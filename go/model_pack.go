// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"encoding/binary"
	"io"
	"slices"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/rocm/internal/gguf"
)

const maxSafetensorsHeaderBytes = 64 << 20

type rocmModelPackConfigProbe struct {
	ModelType             string                       `json:"model_type"`
	Architectures         []string                     `json:"architectures"`
	HiddenSize            int                          `json:"hidden_size"`
	NumHiddenLayers       int                          `json:"num_hidden_layers"`
	NumLayers             int                          `json:"num_layers"`
	NumAttentionHeads     int                          `json:"num_attention_heads"`
	NumKeyValueHeads      int                          `json:"num_key_value_heads"`
	VocabSize             int                          `json:"vocab_size"`
	MaxPositionEmbeddings int                          `json:"max_position_embeddings"`
	MaxSequenceLength     int                          `json:"max_sequence_length"`
	SeqLength             int                          `json:"seq_length"`
	SlidingWindow         int                          `json:"sliding_window"`
	NumLocalExperts       int                          `json:"num_local_experts"`
	NumExperts            int                          `json:"num_experts"`
	NumExpertsPerTok      int                          `json:"num_experts_per_tok"`
	QuantizationConfig    rocmQuantizationConfigProbe  `json:"quantization_config"`
	Quantization          rocmQuantizationConfigProbe  `json:"quantization"`
	TaskSpecificParams    map[string]map[string]string `json:"task_specific_params"`
}

type rocmQuantizationConfigProbe struct {
	QuantMethod  string `json:"quant_method"`
	Bits         int    `json:"bits"`
	GroupSize    int    `json:"group_size"`
	WeightFormat string `json:"weight_format"`
	Format       string `json:"format"`
	LoadIn4Bit   bool   `json:"load_in_4bit"`
	LoadIn8Bit   bool   `json:"load_in_8bit"`
}

type rocmJANGQuantizationInfo struct {
	Version            int                  `json:"version,omitempty"`
	WeightFormat       string               `json:"weight_format,omitempty"`
	Profile            string               `json:"profile,omitempty"`
	Method             string               `json:"method,omitempty"`
	GroupSize          int                  `json:"group_size,omitempty"`
	BitsDefault        int                  `json:"bits_default,omitempty"`
	AttentionBits      int                  `json:"attention_bits,omitempty"`
	SharedExpertBits   int                  `json:"shared_expert_bits,omitempty"`
	RoutedExpertBits   int                  `json:"routed_expert_bits,omitempty"`
	EmbedTokensBits    int                  `json:"embed_tokens_bits,omitempty"`
	LMHeadBits         int                  `json:"lm_head_bits,omitempty"`
	SourceName         string               `json:"source_name,omitempty"`
	SourceOrg          string               `json:"source_org,omitempty"`
	SourceArchitecture string               `json:"source_architecture,omitempty"`
	Capabilities       rocmJANGCapabilities `json:"capabilities,omitempty"`
}

type rocmJANGCapabilities struct {
	ReasoningParser  string `json:"reasoning_parser,omitempty"`
	ToolParser       string `json:"tool_parser,omitempty"`
	ThinkInTemplate  bool   `json:"think_in_template,omitempty"`
	SupportsTools    bool   `json:"supports_tools,omitempty"`
	SupportsThinking bool   `json:"supports_thinking,omitempty"`
	Family           string `json:"family,omitempty"`
	Modality         string `json:"modality,omitempty"`
	CacheType        string `json:"cache_type,omitempty"`
}

type rocmJANGConfigProbe struct {
	Version      int    `json:"version"`
	WeightFormat string `json:"weight_format"`
	Profile      string `json:"profile"`
	SourceModel  struct {
		Name         string `json:"name"`
		Org          string `json:"org"`
		Architecture string `json:"architecture"`
	} `json:"source_model"`
	MXTQBits struct {
		Attention    int `json:"attention"`
		SharedExpert int `json:"shared_expert"`
		RoutedExpert int `json:"routed_expert"`
		EmbedTokens  int `json:"embed_tokens"`
		LMHead       int `json:"lm_head"`
	} `json:"mxtq_bits"`
	Quantization struct {
		Method      string `json:"method"`
		GroupSize   int    `json:"group_size"`
		BitsDefault int    `json:"bits_default"`
	} `json:"quantization"`
	Capabilities rocmJANGCapabilities `json:"capabilities"`
}

type rocmCodebookProfile struct {
	Type         string                      `json:"type,omitempty"`
	Format       string                      `json:"format,omitempty"`
	CodebookSize int                         `json:"codebook_size,omitempty"`
	CodeDim      int                         `json:"code_dim,omitempty"`
	IndexBits    int                         `json:"index_bits,omitempty"`
	Source       string                      `json:"source,omitempty"`
	Tensors      []rocmCodebookTensorProfile `json:"tensors,omitempty"`
}

type rocmCodebookTensorProfile struct {
	Name          string   `json:"name,omitempty"`
	Shape         []uint64 `json:"shape,omitempty"`
	CodesName     string   `json:"codes,omitempty"`
	CodebookName  string   `json:"codebook,omitempty"`
	CodesShape    []uint64 `json:"codes_shape,omitempty"`
	CodebookShape []uint64 `json:"codebook_shape,omitempty"`
	CodebookSize  int      `json:"codebook_size,omitempty"`
	CodeDim       int      `json:"code_dim,omitempty"`
	IndexBits     int      `json:"index_bits,omitempty"`
}

type rocmSafetensorsTensor struct {
	DType       string   `json:"dtype"`
	Shape       []uint64 `json:"shape"`
	DataOffsets []uint64 `json:"data_offsets"`
}

type rocmSafetensorsSummary struct {
	TensorCount  int
	HeaderBytes  uint64
	PayloadBytes uint64
	DTypes       []string
}

func (b *rocmBackend) InspectModelPack(ctx context.Context, path string) (*inference.ModelPackInspection, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	resolvedPath := path
	if abs := core.PathAbs(path); abs.OK {
		resolvedPath = abs.Value.(string)
	}
	stat := core.Stat(resolvedPath)
	if !stat.OK {
		return nil, core.E("rocm.InspectModelPack", "stat model pack", stat.Value.(error))
	}
	info := stat.Value.(core.FsFileInfo)
	root := resolvedPath
	if !info.IsDir() {
		root = core.PathDir(resolvedPath)
	}

	inspection := &inference.ModelPackInspection{
		Path:   resolvedPath,
		Labels: map[string]string{"backend": "rocm", "native_runtime": "hip"},
	}
	weights := discoverROCmWeightFiles(resolvedPath, info)
	inspection.Format = rocmModelPackFormat(weights)
	inspection.Labels["weight_files"] = core.Sprintf("%d", len(weights))
	inspection.Labels["format"] = inspection.Format
	if len(weights) == 0 {
		inspection.Notes = append(inspection.Notes, "no GGUF or safetensors weight files found")
	}

	if cfg, err := readROCmModelConfig(root); err != nil {
		inspection.Notes = append(inspection.Notes, "config.json could not be parsed: "+err.Error())
	} else if cfg != nil {
		applyROCmModelConfig(inspection, *cfg)
	}
	for _, weight := range weights {
		switch core.Lower(core.PathExt(weight)) {
		case ".gguf":
			applyROCmGGUFInspection(inspection, weight)
		case ".safetensors":
			applyROCmSafetensorsInspection(inspection, weight)
		}
	}
	if jang, err := readROCmJANGConfig(root); err != nil {
		inspection.Notes = append(inspection.Notes, "jang_config.json could not be parsed: "+err.Error())
	} else if jang != nil {
		applyROCmJANGInspection(inspection, *jang)
	}
	if codebook, err := readROCmCodebookConfig(root); err != nil {
		inspection.Notes = append(inspection.Notes, "codebook_config.json could not be parsed: "+err.Error())
	} else if codebook != nil {
		applyROCmCodebookInspection(inspection, *codebook)
	}
	applyROCmArchitectureInspection(inspection)
	applyROCmMemoryFitInspection(ctx, b, inspection)
	inspection.Capabilities = append(inspection.Capabilities,
		inference.SupportedCapability(inference.CapabilityModelFit, inference.CapabilityGroupRuntime),
		inference.SupportedCapability(inference.CapabilityMemoryPlanning, inference.CapabilityGroupRuntime),
	)
	inspection.Notes = append(inspection.Notes, "native ROCm decode kernels are not linked yet")
	return inspection, nil
}

func discoverROCmWeightFiles(path string, info core.FsFileInfo) []string {
	if !info.IsDir() {
		if rocmIsWeightFile(path) {
			return []string{path}
		}
		return nil
	}
	weights := []string{}
	_ = core.PathWalkDir(path, func(current string, entry core.FsDirEntry, err error) error {
		if err != nil {
			return nil
		}
		if entry.IsDir() {
			if current != path && core.HasPrefix(core.PathBase(current), ".") {
				return core.PathSkipDir
			}
			return nil
		}
		if rocmIsWeightFile(current) {
			weights = append(weights, current)
		}
		return nil
	})
	slices.Sort(weights)
	return weights
}

func rocmIsWeightFile(path string) bool {
	ext := core.Lower(core.PathExt(path))
	return ext == ".gguf" || ext == ".safetensors"
}

func rocmModelPackFormat(weights []string) string {
	hasGGUF := false
	hasSafetensors := false
	for _, weight := range weights {
		switch core.Lower(core.PathExt(weight)) {
		case ".gguf":
			hasGGUF = true
		case ".safetensors":
			hasSafetensors = true
		}
	}
	switch {
	case hasGGUF && hasSafetensors:
		return "mixed"
	case hasGGUF:
		return "gguf"
	case hasSafetensors:
		return "safetensors"
	default:
		return "missing"
	}
}

func readROCmModelConfig(root string) (*rocmModelPackConfigProbe, error) {
	read := core.ReadFile(core.PathJoin(root, "config.json"))
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return nil, nil
		}
		return nil, read.Value.(error)
	}
	var cfg rocmModelPackConfigProbe
	if result := core.JSONUnmarshal(read.Value.([]byte), &cfg); !result.OK {
		return nil, result.Value.(error)
	}
	return &cfg, nil
}

func applyROCmModelConfig(inspection *inference.ModelPackInspection, cfg rocmModelPackConfigProbe) {
	model := inspection.Model
	model.Architecture = firstNonEmptyString(model.Architecture, rocmConfigArchitecture(cfg))
	model.ContextLength = firstPositiveInt(model.ContextLength, cfg.MaxPositionEmbeddings, cfg.MaxSequenceLength, cfg.SeqLength, cfg.SlidingWindow)
	model.NumLayers = firstPositiveInt(model.NumLayers, cfg.NumHiddenLayers, cfg.NumLayers)
	model.HiddenSize = firstPositiveInt(model.HiddenSize, cfg.HiddenSize)
	model.VocabSize = firstPositiveInt(model.VocabSize, cfg.VocabSize)
	quant := cfg.QuantizationConfig
	if quant.QuantMethod == "" && quant.Bits == 0 && quant.GroupSize == 0 {
		quant = cfg.Quantization
	}
	model.QuantBits = firstPositiveInt(model.QuantBits, rocmQuantConfigBits(quant))
	model.QuantGroup = firstPositiveInt(model.QuantGroup, quant.GroupSize)
	model.QuantType = firstNonEmptyString(model.QuantType, rocmQuantConfigType(quant))
	inspection.Model = model
	if cfg.NumLocalExperts > 0 || cfg.NumExperts > 0 {
		inspection.Labels["moe_experts"] = core.Sprintf("%d", firstPositiveInt(cfg.NumLocalExperts, cfg.NumExperts))
	}
	if cfg.NumExpertsPerTok > 0 {
		inspection.Labels["moe_top_k"] = core.Sprintf("%d", cfg.NumExpertsPerTok)
	}
}

func rocmConfigArchitecture(cfg rocmModelPackConfigProbe) string {
	if cfg.ModelType != "" {
		return normalizeROCmArchitecture(cfg.ModelType)
	}
	for _, architecture := range cfg.Architectures {
		if normalized := normalizeROCmArchitecture(architecture); normalized != "" {
			return normalized
		}
	}
	return ""
}

func rocmQuantConfigBits(quant rocmQuantizationConfigProbe) int {
	if quant.Bits > 0 {
		return quant.Bits
	}
	if quant.LoadIn4Bit {
		return 4
	}
	if quant.LoadIn8Bit {
		return 8
	}
	return 0
}

func rocmQuantConfigType(quant rocmQuantizationConfigProbe) string {
	return core.Lower(firstNonEmptyString(quant.WeightFormat, quant.Format, quant.QuantMethod))
}

func applyROCmGGUFInspection(inspection *inference.ModelPackInspection, path string) {
	info, err := gguf.ReadInfo(path)
	if err != nil {
		inspection.Notes = append(inspection.Notes, "GGUF metadata could not be parsed: "+err.Error())
		return
	}
	metadata := info.Metadata
	model := inspection.Model
	model.Path = path
	model.Architecture = firstNonEmptyString(model.Architecture, normalizeROCmArchitecture(metadata.Architecture))
	model.ContextLength = firstPositiveInt(model.ContextLength, int(metadata.ContextLength))
	model.NumLayers = firstPositiveInt(model.NumLayers, int(metadata.BlockCount))
	bits, group := quantisationFromFileType(metadata.FileType)
	model.QuantBits = firstPositiveInt(model.QuantBits, bits)
	model.QuantGroup = firstPositiveInt(model.QuantGroup, group)
	model.QuantType = firstNonEmptyString(model.QuantType, core.Lower(gguf.FileTypeName(metadata.FileType)))
	inspection.Model = model
	inspection.Labels["gguf_tensors"] = core.Sprintf("%d", len(info.Tensors))
	inspection.Labels["gguf_alignment"] = core.Sprintf("%d", info.Alignment)
	if metadata.FileSize > 0 {
		inspection.Labels["weight_bytes"] = core.Sprintf("%d", metadata.FileSize)
	}
}

func applyROCmSafetensorsInspection(inspection *inference.ModelPackInspection, path string) {
	summary, err := readROCmSafetensorsSummary(path)
	if err != nil {
		inspection.Notes = append(inspection.Notes, "safetensors header could not be parsed: "+err.Error())
		return
	}
	model := inspection.Model
	model.Path = firstNonEmptyString(model.Path, path)
	inspection.Model = model
	inspection.Labels["safetensors_tensors"] = core.Sprintf("%d", summary.TensorCount)
	inspection.Labels["safetensors_header_bytes"] = core.Sprintf("%d", summary.HeaderBytes)
	inspection.Labels["safetensors_payload_bytes"] = core.Sprintf("%d", summary.PayloadBytes)
	if len(summary.DTypes) > 0 {
		inspection.Labels["safetensors_dtypes"] = core.Join(",", summary.DTypes...)
	}
}

func readROCmSafetensorsSummary(path string) (rocmSafetensorsSummary, error) {
	open := core.Open(path)
	if !open.OK {
		return rocmSafetensorsSummary{}, open.Value.(error)
	}
	file := open.Value.(*core.OSFile)
	defer file.Close()
	var headerLength uint64
	if err := binary.Read(file, binary.LittleEndian, &headerLength); err != nil {
		return rocmSafetensorsSummary{}, err
	}
	if headerLength == 0 || headerLength > maxSafetensorsHeaderBytes {
		return rocmSafetensorsSummary{}, core.NewError(core.Sprintf("safetensors header length %d is outside supported bounds", headerLength))
	}
	header := make([]byte, int(headerLength))
	if _, err := io.ReadFull(file, header); err != nil {
		return rocmSafetensorsSummary{}, err
	}
	tensors := map[string]rocmSafetensorsTensor{}
	if result := core.JSONUnmarshal(header, &tensors); !result.OK {
		return rocmSafetensorsSummary{}, result.Value.(error)
	}
	summary := rocmSafetensorsSummary{HeaderBytes: headerLength}
	dtypeSeen := map[string]bool{}
	for name, tensor := range tensors {
		if core.HasPrefix(name, "__") {
			continue
		}
		summary.TensorCount++
		if tensor.DType != "" && !dtypeSeen[tensor.DType] {
			dtypeSeen[tensor.DType] = true
			summary.DTypes = append(summary.DTypes, tensor.DType)
		}
		if len(tensor.DataOffsets) == 2 && tensor.DataOffsets[1] > summary.PayloadBytes {
			summary.PayloadBytes = tensor.DataOffsets[1]
		}
	}
	slices.Sort(summary.DTypes)
	return summary, nil
}

func readROCmJANGConfig(root string) (*rocmJANGQuantizationInfo, error) {
	read := core.ReadFile(core.PathJoin(root, "jang_config.json"))
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return nil, nil
		}
		return nil, read.Value.(error)
	}
	var probe rocmJANGConfigProbe
	if result := core.JSONUnmarshal(read.Value.([]byte), &probe); !result.OK {
		return nil, result.Value.(error)
	}
	return &rocmJANGQuantizationInfo{
		Version:            probe.Version,
		WeightFormat:       probe.WeightFormat,
		Profile:            probe.Profile,
		Method:             probe.Quantization.Method,
		GroupSize:          probe.Quantization.GroupSize,
		BitsDefault:        firstPositiveInt(probe.Quantization.BitsDefault, probe.MXTQBits.RoutedExpert, rocmJANGProfileBits(probe.Profile)),
		AttentionBits:      probe.MXTQBits.Attention,
		SharedExpertBits:   probe.MXTQBits.SharedExpert,
		RoutedExpertBits:   probe.MXTQBits.RoutedExpert,
		EmbedTokensBits:    probe.MXTQBits.EmbedTokens,
		LMHeadBits:         probe.MXTQBits.LMHead,
		SourceName:         probe.SourceModel.Name,
		SourceOrg:          probe.SourceModel.Org,
		SourceArchitecture: normalizeROCmArchitecture(probe.SourceModel.Architecture),
		Capabilities:       probe.Capabilities,
	}, nil
}

func applyROCmJANGInspection(inspection *inference.ModelPackInspection, jang rocmJANGQuantizationInfo) {
	model := inspection.Model
	model.Architecture = firstNonEmptyString(model.Architecture, jang.SourceArchitecture)
	model.QuantBits = firstPositiveInt(model.QuantBits, jang.BitsDefault)
	model.QuantGroup = firstPositiveInt(model.QuantGroup, jang.GroupSize)
	model.QuantType = firstNonEmptyString(model.QuantType, rocmJANGQuantizationType(jang))
	inspection.Model = model
	inspection.Labels["jang_profile"] = jang.Profile
	inspection.Labels["jang_weight_format"] = jang.WeightFormat
	inspection.Labels["jang_method"] = jang.Method
	if jang.GroupSize > 0 {
		inspection.Labels["jang_group_size"] = core.Sprintf("%d", jang.GroupSize)
	}
	if jang.BitsDefault > 0 {
		inspection.Labels["jang_bits_default"] = core.Sprintf("%d", jang.BitsDefault)
	}
	if jang.Capabilities.ReasoningParser != "" || jang.Capabilities.SupportsThinking {
		inspection.Labels["reasoning_parser"] = firstNonEmptyString(jang.Capabilities.ReasoningParser, "native-family")
		inspection.Capabilities = append(inspection.Capabilities, inference.PlannedCapability(inference.CapabilityReasoningParse, inference.CapabilityGroupModel, "JANG reasoning parser metadata is present; ROCm stream parser wiring is pending"))
	}
	if jang.Capabilities.ToolParser != "" || jang.Capabilities.SupportsTools {
		inspection.Labels["tool_parser"] = firstNonEmptyString(jang.Capabilities.ToolParser, "native-family")
		inspection.Capabilities = append(inspection.Capabilities, inference.PlannedCapability(inference.CapabilityToolParse, inference.CapabilityGroupModel, "JANG tool parser metadata is present; ROCm stream parser wiring is pending"))
	}
	if jang.Capabilities.CacheType != "" {
		inspection.Labels["cache_type"] = jang.Capabilities.CacheType
	}
	inspection.Capabilities = append(inspection.Capabilities, inference.PlannedCapability(inference.CapabilityJANGTQ, inference.CapabilityGroupRuntime, "JANG/JANGTQ model-pack metadata is recognised; native ROCm packed kernels are pending"))
	inspection.Notes = append(inspection.Notes, "JANG/JANGTQ kernels are metadata-planned on ROCm")
}

func readROCmCodebookConfig(root string) (*rocmCodebookProfile, error) {
	read := core.ReadFile(core.PathJoin(root, "codebook_config.json"))
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return nil, nil
		}
		return nil, read.Value.(error)
	}
	var profile rocmCodebookProfile
	if result := core.JSONUnmarshal(read.Value.([]byte), &profile); !result.OK {
		return nil, result.Value.(error)
	}
	if profile.Format == "" {
		profile.Format = "vq"
	}
	if profile.Type == "" {
		profile.Type = "codebook"
	}
	if profile.IndexBits == 0 {
		profile.IndexBits = 8
	}
	return &profile, nil
}

func applyROCmCodebookInspection(inspection *inference.ModelPackInspection, profile rocmCodebookProfile) {
	model := inspection.Model
	model.QuantBits = firstPositiveInt(model.QuantBits, profile.IndexBits)
	model.QuantType = firstNonEmptyString(model.QuantType, profile.Type+"."+profile.Format)
	inspection.Model = model
	inspection.Labels["codebook_type"] = profile.Type
	inspection.Labels["codebook_format"] = profile.Format
	inspection.Labels["codebook_tensors"] = core.Sprintf("%d", len(profile.Tensors))
	if profile.CodebookSize > 0 {
		inspection.Labels["codebook_size"] = core.Sprintf("%d", profile.CodebookSize)
	}
	if profile.CodeDim > 0 {
		inspection.Labels["codebook_code_dim"] = core.Sprintf("%d", profile.CodeDim)
	}
	if profile.IndexBits > 0 {
		inspection.Labels["codebook_index_bits"] = core.Sprintf("%d", profile.IndexBits)
	}
	inspection.Capabilities = append(inspection.Capabilities, inference.PlannedCapability(inference.CapabilityCodebookVQ, inference.CapabilityGroupRuntime, "codebook/VQ model-pack metadata is recognised; native ROCm VQ kernels are pending"))
	inspection.Notes = append(inspection.Notes, "codebook/VQ kernels are metadata-planned on ROCm")
}

func applyROCmArchitectureInspection(inspection *inference.ModelPackInspection) {
	architectureOK := supportedNativeArchitecture(inspection.Model.Architecture)
	quantizationOK := supportedNativeQuantization(inspection.Model.QuantBits, inspection.Model.QuantType)
	inspection.Labels["architecture_supported"] = core.Sprintf("%t", architectureOK)
	inspection.Labels["quantization_supported"] = core.Sprintf("%t", quantizationOK)
	inspection.Supported = inspection.Format != "missing" && architectureOK && quantizationOK
	if isROCmMoEArchitecture(inspection.Model.Architecture) || inspection.Labels["moe_experts"] != "" {
		inspection.Capabilities = append(inspection.Capabilities,
			inference.PlannedCapability(inference.CapabilityMoERouting, inference.CapabilityGroupModel, "MoE architecture metadata is recognised; native router kernels are pending"),
			inference.PlannedCapability(inference.CapabilityMoELazyExperts, inference.CapabilityGroupRuntime, "MoE lazy expert residency is required for 16GB-class ROCm devices"),
		)
	}
	if !architectureOK {
		inspection.Notes = append(inspection.Notes, "architecture is not in the native ROCm allow-list yet")
	}
	if !quantizationOK {
		inspection.Notes = append(inspection.Notes, "quantisation is not expected to fit the native ROCm path")
	}
}

func applyROCmMemoryFitInspection(ctx context.Context, backend *rocmBackend, inspection *inference.ModelPackInspection) {
	if backend == nil || inspection == nil {
		return
	}
	report, err := backend.PlanModelFit(ctx, inspection.Model, 0)
	if err != nil || report == nil {
		if err != nil {
			inspection.Notes = append(inspection.Notes, "memory fit planning failed: "+err.Error())
		}
		return
	}
	inspection.Labels["memory_fit"] = core.Sprintf("%t", report.Fits)
	inspection.Labels["memory_plan_machine_class"] = report.MemoryPlan.MachineClass
	inspection.Labels["memory_plan_cache_mode"] = report.MemoryPlan.CacheMode
	inspection.Labels["memory_plan_kv_cache_bytes"] = core.Sprintf("%d", report.MemoryPlan.KVCacheBytes)
	for key, value := range report.MemoryPlan.Labels {
		inspection.Labels["memory_plan_"+key] = value
	}
	inspection.Notes = append(inspection.Notes, report.Notes...)
}

func rocmJANGQuantizationType(jang rocmJANGQuantizationInfo) string {
	lower := core.Lower(core.Concat(jang.Profile, " ", jang.WeightFormat, " ", jang.Method))
	if core.Contains(lower, "jangtq") || core.Contains(lower, "mxtq") {
		return "jangtq"
	}
	return "jang"
}

func rocmJANGProfileBits(profile string) int {
	lower := core.Lower(profile)
	switch {
	case core.Contains(lower, "jangtq"):
		return 2
	case core.Contains(lower, "jang_1"):
		return 1
	case core.Contains(lower, "jang_2"):
		return 2
	case core.Contains(lower, "jang_3"):
		return 3
	case core.Contains(lower, "jang_4"):
		return 4
	default:
		return 0
	}
}

func firstPositiveInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
