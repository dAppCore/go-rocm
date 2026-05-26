// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"io"
	"iter"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const nativeTensorCopyChunkBytes = 16 << 20

type nativeDevicePointer uintptr

type nativeHIPDriver interface {
	Available() bool
	DeviceInfo() nativeDeviceInfo
	Malloc(size uint64) (nativeDevicePointer, error)
	Free(pointer nativeDevicePointer) error
	CopyHostToDevice(pointer nativeDevicePointer, data []byte) error
	CopyDeviceToHost(pointer nativeDevicePointer, data []byte) error
}

type nativeHIPAsyncHostToDevice interface {
	CopyHostToDeviceAsync(pointer nativeDevicePointer, data []byte) error
}

type nativeHIPDeviceMemset interface {
	MemsetAsync(pointer nativeDevicePointer, value byte, size uint64) error
}

func hipCopyHostToDevice(driver nativeHIPDriver, pointer nativeDevicePointer, data []byte) error {
	if async, ok := driver.(nativeHIPAsyncHostToDevice); ok {
		return async.CopyHostToDeviceAsync(pointer, data)
	}
	return driver.CopyHostToDevice(pointer, data)
}

func hipMemsetDevice(driver nativeHIPDriver, pointer nativeDevicePointer, value byte, size uint64) error {
	if size == 0 {
		return nil
	}
	if pointer == 0 {
		return core.E("rocm.hip.MemsetDevice", "device pointer is nil", nil)
	}
	if memset, ok := driver.(nativeHIPDeviceMemset); ok {
		return memset.MemsetAsync(pointer, value, size)
	}
	if size > uint64(int(^uint(0)>>1)) {
		return core.E("rocm.hip.MemsetDevice", "device memset size is out of range", nil)
	}
	payload := make([]byte, int(size))
	if value != 0 {
		for index := range payload {
			payload[index] = value
		}
	}
	return hipCopyHostToDevice(driver, pointer, payload)
}

type hipRuntime struct {
	driver nativeHIPDriver
}

func newSystemNativeRuntime() nativeRuntime {
	return newHIPRuntime(newSystemHIPDriver())
}

func newHIPRuntime(driver nativeHIPDriver) *hipRuntime {
	return &hipRuntime{driver: driver}
}

func (runtime *hipRuntime) Available() bool {
	return runtime != nil && runtime.driver != nil && runtime.driver.Available()
}

func (runtime *hipRuntime) DeviceInfo() nativeDeviceInfo {
	if runtime == nil || runtime.driver == nil {
		return nativeDeviceInfo{}
	}
	return runtime.driver.DeviceInfo()
}

func (runtime *hipRuntime) KernelStatus() hipKernelStatus {
	if runtime == nil || runtime.driver == nil || !runtime.driver.Available() {
		return defaultHIPKernelStatus()
	}
	return normalizeHIPKernelStatus(newHIPRuntimeKernelSet(runtime.driver).Status())
}

func (runtime *hipRuntime) LoadModel(path string, cfg nativeLoadConfig) (nativeModel, error) {
	if runtime == nil || runtime.driver == nil {
		return nil, core.E("rocm.hip.LoadModel", "HIP driver is nil", nil)
	}
	if !runtime.driver.Available() {
		return nil, core.E("rocm.hip.LoadModel", "HIP driver is not available", nil)
	}
	if err := validateHIPLoadConfig(cfg); err != nil {
		return nil, core.E("rocm.hip.LoadModel", "validate tensor plan", err)
	}
	if err := validateHIPTensorFileRanges(path, cfg); err != nil {
		return nil, core.E("rocm.hip.LoadModel", "validate tensor file ranges", err)
	}
	model := &hipLoadedModel{
		driver:      runtime.driver,
		kernels:     newHIPRuntimeKernelSet(runtime.driver),
		modelInfo:   cfg.ModelInfo,
		contextSize: cfg.ContextSize,
		tensors:     make(map[string]hipTensor, len(cfg.Tensors)),
		tokenText:   loadHIPTokenTextDecoderIfPresent(cfg.TokenizerPath),
		createdAt:   time.Now(),
	}
	for _, tensor := range cfg.Tensors {
		if tensor.ByteSize == 0 {
			continue
		}
		pointer, err := runtime.driver.Malloc(tensor.ByteSize)
		if err != nil {
			model.Close()
			return nil, core.E("rocm.hip.LoadModel", "allocate tensor "+tensor.Name, err)
		}
		loaded := hipTensor{info: tensor, pointer: pointer}
		model.tensors[tensor.Name] = loaded
		if err := copyTensorToDevice(runtime.driver, path, cfg.DataOffset, loaded); err != nil {
			model.Close()
			return nil, core.E("rocm.hip.LoadModel", "copy tensor "+tensor.Name, err)
		}
	}
	return model, nil
}

type hipTensor struct {
	info    nativeTensorInfo
	pointer nativeDevicePointer
}

type hipLoadedModel struct {
	driver      nativeHIPDriver
	kernels     hipKernelSet
	modelInfo   inference.ModelInfo
	contextSize int
	tensors     map[string]hipTensor
	adapter     inference.AdapterIdentity
	tinyLoRA    *hipLoadedTinyLoRAAdapter
	smallLoRA   *hipLoadedSmallLoRAAdapter
	classLoRA   *hipLoadedClassifierLoRAAdapter
	tokenText   *hipTokenTextDecoder
	q4ConfigMu  sync.Mutex
	q4Config    hipGemma4Q4ForwardConfig
	q4Layers    int
	q4ConfigOK  bool
	createdAt   time.Time
	closed      bool
}

func (model *hipLoadedModel) Generate(ctx context.Context, prompt string, cfg inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	return model.kernelSet().Generate(ctx, model, prompt, cfg)
}

func (model *hipLoadedModel) Chat(ctx context.Context, messages []inference.Message, cfg inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	return model.kernelSet().Chat(ctx, model, messages, cfg)
}

func (model *hipLoadedModel) Classify(ctx context.Context, prompts []string, cfg inference.GenerateConfig) ([]inference.ClassifyResult, error) {
	return model.kernelSet().Classify(ctx, model, prompts, cfg)
}

func (model *hipLoadedModel) BatchGenerate(ctx context.Context, prompts []string, cfg inference.GenerateConfig) ([]inference.BatchResult, error) {
	return model.kernelSet().BatchGenerate(ctx, model, prompts, cfg)
}

func (model *hipLoadedModel) Project(ctx context.Context, req hipProjectionRequest) ([]float32, error) {
	return model.kernelSet().Project(ctx, model, req)
}

func (model *hipLoadedModel) Prefill(ctx context.Context, req hipPrefillRequest) (hipPrefillResult, error) {
	return model.kernelSet().Prefill(ctx, model, req)
}

func (model *hipLoadedModel) DecodeToken(ctx context.Context, req hipDecodeRequest) (hipDecodeResult, error) {
	return model.kernelSet().Decode(ctx, model, req)
}

func (model *hipLoadedModel) Encode(text string) []int32 {
	if model != nil && model.tokenText != nil {
		return model.tokenText.Encode(text)
	}
	return approximateTokenIDs(text)
}

func (model *hipLoadedModel) Decode(ids []int32) string {
	if model != nil && model.tokenText != nil {
		return model.tokenText.Decode(ids)
	}
	if len(ids) == 0 {
		return ""
	}
	return core.Sprintf("%d tokens", len(ids))
}

func (model *hipLoadedModel) ApplyChatTemplate(messages []inference.Message) (string, error) {
	if model != nil && isROCmGemma4Architecture(model.modelInfo.Architecture) {
		return formatGemma4ChatTemplate(messages), nil
	}
	return formatFallbackChatTemplate(messages), nil
}

func (model *hipLoadedModel) LoadAdapter(path string) (inference.AdapterIdentity, error) {
	if core.Trim(path) == "" {
		return inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "adapter path is required", nil)
	}
	if model == nil || normalizeHIPKernelStatus(model.KernelStatus()).LoRA != hipKernelStatusLinked {
		return inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "native LoRA adapter application is not linked yet: "+path, nil)
	}
	if smallCfg, err := model.loadedSmallDecodeConfig(); err == nil {
		adapter, identity, err := model.loadSmallLoRAAdapter(path, smallCfg)
		if err != nil {
			return inference.AdapterIdentity{}, err
		}
		model.smallLoRA = adapter
		model.tinyLoRA = nil
		model.classLoRA = nil
		model.adapter = cloneAdapterIdentity(identity)
		return cloneAdapterIdentity(identity), nil
	}
	if _, err := model.loadedTinyLMConfig(); err == nil {
		adapter, identity, err := model.loadTinyLoRAAdapter(path)
		if err != nil {
			return inference.AdapterIdentity{}, err
		}
		model.tinyLoRA = adapter
		model.smallLoRA = nil
		model.classLoRA = nil
		model.adapter = cloneAdapterIdentity(identity)
		return cloneAdapterIdentity(identity), nil
	}
	classifier, hasClassifier, err := model.loadedSequenceClassifierConfig()
	if err != nil {
		return inference.AdapterIdentity{}, err
	}
	if hasClassifier {
		adapter, identity, err := model.loadClassifierLoRAAdapter(path, classifier)
		if err != nil {
			return inference.AdapterIdentity{}, err
		}
		model.classLoRA = adapter
		model.tinyLoRA = nil
		model.smallLoRA = nil
		model.adapter = cloneAdapterIdentity(identity)
		return cloneAdapterIdentity(identity), nil
	}
	return inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "no loaded LoRA adapter target supports this model", nil)
}

func (model *hipLoadedModel) UnloadAdapter() error {
	model.adapter = inference.AdapterIdentity{}
	model.tinyLoRA = nil
	model.smallLoRA = nil
	model.classLoRA = nil
	return nil
}

func validateHIPLoadConfig(cfg nativeLoadConfig) error {
	if !hipSupportedModelQuantization(cfg.ModelInfo) {
		return core.E("rocm.hip.Validate", "unsupported quantization", nil)
	}
	if cfg.DataOffset < 0 {
		return core.E("rocm.hip.Validate", "data offset must be non-negative", nil)
	}
	if len(cfg.Tensors) == 0 {
		return core.E("rocm.hip.Validate", "missing token embedding tensor", nil)
	}
	hasEmbedding := false
	hasOutput := false
	layerIDs := map[string]struct{}{}
	tensorNames := map[string]struct{}{}
	for _, tensor := range cfg.Tensors {
		if core.Trim(tensor.Name) == "" {
			return core.E("rocm.hip.Validate", "tensor name is required", nil)
		}
		name := core.Lower(tensor.Name)
		if _, exists := tensorNames[name]; exists {
			return core.E("rocm.hip.Validate", "duplicate tensor name "+tensor.Name, nil)
		}
		tensorNames[name] = struct{}{}
		if err := validateHIPTensorDataOffset(hipTensorDataOffset(cfg, tensor), tensor); err != nil {
			return err
		}
		if !hipSupportedTensorDType(tensor) {
			return core.E("rocm.hip.Validate", "unsupported tensor dtype "+tensor.Name, nil)
		}
		if err := validateHIPTensorShape(cfg.ModelInfo, tensor); err != nil {
			return err
		}
		if isHIPEmbeddingTensor(name) {
			if tensor.ByteSize == 0 {
				return core.E("rocm.hip.Validate", "required tensor has zero byte size "+tensor.Name, nil)
			}
			hasEmbedding = true
		}
		if isHIPOutputTensor(name) {
			if tensor.ByteSize == 0 {
				return core.E("rocm.hip.Validate", "required tensor has zero byte size "+tensor.Name, nil)
			}
			hasOutput = true
		}
		if layerID := hipLayerID(name); layerID != "" {
			layerIDs[layerID] = struct{}{}
		}
	}
	if !hasEmbedding {
		return core.E("rocm.hip.Validate", "missing token embedding tensor", nil)
	}
	if !hasOutput && hipLoadConfigRequiresOutputHead(cfg) {
		return core.E("rocm.hip.Validate", "missing output head tensor", nil)
	}
	if cfg.ModelInfo.NumLayers > 0 && len(layerIDs) > 0 && len(layerIDs) != cfg.ModelInfo.NumLayers {
		return core.E("rocm.hip.Validate", core.Sprintf("mismatched layer count: metadata=%d tensors=%d", cfg.ModelInfo.NumLayers, len(layerIDs)), nil)
	}
	return nil
}

func validateHIPTensorDataOffset(dataOffset int64, tensor nativeTensorInfo) error {
	const maxInt64 = int64(1<<63 - 1)
	if tensor.Offset > uint64(maxInt64-dataOffset) {
		return core.E("rocm.hip.Validate", "tensor data offset overflows int64 "+tensor.Name, nil)
	}
	return nil
}

func validateHIPTensorFileRanges(path string, cfg nativeLoadConfig) error {
	for _, tensor := range cfg.Tensors {
		if tensor.ByteSize == 0 {
			continue
		}
		sourcePath := hipTensorSourcePath(path, tensor)
		stat := core.Stat(sourcePath)
		if !stat.OK {
			return stat.Value.(error)
		}
		size := stat.Value.(core.FsFileInfo).Size()
		if size < 0 {
			return core.E("rocm.hip.Validate", "model file size is invalid", nil)
		}
		start := hipTensorDataOffset(cfg, tensor) + int64(tensor.Offset)
		end, err := hipTensorFileEnd(start, tensor.ByteSize)
		if err != nil {
			return core.E("rocm.hip.Validate", "tensor byte range "+tensor.Name, err)
		}
		if end > size {
			return core.E("rocm.hip.Validate", "tensor byte range exceeds file size "+tensor.Name, nil)
		}
	}
	return nil
}

func hipTensorSourcePath(defaultPath string, tensor nativeTensorInfo) string {
	if tensor.SourcePath != "" {
		return tensor.SourcePath
	}
	return defaultPath
}

func hipTensorDataOffset(cfg nativeLoadConfig, tensor nativeTensorInfo) int64 {
	if tensor.SourcePath != "" || tensor.DataOffset != 0 {
		return tensor.DataOffset
	}
	return cfg.DataOffset
}

func hipTensorFileEnd(start int64, byteSize uint64) (int64, error) {
	const maxInt64 = int64(1<<63 - 1)
	if start < 0 {
		return 0, core.E("rocm.hip.TensorRange", "start offset is negative", nil)
	}
	if byteSize > uint64(maxInt64-start) {
		return 0, core.E("rocm.hip.TensorRange", "end offset overflows int64", nil)
	}
	return start + int64(byteSize), nil
}

func validateHIPTensorShape(info inference.ModelInfo, tensor nativeTensorInfo) error {
	if len(tensor.Dimensions) == 0 {
		return nil
	}
	elements, err := hipTensorElementCount(tensor.Dimensions)
	if err != nil {
		return core.E("rocm.hip.Validate", "invalid tensor dimensions "+tensor.Name, err)
	}
	if expectedBytes, ok := hipExpectedTensorBytes(tensor.Type, elements); ok && tensor.ByteSize > 0 && tensor.ByteSize != expectedBytes {
		return core.E("rocm.hip.Validate", core.Sprintf("tensor byte size mismatch %s: metadata=%d expected=%d", tensor.Name, tensor.ByteSize, expectedBytes), nil)
	}
	name := core.Lower(tensor.Name)
	if !isHIPEmbeddingTensor(name) && !isHIPOutputTensor(name) {
		return nil
	}
	if len(tensor.Dimensions) != 2 {
		return core.E("rocm.hip.Validate", "projection tensor must be rank 2 "+tensor.Name, nil)
	}
	if info.HiddenSize > 0 && !hipTensorDimensionsContainLogical(tensor, uint64(info.HiddenSize), info) {
		return core.E("rocm.hip.Validate", core.Sprintf("projection tensor %s missing hidden size %d", tensor.Name, info.HiddenSize), nil)
	}
	if info.VocabSize > 0 && !hipDimensionsContain(tensor.Dimensions, uint64(info.VocabSize)) {
		return core.E("rocm.hip.Validate", core.Sprintf("projection tensor %s missing vocab size %d", tensor.Name, info.VocabSize), nil)
	}
	return nil
}

func isHIPEmbeddingTensor(name string) bool {
	return core.Contains(name, "tok_embeddings.weight") ||
		core.Contains(name, "token_embd.weight") ||
		core.Contains(name, "embed_tokens.weight") ||
		core.Contains(name, "word_embeddings.weight")
}

func isHIPOutputTensor(name string) bool {
	return core.Contains(name, "output.weight") || core.Contains(name, "lm_head.weight")
}

func hipLoadConfigRequiresOutputHead(cfg nativeLoadConfig) bool {
	if cfg.TiedWordEmbeddings {
		return false
	}
	return normalizeROCmArchitecture(cfg.ModelInfo.Architecture) != "bert"
}

func hipTensorElementCount(dimensions []uint64) (uint64, error) {
	if len(dimensions) == 0 {
		return 0, core.E("rocm.hip.TensorShape", "tensor has no dimensions", nil)
	}
	elements := uint64(1)
	for _, dimension := range dimensions {
		if dimension == 0 {
			return 0, core.E("rocm.hip.TensorShape", "tensor has a zero dimension", nil)
		}
		if elements > ^uint64(0)/dimension {
			return 0, core.E("rocm.hip.TensorShape", "tensor element count overflows uint64", nil)
		}
		elements *= dimension
	}
	return elements, nil
}

func hipExpectedTensorBytes(tensorType uint32, elements uint64) (uint64, bool) {
	blockSize, typeSize, ok := hipTensorBlockSize(tensorType)
	if !ok {
		return 0, false
	}
	blocks := (elements + blockSize - 1) / blockSize
	if blocks > ^uint64(0)/typeSize {
		return 0, false
	}
	return blocks * typeSize, true
}

func hipTensorBlockSize(tensorType uint32) (blockSize, typeSize uint64, ok bool) {
	switch tensorType {
	case 0:
		return 1, 4, true
	case 1, 30:
		return 1, 2, true
	case 2:
		return 32, 18, true
	case 3:
		return 32, 20, true
	case 6:
		return 32, 22, true
	case 7:
		return 32, 24, true
	case 8:
		return 32, 34, true
	case 10:
		return 256, 84, true
	case 11:
		return 256, 110, true
	case 12:
		return 256, 144, true
	case 13:
		return 256, 176, true
	case 14:
		return 256, 210, true
	case 15:
		return 256, 292, true
	case 24:
		return 1, 1, true
	case 25:
		return 1, 2, true
	case 26:
		return 1, 4, true
	case 27, 28:
		return 1, 8, true
	default:
		return 0, 0, false
	}
}

func hipDimensionsContain(dimensions []uint64, value uint64) bool {
	for _, dimension := range dimensions {
		if dimension == value {
			return true
		}
	}
	return false
}

func hipTensorDimensionsContainLogical(tensor nativeTensorInfo, value uint64, info inference.ModelInfo) bool {
	if hipDimensionsContain(tensor.Dimensions, value) {
		return true
	}
	if info.QuantBits == 4 && (tensor.Type == 26 || core.Upper(tensor.TypeName) == "U32") {
		for _, dimension := range tensor.Dimensions {
			if dimension <= ^uint64(0)/8 && dimension*8 == value {
				return true
			}
		}
	}
	return false
}

func hipSupportedModelQuantization(info inference.ModelInfo) bool {
	if info.QuantBits == 0 && info.QuantGroup == 0 {
		return true
	}
	return info.QuantBits == 0 || info.QuantBits == 2 || info.QuantBits == 3 || info.QuantBits == 4 || info.QuantBits == 5 || info.QuantBits == 6 || info.QuantBits == 8 || info.QuantBits == 16 || info.QuantBits == 32
}

func hipSupportedTensorDType(tensor nativeTensorInfo) bool {
	if _, _, ok := hipTensorBlockSize(tensor.Type); ok {
		return true
	}
	name := core.Lower(tensor.TypeName)
	if name == "" {
		return false
	}
	return name == "f32" || name == "f16" || name == "q8_0" || name == "q4_k" || name == "q4_k_m" ||
		core.Contains(name, "jangtq") || core.Contains(name, "mxtq") ||
		core.Contains(name, "codebook") || core.Contains(name, "vq")
}

func hipLayerID(name string) string {
	const marker = "layers."
	index := core.Index(name, marker)
	if index < 0 {
		return ""
	}
	rest := name[index+len(marker):]
	end := 0
	for end < len(rest) && rest[end] >= '0' && rest[end] <= '9' {
		end++
	}
	if end == 0 {
		return ""
	}
	return rest[:end]
}

func (model *hipLoadedModel) ActiveAdapter() inference.AdapterIdentity {
	if model == nil {
		return inference.AdapterIdentity{}
	}
	return cloneAdapterIdentity(model.adapter)
}

func (model *hipLoadedModel) KernelStatus() hipKernelStatus {
	return model.tinyLoadedKernelStatus(model.kernelSet().Status())
}

func (model *hipLoadedModel) Metrics() inference.GenerateMetrics {
	if model == nil {
		return inference.GenerateMetrics{}
	}
	metrics := inference.GenerateMetrics{ActiveMemoryBytes: model.deviceBytes()}
	metrics.PeakMemoryBytes = metrics.ActiveMemoryBytes
	return metrics
}

func (model *hipLoadedModel) Close() error {
	if model == nil || model.closed {
		return nil
	}
	var lastErr error
	for name, tensor := range model.tensors {
		if err := model.driver.Free(tensor.pointer); err != nil {
			lastErr = core.E("rocm.hip.Close", "free tensor "+name, err)
		}
		delete(model.tensors, name)
	}
	model.adapter = inference.AdapterIdentity{}
	model.tinyLoRA = nil
	model.smallLoRA = nil
	model.classLoRA = nil
	model.closed = true
	return lastErr
}

func (model *hipLoadedModel) deviceBytes() uint64 {
	var total uint64
	for _, tensor := range model.tensors {
		total += tensor.info.ByteSize
	}
	return total
}

func copyTensorToDevice(driver nativeHIPDriver, path string, dataOffset int64, tensor hipTensor) error {
	sourcePath := tensor.info.SourcePath
	if sourcePath == "" {
		sourcePath = path
	} else {
		dataOffset = tensor.info.DataOffset
	}
	if tensor.info.SourcePath == "" && tensor.info.DataOffset != 0 {
		dataOffset = tensor.info.DataOffset
	}
	fileResult := core.Open(sourcePath)
	if !fileResult.OK {
		return fileResult.Value.(error)
	}
	file := fileResult.Value.(*core.OSFile)
	defer file.Close()

	start := dataOffset + int64(tensor.info.Offset)
	if _, err := file.Seek(start, io.SeekStart); err != nil {
		return err
	}

	remaining := tensor.info.ByteSize
	buffer := make([]byte, min(uint64(nativeTensorCopyChunkBytes), remaining))
	var copied uint64
	for remaining > 0 {
		chunk := int(min(uint64(len(buffer)), remaining))
		if _, err := io.ReadFull(file, buffer[:chunk]); err != nil {
			return err
		}
		if err := driver.CopyHostToDevice(tensor.pointer+nativeDevicePointer(copied), buffer[:chunk]); err != nil {
			return err
		}
		copied += uint64(chunk)
		remaining -= uint64(chunk)
	}
	return nil
}
