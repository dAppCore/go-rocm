// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"iter"
	"strconv"
	"strings"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/rocm/internal/gguf"
)

const (
	defaultContextLengthCap   = 4096
	memoryGiB                 = uint64(1 << 30)
	memoryClassToleranceBytes = uint64(128 << 20)
)

type rocmBackend struct {
	runtime nativeRuntime
}

type nativeRuntime interface {
	Available() bool
	DeviceInfo() nativeDeviceInfo
	LoadModel(path string, cfg nativeLoadConfig) (nativeModel, error)
}

type nativeDeviceInfo struct {
	Name        string
	MemoryBytes uint64
	FreeBytes   uint64
	Driver      string
}

type nativeLoadConfig struct {
	ContextSize        int
	GPULayerCount      int
	ParallelSlotCount  int
	AdapterPath        string
	ModelInfo          inference.ModelInfo
	TokenizerPath      string
	DataOffset         int64
	Tensors            []nativeTensorInfo
	TiedWordEmbeddings bool
}

type nativeTensorInfo struct {
	Name       string
	Dimensions []uint64
	Type       uint32
	TypeName   string
	SourcePath string
	DataOffset int64
	Offset     uint64
	ByteSize   uint64
}

type nativeModel interface {
	Generate(ctx context.Context, prompt string, cfg inference.GenerateConfig) (iter.Seq[inference.Token], func() error)
	Chat(ctx context.Context, messages []inference.Message, cfg inference.GenerateConfig) (iter.Seq[inference.Token], func() error)
	Classify(ctx context.Context, prompts []string, cfg inference.GenerateConfig) ([]inference.ClassifyResult, error)
	BatchGenerate(ctx context.Context, prompts []string, cfg inference.GenerateConfig) ([]inference.BatchResult, error)
	Encode(text string) []int32
	Decode(ids []int32) string
	ApplyChatTemplate(messages []inference.Message) (string, error)
	LoadAdapter(path string) (inference.AdapterIdentity, error)
	UnloadAdapter() error
	ActiveAdapter() inference.AdapterIdentity
	KernelStatus() hipKernelStatus
	Metrics() inference.GenerateMetrics
	Close() error
}

func newROCmBackendWithRuntime(runtime nativeRuntime) *rocmBackend {
	return &rocmBackend{runtime: runtime}
}

func (b *rocmBackend) Name() string { return "rocm" }

func (b *rocmBackend) Available() bool {
	return b.nativeRuntime().Available()
}

func (b *rocmBackend) Capabilities() inference.CapabilityReport {
	runtime := b.nativeRuntime()
	return rocmCapabilityReport(runtime.DeviceInfo(), inference.ModelIdentity{}, inference.AdapterIdentity{}, runtime.Available(), nativeRuntimeKernelStatus(runtime))
}

func (b *rocmBackend) LoadModel(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
	loadConfig := inference.ApplyLoadOpts(opts)
	if loadConfig.AdapterPath != "" && core.Trim(loadConfig.AdapterPath) == "" {
		return nil, core.E("rocm.LoadModel", "adapter path is required", nil)
	}
	modelPack, err := gguf.ReadInfo(path)
	modelPath := path
	nativeConfig := nativeLoadConfig{}
	modelInfo := inference.ModelInfo{}
	if err == nil {
		metadata := modelPack.Metadata
		modelInfo = modelInfoFromMetadata(metadata)
		nativeConfig = nativeLoadConfig{
			ContextSize:       resolveContextLength(loadConfig.ContextLen, metadata),
			GPULayerCount:     loadConfig.GPULayers,
			ParallelSlotCount: loadConfig.ParallelSlots,
			AdapterPath:       loadConfig.AdapterPath,
			ModelInfo:         modelInfo,
			DataOffset:        modelPack.DataOffset,
			Tensors:           nativeTensorInfos(modelPack.Tensors),
		}
	} else {
		modelPath, nativeConfig, err = b.safetensorsNativeLoadConfig(context.Background(), path, loadConfig)
		if err != nil {
			return nil, core.E("rocm.LoadModel", "read model metadata", err)
		}
		modelInfo = nativeConfig.ModelInfo
	}

	runtime := b.nativeRuntime()
	if !runtime.Available() {
		return nil, core.E("rocm.LoadModel", "native ROCm runtime is not available", nil)
	}

	loaded, err := runtime.LoadModel(modelPath, nativeConfig)
	if err != nil {
		return nil, core.E("rocm.LoadModel", "load native model", err)
	}

	if hipModel, ok := loaded.(*hipLoadedModel); ok &&
		isROCmGemma4Architecture(modelInfo.Architecture) &&
		modelInfo.QuantBits == 4 &&
		modelInfo.NumLayers > 0 {
		if _, err := hipModel.cachedGemma4Q4ForwardConfig(modelInfo.NumLayers); err != nil {
			_ = loaded.Close()
			return nil, core.E("rocm.LoadModel", "prepare Gemma4 q4 forward config", err)
		}
	}

	model := &rocmModel{native: loaded, modelType: modelInfo.Architecture, modelInfo: modelInfo}
	if loadConfig.AdapterPath != "" {
		if _, err := model.LoadAdapter(loadConfig.AdapterPath); err != nil {
			_ = model.Close()
			return nil, core.E("rocm.LoadModel", "load adapter", err)
		}
	}
	return model, nil
}

func (b *rocmBackend) PlanModelFit(ctx context.Context, model inference.ModelIdentity, memoryBytes uint64) (*inference.ModelFitReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if memoryBytes == 0 {
		device := b.nativeRuntime().DeviceInfo()
		memoryBytes = device.MemoryBytes
	}
	if memoryBytes == 0 {
		memoryBytes = 16 * memoryGiB
	}

	contextLength := model.ContextLength
	if contextLength <= 0 {
		contextLength = defaultContextLengthCap
	}
	layers := model.NumLayers
	if layers <= 0 {
		layers = 32
	}
	hidden := model.HiddenSize
	if hidden <= 0 {
		hidden = 4096
	}

	cacheMode := rocmRecommendedCacheMode(memoryBytes, contextLength, model)
	kvBytes := estimateKVCacheBytes(layers, contextLength, hidden, cacheMode, model)
	weightBytes := rocmModelWeightBytes(model)
	runtimeBytes := rocmEstimatedRuntimeBytes(kvBytes, weightBytes)
	fitLimitBytes := memoryBytes * 85 / 100
	architectureOK := supportedNativeArchitecture(model.Architecture)
	quantizationOK := supportedNativeQuantization(model.QuantBits, model.QuantType)
	fits := architectureOK && quantizationOK && kvBytes < memoryBytes*7/10
	if weightBytes > 0 {
		fits = architectureOK && quantizationOK && runtimeBytes < fitLimitBytes
	}
	labels := rocmMemoryPlanLabels(memoryBytes, contextLength, layers, hidden, model, kvBytes, weightBytes, runtimeBytes, cacheMode)
	plan := inference.MemoryPlan{
		MachineClass:      rocmMachineClass(memoryBytes),
		DeviceMemoryBytes: memoryBytes,
		ContextLength:     contextLength,
		BatchSize:         rocmRecommendedBatchSize(memoryBytes),
		CacheMode:         cacheMode,
		Quantization:      rocmQuantizationLabel(model),
		KVCacheBytes:      kvBytes,
		TrainingFeasible:  rocmAtLeastMemoryClass(memoryBytes, 16*memoryGiB) && model.QuantBits <= 8,
		Labels:            labels,
	}
	if !architectureOK {
		plan.Notes = append(plan.Notes, "architecture is not in the native ROCm allow-list yet")
	}
	if !quantizationOK {
		plan.Notes = append(plan.Notes, "quantisation is not expected to fit the native ROCm path")
	}
	if weightBytes > 0 && runtimeBytes >= fitLimitBytes {
		plan.Notes = append(plan.Notes, "weight and KV cache estimate leaves too little memory for workspace")
	} else if kvBytes >= memoryBytes*7/10 {
		plan.Notes = append(plan.Notes, "KV cache estimate leaves too little memory for weights and workspace")
	}
	if memoryBytes <= 16*memoryGiB {
		plan.Notes = append(plan.Notes, "ROCm 16GB plan uses chunked prefill, compact KV cache, and conservative allocator limits")
	}
	if isROCmMoEArchitecture(model.Architecture) {
		plan.Notes = append(plan.Notes, "MoE lazy expert residency is required on 16GB-class ROCm devices")
	}
	if isROCmMetadataQuantization(model.QuantType) {
		plan.Notes = append(plan.Notes, "metadata quantisation is recognised; native ROCm packed kernels are pending")
	}

	return &inference.ModelFitReport{
		Model:          model,
		Fits:           fits,
		MemoryPlan:     plan,
		ArchitectureOK: architectureOK,
		QuantizationOK: quantizationOK,
		Notes:          append([]string(nil), plan.Notes...),
	}, nil
}

func (b *rocmBackend) nativeRuntime() nativeRuntime {
	if b != nil && b.runtime != nil {
		return b.runtime
	}
	return newSystemNativeRuntime()
}

type nativeRuntimeKernelReporter interface {
	KernelStatus() hipKernelStatus
}

type nativeEvalLossKernelModel interface {
	RunEvalCrossEntropyLoss(ctx context.Context, logits [][]float32, targets []int) (hipCrossEntropyLossResult, bool, error)
}

func nativeRuntimeKernelStatus(runtime nativeRuntime) hipKernelStatus {
	if runtime == nil {
		return defaultHIPKernelStatus()
	}
	reporter, ok := runtime.(nativeRuntimeKernelReporter)
	if !ok {
		return defaultHIPKernelStatus()
	}
	return normalizeHIPKernelStatus(reporter.KernelStatus())
}

type rocmModel struct {
	native    nativeModel
	modelType string
	modelInfo inference.ModelInfo

	stateMutex  sync.Mutex
	lastError   error
	lastMetrics inference.GenerateMetrics
	probeSink   inference.ProbeSink
	adapter     inference.AdapterIdentity
	cache       *BlockCacheService
	state       *StateSession
}

func (m *rocmModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.clearLastError()
	if err := rocmContextErr(ctx); err != nil {
		m.setLastFailure(err)
		return emptyTokenSeq
	}
	if m == nil || m.native == nil {
		if m != nil {
			m.setLastFailure(core.E("rocm.Generate", "native model is nil", nil))
		}
		return emptyTokenSeq
	}
	cfg := cloneGenerateConfig(inference.ApplyGenerateOpts(opts))
	promptTokens := m.promptTokenCount(prompt)
	start := time.Now()
	stream, streamError := m.native.Generate(ctx, prompt, cloneGenerateConfig(cfg))
	return m.wrapTokenStream(stream, streamError, promptTokens, start, nil)
}

func (m *rocmModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.clearLastError()
	if err := rocmContextErr(ctx); err != nil {
		m.setLastFailure(err)
		return emptyTokenSeq
	}
	if m == nil || m.native == nil {
		if m != nil {
			m.setLastFailure(core.E("rocm.Chat", "native model is nil", nil))
		}
		return emptyTokenSeq
	}
	if err := validateROCmChatMessages("rocm.Chat", messages); err != nil {
		m.setLastFailure(err)
		return emptyTokenSeq
	}
	cfg := cloneGenerateConfig(inference.ApplyGenerateOpts(opts))
	promptTokens := m.chatPromptTokenCount(messages)
	start := time.Now()
	stream, streamError := m.native.Chat(ctx, append([]inference.Message(nil), messages...), cloneGenerateConfig(cfg))
	return m.wrapTokenStream(stream, streamError, promptTokens, start, nil)
}

func (m *rocmModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	m.clearLastError()
	if err := rocmContextErr(ctx); err != nil {
		m.setLastFailure(err)
		return nil, err
	}
	if m == nil || m.native == nil {
		err := core.E("rocm.Classify", "native model is nil", nil)
		if m != nil {
			m.setLastFailure(err)
		}
		return nil, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := validateROCmPromptBatch("rocm.Classify", prompts); err != nil {
		m.setLastFailure(err)
		return nil, err
	}
	cfg := cloneGenerateConfig(inference.ApplyGenerateOpts(opts))
	start := time.Now()
	results, err := m.native.Classify(ctx, append([]string(nil), prompts...), cloneGenerateConfig(cfg))
	results = cloneClassifyResults(results)
	if !cfg.ReturnLogits {
		stripClassifyLogits(results)
	} else if err == nil {
		m.emitClassifyLogitProbes(results)
	}
	if err != nil {
		m.setLastFailure(err)
	}
	m.recordMetrics(m.promptsTokenCount(prompts), len(results), start, time.Now())
	return results, err
}

func stripClassifyLogits(results []inference.ClassifyResult) {
	for i := range results {
		results[i].Logits = nil
	}
}

func cloneGenerateConfig(cfg inference.GenerateConfig) inference.GenerateConfig {
	cfg.StopTokens = append([]int32(nil), cfg.StopTokens...)
	return cfg
}

func cloneClassifyResults(results []inference.ClassifyResult) []inference.ClassifyResult {
	if len(results) == 0 {
		return results
	}
	out := append([]inference.ClassifyResult(nil), results...)
	for index := range out {
		out[index].Logits = append([]float32(nil), results[index].Logits...)
	}
	return out
}

func (m *rocmModel) emitClassifyLogitProbes(results []inference.ClassifyResult) {
	sink := m.probeSinkSnapshot()
	if sink == nil {
		return
	}
	for index, result := range results {
		if len(result.Logits) == 0 {
			continue
		}
		probeSink := inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
			event.Step = index + 1
			event.Labels = mergeStringMaps(event.Labels, map[string]string{
				"classify_prompt_index": core.Sprintf("%d", index),
				"source":                "classification",
			})
			sink.EmitProbe(event)
		})
		_, _ = rocmReferenceLogitProbe(result.Logits, rocmLogitProbeTopK(len(result.Logits)), nil, probeSink)
		_, _ = rocmReferenceEntropyProbe(result.Logits, probeSink)
	}
}

func (m *rocmModel) probeSinkSnapshot() inference.ProbeSink {
	if m == nil {
		return nil
	}
	m.stateMutex.Lock()
	defer m.stateMutex.Unlock()
	return m.probeSink
}

func rocmLogitProbeTopK(vocabularySize int) int {
	if vocabularySize <= 0 {
		return 0
	}
	if vocabularySize < 5 {
		return vocabularySize
	}
	return 5
}

func (m *rocmModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.BatchResult, error) {
	m.clearLastError()
	if err := rocmContextErr(ctx); err != nil {
		m.setLastFailure(err)
		return nil, err
	}
	if m == nil || m.native == nil {
		err := core.E("rocm.BatchGenerate", "native model is nil", nil)
		if m != nil {
			m.setLastFailure(err)
		}
		return nil, err
	}
	if err := validateROCmPromptBatch("rocm.BatchGenerate", prompts); err != nil {
		m.setLastFailure(err)
		return nil, err
	}
	start := time.Now()
	cfg := cloneGenerateConfig(inference.ApplyGenerateOpts(opts))
	results, err := m.native.BatchGenerate(ctx, append([]string(nil), prompts...), cloneGenerateConfig(cfg))
	results = cloneBatchResults(results)
	generated := 0
	for _, result := range results {
		generated += len(result.Tokens)
	}
	if err != nil {
		m.setLastFailure(err)
	} else if resultErr := firstBatchResultError(results); resultErr != nil {
		m.setLastFailure(resultErr)
	}
	m.recordMetrics(m.promptsTokenCount(prompts), generated, start, time.Now())
	return results, err
}

func cloneBatchResults(results []inference.BatchResult) []inference.BatchResult {
	if len(results) == 0 {
		return results
	}
	out := append([]inference.BatchResult(nil), results...)
	for index := range out {
		out[index].Tokens = append([]inference.Token(nil), results[index].Tokens...)
	}
	return out
}

func firstBatchResultError(results []inference.BatchResult) error {
	for _, result := range results {
		if result.Err != nil {
			return result.Err
		}
	}
	return nil
}

func (m *rocmModel) ModelType() string {
	if m == nil {
		return ""
	}
	return m.modelType
}

func (m *rocmModel) Info() inference.ModelInfo {
	if m == nil {
		return inference.ModelInfo{}
	}
	return m.modelInfo
}

func (m *rocmModel) Capabilities() inference.CapabilityReport {
	if m == nil {
		return rocmCapabilityReport(nativeDeviceInfo{}, inference.ModelIdentity{}, inference.AdapterIdentity{}, false, defaultHIPKernelStatus())
	}
	return rocmCapabilityReport(nativeDeviceInfo{}, m.modelIdentity(), m.ActiveAdapter(), m.native != nil, m.kernelStatus(), rocmCapabilityReportOption{
		ClassifyLinked:         m.classifyLinked(),
		Gemma4Q4GenerateLinked: m.gemma4Q4GenerateLinked(),
	})
}

func (m *rocmModel) classifyLinked() bool {
	if m == nil {
		return false
	}
	loaded, ok := m.native.(*hipLoadedModel)
	if !ok || loaded == nil {
		return false
	}
	classifier, hasClassifier, err := loaded.loadedSequenceClassifierConfig()
	if err != nil || !hasClassifier || classifier.NumLabels <= 0 {
		return false
	}
	status := normalizeHIPKernelStatus(loaded.kernelSet().Status())
	return status.Embedding == hipKernelStatusLinked && status.Projection == hipKernelStatusLinked
}

func (m *rocmModel) gemma4Q4GenerateLinked() bool {
	if m == nil {
		return false
	}
	loaded, ok := m.native.(*hipLoadedModel)
	if !ok || loaded == nil {
		return false
	}
	if !isROCmGemma4Architecture(loaded.modelInfo.Architecture) || loaded.modelInfo.QuantBits != 4 || loaded.modelInfo.NumLayers <= 0 {
		return false
	}
	_, err := loaded.cachedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	return err == nil
}

func (m *rocmModel) Metrics() inference.GenerateMetrics {
	if m == nil {
		return inference.GenerateMetrics{}
	}
	m.stateMutex.Lock()
	defer m.stateMutex.Unlock()
	return m.lastMetrics
}

func (m *rocmModel) Err() error {
	if m == nil {
		return nil
	}
	m.stateMutex.Lock()
	defer m.stateMutex.Unlock()
	return m.lastError
}

func (m *rocmModel) Close() (err error) {
	if m == nil {
		return nil
	}
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	m.stateMutex.Lock()
	native := m.native
	cache := m.cache
	state := m.state
	if native == nil && cache == nil && state == nil {
		m.stateMutex.Unlock()
		return nil
	}
	m.stateMutex.Unlock()
	if err := state.Close(); err != nil {
		return err
	}
	if err := cache.Close(); err != nil {
		return err
	}
	if native != nil {
		if err := native.Close(); err != nil {
			return err
		}
	}
	m.stateMutex.Lock()
	m.native = nil
	m.adapter = inference.AdapterIdentity{}
	m.cache = nil
	m.state = nil
	m.stateMutex.Unlock()
	return nil
}

func (m *rocmModel) Encode(text string) []int32 {
	if m == nil || m.native == nil {
		return approximateTokenIDs(text)
	}
	return append([]int32(nil), m.native.Encode(text)...)
}

func (m *rocmModel) Decode(ids []int32) string {
	if m == nil || m.native == nil {
		return ""
	}
	return m.native.Decode(append([]int32(nil), ids...))
}

func (m *rocmModel) promptTokenCount(prompt string) int {
	if m != nil {
		if loaded, ok := m.native.(*hipLoadedModel); ok {
			if tokens, matched, err := hipGemma4Q4PromptTokenIDs(prompt, loaded); err == nil && matched {
				return len(tokens)
			}
		}
	}
	return len(m.Encode(prompt))
}

func (m *rocmModel) promptsTokenCount(prompts []string) int {
	total := 0
	for _, prompt := range prompts {
		total += m.promptTokenCount(prompt)
	}
	return total
}

func (m *rocmModel) chatPromptTokenCount(messages []inference.Message) int {
	if m == nil || m.native == nil {
		return approximateMessageTokens(messages)
	}
	prompt, err := m.applyChatTemplate(messages)
	if err != nil {
		return approximateMessageTokens(messages)
	}
	if loaded, ok := m.native.(*hipLoadedModel); ok {
		if _, q4, q4Err := loaded.loadedGemma4Q4PackageForwardConfig(); q4 && q4Err == nil {
			return m.promptTokenCount("text:" + prompt)
		}
	}
	return m.promptTokenCount(prompt)
}

func (m *rocmModel) evalSampleTokenCount(sample inference.DatasetSample) int {
	switch {
	case sample.Text != "":
		return m.promptTokenCount(sample.Text)
	case sample.Prompt != "" || sample.Response != "":
		return m.promptTokenCount(core.Trim(sample.Prompt + " " + sample.Response))
	case len(sample.Messages) > 0:
		return m.chatPromptTokenCount(sample.Messages)
	default:
		return m.promptTokenCount(sample.Reasoning)
	}
}

func (m *rocmModel) ApplyChatTemplate(messages []inference.Message) (text string, err error) {
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	return m.applyChatTemplate(messages)
}

func (m *rocmModel) applyChatTemplate(messages []inference.Message) (string, error) {
	if m == nil || m.native == nil {
		return formatFallbackChatTemplate(messages), nil
	}
	return m.native.ApplyChatTemplate(append([]inference.Message(nil), messages...))
}

func (m *rocmModel) LoadAdapter(path string) (identity inference.AdapterIdentity, err error) {
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	if core.Trim(path) == "" {
		return inference.AdapterIdentity{}, core.E("rocm.LoadAdapter", "adapter path is required", nil)
	}
	if m == nil || m.native == nil {
		return inference.AdapterIdentity{}, core.E("rocm.LoadAdapter", "native model is nil", nil)
	}
	m.stateMutex.Lock()
	state := m.state
	cache := m.cache
	m.stateMutex.Unlock()
	if err := state.Close(); err != nil {
		return inference.AdapterIdentity{}, core.E("rocm.LoadAdapter", "close state runtime", err)
	}
	if err := cache.Close(); err != nil {
		return inference.AdapterIdentity{}, core.E("rocm.LoadAdapter", "close cache runtime", err)
	}
	m.stateMutex.Lock()
	if m.state == state {
		m.state = nil
	}
	if m.cache == cache {
		m.cache = nil
	}
	m.stateMutex.Unlock()
	identity, err = m.native.LoadAdapter(path)
	if err != nil {
		return inference.AdapterIdentity{}, err
	}
	if identity.Format == "" {
		identity.Format = "lora"
	}
	if identity.Path == "" {
		identity.Path = path
	}
	identity = cloneAdapterIdentity(identity)
	m.stateMutex.Lock()
	m.adapter = identity
	m.cache = nil
	m.state = nil
	m.stateMutex.Unlock()
	return cloneAdapterIdentity(identity), nil
}

func (m *rocmModel) UnloadAdapter() (err error) {
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	if m == nil || m.native == nil {
		return core.E("rocm.UnloadAdapter", "native model is nil", nil)
	}
	m.stateMutex.Lock()
	state := m.state
	cache := m.cache
	m.stateMutex.Unlock()
	if err := state.Close(); err != nil {
		return core.E("rocm.UnloadAdapter", "close state runtime", err)
	}
	if err := cache.Close(); err != nil {
		return core.E("rocm.UnloadAdapter", "close cache runtime", err)
	}
	m.stateMutex.Lock()
	if m.state == state {
		m.state = nil
	}
	if m.cache == cache {
		m.cache = nil
	}
	m.stateMutex.Unlock()
	if err := m.native.UnloadAdapter(); err != nil {
		return err
	}
	m.stateMutex.Lock()
	m.adapter = inference.AdapterIdentity{}
	m.cache = nil
	m.state = nil
	m.stateMutex.Unlock()
	return nil
}

func (m *rocmModel) ActiveAdapter() inference.AdapterIdentity {
	if m == nil {
		return inference.AdapterIdentity{}
	}
	m.stateMutex.Lock()
	adapter := m.adapter
	native := m.native
	m.stateMutex.Unlock()
	if !adapterIdentityIsZero(adapter) {
		return cloneAdapterIdentity(adapter)
	}
	if native == nil {
		return inference.AdapterIdentity{}
	}
	return cloneAdapterIdentity(native.ActiveAdapter())
}

func (m *rocmModel) kernelStatus() hipKernelStatus {
	if m == nil || m.native == nil {
		return defaultHIPKernelStatus()
	}
	status := normalizeHIPKernelStatus(m.native.KernelStatus())
	if _, ok := m.native.(nativeEmbeddingModel); !ok {
		status.Embedding = hipKernelStatusNotLinked
	}
	if _, ok := m.native.(nativeRerankModel); !ok {
		status.Rerank = hipKernelStatusNotLinked
	}
	return status
}

func adapterIdentityIsZero(identity inference.AdapterIdentity) bool {
	return identity.Path == "" && identity.Hash == "" && identity.Format == "" && identity.Rank == 0 && identity.Alpha == 0 && len(identity.TargetKeys) == 0 && identity.BaseModelHash == "" && len(identity.Labels) == 0
}

func cloneAdapterIdentity(identity inference.AdapterIdentity) inference.AdapterIdentity {
	identity.TargetKeys = append([]string(nil), identity.TargetKeys...)
	identity.Labels = cloneStringMap(identity.Labels)
	return identity
}

func (m *rocmModel) SetProbeSink(sink inference.ProbeSink) {
	if m == nil {
		return
	}
	m.stateMutex.Lock()
	m.probeSink = sink
	m.stateMutex.Unlock()
}

func (m *rocmModel) Benchmark(ctx context.Context, cfg inference.BenchConfig) (report *inference.BenchReport, err error) {
	m.clearLastError()
	if m == nil {
		return nil, core.E("rocm.Benchmark", "model is nil", nil)
	}
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	if ctx == nil {
		ctx = context.Background()
	}
	prompts := cfg.Prompts
	if len(prompts) == 0 {
		prompts = []string{"hello"}
	}
	measuredRuns := cfg.MeasuredRuns
	if measuredRuns <= 0 {
		measuredRuns = 1
	}
	warmupRuns := cfg.WarmupRuns
	if warmupRuns < 0 {
		warmupRuns = 0
	}
	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 32
	}
	var stopSequences []string
	if err := m.benchmarkWarmupRuns(ctx, prompts, maxTokens, warmupRuns, stopSequences); err != nil {
		return nil, err
	}
	probeCounter, restoreProbeSink := m.beginBenchmarkProbeCounter()
	defer restoreProbeSink()

	aggregate, err := m.benchmarkMeasuredRuns(ctx, prompts, maxTokens, measuredRuns, stopSequences)
	if err != nil {
		return nil, err
	}
	cacheStats, err := m.CacheStats(ctx)
	if err != nil {
		return nil, err
	}
	kernelStatus := m.kernelStatus()
	gemma4Q4GenerateLinked := m.gemma4Q4GenerateLinked()
	decodeHelperStatus := rocmDecodeHelperStatusLabel(kernelStatus, gemma4Q4GenerateLinked)
	operationCount := benchmarkOperationCount(prompts, measuredRuns)
	labels := map[string]string{
		"backend":                "rocm",
		"cache.blocks":           "experimental",
		"cache.disk":             "experimental",
		"cache.mode":             firstNonEmptyString(cacheStats.CacheMode, "block-prefix"),
		"cache.warm":             "experimental",
		"decode_duration_ms":     durationMillisecondsLabel(aggregate.DecodeDuration),
		"first_token_latency_ms": averageDurationMillisecondsLabel(aggregate.PrefillDuration, operationCount),
		"measured_runs":          core.Sprintf("%d", measuredRuns),
		"native_runtime":         "hip",
		"operation_count":        core.Sprintf("%d", operationCount),
		"memory_active_bytes":    core.Sprintf("%d", aggregate.ActiveMemoryBytes),
		"memory_peak_bytes":      core.Sprintf("%d", aggregate.PeakMemoryBytes),
		"prefill_duration_ms":    durationMillisecondsLabel(aggregate.PrefillDuration),
		"probe.events":           "stream_tokens",
		"prompt_count":           core.Sprintf("%d", len(prompts)),
		"prompt.cache":           "experimental",
		"prompt.lookup.decode":   decodeHelperStatus,
		"queue_latency_ms":       durationMillisecondsLabel(0),
		"request.cancel":         "supported",
		"scheduler":              "supported",
		"speculative.decode":     decodeHelperStatus,
		"total_duration_ms":      durationMillisecondsLabel(metricsTotalDuration(aggregate)),
		"warmup_runs":            core.Sprintf("%d", warmupRuns),
	}
	for key, value := range kernelStatus.Labels() {
		labels[key] = value
	}
	if kernelStatus.Decode == hipKernelStatusLinked {
		rocmAddReportLabels(labels, rocmDecodeCapabilityLabels(kernelStatus, m.modelIdentity()))
	}
	if gemma4Q4GenerateLinked {
		rocmAddReportLabels(labels, rocmGemma4Q4BenchmarkCapabilityLabels(m.modelIdentity()))
		labels["prompt.lookup.decode"] = "experimental"
		labels["prompt.lookup.decode.source"] = "gemma4_q4_generate"
		labels["speculative.decode"] = "experimental"
		labels["speculative.decode.source"] = "gemma4_q4_generate"
	}
	for key, value := range cacheStats.Labels {
		if value != "" {
			labels["cache."+key] = value
		}
	}
	m.addLoRAOverheadBenchLabels(ctx, labels, prompts, maxTokens, measuredRuns, stopSequences, aggregate)
	m.clearLastError()
	m.setLastMetrics(aggregate)
	report = &inference.BenchReport{
		Model:                 m.modelIdentity(),
		Adapter:               m.ActiveAdapter(),
		PromptTokens:          aggregate.PromptTokens,
		GeneratedTokens:       aggregate.GeneratedTokens,
		PrefillTokensPerSec:   tokensPerSecond(aggregate.PromptTokens, aggregate.PrefillDuration),
		DecodeTokensPerSec:    tokensPerSecond(aggregate.GeneratedTokens, aggregate.DecodeDuration),
		PeakMemoryBytes:       aggregate.PeakMemoryBytes,
		PromptCacheHitRate:    cacheStats.HitRate,
		KVRestoreMilliseconds: cacheStats.RestoreMillis,
		Labels:                labels,
	}
	m.emitCachePressureProbe(report.PromptTokens, report.GeneratedTokens, cacheStats)
	m.emitMemoryPressureProbe(aggregate.ActiveMemoryBytes, aggregate.PeakMemoryBytes, 0)
	labels["probe_count"] = core.Sprintf("%d", probeCounter.Count())
	labels["probe_count_status"] = "measured"
	return report, nil
}

func rocmDecodeHelperStatusLabel(status hipKernelStatus, gemma4Q4GenerateLinked bool) string {
	if gemma4Q4GenerateLinked {
		return "experimental"
	}
	if normalizeHIPKernelStatus(status).Decode == hipKernelStatusLinked {
		return "experimental"
	}
	return "planned"
}

type rocmBenchmarkProbeCounter struct {
	mu         sync.Mutex
	count      int
	downstream inference.ProbeSink
}

func (counter *rocmBenchmarkProbeCounter) EmitProbe(event inference.ProbeEvent) {
	if counter == nil {
		return
	}
	counter.mu.Lock()
	counter.count++
	downstream := counter.downstream
	counter.mu.Unlock()
	if downstream != nil {
		downstream.EmitProbe(event)
	}
}

func (counter *rocmBenchmarkProbeCounter) Count() int {
	if counter == nil {
		return 0
	}
	counter.mu.Lock()
	defer counter.mu.Unlock()
	return counter.count
}

func (m *rocmModel) beginBenchmarkProbeCounter() (*rocmBenchmarkProbeCounter, func()) {
	counter := &rocmBenchmarkProbeCounter{}
	if m == nil {
		return counter, func() {}
	}
	m.stateMutex.Lock()
	previous := m.probeSink
	counter.downstream = previous
	m.probeSink = counter
	m.stateMutex.Unlock()
	return counter, func() {
		m.stateMutex.Lock()
		if m.probeSink == counter {
			m.probeSink = previous
		}
		m.stateMutex.Unlock()
	}
}

func (m *rocmModel) suspendProbeSink() func() {
	if m == nil {
		return func() {}
	}
	m.stateMutex.Lock()
	previous := m.probeSink
	m.probeSink = nil
	m.stateMutex.Unlock()
	return func() {
		m.stateMutex.Lock()
		if m.probeSink == nil {
			m.probeSink = previous
		}
		m.stateMutex.Unlock()
	}
}

func (m *rocmModel) benchmarkWarmupRuns(ctx context.Context, prompts []string, maxTokens, warmupRuns int, stopSequences []string) error {
	opts := benchmarkGenerateOptions(maxTokens, stopSequences)
	for i := 0; i < warmupRuns; i++ {
		for _, prompt := range prompts {
			for range m.Generate(ctx, m.generatedPrompt(prompt), opts...) {
			}
			if err := m.Err(); err != nil {
				return err
			}
		}
	}
	return nil
}

func (m *rocmModel) benchmarkMeasuredRuns(ctx context.Context, prompts []string, maxTokens, measuredRuns int, stopSequences []string) (inference.GenerateMetrics, error) {
	var aggregate inference.GenerateMetrics
	opts := benchmarkGenerateOptions(maxTokens, stopSequences)
	for i := 0; i < measuredRuns; i++ {
		for _, prompt := range prompts {
			for range m.Generate(ctx, m.generatedPrompt(prompt), opts...) {
			}
			if err := m.Err(); err != nil {
				return inference.GenerateMetrics{}, err
			}
			metrics := m.Metrics()
			aggregate.PromptTokens += metrics.PromptTokens
			aggregate.GeneratedTokens += metrics.GeneratedTokens
			aggregate.PrefillDuration += metrics.PrefillDuration
			aggregate.DecodeDuration += metrics.DecodeDuration
			aggregate.TotalDuration += metrics.TotalDuration
			if metrics.PeakMemoryBytes > aggregate.PeakMemoryBytes {
				aggregate.PeakMemoryBytes = metrics.PeakMemoryBytes
			}
			if metrics.ActiveMemoryBytes > aggregate.ActiveMemoryBytes {
				aggregate.ActiveMemoryBytes = metrics.ActiveMemoryBytes
			}
		}
	}
	return aggregate, nil
}

func (m *rocmModel) generatedPrompt(prompt string) string {
	if m == nil || !m.gemma4Q4TextPromptSupported() || hipGemma4Q4PromptHasExplicitMode(prompt) {
		return prompt
	}
	return "text:" + prompt
}

func (m *rocmModel) gemma4Q4TextPromptSupported() bool {
	if m == nil {
		return false
	}
	loaded, ok := m.native.(*hipLoadedModel)
	if !ok || loaded == nil || loaded.tokenText == nil {
		return false
	}
	return isROCmGemma4Architecture(loaded.modelInfo.Architecture) && loaded.modelInfo.QuantBits == 4
}

func hipGemma4Q4PromptHasExplicitMode(prompt string) bool {
	trimmed := strings.ToLower(strings.TrimSpace(prompt))
	return strings.HasPrefix(trimmed, "tokens:") || strings.HasPrefix(trimmed, "text:")
}

func benchmarkGenerateOptions(maxTokens int, stopSequences []string) []inference.GenerateOption {
	opts := []inference.GenerateOption{inference.WithMaxTokens(maxTokens)}
	return opts
}

func (m *rocmModel) benchmarkMeasuredRunsWithoutProbes(ctx context.Context, prompts []string, maxTokens, measuredRuns int, stopSequences []string) (inference.GenerateMetrics, error) {
	restoreProbeSink := m.suspendProbeSink()
	defer restoreProbeSink()
	return m.benchmarkMeasuredRuns(ctx, prompts, maxTokens, measuredRuns, stopSequences)
}

func (m *rocmModel) addLoRAOverheadBenchLabels(ctx context.Context, labels map[string]string, prompts []string, maxTokens, measuredRuns int, stopSequences []string, active inference.GenerateMetrics) {
	if labels == nil {
		return
	}
	adapter := m.ActiveAdapter()
	if adapterIdentityIsZero(adapter) {
		labels["lora_overhead"] = "not_applicable"
		labels["lora_overhead_status"] = "no_active_adapter"
		return
	}
	labels["lora_overhead"] = "attempted"
	labels["lora_overhead_status"] = "active_adapter"
	if adapter.Format != "" {
		labels["lora_adapter_format"] = adapter.Format
	}
	if adapter.Hash != "" {
		labels["lora_adapter_hash"] = adapter.Hash
	}
	if adapter.Rank > 0 {
		labels["lora_adapter_rank"] = core.Sprintf("%d", adapter.Rank)
	}
	if adapter.Alpha > 0 {
		labels["lora_adapter_alpha"] = core.Sprintf("%.6g", adapter.Alpha)
	}
	if adapter.Path == "" || m == nil || m.native == nil {
		labels["lora_overhead_status"] = "missing_adapter_path"
		return
	}
	m.stateMutex.Lock()
	state := m.state
	m.stateMutex.Unlock()
	if err := state.Close(); err != nil {
		labels["lora_overhead_status"] = "state_close_failed"
		labels["lora_overhead_error"] = err.Error()
		return
	}
	m.stateMutex.Lock()
	if m.state == state {
		m.state = nil
	}
	m.stateMutex.Unlock()
	if err := m.native.UnloadAdapter(); err != nil {
		labels["lora_overhead_status"] = "unload_failed"
		labels["lora_overhead_error"] = err.Error()
		return
	}
	m.stateMutex.Lock()
	m.adapter = inference.AdapterIdentity{}
	m.cache = nil
	m.state = nil
	m.stateMutex.Unlock()
	baseline, baselineErr := m.benchmarkMeasuredRunsWithoutProbes(ctx, prompts, maxTokens, measuredRuns, stopSequences)
	_, restoreErr := m.native.LoadAdapter(adapter.Path)
	if restoreErr == nil {
		m.stateMutex.Lock()
		m.adapter = adapter
		m.stateMutex.Unlock()
	}
	if baselineErr != nil {
		labels["lora_overhead_status"] = "baseline_failed"
		labels["lora_overhead_error"] = baselineErr.Error()
		return
	}
	if restoreErr != nil {
		labels["lora_overhead_status"] = "restore_failed"
		labels["lora_overhead_error"] = restoreErr.Error()
		return
	}
	activeDuration := metricsTotalDuration(active)
	baselineDuration := metricsTotalDuration(baseline)
	overhead := activeDuration - baselineDuration
	labels["lora_overhead"] = "measured"
	labels["lora_overhead_status"] = "measured"
	labels["lora_adapter_duration_ms"] = durationMillisecondsLabel(activeDuration)
	labels["lora_baseline_duration_ms"] = durationMillisecondsLabel(baselineDuration)
	labels["lora_overhead_ms"] = durationMillisecondsLabel(overhead)
	if baselineDuration > 0 {
		labels["lora_overhead_ratio"] = core.Sprintf("%.6f", float64(activeDuration)/float64(baselineDuration))
	}
}

func metricsTotalDuration(metrics inference.GenerateMetrics) time.Duration {
	if metrics.TotalDuration > 0 {
		return metrics.TotalDuration
	}
	return metrics.PrefillDuration + metrics.DecodeDuration
}

func benchmarkOperationCount(prompts []string, measuredRuns int) int {
	if len(prompts) <= 0 || measuredRuns <= 0 {
		return 0
	}
	return len(prompts) * measuredRuns
}

func averageDurationMillisecondsLabel(duration time.Duration, count int) string {
	if count <= 0 {
		return durationMillisecondsLabel(0)
	}
	return durationMillisecondsLabel(duration / time.Duration(count))
}

func durationMillisecondsLabel(duration time.Duration) string {
	return core.Sprintf("%.3f", float64(duration)/float64(time.Millisecond))
}

func (m *rocmModel) Evaluate(ctx context.Context, dataset inference.DatasetStream, cfg inference.EvalConfig) (report *inference.EvalReport, err error) {
	m.clearLastError()
	if m == nil {
		return nil, core.E("rocm.Evaluate", "model is nil", nil)
	}
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	if ctx == nil {
		ctx = context.Background()
	}
	if dataset == nil {
		return nil, core.E("rocm.Evaluate", "dataset stream is nil", nil)
	}
	maxSamples := cfg.MaxSamples
	if maxSamples <= 0 {
		maxSamples = 1 << 30
	}
	lossBatchSize := firstPositiveInt(cfg.BatchSize, 1)
	metrics := inference.EvalMetrics{}
	loss := rocmEvalLossAccumulator{batchSize: lossBatchSize}
	lossBatch := make([]rocmEvalLossCandidate, 0, lossBatchSize)
	for metrics.Samples < maxSamples {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		sample, ok, err := dataset.Next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		metrics.Samples++
		metrics.Tokens += m.evalSampleTokenCount(sample)
		candidate, ok := m.evalLossCandidate(sample)
		if !ok {
			loss.skipped++
			continue
		}
		lossBatch = append(lossBatch, candidate)
		if len(lossBatch) >= lossBatchSize {
			if err := m.observeEvalLossBatch(ctx, lossBatch, &loss); err != nil {
				return nil, err
			}
			lossBatch = lossBatch[:0]
		}
	}
	if len(lossBatch) > 0 {
		if err := m.observeEvalLossBatch(ctx, lossBatch, &loss); err != nil {
			return nil, err
		}
	}
	if metrics.Samples == 0 {
		return nil, core.E("rocm.Evaluate", "dataset produced no samples", nil)
	}
	kernelStatus := m.kernelStatus()
	classifyLinked := m.classifyLinked()
	lossLabel := "unsupported_until_prefill_kernels"
	lossStatus := "unsupported"
	if kernelStatus.Prefill == hipKernelStatusLinked || classifyLinked {
		lossLabel = "not_requested"
		lossStatus = "not_requested"
	}
	labels := map[string]string{
		"backend":           "rocm",
		"eval.batch_size":   core.Sprintf("%d", lossBatchSize),
		"eval.samples":      core.Sprintf("%d", metrics.Samples),
		"eval.tokens":       core.Sprintf("%d", metrics.Tokens),
		"loss":              lossLabel,
		"loss_kernel":       kernelStatus.CrossEntropy,
		"loss_kernel_name":  hipKernelNameCrossEntropy,
		"loss_scope":        "toy_cross_entropy",
		"loss_status":       lossStatus,
		"perplexity":        lossLabel,
		"perplexity_status": lossStatus,
	}
	loss.apply(ctx, m, &metrics, labels)
	probes, failures, probeError, err := m.evaluateQualityProbes(ctx, cfg.Probes, firstPositiveInt(cfg.MaxSeqLen, 32), nil)
	if err != nil {
		return nil, err
	}
	if len(cfg.Probes) > 0 {
		labels["quality_probe_count"] = core.Sprintf("%d", len(probes))
		labels["quality_probes"] = "completed"
		labels["quality_probe_failures"] = core.Sprintf("%d", failures)
		labels["quality_probe_passes"] = core.Sprintf("%d", len(probes)-failures)
		if failures > 0 {
			labels["quality_probe_status"] = "generation_unavailable"
			if probeError != "" {
				labels["quality_probe_error"] = probeError
			}
		} else {
			labels["quality_probe_status"] = "passed"
		}
	}
	for key, value := range kernelStatus.Labels() {
		labels[key] = value
	}
	if kernelStatus.Prefill == hipKernelStatusLinked || classifyLinked {
		rocmAddReportLabels(labels, rocmClassifyCapabilityLabels(kernelStatus, m.modelIdentity(), rocmCapabilityReportOption{ClassifyLinked: classifyLinked}))
	}
	if len(cfg.Probes) > 0 && kernelStatus.Decode == hipKernelStatusLinked {
		rocmAddReportLabels(labels, rocmDecodeCapabilityLabels(kernelStatus, m.modelIdentity()))
	}
	if classifyLinked && kernelStatus.Prefill != hipKernelStatusLinked {
		labels["classify_path"] = "bert_sequence_classifier"
		labels["classify_status"] = string(inference.FeatureRuntimeExperimental)
	}
	report = &inference.EvalReport{
		Model:   m.modelIdentity(),
		Adapter: m.ActiveAdapter(),
		Metrics: metrics,
		Probes:  probes,
		Labels:  labels,
	}
	m.clearLastError()
	return report, nil
}

type rocmEvalLossCandidate struct {
	prompt string
	target int
}

type rocmEvalLossAccumulator struct {
	logits     [][]float32
	targets    []int
	candidates int
	batches    int
	batchSize  int
	skipped    int
	source     string
	status     string
	err        string
}

func (m *rocmModel) observeEvalLossBatch(ctx context.Context, candidates []rocmEvalLossCandidate, loss *rocmEvalLossAccumulator) error {
	if loss == nil {
		return nil
	}
	if len(candidates) == 0 {
		return nil
	}
	loss.candidates += len(candidates)
	loss.batches++
	if ok, err := m.observeGemma4Q4EvalLossBatch(ctx, candidates, loss); ok || err != nil {
		return err
	}
	prompts := make([]string, len(candidates))
	for i, candidate := range candidates {
		prompts[i] = candidate.prompt
	}
	results, err := m.Classify(ctx, prompts, inference.WithLogits())
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		if loss.err == "" {
			loss.err = err.Error()
		}
		loss.status = "classify_unavailable"
		return nil
	}
	if len(results) < len(candidates) {
		loss.status = "logits_unavailable"
		return nil
	}
	for i, candidate := range candidates {
		if len(results[i].Logits) == 0 {
			loss.status = "logits_unavailable"
			continue
		}
		loss.logits = append(loss.logits, append([]float32(nil), results[i].Logits...))
		loss.targets = append(loss.targets, candidate.target)
	}
	return nil
}

func (m *rocmModel) observeGemma4Q4EvalLossBatch(ctx context.Context, candidates []rocmEvalLossCandidate, loss *rocmEvalLossAccumulator) (bool, error) {
	if m == nil || loss == nil || len(candidates) == 0 {
		return false, nil
	}
	loaded, ok := m.native.(*hipLoadedModel)
	if !ok || loaded == nil ||
		!isROCmGemma4Architecture(loaded.modelInfo.Architecture) ||
		loaded.modelInfo.QuantBits != 4 {
		return false, nil
	}
	loss.source = "gemma4_q4_package_prefill"
	for _, candidate := range candidates {
		if err := ctx.Err(); err != nil {
			return true, err
		}
		prompt := m.generatedPrompt(candidate.prompt)
		prefill, err := loaded.Prefill(ctx, hipPrefillRequest{
			Prompt:    prompt,
			CacheMode: rocmKVCacheModeKQ8VQ4,
		})
		if err != nil {
			if loss.err == "" {
				loss.err = err.Error()
			}
			loss.status = "gemma4_q4_prefill_unavailable"
			return true, nil
		}
		if err := prefill.Gemma4Q4DeviceState.Close(); err != nil {
			if loss.err == "" {
				loss.err = err.Error()
			}
			loss.status = "gemma4_q4_prefill_close_failed"
			return true, nil
		}
		if len(prefill.Logits) == 0 {
			loss.status = "logits_unavailable"
			continue
		}
		if candidate.target < 0 || candidate.target >= len(prefill.Logits) {
			loss.status = "target_out_of_vocab"
			continue
		}
		loss.logits = append(loss.logits, append([]float32(nil), prefill.Logits...))
		loss.targets = append(loss.targets, candidate.target)
	}
	return true, nil
}

func (m *rocmModel) evalLossCandidate(sample inference.DatasetSample) (rocmEvalLossCandidate, bool) {
	target, ok := evalLossTargetFromLabels(sample.Labels)
	if !ok {
		if response := core.Trim(sample.Response); response != "" {
			ids := m.Encode(response)
			if len(ids) == 0 || ids[0] < 0 {
				return rocmEvalLossCandidate{}, false
			}
			target = int(ids[0])
			ok = true
		}
	}
	if !ok {
		return rocmEvalLossCandidate{}, false
	}
	prompt := core.Trim(sample.Prompt)
	if prompt == "" && len(sample.Messages) > 0 {
		prompt = core.Trim(formatFallbackChatTemplate(sample.Messages))
	}
	if prompt == "" {
		prompt = core.Trim(sample.Text)
	}
	if prompt == "" {
		return rocmEvalLossCandidate{}, false
	}
	return rocmEvalLossCandidate{prompt: prompt, target: target}, true
}

func evalLossTargetFromLabels(labels map[string]string) (int, bool) {
	for _, key := range []string{"target_token_id", "target_id", "next_token_id"} {
		raw := core.Trim(labels[key])
		if raw == "" {
			continue
		}
		id, err := strconv.Atoi(raw)
		if err != nil || id < 0 {
			return 0, false
		}
		return id, true
	}
	return 0, false
}

func (loss rocmEvalLossAccumulator) apply(ctx context.Context, model *rocmModel, metrics *inference.EvalMetrics, labels map[string]string) {
	if metrics == nil || labels == nil {
		return
	}
	if loss.candidates > 0 {
		labels["eval.loss_candidates"] = core.Sprintf("%d", loss.candidates)
	}
	if loss.batches > 0 {
		labels["eval.loss_batches"] = core.Sprintf("%d", loss.batches)
	}
	if loss.batchSize > 0 {
		labels["eval.loss_batch_size"] = core.Sprintf("%d", loss.batchSize)
	}
	if loss.source != "" {
		labels["eval.loss_logits_source"] = loss.source
	}
	if loss.skipped > 0 {
		labels["eval.loss_skipped"] = core.Sprintf("%d", loss.skipped)
	}
	if len(loss.logits) == 0 {
		if loss.status != "" {
			labels["loss_status"] = loss.status
			labels["perplexity_status"] = loss.status
		}
		if loss.err != "" {
			labels["loss_error"] = loss.err
		}
		return
	}
	if result, ok, err := model.runEvalCrossEntropyLoss(ctx, loss.logits, loss.targets); ok {
		labels["loss_backend"] = "hip"
		labels["loss_kernel"] = hipKernelStatusLinked
		labels["loss_kernel_name"] = hipKernelNameCrossEntropy
		if err != nil {
			labels["loss_status"] = "error"
			labels["perplexity_status"] = "error"
			labels["loss_error"] = err.Error()
			return
		}
		metrics.Loss = result.Loss
		metrics.Perplexity = result.Perplexity
		labels["loss"] = core.Sprintf("%.6f", result.Loss)
		labels["loss_status"] = "experimental"
		labels["perplexity"] = core.Sprintf("%.6f", result.Perplexity)
		labels["perplexity_status"] = "experimental"
		labels["eval.loss_tokens"] = core.Sprintf("%d", len(loss.logits))
		return
	}
	value, perplexity, err := rocmReferenceCrossEntropyLoss(loss.logits, loss.targets)
	if err != nil {
		labels["loss_status"] = "error"
		labels["perplexity_status"] = "error"
		labels["loss_error"] = err.Error()
		return
	}
	labels["loss_backend"] = "reference"
	metrics.Loss = value
	metrics.Perplexity = perplexity
	labels["loss"] = core.Sprintf("%.6f", value)
	labels["loss_status"] = "experimental"
	labels["perplexity"] = core.Sprintf("%.6f", perplexity)
	labels["perplexity_status"] = "experimental"
	labels["eval.loss_tokens"] = core.Sprintf("%d", len(loss.logits))
}

func (m *rocmModel) runEvalCrossEntropyLoss(ctx context.Context, logits [][]float32, targets []int) (hipCrossEntropyLossResult, bool, error) {
	if m == nil || m.native == nil {
		return hipCrossEntropyLossResult{}, false, nil
	}
	runner, ok := m.native.(nativeEvalLossKernelModel)
	if !ok {
		return hipCrossEntropyLossResult{}, false, nil
	}
	return runner.RunEvalCrossEntropyLoss(ctx, logits, targets)
}

func (m *rocmModel) evaluateQualityProbes(ctx context.Context, probes []inference.QualityProbe, maxTokens int, stopSequences []string) ([]inference.QualityProbeResult, int, string, error) {
	if len(probes) == 0 {
		return nil, 0, "", nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if maxTokens <= 0 {
		maxTokens = 32
	}
	opts := benchmarkGenerateOptions(maxTokens, stopSequences)
	results := make([]inference.QualityProbeResult, 0, len(probes))
	failures := 0
	firstFailure := ""
	for _, probe := range probes {
		if err := ctx.Err(); err != nil {
			return nil, 0, "", err
		}
		name := firstNonEmptyString(probe.Name, probe.Prompt)
		prompt := m.generatedPrompt(firstNonEmptyString(probe.Prompt, probe.Name))
		builder := core.NewBuilder()
		for token := range m.Generate(ctx, prompt, opts...) {
			builder.WriteString(token.Text)
		}
		if err := ctx.Err(); err != nil {
			return nil, 0, "", err
		}
		text := builder.String()
		result := inference.QualityProbeResult{Name: name, Text: text}
		if err := m.Err(); err != nil {
			failures++
			if firstFailure == "" {
				firstFailure = err.Error()
			}
			result.Passed = false
			result.Score = 0
			results = append(results, result)
			continue
		}
		result.Passed = core.Trim(text) != ""
		if result.Passed {
			result.Score = 1
		}
		results = append(results, result)
	}
	return results, failures, firstFailure, nil
}

func (m *rocmModel) wrapTokenStream(stream iter.Seq[inference.Token], streamError func() error, promptTokens int, start time.Time, stopSequences []string) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		var count int
		var firstTokenAt time.Time
		sink := m.probeSinkSnapshot()
		emit := func(token inference.Token) bool {
			if firstTokenAt.IsZero() {
				firstTokenAt = time.Now()
			}
			count++
			if sink != nil {
				emitTokenProbeTo(sink, token, promptTokens, count)
			}
			return yield(token)
		}
		stops := nonEmptyStopSequences(stopSequences)
		if len(stops) == 0 {
			for token := range stream {
				if !emit(token) {
					break
				}
			}
		} else {
			var buffer string
			var lastToken inference.Token
			stopped := false
			for token := range stream {
				lastToken = token
				buffer += token.Text
				if cut, ok := firstStopSequenceCut(buffer, stops); ok {
					if cut > 0 {
						out := token
						out.Text = buffer[:cut]
						_ = emit(out)
					}
					stopped = true
					break
				}
				hold := stopSequencePrefixHold(buffer, stops)
				emitLen := len(buffer) - hold
				if emitLen <= 0 {
					continue
				}
				out := token
				out.Text = buffer[:emitLen]
				if !emit(out) {
					buffer = ""
					break
				}
				buffer = buffer[emitLen:]
			}
			if !stopped && buffer != "" {
				lastToken.Text = buffer
				_ = emit(lastToken)
			}
		}
		if streamError != nil {
			if err := streamError(); err != nil {
				m.setLastFailure(err)
			}
		}
		if firstTokenAt.IsZero() && count > 0 {
			firstTokenAt = time.Now()
		}
		m.recordMetrics(promptTokens, count, start, firstTokenAt)
	}
}

func applyBatchStopSequences(results []inference.BatchResult, stopSequences []string) {
	stops := nonEmptyStopSequences(stopSequences)
	if len(stops) == 0 {
		return
	}
	for index := range results {
		results[index].Tokens = truncateTokensAtStopSequences(results[index].Tokens, stops)
	}
}

func truncateTokensAtStopSequences(tokens []inference.Token, stops []string) []inference.Token {
	if len(tokens) == 0 || len(stops) == 0 {
		return tokens
	}
	out := make([]inference.Token, 0, len(tokens))
	var buffer string
	var lastToken inference.Token
	for _, token := range tokens {
		lastToken = token
		buffer += token.Text
		if cut, ok := firstStopSequenceCut(buffer, stops); ok {
			if cut > 0 {
				token.Text = buffer[:cut]
				out = append(out, token)
			}
			return out
		}
		hold := stopSequencePrefixHold(buffer, stops)
		emitLen := len(buffer) - hold
		if emitLen <= 0 {
			continue
		}
		token.Text = buffer[:emitLen]
		out = append(out, token)
		buffer = buffer[emitLen:]
	}
	if buffer != "" {
		lastToken.Text = buffer
		out = append(out, lastToken)
	}
	return out
}

func nonEmptyStopSequences(sequences []string) []string {
	if len(sequences) == 0 {
		return nil
	}
	out := make([]string, 0, len(sequences))
	for _, sequence := range sequences {
		if sequence != "" {
			out = append(out, sequence)
		}
	}
	return out
}

func firstStopSequenceCut(text string, stops []string) (int, bool) {
	best := -1
	for _, stop := range stops {
		index := strings.Index(text, stop)
		if index >= 0 && (best < 0 || index < best) {
			best = index
		}
	}
	if best < 0 {
		return 0, false
	}
	return best, true
}

func stopSequencePrefixHold(text string, stops []string) int {
	hold := 0
	for _, stop := range stops {
		max := len(stop) - 1
		if max > len(text) {
			max = len(text)
		}
		for size := 1; size <= max; size++ {
			if size > hold && strings.HasSuffix(text, stop[:size]) {
				hold = size
			}
		}
	}
	return hold
}

func (m *rocmModel) emitTokenProbe(token inference.Token, promptTokens, generatedTokens int) {
	if m == nil {
		return
	}
	emitTokenProbeTo(m.probeSinkSnapshot(), token, promptTokens, generatedTokens)
}

func emitTokenProbeTo(sink inference.ProbeSink, token inference.Token, promptTokens, generatedTokens int) {
	if sink == nil {
		return
	}
	sink.EmitProbe(inference.ProbeEvent{
		Kind:  inference.ProbeEventToken,
		Phase: inference.ProbePhaseDecode,
		Token: &inference.ProbeToken{ID: token.ID, Text: token.Text, PromptTokens: promptTokens, GeneratedTokens: generatedTokens},
	})
}

func (m *rocmModel) emitCachePressureProbe(promptTokens, generatedTokens int, stats inference.CacheStats) {
	labels := mergeStringMaps(stats.Labels, map[string]string{
		"backend": "rocm",
		"source":  "benchmark",
	})
	m.emitProbe(inference.ProbeEvent{
		Kind:   inference.ProbeEventCachePressure,
		Phase:  inference.ProbePhasePrefill,
		Labels: labels,
		Cache: &inference.ProbeCachePressure{
			PromptTokens:    promptTokens,
			GeneratedTokens: generatedTokens,
			CachedTokens:    cacheStatsCachedTokens(stats),
			CacheMode:       firstNonEmptyString(stats.CacheMode, "block-prefix"),
			HitRate:         stats.HitRate,
		},
	})
}

func cacheStatsCachedTokens(stats inference.CacheStats) int {
	if cached, err := positiveIntLabel(stats.Labels, "cached_tokens", 0); err == nil && cached > 0 {
		return cached
	}
	if cached, err := positiveIntLabel(stats.Labels, "kv_tokens", 0); err == nil && cached > 0 {
		return cached
	}
	return 0
}

func (m *rocmModel) emitMemoryPressureProbe(activeBytes, peakBytes, limitBytes uint64) {
	m.emitProbe(inference.ProbeEvent{
		Kind:  inference.ProbeEventMemoryPressure,
		Phase: inference.ProbePhaseDecode,
		Labels: map[string]string{
			"backend": "rocm",
			"source":  "benchmark",
		},
		Memory: &inference.ProbeMemoryPressure{
			ActiveBytes: activeBytes,
			PeakBytes:   peakBytes,
			LimitBytes:  limitBytes,
		},
	})
}

func (m *rocmModel) emitProbe(event inference.ProbeEvent) {
	if m == nil {
		return
	}
	m.stateMutex.Lock()
	sink := m.probeSink
	m.stateMutex.Unlock()
	if sink == nil {
		return
	}
	sink.EmitProbe(event)
}

func (m *rocmModel) recordMetrics(promptTokens, generatedTokens int, start, firstTokenAt time.Time) {
	prefill, decode := splitDurations(start, firstTokenAt, time.Now())
	m.recordMetricsDurations(promptTokens, generatedTokens, prefill, decode)
}

func (m *rocmModel) recordMetricsDurations(promptTokens, generatedTokens int, prefill, decode time.Duration) {
	if m == nil {
		return
	}
	if prefill < 0 {
		prefill = 0
	}
	if decode < 0 {
		decode = 0
	}
	metrics := inference.GenerateMetrics{
		PromptTokens:        promptTokens,
		GeneratedTokens:     generatedTokens,
		PrefillDuration:     prefill,
		DecodeDuration:      decode,
		TotalDuration:       prefill + decode,
		PrefillTokensPerSec: tokensPerSecond(promptTokens, prefill),
		DecodeTokensPerSec:  tokensPerSecond(generatedTokens, decode),
		PeakMemoryBytes:     nativePeakMemoryBytes(),
		ActiveMemoryBytes:   nativePeakMemoryBytes(),
	}
	if m.native != nil {
		nativeMetrics := m.native.Metrics()
		if nativeMetrics.PeakMemoryBytes > metrics.PeakMemoryBytes {
			metrics.PeakMemoryBytes = nativeMetrics.PeakMemoryBytes
		}
		if nativeMetrics.ActiveMemoryBytes > 0 {
			metrics.ActiveMemoryBytes = nativeMetrics.ActiveMemoryBytes
		}
	}
	m.stateMutex.Lock()
	m.lastMetrics = metrics
	m.stateMutex.Unlock()
}

func (m *rocmModel) setLastMetrics(metrics inference.GenerateMetrics) {
	if m == nil {
		return
	}
	m.stateMutex.Lock()
	m.lastMetrics = metrics
	m.stateMutex.Unlock()
}

func (m *rocmModel) clearLastError() { m.setLastFailure(nil) }

func (m *rocmModel) setLastFailure(err error) {
	if m == nil {
		return
	}
	m.stateMutex.Lock()
	m.lastError = err
	m.stateMutex.Unlock()
}

func rocmContextErr(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}

func (m *rocmModel) modelIdentity() inference.ModelIdentity {
	info := m.Info()
	return inference.ModelIdentity{
		Architecture: info.Architecture,
		VocabSize:    info.VocabSize,
		NumLayers:    info.NumLayers,
		HiddenSize:   info.HiddenSize,
		QuantBits:    info.QuantBits,
		QuantGroup:   info.QuantGroup,
	}
}

type rocmCapabilityReportOption struct {
	ClassifyLinked         bool
	Gemma4Q4GenerateLinked bool
}

func rocmCapabilityReport(device nativeDeviceInfo, model inference.ModelIdentity, adapter inference.AdapterIdentity, available bool, kernelStatus hipKernelStatus, options ...rocmCapabilityReportOption) inference.CapabilityReport {
	option := rocmCapabilityReportOption{}
	if len(options) > 0 {
		option = options[0]
	}
	kernelStatus = normalizeHIPKernelStatus(kernelStatus)
	labels := map[string]string{
		"library":         "go-rocm",
		"metadata_status": "supported",
		"runtime_status":  "unavailable",
	}
	if available {
		labels["runtime_status"] = "available"
	}
	for key, value := range kernelStatus.Labels() {
		labels[key] = value
	}
	if device.FreeBytes > 0 {
		labels["free_bytes"] = core.Sprintf("%d", device.FreeBytes)
	}
	runtimeLabels := map[string]string{}
	if device.Driver != "" {
		runtimeLabels["driver"] = device.Driver
	}
	if device.MemoryBytes > 0 {
		runtimeLabels["memory_bytes"] = core.Sprintf("%d", device.MemoryBytes)
	}
	if len(runtimeLabels) == 0 {
		runtimeLabels = nil
	}
	generateCapability := inference.PlannedCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel, "native decode kernels are not linked yet")
	chatCapability := inference.PlannedCapability(inference.CapabilityChat, inference.CapabilityGroupModel, "native decode kernels are not linked yet")
	batchCapability := inference.PlannedCapability(inference.CapabilityBatchGenerate, inference.CapabilityGroupModel, "native decode kernels are not linked yet")
	if kernelStatus.Decode == hipKernelStatusLinked {
		generateCapability = inference.ExperimentalCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel, "native decode kernel is linked; ROCm generation remains experimental")
		chatCapability = inference.ExperimentalCapability(inference.CapabilityChat, inference.CapabilityGroupModel, "native decode kernel is linked; ROCm chat remains experimental")
		batchCapability = inference.ExperimentalCapability(inference.CapabilityBatchGenerate, inference.CapabilityGroupModel, "native decode kernel is linked; ROCm batch generation remains experimental")
		decodeLabels := rocmDecodeCapabilityLabels(kernelStatus, model)
		generateCapability.Labels = cloneStringMap(decodeLabels)
		chatCapability.Labels = cloneStringMap(decodeLabels)
		batchCapability.Labels = cloneStringMap(decodeLabels)
	} else if option.Gemma4Q4GenerateLinked {
		generateCapability = inference.ExperimentalCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel, "loaded Gemma4 MLX-q4 token/text prompt generation is linked through the experimental q4 path; production native prefill/decode remain pending")
		generateCapability.Labels = rocmGemma4Q4GenerateCapabilityLabels(model)
		chatCapability = inference.ExperimentalCapability(inference.CapabilityChat, inference.CapabilityGroupModel, "loaded Gemma4 MLX-q4 chat generation is linked through the Gemma4 chat template and the experimental q4 text prompt path; production native prefill/decode remain pending")
		chatCapability.Labels = rocmGemma4Q4ChatCapabilityLabels(model)
		batchCapability = inference.ExperimentalCapability(inference.CapabilityBatchGenerate, inference.CapabilityGroupModel, "loaded Gemma4 MLX-q4 batch generation is linked through the experimental q4 path; production native prefill/decode remain pending")
		batchCapability.Labels = rocmGemma4Q4BatchGenerateCapabilityLabels(model)
	}
	classifyCapability := inference.PlannedCapability(inference.CapabilityClassify, inference.CapabilityGroupModel, "native prefill kernels are not linked yet")
	classifyLinked := kernelStatus.Prefill == hipKernelStatusLinked || option.ClassifyLinked
	classifyLabels := rocmClassifyCapabilityLabels(kernelStatus, model, option)
	if classifyLinked {
		classifyCapability = inference.ExperimentalCapability(inference.CapabilityClassify, inference.CapabilityGroupModel, "native prefill kernel is linked; ROCm classification remains experimental")
		if option.ClassifyLinked && kernelStatus.Prefill != hipKernelStatusLinked {
			classifyCapability.Detail = "loaded BERT sequence-classifier path is linked through embedding mean-pool plus projection; ROCm classification remains experimental"
		}
		classifyCapability.Labels = classifyLabels
	}
	if option.Gemma4Q4GenerateLinked {
		classifyCapability = inference.ExperimentalCapability(inference.CapabilityClassify, inference.CapabilityGroupModel, "loaded Gemma4 MLX-q4 classification is linked through the experimental q4 package Prefill path; production native prefill remains pending")
		classifyCapability.Labels = rocmGemma4Q4ClassifyCapabilityLabels(model)
	}
	logitProbeCapability := inference.PlannedCapability(inference.CapabilityLogitProbe, inference.CapabilityGroupProbe, "logit probes need native prefill kernels first")
	if classifyLinked {
		logitProbeCapability = inference.ExperimentalCapability(inference.CapabilityLogitProbe, inference.CapabilityGroupProbe, "classification logits can emit compact logit and entropy probe summaries")
		logitProbeCapability.Labels = classifyLabels
	}
	if option.Gemma4Q4GenerateLinked {
		logitProbeCapability = inference.ExperimentalCapability(inference.CapabilityLogitProbe, inference.CapabilityGroupProbe, "loaded Gemma4 MLX-q4 classification logits can emit compact logit and entropy probe summaries through the experimental q4 package Prefill path")
		logitProbeCapability.Labels = rocmGemma4Q4LogitProbeCapabilityLabels(model)
	}
	benchmarkCapability := inference.ExperimentalCapability(inference.CapabilityBenchmark, inference.CapabilityGroupRuntime, "benchmark wrapper is available; native decode kernels are not linked yet")
	if kernelStatus.Decode == hipKernelStatusLinked {
		benchmarkCapability = inference.ExperimentalCapability(inference.CapabilityBenchmark, inference.CapabilityGroupRuntime, "benchmark wrapper can exercise the experimental linked ROCm decode path")
		benchmarkCapability.Labels = rocmDecodeCapabilityLabels(kernelStatus, model)
	}
	if option.Gemma4Q4GenerateLinked {
		benchmarkCapability = inference.ExperimentalCapability(inference.CapabilityBenchmark, inference.CapabilityGroupRuntime, "benchmark wrapper can exercise the experimental Gemma4 q4 generation path with explicit text prompt normalization; production native prefill/decode remain pending")
		benchmarkCapability.Labels = rocmGemma4Q4BenchmarkCapabilityLabels(model)
	}
	evaluationCapability := inference.ExperimentalCapability(inference.CapabilityEvaluation, inference.CapabilityGroupRuntime, "token-count eval is available before prefill kernels are linked")
	evaluationCapability.Labels = rocmEvaluationCapabilityLabels(kernelStatus, nil)
	if classifyLinked {
		evaluationCapability = inference.ExperimentalCapability(inference.CapabilityEvaluation, inference.CapabilityGroupRuntime, "eval can exercise the experimental linked ROCm prefill/classification path")
		evaluationCapability.Labels = rocmEvaluationCapabilityLabels(kernelStatus, classifyLabels)
	}
	if kernelStatus.CrossEntropy == hipKernelStatusLinked {
		evaluationCapability = inference.ExperimentalCapability(inference.CapabilityEvaluation, inference.CapabilityGroupRuntime, "eval can use the linked HIP cross-entropy/perplexity loss fixture")
		if classifyLinked {
			evaluationCapability.Detail = "eval can exercise the experimental linked ROCm prefill/classification path and linked HIP cross-entropy/perplexity loss fixture"
		}
		evaluationCapability.Labels = rocmEvaluationCapabilityLabels(kernelStatus, classifyLabels)
	}
	if option.Gemma4Q4GenerateLinked {
		detail := "eval can use experimental Gemma4 q4 package Prefill logits for loss/perplexity; production native prefill/decode remain pending"
		if kernelStatus.CrossEntropy == hipKernelStatusLinked {
			detail = "eval can use experimental Gemma4 q4 package Prefill logits with the linked HIP cross-entropy/perplexity loss fixture; production native prefill/decode remain pending"
		}
		evaluationCapability = inference.ExperimentalCapability(inference.CapabilityEvaluation, inference.CapabilityGroupRuntime, detail)
		evaluationCapability.Labels = rocmEvaluationCapabilityLabels(kernelStatus, classifyLabels)
		rocmAddReportLabels(evaluationCapability.Labels, rocmGemma4Q4EvaluationCapabilityLabels(model))
	}
	loraCapability := inference.PlannedCapability(inference.CapabilityLoRAInference, inference.CapabilityGroupModel, "native LoRA application is not linked yet")
	if model.Architecture != "" && kernelStatus.LoRA == hipKernelStatusLinked {
		loraCapability = inference.ExperimentalCapability(inference.CapabilityLoRAInference, inference.CapabilityGroupModel, "native LoRA projection kernel is linked for loaded tiny, Qwen/Gemma small LM-head, and BERT classifier adapters; production adapter application remains experimental")
		loraCapability.Labels = map[string]string{
			"kernel_name":                    hipKernelNameLoRA,
			"kernel_scope":                   "loaded_adapter_fixtures",
			"lora_kernel":                    kernelStatus.LoRA,
			"production_adapter_application": hipKernelStatusNotLinked,
			"runtime_status":                 string(inference.FeatureRuntimeExperimental),
			"supported_adapter_scopes":       "tiny_output_head,qwen_gemma_small_lm_head,bert_sequence_classifier",
		}
	}
	embeddingCapability := inference.PlannedCapability(inference.CapabilityEmbeddings, inference.CapabilityGroupModel, "embedding contract is available; native ROCm embedding kernels are pending")
	if kernelStatus.Embedding == hipKernelStatusLinked {
		embeddingCapability = inference.ExperimentalCapability(inference.CapabilityEmbeddings, inference.CapabilityGroupModel, "native embedding mean-pool kernel is linked for loaded f32 token/word embedding tables including BERT-style embedding-only packs; production embedding models remain experimental")
		embeddingCapability.Labels = map[string]string{
			"embedding_kernel":            kernelStatus.Embedding,
			"embedding_kernel_name":       hipKernelNameEmbedMean,
			"kernel_name":                 hipKernelNameEmbedMean,
			"kernel_scope":                "loaded_embedding_fixtures",
			"production_embedding_models": hipKernelStatusNotLinked,
			"runtime_status":              string(inference.FeatureRuntimeExperimental),
			"supported_embedding_scopes":  "tiny_token_embeddings,bert_word_embeddings",
		}
	}
	rerankCapability := inference.PlannedCapability(inference.CapabilityRerank, inference.CapabilityGroupModel, "rerank contract is available; native ROCm scorer is pending")
	if kernelStatus.Rerank == hipKernelStatusLinked {
		rerankCapability = inference.ExperimentalCapability(inference.CapabilityRerank, inference.CapabilityGroupModel, "native rerank cosine kernel is linked over loaded f32 embedding-table mean-pool vectors; production cross-encoder/scorer models remain experimental")
		rerankCapability.Labels = map[string]string{
			"kernel_name":              hipKernelNameRerank,
			"kernel_scope":             "loaded_rerank_fixtures",
			"production_rerank_models": hipKernelStatusNotLinked,
			"rerank_kernel":            kernelStatus.Rerank,
			"rerank_kernel_name":       hipKernelNameRerank,
			"runtime_status":           string(inference.FeatureRuntimeExperimental),
			"supported_rerank_scopes":  "embedding_cosine,bert_sequence_classifier",
		}
		if kernelStatus.Embedding != "" {
			rerankCapability.Labels["embedding_kernel"] = kernelStatus.Embedding
			rerankCapability.Labels["embedding_kernel_name"] = hipKernelNameEmbedMean
		}
	}
	speculativeCapability := inference.PlannedCapability(inference.CapabilitySpeculativeDecode, inference.CapabilityGroupModel, "speculative decode needs native decode kernels first")
	promptLookupCapability := inference.PlannedCapability(inference.CapabilityPromptLookupDecode, inference.CapabilityGroupModel, "prompt lookup decode needs native prefill/decode kernels first")
	if kernelStatus.Decode == hipKernelStatusLinked {
		speculativeCapability = inference.ExperimentalCapability(inference.CapabilitySpeculativeDecode, inference.CapabilityGroupModel, "shared speculative decode helper is available over the experimental ROCm generation path")
		speculativeCapability.Labels = rocmDecodeCapabilityLabels(kernelStatus, model)
		promptLookupCapability = inference.ExperimentalCapability(inference.CapabilityPromptLookupDecode, inference.CapabilityGroupModel, "shared prompt-lookup decode helper is available over the experimental ROCm generation path")
		promptLookupCapability.Labels = rocmDecodeCapabilityLabels(kernelStatus, model)
	}
	if option.Gemma4Q4GenerateLinked {
		speculativeCapability = inference.ExperimentalCapability(inference.CapabilitySpeculativeDecode, inference.CapabilityGroupModel, "shared speculative decode helper is available over the experimental Gemma4 q4 generation path; production native prefill/decode remain pending")
		speculativeCapability.Labels = rocmGemma4Q4SpeculativeDecodeCapabilityLabels(model)
		promptLookupCapability = inference.ExperimentalCapability(inference.CapabilityPromptLookupDecode, inference.CapabilityGroupModel, "shared prompt-lookup decode helper is available over the experimental Gemma4 q4 generation path; production native prefill/decode remain pending")
		promptLookupCapability.Labels = rocmGemma4Q4PromptLookupDecodeCapabilityLabels(model)
	}
	kvSnapshotCapability := rocmCacheRuntimeCapability(
		inference.CapabilityKVSnapshot,
		"runtime-owned package-local KV snapshots, HIP device-mirror snapshot serialization, loaded-model state wake remirror, and block-cache warm/disk-restore remirror are available; fully HIP-owned restore remains pending",
	)
	promptCacheCapability := rocmCacheRuntimeCapability(
		inference.CapabilityPromptCache,
		"metadata/package-local prompt cache warm, hit accounting, state refs, cold disk-ref rehydrate, and best-effort HIP device remirror are available; native prefill reuse remains pending",
	)
	cacheBlocksCapability := rocmCacheRuntimeCapability(
		inference.CapabilityCacheBlocks,
		"metadata-first in-memory block cache is available with package-local KV pages and optional HIP device remirror; native KV ownership is pending",
	)
	cacheDiskCapability := rocmCacheRuntimeCapability(
		inference.CapabilityCacheDisk,
		"go-inference/state disk refs are available for metadata cache refs and portable package-local KV snapshots, including exact cold rehydrate and best-effort HIP device remirror; fully HIP-owned disk KV remains pending",
	)
	cacheWarmCapability := rocmCacheRuntimeCapability(
		inference.CapabilityCacheWarm,
		"cache warm accounting is available before native prefill kernels, with planner-shaped package-local KV pages and optional HIP device remirror",
	)
	return inference.CapabilityReport{
		Runtime: inference.RuntimeIdentity{
			Backend:       "rocm",
			Device:        device.Name,
			Version:       device.Driver,
			NativeRuntime: true,
			Labels:        runtimeLabels,
		},
		Model:         cloneModelIdentity(model),
		Adapter:       cloneAdapterIdentity(adapter),
		Available:     available,
		Architectures: append([]string(nil), rocmCapabilityArchitectures...),
		Quantizations: append([]string(nil), rocmCapabilityQuantizations...),
		CacheModes:    append([]string(nil), rocmCapabilityCacheModes...),
		Capabilities: []inference.Capability{
			inference.SupportedCapability(inference.CapabilityModelLoad, inference.CapabilityGroupRuntime),
			inference.SupportedCapability(inference.CapabilityModelFit, inference.CapabilityGroupRuntime),
			inference.SupportedCapability(inference.CapabilityMemoryPlanning, inference.CapabilityGroupRuntime),
			inference.SupportedCapability(inference.CapabilityKVCachePlanning, inference.CapabilityGroupRuntime),
			benchmarkCapability,
			evaluationCapability,
			inference.PlannedCapability(inference.CapabilityQuantization, inference.CapabilityGroupRuntime, "GGUF quantisation is owned by shared model-pack tooling for now"),
			inference.PlannedCapability(inference.CapabilityModelMerge, inference.CapabilityGroupRuntime, "model-pack merge is not implemented in the ROCm package yet"),
			generateCapability,
			chatCapability,
			classifyCapability,
			batchCapability,
			inference.ExperimentalCapability(inference.CapabilityTokenizer, inference.CapabilityGroupModel, "Hugging Face tokenizer sidecar encode/decode is wired for loaded safetensors packs; GGUF/native templates remain limited"),
			inference.ExperimentalCapability(inference.CapabilityChatTemplate, inference.CapabilityGroupModel, "fallback chat template until model-native templates are wired"),
			loraCapability,
			inference.ExperimentalCapability(inference.CapabilityStateBundle, inference.CapabilityGroupRuntime, "metadata-only StateBundle capture/restore is available; durable KV payloads remain URI-first through AgentMemorySession wake/sleep"),
			kvSnapshotCapability,
			promptCacheCapability,
			rocmPlannedTrainingCapability(inference.CapabilityLoRATraining, "native ROCm LoRA backward/update kernels are not linked yet", "lora_backward", kernelStatus),
			rocmPlannedTrainingCapability(inference.CapabilityDistillation, "distillation needs teacher/student forward and loss kernels first", "distillation_forward_loss", kernelStatus),
			rocmPlannedTrainingCapability(inference.CapabilityGRPO, "GRPO needs rollout generation and policy-gradient kernels first", "grpo_rollout_policy", kernelStatus),
			inference.ExperimentalCapability(inference.CapabilityProbeEvents, inference.CapabilityGroupProbe, "probe sink is wired around streams; kernel-level probes are pending"),
			inference.PlannedCapability(inference.CapabilityAttentionProbe, inference.CapabilityGroupProbe, "attention probes need native prefill kernels first"),
			logitProbeCapability,
			inference.ExperimentalCapability(inference.CapabilityResponsesAPI, inference.CapabilityGroupRuntime, "non-streaming OpenAI Responses handler and service mux are available; streaming is pending"),
			inference.ExperimentalCapability(inference.CapabilityAnthropicMessages, inference.CapabilityGroupRuntime, "non-streaming Anthropic Messages handler is available; streaming is pending"),
			inference.ExperimentalCapability(inference.CapabilityOllamaCompat, inference.CapabilityGroupRuntime, "non-streaming Ollama chat/generate handlers are available; streaming and model registry endpoints are pending"),
			embeddingCapability,
			rerankCapability,
			inference.SupportedCapability(inference.CapabilityScheduler, inference.CapabilityGroupRuntime),
			inference.SupportedCapability(inference.CapabilityRequestCancel, inference.CapabilityGroupRuntime),
			cacheBlocksCapability,
			cacheDiskCapability,
			cacheWarmCapability,
			inference.SupportedCapability(inference.CapabilityToolParse, inference.CapabilityGroupModel),
			inference.SupportedCapability(inference.CapabilityReasoningParse, inference.CapabilityGroupModel),
			speculativeCapability,
			promptLookupCapability,
			rocmMetadataOnlyCapability(inference.CapabilityMoERouting, inference.CapabilityGroupModel, "MoE architecture metadata is recognised; native router kernels are pending"),
			rocmMetadataOnlyCapability(inference.CapabilityMoELazyExperts, inference.CapabilityGroupRuntime, "MoE lazy expert residency is planned from metadata; native expert paging is pending"),
			rocmMetadataOnlyCapability(inference.CapabilityJANGTQ, inference.CapabilityGroupRuntime, "JANG/JANGTQ metadata can be shared; native ROCm packed kernels are pending"),
			rocmMetadataOnlyCapability(inference.CapabilityCodebookVQ, inference.CapabilityGroupRuntime, "codebook/VQ metadata can be shared; native ROCm VQ kernels are pending"),
			inference.ExperimentalCapability(inference.CapabilityAgentMemory, inference.CapabilityGroupRuntime, "URI-first go-inference/state refs and package-local KV restore are wired; loaded ROCm models best-effort remirror portable KV refs to HIP device pages while fully HIP-owned kernel restore remains pending"),
			inference.ExperimentalCapability(inference.CapabilityStateWake, inference.CapabilityGroupRuntime, "state wake restores portable KV snapshot refs into package-local pages and loaded ROCm models can best-effort remirror them to HIP device pages"),
			inference.ExperimentalCapability(inference.CapabilityStateSleep, inference.CapabilityGroupRuntime, "state sleep serializes runtime-owned package-local and HIP device-mirror KV snapshots into portable refs"),
			inference.ExperimentalCapability(inference.CapabilityStateFork, inference.CapabilityGroupRuntime, "state fork wakes refs into a fresh session and loaded ROCm models can best-effort remirror forked KV refs to HIP device pages; production HIP KV page ownership is pending"),
		},
		Labels: labels,
	}
}

func rocmCacheRuntimeCapability(id inference.CapabilityID, detail string) inference.Capability {
	capability := inference.ExperimentalCapability(id, inference.CapabilityGroupRuntime, detail)
	capability.Labels = map[string]string{
		"disk_cache_restore":   "exact_cold_ref",
		"fully_hip_owned":      "pending",
		"kv_backing":           "package_local",
		"kv_cache_snapshot":    "portable",
		"kv_device_backing":    "best_effort_remirror",
		"native_prefill_reuse": "pending",
		"runtime_status":       string(inference.FeatureRuntimeExperimental),
	}
	return capability
}

func rocmMetadataOnlyCapability(id inference.CapabilityID, group inference.CapabilityGroup, detail string) inference.Capability {
	capability := inference.ExperimentalCapability(id, group, detail)
	capability.Labels = map[string]string{
		"kernel_status":          hipKernelStatusPlanned,
		"metadata_status":        "recognised",
		"production_integration": "pending",
		"runtime_status":         string(inference.FeatureRuntimeMetadataOnly),
	}
	if fixture, required := rocmMetadataOnlyFixtureKernel(id); fixture != "" {
		capability.Labels["fixture_kernel_name"] = fixture
		capability.Labels["required_integration"] = required
	}
	return capability
}

func rocmMetadataOnlyFixtureKernel(id inference.CapabilityID) (string, string) {
	switch id {
	case inference.CapabilityMoERouting:
		return hipKernelNameMoERouter, "model_router_forward"
	case inference.CapabilityMoELazyExperts:
		return hipKernelNameMoELazy, "expert_paging"
	case inference.CapabilityJANGTQ:
		return hipKernelNameJANGTQ, "packed_weight_model_integration"
	case inference.CapabilityCodebookVQ:
		return hipKernelNameCodebook, "codebook_weight_model_integration"
	default:
		return "", ""
	}
}

func rocmDecodeCapabilityLabels(kernelStatus hipKernelStatus, model inference.ModelIdentity) map[string]string {
	labels := map[string]string{
		"decode_kernel":      kernelStatus.Decode,
		"decode_kernel_name": hipKernelNameDecode,
		"kernel_scope":       "native_decode",
		"runtime_status":     string(inference.FeatureRuntimeExperimental),
	}
	if kernelStatus.Prefill != "" {
		labels["prefill_kernel"] = kernelStatus.Prefill
		labels["prefill_kernel_name"] = hipKernelNamePrefill
	}
	if normalizeROCmArchitecture(model.Architecture) == "tiny" {
		labels["decode_kernel_name"] = hipKernelNameTinyDecode
		labels["prefill_kernel_name"] = hipKernelNameTinyPrefill
		labels["kernel_scope"] = "toy_tiny_fixture"
		labels["production_decode"] = hipKernelStatusNotLinked
		labels["production_prefill"] = hipKernelStatusNotLinked
	}
	return labels
}

func rocmGemma4Q4GenerateCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := map[string]string{
		"attention_kv_backing":        "hip_device_descriptor",
		"attention_kv_mode":           rocmKVCacheModeKQ8VQ4,
		"decode_architecture":         "gemma4",
		"decode_quant":                "mlx_q4",
		"gemma4_q4_device_kv_state":   "forward_returned_device_state",
		"gemma4_q4_decode_kernel":     hipKernelStatusLinked,
		"gemma4_q4_decode_name":       "rocm_gemma4_q4_greedy_decode_smoke",
		"kernel_scope":                "loaded_gemma4_q4_experimental_generate",
		"production_decode":           hipKernelStatusNotLinked,
		"production_kv_cache_backing": hipKernelStatusNotLinked,
		"production_prefill":          hipKernelStatusNotLinked,
		"prompt_modes":                "tokens,text,env_plain_text",
		"runtime_status":              string(inference.FeatureRuntimeExperimental),
	}
	if model.NumLayers > 0 {
		labels["decode_layers"] = core.Sprintf("%d", model.NumLayers)
	}
	if model.VocabSize > 0 {
		labels["decode_vocab_size"] = core.Sprintf("%d", model.VocabSize)
	}
	if model.HiddenSize > 0 {
		labels["decode_hidden_size"] = core.Sprintf("%d", model.HiddenSize)
	}
	return labels
}

func rocmGemma4Q4BatchGenerateCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4GenerateCapabilityLabels(model)
	labels["batch_generate_kernel"] = hipKernelStatusLinked
	labels["batch_generate_name"] = "rocm_gemma4_q4_batch_generate_experimental"
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_batch_generate"
	return labels
}

func rocmGemma4Q4ChatCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4GenerateCapabilityLabels(model)
	labels["chat_kernel"] = hipKernelStatusLinked
	labels["chat_name"] = "rocm_gemma4_q4_chat_generate_experimental"
	labels["chat_template"] = "fallback"
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_chat"
	return labels
}

func rocmGemma4Q4EvaluationCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4GenerateCapabilityLabels(model)
	labels["eval_loss_logits_source"] = "gemma4_q4_package_prefill"
	labels["eval_prefill_kernel"] = hipKernelStatusLinked
	labels["eval_prefill_name"] = "rocm_gemma4_q4_package_prefill_experimental"
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_eval"
	labels["production_prefill"] = hipKernelStatusNotLinked
	return labels
}

func rocmGemma4Q4BenchmarkCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4GenerateCapabilityLabels(model)
	labels["benchmark_kernel"] = hipKernelStatusLinked
	labels["benchmark_name"] = "rocm_gemma4_q4_benchmark_experimental"
	labels["benchmark_prompt_mode"] = "explicit_text"
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_benchmark"
	return labels
}

func rocmGemma4Q4ClassifyCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4GenerateCapabilityLabels(model)
	labels["classify_kernel"] = hipKernelStatusLinked
	labels["classify_name"] = "rocm_gemma4_q4_classify_experimental"
	labels["classify_logits_source"] = "gemma4_q4_package_prefill"
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_classify"
	labels["production_prefill"] = hipKernelStatusNotLinked
	return labels
}

func rocmGemma4Q4LogitProbeCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4ClassifyCapabilityLabels(model)
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_logit_probe"
	labels["logit_probe_kernel"] = hipKernelStatusLinked
	labels["logit_probe_source"] = "gemma4_q4_classify_logits"
	return labels
}

func rocmGemma4Q4SpeculativeDecodeCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4GenerateCapabilityLabels(model)
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_speculative_decode"
	labels["speculative_decode_helper"] = hipKernelStatusLinked
	labels["speculative_decode_source"] = "gemma4_q4_generate"
	return labels
}

func rocmGemma4Q4PromptLookupDecodeCapabilityLabels(model inference.ModelIdentity) map[string]string {
	labels := rocmGemma4Q4GenerateCapabilityLabels(model)
	labels["kernel_scope"] = "loaded_gemma4_q4_experimental_prompt_lookup_decode"
	labels["prompt_lookup_decode_helper"] = hipKernelStatusLinked
	labels["prompt_lookup_decode_source"] = "gemma4_q4_generate"
	return labels
}

func rocmAddReportLabels(labels map[string]string, extra map[string]string) {
	if labels == nil {
		return
	}
	for key, value := range extra {
		if value != "" {
			labels[key] = value
		}
	}
}

func rocmClassifyCapabilityLabels(kernelStatus hipKernelStatus, model inference.ModelIdentity, option rocmCapabilityReportOption) map[string]string {
	labels := map[string]string{
		"runtime_status": string(inference.FeatureRuntimeExperimental),
	}
	if kernelStatus.Prefill == hipKernelStatusLinked {
		labels["kernel_status"] = kernelStatus.Prefill
		labels["prefill_kernel"] = kernelStatus.Prefill
		labels["prefill_kernel_name"] = hipKernelNamePrefill
		labels["kernel_scope"] = "native_prefill"
		if normalizeROCmArchitecture(model.Architecture) == "tiny" {
			labels["prefill_kernel_name"] = hipKernelNameTinyPrefill
			labels["kernel_scope"] = "toy_tiny_fixture"
			labels["production_prefill"] = hipKernelStatusNotLinked
		}
		return labels
	}
	if option.ClassifyLinked {
		labels["classify_path"] = "bert_sequence_classifier"
		labels["embedding_kernel"] = kernelStatus.Embedding
		labels["projection_kernel"] = kernelStatus.Projection
	}
	return labels
}

func rocmEvaluationCapabilityLabels(kernelStatus hipKernelStatus, classifyLabels map[string]string) map[string]string {
	labels := map[string]string{
		"loss_kernel":      kernelStatus.CrossEntropy,
		"loss_kernel_name": hipKernelNameCrossEntropy,
		"loss_scope":       "toy_cross_entropy",
		"runtime_status":   string(inference.FeatureRuntimeExperimental),
	}
	for key, value := range classifyLabels {
		labels[key] = value
	}
	return labels
}

func rocmPlannedTrainingCapability(id inference.CapabilityID, detail, requiredKernel string, kernelStatus hipKernelStatus) inference.Capability {
	capability := inference.PlannedCapability(id, inference.CapabilityGroupTraining, detail)
	kernelStatus = normalizeHIPKernelStatus(kernelStatus)
	capability.Labels = map[string]string{
		"kernel_status":      hipKernelStatusPlanned,
		"required_kernel":    requiredKernel,
		"runtime_status":     string(inference.FeatureRuntimePlanned),
		"training_kernel":    hipKernelStatusNotLinked,
		"training_interface": "not_implemented",
	}
	switch id {
	case inference.CapabilityDistillation:
		capability.Labels["fixture_kernel"] = kernelStatus.Distillation
		capability.Labels["fixture_kernel_name"] = hipKernelNameDistillKL
		capability.Labels["fixture_scope"] = "toy_kl_loss"
	case inference.CapabilityGRPO:
		capability.Labels["fixture_kernel"] = kernelStatus.GRPO
		capability.Labels["fixture_kernel_name"] = hipKernelNameGRPOAdvantage
		capability.Labels["fixture_scope"] = "toy_advantage_normalization"
	}
	return capability
}

var (
	rocmCapabilityArchitectures = []string{
		"bert",
		"deepseek",
		"deepseek_r1",
		"gemma",
		"gemma2",
		"gemma3",
		"gemma4",
		"gemma4_text",
		"glm",
		"glm4",
		"gpt-oss",
		"granite",
		"hermes",
		"kimi",
		"llama",
		"minimax",
		"minimax_m2",
		"mistral",
		"mixtral",
		"phi",
		"phi3",
		"qwen2",
		"qwen3",
		"qwen3_moe",
		"qwen3_next",
	}
	rocmCapabilityQuantizations = []string{
		"codebook",
		"f16",
		"f32",
		"iq",
		"jang",
		"jangtq",
		"mxfp4",
		"mxtq",
		"nvfp4",
		"q2",
		"q3",
		"q4",
		"q4_k_m",
		"q5",
		"q5_k_m",
		"q6",
		"q8",
		"q8_0",
		"vq",
	}
	rocmCapabilityCacheModes = []string{
		"disk-l2",
		"fp16",
		"k-q8-v-q4",
		"paged",
		"q8",
	}
)

func resolveContextLength(requestedContextLength int, metadata gguf.Metadata) int {
	if requestedContextLength > 0 {
		return requestedContextLength
	}
	if metadata.ContextLength == 0 {
		return defaultContextLengthCap
	}
	return min(int(metadata.ContextLength), defaultContextLengthCap)
}

func resolveModelContextLength(requestedContextLength, modelContextLength int) int {
	if requestedContextLength > 0 {
		return requestedContextLength
	}
	if modelContextLength <= 0 {
		return defaultContextLengthCap
	}
	return min(modelContextLength, defaultContextLengthCap)
}

func modelInfoFromMetadata(metadata gguf.Metadata) inference.ModelInfo {
	quantBits, quantGroup := quantisationFromFileType(metadata.FileType)
	return inference.ModelInfo{Architecture: normalizeROCmArchitecture(metadata.Architecture), NumLayers: int(metadata.BlockCount), QuantBits: quantBits, QuantGroup: quantGroup}
}

func modelInfoFromIdentity(model inference.ModelIdentity) inference.ModelInfo {
	return inference.ModelInfo{
		Architecture: normalizeROCmArchitecture(model.Architecture),
		VocabSize:    model.VocabSize,
		NumLayers:    model.NumLayers,
		HiddenSize:   model.HiddenSize,
		QuantBits:    model.QuantBits,
		QuantGroup:   model.QuantGroup,
	}
}

func nativeTensorInfos(tensors []gguf.TensorInfo) []nativeTensorInfo {
	out := make([]nativeTensorInfo, len(tensors))
	for i, tensor := range tensors {
		out[i] = nativeTensorInfo{
			Name:       tensor.Name,
			Dimensions: append([]uint64(nil), tensor.Dimensions...),
			Type:       tensor.Type,
			TypeName:   tensor.TypeName,
			Offset:     tensor.Offset,
			ByteSize:   tensor.ByteSize,
		}
	}
	return out
}

func quantisationFromFileType(fileType uint32) (bits, groupSize int) {
	fileTypeName := gguf.FileTypeName(fileType)
	switch {
	case core.HasPrefix(fileTypeName, "Q4_"):
		return 4, 32
	case core.HasPrefix(fileTypeName, "Q5_"):
		return 5, 32
	case core.HasPrefix(fileTypeName, "Q8_"):
		return 8, 32
	case core.HasPrefix(fileTypeName, "Q2_"):
		return 2, 16
	case core.HasPrefix(fileTypeName, "Q3_"):
		return 3, 32
	case core.HasPrefix(fileTypeName, "Q6_"):
		return 6, 64
	case fileTypeName == "F16":
		return 16, 0
	case fileTypeName == "F32":
		return 32, 0
	default:
		return 0, 0
	}
}

func normalizeROCmArchitecture(architecture string) string {
	normalized := core.Lower(architecture)
	normalized = core.Replace(normalized, "-", "_")
	normalized = core.Replace(normalized, ".", "_")
	normalized = core.Replace(normalized, " ", "_")
	switch {
	case normalized == "":
		return ""
	case core.Contains(normalized, "minimax") && core.Contains(normalized, "m2"):
		return "minimax_m2"
	case core.Contains(normalized, "minimax"):
		return "minimax"
	case core.Contains(normalized, "qwen3") && core.Contains(normalized, "moe"):
		return "qwen3_moe"
	case core.Contains(normalized, "qwen3") && core.Contains(normalized, "next"):
		return "qwen3_next"
	case core.Contains(normalized, "qwen3"):
		return "qwen3"
	case core.Contains(normalized, "qwen2"):
		return "qwen2"
	case core.Contains(normalized, "deepseek"):
		if core.Contains(normalized, "r1") {
			return "deepseek_r1"
		}
		return "deepseek"
	case core.Contains(normalized, "gpt_oss") || core.Contains(normalized, "gptoss"):
		return "gpt-oss"
	case core.Contains(normalized, "gemma4"):
		if core.Contains(normalized, "text") || core.Contains(normalized, "forcausallm") {
			return "gemma4_text"
		}
		return "gemma4"
	case core.Contains(normalized, "gemma3"):
		return "gemma3"
	case core.Contains(normalized, "gemma2"):
		return "gemma2"
	case core.Contains(normalized, "gemma"):
		return "gemma"
	case core.Contains(normalized, "mixtral"):
		return "mixtral"
	case core.Contains(normalized, "mistral"):
		return "mistral"
	case core.Contains(normalized, "phi3"):
		return "phi3"
	case core.Contains(normalized, "phi"):
		return "phi"
	case core.Contains(normalized, "bert"):
		return "bert"
	case core.Contains(normalized, "glm4"):
		return "glm4"
	case core.Contains(normalized, "glm"):
		return "glm"
	case core.Contains(normalized, "kimi"):
		return "kimi"
	case core.Contains(normalized, "llama"):
		return "llama"
	case core.Contains(normalized, "hermes"):
		return "hermes"
	case core.Contains(normalized, "granite"):
		return "granite"
	default:
		return normalized
	}
}

func isROCmGemma4Architecture(architecture string) bool {
	architecture = normalizeROCmArchitecture(architecture)
	return architecture == "gemma4" || architecture == "gemma4_text"
}

func supportedNativeArchitecture(architecture string) bool {
	architecture = normalizeROCmArchitecture(architecture)
	if architecture == "" {
		return true
	}
	supported := map[string]struct{}{
		"bert": {}, "deepseek": {}, "deepseek_r1": {}, "gemma": {}, "gemma2": {}, "gemma3": {}, "gemma4": {}, "gemma4_text": {},
		"glm": {}, "glm4": {}, "gpt-oss": {}, "granite": {}, "hermes": {}, "kimi": {}, "llama": {},
		"minimax": {}, "minimax_m2": {}, "mistral": {}, "mixtral": {}, "phi": {}, "phi3": {},
		"qwen2": {}, "qwen3": {}, "qwen3_moe": {}, "qwen3_next": {},
	}
	_, ok := supported[architecture]
	return ok
}

func supportedNativeQuantization(bits int, quantType string) bool {
	if bits == 0 && quantType == "" {
		return true
	}
	if bits > 0 && bits <= 8 {
		return true
	}
	quantType = core.Lower(quantType)
	if quantType == "f16" || quantType == "f32" || quantType == "bf16" {
		return true
	}
	return core.Contains(quantType, "q4") || core.Contains(quantType, "q5") || core.Contains(quantType, "q8") || isROCmMetadataQuantization(quantType)
}

func isROCmMetadataQuantization(quantType string) bool {
	quantType = core.Lower(quantType)
	return core.Contains(quantType, "jang") || core.Contains(quantType, "mxtq") || core.Contains(quantType, "codebook") || core.Contains(quantType, "vq") || core.Contains(quantType, "iq") || core.Contains(quantType, "mxfp4") || core.Contains(quantType, "nvfp4")
}

func isROCmMoEArchitecture(architecture string) bool {
	architecture = normalizeROCmArchitecture(architecture)
	return core.Contains(architecture, "moe") || architecture == "mixtral" || architecture == "minimax_m2"
}

func rocmRecommendedCacheMode(memoryBytes uint64, contextLength int, model inference.ModelIdentity) string {
	if memoryBytes <= 16*memoryGiB && (contextLength > 8192 || isROCmMoEArchitecture(model.Architecture) || isROCmMetadataQuantization(model.QuantType)) {
		return "k-q8-v-q4"
	}
	if memoryBytes <= 24*memoryGiB || contextLength > 8192 {
		return "q8"
	}
	return "fp16"
}

func estimateKVCacheBytes(layers, contextLength, hidden int, cacheMode string, model inference.ModelIdentity) uint64 {
	base := estimateKVCacheElementSpan(layers, contextLength, hidden, model)
	switch cacheMode {
	case "q8", "paged":
		return base * 2
	case "k-q8-v-q4":
		return (base*3 + 1) / 2
	default:
		return base * 4
	}
}

func estimateKVCacheElementSpan(layers, contextLength, hidden int, model inference.ModelIdentity) uint64 {
	if layers <= 0 || contextLength <= 0 || hidden <= 0 {
		return 0
	}
	fullLayers := rocmModelLabelInt(model.Labels, "attention_full_layers")
	slidingLayers := rocmModelLabelInt(model.Labels, "attention_sliding_layers")
	slidingWindow := rocmModelLabelInt(model.Labels, "sliding_window")
	if slidingLayers <= 0 || slidingWindow <= 0 {
		return uint64(layers) * uint64(contextLength) * uint64(hidden)
	}
	if fullLayers < 0 {
		fullLayers = 0
	}
	if fullLayers+slidingLayers > layers {
		overflow := fullLayers + slidingLayers - layers
		if slidingLayers >= overflow {
			slidingLayers -= overflow
		} else {
			fullLayers -= overflow - slidingLayers
			slidingLayers = 0
		}
	}
	remainingLayers := layers - fullLayers - slidingLayers
	if remainingLayers < 0 {
		remainingLayers = 0
	}
	slidingContext := min(contextLength, slidingWindow)
	effectiveLayerTokens := uint64(fullLayers+remainingLayers)*uint64(contextLength) + uint64(slidingLayers)*uint64(slidingContext)
	return effectiveLayerTokens * uint64(hidden)
}

func rocmEstimatedRuntimeBytes(kvBytes, weightBytes uint64) uint64 {
	if weightBytes > ^uint64(0)-kvBytes {
		return ^uint64(0)
	}
	return kvBytes + weightBytes
}

func rocmModelWeightBytes(model inference.ModelIdentity) uint64 {
	if model.Labels == nil {
		return 0
	}
	for _, key := range []string{"weight_bytes", "safetensors_index_total_size", "safetensors_payload_bytes"} {
		if value := rocmLabelUint(model.Labels[key]); value > 0 {
			return value
		}
	}
	return 0
}

func rocmModelLabelInt(labels map[string]string, key string) int {
	if labels == nil {
		return 0
	}
	value := rocmLabelUint(labels[key])
	if value > uint64(^uint(0)>>1) {
		return int(^uint(0) >> 1)
	}
	return int(value)
}

func rocmMemoryPlanLabels(memoryBytes uint64, contextLength, layers, hidden int, model inference.ModelIdentity, kvBytes, weightBytes, runtimeBytes uint64, cacheMode string) map[string]string {
	batch := rocmRecommendedBatchSize(memoryBytes)
	prefillChunk := 2048
	if memoryBytes <= 16*memoryGiB {
		prefillChunk = 512
	} else if memoryBytes <= 24*memoryGiB || contextLength > 8192 {
		prefillChunk = 1024
	}
	allocatorLimit := memoryBytes * 85 / 100
	cacheLimit := memoryBytes * 30 / 100
	kvWidth := layers * hidden
	labels := map[string]string{
		"allocator_limit_bytes":    core.Sprintf("%d", allocatorLimit),
		"cache_limit_bytes":        core.Sprintf("%d", cacheLimit),
		"disk_cache":               "planned",
		"estimated_runtime_bytes":  core.Sprintf("%d", runtimeBytes),
		"kv_cache_bytes":           core.Sprintf("%d", kvBytes),
		"kv_cache_block_size":      core.Sprintf("%d", defaultROCmKVBlockSize),
		"kv_key_width":             core.Sprintf("%d", kvWidth),
		"kv_value_width":           core.Sprintf("%d", kvWidth),
		"max_prefill_batch_tokens": core.Sprintf("%d", prefillChunk*batch),
		"paged_cache":              "planned",
		"prefill_chunk_tokens":     core.Sprintf("%d", prefillChunk),
		"prompt_lookup_decode":     "planned",
		"recommended_cache_mode":   cacheMode,
		"speculative_decode":       "planned",
	}
	if isROCmMoEArchitecture(model.Architecture) {
		labels["moe_lazy_experts"] = "true"
		labels["moe_max_resident_experts"] = "2"
		if memoryBytes >= 24*memoryGiB {
			labels["moe_max_resident_experts"] = "4"
		}
		labels["moe_router_top_k"] = "2"
	} else {
		labels["moe_lazy_experts"] = "false"
	}
	if isROCmMetadataQuantization(model.QuantType) {
		labels["metadata_quantization"] = model.QuantType
	}
	if weightBytes > 0 {
		labels["weight_bytes"] = core.Sprintf("%d", weightBytes)
	}
	for _, key := range []string{
		"sliding_window",
		"attention_full_layers",
		"attention_sliding_layers",
		"attention_heads",
		"attention_kv_heads",
		"attention_global_kv_heads",
		"attention_head_dim",
		"attention_global_head_dim",
		"attention_query_width",
		"attention_kv_width",
		"attention_global_kv_width",
		"attention_gqa",
		"rms_norm_eps",
		"final_logit_softcapping",
	} {
		if model.Labels != nil && model.Labels[key] != "" {
			labels[key] = model.Labels[key]
		}
	}
	return labels
}

func rocmMachineClass(memoryBytes uint64) string {
	switch {
	case rocmAtLeastMemoryClass(memoryBytes, 64*memoryGiB):
		return "rocm-64gb-plus"
	case rocmAtLeastMemoryClass(memoryBytes, 24*memoryGiB):
		return "rocm-24gb"
	case rocmAtLeastMemoryClass(memoryBytes, 16*memoryGiB):
		return "rocm-16gb"
	default:
		return "rocm-small"
	}
}

func rocmRecommendedBatchSize(memoryBytes uint64) int {
	if rocmAtLeastMemoryClass(memoryBytes, 48*memoryGiB) {
		return 8
	}
	if rocmAtLeastMemoryClass(memoryBytes, 24*memoryGiB) {
		return 4
	}
	return 1
}

func rocmAtLeastMemoryClass(memoryBytes, threshold uint64) bool {
	if memoryBytes >= threshold {
		return true
	}
	if threshold <= memoryClassToleranceBytes {
		return false
	}
	return memoryBytes >= threshold-memoryClassToleranceBytes
}

func rocmQuantizationLabel(model inference.ModelIdentity) string {
	if model.QuantType != "" {
		return model.QuantType
	}
	if model.QuantBits > 0 {
		return core.Sprintf("q%d", model.QuantBits)
	}
	return ""
}

func nativePeakMemoryBytes() uint64 {
	info, err := GetVRAMInfo()
	if err != nil {
		return 0
	}
	return info.Used
}

func tokensPerSecond(tokens int, duration time.Duration) float64 {
	if tokens <= 0 || duration <= 0 {
		return 0
	}
	return float64(tokens) / duration.Seconds()
}

func splitDurations(start, firstTokenAt, end time.Time) (time.Duration, time.Duration) {
	if start.IsZero() || end.Before(start) {
		return 0, 0
	}
	if firstTokenAt.IsZero() || firstTokenAt.Before(start) || firstTokenAt.After(end) {
		return end.Sub(start), 0
	}
	return firstTokenAt.Sub(start), end.Sub(firstTokenAt)
}

func approximatePromptTokens(prompt string) int { return len(approximateTokenIDs(prompt)) }

func approximatePromptsTokens(prompts []string) int {
	total := 0
	for _, prompt := range prompts {
		total += approximatePromptTokens(prompt)
	}
	return total
}

func approximateMessageTokens(messages []inference.Message) int {
	total := 0
	for _, message := range messages {
		total += approximatePromptTokens(message.Content)
	}
	return total
}

func approximateTokenIDs(text string) []int32 {
	trimmed := core.Trim(text)
	if trimmed == "" {
		return nil
	}
	parts := core.Split(trimmed, " ")
	ids := make([]int32, len(parts))
	for i := range parts {
		ids[i] = int32(i + 1)
	}
	return ids
}

func formatFallbackChatTemplate(messages []inference.Message) string {
	builder := core.NewBuilder()
	for _, message := range messages {
		builder.WriteString(message.Role)
		builder.WriteString(": ")
		builder.WriteString(message.Content)
		builder.WriteString("\n")
	}
	return builder.String()
}

func formatGemma4ChatTemplate(messages []inference.Message) string {
	builder := core.NewBuilder()
	builder.WriteString("<bos>")
	start := 0
	if len(messages) > 0 {
		role := core.Lower(core.Trim(messages[0].Role))
		if role == "system" || role == "developer" {
			builder.WriteString("<|turn>system\n")
			builder.WriteString(core.Trim(messages[0].Content))
			builder.WriteString("<turn|>\n")
			start = 1
		}
	}
	for _, message := range messages[start:] {
		role := core.Lower(core.Trim(message.Role))
		if role == "assistant" {
			role = "model"
		}
		if role == "" {
			role = "user"
		}
		builder.WriteString("<|turn>")
		builder.WriteString(role)
		builder.WriteByte('\n')
		content := core.Trim(message.Content)
		if role == "model" {
			content = stripGemma4ThinkingChannels(content)
		}
		builder.WriteString(content)
		builder.WriteString("<turn|>\n")
	}
	builder.WriteString("<|turn>model\n")
	return builder.String()
}

func stripGemma4ThinkingChannels(text string) string {
	if text == "" || !strings.Contains(text, "<channel|>") {
		return core.Trim(text)
	}
	var builder strings.Builder
	for _, part := range strings.Split(text, "<channel|>") {
		before, _, found := strings.Cut(part, "<|channel>")
		if found {
			builder.WriteString(before)
			continue
		}
		builder.WriteString(part)
	}
	return core.Trim(builder.String())
}

func sampleText(sample inference.DatasetSample) string {
	switch {
	case sample.Text != "":
		return sample.Text
	case sample.Prompt != "" || sample.Response != "":
		return core.Trim(sample.Prompt + " " + sample.Response)
	case len(sample.Messages) > 0:
		return formatFallbackChatTemplate(sample.Messages)
	default:
		return sample.Reasoning
	}
}

func emptyTokenSeq(func(inference.Token) bool) {}
