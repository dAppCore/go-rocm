// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"iter"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/rocm/internal/gguf"
)

const (
	defaultContextLengthCap = 4096
	memoryGiB               = uint64(1 << 30)
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
	ContextSize       int
	GPULayerCount     int
	ParallelSlotCount int
	AdapterPath       string
	ModelInfo         inference.ModelInfo
	DataOffset        int64
	Tensors           []nativeTensorInfo
}

type nativeTensorInfo struct {
	Name       string
	Dimensions []uint64
	Type       uint32
	TypeName   string
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

func (b *rocmBackend) LoadModel(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
	loadConfig := inference.ApplyLoadOpts(opts)
	modelPack, err := gguf.ReadInfo(path)
	if err != nil {
		return nil, core.E("rocm.LoadModel", "read model metadata", err)
	}
	metadata := modelPack.Metadata

	runtime := b.nativeRuntime()
	if !runtime.Available() {
		return nil, core.E("rocm.LoadModel", "native ROCm runtime is not available", nil)
	}

	contextLength := resolveContextLength(loadConfig.ContextLen, metadata)
	modelInfo := modelInfoFromMetadata(metadata)
	loaded, err := runtime.LoadModel(path, nativeLoadConfig{
		ContextSize:       contextLength,
		GPULayerCount:     loadConfig.GPULayers,
		ParallelSlotCount: loadConfig.ParallelSlots,
		AdapterPath:       loadConfig.AdapterPath,
		ModelInfo:         modelInfo,
		DataOffset:        modelPack.DataOffset,
		Tensors:           nativeTensorInfos(modelPack.Tensors),
	})
	if err != nil {
		return nil, core.E("rocm.LoadModel", "load native model", err)
	}

	model := &rocmModel{native: loaded, modelType: metadata.Architecture, modelInfo: modelInfo}
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

	cacheMode := "fp16"
	if memoryBytes <= 24*memoryGiB || contextLength > 8192 {
		cacheMode = "q8"
	}
	kvBytes := estimateKVCacheBytes(layers, contextLength, hidden, cacheMode)
	architectureOK := supportedNativeArchitecture(model.Architecture)
	quantizationOK := supportedNativeQuantization(model.QuantBits, model.QuantType)
	fits := architectureOK && quantizationOK && kvBytes < memoryBytes*7/10
	plan := inference.MemoryPlan{
		MachineClass:      rocmMachineClass(memoryBytes),
		DeviceMemoryBytes: memoryBytes,
		ContextLength:     contextLength,
		BatchSize:         rocmRecommendedBatchSize(memoryBytes),
		CacheMode:         cacheMode,
		Quantization:      rocmQuantizationLabel(model),
		KVCacheBytes:      kvBytes,
		TrainingFeasible:  memoryBytes >= 16*memoryGiB && model.QuantBits <= 8,
	}
	if !architectureOK {
		plan.Notes = append(plan.Notes, "architecture is not in the native ROCm allow-list yet")
	}
	if !quantizationOK {
		plan.Notes = append(plan.Notes, "quantisation is not expected to fit the native ROCm path")
	}
	if kvBytes >= memoryBytes*7/10 {
		plan.Notes = append(plan.Notes, "KV cache estimate leaves too little memory for weights and workspace")
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

type rocmModel struct {
	native    nativeModel
	modelType string
	modelInfo inference.ModelInfo

	stateMutex  sync.Mutex
	lastError   error
	lastMetrics inference.GenerateMetrics
	probeSink   inference.ProbeSink
	adapter     inference.AdapterIdentity
}

func (m *rocmModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.clearLastError()
	if m == nil || m.native == nil {
		if m != nil {
			m.setLastFailure(core.E("rocm.Generate", "native model is nil", nil))
		}
		return emptyTokenSeq
	}
	cfg := inference.ApplyGenerateOpts(opts)
	promptTokens := len(m.Encode(prompt))
	start := time.Now()
	stream, streamError := m.native.Generate(ctx, prompt, cfg)
	return m.wrapTokenStream(stream, streamError, promptTokens, start)
}

func (m *rocmModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.clearLastError()
	if m == nil || m.native == nil {
		if m != nil {
			m.setLastFailure(core.E("rocm.Chat", "native model is nil", nil))
		}
		return emptyTokenSeq
	}
	cfg := inference.ApplyGenerateOpts(opts)
	promptTokens := approximateMessageTokens(messages)
	start := time.Now()
	stream, streamError := m.native.Chat(ctx, messages, cfg)
	return m.wrapTokenStream(stream, streamError, promptTokens, start)
}

func (m *rocmModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	if m == nil || m.native == nil {
		return nil, core.E("rocm.Classify", "native model is nil", nil)
	}
	start := time.Now()
	results, err := m.native.Classify(ctx, prompts, inference.ApplyGenerateOpts(opts))
	m.recordMetrics(approximatePromptsTokens(prompts), len(results), start, time.Now())
	return results, err
}

func (m *rocmModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.BatchResult, error) {
	if m == nil || m.native == nil {
		return nil, core.E("rocm.BatchGenerate", "native model is nil", nil)
	}
	start := time.Now()
	results, err := m.native.BatchGenerate(ctx, prompts, inference.ApplyGenerateOpts(opts))
	generated := 0
	for _, result := range results {
		generated += len(result.Tokens)
	}
	m.recordMetrics(approximatePromptsTokens(prompts), generated, start, time.Now())
	return results, err
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

func (m *rocmModel) Close() error {
	if m == nil || m.native == nil {
		return nil
	}
	return m.native.Close()
}

func (m *rocmModel) Encode(text string) []int32 {
	if m == nil || m.native == nil {
		return approximateTokenIDs(text)
	}
	return m.native.Encode(text)
}

func (m *rocmModel) Decode(ids []int32) string {
	if m == nil || m.native == nil {
		return ""
	}
	return m.native.Decode(ids)
}

func (m *rocmModel) ApplyChatTemplate(messages []inference.Message) (string, error) {
	if m == nil || m.native == nil {
		return formatFallbackChatTemplate(messages), nil
	}
	return m.native.ApplyChatTemplate(messages)
}

func (m *rocmModel) LoadAdapter(path string) (inference.AdapterIdentity, error) {
	if m == nil || m.native == nil {
		return inference.AdapterIdentity{}, core.E("rocm.LoadAdapter", "native model is nil", nil)
	}
	identity, err := m.native.LoadAdapter(path)
	if err != nil {
		return inference.AdapterIdentity{}, err
	}
	if identity.Format == "" {
		identity.Format = "lora"
	}
	if identity.Path == "" {
		identity.Path = path
	}
	m.stateMutex.Lock()
	m.adapter = identity
	m.stateMutex.Unlock()
	return identity, nil
}

func (m *rocmModel) UnloadAdapter() error {
	if m == nil || m.native == nil {
		return core.E("rocm.UnloadAdapter", "native model is nil", nil)
	}
	if err := m.native.UnloadAdapter(); err != nil {
		return err
	}
	m.stateMutex.Lock()
	m.adapter = inference.AdapterIdentity{}
	m.stateMutex.Unlock()
	return nil
}

func (m *rocmModel) ActiveAdapter() inference.AdapterIdentity {
	if m == nil {
		return inference.AdapterIdentity{}
	}
	m.stateMutex.Lock()
	adapter := m.adapter
	m.stateMutex.Unlock()
	if !adapterIdentityIsZero(adapter) {
		return adapter
	}
	if m.native == nil {
		return inference.AdapterIdentity{}
	}
	return m.native.ActiveAdapter()
}

func adapterIdentityIsZero(identity inference.AdapterIdentity) bool {
	return identity.Path == "" && identity.Hash == "" && identity.Format == "" && identity.Rank == 0 && identity.Alpha == 0 && len(identity.TargetKeys) == 0 && identity.BaseModelHash == "" && len(identity.Labels) == 0
}

func (m *rocmModel) SetProbeSink(sink inference.ProbeSink) {
	if m == nil {
		return
	}
	m.stateMutex.Lock()
	m.probeSink = sink
	m.stateMutex.Unlock()
}

func (m *rocmModel) Benchmark(ctx context.Context, cfg inference.BenchConfig) (*inference.BenchReport, error) {
	if m == nil {
		return nil, core.E("rocm.Benchmark", "model is nil", nil)
	}
	prompts := cfg.Prompts
	if len(prompts) == 0 {
		prompts = []string{"hello"}
	}
	measuredRuns := cfg.MeasuredRuns
	if measuredRuns <= 0 {
		measuredRuns = 1
	}
	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 32
	}
	for i := 0; i < cfg.WarmupRuns; i++ {
		for range m.Generate(ctx, prompts[0], inference.WithMaxTokens(maxTokens)) {
		}
		if err := m.Err(); err != nil {
			return nil, err
		}
	}

	var aggregate inference.GenerateMetrics
	for i := 0; i < measuredRuns; i++ {
		for _, prompt := range prompts {
			for range m.Generate(ctx, prompt, inference.WithMaxTokens(maxTokens)) {
			}
			if err := m.Err(); err != nil {
				return nil, err
			}
			metrics := m.Metrics()
			aggregate.PromptTokens += metrics.PromptTokens
			aggregate.GeneratedTokens += metrics.GeneratedTokens
			aggregate.PrefillDuration += metrics.PrefillDuration
			aggregate.DecodeDuration += metrics.DecodeDuration
			if metrics.PeakMemoryBytes > aggregate.PeakMemoryBytes {
				aggregate.PeakMemoryBytes = metrics.PeakMemoryBytes
			}
		}
	}
	return &inference.BenchReport{
		Model:               m.modelIdentity(),
		Adapter:             m.ActiveAdapter(),
		PromptTokens:        aggregate.PromptTokens,
		GeneratedTokens:     aggregate.GeneratedTokens,
		PrefillTokensPerSec: tokensPerSecond(aggregate.PromptTokens, aggregate.PrefillDuration),
		DecodeTokensPerSec:  tokensPerSecond(aggregate.GeneratedTokens, aggregate.DecodeDuration),
		PeakMemoryBytes:     aggregate.PeakMemoryBytes,
	}, nil
}

func (m *rocmModel) Evaluate(ctx context.Context, dataset inference.DatasetStream, cfg inference.EvalConfig) (*inference.EvalReport, error) {
	if m == nil {
		return nil, core.E("rocm.Evaluate", "model is nil", nil)
	}
	if dataset == nil {
		return nil, core.E("rocm.Evaluate", "dataset stream is nil", nil)
	}
	maxSamples := cfg.MaxSamples
	if maxSamples <= 0 {
		maxSamples = 1 << 30
	}
	metrics := inference.EvalMetrics{}
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
		metrics.Tokens += len(m.Encode(sampleText(sample)))
	}
	return &inference.EvalReport{Model: m.modelIdentity(), Adapter: m.ActiveAdapter(), Metrics: metrics}, nil
}

func (m *rocmModel) wrapTokenStream(stream iter.Seq[inference.Token], streamError func() error, promptTokens int, start time.Time) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		var count int
		var firstTokenAt time.Time
		for token := range stream {
			if firstTokenAt.IsZero() {
				firstTokenAt = time.Now()
			}
			count++
			m.emitTokenProbe(token, promptTokens, count)
			if !yield(token) {
				break
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

func (m *rocmModel) emitTokenProbe(token inference.Token, promptTokens, generatedTokens int) {
	if m == nil {
		return
	}
	m.stateMutex.Lock()
	sink := m.probeSink
	m.stateMutex.Unlock()
	if sink == nil {
		return
	}
	sink.EmitProbe(inference.ProbeEvent{
		Kind:  inference.ProbeEventToken,
		Phase: inference.ProbePhaseDecode,
		Token: &inference.ProbeToken{ID: token.ID, Text: token.Text, PromptTokens: promptTokens, GeneratedTokens: generatedTokens},
	})
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

func (m *rocmModel) clearLastError() { m.setLastFailure(nil) }

func (m *rocmModel) setLastFailure(err error) {
	if m == nil {
		return
	}
	m.stateMutex.Lock()
	m.lastError = err
	m.stateMutex.Unlock()
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

func resolveContextLength(requestedContextLength int, metadata gguf.Metadata) int {
	if requestedContextLength > 0 {
		return requestedContextLength
	}
	if metadata.ContextLength == 0 {
		return defaultContextLengthCap
	}
	return min(int(metadata.ContextLength), defaultContextLengthCap)
}

func modelInfoFromMetadata(metadata gguf.Metadata) inference.ModelInfo {
	quantBits, quantGroup := quantisationFromFileType(metadata.FileType)
	return inference.ModelInfo{Architecture: metadata.Architecture, NumLayers: int(metadata.BlockCount), QuantBits: quantBits, QuantGroup: quantGroup}
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

func supportedNativeArchitecture(architecture string) bool {
	if architecture == "" {
		return true
	}
	supported := map[string]struct{}{
		"bert": {}, "deepseek": {}, "gemma": {}, "gemma2": {}, "gemma3": {}, "gemma4": {},
		"gpt-oss": {}, "llama": {}, "mistral": {}, "mixtral": {}, "phi": {}, "phi3": {},
		"qwen2": {}, "qwen3": {},
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
	return core.Contains(quantType, "q4") || core.Contains(quantType, "q5") || core.Contains(quantType, "q8")
}

func estimateKVCacheBytes(layers, contextLength, hidden int, cacheMode string) uint64 {
	bytesPerElement := uint64(2)
	if cacheMode == "q8" {
		bytesPerElement = 1
	}
	return uint64(layers) * uint64(contextLength) * uint64(hidden) * 2 * bytesPerElement
}

func rocmMachineClass(memoryBytes uint64) string {
	switch {
	case memoryBytes >= 64*memoryGiB:
		return "rocm-64gb-plus"
	case memoryBytes >= 24*memoryGiB:
		return "rocm-24gb"
	case memoryBytes >= 16*memoryGiB:
		return "rocm-16gb"
	default:
		return "rocm-small"
	}
}

func rocmRecommendedBatchSize(memoryBytes uint64) int {
	if memoryBytes >= 48*memoryGiB {
		return 8
	}
	if memoryBytes >= 24*memoryGiB {
		return 4
	}
	return 1
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
