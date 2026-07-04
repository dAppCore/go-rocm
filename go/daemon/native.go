// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"dappco.re/go/inference"
	rocm "dappco.re/go/rocm"
)

const DefaultModelName = "default"

var (
	errRunnerNil             = errors.New("native generate runner is nil")
	errPromptRequired        = errors.New("generate prompt is required")
	errNoModelsConfigured    = errors.New("no native models configured")
	errCacheModelMissing     = errors.New("cache model is required")
	errCancelModelMissing    = errors.New("cancel model is required")
	errGenerateModelMissing  = errors.New("generate model is required")
	errEmbedModelMissing     = errors.New("embed model is required")
	errParserModelMissing    = errors.New("parser model is required")
	errRerankModelMissing    = errors.New("rerank model is required")
	errTokenizerModelMissing = errors.New("tokenizer model is required")
)

// NativeGenerateConfig configures the local daemon generate route.
type NativeGenerateConfig struct {
	ModelPaths       map[string]string
	DefaultModelName string
	DefaultMaxTokens int
	LoadOptions      []inference.LoadOption
	ROCmLoadConfig   rocm.ROCmLoadConfig
}

// NativeGenerateRunner loads ROCm models once and serves daemon generate requests.
type NativeGenerateRunner struct {
	mu              sync.Mutex
	modelPaths      map[string]string
	defaultModel    string
	defaultMaxToken int
	loadOptions     []inference.LoadOption
	rocmLoadConfig  rocm.ROCmLoadConfig
	loadModel       func(string, rocm.ROCmLoadConfig, ...inference.LoadOption) (inference.TextModel, error)
	models          atomic.Pointer[map[string]inference.TextModel]
	defaultOpts     []inference.GenerateOption
	scheduleMu      sync.Mutex
	scheduleCancel  map[string]context.CancelFunc
	scheduleSeq     atomic.Uint64
}

// ROCmModelPackInspector exposes ROCm's metadata-only model-pack inspection to
// daemon clients without forcing a model load.
type ROCmModelPackInspector struct{}

func (ROCmModelPackInspector) InspectModelPack(ctx context.Context, req ModelPackRequest) (*inference.ModelPackInspection, error) {
	return rocm.InspectModelPack(ctx, req.Path)
}

// NewNativeGenerateRunner builds a native ROCm generate backend.
func NewNativeGenerateRunner(cfg NativeGenerateConfig) *NativeGenerateRunner {
	defaultModel := strings.TrimSpace(cfg.DefaultModelName)
	if defaultModel == "" {
		defaultModel = DefaultModelName
	}
	runner := &NativeGenerateRunner{
		modelPaths:      copyStringMap(cfg.ModelPaths),
		defaultModel:    defaultModel,
		defaultMaxToken: cfg.DefaultMaxTokens,
		loadOptions:     append([]inference.LoadOption(nil), cfg.LoadOptions...),
		rocmLoadConfig:  cfg.ROCmLoadConfig,
		loadModel:       rocm.LoadModelWithConfig,
		scheduleCancel:  map[string]context.CancelFunc{},
	}
	empty := map[string]inference.TextModel{}
	runner.models.Store(&empty)
	if cfg.DefaultMaxTokens > 0 {
		runner.defaultOpts = []inference.GenerateOption{inference.WithMaxTokens(cfg.DefaultMaxTokens)}
	}
	return runner
}

// Generate runs a prompt or chat request through a cached native ROCm model.
func (runner *NativeGenerateRunner) Generate(ctx context.Context, req GenerateRequest) (GenerateResult, error) {
	if runner == nil {
		return GenerateResult{}, errRunnerNil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	modelName, modelPath, err := runner.resolveModel(req.Model)
	if err != nil {
		return GenerateResult{}, err
	}
	model, err := runner.modelFor(modelName, modelPath)
	if err != nil {
		return GenerateResult{}, err
	}

	opts := runner.generateOptions(req)
	var builder strings.Builder
	if hint := estimateGenerateBytes(req, runner.defaultMaxToken); hint > 0 {
		builder.Grow(hint)
	}
	if len(req.Messages) > 0 {
		for token := range model.Chat(ctx, toInferenceMessages(req.Messages), opts...) {
			builder.WriteString(token.Text)
		}
	} else {
		if strings.TrimSpace(req.Prompt) == "" {
			return GenerateResult{}, errPromptRequired
		}
		for token := range model.Generate(ctx, req.Prompt, opts...) {
			builder.WriteString(token.Text)
		}
	}
	if err := model.Err(); err != nil {
		return GenerateResult{Text: builder.String(), Model: modelName}, err
	}
	return GenerateResult{
		Text:    builder.String(),
		Model:   modelName,
		Metrics: toDaemonMetrics(model.Metrics()),
	}, nil
}

// Embed runs an embedding request through a cached native ROCm model.
func (runner *NativeGenerateRunner) Embed(ctx context.Context, req inference.EmbeddingRequest) (*inference.EmbeddingResult, error) {
	if runner == nil {
		return nil, errRunnerNil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	modelName, modelPath, err := runner.resolveEmbeddingModel(req.Model)
	if err != nil {
		return nil, err
	}
	model, err := runner.modelFor(modelName, modelPath)
	if err != nil {
		return nil, err
	}
	embeddingModel, ok := model.(inference.EmbeddingModel)
	if !ok {
		return nil, fmt.Errorf("model %q does not support embeddings", modelName)
	}
	req.Model = modelName
	result, err := embeddingModel.Embed(ctx, req)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return nil, errors.New("native embedding result is nil")
	}
	return result, nil
}

// Rerank runs a rerank request through a cached native ROCm model.
func (runner *NativeGenerateRunner) Rerank(ctx context.Context, req inference.RerankRequest) (*inference.RerankResult, error) {
	if runner == nil {
		return nil, errRunnerNil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	modelName, modelPath, err := runner.resolveRerankModel(req.Model)
	if err != nil {
		return nil, err
	}
	model, err := runner.modelFor(modelName, modelPath)
	if err != nil {
		return nil, err
	}
	rerankModel, ok := model.(inference.RerankModel)
	if !ok {
		return nil, fmt.Errorf("model %q does not support rerank", modelName)
	}
	req.Model = modelName
	result, err := rerankModel.Rerank(ctx, req)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return nil, errors.New("native rerank result is nil")
	}
	return result, nil
}

// Schedule enqueues a prompt or chat request and returns a token stream.
func (runner *NativeGenerateRunner) Schedule(ctx context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	if runner == nil {
		return inference.RequestHandle{}, nil, errRunnerNil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	modelName, modelPath, err := runner.resolveModel(req.Model)
	if err != nil {
		return inference.RequestHandle{}, nil, err
	}
	model, err := runner.modelFor(modelName, modelPath)
	if err != nil {
		return inference.RequestHandle{}, nil, err
	}
	req = cloneScheduledRequest(req)
	req.Model = modelName
	if strings.TrimSpace(req.ID) == "" {
		req.ID = fmt.Sprintf("rocm-%d", runner.scheduleSeq.Add(1))
	}
	reqCtx, cancel := context.WithCancel(ctx)
	if err := runner.registerScheduleCancel(req.ID, cancel); err != nil {
		cancel()
		return inference.RequestHandle{}, nil, err
	}
	handle := inference.RequestHandle{
		ID:     req.ID,
		Model:  inference.ModelIdentity{ID: modelName, Path: modelPath},
		Labels: cloneLabels(req.Labels),
	}
	if scheduler, ok := model.(inference.SchedulerModel); ok {
		scheduledHandle, stream, err := scheduler.Schedule(reqCtx, req)
		if err != nil {
			runner.forgetScheduleCancel(req.ID)
			cancel()
			return inference.RequestHandle{}, nil, err
		}
		if scheduledHandle.ID != "" {
			handle.ID = scheduledHandle.ID
		}
		if hasModelIdentity(scheduledHandle.Model) {
			handle.Model = scheduledHandle.Model
		}
		if len(scheduledHandle.Labels) > 0 {
			handle.Labels = cloneLabels(scheduledHandle.Labels)
		}
		registeredIDs := []string{req.ID}
		if handle.ID != req.ID {
			if err := runner.registerScheduleCancel(handle.ID, cancel); err != nil {
				runner.forgetScheduleCancel(req.ID)
				cancel()
				return inference.RequestHandle{}, nil, err
			}
			registeredIDs = append(registeredIDs, handle.ID)
		}
		out := make(chan inference.ScheduledToken, 1)
		go runner.forwardScheduledStream(reqCtx, registeredIDs, handle.ID, stream, out)
		return handle, out, nil
	}
	out := make(chan inference.ScheduledToken, 1)
	go runner.generateScheduledStream(reqCtx, req, model, out)
	return handle, out, nil
}

// Models lists configured model aliases for daemon clients.
func (runner *NativeGenerateRunner) Models(ctx context.Context) ([]ModelRecord, error) {
	if runner == nil {
		return nil, errRunnerNil
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	names := make([]string, 0, len(runner.modelPaths))
	for name := range runner.modelPaths {
		names = append(names, name)
	}
	sort.Strings(names)
	records := make([]ModelRecord, 0, len(names))
	loaded := map[string]inference.TextModel(nil)
	if current := runner.models.Load(); current != nil {
		loaded = *current
	}
	for _, name := range names {
		_, isLoaded := loaded[name]
		records = append(records, ModelRecord{
			ID:      name,
			Path:    runner.modelPaths[name],
			Default: name == runner.defaultModel,
			Loaded:  isLoaded,
		})
	}
	return records, nil
}

// ModelInfo reports loaded-model metadata for daemon clients.
func (runner *NativeGenerateRunner) ModelInfo(ctx context.Context, req ModelRequest) (inference.ModelInfo, error) {
	modelName, modelPath, model, err := runner.introspectionModelFor(ctx, req.Model)
	if err != nil {
		return inference.ModelInfo{}, err
	}
	_ = modelName
	_ = modelPath
	return model.Info(), nil
}

// Capabilities reports loaded-model feature support for daemon clients.
func (runner *NativeGenerateRunner) Capabilities(ctx context.Context, req ModelRequest) (inference.CapabilityReport, error) {
	modelName, modelPath, model, err := runner.introspectionModelFor(ctx, req.Model)
	if err != nil {
		return inference.CapabilityReport{}, err
	}
	report, ok := inference.CapabilitiesOf(model)
	if !ok {
		report = inference.TextModelCapabilities(inference.RuntimeIdentity{Backend: "rocm"}, model)
	}
	if report.Runtime.Backend == "" {
		report.Runtime.Backend = "rocm"
	}
	if report.Model.ID == "" {
		report.Model.ID = modelName
	}
	if report.Model.Path == "" {
		report.Model.Path = modelPath
	}
	return report, nil
}

// EngineFeatures reports loaded-model registry features for daemon clients.
func (runner *NativeGenerateRunner) EngineFeatures(ctx context.Context, req EngineFeaturesRequest) (any, error) {
	profile, err := runner.modelProfileForMetadata(ctx, req.Model, req.Path, req.Labels)
	if err != nil {
		return nil, err
	}
	features := profile.EngineFeatures
	if !hasROCmEngineFeatures(features) {
		features = rocm.ROCmEngineFeaturesForProfile(profile)
	}
	if !hasROCmEngineFeatures(features) {
		return nil, errors.New("ROCm engine features not found")
	}
	return features, nil
}

// ModelProfile reports the loaded-model registry profile for daemon clients.
func (runner *NativeGenerateRunner) ModelProfile(ctx context.Context, req ModelProfileRequest) (any, error) {
	profile, err := runner.modelProfileForMetadata(ctx, req.Model, req.Path, req.Labels)
	if err != nil {
		return nil, err
	}
	return profile, nil
}

// ModelRoutes reports the loaded-model route plan for daemon clients.
func (runner *NativeGenerateRunner) ModelRoutes(ctx context.Context, req ModelRoutesRequest) (any, error) {
	plan, err := runner.modelRoutePlanForMetadata(ctx, req.Model, req.Path, req.Labels)
	if err != nil {
		return nil, err
	}
	return plan, nil
}

// InspectModelPack validates local model metadata without loading tensors.
func (runner *NativeGenerateRunner) InspectModelPack(ctx context.Context, req ModelPackRequest) (*inference.ModelPackInspection, error) {
	return ROCmModelPackInspector{}.InspectModelPack(ctx, req)
}

// Tokenize runs text through the selected native ROCm model tokenizer.
func (runner *NativeGenerateRunner) Tokenize(ctx context.Context, req TokenizeRequest) (TokenizeResult, error) {
	modelName, tokenizer, err := runner.tokenizerFor(ctx, req.Model)
	if err != nil {
		return TokenizeResult{}, err
	}
	return TokenizeResult{
		Model:  modelName,
		Tokens: append([]int32(nil), tokenizer.Encode(req.Text)...),
	}, nil
}

// Detokenize renders token IDs through the selected native ROCm model tokenizer.
func (runner *NativeGenerateRunner) Detokenize(ctx context.Context, req DetokenizeRequest) (DetokenizeResult, error) {
	modelName, tokenizer, err := runner.tokenizerFor(ctx, req.Model)
	if err != nil {
		return DetokenizeResult{}, err
	}
	return DetokenizeResult{
		Model: modelName,
		Text:  tokenizer.Decode(append([]int32(nil), req.Tokens...)),
	}, nil
}

// ApplyChatTemplate renders chat messages through the selected native ROCm model template.
func (runner *NativeGenerateRunner) ApplyChatTemplate(ctx context.Context, req ChatTemplateRequest) (ChatTemplateResult, error) {
	modelName, tokenizer, err := runner.tokenizerFor(ctx, req.Model)
	if err != nil {
		return ChatTemplateResult{}, err
	}
	text, err := tokenizer.ApplyChatTemplate(toInferenceMessages(req.Messages))
	if err != nil {
		return ChatTemplateResult{}, err
	}
	return ChatTemplateResult{Model: modelName, Text: text}, nil
}

// ParseReasoning extracts model-family-specific reasoning channels.
func (runner *NativeGenerateRunner) ParseReasoning(ctx context.Context, req ParseRequest) (inference.ReasoningParseResult, error) {
	modelName, model, err := runner.parserModelFor(ctx, req.Model)
	if err != nil {
		return inference.ReasoningParseResult{}, err
	}
	parser, ok := model.(inference.ReasoningParser)
	if !ok {
		return inference.ReasoningParseResult{}, fmt.Errorf("model %q does not support reasoning parsing", modelName)
	}
	return parser.ParseReasoning(tokenIDsToInferenceTokens(req.Tokens), parserTextForModel(req.Text, req.Tokens, model))
}

// ParseTools extracts model-family-specific tool calls.
func (runner *NativeGenerateRunner) ParseTools(ctx context.Context, req ParseRequest) (inference.ToolParseResult, error) {
	modelName, model, err := runner.parserModelFor(ctx, req.Model)
	if err != nil {
		return inference.ToolParseResult{}, err
	}
	parser, ok := model.(inference.ToolParser)
	if !ok {
		return inference.ToolParseResult{}, fmt.Errorf("model %q does not support tool parsing", modelName)
	}
	return parser.ParseTools(tokenIDsToInferenceTokens(req.Tokens), parserTextForModel(req.Text, req.Tokens, model))
}

// CacheStats reports cache health for the selected native ROCm model.
func (runner *NativeGenerateRunner) CacheStats(ctx context.Context, req CacheRequest) (inference.CacheStats, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	service, _, err := runner.cacheServiceFor(ctx, req.Model)
	if err != nil {
		return inference.CacheStats{}, err
	}
	return service.CacheStats(ctx)
}

// WarmCache prepares prompt or token cache blocks on the selected native ROCm model.
func (runner *NativeGenerateRunner) WarmCache(ctx context.Context, req inference.CacheWarmRequest) (inference.CacheWarmResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	modelName := req.Model.ID
	service, resolvedName, err := runner.cacheServiceFor(ctx, modelName)
	if err != nil {
		return inference.CacheWarmResult{}, err
	}
	req.Model.ID = resolvedName
	return service.WarmCache(ctx, req)
}

// ClearCache clears cache blocks on the selected native ROCm model.
func (runner *NativeGenerateRunner) ClearCache(ctx context.Context, req CacheRequest) (inference.CacheStats, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	service, _, err := runner.cacheServiceFor(ctx, req.Model)
	if err != nil {
		return inference.CacheStats{}, err
	}
	return service.ClearCache(ctx, req.Labels)
}

// CacheEntries lists cache blocks for the selected native ROCm model.
func (runner *NativeGenerateRunner) CacheEntries(ctx context.Context, req CacheRequest) (CacheEntriesResult, error) {
	if runner == nil {
		return CacheEntriesResult{}, errRunnerNil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	modelName, modelPath, err := runner.resolveCacheModel(req.Model)
	if err != nil {
		return CacheEntriesResult{}, err
	}
	model, err := runner.modelFor(modelName, modelPath)
	if err != nil {
		return CacheEntriesResult{}, err
	}
	lister, ok := model.(CacheEntryLister)
	if !ok {
		return CacheEntriesResult{}, fmt.Errorf("model %q does not support cache entry listing", modelName)
	}
	entries, err := lister.CacheEntries(ctx, req.Labels)
	if err != nil {
		return CacheEntriesResult{}, err
	}
	result := CacheEntriesResult{Entries: entries}
	if service, ok := model.(inference.CacheService); ok {
		stats, err := service.CacheStats(ctx)
		if err != nil {
			return CacheEntriesResult{}, err
		}
		result.Stats = &stats
	}
	return result, nil
}

// Cancel cancels an in-flight request on the selected native ROCm model.
func (runner *NativeGenerateRunner) Cancel(ctx context.Context, req CancelRequest) (inference.RequestCancelResult, error) {
	if runner == nil {
		return inference.RequestCancelResult{}, errRunnerNil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if runner.cancelScheduled(req.ID) {
		return inference.RequestCancelResult{ID: req.ID, Cancelled: true, Reason: "scheduled request cancelled"}, nil
	}
	modelName, modelPath, err := runner.resolveCancelModel(req.Model)
	if err != nil {
		return inference.RequestCancelResult{}, err
	}
	model, err := runner.modelFor(modelName, modelPath)
	if err != nil {
		return inference.RequestCancelResult{}, err
	}
	cancellable, ok := model.(inference.CancellableModel)
	if !ok {
		return inference.RequestCancelResult{}, fmt.Errorf("model %q does not support cancellation", modelName)
	}
	return cancellable.CancelRequest(ctx, req.ID)
}

func (runner *NativeGenerateRunner) registerScheduleCancel(id string, cancel context.CancelFunc) error {
	runner.scheduleMu.Lock()
	defer runner.scheduleMu.Unlock()
	if runner.scheduleCancel == nil {
		runner.scheduleCancel = map[string]context.CancelFunc{}
	}
	if _, exists := runner.scheduleCancel[id]; exists {
		return fmt.Errorf("duplicate scheduled request id %q", id)
	}
	runner.scheduleCancel[id] = cancel
	return nil
}

func (runner *NativeGenerateRunner) forgetScheduleCancel(id string) {
	runner.scheduleMu.Lock()
	delete(runner.scheduleCancel, id)
	runner.scheduleMu.Unlock()
}

func (runner *NativeGenerateRunner) forgetScheduleCancels(ids ...string) {
	runner.scheduleMu.Lock()
	for _, id := range ids {
		delete(runner.scheduleCancel, id)
	}
	runner.scheduleMu.Unlock()
}

func (runner *NativeGenerateRunner) cancelScheduled(id string) bool {
	id = strings.TrimSpace(id)
	if id == "" {
		return false
	}
	runner.scheduleMu.Lock()
	cancel := runner.scheduleCancel[id]
	if cancel != nil {
		delete(runner.scheduleCancel, id)
	}
	runner.scheduleMu.Unlock()
	if cancel == nil {
		return false
	}
	cancel()
	return true
}

func (runner *NativeGenerateRunner) forwardScheduledStream(ctx context.Context, registeredIDs []string, handleID string, stream <-chan inference.ScheduledToken, out chan<- inference.ScheduledToken) {
	defer close(out)
	defer runner.forgetScheduleCancels(registeredIDs...)
	if stream == nil {
		return
	}
	for token := range stream {
		if token.RequestID == "" {
			token.RequestID = handleID
		}
		select {
		case out <- cloneScheduledToken(token):
		case <-ctx.Done():
			return
		}
	}
}

func (runner *NativeGenerateRunner) generateScheduledStream(ctx context.Context, req inference.ScheduledRequest, model inference.TextModel, out chan<- inference.ScheduledToken) {
	defer close(out)
	defer runner.forgetScheduleCancel(req.ID)
	opts := generateOptionsFromSampler(req.Sampler)
	var stream func(func(inference.Token) bool)
	if len(req.Messages) > 0 {
		stream = model.Chat(ctx, req.Messages, opts...)
	} else {
		stream = model.Generate(ctx, req.Prompt, opts...)
	}
	labels := cloneLabels(req.Labels)
	for token := range stream {
		scheduled := inference.ScheduledToken{
			RequestID: req.ID,
			Token:     token,
			Metrics:   model.Metrics(),
			Labels:    cloneLabels(labels),
		}
		select {
		case out <- scheduled:
		case <-ctx.Done():
			return
		}
	}
}

// Close releases all loaded native models.
func (runner *NativeGenerateRunner) Close() error {
	if runner == nil {
		return nil
	}
	runner.mu.Lock()
	empty := map[string]inference.TextModel{}
	prev := runner.models.Swap(&empty)
	runner.mu.Unlock()

	var closeErr error
	if prev != nil {
		for _, model := range *prev {
			if model == nil {
				continue
			}
			closeErr = errors.Join(closeErr, model.Close())
		}
	}
	return closeErr
}

func (runner *NativeGenerateRunner) resolveModel(requested string) (string, string, error) {
	if len(runner.modelPaths) == 0 {
		return "", "", errNoModelsConfigured
	}
	modelName := strings.TrimSpace(requested)
	if modelName != "" {
		path := runner.modelPaths[modelName]
		if path == "" {
			return "", "", fmt.Errorf("unknown model %q", modelName)
		}
		return modelName, path, nil
	}
	if runner.defaultModel != "" {
		if path := runner.modelPaths[runner.defaultModel]; path != "" {
			return runner.defaultModel, path, nil
		}
	}
	if path := runner.modelPaths["generate"]; path != "" {
		return "generate", path, nil
	}
	if len(runner.modelPaths) == 1 {
		for name, path := range runner.modelPaths {
			return name, path, nil
		}
	}
	return "", "", errGenerateModelMissing
}

func (runner *NativeGenerateRunner) resolveEmbeddingModel(requested string) (string, string, error) {
	if len(runner.modelPaths) == 0 {
		return "", "", errNoModelsConfigured
	}
	modelName := strings.TrimSpace(requested)
	if modelName != "" {
		path := runner.modelPaths[modelName]
		if path == "" {
			return "", "", fmt.Errorf("unknown model %q", modelName)
		}
		return modelName, path, nil
	}
	if path := runner.modelPaths["embed"]; path != "" {
		return "embed", path, nil
	}
	if runner.defaultModel != "" {
		if path := runner.modelPaths[runner.defaultModel]; path != "" {
			return runner.defaultModel, path, nil
		}
	}
	if len(runner.modelPaths) == 1 {
		for name, path := range runner.modelPaths {
			return name, path, nil
		}
	}
	return "", "", errEmbedModelMissing
}

func (runner *NativeGenerateRunner) resolveRerankModel(requested string) (string, string, error) {
	if len(runner.modelPaths) == 0 {
		return "", "", errNoModelsConfigured
	}
	modelName := strings.TrimSpace(requested)
	if modelName != "" {
		path := runner.modelPaths[modelName]
		if path == "" {
			return "", "", fmt.Errorf("unknown model %q", modelName)
		}
		return modelName, path, nil
	}
	if path := runner.modelPaths["rerank"]; path != "" {
		return "rerank", path, nil
	}
	if runner.defaultModel != "" {
		if path := runner.modelPaths[runner.defaultModel]; path != "" {
			return runner.defaultModel, path, nil
		}
	}
	if len(runner.modelPaths) == 1 {
		for name, path := range runner.modelPaths {
			return name, path, nil
		}
	}
	return "", "", errRerankModelMissing
}

func (runner *NativeGenerateRunner) resolveCacheModel(requested string) (string, string, error) {
	if len(runner.modelPaths) == 0 {
		return "", "", errNoModelsConfigured
	}
	modelName := strings.TrimSpace(requested)
	if modelName != "" {
		path := runner.modelPaths[modelName]
		if path == "" {
			return "", "", fmt.Errorf("unknown model %q", modelName)
		}
		return modelName, path, nil
	}
	if path := runner.modelPaths["cache"]; path != "" {
		return "cache", path, nil
	}
	if runner.defaultModel != "" {
		if path := runner.modelPaths[runner.defaultModel]; path != "" {
			return runner.defaultModel, path, nil
		}
	}
	if path := runner.modelPaths["generate"]; path != "" {
		return "generate", path, nil
	}
	if len(runner.modelPaths) == 1 {
		for name, path := range runner.modelPaths {
			return name, path, nil
		}
	}
	return "", "", errCacheModelMissing
}

func (runner *NativeGenerateRunner) resolveCancelModel(requested string) (string, string, error) {
	modelName, modelPath, err := runner.resolveModel(requested)
	if errors.Is(err, errGenerateModelMissing) {
		return "", "", errCancelModelMissing
	}
	return modelName, modelPath, err
}

func (runner *NativeGenerateRunner) resolveTokenizerModel(requested string) (string, string, error) {
	modelName, modelPath, err := runner.resolveModel(requested)
	if errors.Is(err, errGenerateModelMissing) {
		return "", "", errTokenizerModelMissing
	}
	return modelName, modelPath, err
}

func (runner *NativeGenerateRunner) resolveParserModel(requested string) (string, string, error) {
	modelName, modelPath, err := runner.resolveModel(requested)
	if errors.Is(err, errGenerateModelMissing) {
		return "", "", errParserModelMissing
	}
	return modelName, modelPath, err
}

func (runner *NativeGenerateRunner) cacheServiceFor(ctx context.Context, requested string) (inference.CacheService, string, error) {
	if runner == nil {
		return nil, "", errRunnerNil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	modelName, modelPath, err := runner.resolveCacheModel(requested)
	if err != nil {
		return nil, "", err
	}
	model, err := runner.modelFor(modelName, modelPath)
	if err != nil {
		return nil, "", err
	}
	service, ok := model.(inference.CacheService)
	if !ok {
		return nil, "", fmt.Errorf("model %q does not support cache service", modelName)
	}
	return service, modelName, nil
}

func (runner *NativeGenerateRunner) tokenizerFor(ctx context.Context, requested string) (string, inference.TokenizerModel, error) {
	if runner == nil {
		return "", nil, errRunnerNil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	modelName, modelPath, err := runner.resolveTokenizerModel(requested)
	if err != nil {
		return "", nil, err
	}
	model, err := runner.modelFor(modelName, modelPath)
	if err != nil {
		return "", nil, err
	}
	tokenizer, ok := model.(inference.TokenizerModel)
	if !ok {
		return "", nil, fmt.Errorf("model %q does not support tokenization", modelName)
	}
	return modelName, tokenizer, nil
}

func (runner *NativeGenerateRunner) parserModelFor(ctx context.Context, requested string) (string, inference.TextModel, error) {
	if runner == nil {
		return "", nil, errRunnerNil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return "", nil, err
	}
	modelName, modelPath, err := runner.resolveParserModel(requested)
	if err != nil {
		return "", nil, err
	}
	model, err := runner.modelFor(modelName, modelPath)
	if err != nil {
		return "", nil, err
	}
	return modelName, model, nil
}

func (runner *NativeGenerateRunner) introspectionModelFor(ctx context.Context, requested string) (string, string, inference.TextModel, error) {
	if runner == nil {
		return "", "", nil, errRunnerNil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return "", "", nil, err
	}
	modelName, modelPath, err := runner.resolveModel(requested)
	if err != nil {
		return "", "", nil, err
	}
	model, err := runner.modelFor(modelName, modelPath)
	if err != nil {
		return "", "", nil, err
	}
	return modelName, modelPath, model, nil
}

func (runner *NativeGenerateRunner) metadataModelFor(ctx context.Context, requested, path string) (string, string, inference.TextModel, error) {
	if runner == nil {
		return "", "", nil, errRunnerNil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return "", "", nil, err
	}
	modelName := strings.TrimSpace(requested)
	if modelName != "" {
		return runner.introspectionModelFor(ctx, modelName)
	}
	modelPath := strings.TrimSpace(path)
	if modelPath == "" {
		return runner.introspectionModelFor(ctx, "")
	}
	if len(runner.modelPaths) == 0 {
		return "", "", nil, errNoModelsConfigured
	}
	names := make([]string, 0, len(runner.modelPaths))
	for name := range runner.modelPaths {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if runner.modelPaths[name] != modelPath {
			continue
		}
		return runner.introspectionModelFor(ctx, name)
	}
	return "", "", nil, fmt.Errorf("unknown model path %q", modelPath)
}

func (runner *NativeGenerateRunner) modelProfileForMetadata(ctx context.Context, requested, path string, labels map[string]string) (rocm.ROCmModelProfile, error) {
	modelName, modelPath, model, err := runner.metadataModelFor(ctx, requested, path)
	if err != nil {
		return rocm.ROCmModelProfile{}, err
	}
	return runner.modelProfileForLoadedMetadata(modelName, modelPath, model, labels)
}

func (runner *NativeGenerateRunner) modelRoutePlanForMetadata(ctx context.Context, requested, path string, labels map[string]string) (rocm.ROCmModelRoutePlan, error) {
	modelName, modelPath, model, err := runner.metadataModelFor(ctx, requested, path)
	if err != nil {
		return rocm.ROCmModelRoutePlan{}, err
	}
	profile, err := runner.modelProfileForLoadedMetadata(modelName, modelPath, model, labels)
	if err != nil {
		return rocm.ROCmModelRoutePlan{}, err
	}
	plan := rocm.ROCmModelRoutePlanForProfileAndModel(profile, model)
	if !plan.Matched() {
		return rocm.ROCmModelRoutePlan{}, errors.New("ROCm model routes not found")
	}
	return plan, nil
}

func (runner *NativeGenerateRunner) modelProfileForLoadedMetadata(modelName, modelPath string, model inference.TextModel, labels map[string]string) (rocm.ROCmModelProfile, error) {
	profile, ok := rocm.ResolveROCmModelProfileForModel(model)
	if !ok || !profile.Matched() || shouldRefreshMetadataProfileWithPath(profile, modelPath) {
		infoLabels := mergeMetadataLabels(profile.Model.Labels, labels)
		infoProfile, infoOK := rocm.ResolveROCmModelProfileForInfo(modelPath, model.Info(), infoLabels)
		if infoOK && infoProfile.Matched() {
			profile, ok = infoProfile, true
		}
	}
	if !ok || !profile.Matched() {
		return rocm.ROCmModelProfile{}, fmt.Errorf("ROCm model profile not found for %s", modelPath)
	}
	if profile.Model.ID == "" {
		profile.Model.ID = modelName
	}
	if profile.Model.Path == "" {
		profile.Model.Path = modelPath
	}
	profile.Model.Labels = mergeMetadataLabels(profile.Model.Labels, labels)
	return profile, nil
}

func shouldRefreshMetadataProfileWithPath(profile rocm.ROCmModelProfile, modelPath string) bool {
	return strings.TrimSpace(modelPath) != "" && strings.TrimSpace(profile.Model.Path) == ""
}

func hasROCmEngineFeatures(features rocm.ROCmEngineFeatures) bool {
	return features.Contract != "" ||
		features.Architecture != "" ||
		features.Family != "" ||
		features.RuntimeStatus != "" ||
		features.ReasoningParserID != "" ||
		features.ToolParserID != "" ||
		features.ChatTemplateID != "" ||
		features.NativeRuntime ||
		features.ModelContextWindow ||
		features.TextGenerate ||
		features.DeviceKVState ||
		features.FixedSlidingCache ||
		features.FixedSlidingCacheBound ||
		features.ReasoningParse ||
		features.ToolParse ||
		features.ChatTemplate ||
		features.DefaultThinking ||
		features.Embeddings ||
		features.Rerank ||
		features.MoE ||
		features.SequenceMixer ||
		features.AttachedOnly ||
		len(features.Capabilities) > 0 ||
		len(features.Labels) > 0
}

func mergeMetadataLabels(base, extra map[string]string) map[string]string {
	labels := cloneLabels(base)
	for key, value := range extra {
		if strings.TrimSpace(key) == "" || value == "" {
			continue
		}
		if labels == nil {
			labels = map[string]string{}
		}
		labels[key] = value
	}
	return labels
}

func (runner *NativeGenerateRunner) modelFor(name, path string) (inference.TextModel, error) {
	if current := runner.models.Load(); current != nil {
		if model := (*current)[name]; model != nil {
			return model, nil
		}
	}

	runner.mu.Lock()
	defer runner.mu.Unlock()

	current := runner.models.Load()
	if current != nil {
		if model := (*current)[name]; model != nil {
			return model, nil
		}
	}
	model, err := runner.loadModel(path, runner.rocmLoadConfig, runner.loadOptions...)
	if err != nil {
		return nil, fmt.Errorf("load native model %q: %w", name, err)
	}
	var next map[string]inference.TextModel
	if current == nil {
		next = map[string]inference.TextModel{name: model}
	} else {
		next = make(map[string]inference.TextModel, len(*current)+1)
		maps.Copy(next, *current)
		next[name] = model
	}
	runner.models.Store(&next)
	return model, nil
}

func (runner *NativeGenerateRunner) generateOptions(req GenerateRequest) []inference.GenerateOption {
	if req.MaxTokens == 0 &&
		req.Temperature == 0 &&
		req.TopK == 0 &&
		req.TopP == 0 &&
		req.MinP == 0 &&
		len(req.StopTokens) == 0 &&
		req.RepeatPenalty == 0 &&
		!req.ReturnLogits &&
		req.EnableThinking == nil {
		return runner.defaultOpts
	}
	opts := make([]inference.GenerateOption, 0, 9)
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = runner.defaultMaxToken
	}
	if maxTokens > 0 {
		opts = append(opts, inference.WithMaxTokens(maxTokens))
	}
	if req.Temperature != 0 {
		opts = append(opts, inference.WithTemperature(float32(req.Temperature)))
	}
	if req.TopK != 0 {
		opts = append(opts, inference.WithTopK(req.TopK))
	}
	if req.TopP != 0 {
		opts = append(opts, inference.WithTopP(float32(req.TopP)))
	}
	if req.MinP != 0 {
		opts = append(opts, inference.WithMinP(float32(req.MinP)))
	}
	if len(req.StopTokens) > 0 {
		opts = append(opts, inference.WithStopTokens(req.StopTokens...))
	}
	if req.RepeatPenalty != 0 {
		opts = append(opts, inference.WithRepeatPenalty(float32(req.RepeatPenalty)))
	}
	if req.ReturnLogits {
		opts = append(opts, inference.WithLogits())
	}
	if req.EnableThinking != nil {
		opts = append(opts, inference.WithEnableThinking(req.EnableThinking))
	}
	return opts
}

func generateOptionsFromSampler(cfg inference.SamplerConfig) []inference.GenerateOption {
	opts := make([]inference.GenerateOption, 0, 7)
	if cfg.MaxTokens != 0 {
		opts = append(opts, inference.WithMaxTokens(cfg.MaxTokens))
	}
	if cfg.Temperature != 0 {
		opts = append(opts, inference.WithTemperature(cfg.Temperature))
	}
	if cfg.TopK != 0 {
		opts = append(opts, inference.WithTopK(cfg.TopK))
	}
	if cfg.TopP != 0 {
		opts = append(opts, inference.WithTopP(cfg.TopP))
	}
	if cfg.MinP != 0 {
		opts = append(opts, inference.WithMinP(cfg.MinP))
	}
	if len(cfg.StopTokens) > 0 {
		opts = append(opts, inference.WithStopTokens(cfg.StopTokens...))
	}
	if cfg.RepeatPenalty != 0 {
		opts = append(opts, inference.WithRepeatPenalty(cfg.RepeatPenalty))
	}
	if cfg.ReturnLogits {
		opts = append(opts, inference.WithLogits())
	}
	return opts
}

func cloneScheduledRequest(req inference.ScheduledRequest) inference.ScheduledRequest {
	req.Messages = append([]inference.Message(nil), req.Messages...)
	req.Sampler.StopTokens = append([]int32(nil), req.Sampler.StopTokens...)
	req.Sampler.StopSequences = append([]string(nil), req.Sampler.StopSequences...)
	req.Labels = cloneLabels(req.Labels)
	return req
}

func toInferenceMessages(messages []Message) []inference.Message {
	out := make([]inference.Message, len(messages))
	for i, message := range messages {
		out[i] = inference.Message{Role: message.Role, Content: message.Content}
	}
	return out
}

func tokenIDsToInferenceTokens(ids []int32) []inference.Token {
	if len(ids) == 0 {
		return nil
	}
	tokens := make([]inference.Token, len(ids))
	for i, id := range ids {
		tokens[i] = inference.Token{ID: id}
	}
	return tokens
}

func parserTextForModel(text string, ids []int32, model inference.TextModel) string {
	if strings.TrimSpace(text) != "" || len(ids) == 0 {
		return text
	}
	tokenizer, ok := model.(inference.TokenizerModel)
	if !ok {
		return text
	}
	return tokenizer.Decode(append([]int32(nil), ids...))
}

func estimateGenerateBytes(req GenerateRequest, fallbackMaxTokens int) int {
	const bytesPerToken = 4
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = fallbackMaxTokens
	}
	if maxTokens <= 0 {
		return 0
	}
	return maxTokens * bytesPerToken
}

func copyStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	maps.Copy(out, in)
	return out
}
