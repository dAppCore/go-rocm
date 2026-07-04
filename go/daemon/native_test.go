// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"context"
	"iter"
	"strings"
	"testing"
	"time"

	"dappco.re/go/inference"
	rocm "dappco.re/go/rocm"
	rocmmodel "dappco.re/go/rocm/model"
)

type fakeNativeModel struct {
	generatePrompt string
	chatMessages   []inference.Message
	lastConfig     inference.GenerateConfig
	err            error
	closed         bool
	metrics        inference.GenerateMetrics
	info           inference.ModelInfo
}

type fakeNativeCacheProfileTextModel struct {
	fakeNativeModel
	cacheProfile       rocmmodel.CacheProfile
	cacheProfileCalled bool
}

func (model *fakeNativeCacheProfileTextModel) CacheProfile(context.Context) (rocmmodel.CacheProfile, error) {
	model.cacheProfileCalled = true
	return model.cacheProfile.Clone(), nil
}

type fakeNativeEmbeddingTextModel struct {
	fakeNativeModel
	embedRequest inference.EmbeddingRequest
	embedResult  *inference.EmbeddingResult
}

func (model *fakeNativeEmbeddingTextModel) Embed(_ context.Context, req inference.EmbeddingRequest) (*inference.EmbeddingResult, error) {
	model.embedRequest = req
	if model.embedResult != nil {
		return model.embedResult, nil
	}
	return &inference.EmbeddingResult{
		Model:   inference.ModelIdentity{ID: req.Model, Architecture: "bert"},
		Vectors: [][]float32{{1, 0}},
		Usage:   inference.EmbeddingUsage{PromptTokens: len(req.Input), TotalTokens: len(req.Input)},
		Labels:  map[string]string{"backend": "fake"},
	}, nil
}

type fakeNativeRerankTextModel struct {
	fakeNativeModel
	rerankRequest inference.RerankRequest
	rerankResult  *inference.RerankResult
}

func (model *fakeNativeRerankTextModel) Rerank(_ context.Context, req inference.RerankRequest) (*inference.RerankResult, error) {
	model.rerankRequest = req
	if model.rerankResult != nil {
		return model.rerankResult, nil
	}
	results := make([]inference.RerankScore, 0, len(req.Documents))
	for i := range req.Documents {
		results = append(results, inference.RerankScore{Index: i, Score: 1 / float64(i+1), Text: req.Documents[i]})
	}
	return &inference.RerankResult{
		Model:   inference.ModelIdentity{ID: req.Model, Architecture: "bert"},
		Results: results,
		Labels:  map[string]string{"backend": "fake"},
	}, nil
}

type fakeNativeTokenizerParserTextModel struct {
	fakeNativeModel
	encodedText      string
	decodedIDs       []int32
	templateMessages []inference.Message
	reasoningTokens  []inference.Token
	reasoningText    string
	toolsTokens      []inference.Token
	toolsText        string
}

func (model *fakeNativeTokenizerParserTextModel) Encode(text string) []int32 {
	model.encodedText = text
	return []int32{7, 8, 9}
}

func (model *fakeNativeTokenizerParserTextModel) Decode(ids []int32) string {
	model.decodedIDs = append([]int32(nil), ids...)
	return "decoded text"
}

func (model *fakeNativeTokenizerParserTextModel) ApplyChatTemplate(messages []inference.Message) (string, error) {
	model.templateMessages = append([]inference.Message(nil), messages...)
	return "<user>" + messages[len(messages)-1].Content + "</user>", nil
}

func (model *fakeNativeTokenizerParserTextModel) ParseReasoning(tokens []inference.Token, text string) (inference.ReasoningParseResult, error) {
	model.reasoningTokens = append([]inference.Token(nil), tokens...)
	model.reasoningText = text
	return inference.ReasoningParseResult{
		VisibleText: "answer",
		Reasoning:   []inference.ReasoningSegment{{Text: "plan"}},
	}, nil
}

func (model *fakeNativeTokenizerParserTextModel) ParseTools(tokens []inference.Token, text string) (inference.ToolParseResult, error) {
	model.toolsTokens = append([]inference.Token(nil), tokens...)
	model.toolsText = text
	return inference.ToolParseResult{
		VisibleText: "done",
		Calls:       []inference.ToolCall{{Name: "lookup", ArgumentsJSON: `{"id":7}`}},
	}, nil
}

type fakeNativeCacheCancellableTextModel struct {
	fakeNativeModel
	statsCalled   bool
	warmRequest   inference.CacheWarmRequest
	clearLabels   map[string]string
	entryLabels   map[string]string
	cancelID      string
	statsResult   inference.CacheStats
	warmResult    inference.CacheWarmResult
	clearResult   inference.CacheStats
	entriesResult []inference.CacheBlockRef
	cancelResult  inference.RequestCancelResult
}

type fakeNativeSchedulerTextModel struct {
	fakeNativeModel
	scheduleRequest inference.ScheduledRequest
	scheduleHandle  inference.RequestHandle
	scheduleTokens  []inference.ScheduledToken
}

func (model *fakeNativeSchedulerTextModel) Schedule(_ context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	model.scheduleRequest = req
	ch := make(chan inference.ScheduledToken, len(model.scheduleTokens))
	for _, token := range model.scheduleTokens {
		ch <- token
	}
	close(ch)
	if model.scheduleHandle.ID != "" {
		return model.scheduleHandle, ch, nil
	}
	return inference.RequestHandle{
		ID:     req.ID,
		Model:  inference.ModelIdentity{ID: req.Model},
		Labels: cloneLabels(req.Labels),
	}, ch, nil
}

func (model *fakeNativeCacheCancellableTextModel) CacheStats(context.Context) (inference.CacheStats, error) {
	model.statsCalled = true
	if hasFakeCacheStats(model.statsResult) {
		return model.statsResult, nil
	}
	return inference.CacheStats{Blocks: 1, CacheMode: "block-q8"}, nil
}

func (model *fakeNativeCacheCancellableTextModel) WarmCache(_ context.Context, req inference.CacheWarmRequest) (inference.CacheWarmResult, error) {
	model.warmRequest = req
	if len(model.warmResult.Blocks) != 0 || hasFakeCacheStats(model.warmResult.Stats) || len(model.warmResult.Labels) != 0 {
		return model.warmResult, nil
	}
	return inference.CacheWarmResult{
		Blocks: []inference.CacheBlockRef{{ID: "blk", TokenCount: len(req.Tokens)}},
		Stats:  inference.CacheStats{Blocks: 1, CacheMode: "block-q8"},
	}, nil
}

func (model *fakeNativeCacheCancellableTextModel) ClearCache(_ context.Context, labels map[string]string) (inference.CacheStats, error) {
	model.clearLabels = labels
	if hasFakeCacheStats(model.clearResult) {
		return model.clearResult, nil
	}
	return inference.CacheStats{CacheMode: "block-q8"}, nil
}

func (model *fakeNativeCacheCancellableTextModel) CacheEntries(_ context.Context, labels map[string]string) ([]inference.CacheBlockRef, error) {
	model.entryLabels = labels
	if len(model.entriesResult) != 0 {
		return model.entriesResult, nil
	}
	return []inference.CacheBlockRef{{ID: "blk", TokenCount: 1}}, nil
}

func (model *fakeNativeCacheCancellableTextModel) CancelRequest(_ context.Context, id string) (inference.RequestCancelResult, error) {
	model.cancelID = id
	if model.cancelResult.ID != "" || model.cancelResult.Cancelled || model.cancelResult.Reason != "" || len(model.cancelResult.Labels) != 0 {
		return model.cancelResult, nil
	}
	return inference.RequestCancelResult{ID: id, Cancelled: id != ""}, nil
}

func hasFakeCacheStats(stats inference.CacheStats) bool {
	return stats.Blocks != 0 ||
		stats.MemoryBytes != 0 ||
		stats.DiskBytes != 0 ||
		stats.Hits != 0 ||
		stats.Misses != 0 ||
		stats.Evictions != 0 ||
		stats.HitRate != 0 ||
		stats.RestoreMillis != 0 ||
		stats.CacheMode != "" ||
		len(stats.Labels) != 0
}

func (model *fakeNativeModel) Generate(_ context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	model.generatePrompt = prompt
	model.lastConfig = inference.ApplyGenerateOpts(opts)
	return func(yield func(inference.Token) bool) {
		if !yield(inference.Token{Text: "hel"}) {
			return
		}
		yield(inference.Token{Text: "lo"})
	}
}

func (model *fakeNativeModel) Chat(_ context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	model.chatMessages = append([]inference.Message(nil), messages...)
	model.lastConfig = inference.ApplyGenerateOpts(opts)
	return func(yield func(inference.Token) bool) {
		yield(inference.Token{Text: "chat"})
	}
}

func (model *fakeNativeModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (model *fakeNativeModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (model *fakeNativeModel) ModelType() string { return "fake" }
func (model *fakeNativeModel) Info() inference.ModelInfo {
	if model.info != (inference.ModelInfo{}) {
		return model.info
	}
	return inference.ModelInfo{Architecture: "fake"}
}
func (model *fakeNativeModel) Metrics() inference.GenerateMetrics {
	return model.metrics
}
func (model *fakeNativeModel) Err() error { return model.err }
func (model *fakeNativeModel) Close() error {
	model.closed = true
	return nil
}

func collectScheduledTokens(t *testing.T, stream <-chan inference.ScheduledToken) []inference.ScheduledToken {
	t.Helper()
	tokens := []inference.ScheduledToken{}
	for token := range stream {
		tokens = append(tokens, token)
	}
	return tokens
}

func TestNativeGenerateRunner_Good_GeneratesWithDefaultModel(t *testing.T) {
	native := &fakeNativeModel{metrics: inference.GenerateMetrics{
		PromptTokens:        8,
		GeneratedTokens:     2,
		PrefillDuration:     10 * time.Millisecond,
		DecodeDuration:      20 * time.Millisecond,
		TotalDuration:       30 * time.Millisecond,
		PrefillTokensPerSec: 800,
		DecodeTokensPerSec:  100,
		PeakMemoryBytes:     64,
		ActiveMemoryBytes:   32,
	}}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths:       map[string]string{"default": "/models/main"},
		DefaultMaxTokens: 64,
	})
	var loadedPath string
	runner.loadModel = func(path string, _ rocm.ROCmLoadConfig, _ ...inference.LoadOption) (inference.TextModel, error) {
		loadedPath = path
		return native, nil
	}

	result, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "hello", MaxTokens: 4})

	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if loadedPath != "/models/main" || native.generatePrompt != "hello" {
		t.Fatalf("loaded path/prompt = %q/%q, want /models/main/hello", loadedPath, native.generatePrompt)
	}
	if result.Text != "hello" || result.Model != "default" {
		t.Fatalf("result = %+v, want hello/default", result)
	}
	if result.Metrics.PromptTokens != 8 ||
		result.Metrics.GeneratedTokens != 2 ||
		result.Metrics.PrefillSeconds != 0.01 ||
		result.Metrics.DecodeSeconds != 0.02 ||
		result.Metrics.TotalSeconds != 0.03 ||
		result.Metrics.PeakMemoryBytes != 64 ||
		result.Metrics.ActiveMemoryBytes != 32 {
		t.Fatalf("metrics = %+v, want translated inference metrics", result.Metrics)
	}
	if native.lastConfig.MaxTokens != 4 {
		t.Fatalf("max tokens = %d, want 4", native.lastConfig.MaxTokens)
	}
}

func TestNativeGenerateRunner_Good_SchedulesWithDefaultModel(t *testing.T) {
	native := &fakeNativeModel{metrics: inference.GenerateMetrics{GeneratedTokens: 2}}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	var loadedPath string
	runner.loadModel = func(path string, _ rocm.ROCmLoadConfig, _ ...inference.LoadOption) (inference.TextModel, error) {
		loadedPath = path
		return native, nil
	}

	handle, stream, err := runner.Schedule(context.Background(), inference.ScheduledRequest{
		ID:     "req-1",
		Prompt: "hello",
		Sampler: inference.SamplerConfig{
			MaxTokens:    4,
			StopTokens:   []int32{2},
			ReturnLogits: true,
		},
		Labels: map[string]string{"tenant": "test"},
	})
	if err != nil {
		t.Fatalf("Schedule() error = %v", err)
	}
	tokens := collectScheduledTokens(t, stream)
	if loadedPath != "/models/main" || native.generatePrompt != "hello" {
		t.Fatalf("loaded path/prompt = %q/%q, want /models/main/hello", loadedPath, native.generatePrompt)
	}
	if handle.ID != "req-1" ||
		handle.Model.ID != "default" ||
		handle.Model.Path != "/models/main" ||
		handle.Labels["tenant"] != "test" {
		t.Fatalf("handle = %+v, want default scheduled request handle", handle)
	}
	if len(tokens) != 2 ||
		tokens[0].RequestID != "req-1" ||
		tokens[0].Token.Text != "hel" ||
		tokens[1].Token.Text != "lo" ||
		tokens[0].Labels["tenant"] != "test" ||
		tokens[0].Metrics.GeneratedTokens != 2 {
		t.Fatalf("tokens = %+v, want fallback scheduled token stream", tokens)
	}
	if native.lastConfig.MaxTokens != 4 ||
		len(native.lastConfig.StopTokens) != 1 ||
		native.lastConfig.StopTokens[0] != 2 ||
		!native.lastConfig.ReturnLogits {
		t.Fatalf("generate config = %+v, want sampler-backed generate options", native.lastConfig)
	}
}

func TestNativeGenerateRunner_Good_ForwardsModelScheduler(t *testing.T) {
	native := &fakeNativeSchedulerTextModel{
		scheduleHandle: inference.RequestHandle{
			ID:    "backend-1",
			Model: inference.ModelIdentity{ID: "main", Path: "/models/main", Architecture: "fake"},
		},
		scheduleTokens: []inference.ScheduledToken{
			{Token: inference.Token{ID: 9, Text: "ok"}},
		},
	}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths:       map[string]string{"main": "/models/main"},
		DefaultModelName: "main",
	})
	runner.loadModel = func(string, rocm.ROCmLoadConfig, ...inference.LoadOption) (inference.TextModel, error) {
		return native, nil
	}

	handle, stream, err := runner.Schedule(context.Background(), inference.ScheduledRequest{
		ID:     "client-1",
		Prompt: "hello",
		Labels: map[string]string{"tenant": "test"},
	})
	if err != nil {
		t.Fatalf("Schedule() error = %v", err)
	}
	tokens := collectScheduledTokens(t, stream)
	if native.scheduleRequest.ID != "client-1" ||
		native.scheduleRequest.Model != "main" ||
		native.scheduleRequest.Prompt != "hello" ||
		native.scheduleRequest.Labels["tenant"] != "test" {
		t.Fatalf("schedule request = %+v, want normalized native scheduler request", native.scheduleRequest)
	}
	if handle.ID != "backend-1" || handle.Model.Architecture != "fake" {
		t.Fatalf("handle = %+v, want backend scheduler handle", handle)
	}
	if len(tokens) != 1 || tokens[0].RequestID != "backend-1" || tokens[0].Token.Text != "ok" {
		t.Fatalf("tokens = %+v, want forwarded scheduler token with handle id", tokens)
	}
}

func TestNativeGenerateRunner_Good_ModelIntrospection(t *testing.T) {
	native := &fakeNativeModel{info: inference.ModelInfo{
		Architecture: "gemma4",
		VocabSize:    32000,
		NumLayers:    2,
		HiddenSize:   16,
		QuantBits:    4,
		QuantGroup:   64,
	}}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{
			"default": "/models/main",
			"embed":   "/models/embed",
		},
	})
	var loadedPath string
	runner.loadModel = func(path string, _ rocm.ROCmLoadConfig, _ ...inference.LoadOption) (inference.TextModel, error) {
		loadedPath = path
		return native, nil
	}

	records, err := runner.Models(context.Background())
	if err != nil {
		t.Fatalf("Models() error = %v", err)
	}
	if len(records) != 2 ||
		records[0].ID != "default" ||
		records[0].Path != "/models/main" ||
		!records[0].Default ||
		records[0].Loaded ||
		records[1].ID != "embed" {
		t.Fatalf("records = %+v, want sorted configured model records before load", records)
	}

	info, err := runner.ModelInfo(context.Background(), ModelRequest{Model: "default"})
	if err != nil {
		t.Fatalf("ModelInfo() error = %v", err)
	}
	if loadedPath != "/models/main" || info.Architecture != "gemma4" || info.QuantBits != 4 {
		t.Fatalf("loaded/info = %q/%+v, want default model info", loadedPath, info)
	}

	records, err = runner.Models(context.Background())
	if err != nil {
		t.Fatalf("Models(after load) error = %v", err)
	}
	if !records[0].Loaded {
		t.Fatalf("records = %+v, want default marked loaded after model_info", records)
	}

	report, err := runner.Capabilities(context.Background(), ModelRequest{Model: "default"})
	if err != nil {
		t.Fatalf("Capabilities() error = %v", err)
	}
	if report.Runtime.Backend != "rocm" ||
		report.Model.ID != "default" ||
		report.Model.Path != "/models/main" ||
		!report.Supports(inference.CapabilityGenerate) ||
		!report.Available {
		t.Fatalf("capability report = %+v, want annotated loaded-model report", report)
	}
}

func TestNativeGenerateRunner_Good_ModelMetadataBackends(t *testing.T) {
	const modelPath = "/models/lmstudio-community-gemma-4-e4b-it-6bit"
	native := &fakeNativeCacheProfileTextModel{
		fakeNativeModel: fakeNativeModel{info: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    26,
			HiddenSize:   2304,
			QuantBits:    6,
			QuantGroup:   64,
		}},
		cacheProfile: rocmmodel.CacheProfile{
			Contract:        rocmmodel.CacheProfileContract,
			Architecture:    "gemma4_text",
			TotalCaches:     1,
			QuantizedCaches: 1,
			MaxCacheTokens:  9,
		},
	}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths:       map[string]string{"main": modelPath},
		DefaultModelName: "main",
	})
	var loadedPath string
	loadCount := 0
	runner.loadModel = func(path string, _ rocm.ROCmLoadConfig, _ ...inference.LoadOption) (inference.TextModel, error) {
		loadedPath = path
		loadCount++
		return native, nil
	}

	profileAny, err := runner.ModelProfile(context.Background(), ModelProfileRequest{
		Model:  " main ",
		Labels: map[string]string{"tenant": "test"},
	})
	if err != nil {
		t.Fatalf("ModelProfile() error = %v", err)
	}
	profile, ok := profileAny.(rocm.ROCmModelProfile)
	if !ok {
		t.Fatalf("ModelProfile() = %#v, want ROCmModelProfile", profileAny)
	}
	if loadedPath != modelPath ||
		!profile.Matched() ||
		profile.Architecture != "gemma4_text" ||
		profile.Model.ID != "main" ||
		profile.Model.Path != modelPath ||
		profile.Model.Labels["tenant"] != "test" ||
		profile.LoadStatus.Status == "" {
		t.Fatalf("profile = %+v loaded=%q, want loaded Gemma4 model metadata", profile, loadedPath)
	}

	featuresAny, err := runner.EngineFeatures(context.Background(), EngineFeaturesRequest{Model: "main"})
	if err != nil {
		t.Fatalf("EngineFeatures() error = %v", err)
	}
	features, ok := featuresAny.(rocm.ROCmEngineFeatures)
	if !ok {
		t.Fatalf("EngineFeatures() = %#v, want ROCmEngineFeatures", featuresAny)
	}
	if features.Architecture != "gemma4_text" ||
		features.Family != "gemma4" ||
		features.ChatTemplateID != "gemma4_hf_turn" ||
		!features.TextGenerate {
		t.Fatalf("features = %+v, want Gemma4 reactive feature metadata", features)
	}

	routesAny, err := runner.ModelRoutes(context.Background(), ModelRoutesRequest{
		Path:   modelPath,
		Labels: map[string]string{"tenant": "route"},
	})
	if err != nil {
		t.Fatalf("ModelRoutes(path) error = %v", err)
	}
	routes, ok := routesAny.(rocm.ROCmModelRoutePlan)
	if !ok {
		t.Fatalf("ModelRoutes(path) = %#v, want ROCmModelRoutePlan", routesAny)
	}
	if routes.Contract != rocm.ROCmModelRoutePlanContract ||
		routes.Architecture != "gemma4_text" ||
		routes.Model.ID != "main" ||
		routes.Model.Path != modelPath ||
		routes.Model.Labels["tenant"] != "route" ||
		routes.Labels["engine_route_plan_contract"] != rocm.ROCmModelRoutePlanContract ||
		routes.Labels["engine_route_plan_cache_profile"] != "true" ||
		routes.Labels["engine_route_plan_cache_profile_contract"] != rocmmodel.CacheProfileContract ||
		routes.Labels["engine_route_plan_cache_profile_max_cache_tokens"] != "9" ||
		routes.CacheProfile.MaxCacheTokens != 9 ||
		!routes.FeatureRoute.Matched() ||
		!routes.TokenizerRoute.Matched() {
		t.Fatalf("routes = %+v, want Gemma4 reactive route plan with live cache profile", routes)
	}
	if !native.cacheProfileCalled {
		t.Fatal("ModelRoutes did not ask the loaded model for its live cache profile")
	}
	if loadCount != 1 {
		t.Fatalf("load count = %d, want path-only metadata lookup to reuse loaded alias", loadCount)
	}

	_, err = runner.ModelRoutes(context.Background(), ModelRoutesRequest{Path: "/models/missing"})
	if err == nil || !strings.Contains(err.Error(), "unknown model path") {
		t.Fatalf("ModelRoutes(unknown path) error = %v, want unknown path", err)
	}
}

func TestNativeGenerateRunner_Good_EmbedsWithEmbedModel(t *testing.T) {
	native := &fakeNativeEmbeddingTextModel{}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{
			"default": "/models/main",
			"embed":   "/models/embed",
		},
	})
	var loadedPath string
	runner.loadModel = func(path string, _ rocm.ROCmLoadConfig, _ ...inference.LoadOption) (inference.TextModel, error) {
		loadedPath = path
		return native, nil
	}

	result, err := runner.Embed(context.Background(), inference.EmbeddingRequest{
		Input:     []string{"hello"},
		Normalize: true,
		Labels:    map[string]string{"tenant": "test"},
	})

	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if loadedPath != "/models/embed" {
		t.Fatalf("loaded path = %q, want embed model path", loadedPath)
	}
	if native.embedRequest.Model != "embed" ||
		len(native.embedRequest.Input) != 1 ||
		native.embedRequest.Input[0] != "hello" ||
		!native.embedRequest.Normalize ||
		native.embedRequest.Labels["tenant"] != "test" {
		t.Fatalf("embed request = %+v, want normalized request", native.embedRequest)
	}
	if len(result.Vectors) != 1 || result.Model.ID != "embed" {
		t.Fatalf("result = %+v, want embedding result", result)
	}
}

func TestNativeGenerateRunner_Bad_EmbedUnsupportedModel(t *testing.T) {
	native := &fakeNativeModel{}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"embed": "/models/embed"},
	})
	runner.loadModel = func(string, rocm.ROCmLoadConfig, ...inference.LoadOption) (inference.TextModel, error) {
		return native, nil
	}

	_, err := runner.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"hello"}})

	if err == nil || !strings.Contains(err.Error(), "does not support embeddings") {
		t.Fatalf("Embed() error = %v, want unsupported embedding error", err)
	}
}

func TestNativeGenerateRunner_Good_ReranksWithRerankModel(t *testing.T) {
	native := &fakeNativeRerankTextModel{}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{
			"default": "/models/main",
			"rerank":  "/models/rerank",
		},
	})
	var loadedPath string
	runner.loadModel = func(path string, _ rocm.ROCmLoadConfig, _ ...inference.LoadOption) (inference.TextModel, error) {
		loadedPath = path
		return native, nil
	}

	result, err := runner.Rerank(context.Background(), inference.RerankRequest{
		Query:     "needle",
		Documents: []string{"first", "second"},
		TopN:      1,
		Labels:    map[string]string{"tenant": "test"},
	})

	if err != nil {
		t.Fatalf("Rerank() error = %v", err)
	}
	if loadedPath != "/models/rerank" {
		t.Fatalf("loaded path = %q, want rerank model path", loadedPath)
	}
	if native.rerankRequest.Model != "rerank" ||
		native.rerankRequest.Query != "needle" ||
		len(native.rerankRequest.Documents) != 2 ||
		native.rerankRequest.TopN != 1 ||
		native.rerankRequest.Labels["tenant"] != "test" {
		t.Fatalf("rerank request = %+v, want normalized request", native.rerankRequest)
	}
	if len(result.Results) != 2 || result.Model.ID != "rerank" {
		t.Fatalf("result = %+v, want rerank result", result)
	}
}

func TestNativeGenerateRunner_Bad_RerankUnsupportedModel(t *testing.T) {
	native := &fakeNativeModel{}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"rerank": "/models/rerank"},
	})
	runner.loadModel = func(string, rocm.ROCmLoadConfig, ...inference.LoadOption) (inference.TextModel, error) {
		return native, nil
	}

	_, err := runner.Rerank(context.Background(), inference.RerankRequest{Query: "q", Documents: []string{"doc"}})

	if err == nil || !strings.Contains(err.Error(), "does not support rerank") {
		t.Fatalf("Rerank() error = %v, want unsupported rerank error", err)
	}
}

func TestNativeGenerateRunner_Good_TokenizerAndParserRoutes(t *testing.T) {
	native := &fakeNativeTokenizerParserTextModel{}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	var loadedPath string
	runner.loadModel = func(path string, _ rocm.ROCmLoadConfig, _ ...inference.LoadOption) (inference.TextModel, error) {
		loadedPath = path
		return native, nil
	}

	tokens, err := runner.Tokenize(context.Background(), TokenizeRequest{Text: "hello"})
	if err != nil {
		t.Fatalf("Tokenize() error = %v", err)
	}
	if loadedPath != "/models/main" || tokens.Model != "default" || len(tokens.Tokens) != 3 || native.encodedText != "hello" {
		t.Fatalf("tokenize loaded/result/text = %q/%+v/%q, want default tokenizer", loadedPath, tokens, native.encodedText)
	}

	text, err := runner.Detokenize(context.Background(), DetokenizeRequest{Tokens: []int32{7, 8, 9}})
	if err != nil {
		t.Fatalf("Detokenize() error = %v", err)
	}
	if text.Model != "default" || text.Text != "decoded text" || len(native.decodedIDs) != 3 {
		t.Fatalf("detokenize result/ids = %+v/%+v, want decoded IDs", text, native.decodedIDs)
	}

	template, err := runner.ApplyChatTemplate(context.Background(), ChatTemplateRequest{
		Messages: []Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("ApplyChatTemplate() error = %v", err)
	}
	if template.Model != "default" || template.Text != "<user>hello</user>" || len(native.templateMessages) != 1 {
		t.Fatalf("template result/messages = %+v/%+v, want rendered chat template", template, native.templateMessages)
	}

	reasoning, err := runner.ParseReasoning(context.Background(), ParseRequest{
		Text:   "<think>plan</think>answer",
		Tokens: []int32{1, 2},
	})
	if err != nil {
		t.Fatalf("ParseReasoning() error = %v", err)
	}
	if reasoning.VisibleText != "answer" ||
		len(native.reasoningTokens) != 2 ||
		native.reasoningTokens[0].ID != 1 ||
		native.reasoningText != "<think>plan</think>answer" {
		t.Fatalf("reasoning result/tokens/text = %+v/%+v/%q, want native parser call", reasoning, native.reasoningTokens, native.reasoningText)
	}

	tools, err := runner.ParseTools(context.Background(), ParseRequest{Tokens: []int32{3, 4}})
	if err != nil {
		t.Fatalf("ParseTools() error = %v", err)
	}
	if tools.VisibleText != "done" ||
		len(native.toolsTokens) != 2 ||
		native.toolsText != "decoded text" {
		t.Fatalf("tools result/tokens/text = %+v/%+v/%q, want token-only request decoded for parser", tools, native.toolsTokens, native.toolsText)
	}
}

func TestNativeGenerateRunner_Bad_TokenizerAndParserUnsupportedModels(t *testing.T) {
	native := &fakeNativeModel{}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	runner.loadModel = func(string, rocm.ROCmLoadConfig, ...inference.LoadOption) (inference.TextModel, error) {
		return native, nil
	}

	_, err := runner.Tokenize(context.Background(), TokenizeRequest{Text: "hello"})
	if err == nil || !strings.Contains(err.Error(), "does not support tokenization") {
		t.Fatalf("Tokenize() error = %v, want unsupported tokenization error", err)
	}
	_, err = runner.ParseReasoning(context.Background(), ParseRequest{Text: "hello"})
	if err == nil || !strings.Contains(err.Error(), "does not support reasoning parsing") {
		t.Fatalf("ParseReasoning() error = %v, want unsupported parser error", err)
	}
	_, err = runner.ParseTools(context.Background(), ParseRequest{Text: "hello"})
	if err == nil || !strings.Contains(err.Error(), "does not support tool parsing") {
		t.Fatalf("ParseTools() error = %v, want unsupported tool parser error", err)
	}
}

func TestNativeGenerateRunner_Good_CacheAndCancelControls(t *testing.T) {
	cacheModel := &fakeNativeCacheCancellableTextModel{
		statsResult: inference.CacheStats{Blocks: 4, CacheMode: "block-q8"},
		warmResult: inference.CacheWarmResult{
			Blocks: []inference.CacheBlockRef{{ID: "cache-blk", TokenCount: 3}},
			Stats:  inference.CacheStats{Blocks: 1, CacheMode: "block-q8"},
		},
		clearResult:   inference.CacheStats{CacheMode: "block-q8"},
		entriesResult: []inference.CacheBlockRef{{ID: "cache-entry", TokenCount: 3}},
	}
	defaultModel := &fakeNativeCacheCancellableTextModel{
		cancelResult: inference.RequestCancelResult{ID: "req-1", Cancelled: true, Reason: "cancelled"},
	}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{
			"default": "/models/main",
			"cache":   "/models/cache",
		},
	})
	loaded := map[string]int{}
	runner.loadModel = func(path string, _ rocm.ROCmLoadConfig, _ ...inference.LoadOption) (inference.TextModel, error) {
		loaded[path]++
		switch path {
		case "/models/cache":
			return cacheModel, nil
		case "/models/main":
			return defaultModel, nil
		default:
			return &fakeNativeModel{}, nil
		}
	}

	stats, err := runner.CacheStats(context.Background(), CacheRequest{})
	if err != nil {
		t.Fatalf("CacheStats() error = %v", err)
	}
	if stats.Blocks != 4 || stats.CacheMode != "block-q8" || !cacheModel.statsCalled || loaded["/models/cache"] != 1 {
		t.Fatalf("stats/cache load = %+v/%v/%+v, want cache alias service", stats, cacheModel.statsCalled, loaded)
	}

	warmed, err := runner.WarmCache(context.Background(), inference.CacheWarmRequest{
		Prompt: "prefill",
		Tokens: []int32{1, 2, 3},
		Mode:   "block-q8",
		Labels: map[string]string{"tenant": "test"},
	})
	if err != nil {
		t.Fatalf("WarmCache() error = %v", err)
	}
	if len(warmed.Blocks) != 1 ||
		warmed.Blocks[0].ID != "cache-blk" ||
		cacheModel.warmRequest.Model.ID != "cache" ||
		cacheModel.warmRequest.Prompt != "prefill" ||
		len(cacheModel.warmRequest.Tokens) != 3 ||
		cacheModel.warmRequest.Labels["tenant"] != "test" {
		t.Fatalf("warm result/request = %+v/%+v, want cache alias warm", warmed, cacheModel.warmRequest)
	}

	cleared, err := runner.ClearCache(context.Background(), CacheRequest{Labels: map[string]string{"tenant": "test"}})
	if err != nil {
		t.Fatalf("ClearCache() error = %v", err)
	}
	if cleared.CacheMode != "block-q8" || cacheModel.clearLabels["tenant"] != "test" {
		t.Fatalf("clear result/labels = %+v/%+v, want cache alias clear", cleared, cacheModel.clearLabels)
	}

	entries, err := runner.CacheEntries(context.Background(), CacheRequest{Labels: map[string]string{"tenant": "test"}})
	if err != nil {
		t.Fatalf("CacheEntries() error = %v", err)
	}
	if len(entries.Entries) != 1 ||
		entries.Entries[0].ID != "cache-entry" ||
		entries.Stats == nil ||
		entries.Stats.CacheMode != "block-q8" ||
		cacheModel.entryLabels["tenant"] != "test" {
		t.Fatalf("entries result/labels = %+v/%+v, want cache alias entries with stats", entries, cacheModel.entryLabels)
	}

	cancelled, err := runner.Cancel(context.Background(), CancelRequest{ID: "req-1"})
	if err != nil {
		t.Fatalf("Cancel() error = %v", err)
	}
	if !cancelled.Cancelled ||
		cancelled.ID != "req-1" ||
		defaultModel.cancelID != "req-1" ||
		loaded["/models/main"] != 1 {
		t.Fatalf("cancel result/default load = %+v/%q/%+v, want default model cancellation", cancelled, defaultModel.cancelID, loaded)
	}
}

func TestNativeGenerateRunner_Bad_CacheAndCancelUnsupportedModels(t *testing.T) {
	native := &fakeNativeModel{}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{
			"default": "/models/main",
			"cache":   "/models/cache",
		},
	})
	runner.loadModel = func(string, rocm.ROCmLoadConfig, ...inference.LoadOption) (inference.TextModel, error) {
		return native, nil
	}

	_, err := runner.CacheStats(context.Background(), CacheRequest{})
	if err == nil || !strings.Contains(err.Error(), "does not support cache service") {
		t.Fatalf("CacheStats() error = %v, want unsupported cache service error", err)
	}
	_, err = runner.CacheEntries(context.Background(), CacheRequest{})
	if err == nil || !strings.Contains(err.Error(), "does not support cache entry listing") {
		t.Fatalf("CacheEntries() error = %v, want unsupported cache entry listing error", err)
	}
	_, err = runner.Cancel(context.Background(), CancelRequest{ID: "req-1"})
	if err == nil || !strings.Contains(err.Error(), "does not support cancellation") {
		t.Fatalf("Cancel() error = %v, want unsupported cancellation error", err)
	}
}

func TestNativeGenerateRunner_Bad_UnknownModel(t *testing.T) {
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"main": "/models/main"},
	})

	_, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "hello", Model: "missing"})

	if err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Fatalf("Generate() error = %v, want unknown model", err)
	}
}

func TestNativeGenerateRunner_Ugly_ChatMessagesAndGenerationKnobs(t *testing.T) {
	thinking := false
	native := &fakeNativeModel{}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths:       map[string]string{"default": "/models/main"},
		DefaultMaxTokens: 12,
	})
	runner.loadModel = func(string, rocm.ROCmLoadConfig, ...inference.LoadOption) (inference.TextModel, error) {
		return native, nil
	}

	result, err := runner.Generate(context.Background(), GenerateRequest{
		Messages:       []Message{{Role: "system", Content: "steady"}, {Role: "user", Content: "hello"}},
		Temperature:    0.7,
		TopK:           32,
		TopP:           0.95,
		MinP:           0.02,
		RepeatPenalty:  1.15,
		EnableThinking: &thinking,
	})

	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if result.Text != "chat" || len(native.chatMessages) != 2 {
		t.Fatalf("result/chat messages = %q/%+v, want chat with two messages", result.Text, native.chatMessages)
	}
	if native.chatMessages[0].Role != "system" || native.chatMessages[1].Content != "hello" {
		t.Fatalf("chat messages = %+v", native.chatMessages)
	}
	if native.lastConfig.MaxTokens != 12 ||
		native.lastConfig.Temperature != 0.7 ||
		native.lastConfig.TopK != 32 ||
		native.lastConfig.TopP != 0.95 ||
		native.lastConfig.MinP != 0.02 ||
		native.lastConfig.RepeatPenalty != 1.15 ||
		native.lastConfig.EnableThinking == nil ||
		*native.lastConfig.EnableThinking {
		t.Fatalf("generate config = %+v, want daemon generation knobs", native.lastConfig)
	}
}

func TestNativeGenerateRunner_Close_Good_ClosesLoadedModels(t *testing.T) {
	native := &fakeNativeModel{}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	runner.loadModel = func(string, rocm.ROCmLoadConfig, ...inference.LoadOption) (inference.TextModel, error) {
		return native, nil
	}
	if _, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "hello"}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	if err := runner.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if !native.closed {
		t.Fatal("native model was not closed")
	}
}
