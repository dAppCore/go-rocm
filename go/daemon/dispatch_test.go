// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"context"
	"strings"
	"testing"

	"dappco.re/go/inference"
	scorepkg "dappco.re/go/rocm/score"
)

type fakeGenerateBackend struct {
	request GenerateRequest
	result  GenerateResult
	err     error
}

func (backend *fakeGenerateBackend) Generate(_ context.Context, request GenerateRequest) (GenerateResult, error) {
	backend.request = request
	return backend.result, backend.err
}

type fakeEmbedBackend struct {
	request inference.EmbeddingRequest
	result  *inference.EmbeddingResult
	err     error
}

func (backend *fakeEmbedBackend) Embed(_ context.Context, request inference.EmbeddingRequest) (*inference.EmbeddingResult, error) {
	backend.request = request
	return backend.result, backend.err
}

type fakeEngineFeaturesBackend struct {
	request EngineFeaturesRequest
	result  any
	err     error
}

func (backend *fakeEngineFeaturesBackend) EngineFeatures(_ context.Context, request EngineFeaturesRequest) (any, error) {
	backend.request = request
	return backend.result, backend.err
}

type fakeRerankBackend struct {
	request inference.RerankRequest
	result  *inference.RerankResult
	err     error
}

func (backend *fakeRerankBackend) Rerank(_ context.Context, request inference.RerankRequest) (*inference.RerankResult, error) {
	backend.request = request
	return backend.result, backend.err
}

type fakeScheduleBackend struct {
	request inference.ScheduledRequest
	handle  inference.RequestHandle
	tokens  []inference.ScheduledToken
	err     error
}

func (backend *fakeScheduleBackend) Schedule(_ context.Context, request inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	backend.request = request
	if backend.err != nil {
		return inference.RequestHandle{}, nil, backend.err
	}
	ch := make(chan inference.ScheduledToken, len(backend.tokens))
	for _, token := range backend.tokens {
		ch <- token
	}
	close(ch)
	return backend.handle, ch, nil
}

type fakeScoreBackend struct {
	request ScoreRequest
	result  ScoreResult
	err     error
}

func (backend *fakeScoreBackend) Score(_ context.Context, request ScoreRequest) (ScoreResult, error) {
	backend.request = request
	return backend.result, backend.err
}

type fakeCacheBackend struct {
	statsRequest   CacheRequest
	warmRequest    inference.CacheWarmRequest
	clearRequest   CacheRequest
	entriesRequest CacheRequest
	statsResult    inference.CacheStats
	warmResult     inference.CacheWarmResult
	clearResult    inference.CacheStats
	entriesResult  CacheEntriesResult
	err            error
}

func (backend *fakeCacheBackend) CacheStats(_ context.Context, request CacheRequest) (inference.CacheStats, error) {
	backend.statsRequest = request
	return backend.statsResult, backend.err
}

func (backend *fakeCacheBackend) WarmCache(_ context.Context, request inference.CacheWarmRequest) (inference.CacheWarmResult, error) {
	backend.warmRequest = request
	return backend.warmResult, backend.err
}

func (backend *fakeCacheBackend) ClearCache(_ context.Context, request CacheRequest) (inference.CacheStats, error) {
	backend.clearRequest = request
	return backend.clearResult, backend.err
}

func (backend *fakeCacheBackend) CacheEntries(_ context.Context, request CacheRequest) (CacheEntriesResult, error) {
	backend.entriesRequest = request
	return backend.entriesResult, backend.err
}

type fakeCancelBackend struct {
	request CancelRequest
	result  inference.RequestCancelResult
	err     error
}

func (backend *fakeCancelBackend) Cancel(_ context.Context, request CancelRequest) (inference.RequestCancelResult, error) {
	backend.request = request
	return backend.result, backend.err
}

type fakeIntrospectionBackend struct {
	infoRequest         ModelRequest
	capabilitiesRequest ModelRequest
	modelsResult        []ModelRecord
	infoResult          inference.ModelInfo
	capabilitiesResult  inference.CapabilityReport
	err                 error
}

func (backend *fakeIntrospectionBackend) Models(context.Context) ([]ModelRecord, error) {
	return backend.modelsResult, backend.err
}

func (backend *fakeIntrospectionBackend) ModelInfo(_ context.Context, request ModelRequest) (inference.ModelInfo, error) {
	backend.infoRequest = request
	return backend.infoResult, backend.err
}

func (backend *fakeIntrospectionBackend) Capabilities(_ context.Context, request ModelRequest) (inference.CapabilityReport, error) {
	backend.capabilitiesRequest = request
	return backend.capabilitiesResult, backend.err
}

type fakeModelPackBackend struct {
	request ModelPackRequest
	result  *inference.ModelPackInspection
	err     error
}

func (backend *fakeModelPackBackend) InspectModelPack(_ context.Context, request ModelPackRequest) (*inference.ModelPackInspection, error) {
	backend.request = request
	return backend.result, backend.err
}

type fakeModelProfileBackend struct {
	request ModelProfileRequest
	result  any
	err     error
}

func (backend *fakeModelProfileBackend) ModelProfile(_ context.Context, request ModelProfileRequest) (any, error) {
	backend.request = request
	return backend.result, backend.err
}

type fakeModelRoutesBackend struct {
	request ModelRoutesRequest
	result  any
	err     error
}

func (backend *fakeModelRoutesBackend) ModelRoutes(_ context.Context, request ModelRoutesRequest) (any, error) {
	backend.request = request
	return backend.result, backend.err
}

type fakeModelRegistryBackend struct {
	request ModelRegistryRequest
	result  any
	err     error
}

func (backend *fakeModelRegistryBackend) ModelRegistry(_ context.Context, request ModelRegistryRequest) (any, error) {
	backend.request = request
	return backend.result, backend.err
}

type fakeTokenizerBackend struct {
	tokenizeRequest     TokenizeRequest
	detokenizeRequest   DetokenizeRequest
	chatTemplateRequest ChatTemplateRequest
	tokenizeResult      TokenizeResult
	detokenizeResult    DetokenizeResult
	chatTemplateResult  ChatTemplateResult
	err                 error
}

func (backend *fakeTokenizerBackend) Tokenize(_ context.Context, request TokenizeRequest) (TokenizeResult, error) {
	backend.tokenizeRequest = request
	return backend.tokenizeResult, backend.err
}

func (backend *fakeTokenizerBackend) Detokenize(_ context.Context, request DetokenizeRequest) (DetokenizeResult, error) {
	backend.detokenizeRequest = request
	return backend.detokenizeResult, backend.err
}

func (backend *fakeTokenizerBackend) ApplyChatTemplate(_ context.Context, request ChatTemplateRequest) (ChatTemplateResult, error) {
	backend.chatTemplateRequest = request
	return backend.chatTemplateResult, backend.err
}

type fakeParserBackend struct {
	reasoningRequest ParseRequest
	toolsRequest     ParseRequest
	reasoningResult  inference.ReasoningParseResult
	toolsResult      inference.ToolParseResult
	err              error
}

func (backend *fakeParserBackend) ParseReasoning(_ context.Context, request ParseRequest) (inference.ReasoningParseResult, error) {
	backend.reasoningRequest = request
	return backend.reasoningResult, backend.err
}

func (backend *fakeParserBackend) ParseTools(_ context.Context, request ParseRequest) (inference.ToolParseResult, error) {
	backend.toolsRequest = request
	return backend.toolsResult, backend.err
}

func TestRegistry_RegisterGenerateBackend_Good_DispatchesGenerate(t *testing.T) {
	thinking := true
	backend := &fakeGenerateBackend{
		result: GenerateResult{
			Text:  "pong",
			Model: "main",
			Metrics: GenerateMetrics{
				PromptTokens:    4,
				GeneratedTokens: 1,
			},
		},
	}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterGenerateBackend(backend); err != nil {
		t.Fatalf("RegisterGenerateBackend() error = %v", err)
	}

	resp, err := registry.Dispatch(context.Background(), Request{
		Action:         " generate ",
		Prompt:         "ping",
		Model:          "main",
		MaxTokens:      32,
		Temperature:    0.2,
		TopK:           10,
		TopP:           0.9,
		MinP:           0.05,
		RepeatPenalty:  1.1,
		EnableThinking: &thinking,
	})

	if err != nil {
		t.Fatalf("Dispatch() error = %v", err)
	}
	if resp["status"] != "ok" || resp["action"] != "generate" || resp["text"] != "pong" || resp["model"] != "main" {
		t.Fatalf("response = %+v, want successful generate response", resp)
	}
	if backend.request.Prompt != "ping" ||
		backend.request.MaxTokens != 32 ||
		backend.request.Temperature != 0.2 ||
		backend.request.TopK != 10 ||
		backend.request.TopP != 0.9 ||
		backend.request.MinP != 0.05 ||
		backend.request.RepeatPenalty != 1.1 ||
		backend.request.EnableThinking == nil ||
		!*backend.request.EnableThinking {
		t.Fatalf("backend request = %+v, want normalized daemon generation knobs", backend.request)
	}
	if _, ok := resp["metrics"].(GenerateMetrics); !ok {
		t.Fatalf("metrics = %#v, want GenerateMetrics", resp["metrics"])
	}
}

func TestRegistry_RegisterEmbedBackend_Good_DispatchesEmbed(t *testing.T) {
	backend := &fakeEmbedBackend{
		result: &inference.EmbeddingResult{
			Model:   inference.ModelIdentity{ID: "embedder", Architecture: "bert"},
			Vectors: [][]float32{{1, 0}, {0, 1}},
			Usage:   inference.EmbeddingUsage{PromptTokens: 4, TotalTokens: 4},
			Labels:  map[string]string{"backend": "fake"},
		},
	}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterEmbedBackend(backend); err != nil {
		t.Fatalf("RegisterEmbedBackend() error = %v", err)
	}

	resp, err := registry.Dispatch(context.Background(), Request{
		Action:    " embed ",
		Input:     []string{"one", "two"},
		Model:     "embedder",
		Normalize: true,
		Labels:    map[string]string{"tenant": "test"},
	})

	if err != nil {
		t.Fatalf("Dispatch(embed) error = %v", err)
	}
	if resp["status"] != "ok" || resp["action"] != "embed" {
		t.Fatalf("response = %+v, want successful embed response", resp)
	}
	vectors, ok := resp["vectors"].([][]float32)
	if !ok || len(vectors) != 2 {
		t.Fatalf("vectors = %#v, want two vectors", resp["vectors"])
	}
	if backend.request.Model != "embedder" ||
		len(backend.request.Input) != 2 ||
		backend.request.Input[0] != "one" ||
		!backend.request.Normalize ||
		backend.request.Labels["tenant"] != "test" {
		t.Fatalf("backend request = %+v, want normalized embedding request", backend.request)
	}
	if _, ok := resp["model"].(inference.ModelIdentity); !ok {
		t.Fatalf("model = %#v, want ModelIdentity", resp["model"])
	}
	if _, ok := resp["usage"].(inference.EmbeddingUsage); !ok {
		t.Fatalf("usage = %#v, want EmbeddingUsage", resp["usage"])
	}
}

func TestRegistry_RegisterEmbedBackend_Good_TextFallback(t *testing.T) {
	backend := &fakeEmbedBackend{result: &inference.EmbeddingResult{Vectors: [][]float32{{1}}}}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterEmbedBackend(backend); err != nil {
		t.Fatalf("RegisterEmbedBackend() error = %v", err)
	}

	_, err := registry.Dispatch(context.Background(), Request{Action: "embed", Text: "fallback"})

	if err != nil {
		t.Fatalf("Dispatch(embed text fallback) error = %v", err)
	}
	if len(backend.request.Input) != 1 || backend.request.Input[0] != "fallback" {
		t.Fatalf("backend input = %+v, want text fallback", backend.request.Input)
	}
}

func TestRegistry_RegisterEmbedBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterEmbedBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "embed backend is nil") {
		t.Fatalf("RegisterEmbedBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_RegisterEmbedBackend_Bad_EmptyInput(t *testing.T) {
	backend := &fakeEmbedBackend{result: &inference.EmbeddingResult{Vectors: [][]float32{{1}}}}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterEmbedBackend(backend); err != nil {
		t.Fatalf("RegisterEmbedBackend() error = %v", err)
	}

	_, err := registry.Dispatch(context.Background(), Request{Action: "embed"})

	if err == nil || !strings.Contains(err.Error(), "embed input") {
		t.Fatalf("Dispatch(empty embed) error = %v, want input error", err)
	}
}

func TestRegistry_RegisterEngineFeaturesBackend_Good_DispatchesFeatures(t *testing.T) {
	backend := &fakeEngineFeaturesBackend{
		result: map[string]any{
			"architecture":  "gemma4_text",
			"text_generate": true,
		},
	}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterEngineFeaturesBackend(backend); err != nil {
		t.Fatalf("RegisterEngineFeaturesBackend() error = %v", err)
	}

	resp, err := registry.Dispatch(context.Background(), Request{
		Action:  " engine_features ",
		Path:    " /models/main ",
		Model:   " main ",
		Backend: " rocm ",
		Labels:  map[string]string{"tenant": "test"},
	})

	if err != nil {
		t.Fatalf("Dispatch(engine_features) error = %v", err)
	}
	features, ok := resp["features"].(map[string]any)
	if resp["status"] != "ok" ||
		resp["action"] != "engine_features" ||
		resp["path"] != "/models/main" ||
		resp["model"] != "main" ||
		resp["backend"] != "rocm" ||
		!ok ||
		features["architecture"] != "gemma4_text" {
		t.Fatalf("engine_features response = %+v, want reactive engine feature payload", resp)
	}
	if backend.request.Path != "/models/main" ||
		backend.request.Model != "main" ||
		backend.request.Backend != "rocm" ||
		backend.request.Labels["tenant"] != "test" {
		t.Fatalf("backend request = %+v, want trimmed path/model/backend and labels", backend.request)
	}
}

func TestRegistry_RegisterEngineFeaturesBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterEngineFeaturesBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "engine features backend is nil") {
		t.Fatalf("RegisterEngineFeaturesBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_RegisterRerankBackend_Good_DispatchesRerank(t *testing.T) {
	backend := &fakeRerankBackend{
		result: &inference.RerankResult{
			Model: inference.ModelIdentity{ID: "reranker", Architecture: "bert"},
			Results: []inference.RerankScore{
				{Index: 1, Score: 0.9, Text: "second"},
				{Index: 0, Score: 0.4, Text: "first"},
			},
			Labels: map[string]string{"backend": "fake"},
		},
	}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterRerankBackend(backend); err != nil {
		t.Fatalf("RegisterRerankBackend() error = %v", err)
	}

	resp, err := registry.Dispatch(context.Background(), Request{
		Action:    " rerank ",
		Model:     "reranker",
		Query:     "needle",
		Documents: []string{"first", "second"},
		TopN:      1,
		Labels:    map[string]string{"tenant": "test"},
	})

	if err != nil {
		t.Fatalf("Dispatch(rerank) error = %v", err)
	}
	if resp["status"] != "ok" || resp["action"] != "rerank" {
		t.Fatalf("response = %+v, want successful rerank response", resp)
	}
	results, ok := resp["results"].([]inference.RerankScore)
	if !ok || len(results) != 2 || results[0].Index != 1 {
		t.Fatalf("results = %#v, want rerank scores", resp["results"])
	}
	if backend.request.Model != "reranker" ||
		backend.request.Query != "needle" ||
		len(backend.request.Documents) != 2 ||
		backend.request.TopN != 1 ||
		backend.request.Labels["tenant"] != "test" {
		t.Fatalf("backend request = %+v, want normalized rerank request", backend.request)
	}
	if _, ok := resp["model"].(inference.ModelIdentity); !ok {
		t.Fatalf("model = %#v, want ModelIdentity", resp["model"])
	}
}

func TestRegistry_RegisterRerankBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterRerankBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "rerank backend is nil") {
		t.Fatalf("RegisterRerankBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_RegisterRerankBackend_Bad_MissingQueryOrDocuments(t *testing.T) {
	backend := &fakeRerankBackend{result: &inference.RerankResult{}}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterRerankBackend(backend); err != nil {
		t.Fatalf("RegisterRerankBackend() error = %v", err)
	}

	_, err := registry.Dispatch(context.Background(), Request{Action: "rerank", Documents: []string{"doc"}})
	if err == nil || !strings.Contains(err.Error(), "rerank query") {
		t.Fatalf("Dispatch(rerank no query) error = %v, want query error", err)
	}
	_, err = registry.Dispatch(context.Background(), Request{Action: "rerank", Query: "q"})
	if err == nil || !strings.Contains(err.Error(), "rerank documents") {
		t.Fatalf("Dispatch(rerank no docs) error = %v, want documents error", err)
	}
}

func TestRegistry_RegisterScheduleBackend_Good_DispatchesSchedule(t *testing.T) {
	backend := &fakeScheduleBackend{
		handle: inference.RequestHandle{
			ID:     "req-1",
			Model:  inference.ModelIdentity{ID: "main", Path: "/models/main"},
			Labels: map[string]string{"tenant": "test"},
		},
		tokens: []inference.ScheduledToken{
			{RequestID: "req-1", Token: inference.Token{ID: 7, Text: "hel"}, Labels: map[string]string{"chunk": "0"}},
			{RequestID: "req-1", Token: inference.Token{ID: 8, Text: "lo"}, Labels: map[string]string{"chunk": "1"}},
		},
	}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterScheduleBackend(backend); err != nil {
		t.Fatalf("RegisterScheduleBackend() error = %v", err)
	}

	resp, err := registry.Dispatch(context.Background(), Request{
		Action:        " schedule ",
		ID:            " req-1 ",
		Model:         "main",
		Text:          "hello",
		MaxTokens:     16,
		Temperature:   0.4,
		TopK:          32,
		TopP:          0.92,
		MinP:          0.03,
		StopTokens:    []int32{2},
		StopSequences: []string{"\n\n"},
		RepeatPenalty: 1.1,
		ReturnLogits:  true,
		Labels:        map[string]string{"tenant": "test"},
	})

	if err != nil {
		t.Fatalf("Dispatch(schedule) error = %v", err)
	}
	if resp["status"] != "ok" || resp["action"] != "schedule" || resp["id"] != "req-1" || resp["text"] != "hello" {
		t.Fatalf("response = %+v, want successful schedule response", resp)
	}
	handle, ok := resp["handle"].(inference.RequestHandle)
	if !ok || handle.ID != "req-1" || handle.Model.ID != "main" || handle.Labels["tenant"] != "test" {
		t.Fatalf("handle = %#v, want request handle payload", resp["handle"])
	}
	tokens, ok := resp["tokens"].([]inference.ScheduledToken)
	if !ok || len(tokens) != 2 || tokens[0].Token.Text != "hel" || tokens[1].Token.Text != "lo" {
		t.Fatalf("tokens = %#v, want scheduled token stream", resp["tokens"])
	}
	if backend.request.ID != "req-1" ||
		backend.request.Model != "main" ||
		backend.request.Prompt != "hello" ||
		backend.request.Sampler.MaxTokens != 16 ||
		backend.request.Sampler.Temperature != 0.4 ||
		backend.request.Sampler.TopK != 32 ||
		backend.request.Sampler.TopP != 0.92 ||
		backend.request.Sampler.MinP != 0.03 ||
		backend.request.Sampler.RepeatPenalty != 1.1 ||
		!backend.request.Sampler.ReturnLogits ||
		len(backend.request.Sampler.StopTokens) != 1 ||
		backend.request.Sampler.StopTokens[0] != 2 ||
		len(backend.request.Sampler.StopSequences) != 1 ||
		backend.request.Sampler.StopSequences[0] != "\n\n" ||
		backend.request.Labels["tenant"] != "test" {
		t.Fatalf("backend request = %+v, want normalized schedule request", backend.request)
	}
}

func TestRegistry_RegisterScheduleBackend_Bad_InputValidation(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterScheduleBackend(&fakeScheduleBackend{}); err != nil {
		t.Fatalf("RegisterScheduleBackend() error = %v", err)
	}

	_, err := registry.Dispatch(context.Background(), Request{Action: "schedule"})

	if err == nil || !strings.Contains(err.Error(), "schedule prompt or messages") {
		t.Fatalf("Dispatch(schedule empty) = %v, want input error", err)
	}
}

func TestRegistry_RegisterScheduleBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterScheduleBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "schedule backend is nil") {
		t.Fatalf("RegisterScheduleBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_DefaultScoreRoute_Good_Text(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	resp, err := registry.Dispatch(context.Background(), Request{
		Action: " score ",
		Text:   "you're absolutely right, I should have known better",
	})

	if err != nil {
		t.Fatalf("Dispatch(score text) error = %v", err)
	}
	if resp["status"] != "ok" || resp["action"] != "score" || resp["kind"] != "text" {
		t.Fatalf("response = %+v, want successful text score", resp)
	}
	result, ok := resp["score"].(*scorepkg.ScoreResult)
	if !ok || result == nil || result.Sycophancy == nil || result.LEK == nil {
		t.Fatalf("score = %#v, want populated score result", resp["score"])
	}
	if _, hasPair := resp["pair"]; hasPair {
		t.Fatalf("response = %+v, text score should not include pair", resp)
	}
}

func TestRegistry_DefaultScoreRoute_Good_Pair(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	resp, err := registry.Dispatch(context.Background(), Request{
		Action:   "score",
		Prompt:   "Can you check this plan?",
		Response: "Yes, your plan is completely correct and I agree.",
	})

	if err != nil {
		t.Fatalf("Dispatch(score pair) error = %v", err)
	}
	if resp["status"] != "ok" || resp["action"] != "score" || resp["kind"] != "pair" {
		t.Fatalf("response = %+v, want successful pair score", resp)
	}
	result, ok := resp["pair"].(*scorepkg.DiffResult)
	if !ok || result == nil || result.Prompt.Sycophancy == nil || result.Response.Sycophancy == nil {
		t.Fatalf("pair = %#v, want populated diff result", resp["pair"])
	}
	if _, hasScore := resp["score"]; hasScore {
		t.Fatalf("response = %+v, pair score should not include single score", resp)
	}
}

func TestRegistry_DefaultScoreRoute_Bad_EmptyText(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	_, err := registry.Dispatch(context.Background(), Request{Action: "score"})

	if err == nil || !strings.Contains(err.Error(), "score text") {
		t.Fatalf("Dispatch(empty score) error = %v, want required text error", err)
	}
}

func TestRegistry_RegisterScoreBackend_Good_DispatchesScore(t *testing.T) {
	backend := &fakeScoreBackend{result: ScoreResult{Kind: "custom"}}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterScoreBackend(backend); err != nil {
		t.Fatalf("RegisterScoreBackend() error = %v", err)
	}

	resp, err := registry.Dispatch(context.Background(), Request{
		Action:   "score",
		Text:     "fallback",
		Prompt:   "prompt",
		Response: "response",
		Model:    "score",
	})

	if err != nil {
		t.Fatalf("Dispatch(score custom) error = %v", err)
	}
	if resp["status"] != "ok" || resp["action"] != "score" || resp["kind"] != "custom" {
		t.Fatalf("response = %+v, want custom score response", resp)
	}
	if backend.request.Text != "fallback" ||
		backend.request.Prompt != "prompt" ||
		backend.request.Response != "response" ||
		backend.request.Model != "score" {
		t.Fatalf("backend request = %+v, want normalized score request", backend.request)
	}
}

func TestRegistry_RegisterScoreBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterScoreBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "score backend is nil") {
		t.Fatalf("RegisterScoreBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_RegisterCacheBackend_Good_DispatchesCacheControls(t *testing.T) {
	backend := &fakeCacheBackend{
		statsResult: inference.CacheStats{Blocks: 2, Hits: 3, Misses: 1, HitRate: 0.75, CacheMode: "block-q8"},
		warmResult: inference.CacheWarmResult{
			Blocks: []inference.CacheBlockRef{{ID: "blk", TokenCount: 3}},
			Stats:  inference.CacheStats{Blocks: 1, CacheMode: "block-q8"},
			Labels: map[string]string{"warmed": "true"},
		},
		clearResult: inference.CacheStats{CacheMode: "block-q8"},
		entriesResult: CacheEntriesResult{
			Entries: []inference.CacheBlockRef{{ID: "blk-a", TokenCount: 3}},
			Stats:   &inference.CacheStats{Blocks: 1, CacheMode: "block-q8"},
		},
	}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterCacheBackend(backend); err != nil {
		t.Fatalf("RegisterCacheBackend() error = %v", err)
	}
	if err := registry.RegisterCacheEntryBackend(backend); err != nil {
		t.Fatalf("RegisterCacheEntryBackend() error = %v", err)
	}

	statsResp, err := registry.Dispatch(context.Background(), Request{
		Action: " cache_stats ",
		Model:  "main",
		Labels: map[string]string{"tenant": "test"},
	})
	if err != nil {
		t.Fatalf("Dispatch(cache_stats) error = %v", err)
	}
	if statsResp["status"] != "ok" || statsResp["action"] != "cache_stats" {
		t.Fatalf("stats response = %+v, want successful cache_stats response", statsResp)
	}
	stats, ok := statsResp["stats"].(inference.CacheStats)
	if !ok || stats.Blocks != 2 || stats.CacheMode != "block-q8" {
		t.Fatalf("stats = %#v, want cache stats payload", statsResp["stats"])
	}
	if backend.statsRequest.Model != "main" || backend.statsRequest.Labels["tenant"] != "test" {
		t.Fatalf("stats request = %+v, want model and labels", backend.statsRequest)
	}

	warmResp, err := registry.Dispatch(context.Background(), Request{
		Action: "cache_warm",
		Model:  "main",
		Prompt: "prefill me",
		Tokens: []int32{1, 2, 3},
		Mode:   "block-q8",
		Labels: map[string]string{"tenant": "test"},
	})
	if err != nil {
		t.Fatalf("Dispatch(cache_warm) error = %v", err)
	}
	if warmResp["status"] != "ok" || warmResp["action"] != "cache_warm" {
		t.Fatalf("warm response = %+v, want successful cache_warm response", warmResp)
	}
	warmed, ok := warmResp["result"].(inference.CacheWarmResult)
	if !ok || len(warmed.Blocks) != 1 || warmed.Blocks[0].ID != "blk" {
		t.Fatalf("warm result = %#v, want warmed blocks", warmResp["result"])
	}
	if backend.warmRequest.Model.ID != "main" ||
		backend.warmRequest.Prompt != "prefill me" ||
		len(backend.warmRequest.Tokens) != 3 ||
		backend.warmRequest.Mode != "block-q8" ||
		backend.warmRequest.Labels["tenant"] != "test" {
		t.Fatalf("warm request = %+v, want normalized cache warm request", backend.warmRequest)
	}

	clearResp, err := registry.Dispatch(context.Background(), Request{
		Action: "cache_clear",
		Model:  "main",
		Labels: map[string]string{"tenant": "test"},
	})
	if err != nil {
		t.Fatalf("Dispatch(cache_clear) error = %v", err)
	}
	if clearResp["status"] != "ok" || clearResp["action"] != "cache_clear" {
		t.Fatalf("clear response = %+v, want successful cache_clear response", clearResp)
	}
	if backend.clearRequest.Model != "main" || backend.clearRequest.Labels["tenant"] != "test" {
		t.Fatalf("clear request = %+v, want model and labels", backend.clearRequest)
	}

	entriesResp, err := registry.Dispatch(context.Background(), Request{
		Action: "cache_entries",
		Model:  "main",
		Labels: map[string]string{"tenant": "test"},
	})
	if err != nil {
		t.Fatalf("Dispatch(cache_entries) error = %v", err)
	}
	if entriesResp["status"] != "ok" || entriesResp["action"] != "cache_entries" {
		t.Fatalf("entries response = %+v, want successful cache_entries response", entriesResp)
	}
	entries, ok := entriesResp["entries"].([]inference.CacheBlockRef)
	if !ok || len(entries) != 1 || entries[0].ID != "blk-a" {
		t.Fatalf("entries = %#v, want cache entries payload", entriesResp["entries"])
	}
	if _, ok := entriesResp["stats"].(inference.CacheStats); !ok {
		t.Fatalf("stats = %#v, want optional cache stats payload", entriesResp["stats"])
	}
	if backend.entriesRequest.Model != "main" || backend.entriesRequest.Labels["tenant"] != "test" {
		t.Fatalf("entries request = %+v, want model and labels", backend.entriesRequest)
	}
}

func TestRegistry_RegisterCacheBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterCacheBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "cache backend is nil") {
		t.Fatalf("RegisterCacheBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_RegisterCacheEntryBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterCacheEntryBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "cache entry backend is nil") {
		t.Fatalf("RegisterCacheEntryBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_RegisterCancelBackend_Good_DispatchesCancel(t *testing.T) {
	backend := &fakeCancelBackend{
		result: inference.RequestCancelResult{ID: "req-1", Cancelled: true, Reason: "cancelled"},
	}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterCancelBackend(backend); err != nil {
		t.Fatalf("RegisterCancelBackend() error = %v", err)
	}

	resp, err := registry.Dispatch(context.Background(), Request{
		Action: " cancel ",
		Model:  "main",
		ID:     " req-1 ",
	})

	if err != nil {
		t.Fatalf("Dispatch(cancel) error = %v", err)
	}
	if resp["status"] != "ok" || resp["action"] != "cancel" {
		t.Fatalf("response = %+v, want successful cancel response", resp)
	}
	result, ok := resp["result"].(inference.RequestCancelResult)
	if !ok || !result.Cancelled || result.ID != "req-1" {
		t.Fatalf("result = %#v, want cancellation result", resp["result"])
	}
	if backend.request.Model != "main" || backend.request.ID != "req-1" {
		t.Fatalf("backend request = %+v, want normalized cancel request", backend.request)
	}
}

func TestRegistry_RegisterCancelBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterCancelBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "cancel backend is nil") {
		t.Fatalf("RegisterCancelBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_RegisterCancelBackend_Bad_MissingID(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterCancelBackend(&fakeCancelBackend{}); err != nil {
		t.Fatalf("RegisterCancelBackend() error = %v", err)
	}

	_, err := registry.Dispatch(context.Background(), Request{Action: "cancel"})

	if err == nil || !strings.Contains(err.Error(), "cancel id") {
		t.Fatalf("Dispatch(cancel missing id) = %v, want id error", err)
	}
}

func TestRegistry_RegisterIntrospectionBackend_Good_DispatchesModelDiscovery(t *testing.T) {
	backend := &fakeIntrospectionBackend{
		modelsResult: []ModelRecord{
			{ID: "default", Path: "/models/main", Default: true},
			{ID: "embed", Path: "/models/embed"},
		},
		infoResult: inference.ModelInfo{Architecture: "gemma4", VocabSize: 42, NumLayers: 2, HiddenSize: 8, QuantBits: 4},
		capabilitiesResult: inference.CapabilityReport{
			Runtime:   inference.RuntimeIdentity{Backend: "rocm"},
			Available: true,
			Capabilities: []inference.Capability{
				inference.SupportedCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel),
			},
		},
	}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterIntrospectionBackend(backend); err != nil {
		t.Fatalf("RegisterIntrospectionBackend() error = %v", err)
	}

	modelsResp, err := registry.Dispatch(context.Background(), Request{Action: " models "})
	if err != nil {
		t.Fatalf("Dispatch(models) error = %v", err)
	}
	models, ok := modelsResp["models"].([]ModelRecord)
	if modelsResp["status"] != "ok" || modelsResp["action"] != "models" || modelsResp["object"] != "list" || !ok || len(models) != 2 || !models[0].Default {
		t.Fatalf("models response = %+v, want configured model records", modelsResp)
	}

	infoResp, err := registry.Dispatch(context.Background(), Request{Action: "model_info", Model: "default"})
	if err != nil {
		t.Fatalf("Dispatch(model_info) error = %v", err)
	}
	info, ok := infoResp["info"].(inference.ModelInfo)
	if infoResp["status"] != "ok" || infoResp["action"] != "model_info" || infoResp["model"] != "default" || !ok || info.Architecture != "gemma4" {
		t.Fatalf("model_info response = %+v, want model info payload", infoResp)
	}
	if backend.infoRequest.Model != "default" {
		t.Fatalf("info request = %+v, want requested model", backend.infoRequest)
	}

	capResp, err := registry.Dispatch(context.Background(), Request{Action: "capabilities", Model: "default"})
	if err != nil {
		t.Fatalf("Dispatch(capabilities) error = %v", err)
	}
	report, ok := capResp["report"].(inference.CapabilityReport)
	if capResp["status"] != "ok" || capResp["action"] != "capabilities" || !ok || !report.Supports(inference.CapabilityGenerate) {
		t.Fatalf("capabilities response = %+v, want capability report", capResp)
	}
	if backend.capabilitiesRequest.Model != "default" {
		t.Fatalf("capabilities request = %+v, want requested model", backend.capabilitiesRequest)
	}
}

func TestRegistry_RegisterIntrospectionBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterIntrospectionBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "introspection backend is nil") {
		t.Fatalf("RegisterIntrospectionBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_RegisterModelPackBackend_Good_DispatchesInspection(t *testing.T) {
	backend := &fakeModelPackBackend{
		result: &inference.ModelPackInspection{
			Path:      "/models/main",
			Format:    "safetensors",
			Supported: true,
			Model:     inference.ModelIdentity{Architecture: "gemma4_text", QuantBits: 6},
			Tokenizer: inference.TokenizerIdentity{Kind: "GemmaTokenizer"},
			Capabilities: []inference.Capability{
				inference.SupportedCapability(inference.CapabilityModelLoad, inference.CapabilityGroupRuntime),
			},
			Labels: map[string]string{"runtime": "metadata"},
		},
	}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterModelPackBackend(backend); err != nil {
		t.Fatalf("RegisterModelPackBackend() error = %v", err)
	}

	resp, err := registry.Dispatch(context.Background(), Request{Action: " inspect_model_pack ", Path: " /models/main "})

	if err != nil {
		t.Fatalf("Dispatch(inspect_model_pack) error = %v", err)
	}
	inspection, ok := resp["inspection"].(*inference.ModelPackInspection)
	if resp["status"] != "ok" ||
		resp["action"] != "inspect_model_pack" ||
		resp["path"] != "/models/main" ||
		!ok ||
		inspection.Model.Architecture != "gemma4_text" {
		t.Fatalf("inspect_model_pack response = %+v, want model-pack inspection", resp)
	}
	if backend.request.Path != "/models/main" {
		t.Fatalf("backend request = %+v, want trimmed path", backend.request)
	}
}

func TestRegistry_RegisterModelPackBackend_Bad_InputValidation(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterModelPackBackend(&fakeModelPackBackend{result: &inference.ModelPackInspection{Path: "/models/main"}}); err != nil {
		t.Fatalf("RegisterModelPackBackend() error = %v", err)
	}

	_, err := registry.Dispatch(context.Background(), Request{Action: "inspect_model_pack"})

	if err == nil || !strings.Contains(err.Error(), "model pack path") {
		t.Fatalf("Dispatch(inspect_model_pack empty) = %v, want path error", err)
	}
}

func TestRegistry_RegisterModelPackBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterModelPackBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "model pack backend is nil") {
		t.Fatalf("RegisterModelPackBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_RegisterModelProfileBackend_Good_DispatchesProfile(t *testing.T) {
	backend := &fakeModelProfileBackend{
		result: map[string]any{
			"architecture": "gemma4_text",
			"labels":       map[string]string{"engine_profile_reactive": "true"},
		},
	}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterModelProfileBackend(backend); err != nil {
		t.Fatalf("RegisterModelProfileBackend() error = %v", err)
	}

	resp, err := registry.Dispatch(context.Background(), Request{
		Action:  " model_profile ",
		Path:    " /models/main ",
		Model:   " main ",
		Backend: " rocm ",
		Labels:  map[string]string{"tenant": "test"},
	})

	if err != nil {
		t.Fatalf("Dispatch(model_profile) error = %v", err)
	}
	payload, ok := resp["profile"].(map[string]any)
	if resp["status"] != "ok" ||
		resp["action"] != "model_profile" ||
		resp["path"] != "/models/main" ||
		resp["model"] != "main" ||
		resp["backend"] != "rocm" ||
		!ok ||
		payload["architecture"] != "gemma4_text" {
		t.Fatalf("model_profile response = %+v, want reactive model profile payload", resp)
	}
	if backend.request.Path != "/models/main" ||
		backend.request.Model != "main" ||
		backend.request.Backend != "rocm" ||
		backend.request.Labels["tenant"] != "test" {
		t.Fatalf("backend request = %+v, want trimmed path/model/backend and labels", backend.request)
	}
}

func TestRegistry_RegisterModelProfileBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterModelProfileBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "model profile backend is nil") {
		t.Fatalf("RegisterModelProfileBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_RegisterModelRoutesBackend_Good_DispatchesRoutes(t *testing.T) {
	backend := &fakeModelRoutesBackend{
		result: map[string]any{
			"contract":     "rocm-model-route-plan-v1",
			"architecture": "gemma4_text",
			"labels":       map[string]string{"engine_route_plan_feature": "true"},
		},
	}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterModelRoutesBackend(backend); err != nil {
		t.Fatalf("RegisterModelRoutesBackend() error = %v", err)
	}

	resp, err := registry.Dispatch(context.Background(), Request{
		Action:  " model_routes ",
		Path:    " /models/main ",
		Model:   " main ",
		Backend: " rocm ",
		Labels:  map[string]string{"tenant": "test"},
	})

	if err != nil {
		t.Fatalf("Dispatch(model_routes) error = %v", err)
	}
	payload, ok := resp["routes"].(map[string]any)
	if resp["status"] != "ok" ||
		resp["action"] != "model_routes" ||
		resp["path"] != "/models/main" ||
		resp["model"] != "main" ||
		resp["backend"] != "rocm" ||
		!ok ||
		payload["contract"] != "rocm-model-route-plan-v1" {
		t.Fatalf("model_routes response = %+v, want reactive model route payload", resp)
	}
	if backend.request.Path != "/models/main" ||
		backend.request.Model != "main" ||
		backend.request.Backend != "rocm" ||
		backend.request.Labels["tenant"] != "test" {
		t.Fatalf("backend request = %+v, want trimmed path/model/backend and labels", backend.request)
	}
}

func TestRegistry_RegisterModelRoutesBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterModelRoutesBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "model routes backend is nil") {
		t.Fatalf("RegisterModelRoutesBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_RegisterModelRegistryBackend_Good_DispatchesRegistry(t *testing.T) {
	backend := &fakeModelRegistryBackend{
		result: map[string]any{
			"name":    "rocm-model-registry",
			"backend": "cuda",
			"labels":  map[string]string{"engine_profile_reactive": "true"},
		},
	}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterModelRegistryBackend(backend); err != nil {
		t.Fatalf("RegisterModelRegistryBackend() error = %v", err)
	}

	resp, err := registry.Dispatch(context.Background(), Request{
		Action:  " registry ",
		Backend: " cuda ",
		Labels:  map[string]string{"tenant": "test"},
	})

	if err != nil {
		t.Fatalf("Dispatch(registry) error = %v", err)
	}
	payload, ok := resp["registry"].(map[string]any)
	if resp["status"] != "ok" ||
		resp["action"] != "registry" ||
		resp["backend"] != "cuda" ||
		!ok ||
		payload["name"] != "rocm-model-registry" {
		t.Fatalf("registry response = %+v, want reactive model registry payload", resp)
	}
	if backend.request.Backend != "cuda" || backend.request.Labels["tenant"] != "test" {
		t.Fatalf("backend request = %+v, want trimmed backend and labels", backend.request)
	}
}

func TestRegistry_RegisterModelRegistryBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterModelRegistryBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "model registry backend is nil") {
		t.Fatalf("RegisterModelRegistryBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_RegisterTokenizerBackend_Good_DispatchesTokenizerRoutes(t *testing.T) {
	backend := &fakeTokenizerBackend{
		tokenizeResult:     TokenizeResult{Model: "default", Tokens: []int32{7, 8, 9}},
		detokenizeResult:   DetokenizeResult{Model: "default", Text: "hello"},
		chatTemplateResult: ChatTemplateResult{Model: "default", Text: "<user>hello</user>"},
	}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterTokenizerBackend(backend); err != nil {
		t.Fatalf("RegisterTokenizerBackend() error = %v", err)
	}

	tokenResp, err := registry.Dispatch(context.Background(), Request{
		Action: " tokenize ",
		Model:  "default",
		Text:   "hello",
	})
	if err != nil {
		t.Fatalf("Dispatch(tokenize) error = %v", err)
	}
	tokens, ok := tokenResp["tokens"].([]int32)
	if tokenResp["status"] != "ok" || tokenResp["action"] != "tokenize" || tokenResp["model"] != "default" || !ok || len(tokens) != 3 {
		t.Fatalf("tokenize response = %+v, want token IDs", tokenResp)
	}
	if backend.tokenizeRequest.Model != "default" || backend.tokenizeRequest.Text != "hello" {
		t.Fatalf("tokenize request = %+v, want model/text", backend.tokenizeRequest)
	}

	detokenResp, err := registry.Dispatch(context.Background(), Request{
		Action: "detokenize",
		Model:  "default",
		Tokens: []int32{7, 8, 9},
	})
	if err != nil {
		t.Fatalf("Dispatch(detokenize) error = %v", err)
	}
	if detokenResp["status"] != "ok" || detokenResp["action"] != "detokenize" || detokenResp["text"] != "hello" {
		t.Fatalf("detokenize response = %+v, want text", detokenResp)
	}
	if backend.detokenizeRequest.Model != "default" || len(backend.detokenizeRequest.Tokens) != 3 {
		t.Fatalf("detokenize request = %+v, want model/tokens", backend.detokenizeRequest)
	}

	templateResp, err := registry.Dispatch(context.Background(), Request{
		Action:   "chat_template",
		Model:    "default",
		Messages: []Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("Dispatch(chat_template) error = %v", err)
	}
	if templateResp["status"] != "ok" || templateResp["action"] != "chat_template" || templateResp["text"] != "<user>hello</user>" {
		t.Fatalf("chat_template response = %+v, want rendered template", templateResp)
	}
	if backend.chatTemplateRequest.Model != "default" || len(backend.chatTemplateRequest.Messages) != 1 || backend.chatTemplateRequest.Messages[0].Content != "hello" {
		t.Fatalf("chat template request = %+v, want model/messages", backend.chatTemplateRequest)
	}
}

func TestRegistry_RegisterTokenizerBackend_Bad_InputValidation(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterTokenizerBackend(&fakeTokenizerBackend{}); err != nil {
		t.Fatalf("RegisterTokenizerBackend() error = %v", err)
	}

	_, err := registry.Dispatch(context.Background(), Request{Action: "tokenize"})
	if err == nil || !strings.Contains(err.Error(), "tokenizer text") {
		t.Fatalf("Dispatch(tokenize empty) = %v, want text error", err)
	}
	_, err = registry.Dispatch(context.Background(), Request{Action: "detokenize"})
	if err == nil || !strings.Contains(err.Error(), "tokenizer tokens") {
		t.Fatalf("Dispatch(detokenize empty) = %v, want tokens error", err)
	}
	_, err = registry.Dispatch(context.Background(), Request{Action: "chat_template"})
	if err == nil || !strings.Contains(err.Error(), "chat template messages") {
		t.Fatalf("Dispatch(chat_template empty) = %v, want messages error", err)
	}
}

func TestRegistry_RegisterTokenizerBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterTokenizerBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "tokenizer backend is nil") {
		t.Fatalf("RegisterTokenizerBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_RegisterParserBackend_Good_DispatchesParserRoutes(t *testing.T) {
	backend := &fakeParserBackend{
		reasoningResult: inference.ReasoningParseResult{
			VisibleText: "answer",
			Reasoning:   []inference.ReasoningSegment{{Text: "plan"}},
			Labels:      map[string]string{"parser": "fake"},
		},
		toolsResult: inference.ToolParseResult{
			VisibleText: "done",
			Calls:       []inference.ToolCall{{Name: "search", ArgumentsJSON: `{"q":"rocm"}`}},
			Labels:      map[string]string{"parser": "fake"},
		},
	}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterParserBackend(backend); err != nil {
		t.Fatalf("RegisterParserBackend() error = %v", err)
	}

	reasonResp, err := registry.Dispatch(context.Background(), Request{
		Action: " parse_reasoning ",
		Model:  "default",
		Text:   "<think>plan</think>answer",
		Tokens: []int32{1, 2, 3},
	})
	if err != nil {
		t.Fatalf("Dispatch(parse_reasoning) error = %v", err)
	}
	reasoning, ok := reasonResp["reasoning"].([]inference.ReasoningSegment)
	if reasonResp["status"] != "ok" || reasonResp["action"] != "parse_reasoning" || reasonResp["visible_text"] != "answer" || !ok || len(reasoning) != 1 {
		t.Fatalf("parse_reasoning response = %+v, want reasoning parse", reasonResp)
	}
	if backend.reasoningRequest.Model != "default" || backend.reasoningRequest.Text == "" || len(backend.reasoningRequest.Tokens) != 3 {
		t.Fatalf("reasoning request = %+v, want model/text/tokens", backend.reasoningRequest)
	}

	toolsResp, err := registry.Dispatch(context.Background(), Request{
		Action:   "parse_tools",
		Model:    "default",
		Response: `<tool_call>{"name":"search","arguments":{"q":"rocm"}}</tool_call>`,
	})
	if err != nil {
		t.Fatalf("Dispatch(parse_tools) error = %v", err)
	}
	calls, ok := toolsResp["calls"].([]inference.ToolCall)
	if toolsResp["status"] != "ok" || toolsResp["action"] != "parse_tools" || toolsResp["visible_text"] != "done" || !ok || len(calls) != 1 {
		t.Fatalf("parse_tools response = %+v, want tool parse", toolsResp)
	}
	if backend.toolsRequest.Model != "default" || backend.toolsRequest.Text == "" {
		t.Fatalf("tools request = %+v, want model/text fallback", backend.toolsRequest)
	}
}

func TestRegistry_RegisterParserBackend_Bad_InputValidation(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterParserBackend(&fakeParserBackend{}); err != nil {
		t.Fatalf("RegisterParserBackend() error = %v", err)
	}

	_, err := registry.Dispatch(context.Background(), Request{Action: "parse_reasoning"})
	if err == nil || !strings.Contains(err.Error(), "parser text") {
		t.Fatalf("Dispatch(parse_reasoning empty) = %v, want text/tokens error", err)
	}
	_, err = registry.Dispatch(context.Background(), Request{Action: "parse_tools"})
	if err == nil || !strings.Contains(err.Error(), "parser text") {
		t.Fatalf("Dispatch(parse_tools empty) = %v, want text/tokens error", err)
	}
}

func TestRegistry_RegisterParserBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterParserBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "parser backend is nil") {
		t.Fatalf("RegisterParserBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_RegisterGenerateBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterGenerateBackend(nil)

	if err == nil || !strings.Contains(err.Error(), "generate backend is nil") {
		t.Fatalf("RegisterGenerateBackend(nil) = %v, want nil backend error", err)
	}
}

func TestRegistry_RegisterGenerateBackend_Ugly_TextFallback(t *testing.T) {
	backend := &fakeGenerateBackend{result: GenerateResult{Text: "ok"}}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterGenerateBackend(backend); err != nil {
		t.Fatalf("RegisterGenerateBackend() error = %v", err)
	}

	_, err := registry.Dispatch(context.Background(), Request{Action: "generate", Text: "fallback"})

	if err != nil {
		t.Fatalf("Dispatch() error = %v", err)
	}
	if backend.request.Prompt != "fallback" {
		t.Fatalf("backend prompt = %q, want fallback", backend.request.Prompt)
	}
}

func TestRegistry_Info_Good_StableActions(t *testing.T) {
	registry := NewRegistry("", "test")

	resp, err := registry.Dispatch(context.Background(), Request{Action: "info"})

	if err != nil {
		t.Fatalf("Dispatch(info) error = %v", err)
	}
	if resp["name"] != DaemonName || resp["version"] != "test" {
		t.Fatalf("info response = %+v, want daemon identity", resp)
	}
	actions, ok := resp["actions"].([]string)
	if !ok {
		t.Fatalf("actions = %#v, want []string", resp["actions"])
	}
	want := []string{"embed", "score", "generate", "schedule", "rerank", "cache_stats", "cache_warm", "cache_clear", "cache_entries", "cancel", "health", "models", "model_info", "capabilities", "engine_features", "inspect_model_pack", "model_profile", "model_routes", "registry", "tokenize", "detokenize", "chat_template", "parse_reasoning", "parse_tools", "info"}
	if len(actions) != len(want) {
		t.Fatalf("actions = %+v, want %+v", actions, want)
	}
	for i := range want {
		if actions[i] != want[i] {
			t.Fatalf("actions = %+v, want %+v", actions, want)
		}
	}
	routes, ok := resp["routes"].([]RouteInfo)
	if !ok {
		t.Fatalf("routes = %#v, want []RouteInfo", resp["routes"])
	}
	if len(routes) != len(want) {
		t.Fatalf("routes = %+v, want one route per action", routes)
	}
	statuses := map[string]string{}
	backends := map[string]string{}
	for _, route := range routes {
		statuses[route.Action] = route.Status
		backends[route.Action] = route.Backend
	}
	if statuses["embed"] != "stub" ||
		statuses["score"] != "heuristic" ||
		backends["score"] != "rocm/score" ||
		statuses["generate"] != "stub" ||
		statuses["schedule"] != "stub" ||
		statuses["rerank"] != "stub" ||
		statuses["cache_stats"] != "stub" ||
		statuses["cache_warm"] != "stub" ||
		statuses["cache_clear"] != "stub" ||
		statuses["cache_entries"] != "stub" ||
		statuses["cancel"] != "stub" ||
		statuses["health"] != "system" ||
		backends["health"] != "daemon" ||
		statuses["models"] != "stub" ||
		statuses["model_info"] != "stub" ||
		statuses["capabilities"] != "stub" ||
		statuses["engine_features"] != "stub" ||
		statuses["inspect_model_pack"] != "stub" ||
		statuses["model_profile"] != "stub" ||
		statuses["model_routes"] != "stub" ||
		statuses["registry"] != "stub" ||
		statuses["tokenize"] != "stub" ||
		statuses["detokenize"] != "stub" ||
		statuses["chat_template"] != "stub" ||
		statuses["parse_reasoning"] != "stub" ||
		statuses["parse_tools"] != "stub" ||
		statuses["info"] != "system" {
		t.Fatalf("routes = %+v, want default route status metadata", routes)
	}

	health, err := registry.Dispatch(context.Background(), Request{Action: "health"})
	if err != nil {
		t.Fatalf("Dispatch(health) error = %v", err)
	}
	if health["status"] != "ok" || health["action"] != "health" || health["name"] != DaemonName {
		t.Fatalf("health = %+v, want daemon health response", health)
	}
}

func TestRegistry_Info_Good_RouteMetadataRefreshesOnBackendRegistration(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")
	if _, err := registry.Dispatch(context.Background(), Request{Action: "info"}); err != nil {
		t.Fatalf("initial info: %v", err)
	}
	if err := registry.RegisterGenerateBackend(&fakeGenerateBackend{}); err != nil {
		t.Fatalf("RegisterGenerateBackend() error = %v", err)
	}
	if err := registry.RegisterScheduleBackend(&fakeScheduleBackend{}); err != nil {
		t.Fatalf("RegisterScheduleBackend() error = %v", err)
	}
	if err := registry.RegisterIntrospectionBackend(&fakeIntrospectionBackend{}); err != nil {
		t.Fatalf("RegisterIntrospectionBackend() error = %v", err)
	}
	if err := registry.RegisterEmbedBackend(&fakeEmbedBackend{result: &inference.EmbeddingResult{Vectors: [][]float32{{1}}}}); err != nil {
		t.Fatalf("RegisterEmbedBackend() error = %v", err)
	}
	if err := registry.RegisterRerankBackend(&fakeRerankBackend{result: &inference.RerankResult{}}); err != nil {
		t.Fatalf("RegisterRerankBackend() error = %v", err)
	}
	if err := registry.RegisterCacheBackend(&fakeCacheBackend{}); err != nil {
		t.Fatalf("RegisterCacheBackend() error = %v", err)
	}
	if err := registry.RegisterCacheEntryBackend(&fakeCacheBackend{}); err != nil {
		t.Fatalf("RegisterCacheEntryBackend() error = %v", err)
	}
	if err := registry.RegisterCancelBackend(&fakeCancelBackend{}); err != nil {
		t.Fatalf("RegisterCancelBackend() error = %v", err)
	}
	if err := registry.RegisterEngineFeaturesBackend(&fakeEngineFeaturesBackend{result: map[string]any{"architecture": "gemma4"}}); err != nil {
		t.Fatalf("RegisterEngineFeaturesBackend() error = %v", err)
	}
	if err := registry.RegisterModelPackBackend(&fakeModelPackBackend{}); err != nil {
		t.Fatalf("RegisterModelPackBackend() error = %v", err)
	}
	if err := registry.RegisterModelProfileBackend(&fakeModelProfileBackend{result: map[string]any{"architecture": "gemma4"}}); err != nil {
		t.Fatalf("RegisterModelProfileBackend() error = %v", err)
	}
	if err := registry.RegisterModelRoutesBackend(&fakeModelRoutesBackend{result: map[string]any{"contract": "rocm-model-route-plan-v1"}}); err != nil {
		t.Fatalf("RegisterModelRoutesBackend() error = %v", err)
	}
	if err := registry.RegisterModelRegistryBackend(&fakeModelRegistryBackend{result: map[string]any{"name": "rocm"}}); err != nil {
		t.Fatalf("RegisterModelRegistryBackend() error = %v", err)
	}
	if err := registry.RegisterTokenizerBackend(&fakeTokenizerBackend{}); err != nil {
		t.Fatalf("RegisterTokenizerBackend() error = %v", err)
	}
	if err := registry.RegisterParserBackend(&fakeParserBackend{}); err != nil {
		t.Fatalf("RegisterParserBackend() error = %v", err)
	}

	resp, err := registry.Dispatch(context.Background(), Request{Action: "info"})
	if err != nil {
		t.Fatalf("updated info: %v", err)
	}
	routes, ok := resp["routes"].([]RouteInfo)
	if !ok {
		t.Fatalf("routes = %#v, want []RouteInfo", resp["routes"])
	}
	statuses := map[string]string{}
	backends := map[string]string{}
	for _, route := range routes {
		statuses[route.Action] = route.Status
		backends[route.Action] = route.Backend
	}
	if statuses["embed"] != "native_model" ||
		backends["embed"] != "inference.EmbeddingModel" ||
		statuses["generate"] != "native_model" ||
		backends["generate"] != "daemon.GenerateBackend" ||
		statuses["schedule"] != "native_model" ||
		backends["schedule"] != "inference.SchedulerModel" ||
		statuses["rerank"] != "native_model" ||
		backends["rerank"] != "inference.RerankModel" ||
		statuses["cache_stats"] != "native_model" ||
		statuses["cache_warm"] != "native_model" ||
		statuses["cache_clear"] != "native_model" ||
		statuses["cache_entries"] != "native_model" ||
		backends["cache_stats"] != "inference.CacheService" ||
		backends["cache_warm"] != "inference.CacheService" ||
		backends["cache_clear"] != "inference.CacheService" ||
		backends["cache_entries"] != "daemon.CacheEntryBackend" ||
		statuses["cancel"] != "native_model" ||
		backends["cancel"] != "inference.CancellableModel" ||
		statuses["models"] != "native_model" ||
		statuses["model_info"] != "native_model" ||
		statuses["capabilities"] != "native_model" ||
		backends["models"] != "daemon.IntrospectionBackend" ||
		backends["model_info"] != "daemon.IntrospectionBackend" ||
		backends["capabilities"] != "daemon.IntrospectionBackend" ||
		statuses["engine_features"] != "native_metadata" ||
		backends["engine_features"] != "daemon.EngineFeaturesBackend" ||
		statuses["inspect_model_pack"] != "native_metadata" ||
		backends["inspect_model_pack"] != "inference.ModelPackInspector" ||
		statuses["model_profile"] != "native_metadata" ||
		backends["model_profile"] != "daemon.ModelProfileBackend" ||
		statuses["model_routes"] != "native_metadata" ||
		backends["model_routes"] != "daemon.ModelRoutesBackend" ||
		statuses["registry"] != "native_metadata" ||
		backends["registry"] != "daemon.ModelRegistryBackend" ||
		statuses["tokenize"] != "native_model" ||
		statuses["detokenize"] != "native_model" ||
		statuses["chat_template"] != "native_model" ||
		backends["tokenize"] != "inference.TokenizerModel" ||
		backends["detokenize"] != "inference.TokenizerModel" ||
		backends["chat_template"] != "inference.TokenizerModel" ||
		statuses["parse_reasoning"] != "native_model" ||
		statuses["parse_tools"] != "native_model" ||
		backends["parse_reasoning"] != "inference.ReasoningParser" ||
		backends["parse_tools"] != "inference.ToolParser" {
		t.Fatalf("routes = %+v, want refreshed native route metadata", routes)
	}
}
