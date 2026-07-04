// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"context"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/openai"
)

func TestOpenAI_NewOpenAIResolver_Good_UsesROCmBackend(t *testing.T) {
	resolver := NewOpenAIResolver("/models/qwen3.gguf")
	if resolver == nil {
		t.Fatal("NewOpenAIResolver() returned nil")
	}
	if resolver.BackendName != "rocm" {
		t.Fatalf("BackendName = %q, want rocm", resolver.BackendName)
	}
	if resolver.ModelPath != "/models/qwen3.gguf" {
		t.Fatalf("ModelPath = %q", resolver.ModelPath)
	}
}

func TestOpenAI_NewOpenAIHandler_Good_ReturnsHTTPHandler(t *testing.T) {
	handler := NewOpenAIHandler("/models/qwen3.gguf")
	if handler == nil {
		t.Fatal("NewOpenAIHandler() returned nil")
	}
}

func TestOpenAI_NewOpenAIResponsesHandler_Good_NonStreaming(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	handler := NewOpenAIResponsesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{"model":"qwen","input":[{"role":"user","content":"hello"}]}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"object":"response"`) || !strings.Contains(rec.Body.String(), "ok") {
		t.Fatalf("body = %s, want response payload", rec.Body.String())
	}
}

func TestOpenAI_NewOpenAIResponsesHandler_Good_Streaming(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "hel"}, {Text: "lo"}}}
	handler := NewOpenAIResponsesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{"model":"qwen","stream":true,"input":[{"role":"user","content":"hello"}]}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	body := rec.Body.String()
	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status = %d content-type=%q body=%s, want SSE stream", rec.Code, rec.Header().Get("Content-Type"), body)
	}
	for _, want := range []string{
		`"type":"response.created"`,
		`"type":"response.output_text.delta","delta":"hel"`,
		`"type":"response.output_text.delta","delta":"lo"`,
		`"type":"response.completed"`,
		`"text":"hello"`,
		"data: [DONE]\n\n",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("stream body missing %q:\n%s", want, body)
		}
	}
}

func TestOpenAI_NewOpenAIResponsesHandler_Bad_RejectsBlankInput(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	handler := NewOpenAIResponsesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{"model":"qwen","instructions":"   ","input":[{"role":"user","content":"   "}]}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "input or instructions are required") {
		t.Fatalf("status = %d body=%s, want blank input rejected", rec.Code, rec.Body.String())
	}
}

func TestOpenAI_NewOpenAIResponsesHandler_Ugly_TransportAndResolverFailures(t *testing.T) {
	nilResolverRec := httptest.NewRecorder()
	NewOpenAIResponsesHandler(nil).ServeHTTP(nilResolverRec, httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{}`)))
	if nilResolverRec.Code != http.StatusServiceUnavailable || !strings.Contains(nilResolverRec.Body.String(), "not configured") {
		t.Fatalf("nil resolver status = %d body=%s, want service unavailable", nilResolverRec.Code, nilResolverRec.Body.String())
	}

	nilReqRec := httptest.NewRecorder()
	NewOpenAIResponsesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{})).ServeHTTP(nilReqRec, nil)
	if nilReqRec.Code != http.StatusBadRequest || !strings.Contains(nilReqRec.Body.String(), "request body is nil") {
		t.Fatalf("nil request status = %d body=%s, want bad request", nilReqRec.Code, nilReqRec.Body.String())
	}

	methodRec := httptest.NewRecorder()
	NewOpenAIResponsesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{})).ServeHTTP(methodRec, httptest.NewRequest(http.MethodGet, openaicompat.DefaultResponsesPath, nil))
	if methodRec.Code != http.StatusMethodNotAllowed || methodRec.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("wrong method status = %d allow=%q body=%s, want 405 with Allow", methodRec.Code, methodRec.Header().Get("Allow"), methodRec.Body.String())
	}

	malformedRec := httptest.NewRecorder()
	NewOpenAIResponsesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{})).ServeHTTP(malformedRec, httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{"model":`)))
	if malformedRec.Code != http.StatusBadRequest || !strings.Contains(malformedRec.Body.String(), "invalid request body") {
		t.Fatalf("malformed body status = %d body=%s, want bad request", malformedRec.Code, malformedRec.Body.String())
	}

	resolveRec := httptest.NewRecorder()
	NewOpenAIResponsesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{})).ServeHTTP(resolveRec, httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{"model":"missing","input":[{"role":"user","content":"hello"}]}`)))
	if resolveRec.Code != http.StatusNotFound || !strings.Contains(resolveRec.Body.String(), "missing") {
		t.Fatalf("missing model status = %d body=%s, want resolver failure", resolveRec.Code, resolveRec.Body.String())
	}
}

func TestOpenAI_NewOpenAIServiceMux_Good_MountsServiceEndpoints(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	mux := NewOpenAIServiceMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodGet, openaicompat.DefaultCapabilitiesPath+"?model=qwen", nil)
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"available"`) {
		t.Fatalf("body = %s, want capability report", rec.Body.String())
	}
}

func TestOpenAI_NewOpenAIServiceMux_Good_DelegatesBlankChatMessagesToModel(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	mux := NewOpenAIServiceMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultChatCompletionsPath, strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":"   "}]}`))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "ok") {
		t.Fatalf("status = %d body=%s, want canonical OpenAI handler to delegate blank content", rec.Code, rec.Body.String())
	}
}

func TestOpenAI_NewOpenAIServiceMux_Bad_GenericModelWithBlankEmbeddingInputReportsUnsupported(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	mux := NewOpenAIServiceMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultEmbeddingsPath, strings.NewReader(`{"model":"qwen","input":["hello","   "]}`))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotImplemented || !strings.Contains(rec.Body.String(), "does not support embeddings") {
		t.Fatalf("status = %d body=%s, want canonical unsupported-embedding response", rec.Code, rec.Body.String())
	}
}

func TestOpenAI_NewOpenAIServiceMux_Bad_GenericModelWithBlankRerankDocumentReportsUnsupported(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	mux := NewOpenAIServiceMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultRerankPath, strings.NewReader(`{"model":"qwen","query":"core","documents":["doc","   "]}`))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotImplemented || !strings.Contains(rec.Body.String(), "does not support rerank") {
		t.Fatalf("status = %d body=%s, want canonical unsupported-rerank response", rec.Code, rec.Body.String())
	}
}

func TestOpenAI_NewOpenAIServiceMux_Bad_GenericModelWithoutEmbeddingsAndRerank(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	mux := NewOpenAIServiceMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))

	embeddingReq := httptest.NewRequest(http.MethodPost, openaicompat.DefaultEmbeddingsPath, strings.NewReader(`{"model":"qwen","input":"hello"}`))
	embeddingRec := httptest.NewRecorder()
	mux.ServeHTTP(embeddingRec, embeddingReq)
	if embeddingRec.Code != http.StatusNotImplemented || !strings.Contains(embeddingRec.Body.String(), "does not support embeddings") {
		t.Fatalf("embedding status = %d body=%s, want planned not-implemented response", embeddingRec.Code, embeddingRec.Body.String())
	}

	rerankReq := httptest.NewRequest(http.MethodPost, openaicompat.DefaultRerankPath, strings.NewReader(`{"model":"qwen","query":"core","documents":["a"]}`))
	rerankRec := httptest.NewRecorder()
	mux.ServeHTTP(rerankRec, rerankReq)
	if rerankRec.Code != http.StatusNotImplemented || !strings.Contains(rerankRec.Body.String(), "does not support rerank") {
		t.Fatalf("rerank status = %d body=%s, want planned not-implemented response", rerankRec.Code, rerankRec.Body.String())
	}
}

type openAITestModel struct {
	tokens []inference.Token
	mu     sync.Mutex
	metric inference.GenerateMetrics
	info   inference.ModelInfo
	err    error
}

func (model *openAITestModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return model.stream(ctx, opts...)
}

func (model *openAITestModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return model.stream(ctx, opts...)
}

func (model *openAITestModel) stream(ctx context.Context, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)
	return func(yield func(inference.Token) bool) {
		limit := len(model.tokens)
		if cfg.MaxTokens > 0 && cfg.MaxTokens < limit {
			limit = cfg.MaxTokens
		}
		tokens := append([]inference.Token(nil), model.tokens[:limit]...)
		for i := range tokens {
			if ctx != nil {
				select {
				case <-ctx.Done():
					model.mu.Lock()
					model.err = ctx.Err()
					model.mu.Unlock()
					return
				default:
				}
			}
			if !yield(tokens[i]) {
				return
			}
		}
		model.mu.Lock()
		model.metric = inference.GenerateMetrics{GeneratedTokens: len(tokens)}
		model.err = nil
		model.mu.Unlock()
	}
}

func (model *openAITestModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (model *openAITestModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (model *openAITestModel) ModelType() string { return "test" }
func (model *openAITestModel) Info() inference.ModelInfo {
	if model.info.Architecture != "" {
		return model.info
	}
	return inference.ModelInfo{Architecture: "test"}
}
func (model *openAITestModel) Metrics() inference.GenerateMetrics {
	model.mu.Lock()
	defer model.mu.Unlock()
	return model.metric
}
func (model *openAITestModel) Err() error {
	model.mu.Lock()
	defer model.mu.Unlock()
	return model.err
}
func (model *openAITestModel) Close() error { return nil }
