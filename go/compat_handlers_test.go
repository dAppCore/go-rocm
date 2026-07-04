// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/anthropic"
	"dappco.re/go/inference/ollama"
	openaicompat "dappco.re/go/inference/openai"
)

func TestCompatHandlers_Good_AnthropicMessages(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	handler := NewAnthropicMessagesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, anthropic.DefaultMessagesPath, strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"type":"message"`) || !strings.Contains(rec.Body.String(), "ok") {
		t.Fatalf("body = %s, want Anthropic message response", rec.Body.String())
	}
}

func TestCompatHandlers_Good_AnthropicMessagesStreaming(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "hel"}, {Text: "lo"}}}
	handler := NewAnthropicMessagesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, anthropic.DefaultMessagesPath, strings.NewReader(`{"model":"qwen","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	body := rec.Body.String()
	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status = %d content-type=%q body=%s, want SSE stream", rec.Code, rec.Header().Get("Content-Type"), body)
	}
	for _, want := range []string{
		"event: message_start\n",
		"event: content_block_start\n",
		`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hel"}}`,
		`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}`,
		"event: content_block_stop\n",
		`event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}`,
		"event: message_stop\n",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("stream body missing %q:\n%s", want, body)
		}
	}
}

func TestCompatHandlers_Bad_AnthropicRejectsEmptyMessages(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	handler := NewAnthropicMessagesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, anthropic.DefaultMessagesPath, strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":[{"type":"text","text":"   "}]}]}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "messages or system are required") {
		t.Fatalf("status = %d body=%s, want empty messages rejected", rec.Code, rec.Body.String())
	}
}

func TestCompatHandlers_Good_OllamaChatAndGenerate(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	mux := NewOllamaHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	chatReq := httptest.NewRequest(http.MethodPost, ollama.DefaultChatPath, strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":"hello"}]}`))
	chatRec := httptest.NewRecorder()

	mux.ServeHTTP(chatRec, chatReq)

	if chatRec.Code != http.StatusOK {
		t.Fatalf("chat status = %d body=%s", chatRec.Code, chatRec.Body.String())
	}
	if !strings.Contains(chatRec.Body.String(), `"done":true`) || !strings.Contains(chatRec.Body.String(), "ok") {
		t.Fatalf("chat body = %s, want Ollama chat response", chatRec.Body.String())
	}

	genReq := httptest.NewRequest(http.MethodPost, ollama.DefaultGeneratePath, strings.NewReader(`{"model":"qwen","prompt":"hello"}`))
	genRec := httptest.NewRecorder()
	mux.ServeHTTP(genRec, genReq)

	if genRec.Code != http.StatusOK {
		t.Fatalf("generate status = %d body=%s", genRec.Code, genRec.Body.String())
	}
	if !strings.Contains(genRec.Body.String(), `"done":true`) || !strings.Contains(genRec.Body.String(), "ok") {
		t.Fatalf("generate body = %s, want Ollama generate response", genRec.Body.String())
	}
}

func TestCompatHandlers_Good_OllamaTagsAndShow(t *testing.T) {
	identity := inference.ModelIdentity{
		Path:          "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		Architecture:  "gemma4_text",
		VocabSize:     256000,
		HiddenSize:    2304,
		NumLayers:     26,
		QuantBits:     6,
		QuantGroup:    64,
		ContextLength: 131072,
		Labels: map[string]string{
			"gemma4_size":       "E4B",
			"gemma4_quant_mode": "q6",
		},
	}
	profile, ok := ResolveROCmModelProfile(identity.Path, identity)
	if !ok {
		t.Fatal("ResolveROCmModelProfile(gemma4) ok = false")
	}
	model := &compatReactiveShowModel{
		openAITestModel: openAITestModel{
			tokens: []inference.Token{{Text: "ok"}},
			info: inference.ModelInfo{
				Architecture: identity.Architecture,
				VocabSize:    identity.VocabSize,
				HiddenSize:   identity.HiddenSize,
				NumLayers:    identity.NumLayers,
				QuantBits:    identity.QuantBits,
				QuantGroup:   identity.QuantGroup,
			},
		},
		identity: identity,
		profile:  profile,
	}
	mux := NewOllamaHandler(&compatOllamaNamedResolver{
		names:  []string{"gemma4:e2b-q6", "gemma4:e2b-q6", "qwen"},
		models: map[string]inference.TextModel{"gemma4:e2b-q6": model},
	})
	tagsReq := httptest.NewRequest(http.MethodGet, ollama.DefaultTagsPath, nil)
	tagsRec := httptest.NewRecorder()

	mux.ServeHTTP(tagsRec, tagsReq)

	if tagsRec.Code != http.StatusOK || !strings.Contains(tagsRec.Body.String(), `"name":"gemma4:e2b-q6"`) || strings.Count(tagsRec.Body.String(), "gemma4:e2b-q6") != 2 {
		t.Fatalf("tags status = %d body=%s, want deduplicated Ollama model tag", tagsRec.Code, tagsRec.Body.String())
	}

	showReq := httptest.NewRequest(http.MethodPost, ollama.DefaultShowPath, strings.NewReader(`{"model":"gemma4:e2b-q6"}`))
	showRec := httptest.NewRecorder()
	mux.ServeHTTP(showRec, showReq)

	body := showRec.Body.String()
	if showRec.Code != http.StatusOK ||
		!strings.Contains(body, `"architecture":"gemma4_text"`) ||
		!strings.Contains(body, `"family":"gemma4"`) ||
		!strings.Contains(body, `"model_path":"`+identity.Path+`"`) ||
		!strings.Contains(body, `"context_length":"131072"`) ||
		!strings.Contains(body, `"quantization":"6-bit"`) ||
		!strings.Contains(body, `"quant_type":"q6"`) ||
		!strings.Contains(body, `"num_layers":"26"`) ||
		!strings.Contains(body, `"gemma4_size":"E4B"`) ||
		!strings.Contains(body, `"engine_profile":"gemma4"`) ||
		!strings.Contains(body, `"engine_registry":"`+rocmModelRegistryName+`"`) ||
		!strings.Contains(body, `"chat_template":"gemma4_hf_turn"`) ||
		!strings.Contains(body, `"engine_feature_capabilities":"generate,chat.template,reasoning.parse,tool.parse"`) ||
		!strings.Contains(body, `quant_bits 6`) ||
		!strings.Contains(body, `context_length 131072`) {
		t.Fatalf("show status = %d body=%s, want reactive ROCm model details", showRec.Code, body)
	}
}

func TestCompatHandlers_Good_OllamaStreaming(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "hel"}, {Text: "lo"}}}
	mux := NewOllamaHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	chatReq := httptest.NewRequest(http.MethodPost, ollama.DefaultChatPath, strings.NewReader(`{"model":"qwen","stream":true,"messages":[{"role":"user","content":"hello"}]}`))
	chatRec := httptest.NewRecorder()

	mux.ServeHTTP(chatRec, chatReq)

	chatBody := chatRec.Body.String()
	if chatRec.Code != http.StatusOK || chatRec.Header().Get("Content-Type") != "application/x-ndjson" {
		t.Fatalf("chat status = %d content-type=%q body=%s, want NDJSON stream", chatRec.Code, chatRec.Header().Get("Content-Type"), chatBody)
	}
	for _, want := range []string{
		`"message":{"role":"assistant","content":"hel"}`,
		`"message":{"role":"assistant","content":"lo"}`,
		`"done":true`,
		`"eval_count":2`,
	} {
		if !strings.Contains(chatBody, want) {
			t.Fatalf("chat stream body missing %q:\n%s", want, chatBody)
		}
	}

	generateReq := httptest.NewRequest(http.MethodPost, ollama.DefaultGeneratePath, strings.NewReader(`{"model":"qwen","stream":true,"prompt":"hello"}`))
	generateRec := httptest.NewRecorder()
	mux.ServeHTTP(generateRec, generateReq)

	generateBody := generateRec.Body.String()
	if generateRec.Code != http.StatusOK || generateRec.Header().Get("Content-Type") != "application/x-ndjson" {
		t.Fatalf("generate status = %d content-type=%q body=%s, want NDJSON stream", generateRec.Code, generateRec.Header().Get("Content-Type"), generateBody)
	}
	for _, want := range []string{
		`"response":"hel"`,
		`"response":"lo"`,
		`"done":true`,
		`"eval_count":2`,
	} {
		if !strings.Contains(generateBody, want) {
			t.Fatalf("generate stream body missing %q:\n%s", want, generateBody)
		}
	}
}

func TestCompatHandlers_Bad_OllamaRejectsEmptyChatMessages(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	mux := NewOllamaHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, ollama.DefaultChatPath, strings.NewReader(`{"model":"qwen"}`))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "messages are required") {
		t.Fatalf("status = %d body=%s, want empty messages rejected", rec.Code, rec.Body.String())
	}
}

func TestCompatHandlers_Bad_OllamaRejectsBlankChatMessages(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	mux := NewOllamaHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, ollama.DefaultChatPath, strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":"   "}]}`))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "messages are required") {
		t.Fatalf("status = %d body=%s, want blank messages rejected", rec.Code, rec.Body.String())
	}
}

func TestCompatHandlers_Bad_OllamaRejectsEmptyGeneratePrompt(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	mux := NewOllamaHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, ollama.DefaultGeneratePath, strings.NewReader(`{"model":"qwen"}`))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "prompt is required") {
		t.Fatalf("status = %d body=%s, want empty prompt rejected", rec.Code, rec.Body.String())
	}
}

func TestCompatHandlers_Bad_OllamaShowRejectsEmptyModel(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	mux := NewOllamaHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, ollama.DefaultShowPath, strings.NewReader(`{}`))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "model is required") {
		t.Fatalf("show status = %d body=%s, want empty model rejected", rec.Code, rec.Body.String())
	}
}

func TestCompatHandlers_Ugly_TransportAndResolverFailures(t *testing.T) {
	anthropicRec := httptest.NewRecorder()
	NewAnthropicMessagesHandler(nil).ServeHTTP(anthropicRec, httptest.NewRequest(http.MethodPost, anthropic.DefaultMessagesPath, strings.NewReader(`{}`)))
	if anthropicRec.Code != http.StatusServiceUnavailable || !strings.Contains(anthropicRec.Body.String(), "not configured") {
		t.Fatalf("Anthropic nil resolver status = %d body=%s, want service unavailable", anthropicRec.Code, anthropicRec.Body.String())
	}

	methodRec := httptest.NewRecorder()
	NewAnthropicMessagesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{})).ServeHTTP(methodRec, httptest.NewRequest(http.MethodGet, anthropic.DefaultMessagesPath, nil))
	if methodRec.Code != http.StatusMethodNotAllowed || methodRec.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("Anthropic wrong method status = %d allow=%q body=%s, want 405 with Allow", methodRec.Code, methodRec.Header().Get("Allow"), methodRec.Body.String())
	}

	nilReqRec := httptest.NewRecorder()
	NewAnthropicMessagesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{})).ServeHTTP(nilReqRec, nil)
	if nilReqRec.Code != http.StatusBadRequest || !strings.Contains(nilReqRec.Body.String(), "request is nil") {
		t.Fatalf("Anthropic nil request status = %d body=%s, want bad request", nilReqRec.Code, nilReqRec.Body.String())
	}

	ollamaRec := httptest.NewRecorder()
	NewOllamaHandler(nil).ServeHTTP(ollamaRec, httptest.NewRequest(http.MethodPost, ollama.DefaultGeneratePath, strings.NewReader(`{}`)))
	if ollamaRec.Code != http.StatusServiceUnavailable || !strings.Contains(ollamaRec.Body.String(), "not configured") {
		t.Fatalf("Ollama nil resolver status = %d body=%s, want service unavailable", ollamaRec.Code, ollamaRec.Body.String())
	}

	resolveRec := httptest.NewRecorder()
	NewOllamaHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{})).ServeHTTP(resolveRec, httptest.NewRequest(http.MethodPost, ollama.DefaultGeneratePath, strings.NewReader(`{"model":"missing","prompt":"hello"}`)))
	if resolveRec.Code != http.StatusNotFound || !strings.Contains(resolveRec.Body.String(), "missing") {
		t.Fatalf("Ollama missing model status = %d body=%s, want resolver failure", resolveRec.Code, resolveRec.Body.String())
	}

	tagsMethodRec := httptest.NewRecorder()
	NewOllamaHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{})).ServeHTTP(tagsMethodRec, httptest.NewRequest(http.MethodPost, ollama.DefaultTagsPath, nil))
	if tagsMethodRec.Code != http.StatusMethodNotAllowed || tagsMethodRec.Header().Get("Allow") != http.MethodGet {
		t.Fatalf("Ollama tags wrong method status = %d allow=%q body=%s, want 405 with Allow", tagsMethodRec.Code, tagsMethodRec.Header().Get("Allow"), tagsMethodRec.Body.String())
	}
}

func BenchmarkCompatHandlers_CollectROCmWireTokenText_512Tokens(b *testing.B) {
	tokens := make([]inference.Token, 512)
	for i := range tokens {
		tokens[i] = inference.Token{Text: "word "}
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		compatHandlerTextSink = collectROCmWireTokenText(func(yield func(inference.Token) bool) {
			for _, token := range tokens {
				if !yield(token) {
					return
				}
			}
		})
	}
}

func BenchmarkCompatHandlers_WriteAnthropicSSEDelta(b *testing.B) {
	payload := anthropic.AppendContentBlockDeltaEvent(nil, 0, "word ")
	writer := rocmDiscardResponseWriter{}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		writeROCmAnthropicSSEEvent(writer, "content_block_delta", payload)
	}
}

func BenchmarkCompatHandlers_WriteAnthropicSSEDeltaWithPayloadReuse(b *testing.B) {
	writer := rocmDiscardResponseWriter{}
	payload := make([]byte, 0, 256)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		payload = anthropic.AppendContentBlockDeltaEvent(payload[:0], 0, "word ")
		writeROCmAnthropicSSEEvent(writer, "content_block_delta", payload)
	}
}

var compatHandlerTextSink string

type rocmDiscardResponseWriter struct{}

func (rocmDiscardResponseWriter) Header() http.Header { return http.Header{} }
func (rocmDiscardResponseWriter) Write(p []byte) (int, error) {
	return len(p), nil
}
func (rocmDiscardResponseWriter) WriteHeader(int) {}

type compatReactiveShowModel struct {
	openAITestModel
	identity inference.ModelIdentity
	profile  ROCmModelProfile
}

func (model *compatReactiveShowModel) ModelIdentity() inference.ModelIdentity {
	return model.identity
}

func (model *compatReactiveShowModel) ModelProfile() ROCmModelProfile {
	return model.profile
}

type compatOllamaNamedResolver struct {
	names  []string
	models map[string]inference.TextModel
}

func (resolver *compatOllamaNamedResolver) OllamaModelNames(context.Context) ([]string, error) {
	return append([]string(nil), resolver.names...), nil
}

func (resolver *compatOllamaNamedResolver) ResolveModel(_ context.Context, name string) (inference.TextModel, error) {
	model, ok := resolver.models[name]
	if !ok {
		return nil, core.E("compatOllamaNamedResolver", "model not found", nil)
	}
	return model, nil
}
