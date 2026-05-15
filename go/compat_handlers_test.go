// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

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

func TestCompatHandlers_Bad_AnthropicRejectsStreaming(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	handler := NewAnthropicMessagesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, anthropic.DefaultMessagesPath, strings.NewReader(`{"model":"qwen","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
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

func TestCompatHandlers_Good_OllamaGenerateAppliesStopSequences(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "visible "}, {Text: "EN"}, {Text: "D hidden"}}}
	mux := NewOllamaHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, ollama.DefaultGeneratePath, strings.NewReader(`{"model":"qwen","prompt":"hello","options":{"stop":["END"]}}`))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"response":"visible "`) || strings.Contains(rec.Body.String(), "hidden") {
		t.Fatalf("body = %s, want response truncated before stop sequence", rec.Body.String())
	}
}

func TestCompatHandlers_Bad_OllamaRejectsStreaming(t *testing.T) {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	mux := NewOllamaHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, ollama.DefaultGeneratePath, strings.NewReader(`{"model":"qwen","stream":true,"prompt":"hello"}`))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
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
}
