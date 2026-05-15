// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"net/http"
	"net/http/httptest"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/anthropic"
	"dappco.re/go/inference/ollama"
	openaicompat "dappco.re/go/inference/openai"
)

func ExampleNewAnthropicMessagesHandler() {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	handler := NewAnthropicMessagesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, anthropic.DefaultMessagesPath, strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	core.Println(rec.Code)
	// Output: 200
}

func ExampleNewOllamaHandler() {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	mux := NewOllamaHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, ollama.DefaultGeneratePath, strings.NewReader(`{"model":"qwen","prompt":"hello"}`))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	core.Println(rec.Code)
	// Output: 200
}
