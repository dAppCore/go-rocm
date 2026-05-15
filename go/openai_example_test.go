// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"net/http"
	"net/http/httptest"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/openai"
)

func ExampleNewOpenAIResponsesHandler() {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	handler := NewOpenAIResponsesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{"model":"qwen","input":[{"role":"user","content":"hello"}]}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	core.Println(rec.Code)
	// Output: 200
}

func ExampleNewOpenAIServiceMux() {
	model := &openAITestModel{tokens: []inference.Token{{Text: "ok"}}}
	mux := NewOpenAIServiceMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodGet, openaicompat.DefaultCapabilitiesPath+"?model=qwen", nil)
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	core.Println(rec.Code)
	// Output: 200
}
