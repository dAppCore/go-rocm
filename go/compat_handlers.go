// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"encoding/json"
	"iter"
	"net/http"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/anthropic"
	"dappco.re/go/inference/ollama"
	openaicompat "dappco.re/go/inference/openai"
)

// NewAnthropicMessagesHandler exposes the non-streaming Anthropic Messages
// endpoint over a caller-provided resolver.
func NewAnthropicMessagesHandler(resolver openaicompat.Resolver) http.Handler {
	return &anthropicMessagesHandler{resolver: resolver}
}

// NewOllamaHandler exposes non-streaming Ollama chat and generate endpoints
// over a caller-provided resolver.
func NewOllamaHandler(resolver openaicompat.Resolver) *http.ServeMux {
	mux := http.NewServeMux()
	handler := &ollamaCompatHandler{resolver: resolver}
	mux.Handle(ollama.DefaultChatPath, http.HandlerFunc(handler.chat))
	mux.Handle(ollama.DefaultGeneratePath, http.HandlerFunc(handler.generate))
	return mux
}

type anthropicMessagesHandler struct {
	resolver openaicompat.Resolver
}

func (handler *anthropicMessagesHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if handler == nil || handler.resolver == nil {
		writeROCmOpenAIError(w, http.StatusServiceUnavailable, "anthropic messages handler is not configured", "model")
		return
	}
	if !requireROCmWireMethod(w, r, http.MethodPost) {
		return
	}
	var req anthropic.MessageRequest
	if !decodeROCmWireRequest(w, r, &req) {
		return
	}
	if core.Trim(req.Model) == "" {
		writeROCmOpenAIError(w, http.StatusBadRequest, "model is required", "model")
		return
	}
	if req.Stream {
		writeROCmOpenAIError(w, http.StatusNotImplemented, "streaming Anthropic messages are not implemented by go-rocm yet", "stream")
		return
	}
	messages := anthropic.InferenceMessages(req)
	if !hasROCmWireMessages(messages) {
		writeROCmOpenAIError(w, http.StatusBadRequest, "messages or system are required", "messages")
		return
	}
	model, ok := resolveROCmWireModel(w, r, handler.resolver, req.Model)
	if !ok {
		return
	}
	text, ok := runROCmWireChat(w, r, model, messages, anthropic.GenerateOptions(req)...)
	if !ok {
		return
	}
	writeROCmOpenAIJSON(w, http.StatusOK, anthropic.NewTextResponse("msg_rocm", req.Model, text, model.Metrics()))
}

type ollamaCompatHandler struct {
	resolver openaicompat.Resolver
}

func (handler *ollamaCompatHandler) chat(w http.ResponseWriter, r *http.Request) {
	if handler == nil || handler.resolver == nil {
		writeROCmOpenAIError(w, http.StatusServiceUnavailable, "ollama handler is not configured", "model")
		return
	}
	if !requireROCmWireMethod(w, r, http.MethodPost) {
		return
	}
	var req ollama.ChatRequest
	if !decodeROCmWireRequest(w, r, &req) {
		return
	}
	if core.Trim(req.Model) == "" {
		writeROCmOpenAIError(w, http.StatusBadRequest, "model is required", "model")
		return
	}
	if req.Stream {
		writeROCmOpenAIError(w, http.StatusNotImplemented, "streaming Ollama chat is not implemented by go-rocm yet", "stream")
		return
	}
	messages := ollama.InferenceMessages(req.Messages)
	if !hasROCmWireMessages(messages) {
		writeROCmOpenAIError(w, http.StatusBadRequest, "messages are required", "messages")
		return
	}
	model, ok := resolveROCmWireModel(w, r, handler.resolver, req.Model)
	if !ok {
		return
	}
	text, ok := runROCmWireChat(w, r, model, messages, ollama.GenerateOptions(req.Options)...)
	if !ok {
		return
	}
	writeROCmOpenAIJSON(w, http.StatusOK, ollama.NewChatResponse(req.Model, text, model.Metrics()))
}

func (handler *ollamaCompatHandler) generate(w http.ResponseWriter, r *http.Request) {
	if handler == nil || handler.resolver == nil {
		writeROCmOpenAIError(w, http.StatusServiceUnavailable, "ollama handler is not configured", "model")
		return
	}
	if !requireROCmWireMethod(w, r, http.MethodPost) {
		return
	}
	var req ollama.GenerateRequest
	if !decodeROCmWireRequest(w, r, &req) {
		return
	}
	if core.Trim(req.Model) == "" {
		writeROCmOpenAIError(w, http.StatusBadRequest, "model is required", "model")
		return
	}
	if req.Stream {
		writeROCmOpenAIError(w, http.StatusNotImplemented, "streaming Ollama generate is not implemented by go-rocm yet", "stream")
		return
	}
	if core.Trim(req.Prompt) == "" {
		writeROCmOpenAIError(w, http.StatusBadRequest, "prompt is required", "prompt")
		return
	}
	model, ok := resolveROCmWireModel(w, r, handler.resolver, req.Model)
	if !ok {
		return
	}
	text := collectROCmWireTokenText(model.Generate(r.Context(), req.Prompt, ollama.GenerateOptions(req.Options)...))
	if err := model.Err(); err != nil {
		writeROCmOpenAIError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	writeROCmOpenAIJSON(w, http.StatusOK, ollama.NewGenerateResponse(req.Model, text, model.Metrics()))
}

func decodeROCmWireRequest(w http.ResponseWriter, r *http.Request, into any) bool {
	if r == nil || r.Body == nil {
		writeROCmOpenAIError(w, http.StatusBadRequest, "request body is nil", "body")
		return false
	}
	if err := json.NewDecoder(r.Body).Decode(into); err != nil {
		writeROCmOpenAIError(w, http.StatusBadRequest, "invalid request body", "body")
		return false
	}
	return true
}

func requireROCmWireMethod(w http.ResponseWriter, r *http.Request, method string) bool {
	if r == nil {
		writeROCmOpenAIError(w, http.StatusBadRequest, "request is nil", "request")
		return false
	}
	if r.Method != method {
		w.Header().Set("Allow", method)
		writeROCmOpenAIError(w, http.StatusMethodNotAllowed, "method not allowed", "method")
		return false
	}
	return true
}

func resolveROCmWireModel(w http.ResponseWriter, r *http.Request, resolver openaicompat.Resolver, name string) (inference.TextModel, bool) {
	model, err := resolver.ResolveModel(r.Context(), name)
	if err != nil {
		writeROCmOpenAIError(w, http.StatusNotFound, err.Error(), "model")
		return nil, false
	}
	return model, true
}

func runROCmWireChat(w http.ResponseWriter, r *http.Request, model inference.TextModel, messages []inference.Message, opts ...inference.GenerateOption) (string, bool) {
	text := collectROCmWireTokenText(model.Chat(r.Context(), messages, opts...))
	if err := model.Err(); err != nil {
		writeROCmOpenAIError(w, http.StatusInternalServerError, err.Error(), "model")
		return "", false
	}
	return text, true
}

// collectROCmWireTokenText accumulates streamed token text into a single
// response string with a single growable buffer, avoiding the O(n²) string
// re-allocation of `text += token.Text` on each decoded token.
//
//	text := collectROCmWireTokenText(model.Chat(ctx, messages))
func collectROCmWireTokenText(tokens iter.Seq[inference.Token]) string {
	var builder core.Builder
	for token := range tokens {
		builder.WriteString(token.Text)
	}
	return builder.String()
}

func hasROCmWireMessages(messages []inference.Message) bool {
	for _, message := range messages {
		if core.Trim(message.Content) != "" {
			return true
		}
	}
	return false
}
