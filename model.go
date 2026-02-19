//go:build linux && amd64

package rocm

import (
	"context"
	"fmt"
	"iter"
	"sync"

	"forge.lthn.ai/core/go-inference"
	"forge.lthn.ai/core/go-rocm/internal/llamacpp"
)

// rocmModel implements inference.TextModel using a llama-server subprocess.
type rocmModel struct {
	srv       *server
	modelType string

	mu      sync.Mutex
	lastErr error
}

// Generate streams tokens for the given prompt via llama-server's /v1/completions endpoint.
func (m *rocmModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.mu.Lock()
	m.lastErr = nil
	m.mu.Unlock()

	if !m.srv.alive() {
		m.mu.Lock()
		if m.srv.exitErr != nil {
			m.lastErr = fmt.Errorf("rocm: server has exited: %w", m.srv.exitErr)
		} else {
			m.lastErr = fmt.Errorf("rocm: server has exited unexpectedly")
		}
		m.mu.Unlock()
		return func(yield func(inference.Token) bool) {}
	}

	cfg := inference.ApplyGenerateOpts(opts)

	req := llamacpp.CompletionRequest{
		Prompt:        prompt,
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		RepeatPenalty: cfg.RepeatPenalty,
	}

	chunks, errFn := m.srv.client.Complete(ctx, req)

	return func(yield func(inference.Token) bool) {
		for text := range chunks {
			if !yield(inference.Token{Text: text}) {
				break
			}
		}
		if err := errFn(); err != nil {
			m.mu.Lock()
			m.lastErr = err
			m.mu.Unlock()
		}
	}
}

// Chat streams tokens from a multi-turn conversation via llama-server's /v1/chat/completions endpoint.
func (m *rocmModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.mu.Lock()
	m.lastErr = nil
	m.mu.Unlock()

	if !m.srv.alive() {
		m.mu.Lock()
		if m.srv.exitErr != nil {
			m.lastErr = fmt.Errorf("rocm: server has exited: %w", m.srv.exitErr)
		} else {
			m.lastErr = fmt.Errorf("rocm: server has exited unexpectedly")
		}
		m.mu.Unlock()
		return func(yield func(inference.Token) bool) {}
	}

	cfg := inference.ApplyGenerateOpts(opts)

	chatMsgs := make([]llamacpp.ChatMessage, len(messages))
	for i, msg := range messages {
		chatMsgs[i] = llamacpp.ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	req := llamacpp.ChatRequest{
		Messages:      chatMsgs,
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		RepeatPenalty: cfg.RepeatPenalty,
	}

	chunks, errFn := m.srv.client.ChatComplete(ctx, req)

	return func(yield func(inference.Token) bool) {
		for text := range chunks {
			if !yield(inference.Token{Text: text}) {
				break
			}
		}
		if err := errFn(); err != nil {
			m.mu.Lock()
			m.lastErr = err
			m.mu.Unlock()
		}
	}
}

// ModelType returns the architecture identifier (e.g. "gemma3", "qwen3", "llama3").
func (m *rocmModel) ModelType() string { return m.modelType }

// Err returns the error from the last Generate/Chat call, if any.
func (m *rocmModel) Err() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastErr
}

// Close releases the llama-server subprocess and all associated resources.
func (m *rocmModel) Close() error {
	return m.srv.stop()
}
