//go:build linux && amd64

package rocm

import (
	"context"
	"fmt"
	"iter"
	"strings"
	"sync"
	"time"

	"forge.lthn.ai/core/go-inference"
	"forge.lthn.ai/core/go-rocm/internal/llamacpp"
)

// rocmModel implements inference.TextModel using a llama-server subprocess.
type rocmModel struct {
	srv       *server
	modelType string
	modelInfo inference.ModelInfo

	mu      sync.Mutex
	lastErr error
	metrics inference.GenerateMetrics
}

// Generate streams tokens for the given prompt via llama-server's /v1/completions endpoint.
func (m *rocmModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.mu.Lock()
	m.lastErr = nil
	m.mu.Unlock()

	if !m.srv.alive() {
		m.setServerExitErr()
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

	start := time.Now()
	chunks, errFn := m.srv.client.Complete(ctx, req)

	return func(yield func(inference.Token) bool) {
		var count int
		decodeStart := time.Now()
		for text := range chunks {
			count++
			if !yield(inference.Token{Text: text}) {
				break
			}
		}
		if err := errFn(); err != nil {
			m.mu.Lock()
			m.lastErr = err
			m.mu.Unlock()
		}
		m.recordMetrics(0, count, start, decodeStart)
	}
}

// Chat streams tokens from a multi-turn conversation via llama-server's /v1/chat/completions endpoint.
func (m *rocmModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.mu.Lock()
	m.lastErr = nil
	m.mu.Unlock()

	if !m.srv.alive() {
		m.setServerExitErr()
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

	start := time.Now()
	chunks, errFn := m.srv.client.ChatComplete(ctx, req)

	return func(yield func(inference.Token) bool) {
		var count int
		decodeStart := time.Now()
		for text := range chunks {
			count++
			if !yield(inference.Token{Text: text}) {
				break
			}
		}
		if err := errFn(); err != nil {
			m.mu.Lock()
			m.lastErr = err
			m.mu.Unlock()
		}
		m.recordMetrics(0, count, start, decodeStart)
	}
}

// Classify runs batched prefill-only inference via llama-server.
// Each prompt gets a single-token completion (max_tokens=1, temperature=0).
// llama-server has no native classify endpoint, so this simulates it.
func (m *rocmModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	if !m.srv.alive() {
		m.setServerExitErr()
		return nil, m.Err()
	}

	start := time.Now()
	results := make([]inference.ClassifyResult, len(prompts))

	for i, prompt := range prompts {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		req := llamacpp.CompletionRequest{
			Prompt:      prompt,
			MaxTokens:   1,
			Temperature: 0,
		}

		chunks, errFn := m.srv.client.Complete(ctx, req)
		var text strings.Builder
		for chunk := range chunks {
			text.WriteString(chunk)
		}
		if err := errFn(); err != nil {
			return nil, fmt.Errorf("rocm: classify prompt %d: %w", i, err)
		}

		results[i] = inference.ClassifyResult{
			Token: inference.Token{Text: text.String()},
		}
	}

	m.recordMetrics(len(prompts), len(prompts), start, start)
	return results, nil
}

// BatchGenerate runs batched autoregressive generation via llama-server.
// Each prompt is decoded sequentially up to MaxTokens.
func (m *rocmModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.BatchResult, error) {
	if !m.srv.alive() {
		m.setServerExitErr()
		return nil, m.Err()
	}

	cfg := inference.ApplyGenerateOpts(opts)
	start := time.Now()
	results := make([]inference.BatchResult, len(prompts))
	var totalGenerated int

	for i, prompt := range prompts {
		if ctx.Err() != nil {
			results[i].Err = ctx.Err()
			continue
		}

		req := llamacpp.CompletionRequest{
			Prompt:        prompt,
			MaxTokens:     cfg.MaxTokens,
			Temperature:   cfg.Temperature,
			TopK:          cfg.TopK,
			TopP:          cfg.TopP,
			RepeatPenalty: cfg.RepeatPenalty,
		}

		chunks, errFn := m.srv.client.Complete(ctx, req)
		var tokens []inference.Token
		for text := range chunks {
			tokens = append(tokens, inference.Token{Text: text})
		}
		if err := errFn(); err != nil {
			results[i].Err = fmt.Errorf("rocm: batch prompt %d: %w", i, err)
		}
		results[i].Tokens = tokens
		totalGenerated += len(tokens)
	}

	m.recordMetrics(len(prompts), totalGenerated, start, start)
	return results, nil
}

// ModelType returns the architecture identifier (e.g. "gemma3", "qwen3", "llama3").
func (m *rocmModel) ModelType() string { return m.modelType }

// Info returns metadata about the loaded model.
func (m *rocmModel) Info() inference.ModelInfo { return m.modelInfo }

// Metrics returns performance metrics from the last inference operation.
func (m *rocmModel) Metrics() inference.GenerateMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.metrics
}

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

// setServerExitErr stores an appropriate error when the server is dead.
func (m *rocmModel) setServerExitErr() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.srv.exitErr != nil {
		m.lastErr = fmt.Errorf("rocm: server has exited: %w", m.srv.exitErr)
	} else {
		m.lastErr = fmt.Errorf("rocm: server has exited unexpectedly")
	}
}

// recordMetrics captures timing data from an inference operation.
func (m *rocmModel) recordMetrics(promptTokens, generatedTokens int, start, decodeStart time.Time) {
	now := time.Now()
	total := now.Sub(start)
	decode := now.Sub(decodeStart)
	prefill := total - decode

	met := inference.GenerateMetrics{
		PromptTokens:    promptTokens,
		GeneratedTokens: generatedTokens,
		PrefillDuration: prefill,
		DecodeDuration:  decode,
		TotalDuration:   total,
	}
	if prefill > 0 && promptTokens > 0 {
		met.PrefillTokensPerSec = float64(promptTokens) / prefill.Seconds()
	}
	if decode > 0 && generatedTokens > 0 {
		met.DecodeTokensPerSec = float64(generatedTokens) / decode.Seconds()
	}

	// Try to get VRAM stats — best effort.
	if vram, err := GetVRAMInfo(); err == nil {
		met.PeakMemoryBytes = vram.Used
		met.ActiveMemoryBytes = vram.Used
	}

	m.mu.Lock()
	m.metrics = met
	m.mu.Unlock()
}
