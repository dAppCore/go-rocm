//go:build linux && amd64

package rocm

import (
	"context"
	"fmt"
	"iter"
	"strings"
	"sync"
	"time"

	coreerr "dappco.re/go/core/log"
	"dappco.re/go/rocm/internal/llamacpp"
	"forge.lthn.ai/core/go-inference"
)

// rocmModel implements inference.TextModel using a llama-server subprocess.
type rocmModel struct {
	server    *server
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

	if !m.server.alive() {
		m.setServerExitErr()
		return func(yield func(inference.Token) bool) {}
	}

	generateConfig := inference.ApplyGenerateOpts(opts)
	request := completionRequest(prompt, generateConfig)
	promptTokens := approximatePromptTokens(prompt)

	start := time.Now()
	chunks, errFn := m.server.client.Complete(ctx, request)

	return func(yield func(inference.Token) bool) {
		var count int
		var firstTokenAt time.Time
		for text := range chunks {
			if firstTokenAt.IsZero() {
				firstTokenAt = time.Now()
			}
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
		m.recordMetrics(promptTokens, count, start, firstTokenAt)
	}
}

// Chat streams tokens from a multi-turn conversation via llama-server's /v1/chat/completions endpoint.
func (m *rocmModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.mu.Lock()
	m.lastErr = nil
	m.mu.Unlock()

	if !m.server.alive() {
		m.setServerExitErr()
		return func(yield func(inference.Token) bool) {}
	}

	generateConfig := inference.ApplyGenerateOpts(opts)
	promptTokens := approximateMessageTokens(messages)

	chatMsgs := make([]llamacpp.ChatMessage, len(messages))
	for i, msg := range messages {
		chatMsgs[i] = llamacpp.ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}
	request := chatRequest(chatMsgs, generateConfig)

	start := time.Now()
	chunks, errFn := m.server.client.ChatComplete(ctx, request)

	return func(yield func(inference.Token) bool) {
		var count int
		var firstTokenAt time.Time
		for text := range chunks {
			if firstTokenAt.IsZero() {
				firstTokenAt = time.Now()
			}
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
		m.recordMetrics(promptTokens, count, start, firstTokenAt)
	}
}

// Classify runs batched prefill-only inference via llama-server.
// Each prompt gets a single-token completion (max_tokens=1) while honoring
// the sampling settings from opts. llama-server has no native classify
// endpoint, so this simulates it.
func (m *rocmModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	if !m.server.alive() {
		m.setServerExitErr()
		return nil, m.Err()
	}

	generateConfig := inference.ApplyGenerateOpts(opts)
	results := make([]inference.ClassifyResult, len(prompts))
	totalPromptTokens := 0
	totalGenerated := 0
	var totalPrefill time.Duration
	var totalDecode time.Duration

	for promptIndex, prompt := range prompts {
		if contextError := ctx.Err(); contextError != nil {
			m.recordMetricsDurations(totalPromptTokens, totalGenerated, totalPrefill, totalDecode)
			return nil, coreerr.E("rocm.Classify", fmt.Sprintf("classify cancelled before prompt %d", promptIndex), contextError)
		}

		totalPromptTokens += approximatePromptTokens(prompt)
		request := completionRequest(prompt, generateConfig)
		request.MaxTokens = 1

		requestStart := time.Now()
		chunks, errFn := m.server.client.Complete(ctx, request)
		var text strings.Builder
		var firstTokenAt time.Time
		var generated int
		for chunk := range chunks {
			if firstTokenAt.IsZero() {
				firstTokenAt = time.Now()
			}
			generated++
			text.WriteString(chunk)
		}
		requestEnd := time.Now()
		prefill, decode := splitDurations(requestStart, firstTokenAt, requestEnd)
		totalPrefill += prefill
		totalDecode += decode
		totalGenerated += generated

		if err := errFn(); err != nil {
			m.recordMetricsDurations(totalPromptTokens, totalGenerated, totalPrefill, totalDecode)
			return nil, coreerr.E("rocm.Classify", fmt.Sprintf("classify prompt %d", promptIndex), err)
		}

		results[promptIndex] = inference.ClassifyResult{
			Token: inference.Token{Text: text.String()},
		}
	}

	m.recordMetricsDurations(totalPromptTokens, totalGenerated, totalPrefill, totalDecode)
	return results, nil
}

// BatchGenerate runs batched autoregressive generation via llama-server.
// Each prompt is decoded sequentially up to MaxTokens.
func (m *rocmModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.BatchResult, error) {
	if !m.server.alive() {
		m.setServerExitErr()
		return nil, m.Err()
	}

	generateConfig := inference.ApplyGenerateOpts(opts)
	results := make([]inference.BatchResult, len(prompts))
	totalPromptTokens := 0
	var totalGenerated int
	var totalPrefill time.Duration
	var totalDecode time.Duration

	for promptIndex, prompt := range prompts {
		if contextError := ctx.Err(); contextError != nil {
			results[promptIndex].Err = coreerr.E("rocm.BatchGenerate", fmt.Sprintf("batch prompt %d cancelled before start", promptIndex), contextError)
			continue
		}

		totalPromptTokens += approximatePromptTokens(prompt)
		request := completionRequest(prompt, generateConfig)

		requestStart := time.Now()
		chunks, errFn := m.server.client.Complete(ctx, request)
		var tokens []inference.Token
		var firstTokenAt time.Time
		for text := range chunks {
			if firstTokenAt.IsZero() {
				firstTokenAt = time.Now()
			}
			tokens = append(tokens, inference.Token{Text: text})
		}
		requestEnd := time.Now()
		prefill, decode := splitDurations(requestStart, firstTokenAt, requestEnd)
		totalPrefill += prefill
		totalDecode += decode
		results[promptIndex].Tokens = tokens
		totalGenerated += len(tokens)

		if err := errFn(); err != nil {
			results[promptIndex].Err = coreerr.E("rocm.BatchGenerate", fmt.Sprintf("batch prompt %d", promptIndex), err)
		}
	}

	m.recordMetricsDurations(totalPromptTokens, totalGenerated, totalPrefill, totalDecode)
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
	return m.server.stop()
}

// setServerExitErr stores an appropriate error when the server is dead.
func (m *rocmModel) setServerExitErr() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.server.exitErr != nil {
		m.lastErr = m.server.wrapProcessError("rocm.setServerExitErr", "server has exited", m.server.exitErr)
	} else {
		m.lastErr = coreerr.E("rocm.setServerExitErr", m.server.messageWithProcessOutput("server has exited unexpectedly"), nil)
	}
}

// recordMetrics captures timing data from an inference operation.
func (m *rocmModel) recordMetrics(promptTokens, generatedTokens int, start, firstTokenAt time.Time) {
	prefill, decode := splitDurations(start, firstTokenAt, time.Now())
	m.recordMetricsDurations(promptTokens, generatedTokens, prefill, decode)
}

func (m *rocmModel) recordMetricsDurations(promptTokens, generatedTokens int, prefill, decode time.Duration) {
	if prefill < 0 {
		prefill = 0
	}
	if decode < 0 {
		decode = 0
	}
	total := prefill + decode

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

func completionRequest(prompt string, cfg inference.GenerateConfig) llamacpp.CompletionRequest {
	return llamacpp.CompletionRequest{
		Prompt:        prompt,
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		RepeatPenalty: cfg.RepeatPenalty,
	}
}

func chatRequest(messages []llamacpp.ChatMessage, cfg inference.GenerateConfig) llamacpp.ChatRequest {
	return llamacpp.ChatRequest{
		Messages:      messages,
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		RepeatPenalty: cfg.RepeatPenalty,
	}
}

func splitDurations(start, firstTokenAt, end time.Time) (time.Duration, time.Duration) {
	if start.IsZero() || end.Before(start) {
		return 0, 0
	}
	if firstTokenAt.IsZero() || firstTokenAt.Before(start) || firstTokenAt.After(end) {
		return end.Sub(start), 0
	}
	return firstTokenAt.Sub(start), end.Sub(firstTokenAt)
}

// llama-server's streaming API does not expose prompt token counts, so metrics
// use a lightweight whitespace-token approximation for prefill throughput.
func approximatePromptTokens(prompt string) int {
	return len(strings.Fields(prompt))
}

func approximateMessageTokens(messages []inference.Message) int {
	total := 0
	for _, msg := range messages {
		total += approximatePromptTokens(msg.Content)
	}
	return total
}
