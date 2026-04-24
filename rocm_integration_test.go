//go:build rocm

package rocm

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"forge.lthn.ai/core/go-inference"
)

const testModel = "/data/lem/gguf/LEK-Gemma3-1B-layered-v2-Q5_K_M.gguf"

func skipIfNoModel(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(testModel); os.IsNotExist(err) {
		t.Skip("test model not available (SMB mount down?)")
	}
}

func skipIfNoROCm(t *testing.T) {
	t.Helper()
	if _, err := os.Stat("/dev/kfd"); err != nil {
		t.Skip("no ROCm hardware")
	}
	if _, err := findLlamaServer(); err != nil {
		t.Skip("llama-server not found")
	}
}

func TestROCm_LoadAndGenerate(t *testing.T) {
	skipIfNoROCm(t)
	skipIfNoModel(t)

	b := &rocmBackend{}
	if !b.Available() {
		t.Fatal("b.Available() = false, want true")
	}

	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	if got := m.ModelType(); got != "gemma3" {
		t.Errorf("ModelType() = %q, want %q", got, "gemma3")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var tokens []string
	for tok := range m.Generate(ctx, "The capital of France is", inference.WithMaxTokens(16)) {
		tokens = append(tokens, tok.Text)
	}

	if err := m.Err(); err != nil {
		t.Fatalf("m.Err(): %v", err)
	}
	if len(tokens) == 0 {
		t.Fatal("expected at least one token")
	}

	full := ""
	for _, tok := range tokens {
		full += tok
	}
	t.Logf("Generated: %s", full)
}

func TestROCm_Chat(t *testing.T) {
	skipIfNoROCm(t)
	skipIfNoModel(t)

	b := &rocmBackend{}
	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	messages := []inference.Message{
		{Role: "user", Content: "Say hello in exactly three words."},
	}

	var tokens []string
	for tok := range m.Chat(ctx, messages, inference.WithMaxTokens(32)) {
		tokens = append(tokens, tok.Text)
	}

	if err := m.Err(); err != nil {
		t.Fatalf("m.Err(): %v", err)
	}
	if len(tokens) == 0 {
		t.Fatal("expected at least one token")
	}

	full := ""
	for _, tok := range tokens {
		full += tok
	}
	t.Logf("Chat response: %s", full)
}

func TestROCm_ContextCancellation(t *testing.T) {
	skipIfNoROCm(t)
	skipIfNoModel(t)

	b := &rocmBackend{}
	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	ctx, cancel := context.WithCancel(context.Background())

	var count int
	for tok := range m.Generate(ctx, "Write a very long story about dragons", inference.WithMaxTokens(256)) {
		_ = tok
		count++
		if count >= 3 {
			cancel()
		}
	}

	t.Logf("Got %d tokens before cancel", count)
	if count < 3 {
		t.Errorf("count = %d, want >= 3", count)
	}
}

func TestROCm_GracefulShutdown(t *testing.T) {
	skipIfNoROCm(t)
	skipIfNoModel(t)

	b := &rocmBackend{}
	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	// Cancel mid-stream.
	ctx1, cancel1 := context.WithCancel(context.Background())
	var count1 int
	for tok := range m.Generate(ctx1, "Write a long story about space exploration", inference.WithMaxTokens(256)) {
		_ = tok
		count1++
		if count1 >= 5 {
			cancel1()
		}
	}
	t.Logf("First generation: %d tokens before cancel", count1)

	// Generate again on the same model — server should still be alive.
	ctx2, cancel2 := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel2()

	var count2 int
	for tok := range m.Generate(ctx2, "The capital of France is", inference.WithMaxTokens(16)) {
		_ = tok
		count2++
	}

	if err := m.Err(); err != nil {
		t.Fatalf("m.Err(): %v", err)
	}
	if count2 == 0 {
		t.Error("expected tokens from second generation after cancel")
	}
	t.Logf("Second generation: %d tokens", count2)
}

func TestROCm_ConcurrentRequests(t *testing.T) {
	skipIfNoROCm(t)
	skipIfNoModel(t)

	b := &rocmBackend{}
	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	const numGoroutines = 3
	results := make([]string, numGoroutines)

	prompts := []string{
		"The capital of France is",
		"The capital of Germany is",
		"The capital of Italy is",
	}

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := range numGoroutines {
		go func(idx int) {
			defer wg.Done()

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			var sb strings.Builder
			for tok := range m.Generate(ctx, prompts[idx], inference.WithMaxTokens(16)) {
				sb.WriteString(tok.Text)
			}
			results[idx] = sb.String()
		}(i)
	}

	wg.Wait()

	for i, result := range results {
		t.Logf("Goroutine %d: %s", i, result)
		if result == "" {
			t.Errorf("goroutine %d produced no output", i)
		}
	}
}

func TestROCm_Classify(t *testing.T) {
	skipIfNoROCm(t)
	skipIfNoModel(t)

	b := &rocmBackend{}
	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	prompts := []string{
		"The capital of France is",
		"2 + 2 =",
	}

	results, err := m.Classify(ctx, prompts)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("len(results) = %d, want 2", len(results))
	}

	for i, r := range results {
		if r.Token.Text == "" {
			t.Errorf("classify result %d should have a token", i)
		}
		t.Logf("Classify %d: %q", i, r.Token.Text)
	}
}

func TestROCm_BatchGenerate(t *testing.T) {
	skipIfNoROCm(t)
	skipIfNoModel(t)

	b := &rocmBackend{}
	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	prompts := []string{
		"The capital of France is",
		"The capital of Germany is",
	}

	results, err := m.BatchGenerate(ctx, prompts, inference.WithMaxTokens(8))
	if err != nil {
		t.Fatalf("BatchGenerate: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("len(results) = %d, want 2", len(results))
	}

	for i, r := range results {
		if r.Err != nil {
			t.Fatalf("batch result %d error: %v", i, r.Err)
		}
		if len(r.Tokens) == 0 {
			t.Errorf("batch result %d should have tokens", i)
		}

		var sb strings.Builder
		for _, tok := range r.Tokens {
			sb.WriteString(tok.Text)
		}
		t.Logf("Batch %d: %s", i, sb.String())
	}
}

func TestROCm_InfoAndMetrics(t *testing.T) {
	skipIfNoROCm(t)
	skipIfNoModel(t)

	b := &rocmBackend{}
	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	// Info should be populated from GGUF metadata.
	info := m.Info()
	if info.Architecture != "gemma3" {
		t.Errorf("Architecture = %q, want %q", info.Architecture, "gemma3")
	}
	if info.NumLayers == 0 {
		t.Error("expected non-zero layer count")
	}
	if info.QuantBits == 0 {
		t.Error("expected non-zero quant bits")
	}
	t.Logf("Info: arch=%s layers=%d quant=%d-bit group=%d",
		info.Architecture, info.NumLayers, info.QuantBits, info.QuantGroup)

	// Generate some tokens to populate metrics.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	for range m.Generate(ctx, "Hello", inference.WithMaxTokens(4)) {
	}
	if err := m.Err(); err != nil {
		t.Fatalf("m.Err(): %v", err)
	}

	met := m.Metrics()
	if met.GeneratedTokens == 0 {
		t.Error("expected generated tokens")
	}
	if met.TotalDuration == 0 {
		t.Error("expected non-zero duration")
	}
	if met.DecodeTokensPerSec <= 0 {
		t.Error("expected non-zero decode throughput")
	}
	t.Logf("Metrics: gen=%d tok, total=%s, decode=%.1f tok/s, vram=%d MiB",
		met.GeneratedTokens, met.TotalDuration, met.DecodeTokensPerSec,
		met.ActiveMemoryBytes/(1024*1024))
}

func TestROCm_DiscoverModels(t *testing.T) {
	dir := filepath.Dir(testModel)
	if _, err := os.Stat(dir); err != nil {
		t.Skip("model directory not available")
	}

	models, err := DiscoverModels(dir)
	if err != nil {
		t.Fatalf("DiscoverModels: %v", err)
	}
	if len(models) == 0 {
		t.Fatalf("expected at least one model in %s", dir)
	}

	for _, m := range models {
		t.Logf("Found: %s (%s %s %s, ctx=%d)", filepath.Base(m.Path), m.Architecture, m.Parameters, m.Quantisation, m.ContextLen)
		if m.Architecture == "" {
			t.Errorf("empty Architecture for %s", m.Path)
		}
		if m.Name == "" {
			t.Errorf("empty Name for %s", m.Path)
		}
		if m.FileSize <= 0 {
			t.Errorf("FileSize = %d for %s, want > 0", m.FileSize, m.Path)
		}
	}
}
