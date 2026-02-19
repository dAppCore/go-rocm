//go:build rocm

package rocm

import (
	"context"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"forge.lthn.ai/core/go-inference"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	require.True(t, b.Available())

	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	require.NoError(t, err)
	defer m.Close()

	assert.Equal(t, "gemma3", m.ModelType())

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var tokens []string
	for tok := range m.Generate(ctx, "The capital of France is", inference.WithMaxTokens(16)) {
		tokens = append(tokens, tok.Text)
	}

	require.NoError(t, m.Err())
	require.NotEmpty(t, tokens, "expected at least one token")

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
	require.NoError(t, err)
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

	require.NoError(t, m.Err())
	require.NotEmpty(t, tokens, "expected at least one token")

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
	require.NoError(t, err)
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
	assert.GreaterOrEqual(t, count, 3)
}

func TestROCm_GracefulShutdown(t *testing.T) {
	skipIfNoROCm(t)
	skipIfNoModel(t)

	b := &rocmBackend{}
	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	require.NoError(t, err)
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

	require.NoError(t, m.Err())
	assert.Greater(t, count2, 0, "expected tokens from second generation after cancel")
	t.Logf("Second generation: %d tokens", count2)
}

func TestROCm_ConcurrentRequests(t *testing.T) {
	skipIfNoROCm(t)
	skipIfNoModel(t)

	b := &rocmBackend{}
	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	require.NoError(t, err)
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
		assert.NotEmpty(t, result, "goroutine %d produced no output", i)
	}
}
