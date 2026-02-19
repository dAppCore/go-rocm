//go:build rocm

package rocm

import (
	"context"
	"os"
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
