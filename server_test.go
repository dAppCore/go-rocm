//go:build linux && amd64

package rocm

import (
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFindLlamaServer_InPATH(t *testing.T) {
	// llama-server is at /usr/local/bin/llama-server on this machine.
	path, err := findLlamaServer()
	require.NoError(t, err)
	assert.Contains(t, path, "llama-server")
}

func TestFindLlamaServer_EnvOverride(t *testing.T) {
	t.Setenv("ROCM_LLAMA_SERVER_PATH", "/usr/local/bin/llama-server")
	path, err := findLlamaServer()
	require.NoError(t, err)
	assert.Equal(t, "/usr/local/bin/llama-server", path)
}

func TestFindLlamaServer_EnvNotFound(t *testing.T) {
	t.Setenv("ROCM_LLAMA_SERVER_PATH", "/nonexistent/llama-server")
	_, err := findLlamaServer()
	assert.ErrorContains(t, err, "not found")
}

func TestFreePort(t *testing.T) {
	port, err := freePort()
	require.NoError(t, err)
	assert.Greater(t, port, 0)
	assert.Less(t, port, 65536)
}

func TestFreePort_UniquePerCall(t *testing.T) {
	p1, err := freePort()
	require.NoError(t, err)
	p2, err := freePort()
	require.NoError(t, err)
	_ = p1
	_ = p2
}

func TestServerEnv_HIPVisibleDevices(t *testing.T) {
	env := serverEnv()
	var hipVals []string
	for _, e := range env {
		if strings.HasPrefix(e, "HIP_VISIBLE_DEVICES=") {
			hipVals = append(hipVals, e)
		}
	}
	assert.Equal(t, []string{"HIP_VISIBLE_DEVICES=0"}, hipVals)
}

func TestServerEnv_FiltersExistingHIP(t *testing.T) {
	t.Setenv("HIP_VISIBLE_DEVICES", "1")
	env := serverEnv()
	var hipVals []string
	for _, e := range env {
		if strings.HasPrefix(e, "HIP_VISIBLE_DEVICES=") {
			hipVals = append(hipVals, e)
		}
	}
	assert.Equal(t, []string{"HIP_VISIBLE_DEVICES=0"}, hipVals)
}

func TestAvailable(t *testing.T) {
	b := &rocmBackend{}
	if _, err := os.Stat("/dev/kfd"); err != nil {
		t.Skip("no ROCm hardware")
	}
	assert.True(t, b.Available())
}

func TestGuessModelType(t *testing.T) {
	tests := []struct {
		path     string
		expected string
	}{
		{"/data/lem/gguf/LEK-Gemma3-4B-Q4_K_M.gguf", "gemma3"},
		{"/data/models/Qwen3-8B-Q4_K_M.gguf", "qwen3"},
		{"/data/models/Llama-3.1-8B-Q4_K_M.gguf", "llama3"},
		{"/data/models/Mistral-7B-v0.3-Q4_K_M.gguf", "mistral"},
		{"/data/models/random-model.gguf", "unknown"},
	}
	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			assert.Equal(t, tt.expected, guessModelType(tt.path))
		})
	}
}
