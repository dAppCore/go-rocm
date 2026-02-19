//go:build linux && amd64

package rocm

import (
	"os"
	"path/filepath"
	"strings"

	"forge.lthn.ai/core/go-inference"
)

// rocmBackend implements inference.Backend for AMD ROCm GPUs.
type rocmBackend struct{}

func (b *rocmBackend) Name() string { return "rocm" }

// Available reports whether ROCm GPU inference can run on this machine.
// Checks for the ROCm kernel driver (/dev/kfd) and a findable llama-server binary.
func (b *rocmBackend) Available() bool {
	if _, err := os.Stat("/dev/kfd"); err != nil {
		return false
	}
	if _, err := findLlamaServer(); err != nil {
		return false
	}
	return true
}

// LoadModel loads a GGUF model onto the AMD GPU via llama-server.
func (b *rocmBackend) LoadModel(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
	cfg := inference.ApplyLoadOpts(opts)

	binary, err := findLlamaServer()
	if err != nil {
		return nil, err
	}

	srv, err := startServer(binary, path, cfg.GPULayers, cfg.ContextLen)
	if err != nil {
		return nil, err
	}

	return &rocmModel{
		srv:       srv,
		modelType: guessModelType(path),
	}, nil
}

// guessModelType extracts a model type hint from the filename.
// Hyphens are stripped so that "Llama-3" matches "llama3".
func guessModelType(path string) string {
	name := strings.ToLower(filepath.Base(path))
	name = strings.ReplaceAll(name, "-", "")
	for _, arch := range []string{"gemma3", "gemma2", "gemma", "qwen3", "qwen2", "qwen", "llama3", "llama", "mistral", "phi"} {
		if strings.Contains(name, arch) {
			return arch
		}
	}
	return "unknown"
}
