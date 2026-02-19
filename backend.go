//go:build linux && amd64

package rocm

import (
	"fmt"
	"os"

	"forge.lthn.ai/core/go-inference"
	"forge.lthn.ai/core/go-rocm/internal/gguf"
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
// Model architecture is read from GGUF metadata (replacing filename-based guessing).
// If no context length is specified, defaults to min(model_context_length, 4096)
// to prevent VRAM exhaustion on models with 128K+ native context.
func (b *rocmBackend) LoadModel(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
	cfg := inference.ApplyLoadOpts(opts)

	binary, err := findLlamaServer()
	if err != nil {
		return nil, err
	}

	meta, err := gguf.ReadMetadata(path)
	if err != nil {
		return nil, fmt.Errorf("rocm: read model metadata: %w", err)
	}

	ctxLen := cfg.ContextLen
	if ctxLen == 0 && meta.ContextLength > 0 {
		ctxLen = int(min(meta.ContextLength, 4096))
	}

	srv, err := startServer(binary, path, cfg.GPULayers, ctxLen)
	if err != nil {
		return nil, err
	}

	return &rocmModel{
		srv:       srv,
		modelType: meta.Architecture,
	}, nil
}
