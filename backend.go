//go:build linux && amd64

package rocm

import (
	"os"
	"strings"

	coreerr "forge.lthn.ai/core/go-log"
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
		return nil, coreerr.E("rocm.LoadModel", "read model metadata", err)
	}

	ctxLen := cfg.ContextLen
	if ctxLen == 0 && meta.ContextLength > 0 {
		ctxLen = int(min(meta.ContextLength, 4096))
	}

	srv, err := startServer(binary, path, cfg.GPULayers, ctxLen, cfg.ParallelSlots)
	if err != nil {
		return nil, err
	}

	// Map quantisation file type to bit width.
	quantBits := 0
	quantGroup := 0
	ftName := gguf.FileTypeName(meta.FileType)
	switch {
	case strings.HasPrefix(ftName, "Q4_"):
		quantBits = 4
		quantGroup = 32
	case strings.HasPrefix(ftName, "Q5_"):
		quantBits = 5
		quantGroup = 32
	case strings.HasPrefix(ftName, "Q8_"):
		quantBits = 8
		quantGroup = 32
	case strings.HasPrefix(ftName, "Q2_"):
		quantBits = 2
		quantGroup = 16
	case strings.HasPrefix(ftName, "Q3_"):
		quantBits = 3
		quantGroup = 32
	case strings.HasPrefix(ftName, "Q6_"):
		quantBits = 6
		quantGroup = 64
	case ftName == "F16":
		quantBits = 16
	case ftName == "F32":
		quantBits = 32
	}

	return &rocmModel{
		srv:       srv,
		modelType: meta.Architecture,
		modelInfo: inference.ModelInfo{
			Architecture: meta.Architecture,
			NumLayers:    int(meta.BlockCount),
			QuantBits:    quantBits,
			QuantGroup:   quantGroup,
		},
	}, nil
}
