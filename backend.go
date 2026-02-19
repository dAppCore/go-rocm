//go:build linux && amd64

package rocm

import "forge.lthn.ai/core/go-inference"

// rocmBackend implements inference.Backend for AMD ROCm GPUs.
// Uses llama-server (llama.cpp built with HIP) as the inference engine.
type rocmBackend struct{}

func (b *rocmBackend) Name() string { return "rocm" }

func (b *rocmBackend) Available() bool {
	// TODO: Check for ROCm runtime + GPU presence
	// - /dev/kfd exists (ROCm kernel driver)
	// - rocm-smi detects a GPU
	// - llama-server binary is findable
	return false // Stub until Phase 1 implementation
}

func (b *rocmBackend) LoadModel(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
	// TODO: Phase 1 implementation
	// 1. Find llama-server binary (PATH or configured location)
	// 2. Spawn llama-server with --model path --port <free> --n-gpu-layers cfg.GPULayers
	// 3. Wait for health endpoint to respond
	// 4. Return rocmModel wrapping the HTTP client
	return nil, nil
}
