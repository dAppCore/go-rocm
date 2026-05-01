// Package rocm provides AMD ROCm GPU inference for Linux.
//
// This package implements the inference.Backend and inference.TextModel interfaces
// using llama.cpp compiled with HIP/ROCm for AMD GPUs (RDNA 3+).
//
// # Quick Start
//
//	import (
//	    "dappco.re/go/inference"
//	    _ "dappco.re/go/rocm" // auto-registers ROCm backend
//	)
//
//	m, err := inference.LoadModel("/path/to/model.gguf")
//	defer m.Close()
//	for tok := range m.Generate(ctx, "Hello", inference.WithMaxTokens(128)) {
//	    fmt.Print(tok.Text)
//	}
//
// # Requirements
//
//   - Linux (amd64)
//   - AMD GPU with ROCm support (RDNA 2+ / gfx10xx+, tested on RDNA 3 / gfx1100)
//   - ROCm 6.x+ installed
//   - llama-server binary (from llama.cpp built with -DGGML_HIP=ON)
package rocm

// VRAMInfo reports GPU video memory usage in bytes.
type VRAMInfo struct {
	Total uint64
	Used  uint64
	Free  uint64
}

// ModelInfo describes a GGUF model file discovered on disk.
type ModelInfo struct {
	Path         string // full path to .gguf file
	Architecture string // GGUF architecture (e.g. "gemma3", "llama", "qwen2")
	Name         string // human-readable model name from GGUF metadata
	Quantisation string // quantisation level (e.g. "Q4_K_M", "Q8_0")
	Parameters   string // parameter size label (e.g. "1B", "8B")
	FileSize     int64  // file size in bytes
	ContextLen   uint32 // native context window length
}

type rocmFailure interface {
	Error() string
}
