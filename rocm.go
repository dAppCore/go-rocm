// Package rocm provides AMD ROCm GPU inference for Linux.
//
// This package implements the inference.Backend and inference.TextModel interfaces
// using llama.cpp compiled with HIP/ROCm for AMD GPUs (RDNA 3+).
//
// # Quick Start
//
//	import (
//	    "forge.lthn.ai/core/go-inference"
//	    _ "forge.lthn.ai/core/go-rocm" // auto-registers ROCm backend
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
