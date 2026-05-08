// Package rocm provides the AMD ROCm backend for the Core Go inference stack.
//
// The default linux/amd64 build is native-first: it registers the ROCm backend
// through go-inference, exposes model-fit planning, probing, benchmarking,
// evaluation, tokenizer, and adapter contracts, and avoids the previous
// OpenAI-compatible llama-server subprocess path.
//
// The native HIP loader is intentionally explicit. Until it is linked in,
// Available reports false instead of hiding behind a server fallback. The old
// subprocess backend is retained only behind the rocm_legacy_server build tag.
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
//   - Linux (amd64) for the ROCm runtime build
//   - AMD GPU with ROCm support (RDNA 2+ / gfx10xx+ target class)
//   - ROCm/HIP runtime for the forthcoming native loader
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
