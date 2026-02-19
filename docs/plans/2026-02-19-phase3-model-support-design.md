# Phase 3: Model Support Design

Approved 19 Feb 2026.

## 1. GGUF Metadata Parser (`internal/gguf/`)

New internal package that reads GGUF binary file headers. Extracts metadata key-value pairs without reading tensor data. Completes in <1ms per file.

```go
type Metadata struct {
    Architecture  string   // "gemma2", "llama", "qwen2"
    Name          string   // human-readable name
    ContextLength uint32   // native context window
    BlockCount    uint32   // transformer layers
    FileType      uint32   // GGML file type (quant level)
    FileSize      int64    // file size on disk
}

func ReadMetadata(path string) (Metadata, error)
```

GGUF format: magic "GGUF" (4 bytes), version uint32, tensor count uint64, metadata KV count uint64, then KV pairs. Each KV has: key length uint64, key string, value type uint32, value (variable). We only need a handful of keys: `general.architecture`, `general.name`, `general.file_type`, `<arch>.context_length`, `<arch>.block_count`.

## 2. Model Discovery (`discover.go`)

Package-level function that scans a directory for `.gguf` files and returns structured inventory.

```go
type ModelInfo struct {
    Path         string
    Architecture string
    Name         string
    Quantisation string   // "Q4_K_M", "Q8_0", etc.
    Parameters   string   // "1B", "4B", "8B"
    FileSize     int64
    ContextLen   uint32
}

func DiscoverModels(dir string) ([]ModelInfo, error)
```

Quantisation and parameter strings derived from GGUF `general.file_type` and `<arch>.block_count` respectively. Stub on non-Linux returns empty slice + error.

## 3. LoadModel Enrichment

At LoadModel time, read GGUF metadata to:
- Get real architecture (replaces filename-based `guessModelType`)
- If user didn't specify ContextLen (0), default to `min(model_context_length, 4096)` to prevent VRAM exhaustion on models that support 128K+ context

## 4. Chat Templates

llama-server reads `tokenizer.chat_template` from the GGUF and applies it automatically on `/v1/chat/completions`. No go-rocm code needed. Verify with integration test.

## 5. Testing

- GGUF parser: unit tests with a binary test fixture (first few KB of a real GGUF)
- Discovery: unit test with temp dir + test fixtures
- LoadModel enrichment: integration test verifying ModelType() returns correct architecture
- Chat templates: integration test verifying Chat() works on Gemma3
