# go-rocm Architecture

## Overview

go-rocm provides AMD ROCm GPU inference for Linux by managing llama-server as a subprocess. It implements the `inference.Backend` and `inference.TextModel` interfaces from go-inference, making the AMD GPU available to the broader Go ML ecosystem (go-ml, go-ai, go-i18n) without any CGO in the package itself.

Module path: `dappco.re/go/rocm`

## Design Choice: Subprocess over CGO

The package uses llama-server (from llama.cpp) as a managed subprocess rather than direct HIP CGO bindings. This decision was deliberate:

- llama-server supports 50+ model architectures via GGUF without any additional work in this package
- llama.cpp's ROCm/HIP compatibility is maintained by the llama.cpp team
- GPU crashes in the subprocess cannot take down the host Go process
- The same subprocess pattern works for NVIDIA (CUDA build) and Intel (SYCL build) with minimal code changes

The trade-offs are modest: a small HTTP overhead compared to in-process function calls, and an additional ~50ms latency during model load while the server process starts. For inference workloads these costs are negligible.

The sibling package go-mlx takes the CGO approach because MLX is a C library designed for embedding. llama.cpp's primary supported interface is its server mode.

## Package Structure

```
go-rocm/
├── rocm.go              Package doc and exported types (VRAMInfo, ModelInfo)
├── register_rocm.go     linux && amd64 — auto-registers via init()
├── rocm_stub.go         !linux || !amd64 — safe stubs for cross-compilation
├── backend.go           inference.Backend implementation
├── model.go             inference.TextModel implementation
├── server.go            llama-server lifecycle management
├── vram.go              VRAM monitoring via sysfs
├── discover.go          GGUF model discovery
└── internal/
    ├── llamacpp/
    │   ├── health.go    HTTP client and health check
    │   └── client.go    OpenAI-compatible streaming client
    └── gguf/
        └── gguf.go      GGUF v2/v3 binary metadata parser
```

## Build Tags

The package uses build constraints to ensure correctness across platforms:

- `//go:build linux && amd64` — all GPU-touching code: `backend.go`, `model.go`, `server.go`, `vram.go`, `register_rocm.go`
- `//go:build !linux || !amd64` — `rocm_stub.go` provides `ROCmAvailable() bool { return false }` and a `GetVRAMInfo()` that returns an error, allowing the package to compile everywhere
- `//go:build rocm` — integration tests and benchmarks, gated behind an explicit tag to keep `go test ./...` fast on machines without GPU hardware
- `discover.go` has no build constraint; GGUF file parsing is portable

## Auto-Registration

On Linux/amd64, `register_rocm.go` calls `inference.Register(&rocmBackend{})` in an `init()` function. Any program that blank-imports go-rocm gets the backend automatically:

```go
import _ "dappco.re/go/rocm"
```

The backend is then available to `inference.LoadModel()` from go-inference, which iterates registered backends and calls `Available()` on each to select one.

## Inference Flow

### 1. Availability Check

`rocmBackend.Available()` verifies two conditions:

- `/dev/kfd` exists — confirms the amdgpu kernel driver is loaded and ROCm is functional
- `findLlamaServer()` succeeds — checks `ROCM_LLAMA_SERVER_PATH` env var first, then PATH

If either check fails, `Available()` returns false and the backend is skipped.

### 2. Model Loading

`LoadModel(path, opts...)` orchestrates the full startup sequence:

1. Calls `findLlamaServer()` to locate the binary
2. Calls `gguf.ReadMetadata(path)` to extract the model's native context length and architecture without loading tensors
3. Applies the context length cap: `min(model_context_length, 4096)` when the caller has not specified a context length explicitly. This prevents VRAM exhaustion on models with 128K+ native context
4. Calls `startServer()` with the resolved parameters
5. Returns a `*rocmModel` wrapping the running server

### 3. Server Lifecycle

`startServer()` in `server.go` manages the subprocess:

**Port selection**: `freePort()` asks the kernel for an available TCP port by listening on `127.0.0.1:0` and recording the assigned port before closing the listener.

**Environment preparation**: `serverEnv()` copies the current process environment, strips any existing `HIP_VISIBLE_DEVICES` entry (even if the operator has set it to something else), and appends `HIP_VISIBLE_DEVICES=0`. This is critical: the Ryzen 9 9950X's integrated GPU appears as ROCm Device 1 and reports approximately 100 GB free (it is using system RAM). Without masking, llama-server's auto-fit logic splits tensors across both devices and crashes with `ROCm error: unspecified launch failure`.

**Process start**: `exec.Command` spawns llama-server with:
```
--model <path>
--host 127.0.0.1
--port <port>
--n-gpu-layers <layers>   (999 by default = all layers on GPU)
--ctx-size <N>            (when specified)
--parallel <N>            (when ParallelSlots > 0)
```

**Readiness polling**: `waitReady()` polls `GET /health` every 100ms with a 60-second deadline. It selects across three channels simultaneously: the context deadline, the `exited` channel (process died before becoming ready), and the ticker. Model load time is typically 6–10 seconds for a 4–8B model.

**Retry on port conflict**: If the process exits during startup (exit before the health check passes), `startServer()` retries up to 3 times with a freshly selected port. Timeouts are not retried — a stuck server is a different failure mode.

**Shutdown**: `server.stop()` sends SIGTERM and waits up to 5 seconds for a clean exit. If the process has not exited after 5 seconds, it sends SIGKILL and waits for the channel to close.

### 4. Token Streaming

`rocmModel.Generate()` maps to `/v1/completions`. `rocmModel.Chat()` maps to `/v1/chat/completions`. Both:

1. Check `server.alive()` by reading from the `exited` channel non-blockingly. If the server has died, an error is recorded in `lastErr` and an empty iterator is returned immediately
2. Build the request struct with sampling parameters (temperature, top-k, top-p, repeat penalty, max tokens)
3. Call the appropriate client method, which returns `(iter.Seq[string], func() error)`
4. Wrap the chunk iterator into an `iter.Seq[inference.Token]`, setting `Token.Text` from each chunk and leaving `Token.ID` as zero (llama-server's OpenAI-compatible streaming API does not return token IDs)
5. After the iterator completes, call the error function and store any error in `lastErr` under the mutex

The SSE parser in `internal/llamacpp/client.go` uses a `bufio.Scanner` to read `data: ` prefixed lines, stops at `[DONE]`, and propagates scan errors via a pointer. Response bodies are closed exactly once via `sync.Once`.

### 5. Chat Templates

llama-server reads `tokenizer.chat_template` from the GGUF file and applies it automatically on the `/v1/chat/completions` endpoint. go-rocm does not implement any template logic.

## GGUF Metadata Parser

`internal/gguf/` is a standalone binary metadata reader. It supports GGUF v2 (uint32 tensor/KV counts) and v3 (uint64 counts).

The parser reads the file header sequentially:

1. Magic number validation (`0x46554747`, the ASCII string "GGUF" in little-endian)
2. Version field (2 or 3; others return an error)
3. Tensor count and KV count (width depends on version)
4. All KV pairs in sequence

For each KV pair, the key string is read first, then the value type, then the value. Interesting keys are:
- `general.architecture` — architecture identifier (e.g. `gemma3`, `llama`, `qwen2`)
- `general.name` — human-readable model name
- `general.file_type` — GGML quantisation type code
- `general.size_label` — parameter count label (e.g. `1B`, `8B`)
- Any key with suffix `.context_length`
- Any key with suffix `.block_count`

Architecture-specific keys like `llama.context_length` are collected into candidate maps and resolved after the architecture is known. Uninteresting keys are skipped without allocation.

String values are capped at 1 MiB to prevent memory exhaustion from malformed files. `uint64` values for context length and block count are downcast to `uint32` when they fit (some producers write uint64 for these fields).

The parser reads only the header, not tensor data. Parsing a 5 GB model file takes under 1 ms.

## VRAM Monitoring

`GetVRAMInfo()` reads `mem_info_vram_total` and `mem_info_vram_used` from sysfs (`/sys/class/drm/cardN/device/`). It identifies the discrete GPU by selecting the card with the largest VRAM total, which correctly distinguishes the RX 7800 XT (16 GB) from the Ryzen iGPU (2 GB) without hardcoding card numbers.

`Free` is computed as `Total - Used` with a guard against uint64 underflow: if `Used > Total` due to a non-atomic sysfs read during heavy allocation, `Free` is clamped to zero.

## Model Discovery

`DiscoverModels(dir)` globs for `*.gguf` files in a directory, calls `gguf.ReadMetadata()` on each, and returns a `[]ModelInfo` slice. Files that fail to parse are silently skipped.

## go-inference Interface Contract

The package implements two interfaces from `forge.lthn.ai/core/go-inference`:

**inference.Backend**:
- `Name() string` — returns `"rocm"`
- `Available() bool` — /dev/kfd + llama-server present
- `LoadModel(path string, opts ...LoadOption) (TextModel, error)` — spawns llama-server

**inference.TextModel**:
- `Generate(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token]`
- `Chat(ctx context.Context, messages []Message, opts ...GenerateOption) iter.Seq[Token]`
- `ModelType() string` — GGUF architecture string
- `Err() error` — last error from Generate/Chat, mutex-protected
- `Close() error` — SIGTERM/SIGKILL shutdown

Known limitation: `Err()` is a single shared field. With concurrent Generate/Chat calls on the same model, errors from simultaneous callers can overwrite each other (last writer wins). This is a known constraint of the go-inference interface design, not a bug in this package.

`StopTokens []int32` from `GenerateConfig` is ignored. llama-server's OpenAI-compatible API accepts stop sequences as strings, not token IDs, and mapping between them requires a tokeniser. No current consumer of go-rocm uses StopTokens.

## Concurrency and Parallel Slots

llama-server serialises concurrent requests through its slot system. With the default of one slot, simultaneous calls to `Generate()` on the same model are queued server-side. Aggregate throughput still scales because the GPU is not idle during serialised requests.

`inference.WithParallelSlots(n)` passes `--parallel N` to llama-server, enabling true parallel inference across N context slots. Each slot maintains its own KV cache, so VRAM usage scales with `parallelSlots * contextLen`. With 4 slots at ctx=2048 on the RX 7800 XT, the additional VRAM cost is approximately 200 MiB for Gemma3-4B.

## go-inference Ecosystem Position

```
go-inference    — shared TextModel/Backend interfaces (no deps)
     |
go-rocm         — AMD ROCm backend (this package)
go-mlx          — Apple Metal backend (macOS, CGO, Safetensors)
     |
go-ml           — scoring engine, wraps both backends transparently
     |
go-ai           — MCP server + facade, imports go-ml
go-i18n         — grammar engine, may use for batch classification
```

go-rocm registers itself automatically. go-ml selects the appropriate backend at runtime based on `Available()`.
