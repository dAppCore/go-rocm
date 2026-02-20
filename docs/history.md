# go-rocm Project History

## Origin

go-rocm was created on 19 February 2026 by Virgil (orchestrator) as the AMD GPU backend for the go-inference ecosystem. The sibling package go-mlx provides the same interface on macOS using Apple Metal and direct CGO; go-rocm targets the Linux homelab's AMD Radeon RX 7800 XT.

The package was built by Charon (test coverage and build agent, running on the Linux homelab) in a single day across four phases: environment validation, core implementation, robustness, model support, and performance tuning.

---

## Phase 0: Environment Validation (19 Feb 2026)

**Purpose**: Confirm the homelab hardware, ROCm installation, and llama.cpp build before writing any Go code.

**Findings:**

- GPU architecture confirmed as gfx1100 via `rocminfo`. Virgil's initial notes stated gfx1101; the physical hardware is gfx1100. No `HSA_OVERRIDE_GFX_VERSION` override is required.
- ROCm version: 7.2.0 (minimum required is 6.x).
- Kernel: 6.17.0-14-generic.
- llama.cpp built from commit `11c325c` with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DGGML_HIP_ROCWMMA_FATTN=ON`. Binary installed to `/usr/local/bin/llama-server`.

**Critical discovery: iGPU crash**

The Ryzen 9 9950X has an integrated GPU that ROCm detects as a second device:
- Device 0: RX 7800 XT (gfx1100), 16 GB VRAM
- Device 1: Radeon Graphics iGPU (gfx1100), reports ~100 GB free (system RAM)

llama-server's auto-fit logic splits the model across both devices. Loading tensors to Device 1 triggers `ROCm error: unspecified launch failure` and a core dump. The fix is `HIP_VISIBLE_DEVICES=0`, which must be set unconditionally when spawning llama-server.

**Baseline benchmark (Gemma3-4B-Q4_K_M):**

| Metric | Value |
|--------|-------|
| Prefill speed | 396 tok/s |
| Decode speed | 109 tok/s |
| Time to first token | ~40 ms (16-token prompt) |
| Server startup | ~6 s |
| VRAM used | ~3.4 GB of 16 GB |

---

## Phase 1: Core Implementation (19 Feb 2026)

**Commits**: `1d8d65f`, `9aa7f62`, `3c75677`, `def3167`, `a8c4947`, `0e68d71`

**GPU detection** (`1d8d65f`): `Available()` checks `/dev/kfd` and `findLlamaServer()`. Returns false if either is absent. `findLlamaServer()` checks `ROCM_LLAMA_SERVER_PATH` env var first, then PATH.

**Server lifecycle** (`9aa7f62`): `server.go` implements `startServer()`, `waitReady()`, and `stop()`. Health polling at 100ms intervals with a 60-second startup timeout. Graceful shutdown sends SIGTERM, waits 5 seconds, then SIGKILL. `serverEnv()` filters and overrides `HIP_VISIBLE_DEVICES` using a filter-then-append pattern to avoid duplicate env var entries (last-write-wins is platform-specific).

**HTTP client** (`3c75677`, `def3167`): `internal/llamacpp/` provides `Client` with `Health()`, `ChatComplete()`, and `Complete()`. Both completion methods return `(iter.Seq[string], func() error)`. The SSE parser reads `data: ` prefixed lines from the response body using a `bufio.Scanner`, stops at `[DONE]`, and propagates I/O errors via a pointer.

**TextModel implementation** (`a8c4947`): `model.go` wraps the server and client. `Generate()` calls `/v1/completions`; `Chat()` calls `/v1/chat/completions`. Both check `server.alive()` before dispatching and record errors in `lastErr` under a mutex.

**Integration tests** (`0e68d71`): `TestROCm_LoadAndGenerate`, `TestROCm_Chat`, `TestROCm_ContextCancellation` all pass on the RX 7800 XT using Gemma3-1B. Tests gated behind `//go:build rocm`.

**Design decisions recorded in FINDINGS.md:**

- `Token.ID` left as zero; llama-server's streaming API does not return token IDs. No downstream consumer uses the ID field.
- `StopTokens []int32` silently ignored; the llama-server API expects stop sequences as strings, not token IDs. YAGNI.

---

## Phase 2: Robustness (19 Feb 2026)

**Commits**: `2c4966e`, `c07f37a`, `c50a8e9`, `b7342ec`, `a6e647c`, `501de83`, `954c570`

**Server crash recovery** (`2c4966e`, `c07f37a`): `server.alive()` reads from the `exited` channel non-blockingly. `Generate()` and `Chat()` return an empty iterator immediately if the server has died, recording the exit error in `lastErr`.

**Port conflict handling** (`c50a8e9`, `b7342ec`): `startServer()` retries up to 3 times with a fresh port on process exit during startup. Timeouts are not retried (a stuck server is a distinct failure mode from a port conflict).

**Graceful shutdown** (`a6e647c`): Integration test `TestROCm_GracefulShutdown` confirms the server survives a mid-stream context cancel and accepts subsequent Generate calls. Already worked from Phase 1; integration test added to prevent regression.

**VRAM monitoring** (`501de83`, `954c570`): `GetVRAMInfo()` reads sysfs (`/sys/class/drm/cardN/device/mem_info_vram_*`). Selects the dGPU by highest total VRAM, correctly distinguishing the RX 7800 XT (16 GB) from the Ryzen iGPU (2 GB) without hardcoding card numbers. Uint64 underflow guard: `Free` is clamped to zero if `Used > Total` due to non-atomic sysfs reads.

**Concurrent requests** (`a6e647c`): Three goroutines calling `Generate()` simultaneously all receive output. llama-server serialises via its default single-slot configuration. No Go-level locking needed on the model for concurrent Generate calls.

**Known limitation recorded**: `Err()` is a single shared field. Concurrent callers can overwrite each other's errors. This matches the go-inference interface contract and is not a bug.

---

## Phase 3: Model Support (19 Feb 2026)

**Commits**: `c7c9389`, `af23565`, `2c77f6f`

**GGUF metadata parser** (`c7c9389`): `internal/gguf/` reads GGUF v2/v3 binary headers. Extracts architecture, name, file type, size label, context length, and block count without reading tensor data. Supports all 13 GGUF value type codes. String length capped at 1 MiB. Handles uint64 values for context_length/block_count (some producers use uint64 instead of uint32). Architecture-specific keys are collected as candidates and resolved after `general.architecture` is known, handling the case where architecture-specific keys appear before the architecture key in the KV stream.

**Model discovery** (`af23565`): `DiscoverModels(dir)` globs for `*.gguf` files, parses each via the GGUF parser, and returns `[]ModelInfo`. Unparseable files are skipped silently.

**LoadModel enrichment** (`2c77f6f`): Replaced filename-based architecture guessing with GGUF metadata. `meta.Architecture` is now set from `general.architecture`, which is more accurate: Mistral-7B-v0.3 correctly reports `"llama"` (not `"mistral"`), and Qwen-2.5-7B correctly reports `"qwen2"`. Context auto-capped at `min(model_context_length, 4096)` when the caller does not specify a context length, preventing VRAM exhaustion on models with 128K+ native context (Gemma3-4B/12B/27B and Llama-3.1-8B all have 131072-token native context).

**Chat templates**: Confirmed that llama-server reads `tokenizer.chat_template` from GGUF and applies it on `/v1/chat/completions`. No go-rocm code required.

**Model inventory discovered** (at `/data/lem/gguf/`):

| Model | Architecture | Quant | Context |
|-------|-------------|-------|---------|
| Gemma3-1B-layered-v2 | gemma3 | Q5_K_M / Q8_0 | 32768 |
| Gemma3-4B | gemma3 | Q4_K_M | 131072 |
| Gemma3-12B | gemma3 | Q4_K_M | 131072 |
| Gemma3-27B | gemma3 | Q4_K_M | 131072 |
| Llama-3.1-8B | llama | Q4_K_M | 131072 |
| Mistral-7B-v0.3 | llama | Q4_K_M | 32768 |
| Qwen-2.5-7B | qwen2 | Q4_K_M | 32768 |

---

## Phase 4: Performance (19 Feb 2026)

**Commits**: `870ee23` (benchmarks), `3719734` (go-inference: ParallelSlots), `72120bb` (go-rocm: --parallel support)

**Benchmark suite** (`870ee23`): Three benchmarks gated behind `//go:build rocm`:
- `BenchmarkDecode` — 128-token generation, reports tok/s
- `BenchmarkTTFT` — single-token generation, reports µs/first-tok
- `BenchmarkConcurrent` — 4 goroutines, 4 parallel slots, reports tok/s-aggregate

All three run across Gemma3-4B, Llama3.1-8B, and Qwen2.5-7B. Model load time is excluded via `b.StopTimer()` / `b.StartTimer()`.

**Flash attention comparison**: llama-server built with and without `-DGGML_HIP_ROCWMMA_FATTN=ON` at ctx=2048. No significant difference (≤4.4% variation, within noise). rocWMMA flash attention is designed for large context windows where the KV cache dominates. At ctx=2048, standard attention is as fast or faster. Flash attention auto-activates only when beneficial and does not degrade performance at small context sizes. The flag remains enabled in the build configuration.

**Parallel slots** (`3719734`, `72120bb`): `ParallelSlots int` added to go-inference's `LoadConfig`. `inference.WithParallelSlots(n)` passes `--parallel N` to llama-server. Aggregate throughput with 4 slots at ctx=2048:

| Model | Single-slot tok/s | 4-slot aggregate tok/s | Ratio |
|-------|------------------|----------------------|-------|
| Gemma3-4B-Q4_K_M | 102.5 | 238.9 | 2.3x |
| Llama-3.1-8B-Q4_K_M | 77.1 | 166.2 | 2.2x |
| Qwen-2.5-7B-Q4_K_M | 84.4 | 178.0 | 2.1x |

---

## Known Limitations

**Token IDs**: `inference.Token.ID` is always zero. llama-server's OpenAI-compatible streaming API does not return token IDs. Adding token IDs would require `logprobs: true` in the request and additional parsing overhead. No current consumer uses token IDs.

**StopTokens**: `GenerateConfig.StopTokens []int32` is ignored. llama-server's `/v1/completions` and `/v1/chat/completions` endpoints accept stop sequences as strings (`"stop": [...]`), not token IDs. Mapping between them requires a tokeniser that is not available in this package. No current consumer uses StopTokens.

**Err() concurrency**: `rocmModel.Err()` returns the last error from any Generate/Chat call. With multiple concurrent callers, errors can be overwritten. The single `Err() error` method is an go-inference interface constraint, not a go-rocm decision. Per-call error returns would require an interface change in go-inference.

**VRAM reads are non-atomic**: `GetVRAMInfo()` reads `mem_info_vram_total` and `mem_info_vram_used` in two separate sysfs reads. Under heavy VRAM allocation churn, transient inconsistency is possible. `Free` is clamped to zero to prevent uint64 underflow.

**Model directory**: Models are on an SMB mount (`/data` = `//10.69.69.108/Data`). Integration tests and benchmarks skip when the mount is unavailable. For offline testing, copy a small model (the 1B Q5_K_M is approximately 0.9 GB) to a local path and update the `testModel` constant in `rocm_integration_test.go`.

**Single-model-per-server**: Each `rocmModel` owns exactly one llama-server subprocess. Loading multiple models simultaneously requires multiple `LoadModel` calls, each consuming its own VRAM share. There is no shared server or model-switching mechanism.

---

## Future Considerations

**Direct HIP CGO** (Phase 5, unscheduled): Direct HIP CGO bindings would eliminate the HTTP overhead and process boundary. Only worth pursuing if the subprocess approach becomes a measurable bottleneck. Estimated cost: months of implementation to match llama.cpp's model support breadth.

**vLLM backend** (Phase 5, unscheduled): vLLM supports ROCm and provides better batching semantics for high-throughput scenarios. Would be a parallel subprocess backend alongside llama-server, selectable via configuration.

**Model-switching**: The current design loads one model per server instance. A pool-based approach could share llama-server instances across model loads, though this would require llama-server to support hot-swapping models (it does not currently).

**go-i18n integration**: go-i18n may use go-rocm for batch text classification on the Linux homelab once Phase 2 of go-i18n is unblocked. The `WithParallelSlots` option makes the backend well-suited for batch workloads.
