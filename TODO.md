# TODO.md — go-rocm Task Queue

Dispatched from core/go orchestration. Pick up tasks in order.

---

## Phase 0: Environment Setup (on Linux homelab)

- [x] **Install ROCm 6.x** — ROCm 7.2.0 already installed. `rocm-smi` shows RX 7800 XT (gfx1100). Kernel 6.17.0. (Charon, 19 Feb 2026)
- [x] **Build llama-server with HIP** — Built from llama.cpp `11c325c`. Installed to `/usr/local/bin/llama-server`. (Charon, 19 Feb 2026)
- [x] **Test manual inference** — Gemma3-4B-Q4_K_M: 109 tok/s decode, 396 tok/s prefill. See FINDINGS.md for full results. (Charon, 19 Feb 2026)
- [x] **HSA_OVERRIDE_GFX_VERSION benchmark** — N/A: GPU is actually gfx1100 (not gfx1101 as Virgil noted). No override needed. (Charon, 19 Feb 2026)

### Critical Discovery: iGPU Crash

The Ryzen 9 9950X iGPU shows up as ROCm Device 1, reports 100GB free (system RAM), and crashes llama-server when it tries to split tensors across devices. **`HIP_VISIBLE_DEVICES=0` is REQUIRED** when spawning llama-server. See FINDINGS.md for details.

## Phase 1: Core Implementation

- [x] **GPU detection** — `Available()` checks `/dev/kfd` + `findLlamaServer()`. Commit `1d8d65f`. (19 Feb 2026)
- [x] **Server lifecycle** — `server.go`: spawn, health poll (100ms/60s timeout), SIGTERM/SIGKILL shutdown. `serverEnv()` filters HIP_VISIBLE_DEVICES. Commit `9aa7f62`. (19 Feb 2026)
- [x] **HTTP client** — `internal/llamacpp/`: health check, SSE parser, ChatComplete + Complete with `iter.Seq[string]`. Commits `3c75677`, `def3167`. (19 Feb 2026)
- [x] **TextModel implementation** — `model.go`: wraps llamacpp client, maps inference types, mutex-protected Err(). Commit `a8c4947`. (19 Feb 2026)
- [x] **Integration test** — 3 tests (Generate, Chat, ContextCancellation) with Gemma3-1B on RX 7800 XT. All pass. Commit `0e68d71`. (19 Feb 2026)

## Phase 2: Robustness

- [x] **Server crash recovery** — `server.alive()` detects process exit; Generate/Chat return error immediately if dead. Commits `2c4966e`, `c07f37a`. (Charon, 19 Feb 2026)
- [x] **Port conflict handling** — `startServer()` retries up to 3 times with new port on process exit. Only retries on exit, not timeout. Commits `c50a8e9`, `b7342ec`. (Charon, 19 Feb 2026)
- [x] **Graceful shutdown** — Already worked in Phase 1. Integration test confirms server survives context cancellation and generates again. Commit `a6e647c`. (Charon, 19 Feb 2026)
- [x] **Memory monitoring** — `GetVRAMInfo()` reads sysfs, auto-detects dGPU by largest VRAM. Uint64 underflow guard on Free. Commits `501de83`, `954c570`. (Charon, 19 Feb 2026)
- [x] **Concurrent requests** — 3 goroutines calling Generate() simultaneously all get output. llama-server serialises via 1 slot (default). Commit `a6e647c`. (Charon, 19 Feb 2026)

## Phase 3: Model Support

- [x] **GGUF metadata parser** — `internal/gguf/` reads GGUF v2/v3 binary headers. Extracts architecture, name, file type, size label, context length, block count. String length limits for malformed input protection. Commit `c7c9389`. (Charon, 19 Feb 2026)
- [x] **GGUF model discovery** — `DiscoverModels(dir)` scans directory for `.gguf` files, parses metadata via GGUF parser, returns `[]ModelInfo`. Commit `af23565`. (Charon, 19 Feb 2026)
- [x] **LoadModel enrichment** — Replaced `guessModelType` with GGUF metadata for real architecture. Auto-caps context at 4096 when user doesn't specify. Commit `2c77f6f`. (Charon, 19 Feb 2026)
- [x] **Chat templates** — llama-server reads `tokenizer.chat_template` from GGUF natively on `/v1/chat/completions`. No go-rocm code needed. Verified with Gemma3 integration test. (Charon, 19 Feb 2026)
- [x] **Context window sizing** — Auto-detected from GGUF metadata. Default caps at `min(model_context_length, 4096)` to prevent VRAM exhaustion. (Charon, 19 Feb 2026)

## Phase 4: Performance

- [ ] **Benchmark suite** — Measure: tokens/sec (prefill + decode), time-to-first-token, VRAM usage, for Qwen3-8B-Q4, Gemma3-4B, Llama3-8B on the RX 7800 XT. Compare with mlx on M3 Ultra.
- [ ] **Flash attention** — Verify `-DGGML_HIP_ROCWMMA_FATTN=ON` gives real speedup on gfx1100. Benchmark with and without.
- [ ] **Batch inference** — llama-server supports multiple slots for concurrent inference. Test parallel prompts for go-i18n's batch classification use case.

## Phase 5: Alternative Backends

- [ ] **Direct HIP/CGO** — Evaluate whether direct HIP CGO bindings (like go-mlx does for Metal) would be worth the effort. Only if llama-server subprocess becomes a bottleneck.
- [ ] **vLLM backend** — vLLM supports ROCm and has better batching. Could be an alternative subprocess backend for high-throughput scenarios.

---

## Model Inventory (on Linux homelab)

Download to `/data/models/` (or wherever the homelab stores data):
- [ ] Qwen3-8B-Q4_K_M.gguf (~5GB, fits 16GB VRAM with room for context)
- [ ] Gemma3-4B-Q4_K_M.gguf (~3GB)
- [ ] Llama-3.1-8B-Q4_K_M.gguf (~5GB)

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ROCM_LLAMA_SERVER_PATH` | `llama-server` (PATH lookup) | Path to llama-server binary |
| `HIP_VISIBLE_DEVICES` | `0` (MUST set) | Mask iGPU — Ryzen 9 iGPU crashes llama-server |
| `HSA_OVERRIDE_GFX_VERSION` | unset | Not needed (GPU is native gfx1100) |
| `ROCM_MODEL_DIR` | none | Default directory for model discovery |

## Upstream Dependencies

- **go-inference** defines the TextModel/Backend interfaces this package implements
- **go-ml** will wrap this backend (Virgil creates backend_rocm.go when the API is ready)
- **go-i18n** may use this for batch classification on Linux (Phase 4)

## Workflow

1. Virgil in core/go writes tasks here after research
2. This repo's session (on Linux homelab) picks up tasks in phase order
3. Mark `[x]` when done, note commit hash
4. New discoveries → add tasks, flag in FINDINGS.md
