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

- [ ] **GPU detection** — Implement `Available()` in backend.go. Check: `/dev/kfd` exists (ROCm kernel driver), `rocm-smi` detects GPU, llama-server binary is findable (PATH or `ROCM_LLAMA_SERVER_PATH` env).
- [ ] **Server lifecycle** — Create `server.go`: spawn llama-server with `--model`, `--port` (random free port), `--n-gpu-layers` (from LoadConfig.GPULayers), `--ctx-size` (from LoadConfig.ContextLen). Wait for `/health` endpoint. Handle SIGTERM on Close().
- [ ] **HTTP client** — Create `internal/llamacpp/client.go`: POST `/v1/chat/completions` with streaming (SSE). Parse `data: {"choices":[{"delta":{"content":"..."}}]}` into inference.Token stream.
- [ ] **TextModel implementation** — Create `model.go`: implement inference.TextModel wrapping the HTTP client. Generate() sends single-turn prompt, Chat() sends multi-turn messages. Both stream via iter.Seq[Token]. Err() returns last error.
- [ ] **Integration test** — Test end-to-end: LoadModel → Generate → tokens received → Close. Requires GGUF model on disk. Use `t.Skip()` when model/GPU unavailable.

## Phase 2: Robustness

- [ ] **Server crash recovery** — If llama-server dies mid-generation, detect via process exit, return error via Err(), allow re-load.
- [ ] **Port conflict handling** — If the random port is taken, retry with a different port.
- [ ] **Graceful shutdown** — On context cancellation, stop the current request cleanly (close SSE stream), don't kill the server. Only Close() kills the server.
- [ ] **Memory monitoring** — Use `rocm-smi --showmeminfo vram` or HIP API to report VRAM usage. Expose via package-level functions (like go-mlx's GetActiveMemory).
- [ ] **Concurrent requests** — llama-server supports concurrent slots. Test with multiple goroutines calling Generate() simultaneously. Document max concurrency.

## Phase 3: Model Support

- [ ] **GGUF model discovery** — Implement model path scanning: find .gguf files, parse metadata (model name, params, quant level, size). Return structured inventory.
- [ ] **Chat templates** — llama-server handles chat templates natively via `--chat-template`. Verify Gemma3, Qwen3, Llama3 templates work. If not, add template formatting in model.go.
- [ ] **Context window sizing** — Auto-detect optimal context window from model metadata. Default to 4096 if unknown.

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
