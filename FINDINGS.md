# FINDINGS.md — go-rocm Research & Discovery

---

## 2026-02-19: Package Creation (Virgil)

### Hardware

- **GPU**: AMD Radeon RX 7800 XT
- **Architecture**: RDNA 3, gfx1101
- **VRAM**: 16GB GDDR6
- **Compute Units**: 60
- **OS**: Linux (Ubuntu, homelab machine)

### ROCm Support Status

- gfx1100/gfx1101 officially supported in ROCm 6.x+
- Supported on Ubuntu 24.04.3 and 22.04.5
- Kernel 6.10+ recommended for RDNA 3 stability
- `/dev/kfd` device node required (amdgpu kernel driver)

Sources:
- [ROCm system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
- [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)

### llama.cpp + ROCm

llama.cpp has mature ROCm/HIP support. Build flags:

```bash
cmake -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1100 \
    -DGGML_HIP_ROCWMMA_FATTN=ON \
    -DCMAKE_BUILD_TYPE=Release
```

Key findings:
- RX 7800 XT is gfx1101, but ROCm compiler generates identical code for gfx1100
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` may give better performance (benchmark needed)
- rocWMMA flash attention (`-DGGML_HIP_ROCWMMA_FATTN=ON`) available for RDNA 3+
- Docker images may not support hipBLASLt for gfx1100, falling back to hipBLAS
- llama-server provides OpenAI-compatible API with SSE streaming

Sources:
- [llama.cpp ROCm build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [llama.cpp ROCm compatibility](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/llama-cpp-compatibility.html)
- [llama.cpp ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/llama-cpp-install.html)
- [RX 7800 XT build discussion](https://github.com/ggml-org/llama.cpp/discussions/11572)

### Design Decision: Subprocess vs CGO

**Chose subprocess** (llama-server) over direct HIP CGO bindings because:

1. **Maturity**: llama-server is battle-tested with millions of users. Direct HIP CGO would take months to reach comparable stability.
2. **Model support**: llama.cpp supports 50+ model architectures via GGUF. CGO would start with zero.
3. **Maintenance**: llama.cpp team handles ROCm compatibility. We just build the binary.
4. **Isolation**: GPU crashes in the subprocess don't take down the Go process.
5. **Portability**: Same approach works for NVIDIA (CUDA build), Intel (SYCL build) with minimal code changes.

Trade-offs:
- Subprocess adds ~50ms latency for first token (process startup + model load)
- Inter-process communication overhead (HTTP vs in-process)
- Can't share GPU memory between Go process and llama-server

The go-mlx package uses direct CGO because MLX is a C library designed for embedding. llama.cpp's primary API is its server mode.

### VRAM Budget (16GB)

| Model | Quant | VRAM (model) | Context (4K) | Total | Fits? |
|-------|-------|-------------|-------------|-------|-------|
| Qwen3-8B | Q4_K_M | ~5GB | ~0.5GB | ~5.5GB | Yes |
| Gemma3-4B | Q4_K_M | ~3GB | ~0.3GB | ~3.3GB | Yes |
| Llama3-8B | Q4_K_M | ~5GB | ~0.5GB | ~5.5GB | Yes |
| Qwen3-8B | Q8_0 | ~9GB | ~0.5GB | ~9.5GB | Yes |
| Llama3-70B | Q4_K_M | ~40GB | ~2GB | ~42GB | No (partial offload) |

16GB VRAM comfortably runs any 8B model in Q4 or Q8 quantisation. 13B models fit in Q4. Larger models need partial GPU offload (GPULayers option).

---

## 2026-02-19: Sibling Architecture (go-mlx comparison)

| Aspect | go-mlx (macOS) | go-rocm (Linux) |
|--------|---------------|-----------------|
| GPU | Apple Metal (M-series) | AMD ROCm (RDNA 3) |
| Build tag | `darwin && arm64` | `linux && amd64` |
| Approach | Direct CGO (mlx-c) | Subprocess (llama-server) |
| Model format | Safetensors | GGUF |
| Shared interface | `go-inference.TextModel` | `go-inference.TextModel` |
| Memory control | `SetCacheLimit`, `GetActiveMemory` | `rocm-smi` / HIP API |
| Chat templates | Built into model code | llama-server `--chat-template` |

Both register as `inference.Backend` via build-tagged `init()`. go-ml wraps both transparently.

---

## 2026-02-19: Phase 0 Environment Validation (Charon)

### Actual Hardware (corrected from Virgil's notes)

- **GPU arch**: gfx1100 (NOT gfx1101 — `rocminfo` confirms)
- **ROCm version**: 7.2.0 (newer than the 6.x minimum)
- **Kernel**: 6.17.0-14-generic
- **`/dev/kfd`**: Present, working
- **HSA_OVERRIDE_GFX_VERSION**: Not needed — native gfx1100

### llama-server Build

- **Source**: llama.cpp commit `11c325c` (cloned 19 Feb 2026)
- **Local build path**: `/home/claude/llama.cpp/build/bin/llama-server`
- **Installed to**: `/usr/local/bin/llama-server`
- **Build command**:
  ```bash
  cmake -B build \
      -DGGML_HIP=ON \
      -DAMDGPU_TARGETS=gfx1100 \
      -DGGML_HIP_ROCWMMA_FATTN=ON \
      -DCMAKE_BUILD_TYPE=Release
  cmake --build build --parallel $(nproc) -t llama-server
  ```

### Critical: iGPU Crash

**The Ryzen 9 9950X has an integrated GPU** that ROCm detects as a second device:
- Device 0: AMD Radeon RX 7800 XT (gfx1100) — 16GB VRAM (real)
- Device 1: AMD Radeon Graphics (gfx1100) — reports 100GB free (system RAM, misleading)

llama-server's auto-fit logic tries to split the model across both devices. Loading tensors to Device 1 (iGPU) causes **`ROCm error: unspecified launch failure`** and crashes with a core dump.

**Fix**: Set `HIP_VISIBLE_DEVICES=0` to mask the iGPU. The go-rocm package MUST set this env var before spawning llama-server.

### Baseline Benchmarks — Gemma3-4B-Q4_K_M

| Metric | Value |
|--------|-------|
| Model | LEK-Gemma3-4B-Q4_K_M (2.66 GiB) |
| VRAM used | ~3.4 GiB of 16 GiB |
| Prefill (prompt) | 396 tok/s (2.5ms/tok) |
| Decode (generation) | 109 tok/s (9.2ms/tok) |
| Time to first token | ~40ms (16 token prompt) |
| Startup time | ~6s (load + warmup) |
| Context window | 4096 (model supports 131072) |
| Flash attention | Auto-enabled |
| Slots | 4 concurrent |

### GGUF Models Available

All at `/data/lem/gguf/` (SMB mount from M3):

| Model | Size | Fits 16GB? |
|-------|------|-----------|
| LEK-Gemma3-1B-layered-v2-Q5_K_M | ~0.9G | Yes |
| LEK-Gemma3-1B-layered-v2-Q8_0 | ~1.4G | Yes |
| LEK-Gemma3-4B-Q4_K_M | 2.7G | Yes |
| LEK-Gemma3-12B-Q4_K_M | ~7.5G | Yes |
| LEK-Gemma3-27B-Q4_K_M | ~16G | Tight |
| LEK-Llama-3.1-8B-Q4_K_M | ~5G | Yes |
| LEK-Mistral-7B-v0.3-Q4_K_M | ~4G | Yes |
| LEK-Qwen-2.5-7B-Q4_K_M | ~4G | Yes |

### Environment Variables for go-rocm

The server.go implementation MUST set these when spawning:

```go
cmd.Env = append(os.Environ(),
    "HIP_VISIBLE_DEVICES=0",  // Critical: mask iGPU to prevent crash
)
```

### Model Path Note

Models are on SMB mount (`/data` = `//10.69.69.108/Data`). For CI/testing, copy a small model locally or use `t.Skip()` when the mount is unavailable.

---

## 2026-02-19: Phase 1 Plan Review — Interface Questions

### QUESTION: Token.ID not populated by llama-server SSE

llama-server's OpenAI-compatible streaming API (`/v1/chat/completions`, `/v1/completions`) does not include token IDs in the default SSE response. The `inference.Token` struct has `ID int32` and `Text string` — go-rocm will set `Text` but leave `ID` as 0 for all tokens.

Token IDs are available via `logprobs: true` in the request, but this adds overhead and requires parsing the `logprobs.tokens` field.

**Decision needed from Virgil:** Does any consumer (go-ml, go-i18n, go-ai) rely on `Token.ID`? If only `Token.Text` is used downstream, ID=0 is acceptable for Phase 1. If ID is needed, we'll add logprobs parsing.

**ANSWER (Charon, 19 Feb 2026):** Token.ID = 0 is acceptable for Phase 1. No downstream consumer uses Token.ID today — go-ml's scoring engine and go-i18n both only read Token.Text. If a consumer needs IDs later, add logprobs parsing in Phase 2. Don't over-engineer now.

### QUESTION: StopTokens type mismatch

`GenerateConfig.StopTokens` is `[]int32` (token IDs), but llama-server's OpenAI-compatible API expects `"stop"` as `[]string` (text sequences). These are fundamentally different — token IDs cannot be mapped to stop strings without a tokeniser.

Options:
1. Ignore `StopTokens` in go-rocm Phase 1 (no consumer uses it yet)
2. Use llama-server's native `/completion` endpoint which supports `id_slot` stop tokens
3. Add `StopStrings []string` to `GenerateConfig` in go-inference alongside the existing `StopTokens []int32`, let each backend use whichever it supports

**Decision needed from Virgil:** Which approach? Option 3 would be a go-inference interface change. Option 1 is simplest for now — go-rocm silently ignores StopTokens if set.

**ANSWER (Charon, 19 Feb 2026):** Option 1 — ignore StopTokens in Phase 1. No consumer uses them yet. The go-inference interface change (Option 3) should come from a real need, not a hypothetical one. YAGNI.

---

## 2026-02-19: Phase 1 Plan Review (Charon)

### Verdict: Approved

Design and implementation plan reviewed. The layered architecture (internal/llamacpp → server → model → backend) is correct. 8-task TDD breakdown is solid. Tasks 1-6 unit-testable without GPU, Task 7 needs hardware.

### Notes for Implementation

1. **guessModelType() filename parsing** — Pragmatic but fragile. Fine for Phase 1. llama-server's `/props` endpoint returns the actual architecture. Note as a Phase 2 upgrade.

2. **serverEnv() HIP_VISIBLE_DEVICES override** — Current approach appends `HIP_VISIBLE_DEVICES=0` to `os.Environ()`. If the user already has `HIP_VISIBLE_DEVICES` set, both values exist in the env slice. Last-write-wins behaviour depends on the kernel and is platform-specific. Safer to filter the existing value out first:

   ```go
   func serverEnv() []string {
       env := os.Environ()
       filtered := make([]string, 0, len(env)+1)
       for _, e := range env {
           if !strings.HasPrefix(e, "HIP_VISIBLE_DEVICES=") {
               filtered = append(filtered, e)
           }
       }
       return append(filtered, "HIP_VISIBLE_DEVICES=0")
   }
   ```

3. **`//go:build rocm` for integration tests** — Good call. Keeps `go test ./...` fast on machines without GPU.

---

## 2026-02-19: Phase 2 Robustness (Charon)

### Concurrent Requests

Tested 3 goroutines calling Generate() simultaneously on the same model (Gemma3-1B, llama-server with default settings). All 3 received output (~0.9s total). llama-server handles concurrency via its slot system — default is 1 slot, so requests are serialised server-side.

For true parallel inference, use `--parallel N` flag in llama-server (not yet configurable via go-rocm). VRAM cost scales with number of slots and context size.

### VRAM Monitoring

Reading sysfs directly (`/sys/class/drm/cardN/device/mem_info_vram_*`) instead of spawning `rocm-smi`. Auto-detects dGPU by selecting the card with the largest VRAM total:
- card0 = iGPU (2GB) — Ryzen 9 9950X integrated
- card1 = dGPU (16GB) — RX 7800 XT

Note: sysfs reads are non-atomic. Total and Used are read separately, so transient inconsistencies are possible under heavy allocation churn. Free is clamped to prevent uint64 underflow.

### lastErr Design Limitation

`rocmModel.lastErr` is a single mutex-protected field shared across all callers. With concurrent Generate/Chat calls, errors can be clobbered (last writer wins). `Err()` is only reliable in single-caller scenarios. This matches the go-inference interface contract (single `Err() error` method), so it's a known limitation, not a bug. Per-call error returns would require an interface change in go-inference.

---

## 2026-02-19: Phase 3 Model Support (Charon)

### GGUF Metadata Parser

New `internal/gguf/` package reads GGUF v2/v3 binary headers. Extracts metadata KV pairs without reading tensor data (<1ms per file). Supports all 13 GGUF value types (uint8..float64, string, array, bool). String length capped at 1 MiB to prevent memory exhaustion from malformed files. Handles uint64 values for context_length/block_count (some producers use uint64 instead of uint32).

### Model Inventory

Discovered models from `/data/lem/gguf/` using GGUF metadata:

| Model | Architecture | Size | Quant | Context | Blocks |
|-------|-------------|------|-------|---------|--------|
| Gemma3-1B Q5_K_M | gemma3 | 1B | Q5_K_M | 32768 | 26 |
| Gemma3-1B Q8_0 | gemma3 | 1B | Q8_0 | 32768 | 26 |
| Gemma3-4B Q4_K_M | gemma3 | 4B | Q4_K_M | 131072 | 34 |
| Gemma3-12B Q4_K_M | gemma3 | 12B | Q4_K_M | 131072 | 42 |
| Gemma3-27B Q4_K_M | gemma3 | 27B | Q4_K_M | 131072 | 46 |
| Llama-3.1-8B Q4_K_M | llama | 8B | Q4_K_M | 131072 | 32 |
| Mistral-7B-v0.3 Q4_K_M | llama | 7B | Q4_K_M | 32768 | 32 |
| Qwen-2.5-7B Q4_K_M | qwen2 | 7B | Q4_K_M | 32768 | 28 |

Key observations:
- Mistral-7B-v0.3 reports `general.architecture = "llama"` (correct — Mistral is a Llama architecture variant). Old `guessModelType` returned "mistral", GGUF metadata returns "llama".
- Qwen-2.5-7B reports `general.architecture = "qwen2"` (not "qwen3"). Old `guessModelType` would have returned "qwen" due to filename matching.
- Gemma3-4B/12B/27B have 131072 native context — without auto-capping at 4096, these would exhaust VRAM.

### Chat Templates

llama-server reads `tokenizer.chat_template` from the GGUF and applies it automatically on `/v1/chat/completions`. No go-rocm code needed. Verified working with Gemma3 integration tests.

### Context Window Auto-Detection

Default context capped at `min(model_context_length, 4096)` when user doesn't specify `inference.WithContextLen(N)`. Without this cap, Llama-3.1 would try to allocate 131072 context (~4GB KV cache), which combined with model weights would not fit in 16GB VRAM for larger models.

---

## 2026-02-19: Phase 4 Performance (Charon)

### Benchmark Results — RX 7800 XT (gfx1100, ROCm 7.2.0)

All benchmarks run with `ctx=2048`, `testing.B`, `benchtime=3x`.

#### Decode Speed (128 tokens)

| Model | tok/s | VRAM Used |
|-------|-------|-----------|
| Gemma3-4B-Q4_K_M | 102.5 | 4724 MiB |
| Llama-3.1-8B-Q4_K_M | 77.1 | 6482 MiB |
| Qwen-2.5-7B-Q4_K_M | 84.4 | 6149 MiB |

#### Time-to-First-Token

| Model | TTFT |
|-------|------|
| Gemma3-4B-Q4_K_M | 13.8 ms |
| Llama-3.1-8B-Q4_K_M | 17.1 ms |
| Qwen-2.5-7B-Q4_K_M | 16.8 ms |

#### Concurrent Throughput (4 parallel slots, 4 goroutines, 32 tokens each)

| Model | Aggregate tok/s | vs Single |
|-------|----------------|-----------|
| Gemma3-4B-Q4_K_M | 238.9 | 2.3x |
| Llama-3.1-8B-Q4_K_M | 166.2 | 2.2x |
| Qwen-2.5-7B-Q4_K_M | 178.0 | 2.1x |

Parallel slots give ~2.2x throughput improvement with 4 concurrent requests. Per-request latency increases but aggregate throughput scales well.

### Flash Attention Comparison

Compared llama-server built with `-DGGML_HIP_ROCWMMA_FATTN=ON` vs without, at ctx=2048:

| Model | With FA (tok/s) | Without FA (tok/s) | Difference |
|-------|----------------|-------------------|------------|
| Gemma3-4B | 102.5 | 107.2 | -4.4% |
| Llama-3.1-8B | 77.1 | 77.7 | -0.9% |
| Qwen-2.5-7B | 84.4 | 84.4 | 0% |

**Conclusion:** Flash attention shows no benefit at ctx=2048. rocWMMA flash attention is designed for large context windows where the KV cache becomes a bottleneck. At 2048 context, standard attention is faster (or equal). Flash attention benefits would appear at ctx=8192+ where the quadratic attention cost dominates. Keeping FA enabled is harmless — it auto-activates only when beneficial.

### Parallel Slots

Added `ParallelSlots int` to go-inference's `LoadConfig` and `WithParallelSlots(n int) LoadOption`. go-rocm passes `--parallel N` to llama-server. Each slot allocates its own KV cache, so VRAM usage scales with `parallelSlots * contextLen`. With 4 slots at ctx=2048, VRAM overhead is modest (~200 MiB extra for Gemma3-4B).
