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
