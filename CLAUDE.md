# go-rocm — AMD ROCm GPU Inference

## What This Is

AMD ROCm GPU inference for Linux via the native package-first ROCm backend.
Module: `dappco.re/go/rocm`. The managed `llama-server` subprocess path is
legacy-only behind the `rocm_legacy_server` build tag.

Implements `inference.Backend` and `inference.TextModel` (from `core/go-inference`) using llama.cpp compiled with `-DGGML_HIP=ON`. Targets AMD RDNA 2+ GPUs (tested on Radeon RX 7800 XT, gfx1100).

Sibling to `go-mlx` (Metal on macOS). Both expose the same interface; users select at runtime based on `Available()`.

## Key Facts

- **Native model:** default Linux builds load model packs and route through HIP-owned buffers
- **GGUF parser:** Reads model metadata (v2/v3) without loading tensors — enables fast discovery
- **VRAM monitoring:** sysfs-based (no ROCm runtime library dependency)
- **GPU selection:** native live tests pin the RX 7800 XT with `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85`; do not assume ordinal 0 is the dGPU
- **Auto-register:** `init()` registers backend into `inference.Register()` on linux && amd64
- **Platform stubs:** Exports no-op funcs on non-Linux/amd64 to avoid build failures
- **Error wrapping:** All errors use `coreerr.E(scope, msg, cause)` from `go-log`

## Hardware & OS

| Component | Value |
|-----------|-------|
| GPU | Radeon RX 7800 XT (gfx1100, RDNA 3, 16 GB) |
| CPU | Ryzen 9 9950X |
| OS | Ubuntu 24.04 LTS |
| ROCm | 7.2.0 |
| Kernel | 6.17.0 |

## Architecture

```
dappco.re/go/rocm/
├── Public:
│   ├── rocm.go              [VRAMInfo, ModelInfo types]
│   ├── discover.go          [DiscoverModels(dir) -> []ModelInfo]
│   ├── register_rocm.go     [init() register]
│
├── Backend/Model (linux && amd64):
│   ├── backend.go           [rocmBackend impl]
│   ├── model.go             [rocmModel impl, metrics, streaming]
│   ├── server.go            [subprocess lifecycle, port mgmt]
│   ├── vram.go              [GetVRAMInfo() via sysfs]
│   ├── rocm_stub.go         [stubs for other platforms]
│
└── Internal:
    ├── internal/gguf/
    │   └── gguf.go          [GGUF v2/v3 binary header parser]
    │
    └── internal/llamacpp/
        ├── client.go        [HTTP client, Complete, ChatComplete]
        └── health.go        [/health endpoint polling]
```

## Critical Rules

1. **Pin the real dGPU:** Native HIP tests and benchmarks use `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85`. Do not rely on device/card ordinal 0 because it may resolve to the onboard GPU. The old `serverEnv()` `HIP_VISIBLE_DEVICES=0` rule applies only to the legacy llama-server tag.

2. **Platform-specific:** Build tags `linux && amd64` for GPU code. Stubs on other platforms prevent build errors.

3. **Subprocess isolation:** llama-server is not trusted. Runs at default perms, minimal env, auto-killed on exit.

4. **Error scope:** All errors use `coreerr.E()`. No `fmt.Errorf`, no `errors.New`, no `log` package.

5. **Banned imports:** `fmt`, `log`, `errors`, `os/exec` use their core.* equivalents. (Note: `os` used directly for file/env ops, justified by GPU module weight constraints.)

6. **Metrics best-effort:** VRAM stats read non-atomically from sysfs. Under heavy churn, transient gaps expected. Recording is not real-time.

## Spec Index

See `/sessions/vibrant-sharp-fermat/mnt/plans/code/core/go/rocm/RFC.md`:

- **§1–2:** Overview & package layout
- **§3:** Type definitions (VRAMInfo, ModelInfo, rocmBackend, rocmModel, server)
- **§4:** Inference pipeline (Load, Generate, Chat, metrics)
- **§5:** GGUF parser internals
- **§6:** llama-server HTTP bridge
- **§7–9:** VRAM discovery, model discovery, platform support
- **§10–16:** Error handling, config, quantisation, design notes, cross-refs

## Working Commands

```bash
# Unit tests (no GPU required)
go test ./...

# Integration tests + benchmarks (GPU required, gfx1100)
go test -tags rocm ./...

# Full GPU tests only
go test -tags rocm -v -run TestROCm ./...

# Benchmarks
go test -tags rocm -bench=. -benchtime=3x ./...

# Format & lint
go fmt ./...
```

## Building llama-server

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1100 \
    -DGGML_HIP_ROCWMMA_FATTN=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc) -t llama-server
sudo cp build/bin/llama-server /usr/local/bin/llama-server
```

## Coordination

- **Virgil** (forge.lthn.ai/core) — orchestrator, task writer, PR reviewer
- **go-mlx** — sibling Metal backend (same interface contract)
- **go-inference** — shared TextModel/Backend interface definitions
- **go-ml** — scoring engine wrapping both backends
- **LEM training** — uses go-rocm for model eval on Charon homelab

## Test Naming

Format: `TestFilename_Function_{Good,Bad,Ugly}` — all three categories mandatory.

Example: `TestModel_Generate_Good`, `TestModel_Generate_Bad`, `TestModel_Generate_Ugly`.

## Commit Style

```
type(scope): description

Co-Authored-By: Virgil <virgil@lethean.io>
```

Example: `feat(rocm): add VRAM monitoring via sysfs`
