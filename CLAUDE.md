# CLAUDE.md

## What This Is

AMD ROCm GPU inference for Linux. Module: `forge.lthn.ai/core/go-rocm`

Implements `inference.Backend` and `inference.TextModel` (from `core/go-inference`) using llama.cpp compiled with HIP/ROCm. Targets AMD RDNA 3+ GPUs.

## Target Hardware

- **GPU**: AMD Radeon RX 7800 XT (gfx1100, RDNA 3, 16 GB VRAM) — confirmed gfx1100, not gfx1101
- **OS**: Ubuntu 24.04 LTS (linux/amd64)
- **ROCm**: 7.2.0 installed
- **Kernel**: 6.17.0

## Commands

```bash
go test ./...                       # Unit tests (no GPU required)
go test -tags rocm ./...            # Integration tests + benchmarks (GPU required)
go test -tags rocm -v -run TestROCm ./...   # Full GPU tests only
go test -tags rocm -bench=. -benchtime=3x ./...  # Benchmarks
```

## Architecture

See `docs/architecture.md` for full detail.

```
go-rocm/
├── backend.go           inference.Backend (linux && amd64)
├── model.go             inference.TextModel (linux && amd64)
├── server.go            llama-server subprocess lifecycle
├── vram.go              VRAM monitoring via sysfs
├── discover.go          GGUF model discovery
├── register_rocm.go     auto-registers via init() (linux && amd64)
├── rocm_stub.go         stubs for non-linux/non-amd64
└── internal/
    ├── llamacpp/        llama-server HTTP client + health check
    └── gguf/            GGUF v2/v3 binary metadata parser
```

## Critical: iGPU Crash

The Ryzen 9 9950X iGPU appears as ROCm Device 1. llama-server crashes trying to split tensors across it. `serverEnv()` always sets `HIP_VISIBLE_DEVICES=0`. Do not remove or weaken this.

## Building llama-server with ROCm

```bash
cmake -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1100 \
    -DGGML_HIP_ROCWMMA_FATTN=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc) -t llama-server
sudo cp build/bin/llama-server /usr/local/bin/llama-server
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ROCM_LLAMA_SERVER_PATH` | PATH lookup | Path to llama-server binary |
| `HIP_VISIBLE_DEVICES` | overridden to `0` | Always forced to 0 — do not rely on ambient value |

## Coding Standards

- UK English
- Tests: testify assert/require
- Build tags: `linux && amd64` for GPU code, `rocm` for integration tests
- Conventional commits
- Co-Author: `Co-Authored-By: Virgil <virgil@lethean.io>`
- Licence: EUPL-1.2

## Coordination

- **Virgil** (core/go) is the orchestrator — writes tasks and reviews PRs
- **go-mlx** is the sibling — Metal backend on macOS, same interface contract
- **go-inference** defines the shared TextModel/Backend interfaces both backends implement
- **go-ml** wraps both backends into the scoring engine

## Documentation

- `docs/architecture.md` — component design, data flow, interface contracts
- `docs/development.md` — prerequisites, test commands, benchmarks, coding standards
- `docs/history.md` — completed phases, commit hashes, known limitations
- `docs/plans/` — phase design documents (read-only reference)
