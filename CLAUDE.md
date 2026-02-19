# CLAUDE.md

## What This Is

AMD ROCm GPU inference for Linux. Module: `forge.lthn.ai/core/go-rocm`

Implements `inference.Backend` and `inference.TextModel` (from `core/go-inference`) using llama.cpp compiled with HIP/ROCm. Targets AMD RDNA 3+ GPUs.

## Target Hardware

- **GPU**: AMD Radeon RX 7800 XT (gfx1100, RDNA 3, 16GB VRAM) — NOTE: gfx1100 not gfx1101
- **OS**: Ubuntu 24.04 LTS (linux/amd64)
- **ROCm**: 6.x+ (gfx1100/gfx1101 officially supported)
- **Kernel**: 6.10+ recommended for RDNA 3 stability

## Commands

```bash
go test ./...                       # Run all tests (stubs on non-Linux)
go test -tags rocm ./...            # Run with ROCm integration tests

# On the Linux homelab:
go test -v -run TestROCm ./...      # Full GPU tests
```

## Architecture

```
go-rocm (this package)
├── rocm.go              Package doc
├── register_rocm.go     //go:build linux && amd64 — auto-registers via init()
├── rocm_stub.go         //go:build !linux || !amd64 — ROCmAvailable() false
├── backend.go           inference.Backend implementation
├── model.go             inference.TextModel implementation (TODO)
├── server.go            llama-server lifecycle management (TODO)
└── internal/
    └── llamacpp/        llama-server HTTP client (TODO)
        ├── client.go    OpenAI-compatible API client
        └── health.go    Health check + readiness probe
```

### How It Works

1. `LoadModel()` spawns `llama-server` (llama.cpp) as a subprocess
2. llama-server loads the GGUF model onto the AMD GPU via HIP/ROCm
3. `Generate()` / `Chat()` make HTTP requests to llama-server's OpenAI-compatible API
4. Token streaming via SSE (Server-Sent Events) from llama-server
5. `Close()` sends SIGTERM to llama-server, waits for clean exit

This is the subprocess approach (not CGO). It's simpler, more maintainable, and llama.cpp's server mode is battle-tested.

### Dependencies

- `forge.lthn.ai/core/go-inference` — shared TextModel/Backend interfaces
- llama-server binary (external, not Go dependency) built with `-DGGML_HIP=ON`

## Building llama-server with ROCm

```bash
# On the Linux homelab:
sudo apt install rocm-dev rocm-libs  # ROCm 6.x

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1100 \
    -DGGML_HIP_ROCWMMA_FATTN=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel -t llama-server

# Binary at build/bin/llama-server
# Copy to /usr/local/bin/ or set ROCM_LLAMA_SERVER_PATH
```

### Performance Tip

The RX 7800 XT is gfx1101 but the ROCm compiler generates identical code for gfx1100. Setting:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```
...gives better performance on some ROCm versions. Benchmark both.

## Coding Standards

- UK English
- Tests: testify assert/require
- Conventional commits
- Co-Author: `Co-Authored-By: Virgil <virgil@lethean.io>`
- Licence: EUPL-1.2

## Coordination

- **Virgil** (core/go) is the orchestrator — writes tasks here
- **go-mlx Claude** is the sibling — Metal backend on macOS, same interface contract
- **go-inference** defines the shared TextModel/Backend interfaces both backends implement
- **go-ml** wraps both backends into the scoring engine

## Task Queue

See `TODO.md` for prioritised work.
See `FINDINGS.md` for research notes.
