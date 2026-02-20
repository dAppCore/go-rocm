# go-rocm Development Guide

## Prerequisites

### Hardware

- AMD GPU with ROCm support. Tested hardware: AMD Radeon RX 7800 XT (gfx1100, RDNA 3, 16 GB VRAM)
- Linux, amd64. The package does not build or run on any other platform

### Operating System

- Ubuntu 24.04 LTS (recommended; also supported: Ubuntu 22.04.5)
- Kernel 6.10+ recommended for RDNA 3 stability. The homelab currently runs 6.17.0
- The amdgpu kernel driver must be loaded (`/dev/kfd` must be present)

### ROCm

Install ROCm 6.x or later. ROCm 7.2.0 is installed on the homelab:

```bash
sudo apt install rocm-dev rocm-libs
rocm-smi           # verify GPU is detected
rocminfo           # verify gfx architecture
```

Confirm `/dev/kfd` exists and is accessible to your user. Add yourself to the `render` and `video` groups if needed:

```bash
sudo usermod -aG render,video $USER
```

### llama-server

llama-server must be built from llama.cpp with HIP/ROCm support. The package does not ship or download the binary.

**Build steps** (from the homelab):

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1100 \
    -DGGML_HIP_ROCWMMA_FATTN=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel $(nproc) -t llama-server
```

The production binary on the homelab was built from commit `11c325c` (cloned 19 Feb 2026). Install to PATH:

```bash
sudo cp build/bin/llama-server /usr/local/bin/llama-server
llama-server --version
```

Alternatively, set `ROCM_LLAMA_SERVER_PATH` to the full binary path.

**Architecture note**: The RX 7800 XT is physically gfx1100. Earlier documentation from Virgil stated gfx1101; `rocminfo` on the actual hardware confirms gfx1100. Use `-DAMDGPU_TARGETS=gfx1100`. No `HSA_OVERRIDE_GFX_VERSION` override is required.

### Go

Go 1.25.5 or later (as specified in `go.mod`). The module uses Go 1.22+ range-over-integer syntax and Go 1.23 `iter.Seq`.

### go-inference

go-rocm depends on `forge.lthn.ai/core/go-inference`. The `go.mod` replaces it with a local path (`../go-inference`). The go-inference directory must be present as a sibling of go-rocm:

```
Code/
├── go-rocm/
└── go-inference/
```

If checking out go-rocm independently: `go work sync` or adjust the `replace` directive.

## Running Tests

### Unit Tests (no GPU required)

The standard test invocation runs unit tests that do not touch GPU hardware:

```bash
go test ./...
```

This covers:
- `server_test.go` — `findLlamaServer`, `freePort`, `serverEnv`, `server.alive()`, dead-server error handling, retry behaviour
- `vram_test.go` — sysfs parsing logic
- `discover_test.go` — model discovery
- `internal/llamacpp/health_test.go` and `client_test.go` — HTTP client and SSE parser
- `internal/gguf/gguf_test.go` — GGUF binary parser

Some unit tests in `server_test.go` have the `//go:build linux && amd64` constraint and will only run on Linux. They do not require a GPU but do require llama-server to be present in PATH.

### Integration Tests (GPU required)

Integration tests are gated behind the `rocm` build tag:

```bash
go test -tags rocm -v -run TestROCm ./...
```

These tests require:
- `/dev/kfd` present
- `llama-server` in PATH or `ROCM_LLAMA_SERVER_PATH` set
- The test model at `/data/lem/gguf/LEK-Gemma3-1B-layered-v2-Q5_K_M.gguf` (SMB mount from M3)

Each test calls `skipIfNoROCm(t)` and `skipIfNoModel(t)` so they skip cleanly when hardware or the model mount is unavailable.

**Available integration tests:**

| Test | What it verifies |
|------|-----------------|
| `TestROCm_LoadAndGenerate` | Full load + Generate, checks architecture from GGUF metadata |
| `TestROCm_Chat` | Multi-turn Chat with chat template applied by llama-server |
| `TestROCm_ContextCancellation` | Context cancel stops iteration mid-stream |
| `TestROCm_GracefulShutdown` | Server survives context cancel; second Generate succeeds |
| `TestROCm_ConcurrentRequests` | Three goroutines calling Generate simultaneously |
| `TestROCm_DiscoverModels` | DiscoverModels returns non-empty result for model directory |

### Benchmarks (GPU required)

```bash
go test -tags rocm -bench=. -benchtime=3x ./...
```

Benchmarks test three models in sequence (Gemma3-4B, Llama3.1-8B, Qwen2.5-7B). They skip if any model file is absent:

| Benchmark | Metric reported |
|-----------|----------------|
| `BenchmarkDecode` | tok/s for 128-token generation |
| `BenchmarkTTFT` | µs/first-tok (time to first token) |
| `BenchmarkConcurrent` | tok/s-aggregate with 4 goroutines and 4 parallel slots |

Model load time is excluded from benchmark timing via `b.StopTimer()` / `b.StartTimer()`. VRAM usage is logged after each load via `GetVRAMInfo()`.

**Reference results (RX 7800 XT, ROCm 7.2.0, ctx=2048, benchtime=3x):**

Decode speed:

| Model | tok/s | VRAM Used |
|-------|-------|-----------|
| Gemma3-4B-Q4_K_M | 102.5 | 4724 MiB |
| Llama-3.1-8B-Q4_K_M | 77.1 | 6482 MiB |
| Qwen-2.5-7B-Q4_K_M | 84.4 | 6149 MiB |

Time to first token:

| Model | TTFT |
|-------|------|
| Gemma3-4B-Q4_K_M | 13.8 ms |
| Llama-3.1-8B-Q4_K_M | 17.1 ms |
| Qwen-2.5-7B-Q4_K_M | 16.8 ms |

Concurrent throughput (4 parallel slots, 4 goroutines, 32 tokens each):

| Model | Aggregate tok/s | vs single-slot |
|-------|----------------|---------------|
| Gemma3-4B-Q4_K_M | 238.9 | 2.3x |
| Llama-3.1-8B-Q4_K_M | 166.2 | 2.2x |
| Qwen-2.5-7B-Q4_K_M | 178.0 | 2.1x |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ROCM_LLAMA_SERVER_PATH` | PATH lookup | Explicit path to llama-server binary |
| `HIP_VISIBLE_DEVICES` | overridden to `0` | go-rocm always sets this to 0 when spawning llama-server |
| `HSA_OVERRIDE_GFX_VERSION` | unset | Not required; GPU is native gfx1100 |
| `ROCM_MODEL_DIR` | none | Conventional directory for model files (not read by go-rocm itself) |

`HIP_VISIBLE_DEVICES=0` is set unconditionally by `serverEnv()`, overriding any value in the calling process's environment. This masks the Ryzen 9 9950X's iGPU (Device 1), which otherwise causes llama-server to crash when it attempts to split tensors across the iGPU and dGPU.

## VRAM Budget

With 16 GB VRAM on the RX 7800 XT, the following models fit comfortably:

| Model | Quant | VRAM (model) | Context 4K | Total | Fits? |
|-------|-------|-------------|-----------|-------|-------|
| Qwen3-8B | Q4_K_M | ~5 GB | ~0.5 GB | ~5.5 GB | Yes |
| Gemma3-4B | Q4_K_M | ~3 GB | ~0.3 GB | ~3.3 GB | Yes |
| Llama3-8B | Q4_K_M | ~5 GB | ~0.5 GB | ~5.5 GB | Yes |
| Qwen3-8B | Q8_0 | ~9 GB | ~0.5 GB | ~9.5 GB | Yes |
| Gemma3-12B | Q4_K_M | ~7.5 GB | ~0.8 GB | ~8.3 GB | Yes |
| Gemma3-27B | Q4_K_M | ~16 GB | ~1.5 GB | ~17.5 GB | Tight |
| Llama3-70B | Q4_K_M | ~40 GB | ~2 GB | ~42 GB | No (partial offload) |

The context cap (`min(model_context_length, 4096)` by default) is essential for models like Gemma3-4B and Llama-3.1-8B, which have 131072-token native context. Without the cap, the KV cache allocation alone would exhaust VRAM.

## Test Patterns

Tests use `github.com/stretchr/testify/assert` and `require`. The naming convention from the broader go ecosystem applies:

- `_Good` suffix — happy path
- `_Bad` suffix — expected error conditions
- `_Ugly` suffix — panic or edge cases

Integration tests use `skipIfNoROCm(t)` and `skipIfNoModel(t)` guards. Never use `t.Fatal` to skip; always use `t.Skip`.

When writing new unit tests that do not need GPU hardware, do not add the `rocm` build tag. The `linux && amd64` tag is sufficient for tests that test Linux-specific code paths.

## Coding Standards

- **Language**: UK English throughout. Colour, organisation, initialise, behaviour — never American spellings
- **Strict types**: `declare(strict_types=1)` is a PHP convention, but the Go equivalent applies: use concrete types, avoid `any` except where the interface demands it
- **Error messages**: Lower case, no trailing punctuation. Prefixed with the package context: `"rocm: ..."`, `"llamacpp: ..."`, `"gguf: ..."`
- **Formatting**: `gofmt` / `goimports`. No exceptions
- **Licence**: EUPL-1.2. All new files must include the licence header if adding a file header comment

## Conventional Commits

Use the conventional commits format:

```
type(scope): description

feat(server): add GPU layer count override via environment variable
fix(gguf): handle uint64 context_length from v3 producers
test(integration): add DiscoverModels test for SMB mount
docs(architecture): update VRAM budget table
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `chore`

## Co-Authorship

All commits must include the co-author trailer:

```
Co-Authored-By: Virgil <virgil@lethean.io>
```

## Adding a New Backend Feature

The typical sequence for a new go-rocm feature:

1. If the feature requires a go-inference interface change (new `LoadOption`, `GenerateOption`, or `TextModel` method), write that change first in go-inference and coordinate with Virgil (the orchestrator) before implementing the consumer side
2. Write unit tests first; most server and client behaviour is testable without GPU hardware
3. If integration testing on the homelab is needed, use the `//go:build rocm` tag
4. Update `docs/architecture.md` if the data flow or component structure changes
5. Record benchmark results in `docs/history.md` under the relevant phase if performance characteristics change materially
