# go-rocm Architecture

## Overview

go-rocm is the AMD ROCm backend for the Core Go inference ecosystem. Its target shape is native and package-first, matching the role `go-mlx` plays on Apple Silicon: `go-inference` owns shared contracts, `go-rocm` owns AMD runtime execution, and higher packages (`go-ai`, `go-ml`) consume the common surface without circular dependencies.

The default backend does not depend on a managed `llama-server` subprocess. That server-backed implementation was useful as a fast bootstrap path, but it hides model state behind an HTTP harness and blocks research-grade features such as shared KV cache ownership, portable state bundles, adapter identity checks, probe streams, and native training hooks.

## Native Runtime Surface

The default `linux/amd64` build provides:

- `inference.Backend` registration for the `rocm` backend
- `inference.ModelFitPlanner` for ROCm memory/context/cache recommendations
- `inference.TokenizerModel`, `AdapterModel`, `ProbeableModel`, `BenchableModel`, and `Evaluator` on loaded models
- GGUF metadata inspection without loading tensors
- sysfs VRAM monitoring for memory planning and metrics
- an internal `nativeRuntime` seam for the HIP loader and kernels

Until the HIP runtime loader is linked, `Available()` intentionally returns false. That is preferable to silently falling back to an HTTP server because callers can make an explicit backend choice and tests can prove whether native execution is actually present.

## Package Structure

```text
go-rocm/
├── go/rocm.go                 package doc and exported discovery types
├── go/register_rocm.go        linux && amd64 registration
├── go/rocm_stub.go            non-linux or non-amd64 stubs
├── go/native.go               default native ROCm contract surface
├── go/backend.go              legacy server backend, rocm_legacy_server only
├── go/model.go                legacy server model, rocm_legacy_server only
├── go/server.go               legacy server lifecycle, rocm_legacy_server only
├── go/vram.go                 sysfs VRAM monitoring
├── go/discover.go             GGUF model discovery
└── go/internal/gguf/          GGUF v2/v3 metadata parser
```

`go/internal/llamacpp/` remains only for the legacy server tag while the native runtime is being filled in.

## Build Tags

- `linux && amd64 && !rocm_legacy_server`: default native ROCm surface.
- `linux && amd64 && rocm_legacy_server`: previous managed `llama-server` backend.
- `!linux || !amd64`: safe stubs for development on non-ROCm machines.

On a non-Linux development machine, compile the default Linux surface with:

```bash
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test -exec=/usr/bin/true ./...
```

The `-exec=/usr/bin/true` gate proves compilation without trying to execute Linux test binaries on macOS.

## Native Load Flow

1. `register_rocm.go` registers `&rocmBackend{}` with `go-inference`.
2. `LoadModel(path, opts...)` reads GGUF metadata with `gguf.ReadMetadata`.
3. Load options become a `nativeLoadConfig`: context size, GPU layers, parallel slots, adapter path, and model metadata.
4. `nativeRuntime.LoadModel` returns a backend-owned `nativeModel` once the HIP loader exists.
5. `rocmModel` wraps that native model and exposes shared `go-inference` generation, tokenizer, adapter, probe, bench, and eval contracts.

The native seam is deliberately small. HIP-specific ownership of tensors, graph execution, KV cache, adapter application, and eventual training stays behind `nativeModel` rather than leaking into consumers.

## Memory Planning

`PlanModelFit` is available before the HIP loader exists. It uses model identity plus measured or supplied memory to estimate:

- machine class (`rocm-16gb`, `rocm-24gb`, `rocm-64gb-plus`)
- recommended context length and batch size
- cache mode (`q8` on constrained memory or long context, otherwise `fp16`)
- estimated KV cache bytes
- architecture and quantisation support flags
- LoRA/training feasibility hints

The estimates are intentionally conservative because ROCm users are likely to run 16 GB cards where context and KV decisions matter more than generic defaults.

## Probe And Eval Surface

`rocmModel` emits shared `inference.ProbeEvent` token events during generation. Native kernels can add logits, entropy, head selection, layer coherence, router decisions, residual summaries, cache pressure, and memory pressure through the same sink without changing consumers.

The current benchmark/eval hooks are lightweight wrappers over the model surface. Once native logits/loss are exposed, the same shared report structs can carry real perplexity and quality probes.

## GGUF Metadata Parser

`internal/gguf/` is a standalone binary metadata reader. It supports GGUF v2 and v3, validates the magic/version, reads metadata KV pairs, and extracts architecture, name, file type, size label, context length, block count, and file size without loading tensor data.

The parser reads only the header, not tensor payloads, so model discovery and fit planning remain cheap even for multi-GB packs.

## VRAM Monitoring

`GetVRAMInfo()` reads `mem_info_vram_total` and `mem_info_vram_used` from sysfs (`/sys/class/drm/cardN/device/`). It selects the card with the largest VRAM total, which avoids hardcoding card numbers on machines with an iGPU plus dGPU.

Reads are non-atomic, so `Free` is clamped to zero if a transient read reports `Used > Total`.

## Legacy Server Path

The old server-backed implementation is still buildable with `-tags rocm_legacy_server`. It starts `llama-server`, uses OpenAI-compatible streaming endpoints, and keeps the old subprocess lifecycle tests. That path exists as a compatibility/debug bridge only; it is not the default architecture.
