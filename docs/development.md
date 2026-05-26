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

### llama-server Legacy Path

llama-server is only required for `-tags rocm_legacy_server`. The default native package path does not spawn a server.

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

Go 1.26.x as specified in `go.work` and `go/go.mod`.

### Workspace

This repository uses the Core `go/` subtree layout. Root `go.work` links local submodules under `external/`, and the root module is a workspace gate for `go test ./...`: native cgo test runs delegate to the real `go/` module and shared `external/go-inference/go` contract tests, while static cross-builds compile the ROCm package through local replacements without trying to execute foreign test binaries. The native cgo bridge also links local `external/go-cgo/go` for managed C strings/scopes and `core.PinnedView` handoff patterns. For tight iteration, run the same commands from `go/` or a specific submodule.

## Running Tests

### Unit Tests (no GPU required)

The standard test invocation runs unit tests that do not touch GPU hardware:

```bash
go test ./... -count=1
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
go test -tags rocm_legacy_server ./... -count=1
```

When running directly inside `go/` from a Linux host and compiling the Darwin
surface, add an execution shim so Go does not try to run a macOS test binary:

```bash
cd go
GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test -exec=/bin/true ./... -count=1
```

On non-Linux or non-amd64 platforms, the ROCm package registers an unavailable
`rocm` backend with `go-inference`; `ROCmAvailable()` returns false and
`LoadModel` returns a platform-unavailable error. The cross-build command above
checks that stub surface compiles, but the final developer-Mac gate should still
run on macOS so those stub tests and examples execute.

This covers:
- native ROCm contract reporting
- model-pack inspection for GGUF, safetensors, tokenizer sidecars, malformed-weight rejection, Gemma4 nested text-config/tied-embedding metadata, BERT embedding/rerank/classifier metadata hints, metadata-only MoE/JANGTQ/codebook capabilities (`runtime_status=metadata_only`) with fixture-kernel and pending production-integration labels, and architecture aliases
- scheduler/cancellation wrapper behaviour
- cache warm/stats/clear compatibility logic, optional metadata and portable KV snapshot disk refs, exact cold disk-ref rehydrate, disk-byte accounting, and best-effort HIP device remirroring for warmed/cold-restored portable KV snapshots
- speculative and prompt-lookup decode package helpers over the shared `go-inference/decode` harness, including model stream error propagation
- OpenAI service mux cache/cancel endpoint routing and cache-warm validation over ROCm wrappers
- parser registry behaviour
- benchmark warmup runs across all prompts, measured-run and measured latency/duration labels, shared active/peak memory fields plus memory labels, measured probe counts, cache-pressure and memory-pressure probe emission, cache-label mirroring for KV shape and optional disk refs, and active-adapter LoRA overhead labels
- eval token-count metrics, batched experimental classification loss/perplexity from `target_token_id` plus logits when available, explicit native cross-entropy fixture capability labels and eval-result labels (`loss_kernel`, `loss_kernel_name`, `loss_scope`) even when loss is unsupported or not requested, compact classification logit/entropy probe events for `WithLogits`, unsupported/logits-unavailable loss labels otherwise, and qualitative probe results that record unavailable generation without failing the eval report
- state wake/sleep/fork metadata lifecycle, metadata-only `StatefulModel` bundle capture/restore, package-local KV snapshot sleep/wake refs, raw KV block-bundle State refs with `BorrowRefBytes` wake, HIP device-mirror snapshot refs, loaded-model best-effort wake/fork remirror plus direct HIP block-stream restore with package-local fallback until production kernels own restore, and state-runtime close coverage when wake/restore/close replaces owned handles
- package-local fp16/q8/k-q8-v-q4 KV cache page round trips, paging, byte counts, hit rate, restore timing, binary snapshot refs, constructibility of planner-selected cache modes through cache warm, and fake-driver HIP device mirror allocation/copy/free accounting plus incremental decoded-token page appends, device-to-host portable snapshots, fixed descriptor byte layout, descriptor-table copy/rollback, 64-byte KV launch descriptor validation, 64-byte prefill launch-packet encoding, and 96-byte decode launch-packet encoding
- CPU reference MoE routing/lazy residency, residual summaries, JANGTQ/MXTQ packed projection, codebook lookup, embedding mean-pool, rerank cosine, and LoRA projection
- HIP tensor validation/allocation/copy, typed prefill/decode/projection kernel seams carrying device KV mirrors plus descriptor tables, KV launch descriptors, prefill/decode launch packets, 96-byte projection and JANGTQ-projection launch packets, 64-byte codebook-lookup/embedding-mean-pool/rerank-cosine/RMSNorm/RoPE/greedy-sampler/MoE-router/MoE-lazy-expert launch packets, 96-byte attention launch packets, 128-byte LoRA-projection launch packets, 160-byte tiny-prefill/tiny-decode launch packets with fp32/fp16/q8 output-head encodings, native GGUF plus safetensors weight loading into HIP memory, including sharded packs, Gemma4-E2B-style tied U32 4-bit embedding validation, loaded tiny-model generation/classification through device-resident f32 embeddings and f32/f16/raw-q8/JANGTQ/codebook output heads, BERT-style f32 word-embedding-only model load with experimental mean-pool embedding, embedding-cosine rerank, f32/f16 sequence-classifier rerank scoring, and BERT classifier LoRA adapter application, experimental loaded tiny-model `rocm-tiny-lora` adapter application and Qwen/Gemma small LM-head adapter application through `rocm_lora_projection`, typed loaded Qwen/Gemma small decode smoke coverage that reads the request token embedding row from loaded device memory, appends package-local KV, and incrementally appends supplied device KV mirrors, token/projection/transformer device-buffer upload/rollback coverage, device-to-host copy coverage, optional fake-testable kernel launch config validation, fake prefill/decode packet and referenced-memory validation, fake projection/JANGTQ-projection/codebook-lookup/LoRA-projection/embedding-mean-pool/rerank-cosine/RMSNorm/RoPE/greedy/attention/MoE-router/MoE-lazy-expert/tiny-prefill/tiny-decode launch output readback, not-linked decode launch preflight for supplied device KV resources, projection rank/model-dimension/byte-size checks, token embedding lookup, single-head and multi-head attention, causal prefill attention, decode-with-KV, integrated tiny LM prefill/decode, composed Qwen/Gemma small decode smoke coverage using fixture and loaded device-resident weights, fp16/q8/f32 projection, JANGTQ/MXTQ packed projection and codebook lookup including loaded tiny output-head logits, LoRA projection, RMSNorm, RoPE, MoE top-k routing and lazy expert residency bitmaps, greedy and top-k/temperature sampler references, logit, entropy, selected-head, and layer-coherence probe summaries, prompt-lookup draft, speculative-accept, embedding mean-pool, rerank cosine, cross-entropy/perplexity, distillation KL, and GRPO advantage reference fixtures, kernel status labels, and kernel-not-linked errors
- planned training capability labels for `lora.training`, `distillation`, and `grpo`, including `runtime_status=planned`, `training_kernel=not_linked`, `training_interface=not_implemented`, exact `required_kernel` values (`lora_backward`, `distillation_forward_loss`, and `grpo_rollout_policy`), explicit `distillation_kernel`/`grpo_kernel`-aware toy fixture labels for distillation KL and GRPO advantage normalization, and package-local loaded-model hooks for those toy fixtures without shared training-interface support
- fake HIP driver nil/unavailable/malloc/free failure handling
- import-boundary enforcement against concrete workflow/runtime package imports, including skip-safe scans of local `go-ai`, `go-ml`, and `go-inference` checkouts when present
- `internal/gguf/gguf_test.go` — GGUF binary parser

Legacy server tests are covered by `go test -tags rocm_legacy_server ./... -count=1` and do not require hardware.

### Integration Tests (GPU required)

Hardware tests are opt-in through environment variables:

```bash
GO_ROCM_RUN_HIP_TESTS=1 go test ./go -run 'TestHIP|TestNative' -count=1
GO_ROCM_RUN_MODEL_TESTS=1 go test ./go -run 'Test.*Smoke|Test.*Generate|Test.*Decode' -count=1
GO_ROCM_RUN_CACHE_TESTS=1 go test ./go -run 'Test.*KV|Test.*Cache' -count=1
```

Hardware tests must skip with a clear message when the environment variable is not set. Model smoke tests also require `GO_ROCM_MODEL_PATH` to point at a local GGUF or safetensors model pack; the Gemma4-E2B smokes use `/data/lem/models/gemma4/LEM-Gemma4-E2B` for the BF16 correctness anchor and `/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit` for the faster MLX-q4 loop when present. The cache smoke mirrors a toy package-local KV cache into HIP allocations and verifies device-mirror stats, descriptor fields, descriptor-table allocation, token-buffer prefill launch packets, and projection launch packets; production decode kernels still cannot consume those pages. Until production kernels are linked, generic generation returns `native decode kernels are not linked yet` and capability reports set `kernel_status=not_linked`, while the loaded Gemma4 MLX-q4 path exposes experimental package-local Generate/Chat/BatchGenerate/Classify/Benchmark/Eval plus speculative and prompt-lookup helper smokes with production decode/prefill/KV labels still `not_linked`. Gemma4 text-only metadata follows the current `go-mlx` dev alias convention: `gemma4_text`, `Gemma4ForCausalLM`, and `Gemma4TextForCausalLM` inspect as text-model packs while conditional-generation Gemma4 packs remain `gemma4`; both aliases reach the experimental q4 text route.

The current live q4 smoke for the RX 7800 XT pins the real discrete card by UUID because device 0 may otherwise resolve to the onboard GPU:

```bash
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
  GO_ROCM_RUN_MODEL_TESTS=1 \
  GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
  GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
  GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
  GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' \
  go test ./go -run 'TestNative(DecodeSmokeKernelStatus|ModelPackSmokeGemma4E2B)_Good' -count=1 -v
```

On 2026-05-13 this passed with q4 package `Prefill`/`Decode` output and public q4 `Generate` for BOS-aware prompt tokens `[2 10979]`, generated tokens `[236764 3307]`, and decoded text `["," "my"]` over `/tmp/go-rocm-kernels-gfx1100.hsaco`. The BF16 correctness anchor also passed `TestNativeDecodeSmokeKernelStatus_Good` on the same pinned RX 7800 XT against `/data/lem/models/gemma4/LEM-Gemma4-E2B`, reaching the layer-0 tied LM-head greedy check.

The optional native launcher expects a precompiled HSACO path in `GO_ROCM_KERNEL_HSACO`. Kernels are looked up by the internal names `rocm_prefill`, `rocm_decode`, `rocm_projection`, `rocm_jangtq_projection`, `rocm_codebook_lookup`, `rocm_lora_projection`, `rocm_embedding_mean_pool`, `rocm_rerank_cosine`, `rocm_rms_norm`, `rocm_rope`, `rocm_greedy_sample`, `rocm_attention`, `rocm_moe_router`, `rocm_moe_lazy_experts`, `rocm_tiny_prefill`, `rocm_tiny_decode`, `rocm_cross_entropy_loss`, `rocm_distillation_kl_loss`, and `rocm_grpo_advantage`, and receive one device pointer to the fixed little-endian launch packet. The first source artifact is `kernels/rocm_kernels.hip`; build it with `hipcc --std=c++23 --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o build/rocm_kernels_gfx1100.hsaco`. If the variable is unset or the module lacks a compatible symbol, launch attempts fail before any fallback path is used. The ROCm 7.2 homelab compiler accepts `--std=c++23`, but its host headers currently do not provide `<mdspan>`; use `core.PinnedView`/`go-cgo` on the Go side now, and only include `go-cgo/go/cgo_pinned_view.hpp` in HIP/C++ once the toolchain provides mdspan or a compatible header. When the variable is set, loaded HIP models select a projection/embedding/rerank/LoRA-linked kernel set for package-local projection requests; tiny vocab-major f32 loaded models with f32/f16/raw-q8/JANGTQ/codebook output heads can run toy prefill/decode/generate through the compiled tiny kernels plus packed/codebook output logits through `rocm_jangtq_projection`, `rocm_codebook_lookup`, and `rocm_projection`, experimental model-level embedding mean-pool/rerank calls, and experimental `rocm-tiny-lora` output-head adapter application through `rocm_lora_projection`; BERT sequence-classification packs can run classifier-head rerank plus experimental classifier LoRA adapters through the same LoRA projection kernel; fixture-scale Qwen/Gemma loaded models can run typed `DecodeToken` by reading a loaded f32 token embedding row, composing the first-layer primitive kernels, appending package-local KV, incrementally appending supplied device KV mirrors, and applying an experimental LM-head LoRA adapter through `rocm_lora_projection`; and loaded HIP models can run package-local toy cross-entropy, distillation KL, and GRPO advantage fixtures through the compiled loss kernels. Capability, benchmark, and eval quality-probe labels for linked tiny generation/decode helpers include `kernel_scope=toy_tiny_fixture` plus `production_decode=not_linked` and `production_prefill=not_linked`; embedding/rerank capability labels include loaded fixture scopes plus `production_embedding_models=not_linked` and `production_rerank_models=not_linked`; LoRA capability labels include `kernel_scope=loaded_adapter_fixtures`, `supported_adapter_scopes=tiny_output_head,qwen_gemma_small_lm_head,bert_sequence_classifier`, and `production_adapter_application=not_linked`; future production decode/prefill labels use `rocm_decode`/`rocm_prefill`. Full production model-family decode/prefill generation and shared training interfaces still report not-linked status; the loaded Gemma4 MLX-q4 route is an experimental package-local exception for development smoke coverage and labels its production decode/prefill/KV backing as not linked. `GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=... go test ./go -run 'TestHIPHardware.*KernelSource' -count=1 -v` verifies direct and loaded-model fp16/q8 projection launch, direct JANGTQ/MXTQ packed projection launch, direct codebook lookup launch, direct LoRA projection launch, direct embedding mean-pool and rerank cosine launch, direct RMSNorm/RoPE/greedy/attention/MoE-router/MoE-lazy-expert primitive launch, direct cross-entropy/distillation/GRPO loss fixture launch, composed Qwen3 small decode smokes through those primitives using fixture and loaded device-resident weights, typed loaded-model Qwen3 small `DecodeToken` from a device embedding row with fp16 package-local KV plus q8/k-q8-v-q4 package-local and incrementally appended device-KV paths, direct toy tiny-prefill logits/attention/KV/greedy readback, direct toy tiny-decode updated-KV/logits/attention/greedy readback using prefill-written KV, loaded tiny-model generation from device-resident tensors including raw q8 and codebook output heads, fp16/q8 tiny output-head variants, and prefill/decode packet-consumer launches on hardware for fp16, q8, and k-q8-v-q4 cache modes, including device-written status markers from reserved launch-packet fields. `GO_ROCM_RUN_CACHE_TESTS=1 go test ./go -run TestHIPHardwareKVCacheSmoke_Good -count=1 -v` also verifies block-cache warm remirroring into HIP device pages before the lower-level descriptor and launch-packet smoke checks.

The source and restore layers should stay HIP-generic. On NVIDIA systems, HIPCC can target the CUDA/NVCC backend with `HIP_PLATFORM=nvidia`; that is the preferred route for a future NVIDIA profile. `GO_ROCM_RUN_NVIDIA_HIP_COMPILE_TESTS=1 CUDA_PATH=/usr/local/cuda go test ./go -run TestHIPKernelSource_NVIDIAHIPCompile_Good -count=1 -v` compiles `kernels/rocm_kernels.hip` through that backend as a no-GPU-required proof. The default NVIDIA arch is `sm_75` and can be changed with `GO_ROCM_NVIDIA_HIP_ARCH`; the default NVIDIA standard is `c++20` because CUDA 12.8 `nvcc` rejects `--std=c++23`, while the AMD ROCm build remains `--std=c++23`.

ZLUDA is the CUDA-on-non-NVIDIA acceptance runner for AMD smoke tests. Unpack ZLUDA v5 so `GO_ROCM_ZLUDA_DIR` points at the directory containing `libcuda.so` (`/opt/zluda/v5/zluda` on the homelab box), install or expose a compatible ROCm 6.x HIP runtime path such as `/opt/rocm-6.4.4/lib`, then run `GO_ROCM_RUN_ZLUDA_CUDA_TESTS=1 ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 go test ./go -run TestHIPKernelSource_ZLUDACUDARuntimeSmoke_Good -count=1 -v`. This compiles a tiny CUDA binary with `nvcc`, loads ZLUDA's `libcuda.so`, and verifies a kernel launch/readback on the AMD GPU. ZLUDA is separate from HIP source compiled through HIPCC; both proofs are useful and cover different failure modes.

Hot-path changes should follow AX-11: per-token, per-page, per-request, per-frame, public frequently imported, or cross-product functions get `BenchmarkFunctionName_Scenario` coverage with `b.ReportAllocs()`. For the State/KV restore path, `BenchmarkROCmKVCacheBlockFromRawPayload_KQ8VQ4Page` tracks host decode cost and `BenchmarkROCmDeviceKVPageFromRawPayload_KQ8VQ4PinnedCopy` tracks the direct pinned HIP page restore cost.

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
| `ROCR_VISIBLE_DEVICES` | process default | Native HIP hardware smokes can pin the real dGPU by UUID, e.g. `GPU-880ed6479d653a85` for the RX 7800 XT when the onboard GPU is also visible |
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
