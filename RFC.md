# RFC: go-rocm Parity With go-mlx

Status: Draft for `/goal` execution  
Target repo: `/Users/snider/Code/core/go-rocm`  
Primary contract repo: `/Users/snider/Code/core/go-rocm/external/go-inference/go`  
Reference backend: `/Users/snider/Code/core/go-mlx`

## 1. Summary

`go-rocm` should become the AMD ROCm backend sibling of `go-mlx`. The target is
not a copied implementation. The target is contract parity through
`go-inference`, package-first ROCm-native implementation, and hardware-specific
work isolated behind ROCm build tags.

`go-mlx` is ahead in the following areas:

- Scheduler and request cancellation.
- Block-prefix prompt/KV cache service.
- Parser registry for thinking and tool channels.
- State bundle and memvid-backed session lifecycle.
- Fast eval/bench harness fields.
- JANG/JANGTQ, codebook/VQ, MiniMax M2, and MoE metadata/probe support.
- Native KV cache modes, including q8, paged, and k-q8-v-q4 planning.
- Training/eval workflow hooks exposed through shared interfaces.

`go-rocm` already has the right direction: native HIP availability, model-pack
inspection, ROCm 16GB memory planning, and planned capability reports. This RFC
turns that groundwork into an executable sequence.

## 2. Non-Goals

- No UI, TUI, desktop app, or web app.
- No provider policy, API keys, `go-ai` fallback routing, or rate limiting.
- No import from `go-mlx`.
- No default subprocess inference path. The legacy server path remains only
  behind `rocm_legacy_server`.
- No native CUDA or TPU work in this repo.
- No full fine-tuning before basic ROCm forward/decode is real.

## 3. Dependency Rules

Allowed:

- `go-rocm` imports `dappco.re/go`.
- `go-rocm` imports `dappco.re/go/inference`.
- `go-rocm` imports package-local internals such as `go/internal/gguf`.
- `go-rocm/external/go-inference` may be synchronised with the shared local
  `go-inference` contract layer.

Forbidden:

- `go-rocm` importing `go-mlx`.
- `go-inference` importing any concrete runtime.
- `go-rocm` importing `go-ai`, `go-ml`, `go-rag`, provider packages, or
  `core/api`.
- Moving provider policy into `go-rocm`.

## 4. Current go-rocm State

Observed files and roles:

- `go/native.go`: native backend and model wrapper, capability report, memory
  planning, fallback tokenizer/template, bench/eval/probe wrappers.
- `go/hip_runtime.go`: native HIP runtime abstraction, tensor allocation and
  host-to-device copy.
- `go/hip_driver_cgo.go`: dynamic `libamdhip64` loading and narrow HIP calls.
- `go/hip_driver_nocgo.go`: no-cgo fallback.
- `go/model_pack.go`: model-pack inspection for GGUF, safetensors,
  `config.json`, `jang_config.json`, and `codebook_config.json`.
- `go/native_contract_test.go`: contract tests for the current ROCm native
  shape.
- `go/internal/gguf`: GGUF metadata parser.
- `go/model.go`, `go/server.go`, `go/internal/llamacpp`: legacy server path
  behind `rocm_legacy_server`.
- `external/go-inference/go/capability.go`: expanded capability IDs.
- `external/go-inference/go/contracts.go`: optional shared interfaces for
  scheduler, cache, embeddings, rerank, parser, and model-pack inspection.

Current capability stance:

- Runtime/model metadata: usable.
- Model fit planning: usable.
- Evaluation: experimental token-count path.
- Probe events: experimental stream-token path.
- Tokenizer/chat template: experimental fallback path.
- Generate/chat/classify/batch: planned until kernels are linked.
- Scheduler/cancel/cache/parsers/state: planned until wrapper/cache/state code
  is implemented.
- JANGTQ/codebook/MoE: metadata recognised, native kernels pending.

## 5. Target Capability Matrix

| Capability | Contract | ROCm target status before HIP kernels | ROCm target status after narrow kernels |
| --- | --- | --- | --- |
| `model.load` | `inference.Backend` | supported for metadata plus HIP tensor copy | supported |
| `generate` | `inference.TextModel` | planned with clear kernel error | experimental for one small model family |
| `chat` | `inference.TextModel` | planned with clear kernel error | experimental with template support |
| `classify` | `inference.TextModel` | planned with clear kernel error | experimental after prefill kernel |
| `batch.generate` | `inference.TextModel` | planned | experimental after decode loop |
| `tokenizer` | `inference.TokenizerModel` | experimental fallback | supported when tokenizer files are loaded |
| `chat.template` | `inference.TokenizerModel` | experimental fallback | supported when template parsing is loaded |
| `model.fit` | `inference.ModelFitPlanner` | supported | supported |
| `memory.planning` | `inference.MemoryPlan` | supported | supported with measured memory |
| `scheduler` | `inference.SchedulerModel` | supported wrapper | supported wrapper |
| `request.cancel` | `inference.CancellableModel` | supported wrapper | supported wrapper plus kernel cancellation |
| `cache.blocks` | `inference.CacheService` | experimental metadata/in-memory blocks | supported with native KV cache |
| `cache.warm` | `inference.CacheService` | experimental no-kernel warm accounting | supported with prefill cache |
| `cache.disk` | `inference.CacheService` | planned | experimental with `go-inference/state` refs |
| `reasoning.parse` | `inference.ReasoningParser` | supported parser registry | supported |
| `tool.parse` | `inference.ToolParser` | supported parser registry | supported |
| `state.bundle` | `inference.StatefulModel` or `go-inference/state` lifecycle | planned | experimental after KV ownership |
| `kv.snapshot` | backend-owned KV snapshot | planned | experimental after KV ownership |
| `prompt.cache` | cache service plus state refs | planned | experimental after KV ownership |
| `probe.events` | `inference.ProbeableModel` | experimental stream/scheduler/cache events | supported with kernel probes |
| `benchmark` | `inference.BenchableModel` | experimental fake/native-error fields | supported with real timings |
| `evaluation` | `inference.Evaluator` | experimental token counts | supported with loss/perplexity |
| `lora.inference` | `inference.AdapterModel` | planned | experimental after tensor overlays |
| `lora.training` | training contracts | planned | planned until forward/backward kernels exist |
| `distillation` | training contracts | planned | planned until logits/teacher path exists |
| `grpo` | training contracts | planned | planned until rollout generation exists |
| `embeddings` | `inference.EmbeddingModel` | planned | experimental for BERT/embed models |
| `rerank` | `inference.RerankModel` | planned | experimental for scorer models |
| `speculative.decode` | package/runtime helper | planned | experimental after decode loop |
| `prompt.lookup.decode` | package/runtime helper | planned | experimental after cache lookup |
| `moe.routing` | model-pack/runtime probes | metadata-only experimental | experimental with router kernel |
| `moe.lazy_experts` | memory planner plus runtime residency | metadata-only experimental | experimental with expert page-in |
| `jangtq` | metadata plus packed kernels | metadata-only experimental | experimental with MXTQ dequant/projection |
| `codebook.vq` | metadata plus VQ kernels | metadata-only experimental | experimental with codebook lookup |
| `responses.api` | `go-inference/openai` | supported when handlers mount | supported |
| `anthropic.messages` | `go-inference/anthropic` | supported when handlers mount | supported |
| `ollama.compat` | `go-inference/ollama` | supported when handlers mount | supported |

## 6. File Ownership Plan

### Shared contracts

- Modify: `external/go-inference/go/capability.go`
- Modify: `external/go-inference/go/contracts.go`
- Create or sync: `external/go-inference/go/state/*.go`
- Test: `external/go-inference/go/capability_test.go`
- Test: `external/go-inference/go/contracts_test.go`
- Test: `external/go-inference/go/state/state_test.go`

Only add backend-neutral contracts here. Do not import ROCm, MLX, CUDA, TPU,
provider packages, rate-limiters, or API hosts.

### ROCm public runtime

- Modify: `go/native.go`
- Modify: `go/register_rocm.go`
- Modify: `go/rocm.go`
- Modify: `go/rocm_stub.go`
- Test: `go/native_contract_test.go`
- Test: `go/register_rocm_test.go`
- Test: `go/rocm_stub_test.go`

The loaded model should expose the same optional interfaces as MLX where
callable, and report planned status where kernels are not linked.

### Scheduler and cancellation

- Create: `go/scheduler.go`
- Create: `go/scheduler_test.go`
- Example: `go/scheduler_example_test.go`

The scheduler wraps an `inference.TextModel` and implements
`inference.SchedulerModel`, `inference.CancellableModel`, and
`inference.ProbeableModel`. It must support bounded queueing, request IDs,
cancel-before-start, cancel-during-stream, slow consumer backpressure, queue
latency, first-token latency, and scheduler probe events.

### Cache service

- Create: `go/cache.go`
- Create: `go/cache_test.go`
- Example: `go/cache_example_test.go`

The first cache service can be in-memory and metadata-first. It must implement
`inference.CacheService` and return honest stats for blocks, memory bytes, disk
bytes, hits, misses, hit rate, restore milliseconds, cache mode, and labels.
Native KV ownership can replace the storage backend without changing the
interface.

### Parser registry

- Create: `go/parser_registry.go`
- Create: `go/parser_registry_test.go`
- Example: `go/parser_registry_example_test.go`

Support Qwen `<think>...</think>`, Gemma-style channels, MiniMax/JANG capability
hints, DeepSeek R1 thinking spans, GPT-OSS channels, Mistral/Hermes/Granite JSON
tool calls, and generic XML/JSON fallback. The parser registry should implement
`inference.ReasoningParser` and `inference.ToolParser`.

### State lifecycle

- Create: `go/state_session.go`
- Create: `go/state_session_test.go`
- Example: `go/state_session_example_test.go`

Use `go-inference/state` for `WakeRequest`, `WakeResult`, `SleepRequest`,
`SleepResult`, `Session`, and `Forker`. The ROCm layer owns the runtime handles.
The shared state package owns the portable metadata shape. Memvid-like stores
must remain URI-first and opaque to `go-rocm`.

### Model-pack inspection

- Modify: `go/model_pack.go`
- Modify: `go/internal/gguf/gguf.go`
- Test: `go/native_contract_test.go`
- Test: `go/internal/gguf/gguf_test.go`

Inspection must cover:

- GGUF metadata and tensor table.
- Safetensors header count, dtype list, payload bytes, tensor shapes.
- `config.json` architectures and aliases.
- `tokenizer.json`, `tokenizer_config.json`, and chat template metadata when
  present.
- `jang_config.json` profile, MXTQ format, group size, bit layout, source
  architecture, parser hints, and cache hints.
- `codebook_config.json` VQ/codebook shape metadata.
- Context limits and memory-fit hints for 16GB ROCm.

### HIP runtime

- Modify: `go/hip_runtime.go`
- Modify: `go/hip_driver_cgo.go`
- Modify: `go/hip_driver_nocgo.go`
- Create: `go/hip_kernels.go`
- Create: `go/hip_kernels_stub.go`
- Test: `go/hip_runtime_test.go`
- Test: `go/hip_driver_fake_test.go`
- Opt-in hardware test: `go/hip_hardware_test.go`

The kernel ladder must be narrow:

1. Validate tensor shapes and dtypes without launching kernels.
2. Add a tiny fp16/q8 matmul or projection fixture.
3. Add RMSNorm/RoPE/attention/sampler pieces only after the previous fixture is
   deterministic.
4. Add one decode path for Qwen3 or Gemma small.
5. Add KV cache ownership.
6. Add q8 and k-q8-v-q4 cache modes.
7. Add MoE router and lazy expert residency.
8. Add JANGTQ/MXTQ packed dequant/projection.

## 7. Detailed Task Sequence

### Task 0: Baseline and Contract Audit

1. Run:

   ```sh
   git status --short
   go test ./... -count=1
   GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
   go test -tags rocm_legacy_server ./... -count=1
   ```

2. Save the exact baseline failures in the `/goal` working notes.
3. Do not edit unrelated dirty files.

Acceptance:

- The agent can name current failing packages, if any.
- Dirty user changes are preserved.

### Task 1: Contract Synchronisation

Write failing tests first:

- `external/go-inference/go/contracts_test.go`
  - Assert `SchedulerModel`, `CancellableModel`, `CacheService`,
    `ReasoningParser`, `ToolParser`, `EmbeddingModel`, `RerankModel`, and
    `ModelPackInspector` can be implemented by a fake model.
- `external/go-inference/go/capability_test.go`
  - Assert all capability IDs in `GOAL.md` are unique and reportable.
- `external/go-inference/go/state/state_test.go`
  - Assert wake/sleep refs clone labels and preserve URI, token, model, and
    tokenizer metadata.

Implementation:

- Sync missing shared contract files from the canonical local `go-inference`.
- Keep `go-inference` free of provider policy and concrete runtime imports.

Acceptance:

```sh
go test ./external/go-inference/go/... -count=1
```

### Task 2: Capability Parity Report

Write failing tests in `go/native_contract_test.go`:

- Backend report includes runtime identity, device memory labels, architecture
  list, quantization list, and cache modes.
- Loaded model report includes model identity, active adapter identity, and all
  shared capability IDs.
- Supported, experimental, and planned statuses match actual callability.
- `generate`, `chat`, `classify`, and `batch.generate` remain planned until a
  native kernel path exists.
- Parser/cache/scheduler capabilities become supported only after those wrapper
  implementations exist.

Implementation:

- Update `rocmCapabilityReport` in `go/native.go`.
- Add helper functions for capability construction if the current list becomes
  too large to audit safely.
- Add labels for `runtime_status`, `kernel_status`, and `metadata_status` where
  a capability is visible but not fully native.

Acceptance:

```sh
go test ./go -run 'TestNativeContract_Rocm.*Capabilities' -count=1
```

### Task 3: Scheduler and Cancellation

Write tests in `go/scheduler_test.go`:

- `TestScheduler_Good_StreamsQueuedRequest`
- `TestScheduler_Good_CancelsBeforeStart`
- `TestScheduler_Good_CancelsDuringDecode`
- `TestScheduler_Bad_RejectsNilModel`
- `TestScheduler_Ugly_SlowConsumerDoesNotDeadlock`
- `TestScheduler_Good_EmitsProbeEvents`

Implementation:

- Add `NewScheduledModel(model inference.TextModel, cfg SchedulerConfig)`.
- Implement `Schedule(ctx, req)`.
- Implement `CancelRequest(ctx, id)`.
- Forward `Generate`, `Chat`, `Classify`, `BatchGenerate`, `Info`, `Metrics`,
  `Err`, and `Close` to the wrapped model.
- Use a bounded queue and per-request context cancellation.
- Emit `inference.ProbeEventScheduler` events with queue latency,
  first-token latency, request ID, and cancellation status.

Acceptance:

```sh
go test ./go -run 'TestScheduler' -count=1
```

### Task 4: Cache Service

Write tests in `go/cache_test.go`:

- `TestCacheService_Good_WarmStatsClear`
- `TestCacheService_Good_RecordsHitsForOverlappingPrefix`
- `TestCacheService_Bad_RejectsTokenizerMismatch`
- `TestCacheService_Bad_RejectsAdapterMismatch`
- `TestCacheService_Ugly_ClearByLabelsOnlyClearsMatchingBlocks`

Implementation:

- Add `BlockCacheService` implementing `inference.CacheService`.
- `WarmCache` must accept prompt or token IDs.
- Create stable block IDs from model hash, adapter hash, tokenizer hash, token
  span, cache mode, and token contents.
- `CacheStats` must report blocks, memory bytes, disk bytes, hits, misses,
  evictions, hit rate, restore milliseconds, cache mode, and labels.
- `ClearCache` must support label filtering.
- Start with metadata/in-memory blocks. HIP KV backing can replace internals.

Acceptance:

```sh
go test ./go -run 'TestCacheService' -count=1
```

### Task 5: Parser Registry

Write tests in `go/parser_registry_test.go`:

- Qwen `<think>hidden</think>visible` splits reasoning from visible text.
- Gemma channel markers split reasoning from visible text.
- DeepSeek R1 thinking spans parse.
- JSON tool calls parse into `inference.ToolCall`.
- Generic XML fallback parses `<tool name="x">{"a":1}</tool>`.
- Unknown model returns visible text unchanged.

Implementation:

- Add architecture-aware parser lookup.
- Implement `ParseReasoning(tokens []inference.Token, text string)`.
- Implement `ParseTools(tokens []inference.Token, text string)`.
- Wire `rocmModel` or a thin adapter to expose parser interfaces.

Acceptance:

```sh
go test ./go -run 'TestParserRegistry' -count=1
```

### Task 6: Model-Pack Inspection Hardening

Write tests in `go/model_pack_test.go` or extend `go/native_contract_test.go`:

- MiniMax M2 JANGTQ safetensors pack reports `minimax_m2`, `jangtq`, MXTQ
  profile labels, MoE labels, parser labels, and codebook labels.
- Qwen3 GGUF reports architecture, context length, quantization, tokenizer
  hints, and memory plan.
- Gemma pack reports shared-KV or Gemma-specific labels when config indicates
  that shape.
- BERT-like config reports embeddings capability.
- Reranker-like config reports rerank capability.
- Malformed safetensors header fails with a bounded error before reading a huge
  payload.

Implementation:

- Keep header reads bounded by `maxSafetensorsHeaderBytes`.
- Parse tokenizer/chat-template sidecars when present.
- Normalize architecture aliases in one helper.
- Normalize quantization aliases in one helper.
- Keep memory-fit labels prefixed with `memory_plan_`.

Acceptance:

```sh
go test ./go -run 'Test.*ModelPack|TestNativeContract_ModelPack' -count=1
```

### Task 7: State Lifecycle Groundwork

Write tests in `go/state_session_test.go`:

- `WakeState` rejects incompatible model hash unless skip flag is set.
- `WakeState` returns prefix token count, block size, blocks read, and labels.
- `SleepState` writes URI-first refs without exposing runtime handles in JSON.
- `ForkState` creates an independent session handle using the same state refs.
- Missing store returns a Core error with operation context.

Implementation:

- Use `go-inference/state` request/result structs.
- Add ROCm session wrapper that stores runtime-owned handles behind `any`
  fields from the shared request.
- Do not define memvid-specific concrete types in `go-rocm`.
- Return planned or unsupported errors for actual KV restore until ROCm owns KV
  cache pages.

Acceptance:

```sh
go test ./go -run 'TestStateSession' -count=1
```

### Task 8: HIP Runtime Kernel Gate

Write tests before kernels:

- Fake driver allocation/copy/free path is deterministic.
- Shape validation catches missing embeddings, missing output head, mismatched
  layer count, unsupported dtype, and unsupported quantization.
- Decode calls return "native decode kernels are not linked yet" until the
  kernel path exists.

Implementation:

- Keep `hipLoadedModel.Generate`, `Chat`, `Classify`, and `BatchGenerate`
  returning clear kernel errors until a tested kernel path replaces them.
- Add `hipKernelSet` or equivalent narrow interface so tests can inject fake
  kernels.
- Do not make fake tokens look like real model support.

Acceptance:

```sh
go test ./go -run 'TestHIP|TestNativeContract_LoadModel' -count=1
```

### Task 9: First Real Decode Path

Hardware target:

- AMD ROCm Linux machine with 16GB VRAM.
- One small Qwen3 or Gemma GGUF/safetensors fixture.

Implementation order:

1. Token embedding lookup.
2. RMSNorm.
3. Linear projection.
4. RoPE.
5. Single-layer attention fixture.
6. Full small-model prefill.
7. Single-token decode.
8. Greedy sampler.
9. Chat template path.
10. Metrics and probe emission.

Acceptance:

```sh
GO_ROCM_RUN_MODEL_TESTS=1 go test ./go -run 'Test.*Generate|Test.*Decode|Test.*Smoke' -count=1
```

The first smoke test can be tiny. It must prove non-empty deterministic output
and clear metrics. It does not need to be fast.

### Task 10: KV Cache Modes

Write tests:

- fp16 cache stores and restores exact fake blocks.
- q8 cache round-trips within a known tolerance.
- k-q8-v-q4 cache round-trips with larger tolerance and lower byte count.
- Paged block allocation avoids full concatenation for append paths.
- Cache stats report hit rate and restore time.

Implementation:

- Start in package-local fake tensors.
- Add HIP-backed cache pages only after fake tests pass.
- Connect memory planner cache-mode choices to actual cache constructors.

Acceptance:

```sh
go test ./go -run 'Test.*KV|Test.*Cache' -count=1
```

### Task 11: MoE, JANGTQ, and Codebook Runtime

Write tests:

- Router selects top-k experts and emits probe events.
- Lazy expert residency loads only selected experts in 16GB plan.
- JANGTQ/MXTQ descriptor validation rejects invalid bit layouts.
- Packed dequant/projection matches CPU reference for a tiny tensor.
- Codebook lookup matches CPU reference for a tiny tensor.

Implementation:

- Reuse metadata parsing from `go/model_pack.go`.
- Keep CPU reference functions in tests or package-local non-HIP helpers.
- Add HIP kernels only after descriptor validation and CPU reference pass.

Acceptance:

```sh
go test ./go -run 'Test.*MoE|Test.*JANG|Test.*Codebook' -count=1
```

### Task 12: Bench, Eval, and Probe Parity

Write tests:

- Benchmark reports prefill tok/s, decode tok/s, memory peak, active memory,
  cache hit rate, restore time, queue latency, first-token latency, and labels.
- Eval reports samples, token count, loss/perplexity fields when supported, and
  clear unsupported labels when loss is unavailable.
- Probe sink receives scheduler, token, cache pressure, memory pressure, router,
  and residual-summary events when the relevant component is active.

Implementation:

- Keep report structs from `go-inference`.
- Do not invent ROCm-only report schemas unless labels are enough.
- Fake-device tests must be deterministic.

Acceptance:

```sh
go test ./go -run 'Test.*Benchmark|Test.*Evaluate|Test.*Probe' -count=1
```

## 8. Hardware Gate Policy

Default tests must not require ROCm hardware. Hardware tests must check an env
var and skip clearly.

Use these gates:

```sh
GO_ROCM_RUN_HIP_TESTS=1 go test ./go -run 'TestHIP' -count=1
GO_ROCM_RUN_MODEL_TESTS=1 go test ./go -run 'Test.*Smoke|Test.*Generate|Test.*Decode' -count=1
GO_ROCM_RUN_CACHE_TESTS=1 go test ./go -run 'Test.*KV|Test.*Cache' -count=1
```

Expected skip message when disabled:

```text
set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware tests
```

## 9. Error Message Policy

Use clear operation-scoped Core errors:

- `rocm.LoadModel: read model metadata: ...`
- `rocm.hip.LoadModel: HIP driver is not available`
- `rocm.hip.Generate: native decode kernels are not linked yet`
- `rocm.CacheWarm: tokenizer hash mismatch`
- `rocm.WakeState: model hash mismatch`
- `rocm.Parser: unsupported parser family "x"`

Do not silently fall back to CPU, subprocess inference, or an HTTP server.

## 10. Documentation Updates

Update these after implementation phases:

- `README.md`
  - Current supported, experimental, and planned features.
  - Native HIP status.
  - Legacy server build-tag note.
- `docs/architecture.md`
  - Runtime layers: `go-inference` contracts, ROCm backend, HIP driver, kernels,
    cache, state.
  - Memory planner and 16GB ROCm policy.
- `docs/development.md`
  - Non-hardware test gates.
  - Hardware env-var gates.
  - Fixture requirements.
- `docs/history.md`
  - Completed parity phases and remaining kernel work.

## 11. Final Handoff Checklist

- [ ] `git status --short` reviewed and unrelated user changes preserved.
- [ ] `go test ./... -count=1` passes or documented existing failures remain.
- [ ] `GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1` passes.
- [ ] `go test -tags rocm_legacy_server ./... -count=1` passes.
- [ ] Hardware tests are skip-safe without env vars.
- [ ] Capability reports match real callability.
- [ ] Planned features return precise errors rather than pretending to work.
- [ ] No import path in `go-rocm` points to `go-mlx`.
- [ ] Public symbols have tests and examples.
- [ ] Docs explain what is supported, experimental, and planned.
