# go-rocm Goal Working Notes

## 2026-05-11 Baseline

- Starting branch: `dev` tracking `origin/dev`.
- Initial `git status --short`: clean.
- User requested all submodules move to their `dev` branches.
- Current submodules after update:
  - `external/go`: `dev` at `7d10b4486bc50788e35ddb37f477b49b46fb1d94`.
  - `external/go-inference`: `dev` at `f9d1f0367b89c24c794b89853b0a5d81df76acd3`.
  - `external/go-log`: `dev` at `96c2e4700d50e0a48a6c41b112a4fc62fe1a6525`.
- Parent repo now shows modified gitlinks for `external/go`, `external/go-inference`, and `external/go-log`.

### Required Repo-Root Gates

The required commands fail before package compilation because the repo root is
not itself a Go module listed in `go.work`; the managed module subtree is `go/`.

```text
go test ./... -count=1
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
go test -tags rocm_legacy_server ./... -count=1

FAIL    ./... [setup failed]
# ./...
pattern ./...: directory prefix . does not contain modules listed in go.work or their selected dependencies
FAIL
```

### Module-Targeted Baseline Gates

`go test ./... -count=1` from `go/` fails:

```text
--- FAIL: TestBackend_Backend_Available_Bad
    backend_test.go:44: AssertFalse want=false got=true
--- FAIL: TestNativeContract_ModelPackInspectorReadsSidecars_Good
    native_contract_test.go:300: want supported MiniMax/JANGTQ safetensors pack
FAIL    dappco.re/go/rocm
ok      dappco.re/go/rocm/internal/gguf
```

`GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1` from `go/`
fails:

```text
--- FAIL: TestNativeContract_ModelPackInspectorReadsSidecars_Good
    native_contract_test.go:300: want supported MiniMax/JANGTQ safetensors pack
--- FAIL: ExampleROCmAvailable
got:
false
want:
true
FAIL    dappco.re/go/rocm
ok      dappco.re/go/rocm/internal/gguf
```

`go test -tags rocm_legacy_server ./... -count=1` from `go/` fails:

```text
--- FAIL: TestBackend_Backend_Available_Bad
    backend_test.go:44: AssertFalse want=false got=true
--- FAIL: TestModel_Model_Close_Good
panic: runtime error: invalid memory address or nil pointer dereference
dappco.re/go/rocm.(*server).stop(...)
    go/server.go:236
dappco.re/go/rocm.(*rocmModel).Close(...)
    go/model.go:242
FAIL    dappco.re/go/rocm
ok      dappco.re/go/rocm/internal/gguf
ok      dappco.re/go/rocm/internal/llamacpp
```

`go test ./... -count=1` from `external/go-inference/go` passes:

```text
ok dappco.re/go/inference
ok dappco.re/go/inference/anthropic
ok dappco.re/go/inference/ollama
ok dappco.re/go/inference/openai
ok dappco.re/go/inference/parser
ok dappco.re/go/inference/quant/codebook
ok dappco.re/go/inference/quant/jang
ok dappco.re/go/inference/state
ok dappco.re/go/inference/state/filestore
```

## 2026-05-11 Progress

- Updated `external/go`, `external/go-inference`, and `external/go-log` submodules to their latest `dev` branches; refreshed recursive `dev` submodules after `go-inference` advanced to `cb3dc24`.
- Fixed baseline ROCm module failures:
  - environment-dependent availability test expectation,
  - JANGTQ/MXTQ quantization normalization,
  - no-cgo `ROCmAvailable` example,
  - legacy nil server close/generate handling.
- Added or updated:
  - `go-inference` scheduler probe event constants,
  - backend-neutral `go-inference/parser` coverage for Gemma bracket channels, analysis/final channels, Mistral `[TOOL_CALLS]` arrays, and XML tool fallback,
  - backend-neutral `go-inference/parser` and `go-inference/quant` examples, with JANG and codebook validators wired into ROCm model-pack inspection,
  - ROCm scheduler/cancellation wrapper with duplicate in-flight request ID rejection and queue/first-token latency probe labels,
  - scheduler queue-full behavior is explicit and non-blocking for callers once the bounded queue is saturated,
  - scheduler enqueue is synchronized with close and rejects closed or pre-cancelled requests with clear errors,
  - scheduler request IDs are now trimmed before scheduling/cancellation so whitespace IDs are treated as missing rather than stable external IDs,
  - scheduled request labels are cloned at enqueue/handle/stream boundaries so caller mutation cannot alter in-flight scheduler metadata,
  - shared `go-inference/scheduler` now applies the same request-ID trimming and label cloning boundaries as the ROCm scheduler wrapper,
  - shared `go-inference/scheduler` now rejects duplicate in-flight request IDs instead of overwriting the active request map,
  - shared `go-inference/scheduler` close is now idempotent, cancels active requests, stops workers, and rejects schedules after close,
  - scheduler cancellation now delegates unknown request IDs to an underlying cancellable model when one exists, matching the shared scheduler behavior,
  - shared `go-inference/scheduler` cancellation now honors cancelled contexts before delegating unknown request IDs and forwards the caller context to underlying cancellable models, and `SetProbeSink` now forwards sinks to wrapped probeable models like the ROCm scheduler wrapper,
  - scheduler probe events now carry the typed `ProbeScheduler` payload in addition to labels so consumers can read queue depth and latency fields without label parsing,
  - scheduled-model `Generate` and `Chat` now route through the scheduler queue and record enqueue failures in `Err()` instead of bypassing request lifecycle controls,
  - shared `go-inference/scheduler` and ROCm scheduled-model `Classify`/`BatchGenerate` now clear stale scheduler errors on entry and record delegate errors in `Err()`, while the shared scheduler also clears stale errors for successful `Generate`/`Chat`, keeping request bookkeeping consistent across scheduled generation and direct non-streaming delegates,
  - scheduled-model `Close` is now idempotent for the wrapped model instead of only guarding queue shutdown,
  - metadata-first cache service with prefix-hit warm accounting, deterministic restore milliseconds, full-block creation after prefix hits, optional opaque `go-inference/state` disk refs, and disk-byte accounting,
  - cache block IDs and prefix hits now include effective request model/adapter/tokenizer hashes so unconfigured services cannot cross-hit incompatible requests,
  - cache warm byte accounting now accepts `kv_key_width`/`kv_value_width` hints and constructs vector-shaped package-local KV pages for planner-aligned cache sizing,
  - cache warm byte accounting now accepts `kv_cache_block_size`, and cache block identity/prefix-hit matching include KV backing, block size, and key/value widths so different KV shapes cannot collide on the same token prefix,
  - cache warm results now clone returned block labels so callers cannot mutate cache service compatibility metadata through result refs,
  - cache service coverage now proves constructor labels, request labels, warm result labels, warm embedded stats labels, returned block-ref labels, and `CacheStats` labels are isolated from caller mutation while preserving label-based clear behavior,
  - cache warm hits now return the cached block's effective labels so disk refs and KV shape telemetry remain visible on exact cache hits,
  - disk-backed cache ref payloads now include their URI in serialized labels before writing to the opaque state store,
  - cache stats and benchmark cache labels now report the effective warmed block cache mode when a warm call overrides the service default mode,
  - memory-planner labels now publish KV block size and per-token key/value widths so planned cache modes can be warmed with the same package-local KV constructor path,
  - cache stats and benchmark cache-pressure probes now carry cached-token counts plus KV width/backing labels for planner-shaped cache telemetry,
  - benchmark reports now mirror cache stats labels under `cache.*`, including cached tokens, KV block size, KV key/value widths, and opaque disk-ref labels when configured,
  - loaded-model adapter load/unload now resets cache service state so pre-adapter prompt blocks are not reused across adapter changes,
  - parser registry backed by the shared parser package with explicit Qwen/Gemma/MiniMax/DeepSeek/GPT-OSS/Mistral/Kimi/GLM/Hermes/Granite coverage, Mistral `[TOOL_CALLS]` array support, and reasoning/tool examples,
  - ROCm parser registry fixtures now directly cover MiniMax thinking tags and GPT-OSS channel markers in addition to the shared parser tests,
  - loaded-model parser helpers now fall back to `modelType` when `modelInfo.Architecture` is empty, avoiding generic parser selection for models that only carry family metadata there,
  - state wake/sleep/fork groundwork with examples, wake/sleep compatibility checks, planned labels for non-KV placeholders, and binary `rocm/kv-cache+json` sleep/wake refs for package-local KV pages,
  - state wake/sleep compatibility now falls back to model architecture and tokenizer kind when hash metadata is absent, while retaining hash mismatch checks as the strongest guard,
  - state wake now rejects missing entry/index URIs with an explicit ROCm operation-context error before calling the backing store,
  - state sleep placeholder writes now carry merged backend/session/request tags into the state store, matching runtime-owned KV snapshot refs,
  - runtime-owned KV state sleep/wake labels now carry KV block size, page/token counts, and key/value widths from package-local cache stats,
  - sleep result `StateRef` labels now mirror state/cache labels plus chunk IDs so URI-first refs remain self-describing without runtime handles,
  - planned/non-KV wake bundle refs now mirror wake labels like runtime-owned KV wake refs, avoiding unlabeled planned state refs,
  - sleep result bundle refs now point at the URI actually written by the ROCm state store path instead of reporting an unwritten requested bundle URI,
  - state wake/sleep results now clone top-level and nested ref labels so caller mutation cannot alter session/request metadata or sibling refs across subsequent wake/sleep calls,
  - state fork coverage now proves runtime-owned KV snapshots restore into independent fork sessions without aliasing the source runtime cache,
  - loaded ROCm models now keep a persistent state session so `WakeState` restored package-local KV runtime can be serialized by a later `SleepState` call instead of falling back to placeholder state,
  - metadata `RestoreState` now closes the previous runtime before installing the replacement session and keeps the previous state/runtime intact when close fails, so failed metadata restores cannot orphan an unclosed runtime behind new labels,
  - model close, adapter load, and adapter unload now close persistent state runtimes before clearing model pointers or mutating native adapter state; state close failures keep the old runtime reachable and avoid native adapter mutation so failed lifecycle transitions can be retried without orphaning restored KV state,
  - loaded-model adapter changes and close now reset the persistent state session along with cache state so restored KV refs cannot survive an adapter/runtime identity change,
  - OpenAI-compatible chat, non-streaming Responses, Anthropic Messages, Ollama chat/generate, capability/cache/cancel/embedding/rerank service helpers,
  - examples for OpenAI Responses/service mux and Anthropic/Ollama compatibility handlers,
  - ROCm service mux coverage now exercises mounted cache warm/stats/clear endpoints against `rocmModel` and cancel against `ScheduledModel`,
  - shared OpenAI cache-warm handler now rejects empty prompt/token requests as HTTP 400 before calling backend cache services, with ROCm mux coverage for that validation path,
  - shared OpenAI cache-clear and cancel handlers now reject empty model IDs explicitly before resolving models, and cancel rejects empty request IDs before calling cancellable models,
  - shared OpenAI service helpers now include examples for embeddings, rerank, capabilities, cache stats, cache warm, cache clear, and cancel handlers,
  - ROCm wire-service tests proving embedding/rerank endpoints return not-implemented responses until native kernels exist,
  - ROCm OpenAI service mux coverage now proves native `rocmModel` embedding/rerank endpoints surface explicit kernel-not-linked errors when optional native kernels are absent,
  - Anthropic compatibility now rejects empty converted message content before resolving/running models,
  - Ollama compatibility handlers now reject empty chat messages and empty generate prompts before resolving or running models,
  - model-pack tokenizer/config/architecture fixtures for Qwen/Gemma/Mistral/Mixtral/Phi/DeepSeek/GPT-OSS/Kimi/GLM/Hermes/Granite/BERT,
  - additional model-pack alias fixtures and normalization coverage for DeepSeek R1, generic MiniMax, Llama, Qwen3 MoE, and Qwen3 Next,
  - model-pack quantization alias coverage for `quantization_config` MXFP4/NVFP4 metadata, including `format`-only configs,
  - tokenizer `model_max_length` now fills missing model-pack context length while retaining tokenizer metadata labels,
  - model-pack task-specific params now tolerate non-string values so BERT rerank metadata is not rejected by bool/numeric config fields,
  - model-pack inspection now labels BERT sequence-classification packs with `classifier_model=true` and a planned `classify` capability carrying `classify_path=bert_sequence_classifier` alongside rerank metadata,
  - BERT `text-classification` task-specific params now prefer classifier metadata over the generic BERT embedding hint so classification packs do not publish an embedding capability by mistake,
  - BERT sequence-classification architecture metadata no longer implies rerank by itself; rerank labels/capabilities now require an explicit rerank model or task hint while plain sequence classifiers stay classify-only,
  - model-pack `task_specific_params` now accept arbitrary JSON values, including scalar boolean/string entries, so loose HF-style rerank/classification task hints do not make `config.json` fail parsing,
  - tokenizer config parsing now accepts scalar or array token IDs so array `eos_token_id` sidecars do not discard tokenizer/chat-template metadata,
  - safetensors inspection now rejects malformed or reversed tensor `data_offsets` instead of reporting misleading tensor summaries,
  - safetensors inspection now checks tensor `data_offsets` against the actual file payload length so truncated weight files cannot report valid tensor metadata or `Supported=true`,
  - safetensors inspection now rejects tensor entries missing required `dtype` or `shape` fields before reporting tensor counts or support,
  - safetensors inspection now rejects headers that contain only `__metadata__` and no real tensor entries,
  - safetensors inspection now skips only the reserved `__metadata__` entry; other double-underscore keys are validated as tensor entries instead of silently hidden,
  - safetensors inspection now rejects duplicate top-level tensor keys before JSON map decoding can overwrite earlier tensor metadata,
  - safetensors inspection now rejects unsupported tensor dtypes and tensor byte spans that do not match `shape * dtype-size`, including overflow-safe shape byte accounting,
  - safetensors inspection now rejects overlapping tensor `data_offsets` so tensor payload ranges cannot alias each other,
  - sharded safetensors inspection now aggregates tensor counts, header bytes, payload bytes, and dtype sets across valid shards instead of reporting only the last shard,
  - malformed GGUF/safetensors weight metadata now sets `weight_metadata_valid=false` and keeps model-pack reports from claiming `Supported=true` based only on parseable sidecars,
  - mixed-shard model packs now clear partial GGUF/safetensors summary labels when any discovered weight file is malformed,
  - safetensors model-pack reports now require detected architecture metadata, so missing or malformed `config.json` cannot claim `Supported=true` based only on a valid weight header,
  - model-pack memory-fit labels are now skipped for unsupported inspection reports so malformed packs do not mix `Supported=false` with fit telemetry,
  - memory-planner tests for small, 24GB, 64GB, and long-context cache-mode transitions,
  - cache warm byte accounting backed by constructible package-local KV cache pages for planner-selected `fp16`, `q8`, and `k-q8-v-q4` modes,
  - HIP tensor validation for dtype, quantization, projection shape/model dimensions, byte sizes, kernel status labels, and kernel-not-linked tests,
  - HIP dtype validation now accepts known GGUF numeric tensor block types, including quantized tensors without a `TypeName`, while still rejecting unknown tensor types,
  - fake HIP driver nil/unavailable/malloc/free failure coverage, idempotent loaded-model close semantics, load-time adapter cleanup coverage, and tensor-copy cleanup coverage for copy/read failures,
  - HIP load validation now rejects negative/overflowing tensor data offsets and required zero-byte embedding/output tensors before device allocation,
  - HIP load validation now checks tensor byte ranges against the model file before any device allocation, avoiding malloc/copy work for impossible offsets,
  - HIP load validation now rejects empty and duplicate tensor names before allocation to prevent tensor map overwrites and leaked device pointers,
  - explicit typed `hipKernelSet` dispatch/not-linked stub seams for future decode/prefill/projection kernels,
  - explicit LoRA adapter load validation and HIP not-linked failure coverage that preserves active adapter state on failed loads,
  - direct LoRA adapter validator coverage now exercises tiny output-head, small LM-head, and classifier adapter format/target/shape/rank/alpha/bias failures plus active-adapter preconditions, device-backed output/LM-head weight read failures, classifier base-weight validation, classifier bias merge, and tiny attention-weighted-output helper validation,
  - public and loaded HIP adapter identity results now clone label maps and target-key slices at `LoadAdapter`/`ActiveAdapter` boundaries so caller mutation cannot corrupt active adapter metadata or native fallback identity,
  - deterministic token embedding lookup, single-head and multi-head attention, causal prefill attention, decode-with-KV, integrated tiny LM prefill/decode, fp16/q8 projection, LoRA projection, RMSNorm, RoPE, greedy and top-k/temperature sampler references, logit, entropy, selected-head, and layer-coherence probe summaries, prompt-lookup draft, speculative-accept, embedding mean-pool, rerank cosine, cross-entropy/perplexity, distillation KL, and GRPO advantage reference fixtures for the first HIP kernel ladder steps,
  - transformer reference coverage now separately exercises embedding/table/token validation, tiny-LM config and KV-state shape failures, RMSNorm empty/zero/non-finite cases, RoPE position/base/non-finite cases, attention shape failures, sampler tie-breaks, invalid top-k/temperature, and vector-split validation before future HIP transformer kernels rely on the fixture,
  - HIP projection reference coverage now separately exercises invalid rows/cols, input/weight/bias shape failures, q8 zero/negative/non-finite scale rejection, and float16 signed-zero/subnormal/Inf/NaN conversion paths before future classifier/projection kernels rely on the fixture,
  - decode helper/reference coverage now includes prompt-lookup truncation, too-short/no-match and longest-suffix behavior, speculative first-mismatch/draft-longer/empty-draft acceptance cases, encoder-derived lookup tokens, decoder text mapping, and missing-encoder lookup validation,
  - MoE/JANGTQ/codebook reference coverage now includes router tie-breaks and invalid top-k/logits, non-finite router/JANGTQ/codebook/residual inputs, lazy-expert invalid counts/routes, JANGTQ descriptor format/group/scale/shape/short-pack failures plus signed unpack checks, codebook empty-code and shape validation, and residual empty-value validation,
  - LoRA projection reference coverage now separately exercises base input/weight/bias shape failures, LoRA A/B shape failures, rank validation, zero alpha, and non-finite alpha rejection before future HIP LoRA kernels rely on the fixture,
  - embedding/rerank reference coverage now exercises empty token batches, empty/mismatched dimensions, zero-vector normalization, cosine similarity, empty document sets, text-count mismatch, vector-width mismatch, zero-vector rerank errors, and negative `topN` all-results behavior,
  - probe reference coverage now splits validation paths for logits/head-selection/layer-coherence/entropy helpers and includes stable large-logit entropy expectations for future HIP probe kernels,
  - training reference helper coverage now includes large-logit cross-entropy stability plus empty/mismatched input, empty row/vocabulary, invalid target, invalid/non-finite temperature, non-finite logits/rewards, and empty-reward validation paths for the CPU reference loss/distillation/GRPO fixtures,
  - import-boundary enforcement now blocks current `forge.lthn.sh`, legacy `forge.lthn.ai`, canonical `dappco.re`, and GitHub aliases for forbidden workflow/runtime siblings (`go-ai`, `go-ml`, `go-mlx`, `go-rag`, and `go-ratelimit`) so package-first ROCm code cannot accidentally adopt concrete workflow/provider/rate-limit dependencies, and the shared `go-inference` contract subtree is explicitly scanned for both workflow siblings and concrete ROCm runtime imports,
  - package-local KV cache pages covering fp16, q8, and k-q8-v-q4 round trips, paged appends, byte counts, hit rate, restore timing, and validated binary snapshot refs,
  - package-local KV stats now label block size, page count, token count, and active key/value widths for probe/bench/cache telemetry without exposing tensors,
  - package-local KV cache pages now preserve per-token key/value vector widths across paged append, restore, and snapshots while retaining scalar fake-block compatibility,
  - package-local KV pages now insert deterministically by token range and reject overlapping appended or snapshot-restored pages so runtime-owned snapshots cannot alias ambiguous token spans,
  - package-local KV caches now enforce one key/value vector shape across appended and snapshot-restored pages so decode cannot silently mix incompatible KV widths,
  - cache disk KV refs now annotate portable KV snapshots with cache block and compatibility identity metadata, keep legacy raw snapshots readable, and cold restore rejects mismatched raw or annotated snapshot refs, including contradictory embedded identity/cache-shape labels, instead of accepting same-mode/same-length stale KV payloads as hits,
  - cache disk metadata refs now reject mismatched kind, model, adapter, tokenizer, token-start, size, identity-label, cache-shape, and runtime-only KV/device labels even when a corrupted ref carries the requested block ID, while live warm requests scrub caller-supplied disk runtime labels and unbacked disk URIs, and metadata warm requests scrub cache-shape/runtime-only KV/device labels before computing/reporting block/result/stats metadata,
  - package-local KV pages can now be mirrored into HIP device allocations with explicit key/value page copies, idempotent cleanup, and rollback on copy failure; kernels still cannot consume those pages yet,
  - HIP KV device mirrors now expose `inference.CacheStats` labels (`kv_backing=hip_device_mirror`, `kv_device_backing=mirrored`, page/token/width counts) for probes and opt-in hardware smoke checks,
  - HIP KV device mirrors can now snapshot mirrored pages back through device-to-host copies into the portable `rocm/kv-cache+json` encoding, and state sleep records those refs with `kv_serialize=device_mirror` while wake restores them as package-local pages,
  - state wake coverage now proves HIP device-mirror snapshot refs restore through the public wake path as package-local KV pages, loaded `rocmModel` instances best-effort remirror restored portable KV refs back into HIP device pages, and capability details distinguish device-mirror serialization/remirror from still-pending fully HIP-owned restore,
  - state sessions now clone model/tokenizer identity label maps at construction and fork boundaries, so caller-owned identity metadata and forked-session metadata cannot alias each other,
  - state runtime ownership now closes replaced HIP device KV mirrors when wake installs a new runtime, metadata `RestoreState` replaces a session, model close runs without a native runtime, or the benchmark LoRA baseline clears wrapper state,
  - HIP KV device mirrors can now extend a decode result by allocating and copying only the appended token's key/value page, then refreshing the descriptor table and transferring old page ownership only after success; append telemetry labels before/after page/token counts and descriptor-refresh success, and scratch append mirrors now track borrowed vs owned pages so uncommitted cleanup cannot free source pages,
  - HIP KV device mirrors now expose kernel descriptors with token ranges, key/value widths, device pointers, byte sizes, and encodings, and reject descriptor access after close,
  - HIP KV device mirrors now serialize those kernel descriptors into a versioned fixed-width little-endian header/page table and can copy that table into HIP memory with rollback/idempotent cleanup, so future HIP launch code has a stable device-resident cache descriptor ABI,
  - HIP KV device mirrors plus descriptor tables now produce a validated launch descriptor carrying descriptor pointer, byte count, version, mode code, block/page/token counts, and vector widths, plus fixed 64-byte KV launch bytes and 96-byte decode launch packets carrying token/position metadata for future kernel launch plumbing,
  - HIP token IDs now have a little-endian device-buffer upload helper with nil/unavailable-driver validation, non-negative token validation, copy-failure rollback, and idempotent cleanup coverage, plus fixed 64-byte prefill launch packets carrying token-buffer pointer/count/bytes, cache mode, block size, and KV vector widths for future prefill kernels,
  - HIP projection requests now have temporary device buffers for input, fp16/q8/f32 weights, optional bias, and output plus a fixed 96-byte projection launch packet covering pointers, byte counts, shape, encoding, bias flags, and q8 scale, with output readback rejecting wrong-width or non-finite kernel results,
  - HIP LoRA projection requests now have temporary device buffers for input, fp32 base weights, fp32 A/B adapter weights, optional bias, and output plus a fixed 128-byte launch packet covering pointers, byte counts, shape, rank, alpha, and bias flags, with output readback rejecting wrong-width or non-finite kernel results,
  - HIP embedding mean-pool and rerank cosine requests now have temporary device buffers plus fixed 64-byte launch packets covering pointers, byte counts, vector shapes, normalization flags, and output score vectors for future embedding/rerank kernels, with output readback rejecting wrong-width or non-finite kernel results,
  - HIP launch plumbing now has an optional fake-testable kernel launcher contract with one-dimensional launch config validation for prefill, decode, and projection packets; the cgo system driver can load a caller-supplied HSACO from `GO_ROCM_KERNEL_HSACO`, copy the launch packet to device memory, and call `hipModuleLaunchKernel`, while default builds still fail clearly until a compatible module is supplied,
  - HIP drivers now include device-to-host copy support; the fake projection launcher consumes the 96-byte projection packet, writes the output device buffer, and fake linked projection reads that buffer back before returning,
  - fake HIP driver allocations now use non-overlapping address-like pointer ranges so tests that exercise byte-offset device pointer arithmetic, such as loaded embedding-row reads, cannot alias unrelated fake allocations,
  - fake prefill/decode launchers now consume their fixed launch packets and validate the referenced token buffer, KV launch descriptor, and descriptor-table memory before reporting successful launches,
  - HIP RMSNorm, RoPE, greedy sampler, attention, and toy tiny prefill/decode readbacks now reject wrong-width output buffers, non-finite vectors or scores, invalid attention probability values, out-of-range result token IDs, and device-output copy failures before results leave the launch wrappers,
  - loaded Qwen/Gemma small-decode device-weight helper paths now reuse the validated RMSNorm/projection readback boundary and reject non-finite loaded embedding rows before composing typed decode results,
  - loaded embedding tables, loaded f32 classifier/bias/codebook-style tensors, converted fp16 classifier base weights, and decoded tiny output-head weights now reject non-finite values before host-side composition or adapter application can consume them,
  - HIP cross-entropy/perplexity, distillation KL, and GRPO advantage normalization now have fixed 64-byte launch packets, device buffers for logits/targets/teacher/reward outputs, fake-driver execution through the CPU references, validated float64 readback, kernel-source ABI coverage, and opt-in hardware-smoke coverage for the compiled `rocm_cross_entropy_loss`, `rocm_distillation_kl_loss`, and `rocm_grpo_advantage` symbols,
  - RMSNorm and RoPE launch preflight now rejects non-finite epsilon/base values at both request and binary-packet boundaries so fake and real HIP kernels do not receive NaN/Inf scalar launch parameters,
  - composed Qwen/Gemma small-decode request validation now rejects non-finite RMSNorm epsilon and RoPE base values in both direct fixture and loaded-model paths before primitive launch requests are built,
  - tiny prefill/decode launch coverage now exercises missing and ambiguous output-head encodings, output-weight length mismatches, unsupported manual packet encodings, q8 packet-scale rejection, prior-KV length mismatches, and output-weight payload decoding failures,
  - loaded tiny-model config coverage now exercises embedding/output rank and dimension validation, model metadata shape mismatch handling, byte-count validation, and required device-pointer checks before tiny launch packets are built,
  - loaded tiny-model packed/codebook output-head coverage now exercises packed JANGTQ byte-count validation plus codebook output-code byte counts, table dtype/rank/dimension/byte-count validation, and required codebook table device pointers,
  - loaded tiny-model request validation coverage now exercises tiny-specific out-of-vocabulary prefill/decode tokens and KV-width mismatches after generic request preflight but before tiny launch dispatch,
  - `kernels/rocm_kernels.hip` now provides ABI-checked `rocm_prefill`, `rocm_decode`, `rocm_projection`, `rocm_jangtq_projection`, `rocm_codebook_lookup`, `rocm_lora_projection`, `rocm_embedding_mean_pool`, `rocm_rerank_cosine`, `rocm_rms_norm`, `rocm_rope`, `rocm_greedy_sample`, `rocm_attention`, `rocm_moe_router`, `rocm_moe_lazy_experts`, `rocm_tiny_prefill`, `rocm_tiny_decode`, `rocm_cross_entropy_loss`, `rocm_distillation_kl_loss`, and `rocm_grpo_advantage` symbols: prefill/decode consume validated launch packets and referenced memory, projection performs the toy fp16/q8 row projection used by the Go fixtures, JANGTQ/MXTQ projection unpacks signed 2/4/8-bit toy weights, codebook lookup expands toy VQ code IDs into float vectors, LoRA projection applies a toy fp32 low-rank delta over a base projection, embedding mean-pool and rerank cosine execute toy vector fixtures, RMSNorm/RoPE/greedy sampling/single-head attention/MoE top-k routing/lazy-expert bitmaps execute deterministic transformer primitive fixtures, cross-entropy, distillation KL, and GRPO advantage normalization write toy loss/perplexity/KL/advantage outputs for eval/training stepping stones, package-local Qwen/Gemma small decode smokes compose those primitives through RMSNorm, fp16 Q/K/V/O and LM-head projections, RoPE, attention, and greedy sampling using fixture and loaded device-resident weights, tiny prefill runs a toy embedding-attention-output fixture that writes toy KV buffers, logits, final-token attention weights, and a greedy result buffer, and tiny decode consumes those toy prior KV vectors, appends the decoded token embedding, and writes updated KV/logits/attention/greedy buffers; tiny prefill/decode now accept fp32, fp16, or q8 output heads through explicit launch-packet encodings, while loaded tiny JANGTQ/codebook output heads reuse the tiny attention/KV step and recompute logits through `rocm_jangtq_projection` or `rocm_codebook_lookup` plus `rocm_projection`,
  - loaded HIP models now select a projection/embedding/rerank/LoRA-linked kernel set when the driver supports kernel launch and `GO_ROCM_KERNEL_HSACO` is set; tiny vocab-major f32 loaded models with f32/f16/raw-q8/JANGTQ/codebook output heads can run toy prefill/decode/generate plus experimental model-level embedding mean-pool, rerank cosine, and `rocm-tiny-lora` output-head adapter application against device-resident tensor pointers; fixture-scale Qwen/Gemma loaded models can route typed `DecodeToken` through the small decode composition by reading the request token's f32 embedding row from device memory, appending package-local KV across fp16, q8, and k-q8-v-q4 cache modes, optionally applying an experimental LM-head LoRA adapter through `rocm_lora_projection`, and incrementally appending supplied device KV mirrors transactionally on success while preserving the original host/device KV on append failure; normal generation/classification still report decode/prefill not-linked status until real forward kernels land,
  - model capability reports now make benchmark and evaluation details/labels follow linked decode/prefill/classification status so loaded experimental kernels no longer advertise stale not-linked caveats for those wrappers,
  - capability reports now clone model identity label maps and adapter label/target-key metadata at report construction, keeping report mutation isolated from caller-owned identity inputs,
  - eval reports now distinguish linked classification paths with no target labels as `loss_status=not_requested` instead of reusing the prefill-not-linked loss/perplexity labels, label BERT classifier-backed eval as `classify_path=bert_sequence_classifier`, and emit the shared classification logit/entropy probe events,
  - eval loss now uses an optional native cross-entropy loss hook when a loaded HIP model reports `cross_entropy_kernel=linked`, labels the result with `loss_backend=hip`, `loss_kernel=linked`, and `loss_kernel_name=rocm_cross_entropy_loss`, and preserves the CPU reference fallback plus non-failing error labels for non-HIP or failed loss-kernel paths,
  - eval reports now carry cross-entropy fixture status labels (`loss_kernel`, `loss_kernel_name`, `loss_scope`) even when loss is unsupported, not requested, or satisfied by the CPU reference fallback,
  - BERT-style f32 `word_embeddings.weight` packs can now load without an LM head and run the experimental loaded embedding mean-pool plus embedding-cosine rerank path when the embedding/rerank kernel set is linked,
  - public embedding/rerank wrappers now clone native-owned result vectors, score labels, and result labels before stamping ROCm model identity so optional native implementations cannot leak mutable result storage to callers,
  - public classify/batch-generate wrappers now clone native-owned logits and generated-token slices before stripping logits, emitting classification probes, recording metrics, or returning results, so non-streaming text paths no longer expose mutable native result storage,
  - BERT sequence-classification packs with f32/f16 `classifier.weight`/`score.weight` and optional f32/f16 `classifier.bias`/`score.bias` can now classify prompts and rank query/document pairs through the loaded f32 embedding table, mean-pool kernel, and projection path, deterministically pair classifier/scorer weights with their matching bias aliases, and report experimental BERT sequence-classifier capability/status labels while keeping production cross-encoder status caveated,
  - HIP projection launch/reference/source paths now support f32 weights alongside fp16 and q8, covering common classifier-head tensors without fake quantization,
  - loaded BERT sequence-classifier models can now load `rocm-classifier-lora` adapters targeting `classifier.weight`/`score.weight`; direct classify and rerank apply the active adapter through `rocm_lora_projection`, label the adapter runtime, and preserve the existing load-failure behavior that keeps the active adapter unchanged,
  - scorer-head alias and fp16 classifier-head coverage now proves `score.weight` plus optional `score.bias` routes through the same BERT direct classify, positive-logit rerank, and classifier LoRA adapter path as `classifier.weight`,
  - `Example_bertClassifierLoRAAdapter` now documents the fixture `rocm-classifier-lora` JSON format and proves the active adapter changes the experimental BERT sequence-classifier rerank route through `rocm_lora_projection`,
  - q8 projection and tiny output-head paths now reject zero, negative, NaN, and infinite q8 scales before packet creation or loaded tiny-model dispatch,
  - tiny loaded-model JANGTQ/MXTQ output-head metadata now validates bits, group size, scale, and packed byte count before dispatch; invalid packed-output metadata fails before allocation or kernel launch,
  - tiny loaded-model codebook/VQ output-head metadata now validates scalar code dimensions, code byte counts, table presence, f32 table dtype, table shape, and table byte count before dispatch; toy codebook output heads expand through `rocm_codebook_lookup` and then project logits through `rocm_projection`,
  - tiny loaded-model dispatch is now gated to the `tiny` architecture fixture so fixture-scale Qwen/Gemma small-decode models do not accidentally claim the toy tiny prefill/decode/generate route,
  - backend capability reports now use runtime-selected kernel status, so an explicitly configured HSACO is visible in `projection_kernel`, `embedding_kernel`, `rerank_kernel`, and `lora_kernel` labels without falsely marking decode/generation as supported,
  - opt-in hardware smoke coverage now verifies compiled fp16 and q8 `rocm_projection` output through direct and loaded-model launches, compiled `rocm_jangtq_projection` output for the packed MXTQ/JANGTQ fixture, compiled `rocm_codebook_lookup` output for the VQ lookup fixture, compiled `rocm_lora_projection` output for the low-rank adapter fixture, compiled `rocm_embedding_mean_pool` and `rocm_rerank_cosine` output for embedding/rerank fixtures, launches the compiled `rocm_prefill` and `rocm_decode` packet-consumer symbols against real token buffers, HIP KV mirrors, and descriptor-table allocations for fp16, q8, and k-q8-v-q4 cache modes, checks compiled `rocm_rms_norm`, `rocm_rope`, `rocm_greedy_sample`, `rocm_attention`, `rocm_moe_router`, `rocm_moe_lazy_experts`, composed Qwen3 small decode smokes using fixture and loaded device-resident weights, typed loaded-model Qwen3 small `DecodeToken` from a device embedding row with fp16 package-local KV plus q8/k-q8-v-q4 package-local and incrementally appended device-KV paths, toy `rocm_tiny_prefill` KV/logits/attention/greedy readback, toy `rocm_tiny_decode` consuming prefill-written KV, loaded tiny-model generation from device-resident tensors, and fp16/q8/codebook tiny output-head variants, and verifies device-written status markers from reserved launch-packet fields,
  - typed HIP prefill/decode request/results can now carry both a device KV mirror and its HIP descriptor table, validate them against the package-local KV shape before decode, and fake linked kernels incrementally append decoded device KV pages for future real-kernel ownership,
  - the default not-linked decode stub now runs decode launch-packet preflight when callers provide device KV resources, so malformed token, position, or device metadata fails before the expected kernel-not-linked error,
  - the default not-linked prefill stub now validates prompt/token, cache-mode, token ID, and KV width inputs before returning the expected prefill kernel-not-linked error for valid requests,
  - the default not-linked projection stub now validates projection weights, shape, encoding exclusivity, and q8 scale before returning the expected projection kernel-not-linked error for valid requests,
  - the default not-linked chat stub now validates message arrays, known roles, and non-empty content before returning the expected decode kernel-not-linked error for valid chat requests,
  - the default not-linked classification stub now validates prompt batches and empty prompt strings before returning the expected prefill kernel-not-linked error for valid classification requests,
  - the default not-linked batch-generate stub now validates prompt batches and empty prompt strings before returning the expected decode kernel-not-linked error for valid batch requests,
  - public ROCm chat wrappers now validate message arrays before native dispatch so malformed requests do not get hidden behind kernel-not-linked status or fake native streams,
  - public ROCm classify and batch-generate wrappers now validate prompt batches before native dispatch so malformed requests do not silently succeed through fake or metadata-only native models,
  - public ROCm batch-generate wrappers now record native batch errors in `Err()` so callers can inspect the last failure consistently with streamed generation and classification paths,
  - public ROCm classify and batch-generate wrappers now clear stale `Err()` state at the start of each call so a successful non-streaming text request cannot leave an older validation or native failure visible,
  - repo-root Go test gates now use a workspace module with local replacements and cgo-only root meta-tests that delegate to the real `go/` and `external/go-inference/go` packages instead of failing at workspace pattern expansion,
  - HIP reference matrix flattening now lives in package code rather than test-only code so compile-only root imports catch non-test package dependencies,
  - linked HIP tiny text paths now validate chat message arrays and classify/batch prompt batches before HSACO-backed generation/classification dispatch, matching the not-linked stub and public wrapper preflight behavior,
  - linked HIP tiny text paths now return a cancelled context before validation or zero-token generation shortcuts, matching the default not-linked stub cancellation ordering,
  - public ROCm generation/chat/classify/batch/embedding/rerank wrappers now return a cancelled context before validation or native dispatch, so cancellation-before-prefill behavior is consistent from the public package surface down to linked and not-linked HIP paths,
  - loaded HIP embedding and rerank calls now validate empty input text, missing query/documents, and empty rerank documents before returning kernel-not-linked status for otherwise valid requests,
  - public ROCm embedding and rerank wrappers now validate request shape before optional-interface dispatch, so malformed requests do not get hidden behind kernel-not-linked status when the native model lacks those optional methods,
  - fake linked decode now treats device-mirrored KV and descriptor-table updates transactionally by cloning package-local KV before append and only transferring old device pages after the appended key/value page and new descriptor table succeed, preserving original host/device KV on copy failure,
  - opt-in `GO_ROCM_RUN_CACHE_TESTS=1` now exercises HIP KV page allocation/copy/free for a toy package-local cache instead of unconditionally skipping native KV cache ownership,
  - fake linked decode now requires a prefill KV cache and appends the decoded token into package-local KV pages, keeping native decode fixtures honest about KV ownership,
  - HIP prefill/decode kernel requests now carry cache mode and KV key/value vector-width contracts so fake linked kernels exercise planner-shaped KV pages before real kernels land,
  - capability report status for package-local KV snapshots as experimental while HIP device-mirror snapshots are a stepping stone and fully HIP-owned production KV snapshots remain pending,
  - CPU reference MoE top-k routing probes, lazy expert residency, residual-summary probes, JANGTQ/MXTQ packed dequant/projection, and codebook lookup, plus fixed HIP MoE-router, MoE-lazy-expert, JANGTQ/MXTQ packed-projection, and codebook-lookup launch fixtures with fake-driver and hardware readback coverage,
  - HIP MoE-router, MoE-lazy-expert, JANGTQ, and codebook launch coverage now rejects non-finite router/JANGTQ/codebook inputs before device upload, rejects missing/shape-mismatched device buffers, propagates device-output copy failures, requires the MoE router status pointer plus exact one-word status buffer shape in serialized launch packets, rejects router readback when the kernel status marker is missing/mismatched or the returned expert IDs/probabilities are outside the expected domain, rejects lazy-expert readback when resident bitmaps have the wrong byte count or non-binary flags, and rejects JANGTQ/codebook readback when output buffers have the wrong width or non-finite values,
  - metadata-only MoE/JANGTQ/codebook capabilities now label their fixture launch kernels (`rocm_moe_router`, `rocm_moe_lazy_experts`, `rocm_jangtq_projection`, `rocm_codebook_lookup`) and required production integration while staying `runtime_status=metadata_only`,
  - `Example_metadataOnlyFixtureCapabilities` documents the metadata-only fixture-kernel and pending production-integration labels for MoE/JANGTQ/codebook capability reports,
  - loaded-model capability, bench, and eval reports driven by native kernel status,
  - capability guards keeping production model-family LoRA application plus SFT/distillation/GRPO training planned until native adapter wiring, training interfaces, and kernels exist,
  - planned LoRA-training/distillation/GRPO capabilities now carry explicit `runtime_status=planned`, `training_kernel=not_linked`, `training_interface=not_implemented`, and exact required future-kernel labels (`lora_backward`, `distillation_forward_loss`, `grpo_rollout_policy`); distillation and GRPO also expose explicit `distillation_kernel`/`grpo_kernel`-aware toy fixture-kernel labels for `rocm_distillation_kl_loss` and `rocm_grpo_advantage` without advertising production training-interface support,
  - loaded HIP models now expose package-local `RunDistillationKLLoss` and `RunGRPOAdvantage` hooks for the toy distillation KL and GRPO advantage fixture kernels when native kernels are linked, while still avoiding the shared `DistillTrainer` and `GRPOTrainer` interfaces,
  - `Example_trainingCapabilityReport` documents the planned/not-linked training capability labels and exact required-kernel mapping for callers that inspect backend reports,
  - `Example_evaluationLossCapabilityReport` documents the eval loss fixture labels, including default `loss_kernel=not_linked` and linked `loss_kernel=linked` states for `rocm_cross_entropy_loss`,
  - eval quality probes that use the generation surface when available and record unavailable native decode without failing token-count eval,
  - benchmark/eval surfaces now normalize nil contexts before generation, dataset iteration, and qualitative probes,
  - opt-in HIP/model/cache smoke tests that skip without hardware env vars and `GO_ROCM_MODEL_PATH`,
  - in-module import-boundary test preventing `go-rocm` from importing forbidden concrete workflow/runtime packages,
  - skip-safe higher-level import-boundary test scanning local `go-ai`, `go-ml`, and `go-inference` checkouts for direct `go-rocm` runtime imports,
  - bench/eval shared cache, benchmark cache/memory pressure probe emission, and unsupported loss/perplexity labels,
  - benchmark reports now propagate cache stats errors instead of emitting partial cache telemetry,
  - benchmark reports now run `WarmupRuns` over every configured prompt without polluting measured counters, label warmup/measured run shape, and report measured probe counts while forwarding events to any configured sink while excluding warmup and internal LoRA baseline comparison probes,
  - benchmark reports now derive first-token latency plus prefill/decode/total duration labels from measured runs and label the measured operation count,
  - benchmark reports now include active memory as a shared field plus active/peak memory byte labels in addition to memory-pressure probe payloads,
  - benchmark reports now measure active LoRA adapter overhead by comparing the active adapter run with a temporary native baseline run and restoring the original adapter identity afterward,
  - benchmark LoRA overhead measurement now clears wrapper adapter/cache/state identity during the no-adapter baseline and leaves the active adapter cleared if native restore fails, so labels do not claim an adapter is active after a failed restore,
  - benchmark LoRA overhead measurement now closes restored state runtimes before temporarily unloading the native adapter and reports `state_close_failed` without touching native adapter state when close fails, matching the public lifecycle transition policy,
  - classification now strips native logits unless callers opt in with `WithLogits`, emits compact logit and entropy probe summaries when logits are requested, and reports `probe.logits` as experimental only when prefill/classification kernels are linked,
  - eval reports now compute experimental classification cross-entropy/perplexity when dataset samples provide `target_token_id` labels and the model returns classification logits, batch optional loss-logit classification according to `EvalConfig.BatchSize`, and preserve non-failing unsupported/logits-unavailable labels otherwise,
  - eval reports now label sample/token counts, unsupported loss/perplexity status, and quality-probe pass/fail counts,
  - eval quality-probe failures now preserve the first operation-scoped generation error in labels while keeping token-count eval non-failing,
  - cache-disk capability and benchmark labels now report experimental `go-inference/state` metadata refs plus portable package-local KV snapshot disk refs, exact cold disk-ref rehydrate, and `disk_cache_restore` labels while still documenting fully HIP-owned disk KV as pending,
  - block-cache KV warming now optionally mirrors warmed and cold disk-restored portable KV snapshots into HIP device pages when a live ROCm driver is present, labels mirrored device bytes/pages/tokens, closes mirrored pages on cache clear, adapter changes, and model close, and keeps portable package-local blocks with failure labels when device remirroring fails,
  - cache, prompt-cache, KV-snapshot, and cache-disk capability reports now advertise best-effort HIP device remirroring with `kv_device_backing=best_effort_remirror` while keeping `fully_hip_owned` and `native_prefill_reuse` labels pending,
  - benchmark cache labels and cache-pressure probes now preserve block-cache device-remirror labels (`kv_device_backing`, page/token counts, and device bytes) from warmed KV cache stats,
  - direct `BlockCacheService.Close` coverage and examples now prove mirrored HIP KV pages are freed, close is idempotent, and stats are empty afterward,
  - mirrored block-cache close failure coverage now proves HIP page-free errors propagate through `Close`/`ClearCache` and stop `rocmModel` close/load-adapter/unload-adapter before native state is cleared or native adapter/close calls run,
  - state wake close-failure coverage now proves previous HIP device KV runtime free errors are returned with `close previous state runtime` context and prevent both `StateSession` and `rocmModel` wake from installing the newly restored snapshot runtime,
  - `StateSession.Close` now has an example covering HIP device KV cleanup and matching free/allocation accounting,
  - direct `StateSession.Close` bad-path coverage now proves runtime cleanup errors propagate and keep the runtime reference intact for callers to inspect,
  - `StateSession.SleepState` now has device-mirror snapshot failure coverage proving device-to-host copy errors are wrapped with sleep context, keep the runtime intact, and do not write a partial state ref,
  - `StateSession.SleepState` now has package-local and HIP device KV write-failure coverage proving `PutBytes` errors are wrapped with the correct state-ref context and leave the live runtime installed,
  - `StateSession.SleepState` now rejects text-only stores for package-local and HIP device KV runtimes with `binary state store is missing` while leaving the active runtime untouched,
  - `StateSession.SleepState` placeholder refs now have missing-store, non-writer, and text-write-failure coverage with `rocm.SleepState`/`write state ref` context and preserved metadata tags,
  - loaded `rocmModel.ForkState` now best-effort remirrors restored portable KV refs into HIP device pages for forked sessions, labels mirrored or failed remirror status, and leaves forked sessions package-local when remirroring fails,
  - `StateSession.ForkState` now has nil-session and wake-failure wrapping coverage that preserves `rocm.ForkState` plus `wake forked state` context without returning partial fork/wake results,
  - `Example_rocmModel_ForkState` now documents loaded-model fork remirroring labels and verifies the forked session owns a HIP device KV runtime,
  - opt-in `GO_ROCM_RUN_CACHE_TESTS=1` hardware smoke now includes block-cache warm remirroring into HIP device pages before the lower-level descriptor and launch-packet checks,
  - prompt-cache capability and benchmark labels now report experimental for metadata/package-local warm, hit accounting, and state refs while keeping native prefill reuse pending,
  - exported `SpeculativeDecode` and `PromptLookupDecode` package helpers now wrap the shared `go-inference/decode` acceptance harness with examples, Good/Bad/Ugly tests, model stream error propagation, and capability status that stays planned until a loaded experimental decode path is linked,
  - `rocmModel` now implements the shared `StatefulModel` metadata-only `StateBundle` capture/restore surface while keeping durable KV bytes on the URI-first wake/sleep path, with examples plus nil/cancelled/incompatible bad-path tests,
  - shared `go-inference` generation, benchmark, and eval quality-probe options now carry cloned textual stop sequences, and ROCm state bundles, scheduler sampler replay, OpenAI/Responses/Anthropic/Ollama adapters, public generate forwarding, benchmark warmup/measured/LoRA-baseline runs, eval quality probes, legacy llama-server requests, and public ROCm stream/batch stop enforcement preserve them without aliasing caller-owned slices or leaking stop text across token chunks,
  - public tokenizer/template boundaries now clone native-owned encoded token slices and caller-owned decode/template slices before returning or dispatching, so native tokenizer implementations cannot leak mutable token buffers or mutate caller messages/token IDs,
  - README/development/architecture/history docs.

### Current Passing Gates

Run from `go/`:

```text
go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test ./... -run 'TestHIPKernels|TestHIPRuntime|TestHIPDriverFake' -count=1 -v
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test -exec=/bin/true ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test ./... -run 'TestHIPHardware|TestNativeDecodeSmoke|TestHIPRuntimeHardwareCache' -count=1 -v
PASS with skips for GO_ROCM_RUN_HIP_TESTS, GO_ROCM_RUN_MODEL_TESTS, and GO_ROCM_RUN_CACHE_TESTS

GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./... -run 'TestHIPHardware.*KernelSource' -count=1 -v
PASS, including jangtq-projection, codebook-lookup, moe-router, moe-lazy-experts, loaded-small-q8, and loaded-small-k-q8-v-q4 typed DecodeToken hardware subtests

go test ./go -run 'TestHIPRuntime_LoadModelRunsBERTSequenceClassifier|TestHIPRuntime_LoadModelBERTSequenceClassifier|TestNativeContract_RocmModelCapabilitiesUseLoRAKernelStatus' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference
ok dappco.re/go/inference/anthropic
ok dappco.re/go/inference/bench
ok dappco.re/go/inference/decode
? dappco.re/go/inference/eval [no test files]
ok dappco.re/go/inference/ollama
ok dappco.re/go/inference/openai
ok dappco.re/go/inference/parser
ok dappco.re/go/inference/quant/codebook
ok dappco.re/go/inference/quant/jang
ok dappco.re/go/inference/scheduler
ok dappco.re/go/inference/state
ok dappco.re/go/inference/state/filestore

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS

Repeated after BERT classifier LoRA adapter slice:

go test ./go -run 'TestHIPRuntime_LoadModelRunsBERTSequenceClassifier|TestHIPRuntime_LoadModelBERTSequenceClassifier|TestNativeContract_RocmModelCapabilitiesUseLoRAKernelStatus' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference
ok dappco.re/go/inference/anthropic
ok dappco.re/go/inference/bench
ok dappco.re/go/inference/decode
? dappco.re/go/inference/eval [no test files]
ok dappco.re/go/inference/ollama
ok dappco.re/go/inference/openai
ok dappco.re/go/inference/parser
ok dappco.re/go/inference/quant/codebook
ok dappco.re/go/inference/quant/jang
ok dappco.re/go/inference/scheduler
ok dappco.re/go/inference/state
ok dappco.re/go/inference/state/filestore

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Run from the repo root:

```text
go test ./go -run 'TestHIPKernelSource|TestHIPKernels_(TinyPrefill|TinyDecode|Attention|RMSNorm|RoPE|Greedy|Projection)|TestHIPMoE|TestHIPJANGTQ|TestHIPCodebook|TestHIPTransformerReferenceTiny(Prefill|Decode)' -count=1 -v
PASS

go test ./go -run 'TestHIPSmallDecode|TestHIPRuntime_Load(ModelRunsSmallDecodeSmokeWhenHSACOConfigured|edSmallDecodeConfig)|TestHIPRuntime_LoadModelRunsTinyPrefillDecodeWhenHSACOConfigured_Good|TestHIPKernelSource|TestHIPKernels_(TinyPrefill|TinyDecode|Attention|RMSNorm|RoPE|Greedy|Projection|ProjectionLaunchArgs)|TestHIPMoE|TestHIPJANGTQ|TestHIPCodebook|TestHIPProjectionReference|TestKVCache_Good_MirrorsPagesToHIPDevice|TestKVCache_Bad_Device' -count=1 -v
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS

GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'TestHIPHardwareTransformerKernelSource' -count=1 -v
PASS, including jangtq-projection, codebook-lookup, moe-router, moe-lazy-experts, loaded-small-q8, and loaded-small-k-q8-v-q4 typed DecodeToken hardware subtests

git diff --check
PASS

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test -c ./go -o /tmp/go-rocm-darwin-arm64.test
PASS

GOOS=darwin GOARCH=amd64 CGO_ENABLED=0 go test -c ./go -o /tmp/go-rocm-darwin-amd64.test
PASS

go test ./go -run 'TestImportBoundary|TestNativeContract_RocmBackendCapabilities_Good|TestNativeContract_RocmModelCapabilities' -count=1 -v
--- PASS: TestImportBoundary_NoForbiddenRuntimeImports_Good (0.00s)
--- PASS: TestImportBoundary_HigherLevelPackagesDoNotImportROCm_Good (0.00s)
--- PASS: TestImportBoundary_SharedContractsDoNotImportRuntimeOrWorkflowPackages_Good (0.00s)
--- PASS: TestNativeContract_RocmBackendCapabilities_Good (0.00s)
--- PASS: TestNativeContract_RocmModelCapabilities_Ugly (0.00s)
--- PASS: TestNativeContract_RocmModelCapabilitiesUseNativeKernelStatus_Good (0.00s)
--- PASS: TestNativeContract_RocmModelCapabilitiesUseEmbeddingRerankKernelStatus_Good (0.00s)
--- PASS: TestNativeContract_RocmModelCapabilitiesUseLoRAKernelStatus_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.011s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference 0.008s
ok dappco.re/go/inference/anthropic 0.002s
ok dappco.re/go/inference/bench 0.003s
ok dappco.re/go/inference/decode 0.003s
? dappco.re/go/inference/eval [no test files]
ok dappco.re/go/inference/ollama 0.002s
ok dappco.re/go/inference/openai 0.003s
ok dappco.re/go/inference/parser 0.003s
ok dappco.re/go/inference/quant/codebook 0.002s
ok dappco.re/go/inference/quant/jang 0.002s
ok dappco.re/go/inference/scheduler 0.053s
ok dappco.re/go/inference/state 0.002s
ok dappco.re/go/inference/state/filestore 0.003s

(cd external/go && go test ./... -count=1)
ok dappco.re/go 0.270s

(cd external/go-log/go && go test ./... -count=1)
ok dappco.re/go/log 0.002s
ok dappco.re/go/log/tests/cli/log 0.002s
```

Run from `external/go-inference/go`:

```text
go test ./... -count=1
ok dappco.re/go/inference
ok dappco.re/go/inference/anthropic
ok dappco.re/go/inference/bench
ok dappco.re/go/inference/decode
?  dappco.re/go/inference/eval [no test files]
ok dappco.re/go/inference/ollama
ok dappco.re/go/inference/openai
ok dappco.re/go/inference/parser
ok dappco.re/go/inference/quant/codebook
ok dappco.re/go/inference/quant/jang
ok dappco.re/go/inference/scheduler
ok dappco.re/go/inference/state
ok dappco.re/go/inference/state/filestore
```

Latest focused verification after cache-disk capability reporting:

```text
go test ./go -run 'TestNativeContract|TestCacheService' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS
```

Latest verification after decode helper surface:

```text
go test ./go -run 'TestDecode|TestNativeContract|Example.*Decode' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS
```

Latest verification after prompt-cache capability reporting:

```text
go test ./go -run 'TestNativeContract|TestCacheService' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS
```

Latest focused verification after metadata-only StateBundle capture/restore:

```text
go test ./go -run 'TestStateSession|TestNativeContract' -count=1
ok dappco.re/go/rocm
```

Latest full verification after metadata-only StateBundle capture/restore:

```text
cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS
```

Latest verification after portable KV snapshot cache-disk refs and StateBundle examples:

```text
go test ./go -run 'TestCacheService|TestNativeContract' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestStateSession|Example.*State|Example_rocmModel' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS
```

Latest verification after loaded-model wake remirrors portable KV refs to HIP device mirrors:

```text
go test ./go -run 'TestStateSession|TestNativeContract' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS
```

Latest verification after exact cold disk-ref cache rehydrate:

```text
go test ./go -run 'TestCacheService|TestNativeContract' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS
```

Latest verification after BERT-style embedding-only loaded path:

```text
go test ./go -run 'TestHIPRuntime_LoadModelRuns(BERT|TinyEmbed)|TestHIPRuntime_LoadModelRejects|TestNativeContract_RocmModelCapabilitiesUseEmbedding|TestNativeContract_RocmModelEmbeddings' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS
```

Latest verification after BERT classifier LoRA example and score alias coverage:

```text
go test ./go -run 'Example_bertClassifierLoRAAdapter|TestHIPRuntime_LoadModelRunsBERT.*(Classifier|Score).*LoRA|TestHIPRuntime_LoadModelBERTSequenceClassifier|TestNativeContract_RocmModelCapabilitiesUseLoRAKernelStatus' -count=1 -v
PASS

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after training capability labels and example:

```text
go test ./go -run 'Example_trainingCapabilityReport|TestNativeContract_RocmBackendCapabilities_Good|TestNativeContract_RocmModelDoesNotImplementTrainingSurfaces_Ugly' -count=1 -v
PASS

go test ./go -run 'TestNativeContract_RocmBackendCapabilities_Good|TestNativeContract_RocmModelDoesNotImplementTrainingSurfaces_Ugly|TestNativeContract_RocmModelCapabilities' -count=1 -v
PASS

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Repeated after tightening exact training required-kernel labels and development docs:

```text
go test ./go -run 'Example_trainingCapabilityReport|TestNativeContract_RocmBackendCapabilities_Good|TestNativeContract_RocmModelDoesNotImplementTrainingSurfaces_Ugly' -count=1 -v
PASS

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after metadata-only MoE/JANGTQ/codebook fixture labels and example:

```text
go test ./go -run 'Example_(trainingCapabilityReport|metadataOnlyFixtureCapabilities)|TestNativeContract_RocmBackendCapabilities_Good|TestNativeContract_ModelPackInspectorReadsSidecars_Good' -count=1 -v
PASS

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after loaded tiny JANGTQ output-head logits:

```text
go test ./go -run 'TestHIPRuntime_LoadModelRunsTinyPrefillDecodeWhenHSACOConfigured_Good|TestHIPRuntime_LoadedTinyJANGTQOutputValidation_Bad|TestHIPRuntime_LoadedTinyQ8ScaleValidation_Bad|TestHIPRuntime_LoadModelRunsTinyLoRAAdapterWhenHSACOConfigured_Good|TestHIPJANGTQProjectionLaunch' -count=1 -v
PASS

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after loaded tiny codebook output-head logits:

```text
go test ./go -run 'TestHIPRuntime_LoadModelRunsTinyPrefillDecodeWhenHSACOConfigured_Good|TestHIPRuntime_LoadedTiny(Codebook|JANGTQ)OutputValidation_Bad|TestHIPRuntime_LoadedTinyQ8ScaleValidation_Bad|TestHIPRuntime_LoadModelRunsTinyLoRAAdapter(WithCodebookOutput)?WhenHSACOConfigured_Good|TestHIP(CodebookLookup|JANGTQProjection)Launch' -count=1 -v
PASS

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after Qwen/Gemma small-decode LM-head LoRA adapter:

```text
go test ./go -run 'TestHIPRuntime_LoadModelRunsSmallDecode(LoRAAdapter|Smoke)WhenHSACOConfigured_Good|TestHIPRuntime_LoadModelRunsTinyPrefillDecodeWhenHSACOConfigured_Good|TestHIPRuntime_LoadModelRunsTinyLoRAAdapter(WithCodebookOutput)?WhenHSACOConfigured_Good|TestHIPLoRAProjectionLaunch' -count=1 -v
PASS

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after state runtime replacement cleanup:

```text
go test ./go -run 'TestStateSession_(Good_WakeStateClosesPreviousRuntime|Good_RocmModelRestoreStateClosesPreviousRuntime|Good_RocmModelCloseClosesStateWithoutNative|Good_RocmModelWakeStateRemirrorsKVSnapshotToHIPDevice|Good_RocmModelWakeStateKeepsPackageLocalKVOnDeviceMirrorFailure)' -count=1 -v
PASS

go test ./go -run 'Test(StateSession|KVCache|CacheService)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after incremental decode device-KV append:

```text
go test ./go -run 'TestKVCache_Good_DeviceMirrorAppendsDecodeTokenIncrementally|TestHIPRuntime_LoadModelRunsTinyPrefillDecodeWhenHSACOConfigured_Good|TestHIPRuntime_LoadModelRunsSmallDecodeSmokeWhenHSACOConfigured_Good|TestHIPSmallDecode_Bad|TestHIPRuntime_LoadedSmallDecodeConfig_Bad' -count=1 -v
PASS

go test ./go -run 'TestKVCache_(Good_DeviceMirrorAppendsDecodeTokenIncrementally|Bad_DeviceMirrorAppendRollbackOnDescriptorFailure|Bad_DeviceMirrorAppendScratchCloseDoesNotFreeSourcePages|Good_MirrorsPagesToHIPDevice)' -count=1 -v
PASS

go test ./go -run 'Test(KVCache|CacheService|HIPRuntime_LoadModelRunsTinyPrefillDecodeWhenHSACOConfigured_Good|HIPRuntime_LoadModelRunsSmallDecode(LoRAAdapter|Smoke)WhenHSACOConfigured_Good|HIPSmallDecode_Bad|HIPRuntime_LoadedSmallDecodeConfig_Bad)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestHIPRuntime_LoadModelRunsTinyLoRAAdapter(WithCodebookOutput)?WhenHSACOConfigured_Good|TestHIPRuntime_LoadModelRunsSmallDecodeLoRAAdapterWhenHSACOConfigured_Good|TestHIPLoRAProjectionLaunch|TestHIP(CodebookLookup|JANGTQProjection)Launch' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after append telemetry descriptor-refresh labels:

```text
go test ./go -run 'TestKVCache|TestHIPRuntime_LoadModelRunsTinyPrefillDecodeWhenHSACOConfigured_Good|TestHIPRuntime_LoadModelRunsSmallDecodeSmokeWhenHSACOConfigured_Good|TestHIPSmallDecode_Bad|TestHIPRuntime_LoadedSmallDecodeConfig_Bad' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestHIPRuntime_LoadModelRunsTinyLoRAAdapter(WithCodebookOutput)?WhenHSACOConfigured_Good|TestHIPRuntime_LoadModelRunsSmallDecodeLoRAAdapterWhenHSACOConfigured_Good|TestHIPLoRAProjectionLaunch|TestHIP(CodebookLookup|JANGTQProjection)Launch' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after benchmark/eval capability kernel-status labels:

```text
go test ./go -run 'TestNativeContract|TestNativeCapability|Example_metadataOnlyFixtureCapabilities|Example_trainingCapabilityReport' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestHIPRuntime_LoadModelRunsTinyPrefillDecodeWhenHSACOConfigured_Good|TestHIPRuntime_LoadModelRunsSmallDecodeSmokeWhenHSACOConfigured_Good' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after linked-prefill eval loss labels:

```text
go test ./go -run 'TestNativeContract_Evaluate|TestNativeContract_BenchmarkAndEvaluateUseModelSurface_Ugly|TestNativeContract_RocmModelCapabilitiesUseNativeKernelStatus_Good' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after BERT sequence-classifier direct classify path:

```text
go test ./go -run 'TestHIPRuntime_LoadModelRunsBERT(EmbedAndRerankWithoutOutputHead|SequenceClassifier(Rerank|LoRAAdapter|RerankWithF16Head)|ScoreHead)|TestHIPRuntime_LoadModelBERTSequenceClassifier|TestNativeContract_RocmModelCapabilitiesUse(NativeKernelStatus|EmbeddingRerankKernelStatus)_Good|TestNativeContract_Evaluate' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after BERT classifier-backed eval loss labels:

```text
go test ./go -run 'TestHIPRuntime_LoadModelRunsBERTSequenceClassifierRerankWhenHSACOConfigured_Good|TestNativeContract_Evaluate(LinkedPrefillWithoutLossTargetsLabelsNotRequested_Good|UsesClassifyLogitsForLoss_Good|BatchesClassifyLogitLoss_Good)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after direct BERT classifier LoRA classify coverage:

```text
go test ./go -run 'TestHIPRuntime_LoadModelRunsBERTSequenceClassifier(LoRAAdapter|Rerank)WhenHSACOConfigured_Good|TestHIPRuntime_LoadModelRunsBERTScoreTensorLoRAAdapterWhenHSACOConfigured_Good' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after BERT classifier-backed eval probe coverage:

```text
go test ./go -run 'TestHIPRuntime_LoadModelRunsBERTSequenceClassifierRerankWhenHSACOConfigured_Good|TestNativeContract_ClassifyLogitProbes_Good|TestNativeContract_Evaluate(LinkedPrefillWithoutLossTargetsLabelsNotRequested_Good|UsesClassifyLogitsForLoss_Good|BatchesClassifyLogitLoss_Good)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after BERT classifier eval `classify_path` labels:

```text
go test ./go -run 'TestHIPRuntime_LoadModelRunsBERTSequenceClassifierRerankWhenHSACOConfigured_Good|TestNativeContract_ClassifyLogitProbes_Good|TestNativeContract_Evaluate(LinkedPrefillWithoutLossTargetsLabelsNotRequested_Good|UsesClassifyLogitsForLoss_Good|BatchesClassifyLogitLoss_Good)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after BERT classifier direct-classify fp16/score-alias coverage:

```text
go test ./go -run 'TestHIPRuntime_LoadModelRunsBERT(SequenceClassifier(Rerank|RerankWithF16Head|LoRAAdapter)|ScoreTensorLoRAAdapter)WhenHSACOConfigured_Good' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest verification after BERT sequence-classifier model-pack classify metadata:

```text
go test ./go -run 'TestNativeContract_ModelPackInspector(RerankTaskParamsAllowNonStrings_Good|ArchitectureFixtures_Good)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after BERT text-classification task metadata precedence:

```text
go test ./go -run 'TestNativeContract_ModelPackInspector(RerankTaskParamsAllowNonStrings_Good|TextClassificationTaskParamsPreferClassifier_Good|ArchitectureFixtures_Good)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after BERT sequence-classification rerank disambiguation:

```text
go test ./go -run 'TestNativeContract_ModelPackInspector(RerankTaskParamsAllowNonStrings_Good|TextClassificationTaskParamsPreferClassifier_Good|SequenceClassificationWithoutRerankIsClassifierOnly_Good|ArchitectureFixtures_Good)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after scalar `task_specific_params` parsing:

```text
go test ./go -run 'TestNativeContract_ModelPackInspector(RerankTaskParamsAllowNonStrings_Good|TextClassificationTaskParamsPreferClassifier_Good|SequenceClassificationWithoutRerankIsClassifierOnly_Good|TaskParamsAllowScalarValues_Good|ArchitectureFixtures_Good)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after safetensors payload-bound validation:

```text
go test ./go -run 'TestNativeContract_ModelPackInspector(MalformedSafetensors_Bad|MalformedSafetensorsOffsets_Bad|TruncatedSafetensorsPayload_Bad|MalformedSafetensorsShardClearsWeightLabels_Bad|ArchitectureFixtures_Good)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after safetensors required-field validation:

```text
go test ./go -run 'TestNativeContract_ModelPackInspector(MalformedSafetensors_Bad|MalformedSafetensorsOffsets_Bad|TruncatedSafetensorsPayload_Bad|SafetensorsMissingRequiredFields_Bad|MalformedSafetensorsShardClearsWeightLabels_Bad|ArchitectureFixtures_Good)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after safetensors no-tensor validation:

```text
go test ./go -run 'TestNativeContract_ModelPackInspector(MalformedSafetensors_Bad|MalformedSafetensorsOffsets_Bad|TruncatedSafetensorsPayload_Bad|SafetensorsMissingRequiredFields_Bad|SafetensorsRequiresTensorEntries_Bad|MalformedSafetensorsShardClearsWeightLabels_Bad|ArchitectureFixtures_Good)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after safetensors dtype/shape-byte validation:

```text
go test ./go -run 'TestNativeContract_ModelPackInspector(MalformedSafetensors_Bad|MalformedSafetensorsOffsets_Bad|TruncatedSafetensorsPayload_Bad|SafetensorsMissingRequiredFields_Bad|SafetensorsRequiresTensorEntries_Bad|SafetensorsValidatesDTypeShapeByteSpan_Bad|MalformedSafetensorsShardClearsWeightLabels_Bad|ArchitectureFixtures_Good)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after sharded safetensors summary aggregation:

```text
go test ./go -run 'TestNativeContract_ModelPackInspector(AggregatesSafetensorsShardSummaries_Good|MalformedSafetensors_Bad|MalformedSafetensorsOffsets_Bad|TruncatedSafetensorsPayload_Bad|SafetensorsMissingRequiredFields_Bad|SafetensorsRequiresTensorEntries_Bad|SafetensorsValidatesDTypeShapeByteSpan_Bad|MalformedSafetensorsShardClearsWeightLabels_Bad|ArchitectureFixtures_Good)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after exact safetensors `__metadata__` skip handling:

```text
go test ./go -run 'TestNativeContract_ModelPackInspector(AggregatesSafetensorsShardSummaries_Good|MalformedSafetensors_Bad|MalformedSafetensorsOffsets_Bad|TruncatedSafetensorsPayload_Bad|SafetensorsMissingRequiredFields_Bad|SafetensorsRequiresTensorEntries_Bad|SafetensorsValidatesNonMetadataDoubleUnderscoreKeys_Bad|SafetensorsValidatesDTypeShapeByteSpan_Bad|MalformedSafetensorsShardClearsWeightLabels_Bad|ArchitectureFixtures_Good)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after overlapping safetensors offset rejection:

```text
go test ./go -run 'TestNativeContract_ModelPackInspector(AggregatesSafetensorsShardSummaries_Good|MalformedSafetensors_Bad|MalformedSafetensorsOffsets_Bad|TruncatedSafetensorsPayload_Bad|SafetensorsMissingRequiredFields_Bad|SafetensorsRequiresTensorEntries_Bad|SafetensorsValidatesNonMetadataDoubleUnderscoreKeys_Bad|SafetensorsValidatesDTypeShapeByteSpan_Bad|SafetensorsRejectsOverlappingTensorOffsets_Bad|MalformedSafetensorsShardClearsWeightLabels_Bad|ArchitectureFixtures_Good)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after not-linked prefill request preflight:

```text
go test ./go -run 'TestHIPKernels_(NotLinkedErrors_Bad|NotLinkedPrefillPreflightsRequest_Bad|NotLinkedDecodePreflightsDeviceKV_Bad|CancelledContext_Ugly|RequestValidation_Bad)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestHIPKernels' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after not-linked projection request preflight:

```text
go test ./go -run 'TestHIPKernels_(NotLinkedErrors_Bad|NotLinkedProjectPreflightsRequest_Bad|NotLinkedPrefillPreflightsRequest_Bad|CancelledContext_Ugly|RequestValidation_Bad)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestHIPKernels' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after not-linked classification prompt preflight:

```text
go test ./go -run 'TestHIPKernels_(NotLinkedErrors_Bad|NotLinkedClassifyPreflightsPrompts_Bad|NotLinkedProjectPreflightsRequest_Bad|NotLinkedPrefillPreflightsRequest_Bad|CancelledContext_Ugly|RequestValidation_Bad)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestHIPKernels' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after not-linked batch-generate prompt preflight:

```text
go test ./go -run 'TestHIPKernels_(NotLinkedErrors_Bad|NotLinkedBatchGeneratePreflightsPrompts_Bad|NotLinkedClassifyPreflightsPrompts_Bad|NotLinkedProjectPreflightsRequest_Bad|NotLinkedPrefillPreflightsRequest_Bad|CancelledContext_Ugly|RequestValidation_Bad)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestHIPKernels' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after loaded embedding/rerank request preflight:

```text
go test ./go -run 'TestHIPRuntime_LoadedTinyEmbedAndRerank(NotLinked_Bad|PreflightBeforeNotLinked_Bad)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestHIPRuntime_LoadedTinyEmbedAndRerank|TestHIPRuntime_LoadModelRuns.*Rerank|TestHIPRuntime_LoadModelRuns.*Embed' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract_RocmModelEmbeddingsAndRerank|TestOpenAI_NewOpenAIServiceMux_Bad_ROCmModel(Embedding|Rerank)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after public embedding/rerank wrapper preflight:

```text
go test ./go -run 'TestNativeContract_RocmModelEmbeddingsAndRerank(Dispatch_Good|NotLinked_Bad|PreflightBeforeNotLinked_Bad)|TestHIPRuntime_LoadedTinyEmbedAndRerank(NotLinked_Bad|PreflightBeforeNotLinked_Bad)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract_RocmModelEmbeddingsAndRerank|TestOpenAI_NewOpenAIServiceMux_Bad_ROCmModel(Embedding|Rerank)|TestHIPRuntime_LoadedTinyEmbedAndRerank|TestHIPRuntime_LoadModelRuns.*Rerank|TestHIPRuntime_LoadModelRuns.*Embed' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after public classify/batch prompt preflight:

```text
go test ./go -run 'TestNativeContract_(TextBatchPreflightRejectsEmptyPrompts_Bad|ClassifyWithLogitsEmitsLogitAndEntropyProbes_Good|ClassifyWithoutLogitsStripsNativeLogits_Bad)|TestHIPKernels_(NotLinkedBatchGeneratePreflightsPrompts_Bad|NotLinkedClassifyPreflightsPrompts_Bad)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract_(TextBatchPreflightRejectsEmptyPrompts_Bad|Classify|EvaluateBatchesClassify|EvaluateUsesClassify|RocmModelCapabilities)|TestHIPKernels' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after duplicate safetensors tensor-key rejection:

```text
go test ./go -run 'TestNativeContract_ModelPackInspector(AggregatesSafetensorsShardSummaries_Good|MalformedSafetensors_Bad|MalformedSafetensorsOffsets_Bad|TruncatedSafetensorsPayload_Bad|SafetensorsMissingRequiredFields_Bad|SafetensorsRequiresTensorEntries_Bad|SafetensorsValidatesNonMetadataDoubleUnderscoreKeys_Bad|SafetensorsRejectsDuplicateTensorKeys_Bad|SafetensorsValidatesDTypeShapeByteSpan_Bad|SafetensorsRejectsOverlappingTensorOffsets_Bad|MalformedSafetensorsShardClearsWeightLabels_Bad|ArchitectureFixtures_Good)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after public/HIP chat message preflight:

```text
go test ./go -run 'TestNativeContract_ChatPreflightRejectsInvalidMessages_Bad' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestHIPKernels_NotLinkedChatPreflightsMessages_Bad' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract_(TextBatchPreflightRejectsEmptyPrompts_Bad|ChatPreflightRejectsInvalidMessages_Bad|ProbeSinkReceivesGeneratedTokens_Good|Benchmark|Eval)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestHIPKernels_NotLinked(Errors_Bad|ChatPreflightsMessages_Bad|ClassifyPreflightsPrompts_Bad|BatchGeneratePreflightsPrompts_Bad)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after batch-generate native error recording:

```text
go test ./go -run 'TestNativeContract_(BatchGenerateRecordsNativeError_Bad|TextBatchPreflightRejectsEmptyPrompts_Bad|ChatPreflightRejectsInvalidMessages_Bad|Classify|Benchmark|Eval)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after non-streaming text `Err()` clearing:

```text
go test ./go -run 'TestNativeContract_(NonStreamingTextSuccessClearsLastError_Good|BatchGenerateRecordsNativeError_Bad|TextBatchPreflightRejectsEmptyPrompts_Bad|ChatPreflightRejectsInvalidMessages_Bad|Classify|Benchmark|Eval)' -count=1
ok dappco.re/go/rocm

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after repo-root workspace gate:

```text
go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

GOWORK=off go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOWORK=off go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

GOWORK=off GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after linked tiny text-path request preflight:

```text
go test ./go -run 'TestHIPRuntime_LoadedTinyTextPathsPreflightRequests_Bad' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestHIPRuntime_LoadModelRunsTiny(PrefillDecode|LoRAAdapter|EmbedAndRerank)|TestHIPKernels_NotLinked(ChatPreflightsMessages_Bad|ClassifyPreflightsPrompts_Bad|BatchGeneratePreflightsPrompts_Bad)' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]
```

Latest focused verification after linked tiny text-path cancellation precedence:

```text
go test ./go -run 'TestHIPRuntime_LoadedTinyTextPaths(PreferCancelledContext_Ugly|PreflightRequests_Bad)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestHIPRuntime_LoadModelRunsTiny(PrefillDecode|LoRAAdapter|EmbedAndRerank)|TestHIPKernels_CancelledContext_Ugly|TestHIPKernels_NotLinked(ChatPreflightsMessages_Bad|ClassifyPreflightsPrompts_Bad|BatchGeneratePreflightsPrompts_Bad)' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after public wrapper cancellation precedence:

```text
go test ./go -run 'TestNativeContract_PublicWrappersPreferCancelledContext_Ugly' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract_(PublicWrappersPreferCancelledContext_Ugly|TextBatchPreflightRejectsEmptyPrompts_Bad|ChatPreflightRejectsInvalidMessages_Bad|RocmModelEmbeddingsAndRerankPreflightBeforeNotLinked_Bad|NonStreamingTextSuccessClearsLastError_Good|BatchGenerateRecordsNativeError_Bad)' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after shared scheduler cancellation/probe forwarding and scheduled-model `Err()` bookkeeping:

```text
go test ./go -run 'TestScheduler_(Good_NonStreamingDelegatesClearSchedulerErr|Bad_NonStreamingDelegatesRecordErr)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestScheduler' -count=1
ok dappco.re/go/rocm

go test ./external/go-inference/go/scheduler -run 'TestModel_(GenerateAndChatClearStoredErr_Good|NonStreamingDelegatesClearStoredErr_Good|NonStreamingDelegatesRecordErr_Bad)' -count=1
ok dappco.re/go/inference/scheduler

go test ./external/go-inference/go/scheduler -run 'TestModel_(CancelRequest_UsesContextAndDelegatesIt_Bad|SetProbeSink_ForwardsToBaseProbeable_Good)' -count=1
ok dappco.re/go/inference/scheduler

go test ./external/go-inference/go/scheduler -count=1
ok dappco.re/go/inference/scheduler

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after transactional metadata `RestoreState` replacement:

```text
go test ./go -run 'TestStateSession_(Good_RocmModelRestoreStateClosesPreviousRuntime|Bad_RocmModelRestoreStateCloseFailureKeepsPreviousState)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestStateSession' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestCacheService|TestKVCache' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract_(RocmModelRestores|RocmModel|.*State|AdapterLifecycle|CloseGoodIdempotentClearsRuntimeState)' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after transactional close/adapter state-runtime transitions:

```text
go test ./go -run 'TestNativeContract_(CloseBadStateCloseFailureKeepsRuntime_Bad|LoadAdapterBadStateCloseFailureDoesNotCallNative_Bad|UnloadAdapterBadStateCloseFailureDoesNotCallNative_Bad|AdapterLifecycle_Good|LoadAdapterBadNativeFailureKeepsActiveAdapter_Bad|CloseGoodIdempotentClearsRuntimeState)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestStateSession' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestCacheService|TestKVCache' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after LoRA benchmark state-runtime close failure handling:

```text
go test ./go -run 'TestNativeContract_BenchmarkLoRA(StateCloseFailureSkipsNativeUnload_Bad|RestoreFailureClearsActiveAdapter_Bad|MeasuresActiveLoRAOverhead_Good)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestStateSession|TestCacheService|TestKVCache' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract_Benchmark' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after cache result-label isolation coverage:

```text
go test ./go -run 'TestCacheService_Good_ReturnsClonedResultLabelsAndStats|TestCacheService_Good_WarmCacheReturnsClonedBlockLabels' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestCacheService|TestKVCache|TestStateSession' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after adapter identity clone boundaries:

```text
go test ./go -run 'TestHIPRuntime_LoadModelRunsTinyLoRAAdapterWhenHSACOConfigured_Good|TestHIPRuntime_LoadTinyLoRAAdapterBadValidationKeepsActiveAdapter_Bad|TestNativeContract_(AdapterIdentityClonedAtPublicBoundary_Good|ActiveAdapterClonesNativeFallback_Good|HIPLoadedModelActiveAdapterClonesIdentity_Good)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestHIPRuntime' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestHIP.*LoRA|Test.*LoRAAdapter|Test.*ActiveAdapter|Test.*AdapterLifecycle' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after state wake/sleep label isolation:

```text
go test ./go -run 'TestStateSession_Good_(WakeStateReturnsClonedLabels|SleepStateReturnsClonedLabels|WakeStateReturnsRefs|SleepStateWritesMergedPlaceholderTags)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestStateSession|TestCacheService|TestKVCache' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract_(Close|LoadAdapter|UnloadAdapter|RestoreState|AdapterLifecycle|AdapterIdentity|ActiveAdapter|HIPLoadedModelActiveAdapter)' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after public embedding/rerank result cloning:

```text
go test ./go -run 'TestNativeContract_(EmbeddingResultClonedAtPublicBoundary_Good|RerankResultClonedAtPublicBoundary_Good|PublicWrappersPreferCancelledContext_Ugly)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestHIPRuntime_LoadModelRuns(BERT|Tiny).*Rerank|TestHIPRuntime_LoadModelRuns(BERT|Tiny).*Embed|TestHIPRuntime_LoadedTinyEmbedAndRerank|TestHIPRuntime_LoadedTinyEmbedAndRerankPreflight' -count=1
ok dappco.re/go/rocm

go test ./go -run 'Test.*Embed|Test.*Rerank|Test.*Embedding' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco
PASS
```

Latest focused verification after public classify/batch-generate result cloning:

```text
go test ./go -run 'TestNativeContract_(ClassifyResultsClonedAtPublicBoundary_Good|ClassifyWithoutLogitsDoesNotMutateNativeResult_Bad|BatchGenerateResultsClonedAtPublicBoundary_Good|ClassifyWithLogitsEmitsLogitAndEntropyProbes_Good|ClassifyWithoutLogitsStripsNativeLogits_Bad|BatchGenerateRecordsNativeError_Bad)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest focused verification after state-session identity label cloning:

```text
go test ./go -run 'TestStateSession_Good_(IdentityLabelsCloned|WakeStateReturnsClonedLabels|SleepStateReturnsClonedLabels|ForkStateCreatesIndependentSession|ForkStateRestoresIndependentRuntimeOwnedKVSnapshot)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestStateSession' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest focused verification after capability report identity cloning:

```text
go test ./go -run 'TestNativeContract_(CapabilityReportClonesIdentityMetadata_Good|RocmModelCapabilities_Ugly|RocmModelCapabilitiesUseNativeKernelStatus_Good|RocmBackendUnavailableRuntime_Bad)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest focused verification after cache disk KV snapshot identity hardening:

```text
go test ./go -run 'TestCacheService_(Good_WritesPortableKVSnapshotDiskRefsWithOpaqueStateStore|Good_RestoresPortableKVSnapshotDiskRefOnColdWarm|Bad_RejectsMismatchedPortableKVSnapshotDiskRef|Good_RestoresLegacyRawPortableKVSnapshotDiskRef|Bad_RejectsMismatchedLegacyRawPortableKVSnapshotDiskRef|Good_WritesMetadataDiskRefsForBlockPrefixCache|Good_RestoresMetadataDiskRefOnColdWarm)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestCacheService|TestKVCache|TestStateSession_Good_(SleepStateSerializesRuntimeOwnedKVSnapshot|WakeStateRestoresRuntimeOwnedKVSnapshot|SleepStateSerializesHIPDeviceKVSnapshot|WakeStateRestoresHIPDeviceKVSnapshotAsPackageLocal|RocmModelWakeStateRemirrorsKVSnapshotToHIPDevice)' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after textual stop-sequence generation option parity:

```text
go test ./external/go-inference/go -run 'TestOptions|TestIdentity|ExampleWithStop' -count=1
ok dappco.re/go/inference

go test ./external/go-inference/go/scheduler ./external/go-inference/go/bench ./external/go-inference/go/openai ./external/go-inference/go/anthropic -run 'TestModel_ErrAndHelpers_Good|TestConfigGenerateOptions|TestNormalizeConfig_ClonesSlices_Good|TestOpenAI_DecodeRequest_Good_StopStringAndDefaults|TestResponses_ResponseGenerateOptions_Good|TestAnthropic_GenerateOptions_Good' -count=1
ok dappco.re/go/inference/scheduler
ok dappco.re/go/inference/bench
ok dappco.re/go/inference/openai
ok dappco.re/go/inference/anthropic

go test ./go -run 'TestStateSession_Good_RocmModelCapturesMetadataStateBundle|TestScheduler_Good_StreamsQueuedRequest|TestNativeContract_GeneratePassesStopSequences_Good' -count=1
ok dappco.re/go/rocm

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

go test ./go -run 'Test(StateSession|Scheduler|NativeContract)' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after public ROCm stop-sequence enforcement:

```text
go test ./go -run 'TestNativeContract_(GeneratePassesStopSequences_Good|GenerateEnforcesStopSequencesAcrossChunks_Good|BatchGenerateEnforcesStopSequencesAcrossChunks_Good|BatchGenerateResultsClonedAtPublicBoundary_Good)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestDecodeHelpers_Good_PromptLookupDecodeUsesLookupDraft|ExampleSpeculativeDecode|ExamplePromptLookupDecode|TestNativeContract_(GenerateEnforcesStopSequencesAcrossChunks_Good|BatchGenerateEnforcesStopSequencesAcrossChunks_Good)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'Test(NativeContract|DecodeHelpers)|Example.*Decode' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

git diff --check
PASS
```

Latest verification after Ollama-compatible stop-sequence options:

```text
go test ./external/go-inference/go/ollama -run 'TestOllama_GenerateOptions_Good' -count=1
ok dappco.re/go/inference/ollama

go test ./go -run 'TestCompatHandlers_Good_Ollama(GenerateAppliesStopSequences|ChatAndGenerate)|TestNativeContract_(GenerateEnforcesStopSequencesAcrossChunks_Good|BatchGenerateEnforcesStopSequencesAcrossChunks_Good)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestCompatHandlers_Good_OllamaGenerateAppliesStopSequences|TestOpenAI' -count=1
ok dappco.re/go/rocm

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

go test ./go -run 'Test(CompatHandlers|NativeContract|Scheduler|StateSession)|Example.*Decode' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after ROCm benchmark stop-sequence propagation:

```text
go test ./go -run 'TestNativeContract_Benchmark(PassesStopSequences_Good|WarmupRunsAllPromptsWithoutMeasuredCounters_Good|MeasuresActiveLoRAOverhead_Good)|TestNativeContract_GenerateEnforcesStopSequencesAcrossChunks_Good' -count=1
ok dappco.re/go/rocm

go test ./external/go-inference/go -run 'Test.*Bench|Test.*Dataset|TestCapability' -count=1
ok dappco.re/go/inference

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

go test ./go -run 'Test(NativeContract|CompatHandlers|Scheduler|StateSession)|Example.*Decode' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after eval quality-probe stop-sequence propagation:

```text
go test ./go -run 'TestNativeContract_EvaluateQualityProbes(PassStopSequences_Good|_Good|_Bad_RecordsUnavailableGeneration)|TestNativeContract_BenchmarkPassesStopSequences_Good' -count=1
ok dappco.re/go/rocm

go test ./external/go-inference/go -run 'Test.*Dataset|Test.*Eval|TestCapability' -count=1
ok dappco.re/go/inference

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

go test ./go -run 'Test(NativeContract|CompatHandlers|Scheduler|StateSession)|Example.*Decode' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after tokenizer/template mutable-boundary hardening:

```text
go test ./go -run 'TestNativeContract_TokenizerBoundariesCloneMutableSlices_Good|TestNativeContract_GeneratePassesStopSequences_Good|TestNativeContract_BenchmarkPassesStopSequences_Good|TestNativeContract_EvaluateQualityProbesPassStopSequences_Good' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract' -count=1
ok dappco.re/go/rocm

go test ./external/go-inference/go/... -count=1
ok dappco.re/go/inference/...

go test ./go -run 'Test(NativeContract|CompatHandlers|Scheduler|StateSession)|Example.*Decode' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after block-cache HIP device remirroring for portable KV snapshots:

```text
go test ./go -run 'TestCacheService_(Good_MirrorsWarmKVSnapshotToHIPDevice|Good_RestoresDiskKVSnapshotToHIPDeviceOnColdWarm|Bad_DeviceMirrorFailureKeepsPortableKVBlock|Good_RestoresPortableKVSnapshotDiskRefOnColdWarm|Good_WarmStatsClear)|TestKVCache' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestCacheService_(Good_MirrorsWarmKVSnapshotToHIPDevice|Good_RestoresDiskKVSnapshotToHIPDeviceOnColdWarm|Bad_DeviceMirrorFailureKeepsPortableKVBlock|Good_RocmModelWarmCacheUsesHIPDeviceDriver|Good_RocmModelAdapterChangeClosesMirroredCache|Good_RocmModelCloseClosesMirroredCache|Good_RocmModelAdapterChangeResetsCache)|TestNativeContract_BenchmarkLoRAStateCloseFailureSkipsNativeUnload_Bad' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestCacheService|TestStateSession|TestNativeContract_RocmModel|TestHIPRuntime_LoadModel|TestHIPKernels|TestKVCache' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after cache capability device-remirror reporting:

```text
go test ./go -run 'TestNativeContract_RocmBackendCapabilities_Good|TestNativeContract_RocmModelCapabilities_Ugly|TestNativeContract_RocmModelCapabilitiesUseNativeKernelStatus_Good|TestCacheService_Good_RocmModelWarmCacheUsesHIPDeviceDriver' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after benchmark cache device-remirror telemetry coverage:

```text
go test ./go -run 'TestNativeContract_Benchmark(MirrorsDeviceCacheLabels_Good|EmitsCacheAndMemoryProbeEvents_Good)|TestCacheService_Good_MirrorsWarmKVSnapshotToHIPDevice' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after opt-in cache hardware smoke remirror coverage:

```text
go test ./go -run 'TestHIPHardwareKVCacheSmoke_Good|TestCacheService_Good_MirrorsWarmKVSnapshotToHIPDevice|TestNativeContract_BenchmarkMirrorsDeviceCacheLabels_Good' -count=1 -v
PASS with TestHIPHardwareKVCacheSmoke_Good skipped unless GO_ROCM_RUN_CACHE_TESTS=1

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after direct BlockCacheService close coverage:

```text
go test ./go -run 'TestCacheService_Good_(CloseClosesMirroredKVPages|MirrorsWarmKVSnapshotToHIPDevice|RocmModelCloseClosesMirroredCache)|TestNativeContract_BenchmarkMirrorsDeviceCacheLabels_Good' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after BlockCacheService Close example coverage:

```text
go test ./go -run 'ExampleBlockCacheService_(Close|WarmCache)|TestCacheService_Good_CloseClosesMirroredKVPages' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after mirrored block-cache close-failure coverage:

```text
go test ./go -run 'TestCacheService_(Bad_(ClosePropagatesDeviceFreeFailure|ClearPropagatesDeviceFreeFailure|RocmModelLoadAdapterStopsOnCacheCloseFailure|RocmModelUnloadAdapterStopsOnCacheCloseFailure|RocmModelCloseStopsOnCacheCloseFailure)|Good_(CloseClosesMirroredKVPages|RocmModelAdapterChangeClosesMirroredCache|RocmModelCloseClosesMirroredCache))' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after state wake device-runtime close-failure coverage:

```text
go test ./go -run 'TestStateSession_(Bad_(WakeStateClosePreviousDeviceRuntimeFailureDoesNotInstallSnapshot|RocmModelWakeStateClosePreviousDeviceRuntimeFailureKeepsPreviousState)|Good_(WakeStateClosesPreviousRuntime|RocmModelWakeStateRemirrorsKVSnapshotToHIPDevice|RocmModelWakeStateKeepsPackageLocalKVOnDeviceMirrorFailure))' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after StateSession Close example coverage:

```text
go test ./go -run 'ExampleStateSession_(Close|SleepState_kvSnapshot|WakeState)|TestStateSession_Good_RocmModelCloseClosesStateWithoutNative' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after direct StateSession Close failure coverage:

```text
go test ./go -run 'TestStateSession_Bad_(CloseFailureKeepsRuntime|WakeStateClosePreviousDeviceRuntimeFailureDoesNotInstallSnapshot|RocmModelWakeStateClosePreviousDeviceRuntimeFailureKeepsPreviousState)|ExampleStateSession_Close' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after StateSession SleepState device snapshot failure coverage:

```text
go test ./go -run 'TestStateSession_(Bad_SleepStateDeviceKVSnapshotFailureDoesNotWriteStateRef|Good_SleepStateSerializesHIPDeviceKVSnapshot|Good_WakeStateRestoresHIPDeviceKVSnapshotAsPackageLocal)|TestKVCache_Bad_DeviceMirrorSnapshotRejectsClosedAndCopyFailure' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after StateSession SleepState KV write-failure coverage:

```text
go test ./go -run 'TestStateSession_(Bad_SleepState(RuntimeOwnedKVWriteFailureKeepsRuntime|DeviceKVWriteFailureKeepsRuntime|DeviceKVSnapshotFailureDoesNotWriteStateRef)|Good_SleepStateSerializes(RuntimeOwnedKVSnapshot|HIPDeviceKVSnapshot))' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after StateSession SleepState binary-writer requirement coverage:

```text
go test ./go -run 'TestStateSession_Bad_SleepState(RuntimeOwnedKVRequiresBinaryWriter|DeviceKVRequiresBinaryWriter|RuntimeOwnedKVWriteFailureKeepsRuntime|DeviceKVWriteFailureKeepsRuntime)' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after StateSession SleepState placeholder store failure coverage:

```text
go test ./go -run 'TestStateSession_(Good_SleepStateWritesMergedPlaceholderTags|Bad_SleepState(RequiresStore|PlaceholderRequiresWriter|PlaceholderWriteFailure))' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after loaded-model fork-state device remirroring:

```text
go test ./go -run 'TestStateSession_Good_(ForkStateRestoresIndependentRuntimeOwnedKVSnapshot|RocmModelForkStateRemirrorsKVSnapshotToHIPDevice|RocmModelForkStateKeepsPackageLocalKVOnDeviceMirrorFailure)|TestNativeContract_RocmBackendCapabilities_Good|TestNativeContract_RocmModelCapabilities_Ugly' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after StateSession ForkState error coverage:

```text
go test ./go -run 'TestStateSession_(Bad_ForkState(RejectsNilSession|WrapsWakeFailure)|Good_(ForkStateCreatesIndependentSession|ForkStateRestoresIndependentRuntimeOwnedKVSnapshot|RocmModelForkStateRemirrorsKVSnapshotToHIPDevice|RocmModelForkStateKeepsPackageLocalKVOnDeviceMirrorFailure))' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after loaded-model ForkState example coverage:

```text
go test ./go -run 'Example_rocmModel_ForkState|ExampleStateSession_ForkState|TestStateSession_Good_(RocmModelForkStateRemirrorsKVSnapshotToHIPDevice|RocmModelForkStateKeepsPackageLocalKVOnDeviceMirrorFailure)' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after training reference edge coverage:

```text
go test ./go -run 'TestTrainingReference' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after import-boundary alias coverage:

```text
go test ./go -run 'TestImportBoundary' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after shared-contract import-boundary coverage:

```text
go test ./go -run 'TestImportBoundary' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after probe reference edge coverage:

```text
go test ./go -run 'TestProbeReference' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after embedding/rerank reference edge coverage:

```text
go test ./go -run 'Test(EmbeddingReference|RerankReference)' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after LoRA reference edge coverage:

```text
go test ./go -run 'TestLoRAReference' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after MoE/JANGTQ/codebook reference edge coverage:

```text
go test ./go -run 'Test(MoEReference|JANGTQReference|CodebookReference|ResidualReference)' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after decode reference/helper edge coverage:

```text
go test ./go -run 'TestDecode(Reference|Helpers)' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after projection reference edge coverage:

```text
go test ./go -run 'TestHIPProjectionReference|TestHIPFloat16ToFloat32' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after transformer reference edge coverage:

```text
go test ./go -run 'TestHIPTransformerReference' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after RMSNorm/RoPE launch finite validation:

```text
go test ./go -run 'TestHIPKernels_(RMSNorm|RoPE)LaunchArgs' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after small-decode finite validation:

```text
go test ./go -run 'TestHIPSmallDecode_Bad|TestHIPRuntime_LoadedSmallDecodeRequestFiniteValidation_Bad' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after tiny prefill/decode packet bad-path coverage:

```text
go test ./go -run 'TestHIPKernels_Tiny(Prefill|Decode)LaunchArgs_Bad|TestHIPKernels_TinyOutputWeightValues_Bad' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after loaded tiny-model config shape coverage:

```text
go test ./go -run 'TestHIPRuntime_LoadedTinyLMConfigShapeValidation_Bad' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after loaded tiny packed/codebook output metadata coverage:

```text
go test ./go -run 'TestHIPRuntime_LoadedTiny(JANGTQ|Codebook)OutputValidation_Bad' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after loaded tiny request validation coverage:

```text
go test ./go -run 'TestHIPRuntime_LoadedTiny(RequestValidation|JANGTQOutputValidation|CodebookOutputValidation)_Bad' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after LoRA adapter validator edge coverage:

```text
go test ./go -run 'TestHIPLoRAModel_(TinyAdapterValidation|SmallAdapterValidation|ClassifierAdapterValidation|HelperValidation|RunProjectionRequiresActiveAdapter|LoadedWeightHelpersValidation)_Bad' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after MoE/JANGTQ/codebook launch-packet bad-path coverage:

```text
go test ./go -run 'Test.*MoE|Test.*JANG|Test.*Codebook' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest verification after cache metadata disk-ref identity and live metadata label scrub hardening:

```text
go test ./go -run 'TestCacheService_(Good_MetadataWarmScrubsRuntimeOnlyLabels|Good_MetadataWarmScrubsKVShapeLabels|Good_WarmScrubsDiskRuntimeLabels|Good_WarmAllowsDiskURIWithStore|Good_RestoresMetadataDiskRefOnColdWarm|Bad_RejectsMismatchedMetadataDiskRef|Bad_RejectsMismatchedPortableKVSnapshotDiskRef|Bad_RejectsMismatchedPortableKVSnapshotLabels|Bad_RejectsMismatchedLegacyRawPortableKVSnapshotDiskRef)' -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

cd go && go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

git diff --check
PASS
```

Latest focused verification after training reference non-finite input hardening:

```text
go test ./go -run 'TestHIP(Kernels_Projection|LoRAProjection)' -count=1
ok dappco.re/go/rocm

go test ./go -run 'Test(HIPMoE|HIPJANGTQ|HIPCodebook|MoEReference|JANGTQReference|CodebookReference|ResidualReference|TrainingReference)' -count=1
ok dappco.re/go/rocm

go test ./go -count=1
ok dappco.re/go/rocm

git diff --check
PASS
```

Latest focused verification after deterministic BERT classifier/scorer head selection:

```text
go test ./go -run 'TestHIPRuntime_LoadedSequenceClassifierConfig|TestHIPRuntime_LoadModelRunsBERT(SequenceClassifier|ScoreTensor)|TestHIPRuntime_LoadModelBERTSequenceClassifier' -count=1
ok dappco.re/go/rocm

go test ./go -count=1
ok dappco.re/go/rocm

git diff --check
PASS
```

Latest focused verification after embedding/rerank output readback hardening:

```text
go test ./go -run 'TestHIP(EmbeddingMeanPool|RerankCosine|EmbeddingAndRerank)' -count=1
ok dappco.re/go/rocm
```

Latest focused verification after transformer primitive readback hardening:

```text
go test ./go -run 'TestHIPKernels_(RMSNorm|RoPE|GreedySample|Attention|TinyPrefill|TinyDecode|TransformerPrimitiveReadOutputValidation|TinyReadOutputValidation)' -count=1
ok dappco.re/go/rocm
```

Latest focused verification after loaded small-decode readback hardening:

```text
go test ./go -run 'TestHIP(SmallDecode|Runtime_LoadedSmallDecode|Runtime_LoadModelRunsSmallDecode)' -count=1
ok dappco.re/go/rocm
```

Latest focused verification after loaded host-tensor finite hardening:

```text
go test ./go -run 'TestHIP(Kernels_TinyOutputWeightValues|LoRAModel_LoadedWeightHelpersValidation|Runtime_LoadedSmallDecodeEmbeddingReadFiniteValidation|Runtime_LoadedEmbeddingTableFiniteValidation|Runtime_LoadedSmallDecode|SmallDecode)' -count=1
ok dappco.re/go/rocm
```

Latest focused verification after HIP cross-entropy/distillation/GRPO training launch fixtures:

```text
go test ./go -run 'TestHIPTraining(CrossEntropyLoss|DistillationKLLoss|GRPOAdvantage)|TestTrainingReference(CrossEntropy|DistillationKL|NormalizeAdvantages)|TestHIPKernelSource_ExportsLaunchABI|TestHIPHardwareTransformerKernelSource' -count=1
ok dappco.re/go/rocm
```

Latest package verification after HIP GRPO advantage launch fixtures:

```text
go test ./go -count=1
ok dappco.re/go/rocm

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -run 'TestHIPTraining(CrossEntropyLoss|DistillationKLLoss|GRPOAdvantage)|TestHIPKernelSource_ExportsLaunchABI' -count=1
ok dappco.re/go/rocm

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm

git diff --check
PASS

go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace
```

Latest focused verification after HIP eval cross-entropy loss integration:

```text
go test ./go -run 'TestNativeContract_Evaluate(UsesClassifyLogitsForLoss|UsesNativeCrossEntropyLossKernel|LossKernelErrorDoesNotFailEval|BatchesClassifyLogitLoss|BadLossTargetWithoutLogits|LinkedPrefillWithoutLossTargets)|TestHIPRuntime_LoadModelRunsBERTSequenceClassifierRerankWhenHSACOConfigured' -count=1
ok dappco.re/go/rocm

go test ./go -count=1
ok dappco.re/go/rocm

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -run 'TestNativeContract_Evaluate(UsesNativeCrossEntropyLossKernel|LossKernelErrorDoesNotFailEval)|TestHIPTrainingCrossEntropyLoss' -count=1
ok dappco.re/go/rocm

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

git diff --check
PASS
```

Latest focused verification after native-status-aware planned training fixture labels:

```text
go test ./go -run 'TestNativeContract_Rocm(BackendCapabilities_Good|ModelDoesNotImplementTrainingSurfaces_Ugly|ModelCapabilitiesUseNativeKernelStatus_Good)|Example_trainingCapabilityReport' -count=1
ok dappco.re/go/rocm

go test ./go -count=1
ok dappco.re/go/rocm

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -run 'TestNativeContract_Rocm(BackendCapabilities_Good|ModelDoesNotImplementTrainingSurfaces_Ugly|ModelCapabilitiesUseNativeKernelStatus_Good)' -count=1
ok dappco.re/go/rocm

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

git diff --check
PASS
```

Latest focused verification after loaded-model toy distillation/GRPO hooks:

```text
go test ./go -run 'TestHIPTraining(LoadedModel|Distillation|GRPO|CrossEntropy)' -count=1
ok dappco.re/go/rocm

go test ./go -count=1
ok dappco.re/go/rocm

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -run 'TestHIPTraining(LoadedModel|Distillation|GRPO|CrossEntropy)' -count=1
ok dappco.re/go/rocm

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

git diff --check
PASS
```

Latest focused verification after explicit loss-fixture kernel status fields:

```text
go test ./go -run 'TestHIPKernels_StatusLabels_Good|TestHIPRuntime_LoadModelLinksProjectionKernelWhenHSACOConfigured_Good|TestHIPTrainingLoadedModel|TestNativeContract_Rocm(BackendCapabilities_Good|BackendCapabilitiesUseRuntimeKernelStatus_Good|ModelCapabilitiesUseNativeKernelStatus_Good|ModelDoesNotImplementTrainingSurfaces_Ugly)|TestNativeContract_Evaluate(UsesNativeCrossEntropyLossKernel|LossKernelErrorDoesNotFailEval)' -count=1
ok dappco.re/go/rocm

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -run 'TestHIPKernels_StatusLabels_Good|TestHIPTrainingLoadedModel|TestNativeContract_Rocm(BackendCapabilities_Good|BackendCapabilitiesUseRuntimeKernelStatus_Good|ModelCapabilitiesUseNativeKernelStatus_Good|ModelDoesNotImplementTrainingSurfaces_Ugly)' -count=1
ok dappco.re/go/rocm

go test ./go -count=1
ok dappco.re/go/rocm

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

git diff --check
PASS
```

Latest focused verification after eval loss capability example:

```text
go test ./go -run 'Example_(evaluationLossCapabilityReport|trainingCapabilityReport|metadataOnlyFixtureCapabilities)' -count=1 -v
ok dappco.re/go/rocm

go test ./go -count=1
ok dappco.re/go/rocm

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

git diff --check
PASS
```

Latest focused verification after eval loss fixture labels on runtime reports:

```text
go test ./go -run 'TestNativeContract_(BenchmarkAndEvaluateUseModelSurface_Ugly|EvaluateUsesClassifyLogitsForLoss_Good|EvaluateUsesNativeCrossEntropyLossKernel_Good|EvaluateLinkedPrefillWithoutLossTargetsLabelsNotRequested_Good)|TestHIPRuntime_LoadBERTSequenceClassifierRerankAndEval_Good' -count=1
ok dappco.re/go/rocm

go test ./go -count=1
ok dappco.re/go/rocm

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

git diff --check
PASS
```

Latest focused verification after linked decode/prefill capability scope labels:

```text
go test ./go -run 'TestNativeContract_Rocm(ModelCapabilitiesUseNativeKernelStatus|TinyFixtureCapabilitiesLabelProductionPending|BackendCapabilities|BackendCapabilitiesUseRuntimeKernelStatus).*' -count=1
ok dappco.re/go/rocm
```

Latest package/repo verification after linked decode/prefill capability scope labels:

```text
go test ./go -count=1
ok dappco.re/go/rocm

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

git diff --check
PASS
```

Latest focused verification after benchmark/eval report fixture-scope labels:

```text
go test ./go -run 'TestNativeContract_BenchmarkAndEvaluateTinyFixtureLabelsProductionPending_Good|TestNativeContract_BenchmarkAndEvaluateUseModelSurface_Ugly|TestNativeContract_EvaluateLinkedPrefillWithoutLossTargetsLabelsNotRequested_Good' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract_BenchmarkAndEvaluateTinyFixtureLabelsProductionPending_Good|TestNativeContract_EvaluateQualityProbes_Good|TestNativeContract_EvaluateLinkedPrefillWithoutLossTargetsLabelsNotRequested_Good' -count=1
ok dappco.re/go/rocm

go test ./go -run 'TestNativeContract_(Benchmark|Evaluate)' -count=1
ok dappco.re/go/rocm
```

Latest package/repo verification after benchmark/eval report fixture-scope labels:

```text
go test ./go -count=1
ok dappco.re/go/rocm

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

git diff --check
PASS
```

Latest package/repo verification after eval quality-probe decode fixture-scope labels:

```text
go test ./go -count=1
ok dappco.re/go/rocm

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

git diff --check
PASS
```

Latest focused verification after LoRA capability fixture-scope labels:

```text
go test ./go -run 'TestNativeContract_RocmModelCapabilitiesUse(LoRAKernelStatus|NativeKernelStatus)|TestNativeContract_RocmBackendCapabilities_Good' -count=1
ok dappco.re/go/rocm
```

Latest package/repo verification after LoRA capability fixture-scope labels:

```text
go test ./go -count=1
ok dappco.re/go/rocm

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm

go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace

git diff --check
PASS
```

Latest focused verification after Gemma4-E2B safetensors HIP load:

```text
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 rocminfo | grep -E 'Agent |Name:|Marketing Name:|Uuid:|gfx'
Agent 2
  Name:                    gfx1100
  Uuid:                    GPU-880ed6479d653a85
  Marketing Name:          AMD Radeon RX 7800 XT

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 go test ./go -run TestHIPHardwareAvailabilitySmoke_Good -count=1 -v
--- PASS: TestHIPHardwareAvailabilitySmoke_Good (0.03s)
PASS
ok dappco.re/go/rocm 0.027s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit go test ./go -run TestNativeModelPackSmokeGemma4E2B_Good -count=1 -v
--- PASS: TestNativeModelPackSmokeGemma4E2B_Good (0.86s)
PASS
ok dappco.re/go/rocm 0.874s

go test ./go -run 'TestNativeContract_LoadModelSafetensorsGemma4UsesNativeRuntime_Good|TestNativeContract_ModelPackInspectorGemma4NestedTextConfig_Good|TestNativeContract_PlanModelFit_MemoryClassesAndCacheModes_Good|TestHIPRuntime_Validate_Good(GGUFTokenEmbeddingAlias|Gemma4TiedSafetensorsEmbedding)' -count=1 -v
PASS
ok dappco.re/go/rocm 0.011s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (6.06s)
PASS
ok dappco.re/go/rocm 6.082s

go test ./go -count=1
ok dappco.re/go/rocm 0.134s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.104s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.612s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.196s

git diff --check
PASS
```

Latest focused verification after sharded safetensors HIP load support:

```text
go test ./go -run 'TestNativeContract_LoadModelSafetensors(Gemma4UsesNativeRuntime_Good|ShardedPackUsesNativeRuntime_Good)|TestHIPRuntime_(Validate_GoodGemma4TiedSafetensorsEmbedding|LoadModelCopiesShardedSafetensorsSources_Good)' -count=1 -v
--- PASS: TestHIPRuntime_Validate_GoodGemma4TiedSafetensorsEmbedding (0.00s)
--- PASS: TestHIPRuntime_LoadModelCopiesShardedSafetensorsSources_Good (0.00s)
--- PASS: TestNativeContract_LoadModelSafetensorsGemma4UsesNativeRuntime_Good (0.00s)
--- PASS: TestNativeContract_LoadModelSafetensorsShardedPackUsesNativeRuntime_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.009s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (16.44s)
PASS
ok dappco.re/go/rocm 16.456s

go test ./go -count=1
ok dappco.re/go/rocm 0.137s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.112s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.012s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.618s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.187s

git diff --check
PASS
```

Latest focused verification after BF16 Gemma4-E2B model-pack smoke:

```text
go test ./go -run 'TestNativeContract_ModelPackInspectorGemma4(BF16DType|NestedTextConfig)_Good|TestNativeContract_LoadModelSafetensorsShardedPackUsesNativeRuntime_Good' -count=1 -v
--- PASS: TestNativeContract_LoadModelSafetensorsShardedPackUsesNativeRuntime_Good (0.00s)
--- PASS: TestNativeContract_ModelPackInspectorGemma4NestedTextConfig_Good (0.00s)
--- PASS: TestNativeContract_ModelPackInspectorGemma4BF16DType_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.009s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B go test ./go -run TestNativeModelPackSmokeGemma4E2B_Good -count=1 -v
--- PASS: TestNativeModelPackSmokeGemma4E2B_Good (0.24s)
PASS
ok dappco.re/go/rocm 0.249s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (16.41s)
PASS
ok dappco.re/go/rocm 16.426s

go test ./go -count=1
ok dappco.re/go/rocm 0.141s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.104s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.614s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.187s
```

Latest focused verification after safetensors index validation:

```text
go test ./go -run 'TestNativeContract_ModelPackInspector(Gemma4BF16DType_Good|SafetensorsIndexMissingShard_Bad|AggregatesSafetensorsShardSummaries_Good)' -count=1 -v
--- PASS: TestNativeContract_ModelPackInspectorGemma4BF16DType_Good (0.00s)
--- PASS: TestNativeContract_ModelPackInspectorSafetensorsIndexMissingShard_Bad (0.00s)
--- PASS: TestNativeContract_ModelPackInspectorAggregatesSafetensorsShardSummaries_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B go test ./go -run TestNativeModelPackSmokeGemma4E2B_Good -count=1 -v
--- PASS: TestNativeModelPackSmokeGemma4E2B_Good (0.26s)
PASS
ok dappco.re/go/rocm 0.271s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (16.65s)
PASS
ok dappco.re/go/rocm 16.668s

go test ./go -count=1
ok dappco.re/go/rocm 0.144s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.108s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.012s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.618s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.185s

git diff --check
PASS
```

Latest focused verification after known weight bytes in memory fit:

```text
go test ./go -run 'TestNativeContract_PlanModelFit_UsesKnownWeightBytes_Bad|TestNativeContract_ModelPackInspector(Gemma4BF16DType_Good|SafetensorsIndexMissingShard_Bad)|TestNativeModelPackSmokeGemma4E2B_Good' -count=1 -v
--- SKIP: TestNativeModelPackSmokeGemma4E2B_Good (0.00s)
--- PASS: TestNativeContract_PlanModelFit_UsesKnownWeightBytes_Bad (0.00s)
--- PASS: TestNativeContract_ModelPackInspectorGemma4BF16DType_Good (0.00s)
--- PASS: TestNativeContract_ModelPackInspectorSafetensorsIndexMissingShard_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.009s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B go test ./go -run TestNativeModelPackSmokeGemma4E2B_Good -count=1 -v
--- PASS: TestNativeModelPackSmokeGemma4E2B_Good (0.25s)
PASS
ok dappco.re/go/rocm 0.263s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (16.19s)
PASS
ok dappco.re/go/rocm 16.204s

go test ./go -count=1
ok dappco.re/go/rocm 0.144s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.108s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.012s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.608s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.197s

git diff --check
PASS
```

Latest focused verification after Gemma4 sliding-attention memory fit:

```text
go test ./go -run 'TestNativeContract_PlanModelFit_(UsesKnownWeightBytes_Bad|Gemma4SlidingAttentionWeightBytes_Good)|TestNativeContract_ModelPackInspectorGemma4(BF16DType|NestedTextConfig)_Good' -count=1 -v
--- PASS: TestNativeContract_PlanModelFit_UsesKnownWeightBytes_Bad (0.00s)
--- PASS: TestNativeContract_PlanModelFit_Gemma4SlidingAttentionWeightBytes_Good (0.00s)
--- PASS: TestNativeContract_ModelPackInspectorGemma4NestedTextConfig_Good (0.00s)
--- PASS: TestNativeContract_ModelPackInspectorGemma4BF16DType_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.008s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B go test ./go -run TestNativeModelPackSmokeGemma4E2B_Good -count=1 -v
--- PASS: TestNativeModelPackSmokeGemma4E2B_Good (0.26s)
PASS
ok dappco.re/go/rocm 0.273s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (16.17s)
PASS
ok dappco.re/go/rocm 16.184s

go test ./go -count=1
ok dappco.re/go/rocm 0.137s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.009s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.618s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.168s

git diff --check
PASS
```

Latest RX 7800 XT hardware gate verification after making the availability smoke HSACO-aware:

```text
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'TestHIP|TestNative' -count=1 -v
--- PASS: TestHIPHardwareAvailabilitySmoke_Good (0.03s)
--- PASS: TestHIPHardwareProjectionKernelSource_Good (0.02s)
--- PASS: TestHIPHardwareEmbeddingKernelSource_Good (0.00s)
--- PASS: TestHIPHardwareTransformerKernelSource_Good (0.08s)
--- PASS: TestHIPHardwarePrefillDecodeKernelSource_Good (0.01s)
--- PASS: TestNativeContract_LoadModelSafetensorsGemma4UsesNativeRuntime_Good (0.00s)
--- PASS: TestNativeContract_LoadModelSafetensorsShardedPackUsesNativeRuntime_Good (0.00s)
--- PASS: TestNativeContract_PlanModelFit_Gemma4SlidingAttentionWeightBytes_Good (0.00s)
--- PASS: TestNativeContract_ModelPackInspectorGemma4NestedTextConfig_Good (0.00s)
--- PASS: TestNativeContract_ModelPackInspectorGemma4BF16DType_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.221s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'Test.*Smoke|Test.*Generate|Test.*Decode' -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (16.13s)
--- PASS: TestNativeModelPackSmokeGemma4E2B_Good (0.21s)
--- PASS: TestHIPRuntime_LoadModelRunsTinyPrefillDecodeWhenHSACOConfigured_Good (0.00s)
--- PASS: TestHIPSmallDecode_Good_QwenGemmaSmoke (0.00s)
--- PASS: TestHIPRuntime_LoadModelRunsSmallDecodeSmokeWhenHSACOConfigured_Good (0.00s)
--- PASS: TestHIPRuntime_LoadModelRunsSmallDecodeLoRAAdapterWhenHSACOConfigured_Good (0.00s)
PASS
ok dappco.re/go/rocm 16.386s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_CACHE_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'Test.*KV|Test.*Cache' -count=1 -v
--- PASS: TestHIPHardwareKVCacheSmoke_Good (0.04s)
PASS
ok dappco.re/go/rocm 0.067s

go test ./go -count=1
ok dappco.re/go/rocm 0.133s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.615s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.170s

git diff --check
PASS
```

Latest focused verification after Gemma4 GQA/head-geometry metadata labels:

```text
go test ./go -run 'TestNativeContract_ModelPackInspectorGemma4NestedTextConfig_Good|TestNativeContract_PlanModelFit_Gemma4SlidingAttentionWeightBytes_Good' -count=1 -v
--- PASS: TestNativeContract_PlanModelFit_Gemma4SlidingAttentionWeightBytes_Good (0.00s)
--- PASS: TestNativeContract_ModelPackInspectorGemma4NestedTextConfig_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.006s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeModelPackSmokeGemma4E2B_Good -count=1 -v
--- PASS: TestNativeModelPackSmokeGemma4E2B_Good (0.22s)
PASS
ok dappco.re/go/rocm 0.232s

go test ./go -count=1
ok dappco.re/go/rocm 0.133s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.610s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.168s

git diff --check
PASS
```

Latest focused verification after adding BF16 projection support and a device-resident Gemma4 tensor smoke:

```text
go test ./go -run 'TestHIPKernels_Projection|TestHIPProjectionReference|TestHIPFloat16|TestHIPBFloat16' -count=1 -v
PASS
ok dappco.re/go/rocm 0.006s

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestHIPHardwareProjectionKernelSource_Good -count=1 -v
--- PASS: TestHIPHardwareProjectionKernelSource_Good (0.05s)
PASS
ok dappco.re/go/rocm 0.059s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (16.27s)
PASS
ok dappco.re/go/rocm 16.287s
```

Latest focused verification after adding f32/BF16 embedding lookup and wiring the Gemma4 smoke to the loaded device-resident `embed_tokens.weight` table:

```text
go test ./go -run 'TestHIPEmbedding|TestHIPKernelSource_ExportsLaunchABI_Good|TestHIPBFloat16' -count=1 -v
PASS
ok dappco.re/go/rocm 0.006s

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestHIPHardwareEmbeddingKernelSource_Good -count=1 -v
--- PASS: TestHIPHardwareEmbeddingKernelSource_Good (0.04s)
PASS
ok dappco.re/go/rocm 0.056s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (16.54s)
PASS
ok dappco.re/go/rocm 16.555s
```

Latest focused verification after adding BF16 RMSNorm weights and wiring the Gemma4 smoke to loaded device-resident `input_layernorm.weight`:

```text
go test ./go -run 'TestHIPKernels_RMSNorm|TestHIPKernelSource_ExportsLaunchABI_Good|TestHIPTransformerReferenceRMSNorm' -count=1 -v
PASS
ok dappco.re/go/rocm 0.006s

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestHIPHardwareTransformerKernelSource_Good -count=1 -v
--- PASS: TestHIPHardwareTransformerKernelSource_Good (0.12s)
PASS
ok dappco.re/go/rocm 0.136s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (17.02s)
PASS
ok dappco.re/go/rocm 17.036s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 rocminfo | rg -n "Name:|Uuid:|Marketing Name"
87:  Name:                    gfx1100
88:  Uuid:                    GPU-880ed6479d653a85
89:  Marketing Name:          AMD Radeon RX 7800 XT
```

Latest broad verification after the BF16 RMSNorm Gemma4 smoke:

```text
go test ./go -count=1
ok dappco.re/go/rocm 0.140s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.104s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.614s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.184s

(cd go && go test ./... -count=1)
ok dappco.re/go/rocm 0.137s
ok dappco.re/go/rocm/internal/gguf 0.002s

(cd go && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1)
ok dappco.re/go/rocm 0.106s
ok dappco.re/go/rocm/internal/gguf 0.002s

(cd go && go test -tags rocm_legacy_server ./... -count=1)
ok dappco.re/go/rocm 0.010s
ok dappco.re/go/rocm/internal/gguf 0.002s
ok dappco.re/go/rocm/internal/llamacpp 0.003s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'TestHIP|TestNative' -count=1 -v
PASS
ok dappco.re/go/rocm 0.227s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'Test.*Smoke|Test.*Generate|Test.*Decode' -count=1 -v
PASS
ok dappco.re/go/rocm 17.339s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_CACHE_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'Test.*KV|Test.*Cache' -count=1 -v
PASS
ok dappco.re/go/rocm 0.069s

git diff --check
PASS
```

Latest focused verification after widening the Gemma4 smoke to layer-0 BF16 attention projections:

```text
go test ./go -run 'TestHIPHardwareTransformerKernelSource_Good|TestHIPKernels_Projection|TestHIPKernels_RMSNorm' -count=1
ok dappco.re/go/rocm 0.006s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (16.60s)
PASS
ok dappco.re/go/rocm 16.614s

go test ./go -count=1
ok dappco.re/go/rocm 0.139s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.105s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s
```

Latest focused verification after extending the BF16 Gemma4 smoke through q/k RoPE, 8-head one-token GQA attention, and `o_proj` from the attention concat, plus confirming the faster q4 development loop:

```text
go test ./go -run 'TestHIPKernels_Attention|TestHIPTransformerReferenceAttention|TestHIPHardwareTransformerKernelSource_Good' -count=1
ok dappco.re/go/rocm 0.007s

go test ./go -run 'TestHIPKernels_RoPE|TestHIPKernels_Attention|TestHIPTransformerReference(RoPE|Attention)' -count=1
ok dappco.re/go/rocm 0.006s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (16.29s)
PASS
ok dappco.re/go/rocm 16.310s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeModelPackSmokeGemma4E2B_Good -count=1 -v
--- PASS: TestNativeModelPackSmokeGemma4E2B_Good (0.21s)
PASS
ok dappco.re/go/rocm 0.222s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (6.02s)
PASS
ok dappco.re/go/rocm 6.039s
```

BF16 remains the live correctness anchor for raw safetensors tensor-row comparisons. The local q4 pack at `/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit` is suitable for faster metadata/load/decode-status iteration that does not need BF16 row-level checks.

Latest focused verification after adding `rocm_vector_add` and `rocm_swiglu`, then extending the BF16 Gemma4 smoke through layer-0 post-attention RMSNorm and MLP:

```text
go test ./go -run 'TestHIPKernels_(VectorAdd|SwiGLU|Attention|RMSNorm)|TestHIPKernelSource_ExportsLaunchABI_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 0.006s

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestHIPHardwareTransformerKernelSource_Good -count=1 -v
--- PASS: TestHIPHardwareTransformerKernelSource_Good (0.13s)
PASS
ok dappco.re/go/rocm 0.142s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (16.45s)
PASS
ok dappco.re/go/rocm 16.467s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (6.32s)
PASS
ok dappco.re/go/rocm 6.334s

go test ./go -count=1
ok dappco.re/go/rocm 0.142s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.115s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s
```

Latest focused verification after adding the MLX affine q4 projection primitive (`rocm_mlx_q4_projection`) and wiring the q4 Gemma4 smoke to launch it against the loaded packed `layers.0.self_attn.q_proj` tensor:

```text
go test ./go -run 'TestHIPProjectionReferenceMLXQ4|TestHIPKernels_MLXQ4Projection|TestHIPKernelSource_ExportsLaunchABI_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 0.006s

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestHIPHardwareProjectionKernelSource_Good -count=1 -v
--- PASS: TestHIPHardwareProjectionKernelSource_Good (0.05s)
PASS
ok dappco.re/go/rocm 0.057s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (6.11s)
PASS
ok dappco.re/go/rocm 6.131s
```

Latest focused verification after extending `rocm_embedding_lookup` to support MLX affine q4 embedding tables and wiring the q4 Gemma4 smoke to launch it against the loaded packed `embed_tokens` tensors:

```text
go test ./go -run 'TestHIPEmbeddingLookupLaunch|TestHIPTransformerReferenceEmbeddingLookup|TestHIPKernelSource_ExportsLaunchABI_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 0.007s

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestHIPHardwareEmbeddingKernelSource_Good -count=1 -v
--- PASS: TestHIPHardwareEmbeddingKernelSource_Good (0.04s)
PASS
ok dappco.re/go/rocm 0.054s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (7.21s)
PASS
ok dappco.re/go/rocm 7.221s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (20.96s)
PASS
ok dappco.re/go/rocm 20.982s

go test ./go -count=1
ok dappco.re/go/rocm 0.140s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.104s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'Test.*Smoke|Test.*Generate|Test.*Decode' -count=1 -v
PASS
ok dappco.re/go/rocm 7.143s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.618s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.187s
```

Latest focused verification after extending the q4 Gemma4 smoke from isolated embedding/q_proj checks to a full layer-0 primitive chain using loaded packed q4 tensors, then final RMSNorm, tied q4 LM-head projection, and greedy sampling:

```text
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (6.71s)
PASS
ok dappco.re/go/rocm 6.728s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (16.60s)
PASS
ok dappco.re/go/rocm 16.617s

go test ./go -count=1
ok dappco.re/go/rocm 0.140s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'Test.*Smoke|Test.*Generate|Test.*Decode' -count=1 -v
PASS
ok dappco.re/go/rocm 7.346s

git diff --check
```

The q4 smoke now launches `rocm_embedding_lookup` against packed `embed_tokens`, BF16 RMSNorm against q4-pack norm tensors, `rocm_mlx_q4_projection` for layer-0 q/k/v/o and MLP gate/up/down packed tensors, RoPE, one-token GQA attention, residual adds, SwiGLU, final RMSNorm, tied q4 `embed_tokens` projection to vocab logits, and `rocm_greedy_sample` over those logits. CPU comparisons cover selected packed rows for each q4 projection plus CPU references for the non-projection primitives and greedy result.

Latest focused verification after aligning the Gemma4 layer-0 smoke with the `go-mlx` q/k norm order by inserting HIP RMSNorm launches for `self_attn.q_norm.weight` and `self_attn.k_norm.weight` before RoPE:

```text
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (6.70s)
PASS
ok dappco.re/go/rocm 6.716s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (19.93s)
PASS
ok dappco.re/go/rocm 19.950s
```

Latest focused verification after adding the RMSNorm launch flag for Gemma-style `(1 + weight)` norms and correcting the layer residual order to match `go-mlx`: post-attention norm is applied to the attention projection before the attention residual add, then pre-feedforward norm feeds MLP, then post-feedforward norm is applied before the final residual add.

```text
go test ./go -run 'TestHIPKernels_RMSNorm|TestHIPKernelSource_ExportsLaunchABI_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 0.005s

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (5.99s)
PASS
ok dappco.re/go/rocm 6.006s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (16.25s)
PASS
ok dappco.re/go/rocm 16.267s

go test ./go -count=1
ok dappco.re/go/rocm 0.138s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'Test.*Smoke|Test.*Generate|Test.*Decode' -count=1 -v
PASS
ok dappco.re/go/rocm 6.263s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.672s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.187s
```

Latest focused verification after adding `rocm_vector_scale` and using it to match `go-mlx` Gemma4 embedding scaling (`sqrt(hidden_size)`) before the first input RMSNorm:

```text
go test ./go -run 'TestHIPKernels_VectorScale|TestHIPKernelSource_ExportsLaunchABI_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 0.005s

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestHIPHardwareTransformerKernelSource_Good -count=1 -v
--- PASS: TestHIPHardwareTransformerKernelSource_Good (0.14s)
PASS
ok dappco.re/go/rocm 0.152s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (62.67s)
PASS
ok dappco.re/go/rocm 62.682s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (103.35s)
PASS
ok dappco.re/go/rocm 103.361s

go test ./go -count=1
ok dappco.re/go/rocm 0.140s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.105s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'Test.*Smoke|Test.*Generate|Test.*Decode' -count=1 -v
PASS
ok dappco.re/go/rocm 6.602s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.612s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.182s
```

The BF16 and q4 Gemma4 smokes now run the embedding lookup through `rocm_vector_scale` before Gemma-style `(1 + weight)` RMSNorm, matching the `go-mlx` embedding scale plus norm/residual order. The q4 loop remains the fastest full layer-0 packed-tensor development path through tied LM-head logits and greedy sampling; production prefill/decode/generate remains explicitly not linked.

Latest focused verification after moving device-resident embedding lookup and vector-add/vector-scale/SwiGLU launches behind reusable package runner helpers instead of bespoke Gemma4 hardware-test packet assembly:

```text
go test ./go -run 'TestHIPEmbeddingLookupLaunch_Good|TestHIPEmbeddingAndRerankLaunch_Bad|TestHIPKernels_VectorAddLaunchArgs_Good|TestHIPKernels_VectorScaleLaunchArgs_Good|TestHIPKernels_SwiGLULaunchArgs_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 0.010s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (6.53s)
PASS
ok dappco.re/go/rocm 6.542s

go test ./go -count=1
ok dappco.re/go/rocm 0.142s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.105s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check
```

The new reusable helpers are `hipRunEmbeddingLookupKernelWithDeviceTable`, `hipRunVectorAddKernel`, `hipRunVectorScaleKernel`, and `hipRunSwiGLUKernel`. They are now fake-driver tested and used by the loaded Gemma4 q4 smoke, which keeps the layer-0 packed-tensor loop aligned with future non-test decode code while production prefill/decode remains not linked.

Latest focused verification after adding reusable device-resident RMSNorm launch support for BF16 weights plus Gemma's add-unit-weight flag:

```text
go test ./go -run 'TestHIPKernels_RMSNormLaunchArgs' -count=1 -v
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (6.15s)
PASS
ok dappco.re/go/rocm 6.169s

go test ./go -count=1
ok dappco.re/go/rocm 0.140s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.105s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check
```

`hipRunRMSNormKernelWithDeviceWeightConfig` now covers the loaded Gemma4 BF16 norm tensors with `hipRMSNormWeightEncodingBF16` and `hipRMSNormLaunchFlagAddUnitWeight`, replacing another bespoke hardware-test packet path. This is another reusable primitive needed by a future package-level Gemma4 decode composition.

Latest focused verification after switching the loaded Gemma4 RoPE, one-token GQA attention, and greedy-sampling smoke steps onto reusable package runner helpers:

```text
go test ./go -run 'TestHIPKernels_RoPELaunchArgs_Good|TestHIPKernels_GreedySampleLaunchArgs_Good|TestHIPKernels_AttentionLaunchArgs_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 0.007s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (6.10s)
PASS
ok dappco.re/go/rocm 6.119s

go test ./go -count=1
ok dappco.re/go/rocm 0.142s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.104s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check
```

The loaded Gemma4 q4 smoke now reaches RoPE, attention, and greedy sampling through `hipRunRoPEKernel`, `hipRunAttentionKernel`, and `hipRunGreedyKernel`, reducing remaining test-only launch assembly in the real-model primitive chain.

Latest focused verification after extracting the MLX q4 projection device-weight call into a config-based package runner:

```text
go test ./go -run 'TestHIPKernels_MLXQ4ProjectionLaunchArgs' -count=1 -v
=== RUN   TestHIPKernels_MLXQ4ProjectionLaunchArgs_Good
--- PASS: TestHIPKernels_MLXQ4ProjectionLaunchArgs_Good (0.00s)
=== RUN   TestHIPKernels_MLXQ4ProjectionLaunchArgs_Bad
--- PASS: TestHIPKernels_MLXQ4ProjectionLaunchArgs_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (6.13s)
PASS
ok dappco.re/go/rocm 6.150s

go test ./go -count=1
ok dappco.re/go/rocm 0.134s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.622s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.169s
```

`hipRunMLXQ4ProjectionKernelWithDeviceWeightConfig` now owns the reusable device-resident packed-weight/scales/biases path for loaded Gemma4 q4 projections. The loaded q4 smoke uses the config runner for q/k/v/o, MLP gate/up/down, and the tied LM head, while the old positional helper remains as a compatibility wrapper.

Latest focused verification after moving the Gemma4 q4 layer-0 primitive chain into package-level runtime plumbing:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0|TestHIPKernels_MLXQ4ProjectionLaunchArgs' -count=1 -v
=== RUN   TestHIPKernels_MLXQ4ProjectionLaunchArgs_Good
--- PASS: TestHIPKernels_MLXQ4ProjectionLaunchArgs_Good (0.00s)
=== RUN   TestHIPKernels_MLXQ4ProjectionLaunchArgs_Bad
--- PASS: TestHIPKernels_MLXQ4ProjectionLaunchArgs_Bad (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.005s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (6.10s)
PASS
ok dappco.re/go/rocm 6.121s

go test ./go -count=1
ok dappco.re/go/rocm 0.136s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.104s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.620s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.168s
```

`hipRunGemma4Q4Layer0` now loads a q4 token embedding through `rocm_embedding_lookup`, applies Gemma embedding scale, BF16 add-unit RMSNorms, q4 q/k/v/o and MLP projections, q/k norm, RoPE, one-token GQA attention, residual adds, SwiGLU, final norm, tied q4 LM-head projection, and greedy sampling from package code. The hardware smoke now calls that package runner against the loaded Gemma4-E2B q4 tensors while preserving the explicit `production_decode=not_linked` status.

Latest focused verification after splitting the Gemma4 q4 decoder-layer body from token embedding/LM-head work and adding a multi-layer single-token forward wrapper:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (6.63s)
PASS
ok dappco.re/go/rocm 6.641s

go test ./go -count=1
ok dappco.re/go/rocm 0.139s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.619s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.180s
```

`loadedGemma4Q4LayerConfig(layer)` now resolves q4 projection and BF16 norm tensors by layer index, `hipRunGemma4Q4DecoderLayer` runs the reusable per-layer body from an input hidden vector to the next hidden vector, and `hipRunGemma4Q4SingleTokenForward` loops one or more layer configs before final norm, tied q4 LM-head projection, and greedy sampling. The loaded Gemma4-E2B q4 hardware smoke uses this forward wrapper for one layer so the real tensor path now exercises the same sequential-forward scaffolding without claiming production decode support.

Follow-up verification after checking the full loaded Gemma4-E2B q4 layer config set:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (5.96s)
PASS
ok dappco.re/go/rocm 5.974s

go test ./go -count=1
ok dappco.re/go/rocm 0.141s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.628s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.184s
```

The full-layer config check exposed and now allows Gemma4's real per-layer geometry changes: full-attention layers use 512-dim KV heads instead of the 256-dim sliding-attention heads, and later layers use double-wide MLP projections. The cross-layer forward validator now requires only the invariant hidden/vocab/q4-group dimensions while allowing attention and intermediate width to vary by layer.

Latest focused verification after executing selected variable-geometry Gemma4 q4 layers:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (7.14s)
PASS
ok dappco.re/go/rocm 7.158s

go test ./go -count=1
ok dappco.re/go/rocm 0.144s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.629s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.181s
```

The fake-driver test now runs a second layer with different attention width and double-wide MLP shape. The RX 7800 XT q4 model smoke validates all 35 layer configs, then executes layer 0 through logits, plus layer 4's full-attention 512-dim KV geometry and layer 15's 12288-wide MLP decoder-layer body against loaded device-resident tensors.

Latest focused verification after adding per-layer Gemma4 q4 RoPE-base defaults:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (64.03s)
PASS
ok dappco.re/go/rocm 64.047s

go test ./go -count=1
ok dappco.re/go/rocm 0.143s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.105s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.616s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.181s
```

`hipGemma4Q4Layer0Config` now carries a layer RoPE base. Sliding layers default to `10000`; the full-attention 512-dim KV layers default to `1000000`, matching Gemma4-E2B metadata. Decoder-layer requests may override the base, but a zero request base now uses the loaded layer default. The selected real layer-4 smoke therefore executes with the full-attention base instead of the sliding-layer base. This still does not implement Gemma4 full-attention partial rotary semantics; production decode remains not linked.

Latest focused verification after adding Gemma4 q4 partial RoPE support:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (8.15s)
PASS
ok dappco.re/go/rocm 8.166s

go test ./go -count=1
ok dappco.re/go/rocm 0.141s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.613s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.182s
```

`hipGemma4Q4Layer0Config` now also carries `RoPERotaryDim`. Sliding layers rotate the full head; full-attention 512-dim heads rotate only the first 128 dims and leave the remaining head suffix unchanged. The fake-driver test directly checks that partial-RoPE suffix preservation, and the RX 7800 XT smoke now executes selected layer 4 with the full-attention base plus 128-dim partial RoPE.

### Completion Audit Snapshot

Objective restated: continue `GOAL.md` toward ROCm sibling/runtime parity with `go-mlx`, using Gemma4-E2B as the live model evidence where model hardware work is needed.

Prompt-to-artifact checklist:

- `GOAL.md` Phase 0 baseline: `AGENTS.md`, `README.md`, `docs/architecture.md`, `external/go-inference/go/capability.go`, and `external/go-inference/go/contracts.go` are present/read; `git status --short` remains dirty with intentional tracked/untracked work; non-hardware gates are recorded above.
- Shared contracts: `external/go-inference/go` has expanded capabilities, optional contracts, parser/cache/scheduler/bench/eval/openai/anthropic/ollama/decode/quant/state packages; `(cd external/go-inference/go && go test ./... -count=1)` passes.
- Capability honesty/completeness: `TestNativeContract_RocmBackendCapabilities_Good`, `TestNativeContract_RocmModelCapabilities_*`, and `TestImportBoundary_*` pass; production prefill/decode are explicitly `not_linked` while fixture kernels report specific linked/planned statuses.
- Model-pack inspection: `go/model_pack.go` covers GGUF/safetensors/config/tokenizer/JANG/codebook/architecture aliases/quantization/memory labels; Gemma4-E2B nested `text_config`, tied embeddings, BF16, sharded safetensors, safetensors index, known weight bytes, and sliding-attention metadata have focused tests and RX 7800 XT smokes.
- Scheduler/cancel/cache/parser/state/wire APIs: package files and examples exist for scheduler, cache, parser registry, state session, OpenAI/Anthropic/Ollama compat, decode helpers, and native capability examples; `go test ./go -count=1` and focused gate output above pass.
- HIP stepping stones: tensor validation, device copy, launch packet ABI, projection/JANGTQ/codebook/LoRA/embedding/rerank/transformer primitive/tiny prefill-decode/loss fixture tests pass; the broad RX 7800 XT `GO_ROCM_RUN_HIP_TESTS=1` gate passes with `GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco`.
- Gemma4-E2B hardware evidence: `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85` pins the real RX 7800 XT; `rocminfo` reports that UUID as `AMD Radeon RX 7800 XT`; BF16 Gemma4-E2B model-pack smoke and decode-status smoke pass against `/data/lem/models/gemma4/LEM-Gemma4-E2B`, and the decode-status smoke launches `rocm_embedding_lookup` against the loaded device-resident BF16 `embed_tokens.weight` tensor, `rocm_vector_scale` for Gemma's `sqrt(hidden_size)` embedding scale, Gemma-style `(1 + weight)` `rocm_rms_norm` against loaded device-resident BF16 layer-0 `input_layernorm.weight`, `rocm_projection` against loaded device-resident BF16 layer-0 `q_proj`, `k_proj`, and `v_proj`, HIP RMSNorm against BF16 `self_attn.q_norm.weight`/`k_norm.weight`, `rocm_rope` over all 8 real q heads plus the KV head, `rocm_attention` over each one-token GQA head, `rocm_projection` against loaded device-resident BF16 layer-0 `o_proj` using the 2048-wide attention concat, Gemma-style `post_attention_layernorm` before the attention residual add, `pre_feedforward_layernorm` before MLP, BF16 `mlp.gate_proj`/`mlp.up_proj`, `rocm_swiglu`, BF16 `mlp.down_proj`, Gemma-style `post_feedforward_layernorm`, and final residual add, with CPU reference comparisons. The local q4 Gemma4-E2B pack now also launches `rocm_embedding_lookup` against loaded packed U32/BF16-scale/BF16-bias `embed_tokens`, `rocm_vector_scale` for the same embedding scale, Gemma-style BF16 RMSNorm against q4-pack hidden/q-k/feedforward norm tensors, `rocm_mlx_q4_projection` against loaded packed q/k/v/o and MLP gate/up/down tensors plus the tied q4 LM head, `rocm_rope`, `rocm_attention`, `rocm_vector_add`, `rocm_swiglu`, and `rocm_greedy_sample`, so the faster q4 loop covers the full layer-0 primitive chain through vocab logits and a greedy token from real packed tensors as well as load/decode-status iteration.
- Cache/KV hardware evidence: the RFC `GO_ROCM_RUN_CACHE_TESTS=1` gate now passes, including `TestHIPHardwareKVCacheSmoke_Good`.
- Build gates: `go test ./go`, Linux no-cgo, legacy server tag, root workspace gates, `git diff --check`, external contract/core/log gates, and Darwin no-cgo test-binary cross-compiles pass.

Not complete / weakly verified:

- Actual macOS execution is not verified from this Linux ROCm host; only Darwin no-cgo test binaries were cross-compiled.
- Production full-model Qwen/Gemma/Gemma4 prefill/decode/generate kernels remain intentionally `not_linked`; current live Gemma4 evidence proves safetensors inspection, memory fit, HIP weight copy, device-resident BF16 embedding lookup, RMSNorm, q/k/v projection, all-query-head RoPE, one-token GQA attention, o projection, residual add, post-attention RMSNorm, gate/up projection, SwiGLU, down projection, and final residual add from real Gemma4 tensors, plus explicit decode-not-linked status, not real text generation from Gemma4-E2B.
- Fully HIP-owned KV restore/reuse, production MoE paging, production JANGTQ/codebook integration, production model-family LoRA, production scorer/rerank integration, and production training kernels remain planned/experimental rather than complete runtime implementations.

### Still Open

- Real native prefill/decode kernels remain unimplemented beyond launch-packet/device-memory consumers, composed Qwen/Gemma small decode smokes using fixture and loaded device-resident weights, the typed loaded-model small decode smoke, the toy `rocm_tiny_prefill`/`rocm_tiny_decode` fixtures, full Gemma4-E2B safetensors weight load into HIP memory, real Gemma4 BF16 `embed_tokens.weight` lookup, Gemma embedding scale, real Gemma4 layer-0 BF16 `input_layernorm.weight` RMSNorm, real Gemma4 layer-0 BF16 `q_proj`/`k_proj`/`v_proj` projections, all-query-head q/k RoPE, one-token GQA attention, real Gemma4 BF16 `o_proj` projection from attention concat, residual add, `post_attention_layernorm`, BF16 MLP gate/up/down projections, SwiGLU, and final residual smokes with explicit decode-not-linked status.
- The loaded-model path can launch the toy projection kernel when `GO_ROCM_KERNEL_HSACO` is set, tiny vocab-major f32 loaded models with f32/f16/raw-q8/JANGTQ/codebook output heads can run toy generation/classification through `rocm_tiny_prefill`/`rocm_tiny_decode` plus packed/codebook output logits through `rocm_jangtq_projection`, `rocm_codebook_lookup`, and `rocm_projection`, experimental embedding/rerank through `rocm_embedding_mean_pool`/`rocm_rerank_cosine`, experimental `rocm-tiny-lora` output-head adapter application through `rocm_lora_projection`, and package-local toy cross-entropy/distillation KL/GRPO advantage hooks through `rocm_cross_entropy_loss`/`rocm_distillation_kl_loss`/`rocm_grpo_advantage`; tiny linked decode/prefill capability, benchmark, and eval quality-probe labels now mark `kernel_scope=toy_tiny_fixture` with `production_decode=not_linked` and `production_prefill=not_linked`, LoRA capability labels mark `kernel_scope=loaded_adapter_fixtures` with `production_adapter_application=not_linked`, and the toy loss hooks gate on explicit `cross_entropy_kernel`, `distillation_kernel`, and `grpo_kernel` status fields rather than broad overall kernel status. BERT sequence-classification packs can run experimental positive-logit rerank over f32/f16 classifier heads plus experimental classifier LoRA adapters, and fixture-scale Qwen/Gemma loaded models can run one typed small decode step from a loaded embedding row with fp16/q8/k-q8-v-q4 package-local/device-KV incremental-append success and append-failure coverage plus experimental LM-head LoRA; production Qwen/Gemma prefill/decode/generate remains explicitly not linked.
- The HIP kernel-source hardware gate passed with `GO_ROCM_RUN_HIP_TESTS=1` and `GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco`, including the vector-scale primitive; BF16/q4 Gemma4-E2B model-pack and decode-status smokes pass on the UUID-pinned RX 7800 XT; and the opt-in cache/KV hardware smoke now passes with `GO_ROCM_RUN_CACHE_TESTS=1`.
- MoE/JANGTQ/codebook metadata is recognised as experimental `runtime_status=metadata_only`; toy MoE router/lazy-expert, JANGTQ/MXTQ packed-projection, loaded tiny JANGTQ/codebook output-head logits, and codebook lookup launch fixtures exist, but production MoE expert paging and general model-integrated packed/codebook execution are still pending.
- Fully HIP-owned KV cache pages, disk-backed HIP KV cache beyond portable snapshot remirroring, production MoE router/lazy experts, production JANGTQ/codebook integration, production model-family LoRA adapter application, production cross-encoder/scorer rerank integration, and production training kernels beyond the toy cross-entropy/distillation/GRPO fixtures remain planned.

Latest verification after adding an opt-in Gemma4 q4 forward-depth smoke:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 rocminfo | rg -n "Name:|Uuid:|Marketing Name|gfx"
22:  Name:                    AMD Ryzen 9 9950X 16-Core Processor
23:  Uuid:                    CPU-XX
24:  Marketing Name:          AMD Ryzen 9 9950X 16-Core Processor
25:  Vendor Name:             CPU
87:  Name:                    gfx1100
88:  Uuid:                    GPU-880ed6479d653a85
89:  Marketing Name:          AMD Radeon RX 7800 XT
90:  Vendor Name:             AMD
163:      Name:                    amdgcn-amd-amdhsa--gfx1100
181:      Name:                    amdgcn-amd-amdhsa--gfx11-generic

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_FORWARD_LAYERS=5 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:91: Gemma4 q4 5-layer single-token forward greedy token=249318 score=135.845215
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (6.84s)
PASS
ok dappco.re/go/rocm 6.858s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_FORWARD_LAYERS=16 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:91: Gemma4 q4 16-layer single-token forward greedy token=198629 score=81.435280
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (6.81s)
PASS
ok dappco.re/go/rocm 6.824s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_FORWARD_LAYERS=35 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:91: Gemma4 q4 35-layer single-token forward greedy token=236978 score=63.046680
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (7.72s)
PASS
ok dappco.re/go/rocm 7.731s

go test ./go -count=1
ok dappco.re/go/rocm 0.141s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.105s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.636s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.183s
```

`GO_ROCM_GEMMA4_Q4_FORWARD_LAYERS` is now an opt-in model-smoke depth knob. The default q4 model smoke remains fast and still runs layer 0 plus selected layer-geometry checks. When the env var is set, the smoke runs a real q4 single-token forward over the first N loaded Gemma4 decoder layers, then final norm, the tied q4 LM-head projection, and greedy sampling. The 35-layer run above is a full-depth device-resident packed-tensor forward on the UUID-pinned RX 7800 XT. It is still not production generation: there is no sequence prefill, reusable KV cache, decode loop, tokenizer text emission, or production prefill/decode capability link yet.

Follow-up correction after checking the `go-mlx` dev e2e path:

Gemma decoder residual ordering now matches the sibling implementation: the post-attention norm is added back to the pre-norm hidden state, not to the input-layernorm output. `hipRunGemma4Q4DecoderLayer` was corrected accordingly, and the BF16 hardware smoke now uses the scaled embedding as the attention residual left-hand side. The fake-driver q4 test has a nonzero-input regression check that would fail if the residual accidentally used the normalized layer input again.

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_FORWARD_LAYERS=35 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:91: Gemma4 q4 35-layer single-token forward greedy token=97241 score=75.844208
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (7.94s)
PASS
ok dappco.re/go/rocm 7.960s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (17.95s)
PASS
ok dappco.re/go/rocm 17.966s

go test ./go -count=1
ok dappco.re/go/rocm 0.142s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.105s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.617s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.185s
```

Latest verification after adding experimental Gemma4 q4 cached-token decode smoke:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=5 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:91: Gemma4 q4 5-layer greedy decode generated tokens=[158750 158750]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (6.59s)
PASS
ok dappco.re/go/rocm 6.607s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=35 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:91: Gemma4 q4 35-layer greedy decode generated tokens=[97241 238036]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (9.55s)
PASS
ok dappco.re/go/rocm 9.563s

go test ./go -count=1
ok dappco.re/go/rocm 0.145s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.616s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.187s
```

The q4 package runner now has an explicit experimental K/V state path. `hipRunGemma4Q4DecoderLayer` can take prior RoPE'd keys and values, append the current layer K/V, and attend over the resulting token history. `hipRunGemma4Q4SingleTokenForwardWithState` threads that state through all loaded layers, and `hipRunGemma4Q4GreedyDecode` samples and feeds tokens back through the cached state. The opt-in hardware knobs are `GO_ROCM_GEMMA4_Q4_DECODE_LAYERS` and `GO_ROCM_GEMMA4_Q4_DECODE_TOKENS`; both must be set, and token count must be greater than 1 so the smoke exercises an actual cached decode step.

This is still not production decode. K/V is held in package-local Go slices and re-uploaded to the attention primitive each step, there is no tokenizer text emission, no sequence prefill batching, no HIP-owned reusable KV page table integration, and `production_decode` plus `production_kv_cache_backing` stay `not_linked`. It does prove that the real q4 Gemma4-E2B pack can run a full-depth two-token greedy decode-style loop on the UUID-pinned RX 7800 XT using loaded device-resident packed tensors.

Latest verification after applying Gemma4 final-logit softcap in the q4 path:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_FORWARD_LAYERS=35 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:91: Gemma4 q4 35-layer single-token forward greedy token=97241 score=29.620268
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (7.71s)
PASS
ok dappco.re/go/rocm 7.727s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=35 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:91: Gemma4 q4 35-layer greedy decode generated tokens=[97241 238036]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (9.66s)
PASS
ok dappco.re/go/rocm 9.675s

go test ./go -count=1
ok dappco.re/go/rocm 0.140s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.108s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.622s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.179s
```

Loaded Gemma4 q4 layer configs now default `final_logit_softcap` to `30`. Q4 layer-0, single-token forward, and cached greedy decode labels include `logit_softcap`, and logits are transformed with `tanh(logit/softcap)*softcap` before greedy sampling. The greedy token ID stayed stable after softcapping; the 35-layer first-token score changed from the uncapped `75.844208` to `29.620268`, which confirms the cap is applied in the real RX 7800 XT path.

Latest verification after extending the q4 cached decode smoke to multi-token prompts:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=35 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_PROMPT_TOKENS=0,1 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:91: Gemma4 q4 35-layer greedy decode prompt=[0 1] generated tokens=[194680 194680]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (11.39s)
PASS
ok dappco.re/go/rocm 11.412s

go test ./go -count=1
ok dappco.re/go/rocm 0.142s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.105s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.634s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.185s
```

`GO_ROCM_GEMMA4_Q4_DECODE_PROMPT_TOKENS` can now provide a comma-separated token-ID prompt for the opt-in q4 decode smoke. The default remains token `0`. The fake-driver test now uses a two-token prompt and verifies the expected three forward steps/state tokens for a two-token generation request. The full 35-layer RX 7800 XT smoke above proves the package-runner state path handles a real prompt length greater than one before cached generation.

Latest verification after adding Gemma4 q4 sliding-window K/V trimming:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=35 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_PROMPT_TOKENS=0,1 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:91: Gemma4 q4 35-layer greedy decode prompt=[0 1] generated tokens=[194680 194680]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (14.20s)
PASS
ok dappco.re/go/rocm 14.217s

go test ./go -count=1
ok dappco.re/go/rocm 0.142s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.105s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.617s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.197s
```

Layer configs now carry `SlidingWindow`. Real Gemma4 q4 sliding layers default to `512` retained K/V tokens, while 512-dim full-attention layers default to `0` for unbounded package-runner state. `hipRunGemma4Q4DecoderLayer` trims appended K/V to the layer window before attention. The fake-driver test forces a two-token window on one layer and verifies it trims while a full-attention-style layer keeps all three forward-step tokens. The selected real-layer smoke also checks the inferred window values for layer 4 and layer 15.

Latest verification after adding an explicit Gemma4 q4 token-ID prompt path through the public `Generate` stream:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT=tokens:0,1 GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:92: Gemma4 q4 public Generate prompt="tokens:0,1" generated tokens=[194680 194680] text=[" عج" " عج"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (11.53s)
PASS
ok dappco.re/go/rocm 11.547s

go test ./go -count=1
ok dappco.re/go/rocm 0.137s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.628s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.167s
```

`hipNativeProjectionKernelSet.Generate` now recognizes prompts with an explicit `tokens:` prefix and routes them through the experimental Gemma4 q4 cached decode runner. The public smoke uses `GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT=tokens:0,1` and `GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2`, so it exercises the normal `TextModel.Generate` stream while still using token IDs for the prompt. Safetensors loads now carry the tokenizer sidecar path into the HIP model, and generated token IDs decode through the loaded Hugging Face vocab/added-token sidecar when present; the live q4 smoke produced decoded text `[" عج" " عج"]` instead of `<token:N>` placeholders. A normal text prompt such as `Generate("hello")` still returns the existing `native decode kernels are not linked yet` error because plain text prompts are not routed into the experimental q4 path by default, and production prefill/decode capability remains not-linked.

Latest verification after adding Hugging Face BPE sidecar encode plus an explicit `text:` prompt route for the experimental Gemma4 q4 public stream:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:92: Gemma4 q4 public Generate prompt="text:Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (11.81s)
PASS
ok dappco.re/go/rocm 11.833s

go test ./go -count=1
ok dappco.re/go/rocm 0.136s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.104s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.615s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.173s
```

Loaded safetensors HIP models now use the tokenizer sidecar for `Encode`/`Decode` when it is present. The q4 public stream accepts either `tokens:` token-ID prompts or `text:` prompts that are encoded through that sidecar; the real Gemma4 tokenizer hardware smoke verified `Encode("Hello world") == [9259 1902]` before generation. This remains an explicit experimental q4 path: plain `Generate("hello")` still exercises the production surface and returns the native decode not-linked error, and the q4 loop still uses package-local Go K/V state rather than HIP-owned reusable KV pages.

Latest verification after adding an explicit env-gated plain-text q4 Gemma4 route:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:94: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (11.81s)
PASS
ok dappco.re/go/rocm 11.832s

go test ./go -count=1
ok dappco.re/go/rocm 0.139s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.108s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.622s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.168s
```

`GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1` lets a loaded Gemma4 q4 model route an ordinary prompt string through the experimental q4 GPU loop without requiring the `text:` prefix. The default remains unchanged when the env var is absent, so normal production-surface generation still returns the explicit native decode not-linked error. The plain-prompt route preserves prompt whitespace, uses the Hugging Face BPE sidecar for tokenization, and remains limited by the package-local q4 K/V state path.

Latest verification after changing the experimental q4 public stream from precomputing all generated tokens to yielding one token at a time:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:94: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (12.69s)
PASS
ok dappco.re/go/rocm 12.714s

go test ./go -count=1
ok dappco.re/go/rocm 0.137s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.612s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.173s
```

`hipGemma4Q4GenerateTokenSeq` now runs prompt tokens to seed q4 K/V state, yields the current greedy token immediately, and only executes the next q4 forward step after the consumer accepts the yielded token. The fake-driver test counts embedding launches to prove a consumer break after the first yielded token does not precompute the next decode step. The hardware public-stream smoke now also asserts generated-token metrics and, for env-gated plain prompts, tokenizer prompt-token metrics. This improves cancellation/early-stop behavior for the experimental public stream, but it still uses package-local Go K/V slices and does not change production decode/prefill capability status.

Latest verification after exposing the loaded Gemma4 q4 public generation route in capability reporting:

```text
go test ./go -run 'TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability|TestNativeContract_RocmModelCapabilitiesUseNativeKernelStatus|TestNativeContract_RocmTinyFixtureCapabilitiesLabelProductionPending' -count=1 -v
=== RUN   TestNativeContract_RocmModelCapabilitiesUseNativeKernelStatus_Good
--- PASS: TestNativeContract_RocmModelCapabilitiesUseNativeKernelStatus_Good (0.00s)
=== RUN   TestNativeContract_RocmTinyFixtureCapabilitiesLabelProductionPending_Good
--- PASS: TestNativeContract_RocmTinyFixtureCapabilitiesLabelProductionPending_Good (0.00s)
=== RUN   TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability_Good
--- PASS: TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.008s

go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:94: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (12.73s)
PASS
ok dappco.re/go/rocm 12.750s

go test ./go -count=1
ok dappco.re/go/rocm 0.142s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.626s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.182s
```

`rocmModel.Capabilities()` now detects loaded Gemma4 MLX-q4 models whose q4 forward config is available and marks only `CapabilityGenerate` experimental with `kernel_scope=loaded_gemma4_q4_experimental_generate`, `gemma4_q4_decode_kernel=linked`, `decode_quant=mlx_q4`, and prompt modes `tokens,text,env_plain_text`. `CapabilityChat`, `CapabilityBatchGenerate`, speculative decode, and prompt-lookup decode remain planned unless the production/native decode kernel is linked. The q4 labels explicitly keep `production_decode=not_linked` and `production_kv_cache_backing=not_linked`, so capability clients can discover the live q4 route without mistaking it for production decode parity.

Latest verification after adding a Gemma4 q4 device KV mirror wrapper:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[158750 158750]
    hip_hardware_test.go:94: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (12.85s)
PASS
ok dappco.re/go/rocm 12.868s

go test ./go -count=1
ok dappco.re/go/rocm 0.139s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.620s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.187s
```

`hipMirrorGemma4Q4DecodeState` now mirrors each Gemma4 q4 decoder layer's package-local K/V vectors into `rocmDeviceKVCache` HIP pages, creates per-layer descriptor tables and launch descriptors, reports device-mirror labels, and can copy the mirrored state back to host for validation. The fake-driver q4 test proves layer token counts, descriptor/page cleanup, bad input errors, and fp16 round-trip tolerance. The hardware q4 decode smoke now exercises the mirror and host round-trip on the RX 7800 XT. This is still a mirror/descriptor stepping stone: q4 attention kernels continue to consume host slices today, so `production_kv_cache_backing` remains `not_linked`.

Latest verification after carrying the Gemma4 q4 device KV mirror through greedy decode instead of reconstructing it after decode:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[158750 158750]
    hip_hardware_test.go:94: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (12.71s)
PASS
ok dappco.re/go/rocm 12.738s

go test ./go -count=1
ok dappco.re/go/rocm 0.139s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.619s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.178s
```

`hipRunGemma4Q4GreedyDecode` now accepts `MirrorDeviceKV`/`DeviceKVMode` and returns a carried `DeviceState`. Each forward step updates the mirror from the previous host K/V state to the next host K/V state: append-only layers reuse existing HIP pages with one appended page plus a refreshed descriptor table, while sliding-window or otherwise non-appendable layers are remirrored. Decode labels now include Gemma4 q4 device-KV labels, and the fake-driver test proves append/remirror accounting, descriptor cleanup, closed-state errors, and invalid mode rejection. This still does not make attention consume paged device descriptors; production K/V cache backing remains `not_linked` until that kernel/API path exists.

Latest verification after adding descriptor-backed attention and routing the Gemma4 q4 decode smoke through it:

```text
go test ./go -run 'TestHIPKernels_AttentionLaunchArgs|TestHIPKernelSource_ExportsLaunchABI|TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPKernelSource_ExportsLaunchABI_Good
--- PASS: TestHIPKernelSource_ExportsLaunchABI_Good (0.00s)
=== RUN   TestHIPKernels_AttentionLaunchArgs_Good
--- PASS: TestHIPKernels_AttentionLaunchArgs_Good (0.00s)
=== RUN   TestHIPKernels_AttentionLaunchArgs_Bad
--- PASS: TestHIPKernels_AttentionLaunchArgs_Bad (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.007s

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestHIPHardwareTransformerKernelSource_Good -count=1 -v
=== RUN   TestHIPHardwareTransformerKernelSource_Good
=== RUN   TestHIPHardwareTransformerKernelSource_Good/moe-router
=== RUN   TestHIPHardwareTransformerKernelSource_Good/moe-lazy-experts
=== RUN   TestHIPHardwareTransformerKernelSource_Good/loaded-small-q8
=== RUN   TestHIPHardwareTransformerKernelSource_Good/loaded-small-k-q8-v-q4
=== RUN   TestHIPHardwareTransformerKernelSource_Good/fp16-output
=== RUN   TestHIPHardwareTransformerKernelSource_Good/q8-output
=== RUN   TestHIPHardwareTransformerKernelSource_Good/loaded-tiny-f32
=== RUN   TestHIPHardwareTransformerKernelSource_Good/loaded-tiny-q8
--- PASS: TestHIPHardwareTransformerKernelSource_Good (0.13s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/moe-router (0.00s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/moe-lazy-experts (0.00s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/loaded-small-q8 (0.02s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/loaded-small-k-q8-v-q4 (0.02s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/fp16-output (0.00s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/q8-output (0.00s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/loaded-tiny-f32 (0.00s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/loaded-tiny-q8 (0.00s)
PASS
ok dappco.re/go/rocm 0.144s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[158750 158750]
    hip_hardware_test.go:94: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (12.17s)
PASS
ok dappco.re/go/rocm 12.197s

go test ./go -count=1
ok dappco.re/go/rocm 0.144s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.636s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.181s
```

`rocm_attention` now supports the existing contiguous f32 K/V path and an fp16 device-KV descriptor path using the reserved fields in the same 96-byte launch packet. `hipAttentionRequest` can launch attention directly over a `rocmDeviceKVCache` plus descriptor table, and fake-driver coverage decodes the descriptor/table/pages instead of reading uploaded contiguous K/V buffers. Gemma4 q4 decode can opt into `DeviceKVAttention`, which mirrors the updated per-layer K/V into fp16 HIP pages and runs attention over the device descriptor; the fake q4 test checks descriptor-backed attention launches, and the RX 7800 XT q4 smoke verifies this path with the q4 model. This is a real device-KV attention stepping stone, but q4 still remirrors/updates host-derived K/V around the layer and production prefill/decode are still not linked.

Latest verification after extending descriptor-backed attention to q8 and k-q8-v-q4 device-KV pages:

```text
go test ./go -run 'TestHIPKernels_AttentionLaunchArgs|TestHIPKernelSource_ExportsLaunchABI|TestHIPGemma4Q4Layer0' -count=1 -v
=== RUN   TestHIPKernelSource_ExportsLaunchABI_Good
--- PASS: TestHIPKernelSource_ExportsLaunchABI_Good (0.00s)
=== RUN   TestHIPKernels_AttentionLaunchArgs_Good
--- PASS: TestHIPKernels_AttentionLaunchArgs_Good (0.00s)
=== RUN   TestHIPKernels_AttentionLaunchArgs_Bad
--- PASS: TestHIPKernels_AttentionLaunchArgs_Bad (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.007s

hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o /tmp/go-rocm-kernels-gfx1100.hsaco

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestHIPHardwareTransformerKernelSource_Good -count=1 -v
=== RUN   TestHIPHardwareTransformerKernelSource_Good
=== RUN   TestHIPHardwareTransformerKernelSource_Good/moe-router
=== RUN   TestHIPHardwareTransformerKernelSource_Good/moe-lazy-experts
=== RUN   TestHIPHardwareTransformerKernelSource_Good/loaded-small-q8
=== RUN   TestHIPHardwareTransformerKernelSource_Good/loaded-small-k-q8-v-q4
=== RUN   TestHIPHardwareTransformerKernelSource_Good/fp16-output
=== RUN   TestHIPHardwareTransformerKernelSource_Good/q8-output
=== RUN   TestHIPHardwareTransformerKernelSource_Good/loaded-tiny-f32
=== RUN   TestHIPHardwareTransformerKernelSource_Good/loaded-tiny-q8
--- PASS: TestHIPHardwareTransformerKernelSource_Good (0.14s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/moe-router (0.00s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/moe-lazy-experts (0.00s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/loaded-small-q8 (0.02s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/loaded-small-k-q8-v-q4 (0.02s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/fp16-output (0.00s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/q8-output (0.00s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/loaded-tiny-f32 (0.00s)
    --- PASS: TestHIPHardwareTransformerKernelSource_Good/loaded-tiny-q8 (0.00s)
PASS
ok dappco.re/go/rocm 0.154s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[158750 158750]
    hip_hardware_test.go:94: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (12.09s)
PASS
ok dappco.re/go/rocm 12.107s

go test ./go -count=1
ok dappco.re/go/rocm 0.142s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.612s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.167s
```

The attention kernel now decodes fp16, q8, and q4 descriptor page encodings directly from HIP device-KV pages. The fake attention path verifies q8 and k-q8-v-q4 descriptor output against restored quantized K/V reference tensors, the Gemma4 q4 fake forward path exercises k-q8-v-q4 descriptor-backed attention labels and launches, and the RX 7800 XT hardware transformer smoke covers q8 and k-q8-v-q4 descriptor-backed attention. This still leaves production decode and production KV-cache backing as `not_linked`; the Gemma4 q4 public Generate path remains explicitly experimental.

Latest verification after letting Gemma4 q4 descriptor-backed attention reuse carried device-KV pages when the layer can append:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0|TestHIPKernels_AttentionLaunchArgs' -count=1 -v
=== RUN   TestHIPKernels_AttentionLaunchArgs_Good
--- PASS: TestHIPKernels_AttentionLaunchArgs_Good (0.00s)
=== RUN   TestHIPKernels_AttentionLaunchArgs_Bad
--- PASS: TestHIPKernels_AttentionLaunchArgs_Bad (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.006s

go test ./go -count=1
ok dappco.re/go/rocm 0.139s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.109s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[158750 158750]
    hip_hardware_test.go:94: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (12.19s)
PASS
ok dappco.re/go/rocm 12.210s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.611s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.170s
```

Gemma4 q4 greedy decode now passes the carried `hipGemma4Q4DeviceDecodeState` into the next forward step when `MirrorDeviceKV` and `DeviceKVAttention` are both enabled. Each decoder layer appends the current token to the prior device-KV pages for descriptor-backed attention when the host state is append-compatible, and falls back to remirroring host K/V for first-token and sliding-window cases. Forward labels now count `attention_kv_append_layers` and `attention_kv_remirror_layers`; fake coverage proves the expected append/remirror mix across a two-layer sliding-window decode. This reduces avoidable full-layer attention remirrors, but the final state update still re-appends/remirrors after the step and production decode/KV ownership remain `not_linked`.

Follow-up hardening in the same slice rejects direct decoder-layer calls whose prior device-KV cache is closed or has mismatched mode, token count, or vector widths before any attention launch. After that validation change, the package gates, workspace gates, and RX 7800 XT Gemma4 q4 smoke were rerun; the live model smoke again produced prompt tokens `[9259 1902]`, generated tokens `[20842 110288]`, and decoded fragments `["ాత" "ði"]` with `ok dappco.re/go/rocm 12.317s`.

Latest verification after making the Gemma4 q4 forward step return the persistent device-KV state it already built for descriptor-backed attention:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0|TestHIPKernels_AttentionLaunchArgs|TestKVCache_Good_DeviceMirrorAppendsDecodeTokenIncrementally|TestKVCache_Bad_DeviceMirrorAppend' -count=1 -v
=== RUN   TestHIPKernels_AttentionLaunchArgs_Good
--- PASS: TestHIPKernels_AttentionLaunchArgs_Good (0.00s)
=== RUN   TestHIPKernels_AttentionLaunchArgs_Bad
--- PASS: TestHIPKernels_AttentionLaunchArgs_Bad (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
=== RUN   TestKVCache_Good_DeviceMirrorAppendsDecodeTokenIncrementally
--- PASS: TestKVCache_Good_DeviceMirrorAppendsDecodeTokenIncrementally (0.00s)
=== RUN   TestKVCache_Bad_DeviceMirrorAppendRollbackOnDescriptorFailure
--- PASS: TestKVCache_Bad_DeviceMirrorAppendRollbackOnDescriptorFailure (0.00s)
=== RUN   TestKVCache_Bad_DeviceMirrorAppendScratchCloseDoesNotFreeSourcePages
--- PASS: TestKVCache_Bad_DeviceMirrorAppendScratchCloseDoesNotFreeSourcePages (0.00s)
PASS
ok dappco.re/go/rocm 0.009s

go test ./go -count=1
ok dappco.re/go/rocm 0.138s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[158750 158750]
    hip_hardware_test.go:94: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (12.09s)
PASS
ok dappco.re/go/rocm 12.116s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.611s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.168s
```

`hipRunGemma4Q4SingleTokenForwardWithState` can now return a retained `hipGemma4Q4DeviceDecodeState` when `ReturnDeviceState` is set. Greedy decode uses that path whenever both `MirrorDeviceKV` and `DeviceKVAttention` are enabled, so the same per-layer device pages used by descriptor-backed attention become the next persistent decode state after the token step succeeds. Ownership transfer from the previous state happens only after the full forward step, logits, and greedy sample succeed; failures close only scratch/new pages and leave the previous state intact. This removes the previous post-forward duplicate append/remirror path for q4 descriptor attention, while production decode and production KV-cache backing still remain `not_linked`.

Latest verification after wiring the public Gemma4 q4 `Generate` stream to the same carried descriptor-backed device-KV path:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0|TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability|TestHIPKernels_AttentionLaunchArgs' -count=1 -v
=== RUN   TestHIPKernels_AttentionLaunchArgs_Good
--- PASS: TestHIPKernels_AttentionLaunchArgs_Good (0.00s)
=== RUN   TestHIPKernels_AttentionLaunchArgs_Bad
--- PASS: TestHIPKernels_AttentionLaunchArgs_Bad (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
=== RUN   TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability_Good
--- PASS: TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.007s

go test ./go -count=1
ok dappco.re/go/rocm 0.136s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[158750 158750]
    hip_hardware_test.go:94: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (12.57s)
PASS
ok dappco.re/go/rocm 12.597s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.617s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.169s
```

The experimental public Gemma4 q4 `Generate` path now uses `DeviceKVAttention`, a forward-returned device state, and fp16 descriptor-backed KV pages instead of the old contiguous host-KV attention path. The stream closes the carried device state on normal completion, errors, stop-token exits, and early consumer cancellation. Fake coverage now asserts descriptor-backed attention launches for normal and early-stopped public q4 generation, and the capability labels advertise `attention_kv_backing=hip_device_descriptor` plus `gemma4_q4_device_kv_state=forward_returned_device_state`. Production decode and production KV-cache backing remain explicitly `not_linked`.

Latest verification after switching experimental public Gemma4 q4 `Generate` from fp16 device-KV pages to compact `k-q8-v-q4` pages:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0|TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability|TestHIPKernels_AttentionLaunchArgs' -count=1 -v
=== RUN   TestHIPKernels_AttentionLaunchArgs_Good
--- PASS: TestHIPKernels_AttentionLaunchArgs_Good (0.00s)
=== RUN   TestHIPKernels_AttentionLaunchArgs_Bad
--- PASS: TestHIPKernels_AttentionLaunchArgs_Bad (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
=== RUN   TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability_Good
--- PASS: TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.008s

go test ./go -count=1
ok dappco.re/go/rocm 0.136s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[158750 158750]
    hip_hardware_test.go:94: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (12.92s)
PASS
ok dappco.re/go/rocm 12.942s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.617s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.171s
```

Public Gemma4 q4 `Generate` now uses compact descriptor-backed K/V with q8 keys and q4 values (`attention_kv_mode=k-q8-v-q4`) while retaining the forward-returned device state. The RX 7800 XT smoke produced the same prompt/generated token IDs as the fp16-page path for the two-token `Hello world` smoke. This is still an experimental generation route; production decode and production KV-cache backing remain explicitly `not_linked`.

Latest verification after switching the separate Gemma4 q4 greedy-decode smoke to compact `k-q8-v-q4` device-KV pages:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0|TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability|TestHIPKernels_AttentionLaunchArgs' -count=1 -v
=== RUN   TestHIPKernels_AttentionLaunchArgs_Good
--- PASS: TestHIPKernels_AttentionLaunchArgs_Good (0.00s)
=== RUN   TestHIPKernels_AttentionLaunchArgs_Bad
--- PASS: TestHIPKernels_AttentionLaunchArgs_Bad (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Good
--- PASS: TestHIPGemma4Q4Layer0_Good (0.00s)
=== RUN   TestHIPGemma4Q4Layer0_Bad
--- PASS: TestHIPGemma4Q4Layer0_Bad (0.00s)
=== RUN   TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability_Good
--- PASS: TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.007s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[236758 39905]
    hip_hardware_test.go:94: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (14.00s)
PASS
ok dappco.re/go/rocm 14.019s

go test ./go -count=1
ok dappco.re/go/rocm 0.135s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.613s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.170s
```

The q4 greedy-decode fake and hardware smokes now request `DeviceKVMode=k-q8-v-q4` and assert compact attention/device-state labels. Device-state round-trip checks compare against a host-side compact-KV reference that uses the actual device page segmentation, so q8/q4 per-page scales are validated without pretending the restored cache should equal the original fp32 host values. The compact greedy decode path changes the direct smoke token IDs, as expected from q4 value-cache attention numerics; public `Generate("Hello world")` remained stable in the two-token smoke. Production decode and production KV-cache backing remain explicitly `not_linked`.

Latest verification after adding a package-local Gemma4 q4 prefill/decode bridge over the carried descriptor-backed device state:

```text
go test ./go -run 'TestHIPGemma4Q4PackagePrefillDecode|TestHIPGemma4Q4Layer0|TestHIPKernels_NotLinked|TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability' -count=1 -v
PASS
ok dappco.re/go/rocm 0.009s

go test ./go -count=1
ok dappco.re/go/rocm 0.140s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[236758 39905]
    hip_hardware_test.go:94: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (13.35s)
PASS
ok dappco.re/go/rocm 13.373s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.632s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.182s
```

The internal HIP `Prefill`/`Decode` result/request structs now have package-local Gemma4 q4 state slots. The new q4 package bridge runs prompt prefill through the same `k-q8-v-q4` descriptor-backed single-token forwards used by public Generate, returns per-layer host state plus a forward-retained device state, and lets the next q4 decode consume that state without remirroring every layer. Result labels advertise `gemma4_q4_prefill_kernel`/`gemma4_q4_decode_kernel` as experimental while keeping `prefill_kernel`, `decode_kernel`, `production_prefill`, `production_decode`, and `production_kv_cache_backing` explicitly `not_linked`.

Follow-up verification after adding the package bridge to the live Gemma4 q4 smoke:

```text
go test ./go -run 'TestHIPGemma4Q4PackagePrefillDecode|TestHIPGemma4Q4Layer0|TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability' -count=1 -v
PASS
ok dappco.re/go/rocm 0.008s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[236758 39905]
    hip_hardware_test.go:94: Gemma4 q4 package Prefill/Decode prompt=[0] next=150930 decoded=1865 text="ology"
    hip_hardware_test.go:95: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (16.99s)
PASS
ok dappco.re/go/rocm 17.010s

go test ./go -count=1
ok dappco.re/go/rocm 0.144s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.651s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.191s
```

The opt-in hardware smoke now calls `hipLoadedModel.Prefill` and `DecodeToken` for loaded Gemma4 q4, validating the package-first handoff on the RX 7800 XT rather than only through fixture helpers. The live package prefill/decode path returns experimental q4 labels and closes/transfers the prior device state during decode; public Generate remains stable on the same model and kernel build.

Latest verification after wiring public Gemma4 q4 `BatchGenerate` over the same experimental q4 Generate path:

```text
go test ./go -run 'TestHIPGemma4Q4PackagePrefillDecode|TestHIPGemma4Q4Layer0|TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability' -count=1 -v
PASS
ok dappco.re/go/rocm 0.008s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[236758 39905]
    hip_hardware_test.go:94: Gemma4 q4 package Prefill/Decode prompt=[0] next=150930 decoded=1865 text="ology"
    hip_hardware_test.go:95: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (18.50s)
PASS
ok dappco.re/go/rocm 18.526s

go test ./go -count=1
ok dappco.re/go/rocm 0.142s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.616s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.180s
```

Gemma4 q4 `BatchGenerate` now validates prompts, resolves `tokens:`/experimental text prompts through the same q4 prompt parser used by `Generate`, and records per-prompt not-linked errors for plain prompts that cannot use the q4 route. Capability reporting now marks `CapabilityBatchGenerate` experimental for loaded Gemma4 MLX-q4 with `kernel_scope=loaded_gemma4_q4_experimental_batch_generate`, while `Chat`, speculative decode, and prompt-lookup decode remain planned until production decode is linked. The live hardware smoke calls public `BatchGenerate("tokens:0")` in addition to Generate and package Prefill/Decode.

Latest verification after routing Gemma4 q4 `Chat` through explicit `text:` prompt generation:

```text
go test ./go -run 'TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability|TestHIPGemma4Q4PackagePrefillDecode|TestHIPGemma4Q4Layer0' -count=1 -v
PASS
ok dappco.re/go/rocm 0.008s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[236758 39905]
    hip_hardware_test.go:94: Gemma4 q4 package Prefill/Decode prompt=[0] next=150930 decoded=1865 text="ology"
    hip_hardware_test.go:95: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (24.61s)
PASS
ok dappco.re/go/rocm 24.637s

go test ./go -count=1
ok dappco.re/go/rocm 0.145s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.622s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.184s
```

For loaded Gemma4 q4, `Chat` now applies the fallback chat template and sends it through the q4 text prompt path directly, so it no longer depends on the plain-text environment gate. Capability reporting marks `CapabilityChat` experimental with `kernel_scope=loaded_gemma4_q4_experimental_chat`, while keeping production decode/KV labels as `not_linked`. The live hardware smoke now covers public Chat alongside Generate, BatchGenerate, and package Prefill/Decode.

Latest verification after making `Benchmark` wrap Gemma4 q4 plain prompts as explicit `text:` prompts internally:

```text
go test ./go -run 'TestNativeContract_BenchmarkUsesExplicitGemma4Q4TextPrompt|TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability|TestNativeContract_Benchmark' -count=1 -v
PASS
ok dappco.re/go/rocm 0.035s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[236758 39905]
    hip_hardware_test.go:94: Gemma4 q4 package Prefill/Decode prompt=[0] next=150930 decoded=1865 text="ology"
    hip_hardware_test.go:95: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (26.40s)
PASS
ok dappco.re/go/rocm 26.424s

go test ./go -count=1
ok dappco.re/go/rocm 0.143s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.620s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.181s
```

`rocmModel.Benchmark` now leaves explicit `tokens:`/`text:` prompts alone, but wraps plain prompts as `text:` for loaded Gemma4 q4 models with a tokenizer. This lets benchmark callers use ordinary benchmark prompts without enabling the plain-text Generate environment gate, while public Generate behavior remains unchanged. The live smoke temporarily unsets `GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE` before running a one-token q4 benchmark and asserts generated output plus not-linked production prefill/decode labels.

Latest verification after applying the same explicit Gemma4 q4 text prompt wrapping to evaluation quality probes:

```text
go test ./go -run 'TestNativeContract_GeneratedPromptUsesExplicitGemma4Q4TextMode|TestNativeContract_EvaluateQualityProbes|TestNativeContract_Benchmark' -count=1
ok dappco.re/go/rocm 0.035s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[236758 39905]
    hip_hardware_test.go:94: Gemma4 q4 package Prefill/Decode prompt=[0] next=150930 decoded=1865 text="ology"
    hip_hardware_test.go:95: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (28.18s)
PASS
ok dappco.re/go/rocm 28.207s

go test ./go -count=1
ok dappco.re/go/rocm 0.144s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.628s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.188s
```

Evaluation quality probes now use the shared generated-prompt normalization: explicit `tokens:`/`text:` prompts are preserved, and plain prompts are wrapped as `text:` only for loaded Gemma4 q4 models with a tokenizer. The live smoke keeps the plain-text Generate env gate unset after public Generate and then validates both benchmark and evaluation probe generation through the explicit text route, with production prefill/decode labels still `not_linked`.

Latest verification after letting Eval loss/perplexity use Gemma4 q4 package Prefill logits:

```text
go test ./go -run 'TestNativeContract_Evaluate|TestNativeContract_GeneratedPromptUsesExplicitGemma4Q4TextMode|TestHIPGemma4Q4PackagePrefillDecode' -count=1
ok dappco.re/go/rocm 0.010s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:93: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[236758 39905]
    hip_hardware_test.go:94: Gemma4 q4 package Prefill/Decode prompt=[0] next=150930 decoded=1865 text="ology"
    hip_hardware_test.go:95: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (34.25s)
PASS
ok dappco.re/go/rocm 34.272s

go test ./go -count=1
ok dappco.re/go/rocm 0.145s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.108s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.626s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.179s
```

Eval loss collection now detects loaded Gemma4 q4 models and obtains logits through the package Prefill bridge instead of falling through to Classify. It closes the returned q4 device state immediately after collecting logits, records `eval.loss_logits_source=gemma4_q4_package_prefill`, and then reuses the existing reference/HIP cross-entropy path. The live smoke evaluates a q4 sample with `target_token_id=0` and, with HSACO configured, asserts HIP loss backend plus experimental loss status while production prefill/decode labels remain `not_linked`.

Follow-up verification after aligning capability reporting with the Gemma4 q4 eval-loss path:

```text
go test ./go -run 'TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability|TestNativeContract_Evaluate|TestNativeContract_GeneratedPromptUsesExplicitGemma4Q4TextMode' -count=1 -v
PASS
ok dappco.re/go/rocm 0.010s

go test ./go -count=1
ok dappco.re/go/rocm 0.145s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.105s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.616s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.181s
```

Loaded Gemma4 q4 capability reports now mark `CapabilityEvaluation` experimental with `kernel_scope=loaded_gemma4_q4_experimental_eval`, `eval_loss_logits_source=gemma4_q4_package_prefill`, `eval_prefill_kernel=linked`, and the same compact descriptor-backed KV labels as Generate. The report still keeps `production_prefill`, `production_decode`, and `production_kv_cache_backing` as `not_linked`.

Follow-up verification after aligning benchmark capability reporting with the Gemma4 q4 benchmark route:

```text
go test ./go -run 'TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability|TestNativeContract_Benchmark|TestNativeContract_Evaluate' -count=1
ok dappco.re/go/rocm 0.036s

go test ./go -count=1
ok dappco.re/go/rocm 0.141s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.630s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.185s
```

Loaded Gemma4 q4 capability reports now mark `CapabilityBenchmark` experimental with `kernel_scope=loaded_gemma4_q4_experimental_benchmark`, `benchmark_prompt_mode=explicit_text`, and the compact descriptor-backed KV labels from Generate. Production decode/KV labels remain `not_linked`.

Follow-up verification after wiring loaded Gemma4 q4 Classify through the package Prefill path:

```text
go test ./go -run 'TestHIPGemma4Q4PackagePrefillDecode|TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability|TestNativeContract_Classify' -count=1 -v
PASS
ok dappco.re/go/rocm 0.009s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (36.36s)
PASS
ok dappco.re/go/rocm 36.382s

go test ./go -count=1
ok dappco.re/go/rocm 0.140s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.619s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.171s
```

Loaded Gemma4 q4 `Classify` now uses q4 package Prefill logits, closes the returned device KV state, returns greedy token text through the loaded tokenizer, and only retains full logits when `WithLogits` is requested. Capability reporting marks `CapabilityClassify` experimental with `kernel_scope=loaded_gemma4_q4_experimental_classify`, `classify_logits_source=gemma4_q4_package_prefill`, descriptor-backed q4 KV labels, and production prefill/decode/KV backing still `not_linked`. The live smoke exercises public Generate, BatchGenerate, Chat, Classify, Benchmark, Eval, and the 1-layer q4 decode fixture on the RX 7800 XT.

Follow-up verification after extending the BF16 Gemma4-E2B live smoke through final norm, tied `embed_tokens` LM-head logits, Gemma final-logit softcap, and HIP greedy sampling:

```text
go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- SKIP: TestNativeDecodeSmokeKernelStatus_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

go test ./go -count=1
ok dappco.re/go/rocm 0.137s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:91: Gemma4 BF16 layer0 tied LM-head greedy token=158750 score=28.711908
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (17.40s)
PASS
ok dappco.re/go/rocm 17.422s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.641s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.188s
```

The unquantized BF16 smoke now covers a complete layer-0 path from loaded BF16 embeddings through attention, MLP, final RMSNorm, tied BF16 LM-head projection, softcapped logits, and HIP greedy sampling. This keeps BF16 as the raw-tensor correctness anchor while the q4 package path remains the faster experimental Generate/Chat/Classify/BatchGenerate/Benchmark/Eval development route.

Follow-up verification after aligning `CapabilityLogitProbe` with the loaded Gemma4 q4 Classify logits path:

```text
go test ./go -run 'TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability|TestNativeContract_ClassifyWithLogitsEmitsLogitAndEntropyProbes' -count=1 -v
PASS
ok dappco.re/go/rocm 0.007s

go test ./go -count=1
ok dappco.re/go/rocm 0.142s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.106s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.647s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.180s
```

Loaded Gemma4 q4 reports `CapabilityLogitProbe` as experimental when the q4 Classify logits path is linked, with `kernel_scope=loaded_gemma4_q4_experimental_logit_probe`, `logit_probe_source=gemma4_q4_classify_logits`, and the same package-Prefill/descriptor-backed KV labels as q4 Classify. Production prefill/decode/KV backing remain `not_linked`.

Follow-up verification after asserting q4 public Classify emits the advertised logit and entropy probe events in the live smoke:

```text
go test ./go -run 'TestNativeDecodeSmokeKernelStatus_Good|TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability' -count=1 -v
--- SKIP: TestNativeDecodeSmokeKernelStatus_Good (0.00s)
--- PASS: TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.006s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (42.32s)
PASS
ok dappco.re/go/rocm 42.347s

go test ./go -count=1
ok dappco.re/go/rocm 0.144s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.633s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.181s
```

The q4 live smoke now attaches a public `ProbeSink` around `Classify(..., WithLogits())` and verifies both `ProbeEventLogits` and `ProbeEventEntropy` are emitted from the classification wrapper while using the loaded Gemma4 q4 package Prefill logits.

Broader RX 7800 XT hardware gate verification after the BF16/q4 probe updates:

```text
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'TestHIP|TestNative' -count=1 -v
--- PASS: TestHIPHardwareAvailabilitySmoke_Good (0.03s)
--- PASS: TestHIPHardwareProjectionKernelSource_Good (0.02s)
--- PASS: TestHIPHardwareEmbeddingKernelSource_Good (0.01s)
--- PASS: TestHIPHardwareTransformerKernelSource_Good (0.10s)
--- PASS: TestHIPHardwarePrefillDecodeKernelSource_Good (0.01s)
PASS
ok dappco.re/go/rocm 0.236s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run 'Test.*Smoke|Test.*Generate|Test.*Decode' -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (42.22s)
--- PASS: TestNativeModelPackSmokeGemma4E2B_Good (0.23s)
PASS
ok dappco.re/go/rocm 42.500s
```

The broad HIP gate exercised the linked HSACO fixture kernels on `GPU-880ed6479d653a85`. The broad model-smoke gate used the q4 Gemma4-E2B pack and covered the package q4 Prefill/Decode bridge, public q4 Generate/BatchGenerate/Chat/Classify/Benchmark/Eval path, q4 Classify probe events, and Gemma4-E2B safetensors model-pack inspection.

Follow-up verification after promoting q4 speculative/prompt-lookup decode helper capabilities and exercising both helpers over the live q4 Generate path:

```text
go test ./go -run 'TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability|TestDecodeHelpers_Good_SpeculativeDecodeUsesSharedHarness|TestDecodeHelpers_Good_PromptLookupDecodeUsesLookupDraft|TestNativeDecodeSmokeKernelStatus_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 0.010s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (53.85s)
PASS
ok dappco.re/go/rocm 53.871s

go test ./go -count=1
ok dappco.re/go/rocm 0.143s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.108s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.632s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.182s
```

Loaded Gemma4 q4 now reports `CapabilitySpeculativeDecode` and `CapabilityPromptLookupDecode` as experimental when the q4 Generate path is linked. The live q4 smoke runs one-token `SpeculativeDecode` using the same q4 model as target and draft, then one-token `PromptLookupDecode` using the first generated token as the lookup candidate. Production native decode and production KV cache backing remain explicitly `not_linked`.

Follow-up verification after aligning q4 Benchmark report labels with the experimental q4 speculative/prompt-lookup helper capabilities:

```text
go test ./go -run 'TestNativeContract_BenchmarkDecodeHelperStatusUsesQ4Generate|TestNativeContract_Benchmark|TestNativeDecodeSmokeKernelStatus_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 0.036s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (53.40s)
PASS
ok dappco.re/go/rocm 53.426s

go test ./go -count=1
ok dappco.re/go/rocm 0.144s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.629s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.182s
```

Loaded q4 Benchmark reports now mark `speculative.decode` and `prompt.lookup.decode` as `experimental`, include `.source=gemma4_q4_generate`, and merge the q4 benchmark labels (`kernel_scope=loaded_gemma4_q4_experimental_benchmark`, `benchmark_prompt_mode=explicit_text`) while keeping `decode_kernel`, `prefill_kernel`, production decode, and production KV cache backing explicitly `not_linked`.

Follow-up documentation/gate pass after updating README and architecture/development/history docs to describe the Gemma4 MLX-q4 experimental public Generate/Chat/BatchGenerate/Classify/Benchmark/Eval route, q4 speculative/prompt-lookup helpers, q4 Classify logit probes, BF16 final-logit anchor, and the remaining production decode/prefill/KV not-linked caveats:

```text
rg -n "Normal model generation remains planned|Full model decode/prefill generation|local GGUF model|Full Qwen/Gemma generation remains planned|shared speculative/prompt-lookup decode helpers over experimental generation" README.md docs/architecture.md docs/development.md docs/history.md
(no matches)

git diff --check
(no output)

go test ./go -count=1
ok dappco.re/go/rocm 0.150s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.123s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.013s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.651s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.200s
```

Follow-up verification after tightening Gemma4 q4 BatchGenerate prompt coverage:

```text
go test ./go -run 'TestHIPGemma4Q4PackagePrefillDecode_Good|TestNativeDecodeSmokeKernelStatus_Good' -count=1 -v
--- SKIP: TestNativeDecodeSmokeKernelStatus_Good (0.00s)
--- PASS: TestHIPGemma4Q4PackagePrefillDecode_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.006s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (55.44s)
PASS
ok dappco.re/go/rocm 55.460s

go test ./go -count=1
ok dappco.re/go/rocm 0.146s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check
(no output)

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.641s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.186s
```

The fake-device q4 package test now covers explicit `text:` BatchGenerate and env-enabled plain text BatchGenerate. The live RX 7800 XT smoke now runs public q4 BatchGenerate with the same plain text prompt as Generate (`Hello world`) instead of only the synthetic `tokens:` prompt. Production decode, prefill, and KV cache backing remain deliberately labelled `not_linked`.

Follow-up verification after changing non-streaming text metrics to use tokenizer-aware prompt counts for Classify and BatchGenerate, matching the existing Generate prompt-token accounting:

```text
go test ./go -run 'TestNativeContract_NonStreamingTextMetricsUseTokenizerPromptCounts|TestNativeContract_BatchGenerateEnforcesStopSequencesAcrossChunks|TestHIPGemma4Q4PackagePrefillDecode_Good' -count=1 -v
--- PASS: TestHIPGemma4Q4PackagePrefillDecode_Good (0.00s)
--- PASS: TestNativeContract_BatchGenerateEnforcesStopSequencesAcrossChunks_Good (0.00s)
--- PASS: TestNativeContract_NonStreamingTextMetricsUseTokenizerPromptCounts_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.009s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (55.37s)
PASS
ok dappco.re/go/rocm 55.394s

go test ./go -count=1
ok dappco.re/go/rocm 0.142s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check
(no output)

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.623s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.182s
```

The live q4 smoke now verifies public q4 BatchGenerate metrics after batching the plain text prompt: one generated token and the real tokenizer prompt length from `Hello world` (`[9259 1902]`). Production decode, prefill, and KV cache backing remain deliberately labelled `not_linked`.

Follow-up verification after changing Chat metrics to count the applied chat template through the tokenizer, with the Gemma4 q4 path using the same trimmed `text:` prompt parser as native q4 Chat:

```text
go test ./go -run 'TestNativeContract_ChatMetricsUseTemplateTokenizerPromptCount|TestNativeContract_NonStreamingTextMetricsUseTokenizerPromptCounts|TestNativeDecodeSmokeKernelStatus_Good' -count=1 -v
--- SKIP: TestNativeDecodeSmokeKernelStatus_Good (0.00s)
--- PASS: TestNativeContract_NonStreamingTextMetricsUseTokenizerPromptCounts_Good (0.00s)
--- PASS: TestNativeContract_ChatMetricsUseTemplateTokenizerPromptCount_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.008s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (55.52s)
PASS
ok dappco.re/go/rocm 55.539s

go test ./go -count=1
ok dappco.re/go/rocm 0.144s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.107s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check
(no output)

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.626s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.185s
```

The first live assertion used raw `loaded.Encode(chatPrompt)` and failed because q4 `text:` prompts trim the template body before tokenization. The final assertion now uses `hipGemma4Q4TextPromptIDs("text:"+chatPrompt, loaded)`, matching the native q4 Chat route exactly. Production decode, prefill, and KV cache backing remain deliberately labelled `not_linked`.

Follow-up verification after changing eval token-count metrics to use the same tokenizer/template prompt-count helpers as Generate/Chat/Classify/BatchGenerate:

```text
go test ./go -run 'TestNativeContract_EvaluateMetricsUseTemplateTokenizerTokenCounts|TestNativeContract_ChatMetricsUseTemplateTokenizerPromptCount|TestNativeDecodeSmokeKernelStatus_Good' -count=1 -v
--- SKIP: TestNativeDecodeSmokeKernelStatus_Good (0.00s)
--- PASS: TestNativeContract_ChatMetricsUseTemplateTokenizerPromptCount_Good (0.00s)
--- PASS: TestNativeContract_EvaluateMetricsUseTemplateTokenizerTokenCounts_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.008s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (55.59s)
PASS
ok dappco.re/go/rocm 55.610s

go test ./go -count=1
ok dappco.re/go/rocm 0.145s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.109s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check
(no output)

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.625s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.186s
```

Eval sample token counts now use `promptTokenCount` for text/prompt samples and `chatPromptTokenCount` for message samples, so explicit q4 `text:`/`tokens:` prompts and applied chat templates are counted through the same tokenizer path as generation. The live q4 eval smoke now asserts `eval.Metrics.Tokens` and `eval.tokens` labels against the real Gemma tokenizer for `Hi`. Production decode, prefill, and KV cache backing remain deliberately labelled `not_linked`.

Follow-up verification after making public `BatchGenerate` record per-prompt result errors in `model.Err()` when the native batch call itself returns a nil top-level error:

```text
go test ./go -run 'TestNativeContract_BatchGenerateRecordsPerPromptError|TestNativeContract_BatchGenerateRecordsNativeError|TestHIPGemma4Q4PackagePrefillDecode_Good' -count=1 -v
--- PASS: TestHIPGemma4Q4PackagePrefillDecode_Good (0.00s)
--- PASS: TestNativeContract_BatchGenerateRecordsNativeError_Bad (0.00s)
--- PASS: TestNativeContract_BatchGenerateRecordsPerPromptError_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.009s

go test ./go -count=1
ok dappco.re/go/rocm 0.143s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.110s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check
(no output)

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.619s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.181s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
    hip_hardware_test.go:94: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[236758 39905]
    hip_hardware_test.go:95: Gemma4 q4 package Prefill/Decode prompt=[0] next=150930 decoded=1865 text="ology"
    hip_hardware_test.go:96: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (56.16s)
PASS
ok dappco.re/go/rocm 56.187s
```

The wrapper now preserves partial batch observability for q4 mixed batches: successful prompt tokens still contribute to metrics, and the first per-prompt failure is visible through the public `Err()` slot. Production decode, prefill, and KV cache backing remain deliberately labelled `not_linked`.

Follow-up verification after extending the RX 7800 XT q4 public smoke to assert that an invalid `text:` batch prompt returns a per-prompt error and records that error in the public `Err()` slot:

```text
go test ./go -run 'TestNativeContract_BatchGenerateRecordsPerPromptError|TestNativeDecodeSmokeKernelStatus_Good' -count=1 -v
--- SKIP: TestNativeDecodeSmokeKernelStatus_Good (0.00s)
--- PASS: TestNativeContract_BatchGenerateRecordsPerPromptError_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.006s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
    hip_hardware_test.go:94: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[236758 39905]
    hip_hardware_test.go:95: Gemma4 q4 package Prefill/Decode prompt=[0] next=150930 decoded=1865 text="ology"
    hip_hardware_test.go:96: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (55.91s)
PASS
ok dappco.re/go/rocm 55.935s

go test ./go -count=1
ok dappco.re/go/rocm 0.146s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.108s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check
(no output)

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.647s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.186s
```

The live q4 smoke now covers the partial batch failure path without launching a second successful generation: `text:` fails before token execution, returns a per-result error, and confirms the public wrapper exposes it through `Err()` before the following Chat call clears stale state.

Follow-up verification after carrying the same partial-batch error recording through `ScheduledModel.BatchGenerate`, so scheduler-wrapped models expose per-prompt batch failures in the scheduler `Err()` slot even when the wrapped model returns a nil top-level error:

```text
go test ./go -run 'TestScheduler_Bad_BatchGenerateRecordsPerPromptErr|TestScheduler_Bad_NonStreamingDelegatesRecordErr|TestNativeContract_BatchGenerateRecordsPerPromptError' -count=1 -v
--- PASS: TestNativeContract_BatchGenerateRecordsPerPromptError_Bad (0.00s)
--- PASS: TestScheduler_Bad_NonStreamingDelegatesRecordErr (0.00s)
--- PASS: TestScheduler_Bad_BatchGenerateRecordsPerPromptErr (0.00s)
PASS
ok dappco.re/go/rocm 0.008s

go test ./go -count=1
ok dappco.re/go/rocm 0.143s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check
(no output)

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.638s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.180s
```

This keeps scheduler/cancellation wrapper observability aligned with the public ROCm model wrapper and the q4 mixed-batch behavior.

Follow-up verification after making `ScheduledModel.Classify` and `ScheduledModel.BatchGenerate` reject already-cancelled contexts before delegating to the wrapped model:

```text
go test ./go -run 'TestScheduler_Bad_NonStreamingDelegatesPreferCancelledContext|TestScheduler_Bad_BatchGenerateRecordsPerPromptErr|TestScheduler_Good_NonStreamingDelegatesClearSchedulerErr' -count=1 -v
--- PASS: TestScheduler_Good_NonStreamingDelegatesClearSchedulerErr (0.00s)
--- PASS: TestScheduler_Bad_NonStreamingDelegatesPreferCancelledContext (0.00s)
--- PASS: TestScheduler_Bad_BatchGenerateRecordsPerPromptErr (0.00s)
PASS
ok dappco.re/go/rocm 0.008s

go test ./go -count=1
ok dappco.re/go/rocm 0.144s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.109s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check
(no output)

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.621s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.181s
```

This closes a scheduler/cancellation boundary gap for the direct non-streaming delegates: they now match Generate/Chat behavior by recording `context canceled` in the scheduler `Err()` slot without dispatching work to the wrapped model.

Follow-up verification after cloning non-streaming delegate results at the scheduler boundary:

```text
go test ./go -run 'TestScheduler_Good_NonStreamingDelegateResultsCloned|TestScheduler_Bad_BatchGenerateRecordsPerPromptErr|TestNativeContract_BatchGenerateResultsClonedAtPublicBoundary' -count=1 -v
--- PASS: TestNativeContract_BatchGenerateResultsClonedAtPublicBoundary_Good (0.00s)
--- PASS: TestScheduler_Good_NonStreamingDelegateResultsCloned (0.00s)
--- PASS: TestScheduler_Bad_BatchGenerateRecordsPerPromptErr (0.00s)
PASS
ok dappco.re/go/rocm 0.008s

go test ./go -count=1
ok dappco.re/go/rocm 0.145s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.108s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check
(no output)

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.617s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.179s
```

`ScheduledModel.Classify` now clones returned logits and `ScheduledModel.BatchGenerate` clones returned token slices before recording errors or returning to callers, keeping scheduler-wrapped models isolated from mutable native storage.

Follow-up verification after cloning public non-streaming inputs before dispatch to native or wrapped models:

```text
go test ./go -run 'TestNativeContract_NonStreamingPromptInputsClonedAtNativeBoundary|TestNativeContract_ChatMessagesClonedAtNativeBoundary|TestScheduler_Good_NonStreamingDelegateInputsCloned|TestScheduler_Good_NonStreamingDelegateResultsCloned' -count=1 -v
--- PASS: TestNativeContract_NonStreamingPromptInputsClonedAtNativeBoundary_Good (0.00s)
--- PASS: TestNativeContract_ChatMessagesClonedAtNativeBoundary_Good (0.00s)
--- PASS: TestScheduler_Good_NonStreamingDelegateResultsCloned (0.00s)
--- PASS: TestScheduler_Good_NonStreamingDelegateInputsCloned (0.00s)
PASS
ok dappco.re/go/rocm 0.011s

go test ./go -count=1
ok dappco.re/go/rocm 0.142s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.108s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check
(no output)

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.625s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.184s
```

`rocmModel.Chat`, `rocmModel.Classify`, `rocmModel.BatchGenerate`, `ScheduledModel.Classify`, and `ScheduledModel.BatchGenerate` now hand cloned message/prompt slices to native or wrapped implementations, so downstream mutation cannot alter caller-owned request data.

Follow-up verification after separating the generation config used by wrapper-side stop handling from the config passed to native implementations:

```text
go test ./go -run 'TestNativeContract_GenerateStopSequencesSurviveNativeConfigMutation|TestNativeContract_BatchGenerateStopSequencesSurviveNativeConfigMutation|TestNativeContract_GenerateEnforcesStopSequencesAcrossChunks|TestNativeContract_BatchGenerateEnforcesStopSequencesAcrossChunks' -count=1 -v
--- PASS: TestNativeContract_GenerateEnforcesStopSequencesAcrossChunks_Good (0.00s)
--- PASS: TestNativeContract_GenerateStopSequencesSurviveNativeConfigMutation_Good (0.00s)
--- PASS: TestNativeContract_BatchGenerateEnforcesStopSequencesAcrossChunks_Good (0.00s)
--- PASS: TestNativeContract_BatchGenerateStopSequencesSurviveNativeConfigMutation_Good (0.00s)
PASS
ok dappco.re/go/rocm 0.012s

go test ./go -count=1
ok dappco.re/go/rocm 0.146s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.109s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

git diff --check
(no output)

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.625s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.183s
```

`rocmModel.Generate`, `Chat`, `Classify`, and `BatchGenerate` now clone `GenerateConfig` slice fields before dispatch. Native mutation of stop tokens or stop sequences cannot corrupt wrapper-owned stop enforcement for streaming or batch generation.

RX 7800 XT q4 smoke recheck after the wrapper input/config isolation changes:

```text
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=2 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
    hip_hardware_test.go:94: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[236758 39905]
    hip_hardware_test.go:95: Gemma4 q4 package Prefill/Decode prompt=[0] next=150930 decoded=1865 text="ology"
    hip_hardware_test.go:96: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842 110288] text=["ాత" "ði"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (55.48s)
PASS
ok dappco.re/go/rocm 55.503s
```

Follow-up verification after cloning direct `Schedule` request messages and sampler slice fields at enqueue time in both the ROCm scheduler wrapper and the shared `go-inference/scheduler` package:

```text
go test ./go -run 'TestScheduler_Good_ClonesQueuedRequestMessagesAndSampler|TestScheduler_Good_ClonesRequestLabels' -count=1 -v
--- PASS: TestScheduler_Good_ClonesRequestLabels (0.00s)
--- PASS: TestScheduler_Good_ClonesQueuedRequestMessagesAndSampler (0.00s)
PASS
ok dappco.re/go/rocm 0.006s

go test ./external/go-inference/go/scheduler -run 'TestModel_ClonesQueuedRequestMessagesAndSampler|TestModel_NormalizesRequestIDAndClonesLabels' -count=1 -v
--- PASS: TestModel_NormalizesRequestIDAndClonesLabels_Good (0.00s)
--- PASS: TestModel_ClonesQueuedRequestMessagesAndSampler_Good (0.00s)
PASS
ok dappco.re/go/inference/scheduler 0.002s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/scheduler 0.052s

go test ./go -count=1
ok dappco.re/go/rocm 0.146s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.012s

git diff --check
(no output)

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.623s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.189s
```

Direct scheduled requests now preserve caller-owned messages, stop tokens, and stop sequences even while queued behind another request. This keeps ROCm scheduler behavior aligned with the shared scheduler contract.

Follow-up verification after aligning shared `go-inference/scheduler` non-streaming delegates with the ROCm scheduler wrapper for cancelled contexts, input/result cloning, and per-prompt batch errors:

```text
go test ./external/go-inference/go/scheduler -run 'TestModel_NonStreamingDelegates(CloneInputsAndResults|PreferCancelledContext)|TestModel_BatchGenerateRecordsPerPromptErr|TestModel_NonStreamingDelegatesRecordErr' -count=1 -v
--- PASS: TestModel_NonStreamingDelegatesCloneInputsAndResults_Good (0.00s)
--- PASS: TestModel_NonStreamingDelegatesRecordErr_Bad (0.00s)
--- PASS: TestModel_NonStreamingDelegatesPreferCancelledContext_Bad (0.00s)
--- PASS: TestModel_BatchGenerateRecordsPerPromptErr_Bad (0.00s)
PASS
ok dappco.re/go/inference/scheduler 0.002s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/scheduler 0.052s

go test ./go -count=1
ok dappco.re/go/rocm 0.145s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.114s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check
(no output)

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.643s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.186s
```

The shared scheduler now rejects cancelled non-streaming delegate contexts before dispatch, clones prompt/result slices at the public boundary, and records per-result batch errors in `Err()` just like the ROCm wrapper.

Follow-up verification after making `rocmModel.Classify` and `rocmModel.BatchGenerate` record nil-native failures in `Err()`, matching Generate/Chat/Embed/Rerank behavior:

```text
go test ./go -run 'TestNativeContract_NonStreamingNilNativeRecordsErr|TestNativeContract_TextBatchPreflightRejectsEmptyPrompts' -count=1 -v
--- PASS: TestNativeContract_TextBatchPreflightRejectsEmptyPrompts_Bad (0.00s)
--- PASS: TestNativeContract_NonStreamingNilNativeRecordsErr_Bad (0.00s)
PASS
ok dappco.re/go/rocm 0.006s

go test ./go -count=1
ok dappco.re/go/rocm 0.145s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

git diff --check
(no output)

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.622s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.183s
```

The non-streaming public wrappers now preserve the same last-error semantics across validation, cancellation, native failures, per-result batch failures, and closed/nil native state.

Follow-up verification after making scheduler non-streaming nil wrapped-model paths record `Err()` consistently in both ROCm and shared `go-inference/scheduler`:

```text
go test ./go -run 'TestScheduler_Bad_NilWrappedModelRecordsErr|TestScheduler_Bad_NonStreamingDelegatesRecordErr|TestScheduler_Bad_NonStreamingDelegatesPreferCancelledContext|TestScheduler_Bad_BatchGenerateRecordsPerPromptErr' -count=1 -v
--- PASS: TestScheduler_Bad_NonStreamingDelegatesRecordErr (0.00s)
--- PASS: TestScheduler_Bad_NonStreamingDelegatesPreferCancelledContext (0.00s)
--- PASS: TestScheduler_Bad_BatchGenerateRecordsPerPromptErr (0.00s)
--- PASS: TestScheduler_Bad_NilWrappedModelRecordsErr (0.00s)
PASS
ok dappco.re/go/rocm 0.009s

go test ./external/go-inference/go/scheduler -run 'TestModel_NilAndErrorPaths|TestModel_NonStreamingDelegatesRecordErr|TestModel_NonStreamingDelegatesPreferCancelledContext|TestModel_BatchGenerateRecordsPerPromptErr' -count=1 -v
--- PASS: TestModel_NonStreamingDelegatesRecordErr_Bad (0.00s)
--- PASS: TestModel_NonStreamingDelegatesPreferCancelledContext_Bad (0.00s)
--- PASS: TestModel_BatchGenerateRecordsPerPromptErr_Bad (0.00s)
--- PASS: TestModel_NilAndErrorPaths_Bad (0.00s)
PASS
ok dappco.re/go/inference/scheduler 0.002s

go test ./go -count=1
ok dappco.re/go/rocm 0.146s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.109s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/scheduler 0.052s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.635s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.184s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Latest follow-up at EOF after Gemma4 q4 BOS parity:

- Used `go-mlx` dev as the e2e reference and matched its BOS behavior for text tokenization.
- ROCm loaded tokenizers now record `<bos>` and prepend it for normal text prompts unless the prompt already starts with `<bos>`.
- Updated fake-device BOS coverage and the live Gemma4 q4 tokenizer smoke to expect `Hello world -> [2 9259 1902]`.
- Re-ran live Gemma4 q4 on the RX 7800 XT with `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85`.
- `text:Hi` now uses prompt tokens `[2 10979]` and generated decoded tokens `[236764 3307]` with text `["," "my"]`.
- `text:Hello, my name is` now uses prompt tokens `[2 9259 236764 1041 1463 563]` and generated decoded tokens `[870 11069 1567 1604]` with text `[" [" "Your" "Name" "],"]`.
- This removes the no-BOS prompt mismatch and moves live q4 output into normal decoded tokenizer text. The path remains experimental and slow because parts of Gemma4 q4 are still host-side or smoke-kernel based.

Verification:

```text
go test ./go -run 'TestHIPGemma4Q4(Layer0|PackagePrefillDecode|SharedKV|PerLayerInputPrecompute)_Good|TestHIPGemma4Q4Layer0_Bad' -count=1 -v
PASS
ok dappco.re/go/rocm 0.009s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' go test ./go -run 'TestNativeDecodeSmokeKernelStatus_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 74.534s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hello, my name is' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=4 go test ./go -run 'TestNativeDecodeSmokeKernelStatus_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 136.436s

go test ./go -count=1
ok dappco.re/go/rocm 0.149s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.637s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.184s

(external/go-inference/go) go test ./... -count=1
PASS

git diff --check
(no output)

(external/go-inference) git diff --check
(no output)
```

## 2026-05-13 Coverage, Docs, And Benchmark Audit

- Added `codecov.yml` with a strict 90% project target and `0%` threshold.
- Added deterministic non-hardware coverage for ROCm model accessors, wire
  handlers, discovery, native helper branches, HIP token-text decoding, GGUF
  `skipValue`/`discardBytes`, embedding/rerank wrappers, and capability/helper
  contracts.
- Documented current feature surface in `doc/README.md`.
- Documented the RX 7800 XT Gemma4 q4 inference benchmark and optimization
  candidates in `doc/inference-benchmark.md`.
- Default Codecov-filtered coverage now reports:

```text
total: (statements) 90.6%
```

- Live model evidence already collected on the discrete RX 7800 XT:
  - BF16 Gemma4-E2B smoke: `Gemma4 BF16 layer0 tied LM-head greedy token=158750 score=28.510628`.
  - q4 Gemma4-E2B smoke: `text:Hi` prompt tokens `[2 10979]`, generated tokens `[236764 3307]`, text `["," "my"]`.
  - q4 benchmark: `4254553228 ns/op`, `0.2350 tok/s`, `349845704 B/op`, `83394 allocs/op`.
- Main optimization findings: cache HSACO module/function handles, reduce
  `copyTensorToDevice` load allocations, cache tokenizer decoding artifacts,
  keep q4 intermediate/KV buffers device-resident across Generate calls, and
  rerun longer-token benchmarks after startup costs are amortized.

Verification:

```text
go test ./go/... -coverpkg=./go/... -coverprofile=/tmp/go-rocm-final.cover -covermode=atomic -count=1
PASS

go tool cover -func=/tmp/go-rocm-codecov-final.cover | tail -n 1
total: (statements) 90.6%

go test ./go/... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test ./... -count=1
ok dappco.re/go/rocm/workspace

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go/... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf

go test -tags rocm_legacy_server ./go/... -count=1
ok dappco.re/go/rocm
ok dappco.re/go/rocm/internal/gguf
ok dappco.re/go/rocm/internal/llamacpp

git diff --check
(no output)

git -C external/go-inference diff --check
(no output)
```

Final completion audit after real macOS gate and live hardware refresh (2026-05-13 02:03 UTC):

Objective restated:

- Continue `~/Code/core/go-rocm/GOAL.md` until `go-rocm` is the ROCm sibling of `go-mlx`, using Gemma4-E2B as the live model evidence.
- Keep `go-rocm` independent of `go-mlx`, but use the local `go-mlx` `dev` branch as a reference when needed.
- Run the no-hardware, Linux safe-build, legacy, real macOS/no-ROCm, and ROCm hardware gates.
- Use the real RX 7800 XT, not the onboard GPU, for live ROCm checks.

Prompt-to-artifact checklist:

- `~/Code/core/go-rocm/GOAL.md`: implementation lives in the dirty `go-rocm` working tree and notes are tracked here in `GOAL_NOTES.md`.
- Gemma4-E2B model requirement: local packs exist at `/data/lem/models/gemma4/LEM-Gemma4-E2B` (8.7G) and `/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit` (2.5G), so no Hugging Face download was needed.
- Real GPU requirement: `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 rocminfo` resolves to `AMD Radeon RX 7800 XT`, `gfx1100`; the onboard CPU/iGPU device is not selected.
- Kernel artifact: `/tmp/go-rocm-kernels-gfx1100.hsaco` exists and is used by hardware gates.
- Sibling `go-mlx` reference: `/home/claude/Code/core/go-mlx` is on `dev`, `HEAD=c95ae46`, matching `origin/dev=c95ae46`.
- Core sibling setup: `/home/claude/Code/core/go-ai`, `/home/claude/Code/core/go-ml`, `/home/claude/Code/core/go-inference`, `/home/claude/Code/core/go-mlx`, and `/home/claude/Code/core/go-rocm` are all on `dev` and match `origin/dev` after fetch.
- Core repo origin URLs: `go-ai`, `go-ml`, and `go-inference` use `https://forge.lthn.sh/core/go-*.git`.
- go-rocm submodules: `external/go`, `external/go-inference`, and `external/go-log` are on `dev` and match their `origin/dev` refs after fetch.
- No `go-mlx` import boundary: `go/import_boundary_test.go` is present and the full `go test ./go/...` gate passes.
- Shared contract synchronization: `external/go-inference/go` full test suite passes.
- Capability parity and deliberate status reporting: `go/native_contract_test.go`, `go/hip_kernels_test.go`, docs, and the passing focused/full gates cover supported/experimental/planned capability status, including explicit production `not_linked` labels where kernels are not production-ready.
- Model-pack inspection: `go/model_pack.go` plus `TestNativeContract_ModelPackInspector*` cover GGUF/safetensors, Gemma4 aliases and nested text metadata, quant metadata, BERT/rerank/classifier hints, malformed packs, and memory planning.
- Scheduler/cancellation: `go/scheduler.go`, scheduler examples/tests, and `TestScheduler*` pass.
- Parser registry: `go/parser_registry.go`, parser examples/tests, and `TestParserRegistry*` pass, including Gemma4 turn-marker coverage recorded earlier.
- Cache/state: `go/cache.go`, `go/kv_cache.go`, `go/state_session.go`, `go/state_bundle.go`, examples/tests, and the refreshed cache hardware gate pass.
- HIP stepping stones: HIP driver, launch, kernel, projection, transformer, LoRA, MoE, codebook, q4, and tiny/small decode tests pass; production model-family decode/prefill remains deliberately reported as not linked outside the experimental Gemma4 q4 route.
- Bench/eval/probes: `TestNativeContract_Benchmark*`, `TestNativeContract_Evaluate*`, probe/reference tests, and docs cover the shared report fields and q4 experimental route labels.
- Docs: `README.md`, `docs/architecture.md`, `docs/development.md`, and `docs/history.md` describe current supported/experimental/planned status, Mac stub behavior, Linux/Darwin gates, Gemma4 BF16/q4 evidence, and production not-linked caveats.
- Good/Bad/Ugly coverage: new public surfaces are covered through package-local `*_test.go` and `*_example_test.go`; full `go test ./go/...`, Linux no-cgo, legacy, root workspace, external contract, and macOS gates pass.

Fresh verification:

```text
ssh -o BatchMode=yes m3 'cd ~/tmp/go-rocm-macgate && go env GOWORK && go test ./go/... -count=1'
/Users/claude/tmp/go-rocm-macgate/go.work
ok dappco.re/go/rocm 0.355s
ok dappco.re/go/rocm/internal/gguf 0.640s

go test ./go/... -count=1
ok dappco.re/go/rocm 0.144s
ok dappco.re/go/rocm/internal/gguf 0.002s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go/... -count=1
ok dappco.re/go/rocm 0.113s
ok dappco.re/go/rocm/internal/gguf 0.002s

go test -tags rocm_legacy_server ./go/... -count=1
ok dappco.re/go/rocm 0.011s
ok dappco.re/go/rocm/internal/gguf 0.002s
ok dappco.re/go/rocm/internal/llamacpp 0.004s

go test ./go -count=1
ok dappco.re/go/rocm 0.150s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.645s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.184s

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test -exec=/bin/true ./go/... -count=1
ok dappco.re/go/rocm 0.000s
ok dappco.re/go/rocm/internal/gguf 0.000s

GOOS=darwin GOARCH=amd64 CGO_ENABLED=0 go test -exec=/bin/true ./go/... -count=1
ok dappco.re/go/rocm 0.000s
ok dappco.re/go/rocm/internal/gguf 0.000s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference ...
ok dappco.re/go/inference/state/filestore 0.003s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'TestHIP|TestNative' -count=1 -v
PASS
ok dappco.re/go/rocm 0.251s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
hip_hardware_test.go:91: Gemma4 BF16 layer0 tied LM-head greedy token=158750 score=28.510628
PASS
ok dappco.re/go/rocm 17.317s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' go test ./go -run 'Test.*Smoke|Test.*Generate|Test.*Decode' -count=1 -v
hip_hardware_test.go:95: Gemma4 q4 package Prefill/Decode prompt=[0] next=258073 decoded=258073 text="<unused2161>"
hip_hardware_test.go:96: Gemma4 q4 public Generate prompt="text:Hi" prompt_tokens=[2 10979] generated tokens=[236764 3307] text=["," "my"]
PASS
ok dappco.re/go/rocm 72.956s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_CACHE_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'Test.*KV|Test.*Cache' -count=1 -v
PASS
ok dappco.re/go/rocm 0.067s

git diff --check
(no output)

git -C external/go-inference diff --check
(no output)
```

Audit conclusion:

- The previous blocker, real developer-Mac/no-ROCm execution, is closed by the `m3` Darwin arm64 run.
- The live model requirement is closed by BF16 and q4 Gemma4-E2B gates on the pinned RX 7800 XT.
- Remaining production decode/prefill/training limitations are not hidden; they are deliberately reported as `planned`/`not_linked` as required by `GOAL.md`.
- No uncovered requirement remains in the objective or `GOAL.md` definition of done.

Latest final-gate refresh after stub/docs hardening (2026-05-13 01:58 UTC):

- `go-mlx` local checkout is on `dev`; no additional reference copy was needed for this pass.
- Non-Linux/non-amd64 stub code now registers an unavailable `rocm` backend, keeps `ROCmAvailable()` false, and returns a platform-unavailable `LoadModel` error. README, architecture, development, and history docs call out that this protects developer-Mac blank imports while still making the native ROCm runtime absence explicit.
- The real macOS/no-ROCm execution gate is still not run here. Linux-hosted Darwin `-exec=/bin/true` checks prove the selected file set compiles for Darwin; they do not execute the stub tests on a developer Mac.

Verification:

```text
go test ./go -count=1
ok dappco.re/go/rocm 0.150s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.645s

GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test -exec=/bin/true ./go/... -count=1
ok dappco.re/go/rocm 0.000s
ok dappco.re/go/rocm/internal/gguf 0.000s

GOOS=darwin GOARCH=amd64 CGO_ENABLED=0 go test -exec=/bin/true ./go/... -count=1
ok dappco.re/go/rocm 0.000s
ok dappco.re/go/rocm/internal/gguf 0.000s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.184s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference 0.009s
ok dappco.re/go/inference/anthropic 0.002s
ok dappco.re/go/inference/bench 0.002s
ok dappco.re/go/inference/decode 0.002s
ok dappco.re/go/inference/eval 0.002s
ok dappco.re/go/inference/ollama 0.002s
ok dappco.re/go/inference/openai 0.003s
ok dappco.re/go/inference/parser 0.002s
ok dappco.re/go/inference/quant/codebook 0.002s
ok dappco.re/go/inference/quant/jang 0.002s
ok dappco.re/go/inference/scheduler 0.053s
ok dappco.re/go/inference/state 0.002s
ok dappco.re/go/inference/state/filestore 0.003s

git diff --check
(no output)

git -C external/go-inference diff --check
(no output)

git diff --no-index --check /dev/null GOAL_NOTES.md
(no whitespace diagnostics; expected nonzero diff comparison treated as clean)
```

Completion status remains blocked only on the true developer-Mac command:

```sh
go test ./go/... -count=1
```

Latest completion audit after refreshing required gates on 2026-05-13:

- Objective restated as deliverables: make `go-rocm` a package-first ROCm sibling to `go-mlx` without importing `go-mlx`; keep legacy server code behind `rocm_legacy_server`; expose/label every shared `go-inference` capability accurately; provide package-first scheduler, cancellation, cache, parser, state, bench, eval, probe, model-pack, and HIP stepping-stone surfaces; keep not-yet-production decode/prefill paths explicit; prove Linux safe builds, Linux cgo runtime status, opt-in ROCm hardware gates, and Mac/no-ROCm behavior.
- Prompt-to-artifact checklist:
  - `GOAL.md` / `RFC.md` read: confirmed current contract and RFC target matrix in this audit.
  - `AGENTS.md` read: repo uses Core `go/` subtree layout with workspace metadata at repo root.
  - No `go-mlx` import: `go/import_boundary_test.go` covers forbidden runtime imports and `rg` only finds allowed docs/tests/labels such as `mlx_q4`, not a runtime import.
  - Shared contracts: `external/go-inference/go/capability.go` and `contracts.go` contain the expanded capability IDs and optional scheduler/cache/parser/state/model-pack/embedding/rerank interfaces; `go test ./... -count=1` in `external/go-inference/go` passed.
  - Package-first ROCm APIs: current `go test -list` output shows scheduler, cache, parser, state, bench/eval/probe, compatibility handler, decode helper, model-pack, and native capability tests/examples in `./go`.
  - Non-hardware required gates: `go test ./... -count=1`, `GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1`, and `go test -tags rocm_legacy_server ./... -count=1` all passed from repo root.
  - Darwin/no-ROCm proxy: `GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test -exec=/bin/true ./... -count=1` and `GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test -exec=/bin/true ./go/... -count=1` passed, and non-Linux stubs have `go/rocm_stub_test.go` Good/Bad/Ugly coverage. This is still compile-proxy evidence only, not a real developer-Mac test execution.
  - Darwin/no-ROCm stub hardening after this audit: non-Linux/non-amd64 builds now register an unavailable `rocm` backend with `go-inference`, keep `ROCmAvailable()` false, and return a platform-unavailable `LoadModel` error. `go/rocm_stub_test.go` and `go/rocm_stub_example_test.go` cover the registered-unavailable backend path for a real Mac run; Linux-hosted verification remains `GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test -exec=/bin/true ./go/... -count=1`.
  - Darwin file-selection evidence after stub hardening: `GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go list -json ./go` and `GOOS=darwin GOARCH=amd64 CGO_ENABLED=0 go list -json ./go` both report `GoFiles` as `compat_handlers.go`, `discover.go`, `openai.go`, `rocm.go`, and `rocm_stub.go`; they ignore Linux/HIP/native files and include `rocm_stub_test.go` plus `rocm_stub_example_test.go` in `TestGoFiles`. The `darwin/amd64` compile proxy also passed with `GOOS=darwin GOARCH=amd64 CGO_ENABLED=0 go test -exec=/bin/true ./go/... -count=1`.
  - RFC focused non-hardware phase gates rerun after the stub/doc hardening: `go test ./go -run 'TestNativeContract_Rocm.*Capabilities' -count=1`, `go test ./go -run 'TestScheduler' -count=1`, `go test ./go -run 'TestCacheService' -count=1`, `go test ./go -run 'TestParserRegistry' -count=1`, `go test ./go -run 'Test.*ModelPack|TestNativeContract_ModelPack' -count=1`, `go test ./go -run 'TestStateSession' -count=1`, `go test ./go -run 'TestHIP|TestNativeContract_LoadModel' -count=1`, `go test ./go -run 'Test.*KV|Test.*Cache' -count=1`, `go test ./go -run 'Test.*MoE|Test.*JANG|Test.*Codebook' -count=1`, `go test ./go -run 'Test.*Benchmark|Test.*Evaluate|Test.*Probe' -count=1`, and `go test ./go -run 'TestImportBoundary|TestHIPKernels_NotLinked|TestHIPRuntime_DecodeKernelsNotLinked' -count=1` all passed.
  - Linux cgo/HIP status: `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'TestHIP|TestNative' -count=1` passed on the pinned RX 7800 XT.
  - Hardware skip-safety without env vars: `go test ./go -run 'TestHIPHardware|TestNativeDecodeSmokeKernelStatus|TestNativeModelPackSmokeGemma4E2B' -count=1 -v` passed with explicit skips for `GO_ROCM_RUN_HIP_TESTS`, `GO_ROCM_RUN_MODEL_TESTS`, and `GO_ROCM_RUN_CACHE_TESTS`.
  - ROCm cache hardware gate: `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_CACHE_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'Test.*KV|Test.*Cache' -count=1 -v` passed in `0.067s`, including `TestHIPHardwareKVCacheSmoke_Good`.
  - Required model hardware gate: `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' go test ./go -run 'Test.*Smoke|Test.*Generate|Test.*Decode' -count=1 -v` passed in `73.088s`; live q4 public Generate used prompt tokens `[2 10979]`, generated tokens `[236764 3307]`, and decoded text `["," "my"]`.
  - BF16 Gemma4-E2B anchor: earlier 2026-05-13 evidence in this file records `TestNativeDecodeSmokeKernelStatus_Good` passing on `/data/lem/models/gemma4/LEM-Gemma4-E2B` with the layer-0 tied LM-head greedy check.
  - Clear not-linked behavior: `go/hip_kernels_test.go`, `go/hip_runtime_test.go`, `go/native_contract_test.go`, docs, and q4 capability assertions prove production decode/prefill/KV remain labelled or errored as not linked outside the experimental package-local q4 route.
  - Documentation: `README.md`, `docs/architecture.md`, `docs/development.md`, and `docs/history.md` describe current supported/experimental/planned states, hardware commands, and Gemma4-E2B q4/BF16 evidence.
- Completion result: do not mark the active goal complete. The local Linux evidence is strong and the Gemma4 q4/BF16 hardware requirements are covered, but a true macOS/no-ROCm test run has not been executed on a developer Mac. Treat the Darwin cross-build plus stub tests as useful evidence, not as final completion proof.

Continuation marker, 2026-05-13:

- Latest audit/continuation notes are recorded above at the `Latest continuation/audit after rerunning Gemma4-E2B BF16 and q4 on the pinned RX 7800 XT` section.
- The current live evidence from this turn is BF16 `TestNativeDecodeSmokeKernelStatus_Good` passing on `/data/lem/models/gemma4/LEM-Gemma4-E2B` in `17.140s`, q4 `TestNative(DecodeSmokeKernelStatus|ModelPackSmokeGemma4E2B)_Good` passing on `/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit` in `72.245s`, and the BOS-aware q4 prompt `text:Hi` producing prompt tokens `[2 10979]` and decoded generated text `["," "my"]`.
- Do not mark the broad `GOAL.md` complete yet: the Linux/ROCm and cross-build evidence is strong, but the real Mac/no-ROCm gate is only compile-proxied from Linux and production model-family prefill/decode remain intentionally `not_linked` outside the experimental Gemma4 q4 route.
- Capability honesty follow-up: `rocmGemma4Q4GenerateCapabilityLabels` now also advertises `production_prefill=not_linked`, so Generate, BatchGenerate, Chat, Benchmark, SpeculativeDecode, and PromptLookupDecode inherit the same prefill/decode/KV caveats as the package Prefill/Eval/Classify path. Contract and hardware q4 assertions were updated to require the label and the q4 capability detail now says production native prefill/decode remain pending.
- Verification after that label change: `go test ./go -run 'TestNativeContract_RocmGemma4Q4ExperimentalGenerateCapability_Good' -count=1 -v`, `go test ./go -count=1`, and the live RX 7800 XT q4 gate with `GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi'` passed in `72.792s` with public Generate still producing prompt tokens `[2 10979]`, generated tokens `[236764 3307]`, and text `["," "my"]`; `git diff --check` and `git -C external/go-inference diff --check` were clean.

Latest continuation/audit after rerunning Gemma4-E2B BF16 and q4 on the pinned RX 7800 XT:

- Objective restated as deliverables: continue `GOAL.md` toward package-first `go-rocm`/`go-mlx` parity, use local Gemma4-E2B packs for hardware evidence, avoid importing `go-mlx`, keep no-hardware/cross-build gates green, and keep production decode/prefill status honest where kernels are not linked.
- Gemma4-E2B assets are already present locally, so no Hugging Face download was needed:
  - BF16 anchor: `/data/lem/models/gemma4/LEM-Gemma4-E2B`
  - fast q4 loop: `/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit`
- The BF16 smoke was rerun with `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85`, `GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B`, and `/tmp/go-rocm-kernels-gfx1100.hsaco`; `TestNativeDecodeSmokeKernelStatus_Good` passed in `17.140s` and reached the layer-0 tied LM-head greedy check with token `158750`.
- The q4 model gate was rerun with `GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1` and `GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi'`; `TestNative(DecodeSmokeKernelStatus|ModelPackSmokeGemma4E2B)_Good` passed in `72.245s`.
- Current q4 live output after BOS parity:
  - package Prefill/Decode: `prompt=[0] next=258073 decoded=258073 text="<unused2161>"`
  - public Generate: `prompt_tokens=[2 10979] generated tokens=[236764 3307] text=["," "my"]`
- Audit evidence checked in this continuation:
  - `go test ./... -count=1` from repo root: passed.
  - `GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1` from repo root: passed.
  - `go test -tags rocm_legacy_server ./... -count=1` from repo root: passed.
  - `go test ./... -count=1` from `external/go-inference/go`: passed.
  - `GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test -exec=/bin/true ./... -count=1` from repo root and `go/`: passed as a compile-only proxy for the Mac/no-ROCm gate.
  - `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'TestHIP|TestNative' -count=1`: passed.
  - `git diff --check && git -C external/go-inference diff --check`: passed before the docs refresh.
- Completion audit result: do not mark the overall `GOAL.md` complete yet. The active code and docs now have strong evidence for the package-first API surfaces, no-hardware gates, BF16 primitive anchor, and q4 live route, but the true Mac runtime gate is only compile-proxied from Linux and production model-family prefill/decode remain intentionally reported as `not_linked` outside the experimental Gemma4 q4 route.

Latest Gemma4 q4 BOS parity follow-up:

- Checked the local `go-mlx` dev branch again as the e2e reference. Its tokenizer prepends `<bos>` when the tokenizer defines one.
- Updated the ROCm token-text shim to record `<bos>` from `tokenizer.json` and prepend it for normal text prompts unless the prompt already starts with the BOS text.
- Updated the Gemma4 q4 hardware smoke tokenizer assertion from `Hello world -> [9259 1902]` to `Hello world -> [2 9259 1902]`.
- Added fake-device unit coverage proving a BOS-aware decoder encodes `he` and `<bos>he` to the same `[2 12]` sequence.
- Re-ran live Gemma4 q4 on the RX 7800 XT using `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85`; device 0 remains avoided.
- BOS changed the public `text:Hi` prompt from `[10979]` to `[2 10979]` and moved output from fallback/control-ish tokens into decoded tokenizer text:
  - `text:Hi`: generated tokens `[236764 3307]`, text `["," "my"]`
  - `text:Hello, my name is`: prompt tokens `[2 9259 236764 1041 1463 563]`, generated tokens `[870 11069 1567 1604]`, text `[" [" "Your" "Name" "],"]`
- The q4 path is now a real live-model path with normal decoded output, but still experimental and slow. Remaining quality/perf work is likely kernel fusion/host-side GELU removal and deeper parity checks, not the old missing-BOS prompt issue.

Verification:

```text
go test ./go -run 'TestHIPGemma4Q4(Layer0|PackagePrefillDecode|SharedKV|PerLayerInputPrecompute)_Good|TestHIPGemma4Q4Layer0_Bad' -count=1 -v
PASS
ok dappco.re/go/rocm 0.009s

go test ./go -run 'TestNativeContract.*Tokenizer|TestNative.*Tokenizer|TestHIPKernels_MLXQ4ProjectionLaunchArgs_Good' -count=1
ok dappco.re/go/rocm 0.008s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' \
go test ./go -run 'TestNativeDecodeSmokeKernelStatus_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 74.534s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hello, my name is' \
GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=4 \
go test ./go -run 'TestNativeDecodeSmokeKernelStatus_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 136.436s

go test ./go -count=1
ok dappco.re/go/rocm 0.149s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.637s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.184s

(external/go-inference/go) go test ./... -count=1
ok dappco.re/go/inference 0.008s
ok dappco.re/go/inference/anthropic 0.002s
ok dappco.re/go/inference/bench 0.002s
ok dappco.re/go/inference/decode 0.010s
ok dappco.re/go/inference/eval 0.010s
ok dappco.re/go/inference/ollama 0.010s
ok dappco.re/go/inference/openai 0.010s
ok dappco.re/go/inference/parser 0.010s
ok dappco.re/go/inference/quant/codebook 0.010s
ok dappco.re/go/inference/quant/jang 0.010s
ok dappco.re/go/inference/scheduler 0.053s
ok dappco.re/go/inference/state 0.010s
ok dappco.re/go/inference/state/filestore 0.012s

git diff --check
(no output)

(external/go-inference) git diff --check
(no output)
```

Latest follow-up after adding Gemma4 q4 per-layer input parity:

- Used the local `go-mlx` dev branch as a reference only. ROCm still does not import `go-mlx`.
- Added a BF16 device-weight projection path so loaded BF16 tensors can use the existing projection kernel without pretending they are FP16.
- Added Gemma4 q4 per-layer input loading for:
  - `language_model.model.embed_tokens_per_layer.{weight,scales,biases}`
  - `language_model.model.per_layer_model_projection.weight`
  - `language_model.model.per_layer_projection_norm.weight`
  - `language_model.model.layers.N.per_layer_input_gate.*`
  - `language_model.model.layers.N.per_layer_projection.*`
  - `language_model.model.layers.N.post_per_layer_input_norm.weight`
- Wired single-token forward to precompute the per-layer input vectors once per token, split them per layer, and feed each decoder layer's gate/projection path before `layer_scalar`.
- The per-layer GELU and elementwise multiply are currently host-side in the experimental q4 path; projection, RMSNorm, embedding, vector scale/add, attention, and sampling still use HIP primitives.
- Added fake-device coverage for direct per-layer decoder application and for global per-layer precompute labels/launches.
- Re-ran live q4 on the RX 7800 XT. Per-layer input tensors are now loaded and affect logits:
  - package Prefill/Decode: `prompt=[0] next=855 decoded=8704 text=" Den"`
  - public Generate `text:Hi`: `prompt_tokens=[10979] generated tokens=[240709 35843] text=["ሻ" " mó"]`
- Output is still not coherent chat. The remaining likely Gemma4 parity gap is shared/global KV layout or another architecture-specific attention detail, not the now-linked per-layer input path.
- Do not combine `GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1` with the broad `TestHIP|TestNative` unit sweep; it intentionally changes default text-prompt mode and trips `TestHIPGemma4Q4Layer0_Good`. Keep HIP hardware and q4 model gates separate.

Verification:

```text
go test ./go -run 'TestHIPGemma4Q4(Layer0|PerLayerInputPrecompute)_Good|TestHIPGemma4Q4Layer0_Bad' -count=1 -v
PASS
ok dappco.re/go/rocm 0.007s

go test ./go -count=1
ok dappco.re/go/rocm 0.146s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.112s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' \
go test ./go -run 'TestNative(DecodeSmokeKernelStatus|ModelPackSmokeGemma4E2B)_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 53.720s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' \
go test ./go -run 'Test.*Smoke|Test.*Generate|Test.*Decode' -count=1 -v
PASS
ok dappco.re/go/rocm 53.729s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_HIP_TESTS=1 \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
go test ./go -run 'TestHIP|TestNative' -count=1
ok dappco.re/go/rocm 0.249s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.628s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.182s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference 0.008s
ok dappco.re/go/inference/anthropic 0.002s
ok dappco.re/go/inference/bench 0.002s
ok dappco.re/go/inference/decode 0.002s
ok dappco.re/go/inference/eval 0.002s
ok dappco.re/go/inference/ollama 0.002s
ok dappco.re/go/inference/openai 0.003s
ok dappco.re/go/inference/parser 0.002s
ok dappco.re/go/inference/quant/codebook 0.002s
ok dappco.re/go/inference/quant/jang 0.002s
ok dappco.re/go/inference/scheduler 0.053s
ok dappco.re/go/inference/state 0.002s
ok dappco.re/go/inference/state/filestore 0.003s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Live q4 Gemma4-E2B hardware refresh on the RX 7800 XT after parser contract work:

```text
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
  GO_ROCM_RUN_MODEL_TESTS=1 \
  GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
  GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
  GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
  GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' \
  go test ./go -run 'TestNative(DecodeSmokeKernelStatus|ModelPackSmokeGemma4E2B)_Good' -count=1 -v

hip_hardware_test.go:95: Gemma4 q4 package Prefill/Decode prompt=[0] next=150930 decoded=1865 text="ology"
hip_hardware_test.go:96: Gemma4 q4 public Generate prompt="text:Hi" prompt_tokens=[10979] generated tokens=[18070 35506] text=["anth" "思い"]
PASS
ok dappco.re/go/rocm 45.705s
```

Follow-up after tightening model parser error ownership:

- `rocmModel.ParseReasoning` and `rocmModel.ParseTools` now clear stale public errors on entry and record parser failures in `Err()`.
- Added model-level parser examples for reasoning and tool parsing alongside the existing standalone `ParserRegistry` examples.
- The fast Gemma4 q4 live smoke was rerun on the RX 7800 XT after this contract pass; generation behavior is unchanged and still passes.

Verification:

```text
go test ./go -run 'TestParserRegistry_(Good_RocmModel(ImplementsParserContracts|UsesModelTypeFallback|ParseReasoningClearsStaleErr)|Bad_RocmModelParseToolsRecordsErrAndSuccessClears_Bad)|Example_rocmModel_Parse(Reasoning|Tools)$|ExampleParserRegistry_Parse(Reasoning|Tools)$' -count=1 -v
PASS
ok dappco.re/go/rocm 0.012s

go test ./go -count=1
ok dappco.re/go/rocm 0.149s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.769s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.178s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference 0.009s
ok dappco.re/go/inference/scheduler 0.054s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Latest follow-up:

- Pulled the current sibling `go-mlx` dev reference from `https://forge.lthn.sh/core/go-mlx.git` to `c95ae46`; its Gemma4 text runtime uses `gemma4_text` for `Gemma4ForCausalLM`/`Gemma4TextForCausalLM`.
- Mirrored that ROCm-side without importing `go-mlx`: standalone Gemma4 text packs now inspect as `gemma4_text`, while conditional-generation/nested multimodal Gemma4 packs remain `gemma4`.
- Updated Gemma4 q4 runtime gates so `gemma4_text` still reaches text prompt generation, q4 package prefill/decode, layer0, small-decode, and opt-in hardware smoke helpers.
- Hardened ROCm `Benchmark`/`Evaluate` `Err()` ownership: returned failures are recorded, successful wrapper calls clear stale/internal best-effort errors, and eval stream/empty-dataset failures are now observable through `Err()`.

Latest verification:

```text
go test ./go -run 'TestNativeContract_(BenchmarkBad_PropagatesCacheStatsError|EvaluateQualityProbes_Bad_RecordsUnavailableGeneration|EvaluateSuccessClearsLastError_Good|EvaluateBadRecordsFailure|EvaluateBadRejectsEmptyDataset)' -count=1 -v
PASS
ok dappco.re/go/rocm 0.002s

go test ./go -run 'TestNativeContract_(ModelPackInspectorArchitectureFixtures_Good|GeneratedPromptUsesExplicitGemma4Q4TextMode_Good)|TestHIPGemma4Q4Layer0_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 0.010s

go test ./go -count=1
ok dappco.re/go/rocm 0.143s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/scheduler 0.054s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.767s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.176s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Follow-up work after pulling the current `go-mlx` dev branch reference:

- Updated sibling `/home/claude/Code/core/go-mlx` origin to `https://forge.lthn.sh/core/go-mlx.git` and fast-forwarded `dev` to `c95ae46`; the checkout is clean.
- ROCm `Benchmark` and `Evaluate` now own their `Err()` lifecycle like the other public model wrappers: they clear stale errors on entry, record returned failures, and clear internal best-effort probe/loss errors when the wrapper succeeds.
- `Evaluate` now has focused coverage for empty datasets, dataset stream errors, stale error clearing, and successful eval with failed quality probes keeping the failure in report labels instead of poisoning `Err()`.
- Gemma4 text-model aliases now follow the current go-mlx dev convention: `gemma4_text`, `Gemma4ForCausalLM`, and `Gemma4TextForCausalLM` normalize as text-model architecture while `Gemma4ForConditionalGeneration` remains `gemma4`.
- Gemma4 q4 runtime gates now accept both `gemma4` and `gemma4_text`, including text-prompt generation, q4 package prefill/decode, layer0 config, small-decode smoke, and opt-in hardware smoke guards.

Focused verification:

```text
go test ./go -run 'TestNativeContract_(BenchmarkBad_PropagatesCacheStatsError|EvaluateQualityProbes_Bad_RecordsUnavailableGeneration|EvaluateSuccessClearsLastError_Good|EvaluateBadRecordsFailure|EvaluateBadRejectsEmptyDataset)' -count=1 -v
PASS
ok dappco.re/go/rocm 0.002s

go test ./go -run 'TestNativeContract_(ModelPackInspectorArchitectureFixtures_Good|GeneratedPromptUsesExplicitGemma4Q4TextMode_Good)|TestHIPGemma4Q4Layer0_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 0.010s
```

Non-hardware gates after the follow-up patch:

```text
go test ./go -count=1
ok dappco.re/go/rocm 0.139s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference 0.008s
ok dappco.re/go/inference/anthropic 0.002s
ok dappco.re/go/inference/bench 0.002s
ok dappco.re/go/inference/decode 0.002s
ok dappco.re/go/inference/eval 0.002s
ok dappco.re/go/inference/ollama 0.002s
ok dappco.re/go/inference/openai 0.003s
ok dappco.re/go/inference/parser 0.003s
ok dappco.re/go/inference/quant/codebook 0.002s
ok dappco.re/go/inference/quant/jang 0.002s
ok dappco.re/go/inference/scheduler 0.054s
ok dappco.re/go/inference/state 0.002s
ok dappco.re/go/inference/state/filestore 0.003s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.767s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.176s

git diff --check && git -C external/go-inference diff --check
(no output)
```

ROCm `Evaluate` now matches the shared eval contract by rejecting an empty
dataset instead of returning a zero-sample report.

Verification:

```text
go test ./go -run 'TestNativeContract_EvaluateBadRejectsEmptyDataset|TestNativeContract_BenchmarkAndEvaluateUseModelSurface_Ugly|TestNativeContract_EvaluateQualityProbes_Good' -count=1 -v
--- PASS: TestNativeContract_BenchmarkAndEvaluateUseModelSurface_Ugly (0.00s)
--- PASS: TestNativeContract_EvaluateQualityProbes_Good (0.00s)
--- PASS: TestNativeContract_EvaluateBadRejectsEmptyDataset (0.00s)
PASS
ok dappco.re/go/rocm 0.009s

go test ./go -count=1
ok dappco.re/go/rocm 0.148s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.112s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/eval 0.003s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.646s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.183s

git diff --check && git -C external/go-inference diff --check
(no output)
```

The shared eval package now has local Good/Bad/Ugly tests for weighted
perplexity aggregation, adapter loading, quality probes, dataset/runner error
paths, invalid metrics, and cancelled context handling. Its public errors are
now backend-neutral `eval:` messages instead of the previous `mlx:` prefix.

Verification:

```text
go test ./external/go-inference/go/eval -count=1 -v
--- PASS: TestRunDataset_Good_AggregatesWeightedLossAndQuality (0.00s)
--- PASS: TestRunDataset_Good_LoadsAdapterAndRefreshesInfo (0.00s)
--- PASS: TestRunDataset_Bad_UsesBackendNeutralErrors (0.00s)
--- PASS: TestRunDataset_Bad_RejectsAdapterPathWhenUnsupported (0.00s)
--- PASS: TestRunDataset_Bad_PropagatesDatasetError (0.00s)
--- PASS: TestRunDataset_Ugly_CancelledContextBeforeRead (0.00s)
--- PASS: TestRunDataset_Bad_RejectsEmptyAndInvalidMetrics (0.00s)
PASS
ok dappco.re/go/inference/eval 0.002s

rg -n "mlx: eval" external/go-inference/go/eval external/go-inference/go -g '*.go'
(no output)

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/eval 0.002s

go test ./go -count=1
ok dappco.re/go/rocm 0.143s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.660s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.184s

git diff --check && git -C external/go-inference diff --check
(no output)
```

OpenAI service embeddings/rerank validation now mirrors the ROCm native
preflight: blank embedding input items and blank rerank document items are
rejected at the HTTP boundary before optional-interface dispatch.

Verification:

```text
go test ./external/go-inference/go/openai -run 'TestOpenAI_(EmbeddingsHandler_Bad_RejectsBlankInputText|RerankHandler_Bad_RejectsBlankDocuments|EmbeddingsHandler_Good_UsesEmbeddingModel|RerankHandler_Good_UsesRerankModel)' -count=1 -v
--- PASS: TestOpenAI_EmbeddingsHandler_Good_UsesEmbeddingModel (0.00s)
--- PASS: TestOpenAI_RerankHandler_Good_UsesRerankModel (0.00s)
--- PASS: TestOpenAI_EmbeddingsHandler_Bad_RejectsBlankInputText (0.00s)
--- PASS: TestOpenAI_RerankHandler_Bad_RejectsBlankDocuments (0.00s)
PASS
ok dappco.re/go/inference/openai 0.002s

go test ./go -run 'TestOpenAI_NewOpenAIServiceMux_Bad_RejectsBlank(EmbeddingInput|RerankDocument|ChatMessages)|TestOpenAI_NewOpenAIServiceMux_Bad_GenericModelWithoutEmbeddingsAndRerank' -count=1 -v
--- PASS: TestOpenAI_NewOpenAIServiceMux_Bad_RejectsBlankChatMessages (0.00s)
--- PASS: TestOpenAI_NewOpenAIServiceMux_Bad_RejectsBlankEmbeddingInput (0.00s)
--- PASS: TestOpenAI_NewOpenAIServiceMux_Bad_RejectsBlankRerankDocument (0.00s)
--- PASS: TestOpenAI_NewOpenAIServiceMux_Bad_GenericModelWithoutEmbeddingsAndRerank (0.00s)
PASS
ok dappco.re/go/rocm 0.008s

go test ./external/go-inference/go/openai -count=1
ok dappco.re/go/inference/openai 0.002s

go test ./go -count=1
ok dappco.re/go/rocm 0.149s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.110s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/openai 0.003s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.624s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.184s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Shared OpenAI chat validation now rejects requests where every message content
field is blank, and the ROCm OpenAI service mux has boundary coverage for that
path. `ResponseGenerateOptions` now includes `instructions` in its chat-shaped
validation input even when `input` is present, so nonblank instructions can
satisfy content validation consistently with `ResponseMessages`.

Verification:

```text
go test ./external/go-inference/go/openai -run 'TestOpenAI_Handler_Bad_RejectsBlankMessageContent|TestResponses_ResponseGenerateOptions_Good_InstructionsSatisfyContentValidation' -count=1 -v
--- PASS: TestOpenAI_Handler_Bad_RejectsBlankMessageContent (0.00s)
--- PASS: TestResponses_ResponseGenerateOptions_Good_InstructionsSatisfyContentValidation (0.00s)
PASS
ok dappco.re/go/inference/openai 0.002s

go test ./go -run 'TestOpenAI_NewOpenAIServiceMux_Bad_RejectsBlankChatMessages|TestOpenAI_NewOpenAIResponsesHandler_Bad_RejectsBlankInput|TestOpenAI_NewOpenAIResponsesHandler_Good_NonStreaming' -count=1 -v
--- PASS: TestOpenAI_NewOpenAIResponsesHandler_Good_NonStreaming (0.00s)
--- PASS: TestOpenAI_NewOpenAIResponsesHandler_Bad_RejectsBlankInput (0.00s)
--- PASS: TestOpenAI_NewOpenAIServiceMux_Bad_RejectsBlankChatMessages (0.00s)
PASS
ok dappco.re/go/rocm 0.008s

go test ./go -count=1
ok dappco.re/go/rocm 0.142s

go test ./external/go-inference/go/openai -count=1
ok dappco.re/go/inference/openai 0.002s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.110s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.012s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/openai 0.003s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.627s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.182s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Follow-up verification after making direct scheduler enqueue failures update `Err()` consistently:

```text
go test ./go -run 'TestScheduler_Bad_RejectsClosedScheduler|TestScheduler_Bad_RejectsCancelledContextBeforeEnqueue|TestScheduler_Bad_RejectsDuplicateInFlightRequestID|TestScheduler_Bad_RejectsFullQueueWithoutBlocking|TestScheduler_Bad_NilWrappedModelRecordsErr' -count=1 -v
--- PASS: TestScheduler_Bad_NilWrappedModelRecordsErr (0.00s)
--- PASS: TestScheduler_Bad_RejectsClosedScheduler (0.00s)
--- PASS: TestScheduler_Bad_RejectsCancelledContextBeforeEnqueue (0.00s)
--- PASS: TestScheduler_Bad_RejectsDuplicateInFlightRequestID (0.00s)
--- PASS: TestScheduler_Bad_RejectsFullQueueWithoutBlocking (0.00s)
PASS
ok dappco.re/go/rocm 0.010s

go test ./external/go-inference/go/scheduler -run 'TestModel_RejectsFullQueue|TestModel_RejectsDuplicateInFlightRequestID|TestModel_CloseRejectsNewWorkAndIsIdempotent|TestModel_NilAndErrorPaths' -count=1 -v
--- PASS: TestModel_RejectsFullQueue_Bad (0.00s)
--- PASS: TestModel_RejectsDuplicateInFlightRequestID_Bad (0.00s)
--- PASS: TestModel_CloseRejectsNewWorkAndIsIdempotent_Bad (0.00s)
--- PASS: TestModel_NilAndErrorPaths_Bad (0.00s)
PASS
ok dappco.re/go/inference/scheduler 0.002s

go test ./go -count=1
ok dappco.re/go/rocm 0.145s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.112s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/scheduler 0.053s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.631s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.185s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Direct `Schedule` now clears stale scheduler errors on successful enqueue and records cancelled-context, closed-scheduler, duplicate-request, full-queue, nil-base, and nil/partial ROCm wrapper failures in `Err()` without relying on `Generate`/`Chat` to mirror the returned error.

Live ROCm verification on the RX 7800 XT after scheduler hardening, forcing the real discrete GPU with `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85`:

```text
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 rocminfo
Agent 2
  Name: AMD Radeon RX 7800 XT
  Uuid: GPU-880ed6479d653a85
  Name: gfx1100
  Pool 1 GLOBAL coarse-grained size: 16760832 KB

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestHIPHardwareAvailabilitySmoke_Good -count=1 -v
--- PASS: TestHIPHardwareAvailabilitySmoke_Good (0.03s)
PASS
ok dappco.re/go/rocm 0.028s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeModelPackSmokeGemma4E2B_Good -count=1 -v
--- PASS: TestNativeModelPackSmokeGemma4E2B_Good (0.23s)
PASS
ok dappco.re/go/rocm 0.234s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run 'TestHIPHardware(Kernel|Embedding|RMSNorm)' -count=1 -v
--- PASS: TestHIPHardwareEmbeddingKernelSource_Good (0.04s)
PASS
ok dappco.re/go/rocm 0.046s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
hip_hardware_test.go:91: Gemma4 BF16 layer0 tied LM-head greedy token=158750 score=28.711908
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (17.50s)
PASS
ok dappco.re/go/rocm 17.519s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='Hello world' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=1 GO_ROCM_GEMMA4_Q4_DECODE_LAYERS=1 GO_ROCM_GEMMA4_Q4_DECODE_TOKENS=2 go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
hip_hardware_test.go:94: Gemma4 q4 1-layer greedy decode prompt=[0] generated tokens=[236758 39905]
hip_hardware_test.go:95: Gemma4 q4 package Prefill/Decode prompt=[0] next=150930 decoded=1865 text="ology"
hip_hardware_test.go:96: Gemma4 q4 public Generate prompt="Hello world" prompt_tokens=[9259 1902] generated tokens=[20842] text=["ాత"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (53.65s)
PASS
ok dappco.re/go/rocm 53.677s
```

BF16 now has live loaded Gemma4-E2B smoke coverage on the RX 7800 XT for model-pack metadata plus linked embedding/projection/RMS/attention fixture kernels up through a layer-0 tied LM-head greedy check. q4 remains the faster live development path and passed package prefill/decode plus public Generate over the same discrete GPU. Production native prefill/decode labels remain intentionally `not_linked`.

Follow-up verification after aligning direct `CancelRequest` with scheduler last-error semantics in both ROCm and shared `go-inference/scheduler`:

```text
go test ./go -run 'TestScheduler_(Bad_NilWrappedModelRecordsErr|Bad_RejectsBlankCancelID|Good_DelegatesUnknownCancelToBaseModel|Bad_CancelRequestRecordsErr|Good_CancelsBeforeStart|Good_CancelsDuringDecode)' -count=1 -v
--- PASS: TestScheduler_Good_CancelsBeforeStart (0.00s)
--- PASS: TestScheduler_Good_CancelsDuringDecode (0.01s)
--- PASS: TestScheduler_Bad_NilWrappedModelRecordsErr (0.00s)
--- PASS: TestScheduler_Bad_RejectsBlankCancelID (0.00s)
--- PASS: TestScheduler_Good_DelegatesUnknownCancelToBaseModel (0.00s)
--- PASS: TestScheduler_Bad_CancelRequestRecordsErr (0.00s)
PASS
ok dappco.re/go/rocm 0.007s

go test ./external/go-inference/go/scheduler -run 'TestModel_CancelRequest_(UsesContextAndDelegatesIt|CancelsQueuedRequest)|TestModel_NilAndErrorPaths' -count=1 -v
--- PASS: TestModel_CancelRequest_CancelsQueuedRequest_Good (0.03s)
--- PASS: TestModel_CancelRequest_UsesContextAndDelegatesIt_Bad (0.00s)
--- PASS: TestModel_NilAndErrorPaths_Bad (0.00s)
PASS
ok dappco.re/go/inference/scheduler 0.028s

go test ./go -count=1
ok dappco.re/go/rocm 0.146s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.109s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/scheduler 0.052s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.630s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.187s

git diff --check && git -C external/go-inference diff --check
(no output)
```

`CancelRequest` now records cancelled contexts, ROCm blank IDs, nil/partial ROCm scheduler state, and delegate cancellation failures in `Err()`, while successful active or delegated cancellation clears stale scheduler errors.

Follow-up verification after adding explicit coverage that active queued/running cancellation clears stale scheduler errors:

```text
go test ./go -run 'TestScheduler_Good_Cancels(BeforeStart|DuringDecode)|TestScheduler_Good_DelegatesUnknownCancelToBaseModel|TestScheduler_Bad_CancelRequestRecordsErr' -count=1 -v
--- PASS: TestScheduler_Good_CancelsBeforeStart (0.00s)
--- PASS: TestScheduler_Good_CancelsDuringDecode (0.01s)
--- PASS: TestScheduler_Good_DelegatesUnknownCancelToBaseModel (0.00s)
--- PASS: TestScheduler_Bad_CancelRequestRecordsErr (0.00s)
PASS
ok dappco.re/go/rocm 0.013s

go test ./external/go-inference/go/scheduler -run 'TestModel_CancelRequest_(CancelsQueuedRequest|UsesContextAndDelegatesIt)' -count=1 -v
--- PASS: TestModel_CancelRequest_CancelsQueuedRequest_Good (0.03s)
--- PASS: TestModel_CancelRequest_UsesContextAndDelegatesIt_Bad (0.00s)
PASS
ok dappco.re/go/inference/scheduler 0.027s

go test ./go -count=1
ok dappco.re/go/rocm 0.147s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.110s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/scheduler 0.052s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.640s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.183s

git diff --check && git -C external/go-inference diff --check
(no output)
```

The scheduler cancellation contract is now covered for active queued cancellation, active decode cancellation, delegated success, delegated failure, cancelled context, and ROCm nil/blank/partial-wrapper bad paths.

Follow-up verification after tightening OpenAI-compatible Responses validation so whitespace-only `input`/`instructions` are rejected before reaching the model, matching the Anthropic/Ollama non-blank message guard:

```text
go test ./go -run 'TestOpenAI_NewOpenAIResponsesHandler_(Good_NonStreaming|Bad_RejectsStreaming|Bad_RejectsBlankInput)|TestCompatHandlers_Bad_AnthropicRejectsEmptyMessages|TestCompatHandlers_Bad_OllamaRejectsEmptyChatMessages|TestCompatHandlers_Bad_OllamaRejectsEmptyGeneratePrompt' -count=1 -v
--- PASS: TestCompatHandlers_Bad_AnthropicRejectsEmptyMessages (0.00s)
--- PASS: TestCompatHandlers_Bad_OllamaRejectsEmptyChatMessages (0.00s)
--- PASS: TestCompatHandlers_Bad_OllamaRejectsEmptyGeneratePrompt (0.00s)
--- PASS: TestOpenAI_NewOpenAIResponsesHandler_Good_NonStreaming (0.00s)
--- PASS: TestOpenAI_NewOpenAIResponsesHandler_Bad_RejectsStreaming (0.00s)
--- PASS: TestOpenAI_NewOpenAIResponsesHandler_Bad_RejectsBlankInput (0.00s)
PASS
ok dappco.re/go/rocm 0.011s

go test ./go -count=1
ok dappco.re/go/rocm 0.148s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/openai 0.003s
ok dappco.re/go/inference/scheduler 0.053s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.625s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.182s

git diff --check && git -C external/go-inference diff --check
(no output)
```

The package-first OpenAI Responses endpoint now requires at least one non-blank converted message, so blank prompts do not consume scheduler/model work and return the same `input or instructions are required` bad-request shape as missing input.

Follow-up verification after tightening Ollama-compatible chat validation so whitespace-only message arrays are rejected before reaching the model, matching Anthropic and Responses non-blank input behavior:

```text
go test ./go -run 'TestCompatHandlers_(Good_OllamaChatAndGenerate|Bad_OllamaRejectsEmptyChatMessages|Bad_OllamaRejectsBlankChatMessages|Bad_OllamaRejectsEmptyGeneratePrompt)|TestOpenAI_NewOpenAIResponsesHandler_Bad_RejectsBlankInput|TestCompatHandlers_Bad_AnthropicRejectsEmptyMessages' -count=1 -v
--- PASS: TestCompatHandlers_Bad_AnthropicRejectsEmptyMessages (0.00s)
--- PASS: TestCompatHandlers_Good_OllamaChatAndGenerate (0.00s)
--- PASS: TestCompatHandlers_Bad_OllamaRejectsEmptyChatMessages (0.00s)
--- PASS: TestCompatHandlers_Bad_OllamaRejectsBlankChatMessages (0.00s)
--- PASS: TestCompatHandlers_Bad_OllamaRejectsEmptyGeneratePrompt (0.00s)
--- PASS: TestOpenAI_NewOpenAIResponsesHandler_Bad_RejectsBlankInput (0.00s)
PASS
ok dappco.re/go/rocm 0.011s

go test ./go -count=1
ok dappco.re/go/rocm 0.147s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.110s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/ollama 0.003s
ok dappco.re/go/inference/openai 0.003s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.627s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.184s

git diff --check && git -C external/go-inference diff --check
(no output)
```

The package-first wire compatibility layer now consistently rejects blank-only user work across OpenAI Responses, Anthropic Messages, Ollama chat, and Ollama generate before invoking the model or scheduler.

Follow-up verification after adding explicit ROCm parser coverage for Gemma4/Gemma4-E2B reasoning turn markers:

```text
go test ./go -run 'TestParserRegistry_Good_(GemmaChannels|Gemma4E2BTurnMarkers|RocmModelImplementsParserContracts|RocmModelUsesModelTypeFallback)' -count=1 -v
--- PASS: TestParserRegistry_Good_GemmaChannels (0.00s)
--- PASS: TestParserRegistry_Good_Gemma4E2BTurnMarkers (0.00s)
--- PASS: TestParserRegistry_Good_RocmModelImplementsParserContracts (0.00s)
--- PASS: TestParserRegistry_Good_RocmModelUsesModelTypeFallback (0.00s)
PASS
ok dappco.re/go/rocm 0.002s

go test ./external/go-inference/go/parser -run 'TestRegistry_DefaultLookup_Good_ModelFamilies|TestReasoning_BuiltinParsers_Good' -count=1 -v
--- PASS: TestReasoning_BuiltinParsers_Good (0.00s)
--- PASS: TestRegistry_DefaultLookup_Good_ModelFamilies (0.00s)
PASS
ok dappco.re/go/inference/parser 0.002s

go test ./go -count=1
ok dappco.re/go/rocm 0.147s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/parser 0.003s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.639s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.182s

git diff --check && git -C external/go-inference diff --check
(no output)
```

The parser registry is now package-locally covered for `gemma4`, `gemma4_text`, and `Gemma4ForCausalLM` aliases using Gemma turn markers such as `<start_of_turn>analysis`.

ROCm `ScheduledModel.Err()` now returns a scheduler-local stored error before checking the wrapped model, so zero-value/nil-wrapped scheduler failures are observable. `Classify` and `BatchGenerate` now set that stored error on nil wrapped-model failures; the shared scheduler mirrors this for `New(nil, ...)` non-streaming delegates.

Follow-up verification after making scheduler `Close` zero-value safe and making ROCm direct `Schedule` reject nil wrapped/partially initialized scheduler instances before touching queue internals:

```text
go test ./go -run 'TestScheduler_Bad_NilWrappedModelRecordsErr|TestScheduler_Bad_RejectsClosedScheduler|TestScheduler_Good_CloseIsIdempotent' -count=1 -v
--- PASS: TestScheduler_Bad_NilWrappedModelRecordsErr (0.00s)
--- PASS: TestScheduler_Bad_RejectsClosedScheduler (0.00s)
--- PASS: TestScheduler_Good_CloseIsIdempotent (0.00s)
PASS
ok dappco.re/go/rocm 0.007s

go test ./external/go-inference/go/scheduler -run 'TestModel_NilAndErrorPaths|TestModel_CloseRejectsNewWorkAndIsIdempotent' -count=1 -v
--- PASS: TestModel_CloseRejectsNewWorkAndIsIdempotent_Bad (0.00s)
--- PASS: TestModel_NilAndErrorPaths_Bad (0.00s)
PASS
ok dappco.re/go/inference/scheduler 0.002s

go test ./go -count=1
ok dappco.re/go/rocm 0.146s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/scheduler 0.052s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.627s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.179s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Latest follow-up:

- Pulled the current sibling `go-mlx` dev reference from `https://forge.lthn.sh/core/go-mlx.git` to `c95ae46`; its Gemma4 text runtime uses `gemma4_text` for `Gemma4ForCausalLM`/`Gemma4TextForCausalLM`.
- Mirrored that ROCm-side without importing `go-mlx`: standalone Gemma4 text packs now inspect as `gemma4_text`, while conditional-generation/nested multimodal Gemma4 packs remain `gemma4`.
- Updated Gemma4 q4 runtime gates so `gemma4_text` still reaches text prompt generation, q4 package prefill/decode, layer0, small-decode, and opt-in hardware smoke helpers.
- Hardened ROCm `Benchmark`/`Evaluate` `Err()` ownership: returned failures are recorded, successful wrapper calls clear stale/internal best-effort errors, and eval stream/empty-dataset failures are now observable through `Err()`.

Latest verification:

```text
go test ./go -run 'TestNativeContract_(BenchmarkBad_PropagatesCacheStatsError|EvaluateQualityProbes_Bad_RecordsUnavailableGeneration|EvaluateSuccessClearsLastError_Good|EvaluateBadRecordsFailure|EvaluateBadRejectsEmptyDataset)' -count=1 -v
PASS
ok dappco.re/go/rocm 0.002s

go test ./go -run 'TestNativeContract_(ModelPackInspectorArchitectureFixtures_Good|GeneratedPromptUsesExplicitGemma4Q4TextMode_Good)|TestHIPGemma4Q4Layer0_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 0.010s

go test ./go -count=1
ok dappco.re/go/rocm 0.143s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/scheduler 0.054s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.767s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.176s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Live RX 7800 XT q4 recheck after the `gemma4_text` alias alignment, forcing the discrete GPU with `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85`:

```text
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
  GO_ROCM_RUN_MODEL_TESTS=1 \
  GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
  GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
  GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
  GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' \
  go test ./go -run 'TestNative(DecodeSmokeKernelStatus|ModelPackSmokeGemma4E2B)_Good' -count=1 -v

=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:95: Gemma4 q4 package Prefill/Decode prompt=[0] next=150930 decoded=1865 text="ology"
    hip_hardware_test.go:96: Gemma4 q4 public Generate prompt="text:Hi" prompt_tokens=[10979] generated tokens=[18070 35506] text=["anth" "思い"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (45.66s)
=== RUN   TestNativeModelPackSmokeGemma4E2B_Good
--- PASS: TestNativeModelPackSmokeGemma4E2B_Good (0.22s)
PASS
ok   dappco.re/go/rocm 45.906s
```

The q4 route remains the fast Gemma4-E2B development loop on the real RX 7800 XT, including public Generate through the experimental text path. Production decode, prefill, and production KV backing remain explicitly `not_linked`.

Repository setup recheck for the original workspace request:

- `/home/claude/Code/core/go-ai`, `/home/claude/Code/core/go-ml`, and `/home/claude/Code/core/go-inference` all have `origin` set to `https://forge.lthn.sh/core/go-*.git`, are on `dev`, and `git pull --ff-only origin dev` reported already up to date.
- Recursive Core submodules under those repos are on `dev` and fast-forward pulls reported up to date. Third-party MLX library submodules under `go-ml/external/go-mlx/lib/*` are intentionally detached and were skipped.

Follow-up after tightening optional ROCm service error ownership:

- `rocmModel` cache wrappers (`CacheStats`, `WarmCache`, `ClearCache`) now clear stale public errors on entry and record returned failures in `Err()`.
- `rocmModel` state wrappers (`CaptureState`, `RestoreState`, `WakeState`, `SleepState`, `ForkState`) now do the same, keeping optional cache/state surfaces aligned with Generate/Classify/Embed/Benchmark/Eval error observability.

Verification:

```text
go test ./go -run 'TestCacheService_Bad_RocmModelWarmCacheRecordsErr|TestStateSession_Bad_RocmModelRestoreStateRecordsErr|TestStateSession_Good_RocmModelRestoresMetadataStateBundle' -count=1 -v
PASS
ok dappco.re/go/rocm 0.008s

go test ./go -count=1
ok dappco.re/go/rocm 0.139s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.109s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.768s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.175s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/scheduler 0.052s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Fresh live q4 hardware recheck after the optional cache/state `Err()` wrapper changes:

```text
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
  GO_ROCM_RUN_MODEL_TESTS=1 \
  GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
  GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
  GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
  GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' \
  go test ./go -run 'TestNative(DecodeSmokeKernelStatus|ModelPackSmokeGemma4E2B)_Good' -count=1 -v

=== RUN   TestNativeDecodeSmokeKernelStatus_Good
    hip_hardware_test.go:95: Gemma4 q4 package Prefill/Decode prompt=[0] next=150930 decoded=1865 text="ology"
    hip_hardware_test.go:96: Gemma4 q4 public Generate prompt="text:Hi" prompt_tokens=[10979] generated tokens=[18070 35506] text=["anth" "思い"]
--- PASS: TestNativeDecodeSmokeKernelStatus_Good (45.14s)
=== RUN   TestNativeModelPackSmokeGemma4E2B_Good
--- PASS: TestNativeModelPackSmokeGemma4E2B_Good (0.23s)
PASS
ok dappco.re/go/rocm 45.392s
```

Follow-up after tightening adapter lifecycle error ownership:

- `rocmModel.LoadAdapter` and `rocmModel.UnloadAdapter` now clear stale public errors on entry and record returned failures in `Err()`, aligning adapter lifecycle observability with text, embedding/rerank, benchmark/eval, cache, and state wrappers.
- Added focused adapter lifecycle tests proving native load/unload failures are observable through `Err()` and that subsequent successful adapter calls clear stale failures.

Verification:

```text
go test ./go -run 'TestNativeContract_(AdapterLifecycle_Good|LoadAdapterBadNativeFailureKeepsActiveAdapter_Bad|LoadAdapterBadRecordsErrAndSuccessClears_Bad|LoadAdapterBadStateCloseFailureDoesNotCallNative_Bad|UnloadAdapterBadRecordsErrAndSuccessClears_Bad|UnloadAdapterBadStateCloseFailureDoesNotCallNative_Bad)' -count=1 -v
PASS
ok dappco.re/go/rocm 0.002s

go test ./go -count=1
ok dappco.re/go/rocm 0.141s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.109s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.012s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.764s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.178s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/scheduler 0.053s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Follow-up after filling cache service example coverage:

- Added `BlockCacheService.CacheStats` and `BlockCacheService.ClearCache` examples so the public cache service surface now has runnable examples for warm, disk refs, stats, clear, and close.

Verification:

```text
go test ./go -run 'ExampleBlockCacheService_(WarmCache|WarmCache_diskRefs|CacheStats|ClearCache|Close)$' -count=1 -v
PASS
ok dappco.re/go/rocm 0.002s

go test ./go -count=1
ok dappco.re/go/rocm 0.145s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.113s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.651s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.186s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/scheduler 0.053s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Follow-up after tightening the native tokenizer surface and adding optional-surface examples:

- `rocmModel.ApplyChatTemplate` now clears stale public errors on entry and records native template failures in `Err()`.
- Internal token-count helpers use a non-recording template helper so best-effort prompt counting does not poison successful Chat/Eval calls.
- Added native-only examples for `rocmModel.ApplyChatTemplate` and adapter load/unload.

Verification:

```text
go test ./go -run 'TestNativeContract_(TokenizerBoundariesCloneMutableSlices_Good|ApplyChatTemplateBadRecordsErrAndSuccessClears_Bad)|Example_rocmModel_(ApplyChatTemplate|LoadAdapter)$' -count=1 -v
PASS
ok dappco.re/go/rocm 0.008s

go test ./go -count=1
ok dappco.re/go/rocm 0.146s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.110s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.010s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.630s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.194s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/scheduler 0.053s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Follow-up after filling native optional embedding/rerank examples:

- Added native wrapper examples for `rocmModel.Embed` and `rocmModel.Rerank`, using the fake loaded embedding model so examples exercise the package-first optional interfaces without hardware.

Verification:

```text
go test ./go -run 'Example_rocmModel_(ApplyChatTemplate|LoadAdapter|Embed|Rerank)$' -count=1 -v
PASS
ok dappco.re/go/rocm 0.002s

go test ./go -count=1
ok dappco.re/go/rocm 0.145s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.109s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.643s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.188s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/scheduler 0.053s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Follow-up after tightening `Close` error ownership:

- `rocmModel.Close` now clears stale public errors on entry and records state/cache/native close failures in `Err()`.
- Added native close failure coverage, and extended the idempotent close test to prove successful close clears stale `Err()` while failure keeps runtime state intact.

Verification:

```text
go test ./go -run 'TestNativeContract_Close(GoodIdempotentClearsRuntimeState|BadStateCloseFailureKeepsRuntime_Bad|BadNativeCloseFailureKeepsRuntime_Bad)' -count=1 -v
PASS
ok dappco.re/go/rocm 0.002s

go test ./go -count=1
ok dappco.re/go/rocm 0.143s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.110s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.645s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.187s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference/scheduler 0.053s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Latest follow-up after tightening model parser error ownership:

- `rocmModel.ParseReasoning` and `rocmModel.ParseTools` now clear stale public errors on entry and record parser failures in `Err()`.
- Added model-level parser examples for reasoning and tool parsing alongside the standalone `ParserRegistry` examples.
- Reran the fast Gemma4-E2B q4 live smoke on `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85`; package Prefill/Decode and public `Generate("text:Hi")` passed on the RX 7800 XT.

Verification:

```text
go test ./go -run 'TestParserRegistry_(Good_RocmModel(ImplementsParserContracts|UsesModelTypeFallback|ParseReasoningClearsStaleErr)|Bad_RocmModelParseToolsRecordsErrAndSuccessClears_Bad)|Example_rocmModel_Parse(Reasoning|Tools)$|ExampleParserRegistry_Parse(Reasoning|Tools)$' -count=1 -v
PASS
ok dappco.re/go/rocm 0.012s

go test ./go -count=1
ok dappco.re/go/rocm 0.149s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.769s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.178s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference 0.009s
ok dappco.re/go/inference/scheduler 0.054s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' go test ./go -run 'TestNative(DecodeSmokeKernelStatus|ModelPackSmokeGemma4E2B)_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 45.705s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Latest follow-up after filling model-pack inspection/planning coverage:

- Added `InspectModelPack` Ugly coverage for a cancelled context.
- Added Gemma4-style examples for `InspectModelPack` and `PlanModelFit`, including RX 7800 XT-sized memory planning and the expected `rocm-16gb` / `k-q8-v-q4` plan.
- No GPU smoke was rerun in this pass because only CPU-side examples/tests and inspection preflight behavior changed.

Verification:

```text
go test ./go -run 'TestNativeContract_(PlanModelFit_Good|ModelPackInspectorReadsSidecars_Good|ModelPackInspectorUgly_CancelledContext)$|Example_(inspectModelPack|planModelFit)$' -count=1 -v
PASS
ok dappco.re/go/rocm 0.009s

go test ./go -count=1
ok dappco.re/go/rocm 0.155s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.110s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.661s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.171s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference 0.008s
ok dappco.re/go/inference/scheduler 0.053s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Latest follow-up after filling compatibility handler Ugly coverage:

- Added Ugly transport/resolver coverage for the Anthropic and Ollama compatibility handlers: nil resolver, wrong HTTP method, nil request, and missing-model resolver failure.
- No GPU smoke was rerun in this pass because only HTTP compatibility handler tests changed.

Verification:

```text
go test ./go -run 'TestCompatHandlers_(Good|Bad|Ugly)|ExampleNew(AnthropicMessagesHandler|OllamaHandler)$' -count=1 -v
PASS
ok dappco.re/go/rocm 0.007s

go test ./go -count=1
ok dappco.re/go/rocm 0.154s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.110s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.012s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.688s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.174s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference 0.009s
ok dappco.re/go/inference/scheduler 0.053s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Latest follow-up after filling OpenAI Responses handler Ugly coverage:

- Added Ugly transport/resolver coverage for `NewOpenAIResponsesHandler`: nil resolver, nil request, wrong HTTP method, malformed JSON body, and missing-model resolver failure.
- No GPU smoke was rerun in this pass because only HTTP compatibility handler tests changed.

Verification:

```text
go test ./go -run 'TestOpenAI_NewOpenAI(Resolver|Handler|ResponsesHandler|ServiceMux)|ExampleNewOpenAI(ResponsesHandler|ServiceMux)$' -count=1 -v
PASS
ok dappco.re/go/rocm 0.008s

go test ./go -count=1
ok dappco.re/go/rocm 0.154s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.110s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.012s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.681s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.172s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference 0.008s
ok dappco.re/go/inference/scheduler 0.053s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Latest follow-up after live RX 7800 XT hardware/model gates:

- Ran the broader HIP/native gate pinned to the discrete RX 7800 XT UUID with the gfx1100 HSACO. The gate passed; the model-only smokes inside it skipped as expected because `GO_ROCM_RUN_MODEL_TESTS` was not set for that command, and cache hardware skipped because `GO_ROCM_RUN_CACHE_TESTS` was not set.
- Ran the q4 Gemma4-E2B model gate pinned to the same GPU and HSACO. The live q4 package prefill/decode smoke produced `prompt=[0] next=150930 decoded=1865 text="ology"`. The public generation smoke for `text:Hi` produced `prompt_tokens=[10979] generated tokens=[18070 35506]`; the first decoded piece was `"anth"` and the second was non-ASCII text in the test output.
- This refreshes the live hardware evidence, but it does not close the main GOAL because production native model-family prefill/decode is still labelled pending/not-linked outside the experimental q4 Gemma4 path, and the production cache/training/model-merge integration items remain open.

Verification:

```text
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_HIP_TESTS=1 \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
go test ./go -run 'TestHIP|TestNative' -count=1 -v
PASS
ok dappco.re/go/rocm 0.252s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' \
go test ./go -run 'Test.*Smoke|Test.*Generate|Test.*Decode' -count=1 -v
PASS
ok dappco.re/go/rocm 45.987s
```

Latest follow-up after BF16 Gemma4-E2B primitive smoke:

- Ran the BF16 Gemma4-E2B model pack on the pinned RX 7800 XT with the gfx1100 HSACO. This exercises the BF16 embedding/RMSNorm/projection/attention/logit primitive chain in `TestNativeDecodeSmokeKernelStatus_Good`; it does not yet mean production model-family `Generate` is linked for BF16.
- The live BF16 layer0/tied-LM-head smoke produced greedy token `158750` with score `28.711908`.

Verification:

```text
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
go test ./go -run 'TestNativeDecodeSmokeKernelStatus_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 17.356s
```

Latest follow-up after aligning Gemma4 q4 positions with go-mlx:

- Compared the q4 generate shape with the local `go-mlx` dev reference. MLX uses zero-based RoPE/cache offsets: first prompt token at position `0`, then decode at the current cache length.
- Updated the experimental ROCm Gemma4 q4 package path to match: package prefill now starts at position `0`, package decode defaults to `state.tokenCount`, and public q4 token generation starts at position `0`.
- Added focused q4 package assertions for zero-based `decode_position` labels: a two-token prefill reports last prefill position `1`, and the following decode reports position `2`.
- Re-ran focused live q4 hardware smoke on the RX 7800 XT. The one-token `text:Hi` prompt still produced the same generated IDs, which is expected for this tiny prompt because the relative RoPE relationship remains unchanged.

Verification:

```text
go test ./go -run 'TestHIPGemma4Q4PackagePrefillDecode_Good|TestHIPGemma4Q4Layer0_Good|TestNativeContract_GeneratedPromptUsesExplicitGemma4Q4TextMode_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 0.009s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' \
go test ./go -run 'TestNative(DecodeSmokeKernelStatus|ModelPackSmokeGemma4E2B)_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 48.625s

go test ./go -count=1
ok dappco.re/go/rocm 0.146s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.113s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.636s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.184s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference 0.008s
ok dappco.re/go/inference/anthropic 0.002s
ok dappco.re/go/inference/bench 0.002s
ok dappco.re/go/inference/decode 0.002s
ok dappco.re/go/inference/eval 0.002s
ok dappco.re/go/inference/ollama 0.002s
ok dappco.re/go/inference/openai 0.003s
ok dappco.re/go/inference/parser 0.003s
ok dappco.re/go/inference/quant/codebook 0.011s
ok dappco.re/go/inference/quant/jang 0.002s
ok dappco.re/go/inference/scheduler 0.062s
ok dappco.re/go/inference/state 0.011s
ok dappco.re/go/inference/state/filestore 0.012s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Latest follow-up after adding Gemma4 q4 layer-scalar runtime parity:

- The local `go-mlx` Gemma4 decoder applies each layer's `layer_scalar` after the layer residual/MLP path. ROCm q4 was not consuming these BF16 scalar tensors.
- Added ROCm q4 layer-scalar loading from `language_model.model.layers.N.layer_scalar`, converting BF16 to float32 once while building the loaded layer config.
- Applied the scalar after the decoder layer final residual using the existing HIP vector-scale primitive. Missing scalar tensors still default to `1` for fixtures and partial packs.
- Added focused fake-device coverage proving a `0.5` scalar halves the final hidden state in the q4 decoder-layer path.
- Re-ran focused live q4 smoke on the RX 7800 XT. The generated IDs changed, confirming the real q4 runtime is now consuming layer-scalar tensors:
  - package Prefill/Decode: `prompt=[0] next=47416 decoded=107365`
  - public Generate `text:Hi`: `prompt_tokens=[10979] generated tokens=[128636 238666]`
- Output is still not coherent chat; the next likely Gemma4 parity gaps from go-mlx are the per-layer input projection path and shared-KV layout, not the now-fixed layer scalar.

Verification:

```text
go test ./go -run 'TestHIPGemma4Q4Layer0_Good|TestHIPGemma4Q4PackagePrefillDecode_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 0.007s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' \
go test ./go -run 'TestNative(DecodeSmokeKernelStatus|ModelPackSmokeGemma4E2B)_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 48.921s

go test ./go -count=1
ok dappco.re/go/rocm 0.148s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.636s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.187s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference 0.009s
ok dappco.re/go/inference/anthropic 0.002s
ok dappco.re/go/inference/bench 0.002s
ok dappco.re/go/inference/decode 0.002s
ok dappco.re/go/inference/eval 0.002s
ok dappco.re/go/inference/ollama 0.002s
ok dappco.re/go/inference/openai 0.003s
ok dappco.re/go/inference/parser 0.002s
ok dappco.re/go/inference/quant/codebook 0.009s
ok dappco.re/go/inference/quant/jang 0.009s
ok dappco.re/go/inference/scheduler 0.060s
ok dappco.re/go/inference/state 0.009s
ok dappco.re/go/inference/state/filestore 0.010s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Latest follow-up at EOF after adding Gemma4 q4 per-layer input parity:

- Used the local `go-mlx` dev branch as a reference only. ROCm still does not import `go-mlx`.
- Added a BF16 device-weight projection path so loaded BF16 tensors can use the existing projection kernel without pretending they are FP16.
- Added Gemma4 q4 per-layer input loading for `embed_tokens_per_layer`, `per_layer_model_projection`, `per_layer_projection_norm`, `layers.N.per_layer_input_gate`, `layers.N.per_layer_projection`, and `layers.N.post_per_layer_input_norm`.
- Wired single-token forward to precompute the per-layer input vectors once per token, split them per layer, and feed each decoder layer's gate/projection path before `layer_scalar`.
- The per-layer GELU and elementwise multiply are currently host-side in the experimental q4 path; projection, RMSNorm, embedding, vector scale/add, attention, and sampling still use HIP primitives.
- Added fake-device coverage for direct per-layer decoder application and for global per-layer precompute labels/launches.
- Re-ran live q4 on the RX 7800 XT. Per-layer input tensors are now loaded and affect logits:
  - package Prefill/Decode: `prompt=[0] next=855 decoded=8704 text=" Den"`
  - public Generate `text:Hi`: `prompt_tokens=[10979] generated tokens=[240709 35843] text=["ሻ" " mó"]`
- Output is still not coherent chat. The remaining likely Gemma4 parity gap is shared/global KV layout or another architecture-specific attention detail, not the now-linked per-layer input path.
- Keep HIP hardware and q4 model gates separate: setting `GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1` during the broad `TestHIP|TestNative` unit sweep intentionally changes default text-prompt behavior and trips `TestHIPGemma4Q4Layer0_Good`.

Verification:

```text
go test ./go -run 'TestHIPGemma4Q4(Layer0|PerLayerInputPrecompute)_Good|TestHIPGemma4Q4Layer0_Bad' -count=1 -v
PASS
ok dappco.re/go/rocm 0.007s

go test ./go -count=1
ok dappco.re/go/rocm 0.146s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.112s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' \
go test ./go -run 'TestNative(DecodeSmokeKernelStatus|ModelPackSmokeGemma4E2B)_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 53.720s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' \
go test ./go -run 'Test.*Smoke|Test.*Generate|Test.*Decode' -count=1 -v
PASS
ok dappco.re/go/rocm 53.729s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_HIP_TESTS=1 \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
go test ./go -run 'TestHIP|TestNative' -count=1
ok dappco.re/go/rocm 0.249s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.628s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.182s

(cd external/go-inference/go && go test ./... -count=1)
ok dappco.re/go/inference 0.008s
ok dappco.re/go/inference/anthropic 0.002s
ok dappco.re/go/inference/bench 0.002s
ok dappco.re/go/inference/decode 0.002s
ok dappco.re/go/inference/eval 0.002s
ok dappco.re/go/inference/ollama 0.002s
ok dappco.re/go/inference/openai 0.003s
ok dappco.re/go/inference/parser 0.002s
ok dappco.re/go/inference/quant/codebook 0.002s
ok dappco.re/go/inference/quant/jang 0.002s
ok dappco.re/go/inference/scheduler 0.053s
ok dappco.re/go/inference/state 0.002s
ok dappco.re/go/inference/state/filestore 0.003s

git diff --check && git -C external/go-inference diff --check
(no output)
```

Latest follow-up at EOF after Gemma4 q4 BOS parity:

- Used `go-mlx` dev as the e2e reference and matched its BOS behavior for text tokenization.
- ROCm loaded tokenizers now record `<bos>` and prepend it for normal text prompts unless the prompt already starts with `<bos>`.
- Updated fake-device BOS coverage and the live Gemma4 q4 tokenizer smoke to expect `Hello world -> [2 9259 1902]`.
- Re-ran live Gemma4 q4 on the RX 7800 XT with `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85`.
- `text:Hi` now uses prompt tokens `[2 10979]` and generated decoded tokens `[236764 3307]` with text `["," "my"]`.
- `text:Hello, my name is` now uses prompt tokens `[2 9259 236764 1041 1463 563]` and generated decoded tokens `[870 11069 1567 1604]` with text `[" [" "Your" "Name" "],"]`.
- This removes the no-BOS prompt mismatch and moves live q4 output into normal decoded tokenizer text. The path remains experimental and slow because parts of Gemma4 q4 are still host-side or smoke-kernel based.

Verification:

```text
go test ./go -run 'TestHIPGemma4Q4(Layer0|PackagePrefillDecode|SharedKV|PerLayerInputPrecompute)_Good|TestHIPGemma4Q4Layer0_Bad' -count=1 -v
PASS
ok dappco.re/go/rocm 0.009s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' go test ./go -run 'TestNativeDecodeSmokeKernelStatus_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 74.534s

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 GO_ROCM_RUN_MODEL_TESTS=1 GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hello, my name is' GO_ROCM_GEMMA4_Q4_GENERATE_TOKENS=4 go test ./go -run 'TestNativeDecodeSmokeKernelStatus_Good' -count=1 -v
PASS
ok dappco.re/go/rocm 136.436s

go test ./go -count=1
ok dappco.re/go/rocm 0.149s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./go -count=1
ok dappco.re/go/rocm 0.111s

go test -tags rocm_legacy_server ./go -count=1
ok dappco.re/go/rocm 0.011s

go test ./... -count=1
ok dappco.re/go/rocm/workspace 0.637s

GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
? dappco.re/go/rocm/workspace [no test files]

go test -tags rocm_legacy_server ./... -count=1
ok dappco.re/go/rocm/workspace 0.184s

(external/go-inference/go) go test ./... -count=1
PASS

git diff --check
(no output)

(external/go-inference) git diff --check
(no output)
```
