# go-rocm Architecture

## Overview

go-rocm is the AMD ROCm backend for the Core Go inference ecosystem. Its target shape is native and package-first, matching the role `go-mlx` plays on Apple Silicon: `go-inference` owns shared contracts, `go-rocm` owns AMD runtime execution, and higher packages (`go-ai`, `go-ml`) consume the common surface without circular dependencies.

The default backend does not depend on a managed `llama-server` subprocess. That server-backed implementation was useful as a fast bootstrap path, but it hides model state behind an HTTP harness and blocks research-grade features such as shared KV cache ownership, portable state bundles, adapter identity checks, probe streams, and native training hooks.

## Native Runtime Surface

The default `linux/amd64` build provides:

- `inference.Backend` registration for the `rocm` backend
- `inference.ModelFitPlanner` for ROCm memory/context/cache recommendations
- `inference.TokenizerModel`, `AdapterModel`, `ProbeableModel`, `BenchableModel`, `Evaluator`, `CacheService`, parser, and state lifecycle contracts on loaded models
- `NewScheduledModel` for bounded queueing, request IDs, cancellation, and scheduler probe events
- OpenAI-compatible chat, non-streaming Responses, Anthropic Messages with SSE streaming, Ollama chat/generate/tags/show, capability, cache, cancel, embeddings, and rerank handler mounts via shared `go-inference` DTOs
- GGUF and safetensors model-pack inspection, including BERT embedding, rerank, sequence-classifier metadata hints, and Gemma4 nested text-config metadata
- sysfs VRAM monitoring for memory planning and metrics
- a native HIP runtime loader that validates tensor plans, allocates GPU buffers, and copies GGUF tensor bytes or safetensors payloads, including sharded packs, through a dynamic `libamdhip64` driver

When cgo is disabled or `libamdhip64` cannot be opened, `Available()` intentionally returns false. That is preferable to silently falling back to an HTTP server because callers can make an explicit backend choice and tests can prove whether native execution is actually present.

## Package Structure

```text
go-rocm/
├── go/rocm.go                 package doc and exported discovery types
├── go/register_rocm.go        linux && amd64 registration
├── go/rocm_stub.go            non-linux or non-amd64 stubs
├── go/native.go               default native ROCm contract surface
├── go/scheduler.go            bounded request scheduler and cancellation
├── go/cache.go                metadata-first in-memory cache service
├── go/parser_registry.go      reasoning/tool parser registry
├── go/state_session.go        URI-first state wake/sleep/fork groundwork
├── go/hip_runtime.go          native tensor ownership and load path
├── go/hip_driver_cgo.go       dynamic HIP driver, cgo only
├── go/hip_driver_nocgo.go     unavailable HIP driver for cgo-off builds
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

The non-Linux/non-amd64 stub registers an unavailable `rocm` backend with
`go-inference`, keeps `ROCmAvailable()` false, and returns a
platform-unavailable `LoadModel` error. That keeps blank imports deterministic
on a developer Mac while making runtime absence explicit.

On a non-Linux development machine, compile the default Linux surface with:

```bash
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test -exec=/usr/bin/true ./...
```

The `-exec=/usr/bin/true` gate proves compilation without trying to execute Linux test binaries on macOS.

From a Linux ROCm host, the inverse Darwin compile proxy is:

```bash
GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test -exec=/bin/true ./go/... -count=1
```

That proves the Darwin stub file set compiles. It does not replace running
`go test ./go/...` on macOS, where the stub tests and examples execute.

## Native Load Flow

1. `register_rocm.go` registers `&rocmBackend{}` with `go-inference`.
2. `LoadModel(path, opts...)` reads GGUF metadata and tensor descriptors with `gguf.ReadInfo`.
3. Load options become a `nativeLoadConfig`: context size, GPU layers, parallel slots, adapter path, model metadata, data offset, and tensor map.
4. `hipRuntime.LoadModel` validates embeddings, output head, layer-count hints, tensor dtypes, quantisation, rank-2 projection shapes, model identity dimensions, and GGUF byte sizes before allocation.
5. `hipRuntime.LoadModel` allocates device buffers and copies tensor bytes from the GGUF data section using HIP.
6. `hipLoadedModel` dispatches generation/classification plus typed prefill, decode, and projection calls through a narrow `hipKernelSet` seam. The default build uses an explicit not-linked kernel set. CPU reference token embedding lookup, single-head and multi-head attention, causal prefill attention, decode-with-KV, an integrated tiny LM prefill/decode fixture, fp16/q8/f32 projection, LoRA projection, RMSNorm, RoPE, greedy and top-k/temperature sampler math, logit and entropy probe summaries, prompt-lookup draft, speculative-accept, embedding mean-pool, rerank cosine, cross-entropy/perplexity, distillation KL, and GRPO advantage fixtures define the deterministic outputs future HIP kernels must match.
7. `rocmModel` wraps that native model and exposes shared `go-inference` generation, tokenizer, adapter, embedding, rerank, probe, bench, eval, cache, parser, and state contracts.

The native seam is deliberately small. HIP-specific ownership of tensors, graph execution, KV cache, adapter application, and eventual training stays behind `nativeModel` rather than leaking into consumers. Production decode/prefill and general adapter kernels are not linked into the default model path; normal generation, typed prefill/decode, and generic adapter application return explicit not-linked errors after the model weights are loaded.

The launch layer is an optional driver interface: fake drivers can record validated prefill/decode/projection/JANGTQ-projection/codebook-lookup/LoRA-projection/embedding-mean-pool/rerank-cosine/RMSNorm/RoPE/greedy-sampler/attention/MoE-router/MoE-lazy-experts/tiny-prefill/tiny-decode/cross-entropy/distillation/GRPO launch configs, and the cgo system driver can load a packaged `rocm_kernels_*.hsaco` sidecar or caller override from `GO_ROCM_KERNEL_HSACO`, copy the fixed launch packet to device memory, and invoke `hipModuleLaunchKernel`. When a compatible module is found and the driver can launch kernels, backend and loaded-model capability reports expose linked projection, embedding, rerank, loss-fixture, and LoRA kernels for package-local projection, toy embedding/scoring/loss requests, a narrow loaded tiny-model `rocm-tiny-lora` output-head adapter path, a Qwen/Gemma small LM-head LoRA adapter path, and a BERT classifier-head LoRA adapter path; tiny vocab-major f32 loaded models with f32/f16/raw-q8/JANGTQ/codebook output heads can also run the toy prefill/decode/generate path and experimental model-level embedding mean-pool/rerank calls against already loaded tensor pointers. Tiny loaded models with a JANGTQ/MXTQ output head recompute toy logits through `rocm_jangtq_projection`; tiny loaded models with a scalar codebook/VQ output head expand toy output weights through `rocm_codebook_lookup` and then project through `rocm_projection`.

BERT-style f32 `word_embeddings.weight` packs can load without an LM head and run the same experimental mean-pool embedding plus embedding-cosine rerank path, or use f32/f16 `classifier.weight`/`score.weight` plus optional f32/f16 `classifier.bias`/`score.bias` sequence-classification heads to classify prompts and score paired query/document embeddings from the positive class logit. Fixture-scale Qwen/Gemma loaded models with f32 embeddings plus first-layer f16 Q/K/V/O and LM-head weights can route typed `DecodeToken` through the small decode composition: the request token selects a device-resident embedding row, the primitive kernels compute a single decode step, package-local KV receives the appended key/value vector, and supplied device KV mirrors append only the decoded token page before refreshing the descriptor table. Full production Qwen/Gemma decode/prefill generation remains planned; the loaded Gemma4 MLX-q4 path is an experimental package-local generation route over packed q4 tensors and descriptor-backed k-q8-v-q4 KV state, so its capability, benchmark, and eval labels still carry `production_decode=not_linked`, `production_prefill=not_linked`, and `production_kv_cache_backing=not_linked`.

`kernels/rocm_kernels.hip` is the first source artifact for those symbols: prefill/decode consume validated packets and referenced memory for fp16, q8, and k-q8-v-q4 cache-mode descriptors and can write status markers through reserved launch-packet fields, projection computes the toy fp16/q8/f32 row projections used by fixtures, JANGTQ/MXTQ projection unpacks signed 2/4/8-bit toy weights for deterministic packed-projection fixtures and loaded tiny output-head logits, codebook lookup expands toy VQ code IDs into float vectors for deterministic lookup fixtures and loaded tiny output-head logits, LoRA projection applies a toy fp32 low-rank delta, embedding mean-pool and rerank cosine run toy vector fixtures, RMSNorm/RoPE/greedy sampling/single-head attention/MoE router/lazy-expert bitmap execute deterministic transformer primitive fixtures, package-local Qwen/Gemma small decode smokes compose those primitives through RMSNorm, fp16 Q/K/V/O and LM-head projections, RoPE, attention, and greedy sampling using both fixture-supplied weights and loaded device-resident tensor weights, `rocm_tiny_prefill` runs a toy embedding-attention-output fixture that writes logits, final-token attention, toy KV buffers, and greedy result buffers, and `rocm_tiny_decode` consumes those toy prior KV vectors, appends the decoded token embedding, and writes updated KV/logits/attention/greedy buffers. The loss fixtures `rocm_cross_entropy_loss`, `rocm_distillation_kl_loss`, and `rocm_grpo_advantage` write toy loss/perplexity/KL/advantage outputs for eval and training stepping stones. The tiny kernels support fp32, fp16, and q8 output heads through the same launch packet; the loaded-model toy path currently accepts f32 embeddings with f32/f16/raw-q8/JANGTQ/codebook output heads, and the `rocm-tiny-lora`, Qwen/Gemma small LM-head, and BERT classifier adapter formats can add rank-limited deltas through `rocm_lora_projection`. The MoE router/lazy-expert and generic LoRA kernels are still launch fixtures beyond the narrow loaded-model adapter paths; production MoE expert paging, general packed-weight integration, codebook model integration, production adapter application, and production cross-encoder/scorer rerank integration remain metadata-planned. Without a compatible module, launch attempts still fail clearly and capabilities remain pre-kernel. Backend, loaded-model, benchmark, and eval reports expose `runtime_status`, `kernel_status`, `cross_entropy_kernel`, `distillation_kernel`, `grpo_kernel`, `decode_kernel`, `embedding_kernel`, `prefill_kernel`, `projection_kernel`, `lora_kernel`, and `rerank_kernel` labels; generation/decode-helper capability labels plus benchmark and eval quality-probe runtime reports also expose `decode_kernel_name`, `prefill_kernel_name`, and `kernel_scope`, with tiny fixture paths marked `production_decode=not_linked` and `production_prefill=not_linked`; embedding/rerank capability labels expose `kernel_scope=loaded_embedding_fixtures` or `loaded_rerank_fixtures`, enumerate tiny/BERT embedding and embedding-cosine/classifier rerank fixture scopes, and keep production embedding/rerank model integration not linked; LoRA capability labels expose `kernel_scope=loaded_adapter_fixtures`, `supported_adapter_scopes=tiny_output_head,qwen_gemma_small_lm_head,bert_sequence_classifier`, and `production_adapter_application=not_linked`; eval capability reports also expose `loss_kernel`/`loss_kernel_name`, and loaded-model capability status is driven by the model's kernel set.

## Memory Planning

`PlanModelFit` is available before the HIP loader exists. It uses model identity plus measured or supplied memory to estimate:

- machine class (`rocm-16gb`, `rocm-24gb`, `rocm-64gb-plus`)
- recommended context length and batch size
- cache mode (`q8` on constrained memory or long context, otherwise `fp16`)
- estimated KV cache bytes
- architecture and quantisation support flags
- LoRA/training feasibility hints, while capability reports keep LoRA training, distillation, and GRPO planned with `training_kernel=not_linked`, `training_interface=not_implemented`, and required future-kernel labels until native training APIs exist; toy distillation KL and GRPO advantage fixture labels follow their explicit kernel-status fields, loaded HIP models expose package-local fixture hooks for those toy losses, and planned training capabilities advertise the `RunNativeAdamWUpdatePass` packed-state optimizer helper as update-only without implementing shared training interfaces

The estimates are intentionally conservative because ROCm users are likely to run 16 GB cards where context and KV decisions matter more than generic defaults.

## Probe And Eval Surface

`rocmModel` emits shared `inference.ProbeEvent` token events during generation. Classification calls with `WithLogits` now emit compact logit and entropy summaries through the same sink, and capability reports mark `probe.logits` experimental once prefill/classification kernels are linked. Native kernels can add attention head selection, layer coherence, router decisions, residual summaries, cache pressure, and memory pressure through the same sink without changing consumers.

The current benchmark/eval hooks are lightweight wrappers over the model surface. Benchmark runs apply `WarmupRuns` across every configured prompt without adding those tokens to measured counters, report measured operation counts plus aggregate prefill/decode/total duration, average first-token latency, shared active/peak memory fields, and active/peak memory labels, emit cache-pressure and memory-pressure probe events with shared probe payloads, mirror cache labels, count measured probe events while forwarding to any configured sink, and measure active LoRA adapter overhead by temporarily comparing the same prompt run against the native model with the adapter unloaded before restoring the adapter identity. If restore fails, wrapper adapter/cache/state identity is cleared to match native state. Eval reports token-count metrics without requiring prefill kernels, always label the cross-entropy fixture status with `loss_kernel`, `loss_kernel_name`, and `loss_scope`, can compute experimental classification cross-entropy/perplexity when samples provide `target_token_id` labels and the model returns logits from batched `Classify` calls sized by `EvalConfig.BatchSize`, uses the linked HIP cross-entropy fixture for loss/perplexity when `cross_entropy_kernel=linked`, otherwise records loss/perplexity as unsupported, not requested when a linked classification path is available but no loss target is present, logits-unavailable when classification returns no logits, or CPU-reference fallback when the native loss hook is unavailable, and can run bounded qualitative probes through the generation surface while recording unavailable native decode as probe failures instead of failing token-count eval. Loaded Gemma4 MLX-q4 Benchmark/Eval/Classify reports can now use the q4 Generate/Classify path and emit q4 logit-probe and helper labels while retaining production not-linked labels. Capability details and labels now follow linked decode/prefill/classification/loss status so benchmark and eval reports stop advertising stale caveats once an experimental kernel set is active. Broader native quality probes can reuse the same shared report structs once model-family decode kernels produce production logits.

`SpeculativeDecode` and `PromptLookupDecode` are package-first helpers over the shared `go-inference/decode` harness. They collect tokens from `inference.TextModel.Generate`, propagate stream errors through `Err()`, and report shared acceptance metrics. Capability reports keep these helpers planned while decode kernels are not linked, then mark them experimental for loaded models with an experimental decode path such as Gemma4 MLX-q4 Generate; this avoids presenting the helper as production kernel acceleration.

## Scheduler, Cache, Parser, And State

`NewScheduledModel` wraps any `inference.TextModel` with a single-worker bounded queue. Its direct `Generate` and `Chat` methods also enter that queue, record enqueue failures through `Err()`, return stable request IDs through `Schedule`, reject duplicate in-flight IDs, support cancellation before prefill and during decode, delegate unknown cancellation IDs to an underlying `CancellableModel` when present, closes the wrapped model idempotently, and emits `ProbeEventScheduler` events with typed queue-depth/latency payloads plus labels.

`BlockCacheService` is a metadata-first prompt-cache service. It reports stable block IDs, compatibility labels, memory/disk accounting, hits/misses, hit rate, deterministic restore-time accounting, warm, and label-filtered clear. Cache stats carry cached-token counts, the effective warmed cache mode, KV block size, key/value widths, optional disk-ref labels, and optional device-remirror labels so benchmark `cache.*` labels preserve the warmed cache shape. Capability and benchmark labels report `prompt.cache=experimental` for this metadata/package-local path while native prefill reuse remains pending; cache-related capabilities label device backing as best-effort remirror and fully HIP-owned KV as pending. Longer prompt warms reuse matching prefixes for hit accounting while still creating the requested full block. Planner-selected `fp16`, `q8`, and `k-q8-v-q4` cache modes validate through package-local KV cache pages and use those pages for warm-byte accounting before HIP owns real cache memory. Those package-local pages can now be mirrored into HIP device allocations for opt-in allocation/copy/free smoke coverage and for warmed/cold-restored cache blocks on loaded ROCm models, with explicit `kv_backing=hip_device_mirror` stats, Go kernel descriptors containing token ranges, widths, encodings, byte sizes, and device pointers, a versioned fixed-width little-endian descriptor byte layout, a separately owned HIP descriptor-table allocation, a validated 64-byte KV launch descriptor containing the descriptor pointer plus mode/page/token/width metadata, a 64-byte prefill launch packet carrying token-buffer pointer/count/bytes plus cache mode/block/width metadata, and a 96-byte decode launch packet that adds token ID and decode position to the KV descriptor. Projection launch preflight similarly owns temporary device buffers for input, fp16/q8 weights, optional bias, and output and validates a 96-byte pointer/shape/encoding packet; LoRA projection launch preflight owns fp32 base/A/B adapter tensors and validates a 128-byte pointer/shape/alpha packet. The fake projection and LoRA launchers consume those packets, write output device buffers, and the projection seams read them back through device-to-host copy before returning. Fake prefill/decode launchers also consume their launch packets and validate the referenced token buffer, KV descriptor, and descriptor-table memory. The typed prefill/decode seam can carry a validated device mirror plus descriptor table through fake linked kernels, and fake decode updates clone host KV first, appends only the decoded token page to the device mirror, labels before/after token and page counts plus descriptor-refresh success for the incremental update, tracks borrowed versus owned pages during the scratch append, and refreshes the descriptor table so append or descriptor-table failures do not advance host state ahead of device state. The not-linked decode stub runs decode launch-packet preflight when device KV resources are supplied, but real decode kernels still cannot consume the mirror yet. When configured with a `go-inference/state` binary writer, block-prefix cache warm writes metadata refs, ROCm KV-mode warm writes portable `rocm/kv-cache+json` package-local snapshots, both report `DiskBytes`, and fresh cache services can exact-match cold rehydrate those refs by deterministic URI before counting a miss. CPU reference helpers cover MoE top-k routing probes, lazy expert residency, residual summaries, JANGTQ/MXTQ packed projection, codebook lookup, and LoRA projection so future HIP kernels have deterministic behavior to match. Native HIP KV pages can replace the backing store without changing the `inference.CacheService` surface.

`ParserRegistry` handles common thinking and tool formats for Qwen, Gemma, MiniMax, DeepSeek R1, GPT-OSS, Mistral, Kimi, GLM, Hermes, Granite, JSON tools, Mistral `[TOOL_CALLS]` arrays, and generic XML/JSON fallbacks. It is wired onto `rocmModel` through `ReasoningParser` and `ToolParser`, using `modelType` as a fallback when architecture metadata is not populated.

`StateSession` uses `go-inference/state` wake/sleep/fork requests and URI-first refs. It preserves model/tokenizer compatibility checks on wake and sleep with hash checks plus architecture/tokenizer-kind fallbacks, and keeps runtime handles out of JSON. `rocmModel` also implements the shared `StatefulModel` interface for metadata-only `StateBundle` capture/restore; durable KV payloads remain on the URI-first session path because a bundle contains refs rather than tensor bytes. The retained-generation contract is strict: the `.kv` ref is the source of truth, and wake must fail rather than reconstructing state by replaying prior prompt text, manuscripts, or chat transcripts. Current Gemma4 book runs use that `.kv` ref as an MP4-style vector stream and append only the new user turn to the restored state. When the runtime owns package-local KV pages, sleep writes a binary `rocm/kv-cache+json` state ref and wake restores those pages into a session with mode, block, byte, and token accounting. Loaded `rocmModel` instances keep that session so a later sleep call can serialize the restored KV runtime instead of dropping back to placeholder state; adapter changes, bundle restore, wake replacement, LoRA baseline resets, and close clear it with resource cleanup. HIP device KV mirrors can also be snapshotted through device-to-host page copies into the same portable encoding with explicit `kv_serialize=device_mirror` and `kv_device_backing=mirrored` labels. A loaded ROCm model with an available HIP driver now best-effort remirrors restored portable KV refs into HIP device pages on wake and fork, and falls back to package-local pages with failure labels if remirror fails; production kernels still do not own fully HIP-native state restore.

## Wire Service Mounts

`NewOpenAIHandler` exposes chat completions. `NewOpenAIResponsesHandler` exposes the non-streaming Responses shape over the same resolver. `NewOpenAIServiceMux` mounts chat, responses, capability, cache, cancel, embeddings, and rerank endpoints. `NewAnthropicMessagesHandler` exposes Anthropic Messages, including SSE streaming when `stream` is true, and `NewOllamaHandler` exposes non-streaming Ollama chat/generate plus `/api/tags` and `/api/show` registry shapes. These helpers add no provider policy or external API keys. Responses and Ollama streaming remain pending, so the wire capabilities are reported as experimental.

## GGUF Metadata Parser

`internal/gguf/` is a standalone binary metadata reader. It supports GGUF v2 and v3, validates the magic/version, reads metadata KV pairs, and extracts architecture, name, file type, size label, context length, block count, file size, tensor names, tensor dimensions, tensor type names, tensor byte sizes, tensor offsets, alignment, and model data offset without loading tensor data.

The parser reads only the header, not tensor payloads, so model discovery and fit planning remain cheap even for multi-GB packs.

## VRAM Monitoring

`GetVRAMInfo()` reads `mem_info_vram_total` and `mem_info_vram_used` from sysfs (`/sys/class/drm/cardN/device/`). It selects the card with the largest VRAM total, which avoids hardcoding card numbers on machines with an iGPU plus dGPU.

Reads are non-atomic, so `Free` is clamped to zero if a transient read reports `Used > Total`.

## Legacy Server Path

The old server-backed implementation is still buildable with `-tags rocm_legacy_server`. It starts `llama-server`, uses OpenAI-compatible streaming endpoints, and keeps the old subprocess lifecycle tests. That path exists as a compatibility/debug bridge only; it is not the default architecture.
