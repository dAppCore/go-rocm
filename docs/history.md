# go-rocm Project History

## Origin

go-rocm was created on 19 February 2026 by Virgil (orchestrator) as the AMD GPU backend for the go-inference ecosystem. The sibling package go-mlx provides the same interface on macOS using Apple Metal and direct CGO; go-rocm targets the Linux homelab's AMD Radeon RX 7800 XT.

The package was built by Charon (test coverage and build agent, running on the Linux homelab) in a single day across four phases: environment validation, core implementation, robustness, model support, and performance tuning.

---

## Phase 0: Environment Validation (19 Feb 2026)

**Purpose**: Confirm the homelab hardware, ROCm installation, and llama.cpp build before writing any Go code.

**Findings:**

- GPU architecture confirmed as gfx1100 via `rocminfo`. Virgil's initial notes stated gfx1101; the physical hardware is gfx1100. No `HSA_OVERRIDE_GFX_VERSION` override is required.
- ROCm version: 7.2.0 (minimum required is 6.x).
- Kernel: 6.17.0-14-generic.
- llama.cpp built from commit `11c325c` with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DGGML_HIP_ROCWMMA_FATTN=ON`. Binary installed to `/usr/local/bin/llama-server`.

**Critical discovery: iGPU crash**

The Ryzen 9 9950X has an integrated GPU that ROCm exposes alongside the RX
7800 XT. Early llama-server testing treated the RX 7800 XT as ordinal 0 and the
Radeon Graphics iGPU as the other visible device, but later native HIP testing
made ordinal selection an explicit non-contract because the onboard GPU can
appear as device 0. Current native runs pin the RX 7800 XT by UUID with
`ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85`.

llama-server's auto-fit logic split the model across both devices. Loading
tensors to the iGPU triggered `ROCm error: unspecified launch failure` and a
core dump. The legacy-server workaround was `HIP_VISIBLE_DEVICES=0`; the native
driver acceptance path instead pins the dGPU by UUID.

**Baseline benchmark (Gemma3-4B-Q4_K_M):**

| Metric | Value |
|--------|-------|
| Prefill speed | 396 tok/s |
| Decode speed | 109 tok/s |
| Time to first token | ~40 ms (16-token prompt) |
| Server startup | ~6 s |
| VRAM used | ~3.4 GB of 16 GB |

---

## Phase 1: Core Implementation (19 Feb 2026)

**Commits**: `1d8d65f`, `9aa7f62`, `3c75677`, `def3167`, `a8c4947`, `0e68d71`

**GPU detection** (`1d8d65f`): `Available()` checks `/dev/kfd` and `findLlamaServer()`. Returns false if either is absent. `findLlamaServer()` checks `ROCM_LLAMA_SERVER_PATH` env var first, then PATH.

**Server lifecycle** (`9aa7f62`): `server.go` implements `startServer()`, `waitReady()`, and `stop()`. Health polling at 100ms intervals with a 60-second startup timeout. Graceful shutdown sends SIGTERM, waits 5 seconds, then SIGKILL. `serverEnv()` filters and overrides `HIP_VISIBLE_DEVICES` using a filter-then-append pattern to avoid duplicate env var entries (last-write-wins is platform-specific).

**HTTP client** (`3c75677`, `def3167`): `internal/llamacpp/` provides `Client` with `Health()`, `ChatComplete()`, and `Complete()`. Both completion methods return `(iter.Seq[string], func() error)`. The SSE parser reads `data: ` prefixed lines from the response body using a `bufio.Scanner`, stops at `[DONE]`, and propagates I/O errors via a pointer.

**TextModel implementation** (`a8c4947`): `model.go` wraps the server and client. `Generate()` calls `/v1/completions`; `Chat()` calls `/v1/chat/completions`. Both check `server.alive()` before dispatching and record errors in `lastErr` under a mutex.

**Integration tests** (`0e68d71`): `TestROCm_LoadAndGenerate`, `TestROCm_Chat`, `TestROCm_ContextCancellation` all pass on the RX 7800 XT using Gemma3-1B. Tests gated behind `//go:build rocm`.

**Design decisions recorded in FINDINGS.md:**

- `Token.ID` left as zero; llama-server's streaming API does not return token IDs. No downstream consumer uses the ID field.
- `StopTokens []int32` silently ignored; the llama-server API expects stop sequences as strings, not token IDs. YAGNI.

---

## Phase 2: Robustness (19 Feb 2026)

**Commits**: `2c4966e`, `c07f37a`, `c50a8e9`, `b7342ec`, `a6e647c`, `501de83`, `954c570`

**Server crash recovery** (`2c4966e`, `c07f37a`): `server.alive()` reads from the `exited` channel non-blockingly. `Generate()` and `Chat()` return an empty iterator immediately if the server has died, recording the exit error in `lastErr`.

**Port conflict handling** (`c50a8e9`, `b7342ec`): `startServer()` retries up to 3 times with a fresh port on process exit during startup. Timeouts are not retried (a stuck server is a distinct failure mode from a port conflict).

**Graceful shutdown** (`a6e647c`): Integration test `TestROCm_GracefulShutdown` confirms the server survives a mid-stream context cancel and accepts subsequent Generate calls. Already worked from Phase 1; integration test added to prevent regression.

**VRAM monitoring** (`501de83`, `954c570`): `GetVRAMInfo()` reads sysfs (`/sys/class/drm/cardN/device/mem_info_vram_*`). Selects the dGPU by highest total VRAM, correctly distinguishing the RX 7800 XT (16 GB) from the Ryzen iGPU (2 GB) without hardcoding card numbers. Uint64 underflow guard: `Free` is clamped to zero if `Used > Total` due to non-atomic sysfs reads.

**Concurrent requests** (`a6e647c`): Three goroutines calling `Generate()` simultaneously all receive output. llama-server serialises via its default single-slot configuration. No Go-level locking needed on the model for concurrent Generate calls.

**Known limitation recorded**: `Err()` is a single shared field. Concurrent callers can overwrite each other's errors. This matches the go-inference interface contract and is not a bug.

---

## Phase 3: Model Support (19 Feb 2026)

**Commits**: `c7c9389`, `af23565`, `2c77f6f`

**GGUF metadata parser** (`c7c9389`): `internal/gguf/` reads GGUF v2/v3 binary headers. Extracts architecture, name, file type, size label, context length, and block count without reading tensor data. Supports all 13 GGUF value type codes. String length capped at 1 MiB. Handles uint64 values for context_length/block_count (some producers use uint64 instead of uint32). Architecture-specific keys are collected as candidates and resolved after `general.architecture` is known, handling the case where architecture-specific keys appear before the architecture key in the KV stream.

**Model discovery** (`af23565`): `DiscoverModels(dir)` globs for `*.gguf` files, parses each via the GGUF parser, and returns `[]ModelInfo`. Unparseable files are skipped silently.

**LoadModel enrichment** (`2c77f6f`): Replaced filename-based architecture guessing with GGUF metadata. `meta.Architecture` is now set from `general.architecture`, which is more accurate: Mistral-7B-v0.3 correctly reports `"llama"` (not `"mistral"`), and Qwen-2.5-7B correctly reports `"qwen2"`. Context auto-capped at `min(model_context_length, 4096)` when the caller does not specify a context length, preventing VRAM exhaustion on models with 128K+ native context (Gemma3-4B/12B/27B and Llama-3.1-8B all have 131072-token native context).

**Chat templates**: Confirmed that llama-server reads `tokenizer.chat_template` from GGUF and applies it on `/v1/chat/completions`. No go-rocm code required.

**Model inventory discovered** (at `/data/lem/gguf/`):

| Model | Architecture | Quant | Context |
|-------|-------------|-------|---------|
| Gemma3-1B-layered-v2 | gemma3 | Q5_K_M / Q8_0 | 32768 |
| Gemma3-4B | gemma3 | Q4_K_M | 131072 |
| Gemma3-12B | gemma3 | Q4_K_M | 131072 |
| Gemma3-27B | gemma3 | Q4_K_M | 131072 |
| Llama-3.1-8B | llama | Q4_K_M | 131072 |
| Mistral-7B-v0.3 | llama | Q4_K_M | 32768 |
| Qwen-2.5-7B | qwen2 | Q4_K_M | 32768 |

---

## Phase 4: Performance (19 Feb 2026)

**Commits**: `870ee23` (benchmarks), `3719734` (go-inference: ParallelSlots), `72120bb` (go-rocm: --parallel support)

**Benchmark suite** (`870ee23`): Three benchmarks gated behind `//go:build rocm`:
- `BenchmarkDecode` — 128-token generation, reports tok/s
- `BenchmarkTTFT` — single-token generation, reports µs/first-tok
- `BenchmarkConcurrent` — 4 goroutines, 4 parallel slots, reports tok/s-aggregate

All three run across Gemma3-4B, Llama3.1-8B, and Qwen2.5-7B. Model load time is excluded via `b.StopTimer()` / `b.StartTimer()`.

**Flash attention comparison**: llama-server built with and without `-DGGML_HIP_ROCWMMA_FATTN=ON` at ctx=2048. No significant difference (≤4.4% variation, within noise). rocWMMA flash attention is designed for large context windows where the KV cache dominates. At ctx=2048, standard attention is as fast or faster. Flash attention auto-activates only when beneficial and does not degrade performance at small context sizes. The flag remains enabled in the build configuration.

**Parallel slots** (`3719734`, `72120bb`): `ParallelSlots int` added to go-inference's `LoadConfig`. `inference.WithParallelSlots(n)` passes `--parallel N` to llama-server. Aggregate throughput with 4 slots at ctx=2048:

| Model | Single-slot tok/s | 4-slot aggregate tok/s | Ratio |
|-------|------------------|----------------------|-------|
| Gemma3-4B-Q4_K_M | 102.5 | 238.9 | 2.3x |
| Llama-3.1-8B-Q4_K_M | 77.1 | 166.2 | 2.2x |
| Qwen-2.5-7B-Q4_K_M | 84.4 | 178.0 | 2.1x |

## Native Parity Tranche (11 May 2026)

The native package-first path was brought closer to `go-mlx` contract parity:

- Submodules were moved to their `dev` branches so `go-inference` exposes `contracts.go`, OpenAI service helpers, scheduler/cache/parser contracts, and `go-inference/state`.
- ROCm capability reports now include the expanded shared IDs, including scheduler, request cancellation, cache, parser, agent-memory, and state wake/sleep/fork statuses.
- `NewScheduledModel` adds bounded queueing, duplicate in-flight request ID rejection, cancellation, direct `Generate`/`Chat` scheduling, idempotent wrapped-model close, underlying cancellable-model fallback for unknown request IDs, and scheduler probe events with typed `ProbeScheduler` queue/latency payloads plus labels.
- `BlockCacheService` adds metadata-first block-prefix prompt-cache stats, warm, clear, compatibility checks, stable block IDs, optional `go-inference/state` metadata refs, portable package-local KV snapshot disk refs for ROCm KV cache modes, exact cold rehydrate from deterministic disk-ref URIs, disk-byte accounting, and best-effort HIP device remirroring for warmed/cold-restored portable KV snapshots; cache capability labels report `kv_device_backing=best_effort_remirror` while fully HIP-owned disk KV and native prefill reuse remain pending.
- `SpeculativeDecode` and `PromptLookupDecode` add package-first wrappers around the shared `go-inference/decode` acceptance harness, with examples, stream-error propagation, and capability status that remains planned until a loaded model reports a linked experimental decode path.
- `ParserRegistry` is backed by the shared `go-inference/parser` package and adds reasoning/tool parsers and examples for Qwen/Gemma/MiniMax/DeepSeek/GPT-OSS/Mistral/Kimi/GLM/Hermes/Granite families, including `[TOOL_CALLS]` arrays, loaded-model `modelType` fallback, and generic JSON/XML.
- `StateSession` adds URI-first wake/sleep/fork metadata lifecycle over `go-inference/state`, including examples, hash plus architecture/tokenizer-kind compatibility checks, metadata-only shared `StatefulModel` bundle capture/restore, planned labels for non-KV placeholders, binary `rocm/kv-cache+json` sleep/wake refs when a session owns package-local KV pages, loaded-model persistence for wake-then-sleep KV round trips, adapter/restore/close state reset with owned-runtime cleanup, HIP device-mirror snapshot refs by copying mirrored pages device-to-host into the same portable encoding, and loaded-model best-effort wake/fork remirror of restored portable KV refs back into HIP device pages with package-local fallback until production kernels own restore. The retained-generation contract was tightened so the `.kv` vector ref is the source of truth: wake must fail when durable KV is missing or incompatible, and retained book turns must not rebuild state by replaying prior prompt/manuscript text.
- Model-pack inspection now covers safetensors headers, tokenizer sidecars, MiniMax/JANGTQ, shared `go-inference/quant` JANG/codebook validators, MoE hints, Qwen/Gemma/Mistral/Mixtral/Phi/DeepSeek/GPT-OSS/Kimi/GLM/Hermes/Granite/BERT architecture aliases, BERT embedding/rerank/classifier hints, and bounded malformed GGUF/safetensors/codebook errors; malformed weight metadata or missing architecture metadata keeps the inspection report available but `Supported=false`, mixed-shard failures clear partial weight-summary labels, and unsupported packs do not emit memory-fit labels. MoE/JANGTQ/codebook capabilities report experimental metadata-only status with fixture-kernel labels while production model integration remains pending.
- Memory-planner coverage now pins small, 24GB, 64GB, and long-context cache-mode transitions.
- Package-local KV cache pages now cover fp16, q8, and k-q8-v-q4 round trips, paged appends, byte counts, hit rate, restore timing, binary snapshot refs, constructibility of planner-selected cache modes through cache warm, and a HIP device-mirror allocation/copy/free smoke path with incremental decoded-token page appends, device-to-host portable snapshots, a fixed descriptor byte layout, device-resident descriptor table, 64-byte KV launch descriptor preflight, 64-byte prefill launch-packet encoding, and 96-byte decode launch-packet encoding before kernels consume device KV pages.
- The metadata block cache now accounts deterministic restore milliseconds for exact and prefix hits.
- CPU reference helpers now cover MoE top-k routing probes, lazy expert residency, residual-summary probes, JANGTQ/MXTQ packed dequant/projection, codebook lookup, and LoRA projection before native kernels are linked.
- Wire service mounts now include chat completions, non-streaming Responses, Anthropic Messages with SSE streaming, Ollama chat/generate/tags/show, capability, cache, cancel, embeddings, and rerank endpoints without provider policy, with ROCm mux coverage for cache warm/stats/clear, cache-warm request validation, and scheduled-model cancel routing.
- Wire helpers now include examples for OpenAI Responses/service mux and Anthropic/Ollama compatibility handlers.
- ROCm wire-service tests prove embeddings and rerank endpoints stay explicitly not implemented for generic models until native kernels exist; loaded HIP tiny f32 models now expose experimental model-level embedding mean-pool, rerank, narrow `rocm-tiny-lora` output-head adapter application, and JANGTQ/MXTQ packed plus codebook/VQ output-head logits through the compiled kernel fixtures when `GO_ROCM_KERNEL_HSACO` is set, and BERT-style f32 word-embedding-only packs can load without an LM head for experimental mean-pool embedding plus embedding-cosine rerank. BERT sequence-classification packs with f32/f16 `classifier.weight` or `score.weight` plus optional f32/f16 `classifier.bias`/`score.bias` can now rank query/document pairs through the projection path, load experimental classifier LoRA adapters through `rocm_lora_projection`, and report `experimental_bert_sequence_classifier` labels. Fixture-scale Qwen/Gemma small decode can also apply an experimental LM-head LoRA adapter through `rocm_lora_projection` after composing the loaded first-layer primitive kernels.
- Benchmark reports now run warmups over every configured prompt without counting them as measured tokens, report measured operation counts plus first-token latency, aggregate prefill/decode/total duration, shared active/peak memory fields, and active/peak memory labels, emit cache/memory pressure probes, report measured probe counts while forwarding to configured sinks, mirror cache shape labels including KV block size and optional disk refs, and measure active LoRA adapter overhead by restoring the original adapter identity after a baseline comparison while clearing wrapper adapter/cache/state identity if restore fails; classification strips logits unless `WithLogits` is requested and emits compact logit/entropy probe summaries when logits are requested; eval reports include token-count metrics, batched experimental classification loss/perplexity from `target_token_id` plus logits when available, explicit unsupported/logits-unavailable loss labels otherwise, and qualitative probe results that record unavailable generation without failing pre-kernel evaluation.
- Training capability reports keep LoRA training, distillation, and GRPO planned, label the non-callable state explicitly with `runtime_status=planned`, `training_kernel=not_linked`, `training_interface=not_implemented`, and the required future kernel class, expose explicit `distillation_kernel`/`grpo_kernel`-aware toy fixture labels for distillation KL and GRPO advantage normalization, advertise `RunNativeAdamWUpdatePass` as a packed-state optimizer update-only helper, and keep the loaded-model toy loss hooks package-local instead of implementing shared training interfaces.
- HIP loading now validates tensor shape/dtype/quantisation, projection rank/model dimensions, and byte-size requirements before allocation, adds typed prefill/decode/projection kernel seams carrying device KV mirrors plus descriptor tables, KV launch descriptors, prefill/decode launch packets, 96-byte projection, MLX affine q4 projection, JANGTQ-projection, and f32/BF16/MLX-q4 embedding-lookup launch packets, 64-byte codebook-lookup/embedding-mean-pool/rerank-cosine/RMSNorm/RoPE/greedy-sampler/vector-add/vector-scale/SwiGLU/MoE-router/MoE-lazy-expert launch packets, 96-byte attention launch packets, 128-byte LoRA-projection launch packets, 160-byte tiny-prefill/tiny-decode launch packets with fp32/fp16/q8 output-head encodings, token/projection/transformer device-buffer upload/rollback coverage, device-to-host copy support with incremental decoded-token device-KV appends, fake prefill/decode referenced-memory validation, and fake projection/MLX-q4-projection/JANGTQ-projection/codebook-lookup/LoRA-projection/embedding-lookup/embedding-mean-pool/rerank-cosine/RMSNorm/RoPE/greedy/attention/vector-add/vector-scale/SwiGLU/MoE-router/MoE-lazy-expert/tiny-prefill/tiny-decode launch output readback, an optional fake-testable kernel launch contract, and cgo HSACO module launch plumbing behind `GO_ROCM_KERNEL_HSACO`, deterministic f32/BF16/MLX-q4 token embedding lookup, single-head and multi-head attention, causal prefill attention, decode-with-KV, integrated tiny LM prefill/decode, composed Qwen/Gemma small decode smoke with fixture and loaded device-resident weights, fp16/bf16/q8/f32 projection, BERT f32/f16 sequence-classifier rerank scoring, MLX affine q4 packed projection, JANGTQ/MXTQ packed projection and codebook lookup including loaded tiny output-head logits, LoRA projection, f32/BF16-weight RMSNorm, RoPE, vector add, vector scale, SwiGLU, MoE top-k routing and lazy expert residency bitmaps, greedy and top-k/temperature sampler references, logit, entropy, selected-head, and layer-coherence probe summaries, prompt-lookup draft, speculative-accept, embedding mean-pool, rerank cosine, cross-entropy/perplexity, distillation KL, and GRPO advantage reference fixtures, and routes decode/prefill/projection calls through explicit not-linked kernel stubs with capability, bench, eval, cache-pressure probe, and memory-pressure probe status labels.
- `kernels/rocm_kernels.hip` adds ABI-matched HIP source for `rocm_prefill`, `rocm_decode`, `rocm_projection`, `rocm_mlx_q4_projection`, `rocm_jangtq_projection`, `rocm_codebook_lookup`, `rocm_lora_projection`, `rocm_embedding_lookup`, `rocm_embedding_mean_pool`, `rocm_rerank_cosine`, `rocm_rms_norm`, `rocm_rope`, `rocm_greedy_sample`, `rocm_attention`, `rocm_vector_add`, `rocm_vector_scale`, `rocm_swiglu`, `rocm_moe_router`, `rocm_moe_lazy_experts`, `rocm_tiny_prefill`, `rocm_tiny_decode`, `rocm_cross_entropy_loss`, `rocm_distillation_kl_loss`, and `rocm_grpo_advantage`; `hipcc --genco --offload-arch=gfx1100` compiles it, and opt-in hardware smokes verify fp16/bf16/q8 projection output, MLX affine q4 packed projection output, f32/BF16/MLX-q4 embedding lookup output, JANGTQ/MXTQ packed projection output, codebook lookup output, LoRA projection output, embedding mean-pool output, rerank cosine output, f32/BF16-weight RMSNorm/RoPE/greedy/attention/vector-add/vector-scale/SwiGLU/MoE-router/MoE-lazy-expert primitive output, cross-entropy/distillation/GRPO loss fixture output, composed Qwen3 small decode output through those primitives using fixture and loaded device-resident weights, typed loaded-model Qwen3 small `DecodeToken` from a device embedding row with fp16 package-local KV plus q8/k-q8-v-q4 package-local and incrementally appended device-KV paths, toy tiny-prefill KV/logits/attention/greedy output, toy tiny-decode updated-KV/logits/attention/greedy output using prefill-written KV, loaded tiny-model generation from device-resident f32 tensors with f32/f16/raw-q8/JANGTQ/codebook output heads, fp16/q8 tiny output-head variants, plus prefill/decode packet-consumer launches for fp16, q8, and k-q8-v-q4 cache modes with device-written status markers through `GO_ROCM_KERNEL_HSACO`.
- Loaded HIP models now opt into the compiled projection, embedding, rerank, and LoRA symbols when `GO_ROCM_KERNEL_HSACO` is set and the driver supports kernel launch; tiny vocab-major f32 loaded models with f32/f16/raw-q8/JANGTQ/codebook output heads also opt into toy prefill/decode/generate plus experimental model-level embedding/rerank and `rocm-tiny-lora` output-head adapter application, with capability, benchmark, and eval quality-probe labels marking `kernel_scope=toy_tiny_fixture` and production decode/prefill still not linked; embedding/rerank capability labels mark loaded embedding/rerank fixture scopes and keep production embedding/rerank model integration not linked; LoRA capability labels mark `kernel_scope=loaded_adapter_fixtures`, enumerate the tiny output-head, Qwen/Gemma small LM-head, and BERT sequence-classifier adapter scopes, and keep production adapter application not linked; BERT sequence-classification packs can load experimental classifier LoRA adapters; and fixture-scale Qwen/Gemma loaded models can route typed `DecodeToken` through the small decode composition by reading the request token's f32 embedding row from device memory, appending package-local KV, incrementally appending supplied device KV mirrors, and optionally applying an experimental LM-head LoRA adapter, while normal decode/prefill model generation remains explicitly not linked.
- Native `LoadModel` now accepts validated safetensors packs in addition to GGUF, including Gemma4-E2B-style nested `text_config`, huge tokenizer `model_max_length` sentinels, tied word embeddings without a separate LM head, BF16 dtype inference, safetensors index sidecar validation, BF16 side tensors, sharded payload files, and packed U32 4-bit embedding weights. The RX 7800 XT reports slightly below exact 16 GiB through HIP, so memory-class planning applies a small tolerance before classifying it as `rocm-16gb`. The Gemma4-E2B model smoke now also launches `rocm_embedding_lookup` against the loaded device-resident BF16 `embed_tokens.weight` tensor, `rocm_vector_scale` to apply the Gemma `sqrt(hidden_size)` embedding scale, Gemma-style `(1 + weight)` `rocm_rms_norm` against loaded device-resident BF16 layer-0 `input_layernorm.weight`, `rocm_projection` against the loaded device-resident BF16 layer-0 `q_proj`, `k_proj`, and `v_proj` tensors, HIP RMSNorm against BF16 `self_attn.q_norm.weight`/`k_norm.weight`, `rocm_rope` over all 8 real q heads plus the KV head, `rocm_attention` over each one-token GQA head, `rocm_projection` against the loaded device-resident BF16 layer-0 `o_proj` tensor using the 2048-wide attention concat, Gemma-style `post_attention_layernorm` before the attention residual add, `pre_feedforward_layernorm` before MLP, BF16 `mlp.gate_proj`/`mlp.up_proj`, `rocm_swiglu`, BF16 `mlp.down_proj`, Gemma-style `post_feedforward_layernorm`, and final residual add, checking results against the same safetensors rows/weights and CPU references. The local 4-bit Gemma4-E2B pack now launches `rocm_embedding_lookup` against loaded packed U32 `embed_tokens.weight` with BF16 scales/biases, `rocm_vector_scale` for the same embedding scale, Gemma-style BF16 RMSNorm against q4-pack hidden/q-k/feedforward norm tensors, `rocm_mlx_q4_projection` against loaded packed q/k/v/o and MLP gate/up/down tensors plus the tied q4 LM head, RoPE, one-token GQA attention, residual adds, SwiGLU, and greedy sampling over vocab logits, making it a faster full layer-0 packed-tensor development loop aligned with the `go-mlx` embedding-scale/norm/residual order, while the BF16 sharded pack remains the correctness anchor for raw tensor-row GPU comparisons and production Gemma4 decode/generation remains not linked.
- The Gemma4 MLX-q4 route now reaches public experimental Generate, Chat, BatchGenerate, Classify, Benchmark, Eval, SpeculativeDecode, and PromptLookupDecode through the package-local q4 Generate/Classify path on the RX 7800 XT, including the `gemma4_text` aliases used by current `go-mlx` dev for text-only Gemma4 packs. Hardware smokes cover q4 generated-token output, q4 classification logit probes, q4 benchmark/eval labels, and one-token speculative/prompt-lookup helper acceptance while capability reports keep production decode, prefill, and KV cache backing explicitly `not_linked`; the BF16 Gemma4-E2B smoke remains the raw-tensor correctness anchor and now reaches final-norm plus tied LM-head final-logit output.
- Model-pack memory-fit planning now carries known weight bytes from GGUF/safetensors metadata into `PlanModelFit` and applies Gemma4 sliding-attention metadata, so the BF16 Gemma4-E2B sharded pack reports `memory_fit=true` on the 16GB RX 7800 XT with a 7 full-layer, 28 sliding-layer, 512-token-window KV estimate while still proving the weights can be copied into HIP memory. Gemma4 text metadata now also reports GQA geometry for future kernels: attention heads, KV heads, head dimensions, derived query/KV widths, RMSNorm epsilon, and final-logit softcap.
- Fake HIP driver tests cover nil driver, unavailable driver, allocation failure, and free failure paths without requiring ROCm hardware.
- ROCm hardware/model/cache smoke tests are opt-in and skip by default unless the required environment variables and model path are provided.
- Non-Linux/non-amd64 stubs now register an unavailable `rocm` backend, keep `ROCmAvailable()` false, and return platform-unavailable `LoadModel` errors so developer-Mac blank imports are deterministic; Linux-hosted Darwin compile proxies prove the stub file set builds, but real macOS execution remains the final no-ROCm gate.
- Import-boundary tests prevent `go-rocm` from depending on concrete workflow/runtime packages such as `go-mlx`, `go-ai`, `go-ml`, and `go-rag`, and skip-safely scan local `go-ai`, `go-ml`, and `go-inference` checkouts for direct `go-rocm` runtime imports.
- Non-hardware module gates pass from `go/`: default, Linux no-cgo, and `rocm_legacy_server`.

---

## Known Limitations

**Token IDs**: `inference.Token.ID` is always zero. llama-server's OpenAI-compatible streaming API does not return token IDs. Adding token IDs would require `logprobs: true` in the request and additional parsing overhead. No current consumer uses token IDs.

**StopTokens**: `GenerateConfig.StopTokens []int32` is ignored. llama-server's `/v1/completions` and `/v1/chat/completions` endpoints accept stop sequences as strings (`"stop": [...]`), not token IDs. Mapping between them requires a tokeniser that is not available in this package. No current consumer uses StopTokens.

**Err() concurrency**: `rocmModel.Err()` returns the last error from any Generate/Chat call. With multiple concurrent callers, errors can be overwritten. The single `Err() error` method is an go-inference interface constraint, not a go-rocm decision. Per-call error returns would require an interface change in go-inference.

**VRAM reads are non-atomic**: `GetVRAMInfo()` reads `mem_info_vram_total` and `mem_info_vram_used` in two separate sysfs reads. Under heavy VRAM allocation churn, transient inconsistency is possible. `Free` is clamped to zero to prevent uint64 underflow.

**Model directory**: Models are on an SMB mount (`/data` = `//10.69.69.108/Data`). Integration tests and benchmarks skip when the mount is unavailable. For offline testing, copy a small model (the 1B Q5_K_M is approximately 0.9 GB) to a local path and update the `testModel` constant in `rocm_integration_test.go`.

**Single-model-per-server**: Each `rocmModel` owns exactly one llama-server subprocess. Loading multiple models simultaneously requires multiple `LoadModel` calls, each consuming its own VRAM share. There is no shared server or model-switching mechanism.

---

## Future Considerations

**Direct HIP CGO** (Phase 5, unscheduled): Direct HIP CGO bindings would eliminate the HTTP overhead and process boundary. Only worth pursuing if the subprocess approach becomes a measurable bottleneck. Estimated cost: months of implementation to match llama.cpp's model support breadth.

**vLLM backend** (Phase 5, unscheduled): vLLM supports ROCm and provides better batching semantics for high-throughput scenarios. Would be a parallel subprocess backend alongside llama-server, selectable via configuration.

**Model-switching**: The current design loads one model per server instance. A pool-based approach could share llama-server instances across model loads, though this would require llama-server to support hot-swapping models (it does not currently).

**go-i18n integration**: go-i18n may use go-rocm for batch text classification on the Linux homelab once Phase 2 of go-i18n is unblocked. The `WithParallelSlots` option makes the backend well-suited for batch workloads.
