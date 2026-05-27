# go-rocm

Native AMD ROCm backend for the Core Go inference stack. The default package surface is now the ROCm sibling of `go-mlx`: it implements the shared `go-inference` contracts, exposes model-fit planning, scheduler/cancellation wrappers, an in-memory cache service with opaque disk refs, parser registry, decode-optimisation helpers, state lifecycle groundwork, probing, benchmarking, dataset eval hooks, tokenizer/adapter surfaces, GGUF/safetensors model-pack inspection, sysfs VRAM monitoring, model discovery, and a native HIP weight loader without routing inference through an OpenAI-compatible HTTP server.

The former managed `llama-server` implementation is retained only behind the explicit `rocm_legacy_server` build tag while native kernels are filled in on ROCm hardware. Platform-restricted runtime code is `linux/amd64`; safe stubs compile everywhere else and register an unavailable `rocm` backend so blank imports stay deterministic on developer Macs without ROCm.

**Module**: `dappco.re/go/rocm`
**Licence**: EUPL-1.2
**Language**: Go 1.26

## Quick Start

```go
import (
    "dappco.re/go/inference"
    _ "dappco.re/go/rocm" // registers "rocm" backend via init()
)

model, err := inference.LoadModel("/path/to/model.gguf")
defer model.Close()

for tok := range model.Generate(ctx, "Hello", inference.WithMaxTokens(256)) {
    fmt.Print(tok.Text)
}
```

With cgo enabled on Linux, the native backend dynamically opens `libamdhip64` and loads validated GGUF tensors or safetensors model packs into device memory; the safetensors path covers Gemma4-E2B-style nested text metadata, Gemma4 text-model aliases (`gemma4_text`, `Gemma4ForCausalLM`, and `Gemma4TextForCausalLM`), tied embeddings, BF16 side tensors, sharded payload files, and packed U32 4-bit weights. If cgo is disabled, HIP is missing, or production decode/prefill kernels are not linked yet, failures are explicit rather than falling back to a subprocess. A tiny loaded-model fixture with f32 embeddings and f32/f16/raw-q8/JANGTQ/codebook output heads can use `GO_ROCM_KERNEL_HSACO` for prefill/decode smoke coverage, packed/codebook output logits through `rocm_jangtq_projection`, `rocm_codebook_lookup`, and `rocm_projection`, experimental model-level embedding/rerank calls, package-local toy cross-entropy/distillation/GRPO loss hooks, and narrow experimental `rocm-tiny-lora`, Qwen/Gemma small LM-head LoRA, and BERT classifier LoRA adapter paths through `rocm_lora_projection`; BERT-style f32 word-embedding-only packs can run experimental mean-pool embeddings and cosine rerank without an LM head, and BERT sequence-classification packs with f32/f16 `classifier.weight` or `score.weight` can score rerank pairs through the projection path; a fixture-scale Qwen/Gemma typed decode path can read a loaded f32 embedding row, compose the first-layer primitive kernels, append package-local KV, and append only the decoded token page to supplied device KV mirrors. Capability, benchmark, and eval quality-probe labels distinguish toy tiny fixture kernels with `kernel_scope=toy_tiny_fixture` and `production_decode=not_linked`/`production_prefill=not_linked`; embedding/rerank capability labels mark loaded embedding and rerank fixtures with production model integration still not linked; LoRA capability labels similarly mark `kernel_scope=loaded_adapter_fixtures` and `production_adapter_application=not_linked`. General production model-family generation remains planned, but a loaded Gemma4 MLX-q4 pack, including `gemma4_text` text-model aliases, exposes an experimental package-local Generate/Chat/BatchGenerate/Classify/Benchmark/Eval route through descriptor-backed k-q8-v-q4 KV state, with speculative and prompt-lookup helpers over that q4 Generate path; those reports still label production decode, prefill, and KV cache backing as `not_linked`. Use `-tags rocm_legacy_server` only when intentionally testing the old server path.

## Status

- Supported: backend registration, model-pack inspection for packs with valid weight and architecture metadata, native HIP weight loading for GGUF and validated safetensors packs, including BERT embedding/rerank/classifier metadata hints, memory planning, scheduler/cancellation wrapper, parser registry, and state/cache metadata services.
- Experimental: fallback tokenizer/chat template handling, probe events, loaded Gemma4 MLX-q4 Generate/Chat/BatchGenerate/Classify/Benchmark/Eval over package-local packed tensor launches and descriptor-backed k-q8-v-q4 KV state, including `gemma4_text` text-model aliases from the current `go-mlx` dev convention, shared speculative/prompt-lookup decode helpers over the q4 Generate path, q4 Classify logit probes, metadata-only state bundle capture/restore with owned-runtime cleanup on wake/restore/close replacement, classification logit/entropy probe emission when logits are requested, benchmark/eval wrappers with measured latency/duration labels and measured probe counts, active-adapter LoRA overhead labels, and batched/native-fixture classification-logit loss/perplexity hooks, metadata/package-local prompt cache warm/clear plus `go-inference/state` metadata refs, portable package-local KV snapshot disk refs, exact cold disk-ref rehydrate, and best-effort HIP device remirror for warmed/cold-restored KV snapshots, URI-first wake/sleep/fork state refs, package-local KV snapshot refs, HIP KV device-mirror allocation/copy smoke coverage with incremental decoded-token page appends, portable device-mirror KV snapshots, and loaded-model best-effort wake/fork remirror, a fixed descriptor byte layout, device-resident descriptor table, KV launch descriptor, prefill/decode/projection/JANGTQ-projection/codebook-lookup/LoRA-projection/embedding-mean-pool/rerank-cosine/RMSNorm/RoPE/greedy-sampler/attention/MoE-router/MoE-lazy-experts/tiny-prefill/tiny-decode/cross-entropy/distillation/GRPO launch packets, token/projection/transformer buffer upload helpers, fake prefill/decode packet memory validation, device-to-host projection and transformer primitive readback in fake launches, optional fake-testable kernel launch configs, cgo HSACO module launch plumbing via `GO_ROCM_KERNEL_HSACO`, a buildable `kernels/rocm_kernels.hip` fp16/q8/f32 projection smoke kernel selectable by loaded HIP models plus composed Qwen/Gemma small decode smokes using fixture and loaded device-resident tensor weights, typed loaded-model Qwen/Gemma decode smoke that derives the input vector from a loaded f32 embedding row and appends package-local plus supplied device KV, tiny and BERT-style loaded f32 embedding mean-pool/rerank calls over f32 token/word embeddings including sequence-classifier rerank scoring through f32/f16 `classifier.weight`/`score.weight` with capability labels scoped to loaded embedding/rerank fixtures, a tiny loaded-model f32-embedding prefill/decode/generate fixture with f32/f16/raw-q8/JANGTQ/codebook output heads, package-local loaded-model toy cross-entropy/distillation/GRPO loss hooks, experimental tiny loaded-model `rocm-tiny-lora` output-head adapter application, experimental Qwen/Gemma small LM-head LoRA adapter application, experimental BERT sequence-classifier LoRA adapter application with capability labels scoped to loaded adapter fixtures, prefill/decode packet-consumer smokes across fp16, q8, and k-q8-v-q4 cache modes with device-written status markers, compiled embedding mean-pool, rerank cosine, RMSNorm/RoPE/greedy-sampler/attention/MoE-router/MoE-lazy-experts/JANGTQ-projection/codebook-lookup/LoRA-projection/cross-entropy/distillation/GRPO primitive smokes, compiled toy tiny-prefill and tiny-decode fixtures for logits/attention/greedy/KV readback with fp32/fp16/q8 output heads plus loaded packed/codebook output logits through `rocm_jangtq_projection`, `rocm_codebook_lookup`, and `rocm_projection`, and metadata-only MoE/JANGTQ/codebook recognition (`runtime_status=metadata_only`) with fixture-kernel labels and pending production integration labels.
- Planned: native decode/prefill kernels, fully HIP-owned KV pages and snapshots, disk-backed HIP KV cache, production LoRA model-family application/training, production embedding/rerank model-family integration, production MoE router/lazy-expert integration, production JANGTQ/codebook integration, and production model-family loss/perplexity. Training capabilities report `runtime_status=planned`, `training_kernel=not_linked`, `training_interface=not_implemented`, and a `required_kernel` label so callers can distinguish planning hints from callable training APIs.

## Retained State Contract

Retained generation is state-first. The previous turns live in a durable `.kv`
state ref managed through `go-inference/state`; in current development that ref
is an MP4-style vector stream rather than rebuilt prompt text. Wake/restore must
fail when the KV ref is missing or incompatible. It must never rebuild a session
by concatenating the old prompt, manuscript, or chat transcript.

The Gemma4 10-turn book benchmark follows that contract: each turn restores the
session state, appends only the new Gemma4 chat turn and distractor, writes the
new generated tokens back into state, and captures the full generated book for
arc inspection. Any replay baseline is explicitly unsafe/debug-only.

## Documentation

- [Architecture](docs/architecture.md) — native backend direction, GGUF parser, VRAM monitoring, and legacy server notes
- [Development Guide](docs/development.md) — prerequisites, test commands, benchmarks
- [Project History](docs/history.md) — completed phases, commit hashes, known limitations

## Build & Test

```bash
go test ./... -count=1
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
go test -tags rocm_legacy_server ./... -count=1
```

The repo root contains a workspace gate that delegates native test runs to the
`go/` module and shared `external/go-inference/go` contracts. You can still run
the same commands from `go/` when iterating on only the ROCm module.

On macOS without ROCm hardware, run the real stub tests from the repo root:

```bash
go test ./go/... -count=1
```

From a Linux ROCm host, the Darwin stub compile proxy is:

```bash
GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go test -exec=/bin/true ./go/... -count=1
```

## Licence

European Union Public Licence 1.2 — see [LICENCE](LICENCE) for details.
