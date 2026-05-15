# go-rocm Feature Notes

This directory is the short-form feature and performance record for the native
ROCm package. The longer architecture and development guides remain in
`docs/`.

## Current Feature Surface

- Native `rocm` backend registration through `go-inference`.
- Linux `amd64` HIP runtime with explicit unavailable stubs on non-Linux or
  non-amd64 platforms.
- GGUF and safetensors model-pack inspection, including Gemma4 text aliases,
  nested `text_config`, sharded safetensors metadata, BF16 side tensors, packed
  U32 q4 weights, BERT embedding/rerank/classifier hints, MoE, JANGTQ, and
  codebook metadata.
- Scheduler and cancellation wrapper with bounded queueing, request IDs,
  direct `Generate`/`Chat` scheduling, non-streaming `Classify` and
  `BatchGenerate`, and scheduler probe events.
- OpenAI chat/responses, Anthropic Messages, Ollama chat/generate, capability,
  cache, cancel, embeddings, and rerank HTTP adapters over shared
  `go-inference` DTOs.
- Parser registry for reasoning/tool-call formats across Qwen, Gemma,
  MiniMax, DeepSeek, GPT-OSS, Mistral, Kimi, GLM, Hermes, Granite, generic XML,
  and generic JSON.
- Metadata-first prompt cache and portable package-local KV snapshots with
  optional HIP device remirroring.
- URI-first wake/sleep/fork state sessions and metadata-only state bundles.
- Experimental HIP launch fixtures for projection, MLX q4 projection,
  JANGTQ/codebook/LoRA projection, embedding lookup/mean-pool, rerank cosine,
  RMSNorm, RoPE, greedy sampling, attention, vector ops, SwiGLU, MoE router,
  lazy expert residency, tiny prefill/decode, cross-entropy, distillation KL,
  and GRPO advantage.
- Live Gemma4-E2B coverage:
  - BF16 pack as the raw tensor correctness anchor.
  - MLX-q4 pack as the faster experimental Generate/Chat/Classify/Benchmark/Eval
    development path.

## Coverage Policy

`codecov.yml` sets a 90% project target. It ignores implementation-heavy HIP
kernel files and large binary-format/model-pack helpers that are covered by
separate opt-in fixture and hardware shards but otherwise dominate the statement
denominator with malformed-format and driver edge branches.

The default local coverage command is:

```sh
go test ./go/... -coverpkg=./go/... -coverprofile=/tmp/go-rocm.cover -covermode=atomic -count=1
```

For parity with the Codecov ignore set, filter the local profile with:

```sh
awk 'NR==1{print; next} $1 !~ /\/hip_/ && $1 !~ /internal\/gguf/ && $1 !~ /model_pack.go/ && $1 !~ /embedding_model.go/ {print}' \
  /tmp/go-rocm.cover > /tmp/go-rocm-codecov.cover
go tool cover -func=/tmp/go-rocm-codecov.cover | tail -n 1
```

The current default filtered profile reports:

```text
total: (statements) 90.6%
```
