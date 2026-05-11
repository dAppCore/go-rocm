# Bring go-rocm Into go-mlx Runtime Parity

> For `/goal` agents: execute this file as the mission contract. Read `RFC.md`
> first, then work down this checklist with TDD. Preserve user changes. Do not
> import `go-mlx`.

## Goal

Make `go-rocm` the ROCm sibling of `go-mlx`: a native package-first inference
backend that implements the shared `go-inference` contracts for model loading,
metadata inspection, scheduling, cancellation, cache services, parser services,
state wake/sleep/fork groundwork, probes, evaluation, and benchmarks before
expanding model families and training kernels.

## Hard Boundaries

- `go-rocm` must not import `go-mlx`.
- `go-rocm` may import only shared Core modules such as `dappco.re/go`,
  `dappco.re/go/inference`, and package-local internals.
- Use CoreGO style. Prefer `core.E`, Core filesystem/path/JSON/string helpers,
  and `core.Result` where the local package API expects result values.
- Keep old subprocess/HTTP `llama-server` code behind the
  `rocm_legacy_server` build tag only.
- Do not add UI, TUI, provider policy, external API keys, or `go-ai` routing.
- Use local `go.work` during Core development. Do not force `GOWORK=off` while
  unpublished local `go-inference` contracts are linked.
- Hardware tests must be opt-in and skip cleanly without ROCm hardware.

## Current Baseline

The repo already has useful groundwork:

- Native package shape in `go/native.go`.
- HIP availability, memory, malloc/free, and host-to-device copy in
  `go/hip_runtime.go` and `go/hip_driver_cgo.go`.
- GGUF and safetensors sidecar inspection in `go/model_pack.go`.
- Capability reporting for the expanded shared IDs, mostly as planned.
- ROCm 16GB memory planning with compact cache and MoE lazy-expert hints.
- Bench/eval/probe reporting stubs over the model surface.

Do not delete or simplify this groundwork to make tests easier.

## Definition of Done

This goal is complete when all of the following are true:

- `go-rocm` compiles and tests on the developer Mac without ROCm hardware.
- Linux `amd64` safe builds compile without cgo.
- Linux `amd64` cgo builds expose clear HIP availability and kernel status.
- `go-rocm` implements or deliberately reports every shared capability used by
  `go-mlx`, with accurate supported/experimental/planned status.
- `go-rocm` exposes package-first APIs for scheduler, cancellation, cache stats,
  cache warm/clear, parser registry, model-pack inspection, bench, eval, probes,
  and state lifecycle groundwork.
- Native HIP decode/prefill work returns clear "kernel not linked" errors until
  a kernel is actually implemented.
- No higher-level workflow package imports concrete runtime packages directly.
- All new public surfaces have Good/Bad/Ugly tests and examples matching the
  repo style.

## Work Order

- [ ] Phase 0: Snapshot the tree and establish the baseline.
  - Run `git status --short`.
  - Read `AGENTS.md`, `README.md`, `docs/architecture.md`,
    `external/go-inference/go/capability.go`, and
    `external/go-inference/go/contracts.go`.
  - Run `go test ./... -count=1`.
  - Run `GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1`.
  - Record failures in the working notes before editing.

- [ ] Phase 1: Synchronise shared contracts.
  - Ensure local `external/go-inference/go` contains the contract primitives
    required by `go-mlx`: capability IDs, optional interfaces, OpenAI service
    helpers, parser interfaces, cache interfaces, benchmark/eval structs, and
    the `state` package.
  - Add only backend-neutral contracts to `go-inference`.
  - Add focused `go-inference` tests before touching `go-rocm`.

- [ ] Phase 2: Make ROCm capability reports honest and complete.
  - Update `go/native_contract_test.go` first.
  - Ensure `rocmBackend.Capabilities()` and `rocmModel.Capabilities()` report
    all shared feature IDs with correct status.
  - Supported means usable today. Experimental means callable with known limits.
    Planned means visible to planners but not callable.

- [ ] Phase 3: Model-pack inspection parity.
  - Harden `go/model_pack.go` for GGUF, safetensors headers, `config.json`,
    tokenizer metadata, `jang_config.json`, `codebook_config.json`, architecture
    aliases, quantization aliases, context limits, and memory-fit labels.
  - Add fixtures for MiniMax/JANGTQ, Qwen3, Gemma, Mistral/Mixtral, Phi,
    DeepSeek, GPT-OSS, and BERT metadata.

- [ ] Phase 4: Scheduler and cancellation.
  - Add a ROCm scheduler wrapper implementing `inference.SchedulerModel` and
    `inference.CancellableModel`.
  - Include bounded queueing, request IDs, cancellation before prefill,
    cancellation during decode, queue latency, first-token latency, and probe
    events.

- [ ] Phase 5: Parser registry.
  - Add ROCm reasoning/tool parser support for Qwen, Gemma, MiniMax, DeepSeek
    R1, GPT-OSS, Mistral, Kimi, GLM, Hermes, Granite, and generic XML/JSON.
  - Wire parser capabilities to loaded model metadata and stream output.

- [ ] Phase 6: Cache and state groundwork.
  - Add a ROCm `inference.CacheService` implementation with cache stats,
    warm, clear, block identity, adapter compatibility, tokenizer compatibility,
    and memory/disk accounting.
  - Wire `go-inference/state` wake/sleep/fork primitives without coupling to
    `go-mlx` or `go-ai`.
  - Keep memvid-compatible references URI-first and runtime-owned.

- [ ] Phase 7: HIP runtime stepping stones.
  - Keep allocation/copy tests green.
  - Add tensor shape/dtype validation before kernels.
  - Implement prefill/decode kernels one narrow path at a time:
    1. fp16/q8 toy forward fixture.
    2. Qwen3 or Gemma small decode smoke.
    3. KV cache ownership.
    4. q8 and k-q8-v-q4 KV cache modes.
    5. MoE router and lazy expert residency.
    6. JANGTQ/MXTQ packed dequant/projection.

- [ ] Phase 8: Bench, eval, and probes.
  - Make ROCm bench reports use the same fields as MLX: prefill tok/s, decode
    tok/s, memory peak, cache hit rate, restore time, LoRA overhead, perplexity
    hooks, queue latency, first-token latency, and probe counts.
  - Add fake-device deterministic tests and opt-in hardware smoke tests.

- [ ] Phase 9: Documentation and final gates.
  - Update `README.md`, `docs/architecture.md`, `docs/development.md`, and
    `docs/history.md`.
  - Run all non-hardware gates.
  - On the Linux ROCm machine, run the hardware gates from `RFC.md`.

## Required Test Gates

Run these before handoff:

```sh
go test ./... -count=1
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test ./... -count=1
go test -tags rocm_legacy_server ./... -count=1
```

Run these on the ROCm machine when hardware work begins:

```sh
GO_ROCM_RUN_HIP_TESTS=1 go test ./go -run 'TestHIP|TestNative' -count=1
GO_ROCM_RUN_MODEL_TESTS=1 go test ./go -run 'Test.*Smoke|Test.*Generate|Test.*Decode' -count=1
```

## Handoff Notes

- `RFC.md` is the source of technical detail.
- Keep changes small and commit by phase.
- If a feature cannot be supported yet, add the exact planned capability status,
  a clear error path, and a test proving the failure message.
