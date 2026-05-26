# Book Workload

This is the production acceptance workload for Gemma4 q4 stateful generation on
ROCm. Synthetic 2k/4k/29k prompt benchmarks are diagnostics only; the driver is
done when this 10-turn profile is fast and coherent.

## Profile

1. Turn 1 asks for chapter 1 of a book, without declaring the final chapter
   count up front, from
   `C001_STORY_PERSPECTIVE`: a lighthouse keeper discovers the light has been
   signalling to something in the deep ocean for centuries.
2. Turns 2-10 ask for more chapters by requesting the next chapter.
3. Each follow-up turn injects one fresh creative distractor prompt from the
   prompt bank. The distractor must not replace the chapter-1 story arc, and
   the model should not use the distractor's subject, title, imagery, form, or
   central concept in the chapter.
4. Chapter 10 must still preserve the lighthouse keeper, signalling light, and
   deep-ocean entity arc.
5. The implementation should retain and extend KV state between turns instead
   of replaying the entire manuscript/session prompt.

## Acceptance

- `<=90s` wall time for all 10 turns: success.
- `<=110s` wall time with chapter-10 arc retention: production-candidate; flip
  the best route toward defaults and start removing obsolete slow paths.
- `>110s`: remains experimental/tuning.

The Metal/go-mlx reference completes the 10-turn book profile in about `82s`.
The current ROCm retained-state default uses one-token Gemma4 device-KV pages
and has a passing `74.2s` full-cap run on the RX 7800 XT, but later-turn decode
still decays below the `90-100+ tok/s` driver target.

## ROCm Benchmark

The current ROCm benchmark is a replay baseline so the workload has a stable
measurement surface before retained-state generation is wired into public
Generate. It is intentionally double-gated because full replay can monopolize
the display GPU and is not the path being optimized:

```sh
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_BOOK_BENCHMARKS=1 \
GO_ROCM_RUN_UNSAFE_REPLAY_BOOK_BENCHMARKS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_BOOK_CONTEXT_LEN=48000 \
GO_ROCM_BOOK_CHAPTER_TOKENS=0 \
GO_ROCM_BOOK_TURN_TIMEOUT_SECONDS=60 \
go test ./go -run '^$' -bench '^BenchmarkInferenceGemma4Q4Book10Turn_ReplayBaseline$' -benchmem -benchtime=1x -count=1 -timeout=0
```

When retained-state generation lands, keep the benchmark name stable or add a
parallel retained-state benchmark with the same reported metrics:
`book_wall_s/op`, `book_90s_success`, `book_110s_production_candidate`,
`book_prompt_tokens/op`, `book_generated_tokens/op`, `B/op`, and `allocs/op`.
Use `GO_ROCM_BOOK_TURNS` for short smoke runs. The retained benchmark defaults
to a full-chapter safety cap derived from context length and turn count. With
`GO_ROCM_BOOK_CONTEXT_LEN=48000` and 10 turns, omitted
`GO_ROCM_BOOK_CHAPTER_TOKENS` or `GO_ROCM_BOOK_CHAPTER_TOKENS=0` allows up to
4390 generated tokens per chapter; smaller explicit values such as `512` are
smoke/debug caps and are not the acceptance workload. The retained benchmark
defaults to `GO_ROCM_BOOK_PREFILL_UBATCH_TOKENS=16` to keep individual HIP
kernels short enough for the display-attached RX 7800 XT; raise it only during
deliberate watchdog/error-log profiling. `GO_ROCM_BOOK_LAYERS` is a debug-only
layer cap for proving the retained state path on a small graph; omit it for
acceptance. Set `GO_ROCM_BOOK_WARMUP_PROMPT` to run a discarded prefill before
timing; this warms the engine without sampling a response or entering the
retained book state. Set `GO_ROCM_BOOK_OUTPUT_FILE` to write generated chapters
for arc inspection. Set `GO_ROCM_BOOK_TURN_TIMEOUT_SECONDS=0` only for
deliberate full-run profiling.
