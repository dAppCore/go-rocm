# Inference Benchmark Notes

Benchmark host: Linux `amd64`, Ryzen 9 9950X, RX 7800 XT pinned with
`ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85`.

Model pack:

- `/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit`

Kernel module:

- `/tmp/go-rocm-kernels-gfx1100.hsaco`

Command:

```sh
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_BENCHMARKS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_BENCH_PROMPT='text:Hi' \
GO_ROCM_BENCH_TOKENS=1 \
go test ./go -run '^$' -bench BenchmarkInferenceGemma4Q4Generate -benchtime=1x -count=1 -timeout=10m
```

Current endpoint command:

```sh
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_BENCHMARKS=1 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
GO_ROCM_BENCH_TOKENS=2000 \
go test ./go -run '^$' -bench BenchmarkInferenceGemma4Q4Generate -benchmem -count=1
```

Current corrected result from 2026-05-17:

```text
2000 tokens: BenchmarkInferenceGemma4Q4Generate-32  1  19629810772 ns/op  2000 max_tokens/op  101.9 tok/s  2000 tokens  251745216 B/op  2730011 allocs/op
2048 tokens: BenchmarkInferenceGemma4Q4Generate-32  1  20156670616 ns/op  2048 max_tokens/op  101.6 tok/s  2048 tokens  258176312 B/op  2795040 allocs/op
512 tokens:  BenchmarkInferenceGemma4Q4Generate-32  1   4363243733 ns/op   512 max_tokens/op  117.3 tok/s   512 tokens
```

The current endpoint uses the benchmark's `inference.WithContextLen(128)` load
setting as part of the measured configuration. A later driver/HIP pass fixed
Gemma4 q4 layer setup so sliding-window layers honor that requested context
size while full-attention layers remain uncapped. Older results below are kept
as chronology; the former `58-79 tok/s` plateau and an intermediate 512-token
`100+` result were superseded by three fixes: corrected q4 projection/final
greedy launch geometry, default launch-argument sync-on-wrap instead of
per-packet HIP events, and the q4 effective sliding-window context fix.

## Book Workload Acceptance

The production endpoint is now the 10-turn `book.md` workload, not a synthetic
prompt length. The benchmark asks for chapter 1 of a lighthouse/deep ocean book
without declaring the final chapter count up front, then asks for more chapters
while adding one creative distractor prompt each turn. The distractor is
included only as state-retention interference; the prompt tells the model not
to use the distractor's subject, title, imagery, form, or central concept.
Chapter 10 must still retain the original story arc.

The Metal/go-mlx reference completes that retained-state book profile in about
`82s`. ROCm accepts `<=90s` as success and `<=110s` as production-candidate if
chapter 10 still contains the original arc. Above `110s`, the route remains
experimental/tuning.

Current retained-state ROCm default uses one-token Gemma4 device-KV pages. A
fresh full 10-turn retained sampled run on the RX 7800 XT with the 512-token
prefill ubatch default completed in `46.11s` wall, `38.29s` decode, `3272`
generated tokens, `70.96 tok/s` average decode, `64.66 tok/s` turn-10 decode,
empty stderr, no max-token hits, and chapter-10 arc retention. This is accepted
wall-time evidence, but later-turn decode still decays below the `90-100+ tok/s`
driver target.

Current ROCm benchmark surface:

```sh
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_BOOK_BENCHMARKS=1 \
GO_ROCM_RUN_RETAINED_BOOK_BENCHMARKS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100-current.hsaco \
GO_ROCM_BOOK_CONTEXT_LEN=48000 \
GO_ROCM_BOOK_CHAPTER_TOKENS=0 \
GO_ROCM_BOOK_PREFILL_UBATCH_TOKENS=512 \
GO_ROCM_BOOK_TURN_TIMEOUT_SECONDS=60 \
GO_ROCM_BOOK_OUTPUT_FILE=/tmp/go-rocm-book-retained.md \
go test ./go -run '^$' -bench '^BenchmarkInferenceGemma4Q4Book10Turn_RetainedState$' -benchmem -benchtime=1x -count=1 -timeout=0 \
  2>/tmp/go-rocm-book-retained.err
```

This benchmark is retained-state only and is double-gated because it can
monopolize the display GPU. Each chapter turn appends only the new request and
distractor while the `.kv` state carries the book so far. Rebuilding the
manuscript by replaying prompt text is an error, not a fallback.
`GO_ROCM_BOOK_TURNS` can be used for short smoke runs. Retained book runs
default to a full-chapter safety cap derived from context length and turn count. With
`GO_ROCM_BOOK_CONTEXT_LEN=48000` and 10 turns, omitted
`GO_ROCM_BOOK_CHAPTER_TOKENS` or `GO_ROCM_BOOK_CHAPTER_TOKENS=0` allows up to
4390 generated tokens per chapter; smaller explicit values such as `512` are
smoke/debug caps and are not the acceptance workload.
`GO_ROCM_BOOK_PREFILL_UBATCH_TOKENS` defaults to `512`, matching the Gemma4 q4
production prefill path. Drop to `16` only for display-safety debugging with an
error log/watchdog.
`GO_ROCM_BOOK_LAYERS` is a debug-only layer cap for proving retained-state
mechanics on a small graph; acceptance runs omit it. `GO_ROCM_BOOK_WARMUP_PROMPT`
runs a discarded prefill before timing, so it warms the engine without sampling
a response or entering the retained book state. `GO_ROCM_BOOK_OUTPUT_FILE`
writes generated chapters for arc inspection. `GO_ROCM_BOOK_TURN_TIMEOUT_SECONDS=0`
disables the per-turn timeout only for deliberate full-run profiling.

## Long-Prompt Diagnostic Checks

The short-generation endpoint uses a 128-token benchmark context and proves the
decode loop can exceed `100 tok/s`. It does not prove real harness prefill
throughput. Gemma4 E2B/E4B context windows are 128k tokens, so the active
long-load ladder uses a `48000` context and synthetic `tokens:` prompts:

```sh
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_BENCHMARKS=1 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
GO_ROCM_BENCH_CONTEXT_LEN=48000 \
GO_ROCM_BENCH_PROMPT_TOKEN_COUNT=32000 \
GO_ROCM_BENCH_TOKENS=1 \
go test ./go -run '^$' -bench '^BenchmarkInferenceGemma4Q4Generate$' -benchtime=1x -benchmem -count=1 -timeout=0
```

`GO_ROCM_GEMMA4_Q4_PREFILL_UBATCH_TOKENS` controls the integration planner's
prompt ubatch size and defaults to `512`, matching the llama.cpp default
`n_ubatch`. As of this note, the planner only structures the prompt loop and
output mask. The first reusable batched primitives now launch one multi-token
embedding lookup, one `batch*hidden` scale operation, one row-wise input RMSNorm
grid using `rocm_rms_norm_heads`, and batched MLX q4 projection via
`rocm_mlx_q4_projection_batch` with prompt rows mapped onto `GridY`. The MLP
side now has `rocm_mlx_q4_gelu_tanh_multiply_batch` plus a batched down
projection helper, and the attention side has a first
`rocm_attention_heads_batch_causal` primitive for one-layer descriptor-backed
KV attention output. The one-layer scaffold now also chains batched output
projection, residual/norm steps, MLP, the Gemma4 per-layer input branch, and
final hidden output, and can sample only the selected final prompt row through
final RMSNorm plus fused q4 LM-head softcap+greedy. A first all-layer helper now
drives those layer pieces across a token span, accepts per-layer input matrices
for each layer, and has an initial prior-ubatch append path that carries
descriptor-backed device KV into the next ubatch with the attention
`query_start_token` set to the ubatch start. These pieces do not yet execute the
full batched GPU prefill graph; the prior append path is borrow-based and does
not solve the production full/SWA slot layout.

Current staged results:

```text
prompt  go-rocm result                         prompt tok/s  allocation
2k      20962474329 ns/op                      95.41         251449568 B/op
4k      48788445708 ns/op                      81.99         684743344 B/op
8k      123624998497 ns/op                     64.71         1786065632 B/op
16k     349018316246 ns/op                     45.84         3569176536 B/op
32k     1072604641783 ns/op                    29.83         8239230448 B/op
48k     interrupted after 974s without result  incomplete    pre-refinement run
```

The 32k run used the discrete RX 7800 XT: `rocm-smi` showed GPU[0]
`AMD Radeon RX 7800 XT` busy while the onboard GPU stayed at `0%`.

For an external implementation reference, upstream llama.cpp was cloned to
`/tmp/llama.cpp`, built locally with HIP for `gfx1100`, and run against the
Hugging Face Gemma4 GGUF downloaded outside the LEM model tree:

```text
/home/claude/models/hf/unsloth-gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf
```

Command:

```sh
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
/tmp/llama.cpp/build-hip-gfx1100/bin/llama-bench \
  -m /home/claude/models/hf/unsloth-gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf \
  -p 2048,4096,8192,16384,32768 \
  -n 0 -b 2048 -ub 512 -fa 1 -ngl 99 -r 1 -o md
```

llama.cpp results on the same card:

```text
pp2048   4788.12 tok/s
pp4096   4519.58 tok/s
pp8192   4071.68 tok/s
pp16384  3414.32 tok/s
pp32768  2580.52 tok/s
```

The long-load bottleneck is therefore structural. The ROCm path is still
feeding prompt tokens through `hipRunGemma4Q4SingleTokenForward...`; llama.cpp
uses batched prompt processing, `n_batch`/`n_ubatch` splitting, slot-based KV
preparation, separate full/SWA caches for hybrid attention, output flags so only
requested tokens emit logits, batched quantized matmul, and flash-attention
kernels. Reaching the actual-load target requires a real Gemma4 q4 batched
prefill path, not another single-token decode micro-optimization.

Result from 2026-05-13:

```text
BenchmarkInferenceGemma4Q4Generate-32  1  4254553228 ns/op  1.000 max_tokens/op  0.2350 tok/s  1.000 tokens  349845704 B/op  83394 allocs/op
```

After caching HSACO modules/function handles and reusing the small device
launch-argument packet, the same 1-token benchmark on 2026-05-16 reports:

```text
BenchmarkInferenceGemma4Q4Generate-32  1  1151848888 ns/op  1.000 max_tokens/op  0.8682 tok/s  1.000 tokens  89530672 B/op  79132 allocs/op
```

The 8-token benchmark on 2026-05-16 reports:

```text
BenchmarkInferenceGemma4Q4Generate-32  1  11579650685 ns/op  8.000 max_tokens/op  0.6909 tok/s  8.000 tokens  369906856 B/op  336178 allocs/op
```

After the first device-resident q4 MLP step, attention-output concat,
small-buffer pooling, batched RoPE-query upload, multi-head attention launch,
and q4 validation allocation cleanup, the short benchmark ladder on 2026-05-16
reports:

```text
BenchmarkInferenceGemma4Q4Generate-32  2   738894378 ns/op  1.000 max_tokens/op  1.353 tok/s  2.000 tokens   68661432 B/op    61727 allocs/op
BenchmarkInferenceGemma4Q4Generate-32  1  4011068954 ns/op  8.000 max_tokens/op  1.994 tok/s  8.000 tokens  182386392 B/op   248860 allocs/op
```

Long-window checks on 2026-05-16 did not support a timing-artifact theory:

```text
GO_ROCM_BENCH_TOKENS=2000 was interrupted after 599s without completion.
GO_ROCM_BENCH_TOKENS=64 was interrupted after 573s without completion.
```

After the multi-head attention launch, the 64-token run completed but still
collapsed with context growth:

```text
BenchmarkInferenceGemma4Q4Generate-32  1  197371659447 ns/op  64.00 max_tokens/op  0.3243 tok/s  64.00 tokens  1480160960 B/op  1764878 allocs/op
```

After q4 projection parallelization, device KV descriptor lookup fixes,
shared-device KV, sliding-window device appends, device per-layer input
precompute, async launch-argument staging, unit-weight RMSNorm, device KV token
encoding, device descriptor append, partial-RoPE/global device chaining, shared
descriptor borrowing, parallel RMSNorm/RoPE, token-parallel long-context
attention heads, 64-thread q4 projection row blocks, fused final q4
projection+greedy sampling, grouped q4 projection rows, and dynamic shared
attention weights through 2048 tokens, plus queued `hipMemsetAsync` for the
final greedy result clear, final greedy result buffer reuse, packed-word q4
projection, affine q4 accumulation, an eight-row q4 projection block, batched
query/per-layer setup, two local q4 MLP fusions, and RMSNorm+residual-add
fusion, device-KV key-dot scale hoisting, local device-KV value page/scale
reuse in token-parallel attention, and explicit query dimensions for
device-query attention, q4 projection shuffle reductions, layer-scalar fusion,
inlined q4/q8 value loads in token-parallel attention, and fused non-shared q4
Q/K/V projection, then fixing/defaulting mapped launch-argument packets,
parallelizing the common no-trim device KV descriptor append path, caching each
attention query in shared memory for the score phase, consuming q8 key dots four
values per packed word, using 512-thread attention-head blocks plus paired q4
device-KV value accumulation for contexts of at least 512 tokens, caching q4
value page pointers/scales in dynamic shared memory, switching attention-head
device-KV validation to header-only checks, normalizing softmax weights once,
and hoisting direct one-token page addressing, the then-current 2026-05-16 q4 ladder
is:

```text
1 token:     24482634 ns/op, 40.85 tok/s, 4414351 B/op, 22246 allocs/op
8 tokens:   101731908 ns/op, 78.64 tok/s, 7447574 B/op, 83144 allocs/op
64 tokens:  744144350 ns/op, 86.00 tok/s, 39139076 B/op, 567615 allocs/op
512 tokens: 7959242091 ns/op, 64.33 tok/s, 214885656 B/op, 4462676 allocs/op
2000 tokens: 34170996556 ns/op, 58.53 tok/s, 802486024 B/op, 17323617 allocs/op
2048 tokens: 34952296158 ns/op, 58.59 tok/s, 822547632 B/op, 17738577 allocs/op
```

At that checkpoint, if the path were actually running at Metal-class 100+ tok/s and only the
short benchmark were mis-accounting startup time, the 2048-token run would have
completed in roughly 20 seconds plus fixed overhead. Instead, the completed
2k-token run still takes about `35s`, so the gap to `100+ tok/s` is real. The
benchmark loads the model before `b.ResetTimer()` and computes `tok/s` from
elapsed time inside the timed generation loop.

Fusing Q/K/V projection, mapped launch packets, the descriptor append fast path,
and the long-context attention fixes lifted the 512-token check to
`7948177819 ns/op` at `64.42 tok/s` and the 2k endpoint to
`34231495412 ns/op` at `58.43 tok/s`. The later borrowed-alias/page-slice-pool
cleanup cut 2048-token allocation to `822547632 B/op` but only moved throughput
to `58.59 tok/s`; a fresh `GO_ROCM_BENCH_TOKENS=2000` endpoint check measured
`34170996556 ns/op` at `58.53 tok/s`. This older path was still below the
endpoint requirement once context grew.

After rebuilding the normal `gfx1100 -O2` HSACO from source following those
rejected experiments, the artifact still landed in the same band:

```text
512 tokens:  8069296382 ns/op, 63.45 tok/s, 214434344 B/op, 4462654 allocs/op
2000 tokens: 34126497340 ns/op, 58.61 tok/s, 803767264 B/op, 17323657 allocs/op
```

After unrolling the aligned q8 key-dot loop to consume four packed words per
iteration, q4 smoke still generated `[236764 3307]` and the current artifact
measured:

```text
512 tokens:  7828455100 ns/op, 65.40 tok/s, 214654840 B/op, 4462648 allocs/op
2000 tokens: 33268537694 ns/op, 60.12 tok/s, 803144400 B/op, 17323640 allocs/op
2048 tokens: 34097066745 ns/op, 60.06 tok/s, 822774688 B/op, 17738584 allocs/op
```

After adding the direct q4 value-accumulation branch for the default direct
`k-q8-v-q4` descriptor layout, the current artifact measured:

```text
512 tokens:  7822076555 ns/op, 65.46 tok/s, 214674576 B/op, 4462657 allocs/op
2000 tokens: 32902163632 ns/op, 60.79 tok/s, 803769656 B/op, 17323601 allocs/op
2048 tokens: 33781915323 ns/op, 60.62 tok/s, 823835248 B/op, 17738597 allocs/op
```

A fresh verification pass after the kept attention patches measured
`7834757484 ns/op` at `65.35 tok/s` for 512 tokens and `33742782570 ns/op` at
`60.69 tok/s` for 2048 tokens. A direct q4 value-loop unroll-by-two experiment
was rejected after q4 smoke because it was neutral at 512 tokens (`65.40 tok/s`)
and effectively neutral/slightly below the kept path at 2048 tokens
(`60.67 tok/s`).

Pre-scaling each normalized attention weight by the cached q4 value scale once
before paired q4 value accumulation was rejected. It preserved q4 smoke, but the
guarded exact path measured `60.68 tok/s` at 2000 tokens and `60.54 tok/s` at
2048 tokens.

An opt-in split-score attention experiment that computed q8 score dots in a
separate multi-block kernel was also rejected. It preserved q4 smoke, but 512
tokens measured `65.26 tok/s` and 2048 tokens regressed to `53.76 tok/s`.

The later driver launch-path cleanup removed per-launch HSACO `stat`/key
rebuilding, redundant launch-argument slice copies, repeated HIP availability
queries, and finish-closure allocation around launch-argument packets. Q4 smoke
was unchanged. The final checks measured:

```text
512 tokens:  7799788531 ns/op, 65.64 tok/s,  74929800 B/op,  787760 allocs/op
2000 tokens: 32768500713 ns/op, 61.03 tok/s, 258035584 B/op, 2988572 allocs/op
2048 tokens: 33646087144 ns/op, 60.87 tok/s, 264818936 B/op, 3059627 allocs/op
```

This is a kept driver cleanup, but still below the `100+ tok/s` endpoint.

A follow-up current-source check found and fixed one driver correctness issue
outside the default path: `GO_ROCM_DISABLE_ASYNC_LAUNCH_ARGS=1` reused a single
launch packet before the GPU consumed it and could reproduce as HIP error `700`.
That debug path now synchronizes before packet reuse; live q4 smoke under the
env passed. The default async launch-argument ring is unchanged. The same pass
also checked RX 7800 XT clocking. Forcing high/manual masks did not materially
move throughput, and the default path measured `7814954298 ns/op` at
`65.52 tok/s` for 512 tokens and `32762598372 ns/op` at `61.05 tok/s` for
2000 tokens.

A query-head RMSNorm+RoPE fusion was tested and rejected. It preserved live q4
smoke (`[236764 3307]`) and measured `65.53 tok/s` at 512 tokens, but repeat
2000-token endpoint checks regressed to `58.78 tok/s` and `58.73 tok/s`.
Reverting the fusion returned the same 2000-token check to `32737071937 ns/op`,
`61.09 tok/s`, `258238264 B/op`, `2988563 allocs/op`.

The kept q4 projection-family tweak now loads MLX affine scale/bias once per
q4 group in `rocm_mlx_q4_projection`, triple projection, final
projection+greedy, and q4 GELU projection. Live q4 smoke stayed stable
(`[236764 3307]`). The final kept checks measured:

```text
512 tokens:  7785336740 ns/op, 65.76 tok/s,  74929832 B/op,  787765 allocs/op
2000 tokens: 32642251550 ns/op, 61.27 tok/s, 258875656 B/op, 2988551 allocs/op
2048 tokens: 33542086457 ns/op, 61.06 tok/s, 264819224 B/op, 3059631 allocs/op
```

A fresh 512-token `rocprof --stats` run on the kept source measured
`60.38 tok/s` under profiler overhead. The current split is:

```text
rocm_attention_heads.kd                42.93%
rocm_mlx_q4_projection.kd              17.85%
rocm_mlx_q4_gelu_tanh_multiply.kd      10.90%
rocm_rms_norm_residual_add.kd           6.89%
rocm_mlx_q4_projection_greedy.kd        5.26%
rocm_rms_norm.kd                        5.21%
rocm_mlx_q4_triple_projection.kd        2.24%
rocm_mlx_q4_gelu_tanh_projection.kd     2.03%
__amd_rocclr_copyBuffer.kd              0.20%
```

Applying the same group-tiled loop to the paired q4 gate/up GELU multiply
kernel was rejected: smoke passed, but 512 tokens regressed to
`8157714213 ns/op`, `62.76 tok/s`.

Retuning the grouped q4 projection-family launch from 8 rows/block to
16 rows/block was also rejected. Smoke passed, but 512 tokens regressed to
`7960843350 ns/op`, `64.31 tok/s`, so the kept shape remains 8 rows per
256-thread block.

A dedicated small-column q4 projection kernel for the 256/512-column attention
output projections was rejected as well. Live q4 smoke stayed stable
(`[236764 3307]`), but 512 tokens measured `7815026708 ns/op`, `65.51 tok/s`,
and 2000 tokens measured `32820868597 ns/op`, `60.94 tok/s`, below the kept
`65.76`/`61.27 tok/s` endpoints.

Fusing the post-attention residual-add with pre-FFN RMSNorm is a small kept
layer-graph cleanup. Live q4 smoke stayed stable (`[236764 3307]`), and the
endpoint checks for that pass measured:

```text
512 tokens:  7684228195 ns/op, 66.63 tok/s,  74649224 B/op,  769797 allocs/op
2000 tokens: 32170028940 ns/op, 62.17 tok/s, 257342976 B/op, 2918526 allocs/op
2048 tokens: 32973719595 ns/op, 62.11 tok/s, 263898288 B/op, 2987923 allocs/op
```

The fresh 512-token `rocprof --stats` split after that fused residual/norm pass
is still kernel-bound:

```text
rocm_attention_heads.kd                43.14%
rocm_mlx_q4_projection.kd              17.93%
rocm_mlx_q4_gelu_tanh_multiply.kd      10.93%
rocm_mlx_q4_projection_greedy.kd        5.26%
rocm_rms_norm_residual_add.kd           4.71%
rocm_rms_norm_residual_add_norm.kd      3.78%
rocm_rms_norm.kd                        3.34%
rocm_kv_descriptor_append.kd            1.07%
__amd_rocclr_copyBuffer.kd              0.19%
```

Adding cross-layer input-norm precompute and wave-shuffle max/sum reductions in
token-parallel attention is also kept. The q4 hot path now writes the next
layer's input RMSNorm output while writing the current layer's final hidden
state, then lets the next layer consume that precomputed buffer. Q4 smoke stayed
stable (`[236764 3307]`), and endpoint checks for that pass measured:

```text
512 tokens:  7505556636 ns/op, 68.22 tok/s,  75411088 B/op,  769926 allocs/op
2000 tokens: 31411112083 ns/op, 63.67 tok/s, 259325968 B/op, 2920163 allocs/op
2048 tokens: 32356592135 ns/op, 63.29 tok/s, 265920536 B/op, 2989574 allocs/op
```

The fresh 512-token `rocprof --stats` split after cross-layer norm precompute
is still led by attention and q4 GEMV:

```text
rocm_attention_heads.kd                43.38%
rocm_mlx_q4_projection.kd              18.05%
rocm_mlx_q4_gelu_tanh_multiply.kd      11.02%
rocm_rms_norm_residual_add_norm.kd      7.42%
rocm_mlx_q4_projection_greedy.kd        5.31%
rocm_rms_norm_residual_add.kd           2.49%
rocm_rms_norm.kd                        1.27%
rocm_kv_descriptor_append.kd            1.07%
__amd_rocclr_copyBuffer.kd              0.20%
```

Final-token final RMSNorm precompute plus the short-context attention geometry
fix is now kept. The last decoder layer writes the final model RMSNorm output
for the LM-head path, and `rocm_attention_heads` now enters the 512-thread
paired q4 value path with q4 value metadata cached in dynamic shared memory from
16 tokens onward instead of 512. Q4 smoke stayed stable (`[236764 3307]`), and
that pass's endpoint checks measured:

```text
512 tokens:  5659309906 ns/op, 90.47 tok/s,  75225224 B/op,  769938 allocs/op
2000 tokens: 29611416789 ns/op, 67.54 tok/s, 259603056 B/op, 2920165 allocs/op
2048 tokens: 30357995463 ns/op, 67.46 tok/s, 266414472 B/op, 2989597 allocs/op
```

The fresh 512-token `rocprof --stats` split after that attention threshold fix
shows attention is no longer the sole dominant kernel:

```text
rocm_mlx_q4_projection.kd              26.54%
rocm_attention_heads.kd                17.08%
rocm_mlx_q4_gelu_tanh_multiply.kd      16.17%
rocm_rms_norm_residual_add_norm.kd     10.98%
rocm_mlx_q4_projection_greedy.kd        7.71%
rocm_rms_norm_residual_add.kd           3.56%
rocm_rms_norm.kd                        1.76%
rocm_kv_descriptor_append.kd            1.56%
__amd_rocclr_copyBuffer.kd              0.28%
```

The next kept pass replaced the remaining shared-memory tree reductions in
RMSNorm/residual kernels with wave-shuffle block reductions and fused
RMSNorm+RoPE heads for the q4 query/key vectors. The public q4 smoke stayed
stable (`[236764 3307]`), and that pass's endpoint checks measured:

```text
512 tokens:  5429310468 ns/op, 94.30 tok/s,  73470456 B/op,  726184 allocs/op
2000 tokens: 28966745166 ns/op, 69.04 tok/s, 251720040 B/op, 2749949 allocs/op
2048 tokens: 29755291471 ns/op, 68.83 tok/s, 258122472 B/op, 2815303 allocs/op
```

The fresh 512-token `rocprof --stats` split after that fusion was:

```text
rocm_mlx_q4_projection.kd              27.22%
rocm_attention_heads.kd                17.48%
rocm_mlx_q4_gelu_tanh_multiply.kd      16.63%
rocm_rms_norm_residual_add_norm.kd     10.79%
rocm_mlx_q4_projection_greedy.kd        7.86%
rocm_rms_norm_rope_heads.kd             3.63%
rocm_rms_norm_residual_add.kd           3.41%
rocm_mlx_q4_triple_projection.kd        3.39%
__amd_rocclr_copyBuffer.kd              0.29%
```

The descriptor-trim kept pass fixed the long-context sliding-window descriptor trim
path in `rocm_kv_descriptor_append`. The previous 2k profile showed descriptor
append at `18.48%` of GPU time because the trim fallback copied retained
one-token page descriptors serially. The new branch parallel-copies the common
direct one-token-page trim layout and appends the new descriptor on thread 0.
Q4 smoke stayed stable (`[236764 3307]`), and the descriptor-trim endpoint checks
measured:

```text
512 tokens:  5405693975 ns/op, 94.71 tok/s,  73465472 B/op,  726181 allocs/op
2000 tokens: 25194731733 ns/op, 79.38 tok/s, 252131160 B/op, 2749942 allocs/op
2048 tokens: 25883692325 ns/op, 79.12 tok/s, 259628912 B/op, 2815319 allocs/op
```

The fresh 2000-token `rocprof --stats` split after the descriptor-trim fix was:

```text
rocm_attention_heads.kd                30.86%
rocm_mlx_q4_projection.kd              22.57%
rocm_mlx_q4_gelu_tanh_multiply.kd      13.84%
rocm_rms_norm_residual_add_norm.kd      9.09%
rocm_mlx_q4_projection_greedy.kd        6.64%
rocm_rms_norm_rope_heads.kd             3.03%
rocm_rms_norm_residual_add.kd           2.82%
rocm_mlx_q4_triple_projection.kd        2.79%
rocm_mlx_q4_gelu_tanh_projection.kd     2.56%
rocm_kv_descriptor_append.kd            1.79%
__amd_rocclr_copyBuffer.kd              0.20%
```

The earlier 512-token `rocprof --stats` split after the launch cleanup was
similar:

```text
rocm_attention_heads.kd                42.59%
rocm_mlx_q4_projection.kd              18.47%
rocm_mlx_q4_gelu_tanh_multiply.kd      10.89%
rocm_rms_norm_residual_add.kd           6.85%
rocm_rms_norm.kd                        5.19%
rocm_mlx_q4_projection_greedy.kd        4.98%
rocm_kv_descriptor_append.kd            1.06%
__amd_rocclr_copyBuffer.kd              0.19%
```

`rocminfo` with `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85` resolves the pinned
GPU to `gfx1100`, UUID `GPU-880ed6479d653a85`, marketing name
`AMD Radeon RX 7800 XT`. `rocm-smi` samples during endpoint runs in this pass
showed the RX 7800 XT active (`84-99%`) and the onboard GPU at `0%`, so these
numbers are from the discrete GPU path rather than an iGPU fallback.

At the descriptor-trim checkpoint, the driver/HIP problems had narrowed to:

- Long-context attention still uses one device-KV descriptor/page per token.
  The linear descriptor scan is fixed, key-dot scale decode is hoisted, q4 value
  metadata is cached in dynamic shared memory, queries are cached in shared
  memory during the score phase, q8 key dots consume four values at a time when
  aligned, paired q4 value accumulation is used from the 16-token 512-thread
  block threshold onward, repeated full descriptor validation was removed from
  each head block, and direct one-token page addressing is hoisted. No-trim and
  sliding-window descriptor appends now have parallel fast paths. Device-query
  attention also avoids allocating dummy host query vectors now, but the launch
  still exposes only one block per query head. On Gemma4-E2B that is eight
  blocks per layer-token, so the RX 7800 XT cannot be saturated by attention.
  The reference `go-mlx` Gemma4 path uses a paged cache with 256-token pages and
  fused MLX SDPA for decode, while the ROCm path is still optimized around
  `page_count == token_count`. The next attention/KV task is a real layout and
  kernel change: block/page descriptors need direct addressing while preserving
  per-token q8/q4 scales, and attention needs more blocks per layer-token.
- `rocm_attention_heads` is again the largest single kernel after the descriptor
  trim fix. A fresh 2000-token `rocprof --stats` run reported attention at
  `30.86%`, q4 projection at `22.57%`, q4 GELU/multiply at `13.84%`,
  RMSNorm/residual-add-norm at `9.09%`, final q4 projection+greedy at `6.64%`,
  descriptor append at `1.79%`, and `__amd_rocclr_copyBuffer` at only `0.20%`.
- The q4 layer remains a packet-driven graph of primitive launches. The local
  q4 MLP fusions, RMSNorm+residual-add/layer-scalar fusion, and non-shared Q/K/V
  projection fusion cut launch count. More production margin and broader-context
  performance likely still need fused resident layer kernels and a reusable
  decode workspace.
- The HIP launch wrapper no longer dominates allocation churn: the 2048-token
  endpoint dropped from about `822.8MB/op` and `17.7M allocs/op` to
  `259.6MB/op` and `2.82M allocs/op`. The later attention threshold,
  RMSNorm-reduction, RMSNorm+RoPE fusion, and descriptor-trim work moved 2000
  tokens to `79.38 tok/s`, but the remaining production work was still not
  another launch-wrapper allocation pass.
- `rocm_mlx_q4_projection` is much faster than the original serial row GEMV,
  but it is still not a tiled production q4 projection kernel.

Codegen and q4 projection variants were checked after the fresh 2k run:

- `gfx1101` HSACO builds but fails to load through `hipModuleLoadData` with HIP
  error `200`; keep using the `gfx1100` agent reported by `rocminfo`.
- `gfx1100 -O3` is effectively identical to the `-O2` baseline at 64 tokens
  (`34.12 tok/s`) and at the latest 512-token post-fusion check
  (`94.30 tok/s`).
- `gfx11-generic -O2` loads and passes q4 smoke. On the latest post-fusion
  source it measured `94.54 tok/s` at 512 tokens and `69.24 tok/s` at
  2000 tokens, essentially tied with the kept `gfx1100 -O2` endpoint
  (`69.04 tok/s`) and not a path to `100+ tok/s`.
- A shared-input q4 projection variant passed smoke but regressed 64-token
  throughput to `28.60 tok/s`, so it was reverted.
- Final q4 projection+greedy row-block variants were also rejected: 16 rows per
  block measured `56.08 tok/s` at 64 tokens, and 4 rows per block measured
  `55.85 tok/s`, both below the then-current baseline before attention scale
  hoisting.
- A barrier-heavy shared value page/scale cache for token-parallel attention was
  rejected. It regressed 64 tokens to `60.40 tok/s` and 512 tokens to
  `31.24 tok/s`; the kept variant only reuses page/scale state locally within
  each thread.
- A two-stage final q4 projection+greedy reduction for large vocab projections
  was rejected. It reduced global best-value atomic contention but added an
  intermediate block-best buffer and reduction launch, regressing 64 tokens to
  `62.24 tok/s` and 512 tokens to `34.96 tok/s`.
- An earlier broad descriptor header/base caching experiment inside
  token-parallel attention was rejected. It passed correctness checks, but
  regressed 64 tokens to `62.65 tok/s` and 512 tokens to `34.85 tok/s`. The later
  narrower direct one-token page-addressing path is kept in the current ladder.
- An earlier attempt to cache each head's query vector in token-parallel
  attention shared scratch was rejected before the newer 512-thread / paired
  value shape. The later scoped score-phase query cache is kept in the current
  path because it moved the 2k endpoint from `25.20 tok/s` to `25.30 tok/s` and
  is included in the current `58.43 tok/s` ladder.
- A later q8 key-dot specialization that bypassed the generic descriptor helper
  was rejected. It was neutral at 512 tokens (`64.31 tok/s`) but regressed the
  2k endpoint to `58.20 tok/s` versus the kept `58.43 tok/s` path. Retesting
  1024-thread long-context attention after the descriptor-validation fixes also
  remained worse at `57.67 tok/s`, so the selector stays at 512 threads for
  contexts of at least 512 tokens.
- A narrower aligned q8 key-dot unroll is kept. It consumes four packed q8 words
  per loop iteration and moved the 2000-token endpoint to `60.12 tok/s`; the
  2048-token check measured `60.06 tok/s`.
- A direct q4 value-accumulation branch is also kept for the default direct
  `k-q8-v-q4` descriptor layout. It moved the 2000-token endpoint to
  `60.79 tok/s` and the 2048-token check to `60.62 tok/s`.
- Lowering both the attention block-size selector and q4 value metadata-cache
  threshold to 16 tokens is kept. It moved 512 tokens to `90.47 tok/s`,
  2000 tokens to `67.54 tok/s`, and 2048 tokens to `67.46 tok/s`.
- RMSNorm shuffle reductions and fused RMSNorm+RoPE heads are kept. They moved
  512 tokens to `94.30 tok/s`, 2000 tokens to `69.04 tok/s`, and 2048 tokens
  to `68.83 tok/s`.
- The parallel sliding-window descriptor-trim fast path is kept. It moved 512
  tokens to `94.71 tok/s`, 2000 tokens to `79.38 tok/s`, and 2048 tokens to
  `79.12 tok/s`, and dropped `rocm_kv_descriptor_append` to `1.79%` in the
  fresh 2k profile.
- A fast softmax exp experiment in attention was rejected. It was neutral at
  512 tokens (`64.19 tok/s`) but regressed the 2k endpoint to `58.08 tok/s`.
- Deferring attention softmax normalization into value accumulation passed q4
  smoke and measured `64.21 tok/s` at 512 tokens, but regressed the 2000-token
  endpoint to `56.33 tok/s`, so it was reverted.
- A fast q4 GELU tanh approximation was rejected. It passed q4 smoke, but
  changed generated tokens and regressed the 2048-token endpoint to
  `58.38 tok/s`.
- An attention q4 value-metadata threshold change from `>=512` to `>512` was
  rejected before the later 16-token threshold fix. It passed q4 smoke, but
  regressed 512 tokens to `64.17 tok/s` and 2048 tokens to `55.94 tok/s`.
- Switching the experimental device-KV mode to `fp16` or `q8` failed the public
  q4 smoke on the second generated token, so the default remains `k-q8-v-q4`.
- An experimental `k-q4-v-q4` device-KV mode also failed the public q4 smoke by
  changing the generated token, so q4 keys are not kept as a speed shortcut.
- A group-64 q4 scale/bias layout experiment passed q4 smoke but regressed the
  512-token benchmark to `59.77 tok/s`, so it was reverted.
- `GO_ROCM_ENABLE_ASYNC_H2D=1` was neutral at 512 tokens (`64.30 tok/s`) and is
  not a default endpoint fix.
- `gfx1100 -O2 -ffast-math` passed q4 smoke but regressed 512 tokens to
  `63.05 tok/s`.
- q4 projection/GELU `__restrict__` pointer hints passed q4 smoke but measured
  `64.31 tok/s` at 512 tokens and `58.48 tok/s` at 2000 tokens, so they were
  reverted.
- A guarded group-64 q4 projection/GELU group-index shift branch passed q4
  smoke but regressed two 512-token checks to `62.98 tok/s` and `62.93 tok/s`,
  so it was reverted.
- A dedicated 256/512-column q4 projection kernel passed q4 smoke but regressed
  the kept 512/2000-token endpoints to `65.51`/`60.94 tok/s`.
- Retuning the general q4 projection launch to 4 rows per 256-thread block
  failed q4 projection smoke. Retuning to 16 rows per block preserved q4 smoke
  but regressed 512 tokens to `87.58 tok/s`, below the kept `90.47 tok/s`.
- A q4 shared-input GEMV experiment preserved q4 smoke but regressed 512 tokens
  to `74.40 tok/s`, so it was reverted. A block-level q4 greedy atomic
  reduction also preserved q4 smoke but regressed 512 tokens to `93.79 tok/s`
  versus the kept `94.30 tok/s`, so it was reverted.
- A q4-generation temporary device-buffer arena wrapper passed q4 smoke but did
  not help: 512 tokens measured `64.20 tok/s` enabled versus `64.14 tok/s`
  disabled, while Go allocations increased from `4462661` to `5175655`; the
  2000-token check regressed to `58.33 tok/s`. It was removed.
- Shared-device-KV borrowed aliases now share source page metadata without
  ownership, and hot-path copied KV page slices are pooled. This is kept because
  it substantially reduces B/op, but it does not materially change tok/s.

## Profile Findings

Initial memory profile, `alloc_space`:

```text
copyTensorToDevice                         1105 MB
os.readFileContents                         303 MB
hipTokenTextMergeRanks                      242 MB cumulative
loadHIPTokenTextDecoder                     358 MB cumulative
hipRunGemma4Q4DecoderLayer                  284 MB cumulative
```

CPU profile:

```text
runtime.cgocall                             63.6%
cgoHIPDriver.LaunchKernel                   53.9% cumulative
core_rocm_hip_module_load_data              25.2% cumulative
core_rocm_hip_free                          19.2% cumulative
rocmModel.Generate / wrapTokenStream        59.3% cumulative
```

After module/function and launch-argument caching, `hipModuleLoadData`,
`hipModuleUnload`, and per-launch argument `hipFree` are no longer dominant.
The 8-token CPU profile is now dominated by synchronous device-to-host
readbacks:

```text
runtime.cgocall                             83.0%
hipMemcpyDeviceToHost                       74.7%
hipAttentionDeviceBuffers.ReadOutputOnly    55.1%
hipRunMLXQ4ProjectionKernelWithDeviceWeight 13.3%
hipRunRMSNormKernelWithDeviceWeight          6.4%
```

After the device-buffer MLP, attention concat, cgo small-buffer pool, batched
query upload, and `rocm_attention_heads`, the short 8-token profile shifted
again:

```text
runtime.cgocall                             82.8%
hipMemcpyHostToDevice                       67.5%
cgoHIPDriver.LaunchKernel                   62.1%
cgoHIPDriver.launchArgPointer               61.9%
hipRunAttentionOutputFromDeviceQuery...     49.3%
hipMemcpyDeviceToHost                       14.8%
hipReadFloat32DeviceOutput                  13.8%
```

That means the next large step cannot be another host-side bookkeeping tweak.
The driver still launches too many tiny packet-driven primitives, and the
attention work itself is still too serial for long KV histories.

After device KV token encoding, GPU descriptor append, partial/global device
chaining, and shared descriptor borrowing, the 64-token profile changed again:

```text
BenchmarkInferenceGemma4Q4Generate-32  1  7540477396 ns/op  8.488 tok/s  73127728 B/op  1321564 allocs/op
hipReadGreedyResult / final D2H sync        6.74s cumulative
hipMemcpyHostToDevice                       0.37s cumulative
hipRunGemma4Q4DecoderLayerInternal...       0.53s visible CPU cumulative
```

The final greedy-result read is only 8 bytes, so the `6.74s` cost is the sync
point for queued GPU work. The largest host-copy problems were no longer the
dominant limiter; the pre-endpoint gap was the unfused kernel graph, naive q4
projection, and the decode attention implementation. RMSNorm/RoPE were the next
confirmed primitive bottleneck and are now parallelized, but still separate
launches.

After parallel RMSNorm/RoPE, token-parallel attention heads, and q4 projection
row-block tuning, the 64-token profile/benchmark changed to:

```text
BenchmarkInferenceGemma4Q4Generate-32  1  2892521198 ns/op  22.13 tok/s  73135920 B/op  1321564 allocs/op
hipReadGreedyResult / final D2H sync        2.21s cumulative
hipMemcpyHostToDevice                       0.47s cumulative
cgoHIPDriver.LaunchKernel                   0.51s cumulative
```

The 64-token best measured run after setting q4 projection row blocks to 64
threads was `2660767496 ns/op`, `24.05 tok/s`. Testing 32-thread and 128-thread
row blocks kept correctness but regressed the 64-token benchmark to about
`23.66-23.68 tok/s`, so 64 is the current best measured setting.

Before the latest descriptor-validation and q4 value metadata cache pass, mapped
launch-argument packets became the default, descriptor append copying was
parallelized for the common no-trim path, long-context attention switched to
512-thread blocks, query vectors were cached in shared memory for the score
phase, q8 key dots started consuming four values per packed word, and extra q4
value workers were paired per output dimension. A 512-token `rocprof --stats`
run at that point reported:

```text
BenchmarkInferenceGemma4Q4Generate-32  1  12468225072 ns/op  512.0 max_tokens/op  41.06 tok/s  512.0 tokens  741985256 B/op  4473637 allocs/op
rocm_attention_heads.kd                17955 calls  6.374929478s  66.08%
rocm_mlx_q4_projection.kd              64125 calls  1.050792283s  10.89%
rocm_mlx_q4_gelu_tanh_multiply.kd      17955 calls  0.618871851s   6.42%
rocm_rms_norm_residual_add.kd          53865 calls  0.397453391s   4.12%
rocm_rms_norm.kd                       51813 calls  0.295016397s   3.06%
rocm_mlx_q4_projection_greedy.kd         513 calls  0.283566582s   2.94%
rocm_kv_descriptor_append.kd            7680 calls  0.060575175s   0.63%
__amd_rocclr_copyBuffer.kd              2183 calls  0.011185133s   0.12%
```

That profile is stale after the newer direct benchmarks moved 512 tokens from
`43.27 tok/s` to `64.42 tok/s` and 2k tokens from `28.82 tok/s` to
`58.43 tok/s`, but it still narrows the next real target to attention occupancy
and layer fusion. Driver copy tuning is no longer large enough to bridge the
endpoint gap.

The long-prompt prefill scaffold now has batched Gemma4 q4 embedding,
input-norm, Q/K/V q4 projection, value no-scale RMSNorm, MLP gate/up/down
projection, Q/K RMSNorm+RoPE, batched device-KV row page primitives, and a
first batched causal attention primitive. The Q/K primitive is
`rocm_rms_norm_rope_heads_batch`, which maps prompt rows to `GridY` and applies
`start_position+row` RoPE for query/key buffers. The KV primitive reuses
`rocm_kv_encode_token` over row chunks and creates descriptor pages with
`token_count > 1` instead of appending one page per prompt token. The attention
primitive is `rocm_attention_heads_batch_causal`, which maps query heads to
`GridX`, prompt rows to `GridY`, and applies the per-row causal visible-token
limit over contiguous or descriptor-backed device KV. The scaffold now composes
those pieces for one layer through output projection, residual/norm, MLP, and
the Gemma4 per-layer input branch to final hidden output, then borrows only the
requested output row for final RMSNorm and fused q4 LM-head softcap+greedy. This
removes another single-token-shaped piece, and the first all-layer scaffold can
run that path over a position-0 token span. It is still not the completed
48k/64k endpoint: the prompt path still needs prior-ubatch KV merging plus the
hybrid full/SWA attention policy before it can replace the single-token prompt
loop.

## Optimization Candidates

1. Replace `rocm_attention_heads` with a real long-context decode attention
   kernel. The 512-thread block shape, score-phase query cache, packed q8 key
   dots, paired q4 value accumulation, shared q4 metadata cache, header-only
   validation, normalized weights, and direct page addressing are endpoint wins,
   but the current launch still exposes too little parallel work per layer.

2. Replace the current primitive q4 decode sequence with fused resident kernels.
   The q4 path no longer readbacks every MLP and attention-head intermediate,
   but it still launches a packet-driven primitive graph for norm/projection/
   RoPE/attention/residual/logits. A production path needs at least fused
   per-layer decode, device-resident residual/MLP intermediates, and final
   logits/greedy sampling without host readback between primitives.

3. Reuse q4 intermediate buffers and KV state across Generate calls.
   The benchmark still allocates and copies many per-primitive input/output
   buffers. Buffer arenas keyed by hidden/intermediate/vocab dimensions should
   cut allocation count, `hipMalloc`/`hipFree`, and copy synchronization.

4. Reduce model-load allocation in `copyTensorToDevice`.
   The load path accounts for roughly 1.1 GB of allocation space in this run.
   Use a reusable chunk buffer or mmap-style reader path for tensor copies so
   load-time allocations do not dominate single-shot inference benchmarks.

5. Cache tokenizer decoding artifacts per model path.
   `tokenizer.json` loading and merge-rank parsing allocate hundreds of MB.
   A compact parser for only the fields needed by the q4 path, plus process
   cache keyed by tokenizer path and mtime, would reduce startup overhead.

6. Keep the HSACO module/function and launch-argument caches.
   These were the first confirmed optimization: 1-token throughput improved from
   about 0.235 tok/s to about 0.868 tok/s, and 8-token throughput improved from
   about 0.303 tok/s to about 0.691 tok/s.

## 2026-05-17 Gemma4 q4 2k/4k Current Slice

The long-context ladder is paused until 2k/4k prompt processing is closer to
other runners. Latest RX 7800 XT actual-load checks, using the Gemma4 q4 model
pack at `/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit`, the rebuilt gfx1100
HSACO, 48k context, one generated token, and 512-token ubatches:

```text
2k prompt:  5963139198 ns/op,  343.4 prompt_tok/s, 12329936 B/op, 29787 allocs/op
4k prompt: 12840551074 ns/op,  319.0 prompt_tok/s, 14511024 B/op, 52314 allocs/op
```

The main fix was descriptor-backed attention page lookup. Prompt KV uses
16-token pages, and the old device lookup linearly scanned descriptor pages
unless every page was a single token. The kernel now uses descriptor
`block_size` for direct page indexing and scans only as fallback.

The q4 batch projection path also now uses an 8-token tile for the MLX q4
projection batch, fused GELU-tanh gate/up batch, and per-layer-input GELU
projection batch kernels, reusing q4 weight loads across prompt tokens. A
16-token tile did not improve the 2k run; the per-layer-input tile was flat to
slightly down at 2k but moved the 4k check from `315.8` to `319.0 prompt_tok/s`.

This is above the `100+ tok/s` driver floor, but it is still well behind the
local llama.cpp reference (`~4788 tok/s` at 2k and `~4520 tok/s` at 4k prompt
processing). The remaining gap is kernel design: the q4 prompt path still needs
a production tiled GEMM/dequant implementation rather than row-dot packet
primitives with limited token reuse.
