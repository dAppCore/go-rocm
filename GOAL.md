# Goal: 100+ tok/s ROCm Gemma4 q4 Driver

> For `/goal` agents: execute this file as the mission contract. Read `RFC.md`
> first, then work down this checklist with TDD. Preserve user changes.
> Immediate endpoint: make the native ROCm q4 driver sustain `100+ tok/s` on the
> real RX 7800 XT. Do not import `go-mlx`.

## Goal

Make `go-rocm` the ROCm sibling of `go-mlx`: a native package-first inference
backend that implements the shared `go-inference` contracts for model loading,
metadata inspection, scheduling, cancellation, cache services, parser services,
state wake/sleep/fork groundwork, probes, evaluation, and benchmarks before
expanding model families and training kernels.

The near-term endpoint is stricter than "kernel linked". Gemma4-E2B q4 decode
must become a real ROCm driver path that keeps the token step device-resident
and reaches `100+ tok/s` on a long generation benchmark. Sub-100 tok/s numbers
are not acceptable for handoff.

## Performance Endpoint

The 100+ tok/s goal is complete only when all of these are true:

- `BenchmarkInferenceGemma4Q4Generate` on `/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit`
  with `GO_ROCM_BENCH_TOKENS=2000` or the stricter `2048` check reports at
  least `100 tok/s`.
- The benchmark measures generation time after model load. A 2k-token run must
  finish without manual interruption.
- `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85` is used, `rocminfo` resolves that
  UUID to `AMD Radeon RX 7800 XT` / `gfx1100`, and the onboard GPU remains idle.
- The q4 text smoke still passes for `GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi'`
  with prompt tokens `[2 10979]` and valid decoded generated tokens.
- The BF16 Gemma4-E2B smoke remains a correctness anchor for layer-0 tensor
  math and tied LM-head greedy output.
- Codecov-filtered package coverage stays at or above `90%`.
- Normal, Linux no-cgo, legacy-server, HIP, model, and cache smoke gates keep
  passing or skip cleanly when hardware is absent.

Current status as of 2026-05-17: the q4 performance endpoint is met on the
pinned RX 7800 XT with the live `gfx1100` HSACO. The latest corrected
`GO_ROCM_BENCH_TOKENS=2000` run reports `19629810772 ns/op` and `101.9 tok/s`;
the stricter `2048` run reports `20156670616 ns/op` and `101.6 tok/s`. These
numbers use the benchmark's `inference.WithContextLen(128)` load setting, now
correctly applied to Gemma4 q4 sliding-window layers, and keep full-attention
layers uncapped.

Current dependency/platform baseline as of 2026-05-26: local development uses
Go `1.26.x`, `dappco.re/go v0.10.3`, canonical `go-inference v0.10.0`, and
`dappco.re/go/cgo v0.11.1` from `external/go-cgo/go`. HIP kernels are now built
with `hipcc --std=c++23`; ROCm 7.2 accepts the language mode, but this host
toolchain does not provide `<mdspan>`, so the zero-copy path starts with
Go-side `core.PinnedView`/`go-cgo` scoped pins and should only include the
`go-cgo` mdspan companion header once the toolchain can compile it.
The first `go-mlx` state-pattern parity slice is also present behind
`SleepRequest.Encoding == "rocm/kv-cache-block-bundle+json"`: ROCm can write KV
cache pages as raw per-block State refs plus a manifest and wake by walking
those refs, instead of requiring one monolithic snapshot blob. Loaded HIP models
can now restore those block refs directly into HIP device KV pages using
`BorrowRefBytes` plus `go-cgo`/`core.PinnedView` host handoff, then install the
device runtime without the older host-cache remirror step. Keep this layer
HIP-generic: the same code path should remain usable for a future NVIDIA HIP
backend profile if the local toolchain targets CUDA through HIP.

NVIDIA portability proof is now part of the acceptance surface. The ROCm 7.2
HIP source compiles through CUDA/NVCC with
`GO_ROCM_RUN_NVIDIA_HIP_COMPILE_TESTS=1`; CUDA 12.8 accepts `--std=c++20` for
that backend while the AMD `gfx1100` build remains `--std=c++23`. A ZLUDA v5
CUDA runtime smoke is also available behind `GO_ROCM_RUN_ZLUDA_CUDA_TESTS=1`;
on this ROCm 7.2 host it uses side-by-side ROCm 6.4.4 rpath runtime libraries
from `/opt/rocm-6.4.4/lib` so ZLUDA's `libamdhip64.so.6` dependency is
satisfied without downgrading the real ROCm 7.2 development stack.

AX-11 benchmark rule applies here: any per-token, per-page, per-request, or
cross-product hot path touched for this driver needs a `Benchmark*` with
`b.ReportAllocs()`. The raw KV block restore path now has benchmarks for host
decode and direct pinned HIP page restore, making allocation and copy cost
visible before the 48k/64k context work resumes.

## Book 10-Turn Retained-State Acceptance

The short decode endpoint and synthetic prompt ladders are diagnostics, not the
destination. The real workload is the `book.md` profile: use the creative prompt
bank to ask Gemma4 q4 for chapter 1 of a book without declaring the final
chapter count up front, then run nine follow-up turns asking for more chapters
while injecting one fresh distractor prompt each turn. The chapter-10 output
must still preserve the first prompt's story arc, and the driver must get there
by retaining and extending KV state instead of replaying the whole book/session
prompt every turn.

The reference `go-mlx`/Metal run completes the 10-turn book generation in about
`82s` wall time. That is now the practical endpoint for this driver:

- Keep the existing `100+ tok/s` short-generation endpoint green.
- Run the 10-turn `book.md` workload on the real Gemma4-E2B q4 model.
- Let each chapter run to the model's natural stop condition, bounded only by a
  large context-derived safety cap. Omitted `GO_ROCM_BOOK_CHAPTER_TOKENS` or
  `GO_ROCM_BOOK_CHAPTER_TOKENS=0` is the acceptance mode; explicit small caps
  such as `512` are smoke/debug runs and do not prove the book endpoint.
- Use state wake/sleep/fork or equivalent device-resident retained KV so each
  chapter turn appends only the new chapter request and distractor.
- Track per-turn wall time, restore/wake time, append/prefill time, decode
  tok/s, generated tokens, total retained tokens, VRAM, `B/op`, and `allocs/op`.
- Treat chapter-10 story-arc retention as correctness: lighthouse keeper,
  signalling light, and deep-ocean entity from turn 1 must remain coherent even
  after nine distractors.
- Tune the compute graph until the ROCm path approaches the `82s` Metal
  reference. `<=90s` wall time for the 10-turn book is success. `<=110s` with
  coherent chapter-10 story retention is good enough to flip the route toward
  production defaults and start removing obsolete slow/fallback paths. Above
  `110s` remains experimental/tuning. Synthetic 2k/4k/29k prompt runs are only
  used to identify which graph phase is blocking that book workload.

Current status: the 10-turn full-chapter retained book wall-time endpoint is
met only after changing the Gemma4 device-KV page default to one token per page.
The RX 7800 XT is selected correctly and remains busy during the run, while the
onboard GPU remains idle. The retained-state book benchmark appends only the new
turn prompt plus Gemma4 chat-control tokens; it does not replay prior chapters.
The Gemma4 chat template is matched to the local HF tokenizer for
`<bos><|turn>user\n...<turn|>\n<|turn>model\n`, and the `text:` parser
preserves the final generation-prompt newline. A full-cap 10-turn retained run
with the old 16-token device-KV pages completed cleanly but took `438.9s` wall
and fell from `93.8 tok/s` on turn 1 to `10.6 tok/s` by turn 10. With
one-token device-KV pages and the distractor placed before the final continuation
instruction, the same acceptance shape completed in `74.2s` wall with empty
stderr, `4215` generated tokens, no chapter cap hits, and chapter-10 anchor hits
of `3`. A later direct one-token KQ8/VQ4 value fast path for retained pages
improved the same full-cap retained book route to `60.2s` wall, `55.45s`
decode, `3801` generated tokens, empty stderr, no chapter cap hits, and
chapter-10 anchor hits of `3`. After reducing chunked attention to 128-token
chunks, adding a direct chunked KQ8/VQ4 page path, and making chunked attention
the default route, the full-cap retained book route completed in `41.96s` wall
with empty stderr, `3021` generated tokens, no chapter cap hits, and
chapter-10 anchor hits of `3`. A separate dynamic shared-memory query cache for
chunked stage 1 then kept the same acceptance shape green at `41.31s` wall,
`37.08s` decode, `3021` generated tokens, `73.13 tok/s` average, `63.28 tok/s`
on turn 10, empty stderr, and chapter-10 anchor hits of `3`, while reducing the
book benchmark allocation volume from `887.5MB/op` to `762.4MB/op`. After
typed launch-packet pooling and device-buffer pool bucket retention, the same
route stayed green at `41.29s` wall, `37.04s` decode, `73.17 tok/s` average,
`63.85 tok/s` on turn 10, empty stderr, and chapter-10 anchor hits of `3`,
while reducing object churn from `5.18M` to `3.18M allocs/op`; the one-shot book
`B/op` rose to `1.09GB/op`, so the next allocation pass should target byte
volume and unique temporary buffer sizes rather than only object count. A typed
KV page-slice pool then improved the short 2048-token guard from `3.40M` to
`1.98M allocs/op` overall while keeping decode at `103.5 tok/s`. The next
allocation pass skipped unused production forward labels/host state, reused the
chunked-attention output buffer from the workspace, and pooled the chunked
stage-2 launch packet copy. The short 2048-token guard now reports
`19835935211 ns/op`, `103.2 tok/s`, `160949264 B/op`, and `1781980 allocs/op`.
The retained 10-turn full-cap book route stayed green at `41.21s` wall,
`36.98s` decode, `73.31 tok/s` average, `63.65 tok/s` on turn 10, empty
stderr, and chapter-10 anchor hits of `3`, while reducing the book benchmark to
`439319760 B/op` and `2797196 allocs/op`. Removing per-call q4 projection
launch-validator map literals then kept the short guard at `103.3 tok/s` while
dropping it to `146939376 B/op` and `1689861 allocs/op`; the retained book route
stayed green at `41.19s` wall, `36.96s` decode, `73.34 tok/s` average,
`63.67 tok/s` on turn 10, empty stderr, and chapter-10 anchor hits of `3`, with
`418587400 B/op` and `2660797 allocs/op`. A follow-up 2048-token fast-loop
allocation batch stopped copying borrowed RMSNorm-head launch packets, reused a
single per-layer input device view, and used stack-backed Q/K/V views for the
fused triple projection. The short guard now reports `19783450849 ns/op`,
`103.5 tok/s`, `117716864 B/op`, and `1314741 allocs/op`. Replacing shared-KV
alias objects with explicit borrowed ownership flags then moved the short guard
to `19806588088 ns/op`, `103.4 tok/s`, `111822000 B/op`, and
`1232865 allocs/op`. Reusing the hidden-size attention projection output from
the decode workspace then moved the short guard to `19777223686 ns/op`,
`103.6 tok/s`, `107241912 B/op`, and `1161227 allocs/op`. The retained book
route stayed green after these fast-loop batches at `40.68s` wall, `36.45s`
decode, `3021` generated tokens, `74.26 tok/s` average, `64.68 tok/s` on turn
10, empty stderr, no cap hits, and chapter-10 anchor hits of `3`, while
dropping to `356470968 B/op` and `1835290 allocs/op`. This is the current best
production-candidate route for the book endpoint. A later workspace batch reused
the MLP output, GELU-tanh activation, per-layer input projection, and first
residual-add/norm outputs. The short guard now reports `19849179335 ns/op`,
`103.2 tok/s`, `79731416 B/op`, and `731369 allocs/op`. The retained book route
stayed green at `40.69s` wall, `36.45s` decode, `3021` generated tokens,
`74.25 tok/s` average, `65.44 tok/s` on turn 10, empty stderr, no cap hits, and
chapter-10 anchor hits of `3`, while dropping to `315658912 B/op` and
`1198795 allocs/op`. Reusing query/key RoPE, value RMS no-scale, and
intermediate post-FFN buffers then moved the short guard to `19816510363 ns/op`,
`103.3 tok/s`, `66625400 B/op`, and `526667 allocs/op`. The retained book route
stayed green at `41.23s` wall, `36.99s` decode, `3021` generated tokens,
`73.27 tok/s` average, `64.44 tok/s` on turn 10, empty stderr, no cap hits, and
chapter-10 anchor hits of `3`, while dropping to `296256152 B/op` and
`895702 allocs/op`. Reusing QKV/triple-projection outputs then moved the short
guard to `19776340979 ns/op`, `103.6 tok/s`, `62041912 B/op`, and
`455039 allocs/op`. The retained book route stayed green at `40.74s` wall,
`36.49s` decode, `3021` generated tokens, `74.16 tok/s` average,
`65.31 tok/s` on turn 10, empty stderr, no cap hits, and chapter-10 anchor hits
of `3`, while dropping to `289548824 B/op` and `789629 allocs/op`. Reusing
slotted final-hidden and next-layer-input workspace buffers, then caching the
immutable loaded q4 forward config on the model, moved the short guard to
`19785444116 ns/op`, `103.5 tok/s`, `52871360 B/op`, and `311757 allocs/op`.
The retained book route stayed green at `40.63s` wall, `36.39s` decode,
`3021` generated tokens, `74.36 tok/s` average, `64.91 tok/s` on turn 10,
empty stderr, no cap hits, and chapter-10 anchor hits of `3`, while dropping to
`275889064 B/op` and `577472 allocs/op`. Pooling the internal appended-token
descriptor table wrappers then moved the short guard to `19808234851 ns/op`,
`103.4 tok/s`, `50908888 B/op`, and `281087 allocs/op`. The retained book
route stayed green at `41.23s` wall, `36.96s` decode, `3021` generated tokens,
`73.28 tok/s` average, `64.02 tok/s` on turn 10, empty stderr, no cap hits, and
chapter-10 anchor hits of `3`, while dropping to `273063984 B/op` and
`532044 allocs/op`. Caching decoded token text, reusing a workspace token-ID
buffer plus embedding/scaled-embedding outputs, and moving Gemma4 per-layer
embedding precompute temporaries into the workspace then moved the short guard
to `19785395783 ns/op`, `103.5 tok/s`, `49536176 B/op`, and `255531 allocs/op`.
The retained book route stayed green at `41.22s` wall, `36.98s` decode,
`3021` generated tokens, `73.28 tok/s` average, `64.26 tok/s` on turn 10,
empty stderr, no cap hits, and chapter-10 anchor hits of `3`, while dropping to
`271027784 B/op` and `492943 allocs/op`. Passing retained device KV layer state
by value, caching the shared-KV source table on the loaded forward config, using
a value-backed next-input norm request, and only keeping generated-token history
for host sampling then moved the short guard to `19775658852 ns/op`,
`103.6 tok/s`, `36315928 B/op`, and `110184 allocs/op`. The chapter-shaped
2048-token fast guard at `context_len=4096` stayed above the working threshold at
`22207829048 ns/op`, `92.22 tok/s`, `60786376 B/op`, and `125647 allocs/op`.
The retained book route stayed green at `40.60s` wall, `36.36s` decode, `3021`
generated tokens, `74.41 tok/s` average, `65.31 tok/s` on turn 10, empty stderr,
no cap hits, and chapter-10 anchor hits of `3`, while dropping to
`251419160 B/op` and `277596 allocs/op`. Reusing retained device-state layer
slices and device KV cache owner wrappers, warming the immutable Gemma4 q4
forward config during load, skipping token-probe construction when no probe sink
is installed, avoiding q4 config validation scratch slices, and stack-backing
the greedy readback payload then moved the short guard to `19811032696 ns/op`,
`103.4 tok/s`, `20404472 B/op`, and `72262 allocs/op`. The chapter-shaped
2048-token guard stayed above threshold at `22550870699 ns/op`, `90.82 tok/s`,
`44950928 B/op`, and `87745 allocs/op`. The retained book route stayed green at
`41.23s` wall, `36.97s` decode, `3021` generated tokens, `73.27 tok/s` average,
`64.47 tok/s` on turn 10, empty stderr, no cap hits, and chapter-10 anchor hits
of `3`, while dropping to `232519696 B/op` and `227862 allocs/op`. This is the
current best allocation route for the book endpoint. A follow-up 2048-token
fast-loop batch then cached hot HIP C bridge symbols, snapshotted the probe sink
once per public token stream, matched the KV page-slice pool floor to Gemma4's
512-token local window, and added source guards for q4 projection/GELU row
geometry. The short guard now reports `19763118653 ns/op`, `103.6 tok/s`,
`17650384 B/op`, and `72292 allocs/op`; the chapter-shaped 2048-token guard
reports `22546790165 ns/op`, `90.83 tok/s`, `44646464 B/op`, and
`87792 allocs/op`. The retained book route stayed green at `41.29s` wall,
`37.00s` decode, `3021` generated tokens, `73.17 tok/s` average,
`64.50 tok/s` on turn 10, empty stderr, no cap hits, and chapter-10 anchor hits
of `3`, with `232426568 B/op` and `227955 allocs/op`. Replacing the chunked
stage-1 shared-memory score-lane reduction with an order-preserving lane
shuffle then moved the short 2048-token guard to `19020344753 ns/op`,
`107.7 tok/s`, `17641136 B/op`, and `72291 allocs/op`, while the
chapter-shaped 2048-token guard moved to `20713968694 ns/op`, `98.87 tok/s`,
`44673712 B/op`, and `87783 allocs/op`. The retained book route stayed green at
`38.37s` wall, `34.12s` decode, `3021` generated tokens, `78.74 tok/s`
average, `67.74 tok/s` on turn 10, empty stderr, no cap hits, and chapter-10
anchor hits of `3`, with `232339832 B/op` and `227919 allocs/op`. Folding the
stage-1 dim0/dim1 value partials into one shared-memory pass then moved the
short guard to `18985260922 ns/op`, `107.9 tok/s`, `17640976 B/op`, and
`72290 allocs/op`; the chapter-shaped guard stayed at `20708006858 ns/op`,
`98.90 tok/s`, `44638912 B/op`, and `87796 allocs/op`. The retained book route
stayed green at `38.34s` wall, `34.09s` decode, `3021` generated tokens,
`78.79 tok/s` average, `68.29 tok/s` on turn 10, empty stderr, no cap hits, and
chapter-10 anchor hits of `3`, with `232419864 B/op` and `227947 allocs/op`.
Fresh `rocprof --stats` after those stage-1 reductions shows the hotspot moved:
`rocm_mlx_q4_projection` is now `27.82%`, q4 GELU multiply is `16.16%`, and
chunked stage 1 is down to `15.01%`. Replacing the final q4 projection+greedy
block reduction's repeated shared-memory barriers with one post-sync serial
32-row pass then kept the short guard green at `19034192118 ns/op`,
`107.6 tok/s`, `17649968 B/op`, and `72291 allocs/op`; moved the
chapter-shaped guard to `20630704555 ns/op`, `99.27 tok/s`, `44795392 B/op`,
and `87813 allocs/op`; and kept the retained book route green at `38.34s` wall,
`34.10s` decode, `3021` generated tokens, `78.80 tok/s` average,
`68.61 tok/s` on turn 10, empty stderr, no cap hits, and chapter-10 anchor hits
of `3`, with `232424184 B/op` and `227934 allocs/op`. Specializing the shared
q4 projection row-sum helper for Gemma's `group_size == 64` then moved the
short guard to `18851151583 ns/op`, `108.6 tok/s`, `17649904 B/op`, and
`72289 allocs/op`; kept the chapter-shaped guard green at `20684218707 ns/op`,
`99.01 tok/s`, `44639832 B/op`, and `87777 allocs/op`; and kept the retained
book route green at `38.24s` wall, `33.98s` decode, `3021` generated tokens,
`79.00 tok/s` average, `68.71 tok/s` on turn 10, empty stderr, no cap hits, and
chapter-10 anchor hits of `3`, with `232424368 B/op` and `227952 allocs/op`.
Specializing q4 GELU multiply's existing per-packed path for `group_size == 64`
then moved the short guard to `18811240904 ns/op`, `108.9 tok/s`,
`17649344 B/op`, and `72291 allocs/op`; moved the chapter-shaped guard to
`20612256528 ns/op`, `99.36 tok/s`, `44639752 B/op`, and `87777 allocs/op`;
and kept the retained book route green at `38.18s` wall, `33.96s` decode,
`3021` generated tokens, `79.12 tok/s` average, `68.81 tok/s` on turn 10,
empty stderr, no cap hits, and chapter-10 anchor hits of `3`, with
`232416808 B/op` and `227925 allocs/op`. Reusing the per-layer input device-set
wrapper from the decode workspace then moved the short guard to
`18785434694 ns/op`, `109.0 tok/s`, `17305768 B/op`, and `68198 allocs/op`;
moved the chapter-shaped guard to `20246696622 ns/op`, `101.2 tok/s`,
`44339368 B/op`, and `83691 allocs/op`; and kept the retained book route green
at `37.58s` wall, `33.36s` decode, `3021` generated tokens, `80.39 tok/s`
average, `69.81 tok/s` on turn 10, empty stderr, no cap hits, and chapter-10
anchor hits of `3`, with `231825488 B/op` and `221859 allocs/op`. Moving the
suppressed-token fallback onto a second device greedy pass with a workspace
cached suppress-token buffer then cut the short guard to `7868528 B/op` and the
chapter-shaped guard to `15985280 B/op` while keeping decode at `108.9 tok/s`
and `101.5 tok/s` respectively. The retained book route stayed green at
`37.64s` wall, `33.43s` decode, `3021` generated tokens, `80.26 tok/s` average,
`69.56 tok/s` on turn 10, empty stderr, no cap hits, and chapter-10 anchor hits
of `3`, with `231896312 B/op` and `221840 allocs/op`. Caching the workspace
single-token device buffer value then removed the duplicate per-forward token
upload used by the base embedding and per-layer embedding paths. The 2048-token
guards stayed above the speed floor at `108.8 tok/s`, `7821264 B/op`, and
`62060 allocs/op` for `text:Hi`, and `100.0 tok/s`, `15935904 B/op`, and
`77529 allocs/op` for the chapter-shaped prompt. The retained book route stayed
green at `38.21s` wall, `33.99s` decode, `3021` generated tokens, `79.06 tok/s`
average, `67.79 tok/s` on turn 10, empty stderr, no cap hits, and chapter-10
anchor hits of `3`, with `231843976 B/op` and `212778 allocs/op`. Reading the
packed q4 greedy result directly from the embedding kernels then removed the
per-token host-to-device token upload on the greedy decode path. The
2048-token guards stayed green at `109.0 tok/s`, `7770440 B/op`, and
`55915 allocs/op` for `text:Hi`, and `101.5 tok/s`, `15932504 B/op`, and
`71533 allocs/op` for the chapter-shaped prompt. The retained book route stayed
green at `37.68s` wall, `33.46s` decode, `3021` generated tokens,
`80.17 tok/s` average, `69.77 tok/s` on turn 10, empty stderr, no cap hits, and
chapter-10 anchor hits of `3`, with `231837680 B/op` and `212770 allocs/op`.
Replacing Go-side cgo output pointer calls in the hot HIP bridge with
result-return C wrappers then removed a large class of per-launch/per-allocation
Go heap objects without changing HIP behavior. A small 8-entry initial free-list
bucket for the cgo device-memory pool then cut a few thousand more objects
without the `B/op` bloat of the rejected 64/512-entry buckets. The 2048-token
guards stayed green at `109.0 tok/s`, `7393992 B/op`, and `11646 allocs/op` for
`text:Hi`, and `101.4 tok/s`, `15387256 B/op`, and `13344 allocs/op` for the
chapter-shaped prompt. The retained book route stayed green at `37.65s` wall,
`33.43s` decode, `3021` generated tokens, `80.24 tok/s` average,
`69.97 tok/s` on turn 10, empty stderr, no cap hits, and chapter-10 anchor hits
of `3`, with `230971104 B/op` and `109680 allocs/op`.
Collapsing duplicate VRAM metric reads and caching the selected sysfs VRAM path
then trimmed the 2048-token benchmark accounting path to `108.9 tok/s`,
`7339968 B/op`, and `10853 allocs/op` for `text:Hi`, and `101.4 tok/s`,
`15369792 B/op`, and `12555 allocs/op` for the chapter-shaped prompt. The
retained book route stayed green at `37.71s` wall, `33.49s` decode, `3021`
generated tokens, `80.11 tok/s` average, `69.90 tok/s` on turn 10, empty
stderr, no cap hits, and chapter-10 anchor hits of `3`, with `230972112 B/op`
and `109684 allocs/op`.
Reading the final greedy q4 result through a scalar cgo `uint64` copy then
removed the per-token escaping Go byte slice, and the cgo device-memory pool now
keeps a first pointer inline per size so one-off retained descriptor sizes do
not allocate a backing slice just to cache one pointer. The 2048-token guards
stayed green at `108.9 tok/s`, `7392408 B/op`, and `8814 allocs/op` for
`text:Hi`, and `101.5 tok/s`, `15384152 B/op`, and `10428 allocs/op` for the
chapter-shaped prompt. The retained book route stayed green at `37.62s` wall,
`33.40s` decode, `3021` generated tokens, `80.31 tok/s` average,
`69.87 tok/s` on turn 10, empty stderr, no cap hits, and chapter-10 anchor hits
of `3`, with `230986816 B/op` and `106647 allocs/op`. A same-batch attempt to
avoid the second local-window page-slice copy was rejected: it reduced the
chapter-shaped 2048-token allocation count but slowed the retained book to
`38.24s` wall and `79.01 tok/s`.
This is still not the final driver endpoint: later-turn decode is only
`69.87 tok/s`, below the `90-100+ tok/s` target, and the visible output remains
repetitive. Keep tuning retained long-context attention and state quality until
the later turns stay near the target.

Current decode-scaling status as of 2026-05-26: Gemma4 E2B/E4B context is
`128k` tokens, not `128` tokens; the context-128 short decode numbers remain a
hot-loop diagnostic only. With the chapter-1 prompt changed to avoid saying
`10 chapter` up front, the RX 7800 XT `gfx1100` q4 path now shows:

```text
context_len=4096, prompt="chapter 1 of a book..." on Gemma4-E2B q4

max_new_tokens  route                    tok/s   B/op       allocs/op
1024            chunked-128 device KV     95.23  181.78M     1734305
2048            chunked-128 device KV     90.53  314.81M     3426653
4096            non-chunked value-fast    72.45  633.26M     6833550
4096            chunked-128 query cache   80.33  885.71M     6749654
2048 text:Hi    descriptor wrapper pool  103.4    50.91M      281087
2048 text:Hi    token/PLE workspace      103.5    49.54M      255531
2048 text:Hi    state handoff cleanup    103.6    36.32M      110184
2048 chapter    state handoff cleanup     92.22   60.79M      125647
2048 text:Hi    cache/probe cleanup      103.4    20.40M       72262
2048 chapter    cache/probe cleanup       90.82   44.95M       87745
2048 text:Hi    per-layer set workspace  109.0    17.31M       68198
2048 chapter    per-layer set workspace  101.2    44.34M       83691
2048 text:Hi    device suppress fallback 108.9     7.87M       68194
2048 chapter    device suppress fallback 101.5    15.99M       83672
2048 text:Hi    token value cache       108.8     7.82M       62060
2048 chapter    token value cache       100.0    15.94M       77529
2048 text:Hi    greedy-token embedding 109.0     7.77M       55915
2048 chapter    greedy-token embedding 101.5    15.93M       71533
2048 text:Hi    cgo/free-list cleanup  109.0     7.39M       11646
2048 chapter    cgo/free-list cleanup  101.4    15.39M       13344
2048 text:Hi    VRAM metrics cache     108.9     7.34M       10853
2048 chapter    VRAM metrics cache     101.4    15.37M       12555
2048 text:Hi    scalar greedy read     108.9     7.39M        8814
2048 chapter    scalar greedy read     101.5    15.38M       10428
```

The earlier 256-token chunked route was rejected because it fell to `58.24
tok/s` at 4096 tokens and made the retained book slower. The current 128-token
chunked route is accepted as the default because it keeps the short `text:Hi`
2048-token hot-loop gate above `100 tok/s`, improves the chapter-prompt
4096-token diagnostic from `72.45` to `80.33 tok/s`, and improves the full-cap
retained book route from `60.2s` to `41.3s` wall. Book acceptance is based on
the retained-state arc gate and measured wall/decode metrics, not byte-identical
story text across separate generations.
`GO_ROCM_GEMMA4_Q4_CHUNKED_ATTENTION=0` remains the explicit escape hatch.

Fresh long-decode evidence after removing the invalid 512-token chapter cap:
a single-stream 4096-token generation from the chapter prompt originally
completed with empty stderr at `59315672504 ns/op`, `69.05 tok/s`,
`715310248 B/op`, and `6833978 allocs/op`. After adding the direct retained
KQ8/VQ4 one-token value fast path for the no-shared-metadata region, the same
4096-token diagnostic byte-matches the previous greedy output and reports
`56533521855 ns/op`, `72.45 tok/s`, `633262152 B/op`, and `6833550 allocs/op`.
With the default chunked-128 route plus the separate stage-1 query cache, the
same 4096-token diagnostic reports `50991794554 ns/op`, `80.33 tok/s`,
`885711080 B/op`, and `6749654 allocs/op`, with empty stderr.
`rocprof --stats` on the pre-fast-path shape showed the kernel ceiling clearly:
`rocm_attention_heads` was `46.48%` of GPU kernel time (`22.203s` over `143325`
launches), followed by `rocm_mlx_q4_projection` at `17.67%`. After the value
fast path, `rocm_attention_heads` is still first at `44.20%` (`20.228s` over
the same `143325` launches), with q4 projection second at `18.52%`. The slowdown
is therefore not a timing artifact or retained-state replay bug. With the
default chunked-128 route, the active profile is better but still bounded by the
attention/projection hot path: `rocm_attention_heads_chunked_stage1` is `34.23%`
of GPU kernel time, q4 projection is `21.71%`, and chunked stage2 is only
`2.06%`. The next target is stage1 score/value work and q4 projection, not the
old single-kernel full-attention fallback.

Rejected path: a metadata-only shared-memory cache for direct-token K/V value
pointers/scales after weights move to global memory. A 64 KiB version hit
`HSA_STATUS_ERROR_INVALID_ALLOCATION` and timed out, with the error captured in
`book-retained-10turn-fullcap-greedy-default-block1-metadata.err`. A conservative
48 KiB guard completed with empty stderr but changed deterministic generation
enough to fail chapter-10 arc retention (`2` anchor hits), took `114.4s` wall,
and only reached `36.4 tok/s` on turn 10. Do not restore that path as-is.

Rejected micro-optimization: a direct one-token Q8 key-dot shortcut for the same
KQ8/VQ4 descriptor layout. It preserved deterministic 4096-token output and kept
stderr empty, but measured `72.38 tok/s`, slightly below the value-only fast
path, so it was removed.

Retained book smoke also exposed a separate production blocker: sampled book
generation still uses host-side sampling, which forces the full logits path and
destroys wall time. A 2-turn, 8-token-per-turn retained smoke with the revised
"more chapters" prompt completed in `101.2s` with sampling defaults
(`book_temperature=1`, `top_p=0.95`, `top_k=64`) and spent `86.88s` in decode
for only `16` generated tokens. The same run with greedy/device sampling
(`GO_ROCM_BOOK_TEMPERATURE=0`, `GO_ROCM_BOOK_TOP_P=0`,
`GO_ROCM_BOOK_TOP_K=0`) completed in `0.644s`, with `0.1637s` decode for
`16` tokens, about `97.7 tok/s` decode. The next production decision is
therefore either a device-side sampler for top-k/top-p or making the retained
book production route greedy/device-resident until that sampler lands.

The replay-style book benchmark is deliberately double-gated with
`GO_ROCM_RUN_UNSAFE_REPLAY_BOOK_BENCHMARKS=1` and has a per-turn timeout because
it can monopolize a display GPU. Use it only as a baseline/debug aid. The route
to production is a retained-state book benchmark, not repeated full-manuscript
prompt replay.

```text
Diagnostic go-rocm Gemma4-E2B q4, context_len=48000, max_new_tokens=1

prompt  best current result      prompt tok/s  notes
2k      5963139198 ns/op         343.4         batched prefill, 512-token ubatches
4k      12840551074 ns/op        319.0         batched prefill, 512-token ubatches
29k     189761718960 ns/op       152.8         true opencode/IDE session-start base
48k     not rerun post-batch     unknown       only run after 29k improves
```

For comparison, upstream llama.cpp built locally with HIP for `gfx1100` and run
against the Hugging Face Gemma4 GGUF
`/home/claude/models/hf/unsloth-gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf`
on the same `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85` device reports:

```text
llama-bench -p 2048,4096,8192,16384,32768 -n 0 -b 2048 -ub 512 -fa 1 -ngl 99

pp2048   4788.12 tok/s
pp4096   4519.58 tok/s
pp8192   4071.68 tok/s
pp16384  3414.32 tok/s
pp32768  2580.52 tok/s
```

This is the implementation target shape. The gap is not a Go benchmark timing
bug. The ROCm driver now has a batched prefill scaffold, but it still needs the
production kernel layout llama.cpp relies on: larger tiled q4 matmul/dequant,
slot-based KV preparation, separate base/full and SWA KV caches for hybrid
Gemma attention, output flags so only requested tokens emit logits, and
flash-attention-style prompt attention.

## Current Performance Evidence

This chronology starts from the point where the RX 7800 XT was selected
correctly but the q4 path was still far from the endpoint:

- Baseline q4 before driver launch caching:
  - 1 token: `4244871937 ns/op`, `0.2356 tok/s`.
  - 4 tokens: `12228533217 ns/op`, `0.3271 tok/s`.
  - 8 tokens: `26421638390 ns/op`, `0.3028 tok/s`.
- After HSACO/function caching, launch-argument reuse, and attention output-only
  readback:
  - 1 token: `1151848888 ns/op`, `0.8682 tok/s`, `89530672 B/op`, `79132 allocs/op`.
  - 8 tokens: `11579650685 ns/op`, `0.6909 tok/s`, `369906856 B/op`, `336178 allocs/op`.
- After device-buffer q4 MLP, attention concat, cgo small-buffer pooling,
  batched RoPE-query upload, `rocm_attention_heads`, and validation allocation
  cleanup:
  - 1 token: `738894378 ns/op`, `1.353 tok/s`, `68661432 B/op`, `61727 allocs/op`.
  - 8 tokens: `4011068954 ns/op`, `1.994 tok/s`, `182386392 B/op`, `248860 allocs/op`.
  - 64 tokens: `197371659447 ns/op`, `0.3243 tok/s`, `1480160960 B/op`, `1764878 allocs/op`.
- After fixing the `rocm_attention_heads` parallel-thread barrier bug:
  - 1 token: `714107200 ns/op`, `1.400 tok/s`, `42511756 B/op`, `58858 allocs/op`.
  - 8 tokens: `3104406559 ns/op`, `2.577 tok/s`, `182394600 B/op`, `248861 allocs/op`.
  - 64 tokens: `23042303885 ns/op`, `2.778 tok/s`, `1480171840 B/op`, `1764903 allocs/op`.
  - 512 tokens: `472847717598 ns/op`, `1.083 tok/s`, `22746415544 B/op`, `13935245 allocs/op`.
- After moving final q4 generation softcap+greedy sampling onto the device:
  - 8 tokens: `2956576812 ns/op`, `2.706 tok/s`, `144474512 B/op`, `253281 allocs/op`.
  - 64 tokens: `22031539496 ns/op`, `2.905 tok/s`, `1160976584 B/op`, `1792276 allocs/op`.
- After chaining more decoder residual/norm/MLP buffers on device, parallelizing
  the MLX q4 projection kernel across 256 threads per output row, letting
  `rocm_attention_heads` use the launch-provided device weights buffer instead
  of a fixed 1024-token shared array, and keeping the query-side input norm,
  q projection, per-head q norm, q RoPE, full-rotary K/V projection/norm/RoPE,
  layer-to-layer final hidden handoff, and most generation KV state
  device-resident:
  - 8 tokens: `1191835181 ns/op`, `6.712 tok/s`, `36596736 B/op`, `190560 allocs/op`.
  - 64 tokens: `8736626007 ns/op`, `7.325 tok/s`, `367727560 B/op`, `1343699 allocs/op`.
  - 512 tokens: `83431887777 ns/op`, `6.137 tok/s`, `10407447176 B/op`, `10714740 allocs/op`.
  - 2000 tokens: `560469026224 ns/op`, `3.568 tok/s`, `102782893072 B/op`, `47106776 allocs/op`.
- After letting shared-KV layers alias the source layer's device KV cache during
  generation and making device-only appends respect sliding-window token limits:
  - 8 tokens: `1184829954 ns/op`, `6.752 tok/s`, `33227640 B/op`, `188352 allocs/op`.
  - 64 tokens: `8690617315 ns/op`, `7.364 tok/s`, `236095352 B/op`, `1328803 allocs/op`.
  - 512 tokens: `81569911331 ns/op`, `6.277 tok/s`, `2983945448 B/op`, `10596655 allocs/op`.
  - 2000 tokens: `399875368790 ns/op`, `5.002 tok/s`, `18552099048 B/op`, `41802725 allocs/op`.
- After keeping Gemma4 per-layer input precompute on device in generation mode,
  adding an async HIP launch-argument ring, and removing the synthetic
  RMSNormNoScale weight upload with a unit-weight RMSNorm mode:
  - 8 tokens: `1076450260 ns/op`, `7.432 tok/s`, `26174952 B/op`, `218629 allocs/op`.
  - 64 tokens: `7925118540 ns/op`, `8.076 tok/s`, `180647680 B/op`, `1393308 allocs/op`.
  - 512 tokens: `75716855758 ns/op`, `6.762 tok/s`, `2541431872 B/op`, `10933834 allocs/op`.
  - 2000 tokens: `375619197263 ns/op`, `5.325 tok/s`, `16826757936 B/op`, `43054536 allocs/op`.
- After encoding appended q4 KV tokens on device, building appended KV
  descriptor tables on device, moving the partial-RoPE/global-attention branch
  onto device buffers, and letting shared-KV layers borrow descriptor tables:
  - 64 tokens: `7551867334 ns/op`, `8.475 tok/s`, `73137264 B/op`, `1321653 allocs/op`.
  - 512 tokens: `73320058318 ns/op`, `6.983 tok/s`, `1003985368 B/op`, `10294629 allocs/op`.
  - 2000 tokens: `371838641006 ns/op`, `5.379 tok/s`, `7182703848 B/op`, `40270820 allocs/op`.
- After parallelizing the RMSNorm and RoPE hot-path kernels, adding a
  token-parallel path for long-context `rocm_attention_heads`, and tuning the
  MLX q4 projection row block from 256 to 64 threads:
  - 64 tokens: `2660767496 ns/op`, `24.05 tok/s`, `73145736 B/op`, `1321665 allocs/op`.
  - 512 tokens: `29556806238 ns/op`, `17.32 tok/s`, `1003999768 B/op`, `10294665 allocs/op`.
  - 2000 tokens: `171399522008 ns/op`, `11.67 tok/s`, `7182709984 B/op`, `40270831 allocs/op`.
- After fusing the final q4 LM-head projection with greedy sampling, grouping
  four q4 projection rows per block, using dynamic shared memory for
  attention-head weights through 2048 tokens, and replacing the final greedy
  result host clear with queued `hipMemsetAsync`, with the final greedy result
  buffer reused across public q4 generation:
  - 64 tokens: `2662322096 ns/op`, `24.04 tok/s`, `72951368 B/op`, `1315976 allocs/op`.
  - 512 tokens: `29109745679 ns/op`, `17.59 tok/s`, `1001904552 B/op`, `10180944 allocs/op`.
  - 2000 tokens: `166868660548 ns/op`, `11.99 tok/s`, `7172691968 B/op`, `39620903 allocs/op`.
- After changing the q4 projection kernels to consume one packed q4 word at a
  time, applying affine scale/bias once per packed word, and retuning the row
  block to eight rows per 256-thread block:
  - 64 tokens: `1875609269 ns/op`, `34.12 tok/s`, `72968536 B/op`, `1316130 allocs/op`.
  - 512 tokens: `22809017763 ns/op`, `22.45 tok/s`, `1001905288 B/op`, `10180966 allocs/op`.
  - 2000 tokens: `142007905804 ns/op`, `14.08 tok/s`, `7172698912 B/op`, `39620987 allocs/op`.
- After batching query-head RMSNorm/RoPE, batching per-layer input precompute,
  fusing q4 MLP gate/up projection plus GELU multiply, and fusing the
  per-layer input gate projection plus GELU/multiplier:
  - 64 tokens: `1223892467 ns/op`, `52.29 tok/s`, `48503240 B/op`, `745012 allocs/op`.
  - 512 tokens: `17538093964 ns/op`, `29.19 tok/s`, `808699528 B/op`, `5674501 allocs/op`.
  - 2000 tokens: `119662262683 ns/op`, `16.71 tok/s`, `6418807824 B/op`, `22025505 allocs/op`.
- After adding `rocm_rms_norm_residual_add` and wiring the q4 decoder layer's
  post-attention, post-MLP, and per-layer-input residual paths through it:
  - 1 token: `34607793 ns/op`, `28.90 tok/s`, `4572542 B/op`, `24287 allocs/op`.
  - 8 tokens: `148062716 ns/op`, `54.03 tok/s`, `8101539 B/op`, `91284 allocs/op`.
  - 64 tokens: `1136882187 ns/op`, `56.29 tok/s`, `44681336 B/op`, `649458 allocs/op`.
  - 512 tokens: `16919381071 ns/op`, `30.26 tok/s`, `778538352 B/op`, `4920554 allocs/op`.
  - 2000 tokens: `117417290830 ns/op`, `17.03 tok/s`, `6301139640 B/op`, `19085556 allocs/op`.
- After hoisting device-KV q8/q4 scale decoding out of the per-dimension key
  dot-product loop in `rocm_attention_heads`:
  - 1 token: `30529386 ns/op`, `32.76 tok/s`, `4570808 B/op`, `24215 allocs/op`.
  - 8 tokens: `130538347 ns/op`, `61.28 tok/s`, `8103945 B/op`, `91153 allocs/op`.
  - 64 tokens: `1012688129 ns/op`, `63.20 tok/s`, `44675504 B/op`, `649445 allocs/op`.
  - 512 tokens: `14806875246 ns/op`, `34.58 tok/s`, `778538288 B/op`, `4920555 allocs/op`.
  - 2000 tokens: `100736540182 ns/op`, `19.85 tok/s`, `6301145240 B/op`, `19085555 allocs/op`.
- After locally reusing the resolved device-KV value page and scale across the
  two possible output dimensions handled by each attention thread:
  - 1 token: `30748111 ns/op`, `32.52 tok/s`, `4570611 B/op`, `24214 allocs/op`.
  - 8 tokens: `129666170 ns/op`, `61.70 tok/s`, `8103189 B/op`, `91156 allocs/op`.
  - 64 tokens: `1024360039 ns/op`, `62.48 tok/s`, `44688848 B/op`, `649451 allocs/op`.
  - 512 tokens: `14613206249 ns/op`, `35.04 tok/s`, `778542784 B/op`, `4920565 allocs/op`.
  - 2000 tokens: `97921265524 ns/op`, `20.42 tok/s`, `6301157384 B/op`, `19085551 allocs/op`.
- After replacing the dummy host query vector in the device-query attention path
  with an explicit query dimension:
  - 1 token: `31172883 ns/op`, `32.08 tok/s`, `4484072 B/op`, `24144 allocs/op`.
  - 8 tokens: `130246232 ns/op`, `61.42 tok/s`, `7709432 B/op`, `90839 allocs/op`.
  - 64 tokens: `1023191242 ns/op`, `62.55 tok/s`, `41893224 B/op`, `647174 allocs/op`.
  - 512 tokens: `14611411980 ns/op`, `35.04 tok/s`, `756479528 B/op`, `4902607 allocs/op`.
  - 2000 tokens: `97825149305 ns/op`, `20.44 tok/s`, `6215093488 B/op`, `19015571 allocs/op`.
- After switching q4 projection reductions from shared-memory barriers to
  lane-local shuffle reductions, folding per-layer `layer_scalar` into
  RMSNorm+residual-add, and inlining q4/q8 value loads in token-parallel
  device-KV attention:
  - 1 token: `29681362 ns/op`, `33.69 tok/s`, `4441919 B/op`, `23154 allocs/op`.
  - 8 tokens: `126070629 ns/op`, `63.46 tok/s`, `7546088 B/op`, `86434 allocs/op`.
  - 64 tokens: `990174190 ns/op`, `64.64 tok/s`, `39632972 B/op`, `589064 allocs/op`.
  - 512 tokens: `14238000565 ns/op`, `35.96 tok/s`, `745887336 B/op`, `4656240 allocs/op`.
  - 2000 tokens: `96314243796 ns/op`, `20.77 tok/s`, `6174286240 B/op`, `18113302 allocs/op`.
- After fusing non-shared q4 Q/K/V projections into one HIP launch per decoder
  layer:
  - 1 token: `29023005 ns/op`, `34.46 tok/s`, `4426590 B/op`, `22484 allocs/op`.
  - 8 tokens: `120672226 ns/op`, `66.30 tok/s`, `7480734 B/op`, `83552 allocs/op`.
  - 64 tokens: `962144028 ns/op`, `66.52 tok/s`, `39138944 B/op`, `567611 allocs/op`.
  - 512 tokens: `13910662661 ns/op`, `36.81 tok/s`, `741971008 B/op`, `4481810 allocs/op`.
  - 2000 tokens: `93696267363 ns/op`, `21.35 tok/s`, `6158484288 B/op`, `17373090 allocs/op`.
- After fixing/defaulting mapped launch-argument packets, parallelizing the
  no-trim device KV descriptor append path, caching each head query in shared
  memory for the token-parallel score phase, consuming q8 key dots four values
  per packed word, using 512-thread attention-head blocks for contexts of at
  least 512 tokens, and splitting q4 device-KV value accumulation across paired
  output dimensions:
  - 1 token: `24732456 ns/op`, `40.43 tok/s`, `4414922 B/op`, `22246 allocs/op`.
  - 8 tokens: `104624440 ns/op`, `76.46 tok/s`, `7448191 B/op`, `83144 allocs/op`.
  - 64 tokens: `808117325 ns/op`, `79.20 tok/s`, `39175988 B/op`, `567619 allocs/op`.
  - 512 tokens: `11832510557 ns/op`, `43.27 tok/s`, `741984440 B/op`, `4473626 allocs/op`.
  - 2000 tokens: `69384374786 ns/op`, `28.82 tok/s`, `6157314256 B/op`, `17364758 allocs/op`.
- After caching q4 value page pointers/scales in dynamic shared memory,
  switching `rocm_attention_heads` device-KV validation from a full per-head
  page walk to header validation, normalizing softmax weights once before value
  accumulation, and hoisting direct one-token page addressing in the attention
  block:
  - 1 token: `24482634 ns/op`, `40.85 tok/s`, `4414351 B/op`, `22246 allocs/op`.
  - 8 tokens: `101731908 ns/op`, `78.64 tok/s`, `7447574 B/op`, `83144 allocs/op`.
  - 64 tokens: `744144350 ns/op`, `86.00 tok/s`, `39139076 B/op`, `567615 allocs/op`.
  - 512 tokens: `7948177819 ns/op`, `64.42 tok/s`, `741984392 B/op`, `4473624 allocs/op`.
  - 2000 tokens: `34231495412 ns/op`, `58.43 tok/s`, `6157313288 B/op`, `17364747 allocs/op`.
- After making shared-device-KV borrowed aliases share the source page slice
  without owning/freeing source pages, and pooling hot-path copied KV page
  metadata slices:
  - 2000 tokens: `34170996556 ns/op`, `58.53 tok/s`, `802486024 B/op`, `17323617 allocs/op`.
  - 512 tokens: `7959242091 ns/op`, `64.33 tok/s`, `214885656 B/op`, `4462676 allocs/op`.
  - 2048 tokens: `34952296158 ns/op`, `58.59 tok/s`, `822547632 B/op`, `17738577 allocs/op`.
  This removes most remaining host-side KV page metadata allocation, but it does
  not move the endpoint enough to change the kernel-side diagnosis.
- After rebuilding the normal `gfx1100 -O2` HSACO from the reverted source after
  the latest rejected experiments:
  - 512 tokens: `8069296382 ns/op`, `63.45 tok/s`, `214434344 B/op`, `4462654 allocs/op`.
  - 2000 tokens: `34126497340 ns/op`, `58.61 tok/s`, `803767264 B/op`, `17323657 allocs/op`.
  These fresh checks keep the endpoint at roughly the same `58-59 tok/s`
  long-context plateau.
- After unrolling the default q8 device-KV key dot over four packed words per
  loop iteration:
  - 512 tokens: `7828455100 ns/op`, `65.40 tok/s`, `214654840 B/op`, `4462648 allocs/op`.
  - 2000 tokens: `33268537694 ns/op`, `60.12 tok/s`, `803144400 B/op`, `17323640 allocs/op`.
  - 2048 tokens: `34097066745 ns/op`, `60.06 tok/s`, `822774688 B/op`, `17738584 allocs/op`.
  This is a small kept attention gain, but it still leaves the endpoint far
  below `100 tok/s`.
- After adding a direct q4 value-accumulation branch for the default direct
  `k-q8-v-q4` descriptor layout in `rocm_attention_heads`:
  - 512 tokens: `7822076555 ns/op`, `65.46 tok/s`, `214674576 B/op`, `4462657 allocs/op`.
  - 2000 tokens: `32902163632 ns/op`, `60.79 tok/s`, `803769656 B/op`, `17323601 allocs/op`.
  - 2048 tokens: `33781915323 ns/op`, `60.62 tok/s`, `823835248 B/op`, `17738597 allocs/op`.
  This was the kept long-context plateau before the later q4 value pre-scale;
  the endpoint was still well below `100 tok/s`.
- A fresh verification pass after the kept attention patches measured:
  - 512 tokens: `7834757484 ns/op`, `65.35 tok/s`, `214893064 B/op`, `4462717 allocs/op`.
  - 2048 tokens: `33742782570 ns/op`, `60.69 tok/s`, `822770152 B/op`, `17738590 allocs/op`.
  This confirms the direct q4 value branch did not get the driver close to the
  endpoint, and the long-run timer is not the measurement bug.
- After cleaning up the HIP launch path by caching process-level availability,
  caching HSACO modules by path without per-launch `stat`/key rebuilding,
  avoiding redundant launch-argument slice copies, and replacing the
  per-launch launch-argument finish closure with a value lease:
  - 512 tokens: `7799788531 ns/op`, `65.64 tok/s`, `74929800 B/op`, `787760 allocs/op`.
  - 2000 tokens: `32768500713 ns/op`, `61.03 tok/s`, `258035584 B/op`, `2988572 allocs/op`.
  - 2048 tokens: `33646087144 ns/op`, `60.87 tok/s`, `264818936 B/op`, `3059627 allocs/op`.
  This is a kept driver cleanup because it removes millions of Go allocations
  from the endpoint runs, but it is not a route to `100+ tok/s` by itself.
- A follow-up driver/HIP pass found two more runner/launch details:
  - `GO_ROCM_DISABLE_ASYNC_LAUNCH_ARGS=1` was unsafe because the single launch
    packet could be reused before queued kernels had consumed it, producing HIP
    error `700`. That opt-in path now synchronizes before releasing the single
    packet; q4 smoke under the env passed with prompt tokens `[2 10979]` and
    generated token `[236764]`. The default async launch-argument ring is
    unchanged.
  - Forcing the RX 7800 XT into high/manual clocks did not materially move the
    benchmark. The default path still measured `7814954298 ns/op` at
    `65.52 tok/s` for 512 tokens and `32762598372 ns/op` at `61.05 tok/s` for
    2000 tokens. A fresh 512-token `rocprof --stats` still puts
    `rocm_attention_heads` at `42.59%`, q4 projection at `18.47%`, q4
    GELU/multiply at `10.89%`, RMSNorm/residual at about `12.04%`, descriptor
    append at `1.06%`, and copy kernels at only `0.19%`.
- A query-head RMSNorm+RoPE fusion experiment passed q4 smoke with generated
  tokens `[236764 3307]` and was neutral at 512 tokens (`7813047170 ns/op`,
  `65.53 tok/s`, `72103632 B/op`, `739059 allocs/op`), but it regressed repeat
  2000-token endpoint checks to `58.78 tok/s` and `58.73 tok/s`. After reverting
  the fusion and rebuilding the `gfx1100` HSACO, the 2000-token run returned to
  `32737071937 ns/op`, `61.09 tok/s`, `258238264 B/op`, `2988563 allocs/op`.
  This confirmed small query-side launch fusion was not the post-60 tok/s bottleneck;
  keep attention/q4 GEMV/MLP fusion as the main endpoint path.
- A group-tiled MLX q4 projection row-sum helper is kept for
  `rocm_mlx_q4_projection`, triple projection, final projection+greedy, and
  q4 GELU projection. It loads affine scale/bias once per q4 group instead of
  once per packed 8-column word. Q4 smoke still generated `[236764 3307]`;
  checks measured:
  - 512 tokens: `7785336740 ns/op`, `65.76 tok/s`, `74929832 B/op`, `787765 allocs/op`.
  - 2000 tokens: `32642251550 ns/op`, `61.27 tok/s`, `258875656 B/op`, `2988551 allocs/op`.
  - 2048 tokens: `33542086457 ns/op`, `61.06 tok/s`, `264819224 B/op`, `3059631 allocs/op`.
  This is a small kept q4 GEMV cleanup, still far below the `100+ tok/s`
  endpoint. A profiled 512-token run after this change measured `60.38 tok/s`
  under `rocprof --stats`; the split was `rocm_attention_heads` `42.93%`,
  q4 projection `17.85%`, q4 GELU/multiply `10.90%`, RMSNorm/residual
  `12.10%`, q4 final projection+greedy `5.26%`, and copies `0.20%`.
- Applying the same group-tiled scale/bias loop to the paired gate/up
  `rocm_mlx_q4_gelu_tanh_multiply` kernel preserved q4 smoke but regressed the
  512-token check to `8157714213 ns/op`, `62.76 tok/s`, so that part was
  reverted.
- Retuning the q4 projection-family launch from 8 rows/block to 16 rows/block
  after the group-tiled row-sum helper also preserved q4 smoke but regressed the
  512-token check to `7960843350 ns/op`, `64.31 tok/s`, so the kept launch
  shape remains 8 rows per 256-thread block.
- A dedicated small-column q4 projection kernel for the 256/512-column attention
  output projections also preserved live q4 smoke (`[236764 3307]`), but it
  measured `7815026708 ns/op`, `65.51 tok/s` at 512 tokens and
  `32820868597 ns/op`, `60.94 tok/s` at 2000 tokens, below the kept
  `65.76`/`61.27 tok/s` endpoints, so it was reverted.
- A fused post-attention residual-add plus pre-FFN RMSNorm kernel,
  `rocm_rms_norm_residual_add_norm`, is kept. It removes one launch and one
  device-buffer round trip from each decoder layer's post-attention path. Q4
  smoke still generated `[236764 3307]`; checks measured:
  - 512 tokens: `7684228195 ns/op`, `66.63 tok/s`, `74649224 B/op`, `769797 allocs/op`.
  - 2000 tokens: `32170028940 ns/op`, `62.17 tok/s`, `257342976 B/op`, `2918526 allocs/op`.
  - 2048 tokens: `32973719595 ns/op`, `62.11 tok/s`, `263898288 B/op`, `2987923 allocs/op`.
  This is a small kept layer-graph cleanup, still far below the `100+ tok/s`
  endpoint. A profiled 512-token run after this change measured `61.83 tok/s`
  under `rocprof --stats`; the split was `rocm_attention_heads` `43.14%`,
  q4 projection `17.93%`, q4 GELU/multiply `10.93%`, RMSNorm/residual
  `11.83%`, q4 final projection+greedy `5.26%`, and copies `0.19%`.
- Cross-layer input-norm precompute is kept together with a wave-shuffle
  attention max/sum reduction. The q4 hot path now lets a layer's final
  residual-add launch also write the next layer's input RMSNorm output, and the
  next layer consumes that precomputed buffer instead of launching a separate
  `rocm_rms_norm`. Q4 smoke still generated `[236764 3307]`, and the BF16
  layer-0 tied LM-head anchor still reported token `158750`. Checks measured:
  - 512 tokens: `7505556636 ns/op`, `68.22 tok/s`, `75411088 B/op`, `769926 allocs/op`.
  - 2000 tokens: `31411112083 ns/op`, `63.67 tok/s`, `259325968 B/op`, `2920163 allocs/op`.
  - 2048 tokens: `32356592135 ns/op`, `63.29 tok/s`, `265920536 B/op`, `2989574 allocs/op`.
  This is a kept layer-graph cleanup, but it still leaves the endpoint far
  below `100+ tok/s`. A profiled 512-token run after this change measured
  `63.71 tok/s` under `rocprof --stats`; the split was
  `rocm_attention_heads` `43.38%`, q4 projection `18.05%`, q4 GELU/multiply
  `11.02%`, RMSNorm/residual-family kernels about `11.18%`, q4 final
  projection+greedy `5.31%`, and copies `0.20%`.
- Final-token final RMSNorm precompute and the short-context attention geometry
  fix are kept. The last decoder layer now writes the final model RMSNorm output
  for the LM-head path, `rocm_attention_heads` uses the 512-thread paired q4
  value path from 16 tokens onward, and q4 value page pointers/scales are cached
  in dynamic shared memory from that same threshold instead of waiting until
  512 tokens. Q4 smoke still generated `[236764 3307]`; checks measured:
  - 512 tokens: `5659309906 ns/op`, `90.47 tok/s`, `75225224 B/op`, `769938 allocs/op`.
  - 2000 tokens: `29611416789 ns/op`, `67.54 tok/s`, `259603056 B/op`, `2920165 allocs/op`.
  - 2048 tokens: `30357995463 ns/op`, `67.46 tok/s`, `266414472 B/op`, `2989597 allocs/op`.
  A profiled 512-token run after this change measured `82.64 tok/s` under
  `rocprof --stats`; the split moved to `rocm_mlx_q4_projection` `26.54%`,
  `rocm_attention_heads` `17.08%`, q4 GELU/multiply `16.17%`,
  RMSNorm/residual-family kernels about `16.30%`, q4 final projection+greedy
  `7.71%`, descriptor append `1.56%`, and copies `0.28%`.
- Wave-shuffle block reductions for RMSNorm/residual kernels and fused
  RMSNorm+RoPE heads for q4 query/key vectors are kept. The fused kernel
  removes the separate query-head RMSNorm/RoPE launches and the key
  RMSNorm/RoPE launch pair from each decoder layer. Q4 smoke still generated
  `[236764 3307]`; checks measured:
  - 512 tokens: `5429310468 ns/op`, `94.30 tok/s`, `73470456 B/op`, `726184 allocs/op`.
  - 2000 tokens: `28966745166 ns/op`, `69.04 tok/s`, `251720040 B/op`, `2749949 allocs/op`.
  - 2048 tokens: `29755291471 ns/op`, `68.83 tok/s`, `258122472 B/op`, `2815303 allocs/op`.
  A profiled 512-token run after this change measured `86.45 tok/s` under
  `rocprof --stats`; the split was `rocm_mlx_q4_projection` `27.22%`,
  `rocm_attention_heads` `17.48%`, q4 GELU/multiply `16.63%`,
  `rocm_rms_norm_residual_add_norm` `10.79%`, q4 final projection+greedy
  `7.86%`, fused RMSNorm+RoPE heads `3.63%`, residual-add `3.41%`,
  q4 triple projection `3.39%`, and copies `0.29%`.
  Two follow-up experiments were rejected: shared-input q4 GEMV caching
  regressed 512 tokens to `74.40 tok/s`, and block-level q4 greedy atomics
  regressed 512 tokens to `93.79 tok/s` versus the kept `94.30 tok/s`.
- A fast sliding-window descriptor-trim path in `rocm_kv_descriptor_append` is
  kept. The 2k profile showed the old trim fallback had become a long-context
  bottleneck: descriptor append consumed `18.48%` of profiled GPU time because
  every sliding-window step copied retained one-token page descriptors on one
  thread. The new branch parallel-copies the common direct one-token-page trim
  layout and then appends the new page. Q4 smoke still generated
  `[236764 3307]`; checks measured:
  - 512 tokens: `5405693975 ns/op`, `94.71 tok/s`, `73465472 B/op`, `726181 allocs/op`.
  - 2000 tokens: `25194731733 ns/op`, `79.38 tok/s`, `252131160 B/op`, `2749942 allocs/op`.
  - 2048 tokens: `25883692325 ns/op`, `79.12 tok/s`, `259628912 B/op`, `2815319 allocs/op`.
  A fresh 2000-token `rocprof --stats` after this patch measured `73.24 tok/s`
  under profiler and showed `rocm_kv_descriptor_append` down to `1.79%`.
  The remaining profile split is `rocm_attention_heads` `30.86%`,
  q4 projection `22.57%`, q4 GELU/multiply `13.84%`,
  `rocm_rms_norm_residual_add_norm` `9.09%`, final q4 projection+greedy
  `6.64%`, fused RMSNorm+RoPE heads `3.03%`, residual-add `2.82%`,
  q4 triple projection `2.79%`, and q4 GELU projection `2.56%`.
- A later correction pass found three endpoint-critical driver/HIP issues and
  keeps the fixes:
  - `rocm_mlx_q4_projection_greedy` was using the normal q4 projection geometry
    constants, while the normal projection path used the greedy constants. That
    contaminated later final-LM-head benchmark numbers because the fused greedy
    sampler did not cover the intended vocabulary shape. The normal projection
    path now uses eight rows per 256-thread block, the final projection+greedy
    path uses 32 rows per block, and a source regression test checks the HIP
    geometry constants against the Go launch config.
  - The launch-argument ring no longer records a HIP event for every launch
    packet in the default path. It now synchronizes only when the ring wraps
    before reusing slot 0; the old per-packet event mode remains available with
    `GO_ROCM_ENABLE_LAUNCH_ARG_EVENTS=1`.
  - The q4 layer config now respects the loaded model's requested context size.
    The benchmark loads with `inference.WithContextLen(128)`, so sliding-window
    layers use an effective window of 128 instead of the raw Gemma window of
    512; full-attention layers still use `0`.
  With the corrected q4 geometry, default sync-on-wrap launch args, fast
  attention exponentials, and effective sliding-window context, current endpoint
  checks are:
  - 512 tokens: `4363243733 ns/op`, `117.3 tok/s`, `512 tokens`.
  - 2000 tokens: `19629810772 ns/op`, `101.9 tok/s`, `2000 tokens`,
    `251745216 B/op`, `2730011 allocs/op`.
  - 2048 tokens: `20156670616 ns/op`, `101.6 tok/s`, `2048 tokens`,
    `258176312 B/op`, `2795040 allocs/op`.
  The older "mid-60s to 79 tok/s" and the intermediate "100+ at 512 only" notes
  below are historical stepping stones; the geometry/context fixes above are the
  current source of truth.
- Earlier post-barrier `GO_ROCM_BENCH_TOKENS=2000` runs were still active after
  `600s` and were interrupted by timeout. After the q4 projection, host-KV
  omission, device-KV descriptor fast-path, shared-device-KV, and sliding-window
  device-append fixes, per-layer device precompute, async launch-argument
  staging, unit-weight RMSNorm, device KV token encoding, device descriptor
  append, partial-RoPE/global device chaining, shared descriptor borrowing,
  parallel RMSNorm/RoPE, token-parallel long-context attention, 64-thread q4
  projection row blocks, fused final q4 projection+greedy sampling, grouped q4
  projection rows, dynamic shared attention weights, queued device-side final
  greedy result clearing, final greedy buffer reuse, packed-word q4 projection,
  affine q4 accumulation, batched query-head/per-layer precompute, q4 MLP local
  fusions, RMSNorm+residual-add fusion, device-KV key-dot scale hoisting,
  local value-page/scale reuse in attention, and dummy host-query allocation
  removal, q4 projection shuffle reductions, layer-scalar fusion, inlined
  q4/q8 device-KV value loads, fused q4 Q/K/V projection, mapped launch-argument
  packets, parallel descriptor append, shared-query caching for the attention
  score phase, packed q8 key-dot loads, 512-thread long-context attention head
  blocks, paired q4 value accumulation, shared q4 value metadata caching,
  header-only attention-head descriptor validation, normalized softmax weights,
  direct one-token page addressing, borrowed shared-KV aliases, pooled KV
  page metadata slices, HIP launch-path allocation cleanup, grouped q4
  projection row sums, fused post-attention residual-add/pre-FFN RMSNorm,
  attention max/sum shuffle reductions, cross-layer input-norm precompute,
  final RMSNorm precompute, the 16-token attention block/q4 metadata-cache
  threshold, RMSNorm shuffle reductions, fused RMSNorm+RoPE heads, and the
  parallel sliding-window descriptor-trim path, the descriptor-trim-era endpoint checks
  finish in `25194731733 ns/op` at `79.38 tok/s` for 2000 tokens and
  `25883692325 ns/op` at `79.12 tok/s` for 2048 tokens, so this is not a
  benchmark accounting artifact hiding `100+ tok/s`. This paragraph is
  superseded by the later q4 geometry, launch-argument, and effective-context
  fixes recorded above.
- During long endpoint runs in this pass, `rocm-smi` samples showed the RX
  7800 XT active (`84-99%`) and the onboard GPU at `0%`, so the blocker at that
  point was the ROCm driver/HIP path, not device choice.
- Fresh 2000-token `rocprof --stats` evidence after the descriptor-trim fix puts
  kernel time at roughly: `rocm_attention_heads` `30.86%`,
  `rocm_mlx_q4_projection` `22.57%`, `rocm_mlx_q4_gelu_tanh_multiply` `13.84%`,
  `rocm_rms_norm_residual_add_norm` `9.09%`, final q4 projection+greedy
  `6.64%`, fused RMSNorm+RoPE heads `3.03%`, residual-add `2.82%`, q4 triple
  projection `2.79%`, q4 GELU projection `2.56%`, descriptor append `1.79%`,
  and copy kernels only `0.20%`. The descriptor-trim-era endpoint gap was a real
  attention/projection/layer-fusion problem, not a descriptor-copy, bulk-transfer,
  or model-load problem.

## Problems Found In The Driver/HIP Path

This section records the driver/HIP issues found during the endpoint work. Some
items are historical stepping stones; the current corrected performance endpoint
is the `101.9 tok/s` 2k-token run recorded above. Remaining items are production
headroom and broader-context hardening, not blockers for the current benchmark
endpoint.

- `go/hip_gemma4_q4_layer.go` still orchestrates Gemma4 q4 decode as many tiny
  HIP primitives. The latest pass removed the largest stale host-copy paths
  from per-layer input precompute, appended KV encoding, KV descriptor rebuild,
  shared-KV descriptor setup, the partial-RoPE/global-attention branch, and
  scalar RMSNorm/RoPE loops, then batched query-head/per-layer precompute,
  fused the two most obvious q4 MLP local subgraphs, fused three
  RMSNorm+residual-add chains, fused the post-attention residual output into
  pre-FFN RMSNorm, and precomputes the next layer's input RMSNorm while writing
  the current layer's final hidden state. The token step is still not a fused
  resident layer kernel.
- `go/hip_transformer_launch.go` still has host-read helpers for tests,
  fallbacks, and debug/probe outputs, but the hot q4 generate path now mostly
  avoids full-vector readback. The 64-token profile after descriptor borrowing
  shows host-to-device copies down to about `0.37s`; the dominant apparent
  readback is the final 8-byte greedy-result copy at about `6.74s` cumulative
  because it synchronizes all previously queued GPU work. Later primitive
  parallelization moved the path into low double-digit tok/s. A fresh 64-token
  profile after packed q4 projection reports the timed generation section at
  `1.85s`; the visible host time is still mostly cgo/HIP synchronization around
  queued GPU work, including about `1.18s` in the final greedy-result D2H sync,
  `0.56s` in kernel launch plumbing, and `0.55s` in H2D launch/input staging.
  The benchmark resets the timer after `LoadModel`, so the remaining gap is real
  GPU/kernel work, not a load-time accounting artifact.
- `go/hip_small_decode.go` now has `rocm_attention_heads` for one launch across
  q heads. The first 256-thread version was incorrect because
  `rocm_attention_heads` returned non-zero threads before calling a helper that
  uses `__syncthreads()`. That barrier-participation bug made q4 output
  nondeterministic. It is fixed, and the helper no longer falls back to serial
  attention above 1024 tokens.
- The device-KV attention path had a severe descriptor lookup issue: every
  key/value element access linearly scanned the page table. Generation appends
  one device page per token, so long contexts multiplied attention work by page
  count. `rocm_attention_device_kv_page` now uses a direct token-to-page fast
  path for the common one-token-page layout, which moved 512-token generation
  from `1.277 tok/s` to `6.137 tok/s` and allowed the 2k benchmark to finish.
  Shared-device KV aliasing and sliding-window device append then moved the 2k
  run to `5.002 tok/s` and cut allocation from `102.8GB/op` to `18.6GB/op`.
  Per-layer input device precompute, async launch-argument staging, and
  unit-weight RMSNorm moved the 2k run to `5.325 tok/s` and `16.8GB/op`.
  Device KV token encoding, device descriptor append, partial/global device
  chaining, and shared descriptor borrowing moved the 2k run to `5.379 tok/s`
  and `7.18GB/op`. Parallel RMSNorm/RoPE, token-parallel long-context
  attention, and 64-thread q4 projection row blocks moved the 2k run to
  `11.67 tok/s` and `7.18GB/op`. Fusing final q4 projection+greedy sampling,
  grouping q4 projection rows, using dynamic shared attention weights, replacing
  the final greedy result host clear with queued `hipMemsetAsync`, reusing the
  final greedy result buffer, consuming packed q4 words, batching query-head and
  per-layer setup, fusing q4 MLP local subgraphs, fusing RMSNorm+residual-add
  chains, hoisting q8/q4 scale decode out of the device-KV key dot-product
  inner loop, reusing the value page/scale locally across each attention
  thread's two output dimensions, replacing dummy host query vectors with an
  explicit device-query dimension, folding layer scalar into RMSNorm+residual-add,
  inlining q4/q8 value loads, fusing non-shared q4 Q/K/V projections, fixing
  mapped launch-argument packet reuse, parallelizing the common no-trim device
  KV descriptor append, caching attention queries in shared memory, consuming
  q8 key dots four values per packed word, moving long-context attention-head
  launches to 512-thread blocks, splitting q4 value accumulation across paired
  output dimensions, caching q4 value metadata in shared memory, skipping the
  repeated full descriptor page walk in `rocm_attention_heads`, normalizing
  softmax weights once, and directly addressing one-token descriptor pages moved
  the 2k run to `58.43 tok/s` and `6.16GB/op`. Borrowed shared-KV aliases and
  pooled KV page metadata slices then cut the 2048-token allocation to
  `822.5MB/op` while measuring `58.59 tok/s`. Later launch-path allocation
  cleanup, grouped q4 row sums, residual/norm graph fusion, cross-layer norm
  precompute, final norm precompute, the attention threshold fix, RMSNorm
  shuffle reductions, and fused RMSNorm+RoPE heads lifted the 2k run to
  `69.04 tok/s` and `251.7MB/op`. The later parallel sliding-window descriptor
  trim path lifted that 2k endpoint to `79.38 tok/s` with `252.1MB/op`. The
  corrected q4 projection geometry, default sync-on-wrap launch args, and
  effective sliding-window context later lifted the current 2k benchmark to
  `101.9 tok/s`.
- The device KV path still appends one tiny K/V page per token and carries a
  descriptor table with one page descriptor per token. The descriptor append is
  now device-side and both no-trim and sliding-window trim copies have parallel
  fast paths, but `rocm_attention_heads` still launches only one block per query
  head and still streams through per-token K/V pointers. On Gemma4-E2B that is
  only eight blocks per layer-token, so the RX 7800 XT cannot be saturated by
  the attention dispatch. More production headroom likely needs a contiguous or
  paged arena layout plus a multi-block/paged attention kernel. The `go-mlx` Gemma4
  paged cache groups decode history into 256-token pages and calls fused MLX
  SDPA, while the ROCm hot path currently depends on the
  `page_count == token_count` one-token-page fast path. That is now a concrete
  design mismatch to fix, not just a micro-optimization opportunity.
- `go/hip_gemma4_q4_layer.go` no longer runs q4 MLP GELU/multiply on the host
  for the main q4 path; it now uses device GELU-tanh multiply buffers. The
  broader layer graph is still not fused.
- `rocm_mlx_q4_projection` was a one-thread-per-output-row GEMV. That was a real
  HIP bug: the RX 7800 XT could show `100%` busy while each row serially walked
  every input column. It now launches one block per output row and reduces in
  shared memory, roughly doubling short-run q4 throughput. A
  64-thread row block was initially best among 32/64/128/256. After the packed
  q4-word loop, eight rows per 256-thread block is best among 4/8/16 rows,
  moving 64 tokens to `34.12 tok/s` and 2k tokens to `14.08 tok/s`. Batched
  setup, local MLP fusions, and RMSNorm+residual-add fusion moved an
  intermediate ladder to `56.29 tok/s` at 64 tokens and `17.03 tok/s` at 2k
  tokens; the later attention scale-hoist, value-side local reuse, and query
  allocation cleanup moved the ladder to `62.55 tok/s` at 64 tokens and
  `20.44 tok/s` at 2k tokens. q4 projection shuffle reductions, layer-scalar
  fusion, q4/q8 attention value-load inlining, fused non-shared q4 Q/K/V
  projection, mapped launch packets, the parallel descriptor append fast path,
  shared-query caching, packed q8 key-dot loads, 512-thread long-context
  attention, paired q4 device-KV value accumulation, shared q4 value metadata,
  header-only attention-head descriptor validation, normalized softmax weights,
  and direct one-token page addressing
  moved the ladder to `86.00 tok/s` at 64 tokens and `58.43 tok/s`
  at 2k tokens. Borrowed shared-KV aliases plus pooled KV page metadata moved
  the fresh 2048-token run only to `58.59 tok/s`, while cutting allocation
  pressure. Launch-path cleanup, grouped q4 row sums, residual/norm graph
  fusion, cross-layer norm precompute, final norm precompute, and the 16-token
  attention block/q4 metadata-cache threshold then lifted 512 tokens to
  `90.47 tok/s` and 2k tokens to `67.54 tok/s`. The corrected final-greedy
  geometry now uses 32 rows per block and contributes to the current
  `101.9 tok/s` 2k endpoint. More headroom still likely needs a tiled q4
  projection path that reuses the input vector, uses wave/block reductions
  intentionally, and avoids one launch per tiny projection where possible.
- A shared-input variant of `rocm_mlx_q4_projection` was tested after fixing a
  potential `__syncthreads()` participation bug in the final partial row block.
  It passed the q4 smoke but regressed the 64-token benchmark to `28.60 tok/s`
  because the extra shared-memory barriers cost more than the input reuse saved.
  The experiment was reverted.
- A separate row-block shape for the final q4 projection+greedy kernel was also
  tested after the RMSNorm/residual pass. Greedy-only 16 rows per block passed
  smoke but measured `56.08 tok/s` at 64 tokens, and greedy-only 4 rows per
  block measured `55.85 tok/s`, both below the current `56.29 tok/s` baseline
  before the later attention scale-hoist. The specialization was reverted.
- Two earlier token-parallel attention cache experiments were rejected or
  superseded. Descriptor header/base caching regressed 512 tokens to
  `34.85 tok/s`. An earlier shared-scratch query cache needed an extra barrier
  and measured only `64.63 tok/s` at 64 tokens before the newer 512-thread /
  paired-value attention shape. The later scoped score-phase query cache is now
  kept because it moved the 2k endpoint from `25.20 tok/s` to `25.30 tok/s` and
  is included in the current `58.43 tok/s` ladder.
- A later q8 key-dot specialization that bypassed the generic descriptor helper
  was also rejected. It was neutral at 512 tokens (`64.31 tok/s`) but regressed
  the 2k endpoint to `58.20 tok/s` versus the kept `58.43 tok/s` path. Retesting
  1024-thread long-context attention after the descriptor-validation fixes also
  remained worse at `57.67 tok/s`, so the selector stays at 512 threads for
  contexts of at least 512 tokens.
- A narrower q8 key-dot unroll is kept: `rocm_attention_device_kv_dot_from_page`
  now consumes four packed q8 words per aligned loop iteration. It preserved q4
  smoke and moved the 2000-token endpoint to `60.12 tok/s`; the 2048-token check
  measured `60.06 tok/s`.
- A direct q4 value branch for the default direct `k-q8-v-q4` descriptor layout
  is also kept. It preserved q4 smoke and moved the endpoint to `60.79 tok/s`
  at 2000 tokens and `60.62 tok/s` at 2048 tokens.
- The short-context attention launch selector and q4 value metadata-cache
  threshold are now both kept at 16 tokens. This preserved q4 smoke and moved
  512 tokens to `90.47 tok/s`, 2000 tokens to `67.54 tok/s`, and 2048 tokens to
  `67.46 tok/s`. The previous 512-token threshold had left the paired q4 value
  path and metadata cache disabled for most of the short-context benchmark.
- Unrolling that direct q4 value token loop by two was rejected. It preserved q4
  smoke, but 512 tokens stayed at `65.40 tok/s` and 2048 tokens measured
  `60.67 tok/s`, effectively neutral/slightly below the kept path.
- Pre-scaling normalized direct q4 attention weights by cached q4 value scale
  before paired value accumulation was rejected. It preserved q4 smoke, but the
  guarded exact path measured `60.68 tok/s` at 2000 and `60.54 tok/s` at 2048.
- Splitting attention score dots into a separate multi-block kernel was
  rejected. It preserved q4 smoke, but 512 tokens measured `65.26 tok/s` and
  2048 tokens regressed to `53.76 tok/s`.
- A fast softmax exp experiment in attention was rejected. It was neutral at
  512 tokens (`64.19 tok/s`) but regressed the 2k endpoint to `58.08 tok/s`.
- Deferring softmax normalization in `rocm_attention_heads` until value
  accumulation was also rejected. Q4 smoke passed and 512 tokens measured
  `64.21 tok/s`, but the 2000-token endpoint regressed to `56.33 tok/s`; the
  separate normalized-weight pass is kept.
- A fast q4 GELU tanh approximation was rejected. It passed the q4 smoke but
  changed generated tokens, was neutral at 512 tokens (`64.44 tok/s`), and
  regressed the 2048-token endpoint to `58.38 tok/s`.
- Retesting the old attention value-metadata cache threshold at `>512` instead
  of `>=512` was rejected before the later 16-token fix. It passed smoke but
  regressed 512 tokens to `64.17 tok/s` and 2048 tokens to `55.94 tok/s`.
- Switching the experimental device KV mode to `fp16` or `q8` is not a shortcut
  to speed. Both modes failed the public q4 smoke on the second generated token,
  so the default remains `k-q8-v-q4`.
- An experimental `k-q4-v-q4` key/value cache mode was also rejected. It loaded
  and ran, but the public q4 smoke changed the generated token and failed the
  accepted-token check, so q4 keys are not a safe endpoint shortcut.
- A group-64 interpretation of the q4 scale/bias layout in
  `rocm_mlx_q4_projection` passed the q4 smoke, but regressed the 512-token
  benchmark to `59.77 tok/s` versus the kept `63-64 tok/s` path. It was
  reverted.
- The `GO_ROCM_ENABLE_ASYNC_H2D=1` runtime knob was effectively neutral on the
  current path (`64.30 tok/s` at 512 tokens), so it is not a default-fix for the
  endpoint.
- Building the kernels with toolchain `-ffast-math` passed q4 smoke but
  regressed 512 tokens to `63.05 tok/s`. Keep the normal `gfx1100 -O2` build.
- Adding `__restrict__` pointer hints to the q4 projection/GELU kernels passed
  q4 smoke, but measured `64.31 tok/s` at 512 tokens and `58.48 tok/s` at 2000
  tokens, neutral to slightly worse than the kept source. It was reverted.
- A q4-generation temporary device-buffer arena around `hipAllocateByteBuffer`
  preserved q4 smoke but increased Go allocation churn, was neutral at 512
  tokens (`64.20 tok/s` enabled versus `64.14 tok/s` disabled), and regressed
  the 2000-token endpoint to `58.33 tok/s`. It was removed. A real workspace
  still needs fewer primitive allocations and launches, not a wrapper that adds
  map churn around the current graph.
- Retuning the general q4 projection-family launch to 4 rows per 256-thread
  block was rejected because it failed the q4 projection smoke. Retuning to
  16 rows per block preserved q4 smoke but regressed 512 tokens to `87.58 tok/s`
  versus the kept `90.47 tok/s`, so the general q4 projection shape remains
  8 rows per 256-thread block.
- `rocm_rms_norm` and `rocm_rope` no longer run as single-thread hot-path
  kernels. RMSNorm now does a block reduction/writeback and RoPE maps rotary
  pairs across the grid. RMSNorm+residual-add is now fused for the three
  q4-layer residual paths and also applies Gemma4 `layer_scalar` without a
  separate vector-scale launch. Attention now avoids reloading the q8/q4 KV
  scale on every key dimension, locally reuses each resolved page and scale
  across the two possible output dimensions handled by a thread, and inlines the
  hot q4/q8 value loads. Non-shared Q/K/V q4 projections now run in one HIP
  launch per decoder layer. The subsequent mapped launch-argument,
  descriptor-append, shared-query cache, packed q8 key-dot, 512-thread
  long-context attention, paired q4 value accumulation, shared q4 value
  metadata, header-only descriptor validation, normalized weights, and direct
  page addressing fixes moved the latest 2k run to `58.43 tok/s`.
  The later host metadata pooling pass lowered 2048-token allocation to
  `822.5MB/op` but left throughput at `58.59 tok/s`. Cross-layer/final norm
  precompute and the 16-token attention threshold fix then moved 2000 tokens to
  `67.54 tok/s` and 2048 tokens to `67.46 tok/s`.
  Other norm/RoPE launches are still separate primitives that should be fused
  into the layer graph.
- `go/hip_projection_launch.go` still owns many primitive projection launches.
  The device-input path and partial/global branch conversion avoid the previous
  hot readbacks, but the graph still pays per-primitive launch and allocation
  overhead instead of running a planned q4 decode workspace.
- `go/hip_driver_cgo.go` launch overhead was a real problem and is partly fixed:
  HSACO modules/functions are cached, small device buffers are pooled, and the
  small launch-argument packet now uses an async ring. A stale mapped-host packet
  bug was found in the earlier opt-in path: one mapped launch packet was reused
  before the GPU had consumed it, producing HIP 700 failures or bad tokens. The
  mapped path now uses the synchronized async ring and is the default unless
  `GO_ROCM_DISABLE_MAPPED_LAUNCH_ARGS` is set. The remaining
  `GO_ROCM_DISABLE_ASYNC_LAUNCH_ARGS=1` single-packet debug path now calls
  `hipDeviceSynchronize` before reuse so it is safe, although intentionally
  slow. A later driver pass removed per-launch HSACO `stat`/key rebuilding,
  redundant launch-argument slice copies, process-level HIP availability
  checks, and per-launch finish-closure allocation. That cut the 2048-token
  endpoint allocation from about `822.8MB/op` and `17.7M allocs/op` to
  `264.8MB/op` and `3.06M allocs/op`, while only moving throughput to
  `60.87 tok/s`. The later attention threshold fix moved 2000 tokens to
  `67.54 tok/s`. A fresh 512-token `rocprof --stats` run after that fix shows
  `__amd_rocclr_copyBuffer` at only `0.28%`, so the next step is
  fewer launches through a device-resident/fused token step, not more copy
  tuning.
- `go/hip_tiny_model.go` was also revalidating the internally-produced q4
  forward state on every public generated token. That repeat validation has now
  been skipped in the public q4 stream, but benchmark results did not materially
  change, confirming validation is not the main limiter.
- Public q4 generation no longer reads the full LM-head logits back to Go only
  to softcap on CPU and upload them again for greedy sampling. It uses
  `rocm_softcap_greedy_sample` over the device logits and reads back only the
  8-byte greedy result. Streaming generation also omits debug-only per-layer
  tensor result fields such as the raw attention concat readback. Later device
  KV and descriptor work reduced the 64-token allocation to about `73MB/op`.
  Later primitive parallelization, projection tuning, mapped launch packets,
  descriptor append parallelization, shared-query caching, packed q8 key-dot
  loads, 512-thread long-context attention blocks, paired q4 device-KV value
  accumulation, q4 value metadata caching, header-only descriptor validation,
  normalized weights, and direct page addressing moved throughput to
  `86.00 tok/s` at 64 tokens, `64.42 tok/s` at 512 tokens, and `58.43 tok/s`
  at 2k tokens. The borrowed-alias/page-slice-pool pass then measured
  `64.33 tok/s` at 512 tokens and `58.59 tok/s` at 2048 tokens; final norm
  precompute, the 16-token attention threshold fix, RMSNorm shuffle
  reductions, and fused RMSNorm+RoPE heads then measured `94.30 tok/s` at
  512 tokens and `69.04 tok/s` at 2k tokens. The parallel sliding-window
  descriptor-trim fix then measured `94.71 tok/s` at 512 tokens and
  `79.38 tok/s` at 2k tokens. The remaining
  bottleneck is still the queued GPU primitive graph rather than bulk logits
  readback or Go-side KV page metadata allocation.
- Alloc profiles still show load-time pressure in `copyTensorToDevice` and
  tokenizer setup. The latest generation-time metadata cleanup reduces B/op
  substantially, so the runtime blocker is now kernel time and primitive launch
  shape more than host allocation.
- `rocminfo` resolves the selected GPU UUID to `gfx1100`; `rocm-smi` reports the
  same board's product GFX version as `gfx1101`. A `gfx1101` HSACO built
  successfully but failed `hipModuleLoadData` with HIP error `200`, while the
  `gfx1100` HSACO loads and runs. Fresh post-fusion checks confirmed this is
  not hiding the 100+ tok/s endpoint: `gfx1100 -O3` preserved smoke and measured
  `94.30 tok/s` at 512 tokens, effectively identical to the kept `gfx1100 -O2`
  path; `gfx11-generic -O2` preserved smoke and measured `94.54 tok/s` at
  512 tokens and `69.24 tok/s` at 2000 tokens on the pre-descriptor-trim source,
  only noise-level above that kept `69.04 tok/s` endpoint. Keep using
  `gfx1100 -O2` unless ROCm toolchain
  behavior changes enough to justify another endpoint retest.

## 100+ tok/s Work Order

- [x] Keep the current q4/BF16 smoke tests and benchmarks as regression gates.
- [x] Verify the 2k benchmark is timing generation after model load, not model
  load/setup time.
- [x] Verify the pinned UUID runs on the RX 7800 XT and the onboard GPU remains
  idle during the 2k benchmark.
- [x] Rule out `gfx1101`, `gfx1100 -O3`, `gfx11-generic`, and shared-input q4
  projection as easy HIP/compiler fixes for the pre-endpoint bottleneck.
- [x] Encode appended q4 KV pages on device with `rocm_kv_encode_token`.
- [x] Build appended device KV descriptor tables on device and let shared-KV
  layers borrow source descriptor tables.
- [x] Move the partial-RoPE/global-attention Q/K/V/norm/RoPE/KV-append branch
  onto device buffers for the q4 generate hot path.
- [x] Replace the serial primitive `rocm_rms_norm` and `rocm_rope` hot-path
  behavior with parallel kernels.
- [x] Fuse the local q4 MLP gate/up plus GELU multiply, the per-layer input
  gate plus GELU/multiplier path, and three RMSNorm+residual-add chains.
- [x] Fix/default mapped launch-argument packets without reusing stale mapped
  host memory before GPU consumption.
- [x] Parallelize the common no-trim device KV descriptor append path.
- [x] Use 512-thread attention-head blocks for contexts of at least 512 tokens,
  while keeping shorter contexts on the measured 256-thread path.
- [x] Cache each head query in shared memory for the token-parallel attention
  score phase after the newer 512-thread attention shape.
- [x] Consume q8 attention key dots four int8 values per packed 32-bit word when
  the device-KV page offset is aligned.
- [x] Unroll the aligned q8 key-dot loop to consume four packed q8 words per
  iteration. This is distinct from the rejected generic-helper bypass and moved
  the 2000-token endpoint to `60.12 tok/s`.
- [x] Add a guarded direct q4 value-accumulation branch for the default direct
  `k-q8-v-q4` descriptor layout. This moved the endpoint to `60.79 tok/s` at
  2000 tokens and `60.62 tok/s` at 2048 tokens.
- [x] Reject direct q4 value token-loop unroll-by-two after q4 smoke and
  endpoint benchmarks showed no useful speedup.
- [x] Reject pre-scaling normalized direct q4 attention weights by cached value
  scale because the guarded exact path regressed the 2048-token endpoint to
  `60.54 tok/s`.
- [x] Reject split-score multi-block attention because the extra launch
  preserved smoke but regressed the 2048-token endpoint to `53.76 tok/s`.
- [x] Split q4 device-KV attention value accumulation across paired output
  dimensions when long-context 512-thread blocks have more threads than output
  dimensions.
- [x] Cache q4 value page pointers/scales in dynamic shared memory for
  long-context device-KV attention.
- [x] Replace repeated full per-head device-KV page-table validation in
  `rocm_attention_heads` with header-only validation.
- [x] Normalize token-parallel attention softmax weights once and hoist direct
  one-token page addressing in the score/value loops.
- [x] Reject the post-validation q8 key-dot specialization and 1024-thread
  attention retest because both were slower than the kept path.
- [x] Make shared-device-KV borrowed aliases share source page metadata without
  ownership, and pool hot-path copied KV page slices. This cuts long-run
  allocation pressure but leaves throughput kernel-bound.
- [x] Remove launch-path allocation churn from `go/hip_driver_cgo.go` and launch
  config creation: cache HIP availability, avoid per-launch HSACO `stat`/key
  construction, stop copying launch args into configs, and use a value lease
  instead of a finish closure. This cut 2048-token allocation to `264.8MB/op`
  and `3.06M allocs/op`, but throughput is still only `60.87 tok/s`.
- [x] Fix the opt-in single-packet launch-argument path
  (`GO_ROCM_DISABLE_ASYNC_LAUNCH_ARGS=1`) so it synchronizes before reusing the
  packet. This removes a HIP 700 correctness failure but is not the default
  endpoint path.
- [x] Reject `fp16`/`q8` KV-mode shortcuts, group-64 q4 layout, async H2D as a
  default, `-ffast-math`, and q4 `__restrict__` hints because they either failed
  smoke or did not improve the 2k endpoint.
- [x] Reject a guarded group-64 q4 projection/GELU group-index shift branch. It
  preserved q4 smoke but regressed two 512-token checks to about `62.9 tok/s`
  versus the `65.5 tok/s` kept source.
- [x] Reject the temporary device-buffer arena wrapper because it preserved q4
  smoke but regressed the 2k endpoint and increased Go allocation churn.
- [x] Reject query-head RMSNorm+RoPE fusion. It preserved q4 smoke and reduced
  some allocation pressure at 512 tokens, but repeat 2k endpoint runs regressed
  to `58.78`/`58.73 tok/s`; reverting restored `61.09 tok/s`.
- [x] Keep group-tiled scale/bias loading in the MLX q4 projection-family row
  sum. It preserved q4 smoke and moved the kept 2k endpoint to `61.27 tok/s`.
- [x] Reject the same group-tiled scale/bias loop in the paired q4 gate/up GELU
  multiply kernel because it regressed 512 tokens to `62.76 tok/s`.
- [x] Reject 16 q4 projection rows per 256-thread block after the grouped
  row-sum helper because it regressed 512 tokens to `64.31 tok/s`; reject the
  later 4-row retest because it failed q4 projection smoke, and reject the later
  16-row retest because it regressed the kept 512-token endpoint to
  `87.58 tok/s`.
- [x] Reject a dedicated 256/512-column q4 projection kernel because it
  preserved smoke but regressed the kept 512/2000-token endpoints to
  `65.51`/`60.94 tok/s`.
- [x] Keep the post-attention residual-add/pre-FFN RMSNorm fusion and the
  cross-layer final-hidden/next-input RMSNorm precompute. Together with
  attention max/sum shuffle reductions, this moved the kept 2k endpoint to
  `63.67 tok/s`.
- [x] Keep final-token final RMSNorm precompute plus the 16-token
  `rocm_attention_heads` block/q4 value metadata-cache threshold. This preserved
  q4 smoke and moved the kept endpoint to `90.47 tok/s` at 512 tokens,
  `67.54 tok/s` at 2000 tokens, and `67.46 tok/s` at 2048 tokens.
- [x] Keep RMSNorm shuffle reductions and fused q4 RMSNorm+RoPE heads. This
  preserved q4 smoke and moved the kept endpoint to `94.30 tok/s` at
  512 tokens, `69.04 tok/s` at 2000 tokens, and `68.83 tok/s` at 2048 tokens.
- [x] Keep the parallel sliding-window descriptor-trim fast path. This preserved
  q4 smoke, reduced `rocm_kv_descriptor_append` from `18.48%` to `1.79%` in a
  2k profile, and moved the kept endpoint to `94.71 tok/s` at 512 tokens,
  `79.38 tok/s` at 2000 tokens, and `79.12 tok/s` at 2048 tokens.
- [x] Fix the q4 projection/final-greedy launch geometry mismatch. The previous
  mixed constants produced contaminated fused-greedy timing; the corrected
  source uses normal projection rows-per-block for normal projection kernels and
  greedy rows-per-block for the final projection+greedy kernel.
- [x] Retune the corrected q4 projection geometry. The kept source uses eight
  rows per block for normal q4 projection and 32 rows per block for the fused
  q4 final projection+greedy kernel.
- [x] Remove default per-launch HIP event overhead from async launch-argument
  reuse. The ring now synchronizes on wrap by default, with the old event mode
  behind `GO_ROCM_ENABLE_LAUNCH_ARG_EVENTS=1`.
- [x] Respect `LoadModel` context size in the Gemma4 q4 sliding-window layer
  config. This makes the benchmark's `WithContextLen(128)` real for
  sliding-window layers without capping full-attention layers.
- [x] Reach the 2k performance endpoint on the pinned RX 7800 XT. The corrected
  current path measures `101.9 tok/s` for 2000 tokens and `101.6 tok/s` for
  2048 tokens using `/tmp/go-rocm-kernels-gfx1100.hsaco`.
- [x] Reject an experimental `k-q4-v-q4` device KV cache mode because it changed
  public q4 generation output and failed the accepted-token smoke check.
- [x] Add the Gemma4 q4 prefill planner and wire the current prompt loop through
  ubatch/output-mask semantics. The default ubatch size is 512 and can be
  overridden with `GO_ROCM_GEMMA4_Q4_PREFILL_UBATCH_TOKENS`. This is only
  integration groundwork; it still delegates each prompt token to the existing
  single-token forward path.
- [x] Add the Gemma4 q4 batched prefill embedding primitive. It accepts a token
  span, launches one multi-token embedding lookup, applies the normal
  `sqrt(hidden)` scale across `batch*hidden` values, and returns scaled device
  embeddings for the future batched layer graph. This is still groundwork; it
  does not update KV slots or run batched projection/attention yet.
- [x] Add the Gemma4 q4 batched prefill input-norm primitive. It reuses
  `rocm_rms_norm_heads` with prompt rows mapped to the launch `head_count`, so
  a prompt ubatch can normalize `batch` independent hidden vectors in one grid
  launch. This prepares the first per-layer prefill row operation, but the
  layer still needs batched q4 projection, KV writes, and attention.
- [x] Add the first batched MLX q4 projection primitive for prefill. The new
  `rocm_mlx_q4_projection_batch` kernel maps prompt rows onto `GridY`, projects
  a `[batch, cols]` activation matrix to `[batch, rows]` with one launch, and is
  wrapped by a Gemma4 q4 Q/K/V prefill helper. This replaces one class of
  per-token projection launches, but batched MLP, KV writes, RoPE, and attention
  are still open.
- [x] Add the first batched q4 MLP primitive for prefill. The new
  `rocm_mlx_q4_gelu_tanh_multiply_batch` kernel maps prompt rows onto `GridY`
  for fused gate/up projection plus GELU multiply, and the Gemma4 q4 prefill
  MLP helper follows it with the batched down projection. This covers the MLP
  launch shape, but it is not yet wired into the full prompt path.
- [x] Add batched Q/K RMSNorm+RoPE for Gemma4 q4 prefill. The new
  `rocm_rms_norm_rope_heads_batch` kernel maps prompt rows onto `GridY`,
  normalizes `[batch, heads, head_dim]`, and applies `start_position+row` RoPE
  in one launch per Q or K buffer. The Gemma4 q4 prefill helper now launches
  batched query and key norm/RoPE with the same full-rotary vs partial-rotary
  geometry used by decode.
- [x] Add batched Gemma4 q4 value RMSNorm for prefill. It reuses
  `rocm_rms_norm_heads` with unit weights to normalize `[batch, head_dim]`
  value rows without host readback, matching the decode path's no-scale value
  normalization.
- [x] Add batched device-KV row pages for Gemma4 q4 prefill. The device cache
  path can now encode contiguous `[batch, head_dim]` key/value buffers into
  block-sized pages with `token_count > 1`, build a descriptor table, and return
  a launch descriptor for the prefill attention graph. This removes another
  one-token append assumption from the scaffold, but full masked prefill
  attention is still open.
- [x] Compose the Gemma4 q4 one-layer prefill KV setup. The helper chains
  batched input RMSNorm, Q/K/V q4 projection, Q/K RMSNorm+RoPE, value
  RMSNorm, and batched device-KV descriptor creation for one layer. This proves
  the completed row-batch primitives can be driven together up to KV state, but
  it still stops before masked attention, residuals, MLP, and logits.
- [x] Add a first batched causal prefill attention primitive. The new
  `rocm_attention_heads_batch_causal` kernel maps query heads to `GridX`,
  prompt rows to `GridY`, reuses the existing single-head softmax/value math,
  supports contiguous or descriptor-backed device KV, and lets a Gemma4 q4
  one-layer prefill helper produce `[tokens, query_heads*head_dim]` attention
  output from the batched Q/K/V setup. This is still one-layer scaffolding; it
  does not yet implement the hybrid full/SWA prompt graph across all layers.
- [x] Compose the Gemma4 q4 one-layer prefill body through final hidden output.
  The helper now chains batched causal attention, batched output projection,
  post-attention residual/pre-FFN norm, batched q4 MLP, and post-FFN residual
  output for one layer without host readback of the intermediate matrices. This
  still is not full prefill: it is not wired across all layers, does not produce
  selected logits, and does not yet split full-context vs sliding-window
  attention policy.
- [x] Add the Gemma4 q4 batched prefill per-layer input branch. The new
  `rocm_mlx_q4_gelu_tanh_projection_batch` kernel maps prompt rows onto
  `GridY` for `gelu(input_gate(hidden))*per_layer_input`, then the prefill
  layer body applies the batched per-layer projection plus post-input residual
  path before layer-scalar scaling. This closes the one-layer architecture
  branch that Gemma4 needs, but still leaves multi-layer orchestration,
  selected logits, and full/SWA policy as the actual-load blockers.
- [x] Add selected-row final sampling for the batched prefill scaffold. The
  helper borrows only the requested prompt row, applies final RMSNorm, and runs
  fused q4 LM-head softcap+greedy against that row instead of projecting logits
  for every prompt token. This matches the output-mask requirement for the
  scaffold, but the result still needs all-layer integration before it can
  replace the single-token prompt loop.
- [x] Add a first all-layer batched prefill scaffold. The helper accepts a token
  span, launches batched embedding once, runs each Gemma4 q4 layer config
  through the existing batched KV/body path, accepts per-layer input matrices,
  and can sample selected output rows. This is not yet the real long-prompt
  replacement because it still needs prior-ubatch integration and the full/SWA
  prompt attention policy.
- [x] Add a first prior-ubatch device-KV append path for batched prefill. The
  `WithPrior` forward helper accepts one prior device-KV cache per layer,
  appends the current token span into descriptor-backed device KV, and launches
  batched causal attention with `query_start_token` set to the ubatch start. The
  current cache ownership is still borrow-based and this path deliberately
  handles only untrimmed prior windows, so production full/SWA slot layout and
  ownership transfer remain open.
- [ ] Implement Gemma4 q4 batched prefill. This must accept a token span,
  process prompt tokens in ubatches, update KV slots for every token, request
  logits only for selected output tokens, and avoid feeding large prompts
  through the single-token decode loop.
- [ ] Replace the one-token page KV layout used during prompt load with
  contiguous/slot-based full and SWA KV caches. The target is llama.cpp's shape:
  prepare base/full and sliding-window caches per ubatch, then execute the graph
  against device-side K/V indices and masks.
- [ ] Add batched q4 projection/MLP kernels for prompt matrices. Single-token
  GEMV launch chains are acceptable for decode, but long prompt prefill needs
  GEMM/MMQ-style work over `[hidden, batch]` activations.
- [ ] Add batched masked attention for Gemma4 prefill, preferably a
  flash-attention style kernel with separate full-context and SWA paths.
- [ ] Add a device-resident q4 decode workspace/arena for hidden vectors,
  projection outputs, attention scratch, MLP scratch, logits, token IDs, and KV
  cache updates. The per-token hot path must not allocate/free device buffers.
- [ ] Fuse the per-layer q4 decode path. A layer should accept device pointers
  for hidden state, q/k/v/o weights, norms, MLP weights, KV cache descriptors,
  and scratch; it should return the next hidden state on device.
- [ ] Replace the current naive MLX q4 GEMV with a production q4 projection
  kernel: tiled input reuse, wave/block reductions, packed-weight coalescing,
  and a launch shape tuned for large LM-head rows and small Q/K/V/O rows.
- [ ] Replace the current helper attention kernel with a real long-context decode
  attention kernel that keeps scores/probabilities on device, avoids per-token
  barrier-heavy loops, supports 2k+ tokens without serial fallback, and writes
  the attention concat directly into the next projection input.
- [ ] Replace one-token device-KV pages with a production block/page layout.
  The current direct-token descriptor fast path is fast for the existing layout
  but forces one descriptor and one tiny encoded K/V allocation per retained
  token. A 100+ tok/s driver needs block/page descriptors that attention can
  address directly, with per-token q8/q4 scales preserved without pointer
  chasing a tiny page for every key/value access.
- [ ] Move GELU/SwiGLU/multiply and residual/norm chaining into HIP kernels or a
  fused per-layer kernel. No q4 MLP intermediate should become a Go `[]float32`.
- [ ] Keep attention update and KV cache reads on device while replacing the
  current paged-attention primitive with a production decode attention kernel.
  Long-context decode must avoid per-head or per-position host readback and
  should use packed/paged descriptors when history grows.
- [x] Make the full token step device-resident for the current q4 Generate path:
  embedding, all layers, final
  norm, LM head, and greedy/sample stay on the GPU. Only the selected token ID
  comes back to Go for tokenizer/loop control. The initial device KV cache is
  now bootstrapped from device K/V token buffers instead of host K/V vectors.
- [ ] Add a performance ladder and fail fast on regressions: 1, 8, 64, 512, and
  2000 generated tokens.
- [ ] Once BF16 correctness is stable, use q4 for fast iteration, but do not
  accept q4 speedups that bypass the real Gemma4 q4 packed tensors.

## Performance Gates

Use the real discrete GPU, not device 0:

```sh
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 rocminfo | rg -n "Name:|Uuid:|Marketing Name|gfx"
```

Short q4 speed checks:

```sh
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_BENCHMARKS=1 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
GO_ROCM_BENCH_TOKENS=1 \
go test ./go -run '^$' -bench BenchmarkInferenceGemma4Q4Generate -benchmem -count=1

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_BENCHMARKS=1 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
GO_ROCM_BENCH_TOKENS=8 \
go test ./go -run '^$' -bench BenchmarkInferenceGemma4Q4Generate -benchmem -count=1
```

Endpoint q4 benchmark:

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

Use `GO_ROCM_BENCH_TOKENS=2048` for the stricter 2k-token check used in the
latest fresh-quota pass.

Correctness anchors:

```sh
ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE=1 \
GO_ROCM_GEMMA4_Q4_GENERATE_PROMPT='text:Hi' \
go test ./go -run 'TestNative(DecodeSmokeKernelStatus|ModelPackSmokeGemma4E2B)_Good' -count=1 -v

ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH=/data/lem/models/gemma4/LEM-Gemma4-E2B \
GO_ROCM_KERNEL_HSACO=/tmp/go-rocm-kernels-gfx1100.hsaco \
go test ./go -run TestNativeDecodeSmokeKernelStatus_Good -count=1 -v
```

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

This goal is complete when the performance endpoint and the package/runtime
contract are both true:

- Gemma4-E2B q4 generation on the RX 7800 XT sustains `100+ tok/s` on the
  `GO_ROCM_BENCH_TOKENS=2000` benchmark.
- The token-generation hot path is device-resident. Only final token IDs and
  explicitly requested debug/probe outputs cross back to host during decode.
- No per-token host `[]float32` intermediate is required between embedding,
  decoder layers, final norm, LM head, and sampling.
- The q4 path uses the real packed Gemma4-E2B q4 tensors from
  `/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit`; test-only toy weights do not
  satisfy the endpoint.
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

## Gemma4 q4 100+ Driver Endpoint Addendum

Current endpoint: Gemma4 q4 prompt prefill on the pinned RX 7800 XT must stay
above `100 prompt_tok/s` for actual model loads while moving toward llama.cpp-
class prompt processing. The near-term comparison set is 2k and 4k prompts at a
48k context, not 16k+ ladder runs, until the 2k/4k gap is much closer.

Latest measured state on `ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85` with
`/data/lem/models/gemma4/LEM-Gemma4-E2B-4bit` and the rebuilt gfx1100 HSACO:

- 2k prompt: `5963139198 ns/op`, `343.4 prompt_tok/s`.
- 4k prompt: `12840551074 ns/op`, `319.0 prompt_tok/s`.
- Local llama.cpp reference remains much faster: about `4788 tok/s` at 2k,
  `4520 tok/s` at 4k, and `4072 tok/s` at 8k prompt processing.

Confirmed driver/kernel findings:

- Fixed a real descriptor-backed attention bug: `rocm_attention_device_kv_page`
  was linearly scanning descriptor pages for 16-token prompt KV pages. It now
  indexes by descriptor `block_size` first and only scans as fallback.
- Added 8-token q4 batch tiles for the MLX q4 projection batch, fused
  GELU-tanh gate/up batch, and per-layer-input GELU projection batch kernels.
  This reuses q4 weight loads across prompt tokens and moved the 2k run from
  roughly `272 prompt_tok/s` after the descriptor fix to the mid-`340s`; the
  per-layer-input tile is mixed at 2k but improved the 4k check to
  `319.0 prompt_tok/s`.
- A 16-token tile did not improve the 2k run, so the kept tile is 8.

Remaining blocker:

- The code is now comfortably above the original `100+ tok/s` floor, but it is
  still not "fixed" relative to llama.cpp. The remaining gap is the q4 prompt
  matmul algorithm: current kernels are still row-dot packet primitives with
  limited tiling, not a production tiled GEMM/dequant path with broad reuse of
  input/weights across rows and tokens.
- The 2048-token fast loop is useful for accepting allocation/plumbing cleanup,
  but retained-book quality remains the gate for numerical launch changes. A
  512-thread RMS launch-shape experiment improved 2048 tok/s, then failed the
  retained chapter-10 arc check, so it was rejected. The latest accepted
  launch-plumbing pass is quality-preserving and flat on retained speed:
  `37.67s` wall, `80.20 tok/s` average, turn 10 `69.63 tok/s`,
  `230999296 B/op`, `106651 allocs/op`, `chapter10_arc_anchor_hits=3`.

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
