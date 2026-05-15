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

Result from 2026-05-13:

```text
BenchmarkInferenceGemma4Q4Generate-32  1  4254553228 ns/op  1.000 max_tokens/op  0.2350 tok/s  1.000 tokens  349845704 B/op  83394 allocs/op
```

## Profile Findings

Memory profile, `alloc_space`:

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

## Optimization Candidates

1. Cache loaded HSACO modules and function handles per driver/process.
   The benchmark currently spends a large share of CPU samples under cgo module
   load/unload and launch plumbing. Reusing module handles should remove repeat
   `hipModuleLoadData` cost from per-token inference.

2. Reduce model-load allocation in `copyTensorToDevice`.
   The load path accounts for roughly 1.1 GB of allocation space in this run.
   Use a reusable chunk buffer or mmap-style reader path for tensor copies so
   load-time allocations do not dominate single-shot inference benchmarks.

3. Cache tokenizer decoding artifacts per model path.
   `tokenizer.json` loading and merge-rank parsing allocate hundreds of MB.
   A compact parser for only the fields needed by the q4 path, plus process
   cache keyed by tokenizer path and mtime, would reduce startup overhead.

4. Keep q4 intermediate buffers and KV state device-resident across Generate
   calls.
   The experimental q4 path still has host-side control and readback work.
   Reusing descriptor tables, KV pages, and output buffers across decode steps
   should reduce allocation count and cgo launch overhead.

5. Add a longer-token benchmark once module/tokenizer caching lands.
   The current one-token benchmark is intentionally a startup-heavy worst case.
   After the startup costs are amortized, run `GO_ROCM_BENCH_TOKENS=8` and
   compare steady-state decode tok/s against this baseline.
