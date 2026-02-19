# Phase 4: Performance Design

Approved 19 Feb 2026.

## 1. Benchmark Suite (`rocm_benchmark_test.go`)

Go benchmark tests (`testing.B`) with `b.ReportMetric()`, build-tagged `//go:build rocm`.

Three benchmarks per model:

- **BenchmarkDecode_{Model}** — Generate 128 tokens, report tok/s
- **BenchmarkTTFT_{Model}** — Time-to-first-token latency (ns)
- **BenchmarkConcurrent_{Model}** — N goroutines generating simultaneously, report aggregate tok/s

Models: Gemma3-4B-Q4_K_M, Llama-3.1-8B-Q4_K_M, Qwen-2.5-7B-Q4_K_M. Each loads once per benchmark function (load time excluded via `b.StopTimer`/`b.StartTimer`). VRAM logged after load via `GetVRAMInfo()`.

Run: `go test -tags rocm -bench . -benchtime 3x -timeout 600s`

## 2. Flash Attention (Manual)

No code changes. Build a second llama-server without `-DGGML_HIP_ROCWMMA_FATTN=ON`, run the benchmark suite twice with `ROCM_LLAMA_SERVER_PATH` pointing at each binary, record results in FINDINGS.md.

## 3. Parallel Slots

Add `ParallelSlots int` to go-inference's `LoadConfig` and `WithParallelSlots(n int) LoadOption`. Pass `--parallel N` to llama-server in `startServer` when > 0. Benchmark concurrent requests with 1 slot vs 4 slots.

## 4. Testing

Benchmarks are the tests — they produce the numbers. Existing integration tests remain as correctness checks. All results documented in FINDINGS.md.
