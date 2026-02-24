# Phase 4: Performance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add parallel inference slots, a Go benchmark suite measuring decode speed / TTFT / concurrent throughput, and document flash attention results.

**Architecture:** Add `ParallelSlots` to go-inference's `LoadConfig`, wire `--parallel N` through go-rocm's `startServer`, then benchmark three models with `testing.B` and `b.ReportMetric`.

**Tech Stack:** Go testing.B, go-inference options, llama-server `--parallel` flag

---

### Task 1: Add ParallelSlots to go-inference

Add `ParallelSlots int` field to `LoadConfig` and a `WithParallelSlots(n int) LoadOption` constructor in the go-inference package.

**Files:**
- Modify: `/home/claude/Code/core/go-inference/options.go`

**Step 1: Add field and option to go-inference**

In `/home/claude/Code/core/go-inference/options.go`, add to `LoadConfig` struct (after `GPULayers`):

```go
ParallelSlots int // Number of concurrent inference slots (0 = server default)
```

Add option function after `WithGPULayers`:

```go
// WithParallelSlots sets the number of concurrent inference slots.
// Higher values allow parallel Generate/Chat calls but increase VRAM usage.
// 0 or unset uses the server default (typically 1).
func WithParallelSlots(n int) LoadOption {
	return func(c *LoadConfig) { c.ParallelSlots = n }
}
```

**Step 2: Verify go-inference builds**

Run: `cd /home/claude/Code/core/go-inference && go build ./...`
Expected: Clean

**Step 3: Commit go-inference**

```bash
cd /home/claude/Code/core/go-inference
git add options.go
git commit -m "feat: add ParallelSlots to LoadConfig for concurrent inference

Co-Authored-By: Virgil <virgil@lethean.io>"
```

---

### Task 2: Wire parallel slots through go-rocm

Pass `--parallel N` to llama-server when `ParallelSlots > 0`. Update `startServer` and `LoadModel`.

**Files:**
- Modify: `server.go` (add parallelSlots param to startServer)
- Modify: `backend.go` (pass cfg.ParallelSlots)

**Step 1: Add parallelSlots to startServer**

In `server.go`, change the `startServer` signature from:

```go
func startServer(binary, modelPath string, gpuLayers, ctxSize int) (*server, error) {
```

to:

```go
func startServer(binary, modelPath string, gpuLayers, ctxSize, parallelSlots int) (*server, error) {
```

In the args-building section (after the `--ctx-size` block at line ~106), add:

```go
if parallelSlots > 0 {
	args = append(args, "--parallel", strconv.Itoa(parallelSlots))
}
```

**Step 2: Update LoadModel in backend.go**

Change the `startServer` call from:

```go
srv, err := startServer(binary, path, cfg.GPULayers, ctxLen)
```

to:

```go
srv, err := startServer(binary, path, cfg.GPULayers, ctxLen, cfg.ParallelSlots)
```

**Step 3: Update TestStartServer_RetriesOnProcessExit in server_test.go**

The test calls `startServer` directly. Update from:

```go
_, err := startServer("/bin/false", "/nonexistent/model.gguf", 999, 0)
```

to:

```go
_, err := startServer("/bin/false", "/nonexistent/model.gguf", 999, 0, 0)
```

**Step 4: Run tests**

Run: `go test ./...`
Expected: All tests PASS

Run: `go vet ./...`
Expected: Clean

**Step 5: Commit**

```bash
git add server.go backend.go server_test.go
git commit -m "feat: pass --parallel N to llama-server for concurrent inference slots

Co-Authored-By: Virgil <virgil@lethean.io>"
```

---

### Task 3: Benchmark Suite

Go benchmark tests measuring decode speed, time-to-first-token, and concurrent throughput across three models.

**Files:**
- Create: `rocm_benchmark_test.go`

**Step 1: Write the benchmark file**

Create `rocm_benchmark_test.go`:

```go
//go:build rocm

package rocm

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"forge.lthn.ai/core/go-inference"
)

// benchModels lists the models to benchmark.
// Each loads ~3-9 GB of VRAM, so they run sequentially (one at a time).
var benchModels = []struct {
	name string
	path string
}{
	{"Gemma3-4B-Q4_K_M", "/data/lem/gguf/LEK-Gemma3-4B-Q4_K_M.gguf"},
	{"Llama3.1-8B-Q4_K_M", "/data/lem/gguf/LEK-Llama-3.1-8B-Q4_K_M.gguf"},
	{"Qwen2.5-7B-Q4_K_M", "/data/lem/gguf/LEK-Qwen-2.5-7B-Q4_K_M.gguf"},
}

func skipBenchIfUnavailable(b *testing.B) {
	b.Helper()
	if _, err := os.Stat("/dev/kfd"); err != nil {
		b.Skip("no ROCm hardware")
	}
	if _, err := findLlamaServer(); err != nil {
		b.Skip("llama-server not found")
	}
}

// loadBenchModel loads a model for benchmarking. Caller must defer m.Close().
// Stops the benchmark timer during loading so load time isn't measured.
func loadBenchModel(b *testing.B, path string, opts ...inference.LoadOption) inference.TextModel {
	b.Helper()
	if _, err := os.Stat(path); err != nil {
		b.Skipf("model not available: %s", path)
	}

	b.StopTimer()
	backend := &rocmBackend{}
	defaults := []inference.LoadOption{inference.WithContextLen(2048)}
	m, err := backend.LoadModel(path, append(defaults, opts...)...)
	if err != nil {
		b.Fatalf("load model: %v", err)
	}

	if vram, err := GetVRAMInfo(); err == nil {
		b.Logf("VRAM after load: %d MiB used / %d MiB total",
			vram.Used/(1024*1024), vram.Total/(1024*1024))
	}

	b.StartTimer()
	return m
}

// BenchmarkDecode measures token generation speed (tok/s).
// Generates 128 tokens per iteration, reports tokens/second.
func BenchmarkDecode(b *testing.B) {
	skipBenchIfUnavailable(b)

	for _, model := range benchModels {
		b.Run(model.name, func(b *testing.B) {
			m := loadBenchModel(b, model.path)
			defer m.Close()

			const maxTok = 128

			b.ResetTimer()
			for range b.N {
				ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
				var count int
				for range m.Generate(ctx, "Explain the theory of relativity in detail.", inference.WithMaxTokens(maxTok)) {
					count++
				}
				cancel()

				if count > 0 {
					tokPerSec := float64(count) / (b.Elapsed().Seconds() / float64(b.N))
					b.ReportMetric(tokPerSec, "tok/s")
				}
			}
		})
	}
}

// BenchmarkTTFT measures time-to-first-token latency.
// Reports the time from request start to first token received.
func BenchmarkTTFT(b *testing.B) {
	skipBenchIfUnavailable(b)

	for _, model := range benchModels {
		b.Run(model.name, func(b *testing.B) {
			m := loadBenchModel(b, model.path)
			defer m.Close()

			b.ResetTimer()
			for range b.N {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				start := time.Now()
				var ttft time.Duration
				for range m.Generate(ctx, "Hello", inference.WithMaxTokens(1)) {
					if ttft == 0 {
						ttft = time.Since(start)
					}
				}
				cancel()

				if ttft > 0 {
					b.ReportMetric(float64(ttft.Microseconds()), "µs/first-tok")
				}
			}
		})
	}
}

// BenchmarkConcurrent measures throughput with multiple goroutines.
// Uses 4 parallel slots and 4 goroutines generating simultaneously.
func BenchmarkConcurrent(b *testing.B) {
	skipBenchIfUnavailable(b)

	for _, model := range benchModels {
		b.Run(model.name, func(b *testing.B) {
			m := loadBenchModel(b, model.path, inference.WithParallelSlots(4))
			defer m.Close()

			const numWorkers = 4
			const maxTok = 32

			prompts := []string{
				"The capital of France is",
				"The capital of Germany is",
				"The capital of Italy is",
				"The capital of Spain is",
			}

			b.ResetTimer()
			for range b.N {
				var wg sync.WaitGroup
				var mu sync.Mutex
				totalTokens := 0

				wg.Add(numWorkers)
				start := time.Now()
				for i := range numWorkers {
					go func(idx int) {
						defer wg.Done()
						ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
						defer cancel()

						count := 0
						for range m.Generate(ctx, prompts[idx], inference.WithMaxTokens(maxTok)) {
							count++
						}

						mu.Lock()
						totalTokens += count
						mu.Unlock()
					}(i)
				}
				wg.Wait()
				elapsed := time.Since(start)

				if totalTokens > 0 {
					b.ReportMetric(float64(totalTokens)/elapsed.Seconds(), "tok/s-aggregate")
					b.ReportMetric(float64(totalTokens), "total-tok")
				}
			}
		})
	}
}
```

**Step 2: Run benchmarks**

Run: `go test -tags rocm -bench BenchmarkDecode/Gemma3 -benchtime 1x -timeout 120s -v`
Expected: PASS with tok/s metric reported

Run full suite (takes several minutes — each model loads separately):
`go test -tags rocm -bench . -benchtime 3x -timeout 600s -v`
Expected: All benchmarks PASS

**Step 3: Verify unit tests still pass**

Run: `go test ./...`
Expected: All PASS (benchmark file is `//go:build rocm`, won't run without tag)

**Step 4: Commit**

```bash
git add rocm_benchmark_test.go
git commit -m "feat: benchmark suite for decode speed, TTFT, and concurrent throughput

Co-Authored-By: Virgil <virgil@lethean.io>"
```

---

### Task 4: Flash Attention Comparison (Manual)

Build a second llama-server without flash attention and compare performance. No code changes — just benchmarks and documentation.

**Step 1: Build llama-server without flash attention**

```bash
cd /home/claude/llama.cpp
cmake -B build-nofa \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1100 \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-nofa --parallel $(nproc) -t llama-server
```

Note: This is the same as the original build but WITHOUT `-DGGML_HIP_ROCWMMA_FATTN=ON`.

**Step 2: Run benchmarks with flash attention (current binary)**

```bash
go test -tags rocm -bench BenchmarkDecode -benchtime 3x -timeout 600s -v 2>&1 | tee /tmp/bench-fa.txt
```

**Step 3: Run benchmarks without flash attention**

```bash
ROCM_LLAMA_SERVER_PATH=/home/claude/llama.cpp/build-nofa/bin/llama-server \
go test -tags rocm -bench BenchmarkDecode -benchtime 3x -timeout 600s -v 2>&1 | tee /tmp/bench-nofa.txt
```

**Step 4: Record results in FINDINGS.md**

Add a section to FINDINGS.md documenting the comparison.

---

### Task 5: Update TODO.md and FINDINGS.md

**Files:**
- Modify: `TODO.md` (mark Phase 4 items done)
- Modify: `FINDINGS.md` (add benchmark results, flash attention comparison)

**Step 1: Update TODO.md**

Mark all Phase 4 items `[x]` with commit references and dates.

**Step 2: Update FINDINGS.md**

Add a `## 2026-02-19: Phase 4 Performance (Charon)` section with:
- Benchmark results table (tok/s, TTFT, VRAM for each model)
- Flash attention comparison
- Concurrent throughput numbers
- Notes on parallel slots and VRAM impact

**Step 3: Commit**

```bash
git add TODO.md FINDINGS.md
git commit -m "docs: Phase 4 complete — benchmarks, flash attention, parallel slots

Co-Authored-By: Virgil <virgil@lethean.io>"
```

---

## Summary

| Task | What | Files | Test Type |
|------|------|-------|-----------|
| 1 | ParallelSlots in go-inference | go-inference/options.go | Build check |
| 2 | Wire --parallel to llama-server | server.go, backend.go, server_test.go | Unit |
| 3 | Benchmark suite | rocm_benchmark_test.go | Benchmark (GPU) |
| 4 | Flash attention comparison | (manual) | Benchmark (GPU) |
| 5 | Documentation | TODO.md, FINDINGS.md | N/A |
