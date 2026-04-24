//go:build rocm

package rocm

import (
	"context"
	"os"
	"sync"
	"testing"
	"time"

	"dappco.re/go/inference"
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
// Stops the benchmark timer during loading so load time is excluded.
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

			var totalTokens int
			b.ResetTimer()
			for range b.N {
				ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
				var count int
				for range m.Generate(ctx, "Explain the theory of relativity in detail.", inference.WithMaxTokens(maxTok)) {
					count++
				}
				cancel()
				totalTokens += count
			}
			b.StopTimer()

			if totalTokens > 0 {
				elapsed := b.Elapsed()
				tokPerSec := float64(totalTokens) / elapsed.Seconds()
				b.ReportMetric(tokPerSec, "tok/s")
			}
		})
	}
}

// BenchmarkTTFT measures time-to-first-token latency.
// Reports the average time from request start to first token received.
func BenchmarkTTFT(b *testing.B) {
	skipBenchIfUnavailable(b)

	for _, model := range benchModels {
		b.Run(model.name, func(b *testing.B) {
			m := loadBenchModel(b, model.path)
			defer m.Close()

			var totalTTFT time.Duration
			b.ResetTimer()
			for range b.N {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				start := time.Now()
				for range m.Generate(ctx, "Hello", inference.WithMaxTokens(1)) {
					totalTTFT += time.Since(start)
				}
				cancel()
			}
			b.StopTimer()

			if b.N > 0 {
				avgTTFT := totalTTFT / time.Duration(b.N)
				b.ReportMetric(float64(avgTTFT.Microseconds()), "µs/first-tok")
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

			var totalTokens int
			b.ResetTimer()
			for range b.N {
				var wg sync.WaitGroup
				var mu sync.Mutex
				iterTokens := 0

				wg.Add(numWorkers)
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
						iterTokens += count
						mu.Unlock()
					}(i)
				}
				wg.Wait()
				totalTokens += iterTokens
			}
			b.StopTimer()

			if totalTokens > 0 {
				elapsed := b.Elapsed()
				b.ReportMetric(float64(totalTokens)/elapsed.Seconds(), "tok/s-aggregate")
				b.ReportMetric(float64(totalTokens)/float64(b.N), "tok/iter")
			}
		})
	}
}
