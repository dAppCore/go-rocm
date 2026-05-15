// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"os"
	"strconv"
	"testing"
	"time"

	"dappco.re/go/inference"
)

func BenchmarkInferenceGemma4Q4Generate(b *testing.B) {
	if os.Getenv("GO_ROCM_RUN_BENCHMARKS") != "1" {
		b.Skip("set GO_ROCM_RUN_BENCHMARKS=1 to run ROCm inference benchmarks")
	}
	modelPath := os.Getenv("GO_ROCM_MODEL_PATH")
	if modelPath == "" {
		b.Skip("set GO_ROCM_MODEL_PATH to a local Gemma4 q4 model pack")
	}
	maxTokens := 1
	if value := os.Getenv("GO_ROCM_BENCH_TOKENS"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed <= 0 {
			b.Fatalf("GO_ROCM_BENCH_TOKENS=%q, want positive integer", value)
		}
		maxTokens = parsed
	}
	prompt := os.Getenv("GO_ROCM_BENCH_PROMPT")
	if prompt == "" {
		prompt = "text:Hi"
	}
	b.Setenv("GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE", "1")

	model, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModel(modelPath, inference.WithContextLen(128))
	if err != nil {
		b.Fatalf("LoadModel(%q): %v", modelPath, err)
	}
	defer model.Close()

	b.ReportAllocs()
	b.ResetTimer()
	totalTokens := 0
	start := time.Now()
	for i := 0; i < b.N; i++ {
		generated := 0
		for range model.Generate(context.Background(), prompt, inference.WithMaxTokens(maxTokens)) {
			generated++
		}
		if err := model.Err(); err != nil {
			b.Fatalf("Generate: %v", err)
		}
		totalTokens += generated
	}
	elapsed := time.Since(start)
	b.StopTimer()
	if elapsed > 0 {
		b.ReportMetric(float64(totalTokens)/elapsed.Seconds(), "tok/s")
	}
	b.ReportMetric(float64(totalTokens), "tokens")
	b.ReportMetric(float64(maxTokens), "max_tokens/op")
}
