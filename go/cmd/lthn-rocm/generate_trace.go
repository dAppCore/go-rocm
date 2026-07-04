// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"fmt"
	"io"
	"strconv"
	"time"

	"dappco.re/go/inference"
	rocm "dappco.re/go/rocm"
)

type traceStateResetter interface {
	ResetState() error
}

func runGenerateTrace(ctx context.Context, modelPath, prompt string, maxTokens int, sampleTemperature float32, enableThinking bool, rocmLoadCfg rocm.ROCmLoadConfig, loadOpts []inference.LoadOption, stdout, stderr io.Writer) int {
	model, err := loadROCmCLITextModel(modelPath, rocmLoadCfg, loadOpts)
	if err != nil {
		fmt.Fprintf(stderr, "%s generate: load: %v\n", cliName(), err)
		return 1
	}
	defer model.Close()

	run := func(lane string, temp float32, limit int, report bool) bool {
		start := time.Now()
		count := 0
		for range model.Chat(ctx, []inference.Message{{Role: "user", Content: prompt}},
			rocmCLIGenerateOptions(limit, temp, enableThinking)...,
		) {
			count++
		}
		wall := time.Since(start)
		if err := model.Err(); err != nil {
			fmt.Fprintf(stderr, "%s generate: %s: %v\n", cliName(), lane, err)
			return false
		}
		if !report {
			return true
		}
		metrics := model.Metrics()
		metrics = completeTraceMetrics(metrics, count, wall)
		printGenerateTraceLane(stdout, lane, metrics)
		return true
	}

	if !run("warm", 0, minPositive(maxTokens, 8), false) {
		return 1
	}
	if !resetGenerateTraceState(model, stderr) {
		return 1
	}
	for _, lane := range []struct {
		name string
		temp float32
	}{
		{"greedy (temp=0)", 0},
		{"sampled (temp=" + formatTraceTemperature(sampleTemperature) + ")", sampleTemperature},
	} {
		if !run(lane.name, lane.temp, maxTokens, true) {
			return 1
		}
		if !resetGenerateTraceState(model, stderr) {
			return 1
		}
	}
	fmt.Fprintln(stdout, "\nphase trace: ROCm currently exposes aggregate prefill/decode timing; per-token host/GPU phase buckets are backend-internal work.")
	return 0
}

func formatTraceTemperature(temp float32) string {
	return strconv.FormatFloat(float64(temp), 'g', -1, 32)
}

func resetGenerateTraceState(model inference.TextModel, stderr io.Writer) bool {
	resetter, ok := model.(traceStateResetter)
	if !ok {
		return true
	}
	if err := resetter.ResetState(); err != nil {
		fmt.Fprintf(stderr, "%s generate: reset trace state: %v\n", cliName(), err)
		return false
	}
	return true
}

func completeTraceMetrics(metrics inference.GenerateMetrics, generated int, wall time.Duration) inference.GenerateMetrics {
	if metrics.GeneratedTokens == 0 {
		metrics.GeneratedTokens = generated
	}
	if metrics.TotalDuration <= 0 {
		metrics.TotalDuration = metrics.PrefillDuration + metrics.DecodeDuration
	}
	if metrics.TotalDuration <= 0 {
		metrics.TotalDuration = wall
	}
	if metrics.DecodeDuration <= 0 {
		if metrics.PrefillDuration > 0 && metrics.TotalDuration > metrics.PrefillDuration {
			metrics.DecodeDuration = metrics.TotalDuration - metrics.PrefillDuration
		} else {
			metrics.DecodeDuration = metrics.TotalDuration
		}
	}
	if metrics.PrefillTokensPerSec == 0 && metrics.PromptTokens > 0 && metrics.PrefillDuration > 0 {
		metrics.PrefillTokensPerSec = float64(metrics.PromptTokens) / metrics.PrefillDuration.Seconds()
	}
	if metrics.DecodeTokensPerSec == 0 && metrics.GeneratedTokens > 0 && metrics.DecodeDuration > 0 {
		metrics.DecodeTokensPerSec = float64(metrics.GeneratedTokens) / metrics.DecodeDuration.Seconds()
	}
	return metrics
}

func printGenerateTraceLane(stdout io.Writer, lane string, metrics inference.GenerateMetrics) {
	fmt.Fprintf(stdout, "\n%s\n", lane)
	fmt.Fprintf(stdout, "  tokens   prompt %d · generated %d\n", metrics.PromptTokens, metrics.GeneratedTokens)
	fmt.Fprintf(stdout, "  prefill  %8.3f ms  %8.1f tok/s\n", durationMS(metrics.PrefillDuration), metrics.PrefillTokensPerSec)
	fmt.Fprintf(stdout, "  decode   %8.3f ms  %8.1f tok/s\n", durationMS(metrics.DecodeDuration), metrics.DecodeTokensPerSec)
	fmt.Fprintf(stdout, "  total    %8.3f ms\n", durationMS(metrics.TotalDuration))
	if metrics.ActiveMemoryBytes != 0 || metrics.PeakMemoryBytes != 0 {
		fmt.Fprintf(stdout, "  memory   active %.1f MiB · peak %.1f MiB\n", mib(metrics.ActiveMemoryBytes), mib(metrics.PeakMemoryBytes))
	}
}

func durationMS(d time.Duration) float64 {
	return float64(d.Microseconds()) / 1000.0
}

func mib(bytes uint64) float64 {
	return float64(bytes) / float64(1<<20)
}

func minPositive(a, b int) int {
	if a <= 0 {
		return b
	}
	if b <= 0 || a < b {
		return a
	}
	return b
}
