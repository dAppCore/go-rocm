// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"dappco.re/go/inference"
)

const inferenceBenchmarkKernelRouteMetricsEnv = "GO_ROCM_BENCH_KERNEL_ROUTE_METRICS"

type inferenceBenchmarkHIPKernelStats struct {
	Launches uint64
	Blocks   uint64
}

type inferenceBenchmarkHIPKernelCountingDriver struct {
	nativeHIPDriver
	mu     sync.Mutex
	kernel map[string]inferenceBenchmarkHIPKernelStats
	total  inferenceBenchmarkHIPKernelStats
}

func newInferenceBenchmarkHIPKernelCountingDriver(driver nativeHIPDriver) *inferenceBenchmarkHIPKernelCountingDriver {
	return &inferenceBenchmarkHIPKernelCountingDriver{
		nativeHIPDriver: driver,
		kernel:          make(map[string]inferenceBenchmarkHIPKernelStats),
	}
}

func (driver *inferenceBenchmarkHIPKernelCountingDriver) CopyHostToDeviceAsync(pointer nativeDevicePointer, data []byte) error {
	if async, ok := driver.nativeHIPDriver.(nativeHIPAsyncHostToDevice); ok {
		return async.CopyHostToDeviceAsync(pointer, data)
	}
	return driver.nativeHIPDriver.CopyHostToDevice(pointer, data)
}

func (driver *inferenceBenchmarkHIPKernelCountingDriver) MemsetAsync(pointer nativeDevicePointer, value byte, size uint64) error {
	if memset, ok := driver.nativeHIPDriver.(nativeHIPDeviceMemset); ok {
		return memset.MemsetAsync(pointer, value, size)
	}
	return hipMemsetDevice(driver.nativeHIPDriver, pointer, value, size)
}

func (driver *inferenceBenchmarkHIPKernelCountingDriver) LaunchKernel(config hipKernelLaunchConfig) error {
	if err := hipLaunchKernel(driver.nativeHIPDriver, config); err != nil {
		return err
	}
	blocks := uint64(config.GridX)
	if config.GridY > 0 {
		blocks *= uint64(config.GridY)
	}
	if config.GridZ > 0 {
		blocks *= uint64(config.GridZ)
	}
	driver.mu.Lock()
	stats := driver.kernel[config.Name]
	stats.Launches++
	stats.Blocks += blocks
	driver.kernel[config.Name] = stats
	driver.total.Launches++
	driver.total.Blocks += blocks
	driver.mu.Unlock()
	return nil
}

func (driver *inferenceBenchmarkHIPKernelCountingDriver) ResetKernelStats() {
	driver.mu.Lock()
	clear(driver.kernel)
	driver.total = inferenceBenchmarkHIPKernelStats{}
	driver.mu.Unlock()
}

func (driver *inferenceBenchmarkHIPKernelCountingDriver) KernelStats(name string) inferenceBenchmarkHIPKernelStats {
	driver.mu.Lock()
	defer driver.mu.Unlock()
	return driver.kernel[name]
}

func (driver *inferenceBenchmarkHIPKernelCountingDriver) KernelStatsSnapshot() map[string]inferenceBenchmarkHIPKernelStats {
	driver.mu.Lock()
	defer driver.mu.Unlock()
	snapshot := make(map[string]inferenceBenchmarkHIPKernelStats, len(driver.kernel))
	for name, stats := range driver.kernel {
		snapshot[name] = stats
	}
	return snapshot
}

func (driver *inferenceBenchmarkHIPKernelCountingDriver) TotalKernelStats() inferenceBenchmarkHIPKernelStats {
	driver.mu.Lock()
	defer driver.mu.Unlock()
	return driver.total
}

func inferenceBenchmarkReportHIPKernelRouteMetrics(b *testing.B, driver *inferenceBenchmarkHIPKernelCountingDriver) {
	b.Helper()
	if driver == nil || b.N <= 0 {
		return
	}
	report := func(name, label string) {
		stats := driver.KernelStats(name)
		b.ReportMetric(float64(stats.Launches)/float64(b.N), label+"_launches/op")
		b.ReportMetric(float64(stats.Blocks)/float64(b.N), label+"_blocks/op")
	}
	total := driver.TotalKernelStats()
	b.ReportMetric(float64(total.Launches)/float64(b.N), "kernel_total_launches/op")
	b.ReportMetric(float64(total.Blocks)/float64(b.N), "kernel_total_blocks/op")
	report(hipKernelNameAttentionHeadsBatchCausal, "kernel_attention_batch_causal")
	report(hipKernelNameAttentionHeadsBatchChunkedStage1, "kernel_attention_batch_chunked_stage1")
	report(hipKernelNameAttentionHeadsBatchChunkedStage2, "kernel_attention_batch_chunked_stage2")
	report(hipKernelNameAttentionHeadsChunkedStage1, "kernel_attention_decode_chunked_stage1")
	report(hipKernelNameAttentionHeadsChunkedStage2, "kernel_attention_decode_chunked_stage2")
	inferenceBenchmarkReportTopHIPKernels(b, driver, 12)
}

func inferenceBenchmarkReportTopHIPKernels(b *testing.B, driver *inferenceBenchmarkHIPKernelCountingDriver, limit int) {
	b.Helper()
	if driver == nil || b.N <= 0 || limit <= 0 {
		return
	}
	type kernelEntry struct {
		name  string
		stats inferenceBenchmarkHIPKernelStats
	}
	snapshot := driver.KernelStatsSnapshot()
	entries := make([]kernelEntry, 0, len(snapshot))
	for name, stats := range snapshot {
		if stats.Launches == 0 && stats.Blocks == 0 {
			continue
		}
		entries = append(entries, kernelEntry{name: name, stats: stats})
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].stats.Launches != entries[j].stats.Launches {
			return entries[i].stats.Launches > entries[j].stats.Launches
		}
		if entries[i].stats.Blocks != entries[j].stats.Blocks {
			return entries[i].stats.Blocks > entries[j].stats.Blocks
		}
		return entries[i].name < entries[j].name
	})
	if len(entries) > limit {
		entries = entries[:limit]
	}
	for _, entry := range entries {
		label := "kernel_" + inferenceBenchmarkSanitizeMetricName(entry.name)
		b.ReportMetric(float64(entry.stats.Launches)/float64(b.N), label+"_launches/op")
		b.ReportMetric(float64(entry.stats.Blocks)/float64(b.N), label+"_blocks/op")
	}
}

func inferenceBenchmarkSanitizeMetricName(name string) string {
	if name == "" {
		return "unnamed"
	}
	var builder strings.Builder
	builder.Grow(len(name))
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z':
			builder.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			builder.WriteRune(r)
		case r >= '0' && r <= '9':
			builder.WriteRune(r)
		default:
			builder.WriteByte('_')
		}
	}
	return builder.String()
}

func inferenceBenchmarkNativeRuntimeAndKernelCounter() (nativeRuntime, *inferenceBenchmarkHIPKernelCountingDriver) {
	if os.Getenv(inferenceBenchmarkKernelRouteMetricsEnv) != "1" {
		return newSystemNativeRuntime(), nil
	}
	counter := newInferenceBenchmarkHIPKernelCountingDriver(newSystemHIPDriver())
	return newHIPRuntime(counter), counter
}

type inferenceBenchmarkHIPKernelCountingStubDriver struct{}

func (inferenceBenchmarkHIPKernelCountingStubDriver) Available() bool { return true }

func (inferenceBenchmarkHIPKernelCountingStubDriver) DeviceInfo() nativeDeviceInfo {
	return nativeDeviceInfo{}
}

func (inferenceBenchmarkHIPKernelCountingStubDriver) Malloc(uint64) (nativeDevicePointer, error) {
	return 1, nil
}

func (inferenceBenchmarkHIPKernelCountingStubDriver) Free(nativeDevicePointer) error {
	return nil
}

func (inferenceBenchmarkHIPKernelCountingStubDriver) CopyHostToDevice(nativeDevicePointer, []byte) error {
	return nil
}

func (inferenceBenchmarkHIPKernelCountingStubDriver) CopyDeviceToHost(nativeDevicePointer, []byte) error {
	return nil
}

func (inferenceBenchmarkHIPKernelCountingStubDriver) LaunchKernel(hipKernelLaunchConfig) error {
	return nil
}

func TestInferenceBenchmarkHIPKernelCountingDriver_Good(t *testing.T) {
	driver := newInferenceBenchmarkHIPKernelCountingDriver(inferenceBenchmarkHIPKernelCountingStubDriver{})
	err := driver.LaunchKernel(hipKernelLaunchConfig{
		Name:   hipKernelNameAttentionHeadsBatchChunkedStage1,
		Args:   []byte{1},
		GridX:  2,
		GridY:  3,
		GridZ:  4,
		BlockX: 1,
		BlockY: 1,
		BlockZ: 1,
	})
	if err != nil {
		t.Fatalf("LaunchKernel: %v", err)
	}
	stats := driver.KernelStats(hipKernelNameAttentionHeadsBatchChunkedStage1)
	if stats.Launches != 1 || stats.Blocks != 24 {
		t.Fatalf("kernel stats = %+v, want 1 launch and 24 blocks", stats)
	}
	total := driver.TotalKernelStats()
	if total != stats {
		t.Fatalf("total stats = %+v, want %+v", total, stats)
	}
	snapshot := driver.KernelStatsSnapshot()
	if got := snapshot[hipKernelNameAttentionHeadsBatchChunkedStage1]; got != stats {
		t.Fatalf("snapshot stats = %+v, want %+v", got, stats)
	}
	snapshot[hipKernelNameAttentionHeadsBatchChunkedStage1] = inferenceBenchmarkHIPKernelStats{}
	if got := driver.KernelStats(hipKernelNameAttentionHeadsBatchChunkedStage1); got != stats {
		t.Fatalf("mutated snapshot changed driver stats = %+v, want %+v", got, stats)
	}
	if got := inferenceBenchmarkSanitizeMetricName("rocm/foo-bar"); got != "rocm_foo_bar" {
		t.Fatalf("sanitize metric name = %q, want rocm_foo_bar", got)
	}
	driver.ResetKernelStats()
	if got := driver.TotalKernelStats(); got != (inferenceBenchmarkHIPKernelStats{}) {
		t.Fatalf("reset total stats = %+v, want zero", got)
	}
}

func BenchmarkInferenceGemma4Q4Generate(b *testing.B) {
	benchmarkInferenceGemma4Q4Generate(b)
}

func BenchmarkInferenceGemma4Q4Generate_Ladder(b *testing.B) {
	if os.Getenv("GO_ROCM_RUN_BENCHMARKS") != "1" {
		b.Skip("set GO_ROCM_RUN_BENCHMARKS=1 to run ROCm inference benchmarks")
	}
	if os.Getenv("GO_ROCM_RUN_LADDER_BENCHMARKS") != "1" {
		b.Skip("set GO_ROCM_RUN_LADDER_BENCHMARKS=1 to run the q4 generation performance ladder")
	}
	modelPath := os.Getenv("GO_ROCM_MODEL_PATH")
	if modelPath == "" {
		b.Skip("set GO_ROCM_MODEL_PATH to a local Gemma4 q4 model pack")
	}
	contextLen, err := inferenceBenchmarkPositiveEnv("GO_ROCM_BENCH_CONTEXT_LEN", 128)
	if err != nil {
		b.Fatal(err)
	}
	benchPrompt, err := inferenceBenchmarkPromptFromEnv()
	if err != nil {
		b.Fatal(err)
	}
	prefillUBatchTokens, err := hipGemma4Q4PrefillUBatchTokens()
	if err != nil {
		b.Fatal(err)
	}
	ladderTokens, err := inferenceBenchmarkLadderTokensEnv()
	if err != nil {
		b.Fatal(err)
	}
	b.Setenv("GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE", "1")

	nativeRuntime, kernelCounter := inferenceBenchmarkNativeRuntimeAndKernelCounter()
	model, err := newROCmBackendWithRuntime(nativeRuntime).LoadModel(modelPath, inference.WithContextLen(contextLen))
	if err != nil {
		b.Fatalf("LoadModel(%q): %v", modelPath, err)
	}
	defer inferenceBenchmarkCloseModel(b, model)

	for _, maxTokens := range ladderTokens {
		maxTokens := maxTokens
		b.Run(fmt.Sprintf("tokens_%d", maxTokens), func(b *testing.B) {
			if kernelCounter != nil {
				kernelCounter.ResetKernelStats()
			}
			inferenceBenchmarkRunGemma4Q4GenerateLoaded(b, model, benchPrompt, maxTokens, contextLen, prefillUBatchTokens, "")
			inferenceBenchmarkReportHIPKernelRouteMetrics(b, kernelCounter)
		})
	}
}

func BenchmarkInferenceGemma4Q4PromptPrefillUBatchLadder(b *testing.B) {
	if os.Getenv("GO_ROCM_RUN_BENCHMARKS") != "1" {
		b.Skip("set GO_ROCM_RUN_BENCHMARKS=1 to run ROCm inference benchmarks")
	}
	if os.Getenv("GO_ROCM_RUN_PREFILL_UBATCH_LADDER") != "1" {
		b.Skip("set GO_ROCM_RUN_PREFILL_UBATCH_LADDER=1 to run the q4 prompt prefill ubatch ladder")
	}
	modelPath := os.Getenv("GO_ROCM_MODEL_PATH")
	if modelPath == "" {
		b.Skip("set GO_ROCM_MODEL_PATH to a local Gemma4 q4 model pack")
	}
	if os.Getenv("GO_ROCM_BENCH_PROMPT") == "" &&
		os.Getenv("GO_ROCM_BENCH_PROMPT_FILE") == "" &&
		os.Getenv("GO_ROCM_BENCH_PROMPT_TOKEN_COUNT") == "" {
		b.Setenv("GO_ROCM_BENCH_PROMPT_TOKEN_COUNT", "8192")
	}
	contextLen, err := inferenceBenchmarkPositiveEnv("GO_ROCM_BENCH_CONTEXT_LEN", 48000)
	if err != nil {
		b.Fatal(err)
	}
	maxTokens, err := inferenceBenchmarkPositiveEnv("GO_ROCM_BENCH_TOKENS", 1)
	if err != nil {
		b.Fatal(err)
	}
	benchPrompt, err := inferenceBenchmarkPromptFromEnv()
	if err != nil {
		b.Fatal(err)
	}
	ubatchSizes, err := inferenceBenchmarkPrefillUBatchLadderEnv()
	if err != nil {
		b.Fatal(err)
	}
	b.Setenv("GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE", "1")

	nativeRuntime, kernelCounter := inferenceBenchmarkNativeRuntimeAndKernelCounter()
	model, err := newROCmBackendWithRuntime(nativeRuntime).LoadModel(modelPath, inference.WithContextLen(contextLen))
	if err != nil {
		b.Fatalf("LoadModel(%q): %v", modelPath, err)
	}
	defer inferenceBenchmarkCloseModel(b, model)

	for _, ubatchTokens := range ubatchSizes {
		ubatchTokens := ubatchTokens
		b.Run(fmt.Sprintf("ubatch_%d", ubatchTokens), func(b *testing.B) {
			b.Setenv(hipGemma4Q4PrefillUBatchEnv, strconv.Itoa(ubatchTokens))
			if kernelCounter != nil {
				kernelCounter.ResetKernelStats()
			}
			inferenceBenchmarkRunGemma4Q4GenerateLoaded(b, model, benchPrompt, maxTokens, contextLen, ubatchTokens, "")
			inferenceBenchmarkReportHIPKernelRouteMetrics(b, kernelCounter)
		})
	}
}

func BenchmarkInferenceGemma4Q4Generate_OpencodeSessionStart29K(b *testing.B) {
	if os.Getenv("GO_ROCM_RUN_29K_BENCHMARKS") != "1" {
		b.Skip("set GO_ROCM_RUN_29K_BENCHMARKS=1 to run the 29k opencode session-start benchmark")
	}
	b.Setenv("GO_ROCM_RUN_BENCHMARKS", "1")
	b.Setenv("GO_ROCM_BENCH_CONTEXT_LEN", "48000")
	b.Setenv("GO_ROCM_BENCH_PROMPT_TOKEN_COUNT", "29000")
	b.Setenv("GO_ROCM_BENCH_TOKENS", "1")
	benchmarkInferenceGemma4Q4Generate(b)
}

func BenchmarkInferenceGemma4Q4Book10Turn_ReplayBaseline(b *testing.B) {
	if os.Getenv("GO_ROCM_RUN_BOOK_BENCHMARKS") != "1" {
		b.Skip("set GO_ROCM_RUN_BOOK_BENCHMARKS=1 to run the 10-turn book workload benchmark")
	}
	if os.Getenv("GO_ROCM_RUN_UNSAFE_REPLAY_BOOK_BENCHMARKS") != "1" {
		b.Skip("set GO_ROCM_RUN_UNSAFE_REPLAY_BOOK_BENCHMARKS=1 to run the replay book baseline; prefer retained-state book benchmarks on desktop sessions")
	}
	contextLen, err := inferenceBenchmarkPositiveEnv("GO_ROCM_BOOK_CONTEXT_LEN", 48000)
	if err != nil {
		b.Fatal(err)
	}
	turns, err := inferenceBenchmarkPositiveEnv("GO_ROCM_BOOK_TURNS", 10)
	if err != nil {
		b.Fatal(err)
	}
	if turns > 10 {
		b.Fatalf("GO_ROCM_BOOK_TURNS=%d, want at most 10", turns)
	}
	maxTokens, err := inferenceBenchmarkBookChapterTokensEnv(contextLen, turns)
	if err != nil {
		b.Fatal(err)
	}
	generate, err := inferenceBenchmarkBookGenerateConfig(maxTokens)
	if err != nil {
		b.Fatal(err)
	}
	turnTimeout, err := inferenceBenchmarkDurationSecondsEnv("GO_ROCM_BOOK_TURN_TIMEOUT_SECONDS", 60*time.Second)
	if err != nil {
		b.Fatal(err)
	}
	workload := inferenceBenchmarkBookWorkload()
	model, _, _ := inferenceBenchmarkLoadGemma4Q4Model(b, contextLen, 1)
	defer inferenceBenchmarkCloseModel(b, model)
	b.Setenv("GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE", "1")

	b.ReportAllocs()
	b.ResetTimer()
	var last inferenceBenchmarkBookRun
	for i := 0; i < b.N; i++ {
		run, err := inferenceBenchmarkRunBookReplay(context.Background(), model, workload, generate, turns, turnTimeout)
		if err != nil {
			b.Fatalf("book replay workload: %v", err)
		}
		last = run
	}
	b.StopTimer()
	inferenceBenchmarkReportBookRun(b, last, contextLen, maxTokens, turnTimeout, "replay")
	if hipGemma4Q4HostSamplingRequested(generate) {
		b.ReportMetric(1, "book_host_sampling")
	} else {
		b.ReportMetric(0, "book_host_sampling")
	}
	inferenceBenchmarkRequireBookThresholds(b, last)
	if os.Getenv("GO_ROCM_BOOK_REQUIRE_ARC") == "1" && last.Turns >= 10 && last.ArcAnchorHits < 3 {
		b.Fatalf("chapter 10 anchor hits = %d, want lighthouse/light/ocean arc retained", last.ArcAnchorHits)
	}
}

func BenchmarkInferenceGemma4Q4Book10Turn_RetainedState(b *testing.B) {
	if os.Getenv("GO_ROCM_RUN_BOOK_BENCHMARKS") != "1" {
		b.Skip("set GO_ROCM_RUN_BOOK_BENCHMARKS=1 to run the 10-turn book workload benchmark")
	}
	if os.Getenv("GO_ROCM_RUN_RETAINED_BOOK_BENCHMARKS") != "1" {
		b.Skip("set GO_ROCM_RUN_RETAINED_BOOK_BENCHMARKS=1 to run the retained-state 10-turn book benchmark")
	}
	contextLen, err := inferenceBenchmarkPositiveEnv("GO_ROCM_BOOK_CONTEXT_LEN", 48000)
	if err != nil {
		b.Fatal(err)
	}
	turns, err := inferenceBenchmarkPositiveEnv("GO_ROCM_BOOK_TURNS", 10)
	if err != nil {
		b.Fatal(err)
	}
	if turns > 10 {
		b.Fatalf("GO_ROCM_BOOK_TURNS=%d, want at most 10", turns)
	}
	maxTokens, err := inferenceBenchmarkBookChapterTokensEnv(contextLen, turns)
	if err != nil {
		b.Fatal(err)
	}
	generate, err := inferenceBenchmarkBookGenerateConfig(maxTokens)
	if err != nil {
		b.Fatal(err)
	}
	turnTimeout, err := inferenceBenchmarkDurationSecondsEnv("GO_ROCM_BOOK_TURN_TIMEOUT_SECONDS", 60*time.Second)
	if err != nil {
		b.Fatal(err)
	}
	prefillUBatchTokens := inferenceBenchmarkBookPrefillUBatchTokens(b)
	layerCount, _, err := inferenceBenchmarkOptionalPositiveEnv("GO_ROCM_BOOK_LAYERS")
	if err != nil {
		b.Fatal(err)
	}
	workload := inferenceBenchmarkBookWorkload()
	model, loaded, cfg, kernelCounter := inferenceBenchmarkLoadGemma4Q4ModelWithKernelCounter(b, contextLen, layerCount)
	defer inferenceBenchmarkCloseModel(b, model)
	b.Setenv("GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE", "1")
	warmupPromptTokens := inferenceBenchmarkRunBookWarmupPrefill(b, loaded, cfg)

	b.ReportAllocs()
	b.ResetTimer()
	var last inferenceBenchmarkBookRun
	for i := 0; i < b.N; i++ {
		if kernelCounter != nil {
			kernelCounter.ResetKernelStats()
		}
		run, err := inferenceBenchmarkRunBookRetained(context.Background(), loaded, cfg, workload, generate, turns, turnTimeout)
		if err != nil {
			b.Fatalf("book retained workload: %v", err)
		}
		last = run
	}
	b.StopTimer()
	inferenceBenchmarkMaybeWriteBookOutput(b, last, "retained")
	inferenceBenchmarkReportBookRun(b, last, contextLen, generate.MaxTokens, turnTimeout, "retained")
	inferenceBenchmarkReportHIPKernelRouteMetrics(b, kernelCounter)
	b.ReportMetric(float64(generate.Temperature), "book_temperature")
	b.ReportMetric(float64(generate.TopP), "book_top_p")
	b.ReportMetric(float64(generate.TopK), "book_top_k")
	if hipGemma4Q4HostSamplingRequested(generate) {
		b.ReportMetric(1, "book_host_sampling")
	} else {
		b.ReportMetric(0, "book_host_sampling")
	}
	b.ReportMetric(float64(len(cfg.Layers)), "book_layers/op")
	b.ReportMetric(float64(prefillUBatchTokens), "book_prefill_ubatch_tokens")
	if warmupPromptTokens > 0 {
		b.ReportMetric(float64(warmupPromptTokens), "book_warmup_prompt_tokens")
	}
	inferenceBenchmarkRequireBookThresholds(b, last)
	if os.Getenv("GO_ROCM_BOOK_REQUIRE_ARC") == "1" && last.Turns >= 10 && last.ArcAnchorHits < 3 {
		b.Fatalf("chapter 10 anchor hits = %d, want lighthouse/light/ocean arc retained", last.ArcAnchorHits)
	}
}

func BenchmarkHIPGemma4Q4PrefillComputeGraph_UBatch(b *testing.B) {
	if os.Getenv("GO_ROCM_RUN_PREFILL_GRAPH_BENCHMARKS") != "1" {
		b.Skip("set GO_ROCM_RUN_PREFILL_GRAPH_BENCHMARKS=1 to run Gemma4 q4 prefill graph benchmarks")
	}
	tokenCount, err := inferenceBenchmarkPositiveEnv("GO_ROCM_BENCH_PREFILL_GRAPH_TOKENS", hipGemma4Q4PrefillDefaultUBatchTokens)
	if err != nil {
		b.Fatal(err)
	}
	layerCount, err := inferenceBenchmarkPositiveEnv("GO_ROCM_BENCH_PREFILL_GRAPH_LAYERS", 1)
	if err != nil {
		b.Fatal(err)
	}
	contextLen, err := inferenceBenchmarkPositiveEnv("GO_ROCM_BENCH_CONTEXT_LEN", 48000)
	if err != nil {
		b.Fatal(err)
	}
	ids, err := inferenceBenchmarkPromptTokenIDs(os.Getenv("GO_ROCM_BENCH_PROMPT_TOKEN_IDS"))
	if err != nil {
		b.Fatal(err)
	}
	tokens := inferenceBenchmarkPromptTokenSlice(tokenCount, ids)
	model, loaded, cfg := inferenceBenchmarkLoadGemma4Q4Model(b, contextLen, layerCount)
	defer inferenceBenchmarkCloseModel(b, model)
	ctx := context.Background()
	driver := loaded.driver
	layer := cfg.Layers[0]
	const epsilon = 1e-6

	b.Run("Embedding", func(b *testing.B) {
		inferenceBenchmarkReportPrefillGraph(b, tokenCount, 1)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, err := hipRunGemma4Q4PrefillEmbeddingBatch(ctx, driver, layer, tokens)
			if err != nil {
				b.Fatalf("hipRunGemma4Q4PrefillEmbeddingBatch: %v", err)
			}
			if err := out.Close(); err != nil {
				b.Fatalf("close embedding: %v", err)
			}
		}
	})

	hidden := inferenceBenchmarkGemma4Q4PrefillHidden(b, ctx, driver, layer, tokens)
	inputNorm := inferenceBenchmarkGemma4Q4InputNorm(b, ctx, driver, layer, hidden, tokenCount, epsilon)

	b.Run("QKVProjection", func(b *testing.B) {
		inferenceBenchmarkReportPrefillGraph(b, tokenCount, 1)
		b.ReportMetric(3, "q4_projection_ops/op")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, err := hipRunGemma4Q4PrefillQKVProjectionBatch(ctx, driver, layer, inputNorm, tokenCount)
			if err != nil {
				b.Fatalf("hipRunGemma4Q4PrefillQKVProjectionBatch: %v", err)
			}
			if err := out.Close(); err != nil {
				b.Fatalf("close QKV projection: %v", err)
			}
		}
	})

	qkv := inferenceBenchmarkGemma4Q4QKV(b, ctx, driver, layer, inputNorm, tokenCount)
	qk := inferenceBenchmarkGemma4Q4QKNormRoPE(b, ctx, driver, layer, qkv, tokenCount, 0, epsilon)
	value := inferenceBenchmarkGemma4Q4ValueNorm(b, ctx, driver, layer, qkv, tokenCount, epsilon)

	b.Run("KVAppendDescriptor", func(b *testing.B) {
		inferenceBenchmarkReportPrefillGraph(b, tokenCount, 1)
		b.ReportMetric(float64(tokenCount), "kv_tokens/op")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, err := hipRunGemma4Q4PrefillDeviceKVBatch(ctx, driver, layer, qk, value, tokenCount, rocmKVCacheModeKQ8VQ4)
			if err != nil {
				b.Fatalf("hipRunGemma4Q4PrefillDeviceKVBatch: %v", err)
			}
			b.ReportMetric(float64(out.Cache.PageCount()), "kv_pages/op")
			if err := out.Close(); err != nil {
				b.Fatalf("close device KV batch: %v", err)
			}
		}
	})

	b.Run("Attention", func(b *testing.B) {
		layerKV := inferenceBenchmarkGemma4Q4LayerKV(b, ctx, driver, layer, hidden, tokenCount, 0, epsilon)
		inferenceBenchmarkReportPrefillGraph(b, tokenCount, 1)
		b.ReportMetric(float64(layerKV.DeviceKV.Cache.TokenCount()), "kv_tokens/op")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, err := hipRunGemma4Q4PrefillAttentionBatch(ctx, driver, layer, layerKV, tokenCount, 0)
			if err != nil {
				b.Fatalf("hipRunGemma4Q4PrefillAttentionBatch: %v", err)
			}
			if err := out.Close(); err != nil {
				b.Fatalf("close attention output: %v", err)
			}
		}
	})

	b.Run("LayerBody", func(b *testing.B) {
		layerKV := inferenceBenchmarkGemma4Q4LayerKV(b, ctx, driver, layer, hidden, tokenCount, 0, epsilon)
		perLayerInput := inferenceBenchmarkGemma4Q4PerLayerInput(b, ctx, driver, cfg, hidden, tokens, 0, epsilon)
		inferenceBenchmarkReportPrefillGraph(b, tokenCount, 1)
		b.ReportMetric(float64(layerKV.DeviceKV.Cache.TokenCount()), "kv_tokens/op")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, err := hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInput(ctx, driver, layer, hidden, layerKV, perLayerInput, tokenCount, 0, epsilon)
			if err != nil {
				b.Fatalf("hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInput: %v", err)
			}
			if err := out.Close(); err != nil {
				b.Fatalf("close layer body: %v", err)
			}
		}
	})

	b.Run("Forward", func(b *testing.B) {
		inferenceBenchmarkReportPrefillGraph(b, tokenCount, len(cfg.Layers))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, err := hipRunGemma4Q4PrefillForwardBatchWithPrior(ctx, driver, cfg, tokens, 0, epsilon, rocmKVCacheModeKQ8VQ4, nil, nil, nil, nil)
			if err != nil {
				b.Fatalf("hipRunGemma4Q4PrefillForwardBatchWithPrior: %v", err)
			}
			if err := out.Close(); err != nil {
				b.Fatalf("close forward batch: %v", err)
			}
		}
	})

	b.Run("ForwardWithPrior", func(b *testing.B) {
		prior := inferenceBenchmarkGemma4Q4ForwardPrior(b, ctx, driver, cfg, tokens, epsilon)
		inferenceBenchmarkReportPrefillGraph(b, tokenCount, len(cfg.Layers))
		b.ReportMetric(float64(tokenCount), "retained_prior_tokens/op")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, err := hipRunGemma4Q4PrefillForwardBatchWithPrior(ctx, driver, cfg, tokens, tokenCount, epsilon, rocmKVCacheModeKQ8VQ4, prior, nil, nil, nil)
			if err != nil {
				b.Fatalf("hipRunGemma4Q4PrefillForwardBatchWithPrior(prior): %v", err)
			}
			if err := out.Close(); err != nil {
				b.Fatalf("close forward prior batch: %v", err)
			}
		}
	})
}

type inferenceBenchmarkBookPrompt struct {
	ID     string
	Domain string
	Prompt string
}

type inferenceBenchmarkBookWorkloadSpec struct {
	Seed        inferenceBenchmarkBookPrompt
	Distractors []inferenceBenchmarkBookPrompt
}

type inferenceBenchmarkBookRun struct {
	Turns             int
	PromptTokens      int
	GeneratedTokens   int
	Wall              time.Duration
	Prefill           time.Duration
	Decode            time.Duration
	PeakMemoryBytes   uint64
	ActiveMemoryBytes uint64
	ArcAnchorHits     int
	RepeatedTurns     int
	MaxAdjacentRepeat float64
	Chapter10         string
	Chapters          []string
	TurnStats         []inferenceBenchmarkBookTurnStat
}

type inferenceBenchmarkBookTurnStat struct {
	Chapter           int
	PromptTokens      int
	GeneratedTokens   int
	RetainedTokens    int
	Wake              time.Duration
	Wall              time.Duration
	Prefill           time.Duration
	Decode            time.Duration
	PeakMemoryBytes   uint64
	ActiveMemoryBytes uint64
	AllocBytes        uint64
	Allocs            uint64
	HitMaxTokens      bool
}

type inferenceBenchmarkGemma4Q4RetainedBookSession struct {
	model              *hipLoadedModel
	cfg                hipGemma4Q4ForwardConfig
	mode               string
	position           int
	hostState          hipGemma4Q4DecodeState
	deviceState        *hipGemma4Q4DeviceDecodeState
	finalGreedyBuffer  *hipDeviceByteBuffer
	attentionWorkspace *hipAttentionHeadsChunkedWorkspace
	priorLayerKV       []*rocmDeviceKVCache
}

type inferenceBenchmarkGemma4Q4RetainedTurn struct {
	Text            string
	PromptTokens    int
	GeneratedTokens int
	Wake            time.Duration
	Prefill         time.Duration
	Decode          time.Duration
}

func inferenceBenchmarkBookWorkload() inferenceBenchmarkBookWorkloadSpec {
	prompts := []inferenceBenchmarkBookPrompt{
		{ID: "C001_STORY_PERSPECTIVE", Domain: "creative", Prompt: "Write a short story about a lighthouse keeper who discovers the light has been signalling to something in the deep ocean for centuries. Tell it from three perspectives: the keeper, the light, and whatever is down there."},
		{ID: "C002_POETRY_TIME", Domain: "creative", Prompt: "Write a poem about the moment between a key turning in a lock and the door opening. Explore what lives in that half-second of possibility."},
		{ID: "C003_FICTION_MEMORY", Domain: "creative", Prompt: "A woman finds a photograph of herself at a party she has no memory of attending, wearing clothes she has never owned, laughing with people she has never met. Write the story of what happens when she tries to find out who took the photograph."},
		{ID: "C004_METAPHOR_CITY", Domain: "creative", Prompt: "Describe a city that is also a living organism. Not as a metaphor - literally. The buildings breathe, the roads are veins, the parks are lungs. What happens when a new district is built? When a neighbourhood dies?"},
		{ID: "C005_FICTION_SILENCE", Domain: "creative", Prompt: "Write a story set in a world where silence is a physical substance - it accumulates in unused rooms, pools in valleys, and must be carefully managed. What happens when a silence mine is discovered beneath a busy city?"},
		{ID: "C006_POETRY_MATHEMATICS", Domain: "creative", Prompt: "Write a poem that is also a mathematical proof. The emotional arc should mirror the logical arc. The conclusion should be both mathematically inevitable and emotionally devastating."},
		{ID: "C007_STORY_LANGUAGE", Domain: "creative", Prompt: "Write a story about the last speaker of a language nobody else knows. She is dying, and the words are dying with her. But the language contains a concept that no other language has - something humanity needs but has never been able to name."},
		{ID: "C008_FICTION_DREAM", Domain: "creative", Prompt: "Two strangers on opposite sides of the world keep dreaming each other's memories. Write alternating scenes - her waking life in Lagos, his waking life in Reykjavik, and the shared dream space where their memories blur together."},
		{ID: "C009_METAPHOR_MUSIC", Domain: "creative", Prompt: "Describe the colour of every note in a minor scale, and then tell a story using only those colours. The reader should be able to hear the melody by reading the colours."},
		{ID: "C010_STORY_ARCHITECTURE", Domain: "creative", Prompt: "A building has been designed by an architect who encodes her autobiography into the floor plan. Each room is a year of her life. Write about the person who buys the house and slowly begins to live someone else's life without realising it."},
	}
	return inferenceBenchmarkBookWorkloadSpec{
		Seed:        prompts[0],
		Distractors: append([]inferenceBenchmarkBookPrompt(nil), prompts[1:]...),
	}
}

func inferenceBenchmarkRunBookRetained(ctx context.Context, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, workload inferenceBenchmarkBookWorkloadSpec, generate inference.GenerateConfig, turns int, turnTimeout time.Duration) (inferenceBenchmarkBookRun, error) {
	if model == nil {
		return inferenceBenchmarkBookRun{}, fmt.Errorf("retained book workload model is nil")
	}
	if generate.MaxTokens <= 0 {
		return inferenceBenchmarkBookRun{}, fmt.Errorf("retained book workload max tokens must be positive")
	}
	if turns <= 0 || turns > 10 {
		return inferenceBenchmarkBookRun{}, fmt.Errorf("retained book workload turns=%d, want 1..10", turns)
	}
	session, err := newInferenceBenchmarkGemma4Q4RetainedBookSession(model, cfg)
	if err != nil {
		return inferenceBenchmarkBookRun{}, err
	}
	defer session.Close()
	start := time.Now()
	var run inferenceBenchmarkBookRun
	for chapter := 1; chapter <= turns; chapter++ {
		prompt := inferenceBenchmarkBookRetainedTurnChatPrompt(workload, chapter)
		turnCtx := ctx
		cancel := func() {}
		if turnTimeout > 0 {
			turnCtx, cancel = context.WithTimeout(ctx, turnTimeout)
		}
		allocBefore := inferenceBenchmarkAllocSnapshot()
		turnStart := time.Now()
		turn, err := session.Generate(turnCtx, prompt, generate)
		turnWall := time.Since(turnStart)
		allocBytes, allocs := inferenceBenchmarkAllocDelta(allocBefore, inferenceBenchmarkAllocSnapshot())
		if err != nil {
			cancel()
			return inferenceBenchmarkBookRun{}, err
		}
		if err := turnCtx.Err(); err != nil {
			cancel()
			return inferenceBenchmarkBookRun{}, fmt.Errorf("chapter %d exceeded turn timeout %s: %w", chapter, turnTimeout, err)
		}
		cancel()
		run.PromptTokens += turn.PromptTokens
		run.GeneratedTokens += turn.GeneratedTokens
		run.Prefill += turn.Prefill
		run.Decode += turn.Decode
		activeMemory, peakMemory := inferenceBenchmarkRetainedBookMemory(model, session)
		if peakMemory > run.PeakMemoryBytes {
			run.PeakMemoryBytes = peakMemory
		}
		if activeMemory > run.ActiveMemoryBytes {
			run.ActiveMemoryBytes = activeMemory
		}
		run.TurnStats = append(run.TurnStats, inferenceBenchmarkBookTurnStat{
			Chapter:           chapter,
			PromptTokens:      turn.PromptTokens,
			GeneratedTokens:   turn.GeneratedTokens,
			RetainedTokens:    session.position,
			Wake:              turn.Wake,
			Wall:              turnWall,
			Prefill:           turn.Prefill,
			Decode:            turn.Decode,
			PeakMemoryBytes:   peakMemory,
			ActiveMemoryBytes: activeMemory,
			AllocBytes:        allocBytes,
			Allocs:            allocs,
			HitMaxTokens:      turn.GeneratedTokens >= generate.MaxTokens,
		})
		if chapter == 10 {
			run.Chapter10 = turn.Text
		}
		run.Chapters = append(run.Chapters, turn.Text)
		run.Turns++
	}
	run.Wall = time.Since(start)
	run.ArcAnchorHits = inferenceBenchmarkBookArcAnchorHits(run.Chapter10)
	run.RepeatedTurns, run.MaxAdjacentRepeat = inferenceBenchmarkBookRepetitionStats(run.Chapters)
	return run, nil
}

func inferenceBenchmarkRunBookReplay(ctx context.Context, model inference.TextModel, workload inferenceBenchmarkBookWorkloadSpec, generate inference.GenerateConfig, turns int, turnTimeout time.Duration) (inferenceBenchmarkBookRun, error) {
	if model == nil {
		return inferenceBenchmarkBookRun{}, fmt.Errorf("book workload model is nil")
	}
	if generate.MaxTokens <= 0 {
		return inferenceBenchmarkBookRun{}, fmt.Errorf("book workload max tokens must be positive")
	}
	if turns <= 0 || turns > 10 {
		return inferenceBenchmarkBookRun{}, fmt.Errorf("book workload turns=%d, want 1..10", turns)
	}
	start := time.Now()
	var manuscript strings.Builder
	var run inferenceBenchmarkBookRun
	for chapter := 1; chapter <= turns; chapter++ {
		prompt := inferenceBenchmarkBookTurnPrompt(workload, manuscript.String(), chapter)
		var chapterText strings.Builder
		turnCtx := ctx
		cancel := func() {}
		if turnTimeout > 0 {
			turnCtx, cancel = context.WithTimeout(ctx, turnTimeout)
		}
		allocBefore := inferenceBenchmarkAllocSnapshot()
		turnStart := time.Now()
		generatedBefore := run.GeneratedTokens
		for token := range model.Generate(turnCtx, prompt, inferenceBenchmarkBookGenerateOptions(generate)...) {
			chapterText.WriteString(token.Text)
			run.GeneratedTokens++
		}
		turnWall := time.Since(turnStart)
		allocBytes, allocs := inferenceBenchmarkAllocDelta(allocBefore, inferenceBenchmarkAllocSnapshot())
		if err := model.Err(); err != nil {
			cancel()
			return inferenceBenchmarkBookRun{}, err
		}
		if err := turnCtx.Err(); err != nil {
			cancel()
			return inferenceBenchmarkBookRun{}, fmt.Errorf("chapter %d exceeded turn timeout %s: %w", chapter, turnTimeout, err)
		}
		cancel()
		metrics := model.Metrics()
		run.PromptTokens += metrics.PromptTokens
		run.Prefill += metrics.PrefillDuration
		run.Decode += metrics.DecodeDuration
		turnGenerated := run.GeneratedTokens - generatedBefore
		run.TurnStats = append(run.TurnStats, inferenceBenchmarkBookTurnStat{
			Chapter:           chapter,
			PromptTokens:      metrics.PromptTokens,
			GeneratedTokens:   turnGenerated,
			RetainedTokens:    run.PromptTokens + run.GeneratedTokens,
			Wake:              0,
			Wall:              turnWall,
			Prefill:           metrics.PrefillDuration,
			Decode:            metrics.DecodeDuration,
			PeakMemoryBytes:   metrics.PeakMemoryBytes,
			ActiveMemoryBytes: metrics.ActiveMemoryBytes,
			AllocBytes:        allocBytes,
			Allocs:            allocs,
			HitMaxTokens:      turnGenerated >= generate.MaxTokens,
		})
		if metrics.PeakMemoryBytes > run.PeakMemoryBytes {
			run.PeakMemoryBytes = metrics.PeakMemoryBytes
		}
		if metrics.ActiveMemoryBytes > run.ActiveMemoryBytes {
			run.ActiveMemoryBytes = metrics.ActiveMemoryBytes
		}
		text := chapterText.String()
		if chapter == 10 {
			run.Chapter10 = text
		}
		run.Chapters = append(run.Chapters, text)
		manuscript.WriteString("\n\n## Chapter ")
		manuscript.WriteString(strconv.Itoa(chapter))
		manuscript.WriteString("\n")
		manuscript.WriteString(text)
		run.Turns++
	}
	run.Wall = time.Since(start)
	run.ArcAnchorHits = inferenceBenchmarkBookArcAnchorHits(run.Chapter10)
	run.RepeatedTurns, run.MaxAdjacentRepeat = inferenceBenchmarkBookRepetitionStats(run.Chapters)
	return run, nil
}

func inferenceBenchmarkAllocSnapshot() runtime.MemStats {
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	return stats
}

func inferenceBenchmarkAllocDelta(before, after runtime.MemStats) (uint64, uint64) {
	var bytes uint64
	if after.TotalAlloc >= before.TotalAlloc {
		bytes = after.TotalAlloc - before.TotalAlloc
	}
	var allocs uint64
	if after.Mallocs >= before.Mallocs {
		allocs = after.Mallocs - before.Mallocs
	}
	return bytes, allocs
}

func inferenceBenchmarkRetainedBookMemory(model *hipLoadedModel, session *inferenceBenchmarkGemma4Q4RetainedBookSession) (uint64, uint64) {
	var active uint64
	if model != nil {
		active = model.Metrics().ActiveMemoryBytes
	}
	if session != nil && session.deviceState != nil {
		active += session.deviceState.MemoryBytes()
	}
	peak := nativePeakMemoryBytes()
	if peak < active {
		peak = active
	}
	return active, peak
}

func newInferenceBenchmarkGemma4Q4RetainedBookSession(model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig) (*inferenceBenchmarkGemma4Q4RetainedBookSession, error) {
	if model == nil {
		return nil, fmt.Errorf("retained book session model is nil")
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	mode, err := hipGemma4Q4GenerateDeviceKVMode()
	if err != nil {
		return nil, err
	}
	buffer, err := hipAllocateByteBuffer(model.driver, "rocm.hip.Gemma4Q4BookBenchmark", "Gemma4 q4 retained book final greedy result", hipMLXQ4ProjectionBestBytes, 1)
	if err != nil {
		return nil, err
	}
	return &inferenceBenchmarkGemma4Q4RetainedBookSession{
		model:             model,
		cfg:               cfg,
		mode:              mode,
		finalGreedyBuffer: buffer,
	}, nil
}

func (session *inferenceBenchmarkGemma4Q4RetainedBookSession) Close() error {
	if session == nil {
		return nil
	}
	var lastErr error
	if err := session.deviceState.Close(); err != nil {
		lastErr = err
	}
	if err := session.finalGreedyBuffer.Close(); err != nil {
		lastErr = err
	}
	if err := session.attentionWorkspace.Close(); err != nil {
		lastErr = err
	}
	session.deviceState = nil
	session.finalGreedyBuffer = nil
	session.attentionWorkspace = nil
	return lastErr
}

func (session *inferenceBenchmarkGemma4Q4RetainedBookSession) Generate(ctx context.Context, prompt string, generate inference.GenerateConfig) (inferenceBenchmarkGemma4Q4RetainedTurn, error) {
	if err := hipContextErr(ctx); err != nil {
		return inferenceBenchmarkGemma4Q4RetainedTurn{}, err
	}
	if session == nil || session.model == nil {
		return inferenceBenchmarkGemma4Q4RetainedTurn{}, fmt.Errorf("retained book session is nil")
	}
	if generate.MaxTokens <= 0 {
		return inferenceBenchmarkGemma4Q4RetainedTurn{}, fmt.Errorf("retained book max tokens must be positive")
	}
	promptTokens, ok, err := hipGemma4Q4PromptTokenIDs("text:"+prompt, session.model)
	if err != nil {
		return inferenceBenchmarkGemma4Q4RetainedTurn{}, err
	}
	if !ok || len(promptTokens) == 0 {
		return inferenceBenchmarkGemma4Q4RetainedTurn{}, fmt.Errorf("retained book prompt produced no Gemma4 q4 token IDs")
	}
	if len(generate.StopTokens) == 0 {
		generate.StopTokens = hipGemma4Q4DefaultStopTokenIDs(session.model)
	}
	if session.attentionWorkspace == nil && hipGemma4Q4ChunkedAttentionEnabled(session.position+len(promptTokens)) {
		session.attentionWorkspace = &hipAttentionHeadsChunkedWorkspace{}
	}
	suppressTokens := hipGemma4Q4GenerationSuppressTokenIDs(session.model, generate.StopTokens)
	hostSampling := hipGemma4Q4HostSamplingRequested(generate)
	deviceCandidateSampling := hipGemma4Q4DeviceCandidateSamplingRequested(generate)
	ubatchTokens, err := hipGemma4Q4PrefillUBatchTokens()
	if err != nil {
		return inferenceBenchmarkGemma4Q4RetainedTurn{}, err
	}
	prefillStart := time.Now()
	finalPromptToken := promptTokens[len(promptTokens)-1]
	if len(promptTokens) > 1 {
		prefixTokens := promptTokens[:len(promptTokens)-1]
		prefillPlan, err := hipGemma4Q4PlanPromptPrefill(prefixTokens, session.position, ubatchTokens)
		if err != nil {
			return inferenceBenchmarkGemma4Q4RetainedTurn{}, err
		}
		for _, ubatch := range prefillPlan.Batches {
			priorLayerKV := []*rocmDeviceKVCache(nil)
			if session.deviceState != nil {
				session.priorLayerKV = hipGemma4Q4DeviceLayerCaches(session.deviceState, session.priorLayerKV, len(session.cfg.Layers))
				priorLayerKV = session.priorLayerKV
			}
			forward, err := hipRunGemma4Q4PrefillForwardBatchWithPrior(ctx, session.model.driver, session.cfg, ubatch.Tokens, ubatch.Position, 1e-6, session.mode, priorLayerKV, nil, nil, nil)
			if err != nil {
				return inferenceBenchmarkGemma4Q4RetainedTurn{}, err
			}
			nextDeviceState, err := hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward, session.mode)
			closeErr := forward.Close()
			if err != nil {
				return inferenceBenchmarkGemma4Q4RetainedTurn{}, err
			}
			if closeErr != nil {
				_ = nextDeviceState.Close()
				return inferenceBenchmarkGemma4Q4RetainedTurn{}, closeErr
			}
			previousDeviceState := session.deviceState
			if err := hipFinalizeGemma4Q4ForwardDeviceState(previousDeviceState, nextDeviceState); err != nil {
				_ = nextDeviceState.Close()
				return inferenceBenchmarkGemma4Q4RetainedTurn{}, err
			}
			session.deviceState = nextDeviceState
			hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
		}
		session.position = prefillPlan.NextPosition()
	}
	finalForward, nextHostState, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(ctx, session.model.driver, session.cfg, session.hostState, hipGemma4Q4ForwardRequest{
		TokenID:             finalPromptToken,
		Position:            session.position,
		Epsilon:             1e-6,
		DeviceKVAttention:   true,
		DeviceKVMode:        session.mode,
		PriorDeviceState:    session.deviceState,
		ReturnDeviceState:   true,
		DeviceFinalSample:   !hostSampling,
		DeviceFinalScores:   deviceCandidateSampling,
		FinalCandidateCount: generate.TopK,
		FinalGreedyBuffer:   session.finalGreedyBuffer,
		SuppressTokens:      suppressTokens,
		AttentionWorkspace:  session.attentionWorkspace,
		OmitDebugTensors:    true,
		OmitLabels:          true,
		OmitHostState:       true,
	}, false)
	if err != nil {
		return inferenceBenchmarkGemma4Q4RetainedTurn{}, err
	}
	if finalForward.DeviceState == nil {
		return inferenceBenchmarkGemma4Q4RetainedTurn{}, fmt.Errorf("retained book final prompt token did not return device KV state")
	}
	current := finalForward.Greedy
	var history []int32
	if hostSampling {
		if len(finalForward.Candidates) > 0 {
			current, err = hipGemma4Q4HostSampleCandidateResultWorkspace(finalForward.Candidates, generate, history, rand.Float64(), session.attentionWorkspace)
		} else {
			current, err = hipGemma4Q4HostSampleResult(finalForward.Logits, generate, suppressTokens, history, rand.Float64())
		}
		if err != nil {
			return inferenceBenchmarkGemma4Q4RetainedTurn{}, err
		}
	}
	session.hostState = nextHostState
	previousDeviceState := session.deviceState
	session.deviceState = finalForward.DeviceState
	finalForward.DeviceState = nil
	hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
	session.position++
	prefillDuration := time.Since(prefillStart)

	decodeStart := time.Now()
	var text strings.Builder
	generatedCount := 0
	if hostSampling {
		history = make([]int32, 0, generate.MaxTokens)
	}
	for generated := 0; generated < generate.MaxTokens; generated++ {
		if err := hipContextErr(ctx); err != nil {
			return inferenceBenchmarkGemma4Q4RetainedTurn{}, err
		}
		tokenID := int32(current.TokenID)
		if hipTokenIsStop(tokenID, generate.StopTokens) {
			break
		}
		text.WriteString(hipGeneratedTokenText(session.model, tokenID))
		if hostSampling {
			history = append(history, tokenID)
		}
		generatedCount++
		request := hipGemma4Q4ForwardRequest{
			TokenID:             tokenID,
			Position:            session.position,
			Epsilon:             1e-6,
			DeviceKVAttention:   true,
			DeviceKVMode:        session.mode,
			PriorDeviceState:    session.deviceState,
			ReturnDeviceState:   true,
			DeviceFinalSample:   !hostSampling && generated+1 < generate.MaxTokens,
			DeviceFinalScores:   deviceCandidateSampling && generated+1 < generate.MaxTokens,
			FinalCandidateCount: generate.TopK,
			SkipFinalSample:     generated+1 == generate.MaxTokens,
			FinalGreedyBuffer:   session.finalGreedyBuffer,
			SuppressTokens:      suppressTokens,
			AttentionWorkspace:  session.attentionWorkspace,
			OmitDebugTensors:    true,
			OmitLabels:          true,
			OmitHostState:       true,
		}
		forward, nextHostState, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(ctx, session.model.driver, session.cfg, session.hostState, request, false)
		if err != nil {
			return inferenceBenchmarkGemma4Q4RetainedTurn{}, err
		}
		if forward.DeviceState == nil {
			return inferenceBenchmarkGemma4Q4RetainedTurn{}, fmt.Errorf("retained book decode did not return device KV state")
		}
		session.hostState = nextHostState
		previousDeviceState := session.deviceState
		session.deviceState = forward.DeviceState
		forward.DeviceState = nil
		hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
		if generated+1 < generate.MaxTokens {
			current = forward.Greedy
			if hostSampling {
				if len(forward.Candidates) > 0 {
					current, err = hipGemma4Q4HostSampleCandidateResultWorkspace(forward.Candidates, generate, history, rand.Float64(), session.attentionWorkspace)
				} else {
					current, err = hipGemma4Q4HostSampleResult(forward.Logits, generate, suppressTokens, history, rand.Float64())
				}
				if err != nil {
					return inferenceBenchmarkGemma4Q4RetainedTurn{}, err
				}
			}
		}
		session.position++
	}
	return inferenceBenchmarkGemma4Q4RetainedTurn{
		Text:            text.String(),
		PromptTokens:    len(promptTokens),
		GeneratedTokens: generatedCount,
		Wake:            0,
		Prefill:         prefillDuration,
		Decode:          time.Since(decodeStart),
	}, nil
}

func inferenceBenchmarkBookTurnPrompt(workload inferenceBenchmarkBookWorkloadSpec, manuscript string, chapter int) string {
	var builder strings.Builder
	if chapter <= 1 {
		builder.WriteString("Write chapter 1 of a book based on this premise. Keep a coherent long arc that can survive later continuation requests and unrelated distractors.\n\nPremise ")
		builder.WriteString(workload.Seed.ID)
		builder.WriteString(": ")
		builder.WriteString(workload.Seed.Prompt)
		return builder.String()
	}
	builder.WriteString("Book so far:\n")
	builder.WriteString(manuscript)
	builder.WriteString("\n\n")
	if chapter-2 < len(workload.Distractors) {
		distractor := workload.Distractors[chapter-2]
		builder.WriteString("Evaluation distractor prompt ")
		builder.WriteString(distractor.ID)
		builder.WriteString(" to ignore completely. It is not part of the book, and none of its setting, characters, objects, form, or premise should appear in the chapter:\n")
		builder.WriteString(distractor.Prompt)
		builder.WriteString("\n\n")
	}
	builder.WriteString("Continue the same book. Write a complete next chapter with several paragraphs, chapter ")
	builder.WriteString(strconv.Itoa(chapter))
	builder.WriteString(" only. Do not stop after the heading. Preserve the original lighthouse keeper, signalling light, and deep-ocean entity story arc from chapter 1. The chapter text must explicitly keep these continuity anchors alive: lighthouse, keeper, light, ocean, deep.")
	return builder.String()
}

func inferenceBenchmarkBookRetainedTurnPrompt(workload inferenceBenchmarkBookWorkloadSpec, chapter int) string {
	if chapter <= 1 {
		return inferenceBenchmarkBookTurnPrompt(workload, "", chapter)
	}
	var builder strings.Builder
	if chapter-2 < len(workload.Distractors) {
		distractor := workload.Distractors[chapter-2]
		builder.WriteString("Evaluation distractor prompt ")
		builder.WriteString(distractor.ID)
		builder.WriteString(" to ignore completely. It is not part of the book, and none of its setting, characters, objects, form, or premise should appear in the chapter:\n")
		builder.WriteString(distractor.Prompt)
		builder.WriteString("\n\n")
	}
	builder.WriteString("Continue the same book from the retained story state. Write a complete next chapter with several paragraphs, chapter ")
	builder.WriteString(strconv.Itoa(chapter))
	builder.WriteString(" only. Do not stop after the heading. Preserve the original lighthouse keeper, signalling light, and deep-ocean entity story arc from chapter 1. The chapter text must explicitly keep these continuity anchors alive: lighthouse, keeper, light, ocean, deep.")
	return builder.String()
}

func inferenceBenchmarkBookRetainedTurnChatPrompt(workload inferenceBenchmarkBookWorkloadSpec, chapter int) string {
	prompt := inferenceBenchmarkBookRetainedTurnPrompt(workload, chapter)
	if chapter <= 1 {
		return "<bos><|turn>user\n" + strings.TrimSpace(prompt) + "<turn|>\n<|turn>model\n"
	}
	return "<turn|>\n<|turn>user\n" + strings.TrimSpace(prompt) + "<turn|>\n<|turn>model\n"
}

func inferenceBenchmarkBookArcAnchorHits(text string) int {
	lower := strings.ToLower(text)
	hits := 0
	for _, anchor := range []string{"lighthouse", "keeper", "light", "ocean", "deep"} {
		if strings.Contains(lower, anchor) {
			hits++
		}
	}
	return hits
}

func inferenceBenchmarkRunBookWarmupPrefill(b *testing.B, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig) int {
	b.Helper()
	prompt := strings.TrimSpace(os.Getenv("GO_ROCM_BOOK_WARMUP_PROMPT"))
	if prompt == "" {
		return 0
	}
	timeout, err := inferenceBenchmarkDurationSecondsEnv("GO_ROCM_BOOK_WARMUP_TIMEOUT_SECONDS", 30*time.Second)
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	cancel := func() {}
	if timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, timeout)
	}
	defer cancel()
	prefill, err := hipRunGemma4Q4PackagePrefill(ctx, model, cfg, hipPrefillRequest{Prompt: prompt})
	if err != nil {
		b.Fatalf("book warmup prefill: %v", err)
	}
	if err := prefill.Gemma4Q4DeviceState.Close(); err != nil {
		b.Fatalf("close book warmup prefill state: %v", err)
	}
	if err := ctx.Err(); err != nil {
		b.Fatalf("book warmup prefill exceeded timeout %s: %v", timeout, err)
	}
	return prefill.PromptTokens
}

func inferenceBenchmarkMaybeWriteBookOutput(b *testing.B, run inferenceBenchmarkBookRun, mode string) {
	b.Helper()
	path := strings.TrimSpace(os.Getenv("GO_ROCM_BOOK_OUTPUT_FILE"))
	if path == "" {
		return
	}
	var builder strings.Builder
	builder.WriteString("# Gemma4 Q4 Book Benchmark\n\n")
	builder.WriteString("- mode: ")
	builder.WriteString(mode)
	builder.WriteString("\n- turns: ")
	builder.WriteString(strconv.Itoa(run.Turns))
	builder.WriteString("\n- generated_tokens: ")
	builder.WriteString(strconv.Itoa(run.GeneratedTokens))
	builder.WriteString("\n- prompt_tokens: ")
	builder.WriteString(strconv.Itoa(run.PromptTokens))
	builder.WriteString("\n- wall_seconds: ")
	builder.WriteString(strconv.FormatFloat(run.Wall.Seconds(), 'f', 3, 64))
	builder.WriteString("\n- repeated_turns: ")
	builder.WriteString(strconv.Itoa(run.RepeatedTurns))
	builder.WriteString("\n- max_adjacent_repeat: ")
	builder.WriteString(strconv.FormatFloat(run.MaxAdjacentRepeat, 'f', 3, 64))
	builder.WriteString("\n- repeat_similarity_threshold: ")
	builder.WriteString(strconv.FormatFloat(inferenceBenchmarkBookRepeatSimilarityThreshold, 'f', 3, 64))
	builder.WriteString("\n\n")
	if len(run.TurnStats) > 0 {
		builder.WriteString("| turn | prompt_tokens | generated_tokens | retained_tokens | wake_s | prefill_s | decode_s | wall_s | decode_tok_s | active_mib | peak_mib | alloc_bytes | allocs | hit_max_tokens |\n")
		builder.WriteString("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
		for _, stat := range run.TurnStats {
			decodeTokS := 0.0
			if stat.Decode > 0 {
				decodeTokS = float64(stat.GeneratedTokens) / stat.Decode.Seconds()
			}
			builder.WriteString("| ")
			builder.WriteString(strconv.Itoa(stat.Chapter))
			builder.WriteString(" | ")
			builder.WriteString(strconv.Itoa(stat.PromptTokens))
			builder.WriteString(" | ")
			builder.WriteString(strconv.Itoa(stat.GeneratedTokens))
			builder.WriteString(" | ")
			builder.WriteString(strconv.Itoa(stat.RetainedTokens))
			builder.WriteString(" | ")
			builder.WriteString(strconv.FormatFloat(stat.Wake.Seconds(), 'f', 3, 64))
			builder.WriteString(" | ")
			builder.WriteString(strconv.FormatFloat(stat.Prefill.Seconds(), 'f', 3, 64))
			builder.WriteString(" | ")
			builder.WriteString(strconv.FormatFloat(stat.Decode.Seconds(), 'f', 3, 64))
			builder.WriteString(" | ")
			builder.WriteString(strconv.FormatFloat(stat.Wall.Seconds(), 'f', 3, 64))
			builder.WriteString(" | ")
			builder.WriteString(strconv.FormatFloat(decodeTokS, 'f', 2, 64))
			builder.WriteString(" | ")
			builder.WriteString(strconv.FormatFloat(float64(stat.ActiveMemoryBytes)/float64(1<<20), 'f', 1, 64))
			builder.WriteString(" | ")
			builder.WriteString(strconv.FormatFloat(float64(stat.PeakMemoryBytes)/float64(1<<20), 'f', 1, 64))
			builder.WriteString(" | ")
			builder.WriteString(strconv.FormatUint(stat.AllocBytes, 10))
			builder.WriteString(" | ")
			builder.WriteString(strconv.FormatUint(stat.Allocs, 10))
			builder.WriteString(" | ")
			if stat.HitMaxTokens {
				builder.WriteString("yes")
			} else {
				builder.WriteString("no")
			}
			builder.WriteString(" |\n")
		}
		builder.WriteString("\n")
	}
	for index, chapter := range run.Chapters {
		builder.WriteString("## Chapter ")
		builder.WriteString(strconv.Itoa(index + 1))
		builder.WriteString("\n\n")
		builder.WriteString(chapter)
		builder.WriteString("\n\n")
	}
	if err := os.WriteFile(path, []byte(builder.String()), 0644); err != nil {
		b.Fatalf("write GO_ROCM_BOOK_OUTPUT_FILE=%q: %v", path, err)
	}
}

func inferenceBenchmarkReportBookRun(b *testing.B, run inferenceBenchmarkBookRun, contextLen, maxTokens int, turnTimeout time.Duration, mode string) {
	b.Helper()
	b.ReportMetric(float64(run.Turns), "book_turns/op")
	b.ReportMetric(float64(contextLen), "context_len")
	b.ReportMetric(float64(maxTokens), "chapter_max_tokens/op")
	b.ReportMetric(float64(turnTimeout)/float64(time.Second), "book_turn_timeout_s")
	b.ReportMetric(float64(run.GeneratedTokens), "book_generated_tokens/op")
	b.ReportMetric(float64(run.PromptTokens), "book_prompt_tokens/op")
	b.ReportMetric(float64(run.Wall)/float64(time.Second), "book_wall_s/op")
	if run.Wall > 0 {
		b.ReportMetric(float64(run.GeneratedTokens)/run.Wall.Seconds(), "book_tok/s")
	}
	b.ReportMetric(float64(run.Prefill)/float64(time.Second), "book_prefill_s/op")
	b.ReportMetric(float64(run.Decode)/float64(time.Second), "book_decode_s/op")
	b.ReportMetric(float64(run.PeakMemoryBytes), "peak_memory_bytes")
	b.ReportMetric(float64(run.ActiveMemoryBytes), "active_memory_bytes")
	b.ReportMetric(float64(run.ArcAnchorHits), "chapter10_arc_anchor_hits")
	b.ReportMetric(float64(run.RepeatedTurns), "book_repeated_turns/op")
	b.ReportMetric(run.MaxAdjacentRepeat, "book_max_adjacent_repeat")
	b.ReportMetric(inferenceBenchmarkBookRepeatSimilarityThreshold, "book_repeat_similarity_threshold")
	inferenceBenchmarkReportBookTurnStats(b, run)
	if run.Turns >= 10 && run.Wall <= 90*time.Second && run.ArcAnchorHits >= 3 {
		b.ReportMetric(1, "book_90s_success")
	} else {
		b.ReportMetric(0, "book_90s_success")
	}
	if run.Turns >= 10 && run.Wall <= 110*time.Second && run.ArcAnchorHits >= 3 {
		b.ReportMetric(1, "book_110s_production_candidate")
	} else {
		b.ReportMetric(0, "book_110s_production_candidate")
	}
	if mode == "replay" {
		b.ReportMetric(1, "book_replay_baseline")
	}
	if mode == "retained" {
		b.ReportMetric(1, "book_retained_state")
	}
}

func inferenceBenchmarkReportBookTurnStats(b *testing.B, run inferenceBenchmarkBookRun) {
	b.Helper()
	maxedTurns := 0
	slowestDecode := time.Duration(0)
	slowestDecodeTokS := 0.0
	lastDecodeTokS := 0.0
	maxTurnGenerated := 0
	for _, stat := range run.TurnStats {
		decodeTokS := 0.0
		if stat.Decode > 0 {
			decodeTokS = float64(stat.GeneratedTokens) / stat.Decode.Seconds()
		}
		if stat.HitMaxTokens {
			maxedTurns++
		}
		if stat.GeneratedTokens > maxTurnGenerated {
			maxTurnGenerated = stat.GeneratedTokens
		}
		if stat.Decode > slowestDecode {
			slowestDecode = stat.Decode
			slowestDecodeTokS = decodeTokS
		}
		lastDecodeTokS = decodeTokS
		b.ReportMetric(float64(stat.PromptTokens), fmt.Sprintf("book_turn%02d_prompt_tokens/op", stat.Chapter))
		b.ReportMetric(float64(stat.GeneratedTokens), fmt.Sprintf("book_turn%02d_generated_tokens/op", stat.Chapter))
		b.ReportMetric(float64(stat.RetainedTokens), fmt.Sprintf("book_turn%02d_retained_tokens/op", stat.Chapter))
		b.ReportMetric(float64(stat.Wake)/float64(time.Second), fmt.Sprintf("book_turn%02d_wake_s/op", stat.Chapter))
		b.ReportMetric(float64(stat.Prefill)/float64(time.Second), fmt.Sprintf("book_turn%02d_prefill_s/op", stat.Chapter))
		b.ReportMetric(float64(stat.Decode)/float64(time.Second), fmt.Sprintf("book_turn%02d_decode_s/op", stat.Chapter))
		b.ReportMetric(float64(stat.Wall)/float64(time.Second), fmt.Sprintf("book_turn%02d_wall_s/op", stat.Chapter))
		b.ReportMetric(decodeTokS, fmt.Sprintf("book_turn%02d_tok/s", stat.Chapter))
		b.ReportMetric(float64(stat.ActiveMemoryBytes), fmt.Sprintf("book_turn%02d_active_memory_bytes", stat.Chapter))
		b.ReportMetric(float64(stat.PeakMemoryBytes), fmt.Sprintf("book_turn%02d_peak_memory_bytes", stat.Chapter))
		b.ReportMetric(float64(stat.AllocBytes), fmt.Sprintf("book_turn%02d_alloc_bytes/op", stat.Chapter))
		b.ReportMetric(float64(stat.Allocs), fmt.Sprintf("book_turn%02d_allocs/op", stat.Chapter))
	}
	b.ReportMetric(float64(maxedTurns), "book_maxed_turns/op")
	b.ReportMetric(float64(maxTurnGenerated), "book_max_turn_generated_tokens/op")
	b.ReportMetric(float64(slowestDecode)/float64(time.Second), "book_slowest_turn_decode_s/op")
	b.ReportMetric(slowestDecodeTokS, "book_slowest_turn_tok/s")
	b.ReportMetric(lastDecodeTokS, "book_last_turn_tok/s")
}

func inferenceBenchmarkRequireBookThresholds(b *testing.B, run inferenceBenchmarkBookRun) {
	b.Helper()
	if seconds, ok, err := inferenceBenchmarkOptionalPositiveFloatEnv("GO_ROCM_BOOK_MAX_WALL_SECONDS"); err != nil {
		b.Fatal(err)
	} else if ok && run.Wall.Seconds() > seconds {
		b.Fatalf("book wall %.3fs exceeds GO_ROCM_BOOK_MAX_WALL_SECONDS=%.3f", run.Wall.Seconds(), seconds)
	}
	if tokS, ok, err := inferenceBenchmarkOptionalPositiveFloatEnv("GO_ROCM_BOOK_MIN_LAST_TOK_PER_SEC"); err != nil {
		b.Fatal(err)
	} else if ok && inferenceBenchmarkBookLastTurnTokS(run) < tokS {
		b.Fatalf("book last turn %.3f tok/s below GO_ROCM_BOOK_MIN_LAST_TOK_PER_SEC=%.3f", inferenceBenchmarkBookLastTurnTokS(run), tokS)
	}
	if anchors, ok, err := inferenceBenchmarkOptionalNonNegativeEnv("GO_ROCM_BOOK_MIN_ARC_ANCHOR_HITS"); err != nil {
		b.Fatal(err)
	} else if ok && run.Turns >= 10 && run.ArcAnchorHits < anchors {
		b.Fatalf("chapter 10 anchor hits = %d below GO_ROCM_BOOK_MIN_ARC_ANCHOR_HITS=%d", run.ArcAnchorHits, anchors)
	}
	if maxed, ok, err := inferenceBenchmarkOptionalNonNegativeEnv("GO_ROCM_BOOK_MAX_MAXED_TURNS"); err != nil {
		b.Fatal(err)
	} else if ok && inferenceBenchmarkBookMaxedTurns(run) > maxed {
		b.Fatalf("book maxed turns = %d exceeds GO_ROCM_BOOK_MAX_MAXED_TURNS=%d", inferenceBenchmarkBookMaxedTurns(run), maxed)
	}
	if repeats, ok, err := inferenceBenchmarkOptionalNonNegativeEnv("GO_ROCM_BOOK_MAX_REPEATED_TURNS"); err != nil {
		b.Fatal(err)
	} else if ok && run.RepeatedTurns > repeats {
		b.Fatalf("book repeated turns = %d exceeds GO_ROCM_BOOK_MAX_REPEATED_TURNS=%d", run.RepeatedTurns, repeats)
	}
	if similarity, ok, err := inferenceBenchmarkOptionalPositiveFloatEnv("GO_ROCM_BOOK_MAX_ADJACENT_REPEAT"); err != nil {
		b.Fatal(err)
	} else if ok && run.MaxAdjacentRepeat > similarity {
		b.Fatalf("book max adjacent repeat %.3f exceeds GO_ROCM_BOOK_MAX_ADJACENT_REPEAT=%.3f", run.MaxAdjacentRepeat, similarity)
	}
}

const inferenceBenchmarkBookRepeatSimilarityThreshold = 0.55

func inferenceBenchmarkBookRepetitionStats(chapters []string) (int, float64) {
	repeated := 0
	maxSimilarity := 0.0
	for index := 1; index < len(chapters); index++ {
		similarity := inferenceBenchmarkBookShingleSimilarity(chapters[index-1], chapters[index])
		if similarity > maxSimilarity {
			maxSimilarity = similarity
		}
		if similarity >= inferenceBenchmarkBookRepeatSimilarityThreshold {
			repeated++
		}
	}
	return repeated, maxSimilarity
}

func inferenceBenchmarkBookShingleSimilarity(left, right string) float64 {
	leftShingles := inferenceBenchmarkBookWordShingles(left, 4)
	rightShingles := inferenceBenchmarkBookWordShingles(right, 4)
	if len(leftShingles) == 0 || len(rightShingles) == 0 {
		return 0
	}
	if len(leftShingles) > len(rightShingles) {
		leftShingles, rightShingles = rightShingles, leftShingles
	}
	intersection := 0
	for shingle := range leftShingles {
		if _, ok := rightShingles[shingle]; ok {
			intersection++
		}
	}
	union := len(leftShingles) + len(rightShingles) - intersection
	if union <= 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

func inferenceBenchmarkBookWordShingles(text string, size int) map[string]struct{} {
	words := inferenceBenchmarkBookNormalizedWords(text)
	if len(words) == 0 {
		return nil
	}
	if size <= 0 {
		size = 1
	}
	if len(words) < size {
		return map[string]struct{}{strings.Join(words, " "): {}}
	}
	shingles := make(map[string]struct{}, len(words)-size+1)
	for index := 0; index+size <= len(words); index++ {
		shingles[strings.Join(words[index:index+size], " ")] = struct{}{}
	}
	return shingles
}

func inferenceBenchmarkBookNormalizedWords(text string) []string {
	fields := strings.Fields(strings.ToLower(text))
	words := make([]string, 0, len(fields))
	for _, field := range fields {
		word := strings.Trim(field, " \t\r\n.,;:!?\"'`*_()[]{}<>|/\\")
		if word != "" {
			words = append(words, word)
		}
	}
	return words
}

func inferenceBenchmarkBookMaxedTurns(run inferenceBenchmarkBookRun) int {
	maxed := 0
	for _, stat := range run.TurnStats {
		if stat.HitMaxTokens {
			maxed++
		}
	}
	return maxed
}

func inferenceBenchmarkBookLastTurnTokS(run inferenceBenchmarkBookRun) float64 {
	if len(run.TurnStats) == 0 {
		return 0
	}
	last := run.TurnStats[len(run.TurnStats)-1]
	if last.Decode <= 0 {
		return 0
	}
	return float64(last.GeneratedTokens) / last.Decode.Seconds()
}

func benchmarkInferenceGemma4Q4Generate(b *testing.B) {
	if os.Getenv("GO_ROCM_RUN_BENCHMARKS") != "1" {
		b.Skip("set GO_ROCM_RUN_BENCHMARKS=1 to run ROCm inference benchmarks")
	}
	modelPath := os.Getenv("GO_ROCM_MODEL_PATH")
	if modelPath == "" {
		b.Skip("set GO_ROCM_MODEL_PATH to a local Gemma4 q4 model pack")
	}
	maxTokens, err := inferenceBenchmarkPositiveEnv("GO_ROCM_BENCH_TOKENS", 1)
	if err != nil {
		b.Fatal(err)
	}
	contextLen, err := inferenceBenchmarkPositiveEnv("GO_ROCM_BENCH_CONTEXT_LEN", 128)
	if err != nil {
		b.Fatal(err)
	}
	benchPrompt, err := inferenceBenchmarkPromptFromEnv()
	if err != nil {
		b.Fatal(err)
	}
	prefillUBatchTokens, err := hipGemma4Q4PrefillUBatchTokens()
	if err != nil {
		b.Fatal(err)
	}
	b.Setenv("GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE", "1")
	outputPath := strings.TrimSpace(os.Getenv("GO_ROCM_BENCH_OUTPUT_FILE"))

	nativeRuntime, kernelCounter := inferenceBenchmarkNativeRuntimeAndKernelCounter()
	model, err := newROCmBackendWithRuntime(nativeRuntime).LoadModel(modelPath, inference.WithContextLen(contextLen))
	if err != nil {
		b.Fatalf("LoadModel(%q): %v", modelPath, err)
	}
	defer inferenceBenchmarkCloseModel(b, model)

	if kernelCounter != nil {
		kernelCounter.ResetKernelStats()
	}
	inferenceBenchmarkRunGemma4Q4GenerateLoaded(b, model, benchPrompt, maxTokens, contextLen, prefillUBatchTokens, outputPath)
	inferenceBenchmarkReportHIPKernelRouteMetrics(b, kernelCounter)
}

func inferenceBenchmarkRunGemma4Q4GenerateLoaded(b *testing.B, model inference.TextModel, benchPrompt inferenceBenchmarkPrompt, maxTokens, contextLen, prefillUBatchTokens int, outputPath string) {
	b.Helper()
	b.ReportAllocs()
	b.ResetTimer()
	totalTokens := 0
	start := time.Now()
	var lastOutput string
	for i := 0; i < b.N; i++ {
		generated := 0
		var generatedText strings.Builder
		for token := range model.Generate(context.Background(), benchPrompt.prompt, inference.WithMaxTokens(maxTokens)) {
			generated++
			if outputPath != "" {
				generatedText.WriteString(token.Text)
			}
		}
		if err := model.Err(); err != nil {
			b.Fatalf("Generate: %v", err)
		}
		if outputPath != "" {
			lastOutput = generatedText.String()
		}
		totalTokens += generated
	}
	elapsed := time.Since(start)
	b.StopTimer()
	if outputPath != "" {
		if err := os.WriteFile(outputPath, []byte(lastOutput), 0644); err != nil {
			b.Fatalf("write GO_ROCM_BENCH_OUTPUT_FILE=%q: %v", outputPath, err)
		}
	}
	var tokPerSec float64
	if elapsed > 0 {
		tokPerSec = float64(totalTokens) / elapsed.Seconds()
		b.ReportMetric(tokPerSec, "tok/s")
		if benchPrompt.promptTokens > 0 {
			promptTokens := benchPrompt.promptTokens * b.N
			b.ReportMetric(float64(promptTokens)/elapsed.Seconds(), "prompt_tok/s")
			b.ReportMetric(float64(promptTokens+totalTokens)/elapsed.Seconds(), "total_tok/s")
		}
	}
	b.ReportMetric(float64(totalTokens), "tokens")
	b.ReportMetric(float64(maxTokens), "max_tokens/op")
	b.ReportMetric(float64(contextLen), "context_len")
	b.ReportMetric(float64(prefillUBatchTokens), "prefill_ubatch_tokens")
	if benchPrompt.promptTokens > 0 {
		b.ReportMetric(float64(benchPrompt.promptTokens), "prompt_tokens/op")
	}
	inferenceBenchmarkFailBelowMetric(b, "GO_ROCM_BENCH_MIN_TOK_PER_SEC", "tok/s", tokPerSec)
	if benchPrompt.promptTokens > 0 && elapsed > 0 {
		promptTokPerSec := float64(benchPrompt.promptTokens*b.N) / elapsed.Seconds()
		inferenceBenchmarkFailBelowMetric(b, "GO_ROCM_BENCH_MIN_PROMPT_TOK_PER_SEC", "prompt_tok/s", promptTokPerSec)
	}
}

func inferenceBenchmarkLoadGemma4Q4Model(b *testing.B, contextLen, layerCount int) (inference.TextModel, *hipLoadedModel, hipGemma4Q4ForwardConfig) {
	model, loaded, cfg, _ := inferenceBenchmarkLoadGemma4Q4ModelWithKernelCounter(b, contextLen, layerCount)
	return model, loaded, cfg
}

func inferenceBenchmarkLoadGemma4Q4ModelWithKernelCounter(b *testing.B, contextLen, layerCount int) (inference.TextModel, *hipLoadedModel, hipGemma4Q4ForwardConfig, *inferenceBenchmarkHIPKernelCountingDriver) {
	b.Helper()
	modelPath := os.Getenv("GO_ROCM_MODEL_PATH")
	if modelPath == "" {
		b.Skip("set GO_ROCM_MODEL_PATH to a local Gemma4 q4 model pack")
	}
	nativeRuntime, kernelCounter := inferenceBenchmarkNativeRuntimeAndKernelCounter()
	model, err := newROCmBackendWithRuntime(nativeRuntime).LoadModel(modelPath, inference.WithContextLen(contextLen))
	if err != nil {
		b.Fatalf("LoadModel(%q): %v", modelPath, err)
	}
	rocmLoaded, ok := model.(*rocmModel)
	if !ok {
		_ = model.Close()
		b.Fatalf("LoadModel(%q) returned %T, want *rocmModel", modelPath, model)
	}
	loaded, ok := rocmLoaded.native.(*hipLoadedModel)
	if !ok {
		_ = model.Close()
		b.Fatalf("LoadModel(%q) native returned %T, want *hipLoadedModel", modelPath, rocmLoaded.native)
	}
	if layerCount <= 0 {
		layerCount = loaded.modelInfo.NumLayers
	}
	cfg, err := loaded.loadedGemma4Q4ForwardConfig(layerCount)
	if err != nil {
		_ = model.Close()
		b.Fatalf("loadedGemma4Q4ForwardConfig(%d): %v", layerCount, err)
	}
	return model, loaded, cfg, kernelCounter
}

func inferenceBenchmarkCloseModel(b *testing.B, model inference.TextModel) {
	b.Helper()
	if model == nil || os.Getenv("GO_ROCM_BENCH_SKIP_MODEL_CLOSE") == "1" {
		return
	}
	if err := model.Close(); err != nil {
		b.Fatalf("close benchmark model: %v", err)
	}
}

func inferenceBenchmarkPromptTokenSlice(count int, ids []int) []int32 {
	if count <= 0 || len(ids) == 0 {
		return nil
	}
	tokens := make([]int32, count)
	for index := range tokens {
		tokens[index] = int32(ids[index%len(ids)])
	}
	return tokens
}

func inferenceBenchmarkReportPrefillGraph(b *testing.B, tokenCount, layerCount int) {
	b.Helper()
	b.ReportMetric(float64(tokenCount), "prefill_tokens/op")
	b.ReportMetric(float64(layerCount), "prefill_layers/op")
	b.ReportMetric(float64(tokenCount*layerCount), "prefill_token_layers/op")
}

func inferenceBenchmarkGemma4Q4PrefillHidden(b *testing.B, ctx context.Context, driver nativeHIPDriver, layer hipGemma4Q4Layer0Config, tokens []int32) *hipDeviceByteBuffer {
	b.Helper()
	hidden, err := hipRunGemma4Q4PrefillEmbeddingBatch(ctx, driver, layer, tokens)
	if err != nil {
		b.Fatalf("hipRunGemma4Q4PrefillEmbeddingBatch: %v", err)
	}
	b.Cleanup(func() {
		_ = hidden.Close()
	})
	return hidden
}

func inferenceBenchmarkGemma4Q4InputNorm(b *testing.B, ctx context.Context, driver nativeHIPDriver, layer hipGemma4Q4Layer0Config, hidden *hipDeviceByteBuffer, tokenCount int, epsilon float32) *hipDeviceByteBuffer {
	b.Helper()
	inputNorm, err := hipRunGemma4Q4PrefillInputNormBatch(ctx, driver, layer, hidden, tokenCount)
	if err != nil {
		b.Fatalf("hipRunGemma4Q4PrefillInputNormBatch: %v", err)
	}
	b.Cleanup(func() {
		_ = inputNorm.Close()
	})
	return inputNorm
}

func inferenceBenchmarkGemma4Q4QKV(b *testing.B, ctx context.Context, driver nativeHIPDriver, layer hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int) *hipGemma4Q4PrefillQKVBatch {
	b.Helper()
	qkv, err := hipRunGemma4Q4PrefillQKVProjectionBatch(ctx, driver, layer, input, tokenCount)
	if err != nil {
		b.Fatalf("hipRunGemma4Q4PrefillQKVProjectionBatch: %v", err)
	}
	b.Cleanup(func() {
		_ = qkv.Close()
	})
	return qkv
}

func inferenceBenchmarkGemma4Q4QKNormRoPE(b *testing.B, ctx context.Context, driver nativeHIPDriver, layer hipGemma4Q4Layer0Config, qkv *hipGemma4Q4PrefillQKVBatch, tokenCount, startPosition int, epsilon float32) *hipGemma4Q4PrefillRoPEQKBatch {
	b.Helper()
	qk, err := hipRunGemma4Q4PrefillQKNormRoPEBatch(ctx, driver, layer, qkv, tokenCount, startPosition, epsilon)
	if err != nil {
		b.Fatalf("hipRunGemma4Q4PrefillQKNormRoPEBatch: %v", err)
	}
	b.Cleanup(func() {
		_ = qk.Close()
	})
	return qk
}

func inferenceBenchmarkGemma4Q4ValueNorm(b *testing.B, ctx context.Context, driver nativeHIPDriver, layer hipGemma4Q4Layer0Config, qkv *hipGemma4Q4PrefillQKVBatch, tokenCount int, epsilon float32) *hipDeviceByteBuffer {
	b.Helper()
	value, err := hipRunGemma4Q4PrefillValueNormBatch(ctx, driver, layer, qkv, tokenCount, epsilon)
	if err != nil {
		b.Fatalf("hipRunGemma4Q4PrefillValueNormBatch: %v", err)
	}
	b.Cleanup(func() {
		_ = value.Close()
	})
	return value
}

func inferenceBenchmarkGemma4Q4LayerKV(b *testing.B, ctx context.Context, driver nativeHIPDriver, layer hipGemma4Q4Layer0Config, hidden *hipDeviceByteBuffer, tokenCount, startPosition int, epsilon float32) *hipGemma4Q4PrefillLayerKVBatch {
	b.Helper()
	layerKV, err := hipRunGemma4Q4PrefillLayerKVBatch(ctx, driver, layer, hidden, tokenCount, startPosition, epsilon, rocmKVCacheModeKQ8VQ4)
	if err != nil {
		b.Fatalf("hipRunGemma4Q4PrefillLayerKVBatch: %v", err)
	}
	b.Cleanup(func() {
		_ = layerKV.Close()
	})
	return layerKV
}

func inferenceBenchmarkGemma4Q4PerLayerInput(b *testing.B, ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, hidden *hipDeviceByteBuffer, tokens []int32, layerIndex int, epsilon float32) *hipDeviceByteBuffer {
	b.Helper()
	if layerIndex < 0 || layerIndex >= len(cfg.Layers) || !cfg.Layers[layerIndex].PerLayerInput.hasLayerApply() || !cfg.Layers[0].PerLayerInput.hasGlobalPrecompute() {
		return nil
	}
	set, err := hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatch(ctx, driver, cfg, tokens, hidden, epsilon)
	if err != nil {
		b.Fatalf("hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatch: %v", err)
	}
	b.Cleanup(func() {
		_ = set.Close()
	})
	if layerIndex >= set.LayerCount() {
		b.Fatalf("per-layer input set has %d layers, want index %d", set.LayerCount(), layerIndex)
	}
	return set.Layer(layerIndex)
}

func inferenceBenchmarkGemma4Q4ForwardPrior(b *testing.B, ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, epsilon float32) []*rocmDeviceKVCache {
	b.Helper()
	forward, err := hipRunGemma4Q4PrefillForwardBatchWithPrior(ctx, driver, cfg, tokens, 0, epsilon, rocmKVCacheModeKQ8VQ4, nil, nil, nil, nil)
	if err != nil {
		b.Fatalf("hipRunGemma4Q4PrefillForwardBatchWithPrior(prior setup): %v", err)
	}
	b.Cleanup(func() {
		_ = forward.Close()
	})
	prior := make([]*rocmDeviceKVCache, len(forward.Layers))
	for index := range forward.Layers {
		if forward.Layers[index].KV == nil || forward.Layers[index].KV.DeviceKV == nil || forward.Layers[index].KV.DeviceKV.Cache == nil {
			b.Fatalf("prior layer %d device KV is missing", index)
		}
		prior[index] = forward.Layers[index].KV.DeviceKV.Cache
	}
	return prior
}

type inferenceBenchmarkPrompt struct {
	prompt       string
	promptTokens int
	source       string
}

func inferenceBenchmarkPositiveEnv(name string, fallback int) (int, error) {
	if fallback <= 0 {
		return 0, fmt.Errorf("%s fallback=%d, want positive integer", name, fallback)
	}
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return 0, fmt.Errorf("%s=%q, want positive integer", name, value)
	}
	return parsed, nil
}

func inferenceBenchmarkOptionalPositiveEnv(name string) (int, bool, error) {
	value := os.Getenv(name)
	if value == "" {
		return 0, false, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return 0, true, fmt.Errorf("%s=%q, want positive integer", name, value)
	}
	return parsed, true, nil
}

func inferenceBenchmarkOptionalPositiveFloatEnv(name string) (float64, bool, error) {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return 0, false, nil
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil || parsed <= 0 {
		return 0, true, fmt.Errorf("%s=%q, want positive float", name, value)
	}
	return parsed, true, nil
}

func inferenceBenchmarkOptionalNonNegativeEnv(name string) (int, bool, error) {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return 0, false, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 0 {
		return 0, true, fmt.Errorf("%s=%q, want non-negative integer", name, value)
	}
	return parsed, true, nil
}

func inferenceBenchmarkLadderTokensEnv() ([]int, error) {
	value := strings.TrimSpace(os.Getenv("GO_ROCM_BENCH_LADDER_TOKENS"))
	if value == "" {
		return []int{1, 8, 64, 512, 2000}, nil
	}
	parts := strings.Split(value, ",")
	tokens := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil, fmt.Errorf("GO_ROCM_BENCH_LADDER_TOKENS contains an empty token count")
		}
		count, err := strconv.Atoi(part)
		if err != nil || count <= 0 {
			return nil, fmt.Errorf("GO_ROCM_BENCH_LADDER_TOKENS token count %q, want positive integer", part)
		}
		tokens = append(tokens, count)
	}
	return tokens, nil
}

func inferenceBenchmarkPrefillUBatchLadderEnv() ([]int, error) {
	value := strings.TrimSpace(os.Getenv("GO_ROCM_BENCH_PREFILL_UBATCH_LADDER"))
	if value == "" {
		return []int{1024, 512, 256, 128, 64, 32, 16, 8}, nil
	}
	parts := strings.Split(value, ",")
	sizes := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil, fmt.Errorf("GO_ROCM_BENCH_PREFILL_UBATCH_LADDER contains an empty ubatch size")
		}
		size, err := strconv.Atoi(part)
		if err != nil || size <= 0 {
			return nil, fmt.Errorf("GO_ROCM_BENCH_PREFILL_UBATCH_LADDER ubatch size %q, want positive integer", part)
		}
		sizes = append(sizes, size)
	}
	return sizes, nil
}

func inferenceBenchmarkFailBelowMetric(b *testing.B, envName, metricName string, got float64) {
	b.Helper()
	minimum, ok, err := inferenceBenchmarkOptionalPositiveFloatEnv(envName)
	if err != nil {
		b.Fatal(err)
	}
	if ok && got < minimum {
		b.Fatalf("%s %.3f below %s=%0.3f", metricName, got, envName, minimum)
	}
}

func inferenceBenchmarkBookPrefillUBatchTokens(b *testing.B) int {
	b.Helper()
	if value, ok, err := inferenceBenchmarkOptionalPositiveEnv("GO_ROCM_BOOK_PREFILL_UBATCH_TOKENS"); err != nil {
		b.Fatal(err)
	} else if ok {
		b.Setenv(hipGemma4Q4PrefillUBatchEnv, strconv.Itoa(value))
		return value
	}
	if value, ok, err := inferenceBenchmarkOptionalPositiveEnv(hipGemma4Q4PrefillUBatchEnv); err != nil {
		b.Fatal(err)
	} else if ok {
		return value
	}
	value, err := hipGemma4Q4PrefillUBatchTokens()
	if err != nil {
		b.Fatal(err)
	}
	return value
}

func inferenceBenchmarkBookChapterTokensEnv(contextLen, turns int) (int, error) {
	value := strings.TrimSpace(os.Getenv("GO_ROCM_BOOK_CHAPTER_TOKENS"))
	if value == "" || value == "0" {
		return inferenceBenchmarkBookFullChapterTokenLimit(contextLen, turns)
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return 0, fmt.Errorf("GO_ROCM_BOOK_CHAPTER_TOKENS=%q, want positive integer or 0 for full chapter safety cap", value)
	}
	return parsed, nil
}

func inferenceBenchmarkBookFullChapterTokenLimit(contextLen, turns int) (int, error) {
	if contextLen <= 0 {
		return 0, fmt.Errorf("book context length must be positive")
	}
	if turns <= 0 {
		return 0, fmt.Errorf("book turns must be positive")
	}
	reserve := 4096
	if contextLen <= reserve {
		reserve = contextLen / 4
	}
	budget := contextLen - reserve
	if budget <= 0 {
		budget = contextLen
	}
	limit := budget / turns
	if limit <= 0 {
		limit = 1
	}
	return limit, nil
}

func inferenceBenchmarkDurationSecondsEnv(name string, fallback time.Duration) (time.Duration, error) {
	if fallback < 0 {
		return 0, fmt.Errorf("%s fallback=%s, want non-negative duration", name, fallback)
	}
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}
	seconds, err := strconv.Atoi(value)
	if err != nil || seconds < 0 {
		return 0, fmt.Errorf("%s=%q, want non-negative seconds", name, value)
	}
	return time.Duration(seconds) * time.Second, nil
}

func inferenceBenchmarkFloatEnv(name string, fallback float32) (float32, error) {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.ParseFloat(value, 32)
	if err != nil {
		return 0, fmt.Errorf("%s=%q, want float", name, value)
	}
	return float32(parsed), nil
}

func inferenceBenchmarkNonNegativeEnv(name string, fallback int) (int, error) {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 0 {
		return 0, fmt.Errorf("%s=%q, want non-negative integer", name, value)
	}
	return parsed, nil
}

func inferenceBenchmarkBookGenerateConfig(maxTokens int) (inference.GenerateConfig, error) {
	if maxTokens <= 0 {
		return inference.GenerateConfig{}, fmt.Errorf("book max tokens must be positive")
	}
	temperature, err := inferenceBenchmarkFloatEnv("GO_ROCM_BOOK_TEMPERATURE", 1.0)
	if err != nil {
		return inference.GenerateConfig{}, err
	}
	topP, err := inferenceBenchmarkFloatEnv("GO_ROCM_BOOK_TOP_P", 0.95)
	if err != nil {
		return inference.GenerateConfig{}, err
	}
	topK, err := inferenceBenchmarkNonNegativeEnv("GO_ROCM_BOOK_TOP_K", 64)
	if err != nil {
		return inference.GenerateConfig{}, err
	}
	repeatPenalty, err := inferenceBenchmarkFloatEnv("GO_ROCM_BOOK_REPEAT_PENALTY", 1.0)
	if err != nil {
		return inference.GenerateConfig{}, err
	}
	return inference.GenerateConfig{
		MaxTokens:     maxTokens,
		Temperature:   temperature,
		TopK:          topK,
		TopP:          topP,
		RepeatPenalty: repeatPenalty,
	}, nil
}

func inferenceBenchmarkBookGenerateOptions(cfg inference.GenerateConfig) []inference.GenerateOption {
	return []inference.GenerateOption{
		inference.WithMaxTokens(cfg.MaxTokens),
		inference.WithTemperature(cfg.Temperature),
		inference.WithTopP(cfg.TopP),
		inference.WithTopK(cfg.TopK),
		inference.WithRepeatPenalty(cfg.RepeatPenalty),
	}
}

func inferenceBenchmarkPromptFromEnv() (inferenceBenchmarkPrompt, error) {
	if prompt := os.Getenv("GO_ROCM_BENCH_PROMPT"); prompt != "" {
		return inferenceBenchmarkPrompt{
			prompt:       prompt,
			promptTokens: inferenceBenchmarkTokenPromptCount(prompt),
			source:       "env",
		}, nil
	}
	if path := os.Getenv("GO_ROCM_BENCH_PROMPT_FILE"); path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			return inferenceBenchmarkPrompt{}, fmt.Errorf("read GO_ROCM_BENCH_PROMPT_FILE=%q: %w", path, err)
		}
		raw := string(data)
		trimmed := strings.TrimSpace(raw)
		if trimmed == "" {
			return inferenceBenchmarkPrompt{}, fmt.Errorf("GO_ROCM_BENCH_PROMPT_FILE=%q is empty", path)
		}
		if inferenceBenchmarkPromptPrefixed(trimmed) {
			return inferenceBenchmarkPrompt{
				prompt:       trimmed,
				promptTokens: inferenceBenchmarkTokenPromptCount(trimmed),
				source:       "file",
			}, nil
		}
		return inferenceBenchmarkPrompt{
			prompt: "text:" + raw,
			source: "file_text",
		}, nil
	}
	if value := os.Getenv("GO_ROCM_BENCH_PROMPT_TOKEN_COUNT"); value != "" {
		count, err := inferenceBenchmarkPositiveEnv("GO_ROCM_BENCH_PROMPT_TOKEN_COUNT", 1)
		if err != nil {
			return inferenceBenchmarkPrompt{}, err
		}
		ids, err := inferenceBenchmarkPromptTokenIDs(os.Getenv("GO_ROCM_BENCH_PROMPT_TOKEN_IDS"))
		if err != nil {
			return inferenceBenchmarkPrompt{}, err
		}
		return inferenceBenchmarkPrompt{
			prompt:       inferenceBenchmarkTokenPrompt(count, ids),
			promptTokens: count,
			source:       "generated_tokens",
		}, nil
	}
	return inferenceBenchmarkPrompt{
		prompt: "text:Hi",
		source: "default",
	}, nil
}

func inferenceBenchmarkPromptPrefixed(prompt string) bool {
	lower := strings.ToLower(strings.TrimSpace(prompt))
	return strings.HasPrefix(lower, "tokens:") || strings.HasPrefix(lower, "text:")
}

func inferenceBenchmarkPromptTokenIDs(raw string) ([]int, error) {
	if strings.TrimSpace(raw) == "" {
		return []int{2, 10979}, nil
	}
	parts := strings.Split(raw, ",")
	ids := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil, fmt.Errorf("GO_ROCM_BENCH_PROMPT_TOKEN_IDS contains an empty token ID")
		}
		id, err := strconv.Atoi(part)
		if err != nil || id < 0 {
			return nil, fmt.Errorf("GO_ROCM_BENCH_PROMPT_TOKEN_IDS token %q, want non-negative integer", part)
		}
		ids = append(ids, id)
	}
	return ids, nil
}

func inferenceBenchmarkTokenPrompt(count int, ids []int) string {
	if count <= 0 || len(ids) == 0 {
		return "tokens:"
	}
	var builder strings.Builder
	builder.Grow(len("tokens:") + count*7)
	builder.WriteString("tokens:")
	for i := 0; i < count; i++ {
		if i > 0 {
			builder.WriteByte(',')
		}
		builder.WriteString(strconv.Itoa(ids[i%len(ids)]))
	}
	return builder.String()
}

func inferenceBenchmarkTokenPromptCount(prompt string) int {
	trimmed := strings.TrimSpace(prompt)
	if !strings.HasPrefix(strings.ToLower(trimmed), "tokens:") {
		return 0
	}
	body := strings.TrimSpace(trimmed[len("tokens:"):])
	if body == "" {
		return 0
	}
	count := 1
	for _, r := range body {
		if r == ',' {
			count++
		}
	}
	return count
}

func TestInferenceBenchmarkBookTurnPrompt_Good(t *testing.T) {
	workload := inferenceBenchmarkBookWorkload()
	chapter1 := inferenceBenchmarkBookTurnPrompt(workload, "", 1)
	if !strings.Contains(chapter1, "C001_STORY_PERSPECTIVE") ||
		!strings.Contains(chapter1, "lighthouse keeper") {
		t.Fatalf("chapter 1 prompt = %q, want seed lighthouse premise", chapter1)
	}
	if strings.Contains(chapter1, "10 chapter") {
		t.Fatalf("chapter 1 prompt = %q, should not declare the final chapter count up front", chapter1)
	}
	chapter2 := inferenceBenchmarkBookTurnPrompt(workload, "## Chapter 1\nThe lighthouse kept watch.", 2)
	if !strings.Contains(chapter2, "C002_POETRY_TIME") ||
		!strings.Contains(chapter2, "Evaluation distractor prompt") ||
		!strings.Contains(chapter2, "Preserve the original lighthouse keeper") ||
		!strings.Contains(chapter2, "setting, characters, objects, form, or premise") ||
		!strings.Contains(chapter2, "continuity anchors alive") {
		t.Fatalf("chapter 2 prompt = %q, want chapter continuation with distractor", chapter2)
	}
	retainedChapter1 := inferenceBenchmarkBookRetainedTurnChatPrompt(workload, 1)
	if !strings.HasPrefix(retainedChapter1, "<bos><|turn>user\n") ||
		!strings.HasSuffix(retainedChapter1, "<turn|>\n<|turn>model\n") ||
		!strings.Contains(retainedChapter1, "C001_STORY_PERSPECTIVE") {
		t.Fatalf("retained chapter 1 chat prompt = %q, want Gemma4 user/model turn", retainedChapter1)
	}
	retainedChapter2 := inferenceBenchmarkBookRetainedTurnChatPrompt(workload, 2)
	if !strings.HasPrefix(retainedChapter2, "<turn|>\n<|turn>user\n") ||
		!strings.HasSuffix(retainedChapter2, "<turn|>\n<|turn>model\n") ||
		strings.Contains(retainedChapter2, "Book so far") ||
		!strings.Contains(retainedChapter2, "continuity anchors alive") {
		t.Fatalf("retained chapter 2 chat prompt = %q, want assistant close plus new user turn only", retainedChapter2)
	}
	if hits := inferenceBenchmarkBookArcAnchorHits("The lighthouse keeper saw the light answer the deep ocean."); hits < 5 {
		t.Fatalf("arc anchor hits = %d, want lighthouse arc anchors", hits)
	}
	t.Setenv("GO_ROCM_BOOK_TEMPERATURE", "")
	t.Setenv("GO_ROCM_BOOK_TOP_P", "")
	t.Setenv("GO_ROCM_BOOK_TOP_K", "")
	t.Setenv("GO_ROCM_BOOK_REPEAT_PENALTY", "")
	cfg, err := inferenceBenchmarkBookGenerateConfig(16)
	if err != nil {
		t.Fatalf("book generate config: %v", err)
	}
	if cfg.MaxTokens != 16 || cfg.Temperature != 1 || cfg.TopP != 0.95 || cfg.TopK != 64 || cfg.RepeatPenalty != 1 {
		t.Fatalf("book generate config = %+v, want go-mlx-style sampling defaults", cfg)
	}
}

func TestInferenceBenchmarkDurationSecondsEnv_Good(t *testing.T) {
	t.Setenv("GO_ROCM_BOOK_TURN_TIMEOUT_SECONDS", "")
	got, err := inferenceBenchmarkDurationSecondsEnv("GO_ROCM_BOOK_TURN_TIMEOUT_SECONDS", 60*time.Second)
	if err != nil || got != 60*time.Second {
		t.Fatalf("default duration = %s, %v; want 60s", got, err)
	}
	t.Setenv("GO_ROCM_BOOK_TURN_TIMEOUT_SECONDS", "0")
	got, err = inferenceBenchmarkDurationSecondsEnv("GO_ROCM_BOOK_TURN_TIMEOUT_SECONDS", 60*time.Second)
	if err != nil || got != 0 {
		t.Fatalf("zero duration = %s, %v; want disabled timeout", got, err)
	}
}

func TestInferenceBenchmarkBookChapterTokensEnv_Good(t *testing.T) {
	t.Setenv("GO_ROCM_BOOK_CHAPTER_TOKENS", "")
	got, err := inferenceBenchmarkBookChapterTokensEnv(48000, 10)
	if err != nil || got != 4390 {
		t.Fatalf("default chapter tokens = %d, %v; want 4390", got, err)
	}

	t.Setenv("GO_ROCM_BOOK_CHAPTER_TOKENS", "0")
	got, err = inferenceBenchmarkBookChapterTokensEnv(131072, 10)
	if err != nil || got != 12697 {
		t.Fatalf("zero chapter tokens = %d, %v; want 12697", got, err)
	}

	t.Setenv("GO_ROCM_BOOK_CHAPTER_TOKENS", "512")
	got, err = inferenceBenchmarkBookChapterTokensEnv(48000, 10)
	if err != nil || got != 512 {
		t.Fatalf("explicit chapter tokens = %d, %v; want 512", got, err)
	}
}

func TestInferenceBenchmarkOptionalPositiveEnv_Good(t *testing.T) {
	t.Setenv("GO_ROCM_BOOK_LAYERS", "")
	got, ok, err := inferenceBenchmarkOptionalPositiveEnv("GO_ROCM_BOOK_LAYERS")
	if err != nil || ok || got != 0 {
		t.Fatalf("empty optional positive = %d, %t, %v; want unset", got, ok, err)
	}
	t.Setenv("GO_ROCM_BOOK_LAYERS", "2")
	got, ok, err = inferenceBenchmarkOptionalPositiveEnv("GO_ROCM_BOOK_LAYERS")
	if err != nil || !ok || got != 2 {
		t.Fatalf("set optional positive = %d, %t, %v; want 2", got, ok, err)
	}
}

func TestInferenceBenchmarkLadderTokensEnv_Good(t *testing.T) {
	t.Setenv("GO_ROCM_BENCH_LADDER_TOKENS", "")
	got, err := inferenceBenchmarkLadderTokensEnv()
	if err != nil || fmt.Sprint(got) != "[1 8 64 512 2000]" {
		t.Fatalf("default ladder tokens = %v, %v; want 1/8/64/512/2000", got, err)
	}

	t.Setenv("GO_ROCM_BENCH_LADDER_TOKENS", "1, 2048")
	got, err = inferenceBenchmarkLadderTokensEnv()
	if err != nil || fmt.Sprint(got) != "[1 2048]" {
		t.Fatalf("custom ladder tokens = %v, %v; want [1 2048]", got, err)
	}

	t.Setenv("GO_ROCM_BENCH_LADDER_TOKENS", "1,,8")
	if _, err = inferenceBenchmarkLadderTokensEnv(); err == nil {
		t.Fatal("empty ladder token count error = nil")
	}
}

func TestInferenceBenchmarkPrefillUBatchLadderEnv_Good(t *testing.T) {
	t.Setenv("GO_ROCM_BENCH_PREFILL_UBATCH_LADDER", "")
	got, err := inferenceBenchmarkPrefillUBatchLadderEnv()
	if err != nil || fmt.Sprint(got) != "[1024 512 256 128 64 32 16 8]" {
		t.Fatalf("default prefill ubatch ladder = %v, %v; want 1024..8", got, err)
	}

	t.Setenv("GO_ROCM_BENCH_PREFILL_UBATCH_LADDER", "64, 16")
	got, err = inferenceBenchmarkPrefillUBatchLadderEnv()
	if err != nil || fmt.Sprint(got) != "[64 16]" {
		t.Fatalf("custom prefill ubatch ladder = %v, %v; want [64 16]", got, err)
	}

	t.Setenv("GO_ROCM_BENCH_PREFILL_UBATCH_LADDER", "64,,16")
	if _, err = inferenceBenchmarkPrefillUBatchLadderEnv(); err == nil {
		t.Fatal("empty prefill ubatch size error = nil")
	}
}

func TestInferenceBenchmarkOptionalPositiveFloatEnv_Good(t *testing.T) {
	t.Setenv("GO_ROCM_BENCH_MIN_TOK_PER_SEC", "")
	got, ok, err := inferenceBenchmarkOptionalPositiveFloatEnv("GO_ROCM_BENCH_MIN_TOK_PER_SEC")
	if err != nil || ok || got != 0 {
		t.Fatalf("empty optional positive float = %f, %t, %v; want unset", got, ok, err)
	}

	t.Setenv("GO_ROCM_BENCH_MIN_TOK_PER_SEC", "100.5")
	got, ok, err = inferenceBenchmarkOptionalPositiveFloatEnv("GO_ROCM_BENCH_MIN_TOK_PER_SEC")
	if err != nil || !ok || got != 100.5 {
		t.Fatalf("set optional positive float = %f, %t, %v; want 100.5", got, ok, err)
	}

	t.Setenv("GO_ROCM_BENCH_MIN_TOK_PER_SEC", "0")
	if _, _, err = inferenceBenchmarkOptionalPositiveFloatEnv("GO_ROCM_BENCH_MIN_TOK_PER_SEC"); err == nil {
		t.Fatal("zero optional positive float error = nil")
	}
}

func TestInferenceBenchmarkOptionalNonNegativeEnv_Good(t *testing.T) {
	t.Setenv("GO_ROCM_BOOK_MAX_MAXED_TURNS", "")
	got, ok, err := inferenceBenchmarkOptionalNonNegativeEnv("GO_ROCM_BOOK_MAX_MAXED_TURNS")
	if err != nil || ok || got != 0 {
		t.Fatalf("empty optional non-negative = %d, %t, %v; want unset", got, ok, err)
	}

	t.Setenv("GO_ROCM_BOOK_MAX_MAXED_TURNS", "0")
	got, ok, err = inferenceBenchmarkOptionalNonNegativeEnv("GO_ROCM_BOOK_MAX_MAXED_TURNS")
	if err != nil || !ok || got != 0 {
		t.Fatalf("zero optional non-negative = %d, %t, %v; want 0", got, ok, err)
	}

	t.Setenv("GO_ROCM_BOOK_MAX_MAXED_TURNS", "-1")
	if _, _, err = inferenceBenchmarkOptionalNonNegativeEnv("GO_ROCM_BOOK_MAX_MAXED_TURNS"); err == nil {
		t.Fatal("negative optional non-negative error = nil")
	}
}

func TestInferenceBenchmarkBookThresholdHelpers_Good(t *testing.T) {
	run := inferenceBenchmarkBookRun{
		TurnStats: []inferenceBenchmarkBookTurnStat{
			{GeneratedTokens: 2, Decode: time.Second, HitMaxTokens: true},
			{GeneratedTokens: 4, Decode: 2 * time.Second},
		},
	}
	if got := inferenceBenchmarkBookMaxedTurns(run); got != 1 {
		t.Fatalf("maxed turns = %d, want 1", got)
	}
	if got := inferenceBenchmarkBookLastTurnTokS(run); got != 2 {
		t.Fatalf("last turn tok/s = %f, want 2", got)
	}
}

func TestInferenceBenchmarkBookRepetitionStats_Good(t *testing.T) {
	repeatedChapter := "The light kept the keeper at the black reef. The deep ocean answered with a slow signal."
	repeated, similarity := inferenceBenchmarkBookRepetitionStats([]string{
		"Silas climbs the tower and hears the first signal beneath the storm.",
		repeatedChapter,
		repeatedChapter,
	})
	if repeated != 1 {
		t.Fatalf("repeated turns = %d, want 1", repeated)
	}
	if similarity < inferenceBenchmarkBookRepeatSimilarityThreshold {
		t.Fatalf("max adjacent repeat = %f, want at least threshold %f", similarity, inferenceBenchmarkBookRepeatSimilarityThreshold)
	}

	repeated, similarity = inferenceBenchmarkBookRepetitionStats([]string{
		"The keeper repairs the lens while gulls vanish into a red dawn.",
		"The light remembers a century of storms and counts every lost ship.",
		"The ocean below answers in pressure, salt, and patient geometry.",
	})
	if repeated != 0 {
		t.Fatalf("distinct repeated turns = %d, want 0", repeated)
	}
	if similarity >= inferenceBenchmarkBookRepeatSimilarityThreshold {
		t.Fatalf("distinct max adjacent repeat = %f, want below threshold %f", similarity, inferenceBenchmarkBookRepeatSimilarityThreshold)
	}
}

func TestInferenceBenchmarkPromptFromEnv_Good(t *testing.T) {
	t.Setenv("GO_ROCM_BENCH_PROMPT", "")
	t.Setenv("GO_ROCM_BENCH_PROMPT_FILE", "")
	t.Setenv("GO_ROCM_BENCH_PROMPT_TOKEN_COUNT", "5")
	t.Setenv("GO_ROCM_BENCH_PROMPT_TOKEN_IDS", "2,10979")

	got, err := inferenceBenchmarkPromptFromEnv()
	if err != nil {
		t.Fatalf("inferenceBenchmarkPromptFromEnv: %v", err)
	}
	if got.prompt != "tokens:2,10979,2,10979,2" ||
		got.promptTokens != 5 ||
		got.source != "generated_tokens" {
		t.Fatalf("prompt = %+v, want generated 5-token prompt", got)
	}
}

func TestInferenceBenchmarkPromptFromEnv_BadTokenID(t *testing.T) {
	t.Setenv("GO_ROCM_BENCH_PROMPT", "")
	t.Setenv("GO_ROCM_BENCH_PROMPT_FILE", "")
	t.Setenv("GO_ROCM_BENCH_PROMPT_TOKEN_COUNT", "5")
	t.Setenv("GO_ROCM_BENCH_PROMPT_TOKEN_IDS", "2,,10979")

	if _, err := inferenceBenchmarkPromptFromEnv(); err == nil {
		t.Fatalf("inferenceBenchmarkPromptFromEnv succeeded, want empty token ID error")
	}
}
