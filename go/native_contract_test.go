// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"encoding/binary"
	"iter"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestNativeContract_RocmBackendImplementsSharedPlanner_Good(t *testing.T) {
	var _ inference.ModelFitPlanner = (*rocmBackend)(nil)
}

func TestNativeContract_RocmModelImplementsSharedContracts_Good(t *testing.T) {
	var _ inference.TokenizerModel = (*rocmModel)(nil)
	var _ inference.AdapterModel = (*rocmModel)(nil)
	var _ inference.ProbeableModel = (*rocmModel)(nil)
	var _ inference.BenchableModel = (*rocmModel)(nil)
	var _ inference.Evaluator = (*rocmModel)(nil)
}

func TestNativeContract_LoadModelUsesNativeRuntimeWithoutServer_Good(t *testing.T) {
	runtime := &fakeNativeRuntime{
		available: true,
		model:     &fakeNativeModel{tokens: []inference.Token{{ID: 17, Text: "ok"}}},
	}
	backend := newROCmBackendWithRuntime(runtime)
	t.Setenv("PATH", "")
	t.Setenv("ROCM_LLAMA_SERVER_PATH", "")

	model, err := backend.LoadModel(nativeContractGGUF(t), inference.WithContextLen(8192), inference.WithAdapterPath("adapter.safetensors"))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer model.Close()

	if runtime.loadPath == "" || runtime.loadConfig.ContextSize != 8192 {
		t.Fatalf("native runtime load = path %q config %+v, want direct native load", runtime.loadPath, runtime.loadConfig)
	}
	if runtime.loadConfig.AdapterPath != "adapter.safetensors" {
		t.Fatalf("adapter path = %q, want load-time adapter path forwarded", runtime.loadConfig.AdapterPath)
	}
	if model.ModelType() != "qwen3" {
		t.Fatalf("ModelType = %q, want qwen3", model.ModelType())
	}
}

func TestNativeContract_PlanModelFit_Good(t *testing.T) {
	runtime := &fakeNativeRuntime{device: nativeDeviceInfo{MemoryBytes: 16 * memoryGiB, Name: "gfx1100"}}
	report, err := newROCmBackendWithRuntime(runtime).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture:  "qwen3",
		QuantBits:     4,
		ContextLength: 32768,
		NumLayers:     28,
		HiddenSize:    2048,
	}, 0)
	if err != nil {
		t.Fatalf("PlanModelFit: %v", err)
	}
	if report == nil || !report.Fits || !report.ArchitectureOK || !report.QuantizationOK {
		t.Fatalf("fit report = %+v, want supported fitting qwen3 q4", report)
	}
	if report.MemoryPlan.CacheMode == "" || report.MemoryPlan.KVCacheBytes == 0 {
		t.Fatalf("memory plan = %+v, want cache sizing", report.MemoryPlan)
	}
}

func TestNativeContract_PlanModelFit_Bad(t *testing.T) {
	report, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture: "unknown",
		QuantBits:    16,
	}, 8*memoryGiB)
	if err != nil {
		t.Fatalf("PlanModelFit: %v", err)
	}
	if report == nil || report.ArchitectureOK || report.QuantizationOK || report.Fits {
		t.Fatalf("fit report = %+v, want unsupported model", report)
	}
}

func TestNativeContract_PlanModelFit_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	report, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).PlanModelFit(ctx, inference.ModelIdentity{Architecture: "qwen3"}, 0)
	if err == nil {
		t.Fatalf("PlanModelFit cancelled error = nil, report=%+v", report)
	}
}

func TestNativeContract_ProbeSinkReceivesGeneratedTokens_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &fakeNativeModel{tokens: []inference.Token{{ID: 9, Text: "hi"}}},
	}
	var got inference.ProbeEvent
	model.SetProbeSink(inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		got = event
	}))

	for range model.Generate(context.Background(), "hello") {
	}

	if got.Kind != inference.ProbeEventToken || got.Token == nil || got.Token.ID != 9 || got.Token.Text != "hi" {
		t.Fatalf("probe event = %+v, want generated token event", got)
	}
}

func TestNativeContract_AdapterLifecycle_Good(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{}}
	identity, err := model.LoadAdapter("domain.safetensors")
	if err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}
	if identity.Path != "domain.safetensors" || identity.Format != "lora" {
		t.Fatalf("adapter identity = %+v, want lora path", identity)
	}
	if model.ActiveAdapter().Path != "domain.safetensors" {
		t.Fatalf("active adapter = %+v, want loaded adapter", model.ActiveAdapter())
	}
	if err := model.UnloadAdapter(); err != nil {
		t.Fatalf("UnloadAdapter: %v", err)
	}
	if !adapterIdentityIsZero(model.ActiveAdapter()) {
		t.Fatalf("active adapter after unload = %+v, want zero", model.ActiveAdapter())
	}
}

func TestNativeContract_BenchmarkAndEvaluateUseModelSurface_Ugly(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "a"}, {ID: 2, Text: "b"}}},
	}
	bench, err := model.Benchmark(context.Background(), inference.BenchConfig{Prompts: []string{"hi"}, MaxTokens: 2, MeasuredRuns: 1})
	if err != nil {
		t.Fatalf("Benchmark: %v", err)
	}
	if bench.GeneratedTokens != 2 || bench.DecodeTokensPerSec == 0 {
		t.Fatalf("bench = %+v, want generated token throughput", bench)
	}

	eval, err := model.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{Text: "hello world"}}, inference.EvalConfig{MaxSamples: 1})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if eval.Metrics.Samples != 1 || eval.Metrics.Tokens == 0 {
		t.Fatalf("eval = %+v, want token counts", eval)
	}
}

type fakeNativeRuntime struct {
	available  bool
	device     nativeDeviceInfo
	model      nativeModel
	loadPath   string
	loadConfig nativeLoadConfig
}

func (runtime *fakeNativeRuntime) Available() bool { return runtime.available }
func (runtime *fakeNativeRuntime) DeviceInfo() nativeDeviceInfo {
	return runtime.device
}
func (runtime *fakeNativeRuntime) LoadModel(path string, cfg nativeLoadConfig) (nativeModel, error) {
	runtime.loadPath = path
	runtime.loadConfig = cfg
	if runtime.model == nil {
		runtime.model = &fakeNativeModel{}
	}
	return runtime.model, nil
}

type fakeNativeModel struct {
	tokens  []inference.Token
	adapter inference.AdapterIdentity
}

func (model *fakeNativeModel) Generate(_ context.Context, _ string, _ inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	return func(yield func(inference.Token) bool) {
		for _, token := range model.tokens {
			if !yield(token) {
				return
			}
		}
	}, func() error { return nil }
}
func (model *fakeNativeModel) Chat(ctx context.Context, _ []inference.Message, cfg inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	return model.Generate(ctx, "", cfg)
}
func (model *fakeNativeModel) Classify(_ context.Context, prompts []string, _ inference.GenerateConfig) ([]inference.ClassifyResult, error) {
	out := make([]inference.ClassifyResult, len(prompts))
	for i := range prompts {
		out[i] = inference.ClassifyResult{Token: inference.Token{ID: int32(i + 1), Text: "ok"}}
	}
	return out, nil
}
func (model *fakeNativeModel) BatchGenerate(_ context.Context, prompts []string, _ inference.GenerateConfig) ([]inference.BatchResult, error) {
	out := make([]inference.BatchResult, len(prompts))
	for i := range prompts {
		out[i] = inference.BatchResult{Tokens: append([]inference.Token(nil), model.tokens...)}
	}
	return out, nil
}
func (model *fakeNativeModel) Encode(text string) []int32 {
	if core.Trim(text) == "" {
		return nil
	}
	parts := core.Split(core.Trim(text), " ")
	ids := make([]int32, len(parts))
	for i := range parts {
		ids[i] = int32(i + 1)
	}
	return ids
}
func (model *fakeNativeModel) Decode(ids []int32) string {
	return core.Sprintf("%d tokens", len(ids))
}
func (model *fakeNativeModel) ApplyChatTemplate(messages []inference.Message) (string, error) {
	var text string
	for _, message := range messages {
		text += message.Role + ":" + message.Content + "\n"
	}
	return text, nil
}
func (model *fakeNativeModel) LoadAdapter(path string) (inference.AdapterIdentity, error) {
	model.adapter = inference.AdapterIdentity{Path: path, Format: "lora"}
	return model.adapter, nil
}
func (model *fakeNativeModel) UnloadAdapter() error {
	model.adapter = inference.AdapterIdentity{}
	return nil
}
func (model *fakeNativeModel) ActiveAdapter() inference.AdapterIdentity { return model.adapter }
func (model *fakeNativeModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{GeneratedTokens: len(model.tokens), DecodeDuration: time.Millisecond}
}
func (model *fakeNativeModel) Close() error { return nil }

type singleInferenceSample struct {
	sample inference.DatasetSample
	done   bool
}

func (stream *singleInferenceSample) Next() (inference.DatasetSample, bool, error) {
	if stream.done {
		return inference.DatasetSample{}, false, nil
	}
	stream.done = true
	return stream.sample, true, nil
}

func nativeContractGGUF(t *testing.T) string {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "native-contract.gguf")
	buf := core.NewBuffer()
	writeUint32 := func(v uint32) { core.RequireNoError(t, binary.Write(buf, binary.LittleEndian, v)) }
	writeUint64 := func(v uint64) { core.RequireNoError(t, binary.Write(buf, binary.LittleEndian, v)) }
	writeString := func(v string) {
		writeUint64(uint64(len(v)))
		_, err := buf.Write([]byte(v))
		core.RequireNoError(t, err)
	}
	writeKVString := func(key, value string) {
		writeString(key)
		writeUint32(8)
		writeString(value)
	}
	writeKVUint32 := func(key string, value uint32) {
		writeString(key)
		writeUint32(4)
		writeUint32(value)
	}

	writeUint32(0x46554747)
	writeUint32(3)
	writeUint64(0)
	writeUint64(6)
	writeKVString("general.architecture", "qwen3")
	writeKVString("general.name", "native-test")
	writeKVString("general.size_label", "0B")
	writeKVUint32("general.file_type", 15)
	writeKVUint32("qwen3.context_length", 32768)
	writeKVUint32("qwen3.block_count", 28)

	result := core.WriteFile(path, buf.Bytes(), 0o644)
	core.RequireTrue(t, result.OK)
	return path
}
