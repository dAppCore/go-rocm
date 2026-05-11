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
	var _ inference.CapabilityReporter = (*rocmBackend)(nil)
	var _ inference.ModelPackInspector = (*rocmBackend)(nil)
}

func TestNativeContract_RocmModelImplementsSharedContracts_Good(t *testing.T) {
	var _ inference.TokenizerModel = (*rocmModel)(nil)
	var _ inference.AdapterModel = (*rocmModel)(nil)
	var _ inference.ProbeableModel = (*rocmModel)(nil)
	var _ inference.BenchableModel = (*rocmModel)(nil)
	var _ inference.Evaluator = (*rocmModel)(nil)
	var _ inference.CapabilityReporter = (*rocmModel)(nil)
}

func TestNativeContract_RocmBackendCapabilities_Good(t *testing.T) {
	runtime := &fakeNativeRuntime{
		available: true,
		device:    nativeDeviceInfo{Name: "gfx1100", MemoryBytes: 16 * memoryGiB, FreeBytes: 8 * memoryGiB, Driver: "hip-test"},
	}

	report := newROCmBackendWithRuntime(runtime).Capabilities()

	if report.Runtime.Backend != "rocm" || !report.Runtime.NativeRuntime || report.Runtime.Device != "gfx1100" {
		t.Fatalf("runtime = %+v, want native ROCm device", report.Runtime)
	}
	if !report.Available {
		t.Fatalf("Available = false, want true")
	}
	if !report.Supports(inference.CapabilityModelLoad) || !report.Supports(inference.CapabilityModelFit) {
		t.Fatalf("capabilities = %+v, want load and fit planning", report.CapabilityIDs())
	}
	if report.Supports(inference.CapabilityGenerate) {
		t.Fatalf("generate should be planned until native decode kernels are linked: %+v", report.CapabilityIDs())
	}
	if !report.Supports(inference.CapabilityTokenizer) || !report.Supports(inference.CapabilityProbeEvents) {
		t.Fatalf("capabilities = %+v, want fallback tokenizer and probe stream", report.CapabilityIDs())
	}
	if cap, ok := report.Capability(inference.CapabilityJANGTQ); !ok || cap.Status != inference.CapabilityStatusPlanned {
		t.Fatalf("JANGTQ capability = %+v ok=%v, want planned groundwork", cap, ok)
	}
	if cap, ok := report.Capability(inference.CapabilityScheduler); !ok || cap.Status != inference.CapabilityStatusPlanned {
		t.Fatalf("scheduler capability = %+v ok=%v, want planned native scheduler", cap, ok)
	}
	for _, id := range []inference.CapabilityID{
		inference.CapabilityRequestCancel,
		inference.CapabilityCacheBlocks,
		inference.CapabilityCacheWarm,
		inference.CapabilityCacheDisk,
		inference.CapabilityReasoningParse,
		inference.CapabilityToolParse,
		inference.CapabilitySpeculativeDecode,
		inference.CapabilityPromptLookupDecode,
		inference.CapabilityMoERouting,
		inference.CapabilityMoELazyExperts,
		inference.CapabilityJANGTQ,
		inference.CapabilityCodebookVQ,
		inference.CapabilityEmbeddings,
		inference.CapabilityRerank,
	} {
		if _, ok := report.Capability(id); !ok {
			t.Fatalf("capability %q missing from ROCm report: %+v", id, report.CapabilityIDs())
		}
	}
	if len(report.Architectures) == 0 || len(report.Quantizations) == 0 || len(report.CacheModes) == 0 {
		t.Fatalf("report = %+v, want architecture/quant/cache metadata", report)
	}
}

func TestNativeContract_RocmModelCapabilities_Ugly(t *testing.T) {
	model := &rocmModel{
		modelType: "qwen3",
		modelInfo: inference.ModelInfo{Architecture: "qwen3", NumLayers: 28, QuantBits: 4},
		native:    &fakeNativeModel{adapter: inference.AdapterIdentity{Path: "domain.safetensors", Format: "lora"}},
	}

	report := model.Capabilities()

	if !report.Available || report.Model.Architecture != "qwen3" || report.Adapter.Path != "domain.safetensors" {
		t.Fatalf("report = %+v, want loaded model and adapter identity", report)
	}
	if report.Supports(inference.CapabilityLoRAInference) {
		t.Fatalf("LoRA inference should be planned until HIP adapter application is linked")
	}
	if !report.Supports(inference.CapabilityEvaluation) {
		t.Fatalf("evaluation should be experimentally available: %+v", report.CapabilityIDs())
	}
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

func TestNativeContract_PlanModelFit_Rocm16GBMoELazyExperts_Good(t *testing.T) {
	runtime := &fakeNativeRuntime{device: nativeDeviceInfo{MemoryBytes: 16 * memoryGiB, Name: "gfx1100"}}
	report, err := newROCmBackendWithRuntime(runtime).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture:  "Qwen3MoeForCausalLM",
		QuantBits:     2,
		QuantType:     "jangtq",
		QuantGroup:    64,
		ContextLength: 32768,
		NumLayers:     24,
		HiddenSize:    2048,
	}, 0)
	if err != nil {
		t.Fatalf("PlanModelFit: %v", err)
	}
	if report == nil || !report.Fits || report.MemoryPlan.MachineClass != "rocm-16gb" {
		t.Fatalf("fit report = %+v, want fitting ROCm 16GB MoE plan", report)
	}
	if report.MemoryPlan.CacheMode != "k-q8-v-q4" || report.MemoryPlan.Labels["moe_lazy_experts"] != "true" || report.MemoryPlan.Labels["prefill_chunk_tokens"] != "512" {
		t.Fatalf("memory plan = %+v, want compact KV, lazy experts, and chunked prefill", report.MemoryPlan)
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
	if bench.Labels["scheduler"] != "planned" || bench.Labels["cache.blocks"] != "planned" || bench.Labels["probe.events"] != "stream_tokens" {
		t.Fatalf("bench labels = %+v, want ROCm parity probe/cache/scheduler fields", bench.Labels)
	}

	eval, err := model.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{Text: "hello world"}}, inference.EvalConfig{MaxSamples: 1})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if eval.Metrics.Samples != 1 || eval.Metrics.Tokens == 0 {
		t.Fatalf("eval = %+v, want token counts", eval)
	}
}

func TestNativeContract_ModelPackInspectorReadsSidecars_Good(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type":"MiniMaxM2ForCausalLM",
		"hidden_size":2048,
		"num_hidden_layers":24,
		"vocab_size":32000,
		"max_position_embeddings":32768,
		"num_local_experts":32,
		"num_experts_per_tok":2,
		"quantization_config":{"quant_method":"jangtq","bits":2,"group_size":64,"weight_format":"mxtq"}
	}`)
	writeNativeContractSafetensors(t, core.PathJoin(dir, "model.safetensors"))
	writeNativeContractFile(t, core.PathJoin(dir, "jang_config.json"), `{
		"version":1,
		"weight_format":"mxtq",
		"profile":"JANGTQ",
		"source_model":{"name":"MiniMax-M2.7","org":"dealignai","architecture":"MiniMaxM2ForCausalLM"},
		"mxtq_bits":{"attention":8,"shared_expert":4,"routed_expert":2,"embed_tokens":8,"lm_head":8},
		"quantization":{"method":"affine+mxtq","group_size":64,"bits_default":2},
		"capabilities":{"reasoning_parser":"minimax","tool_parser":"json","supports_tools":true,"supports_thinking":true,"cache_type":"block-prefix"}
	}`)
	writeNativeContractFile(t, core.PathJoin(dir, "codebook_config.json"), `{
		"type":"codebook",
		"format":"vq",
		"codebook_size":16,
		"code_dim":2,
		"index_bits":8,
		"tensors":[{"name":"model.layers.0.mlp.down_proj.weight","shape":[2,4],"codes":"codes","codebook":"table"}]
	}`)

	backend := newROCmBackendWithRuntime(&fakeNativeRuntime{device: nativeDeviceInfo{MemoryBytes: 16 * memoryGiB, Name: "gfx1100"}})
	inspection, err := backend.InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !inspection.Supported || inspection.Format != "safetensors" || inspection.Model.Architecture != "minimax_m2" || inspection.Model.QuantType != "jangtq" {
		t.Fatalf("inspection = %+v, want supported MiniMax/JANGTQ safetensors pack", inspection)
	}
	if inspection.Labels["memory_plan_machine_class"] != "rocm-16gb" || inspection.Labels["memory_plan_moe_lazy_experts"] != "true" || inspection.Labels["codebook_format"] != "vq" {
		t.Fatalf("inspection labels = %+v, want memory fit and codebook metadata", inspection.Labels)
	}
	if !nativeInspectionHasCapability(inspection, inference.CapabilityJANGTQ) || !nativeInspectionHasCapability(inspection, inference.CapabilityCodebookVQ) || !nativeInspectionHasCapability(inspection, inference.CapabilityMoELazyExperts) {
		t.Fatalf("inspection capabilities = %+v, want JANGTQ/codebook/MoE metadata", inspection.Capabilities)
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

func writeNativeContractFile(t *testing.T, path, content string) {
	t.Helper()
	result := core.WriteFile(path, []byte(content), 0o644)
	core.RequireTrue(t, result.OK)
}

func writeNativeContractSafetensors(t *testing.T, path string) {
	t.Helper()
	header := []byte(`{"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[0,16]},"__metadata__":{"format":"pt"}}`)
	buf := core.NewBuffer()
	core.RequireNoError(t, binary.Write(buf, binary.LittleEndian, uint64(len(header))))
	_, err := buf.Write(header)
	core.RequireNoError(t, err)
	_, err = buf.Write(make([]byte, 16))
	core.RequireNoError(t, err)
	result := core.WriteFile(path, buf.Bytes(), 0o644)
	core.RequireTrue(t, result.OK)
}

func nativeInspectionHasCapability(inspection *inference.ModelPackInspection, id inference.CapabilityID) bool {
	if inspection == nil {
		return false
	}
	for _, capability := range inspection.Capabilities {
		if capability.ID == id {
			return true
		}
	}
	return false
}
