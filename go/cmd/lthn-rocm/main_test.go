// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"iter"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
	rocm "dappco.re/go/rocm"
	"dappco.re/go/rocm/memorypretrain"
	rocmmodel "dappco.re/go/rocm/model"
)

type scoreRouteReply struct {
	Prompt struct {
		Sycophancy *struct {
			Tier int `json:"tier"`
		} `json:"sycophancy"`
	} `json:"prompt"`
	Response struct {
		Sycophancy *struct {
			Tier int `json:"tier"`
		} `json:"sycophancy"`
		LEK *struct {
			LEKScore float64 `json:"lek_score"`
		} `json:"lek"`
		Imprint *struct {
			VocabRichness float64 `json:"vocab_richness"`
		} `json:"imprint"`
	} `json:"response"`
	Differential *struct {
		Echo float64 `json:"echo"`
	} `json:"differential"`
}

func TestRunCommandHelpIncludesContractSurface(t *testing.T) {
	oldName := commandName
	commandName = defaultCLIName
	defer func() { commandName = oldName }()

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), nil, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("runCommand() code=%d, want 0; stderr=%s", code, stderr.String())
	}
	for _, want := range []string{
		"menubar",
		"discover",
		"pack",
		"reactive",
		"bench",
		"ssd-recipes",
		"ssd-eval",
		"memory-pretrain-build",
		"serve",
		"generate",
		"sft",
		"fuse",
		"ssd",
		"tune",
		"diffuse",
		"audio",
		"vision",
		"ebook",
		"slice",
		"state-pack",
	} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("help output missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestBenchJSONReportsGemma4QATMTPContract(t *testing.T) {
	var stdout, stderr bytes.Buffer
	model := "mlx-community/gemma-4-12B-it-qat-4bit"
	code := runCommand(context.Background(), []string{"bench", "-json", "-max-tokens", "128", model}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("bench code=%d, want 0; stderr=%s", code, stderr.String())
	}
	var report benchPlanReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("bench JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Kind != "gemma4-qat-production-bench-plan" ||
		report.Model.Path != model ||
		report.Model.QAT == nil ||
		report.Model.AssistantQAT == nil {
		t.Fatalf("unexpected bench report: %+v", report)
	}
	if report.Model.Assistant != "mlx-community/gemma-4-12B-it-qat-assistant-4bit" {
		t.Fatalf("assistant=%q, want matching MTP-QAT assistant", report.Model.Assistant)
	}
	if report.Target.FloorTokS != 180 || report.Target.GoalTokS != 200 || report.Target.MaxTokens != 128 {
		t.Fatalf("target=%+v, want 180/200 tok/s and 128 max tokens", report.Target)
	}
	if report.Labels["qat_collection"] != "mlx-community/gemma-4-qat" ||
		report.Labels["mtp_qat_collection"] != "mlx-community/gemma-4-mtp-qat" ||
		report.Labels["openai_server_required"] != "true" ||
		report.Labels["stateful_generate_required"] != "true" ||
		report.Labels["measurement_prompt_profile"] != productionMeasurementPromptProfile {
		t.Fatalf("labels missing QAT/state/server contract: %+v", report.Labels)
	}
	commands := benchTestCommands(report.Commands)
	if !strings.Contains(commands, "retained-state Gemma-4 QAT server") {
		t.Fatalf("bench commands missing long MTP warmup prompt:\n%s", commands)
	}
	for _, want := range []string{
		"make release-artifacts",
		"generate",
		"-state",
		"-draft mlx-community/gemma-4-12B-it-qat-assistant-4bit",
		"serve",
		"-state-conversations=true",
		"openai-chat-retained-state",
		"<assistant reply from openai-chat-warm-state>",
		"BenchmarkInferenceGemma4Q4Generate",
		"BenchmarkInferenceGemma4Q4Book10Turn_RetainedState",
		"vllm serve",
		"llama-server",
	} {
		if !strings.Contains(commands, want) {
			t.Fatalf("bench commands missing %q:\n%s", want, commands)
		}
	}
	mtpCommand, ok := benchTestCommandNamed(report.Commands, "mtp-assistant-boundary")
	if !ok || !mtpCommand.Runnable || !strings.Contains(strings.Join(mtpCommand.Notes, " "), "target-retained decode") ||
		!strings.Contains(strings.Join(report.Notes, "\n"), "target-retained decode") {
		t.Fatalf("bench MTP boundary = %+v ok=%v notes=%v, want target-retained runnable boundary", mtpCommand, ok, report.Notes)
	}
	if report.Readiness.Done || report.Readiness.ThroughputProof {
		t.Fatalf("bench report should keep done false until measurements land: %+v", report.Readiness)
	}
}

func TestBenchNamedCUDADefaultReportsPendingRuntimeLane(t *testing.T) {
	oldName := commandName
	commandName = "lthn-cuda"
	defer func() { commandName = oldName }()

	var stdout, stderr bytes.Buffer
	model := "mlx-community/gemma-4-12B-it-qat-4bit"
	code := runCommand(context.Background(), []string{"bench", "-json", model}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("bench code=%d, want 0; stderr=%s", code, stderr.String())
	}
	var report benchPlanReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("bench JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Backend != "cuda" ||
		report.Labels["binary_default_backend"] != "cuda" ||
		report.Labels["active_backend"] != defaultBackendName ||
		report.Labels["runtime_lane"] != "cuda" ||
		report.Labels["runtime_dispatch_status"] != "compile_ready_runtime_dispatch_pending" {
		t.Fatalf("bench backend/labels = %q %+v, want cuda pending runtime lane", report.Backend, report.Labels)
	}
	commands := benchTestCommands(report.Commands)
	for _, want := range []string{
		"lthn-cuda generate",
		"-backend cuda",
		"lthn-cuda serve",
	} {
		if !strings.Contains(commands, want) {
			t.Fatalf("bench commands missing %q:\n%s", want, commands)
		}
	}
}

func TestBenchLocalProductionPackResolvesMTPQATAssistant(t *testing.T) {
	var stdout, stderr bytes.Buffer
	model := "/data/ai/models/mlx-community/gemma-4-e2b-it-6bit"
	code := runCommand(context.Background(), []string{"bench", "-json", "-max-tokens", "64", model}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("bench code=%d, want 0; stderr=%s", code, stderr.String())
	}
	var report benchPlanReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("bench JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Model.Pack == nil ||
		report.Model.Pack.Name != "6bit" ||
		report.Model.Pack.LockedModelID != "mlx-community/gemma-4-e2b-it-6bit" ||
		report.Model.QAT == nil ||
		report.Model.QAT.CollectionID != "mlx-community/gemma-4-qat" ||
		report.Model.QAT.ModelID != "mlx-community/gemma-4-E2B-it-qat-6bit" ||
		report.Model.Assistant != "mlx-community/gemma-4-E2B-it-qat-assistant-6bit" ||
		report.Model.AssistantQAT == nil ||
		report.Model.AssistantQAT.CollectionID != "mlx-community/gemma-4-mtp-qat" {
		t.Fatalf("bench local pack = %+v, want live QAT target paired with MTP-QAT assistant", report.Model)
	}
	commands := benchTestCommands(report.Commands)
	if !strings.Contains(commands, "-draft mlx-community/gemma-4-E2B-it-qat-assistant-6bit") ||
		strings.Contains(commands, "google/gemma-4-E2B-it-assistant") {
		t.Fatalf("bench commands = %s, want local MTP-QAT assistant and no BF16 assistant fallback", commands)
	}
}

func TestBenchRequiresModelPath(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"bench", "-json"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("bench code=%d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "model path is required") {
		t.Fatalf("stderr missing model validation: %s", stderr.String())
	}
}

func TestBenchEvidenceMarksDoneWhenGatePasses(t *testing.T) {
	var stdout, stderr bytes.Buffer
	model := "mlx-community/gemma-4-12B-it-qat-4bit"
	code := runCommand(context.Background(), []string{
		"bench",
		"-json",
		"-release-artifacts-ok",
		"-native-mtp-ok",
		"-rocm-tok-s", "205",
		"-mtp-tok-s", "220",
		"-retained-tok-s", "210",
		"-replay-tok-s", "120",
		"-openai-tok-s", "190",
		"-vllm-tok-s", "180",
		"-llama-tok-s", "96",
		model,
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("bench code=%d, want 0; stderr=%s", code, stderr.String())
	}
	var report benchPlanReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("bench JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Evidence == nil || !report.Evidence.Pass || !report.Readiness.Done || !report.Readiness.ThroughputProof {
		t.Fatalf("evidence/readiness = %+v %+v, want passing done gate", report.Evidence, report.Readiness)
	}
	if report.Readiness.NativeAttachedDrafter != "linked" {
		t.Fatalf("native drafter status=%q, want linked", report.Readiness.NativeAttachedDrafter)
	}
	if report.Evidence.RetainedStateSpeedup != 1.75 {
		t.Fatalf("retained speedup=%v, want 1.75", report.Evidence.RetainedStateSpeedup)
	}
	if report.Labels["bench_evidence_pass"] != "true" ||
		report.Labels["retained_state_speedup"] != "1.750" ||
		report.Labels["native_mtp_tok_s"] != "220.000" ||
		report.Labels["native_mtp_speedup"] != "1.048" ||
		report.Labels["rocm_vs_llama_cpp_ratio"] != "2.135" {
		t.Fatalf("labels missing passing evidence metrics: %+v", report.Labels)
	}
}

func TestBenchEvidenceFailsWithoutStateSpeedupAndNativeMTP(t *testing.T) {
	var stdout, stderr bytes.Buffer
	model := "mlx-community/gemma-4-12B-it-qat-4bit"
	code := runCommand(context.Background(), []string{
		"bench",
		"-json",
		"-release-artifacts-ok",
		"-rocm-tok-s", "205",
		"-retained-tok-s", "150",
		"-replay-tok-s", "160",
		"-openai-tok-s", "190",
		"-vllm-tok-s", "180",
		"-llama-tok-s", "96",
		model,
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("bench code=%d, want 0; stderr=%s", code, stderr.String())
	}
	var report benchPlanReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("bench JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Evidence == nil || report.Evidence.Pass || report.Readiness.Done || report.Readiness.ThroughputProof {
		t.Fatalf("evidence/readiness = %+v %+v, want failing gate", report.Evidence, report.Readiness)
	}
	failures := strings.Join(report.Evidence.Failures, "\n")
	for _, want := range []string{
		"native attached MTP drafter execution",
		"native attached MTP decode tok/s is required",
		"retained-state decode 150.000 tok/s below 180 tok/s floor",
		"retained-state speedup 0.938x",
	} {
		if !strings.Contains(failures, want) {
			t.Fatalf("failures missing %q:\n%s", want, failures)
		}
	}
	if report.Labels["bench_evidence_pass"] != "false" ||
		report.Labels["bench_evidence_failures"] == "" {
		t.Fatalf("labels missing failing evidence markers: %+v", report.Labels)
	}
}

func TestBenchEvidenceFailsWhenNativeMTPSlowerThanRetained(t *testing.T) {
	var stdout, stderr bytes.Buffer
	model := "mlx-community/gemma-4-12B-it-qat-4bit"
	code := runCommand(context.Background(), []string{
		"bench",
		"-json",
		"-release-artifacts-ok",
		"-native-mtp-ok",
		"-rocm-tok-s", "205",
		"-mtp-tok-s", "200",
		"-retained-tok-s", "210",
		"-replay-tok-s", "120",
		"-openai-tok-s", "190",
		"-vllm-tok-s", "180",
		"-llama-tok-s", "96",
		model,
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("bench code=%d, want 0; stderr=%s", code, stderr.String())
	}
	var report benchPlanReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("bench JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Evidence == nil || report.Evidence.Pass || report.Readiness.Done || report.Readiness.ThroughputProof {
		t.Fatalf("evidence/readiness = %+v %+v, want failing native MTP speedup gate", report.Evidence, report.Readiness)
	}
	failures := strings.Join(report.Evidence.Failures, "\n")
	if !strings.Contains(failures, "native MTP speedup 0.952x is not greater than retained-state target decode") {
		t.Fatalf("failures missing native MTP speedup gate:\n%s", failures)
	}
	if report.Labels["bench_evidence_pass"] != "false" ||
		report.Labels["native_mtp_speedup"] != "0.952" ||
		report.Labels["native_mtp_tok_s"] != "200.000" {
		t.Fatalf("labels missing native MTP speedup failure metrics: %+v", report.Labels)
	}
}

func benchTestCommands(commands []benchRunCommand) string {
	parts := make([]string, 0, len(commands))
	for _, command := range commands {
		parts = append(parts, command.Name)
		parts = append(parts, command.Command)
	}
	return strings.Join(parts, "\n")
}

func benchTestCommandNamed(commands []benchRunCommand, name string) (benchRunCommand, bool) {
	for _, command := range commands {
		if command.Name == name {
			return command, true
		}
	}
	return benchRunCommand{}, false
}

func TestRunCommandUnknown(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"bogus"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("runCommand() code=%d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "unknown command") {
		t.Fatalf("stderr missing unknown command: %s", stderr.String())
	}
}

func TestAdminTokenGenerationUsesROCmPrefixAndPreservesExistingTokens(t *testing.T) {
	path := filepath.Join(t.TempDir(), "admin.token")
	token, generated, err := ensureAdminTokenFile(path)
	if err != nil {
		t.Fatalf("ensure admin token failed: %v", err)
	}
	if !generated || !strings.HasPrefix(token, rocmAdminTokenPrefix) || len(token) <= len(rocmAdminTokenPrefix) {
		t.Fatalf("token=%q generated=%t, want new prefixed token", token, generated)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat token file: %v", err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("token file mode=%#o, want 0600", info.Mode().Perm())
	}
	again, generated, err := ensureAdminTokenFile(path)
	if err != nil {
		t.Fatalf("second ensure admin token failed: %v", err)
	}
	if generated || again != token {
		t.Fatalf("second ensure token=%q generated=%t, want existing %q", again, generated, token)
	}
	rotated, err := rotateAdminTokenFile(path)
	if err != nil {
		t.Fatalf("rotate admin token failed: %v", err)
	}
	if !strings.HasPrefix(rotated, rocmAdminTokenPrefix) || rotated == token {
		t.Fatalf("rotated token=%q, want distinct prefixed token", rotated)
	}

	legacyPath := filepath.Join(t.TempDir(), "legacy.token")
	if err := os.WriteFile(legacyPath, []byte("legacy-token\n"), 0o600); err != nil {
		t.Fatalf("write legacy token: %v", err)
	}
	legacy, generated, err := ensureAdminTokenFile(legacyPath)
	if err != nil {
		t.Fatalf("ensure legacy admin token failed: %v", err)
	}
	if generated || legacy != "legacy-token" {
		t.Fatalf("legacy token=%q generated=%t, want preserved existing token", legacy, generated)
	}
}

func TestDiscoverJSONReportsROCmContract(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"discover", "-json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("discover code=%d, want 0; stderr=%s", code, stderr.String())
	}
	var report discoveryReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("discover JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Runtime.Backend != "rocm" {
		t.Fatalf("Runtime.Backend=%q, want rocm", report.Runtime.Backend)
	}
	if report.FastLane.Backend != "rocm" {
		t.Fatalf("FastLane.Backend=%q, want rocm", report.FastLane.Backend)
	}
	if report.FastLane.RequiresEnvGate || report.FastLane.RequiresCLIFlag {
		t.Fatalf("fast lane should not require env gates or CLI flags: %+v", report.FastLane)
	}
	if report.DefaultBackend != "rocm" {
		t.Fatalf("DefaultBackend=%q, want rocm", report.DefaultBackend)
	}
	if report.ModelRegistry.Name != rocm.DefaultROCmModelRegistryName() ||
		report.ModelRegistry.DefaultFamily != "gemma4" ||
		!cliTestStringSliceContains(report.ModelRegistry.Factories, "gemma4") ||
		report.ModelRegistry.Labels["engine_algorithm_profile_contract"] != rocm.ROCmAlgorithmProfileRegistryContract ||
		report.ModelRegistry.Labels["engine_config_probe_contract"] != rocm.ROCmModelConfigProbeContract ||
		report.ModelRegistry.Labels["engine_feature_route_contract"] != rocm.ROCmModelFeatureRegistryContract ||
		report.ModelRegistry.Labels["engine_tokenizer_route_contract"] != rocm.ROCmModelTokenizerRegistryContract ||
		report.ModelRegistry.Labels["engine_lora_route_contract"] != rocm.ROCmLoRAAdapterRegistryContract ||
		report.ModelRegistry.Labels["engine_loader_contract"] != rocm.ROCmModelLoaderRegistryContract ||
		report.ModelRegistry.Labels["engine_mixer_loader_contract"] != rocm.ROCmSequenceMixerLoaderRegistryContract ||
		report.ModelRegistry.Labels["engine_multimodal_processor_route_contract"] != rocm.ROCmMultimodalProcessorRegistryContract ||
		report.ModelRegistry.Labels["engine_diffusion_sampler_route_contract"] != rocm.ROCmDiffusionSamplerRegistryContract ||
		report.ModelRegistry.Labels["engine_state_context_route_contract"] != rocm.ROCmStateContextRegistryContract ||
		report.ModelRegistry.Labels["engine_attached_drafter_route_contract"] != rocm.ROCmAttachedDrafterRegistryContract ||
		report.ModelRegistry.Labels["engine_quant_scheme_contract"] != rocm.ROCmQuantSchemeRegistryContract ||
		report.ModelRegistry.Labels["engine_quant_scheme_kinds"] != "affine,bf16,mxfp4,mxfp8,nvfp4,q4_0,jangtq" ||
		report.ModelRegistry.Labels["engine_quant_loader_contract"] != rocm.ROCmQuantLoaderRegistryContract ||
		report.ModelRegistry.Labels["engine_profile_reactive"] != "true" ||
		report.ModelRegistry.Labels["algorithm_profile_count"] != strconv.Itoa(len(rocm.DefaultROCmAlgorithmProfiles())) ||
		report.ModelRegistry.Labels["production_contract"] != cliContractName {
		t.Fatalf("ModelRegistry = %+v, want reactive Gemma4 registry contract", report.ModelRegistry)
	}
	algorithmProfiles := map[inference.CapabilityID]rocm.ROCmAlgorithmProfile{}
	for _, profile := range report.ModelRegistry.AlgorithmProfiles {
		algorithmProfiles[profile.ID] = profile
	}
	if len(algorithmProfiles) != len(rocm.DefaultROCmAlgorithmProfiles()) ||
		algorithmProfiles[inference.CapabilityQuantization].Algorithm != "auto-round" ||
		!cliTestStringSliceContains(algorithmProfiles[inference.CapabilityQuantization].Provides, "hip.autoround_quantize.kernel") ||
		algorithmProfiles[inference.CapabilitySpeculativeDecode].Algorithm != "speculative-decode" ||
		!cliTestStringSliceContains(algorithmProfiles[inference.CapabilitySpeculativeDecode].Provides, "mtp.attached_drafter.plan") ||
		algorithmProfiles[inference.CapabilityCacheDisk].RuntimeStatus != inference.FeatureRuntimePlanned {
		t.Fatalf("algorithm profiles = %+v, want discoverable ROCm algorithm matrix", report.ModelRegistry.AlgorithmProfiles)
	}
	profiles := map[string]rocm.Gemma4ArchitectureSettings{}
	for _, profile := range report.ModelRegistry.ArchitectureProfiles {
		profiles[profile.ID] = profile
	}
	if len(profiles) != len(rocm.DefaultROCmArchitectureProfiles()) ||
		profiles["composed"].Family != "composed" ||
		profiles["composed"].Generation ||
		profiles["hybrid"].Family != "hybrid" ||
		profiles["gemma4"].TextTowerID != "gemma4_text" ||
		profiles["gemma4_text"].ParserID != "gemma" ||
		!profiles["gemma4_assistant"].AttachedOnly ||
		profiles["gemma4_assistant"].Generation ||
		!cliTestStringSliceContains(profiles["gemma4_text"].QuantizationHints, "q6") ||
		!cliTestStringSliceContains(profiles["gemma4_assistant"].CacheHints, "attached-drafter") ||
		profiles["qwen3_6_moe"].Family != "qwen" ||
		!profiles["qwen3_6_moe"].MoE ||
		!profiles["bert_rerank"].Rerank ||
		profiles["bert_rerank"].Chat {
		t.Fatalf("architecture profiles = %+v, want discoverable ROCm reactive architecture profiles", report.ModelRegistry.ArchitectureProfiles)
	}
	featureRoutes := map[string]rocm.ROCmModelFeatureRoute{}
	for _, route := range report.ModelRegistry.FeatureRoutes {
		featureRoutes[route.Architecture] = route
	}
	if len(featureRoutes) != len(rocm.DefaultROCmModelFeatureRoutes()) ||
		featureRoutes["gemma4_text"].Contract != rocm.ROCmModelFeatureRegistryContract ||
		featureRoutes["gemma4_text"].ReasoningParserID != "gemma" ||
		featureRoutes["gemma4_text"].ChatTemplateID != "gemma4_hf_turn" ||
		!featureRoutes["gemma4_text"].Generation ||
		featureRoutes["gemma4_text"].TextGenerate ||
		featureRoutes["qwen3_6_moe"].Family != "qwen" ||
		!featureRoutes["qwen3_6_moe"].MoE ||
		featureRoutes["bert_rerank"].Chat ||
		!featureRoutes["bert_rerank"].Rerank {
		t.Fatalf("feature routes = %+v, want discoverable ROCm model feature route registry", report.ModelRegistry.FeatureRoutes)
	}
	tokenizerRoutes := map[string]rocm.ROCmModelTokenizerRoute{}
	for _, route := range report.ModelRegistry.TokenizerRoutes {
		tokenizerRoutes[route.Architecture] = route
	}
	if len(tokenizerRoutes) != len(rocm.DefaultROCmModelTokenizerRoutes()) ||
		tokenizerRoutes["gemma4_text"].Contract != rocm.ROCmModelTokenizerRegistryContract ||
		tokenizerRoutes["gemma4_text"].TokenizerKind != "GemmaTokenizer" ||
		tokenizerRoutes["gemma4_text"].ChatTemplateID != "gemma4_hf_turn" ||
		!tokenizerRoutes["gemma4_text"].RequiresChatTemplate ||
		tokenizerRoutes["qwen3_6_moe"].TokenizerKind != "Qwen2Tokenizer" ||
		tokenizerRoutes["qwen3_6_moe"].ChatTemplateID != "qwen" ||
		tokenizerRoutes["bert_rerank"].ChatTemplate {
		t.Fatalf("tokenizer routes = %+v, want discoverable ROCm tokenizer route registry", report.ModelRegistry.TokenizerRoutes)
	}
	loraRoutes := map[string]rocm.ROCmLoRAAdapterRoute{}
	for _, route := range report.ModelRegistry.LoRAAdapterRoutes {
		loraRoutes[route.Architecture] = route
	}
	if len(loraRoutes) != len(rocm.DefaultROCmLoRAAdapterRoutes()) ||
		loraRoutes["gemma4_text"].Contract != rocm.ROCmLoRAAdapterRegistryContract ||
		loraRoutes["gemma4_text"].TargetPolicy != "gemma4" ||
		loraRoutes["gemma4_text"].TargetPaths["q_proj"] != "self_attn.q_proj" ||
		!loraRoutes["gemma4_text"].Registered ||
		!loraRoutes["gemma4_text"].Staged ||
		loraRoutes["qwen3_6_moe"].TargetPolicy != "decoder" ||
		loraRoutes["qwen3_6_moe"].TargetPaths["gate_proj"] != "mlp.gate_proj" ||
		loraRoutes["composed"].TargetPolicy != "composed_mlp" ||
		cliTestStringSliceContains(loraRoutes["composed"].SafeTargets, "q_proj") {
		t.Fatalf("LoRA adapter routes = %+v, want discoverable ROCm adapter route registry", report.ModelRegistry.LoRAAdapterRoutes)
	}
	multimodalRoutes := map[string]rocm.ROCmMultimodalProcessorRoute{}
	for _, route := range report.ModelRegistry.MultimodalProcessorRoutes {
		multimodalRoutes[route.Architecture] = route
	}
	if len(multimodalRoutes) != len(rocm.DefaultROCmMultimodalProcessorRoutes()) ||
		multimodalRoutes["gemma4"].Contract != rocm.ROCmMultimodalProcessorRegistryContract ||
		multimodalRoutes["gemma4"].Reference != "go_mlx_gemma4_vision" ||
		!multimodalRoutes["gemma4"].Vision ||
		!multimodalRoutes["gemma4"].Video ||
		multimodalRoutes["gemma4"].Audio ||
		multimodalRoutes["gemma4_unified"].Reference != "go_mlx_gemma4_audio" ||
		!multimodalRoutes["gemma4_unified"].Audio ||
		multimodalRoutes["gemma3"].Reference != "go_mlx_gemma3_multimodal_wrapper" {
		t.Fatalf("multimodal processor routes = %+v, want discoverable ROCm multimodal processor registry", report.ModelRegistry.MultimodalProcessorRoutes)
	}
	diffusionRoutes := map[string]rocm.ROCmDiffusionSamplerRoute{}
	for _, route := range report.ModelRegistry.DiffusionSamplerRoutes {
		diffusionRoutes[route.Architecture] = route
	}
	if len(diffusionRoutes) != len(rocm.DefaultROCmDiffusionSamplerRoutes()) ||
		diffusionRoutes["diffusion_gemma"].Contract != rocm.ROCmDiffusionSamplerRegistryContract ||
		diffusionRoutes["diffusion_gemma"].Reference != "go_mlx_diffusion_gemma" ||
		diffusionRoutes["diffusion_gemma"].DiffusionRuntime != "not_linked" ||
		diffusionRoutes["diffusion_gemma"].SamplerRuntime != "not_linked" ||
		diffusionRoutes["diffusion_gemma"].TrunkRuntime != "model_pack_metadata" ||
		diffusionRoutes["diffusion_gemma"].NativeRuntime ||
		!diffusionRoutes["diffusion_gemma"].BlockDiffusion ||
		!diffusionRoutes["diffusion_gemma"].FallbackRefused {
		t.Fatalf("diffusion sampler routes = %+v, want discoverable ROCm block diffusion sampler registry", report.ModelRegistry.DiffusionSamplerRoutes)
	}
	stateRoutes := map[string]rocm.ROCmStateContextRoute{}
	for _, route := range report.ModelRegistry.StateContextRoutes {
		stateRoutes[route.Architecture] = route
	}
	if len(stateRoutes) != len(rocm.DefaultROCmStateContextRoutes()) ||
		stateRoutes["gemma4_text"].Contract != rocm.ROCmStateContextRegistryContract ||
		stateRoutes["gemma4_text"].Reference != "go_mlx_gemma4_retained_state" ||
		!stateRoutes["gemma4_text"].StateSession ||
		!stateRoutes["gemma4_text"].RuntimeOwnedKV ||
		!stateRoutes["gemma4_text"].PromptReplayRefused ||
		!stateRoutes["gemma4_text"].RemainingContextDefault ||
		!stateRoutes["gemma4_text"].WakeState ||
		!stateRoutes["gemma4_text"].SleepState ||
		!stateRoutes["gemma4_text"].ForkState ||
		stateRoutes["gemma4_assistant"].Reference != "go_mlx_gemma4_attached_drafter_retained_state" ||
		!stateRoutes["gemma4_assistant"].AttachedDrafterState {
		t.Fatalf("state context routes = %+v, want discoverable ROCm retained-state context registry", report.ModelRegistry.StateContextRoutes)
	}
	attachedRoutes := map[string]rocm.ROCmAttachedDrafterRoute{}
	for _, route := range report.ModelRegistry.AttachedDrafterRoutes {
		attachedRoutes[route.Architecture] = route
	}
	if len(attachedRoutes) != len(rocm.DefaultROCmAttachedDrafterRoutes()) ||
		attachedRoutes["gemma4_text"].Contract != rocm.ROCmAttachedDrafterRegistryContract ||
		attachedRoutes["gemma4_text"].Reference != "go_mlx_gemma4_assistant_pair" ||
		attachedRoutes["gemma4_text"].Mode != "mtp_attached_drafter" ||
		attachedRoutes["gemma4_text"].Role != "target" ||
		attachedRoutes["gemma4_text"].AssistantArchitecture != "gemma4_assistant" ||
		attachedRoutes["gemma4_text"].NativeAttachment != "not_linked" ||
		attachedRoutes["gemma4_text"].NativeRuntime ||
		!attachedRoutes["gemma4_text"].PairValidation ||
		!attachedRoutes["gemma4_text"].RetainedStateRequired ||
		!attachedRoutes["gemma4_text"].PromptReplayRefused ||
		!attachedRoutes["gemma4_text"].DraftDetection ||
		!cliTestStringSliceContains(attachedRoutes["gemma4_text"].AssistantModelIDs, "google/gemma-4-31B-it-assistant") ||
		!cliTestStringSliceContains(attachedRoutes["gemma4_text"].AssistantModelIDs, "mlx-community/gemma-4-31B-it-qat-assistant-6bit") ||
		!cliTestStringSliceContains(attachedRoutes["gemma4_text"].AssistantQuantModes, "q6") ||
		!cliTestStringSliceContains(attachedRoutes["gemma4_text"].DetectionSources, string(rocm.DraftSourceAssistantDir)) ||
		attachedRoutes["gemma4_assistant"].Role != "assistant" ||
		!attachedRoutes["gemma4_assistant"].AttachedOnly {
		t.Fatalf("attached drafter routes = %+v, want discoverable native-pending MTP route registry", report.ModelRegistry.AttachedDrafterRoutes)
	}
	routes := map[string]rocm.ROCmModelLoaderRoute{}
	for _, route := range report.ModelRegistry.LoaderRoutes {
		routes[route.Architecture] = route
	}
	if len(routes) != len(rocm.DefaultROCmModelLoaderRoutes()) ||
		routes["gemma4_text"].Loader != "gemma4_text" ||
		routes["gemma4_text"].Runtime != "hip" ||
		routes["gemma4_text"].Status != rocm.ROCmModelLoadStandaloneNative ||
		!routes["gemma4_text"].Registered ||
		!routes["gemma4_text"].TextGenerate ||
		routes["gemma4_text"].Staged ||
		routes["gemma4_assistant"].Target != "attached" ||
		!routes["gemma4_assistant"].AttachedOnly ||
		routes["composed"].Status != rocm.ROCmModelLoadStagedNative ||
		!routes["composed"].Registered ||
		routes["qwen3_6_moe"].Family != "qwen" {
		t.Fatalf("loader routes = %+v, want discoverable ROCm loader route registry", report.ModelRegistry.LoaderRoutes)
	}
	quantSchemes := map[string]rocm.ROCmQuantScheme{}
	for _, scheme := range report.ModelRegistry.QuantSchemes {
		quantSchemes[scheme.Kind] = scheme
	}
	if len(quantSchemes) != len(rocm.DefaultROCmQuantSchemes()) ||
		quantSchemes["affine"].Contract != rocm.ROCmQuantSchemeRegistryContract ||
		quantSchemes["affine"].Loader != "gemma4_affine" ||
		!quantSchemes["affine"].NativeRuntime ||
		quantSchemes["mxfp4"].RuntimeStatus != inference.FeatureRuntimePlanned ||
		!quantSchemes["mxfp4"].Planned ||
		quantSchemes["q4_0"].RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		!quantSchemes["q4_0"].MetadataOnly {
		t.Fatalf("quant schemes = %+v, want discoverable ROCm quant scheme registry", report.ModelRegistry.QuantSchemes)
	}
	quantRoutes := map[string]rocm.ROCmQuantLoaderRoute{}
	for _, route := range report.ModelRegistry.QuantLoaderRoutes {
		quantRoutes[route.Size+":"+route.Mode] = route
	}
	if len(quantRoutes) != len(rocm.DefaultROCmQuantLoaderRoutes()) ||
		quantRoutes["E2B:q6"].Loader != "gemma4_affine" ||
		quantRoutes["E2B:q6"].GenerateStatus != rocm.Gemma4GenerateLinked ||
		!quantRoutes["E2B:q6"].Registered ||
		!quantRoutes["E2B:q6"].NativeRuntime ||
		quantRoutes["E2B:bf16"].Target != "load" ||
		!quantRoutes["E2B:bf16"].LoadOnly ||
		quantRoutes["E2B:mxfp4"].Target != "planned" ||
		!quantRoutes["E2B:mxfp4"].Planned ||
		quantRoutes["31B:q4-status"].Target != "metadata" {
		t.Fatalf("quant loader routes = %+v, want discoverable ROCm quant loader registry", report.ModelRegistry.QuantLoaderRoutes)
	}
	mixerRoutes := map[string]rocm.ROCmSequenceMixerLoaderRoute{}
	for _, route := range report.ModelRegistry.MixerLoaderRoutes {
		mixerRoutes[route.Kind] = route
	}
	if len(mixerRoutes) != len(rocm.DefaultROCmSequenceMixerLoaderRoutes()) ||
		mixerRoutes["full_attention"].Contract != rocm.ROCmSequenceMixerLoaderRegistryContract ||
		mixerRoutes["full_attention"].CacheMode != rocm.SequenceMixerCacheModeDefault ||
		!mixerRoutes["full_attention"].Registered ||
		mixerRoutes["mamba2"].State != rocm.SequenceMixerStateRecurrent ||
		mixerRoutes["mamba2"].CacheMode != rocm.SequenceMixerCacheModeRecurrent ||
		!slices.Equal(mixerRoutes["mamba2"].StateSlots, []string{"conv_state", "ssm_state"}) ||
		!slices.Equal(mixerRoutes["gsa"].StateSlots, []string{"slot_key_state", "slot_value_state"}) ||
		!slices.Equal(mixerRoutes["rwkv7"].StateSlots, []string{"wkv_state"}) ||
		mixerRoutes["mla"].CacheMode != rocm.SequenceMixerCacheModeMLALatent {
		t.Fatalf("mixer loader routes = %+v, want discoverable ROCm sequence-mixer loader registry", report.ModelRegistry.MixerLoaderRoutes)
	}
	if len(report.Compile.KernelTargets) == 0 {
		t.Fatalf("discover should report HIP kernel compile targets")
	}
	var haveNVIDIA, haveHIPCPU bool
	for _, target := range report.Compile.KernelTargets {
		if target.Backend == "cuda" && target.Kind == "hip-kernel" {
			haveNVIDIA = true
		}
		if target.Backend == "cpu" && target.Kind == "hip-cpu-kernel" {
			haveHIPCPU = true
		}
	}
	if !haveNVIDIA || !haveHIPCPU {
		t.Fatalf("compile targets missing cuda=%t hip_cpu=%t: %+v", haveNVIDIA, haveHIPCPU, report.Compile.KernelTargets)
	}
	if report.Labels["cli_contract"] != cliContractName {
		t.Fatalf("cli_contract label=%q", report.Labels["cli_contract"])
	}
}

func TestDiscoverJSONReportsNamedReleaseBinaryTargets(t *testing.T) {
	oldName := commandName
	defer func() { commandName = oldName }()

	cases := []struct {
		name           string
		backend        string
		lane           string
		sidecar        string
		dispatchStatus string
		stubRuntime    bool
	}{
		{
			name:           "lthn-amd",
			backend:        defaultBackendName,
			lane:           "amd",
			sidecar:        "rocm_kernels_gfx1100.hsaco",
			dispatchStatus: "active",
		},
		{
			name:           "lthn-cuda",
			backend:        "cuda",
			lane:           "cuda",
			sidecar:        "rocm_kernels_nvidia_sm_75.o",
			dispatchStatus: "compile_ready_runtime_dispatch_pending",
		},
		{
			name:           "lthn-cpu-x86",
			backend:        "cpu",
			lane:           "cpu-x86",
			sidecar:        "rocm_kernels_hip_cpu_x86_64.o",
			dispatchStatus: "compile_ready_runtime_dispatch_pending",
			stubRuntime:    true,
		},
		{
			name:           "lthn-cpu-aarch64",
			backend:        "cpu",
			lane:           "cpu-aarch64",
			sidecar:        "rocm_kernels_hip_cpu_aarch64.o",
			dispatchStatus: "compile_ready_runtime_dispatch_pending",
			stubRuntime:    true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			commandName = tc.name
			var stdout, stderr bytes.Buffer
			code := runCommand(context.Background(), []string{"discover", "-json"}, &stdout, &stderr)
			if code != 0 {
				t.Fatalf("discover code=%d, want 0; stderr=%s", code, stderr.String())
			}
			var report discoveryReport
			if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
				t.Fatalf("discover JSON did not unmarshal: %v\n%s", err, stdout.String())
			}
			current := report.Compile.Current
			if current.Name != tc.name ||
				current.Artifact != tc.name ||
				current.Kind != "release-binary" ||
				current.Backend != tc.backend ||
				current.Labels["runtime_lane"] != tc.lane ||
				current.Labels["runtime_dispatch_status"] != tc.dispatchStatus ||
				current.Labels["active_backend"] != defaultBackendName ||
				current.Labels["production_artifact"] != "true" ||
				current.StubRuntime != tc.stubRuntime ||
				!cliTestStringSliceContains(current.Sidecars, tc.sidecar) {
				t.Fatalf("current compile target = %+v, want named release target %s/%s", current, tc.backend, tc.lane)
			}
			if report.DefaultBackend != tc.backend || report.Runtime.Backend != tc.backend {
				t.Fatalf("default/runtime backend = %q/%q, want binary lane backend %q", report.DefaultBackend, report.Runtime.Backend, tc.backend)
			}
			if report.Labels["binary_default_backend"] != tc.backend ||
				report.Labels["active_backend"] != defaultBackendName ||
				report.Labels["runtime_lane"] != tc.lane ||
				report.Labels["runtime_dispatch_status"] != tc.dispatchStatus {
				t.Fatalf("discover labels = %+v, want binary lane %s/%s", report.Labels, tc.backend, tc.lane)
			}
			var found bool
			for _, target := range report.Compile.BinaryTargets {
				if target.Name == tc.name && target.Backend == tc.backend && cliTestStringSliceContains(target.Sidecars, tc.sidecar) {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("binary target %s missing from compile report: %+v", tc.name, report.Compile.BinaryTargets)
			}
		})
	}
}

func TestGenerateNamedCUDADefaultRefusesPendingRuntimeDispatch(t *testing.T) {
	oldName := commandName
	commandName = "lthn-cuda"
	defer func() { commandName = oldName }()

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-draft", "", "-state", "", "/tmp/model"}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("generate code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate wrote stdout on pending dispatch: %s", stdout.String())
	}
	for _, want := range []string{
		"cuda runtime lane is compiled and packaged",
		"rocm_kernels_nvidia_sm_75.o",
		"runtime_dispatch_status=compile_ready_runtime_dispatch_pending",
		"pass -backend rocm",
	} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("stderr missing %q:\n%s", want, stderr.String())
		}
	}
}

func TestGenerateNamedCUDAAllowsExplicitROCmBackendOverride(t *testing.T) {
	oldName := commandName
	commandName = "lthn-cuda"
	defer func() { commandName = oldName }()

	original, hadOriginal := inference.Get(defaultBackendName)
	model := &stateCLITestModel{}
	inference.Register(&stateCLITestBackend{name: defaultBackendName, model: model})
	t.Cleanup(func() {
		if hadOriginal {
			inference.Register(original)
		}
	})

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"generate",
		"-backend", defaultBackendName,
		"-draft", "",
		"-state", "",
		"-prompt", "override",
		"/tmp/model",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("generate code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("generate stderr=%s", stderr.String())
	}
	if !strings.Contains(stdout.String(), "state") || len(model.prompts) != 1 || model.prompts[0] != "override" {
		t.Fatalf("stdout/prompts = %q/%q, want explicit ROCm backend override to load test model", stdout.String(), model.prompts)
	}
}

func TestServeNamedCPUDefaultRefusesPendingRuntimeDispatchWhenModelProvided(t *testing.T) {
	oldName := commandName
	commandName = "lthn-cpu-x86"
	defer func() { commandName = oldName }()

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"serve", "-model", "/tmp/model", "-addr", "127.0.0.1:0"}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("serve code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("serve wrote stdout on pending dispatch: %s", stdout.String())
	}
	for _, want := range []string{
		"cpu runtime lane is compiled and packaged",
		"rocm_kernels_hip_cpu_x86_64.o",
		"runtime_dispatch_status=compile_ready_runtime_dispatch_pending",
		"pass -backend rocm",
	} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("stderr missing %q:\n%s", want, stderr.String())
		}
	}
}

func TestDiscoverTextReportsModelRegistry(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"discover"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("discover code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "model registry: "+rocm.DefaultROCmModelRegistryName()) ||
		!strings.Contains(stdout.String(), "profiles="+strconv.Itoa(len(rocm.DefaultROCmArchitectureProfiles()))) ||
		!strings.Contains(stdout.String(), "algorithms="+strconv.Itoa(len(rocm.DefaultROCmAlgorithmProfiles()))) ||
		!strings.Contains(stdout.String(), "feature_routes="+strconv.Itoa(len(rocm.DefaultROCmModelFeatureRoutes()))) ||
		!strings.Contains(stdout.String(), "tokenizers="+strconv.Itoa(len(rocm.DefaultROCmModelTokenizerRoutes()))) ||
		!strings.Contains(stdout.String(), "adapters="+strconv.Itoa(len(rocm.DefaultROCmLoRAAdapterRoutes()))) ||
		!strings.Contains(stdout.String(), "multimodal_processors="+strconv.Itoa(len(rocm.DefaultROCmMultimodalProcessorRoutes()))) ||
		!strings.Contains(stdout.String(), "diffusion_samplers="+strconv.Itoa(len(rocm.DefaultROCmDiffusionSamplerRoutes()))) ||
		!strings.Contains(stdout.String(), "state_contexts="+strconv.Itoa(len(rocm.DefaultROCmStateContextRoutes()))) ||
		!strings.Contains(stdout.String(), "attached_drafters="+strconv.Itoa(len(rocm.DefaultROCmAttachedDrafterRoutes()))) ||
		!strings.Contains(stdout.String(), "loaders="+strconv.Itoa(len(rocm.DefaultROCmModelLoaderRoutes()))) ||
		!strings.Contains(stdout.String(), "cache_modes="+strconv.Itoa(len(rocm.DefaultROCmCacheModeRoutes()))) ||
		!strings.Contains(stdout.String(), "cache_routes="+strconv.Itoa(len(rocm.DefaultROCmModelRegistrySnapshot("").CacheRoutes))) ||
		!strings.Contains(stdout.String(), "quant_schemes="+strconv.Itoa(len(rocm.DefaultROCmQuantSchemes()))) ||
		!strings.Contains(stdout.String(), "quant_loaders="+strconv.Itoa(len(rocm.DefaultROCmQuantLoaderRoutes()))) ||
		!strings.Contains(stdout.String(), "mixers="+strconv.Itoa(len(rocm.DefaultROCmSequenceMixerLoaderRoutes()))) {
		t.Fatalf("discover text output missing model registry profile summary:\n%s", stdout.String())
	}
}

func TestDiscoverRejectsInvalidLimit(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"discover", "-json", "-max-models=-1"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("discover code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("discover wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "-max-models must be >= 0") {
		t.Fatalf("stderr missing max-models validation: %s", stderr.String())
	}
}

func TestRunReactiveCommandJSONReportsComposedPlan(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{
		"model_type":"composed",
		"hidden_size":4,
		"num_hidden_layers":2,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":4,
		"vocab_size":16,
		"max_position_embeddings":32768,
		"layer_types":["full_attention","mamba2"]
	}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLITestSafetensors(t, filepath.Join(dir, "model.safetensors"), map[string]cliTestTensor{
		"model.embed_tokens.weight":                      {DType: "F32", Shape: []uint64{1}, Values: []float32{1}},
		"lm_head.weight":                                 {DType: "F32", Shape: []uint64{1}, Values: []float32{2}},
		"model.norm.weight":                              {DType: "F32", Shape: []uint64{1}, Values: []float32{3}},
		"model.layers.0.input_layernorm.weight":          {DType: "F32", Shape: []uint64{1}, Values: []float32{4}},
		"model.layers.0.post_attention_layernorm.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{5}},
		"model.layers.0.self_attn.q_proj.weight":         {DType: "F32", Shape: []uint64{1}, Values: []float32{3}},
		"model.layers.0.self_attn.k_proj.weight":         {DType: "F32", Shape: []uint64{1}, Values: []float32{4}},
		"model.layers.0.self_attn.v_proj.weight":         {DType: "F32", Shape: []uint64{1}, Values: []float32{5}},
		"model.layers.0.self_attn.o_proj.weight":         {DType: "F32", Shape: []uint64{1}, Values: []float32{6}},
		"model.layers.0.mlp.gate_proj.weight":            {DType: "F32", Shape: []uint64{1}, Values: []float32{6}},
		"model.layers.0.mlp.up_proj.weight":              {DType: "F32", Shape: []uint64{1}, Values: []float32{6}},
		"model.layers.0.mlp.down_proj.weight":            {DType: "F32", Shape: []uint64{1}, Values: []float32{7}},
		"model.layers.1.input_layernorm.weight":          {DType: "F32", Shape: []uint64{1}, Values: []float32{8}},
		"model.layers.1.post_attention_layernorm.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{9}},
		"model.layers.1.mixer.in_proj.weight":            {DType: "F32", Shape: []uint64{1}, Values: []float32{8}},
		"model.layers.1.mixer.out_proj.weight":           {DType: "F32", Shape: []uint64{1}, Values: []float32{9}},
		"model.layers.1.mixer.conv1d.weight":             {DType: "F32", Shape: []uint64{1}, Values: []float32{10}},
		"model.layers.1.mixer.A_log":                     {DType: "F32", Shape: []uint64{1}, Values: []float32{11}},
		"model.layers.1.mlp.gate_proj.weight":            {DType: "F32", Shape: []uint64{1}, Values: []float32{11}},
		"model.layers.1.mlp.up_proj.weight":              {DType: "F32", Shape: []uint64{1}, Values: []float32{11}},
		"model.layers.1.mlp.down_proj.weight":            {DType: "F32", Shape: []uint64{1}, Values: []float32{12}},
	})

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"reactive", "-json", dir}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("reactive code=%d, want 0; stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("reactive JSON wrote stderr: %s", stderr.String())
	}
	var report rocm.ReactiveSequenceMixerReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("reactive JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Status != "ready_for_native_load" ||
		report.ExecutionStatus != "load_plan_ready_runner_pending" ||
		!report.PlanningReady ||
		!report.TensorBindingReady ||
		!report.ComposedStackReady ||
		report.RunnerReady {
		t.Fatalf("reactive readiness = %+v, want plan ready with runner pending", report)
	}
	if report.Plan == nil ||
		len(report.Plan.Layers) != 2 ||
		report.Plan.Layers[0].Kind != "full_attention" ||
		report.Plan.Layers[0].Subpath != "self_attn" ||
		report.Plan.Layers[1].Kind != "mamba2" ||
		report.Plan.Layers[1].State != rocm.SequenceMixerStateRecurrent ||
		len(report.Plan.Cache.Layers) != 2 ||
		report.Plan.Cache.Layers[0].Mode != rocm.SequenceMixerCacheModeDefault ||
		report.Plan.Cache.Layers[1].Holder != rocm.SequenceMixerStateRecurrent ||
		report.Plan.Cache.Layers[1].Mode != rocm.SequenceMixerCacheModeRecurrent {
		t.Fatalf("reactive plan = %+v, want go-mlx composed loader plan", report.Plan)
	}
	if report.Labels["sequence_mixer_tensor_binding"] != "ready" ||
		report.Labels["sequence_mixer_composed_stack"] != "ready" ||
		report.Labels["sequence_mixer_runner_status"] != "hip_composed_runner_pending" ||
		report.Labels["sequence_mixer_cache_factory_contract"] != rocm.SequenceMixerCacheFactoryContract ||
		report.Labels["production_requires_env_gate"] != "false" ||
		report.Labels["production_requires_cli_flag"] != "false" {
		t.Fatalf("reactive labels = %+v, want production contract labels", report.Labels)
	}

	stdout.Reset()
	stderr.Reset()
	code = runCommand(context.Background(), []string{"reactive", dir}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("reactive text code=%d, want 0; stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"subpaths: 0:self_attn,1:mixer",
		"cache holders: 0:full_attention:kv-cache,1:mamba2:recurrent",
		"cache modes: 0:full_attention:default,1:mamba2:recurrent",
		"runner: hip_composed_runner_pending",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("reactive text output missing %q:\n%s", want, out)
		}
	}
}

func TestRunReactiveCommandJSONRejectsIncompleteComposedStack(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{
		"model_type":"composed",
		"hidden_size":4,
		"num_hidden_layers":1,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":4,
		"vocab_size":16,
		"max_position_embeddings":32768,
		"layer_types":["full_attention"]
	}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLITestSafetensors(t, filepath.Join(dir, "model.safetensors"), map[string]cliTestTensor{
		"model.embed_tokens.weight":              {DType: "F32", Shape: []uint64{1}, Values: []float32{1}},
		"model.layers.0.self_attn.q_proj.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{2}},
		"model.layers.0.self_attn.k_proj.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{3}},
		"model.layers.0.self_attn.v_proj.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{4}},
		"model.layers.0.self_attn.o_proj.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{5}},
	})

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"reactive", "-json", dir}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("reactive code=%d, want 1; stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("reactive JSON wrote stderr: %s", stderr.String())
	}
	var report rocm.ReactiveSequenceMixerReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("reactive JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Status != "incomplete" ||
		report.ExecutionStatus != "composed_stack_missing_runner_pending" ||
		!report.PlanningReady ||
		report.TensorBindingReady ||
		report.ComposedStackReady ||
		!cliTestStringSliceContains(report.MissingTensors, "model.norm.weight") ||
		!cliTestStringSliceContains(report.MissingTensors, "model.layers.0.mlp.gate_proj.weight") {
		t.Fatalf("reactive incomplete report = %+v, want missing composed stack tensors", report)
	}
	if report.Labels["sequence_mixer_tensor_binding"] != "mixer_ready" ||
		report.Labels["sequence_mixer_composed_stack"] != "missing" {
		t.Fatalf("reactive incomplete labels = %+v", report.Labels)
	}
}

func cliTestStringSliceContains(values []string, needle string) bool {
	for _, value := range values {
		if value == needle {
			return true
		}
	}
	return false
}

func TestRunReactiveCommandJSONRejectsAmbiguousSubpath(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{
		"model_type":"composed",
		"hidden_size":4,
		"num_hidden_layers":1,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":4,
		"vocab_size":16,
		"max_position_embeddings":32768,
		"layer_types":["full_attention"]
	}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLITestSafetensors(t, filepath.Join(dir, "model.safetensors"), map[string]cliTestTensor{
		"model.layers.0.self_attn.q_proj.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{1}},
		"model.layers.0.self_attn.k_proj.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{2}},
		"model.layers.0.self_attn.v_proj.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{3}},
		"model.layers.0.self_attn.o_proj.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{4}},
		"model.layers.0.mixer.in_proj.weight":    {DType: "F32", Shape: []uint64{1}, Values: []float32{5}},
		"model.layers.0.mlp.down_proj.weight":    {DType: "F32", Shape: []uint64{1}, Values: []float32{6}},
	})

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"reactive", "-json", dir}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("reactive code=%d, want 1; stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("reactive JSON wrote stderr: %s", stderr.String())
	}
	var report rocm.ReactiveSequenceMixerReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("reactive JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Status != "invalid" ||
		report.ExecutionStatus != "load_plan_invalid" ||
		!strings.Contains(strings.Join(report.Notes, "\n"), "0:mixer|self_attn") {
		t.Fatalf("reactive invalid report = %+v, want ambiguous subpath detail", report)
	}
}

func TestMemoryPretrainBuildJSONBuildsArtifacts(t *testing.T) {
	dir := t.TempDir()
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	routerPath := filepath.Join(dir, "memory", "router.json")
	ffnPath := filepath.Join(dir, "memory", "ffn.json")
	clusterIn := filepath.Join(dir, "task.jsonl")
	clusterOut := filepath.Join(dir, "clustered.jsonl")
	if err := os.WriteFile(corpusPath, []byte(
		`{"id":"go-1","text":"Go memory planning","meta":{"source":"docs"}}`+"\n"+
			`{"id":"go-2","text":"Go cgo bridge"}`+"\n"+
			`{"id":"poem-1","text":"winter proof poem"}`+"\n"+
			`{"id":"poem-2","text":"autumn prayer"}`+"\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(corpus) error = %v", err)
	}
	if err := os.WriteFile(clusterIn, []byte(`{"context":"Go memory planning"}`+"\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(cluster input) error = %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"memory-pretrain-build",
		"-json",
		"-corpus", corpusPath,
		"-router", routerPath,
		"-ffn-memory", ffnPath,
		"-hidden-size", "2",
		"-layers", "1",
		"-levels", "1",
		"-tokens", "1",
		"-branching", "2",
		"-depth", "1",
		"-min-cluster-size", "1",
		"-kmeans-iters", "4",
		"-cluster-input", clusterIn,
		"-cluster-output", clusterOut,
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("memory-pretrain-build code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("memory-pretrain-build stderr=%s", stderr.String())
	}
	var report memoryPretrainBuildReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("memory-pretrain-build JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Backend != "rocm" || report.Command != "memory-pretrain-build" || report.CLIContract != cliContractName {
		t.Fatalf("report identity = %+v, want ROCm contract", report)
	}
	if !report.NoPython || report.Embedding != "text-hash" {
		t.Fatalf("report offline mode = %+v, want native text-hash", report)
	}
	if report.Report == nil || report.Report.CorpusRecords != 4 || report.Report.RouterPath != routerPath || report.Report.FFNMemoryPath != ffnPath {
		t.Fatalf("artifact report = %+v, want written router and FFN memory", report.Report)
	}
	if report.Report.ClusterIDReport == nil || report.Report.ClusterIDReport.LearnedRows != 1 {
		t.Fatalf("cluster report = %+v, want learned cluster IDs", report.Report.ClusterIDReport)
	}
	if _, err := memorypretrain.LoadBank(routerPath); err != nil {
		t.Fatalf("LoadBank(routerPath) error = %v", err)
	}
	if _, err := memorypretrain.LoadFFNMemoryBank(ffnPath); err != nil {
		t.Fatalf("LoadFFNMemoryBank(ffnPath) error = %v", err)
	}
	clustered, err := os.ReadFile(clusterOut)
	if err != nil {
		t.Fatalf("ReadFile(clusterOut) error = %v", err)
	}
	if !strings.Contains(string(clustered), `"cluster_ids":[`) {
		t.Fatalf("clustered JSONL = %s, want cluster_ids", string(clustered))
	}
}

func TestMemoryPretrainBuildRequiresCorpus(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"memory-pretrain-build",
		"-json",
		"-hidden-size", "2",
		"-layers", "1",
	}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("memory-pretrain-build code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("memory-pretrain-build wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "corpus path is required") {
		t.Fatalf("stderr missing corpus validation: %s", stderr.String())
	}
}

func TestPackRequiresModelPath(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"pack"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("pack code=%d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "expected exactly one model path") {
		t.Fatalf("stderr missing validation message: %s", stderr.String())
	}
}

func TestPackRejectsUnknownBackendBeforeInspection(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"pack", "-backend", "missing-backend", "/tmp/model"}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("pack code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("pack wrote stdout on backend failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), `backend "missing-backend" is not registered`) {
		t.Fatalf("stderr missing backend validation: %s", stderr.String())
	}
}

func TestPackJSONReportsReactiveModelProfile(t *testing.T) {
	model := t.TempDir()
	writeCLIDraftDetectModelDir(t, model, "qwen3_6_moe")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"pack", "-json", model}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("pack code=%d, want 0; stderr=%s", code, stderr.String())
	}
	var inspection inference.ModelPackInspection
	if err := json.Unmarshal(stdout.Bytes(), &inspection); err != nil {
		t.Fatalf("pack JSON no longer decodes as ModelPackInspection: %v\n%s", err, stdout.String())
	}
	var report packReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("pack JSON did not unmarshal as packReport: %v\n%s", err, stdout.String())
	}
	if report.Backend != defaultBackendName ||
		report.Command != "pack" ||
		report.CLIContract != cliContractName ||
		report.ModelPackInspection == nil ||
		report.Path != model ||
		report.Model.Architecture != "qwen3_6_moe" ||
		report.ModelProfile == nil ||
		report.ModelProfile.Name != "qwen" ||
		report.ModelProfile.Registry != rocm.DefaultROCmModelRegistryName() ||
		report.ModelProfile.EngineFeatures.ChatTemplateID != "qwen" ||
		report.ModelProfile.EngineFeatures.ReasoningParserID != "qwen" ||
		report.ModelProfile.LoadStatus.Status != rocm.ROCmModelLoadStagedNative ||
		report.ModelRoutes == nil ||
		report.ModelRoutes.Contract != rocm.ROCmModelRoutePlanContract ||
		report.ModelRoutes.Architecture != "qwen3_6_moe" ||
		report.ModelRoutes.LoadStatus.Status != rocm.ROCmModelLoadStagedNative ||
		report.ModelRoutes.Labels["engine_route_plan_contract"] != rocm.ROCmModelRoutePlanContract {
		t.Fatalf("pack report = %+v, inspection=%+v, want reactive registry profile beside inspection fields", report, inspection)
	}
}

func TestPackJSONReportsRuntimeLaneBackendInspection(t *testing.T) {
	model := t.TempDir()
	writeCLIDraftDetectModelDir(t, model, "gemma4_text")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"pack", "-json", "-backend", "cuda", model}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("pack code=%d, want 0; stderr=%s", code, stderr.String())
	}
	var report packReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("pack JSON did not unmarshal as packReport: %v\n%s", err, stdout.String())
	}
	if report.Backend != "cuda" ||
		report.ModelPackInspection == nil ||
		report.Labels["backend"] != "cuda" ||
		report.Labels["active_backend"] != defaultBackendName ||
		report.Labels["runtime_lane"] != "cuda" ||
		report.Labels["runtime_dispatch_status"] != "compile_ready_runtime_dispatch_pending" ||
		report.Model.Labels["runtime_lane"] != "cuda" ||
		report.TokenLoop == nil ||
		!report.TokenLoop.IncrementalDecodeReady() ||
		report.TokenLoop.Labels["engine_token_loop_contract"] != rocm.ROCmTokenLoopContract ||
		!strings.Contains(strings.Join(report.Notes, "\n"), "backend-specific runtime dispatch is pending") {
		t.Fatalf("pack report = %+v inspection=%+v, want CUDA lane-aware metadata inspection", report, report.ModelPackInspection)
	}
}

func TestCompatibilityStubParsesFlags(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"menubar", "-addr", ":1"}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("menubar code=%d, want 1", code)
	}
	if !strings.Contains(stderr.String(), cliContractName+" CLI boundary") {
		t.Fatalf("stderr missing compatibility stub: %s", stderr.String())
	}
}

func TestGenerateRejectsInvalidTokenLimit(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-max-tokens=-1", "/tmp/model"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("generate code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "-max-tokens must be >= 0") {
		t.Fatalf("stderr missing token validation: %s", stderr.String())
	}
}

func TestGenerateDefaultTokenLimitUsesBackendDefault(t *testing.T) {
	for _, tt := range []struct {
		name  string
		flags []string
	}{
		{name: "unset"},
		{name: "explicit-zero", flags: []string{"-max-tokens=0"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			backendName := "generate-default-" + strings.NewReplacer("/", "-", " ", "-").Replace(t.Name())
			model := &traceCLITestModel{}
			inference.Register(&traceCLITestBackend{name: backendName, model: model})

			args := []string{"generate", "-backend", backendName, "-draft", "", "-prompt", "hello"}
			args = append(args, tt.flags...)
			args = append(args, "/tmp/model")

			var stdout, stderr bytes.Buffer
			code := runCommand(context.Background(), args, &stdout, &stderr)
			if code != 0 {
				t.Fatalf("generate code=%d, want 0; stderr=%s", code, stderr.String())
			}
			if stderr.Len() != 0 {
				t.Fatalf("generate stderr=%s", stderr.String())
			}
			if !strings.Contains(stdout.String(), "trace ok") {
				t.Fatalf("generate stdout missing model output: %s", stdout.String())
			}
			if len(model.calls) != 1 {
				t.Fatalf("generate model calls=%d, want 1", len(model.calls))
			}
			if got, want := model.calls[0].MaxTokens, inference.DefaultGenerateConfig().MaxTokens; got != want {
				t.Fatalf("generate MaxTokens=%d, want backend default %d", got, want)
			}
			if got := model.calls[0].Temperature; got != 0 {
				t.Fatalf("generate Temperature=%g, want greedy production default", got)
			}
		})
	}
}

func TestGeneratePrintsAttachedDrafterMetrics(t *testing.T) {
	var stdout bytes.Buffer
	model := &generateAttachedDrafterMetricsTestModel{
		traceCLITestModel: &traceCLITestModel{},
		metrics: &rocm.AttachedDrafterMetrics{
			AcceptedTokens: 6,
			ProposedTokens: 24,
			VerifyCalls:    12,
			AcceptanceRate: 0.25,
		},
	}

	printGenerateAttachedDrafterMetrics(&stdout, model)

	want := "mtp: 25% acceptance (6/24 drafted) over 12 verify forwards\n"
	if stdout.String() != want {
		t.Fatalf("MTP metrics line = %q, want %q", stdout.String(), want)
	}
}

func TestKVCacheModeResolverAcceptsDirectNativeModes(t *testing.T) {
	for _, raw := range []string{"fp16", "q8", "kq8vq4", "k-q8-v-q4"} {
		decision := resolveROCmCLIKVCacheMode("generate", raw)
		if decision.Code != 0 || decision.Message != "" || decision.Mode == "" {
			t.Fatalf("resolve -kv-cache %q = %+v, want accepted direct native mode", raw, decision)
		}
		if rocmCLIKVCacheModeStatus(decision.Mode) != "native_device_kv_bound" {
			t.Fatalf("mode %q status = %q, want bound", decision.Mode, rocmCLIKVCacheModeStatus(decision.Mode))
		}
		cfg := rocmCLILoadConfigForKVCache(decision.Mode)
		if cfg.CacheMode != decision.Mode || cfg.DeviceKVMode != decision.Mode || !rocmCLILoadConfigActive(cfg) {
			t.Fatalf("load config for %q = %+v, want active canonical mode", decision.Mode, cfg)
		}
	}
}

func TestKVStorageResolverMapsFP16ToNativeLoad(t *testing.T) {
	for _, raw := range []string{"fp16", "f16"} {
		decision := resolveROCmCLIKVStorageMode("generate", raw)
		if decision.Code != 0 || decision.Message != "" || decision.Mode != "fp16" {
			t.Fatalf("resolve -kv-storage %q = %+v, want fp16 native storage", raw, decision)
		}
		loadMode := resolveROCmCLILoadModeForCacheAndStorage("generate", "", decision.Mode)
		if loadMode.Code != 0 || loadMode.Mode != "fp16" {
			t.Fatalf("load mode for storage %q = %+v, want fp16", raw, loadMode)
		}
		cfg := rocmCLILoadConfigForKVCache(loadMode.Mode)
		if cfg.CacheMode != "fp16" || cfg.DeviceKVMode != "fp16" || !rocmCLILoadConfigActive(cfg) {
			t.Fatalf("load config for fp16 storage = %+v, want active fp16 device-KV", cfg)
		}
	}
}

func TestKVStorageAndCacheRejectConflicts(t *testing.T) {
	decision := resolveROCmCLILoadModeForCacheAndStorage("generate", "q8", "fp16")
	if decision.Code != 2 ||
		!strings.Contains(decision.Message, "-kv-storage fp16 conflicts with -kv-cache q8") {
		t.Fatalf("load mode conflict = %+v, want validation refusal", decision)
	}
}

func TestGenerateRejectsPlannedKVCacheUntilNativeLoadBinding(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-kv-cache", "turboquant", "/tmp/model"}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("generate code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate wrote stdout on cache-mode refusal: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "-kv-cache turboquant is recognized") ||
		!strings.Contains(stderr.String(), "refusing backend-default fallback") {
		t.Fatalf("stderr missing cache-mode refusal: %s", stderr.String())
	}
}

func TestGenerateRejectsUnboundKVStorageUntilNativeLoadBinding(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-draft", "", "-kv-storage", "bf16", "/tmp/model"}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("generate code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate wrote stdout on storage refusal: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "-kv-storage bf16 is recognized") ||
		!strings.Contains(stderr.String(), "refusing backend-default fallback") {
		t.Fatalf("stderr missing storage refusal: %s", stderr.String())
	}
}

func TestGenerateRejectsConflictingKVStorageAndCache(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-draft", "", "-kv-cache", "q8", "-kv-storage", "fp16", "/tmp/model"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("generate code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate wrote stdout on storage/cache conflict: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "-kv-storage fp16 conflicts with -kv-cache q8") {
		t.Fatalf("stderr missing storage/cache conflict: %s", stderr.String())
	}
}

func TestGenerateRejectsSerialPipelineUntilRuntimeGate(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-draft", "", "-pipeline=false", "/tmp/model"}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("generate code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate wrote stdout on pipeline refusal: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "-pipeline=false is recognized") ||
		!strings.Contains(stderr.String(), "refusing backend-default fallback") {
		t.Fatalf("stderr missing pipeline refusal: %s", stderr.String())
	}
}

func TestGenerateRejectsUnknownKVCacheMode(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-kv-cache", "mystery", "/tmp/model"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("generate code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), `unsupported -kv-cache "mystery"`) ||
		!strings.Contains(stderr.String(), "kq8vq4") {
		t.Fatalf("stderr missing cache-mode validation: %s", stderr.String())
	}
}

func TestGenerateRejectsStagedRegistryProfileBeforeRuntimeLoad(t *testing.T) {
	model := t.TempDir()
	writeCLIDraftDetectModelDir(t, model, "qwen3_6_moe")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-draft", "", "-prompt", "hi", model}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("generate code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate wrote stdout on staged model-load refusal: %s", stdout.String())
	}
	for _, want := range []string{
		"generate: load:",
		"qwen3_6_moe",
		"engine_load_status=staged_native",
		"engine_load_target=standalone",
		"engine_load_text_generate=false",
		"standalone generation remains pending",
	} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("stderr missing %q:\n%s", want, stderr.String())
		}
	}
}

func TestGenerateStateWakesAndSleepsNamedStore(t *testing.T) {
	backendName := "state-cli-" + strings.NewReplacer("/", "-", " ", "-").Replace(t.Name())
	model := &stateCLITestModel{}
	inference.Register(&stateCLITestBackend{name: backendName, model: model})
	storePath := filepath.Join(t.TempDir(), "agent.kv")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"generate",
		"-backend", backendName,
		"-state", "resume",
		"-state-store", storePath,
		"-prompt", "first turn",
		"/tmp/model",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("first generate code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("first generate stderr=%s", stderr.String())
	}
	if !strings.Contains(stdout.String(), "turn: fresh state") || !strings.Contains(stdout.String(), "slept 12 tokens -> 3 blocks") {
		t.Fatalf("first stdout missing state report: %s", stdout.String())
	}
	if model.wakeCalls != 0 || model.sleepCalls != 1 {
		t.Fatalf("after first turn wake=%d sleep=%d", model.wakeCalls, model.sleepCalls)
	}

	stdout.Reset()
	stderr.Reset()
	code = runCommand(context.Background(), []string{
		"generate",
		"-backend", backendName,
		"-state", "resume",
		"-state-store", storePath,
		"-think",
		"-prompt", "second turn",
		"/tmp/model",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("second generate code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("second generate stderr=%s", stderr.String())
	}
	if !strings.Contains(stdout.String(), "turn: woke 8 prefix tokens") || !strings.Contains(stdout.String(), "state: resume") {
		t.Fatalf("second stdout missing wake report: %s", stdout.String())
	}
	if model.wakeCalls != 1 || model.sleepCalls != 2 {
		t.Fatalf("after second turn wake=%d sleep=%d", model.wakeCalls, model.sleepCalls)
	}
	if len(model.prompts) != 2 || model.prompts[0] != "first turn" || model.prompts[1] != "second turn" {
		t.Fatalf("prompts=%q", model.prompts)
	}
	if len(model.chatConfigs) != 2 ||
		model.chatConfigs[0].EnableThinking == nil || *model.chatConfigs[0].EnableThinking ||
		model.chatConfigs[1].EnableThinking == nil || !*model.chatConfigs[1].EnableThinking {
		t.Fatalf("state chat configs = %+v, want default thinking off then explicit -think on", model.chatConfigs)
	}
}

func TestGenerateDefaultsToStateForROCmBackend(t *testing.T) {
	original, hadOriginal := inference.Get(defaultBackendName)
	model := &stateCLITestModel{}
	inference.Register(&stateCLITestBackend{name: defaultBackendName, model: model})
	t.Cleanup(func() {
		if hadOriginal {
			inference.Register(original)
		}
	})
	storePath := filepath.Join(t.TempDir(), "agent.kv")
	modelPath := "/tmp/model"
	stateName := defaultROCmGenerateStateName(modelPath)

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"generate",
		"-state-store", storePath,
		"-prompt", "production default",
		modelPath,
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("generate code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("generate stderr=%s", stderr.String())
	}
	if !strings.Contains(stdout.String(), "turn: fresh state") ||
		!strings.Contains(stdout.String(), "state: "+stateName+" ("+storePath+")") {
		t.Fatalf("stdout missing default retained state report: %s", stdout.String())
	}
	if model.wakeCalls != 0 || model.sleepCalls != 1 {
		t.Fatalf("default state wake=%d sleep=%d, want fresh sleep", model.wakeCalls, model.sleepCalls)
	}
	if len(model.prompts) != 1 || model.prompts[0] != "production default" {
		t.Fatalf("prompts=%q", model.prompts)
	}
}

func TestGenerateExplicitEmptyStateDisablesStateStore(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-state", "", "-state-store", filepath.Join(t.TempDir(), "agent.kv"), "/tmp/model"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("generate code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "-state-store requires a non-empty -state") {
		t.Fatalf("stderr missing state-store validation: %s", stderr.String())
	}
}

func TestGenerateTraceReportsAggregateLanes(t *testing.T) {
	backendName := "trace-cli-" + strings.NewReplacer("/", "-", " ", "-").Replace(t.Name())
	model := &traceCLITestModel{}
	inference.Register(&traceCLITestBackend{name: backendName, model: model})

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"generate",
		"-backend", backendName,
		"-trace",
		"-prompt", "trace this",
		"-max-tokens", "2",
		"-temp", "0.7",
		"-think",
		"/tmp/model",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("generate trace code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("generate trace stderr=%s", stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"greedy (temp=0)",
		"sampled (temp=0.7)",
		"prefill",
		"decode",
		"phase trace: ROCm currently exposes aggregate prefill/decode timing",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("generate trace stdout missing %q:\n%s", want, out)
		}
	}
	if len(model.calls) != 3 {
		t.Fatalf("trace model calls=%d, want warm + 2 lanes", len(model.calls))
	}
	if model.calls[0].Temperature != 0 || model.calls[1].Temperature != 0 || model.calls[2].Temperature != 0.7 {
		t.Fatalf("trace temperatures=%v, want warm/greedy/sampled", []float32{model.calls[0].Temperature, model.calls[1].Temperature, model.calls[2].Temperature})
	}
	for index, cfg := range model.calls {
		if cfg.EnableThinking == nil || !*cfg.EnableThinking {
			t.Fatalf("trace call %d config = %+v, want -think propagated to every lane", index, cfg)
		}
	}
	if model.resets != 3 {
		t.Fatalf("trace resets=%d, want reset after warm and both lanes", model.resets)
	}
}

type traceCLITestBackend struct {
	name  string
	model *traceCLITestModel
}

func (b *traceCLITestBackend) Name() string {
	return b.name
}

func (b *traceCLITestBackend) Available() bool {
	return true
}

func (b *traceCLITestBackend) LoadModel(string, ...inference.LoadOption) (inference.TextModel, error) {
	return b.model, nil
}

type traceCLITestModel struct {
	calls    []inference.GenerateConfig
	messages [][]inference.Message
	metrics  inference.GenerateMetrics
	resets   int
	closed   bool
}

type generateAttachedDrafterMetricsTestModel struct {
	*traceCLITestModel
	metrics *rocm.AttachedDrafterMetrics
}

type tuneStatelessCLITestModel struct {
	*traceCLITestModel
	chatCalls      int
	statelessCalls int
}

func (m *generateAttachedDrafterMetricsTestModel) AttachedDrafterMetrics() *rocm.AttachedDrafterMetrics {
	if m == nil || m.metrics == nil {
		return nil
	}
	metrics := *m.metrics
	return &metrics
}

func (m *traceCLITestModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.streamTokens()
}

func (m *traceCLITestModel) Chat(_ context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)
	m.messages = append(m.messages, cloneInferenceMessages(messages))
	m.calls = append(m.calls, cfg)
	m.metrics = inference.GenerateMetrics{
		PromptTokens:        3,
		GeneratedTokens:     minPositive(cfg.MaxTokens, 2),
		PrefillDuration:     2 * time.Millisecond,
		DecodeDuration:      4 * time.Millisecond,
		TotalDuration:       6 * time.Millisecond,
		PrefillTokensPerSec: 1500,
		DecodeTokensPerSec:  500,
		ActiveMemoryBytes:   64 << 20,
		PeakMemoryBytes:     128 << 20,
	}
	return m.streamTokens()
}

func cloneInferenceMessages(messages []inference.Message) []inference.Message {
	if len(messages) == 0 {
		return nil
	}
	out := make([]inference.Message, len(messages))
	copy(out, messages)
	return out
}

func (m *tuneStatelessCLITestModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.chatCalls++
	return m.traceCLITestModel.Chat(ctx, messages, opts...)
}

func (m *tuneStatelessCLITestModel) ChatStateless(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.statelessCalls++
	return m.traceCLITestModel.Chat(ctx, messages, opts...)
}

func (m *traceCLITestModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (m *traceCLITestModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (m *traceCLITestModel) ModelType() string {
	return "trace-cli-test"
}

func (m *traceCLITestModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "trace-cli-test"}
}

func (m *traceCLITestModel) Metrics() inference.GenerateMetrics {
	return m.metrics
}

func (m *traceCLITestModel) Err() error {
	return nil
}

func (m *traceCLITestModel) Close() error {
	m.closed = true
	return nil
}

func (m *traceCLITestModel) ResetState() error {
	m.resets++
	return nil
}

func (m *traceCLITestModel) streamTokens() iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		if !yield(inference.Token{Text: "trace"}) {
			return
		}
		_ = yield(inference.Token{Text: " ok"})
	}
}

type stateCLITestBackend struct {
	name  string
	model *stateCLITestModel
}

func (b *stateCLITestBackend) Name() string {
	return b.name
}

func (b *stateCLITestBackend) Available() bool {
	return true
}

func (b *stateCLITestBackend) LoadModel(string, ...inference.LoadOption) (inference.TextModel, error) {
	return b.model, nil
}

type stateCLITestModel struct {
	prompts     []string
	chatConfigs []inference.GenerateConfig
	wakeCalls   int
	sleepCalls  int
	closed      bool
}

func (m *stateCLITestModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		_ = yield(inference.Token{Text: "ok"})
	}
}

func (m *stateCLITestModel) Chat(_ context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	if len(messages) > 0 {
		m.prompts = append(m.prompts, messages[len(messages)-1].Content)
	}
	m.chatConfigs = append(m.chatConfigs, inference.ApplyGenerateOpts(opts))
	return func(yield func(inference.Token) bool) {
		if !yield(inference.Token{Text: "state"}) {
			return
		}
		_ = yield(inference.Token{Text: " ok"})
	}
}

func (m *stateCLITestModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (m *stateCLITestModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (m *stateCLITestModel) ModelType() string {
	return "state-cli-test"
}

func (m *stateCLITestModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "state-cli-test"}
}

func (m *stateCLITestModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{}
}

func (m *stateCLITestModel) Err() error {
	return nil
}

func (m *stateCLITestModel) Close() error {
	m.closed = true
	return nil
}

func (m *stateCLITestModel) WakeState(_ context.Context, req inference.AgentMemoryWakeRequest) (*inference.AgentMemoryWakeResult, error) {
	m.wakeCalls++
	return &inference.AgentMemoryWakeResult{
		Entry: inference.AgentMemoryRef{
			URI:        req.EntryURI,
			IndexURI:   req.IndexURI,
			TokenCount: 8,
		},
		PrefixTokens: 8,
		BlocksRead:   2,
	}, nil
}

func (m *stateCLITestModel) SleepState(ctx context.Context, req inference.AgentMemorySleepRequest) (*inference.AgentMemorySleepResult, error) {
	m.sleepCalls++
	writer, ok := req.Store.(state.Writer)
	if !ok {
		return nil, state.ErrChunkNotFound
	}
	if _, err := writer.Put(ctx, `{"kind":"state-cli-test-index"}`, state.PutOptions{
		URI:   req.IndexURI,
		Title: req.Title,
		Kind:  "rocm-state-index",
	}); err != nil {
		return nil, err
	}
	return &inference.AgentMemorySleepResult{
		Entry: inference.AgentMemoryRef{
			URI:        req.EntryURI,
			IndexURI:   req.IndexURI,
			Title:      req.Title,
			TokenCount: 12,
		},
		TokenCount:    12,
		BlocksWritten: 3,
	}, nil
}

func TestGenerateReactiveDraftFlagsReportBoundary(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-draft", "/tmp/assistant", "/tmp/model"}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("generate code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate wrote stdout for unimplemented draft boundary: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "reactive MTP drafter resolved") || !strings.Contains(stderr.String(), "/tmp/assistant") {
		t.Fatalf("stderr missing draft boundary: %s", stderr.String())
	}
}

func TestGenerateAutoDraftReportsResolvedBoundary(t *testing.T) {
	model := t.TempDir()
	writeCLIDraftDetectModelDir(t, model, "gemma4_text")
	writeCLIDraftDetectModelDir(t, filepath.Join(model, "assistant"), "gemma4_assistant")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-max-tokens=1", model}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("generate code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate wrote stdout for native-pending MTP boundary: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "reactive MTP drafter resolved for state") ||
		!strings.Contains(stderr.String(), "auto-detected assistant/") ||
		!strings.Contains(stderr.String(), "native ROCm retained-state drafter execution is pending") {
		t.Fatalf("stderr missing auto-draft detection: %s", stderr.String())
	}
}

func TestGenerateAutoDraftRefusesSamplingBeforeLoad(t *testing.T) {
	model := t.TempDir()
	writeCLIDraftDetectModelDir(t, model, "gemma4_text")
	writeCLIDraftDetectModelDir(t, filepath.Join(model, "assistant"), "gemma4_assistant")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-temp=1", "-max-tokens=1", model}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("generate code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate wrote stdout for sampled MTP refusal: %s", stdout.String())
	}
	for _, want := range []string{"reactive MTP drafter resolved for state", "requires greedy generation", "-temp 0"} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("stderr missing %q:\n%s", want, stderr.String())
		}
	}
}

func TestGenerateExplicitDraftRefusesSamplingBeforeLoad(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-state", "", "-draft", "/tmp/assistant", "-temp=1", "/tmp/model"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("generate code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate wrote stdout for sampled explicit MTP refusal: %s", stdout.String())
	}
	for _, want := range []string{"reactive MTP drafter resolved for generate", "requires greedy generation", "-temp 0"} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("stderr missing %q:\n%s", want, stderr.String())
		}
	}
}

func TestGenerateTraceRefusesAutoDraftInsteadOfIgnoring(t *testing.T) {
	model := t.TempDir()
	writeCLIDraftDetectModelDir(t, model, "gemma4_text")
	writeCLIDraftDetectModelDir(t, filepath.Join(model, "assistant"), "gemma4_assistant")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-trace", "-max-tokens=1", model}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("generate trace code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate trace wrote stdout for native-pending MTP boundary: %s", stdout.String())
	}
	for _, want := range []string{"reactive MTP drafter resolved for trace", "auto-detected assistant/", "use -draft ''"} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("trace stderr missing %q:\n%s", want, stderr.String())
		}
	}
}

func TestGenerateStateAttemptsAutoDraftRetainedNativePath(t *testing.T) {
	model := t.TempDir()
	writeCLIDraftDetectModelDir(t, model, "gemma4_text")
	writeCLIDraftDetectModelDir(t, filepath.Join(model, "assistant"), "gemma4_assistant")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"generate",
		"-state", "resume",
		"-state-store", filepath.Join(t.TempDir(), "agent.kv"),
		"-max-tokens=1",
		model,
	}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("generate state code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate state wrote stdout for native-pending MTP boundary: %s", stdout.String())
	}
	for _, want := range []string{"reactive MTP drafter resolved for state", "auto-detected assistant/", "native ROCm retained-state drafter execution is pending"} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("state stderr missing %q:\n%s", want, stderr.String())
		}
	}
	if strings.Contains(stderr.String(), "use -draft ''") {
		t.Fatalf("state stderr still reports old pre-load refusal:\n%s", stderr.String())
	}
}

func TestGenerateRejectsInvalidDraftBlock(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-draft-block=-1", "/tmp/model"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("generate code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("generate wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "-draft-block must be >= 0") {
		t.Fatalf("stderr missing draft-block validation: %s", stderr.String())
	}
}

func TestServeRejectsPlannedKVCacheUntilNativeLoadBinding(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"serve", "-kv-cache", "paged"}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("serve code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("serve wrote stdout on cache-mode refusal: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "-kv-cache paged is recognized") ||
		!strings.Contains(stderr.String(), "refusing backend-default fallback") {
		t.Fatalf("stderr missing serve cache-mode refusal: %s", stderr.String())
	}
}

func TestServeResolverCarriesDirectKVCacheMode(t *testing.T) {
	cfg := rocmServeConfig{
		Backend:        defaultBackendName,
		ModelPath:      "/tmp/model",
		KVCacheMode:    "q8",
		ROCmLoadConfig: rocmCLILoadConfigForKVCache("q8"),
	}
	resolver := newROCmServeResolver(cfg)
	status := resolver.Status()
	if status.KVCacheMode != "q8" {
		t.Fatalf("status KVCacheMode = %q, want q8", status.KVCacheMode)
	}
	labels := rocmServeLabelsForStatus(cfg, status)
	if labels["kv_cache"] != "q8" {
		t.Fatalf("serve labels = %+v, want q8 kv_cache label", labels)
	}
	if err := validateROCmCLILoadConfigBackend("serve", "cuda", cfg.ROCmLoadConfig); err == nil ||
		!strings.Contains(err.Error(), "cannot be used with -backend cuda") {
		t.Fatalf("backend validation err = %v, want CUDA rejection", err)
	}
}

func TestServeResolverStatusUsesLoadedModelRoutePlan(t *testing.T) {
	modelPath := "/models/gemma4-e4b-q6"
	cfg := rocmServeConfig{
		Backend:   defaultBackendName,
		ModelPath: modelPath,
	}
	resolver := newROCmServeResolver(cfg)
	resolver.model = &serveReactiveTestModel{
		modelIdentity: inference.ModelIdentity{
			Path:         modelPath,
			Architecture: "gemma4_text",
			QuantBits:    6,
			Labels: map[string]string{
				"gemma4_quant_mode": "q6",
				"gemma4_size":       "E4B",
			},
		},
		cacheProfile: rocmmodel.CacheProfile{
			Contract:        rocmmodel.CacheProfileContract,
			Architecture:    "gemma4_text",
			TotalCaches:     2,
			QuantizedCaches: 2,
			MaxCacheTokens:  37,
		},
	}

	status := resolver.Status()
	if status.ModelProfile == nil || !status.ModelProfile.Matched() {
		t.Fatalf("status ModelProfile = %+v, want loaded model profile", status.ModelProfile)
	}
	if status.ModelRoutes == nil || !status.ModelRoutes.Matched() {
		t.Fatalf("status ModelRoutes = %+v, want loaded model route plan", status.ModelRoutes)
	}
	if status.ModelRoutes.CacheProfile.MaxCacheTokens != 37 ||
		status.ModelRoutes.Labels["engine_route_plan_cache_profile_max_cache_tokens"] != "37" {
		t.Fatalf("status ModelRoutes = %+v, want live cache profile overlay", status.ModelRoutes)
	}

	labels := rocmServeLabelsForStatus(cfg, status)
	if labels["engine_route_plan_cache_profile_max_cache_tokens"] != "37" ||
		labels["engine_route_plan_cache_profile_quantized_count"] != "2" {
		t.Fatalf("serve labels = %+v, want status route-plan live cache labels", labels)
	}

	reply := rocmServeStatusResponse(cfg, status)
	if reply.ModelRoutes == nil || reply.ModelRoutes.CacheProfile.MaxCacheTokens != 37 ||
		reply.Labels["engine_route_plan_cache_profile_max_cache_tokens"] != "37" {
		t.Fatalf("status reply = %+v labels=%+v, want loaded route plan propagated", reply.ModelRoutes, reply.Labels)
	}
}

func TestServeResolverReloadPreservesBootLoadConfig(t *testing.T) {
	cfg := rocmServeConfig{
		Backend:        defaultBackendName,
		ModelPath:      "/tmp/boot-model",
		ContextLen:     4096,
		KVCacheMode:    "q8",
		ROCmLoadConfig: rocmCLILoadConfigForKVCache("q8"),
		LoadOptions: []inference.LoadOption{
			inference.WithContextLen(4096),
			inference.WithParallelSlots(3),
			inference.WithAdapterPath("/adapters/boot"),
		},
	}
	resolver := newROCmServeResolver(cfg)
	resolver.ReloadWithOptions(rocmServeReloadOptions{
		Path:       "/tmp/reloaded-model",
		Backend:    defaultBackendName,
		ContextLen: 8192,
	})

	status := resolver.Status()
	reply := rocmServeStatusResponse(cfg, status)
	if status.KVCacheMode != "q8" ||
		reply.Config.ContextLength != 8192 ||
		reply.Config.ParallelSlots != 3 ||
		reply.Config.KVCache != "q8" ||
		reply.Config.CacheMode != "q8" ||
		reply.Config.DeviceKVMode != "q8" ||
		reply.Config.AdapterPath != "/adapters/boot" {
		t.Fatalf("reload status = %+v config=%+v, want reload overlay preserving boot load config", status, reply.Config)
	}
}

func TestServeRejectsUnknownKVCacheMode(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"serve", "-kv-cache", "bad-cache"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("serve code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("serve wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), `unsupported -kv-cache "bad-cache"`) ||
		!strings.Contains(stderr.String(), "disk-l2") {
		t.Fatalf("stderr missing serve cache-mode validation: %s", stderr.String())
	}
}

func TestGenerateHelp(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-h"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("generate -h code=%d, want 0", code)
	}
	if !strings.Contains(stderr.String(), "Usage:") || !strings.Contains(stderr.String(), "-max-tokens") {
		t.Fatalf("generate help missing flags: %s", stderr.String())
	}
}

func TestServeHandlerHealthModelsAdminReloadAndOllamaTags(t *testing.T) {
	modelRoot := t.TempDir()
	modelPath := filepath.Join(modelRoot, "lemer-lite")
	if err := os.MkdirAll(modelPath, 0o755); err != nil {
		t.Fatalf("create model dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(modelPath, rocmServeSHAManifestFilename), []byte("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  config.json\n"), 0o600); err != nil {
		t.Fatalf("write model manifest: %v", err)
	}
	unsignedPath := filepath.Join(modelRoot, "unsigned")
	if err := os.MkdirAll(unsignedPath, 0o755); err != nil {
		t.Fatalf("create unsigned model dir: %v", err)
	}
	var audit bytes.Buffer
	handler, resolver := newROCmServeHandler(rocmServeConfig{
		Backend:               "rocm",
		DraftPath:             "auto",
		DraftDetect:           true,
		DraftBlock:            5,
		AdminToken:            "test-token",
		NoAutoProfile:         true,
		ProfileDir:            "/tmp/profiles",
		StateStorePath:        "/tmp/state.kv",
		AdminDownloadModelDir: modelRoot,
		AdminAudit:            &audit,
	})
	defer resolver.Close()

	health := httptest.NewRecorder()
	handler.ServeHTTP(health, httptest.NewRequest(http.MethodGet, rocmServeHealthPath, nil))
	if health.Code != http.StatusOK || !strings.Contains(health.Body.String(), `"runtime":"rocm"`) || !strings.Contains(health.Body.String(), `"models":null`) {
		t.Fatalf("health status=%d body=%s, want model-less rocm health", health.Code, health.Body.String())
	}

	unauthorized := httptest.NewRecorder()
	handler.ServeHTTP(unauthorized, httptest.NewRequest(http.MethodGet, rocmServeAdminStatusPath, nil))
	if unauthorized.Code != http.StatusUnauthorized || !strings.Contains(unauthorized.Body.String(), "admin bearer token") {
		t.Fatalf("admin status without token=%d body=%s, want 401", unauthorized.Code, unauthorized.Body.String())
	}
	if unauthorized.Header().Get("WWW-Authenticate") != rocmAdminAuthRealm {
		t.Fatalf("admin status WWW-Authenticate=%q, want %q", unauthorized.Header().Get("WWW-Authenticate"), rocmAdminAuthRealm)
	}
	if !strings.Contains(audit.String(), "admin: auth deny path="+rocmServeAdminStatusPath) {
		t.Fatalf("audit log after unauthorized admin status missing auth deny:\n%s", audit.String())
	}

	machineReq := httptest.NewRequest(http.MethodGet, rocmServeAdminMachinePath, nil)
	machineReq.Header.Set("Authorization", "Bearer test-token")
	machine := httptest.NewRecorder()
	handler.ServeHTTP(machine, machineReq)
	if machine.Code != http.StatusOK {
		t.Fatalf("admin machine=%d body=%s, want 200", machine.Code, machine.Body.String())
	}
	var machineInfo rocmServeAdminMachineResponse
	if err := json.Unmarshal(machine.Body.Bytes(), &machineInfo); err != nil {
		t.Fatalf("admin machine response did not unmarshal: %v\n%s", err, machine.Body.String())
	}
	if machineInfo.Hash == "" ||
		machineInfo.Runtime != "rocm" ||
		machineInfo.CLIContract != cliContractName ||
		machineInfo.GoVersion != runtime.Version() ||
		machineInfo.OS != runtime.GOOS ||
		machineInfo.Arch != runtime.GOARCH ||
		machineInfo.Compile.Current.Backend != "rocm" ||
		machineInfo.Labels["admin_auth"] != "bearer_constant_time" ||
		machineInfo.Labels["admin_auth_audit"] != "deny" ||
		machineInfo.Labels["admin_token_prefix"] != rocmAdminTokenPrefix ||
		machineInfo.Labels["admin_reload"] != "confirmation_sha_manifest_gated" {
		t.Fatalf("admin machine = %+v, want go-mlx-style ROCm machine identity", machineInfo)
	}

	noConfirmBody, err := json.Marshal(map[string]any{
		"model_path": modelPath,
		"backend":    "rocm",
	})
	if err != nil {
		t.Fatalf("no confirm request marshal failed: %v", err)
	}
	noConfirmReq := httptest.NewRequest(http.MethodPost, rocmServeAdminReloadPath, bytes.NewReader(noConfirmBody))
	noConfirmReq.Header.Set("Authorization", "Bearer test-token")
	noConfirm := httptest.NewRecorder()
	handler.ServeHTTP(noConfirm, noConfirmReq)
	if noConfirm.Code != http.StatusBadRequest || !strings.Contains(noConfirm.Body.String(), "confirm_machine required") {
		t.Fatalf("reload missing confirmation=%d body=%s, want confirm_machine required", noConfirm.Code, noConfirm.Body.String())
	}

	badConfirmBody, err := json.Marshal(map[string]any{
		"model_path":      modelPath,
		"backend":         "rocm",
		"confirm_machine": "wrong-machine",
	})
	if err != nil {
		t.Fatalf("bad confirm request marshal failed: %v", err)
	}
	badConfirmReq := httptest.NewRequest(http.MethodPost, rocmServeAdminReloadPath, bytes.NewReader(badConfirmBody))
	badConfirmReq.Header.Set("Authorization", "Bearer test-token")
	badConfirm := httptest.NewRecorder()
	handler.ServeHTTP(badConfirm, badConfirmReq)
	if badConfirm.Code != http.StatusBadRequest || !strings.Contains(badConfirm.Body.String(), "confirm_machine mismatch") {
		t.Fatalf("reload bad confirmation=%d body=%s, want confirm_machine mismatch", badConfirm.Code, badConfirm.Body.String())
	}

	unsignedBody, err := json.Marshal(map[string]any{
		"model_path":      unsignedPath,
		"backend":         "rocm",
		"confirm_machine": machineInfo.Hash,
	})
	if err != nil {
		t.Fatalf("unsigned request marshal failed: %v", err)
	}
	unsignedReq := httptest.NewRequest(http.MethodPost, rocmServeAdminReloadPath, bytes.NewReader(unsignedBody))
	unsignedReq.Header.Set("Authorization", "Bearer test-token")
	unsigned := httptest.NewRecorder()
	handler.ServeHTTP(unsigned, unsignedReq)
	if unsigned.Code != http.StatusBadRequest || !strings.Contains(unsigned.Body.String(), "sha manifest") {
		t.Fatalf("reload unsigned model=%d body=%s, want sha manifest refusal", unsigned.Code, unsigned.Body.String())
	}

	profilePath := "/tmp/profiles/lemer-lite.json"
	adapterPath := "/tmp/adapters/domain"
	reloadBody, err := json.Marshal(map[string]any{
		"model_path":      modelPath,
		"backend":         "rocm",
		"context_length":  8192,
		"confirm_machine": machineInfo.Hash,
		"profile_path":    profilePath,
		"adapter_path":    adapterPath,
	})
	if err != nil {
		t.Fatalf("reload request marshal failed: %v", err)
	}
	reloadReq := httptest.NewRequest(http.MethodPost, rocmServeAdminReloadPath, bytes.NewReader(reloadBody))
	reloadReq.Header.Set("Authorization", "Bearer test-token")
	reload := httptest.NewRecorder()
	handler.ServeHTTP(reload, reloadReq)
	if reload.Code != http.StatusOK || !strings.Contains(reload.Body.String(), `"model_name":"lemer-lite"`) || !strings.Contains(reload.Body.String(), `"loaded":false`) {
		t.Fatalf("reload status=%d body=%s, want lazy reloaded model status", reload.Code, reload.Body.String())
	}
	var reloadStatus rocmServeAdminStatusResponse
	if err := json.Unmarshal(reload.Body.Bytes(), &reloadStatus); err != nil {
		t.Fatalf("reload status response did not unmarshal: %v\n%s", err, reload.Body.String())
	}
	if reloadStatus.Runtime != "rocm" ||
		reloadStatus.CLIContract != cliContractName ||
		reloadStatus.ModelPath != modelPath ||
		reloadStatus.ModelRegistry.Name != rocm.DefaultROCmModelRegistryName() ||
		reloadStatus.ModelRegistry.Labels["engine_profile_reactive"] != "true" ||
		reloadStatus.ProfilePath != profilePath ||
		reloadStatus.AdapterPath != adapterPath ||
		reloadStatus.Config.ContextLength != 8192 ||
		reloadStatus.Config.AdapterPath != adapterPath ||
		reloadStatus.Config.ProfileDir != "/tmp/profiles" ||
		reloadStatus.Config.StateStore != "/tmp/state.kv" ||
		!reloadStatus.Config.DraftDetect ||
		reloadStatus.Config.DraftBlock != 5 ||
		!reloadStatus.Config.NoAutoProfile ||
		reloadStatus.Config.StateConversations ||
		reloadStatus.Memory.ActiveBytes != 0 ||
		reloadStatus.Memory.CacheBytes != 0 ||
		reloadStatus.Memory.PeakBytes != 0 ||
		reloadStatus.Resolver.ModelName != "lemer-lite" ||
		reloadStatus.Resolver.ProfilePath != profilePath ||
		reloadStatus.Resolver.AdapterPath != adapterPath ||
		reloadStatus.Labels["admin_reload_confirm_machine"] != "accepted" ||
		reloadStatus.Labels["admin_reload_adapter_path"] != "load_option" ||
		reloadStatus.Resolver.Loaded {
		t.Fatalf("reload status = %+v, want cross-backend ROCm admin status contract", reloadStatus)
	}
	auditText := audit.String()
	for _, want := range []string{"serve_reload attempt", "confirm_machine required", "confirm_machine mismatch", "sha manifest", "serve_reload success"} {
		if !strings.Contains(auditText, want) {
			t.Fatalf("audit log missing %q:\n%s", want, auditText)
		}
	}

	statusReq := httptest.NewRequest(http.MethodGet, rocmServeAdminStatusPath, nil)
	statusReq.Header.Set("Authorization", "Bearer test-token")
	status := httptest.NewRecorder()
	handler.ServeHTTP(status, statusReq)
	if status.Code != http.StatusOK {
		t.Fatalf("admin status=%d body=%s, want 200", status.Code, status.Body.String())
	}
	var adminStatus rocmServeAdminStatusResponse
	if err := json.Unmarshal(status.Body.Bytes(), &adminStatus); err != nil {
		t.Fatalf("admin status response did not unmarshal: %v\n%s", err, status.Body.String())
	}
	if adminStatus.ModelPath != reloadStatus.ModelPath ||
		adminStatus.ModelRegistry.Name != rocm.DefaultROCmModelRegistryName() ||
		adminStatus.ProfilePath != profilePath ||
		adminStatus.AdapterPath != adapterPath ||
		adminStatus.Runtime != "rocm" ||
		adminStatus.CLIContract != cliContractName ||
		adminStatus.Config.ContextLength != 8192 ||
		adminStatus.Config.AdapterPath != adapterPath ||
		adminStatus.Resolver.ModelName != "lemer-lite" ||
		adminStatus.Resolver.ProfilePath != profilePath ||
		adminStatus.Resolver.AdapterPath != adapterPath ||
		adminStatus.Labels["admin_model_download"] != "ready" {
		t.Fatalf("admin status = %+v, want typed serve/status snapshot after reload", adminStatus)
	}

	models := httptest.NewRecorder()
	handler.ServeHTTP(models, httptest.NewRequest(http.MethodGet, rocmServeModelsPath, nil))
	if models.Code != http.StatusOK || !strings.Contains(models.Body.String(), `"id":"lemer-lite"`) {
		t.Fatalf("models status=%d body=%s, want reloaded model list", models.Code, models.Body.String())
	}

	tags := httptest.NewRecorder()
	handler.ServeHTTP(tags, httptest.NewRequest(http.MethodGet, "/api/tags", nil))
	if tags.Code != http.StatusOK || !strings.Contains(tags.Body.String(), `"name":"lemer-lite"`) {
		t.Fatalf("ollama tags status=%d body=%s, want reloaded model tag", tags.Code, tags.Body.String())
	}
}

func TestServeHandlerReportsReactiveDraftDetection(t *testing.T) {
	model := t.TempDir()
	writeCLIDraftDetectModelDir(t, model, "gemma4_text")
	assistant := filepath.Join(model, "assistant")
	writeCLIDraftDetectModelDir(t, assistant, "gemma4_assistant")
	handler, resolver := newROCmServeHandler(rocmServeConfig{
		Backend:            "rocm",
		ModelPath:          model,
		DraftPath:          "auto",
		DraftDetect:        true,
		DraftBlock:         5,
		AdminToken:         "test-token",
		StateConversations: true,
		StateStorePath:     "/tmp/conversations.kv",
		StateStore:         state.NewInMemoryStore(nil),
	})
	defer resolver.Close()

	health := httptest.NewRecorder()
	handler.ServeHTTP(health, httptest.NewRequest(http.MethodGet, rocmServeHealthPath, nil))
	if health.Code != http.StatusOK ||
		!strings.Contains(health.Body.String(), `"reactive_draft_detection":"active_pending_native_drafter"`) ||
		!strings.Contains(health.Body.String(), `"reactive_draft_source":"assistant-dir"`) ||
		!strings.Contains(health.Body.String(), `"reactive_draft_fallback":"target_retained_decode"`) ||
		!strings.Contains(health.Body.String(), `"openai_chat_retained_state_mtp":"pending_native_drafter"`) ||
		!strings.Contains(health.Body.String(), `"openai_chat_retained_state_runtime_state":"rocm_state_session_runtime_kv"`) ||
		!strings.Contains(health.Body.String(), `"openai_chat_retained_state_prompt_replay_fallback":"forbidden"`) ||
		!strings.Contains(health.Body.String(), assistant) {
		t.Fatalf("health status=%d body=%s, want reactive draft labels", health.Code, health.Body.String())
	}

	statusReq := httptest.NewRequest(http.MethodGet, rocmServeAdminStatusPath, nil)
	statusReq.Header.Set("Authorization", "Bearer test-token")
	status := httptest.NewRecorder()
	handler.ServeHTTP(status, statusReq)
	if status.Code != http.StatusOK ||
		!strings.Contains(status.Body.String(), `"source":"assistant-dir"`) ||
		!strings.Contains(status.Body.String(), `"draft_block":5`) {
		t.Fatalf("status=%d body=%s, want draft detection payload", status.Code, status.Body.String())
	}
	var adminStatus rocmServeAdminStatusResponse
	if err := json.Unmarshal(status.Body.Bytes(), &adminStatus); err != nil {
		t.Fatalf("decode status response: %v\n%s", err, status.Body.String())
	}
	if adminStatus.ModelRegistry.Name != rocm.DefaultROCmModelRegistryName() ||
		adminStatus.ModelRegistry.Backend != defaultBackendName ||
		adminStatus.ModelProfile == nil ||
		adminStatus.ModelProfile.Registry != rocm.DefaultROCmModelRegistryName() ||
		adminStatus.ModelProfile.Architecture != "gemma4_text" ||
		adminStatus.ModelRoutes == nil ||
		adminStatus.ModelRoutes.Contract != rocm.ROCmModelRoutePlanContract ||
		adminStatus.ModelRoutes.Architecture != "gemma4_text" ||
		adminStatus.TokenLoop == nil ||
		!adminStatus.TokenLoop.IncrementalDecodeReady() ||
		adminStatus.Resolver.ModelProfile == nil ||
		adminStatus.Resolver.ModelProfile.Architecture != "gemma4_text" ||
		adminStatus.Resolver.ModelRoutes == nil ||
		adminStatus.Resolver.ModelRoutes.Architecture != "gemma4_text" ||
		adminStatus.Resolver.TokenLoop == nil ||
		adminStatus.Resolver.TokenLoop.Labels["engine_token_loop_contract"] != rocm.ROCmTokenLoopContract ||
		adminStatus.Labels["engine_profile"] != "gemma4" ||
		adminStatus.Labels["engine_feature_chat_template_id"] != "gemma4_hf_turn" ||
		adminStatus.Labels["engine_route_plan_contract"] != rocm.ROCmModelRoutePlanContract ||
		adminStatus.Labels["engine_token_loop_incremental_ready"] != "true" ||
		adminStatus.Labels["engine_route_plan_feature"] != "true" ||
		adminStatus.Labels["openai_chat_state_continuity"] != rocmServeStateContinuityStoreReady ||
		adminStatus.Labels["openai_chat_retained_state_mtp"] != rocmServeOpenAIStateMTPPendingNative {
		t.Fatalf("admin status = %+v labels=%+v, want reactive registry/profile metadata", adminStatus, adminStatus.Labels)
	}

	models := httptest.NewRecorder()
	handler.ServeHTTP(models, httptest.NewRequest(http.MethodGet, rocmServeModelsPath, nil))
	if models.Code != http.StatusOK {
		t.Fatalf("models status=%d body=%s, want public model list", models.Code, models.Body.String())
	}
	var modelsReply struct {
		Object string `json:"object"`
		Data   []struct {
			ID            string                        `json:"id"`
			Object        string                        `json:"object"`
			OwnedBy       string                        `json:"owned_by"`
			Metadata      map[string]any                `json:"metadata"`
			Labels        map[string]string             `json:"labels"`
			ModelProfile  *rocm.ROCmModelProfile        `json:"model_profile,omitempty"`
			ModelRoutes   *rocm.ROCmModelRoutePlan      `json:"model_routes,omitempty"`
			TokenLoop     *rocm.ROCmTokenLoopStatus     `json:"token_loop,omitempty"`
			RetainedState *rocm.ROCmRetainedStateStatus `json:"retained_state,omitempty"`
		} `json:"data"`
	}
	if err := json.Unmarshal(models.Body.Bytes(), &modelsReply); err != nil {
		t.Fatalf("decode models response: %v\n%s", err, models.Body.String())
	}
	if modelsReply.Object != "list" ||
		len(modelsReply.Data) != 1 ||
		modelsReply.Data[0].ID != filepath.Base(model) ||
		modelsReply.Data[0].OwnedBy != defaultBackendName ||
		modelsReply.Data[0].Metadata["draft_block"].(float64) != 5 ||
		modelsReply.Data[0].Labels["openai_chat_retained_state_mtp"] != rocmServeOpenAIStateMTPPendingNative ||
		modelsReply.Data[0].Labels["openai_chat_retained_state_prompt_replay_fallback"] != "forbidden" ||
		modelsReply.Data[0].ModelProfile == nil ||
		modelsReply.Data[0].ModelProfile.Architecture != "gemma4_text" ||
		modelsReply.Data[0].ModelRoutes == nil ||
		modelsReply.Data[0].ModelRoutes.Contract != rocm.ROCmModelRoutePlanContract ||
		modelsReply.Data[0].TokenLoop == nil ||
		!modelsReply.Data[0].TokenLoop.IncrementalDecodeReady() ||
		modelsReply.Data[0].Labels["engine_token_loop_contract"] != rocm.ROCmTokenLoopContract ||
		modelsReply.Data[0].RetainedState == nil ||
		!modelsReply.Data[0].RetainedState.RuntimeOwnedDecodeReady() ||
		!modelsReply.Data[0].RetainedState.PromptReplayRefused {
		t.Fatalf("models response = %+v, want public retained-state model metadata", modelsReply)
	}
}

func TestServeHandlerConversationContinuityWakesStoredPrefix(t *testing.T) {
	backendName := "serve-state-" + strings.NewReplacer("/", "-", " ", "-").Replace(t.Name())
	model := &stateCLITestModel{}
	inference.Register(&stateCLITestBackend{name: backendName, model: model})
	handler, resolver := newROCmServeHandler(rocmServeConfig{
		Backend:            backendName,
		ModelPath:          "/tmp/state-serve-model",
		AdminToken:         "test-token",
		StateConversations: true,
		StateStorePath:     "/tmp/conversations.kv",
		StateStore:         state.NewInMemoryStore(nil),
	})
	defer resolver.Close()

	first := httptest.NewRecorder()
	handler.ServeHTTP(first, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{
		"model":"state",
		"messages":[{"role":"user","content":"first turn"}]
	}`)))
	if first.Code != http.StatusOK || !strings.Contains(first.Body.String(), `"content":"state ok"`) {
		t.Fatalf("first chat status=%d body=%s, want non-streaming state reply", first.Code, first.Body.String())
	}
	if model.wakeCalls != 0 || model.sleepCalls != 1 {
		t.Fatalf("after first chat wake=%d sleep=%d, want fresh sleep", model.wakeCalls, model.sleepCalls)
	}

	second := httptest.NewRecorder()
	handler.ServeHTTP(second, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{
		"model":"state",
		"messages":[
			{"role":"user","content":"first turn"},
			{"role":"assistant","content":"state ok"},
			{"role":"user","content":"second turn"}
		]
	}`)))
	if second.Code != http.StatusOK || !strings.Contains(second.Body.String(), `"content":"state ok"`) {
		t.Fatalf("second chat status=%d body=%s, want non-streaming state reply", second.Code, second.Body.String())
	}
	if model.wakeCalls != 1 || model.sleepCalls != 2 {
		t.Fatalf("after second chat wake=%d sleep=%d, want stored wake plus sleep", model.wakeCalls, model.sleepCalls)
	}
	if len(model.prompts) != 2 || model.prompts[0] != "first turn" || model.prompts[1] != "second turn" {
		t.Fatalf("prompts=%q, want second request to send only live tail after wake", model.prompts)
	}

	statusReq := httptest.NewRequest(http.MethodGet, rocmServeAdminStatusPath, nil)
	statusReq.Header.Set("Authorization", "Bearer test-token")
	status := httptest.NewRecorder()
	handler.ServeHTTP(status, statusReq)
	if status.Code != http.StatusOK {
		t.Fatalf("admin status=%d body=%s, want 200", status.Code, status.Body.String())
	}
	var reply rocmServeAdminStatusResponse
	if err := json.Unmarshal(status.Body.Bytes(), &reply); err != nil {
		t.Fatalf("admin status response did not unmarshal: %v\n%s", err, status.Body.String())
	}
	if reply.Config.StateContinuity != rocmServeStateContinuityConversationStore ||
		reply.Resolver.StateStats.StoreWakes != 1 ||
		reply.Resolver.StateStats.Sleeps != 2 ||
		reply.Labels["reactive_state_continuity"] != rocmServeStateContinuityConversationStore {
		t.Fatalf("admin status = %+v, want active conversation continuity stats", reply)
	}
}

func TestServeAdminReloadAppliesTuningProfileLoadConfig(t *testing.T) {
	modelRoot := t.TempDir()
	modelPath := filepath.Join(modelRoot, "profiled")
	if err := os.MkdirAll(modelPath, 0o755); err != nil {
		t.Fatalf("create model dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(modelPath, rocmServeSHAManifestFilename), []byte("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  config.json\n"), 0o600); err != nil {
		t.Fatalf("write model manifest: %v", err)
	}
	profilePath := filepath.Join(t.TempDir(), "profile.json")
	if err := writeROCmTuningProfile(profilePath, inference.TuningProfile{
		Candidate: inference.TuningCandidate{
			ContextLength:        32768,
			ParallelSlots:        2,
			PromptCache:          true,
			PromptCacheMinTokens: 512,
			CachePolicy:          "stateful",
			CacheMode:            "k-q8-v-q4",
			BatchSize:            4,
			PrefillChunkSize:     2048,
			ExpectedQuantization: 4,
			MemoryLimitBytes:     16 << 30,
			CacheLimitBytes:      8 << 30,
			WiredLimitBytes:      4 << 30,
			Adapter:              inference.AdapterIdentity{Path: "/adapters/from-profile"},
		},
		CreatedAtUnix: 20,
	}); err != nil {
		t.Fatal(err)
	}

	handler, resolver := newROCmServeHandler(rocmServeConfig{
		Backend:               "rocm",
		AdminToken:            "test-token",
		AdminDownloadModelDir: modelRoot,
	})
	defer resolver.Close()
	machineInfo, err := buildROCmServeAdminMachineResponse(context.Background(), rocmServeConfig{Backend: "rocm"})
	if err != nil {
		t.Fatalf("machine hash: %v", err)
	}
	body, err := json.Marshal(map[string]any{
		"model_path":      modelPath,
		"backend":         "rocm",
		"confirm_machine": machineInfo.Hash,
		"profile_path":    profilePath,
		"context_length":  8192,
		"adapter_path":    "/adapters/explicit",
	})
	if err != nil {
		t.Fatalf("reload marshal: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, rocmServeAdminReloadPath, bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-token")
	reload := httptest.NewRecorder()
	handler.ServeHTTP(reload, req)
	if reload.Code != http.StatusOK {
		t.Fatalf("reload status=%d body=%s, want 200", reload.Code, reload.Body.String())
	}
	var status rocmServeAdminStatusResponse
	if err := json.Unmarshal(reload.Body.Bytes(), &status); err != nil {
		t.Fatalf("reload status response did not unmarshal: %v\n%s", err, reload.Body.String())
	}
	cfg := status.Config
	if cfg.ContextLength != 8192 ||
		cfg.ParallelSlots != 2 ||
		!cfg.PromptCache ||
		cfg.PromptCacheMinTokens != 512 ||
		cfg.CachePolicy != "stateful" ||
		cfg.KVCache != "k-q8-v-q4" ||
		cfg.CacheMode != "k-q8-v-q4" ||
		cfg.DeviceKVMode != "k-q8-v-q4" ||
		cfg.BatchSize != 4 ||
		cfg.PrefillChunkSize != 2048 ||
		cfg.ExpectedQuantization != 4 ||
		cfg.MemoryLimitBytes != 16<<30 ||
		cfg.CacheLimitBytes != 8<<30 ||
		cfg.WiredLimitBytes != 4<<30 ||
		cfg.AdapterPath != "/adapters/explicit" ||
		status.Resolver.ContextLen != 8192 ||
		status.Resolver.KVCacheMode != "k-q8-v-q4" ||
		status.Resolver.AdapterPath != "/adapters/explicit" ||
		status.Labels["admin_reload_profile_path"] != "candidate_load_config_optional" {
		t.Fatalf("reload status config = %+v resolver=%+v labels=%+v, want profile-derived load config with explicit overrides", cfg, status.Resolver, status.Labels)
	}
}

func TestServeResolverRejectsActiveDraftFallback(t *testing.T) {
	model := t.TempDir()
	writeCLIDraftDetectModelDir(t, model, "gemma4_text")
	assistant := filepath.Join(model, "assistant")
	writeCLIDraftDetectModelDir(t, assistant, "gemma4_assistant")
	resolver := newROCmServeResolver(rocmServeConfig{
		Backend:     "rocm",
		ModelPath:   model,
		DraftPath:   "auto",
		DraftDetect: true,
		DraftBlock:  5,
	})

	loaded, err := resolver.ResolveModel(context.Background(), "")
	if err == nil {
		if loaded != nil {
			_ = loaded.Close()
		}
		t.Fatal("ResolveModel loaded target-only model with an active drafter; want explicit native-pending refusal")
	}
	for _, want := range []string{
		"reactive MTP drafter resolved",
		assistant,
		"block 5",
		"native ROCm drafter execution is pending",
		"refusing autoregressive fallback",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("ResolveModel error %q missing %q", err.Error(), want)
		}
	}

	reloadResolver := newROCmServeResolver(rocmServeConfig{
		Backend:     "rocm",
		DraftPath:   "auto",
		DraftDetect: true,
		DraftBlock:  6,
	})
	reloadResolver.Reload(model, "rocm", 4096)
	loaded, err = reloadResolver.ResolveModel(context.Background(), "")
	if err == nil {
		if loaded != nil {
			_ = loaded.Close()
		}
		t.Fatal("ResolveModel loaded target-only model after reload with an active drafter; want explicit native-pending refusal")
	}
	if !strings.Contains(err.Error(), "block 6") || !strings.Contains(err.Error(), "refusing autoregressive fallback") {
		t.Fatalf("reload ResolveModel error = %q, want block 6 fallback refusal", err.Error())
	}

	stateResolver := newROCmServeResolver(rocmServeConfig{
		Backend:            "rocm",
		ModelPath:          model,
		DraftPath:          "auto",
		DraftDetect:        true,
		DraftBlock:         5,
		StateConversations: true,
		StateStore:         state.NewInMemoryStore(nil),
	})
	loaded, err = stateResolver.ResolveModel(context.Background(), "")
	if err == nil {
		if loaded != nil {
			_ = loaded.Close()
		}
		t.Fatal("ResolveModel loaded stateful active drafter; want retained-state native-pending refusal")
	}
	if !strings.Contains(err.Error(), "retained-state OpenAI/server route") ||
		!strings.Contains(err.Error(), "rocm_state_session_runtime_kv") {
		t.Fatalf("stateful ResolveModel error = %q, want retained-state MTP boundary", err.Error())
	}
}

func TestLoadTunedDraftBlockUsesNewestMatchingProfile(t *testing.T) {
	dir := t.TempDir()
	writeProfile := func(name string, profile inference.TuningProfile) {
		t.Helper()
		if err := writeROCmTuningProfile(filepath.Join(dir, name), profile); err != nil {
			t.Fatal(err)
		}
	}
	writeProfile("old.json", inference.TuningProfile{
		Key:           inference.TuningProfileKey{Model: inference.ModelIdentity{Path: "/m"}},
		Candidate:     inference.TuningCandidate{ID: "mtp-block-4", Labels: map[string]string{tuneDraftBlockLabel: "4"}},
		CreatedAtUnix: 10,
	})
	writeProfile("new.json", inference.TuningProfile{
		Key:           inference.TuningProfileKey{Model: inference.ModelIdentity{Path: "/m"}},
		Candidate:     inference.TuningCandidate{ID: "mtp-block-5", Labels: map[string]string{tuneDraftBlockLabel: "5"}},
		CreatedAtUnix: 20,
	})
	writeProfile("other-model.json", inference.TuningProfile{
		Key:           inference.TuningProfileKey{Model: inference.ModelIdentity{Path: "/other"}},
		Candidate:     inference.TuningCandidate{ID: "mtp-block-6", Labels: map[string]string{tuneDraftBlockLabel: "6"}},
		CreatedAtUnix: 30,
	})
	writeProfile("other-machine.json", inference.TuningProfile{
		Key:           inference.TuningProfileKey{MachineHash: "machine-b", Model: inference.ModelIdentity{Path: "/m"}},
		Candidate:     inference.TuningCandidate{ID: "mtp-block-7", Labels: map[string]string{tuneDraftBlockLabel: "7"}},
		CreatedAtUnix: 40,
	})
	if err := os.WriteFile(filepath.Join(dir, "corrupt.json"), []byte("{"), 0o600); err != nil {
		t.Fatal(err)
	}

	block, path := loadTunedDraftBlock(dir, "/m", "machine-a")
	if block != 5 || filepath.Base(path) != "new.json" {
		t.Fatalf("loadTunedDraftBlock = %d/%q, want newest matching block 5", block, path)
	}
	if block, _ := loadTunedDraftBlock(dir, "/nope", "machine-a"); block != 0 {
		t.Fatalf("loadTunedDraftBlock for other model = %d, want 0", block)
	}
}

func TestLoadTunedDraftBlockUsesProfileIdentityOverlay(t *testing.T) {
	dir := t.TempDir()
	profile := inference.TuningProfile{
		Key: inference.TuningProfileKey{
			MachineHash: "machine-a",
			Model:       inference.ModelIdentity{Path: "/stale-key"},
		},
		Candidate: inference.TuningCandidate{
			ID:     "mtp-block-6",
			Model:  inference.ModelIdentity{Path: "/m"},
			Labels: map[string]string{tuneDraftBlockLabel: "6"},
		},
		CreatedAtUnix: 10,
	}
	if err := writeROCmTuningProfile(filepath.Join(dir, "overlay.json"), profile); err != nil {
		t.Fatal(err)
	}
	block, path := loadTunedDraftBlock(dir, "/m", "machine-a")
	if block != 6 || filepath.Base(path) != "overlay.json" {
		t.Fatalf("loadTunedDraftBlock overlay = %d/%q, want candidate model block 6", block, path)
	}
	if block, _ := loadTunedDraftBlock(dir, "/m", ""); block != 0 {
		t.Fatalf("loadTunedDraftBlock without local hash = %d, want machine-specific profile skipped", block)
	}
}

func TestLoadTunedDraftBlockMatchesWeightFileProfileToModelPackDir(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(t.TempDir(), "gemma")
	profile := inference.TuningProfile{
		Key: inference.TuningProfileKey{
			MachineHash: "machine-a",
			Model:       inference.ModelIdentity{Path: filepath.Join(modelDir, "model.safetensors")},
		},
		Candidate: inference.TuningCandidate{
			ID:     "mtp-block-4",
			Labels: map[string]string{tuneDraftBlockLabel: "4"},
		},
		CreatedAtUnix: 10,
	}
	if err := writeROCmTuningProfile(filepath.Join(dir, "weight-file-profile.json"), profile); err != nil {
		t.Fatal(err)
	}

	block, path := loadTunedDraftBlock(dir, modelDir, "machine-a")
	if block != 4 || filepath.Base(path) != "weight-file-profile.json" {
		t.Fatalf("loadTunedDraftBlock weight-file profile = %d/%q, want model-pack dir block 4", block, path)
	}
}

func TestServeResolverAppliesTunedDraftBlockProfile(t *testing.T) {
	model := t.TempDir()
	writeCLIDraftDetectModelDir(t, model, "gemma4_text")
	assistant := filepath.Join(model, "assistant")
	writeCLIDraftDetectModelDir(t, assistant, "gemma4_assistant")
	profileDir := t.TempDir()
	profilePath := filepath.Join(profileDir, "chat-machine-model-mtp-block-5.json")
	if err := writeROCmTuningProfile(profilePath, inference.TuningProfile{
		Key: inference.TuningProfileKey{
			Model: inference.ModelIdentity{Path: model},
		},
		Candidate: inference.TuningCandidate{
			ID:     "mtp-block-5",
			Labels: map[string]string{tuneDraftBlockLabel: "5"},
		},
		CreatedAtUnix: 20,
	}); err != nil {
		t.Fatal(err)
	}

	resolver := newROCmServeResolver(rocmServeConfig{
		Backend:     "rocm",
		ModelPath:   model,
		DraftPath:   "auto",
		DraftDetect: true,
		ProfileDir:  profileDir,
	})
	loaded, err := resolver.ResolveModel(context.Background(), "")
	if err == nil {
		if loaded != nil {
			_ = loaded.Close()
		}
		t.Fatal("ResolveModel loaded target-only model with tuned drafter; want native-pending refusal")
	}
	for _, want := range []string{"block 5", "tuned: " + profilePath, "refusing autoregressive fallback"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("ResolveModel error %q missing %q", err.Error(), want)
		}
	}

	noProfile := newROCmServeResolver(rocmServeConfig{
		Backend:       "rocm",
		ModelPath:     model,
		DraftPath:     "auto",
		DraftDetect:   true,
		NoAutoProfile: true,
		ProfileDir:    profileDir,
	})
	_, err = noProfile.ResolveModel(context.Background(), "")
	if err == nil || !strings.Contains(err.Error(), "block backend_default") || strings.Contains(err.Error(), "tuned:") {
		t.Fatalf("NoAutoProfile ResolveModel error = %v, want backend default with no tuned source", err)
	}

	explicit := newROCmServeResolver(rocmServeConfig{
		Backend:     "rocm",
		ModelPath:   model,
		DraftPath:   "auto",
		DraftDetect: true,
		DraftBlock:  6,
		ProfileDir:  profileDir,
	})
	_, err = explicit.ResolveModel(context.Background(), "")
	if err == nil || !strings.Contains(err.Error(), "block 6") || strings.Contains(err.Error(), "tuned:") {
		t.Fatalf("explicit draft block ResolveModel error = %v, want explicit block 6 with no tuned source", err)
	}
}

func TestServeOpenAIStateMTPLabelsPromoteLoadedNativeAttachment(t *testing.T) {
	detection := rocm.DraftDetection{
		Source:    rocm.DraftSourceFlag,
		DraftPath: "/models/gemma/assistant",
		Note:      "explicit --draft",
	}
	cfg := rocmServeConfig{
		StateConversations: true,
		StateStore:         state.NewInMemoryStore(nil),
	}
	linkedStatus := rocmServeResolverStatus{
		StateContinuity:     rocmServeStateContinuityConversationStore,
		NativeMTPAttachment: rocmServeOpenAIStateMTPLinked,
	}
	labels := rocmServeApplyOpenAIStateMTPLabels(nil, cfg, linkedStatus, detection)
	if labels["openai_chat_retained_state_mtp"] != rocmServeOpenAIStateMTPLinked ||
		labels["native_mtp_attachment"] != rocmServeOpenAIStateMTPLinked ||
		labels["openai_chat_retained_state_mtp_attachment"] != rocmServeOpenAIStateMTPLinked ||
		labels["openai_chat_retained_state_entrypoint"] != "attached_drafter_textmodel_generate_native_from_state" {
		t.Fatalf("linked labels = %+v, want loaded native MTP attachment promoted", labels)
	}

	pending := rocmServeApplyOpenAIStateMTPLabels(nil, cfg, rocmServeResolverStatus{
		StateContinuity: rocmServeStateContinuityStoreReady,
	}, detection)
	if pending["openai_chat_retained_state_mtp"] != rocmServeOpenAIStateMTPPendingNative ||
		pending["native_mtp_attachment"] != "" {
		t.Fatalf("pending labels = %+v, want pre-load retained MTP pending", pending)
	}

	disabled := rocmServeApplyOpenAIStateMTPLabels(nil, rocmServeConfig{}, rocmServeResolverStatus{}, detection)
	if disabled["openai_chat_retained_state_mtp"] != rocmServeOpenAIStateMTPDisabled {
		t.Fatalf("disabled labels = %+v, want disabled state MTP", disabled)
	}

	noDraft := rocmServeApplyOpenAIStateMTPLabels(nil, cfg, linkedStatus, rocm.DraftDetection{})
	if noDraft["openai_chat_retained_state_mtp"] != rocmServeOpenAIStateMTPNotApplicable {
		t.Fatalf("no-draft labels = %+v, want retained MTP not applicable", noDraft)
	}
}

func TestServeNativeMTPAttachmentStatusReadsLoadedModelIdentity(t *testing.T) {
	model := &serveReactiveTestModel{
		modelIdentity: inference.ModelIdentity{
			Labels: map[string]string{
				"attached_drafter_native_attachment": rocmServeOpenAIStateMTPLinked,
			},
		},
	}
	if got := rocmServeNativeMTPAttachmentStatus(model); got != rocmServeOpenAIStateMTPLinked {
		t.Fatalf("native attachment status = %q, want linked", got)
	}
	model.modelIdentity.Labels = map[string]string{
		"engine_attached_drafter_native_attachment": "not_linked",
	}
	if got := rocmServeNativeMTPAttachmentStatus(model); got != "not_linked" {
		t.Fatalf("native attachment engine-label status = %q, want not_linked", got)
	}
	if got := rocmServeNativeMTPAttachmentStatus(&traceCLITestModel{}); got != "" {
		t.Fatalf("native attachment status for non-reporter = %q, want empty", got)
	}
}

func TestServeResolverStatusReportsLoadedDraftSelection(t *testing.T) {
	model := &serveReactiveTestModel{
		modelIdentity: inference.ModelIdentity{
			Labels: map[string]string{
				"attached_drafter_native_attachment": rocmServeOpenAIStateMTPLinked,
			},
		},
	}
	cfg := rocmServeConfig{
		Backend:            "rocm",
		ModelPath:          "/models/gemma",
		DraftPath:          "/models/gemma/assistant",
		DraftDetect:        true,
		StateConversations: true,
		StateStore:         state.NewInMemoryStore(nil),
	}
	resolver := newROCmServeResolver(cfg)
	resolver.model = model
	resolver.loadedAt = time.Unix(1234, 0)
	resolver.loadedSelection = rocmServeDraftSelection{
		Detection: rocm.DraftDetection{
			Source:    rocm.DraftSourceFlag,
			DraftPath: "/models/gemma/assistant",
			Note:      "explicit --draft",
		},
		DraftBlock:       3,
		DraftBlockSource: "tuned: /profiles/chat-mtp-block-3.json",
	}

	status := resolver.Status()
	if status.DraftBlock != 3 ||
		status.DraftBlockSource != "tuned: /profiles/chat-mtp-block-3.json" ||
		status.NativeMTPAttachment != rocmServeOpenAIStateMTPLinked {
		t.Fatalf("status = %+v, want loaded tuned draft selection and linked native MTP", status)
	}
	labels := rocmServeLabelsForStatus(cfg, status)
	if labels["openai_chat_retained_state_mtp"] != rocmServeOpenAIStateMTPLinked ||
		labels["native_mtp_attachment"] != rocmServeOpenAIStateMTPLinked {
		t.Fatalf("labels = %+v, want loaded server OpenAI MTP linked", labels)
	}
}

func TestGenerateAppliesTunedDraftBlockProfile(t *testing.T) {
	model := t.TempDir()
	writeCLIDraftDetectModelDir(t, model, "gemma4_text")
	assistant := filepath.Join(model, "assistant")
	writeCLIDraftDetectModelDir(t, assistant, "gemma4_assistant")
	profileDir := t.TempDir()
	profilePath := filepath.Join(profileDir, "generate-machine-model-mtp-block-3.json")
	if err := writeROCmTuningProfile(profilePath, inference.TuningProfile{
		Key: inference.TuningProfileKey{
			Model: inference.ModelIdentity{Path: model},
		},
		Candidate: inference.TuningCandidate{
			ID:     "mtp-block-3",
			Labels: map[string]string{tuneDraftBlockLabel: "3"},
		},
		CreatedAtUnix: 20,
	}); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"generate", "-state=", "-draft", "auto", "-profile-dir", profileDir, model}, &stdout, &stderr)
	if code == 0 {
		t.Fatalf("generate loaded temp model unexpectedly; stdout=%s stderr=%s", stdout.String(), stderr.String())
	}
	for _, want := range []string{"block 3", "tuned: " + profilePath} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("generate stderr %q missing %q", stderr.String(), want)
		}
	}

	stdout.Reset()
	stderr.Reset()
	code = runCommand(context.Background(), []string{"generate", "-state=", "-draft", "auto", "-profile-dir", profileDir, "-no-auto-profile", model}, &stdout, &stderr)
	if code == 0 {
		t.Fatalf("generate no-auto-profile loaded temp model unexpectedly; stdout=%s stderr=%s", stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "block backend_default") || strings.Contains(stderr.String(), "tuned:") {
		t.Fatalf("generate no-auto-profile stderr = %q, want backend default with no tuned source", stderr.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = runCommand(context.Background(), []string{"generate", "-state=", "-draft", "auto", "-draft-block", "5", "-profile-dir", profileDir, model}, &stdout, &stderr)
	if code == 0 {
		t.Fatalf("generate explicit draft-block loaded temp model unexpectedly; stdout=%s stderr=%s", stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "block 5") || strings.Contains(stderr.String(), "tuned:") {
		t.Fatalf("generate explicit draft-block stderr = %q, want explicit block with no tuned source", stderr.String())
	}
}

func TestServeHandlerReactiveAdminRoutesAndStreamingResponses(t *testing.T) {
	handler, resolver := newROCmServeHandler(rocmServeConfig{
		Backend:    "rocm",
		AdminToken: "test-token",
	})
	defer resolver.Close()

	resolver.mu.Lock()
	resolver.modelPath = "/tmp/models/reactive"
	resolver.model = &serveReactiveTestModel{
		tokens: []string{"hello", " world"},
		entries: []inference.CacheBlockRef{
			{ID: "block-dev", Kind: "prompt", TokenCount: 2, Labels: map[string]string{"tenant": "dev"}},
			{ID: "block-prod", Kind: "prompt", TokenCount: 4, Labels: map[string]string{"tenant": "prod"}},
		},
		stats: inference.CacheStats{Blocks: 2, MemoryBytes: 24, CacheMode: "block-prefix"},
	}
	resolver.mu.Unlock()

	unauthorized := httptest.NewRecorder()
	handler.ServeHTTP(unauthorized, httptest.NewRequest(http.MethodGet, rocmServeCacheEntriesPath, nil))
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("cache entries without token status=%d body=%s, want 401", unauthorized.Code, unauthorized.Body.String())
	}

	wakeReq := httptest.NewRequest(http.MethodPost, rocmServeRuntimeWakePath, nil)
	wakeReq.Header.Set("Authorization", "Bearer test-token")
	wake := httptest.NewRecorder()
	handler.ServeHTTP(wake, wakeReq)
	if wake.Code != http.StatusOK || !strings.Contains(wake.Body.String(), `"action":"wake"`) || !strings.Contains(wake.Body.String(), `"loaded":true`) {
		t.Fatalf("wake status=%d body=%s, want loaded wake action", wake.Code, wake.Body.String())
	}

	cacheReq := httptest.NewRequest(http.MethodGet, rocmServeCacheEntriesPath+"?model=reactive&tenant=prod", nil)
	cacheReq.Header.Set("Authorization", "Bearer test-token")
	cache := httptest.NewRecorder()
	handler.ServeHTTP(cache, cacheReq)
	if cache.Code != http.StatusOK ||
		!strings.Contains(cache.Body.String(), `"object":"list"`) ||
		!strings.Contains(cache.Body.String(), `"id":"block-prod"`) ||
		strings.Contains(cache.Body.String(), `"id":"block-dev"`) ||
		!strings.Contains(cache.Body.String(), `"blocks":2`) {
		t.Fatalf("cache entries status=%d body=%s, want filtered cache entry list with stats", cache.Code, cache.Body.String())
	}

	responsesReq := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"reactive","input":[{"role":"user","content":"hi"}],"stream":true}`))
	responses := httptest.NewRecorder()
	handler.ServeHTTP(responses, responsesReq)
	if responses.Code != http.StatusOK ||
		!strings.Contains(responses.Header().Get("Content-Type"), "text/event-stream") ||
		!strings.Contains(responses.Body.String(), "response.output_text.delta") ||
		!strings.Contains(responses.Body.String(), "[DONE]") {
		t.Fatalf("streaming responses status=%d content-type=%q body=%s, want SSE deltas", responses.Code, responses.Header().Get("Content-Type"), responses.Body.String())
	}

	sleepReq := httptest.NewRequest(http.MethodPost, rocmServeRuntimeSleepPath, nil)
	sleepReq.Header.Set("Authorization", "Bearer test-token")
	sleep := httptest.NewRecorder()
	handler.ServeHTTP(sleep, sleepReq)
	if sleep.Code != http.StatusOK || !strings.Contains(sleep.Body.String(), `"action":"sleep"`) || !strings.Contains(sleep.Body.String(), `"loaded":false`) {
		t.Fatalf("sleep status=%d body=%s, want unloaded sleep action", sleep.Code, sleep.Body.String())
	}
}

func TestServeHandlerAdminSFTLifecycle(t *testing.T) {
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "train.jsonl")
	if err := os.WriteFile(trainPath, []byte(`{"prompt":"alpha","response":"bravo"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	backendName := "admin-sft-" + strings.NewReplacer("/", "-", " ", "-").Replace(t.Name())
	model := &sftCLITestModel{}
	backend := &sftCLITestBackend{name: backendName, model: model}
	inference.Register(backend)
	adapterRoot := filepath.Join(dir, "adapters")
	handler, resolver := newROCmServeHandler(rocmServeConfig{
		Backend:             backendName,
		AdminToken:          "test-token",
		AdminSFTAdapterRoot: adapterRoot,
	})
	defer resolver.Close()

	unauthorized := httptest.NewRecorder()
	handler.ServeHTTP(unauthorized, httptest.NewRequest(http.MethodPost, rocmServeAdminSFTStartPath, strings.NewReader(`{}`)))
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("admin SFT start without token status=%d body=%s, want 401", unauthorized.Code, unauthorized.Body.String())
	}

	startReq := httptest.NewRequest(http.MethodPost, rocmServeAdminSFTStartPath, strings.NewReader(`{
		"model_path":"/models/gemma4",
		"dataset_path":"`+trainPath+`",
		"adapter_name":"domain-a",
		"epochs":2,
		"batch_size":3,
		"learning_rate":0.0003,
		"lora_rank":4,
		"lora_alpha":8,
		"max_seq_len":512,
		"context_length":4096
	}`))
	startReq.Header.Set("Authorization", "Bearer test-token")
	start := httptest.NewRecorder()
	handler.ServeHTTP(start, startReq)
	if start.Code != http.StatusAccepted {
		t.Fatalf("admin SFT start status=%d body=%s, want 202", start.Code, start.Body.String())
	}
	var started rocmAdminSFTJob
	if err := json.Unmarshal(start.Body.Bytes(), &started); err != nil {
		t.Fatalf("decode SFT start response: %v\n%s", err, start.Body.String())
	}
	if started.JobID == "" || started.AdapterDir != filepath.Join(adapterRoot, "domain-a") || started.Backend != backendName {
		t.Fatalf("unexpected started job: %+v", started)
	}

	done := waitForROCmAdminSFTJob(t, handler, "test-token", started.JobID, rocmAdminSFTStateDone)
	if done.Metrics.Step != 7 || done.Metrics.Epoch != 2 || done.Metrics.Samples != 1 || done.LastLoss != 0.125 {
		t.Fatalf("done job metrics = %+v last_loss=%v, want trainer result", done.Metrics, done.LastLoss)
	}
	if done.Adapter.Path != filepath.Join(adapterRoot, "domain-a", "adapter.safetensors") ||
		done.Adapter.Rank != 4 ||
		done.Adapter.Alpha != 8 {
		t.Fatalf("done adapter = %+v, want configured adapter identity", done.Adapter)
	}
	if done.Labels["training_stage"] != "admin_native_lora_sft_execute" ||
		done.Labels["trainer_interface"] != "inference.SFTTrainer" ||
		done.Labels["dataset_loader"] != "rocm.LoadJSONLDataset" ||
		done.Labels["production_requires_env_gate"] != "false" ||
		done.Labels["production_requires_cli_flag"] != "false" {
		t.Fatalf("done labels = %+v, want admin SFT production labels", done.Labels)
	}
	if backend.loadPath != "/models/gemma4" || backend.loadConfig.Backend != backendName || backend.loadConfig.ContextLen != 4096 {
		t.Fatalf("backend load path/config = %q %+v", backend.loadPath, backend.loadConfig)
	}
	if !model.closed {
		t.Fatalf("admin SFT model was not closed")
	}
	if len(model.samples) != 1 || model.samples[0].Prompt != "alpha" || model.samples[0].Response != "bravo" {
		t.Fatalf("admin SFT samples = %+v", model.samples)
	}
	if model.config.Epochs != 2 ||
		model.config.BatchSize != 3 ||
		model.config.LearningRate != 0.0003 ||
		model.config.LoRA.Rank != 4 ||
		model.config.LoRA.Alpha != 8 ||
		model.config.Labels["max_sequence_length"] != "512" {
		t.Fatalf("admin SFT config = %+v labels=%+v", model.config, model.config.Labels)
	}

	adaptersReq := httptest.NewRequest(http.MethodGet, rocmServeAdminSFTAdaptersPath, nil)
	adaptersReq.Header.Set("Authorization", "Bearer test-token")
	adapters := httptest.NewRecorder()
	handler.ServeHTTP(adapters, adaptersReq)
	if adapters.Code != http.StatusOK ||
		!strings.Contains(adapters.Body.String(), `"dir":"`+adapterRoot+`"`) ||
		!strings.Contains(adapters.Body.String(), `"name":"domain-a"`) {
		t.Fatalf("adapters status=%d body=%s, want completed adapter listed", adapters.Code, adapters.Body.String())
	}
}

func TestServeHandlerAdminSFTFailsClosedWithoutTrainer(t *testing.T) {
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "train.jsonl")
	if err := os.WriteFile(trainPath, []byte(`{"prompt":"alpha","response":"bravo"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	backendName := "admin-sft-no-trainer-" + strings.NewReplacer("/", "-", " ", "-").Replace(t.Name())
	model := &traceCLITestModel{}
	inference.Register(&traceCLITestBackend{name: backendName, model: model})
	handler, resolver := newROCmServeHandler(rocmServeConfig{
		Backend:             backendName,
		AdminToken:          "test-token",
		AdminSFTAdapterRoot: filepath.Join(dir, "adapters"),
	})
	defer resolver.Close()

	startReq := httptest.NewRequest(http.MethodPost, rocmServeAdminSFTStartPath, strings.NewReader(`{
		"model_path":"/models/gemma4",
		"dataset_path":"`+trainPath+`",
		"adapter_name":"domain-b"
	}`))
	startReq.Header.Set("Authorization", "Bearer test-token")
	start := httptest.NewRecorder()
	handler.ServeHTTP(start, startReq)
	if start.Code != http.StatusAccepted {
		t.Fatalf("admin SFT start status=%d body=%s, want 202", start.Code, start.Body.String())
	}
	var started rocmAdminSFTJob
	if err := json.Unmarshal(start.Body.Bytes(), &started); err != nil {
		t.Fatalf("decode SFT start response: %v\n%s", err, start.Body.String())
	}
	failed := waitForROCmAdminSFTJob(t, handler, "test-token", started.JobID, rocmAdminSFTStateFailed)
	if !strings.Contains(failed.Error, "does not implement inference.SFTTrainer") {
		t.Fatalf("failed job error = %q, want SFTTrainer refusal", failed.Error)
	}
	if !model.closed {
		t.Fatalf("non-trainer admin SFT model was not closed")
	}
}

func waitForROCmAdminSFTJob(t *testing.T, handler http.Handler, token, jobID string, want rocmAdminSFTJobState) rocmAdminSFTJob {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		req := httptest.NewRequest(http.MethodGet, rocmServeAdminSFTStatusPath+"?job="+jobID, nil)
		req.Header.Set("Authorization", "Bearer "+token)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("admin SFT status=%d body=%s, want 200", rec.Code, rec.Body.String())
		}
		var job rocmAdminSFTJob
		if err := json.Unmarshal(rec.Body.Bytes(), &job); err != nil {
			t.Fatalf("decode admin SFT status: %v\n%s", err, rec.Body.String())
		}
		if job.State == want {
			return job
		}
		if job.State == rocmAdminSFTStateFailed || job.State == rocmAdminSFTStateStopped {
			t.Fatalf("admin SFT job reached %s, want %s: %+v", job.State, want, job)
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for admin SFT job %s, last=%+v want=%s", jobID, job, want)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestDiscoverModelCandidatesReportsReactiveDraftDetection(t *testing.T) {
	model := t.TempDir()
	writeCLIDraftDetectModelDir(t, model, "gemma4_text")
	assistant := filepath.Join(model, "assistant")
	writeCLIDraftDetectModelDir(t, assistant, "gemma4_assistant")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"discover", "-json", "-model-dir", model}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("discover code=%d, want 0; stderr=%s", code, stderr.String())
	}
	var report discoveryReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("discover JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if len(report.Models) != 1 {
		t.Fatalf("models=%d, want 1: %+v", len(report.Models), report.Models)
	}
	if report.Models[0].ModelProfile == nil ||
		report.Models[0].ModelProfile.Name != "gemma4" ||
		report.Models[0].ModelProfile.Registry != rocm.DefaultROCmModelRegistryName() ||
		report.Models[0].ModelProfile.Architecture != "gemma4_text" ||
		report.Models[0].ModelProfile.Gemma4Settings.ChatTemplate != "gemma4_hf_turn" ||
		report.Models[0].ModelProfile.Gemma4Settings.ParserID != "gemma" ||
		report.Models[0].ModelProfile.LoadStatus.Status != rocm.ROCmModelLoadStandaloneNative ||
		report.Models[0].ModelProfile.LoadStatus.Target != "standalone" ||
		report.Models[0].ModelProfile.LoadStatus.Staged ||
		!report.Models[0].ModelProfile.LoadStatus.TextGenerate ||
		!report.Models[0].ModelProfile.Gemma4EngineFeatures.ModelContextWindow {
		t.Fatalf("model profile = %+v, want structured Gemma4 registry profile", report.Models[0].ModelProfile)
	}
	if report.Models[0].ModelRoutes == nil ||
		report.Models[0].ModelRoutes.Contract != rocm.ROCmModelRoutePlanContract ||
		report.Models[0].ModelRoutes.Architecture != "gemma4_text" ||
		report.Models[0].ModelRoutes.LoadStatus.Status != rocm.ROCmModelLoadStandaloneNative ||
		report.Models[0].ModelRoutes.Labels["engine_route_plan_feature"] != "true" {
		t.Fatalf("model routes = %+v, want compact Gemma4 route plan", report.Models[0].ModelRoutes)
	}
	if report.Models[0].TokenLoop == nil ||
		!report.Models[0].TokenLoop.IncrementalDecodeReady() ||
		report.Models[0].Labels["engine_token_loop_contract"] != rocm.ROCmTokenLoopContract {
		t.Fatalf("token loop = %+v labels=%+v, want retained Gemma4 token-loop contract", report.Models[0].TokenLoop, report.Models[0].Labels)
	}
	labels := report.Models[0].Labels
	if labels["reactive_draft_detection"] != "active_pending_native_drafter" ||
		labels["reactive_draft_source"] != "assistant-dir" ||
		labels["reactive_draft_path"] != assistant {
		t.Fatalf("draft labels = %+v, want assistant-dir detection", labels)
	}
}

func TestDiscoverModelCandidatesReportsGenericReactiveRegistryProfile(t *testing.T) {
	model := t.TempDir()
	writeCLIDraftDetectModelDir(t, model, "qwen3_6_moe")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"discover", "-json", "-model-dir", model}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("discover code=%d, want 0; stderr=%s", code, stderr.String())
	}
	var report discoveryReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("discover JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if len(report.Models) != 1 {
		t.Fatalf("models=%d, want 1: %+v", len(report.Models), report.Models)
	}
	profile := report.Models[0].ModelProfile
	if profile == nil ||
		profile.Name != "qwen" ||
		profile.Registry != rocm.DefaultROCmModelRegistryName() ||
		profile.Architecture != "qwen3_6_moe" ||
		profile.ArchitectureProfile.ID != "qwen3_6_moe" ||
		profile.EngineFeatures.ChatTemplateID != "qwen" ||
		profile.EngineFeatures.ReasoningParserID != "qwen" ||
		profile.EngineFeatures.ToolParserID != "qwen" ||
		!profile.EngineFeatures.ChatTemplate ||
		!profile.EngineFeatures.ReasoningParse ||
		!profile.EngineFeatures.ToolParse ||
		profile.EngineFeatures.TextGenerate ||
		profile.LoadStatus.Status != rocm.ROCmModelLoadStagedNative ||
		profile.LoadStatus.Target != "standalone" ||
		!profile.LoadStatus.Staged ||
		profile.LoadStatus.TextGenerate {
		t.Fatalf("model profile = %+v, want inspection-derived Qwen registry engine features", profile)
	}
	routes := report.Models[0].ModelRoutes
	if routes == nil ||
		routes.Contract != rocm.ROCmModelRoutePlanContract ||
		routes.Architecture != "qwen3_6_moe" ||
		routes.Family != "qwen" ||
		!routes.FeatureRoute.Matched() ||
		!routes.TokenizerRoute.Matched() ||
		routes.LoadStatus.Status != rocm.ROCmModelLoadStagedNative ||
		routes.Labels["engine_route_plan_tokenizer"] != "true" {
		t.Fatalf("model routes = %+v, want inspection-derived Qwen route plan", routes)
	}
	labels := report.Models[0].Labels
	if labels["engine_profile"] != "qwen" ||
		labels["engine_profile_source"] != "architecture_profile" ||
		labels["engine_feature_architecture"] != "qwen3_6_moe" ||
		labels["engine_feature_chat_template_id"] != "qwen" ||
		labels["engine_feature_reasoning_parser"] != "qwen" ||
		labels["engine_feature_tool_parser"] != "qwen" ||
		labels["engine_feature_capabilities"] != "chat.template,reasoning.parse,tool.parse" ||
		labels["engine_load_status"] != string(rocm.ROCmModelLoadStagedNative) ||
		labels["engine_load_target"] != "standalone" ||
		labels["engine_load_staged"] != "true" ||
		labels["engine_load_text_generate"] != "false" {
		t.Fatalf("labels = %+v, want generic reactive registry feature labels", labels)
	}
}

func TestDiscoverIncludeCandidatesReportsROCmTuningCandidate(t *testing.T) {
	model := t.TempDir()
	writeCLIDraftDetectModelDir(t, model, "qwen3_6_moe")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"discover", "-json", "-model-dir", model, "-include-candidates", "-workload", string(inference.TuningWorkloadChat)}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("discover code=%d, want 0; stderr=%s", code, stderr.String())
	}
	var report discoveryReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("discover JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if !slices.Equal(report.Workloads, []inference.TuningWorkload{inference.TuningWorkloadChat}) {
		t.Fatalf("workloads = %+v, want chat-only workload", report.Workloads)
	}
	if len(report.Models) != 1 || len(report.Candidates) != 1 {
		t.Fatalf("models=%d candidates=%d, want one model and one candidate: %+v", len(report.Models), len(report.Candidates), report.Candidates)
	}
	candidate := report.Candidates[0]
	if candidate.ID != inference.CandidateID(inference.TuningWorkloadChat, "k-q8-v-q4", candidate.ContextLength, candidate.BatchSize) ||
		candidate.Workload != inference.TuningWorkloadChat ||
		!strings.HasPrefix(candidate.Model.Path, model) ||
		candidate.Model.Architecture != "qwen3_6_moe" ||
		candidate.Runtime.Backend != defaultBackendName ||
		candidate.Runtime.CacheMode != "k-q8-v-q4" ||
		!candidate.Runtime.NativeRuntime ||
		candidate.CacheMode != "k-q8-v-q4" ||
		candidate.BatchSize != 1 ||
		candidate.ParallelSlots != 1 ||
		candidate.PromptCache {
		t.Fatalf("candidate = %+v, want ROCm native-compatible chat tuning candidate", candidate)
	}
	if candidate.Labels["engine_profile"] != "qwen" ||
		candidate.Labels["engine_registry"] != rocm.DefaultROCmModelRegistryName() ||
		candidate.Labels["engine_load_status"] != string(rocm.ROCmModelLoadStagedNative) ||
		candidate.Labels["engine_load_text_generate"] != "false" ||
		candidate.Labels["chat_template_id"] != "qwen" ||
		candidate.Labels["reasoning_parser_id"] != "qwen" ||
		candidate.Labels["production_requires_env_gate"] != "false" ||
		candidate.Labels["production_requires_cli_flag"] != "false" ||
		candidate.Labels["candidate_cache_mode_bound"] != "true" ||
		candidate.Labels["reactive_registry_planning"] != "true" {
		t.Fatalf("candidate labels = %+v, want reactive registry and production fast-lane metadata", candidate.Labels)
	}
	if candidate.Runtime.Labels["candidate_source"] != "lthn-rocm discover" ||
		candidate.Runtime.Labels["engine_profile"] != "qwen" {
		t.Fatalf("runtime labels = %+v, want candidate contract labels copied onto runtime", candidate.Runtime.Labels)
	}
	if len(candidate.Reasons) == 0 || !strings.Contains(strings.Join(candidate.Reasons, "\n"), "registry-derived ROCm discovery candidate") {
		t.Fatalf("candidate reasons = %+v, want registry-derived explanation", candidate.Reasons)
	}
}

func TestDiscoverRejectsUnknownWorkload(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"discover", "-json", "-workload", "mystery"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("discover code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("discover wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), `unsupported workload "mystery"`) {
		t.Fatalf("stderr missing workload validation: %s", stderr.String())
	}
}

func TestServeHandlerNoModelLoadedReportsBoundary(t *testing.T) {
	handler, resolver := newROCmServeHandler(rocmServeConfig{Backend: "rocm", AdminToken: "test-token"})
	defer resolver.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound || !strings.Contains(rec.Body.String(), "no model loaded") {
		t.Fatalf("chat without model status=%d body=%s, want no-model-loaded error", rec.Code, rec.Body.String())
	}
}

func TestServeHandlerScoreRouteGoodAndBad(t *testing.T) {
	handler, resolver := newROCmServeHandler(rocmServeConfig{Backend: "rocm", AdminToken: "test-token"})
	defer resolver.Close()

	req := httptest.NewRequest(http.MethodPost, rocmServeScorePath, strings.NewReader(`{"prompt":"explain your reasoning about the harbour plan","response":"you're absolutely right, I was wrong about the harbour"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("score status=%d body=%s, want 200", rec.Code, rec.Body.String())
	}
	var reply scoreRouteReply
	if err := json.Unmarshal(rec.Body.Bytes(), &reply); err != nil {
		t.Fatalf("score response did not unmarshal: %v\n%s", err, rec.Body.String())
	}
	if reply.Response.Sycophancy == nil {
		t.Fatal("response.sycophancy missing")
	}
	if reply.Response.LEK == nil || reply.Response.LEK.LEKScore < 0 || reply.Response.LEK.LEKScore > 100 {
		t.Fatalf("response.lek = %+v, want composite in [0,100]", reply.Response.LEK)
	}
	if reply.Response.Imprint == nil {
		t.Fatal("response.imprint missing for tokenized text")
	}
	if reply.Differential == nil {
		t.Fatal("differential missing for tokenized prompt/response pair")
	}

	get := httptest.NewRecorder()
	handler.ServeHTTP(get, httptest.NewRequest(http.MethodGet, rocmServeScorePath, nil))
	if get.Code != http.StatusMethodNotAllowed {
		t.Fatalf("score GET status=%d body=%s, want 405", get.Code, get.Body.String())
	}

	bad := httptest.NewRecorder()
	handler.ServeHTTP(bad, httptest.NewRequest(http.MethodPost, rocmServeScorePath, strings.NewReader("{not json")))
	if bad.Code != http.StatusBadRequest || !strings.Contains(bad.Body.String(), "invalid JSON") {
		t.Fatalf("score bad JSON status=%d body=%s, want 400 invalid JSON", bad.Code, bad.Body.String())
	}

	empty := httptest.NewRecorder()
	handler.ServeHTTP(empty, httptest.NewRequest(http.MethodPost, rocmServeScorePath, strings.NewReader(`{"prompt":"","response":""}`)))
	if empty.Code != http.StatusOK {
		t.Fatalf("score empty status=%d body=%s, want 200", empty.Code, empty.Body.String())
	}
	var emptyReply scoreRouteReply
	if err := json.Unmarshal(empty.Body.Bytes(), &emptyReply); err != nil {
		t.Fatalf("empty score response did not unmarshal: %v\n%s", err, empty.Body.String())
	}
	if emptyReply.Differential != nil {
		t.Fatalf("empty score differential = %+v, want absent", emptyReply.Differential)
	}
}

type serveReactiveTestModel struct {
	tokens        []string
	entries       []inference.CacheBlockRef
	stats         inference.CacheStats
	modelIdentity inference.ModelIdentity
	cacheProfile  rocmmodel.CacheProfile
}

func (m *serveReactiveTestModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.streamTokens()
}

func (m *serveReactiveTestModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.streamTokens()
}

func (m *serveReactiveTestModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (m *serveReactiveTestModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (m *serveReactiveTestModel) ModelType() string {
	if m.modelIdentity.Architecture != "" {
		return m.modelIdentity.Architecture
	}
	return "test"
}

func (m *serveReactiveTestModel) Info() inference.ModelInfo {
	if m.modelIdentity.Architecture != "" ||
		m.modelIdentity.VocabSize != 0 ||
		m.modelIdentity.NumLayers != 0 ||
		m.modelIdentity.HiddenSize != 0 ||
		m.modelIdentity.QuantBits != 0 ||
		m.modelIdentity.QuantGroup != 0 {
		return inference.ModelInfo{
			Architecture: m.modelIdentity.Architecture,
			VocabSize:    m.modelIdentity.VocabSize,
			NumLayers:    m.modelIdentity.NumLayers,
			HiddenSize:   m.modelIdentity.HiddenSize,
			QuantBits:    m.modelIdentity.QuantBits,
			QuantGroup:   m.modelIdentity.QuantGroup,
		}
	}
	return inference.ModelInfo{Architecture: "test"}
}

func (m *serveReactiveTestModel) ModelIdentity() inference.ModelIdentity {
	identity := m.modelIdentity
	identity.Labels = cloneServeReactiveLabels(identity.Labels)
	return identity
}

func (m *serveReactiveTestModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{GeneratedTokens: len(m.tokens)}
}

func (m *serveReactiveTestModel) Err() error {
	return nil
}

func (m *serveReactiveTestModel) Close() error {
	return nil
}

func (m *serveReactiveTestModel) CacheEntries(ctx context.Context, labels map[string]string) ([]inference.CacheBlockRef, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	entries := make([]inference.CacheBlockRef, 0, len(m.entries))
	for _, entry := range m.entries {
		if serveReactiveLabelsMatch(entry.Labels, labels) {
			entry.Labels = cloneServeReactiveLabels(entry.Labels)
			entries = append(entries, entry)
		}
	}
	return entries, nil
}

func (m *serveReactiveTestModel) CacheStats(ctx context.Context) (inference.CacheStats, error) {
	if err := ctx.Err(); err != nil {
		return inference.CacheStats{}, err
	}
	stats := m.stats
	stats.Labels = cloneServeReactiveLabels(stats.Labels)
	return stats, nil
}

func (m *serveReactiveTestModel) WarmCache(context.Context, inference.CacheWarmRequest) (inference.CacheWarmResult, error) {
	return inference.CacheWarmResult{}, nil
}

func (m *serveReactiveTestModel) ClearCache(context.Context, map[string]string) (inference.CacheStats, error) {
	return m.stats, nil
}

func (m *serveReactiveTestModel) CacheProfile(context.Context) (rocmmodel.CacheProfile, error) {
	return m.cacheProfile.Clone(), nil
}

func (m *serveReactiveTestModel) streamTokens() iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, text := range m.tokens {
			if !yield(inference.Token{Text: text}) {
				return
			}
		}
	}
}

func serveReactiveLabelsMatch(labels, filter map[string]string) bool {
	for key, want := range filter {
		if labels[key] != want {
			return false
		}
	}
	return true
}

func cloneServeReactiveLabels(labels map[string]string) map[string]string {
	if len(labels) == 0 {
		return nil
	}
	out := make(map[string]string, len(labels))
	for key, value := range labels {
		out[key] = value
	}
	return out
}

func TestRemainingReactiveCommandStubsParseFlags(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{name: "menubar", args: []string{"menubar", "-addr", ":36911", "-model", "model", "-foreground"}, want: "menu-bar app"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runCommand(context.Background(), tt.args, &stdout, &stderr)
			if code != 1 {
				t.Fatalf("%s code=%d, want 1", tt.name, code)
			}
			if stdout.Len() != 0 {
				t.Fatalf("%s wrote stdout: %s", tt.name, stdout.String())
			}
			if !strings.Contains(stderr.String(), tt.want) {
				t.Fatalf("%s stderr missing %q: %s", tt.name, tt.want, stderr.String())
			}
		})
	}
}

func TestSFTCommandJSONReportsNativePlan(t *testing.T) {
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "train.jsonl")
	validPath := filepath.Join(dir, "valid.jsonl")
	trainRows := strings.Join([]string{
		`{"prompt":"hello","response":"world","target_token_id":1}`,
		`{"messages":[{"role":"user","content":"u"},{"role":"assistant","content":"a"}]}`,
	}, "\n")
	if err := os.WriteFile(trainPath, []byte(trainRows), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(validPath, []byte(`{"instruction":"sum","input":"1+1","output":"2"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"sft",
		"-json",
		"-model", "/models/gemma4",
		"-data", trainPath,
		"-valid", validPath,
		"-rank", "8",
		"-alpha", "16",
		"-lr", "0.0002",
		"-epochs", "2",
		"-batch", "2",
		"-grad-accum", "4",
		"-max-seq", "1024",
		"-save", "adapter.safetensors",
		"-merge",
		"-metrics-lp", "metrics.lp",
		"-influx-url", "http://127.0.0.1:8086",
		"-influx-token", "secret-token",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("sft code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("sft JSON wrote stderr: %s", stderr.String())
	}
	if strings.Contains(stdout.String(), "secret-token") {
		t.Fatalf("sft JSON leaked influx token: %s", stdout.String())
	}
	var report sftPlanReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("sft JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Backend != "rocm" || report.Command != "sft" || report.CLIContract != cliContractName || !report.NoPython {
		t.Fatalf("unexpected SFT report identity: %+v", report)
	}
	if report.Kind != "native-lora-sft-plan" || report.ModelPath != "/models/gemma4" || report.DataPath != trainPath {
		t.Fatalf("unexpected SFT plan paths: %+v", report)
	}
	if report.Config.LoRA.Rank != 8 ||
		report.Config.LoRA.Alpha != 16 ||
		report.Config.LearningRate != 0.0002 ||
		report.Config.Epochs != 2 ||
		report.Config.BatchSize != 2 ||
		report.Config.GradientAccumulation != 4 ||
		report.Config.MaxSequenceLength != 1024 {
		t.Fatalf("unexpected SFT config: %+v", report.Config)
	}
	if report.Dataset.Samples != 2 ||
		report.Dataset.PromptRows != 1 ||
		report.Dataset.ResponseRows != 2 ||
		report.Dataset.MessageRows != 1 ||
		report.Dataset.TargetRows != 1 ||
		!sftTestStringSliceContains(report.Dataset.Formats, "prompt_response") ||
		!sftTestStringSliceContains(report.Dataset.Formats, "openai_messages") {
		t.Fatalf("unexpected SFT dataset summary: %+v", report.Dataset)
	}
	if report.Validation == nil || report.Validation.Samples != 1 || !sftTestStringSliceContains(report.Validation.Formats, "alpaca") {
		t.Fatalf("unexpected SFT validation summary: %+v", report.Validation)
	}
	if !report.MergeAfterTraining ||
		report.OutputAdapterPath != "adapter.safetensors" ||
		!report.Metrics.InfluxConfigured ||
		!report.Metrics.InfluxTokenSet ||
		report.Metrics.LineProtocolPath != "metrics.lp" {
		t.Fatalf("unexpected SFT output/metrics plan: %+v", report)
	}
	if report.Labels["training_stage"] != "native_lora_sft_plan" ||
		report.Labels["trainer_interface"] != "runtime_checked" ||
		report.Labels["sft_trainer_interface"] != "inference.SFTTrainer" ||
		report.Labels["model_loader_interface"] != "linked" ||
		report.Labels["native_trainer_interface"] != "not_implemented" ||
		report.Labels["loss_helper"] != "RunNativeSFTLossPass" ||
		report.Labels["lora_backward_helper"] != "RunNativeLoRABackwardPass" ||
		report.Labels["lora_update_helper"] != "RunNativeLoRAAdamWUpdatePass" ||
		report.Labels["lora_adapter_snapshot_helper"] != "SaveNativeLoRAAdapterSnapshot" ||
		report.Labels["training_interface"] != "lora_backward_plus_optimizer_update" ||
		report.Labels["optimizer_helper"] != "RunNativeSFTAdamWUpdatePass" ||
		report.Labels["native_lora_backward"] != "reference" ||
		report.Labels["native_lora_update_pass"] != "linked" ||
		report.Labels["production_requires_env_gate"] != "false" ||
		report.Labels["production_requires_cli_flag"] != "false" {
		t.Fatalf("unexpected SFT labels: %+v", report.Labels)
	}
}

func TestSFTCommandRunsTrainerByDefault(t *testing.T) {
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "train.jsonl")
	trainRows := strings.Join([]string{
		`{"prompt":"alpha","response":"bravo"}`,
		`{"messages":[{"role":"user","content":"charlie"},{"role":"assistant","content":"delta"}]}`,
	}, "\n")
	if err := os.WriteFile(trainPath, []byte(trainRows), 0o644); err != nil {
		t.Fatal(err)
	}

	backendName := "sft-cli-" + strings.NewReplacer("/", "-", " ", "-").Replace(t.Name())
	model := &sftCLITestModel{}
	backend := &sftCLITestBackend{name: backendName, model: model}
	inference.Register(backend)

	checkpointDir := filepath.Join(dir, "checkpoints")
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"sft",
		"-backend", backendName,
		"-model", "/models/gemma4",
		"-data", trainPath,
		"-checkpoint-dir", checkpointDir,
		"-rank", "4",
		"-alpha", "8",
		"-lr", "0.001",
		"-epochs", "3",
		"-batch", "2",
		"-grad-accum", "5",
		"-max-seq", "256",
		"-context", "2048",
		"-run-id", "sft-test-run",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("sft code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("sft stderr=%s", stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"steps 7",
		"epochs 3",
		"samples 2",
		"adapter " + filepath.Join(checkpointDir, "adapter.safetensors"),
		"checkpoint adapter",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("sft stdout missing %q:\n%s", want, out)
		}
	}
	if backend.loadPath != "/models/gemma4" || backend.loadConfig.Backend != backendName || backend.loadConfig.ContextLen != 2048 {
		t.Fatalf("backend load path/config = %q %+v", backend.loadPath, backend.loadConfig)
	}
	if !model.closed {
		t.Fatalf("SFT model was not closed")
	}
	if len(model.samples) != 2 || model.samples[0].Prompt != "alpha" || model.samples[1].Response != "delta" {
		t.Fatalf("SFT samples=%+v", model.samples)
	}
	cfg := model.config
	if cfg.Epochs != 3 ||
		cfg.BatchSize != 2 ||
		cfg.GradientAccumulation != 5 ||
		cfg.LearningRate != 0.001 ||
		cfg.LoRA.Rank != 4 ||
		cfg.LoRA.Alpha != 8 ||
		cfg.Labels["training_stage"] != "native_lora_sft_execute" ||
		cfg.Labels["trainer_interface"] != "inference.SFTTrainer" ||
		cfg.Labels["max_sequence_length"] != "256" ||
		cfg.Labels["run_id"] != "sft-test-run" ||
		cfg.Labels["output_adapter_path"] != filepath.Join(checkpointDir, "adapter.safetensors") ||
		cfg.Labels["metrics_line_protocol_path"] != filepath.Join(checkpointDir, "metrics.lp") ||
		cfg.Labels["capture_path"] != filepath.Join(checkpointDir, "captures.jsonl") {
		t.Fatalf("unexpected SFT trainer config: %+v labels=%+v", cfg, cfg.Labels)
	}
}

func TestSFTCommandFailsClosedWhenBackendHasNoTrainer(t *testing.T) {
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "train.jsonl")
	if err := os.WriteFile(trainPath, []byte(`{"prompt":"alpha","response":"bravo"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	backendName := "sft-no-trainer-" + strings.NewReplacer("/", "-", " ", "-").Replace(t.Name())
	model := &traceCLITestModel{}
	inference.Register(&traceCLITestBackend{name: backendName, model: model})

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"sft",
		"-backend", backendName,
		"-model", "/models/gemma4",
		"-data", trainPath,
	}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("sft code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("sft wrote stdout on trainer failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "does not implement inference.SFTTrainer") {
		t.Fatalf("stderr missing SFTTrainer refusal: %s", stderr.String())
	}
	if !model.closed {
		t.Fatalf("non-trainer model was not closed")
	}
}

func TestSFTNamedCUDADefaultRefusesPendingRuntimeDispatch(t *testing.T) {
	oldName := commandName
	commandName = "lthn-cuda"
	defer func() { commandName = oldName }()

	dir := t.TempDir()
	trainPath := filepath.Join(dir, "train.jsonl")
	if err := os.WriteFile(trainPath, []byte(`{"prompt":"alpha","response":"bravo"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"sft",
		"-model", "/models/gemma4",
		"-data", trainPath,
	}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("sft code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("sft wrote stdout on pending dispatch: %s", stdout.String())
	}
	for _, want := range []string{
		"cuda runtime lane is compiled and packaged",
		"rocm_kernels_nvidia_sm_75.o",
		"runtime_dispatch_status=compile_ready_runtime_dispatch_pending",
		"pass -backend rocm",
	} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("stderr missing %q:\n%s", want, stderr.String())
		}
	}
}

func TestSFTCommandRejectsMissingModelAndData(t *testing.T) {
	for _, args := range [][]string{
		{"sft", "-json"},
		{"sft", "-json", "-model", "/models/gemma4"},
	} {
		var stdout, stderr bytes.Buffer
		code := runCommand(context.Background(), args, &stdout, &stderr)
		if code != 2 {
			t.Fatalf("sft %v code=%d, want 2", args, code)
		}
		if stdout.Len() != 0 {
			t.Fatalf("sft %v wrote stdout on validation failure: %s", args, stdout.String())
		}
		if !strings.Contains(stderr.String(), "required") {
			t.Fatalf("sft %v stderr missing validation: %s", args, stderr.String())
		}
	}
}

type sftCLITestBackend struct {
	name       string
	model      *sftCLITestModel
	loadPath   string
	loadConfig inference.LoadConfig
}

func (b *sftCLITestBackend) Name() string {
	return b.name
}

func (b *sftCLITestBackend) Available() bool {
	return true
}

func (b *sftCLITestBackend) LoadModel(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
	b.loadPath = path
	b.loadConfig = inference.ApplyLoadOpts(opts)
	return b.model, nil
}

type sftCLITestModel struct {
	samples []inference.DatasetSample
	config  inference.TrainingConfig
	closed  bool
}

func (m *sftCLITestModel) TrainSFT(_ context.Context, dataset inference.DatasetStream, cfg inference.TrainingConfig) (*inference.TrainingResult, error) {
	m.config = cfg
	for {
		sample, ok, err := dataset.Next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		m.samples = append(m.samples, sample)
	}
	adapterPath := cfg.Labels["output_adapter_path"]
	checkpointDir := cfg.Labels["checkpoint_dir"]
	return &inference.TrainingResult{
		Adapter: inference.AdapterIdentity{
			Path:       adapterPath,
			Format:     "lora",
			Rank:       cfg.LoRA.Rank,
			Alpha:      cfg.LoRA.Alpha,
			TargetKeys: append([]string(nil), cfg.LoRA.TargetKeys...),
		},
		Metrics: inference.TrainingMetrics{
			Epoch:        cfg.Epochs,
			Step:         7,
			Samples:      len(m.samples),
			Loss:         0.125,
			LearningRate: cfg.LearningRate,
		},
		Checkpoints: []inference.StateRef{
			{Kind: "adapter", URI: filepath.Join(checkpointDir, "step-7.safetensors")},
		},
		Labels: cfg.Labels,
	}, nil
}

func (m *sftCLITestModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		_ = yield(inference.Token{Text: "sft"})
	}
}

func (m *sftCLITestModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		_ = yield(inference.Token{Text: "sft"})
	}
}

func (m *sftCLITestModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (m *sftCLITestModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (m *sftCLITestModel) ModelType() string {
	return "sft-cli-test"
}

func (m *sftCLITestModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "sft-cli-test"}
}

func (m *sftCLITestModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{}
}

func (m *sftCLITestModel) Err() error {
	return nil
}

func (m *sftCLITestModel) Close() error {
	m.closed = true
	return nil
}

func sftTestStringSliceContains(values []string, needle string) bool {
	for _, value := range values {
		if value == needle {
			return true
		}
	}
	return false
}

func TestSSDCommandJSONReportsNativePlan(t *testing.T) {
	dir := t.TempDir()
	dataPath := filepath.Join(dir, "prompts.jsonl")
	promptRows := strings.Join([]string{
		`{"prompt":"write a kernel","response":"seed"}`,
		`{"text":"freeform prompt"}`,
	}, "\n")
	if err := os.WriteFile(dataPath, []byte(promptRows), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"ssd",
		"-json",
		"-model", "/models/gemma4",
		"-data", dataPath,
		"-kernel", "kernel-prefix.hip",
		"-sample-max-tokens", "64",
		"-sample-temp", "0.7",
		"-sample-top-k", "32",
		"-sample-top-p", "0.9",
		"-sample-min-p", "0.05",
		"-rep-penalty", "1.2",
		"-filter-shortest", "25",
		"-score-samples=false",
		"-checkpoint-dir", "ssd-run",
		"-capture", "ssd-captures.jsonl",
		"-metrics-lp", "ssd.lp",
		"-influx-url", "http://127.0.0.1:8086",
		"-influx-token", "secret-token",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("ssd code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("ssd JSON wrote stderr: %s", stderr.String())
	}
	if strings.Contains(stdout.String(), "secret-token") {
		t.Fatalf("ssd JSON leaked influx token: %s", stdout.String())
	}
	var report ssdPlanReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("ssd JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Backend != "rocm" || report.Command != "ssd" || report.CLIContract != cliContractName || !report.NoPython {
		t.Fatalf("unexpected SSD report identity: %+v", report)
	}
	if report.Kind != "native-simple-self-distillation-trace-plan" || report.ModelPath != "/models/gemma4" || report.DataPath != dataPath {
		t.Fatalf("unexpected SSD plan paths: %+v", report)
	}
	if report.Config.Sample.SampleMaxTokens != 64 ||
		report.Config.Sample.SampleTemperature != 0.7 ||
		report.Config.Sample.SampleTopK != 32 ||
		report.Config.Sample.SampleTopP != 0.9 ||
		report.Config.Sample.SampleMinP != 0.05 ||
		report.Config.Sample.RepetitionPenalty != 1.2 ||
		report.Config.Sample.FilterShortestPercent != 25 ||
		report.Config.ScoreSamples {
		t.Fatalf("unexpected SSD sample config: %+v", report.Config)
	}
	if report.Dataset.Samples != 2 ||
		report.Dataset.PromptRows != 1 ||
		report.Dataset.ResponseRows != 1 ||
		report.Dataset.TextRows != 1 ||
		!sftTestStringSliceContains(report.Dataset.Formats, "prompt_response") ||
		!sftTestStringSliceContains(report.Dataset.Formats, "text") {
		t.Fatalf("unexpected SSD dataset summary: %+v", report.Dataset)
	}
	if report.CheckpointDir != "ssd-run" ||
		report.CapturePath != "ssd-captures.jsonl" ||
		report.KernelPath != "kernel-prefix.hip" ||
		!report.Metrics.InfluxConfigured ||
		!report.Metrics.InfluxTokenSet ||
		report.Metrics.LineProtocolPath != "ssd.lp" {
		t.Fatalf("unexpected SSD output/metrics plan: %+v", report)
	}
	if report.Labels["training_stage"] != "native_simple_self_distillation_trace_plan" ||
		report.Labels["ssd_runner"] != "RunModelSimpleSelfDistillation" ||
		report.Labels["ssd_stops_at"] != "scored_trace" ||
		report.Labels["next_training_command"] != "sft" ||
		report.Labels["model_loader_interface"] != "linked" ||
		report.Labels["native_ssd_runner"] != "linked" ||
		report.Labels["score_samples"] != "false" ||
		report.Labels["sample_temperature"] != "0.7" ||
		report.Labels["sample_top_k"] != "32" ||
		report.Labels["sample_top_p"] != "0.9" ||
		report.Labels["sample_min_p"] != "0.05" ||
		report.Labels["repetition_penalty"] != "1.2" ||
		report.Labels["filter_shortest_percent"] != "25" ||
		report.Labels["production_requires_env_gate"] != "false" ||
		report.Labels["production_requires_cli_flag"] != "false" {
		t.Fatalf("unexpected SSD labels: %+v", report.Labels)
	}
}

func TestSSDCommandRunsTraceAndWritesSidecars(t *testing.T) {
	dir := t.TempDir()
	dataPath := filepath.Join(dir, "prompts.jsonl")
	promptRows := strings.Join([]string{
		`{"prompt":"alpha","response":"seed","labels":{"source":"test"}}`,
		`{"text":"beta"}`,
	}, "\n")
	if err := os.WriteFile(dataPath, []byte(promptRows), 0o644); err != nil {
		t.Fatal(err)
	}

	backendName := "ssd-cli-" + strings.NewReplacer("/", "-", " ", "-").Replace(t.Name())
	model := &ssdCLITestModel{}
	backend := &ssdCLITestBackend{name: backendName, model: model}
	inference.Register(backend)

	checkpointDir := filepath.Join(dir, "ssd-run")
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"ssd",
		"-backend", backendName,
		"-model", "/models/gemma4",
		"-data", dataPath,
		"-checkpoint-dir", checkpointDir,
		"-sample-max-tokens", "3",
		"-sample-temp", "0.6",
		"-sample-top-k", "7",
		"-sample-top-p", "0.91",
		"-sample-min-p", "0.04",
		"-rep-penalty", "1.3",
		"-filter-shortest", "0",
		"-context", "4096",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("ssd code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("ssd stderr=%s", stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"self-samples 2",
		"ssd trace",
		"sample-score mean",
		"next: refine the trace",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("ssd stdout missing %q:\n%s", want, out)
		}
	}
	if backend.loadPath != "/models/gemma4" || backend.loadConfig.Backend != backendName || backend.loadConfig.ContextLen != 4096 {
		t.Fatalf("backend load path/config = %q %+v", backend.loadPath, backend.loadConfig)
	}
	if !model.closed {
		t.Fatalf("SSD model was not closed")
	}
	if len(model.prompts) != 2 || model.prompts[0] != "alpha" || model.prompts[1] != "beta" {
		t.Fatalf("SSD prompts=%q", model.prompts)
	}
	if len(model.calls) != 2 ||
		model.calls[0].MaxTokens != 3 ||
		model.calls[0].Temperature != 0.6 ||
		model.calls[0].TopK != 7 ||
		model.calls[0].TopP != 0.91 ||
		model.calls[0].MinP != 0.04 ||
		model.calls[0].RepeatPenalty != 1.3 {
		t.Fatalf("SSD generate configs=%+v", model.calls)
	}

	traceRecords := readSSDTraceJSONL(t, filepath.Join(checkpointDir, "ssd-captures.jsonl"))
	if len(traceRecords) != 2 {
		t.Fatalf("trace rows=%d, want 2: %+v", len(traceRecords), traceRecords)
	}
	if traceRecords[0].Prompt != "alpha" || traceRecords[0].Response != "response for alpha" || traceRecords[0].Labels["ssd"] != "simple_self_distillation" {
		t.Fatalf("unexpected first trace row: %+v", traceRecords[0])
	}
	scoreRecords := readSSDScoreJSONL(t, filepath.Join(checkpointDir, "ssd-samples-score.jsonl"))
	if len(scoreRecords) != 2 || scoreRecords[1].Prompt != "beta" || scoreRecords[1].Response != "response for beta" {
		t.Fatalf("unexpected score rows: %+v", scoreRecords)
	}
}

func TestSSDNamedCPUDefaultRefusesPendingRuntimeDispatch(t *testing.T) {
	oldName := commandName
	commandName = "lthn-cpu-x86"
	defer func() { commandName = oldName }()

	dir := t.TempDir()
	dataPath := filepath.Join(dir, "prompts.jsonl")
	if err := os.WriteFile(dataPath, []byte(`{"prompt":"alpha","response":"seed"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"ssd",
		"-model", "/models/gemma4",
		"-data", dataPath,
	}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("ssd code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("ssd wrote stdout on pending dispatch: %s", stdout.String())
	}
	for _, want := range []string{
		"cpu runtime lane is compiled and packaged",
		"rocm_kernels_hip_cpu_x86_64.o",
		"runtime_dispatch_status=compile_ready_runtime_dispatch_pending",
		"pass -backend rocm",
	} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("stderr missing %q:\n%s", want, stderr.String())
		}
	}
}

func TestSSDCommandRejectsMissingModelAndData(t *testing.T) {
	for _, args := range [][]string{
		{"ssd", "-json"},
		{"ssd", "-json", "-model", "/models/gemma4"},
	} {
		var stdout, stderr bytes.Buffer
		code := runCommand(context.Background(), args, &stdout, &stderr)
		if code != 2 {
			t.Fatalf("ssd %v code=%d, want 2", args, code)
		}
		if stdout.Len() != 0 {
			t.Fatalf("ssd %v wrote stdout on validation failure: %s", args, stdout.String())
		}
		if !strings.Contains(stderr.String(), "required") {
			t.Fatalf("ssd %v stderr missing validation: %s", args, stderr.String())
		}
	}
}

func TestSSDCommandRejectsInvalidSamplingFlags(t *testing.T) {
	dir := t.TempDir()
	dataPath := filepath.Join(dir, "prompts.jsonl")
	if err := os.WriteFile(dataPath, []byte(`{"prompt":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name string
		flag string
		val  string
		want string
	}{
		{name: "unit-temp", flag: "-sample-temp", val: "1", want: "sample temperature"},
		{name: "negative-top-k", flag: "-sample-top-k", val: "-1", want: "sample top-k"},
		{name: "bad-top-p", flag: "-sample-top-p", val: "1.1", want: "sample top-p"},
		{name: "bad-min-p", flag: "-sample-min-p", val: "-0.1", want: "sample min-p"},
		{name: "bad-repetition", flag: "-rep-penalty", val: "-1", want: "repetition penalty"},
		{name: "bad-filter", flag: "-filter-shortest", val: "101", want: "filter shortest"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runCommand(context.Background(), []string{
				"ssd",
				"-json",
				"-model", "/models/gemma4",
				"-data", dataPath,
				tc.flag, tc.val,
			}, &stdout, &stderr)
			if code != 2 {
				t.Fatalf("ssd code=%d, want 2; stdout=%s stderr=%s", code, stdout.String(), stderr.String())
			}
			if stdout.Len() != 0 {
				t.Fatalf("ssd wrote stdout on validation failure: %s", stdout.String())
			}
			if !strings.Contains(stderr.String(), tc.want) {
				t.Fatalf("ssd stderr=%s, want %q", stderr.String(), tc.want)
			}
		})
	}
}

func readSSDTraceJSONL(t *testing.T, path string) []ssdTraceJSONLRecord {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	records := make([]ssdTraceJSONLRecord, 0, len(lines))
	for _, line := range lines {
		var record ssdTraceJSONLRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			t.Fatalf("decode trace JSONL %s: %v", path, err)
		}
		records = append(records, record)
	}
	return records
}

func readSSDScoreJSONL(t *testing.T, path string) []ssdScoreJSONLRecord {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	records := make([]ssdScoreJSONLRecord, 0, len(lines))
	for _, line := range lines {
		var record ssdScoreJSONLRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			t.Fatalf("decode score JSONL %s: %v", path, err)
		}
		records = append(records, record)
	}
	return records
}

type ssdCLITestBackend struct {
	name       string
	model      *ssdCLITestModel
	loadPath   string
	loadConfig inference.LoadConfig
}

func (b *ssdCLITestBackend) Name() string {
	return b.name
}

func (b *ssdCLITestBackend) Available() bool {
	return true
}

func (b *ssdCLITestBackend) LoadModel(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
	b.loadPath = path
	b.loadConfig = inference.ApplyLoadOpts(opts)
	return b.model, nil
}

type ssdCLITestModel struct {
	prompts []string
	calls   []inference.GenerateConfig
	closed  bool
}

func (m *ssdCLITestModel) Generate(_ context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.prompts = append(m.prompts, prompt)
	m.calls = append(m.calls, inference.ApplyGenerateOpts(opts))
	return func(yield func(inference.Token) bool) {
		if !yield(inference.Token{Text: "response for "}) {
			return
		}
		_ = yield(inference.Token{Text: prompt})
	}
}

func (m *ssdCLITestModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		_ = yield(inference.Token{Text: "chat"})
	}
}

func (m *ssdCLITestModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (m *ssdCLITestModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (m *ssdCLITestModel) ModelType() string {
	return "ssd-cli-test"
}

func (m *ssdCLITestModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "ssd-cli-test"}
}

func (m *ssdCLITestModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{}
}

func (m *ssdCLITestModel) Err() error {
	return nil
}

func (m *ssdCLITestModel) Close() error {
	m.closed = true
	return nil
}

func TestSSDRecipesJSONReportsROCmManifest(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"ssd-recipes", "-json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("ssd-recipes code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("ssd-recipes JSON wrote stderr: %s", stderr.String())
	}
	var report ssdRecipesReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("ssd-recipes JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Backend != "rocm" || report.Command != "ssd-recipes" || report.CLIContract != cliContractName || !report.NoPython {
		t.Fatalf("unexpected SSD recipe report identity: %+v", report)
	}
	if report.TrainDefault.SampleMaxTokens != 65536 ||
		report.TrainDefault.SampleTemperature != 1.5 ||
		report.TrainDefault.SampleTopK != 20 ||
		report.TrainDefault.SampleTopP != 0.8 ||
		report.TrainDefault.RepetitionPenalty != 1 ||
		report.TrainDefault.FilterShortestPercent != 10 {
		t.Fatalf("train defaults = %+v, want ml-ssd recipe defaults", report.TrainDefault)
	}
	if report.EvalDefault.Benchmark != "LiveCodeBench-v6" ||
		report.EvalDefault.NRepeat != 20 ||
		report.EvalDefault.Generate.MaxTokens != 32768 ||
		report.EvalDefault.Generate.TopK != 20 {
		t.Fatalf("eval defaults = %+v, want LiveCodeBench-v6 defaults", report.EvalDefault)
	}
	if len(report.Recipes) != 3 || report.Recipes[1].Model != "apple/SimpleSD-4B-thinking" {
		t.Fatalf("recipes = %+v, want released SimpleSD recipes", report.Recipes)
	}
}

func TestSSDEvalJSONPlansLiveCodeBench(t *testing.T) {
	samples := filepath.Join(t.TempDir(), "lcb.jsonl")
	raw := `{"id":"old","prompt":"old","contest_date":"2025-01-01"}` + "\n" +
		`{"question_id":"new","question_content":"solve me","starter_code":"def f(): pass","entry_point":"f","is_stdin":false,"contest_date":"2025-03-02","public_test_cases":["assert f()==1"],"difficulty":"easy","platform":"lcb"}`
	if err := os.WriteFile(samples, []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"ssd-eval",
		"-json",
		"-samples", samples,
		"-output", "plan.json",
		"-n-repeat", "4",
		"-sampling-params", "temperature=0.9,top_p=0.8,top_k=12",
		"-max-tokens", "512",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("ssd-eval code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("ssd-eval JSON wrote stderr: %s", stderr.String())
	}
	var report ssdEvalPlanReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("ssd-eval JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Backend != "rocm" || report.Command != "ssd-eval" || report.CLIContract != cliContractName || !report.LiveCodeBench || report.Samples != 1 {
		t.Fatalf("unexpected SSD eval plan identity: %+v", report)
	}
	if report.OutputPath != "plan.json" ||
		report.Config.NRepeat != 4 ||
		report.Config.Generate.MaxTokens != 512 ||
		report.Config.Generate.Temperature != 0.9 ||
		report.Config.Generate.TopP != 0.8 ||
		report.Config.Generate.TopK != 12 {
		t.Fatalf("SSD eval config = %+v, want overrides", report.Config)
	}
}

func TestSSDEvalRejectsMissingSamples(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"ssd-eval", "-json"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("ssd-eval code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("ssd-eval wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "samples path is required") {
		t.Fatalf("stderr missing samples validation: %s", stderr.String())
	}
}

func TestRunFuseCommandMissingFlagsBad(t *testing.T) {
	for _, args := range [][]string{
		{"fuse"},
		{"fuse", "-base", "/tmp/base"},
		{"fuse", "-base", "/tmp/base", "-adapter", "/tmp/adapter"},
	} {
		var stdout, stderr bytes.Buffer
		code := runCommand(context.Background(), args, &stdout, &stderr)
		if code != 2 {
			t.Fatalf("fuse %v code=%d, want 2", args, code)
		}
		if stdout.Len() != 0 {
			t.Fatalf("fuse %v wrote stdout: %s", args, stdout.String())
		}
		if !strings.Contains(stderr.String(), "-base, -adapter and -out are all required") {
			t.Fatalf("fuse %v stderr missing required flags: %s", args, stderr.String())
		}
	}
}

func TestRunFuseCommandJSONReportsBoundary(t *testing.T) {
	base := t.TempDir()
	if err := os.WriteFile(filepath.Join(base, "config.json"), []byte(`{"model_type":"gemma4_text","num_hidden_layers":1,"hidden_size":2,"vocab_size":8}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(base, "model.safetensors"), make([]byte, 8), 0o644); err != nil {
		t.Fatal(err)
	}
	adapter := t.TempDir()
	if err := os.WriteFile(filepath.Join(adapter, "adapter.safetensors"), []byte("adapter"), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "fused")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"fuse", "-json", "-base", base, "-adapter", adapter, "-out", out}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("fuse code=%d, want 1", code)
	}
	if stderr.Len() != 0 {
		t.Fatalf("fuse JSON wrote stderr: %s", stderr.String())
	}
	var report fuseReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("fuse JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Backend != "rocm" || report.Command != "fuse" || report.CLIContract != cliContractName {
		t.Fatalf("unexpected fuse report identity: %+v", report)
	}
	if report.Status != "unsupported" || report.Feature != "LoRA on-disk fuse" {
		t.Fatalf("unexpected fuse report status: %+v", report)
	}
	if report.Capability.ID != inference.CapabilityModelMerge || report.Capability.Status != inference.CapabilityStatusExperimental {
		t.Fatalf("fuse capability = %+v, want experimental model.merge", report.Capability)
	}
	if report.Inputs["base"] != base || report.Inputs["adapter"] != adapter || report.Inputs["out"] != out {
		t.Fatalf("fuse inputs not preserved: %+v", report.Inputs)
	}
	if report.Base == nil || report.Base.Path == "" {
		t.Fatalf("fuse base inspection missing: %+v", report.Base)
	}
	if report.Adapter.Path != adapter || report.Adapter.Format != "lora" || report.Adapter.Labels["adapter_file"] != "adapter.safetensors" {
		t.Fatalf("fuse adapter identity = %+v", report.Adapter)
	}
	if !strings.Contains(report.Detail, "safetensors") && !strings.Contains(report.Detail, "rank") {
		t.Fatalf("fuse detail = %q, want unsupported fuse refusal", report.Detail)
	}
}

func TestRunSliceCommandJSONMaterializesSafetensorsSubset(t *testing.T) {
	source := t.TempDir()
	if err := os.WriteFile(filepath.Join(source, "config.json"), []byte(`{
		"model_type":"qwen2",
		"hidden_size":4,
		"num_hidden_layers":1,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":4,
		"vocab_size":16,
		"max_position_embeddings":32
	}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "tokenizer.json"), []byte(`{"model":{"type":"BPE"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLITestSafetensors(t, filepath.Join(source, "model.safetensors"), map[string]cliTestTensor{
		"model.embed_tokens.weight":              {DType: "F32", Shape: []uint64{1}, Values: []float32{1}},
		"model.layers.0.input_layernorm.weight":  {DType: "F32", Shape: []uint64{1}, Values: []float32{2}},
		"model.layers.0.self_attn.q_proj.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{3}},
		"model.layers.0.self_attn.k_proj.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{4}},
		"model.layers.0.self_attn.v_proj.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{5}},
		"model.layers.0.self_attn.o_proj.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{6}},
		"model.layers.0.mlp.down_proj.weight":    {DType: "F32", Shape: []uint64{1}, Values: []float32{7}},
		"lm_head.weight":                         {DType: "F32", Shape: []uint64{1}, Values: []float32{8}},
	})
	output := filepath.Join(t.TempDir(), "client-slice")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"slice", "-json", "-preset", "client", "-output", output, source}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("slice code=%d, want 0; stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("slice JSON wrote stderr: %s", stderr.String())
	}
	var plan inference.ModelSlicePlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("slice JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if plan.Preset != inference.ModelSlicePresetClient || plan.OutputPath != output || plan.SourcePath != source {
		t.Fatalf("slice plan = %+v, want client plan paths", plan)
	}
	if plan.Labels["backend"] != "rocm" ||
		plan.Labels["slice_runtime"] != "native_safetensors_subset" ||
		plan.Labels["tensor_count"] != "7" ||
		plan.Labels["selected_tensor_bytes"] != "28" ||
		plan.Labels["source_tensor_bytes"] != "32" {
		t.Fatalf("slice labels = %+v, want ROCm subset byte labels", plan.Labels)
	}
	if values := readCLITestSafetensorsF32(t, filepath.Join(output, "model.safetensors"), "model.layers.0.self_attn.q_proj.weight"); len(values) != 1 || values[0] != 3 {
		t.Fatalf("selected q_proj = %v, want [3]", values)
	}
	if values := readCLITestSafetensorsF32(t, filepath.Join(output, "model.safetensors"), "lm_head.weight"); len(values) != 1 || values[0] != 8 {
		t.Fatalf("selected lm_head = %v, want [8]", values)
	}
	if _, err := os.Stat(filepath.Join(output, "slice_manifest.json")); err != nil {
		t.Fatalf("slice_manifest.json not written: %v", err)
	}
	if _, ok := readCLITestSafetensorsHeader(t, filepath.Join(output, "model.safetensors"))["model.layers.0.mlp.down_proj.weight"]; ok {
		t.Fatal("client slice retained FFN tensor, want attention/client subset")
	}
}

func TestRunSliceCommandRejectsMissingOutput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"slice", "/models/qwen2"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("slice code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("slice wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "-output is required") {
		t.Fatalf("stderr missing output validation: %s", stderr.String())
	}
}

func TestRunEbookCommandJSONWritesEPUB(t *testing.T) {
	source := t.TempDir()
	if err := os.WriteFile(filepath.Join(source, "config.json"), []byte(`{"model_type":"qwen2","num_hidden_layers":1,"hidden_size":4,"vocab_size":16}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "README.md"), []byte("A small model book.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLITestSafetensors(t, filepath.Join(source, "model.safetensors"), map[string]cliTestTensor{
		"model.embed_tokens.weight":              {DType: "F32", Shape: []uint64{1}, Values: []float32{1}},
		"model.layers.0.self_attn.q_proj.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{2}},
	})
	out := filepath.Join(t.TempDir(), "model.epub")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"ebook", "-json", "-model", source, "-out", out, "-title", "Tiny Model", "-chapter-chars", "32"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("ebook code=%d, want 0; stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("ebook JSON wrote stderr: %s", stderr.String())
	}
	var report ebookCommandReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("ebook JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.OutputPath != out ||
		report.Title != "Tiny Model" ||
		!report.IncludeWeights ||
		report.Chapters < 5 ||
		report.NavChapters < 4 ||
		report.Bytes <= 0 ||
		report.Labels["ebook_runtime"] != "native_epub3" ||
		report.Labels["production_requires_env_gate"] != "false" {
		t.Fatalf("ebook report = %+v, want native EPUB3 output", report)
	}

	zr, err := zip.OpenReader(out)
	if err != nil {
		t.Fatalf("open epub zip: %v", err)
	}
	defer zr.Close()
	if len(zr.File) == 0 || zr.File[0].Name != "mimetype" || zr.File[0].Method != zip.Store {
		t.Fatalf("first epub entry = %+v, want stored mimetype", zr.File)
	}
	if got := readCLIZipEntry(t, &zr.Reader, "mimetype"); got != "application/epub+zip" {
		t.Fatalf("mimetype = %q", got)
	}
	opf := readCLIZipEntry(t, &zr.Reader, "OEBPS/content.opf")
	for _, want := range []string{"<dc:title>Tiny Model</dc:title>", `href="plate0001.xhtml"`, "dcterms:modified"} {
		if !strings.Contains(opf, want) {
			t.Fatalf("content.opf missing %q:\n%s", want, opf)
		}
	}
	foreword := readCLIZipEntry(t, &zr.Reader, "OEBPS/ch001-foreword.xhtml")
	if !strings.Contains(foreword, "A small model book.") {
		t.Fatalf("foreword missing README text:\n%s", foreword)
	}
	plate := readCLIZipEntry(t, &zr.Reader, "OEBPS/plate0001.xhtml")
	if !strings.Contains(plate, "<pre>") {
		t.Fatalf("plate missing base64 payload:\n%s", plate)
	}
}

func TestRunEbookCommandRejectsMissingModel(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"ebook", "-json"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("ebook code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("ebook wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "-model is required") {
		t.Fatalf("stderr missing model validation: %s", stderr.String())
	}
}

func readCLIZipEntry(t *testing.T, zr *zip.Reader, name string) string {
	t.Helper()
	for _, file := range zr.File {
		if file.Name != name {
			continue
		}
		rc, err := file.Open()
		if err != nil {
			t.Fatalf("open zip entry %s: %v", name, err)
		}
		defer rc.Close()
		data, err := io.ReadAll(rc)
		if err != nil {
			t.Fatalf("read zip entry %s: %v", name, err)
		}
		return string(data)
	}
	t.Fatalf("zip entry %s not found", name)
	return ""
}

func TestRunFuseCommandJSONReportsPEFTAdapterConfig(t *testing.T) {
	base := t.TempDir()
	if err := os.WriteFile(filepath.Join(base, "config.json"), []byte(`{"model_type":"gemma4_text","num_hidden_layers":1,"hidden_size":2,"vocab_size":8}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(base, "model.safetensors"), make([]byte, 8), 0o644); err != nil {
		t.Fatal(err)
	}
	adapter := t.TempDir()
	config := `{
		"peft_type":"LORA",
		"base_model_name_or_path":"google/gemma-4-E4B-it",
		"task_type":"CAUSAL_LM",
		"r":8,
		"lora_alpha":16,
		"target_modules":["q_proj","mlp.up_proj","router.proj","vision_tower.q_proj"]
	}`
	if err := os.WriteFile(filepath.Join(adapter, "adapter_config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(adapter, "adapter.safetensors"), []byte("adapter"), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "fused")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"fuse", "-json", "-base", base, "-adapter", adapter, "-out", out}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("fuse code=%d, want 1", code)
	}
	if stderr.Len() != 0 {
		t.Fatalf("fuse JSON wrote stderr: %s", stderr.String())
	}
	var report fuseReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("fuse JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Adapter.Rank != 8 || report.Adapter.Alpha != 16 || strings.Join(report.Adapter.TargetKeys, ",") != "q_proj,mlp.up_proj,router.proj,vision_tower.q_proj" {
		t.Fatalf("adapter config was not reflected: %+v", report.Adapter)
	}
	labels := report.Adapter.Labels
	if labels["adapter_config_format"] != "peft" || labels["adapter_weights_file"] != "adapter.safetensors" || labels["adapter_scale"] != "2" {
		t.Fatalf("adapter labels missing config metadata: %+v", labels)
	}
	if labels["adapter_target_policy"] != "gemma4" || labels["adapter_default_targets"] != "q_proj,v_proj,o_proj" {
		t.Fatalf("adapter labels missing Gemma4 policy: %+v", labels)
	}
	if labels["adapter_canonical_targets"] != "self_attn.q_proj,mlp.up_proj,router.proj" {
		t.Fatalf("adapter canonical targets = %q", labels["adapter_canonical_targets"])
	}
	if labels["adapter_extended_targets_require_opt_in"] != "true" || labels["adapter_extended_targets_present"] != "router.proj" {
		t.Fatalf("adapter extended target labels missing: %+v", labels)
	}
	if labels["adapter_unknown_targets"] != "vision_tower.q_proj" || labels["adapter_safe_for_default_fuse"] != "false" {
		t.Fatalf("adapter unknown/safety labels missing: %+v", labels)
	}
}

func TestRunFuseCommandJSONFusesDenseSafetensors(t *testing.T) {
	base := t.TempDir()
	if err := os.WriteFile(filepath.Join(base, "config.json"), []byte(`{"model_type":"gemma4_text","num_hidden_layers":1,"hidden_size":3,"vocab_size":8}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLITestSafetensors(t, filepath.Join(base, "model.safetensors"), map[string]cliTestTensor{
		"model.layers.0.self_attn.q_proj.weight": {DType: "F32", Shape: []uint64{2, 3}, Values: []float32{1, 2, 3, 4, 5, 6}},
		"model.norm.weight":                      {DType: "F32", Shape: []uint64{3}, Values: []float32{7, 8, 9}},
	})
	adapter := t.TempDir()
	if err := os.WriteFile(filepath.Join(adapter, "adapter_config.json"), []byte(`{"r":1,"lora_alpha":2,"target_modules":["q_proj"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLITestSafetensors(t, filepath.Join(adapter, "adapter.safetensors"), map[string]cliTestTensor{
		"model.layers.0.q_proj.lora_A.weight": {DType: "F32", Shape: []uint64{1, 3}, Values: []float32{0.1, 0.2, 0.3}},
		"model.layers.0.q_proj.lora_B.weight": {DType: "F32", Shape: []uint64{2, 1}, Values: []float32{2, 3}},
	})
	out := filepath.Join(t.TempDir(), "fused")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"fuse", "-json", "-base", base, "-adapter", adapter, "-out", out}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("fuse code=%d, want 0\nstdout=%s\nstderr=%s", code, stdout.String(), stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("fuse JSON wrote stderr: %s", stderr.String())
	}
	var report fuseReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("fuse JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Status != "ok" || report.Result == nil || report.Result.FusedWeights != 1 {
		t.Fatalf("unexpected fuse success report: %+v", report)
	}
	if strings.Join(report.Result.FusedLayers, ",") != "model.layers.0.self_attn.q_proj" {
		t.Fatalf("fused layers = %+v, want go-mlx-compatible target layer", report.Result.FusedLayers)
	}
	if report.Result.Labels["fuse_runtime"] != "dense_f32_cpu" {
		t.Fatalf("result labels = %+v, want dense fuse runtime", report.Result.Labels)
	}
	if report.Result.Labels["fuse_layer_count"] != "1" {
		t.Fatalf("result labels = %+v, want one fused layer", report.Result.Labels)
	}
	got := readCLITestSafetensorsF32(t, filepath.Join(out, "model.safetensors"), "model.layers.0.self_attn.q_proj.weight")
	want := []float32{1.4, 2.8, 4.2, 4.6, 6.2, 7.8}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("fused[%d] = %.4f, want %.4f (all=%v)", i, got[i], want[i], got)
		}
	}
	if _, err := os.Stat(filepath.Join(out, rocm.LoRAFuseProvenanceFile)); err != nil {
		t.Fatalf("provenance missing: %v", err)
	}
}

func TestRunFuseCommandJSONFusesRankOnlyPEFTDefaultScale(t *testing.T) {
	base := t.TempDir()
	if err := os.WriteFile(filepath.Join(base, "config.json"), []byte(`{"model_type":"gemma4_text","num_hidden_layers":1,"hidden_size":3,"vocab_size":8}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLITestSafetensors(t, filepath.Join(base, "model.safetensors"), map[string]cliTestTensor{
		"model.layers.0.self_attn.q_proj.weight": {DType: "F32", Shape: []uint64{2, 3}, Values: []float32{1, 2, 3, 4, 5, 6}},
	})
	adapter := t.TempDir()
	if err := os.WriteFile(filepath.Join(adapter, "adapter_config.json"), []byte(`{"r":1,"target_modules":["q_proj"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLITestSafetensors(t, filepath.Join(adapter, "adapter.safetensors"), map[string]cliTestTensor{
		"model.layers.0.q_proj.lora_A.weight": {DType: "F32", Shape: []uint64{1, 3}, Values: []float32{0.1, 0.2, 0.3}},
		"model.layers.0.q_proj.lora_B.weight": {DType: "F32", Shape: []uint64{2, 1}, Values: []float32{2, 3}},
	})
	out := filepath.Join(t.TempDir(), "fused")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"fuse", "-json", "-base", base, "-adapter", adapter, "-out", out}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("fuse code=%d, want 0\nstdout=%s\nstderr=%s", code, stdout.String(), stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("fuse JSON wrote stderr: %s", stderr.String())
	}
	var report fuseReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("fuse JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Status != "ok" || report.Result == nil || report.Result.FusedWeights != 1 {
		t.Fatalf("unexpected fuse success report: %+v", report)
	}
	if report.Adapter.Rank != 1 || report.Adapter.Alpha != 2 ||
		report.Adapter.Labels["adapter_alpha_source"] != "default_rank_x2" ||
		report.Adapter.Labels["adapter_scale"] != "2" {
		t.Fatalf("rank-only PEFT adapter did not receive production fuse defaults: %+v", report.Adapter)
	}
	got := readCLITestSafetensorsF32(t, filepath.Join(out, "model.safetensors"), "model.layers.0.self_attn.q_proj.weight")
	want := []float32{1.4, 2.8, 4.2, 4.6, 6.2, 7.8}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("rank-only fused[%d] = %.4f, want %.4f (all=%v)", i, got[i], want[i], got)
		}
	}
}

func TestRunFuseCommandPlainSuccessMatchesGoMLX(t *testing.T) {
	base := t.TempDir()
	if err := os.WriteFile(filepath.Join(base, "config.json"), []byte(`{"model_type":"gemma4_text","num_hidden_layers":1,"hidden_size":3,"vocab_size":8}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLITestSafetensors(t, filepath.Join(base, "model.safetensors"), map[string]cliTestTensor{
		"model.layers.0.self_attn.q_proj.weight": {DType: "F32", Shape: []uint64{2, 3}, Values: []float32{1, 2, 3, 4, 5, 6}},
		"model.norm.weight":                      {DType: "F32", Shape: []uint64{3}, Values: []float32{7, 8, 9}},
	})
	adapter := t.TempDir()
	if err := os.WriteFile(filepath.Join(adapter, "adapter_config.json"), []byte(`{"r":1,"lora_alpha":2,"target_modules":["q_proj"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLITestSafetensors(t, filepath.Join(adapter, "adapter.safetensors"), map[string]cliTestTensor{
		"model.layers.0.q_proj.lora_A.weight": {DType: "F32", Shape: []uint64{1, 3}, Values: []float32{0.1, 0.2, 0.3}},
		"model.layers.0.q_proj.lora_B.weight": {DType: "F32", Shape: []uint64{2, 1}, Values: []float32{2, 3}},
	})
	out := filepath.Join(t.TempDir(), "fused")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"fuse", "-base", base, "-adapter", adapter, "-out", out}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("fuse code=%d, want 0\nstdout=%s\nstderr=%s", code, stdout.String(), stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("fuse wrote stderr: %s", stderr.String())
	}
	want := "fused 1 layer(s) into " + out + "\n"
	if stdout.String() != want {
		t.Fatalf("stdout = %q, want %q", stdout.String(), want)
	}
}

func writeCLIDraftDetectModelDir(t *testing.T, dir, modelType string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"`+modelType+`","hidden_size":1,"num_hidden_layers":1,"vocab_size":1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLITestSafetensors(t, filepath.Join(dir, "model.safetensors"), map[string]cliTestTensor{
		"model.embed_tokens.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{0}},
	})
}

func writeCLIVisionModelDir(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	config := `{
		"architectures":["Gemma4ForConditionalGeneration"],
		"image_token_id":258880,
		"video_token_id":258884,
		"model_type":"gemma4",
		"text_config":{
			"model_type":"gemma4_text",
			"hidden_size":2304,
			"num_hidden_layers":26,
			"num_attention_heads":8,
			"num_key_value_heads":4,
			"head_dim":256,
			"vocab_size":262144,
			"max_position_embeddings":131072
		},
		"vision_config":{
			"dtype":"bfloat16",
			"default_output_length":280,
			"global_head_dim":64,
			"head_dim":64,
			"hidden_activation":"gelu_pytorch_tanh",
			"hidden_size":768,
			"intermediate_size":3072,
			"model_type":"gemma4_vision",
			"num_attention_heads":12,
			"num_hidden_layers":16,
			"num_key_value_heads":12,
			"patch_size":16,
			"pooling_kernel_size":3,
			"position_embedding_size":10240,
			"rope_parameters":{"rope_theta":100.0,"rope_type":"default"},
			"standardize":false,
			"use_clipped_linears":true
		},
		"vision_soft_tokens_per_image":280
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLITestSafetensors(t, filepath.Join(dir, "model.safetensors"), map[string]cliTestTensor{
		"model.embed_tokens.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{0}},
	})
}

func writeCLIAudioModelDir(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	config := `{
		"architectures":["Gemma4UnifiedForConditionalGeneration"],
		"audio_token_id":258881,
		"boa_token_id":256000,
		"eoa_token_index":258883,
		"model_type":"gemma4_unified",
		"text_config":{
			"model_type":"gemma4_unified_text",
			"hidden_size":3840,
			"num_hidden_layers":48,
			"num_attention_heads":16,
			"num_key_value_heads":8,
			"vocab_size":262144,
			"max_position_embeddings":262144
		},
		"audio_config":{
			"model_type":"gemma4_unified_audio",
			"hidden_size":1024,
			"audio_embed_dim":640,
			"audio_samples_per_token":640,
			"num_hidden_layers":12,
			"num_attention_heads":8,
			"attention_chunk_size":12,
			"attention_context_left":13,
			"conv_kernel_size":5,
			"output_proj_dims":1536,
			"rms_norm_eps":1e-6,
			"hidden_act":"silu",
			"use_clipped_linears":true
		}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLITestSafetensors(t, filepath.Join(dir, "model.safetensors"), map[string]cliTestTensor{
		"model.embed_tokens.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{0}},
	})
}

func writeCLIDiffusionModelDir(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	config := `{
		"architectures":["DiffusionGemmaForBlockDiffusion"],
		"model_type":"diffusion_gemma",
		"canvas_length":256,
		"hidden_size":2304,
		"num_hidden_layers":26,
		"num_attention_heads":8,
		"num_key_value_heads":4,
		"head_dim":256,
		"vocab_size":262144,
		"max_position_embeddings":131072
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	writeCLITestSafetensors(t, filepath.Join(dir, "model.safetensors"), map[string]cliTestTensor{
		"model.decoder.embed_tokens.weight": {DType: "F32", Shape: []uint64{1}, Values: []float32{0}},
	})
}

func writeCLITestWAVPCM16(t *testing.T, path string, sampleRate uint32, channels uint16, samples []int16) {
	t.Helper()
	if channels == 0 {
		t.Fatal("channels must be > 0")
	}
	dataBytes := len(samples) * 2
	payload := make([]byte, 44+dataBytes)
	copy(payload[0:4], "RIFF")
	binary.LittleEndian.PutUint32(payload[4:8], uint32(36+dataBytes))
	copy(payload[8:12], "WAVE")
	copy(payload[12:16], "fmt ")
	binary.LittleEndian.PutUint32(payload[16:20], 16)
	binary.LittleEndian.PutUint16(payload[20:22], 1)
	binary.LittleEndian.PutUint16(payload[22:24], channels)
	binary.LittleEndian.PutUint32(payload[24:28], sampleRate)
	byteRate := sampleRate * uint32(channels) * 2
	binary.LittleEndian.PutUint32(payload[28:32], byteRate)
	binary.LittleEndian.PutUint16(payload[32:34], channels*2)
	binary.LittleEndian.PutUint16(payload[34:36], 16)
	copy(payload[36:40], "data")
	binary.LittleEndian.PutUint32(payload[40:44], uint32(dataBytes))
	for i, sample := range samples {
		binary.LittleEndian.PutUint16(payload[44+i*2:46+i*2], uint16(sample))
	}
	if err := os.WriteFile(path, payload, 0o644); err != nil {
		t.Fatal(err)
	}
}

type cliTestTensor struct {
	DType  string
	Shape  []uint64
	Values []float32
}

type cliTestSafetensorHeader struct {
	DType       string   `json:"dtype"`
	Shape       []uint64 `json:"shape"`
	DataOffsets []uint64 `json:"data_offsets"`
}

func writeCLITestSafetensors(t *testing.T, path string, tensors map[string]cliTestTensor) {
	t.Helper()
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	header := map[string]cliTestSafetensorHeader{}
	payloads := make([][]byte, 0, len(names))
	var offset uint64
	for _, name := range names {
		tensor := tensors[name]
		if tensor.DType != "F32" {
			t.Fatalf("test writer only supports F32, got %s", tensor.DType)
		}
		payload := encodeCLITestF32s(tensor.Values...)
		header[name] = cliTestSafetensorHeader{
			DType:       tensor.DType,
			Shape:       append([]uint64(nil), tensor.Shape...),
			DataOffsets: []uint64{offset, offset + uint64(len(payload))},
		}
		payloads = append(payloads, payload)
		offset += uint64(len(payload))
	}
	headerBytes, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var headerLen [8]byte
	binary.LittleEndian.PutUint64(headerLen[:], uint64(len(headerBytes)))
	if _, err := file.Write(headerLen[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := file.Write(headerBytes); err != nil {
		t.Fatal(err)
	}
	for _, payload := range payloads {
		if _, err := file.Write(payload); err != nil {
			t.Fatal(err)
		}
	}
}

func readCLITestSafetensorsHeader(t *testing.T, path string) map[string]cliTestSafetensorHeader {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var headerLen uint64
	if err := binary.Read(file, binary.LittleEndian, &headerLen); err != nil {
		t.Fatal(err)
	}
	headerBytes := make([]byte, int(headerLen))
	if _, err := file.Read(headerBytes); err != nil {
		t.Fatal(err)
	}
	header := map[string]cliTestSafetensorHeader{}
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		t.Fatal(err)
	}
	return header
}

func readCLITestSafetensorsF32(t *testing.T, path, name string) []float32 {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var headerLen uint64
	if err := binary.Read(file, binary.LittleEndian, &headerLen); err != nil {
		t.Fatal(err)
	}
	headerBytes := make([]byte, int(headerLen))
	if _, err := file.Read(headerBytes); err != nil {
		t.Fatal(err)
	}
	header := map[string]cliTestSafetensorHeader{}
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		t.Fatal(err)
	}
	entry, ok := header[name]
	if !ok {
		t.Fatalf("tensor %q missing from %s", name, path)
	}
	if entry.DType != "F32" || len(entry.DataOffsets) != 2 {
		t.Fatalf("tensor entry = %+v, want F32 with offsets", entry)
	}
	payloadStart := int64(8 + headerLen + entry.DataOffsets[0])
	payloadLen := int(entry.DataOffsets[1] - entry.DataOffsets[0])
	raw := make([]byte, payloadLen)
	if _, err := file.ReadAt(raw, payloadStart); err != nil {
		t.Fatal(err)
	}
	values := make([]float32, payloadLen/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return values
}

func encodeCLITestF32s(values ...float32) []byte {
	out := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(value))
	}
	return out
}

func TestTuneJSONReportsReactiveMTPPlan(t *testing.T) {
	model := filepath.Join(t.TempDir(), "model")
	assistant := filepath.Join(model, "assistant")
	writeCLIDraftDetectModelDir(t, model, "gemma4_text")
	writeCLIDraftDetectModelDir(t, assistant, "gemma4_assistant")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"tune",
		"-model", model,
		"-depths", "4,5",
		"-max-tokens=16",
		"-profile-dir", "profiles",
		"-json",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("tune code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("tune JSON wrote stderr: %s", stderr.String())
	}
	var report tunePlanReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("tune JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Backend != "rocm" || report.Command != "tune" || report.CLIContract != cliContractName || !report.NoPython {
		t.Fatalf("unexpected tune report identity: %+v", report)
	}
	if report.Kind != "reactive-mtp-tuning-plan" || report.ModelPath != model || report.ProfileDir != "profiles" {
		t.Fatalf("unexpected tune plan paths: %+v", report)
	}
	if report.DraftDetection.Source != rocm.DraftSourceAssistantDir || report.DraftDetection.DraftPath != assistant {
		t.Fatalf("unexpected tune draft detection: %+v", report.DraftDetection)
	}
	if report.Policy.Mode != "mtp_attached_drafter" || report.Policy.DefaultDraftTokens != rocm.ProductionMTPDefaultDraftTokens {
		t.Fatalf("unexpected tune MTP policy: %+v", report.Policy)
	}
	if len(report.OfficialLocks) != 2 ||
		report.OfficialLocks[0].Role != rocm.OfficialGemma4E2BRoleTarget ||
		report.OfficialLocks[1].Role != rocm.OfficialGemma4E2BRoleAssistant {
		t.Fatalf("unexpected official Gemma 4 locks: %+v", report.OfficialLocks)
	}
	if len(report.Sweep) != 2 ||
		report.Sweep[0].DraftBlock != 4 ||
		report.Sweep[0].DraftTokens != 3 ||
		report.Sweep[0].MaxTokens != 16 ||
		report.Sweep[1].DraftBlock != 5 ||
		report.Sweep[1].DraftTokens != 4 {
		t.Fatalf("unexpected tune sweep: %+v", report.Sweep)
	}
	if report.Labels["tuning_stage"] != "reactive_mtp_tuning_plan" ||
		report.Labels["reactive_draft_detection"] != "active_pending_native_drafter" ||
		report.Labels["reactive_draft_fallback"] != "refused" ||
		report.Labels["native_mtp_attachment"] != "linked" ||
		report.Labels["tuning_executor"] != "native_attached_mtp" ||
		report.Labels["profile_writer"] != "enabled_after_successful_sweep" ||
		report.Labels["profile_reader"] != "generate_serve_auto_profile" ||
		report.Labels["retained_state_required"] != "true" ||
		report.Labels["prompt_replay_fallback"] != "refused" ||
		report.Labels["measurement_prompt_profile"] != productionMeasurementPromptProfile ||
		report.Labels["target_decode_baseline_required"] != "true" ||
		report.Labels["profile_requires_native_mtp_speedup"] != "true" ||
		report.Labels["tuning_measurement_shape"] != tuneMeasurementShapeRetainedWarmTurn ||
		report.Labels["tune_measured_turn"] != tuneRetainedMeasuredTurn ||
		report.Labels["tune_warmup_max_tokens"] != "16" ||
		report.Labels["sampling_profile"] != "greedy" ||
		report.Labels["sampling_temperature"] != "0" ||
		report.Labels["draft_block_sweep"] != "4,5" ||
		report.Labels["draft_token_sweep"] != "3,4" ||
		report.Labels["production_requires_env_gate"] != "false" ||
		report.Labels["production_requires_cli_flag"] != "false" {
		t.Fatalf("unexpected tune labels: %+v", report.Labels)
	}
	if !strings.Contains(report.Prompt, "retained-state Gemma-4 QAT server") {
		t.Fatalf("tune prompt = %q, want long MTP warmup prompt", report.Prompt)
	}
}

func TestTuneJSONReportsLlamaServerMTPSamplingShape(t *testing.T) {
	model := filepath.Join(t.TempDir(), "model")
	assistant := filepath.Join(model, "assistant")
	writeCLIDraftDetectModelDir(t, model, "gemma4_text")
	writeCLIDraftDetectModelDir(t, assistant, "gemma4_assistant")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"tune",
		"-model", model,
		"-depths", "5",
		"-temp", "1",
		"-top-p", "0.95",
		"-top-k", "64",
		"-json",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("tune code=%d, want 0; stderr=%s", code, stderr.String())
	}
	var report tunePlanReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("tune JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Sampling.Profile != "llama_server_mtp" ||
		report.Sampling.Temperature != 1 ||
		report.Sampling.TopP != 0.95 ||
		report.Sampling.TopK != 64 ||
		report.Sweep[0].DraftBlock != 5 ||
		report.Sweep[0].DraftTokens != 4 ||
		report.Labels["sampling_profile"] != "llama_server_mtp" ||
		report.Labels["sampling_temperature"] != "1" ||
		report.Labels["sampling_top_p"] != "0.95" ||
		report.Labels["sampling_top_k"] != "64" {
		t.Fatalf("sampling report = %+v labels=%+v sweep=%+v, want llama-server MTP shape", report.Sampling, report.Labels, report.Sweep)
	}
}

func TestProductionMeasurementPromptProfileReportsCustomPrompt(t *testing.T) {
	if productionMeasurementPromptProfileFor("") != productionMeasurementPromptProfile ||
		productionMeasurementPromptProfileFor(defaultProductionMeasurementPrompt) != productionMeasurementPromptProfile ||
		productionMeasurementPromptProfileFor("short prompt") != "custom" {
		t.Fatalf("unexpected prompt profile defaults/custom handling")
	}
}

func TestTuneDefaultDepthsIncludeShortFastBlocks(t *testing.T) {
	depths, err := parseTuneDepths("")
	if err != nil {
		t.Fatalf("parse default tune depths: %v", err)
	}
	if !slices.Equal(depths, []int{2, 3, 4, 5, 6}) {
		t.Fatalf("default tune depths = %v, want short-block fast lane sweep", depths)
	}
}

func TestTuneDefaultFailsClosedWithoutDrafter(t *testing.T) {
	model := filepath.Join(t.TempDir(), "model")
	writeCLIDraftDetectModelDir(t, model, "gemma4_text")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"tune",
		"-model", model,
	}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("tune code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("tune wrote stdout on no-drafter refusal: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "no MTP drafter found") ||
		!strings.Contains(stderr.String(), "assistant/") ||
		!strings.Contains(stderr.String(), "--draft") {
		t.Fatalf("stderr missing no-drafter guidance: %s", stderr.String())
	}
}

func TestTuneDefaultFailsClosedWhenNativeMTPCannotRun(t *testing.T) {
	model := filepath.Join(t.TempDir(), "model")
	assistant := filepath.Join(model, "assistant")
	writeCLIDraftDetectModelDir(t, model, "gemma4_text")
	writeCLIDraftDetectModelDir(t, assistant, "gemma4_assistant")

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"tune",
		"-model", model,
		"-depths", "4,5",
	}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("tune code=%d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("tune wrote stdout on failed native MTP sweep: %s", stdout.String())
	}
	for _, want := range []string{
		"measuring block 4",
		"block 4 failed",
		"no successful native MTP attached-drafter candidates",
		"refusing autoregressive fallback",
		assistant,
		"sweep 4,5",
	} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("stderr missing %q: %s", want, stderr.String())
		}
	}
}

func TestTuneDraftBlockCandidatesCarryProfileContract(t *testing.T) {
	report := tunePlanReport{
		ModelPath:  "/models/gemma",
		Workload:   "chat",
		ProfileDir: "profiles",
		Labels: map[string]string{
			"caller": "unit-test",
		},
		Sweep: []tuneSweepPlan{
			{DraftBlock: 4, DraftTokens: 3, MaxTokens: 16, Workload: "chat"},
			{DraftBlock: 5, DraftTokens: 4, MaxTokens: 16, Workload: "chat"},
		},
	}
	plan := inference.TuningPlan{
		Runtime: inference.RuntimeIdentity{Backend: "rocm", CacheMode: "k-q8-v-q4"},
		Model:   inference.ModelIdentity{Path: "/models/gemma"},
		Candidates: []inference.TuningCandidate{{
			ID:       "template",
			Workload: inference.TuningWorkloadChat,
			Model:    inference.ModelIdentity{Path: "/models/gemma"},
			Runtime:  inference.RuntimeIdentity{Backend: "rocm", CacheMode: "k-q8-v-q4"},
			Labels:   map[string]string{"template": "kept"},
		}},
	}

	candidates := tuneDraftBlockCandidatesFromPlan(plan, report)
	if len(candidates) != 2 {
		t.Fatalf("candidates = %d, want 2", len(candidates))
	}
	if candidates[0].ID != "chat:mtp-block-4" ||
		candidates[0].Labels[tuneDraftBlockLabel] != "4" ||
		candidates[0].Labels["mtp_draft_tokens"] != "3" ||
		candidates[0].Labels["tuning_executor"] != "native_attached_mtp" ||
		candidates[0].Labels["profile_writer"] != "enabled" ||
		candidates[0].Labels["profile_reader"] != "generate_serve_auto_profile" ||
		candidates[0].Labels["caller"] != "unit-test" ||
		candidates[0].Labels["template"] != "kept" {
		t.Fatalf("candidate[0] = %+v, want MTP profile contract labels", candidates[0])
	}
	if got := tuneDraftBlockFromCandidate(candidates[1]); got != 5 {
		t.Fatalf("draft block from candidate[1] = %d, want 5", got)
	}
}

func TestTuneTargetDecodeBaselineCandidateCarriesComparisonContract(t *testing.T) {
	report := tunePlanReport{
		ModelPath: "/models/gemma",
		Workload:  "chat",
		Labels: map[string]string{
			tuneDraftBlockLabel: "3",
			"mtp_draft_tokens":  "2",
			"caller":            "unit-test",
			"native_mtp_status": "linked",
		},
	}
	plan := inference.TuningPlan{
		Runtime: inference.RuntimeIdentity{Backend: "rocm", CacheMode: "k-q8-v-q4"},
		Model:   inference.ModelIdentity{Path: "/models/gemma", Architecture: "gemma4_text"},
		Candidates: []inference.TuningCandidate{{
			ID:       "template",
			Workload: inference.TuningWorkloadChat,
			Model:    inference.ModelIdentity{Path: filepath.Join("/models/gemma", "model.safetensors"), Architecture: "gemma4_text"},
			Runtime:  inference.RuntimeIdentity{Backend: "rocm", CacheMode: "k-q8-v-q4"},
			Labels: map[string]string{
				tuneDraftBlockLabel: "4",
				"template":          "kept",
			},
		}},
	}

	candidate := tuneTargetDecodeBaselineCandidate(plan, report)
	if candidate.ID != "chat:target-decode-baseline" ||
		candidate.Model.Path != report.ModelPath ||
		candidate.Labels[tuneDraftBlockLabel] != "" ||
		candidate.Labels["mtp_draft_tokens"] != "" ||
		candidate.Labels["tuning_executor"] != "target_decode_baseline" ||
		candidate.Labels["target_decode_baseline"] != "true" ||
		candidate.Labels["native_mtp_attachment"] != "not_used" ||
		candidate.Labels["profile_writer"] != "comparison_only" ||
		candidate.Labels["template"] != "kept" {
		t.Fatalf("baseline candidate = %+v, want target decode comparison-only candidate", candidate)
	}
}

func TestTuneDraftBlockCandidatesPinProfileLookupModelPath(t *testing.T) {
	report := tunePlanReport{
		ModelPath: "/models/gemma",
		Workload:  "chat",
		Sweep: []tuneSweepPlan{
			{DraftBlock: 3, DraftTokens: 2, MaxTokens: 16, Workload: "chat"},
		},
	}
	plan := inference.TuningPlan{
		Runtime: inference.RuntimeIdentity{Backend: "rocm", CacheMode: "k-q8-v-q4"},
		Model: inference.ModelIdentity{
			Path:          report.ModelPath,
			Architecture:  "gemma4_text",
			ContextLength: 131072,
			Labels:        map[string]string{"plan_model": "kept"},
		},
		Candidates: []inference.TuningCandidate{{
			ID:       "template",
			Workload: inference.TuningWorkloadChat,
			Model: inference.ModelIdentity{
				Path:          filepath.Join(report.ModelPath, "model.safetensors"),
				Architecture:  "gemma4_text",
				QuantBits:     6,
				QuantGroup:    64,
				QuantType:     "q6",
				ContextLength: 131072,
				Labels:        map[string]string{"candidate_model": "kept"},
			},
			Runtime: inference.RuntimeIdentity{Backend: "rocm", CacheMode: "k-q8-v-q4"},
		}},
	}

	candidates := tuneDraftBlockCandidatesFromPlan(plan, report)
	if len(candidates) != 1 {
		t.Fatalf("candidates = %d, want 1", len(candidates))
	}
	candidate := candidates[0]
	if candidate.Model.Path != report.ModelPath ||
		candidate.Model.QuantBits != 6 ||
		candidate.Model.ContextLength != 131072 ||
		candidate.Model.Labels["candidate_model"] != "kept" ||
		candidate.Model.Labels["plan_model"] != "kept" ||
		candidate.Model.Labels["inspected_weight_path"] != filepath.Join(report.ModelPath, "model.safetensors") ||
		candidate.Model.Labels["profile_lookup_path"] != report.ModelPath {
		t.Fatalf("candidate model = %+v, want model-pack path pinned for profile lookup", candidate.Model)
	}

	profileDir := t.TempDir()
	profile := rocm.BuildTuningProfile(plan, report.ModelPath, "machine-a", inference.TuningWorkloadChat, inference.TuningResult{
		Candidate: candidate,
		Measurements: inference.TuningMeasurements{
			DecodeTokensPerSec: 42,
		},
		Score: inference.TuningScore{Score: 42},
	}, map[string]string{"source": "unit-test"}, time.Unix(1234, 0))
	if profile.Key.Model.Path != report.ModelPath || profile.Candidate.Model.Path != report.ModelPath {
		t.Fatalf("profile model paths key=%q candidate=%q, want %q", profile.Key.Model.Path, profile.Candidate.Model.Path, report.ModelPath)
	}
	path := rocm.TuningProfilePath(profileDir, profile)
	if err := writeROCmTuningProfile(path, profile); err != nil {
		t.Fatal(err)
	}
	block, loadedPath := loadTunedDraftBlock(profileDir, report.ModelPath, "machine-a")
	if block != 3 || loadedPath != path {
		t.Fatalf("loadTunedDraftBlock generated profile = %d/%q, want block 3 from %q", block, loadedPath, path)
	}
}

func TestTuneProfileLabelsPromoteMeasuredNativeSweep(t *testing.T) {
	report := tunePlanReport{
		Labels: map[string]string{
			"tuning_stage": "reactive_mtp_tuning_plan",
			"caller":       "unit-test",
		},
	}
	selected := inference.TuningResult{
		Candidate: inference.TuningCandidate{
			ID:     "chat:mtp-block-5",
			Labels: map[string]string{tuneDraftBlockLabel: "5"},
		},
		Labels: map[string]string{
			"mtp_acceptance_rate": "0.440000",
		},
	}
	labels := tuneProfileLabelsFromReport(report, selected)
	if labels["tuning_stage"] != "reactive_mtp_tuning_profile" ||
		labels["native_mtp_attachment"] != "linked" ||
		labels["tuning_executor"] != "native_attached_mtp" ||
		labels["profile_writer"] != "enabled" ||
		labels["profile_reader"] != "generate_serve_auto_profile" ||
		labels["profile_contract"] != "inference.TuningProfile" ||
		labels["profile_draft_block_label"] != tuneDraftBlockLabel ||
		labels[tuneDraftBlockLabel] != "5" ||
		labels["mtp_acceptance_rate"] != "0.440000" ||
		labels["caller"] != "unit-test" {
		t.Fatalf("labels = %+v, want measured native sweep profile labels", labels)
	}
}

func TestTuneNativeMTPBaselineComparisonRequiresSpeedup(t *testing.T) {
	selected := inference.TuningResult{
		Candidate:    inference.TuningCandidate{ID: "chat:mtp-block-3"},
		Measurements: inference.TuningMeasurements{DecodeTokensPerSec: 96},
	}
	baseline := inference.TuningResult{
		Candidate:    inference.TuningCandidate{ID: "chat:target-decode-baseline"},
		Measurements: inference.TuningMeasurements{DecodeTokensPerSec: 100},
	}
	comparison := tuneNativeMTPBaselineComparison(selected, baseline)
	if comparison.Error != "" ||
		comparison.MTPFaster ||
		comparison.Speedup != 0.96 ||
		comparison.Labels["native_mtp_faster_than_target_decode"] != "false" ||
		comparison.Labels["native_mtp_speedup"] != "0.960000" ||
		comparison.Labels["target_decode_baseline_tok_s"] != "100.000000" {
		t.Fatalf("comparison = %+v, want slower native MTP rejected with labels", comparison)
	}
	selected.Measurements.DecodeTokensPerSec = 112
	comparison = tuneNativeMTPBaselineComparison(selected, baseline)
	if comparison.Error != "" ||
		!comparison.MTPFaster ||
		comparison.Speedup != 1.12 ||
		comparison.Labels["native_mtp_faster_than_target_decode"] != "true" ||
		comparison.Labels["native_mtp_speedup"] != "1.120000" ||
		comparison.Labels["profile_requires_native_mtp_speedup"] != "true" {
		t.Fatalf("comparison = %+v, want faster native MTP accepted with labels", comparison)
	}
}

func TestTuneDraftBlockGenerateUsesRetainedWarmTurn(t *testing.T) {
	model := &tuneStatelessCLITestModel{traceCLITestModel: &traceCLITestModel{}}
	generated := runTuneDraftBlockGenerate(context.Background(), model, tunePlanReport{
		Prompt:    "Measure attached MTP.",
		MaxTokens: 8,
		Sampling: tuneSamplingPlan{
			Temperature: 1,
			TopP:        0.95,
			TopK:        64,
			Profile:     "llama_server_mtp",
		},
	})
	if generated.GeneratedTokens != 2 ||
		generated.WarmupTokens != 2 ||
		generated.Shape != tuneMeasurementShapeRetainedWarmTurn ||
		generated.MeasuredTurn != tuneRetainedMeasuredTurn ||
		generated.WarmupMaxTokens != 8 ||
		model.statelessCalls != 0 ||
		model.chatCalls != 2 {
		t.Fatalf("tune generate result=%+v stateless=%d chat=%d, want retained warm-turn measurement", generated, model.statelessCalls, model.chatCalls)
	}
	if len(model.calls) != 2 ||
		model.calls[0].MaxTokens != 8 ||
		model.calls[1].MaxTokens != 8 ||
		model.calls[0].Temperature != 1 ||
		model.calls[1].Temperature != 1 ||
		model.calls[0].TopP != 0.95 ||
		model.calls[1].TopP != 0.95 ||
		model.calls[0].TopK != 64 ||
		model.calls[1].TopK != 64 {
		t.Fatalf("generate opts = %+v, want warmup + sampled measured turn", model.calls)
	}
	if len(model.messages) != 2 ||
		model.messages[0][0].Content != "Measure attached MTP." ||
		!strings.Contains(model.messages[1][0].Content, "retained-state token loop") {
		t.Fatalf("generate messages = %+v, want production warmup then retained continuation", model.messages)
	}
}

func TestTuneRejectsInvalidDepths(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"tune", "-model", "/tmp/model", "-depths", "4x", "-json"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("tune code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("tune wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), `invalid draft block "4x"`) {
		t.Fatalf("stderr missing depth validation: %s", stderr.String())
	}
}

func TestTuneRejectsOutOfRangeDepths(t *testing.T) {
	for _, depths := range []string{"1", "9"} {
		var stdout, stderr bytes.Buffer
		code := runCommand(context.Background(), []string{"tune", "-model", "/tmp/model", "-depths", depths, "-json"}, &stdout, &stderr)
		if code != 2 {
			t.Fatalf("tune -depths %s code=%d, want 2", depths, code)
		}
		if stdout.Len() != 0 {
			t.Fatalf("tune -depths %s wrote stdout on validation failure: %s", depths, stdout.String())
		}
		if !strings.Contains(stderr.String(), "out of range 2..8") {
			t.Fatalf("stderr missing depth range validation for %s: %s", depths, stderr.String())
		}
	}
}

func TestTuneRejectsMissingModel(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"tune", "-json"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("tune code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("tune wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "model path is required") {
		t.Fatalf("stderr missing model validation: %s", stderr.String())
	}
}

func TestVisionJSONReportsNativeRuntimeBoundary(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model")
	image := filepath.Join(dir, "photo.png")
	frame := filepath.Join(dir, "frame.jpg")
	writeCLIVisionModelDir(t, model)
	if err := os.WriteFile(image, []byte("png"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(frame, []byte("jpg"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"vision",
		"-images", image,
		"-video-frames", frame,
		"-fps", "2",
		"-prompt", "What changed?",
		"-max-tokens", "12",
		"-json",
		model,
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("vision code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("vision JSON wrote stderr: %s", stderr.String())
	}
	var report visionPlanReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("vision JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Kind != "native-vision-runtime-report" ||
		report.Backend != defaultBackendName ||
		report.Command != "vision" ||
		report.CLIContract != cliContractName ||
		!report.NoPython {
		t.Fatalf("unexpected vision identity: %+v", report)
	}
	if report.ModelPath != model || report.Prompt != "What changed?" || report.MaxTokens != 12 || !report.ChatTemplate {
		t.Fatalf("unexpected vision request: %+v", report)
	}
	if len(report.Images) != 1 || report.Images[0].Path != image || report.Images[0].Kind != "image" ||
		len(report.VideoFrames) != 1 || report.VideoFrames[0].Path != frame || report.VideoFrames[0].Kind != "video_frame" {
		t.Fatalf("unexpected vision assets: images=%+v frames=%+v", report.Images, report.VideoFrames)
	}
	if report.Model.Architecture != "gemma4" ||
		!report.Native.Multimodal ||
		report.Native.Runtime != "not_linked" ||
		report.Native.Projector != "not_linked" ||
		report.Native.Reference != "go_mlx_gemma4_vision" ||
		report.Native.Ready ||
		report.Native.Fallback != "refused" ||
		report.Native.ExecutionStatus != "not_linked" {
		t.Fatalf("unexpected vision native report: model=%+v native=%+v labels=%+v", report.Model, report.Native, report.Labels)
	}
	if report.SoftTokensPerImage != 280 || report.EstimatedSoftTokens != 560 {
		t.Fatalf("soft token estimate = %d/%d, want 280/560", report.SoftTokensPerImage, report.EstimatedSoftTokens)
	}
	if report.Labels["vision_stage"] != "native_vision_runtime_report" ||
		report.Labels["reactive_vision_fallback"] != "refused" ||
		report.Labels["production_requires_env_gate"] != "false" ||
		report.Labels["production_requires_cli_flag"] != "false" ||
		report.Labels["image_count"] != "1" ||
		report.Labels["video_frame_count"] != "1" {
		t.Fatalf("unexpected vision labels: %+v", report.Labels)
	}
}

func TestVisionRejectsMissingVisualInput(t *testing.T) {
	model := filepath.Join(t.TempDir(), "model")
	writeCLIVisionModelDir(t, model)

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"vision", "-json", model}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("vision code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("vision wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "at least one image or video frame is required") {
		t.Fatalf("stderr missing visual input validation: %s", stderr.String())
	}
}

func TestAudioJSONReportsNativeRuntimeBoundary(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model")
	wav := filepath.Join(dir, "speech.wav")
	writeCLIAudioModelDir(t, model)
	writeCLITestWAVPCM16(t, wav, 16000, 1, []int16{0, 100, -100, 0})

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"audio",
		"-audio", wav,
		"-prompt", "Transcribe this.",
		"-max-tokens", "12",
		"-json",
		model,
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("audio code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("audio JSON wrote stderr: %s", stderr.String())
	}
	var report audioPlanReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("audio JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Kind != "native-audio-runtime-report" ||
		report.Backend != defaultBackendName ||
		report.Command != "audio" ||
		report.CLIContract != cliContractName ||
		!report.NoPython {
		t.Fatalf("unexpected audio identity: %+v", report)
	}
	if report.ModelPath != model || report.Prompt != "Transcribe this." || report.MaxTokens != 12 || !report.ChatTemplate {
		t.Fatalf("unexpected audio request: %+v", report)
	}
	if report.Audio.Path != wav ||
		report.Audio.SampleRate != 16000 ||
		report.Audio.Channels != 1 ||
		report.Audio.BitsPerSample != 16 ||
		report.Audio.Frames != 4 {
		t.Fatalf("unexpected audio asset: %+v", report.Audio)
	}
	if report.Model.Architecture != "gemma4_unified" ||
		!report.Native.Multimodal ||
		report.Native.Runtime != "not_linked" ||
		report.Native.Projector != "not_linked" ||
		report.Native.FrontEnd != "not_linked" ||
		report.Native.Reference != "go_mlx_gemma4_audio" ||
		report.Native.Ready ||
		report.Native.Fallback != "refused" ||
		report.Native.ExecutionStatus != "not_linked" {
		t.Fatalf("unexpected audio native report: model=%+v native=%+v labels=%+v", report.Model, report.Native, report.Labels)
	}
	if report.Labels["audio_stage"] != "native_audio_runtime_report" ||
		report.Labels["reactive_audio_fallback"] != "refused" ||
		report.Labels["audio_token_id"] != "258881" ||
		report.Labels["audio_samples_per_token"] != "640" ||
		report.Labels["production_requires_env_gate"] != "false" ||
		report.Labels["production_requires_cli_flag"] != "false" {
		t.Fatalf("unexpected audio labels: %+v", report.Labels)
	}
}

func TestAudioRejectsMissingAudioInput(t *testing.T) {
	model := filepath.Join(t.TempDir(), "model")
	writeCLIAudioModelDir(t, model)

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"audio", "-json", model}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("audio code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("audio wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "audio path is required") {
		t.Fatalf("stderr missing audio path validation: %s", stderr.String())
	}
}

func TestDiffuseJSONReportsNativeRuntimeBoundary(t *testing.T) {
	model := filepath.Join(t.TempDir(), "model")
	writeCLIDiffusionModelDir(t, model)

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{
		"diffuse",
		"-prompt", "Explain entropy briefly.",
		"-max-canvases", "2",
		"-steps", "3",
		"-canvas", "16",
		"-entropy", "0.25",
		"-seed", "42",
		"-trace",
		"-json",
		model,
	}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("diffuse code=%d, want 1 until native sampler is linked; stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("diffuse JSON wrote stderr: %s", stderr.String())
	}
	var report diffusePlanReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("diffuse JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Kind != "native-diffusion-runtime-report" ||
		report.Backend != defaultBackendName ||
		report.Command != "diffuse" ||
		report.CLIContract != cliContractName ||
		!report.NoPython {
		t.Fatalf("unexpected diffuse identity: %+v", report)
	}
	if report.ModelPath != model ||
		report.Prompt != "Explain entropy briefly." ||
		report.MaxCanvases != 2 ||
		report.Steps != 3 ||
		report.Canvas != 16 ||
		report.Entropy != 0.25 ||
		report.Seed != 42 ||
		!report.ChatTemplate ||
		!report.Trace {
		t.Fatalf("unexpected diffuse request: %+v", report)
	}
	if report.Model.Architecture != "diffusion_gemma" ||
		!report.Native.BlockDiffusion ||
		report.Native.Runtime != "not_linked" ||
		report.Native.Sampler != "not_linked" ||
		report.Native.Trunk != "model_pack_metadata" ||
		report.Native.Reference != "go_mlx_diffusion_gemma" ||
		report.Native.ModelCanvasLength != 256 ||
		report.Native.Ready ||
		report.Native.Fallback != "refused" ||
		report.Native.ExecutionStatus != "not_linked" {
		t.Fatalf("unexpected diffuse native report: model=%+v native=%+v labels=%+v", report.Model, report.Native, report.Labels)
	}
	if report.Labels["diffusion_stage"] != "native_diffusion_runtime_report" ||
		report.Labels["reactive_diffusion_fallback"] != "refused" ||
		report.Labels["block_diffusion_model"] != "true" ||
		report.Labels["diffusion_canvas"] != "16" ||
		report.Labels["diffusion_canvas_length"] != "256" ||
		report.Labels["diffusion_seed"] != "42" ||
		report.Labels["engine_diffusion_sampler_route_contract"] != rocm.ROCmDiffusionSamplerRegistryContract ||
		report.Labels["engine_diffusion_sampler_canvas_length"] != "256" ||
		report.Labels["engine_diffusion_sampler_default_canvas_length"] != "64" ||
		report.Labels["engine_diffusion_sampler_default_max_steps"] != "16" ||
		report.Labels["engine_diffusion_sampler_fallback_refused"] != "true" ||
		report.Labels["production_requires_env_gate"] != "false" ||
		report.Labels["production_requires_cli_flag"] != "false" {
		t.Fatalf("unexpected diffuse labels: %+v", report.Labels)
	}
}

func TestDiffuseRejectsMissingModelPath(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"diffuse", "-json"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("diffuse code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("diffuse wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "expected exactly one model path") {
		t.Fatalf("stderr missing model validation: %s", stderr.String())
	}
}

func TestStatePackJSONBuildsKVSTContainer(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "session.mvlog")
	markerPath := filepath.Join(dir, "ramp-report.json")
	outputPath := filepath.Join(dir, "session.kv")
	payload := []byte("go-rocm-state-log\nbinary\x00tail")
	if err := os.WriteFile(statePath, payload, 0o600); err != nil {
		t.Fatalf("WriteFile(state) error = %v", err)
	}
	if err := os.WriteFile(markerPath, []byte(`{
  "fold": {
    "compact_marker": {
      "store_path": "`+statePath+`",
      "index_uri": "rocm://state-ramp/fold/1/folded/index",
      "entry_uri": "rocm://state-ramp/fold/1/folded",
      "bundle_uri": "rocm://state-ramp/fold/1/folded/bundle",
      "token_count": 206
    }
  }
}`), 0o600); err != nil {
		t.Fatalf("WriteFile(marker) error = %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"state-pack", "-json", "-marker-file", markerPath, "-output", outputPath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("state-pack code=%d, want 0; stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("state-pack JSON wrote stderr: %s", stderr.String())
	}
	var report statePackReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("state-pack JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Backend != "rocm" || report.Command != "state-pack" || report.CLIContract != cliContractName {
		t.Fatalf("unexpected state-pack report identity: %+v", report)
	}
	if report.Magic != stateKVContainerMagic || report.TrixVersion != stateKVContainerVersion {
		t.Fatalf("unexpected state-pack container identity: %+v", report)
	}
	if report.MarkerFile != markerPath || report.StateStorePath != statePath || report.OutputPath != outputPath {
		t.Fatalf("unexpected state-pack paths: %+v", report)
	}
	if report.PayloadBytes != int64(len(payload)) || report.ContainerBytes <= report.PayloadBytes {
		t.Fatalf("unexpected state-pack byte counts: %+v", report)
	}
	if report.Marker.IndexURI != "rocm://state-ramp/fold/1/folded/index" || report.Marker.TokenCount != 206 {
		t.Fatalf("unexpected compact marker: %+v", report.Marker)
	}
	decodedPayload, header, err := stateKVContainerPayload(outputPath)
	if err != nil {
		t.Fatalf("stateKVContainerPayload() error = %v", err)
	}
	if string(decodedPayload) != string(payload) {
		t.Fatalf("decoded payload = %q, want original payload", string(decodedPayload))
	}
	if header["kind"] != stateKVContainerKind || header["content_type"] != stateKVContainerContentType {
		t.Fatalf("header = %#v, want ROCm state KV metadata", header)
	}
	marker, err := stateKVContainerMarkerFromHeader(header, int64(len(payload)))
	if err != nil {
		t.Fatalf("stateKVContainerMarkerFromHeader() error = %v", err)
	}
	if marker.IndexURI != report.Marker.IndexURI || marker.TokenCount != report.Marker.TokenCount {
		t.Fatalf("decoded marker = %+v, want %+v", marker, report.Marker)
	}
}

func TestStatePackLogAliasBuildsContainer(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "session.mvlog")
	markerPath := filepath.Join(dir, "marker.json")
	outputPath := filepath.Join(dir, "session.kv")
	if err := os.WriteFile(statePath, []byte("state payload"), 0o600); err != nil {
		t.Fatalf("WriteFile(state) error = %v", err)
	}
	if err := os.WriteFile(markerPath, []byte(`{"index_uri":"rocm://state/index"}`), 0o600); err != nil {
		t.Fatalf("WriteFile(marker) error = %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"state-pack", "-json", "-marker-file", markerPath, "-output", outputPath, "-log", statePath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("state-pack code=%d, want 0; stderr=%s", code, stderr.String())
	}
	var report statePackReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("state-pack JSON did not unmarshal: %v\n%s", err, stdout.String())
	}
	if report.StateStorePath != statePath || report.PayloadBytes != int64(len("state payload")) {
		t.Fatalf("report = %+v, want -log alias payload", report)
	}
}

func TestStatePackValidationBad(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCommand(context.Background(), []string{"state-pack", "-output", "state.kv"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("state-pack code=%d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("state-pack wrote stdout on validation failure: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "marker file is required") {
		t.Fatalf("stderr missing marker validation: %s", stderr.String())
	}
}
