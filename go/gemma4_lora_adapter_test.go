// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestGemma4LoRAAdapter_Good_LoadCarriesBaseModelLabels(t *testing.T) {
	native := &fakeNativeModel{
		loadAdapterIdentity: inference.AdapterIdentity{
			Path:       "domain.safetensors",
			Format:     "lora",
			TargetKeys: []string{"model.layers.0.self_attn.q_proj"},
			Labels:     map[string]string{"adapter_runtime": "hip_loaded"},
		},
	}
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    26,
			HiddenSize:   2304,
			VocabSize:    262144,
		},
		native: native,
	}

	loaded, err := model.LoadAdapter("domain.safetensors")
	core.RequireNoError(t, err)
	assertGemma4LoRABaseLabels(t, loaded, "gemma4_text", "E4B", "q6", Gemma4RuntimeMLXAffine, Gemma4GenerateLinked, "true", "true")
	core.AssertEqual(t, "hip_loaded", loaded.Labels["adapter_runtime"])
	core.AssertEqual(t, "gemma4_mlx_affine", loaded.Labels["production_quant_policy"])
	core.AssertEqual(t, "lmstudio-community/gemma-4-E4B-it-MLX-6bit", loaded.Labels["adapter_base_production_quant_model"])
	core.AssertEqual(t, "", loaded.Labels["adapter_base_production_quant_locked_model"])

	loaded.Labels["adapter_base_gemma4_size"] = "mutated"
	active := model.ActiveAdapter()
	assertGemma4LoRABaseLabels(t, active, "gemma4_text", "E4B", "q6", Gemma4RuntimeMLXAffine, Gemma4GenerateLinked, "true", "true")
	core.AssertEqual(t, "hip_loaded", active.Labels["adapter_runtime"])
	core.AssertEqual(t, "lmstudio-community/gemma-4-E4B-it-MLX-6bit", active.Labels["adapter_base_production_quant_model"])
}

func TestGemma4LoRAAdapter_Good_AllProductionSizesCarryBaseModelLabels(t *testing.T) {
	tests := []struct {
		name         string
		architecture string
		path         string
		size         string
		mode         string
		runtime      string
		status       string
		runnable     string
	}{
		{name: "e2b_bf16", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-e2b-it-bf16", size: "E2B", mode: "bf16", runtime: Gemma4RuntimeBF16, status: Gemma4GenerateLoadOnly, runnable: "true"},
		{name: "e2b_q8", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-e2b-it-8bit", size: "E2B", mode: "q8", runtime: Gemma4RuntimeMLXAffine, status: Gemma4GenerateLinked, runnable: "true"},
		{name: "e2b_q6", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-e2b-it-6bit", size: "E2B", mode: "q6", runtime: Gemma4RuntimeMLXAffine, status: Gemma4GenerateLinked, runnable: "true"},
		{name: "e2b_q4", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-e2b-it-4bit", size: "E2B", mode: "q4", runtime: Gemma4RuntimeMLXAffine, status: Gemma4GenerateLinked, runnable: "true"},
		{name: "e2b_mxfp8", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-e2b-it-mxfp8", size: "E2B", mode: "mxfp8", runtime: Gemma4RuntimePlanned, status: Gemma4GeneratePlannedOnly, runnable: "true"},
		{name: "e2b_mxfp4", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-e2b-it-mxfp4", size: "E2B", mode: "mxfp4", runtime: Gemma4RuntimePlanned, status: Gemma4GeneratePlannedOnly, runnable: "true"},
		{name: "e4b_bf16", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-e4b-it-bf16", size: "E4B", mode: "bf16", runtime: Gemma4RuntimeBF16, status: Gemma4GenerateLoadOnly, runnable: "true"},
		{name: "e4b_q8", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-e4b-it-8bit", size: "E4B", mode: "q8", runtime: Gemma4RuntimeMLXAffine, status: Gemma4GenerateLinked, runnable: "true"},
		{name: "e4b_q6", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-e4b-it-6bit", size: "E4B", mode: "q6", runtime: Gemma4RuntimeMLXAffine, status: Gemma4GenerateLinked, runnable: "true"},
		{name: "e4b_q4", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-e4b-it-4bit", size: "E4B", mode: "q4", runtime: Gemma4RuntimeMLXAffine, status: Gemma4GenerateLinked, runnable: "true"},
		{name: "e4b_mxfp8", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-e4b-it-mxfp8", size: "E4B", mode: "mxfp8", runtime: Gemma4RuntimePlanned, status: Gemma4GeneratePlannedOnly, runnable: "true"},
		{name: "e4b_mxfp4", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-e4b-it-mxfp4", size: "E4B", mode: "mxfp4", runtime: Gemma4RuntimePlanned, status: Gemma4GeneratePlannedOnly, runnable: "true"},
		{name: "12b_q6", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-12b-it-6bit", size: "12B", mode: "q6", runtime: Gemma4RuntimeMLXAffine, status: Gemma4GenerateLinked, runnable: "true"},
		{name: "26b_a4b_q8_status", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-26b-a4b-it-8bit", size: "26B-A4B", mode: "q8-status", runtime: Gemma4RuntimePlanned, status: Gemma4GeneratePlannedOnly, runnable: "false"},
		{name: "26b_a4b_q6_status", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-26b-a4b-it-6bit", size: "26B-A4B", mode: "q6-status", runtime: Gemma4RuntimePlanned, status: Gemma4GeneratePlannedOnly, runnable: "false"},
		{name: "26b_a4b_q4_status", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-26b-a4b-it-4bit", size: "26B-A4B", mode: "q4-status", runtime: Gemma4RuntimePlanned, status: Gemma4GeneratePlannedOnly, runnable: "false"},
		{name: "31b_q8_status", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-31b-it-8bit", size: "31B", mode: "q8-status", runtime: Gemma4RuntimePlanned, status: Gemma4GeneratePlannedOnly, runnable: "false"},
		{name: "31b_q6_status", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-31b-it-6bit", size: "31B", mode: "q6-status", runtime: Gemma4RuntimePlanned, status: Gemma4GeneratePlannedOnly, runnable: "false"},
		{name: "31b_q4_status", architecture: "gemma4_text", path: "/models/lmstudio-community-gemma-4-31b-it-4bit", size: "31B", mode: "q4-status", runtime: Gemma4RuntimePlanned, status: Gemma4GeneratePlannedOnly, runnable: "false"},
		{name: "mtp_assistant_e2b_bf16", architecture: "gemma4_assistant", path: "google/gemma-4-E2B-it-assistant", size: "E2B", mode: "bf16", runtime: Gemma4RuntimeBF16, status: Gemma4GenerateLoadOnly, runnable: "true"},
		{name: "mtp_assistant_e4b_bf16", architecture: "gemma4_assistant", path: "google/gemma-4-E4B-it-assistant", size: "E4B", mode: "bf16", runtime: Gemma4RuntimeBF16, status: Gemma4GenerateLoadOnly, runnable: "true"},
		{name: "mtp_assistant_12b_bf16", architecture: "gemma4_assistant", path: "google/gemma-4-12B-it-assistant", size: "12B", mode: "bf16", runtime: Gemma4RuntimeBF16, status: Gemma4GenerateLoadOnly, runnable: "true"},
		{name: "mtp_assistant_26b_a4b_bf16", architecture: "gemma4_assistant", path: "google/gemma-4-26B-A4B-it-assistant", size: "26B-A4B", mode: "bf16", runtime: Gemma4RuntimeBF16, status: Gemma4GenerateLoadOnly, runnable: "true"},
		{name: "mtp_assistant_31b_bf16", architecture: "gemma4_assistant", path: "google/gemma-4-31B-it-assistant", size: "31B", mode: "bf16", runtime: Gemma4RuntimeBF16, status: Gemma4GenerateLoadOnly, runnable: "true"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			model := &rocmModel{
				modelPath: tc.path,
				modelType: tc.architecture,
				modelInfo: inference.ModelInfo{
					Architecture: tc.architecture,
					VocabSize:    262144,
				},
				native: &fakeNativeModel{},
			}

			loaded, err := model.LoadAdapter("domain.safetensors")
			core.RequireNoError(t, err)
			assertGemma4LoRABaseLabels(t, loaded, tc.architecture, tc.size, tc.mode, tc.runtime, tc.status, "true", tc.runnable)
			core.AssertEqual(t, "gemma4_mlx_affine", loaded.Labels["production_quant_policy"])
		})
	}
}

func TestGemma4LoRAAdapter_Good_GGUFBaseCarriesLoadOnlyLabels(t *testing.T) {
	path := "/models/lmstudio-community/gemma-4-e2b-it-q4.gguf"
	labels := rocmGGUFNativeLoadLabels(inference.ModelInfo{
		Architecture: "gemma4_text",
		NumLayers:    productionLaneGemma4E2BLayers,
		HiddenSize:   productionLaneGemma4E2BHiddenSize,
		VocabSize:    262144,
		QuantBits:    4,
		QuantGroup:   64,
	}, path)
	model := inference.ModelIdentity{
		Architecture: "gemma4_text",
		Path:         path,
		NumLayers:    productionLaneGemma4E2BLayers,
		HiddenSize:   productionLaneGemma4E2BHiddenSize,
		VocabSize:    262144,
		QuantBits:    4,
		QuantGroup:   64,
		Labels:       labels,
	}

	identity := rocmAdapterIdentityForModel(inference.AdapterIdentity{Path: "domain.safetensors", Format: "lora"}, model)

	assertGemma4LoRABaseLabels(t, identity, "gemma4_text", "E2B", "q4", Gemma4RuntimeGGUF, Gemma4GenerateLoadOnly, "true", "true")
	core.AssertEqual(t, "gguf", identity.Labels["gemma4_source_format"])
	core.AssertEqual(t, Gemma4RuntimeGGUF, identity.Labels["production_quant_runtime"])
	core.AssertEqual(t, Gemma4GenerateLoadOnly, identity.Labels["production_quant_generate_status"])
}

func TestGemma4LoRAAdapter_Good_MTPAssistantBaseCarriesAssistantProductionLabels(t *testing.T) {
	native := &fakeNativeModel{
		loadAdapterIdentity: inference.AdapterIdentity{
			Path:   "assistant-domain.safetensors",
			Format: "lora",
		},
	}
	model := &rocmModel{
		modelPath: "google/gemma-4-E4B-it-assistant",
		modelType: officialGemma4E2BAssistantArchitecture,
		modelInfo: inference.ModelInfo{
			Architecture: officialGemma4E2BAssistantArchitecture,
			NumLayers:    4,
			HiddenSize:   2304,
			VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
			QuantBits:    16,
		},
		native: native,
	}

	loaded, err := model.LoadAdapter("assistant-domain.safetensors")

	core.RequireNoError(t, err)
	assertGemma4LoRABaseLabels(t, loaded, officialGemma4E2BAssistantArchitecture, "E4B", "bf16", Gemma4RuntimeBF16, Gemma4GenerateLoadOnly, "true", "true")
	core.AssertEqual(t, "google/gemma-4-E4B-it-assistant", loaded.Labels["adapter_base_production_quant_model"])
	core.AssertEqual(t, "google/gemma-4-E4B-it-assistant", loaded.Labels["adapter_base_production_quant_assistant_model"])
	core.AssertEqual(t, "mtp-assistant", loaded.Labels["adapter_base_production_quant_tier"])
	core.AssertEqual(t, "true", loaded.Labels["adapter_base_production_quant_mtp_assistant"])
	core.AssertEqual(t, "gemma4", loaded.Labels["adapter_base_production_quant_target_family"])

	report := model.Capabilities()
	core.AssertEqual(t, "google/gemma-4-E4B-it-assistant", report.Labels["adapter_base_production_quant_assistant_model"])
	core.AssertEqual(t, "mtp-assistant", report.Labels["adapter_base_production_quant_tier"])
	core.AssertEqual(t, "true", report.Labels["adapter_base_production_quant_mtp_assistant"])
}

func TestGemma4LoRAAdapter_Good_ActiveNativeFallbackCarriesStatusOnlyLabels(t *testing.T) {
	native := &fakeNativeModel{
		adapter: inference.AdapterIdentity{
			Path:       "assistant.safetensors",
			Format:     "lora",
			TargetKeys: []string{"model.layers.0.mlp.gate_proj"},
			Labels:     map[string]string{"adapter_runtime": "native_fallback"},
		},
	}
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-31b-it-6bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    62,
			HiddenSize:   4096,
			VocabSize:    262144,
		},
		native: native,
	}

	active := model.ActiveAdapter()
	assertGemma4LoRABaseLabels(t, active, "gemma4_text", "31B", "q6-status", Gemma4RuntimePlanned, Gemma4GeneratePlannedOnly, "true", "false")
	core.AssertEqual(t, "native_fallback", active.Labels["adapter_runtime"])

	active.TargetKeys[0] = "mutated"
	active.Labels["adapter_runtime"] = "mutated"
	core.AssertEqual(t, "model.layers.0.mlp.gate_proj", native.adapter.TargetKeys[0])
	core.AssertEqual(t, "native_fallback", native.adapter.Labels["adapter_runtime"])
}

func TestGemma4LoRAAdapter_Good_StateBundlePreservesAdapterBaseLabels(t *testing.T) {
	native := &fakeNativeModel{
		loadAdapterIdentity: inference.AdapterIdentity{
			Path:   "domain.safetensors",
			Format: "lora",
			Hash:   "adapter-hash",
			Labels: map[string]string{"adapter_runtime": "hip_loaded"},
		},
	}
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e2b-it-8bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    35,
			HiddenSize:   1536,
			VocabSize:    262144,
		},
		native: native,
	}
	_, err := model.LoadAdapter("domain.safetensors")
	core.RequireNoError(t, err)

	bundle, err := model.CaptureState(context.Background(), "tokens:1,2,3")
	core.RequireNoError(t, err)
	assertGemma4LoRABaseLabels(t, bundle.Adapter, "gemma4_text", "E2B", "q8", Gemma4RuntimeMLXAffine, Gemma4GenerateLinked, "true", "true")
	core.AssertEqual(t, ProductionLaneCurrentQualityModelID, bundle.Adapter.Labels["adapter_base_production_quant_model"])
	core.AssertEqual(t, "mlx-community/gemma-4-e2b-it-8bit", bundle.Adapter.Labels["adapter_base_production_quant_locked_model"])
	core.AssertEqual(t, "metadata_only", bundle.Labels["state_adapter"])
	core.AssertEqual(t, "E2B", bundle.Labels["adapter_base_gemma4_size"])
	core.AssertEqual(t, "q8", bundle.Labels["adapter_base_gemma4_quant_mode"])
	core.AssertEqual(t, "64", bundle.Labels["adapter_base_gemma4_quant_group"])
	core.AssertEqual(t, ProductionLaneCurrentQualityModelID, bundle.Labels["adapter_base_production_quant_model"])
	core.AssertEqual(t, "mlx-community/gemma-4-e2b-it-8bit", bundle.Labels["adapter_base_production_quant_locked_model"])

	restored := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e2b-it-8bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    35,
			HiddenSize:   1536,
			VocabSize:    262144,
		},
	}
	err = restored.RestoreState(context.Background(), bundle)
	core.RequireNoError(t, err)
	core.AssertEqual(t, "metadata_only", restored.state.labels["state_adapter"])
	core.AssertEqual(t, "E2B", restored.state.labels["adapter_base_gemma4_size"])
	core.AssertEqual(t, "q8", restored.state.labels["adapter_base_gemma4_quant_mode"])
	core.AssertEqual(t, "64", restored.state.labels["adapter_base_gemma4_quant_group"])
	core.AssertEqual(t, Gemma4GenerateLinked, restored.state.labels["adapter_base_gemma4_generate_status"])
	core.AssertEqual(t, ProductionLaneCurrentQualityModelID, restored.state.labels["adapter_base_production_quant_model"])
	core.AssertEqual(t, "mlx-community/gemma-4-e2b-it-8bit", restored.state.labels["adapter_base_production_quant_locked_model"])

	report := model.Capabilities()
	core.AssertEqual(t, "true", report.Labels["active_adapter"])
	core.AssertEqual(t, "E2B", report.Labels["adapter_base_gemma4_size"])
	core.AssertEqual(t, "q8", report.Labels["adapter_base_gemma4_quant_mode"])
	core.AssertEqual(t, "64", report.Labels["adapter_base_gemma4_quant_group"])
	core.AssertEqual(t, ProductionLaneCurrentQualityModelID, report.Labels["adapter_base_production_quant_model"])
	core.AssertEqual(t, "mlx-community/gemma-4-e2b-it-8bit", report.Labels["adapter_base_production_quant_locked_model"])
	core.AssertEqual(t, "E2B", report.Adapter.Labels["adapter_base_gemma4_size"])
	core.AssertEqual(t, "64", report.Adapter.Labels["adapter_base_gemma4_quant_group"])
	core.AssertEqual(t, ProductionLaneCurrentQualityModelID, report.Adapter.Labels["adapter_base_production_quant_model"])
	core.AssertEqual(t, "mlx-community/gemma-4-e2b-it-8bit", report.Adapter.Labels["adapter_base_production_quant_locked_model"])
	for _, id := range []inference.CapabilityID{
		inference.CapabilityModelLoad,
		inference.CapabilityGenerate,
		inference.CapabilityChatTemplate,
		inference.CapabilityLoRAInference,
	} {
		capability, ok := report.Capability(id)
		core.AssertTrue(t, ok)
		core.AssertEqual(t, "true", capability.Labels["active_adapter"])
		core.AssertEqual(t, "E2B", capability.Labels["adapter_base_gemma4_size"])
		core.AssertEqual(t, "q8", capability.Labels["adapter_base_gemma4_quant_mode"])
		core.AssertEqual(t, "64", capability.Labels["adapter_base_gemma4_quant_group"])
		core.AssertEqual(t, Gemma4GenerateLinked, capability.Labels["adapter_base_gemma4_generate_status"])
		core.AssertEqual(t, ProductionLaneCurrentQualityModelID, capability.Labels["adapter_base_production_quant_model"])
		core.AssertEqual(t, "mlx-community/gemma-4-e2b-it-8bit", capability.Labels["adapter_base_production_quant_locked_model"])
	}
}

func TestGemma4LoRAAdapter_Good_BenchmarkLabelsPreserveAdapterBaseLabels(t *testing.T) {
	native := &fakeNativeModel{
		tokens: []inference.Token{{ID: 1, Text: "a"}},
		adapter: inference.AdapterIdentity{
			Path:   "domain.safetensors",
			Format: "lora",
			Hash:   "adapter-hash",
		},
	}
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-12b-it-6bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    48,
			HiddenSize:   3840,
			VocabSize:    262144,
		},
		native: native,
	}

	bench, err := model.Benchmark(context.Background(), inference.BenchConfig{Prompts: []string{"tokens:1"}, MaxTokens: 1, MeasuredRuns: 1})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "12B", bench.Adapter.Labels["adapter_base_gemma4_size"])
	core.AssertEqual(t, "q6", bench.Adapter.Labels["adapter_base_gemma4_quant_mode"])
	core.AssertEqual(t, "64", bench.Adapter.Labels["adapter_base_gemma4_quant_group"])
	core.AssertEqual(t, "12B", bench.Labels["adapter_base_gemma4_size"])
	core.AssertEqual(t, "q6", bench.Labels["adapter_base_gemma4_quant_mode"])
	core.AssertEqual(t, "64", bench.Labels["adapter_base_gemma4_quant_group"])
	core.AssertEqual(t, Gemma4RuntimeMLXAffine, bench.Labels["adapter_base_gemma4_runtime"])
	core.AssertEqual(t, Gemma4GenerateLinked, bench.Labels["adapter_base_gemma4_generate_status"])
}

func TestGemma4LoRAAdapter_Good_EvaluateLabelsPreserveAdapterBaseLabels(t *testing.T) {
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e2b-it-4bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    35,
			HiddenSize:   1536,
			VocabSize:    262144,
		},
		native: &fakeNativeModel{
			adapter: inference.AdapterIdentity{
				Path:   "domain.safetensors",
				Format: "lora",
				Hash:   "adapter-hash",
			},
		},
	}

	eval, err := model.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{Text: "hello world"}}, inference.EvalConfig{MaxSamples: 1})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "E2B", eval.Adapter.Labels["adapter_base_gemma4_size"])
	core.AssertEqual(t, "q4", eval.Adapter.Labels["adapter_base_gemma4_quant_mode"])
	core.AssertEqual(t, "64", eval.Adapter.Labels["adapter_base_gemma4_quant_group"])
	core.AssertEqual(t, "E2B", eval.Labels["adapter_base_gemma4_size"])
	core.AssertEqual(t, "q4", eval.Labels["adapter_base_gemma4_quant_mode"])
	core.AssertEqual(t, "64", eval.Labels["adapter_base_gemma4_quant_group"])
	core.AssertEqual(t, Gemma4RuntimeMLXAffine, eval.Labels["adapter_base_gemma4_runtime"])
	core.AssertEqual(t, Gemma4GenerateLinked, eval.Labels["adapter_base_gemma4_generate_status"])
}

func TestGemma4LoRAAdapter_Bad_LoadRejectsMismatchedBaseLabels(t *testing.T) {
	native := &fakeNativeModel{
		loadAdapterIdentity: inference.AdapterIdentity{
			Path:   "domain.safetensors",
			Format: "lora",
			Labels: map[string]string{
				"adapter_base_architecture":      "gemma4_text",
				"adapter_base_gemma4_size":       "E2B",
				"adapter_base_gemma4_quant_mode": "q6",
			},
		},
	}
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    26,
			HiddenSize:   2304,
			VocabSize:    262144,
		},
		native: native,
	}

	identity, err := model.LoadAdapter("domain.safetensors")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "adapter base Gemma4 size mismatch")
	if !adapterIdentityIsZero(identity) {
		t.Fatalf("identity = %+v, want zero on mismatched adapter", identity)
	}
	if !adapterIdentityIsZero(model.ActiveAdapter()) {
		t.Fatalf("active adapter = %+v, want cleared after mismatched adapter", model.ActiveAdapter())
	}
	core.AssertEqual(t, 1, native.unloadCalls)
	if !adapterIdentityIsZero(native.adapter) {
		t.Fatalf("native adapter = %+v, want unloaded after mismatch", native.adapter)
	}
}

func TestGemma4LoRAAdapter_Bad_LoadRejectsMismatchedBaseSupportLabels(t *testing.T) {
	for _, tc := range []struct {
		name   string
		labels map[string]string
		want   string
	}{
		{
			name: "quant_group",
			labels: map[string]string{
				"adapter_base_architecture":            "gemma4_text",
				"adapter_base_gemma4_size":             "E4B",
				"adapter_base_gemma4_quant_mode":       "q6",
				"adapter_base_gemma4_quant_group":      "32",
				"adapter_base_gemma4_pack_supported":   "true",
				"adapter_base_gemma4_runnable_on_card": "true",
			},
			want: "adapter base Gemma4 quant group mismatch",
		},
		{
			name: "pack_supported",
			labels: map[string]string{
				"adapter_base_architecture":            "gemma4_text",
				"adapter_base_gemma4_size":             "E4B",
				"adapter_base_gemma4_quant_mode":       "q6",
				"adapter_base_gemma4_pack_supported":   "false",
				"adapter_base_gemma4_runnable_on_card": "true",
			},
			want: "adapter base Gemma4 pack support mismatch",
		},
		{
			name: "runnable_on_card",
			labels: map[string]string{
				"adapter_base_architecture":            "gemma4_text",
				"adapter_base_gemma4_size":             "E4B",
				"adapter_base_gemma4_quant_mode":       "q6",
				"adapter_base_gemma4_pack_supported":   "true",
				"adapter_base_gemma4_runnable_on_card": "false",
			},
			want: "adapter base Gemma4 runnable status mismatch",
		},
		{
			name: "production_pack",
			labels: map[string]string{
				"adapter_base_architecture":            "gemma4_text",
				"adapter_base_gemma4_size":             "E4B",
				"adapter_base_gemma4_quant_mode":       "q6",
				"adapter_base_gemma4_quant_group":      "64",
				"adapter_base_gemma4_pack_supported":   "true",
				"adapter_base_gemma4_runnable_on_card": "true",
				"adapter_base_production_quant_pack":   "E2B:q6",
			},
			want: "adapter base production quant pack mismatch",
		},
		{
			name: "assistant_role_on_text_base",
			labels: map[string]string{
				"adapter_base_architecture":                     "gemma4_text",
				"adapter_base_gemma4_size":                      "E4B",
				"adapter_base_gemma4_quant_mode":                "q6",
				"adapter_base_gemma4_quant_group":               "64",
				"adapter_base_gemma4_pack_supported":            "true",
				"adapter_base_gemma4_runnable_on_card":          "true",
				"adapter_base_production_quant_assistant_model": "google/gemma-4-E4B-it-assistant",
				"adapter_base_production_quant_mtp_assistant":   "true",
				"adapter_base_production_quant_target_family":   "gemma4",
			},
			want: "adapter base production quant assistant model mismatch",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			native := &fakeNativeModel{
				loadAdapterIdentity: inference.AdapterIdentity{
					Path:   "domain.safetensors",
					Format: "lora",
					Labels: tc.labels,
				},
			}
			model := &rocmModel{
				modelPath: "/models/lmstudio-community-gemma-4-e4b-it-6bit",
				modelType: "gemma4_text",
				modelInfo: inference.ModelInfo{
					Architecture: "gemma4_text",
					NumLayers:    26,
					HiddenSize:   2304,
					VocabSize:    262144,
				},
				native: native,
			}

			identity, err := model.LoadAdapter("domain.safetensors")

			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tc.want)
			if !adapterIdentityIsZero(identity) {
				t.Fatalf("identity = %+v, want zero on mismatched adapter", identity)
			}
			if !adapterIdentityIsZero(model.ActiveAdapter()) {
				t.Fatalf("active adapter = %+v, want cleared after mismatched adapter", model.ActiveAdapter())
			}
			core.AssertEqual(t, 1, native.unloadCalls)
		})
	}
}

func TestGemma4LoRAAdapter_Bad_GGUFBaseRejectsLinkedAdapterStatus(t *testing.T) {
	path := "/models/lmstudio-community/gemma-4-e2b-it-q4.gguf"
	labels := rocmGGUFNativeLoadLabels(inference.ModelInfo{
		Architecture: "gemma4_text",
		NumLayers:    productionLaneGemma4E2BLayers,
		HiddenSize:   productionLaneGemma4E2BHiddenSize,
		VocabSize:    262144,
		QuantBits:    4,
		QuantGroup:   64,
	}, path)
	model := inference.ModelIdentity{
		Architecture: "gemma4_text",
		Path:         path,
		NumLayers:    productionLaneGemma4E2BLayers,
		HiddenSize:   productionLaneGemma4E2BHiddenSize,
		VocabSize:    262144,
		QuantBits:    4,
		QuantGroup:   64,
		Labels:       labels,
	}
	baseLabels := map[string]string{
		"adapter_base_architecture":            "gemma4_text",
		"adapter_base_gemma4_size":             "E2B",
		"adapter_base_gemma4_quant_mode":       "q4",
		"adapter_base_gemma4_quant_group":      "64",
		"adapter_base_gemma4_pack_supported":   "true",
		"adapter_base_gemma4_runnable_on_card": "true",
	}
	tests := []struct {
		name   string
		labels map[string]string
		want   string
	}{
		{
			name: "runtime",
			labels: cloneGemma4LoRATestLabels(baseLabels, map[string]string{
				"adapter_base_gemma4_runtime":         Gemma4RuntimeMLXAffine,
				"adapter_base_gemma4_generate_status": Gemma4GenerateLoadOnly,
			}),
			want: "adapter base Gemma4 runtime mismatch",
		},
		{
			name: "generate_status",
			labels: cloneGemma4LoRATestLabels(baseLabels, map[string]string{
				"adapter_base_gemma4_runtime":         Gemma4RuntimeGGUF,
				"adapter_base_gemma4_generate_status": Gemma4GenerateLinked,
			}),
			want: "adapter base Gemma4 generate status mismatch",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := checkROCmAdapterModelCompatibility("load adapter", model, inference.AdapterIdentity{
				Path:   "domain.safetensors",
				Format: "lora",
				Labels: tc.labels,
			})

			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tc.want)
		})
	}
}

func TestGemma4LoRAAdapter_Bad_LoadRejectsIncompleteGemma4BaseIdentity(t *testing.T) {
	for _, tc := range []struct {
		name   string
		labels map[string]string
	}{
		{
			name: "runtime_only",
			labels: map[string]string{
				"adapter_base_gemma4_runtime": Gemma4RuntimeMLXAffine,
			},
		},
		{
			name: "generate_status_only",
			labels: map[string]string{
				"gemma4_generate_status": Gemma4GenerateLinked,
			},
		},
		{
			name: "architecture_and_support_only",
			labels: map[string]string{
				"adapter_base_architecture":          "gemma4_text",
				"adapter_base_gemma4_pack_supported": "true",
			},
		},
		{
			name: "size_without_quant",
			labels: map[string]string{
				"adapter_base_gemma4_size":             "E4B",
				"adapter_base_gemma4_runnable_on_card": "true",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			native := &fakeNativeModel{
				loadAdapterIdentity: inference.AdapterIdentity{
					Path:   "domain.safetensors",
					Format: "lora",
					Labels: tc.labels,
				},
			}
			model := &rocmModel{
				modelPath: "/models/lmstudio-community-gemma-4-e4b-it-6bit",
				modelType: "gemma4_text",
				modelInfo: inference.ModelInfo{
					Architecture: "gemma4_text",
					NumLayers:    26,
					HiddenSize:   2304,
					VocabSize:    262144,
				},
				native: native,
			}

			identity, err := model.LoadAdapter("domain.safetensors")

			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), "adapter base Gemma4 identity is incomplete")
			if !adapterIdentityIsZero(identity) {
				t.Fatalf("identity = %+v, want zero on incomplete Gemma4 adapter", identity)
			}
			if !adapterIdentityIsZero(model.ActiveAdapter()) {
				t.Fatalf("active adapter = %+v, want cleared after incomplete Gemma4 adapter", model.ActiveAdapter())
			}
			core.AssertEqual(t, 1, native.unloadCalls)
		})
	}
}

func TestGemma4LoRAAdapter_Good_LoadInfersBaseLabelsFromAdapterBasePath(t *testing.T) {
	native := &fakeNativeModel{
		loadAdapterIdentity: inference.AdapterIdentity{
			Path:   "domain.safetensors",
			Format: "lora",
			Labels: map[string]string{
				"adapter_base_model_path": "/models/lmstudio-community-gemma-4-e4b-it-6bit",
			},
		},
	}
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    26,
			HiddenSize:   2304,
			VocabSize:    262144,
		},
		native: native,
	}

	identity, err := model.LoadAdapter("domain.safetensors")

	core.RequireNoError(t, err)
	assertGemma4LoRABaseLabels(t, identity, "gemma4_text", "E4B", "q6", Gemma4RuntimeMLXAffine, Gemma4GenerateLinked, "true", "true")
	core.AssertEqual(t, "/models/lmstudio-community-gemma-4-e4b-it-6bit", identity.Labels["adapter_base_model_path"])
}

func TestGemma4LoRAAdapter_Good_LoadInfersBaseLabelsFromProductionQuantModel(t *testing.T) {
	native := &fakeNativeModel{
		loadAdapterIdentity: inference.AdapterIdentity{
			Path:   "domain.safetensors",
			Format: "lora",
			Labels: map[string]string{
				"adapter_base_production_quant_model": ProductionLaneCurrentModelID,
			},
		},
	}
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e2b-it-6bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    productionLaneGemma4E2BLayers,
			HiddenSize:   productionLaneGemma4E2BHiddenSize,
			VocabSize:    262144,
		},
		native: native,
	}

	identity, err := model.LoadAdapter("domain.safetensors")

	core.RequireNoError(t, err)
	assertGemma4LoRABaseLabels(t, identity, "gemma4_text", "E2B", "q6", Gemma4RuntimeMLXAffine, Gemma4GenerateLinked, "true", "true")
	core.AssertEqual(t, ProductionLaneCurrentModelID, identity.Labels["adapter_base_production_quant_model"])
	core.AssertEqual(t, ProductionLaneModelID, identity.Labels["adapter_base_production_quant_locked_model"])
	core.AssertEqual(t, "/models/lmstudio-community-gemma-4-e2b-it-6bit", identity.Labels["adapter_base_model_path"])
}

func TestGemma4LoRAAdapter_Good_LoadInfersAssistantBaseLabelsFromProductionQuantAssistantModel(t *testing.T) {
	native := &fakeNativeModel{
		loadAdapterIdentity: inference.AdapterIdentity{
			Path:   "assistant-domain.safetensors",
			Format: "lora",
			Labels: map[string]string{
				"adapter_base_production_quant_assistant_model": rocmGemma4MTPAssistantPath("E4B", "bf16"),
			},
		},
	}
	model := &rocmModel{
		modelPath: rocmGemma4MTPAssistantPath("E4B", "bf16"),
		modelType: officialGemma4E2BAssistantArchitecture,
		modelInfo: inference.ModelInfo{
			Architecture: officialGemma4E2BAssistantArchitecture,
			NumLayers:    4,
			HiddenSize:   2304,
			VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
			QuantBits:    16,
		},
		native: native,
	}

	identity, err := model.LoadAdapter("assistant-domain.safetensors")

	core.RequireNoError(t, err)
	assertGemma4LoRABaseLabels(t, identity, officialGemma4E2BAssistantArchitecture, "E4B", "bf16", Gemma4RuntimeBF16, Gemma4GenerateLoadOnly, "true", "true")
	core.AssertEqual(t, rocmGemma4MTPAssistantPath("E4B", "bf16"), identity.Labels["adapter_base_production_quant_assistant_model"])
	core.AssertEqual(t, "E4B:assistant-bf16", identity.Labels["adapter_base_production_quant_pack"])
	core.AssertEqual(t, "true", identity.Labels["adapter_base_production_quant_mtp_assistant"])
	core.AssertEqual(t, "gemma4", identity.Labels["adapter_base_production_quant_target_family"])
}

func TestGemma4LoRAAdapter_Bad_LoadRejectsMismatchedBasePathOnlyAdapter(t *testing.T) {
	tests := []struct {
		name        string
		modelPath   string
		modelArch   string
		modelLayers int
		modelHidden int
		basePath    string
		want        string
	}{
		{
			name:        "size",
			modelPath:   "/models/lmstudio-community-gemma-4-e4b-it-6bit",
			modelArch:   "gemma4_text",
			modelLayers: 26,
			modelHidden: 2304,
			basePath:    "/models/lmstudio-community-gemma-4-e2b-it-6bit",
			want:        "adapter base Gemma4 size mismatch",
		},
		{
			name:        "quant",
			modelPath:   "/models/lmstudio-community-gemma-4-e2b-it-8bit",
			modelArch:   "gemma4_text",
			modelLayers: productionLaneGemma4E2BLayers,
			modelHidden: productionLaneGemma4E2BHiddenSize,
			basePath:    "/models/lmstudio-community-gemma-4-e2b-it-6bit",
			want:        "adapter base Gemma4 quant mismatch",
		},
		{
			name:        "non-gemma",
			modelPath:   "/models/qwen3",
			modelArch:   "qwen3",
			modelLayers: 24,
			modelHidden: 2048,
			basePath:    "/models/lmstudio-community-gemma-4-e2b-it-6bit",
			want:        "adapter Gemma4 base model mismatch",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			native := &fakeNativeModel{
				loadAdapterIdentity: inference.AdapterIdentity{
					Path:   "domain.safetensors",
					Format: "lora",
					Labels: map[string]string{
						"adapter_base_model_path": tc.basePath,
					},
				},
			}
			model := &rocmModel{
				modelPath: tc.modelPath,
				modelType: tc.modelArch,
				modelInfo: inference.ModelInfo{
					Architecture: tc.modelArch,
					NumLayers:    tc.modelLayers,
					HiddenSize:   tc.modelHidden,
					VocabSize:    262144,
				},
				native: native,
			}

			identity, err := model.LoadAdapter("domain.safetensors")

			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tc.want)
			if !adapterIdentityIsZero(identity) {
				t.Fatalf("identity = %+v, want zero on mismatched adapter", identity)
			}
			if !adapterIdentityIsZero(model.ActiveAdapter()) {
				t.Fatalf("active adapter = %+v, want cleared after mismatched adapter", model.ActiveAdapter())
			}
		})
	}
}

func TestGemma4LoRAAdapter_Bad_RestoreStateRejectsMismatchedAdapterBaseLabels(t *testing.T) {
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    26,
			HiddenSize:   2304,
			VocabSize:    262144,
		},
	}
	bundle := &inference.StateBundle{
		Model: inference.ModelIdentity{Architecture: "gemma4_text"},
		Adapter: inference.AdapterIdentity{
			Path:   "domain.safetensors",
			Format: "lora",
			Labels: map[string]string{
				"adapter_base_architecture":      "gemma4_text",
				"adapter_base_gemma4_size":       "E2B",
				"adapter_base_gemma4_quant_mode": "q6",
			},
		},
	}

	err := model.RestoreState(context.Background(), bundle)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "adapter base Gemma4 size mismatch")
	core.AssertNil(t, model.state)
}

func TestGemma4LoRAAdapter_Bad_RestoreStateRejectsMismatchedAdapterSupportLabels(t *testing.T) {
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-31b-it-6bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    64,
			HiddenSize:   4096,
			VocabSize:    262144,
		},
	}
	bundle := &inference.StateBundle{
		Model: inference.ModelIdentity{Architecture: "gemma4_text"},
		Adapter: inference.AdapterIdentity{
			Path:   "domain.safetensors",
			Format: "lora",
			Labels: map[string]string{
				"adapter_base_architecture":            "gemma4_text",
				"adapter_base_gemma4_size":             "31B",
				"adapter_base_gemma4_quant_mode":       "q6-status",
				"adapter_base_gemma4_generate_status":  Gemma4GeneratePlannedOnly,
				"adapter_base_gemma4_pack_supported":   "true",
				"adapter_base_gemma4_runnable_on_card": "true",
			},
		},
	}

	err := model.RestoreState(context.Background(), bundle)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "adapter base Gemma4 runnable status mismatch")
	core.AssertNil(t, model.state)
}

func assertGemma4LoRABaseLabels(t *testing.T, identity inference.AdapterIdentity, architecture, size, mode, runtime, status, supported, runnable string) {
	t.Helper()
	group := gemma4LoRAExpectedQuantGroup(mode)
	core.AssertEqual(t, size, identity.Labels["gemma4_size"])
	core.AssertEqual(t, mode, identity.Labels["gemma4_quant_mode"])
	core.AssertEqual(t, group, identity.Labels["gemma4_quant_group"])
	core.AssertEqual(t, runtime, identity.Labels["gemma4_runtime"])
	core.AssertEqual(t, status, identity.Labels["gemma4_generate_status"])
	core.AssertEqual(t, supported, identity.Labels["gemma4_pack_supported"])
	core.AssertEqual(t, runnable, identity.Labels["gemma4_runnable_on_card"])
	core.AssertEqual(t, architecture, identity.Labels["adapter_base_architecture"])
	core.AssertEqual(t, size, identity.Labels["adapter_base_gemma4_size"])
	core.AssertEqual(t, mode, identity.Labels["adapter_base_gemma4_quant_mode"])
	core.AssertEqual(t, group, identity.Labels["adapter_base_gemma4_quant_group"])
	core.AssertEqual(t, runtime, identity.Labels["adapter_base_gemma4_runtime"])
	core.AssertEqual(t, status, identity.Labels["adapter_base_gemma4_generate_status"])
	core.AssertEqual(t, supported, identity.Labels["adapter_base_gemma4_pack_supported"])
	core.AssertEqual(t, runnable, identity.Labels["adapter_base_gemma4_runnable_on_card"])
}

func gemma4LoRAExpectedQuantGroup(mode string) string {
	switch mode {
	case "q8", "q6", "q4":
		return "64"
	case "mxfp8", "mxfp4":
		return "32"
	default:
		return ""
	}
}

func cloneGemma4LoRATestLabels(base, overrides map[string]string) map[string]string {
	out := cloneStringMap(base)
	if out == nil {
		out = map[string]string{}
	}
	for key, value := range overrides {
		out[key] = value
	}
	return out
}
