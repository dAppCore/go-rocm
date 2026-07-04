// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && rocm_legacy_server

package rocm

import (
	"testing"

	"dappco.re/go/inference"
)

func TestLegacyModel_Good_ReportsReactiveRegistryIdentityAndProfile(t *testing.T) {
	var _ ROCmModelIdentityReporter = (*rocmModel)(nil)
	var _ ROCmModelProfileReporter = (*rocmModel)(nil)
	var _ inference.CapabilityReporter = (*rocmModel)(nil)

	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e4b-it-6bit.gguf",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			VocabSize:    262144,
			NumLayers:    26,
			HiddenSize:   2304,
			QuantBits:    6,
			QuantGroup:   64,
		},
		contextLength: 131072,
	}

	identity := model.ModelIdentity()
	if identity.Path != model.modelPath ||
		identity.Architecture != "gemma4_text" ||
		identity.ContextLength != 131072 ||
		identity.QuantType != "q6" ||
		identity.Labels["gemma4_size"] != "E4B" ||
		identity.Labels["gemma4_quant_mode"] != "q6" ||
		identity.Labels["gemma4_source_format"] != "gguf" {
		t.Fatalf("ModelIdentity = %+v, want path/context-aware Gemma4 legacy identity", identity)
	}
	identity.Labels["gemma4_size"] = "mutated"
	if next := model.ModelIdentity(); next.Labels["gemma4_size"] == "mutated" {
		t.Fatalf("ModelIdentity returned aliased labels: %+v", next.Labels)
	}

	profile := model.ModelProfile()
	if !profile.Matched() ||
		profile.Model.Path != model.modelPath ||
		profile.Model.ContextLength != 131072 ||
		profile.Model.QuantType != "q6" ||
		profile.Model.Labels["gemma4_size"] != "E4B" ||
		profile.Model.Labels["gemma4_source_format"] != "gguf" ||
		profile.Labels["engine_profile_reactive"] != "true" {
		t.Fatalf("ModelProfile = %+v, want resolved path/context-aware Gemma4 registry profile", profile)
	}
	profile.Model.Labels["gemma4_size"] = "mutated"
	profile.Labels["engine_profile_reactive"] = "mutated"
	nextProfile := model.ModelProfile()
	if nextProfile.Model.Labels["gemma4_size"] == "mutated" ||
		nextProfile.Labels["engine_profile_reactive"] == "mutated" {
		t.Fatalf("ModelProfile returned aliased profile data: %+v", nextProfile)
	}

	report := model.Capabilities()
	if report.Runtime.Backend != "rocm" ||
		report.Runtime.Labels["native_runtime"] != "llama_server" ||
		report.Available ||
		report.Model.ContextLength != 131072 ||
		report.Model.Labels["gemma4_size"] != "E4B" ||
		report.Labels["native_runtime"] != "llama_server" ||
		report.Labels["production_requires_env_gate"] != "false" ||
		report.Labels["production_requires_cli_flag"] != "false" ||
		report.Labels["engine_profile_reactive"] != "true" ||
		!report.Supports(inference.CapabilityGenerate) ||
		!report.Supports(inference.CapabilityChat) ||
		!report.Supports(inference.CapabilityClassify) ||
		!report.Supports(inference.CapabilityBatchGenerate) ||
		!report.Supports(inference.CapabilityModelLoad) ||
		!report.Supports(inference.CapabilityChatTemplate) {
		t.Fatalf("Capabilities = %+v, want legacy llama-server model capability report with reactive profile labels", report)
	}
	generate, ok := report.Capability(inference.CapabilityGenerate)
	if !ok ||
		generate.Labels["native_runtime"] != "llama_server" ||
		generate.Labels["gemma4_size"] != "E4B" ||
		generate.Labels["gemma4_source_format"] != "gguf" {
		t.Fatalf("generate capability = %+v ok=%v, want legacy model labels", generate, ok)
	}
	report.Model.Labels["gemma4_size"] = "mutated"
	report.Runtime.Labels["native_runtime"] = "mutated"
	report.Labels["engine_profile_reactive"] = "mutated"
	for index := range report.Capabilities {
		if report.Capabilities[index].ID == inference.CapabilityGenerate {
			report.Capabilities[index].Labels["gemma4_size"] = "mutated"
		}
	}
	nextReport := model.Capabilities()
	nextGenerate, ok := nextReport.Capability(inference.CapabilityGenerate)
	if !ok ||
		nextReport.Model.Labels["gemma4_size"] == "mutated" ||
		nextReport.Runtime.Labels["native_runtime"] == "mutated" ||
		nextReport.Labels["engine_profile_reactive"] == "mutated" ||
		nextGenerate.Labels["gemma4_size"] == "mutated" {
		t.Fatalf("Capabilities returned aliased report data: %+v capability=%+v ok=%v", nextReport, nextGenerate, ok)
	}
}

func TestLegacyModel_Good_InfoCanonicalizesRegistryMetadata(t *testing.T) {
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e4b-it-6bit.gguf",
		modelType: "Gemma4ForCausalLM",
		modelInfo: inference.ModelInfo{
			Architecture: "Gemma4ForCausalLM",
			VocabSize:    262144,
			NumLayers:    26,
			HiddenSize:   2304,
		},
		contextLength: 131072,
	}

	info := model.Info()
	if info.Architecture != "gemma4_text" ||
		info.VocabSize != 262144 ||
		info.NumLayers != 26 ||
		info.HiddenSize != 2304 ||
		info.QuantBits != 6 ||
		info.QuantGroup != 64 {
		t.Fatalf("Info = %+v, want canonical Gemma4 text q6 metadata inferred from legacy model path", info)
	}
}
