// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	modelgemma4 "dappco.re/go/rocm/model/gemma4"
)

func TestGemma4MTPAssistantIdentityForTarget_FiveSizesBF16(t *testing.T) {
	tests := []struct {
		name         string
		targetPath   string
		wantSize     string
		wantHidden   int
		wantOfficial bool
		wantFamily   bool
	}{
		{name: "E2B", targetPath: "/models/lmstudio-community-gemma-4-e2b-it-6bit", wantSize: "E2B", wantHidden: productionLaneGemma4E2BHiddenSize, wantOfficial: true, wantFamily: true},
		{name: "E4B", targetPath: "/models/lmstudio-community-gemma-4-e4b-it-8bit", wantSize: "E4B", wantHidden: 2304, wantFamily: true},
		{name: "12B", targetPath: "/models/lmstudio-community-gemma-4-12b-it-6bit", wantSize: "12B", wantHidden: 3840, wantFamily: true},
		{name: "26B-A4B", targetPath: "/models/lmstudio-community-gemma-4-26b-a4b-it-6bit", wantSize: "26B-A4B", wantHidden: 4096},
		{name: "31B", targetPath: "/models/lmstudio-community-gemma-4-31b-it-6bit", wantSize: "31B", wantHidden: 4096},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			target := rocmGemma4ModelWithInferredPathQuant(inference.ModelIdentity{
				Path:         tt.targetPath,
				Architecture: "gemma4_text",
			})

			assistant := rocmGemma4MTPAssistantIdentityForTarget(target)

			core.AssertEqual(t, "google/gemma-4-"+tt.wantSize+"-it-assistant", assistant.Path)
			core.AssertEqual(t, officialGemma4E2BAssistantArchitecture, assistant.Architecture)
			core.AssertEqual(t, 4, assistant.NumLayers)
			core.AssertEqual(t, tt.wantHidden, assistant.HiddenSize)
			core.AssertEqual(t, ProductionMTPAssistantTokenOrderingVocabSize, assistant.VocabSize)
			core.AssertEqual(t, 16, assistant.QuantBits)
			core.AssertEqual(t, "bf16", assistant.QuantType)
			core.AssertEqual(t, tt.wantSize, assistant.Labels["gemma4_size"])
			core.AssertEqual(t, "bf16", assistant.Labels["gemma4_quant_mode"])
			core.AssertEqual(t, Gemma4RuntimeBF16, assistant.Labels["gemma4_runtime"])
			core.AssertEqual(t, Gemma4GenerateLoadOnly, assistant.Labels["gemma4_generate_status"])
			core.AssertEqual(t, "true", assistant.Labels["gemma4_pack_supported"])
			core.AssertEqual(t, "true", assistant.Labels["gemma4_runnable_on_card"])
			core.AssertEqual(t, tt.wantSize+":assistant-bf16", assistant.Labels["production_quant_pack"])
			core.AssertEqual(t, "mtp-assistant", assistant.Labels["production_quant_tier"])
			core.AssertEqual(t, "google/gemma-4-"+tt.wantSize+"-it-assistant", assistant.Labels["production_quant_model"])
			core.AssertEqual(t, "google/gemma-4-"+tt.wantSize+"-it-assistant", assistant.Labels["production_quant_assistant_model"])
			core.AssertEqual(t, "true", assistant.Labels["production_quant_mtp_assistant"])
			core.AssertEqual(t, "gemma4", assistant.Labels["production_quant_target_family"])
			core.AssertEqual(t, rocmModelRegistryName, assistant.Labels["engine_registry"])
			core.AssertEqual(t, "gemma4", assistant.Labels["engine_profile"])
			core.AssertEqual(t, "gemma4_assistant", assistant.Labels["engine_profile_architecture"])
			core.AssertEqual(t, "gemma4_assistant", assistant.Labels["engine_architecture_profile"])
			core.AssertEqual(t, string(inference.FeatureRuntimeNative), assistant.Labels["engine_architecture_runtime_status"])
			core.AssertEqual(t, "gemma", assistant.Labels["engine_architecture_reasoning_parser"])
			core.AssertEqual(t, "retained-state,attached-drafter", assistant.Labels["engine_architecture_cache_hints"])
			core.AssertEqual(t, "true", assistant.Labels["engine_architecture_attached_only"])
			core.AssertEqual(t, "false", assistant.Labels["engine_architecture_generation"])
			core.AssertEqual(t, "false", assistant.Labels["engine_architecture_chat"])
			core.AssertEqual(t, "", assistant.Labels["engine_chat_template"])
			core.AssertEqual(t, "", assistant.Labels["gemma4_lora_default_targets"])
			core.AssertEqual(t, "", assistant.Labels["gemma4_weight_policy"])
			core.AssertEqual(t, tt.wantOfficial, rocmGemma4AttachedDrafterOfficialPairVerified(target, assistant))
			core.AssertEqual(t, tt.wantFamily, rocmGemma4AttachedDrafterFamilyPairVerified(target, assistant))

			pack, ok := rocmGemma4ProductionQuantPackForModel(assistant)
			core.AssertTrue(t, ok)
			core.AssertEqual(t, tt.wantSize, pack.Size)
			core.AssertEqual(t, "bf16", pack.QuantMode)
			core.AssertEqual(t, "mtp-assistant", pack.ProductRole)
			core.AssertEqual(t, "google/gemma-4-"+tt.wantSize+"-it-assistant", pack.ModelID)
			core.AssertEqual(t, Gemma4RuntimeBF16, pack.Runtime)
			core.AssertEqual(t, Gemma4GenerateLoadOnly, pack.GenerateStatus)
			core.AssertEqual(t, true, pack.RunnableOnCard)

			alias, ok := ProductionQuantizationPackByName("google/gemma-4-" + tt.wantSize + "-it-assistant")
			core.AssertTrue(t, ok)
			core.AssertEqual(t, pack.ModelID, alias.ModelID)

			labels := map[string]string{}
			rocmApplyGemma4ProductionQuantLabels(labels, assistant)
			core.AssertEqual(t, tt.wantSize+":assistant-bf16", labels["production_quant_pack"])
			core.AssertEqual(t, "mtp-assistant", labels["production_quant_tier"])
			core.AssertEqual(t, pack.ModelID, labels["production_quant_model"])
			core.AssertEqual(t, pack.ModelID, labels["production_quant_assistant_model"])
			core.AssertEqual(t, "true", labels["production_quant_mtp_assistant"])
			core.AssertEqual(t, "", labels["production_quant_target_model"])
		})
	}
}

func TestGemma4MTPAssistantIdentity_Good_UsesAssistantPathMetadata(t *testing.T) {
	assistant := rocmGemma4ModelWithInferredPathQuant(inference.ModelIdentity{
		Path:         "google/gemma-4-E2B-it-assistant",
		Architecture: officialGemma4E2BAssistantArchitecture,
		NumLayers:    4,
		HiddenSize:   productionLaneGemma4E2BHiddenSize,
		VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		QuantBits:    16,
	})

	core.AssertEqual(t, "E2B", assistant.Labels["gemma4_size"])
	core.AssertEqual(t, "bf16", assistant.Labels["gemma4_quant_mode"])
	core.AssertEqual(t, Gemma4RuntimeBF16, assistant.Labels["gemma4_runtime"])
	core.AssertEqual(t, Gemma4GenerateLoadOnly, assistant.Labels["gemma4_generate_status"])
	core.AssertEqual(t, "true", assistant.Labels["gemma4_pack_supported"])
	core.AssertEqual(t, "true", assistant.Labels["gemma4_runnable_on_card"])
	core.AssertEqual(t, "E2B:assistant-bf16", assistant.Labels["production_quant_pack"])
	core.AssertEqual(t, "mtp-assistant", assistant.Labels["production_quant_tier"])
	core.AssertEqual(t, "google/gemma-4-E2B-it-assistant", assistant.Labels["production_quant_model"])
	core.AssertEqual(t, "true", assistant.Labels["production_quant_mtp_assistant"])
	core.AssertEqual(t, "gemma4", assistant.Labels["production_quant_target_family"])
	core.AssertEqual(t, "gemma4", assistant.Labels["engine_profile"])
	core.AssertEqual(t, "gemma4_assistant", assistant.Labels["engine_architecture_profile"])
	core.AssertEqual(t, string(inference.FeatureRuntimeNative), assistant.Labels["engine_architecture_runtime_status"])
	core.AssertEqual(t, "true", assistant.Labels["engine_architecture_attached_only"])
}

func TestGemma4MTPAssistantIdentityForTarget_Good_QATTargetUsesMTPQATAssistant(t *testing.T) {
	target := rocmGemma4ModelWithInferredPathQuant(inference.ModelIdentity{
		Path:         "mlx-community/gemma-4-12B-it-qat-4bit",
		Architecture: "gemma4_text",
	})

	assistant := rocmGemma4MTPAssistantIdentityForTarget(target)

	core.AssertEqual(t, "mlx-community/gemma-4-12B-it-qat-assistant-4bit", assistant.Path)
	core.AssertEqual(t, officialGemma4E2BAssistantArchitecture, assistant.Architecture)
	core.AssertEqual(t, "12B", assistant.Labels["gemma4_size"])
	core.AssertEqual(t, "q4", assistant.Labels["gemma4_quant_mode"])
	core.AssertEqual(t, 4, assistant.QuantBits)
	core.AssertEqual(t, 64, assistant.QuantGroup)
	core.AssertEqual(t, Gemma4RuntimeMLXAffine, assistant.Labels["gemma4_runtime"])
	core.AssertEqual(t, Gemma4GenerateLoadOnly, assistant.Labels["gemma4_generate_status"])
	core.AssertEqual(t, modelgemma4.MTPQATCollectionID, assistant.Labels["production_quant_collection"])
	core.AssertEqual(t, "12B:assistant-q4", assistant.Labels["production_quant_pack"])
	core.AssertEqual(t, "mlx-community/gemma-4-12B-it-qat-assistant-4bit", assistant.Labels["production_quant_model"])
	core.AssertEqual(t, false, rocmGemma4AttachedDrafterOfficialPairVerified(target, assistant))
	core.AssertEqual(t, true, rocmGemma4AttachedDrafterFamilyPairVerified(target, assistant))
}
