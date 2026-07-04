// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestModelPackGemma4Assistant_ConfigEvidence_Good(t *testing.T) {
	config := `{
		"model_type":"gemma4_assistant",
		"backbone_hidden_size":2304,
		"hidden_size":1536,
		"vocab_size":262144,
		"num_hidden_layers":4,
		"num_attention_heads":4,
		"head_dim":512,
		"num_centroids":2048,
		"centroid_intermediate_top_k":32,
		"use_ordered_embeddings":true,
		"max_position_embeddings":131072
	}`
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, config))
	core.RequireNoError(t, err)

	labels := inspection.Labels
	core.AssertEqual(t, "gemma4_assistant", inspection.Model.Architecture)
	core.AssertEqual(t, "2304", labels["attached_drafter_assistant_backbone_hidden_size"])
	core.AssertEqual(t, "4", labels["attached_drafter_assistant_layer_count"])
	core.AssertEqual(t, "true", labels["attached_drafter_assistant_four_layer_drafter"])
	core.AssertEqual(t, "true", labels["attached_drafter_assistant_ordered_embeddings"])
	core.AssertEqual(t, "2048", labels["attached_drafter_assistant_centroids"])
	core.AssertEqual(t, "32", labels["attached_drafter_assistant_centroid_intermediate_top_k"])
	core.AssertEqual(t, "2048x128", labels["attached_drafter_assistant_token_ordering_shape"])
	core.AssertEqual(t, "false", labels["attached_drafter_official_pair_verified"])
	core.AssertEqual(t, "false", labels["attached_drafter_gemma4_family_pair_verified"])
}

func TestModelPackGemma4Assistant_FiveBF16PacksCarrySupportLabels(t *testing.T) {
	config := `{
		"model_type":"gemma4_assistant",
		"dtype":"bfloat16",
		"backbone_hidden_size":2304,
		"hidden_size":1536,
		"vocab_size":262144,
		"num_hidden_layers":4,
		"num_attention_heads":4,
		"head_dim":512,
		"num_centroids":2048,
		"centroid_intermediate_top_k":32,
		"use_ordered_embeddings":true,
		"max_position_embeddings":131072
	}`
	tests := []struct {
		name string
		size string
		path string
	}{
		{name: "E2B", size: "E2B", path: "google/gemma-4-E2B-it-assistant"},
		{name: "E4B", size: "E4B", path: "google/gemma-4-E4B-it-assistant"},
		{name: "12B", size: "12B", path: "google/gemma-4-12B-it-assistant"},
		{name: "26B-A4B", size: "26B-A4B", path: "google/gemma-4-26B-A4B-it-assistant"},
		{name: "31B", size: "31B", path: "google/gemma-4-31B-it-assistant"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), gemma4AssistantSafetensorsPack(t, tt.path, config))
			core.RequireNoError(t, err)

			labels := inspection.Labels
			core.AssertTrue(t, inspection.Supported)
			core.AssertEqual(t, "gemma4_assistant", inspection.Model.Architecture)
			core.AssertEqual(t, 16, inspection.Model.QuantBits)
			core.AssertEqual(t, "bf16", inspection.Model.QuantType)
			core.AssertEqual(t, tt.size, labels["gemma4_size"])
			core.AssertEqual(t, "bf16", labels["gemma4_quant_mode"])
			core.AssertEqual(t, Gemma4RuntimeBF16, labels["gemma4_runtime"])
			core.AssertEqual(t, Gemma4GenerateLoadOnly, labels["gemma4_generate_status"])
			core.AssertEqual(t, "true", labels["gemma4_pack_supported"])
			core.AssertEqual(t, "true", labels["gemma4_runnable_on_card"])
			core.AssertEqual(t, "gemma4_assistant", labels["attached_drafter_role"])
			core.AssertEqual(t, "false", labels["attached_drafter_official_pair_verified"])
			core.AssertEqual(t, "false", labels["attached_drafter_gemma4_family_pair_verified"])
			load, ok := nativeInspectionCapability(inspection, inference.CapabilityModelLoad)
			core.AssertTrue(t, ok)
			core.AssertEqual(t, tt.size, load.Labels["gemma4_size"])
			core.AssertEqual(t, "bf16", load.Labels["gemma4_quant_mode"])
			core.AssertEqual(t, Gemma4GenerateLoadOnly, load.Labels["gemma4_generate_status"])
			core.AssertEqual(t, "gemma4_assistant", load.Labels["attached_drafter_role"])
			core.AssertEqual(t, "true", load.Labels["attached_drafter_retained_state_required"])
			core.AssertEqual(t, "forbidden", load.Labels["attached_drafter_prompt_replay_fallback"])
			core.AssertEqual(t, "gemma4_mlx_affine", load.Labels["production_quant_policy"])
			core.AssertEqual(t, Gemma4RuntimeBF16, load.Labels["production_quant_runtime"])
			core.AssertEqual(t, Gemma4GenerateLoadOnly, load.Labels["production_quant_generate_status"])
			for _, id := range []inference.CapabilityID{
				inference.CapabilityModelFit,
				inference.CapabilityMemoryPlanning,
				inference.CapabilityKVCachePlanning,
			} {
				capability, ok := nativeInspectionCapability(inspection, id)
				core.AssertTrue(t, ok)
				core.AssertEqual(t, tt.size, capability.Labels["gemma4_size"])
				core.AssertEqual(t, "bf16", capability.Labels["gemma4_quant_mode"])
				core.AssertEqual(t, Gemma4GenerateLoadOnly, capability.Labels["gemma4_generate_status"])
				core.AssertEqual(t, "gemma4_assistant", capability.Labels["attached_drafter_role"])
				core.AssertEqual(t, "drafter", capability.Labels["mtp_role"])
				core.AssertEqual(t, "gemma4", capability.Labels["mtp_target_family"])
				core.AssertEqual(t, "gemma4_mlx_affine", capability.Labels["production_quant_policy"])
				core.AssertEqual(t, Gemma4RuntimeBF16, capability.Labels["production_quant_runtime"])
				core.AssertEqual(t, Gemma4GenerateLoadOnly, capability.Labels["production_quant_generate_status"])
			}
		})
	}
}

func TestModelPackGemma4Assistant_Good_QATAssistantLoadsClosed(t *testing.T) {
	config := `{
		"model_type":"gemma4_assistant",
		"quantization_config":{"bits":4,"quant_method":"q4"},
		"vocab_size":262144,
		"num_hidden_layers":4,
		"num_centroids":2048,
		"centroid_intermediate_top_k":32,
		"use_ordered_embeddings":true,
		"max_position_embeddings":131072
	}`

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), gemma4AssistantSafetensorsPack(t, "mlx-community-gemma-4-E2B-it-assistant-q4", config))
	core.RequireNoError(t, err)

	core.AssertEqual(t, true, inspection.Supported)
	core.AssertEqual(t, "E2B", inspection.Labels["gemma4_size"])
	core.AssertEqual(t, "q4", inspection.Labels["gemma4_quant_mode"])
	core.AssertEqual(t, "true", inspection.Labels["gemma4_pack_supported"])
	core.AssertEqual(t, Gemma4GenerateLoadOnly, inspection.Labels["gemma4_generate_status"])
	if nativeInspectionHasCapability(inspection, inference.CapabilityGenerate) {
		t.Fatalf("capabilities = %+v, assistant q4 must not expose linked generation", inspection.Capabilities)
	}
}

func TestModelPackGemma4Assistant_Bad_UnsupportedQATAssistantFailsClosed(t *testing.T) {
	config := `{
		"model_type":"gemma4_assistant",
		"quantization_config":{"bits":3,"quant_method":"q3"},
		"vocab_size":262144,
		"num_hidden_layers":4,
		"num_centroids":2048,
		"centroid_intermediate_top_k":32,
		"use_ordered_embeddings":true,
		"max_position_embeddings":131072
	}`

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), gemma4AssistantSafetensorsPack(t, "mlx-community-gemma-4-E2B-it-assistant-q3", config))
	core.RequireNoError(t, err)

	core.AssertFalse(t, inspection.Supported)
	core.AssertEqual(t, "E2B", inspection.Labels["gemma4_size"])
	core.AssertEqual(t, "q3", inspection.Labels["gemma4_quant_mode"])
	core.AssertEqual(t, "false", inspection.Labels["gemma4_pack_supported"])
	if nativeInspectionHasCapability(inspection, inference.CapabilityGenerate) {
		t.Fatalf("capabilities = %+v, assistant q3 must not expose linked generation", inspection.Capabilities)
	}
}

func TestModelPackGemma4Assistant_ConfigEvidence_Bad_RejectsStaticOfficialPromotion(t *testing.T) {
	config := `{
		"model_type":"gemma4_assistant",
		"backbone_hidden_size":2304,
		"hidden_size":1536,
		"vocab_size":262144,
		"num_hidden_layers":6,
		"num_attention_heads":4,
		"head_dim":512,
		"num_centroids":1024,
		"centroid_intermediate_top_k":16,
		"use_ordered_embeddings":false,
		"max_position_embeddings":131072
	}`
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, config))
	core.RequireNoError(t, err)

	labels := inspection.Labels
	core.AssertEqual(t, "gemma4_assistant", inspection.Model.Architecture)
	core.AssertEqual(t, "6", labels["attached_drafter_assistant_layer_count"])
	core.AssertEqual(t, "false", labels["attached_drafter_assistant_four_layer_drafter"])
	core.AssertEqual(t, "false", labels["attached_drafter_assistant_ordered_embeddings"])
	core.AssertEqual(t, "1024", labels["attached_drafter_assistant_centroids"])
	core.AssertEqual(t, "16", labels["attached_drafter_assistant_centroid_intermediate_top_k"])
	core.AssertEqual(t, "1024x256", labels["attached_drafter_assistant_token_ordering_shape"])
	core.AssertEqual(t, "false", labels["attached_drafter_official_pair_verified"])
	core.AssertEqual(t, "false", labels["attached_drafter_gemma4_family_pair_verified"])
	core.AssertTrue(t, nativeContractHasNoteContaining(inspection.Notes, "does not match the locked official E2B assistant layout"))
}

func gemma4AssistantSafetensorsPack(t *testing.T, dirName, config string) string {
	t.Helper()
	dir := core.PathJoin(t.TempDir(), dirName)
	core.RequireNoError(t, os.MkdirAll(dir, 0o755))
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), config)
	writeNativeContractSafetensors(t, core.PathJoin(dir, "model.safetensors"))
	return dir
}
