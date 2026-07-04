// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"
)

func TestModelPackGemma3MultimodalWrapper_MetadataLabels_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, `{
		"architectures":["Gemma3ForConditionalGeneration"],
		"boi_token_index":255999,
		"eoi_token_index":256000,
		"image_token_index":262144,
		"mm_tokens_per_image":256,
		"model_type":"gemma3",
		"text_config":{
			"model_type":"gemma3_text",
			"hidden_size":3840,
			"intermediate_size":15360,
			"num_hidden_layers":48,
			"num_attention_heads":16,
			"num_key_value_heads":8,
			"head_dim":256,
			"vocab_size":262208,
			"max_position_embeddings":131072,
			"sliding_window":1024,
			"sliding_window_pattern":6,
			"rope_local_base_freq":10000
		}
	}`))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "gemma3" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported Gemma3 multimodal wrapper", inspection, inspection.Labels)
	}
	labels := inspection.Labels
	if labels["multimodal_model"] != "true" ||
		labels["gemma3_multimodal"] != "true" ||
		labels["gemma4_multimodal"] != "" ||
		labels["vision_runtime"] != hipKernelStatusNotLinked ||
		labels["vision_projector_runtime"] != hipKernelStatusNotLinked ||
		labels["vision_reference"] != "go_mlx_gemma3_multimodal_wrapper" ||
		labels["image_token_id"] != "262144" ||
		labels["boi_token_index"] != "255999" ||
		labels["eoi_token_index"] != "256000" ||
		labels["vision_soft_tokens_per_image"] != "256" ||
		labels["sliding_window"] != "1024" ||
		labels["attention_layer_types"] == "" {
		t.Fatalf("labels = %+v, want Gemma3 multimodal wrapper labels", labels)
	}
}

func TestModelPackGemma3TextStandalone_MetadataLabels_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, `{
		"architectures":["Gemma3TextForCausalLM"],
		"model_type":"gemma3_text",
		"hidden_size":3840,
		"intermediate_size":15360,
		"num_hidden_layers":48,
		"num_attention_heads":16,
		"num_key_value_heads":8,
		"head_dim":256,
		"vocab_size":262208,
		"max_position_embeddings":131072,
		"sliding_window":1024,
		"sliding_window_pattern":6
	}`))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "gemma3_text" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported standalone Gemma3 text model", inspection, inspection.Labels)
	}
	labels := inspection.Labels
	if labels["architecture_resolution_source"] != "model_type" ||
		labels["architecture_model_type"] != "gemma3_text" ||
		labels["engine_loader"] != "gemma3_text" ||
		labels["engine_load_architecture"] != "gemma3_text" ||
		labels["engine_route_set_loader_name"] != "gemma3_text" ||
		labels["engine_tokenizer_architecture"] != "gemma3_text" ||
		labels["engine_tokenizer_kind"] != "GemmaTokenizer" ||
		labels["multimodal_model"] != "" ||
		labels["gemma3_multimodal"] != "" {
		t.Fatalf("labels = %+v, want standalone Gemma3 text routing labels", labels)
	}
}
