// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"
)

func TestModelPackGemma4Audio_MetadataLabels_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, `{
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
			"attention_context_right":0,
			"conv_kernel_size":5,
			"output_proj_dims":1536,
			"rms_norm_eps":1e-6,
			"gradient_clipping":1e10,
			"residual_weight":0.5,
			"hidden_act":"silu",
			"use_clipped_linears":true
		}
	}`))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "gemma4_unified" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported Gemma4 unified audio pack", inspection, inspection.Labels)
	}
	labels := inspection.Labels
	if labels["multimodal_model"] != "true" ||
		labels["architecture_resolved"] != "gemma4_unified" ||
		labels["architecture_resolution_source"] != "model_type" ||
		labels["engine_architecture_resolved_family"] != "gemma4" ||
		labels["engine_architecture_resolved_parser"] != "gemma" ||
		labels["engine_architecture_resolved_chat_template"] != "gemma4_hf_turn" ||
		labels["gemma4_multimodal"] != "true" ||
		labels["audio_runtime"] != hipKernelStatusNotLinked ||
		labels["audio_projector_runtime"] != hipKernelStatusNotLinked ||
		labels["audio_reference"] != "go_mlx_gemma4_audio" ||
		labels["audio_token_id"] != "258881" ||
		labels["boa_token_id"] != "256000" ||
		labels["eoa_token_index"] != "258883" ||
		labels["audio_model_type"] != "gemma4_unified_audio" ||
		labels["audio_hidden_size"] != "1024" ||
		labels["audio_embed_dim"] != "640" ||
		labels["audio_samples_per_token"] != "640" ||
		labels["audio_num_hidden_layers"] != "12" ||
		labels["audio_attention_heads"] != "8" ||
		labels["audio_attention_chunk_size"] != "12" ||
		labels["audio_attention_context_left"] != "13" ||
		labels["audio_conv_kernel_size"] != "5" ||
		labels["audio_output_proj_dims"] != "1536" ||
		labels["audio_rms_norm_eps"] != "1e-06" ||
		labels["audio_gradient_clipping"] != "1e+10" ||
		labels["audio_residual_weight"] != "0.5" ||
		labels["audio_hidden_act"] != "silu" ||
		labels["audio_use_clipped_linears"] != "true" ||
		labels["engine_multimodal_processor_route_contract"] != ROCmMultimodalProcessorRegistryContract ||
		labels["engine_multimodal_processor_architecture"] != "gemma4_unified" ||
		labels["engine_multimodal_processor_vision"] != "false" ||
		labels["engine_multimodal_processor_audio"] != "true" ||
		labels["engine_multimodal_processor_audio_token_id"] != "258881" ||
		labels["engine_multimodal_processor_boa_token_id"] != "256000" ||
		labels["engine_multimodal_processor_eoa_token_index"] != "258883" ||
		labels["engine_multimodal_processor_audio_model_type"] != "gemma4_unified_audio" ||
		labels["engine_multimodal_processor_audio_hidden_size"] != "1024" ||
		labels["engine_multimodal_processor_audio_samples_per_token"] != "640" ||
		labels["engine_multimodal_processor_audio_projector_runtime"] != hipKernelStatusNotLinked ||
		labels["engine_multimodal_processor_audio_front_end_runtime"] != hipKernelStatusNotLinked {
		t.Fatalf("labels = %+v, want Gemma4 audio metadata/runtime labels", labels)
	}
	route, ok := ROCmMultimodalProcessorRouteForInspection(inspection)
	if !ok ||
		route.Architecture != "gemma4_unified" ||
		route.Family != "gemma4" ||
		route.Reference != "go_mlx_gemma4_audio" ||
		!route.Multimodal ||
		route.Vision ||
		!route.Audio ||
		route.NativeRuntime ||
		route.AudioTokenID != 258881 ||
		route.BOATokenID != 256000 ||
		route.EOATokenIndex != 258883 ||
		route.AudioModelType != "gemma4_unified_audio" ||
		route.AudioHiddenSize != 1024 ||
		route.AudioSamplesPerToken != 640 ||
		route.AudioFrontEndRuntime != hipKernelStatusNotLinked {
		t.Fatalf("ROCmMultimodalProcessorRouteForInspection = %+v ok=%v, want loaded Gemma4 audio processor route", route, ok)
	}
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	appliedLabels := rocmApplyModelProfileLabels(nil, profile)
	if !ok ||
		profile.Gemma4DeclaredFeatures.Vision ||
		!profile.Gemma4DeclaredFeatures.Audio ||
		appliedLabels["gemma4_vision"] != "" ||
		appliedLabels["gemma4_audio"] != "true" {
		t.Fatalf("ROCm model profile = %+v labels=%+v ok=%v, want Gemma4 declared audio feature", profile, appliedLabels, ok)
	}
}
