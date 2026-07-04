// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"path/filepath"
	"testing"
)

func TestModelPackGemma4Vision_MetadataLabels_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, `{
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
			"max_position_embeddings":131072,
			"model_type":"gemma4_vision",
			"num_attention_heads":12,
			"num_hidden_layers":16,
			"num_key_value_heads":12,
			"patch_size":16,
			"pooling_kernel_size":3,
			"position_embedding_size":10240,
			"rms_norm_eps":1e-6,
			"rope_parameters":{"rope_theta":100.0,"rope_type":"default"},
			"standardize":false,
			"use_clipped_linears":true
		},
		"vision_soft_tokens_per_image":280
	}`))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "gemma4" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported Gemma4 multimodal pack", inspection, inspection.Labels)
	}
	labels := inspection.Labels
	if labels["multimodal_model"] != "true" ||
		labels["architecture_resolved"] != "gemma4_text" ||
		labels["architecture_resolution_source"] != "model_type_text_tower" ||
		labels["engine_architecture_resolved_family"] != "gemma4" ||
		labels["engine_architecture_resolved_parser"] != "gemma" ||
		labels["engine_architecture_resolved_chat_template"] != "gemma4_hf_turn" ||
		labels["gemma4_multimodal"] != "true" ||
		labels["vision_runtime"] != hipKernelStatusNotLinked ||
		labels["vision_projector_runtime"] != hipKernelStatusNotLinked ||
		labels["vision_reference"] != "go_mlx_gemma4_vision" ||
		labels["image_token_id"] != "258880" ||
		labels["video_token_id"] != "258884" ||
		labels["vision_soft_tokens_per_image"] != "280" ||
		labels["vision_model_type"] != "gemma4_vision" ||
		labels["vision_dtype"] != "bf16" ||
		labels["vision_patch_size"] != "16" ||
		labels["vision_hidden_size"] != "768" ||
		labels["vision_intermediate_size"] != "3072" ||
		labels["vision_num_hidden_layers"] != "16" ||
		labels["vision_attention_heads"] != "12" ||
		labels["vision_kv_heads"] != "12" ||
		labels["vision_head_dim"] != "64" ||
		labels["vision_global_head_dim"] != "64" ||
		labels["vision_pooling_kernel_size"] != "3" ||
		labels["vision_position_embedding_size"] != "10240" ||
		labels["vision_hidden_activation"] != "gelu_pytorch_tanh" ||
		labels["vision_rope_theta"] != "100" ||
		labels["vision_rope_type"] != "default" ||
		labels["vision_standardize"] != "false" ||
		labels["vision_use_clipped_linears"] != "true" ||
		labels["engine_multimodal_processor_route_contract"] != ROCmMultimodalProcessorRegistryContract ||
		labels["engine_multimodal_processor_architecture"] != "gemma4" ||
		labels["engine_multimodal_processor_vision"] != "true" ||
		labels["engine_multimodal_processor_audio"] != "false" ||
		labels["engine_multimodal_processor_image_token_id"] != "258880" ||
		labels["engine_multimodal_processor_video_token_id"] != "258884" ||
		labels["engine_multimodal_processor_soft_tokens_per_image"] != "280" ||
		labels["engine_multimodal_processor_vision_model_type"] != "gemma4_vision" ||
		labels["engine_multimodal_processor_vision_hidden_size"] != "768" ||
		labels["engine_multimodal_processor_vision_projector_runtime"] != hipKernelStatusNotLinked {
		t.Fatalf("labels = %+v, want Gemma4 vision metadata/runtime labels", labels)
	}
	route, ok := ROCmMultimodalProcessorRouteForInspection(inspection)
	if !ok ||
		route.Architecture != "gemma4" ||
		route.Family != "gemma4" ||
		route.Reference != "go_mlx_gemma4_vision" ||
		!route.Multimodal ||
		!route.Vision ||
		!route.Video ||
		route.Audio ||
		route.NativeRuntime ||
		route.ImageTokenID != 258880 ||
		route.VideoTokenID != 258884 ||
		route.SoftTokensPerImage != 280 ||
		route.VisionModelType != "gemma4_vision" ||
		route.VisionHiddenSize != 768 ||
		route.VisionLayers != 16 {
		t.Fatalf("ROCmMultimodalProcessorRouteForInspection = %+v ok=%v, want loaded Gemma4 vision processor route", route, ok)
	}
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	appliedLabels := rocmApplyModelProfileLabels(nil, profile)
	if !ok ||
		!profile.Gemma4DeclaredFeatures.Vision ||
		profile.Gemma4DeclaredFeatures.Audio ||
		appliedLabels["gemma4_vision"] != "true" ||
		appliedLabels["gemma4_audio"] != "" {
		t.Fatalf("ROCm model profile = %+v labels=%+v ok=%v, want Gemma4 declared vision feature", profile, appliedLabels, ok)
	}
}

func TestModelPackGemma4ProcessorConfig_MetadataLabels_Good(t *testing.T) {
	dir := nativeContractSafetensorsPack(t, `{
		"model_type":"gemma4",
		"text_config":{
			"model_type":"gemma4_text",
			"hidden_size":2304,
			"num_hidden_layers":26,
			"vocab_size":262144,
			"max_position_embeddings":131072
		}
	}`)
	writeNativeContractFile(t, filepath.Join(dir, "processor_config.json"), `{
		"audio_ms_per_token":40,
		"audio_seq_length":750,
		"image_processor":{"max_soft_tokens":280,"do_resize":true},
		"video_processor":{"max_soft_tokens":70,"do_resize":true,"num_frames":32},
		"feature_extractor":{"sampling_rate":16000,"num_mel_filters":128,"fft_length":512,"hop_length":160}
	}`)

	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	labels := inspection.Labels
	for key, want := range map[string]string{
		"processor_config":                 "true",
		"multimodal_model":                 "true",
		"gemma4_multimodal":                "true",
		"vision_processor_config":          "true",
		"audio_processor_config":           "true",
		"image_processor":                  "true",
		"image_processor_patch_size":       "16",
		"image_processor_max_soft_tokens":  "280",
		"video_processor":                  "true",
		"video_processor_max_soft_tokens":  "70",
		"video_processor_num_frames":       "32",
		"audio_feature_extractor":          "true",
		"processor_audio_ms_per_token":     "40",
		"processor_audio_seq_length":       "750",
		"audio_feature_size":               "128",
		"audio_feature_frame_length":       "320",
		"audio_feature_hop_length":         "160",
		"audio_feature_fft_length":         "512",
		"audio_feature_max_length_samples": "480000",
		"audio_feature_pad_to_multiple":    "128",
		"vision_runtime":                   hipKernelStatusNotLinked,
		"vision_reference":                 "go_mlx_gemma4_vision",
		"audio_frontend_runtime":           hipKernelStatusNotLinked,
		"audio_reference":                  "go_mlx_gemma4_audio",
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok || !profile.Gemma4DeclaredFeatures.Vision || !profile.Gemma4DeclaredFeatures.Audio {
		t.Fatalf("ROCm model profile = %+v ok=%v labels=%+v, want processor config to declare Gemma4 multimodal features", profile, ok, labels)
	}
}
