// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "testing"

func TestFeaturesOf_Good_DerivesMultimodalFromDetailedConfigs(t *testing.T) {
	features := FeaturesOf(TextConfig{
		VisionConfig: VisionConfig{ModelType: "gemma4_vision"},
		AudioConfig:  AudioConfig{AudioTokenID: 258881},
	})

	if !features.Vision || !features.Audio {
		t.Fatalf("FeaturesOf(multimodal configs) = %+v, want vision and audio", features)
	}
}

func TestApplyVisionConfigLabels_Good_WritesGemma4VisionSurface(t *testing.T) {
	labels := ApplyVisionConfigLabels(nil, VisionConfig{
		ImageTokenID:          258880,
		VideoTokenID:          258884,
		BOITokenID:            256001,
		EOITokenIndex:         258882,
		SoftTokensPerImage:    280,
		ModelType:             "Gemma4-Vision",
		DType:                 "bfloat16",
		PatchSize:             16,
		HiddenSize:            768,
		IntermediateSize:      3072,
		NumHiddenLayers:       16,
		NumAttentionHeads:     12,
		NumKeyValueHeads:      12,
		HeadDim:               64,
		GlobalHeadDim:         64,
		PoolingKernelSize:     3,
		PositionEmbeddingSize: 10240,
		HiddenActivation:      "gelu_pytorch_tanh",
		RMSNormEps:            1e-6,
		RoPEParameters:        RoPEParameters{RopeTheta: 100, RopeType: "default"},
		UseClippedLinears:     true,
	})

	for key, want := range map[string]string{
		"image_token_id":                 "258880",
		"video_token_id":                 "258884",
		"boi_token_id":                   "256001",
		"eoi_token_index":                "258882",
		"vision_soft_tokens_per_image":   "280",
		"vision_model_type":              "gemma4_vision",
		"vision_dtype":                   "bf16",
		"vision_patch_size":              "16",
		"vision_hidden_size":             "768",
		"vision_intermediate_size":       "3072",
		"vision_num_hidden_layers":       "16",
		"vision_attention_heads":         "12",
		"vision_kv_heads":                "12",
		"vision_head_dim":                "64",
		"vision_global_head_dim":         "64",
		"vision_pooling_kernel_size":     "3",
		"vision_position_embedding_size": "10240",
		"vision_hidden_activation":       "gelu_pytorch_tanh",
		"vision_rms_norm_eps":            "1e-06",
		"vision_rope_theta":              "100",
		"vision_rope_type":               "default",
		"vision_standardize":             "false",
		"vision_use_clipped_linears":     "true",
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}
}

func TestApplyAudioConfigLabels_Good_WritesGemma4AudioSurface(t *testing.T) {
	labels := ApplyAudioConfigLabels(nil, AudioConfig{
		AudioTokenID:                258881,
		BOATokenID:                  256000,
		EOATokenIndex:               258883,
		ModelType:                   "Gemma4-Unified-Audio",
		HiddenSize:                  1024,
		AudioEmbedDim:               640,
		AudioSamplesPerToken:        640,
		NumHiddenLayers:             12,
		NumAttentionHeads:           8,
		AttentionChunkSize:          12,
		AttentionContextLeft:        13,
		AttentionInvalidLogitsValue: -1e9,
		ConvKernelSize:              5,
		OutputProjDims:              1536,
		RMSNormEps:                  1e-6,
		GradientClipping:            1e10,
		ResidualWeight:              0.5,
		HiddenAct:                   "silu",
		UseClippedLinears:           true,
	})

	for key, want := range map[string]string{
		"audio_token_id":                       "258881",
		"boa_token_id":                         "256000",
		"eoa_token_index":                      "258883",
		"audio_model_type":                     "gemma4_unified_audio",
		"audio_hidden_size":                    "1024",
		"audio_embed_dim":                      "640",
		"audio_samples_per_token":              "640",
		"audio_num_hidden_layers":              "12",
		"audio_attention_heads":                "8",
		"audio_attention_chunk_size":           "12",
		"audio_attention_context_left":         "13",
		"audio_attention_invalid_logits_value": "-1e+09",
		"audio_conv_kernel_size":               "5",
		"audio_output_proj_dims":               "1536",
		"audio_rms_norm_eps":                   "1e-06",
		"audio_gradient_clipping":              "1e+10",
		"audio_residual_weight":                "0.5",
		"audio_hidden_act":                     "silu",
		"audio_use_clipped_linears":            "true",
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}
}
