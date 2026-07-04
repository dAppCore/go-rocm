// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	"dappco.re/go/inference"
)

func TestFeaturesOf_Good_ConfigDeclaresReactiveSurface(t *testing.T) {
	features := FeaturesOf(TextConfig{
		EnableMoEBlock:       true,
		NumExperts:           128,
		TopKExperts:          8,
		Vision:               true,
		Audio:                true,
		SlidingWindow:        1024,
		SlidingWindowPattern: 6,
		KVSharedLayers:       4,
		QuantBits:            6,
		QuantMode:            "q6",
	})
	if !features.Mixture ||
		features.NumExperts != 128 ||
		features.TopKExperts != 8 ||
		!features.Vision ||
		!features.Audio ||
		!features.Attention.Hybrid() ||
		features.Attention.SlidingWindow != 1024 ||
		features.Attention.SlidingPattern != 6 ||
		features.Attention.SharedKVLayers != 4 ||
		!features.Quantization.Q6Bitstream() {
		t.Fatalf("FeaturesOf(config) = %+v, want MoE, multimodal, hybrid attention, and q6 quantization", features)
	}

	dense := FeaturesOf(TextConfig{NumExperts: 128, TopKExperts: 8})
	if dense.Mixture || dense.NumExperts != 0 || dense.TopKExperts != 0 || dense.Attention.Hybrid() || dense.Quantization.Q6Bitstream() {
		t.Fatalf("FeaturesOf(dense config) = %+v, want dense text-only surface", dense)
	}
}

func TestFeaturesOfIdentity_Good_LabelsDeclareReactiveSurface(t *testing.T) {
	features := FeaturesOfIdentity(inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels: map[string]string{
			"gemma4_enable_moe_block":           " true ",
			"gemma4_num_experts":                "128",
			"gemma4_top_k_experts":              "8",
			"sliding_window":                    "1024",
			"sliding_window_pattern":            "6",
			"attention_kv_shared_layers":        "4",
			"vision_model_type":                 "gemma4_vision",
			"engine_multimodal_processor_audio": "yes",
			"gemma4_quant_mode":                 "q6",
		},
	})
	if !features.Mixture ||
		features.NumExperts != 128 ||
		features.TopKExperts != 8 ||
		!features.Vision ||
		!features.Audio ||
		features.Attention.SlidingWindow != 1024 ||
		features.Attention.SlidingPattern != 6 ||
		features.Attention.SharedKVLayers != 4 ||
		!features.Quantization.Q6Bitstream() {
		t.Fatalf("FeaturesOfIdentity(labels) = %+v, want label-derived surface", features)
	}
}

func TestEngineFeaturesOf_Good_CacheSelectionFollowsFeatures(t *testing.T) {
	hybrid := EngineFeaturesOf(Features{
		Attention:    AttentionClass{SlidingWindow: 1024},
		Quantization: QuantizationClass{Bits: 6},
	})
	if !hybrid.ModelContextWindow || !hybrid.FixedSlidingCache || !hybrid.FixedSlidingCacheBound || !hybrid.NativeQ6BitstreamMatVec {
		t.Fatalf("EngineFeaturesOf(hybrid q6) = %+v, want model-context, bounded fixed-sliding cache, and q6 bitstream selection", hybrid)
	}
	if hybrid.NativeFixedSlidingAttention || hybrid.GenerationStream {
		t.Fatalf("EngineFeaturesOf(hybrid) = %+v, want config-only features before linked runtime is known", hybrid)
	}
	if hybrid.AsyncDecodePrefetch {
		t.Fatalf("EngineFeaturesOf(hybrid) = %+v, want linked-only prefetch disabled before runtime is known", hybrid)
	}

	dense := EngineFeaturesOf(Features{Quantization: QuantizationClass{Bits: 4}})
	if !dense.ModelContextWindow || dense.FixedSlidingCache || dense.FixedSlidingCacheBound || dense.NativeQ6BitstreamMatVec {
		t.Fatalf("EngineFeaturesOf(dense q4) = %+v, want model-context without fixed-sliding cache or q6 bitstream selection", dense)
	}
}

func TestLinkedGenerationEngineFeatures_Good_EnablesROCmFastPaths(t *testing.T) {
	hybrid := LinkedGenerationEngineFeatures(EngineFeaturesOf(Features{
		Attention:    AttentionClass{SlidingWindow: 1024},
		Quantization: QuantizationClass{Bits: 6},
	}))
	if !hybrid.DirectGreedyToken ||
		!hybrid.NativeMLPMatVec ||
		!hybrid.NativeLinearMatVec ||
		!hybrid.NativeQ6BitstreamMatVec ||
		!hybrid.NativeAttentionOMatVec ||
		!hybrid.NativeFixedSlidingAttention ||
		!hybrid.GenerationStream ||
		!hybrid.AsyncDecodePrefetch ||
		!hybrid.FixedSlidingCache ||
		!hybrid.FixedSlidingCacheBound {
		t.Fatalf("LinkedGenerationEngineFeatures(hybrid) = %+v, want linked ROCm runtime fast paths", hybrid)
	}
	if hybrid.CompiledLayerDecode || hybrid.PipelinedDecode {
		t.Fatalf("LinkedGenerationEngineFeatures(hybrid) = %+v, compiled/pipelined decode should stay false until implemented", hybrid)
	}

	dense := LinkedGenerationEngineFeatures(EngineFeaturesOf(Features{Quantization: QuantizationClass{Bits: 4}}))
	if !dense.DirectGreedyToken ||
		!dense.NativeMLPMatVec ||
		!dense.NativeLinearMatVec ||
		!dense.NativeAttentionOMatVec ||
		!dense.GenerationStream ||
		!dense.AsyncDecodePrefetch ||
		dense.NativeQ6BitstreamMatVec ||
		dense.NativeFixedSlidingAttention ||
		dense.FixedSlidingCache ||
		dense.PipelinedDecode {
		t.Fatalf("LinkedGenerationEngineFeatures(dense q4) = %+v, want dense linked paths without q6/fixed-sliding extras", dense)
	}
}

func TestEngineFeaturesOfIdentity_Good_UsesDeclaredAttention(t *testing.T) {
	features := EngineFeaturesOfIdentity(inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels: map[string]string{
			"gemma4_sliding_window": "1024",
		},
	})
	if !features.FixedSlidingCache || !features.FixedSlidingCacheBound {
		t.Fatalf("EngineFeaturesOfIdentity(sliding labels) = %+v, want fixed-sliding cache selection", features)
	}
}

func TestNeedsThoughtChannelSuppressorForIdentity_Good_UsesAttentionHeadBoundary(t *testing.T) {
	large, ok := NeedsThoughtChannelSuppressorForIdentity(inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels: map[string]string{
			"attention_heads": "16",
		},
	})
	if !ok || !large {
		t.Fatalf("NeedsThoughtChannelSuppressorForIdentity(heads=16) = %t,%t, want true,true", large, ok)
	}

	small, ok := NeedsThoughtChannelSuppressorForIdentity(inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels: map[string]string{
			"attention_heads": "8",
		},
	})
	if !ok || small {
		t.Fatalf("NeedsThoughtChannelSuppressorForIdentity(heads=8) = %t,%t, want false,true", small, ok)
	}

	missing, ok := NeedsThoughtChannelSuppressorForIdentity(inference.ModelIdentity{Architecture: "gemma4_text"})
	if ok || missing {
		t.Fatalf("NeedsThoughtChannelSuppressorForIdentity(no heads) = %t,%t, want false,false", missing, ok)
	}
}

func TestSizeNeedsThoughtChannelSuppressor_Good_RetainsROCmMetadataFallback(t *testing.T) {
	for _, size := range []string{"12B", "26b-a4b", " 31b "} {
		if !SizeNeedsThoughtChannelSuppressor(size) {
			t.Fatalf("SizeNeedsThoughtChannelSuppressor(%q) = false, want true", size)
		}
	}
	for _, size := range []string{"E2B", "E4B", ""} {
		if SizeNeedsThoughtChannelSuppressor(size) {
			t.Fatalf("SizeNeedsThoughtChannelSuppressor(%q) = true, want false", size)
		}
	}
}

func TestApplyConfigFeatureLabels_Good_WritesConfigAliases(t *testing.T) {
	labels := ApplyConfigFeatureLabels(nil, Features{
		Mixture:     true,
		NumExperts:  16,
		TopKExperts: 2,
		Vision:      true,
		Quantization: QuantizationClass{
			Bits: 6,
			Mode: "q6",
		},
		Attention: AttentionClass{
			SlidingWindow:  512,
			SlidingPattern: 5,
			SharedKVLayers: 2,
		},
	})
	for key, want := range map[string]string{
		"sliding_window":                    "512",
		"gemma4_sliding_window":             "512",
		"sliding_window_pattern":            "5",
		"gemma4_sliding_window_pattern":     "5",
		"attention_kv_shared_layers":        "2",
		"gemma4_attention_kv_shared_layers": "2",
		"gemma4_enable_moe_block":           "true",
		"gemma4_num_experts":                "16",
		"gemma4_top_k_experts":              "2",
		"gemma4_multimodal":                 "true",
		"gemma4_vision":                     "true",
		"gemma4_quant_bits":                 "6",
		"gemma4_quant_mode":                 "q6",
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}
}

func TestApplyConfigLabels_Good_WritesFullConfigSurface(t *testing.T) {
	labels := ApplyConfigLabels(nil, TextConfig{
		EnableMoEBlock:       true,
		NumExperts:           16,
		TopKExperts:          2,
		Vision:               true,
		Audio:                true,
		SlidingWindow:        512,
		SlidingWindowPattern: 5,
		KVSharedLayers:       0,
		KVSharedLayersSet:    true,
		HiddenSizePerLayer:   256,
		VocabSizePerLayer:    262144,
		UseDoubleWideMLP:     true,
		MoEIntermediateSize:  32,
		QuantBits:            6,
		QuantMode:            "q6",
	})
	for key, want := range map[string]string{
		"sliding_window":                     "512",
		"gemma4_sliding_window":              "512",
		"sliding_window_pattern":             "5",
		"gemma4_sliding_window_pattern":      "5",
		"attention_kv_shared_layers":         "0",
		"gemma4_attention_kv_shared_layers":  "0",
		"gemma4_enable_moe_block":            "true",
		"gemma4_num_experts":                 "16",
		"gemma4_top_k_experts":               "2",
		"gemma4_multimodal":                  "true",
		"gemma4_vision":                      "true",
		"gemma4_audio":                       "true",
		"gemma4_hidden_size_per_layer_input": "256",
		"gemma4_vocab_size_per_layer_input":  "262144",
		"gemma4_use_double_wide_mlp":         "true",
		"gemma4_moe_intermediate_size":       "32",
		"gemma4_quant_bits":                  "6",
		"gemma4_quant_mode":                  "q6",
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}
}

func TestApplyDeclaredFeatureLabels_Good_WritesEngineFacingLabels(t *testing.T) {
	labels := ApplyDeclaredFeatureLabels(map[string]string{}, Features{
		Mixture:     true,
		NumExperts:  16,
		TopKExperts: 2,
		Audio:       true,
		Attention: AttentionClass{
			SlidingWindow:  512,
			SlidingPattern: 5,
			SharedKVLayers: 2,
		},
		Quantization: QuantizationClass{
			Bits: 6,
			Mode: "q6",
		},
	})
	for key, want := range map[string]string{
		"gemma4_attention_sliding_window":   "512",
		"gemma4_attention_sliding_pattern":  "5",
		"gemma4_attention_kv_shared_layers": "2",
		"gemma4_mixture":                    "true",
		"gemma4_num_experts":                "16",
		"gemma4_top_k_experts":              "2",
		"gemma4_multimodal":                 "true",
		"gemma4_audio":                      "true",
		"gemma4_quant_bits":                 "6",
		"gemma4_quant_mode":                 "q6",
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}
	if labels["gemma4_vision"] != "" {
		t.Fatalf("labels[gemma4_vision] = %q, want unset for audio-only features", labels["gemma4_vision"])
	}
}
