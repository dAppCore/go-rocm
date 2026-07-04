// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"slices"
	"testing"
)

func TestStructurePlanOf_Good_DerivesReactiveLoadPlan(t *testing.T) {
	plan := StructurePlanOf(TextConfig{
		NumLayers:            4,
		LayerTypes:           []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"},
		AttentionKEqV:        true,
		EnableMoEBlock:       true,
		NumExperts:           128,
		TopKExperts:          8,
		KVSharedLayers:       2,
		HiddenSizePerLayer:   256,
		VocabSizePerLayer:    262144,
		UseDoubleWideMLP:     true,
		MoEIntermediateSize:  512,
		SlidingWindow:        1024,
		SlidingWindowPattern: 2,
	})
	if plan.LayerCount != 4 ||
		!slices.Equal(plan.LayerTypes, []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}) ||
		!plan.AttentionKEqV ||
		!plan.AttentionKEqVDeclared ||
		!plan.HasPerLayerInputs() ||
		!plan.UseDoubleWideMLP ||
		!plan.SharedKVEnabled() ||
		!plan.HasMoERouter() ||
		!plan.FusedExpertGateUpEligible {
		t.Fatalf("StructurePlanOf(config) = %+v, want reactive Gemma4 structure", plan)
	}
}

func TestStructurePlanOf_Good_ExpertCountsDoNotImplyMoERouter(t *testing.T) {
	plan := StructurePlanOf(TextConfig{
		NumExperts:  128,
		TopKExperts: 8,
	})
	if plan.HasMoERouter() || plan.FusedExpertGateUpEligible {
		t.Fatalf("StructurePlanOf(expert counts without MoE) = %+v, want dense plan", plan)
	}

	fromLabels := StructurePlanOfLabels(map[string]string{
		"gemma4_num_experts":      "128",
		"gemma4_top_k_experts":    "8",
		"moe_intermediate_size":   "512",
		"gemma4_per_layer_inputs": "true",
	})
	if fromLabels.HasMoERouter() || fromLabels.FusedExpertGateUpEligible {
		t.Fatalf("StructurePlanOfLabels(expert counts without MoE label) = %+v, want dense plan", fromLabels)
	}
}

func TestStructurePlanOfLabels_Good_ReconstructsLoadedModelPlan(t *testing.T) {
	plan := StructurePlanOfLabels(map[string]string{
		"gemma4_attention_layer_types":         "sliding_attention,full_attention",
		"gemma4_attention_k_eq_v":              "false",
		"gemma4_hidden_size_per_layer_input":   "256",
		"gemma4_vocab_size_per_layer_input":    "262144",
		"gemma4_use_double_wide_mlp":           "true",
		"gemma4_attention_kv_shared_layers":    "2",
		"gemma4_enable_moe_block":              "true",
		"gemma4_num_experts":                   "128",
		"gemma4_top_k_experts":                 "8",
		"gemma4_moe_intermediate_size":         "512",
		"gemma4_fused_expert_gate_up_eligible": "true",
	})
	if plan.LayerCount != 2 ||
		!slices.Equal(plan.LayerTypes, []string{"sliding_attention", "full_attention"}) ||
		plan.AttentionKEqV ||
		!plan.AttentionKEqVDeclared ||
		!plan.HasPerLayerInputs() ||
		!plan.UseDoubleWideMLP ||
		!plan.SharedKVEnabled() ||
		!plan.HasMoERouter() ||
		!plan.FusedExpertGateUpEligible {
		t.Fatalf("StructurePlanOfLabels(labels) = %+v, want label-derived loaded-model plan", plan)
	}
}

func TestApplyStructurePlanLabels_Good_WritesFactorySurface(t *testing.T) {
	labels := ApplyStructurePlanLabels(nil, StructurePlan{
		LayerCount:                4,
		LayerTypes:                []string{"sliding_attention", "full_attention"},
		AttentionKEqV:             false,
		AttentionKEqVDeclared:     true,
		PerLayerInputs:            true,
		HiddenSizePerLayerInput:   256,
		VocabSizePerLayerInput:    262144,
		UseDoubleWideMLP:          true,
		UsesSharedKV:              true,
		SharedKVLayers:            2,
		MoERouter:                 true,
		NumExperts:                128,
		TopKExperts:               8,
		MoEIntermediateSize:       512,
		FusedExpertGateUpEligible: true,
	})
	for key, want := range map[string]string{
		"gemma4_structure_plan_reactive":       "true",
		"num_hidden_layers":                    "4",
		"gemma4_num_hidden_layers":             "4",
		"layer_types":                          "sliding_attention,full_attention",
		"gemma4_layer_types":                   "sliding_attention,full_attention",
		"attention_k_eq_v":                     "false",
		"gemma4_attention_k_eq_v":              "false",
		"per_layer_inputs":                     "true",
		"gemma4_per_layer_inputs":              "true",
		"hidden_size_per_layer_input":          "256",
		"gemma4_hidden_size_per_layer_input":   "256",
		"vocab_size_per_layer_input":           "262144",
		"gemma4_vocab_size_per_layer_input":    "262144",
		"use_double_wide_mlp":                  "true",
		"gemma4_use_double_wide_mlp":           "true",
		"attention_shared_kv":                  "true",
		"gemma4_shared_kv":                     "true",
		"attention_kv_shared_layers":           "2",
		"gemma4_attention_kv_shared_layers":    "2",
		"gemma4_moe_router":                    "true",
		"gemma4_enable_moe_block":              "true",
		"num_experts":                          "128",
		"gemma4_num_experts":                   "128",
		"top_k_experts":                        "8",
		"gemma4_top_k_experts":                 "8",
		"moe_intermediate_size":                "512",
		"gemma4_moe_intermediate_size":         "512",
		"gemma4_fused_expert_gate_up_eligible": "true",
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}
}

func TestFeaturesOfLabels_Good_IncludesStructurePlan(t *testing.T) {
	features := FeaturesOfLabels(map[string]string{
		"gemma4_attention_layer_count":       "2",
		"gemma4_attention_layer_types":       "sliding_attention,full_attention",
		"gemma4_hidden_size_per_layer_input": "256",
		"gemma4_enable_moe_block":            "true",
		"gemma4_num_experts":                 "128",
		"gemma4_top_k_experts":               "8",
	})
	if features.Structure.LayerCount != 2 ||
		!features.Structure.HasPerLayerInputs() ||
		!features.Structure.HasMoERouter() ||
		!slices.Equal(features.Structure.LayerTypes, []string{"sliding_attention", "full_attention"}) {
		t.Fatalf("FeaturesOfLabels structure = %+v, want reactive structure plan", features.Structure)
	}
}
