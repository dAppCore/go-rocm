// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "testing"

func TestRoPEPolicyOf_Good_MergesGemma4Defaults(t *testing.T) {
	policy := RoPEPolicyOf(TextConfig{GlobalPartialRotaryFactor: 0.25})

	full := policy.Parameters[LayerTypeFullAttention]
	assertFloatEqual(t, full.PartialRotaryFactor, 0.25, "full.PartialRotaryFactor")
	assertFloatEqual(t, full.RopeTheta, 1000000, "full.RopeTheta")
	assertStringEqual(t, full.RopeType, "proportional", "full.RopeType")
	assertFloatEqual(t, full.Factor, 1, "full.Factor")

	sliding := policy.Parameters[LayerTypeSlidingAttention]
	assertFloatEqual(t, sliding.PartialRotaryFactor, 1, "sliding.PartialRotaryFactor")
	assertFloatEqual(t, sliding.RopeTheta, 10000, "sliding.RopeTheta")
	assertStringEqual(t, sliding.RopeType, "default", "sliding.RopeType")
	assertFloatEqual(t, sliding.Factor, 1, "sliding.Factor")
}

func TestRoPEPolicyOf_Good_InferGlobalPartialFromFullAttention(t *testing.T) {
	policy := RoPEPolicyOf(TextConfig{
		RoPEParameters: map[string]RoPEParameters{
			LayerTypeFullAttention: {PartialRotaryFactor: 0.125},
		},
	})

	full := policy.Parameters[LayerTypeFullAttention]
	assertFloatEqual(t, full.PartialRotaryFactor, 0.125, "full.PartialRotaryFactor")
	assertFloatEqual(t, full.RopeTheta, 1000000, "full.RopeTheta")
	assertStringEqual(t, full.RopeType, "proportional", "full.RopeType")
	assertFloatEqual(t, full.Factor, 1, "full.Factor")
}

func TestOverlayRoPEParameters_Good_UsesNonZeroFields(t *testing.T) {
	base := map[string]RoPEParameters{
		LayerTypeFullAttention: {
			PartialRotaryFactor: 0.25,
			RopeTheta:           1000000,
			RopeType:            "proportional",
			Factor:              1,
		},
	}
	merged := OverlayRoPEParameters(base, map[string]RoPEParameters{
		LayerTypeFullAttention: {
			RopeTheta: 2000000,
			Factor:    2,
		},
	})

	full := merged[LayerTypeFullAttention]
	assertFloatEqual(t, full.PartialRotaryFactor, 0.25, "full.PartialRotaryFactor")
	assertFloatEqual(t, full.RopeTheta, 2000000, "full.RopeTheta")
	assertStringEqual(t, full.RopeType, "proportional", "full.RopeType")
	assertFloatEqual(t, full.Factor, 2, "full.Factor")
}

func TestMergeRoPEParameters_Good_KeepsAdditionalAttentionClass(t *testing.T) {
	merged := MergeRoPEParameters(DefaultRoPEParameters(0.25), map[string]RoPEParameters{
		"linear_attention": {
			RopeTheta: 1234,
		},
	})

	linear := merged["linear_attention"]
	assertFloatEqual(t, linear.RopeTheta, 1234, "linear.RopeTheta")
	assertFloatEqual(t, linear.Factor, 1, "linear.Factor")
}

func TestApplyRoPEPolicyLabels_Good_WritesROCmAndGemma4Aliases(t *testing.T) {
	labels := ApplyConfigLabels(nil, TextConfig{
		RoPEParameters: map[string]RoPEParameters{
			LayerTypeFullAttention: {
				PartialRotaryFactor: 0.125,
				RopeTheta:           1000000,
				RopeType:            "proportional",
				Factor:              2,
			},
		},
	})

	for key, want := range map[string]string{
		"attention_rope_full_theta":                           "1e+06",
		"attention_rope_full_partial_rotary_factor":           "0.125",
		"attention_rope_full_type":                            "proportional",
		"attention_rope_full_factor":                          "2",
		"gemma4_attention_rope_full_theta":                    "1e+06",
		"gemma4_attention_rope_full_partial_rotary_factor":    "0.125",
		"attention_rope_sliding_theta":                        "10000",
		"attention_rope_sliding_partial_rotary_factor":        "1",
		"attention_rope_sliding_type":                         "default",
		"attention_rope_sliding_factor":                       "1",
		"gemma4_attention_rope_sliding_partial_rotary_factor": "1",
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}
}

func assertFloatEqual(t *testing.T, got, want float64, name string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %v, want %v", name, got, want)
	}
}

func assertStringEqual(t *testing.T, got, want, name string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %q, want %q", name, got, want)
	}
}
