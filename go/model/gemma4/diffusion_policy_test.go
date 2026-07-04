// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"math"
	"testing"
)

func TestDiffusionGeneratePolicyOf_Good_MatchesGoMLXFastLaneDefaults(t *testing.T) {
	policy := DiffusionGeneratePolicyOf(DiffusionPolicyConfig{VocabSize: 262144, ReferenceCanvasLength: 256})
	if policy.CanvasLength != DiffusionDefaultCanvasLength ||
		policy.ReferenceCanvasLength != DiffusionReferenceCanvasLength ||
		policy.MaxSteps != DiffusionDefaultMaxSteps ||
		policy.ReferenceMaxSteps != DiffusionReferenceMaxSteps ||
		policy.StabilityThreshold != DiffusionDefaultStabilitySteps ||
		policy.ConfidenceThreshold != DiffusionDefaultConfidence ||
		policy.MaxCanvases != 1 ||
		policy.Step.TextVocabSize != 262144 ||
		policy.Step.EntropyBound != DiffusionDefaultEntropyBound ||
		policy.Step.ReferenceEntropy != DiffusionReferenceEntropyBound ||
		policy.Step.MaxTemperature != DiffusionDefaultMaxTemperature ||
		policy.Step.MinTemperature != DiffusionDefaultMinTemperature ||
		policy.Step.Exponent != DiffusionDefaultTempExponent {
		t.Fatalf("DiffusionGeneratePolicyOf(defaults) = %+v, want go-mlx tuned fast-lane defaults", policy)
	}
}

func TestDiffusionGeneratePolicyOf_Good_OverlaysRuntimeConfig(t *testing.T) {
	policy := DiffusionGeneratePolicyOf(DiffusionPolicyConfig{
		CanvasLength:          128,
		ReferenceCanvasLength: 512,
		MaxSteps:              24,
		ReferenceMaxSteps:     96,
		StabilityThreshold:    3,
		ConfidenceThreshold:   0.01,
		MaxCanvases:           2,
		TextVocabSize:         1024,
		Seed:                  7,
		EntropyBound:          0.2,
		MaxTemperature:        0.9,
		MinTemperature:        0.2,
		TemperatureExponent:   2,
		StopTokens:            []int32{1, 2},
	})
	if policy.CanvasLength != 128 ||
		policy.ReferenceCanvasLength != 512 ||
		policy.MaxSteps != 24 ||
		policy.ReferenceMaxSteps != 96 ||
		policy.StabilityThreshold != 3 ||
		policy.ConfidenceThreshold != 0.01 ||
		policy.MaxCanvases != 2 ||
		policy.Step.TextVocabSize != 1024 ||
		policy.Step.Seed != 7 ||
		policy.Step.EntropyBound != 0.2 ||
		policy.Step.MaxTemperature != 0.9 ||
		policy.Step.MinTemperature != 0.2 ||
		policy.Step.Exponent != 2 ||
		len(policy.StopTokens) != 2 ||
		policy.StopTokens[0] != 1 ||
		policy.StopTokens[1] != 2 {
		t.Fatalf("DiffusionGeneratePolicyOf(overrides) = %+v, want explicit runtime config", policy)
	}
}

func TestDiffusionSchedule_Good_MatchesGoMLXSamplerMath(t *testing.T) {
	step := DefaultDiffusionStepPolicy(262144)
	for _, tc := range []struct {
		step int
		want float64
	}{
		{step: 0, want: 1.0},
		{step: 8, want: 0.5},
		{step: 16, want: 0.0},
	} {
		got := DiffusionNoiseAtStep(tc.step, 16)
		if math.Abs(got-tc.want) > 1e-9 {
			t.Fatalf("DiffusionNoiseAtStep(%d,16) = %v, want %v", tc.step, got, tc.want)
		}
	}
	if temp := DiffusionTemperature(1.0, step); math.Abs(temp-DiffusionDefaultMaxTemperature) > 1e-9 {
		t.Fatalf("DiffusionTemperature(noise=1) = %v, want %v", temp, DiffusionDefaultMaxTemperature)
	}
	if temp := DiffusionTemperature(0.0, step); math.Abs(temp-DiffusionDefaultMinTemperature) > 1e-9 {
		t.Fatalf("DiffusionTemperature(noise=0) = %v, want %v", temp, DiffusionDefaultMinTemperature)
	}
	if temp := DiffusionTemperature(0.25, DiffusionStepPolicy{MaxTemperature: 0.0, MinTemperature: 0.0, Exponent: 0.0}); temp <= 0 {
		t.Fatalf("DiffusionTemperature resolved non-positive temp %v", temp)
	}
}

func TestDiffusionSeeds_Good_MatchGoMLXCanvasDerivation(t *testing.T) {
	const base uint64 = 11
	if got := DiffusionInitialCanvasSeed(base, 0); got != base^(uint64(1)<<32) {
		t.Fatalf("DiffusionInitialCanvasSeed(base,0) = %d, want base xor 1<<32", got)
	}
	if got := DiffusionInitialCanvasSeed(base, 1); got != base^(uint64(2)<<32) {
		t.Fatalf("DiffusionInitialCanvasSeed(base,1) = %d, want base xor 2<<32", got)
	}
	if got := DiffusionCanvasStepSeed(base, 0); got != base {
		t.Fatalf("DiffusionCanvasStepSeed(base,0) = %d, want base", got)
	}
	if got := DiffusionCanvasStepSeed(base, 1); got != base+diffusionCanvasStepSeedIncrement {
		t.Fatalf("DiffusionCanvasStepSeed(base,1) = %d, want incremented seed", got)
	}
}

func TestDiffusionConverged_Good_RequiresStableRunAndConfidence(t *testing.T) {
	policy := DiffusionGeneratePolicyOf(DiffusionPolicyConfig{StabilityThreshold: 2, ConfidenceThreshold: 0.01})
	if DiffusionConverged(1, 0.001, policy) {
		t.Fatal("DiffusionConverged returned true before stability threshold")
	}
	if DiffusionConverged(2, 0.01, policy) {
		t.Fatal("DiffusionConverged returned true at the confidence threshold; go-mlx requires below it")
	}
	if !DiffusionConverged(2, 0.009, policy) {
		t.Fatal("DiffusionConverged returned false for stable low-entropy canvas")
	}
}

func TestApplyDiffusionPolicyLabels_Good_EmitsGemma4PolicyContract(t *testing.T) {
	labels := ApplyDiffusionPolicyLabels(nil, DiffusionGeneratePolicyOf(DiffusionPolicyConfig{
		ReferenceCanvasLength: 256,
		VocabSize:             262144,
	}))
	for key, want := range map[string]string{
		"diffusion_default_canvas_length":          "64",
		"diffusion_reference_canvas_length":        "256",
		"diffusion_default_max_steps":              "16",
		"diffusion_reference_max_steps":            "48",
		"diffusion_stability_threshold":            "1",
		"diffusion_confidence_threshold":           "0.005",
		"diffusion_entropy_bound":                  "0.3",
		"diffusion_reference_entropy_bound":        "0.1",
		"diffusion_max_temperature":                "0.8",
		"diffusion_min_temperature":                "0.4",
		"diffusion_temperature_exponent":           "1",
		"diffusion_text_vocab_size":                "262144",
		"gemma4_diffusion_default_canvas_length":   "64",
		"gemma4_diffusion_reference_canvas_length": "256",
		"gemma4_diffusion_entropy_bound":           "0.3",
		"gemma4_diffusion_reference_entropy_bound": "0.1",
		"gemma4_diffusion_temperature_exponent":    "1",
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}
}
