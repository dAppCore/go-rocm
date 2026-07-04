// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"slices"
	"testing"
)

func TestWeightPolicy_Good_FamilyOwnsCanonicalizationEntryPoint(t *testing.T) {
	weight, ok := CanonicalWeightName("gemma4_text", "language_model.model.layers.0.self_attn.q_proj.weight")
	if !ok || weight != "model.layers.0.self_attn.q_proj.weight" {
		t.Fatalf("CanonicalWeightName = %q ok=%v, want model.layers.0.self_attn.q_proj.weight true", weight, ok)
	}

	if skipped, ok := CanonicalWeightName("gemma4_text", "vision_tower.encoder.weight"); ok || skipped != "" {
		t.Fatalf("CanonicalWeightName vision = %q ok=%v, want skipped", skipped, ok)
	}
	if skipped, ok := CanonicalWeightName("gemma4_text", "model.layers.0.self_attn.rotary_emb.inv_freq"); ok || skipped != "" {
		t.Fatalf("CanonicalWeightName rotary = %q ok=%v, want skipped", skipped, ok)
	}

	unknown := "model.layers.0.self_attn.q_proj.weight"
	if got, ok := CanonicalWeightName("qwen3", unknown); !ok || got != unknown {
		t.Fatalf("CanonicalWeightName unknown = %q ok=%v, want passthrough", got, ok)
	}
}

func TestWeightPolicy_Good_WrapperHelpers(t *testing.T) {
	trimmed, ok := TrimWeightWrapperPrefix("gemma4_text", "model.language_model.model.layers.0.self_attn.q_proj.weight")
	if !ok || trimmed != "layers.0.self_attn.q_proj.weight" {
		t.Fatalf("TrimWeightWrapperPrefix = %q ok=%v, want layers.0.self_attn.q_proj.weight true", trimmed, ok)
	}

	unwrapped := UnwrapWeightName("model.language_model.model.layers.0.self_attn.q_proj.weight")
	if unwrapped != "layers.0.self_attn.q_proj.weight" {
		t.Fatalf("UnwrapWeightName = %q, want layers.0.self_attn.q_proj.weight", unwrapped)
	}

	one, ok := TrimOneWeightWrapper("model.language_model.model.layers.0.self_attn.q_proj.weight")
	if !ok || one != "layers.0.self_attn.q_proj.weight" {
		t.Fatalf("TrimOneWeightWrapper = %q ok=%v, want layers.0.self_attn.q_proj.weight true", one, ok)
	}
}

func TestWeightPolicy_Good_WrapperPrefixesAreCopySafe(t *testing.T) {
	prefixes := WeightWrapperPrefixes()
	if !slices.Contains(prefixes, "model.language_model.model.") {
		t.Fatalf("WeightWrapperPrefixes = %v, want model.language_model.model. prefix", prefixes)
	}
	prefixes[0] = "mutated"

	next := WeightWrapperPrefixes()
	if slices.Contains(next, "mutated") || !slices.Contains(next, "model.language_model.model.") {
		t.Fatalf("WeightWrapperPrefixes returned mutable backing data: %v", next)
	}
}
