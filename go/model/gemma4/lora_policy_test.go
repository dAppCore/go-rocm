// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"slices"
	"testing"
)

func TestLoRATargetPolicyForArchitecture_Good_FamilyOwnedPolicy(t *testing.T) {
	policy, ok := LoRATargetPolicyForArchitecture("Gemma4ForConditionalGeneration")
	if !ok {
		t.Fatal("LoRATargetPolicyForArchitecture(Gemma4ForConditionalGeneration) ok = false")
	}
	if !slices.Equal(policy.DefaultTargets, []string{"q_proj", "v_proj", "o_proj"}) {
		t.Fatalf("DefaultTargets = %v, want q/v/o", policy.DefaultTargets)
	}
	if !slices.Contains(policy.SafeTargets, "gate_proj") || !slices.Contains(policy.ExtendedTargets, "router.proj") {
		t.Fatalf("policy = %+v, want safe MLP targets and extended router target", policy)
	}
	if policy.TargetPaths["q_proj"] != "self_attn.q_proj" ||
		policy.TargetPaths["gate_proj"] != "mlp.gate_proj" ||
		policy.TargetPaths["router.proj"] != "router.proj" {
		t.Fatalf("TargetPaths = %+v, want Gemma4 target path map", policy.TargetPaths)
	}

	policy.DefaultTargets[0] = "mutated"
	policy.TargetPaths["q_proj"] = "mutated"
	next, ok := LoRATargetPolicyForArchitecture("gemma4_text")
	if !ok || next.DefaultTargets[0] != "q_proj" || next.TargetPaths["q_proj"] != "self_attn.q_proj" {
		t.Fatalf("policy returned mutable backing data: %+v ok=%v", next, ok)
	}
}

func TestLoRATargetPolicy_Good_TargetHelpers(t *testing.T) {
	if targets := LoRADefaultTargets("gemma4_text"); !slices.Equal(targets, []string{"q_proj", "v_proj", "o_proj"}) {
		t.Fatalf("LoRADefaultTargets = %v, want q/v/o", targets)
	}
	if path, ok := LoRATargetPath("gemma4_text", "q_proj"); !ok || path != "self_attn.q_proj" {
		t.Fatalf("LoRATargetPath(q_proj) = %q ok=%v, want self_attn.q_proj true", path, ok)
	}
	if !LoRASafeTarget("gemma4_text", "mlp.down_proj") || LoRASafeTarget("gemma4_text", "router.proj") {
		t.Fatal("LoRASafeTarget did not distinguish safe MLP target from extended router target")
	}
	if !LoRAExtendedTarget("gemma4_text", "router.proj") || LoRAExtendedTarget("gemma4_text", "q_proj") {
		t.Fatal("LoRAExtendedTarget did not distinguish extended router target from default q target")
	}
	canonical, ok := LoRACanonicalTarget("gemma4_text", "model.layers.0.q_proj")
	if !ok || canonical != "model.layers.0.self_attn.q_proj" {
		t.Fatalf("LoRACanonicalTarget = %q ok=%v, want model.layers.0.self_attn.q_proj true", canonical, ok)
	}
	if canonical, ok := LoRACanonicalTarget("gemma4_assistant", "model.layers.0.q_proj"); ok || canonical != "" {
		t.Fatalf("LoRACanonicalTarget(assistant) = %q ok=%v, want no standalone assistant targets", canonical, ok)
	}
}
