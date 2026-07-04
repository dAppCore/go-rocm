// SPDX-Licence-Identifier: EUPL-1.2

package profile_test

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	rocmprofile "dappco.re/go/rocm/profile"
)

func TestGemma4ArchitectureSettings_Good_ProfileOwnsModelSettings(t *testing.T) {
	settings, ok := rocmprofile.Gemma4ArchitectureSettingsForArchitecture("Gemma4ForConditionalGeneration")
	if !ok {
		t.Fatal("Gemma4ArchitectureSettingsForArchitecture(Gemma4ForConditionalGeneration) ok = false")
	}
	if settings.ID != "gemma4" ||
		settings.TextTowerID != "gemma4_text" ||
		settings.RuntimeStatus != inference.FeatureRuntimeNative ||
		settings.ParserID != "gemma" ||
		settings.ToolParserID != "gemma" ||
		settings.TokenizerKind != "GemmaTokenizer" ||
		settings.ChatTemplate != "gemma4_hf_turn" ||
		settings.GenerationRole != "model" ||
		!settings.DefaultThinking ||
		!settings.RequiresChatTemplate ||
		!settings.NativeRuntime ||
		!settings.Generation ||
		!settings.Chat ||
		settings.Embeddings ||
		settings.Rerank ||
		settings.MoE ||
		settings.AttachedOnly {
		t.Fatalf("settings = %+v, want Gemma4 target architecture profile settings", settings)
	}
	if !slices.Contains(settings.WeightWrapperPrefixes, "model.language_model.model.") ||
		!slices.Contains(settings.WeightSkipPrefixes, "vision_tower") ||
		!slices.Contains(settings.WeightSkipSubstrings, "self_attn.rotary_emb") ||
		!slices.Contains(settings.WeightModelPrefixes, "layers.") {
		t.Fatalf("settings weight policy = %+v, want Gemma4 weight policy", settings)
	}
	if !slices.Contains(settings.QuantizationHints, "q6") ||
		!slices.Contains(settings.CacheHints, "k-q8-v-q4") ||
		!slices.Contains(settings.Aliases, "Gemma4ForConditionalGeneration") {
		t.Fatalf("settings hints = %+v cache=%+v aliases=%+v, want profile registry metadata", settings.QuantizationHints, settings.CacheHints, settings.Aliases)
	}
	if !slices.Equal(settings.LoRADefaultTargets, []string{"q_proj", "v_proj", "o_proj"}) ||
		!slices.Contains(settings.LoRATargets, "router.proj") ||
		!slices.Contains(settings.LoRAExtendedTargets, "per_layer_projection") ||
		settings.LoRATargetPaths["gate_proj"] != "mlp.gate_proj" {
		t.Fatalf("settings LoRA policy = targets=%+v defaults=%+v extended=%+v paths=%+v, want registry-owned Gemma4 target policy", settings.LoRATargets, settings.LoRADefaultTargets, settings.LoRAExtendedTargets, settings.LoRATargetPaths)
	}

	textSettings, ok := rocmprofile.Gemma4ArchitectureSettingsForArchitecture("gemma4_unified_text")
	if !ok || textSettings.ID != "gemma4_text" || textSettings.TextTowerID != "" {
		t.Fatalf("text settings = %+v ok=%v, want unified text alias as Gemma4 text tower", textSettings, ok)
	}
	textSettings.QuantizationHints[0] = "mutated"
	textSettings, _ = rocmprofile.Gemma4ArchitectureSettingsForArchitecture("gemma4_text")
	if textSettings.QuantizationHints[0] == "mutated" {
		t.Fatalf("settings quantization hints were not defensively copied: %+v", textSettings.QuantizationHints)
	}
}

func TestGemma4ArchitectureSettings_Good_DefaultProfilesAreCopySafe(t *testing.T) {
	profiles := rocmprofile.DefaultGemma4ArchitectureSettings()
	if len(profiles) != 4 {
		t.Fatalf("DefaultGemma4ArchitectureSettings len = %d, want 4: %+v", len(profiles), profiles)
	}
	seen := map[string]rocmprofile.Gemma4ArchitectureSettings{}
	for _, profile := range profiles {
		seen[profile.ID] = profile
	}
	for _, id := range []string{"gemma4", "gemma4_text", "gemma4_unified", "gemma4_assistant"} {
		if _, ok := seen[id]; !ok {
			t.Fatalf("DefaultGemma4ArchitectureSettings missing %q: %+v", id, profiles)
		}
	}
	if seen["gemma4"].TextTowerID != "gemma4_text" ||
		seen["gemma4_text"].TextTowerID != "" ||
		!seen["gemma4_assistant"].AttachedOnly ||
		seen["gemma4_assistant"].Generation ||
		!slices.Contains(seen["gemma4"].QuantizationHints, "q6") ||
		!slices.Contains(seen["gemma4_assistant"].CacheHints, "attached-drafter") {
		t.Fatalf("profiles = %+v, want registry-ready Gemma4 target and assistant profiles", profiles)
	}
	profiles[0].QuantizationHints[0] = "mutated"
	next := rocmprofile.DefaultGemma4ArchitectureSettings()
	if next[0].QuantizationHints[0] == "mutated" {
		t.Fatalf("DefaultGemma4ArchitectureSettings returned mutable backing slices: %+v", next[0])
	}
}

func TestGemma4LoRATargetPolicy_Good_ProfileOwnsModelPolicy(t *testing.T) {
	wantDefault := []string{"q_proj", "v_proj", "o_proj"}
	for _, architecture := range []string{
		"gemma4",
		"gemma4_text",
		"gemma4_unified",
		"Gemma4ForConditionalGeneration",
		"Gemma4UnifiedForConditionalGeneration",
	} {
		t.Run(architecture, func(t *testing.T) {
			policy, ok := rocmprofile.Gemma4LoRATargetPolicyForArchitecture(architecture)
			if !ok {
				t.Fatalf("Gemma4LoRATargetPolicyForArchitecture(%q) ok = false", architecture)
			}
			generic, ok := rocmprofile.LoRATargetPolicyForArchitecture(architecture)
			if !ok {
				t.Fatalf("LoRATargetPolicyForArchitecture(%q) ok = false", architecture)
			}
			if !slices.Equal(policy.DefaultTargets, wantDefault) {
				t.Fatalf("DefaultTargets = %v, want %v", policy.DefaultTargets, wantDefault)
			}
			if !slices.Equal(generic.DefaultTargets, wantDefault) {
				t.Fatalf("generic DefaultTargets = %v, want %v", generic.DefaultTargets, wantDefault)
			}
			cases := []struct {
				target   string
				wantPath string
				wantSafe bool
			}{
				{"q_proj", "self_attn.q_proj", true},
				{"self_attn.q_proj", "self_attn.q_proj", true},
				{"gate_proj", "mlp.gate_proj", true},
				{"mlp.up_proj", "mlp.up_proj", true},
				{"router.proj", "router.proj", false},
				{"per_layer_input_gate", "per_layer_input_gate", false},
			}
			for _, tc := range cases {
				path, ok := rocmprofile.Gemma4LoRATargetPath(architecture, tc.target)
				if !ok || path != tc.wantPath {
					t.Fatalf("Gemma4LoRATargetPath(%q, %q) = %q, %v; want %q, true", architecture, tc.target, path, ok, tc.wantPath)
				}
				canonical, ok := rocmprofile.Gemma4LoRACanonicalTarget(architecture, "model.layers.0."+tc.target)
				if !ok || canonical != "model.layers.0."+tc.wantPath {
					t.Fatalf("Gemma4LoRACanonicalTarget(%q, %q) = %q, %v; want %q, true", architecture, tc.target, canonical, ok, "model.layers.0."+tc.wantPath)
				}
				if safe := rocmprofile.Gemma4LoRASafeTarget(architecture, tc.target); safe != tc.wantSafe {
					t.Fatalf("Gemma4LoRASafeTarget(%q, %q) = %v, want %v", architecture, tc.target, safe, tc.wantSafe)
				}
				if safe := rocmprofile.SafeLoRATarget(architecture, tc.target); safe != tc.wantSafe {
					t.Fatalf("SafeLoRATarget(%q, %q) = %v, want %v", architecture, tc.target, safe, tc.wantSafe)
				}
			}
			if targets := rocmprofile.DefaultLoRATargets(architecture); !slices.Equal(targets, wantDefault) {
				t.Fatalf("DefaultLoRATargets(%q) = %v, want %v", architecture, targets, wantDefault)
			}
			if rocmprofile.SafeLoRATarget(architecture, "router.proj") || !rocmprofile.LoRAExtendedTarget(architecture, "router.proj") {
				t.Fatalf("generic LoRA safety helpers did not preserve extended target policy for %q", architecture)
			}
		})
	}

	policy, ok := rocmprofile.Gemma4LoRATargetPolicyForArchitecture("gemma4_text")
	if !ok {
		t.Fatal("Gemma4LoRATargetPolicyForArchitecture(gemma4_text) ok = false")
	}
	policy.DefaultTargets[0] = "mutated"
	policy.TargetPaths["q_proj"] = "mutated"
	defaults := rocmprofile.DefaultLoRATargets("gemma4_text")
	defaults[0] = "mutated"
	profile, _ := rocmprofile.LookupArchitectureProfile("gemma4_text")
	profile.LoRATargetPaths["q_proj"] = "mutated"
	policy, _ = rocmprofile.Gemma4LoRATargetPolicyForArchitecture("gemma4_text")
	if policy.DefaultTargets[0] != "q_proj" || policy.TargetPaths["q_proj"] != "self_attn.q_proj" {
		t.Fatalf("policy was mutated through returned copy: %+v", policy)
	}
	if path, ok := rocmprofile.LoRATargetPath("gemma4_text", "q_proj"); !ok || path != "self_attn.q_proj" {
		t.Fatalf("LoRATargetPath after mutation = %q ok=%v, want self_attn.q_proj true", path, ok)
	}
	if targets := rocmprofile.Gemma4LoRADefaultTargets("gemma4_assistant"); targets != nil {
		t.Fatalf("Gemma4LoRADefaultTargets(gemma4_assistant) = %v, want nil", targets)
	}
	if targets := rocmprofile.DefaultLoRATargets("gemma4_assistant"); targets != nil {
		t.Fatalf("DefaultLoRATargets(gemma4_assistant) = %v, want nil", targets)
	}
}

func TestGemma4WeightPolicy_Good_ProfileOwnsCanonicalization(t *testing.T) {
	weight, ok := rocmprofile.CanonicalWeightName("gemma4_text", "language_model.model.layers.0.self_attn.q_proj.weight")
	if !ok || weight != "model.layers.0.self_attn.q_proj.weight" {
		t.Fatalf("CanonicalWeightName = %q ok=%v, want model.layers.0.self_attn.q_proj.weight true", weight, ok)
	}
	if skipped, ok := rocmprofile.CanonicalWeightName("gemma4_text", "vision_tower.encoder.weight"); ok || skipped != "" {
		t.Fatalf("CanonicalWeightName vision = %q ok=%v, want skipped", skipped, ok)
	}
	trimmed, ok := rocmprofile.TrimWeightWrapperPrefix("gemma4_text", "model.language_model.model.layers.0.self_attn.q_proj.weight")
	if !ok || trimmed != "layers.0.self_attn.q_proj.weight" {
		t.Fatalf("TrimWeightWrapperPrefix = %q ok=%v, want layers.0.self_attn.q_proj.weight true", trimmed, ok)
	}
	unknown := "model.layers.0.self_attn.q_proj.weight"
	if got, ok := rocmprofile.CanonicalWeightName("qwen3", unknown); !ok || got != unknown {
		t.Fatalf("CanonicalWeightName unknown = %q ok=%v, want passthrough", got, ok)
	}
}
