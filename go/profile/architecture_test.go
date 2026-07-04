// SPDX-Licence-Identifier: EUPL-1.2

package profile_test

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	rocmprofile "dappco.re/go/rocm/profile"
)

func TestArchitectureProfile_Good_BroadRegistryContract(t *testing.T) {
	profiles := rocmprofile.BuiltinArchitectureProfiles()
	if len(profiles) < 24 {
		t.Fatalf("BuiltinArchitectureProfiles len = %d, want broad reactive target list: %+v", len(profiles), profiles)
	}
	seen := map[string]rocmprofile.ArchitectureProfile{}
	for _, profile := range profiles {
		seen[profile.ID] = profile
	}
	for _, id := range []string{"gemma3_text", "gemma4_text", "qwen3_6", "qwen3_6_moe", "minimax_m2", "bert_rerank", "gpt-oss", "mamba2", "mla", "nsa", "rwkv7"} {
		if _, ok := seen[id]; !ok {
			t.Fatalf("BuiltinArchitectureProfiles missing %q: %+v", id, profiles)
		}
	}
	if seen["gemma4_text"].ChatTemplate != "gemma4_hf_turn" ||
		seen["gemma4_text"].TokenizerKind != "GemmaTokenizer" ||
		!seen["gemma4_text"].DefaultThinking ||
		seen["gemma4_text"].GenerationRole != "model" {
		t.Fatalf("gemma4_text profile = %+v, want Gemma4 specialized settings", seen["gemma4_text"])
	}
	if seen["qwen3_6_moe"].Family != "qwen" ||
		seen["qwen3_6_moe"].ParserID != "qwen" ||
		!seen["qwen3_6_moe"].MoE ||
		!seen["qwen3_6_moe"].NativeRuntime ||
		seen["qwen3_6_moe"].Generation ||
		seen["qwen3_6_moe"].Chat ||
		seen["qwen3_6_moe"].RequiresChatTemplate ||
		!slices.Contains(seen["qwen3_6_moe"].CacheHints, "k-q8-v-q4") {
		t.Fatalf("qwen3_6_moe profile = %+v, want staged MoE reactive metadata", seen["qwen3_6_moe"])
	}
	if !slices.Equal(seen["qwen3"].LoRADefaultTargets, []string{"q_proj", "v_proj"}) ||
		!slices.Contains(seen["qwen3"].LoRATargets, "gate_proj") ||
		seen["qwen3"].LoRATargetPaths["q_proj"] != "self_attn.q_proj" ||
		seen["qwen3"].LoRATargetPaths["gate_proj"] != "mlp.gate_proj" {
		t.Fatalf("qwen3 profile LoRA policy = targets=%+v defaults=%+v paths=%+v, want decoder adapter policy in profile registry", seen["qwen3"].LoRATargets, seen["qwen3"].LoRADefaultTargets, seen["qwen3"].LoRATargetPaths)
	}
	if seen["bert_rerank"].Family != "bert" ||
		!seen["bert_rerank"].Rerank ||
		seen["bert_rerank"].Chat ||
		seen["bert_rerank"].RequiresChatTemplate ||
		seen["bert_rerank"].ChatTemplate != "" {
		t.Fatalf("bert_rerank profile = %+v, want rerank non-chat metadata", seen["bert_rerank"])
	}
	if seen["composed"].Family != "composed" ||
		seen["composed"].Generation ||
		!slices.Equal(seen["composed"].CacheHints, []string{"default", "recurrent", "mla-latent"}) ||
		!slices.Equal(seen["composed"].LoRADefaultTargets, []string{"gate_proj", "up_proj", "down_proj"}) ||
		seen["composed"].LoRATargetPaths["up_proj"] != "mlp.up_proj" {
		t.Fatalf("composed profile = %+v, want sequence-mixer cache hints", seen["composed"])
	}
	if seen["mamba2"].NativeRuntime ||
		seen["mamba2"].RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		seen["mamba2"].Generation ||
		seen["mamba2"].Chat ||
		!slices.Equal(seen["mamba2"].CacheHints, []string{"default", "recurrent"}) {
		t.Fatalf("mamba2 profile = %+v, want metadata-only recurrent go-mlx parity profile", seen["mamba2"])
	}
	if seen["mla"].NativeRuntime ||
		!slices.Equal(seen["mla"].CacheHints, []string{"default", "mla-latent"}) {
		t.Fatalf("mla profile = %+v, want metadata-only latent-cache profile", seen["mla"])
	}
	if seen["nsa"].NativeRuntime ||
		!slices.Equal(seen["nsa"].CacheHints, []string{"default", "paged"}) {
		t.Fatalf("nsa profile = %+v, want metadata-only paged-cache profile", seen["nsa"])
	}

	profiles[0].Aliases = []string{"mutated"}
	next, ok := rocmprofile.LookupArchitectureProfile("bert")
	if !ok || slices.Contains(next.Aliases, "mutated") {
		t.Fatalf("LookupArchitectureProfile returned mutable aliases: %+v ok=%v", next, ok)
	}
	if _, ok := rocmprofile.LookupArchitectureProfile("unknown_family"); ok {
		t.Fatal("LookupArchitectureProfile(unknown_family) ok = true, want false")
	}
}

func TestArchitectureProfile_Good_NormalizeAndLookupAliases(t *testing.T) {
	cases := map[string]string{
		"BertForSequenceClassification":         "bert_rerank",
		"Qwen3_5MoeForConditionalGeneration":    "qwen3_6_moe",
		"Qwen3.6ForConditionalGeneration":       "qwen3_6",
		"MiniMaxM2ForCausalLM":                  "minimax_m2",
		"GPTOSSForCausalLM":                     "gpt-oss",
		"DiffusionGemmaForBlockDiffusion":       "diffusion_gemma",
		"Gemma3TextForCausalLM":                 "gemma3_text",
		"Gemma4UnifiedForConditionalGeneration": "gemma4_unified",
		"gemma4_unified_text":                   "gemma4_text",
		"Mamba2ForCausalLM":                     "mamba2",
		"DeltaNetForCausalLM":                   "deltanet",
		"NativeSparseAttentionForCausalLM":      "nsa",
		"RWKV7ForCausalLM":                      "rwkv7",
	}
	for input, want := range cases {
		if got := rocmprofile.ArchitectureID(input); got != want {
			t.Fatalf("ArchitectureID(%q) = %q, want %q", input, got, want)
		}
		profile, ok := rocmprofile.LookupArchitectureProfile(input)
		if !ok || profile.ID != want {
			t.Fatalf("LookupArchitectureProfile(%q) = %+v ok=%v, want id %q", input, profile, ok, want)
		}
	}
}

func TestArchitectureProfile_Good_RuntimeHelpers(t *testing.T) {
	if !rocmprofile.SupportedNativeArchitecture("Gemma4ForCausalLM") ||
		!rocmprofile.SupportedNativeArchitecture("qwen3_6_moe") ||
		rocmprofile.SupportedNativeArchitecture("mamba2") ||
		rocmprofile.SupportedNativeArchitecture("unknown_family") {
		t.Fatalf("SupportedNativeArchitecture mismatch")
	}
	if !rocmprofile.IsMoEArchitecture("qwen3_6_moe") ||
		!rocmprofile.IsMoEArchitecture("mixtral") ||
		rocmprofile.IsMoEArchitecture("qwen3") {
		t.Fatalf("IsMoEArchitecture mismatch")
	}
	if got := rocmprofile.ArchitectureProfileParser("deepseek_r1"); got != "deepseek-r1" {
		t.Fatalf("ArchitectureProfileParser(deepseek_r1) = %q, want deepseek-r1", got)
	}
	if got := rocmprofile.ArchitectureProfileChatTemplate("qwen3"); got != "qwen" {
		t.Fatalf("ArchitectureProfileChatTemplate(qwen3) = %q, want qwen", got)
	}
	if got := rocmprofile.ArchitectureProfileTokenizerKind("Qwen3ForCausalLM"); got != "Qwen2Tokenizer" {
		t.Fatalf("ArchitectureProfileTokenizerKind(qwen3) = %q, want Qwen2Tokenizer", got)
	}
	if got := rocmprofile.ArchitectureProfileTokenizerKind("BertModel"); got != "BertTokenizer" {
		t.Fatalf("ArchitectureProfileTokenizerKind(bert) = %q, want BertTokenizer", got)
	}
	if got := rocmprofile.ArchitectureProfileTokenizerKind("NativeSparseAttentionForCausalLM"); got != "tokenizer.json" {
		t.Fatalf("ArchitectureProfileTokenizerKind(nsa) = %q, want tokenizer.json metadata tokenizer kind", got)
	}
	if got := rocmprofile.ArchitectureProfileChat("qwen3_6_moe"); got {
		t.Fatalf("ArchitectureProfileChat(qwen3_6_moe) = true, want false for staged profile")
	}
	if got := rocmprofile.ArchitectureProfileLoRATargetPolicyName("Qwen3ForCausalLM"); got != "decoder" {
		t.Fatalf("ArchitectureProfileLoRATargetPolicyName(qwen3) = %q, want decoder", got)
	}
	if got := rocmprofile.ArchitectureProfileLoRATargetPolicyName("hybrid"); got != "composed_mlp" {
		t.Fatalf("ArchitectureProfileLoRATargetPolicyName(hybrid) = %q, want composed_mlp", got)
	}
	if defaults := rocmprofile.ArchitectureProfileLoRADefaultTargets("Qwen3ForCausalLM"); !slices.Equal(defaults, []string{"q_proj", "v_proj"}) {
		t.Fatalf("ArchitectureProfileLoRADefaultTargets(qwen3) = %v, want q/v", defaults)
	}
	if path := rocmprofile.ArchitectureProfileLoRATargetPaths("Qwen3ForCausalLM")["down_proj"]; path != "mlp.down_proj" {
		t.Fatalf("ArchitectureProfileLoRATargetPaths(qwen3)[down_proj] = %q, want mlp.down_proj", path)
	}
}

func TestArchitectureProfile_Good_GenericProfileHelpers(t *testing.T) {
	targetCases := []struct {
		architecture string
		wantTarget   bool
	}{
		{architecture: "gemma4", wantTarget: true},
		{architecture: "gemma4_text", wantTarget: true},
		{architecture: "gemma4_unified", wantTarget: true},
		{architecture: "gemma4_unified_text", wantTarget: true},
		{architecture: "Gemma4ForConditionalGeneration", wantTarget: true},
		{architecture: "Gemma4UnifiedForConditionalGeneration", wantTarget: true},
		{architecture: "Gemma4ForCausalLM", wantTarget: true},
		{architecture: "Gemma4AssistantForCausalLM"},
		{architecture: "gemma3"},
		{architecture: "qwen3"},
		{architecture: ""},
	}
	for _, tc := range targetCases {
		if got := rocmprofile.IsGemma4TargetArchitecture(tc.architecture); got != tc.wantTarget {
			t.Fatalf("IsGemma4TargetArchitecture(%q) = %v, want %v", tc.architecture, got, tc.wantTarget)
		}
	}

	if !rocmprofile.IsGemma4LargeVariant("Gemma4ForConditionalGeneration", 16) ||
		!rocmprofile.IsGemma4LargeVariant("gemma4_unified_text", 16) ||
		rocmprofile.IsGemma4LargeVariant("gemma4_text", 8) ||
		rocmprofile.IsGemma4LargeVariant("Gemma4AssistantForCausalLM", 16) ||
		rocmprofile.IsGemma4LargeVariant("qwen3", 16) {
		t.Fatalf("IsGemma4LargeVariant did not follow target architecture and head-count policy")
	}

	if !rocmprofile.DefaultThinkingEnabled("Gemma4ForConditionalGeneration") ||
		!rocmprofile.DefaultThinkingEnabled("gemma4_unified_text") ||
		rocmprofile.DefaultThinkingEnabled("Gemma4AssistantForCausalLM") ||
		rocmprofile.DefaultThinkingEnabled("qwen3") {
		t.Fatalf("DefaultThinkingEnabled did not follow profile registry defaults")
	}
	if !rocmprofile.AttachedOnlyArchitecture("Gemma4AssistantForCausalLM") ||
		rocmprofile.AttachedOnlyArchitecture("Gemma4ForConditionalGeneration") ||
		rocmprofile.AttachedOnlyArchitecture("qwen3") {
		t.Fatalf("AttachedOnlyArchitecture did not follow profile registry settings")
	}

	templateCases := map[string]string{
		"Gemma4ForConditionalGeneration": "gemma4_hf_turn",
		"gemma4_unified_text":            "gemma4_hf_turn",
		"Gemma4AssistantForCausalLM":     "",
		"Gemma3ForCausalLM":              "gemma",
		"qwen3_6_moe":                    "qwen",
		"llama3":                         "llama",
		"MiniMaxM2ForCausalLM":           "",
		"DeepseekV3ForCausalLM":          "",
		"unknown":                        "",
		"":                               "",
	}
	for architecture, want := range templateCases {
		if got := rocmprofile.ChatTemplateName(architecture); got != want {
			t.Fatalf("ChatTemplateName(%q) = %q, want %q", architecture, got, want)
		}
	}
}
