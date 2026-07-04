// SPDX-Licence-Identifier: EUPL-1.2

package profile

import (
	"slices"
	"testing"
)

func TestResolveArchitecture_Good_MatchesReactiveOrder(t *testing.T) {
	cases := []struct {
		name      string
		modelType string
		textTower string
		archs     []string
		want      string
		source    string
	}{
		{name: "qwen2.5 alias", modelType: "qwen2.5", archs: []string{"Qwen2.5ForCausalLM"}, want: "qwen2", source: "model_type"},
		{name: "qwen3.5 to 3.6", modelType: "qwen3_5", archs: []string{"Qwen3_5ForConditionalGeneration"}, want: "qwen3_6", source: "model_type"},
		{name: "qwen3.5 moe", modelType: "qwen3_5_moe", archs: []string{"Qwen3_5MoeForConditionalGeneration"}, want: "qwen3_6_moe", source: "model_type"},
		{name: "text tower fallback", textTower: "qwen3_5_text", archs: []string{"Qwen3_5ForConditionalGeneration"}, want: "qwen3_6", source: "text_config_model_type"},
		{name: "architecture fallback", archs: []string{"MiniMaxM2ForCausalLM"}, want: "minimax_m2", source: "architectures"},
		{name: "gemma4 multimodal text tower", modelType: "gemma4", textTower: "gemma4_text", archs: []string{"Gemma4ForConditionalGeneration"}, want: "gemma4_text", source: "model_type_text_tower"},
		{name: "gemma4 wrapper without tower", modelType: "gemma4", archs: []string{"Gemma4ForConditionalGeneration"}, want: "gemma4", source: "model_type"},
		{name: "gemma4 unified stays wrapper", modelType: "gemma4_unified", textTower: "gemma4_unified_text", archs: []string{"Gemma4UnifiedForConditionalGeneration"}, want: "gemma4_unified", source: "model_type"},
		{name: "gemma4 unified text alias", modelType: "gemma4_unified_text", archs: []string{"Gemma4TextForCausalLM"}, want: "gemma4_text", source: "model_type"},
		{name: "bert plain", modelType: "bert", archs: []string{"BertModel"}, want: "bert", source: "model_type"},
		{name: "bert rerank refined", modelType: "bert", archs: []string{"BertForSequenceClassification"}, want: "bert_rerank", source: "model_type_architecture_refinement"},
		{name: "empty"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resolution := ResolveArchitecture(tc.modelType, tc.textTower, tc.archs)
			if resolution.Architecture != tc.want || resolution.Source != tc.source {
				t.Fatalf("ResolveArchitecture(%q, %q, %v) = %+v, want architecture %q source %q", tc.modelType, tc.textTower, tc.archs, resolution, tc.want, tc.source)
			}
			if got := ResolveArchitectureID(tc.modelType, tc.textTower, tc.archs); got != tc.want {
				t.Fatalf("ResolveArchitectureID(%q, %q, %v) = %q, want %q", tc.modelType, tc.textTower, tc.archs, got, tc.want)
			}
		})
	}
}

func TestResolveArchitecture_Good_RegisteredProfilesDriveRefinements(t *testing.T) {
	restoreRegisteredArchitectureProfilesForTest(t)

	RegisterArchitectureProfile(ArchitectureProfile{
		ID:          "fake_multi",
		Family:      "fake",
		TextTowerID: "fake_text",
		Aliases:     []string{"FakeMultiForConditionalGeneration"},
	})
	RegisterArchitectureProfile(ArchitectureProfile{
		ID:      "fake_text",
		Family:  "fake",
		Aliases: []string{"FakeTextForCausalLM"},
	})
	RegisterArchitectureProfile(ArchitectureProfile{
		ID:      "fake_rerank",
		Family:  "fake",
		Rerank:  true,
		Aliases: []string{"FakeForSequenceClassification"},
	})

	resolution := ResolveArchitecture("FakeMultiForConditionalGeneration", "FakeTextForCausalLM", nil)
	if resolution.Architecture != "fake_text" ||
		resolution.Source != "model_type_text_tower" ||
		resolution.Profile.ID != "fake_text" {
		t.Fatalf("ResolveArchitecture registered text tower = %+v, want fake_text profile", resolution)
	}

	resolution = ResolveArchitecture("fake_multi", "", []string{"FakeForSequenceClassification"})
	if resolution.Architecture != "fake_rerank" ||
		resolution.Source != "model_type_architecture_refinement" ||
		resolution.Profile.ID != "fake_rerank" {
		t.Fatalf("ResolveArchitecture registered rerank = %+v, want fake_rerank profile", resolution)
	}
}

func TestResolveArchitecture_Good_CopySafeAndCleansSignals(t *testing.T) {
	resolution := ResolveArchitecture("gemma4", "gemma4_text", []string{" ", " Gemma4ForConditionalGeneration "})
	if !resolution.Matched() ||
		resolution.Architecture != "gemma4_text" ||
		!slices.Equal(resolution.Architectures, []string{"Gemma4ForConditionalGeneration"}) ||
		resolution.Profile.ID != "gemma4_text" {
		t.Fatalf("ResolveArchitecture = %+v, want cleaned Gemma4 text-tower resolution", resolution)
	}

	resolution.Architectures[0] = "mutated"
	resolution.Profile.QuantizationHints[0] = "mutated"
	next := ResolveArchitecture("gemma4", "gemma4_text", []string{"Gemma4ForConditionalGeneration"})
	if next.Architectures[0] == "mutated" || next.Profile.QuantizationHints[0] == "mutated" {
		t.Fatalf("ResolveArchitecture returned mutable shared data: %+v", next)
	}
}
