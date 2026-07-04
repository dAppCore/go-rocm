// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestGemma4ArchitectureSettings_Good_MatchesReactiveContract(t *testing.T) {
	settings, ok := Gemma4ArchitectureSettingsForArchitecture("Gemma4ForConditionalGeneration")
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
	if len(settings.WeightWrapperPrefixes) == 0 ||
		settings.WeightWrapperPrefixes[0] != "model.language_model.model." ||
		!slices.Contains(settings.WeightSkipPrefixes, "vision_tower") ||
		!slices.Contains(settings.WeightSkipSubstrings, "self_attn.rotary_emb") ||
		!slices.Contains(settings.WeightModelPrefixes, "layers.") {
		t.Fatalf("settings weight policy = %+v, want Gemma4 weight-name policy", settings)
	}
	if !slices.Contains(settings.QuantizationHints, "q6") ||
		!slices.Contains(settings.CacheHints, "k-q8-v-q4") ||
		!slices.Contains(settings.Aliases, "Gemma4ForConditionalGeneration") {
		t.Fatalf("settings hints = %+v cache=%+v aliases=%+v, want ROCm Gemma4 reactive profile metadata", settings.QuantizationHints, settings.CacheHints, settings.Aliases)
	}

	textSettings, ok := Gemma4ArchitectureSettingsForArchitecture("gemma4_unified_text")
	if !ok || textSettings.ID != "gemma4_text" || textSettings.TextTowerID != "" {
		t.Fatalf("text settings = %+v ok=%v, want unified text alias to resolve as Gemma4 text tower", textSettings, ok)
	}
	textSettings.QuantizationHints[0] = "mutated"
	textSettings, _ = Gemma4ArchitectureSettingsForArchitecture("gemma4_text")
	if textSettings.QuantizationHints[0] == "mutated" {
		t.Fatalf("settings quantization hints were not defensively copied: %+v", textSettings.QuantizationHints)
	}
}

func TestDefaultGemma4ArchitectureSettings_Good_DiscoverableProfiles(t *testing.T) {
	profiles := DefaultGemma4ArchitectureSettings()
	if len(profiles) != 4 {
		t.Fatalf("DefaultGemma4ArchitectureSettings len = %d, want 4: %+v", len(profiles), profiles)
	}
	seen := map[string]Gemma4ArchitectureSettings{}
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
	next := DefaultGemma4ArchitectureSettings()
	if next[0].QuantizationHints[0] == "mutated" {
		t.Fatalf("DefaultGemma4ArchitectureSettings returned mutable backing slices: %+v", next[0])
	}
}

func TestROCmResolveArchitecture_Good_MatchesReactiveResolver(t *testing.T) {
	cases := []struct {
		name      string
		modelType string
		textTower string
		archs     []string
		want      string
	}{
		{name: "qwen2.5 alias", modelType: "qwen2.5", archs: []string{"Qwen2.5ForCausalLM"}, want: "qwen2"},
		{name: "qwen3.5 to 3.6", modelType: "qwen3_5", archs: []string{"Qwen3_5ForConditionalGeneration"}, want: "qwen3_6"},
		{name: "qwen3.5 moe", modelType: "qwen3_5_moe", archs: []string{"Qwen3_5MoeForConditionalGeneration"}, want: "qwen3_6_moe"},
		{name: "text tower fallback", textTower: "qwen3_5_text", archs: []string{"Qwen3_5ForConditionalGeneration"}, want: "qwen3_6"},
		{name: "architecture fallback", archs: []string{"MiniMaxM2ForCausalLM"}, want: "minimax_m2"},
		{name: "gemma4 multimodal text tower", modelType: "gemma4", textTower: "gemma4_text", archs: []string{"Gemma4ForConditionalGeneration"}, want: "gemma4_text"},
		{name: "gemma4 wrapper without tower", modelType: "gemma4", archs: []string{"Gemma4ForConditionalGeneration"}, want: "gemma4"},
		{name: "gemma4 unified stays wrapper", modelType: "gemma4_unified", textTower: "gemma4_unified_text", archs: []string{"Gemma4UnifiedForConditionalGeneration"}, want: "gemma4_unified"},
		{name: "gemma4 unified text alias", modelType: "gemma4_unified_text", archs: []string{"Gemma4TextForCausalLM"}, want: "gemma4_text"},
		{name: "bert plain", modelType: "bert", archs: []string{"BertModel"}, want: "bert"},
		{name: "bert rerank refined", modelType: "bert", archs: []string{"BertForSequenceClassification"}, want: "bert_rerank"},
		{name: "empty"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ROCmResolveArchitecture(tc.modelType, tc.textTower, tc.archs); got != tc.want {
				t.Fatalf("ROCmResolveArchitecture(%q, %q, %v) = %q, want %q", tc.modelType, tc.textTower, tc.archs, got, tc.want)
			}
		})
	}
	resolution := ResolveROCmArchitecture("gemma4", "gemma4_text", []string{" Gemma4ForConditionalGeneration "})
	if !resolution.Matched() ||
		resolution.Contract != ROCmArchitectureResolutionContract ||
		resolution.Architecture != "gemma4_text" ||
		resolution.Source != "model_type_text_tower" ||
		resolution.ModelType != "gemma4" ||
		resolution.TextTowerModelType != "gemma4_text" ||
		len(resolution.Architectures) != 1 ||
		resolution.Architectures[0] != "Gemma4ForConditionalGeneration" ||
		resolution.Profile.ID != "gemma4_text" ||
		resolution.Profile.TokenizerKind != "GemmaTokenizer" ||
		resolution.Profile.ChatTemplate != "gemma4_hf_turn" {
		t.Fatalf("ResolveROCmArchitecture structured result = %+v, want copy-safe Gemma4 text-tower dispatch profile", resolution)
	}
	resolution.Architectures[0] = "mutated"
	resolution.Profile.QuantizationHints[0] = "mutated"
	next := ResolveROCmArchitecture("gemma4", "gemma4_text", []string{"Gemma4ForConditionalGeneration"})
	if next.Architectures[0] == "mutated" || next.Profile.QuantizationHints[0] == "mutated" {
		t.Fatalf("ResolveROCmArchitecture returned mutable shared data: %+v", next)
	}
}

func TestROCmArchitectureProfiles_Good_GenericRegistryContract(t *testing.T) {
	profiles := DefaultROCmArchitectureProfiles()
	if len(profiles) < 24 {
		t.Fatalf("DefaultROCmArchitectureProfiles len = %d, want broad reactive target list: %+v", len(profiles), profiles)
	}
	seen := map[string]ROCmArchitectureProfile{}
	for _, profile := range profiles {
		seen[profile.ID] = profile
	}
	for _, id := range []string{"gemma4_text", "qwen3_6", "qwen3_6_moe", "minimax_m2", "bert_rerank", "gpt-oss"} {
		if _, ok := seen[id]; !ok {
			t.Fatalf("DefaultROCmArchitectureProfiles missing %q: %+v", id, profiles)
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
	if seen["bert_rerank"].Family != "bert" ||
		!seen["bert_rerank"].Rerank ||
		seen["bert_rerank"].Chat ||
		seen["bert_rerank"].RequiresChatTemplate ||
		seen["bert_rerank"].ChatTemplate != "" {
		t.Fatalf("bert_rerank profile = %+v, want rerank non-chat metadata", seen["bert_rerank"])
	}
	profiles[0].Aliases = []string{"mutated"}
	next, ok := ROCmArchitectureProfileForArchitecture("bert")
	if !ok || slices.Contains(next.Aliases, "mutated") {
		t.Fatalf("ROCmArchitectureProfileForArchitecture returned mutable aliases: %+v ok=%v", next, ok)
	}
	if _, ok := ROCmArchitectureProfileForArchitecture("unknown_family"); ok {
		t.Fatal("ROCmArchitectureProfileForArchitecture(unknown_family) ok = true, want false")
	}
}

func TestROCmArchitectureProfileHelpers_Good_MatchRegistrySettings(t *testing.T) {
	if !ROCmDefaultThinkingEnabled("Gemma4ForCausalLM") ||
		!ROCmRequiresChatTemplate("gemma4_unified_text") ||
		ROCmAttachedOnlyArchitecture("gemma4_text") {
		t.Fatalf("target helper defaults = thinking:%t requires_template:%t attached:%t, want Gemma4 target registry settings",
			ROCmDefaultThinkingEnabled("Gemma4ForCausalLM"),
			ROCmRequiresChatTemplate("gemma4_unified_text"),
			ROCmAttachedOnlyArchitecture("gemma4_text"))
	}
	template, ok := ROCmChatTemplateID("gemma4_unified")
	if !ok || template != "gemma4_hf_turn" {
		t.Fatalf("ROCmChatTemplateID(gemma4_unified) = %q ok=%v, want gemma4_hf_turn", template, ok)
	}
	role, ok := ROCmGenerationRole("gemma4_text")
	if !ok || role != "model" {
		t.Fatalf("ROCmGenerationRole(gemma4_text) = %q ok=%v, want model", role, ok)
	}
	reasoning, ok := ROCmReasoningParserID("Gemma4TextForCausalLM")
	if !ok || reasoning != "gemma" {
		t.Fatalf("ROCmReasoningParserID(Gemma4TextForCausalLM) = %q ok=%v, want gemma", reasoning, ok)
	}
	tool, ok := ROCmToolParserID("Gemma4UnifiedForConditionalGeneration")
	if !ok || tool != "gemma" {
		t.Fatalf("ROCmToolParserID(Gemma4UnifiedForConditionalGeneration) = %q ok=%v, want gemma", tool, ok)
	}
	tokenizerKind, ok := ROCmTokenizerKind("Gemma4UnifiedForConditionalGeneration")
	if !ok || tokenizerKind != "GemmaTokenizer" {
		t.Fatalf("ROCmTokenizerKind(Gemma4UnifiedForConditionalGeneration) = %q ok=%v, want GemmaTokenizer", tokenizerKind, ok)
	}

	if !ROCmAttachedOnlyArchitecture("Gemma4AssistantForCausalLM") ||
		ROCmDefaultThinkingEnabled("gemma4_assistant") ||
		ROCmRequiresChatTemplate("gemma4_assistant") {
		t.Fatalf("assistant helper defaults = attached:%t thinking:%t requires_template:%t, want attached-only non-chat drafter",
			ROCmAttachedOnlyArchitecture("Gemma4AssistantForCausalLM"),
			ROCmDefaultThinkingEnabled("gemma4_assistant"),
			ROCmRequiresChatTemplate("gemma4_assistant"))
	}
	if template, ok := ROCmChatTemplateID("gemma4_assistant"); ok || template != "" {
		t.Fatalf("ROCmChatTemplateID(gemma4_assistant) = %q ok=%v, want empty false", template, ok)
	}
	if role, ok := ROCmGenerationRole("gemma4_assistant"); ok || role != "" {
		t.Fatalf("ROCmGenerationRole(gemma4_assistant) = %q ok=%v, want empty false", role, ok)
	}
	qwenTemplate, ok := ROCmChatTemplateID("qwen3")
	if !ok || qwenTemplate != "qwen" || !ROCmRequiresChatTemplate("qwen3") || ROCmDefaultThinkingEnabled("qwen3") || ROCmAttachedOnlyArchitecture("qwen3") {
		t.Fatalf("qwen helper defaults = template:%q ok:%v requires:%t thinking:%t attached:%t, want generic qwen chat profile",
			qwenTemplate, ok, ROCmRequiresChatTemplate("qwen3"), ROCmDefaultThinkingEnabled("qwen3"), ROCmAttachedOnlyArchitecture("qwen3"))
	}
	qwenParser, ok := ROCmReasoningParserID("Qwen3_5MoeForConditionalGeneration")
	qwenToolParser, toolOK := ROCmToolParserID("qwen3_6_moe")
	qwenStagedTemplate, templateOK := ROCmChatTemplateID("qwen3_6_moe")
	if !ok || qwenParser != "qwen" || !toolOK || qwenToolParser != "qwen" || !templateOK || qwenStagedTemplate != "qwen" || ROCmRequiresChatTemplate("qwen3_6_moe") {
		t.Fatalf("qwen staged helper defaults = parser:%q/%v tool:%q/%v template:%q/%v requires:%t, want parser/template discovery without standalone chat requirement",
			qwenParser, ok, qwenToolParser, toolOK, qwenStagedTemplate, templateOK, ROCmRequiresChatTemplate("qwen3_6_moe"))
	}
	if tokenizerKind, ok := ROCmTokenizerKind("Qwen3ForCausalLM"); !ok || tokenizerKind != "Qwen2Tokenizer" {
		t.Fatalf("ROCmTokenizerKind(Qwen3ForCausalLM) = %q ok=%v, want Qwen2Tokenizer", tokenizerKind, ok)
	}
	if role, ok := ROCmGenerationRole("qwen3_6_moe"); ok || role != "" {
		t.Fatalf("ROCmGenerationRole(qwen3_6_moe) = %q ok=%v, want empty false for staged no-chat profile", role, ok)
	}
	if parser, ok := ROCmReasoningParserID("BertForSequenceClassification"); !ok || parser != "generic" {
		t.Fatalf("ROCmReasoningParserID(BertForSequenceClassification) = %q ok=%v, want generic true", parser, ok)
	}
	if tokenizerKind, ok := ROCmTokenizerKind("BertForSequenceClassification"); !ok || tokenizerKind != "BertTokenizer" {
		t.Fatalf("ROCmTokenizerKind(BertForSequenceClassification) = %q ok=%v, want BertTokenizer", tokenizerKind, ok)
	}
	if template, ok := ROCmChatTemplateID("MiniMaxM2ForCausalLM"); ok || template != "" {
		t.Fatalf("ROCmChatTemplateID(MiniMaxM2ForCausalLM) = %q ok=%v, want empty false for staged profile", template, ok)
	}
	if ROCmDefaultThinkingEnabled("unknown") ||
		ROCmAttachedOnlyArchitecture("unknown") ||
		ROCmRequiresChatTemplate("unknown") {
		t.Fatalf("unknown helper defaults should be false")
	}
	if parser, ok := ROCmReasoningParserID("unknown"); ok || parser != "" {
		t.Fatalf("ROCmReasoningParserID(unknown) = %q ok=%v, want empty false", parser, ok)
	}
	if tokenizerKind, ok := ROCmTokenizerKind("unknown"); ok || tokenizerKind != "" {
		t.Fatalf("ROCmTokenizerKind(unknown) = %q ok=%v, want empty false", tokenizerKind, ok)
	}
}

func TestGemma4ArchitectureSettings_Good_AssistantAttachedOnly(t *testing.T) {
	settings, ok := Gemma4ArchitectureSettingsForArchitecture("Gemma4AssistantForCausalLM")
	if !ok {
		t.Fatal("Gemma4ArchitectureSettingsForArchitecture(assistant) ok = false")
	}
	if settings.ID != "gemma4_assistant" ||
		settings.RuntimeStatus != inference.FeatureRuntimeNative ||
		settings.ParserID != "gemma" ||
		settings.ToolParserID != "gemma" ||
		!settings.NativeRuntime ||
		!settings.AttachedOnly ||
		settings.Generation ||
		settings.Chat ||
		settings.RequiresChatTemplate ||
		settings.DefaultThinking ||
		settings.ChatTemplate != "" ||
		len(settings.WeightWrapperPrefixes) != 0 ||
		!slices.Contains(settings.CacheHints, "attached-drafter") ||
		len(settings.Notes) == 0 {
		t.Fatalf("assistant settings = %+v, want attached-only drafter profile without target chat/weight policy", settings)
	}
}

func TestGemma4ArchitectureSettings_Good_RegistryLabels(t *testing.T) {
	settings, ok := Gemma4ArchitectureSettingsForArchitecture("gemma4_text")
	if !ok {
		t.Fatal("Gemma4ArchitectureSettingsForArchitecture(gemma4_text) ok = false")
	}
	labels := rocmApplyGemma4ArchitectureSettingsLabels(map[string]string{"chat_template": "present"}, settings)
	if labels["engine_architecture_profile"] != "gemma4_text" ||
		labels["engine_architecture_family"] != "gemma4" ||
		labels["engine_architecture_runtime_status"] != string(inference.FeatureRuntimeNative) ||
		labels["engine_architecture_reasoning_parser"] != "gemma" ||
		labels["engine_architecture_tool_parser"] != "gemma" ||
		labels["engine_architecture_tokenizer_kind"] != "GemmaTokenizer" ||
		labels["reasoning_parser"] != "gemma" ||
		labels["tool_parser"] != "gemma" ||
		labels["engine_architecture_embeddings"] != "false" ||
		labels["engine_architecture_rerank"] != "false" ||
		labels["engine_architecture_moe"] != "false" ||
		labels["engine_architecture_quantization_hints"] != "bf16,q8,q6,q4,mxfp8,mxfp4" ||
		labels["engine_architecture_cache_hints"] != "q8,paged,k-q8-v-q4,retained-state" ||
		labels["engine_architecture_aliases"] != "Gemma4ForCausalLM,Gemma4TextForCausalLM" ||
		labels["engine_chat_template"] != "gemma4_hf_turn" ||
		labels["chat_template"] != "gemma4_hf_turn" ||
		labels["engine_default_thinking"] != "true" ||
		labels["engine_requires_chat_template"] != "true" ||
		labels["engine_generation_role"] != "model" ||
		labels["engine_weight_policy_source"] != "model_registry" ||
		labels["gemma4_weight_policy"] != "model_registry" ||
		labels["gemma4_weight_model_prefixes"] != "layers.,embed_tokens.,embed_tokens_per_layer.,norm.,per_layer_model_projection.,per_layer_projection_norm." {
		t.Fatalf("labels = %+v, want Gemma4 architecture settings labels", labels)
	}
}

func TestGemma4CanonicalWeightName_Good_ReactiveRegistryRules(t *testing.T) {
	cases := []struct {
		name string
		want string
		ok   bool
	}{
		{"language_model.model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight", true},
		{"model.language_model.model.layers.0.mlp.down_proj.scales", "model.layers.0.mlp.down_proj.scales", true},
		{"model.model.norm.weight", "model.norm.weight", true},
		{"vision_tower.encoder.weight", "", false},
		{"language_model.model.layers.0.self_attn.rotary_emb.inv_freq", "", false},
	}
	for _, tc := range cases {
		got, ok := Gemma4CanonicalWeightName("gemma4_text", tc.name)
		if ok != tc.ok || got != tc.want {
			t.Fatalf("Gemma4CanonicalWeightName(%q) = %q, %v; want %q, %v", tc.name, got, ok, tc.want, tc.ok)
		}
	}

	got, ok := Gemma4CanonicalWeightName("qwen3", "language_model.model.layers.0.self_attn.q_proj.weight")
	if !ok || got != "language_model.model.layers.0.self_attn.q_proj.weight" {
		t.Fatalf("non-Gemma4 canonical = %q, %v; want passthrough", got, ok)
	}
}
