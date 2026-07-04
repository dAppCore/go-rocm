// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"context"
	"iter"
	"slices"
	"strconv"
	"testing"

	"dappco.re/go/inference"
)

type rocmRegistryAPITestTextModel struct {
	modelType string
	info      inference.ModelInfo
}

func (m *rocmRegistryAPITestTextModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (m *rocmRegistryAPITestTextModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (m *rocmRegistryAPITestTextModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (m *rocmRegistryAPITestTextModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (m *rocmRegistryAPITestTextModel) ModelType() string { return m.modelType }
func (m *rocmRegistryAPITestTextModel) Info() inference.ModelInfo {
	return m.info
}
func (m *rocmRegistryAPITestTextModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{}
}
func (m *rocmRegistryAPITestTextModel) Err() error   { return nil }
func (m *rocmRegistryAPITestTextModel) Close() error { return nil }

type rocmRegistryAPITestProfileModel struct {
	rocmRegistryAPITestTextModel
	profile ROCmModelProfile
}

func (m *rocmRegistryAPITestProfileModel) ModelProfile() ROCmModelProfile {
	return m.profile
}

type rocmRegistryAPITestIdentityModel struct {
	rocmRegistryAPITestTextModel
	identity inference.ModelIdentity
}

func (m *rocmRegistryAPITestIdentityModel) ModelIdentity() inference.ModelIdentity {
	return m.identity
}

func TestROCmEngineFeaturesForProfile_Good_Gemma4UsesModelProfile(t *testing.T) {
	profile, ok := ResolveROCmModelProfile("", inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		Architecture: "gemma4_text",
		QuantBits:    6,
		NumLayers:    26,
		HiddenSize:   2304,
		Labels: map[string]string{
			"gemma4_size":       "E4B",
			"gemma4_quant_mode": "q6",
			"sliding_window":    "1024",
		},
	})
	if !ok {
		t.Fatal("ResolveROCmModelProfile(gemma4) ok = false")
	}
	features := profile.EngineFeatures
	if features.Contract != rocmEngineFeaturesContract ||
		features.Architecture != "gemma4_text" ||
		features.Family != "gemma4" ||
		features.ReasoningParserID != "gemma" ||
		features.ToolParserID != "gemma" ||
		features.ChatTemplateID != "gemma4_hf_turn" ||
		!features.NativeRuntime ||
		!features.TextGenerate ||
		!features.DirectGreedyToken ||
		!features.NativeMLPMatVec ||
		!features.NativeLinearMatVec ||
		!features.NativeQ6BitstreamMatVec ||
		!features.NativeAttentionOMatVec ||
		!features.NativeFixedSlidingAttention ||
		!features.GenerationStream ||
		!features.AsyncDecodePrefetch ||
		!features.ModelContextWindow ||
		!features.DeviceKVState ||
		!features.FixedSlidingCache ||
		!features.FixedSlidingCacheBound ||
		features.CompiledLayerDecode ||
		features.PipelinedDecode ||
		!features.ChatTemplate ||
		!features.ReasoningParse ||
		!features.ToolParse ||
		!slices.Contains(features.Capabilities, inference.CapabilityGenerate) ||
		!slices.Contains(features.Capabilities, inference.CapabilityChatTemplate) ||
		!slices.Contains(features.Capabilities, inference.CapabilityReasoningParse) ||
		!slices.Contains(features.Capabilities, inference.CapabilityToolParse) {
		t.Fatalf("Gemma4 ROCmEngineFeatures = %+v, want model-profile derived linked generation/cache/parser features", features)
	}
	if profile.LoadStatus.Contract != rocmModelLoadStatusContract ||
		profile.LoadStatus.Status != ROCmModelLoadStandaloneNative ||
		profile.LoadStatus.Target != "standalone" ||
		!profile.LoadStatus.Standalone ||
		profile.LoadStatus.Staged ||
		!profile.LoadStatus.TextGenerate {
		t.Fatalf("Gemma4 load status = %+v, want standalone native generation", profile.LoadStatus)
	}
	if profile.QuantLoaderRoute.Contract != ROCmQuantLoaderRegistryContract ||
		profile.QuantLoaderRoute.Size != "E4B" ||
		profile.QuantLoaderRoute.Mode != "q6" ||
		profile.QuantLoaderRoute.Loader != "gemma4_affine" ||
		profile.QuantLoaderRoute.Runtime != Gemma4RuntimeMLXAffine ||
		profile.QuantLoaderRoute.GenerateStatus != Gemma4GenerateLinked ||
		!profile.QuantLoaderRoute.Registered ||
		!profile.QuantLoaderRoute.NativeRuntime ||
		!profile.QuantLoaderRoute.RunnableOnCard {
		t.Fatalf("Gemma4 quant loader route = %+v, want model-derived q6 affine loader route", profile.QuantLoaderRoute)
	}
	if profile.LoRAAdapterRoute.Contract != ROCmLoRAAdapterRegistryContract ||
		profile.LoRAAdapterRoute.Architecture != "gemma4_text" ||
		profile.LoRAAdapterRoute.TargetPolicy != "gemma4" ||
		profile.LoRAAdapterRoute.TargetPaths["q_proj"] != "self_attn.q_proj" ||
		!profile.LoRAAdapterRoute.Registered ||
		!profile.LoRAAdapterRoute.NativeRuntime ||
		!profile.LoRAAdapterRoute.ApplySupported ||
		profile.LoRAAdapterRoute.Staged ||
		profile.LoRAAdapterRoute.Status != ROCmLoRAAdapterRouteExperimentalNative {
		t.Fatalf("Gemma4 LoRA adapter route = %+v, want loaded model-owned adapter target route", profile.LoRAAdapterRoute)
	}
	capabilities := features.EnabledCapabilities()
	capabilities[0] = inference.CapabilityRerank
	if features.EnabledCapabilities()[0] == inference.CapabilityRerank {
		t.Fatalf("EnabledCapabilities leaked mutable backing slice: %+v", features.EnabledCapabilities())
	}
	labels := ApplyROCmModelProfileLabels(nil, profile)
	if labels["engine_features_contract"] != rocmEngineFeaturesContract ||
		labels["engine_feature_text_generate"] != "true" ||
		labels["engine_feature_direct_greedy_token"] != "true" ||
		labels["engine_feature_native_mlp_matvec"] != "true" ||
		labels["engine_feature_native_linear_matvec"] != "true" ||
		labels["engine_feature_native_q6_bitstream_matvec"] != "true" ||
		labels["engine_feature_native_attention_o_matvec"] != "true" ||
		labels["engine_feature_native_fixed_sliding_attention"] != "true" ||
		labels["engine_feature_generation_stream"] != "true" ||
		labels["engine_feature_async_decode_prefetch"] != "true" ||
		labels["engine_feature_fixed_sliding_cache"] != "true" ||
		labels["engine_feature_compiled_layer_decode"] != "false" ||
		labels["engine_feature_pipelined_decode"] != "false" ||
		labels["engine_feature_capabilities"] != "generate,chat.template,reasoning.parse,tool.parse" ||
		labels["engine_load_status"] != string(ROCmModelLoadStandaloneNative) ||
		labels["engine_load_target"] != "standalone" ||
		labels["engine_load_text_generate"] != "true" ||
		labels["engine_quant_loader_contract"] != ROCmQuantLoaderRegistryContract ||
		labels["engine_quant_loader"] != "gemma4_affine" ||
		labels["engine_quant_loader_mode"] != "q6" ||
		labels["engine_quant_loader_generate_status"] != Gemma4GenerateLinked ||
		labels["engine_lora_route_contract"] != ROCmLoRAAdapterRegistryContract ||
		labels["engine_lora_target_policy"] != "gemma4" ||
		labels["engine_lora_status"] != string(ROCmLoRAAdapterRouteExperimentalNative) ||
		labels["engine_lora_apply_supported"] != "true" ||
		labels["engine_text_generate"] != "true" {
		t.Fatalf("Gemma4 engine feature labels = %+v, want generic and Gemma4-specific feature labels", labels)
	}
}

func TestROCmModelRoutePlanForProfile_Good_CollectsReactiveRoutes(t *testing.T) {
	profile, ok := ResolveROCmModelProfile("", inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		Architecture: "gemma4_text",
		QuantBits:    6,
		NumLayers:    26,
		HiddenSize:   2304,
		Labels: map[string]string{
			"gemma4_size":       "E4B",
			"gemma4_quant_mode": "q6",
			"sliding_window":    "1024",
		},
	})
	if !ok {
		t.Fatal("ResolveROCmModelProfile(gemma4) ok = false")
	}

	plan := ROCmModelRoutePlanForProfile(profile)

	if plan.Contract != ROCmModelRoutePlanContract ||
		plan.Architecture != "gemma4_text" ||
		plan.Family != "gemma4" ||
		!plan.Matched() ||
		!plan.EngineFeatures.TextGenerate ||
		!plan.FeatureRoute.Matched() ||
		!plan.TokenizerRoute.Matched() ||
		!plan.LoRAAdapterRoute.Matched() ||
		!plan.StateContextRoute.Matched() ||
		!plan.AttachedDrafterRoute.Matched() ||
		plan.LoadStatus.Status != ROCmModelLoadStandaloneNative ||
		!plan.LoaderRoute.Matched() ||
		plan.LoaderRoute.Contract != ROCmModelLoaderRegistryContract ||
		!plan.QuantLoaderRoute.Matched() ||
		plan.QuantLoaderRoute.Mode != "q6" ||
		plan.Labels["engine_route_plan_contract"] != ROCmModelRoutePlanContract ||
		plan.Labels["engine_route_plan_architecture"] != "gemma4_text" ||
		plan.Labels["engine_route_plan_feature"] != "true" ||
		plan.Labels["engine_route_plan_tokenizer"] != "true" ||
		plan.Labels["engine_route_plan_lora_adapter"] != "true" ||
		plan.Labels["engine_route_plan_state_context"] != "true" ||
		plan.Labels["engine_route_plan_drafter"] != "true" ||
		plan.Labels["engine_route_plan_loader"] != "true" ||
		plan.Labels["engine_route_plan_quant_loader"] != "true" ||
		plan.Labels["engine_route_plan_load_status"] != string(ROCmModelLoadStandaloneNative) ||
		plan.Labels["engine_route_plan_load_target"] != "standalone" ||
		plan.Labels["engine_route_plan_loader_name"] != "gemma4_text" ||
		plan.Labels["engine_route_plan_loader_runtime"] != rocmModelLoaderRuntimeHIP ||
		plan.Labels["engine_route_plan_quant_size"] != "E4B" ||
		plan.Labels["engine_route_plan_quant_mode"] != "q6" ||
		plan.Labels["engine_route_plan_quant_generate_status"] != Gemma4GenerateLinked ||
		plan.Labels["engine_route_plan_quant_target"] != "generate" ||
		plan.Labels["engine_route_plan_state_context_status"] != string(ROCmStateContextRouteExperimentalRuntime) ||
		plan.Labels["engine_route_plan_state_context_runtime"] != rocmStateContextRuntimeAPI ||
		plan.Labels["engine_route_plan_state_context_retained_state_required"] != "true" ||
		plan.Labels["engine_route_plan_state_context_runtime_owned_kv"] != "true" ||
		plan.Labels["engine_route_plan_state_context_prompt_replay_refused"] != "true" ||
		plan.Labels["engine_route_plan_state_context_remaining_default"] != "true" ||
		plan.Labels["engine_route_plan_drafter_status"] != string(ROCmAttachedDrafterRouteNativePending) ||
		plan.Labels["engine_route_plan_drafter_runtime"] != rocmAttachedDrafterRuntimeMetadata ||
		plan.Labels["engine_route_plan_drafter_role"] != "target" ||
		plan.Labels["engine_route_plan_drafter_native_attachment"] != hipKernelStatusNotLinked ||
		plan.Labels["engine_route_plan_drafter_assistant_architecture"] != "gemma4_assistant" ||
		plan.Labels["engine_route_plan_drafter_retained_state_required"] != "true" ||
		plan.Labels["engine_route_plan_drafter_prompt_replay_refused"] != "true" ||
		plan.Labels["engine_route_plan_drafter_fallback_refused"] != "true" ||
		plan.Labels["engine_route_plan_drafter_default_tokens"] != strconv.Itoa(ProductionMTPDefaultDraftTokens) {
		t.Fatalf("ROCmModelRoutePlanForProfile = %+v, want collected reactive route plan", plan)
	}
	applied := ApplyROCmModelRoutePlanLabels(map[string]string{"caller": "kept"}, plan)
	if applied["caller"] != "kept" ||
		applied["engine_route_plan_contract"] != ROCmModelRoutePlanContract ||
		applied["engine_route_plan_feature"] != "true" ||
		applied["engine_route_plan_loader"] != "true" ||
		applied["engine_route_plan_drafter_native_attachment"] != hipKernelStatusNotLinked ||
		applied["engine_route_plan_state_context_runtime_owned_kv"] != "true" {
		t.Fatalf("ApplyROCmModelRoutePlanLabels = %+v, want route-plan labels plus caller labels", applied)
	}

	plan.Model.Labels["gemma4_size"] = "mutated"
	plan.FeatureRoute.Labels["engine_feature_route_contract"] = "mutated"
	if len(plan.EngineFeatures.Capabilities) > 0 {
		plan.EngineFeatures.Capabilities[0] = inference.CapabilityRerank
	}
	next := ROCmModelRoutePlanForProfile(profile)
	if next.Model.Labels["gemma4_size"] == "mutated" ||
		next.FeatureRoute.Labels["engine_feature_route_contract"] == "mutated" ||
		slices.Contains(next.EngineFeatures.Capabilities, inference.CapabilityRerank) {
		t.Fatalf("ROCmModelRoutePlanForProfile leaked mutable route-plan data: %+v", next)
	}
	byIdentity, ok := ROCmModelRoutePlanForIdentity("", profile.Model)
	if !ok || !byIdentity.Matched() || byIdentity.Architecture != "gemma4_text" {
		t.Fatalf("ROCmModelRoutePlanForIdentity = %+v ok=%v, want Gemma4 route plan", byIdentity, ok)
	}
	assistantProfile, ok := ResolveROCmModelProfile("/models/gemma4-assistant", inference.ModelIdentity{Architecture: "Gemma4AssistantForCausalLM"})
	if !ok {
		t.Fatal("ResolveROCmModelProfile(gemma4 assistant) ok = false")
	}
	assistantPlan := ROCmModelRoutePlanForProfile(assistantProfile)
	if !assistantPlan.Matched() ||
		assistantPlan.LoadStatus.Status != ROCmModelLoadAttachedOnly ||
		assistantPlan.Labels["engine_route_plan_load_attached_only"] != "true" ||
		assistantPlan.Labels["engine_route_plan_state_context_attached_drafter_state"] != "true" ||
		assistantPlan.Labels["engine_route_plan_drafter_role"] != "assistant" ||
		assistantPlan.Labels["engine_route_plan_drafter_attached_only"] != "true" ||
		assistantPlan.Labels["engine_route_plan_drafter_target_architecture"] != "gemma4_text" ||
		assistantPlan.Labels["engine_route_plan_drafter_assistant_architecture"] != "gemma4_assistant" {
		t.Fatalf("ROCmModelRoutePlanForProfile(assistant) = %+v, want attached-only assistant route-plan labels", assistantPlan)
	}
	if plan, ok := ROCmModelRoutePlanForInfo("/models/unknown", inference.ModelInfo{Architecture: "nope"}, nil); ok || plan.Matched() {
		t.Fatalf("ROCmModelRoutePlanForInfo(unknown) = %+v ok=%v, want empty false", plan, ok)
	}
}

func TestResolveROCmModelProfileForModel_Good_UsesModelOwnedProfile(t *testing.T) {
	model := &rocmRegistryAPITestProfileModel{
		profile: ROCmModelProfile{
			Name:     "gemma4",
			Family:   "gemma4",
			Registry: rocmModelRegistryName,
			Model: inference.ModelIdentity{
				Path:          "/models/profile-owned",
				Architecture:  "gemma4_text",
				ContextLength: 8192,
				Labels:        map[string]string{"source": "profile"},
			},
			EngineFeatures: ROCmEngineFeatures{
				Contract:     rocmEngineFeaturesContract,
				Capabilities: []inference.CapabilityID{inference.CapabilityGenerate},
				Labels:       map[string]string{"engine_feature_text_generate": "true"},
			},
			Gemma4EngineFeatures: Gemma4EngineFeatures{
				ModelContextWindow: true,
				TextGenerate:       true,
				MLXAffineDecode:    true,
			},
			Labels: map[string]string{"engine_profile": "gemma4"},
		},
	}

	profile, ok := ResolveROCmModelProfileForModel(model)
	if !ok ||
		profile.Model.ContextLength != 8192 ||
		profile.Model.Labels["source"] != "profile" ||
		!profile.Gemma4EngineFeatures.GenerateLinked() ||
		!slices.Contains(profile.EngineFeatures.Capabilities, inference.CapabilityGenerate) {
		t.Fatalf("ResolveROCmModelProfileForModel(profile reporter) = %+v ok=%v, want model-owned profile", profile, ok)
	}
	profile.Model.Labels["source"] = "mutated"
	profile.EngineFeatures.Capabilities[0] = inference.CapabilityChat
	profile.EngineFeatures.Labels["engine_feature_text_generate"] = "mutated"
	profile.Labels["engine_profile"] = "mutated"
	next, ok := ResolveROCmModelProfileForModel(model)
	if !ok ||
		next.Model.Labels["source"] == "mutated" ||
		next.EngineFeatures.Capabilities[0] == inference.CapabilityChat ||
		next.EngineFeatures.Labels["engine_feature_text_generate"] == "mutated" ||
		next.Labels["engine_profile"] == "mutated" {
		t.Fatalf("ResolveROCmModelProfileForModel returned aliased profile data: %+v ok=%v", next, ok)
	}
}

func TestROCmEngineFeaturesForModel_Good_UsesIdentityReporter(t *testing.T) {
	model := &rocmRegistryAPITestIdentityModel{
		identity: inference.ModelIdentity{
			Path:          "/models/lmstudio-community-gemma-4-e4b-it-6bit",
			Architecture:  "gemma4_text",
			QuantBits:     6,
			NumLayers:     26,
			HiddenSize:    2304,
			ContextLength: 8192,
			Labels: map[string]string{
				"gemma4_size":       "E4B",
				"gemma4_quant_mode": "q6",
				"sliding_window":    "1024",
			},
		},
	}

	profile, ok := ResolveROCmModelProfileForModel(model)
	if !ok ||
		profile.Model.ContextLength != 8192 ||
		profile.Model.Labels["gemma4_size"] != "E4B" ||
		profile.EngineFeatures.ChatTemplateID != "gemma4_hf_turn" {
		t.Fatalf("ResolveROCmModelProfileForModel(identity reporter) = %+v ok=%v, want identity-derived Gemma4 profile", profile, ok)
	}
	features, ok := ROCmEngineFeaturesForModel(model)
	if !ok ||
		features.Contract != rocmEngineFeaturesContract ||
		!features.TextGenerate ||
		!features.ModelContextWindow ||
		!features.FixedSlidingCache ||
		!features.ChatTemplate ||
		features.ChatTemplateID != "gemma4_hf_turn" ||
		!slices.Contains(features.Capabilities, inference.CapabilityGenerate) {
		t.Fatalf("ROCmEngineFeaturesForModel(identity reporter) = %+v ok=%v, want reactive Gemma4 features", features, ok)
	}
	features.Capabilities[0] = inference.CapabilityChat
	features.Labels["engine_feature_text_generate"] = "mutated"
	next, ok := ROCmEngineFeaturesForModel(model)
	if !ok ||
		next.Capabilities[0] == inference.CapabilityChat ||
		next.Labels["engine_feature_text_generate"] == "mutated" {
		t.Fatalf("ROCmEngineFeaturesForModel returned aliased feature data: %+v ok=%v", next, ok)
	}
}

func TestResolveROCmModelProfileForModel_Good_FallsBackToInfoAndModelType(t *testing.T) {
	model := &rocmRegistryAPITestTextModel{
		modelType: "Qwen3_5MoeForConditionalGeneration",
		info: inference.ModelInfo{
			QuantBits: 4,
		},
	}

	profile, ok := ResolveROCmModelProfileForModel(model)
	if !ok ||
		profile.Architecture != "qwen3_6_moe" ||
		profile.Family != "qwen" ||
		profile.EngineFeatures.ChatTemplateID != "qwen" ||
		!profile.EngineFeatures.ChatTemplate ||
		!profile.EngineFeatures.ReasoningParse ||
		profile.EngineFeatures.TextGenerate {
		t.Fatalf("ResolveROCmModelProfileForModel(info fallback) = %+v ok=%v, want model-type-derived Qwen profile", profile, ok)
	}
	features, ok := ROCmEngineFeaturesForModel(model)
	if !ok ||
		features.Architecture != "qwen3_6_moe" ||
		features.ChatTemplateID != "qwen" ||
		!features.ChatTemplate ||
		!features.ToolParse ||
		features.TextGenerate {
		t.Fatalf("ROCmEngineFeaturesForModel(info fallback) = %+v ok=%v, want Qwen parser/template features", features, ok)
	}
}

func TestROCmEngineFeaturesForInfo_Good_GenericArchitectureProfile(t *testing.T) {
	features, ok := ROCmEngineFeaturesForInfo("/models/qwen", inference.ModelInfo{
		Architecture: "Qwen3_5MoeForConditionalGeneration",
		QuantBits:    4,
	}, nil)
	if !ok {
		t.Fatal("ROCmEngineFeaturesForInfo(qwen) ok = false")
	}
	if features.Contract != rocmEngineFeaturesContract ||
		features.Architecture != "qwen3_6_moe" ||
		features.Family != "qwen" ||
		features.ReasoningParserID != "qwen" ||
		features.ToolParserID != "qwen" ||
		features.ChatTemplateID != "qwen" ||
		!features.NativeRuntime ||
		features.TextGenerate ||
		features.ModelContextWindow ||
		!features.ChatTemplate ||
		!features.ReasoningParse ||
		!features.ToolParse ||
		!features.MoE ||
		!slices.Contains(features.Capabilities, inference.CapabilityChatTemplate) ||
		!slices.Contains(features.Capabilities, inference.CapabilityReasoningParse) ||
		!slices.Contains(features.Capabilities, inference.CapabilityToolParse) ||
		slices.Contains(features.Capabilities, inference.CapabilityGenerate) {
		t.Fatalf("Qwen ROCmEngineFeatures = %+v, want staged parser/template/MoE features without standalone generation", features)
	}
	profile, ok := ResolveROCmModelProfile("/models/qwen", inference.ModelIdentity{Architecture: "Qwen3_5MoeForConditionalGeneration"})
	if !ok {
		t.Fatal("ResolveROCmModelProfile(qwen) ok = false")
	}
	if profile.LoadStatus.Status != ROCmModelLoadStagedNative ||
		profile.LoadStatus.Target != "standalone" ||
		!profile.LoadStatus.Standalone ||
		!profile.LoadStatus.Staged ||
		profile.LoadStatus.TextGenerate ||
		profile.LoadStatus.AttachedOnly {
		t.Fatalf("Qwen load status = %+v, want staged standalone native metadata path", profile.LoadStatus)
	}
	if profile.LoRAAdapterRoute.Contract != ROCmLoRAAdapterRegistryContract ||
		profile.LoRAAdapterRoute.Architecture != "qwen3_6_moe" ||
		profile.LoRAAdapterRoute.TargetPolicy != "decoder" ||
		profile.LoRAAdapterRoute.TargetPaths["q_proj"] != "self_attn.q_proj" ||
		!profile.LoRAAdapterRoute.Registered ||
		!profile.LoRAAdapterRoute.Staged ||
		profile.LoRAAdapterRoute.Planned {
		t.Fatalf("Qwen LoRA adapter route = %+v, want staged decoder adapter metadata", profile.LoRAAdapterRoute)
	}
	labels := ApplyROCmModelProfileLabels(nil, profile)
	if labels["engine_features_contract"] != rocmEngineFeaturesContract ||
		labels["engine_feature_architecture"] != "qwen3_6_moe" ||
		labels["engine_feature_family"] != "qwen" ||
		labels["engine_feature_text_generate"] != "false" ||
		labels["engine_feature_chat_template"] != "true" ||
		labels["engine_feature_chat_template_id"] != "qwen" ||
		labels["engine_feature_reasoning_parser"] != "qwen" ||
		labels["engine_feature_tool_parser"] != "qwen" ||
		labels["engine_feature_reasoning_parse"] != "true" ||
		labels["engine_feature_tool_parse"] != "true" ||
		labels["engine_feature_moe"] != "true" ||
		labels["engine_feature_capabilities"] != "chat.template,reasoning.parse,tool.parse" ||
		labels["engine_load_status"] != string(ROCmModelLoadStagedNative) ||
		labels["engine_load_target"] != "standalone" ||
		labels["engine_load_staged"] != "true" ||
		labels["engine_load_text_generate"] != "false" ||
		labels["engine_lora_route_contract"] != ROCmLoRAAdapterRegistryContract ||
		labels["engine_lora_status"] != string(ROCmLoRAAdapterRouteStagedNative) ||
		labels["engine_lora_target_policy"] != "decoder" {
		t.Fatalf("Qwen engine feature labels = %+v, want generic architecture-profile feature contract", labels)
	}
}

func TestROCmEngineFeaturesForInfo_Good_SequenceMixerLabelsEnableReactiveRoute(t *testing.T) {
	labels := map[string]string{
		"attention_layer_types":              "full_attention,mamba2",
		"sequence_mixer_declared_kinds":      "full_attention,mamba2",
		"sequence_mixer_load_plan_status":    "valid",
		"sequence_mixer_load_plan_entries":   "0:full_attention:kv-cache:self_attn:planned_hip,1:mamba2:recurrent:mixer:planned_hip",
		"sequence_mixer_cache_plan_contract": SequenceMixerCachePlanContract,
		"sequence_mixer_cache_plan_entries":  "0:full_attention:kv-cache:default,1:mamba2:recurrent:recurrent",
	}
	profile, ok := ResolveROCmModelProfileForInfo("/models/qwen-next", inference.ModelInfo{
		Architecture: "Qwen3NextForCausalLM",
		NumLayers:    2,
		QuantBits:    4,
	}, labels)
	if !ok {
		t.Fatal("ResolveROCmModelProfileForInfo(qwen-next sequence mixer) ok = false")
	}
	if !profile.EngineFeatures.SequenceMixer ||
		profile.EngineFeatures.Labels["engine_feature_sequence_mixer"] != "true" ||
		!profile.FeatureRoute.SequenceMixer ||
		profile.FeatureRoute.Labels["engine_feature_route_sequence_mixer"] != "true" {
		t.Fatalf("profile = %+v, want sequence-mixer engine feature and route resolved from load-plan labels", profile)
	}
	applied := ApplyROCmModelProfileLabels(nil, profile)
	if applied["engine_feature_sequence_mixer"] != "true" ||
		applied["engine_feature_route_sequence_mixer"] != "true" {
		t.Fatalf("ApplyROCmModelProfileLabels = %+v, want sequence-mixer feature labels", applied)
	}
	labels["sequence_mixer_load_plan_status"] = "mutated"
	if profile.Model.Labels["sequence_mixer_load_plan_status"] != "valid" {
		t.Fatalf("ResolveROCmModelProfileForInfo aliased sequence mixer labels: %+v", profile.Model.Labels)
	}
}

func TestResolveROCmModelProfile_Good_SequenceMixerRoutesFromRouteSet(t *testing.T) {
	mamba, ok := ResolveROCmModelProfile("/models/mamba", inference.ModelIdentity{
		Architecture: "Mamba2ForCausalLM",
	})
	if !ok ||
		mamba.Architecture != "mamba2" ||
		len(mamba.SequenceMixerRoutes) != 1 ||
		mamba.SequenceMixerRoutes[0].Kind != "mamba2" ||
		mamba.SequenceMixerRoutes[0].CacheMode != SequenceMixerCacheModeRecurrent ||
		mamba.Labels["engine_route_set_sequence_mixer"] != "true" ||
		mamba.Labels["engine_route_set_sequence_mixer_kinds"] != "mamba2" {
		t.Fatalf("ResolveROCmModelProfile(mamba2) = %+v ok=%v, want public sequence mixer route", mamba, ok)
	}

	mamba.SequenceMixerRoutes[0].Labels["engine_mixer_loader_kind"] = "mutated"
	next, ok := ResolveROCmModelProfile("/models/mamba", inference.ModelIdentity{
		Architecture: "Mamba2ForCausalLM",
	})
	if !ok || next.SequenceMixerRoutes[0].Labels["engine_mixer_loader_kind"] != "mamba2" {
		t.Fatalf("ResolveROCmModelProfile sequence mixer route leaked mutable labels: %+v ok=%v", next, ok)
	}

	hybrid, ok := ResolveROCmModelProfile("/models/hybrid", inference.ModelIdentity{
		Architecture: "hybrid",
		Labels: map[string]string{
			"layer_types": "full_attention,mamba2,mla,mamba2",
		},
	})
	if !ok ||
		hybrid.Architecture != "hybrid" ||
		len(hybrid.SequenceMixerRoutes) != 3 ||
		hybrid.SequenceMixerRoutes[0].Kind != "full_attention" ||
		hybrid.SequenceMixerRoutes[1].Kind != "mamba2" ||
		hybrid.SequenceMixerRoutes[2].Kind != "mla" ||
		hybrid.Labels["engine_route_set_sequence_mixer_cache_modes"] != "default,recurrent,mla-latent" {
		t.Fatalf("ResolveROCmModelProfile(hybrid) = %+v ok=%v, want composed sequence mixer routes", hybrid, ok)
	}
}

func TestResolveROCmModelProfile_Good_RuntimeContractRouteFromRouteSet(t *testing.T) {
	gemma4, ok := ResolveROCmModelProfile("/models/gemma4", inference.ModelIdentity{
		Architecture: "Gemma4ForConditionalGeneration",
		Labels: map[string]string{
			"engine_architecture_resolved": "gemma4_text",
		},
	})
	if !ok ||
		gemma4.Architecture != "gemma4_text" ||
		!gemma4.RuntimeContractRoute.Matched() ||
		!gemma4.RuntimeContractRoute.LastTokenLogits ||
		!gemma4.RuntimeContractRoute.GreedyToken ||
		!gemma4.RuntimeContractRoute.FixedSlidingCache ||
		!gemma4.RuntimeContractRoute.ModelInfoReporter ||
		!slices.Contains(gemma4.RuntimeContractRoute.ContractIDs, ROCmRuntimeContractThoughtChannelSuppressor) ||
		gemma4.Labels["engine_route_set_runtime_contract"] != "true" {
		t.Fatalf("ResolveROCmModelProfile(gemma4) = %+v ok=%v, want public runtime contract route", gemma4, ok)
	}
	labels := ApplyROCmModelProfileLabels(nil, gemma4)
	if labels["engine_runtime_contract_route_contract"] != ROCmModelRuntimeContractRegistryContract ||
		labels["engine_runtime_contract_fixed_sliding_cache"] != "true" ||
		labels["engine_runtime_contract_model_info_reporter"] != "true" {
		t.Fatalf("ApplyROCmModelProfileLabels = %+v, want runtime contract labels", labels)
	}

	gemma4.RuntimeContractRoute.Labels["engine_runtime_contract_architecture"] = "mutated"
	next, ok := ResolveROCmModelProfile("/models/gemma4", inference.ModelIdentity{
		Architecture: "Gemma4ForConditionalGeneration",
		Labels: map[string]string{
			"engine_architecture_resolved": "gemma4_text",
		},
	})
	if !ok || next.RuntimeContractRoute.Labels["engine_runtime_contract_architecture"] != "gemma4_text" {
		t.Fatalf("ResolveROCmModelProfile runtime contract route leaked mutable labels: %+v ok=%v", next, ok)
	}

	mamba, ok := ResolveROCmModelProfile("/models/mamba", inference.ModelIdentity{
		Architecture: "Mamba2ForCausalLM",
	})
	if !ok ||
		mamba.RuntimeContractRoute.NativeRuntime ||
		!mamba.RuntimeContractRoute.MetadataOnly ||
		!mamba.RuntimeContractRoute.DecodeUnavailableReporter ||
		!slices.Equal(mamba.RuntimeContractRoute.ContractIDs, []ROCmModelRuntimeContractID{ROCmRuntimeContractModelInfoReporter, ROCmRuntimeContractDecodeUnavailableReport}) {
		t.Fatalf("ResolveROCmModelProfile(mamba2) = %+v ok=%v, want metadata-only runtime contracts", mamba, ok)
	}

	hybridMoE, ok := ResolveROCmModelProfile("/models/qwen36-moe", inference.ModelIdentity{
		Architecture: "Qwen3.6MoeForConditionalGeneration",
	})
	if !ok ||
		hybridMoE.Architecture != "qwen3_6_moe" ||
		!hybridMoE.RuntimeContractRoute.MoETextRuntimeReporter ||
		!hybridMoE.RuntimeContractRoute.HybridAttentionCachePlanner ||
		hybridMoE.RuntimeContractRoute.TextGenerate {
		t.Fatalf("ResolveROCmModelProfile(qwen3_6_moe) = %+v ok=%v, want staged hybrid/MoE runtime contracts", hybridMoE, ok)
	}
}

func TestResolveROCmModelProfileForInspection_Good_ConfigResolvedArchitecture(t *testing.T) {
	inspection := &inference.ModelPackInspection{
		Path: "/models/qwen",
		Model: inference.ModelIdentity{
			Architecture: "qwen3",
			Labels: map[string]string{
				"model_label": "present",
			},
		},
		Labels: map[string]string{
			"engine_architecture_resolved": "qwen3_6_moe",
			"inspection_label":             "present",
		},
	}
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok {
		t.Fatal("ResolveROCmModelProfileForInspection(qwen) ok = false")
	}
	if profile.Architecture != "qwen3_6_moe" ||
		profile.EngineFeatures.Family != "qwen" ||
		profile.EngineFeatures.ChatTemplateID != "qwen" ||
		profile.Model.Labels["inspection_label"] != "present" ||
		profile.Model.Labels["model_label"] != "present" {
		t.Fatalf("profile = %+v, want inspection-label resolved Qwen MoE profile", profile)
	}
	features, ok := ROCmEngineFeaturesForInspection(inspection)
	if !ok || features.Architecture != "qwen3_6_moe" || !features.ToolParse {
		t.Fatalf("ROCmEngineFeaturesForInspection = %+v ok=%v, want Qwen MoE parser features", features, ok)
	}
	loadStatus, ok := ROCmModelLoadStatusForInspection(inspection)
	if !ok || loadStatus.Status != ROCmModelLoadStagedNative || loadStatus.Architecture != "qwen3_6_moe" {
		t.Fatalf("ROCmModelLoadStatusForInspection = %+v ok=%v, want Qwen MoE staged-native load status", loadStatus, ok)
	}
	route, ok := ROCmModelLoaderRouteForInspection(inspection)
	if !ok ||
		route.Contract != ROCmModelLoaderRegistryContract ||
		route.Architecture != "qwen3_6_moe" ||
		route.Loader != "qwen3_6_moe" ||
		route.Runtime != "hip" ||
		route.Status != ROCmModelLoadStagedNative ||
		!route.Registered ||
		!route.Staged ||
		route.Labels["engine_loader_contract"] != ROCmModelLoaderRegistryContract {
		t.Fatalf("ROCmModelLoaderRouteForInspection = %+v ok=%v, want Qwen MoE staged HIP loader route", route, ok)
	}
}

func TestResolveROCmModelProfileForInspection_Good_Gemma4ResolvedTextTower(t *testing.T) {
	inspection := &inference.ModelPackInspection{
		Path: "/models/gemma-4-e2b-it-6bit",
		Model: inference.ModelIdentity{
			Path:         "/models/gemma-4-e2b-it-6bit",
			Architecture: "gemma4",
			Labels: map[string]string{
				"model_label": "present",
			},
		},
		Labels: map[string]string{
			"engine_architecture_resolved":     "gemma4_text",
			"architecture_resolution_contract": ROCmArchitectureResolutionContract,
			"architecture_resolution_source":   "model_type_text_tower",
			"inspection_label":                 "present",
		},
	}
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok {
		t.Fatal("ResolveROCmModelProfileForInspection(gemma4 text tower) ok = false")
	}
	if profile.Architecture != "gemma4_text" ||
		profile.Model.Architecture != "gemma4_text" ||
		profile.Gemma4Settings.ID != "gemma4_text" ||
		profile.EngineFeatures.Architecture != "gemma4_text" ||
		!profile.EngineFeatures.ChatTemplate ||
		profile.LoadStatus.Architecture != "gemma4_text" ||
		profile.LoadStatus.Loader != "gemma4_text" ||
		profile.LoadStatus.LoaderRuntime != "hip" ||
		!profile.LoadStatus.LoaderRegistered ||
		profile.LoadStatus.Status != ROCmModelLoadStandaloneNative ||
		profile.LoadStatus.Labels["engine_loader_contract"] != ROCmModelLoaderRegistryContract ||
		profile.Model.Labels["inspection_label"] != "present" ||
		profile.Model.Labels["model_label"] != "present" {
		t.Fatalf("profile = %+v, want Gemma4 factory to honor resolved text-tower dispatch identity", profile)
	}
	route := ROCmModelLoaderRouteForProfile(profile)
	if route.Architecture != "gemma4_text" ||
		route.Loader != "gemma4_text" ||
		route.Status != ROCmModelLoadStandaloneNative ||
		!route.Registered ||
		!route.Standalone ||
		route.Labels["engine_loader_runtime"] != "hip" {
		t.Fatalf("ROCmModelLoaderRouteForProfile = %+v, want Gemma4 text-tower standalone HIP route", route)
	}
	quantRoute, ok := ROCmQuantLoaderRouteForInspection(inspection)
	if !ok ||
		quantRoute.Contract != ROCmQuantLoaderRegistryContract ||
		quantRoute.Size != "E2B" ||
		quantRoute.Mode != "q6" ||
		quantRoute.Loader != "gemma4_affine" ||
		quantRoute.Target != "generate" ||
		!quantRoute.Registered ||
		!quantRoute.NativeRuntime {
		t.Fatalf("ROCmQuantLoaderRouteForInspection = %+v ok=%v, want Gemma4 q6 affine quant route", quantRoute, ok)
	}
	loraRoute, ok := ROCmLoRAAdapterRouteForInspection(inspection)
	if !ok ||
		loraRoute.Contract != ROCmLoRAAdapterRegistryContract ||
		loraRoute.Architecture != "gemma4_text" ||
		loraRoute.TargetPolicy != "gemma4" ||
		loraRoute.TargetPaths["o_proj"] != "self_attn.o_proj" ||
		!loraRoute.Registered ||
		loraRoute.Staged ||
		loraRoute.Status != ROCmLoRAAdapterRouteExperimentalNative {
		t.Fatalf("ROCmLoRAAdapterRouteForInspection = %+v ok=%v, want Gemma4 text-tower adapter route", loraRoute, ok)
	}
}

func TestROCmEngineFeaturesForInfo_Bad_UnknownArchitecture(t *testing.T) {
	if features, ok := ROCmEngineFeaturesForInfo("/models/unknown", inference.ModelInfo{Architecture: "nope"}, nil); ok || !features.empty() {
		t.Fatalf("ROCmEngineFeaturesForInfo(unknown) = %+v ok=%v, want empty false", features, ok)
	}
}

func TestROCmModelFeatureRouteForProfile_Good_LoadedModelRefreshesRuntimeCapabilities(t *testing.T) {
	profile, ok := ResolveROCmModelProfile("", inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		Architecture: "gemma4_text",
		QuantBits:    6,
		NumLayers:    26,
		HiddenSize:   2304,
		Labels: map[string]string{
			"gemma4_size":       "E4B",
			"gemma4_quant_mode": "q6",
			"sliding_window":    "1024",
		},
	})
	if !ok {
		t.Fatal("ResolveROCmModelProfile(gemma4) ok = false")
	}
	if !profile.FeatureRoute.Matched() {
		t.Fatalf("profile.FeatureRoute = %+v, want populated feature route", profile.FeatureRoute)
	}
	route := ROCmModelFeatureRouteForProfile(profile)
	if route.Contract != ROCmModelFeatureRegistryContract ||
		route.Architecture != "gemma4_text" ||
		route.Family != "gemma4" ||
		route.ReasoningParserID != "gemma" ||
		route.ToolParserID != "gemma" ||
		route.ChatTemplateID != "gemma4_hf_turn" ||
		route.GenerationRole != "model" ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.Generation ||
		!route.TextGenerate ||
		!route.Chat ||
		!route.ModelContextWindow ||
		!route.ReasoningParse ||
		!route.ToolParse ||
		!route.ChatTemplate ||
		!route.DefaultThinking ||
		!route.RequiresChatTemplate ||
		!slices.Contains(route.Capabilities, inference.CapabilityGenerate) ||
		route.Labels["engine_feature_route_contract"] != ROCmModelFeatureRegistryContract ||
		route.Labels["engine_feature_route_capabilities"] != "generate,chat.template,reasoning.parse,tool.parse" {
		t.Fatalf("loaded Gemma4 feature route = %+v, want runtime-refreshed parser/template/generate route", route)
	}
	labels := ApplyROCmModelProfileLabels(nil, profile)
	if labels["engine_feature_route_contract"] != ROCmModelFeatureRegistryContract ||
		labels["engine_feature_route_text_generate"] != "true" ||
		labels["engine_feature_route_chat_template_id"] != "gemma4_hf_turn" ||
		labels["engine_feature_route_capabilities"] != "generate,chat.template,reasoning.parse,tool.parse" {
		t.Fatalf("loaded Gemma4 feature route labels = %+v, want route labels applied to profile", labels)
	}
	route.Capabilities[0] = inference.CapabilityRerank
	next := ROCmModelFeatureRouteForProfile(profile)
	if next.Capabilities[0] == inference.CapabilityRerank {
		t.Fatalf("ROCmModelFeatureRouteForProfile leaked mutable capability slice: %+v", next)
	}
	byIdentity, ok := ROCmModelFeatureRouteForIdentity("", profile.Model)
	if !ok || byIdentity.Architecture != "gemma4_text" || !byIdentity.TextGenerate {
		t.Fatalf("ROCmModelFeatureRouteForIdentity = %+v ok=%v, want loaded Gemma4 route", byIdentity, ok)
	}
	if route, ok := ROCmModelFeatureRouteForInfo("/models/unknown", inference.ModelInfo{Architecture: "nope"}, nil); ok || route.Matched() {
		t.Fatalf("ROCmModelFeatureRouteForInfo(unknown) = %+v ok=%v, want empty false", route, ok)
	}
}

func TestROCmLoRAAdapterRouteForProfile_Good_ModelOwnedTargetPolicy(t *testing.T) {
	profile, ok := ResolveROCmModelProfile("/models/qwen", inference.ModelIdentity{Architecture: "Qwen3ForCausalLM"})
	if !ok {
		t.Fatal("ResolveROCmModelProfile(qwen3) ok = false")
	}
	route := profile.LoRAAdapterRoute
	if route.Contract != ROCmLoRAAdapterRegistryContract ||
		route.Architecture != "qwen3" ||
		route.Family != "qwen" ||
		route.TargetPolicy != "decoder" ||
		!slices.Equal(route.DefaultTargets, []string{"q_proj", "v_proj"}) ||
		route.TargetPaths["q_proj"] != "self_attn.q_proj" ||
		route.TargetPaths["mlp.up_proj"] != "mlp.up_proj" ||
		!route.Registered ||
		!route.ApplySupported ||
		!route.LoadSupported ||
		!route.FuseSupported ||
		!route.TrainingSupported ||
		route.Staged ||
		route.Planned ||
		!slices.Contains(route.Capabilities, inference.CapabilityLoRAInference) ||
		!slices.Contains(route.Capabilities, inference.CapabilityLoRATraining) ||
		!slices.Contains(route.Capabilities, inference.CapabilityModelMerge) ||
		route.Labels["engine_lora_route_contract"] != ROCmLoRAAdapterRegistryContract ||
		route.Labels["engine_lora_capabilities"] != "lora.inference,lora.training,model.merge" {
		t.Fatalf("Qwen3 LoRA adapter route = %+v, want decoder adapter target contract", route)
	}
	if path, ok := ROCmLoRATargetPath("Qwen3ForCausalLM", "gate_proj"); !ok || path != "mlp.gate_proj" {
		t.Fatalf("ROCmLoRATargetPath(qwen3, gate_proj) = %q ok=%v, want mlp.gate_proj true", path, ok)
	}
	if canonical, ok := ROCmLoRACanonicalTarget("qwen3", "model.layers.0.q_proj"); !ok || canonical != "model.layers.0.self_attn.q_proj" {
		t.Fatalf("ROCmLoRACanonicalTarget(qwen3) = %q ok=%v, want model.layers.0.self_attn.q_proj true", canonical, ok)
	}
	byIdentity, ok := ROCmLoRAAdapterRouteForIdentity("/models/qwen", profile.Model)
	if !ok || byIdentity.Architecture != "qwen3" || byIdentity.TargetPolicy != "decoder" {
		t.Fatalf("ROCmLoRAAdapterRouteForIdentity = %+v ok=%v, want qwen decoder route", byIdentity, ok)
	}
	route.DefaultTargets[0] = "mutated"
	route.TargetPaths["q_proj"] = "mutated"
	next := ROCmLoRAAdapterRouteForProfile(profile)
	if next.DefaultTargets[0] == "mutated" || next.TargetPaths["q_proj"] == "mutated" {
		t.Fatalf("ROCmLoRAAdapterRouteForProfile leaked mutable target policy: %+v", next)
	}
	if route, ok := ROCmLoRAAdapterRouteForInfo("/models/unknown", inference.ModelInfo{Architecture: "BertForSequenceClassification"}, nil); ok || route.Matched() {
		t.Fatalf("ROCmLoRAAdapterRouteForInfo(bert_rerank) = %+v ok=%v, want empty false", route, ok)
	}
}

func TestROCmModelTokenizerRouteForInspection_Good_SidecarMetadataRefreshesRoute(t *testing.T) {
	inspection := &inference.ModelPackInspection{
		Path: "/models/qwen",
		Model: inference.ModelIdentity{
			Path:         "/models/qwen",
			Architecture: "Qwen3_5MoeForConditionalGeneration",
			Labels: map[string]string{
				"engine_architecture_resolved": "qwen3_6_moe",
			},
		},
		Tokenizer: inference.TokenizerIdentity{
			Kind:         "Qwen2Tokenizer",
			Path:         "/models/qwen/tokenizer_config.json",
			ChatTemplate: "{% for message in messages %}{{ message.content }}{% endfor %}",
			EOSID:        151645,
			PADID:        151643,
			Labels: map[string]string{
				"tokenizer_hash": "tok-a",
			},
		},
		Labels: map[string]string{
			"tokenizer_json":                 "present",
			"tokenizer_config":               "present",
			"chat_template":                  "present",
			"engine_architecture_resolved":   "qwen3_6_moe",
			"architecture_resolution_source": "model_type_architecture_refinement",
		},
	}
	route, ok := ROCmModelTokenizerRouteForInspection(inspection)
	if !ok ||
		route.Contract != ROCmModelTokenizerRegistryContract ||
		route.Architecture != "qwen3_6_moe" ||
		route.Family != "qwen" ||
		route.Loader != "hf-tokenizer-json" ||
		route.TokenizerKind != "Qwen2Tokenizer" ||
		route.TokenizerPath != "/models/qwen/tokenizer_config.json" ||
		route.ChatTemplateID != "qwen" ||
		route.ChatTemplateSource != "sidecar" ||
		!route.SidecarTokenizer ||
		!route.SidecarConfig ||
		!route.SidecarTemplate ||
		!route.ChatTemplate ||
		route.Generation ||
		route.Chat ||
		route.EOSID != 151645 ||
		route.PADID != 151643 ||
		route.Tokenizer.Labels["tokenizer_hash"] != "tok-a" ||
		!slices.Contains(route.Capabilities, inference.CapabilityTokenizer) ||
		!slices.Contains(route.Capabilities, inference.CapabilityChatTemplate) ||
		route.Labels["engine_tokenizer_route_contract"] != ROCmModelTokenizerRegistryContract ||
		route.Labels["engine_tokenizer_chat_template_source"] != "sidecar" ||
		route.Labels["engine_tokenizer_eos_id"] != "151645" {
		t.Fatalf("ROCmModelTokenizerRouteForInspection = %+v ok=%v, want sidecar-refreshed Qwen tokenizer route", route, ok)
	}
	route.RequiredFiles[0] = "mutated"
	route.Tokenizer.Labels["tokenizer_hash"] = "mutated"
	next, ok := ROCmModelTokenizerRouteForInspection(inspection)
	if !ok ||
		next.RequiredFiles[0] == "mutated" ||
		next.Tokenizer.Labels["tokenizer_hash"] != "tok-a" {
		t.Fatalf("ROCmModelTokenizerRouteForInspection leaked mutable state: %+v ok=%v", next, ok)
	}
	labels := ApplyROCmModelProfileLabels(nil, mustResolveROCmProfileForTest(t, inspection))
	if labels["engine_tokenizer_route_contract"] != ROCmModelTokenizerRegistryContract ||
		labels["engine_tokenizer_loader"] != "hf-tokenizer-json" {
		t.Fatalf("profile tokenizer labels = %+v, want default tokenizer route labels", labels)
	}
}

func mustResolveROCmProfileForTest(t *testing.T, inspection *inference.ModelPackInspection) ROCmModelProfile {
	t.Helper()
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok {
		t.Fatalf("ResolveROCmModelProfileForInspection(%+v) ok = false", inspection)
	}
	return profile
}
