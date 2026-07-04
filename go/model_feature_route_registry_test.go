// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestRegisterROCmModelFeatureRoute_Good_ExtendsReactiveFeatureRegistry(t *testing.T) {
	restoreRegisteredROCmModelFeatureRoutesForTest(t)

	RegisterROCmModelFeatureRoute(ROCmModelFeatureRoute{})
	RegisterROCmModelFeatureRoute(ROCmModelFeatureRoute{
		Architecture:      "fake-loader",
		ReasoningParserID: "first-parser",
	})
	RegisterROCmModelFeatureRoute(ROCmModelFeatureRoute{
		Architecture:         "fake-loader",
		Family:               "fake",
		RuntimeStatus:        inference.FeatureRuntimeNative,
		ReasoningParserID:    "fake-reasoning",
		ToolParserID:         "fake-tool",
		ChatTemplateID:       "fake-chat",
		GenerationRole:       "assistant",
		NativeRuntime:        true,
		Generation:           true,
		Chat:                 true,
		DefaultThinking:      true,
		RequiresChatTemplate: true,
		SequenceMixer:        true,
		Capabilities:         []inference.CapabilityID{inference.CapabilityRuntimeDiscovery},
	})

	if got := RegisteredROCmModelFeatureRouteArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmModelFeatureRouteArchitectures = %v, want normalized replacement under one architecture", got)
	}
	registered := RegisteredROCmModelFeatureRouteArchitectures()
	registered[0] = "mutated"
	if next := RegisteredROCmModelFeatureRouteArchitectures(); !slices.Equal(next, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmModelFeatureRouteArchitectures returned mutable state: %v", next)
	}

	route, ok := ROCmModelFeatureRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != ROCmModelFeatureRegistryContract ||
		route.Name != rocmModelFeatureRegistryRouteName ||
		route.Architecture != "fake_loader" ||
		route.Family != "fake" ||
		route.RuntimeStatus != inference.FeatureRuntimeNative ||
		route.ReasoningParserID != "fake-reasoning" ||
		route.ToolParserID != "fake-tool" ||
		route.ChatTemplateID != "fake-chat" ||
		route.GenerationRole != "assistant" ||
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
		!route.SequenceMixer ||
		!slices.Equal(route.Capabilities, []inference.CapabilityID{
			inference.CapabilityGenerate,
			inference.CapabilityChatTemplate,
			inference.CapabilityReasoningParse,
			inference.CapabilityToolParse,
			inference.CapabilityRuntimeDiscovery,
		}) ||
		route.Labels["engine_feature_route_reasoning_parser"] != "fake-reasoning" ||
		route.Labels["engine_feature_route_tool_parser"] != "fake-tool" ||
		route.Labels["engine_feature_route_chat_template_id"] != "fake-chat" ||
		route.Labels["engine_feature_route_sequence_mixer"] != "true" ||
		route.Labels["engine_feature_route_capabilities"] != "generate,chat.template,reasoning.parse,tool.parse,runtime.discovery" {
		t.Fatalf("ROCmModelFeatureRouteForArchitecture(fake-loader) = %+v ok=%v, want registered feature route", route, ok)
	}

	route.Capabilities[0] = inference.CapabilityRerank
	route.Labels["engine_feature_route_reasoning_parser"] = "mutated"
	nextRoute, ok := ROCmModelFeatureRouteForArchitecture("fake-loader")
	if !ok ||
		nextRoute.Capabilities[0] != inference.CapabilityGenerate ||
		nextRoute.Labels["engine_feature_route_reasoning_parser"] != "fake-reasoning" {
		t.Fatalf("ROCmModelFeatureRouteForArchitecture leaked mutable state: %+v ok=%v", nextRoute, ok)
	}
	modelRoute, ok := rocmmodel.RegisteredFeatureRouteForArchitecture("fake-loader")
	if !ok ||
		modelRoute.ReasoningParserID != "fake-reasoning" ||
		modelRoute.Labels["engine_feature_route_reasoning_parser"] != "fake-reasoning" {
		t.Fatalf("model.RegisteredFeatureRouteForArchitecture(fake-loader) = %+v ok=%v, want mirrored route", modelRoute, ok)
	}

	defaults := DefaultROCmModelFeatureRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmModelFeatureRoute) bool {
		return route.Architecture == "fake_loader" && route.ReasoningParserID == "fake-reasoning"
	}) {
		t.Fatalf("DefaultROCmModelFeatureRoutes missing registered feature route: %+v", defaults)
	}

	profile := ROCmModelProfile{
		Name:         "fake",
		Family:       "fake",
		Architecture: "fake_loader",
		ArchitectureProfile: ROCmArchitectureProfile{
			ID:             "fake_loader",
			Family:         "fake",
			RuntimeStatus:  inference.FeatureRuntimeNative,
			NativeRuntime:  true,
			ParserID:       "profile-reasoning",
			ToolParserID:   "profile-tool",
			ChatTemplate:   "profile-chat",
			GenerationRole: "assistant",
			Generation:     true,
			Chat:           true,
		},
	}
	features := ROCmEngineFeaturesForProfile(profile)
	if features.ReasoningParserID != "fake-reasoning" ||
		features.ToolParserID != "fake-tool" ||
		features.ChatTemplateID != "fake-chat" ||
		!features.TextGenerate ||
		!features.SequenceMixer ||
		!slices.Contains(features.Capabilities, inference.CapabilityRuntimeDiscovery) ||
		features.Labels["engine_feature_reasoning_parser"] != "fake-reasoning" {
		t.Fatalf("ROCmEngineFeaturesForProfile registered route = %+v, want engine features to merge registered feature route", features)
	}
	profile.EngineFeatures = features
	profileRoute := ROCmModelFeatureRouteForProfile(profile)
	if profileRoute.ReasoningParserID != "fake-reasoning" ||
		profileRoute.ChatTemplateID != "fake-chat" ||
		!profileRoute.TextGenerate ||
		!profileRoute.SequenceMixer ||
		profileRoute.Labels["engine_feature_route_capabilities"] != "generate,chat.template,reasoning.parse,tool.parse,runtime.discovery" {
		t.Fatalf("ROCmModelFeatureRouteForProfile registered route = %+v, want profile feature route to use registered features", profileRoute)
	}
}

func TestRegisterROCmModelFeatureRoute_Good_SeesModelPackageRegistration(t *testing.T) {
	restoreRegisteredROCmModelFeatureRoutesForTest(t)

	rocmmodel.RegisterFeatureRoute(rocmmodel.FeatureRoute{
		Architecture:      "folder-feature",
		Family:            "folder",
		RuntimeStatus:     inference.FeatureRuntimeNative,
		ReasoningParserID: "folder-reasoning",
		ToolParserID:      "folder-tool",
		ChatTemplateID:    "folder-chat",
		NativeRuntime:     true,
		Generation:        true,
		Chat:              true,
		SequenceMixer:     true,
		Capabilities:      []inference.CapabilityID{inference.CapabilityRuntimeDiscovery},
	})

	route, ok := ROCmModelFeatureRouteForArchitecture("folder-feature")
	if !ok ||
		route.Contract != ROCmModelFeatureRegistryContract ||
		route.Name != rocmModelFeatureRegistryRouteName ||
		route.Architecture != "folder_feature" ||
		route.Family != "folder" ||
		route.RuntimeStatus != inference.FeatureRuntimeNative ||
		route.ReasoningParserID != "folder-reasoning" ||
		route.ToolParserID != "folder-tool" ||
		route.ChatTemplateID != "folder-chat" ||
		!route.NativeRuntime ||
		!route.TextGenerate ||
		!route.SequenceMixer ||
		!slices.Contains(route.Capabilities, inference.CapabilityRuntimeDiscovery) ||
		route.Labels["engine_feature_route_reasoning_parser"] != "folder-reasoning" {
		t.Fatalf("ROCmModelFeatureRouteForArchitecture(folder-feature) = %+v ok=%v, want model package route", route, ok)
	}
	defaults := DefaultROCmModelFeatureRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmModelFeatureRoute) bool {
		return route.Architecture == "folder_feature" && route.ReasoningParserID == "folder-reasoning"
	}) {
		t.Fatalf("DefaultROCmModelFeatureRoutes missing model package route: %+v", defaults)
	}
	features := ROCmEngineFeaturesForProfile(ROCmModelProfile{
		Name:         "folder",
		Family:       "folder",
		Architecture: "folder_feature",
	})
	if features.ReasoningParserID != "folder-reasoning" ||
		features.ToolParserID != "folder-tool" ||
		features.ChatTemplateID != "folder-chat" ||
		!features.TextGenerate ||
		!features.SequenceMixer ||
		!slices.Contains(features.Capabilities, inference.CapabilityRuntimeDiscovery) {
		t.Fatalf("ROCmEngineFeaturesForProfile(folder_feature) = %+v, want model package feature route", features)
	}
}

func TestRegisterROCmModelFeatureRoute_Good_OverridesBuiltinFeatureRoute(t *testing.T) {
	restoreRegisteredROCmModelFeatureRoutesForTest(t)

	RegisterROCmModelFeatureRoute(ROCmModelFeatureRoute{
		Architecture:      "qwen3",
		Family:            "qwen",
		ReasoningParserID: "registered-qwen-reasoning",
		ToolParserID:      "registered-qwen-tool",
		ChatTemplateID:    "registered-qwen-chat",
		SequenceMixer:     true,
		Capabilities:      []inference.CapabilityID{inference.CapabilityRuntimeDiscovery},
	})

	route, ok := ROCmModelFeatureRouteForArchitecture("Qwen3ForCausalLM")
	if !ok ||
		route.Architecture != "qwen3" ||
		route.Family != "qwen" ||
		route.ReasoningParserID != "registered-qwen-reasoning" ||
		route.ToolParserID != "registered-qwen-tool" ||
		route.ChatTemplateID != "registered-qwen-chat" ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.TextGenerate ||
		!route.SequenceMixer ||
		!slices.Contains(route.Capabilities, inference.CapabilityRuntimeDiscovery) ||
		route.Labels["engine_feature_route_reasoning_parser"] != "registered-qwen-reasoning" {
		t.Fatalf("ROCmModelFeatureRouteForArchitecture(qwen3 override) = %+v ok=%v, want registered override", route, ok)
	}

	profile, ok := ResolveROCmModelProfile("/models/qwen", inference.ModelIdentity{Architecture: "Qwen3ForCausalLM"})
	if !ok ||
		profile.EngineFeatures.ReasoningParserID != "registered-qwen-reasoning" ||
		profile.EngineFeatures.ToolParserID != "registered-qwen-tool" ||
		profile.EngineFeatures.ChatTemplateID != "registered-qwen-chat" ||
		!profile.EngineFeatures.SequenceMixer ||
		profile.FeatureRoute.ReasoningParserID != "registered-qwen-reasoning" ||
		profile.FeatureRoute.ChatTemplateID != "registered-qwen-chat" ||
		!profile.FeatureRoute.SequenceMixer {
		t.Fatalf("ResolveROCmModelProfile(qwen3 registered feature override) = %+v ok=%v, want profile to expose registered features", profile, ok)
	}
}

func restoreRegisteredROCmModelFeatureRoutesForTest(t *testing.T) {
	t.Helper()
	modelRoutes := rocmmodel.RegisteredFeatureRoutes()

	t.Cleanup(func() {
		restoreRoutes := make([]rocmmodel.FeatureRoute, 0, len(modelRoutes))
		for _, route := range modelRoutes {
			restoreRoutes = append(restoreRoutes, route.Clone())
		}
		rocmmodel.ReplaceRegisteredFeatureRoutes(restoreRoutes)
	})
}
