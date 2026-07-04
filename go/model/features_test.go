// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestDefaultFeatureRoutes_Good_ArchitectureCatalogue(t *testing.T) {
	routes := DefaultFeatureRoutes()
	if len(routes) == 0 {
		t.Fatal("DefaultFeatureRoutes returned no routes")
	}
	byArchitecture := map[string]FeatureRoute{}
	for _, route := range routes {
		byArchitecture[route.Architecture] = route
	}
	gemma4 := byArchitecture["gemma4"]
	if !gemma4.Matched() ||
		gemma4.Contract != FeatureRegistryContract ||
		gemma4.Name != FeatureRouteName ||
		gemma4.RuntimeStatus != inference.FeatureRuntimeNative ||
		gemma4.ReasoningParserID == "" ||
		gemma4.ChatTemplateID == "" ||
		!gemma4.Registered ||
		!gemma4.NativeRuntime ||
		!gemma4.Generation ||
		!gemma4.TextGenerate ||
		!gemma4.Chat ||
		!gemma4.ModelContextWindow ||
		!gemma4.ReasoningParse ||
		!gemma4.ToolParse ||
		!gemma4.ChatTemplate ||
		gemma4.Labels["engine_feature_route_architecture"] != "gemma4" {
		t.Fatalf("gemma4 feature route = %+v, want native chat/generation route", gemma4)
	}
	bert := byArchitecture["bert"]
	if !bert.Matched() ||
		bert.RuntimeStatus != inference.FeatureRuntimeNative ||
		!bert.Registered ||
		!bert.NativeRuntime ||
		bert.TextGenerate ||
		!bert.Embeddings ||
		!slices.Contains(bert.Capabilities, inference.CapabilityEmbeddings) {
		t.Fatalf("bert feature route = %+v, want embedding-capable non-generation route", bert)
	}
}

func TestFeatureRouteForArchitecture_Good_NormalizesAliases(t *testing.T) {
	route, ok := FeatureRouteForArchitecture("Gemma4ForConditionalGeneration")
	if !ok ||
		route.Architecture != "gemma4" ||
		route.Family != "gemma4" ||
		route.RuntimeStatus != inference.FeatureRuntimeNative ||
		!route.TextGenerate ||
		!route.ChatTemplate {
		t.Fatalf("FeatureRouteForArchitecture(Gemma4ForConditionalGeneration) = %+v ok=%v, want gemma4 route", route, ok)
	}
	if _, ok := FeatureRouteForArchitecture("totally_unknown_feature_route"); ok {
		t.Fatal("FeatureRouteForArchitecture(unknown) ok = true, want false")
	}
}

func TestFeatureRouteForIdentity_Good_ResolvedArchitectureLabelsWin(t *testing.T) {
	route, ok := FeatureRouteForIdentity("/models/gemma4", inference.ModelIdentity{
		Architecture: "Gemma4ForConditionalGeneration",
		Labels: map[string]string{
			"engine_architecture_resolved": "bert_rerank",
		},
	})
	if !ok ||
		route.Architecture != "bert_rerank" ||
		route.Family != "bert" ||
		!route.Rerank ||
		route.TextGenerate {
		t.Fatalf("FeatureRouteForIdentity = %+v ok=%v, want resolved BERT rerank route", route, ok)
	}
}

func TestFeatureRouteForInspection_Good_MergesLabelsCopySafe(t *testing.T) {
	inspectionLabels := map[string]string{"architecture_resolved": "qwen3"}
	modelLabels := map[string]string{"engine_architecture_resolved": "bert_rerank"}
	inspection := &inference.ModelPackInspection{
		Path:   "/models/rerank",
		Labels: inspectionLabels,
		Model: inference.ModelIdentity{
			Architecture: "Qwen3ForCausalLM",
			Labels:       modelLabels,
		},
	}
	route, ok := FeatureRouteForInspection(inspection)
	if !ok ||
		route.Architecture != "bert_rerank" ||
		route.Family != "bert" ||
		!route.Rerank ||
		route.TextGenerate {
		t.Fatalf("FeatureRouteForInspection = %+v ok=%v, want model label refined BERT rerank route", route, ok)
	}
	inspectionLabels["architecture_resolved"] = "mutated"
	modelLabels["engine_architecture_resolved"] = "mutated"
	route.Labels["engine_feature_route_architecture"] = "mutated"
	next, ok := FeatureRouteForArchitecture("bert_rerank")
	if !ok || next.Labels["engine_feature_route_architecture"] != "bert_rerank" {
		t.Fatalf("FeatureRouteForInspection leaked mutable route labels: %+v ok=%v", next, ok)
	}
}

func TestRegisterFeatureRoute_Good_ExtendsReactiveCatalogue(t *testing.T) {
	restoreRegisteredFeaturesForTest(t)

	RegisterFeatureRoute(FeatureRoute{})
	RegisterFeatureRoute(FeatureRoute{Architecture: "fake-loader", ReasoningParserID: "first-parser"})
	RegisterFeatureRoute(FeatureRoute{
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

	if got := RegisteredFeatureArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredFeatureArchitectures = %v, want normalized replacement", got)
	}
	registeredRoutes := RegisteredFeatureRoutes()
	if len(registeredRoutes) != 1 ||
		registeredRoutes[0].Architecture != "fake_loader" ||
		registeredRoutes[0].ReasoningParserID != "fake-reasoning" {
		t.Fatalf("RegisteredFeatureRoutes = %+v, want one replacement route", registeredRoutes)
	}
	registeredRoute, ok := RegisteredFeatureRouteForArchitecture("fake-loader")
	if !ok || registeredRoute.ReasoningParserID != "fake-reasoning" {
		t.Fatalf("RegisteredFeatureRouteForArchitecture(fake-loader) = %+v ok=%v, want registered route", registeredRoute, ok)
	}
	registeredRoutes[0].Labels["engine_feature_route_reasoning_parser"] = "mutated"
	nextRegisteredRoute, ok := RegisteredFeatureRouteForArchitecture("fake-loader")
	if !ok || nextRegisteredRoute.Labels["engine_feature_route_reasoning_parser"] != "fake-reasoning" {
		t.Fatalf("RegisteredFeatureRoutes leaked mutable labels: %+v ok=%v", nextRegisteredRoute, ok)
	}
	ReplaceRegisteredFeatureRoutes(registeredRoutes)
	replaced, ok := RegisteredFeatureRouteForArchitecture("fake-loader")
	if !ok || replaced.Labels["engine_feature_route_reasoning_parser"] != "fake-reasoning" {
		t.Fatalf("ReplaceRegisteredFeatureRoutes = %+v ok=%v, want copy-safe restored route", replaced, ok)
	}

	route, ok := FeatureRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != FeatureRegistryContract ||
		route.Name != FeatureRouteName ||
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
		route.Labels["engine_feature_route_capabilities"] != "generate,chat.template,reasoning.parse,tool.parse,runtime.discovery" {
		t.Fatalf("FeatureRouteForArchitecture(fake-loader) = %+v ok=%v, want registered feature route", route, ok)
	}
	route.Capabilities[0] = inference.CapabilityRerank
	route.Labels["engine_feature_route_reasoning_parser"] = "mutated"
	next, ok := FeatureRouteForArchitecture("fake-loader")
	if !ok ||
		next.Capabilities[0] != inference.CapabilityGenerate ||
		next.Labels["engine_feature_route_reasoning_parser"] != "fake-reasoning" {
		t.Fatalf("FeatureRouteForArchitecture leaked mutable state: %+v ok=%v", next, ok)
	}
	if !slices.ContainsFunc(DefaultFeatureRoutes(), func(route FeatureRoute) bool {
		return route.Architecture == "fake_loader" && route.ReasoningParserID == "fake-reasoning"
	}) {
		t.Fatalf("DefaultFeatureRoutes missing fake_loader registration")
	}
}

func restoreRegisteredFeaturesForTest(t *testing.T) {
	t.Helper()
	order, routes := registeredFeatures.Snapshot()
	for architecture, route := range routes {
		routes[architecture] = route.Clone()
	}
	t.Cleanup(func() {
		restoreRoutes := make(map[string]FeatureRoute, len(routes))
		for architecture, route := range routes {
			restoreRoutes[architecture] = route.Clone()
		}
		registeredFeatures.Restore(order, restoreRoutes)
	})
}
