// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestDefaultTokenizerRoutes_Good_ArchitectureCatalogue(t *testing.T) {
	routes := DefaultTokenizerRoutes()
	if len(routes) == 0 {
		t.Fatal("DefaultTokenizerRoutes returned no routes")
	}
	byArchitecture := map[string]TokenizerRoute{}
	for _, route := range routes {
		byArchitecture[route.Architecture] = route
	}
	gemma4 := byArchitecture["gemma4"]
	if !gemma4.Matched() ||
		gemma4.Contract != TokenizerRegistryContract ||
		gemma4.Name != TokenizerRouteName ||
		gemma4.Loader != TokenizerLoaderHFJSON ||
		gemma4.Runtime != TokenizerRuntimeHost ||
		gemma4.TokenizerKind != "GemmaTokenizer" ||
		gemma4.ChatTemplateID == "" ||
		gemma4.ChatTemplateSource != "registry" ||
		!gemma4.Registered ||
		!gemma4.NativeRuntime ||
		!gemma4.ChatTemplate ||
		!gemma4.RequiresChatTemplate ||
		!gemma4.ModelOwnedTemplate ||
		!gemma4.Generation ||
		!gemma4.Chat ||
		!slices.Contains(gemma4.RequiredFiles, "tokenizer.json") ||
		!slices.Contains(gemma4.Capabilities, inference.CapabilityTokenizer) ||
		!slices.Contains(gemma4.Capabilities, inference.CapabilityChatTemplate) ||
		gemma4.Tokenizer.Kind != "GemmaTokenizer" ||
		gemma4.Tokenizer.ChatTemplate != gemma4.ChatTemplateID ||
		gemma4.Labels["engine_tokenizer_architecture"] != "gemma4" {
		t.Fatalf("gemma4 tokenizer route = %+v, want native tokenizer/template route", gemma4)
	}
	bert := byArchitecture["bert"]
	if !bert.Matched() ||
		bert.TokenizerKind != "BertTokenizer" ||
		bert.ChatTemplate ||
		bert.Generation ||
		bert.Chat ||
		!slices.Contains(bert.Capabilities, inference.CapabilityTokenizer) {
		t.Fatalf("bert tokenizer route = %+v, want tokenizer-only route", bert)
	}
}

func TestTokenizerRouteForArchitecture_Good_NormalizesAliases(t *testing.T) {
	route, ok := TokenizerRouteForArchitecture("Gemma4ForConditionalGeneration")
	if !ok ||
		route.Architecture != "gemma4" ||
		route.Family != "gemma4" ||
		route.TokenizerKind != "GemmaTokenizer" ||
		!route.ChatTemplate {
		t.Fatalf("TokenizerRouteForArchitecture(Gemma4ForConditionalGeneration) = %+v ok=%v, want gemma4 route", route, ok)
	}
	if _, ok := TokenizerRouteForArchitecture("totally_unknown_tokenizer_route"); ok {
		t.Fatal("TokenizerRouteForArchitecture(unknown) ok = true, want false")
	}
}

func TestTokenizerRouteForInspection_Good_MergesIdentityCopySafe(t *testing.T) {
	inspectionLabels := map[string]string{
		"architecture_resolved": "qwen3",
		"tokenizer_json":        "present",
		"tokenizer_config":      "present",
	}
	modelLabels := map[string]string{"engine_architecture_resolved": "gemma4_text"}
	inspection := &inference.ModelPackInspection{
		Path:   "/models/gemma4",
		Labels: inspectionLabels,
		Model: inference.ModelIdentity{
			Architecture: "Qwen3ForCausalLM",
			Labels:       modelLabels,
		},
		Tokenizer: inference.TokenizerIdentity{
			Kind:         "sidecar-tokenizer",
			Path:         "tokenizer.json",
			ChatTemplate: "sidecar-template",
			BOSID:        2,
			EOSID:        1,
			PADID:        0,
			Labels:       map[string]string{"tokenizer": "sidecar"},
		},
	}
	route, ok := TokenizerRouteForInspection(inspection)
	if !ok ||
		route.Architecture != "gemma4_text" ||
		route.TokenizerKind != "sidecar-tokenizer" ||
		route.TokenizerPath != "tokenizer.json" ||
		route.ChatTemplateID != "gemma4_hf_turn" ||
		route.ChatTemplateSource != "sidecar" ||
		!route.SidecarTokenizer ||
		!route.SidecarConfig ||
		!route.SidecarTemplate ||
		route.BOSID != 2 ||
		route.EOSID != 1 ||
		route.Tokenizer.Labels["tokenizer"] != "sidecar" {
		t.Fatalf("TokenizerRouteForInspection = %+v ok=%v, want sidecar identity merged into route", route, ok)
	}
	inspectionLabels["architecture_resolved"] = "mutated"
	modelLabels["engine_architecture_resolved"] = "mutated"
	route.Tokenizer.Labels["tokenizer"] = "mutated"
	route.RequiredFiles[0] = "mutated"
	next, ok := TokenizerRouteForArchitecture("gemma4_text")
	if !ok ||
		next.Tokenizer.Labels["tokenizer"] == "mutated" ||
		next.RequiredFiles[0] != "tokenizer.json" {
		t.Fatalf("TokenizerRouteForInspection leaked mutable state: %+v ok=%v", next, ok)
	}
}

func TestRegisterTokenizerRoute_Good_ExtendsReactiveCatalogue(t *testing.T) {
	restoreRegisteredTokenizersForTest(t)

	RegisterTokenizerRoute(TokenizerRoute{})
	RegisterTokenizerRoute(TokenizerRoute{Architecture: "fake-loader", TokenizerKind: "FirstTokenizer"})
	RegisterTokenizerRoute(TokenizerRoute{
		Architecture:         "fake-loader",
		Family:               "fake",
		TokenizerKind:        "FakeTokenizer",
		TokenizerPath:        "tokenizer.json",
		ConfigPath:           "tokenizer_config.json",
		ChatTemplateID:       "fake-chat",
		ReasoningParserID:    "fake-reasoning",
		ToolParserID:         "fake-tool",
		GenerationRole:       "assistant",
		NativeRuntime:        true,
		RequiresChatTemplate: true,
		Generation:           true,
		Chat:                 true,
		Tokenizer: inference.TokenizerIdentity{
			Path: "tokenizer.json",
		},
	})

	if got := RegisteredTokenizerArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredTokenizerArchitectures = %v, want normalized replacement", got)
	}
	registeredRoutes := RegisteredTokenizerRoutes()
	if len(registeredRoutes) != 1 ||
		registeredRoutes[0].Architecture != "fake_loader" ||
		registeredRoutes[0].TokenizerKind != "FakeTokenizer" {
		t.Fatalf("RegisteredTokenizerRoutes = %+v, want one replacement route", registeredRoutes)
	}
	registeredRoute, ok := RegisteredTokenizerRouteForArchitecture("fake-loader")
	if !ok || registeredRoute.TokenizerKind != "FakeTokenizer" {
		t.Fatalf("RegisteredTokenizerRouteForArchitecture(fake-loader) = %+v ok=%v, want registered route", registeredRoute, ok)
	}
	registeredRoutes[0].Labels["engine_tokenizer_kind"] = "mutated"
	nextRegisteredRoute, ok := RegisteredTokenizerRouteForArchitecture("fake-loader")
	if !ok || nextRegisteredRoute.Labels["engine_tokenizer_kind"] != "FakeTokenizer" {
		t.Fatalf("RegisteredTokenizerRoutes leaked mutable labels: %+v ok=%v", nextRegisteredRoute, ok)
	}
	ReplaceRegisteredTokenizerRoutes(registeredRoutes)
	replaced, ok := RegisteredTokenizerRouteForArchitecture("fake-loader")
	if !ok || replaced.Labels["engine_tokenizer_kind"] != "FakeTokenizer" {
		t.Fatalf("ReplaceRegisteredTokenizerRoutes = %+v ok=%v, want copy-safe restored route", replaced, ok)
	}

	route, ok := TokenizerRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != TokenizerRegistryContract ||
		route.Name != TokenizerRouteName ||
		route.Architecture != "fake_loader" ||
		route.Family != "fake" ||
		route.Loader != TokenizerLoaderHFJSON ||
		route.Runtime != TokenizerRuntimeHost ||
		route.TokenizerKind != "FakeTokenizer" ||
		route.TokenizerPath != "tokenizer.json" ||
		route.ConfigPath != "tokenizer_config.json" ||
		route.ChatTemplateID != "fake-chat" ||
		route.ChatTemplateSource != "registry" ||
		route.ReasoningParserID != "fake-reasoning" ||
		route.ToolParserID != "fake-tool" ||
		route.GenerationRole != "assistant" ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.ChatTemplate ||
		!route.RequiresChatTemplate ||
		!route.ModelOwnedTemplate ||
		!route.Generation ||
		!route.Chat ||
		!slices.Equal(route.RequiredFiles, []string{"tokenizer.json"}) ||
		!slices.Contains(route.Capabilities, inference.CapabilityTokenizer) ||
		!slices.Contains(route.Capabilities, inference.CapabilityChatTemplate) ||
		route.Tokenizer.Kind != "FakeTokenizer" ||
		route.Tokenizer.Path != "tokenizer.json" ||
		route.Tokenizer.ChatTemplate != "fake-chat" ||
		route.Labels["engine_tokenizer_kind"] != "FakeTokenizer" {
		t.Fatalf("TokenizerRouteForArchitecture(fake-loader) = %+v ok=%v, want registered tokenizer route", route, ok)
	}
	route.Labels["engine_tokenizer_kind"] = "mutated"
	route.RequiredFiles[0] = "mutated"
	next, ok := TokenizerRouteForArchitecture("fake-loader")
	if !ok ||
		next.Labels["engine_tokenizer_kind"] != "FakeTokenizer" ||
		next.RequiredFiles[0] != "tokenizer.json" {
		t.Fatalf("TokenizerRouteForArchitecture leaked mutable state: %+v ok=%v", next, ok)
	}
	if !slices.ContainsFunc(DefaultTokenizerRoutes(), func(route TokenizerRoute) bool {
		return route.Architecture == "fake_loader" && route.TokenizerKind == "FakeTokenizer"
	}) {
		t.Fatalf("DefaultTokenizerRoutes missing fake_loader registration")
	}
}

func restoreRegisteredTokenizersForTest(t *testing.T) {
	t.Helper()
	order, routes := registeredTokenizers.Snapshot()
	for architecture, route := range routes {
		routes[architecture] = route.Clone()
	}
	t.Cleanup(func() {
		restoreRoutes := make(map[string]TokenizerRoute, len(routes))
		for architecture, route := range routes {
			restoreRoutes[architecture] = route.Clone()
		}
		registeredTokenizers.Restore(order, restoreRoutes)
	})
}
