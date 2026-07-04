// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestRegisterROCmModelTokenizerRoute_Good_ExtendsReactiveTokenizerRegistry(t *testing.T) {
	restoreRegisteredROCmModelTokenizerRoutesForTest(t)

	RegisterROCmModelTokenizerRoute(ROCmModelTokenizerRoute{})
	RegisterROCmModelTokenizerRoute(ROCmModelTokenizerRoute{
		Architecture:  "fake-loader",
		TokenizerKind: "FirstTokenizer",
	})
	RegisterROCmModelTokenizerRoute(ROCmModelTokenizerRoute{
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

	if got := RegisteredROCmModelTokenizerRouteArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmModelTokenizerRouteArchitectures = %v, want normalized replacement under one architecture", got)
	}
	registered := RegisteredROCmModelTokenizerRouteArchitectures()
	registered[0] = "mutated"
	if next := RegisteredROCmModelTokenizerRouteArchitectures(); !slices.Equal(next, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmModelTokenizerRouteArchitectures returned mutable state: %v", next)
	}

	route, ok := ROCmModelTokenizerRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != ROCmModelTokenizerRegistryContract ||
		route.Name != rocmModelTokenizerRegistryRouteName ||
		route.Architecture != "fake_loader" ||
		route.Family != "fake" ||
		route.Loader != rocmModelTokenizerLoaderHFJSON ||
		route.Runtime != rocmModelTokenizerRuntimeHost ||
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
		route.Labels["engine_tokenizer_loader"] != rocmModelTokenizerLoaderHFJSON ||
		route.Labels["engine_tokenizer_kind"] != "FakeTokenizer" ||
		route.Labels["engine_tokenizer_chat_template_id"] != "fake-chat" ||
		route.Labels["engine_tokenizer_reasoning_parser"] != "fake-reasoning" ||
		route.Labels["engine_tokenizer_tool_parser"] != "fake-tool" {
		t.Fatalf("ROCmModelTokenizerRouteForArchitecture(fake-loader) = %+v ok=%v, want registered tokenizer route", route, ok)
	}

	route.Labels["engine_tokenizer_kind"] = "mutated"
	nextRoute, ok := ROCmModelTokenizerRouteForArchitecture("fake-loader")
	if !ok || nextRoute.Labels["engine_tokenizer_kind"] != "FakeTokenizer" {
		t.Fatalf("ROCmModelTokenizerRouteForArchitecture leaked mutable labels: %+v ok=%v", nextRoute, ok)
	}
	nextRoute.RequiredFiles[0] = "mutated"
	cleanRoute, ok := ROCmModelTokenizerRouteForArchitecture("fake-loader")
	if !ok || !slices.Equal(cleanRoute.RequiredFiles, []string{"tokenizer.json"}) {
		t.Fatalf("ROCmModelTokenizerRouteForArchitecture leaked mutable files: %+v ok=%v", cleanRoute, ok)
	}
	modelRoute, ok := rocmmodel.RegisteredTokenizerRouteForArchitecture("fake-loader")
	if !ok ||
		modelRoute.TokenizerKind != "FakeTokenizer" ||
		modelRoute.Labels["engine_tokenizer_kind"] != "FakeTokenizer" ||
		!slices.Equal(modelRoute.RequiredFiles, []string{"tokenizer.json"}) {
		t.Fatalf("model.RegisteredTokenizerRouteForArchitecture(fake-loader) = %+v ok=%v, want mirrored route", modelRoute, ok)
	}

	defaults := DefaultROCmModelTokenizerRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmModelTokenizerRoute) bool {
		return route.Architecture == "fake_loader" && route.TokenizerKind == "FakeTokenizer"
	}) {
		t.Fatalf("DefaultROCmModelTokenizerRoutes missing registered tokenizer route: %+v", defaults)
	}

	profile := ROCmModelProfile{
		Name:         "fake",
		Family:       "fake",
		Architecture: "fake_loader",
		ArchitectureProfile: ROCmArchitectureProfile{
			ID:                   "fake_loader",
			Family:               "fake",
			RuntimeStatus:        inference.FeatureRuntimeNative,
			ParserID:             "profile-reasoning",
			ToolParserID:         "profile-tool",
			ChatTemplate:         "profile-chat",
			GenerationRole:       "assistant",
			RequiresChatTemplate: true,
			NativeRuntime:        true,
			Generation:           true,
			Chat:                 true,
		},
	}
	profileRoute := ROCmModelTokenizerRouteForProfile(profile)
	if profileRoute.Loader != rocmModelTokenizerLoaderHFJSON ||
		profileRoute.TokenizerKind != "FakeTokenizer" ||
		profileRoute.ChatTemplateID != "fake-chat" ||
		profileRoute.ReasoningParserID != "fake-reasoning" ||
		profileRoute.ToolParserID != "fake-tool" ||
		!profileRoute.Registered ||
		!profileRoute.NativeRuntime ||
		profileRoute.Labels["engine_tokenizer_kind"] != "FakeTokenizer" {
		t.Fatalf("ROCmModelTokenizerRouteForProfile registered route = %+v, want profile to use registered tokenizer", profileRoute)
	}
}

func TestRegisterROCmModelTokenizerRoute_Good_SeesModelPackageRegistration(t *testing.T) {
	restoreRegisteredROCmModelTokenizerRoutesForTest(t)

	rocmmodel.RegisterTokenizerRoute(rocmmodel.TokenizerRoute{
		Architecture:         "folder-tokenizer",
		Family:               "folder",
		TokenizerKind:        "FolderTokenizer",
		TokenizerPath:        "tokenizer.json",
		ConfigPath:           "tokenizer_config.json",
		ChatTemplateID:       "folder-chat",
		ReasoningParserID:    "folder-reasoning",
		ToolParserID:         "folder-tool",
		GenerationRole:       "assistant",
		NativeRuntime:        true,
		RequiresChatTemplate: true,
		Generation:           true,
		Chat:                 true,
		Tokenizer: inference.TokenizerIdentity{
			Path: "tokenizer.json",
		},
	})

	route, ok := ROCmModelTokenizerRouteForArchitecture("folder-tokenizer")
	if !ok ||
		route.Contract != ROCmModelTokenizerRegistryContract ||
		route.Name != rocmModelTokenizerRegistryRouteName ||
		route.Architecture != "folder_tokenizer" ||
		route.Family != "folder" ||
		route.Loader != rocmModelTokenizerLoaderHFJSON ||
		route.Runtime != rocmModelTokenizerRuntimeHost ||
		route.TokenizerKind != "FolderTokenizer" ||
		route.TokenizerPath != "tokenizer.json" ||
		route.ConfigPath != "tokenizer_config.json" ||
		route.ChatTemplateID != "folder-chat" ||
		route.ReasoningParserID != "folder-reasoning" ||
		route.ToolParserID != "folder-tool" ||
		!route.NativeRuntime ||
		!route.ChatTemplate ||
		!route.RequiresChatTemplate ||
		!route.ModelOwnedTemplate ||
		!route.Generation ||
		!route.Chat ||
		!slices.Contains(route.Capabilities, inference.CapabilityTokenizer) ||
		!slices.Contains(route.Capabilities, inference.CapabilityChatTemplate) ||
		route.Labels["engine_tokenizer_kind"] != "FolderTokenizer" {
		t.Fatalf("ROCmModelTokenizerRouteForArchitecture(folder-tokenizer) = %+v ok=%v, want model package route", route, ok)
	}
	defaults := DefaultROCmModelTokenizerRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmModelTokenizerRoute) bool {
		return route.Architecture == "folder_tokenizer" && route.TokenizerKind == "FolderTokenizer"
	}) {
		t.Fatalf("DefaultROCmModelTokenizerRoutes missing model package route: %+v", defaults)
	}
	profileRoute := ROCmModelTokenizerRouteForProfile(ROCmModelProfile{
		Name:         "folder",
		Family:       "folder",
		Architecture: "folder_tokenizer",
	})
	if profileRoute.TokenizerKind != "FolderTokenizer" ||
		profileRoute.ChatTemplateID != "folder-chat" ||
		profileRoute.ReasoningParserID != "folder-reasoning" ||
		profileRoute.ToolParserID != "folder-tool" ||
		!profileRoute.NativeRuntime ||
		!profileRoute.ChatTemplate {
		t.Fatalf("ROCmModelTokenizerRouteForProfile(folder_tokenizer) = %+v, want model package tokenizer route", profileRoute)
	}
}

func TestRegisterROCmModelTokenizerRoute_Good_OverridesBuiltinTokenizerRoute(t *testing.T) {
	restoreRegisteredROCmModelTokenizerRoutesForTest(t)

	RegisterROCmModelTokenizerRoute(ROCmModelTokenizerRoute{
		Architecture:      "qwen3",
		Family:            "qwen",
		Loader:            "registered-tokenizer-loader",
		TokenizerKind:     "RegisteredQwenTokenizer",
		ChatTemplateID:    "registered-qwen-chat",
		ReasoningParserID: "registered-qwen-reasoning",
		ToolParserID:      "registered-qwen-tool",
	})

	route, ok := ROCmModelTokenizerRouteForArchitecture("Qwen3ForCausalLM")
	if !ok ||
		route.Architecture != "qwen3" ||
		route.Family != "qwen" ||
		route.Loader != "registered-tokenizer-loader" ||
		route.TokenizerKind != "RegisteredQwenTokenizer" ||
		route.ChatTemplateID != "registered-qwen-chat" ||
		route.ReasoningParserID != "registered-qwen-reasoning" ||
		route.ToolParserID != "registered-qwen-tool" ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.ChatTemplate ||
		!route.RequiresChatTemplate ||
		!route.Generation ||
		!route.Chat ||
		route.Labels["engine_tokenizer_loader"] != "registered-tokenizer-loader" ||
		route.Labels["engine_tokenizer_kind"] != "RegisteredQwenTokenizer" ||
		route.Labels["engine_tokenizer_chat_template_id"] != "registered-qwen-chat" {
		t.Fatalf("ROCmModelTokenizerRouteForArchitecture(qwen3 override) = %+v ok=%v, want registered override", route, ok)
	}
}

func TestROCmModelTokenizerRoute_Good_Gemma4ThoughtChannelRegistration(t *testing.T) {
	route, ok := ROCmModelTokenizerRouteForArchitecture("Gemma4ForCausalLM")
	if !ok ||
		route.Architecture != "gemma4_text" ||
		!route.ThinkingChannel ||
		route.ThinkingChannelOpen != "<|channel>" ||
		route.ThinkingChannelClose != "<channel|>" ||
		route.Labels["engine_tokenizer_thinking_channel"] != "true" ||
		route.Labels["engine_tokenizer_thinking_channel_open"] != "<|channel>" ||
		route.Labels["engine_tokenizer_thinking_channel_close"] != "<channel|>" {
		t.Fatalf("ROCmModelTokenizerRouteForArchitecture(Gemma4ForCausalLM) = %+v ok=%v, want Gemma4 thought-channel route", route, ok)
	}

	profile, ok := ResolveROCmModelProfile("/models/gemma4", inference.ModelIdentity{Architecture: "Gemma4ForCausalLM"})
	if !ok ||
		!profile.TokenizerRoute.ThinkingChannel ||
		profile.TokenizerRoute.Labels["engine_tokenizer_thinking_channel"] != "true" {
		t.Fatalf("ResolveROCmModelProfile(Gemma4).TokenizerRoute = %+v ok=%v, want thought-channel metadata", profile.TokenizerRoute, ok)
	}
}

func restoreRegisteredROCmModelTokenizerRoutesForTest(t *testing.T) {
	t.Helper()
	modelRoutes := rocmmodel.RegisteredTokenizerRoutes()
	rocmmodel.ReplaceRegisteredTokenizerRoutes(nil)

	t.Cleanup(func() {
		restoreRoutes := make([]rocmmodel.TokenizerRoute, 0, len(modelRoutes))
		for _, route := range modelRoutes {
			restoreRoutes = append(restoreRoutes, route.Clone())
		}
		rocmmodel.ReplaceRegisteredTokenizerRoutes(restoreRoutes)
	})
}
