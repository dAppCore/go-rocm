// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestDefaultLoaderRoutes_Good_ArchitectureCatalogue(t *testing.T) {
	routes := DefaultLoaderRoutes()
	if len(routes) == 0 {
		t.Fatal("DefaultLoaderRoutes returned no routes")
	}
	byArchitecture := map[string]LoaderRoute{}
	for _, route := range routes {
		byArchitecture[route.Architecture] = route
	}
	gemma4 := byArchitecture["gemma4"]
	if !gemma4.Matched() ||
		gemma4.Contract != LoaderRegistryContract ||
		gemma4.Runtime != RuntimeHIP ||
		gemma4.Status != StatusStandaloneNative ||
		gemma4.Target != "standalone" ||
		!gemma4.Registered ||
		!gemma4.NativeRuntime ||
		!gemma4.Standalone ||
		!gemma4.TextGenerate ||
		gemma4.Labels["engine_loader_architecture"] != "gemma4" {
		t.Fatalf("gemma4 loader route = %+v, want native standalone route", gemma4)
	}
	bert := byArchitecture["bert"]
	if !bert.Matched() ||
		bert.Runtime != RuntimeHIP ||
		bert.Status != StatusStagedNative ||
		!bert.Staged ||
		!bert.Standalone ||
		bert.TextGenerate {
		t.Fatalf("bert loader route = %+v, want staged native non-generation route", bert)
	}
	gptOSS := byArchitecture["gpt-oss"]
	if !gptOSS.Matched() ||
		gptOSS.Architecture != "gpt-oss" ||
		gptOSS.Loader != "gpt_oss" ||
		gptOSS.Labels["engine_loader"] != "gpt_oss" {
		t.Fatalf("gpt-oss loader route = %+v, want go-mlx loader token gpt_oss", gptOSS)
	}
}

func TestDefaultLoaderRoutes_Good_MatchesGoMLXModelLoaderTokens(t *testing.T) {
	routes := DefaultLoaderRoutes()
	byArchitecture := map[string]LoaderRoute{}
	for _, route := range routes {
		byArchitecture[route.Architecture] = route
	}
	wantLoaders := map[string]string{
		"bert":            "bert",
		"bert_rerank":     "bert_rerank",
		"composed":        "composed",
		"deepseek":        "deepseek",
		"deltanet":        "deltanet",
		"diffusion_gemma": "diffusion_gemma",
		"gemma2":          "gemma2",
		"gemma3":          "gemma3",
		"gemma3_text":     "gemma3_text",
		"gemma4":          "gemma4",
		"gemma4_text":     "gemma4_text",
		"gemma4_unified":  "gemma4_unified",
		"glm":             "glm",
		"glm4":            "glm",
		"gpt-oss":         "gpt_oss",
		"granite":         "granite",
		"gla":             "gla",
		"gsa":             "gsa",
		"hermes":          "hermes",
		"hybrid":          "hybrid",
		"kimi":            "kimi",
		"llama":           "llama",
		"mamba2":          "mamba2",
		"minimax_m2":      "minimax_m2",
		"mistral":         "mistral",
		"mixtral":         "mixtral",
		"mla":             "mla",
		"moba":            "moba",
		"nsa":             "nsa",
		"phi":             "phi",
		"qwen2":           "qwen2",
		"qwen3":           "qwen3",
		"qwen3_6":         "qwen3_6",
		"qwen3_6_moe":     "qwen3_6_moe",
		"qwen3_moe":       "qwen3_moe",
		"qwen3_next":      "qwen3_next",
		"retnet":          "retnet",
		"rwkv7":           "rwkv7",
	}
	for architecture, loader := range wantLoaders {
		route, ok := byArchitecture[architecture]
		if !ok || !route.Matched() {
			t.Fatalf("DefaultLoaderRoutes missing go-mlx loader token %s for architecture %s", loader, architecture)
		}
		if route.Loader != loader || route.Labels["engine_loader"] != loader {
			t.Fatalf("DefaultLoaderRoutes[%s] = %+v, want loader token %q", architecture, route, loader)
		}
	}
}

func TestLoaderRouteForArchitecture_Good_NormalizesAliases(t *testing.T) {
	route, ok := LoaderRouteForArchitecture("Gemma4ForConditionalGeneration")
	if !ok ||
		route.Architecture != "gemma4" ||
		route.Family != "gemma4" ||
		route.RuntimeStatus != inference.FeatureRuntimeNative ||
		route.Status != StatusStandaloneNative {
		t.Fatalf("LoaderRouteForArchitecture(Gemma4ForConditionalGeneration) = %+v ok=%v, want gemma4 route", route, ok)
	}
	if _, ok := LoaderRouteForArchitecture("totally_unknown_loader"); ok {
		t.Fatal("LoaderRouteForArchitecture(unknown) ok = true, want false")
	}
	gptOSS, ok := LoaderRouteForArchitecture("GPTOSSForCausalLM")
	if !ok ||
		gptOSS.Architecture != "gpt-oss" ||
		gptOSS.Loader != "gpt_oss" ||
		gptOSS.Family != "gpt-oss" {
		t.Fatalf("LoaderRouteForArchitecture(GPTOSSForCausalLM) = %+v ok=%v, want gpt-oss architecture with gpt_oss loader token", gptOSS, ok)
	}
	glm4, ok := LoaderRouteForArchitecture("ChatGLM4ForConditionalGeneration")
	if !ok ||
		glm4.Architecture != "glm4" ||
		glm4.Loader != "glm" ||
		glm4.Family != "glm" {
		t.Fatalf("LoaderRouteForArchitecture(ChatGLM4ForConditionalGeneration) = %+v ok=%v, want glm4 architecture with glm loader token", glm4, ok)
	}
	gemma3Text, ok := LoaderRouteForArchitecture("Gemma3TextForCausalLM")
	if !ok ||
		gemma3Text.Architecture != "gemma3_text" ||
		gemma3Text.Loader != "gemma3_text" ||
		gemma3Text.Family != "gemma" {
		t.Fatalf("LoaderRouteForArchitecture(Gemma3TextForCausalLM) = %+v ok=%v, want Gemma 3 text loader route", gemma3Text, ok)
	}
	mamba2, ok := LoaderRouteForArchitecture("Mamba2ForCausalLM")
	if !ok ||
		mamba2.Architecture != "mamba2" ||
		mamba2.Loader != "mamba2" ||
		mamba2.Runtime != RuntimeMetadata ||
		mamba2.RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		mamba2.Status != StatusMetadataOnly ||
		mamba2.NativeRuntime ||
		mamba2.TextGenerate {
		t.Fatalf("LoaderRouteForArchitecture(Mamba2ForCausalLM) = %+v ok=%v, want metadata-only go-mlx parity route", mamba2, ok)
	}
}

func TestLoaderRouteForIdentity_Good_ResolvedArchitectureLabelsWin(t *testing.T) {
	route, ok := LoaderRouteForIdentity("/models/gemma4", inference.ModelIdentity{
		Architecture: "Gemma4ForConditionalGeneration",
		Labels: map[string]string{
			"engine_architecture_resolved": "gemma4_text",
		},
	})
	if !ok ||
		route.Architecture != "gemma4_text" ||
		route.Family != "gemma4" ||
		route.Status != StatusStandaloneNative {
		t.Fatalf("LoaderRouteForIdentity = %+v ok=%v, want resolved Gemma4 text route", route, ok)
	}
}

func TestLoaderRouteForInspection_Good_MergesLabelsCopySafe(t *testing.T) {
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
	route, ok := LoaderRouteForInspection(inspection)
	if !ok ||
		route.Architecture != "bert_rerank" ||
		route.Family != "bert" ||
		route.Status != StatusStagedNative ||
		route.TextGenerate {
		t.Fatalf("LoaderRouteForInspection = %+v ok=%v, want model label refined BERT rerank route", route, ok)
	}
	inspectionLabels["architecture_resolved"] = "mutated"
	modelLabels["engine_architecture_resolved"] = "mutated"
	route.Labels["engine_loader_architecture"] = "mutated"
	next, ok := LoaderRouteForArchitecture("bert_rerank")
	if !ok || next.Labels["engine_loader_architecture"] != "bert_rerank" {
		t.Fatalf("LoaderRouteForInspection leaked mutable route labels: %+v ok=%v", next, ok)
	}
}

func TestLoaderRouteForInfo_Good_UsesLabels(t *testing.T) {
	route, ok := LoaderRouteForInfo("/models/qwen-next", inference.ModelInfo{
		Architecture: "Qwen3ForCausalLM",
	}, map[string]string{"architecture_resolved": "qwen3_next"})
	if !ok ||
		route.Architecture != "qwen3_next" ||
		route.Family != "qwen" {
		t.Fatalf("LoaderRouteForInfo = %+v ok=%v, want label-refined qwen3_next route", route, ok)
	}
}

func TestRegisterLoaderRoute_Good_ExtendsReactiveCatalogue(t *testing.T) {
	restoreRegisteredLoadersForTest(t)

	RegisterLoaderRoute(LoaderRoute{})
	RegisterLoaderRoute(LoaderRoute{Architecture: "fake-loader", Loader: "first_loader", Status: StatusStagedNative})
	RegisterLoaderRoute(LoaderRoute{
		Architecture: "fake-loader",
		Family:       "fake",
		Loader:       "fake_hip_loader",
		Status:       StatusStandaloneNative,
		Reason:       "registered test route",
	})

	if got := RegisteredLoaderArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredLoaderArchitectures = %v, want normalized replacement", got)
	}
	registeredRoutes := RegisteredLoaderRoutes()
	if len(registeredRoutes) != 1 ||
		registeredRoutes[0].Architecture != "fake_loader" ||
		registeredRoutes[0].Loader != "fake_hip_loader" {
		t.Fatalf("RegisteredLoaderRoutes = %+v, want one replacement route", registeredRoutes)
	}
	registeredRoute, ok := RegisteredLoaderRouteForArchitecture("fake-loader")
	if !ok || registeredRoute.Loader != "fake_hip_loader" {
		t.Fatalf("RegisteredLoaderRouteForArchitecture(fake-loader) = %+v ok=%v, want registered route", registeredRoute, ok)
	}
	registered := RegisteredLoaderArchitectures()
	registered[0] = "mutated"
	if next := RegisteredLoaderArchitectures(); !slices.Equal(next, []string{"fake_loader"}) {
		t.Fatalf("RegisteredLoaderArchitectures returned mutable state: %v", next)
	}
	registeredRoutes[0].Labels["engine_loader"] = "mutated"
	nextRegisteredRoute, ok := RegisteredLoaderRouteForArchitecture("fake-loader")
	if !ok || nextRegisteredRoute.Labels["engine_loader"] != "fake_hip_loader" {
		t.Fatalf("RegisteredLoaderRoutes leaked mutable labels: %+v ok=%v", nextRegisteredRoute, ok)
	}
	ReplaceRegisteredLoaderRoutes(registeredRoutes)
	replaced, ok := RegisteredLoaderRouteForArchitecture("fake-loader")
	if !ok || replaced.Labels["engine_loader"] != "fake_hip_loader" {
		t.Fatalf("ReplaceRegisteredLoaderRoutes = %+v ok=%v, want copy-safe restored route", replaced, ok)
	}

	route, ok := LoaderRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != LoaderRegistryContract ||
		route.Architecture != "fake_loader" ||
		route.Family != "fake" ||
		route.Loader != "fake_hip_loader" ||
		route.Runtime != RuntimeHIP ||
		route.Status != StatusStandaloneNative ||
		route.Target != "standalone" ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.Standalone ||
		!route.TextGenerate ||
		route.Labels["engine_loader_reason"] != "registered test route" {
		t.Fatalf("LoaderRouteForArchitecture(fake-loader) = %+v ok=%v, want registered route", route, ok)
	}
	route.Labels["engine_loader"] = "mutated"
	next, ok := LoaderRouteForArchitecture("fake-loader")
	if !ok || next.Labels["engine_loader"] != "fake_hip_loader" {
		t.Fatalf("LoaderRouteForArchitecture leaked mutable labels: %+v ok=%v", next, ok)
	}
	if !slices.Contains(LoaderArchitectures(), "fake_loader") {
		t.Fatalf("LoaderArchitectures missing fake_loader: %v", LoaderArchitectures())
	}
}

func restoreRegisteredLoadersForTest(t *testing.T) {
	t.Helper()
	order, routes := registeredLoaders.Snapshot()
	for architecture, route := range routes {
		routes[architecture] = route.Clone()
	}
	t.Cleanup(func() {
		restoreRoutes := make(map[string]LoaderRoute, len(routes))
		for architecture, route := range routes {
			restoreRoutes[architecture] = route.Clone()
		}
		registeredLoaders.Restore(order, restoreRoutes)
	})
}
