// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

type rocmModelProfileFactoryTestFactory struct {
	name    string
	payload string
}

func (factory rocmModelProfileFactoryTestFactory) Name() string { return factory.name }

func (factory rocmModelProfileFactoryTestFactory) BuildROCmModelProfile(req ROCmModelProfileRequest) (ROCmModelProfile, bool) {
	if req.Model.Labels["registry_factory_test"] != "true" {
		return ROCmModelProfile{}, false
	}
	architectureProfile, ok := ROCmArchitectureProfileForArchitecture("qwen3")
	if !ok {
		return ROCmModelProfile{}, false
	}
	cacheRoute, _ := ROCmCacheRouteForArchitecture("qwen3")
	model := req.Model
	model.Path = firstNonEmptyString(model.Path, req.Path)
	model.Architecture = "qwen3"
	return ROCmModelProfile{
		Name:                "registered-qwen",
		Family:              "qwen",
		Architecture:        "qwen3",
		Model:               model,
		ArchitectureProfile: architectureProfile,
		CacheRoute:          cacheRoute,
		Labels: map[string]string{
			"factory_payload": factory.payload,
		},
	}, true
}

type blankROCmModelProfileFactoryTestFactory struct{}

func (blankROCmModelProfileFactoryTestFactory) Name() string { return " " }
func (blankROCmModelProfileFactoryTestFactory) BuildROCmModelProfile(ROCmModelProfileRequest) (ROCmModelProfile, bool) {
	return ROCmModelProfile{Name: "blank"}, true
}

type modelPackageRouteSetProfileFactoryTestFactory struct{}

func (modelPackageRouteSetProfileFactoryTestFactory) Name() string {
	return "model-package-route-set"
}

func (modelPackageRouteSetProfileFactoryTestFactory) BuildModelProfile(req rocmmodel.ProfileRequest) (rocmmodel.Profile, bool) {
	if req.Model.Labels["model_route_set_test"] != "true" {
		return rocmmodel.Profile{}, false
	}
	identity := req.Model
	identity.Path = firstNonEmptyString(identity.Path, req.Path)
	identity.Architecture = "model_route_set"
	identity.Labels = cloneStringMap(identity.Labels)
	if identity.Labels == nil {
		identity.Labels = map[string]string{}
	}
	identity.Labels["route_set_identity"] = "kept"
	routeSet := rocmmodel.RouteSet{
		Contract:     rocmmodel.RouteSetContract,
		Architecture: "model_route_set",
		Family:       "model-route",
		Model:        identity,
		FeatureRoute: rocmmodel.FeatureRoute{
			Contract:           rocmmodel.FeatureRegistryContract,
			Name:               rocmmodel.FeatureRouteName,
			Architecture:       "model_route_set",
			Family:             "model-route",
			RuntimeStatus:      inference.FeatureRuntimeNative,
			ReasoningParserID:  "route-reasoning-parser",
			ChatTemplateID:     "route-chat-template",
			GenerationRole:     "assistant",
			Registered:         true,
			NativeRuntime:      true,
			Generation:         true,
			TextGenerate:       true,
			Chat:               true,
			ModelContextWindow: true,
		},
		TokenizerRoute: rocmmodel.TokenizerRoute{
			Contract:       rocmmodel.TokenizerRegistryContract,
			Name:           rocmmodel.TokenizerRouteName,
			Architecture:   "model_route_set",
			Family:         "model-route",
			Loader:         "route-tokenizer-loader",
			Runtime:        rocmmodel.TokenizerRuntimeHost,
			TokenizerKind:  "RouteTokenizer",
			ChatTemplateID: "route-chat-template",
			GenerationRole: "assistant",
			Registered:     true,
			NativeRuntime:  true,
			ChatTemplate:   true,
			Generation:     true,
			Chat:           true,
		},
		LoaderRoute: rocmmodel.LoaderRoute{
			Contract:      rocmmodel.LoaderRegistryContract,
			Name:          "architecture-loader",
			Architecture:  "model_route_set",
			Family:        "model-route",
			Loader:        "route_hip_loader",
			Runtime:       rocmmodel.RuntimeHIP,
			Status:        rocmmodel.StatusStandaloneNative,
			Target:        "standalone",
			RuntimeStatus: inference.FeatureRuntimeNative,
			Reason:        "model package route-set loader",
			Registered:    true,
			NativeRuntime: true,
			Standalone:    true,
			TextGenerate:  true,
		},
		QuantLoaderRoute: rocmmodel.QuantLoaderRoute{
			Contract:       rocmmodel.QuantLoaderRegistryContract,
			Name:           rocmmodel.QuantLoaderRouteName,
			Family:         "model-route",
			Architecture:   "model_route_set",
			Pack:           "route-pack:q8",
			Mode:           "q8",
			Loader:         "route_quant_loader",
			Runtime:        rocmmodel.QuantRuntimeBF16,
			GenerateStatus: rocmmodel.QuantGenerateLinked,
			Target:         "generate",
			Registered:     true,
			NativeRuntime:  true,
			RunnableOnCard: true,
		},
		RuntimeContractRoute: rocmmodel.RuntimeContractRoute{
			Contract:          rocmmodel.RuntimeContractRegistryContract,
			Name:              rocmmodel.RuntimeContractRouteName,
			Architecture:      "model_route_set",
			Family:            "model-route",
			RuntimeStatus:     inference.FeatureRuntimeNative,
			Registered:        true,
			NativeRuntime:     true,
			TextGenerate:      true,
			GreedyToken:       true,
			ModelInfoReporter: true,
		},
		Labels: map[string]string{
			"route_set_payload": "kept",
		},
	}
	routeSet.CacheRoute = rocmmodel.NormalizeCacheRoute(rocmmodel.CacheRoute{
		Architecture:      "model_route_set",
		Family:            "model-route",
		RecommendedMode:   rocmmodel.CacheModeQ8,
		DeviceMode:        rocmmodel.CacheModeQ8,
		NativeRuntime:     true,
		SupportsKV:        true,
		SupportsDevice:    true,
		SupportsRecurrent: false,
	})
	return rocmmodel.Profile{
		Contract:     rocmmodel.ProfileFactoryRegistryContract,
		Name:         "model-package-route-set",
		Family:       "model-route",
		Architecture: "model_route_set",
		Registry:     rocmmodel.ProfileRegistryName,
		Model:        identity,
		RouteSet:     routeSet,
		Labels: map[string]string{
			"profile_payload": "kept",
		},
	}, true
}

func TestRegisterROCmModelProfileFactory_Good_PrependsReactiveFactory(t *testing.T) {
	restoreRegisteredROCmModelProfileFactoriesForTest(t)

	RegisterROCmModelProfileFactory(nil)
	RegisterROCmModelProfileFactory(blankROCmModelProfileFactoryTestFactory{})
	RegisterROCmModelProfileFactory(rocmModelProfileFactoryTestFactory{name: "test-reactive-factory", payload: "first"})
	RegisterROCmModelProfileFactory(rocmModelProfileFactoryTestFactory{name: "test-reactive-factory", payload: "second"})

	registered := RegisteredROCmModelProfileFactoryNames()
	if !slices.Equal(registered, []string{"gemma4", "test-reactive-factory", "architecture-profile"}) {
		t.Fatalf("RegisteredROCmModelProfileFactoryNames = %v, want extensions before the generic fallback", registered)
	}
	registered[0] = "mutated"
	if next := RegisteredROCmModelProfileFactoryNames(); !slices.Equal(next, []string{"gemma4", "test-reactive-factory", "architecture-profile"}) {
		t.Fatalf("RegisteredROCmModelProfileFactoryNames returned mutable state: %v", next)
	}
	if got := rocmmodel.RegisteredProfileFactoryNames(); !slices.Equal(got, []string{"gemma4", "architecture-profile", "test-reactive-factory"}) {
		t.Fatalf("model.RegisteredProfileFactoryNames = %v, want root factory bridged into model registry", got)
	}

	factories := defaultROCmModelProfileRegistry().FactoryNames()
	if !slices.Equal(factories, []string{"gemma4", "test-reactive-factory", "architecture-profile"}) {
		t.Fatalf("FactoryNames = %v, want model-owned Gemma4, registered factory, then generic fallback", factories)
	}

	callerLabels := map[string]string{"registry_factory_test": "true"}
	profile, ok := ResolveROCmModelProfile("/models/qwen", inference.ModelIdentity{
		Architecture: "qwen3",
		Labels:       callerLabels,
	})
	if !ok || !profile.Matched() ||
		profile.Name != "registered-qwen" ||
		profile.Registry != rocmModelRegistryName ||
		profile.Model.Path != "/models/qwen" ||
		profile.EngineFeatures.Contract != rocmEngineFeaturesContract ||
		profile.EngineFeatures.Architecture != "qwen3" ||
		!profile.EngineFeatures.TextGenerate ||
		!profile.CacheRoute.Matched() ||
		profile.CacheRoute.Architecture != "qwen3" ||
		profile.CacheRoute.Labels["engine_cache_factory_architecture"] != "qwen3" ||
		profile.Labels["factory_payload"] != "second" ||
		profile.Labels["engine_profile_source"] != "registered_factory" ||
		profile.Labels["engine_profile_factory"] != "test-reactive-factory" ||
		profile.Labels["engine_profile_reactive"] != "true" ||
		profile.Labels["engine_profile_architecture"] != "qwen3" {
		t.Fatalf("ResolveROCmModelProfile registered factory = %+v ok=%v, want reactive registered profile", profile, ok)
	}

	profile.Model.Labels["registry_factory_test"] = "mutated"
	if callerLabels["registry_factory_test"] != "true" {
		t.Fatalf("ResolveROCmModelProfile aliased caller labels: %+v", callerLabels)
	}
	profile.Labels["factory_payload"] = "mutated"
	profile.CacheRoute.Labels["engine_cache_factory_architecture"] = "mutated"
	next, ok := ResolveROCmModelProfile("/models/qwen", inference.ModelIdentity{
		Architecture: "qwen3",
		Labels:       callerLabels,
	})
	if !ok ||
		next.Labels["factory_payload"] != "second" ||
		next.CacheRoute.Labels["engine_cache_factory_architecture"] != "qwen3" {
		t.Fatalf("ResolveROCmModelProfile returned mutable registered profile data: %+v ok=%v", next, ok)
	}
}

func TestRegisteredModelProfileFactory_Good_PreservesModelOwnedRouteSet(t *testing.T) {
	restoreRegisteredROCmModelProfileFactoriesForTest(t)

	rocmmodel.RegisterProfileFactory(modelPackageRouteSetProfileFactoryTestFactory{})

	profile, ok := ResolveROCmModelProfile("/models/route-set", inference.ModelIdentity{
		Architecture: "model_route_set",
		Labels: map[string]string{
			"model_route_set_test": "true",
		},
	})
	if !ok || !profile.Matched() ||
		profile.Name != "model-package-route-set" ||
		profile.Model.Path != "/models/route-set" ||
		profile.Model.Labels["route_set_identity"] != "kept" ||
		profile.FeatureRoute.ReasoningParserID != "route-reasoning-parser" ||
		profile.FeatureRoute.ChatTemplateID != "route-chat-template" ||
		!profile.FeatureRoute.TextGenerate ||
		profile.TokenizerRoute.Loader != "route-tokenizer-loader" ||
		profile.TokenizerRoute.TokenizerKind != "RouteTokenizer" ||
		profile.LoadStatus.Loader != "route_hip_loader" ||
		profile.LoadStatus.Status != ROCmModelLoadStandaloneNative ||
		profile.LoadStatus.Labels["engine_load_reason"] != "model package route-set loader" ||
		!profile.CacheRoute.Matched() ||
		profile.CacheRoute.RecommendedMode != rocmmodel.CacheModeQ8 ||
		profile.CacheRoute.DeviceMode != rocmmodel.CacheModeQ8 ||
		profile.CacheRoute.Labels["engine_cache_factory_architecture"] != "model_route_set" ||
		profile.QuantLoaderRoute.Loader != "route_quant_loader" ||
		profile.RuntimeContractRoute.GreedyToken != true ||
		profile.Labels["route_set_payload"] != "kept" ||
		profile.Labels["profile_payload"] != "kept" {
		t.Fatalf("ResolveROCmModelProfile model-owned route set = %+v ok=%v, want preserved routes", profile, ok)
	}

	profile.TokenizerRoute.RequiredFiles = append(profile.TokenizerRoute.RequiredFiles, "mutated")
	profile.CacheRoute.Labels["engine_cache_factory_architecture"] = "mutated"
	next, ok := ResolveROCmModelProfile("/models/route-set", inference.ModelIdentity{
		Architecture: "model_route_set",
		Labels: map[string]string{
			"model_route_set_test": "true",
		},
	})
	if !ok ||
		slices.Contains(next.TokenizerRoute.RequiredFiles, "mutated") ||
		next.CacheRoute.Labels["engine_cache_factory_architecture"] != "model_route_set" {
		t.Fatalf("ResolveROCmModelProfile leaked mutable model-owned route data: %+v ok=%v", next, ok)
	}
}

func restoreRegisteredROCmModelProfileFactoriesForTest(t *testing.T) {
	t.Helper()
	factories := rocmmodel.RegisteredProfileFactories()

	t.Cleanup(func() {
		rocmmodel.ReplaceRegisteredProfileFactories(factories)
	})
}
