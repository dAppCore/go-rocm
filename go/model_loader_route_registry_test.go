// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestRegisterROCmModelLoaderRoute_Good_ExtendsReactiveLoaderRegistry(t *testing.T) {
	restoreRegisteredROCmModelLoaderRoutesForTest(t)

	RegisterROCmModelLoaderRoute(ROCmModelLoaderRoute{})
	RegisterROCmModelLoaderRoute(ROCmModelLoaderRoute{
		Architecture: "fake-loader",
		Loader:       "first_loader",
		Status:       ROCmModelLoadStagedNative,
		Reason:       "first registration",
	})
	RegisterROCmModelLoaderRoute(ROCmModelLoaderRoute{
		Architecture: "fake-loader",
		Family:       "fake",
		Loader:       "fake_hip_loader",
		Status:       ROCmModelLoadStandaloneNative,
		Reason:       "registered test route",
	})

	if got := RegisteredROCmModelLoaderRouteArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmModelLoaderRouteArchitectures = %v, want normalized replacement under one architecture", got)
	}
	registered := RegisteredROCmModelLoaderRouteArchitectures()
	registered[0] = "mutated"
	if next := RegisteredROCmModelLoaderRouteArchitectures(); !slices.Equal(next, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmModelLoaderRouteArchitectures returned mutable state: %v", next)
	}
	modelRoute, ok := rocmmodel.RegisteredLoaderRouteForArchitecture("fake-loader")
	if !ok ||
		modelRoute.Architecture != "fake_loader" ||
		modelRoute.Loader != "fake_hip_loader" ||
		modelRoute.Status != string(ROCmModelLoadStandaloneNative) {
		t.Fatalf("model.RegisteredLoaderRouteForArchitecture(fake-loader) = %+v ok=%v, want mirrored registration", modelRoute, ok)
	}

	route, ok := ROCmModelLoaderRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != ROCmModelLoaderRegistryContract ||
		route.Architecture != "fake_loader" ||
		route.Family != "fake" ||
		route.Loader != "fake_hip_loader" ||
		route.Runtime != rocmModelLoaderRuntimeHIP ||
		route.Status != ROCmModelLoadStandaloneNative ||
		route.Target != "standalone" ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.Standalone ||
		!route.TextGenerate ||
		route.Labels["engine_loader"] != "fake_hip_loader" ||
		route.Labels["engine_loader_architecture"] != "fake_loader" ||
		route.Labels["engine_loader_reason"] != "registered test route" {
		t.Fatalf("ROCmModelLoaderRouteForArchitecture(fake-loader) = %+v ok=%v, want registered HIP loader route", route, ok)
	}

	route.Labels["engine_loader"] = "mutated"
	nextRoute, ok := ROCmModelLoaderRouteForArchitecture("fake-loader")
	if !ok || nextRoute.Labels["engine_loader"] != "fake_hip_loader" {
		t.Fatalf("ROCmModelLoaderRouteForArchitecture leaked mutable labels: %+v ok=%v", nextRoute, ok)
	}
	defaults := DefaultROCmModelLoaderRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmModelLoaderRoute) bool {
		return route.Architecture == "fake_loader" && route.Loader == "fake_hip_loader"
	}) {
		t.Fatalf("DefaultROCmModelLoaderRoutes missing registered loader route: %+v", defaults)
	}

	profile := ROCmModelProfile{
		Name:         "fake",
		Family:       "fake",
		Architecture: "fake_loader",
		ArchitectureProfile: ROCmArchitectureProfile{
			ID:            "fake_loader",
			Family:        "fake",
			RuntimeStatus: inference.FeatureRuntimeNative,
			NativeRuntime: true,
			Generation:    true,
			Chat:          true,
		},
		EngineFeatures: ROCmEngineFeatures{
			Contract:      rocmEngineFeaturesContract,
			Architecture:  "fake_loader",
			Family:        "fake",
			NativeRuntime: true,
			TextGenerate:  true,
		},
	}
	status := ROCmModelLoadStatusForProfile(profile)
	if status.Loader != "fake_hip_loader" ||
		status.LoaderRuntime != rocmModelLoaderRuntimeHIP ||
		status.LoaderContract != ROCmModelLoaderRegistryContract ||
		status.Status != ROCmModelLoadStandaloneNative ||
		status.Target != "standalone" ||
		!status.LoaderRegistered ||
		!status.NativeRuntime ||
		!status.Standalone ||
		!status.TextGenerate ||
		status.Labels["engine_loader"] != "fake_hip_loader" ||
		status.Labels["engine_load_reason"] != "registered test route" {
		t.Fatalf("ROCmModelLoadStatusForProfile registered route = %+v, want load status to use registered loader", status)
	}
}

func TestRegisterROCmModelLoaderRoute_Good_SeesModelPackageRegistration(t *testing.T) {
	restoreRegisteredROCmModelLoaderRoutesForTest(t)

	rocmmodel.RegisterLoaderRoute(rocmmodel.LoaderRoute{
		Architecture: "model-only-loader",
		Family:       "model-only",
		Loader:       "model_only_loader",
		Status:       rocmmodel.StatusStandaloneNative,
		Reason:       "model package registration",
	})

	route, ok := ROCmModelLoaderRouteForArchitecture("model-only-loader")
	if !ok ||
		route.Architecture != "model_only_loader" ||
		route.Family != "model-only" ||
		route.Loader != "model_only_loader" ||
		route.Status != ROCmModelLoadStandaloneNative ||
		route.Labels["engine_loader_reason"] != "model package registration" {
		t.Fatalf("ROCmModelLoaderRouteForArchitecture(model-only-loader) = %+v ok=%v, want model-package route", route, ok)
	}
	defaults := DefaultROCmModelLoaderRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmModelLoaderRoute) bool {
		return route.Architecture == "model_only_loader" && route.Loader == "model_only_loader"
	}) {
		t.Fatalf("DefaultROCmModelLoaderRoutes missing model package registration: %+v", defaults)
	}
}

func TestRegisterROCmModelLoaderRoute_Good_OverridesBuiltinLoaderRoute(t *testing.T) {
	restoreRegisteredROCmModelLoaderRoutesForTest(t)

	RegisterROCmModelLoaderRoute(ROCmModelLoaderRoute{
		Architecture: "qwen3",
		Family:       "qwen",
		Loader:       "qwen3_registered_loader",
		Status:       ROCmModelLoadStagedNative,
		Reason:       "registered qwen override",
	})

	route, ok := ROCmModelLoaderRouteForArchitecture("Qwen3ForCausalLM")
	if !ok ||
		route.Architecture != "qwen3" ||
		route.Loader != "qwen3_registered_loader" ||
		route.Status != ROCmModelLoadStagedNative ||
		!route.Registered ||
		!route.Staged ||
		!route.Standalone ||
		route.Labels["engine_loader_reason"] != "registered qwen override" {
		t.Fatalf("ROCmModelLoaderRouteForArchitecture(qwen3 override) = %+v ok=%v, want registered override", route, ok)
	}
}

func restoreRegisteredROCmModelLoaderRoutesForTest(t *testing.T) {
	t.Helper()
	modelRoutes := rocmmodel.RegisteredLoaderRoutes()

	t.Cleanup(func() {
		restoreRoutes := make([]rocmmodel.LoaderRoute, 0, len(modelRoutes))
		for _, route := range modelRoutes {
			restoreRoutes = append(restoreRoutes, route.Clone())
		}
		rocmmodel.ReplaceRegisteredLoaderRoutes(restoreRoutes)
	})
}
