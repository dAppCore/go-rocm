// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"slices"
	"strings"
	"testing"

	"dappco.re/go/inference"
)

func TestROCmNativeModelLoaderRegistry_Good_RoutesResolvedArchitecture(t *testing.T) {
	restoreRegisteredROCmNativeModelLoadersForTest(t)

	before := registeredROCmNativeModelLoaderArchitectures()
	registerROCmNativeModelLoader("", "ignored", func(*hipRuntime, string, nativeLoadConfig) (nativeModel, error) {
		t.Fatal("empty architecture loader should not register")
		return nil, nil
	})
	registerROCmNativeModelLoader("nil-loader", "ignored", nil)
	if after := registeredROCmNativeModelLoaderArchitectures(); !slices.Equal(after, before) {
		t.Fatalf("invalid registrations changed loader architectures: before=%v after=%v", before, after)
	}

	fake := &fakeNativeModel{}
	var gotPath string
	var gotConfig nativeLoadConfig
	registerROCmNativeModelLoader("fake-loader", "first_loader", func(*hipRuntime, string, nativeLoadConfig) (nativeModel, error) {
		t.Fatal("first loader should have been replaced")
		return nil, nil
	})
	registerROCmNativeModelLoader("fake_loader", "fake_hip_loader", func(_ *hipRuntime, path string, cfg nativeLoadConfig) (nativeModel, error) {
		gotPath = path
		gotConfig = cfg
		return fake, nil
	})

	if !slices.Contains(registeredROCmNativeModelLoaderArchitectures(), "fake_loader") {
		t.Fatalf("registered loader architectures = %v, want fake_loader", registeredROCmNativeModelLoaderArchitectures())
	}
	loader, ok := lookupROCmNativeModelLoader("fake-loader")
	if !ok || loader.name != "fake_hip_loader" {
		t.Fatalf("lookupROCmNativeModelLoader(fake-loader) = %+v ok=%v, want replacement loader", loader, ok)
	}

	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel("fake.gguf", nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		ModelLabels: map[string]string{
			"engine_architecture_resolved": "fake-loader",
		},
	})
	if err != nil {
		t.Fatalf("LoadModel routed fake loader: %v", err)
	}
	if model != fake {
		t.Fatalf("LoadModel returned %T, want registered fake native model", model)
	}
	if gotPath != "fake.gguf" || gotConfig.ModelLabels["engine_architecture_resolved"] != "fake-loader" {
		t.Fatalf("registered loader received path=%q cfg=%+v, want original load request", gotPath, gotConfig)
	}
	if len(driver.allocations) != 0 || len(driver.copies) != 0 {
		t.Fatalf("registered loader should own loading without fallback allocations: allocations=%v copies=%v", driver.allocations, driver.copies)
	}
}

func TestROCmNativeModelLoaderRegistry_Good_DefaultRoutesHaveStandaloneLoaders(t *testing.T) {
	restoreRegisteredROCmNativeModelLoadersForTest(t)

	for _, route := range DefaultROCmModelLoaderRoutes() {
		loader, ok := lookupROCmNativeModelLoader(route.Architecture)
		if rocmNativeModelLoaderRouteHasStandaloneLoader(route) {
			if !ok {
				t.Fatalf("lookupROCmNativeModelLoader(%s) ok=false, want standalone native loader for route %+v", route.Architecture, route)
			}
			if loader.name != route.Loader {
				t.Fatalf("lookupROCmNativeModelLoader(%s) = %+v, want loader token %q", route.Architecture, loader, route.Loader)
			}
			continue
		}
		if ok {
			t.Fatalf("lookupROCmNativeModelLoader(%s) = %+v ok=true, want no standalone loader for route %+v", route.Architecture, loader, route)
		}
	}
}

func TestROCmNativeModelLoaderRegistry_Good_PublicSnapshotMatchesRoutes(t *testing.T) {
	restoreRegisteredROCmNativeModelLoadersForTest(t)

	registrations := RegisteredROCmNativeModelLoaderRegistrations()
	if len(registrations) == 0 {
		t.Fatal("RegisteredROCmNativeModelLoaderRegistrations returned no native loaders")
	}
	byArchitecture := map[string]ROCmNativeModelLoaderRegistration{}
	for _, registration := range registrations {
		byArchitecture[registration.Architecture] = registration
	}
	for _, route := range DefaultROCmModelLoaderRoutes() {
		registration, ok := byArchitecture[route.Architecture]
		if rocmNativeModelLoaderRouteHasStandaloneLoader(route) {
			if !ok ||
				registration.Loader != route.Loader ||
				!registration.Registered ||
				!registration.Route.Matched() ||
				registration.Route.Loader != route.Loader ||
				registration.Route.Labels["engine_loader"] != route.Loader ||
				!registration.NativeRuntime ||
				!registration.Standalone ||
				registration.TextGenerate != route.TextGenerate {
				t.Fatalf("native loader registration[%s] = %+v ok=%v, want live standalone loader for route %+v", route.Architecture, registration, ok, route)
			}
			continue
		}
		if ok {
			t.Fatalf("native loader registration[%s] = %+v ok=true, want metadata/attached-only route without live standalone loader", route.Architecture, registration)
		}
	}

	registrations[0].Route.Labels["engine_loader"] = "mutated"
	next, ok := ROCmNativeModelLoaderRegistrationForArchitecture(registrations[0].Architecture)
	if !ok || next.Route.Labels["engine_loader"] == "mutated" {
		t.Fatalf("ROCmNativeModelLoaderRegistrationForArchitecture leaked mutable route labels: %+v ok=%v", next, ok)
	}
	registrations[0].Architecture = "mutated"
	if refreshed := RegisteredROCmNativeModelLoaderRegistrations(); refreshed[0].Architecture == "mutated" {
		t.Fatalf("RegisteredROCmNativeModelLoaderRegistrations leaked mutable slice state: %+v", refreshed)
	}
}

func TestROCmNativeModelLoaderRegistry_Good_PublicLookupNormalizesRegisteredRoute(t *testing.T) {
	restoreRegisteredROCmModelLoaderRoutesForTest(t)
	restoreRegisteredROCmNativeModelLoadersForTest(t)

	RegisterROCmModelLoaderRoute(ROCmModelLoaderRoute{
		Architecture:  "fake-loader",
		Family:        "fake",
		Loader:        "fake_hip_loader",
		Runtime:       rocmModelLoaderRuntimeHIP,
		Status:        ROCmModelLoadStandaloneNative,
		Reason:        "native registry public snapshot",
		NativeRuntime: true,
		Standalone:    true,
		TextGenerate:  true,
	})
	registerROCmNativeModelLoader("fake-loader", "fake_hip_loader", func(*hipRuntime, string, nativeLoadConfig) (nativeModel, error) {
		return &fakeNativeModel{}, nil
	})

	registration, ok := ROCmNativeModelLoaderRegistrationForArchitecture("fake-loader")
	if !ok ||
		registration.Architecture != "fake_loader" ||
		registration.Loader != "fake_hip_loader" ||
		!registration.Registered ||
		!registration.NativeRuntime ||
		!registration.Standalone ||
		!registration.TextGenerate ||
		registration.Route.Contract != ROCmModelLoaderRegistryContract ||
		registration.Route.Architecture != "fake_loader" ||
		registration.Route.Loader != "fake_hip_loader" ||
		registration.Route.Labels["engine_loader_reason"] != "native registry public snapshot" {
		t.Fatalf("ROCmNativeModelLoaderRegistrationForArchitecture(fake-loader) = %+v ok=%v, want normalized live loader registration", registration, ok)
	}
	if !slices.ContainsFunc(RegisteredROCmNativeModelLoaderRegistrations(), func(candidate ROCmNativeModelLoaderRegistration) bool {
		return candidate.Architecture == "fake_loader" && candidate.Loader == "fake_hip_loader"
	}) {
		t.Fatalf("RegisteredROCmNativeModelLoaderRegistrations missing fake loader: %+v", RegisteredROCmNativeModelLoaderRegistrations())
	}
}

func TestROCmNativeModelLoaderRegistry_Good_RoutesCataloguedArchitectureToDefaultHIPLoader(t *testing.T) {
	restoreRegisteredROCmNativeModelLoadersForTest(t)

	path, dataOffset := nativeHIPTensorGGUF(t)
	cfg := validHIPDriverFakeLoadConfigWithOffset(dataOffset)
	loader, ok := rocmNativeModelLoaderForConfig(cfg)
	if !ok || loader.name != "qwen3" {
		t.Fatalf("rocmNativeModelLoaderForConfig(qwen3) = %+v ok=%v, want catalogued qwen3 default HIP loader", loader, ok)
	}

	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(path, cfg)
	if err != nil {
		t.Fatalf("LoadModel fallback: %v", err)
	}
	defer model.Close()
	if _, ok := model.(*hipLoadedModel); !ok {
		t.Fatalf("LoadModel fallback returned %T, want *hipLoadedModel", model)
	}
	if !slices.Equal(driver.allocations, []uint64{16, 16}) || !slices.Equal(driver.copies, []uint64{16, 16}) {
		t.Fatalf("default loader allocations=%v copies=%v, want tensor copy path", driver.allocations, driver.copies)
	}
}

func TestROCmNativeModelLoaderRegistry_Good_AttachedDrafterNotStandalone(t *testing.T) {
	restoreRegisteredROCmNativeModelLoadersForTest(t)

	if loader, ok := lookupROCmNativeModelLoader("gemma4_assistant"); ok {
		t.Fatalf("lookupROCmNativeModelLoader(gemma4_assistant) = %+v ok=true, want no standalone loader", loader)
	}

	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel("assistant.gguf", nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "Gemma4AssistantForCausalLM"},
	})
	if err == nil {
		t.Fatal("LoadModel(gemma4_assistant) error = nil, want attached-only boundary")
	}
	if model != nil {
		t.Fatalf("LoadModel(gemma4_assistant) model = %T, want nil", model)
	}
	if !strings.Contains(err.Error(), "attached drafter") ||
		!strings.Contains(err.Error(), "standalone") ||
		!strings.Contains(err.Error(), "LoadAttachedDrafterPairAsTextModel") {
		t.Fatalf("LoadModel(gemma4_assistant) error = %v, want attached-only standalone boundary", err)
	}
	if len(driver.allocations) != 0 || len(driver.copies) != 0 {
		t.Fatalf("attached-only boundary should happen before tensor loading: allocations=%v copies=%v", driver.allocations, driver.copies)
	}
}

func TestROCmNativeModelLoaderRegistry_Good_MetadataOnlyNotStandalone(t *testing.T) {
	restoreRegisteredROCmNativeModelLoadersForTest(t)

	route, ok := ROCmModelLoaderRouteForArchitecture("Mamba2ForCausalLM")
	if !ok || !route.MetadataOnly || route.NativeRuntime {
		t.Fatalf("ROCmModelLoaderRouteForArchitecture(Mamba2ForCausalLM) = %+v ok=%v, want metadata-only route", route, ok)
	}
	if loader, ok := lookupROCmNativeModelLoader(route.Architecture); ok {
		t.Fatalf("lookupROCmNativeModelLoader(%s) = %+v ok=true, want no standalone loader", route.Architecture, loader)
	}

	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel("mamba2.gguf", nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "Mamba2ForCausalLM"},
	})
	if err == nil {
		t.Fatal("LoadModel(mamba2) error = nil, want metadata-only boundary")
	}
	if model != nil {
		t.Fatalf("LoadModel(mamba2) model = %T, want nil", model)
	}
	if !strings.Contains(err.Error(), "no standalone HIP model loader") ||
		!strings.Contains(err.Error(), string(ROCmModelLoadMetadataOnly)) {
		t.Fatalf("LoadModel(mamba2) error = %v, want metadata-only standalone boundary", err)
	}
	if len(driver.allocations) != 0 || len(driver.copies) != 0 {
		t.Fatalf("metadata-only boundary should happen before tensor loading: allocations=%v copies=%v", driver.allocations, driver.copies)
	}
}

func TestROCmNativeLoadModelProfile_Good_DoesNotApplyRuntimeGatesBeforeLoad(t *testing.T) {
	restoreDirect := SetROCmRuntimeGate(ROCmGateDirectGreedyToken, false)
	restoreMLP := SetROCmRuntimeGate(ROCmGateNativeMLPMatVec, false)
	t.Cleanup(func() {
		restoreMLP()
		restoreDirect()
	})

	cfg := nativeLoadConfig{
		ContextSize: 4096,
		ModelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    26,
			HiddenSize:   2304,
			QuantBits:    6,
			QuantGroup:   32,
		},
		ModelLabels: map[string]string{
			"gemma4_size":       "E4B",
			"gemma4_quant_mode": "q6",
			"sliding_window":    "1024",
		},
	}

	rocmApplyNativeLoadModelProfile("/models/lmstudio-community-gemma-4-e4b-it-6bit", &cfg)

	if !cfg.EngineProfile.Matched() ||
		cfg.EngineProfile.EngineFeatures.empty() ||
		cfg.ModelLabels["engine_feature_native_mlp_matvec"] != "true" ||
		cfg.ModelLabels["engine_runtime_gate_plan_reactive"] != "true" {
		t.Fatalf("rocmApplyNativeLoadModelProfile produced profile=%+v labels=%+v, want reactive profile metadata", cfg.EngineProfile, cfg.ModelLabels)
	}
	if ROCmRuntimeGateEnabled(ROCmGateDirectGreedyToken) ||
		ROCmRuntimeGateEnabled(ROCmGateNativeMLPMatVec) {
		t.Fatalf("rocmApplyNativeLoadModelProfile applied gates before load: %+v", ROCmRuntimeGateSnapshot())
	}
}

func restoreRegisteredROCmNativeModelLoadersForTest(t *testing.T) {
	t.Helper()
	order, values := registeredROCmNativeModelLoaders.Snapshot()
	t.Cleanup(func() {
		registeredROCmNativeModelLoaders.Restore(order, values)
	})
}
