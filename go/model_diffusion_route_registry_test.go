// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestRegisterROCmDiffusionSamplerRoute_Good_ExtendsReactiveSamplerRegistry(t *testing.T) {
	restoreRegisteredROCmDiffusionSamplerRoutesForTest(t)

	RegisterROCmDiffusionSamplerRoute(ROCmDiffusionSamplerRoute{})
	RegisterROCmDiffusionSamplerRoute(ROCmDiffusionSamplerRoute{
		Architecture:    "fake-loader",
		Reference:       "first_diffusion_sampler",
		BlockDiffusion:  true,
		DefaultMaxSteps: 4,
	})
	RegisterROCmDiffusionSamplerRoute(ROCmDiffusionSamplerRoute{
		Architecture:           "fake-loader",
		Family:                 "fake",
		Reference:              "fake_diffusion_sampler",
		BlockDiffusion:         true,
		SelfConditioning:       true,
		EncoderLayerScalars:    true,
		GlobalCanvasMask:       true,
		BlockLocalCanvasMask:   true,
		KVCacheRollback:        true,
		Streaming:              true,
		CanvasLength:           96,
		DefaultCanvasLength:    32,
		ReferenceCanvasLength:  96,
		DefaultMaxSteps:        8,
		ReferenceMaxSteps:      24,
		RequiredFiles:          []string{"config.json", "diffusion_config.json"},
		OptionalFiles:          []string{"tokenizer.json"},
		RequiredWeightLeaves:   []string{"fake.self_conditioning.weight"},
		OptionalWeightPrefixes: []string{"fake.layers."},
	})

	if got := RegisteredROCmDiffusionSamplerRouteArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmDiffusionSamplerRouteArchitectures = %v, want normalized replacement under one architecture", got)
	}
	registered := RegisteredROCmDiffusionSamplerRouteArchitectures()
	registered[0] = "mutated"
	if next := RegisteredROCmDiffusionSamplerRouteArchitectures(); !slices.Equal(next, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmDiffusionSamplerRouteArchitectures returned mutable state: %v", next)
	}

	route, ok := ROCmDiffusionSamplerRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != ROCmDiffusionSamplerRegistryContract ||
		route.Name != rocmDiffusionSamplerRegistryRouteName ||
		route.Architecture != "fake_loader" ||
		route.Family != "fake" ||
		route.Reference != "fake_diffusion_sampler" ||
		route.Runtime != rocmDiffusionSamplerRuntimeMetadata ||
		route.RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		route.Status != ROCmDiffusionSamplerPlannedMetadata ||
		route.DiffusionRuntime != hipKernelStatusNotLinked ||
		route.SamplerRuntime != hipKernelStatusNotLinked ||
		route.TrunkRuntime != "model_pack_metadata" ||
		route.ExecutionStatus != hipKernelStatusNotLinked ||
		route.Fallback != "refused" ||
		!route.Registered ||
		route.NativeRuntime ||
		!route.BlockDiffusion ||
		!route.Sampler ||
		!route.Trunk ||
		!route.Generation ||
		!route.SelfConditioning ||
		!route.EncoderLayerScalars ||
		!route.GlobalCanvasMask ||
		!route.BlockLocalCanvasMask ||
		!route.KVCacheRollback ||
		!route.Streaming ||
		!route.Staged ||
		!route.Planned ||
		!route.FallbackRefused ||
		route.CanvasLength != 96 ||
		route.DefaultCanvasLength != 32 ||
		route.DefaultMaxSteps != 8 ||
		route.ReferenceMaxSteps != 24 ||
		!slices.Equal(route.RequiredFiles, []string{"config.json", "diffusion_config.json"}) ||
		!slices.Equal(route.OptionalFiles, []string{"tokenizer.json"}) ||
		!slices.Equal(route.RequiredWeightLeaves, []string{"fake.self_conditioning.weight"}) ||
		!slices.Equal(route.OptionalWeightPrefixes, []string{"fake.layers."}) ||
		route.Labels["engine_diffusion_sampler_reference"] != "fake_diffusion_sampler" ||
		route.Labels["engine_diffusion_sampler_diffusion_runtime"] != hipKernelStatusNotLinked ||
		route.Labels["engine_diffusion_sampler_fallback_refused"] != "true" ||
		route.Labels["engine_diffusion_sampler_canvas_length"] != "96" {
		t.Fatalf("ROCmDiffusionSamplerRouteForArchitecture(fake-loader) = %+v ok=%v, want registered diffusion route", route, ok)
	}

	route.Labels["engine_diffusion_sampler_reference"] = "mutated"
	route.RequiredWeightLeaves[0] = "mutated"
	nextRoute, ok := ROCmDiffusionSamplerRouteForArchitecture("fake-loader")
	if !ok ||
		nextRoute.Labels["engine_diffusion_sampler_reference"] != "fake_diffusion_sampler" ||
		!slices.Equal(nextRoute.RequiredWeightLeaves, []string{"fake.self_conditioning.weight"}) {
		t.Fatalf("ROCmDiffusionSamplerRouteForArchitecture leaked mutable state: %+v ok=%v", nextRoute, ok)
	}
	modelRoute, ok := rocmmodel.RegisteredDiffusionSamplerRouteForArchitecture("fake-loader")
	if !ok ||
		modelRoute.Reference != "fake_diffusion_sampler" ||
		!slices.Equal(modelRoute.RequiredFiles, []string{"config.json", "diffusion_config.json"}) ||
		!slices.Equal(modelRoute.RequiredWeightLeaves, []string{"fake.self_conditioning.weight"}) ||
		modelRoute.Labels["engine_diffusion_sampler_reference"] != "fake_diffusion_sampler" {
		t.Fatalf("model.RegisteredDiffusionSamplerRouteForArchitecture(fake-loader) = %+v ok=%v, want mirrored route", modelRoute, ok)
	}

	defaults := DefaultROCmDiffusionSamplerRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmDiffusionSamplerRoute) bool {
		return route.Architecture == "fake_loader" && route.Reference == "fake_diffusion_sampler"
	}) {
		t.Fatalf("DefaultROCmDiffusionSamplerRoutes missing registered route: %+v", defaults)
	}

	profileRoute := ROCmDiffusionSamplerRouteForProfile(ROCmModelProfile{
		Name:         "fake",
		Family:       "fake",
		Architecture: "fake_loader",
		ArchitectureProfile: ROCmArchitectureProfile{
			ID:     "fake_loader",
			Family: "fake",
		},
	})
	if profileRoute.Reference != "fake_diffusion_sampler" ||
		!profileRoute.BlockDiffusion ||
		!profileRoute.Sampler ||
		!profileRoute.Generation ||
		profileRoute.CanvasLength != 96 ||
		profileRoute.Labels["engine_diffusion_sampler_reference"] != "fake_diffusion_sampler" {
		t.Fatalf("ROCmDiffusionSamplerRouteForProfile registered route = %+v, want profile to use registered sampler", profileRoute)
	}
}

func TestRegisterROCmDiffusionSamplerRoute_Good_SeesModelPackageRegistration(t *testing.T) {
	restoreRegisteredROCmDiffusionSamplerRoutesForTest(t)

	rocmmodel.RegisterDiffusionSamplerRoute(rocmmodel.DiffusionSamplerRoute{
		Architecture:           "folder-diffusion",
		Family:                 "folder",
		Reference:              "folder_diffusion_sampler",
		BlockDiffusion:         true,
		SelfConditioning:       true,
		EncoderLayerScalars:    true,
		GlobalCanvasMask:       true,
		BlockLocalCanvasMask:   true,
		KVCacheRollback:        true,
		Streaming:              true,
		CanvasLength:           96,
		DefaultCanvasLength:    32,
		ReferenceCanvasLength:  96,
		DefaultMaxSteps:        8,
		ReferenceMaxSteps:      24,
		RequiredFiles:          []string{"config.json", "diffusion_config.json"},
		OptionalFiles:          []string{"tokenizer.json"},
		RequiredWeightLeaves:   []string{"folder.self_conditioning.weight"},
		OptionalWeightPrefixes: []string{"folder.layers."},
	})

	route, ok := ROCmDiffusionSamplerRouteForArchitecture("folder-diffusion")
	if !ok ||
		route.Contract != ROCmDiffusionSamplerRegistryContract ||
		route.Name != rocmDiffusionSamplerRegistryRouteName ||
		route.Architecture != "folder_diffusion" ||
		route.Family != "folder" ||
		route.Reference != "folder_diffusion_sampler" ||
		route.Runtime != rocmDiffusionSamplerRuntimeMetadata ||
		route.RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		route.Status != ROCmDiffusionSamplerPlannedMetadata ||
		route.DiffusionRuntime != hipKernelStatusNotLinked ||
		route.SamplerRuntime != hipKernelStatusNotLinked ||
		route.TrunkRuntime != "model_pack_metadata" ||
		route.ExecutionStatus != hipKernelStatusNotLinked ||
		route.Fallback != "refused" ||
		!route.Registered ||
		route.NativeRuntime ||
		!route.BlockDiffusion ||
		!route.Sampler ||
		!route.Trunk ||
		!route.Generation ||
		!route.SelfConditioning ||
		!route.EncoderLayerScalars ||
		!route.GlobalCanvasMask ||
		!route.BlockLocalCanvasMask ||
		!route.KVCacheRollback ||
		!route.Streaming ||
		!route.Staged ||
		!route.Planned ||
		!route.FallbackRefused ||
		route.CanvasLength != 96 ||
		route.DefaultCanvasLength != 32 ||
		route.DefaultMaxSteps != 8 ||
		route.ReferenceMaxSteps != 24 ||
		!slices.Equal(route.RequiredFiles, []string{"config.json", "diffusion_config.json"}) ||
		!slices.Equal(route.OptionalFiles, []string{"tokenizer.json"}) ||
		!slices.Equal(route.RequiredWeightLeaves, []string{"folder.self_conditioning.weight"}) ||
		!slices.Equal(route.OptionalWeightPrefixes, []string{"folder.layers."}) ||
		route.Labels["engine_diffusion_sampler_reference"] != "folder_diffusion_sampler" {
		t.Fatalf("ROCmDiffusionSamplerRouteForArchitecture(folder-diffusion) = %+v ok=%v, want model package route", route, ok)
	}
	defaults := DefaultROCmDiffusionSamplerRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmDiffusionSamplerRoute) bool {
		return route.Architecture == "folder_diffusion" && route.Reference == "folder_diffusion_sampler"
	}) {
		t.Fatalf("DefaultROCmDiffusionSamplerRoutes missing model package route: %+v", defaults)
	}
	profileRoute := ROCmDiffusionSamplerRouteForProfile(ROCmModelProfile{
		Name:         "folder",
		Family:       "folder",
		Architecture: "folder_diffusion",
	})
	if profileRoute.Reference != "folder_diffusion_sampler" ||
		!profileRoute.BlockDiffusion ||
		!profileRoute.Sampler ||
		!profileRoute.Generation ||
		profileRoute.Labels["engine_diffusion_sampler_reference"] != "folder_diffusion_sampler" {
		t.Fatalf("ROCmDiffusionSamplerRouteForProfile(folder_diffusion) = %+v, want model package sampler route", profileRoute)
	}
}

func TestRegisterROCmDiffusionSamplerRoute_Good_OverridesBuiltinSamplerRoute(t *testing.T) {
	restoreRegisteredROCmDiffusionSamplerRoutesForTest(t)

	RegisterROCmDiffusionSamplerRoute(ROCmDiffusionSamplerRoute{
		Architecture:          "diffusion_gemma",
		Family:                "gemma",
		Reference:             "registered_diffusion_gemma_sampler",
		NativeRuntime:         true,
		BlockDiffusion:        true,
		SelfConditioning:      true,
		KVCacheRollback:       true,
		CanvasLength:          128,
		DefaultCanvasLength:   64,
		ReferenceCanvasLength: 128,
		DefaultMaxSteps:       12,
		ReferenceMaxSteps:     32,
		RequiredFiles:         []string{"config.json", "registered_diffusion.json"},
	})

	route, ok := ROCmDiffusionSamplerRouteForArchitecture("DiffusionGemmaForBlockDiffusion")
	if !ok ||
		route.Architecture != "diffusion_gemma" ||
		route.Reference != "registered_diffusion_gemma_sampler" ||
		route.Runtime != rocmDiffusionSamplerRuntimeHIP ||
		route.Status != ROCmDiffusionSamplerExperimentalNative ||
		route.DiffusionRuntime != hipKernelStatusLinked ||
		route.SamplerRuntime != hipKernelStatusLinked ||
		route.ExecutionStatus != "ready" ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.BlockDiffusion ||
		!route.Sampler ||
		!route.Trunk ||
		!route.Generation ||
		!route.SelfConditioning ||
		!route.KVCacheRollback ||
		route.Staged ||
		route.Planned ||
		route.FallbackRefused ||
		route.CanvasLength != 128 ||
		route.DefaultMaxSteps != 12 ||
		!slices.Equal(route.RequiredFiles, []string{"config.json", "registered_diffusion.json"}) ||
		route.Labels["engine_diffusion_sampler_reference"] != "registered_diffusion_gemma_sampler" ||
		route.Labels["engine_diffusion_sampler_native_runtime"] != "true" ||
		route.Labels["engine_diffusion_sampler_execution_status"] != "ready" {
		t.Fatalf("ROCmDiffusionSamplerRouteForArchitecture(diffusion_gemma override) = %+v ok=%v, want registered native sampler route", route, ok)
	}

	profile, ok := ResolveROCmModelProfile("/models/diffusion-gemma", inference.ModelIdentity{Architecture: "DiffusionGemmaForBlockDiffusion"})
	if !ok ||
		profile.DiffusionSamplerRoute.Reference != "registered_diffusion_gemma_sampler" ||
		!profile.DiffusionSamplerRoute.NativeRuntime ||
		profile.DiffusionSamplerRoute.DiffusionRuntime != hipKernelStatusLinked ||
		profile.DiffusionSamplerRoute.SamplerRuntime != hipKernelStatusLinked ||
		profile.DiffusionSamplerRoute.CanvasLength != 128 ||
		profile.DiffusionSamplerRoute.Labels["engine_diffusion_sampler_reference"] != "registered_diffusion_gemma_sampler" {
		t.Fatalf("ResolveROCmModelProfile(diffusion_gemma registered override) = %+v ok=%v, want profile to expose registered sampler", profile, ok)
	}
}

func restoreRegisteredROCmDiffusionSamplerRoutesForTest(t *testing.T) {
	t.Helper()
	modelRoutes := rocmmodel.RegisteredDiffusionSamplerRoutes()

	t.Cleanup(func() {
		restoreRoutes := make([]rocmmodel.DiffusionSamplerRoute, 0, len(modelRoutes))
		for _, route := range modelRoutes {
			restoreRoutes = append(restoreRoutes, route.Clone())
		}
		rocmmodel.ReplaceRegisteredDiffusionSamplerRoutes(restoreRoutes)
	})
}
