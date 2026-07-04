// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestDefaultDiffusionSamplerRoutes_Good_StaticCatalogue(t *testing.T) {
	routes := DefaultDiffusionSamplerRoutes()
	if len(routes) == 0 {
		t.Fatal("DefaultDiffusionSamplerRoutes returned no routes")
	}
	byArchitecture := map[string]DiffusionSamplerRoute{}
	for _, route := range routes {
		byArchitecture[route.Architecture] = route
	}
	diffusion := byArchitecture["diffusion_gemma"]
	if !diffusion.Matched() ||
		diffusion.Contract != DiffusionSamplerRegistryContract ||
		diffusion.Name != DiffusionSamplerRouteName ||
		diffusion.Reference != "go_mlx_diffusion_gemma" ||
		diffusion.Runtime != DiffusionSamplerRuntimeMetadata ||
		diffusion.RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		diffusion.Status != DiffusionSamplerPlannedMetadata ||
		diffusion.DiffusionRuntime != KernelStatusNotLinked ||
		diffusion.SamplerRuntime != KernelStatusNotLinked ||
		diffusion.TrunkRuntime != "model_pack_metadata" ||
		diffusion.ExecutionStatus != KernelStatusNotLinked ||
		diffusion.Fallback != "refused" ||
		!diffusion.Registered ||
		diffusion.NativeRuntime ||
		!diffusion.BlockDiffusion ||
		!diffusion.Sampler ||
		!diffusion.Trunk ||
		!diffusion.Generation ||
		!diffusion.SelfConditioning ||
		!diffusion.EncoderLayerScalars ||
		!diffusion.GlobalCanvasMask ||
		!diffusion.BlockLocalCanvasMask ||
		!diffusion.KVCacheRollback ||
		!diffusion.Streaming ||
		!diffusion.Staged ||
		!diffusion.Planned ||
		!diffusion.FallbackRefused ||
		diffusion.CanvasLength != 256 ||
		diffusion.DefaultCanvasLength != 64 ||
		diffusion.ReferenceCanvasLength != 256 ||
		diffusion.DefaultMaxSteps != 16 ||
		diffusion.ReferenceMaxSteps != 48 ||
		!slices.Contains(diffusion.RequiredFiles, "config.json") ||
		!slices.Contains(diffusion.RequiredFiles, "tokenizer.json") ||
		diffusion.Labels["engine_diffusion_sampler_reference"] != "go_mlx_diffusion_gemma" ||
		diffusion.Labels["engine_diffusion_sampler_diffusion_runtime"] != KernelStatusNotLinked ||
		diffusion.Labels["engine_diffusion_sampler_fallback_refused"] != "true" {
		t.Fatalf("diffusion_gemma route = %+v, want planned block-diffusion route", diffusion)
	}
}

func TestDiffusionSamplerRouteForInspection_Good_UsesLabels(t *testing.T) {
	inspectionLabels := map[string]string{
		"block_diffusion_model":              "true",
		"architecture_model_type":            "diffusion_gemma",
		"diffusion_reference":                "fixture_diffusion",
		"diffusion_runtime":                  KernelStatusLinked,
		"diffusion_sampler_runtime":          KernelStatusLinked,
		"diffusion_trunk_runtime":            "fixture_trunk",
		"diffusion_canvas_length":            "96",
		"diffusion_default_max_steps":        "8",
		"diffusion_confidence_threshold":     "0.01",
		"diffusion_temperature_exponent":     "1.5",
		"diffusion_execution_status":         "fixture_ready",
		"engine_diffusion_sampler_reference": "ignored_engine_label",
	}
	inspection := &inference.ModelPackInspection{
		Path:   "/models/diffusion-gemma",
		Labels: inspectionLabels,
		Model: inference.ModelIdentity{
			Architecture: "DiffusionGemmaForBlockDiffusion",
			Labels: map[string]string{
				"engine_architecture_resolved": "diffusion_gemma",
			},
		},
	}
	route, ok := DiffusionSamplerRouteForInspection(inspection)
	if !ok ||
		route.Architecture != "diffusion_gemma" ||
		route.Reference != "fixture_diffusion" ||
		route.Runtime != DiffusionSamplerRuntimeHIP ||
		route.RuntimeStatus != inference.FeatureRuntimeExperimental ||
		route.Status != DiffusionSamplerExperimentalNative ||
		route.DiffusionRuntime != KernelStatusLinked ||
		route.SamplerRuntime != KernelStatusLinked ||
		route.TrunkRuntime != "fixture_trunk" ||
		route.ExecutionStatus != "ready" ||
		!route.NativeRuntime ||
		!route.BlockDiffusion ||
		route.Staged ||
		route.Planned ||
		route.CanvasLength != 96 ||
		route.DefaultMaxSteps != 8 ||
		route.ConfidenceThreshold != 0.01 ||
		route.TemperatureExponent != 1.5 ||
		route.Labels["engine_diffusion_sampler_native_runtime"] != "true" {
		t.Fatalf("DiffusionSamplerRouteForInspection = %+v ok=%v, want label-derived linked diffusion route", route, ok)
	}
	inspectionLabels["diffusion_reference"] = "mutated"
	route.RequiredFiles[0] = "mutated"
	next, ok := DiffusionSamplerRouteForArchitecture("diffusion_gemma")
	if !ok ||
		next.Reference != "go_mlx_diffusion_gemma" ||
		next.RequiredFiles[0] != "config.json" {
		t.Fatalf("DiffusionSamplerRouteForInspection leaked mutable state: %+v ok=%v", next, ok)
	}
}

func TestRegisterDiffusionSamplerRoute_Good_ExtendsReactiveCatalogue(t *testing.T) {
	restoreRegisteredDiffusionSamplersForTest(t)

	RegisterDiffusionSamplerRoute(DiffusionSamplerRoute{})
	RegisterDiffusionSamplerRoute(DiffusionSamplerRoute{
		Architecture:    "fake-loader",
		Reference:       "first_diffusion_sampler",
		BlockDiffusion:  true,
		DefaultMaxSteps: 4,
	})
	RegisterDiffusionSamplerRoute(DiffusionSamplerRoute{
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

	if got := RegisteredDiffusionSamplerArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredDiffusionSamplerArchitectures = %v, want normalized replacement", got)
	}
	registeredRoutes := RegisteredDiffusionSamplerRoutes()
	if len(registeredRoutes) != 1 ||
		registeredRoutes[0].Architecture != "fake_loader" ||
		registeredRoutes[0].Reference != "fake_diffusion_sampler" {
		t.Fatalf("RegisteredDiffusionSamplerRoutes = %+v, want one replacement route", registeredRoutes)
	}
	routesForReplace := RegisteredDiffusionSamplerRoutes()
	registeredRoutes[0].Labels["engine_diffusion_sampler_reference"] = "mutated"
	registeredRoutes[0].RequiredWeightLeaves[0] = "mutated"
	nextRegistered, ok := RegisteredDiffusionSamplerRouteForArchitecture("fake-loader")
	if !ok ||
		nextRegistered.Labels["engine_diffusion_sampler_reference"] != "fake_diffusion_sampler" ||
		!slices.Equal(nextRegistered.RequiredWeightLeaves, []string{"fake.self_conditioning.weight"}) {
		t.Fatalf("RegisteredDiffusionSamplerRoutes leaked mutable state: %+v ok=%v", nextRegistered, ok)
	}
	ReplaceRegisteredDiffusionSamplerRoutes(routesForReplace)

	route, ok := DiffusionSamplerRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != DiffusionSamplerRegistryContract ||
		route.Name != DiffusionSamplerRouteName ||
		route.Architecture != "fake_loader" ||
		route.Family != "fake" ||
		route.Reference != "fake_diffusion_sampler" ||
		route.Runtime != DiffusionSamplerRuntimeMetadata ||
		route.RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		route.Status != DiffusionSamplerPlannedMetadata ||
		route.DiffusionRuntime != KernelStatusNotLinked ||
		route.SamplerRuntime != KernelStatusNotLinked ||
		route.TrunkRuntime != "model_pack_metadata" ||
		route.ExecutionStatus != KernelStatusNotLinked ||
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
		route.Labels["engine_diffusion_sampler_canvas_length"] != "96" {
		t.Fatalf("DiffusionSamplerRouteForArchitecture(fake-loader) = %+v ok=%v, want registered route", route, ok)
	}
	route.Labels["engine_diffusion_sampler_reference"] = "mutated"
	route.RequiredWeightLeaves[0] = "mutated"
	next, ok := DiffusionSamplerRouteForArchitecture("fake-loader")
	if !ok ||
		next.Labels["engine_diffusion_sampler_reference"] != "fake_diffusion_sampler" ||
		!slices.Equal(next.RequiredWeightLeaves, []string{"fake.self_conditioning.weight"}) {
		t.Fatalf("DiffusionSamplerRouteForArchitecture leaked mutable state: %+v ok=%v", next, ok)
	}
	if !slices.ContainsFunc(DefaultDiffusionSamplerRoutes(), func(route DiffusionSamplerRoute) bool {
		return route.Architecture == "fake_loader" && route.Reference == "fake_diffusion_sampler"
	}) {
		t.Fatalf("DefaultDiffusionSamplerRoutes missing fake_loader registration")
	}
}

func restoreRegisteredDiffusionSamplersForTest(t *testing.T) {
	t.Helper()
	routes := RegisteredDiffusionSamplerRoutes()
	t.Cleanup(func() {
		ReplaceRegisteredDiffusionSamplerRoutes(routes)
	})
}
