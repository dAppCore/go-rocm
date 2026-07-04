// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestDefaultMultimodalProcessorRoutes_Good_StaticCatalogue(t *testing.T) {
	routes := DefaultMultimodalProcessorRoutes()
	if len(routes) == 0 {
		t.Fatal("DefaultMultimodalProcessorRoutes returned no routes")
	}
	byArchitecture := map[string]MultimodalProcessorRoute{}
	for _, route := range routes {
		byArchitecture[route.Architecture] = route
	}
	gemma4 := byArchitecture["gemma4"]
	if !gemma4.Matched() ||
		gemma4.Contract != MultimodalProcessorRegistryContract ||
		gemma4.Name != MultimodalProcessorRouteName ||
		gemma4.Reference != "go_mlx_gemma4_vision" ||
		gemma4.VisionReference != "go_mlx_gemma4_vision" ||
		gemma4.Runtime != MultimodalProcessorRuntimeMetadata ||
		gemma4.RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		gemma4.Status != MultimodalProcessorPlannedMetadata ||
		gemma4.VisionRuntime != KernelStatusNotLinked ||
		gemma4.VisionProjectorRuntime != KernelStatusNotLinked ||
		!gemma4.Registered ||
		gemma4.NativeRuntime ||
		!gemma4.Multimodal ||
		!gemma4.Vision ||
		gemma4.Audio ||
		!gemma4.Video ||
		!gemma4.Projector ||
		!gemma4.VisionTower ||
		!gemma4.ImageProcessor ||
		!gemma4.Staged ||
		!gemma4.Planned ||
		!slices.Contains(gemma4.RequiredFiles, "config.json") ||
		gemma4.Labels["engine_multimodal_processor_reference"] != "go_mlx_gemma4_vision" {
		t.Fatalf("gemma4 multimodal route = %+v, want planned vision/video route", gemma4)
	}
	audio := byArchitecture["gemma4_unified"]
	if !audio.Matched() ||
		audio.Reference != "go_mlx_gemma4_audio" ||
		!audio.Audio ||
		audio.Vision ||
		audio.AudioRuntime != KernelStatusNotLinked ||
		audio.AudioProjectorRuntime != KernelStatusNotLinked ||
		audio.AudioFrontEndRuntime != KernelStatusNotLinked {
		t.Fatalf("gemma4_unified multimodal route = %+v, want planned audio route", audio)
	}
}

func TestMultimodalProcessorRouteForInspection_Good_UsesLabels(t *testing.T) {
	inspectionLabels := map[string]string{
		"multimodal_model":             "true",
		"architecture_model_type":      "gemma4",
		"vision_reference":             "fixture_vision",
		"vision_runtime":               KernelStatusLinked,
		"vision_projector_runtime":     KernelStatusLinked,
		"image_token_id":               "42",
		"vision_soft_tokens_per_image": "256",
		"vision_model_type":            "fixture_vision_model",
	}
	inspection := &inference.ModelPackInspection{
		Path:   "/models/gemma4",
		Labels: inspectionLabels,
		Model: inference.ModelIdentity{
			Architecture: "Gemma4ForConditionalGeneration",
			Labels: map[string]string{
				"engine_architecture_resolved": "gemma4",
			},
		},
	}
	route, ok := MultimodalProcessorRouteForInspection(inspection)
	if !ok ||
		route.Architecture != "gemma4" ||
		route.Reference != "fixture_vision" ||
		route.VisionReference != "fixture_vision" ||
		route.Runtime != MultimodalProcessorRuntimeHIP ||
		route.RuntimeStatus != inference.FeatureRuntimeExperimental ||
		route.Status != MultimodalProcessorExperimentalNative ||
		route.VisionRuntime != KernelStatusLinked ||
		route.VisionProjectorRuntime != KernelStatusLinked ||
		!route.NativeRuntime ||
		!route.Vision ||
		route.ImageTokenID != 42 ||
		route.SoftTokensPerImage != 256 ||
		route.VisionModelType != "fixture_vision_model" ||
		route.Labels["engine_multimodal_processor_native_runtime"] != "true" {
		t.Fatalf("MultimodalProcessorRouteForInspection = %+v ok=%v, want label-derived linked vision route", route, ok)
	}
	inspectionLabels["vision_reference"] = "mutated"
	route.RequiredFiles[0] = "mutated"
	next, ok := MultimodalProcessorRouteForArchitecture("gemma4")
	if !ok ||
		next.Reference != "go_mlx_gemma4_vision" ||
		next.RequiredFiles[0] != "config.json" {
		t.Fatalf("MultimodalProcessorRouteForInspection leaked mutable state: %+v ok=%v", next, ok)
	}
}

func TestRegisterMultimodalProcessorRoute_Good_ExtendsReactiveCatalogue(t *testing.T) {
	restoreRegisteredMultimodalProcessorsForTest(t)

	RegisterMultimodalProcessorRoute(MultimodalProcessorRoute{})
	RegisterMultimodalProcessorRoute(MultimodalProcessorRoute{
		Architecture:    "fake-loader",
		Reference:       "first_multimodal_processor",
		VisionReference: "first_vision_processor",
		Vision:          true,
	})
	RegisterMultimodalProcessorRoute(MultimodalProcessorRoute{
		Architecture:         "fake-loader",
		Family:               "fake",
		Reference:            "fake_multimodal_processor",
		VisionReference:      "fake_vision_processor",
		AudioReference:       "fake_audio_processor",
		Multimodal:           true,
		Vision:               true,
		Audio:                true,
		Video:                true,
		ImageTokenID:         42,
		AudioTokenID:         43,
		AudioSamplesPerToken: 640,
		VisionModelType:      "fake_vision",
		AudioModelType:       "fake_audio",
		RequiredFiles:        []string{"config.json", "processor_config.json"},
		OptionalFiles:        []string{"preprocessor_config.json"},
	})

	if got := RegisteredMultimodalProcessorArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredMultimodalProcessorArchitectures = %v, want normalized replacement", got)
	}
	registeredRoutes := RegisteredMultimodalProcessorRoutes()
	if len(registeredRoutes) != 1 ||
		registeredRoutes[0].Architecture != "fake_loader" ||
		registeredRoutes[0].Reference != "fake_multimodal_processor" {
		t.Fatalf("RegisteredMultimodalProcessorRoutes = %+v, want one replacement route", registeredRoutes)
	}
	routesForReplace := RegisteredMultimodalProcessorRoutes()
	registeredRoutes[0].Labels["engine_multimodal_processor_reference"] = "mutated"
	registeredRoutes[0].RequiredFiles[0] = "mutated"
	nextRegistered, ok := RegisteredMultimodalProcessorRouteForArchitecture("fake-loader")
	if !ok ||
		nextRegistered.Labels["engine_multimodal_processor_reference"] != "fake_multimodal_processor" ||
		!slices.Equal(nextRegistered.RequiredFiles, []string{"config.json", "processor_config.json"}) {
		t.Fatalf("RegisteredMultimodalProcessorRoutes leaked mutable state: %+v ok=%v", nextRegistered, ok)
	}
	ReplaceRegisteredMultimodalProcessorRoutes(routesForReplace)

	route, ok := MultimodalProcessorRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != MultimodalProcessorRegistryContract ||
		route.Name != MultimodalProcessorRouteName ||
		route.Architecture != "fake_loader" ||
		route.Family != "fake" ||
		route.Reference != "fake_multimodal_processor" ||
		route.VisionReference != "fake_vision_processor" ||
		route.AudioReference != "fake_audio_processor" ||
		route.Runtime != MultimodalProcessorRuntimeMetadata ||
		route.RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		route.Status != MultimodalProcessorPlannedMetadata ||
		route.VisionRuntime != KernelStatusNotLinked ||
		route.VisionProjectorRuntime != KernelStatusNotLinked ||
		route.AudioRuntime != KernelStatusNotLinked ||
		route.AudioProjectorRuntime != KernelStatusNotLinked ||
		route.AudioFrontEndRuntime != KernelStatusNotLinked ||
		!route.Registered ||
		route.NativeRuntime ||
		!route.Multimodal ||
		!route.Vision ||
		!route.Audio ||
		!route.Video ||
		!route.Projector ||
		!route.VisionTower ||
		!route.AudioTower ||
		!route.ImageProcessor ||
		!route.AudioProcessor ||
		!route.Staged ||
		!route.Planned ||
		route.ImageTokenID != 42 ||
		route.AudioTokenID != 43 ||
		route.AudioSamplesPerToken != 640 ||
		route.VisionModelType != "fake_vision" ||
		route.AudioModelType != "fake_audio" ||
		!slices.Equal(route.RequiredFiles, []string{"config.json", "processor_config.json"}) ||
		!slices.Equal(route.OptionalFiles, []string{"preprocessor_config.json"}) ||
		route.Labels["engine_multimodal_processor_reference"] != "fake_multimodal_processor" ||
		route.Labels["engine_multimodal_processor_audio_front_end_runtime"] != KernelStatusNotLinked {
		t.Fatalf("MultimodalProcessorRouteForArchitecture(fake-loader) = %+v ok=%v, want registered route", route, ok)
	}
	route.Labels["engine_multimodal_processor_reference"] = "mutated"
	route.RequiredFiles[0] = "mutated"
	next, ok := MultimodalProcessorRouteForArchitecture("fake-loader")
	if !ok ||
		next.Labels["engine_multimodal_processor_reference"] != "fake_multimodal_processor" ||
		!slices.Equal(next.RequiredFiles, []string{"config.json", "processor_config.json"}) {
		t.Fatalf("MultimodalProcessorRouteForArchitecture leaked mutable state: %+v ok=%v", next, ok)
	}
	if !slices.ContainsFunc(DefaultMultimodalProcessorRoutes(), func(route MultimodalProcessorRoute) bool {
		return route.Architecture == "fake_loader" && route.Reference == "fake_multimodal_processor"
	}) {
		t.Fatalf("DefaultMultimodalProcessorRoutes missing fake_loader registration")
	}
}

func restoreRegisteredMultimodalProcessorsForTest(t *testing.T) {
	t.Helper()
	order, routes := registeredMultimodalProcessors.Snapshot()
	for architecture, route := range routes {
		routes[architecture] = route.Clone()
	}
	t.Cleanup(func() {
		restoreRoutes := make(map[string]MultimodalProcessorRoute, len(routes))
		for architecture, route := range routes {
			restoreRoutes[architecture] = route.Clone()
		}
		registeredMultimodalProcessors.Restore(order, restoreRoutes)
	})
}
