// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestRegisterROCmMultimodalProcessorRoute_Good_ExtendsReactiveProcessorRegistry(t *testing.T) {
	restoreRegisteredROCmMultimodalProcessorRoutesForTest(t)

	RegisterROCmMultimodalProcessorRoute(ROCmMultimodalProcessorRoute{})
	RegisterROCmMultimodalProcessorRoute(ROCmMultimodalProcessorRoute{
		Architecture:    "fake-loader",
		Reference:       "first_multimodal_processor",
		VisionReference: "first_vision_processor",
		Vision:          true,
	})
	RegisterROCmMultimodalProcessorRoute(ROCmMultimodalProcessorRoute{
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

	if got := RegisteredROCmMultimodalProcessorRouteArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmMultimodalProcessorRouteArchitectures = %v, want normalized replacement under one architecture", got)
	}
	registered := RegisteredROCmMultimodalProcessorRouteArchitectures()
	registered[0] = "mutated"
	if next := RegisteredROCmMultimodalProcessorRouteArchitectures(); !slices.Equal(next, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmMultimodalProcessorRouteArchitectures returned mutable state: %v", next)
	}

	route, ok := ROCmMultimodalProcessorRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != ROCmMultimodalProcessorRegistryContract ||
		route.Name != rocmMultimodalProcessorRegistryRouteName ||
		route.Architecture != "fake_loader" ||
		route.Family != "fake" ||
		route.Reference != "fake_multimodal_processor" ||
		route.VisionReference != "fake_vision_processor" ||
		route.AudioReference != "fake_audio_processor" ||
		route.Runtime != rocmMultimodalProcessorRuntimeMetadata ||
		route.RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		route.Status != ROCmMultimodalProcessorPlannedMetadata ||
		route.VisionRuntime != hipKernelStatusNotLinked ||
		route.VisionProjectorRuntime != hipKernelStatusNotLinked ||
		route.AudioRuntime != hipKernelStatusNotLinked ||
		route.AudioProjectorRuntime != hipKernelStatusNotLinked ||
		route.AudioFrontEndRuntime != hipKernelStatusNotLinked ||
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
		route.Labels["engine_multimodal_processor_vision"] != "true" ||
		route.Labels["engine_multimodal_processor_audio"] != "true" ||
		route.Labels["engine_multimodal_processor_video"] != "true" ||
		route.Labels["engine_multimodal_processor_audio_front_end_runtime"] != hipKernelStatusNotLinked {
		t.Fatalf("ROCmMultimodalProcessorRouteForArchitecture(fake-loader) = %+v ok=%v, want registered multimodal route", route, ok)
	}

	route.Labels["engine_multimodal_processor_reference"] = "mutated"
	route.RequiredFiles[0] = "mutated"
	nextRoute, ok := ROCmMultimodalProcessorRouteForArchitecture("fake-loader")
	if !ok ||
		nextRoute.Labels["engine_multimodal_processor_reference"] != "fake_multimodal_processor" ||
		!slices.Equal(nextRoute.RequiredFiles, []string{"config.json", "processor_config.json"}) {
		t.Fatalf("ROCmMultimodalProcessorRouteForArchitecture leaked mutable state: %+v ok=%v", nextRoute, ok)
	}
	modelRoute, ok := rocmmodel.RegisteredMultimodalProcessorRouteForArchitecture("fake-loader")
	if !ok ||
		modelRoute.Reference != "fake_multimodal_processor" ||
		modelRoute.VisionReference != "fake_vision_processor" ||
		modelRoute.AudioReference != "fake_audio_processor" ||
		!slices.Equal(modelRoute.RequiredFiles, []string{"config.json", "processor_config.json"}) ||
		modelRoute.Labels["engine_multimodal_processor_reference"] != "fake_multimodal_processor" {
		t.Fatalf("model.RegisteredMultimodalProcessorRouteForArchitecture(fake-loader) = %+v ok=%v, want mirrored route", modelRoute, ok)
	}

	defaults := DefaultROCmMultimodalProcessorRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmMultimodalProcessorRoute) bool {
		return route.Architecture == "fake_loader" && route.Reference == "fake_multimodal_processor"
	}) {
		t.Fatalf("DefaultROCmMultimodalProcessorRoutes missing registered route: %+v", defaults)
	}

	profileRoute := ROCmMultimodalProcessorRouteForProfile(ROCmModelProfile{
		Name:         "fake",
		Family:       "fake",
		Architecture: "fake_loader",
		ArchitectureProfile: ROCmArchitectureProfile{
			ID:     "fake_loader",
			Family: "fake",
		},
	})
	if profileRoute.Reference != "fake_multimodal_processor" ||
		!profileRoute.Multimodal ||
		!profileRoute.Vision ||
		!profileRoute.Audio ||
		!profileRoute.Video ||
		profileRoute.Labels["engine_multimodal_processor_reference"] != "fake_multimodal_processor" {
		t.Fatalf("ROCmMultimodalProcessorRouteForProfile registered route = %+v, want profile to use registered processor", profileRoute)
	}
}

func TestRegisterROCmMultimodalProcessorRoute_Good_SeesModelPackageRegistration(t *testing.T) {
	restoreRegisteredROCmMultimodalProcessorRoutesForTest(t)

	rocmmodel.RegisterMultimodalProcessorRoute(rocmmodel.MultimodalProcessorRoute{
		Architecture:         "folder-multimodal",
		Family:               "folder",
		Reference:            "folder_multimodal_processor",
		VisionReference:      "folder_vision_processor",
		AudioReference:       "folder_audio_processor",
		Multimodal:           true,
		Vision:               true,
		Audio:                true,
		Video:                true,
		ImageTokenID:         42,
		AudioTokenID:         43,
		AudioSamplesPerToken: 640,
		VisionModelType:      "folder_vision",
		AudioModelType:       "folder_audio",
		RequiredFiles:        []string{"config.json", "processor_config.json"},
		OptionalFiles:        []string{"preprocessor_config.json"},
	})

	route, ok := ROCmMultimodalProcessorRouteForArchitecture("folder-multimodal")
	if !ok ||
		route.Contract != ROCmMultimodalProcessorRegistryContract ||
		route.Name != rocmMultimodalProcessorRegistryRouteName ||
		route.Architecture != "folder_multimodal" ||
		route.Family != "folder" ||
		route.Reference != "folder_multimodal_processor" ||
		route.VisionReference != "folder_vision_processor" ||
		route.AudioReference != "folder_audio_processor" ||
		route.Runtime != rocmMultimodalProcessorRuntimeMetadata ||
		route.RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		route.Status != ROCmMultimodalProcessorPlannedMetadata ||
		route.VisionRuntime != hipKernelStatusNotLinked ||
		route.VisionProjectorRuntime != hipKernelStatusNotLinked ||
		route.AudioRuntime != hipKernelStatusNotLinked ||
		route.AudioProjectorRuntime != hipKernelStatusNotLinked ||
		route.AudioFrontEndRuntime != hipKernelStatusNotLinked ||
		!route.Registered ||
		route.NativeRuntime ||
		!route.Multimodal ||
		!route.Vision ||
		!route.Audio ||
		!route.Video ||
		!route.Projector ||
		!route.Staged ||
		!route.Planned ||
		route.ImageTokenID != 42 ||
		route.AudioTokenID != 43 ||
		route.AudioSamplesPerToken != 640 ||
		route.VisionModelType != "folder_vision" ||
		route.AudioModelType != "folder_audio" ||
		route.Labels["engine_multimodal_processor_reference"] != "folder_multimodal_processor" {
		t.Fatalf("ROCmMultimodalProcessorRouteForArchitecture(folder-multimodal) = %+v ok=%v, want model package route", route, ok)
	}
	defaults := DefaultROCmMultimodalProcessorRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmMultimodalProcessorRoute) bool {
		return route.Architecture == "folder_multimodal" && route.Reference == "folder_multimodal_processor"
	}) {
		t.Fatalf("DefaultROCmMultimodalProcessorRoutes missing model package route: %+v", defaults)
	}
	profileRoute := ROCmMultimodalProcessorRouteForProfile(ROCmModelProfile{
		Name:         "folder",
		Family:       "folder",
		Architecture: "folder_multimodal",
	})
	if profileRoute.Reference != "folder_multimodal_processor" ||
		!profileRoute.Multimodal ||
		!profileRoute.Vision ||
		!profileRoute.Audio ||
		!profileRoute.Video ||
		profileRoute.Labels["engine_multimodal_processor_reference"] != "folder_multimodal_processor" {
		t.Fatalf("ROCmMultimodalProcessorRouteForProfile(folder_multimodal) = %+v, want model package processor route", profileRoute)
	}
}

func TestRegisterROCmMultimodalProcessorRoute_Good_OverridesBuiltinProcessorRoute(t *testing.T) {
	restoreRegisteredROCmMultimodalProcessorRoutesForTest(t)

	RegisterROCmMultimodalProcessorRoute(ROCmMultimodalProcessorRoute{
		Architecture:       "gemma4",
		Family:             "gemma4",
		Reference:          "registered_gemma4_processor",
		VisionReference:    "registered_gemma4_vision",
		NativeRuntime:      true,
		Multimodal:         true,
		Vision:             true,
		Video:              true,
		RequiredFiles:      []string{"config.json", "registered_processor.json"},
		OptionalFiles:      []string{"registered_preprocessor.json"},
		SoftTokensPerImage: 512,
	})

	route, ok := ROCmMultimodalProcessorRouteForArchitecture("Gemma4ForConditionalGeneration")
	if !ok ||
		route.Architecture != "gemma4" ||
		route.Reference != "registered_gemma4_processor" ||
		route.VisionReference != "registered_gemma4_vision" ||
		route.Runtime != rocmMultimodalProcessorRuntimeHIP ||
		route.Status != ROCmMultimodalProcessorExperimentalNative ||
		route.VisionRuntime != hipKernelStatusLinked ||
		route.VisionProjectorRuntime != hipKernelStatusLinked ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.Multimodal ||
		!route.Vision ||
		!route.Video ||
		route.Audio ||
		route.Staged ||
		route.Planned ||
		route.SoftTokensPerImage != 512 ||
		route.Labels["engine_multimodal_processor_reference"] != "registered_gemma4_processor" ||
		route.Labels["engine_multimodal_processor_native_runtime"] != "true" {
		t.Fatalf("ROCmMultimodalProcessorRouteForArchitecture(gemma4 override) = %+v ok=%v, want registered native route", route, ok)
	}

	profile, ok := ResolveROCmModelProfile("/models/gemma4", inference.ModelIdentity{Architecture: "Gemma4ForConditionalGeneration"})
	if !ok ||
		profile.MultimodalProcessorRoute.Reference != "registered_gemma4_processor" ||
		!profile.MultimodalProcessorRoute.NativeRuntime ||
		profile.MultimodalProcessorRoute.VisionRuntime != hipKernelStatusLinked ||
		profile.MultimodalProcessorRoute.VisionProjectorRuntime != hipKernelStatusLinked ||
		profile.MultimodalProcessorRoute.SoftTokensPerImage != 512 ||
		profile.MultimodalProcessorRoute.Labels["engine_multimodal_processor_reference"] != "registered_gemma4_processor" {
		t.Fatalf("ResolveROCmModelProfile(gemma4 registered processor override) = %+v ok=%v, want profile to expose registered processor", profile, ok)
	}
}

func restoreRegisteredROCmMultimodalProcessorRoutesForTest(t *testing.T) {
	t.Helper()
	modelRoutes := rocmmodel.RegisteredMultimodalProcessorRoutes()

	t.Cleanup(func() {
		restoreRoutes := make([]rocmmodel.MultimodalProcessorRoute, 0, len(modelRoutes))
		for _, route := range modelRoutes {
			restoreRoutes = append(restoreRoutes, route.Clone())
		}
		rocmmodel.ReplaceRegisteredMultimodalProcessorRoutes(restoreRoutes)
	})
}
