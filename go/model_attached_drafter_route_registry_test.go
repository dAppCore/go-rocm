// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestRegisterROCmAttachedDrafterRoute_Good_ExtendsReactiveDrafterRegistry(t *testing.T) {
	restoreRegisteredROCmAttachedDrafterRoutesForTest(t)

	RegisterROCmAttachedDrafterRoute(ROCmAttachedDrafterRoute{})
	RegisterROCmAttachedDrafterRoute(ROCmAttachedDrafterRoute{
		Architecture: "fake-loader",
		Reference:    "first_attached_drafter",
		Target:       true,
	})
	RegisterROCmAttachedDrafterRoute(ROCmAttachedDrafterRoute{
		Architecture:                      "fake-loader",
		Family:                            "fake",
		Reference:                         "fake_attached_drafter",
		Target:                            true,
		TargetArchitecture:                "fake_loader",
		AssistantArchitecture:             "fake_assistant",
		TargetFamily:                      "fake",
		AssistantFamily:                   "fake",
		DefaultDraftTokens:                6,
		DefaultDraftBlock:                 4,
		AssistantCentroids:                1024,
		AssistantCentroidIntermediateTopK: 16,
		AssistantTokenOrderingShape:       []int{1024, 64},
		TargetSizes:                       []string{"tiny"},
		TargetQuantModes:                  []string{"q8"},
		AssistantQuantModes:               []string{"bf16"},
		AssistantModelIDs:                 []string{"fake/assistant"},
		DetectionSources:                  []string{"flag"},
		RequiredDraftTokenSweeps:          []int{2, 4},
		TunableDraftBlocks:                []int{4},
		RequiredMetrics:                   []string{"fake_metric"},
	})

	if got := RegisteredROCmAttachedDrafterRouteArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmAttachedDrafterRouteArchitectures = %v, want normalized replacement under one architecture", got)
	}
	registered := RegisteredROCmAttachedDrafterRouteArchitectures()
	registered[0] = "mutated"
	if next := RegisteredROCmAttachedDrafterRouteArchitectures(); !slices.Equal(next, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmAttachedDrafterRouteArchitectures returned mutable state: %v", next)
	}

	route, ok := ROCmAttachedDrafterRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != ROCmAttachedDrafterRegistryContract ||
		route.Name != rocmAttachedDrafterRegistryRouteName ||
		route.Architecture != "fake_loader" ||
		route.Family != "fake" ||
		route.Reference != "fake_attached_drafter" ||
		route.Runtime != rocmAttachedDrafterRuntimeMetadata ||
		route.RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		route.Status != ROCmAttachedDrafterRouteNativePending ||
		route.Role != "target" ||
		route.TargetArchitecture != "fake_loader" ||
		route.AssistantArchitecture != "fake_assistant" ||
		route.NativeAttachment != hipKernelStatusNotLinked ||
		route.ExecutionStatus != hipKernelStatusNotLinked ||
		route.Fallback != "refused" ||
		!route.Registered ||
		route.NativeRuntime ||
		!route.Target ||
		route.Assistant ||
		route.AttachedOnly ||
		route.StandaloneGeneration ||
		!route.PairValidation ||
		!route.FamilyPairRequired ||
		!route.OfficialPairKnown ||
		!route.OfficialPairLocked ||
		!route.RetainedStateRequired ||
		!route.RuntimeOwnedKV ||
		!route.PromptReplayRefused ||
		!route.DraftDetection ||
		!route.ExplicitDraft ||
		!route.AutoDetectAssistantDir ||
		!route.AutoDetectSiblingPair ||
		!route.AutoDetectMTPDir ||
		!route.AutoDetectMTPSiblingGGUF ||
		!route.TuneProfile ||
		!route.FourLayerDrafter ||
		!route.OrderedEmbeddings ||
		!route.CentroidRouting ||
		!route.BorrowTargetKV ||
		!route.VerifyForward ||
		route.NativeGeneration ||
		route.NativeStateGeneration ||
		!route.FallbackRefused ||
		!route.Staged ||
		!route.Planned ||
		route.DefaultDraftTokens != 6 ||
		route.DefaultDraftBlock != 4 ||
		route.AssistantCentroids != 1024 ||
		route.AssistantCentroidIntermediateTopK != 16 ||
		!slices.Equal(route.AssistantTokenOrderingShape, []int{1024, 64}) ||
		!slices.Equal(route.TargetSizes, []string{"tiny"}) ||
		!slices.Equal(route.TargetQuantModes, []string{"q8"}) ||
		!slices.Equal(route.AssistantQuantModes, []string{"bf16"}) ||
		!slices.Equal(route.AssistantModelIDs, []string{"fake/assistant"}) ||
		!slices.Equal(route.DetectionSources, []string{"flag"}) ||
		!slices.Equal(route.RequiredDraftTokenSweeps, []int{2, 4}) ||
		!slices.Equal(route.TunableDraftBlocks, []int{4}) ||
		!slices.Equal(route.RequiredMetrics, []string{"fake_metric"}) ||
		!slices.Equal(route.Capabilities, []inference.CapabilityID{
			inference.CapabilitySpeculativeDecode,
			inference.CapabilityStateBundle,
			inference.CapabilityStateWake,
			inference.CapabilityStateSleep,
			inference.CapabilityStateFork,
		}) ||
		route.Labels["engine_attached_drafter_reference"] != "fake_attached_drafter" ||
		route.Labels["engine_attached_drafter_role"] != "target" ||
		route.Labels["engine_attached_drafter_native_attachment"] != hipKernelStatusNotLinked ||
		route.Labels["engine_attached_drafter_fallback_refused"] != "true" ||
		route.Labels["engine_attached_drafter_capabilities"] != "speculative.decode,state.bundle,state.wake,state.sleep,state.fork" {
		t.Fatalf("ROCmAttachedDrafterRouteForArchitecture(fake-loader) = %+v ok=%v, want registered attached drafter route", route, ok)
	}

	route.Labels["engine_attached_drafter_reference"] = "mutated"
	route.TargetSizes[0] = "mutated"
	route.Capabilities[0] = inference.CapabilityGenerate
	nextRoute, ok := ROCmAttachedDrafterRouteForArchitecture("fake-loader")
	if !ok ||
		nextRoute.Labels["engine_attached_drafter_reference"] != "fake_attached_drafter" ||
		!slices.Equal(nextRoute.TargetSizes, []string{"tiny"}) ||
		nextRoute.Capabilities[0] != inference.CapabilitySpeculativeDecode {
		t.Fatalf("ROCmAttachedDrafterRouteForArchitecture leaked mutable state: %+v ok=%v", nextRoute, ok)
	}
	modelRoute, ok := rocmmodel.RegisteredAttachedDrafterRouteForArchitecture("fake-loader")
	if !ok ||
		modelRoute.Reference != "fake_attached_drafter" ||
		!slices.Equal(modelRoute.TargetSizes, []string{"tiny"}) ||
		!slices.Equal(modelRoute.AssistantModelIDs, []string{"fake/assistant"}) ||
		modelRoute.Labels["engine_attached_drafter_reference"] != "fake_attached_drafter" {
		t.Fatalf("model.RegisteredAttachedDrafterRouteForArchitecture(fake-loader) = %+v ok=%v, want mirrored route", modelRoute, ok)
	}

	defaults := DefaultROCmAttachedDrafterRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmAttachedDrafterRoute) bool {
		return route.Architecture == "fake_loader" && route.Reference == "fake_attached_drafter"
	}) {
		t.Fatalf("DefaultROCmAttachedDrafterRoutes missing registered route: %+v", defaults)
	}

	profileRoute := ROCmAttachedDrafterRouteForProfile(ROCmModelProfile{
		Name:         "fake",
		Family:       "fake",
		Architecture: "fake_loader",
		ArchitectureProfile: ROCmArchitectureProfile{
			ID:     "fake_loader",
			Family: "fake",
		},
	})
	if profileRoute.Reference != "fake_attached_drafter" ||
		!profileRoute.Target ||
		profileRoute.Assistant ||
		profileRoute.DefaultDraftTokens != 6 ||
		profileRoute.AssistantCentroids != 1024 ||
		profileRoute.Labels["engine_attached_drafter_reference"] != "fake_attached_drafter" {
		t.Fatalf("ROCmAttachedDrafterRouteForProfile registered route = %+v, want profile to use registered drafter route", profileRoute)
	}
}

func TestRegisterROCmAttachedDrafterRoute_Good_SeesModelPackageRegistration(t *testing.T) {
	restoreRegisteredROCmAttachedDrafterRoutesForTest(t)

	rocmmodel.RegisterAttachedDrafterRoute(rocmmodel.AttachedDrafterRoute{
		Architecture:                      "folder-drafter",
		Family:                            "folder",
		Reference:                         "folder_attached_drafter",
		Target:                            true,
		TargetArchitecture:                "folder_drafter",
		AssistantArchitecture:             "folder_assistant",
		TargetFamily:                      "folder",
		AssistantFamily:                   "folder",
		DefaultDraftTokens:                6,
		DefaultDraftBlock:                 4,
		AssistantCentroids:                1024,
		AssistantCentroidIntermediateTopK: 16,
		AssistantTokenOrderingShape:       []int{1024, 64},
		TargetSizes:                       []string{"tiny"},
		TargetQuantModes:                  []string{"q8"},
		AssistantQuantModes:               []string{"bf16"},
		AssistantModelIDs:                 []string{"folder/assistant"},
		DetectionSources:                  []string{"flag"},
		RequiredDraftTokenSweeps:          []int{2, 4},
		TunableDraftBlocks:                []int{4},
		RequiredMetrics:                   []string{"folder_metric"},
	})

	route, ok := ROCmAttachedDrafterRouteForArchitecture("folder-drafter")
	if !ok ||
		route.Contract != ROCmAttachedDrafterRegistryContract ||
		route.Name != rocmAttachedDrafterRegistryRouteName ||
		route.Architecture != "folder_drafter" ||
		route.Family != "folder" ||
		route.Reference != "folder_attached_drafter" ||
		route.Runtime != rocmAttachedDrafterRuntimeMetadata ||
		route.RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		route.Status != ROCmAttachedDrafterRouteNativePending ||
		route.Role != "target" ||
		route.TargetArchitecture != "folder_drafter" ||
		route.AssistantArchitecture != "folder_assistant" ||
		route.NativeAttachment != hipKernelStatusNotLinked ||
		route.ExecutionStatus != hipKernelStatusNotLinked ||
		route.Fallback != "refused" ||
		!route.Registered ||
		route.NativeRuntime ||
		!route.Target ||
		route.Assistant ||
		route.AttachedOnly ||
		route.StandaloneGeneration ||
		!route.PairValidation ||
		!route.RetainedStateRequired ||
		!route.RuntimeOwnedKV ||
		!route.PromptReplayRefused ||
		!route.DraftDetection ||
		!route.ExplicitDraft ||
		!route.AutoDetectAssistantDir ||
		!route.AutoDetectSiblingPair ||
		!route.AutoDetectMTPDir ||
		!route.AutoDetectMTPSiblingGGUF ||
		!route.TuneProfile ||
		!route.FourLayerDrafter ||
		!route.OrderedEmbeddings ||
		!route.CentroidRouting ||
		!route.BorrowTargetKV ||
		!route.VerifyForward ||
		route.NativeGeneration ||
		route.NativeStateGeneration ||
		!route.FallbackRefused ||
		!route.Staged ||
		!route.Planned ||
		route.DefaultDraftTokens != 6 ||
		route.DefaultDraftBlock != 4 ||
		route.AssistantCentroids != 1024 ||
		route.AssistantCentroidIntermediateTopK != 16 ||
		!slices.Equal(route.AssistantTokenOrderingShape, []int{1024, 64}) ||
		!slices.Equal(route.TargetSizes, []string{"tiny"}) ||
		!slices.Equal(route.TargetQuantModes, []string{"q8"}) ||
		!slices.Equal(route.AssistantQuantModes, []string{"bf16"}) ||
		!slices.Equal(route.AssistantModelIDs, []string{"folder/assistant"}) ||
		!slices.Equal(route.DetectionSources, []string{"flag"}) ||
		!slices.Equal(route.RequiredDraftTokenSweeps, []int{2, 4}) ||
		!slices.Equal(route.TunableDraftBlocks, []int{4}) ||
		!slices.Equal(route.RequiredMetrics, []string{"folder_metric"}) ||
		route.Labels["engine_attached_drafter_reference"] != "folder_attached_drafter" {
		t.Fatalf("ROCmAttachedDrafterRouteForArchitecture(folder-drafter) = %+v ok=%v, want model package route", route, ok)
	}
	defaults := DefaultROCmAttachedDrafterRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmAttachedDrafterRoute) bool {
		return route.Architecture == "folder_drafter" && route.Reference == "folder_attached_drafter"
	}) {
		t.Fatalf("DefaultROCmAttachedDrafterRoutes missing model package route: %+v", defaults)
	}
	profileRoute := ROCmAttachedDrafterRouteForProfile(ROCmModelProfile{
		Name:         "folder",
		Family:       "folder",
		Architecture: "folder_drafter",
	})
	if profileRoute.Reference != "folder_attached_drafter" ||
		!profileRoute.Target ||
		profileRoute.Assistant ||
		profileRoute.DefaultDraftTokens != 6 ||
		profileRoute.AssistantCentroids != 1024 ||
		profileRoute.Labels["engine_attached_drafter_reference"] != "folder_attached_drafter" {
		t.Fatalf("ROCmAttachedDrafterRouteForProfile(folder_drafter) = %+v, want model package drafter route", profileRoute)
	}
}

func TestRegisterROCmAttachedDrafterRoute_Good_OverridesBuiltinDrafterRoute(t *testing.T) {
	restoreRegisteredROCmAttachedDrafterRoutesForTest(t)

	RegisterROCmAttachedDrafterRoute(ROCmAttachedDrafterRoute{
		Architecture:                      "gemma4_text",
		Family:                            "gemma4",
		Reference:                         "registered_gemma4_assistant_pair",
		Target:                            true,
		TargetArchitecture:                "gemma4_text",
		AssistantArchitecture:             "gemma4_assistant",
		NativeRuntime:                     true,
		NativeAttachment:                  hipKernelStatusLinked,
		NativeGeneration:                  true,
		NativeStateGeneration:             true,
		DefaultDraftTokens:                8,
		DefaultDraftBlock:                 6,
		AssistantCentroids:                4096,
		AssistantCentroidIntermediateTopK: 32,
		AssistantTokenOrderingShape:       []int{4096, 128},
		TargetSizes:                       []string{"12B"},
		TargetQuantModes:                  []string{"q6"},
		AssistantQuantModes:               []string{"bf16"},
		AssistantModelIDs:                 []string{"google/gemma-4-12B-it-assistant"},
	})

	route, ok := ROCmAttachedDrafterRouteForArchitecture("Gemma4ForCausalLM")
	if !ok ||
		route.Architecture != "gemma4_text" ||
		route.Reference != "registered_gemma4_assistant_pair" ||
		route.Runtime != rocmAttachedDrafterRuntimeHIP ||
		route.RuntimeStatus != inference.FeatureRuntimeExperimental ||
		route.Status != ROCmAttachedDrafterRouteNativePending ||
		route.Role != "target" ||
		route.NativeAttachment != hipKernelStatusLinked ||
		route.ExecutionStatus != "ready" ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.Target ||
		route.Assistant ||
		route.AttachedOnly ||
		!route.NativeGeneration ||
		!route.NativeStateGeneration ||
		route.FallbackRefused ||
		route.Staged ||
		route.Planned ||
		route.DefaultDraftTokens != 8 ||
		route.DefaultDraftBlock != 6 ||
		route.AssistantCentroids != 4096 ||
		!slices.Equal(route.TargetSizes, []string{"12B"}) ||
		!slices.Equal(route.TargetQuantModes, []string{"q6"}) ||
		!slices.Equal(route.AssistantModelIDs, []string{"google/gemma-4-12B-it-assistant"}) ||
		route.Labels["engine_attached_drafter_reference"] != "registered_gemma4_assistant_pair" ||
		route.Labels["engine_attached_drafter_native_runtime"] != "true" ||
		route.Labels["engine_attached_drafter_execution_status"] != "ready" {
		t.Fatalf("ROCmAttachedDrafterRouteForArchitecture(gemma4_text override) = %+v ok=%v, want registered native drafter route", route, ok)
	}

	profile, ok := ResolveROCmModelProfile("/models/gemma4", inference.ModelIdentity{Architecture: "Gemma4ForCausalLM"})
	if !ok ||
		profile.AttachedDrafterRoute.Reference != "registered_gemma4_assistant_pair" ||
		!profile.AttachedDrafterRoute.NativeRuntime ||
		profile.AttachedDrafterRoute.NativeAttachment != hipKernelStatusLinked ||
		profile.AttachedDrafterRoute.ExecutionStatus != "ready" ||
		profile.AttachedDrafterRoute.DefaultDraftTokens != 8 ||
		profile.AttachedDrafterRoute.Labels["engine_attached_drafter_reference"] != "registered_gemma4_assistant_pair" {
		t.Fatalf("ResolveROCmModelProfile(gemma4 registered drafter override) = %+v ok=%v, want profile to expose registered drafter route", profile, ok)
	}
}

func restoreRegisteredROCmAttachedDrafterRoutesForTest(t *testing.T) {
	t.Helper()
	modelRoutes := rocmmodel.RegisteredAttachedDrafterRoutes()

	t.Cleanup(func() {
		restoreRoutes := make([]rocmmodel.AttachedDrafterRoute, 0, len(modelRoutes))
		for _, route := range modelRoutes {
			restoreRoutes = append(restoreRoutes, route.Clone())
		}
		rocmmodel.ReplaceRegisteredAttachedDrafterRoutes(restoreRoutes)
	})
}
