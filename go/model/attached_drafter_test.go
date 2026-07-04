// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestDefaultAttachedDrafterRoutes_Good_StaticCatalogue(t *testing.T) {
	routes := DefaultAttachedDrafterRoutes()
	if len(routes) == 0 {
		t.Fatal("DefaultAttachedDrafterRoutes returned no routes")
	}
	byArchitecture := map[string]AttachedDrafterRoute{}
	for _, route := range routes {
		byArchitecture[route.Architecture] = route
	}
	target := byArchitecture["gemma4_text"]
	if !target.Matched() ||
		target.Contract != AttachedDrafterRegistryContract ||
		target.Name != AttachedDrafterRouteName ||
		target.Reference != "go_mlx_gemma4_assistant_pair" ||
		target.Runtime != AttachedDrafterRuntimeMetadata ||
		target.RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		target.Status != AttachedDrafterRouteNativePending ||
		target.Role != "target" ||
		target.TargetArchitecture != "gemma4_text" ||
		target.AssistantArchitecture != "gemma4_assistant" ||
		target.TargetRuntime != AttachedDrafterGemma4RuntimeMLXAffine ||
		target.AssistantRuntime != AttachedDrafterGemma4RuntimeBF16 ||
		target.TargetGenerateStatus != AttachedDrafterGemma4GenerateLinked ||
		target.AssistantGenerateStatus != AttachedDrafterGemma4GenerateLoadOnly ||
		target.NativeAttachment != KernelStatusNotLinked ||
		target.ExecutionStatus != KernelStatusNotLinked ||
		target.Fallback != "refused" ||
		!target.Registered ||
		target.NativeRuntime ||
		!target.Target ||
		target.Assistant ||
		target.AttachedOnly ||
		target.StandaloneGeneration ||
		!target.PairValidation ||
		!target.FamilyPairRequired ||
		!target.OfficialPairKnown ||
		!target.OfficialPairLocked ||
		!target.RetainedStateRequired ||
		!target.RuntimeOwnedKV ||
		!target.PromptReplayRefused ||
		!target.DraftDetection ||
		!target.ExplicitDraft ||
		!target.AutoDetectAssistantDir ||
		!target.AutoDetectSiblingPair ||
		!target.AutoDetectMTPDir ||
		!target.AutoDetectMTPSiblingGGUF ||
		!target.TuneProfile ||
		!target.FourLayerDrafter ||
		!target.OrderedEmbeddings ||
		!target.CentroidRouting ||
		!target.BorrowTargetKV ||
		!target.VerifyForward ||
		target.NativeGeneration ||
		target.NativeStateGeneration ||
		!target.FallbackRefused ||
		!target.Staged ||
		!target.Planned ||
		target.DefaultDraftTokens != AttachedDrafterDefaultDraftTokens ||
		target.DefaultDraftBlock != 5 ||
		target.AssistantCentroids != AttachedDrafterAssistantCentroids ||
		target.AssistantCentroidIntermediateTopK != AttachedDrafterAssistantIntermediateTopK ||
		!slices.Equal(target.AssistantTokenOrderingShape, []int{2048, 128}) ||
		!slices.Contains(target.TargetSizes, "E2B") ||
		!slices.Contains(target.TargetQuantModes, "q5") ||
		!slices.Contains(target.TargetQuantModes, "mxfp4") ||
		!slices.Contains(target.AssistantQuantModes, "q6") ||
		!slices.Contains(target.AssistantQuantModes, "nvfp4") ||
		!slices.Contains(target.AssistantModelIDs, "google/gemma-4-E2B-it-assistant") ||
		!slices.Contains(target.AssistantModelIDs, "mlx-community/gemma-4-E2B-it-qat-assistant-6bit") ||
		!slices.Contains(target.AssistantModelIDs, "mlx-community/gemma-4-12B-it-qat-assistant-4bit") ||
		!slices.Contains(target.DetectionSources, "assistant-dir") ||
		!slices.Equal(target.Capabilities, []inference.CapabilityID{
			inference.CapabilitySpeculativeDecode,
			inference.CapabilityStateBundle,
			inference.CapabilityStateWake,
			inference.CapabilityStateSleep,
			inference.CapabilityStateFork,
		}) ||
		target.Labels["engine_attached_drafter_reference"] != "go_mlx_gemma4_assistant_pair" ||
		target.Labels["engine_attached_drafter_role"] != "target" ||
		target.Labels["engine_attached_drafter_capabilities"] != "speculative.decode,state.bundle,state.wake,state.sleep,state.fork" {
		t.Fatalf("gemma4_text attached drafter route = %+v, want planned target/assistant pair route", target)
	}
	assistant := byArchitecture["gemma4_assistant"]
	if !assistant.Matched() ||
		assistant.Status != AttachedDrafterRouteAttachedOnly ||
		assistant.Role != "assistant" ||
		!assistant.Assistant ||
		!assistant.AttachedOnly ||
		assistant.Target ||
		assistant.TargetArchitecture != "gemma4_text" ||
		assistant.AssistantArchitecture != "gemma4_assistant" ||
		assistant.Labels["engine_attached_drafter_attached_only"] != "true" {
		t.Fatalf("gemma4_assistant attached drafter route = %+v, want attached-only assistant route", assistant)
	}
}

func TestAttachedDrafterRouteForInspection_Good_UsesLabels(t *testing.T) {
	inspectionLabels := map[string]string{
		"engine_architecture_profile":                       "gemma4_text",
		"engine_attached_drafter_reference":                 "fixture_pair",
		"engine_attached_drafter_role":                      "target",
		"engine_attached_drafter_native_attachment":         KernelStatusLinked,
		"engine_attached_drafter_execution_status":          "ready",
		"attached_drafter_target_gemma4_runtime":            AttachedDrafterGemma4RuntimeMLXAffine,
		"attached_drafter_target_gemma4_generate_status":    AttachedDrafterGemma4GenerateLinked,
		"attached_drafter_assistant_gemma4_runtime":         AttachedDrafterGemma4RuntimeBF16,
		"attached_drafter_assistant_gemma4_generate_status": AttachedDrafterGemma4GenerateLoadOnly,
		"attached_drafter_retained_state_required":          "true",
		"attached_drafter_prompt_replay_fallback":           "forbidden",
		"attached_drafter_official_pair_verified":           "true",
		"attached_drafter_gemma4_family_pair_verified":      "true",
		"speculative_draft_tokens":                          "6",
		"reactive_draft_block":                              "4",
	}
	inspection := &inference.ModelPackInspection{
		Path:   "/models/gemma4",
		Labels: inspectionLabels,
		Model: inference.ModelIdentity{
			Architecture: "Gemma4ForCausalLM",
			Labels: map[string]string{
				"engine_architecture_resolved": "gemma4_text",
			},
		},
	}
	route, ok := AttachedDrafterRouteForInspection(inspection)
	if !ok ||
		route.Architecture != "gemma4_text" ||
		route.Reference != "fixture_pair" ||
		route.Runtime != AttachedDrafterRuntimeMetadata ||
		route.RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		route.Status != AttachedDrafterRouteNativePending ||
		route.NativeAttachment != KernelStatusLinked ||
		route.ExecutionStatus != "ready" ||
		route.NativeRuntime ||
		route.NativeGeneration ||
		route.NativeStateGeneration ||
		!route.Staged ||
		!route.Planned ||
		!route.FallbackRefused ||
		route.DefaultDraftTokens != 6 ||
		route.DefaultDraftBlock != 4 ||
		!route.RetainedStateRequired ||
		!route.PromptReplayRefused ||
		!route.OfficialPairLocked ||
		!route.FamilyPairRequired ||
		route.Labels["engine_attached_drafter_reference"] != "fixture_pair" ||
		route.Labels["engine_attached_drafter_execution_status"] != "ready" ||
		route.Labels["engine_attached_drafter_default_draft_tokens"] != "6" {
		t.Fatalf("AttachedDrafterRouteForInspection = %+v ok=%v, want label-derived native attached route", route, ok)
	}
	inspectionLabels["engine_attached_drafter_reference"] = "mutated"
	route.TargetSizes[0] = "mutated"
	next, ok := AttachedDrafterRouteForArchitecture("gemma4_text")
	if !ok ||
		next.Reference != "go_mlx_gemma4_assistant_pair" ||
		slices.Contains(next.TargetSizes, "mutated") {
		t.Fatalf("AttachedDrafterRouteForInspection leaked mutable state: %+v ok=%v", next, ok)
	}
}

func TestRegisterAttachedDrafterRoute_Good_ExtendsReactiveCatalogue(t *testing.T) {
	restoreRegisteredAttachedDraftersForTest(t)

	RegisterAttachedDrafterRoute(AttachedDrafterRoute{})
	RegisterAttachedDrafterRoute(AttachedDrafterRoute{
		Architecture: "fake-loader",
		Reference:    "first_attached_drafter",
		Target:       true,
	})
	RegisterAttachedDrafterRoute(AttachedDrafterRoute{
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

	if got := RegisteredAttachedDrafterArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredAttachedDrafterArchitectures = %v, want normalized replacement", got)
	}
	registeredRoutes := RegisteredAttachedDrafterRoutes()
	if len(registeredRoutes) != 1 ||
		registeredRoutes[0].Architecture != "fake_loader" ||
		registeredRoutes[0].Reference != "fake_attached_drafter" {
		t.Fatalf("RegisteredAttachedDrafterRoutes = %+v, want one replacement route", registeredRoutes)
	}
	routesForReplace := RegisteredAttachedDrafterRoutes()
	registeredRoutes[0].Labels["engine_attached_drafter_reference"] = "mutated"
	registeredRoutes[0].TargetSizes[0] = "mutated"
	nextRegistered, ok := RegisteredAttachedDrafterRouteForArchitecture("fake-loader")
	if !ok ||
		nextRegistered.Labels["engine_attached_drafter_reference"] != "fake_attached_drafter" ||
		!slices.Equal(nextRegistered.TargetSizes, []string{"tiny"}) {
		t.Fatalf("RegisteredAttachedDrafterRoutes leaked mutable state: %+v ok=%v", nextRegistered, ok)
	}
	ReplaceRegisteredAttachedDrafterRoutes(routesForReplace)

	route, ok := AttachedDrafterRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != AttachedDrafterRegistryContract ||
		route.Name != AttachedDrafterRouteName ||
		route.Architecture != "fake_loader" ||
		route.Family != "fake" ||
		route.Reference != "fake_attached_drafter" ||
		route.Runtime != AttachedDrafterRuntimeMetadata ||
		route.RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		route.Status != AttachedDrafterRouteNativePending ||
		route.Role != "target" ||
		route.TargetArchitecture != "fake_loader" ||
		route.AssistantArchitecture != "fake_assistant" ||
		route.NativeAttachment != KernelStatusNotLinked ||
		route.ExecutionStatus != KernelStatusNotLinked ||
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
		route.Labels["engine_attached_drafter_fallback_refused"] != "true" ||
		route.Labels["engine_attached_drafter_capabilities"] != "speculative.decode,state.bundle,state.wake,state.sleep,state.fork" {
		t.Fatalf("AttachedDrafterRouteForArchitecture(fake-loader) = %+v ok=%v, want registered route", route, ok)
	}
	route.Labels["engine_attached_drafter_reference"] = "mutated"
	route.TargetSizes[0] = "mutated"
	route.Capabilities[0] = inference.CapabilityGenerate
	next, ok := AttachedDrafterRouteForArchitecture("fake-loader")
	if !ok ||
		next.Labels["engine_attached_drafter_reference"] != "fake_attached_drafter" ||
		!slices.Equal(next.TargetSizes, []string{"tiny"}) ||
		next.Capabilities[0] != inference.CapabilitySpeculativeDecode {
		t.Fatalf("AttachedDrafterRouteForArchitecture leaked mutable state: %+v ok=%v", next, ok)
	}
	if !slices.ContainsFunc(DefaultAttachedDrafterRoutes(), func(route AttachedDrafterRoute) bool {
		return route.Architecture == "fake_loader" && route.Reference == "fake_attached_drafter"
	}) {
		t.Fatalf("DefaultAttachedDrafterRoutes missing fake_loader registration")
	}
}

func restoreRegisteredAttachedDraftersForTest(t *testing.T) {
	t.Helper()
	routes := RegisteredAttachedDrafterRoutes()
	t.Cleanup(func() {
		ReplaceRegisteredAttachedDrafterRoutes(routes)
	})
}
