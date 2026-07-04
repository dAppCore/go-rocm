// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestRuntimeAuthorPlanForIdentity_Good_Gemma4RuntimeAuthorSurface(t *testing.T) {
	plan, ok := RuntimeAuthorPlanForIdentity("/models/gemma4", inference.ModelIdentity{
		Architecture: "Gemma4ForConditionalGeneration",
		Labels: map[string]string{
			"engine_architecture_resolved": "gemma4_text",
		},
	})
	if !ok ||
		!plan.Matched() ||
		plan.Contract != RuntimeAuthorPlanContract ||
		plan.Architecture != "gemma4_text" ||
		plan.Family != "gemma4" ||
		!plan.NativeRuntime ||
		!plan.TextRuntime ||
		!plan.ModelAccess ||
		!plan.TokenCodec ||
		!plan.RuntimeGuard ||
		!plan.ParallelSlotGate ||
		!plan.PromptCacheLock ||
		!plan.DeviceGuard ||
		!plan.RequestFixedCache ||
		!plan.FixedSlidingCacheSize ||
		!plan.CacheSnapshotSafe ||
		!plan.CacheRestore ||
		!plan.CacheProfile ||
		!plan.ModelProfile ||
		!plan.ModelRoutePlan ||
		!plan.HasCapability(RuntimeAuthorUnderlyingModel) ||
		!plan.HasCapability(RuntimeAuthorRuntimeTokenizer) ||
		!plan.HasCapability(RuntimeAuthorRequireTextRuntime) ||
		!plan.HasCapability(RuntimeAuthorWithDevice) ||
		!plan.HasCapability(RuntimeAuthorRuntimeCachesSnapshotSafe) ||
		!plan.HasCapability(RuntimeAuthorCacheProfile) {
		t.Fatalf("RuntimeAuthorPlanForIdentity = %+v ok=%v, want Gemma4 runtime author plan", plan, ok)
	}
	for _, id := range []RuntimeAuthorCapabilityID{
		RuntimeAuthorUnderlyingModel,
		RuntimeAuthorRuntimeTokenizer,
		RuntimeAuthorRequireTextRuntime,
		RuntimeAuthorAcquireSlot,
		RuntimeAuthorAcquirePromptCache,
		RuntimeAuthorWithDevice,
		RuntimeAuthorNewCachesWithRequestFixedSize,
		RuntimeAuthorGenerationFixedCacheSize,
		RuntimeAuthorRuntimeCachesSnapshotSafe,
		RuntimeAuthorPromptCacheEnabled,
		RuntimeAuthorSetLastErr,
		RuntimeAuthorSetLastMetrics,
		RuntimeAuthorRestoreCaches,
		RuntimeAuthorCacheProfile,
		RuntimeAuthorModelRoutePlan,
	} {
		if !slices.Contains(plan.CapabilityIDs, id) {
			t.Fatalf("RuntimeAuthorPlanForIdentity missing capability %q in %+v", id, plan.CapabilityIDs)
		}
	}
	if plan.Labels["engine_runtime_author_plan_contract"] != RuntimeAuthorPlanContract ||
		plan.Labels["engine_runtime_author_architecture"] != "gemma4_text" ||
		plan.Labels["engine_runtime_author_underlying_model"] != "true" ||
		plan.Labels["engine_runtime_author_cache_profile"] != "true" ||
		plan.Labels["engine_runtime_author_capability_count"] == "" ||
		plan.Labels["engine_runtime_author_capability_ids"] == "" {
		t.Fatalf("RuntimeAuthorPlanForIdentity labels = %+v, want runtime author labels", plan.Labels)
	}
}

func TestRuntimeAuthorPlanForIdentity_Good_AttachedDrafterRuntime(t *testing.T) {
	plan, ok := RuntimeAuthorPlanForIdentity("/models/gemma4/assistant", inference.ModelIdentity{
		Architecture: "Gemma4AssistantForCausalLM",
	})
	if !ok ||
		!plan.Matched() ||
		plan.Architecture != "gemma4_assistant" ||
		plan.TextRuntime ||
		!plan.AttachedDrafterRuntime ||
		!plan.HiddenPromptCache ||
		!plan.HasCapability(RuntimeAuthorAttachedDrafterRuntime) ||
		!plan.HasCapability(RuntimeAuthorPromptCacheEntryHidden) {
		t.Fatalf("RuntimeAuthorPlanForIdentity(assistant) = %+v ok=%v, want attached drafter author surface", plan, ok)
	}
}

func TestRouteSet_Good_CarriesRuntimeAuthorPlan(t *testing.T) {
	set, ok := RouteSetForIdentity("/models/gemma4", inference.ModelIdentity{
		Architecture: "gemma4_text",
	})
	if !ok ||
		!set.Matched() ||
		!set.RuntimeAuthorPlan.Matched() ||
		!set.RuntimeAuthorPlan.HasCapability(RuntimeAuthorCacheProfile) ||
		set.Labels["engine_route_set_runtime_author"] != "true" ||
		set.Labels["engine_route_set_runtime_author_count"] == "" ||
		set.Labels["engine_runtime_author_plan_contract"] != RuntimeAuthorPlanContract {
		t.Fatalf("RouteSetForIdentity = %+v ok=%v, want runtime author plan", set, ok)
	}

	set.RuntimeAuthorPlan.Labels["engine_runtime_author_plan_contract"] = "mutated"
	next, ok := RouteSetForIdentity("/models/gemma4", inference.ModelIdentity{
		Architecture: "gemma4_text",
	})
	if !ok || next.RuntimeAuthorPlan.Labels["engine_runtime_author_plan_contract"] != RuntimeAuthorPlanContract {
		t.Fatalf("RouteSetForIdentity leaked mutable runtime author labels: %+v ok=%v", next.RuntimeAuthorPlan, ok)
	}
}

func TestRuntimeAuthorPlanForIdentity_Bad_UnknownArchitectureNeedsRealRoute(t *testing.T) {
	plan, ok := RuntimeAuthorPlanForIdentity("/models/unknown", inference.ModelIdentity{
		Architecture: "not-a-real-architecture",
	})
	if ok || plan.Matched() {
		t.Fatalf("RuntimeAuthorPlanForIdentity = %+v ok=%v, want no author plan for unknown architecture", plan, ok)
	}
}
