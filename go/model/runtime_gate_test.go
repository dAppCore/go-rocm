// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestRuntimeGatePlan_Good_RouteSetCarriesReactiveGates(t *testing.T) {
	set, ok := RouteSetForIdentity("/models/gemma4-e2b-q6", inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels: map[string]string{
			"engine_feature_native_mlp_matvec":          "true",
			"engine_feature_native_q6_bitstream_matvec": "linked",
			"engine_feature_fixed_sliding_cache_bound":  "ready",
			"engine_feature_compiled_layer_decode":      "enabled",
			"attention_mask_fixed_single_token":         "true",
		},
	})
	if !ok || !set.Matched() || !set.RuntimeGatePlan.Matched() {
		t.Fatalf("RouteSetForIdentity = %+v ok=%v, want runtime gate plan", set, ok)
	}
	plan := set.RuntimeGatePlan
	for _, gate := range []RuntimeGateID{
		GateGenerationStream,
		GateDirectGreedyToken,
		GateFixedSlidingCache,
		GateNativeMLPMatVec,
		GateNativeQ6BitstreamMatVec,
		GateFixedSlidingCacheBound,
		GateFixedSharedMask,
		GateCompiledLayerDecode,
	} {
		if !plan.GateEnabled(gate) {
			t.Fatalf("RuntimeGatePlan missing gate %s: %+v", gate, plan)
		}
	}
	if !slices.Contains(plan.GateIDs, GateNativeMLPMatVec) ||
		plan.Labels["engine_runtime_gate_plan_contract"] != RuntimeGatePlanContract ||
		plan.Labels["engine_runtime_gate_native_mlp_matvec"] != "true" ||
		set.Labels["engine_route_set_runtime_gate"] != "true" ||
		set.Labels["engine_runtime_gate_ambient_env"] != "false" {
		t.Fatalf("RuntimeGatePlan labels = %+v route labels=%+v, want gate labels", plan.Labels, set.Labels)
	}
	plan.Labels["engine_runtime_gate_native_mlp_matvec"] = "mutated"
	next, _ := RouteSetForIdentity("/models/gemma4-e2b-q6", inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels:       map[string]string{"engine_feature_native_mlp_matvec": "true"},
	})
	if next.RuntimeGatePlan.Labels["engine_runtime_gate_native_mlp_matvec"] != "true" {
		t.Fatalf("RuntimeGatePlan leaked mutable labels: %+v", next.RuntimeGatePlan.Labels)
	}
}

func TestRuntimeGatePlan_Good_AmbientEnvIgnored(t *testing.T) {
	t.Setenv("GO_MLX_ENABLE_NATIVE_MLP_MATVEC", "1")
	t.Setenv("GO_MLX_ENABLE_FIXED_SLIDING_CACHE", "1")

	set, ok := RouteSetForIdentity("/models/qwen", inference.ModelIdentity{Architecture: "qwen3"})
	if !ok || !set.RuntimeGatePlan.Matched() {
		t.Fatalf("RouteSetForIdentity(qwen3) = %+v ok=%v, want model-derived runtime gate plan", set, ok)
	}
	if set.RuntimeGatePlan.GateEnabled(GateNativeMLPMatVec) {
		t.Fatalf("RuntimeGatePlan enabled %s from ambient env: %+v", GateNativeMLPMatVec, set.RuntimeGatePlan)
	}
	if set.RuntimeGatePlan.GateEnabled(GateFixedSlidingCache) {
		t.Fatalf("RuntimeGatePlan enabled %s from ambient env: %+v", GateFixedSlidingCache, set.RuntimeGatePlan)
	}
	if !set.RuntimeGatePlan.GateEnabled(GateGenerationStream) {
		t.Fatalf("RuntimeGatePlan missing generation stream from qwen3 feature route: %+v", set.RuntimeGatePlan)
	}
}

func TestRuntimeGatePlan_Good_IdentityResolver(t *testing.T) {
	plan, ok := RuntimeGatePlanForIdentity("/models/moe", inference.ModelIdentity{
		Architecture: "qwen3_moe",
		Labels:       map[string]string{"engine_feature_async_decode_prefetch": "on"},
	})
	if !ok ||
		!plan.GateEnabled(GateSortedExpertPrefill) ||
		!plan.GateEnabled(GateAsyncDecodePrefetch) ||
		plan.Labels["engine_runtime_gate_ids"] == "" {
		t.Fatalf("RuntimeGatePlanForIdentity = %+v ok=%v, want MoE gate plan", plan, ok)
	}
}
