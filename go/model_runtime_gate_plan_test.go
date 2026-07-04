// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestROCmModelRoutePlan_Good_CarriesRuntimeGatePlan(t *testing.T) {
	plan, ok := ROCmModelRoutePlanForIdentity("/models/gemma4-e2b-q6", inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels: map[string]string{
			"engine_feature_native_mlp_matvec":     "true",
			"engine_feature_compiled_layer_decode": "true",
		},
	})
	if !ok ||
		!plan.Matched() ||
		!plan.RuntimeGatePlan.Matched() ||
		!plan.RuntimeGatePlan.GateEnabled(rocmmodel.GateDirectGreedyToken) ||
		!plan.RuntimeGatePlan.GateEnabled(rocmmodel.GateNativeMLPMatVec) ||
		!plan.RuntimeGatePlan.GateEnabled(rocmmodel.GateCompiledLayerDecode) ||
		plan.Labels["engine_route_plan_runtime_gate"] != "true" ||
		plan.Labels["engine_route_plan_runtime_gate_ids"] == "" ||
		plan.Labels["engine_runtime_gate_ambient_env"] != "false" {
		t.Fatalf("ROCmModelRoutePlanForIdentity = %+v ok=%v, want runtime gate plan", plan, ok)
	}

	plan.RuntimeGatePlan.Labels["engine_runtime_gate_ambient_env"] = "mutated"
	next, ok := ROCmModelRoutePlanForIdentity("/models/gemma4-e2b-q6", inference.ModelIdentity{
		Architecture: "gemma4_text",
	})
	if !ok || next.RuntimeGatePlan.Labels["engine_runtime_gate_ambient_env"] != "false" {
		t.Fatalf("ROCmModelRoutePlanForIdentity leaked mutable gate labels: %+v ok=%v", next.RuntimeGatePlan, ok)
	}
}
