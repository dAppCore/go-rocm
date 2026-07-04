// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestROCmModelRoutePlan_Good_CarriesRuntimeContractRoute(t *testing.T) {
	plan, ok := ROCmModelRoutePlanForIdentity("/models/gemma4-e2b-q6", inference.ModelIdentity{
		Architecture: "gemma4_text",
	})
	if !ok ||
		!plan.Matched() ||
		!plan.RuntimeContractRoute.Matched() ||
		!plan.RuntimeContractRoute.LastTokenLogits ||
		!plan.RuntimeContractRoute.CacheTopology ||
		!plan.RuntimeContractRoute.FixedSlidingCache ||
		plan.Labels["engine_route_plan_runtime_contract"] != "true" ||
		plan.Labels["engine_route_plan_runtime_contract_contract"] != ROCmModelRuntimeContractRegistryContract ||
		plan.Labels["engine_route_plan_runtime_contract_fixed_sliding_cache"] != "true" ||
		plan.Labels["engine_runtime_contract_route_contract"] != ROCmModelRuntimeContractRegistryContract ||
		plan.Labels["engine_runtime_contract_fixed_sliding_cache"] != "true" ||
		plan.Labels["engine_route_plan_runtime_contract_ids"] == "" {
		t.Fatalf("ROCmModelRoutePlanForIdentity = %+v ok=%v, want runtime contract route", plan, ok)
	}

	plan.RuntimeContractRoute.Labels["engine_runtime_contract_architecture"] = "mutated"
	if len(plan.RuntimeContractRoute.ContractIDs) > 0 {
		plan.RuntimeContractRoute.ContractIDs[0] = rocmmodel.RuntimeContractID("mutated")
	}
	next, ok := ROCmModelRoutePlanForIdentity("/models/gemma4-e2b-q6", inference.ModelIdentity{
		Architecture: "gemma4_text",
	})
	if !ok ||
		next.RuntimeContractRoute.Labels["engine_runtime_contract_architecture"] != "gemma4_text" ||
		next.RuntimeContractRoute.ContractIDs[0] != rocmmodel.RuntimeContractLastTokenLogits {
		t.Fatalf("ROCmModelRoutePlanForIdentity leaked mutable runtime contract route: %+v ok=%v", next.RuntimeContractRoute, ok)
	}
}
