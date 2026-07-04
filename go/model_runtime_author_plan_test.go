// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestROCmModelRoutePlan_Good_CarriesRuntimeAuthorPlan(t *testing.T) {
	plan, ok := ROCmModelRoutePlanForIdentity("/models/gemma4-e2b-q6", inference.ModelIdentity{
		Architecture: "gemma4_text",
	})
	if !ok ||
		!plan.Matched() ||
		!plan.RuntimeAuthorPlan.Matched() ||
		!plan.RuntimeAuthorPlan.HasCapability(rocmmodel.RuntimeAuthorUnderlyingModel) ||
		!plan.RuntimeAuthorPlan.HasCapability(rocmmodel.RuntimeAuthorRuntimeTokenizer) ||
		!plan.RuntimeAuthorPlan.HasCapability(rocmmodel.RuntimeAuthorCacheProfile) ||
		plan.Labels["engine_route_plan_runtime_author"] != "true" ||
		plan.Labels["engine_route_plan_runtime_author_ids"] == "" ||
		plan.Labels["engine_runtime_author_plan_contract"] != rocmmodel.RuntimeAuthorPlanContract {
		t.Fatalf("ROCmModelRoutePlanForIdentity = %+v ok=%v, want runtime author plan", plan, ok)
	}

	plan.RuntimeAuthorPlan.Labels["engine_runtime_author_plan_contract"] = "mutated"
	next, ok := ROCmModelRoutePlanForIdentity("/models/gemma4-e2b-q6", inference.ModelIdentity{
		Architecture: "gemma4_text",
	})
	if !ok || next.RuntimeAuthorPlan.Labels["engine_runtime_author_plan_contract"] != rocmmodel.RuntimeAuthorPlanContract {
		t.Fatalf("ROCmModelRoutePlanForIdentity leaked mutable author labels: %+v ok=%v", next.RuntimeAuthorPlan, ok)
	}
}
