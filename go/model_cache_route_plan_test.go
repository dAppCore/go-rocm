// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestROCmModelRoutePlan_Good_CarriesCacheFactoryRoute(t *testing.T) {
	plan, ok := ROCmModelRoutePlanForIdentity("/models/gemma4", inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels: map[string]string{
			"kv_cache_mode":  rocmmodel.CacheModeKQ8VQ4,
			"device_kv_mode": rocmmodel.CacheModeQ8,
		},
	})
	if !ok ||
		!plan.Matched() ||
		!plan.CacheRoute.Matched() ||
		plan.CacheRoute.RecommendedMode != rocmmodel.CacheModeKQ8VQ4 ||
		plan.CacheRoute.DeviceMode != rocmmodel.CacheModeQ8 ||
		plan.Labels["engine_route_plan_cache"] != "true" ||
		plan.Labels["engine_route_plan_cache_recommended_mode"] != rocmmodel.CacheModeKQ8VQ4 ||
		plan.Labels["engine_cache_factory_contract"] != rocmmodel.CacheFactoryRouteContract {
		t.Fatalf("ROCmModelRoutePlanForIdentity = %+v ok=%v, want cache factory route", plan, ok)
	}

	plan.CacheRoute.Labels["engine_cache_factory_recommended_mode"] = "mutated"
	next, ok := ROCmModelRoutePlanForIdentity("/models/gemma4", inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels:       map[string]string{"kv_cache_mode": rocmmodel.CacheModeKQ8VQ4},
	})
	if !ok || next.CacheRoute.Labels["engine_cache_factory_recommended_mode"] != rocmmodel.CacheModeKQ8VQ4 {
		t.Fatalf("ROCmModelRoutePlanForIdentity leaked mutable cache labels: %+v ok=%v", next.CacheRoute, ok)
	}
}
