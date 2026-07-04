// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestROCmModelRoutePlanForModel_Good_CarriesLiveCacheProfile(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "gemma4_text"}}

	_, err := model.WarmCache(context.Background(), inference.CacheWarmRequest{
		Mode:   rocmKVCacheModeQ8,
		Tokens: []int32{1, 2, 3, 4, 5},
	})
	if err != nil {
		t.Fatalf("WarmCache error = %v", err)
	}
	plan, ok := ROCmModelRoutePlanForModel(model)
	if !ok {
		t.Fatalf("ROCmModelRoutePlanForModel ok = false, want true")
	}
	assertLiveCacheProfileRoutePlan(t, plan, "ROCmModelRoutePlanForModel")

	direct := model.ModelRoutePlan()
	assertLiveCacheProfileRoutePlan(t, direct, "ModelRoutePlan")

	plan.CacheProfile.Labels["engine_cache_profile_contract"] = "mutated"
	next, ok := ROCmModelRoutePlanForModel(model)
	if !ok ||
		next.CacheProfile.Labels["engine_cache_profile_contract"] != rocmmodel.CacheProfileContract ||
		next.Labels["engine_cache_profile_contract"] != rocmmodel.CacheProfileContract {
		t.Fatalf("ROCmModelRoutePlanForModel leaked mutable cache profile labels: %+v ok=%v", next, ok)
	}

	direct.CacheProfile.Labels["engine_cache_profile_contract"] = "mutated"
	nextDirect := model.ModelRoutePlan()
	if nextDirect.CacheProfile.Labels["engine_cache_profile_contract"] != rocmmodel.CacheProfileContract ||
		nextDirect.Labels["engine_cache_profile_contract"] != rocmmodel.CacheProfileContract {
		t.Fatalf("ModelRoutePlan leaked mutable cache profile labels: %+v", nextDirect)
	}
}

func assertLiveCacheProfileRoutePlan(t *testing.T, plan ROCmModelRoutePlan, label string) {
	t.Helper()
	if !plan.Matched() ||
		!plan.CacheProfile.Matched() ||
		plan.CacheProfile.Architecture != "gemma4_text" ||
		plan.CacheProfile.TotalCaches != 1 ||
		plan.CacheProfile.QuantizedCaches != 1 ||
		plan.CacheProfile.MaxCacheTokens != 5 ||
		plan.Labels["engine_route_plan_cache_profile"] != "true" ||
		plan.Labels["engine_route_plan_cache_profile_contract"] != rocmmodel.CacheProfileContract ||
		plan.Labels["engine_cache_profile_contract"] != rocmmodel.CacheProfileContract ||
		plan.Labels["engine_route_plan_cache_profile_quantized_count"] != "1" {
		t.Fatalf("%s = %+v, want live cache profile route labels", label, plan)
	}
}
