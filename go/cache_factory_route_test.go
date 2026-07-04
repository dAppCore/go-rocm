// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestROCmCacheFactoryRoute_Good_RootAPIMirrorsModelRegistry(t *testing.T) {
	routes := DefaultROCmCacheModeRoutes()
	modes := make([]string, 0, len(routes))
	for _, route := range routes {
		modes = append(modes, route.Mode)
	}
	for _, mode := range []string{
		ROCmCacheModeDefault,
		ROCmCacheModeQ8,
		ROCmCacheModeKQ8VQ4,
		ROCmCacheModeMLALatent,
		ROCmCacheModeRecurrent,
		ROCmCacheModeBlockPrefix,
	} {
		if !slices.Contains(modes, mode) {
			t.Fatalf("DefaultROCmCacheModeRoutes modes = %v, want %s", modes, mode)
		}
	}

	modeRoute, ok := ROCmCacheModeRouteForMode("q8")
	if !ok ||
		modeRoute.Mode != ROCmCacheModeQ8 ||
		modeRoute.Runtime != ROCmCacheRuntimeHIP ||
		!modeRoute.NativeKV ||
		!modeRoute.DeviceKV ||
		!modeRoute.Quantized ||
		modeRoute.Labels["engine_cache_mode"] != ROCmCacheModeQ8 {
		t.Fatalf("ROCmCacheModeRouteForMode(q8) = %+v ok=%v, want native q8 cache mode", modeRoute, ok)
	}

	route, ok := ROCmCacheRouteForIdentity("/models/gemma4", inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels: map[string]string{
			"kv_cache_mode":  ROCmCacheModeKQ8VQ4,
			"device_kv_mode": ROCmCacheModeQ8,
		},
	})
	if !ok ||
		!route.Matched() ||
		route.Contract != ROCmCacheFactoryRouteContract ||
		route.Name != ROCmCacheFactoryRouteName ||
		route.RecommendedMode != ROCmCacheModeKQ8VQ4 ||
		route.DeviceMode != ROCmCacheModeQ8 ||
		!route.SupportsKV ||
		!route.SupportsDevice ||
		route.Labels["engine_cache_factory_recommended_mode"] != ROCmCacheModeKQ8VQ4 ||
		route.Labels["engine_cache_factory_device_mode"] != ROCmCacheModeQ8 {
		t.Fatalf("ROCmCacheRouteForIdentity = %+v ok=%v, want root cache factory route", route, ok)
	}
	route.Labels["engine_cache_factory_recommended_mode"] = "mutated"
	next, ok := ROCmCacheRouteForIdentity("/models/gemma4", inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels:       map[string]string{"kv_cache_mode": ROCmCacheModeKQ8VQ4},
	})
	if !ok || next.Labels["engine_cache_factory_recommended_mode"] != ROCmCacheModeKQ8VQ4 {
		t.Fatalf("ROCmCacheRouteForIdentity leaked mutable labels: %+v ok=%v", next, ok)
	}
}

func TestROCmCacheFactoryRoute_Good_ProfileAndModelResolvers(t *testing.T) {
	profile, ok := ResolveROCmModelProfile("/models/gemma4", inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels: map[string]string{
			"kv_cache_mode":  ROCmCacheModeKQ8VQ4,
			"device_kv_mode": ROCmCacheModeQ8,
		},
	})
	if !ok {
		t.Fatal("ResolveROCmModelProfile(gemma4_text) ok=false")
	}
	if !profile.CacheRoute.Matched() ||
		profile.CacheRoute.RecommendedMode != ROCmCacheModeKQ8VQ4 ||
		profile.CacheRoute.DeviceMode != ROCmCacheModeQ8 {
		t.Fatalf("ResolveROCmModelProfile cache route = %+v, want profile-owned cache route", profile.CacheRoute)
	}

	route, ok := ROCmCacheRouteForProfile(profile)
	if !ok ||
		!route.Matched() ||
		route.Architecture != "gemma4_text" ||
		route.RecommendedMode != ROCmCacheModeKQ8VQ4 ||
		route.DeviceMode != ROCmCacheModeQ8 {
		t.Fatalf("ROCmCacheRouteForProfile = %+v ok=%v, want profile cache route", route, ok)
	}

	model := &rocmRegistryAPITestProfileModel{profile: profile}
	model.modelType = "gemma4_text"
	model.info = inference.ModelInfo{Architecture: "gemma4_text"}
	modelRoute, ok := ROCmCacheRouteForModel(model)
	if !ok ||
		!modelRoute.Matched() ||
		modelRoute.RecommendedMode != ROCmCacheModeKQ8VQ4 ||
		modelRoute.DeviceMode != ROCmCacheModeQ8 {
		t.Fatalf("ROCmCacheRouteForModel = %+v ok=%v, want loaded-model cache route", modelRoute, ok)
	}
	modelRoute.Labels["engine_cache_factory_recommended_mode"] = "mutated"
	next, ok := ROCmCacheRouteForModel(model)
	if !ok || next.Labels["engine_cache_factory_recommended_mode"] != ROCmCacheModeKQ8VQ4 {
		t.Fatalf("ROCmCacheRouteForModel leaked mutable labels: %+v ok=%v", next, ok)
	}
}
