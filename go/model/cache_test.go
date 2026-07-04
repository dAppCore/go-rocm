// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestDefaultCacheModeRoutes_Good_MatchesROCmCacheFactoryModes(t *testing.T) {
	routes := DefaultCacheModeRoutes()
	if len(routes) == 0 {
		t.Fatal("DefaultCacheModeRoutes returned no modes")
	}
	byMode := map[string]CacheModeRoute{}
	for _, route := range routes {
		byMode[route.Mode] = route
	}
	for _, mode := range []string{
		SequenceMixerCacheModeDefault,
		CacheModeFP16,
		CacheModeQ8,
		CacheModeKQ8VQ4,
		CacheModePaged,
		CacheModeFixed,
		CacheModeTurboQuant,
		SequenceMixerCacheModeMLALatent,
		SequenceMixerCacheModeCompaction,
		SequenceMixerCacheModeCompactionFull,
		SequenceMixerCacheModeRecurrent,
		CacheModeBlockPrefix,
		CacheModeRetained,
		CacheModeAttached,
	} {
		if route, ok := byMode[mode]; !ok || !route.Matched() {
			t.Fatalf("DefaultCacheModeRoutes missing %q in %+v", mode, routes)
		}
	}
	if !byMode[CacheModeQ8].NativeKV ||
		!byMode[CacheModeQ8].DeviceKV ||
		!byMode[CacheModeQ8].Quantized ||
		byMode[CacheModeQ8].RuntimeStatus != inference.FeatureRuntimeNative {
		t.Fatalf("q8 route = %+v, want native quantized device KV", byMode[CacheModeQ8])
	}
	if !byMode[SequenceMixerCacheModeRecurrent].Recurrent ||
		!byMode[SequenceMixerCacheModeRecurrent].MetadataOnly {
		t.Fatalf("recurrent route = %+v, want metadata recurrent holder", byMode[SequenceMixerCacheModeRecurrent])
	}
	routes[0].Labels["engine_cache_mode"] = "mutated"
	next := DefaultCacheModeRoutes()
	if next[0].Labels["engine_cache_mode"] == "mutated" {
		t.Fatalf("DefaultCacheModeRoutes leaked mutable labels: %+v", next[0])
	}
}

func TestCacheRouteForIdentity_Good_ProfileHintsAndIdentityModes(t *testing.T) {
	route, ok := CacheRouteForIdentity("/models/gemma4", inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels: map[string]string{
			"kv_cache_mode":  CacheModeKQ8VQ4,
			"device_kv_mode": CacheModeQ8,
		},
	})
	if !ok ||
		!route.Matched() ||
		route.Architecture != "gemma4_text" ||
		route.Family != "gemma4" ||
		route.RecommendedMode != CacheModeKQ8VQ4 ||
		route.DeviceMode != CacheModeQ8 ||
		!slices.Contains(route.CacheHints, CacheModeKQ8VQ4) ||
		!slices.Contains(route.ModeNames, SequenceMixerCacheModeDefault) ||
		!route.SupportsKV ||
		!route.SupportsDevice ||
		route.Labels["engine_cache_factory_recommended_mode"] != CacheModeKQ8VQ4 ||
		route.Labels["engine_cache_factory_device_mode"] != CacheModeQ8 {
		t.Fatalf("CacheRouteForIdentity = %+v ok=%v, want Gemma4 cache route", route, ok)
	}
	route.Labels["engine_cache_factory_recommended_mode"] = "mutated"
	next, ok := CacheRouteForIdentity("/models/gemma4", inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels:       map[string]string{"kv_cache_mode": CacheModeKQ8VQ4},
	})
	if !ok || next.Labels["engine_cache_factory_recommended_mode"] != CacheModeKQ8VQ4 {
		t.Fatalf("CacheRouteForIdentity leaked mutable labels: %+v ok=%v", next, ok)
	}
}

func TestRouteSet_Good_CarriesCacheFactoryRoute(t *testing.T) {
	set, ok := RouteSetForIdentity("/models/hybrid", inference.ModelIdentity{
		Architecture: "hybrid",
		Labels:       map[string]string{"kv_cache_mode": SequenceMixerCacheModeMLALatent},
	})
	if !ok ||
		!set.Matched() ||
		!set.CacheRoute.Matched() ||
		set.CacheRoute.RecommendedMode != SequenceMixerCacheModeMLALatent ||
		!slices.Contains(set.CacheRoute.CacheHints, SequenceMixerCacheModeRecurrent) ||
		set.Labels["engine_route_set_cache"] != "true" ||
		set.Labels["engine_route_set_cache_recommended_mode"] != SequenceMixerCacheModeMLALatent ||
		set.Labels["engine_cache_factory_contract"] != CacheFactoryRouteContract {
		t.Fatalf("RouteSetForIdentity = %+v ok=%v, want cache route", set, ok)
	}
}
