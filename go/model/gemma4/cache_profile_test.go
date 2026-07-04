// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "testing"

func gemma4CacheProfileTestConfig(slidingWindow int) TextConfig {
	return TextConfig{
		NumLayers:      8,
		SlidingWindow:  slidingWindow,
		KVSharedLayers: 2,
		LayerTypes: []string{
			"sliding_attention",
			"sliding_attention",
			"sliding_attention",
			"sliding_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
		},
	}
}

func TestCacheProfileOf_Good_Gemma4LocalWindowBounded(t *testing.T) {
	profile := CacheProfileOf(gemma4CacheProfileTestConfig(512), []CacheObservation{
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 71040, Tokens: 4000, Bounded: true},
	})

	if profile.TotalCaches != 6 || profile.LocalCaches != 5 || profile.GlobalCaches != 1 || profile.SharedLayers != 2 {
		t.Fatalf("CacheProfileOf = %+v, want 6 total, 5 local, 1 global, 2 shared", profile)
	}
	if profile.LocalWindowTokens != 512 || profile.MaxLocalTokens != 512 || profile.MaxLocalCapacity != 512 {
		t.Fatalf("local profile = %+v, want window/tokens/capacity capped at 512", profile)
	}
	if profile.MaxGlobalTokens != 4000 || profile.MaxGlobalCapacity != 71040 {
		t.Fatalf("global profile = %+v, want retained global cache shape", profile)
	}
	if profile.LocalWindowLeaked {
		t.Fatalf("LocalWindowLeaked = true for bounded local caches: %+v", profile)
	}
}

func TestCacheProfileOf_Ugly_Gemma4LocalWindowLeak(t *testing.T) {
	profile := CacheProfileOf(gemma4CacheProfileTestConfig(512), []CacheObservation{
		{Capacity: 71040, Tokens: 2048, Bounded: true},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 71040, Tokens: 4000, Bounded: true},
	})

	if !profile.LocalWindowLeaked {
		t.Fatalf("CacheProfileOf = %+v, want local-window leak flagged", profile)
	}
	if profile.MaxLocalTokens != 2048 || profile.MaxLocalCapacity != 71040 {
		t.Fatalf("local profile = %+v, want oversized local cache recorded", profile)
	}
}

func TestCacheProfileOf_Ugly_UnboundedLocalCacheLeaks(t *testing.T) {
	profile := CacheProfileOf(gemma4CacheProfileTestConfig(512), []CacheObservation{
		{Capacity: 0, Tokens: 64, Bounded: false},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 71040, Tokens: 4000, Bounded: true},
	})

	if !profile.LocalWindowLeaked {
		t.Fatalf("CacheProfileOf = %+v, want unbounded local cache to leak fixed window", profile)
	}
}

func TestApplyCacheProfileLabels_Good_WritesRuntimeTopologySurface(t *testing.T) {
	labels := ApplyCacheProfileLabels(nil, CacheProfileOf(gemma4CacheProfileTestConfig(512), []CacheObservation{
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 512, Tokens: 512, Bounded: true},
		{Capacity: 71040, Tokens: 4000, Bounded: true},
	}))

	for key, want := range map[string]string{
		"attention_cache_profile_total":                      "6",
		"gemma4_attention_cache_profile_total":               "6",
		"attention_cache_profile_observed_layers":            "6",
		"attention_cache_profile_local_window_tokens":        "512",
		"attention_cache_profile_local_count":                "5",
		"attention_cache_profile_global_count":               "1",
		"attention_cache_profile_shared_layers":              "2",
		"attention_cache_profile_max_local_tokens":           "512",
		"attention_cache_profile_max_local_capacity":         "512",
		"attention_cache_profile_max_global_tokens":          "4000",
		"attention_cache_profile_max_global_capacity":        "71040",
		"attention_cache_profile_local_window_leaked":        "false",
		"gemma4_attention_cache_profile_local_window_leaked": "false",
		"gemma4_attention_cache_profile_max_global_capacity": "71040",
		"gemma4_attention_cache_profile_local_window_tokens": "512",
		"gemma4_attention_cache_profile_observed_layers":     "6",
		"gemma4_attention_cache_profile_max_local_tokens":    "512",
		"gemma4_attention_cache_profile_max_local_capacity":  "512",
		"gemma4_attention_cache_profile_max_global_tokens":   "4000",
		"gemma4_attention_cache_profile_global_count":        "1",
		"gemma4_attention_cache_profile_local_count":         "5",
		"gemma4_attention_cache_profile_shared_layers":       "2",
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}
}
