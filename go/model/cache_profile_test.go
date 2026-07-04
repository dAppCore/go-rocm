// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

func TestBuildCacheProfile_Good_GenericRuntimeTopology(t *testing.T) {
	profile := BuildCacheProfile(CacheProfileOptions{
		Architecture:      "gemma4_text",
		LocalWindowTokens: 512,
		SharedLayers:      1,
	}, []CacheObservation{
		{Kind: CacheObservationKindFixed, Tokens: 512, Capacity: 512, ProcessedTokens: 640, Bounded: true, Local: true},
		{Mode: CacheModePaged, Tokens: 4096, Capacity: 8192, Bounded: true, Global: true},
		{Mode: CacheModeQ8, Tokens: 2048, Capacity: 4096, Bounded: true, Global: true},
		{Cacheless: true},
		{Kind: "runtime-specific-cache", Tokens: 12},
	})

	if !profile.Matched() || profile.Contract != CacheProfileContract || profile.Architecture != "gemma4_text" {
		t.Fatalf("BuildCacheProfile = %+v, want matched Gemma4 cache profile", profile)
	}
	if profile.TotalCaches != 4 ||
		profile.LocalCaches != 1 ||
		profile.GlobalCaches != 2 ||
		profile.SharedLayers != 1 ||
		profile.CachelessLayers != 1 {
		t.Fatalf("BuildCacheProfile counts = %+v, want 4 total, 1 local, 2 global, 1 shared, 1 cacheless", profile)
	}
	if profile.FixedCaches != 1 ||
		profile.PagedCaches != 1 ||
		profile.QuantizedCaches != 1 ||
		profile.UnknownCaches != 1 ||
		profile.UnboundedCaches != 1 {
		t.Fatalf("BuildCacheProfile kinds = %+v, want fixed/paged/quantized/unknown/unbounded counts", profile)
	}
	if profile.MaxLocalTokens != 512 ||
		profile.MaxLocalCapacity != 512 ||
		profile.MaxGlobalTokens != 4096 ||
		profile.MaxGlobalCapacity != 8192 ||
		profile.MaxCacheTokens != 4096 ||
		profile.MaxCacheCapacity != 8192 ||
		profile.MaxProcessedTokens != 4096 {
		t.Fatalf("BuildCacheProfile maxima = %+v, want observed max cache dimensions", profile)
	}
	if profile.LocalWindowLeaked {
		t.Fatalf("LocalWindowLeaked = true for bounded local cache: %+v", profile)
	}
}

func TestBuildCacheProfile_Ugly_LocalWindowLeak(t *testing.T) {
	profile := BuildCacheProfile(CacheProfileOptions{
		Architecture:      "gemma4_text",
		LocalWindowTokens: 512,
	}, []CacheObservation{
		{Kind: CacheObservationKindRotating, Tokens: 2048, Capacity: 71040, Bounded: true, Local: true},
	})

	if !profile.LocalWindowLeaked {
		t.Fatalf("BuildCacheProfile = %+v, want oversized local window leak", profile)
	}
	if profile.LocalCaches != 1 || profile.MaxLocalTokens != 2048 || profile.MaxLocalCapacity != 71040 {
		t.Fatalf("local profile = %+v, want oversized local cache recorded", profile)
	}
}

func TestBuildCacheProfile_Ugly_UnboundedLocalCacheLeaks(t *testing.T) {
	profile := BuildCacheProfile(CacheProfileOptions{
		Architecture:      "gemma4_text",
		LocalWindowTokens: 512,
	}, []CacheObservation{
		{Kind: CacheObservationKindFixed, Tokens: 64, Capacity: 0, Bounded: false, Local: true},
	})

	if !profile.LocalWindowLeaked {
		t.Fatalf("BuildCacheProfile = %+v, want unbounded local cache to leak fixed window", profile)
	}
	if profile.UnboundedCaches != 1 {
		t.Fatalf("UnboundedCaches = %d, want 1", profile.UnboundedCaches)
	}
}

func TestApplyCacheProfileLabels_Good_RuntimeContract(t *testing.T) {
	seed := map[string]string{"caller": "kept"}
	profile := BuildCacheProfile(CacheProfileOptions{
		Architecture:      "gemma4_text",
		LocalWindowTokens: 512,
		Labels:            seed,
	}, []CacheObservation{
		{Kind: CacheObservationKindFixed, Tokens: 512, Capacity: 512, ProcessedTokens: 640, Bounded: true, Local: true},
		{Mode: CacheModeQ8, Tokens: 2048, Capacity: 4096, Bounded: true, Global: true},
	})

	if seed["engine_cache_profile_contract"] != "" {
		t.Fatalf("BuildCacheProfile mutated caller labels: %+v", seed)
	}
	for key, want := range map[string]string{
		"caller":                                    "kept",
		"engine_cache_profile_contract":             CacheProfileContract,
		"engine_cache_profile_architecture":         "gemma4_text",
		"engine_cache_profile_total":                "2",
		"engine_cache_profile_local_count":          "1",
		"engine_cache_profile_global_count":         "1",
		"engine_cache_profile_local_window_tokens":  "512",
		"engine_cache_profile_max_local_tokens":     "512",
		"engine_cache_profile_max_local_capacity":   "512",
		"engine_cache_profile_max_global_tokens":    "2048",
		"engine_cache_profile_max_global_capacity":  "4096",
		"engine_cache_profile_max_cache_tokens":     "2048",
		"engine_cache_profile_max_cache_capacity":   "4096",
		"engine_cache_profile_max_processed_tokens": "2048",
		"engine_cache_profile_fixed_count":          "1",
		"engine_cache_profile_quantized_count":      "1",
		"engine_cache_profile_local_window_leaked":  "false",
	} {
		if profile.Labels[key] != want {
			t.Fatalf("profile.Labels[%q] = %q, want %q in %+v", key, profile.Labels[key], want, profile.Labels)
		}
	}

	profile.Labels["engine_cache_profile_contract"] = "mutated"
	next := BuildCacheProfile(CacheProfileOptions{Architecture: "gemma4_text"}, nil)
	if next.Labels["engine_cache_profile_contract"] != CacheProfileContract {
		t.Fatalf("BuildCacheProfile leaked mutable labels: %+v", next.Labels)
	}
}

func TestCacheObservationClone_Good_CopiesLabels(t *testing.T) {
	observation := CacheObservation{Labels: map[string]string{"a": "b"}}
	clone := observation.Clone()
	clone.Labels["a"] = "mutated"
	if observation.Labels["a"] != "b" {
		t.Fatalf("CacheObservation.Clone leaked labels: original=%+v clone=%+v", observation.Labels, clone.Labels)
	}
}
