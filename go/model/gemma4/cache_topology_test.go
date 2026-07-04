// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "testing"

func TestLayerTypesOf_Good_DefaultPatternForcesFinalFull(t *testing.T) {
	got := LayerTypesOf(TextConfig{
		NumLayers:            7,
		SlidingWindowPattern: 3,
	})
	want := []string{
		"sliding_attention",
		"sliding_attention",
		"full_attention",
		"sliding_attention",
		"sliding_attention",
		"full_attention",
		"full_attention",
	}
	assertIntEqual(t, len(got), len(want), "len(layerTypes)")
	for index, wantType := range want {
		if got[index] != wantType {
			t.Fatalf("LayerTypesOf[%d] = %q, want %q in %+v", index, got[index], wantType, got)
		}
	}
}

func TestBuildCacheLayout_Good_SharedOwners(t *testing.T) {
	previous, cacheIndexByLayer := BuildCacheLayout([]string{
		"sliding_attention",
		"full_attention",
		"sliding_attention",
		"full_attention",
	}, 2)

	assertIntSliceEqual(t, previous, []int{0, 1, 0, 1}, "previous")
	assertIntSliceEqual(t, cacheIndexByLayer, []int{0, 1, -1, -1}, "cacheIndexByLayer")
}

func TestBuildCacheLayout_Good_PromotesMissingOwner(t *testing.T) {
	previous, cacheIndexByLayer := BuildCacheLayout([]string{
		"sliding_attention",
		"sliding_attention",
		"sliding_attention",
		"sliding_attention",
		"full_attention",
		"sliding_attention",
	}, 2)

	assertIntSliceEqual(t, previous, []int{0, 1, 2, 3, 4, 3}, "previous")
	assertIntSliceEqual(t, cacheIndexByLayer, []int{0, 1, 2, 3, 4, -1}, "cacheIndexByLayer")
}

func TestCacheTopologyOf_Good_E4BSharedLayoutUsesLayerTypes(t *testing.T) {
	topology := CacheTopologyOf(TextConfig{
		NumLayers:         42,
		LayerTypes:        testPatternLayerTypes(42, 6),
		SlidingWindow:     512,
		KVSharedLayers:    18,
		KVSharedLayersSet: true,
	})

	assertIntEqual(t, topology.OwnerCaches, 24, "OwnerCaches")
	assertIntEqual(t, topology.LocalCaches, 20, "LocalCaches")
	assertIntEqual(t, topology.GlobalCaches, 4, "GlobalCaches")
	assertIntEqual(t, topology.SharedLayers, 18, "SharedLayers")
	assertIntEqual(t, topology.PreviousKVByLayer[24], 22, "PreviousKVByLayer[24]")
	assertIntEqual(t, topology.PreviousKVByLayer[29], 23, "PreviousKVByLayer[29]")
	assertIntEqual(t, topology.PreviousKVByLayer[41], 23, "PreviousKVByLayer[41]")
	assertIntEqual(t, topology.CacheIndexByLayer[24], -1, "CacheIndexByLayer[24]")
	assertIntEqual(t, topology.FixedSlidingPrefillCap, 512, "FixedSlidingPrefillCap")
}

func TestCacheTopologyOf_Good_E2BSharedLayoutUsesLayerTypes(t *testing.T) {
	topology := CacheTopologyOf(TextConfig{
		NumLayers:         35,
		LayerTypes:        testPatternLayerTypes(35, 5),
		SlidingWindow:     512,
		KVSharedLayers:    20,
		KVSharedLayersSet: true,
	})

	assertIntEqual(t, topology.LocalCaches, 12, "LocalCaches")
	assertIntEqual(t, topology.GlobalCaches, 3, "GlobalCaches")
	assertIntEqual(t, topology.OwnerCaches, 15, "OwnerCaches")
	assertIntEqual(t, topology.SharedLayers, 20, "SharedLayers")
	assertIntEqual(t, topology.PreviousKVByLayer[15], 13, "PreviousKVByLayer[15]")
	assertIntEqual(t, topology.PreviousKVByLayer[19], 14, "PreviousKVByLayer[19]")
	assertIntEqual(t, topology.PreviousKVByLayer[34], 14, "PreviousKVByLayer[34]")
}

func TestAttentionCacheLayout_Good_MapsSharedLayersToOwnerCaches(t *testing.T) {
	sharedOwners := AttentionCacheLayout(TextConfig{
		NumLayers:      4,
		LayerTypes:     []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"},
		KVSharedLayers: 2,
	}, 4, 2)
	assertIntSliceEqual(t, sharedOwners, []int{0, 1, 0, 1}, "sharedOwners")

	promotedOwner := AttentionCacheLayout(TextConfig{
		NumLayers:      6,
		LayerTypes:     []string{"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention"},
		KVSharedLayers: 2,
	}, 6, 5)
	assertIntSliceEqual(t, promotedOwner, []int{0, 1, 2, 3, 4, 3}, "promotedOwner")
}

func TestFixedSlidingPrefillChunkLimit_Good_CapsAtSmallerFixedCache(t *testing.T) {
	cfg := TextConfig{SlidingWindow: 512}
	assertIntEqual(t, FixedSlidingPrefillChunkLimit(cfg), 512, "window")
	assertIntEqual(t, FixedSlidingPrefillChunkLimit(cfg, 1024, 256), 256, "smaller fixed cache")
	assertIntEqual(t, FixedSlidingPrefillChunkLimit(TextConfig{}, 256), 0, "no window")
}

func TestApplyConfigLabels_Good_WritesCacheTopologySurface(t *testing.T) {
	labels := ApplyConfigLabels(nil, TextConfig{
		NumLayers:            4,
		LayerTypes:           []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"},
		SlidingWindow:        512,
		SlidingWindowPattern: 2,
		KVSharedLayers:       2,
		KVSharedLayersSet:    true,
	})
	for key, want := range map[string]string{
		"attention_layer_count":                      "4",
		"gemma4_attention_layer_count":               "4",
		"attention_layer_types":                      "sliding_attention,full_attention,sliding_attention,full_attention",
		"gemma4_attention_layer_types":               "sliding_attention,full_attention,sliding_attention,full_attention",
		"attention_cache_owner_by_layer":             "0,1,0,1",
		"gemma4_attention_cache_owner_by_layer":      "0,1,0,1",
		"attention_cache_index_by_layer":             "0,1,-1,-1",
		"gemma4_attention_cache_index_by_layer":      "0,1,-1,-1",
		"attention_cache_local_window_tokens":        "512",
		"gemma4_attention_cache_local_window_tokens": "512",
		"attention_cache_owner_count":                "2",
		"attention_cache_local_count":                "1",
		"attention_cache_global_count":               "1",
		"attention_cache_shared_layers":              "2",
		"gemma4_fixed_sliding_prefill_chunk_limit":   "512",
		"fixed_sliding_prefill_chunk_limit":          "512",
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}
}

func testPatternLayerTypes(numLayers, pattern int) []string {
	layerTypes := make([]string, numLayers)
	for index := range layerTypes {
		if pattern > 1 && (index+1)%pattern != 0 {
			layerTypes[index] = "sliding_attention"
		} else {
			layerTypes[index] = "full_attention"
		}
	}
	if len(layerTypes) > 0 {
		layerTypes[len(layerTypes)-1] = "full_attention"
	}
	return layerTypes
}

func assertIntEqual(t *testing.T, got, want int, name string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %d, want %d", name, got, want)
	}
}

func assertIntSliceEqual(t *testing.T, got, want []int, name string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(%s) = %d, want %d: %+v", name, len(got), len(want), got)
	}
	for index, wantValue := range want {
		if got[index] != wantValue {
			t.Fatalf("%s[%d] = %d, want %d in %+v", name, index, got[index], wantValue, got)
		}
	}
}
