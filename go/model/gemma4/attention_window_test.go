// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "testing"

func TestCachedAttentionAllowed_Good_OffsetsAndWindow(t *testing.T) {
	got := CachedAttentionAllowed(2, 5, 3, 0, 2)
	want := []bool{
		false, false, true, true, false,
		false, false, false, true, true,
	}
	assertBoolSliceEqual(t, got, want, "allowed")

	ranges := CachedAttentionWindows(2, 5, 3, 0, 2)
	assertAttentionWindowEqual(t, ranges[0], AttentionWindowRange{QueryPosition: 3, KeyStart: 2, KeyEnd: 4}, "range[0]")
	assertAttentionWindowEqual(t, ranges[1], AttentionWindowRange{QueryPosition: 4, KeyStart: 3, KeyEnd: 5}, "range[1]")
}

func TestCachedAttentionAllowed_Good_TrimmedKeyStart(t *testing.T) {
	got := CachedAttentionAllowed(2, 5, 8, 5, 4)
	want := []bool{
		true, true, true, true, false,
		false, true, true, true, true,
	}
	assertBoolSliceEqual(t, got, want, "allowed")

	ranges := CachedAttentionWindows(2, 5, 8, 5, 4)
	assertAttentionWindowEqual(t, ranges[0], AttentionWindowRange{QueryPosition: 8, KeyStart: 5, KeyEnd: 9}, "range[0]")
	assertAttentionWindowEqual(t, ranges[1], AttentionWindowRange{QueryPosition: 9, KeyStart: 6, KeyEnd: 10}, "range[1]")
}

func TestCanUseOffsetCausalAttention_Good_FollowsGemma4Bounds(t *testing.T) {
	if !CanUseOffsetCausalAttention(2, 5, 4) {
		t.Fatal("CanUseOffsetCausalAttention(2,5,4) = false, want true")
	}
	if CanUseOffsetCausalAttention(1, 5, 4) {
		t.Fatal("CanUseOffsetCausalAttention(queryLen=1) = true, want false")
	}
	if CanUseOffsetCausalAttention(2, 0, 4) {
		t.Fatal("CanUseOffsetCausalAttention(keyLen=0) = true, want false")
	}
	if !CanUseOffsetCausalAttention(8, 4096, 0) {
		t.Fatal("CanUseOffsetCausalAttention(no window) = false, want true")
	}
	if CanUseOffsetCausalAttention(513, 1024, 512) {
		t.Fatal("CanUseOffsetCausalAttention(query past window) = true, want false")
	}
	if CanUseOffsetCausalAttention(512, 1024, 512) {
		t.Fatal("CanUseOffsetCausalAttention(key range past compact window) = true, want false")
	}
}

func TestSlidingCausalContextLen_Good_CapsToVisibleWindow(t *testing.T) {
	assertIntEqual(t, SlidingCausalContextLen(512, 1024, 512), 1023, "full window chunk")
	assertIntEqual(t, SlidingCausalContextLen(128, 2048, 512), 639, "small chunk")
	assertIntEqual(t, SlidingCausalContextLen(513, 2048, 512), 2048, "query larger than window")
	assertIntEqual(t, SlidingCausalContextLen(2, 5, 0), 5, "no window")
	assertIntEqual(t, SlidingCausalContextLen(1, 5, 512), 5, "single query")
}

func TestFixedSingleTokenCausalWindow_Good_ValidatesCapacityAndOffset(t *testing.T) {
	window, ok := FixedSingleTokenCausalWindow(5, 2)
	if !ok {
		t.Fatal("FixedSingleTokenCausalWindow(5,2) ok = false, want true")
	}
	assertAttentionWindowEqual(t, window, AttentionWindowRange{QueryPosition: 2, KeyStart: 0, KeyEnd: 3}, "window")
	assertBoolSliceEqual(t, FixedSingleTokenCausalAllowed(5, 2), []bool{true, true, true, false, false}, "allowed")

	for _, tc := range []struct {
		capacity int
		offset   int
	}{
		{0, 0},
		{5, -1},
		{5, 5},
	} {
		if got, ok := FixedSingleTokenCausalWindow(tc.capacity, tc.offset); ok || got != (AttentionWindowRange{}) {
			t.Fatalf("FixedSingleTokenCausalWindow(%d,%d) = %+v,%t, want zero,false", tc.capacity, tc.offset, got, ok)
		}
	}
}

func TestMaxSpeculativeVerifyProposals_Good_StaysWithinSlidingWindow(t *testing.T) {
	assertIntEqual(t, MaxSpeculativeVerifyProposals(4, 0), 4, "no window")
	assertIntEqual(t, MaxSpeculativeVerifyProposals(4, 5), 4, "fits window")
	assertIntEqual(t, MaxSpeculativeVerifyProposals(4, 3), 2, "capped by window")
	assertIntEqual(t, MaxSpeculativeVerifyProposals(-1, 3), 0, "negative draft")
}

func TestApplyConfigLabels_Good_WritesAttentionWindowPolicySurface(t *testing.T) {
	labels := ApplyConfigLabels(nil, TextConfig{SlidingWindow: 512})
	for key, want := range map[string]string{
		"attention_window_policy":                         "sliding_causal",
		"gemma4_attention_window_policy":                  "sliding_causal",
		"attention_window_tokens":                         "512",
		"gemma4_attention_window_tokens":                  "512",
		"attention_mask_dense_sliding_prefill":            "true",
		"gemma4_attention_mask_dense_sliding_prefill":     "true",
		"attention_mask_cached_offset_causal":             "true",
		"gemma4_attention_mask_cached_offset_causal":      "true",
		"attention_mask_fixed_single_token":               "true",
		"gemma4_attention_mask_fixed_single_token":        "true",
		"attention_offset_causal_fast_path":               "true",
		"gemma4_attention_offset_causal_fast_path":        "true",
		"attention_sliding_context_trim":                  "true",
		"gemma4_attention_sliding_context_trim":           "true",
		"speculative_verify_proposal_window_limit":        "511",
		"gemma4_speculative_verify_proposal_window_limit": "511",
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}

	dense := ApplyConfigLabels(nil, TextConfig{})
	if dense["attention_window_policy"] != "" || dense["attention_mask_cached_offset_causal"] != "" {
		t.Fatalf("dense labels = %+v, want no sliding-window mask policy", dense)
	}
}

func assertAttentionWindowEqual(t *testing.T, got, want AttentionWindowRange, name string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %+v, want %+v", name, got, want)
	}
}

func assertBoolSliceEqual(t *testing.T, got, want []bool, name string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(%s) = %d, want %d: %+v", name, len(got), len(want), got)
	}
	for index, wantValue := range want {
		if got[index] != wantValue {
			t.Fatalf("%s[%d] = %t, want %t in %+v", name, index, got[index], wantValue, got)
		}
	}
}
