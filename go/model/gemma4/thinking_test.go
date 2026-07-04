// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "testing"

type fakeSpecialTokenizer struct {
	bos bool
}

func (tokenizer fakeSpecialTokenizer) Encode(text string) []int32 {
	var ids []int32
	if tokenizer.bos {
		ids = append(ids, 2)
	}
	switch text {
	case ThinkingChannelOpenMarker:
		return append(ids, 101)
	case ThinkingChannelCloseMarker:
		return append(ids, 102)
	case "split":
		return append(ids, 101, 102)
	default:
		return ids
	}
}

func (tokenizer fakeSpecialTokenizer) HasBOSToken() bool { return tokenizer.bos }

func (fakeSpecialTokenizer) BOSToken() int32 { return 2 }

func TestThinkingChannelTokens_Good_ResolvesAtomicDelimiters(t *testing.T) {
	open, close, ok := ThinkingChannelTokens(fakeSpecialTokenizer{bos: true})
	if !ok || open != 101 || close != 102 {
		t.Fatalf("ThinkingChannelTokens() = %d,%d,%t, want 101,102,true", open, close, ok)
	}

	if _, ok := SpecialTokenID(fakeSpecialTokenizer{bos: true}, "split"); ok {
		t.Fatal("SpecialTokenID(split) ok = true, want false for multi-token marker")
	}
}

func TestApplyThinkingChannelLabels_Good_WritesMarkersAndIDs(t *testing.T) {
	labels := ApplyThinkingChannelLabels(nil, 101, 102)
	for key, want := range map[string]string{
		"gemma4_thinking_channel":          "true",
		"gemma4_thinking_channel_open":     ThinkingChannelOpenMarker,
		"gemma4_thinking_channel_close":    ThinkingChannelCloseMarker,
		"gemma4_thinking_channel_open_id":  "101",
		"gemma4_thinking_channel_close_id": "102",
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}
}
