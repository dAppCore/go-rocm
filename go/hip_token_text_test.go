// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"testing"

	core "dappco.re/go"
)

func TestHIPTokenTextDecoder_Good_LoadEncodeDecode(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "tokenizer.json")
	payload := []byte(`{
		"model": {
			"vocab": {
				"<unk>": 0,
				"<bos>": 2,
				"he": 3,
				"▁": 4,
				"<0x7A>": 5
			},
			"merges": ["h e"]
		},
		"added_tokens": [
			{"id": 2, "content": "<bos>", "special": true},
			{"id": 9, "content": "<turn>", "special": true}
		]
	}`)
	write := core.WriteFile(path, payload, 0o644)
	core.RequireTrue(t, write.OK)

	decoder, err := loadHIPTokenTextDecoder(path)
	core.RequireNoError(t, err)
	core.AssertNotNil(t, decoder)
	core.AssertEqual(t, []int32{2, 3, 9, 5}, decoder.Encode("he<turn>z"))
	core.AssertEqual(t, "he z", decoder.Decode([]int32{2, 3, 4, 5, 9}))
	core.AssertEqual(t, "he", decoder.DecodeToken(3))
	core.AssertEqual(t, "", decoder.DecodeToken(9))
	core.AssertNotNil(t, loadHIPTokenTextDecoderIfPresent(path))
	core.AssertNil(t, loadHIPTokenTextDecoderIfPresent(" "))
	core.AssertNil(t, loadHIPTokenTextDecoderIfPresent(core.PathJoin(t.TempDir(), "missing.json")))
}

func TestHIPTokenTextDecoder_Bad_MergeAndFallbackEdges(t *testing.T) {
	stringRanks := hipTokenTextMergeRanks([]byte(`["a b","bad","c d"]`))
	core.AssertEqual(t, 0, stringRanks["a b"])
	core.AssertEqual(t, 2, stringRanks["c d"])
	arrayRanks := hipTokenTextMergeRanks([]byte(`[["x","y"],["bad"],["y","z"]]`))
	core.AssertEqual(t, 0, arrayRanks["x y"])
	core.AssertEqual(t, 2, arrayRanks["y z"])
	core.AssertEqual(t, 0, len(hipTokenTextMergeRanks(nil)))
	core.AssertEqual(t, 0, len(hipTokenTextMergeRanks([]byte(`{"not":"a merge list"}`))))

	decoder := &hipTokenTextDecoder{
		vocab:      map[string]int32{"<unk>": 7},
		pieces:     map[int32]string{7: "<unk>"},
		hasUnknown: true,
		unknownID:  7,
	}
	core.AssertEqual(t, []int32{7, 7}, decoder.Encode("é"))
	core.AssertEqual(t, "", (*hipTokenTextDecoder)(nil).Decode([]int32{1}))
	core.AssertEqual(t, "", decoder.DecodeToken(404))
}
