// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"bytes"
	"encoding/json"
	"strconv"
	"strings"

	core "dappco.re/go"
)

type hipTokenTextDecoder struct {
	vocab         map[string]int32
	pieces        map[int32]string
	decodedPieces []string
	mergeRanks    map[string]int
	special       map[int32]bool
	specialText   map[string]int32
	bosID         int32
	hasBOS        bool
	unknownID     int32
	hasUnknown    bool
}

type hipTokenTextDecoderJSON struct {
	Model struct {
		Vocab  map[string]int32 `json:"vocab"`
		Merges json.RawMessage  `json:"merges"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
}

func loadHIPTokenTextDecoderIfPresent(path string) *hipTokenTextDecoder {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}
	decoder, err := loadHIPTokenTextDecoder(path)
	if err != nil {
		return nil
	}
	return decoder
}

func loadHIPTokenTextDecoder(path string) (*hipTokenTextDecoder, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, read.Value.(error)
	}
	var payload hipTokenTextDecoderJSON
	if err := json.Unmarshal(read.Value.([]byte), &payload); err != nil {
		return nil, err
	}
	decoder := &hipTokenTextDecoder{
		vocab:       make(map[string]int32, len(payload.Model.Vocab)+len(payload.AddedTokens)),
		pieces:      make(map[int32]string, len(payload.Model.Vocab)+len(payload.AddedTokens)),
		mergeRanks:  hipTokenTextMergeRanks(payload.Model.Merges),
		special:     make(map[int32]bool),
		specialText: make(map[string]int32),
	}
	for piece, id := range payload.Model.Vocab {
		decoder.vocab[piece] = id
		decoder.pieces[id] = piece
	}
	for _, token := range payload.AddedTokens {
		decoder.vocab[token.Content] = token.ID
		decoder.pieces[token.ID] = token.Content
		if token.Special {
			decoder.special[token.ID] = true
			decoder.specialText[token.Content] = token.ID
		}
	}
	if unknownID, ok := decoder.vocab["<unk>"]; ok {
		decoder.unknownID = unknownID
		decoder.hasUnknown = true
	}
	if bosID, ok := decoder.vocab["<bos>"]; ok {
		decoder.bosID = bosID
		decoder.hasBOS = true
	}
	decoder.precomputeDecodedPieces()
	return decoder, nil
}

func (decoder *hipTokenTextDecoder) precomputeDecodedPieces() {
	if decoder == nil || len(decoder.pieces) == 0 {
		return
	}
	maxID := int32(-1)
	for id := range decoder.pieces {
		if id > maxID {
			maxID = id
		}
	}
	if maxID < 0 {
		return
	}
	decoded := make([]string, int(maxID)+1)
	for id, piece := range decoder.pieces {
		if id < 0 || decoder.special[id] {
			continue
		}
		decoded[id] = hipDecodeTokenTextRaw(piece)
	}
	decoder.decodedPieces = decoded
}

func hipTokenTextMergeRanks(raw json.RawMessage) map[string]int {
	if len(raw) == 0 {
		return nil
	}
	index := hipTokenTextSkipJSONSpace(raw, 0)
	if index >= len(raw) || raw[index] != '[' {
		return nil
	}
	ranks := make(map[string]int, hipTokenTextMergeRankCapacity(raw))
	index++
	for rank := 0; index < len(raw); rank++ {
		index = hipTokenTextSkipJSONListSeparator(raw, index)
		if index >= len(raw) || raw[index] == ']' {
			break
		}
		switch raw[index] {
		case '"':
			value, next, ok := hipTokenTextReadJSONString(raw, index)
			index = next
			if ok {
				left, right, ok := strings.Cut(value, " ")
				if !ok {
					continue
				}
				ranks[left+" "+right] = rank
			}
		case '[':
			left, right, next, ok := hipTokenTextReadJSONMergePair(raw, index)
			index = next
			if ok {
				ranks[left+" "+right] = rank
			}
		default:
			index = hipTokenTextSkipJSONValue(raw, index)
		}
	}
	return ranks
}

func hipTokenTextMergeRankCapacity(raw json.RawMessage) int {
	if len(raw) < 4 {
		return 0
	}
	const maxMergeRankCapacity = 1 << 20
	count := bytes.Count(raw, []byte("],"))
	if count == 0 {
		count = bytes.Count(raw, []byte(`","`))
	}
	if count > maxMergeRankCapacity {
		count = maxMergeRankCapacity
	}
	return count + 1
}

func hipTokenTextReadJSONMergePair(raw []byte, index int) (string, string, int, bool) {
	if index >= len(raw) || raw[index] != '[' {
		return "", "", index, false
	}
	index++
	var parts [2]string
	valueCount := 0
	stringParts := 0
	for index < len(raw) {
		index = hipTokenTextSkipJSONListSeparator(raw, index)
		if index >= len(raw) {
			return "", "", index, false
		}
		if raw[index] == ']' {
			index++
			return parts[0], parts[1], index, valueCount == 2 && stringParts == 2
		}
		valueCount++
		if raw[index] == '"' {
			value, next, ok := hipTokenTextReadJSONString(raw, index)
			index = next
			if ok && valueCount <= len(parts) {
				parts[valueCount-1] = value
				stringParts++
			}
			continue
		}
		index = hipTokenTextSkipJSONValue(raw, index)
	}
	return "", "", index, false
}

func hipTokenTextReadJSONString(raw []byte, index int) (string, int, bool) {
	if index >= len(raw) || raw[index] != '"' {
		return "", index, false
	}
	start := index
	index++
	escaped := false
	for index < len(raw) {
		switch raw[index] {
		case '\\':
			escaped = true
			index += 2
			continue
		case '"':
			index++
			if !escaped {
				return string(raw[start+1 : index-1]), index, true
			}
			value, err := strconv.Unquote(string(raw[start:index]))
			return value, index, err == nil
		}
		index++
	}
	return "", len(raw), false
}

func hipTokenTextSkipJSONValue(raw []byte, index int) int {
	index = hipTokenTextSkipJSONSpace(raw, index)
	if index >= len(raw) {
		return index
	}
	switch raw[index] {
	case '"':
		_, next, _ := hipTokenTextReadJSONString(raw, index)
		return next
	case '[', '{':
		depth := 0
		for index < len(raw) {
			switch raw[index] {
			case '"':
				_, next, _ := hipTokenTextReadJSONString(raw, index)
				index = next
				continue
			case '[', '{':
				depth++
			case ']', '}':
				depth--
				index++
				if depth <= 0 {
					return index
				}
				continue
			}
			index++
		}
		return index
	default:
		for index < len(raw) && raw[index] != ',' && raw[index] != ']' && raw[index] != '}' {
			index++
		}
		return index
	}
}

func hipTokenTextSkipJSONListSeparator(raw []byte, index int) int {
	for index < len(raw) {
		switch raw[index] {
		case ' ', '\n', '\r', '\t', ',':
			index++
			continue
		}
		return index
	}
	return index
}

func hipTokenTextSkipJSONSpace(raw []byte, index int) int {
	for index < len(raw) {
		switch raw[index] {
		case ' ', '\n', '\r', '\t':
			index++
			continue
		}
		return index
	}
	return index
}

func (decoder *hipTokenTextDecoder) Encode(text string) []int32 {
	if decoder == nil || text == "" {
		return nil
	}
	tokens := []int32{}
	if decoder.shouldPrependBOS(text) {
		tokens = append(tokens, decoder.bosID)
	}
	remaining := text
	for remaining != "" {
		if id, width, ok := decoder.specialPrefix(remaining); ok {
			tokens = append(tokens, id)
			remaining = remaining[width:]
			continue
		}
		end := len(remaining)
		for special := range decoder.specialText {
			if special == "" {
				continue
			}
			index := strings.Index(remaining, special)
			if index > 0 && index < end {
				end = index
			}
		}
		segment := remaining[:end]
		remaining = remaining[end:]
		tokens = append(tokens, decoder.encodeSegment(segment)...)
	}
	return tokens
}

func (decoder *hipTokenTextDecoder) shouldPrependBOS(text string) bool {
	if decoder == nil || !decoder.hasBOS {
		return false
	}
	bosText := decoder.pieces[decoder.bosID]
	return bosText == "" || !strings.HasPrefix(text, bosText)
}

func (decoder *hipTokenTextDecoder) specialPrefix(text string) (int32, int, bool) {
	for special, id := range decoder.specialText {
		if special != "" && strings.HasPrefix(text, special) {
			return id, len(special), true
		}
	}
	return 0, 0, false
}

func (decoder *hipTokenTextDecoder) encodeSegment(segment string) []int32 {
	normalized := strings.ReplaceAll(segment, " ", "\u2581")
	symbols := hipTokenTextSymbols(normalized)
	symbols = decoder.bpeMerge(symbols)
	tokens := make([]int32, 0, len(symbols))
	for _, symbol := range symbols {
		if id, ok := decoder.vocab[symbol]; ok {
			tokens = append(tokens, id)
			continue
		}
		tokens = append(tokens, decoder.byteFallbackTokens(symbol)...)
	}
	return tokens
}

func hipTokenTextSymbols(text string) []string {
	symbols := make([]string, 0, len(text))
	for _, r := range text {
		symbols = append(symbols, string(r))
	}
	return symbols
}

func (decoder *hipTokenTextDecoder) bpeMerge(symbols []string) []string {
	for len(symbols) > 1 {
		bestRank := -1
		bestIndex := -1
		for index := 0; index < len(symbols)-1; index++ {
			key := symbols[index] + " " + symbols[index+1]
			rank, ok := decoder.mergeRanks[key]
			if ok && (bestRank < 0 || rank < bestRank) {
				bestRank = rank
				bestIndex = index
			}
		}
		if bestIndex < 0 {
			return symbols
		}
		merged := symbols[bestIndex] + symbols[bestIndex+1]
		symbols[bestIndex] = merged
		copy(symbols[bestIndex+1:], symbols[bestIndex+2:])
		symbols[len(symbols)-1] = ""
		symbols = symbols[:len(symbols)-1]
	}
	return symbols
}

func (decoder *hipTokenTextDecoder) byteFallbackTokens(symbol string) []int32 {
	tokens := []int32{}
	for _, b := range []byte(symbol) {
		key := core.Sprintf("<0x%02X>", b)
		if id, ok := decoder.vocab[key]; ok {
			tokens = append(tokens, id)
		} else if decoder.hasUnknown {
			tokens = append(tokens, decoder.unknownID)
		}
	}
	return tokens
}

func (decoder *hipTokenTextDecoder) Decode(ids []int32) string {
	if decoder == nil || len(ids) == 0 {
		return ""
	}
	var raw strings.Builder
	for _, id := range ids {
		if decoder.special[id] {
			continue
		}
		piece, ok := decoder.pieces[id]
		if !ok {
			continue
		}
		raw.WriteString(piece)
	}
	return hipDecodeTokenTextRaw(raw.String())
}

func (decoder *hipTokenTextDecoder) DecodeToken(id int32) string {
	if decoder == nil || decoder.special[id] {
		return ""
	}
	if id >= 0 && int(id) < len(decoder.decodedPieces) {
		if text := decoder.decodedPieces[id]; text != "" {
			return text
		}
	}
	piece, ok := decoder.pieces[id]
	if !ok {
		return ""
	}
	return hipDecodeTokenTextRaw(piece)
}

func hipDecodeTokenTextRaw(raw string) string {
	raw = strings.ReplaceAll(raw, "\u2581", " ")
	return hipDecodeTokenTextByteFallback(raw)
}

func hipDecodeTokenTextByteFallback(raw string) string {
	if !strings.Contains(raw, "<0x") {
		return raw
	}
	var out strings.Builder
	for index := 0; index < len(raw); {
		if index+6 <= len(raw) &&
			raw[index] == '<' &&
			raw[index+1] == '0' &&
			raw[index+2] == 'x' &&
			raw[index+5] == '>' {
			value, err := strconv.ParseUint(raw[index+3:index+5], 16, 8)
			if err == nil {
				out.WriteByte(byte(value))
				index += 6
				continue
			}
		}
		out.WriteByte(raw[index])
		index++
	}
	return out.String()
}
