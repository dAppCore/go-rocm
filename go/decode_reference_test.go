// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	inferdecode "dappco.re/go/inference/decode"
)

func TestDecodeReferencePromptLookup_Good_ReturnsDraftAfterRepeatedSuffix(t *testing.T) {
	draft, err := rocmReferencePromptLookupDraft([]int32{1, 2, 3, 4, 1, 2}, 2, 2)

	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{3, 4}, draft)
}

func TestDecodeReferencePromptLookup_Good_NoMatchReturnsNil(t *testing.T) {
	draft, err := rocmReferencePromptLookupDraft([]int32{1, 2, 3, 4}, 2, 2)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, len(draft))
}

func TestDecodeReferencePromptLookup_Good_TruncatesDraft(t *testing.T) {
	draft, err := rocmReferencePromptLookupDraft([]int32{1, 2, 3, 4, 1, 2}, 2, 1)

	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{3}, draft)
}

func TestDecodeReferencePromptLookup_Good_UsesLongestRepeatedSuffix(t *testing.T) {
	draft, err := rocmReferencePromptLookupDraft([]int32{1, 2, 3, 4, 1, 2, 3}, 2, 4)

	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{4}, draft)
}

func TestDecodeReferencePromptLookup_Good_TooShortReturnsNil(t *testing.T) {
	draft, err := rocmReferencePromptLookupDraft([]int32{1, 2, 1}, 2, 2)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, len(draft))
}

func TestDecodeReferencePromptLookup_Bad_RejectsInvalidConfig(t *testing.T) {
	_, err := rocmReferencePromptLookupDraft([]int32{1, 2}, 0, 1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "min match")

	_, err = rocmReferencePromptLookupDraft([]int32{1, 2}, 1, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "max draft")
}

func TestDecodeReferenceSpeculativeAccept_Good_AcceptsMatchingPrefix(t *testing.T) {
	accepted, rejectedAt := rocmReferenceSpeculativeAccept([]int32{4, 5, 6}, []int32{4, 5, 7})

	core.AssertEqual(t, []int32{4, 5}, accepted)
	core.AssertEqual(t, 2, rejectedAt)
}

func TestDecodeReferenceSpeculativeAccept_Good_AcceptsAllDraftTokens(t *testing.T) {
	accepted, rejectedAt := rocmReferenceSpeculativeAccept([]int32{4, 5}, []int32{4, 5, 6})

	core.AssertEqual(t, []int32{4, 5}, accepted)
	core.AssertEqual(t, -1, rejectedAt)
}

func TestDecodeReferenceSpeculativeAccept_Good_RejectsFirstMismatch(t *testing.T) {
	accepted, rejectedAt := rocmReferenceSpeculativeAccept([]int32{4, 5}, []int32{9, 5})

	core.AssertEqual(t, 0, len(accepted))
	core.AssertEqual(t, 0, rejectedAt)
}

func TestDecodeReferenceSpeculativeAccept_Good_DraftLongerThanTargetRejectsAtTargetEnd(t *testing.T) {
	accepted, rejectedAt := rocmReferenceSpeculativeAccept([]int32{4, 5, 6}, []int32{4, 5})

	core.AssertEqual(t, []int32{4, 5}, accepted)
	core.AssertEqual(t, 2, rejectedAt)
}

func TestDecodeReferenceSpeculativeAccept_Good_EmptyDraftAcceptsAll(t *testing.T) {
	accepted, rejectedAt := rocmReferenceSpeculativeAccept(nil, []int32{4, 5})

	core.AssertEqual(t, 0, len(accepted))
	core.AssertEqual(t, -1, rejectedAt)
}

func TestDecodeHelpers_Good_SpeculativeDecodeUsesSharedHarness(t *testing.T) {
	target := &rocmModel{native: &fakeNativeModel{tokens: []inference.Token{{ID: 4, Text: "a"}, {ID: 5, Text: "b"}, {ID: 7, Text: "c"}}}}
	draft := &rocmModel{native: &fakeNativeModel{tokens: []inference.Token{{ID: 4, Text: "a"}, {ID: 5, Text: "b"}, {ID: 6, Text: "x"}}}}

	result, err := SpeculativeDecode(context.Background(), target, draft, SpeculativeDecodeConfig{Prompt: "p", MaxTokens: 3, DraftTokens: 3})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModeSpeculative, result.Mode)
	core.AssertEqual(t, 2, result.Metrics.AcceptedTokens)
	core.AssertEqual(t, 1, result.Metrics.RejectedTokens)
	core.AssertEqual(t, 3, result.Metrics.EmittedTokens)
}

func TestDecodeHelpers_Good_PromptLookupDecodeUsesLookupDraft(t *testing.T) {
	target := &rocmModel{native: &fakeNativeModel{tokens: []inference.Token{{ID: 3}, {ID: 4}, {ID: 9}}}}

	result, err := PromptLookupDecode(context.Background(), target, PromptLookupDecodeConfig{
		Prompt:       "p",
		MaxTokens:    3,
		LookupTokens: []int32{3, 4, 8},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModePromptLookup, result.Mode)
	core.AssertEqual(t, 2, result.Metrics.AcceptedTokens)
	core.AssertEqual(t, 1, result.Metrics.RejectedTokens)
	core.AssertEqual(t, 3, result.Metrics.LookupTokens)
}

func TestDecodeHelpers_Good_PromptLookupDecodeDerivesLookupTokensFromEncoder(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{
		tokens:       []inference.Token{{ID: 3}, {ID: 4}, {ID: 9}},
		encodeResult: []int32{1, 2, 3, 4, 1, 2},
	}}

	result, err := PromptLookupDecode(context.Background(), model, PromptLookupDecodeConfig{
		Prompt:    "ignored",
		MaxTokens: 3,
		MaxDraft:  2,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModePromptLookup, result.Mode)
	core.AssertEqual(t, 2, result.Metrics.AcceptedTokens)
	core.AssertEqual(t, 0, result.Metrics.RejectedTokens)
	core.AssertEqual(t, 3, result.Metrics.EmittedTokens)
	core.AssertEqual(t, 2, result.Metrics.LookupTokens)
}

func TestDecodeHelpers_Good_PromptLookupTokensUsesDecoderText(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{}}

	tokens, err := rocmPromptLookupTokens(model, PromptLookupDecodeConfig{LookupTokens: []int32{7, 8}})

	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(7), tokens[0].ID)
	core.AssertEqual(t, "1 tokens", tokens[0].Text)
	core.AssertEqual(t, int32(8), tokens[1].ID)
	core.AssertEqual(t, "1 tokens", tokens[1].Text)
}

func TestDecodeHelpers_Bad_RejectsMissingModels(t *testing.T) {
	target := &rocmModel{native: &fakeNativeModel{}}

	_, err := SpeculativeDecode(context.Background(), nil, target, SpeculativeDecodeConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "target model")

	_, err = SpeculativeDecode(context.Background(), target, nil, SpeculativeDecodeConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "draft model")

	_, err = PromptLookupDecode(context.Background(), nil, PromptLookupDecodeConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "target model")
}

func TestDecodeHelpers_Bad_PromptLookupRequiresTokensWithoutEncoder(t *testing.T) {
	model := &minimalDecodeTextModel{}

	_, err := PromptLookupDecode(context.Background(), model, PromptLookupDecodeConfig{Prompt: "1 2 1"})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "lookup tokens")
}

func TestDecodeHelpers_Ugly_PropagatesModelStreamError(t *testing.T) {
	target := &rocmModel{native: &decodeErrorNativeModel{fakeNativeModel: &fakeNativeModel{}, err: core.NewError("decode failed")}}
	draft := &rocmModel{native: &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "a"}}}}

	_, err := SpeculativeDecode(context.Background(), target, draft, SpeculativeDecodeConfig{Prompt: "p", MaxTokens: 1})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model generation failed")
	core.AssertContains(t, err.Error(), "decode failed")
}

type decodeErrorNativeModel struct {
	*fakeNativeModel
	err error
}

func (model *decodeErrorNativeModel) Generate(context.Context, string, inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	return func(yield func(inference.Token) bool) {
		yield(inference.Token{ID: 1, Text: "a"})
	}, func() error { return model.err }
}

type minimalDecodeTextModel struct{}

func (*minimalDecodeTextModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (*minimalDecodeTextModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (*minimalDecodeTextModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (*minimalDecodeTextModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (*minimalDecodeTextModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{}
}

func (*minimalDecodeTextModel) Err() error { return nil }

func (*minimalDecodeTextModel) ModelType() string { return "minimal" }

func (*minimalDecodeTextModel) Info() inference.ModelInfo { return inference.ModelInfo{} }

func (*minimalDecodeTextModel) Close() error { return nil }
