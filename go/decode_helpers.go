// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	inferdecode "dappco.re/go/inference/decode"
)

const (
	defaultROCmPromptLookupMinMatch = 2
	defaultROCmPromptLookupMaxDraft = 16
)

// SpeculativeDecodeConfig configures the ROCm package helper over the shared
// backend-neutral speculative decode harness.
type SpeculativeDecodeConfig struct {
	Prompt      string
	MaxTokens   int
	DraftTokens int
}

// PromptLookupDecodeConfig configures the ROCm package helper over the shared
// backend-neutral prompt-lookup decode harness.
type PromptLookupDecodeConfig struct {
	Prompt       string
	MaxTokens    int
	LookupTokens []int32
	MinMatch     int
	MaxDraft     int
}

// SpeculativeDecode compares draft model output against target model output
// using the shared go-inference/decode acceptance algorithm. It is a package
// helper; it does not imply production ROCm decode kernels are linked.
func SpeculativeDecode(ctx context.Context, target, draft inference.TextModel, cfg SpeculativeDecodeConfig) (inferdecode.Result, error) {
	if target == nil {
		return inferdecode.Result{}, core.E("rocm.SpeculativeDecode", "target model is required", nil)
	}
	if draft == nil {
		return inferdecode.Result{}, core.E("rocm.SpeculativeDecode", "draft model is required", nil)
	}
	return inferdecode.Speculative(ctx, inferdecode.SpeculativeConfig{
		Prompt:         cfg.Prompt,
		MaxTokens:      cfg.MaxTokens,
		DraftTokens:    cfg.DraftTokens,
		TargetGenerate: rocmDecodeGenerateFunc(target),
		DraftGenerate:  rocmDecodeGenerateFunc(draft),
	})
}

// PromptLookupDecode derives or accepts prompt-lookup candidates and compares
// them against target model output using the shared go-inference/decode
// acceptance algorithm.
func PromptLookupDecode(ctx context.Context, target inference.TextModel, cfg PromptLookupDecodeConfig) (inferdecode.Result, error) {
	if target == nil {
		return inferdecode.Result{}, core.E("rocm.PromptLookupDecode", "target model is required", nil)
	}
	lookupTokens, err := rocmPromptLookupTokens(target, cfg)
	if err != nil {
		return inferdecode.Result{}, err
	}
	return inferdecode.PromptLookup(ctx, inferdecode.PromptLookupConfig{
		Prompt:         cfg.Prompt,
		MaxTokens:      cfg.MaxTokens,
		LookupTokens:   lookupTokens,
		TargetGenerate: rocmDecodeGenerateFunc(target),
	})
}

func rocmDecodeGenerateFunc(model inference.TextModel) inferdecode.GenerateFunc {
	return func(ctx context.Context, prompt string, cfg inferdecode.GenerateConfig) (inferdecode.Generation, error) {
		if model == nil {
			return inferdecode.Generation{}, core.E("rocm.Decode.Generate", "model is required", nil)
		}
		var opts []inference.GenerateOption
		if cfg.MaxTokens > 0 {
			opts = append(opts, inference.WithMaxTokens(cfg.MaxTokens))
		}
		tokens := []inferdecode.Token{}
		for token := range model.Generate(ctx, prompt, opts...) {
			tokens = append(tokens, rocmDecodeToken(token))
		}
		if err := model.Err(); err != nil {
			return inferdecode.Generation{}, core.E("rocm.Decode.Generate", "model generation failed", err)
		}
		return inferdecode.Generation{Tokens: tokens, Text: inferdecode.TokensText(tokens)}, nil
	}
}

func rocmPromptLookupTokens(model inference.TextModel, cfg PromptLookupDecodeConfig) ([]inferdecode.Token, error) {
	tokenIDs := append([]int32(nil), cfg.LookupTokens...)
	if len(tokenIDs) == 0 {
		encoder, ok := model.(interface {
			Encode(string) []int32
		})
		if !ok {
			return nil, core.E("rocm.PromptLookupDecode", "lookup tokens are required when model does not expose Encode", nil)
		}
		minMatch := cfg.MinMatch
		if minMatch <= 0 {
			minMatch = defaultROCmPromptLookupMinMatch
		}
		maxDraft := cfg.MaxDraft
		if maxDraft <= 0 {
			maxDraft = cfg.MaxTokens
		}
		if maxDraft <= 0 {
			maxDraft = defaultROCmPromptLookupMaxDraft
		}
		var err error
		tokenIDs, err = rocmReferencePromptLookupDraft(encoder.Encode(cfg.Prompt), minMatch, maxDraft)
		if err != nil {
			return nil, err
		}
	}
	return rocmDecodeTokens(model, tokenIDs), nil
}

func rocmDecodeTokens(model inference.TextModel, ids []int32) []inferdecode.Token {
	out := make([]inferdecode.Token, len(ids))
	decoder, _ := model.(interface {
		Decode([]int32) string
	})
	for i, id := range ids {
		text := core.Sprintf("%d", id)
		if decoder != nil {
			if decoded := decoder.Decode([]int32{id}); decoded != "" {
				text = decoded
			}
		}
		out[i] = inferdecode.Token{ID: id, Text: text}
	}
	return out
}

func rocmDecodeToken(token inference.Token) inferdecode.Token {
	return inferdecode.Token{ID: token.ID, Text: token.Text}
}
