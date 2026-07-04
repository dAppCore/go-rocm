// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"

	"dappco.re/go/inference"
	inferdecode "dappco.re/go/inference/decode"
	root "dappco.re/go/rocm"
)

type (
	SpeculativeDecodeConfig             = root.SpeculativeDecodeConfig
	AttachedDrafterDecodeConfig         = root.AttachedDrafterDecodeConfig
	AttachedDrafterGenerateConfig       = root.AttachedDrafterGenerateConfig
	AttachedDrafterStateGenerateRequest = root.AttachedDrafterStateGenerateRequest
	AttachedDrafterPlan                 = root.AttachedDrafterPlan
	AttachedDrafterAttachment           = root.AttachedDrafterAttachment
	AttachedDrafterPairConfig           = root.AttachedDrafterPairConfig
	AttachedDrafterPair                 = root.AttachedDrafterPair
	PromptLookupDecodeConfig            = root.PromptLookupDecodeConfig
)

func SpeculativeDecode(ctx context.Context, target, draft inference.TextModel, cfg SpeculativeDecodeConfig) (inferdecode.Result, error) {
	return root.SpeculativeDecode(ctx, target, draft, cfg)
}

func AttachedDrafterDecode(ctx context.Context, target, draft inference.TextModel, cfg AttachedDrafterDecodeConfig) (inferdecode.Result, error) {
	return root.AttachedDrafterDecode(ctx, target, draft, cfg)
}

func PlanAttachedDrafter(target, draft inference.TextModel) (AttachedDrafterPlan, error) {
	return root.PlanAttachedDrafter(target, draft)
}

func AttachNativeDrafter(target, draft inference.TextModel) (AttachedDrafterAttachment, error) {
	return root.AttachNativeDrafter(target, draft)
}

func NewAttachedDrafterPair(target, draft inference.TextModel) (*AttachedDrafterPair, error) {
	return root.NewAttachedDrafterPair(target, draft)
}

func LoadAttachedDrafterPair(targetPath, draftPath string, cfg AttachedDrafterPairConfig) (*AttachedDrafterPair, error) {
	return root.LoadAttachedDrafterPair(targetPath, draftPath, cfg)
}

func PromptLookupDecode(ctx context.Context, target inference.TextModel, cfg PromptLookupDecodeConfig) (inferdecode.Result, error) {
	return root.PromptLookupDecode(ctx, target, cfg)
}
