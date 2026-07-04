// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"dappco.re/go/inference"
	root "dappco.re/go/rocm"
)

func LoadAttachedDrafterPairAsTextModel(targetPath, draftPath string, opts ...inference.LoadOption) (inference.TextModel, error) {
	return root.LoadAttachedDrafterPairAsTextModel(targetPath, draftPath, opts...)
}

func LoadAttachedDrafterPairAsTextModelWithConfig(targetPath, draftPath string, cfg ROCmLoadConfig, opts ...inference.LoadOption) (inference.TextModel, error) {
	return root.LoadAttachedDrafterPairAsTextModelWithConfig(targetPath, draftPath, cfg, opts...)
}

func LoadAttachedDrafterPairAsTextModelBlock(targetPath, draftPath string, draftBlock int, opts ...inference.LoadOption) (inference.TextModel, error) {
	return root.LoadAttachedDrafterPairAsTextModelBlock(targetPath, draftPath, draftBlock, opts...)
}

func LoadAttachedDrafterPairAsTextModelBlockWithConfig(targetPath, draftPath string, draftBlock int, cfg ROCmLoadConfig, opts ...inference.LoadOption) (inference.TextModel, error) {
	return root.LoadAttachedDrafterPairAsTextModelBlockWithConfig(targetPath, draftPath, draftBlock, cfg, opts...)
}

func IsAttachedDrafterTextModel(model inference.TextModel) bool {
	return root.IsAttachedDrafterTextModel(model)
}
