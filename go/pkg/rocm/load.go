// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"context"

	"dappco.re/go/inference"
	root "dappco.re/go/rocm"
)

type ROCmLoadConfig = root.ROCmLoadConfig

func ROCmAvailable() bool {
	return root.ROCmAvailable()
}

func LoadModelWithConfig(path string, cfg ROCmLoadConfig, opts ...inference.LoadOption) (inference.TextModel, error) {
	return root.LoadModelWithConfig(path, cfg, opts...)
}

func InspectModelPack(ctx context.Context, path string) (*inference.ModelPackInspection, error) {
	return root.InspectModelPack(ctx, path)
}
