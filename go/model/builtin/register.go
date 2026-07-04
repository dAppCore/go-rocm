// SPDX-Licence-Identifier: EUPL-1.2

// Package builtin registers the model-profile factories the root ROCm package
// enables by default.
package builtin

import (
	"dappco.re/go/rocm/model"
	"dappco.re/go/rocm/model/architecture"
	_ "dappco.re/go/rocm/model/gemma4" // registers Gemma-4 before the generic fallback
)

func init() {
	model.RegisterProfileFactory(architecture.ProfileFactory{})
}
