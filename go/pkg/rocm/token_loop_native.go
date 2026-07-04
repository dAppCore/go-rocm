// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"dappco.re/go/inference"
	root "dappco.re/go/rocm"
)

type (
	StateSession          = root.StateSession
	ROCmRuntimeTokenModel = root.ROCmRuntimeTokenModel
)

func ROCmRuntimeTokenSession(model inference.TextModel) (*StateSession, bool) {
	return root.ROCmRuntimeTokenSession(model)
}
