// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import root "dappco.re/go/rocm"

const (
	DraftSourceNone             = root.DraftSourceNone
	DraftSourceFlag             = root.DraftSourceFlag
	DraftSourceAssistantDir     = root.DraftSourceAssistantDir
	DraftSourceSiblingAssistant = root.DraftSourceSiblingAssistant
	DraftSourceMTPDir           = root.DraftSourceMTPDir
	DraftSourceMTPSibling       = root.DraftSourceMTPSibling
)

type (
	DraftDetectOptions   = root.DraftDetectOptions
	DraftDetectionSource = root.DraftDetectionSource
	DraftDetection       = root.DraftDetection
)

func DetectGemma4DraftPath(modelPath, explicit string, opts DraftDetectOptions) DraftDetection {
	return root.DetectGemma4DraftPath(modelPath, explicit, opts)
}
