// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import root "dappco.re/go/rocm"

const (
	ProductionFastLaneName                          = root.ProductionFastLaneName
	ProductionMTPDefaultDraftTokens                 = root.ProductionMTPDefaultDraftTokens
	ProductionMTPAssistantTokenOrderingVocabSize    = root.ProductionMTPAssistantTokenOrderingVocabSize
	ProductionMTPAssistantOrderedEmbeddingCentroids = root.ProductionMTPAssistantOrderedEmbeddingCentroids
	ProductionMTPAssistantCentroidIntermediateTopK  = root.ProductionMTPAssistantCentroidIntermediateTopK
	OfficialGemma4E2BRoleTarget                     = root.OfficialGemma4E2BRoleTarget
	OfficialGemma4E2BRoleAssistant                  = root.OfficialGemma4E2BRoleAssistant
)

type (
	OfficialGemma4E2BLock                    = root.OfficialGemma4E2BLock
	ProductionFastLane                       = root.ProductionFastLane
	ProductionMTPPolicy                      = root.ProductionMTPPolicy
	ProductionTurboQuantPolicy               = root.ProductionTurboQuantPolicy
	ProductionCombinedMTPAndTurboQuantPolicy = root.ProductionCombinedMTPAndTurboQuantPolicy
)

func DefaultOfficialGemma4E2BLocks() []OfficialGemma4E2BLock {
	return root.DefaultOfficialGemma4E2BLocks()
}

func OfficialGemma4E2BLockByRole(role string) (OfficialGemma4E2BLock, bool) {
	for _, lock := range DefaultOfficialGemma4E2BLocks() {
		if lock.Role == role {
			return lock, true
		}
	}
	return OfficialGemma4E2BLock{}, false
}

func OfficialGemma4E2BTargetLock() OfficialGemma4E2BLock {
	lock, _ := OfficialGemma4E2BLockByRole(OfficialGemma4E2BRoleTarget)
	return lock
}

func OfficialGemma4E2BAssistantLock() OfficialGemma4E2BLock {
	lock, _ := OfficialGemma4E2BLockByRole(OfficialGemma4E2BRoleAssistant)
	return lock
}

func DefaultProductionFastLane() ProductionFastLane {
	return root.DefaultProductionFastLane()
}

func DefaultProductionMTPPolicy() ProductionMTPPolicy {
	return root.DefaultProductionMTPPolicy()
}

func DefaultProductionTurboQuantPolicy() ProductionTurboQuantPolicy {
	return root.DefaultProductionTurboQuantPolicy()
}

func DefaultProductionCombinedMTPAndTurboQuantPolicy() ProductionCombinedMTPAndTurboQuantPolicy {
	return root.DefaultProductionCombinedMTPAndTurboQuantPolicy()
}
