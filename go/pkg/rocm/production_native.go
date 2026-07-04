// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import root "dappco.re/go/rocm"

const (
	ProductionLaneName                    = root.ProductionLaneName
	ProductionLaneModelID                 = root.ProductionLaneModelID
	ProductionLaneCurrentModelID          = root.ProductionLaneCurrentModelID
	ProductionLaneArchitecture            = root.ProductionLaneArchitecture
	ProductionLaneChatTemplate            = root.ProductionLaneChatTemplate
	ProductionLaneProductDefaultQuantBits = root.ProductionLaneProductDefaultQuantBits
	ProductionLaneLongContextLength       = root.ProductionLaneLongContextLength
	ProductionLaneHyperLongContextLength  = root.ProductionLaneHyperLongContextLength
)

type (
	ProductionLane                             = root.ProductionLane
	ProductionQuantizationPolicy               = root.ProductionQuantizationPolicy
	ProductionQuantizationTier                 = root.ProductionQuantizationTier
	ProductionQuantizationSelectionInput       = root.ProductionQuantizationSelectionInput
	ProductionQuantizationChoice               = root.ProductionQuantizationChoice
	ProductionAutoRoundQuantizationSupport     = root.ProductionAutoRoundQuantizationSupport
	ProductionAutoRoundQuantizationProfile     = root.ProductionAutoRoundQuantizationProfile
	ProductionBookGatePolicy                   = root.ProductionBookGatePolicy
	ProductionArchitectureStatusReport         = root.ProductionArchitectureStatusReport
	ProductionArchitectureGap                  = root.ProductionArchitectureGap
	ProductionQuantizationPackLock             = root.ProductionQuantizationPackLock
	ProductionMTPPromotionEvidence             = root.ProductionMTPPromotionEvidence
	ProductionMTPDecodeRunEvidence             = root.ProductionMTPDecodeRunEvidence
	ProductionMTPPromotionDecision             = root.ProductionMTPPromotionDecision
	ProductionTurboQuantPromotionEvidence      = root.ProductionTurboQuantPromotionEvidence
	ProductionTurboQuantPromotionDecision      = root.ProductionTurboQuantPromotionDecision
	ProductionCombinedMTPAndTurboQuantDecision = root.ProductionCombinedMTPAndTurboQuantDecision
)

func DefaultProductionLane() ProductionLane {
	return root.DefaultProductionLane()
}

func DefaultProductionQuantizationPolicy() ProductionQuantizationPolicy {
	return root.DefaultProductionQuantizationPolicy()
}

func SelectProductionQuantizationTier(input ProductionQuantizationSelectionInput) ProductionQuantizationChoice {
	return root.SelectProductionQuantizationTier(input)
}

func DefaultProductionAutoRoundQuantizationSupport() ProductionAutoRoundQuantizationSupport {
	return root.DefaultProductionAutoRoundQuantizationSupport()
}

func DefaultProductionAutoRoundQuantizationProfiles() []ProductionAutoRoundQuantizationProfile {
	return root.DefaultProductionAutoRoundQuantizationProfiles()
}

func DefaultProductionBookGatePolicy() ProductionBookGatePolicy {
	return root.DefaultProductionBookGatePolicy()
}

func DefaultProductionArchitectureStatus() ProductionArchitectureStatusReport {
	return root.DefaultProductionArchitectureStatus()
}

func DefaultProductionQuantizationPackLocks() []ProductionQuantizationPackLock {
	return root.DefaultProductionQuantizationPackLocks()
}

func EvaluateProductionMTPPromotion(policy ProductionMTPPolicy, evidence ProductionMTPPromotionEvidence) ProductionMTPPromotionDecision {
	return root.EvaluateProductionMTPPromotion(policy, evidence)
}

func EvaluateProductionMTPPromotionMetricLabels(labels map[string]string) (ProductionMTPPromotionDecision, error) {
	return root.EvaluateProductionMTPPromotionMetricLabels(labels)
}

func EvaluateProductionMTPPromotionMetricLabelsWithPolicy(policy ProductionMTPPolicy, labels map[string]string) (ProductionMTPPromotionDecision, error) {
	return root.EvaluateProductionMTPPromotionMetricLabelsWithPolicy(policy, labels)
}

func EvaluateProductionTurboQuantPromotion(policy ProductionTurboQuantPolicy, evidence ProductionTurboQuantPromotionEvidence) ProductionTurboQuantPromotionDecision {
	return root.EvaluateProductionTurboQuantPromotion(policy, evidence)
}

func EvaluateProductionTurboQuantPromotionMetricLabels(labels map[string]string) (ProductionTurboQuantPromotionDecision, error) {
	return root.EvaluateProductionTurboQuantPromotionMetricLabels(labels)
}

func EvaluateProductionTurboQuantPromotionMetricLabelsWithPolicy(policy ProductionTurboQuantPolicy, labels map[string]string) (ProductionTurboQuantPromotionDecision, error) {
	return root.EvaluateProductionTurboQuantPromotionMetricLabelsWithPolicy(policy, labels)
}

func EvaluateProductionCombinedMTPAndTurboQuantPromotion(policy ProductionCombinedMTPAndTurboQuantPolicy, mtpEvidence ProductionMTPPromotionEvidence, turboEvidence ProductionTurboQuantPromotionEvidence) ProductionCombinedMTPAndTurboQuantDecision {
	return root.EvaluateProductionCombinedMTPAndTurboQuantPromotion(policy, mtpEvidence, turboEvidence)
}

func EvaluateProductionCombinedMTPAndTurboQuantPromotionMetricLabels(labels map[string]string) (ProductionCombinedMTPAndTurboQuantDecision, error) {
	return root.EvaluateProductionCombinedMTPAndTurboQuantPromotionMetricLabels(labels)
}

func EvaluateProductionCombinedMTPAndTurboQuantPromotionMetricLabelsWithPolicy(policy ProductionCombinedMTPAndTurboQuantPolicy, labels map[string]string) (ProductionCombinedMTPAndTurboQuantDecision, error) {
	return root.EvaluateProductionCombinedMTPAndTurboQuantPromotionMetricLabelsWithPolicy(policy, labels)
}
