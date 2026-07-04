// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"math"
	"testing"

	core "dappco.re/go"
)

func TestProductionRequiredMetricLabels_Good_MTPPassingLabelsComplete(t *testing.T) {
	labels := productionMTPPassingLabels()

	err := ValidateProductionMTPRequiredMetricLabels(labels)

	core.RequireNoError(t, err)
}

func TestProductionRequiredMetricLabels_Bad_MTPRejectsMissingRequiredMetric(t *testing.T) {
	labels := productionMTPPassingLabels()
	delete(labels, "mtp_visible_tokens_per_sec")

	err := ValidateProductionMTPRequiredMetricLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "mtp_visible_tokens_per_sec")
}

func TestProductionRequiredMetricLabels_Good_TurboQuantPassingLabelsComplete(t *testing.T) {
	labels := productionTurboQuantPassingLabels()

	err := ValidateProductionTurboQuantRequiredMetricLabels(labels)

	core.RequireNoError(t, err)
}

func TestProductionRequiredMetricLabels_Bad_TurboQuantRejectsMissingRequiredMetric(t *testing.T) {
	labels := productionTurboQuantPassingLabels()
	delete(labels, "candidate_active_plus_cache_memory_bytes")

	err := ValidateProductionTurboQuantRequiredMetricLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "candidate_active_plus_cache_memory_bytes")
}

func TestProductionRequiredMetricLabels_Good_CombinedPassingLabelsComplete(t *testing.T) {
	labels := productionCombinedPassingLabels()

	err := ValidateProductionCombinedMTPAndTurboQuantRequiredMetricLabels(labels)

	core.RequireNoError(t, err)
}

func TestProductionRequiredMetricLabels_Good_CombinedPrefixedLabelsComplete(t *testing.T) {
	labels := productionCombinedPrefixedMTPPassingLabels()

	err := ValidateProductionCombinedMTPAndTurboQuantRequiredMetricLabels(labels)

	core.RequireNoError(t, err)
}

func TestProductionRequiredMetricLabels_Bad_CombinedRejectsMissingRequiredMetric(t *testing.T) {
	labels := productionCombinedPassingLabels()
	delete(labels, "mtp_target_tokens_per_sec")

	err := ValidateProductionCombinedMTPAndTurboQuantRequiredMetricLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "mtp_target_tokens_per_sec")
}

func TestProductionBookGateMetricLabels_Good_CompleteNumericRow(t *testing.T) {
	labels := productionBookGatePassingLabels()

	err := ValidateProductionBookGateMetricLabels(labels)

	core.RequireNoError(t, err)
}

func TestProductionBookGateMetricLabels_Bad_RejectsMissingMetric(t *testing.T) {
	labels := productionBookGatePassingLabels()
	delete(labels, "production_book_gate_decode")

	err := ValidateProductionBookGateMetricLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "production_book_gate_decode")
}

func TestProductionBookGateMetricLabels_Bad_RejectsMalformedMetric(t *testing.T) {
	labels := productionBookGatePassingLabels()
	labels["production_book_gate_reason_code"] = "passed"

	err := ValidateProductionBookGateMetricLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "production_book_gate_reason_code")
}

func TestProductionBookGateMetricLabels_Bad_RejectsNonFiniteMetric(t *testing.T) {
	labels := productionBookGatePassingLabels()
	labels["production_book_gate_raw_decode_tok/s"] = "NaN"

	err := ValidateProductionBookGateMetricLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "production_book_gate_raw_decode_tok/s")
}

func TestProductionBookGateMetricLabels_Good_EvaluatesPassingRow(t *testing.T) {
	labels := productionBookGatePassingLabels()

	decision, err := EvaluateProductionBookGateMetricLabels(labels)

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, decision.ProductionCandidate)
	core.AssertEqual(t, ProductionBookGateReasonPass, decision.ReasonCode)
	core.AssertEqual(t, true, decision.QuantAccepted)
	core.AssertEqual(t, true, decision.TurnsAccepted)
	core.AssertEqual(t, true, decision.WallAccepted)
	core.AssertEqual(t, true, decision.DecodeAccepted)
	core.AssertEqual(t, true, decision.QualityAccepted)
	core.AssertEqual(t, float64(101.5), decision.RawDecodeTokensPerSec)
	core.AssertEqual(t, float64(89.25), decision.WallSeconds)
	core.AssertEqual(t, 0, decision.QualityFlags)
	core.AssertContains(t, decision.Reason, "passes q6 retained-state")
}

func TestProductionBookGateMetricLabels_Good_EvaluatesDecodeFailureRow(t *testing.T) {
	labels := productionBookGatePassingLabels()
	labels["production_book_gate_candidate"] = "0"
	labels["production_book_gate_reason_code"] = "5"
	labels["production_book_gate_decode"] = "0"
	labels["production_book_gate_raw_decode_tok/s"] = "87.75"

	decision, err := EvaluateProductionBookGateMetricLabels(labels)

	core.RequireNoError(t, err)
	core.AssertEqual(t, false, decision.ProductionCandidate)
	core.AssertEqual(t, ProductionBookGateReasonDecode, decision.ReasonCode)
	core.AssertEqual(t, false, decision.DecodeAccepted)
	core.AssertContains(t, decision.Reason, "87.750 tok/s below 100 tok/s")
}

func TestProductionBookGateMetricLabels_Bad_RejectsInconsistentCandidate(t *testing.T) {
	labels := productionBookGatePassingLabels()
	labels["production_book_gate_candidate"] = "1"
	labels["production_book_gate_reason_code"] = "5"
	labels["production_book_gate_decode"] = "0"

	_, err := EvaluateProductionBookGateMetricLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "inconsistent")
}

func TestProductionBookGateMetricLabels_Bad_RejectsPassReasonWithFailedGate(t *testing.T) {
	labels := productionBookGatePassingLabels()
	labels["production_book_gate_candidate"] = "0"
	labels["production_book_gate_decode"] = "0"
	labels["production_book_gate_raw_decode_tok/s"] = "99"

	_, err := EvaluateProductionBookGateMetricLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "first failing gate")
}

func TestProductionBookGateMetricLabels_Bad_RejectsNonBinaryGateMetric(t *testing.T) {
	labels := productionBookGatePassingLabels()
	labels["production_book_gate_wall"] = "0.5"

	_, err := EvaluateProductionBookGateMetricLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "production_book_gate_wall")
}

func TestProductionBookGateMetricLabels_Good_AddsLabelsFromMetrics(t *testing.T) {
	labels, err := AddProductionBookGateMetricLabels(map[string]string{"source": "bench"}, productionBookGatePassingMetrics())

	core.RequireNoError(t, err)
	core.AssertEqual(t, "bench", labels["source"])
	core.AssertEqual(t, "101.5", labels["production_book_gate_raw_decode_tok/s"])
	decision, err := EvaluateProductionBookGateMetricLabels(labels)
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, decision.ProductionCandidate)
	core.AssertEqual(t, ProductionBookGateReasonPass, decision.ReasonCode)
}

func TestProductionBookGateMetrics_Good_EvaluatesPassingMetrics(t *testing.T) {
	decision, err := EvaluateProductionBookGateMetrics(productionBookGatePassingMetrics())

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, decision.ProductionCandidate)
	core.AssertEqual(t, ProductionBookGateReasonPass, decision.ReasonCode)
	core.AssertEqual(t, float64(101.5), decision.RawDecodeTokensPerSec)
	core.AssertEqual(t, float64(89.25), decision.WallSeconds)
	core.AssertEqual(t, 0, decision.QualityFlags)
}

func TestProductionBookRetainedRouteMetrics_Good_RetainedRoute(t *testing.T) {
	err := ValidateProductionBookRetainedRouteMetrics(productionBookRetainedRoutePassingMetrics())

	core.RequireNoError(t, err)
}

func TestProductionBookRetainedRouteMetrics_Bad_RejectsReplayBaseline(t *testing.T) {
	metrics := productionBookRetainedRoutePassingMetrics()
	metrics["book_replay_baseline"] = 1

	err := ValidateProductionBookRetainedRouteMetrics(metrics)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "book_replay_baseline")
}

func TestProductionBookRetainedRouteMetrics_Bad_RejectsMissingNoReplayMetric(t *testing.T) {
	metrics := productionBookRetainedRoutePassingMetrics()
	delete(metrics, "book_prompt_replay_fallback_forbidden")

	err := ValidateProductionBookRetainedRouteMetrics(metrics)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "book_prompt_replay_fallback_forbidden")
}

func TestProductionBookRetainedRouteMetrics_Bad_RejectsFalseRuntimeKVSource(t *testing.T) {
	metrics := productionBookRetainedRoutePassingMetrics()
	metrics["book_state_source_runtime_kv"] = 0

	err := ValidateProductionBookRetainedRouteMetrics(metrics)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "book_state_source_runtime_kv")
}

func TestProductionBookRetainedArtifactMetrics_Good_EvaluatesPassingArtifact(t *testing.T) {
	decision, err := EvaluateProductionBookRetainedArtifactMetrics(productionBookRetainedArtifactPassingMetrics())

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, decision.RetainedRoute)
	core.AssertEqual(t, true, decision.Gate.ProductionCandidate)
	core.AssertEqual(t, ProductionBookGateReasonPass, decision.Gate.ReasonCode)
}

func TestProductionBookRetainedArtifactMetrics_Good_EvaluatesDecodeFailureArtifact(t *testing.T) {
	metrics := productionBookRetainedArtifactPassingMetrics()
	metrics["production_book_gate_candidate"] = 0
	metrics["production_book_gate_reason_code"] = float64(ProductionBookGateReasonDecode)
	metrics["production_book_gate_decode"] = 0
	metrics["production_book_gate_raw_decode_tok/s"] = 91.25

	decision, err := EvaluateProductionBookRetainedArtifactMetrics(metrics)

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, decision.RetainedRoute)
	core.AssertEqual(t, false, decision.Gate.ProductionCandidate)
	core.AssertEqual(t, ProductionBookGateReasonDecode, decision.Gate.ReasonCode)
}

func TestProductionBookRetainedArtifactMetrics_Bad_RejectsReplayRoute(t *testing.T) {
	metrics := productionBookRetainedArtifactPassingMetrics()
	metrics["book_replay_baseline"] = 1

	_, err := EvaluateProductionBookRetainedArtifactMetrics(metrics)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "book_replay_baseline")
}

func TestProductionBookRetainedArtifactDecisionLabels_Good_PassingArtifact(t *testing.T) {
	decision, err := EvaluateProductionBookRetainedArtifactMetrics(productionBookRetainedArtifactPassingMetrics())
	core.RequireNoError(t, err)

	labels := AddProductionBookRetainedArtifactDecisionLabels(map[string]string{"source": "bench"}, decision)

	core.AssertEqual(t, "bench", labels["source"])
	core.AssertEqual(t, "true", labels["production_book_retained_artifact_candidate"])
	core.AssertEqual(t, "true", labels["production_book_retained_artifact_retained_route"])
	core.AssertContains(t, labels["production_book_retained_artifact_reason"], "passes q6 retained-state")
	core.AssertEqual(t, "true", labels["production_book_retained_artifact_gate_candidate"])
	core.AssertEqual(t, "0", labels["production_book_retained_artifact_gate_reason_code"])
	core.AssertEqual(t, "true", labels["production_book_retained_artifact_gate_q6"])
	core.AssertEqual(t, "true", labels["production_book_retained_artifact_gate_turns"])
	core.AssertEqual(t, "true", labels["production_book_retained_artifact_gate_wall"])
	core.AssertEqual(t, "true", labels["production_book_retained_artifact_gate_decode"])
	core.AssertEqual(t, "true", labels["production_book_retained_artifact_gate_quality"])
	core.AssertEqual(t, "101.500000", labels["production_book_retained_artifact_raw_decode_tok/s"])
	core.AssertEqual(t, "89.250000", labels["production_book_retained_artifact_wall_s"])
	core.AssertEqual(t, "0", labels["production_book_retained_artifact_quality_flags"])
}

func TestProductionBookRetainedArtifactDecisionLabels_Good_DecodeFailureArtifact(t *testing.T) {
	metrics := productionBookRetainedArtifactPassingMetrics()
	metrics["production_book_gate_candidate"] = 0
	metrics["production_book_gate_reason_code"] = float64(ProductionBookGateReasonDecode)
	metrics["production_book_gate_decode"] = 0
	metrics["production_book_gate_raw_decode_tok/s"] = 91.25
	decision, err := EvaluateProductionBookRetainedArtifactMetrics(metrics)
	core.RequireNoError(t, err)

	labels := ProductionBookRetainedArtifactDecisionLabels(decision)

	core.AssertEqual(t, "false", labels["production_book_retained_artifact_candidate"])
	core.AssertEqual(t, "true", labels["production_book_retained_artifact_retained_route"])
	core.AssertContains(t, labels["production_book_retained_artifact_reason"], "91.250 tok/s below 100 tok/s")
	core.AssertEqual(t, "false", labels["production_book_retained_artifact_gate_candidate"])
	core.AssertEqual(t, "5", labels["production_book_retained_artifact_gate_reason_code"])
	core.AssertEqual(t, "false", labels["production_book_retained_artifact_gate_decode"])
	core.AssertEqual(t, "91.250000", labels["production_book_retained_artifact_raw_decode_tok/s"])
}

func TestProductionBookRetainedArtifactDecisionLabels_Good_EvaluatesPassingLabels(t *testing.T) {
	source, err := EvaluateProductionBookRetainedArtifactMetrics(productionBookRetainedArtifactPassingMetrics())
	core.RequireNoError(t, err)
	labels := ProductionBookRetainedArtifactDecisionLabels(source)

	decision, err := EvaluateProductionBookRetainedArtifactDecisionLabels(labels)

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, decision.RetainedRoute)
	core.AssertEqual(t, true, decision.Gate.ProductionCandidate)
	core.AssertEqual(t, ProductionBookGateReasonPass, decision.Gate.ReasonCode)
	core.AssertEqual(t, float64(101.5), decision.Gate.RawDecodeTokensPerSec)
	core.AssertContains(t, decision.Gate.Reason, "passes q6 retained-state")
}

func TestProductionBookRetainedArtifactDecisionLabels_Good_AddsFromMetrics(t *testing.T) {
	labels, err := AddProductionBookRetainedArtifactMetricDecisionLabels(map[string]string{"source": "bench"}, productionBookRetainedArtifactPassingMetrics())

	core.RequireNoError(t, err)
	core.AssertEqual(t, "bench", labels["source"])
	core.AssertEqual(t, "true", labels["production_book_retained_artifact_candidate"])
	core.AssertEqual(t, "true", labels["production_book_retained_artifact_retained_route"])
	core.AssertEqual(t, "true", labels["production_book_retained_artifact_gate_decode"])
	decision, err := EvaluateProductionBookRetainedArtifactDecisionLabels(labels)
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, decision.Gate.ProductionCandidate)
}

func TestProductionBookRetainedArtifactDecisionLabels_Bad_AddFromMetricsRejectsReplayRoute(t *testing.T) {
	metrics := productionBookRetainedArtifactPassingMetrics()
	metrics["book_replay_baseline"] = 1

	_, err := ProductionBookRetainedArtifactMetricDecisionLabels(metrics)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "book_replay_baseline")
}

func TestProductionBookRetainedArtifactDecisionLabels_Bad_RejectsMissingLabel(t *testing.T) {
	decision, err := EvaluateProductionBookRetainedArtifactMetrics(productionBookRetainedArtifactPassingMetrics())
	core.RequireNoError(t, err)
	labels := ProductionBookRetainedArtifactDecisionLabels(decision)
	delete(labels, "production_book_retained_artifact_gate_decode")

	err = ValidateProductionBookRetainedArtifactDecisionLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "production_book_retained_artifact_gate_decode")
}

func TestProductionBookRetainedArtifactDecisionLabels_Bad_RejectsReplayRoute(t *testing.T) {
	decision, err := EvaluateProductionBookRetainedArtifactMetrics(productionBookRetainedArtifactPassingMetrics())
	core.RequireNoError(t, err)
	labels := ProductionBookRetainedArtifactDecisionLabels(decision)
	labels["production_book_retained_artifact_retained_route"] = "false"
	labels["production_book_retained_artifact_candidate"] = "false"

	_, err = EvaluateProductionBookRetainedArtifactDecisionLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "retained_route")
}

func TestProductionBookRetainedArtifactDecisionLabels_Bad_RejectsInconsistentCandidate(t *testing.T) {
	decision, err := EvaluateProductionBookRetainedArtifactMetrics(productionBookRetainedArtifactPassingMetrics())
	core.RequireNoError(t, err)
	labels := ProductionBookRetainedArtifactDecisionLabels(decision)
	labels["production_book_retained_artifact_candidate"] = "false"

	_, err = EvaluateProductionBookRetainedArtifactDecisionLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "candidate")
}

func TestProductionBookRetainedArtifactDecisionLabels_Bad_RejectsInconsistentReason(t *testing.T) {
	decision, err := EvaluateProductionBookRetainedArtifactMetrics(productionBookRetainedArtifactPassingMetrics())
	core.RequireNoError(t, err)
	labels := ProductionBookRetainedArtifactDecisionLabels(decision)
	labels["production_book_retained_artifact_reason"] = "manual override"

	_, err = EvaluateProductionBookRetainedArtifactDecisionLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "reason")
}

func TestProductionBookRetainedArtifactDecisionLabels_Bad_RejectsNonFiniteRawDecode(t *testing.T) {
	decision, err := EvaluateProductionBookRetainedArtifactMetrics(productionBookRetainedArtifactPassingMetrics())
	core.RequireNoError(t, err)
	labels := ProductionBookRetainedArtifactDecisionLabels(decision)
	labels["production_book_retained_artifact_raw_decode_tok/s"] = "NaN"

	_, err = EvaluateProductionBookRetainedArtifactDecisionLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "production_book_retained_artifact_raw_decode_tok/s")
}

func TestProductionBookGateMetrics_Good_EvaluatesDecodeFailureMetrics(t *testing.T) {
	metrics := productionBookGatePassingMetrics()
	metrics["production_book_gate_candidate"] = 0
	metrics["production_book_gate_reason_code"] = float64(ProductionBookGateReasonDecode)
	metrics["production_book_gate_decode"] = 0
	metrics["production_book_gate_raw_decode_tok/s"] = 91.25

	decision, err := EvaluateProductionBookGateMetrics(metrics)

	core.RequireNoError(t, err)
	core.AssertEqual(t, false, decision.ProductionCandidate)
	core.AssertEqual(t, ProductionBookGateReasonDecode, decision.ReasonCode)
	core.AssertEqual(t, false, decision.DecodeAccepted)
	core.AssertContains(t, decision.Reason, "91.250 tok/s below 100 tok/s")
}

func TestProductionBookGateMetricLabels_Bad_AddRejectsMissingMetric(t *testing.T) {
	metrics := productionBookGatePassingMetrics()
	delete(metrics, "production_book_gate_quality_flags")

	_, err := AddProductionBookGateMetricLabels(nil, metrics)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "production_book_gate_quality_flags")
}

func TestProductionBookGateMetrics_Bad_RejectsMissingMetric(t *testing.T) {
	metrics := productionBookGatePassingMetrics()
	delete(metrics, "production_book_gate_quality_flags")

	err := ValidateProductionBookGateMetrics(metrics)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "production_book_gate_quality_flags")
}

func TestProductionBookGateMetrics_Bad_RejectsNonFiniteMetric(t *testing.T) {
	metrics := productionBookGatePassingMetrics()
	metrics["production_book_gate_wall_s"] = math.Inf(1)

	err := ValidateProductionBookGateMetrics(metrics)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "production_book_gate_wall_s")
}

func TestProductionBookGateMetrics_Bad_RejectsInconsistentReason(t *testing.T) {
	metrics := productionBookGatePassingMetrics()
	metrics["production_book_gate_candidate"] = 0
	metrics["production_book_gate_reason_code"] = float64(ProductionBookGateReasonPass)
	metrics["production_book_gate_decode"] = 0
	metrics["production_book_gate_raw_decode_tok/s"] = 99

	_, err := EvaluateProductionBookGateMetrics(metrics)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "first failing gate")
}

func TestProductionBookGateMetrics_Bad_RejectsReasonThatSkipsEarlierFailingGate(t *testing.T) {
	metrics := productionBookGatePassingMetrics()
	metrics["production_book_gate_candidate"] = 0
	metrics["production_book_gate_q6"] = 0
	metrics["production_book_gate_reason_code"] = float64(ProductionBookGateReasonDecode)
	metrics["production_book_gate_decode"] = 0
	metrics["production_book_gate_raw_decode_tok/s"] = 91.25

	_, err := EvaluateProductionBookGateMetrics(metrics)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "first failing gate")
}

func TestProductionBookGateMetrics_Bad_RejectsWallAndDecodeFailureReportedAsDecode(t *testing.T) {
	metrics := productionBookGatePassingMetrics()
	metrics["production_book_gate_candidate"] = 0
	metrics["production_book_gate_reason_code"] = float64(ProductionBookGateReasonDecode)
	metrics["production_book_gate_wall"] = 0
	metrics["production_book_gate_wall_s"] = float64(ProductionLaneBookWallSeconds + 1)
	metrics["production_book_gate_decode"] = 0
	metrics["production_book_gate_raw_decode_tok/s"] = 91.25

	_, err := EvaluateProductionBookGateMetrics(metrics)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "first failing gate")
}

func TestProductionBookGateMetrics_Bad_RejectsDecodeBooleanContradictingRawDecode(t *testing.T) {
	metrics := productionBookGatePassingMetrics()
	metrics["production_book_gate_raw_decode_tok/s"] = 99

	_, err := EvaluateProductionBookGateMetrics(metrics)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "production_book_gate_decode")
}

func TestProductionBookGateMetrics_Bad_RejectsWallBooleanContradictingWallSeconds(t *testing.T) {
	metrics := productionBookGatePassingMetrics()
	metrics["production_book_gate_wall_s"] = float64(ProductionLaneBookWallSeconds + 1)

	_, err := EvaluateProductionBookGateMetrics(metrics)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "production_book_gate_wall")
}

func TestProductionBookGateMetrics_Bad_RejectsQualityBooleanContradictingQualityFlags(t *testing.T) {
	metrics := productionBookGatePassingMetrics()
	metrics["production_book_gate_quality_flags"] = 1

	_, err := EvaluateProductionBookGateMetrics(metrics)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "production_book_gate_quality")
}

func TestProductionBookGateMetricLabels_Bad_RejectsDecodeBooleanContradictingRawDecode(t *testing.T) {
	labels := productionBookGatePassingLabels()
	labels["production_book_gate_raw_decode_tok/s"] = "99"

	_, err := EvaluateProductionBookGateMetricLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "production_book_gate_decode")
}

func TestProductionBookGateMetrics_Good_CustomPolicy(t *testing.T) {
	policy := DefaultProductionBookGatePolicy()
	policy.MinimumRawDecodeTokensSec = 90
	metrics := productionBookGatePassingMetrics()
	metrics["production_book_gate_raw_decode_tok/s"] = 95

	decision, err := EvaluateProductionBookGateMetricsWithPolicy(policy, metrics)

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, decision.ProductionCandidate)
	core.AssertEqual(t, float64(95), decision.RawDecodeTokensPerSec)
}

func TestProductionPromotionDecisionLabels_Good_MTP(t *testing.T) {
	decision := EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), productionMTPPassingEvidence())

	labels := AddProductionMTPPromotionDecisionLabels(map[string]string{"source": "bench"}, decision)

	core.AssertEqual(t, "bench", labels["source"])
	core.AssertEqual(t, "true", labels["production_mtp_enable_by_default"])
	core.AssertContains(t, labels["production_mtp_reason"], "MTP retained workflow")
	core.AssertEqual(t, "1.250000", labels["production_mtp_wall_speedup"])
	core.AssertEqual(t, "1.190476", labels["production_mtp_visible_speedup"])
	core.AssertEqual(t, "1.250000", labels["production_mtp_restore_speedup"])
	core.AssertEqual(t, "0.240000", labels["production_mtp_energy_savings"])
	core.AssertEqual(t, "0.750000", labels["production_mtp_acceptance_rate"])

	parsed, err := EvaluateProductionMTPPromotionDecisionLabels(labels)

	core.RequireNoError(t, err)
	core.AssertEqual(t, decision.EnableByDefault, parsed.EnableByDefault)
	core.AssertEqual(t, decision.Reason, parsed.Reason)
	core.AssertEqual(t, labels["production_mtp_wall_speedup"], productionMetricFloatLabel(parsed.WallSpeedup))
	core.AssertEqual(t, labels["production_mtp_visible_speedup"], productionMetricFloatLabel(parsed.VisibleSpeedup))
	core.AssertEqual(t, labels["production_mtp_restore_speedup"], productionMetricFloatLabel(parsed.RestoreSpeedup))
	core.AssertEqual(t, labels["production_mtp_energy_savings"], productionMetricFloatLabel(parsed.EnergySavings))
	core.AssertEqual(t, labels["production_mtp_acceptance_rate"], productionMetricFloatLabel(parsed.AcceptanceRate))
}

func TestProductionPromotionDecisionLabels_Good_TurboQuant(t *testing.T) {
	decision := EvaluateProductionTurboQuantPromotion(DefaultProductionTurboQuantPolicy(), productionTurboQuantPassingEvidence())

	labels := ProductionTurboQuantPromotionDecisionLabels(decision)

	core.AssertEqual(t, "true", labels["production_turboquant_candidate"])
	core.AssertEqual(t, "true", labels["production_turboquant_enable_by_default"])
	core.AssertContains(t, labels["production_turboquant_reason"], "TurboQuant retained workflow")
	core.AssertEqual(t, "1.111111", labels["production_turboquant_wall_speedup"])
	core.AssertEqual(t, "1.020000", labels["production_turboquant_visible_speedup"])
	core.AssertEqual(t, "1.428571", labels["production_turboquant_restore_speedup"])
	core.AssertEqual(t, "0.285714", labels["production_turboquant_memory_savings_ratio"])
	core.AssertEqual(t, "0.100000", labels["production_turboquant_energy_savings_ratio"])

	parsed, err := EvaluateProductionTurboQuantPromotionDecisionLabels(labels)

	core.RequireNoError(t, err)
	core.AssertEqual(t, decision.ProductionCandidate, parsed.ProductionCandidate)
	core.AssertEqual(t, decision.EnableByDefault, parsed.EnableByDefault)
	core.AssertEqual(t, decision.Reason, parsed.Reason)
	core.AssertEqual(t, labels["production_turboquant_wall_speedup"], productionMetricFloatLabel(parsed.WallSpeedup))
	core.AssertEqual(t, labels["production_turboquant_visible_speedup"], productionMetricFloatLabel(parsed.VisibleSpeedup))
	core.AssertEqual(t, labels["production_turboquant_restore_speedup"], productionMetricFloatLabel(parsed.RestoreSpeedup))
	core.AssertEqual(t, labels["production_turboquant_memory_savings_ratio"], productionMetricFloatLabel(parsed.MemorySavingsRatio))
	core.AssertEqual(t, labels["production_turboquant_energy_savings_ratio"], productionMetricFloatLabel(parsed.EnergySavingsRatio))
}

func TestProductionPromotionDecisionLabels_Good_Combined(t *testing.T) {
	decision := EvaluateProductionCombinedMTPAndTurboQuantPromotion(
		DefaultProductionCombinedMTPAndTurboQuantPolicy(),
		productionCombinedMTPPassingEvidence(),
		productionTurboQuantPassingEvidence(),
	)

	labels := ProductionCombinedMTPAndTurboQuantDecisionLabels(decision)

	core.AssertEqual(t, "true", labels["production_combined_candidate"])
	core.AssertEqual(t, "true", labels["production_combined_enable_by_default"])
	core.AssertEqual(t, "true", labels["production_combined_mtp_eligible"])
	core.AssertEqual(t, "true", labels["production_combined_turboquant_eligible"])
	core.AssertContains(t, labels["production_combined_reason"], "production fast lane")
	core.AssertEqual(t, "1.250000", labels["production_combined_mtp_wall_speedup"])
	core.AssertEqual(t, "1.190476", labels["production_combined_mtp_visible_speedup"])
	core.AssertEqual(t, "0.750000", labels["production_combined_mtp_acceptance_rate"])
	core.AssertEqual(t, "0.285714", labels["production_combined_turboquant_memory_savings_ratio"])
	core.AssertEqual(t, "0.100000", labels["production_combined_turboquant_energy_savings_ratio"])

	parsed, err := EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels(labels)

	core.RequireNoError(t, err)
	core.AssertEqual(t, decision.ProductionCandidate, parsed.ProductionCandidate)
	core.AssertEqual(t, decision.EnableByDefault, parsed.EnableByDefault)
	core.AssertEqual(t, decision.Reason, parsed.Reason)
	core.AssertEqual(t, decision.MTPEligible, parsed.MTPEligible)
	core.AssertEqual(t, decision.TurboQuantEligible, parsed.TurboQuantEligible)
	core.AssertEqual(t, labels["production_combined_mtp_wall_speedup"], productionMetricFloatLabel(parsed.MTPWallSpeedup))
	core.AssertEqual(t, labels["production_combined_mtp_visible_speedup"], productionMetricFloatLabel(parsed.MTPVisibleSpeedup))
	core.AssertEqual(t, labels["production_combined_mtp_acceptance_rate"], productionMetricFloatLabel(parsed.MTPAcceptanceRate))
	core.AssertEqual(t, labels["production_combined_turboquant_memory_savings_ratio"], productionMetricFloatLabel(parsed.TurboQuantMemorySavingsRatio))
	core.AssertEqual(t, labels["production_combined_turboquant_energy_savings_ratio"], productionMetricFloatLabel(parsed.TurboQuantEnergySavingsRatio))
}

func TestProductionPromotionDecisionLabels_Bad_RejectsMissingDecisionLabel(t *testing.T) {
	decision := EvaluateProductionTurboQuantPromotion(DefaultProductionTurboQuantPolicy(), productionTurboQuantPassingEvidence())
	labels := ProductionTurboQuantPromotionDecisionLabels(decision)
	delete(labels, "production_turboquant_reason")

	err := ValidateProductionTurboQuantPromotionDecisionLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "production_turboquant_reason")
}

func TestProductionPromotionDecisionLabels_Bad_RejectsMalformedDecisionLabel(t *testing.T) {
	decision := EvaluateProductionMTPPromotion(DefaultProductionMTPPolicy(), productionMTPPassingEvidence())
	labels := ProductionMTPPromotionDecisionLabels(decision)
	labels["production_mtp_acceptance_rate"] = "NaN"

	_, err := EvaluateProductionMTPPromotionDecisionLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "production_mtp_acceptance_rate")
}

func TestProductionPromotionDecisionLabels_Bad_RejectsInconsistentCombinedCandidate(t *testing.T) {
	decision := EvaluateProductionCombinedMTPAndTurboQuantPromotion(
		DefaultProductionCombinedMTPAndTurboQuantPolicy(),
		productionCombinedMTPPassingEvidence(),
		productionTurboQuantPassingEvidence(),
	)
	labels := ProductionCombinedMTPAndTurboQuantDecisionLabels(decision)
	labels["production_combined_turboquant_eligible"] = "false"

	_, err := EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "both component lanes")
}

func BenchmarkProductionRequiredMetricLabels_ValidateCombined(b *testing.B) {
	labels := productionCombinedPassingLabels()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := ValidateProductionCombinedMTPAndTurboQuantRequiredMetricLabels(labels); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProductionBookGateMetricLabels_ValidatePassingRow(b *testing.B) {
	labels := productionBookGatePassingLabels()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := ValidateProductionBookGateMetricLabels(labels); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProductionBookGateMetricLabels_EvaluatePassingRow(b *testing.B) {
	labels := productionBookGatePassingLabels()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		decision, err := EvaluateProductionBookGateMetricLabels(labels)
		if err != nil {
			b.Fatal(err)
		}
		productionBookGateMetricDecisionSink = decision
	}
}

func BenchmarkProductionBookGateMetricLabels_AddFromMetrics(b *testing.B) {
	metrics := productionBookGatePassingMetrics()
	labels := make(map[string]string, len(productionBookGateMetrics))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		clear(labels)
		var err error
		labels, err = AddProductionBookGateMetricLabels(labels, metrics)
		if err != nil {
			b.Fatal(err)
		}
		productionBookGateMetricLabelsSink = labels
	}
}

func BenchmarkProductionBookGateMetrics_EvaluatePassingRow(b *testing.B) {
	metrics := productionBookGatePassingMetrics()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		decision, err := EvaluateProductionBookGateMetrics(metrics)
		if err != nil {
			b.Fatal(err)
		}
		productionBookGateMetricDecisionSink = decision
	}
}

func BenchmarkProductionBookRetainedRouteMetrics_ValidatePassingRow(b *testing.B) {
	metrics := productionBookRetainedRoutePassingMetrics()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := ValidateProductionBookRetainedRouteMetrics(metrics); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProductionBookRetainedArtifactMetrics_EvaluatePassingRow(b *testing.B) {
	metrics := productionBookRetainedArtifactPassingMetrics()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		decision, err := EvaluateProductionBookRetainedArtifactMetrics(metrics)
		if err != nil {
			b.Fatal(err)
		}
		productionBookRetainedArtifactDecisionSink = decision
	}
}

func BenchmarkProductionBookRetainedArtifactDecisionLabels_AddPassing(b *testing.B) {
	decision, err := EvaluateProductionBookRetainedArtifactMetrics(productionBookRetainedArtifactPassingMetrics())
	if err != nil {
		b.Fatal(err)
	}
	labels := make(map[string]string, 13)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		clear(labels)
		AddProductionBookRetainedArtifactDecisionLabels(labels, decision)
		productionBookGateMetricLabelsSink = labels
	}
}

func BenchmarkProductionBookRetainedArtifactDecisionLabels_EvaluatePassing(b *testing.B) {
	decision, err := EvaluateProductionBookRetainedArtifactMetrics(productionBookRetainedArtifactPassingMetrics())
	if err != nil {
		b.Fatal(err)
	}
	labels := ProductionBookRetainedArtifactDecisionLabels(decision)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		decision, err := EvaluateProductionBookRetainedArtifactDecisionLabels(labels)
		if err != nil {
			b.Fatal(err)
		}
		productionBookRetainedArtifactDecisionSink = decision
	}
}

func BenchmarkProductionBookRetainedArtifactMetricDecisionLabels_AddPassing(b *testing.B) {
	metrics := productionBookRetainedArtifactPassingMetrics()
	labels := make(map[string]string, 13)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		clear(labels)
		labels, err := AddProductionBookRetainedArtifactMetricDecisionLabels(labels, metrics)
		if err != nil {
			b.Fatal(err)
		}
		productionBookGateMetricLabelsSink = labels
	}
}

func BenchmarkProductionPromotionDecisionLabels_AddCombined(b *testing.B) {
	decision := EvaluateProductionCombinedMTPAndTurboQuantPromotion(
		DefaultProductionCombinedMTPAndTurboQuantPolicy(),
		productionCombinedMTPPassingEvidence(),
		productionTurboQuantPassingEvidence(),
	)
	labels := make(map[string]string, 10)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		clear(labels)
		AddProductionCombinedMTPAndTurboQuantDecisionLabels(labels, decision)
	}
}

func BenchmarkProductionPromotionDecisionLabels_EvaluateCombined(b *testing.B) {
	decision := EvaluateProductionCombinedMTPAndTurboQuantPromotion(
		DefaultProductionCombinedMTPAndTurboQuantPolicy(),
		productionCombinedMTPPassingEvidence(),
		productionTurboQuantPassingEvidence(),
	)
	labels := ProductionCombinedMTPAndTurboQuantDecisionLabels(decision)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		decision, err := EvaluateProductionCombinedMTPAndTurboQuantDecisionLabels(labels)
		if err != nil {
			b.Fatal(err)
		}
		productionCombinedMTPAndTurboQuantDecisionSink = decision
	}
}

var productionBookGateMetricDecisionSink ProductionBookGateMetricDecision
var productionBookGateMetricLabelsSink map[string]string
var productionBookRetainedArtifactDecisionSink ProductionBookRetainedArtifactDecision
var productionCombinedMTPAndTurboQuantDecisionSink ProductionCombinedMTPAndTurboQuantDecision

func productionBookGatePassingLabels() map[string]string {
	return map[string]string{
		"production_book_gate_candidate":        "1",
		"production_book_gate_reason_code":      "0",
		"production_book_gate_q6":               "1",
		"production_book_gate_turns":            "1",
		"production_book_gate_wall":             "1",
		"production_book_gate_decode":           "1",
		"production_book_gate_quality":          "1",
		"production_book_gate_raw_decode_tok/s": "101.5",
		"production_book_gate_wall_s":           "89.25",
		"production_book_gate_quality_flags":    "0",
	}
}

func productionBookGatePassingMetrics() map[string]float64 {
	return map[string]float64{
		"production_book_gate_candidate":        1,
		"production_book_gate_reason_code":      0,
		"production_book_gate_q6":               1,
		"production_book_gate_turns":            1,
		"production_book_gate_wall":             1,
		"production_book_gate_decode":           1,
		"production_book_gate_quality":          1,
		"production_book_gate_raw_decode_tok/s": 101.5,
		"production_book_gate_wall_s":           89.25,
		"production_book_gate_quality_flags":    0,
	}
}

func productionBookRetainedRoutePassingMetrics() map[string]float64 {
	return map[string]float64{
		"book_retained_state":                   1,
		"book_retained_state_required":          1,
		"book_prompt_replay_fallback_forbidden": 1,
		"book_state_source_runtime_kv":          1,
		"book_replay_baseline":                  0,
	}
}

func productionBookRetainedArtifactPassingMetrics() map[string]float64 {
	metrics := productionBookGatePassingMetrics()
	for key, value := range productionBookRetainedRoutePassingMetrics() {
		metrics[key] = value
	}
	return metrics
}
