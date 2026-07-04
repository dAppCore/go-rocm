// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"testing"
	"time"

	core "dappco.re/go"
)

var (
	productionTurboQuantPolicySink   ProductionTurboQuantPolicy
	productionTurboQuantDecisionSink ProductionTurboQuantPromotionDecision
)

func TestProductionTurboQuantPolicy_Defaults_Good(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()

	core.AssertEqual(t, ProductionLaneCurrentModelID, policy.TargetModelID)
	core.AssertEqual(t, rocmTurboQuantKVMode, policy.CacheMode)
	core.AssertEqual(t, 3500, policy.TargetEffectiveBitsMilli)
	core.AssertEqual(t, ProductionTurboQuantKVLayoutVersion, policy.RequiredLayoutVersion)
	core.AssertEqual(t, ProductionTurboQuantKeyAlgorithm, policy.RequiredKeyAlgorithm)
	core.AssertEqual(t, ProductionTurboQuantValueAlgorithm, policy.RequiredValueAlgorithm)
	core.AssertEqual(t, ProductionTurboQuantOutlierPolicy, policy.RequiredOutlierPolicy)
	core.AssertEqual(t, ProductionMTPPromotionMinRetainedTurns, policy.MinimumRetainedTurns)
	core.AssertEqual(t, ProductionLaneLongContextLength, policy.NormalContextLength)
	core.AssertEqual(t, ProductionLaneHyperLongContextLength, policy.StressContextLength)
	core.AssertTrue(t, !policy.RequiresExplicitOptIn, "TurboQuant must not require a CLI/API opt-in")
	core.AssertTrue(t, policy.EnabledByDefault, "TurboQuant must be part of the production fast lane")
	core.AssertTrue(t, policy.RequiresSideBySideBenchmark, "TurboQuant promotion needs side-by-side evidence")
	core.AssertEqual(t, []string{rocmKVCacheModeFP16, productionTurboQuantCacheModePaged, rocmKVCacheModeQ8, rocmKVCacheModeKQ8VQ4}, policy.CompareAgainstCacheModes)
	for _, metric := range []string{
		"retained_workflow",
		"turns",
		"quality_matches",
		"quality_flags",
		"baseline_cache_mode",
		"candidate_cache_mode",
		"candidate_layout_version",
		"candidate_key_algorithm",
		"candidate_value_algorithm",
		"candidate_outlier_policy",
		"candidate_effective_bits_milli",
		"candidate_qjl_residual",
		"candidate_metadata_bytes",
		"same_load_policy",
		"baseline_cache_policy",
		"candidate_cache_policy",
		"baseline_context_length",
		"candidate_context_length",
		"normal_context_validated",
		"stress_context_validated",
		"candidate_peak_memory_bytes",
		"baseline_peak_memory_bytes",
		"candidate_active_plus_cache_memory_bytes",
		"baseline_active_plus_cache_memory_bytes",
		"candidate_wall_duration",
		"baseline_wall_duration",
		"candidate_restore_duration",
		"baseline_restore_duration",
		"candidate_visible_tokens_per_sec",
		"baseline_visible_tokens_per_sec",
		"candidate_input_output_tokens_per_sec",
		"baseline_input_output_tokens_per_sec",
		"candidate_energy_joules",
		"baseline_energy_joules",
		"estimated_power_watts",
	} {
		if !stringSliceContains(policy.RequiredMetrics, metric) {
			t.Fatalf("RequiredMetrics = %v, missing %q", policy.RequiredMetrics, metric)
		}
	}
}

func TestProductionTurboQuantPolicy_Good_DefensiveCopies(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()
	policy.CompareAgainstCacheModes[0] = "mutated"
	policy.RequiredMetrics[0] = "mutated"

	next := DefaultProductionTurboQuantPolicy()
	core.AssertEqual(t, rocmKVCacheModeFP16, next.CompareAgainstCacheModes[0])
	core.AssertEqual(t, "retained_workflow", next.RequiredMetrics[0])
}

func TestProductionTurboQuantPromotion_Good_AcceptsExplicitCandidate(t *testing.T) {
	decision := EvaluateProductionTurboQuantPromotion(DefaultProductionTurboQuantPolicy(), productionTurboQuantPassingEvidence())

	if !decision.ProductionCandidate || !decision.EnableByDefault {
		t.Fatalf("decision = %+v, want production candidate with default enablement", decision)
	}
	core.AssertContains(t, decision.Reason, "quality parity")
	if decision.WallSpeedup <= 1 || decision.RestoreSpeedup <= 1 || decision.MemorySavingsRatio <= 0 || decision.EnergySavingsRatio <= 0 {
		t.Fatalf("decision metrics = %+v, want wall/restore/memory/energy improvement", decision)
	}
}

func TestProductionTurboQuantPromotion_Bad_RejectsMissingEvidence(t *testing.T) {
	evidence := productionTurboQuantPassingEvidence()
	evidence.CandidateCacheMode = rocmKVCacheModeQ8
	decision := EvaluateProductionTurboQuantPromotion(DefaultProductionTurboQuantPolicy(), evidence)
	core.AssertEqual(t, false, decision.ProductionCandidate)
	core.AssertContains(t, decision.Reason, "candidate cache mode")

	evidence = productionTurboQuantPassingEvidence()
	evidence.ComparedCacheModes = []string{rocmKVCacheModeFP16}
	decision = EvaluateProductionTurboQuantPromotion(DefaultProductionTurboQuantPolicy(), evidence)
	core.AssertContains(t, decision.Reason, "side by side")

	evidence = productionTurboQuantPassingEvidence()
	evidence.CandidateLayoutVersion = ""
	decision = EvaluateProductionTurboQuantPromotion(DefaultProductionTurboQuantPolicy(), evidence)
	core.AssertContains(t, decision.Reason, "layout version")

	evidence = productionTurboQuantPassingEvidence()
	evidence.CandidateActivePlusCacheMemoryBytes = evidence.BaselineActivePlusCacheMemoryBytes
	decision = EvaluateProductionTurboQuantPromotion(DefaultProductionTurboQuantPolicy(), evidence)
	core.AssertContains(t, decision.Reason, "active+cache memory savings")
}

func TestProductionTurboQuantPromotion_Bad_RejectsQualityRegressions(t *testing.T) {
	evidence := productionTurboQuantPassingEvidence()
	evidence.QualityMatches = false
	decision := EvaluateProductionTurboQuantPromotion(DefaultProductionTurboQuantPolicy(), evidence)
	core.AssertContains(t, decision.Reason, "quality parity")

	evidence = productionTurboQuantPassingEvidence()
	evidence.QualityFlags = []string{"chapter_drift"}
	decision = EvaluateProductionTurboQuantPromotion(DefaultProductionTurboQuantPolicy(), evidence)
	core.AssertContains(t, decision.Reason, "quality flags")

	policy := DefaultProductionTurboQuantPolicy()
	policy.EnabledByDefault = false
	decision = EvaluateProductionTurboQuantPromotion(policy, productionTurboQuantPassingEvidence())
	core.AssertEqual(t, true, decision.ProductionCandidate)
	core.AssertEqual(t, false, decision.EnableByDefault)
}

func TestProductionTurboQuantLabelEvidence_Good_FillsStaticCapabilityLabels(t *testing.T) {
	evidence := ProductionTurboQuantPromotionEvidence{}
	labels := map[string]string{
		"kv_compression":                         rocmTurboQuantKVMode,
		"production_compare_cache_modes":         "fp16,paged,q8,k-q8-v-q4",
		"production_required_key_algorithm":      ProductionTurboQuantKeyAlgorithm,
		"production_required_layout_version":     ProductionTurboQuantKVLayoutVersion,
		"production_required_outlier_policy":     ProductionTurboQuantOutlierPolicy,
		"production_required_value_algorithm":    ProductionTurboQuantValueAlgorithm,
		"production_target_effective_bits_milli": "3500",
	}

	err := ApplyProductionTurboQuantLabelEvidence(&evidence, labels)

	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmTurboQuantKVMode, evidence.CandidateCacheMode)
	core.AssertEqual(t, ProductionTurboQuantKVLayoutVersion, evidence.CandidateLayoutVersion)
	core.AssertEqual(t, ProductionTurboQuantKeyAlgorithm, evidence.CandidateKeyAlgorithm)
	core.AssertEqual(t, ProductionTurboQuantValueAlgorithm, evidence.CandidateValueAlgorithm)
	core.AssertEqual(t, ProductionTurboQuantOutlierPolicy, evidence.CandidateOutlierPolicy)
	core.AssertEqual(t, 3500, evidence.CandidateEffectiveBitsMilli)
	core.AssertEqual(t, []string{rocmKVCacheModeFP16, productionTurboQuantCacheModePaged, rocmKVCacheModeQ8, rocmKVCacheModeKQ8VQ4}, evidence.ComparedCacheModes)
}

func TestProductionTurboQuantLabelEvidence_Good_MeasuredLabelsPromote(t *testing.T) {
	evidence := ProductionTurboQuantPromotionEvidence{}
	labels := productionTurboQuantPassingLabels()

	err := ApplyProductionTurboQuantLabelEvidence(&evidence, labels)
	decision := EvaluateProductionTurboQuantPromotion(DefaultProductionTurboQuantPolicy(), evidence)

	core.RequireNoError(t, err)
	if !decision.ProductionCandidate {
		t.Fatalf("decision = %+v evidence=%+v, want labels to produce passing TurboQuant evidence", decision, evidence)
	}
}

func TestProductionTurboQuantPromotionMetricLabels_Good_EvaluatesPassingLabels(t *testing.T) {
	decision, err := EvaluateProductionTurboQuantPromotionMetricLabels(productionTurboQuantPassingLabels())

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, decision.ProductionCandidate)
	core.AssertContains(t, decision.Reason, "TurboQuant retained workflow")
	core.AssertGreater(t, decision.MemorySavingsRatio, float64(0))
}

func TestProductionTurboQuantPromotionMetricLabels_Good_EvaluatesValidNonPromotingLabels(t *testing.T) {
	labels := productionTurboQuantPassingLabels()
	labels["candidate_active_plus_cache_memory_bytes"] = labels["baseline_active_plus_cache_memory_bytes"]

	decision, err := EvaluateProductionTurboQuantPromotionMetricLabels(labels)

	core.RequireNoError(t, err)
	core.AssertEqual(t, false, decision.ProductionCandidate)
	core.AssertContains(t, decision.Reason, "active+cache memory")
}

func TestProductionTurboQuantPromotionMetricLabels_Bad_RejectsMissingRequiredMetric(t *testing.T) {
	labels := productionTurboQuantPassingLabels()
	delete(labels, "candidate_effective_bits_milli")

	err := ValidateProductionTurboQuantPromotionMetricLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "candidate_effective_bits_milli")
}

func TestProductionTurboQuantPromotionMetricLabels_Bad_RejectsMalformedMetric(t *testing.T) {
	labels := productionTurboQuantPassingLabels()
	labels["candidate_effective_bits_milli"] = "three-point-five"

	_, err := EvaluateProductionTurboQuantPromotionMetricLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "candidate_effective_bits_milli")
}

func TestProductionTurboQuantLabelEvidence_Bad_InvalidMeasuredValue(t *testing.T) {
	evidence := ProductionTurboQuantPromotionEvidence{}
	labels := productionTurboQuantPassingLabels()
	labels["candidate_effective_bits_milli"] = "three-point-five"

	err := ApplyProductionTurboQuantLabelEvidence(&evidence, labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "candidate_effective_bits_milli")
}

func BenchmarkProductionTurboQuantPromotion_PassingEvidence(b *testing.B) {
	policy := DefaultProductionTurboQuantPolicy()
	evidence := productionTurboQuantPassingEvidence()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		productionTurboQuantDecisionSink = EvaluateProductionTurboQuantPromotion(policy, evidence)
	}
}

func BenchmarkProductionTurboQuantPolicy_Default(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		productionTurboQuantPolicySink = DefaultProductionTurboQuantPolicy()
	}
}

func BenchmarkProductionTurboQuantLabelEvidence_ApplyMeasuredLabels(b *testing.B) {
	labels := productionTurboQuantPassingLabels()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var evidence ProductionTurboQuantPromotionEvidence
		if err := ApplyProductionTurboQuantLabelEvidence(&evidence, labels); err != nil {
			b.Fatal(err)
		}
		productionTurboQuantDecisionSink = EvaluateProductionTurboQuantPromotion(DefaultProductionTurboQuantPolicy(), evidence)
	}
}

func BenchmarkProductionTurboQuantPromotionMetricLabels_EvaluatePassing(b *testing.B) {
	labels := productionTurboQuantPassingLabels()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		decision, err := EvaluateProductionTurboQuantPromotionMetricLabels(labels)
		if err != nil {
			b.Fatal(err)
		}
		productionTurboQuantDecisionSink = decision
	}
}

func productionTurboQuantPassingEvidence() ProductionTurboQuantPromotionEvidence {
	return ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:                    true,
		Turns:                               ProductionMTPPromotionMinRetainedTurns,
		QualityMatches:                      true,
		BaselineCacheMode:                   rocmKVCacheModeKQ8VQ4,
		CandidateCacheMode:                  rocmTurboQuantKVMode,
		CandidateLayoutVersion:              ProductionTurboQuantKVLayoutVersion,
		CandidateKeyAlgorithm:               ProductionTurboQuantKeyAlgorithm,
		CandidateValueAlgorithm:             ProductionTurboQuantValueAlgorithm,
		CandidateOutlierPolicy:              ProductionTurboQuantOutlierPolicy,
		CandidateEffectiveBitsMilli:         3500,
		CandidateQJLResidual:                true,
		CandidateMetadataBytes:              8192,
		SameLoadPolicy:                      true,
		BaselineCachePolicy:                 "retained-state",
		CandidateCachePolicy:                "retained-state",
		BaselineContextLength:               ProductionLaneLongContextLength,
		CandidateContextLength:              ProductionLaneLongContextLength,
		ComparedCacheModes:                  []string{rocmKVCacheModeFP16, productionTurboQuantCacheModePaged, rocmKVCacheModeQ8, rocmKVCacheModeKQ8VQ4},
		NormalContextValidated:              true,
		StressContextValidated:              true,
		BaselineVisibleTokensPerSec:         100,
		CandidateVisibleTokensPerSec:        102,
		BaselineInputOutputTokensPerSec:     32000,
		CandidateInputOutputTokensPerSec:    34000,
		BaselineWallDuration:                10 * time.Second,
		CandidateWallDuration:               9 * time.Second,
		BaselineRestoreDuration:             100 * time.Millisecond,
		CandidateRestoreDuration:            70 * time.Millisecond,
		BaselinePeakMemoryBytes:             8 << 30,
		CandidatePeakMemoryBytes:            6 << 30,
		BaselineActivePlusCacheMemoryBytes:  7 << 30,
		CandidateActivePlusCacheMemoryBytes: 5 << 30,
		BaselineEnergyJoules:                500,
		CandidateEnergyJoules:               450,
		EstimatedPowerWatts:                 50,
	}
}

func productionTurboQuantPassingLabels() map[string]string {
	return map[string]string{
		"retained_workflow":                        "true",
		"turns":                                    "10",
		"quality_matches":                          "true",
		"quality_flags":                            "",
		"baseline_cache_mode":                      rocmKVCacheModeKQ8VQ4,
		"candidate_cache_mode":                     rocmTurboQuantKVMode,
		"candidate_layout_version":                 ProductionTurboQuantKVLayoutVersion,
		"candidate_key_algorithm":                  ProductionTurboQuantKeyAlgorithm,
		"candidate_value_algorithm":                ProductionTurboQuantValueAlgorithm,
		"candidate_outlier_policy":                 ProductionTurboQuantOutlierPolicy,
		"candidate_effective_bits_milli":           "3500",
		"candidate_qjl_residual":                   "true",
		"candidate_metadata_bytes":                 "8192",
		"same_load_policy":                         "true",
		"baseline_cache_policy":                    "retained-state",
		"candidate_cache_policy":                   "retained-state",
		"baseline_context_length":                  "32768",
		"candidate_context_length":                 "32768",
		"compared_cache_modes":                     "fp16,paged,q8,k-q8-v-q4",
		"normal_context_validated":                 "true",
		"stress_context_validated":                 "true",
		"baseline_visible_tokens_per_sec":          "100",
		"candidate_visible_tokens_per_sec":         "102",
		"baseline_input_output_tokens_per_sec":     "32000",
		"candidate_input_output_tokens_per_sec":    "34000",
		"baseline_wall_duration":                   "10s",
		"candidate_wall_duration":                  "9s",
		"baseline_restore_duration":                "0.100",
		"candidate_restore_duration":               "70ms",
		"baseline_peak_memory_bytes":               "8589934592",
		"candidate_peak_memory_bytes":              "6442450944",
		"baseline_active_plus_cache_memory_bytes":  "7516192768",
		"candidate_active_plus_cache_memory_bytes": "5368709120",
		"baseline_energy_joules":                   "500",
		"candidate_energy_joules":                  "450",
		"estimated_power_watts":                    "50",
	}
}
