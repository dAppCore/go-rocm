// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"testing"

	core "dappco.re/go"
)

var (
	productionCombinedPolicySink   ProductionCombinedMTPAndTurboQuantPolicy
	productionCombinedDecisionSink ProductionCombinedMTPAndTurboQuantDecision
)

func TestProductionCombinedMTPAndTurboQuantPolicy_Defaults_Good(t *testing.T) {
	policy := DefaultProductionCombinedMTPAndTurboQuantPolicy()

	core.AssertEqual(t, officialGemma4E2BTargetModelID, policy.TargetModelID)
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, policy.AssistantModelID)
	core.AssertEqual(t, ProductionCombinedMTPAndTurboQuantMode, policy.Mode)
	core.AssertEqual(t, rocmTurboQuantKVMode, policy.CacheMode)
	core.AssertTrue(t, policy.EnabledByDefault, "combined lane must be the production fast-lane default")
	core.AssertTrue(t, !policy.RequiresExplicitOptIn, "combined lane must not require a CLI/API opt-in")
	core.AssertTrue(t, policy.RequiresMTPPromotion, "combined lane must require MTP promotion")
	core.AssertTrue(t, policy.RequiresTurboQuantPromotion, "combined lane must require TurboQuant promotion")
	for _, metric := range []string{
		"retained_workflow",
		"turns",
		"quality_matches",
		"mtp_greedy_output_matches",
		"quality_flags",
		"mtp_target_only_cache_mode",
		"mtp_cache_mode",
		"mtp_target_only_visible_tokens_per_sec",
		"mtp_visible_tokens_per_sec",
		"mtp_target_tokens_per_sec",
		"mtp_warm_decode_tokens_per_sec",
		"mtp_target_only_wall_duration",
		"mtp_wall_duration",
		"mtp_target_only_restore_duration",
		"mtp_restore_duration",
		"mtp_target_only_peak_memory_bytes",
		"mtp_peak_memory_bytes",
		"mtp_target_only_active_plus_cache_memory_bytes",
		"mtp_active_plus_cache_memory_bytes",
		"mtp_target_only_energy_joules",
		"mtp_energy_joules",
		"mtp_observed_draft_token_sweeps",
		"mtp_proposed_tokens",
		"mtp_accepted_tokens",
		"mtp_rejected_tokens",
		"mtp_target_verify_calls",
		"mtp_draft_calls",
		"attached_drafter_retained_state_entrypoint",
		"attached_drafter_retained_state_required",
		"attached_drafter_state_source",
		"attached_drafter_prompt_replay_fallback",
		"attached_drafter_native_attachment",
		"attached_drafter_native_handoff",
		"attached_drafter_target_retained_decode",
		"attached_drafter_target_retained_state_decode",
		"attached_drafter_assistant_verify",
		"attached_drafter_assistant_state_verify",
		"attached_drafter_target_gemma4_size",
		"attached_drafter_target_gemma4_quant_mode",
		"attached_drafter_target_gemma4_quant_group",
		"attached_drafter_target_gemma4_runtime",
		"attached_drafter_target_gemma4_generate_status",
		"attached_drafter_assistant_gemma4_size",
		"attached_drafter_assistant_gemma4_quant_mode",
		"attached_drafter_assistant_gemma4_runtime",
		"attached_drafter_assistant_gemma4_generate_status",
		"assistant_architecture",
		"assistant_ordered_embeddings",
		"assistant_centroids",
		"assistant_centroid_intermediate_top_k",
		"assistant_four_layer_drafter",
		"assistant_token_ordering_dtype",
		"assistant_token_ordering_shape",
		"gemma4_family_pair_verified",
		"baseline_cache_mode",
		"turboquant_candidate_cache_mode",
		"same_load_policy",
		"baseline_cache_policy",
		"turboquant_candidate_cache_policy",
		"baseline_context_length",
		"candidate_context_length",
		"compared_cache_modes",
		"turboquant_normal_context_validated",
		"turboquant_stress_context_validated",
		"turboquant_candidate_layout_version",
		"turboquant_candidate_key_algorithm",
		"turboquant_candidate_value_algorithm",
		"turboquant_candidate_outlier_policy",
		"turboquant_candidate_effective_bits_milli",
		"turboquant_candidate_qjl_residual",
		"turboquant_candidate_metadata_bytes",
		"turboquant_quality_flags",
		"baseline_visible_tokens_per_sec",
		"turboquant_candidate_visible_tokens_per_sec",
		"baseline_input_output_tokens_per_sec",
		"turboquant_candidate_input_output_tokens_per_sec",
		"baseline_wall_duration",
		"turboquant_candidate_wall_duration",
		"baseline_restore_duration",
		"turboquant_candidate_restore_duration",
		"baseline_peak_memory_bytes",
		"turboquant_candidate_peak_memory_bytes",
		"baseline_active_plus_cache_memory_bytes",
		"turboquant_candidate_active_plus_cache_memory_bytes",
		"baseline_energy_joules",
		"turboquant_candidate_energy_joules",
		"estimated_power_watts",
		"turboquant_active_plus_cache_memory_savings",
	} {
		if !stringSliceContains(policy.RequiredMetrics, metric) {
			t.Fatalf("RequiredMetrics = %v, missing %q", policy.RequiredMetrics, metric)
		}
	}
}

func TestProductionCombinedMTPAndTurboQuantPolicy_Good_DefensiveCopies(t *testing.T) {
	policy := DefaultProductionCombinedMTPAndTurboQuantPolicy()
	policy.RequiredMetrics[0] = "mutated"

	next := DefaultProductionCombinedMTPAndTurboQuantPolicy()
	core.AssertEqual(t, "retained_workflow", next.RequiredMetrics[0])
}

func TestProductionCombinedMTPAndTurboQuantPromotion_Good_AllowsExplicitCandidate(t *testing.T) {
	decision := EvaluateProductionCombinedMTPAndTurboQuantPromotion(
		DefaultProductionCombinedMTPAndTurboQuantPolicy(),
		productionCombinedMTPPassingEvidence(),
		productionTurboQuantPassingEvidence(),
	)

	if !decision.ProductionCandidate || !decision.EnableByDefault {
		t.Fatalf("decision = %+v, want default production candidate", decision)
	}
	if !decision.MTPEligible || !decision.TurboQuantEligible {
		t.Fatalf("eligibility = mtp:%v turbo:%v, want both component gates passing", decision.MTPEligible, decision.TurboQuantEligible)
	}
	if decision.MTPAcceptanceRate <= 0 || decision.TurboQuantMemorySavingsRatio <= 0 {
		t.Fatalf("decision metrics = %+v, want MTP acceptance and TurboQuant memory savings", decision)
	}
}

func TestProductionCombinedMTPAndTurboQuantPromotion_Bad_RejectsWrongCacheMode(t *testing.T) {
	mtpEvidence := productionMTPPassingEvidence()
	decision := EvaluateProductionCombinedMTPAndTurboQuantPromotion(
		DefaultProductionCombinedMTPAndTurboQuantPolicy(),
		mtpEvidence,
		productionTurboQuantPassingEvidence(),
	)

	core.AssertEqual(t, false, decision.ProductionCandidate)
	core.AssertContains(t, decision.Reason, "must run target-only and MTP with TurboQuant")
}

func TestProductionCombinedMTPAndTurboQuantPromotion_Bad_RejectsComponentRegressions(t *testing.T) {
	mtpEvidence := productionCombinedMTPPassingEvidence()
	mtpEvidence.MTPAcceptedTokens = 0
	mtpEvidence.MTPRejectedTokens = mtpEvidence.MTPProposedTokens
	decision := EvaluateProductionCombinedMTPAndTurboQuantPromotion(
		DefaultProductionCombinedMTPAndTurboQuantPolicy(),
		mtpEvidence,
		productionTurboQuantPassingEvidence(),
	)
	core.AssertContains(t, decision.Reason, "MTP must pass")

	turboEvidence := productionTurboQuantPassingEvidence()
	turboEvidence.QualityFlags = []string{"chapter_drift"}
	decision = EvaluateProductionCombinedMTPAndTurboQuantPromotion(
		DefaultProductionCombinedMTPAndTurboQuantPolicy(),
		productionCombinedMTPPassingEvidence(),
		turboEvidence,
	)
	core.AssertContains(t, decision.Reason, "TurboQuant must pass")

	policy := DefaultProductionCombinedMTPAndTurboQuantPolicy()
	policy.EnabledByDefault = false
	decision = EvaluateProductionCombinedMTPAndTurboQuantPromotion(policy, productionCombinedMTPPassingEvidence(), productionTurboQuantPassingEvidence())
	core.AssertEqual(t, true, decision.ProductionCandidate)
	core.AssertEqual(t, false, decision.EnableByDefault)
}

func TestProductionCombinedMTPAndTurboQuantLabelEvidence_Good_MeasuredLabelsPromote(t *testing.T) {
	var mtpEvidence ProductionMTPPromotionEvidence
	var turboEvidence ProductionTurboQuantPromotionEvidence
	labels := productionCombinedPassingLabels()

	err := ApplyProductionCombinedMTPAndTurboQuantLabelEvidence(&mtpEvidence, &turboEvidence, labels)
	decision := EvaluateProductionCombinedMTPAndTurboQuantPromotion(DefaultProductionCombinedMTPAndTurboQuantPolicy(), mtpEvidence, turboEvidence)

	core.RequireNoError(t, err)
	if !decision.ProductionCandidate {
		t.Fatalf("decision = %+v mtp=%+v turbo=%+v, want combined labels to produce passing evidence", decision, mtpEvidence, turboEvidence)
	}
	core.AssertEqual(t, "forbidden", mtpEvidence.AttachedDrafterPromptReplayFallback)
	core.AssertEqual(t, rocmTurboQuantKVMode, mtpEvidence.MTPCacheMode)
	core.AssertEqual(t, rocmTurboQuantKVMode, turboEvidence.CandidateCacheMode)
}

func TestProductionCombinedMTPAndTurboQuantPromotionMetricLabels_Good_EvaluatesPassingLabels(t *testing.T) {
	decision, err := EvaluateProductionCombinedMTPAndTurboQuantPromotionMetricLabels(productionCombinedPassingLabels())

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, decision.ProductionCandidate)
	core.AssertEqual(t, true, decision.MTPEligible)
	core.AssertEqual(t, true, decision.TurboQuantEligible)
	core.AssertContains(t, decision.Reason, "production fast lane")
}

func TestProductionCombinedMTPAndTurboQuantPromotionMetricLabels_Good_EvaluatesValidNonPromotingLabels(t *testing.T) {
	labels := productionCombinedPassingLabels()
	labels["mtp_visible_tokens_per_sec"] = "99"

	decision, err := EvaluateProductionCombinedMTPAndTurboQuantPromotionMetricLabels(labels)

	core.RequireNoError(t, err)
	core.AssertEqual(t, false, decision.ProductionCandidate)
	core.AssertEqual(t, false, decision.MTPEligible)
	core.AssertEqual(t, true, decision.TurboQuantEligible)
	core.AssertContains(t, decision.Reason, "MTP must pass")
}

func TestProductionCombinedMTPAndTurboQuantPromotionMetricLabels_Bad_RejectsMissingRequiredMetric(t *testing.T) {
	labels := productionCombinedPassingLabels()
	delete(labels, "mtp_draft_calls")

	err := ValidateProductionCombinedMTPAndTurboQuantPromotionMetricLabels(labels)

	core.AssertError(t, err)
	if err != nil {
		core.AssertContains(t, err.Error(), "mtp_draft_calls")
	}
}

func TestProductionCombinedMTPAndTurboQuantPromotionMetricLabels_Bad_RejectsMalformedMetric(t *testing.T) {
	labels := productionCombinedPassingLabels()
	labels["mtp_proposed_tokens"] = "forty"

	_, err := EvaluateProductionCombinedMTPAndTurboQuantPromotionMetricLabels(labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "mtp_proposed_tokens")
}

func TestProductionCombinedMTPAndTurboQuantLabelEvidence_Good_AcceptsRequiredPrefixedMTPMetricNames(t *testing.T) {
	var mtpEvidence ProductionMTPPromotionEvidence
	var turboEvidence ProductionTurboQuantPromotionEvidence
	labels := productionCombinedPrefixedMTPPassingLabels()

	err := ApplyProductionCombinedMTPAndTurboQuantLabelEvidence(&mtpEvidence, &turboEvidence, labels)
	decision := EvaluateProductionCombinedMTPAndTurboQuantPromotion(DefaultProductionCombinedMTPAndTurboQuantPolicy(), mtpEvidence, turboEvidence)

	core.RequireNoError(t, err)
	if !decision.ProductionCandidate {
		t.Fatalf("decision = %+v mtp=%+v turbo=%+v, want prefixed combined MTP labels to produce passing evidence", decision, mtpEvidence, turboEvidence)
	}
	core.AssertEqual(t, rocmTurboQuantKVMode, mtpEvidence.TargetOnlyCacheMode)
	core.AssertEqual(t, uint64(4096), mtpEvidence.TargetOnlyPeakMemoryBytes)
	core.AssertEqual(t, float64(1000), mtpEvidence.TargetOnlyEnergyJoules)
}

func TestProductionCombinedMTPAndTurboQuantLabelEvidence_Bad_InvalidMeasuredValue(t *testing.T) {
	var mtpEvidence ProductionMTPPromotionEvidence
	var turboEvidence ProductionTurboQuantPromotionEvidence
	labels := productionCombinedPassingLabels()
	labels["mtp_proposed_tokens"] = "forty"

	err := ApplyProductionCombinedMTPAndTurboQuantLabelEvidence(&mtpEvidence, &turboEvidence, labels)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "mtp_proposed_tokens")
}

func BenchmarkProductionCombinedMTPAndTurboQuantPromotion_PassingEvidence(b *testing.B) {
	policy := DefaultProductionCombinedMTPAndTurboQuantPolicy()
	mtpEvidence := productionCombinedMTPPassingEvidence()
	turboEvidence := productionTurboQuantPassingEvidence()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		productionCombinedDecisionSink = EvaluateProductionCombinedMTPAndTurboQuantPromotion(policy, mtpEvidence, turboEvidence)
	}
}

func BenchmarkProductionCombinedMTPAndTurboQuantLabelEvidence_ApplyMeasuredLabels(b *testing.B) {
	labels := productionCombinedPassingLabels()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var mtpEvidence ProductionMTPPromotionEvidence
		var turboEvidence ProductionTurboQuantPromotionEvidence
		if err := ApplyProductionCombinedMTPAndTurboQuantLabelEvidence(&mtpEvidence, &turboEvidence, labels); err != nil {
			b.Fatal(err)
		}
		productionCombinedDecisionSink = EvaluateProductionCombinedMTPAndTurboQuantPromotion(DefaultProductionCombinedMTPAndTurboQuantPolicy(), mtpEvidence, turboEvidence)
	}
}

func BenchmarkProductionCombinedMTPAndTurboQuantPromotionMetricLabels_EvaluatePassing(b *testing.B) {
	labels := productionCombinedPassingLabels()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		decision, err := EvaluateProductionCombinedMTPAndTurboQuantPromotionMetricLabels(labels)
		if err != nil {
			b.Fatal(err)
		}
		productionCombinedDecisionSink = decision
	}
}

func BenchmarkProductionCombinedMTPAndTurboQuantPolicy_Default(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		productionCombinedPolicySink = DefaultProductionCombinedMTPAndTurboQuantPolicy()
	}
}

func productionCombinedMTPPassingEvidence() ProductionMTPPromotionEvidence {
	evidence := productionMTPPassingEvidence()
	evidence.TargetOnlyCacheMode = rocmTurboQuantKVMode
	evidence.MTPCacheMode = rocmTurboQuantKVMode
	return evidence
}

func productionCombinedPassingLabels() map[string]string {
	labels := productionTurboQuantPassingLabels()
	rocmAddGemma4AttachedDrafterCapabilityLabels(labels)
	labels["retained_workflow"] = "true"
	labels["turns"] = "10"
	labels["greedy_output_matches"] = "true"
	labels["target_only_visible_tokens_per_sec"] = "105"
	labels["mtp_visible_tokens_per_sec"] = "125"
	labels["mtp_target_tokens_per_sec"] = "110"
	labels["mtp_warm_decode_tokens_per_sec"] = "123"
	labels["target_only_wall_duration"] = "10s"
	labels["mtp_wall_duration"] = "8s"
	labels["target_only_restore_duration"] = "100ms"
	labels["mtp_restore_duration"] = "80ms"
	labels["target_only_peak_memory_bytes"] = "4096"
	labels["mtp_peak_memory_bytes"] = "3584"
	labels["target_only_active_plus_cache_memory_bytes"] = "2560"
	labels["mtp_active_plus_cache_memory_bytes"] = "2304"
	labels["target_only_energy_joules"] = "1000"
	labels["mtp_energy_joules"] = "760"
	labels["same_load_policy"] = "true"
	labels["target_only_cache_mode"] = rocmTurboQuantKVMode
	labels["mtp_cache_mode"] = rocmTurboQuantKVMode
	markProductionMTPNativeHandoffLabelsLinked(labels)
	labels["turboquant_active_plus_cache_memory_savings"] = "2147483648"
	labels["mtp_draft_token_schedule"] = "2,2"
	labels["mtp_observed_draft_token_sweeps"] = "1,2,4"
	labels["mtp_proposed_tokens"] = "40"
	labels["mtp_accepted_tokens"] = "30"
	labels["mtp_rejected_tokens"] = "10"
	labels["mtp_target_verify_calls"] = "20"
	labels["mtp_draft_calls"] = "20"
	return labels
}

func productionCombinedPrefixedMTPPassingLabels() map[string]string {
	labels := map[string]string{
		"retained_workflow":                                   "true",
		"turns":                                               "10",
		"quality_matches":                                     "true",
		"quality_flags":                                       "",
		"baseline_cache_mode":                                 rocmKVCacheModeKQ8VQ4,
		"turboquant_candidate_cache_mode":                     rocmTurboQuantKVMode,
		"turboquant_candidate_layout_version":                 ProductionTurboQuantKVLayoutVersion,
		"turboquant_candidate_key_algorithm":                  ProductionTurboQuantKeyAlgorithm,
		"turboquant_candidate_value_algorithm":                ProductionTurboQuantValueAlgorithm,
		"turboquant_candidate_outlier_policy":                 ProductionTurboQuantOutlierPolicy,
		"turboquant_candidate_effective_bits_milli":           "3500",
		"turboquant_candidate_qjl_residual":                   "true",
		"turboquant_candidate_metadata_bytes":                 "8192",
		"same_load_policy":                                    "true",
		"baseline_cache_policy":                               "retained-state",
		"turboquant_candidate_cache_policy":                   "retained-state",
		"baseline_context_length":                             "32768",
		"candidate_context_length":                            "32768",
		"compared_cache_modes":                                "fp16,paged,q8,k-q8-v-q4",
		"turboquant_normal_context_validated":                 "true",
		"turboquant_stress_context_validated":                 "true",
		"baseline_visible_tokens_per_sec":                     "100",
		"turboquant_candidate_visible_tokens_per_sec":         "102",
		"baseline_input_output_tokens_per_sec":                "32000",
		"turboquant_candidate_input_output_tokens_per_sec":    "34000",
		"baseline_wall_duration":                              "10s",
		"turboquant_candidate_wall_duration":                  "9s",
		"baseline_restore_duration":                           "0.100",
		"turboquant_candidate_restore_duration":               "70ms",
		"baseline_peak_memory_bytes":                          "8589934592",
		"turboquant_candidate_peak_memory_bytes":              "6442450944",
		"baseline_active_plus_cache_memory_bytes":             "7516192768",
		"turboquant_candidate_active_plus_cache_memory_bytes": "5368709120",
		"baseline_energy_joules":                              "500",
		"turboquant_candidate_energy_joules":                  "450",
		"estimated_power_watts":                               "50",
		"turboquant_active_plus_cache_memory_savings":         "2147483648",
	}
	rocmAddGemma4AttachedDrafterCapabilityLabels(labels)
	markProductionMTPNativeHandoffLabelsLinked(labels)
	labels["mtp_retained_workflow"] = "true"
	labels["mtp_turns"] = "10"
	labels["mtp_greedy_output_matches"] = "true"
	labels["mtp_target_only_visible_tokens_per_sec"] = "105"
	labels["mtp_visible_tokens_per_sec"] = "125"
	labels["mtp_target_tokens_per_sec"] = "110"
	labels["mtp_warm_decode_tokens_per_sec"] = "123"
	labels["mtp_target_only_wall_duration"] = "10s"
	labels["mtp_wall_duration"] = "8s"
	labels["mtp_target_only_restore_duration"] = "100ms"
	labels["mtp_restore_duration"] = "80ms"
	labels["mtp_target_only_peak_memory_bytes"] = "4096"
	labels["mtp_peak_memory_bytes"] = "3584"
	labels["mtp_target_only_active_plus_cache_memory_bytes"] = "2560"
	labels["mtp_active_plus_cache_memory_bytes"] = "2304"
	labels["mtp_target_only_energy_joules"] = "1000"
	labels["mtp_energy_joules"] = "760"
	labels["mtp_same_load_policy"] = "true"
	labels["mtp_target_only_cache_mode"] = rocmTurboQuantKVMode
	labels["mtp_cache_mode"] = rocmTurboQuantKVMode
	labels["mtp_draft_token_schedule"] = "2,2"
	labels["mtp_observed_draft_token_sweeps"] = "1,2,4"
	labels["mtp_proposed_tokens"] = "40"
	labels["mtp_accepted_tokens"] = "30"
	labels["mtp_rejected_tokens"] = "10"
	labels["mtp_target_verify_calls"] = "20"
	labels["mtp_draft_calls"] = "20"
	return labels
}
