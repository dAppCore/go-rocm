// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"strconv"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	modelgemma4 "dappco.re/go/rocm/model/gemma4"
)

func TestProductionLane_DefaultGemma4E2B_Good(t *testing.T) {
	lane := DefaultProductionLane()

	if lane.ModelID != ProductionLaneCurrentModelID {
		t.Fatalf("ModelID = %q, want Gemma 4 E2B q6 default", lane.ModelID)
	}
	if lane.Architecture != "gemma4_text" || lane.ChatTemplate != "gemma4" || lane.QuantBits != 6 {
		t.Fatalf("lane identity = %+v, want Gemma 4 text q6 with Gemma chat template", lane)
	}
	if ProductionLaneProductDefaultQuantBits != 6 || ProductionLaneQualityQuantBits != 8 || ProductionLaneConstrainedQuantBits != 4 {
		t.Fatalf("quant constants = default:%d quality:%d constrained:%d, want 6/8/4", ProductionLaneProductDefaultQuantBits, ProductionLaneQualityQuantBits, ProductionLaneConstrainedQuantBits)
	}
	if ProductionLaneLongContextLength != 32768 || ProductionLaneHyperLongContextLength != 131072 || ProductionLaneLongFormMaxTokens != 8192 || ProductionLaneLongContextPrefillChunk != 512 || ProductionLaneLongContextPromptBytes != 4096 || ProductionLanePagedKVPageSize != 2048 || ProductionLaneRetainedKVCacheDType != "fp16" {
		t.Fatalf("long context shape = context:%d hyper:%d tokens:%d prefill:%d prompt:%d page:%d dtype:%s, want retained-state defaults", ProductionLaneLongContextLength, ProductionLaneHyperLongContextLength, ProductionLaneLongFormMaxTokens, ProductionLaneLongContextPrefillChunk, ProductionLaneLongContextPromptBytes, ProductionLanePagedKVPageSize, ProductionLaneRetainedKVCacheDType)
	}
	if lane.ContextLength != 0 {
		t.Fatalf("lane ContextLength = %d, want model-window driver default", lane.ContextLength)
	}
	if lane.MaxTokens != 0 {
		t.Fatalf("lane MaxTokens = %d, want uncapped driver-resolved default", lane.MaxTokens)
	}
	if !lane.TraceTokenPhases || lane.IncludeOutput {
		t.Fatalf("profile reporting = trace:%v include_output:%v, want phase trace without output capture", lane.TraceTokenPhases, lane.IncludeOutput)
	}
}

func TestProductionLane_DefaultProductionFastLane_Good(t *testing.T) {
	fast := DefaultProductionFastLane()

	core.AssertEqual(t, ProductionFastLaneName, fast.Name)
	core.AssertEqual(t, "rocm", fast.Backend)
	core.AssertEqual(t, "go-rocm", fast.Library)
	core.AssertEqual(t, "go-mlx", fast.ReferenceBackend)
	core.AssertEqual(t, ProductionLaneCurrentModelID, fast.ModelID)
	core.AssertEqual(t, ProductionLaneModelID, fast.LockedModelID)
	core.AssertEqual(t, officialGemma4E2BTargetModelID, fast.OfficialTargetModelID)
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, fast.AssistantModelID)
	core.AssertEqual(t, ProductionLaneArchitecture, fast.Architecture)
	core.AssertEqual(t, ProductionLaneChatTemplate, fast.ChatTemplate)
	core.AssertEqual(t, ProductionLaneProductDefaultQuantBits, fast.QuantBits)
	core.AssertEqual(t, "affine", fast.QuantMode)
	core.AssertEqual(t, 64, fast.QuantGroup)
	core.AssertEqual(t, rocmTurboQuantKVMode, fast.CacheMode)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, fast.MTPDefaultDraftTokens)
	core.AssertEqual(t, true, fast.EnabledByDefault)
	core.AssertEqual(t, false, fast.RequiresEnvGate)
	core.AssertEqual(t, false, fast.RequiresCLIFlag)
	for _, metric := range []string{"load_duration", "retained_workflow", "candidate_cache_mode", "mtp_draft_calls", "turboquant_candidate_cache_mode"} {
		if !stringSliceContains(fast.RequiredMetrics, metric) {
			t.Fatalf("fast lane RequiredMetrics = %v, missing %q", fast.RequiredMetrics, metric)
		}
	}
	if fast.Labels["production_fast_lane"] != "true" ||
		fast.Labels["production_default"] != "true" ||
		fast.Labels["production_requires_env_gate"] != "false" ||
		fast.Labels["production_requires_cli_flag"] != "false" ||
		fast.Labels["production_cache_mode"] != rocmTurboQuantKVMode ||
		fast.Labels["production_mtp_assistant_model"] != officialGemma4E2BAssistantModelID {
		t.Fatalf("fast lane labels = %+v, want production default CLI/API contract", fast.Labels)
	}
	fast.RequiredMetrics[0] = "mutated"
	fast.Labels["production_fast_lane"] = "mutated"
	next := DefaultProductionFastLane()
	if next.RequiredMetrics[0] == "mutated" || next.Labels["production_fast_lane"] == "mutated" {
		t.Fatalf("DefaultProductionFastLane aliases caller mutation: %+v", next)
	}
}

func TestProductionLane_DefaultProductionQuantizationPolicy_Good(t *testing.T) {
	policy := DefaultProductionQuantizationPolicy()

	if policy.TargetModelID != ProductionLaneCurrentModelID || policy.ArchivedBaseline != ProductionLaneCurrentConstrainedModelID {
		t.Fatalf("policy identity = %+v, want q6 target plus archived q4 baseline", policy)
	}
	if policy.DefaultBits != 6 || policy.QualityBits != 8 || policy.ConstrainedBits != 4 {
		t.Fatalf("policy bits = default:%d quality:%d constrained:%d, want 6/8/4", policy.DefaultBits, policy.QualityBits, policy.ConstrainedBits)
	}
	if policy.ActiveParameterEstimate != productionLaneActiveParameterEstimate || policy.MinimumVisibleTokensPerSec != 100 {
		t.Fatalf("policy estimates = params:%d min:%d, want retained 100 tok/s target", policy.ActiveParameterEstimate, policy.MinimumVisibleTokensPerSec)
	}
	if productionLaneBookTurnCountLabel != strconv.Itoa(ProductionLaneBookTurnCount) ||
		productionLaneBookWallSecondsLabel != strconv.Itoa(ProductionLaneBookWallSeconds) ||
		productionLaneRetainedVisibleTokensSecLabel != strconv.Itoa(productionLaneRetainedVisibleTokensSec) {
		t.Fatalf("book gate labels = turns:%q wall:%q tok:%q, want numeric production lane constants", productionLaneBookTurnCountLabel, productionLaneBookWallSecondsLabel, productionLaneRetainedVisibleTokensSecLabel)
	}
	if productionLaneGemma4E2BLayersLabel != strconv.Itoa(productionLaneGemma4E2BLayers) ||
		productionLaneGemma4E2BVocabSizeLabel != strconv.Itoa(productionLaneGemma4E2BVocabSize) ||
		productionLaneGemma4E2BHiddenSizeLabel != strconv.Itoa(productionLaneGemma4E2BHiddenSize) {
		t.Fatalf("Gemma4 E2B shape labels = layers:%q vocab:%q hidden:%q, want numeric production geometry", productionLaneGemma4E2BLayersLabel, productionLaneGemma4E2BVocabSizeLabel, productionLaneGemma4E2BHiddenSizeLabel)
	}
	if productionQuantizationLadderLabel != "bf16,q8,q6,q4" {
		t.Fatalf("quant ladder label = %q, want bf16,q8,q6,q4", productionQuantizationLadderLabel)
	}
	autoRound := DefaultProductionAutoRoundQuantizationSupport()
	if strings.Join(autoRound.Algorithms, ",") != productionAutoRoundAlgorithmsLabel ||
		strings.Join(autoRound.Formats, ",") != productionAutoRoundFormatsLabel ||
		strings.Join(autoRound.WeightSchemes, ",") != productionAutoRoundSchemesLabel ||
		strings.Join(autoRound.FloatFormats, ",") != productionAutoRoundFloatFormatsLabel ||
		intsJoinLabel(autoRound.GroupSizes) != productionAutoRoundGroupSizesLabel ||
		productionAutoRoundProfileNames(autoRound.Profiles) != productionAutoRoundProfilesLabel ||
		strings.Join(DefaultProductionAutoRoundCalibrationLabels(), ",") != productionAutoRoundCalibrationLabelsLabel ||
		strings.Join(DefaultProductionAutoRoundCalibrationDecisionLabels(), ",") != productionAutoRoundCalibrationDecisionLabelsLabel ||
		autoRound.Runtime != "planned_hip" ||
		autoRound.HIPKernel != hipKernelStatusNotLinked {
		t.Fatalf("AutoRound support = %+v, want planned ROCm AutoRound/FP quant metadata", autoRound)
	}
	calibrationLabels := DefaultProductionAutoRoundCalibrationLabels()
	calibrationLabels[0] = "mutated"
	if DefaultProductionAutoRoundCalibrationLabels()[0] == "mutated" {
		t.Fatalf("DefaultProductionAutoRoundCalibrationLabels aliases caller mutation")
	}
	decisionLabels := DefaultProductionAutoRoundCalibrationDecisionLabels()
	decisionLabels[0] = "mutated"
	if DefaultProductionAutoRoundCalibrationDecisionLabels()[0] == "mutated" {
		t.Fatalf("DefaultProductionAutoRoundCalibrationDecisionLabels aliases caller mutation")
	}
	autoRound.Algorithms[0] = "mutated"
	autoRound.GroupSizes[0] = 1
	autoRound.Profiles[0].Name = "mutated"
	nextAutoRound := DefaultProductionAutoRoundQuantizationSupport()
	if nextAutoRound.Algorithms[0] == "mutated" || nextAutoRound.GroupSizes[0] == 1 || nextAutoRound.Profiles[0].Name == "mutated" {
		t.Fatalf("DefaultProductionAutoRoundQuantizationSupport aliases caller mutation")
	}
	nvfp4, ok := ProductionAutoRoundQuantizationProfileByName("NVFP4")
	if !ok || nvfp4.Name != "w4a16-nvfp4-g128" || nvfp4.FloatFormat != "nvfp4" || nvfp4.Bits != 4 || nvfp4.GroupSize != 128 || nvfp4.NSamples != 512 || nvfp4.SeqLen != 2048 || nvfp4.Iters != 200 || !nvfp4.RequiresCalibration || !nvfp4.RequiresBench || nvfp4.HIPKernel != hipKernelStatusNotLinked {
		t.Fatalf("ProductionAutoRoundQuantizationProfileByName(NVFP4) = %+v ok=%v, want planned NVFP4 AutoRound profile", nvfp4, ok)
	}
	nvfp4Plan := DefaultProductionAutoRoundCalibrationPlan(nvfp4)
	if nvfp4Plan.ProfileName != nvfp4.Name || nvfp4Plan.NSamples != nvfp4.NSamples || nvfp4Plan.SeqLen != nvfp4.SeqLen || nvfp4Plan.Iters != nvfp4.Iters || nvfp4Plan.Runtime != "planned_hip" || nvfp4Plan.HIPKernel != hipKernelStatusNotLinked || !nvfp4Plan.RequiresCalibration || !nvfp4Plan.RequiresBench {
		t.Fatalf("DefaultProductionAutoRoundCalibrationPlan(NVFP4) = %+v, want profile defaults and planned HIP status", nvfp4Plan)
	}
	labels := map[string]string{}
	ApplyProductionAutoRoundCalibrationPlanLabels(labels, nvfp4Plan)
	if labels["autoround_calibration_profile"] != "w4a16-nvfp4-g128" ||
		labels["autoround_calibration_format"] != "native" ||
		labels["autoround_calibration_float_format"] != "nvfp4" ||
		labels["autoround_calibration_group_size"] != "128" ||
		labels["autoround_calibration_nsamples"] != "512" ||
		labels["autoround_calibration_runtime"] != "planned_hip" ||
		labels["autoround_calibration_hip_kernel"] != hipKernelStatusNotLinked ||
		labels["autoround_calibration_required"] != "true" {
		t.Fatalf("ApplyProductionAutoRoundCalibrationPlanLabels() = %+v, want NVFP4 calibration labels", labels)
	}
	if err := ValidateProductionAutoRoundCalibrationLabels(labels); err != nil {
		t.Fatalf("ValidateProductionAutoRoundCalibrationLabels() error = %v", err)
	}
	var evidence ProductionAutoRoundCalibrationEvidence
	if err := ApplyProductionAutoRoundCalibrationLabelEvidence(&evidence, labels); err != nil {
		t.Fatalf("ApplyProductionAutoRoundCalibrationLabelEvidence() error = %v", err)
	}
	decision := EvaluateProductionAutoRoundCalibrationEvidence(evidence)
	if !decision.CalibrationCandidate || decision.ProfileName != "w4a16-nvfp4-g128" || decision.FloatFormat != "nvfp4" || decision.HIPKernel != hipKernelStatusNotLinked || !decision.RequiresBench {
		t.Fatalf("EvaluateProductionAutoRoundCalibrationEvidence() = %+v evidence=%+v, want planned NVFP4 calibration candidate", decision, evidence)
	}
	decisionOutput := map[string]string{}
	ApplyProductionAutoRoundCalibrationDecisionLabels(decisionOutput, decision)
	if decisionOutput["autoround_calibration_candidate"] != "true" ||
		decisionOutput["autoround_calibration_decision_profile"] != "w4a16-nvfp4-g128" ||
		decisionOutput["autoround_calibration_decision_float_format"] != "nvfp4" ||
		decisionOutput["autoround_calibration_decision_hip_kernel"] != hipKernelStatusNotLinked ||
		decisionOutput["autoround_calibration_decision_requires_bench"] != "true" ||
		!strings.Contains(decisionOutput["autoround_calibration_decision_reason"], "ready") {
		t.Fatalf("ApplyProductionAutoRoundCalibrationDecisionLabels() = %+v, want NVFP4 decision labels", decisionOutput)
	}
	parsedDecision, err := EvaluateProductionAutoRoundCalibrationDecisionLabels(decisionOutput)
	if err != nil {
		t.Fatalf("EvaluateProductionAutoRoundCalibrationDecisionLabels() error = %v", err)
	}
	if parsedDecision != decision {
		t.Fatalf("EvaluateProductionAutoRoundCalibrationDecisionLabels() = %+v, want %+v", parsedDecision, decision)
	}
	combinedOutput := map[string]string{}
	combinedDecision, err := ApplyProductionAutoRoundCalibrationEvidenceDecisionLabels(combinedOutput, labels)
	if err != nil {
		t.Fatalf("ApplyProductionAutoRoundCalibrationEvidenceDecisionLabels() error = %v", err)
	}
	if combinedDecision != decision ||
		combinedOutput["autoround_calibration_candidate"] != "true" ||
		combinedOutput["autoround_calibration_decision_profile"] != "w4a16-nvfp4-g128" {
		t.Fatalf("ApplyProductionAutoRoundCalibrationEvidenceDecisionLabels() = decision:%+v labels:%+v, want NVFP4 decision labels", combinedDecision, combinedOutput)
	}
	if err := ValidateProductionAutoRoundCalibrationDecisionLabels(combinedOutput); err != nil {
		t.Fatalf("ValidateProductionAutoRoundCalibrationDecisionLabels() error = %v", err)
	}
	if err := ValidateProductionAutoRoundCalibrationEvidenceDecisionLabels(labels, combinedOutput); err != nil {
		t.Fatalf("ValidateProductionAutoRoundCalibrationEvidenceDecisionLabels() error = %v", err)
	}
	mismatchedDecision := cloneStringMap(combinedOutput)
	mismatchedDecision["autoround_calibration_decision_profile"] = "w4a16-mxfp4-g128"
	if err := ValidateProductionAutoRoundCalibrationEvidenceDecisionLabels(labels, mismatchedDecision); err == nil {
		t.Fatal("ValidateProductionAutoRoundCalibrationEvidenceDecisionLabels(mismatched profile) error = nil")
	}
	missingCalibration := cloneStringMap(labels)
	delete(missingCalibration, "autoround_calibration_profile")
	if err := ValidateProductionAutoRoundCalibrationLabels(missingCalibration); err == nil {
		t.Fatal("ValidateProductionAutoRoundCalibrationLabels(missing profile) error = nil")
	}
	missingDecision := cloneStringMap(combinedOutput)
	delete(missingDecision, "autoround_calibration_candidate")
	if err := ValidateProductionAutoRoundCalibrationDecisionLabels(missingDecision); err == nil {
		t.Fatal("ValidateProductionAutoRoundCalibrationDecisionLabels(missing candidate) error = nil")
	}
	decisionOutput["autoround_calibration_candidate"] = "maybe"
	if _, err := EvaluateProductionAutoRoundCalibrationDecisionLabels(decisionOutput); err == nil {
		t.Fatal("EvaluateProductionAutoRoundCalibrationDecisionLabels(bad candidate) error = nil")
	}
	evidence.HIPKernel = hipKernelStatusLinked
	decision = EvaluateProductionAutoRoundCalibrationEvidence(evidence)
	if decision.CalibrationCandidate || !strings.Contains(decision.Reason, "not_linked") {
		t.Fatalf("linked-kernel calibration decision = %+v, want fail-closed not-linked reason", decision)
	}
	if err := ApplyProductionAutoRoundCalibrationLabelEvidence(&evidence, map[string]string{"autoround_calibration_bits": "bad"}); err == nil {
		t.Fatal("ApplyProductionAutoRoundCalibrationLabelEvidence(bad bits) error = nil")
	}
	fp8, ok := ProductionAutoRoundQuantizationProfileByName("w8a16_fp8")
	if !ok || fp8.Name != "w8a16-fp8-g64" || fp8.WeightScheme != "W8A16" || fp8.FloatFormat != "fp8" || fp8.Bits != 8 || fp8.GroupSize != 64 {
		t.Fatalf("ProductionAutoRoundQuantizationProfileByName(w8a16_fp8) = %+v ok=%v, want planned FP8 AutoRound profile", fp8, ok)
	}
	autoRoundMXFP8, ok := ProductionAutoRoundQuantizationProfileByName("MXFP8")
	if !ok || autoRoundMXFP8.Name != "w8a16-mxfp8-g64" || autoRoundMXFP8.WeightScheme != "W8A16" || autoRoundMXFP8.FloatFormat != "mxfp8" || autoRoundMXFP8.Bits != 8 || autoRoundMXFP8.GroupSize != 64 || autoRoundMXFP8.ProductRole != "rocm-fp8-planning" {
		t.Fatalf("ProductionAutoRoundQuantizationProfileByName(MXFP8) = %+v ok=%v, want planned MXFP8 AutoRound profile", autoRoundMXFP8, ok)
	}
	if profile, ok := productionAutoRoundQuantizationProfileForFields("W8A16", "mxfp8", 64); !ok || profile.Name != "w8a16-mxfp8-g64" {
		t.Fatalf("productionAutoRoundQuantizationProfileForFields(W8A16, mxfp8, g64) = %+v ok=%v, want MXFP8 profile", profile, ok)
	}
	int2, ok := ProductionAutoRoundQuantizationProfileByName("W2A16_INT2")
	if !ok || int2.Name != "w2a16-int2-g128" || int2.WeightScheme != "W2A16" || int2.FloatFormat != "int2" || int2.Bits != 2 || int2.GroupSize != 128 || int2.ProductRole != "rocm-int2-planning" || int2.HIPKernel != hipKernelStatusNotLinked {
		t.Fatalf("ProductionAutoRoundQuantizationProfileByName(W2A16_INT2) = %+v ok=%v, want planned W2A16 INT2 AutoRound profile", int2, ok)
	}
	if profile, ok := productionAutoRoundQuantizationProfileForFields("W2A16", "int2", 128); !ok || profile.Name != "w2a16-int2-g128" {
		t.Fatalf("productionAutoRoundQuantizationProfileForFields(W2A16, int2, g128) = %+v ok=%v, want INT2 profile", profile, ok)
	}
	if profile, ok := ProductionAutoRoundQuantizationProfileByName("q2"); !ok || profile.Name != "w2a16-int2-g128" {
		t.Fatalf("ProductionAutoRoundQuantizationProfileByName(q2) = %+v ok=%v, want INT2 profile alias", profile, ok)
	}
	if profile, ok := productionAutoRoundQuantizationProfileForFields("W2A16", "q2", 128); !ok || profile.Name != "w2a16-int2-g128" {
		t.Fatalf("productionAutoRoundQuantizationProfileForFields(W2A16, q2, g128) = %+v ok=%v, want INT2 profile alias", profile, ok)
	}
	if _, ok := ProductionAutoRoundQuantizationProfileByName("unknown"); ok {
		t.Fatal("ProductionAutoRoundQuantizationProfileByName(unknown) ok = true")
	}
	if _, ok := productionAutoRoundQuantizationProfileForFields("W4A16", "mxfp4", 64); ok {
		t.Fatal("productionAutoRoundQuantizationProfileForFields(W4A16, mxfp4, g64) ok = true")
	}
	for _, metric := range []string{
		"load_duration",
		"peak_memory_bytes",
		"retained_restore_duration",
		"raw_decode_tokens_per_sec",
		"active_weight_read_bytes_per_token",
		"memory_bandwidth_bytes_per_sec",
		"long_output_quality_flags",
		"step_down_working_set_bytes",
		"context_length",
	} {
		if !stringSliceContains(policy.RequiredBenchmarkMetrics, metric) {
			t.Fatalf("RequiredBenchmarkMetrics = %v, missing %q", policy.RequiredBenchmarkMetrics, metric)
		}
	}
	if strings.Join(policy.RequiredBenchmarkMetrics, ",") != productionQuantizationRequiredMetricsLabel {
		t.Fatalf("required metrics label = %q, want joined policy metrics %v", productionQuantizationRequiredMetricsLabel, policy.RequiredBenchmarkMetrics)
	}
	for _, metric := range []string{
		"production_book_gate_candidate",
		"production_book_gate_reason_code",
		"production_book_gate_q6",
		"production_book_gate_turns",
		"production_book_gate_wall",
		"production_book_gate_decode",
		"production_book_gate_quality",
		"production_book_gate_raw_decode_tok/s",
		"production_book_gate_wall_s",
		"production_book_gate_quality_flags",
	} {
		if !stringSliceContains(productionBookGateMetrics, metric) {
			t.Fatalf("productionBookGateMetrics = %v, missing %q", productionBookGateMetrics, metric)
		}
	}
	if strings.Join(productionBookGateMetrics, ",") != productionBookGateMetricsLabel {
		t.Fatalf("book gate metrics label = %q, want joined metrics %v", productionBookGateMetricsLabel, productionBookGateMetrics)
	}
	for _, metric := range []string{
		"book_retained_state",
		"book_retained_state_required",
		"book_prompt_replay_fallback_forbidden",
		"book_state_source_runtime_kv",
		"book_replay_baseline",
	} {
		if !stringSliceContains(productionBookRetainedRouteMetrics, metric) {
			t.Fatalf("productionBookRetainedRouteMetrics = %v, missing %q", productionBookRetainedRouteMetrics, metric)
		}
	}
	if strings.Join(productionBookRetainedRouteMetrics, ",") != productionBookRetainedRouteMetricsLabel {
		t.Fatalf("book retained route metrics label = %q, want joined metrics %v", productionBookRetainedRouteMetricsLabel, productionBookRetainedRouteMetrics)
	}
	if productionBookRetainedRouteMetricsLabel == "" || strings.Contains(productionBookRetainedRouteMetricsLabel, "production_book_gate_") {
		t.Fatalf("book retained route metrics label = %q, want distinct retained-route metric surface", productionBookRetainedRouteMetricsLabel)
	}
	for _, label := range []string{
		"production_book_retained_artifact_candidate",
		"production_book_retained_artifact_retained_route",
		"production_book_retained_artifact_reason",
		"production_book_retained_artifact_gate_candidate",
		"production_book_retained_artifact_gate_reason_code",
		"production_book_retained_artifact_gate_q6",
		"production_book_retained_artifact_gate_turns",
		"production_book_retained_artifact_gate_wall",
		"production_book_retained_artifact_gate_decode",
		"production_book_retained_artifact_gate_quality",
		"production_book_retained_artifact_raw_decode_tok/s",
		"production_book_retained_artifact_wall_s",
		"production_book_retained_artifact_quality_flags",
	} {
		if !stringSliceContains(productionBookRetainedArtifactLabels, label) {
			t.Fatalf("productionBookRetainedArtifactLabels = %v, missing %q", productionBookRetainedArtifactLabels, label)
		}
	}
	if strings.Join(productionBookRetainedArtifactLabels, ",") != productionBookRetainedArtifactLabelsLabel {
		t.Fatalf("book retained artifact labels = %q, want joined labels %v", productionBookRetainedArtifactLabelsLabel, productionBookRetainedArtifactLabels)
	}
	bookGate := DefaultProductionBookGatePolicy()
	if bookGate.QuantBits != ProductionLaneProductDefaultQuantBits ||
		bookGate.MinimumTurns != ProductionLaneBookTurnCount ||
		bookGate.MaximumWallSeconds != ProductionLaneBookWallSeconds ||
		bookGate.MinimumRawDecodeTokensSec != float64(productionLaneRetainedVisibleTokensSec) ||
		bookGate.MaximumQualityFlags != 0 {
		t.Fatalf("book gate policy = %+v, want q6 10-turn 110s 100 tok/s zero-quality-flags gate", bookGate)
	}
	if strings.Join(bookGate.RequiredMetrics, ",") != productionBookGateMetricsLabel || bookGate.ReasonCodes != productionBookGateReasonCodesLabel {
		t.Fatalf("book gate policy labels = metrics:%v reason:%q, want advertised production labels", bookGate.RequiredMetrics, bookGate.ReasonCodes)
	}
	bookGate.RequiredMetrics[0] = "mutated"
	if DefaultProductionBookGatePolicy().RequiredMetrics[0] == "mutated" {
		t.Fatalf("DefaultProductionBookGatePolicy RequiredMetrics aliases caller mutation")
	}
	for _, code := range []string{"0=pass", "1=quant", "2=metrics", "3=turns", "4=wall", "5=decode", "6=quality"} {
		if !strings.Contains(productionBookGateReasonCodesLabel, code) {
			t.Fatalf("book gate reason code label = %q, missing %q", productionBookGateReasonCodesLabel, code)
		}
	}
	if len(policy.Tiers) != 3 {
		t.Fatalf("tiers = %+v, want quality/default/constrained", policy.Tiers)
	}
	if len(policy.SupportedPacks) != 20 {
		t.Fatalf("supported packs = %+v, want Gemma4 production matrix without benchmark-only q5", policy.SupportedPacks)
	}
	quality := policy.Tiers[0]
	if quality.Bits != 8 || quality.ModelID != ProductionLaneCurrentQualityModelID || !quality.QualityFirst || quality.StepDownToBits != 6 {
		t.Fatalf("quality tier = %+v, want q8 quality-first stepping to q6", quality)
	}
	def := policy.Tiers[1]
	if def.Bits != 6 || def.ModelID != ProductionLaneCurrentModelID || !def.ProductDefault || def.StepDownToBits != 4 {
		t.Fatalf("default tier = %+v, want q6 product default stepping to q4", def)
	}
	constrained := policy.Tiers[2]
	if constrained.Bits != 4 || constrained.ModelID != ProductionLaneCurrentConstrainedModelID || !constrained.ConstrainedOnly || !constrained.ArchivedControl {
		t.Fatalf("constrained tier = %+v, want q4 constrained archived control", constrained)
	}
	if quality.ActiveWeightReadBytesPerToken != 2300000000 || def.ActiveWeightReadBytesPerToken != 1725000000 || constrained.ActiveWeightReadBytesPerToken != 1150000000 {
		t.Fatalf("active read bytes = q8:%d q6:%d q4:%d, want active-param bit estimates", quality.ActiveWeightReadBytesPerToken, def.ActiveWeightReadBytesPerToken, constrained.ActiveWeightReadBytesPerToken)
	}
	if quality.MinimumWorkingSetBytes != 32*memoryGiB || quality.LongContextWorkingSetBytes != 64*memoryGiB ||
		def.MinimumWorkingSetBytes != 16*memoryGiB || def.LongContextWorkingSetBytes != 24*memoryGiB ||
		constrained.MinimumWorkingSetBytes != 8*memoryGiB || constrained.LongContextWorkingSetBytes != 12*memoryGiB {
		t.Fatalf("working set thresholds = q8:%d/%d q6:%d/%d q4:%d/%d, want go-mlx tier thresholds", quality.MinimumWorkingSetBytes, quality.LongContextWorkingSetBytes, def.MinimumWorkingSetBytes, def.LongContextWorkingSetBytes, constrained.MinimumWorkingSetBytes, constrained.LongContextWorkingSetBytes)
	}
	mxfp8, ok := ProductionQuantizationPackByName("MXFP8")
	if !ok || mxfp8.ModelID != "mlx-community/gemma-4-e2b-it-mxfp8" || mxfp8.Bits != 8 || mxfp8.QuantMode != "mxfp8" || !mxfp8.RequiresBench {
		t.Fatalf("ProductionQuantizationPackByName(MXFP8) = %+v ok=%v, want research MXFP8 pack", mxfp8, ok)
	}
	bf16, ok := ProductionQuantizationPackByName("mlx-community/gemma-4-e2b-it-bf16")
	if !ok || bf16.Size != "E2B" || bf16.Bits != 16 || bf16.QuantMode != "bf16" || bf16.ProductRole != "quality-control" || bf16.Runtime != Gemma4RuntimeBF16 || bf16.GenerateStatus != Gemma4GenerateLoadOnly || !bf16.RequiresNative {
		t.Fatalf("ProductionQuantizationPackByName(bf16 model) = %+v ok=%v, want native bf16 control pack", bf16, ok)
	}
	e4b := ProductionQuantizationPacksBySize("e4b")
	if len(e4b) != 6 {
		t.Fatalf("ProductionQuantizationPacksBySize(E4B) = %+v, want BF16/q8/q6/q4 plus MXFP research packs", e4b)
	}
	e4bModes := map[string]ProductionQuantizationPackSupport{}
	for _, pack := range e4b {
		e4bModes[rocmGemma4ProductionQuantPackMode(pack)] = pack
	}
	for _, mode := range []string{"bf16", "q8", "q6", "q4"} {
		pack, found := e4bModes[mode]
		if !found {
			t.Fatalf("E4B packs = %+v, missing %s target", e4b, mode)
		}
		if pack.Size != "E4B" || !pack.RunnableOnCard || !pack.RequiresBench {
			t.Fatalf("E4B %s pack = %+v, want runnable bench target", mode, pack)
		}
	}
	for _, mode := range []string{"mxfp8", "mxfp4"} {
		pack, found := e4bModes[mode]
		if !found {
			t.Fatalf("E4B packs = %+v, missing %s research pack", e4b, mode)
		}
		if pack.Size != "E4B" || pack.Runtime != Gemma4RuntimePlanned || pack.GenerateStatus != Gemma4GeneratePlannedOnly || !pack.RunnableOnCard || !pack.RequiresBench {
			t.Fatalf("E4B %s pack = %+v, want planned-only runnable research pack", mode, pack)
		}
	}
	e4b6, ok := ProductionQuantizationPackByName("mlx-community/gemma-4-e4b-it-6bit")
	if !ok || e4b6.Size != "E4B" || e4b6.Bits != 6 || e4b6.GenerateStatus != Gemma4GenerateLinked || !e4b6.RequiresBench {
		t.Fatalf("ProductionQuantizationPackByName(E4B q6) = %+v ok=%v, want runnable bench target", e4b6, ok)
	}
	lmstudioE4B6, ok := ProductionQuantizationPackByName("lmstudio-community/gemma-4-E4B-it-MLX-6bit")
	if !ok || lmstudioE4B6.Size != "E4B" || lmstudioE4B6.Bits != 6 || lmstudioE4B6.GenerateStatus != Gemma4GenerateLinked {
		t.Fatalf("ProductionQuantizationPackByName(LMStudio E4B q6) = %+v ok=%v, want E4B linked q6 alias", lmstudioE4B6, ok)
	}
	e4bMXFP4, ok := ProductionQuantizationPackByName("mlx-community/gemma-4-e4b-it-mxfp4")
	if !ok || e4bMXFP4.Size != "E4B" || e4bMXFP4.QuantMode != "mxfp4" || e4bMXFP4.GenerateStatus != Gemma4GeneratePlannedOnly || !e4bMXFP4.RequiresBench {
		t.Fatalf("ProductionQuantizationPackByName(E4B MXFP4) = %+v ok=%v, want planned research pack", e4bMXFP4, ok)
	}
	lmstudioE4BMXFP8, ok := ProductionQuantizationPackByName("lmstudio-community/gemma-4-E4B-it-MLX-mxfp8")
	if !ok || lmstudioE4BMXFP8.Size != "E4B" || lmstudioE4BMXFP8.QuantMode != "mxfp8" || lmstudioE4BMXFP8.GenerateStatus != Gemma4GeneratePlannedOnly {
		t.Fatalf("ProductionQuantizationPackByName(LMStudio E4B MXFP8) = %+v ok=%v, want E4B planned MXFP alias", lmstudioE4BMXFP8, ok)
	}
	gemma12b := ProductionQuantizationPacksBySize("12b")
	if len(gemma12b) != 2 {
		t.Fatalf("ProductionQuantizationPacksBySize(12B) = %+v, want q6/q4 linked targets", gemma12b)
	}
	gemma12bModes := map[string]ProductionQuantizationPackSupport{}
	for _, pack := range gemma12b {
		gemma12bModes[rocmGemma4ProductionQuantPackMode(pack)] = pack
	}
	for _, mode := range []string{"q6", "q4"} {
		pack, found := gemma12bModes[mode]
		if !found || pack.Size != "12B" || pack.GenerateStatus != Gemma4GenerateLinked || !pack.RunnableOnCard || !pack.RequiresBench {
			t.Fatalf("12B %s pack = %+v found=%v, want runnable linked bench target", mode, pack, found)
		}
	}
	gemma12BQ4, ok := ProductionQuantizationPackByName("mlx-community/gemma-4-12B-it-qat-4bit")
	if !ok || gemma12BQ4.Size != "12B" || gemma12BQ4.Bits != 4 || gemma12BQ4.GenerateStatus != Gemma4GenerateLinked || gemma12BQ4.SourceCollection != modelgemma4.QATCollectionID {
		t.Fatalf("ProductionQuantizationPackByName(12B QAT q4) = %+v ok=%v, want linked QAT q4 target", gemma12BQ4, ok)
	}
	if lmstudio12B, ok := ProductionQuantizationPackByName("lmstudio-community/gemma-4-12B-it-GGUF"); ok {
		t.Fatalf("ProductionQuantizationPackByName(LMStudio 12B GGUF repo) = %+v ok=%v, want fail-closed without quant filename", lmstudio12B, ok)
	}
	lmstudio12BQ6, ok := ProductionQuantizationPackByName("lmstudio-community/gemma-4-12B-it-GGUF:Q6_K")
	if !ok || lmstudio12BQ6.Size != "12B" || lmstudio12BQ6.Bits != 6 || lmstudio12BQ6.Runtime != Gemma4RuntimeGGUF || lmstudio12BQ6.GenerateStatus != Gemma4GenerateLoadOnly {
		t.Fatalf("ProductionQuantizationPackByName(LMStudio 12B GGUF Q6) = %+v ok=%v, want load-only q6 GGUF alias", lmstudio12BQ6, ok)
	}
	gemma31b := ProductionQuantizationPacksBySize("31b")
	if len(gemma31b) != 3 {
		t.Fatalf("ProductionQuantizationPacksBySize(31B) = %+v, want q8/q6/q4 status-only entries", gemma31b)
	}
	lmstudio31B4, ok := ProductionQuantizationPackByName("lmstudio-community/gemma-4-31B-it-MLX-4bit")
	if !ok || lmstudio31B4.Size != "31B" || lmstudio31B4.QuantMode != "q4-status" || lmstudio31B4.GenerateStatus != Gemma4GeneratePlannedOnly || lmstudio31B4.RunnableOnCard {
		t.Fatalf("ProductionQuantizationPackByName(LMStudio 31B q4) = %+v ok=%v, want status-only q4 alias", lmstudio31B4, ok)
	}
	if _, ok := ProductionQuantizationPackByName("unknown"); ok {
		t.Fatal("ProductionQuantizationPackByName(unknown) ok = true")
	}
}

func TestProductionLane_CurrentGemma4ProductionPacksRetainLockedProvenance_Good(t *testing.T) {
	tests := []struct {
		name        string
		modelID     string
		lockedID    string
		productRole string
	}{
		{
			name:        "quality",
			modelID:     ProductionLaneCurrentQualityModelID,
			lockedID:    "mlx-community/gemma-4-e2b-it-8bit",
			productRole: "quality",
		},
		{
			name:        "default",
			modelID:     ProductionLaneCurrentModelID,
			lockedID:    ProductionLaneModelID,
			productRole: "default",
		},
		{
			name:        "constrained",
			modelID:     ProductionLaneCurrentConstrainedModelID,
			lockedID:    ProductionLaneArchivedBaselineModelID,
			productRole: "constrained",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pack, ok := ProductionQuantizationPackByName(tt.modelID)
			if !ok || pack.ModelID != tt.modelID || pack.LockedModelID != tt.lockedID || pack.ProductRole != tt.productRole {
				t.Fatalf("ProductionQuantizationPackByName(%s) = %+v ok=%v, want locked provenance %s role %s", tt.modelID, pack, ok, tt.lockedID, tt.productRole)
			}
		})
	}
}

func TestProductionLane_SelectProductionQuantizationTier_Good(t *testing.T) {
	wide := inference.MachineDeviceInfo{MemorySize: 96 * memoryGiB, MaxRecommendedWorkingSetSize: 90 * memoryGiB}
	choice := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:        wide,
		ContextLength: ProductionLaneLongContextLength,
	})
	if choice.Tier.Bits != 6 || choice.Tier.ModelID != ProductionLaneCurrentModelID || !choice.Fits {
		t.Fatalf("default wide choice = %+v, want fitting q6", choice)
	}

	quality := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:        wide,
		ContextLength: ProductionLaneLongContextLength,
		QualityFirst:  true,
	})
	if quality.Tier.Bits != 8 || quality.Tier.ModelID != ProductionLaneCurrentQualityModelID || !quality.Fits {
		t.Fatalf("quality wide choice = %+v, want fitting q8", quality)
	}
	if quality.RequestedBits != 8 || quality.StepDownFromBits != 0 || quality.StepDownWorkingSetBytes != 0 || quality.StepDownRequiredWorkingSet != 0 {
		t.Fatalf("quality step-down evidence = %+v, want requested q8 with no step-down", quality)
	}

	qualityStepDown := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:        inference.MachineDeviceInfo{MemorySize: 64 * memoryGiB, MaxRecommendedWorkingSetSize: 48 * memoryGiB},
		ContextLength: ProductionLaneLongContextLength,
		QualityFirst:  true,
	})
	if qualityStepDown.Tier.Bits != 6 || qualityStepDown.Tier.ModelID != ProductionLaneCurrentModelID || !qualityStepDown.Fits {
		t.Fatalf("quality step-down choice = %+v, want fitting q6", qualityStepDown)
	}
	if qualityStepDown.RequestedBits != 8 || qualityStepDown.StepDownFromBits != 8 || qualityStepDown.StepDownWorkingSetBytes != 48*memoryGiB || qualityStepDown.StepDownRequiredWorkingSet != 64*memoryGiB {
		t.Fatalf("quality step-down evidence = %+v, want q8 required 64GiB stepping down at 48GiB working set", qualityStepDown)
	}

	unknownQuality := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		QualityFirst: true,
	})
	if unknownQuality.Tier.Bits != 6 || unknownQuality.RequestedBits != 8 || unknownQuality.StepDownFromBits != 8 || !unknownQuality.Fits {
		t.Fatalf("unknown-memory quality choice = %+v, want q6 default until q8 headroom is measured", unknownQuality)
	}
	if !core.Contains(unknownQuality.Reason, "measured memory headroom") {
		t.Fatalf("unknown-memory quality reason = %q, want measured-headroom explanation", unknownQuality.Reason)
	}

	constrained := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:        inference.MachineDeviceInfo{MemorySize: 16 * memoryGiB, MaxRecommendedWorkingSetSize: 13 * memoryGiB},
		ContextLength: ProductionLaneLongContextLength,
	})
	if constrained.Tier.Bits != 4 || constrained.Tier.ModelID != ProductionLaneCurrentConstrainedModelID || !constrained.Fits {
		t.Fatalf("constrained long-context choice = %+v, want fitting q4 fallback", constrained)
	}
	if constrained.RequestedBits != 6 || constrained.StepDownFromBits != 6 || constrained.StepDownWorkingSetBytes != 13*memoryGiB || constrained.StepDownRequiredWorkingSet != 24*memoryGiB {
		t.Fatalf("constrained step-down evidence = %+v, want q6 required 24GiB stepping down at 13GiB working set", constrained)
	}

	forced := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:              wide,
		ContextLength:       ProductionLaneContextLength,
		ConstrainedFallback: true,
	})
	if forced.Tier.Bits != 4 || !forced.Tier.ConstrainedOnly {
		t.Fatalf("forced constrained choice = %+v, want q4 fallback", forced)
	}
	if forced.RequestedBits != 4 || forced.StepDownFromBits != 0 {
		t.Fatalf("forced constrained evidence = %+v, want requested q4 without step-down", forced)
	}
}

func TestProductionLane_DefaultProductionArchitectureStatus_Good(t *testing.T) {
	status := DefaultProductionArchitectureStatus()
	if status.TotalArchitectures != len(rocmCapabilityArchitectures) || status.NativeArchitectures != len(rocmCapabilityArchitectures) || status.MetadataOnlyArchitectures != 0 {
		t.Fatalf("status counts = total:%d native:%d metadata:%d, want all advertised architectures native/staged", status.TotalArchitectures, status.NativeArchitectures, status.MetadataOnlyArchitectures)
	}
	for _, id := range []string{"gemma3_text", "gemma4", "gemma4_assistant", "gemma4_text", "gemma4_unified", "gemma4_unified_text", "qwen3_6", "qwen3_6_moe", "qwen3_moe", "minimax_m2", "granite", "mixtral", "deepseek", "gpt-oss", "kimi", "bert", "bert_rerank"} {
		if !stringSliceContains(status.NativeIDs, id) {
			t.Fatalf("NativeIDs = %v, missing %q", status.NativeIDs, id)
		}
	}
	if len(status.MetadataOnlyIDs) != 0 {
		t.Fatalf("MetadataOnlyIDs = %v, want empty for advertised native/staged matrix", status.MetadataOnlyIDs)
	}
	if len(status.RemainingGaps) != 0 {
		t.Fatalf("RemainingGaps = %+v, want empty for advertised native/staged matrix", status.RemainingGaps)
	}

	gap := productionArchitectureGap("qwen3_6_moe")
	if gap.Family != "qwen" || !gap.MoE || gap.MissingNative != "hybrid linear attention plus sparse expert router" || !stringSliceContains(gap.NextWork, "linear_attention_kernel") {
		t.Fatalf("qwen3_6_moe gap = %+v, want qwen MoE hybrid-attention next work", gap)
	}
	bert := productionArchitectureGap("bert_rerank")
	if bert.Family != "bert" || !bert.Rerank || bert.Generation || bert.MissingNative != "rerank scorer" {
		t.Fatalf("bert_rerank gap = %+v, want rerank scorer shape", bert)
	}
}

func TestProductionLane_DefaultQuantizationPackLocks_Good(t *testing.T) {
	locks := DefaultProductionQuantizationPackLocks()
	if len(locks) != 6 {
		t.Fatalf("DefaultProductionQuantizationPackLocks() = %d, want six Gemma4 E2B pack locks", len(locks))
	}
	byName := map[string]ProductionQuantizationPackLock{}
	byModel := map[string]ProductionQuantizationPackLock{}
	for _, lock := range locks {
		byName[lock.Name] = lock
		byModel[lock.ModelID] = lock
		if lock.BaseModelID != OfficialGemma4E2BTargetLock().ModelID || lock.BaseRevision != OfficialGemma4E2BTargetLock().Revision {
			t.Fatalf("lock base provenance = %+v, want official E2B target lock", lock)
		}
		if lock.SourceCheckedAt != officialGemma4E2BSourceCheckedAt || lock.SourceURL == "" || lock.ConversionCommand == "" || lock.AccuracySmoke == "" {
			t.Fatalf("lock source/conversion evidence incomplete: %+v", lock)
		}
		if lock.Licence != "apache-2.0" || lock.LicenceURL != "https://ai.google.dev/gemma/docs/gemma_4_license" {
			t.Fatalf("lock licence = %+v, want Gemma4 Apache-2.0 licence metadata", lock)
		}
		if lock.ConfigSHA256 == "" || lock.TokenizerSHA256 == "" || lock.TokenizerConfigSHA256 == "" || lock.SafetensorsIndexSHA256 == "" {
			t.Fatalf("lock hashes incomplete: %+v", lock)
		}
		if !lock.SafetensorsIndexPresent || len(lock.WeightFiles) == 0 {
			t.Fatalf("lock safetensors = present:%v files:%d, want indexed locked weights", lock.SafetensorsIndexPresent, len(lock.WeightFiles))
		}
	}
	for _, modelID := range []string{
		"mlx-community/gemma-4-e2b-it-mxfp4",
		"mlx-community/gemma-4-e2b-it-mxfp8",
		"mlx-community/gemma-4-e2b-it-4bit",
		"mlx-community/gemma-4-e2b-it-6bit",
		"mlx-community/gemma-4-e2b-it-8bit",
		"mlx-community/gemma-4-e2b-it-bf16",
	} {
		if _, ok := byModel[modelID]; !ok {
			t.Fatalf("DefaultProductionQuantizationPackLocks() missing %s: %+v", modelID, locks)
		}
	}

	q8 := byName["quality"]
	if q8.ModelID != "mlx-community/gemma-4-e2b-it-8bit" || q8.Revision != "48ef0737faea4e72556670e49da0ba421027a545" || len(q8.WeightFiles) != 2 {
		t.Fatalf("q8 lock = %+v, want two-shard quality pack", q8)
	}
	q6 := byName["default"]
	if q6.ModelID != ProductionLaneModelID || q6.Revision != "40d43b05f94ee798c0e40fe19fcd9ef49928486b" || len(q6.WeightFiles) != 1 || q6.WeightFiles[0].Name != "model.safetensors" {
		t.Fatalf("q6 lock = %+v, want default one-file q6 pack", q6)
	}
	q4 := byName["constrained"]
	if q4.ModelID != ProductionLaneArchivedBaselineModelID || q4.Revision != "99d9a53ff828d365a8ecae538e45f80a08d612cd" || q4.QuantGroup != 64 || q4.QuantMode != "affine" || q4.WeightFiles[0].Bytes != 3581101896 {
		t.Fatalf("q4 lock = %+v, want constrained affine g64 fallback", q4)
	}

	locks[0].Name = "mutated"
	locks[0].WeightFiles[0].Name = "mutated"
	next := DefaultProductionQuantizationPackLocks()
	if next[0].Name == "mutated" || next[0].WeightFiles[0].Name == "mutated" {
		t.Fatalf("DefaultProductionQuantizationPackLocks leaked mutable state: %+v", next[0])
	}
}

func TestProductionLane_DefaultProductionQuantizationPolicyDefensiveCopies_Good(t *testing.T) {
	policy := DefaultProductionQuantizationPolicy()
	policy.RequiredBenchmarkMetrics[0] = "mutated"
	policy.Tiers[1].Bits = 99
	policy.SupportedPacks[0].Name = "mutated"

	packs := DefaultProductionQuantizationPackSupport()
	packs[0].Name = "mutated"

	next := DefaultProductionQuantizationPolicy()
	nextPacks := DefaultProductionQuantizationPackSupport()
	if next.RequiredBenchmarkMetrics[0] == "mutated" || next.Tiers[1].Bits == 99 || next.SupportedPacks[0].Name == "mutated" || nextPacks[0].Name == "mutated" {
		t.Fatalf("DefaultProductionQuantizationPolicy leaked mutable slices: %+v", next)
	}
}

var (
	productionAutoRoundSupportSink  ProductionAutoRoundQuantizationSupport
	productionAutoRoundProfileSink  ProductionAutoRoundQuantizationProfile
	productionAutoRoundDecisionSink ProductionAutoRoundCalibrationDecision
)

func BenchmarkProductionAutoRoundQuantizationSupport_Default(b *testing.B) {
	for i := 0; i < b.N; i++ {
		productionAutoRoundSupportSink = DefaultProductionAutoRoundQuantizationSupport()
	}
}

func BenchmarkProductionAutoRoundQuantizationProfileByName_NVFP4(b *testing.B) {
	for i := 0; i < b.N; i++ {
		profile, ok := ProductionAutoRoundQuantizationProfileByName("nvfp4")
		if !ok {
			b.Fatal("missing nvfp4 profile")
		}
		productionAutoRoundProfileSink = profile
	}
}

func BenchmarkProductionAutoRoundQuantizationProfileForConfig_MXFP4(b *testing.B) {
	quant := rocmQuantizationConfigProbe{
		QuantMethod:  "auto-round-light",
		Format:       "native",
		WeightFormat: "mxfp4",
		Scheme:       "W4A16",
		Bits:         4,
		GroupSize:    128,
	}
	for i := 0; i < b.N; i++ {
		profile, ok := rocmAutoRoundProfileForQuantConfig(quant)
		if !ok {
			b.Fatal("missing mxfp4 profile")
		}
		productionAutoRoundProfileSink = profile
	}
}

func BenchmarkProductionAutoRoundCalibrationPlanForConfig_MXFP4(b *testing.B) {
	quant := rocmQuantizationConfigProbe{
		QuantMethod:  "auto-round-light",
		Format:       "native",
		WeightFormat: "mxfp4",
		Scheme:       "W4A16",
		Bits:         4,
		GroupSize:    128,
		NSamples:     640,
		SeqLen:       3072,
		Iters:        220,
	}
	for i := 0; i < b.N; i++ {
		plan, ok := rocmAutoRoundCalibrationPlanForQuantConfig(quant)
		if !ok {
			b.Fatal("missing mxfp4 calibration plan")
		}
		productionAutoRoundCalibrationPlanSink = plan
	}
}

func BenchmarkProductionAutoRoundCalibrationPlanLabels_ApplyMXFP4(b *testing.B) {
	quant := rocmQuantizationConfigProbe{
		QuantMethod:  "auto-round-light",
		Format:       "native",
		WeightFormat: "mxfp4",
		Scheme:       "W4A16",
		Bits:         4,
		GroupSize:    128,
		NSamples:     640,
		SeqLen:       3072,
		Iters:        220,
	}
	plan, ok := rocmAutoRoundCalibrationPlanForQuantConfig(quant)
	if !ok {
		b.Fatal("missing mxfp4 calibration plan")
	}
	labels := make(map[string]string, len(productionAutoRoundCalibrationLabels))
	for i := 0; i < b.N; i++ {
		clear(labels)
		ApplyProductionAutoRoundCalibrationPlanLabels(labels, plan)
	}
	if len(labels) != len(productionAutoRoundCalibrationLabels) {
		b.Fatalf("labels = %+v, want %d calibration labels", labels, len(productionAutoRoundCalibrationLabels))
	}
}

func BenchmarkProductionAutoRoundCalibrationLabelEvidence_EvaluateMXFP4(b *testing.B) {
	quant := rocmQuantizationConfigProbe{
		QuantMethod:  "auto-round-light",
		Format:       "native",
		WeightFormat: "mxfp4",
		Scheme:       "W4A16",
		Bits:         4,
		GroupSize:    128,
		NSamples:     640,
		SeqLen:       3072,
		Iters:        220,
	}
	plan, ok := rocmAutoRoundCalibrationPlanForQuantConfig(quant)
	if !ok {
		b.Fatal("missing mxfp4 calibration plan")
	}
	labels := make(map[string]string, len(productionAutoRoundCalibrationLabels))
	ApplyProductionAutoRoundCalibrationPlanLabels(labels, plan)
	for i := 0; i < b.N; i++ {
		var evidence ProductionAutoRoundCalibrationEvidence
		if err := ApplyProductionAutoRoundCalibrationLabelEvidence(&evidence, labels); err != nil {
			b.Fatal(err)
		}
		productionAutoRoundDecisionSink = EvaluateProductionAutoRoundCalibrationEvidence(evidence)
	}
}

func BenchmarkProductionAutoRoundCalibrationDecisionLabels_ApplyMXFP4(b *testing.B) {
	decision := ProductionAutoRoundCalibrationDecision{
		CalibrationCandidate: true,
		RequiresBench:        true,
		Reason:               "AutoRound calibration target ready for ROCm bench planning",
		ProfileName:          "w4a16-mxfp4-g128",
		FloatFormat:          "mxfp4",
		HIPKernel:            hipKernelStatusNotLinked,
	}
	labels := make(map[string]string, len(productionAutoRoundCalibrationDecisionLabels))
	for i := 0; i < b.N; i++ {
		clear(labels)
		ApplyProductionAutoRoundCalibrationDecisionLabels(labels, decision)
	}
	if len(labels) != len(productionAutoRoundCalibrationDecisionLabels) {
		b.Fatalf("labels = %+v, want %d decision labels", labels, len(productionAutoRoundCalibrationDecisionLabels))
	}
}

func BenchmarkProductionAutoRoundCalibrationDecisionLabels_EvaluateMXFP4(b *testing.B) {
	decision := ProductionAutoRoundCalibrationDecision{
		CalibrationCandidate: true,
		RequiresBench:        true,
		Reason:               "AutoRound calibration target ready for ROCm bench planning",
		ProfileName:          "w4a16-mxfp4-g128",
		FloatFormat:          "mxfp4",
		HIPKernel:            hipKernelStatusNotLinked,
	}
	labels := make(map[string]string, len(productionAutoRoundCalibrationDecisionLabels))
	ApplyProductionAutoRoundCalibrationDecisionLabels(labels, decision)
	for i := 0; i < b.N; i++ {
		parsed, err := EvaluateProductionAutoRoundCalibrationDecisionLabels(labels)
		if err != nil {
			b.Fatal(err)
		}
		productionAutoRoundDecisionSink = parsed
	}
}

func BenchmarkProductionAutoRoundCalibrationEvidenceDecisionLabels_ApplyMXFP4(b *testing.B) {
	quant := rocmQuantizationConfigProbe{
		QuantMethod:  "auto-round-light",
		Format:       "native",
		WeightFormat: "mxfp4",
		Scheme:       "W4A16",
		Bits:         4,
		GroupSize:    128,
		NSamples:     640,
		SeqLen:       3072,
		Iters:        220,
	}
	plan, ok := rocmAutoRoundCalibrationPlanForQuantConfig(quant)
	if !ok {
		b.Fatal("missing mxfp4 calibration plan")
	}
	evidenceLabels := make(map[string]string, len(productionAutoRoundCalibrationLabels))
	ApplyProductionAutoRoundCalibrationPlanLabels(evidenceLabels, plan)
	decisionLabels := make(map[string]string, len(productionAutoRoundCalibrationDecisionLabels))
	for i := 0; i < b.N; i++ {
		clear(decisionLabels)
		decision, err := ApplyProductionAutoRoundCalibrationEvidenceDecisionLabels(decisionLabels, evidenceLabels)
		if err != nil {
			b.Fatal(err)
		}
		productionAutoRoundDecisionSink = decision
	}
}

func BenchmarkProductionAutoRoundCalibrationLabels_ValidateMXFP4(b *testing.B) {
	quant := rocmQuantizationConfigProbe{
		QuantMethod:  "auto-round-light",
		Format:       "native",
		WeightFormat: "mxfp4",
		Scheme:       "W4A16",
		Bits:         4,
		GroupSize:    128,
		NSamples:     640,
		SeqLen:       3072,
		Iters:        220,
	}
	plan, ok := rocmAutoRoundCalibrationPlanForQuantConfig(quant)
	if !ok {
		b.Fatal("missing mxfp4 calibration plan")
	}
	labels := make(map[string]string, len(productionAutoRoundCalibrationLabels))
	ApplyProductionAutoRoundCalibrationPlanLabels(labels, plan)
	for i := 0; i < b.N; i++ {
		if err := ValidateProductionAutoRoundCalibrationLabels(labels); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProductionAutoRoundCalibrationDecisionLabels_ValidateMXFP4(b *testing.B) {
	decision := ProductionAutoRoundCalibrationDecision{
		CalibrationCandidate: true,
		RequiresBench:        true,
		Reason:               "AutoRound calibration target ready for ROCm bench planning",
		ProfileName:          "w4a16-mxfp4-g128",
		FloatFormat:          "mxfp4",
		HIPKernel:            hipKernelStatusNotLinked,
	}
	labels := make(map[string]string, len(productionAutoRoundCalibrationDecisionLabels))
	ApplyProductionAutoRoundCalibrationDecisionLabels(labels, decision)
	for i := 0; i < b.N; i++ {
		if err := ValidateProductionAutoRoundCalibrationDecisionLabels(labels); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProductionAutoRoundCalibrationEvidenceDecisionLabels_ValidateMXFP4(b *testing.B) {
	quant := rocmQuantizationConfigProbe{
		QuantMethod:  "auto-round-light",
		Format:       "native",
		WeightFormat: "mxfp4",
		Scheme:       "W4A16",
		Bits:         4,
		GroupSize:    128,
		NSamples:     640,
		SeqLen:       3072,
		Iters:        220,
	}
	plan, ok := rocmAutoRoundCalibrationPlanForQuantConfig(quant)
	if !ok {
		b.Fatal("missing mxfp4 calibration plan")
	}
	evidenceLabels := make(map[string]string, len(productionAutoRoundCalibrationLabels))
	ApplyProductionAutoRoundCalibrationPlanLabels(evidenceLabels, plan)
	decisionLabels := make(map[string]string, len(productionAutoRoundCalibrationDecisionLabels))
	if _, err := ApplyProductionAutoRoundCalibrationEvidenceDecisionLabels(decisionLabels, evidenceLabels); err != nil {
		b.Fatal(err)
	}
	for i := 0; i < b.N; i++ {
		if err := ValidateProductionAutoRoundCalibrationEvidenceDecisionLabels(evidenceLabels, decisionLabels); err != nil {
			b.Fatal(err)
		}
	}
}

func intsJoinLabel(values []int) string {
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = strconv.Itoa(value)
	}
	return strings.Join(parts, ",")
}

func productionAutoRoundProfileNames(profiles []ProductionAutoRoundQuantizationProfile) string {
	names := make([]string, len(profiles))
	for i, profile := range profiles {
		names[i] = profile.Name
	}
	return strings.Join(names, ",")
}
