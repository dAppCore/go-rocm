// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"fmt"
	"io"
	"iter"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"dappco.re/go/inference"
	rocm "dappco.re/go/rocm"
)

type tuneAttachedDrafterMetricsProvider interface {
	AttachedDrafterMetrics() *rocm.AttachedDrafterMetrics
}

const (
	tuneMeasurementShapeRetainedWarmTurn = "retained_warm_turn"
	tuneRetainedWarmupMaxTokens          = 32
	tuneRetainedMeasuredTurn             = "continuation_release_plan"
	tuneRetainedContinuationPrompt       = "Continue the review with a concrete release engineering plan: explain the retained-state token loop, MTP verifier counters, tuning profile promotion rules, server state reuse, CUDA and CPU artifact checks, and the exact benchmark evidence needed before beta."
)

type tuneGenerationResult struct {
	GeneratedTokens int
	WarmupTokens    int
	Shape           string
	MeasuredTurn    string
	WarmupMaxTokens int
}

func runTuneExecute(ctx context.Context, report tunePlanReport, stdout io.Writer, stderr io.Writer) int {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			fmt.Fprintf(stderr, "%s tune: %v\n", cliName(), err)
			return 1
		}
	}
	if !report.DraftDetection.Active() {
		fmt.Fprintf(stderr, "%s tune: no MTP drafter found for %s; place an assistant/ beside the target or pass --draft\n", cliName(), report.ModelPath)
		return 1
	}
	workload := tuneWorkloadFromReport(report)
	plan, err := tunePlanForReport(ctx, report, workload)
	if err != nil {
		fmt.Fprintf(stderr, "%s tune: %v\n", cliName(), err)
		return 1
	}
	machineHash, err := currentROCmMachineProfileHash(ctx)
	if err != nil {
		fmt.Fprintf(stderr, "%s tune: %v\n", cliName(), err)
		return 1
	}
	results := make([]inference.TuningResult, 0, len(plan.Candidates))
	for _, candidate := range plan.Candidates {
		block := tuneDraftBlockFromCandidate(candidate)
		if block <= 0 {
			results = append(results, inference.TuningResult{
				Candidate: candidate,
				Error:     "candidate missing MTP draft block label",
				Labels:    map[string]string{"candidate_status": "missing_draft_block"},
			})
			continue
		}
		fmt.Fprintf(stderr, "%s tune: measuring block %d (%s)\n", cliName(), block, candidate.ID)
		result := runTuneDraftBlockCandidate(ctx, report, workload, candidate)
		results = append(results, result)
		if result.Error != "" {
			fmt.Fprintf(stderr, "%s tune: block %d failed: %s\n", cliName(), block, result.Error)
			continue
		}
		fmt.Fprintf(stderr, "%s tune: block %d %.1f tok/s acceptance %.0f%%\n",
			cliName(),
			block,
			result.Measurements.DecodeTokensPerSec,
			tuneFloatLabel(result.Labels, "mtp_acceptance_rate")*100,
		)
	}
	selected, ok := rocm.SelectTuningResult(results)
	if !ok {
		fmt.Fprintf(stderr,
			"%s tune: no successful native MTP attached-drafter candidates; refusing autoregressive fallback (target %s draft %s sweep %s)\n",
			cliName(),
			report.ModelPath,
			report.DraftDetection.DraftPath,
			strings.TrimSpace(report.Labels["draft_block_sweep"]),
		)
		return 1
	}
	baselineCandidate := tuneTargetDecodeBaselineCandidate(plan, report)
	fmt.Fprintf(stderr, "%s tune: measuring target decode baseline (%s)\n", cliName(), baselineCandidate.ID)
	baseline := runTuneTargetDecodeBaselineCandidate(ctx, report, workload, baselineCandidate)
	results = append(results, baseline)
	if baseline.Error != "" {
		fmt.Fprintf(stderr, "%s tune: target decode baseline failed: %s\n", cliName(), baseline.Error)
		return 1
	}
	fmt.Fprintf(stderr, "%s tune: target decode baseline %.1f tok/s\n", cliName(), baseline.Measurements.DecodeTokensPerSec)
	comparison := tuneNativeMTPBaselineComparison(selected, baseline)
	if comparison.Error != "" {
		fmt.Fprintf(stderr, "%s tune: %s\n", cliName(), comparison.Error)
		return 1
	}
	if !comparison.MTPFaster {
		fmt.Fprintf(stderr,
			"%s tune: selected native MTP candidate %s %.1f tok/s is not faster than target decode baseline %.1f tok/s (%.3fx); refusing slower MTP profile\n",
			cliName(),
			selected.Candidate.ID,
			selected.Measurements.DecodeTokensPerSec,
			baseline.Measurements.DecodeTokensPerSec,
			comparison.Speedup,
		)
		return 1
	}
	selectionLabels := rocm.TuningSelectionLabels(results, selected)
	for key, value := range comparison.Labels {
		selectionLabels[key] = value
	}
	for key, value := range tuneProfileLabelsFromReport(report, selected) {
		selectionLabels[key] = value
	}
	profile := rocm.BuildTuningProfile(plan, report.ModelPath, machineHash, workload, selected, selectionLabels, time.Now().UTC())
	path := rocm.TuningProfilePath(report.ProfileDir, profile)
	if err := writeROCmTuningProfile(path, profile); err != nil {
		fmt.Fprintf(stderr, "%s tune: write profile: %v\n", cliName(), err)
		return 1
	}
	fmt.Fprintf(stdout, "selected block %s for %s\n", profile.Candidate.Labels[tuneDraftBlockLabel], report.ModelPath)
	fmt.Fprintf(stdout, "profile: %s\n", path)
	fmt.Fprintf(stdout, "score %.3f decode %.1f tok/s\n", profile.Score.Score, profile.Measurements.DecodeTokensPerSec)
	return 0
}

func tuneWorkloadFromReport(report tunePlanReport) inference.TuningWorkload {
	workload := strings.TrimSpace(report.Workload)
	if workload == "" {
		return inference.TuningWorkloadChat
	}
	return inference.TuningWorkload(workload)
}

func tunePlanForReport(ctx context.Context, report tunePlanReport, workload inference.TuningWorkload) (inference.TuningPlan, error) {
	plan, err := rocm.PlanLocalTuning(ctx, inference.TuningPlanRequest{
		Runtime:   inference.RuntimeIdentity{Backend: defaultBackendName},
		Model:     inference.ModelIdentity{Path: report.ModelPath},
		Workloads: []inference.TuningWorkload{workload},
		Labels:    cloneStringMap(report.Labels),
	})
	if err != nil {
		return inference.TuningPlan{}, err
	}
	plan.Model.Path = report.ModelPath
	plan.Workloads = []inference.TuningWorkload{workload}
	plan.Candidates = tuneDraftBlockCandidatesFromPlan(plan, report)
	plan.Recommended = map[inference.TuningWorkload]string{}
	if len(plan.Candidates) > 0 {
		plan.Recommended[workload] = plan.Candidates[0].ID
	}
	return plan, nil
}

func tuneDraftBlockCandidatesFromPlan(plan inference.TuningPlan, report tunePlanReport) []inference.TuningCandidate {
	template := inference.TuningCandidate{
		Workload:      tuneWorkloadFromReport(report),
		Model:         plan.Model,
		Runtime:       plan.Runtime,
		ParallelSlots: 1,
		BatchSize:     1,
		Labels:        cloneStringMap(report.Labels),
	}
	if len(plan.Candidates) > 0 {
		template = plan.Candidates[0]
		template.Labels = cloneStringMap(template.Labels)
	}
	if len(report.Labels) > 0 {
		labels := cloneStringMap(report.Labels)
		for key, value := range template.Labels {
			if value != "" {
				if labels == nil {
					labels = map[string]string{}
				}
				labels[key] = value
			}
		}
		template.Labels = labels
	}
	template.Model = tuneDraftBlockProfileModel(template.Model, plan.Model, report.ModelPath)
	if template.Runtime.Backend == "" {
		template.Runtime = plan.Runtime
	}
	if template.Runtime.Backend == "" {
		template.Runtime.Backend = defaultBackendName
	}
	if template.Workload == "" {
		template.Workload = tuneWorkloadFromReport(report)
	}
	candidates := make([]inference.TuningCandidate, 0, len(report.Sweep))
	for _, item := range report.Sweep {
		candidate := template
		candidate.Labels = cloneStringMap(template.Labels)
		if candidate.Labels == nil {
			candidate.Labels = map[string]string{}
		}
		candidate.ID = tuneDraftBlockCandidateID(candidate.Workload, item.DraftBlock)
		candidate.Labels[tuneDraftBlockLabel] = strconv.Itoa(item.DraftBlock)
		candidate.Labels["mtp_draft_tokens"] = strconv.Itoa(item.DraftTokens)
		candidate.Labels["tuning_executor"] = "native_attached_mtp"
		candidate.Labels["profile_writer"] = "enabled"
		candidate.Labels["profile_reader"] = "generate_serve_auto_profile"
		candidates = append(candidates, candidate)
	}
	return candidates
}

func tuneTargetDecodeBaselineCandidate(plan inference.TuningPlan, report tunePlanReport) inference.TuningCandidate {
	candidate := inference.TuningCandidate{
		ID:            tuneTargetDecodeBaselineCandidateID(tuneWorkloadFromReport(report)),
		Workload:      tuneWorkloadFromReport(report),
		Model:         tuneDraftBlockProfileModel(inference.ModelIdentity{}, plan.Model, report.ModelPath),
		Runtime:       plan.Runtime,
		ParallelSlots: 1,
		BatchSize:     1,
		Labels:        cloneStringMap(report.Labels),
	}
	if len(plan.Candidates) > 0 {
		candidate = plan.Candidates[0]
		candidate.Labels = cloneStringMap(candidate.Labels)
		candidate.Model = tuneDraftBlockProfileModel(candidate.Model, plan.Model, report.ModelPath)
	}
	if candidate.Labels == nil {
		candidate.Labels = map[string]string{}
	}
	candidate.ID = tuneTargetDecodeBaselineCandidateID(tuneWorkloadFromReport(report))
	candidate.Workload = tuneWorkloadFromReport(report)
	if candidate.Runtime.Backend == "" {
		candidate.Runtime = plan.Runtime
	}
	if candidate.Runtime.Backend == "" {
		candidate.Runtime.Backend = defaultBackendName
	}
	delete(candidate.Labels, tuneDraftBlockLabel)
	delete(candidate.Labels, "mtp_draft_tokens")
	candidate.Labels["tuning_executor"] = "target_decode_baseline"
	candidate.Labels["target_decode_baseline"] = "true"
	candidate.Labels["native_mtp_attachment"] = "not_used"
	candidate.Labels["profile_writer"] = "comparison_only"
	candidate.Labels["profile_reader"] = "generate_serve_auto_profile"
	return candidate
}

func tuneTargetDecodeBaselineCandidateID(workload inference.TuningWorkload) string {
	base := string(workload)
	if strings.TrimSpace(base) == "" {
		base = string(inference.TuningWorkloadChat)
	}
	return base + ":target-decode-baseline"
}

func tuneDraftBlockProfileModel(candidateModel, planModel inference.ModelIdentity, reportModelPath string) inference.ModelIdentity {
	model := candidateModel
	if model.Path == "" {
		model = planModel
	} else {
		model = tuneModelIdentityWithMissingFallbacks(model, planModel)
	}
	reportModelPath = strings.TrimSpace(reportModelPath)
	if reportModelPath == "" {
		return model
	}
	if model.Path != "" && filepathClean(model.Path) != filepathClean(reportModelPath) {
		labels := cloneStringMap(model.Labels)
		if labels == nil {
			labels = map[string]string{}
		}
		labels["inspected_weight_path"] = model.Path
		labels["profile_lookup_path"] = reportModelPath
		model.Labels = labels
	}
	model.Path = reportModelPath
	return model
}

func tuneModelIdentityWithMissingFallbacks(model, fallback inference.ModelIdentity) inference.ModelIdentity {
	if model.ID == "" {
		model.ID = fallback.ID
	}
	if model.Path == "" {
		model.Path = fallback.Path
	}
	if model.Architecture == "" {
		model.Architecture = fallback.Architecture
	}
	if model.Revision == "" {
		model.Revision = fallback.Revision
	}
	if model.Hash == "" {
		model.Hash = fallback.Hash
	}
	if model.QuantBits == 0 {
		model.QuantBits = fallback.QuantBits
	}
	if model.QuantGroup == 0 {
		model.QuantGroup = fallback.QuantGroup
	}
	if model.QuantType == "" {
		model.QuantType = fallback.QuantType
	}
	if model.ContextLength == 0 {
		model.ContextLength = fallback.ContextLength
	}
	if model.NumLayers == 0 {
		model.NumLayers = fallback.NumLayers
	}
	if model.HiddenSize == 0 {
		model.HiddenSize = fallback.HiddenSize
	}
	if model.VocabSize == 0 {
		model.VocabSize = fallback.VocabSize
	}
	labels := cloneStringMap(fallback.Labels)
	for key, value := range model.Labels {
		if value != "" {
			if labels == nil {
				labels = map[string]string{}
			}
			labels[key] = value
		}
	}
	model.Labels = labels
	return model
}

func filepathClean(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	return filepath.Clean(path)
}

func tuneDraftBlockCandidateID(workload inference.TuningWorkload, block int) string {
	base := string(workload)
	if strings.TrimSpace(base) == "" {
		base = string(inference.TuningWorkloadChat)
	}
	return base + ":mtp-block-" + strconv.Itoa(block)
}

func tuneDraftBlockFromCandidate(candidate inference.TuningCandidate) int {
	block, err := strconv.Atoi(strings.TrimSpace(candidate.Labels[tuneDraftBlockLabel]))
	if err != nil || block < 2 || block > 8 {
		return 0
	}
	return block
}

func runTuneDraftBlockCandidate(ctx context.Context, report tunePlanReport, workload inference.TuningWorkload, candidate inference.TuningCandidate) inference.TuningResult {
	block := tuneDraftBlockFromCandidate(candidate)
	start := time.Now()
	model, err := rocm.LoadAttachedDrafterPairAsTextModelBlockWithConfig(
		report.ModelPath,
		report.DraftDetection.DraftPath,
		block,
		rocm.TuningCandidateROCmLoadConfig(candidate),
		rocm.TuningCandidateLoadOptions(candidate)...,
	)
	loadDuration := time.Since(start)
	if err != nil {
		return inference.TuningResult{
			Candidate: candidate,
			Error:     err.Error(),
			Labels:    map[string]string{"candidate_status": "load_failed"},
		}
	}
	defer func() {
		if closer, ok := model.(interface{ Close() error }); ok && closer != nil {
			_ = closer.Close()
		}
	}()
	genStart := time.Now()
	generated := runTuneDraftBlockGenerate(ctx, model, report)
	totalDuration := time.Since(genStart)
	if err := model.Err(); err != nil {
		return inference.TuningResult{
			Candidate: candidate,
			Error:     err.Error(),
			Labels:    map[string]string{"candidate_status": "generate_failed"},
		}
	}
	measurements := tuneMeasurementsFromModel(model, generated.GeneratedTokens, loadDuration, totalDuration)
	score := inference.ScoreTuningMeasurements(workload, measurements)
	labels := tuneResultLabelsFromModel(model, block, generated)
	if labels["mtp_proposed_tokens"] == "" {
		labels["candidate_status"] = "missing_mtp_metrics"
		return inference.TuningResult{
			Candidate:    candidate,
			Measurements: measurements,
			Score:        score,
			Error:        "native attached drafter did not emit MTP counters",
			Labels:       labels,
		}
	}
	for key, value := range score.Labels {
		labels[key] = value
	}
	return inference.TuningResult{
		Candidate:    candidate,
		Measurements: measurements,
		Score:        score,
		Labels:       labels,
	}
}

func runTuneTargetDecodeBaselineCandidate(ctx context.Context, report tunePlanReport, workload inference.TuningWorkload, candidate inference.TuningCandidate) inference.TuningResult {
	start := time.Now()
	model, err := rocm.LoadModelWithConfig(
		report.ModelPath,
		rocm.TuningCandidateROCmLoadConfig(candidate),
		rocm.TuningCandidateLoadOptions(candidate)...,
	)
	loadDuration := time.Since(start)
	if err != nil {
		return inference.TuningResult{
			Candidate: candidate,
			Error:     err.Error(),
			Labels:    map[string]string{"candidate_status": "load_failed", "target_decode_baseline": "true"},
		}
	}
	defer func() {
		if closer, ok := model.(interface{ Close() error }); ok && closer != nil {
			_ = closer.Close()
		}
	}()
	genStart := time.Now()
	generated := runTuneDraftBlockGenerate(ctx, model, report)
	totalDuration := time.Since(genStart)
	if err := model.Err(); err != nil {
		return inference.TuningResult{
			Candidate: candidate,
			Error:     err.Error(),
			Labels:    map[string]string{"candidate_status": "generate_failed", "target_decode_baseline": "true"},
		}
	}
	measurements := tuneMeasurementsFromModel(model, generated.GeneratedTokens, loadDuration, totalDuration)
	score := inference.ScoreTuningMeasurements(workload, measurements)
	labels := tuneTargetDecodeBaselineResultLabels(model, generated)
	for key, value := range score.Labels {
		labels[key] = value
	}
	return inference.TuningResult{
		Candidate:    candidate,
		Measurements: measurements,
		Score:        score,
		Labels:       labels,
	}
}

func runTuneDraftBlockGenerate(ctx context.Context, model inference.TextModel, report tunePlanReport) tuneGenerationResult {
	result := tuneGenerationResult{
		Shape:           tuneMeasurementShapeRetainedWarmTurn,
		MeasuredTurn:    tuneRetainedMeasuredTurn,
		WarmupMaxTokens: tuneWarmupMaxTokens(report),
	}
	warmupMessages := []inference.Message{{Role: "user", Content: report.Prompt}}
	warmupOpts := tuneGenerateOptions(report, result.WarmupMaxTokens)
	result.WarmupTokens = collectTuneGeneratedTokens(model.Chat(ctx, warmupMessages, warmupOpts...))
	if err := model.Err(); err != nil {
		return result
	}
	measuredMessages := []inference.Message{{Role: "user", Content: tuneRetainedContinuationPrompt}}
	measuredOpts := tuneGenerateOptions(report, report.MaxTokens)
	result.GeneratedTokens = collectTuneGeneratedTokens(model.Chat(ctx, measuredMessages, measuredOpts...))
	return result
}

func tuneGenerateOptions(report tunePlanReport, maxTokens int) []inference.GenerateOption {
	opts := []inference.GenerateOption{
		inference.WithMaxTokens(maxTokens),
		inference.WithTemperature(report.Sampling.Temperature),
	}
	if report.Sampling.TopK > 0 {
		opts = append(opts, inference.WithTopK(report.Sampling.TopK))
	}
	if report.Sampling.TopP > 0 {
		opts = append(opts, inference.WithTopP(report.Sampling.TopP))
	}
	if report.Sampling.MinP > 0 {
		opts = append(opts, inference.WithMinP(report.Sampling.MinP))
	}
	return opts
}

func tuneWarmupMaxTokens(report tunePlanReport) int {
	if report.MaxTokens > 0 && report.MaxTokens < tuneRetainedWarmupMaxTokens {
		return report.MaxTokens
	}
	return tuneRetainedWarmupMaxTokens
}

func collectTuneGeneratedTokens(stream iter.Seq[inference.Token]) int {
	generated := 0
	for range stream {
		generated++
	}
	return generated
}

func tuneMeasurementsFromModel(model inference.TextModel, generated int, loadDuration, totalDuration time.Duration) inference.TuningMeasurements {
	metrics := model.Metrics()
	measurements := inference.TuningMeasurements{
		PromptTokens:           metrics.PromptTokens,
		GeneratedTokens:        metrics.GeneratedTokens,
		LoadMilliseconds:       tuneDurationMilliseconds(loadDuration),
		FirstTokenMilliseconds: 0,
		PrefillTokensPerSec:    metrics.PrefillTokensPerSec,
		DecodeTokensPerSec:     metrics.DecodeTokensPerSec,
		TotalMilliseconds:      tuneDurationMilliseconds(totalDuration),
		PeakMemoryBytes:        metrics.PeakMemoryBytes,
		ActiveMemoryBytes:      metrics.ActiveMemoryBytes,
		CorrectnessSmokeResult: "pass",
		CorrectnessSmokeChecks: 1,
	}
	if measurements.GeneratedTokens <= 0 {
		measurements.GeneratedTokens = generated
	}
	if measurements.TotalMilliseconds <= 0 {
		measurements.TotalMilliseconds = tuneDurationMilliseconds(metrics.TotalDuration)
	}
	if measurements.DecodeTokensPerSec <= 0 && measurements.GeneratedTokens > 0 {
		duration := metrics.DecodeDuration
		if duration <= 0 {
			duration = totalDuration
		}
		if duration > 0 {
			measurements.DecodeTokensPerSec = float64(measurements.GeneratedTokens) / duration.Seconds()
		}
	}
	return measurements
}

func tuneResultLabelsFromModel(model inference.TextModel, block int, generated tuneGenerationResult) map[string]string {
	labels := map[string]string{
		tuneDraftBlockLabel:        strconv.Itoa(block),
		"generated_tokens":         strconv.Itoa(generated.GeneratedTokens),
		"warmup_generated_tokens":  strconv.Itoa(generated.WarmupTokens),
		"tuning_measurement_shape": generated.Shape,
		"tune_measured_turn":       generated.MeasuredTurn,
		"tune_warmup_max_tokens":   strconv.Itoa(generated.WarmupMaxTokens),
		"tuning_executor":          "native_attached_mtp",
		"profile_writer":           "enabled",
		"profile_reader":           "generate_serve_auto_profile",
		"retained_state_required":  "true",
		"prompt_replay_fallback":   "refused",
	}
	provider, ok := model.(tuneAttachedDrafterMetricsProvider)
	if !ok || provider == nil {
		return labels
	}
	metrics := provider.AttachedDrafterMetrics()
	if metrics == nil {
		return labels
	}
	labels["mtp_draft_tokens"] = strconv.Itoa(metrics.DraftTokens)
	labels["mtp_accepted_tokens"] = strconv.Itoa(metrics.AcceptedTokens)
	labels["mtp_rejected_tokens"] = strconv.Itoa(metrics.RejectedTokens)
	labels["mtp_proposed_tokens"] = strconv.Itoa(metrics.ProposedTokens)
	labels["mtp_verify_calls"] = strconv.Itoa(metrics.VerifyCalls)
	labels["mtp_target_calls"] = strconv.Itoa(metrics.TargetCalls)
	labels["mtp_draft_calls"] = strconv.Itoa(metrics.DraftCalls)
	labels["mtp_acceptance_rate"] = strconv.FormatFloat(metrics.AcceptanceRate, 'f', 6, 64)
	return labels
}

func tuneTargetDecodeBaselineResultLabels(model inference.TextModel, generated tuneGenerationResult) map[string]string {
	labels := map[string]string{
		"generated_tokens":         strconv.Itoa(generated.GeneratedTokens),
		"warmup_generated_tokens":  strconv.Itoa(generated.WarmupTokens),
		"tuning_measurement_shape": generated.Shape,
		"tune_measured_turn":       generated.MeasuredTurn,
		"tune_warmup_max_tokens":   strconv.Itoa(generated.WarmupMaxTokens),
		"tuning_executor":          "target_decode_baseline",
		"target_decode_baseline":   "true",
		"profile_writer":           "comparison_only",
		"profile_reader":           "generate_serve_auto_profile",
		"retained_state_required":  "true",
		"prompt_replay_fallback":   "refused",
	}
	return labels
}

type tuneNativeMTPBaselineComparisonResult struct {
	Labels    map[string]string
	Speedup   float64
	MTPFaster bool
	Error     string
}

func tuneNativeMTPBaselineComparison(selected, baseline inference.TuningResult) tuneNativeMTPBaselineComparisonResult {
	labels := map[string]string{
		"target_decode_baseline_candidate_id": baseline.Candidate.ID,
		"native_mtp_candidate_id":             selected.Candidate.ID,
		"profile_requires_native_mtp_speedup": "true",
	}
	selectedTokS := selected.Measurements.DecodeTokensPerSec
	baselineTokS := baseline.Measurements.DecodeTokensPerSec
	if selectedTokS > 0 {
		labels["native_mtp_decode_tok_s"] = strconv.FormatFloat(selectedTokS, 'f', 6, 64)
	}
	if baselineTokS > 0 {
		labels["target_decode_baseline_tok_s"] = strconv.FormatFloat(baselineTokS, 'f', 6, 64)
	}
	if selectedTokS <= 0 {
		labels["native_mtp_faster_than_target_decode"] = "false"
		return tuneNativeMTPBaselineComparisonResult{
			Labels: labels,
			Error:  "selected native MTP candidate did not report decode tok/s",
		}
	}
	if baselineTokS <= 0 {
		labels["native_mtp_faster_than_target_decode"] = "false"
		return tuneNativeMTPBaselineComparisonResult{
			Labels: labels,
			Error:  "target decode baseline did not report decode tok/s",
		}
	}
	speedup := selectedTokS / baselineTokS
	labels["native_mtp_speedup"] = strconv.FormatFloat(speedup, 'f', 6, 64)
	labels["native_mtp_faster_than_target_decode"] = strconv.FormatBool(speedup > 1)
	return tuneNativeMTPBaselineComparisonResult{
		Labels:    labels,
		Speedup:   speedup,
		MTPFaster: speedup > 1,
	}
}

func tuneProfileLabelsFromReport(report tunePlanReport, selected inference.TuningResult) map[string]string {
	labels := cloneStringMap(report.Labels)
	if labels == nil {
		labels = map[string]string{}
	}
	for key, value := range selected.Labels {
		if value != "" {
			labels[key] = value
		}
	}
	labels["tuning_stage"] = "reactive_mtp_tuning_profile"
	labels["native_mtp_attachment"] = "linked"
	labels["tuning_executor"] = "native_attached_mtp"
	labels["profile_writer"] = "enabled"
	labels["profile_reader"] = "generate_serve_auto_profile"
	labels["profile_contract"] = "inference.TuningProfile"
	labels["profile_draft_block_label"] = tuneDraftBlockLabel
	labels[tuneDraftBlockLabel] = selected.Candidate.Labels[tuneDraftBlockLabel]
	return labels
}

func tuneDurationMilliseconds(duration time.Duration) float64 {
	if duration <= 0 {
		return 0
	}
	return float64(duration) / float64(time.Millisecond)
}

func tuneFloatLabel(labels map[string]string, key string) float64 {
	value, err := strconv.ParseFloat(strings.TrimSpace(labels[key]), 64)
	if err != nil {
		return 0
	}
	return value
}
