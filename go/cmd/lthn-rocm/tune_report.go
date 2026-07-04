// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"

	rocm "dappco.re/go/rocm"
)

const defaultTuneDepths = "2,3,4,5,6"

type tuneCommandOptions struct {
	ModelPath   string
	DraftPath   string
	Depths      string
	MaxTokens   int
	Prompt      string
	Workload    string
	ProfileDir  string
	Temperature float32
	TopP        float32
	TopK        int
	MinP        float32
}

type tunePlanReport struct {
	Version        int                          `json:"version"`
	Kind           string                       `json:"kind"`
	Backend        string                       `json:"backend"`
	Command        string                       `json:"command"`
	CLIContract    string                       `json:"cli_contract"`
	NoPython       bool                         `json:"no_python"`
	ModelPath      string                       `json:"model_path"`
	DraftFlag      string                       `json:"draft_flag,omitempty"`
	DraftDetection rocm.DraftDetection          `json:"draft_detection,omitempty"`
	Workload       string                       `json:"workload"`
	Prompt         string                       `json:"prompt"`
	ProfileDir     string                       `json:"profile_dir,omitempty"`
	MaxTokens      int                          `json:"max_tokens"`
	Sampling       tuneSamplingPlan             `json:"sampling"`
	Policy         rocm.ProductionMTPPolicy     `json:"policy"`
	OfficialLocks  []rocm.OfficialGemma4E2BLock `json:"official_locks,omitempty"`
	Sweep          []tuneSweepPlan              `json:"sweep"`
	Labels         map[string]string            `json:"labels,omitempty"`
	Notes          []string                     `json:"notes,omitempty"`
}

type tuneSamplingPlan struct {
	Temperature float32 `json:"temperature"`
	TopP        float32 `json:"top_p,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	MinP        float32 `json:"min_p,omitempty"`
	Profile     string  `json:"profile"`
}

type tuneSweepPlan struct {
	DraftBlock  int    `json:"draft_block"`
	DraftTokens int    `json:"draft_tokens"`
	MaxTokens   int    `json:"max_tokens"`
	Workload    string `json:"workload"`
}

func tunePlanReportFromOptions(opts tuneCommandOptions) (tunePlanReport, error) {
	if strings.TrimSpace(opts.ModelPath) == "" {
		return tunePlanReport{}, fmt.Errorf("model path is required")
	}
	if opts.MaxTokens <= 0 {
		return tunePlanReport{}, fmt.Errorf("max tokens must be > 0")
	}
	if err := validateTuneSamplingOptions(opts); err != nil {
		return tunePlanReport{}, err
	}
	depths, err := parseTuneDepths(opts.Depths)
	if err != nil {
		return tunePlanReport{}, err
	}
	workload := strings.TrimSpace(opts.Workload)
	if workload == "" {
		workload = "chat"
	}
	prompt := strings.TrimSpace(opts.Prompt)
	if prompt == "" {
		prompt = defaultProductionMeasurementPrompt
	}
	profileDir := strings.TrimSpace(opts.ProfileDir)
	if profileDir == "" {
		profileDir = standardTuningProfileDir()
	}
	detection := resolveROCmDraft(opts.ModelPath, opts.DraftPath, true)
	labels := tunePlanLabels(opts, depths, detection)
	report := tunePlanReport{
		Version:        1,
		Kind:           "reactive-mtp-tuning-plan",
		Backend:        defaultBackendName,
		Command:        "tune",
		CLIContract:    cliContractName,
		NoPython:       true,
		ModelPath:      strings.TrimSpace(opts.ModelPath),
		DraftFlag:      strings.TrimSpace(opts.DraftPath),
		DraftDetection: detection,
		Workload:       workload,
		Prompt:         prompt,
		ProfileDir:     profileDir,
		MaxTokens:      opts.MaxTokens,
		Sampling:       tuneSamplingPlanFromOptions(opts),
		Policy:         rocm.DefaultProductionMTPPolicy(),
		OfficialLocks:  rocm.DefaultOfficialGemma4E2BLocks(),
		Sweep:          tuneSweepPlanFromDepths(depths, opts.MaxTokens, workload),
		Labels:         labels,
		Notes: []string{
			"ROCm has the production MTP promotion policy, official Gemma 4 target/assistant lock metadata, and reactive drafter detection linked into this binary.",
			"The tune command measures the native attached-drafter draft-block sweep when the runtime can load the target/assistant pair; it refuses autoregressive fallback when the native route is unavailable.",
			"Successful native MTP sweeps are compared against a target decode baseline before a profile is written, so slower linked drafters do not become the production fast lane.",
			"Profiles use the shared inference.TuningProfile schema; lthn-rocm generate and serve read the same tuned draft-block label written by go-mlx tune.",
		},
	}
	return report, nil
}

func validateTuneSamplingOptions(opts tuneCommandOptions) error {
	if opts.TopK < 0 {
		return fmt.Errorf("top-k must be >= 0")
	}
	for name, value := range map[string]float32{
		"temperature": opts.Temperature,
		"top-p":       opts.TopP,
		"min-p":       opts.MinP,
	} {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("%s must be finite", name)
		}
	}
	if opts.Temperature < 0 {
		return fmt.Errorf("temperature must be >= 0")
	}
	if opts.TopP < 0 || opts.TopP > 1 {
		return fmt.Errorf("top-p must be in [0, 1]")
	}
	if opts.MinP < 0 || opts.MinP > 1 {
		return fmt.Errorf("min-p must be in [0, 1]")
	}
	return nil
}

func tuneSamplingPlanFromOptions(opts tuneCommandOptions) tuneSamplingPlan {
	return tuneSamplingPlan{
		Temperature: opts.Temperature,
		TopP:        opts.TopP,
		TopK:        opts.TopK,
		MinP:        opts.MinP,
		Profile:     tuneSamplingProfile(opts),
	}
}

func tuneSamplingProfile(opts tuneCommandOptions) string {
	if opts.Temperature == 0 && opts.TopP == 0 && opts.TopK == 0 && opts.MinP == 0 {
		return "greedy"
	}
	if opts.Temperature == 1 && opts.TopP == 0.95 && opts.TopK == 64 && opts.MinP == 0 {
		return "llama_server_mtp"
	}
	return "sampled"
}

func parseTuneDepths(value string) ([]int, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		value = defaultTuneDepths
	}
	parts := strings.Split(value, ",")
	depths := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		depth, err := strconv.Atoi(part)
		if err != nil || depth <= 0 {
			return nil, fmt.Errorf("invalid draft block %q", part)
		}
		if depth < 2 || depth > 8 {
			return nil, fmt.Errorf("draft block %d out of range 2..8", depth)
		}
		depths = append(depths, depth)
	}
	if len(depths) == 0 {
		return nil, fmt.Errorf("at least one draft block is required")
	}
	return depths, nil
}

func tuneSweepPlanFromDepths(depths []int, maxTokens int, workload string) []tuneSweepPlan {
	sweep := make([]tuneSweepPlan, 0, len(depths))
	for _, depth := range depths {
		sweep = append(sweep, tuneSweepPlan{
			DraftBlock:  depth,
			DraftTokens: tuneDraftTokensForBlock(depth),
			MaxTokens:   maxTokens,
			Workload:    workload,
		})
	}
	return sweep
}

func tuneDraftTokensForBlock(depth int) int {
	if depth > 1 {
		return depth - 1
	}
	return rocm.ProductionMTPDefaultDraftTokens
}

func tunePlanLabels(opts tuneCommandOptions, depths []int, detection rocm.DraftDetection) map[string]string {
	labels := map[string]string{
		"backend":                             defaultBackendName,
		"cli_contract":                        cliContractName,
		"tuning_stage":                        "reactive_mtp_tuning_plan",
		"mtp_policy":                          "DefaultProductionMTPPolicy",
		"official_locks":                      "DefaultOfficialGemma4E2BLocks",
		"native_mtp_attachment":               "linked",
		"reactive_draft_fallback":             "refused",
		"prompt_replay_fallback":              "refused",
		"profile_writer":                      "enabled_after_successful_sweep",
		"retained_state_required":             "true",
		"tuning_executor":                     "native_attached_mtp",
		"promotion_evidence_required":         "true",
		"measurement_prompt_profile":          productionMeasurementPromptProfileFor(opts.Prompt),
		"target_decode_baseline_required":     "true",
		"profile_requires_native_mtp_speedup": "true",
		"tuning_measurement_shape":            tuneMeasurementShapeRetainedWarmTurn,
		"tune_measured_turn":                  tuneRetainedMeasuredTurn,
		"tune_warmup_max_tokens":              strconv.Itoa(tuneWarmupMaxTokens(tunePlanReport{MaxTokens: opts.MaxTokens})),
		"sampling_profile":                    tuneSamplingProfile(opts),
		"sampling_temperature":                formatTuneFloat32(opts.Temperature),
		"sampling_top_p":                      formatTuneFloat32(opts.TopP),
		"sampling_top_k":                      strconv.Itoa(opts.TopK),
		"sampling_min_p":                      formatTuneFloat32(opts.MinP),
		"production_requires_env_gate":        "false",
		"production_requires_cli_flag":        "false",
		"no_python":                           "true",
		"max_tokens":                          strconv.Itoa(opts.MaxTokens),
		"workload":                            strings.TrimSpace(opts.Workload),
	}
	if labels["workload"] == "" {
		labels["workload"] = "chat"
	}
	if len(depths) > 0 {
		labels["draft_block_sweep"] = joinTuneInts(depths)
		labels["draft_token_sweep"] = joinTuneDraftTokens(depths)
		labels = applyROCmDraftDetectionLabels(labels, detection, depths[0])
	} else {
		labels = applyROCmDraftDetectionLabels(labels, detection, 0)
	}
	labels["native_mtp_attachment"] = "linked"
	labels["reactive_draft_fallback"] = "refused"
	labels["profile_writer"] = "enabled_after_successful_sweep"
	labels["tuning_executor"] = "native_attached_mtp"
	if strings.TrimSpace(opts.ProfileDir) != "" {
		labels["profile_dir"] = strings.TrimSpace(opts.ProfileDir)
	} else if dir := standardTuningProfileDir(); dir != "" {
		labels["profile_dir"] = dir
	}
	labels["profile_reader"] = "generate_serve_auto_profile"
	labels["profile_contract"] = "inference.TuningProfile"
	labels["profile_draft_block_label"] = tuneDraftBlockLabel
	return labels
}

func formatTuneFloat32(value float32) string {
	return strconv.FormatFloat(float64(value), 'f', -1, 32)
}

func joinTuneDraftTokens(depths []int) string {
	values := make([]int, 0, len(depths))
	for _, depth := range depths {
		values = append(values, tuneDraftTokensForBlock(depth))
	}
	return joinTuneInts(values)
}

func joinTuneInts(values []int) string {
	if len(values) == 0 {
		return ""
	}
	parts := make([]string, 0, len(values))
	for _, value := range values {
		parts = append(parts, strconv.Itoa(value))
	}
	return strings.Join(parts, ",")
}

func printTunePlanReport(stdout io.Writer, report tunePlanReport) {
	fmt.Fprintln(stdout, "reactive MTP tuning plan")
	fmt.Fprintf(stdout, "  model: %s\n", report.ModelPath)
	if report.DraftDetection.Active() {
		fmt.Fprintf(stdout, "  draft: %s (%s)\n", report.DraftDetection.DraftPath, report.DraftDetection.Source)
	} else {
		fmt.Fprintln(stdout, "  draft: standing by")
	}
	fmt.Fprintf(stdout, "  workload: %s max_tokens=%d\n", report.Workload, report.MaxTokens)
	fmt.Fprintf(stdout, "  sweep:")
	for _, item := range report.Sweep {
		fmt.Fprintf(stdout, " block=%d/tokens=%d", item.DraftBlock, item.DraftTokens)
	}
	fmt.Fprintln(stdout)
	fmt.Fprintf(stdout, "  policy: %s default_draft_tokens=%d min_turns=%d\n",
		report.Policy.Mode,
		report.Policy.DefaultDraftTokens,
		report.Policy.MinimumRetainedTurns,
	)
	fmt.Fprintf(stdout, "  status: native_mtp_attachment=%s\n", report.Labels["native_mtp_attachment"])
}
