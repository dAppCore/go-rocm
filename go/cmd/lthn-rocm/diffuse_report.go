// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"fmt"
	"io"
	"strconv"
	"strings"

	"dappco.re/go/inference"
)

type diffuseCommandOptions struct {
	Backend         string
	ModelPath       string
	Prompt          string
	MaxCanvases     int
	Steps           int
	Canvas          int
	Entropy         float64
	Seed            uint64
	Chat            bool
	Trace           bool
	CompatMaxTokens int
	CompatTemp      float64
}

type diffusePlanReport struct {
	Version      int                            `json:"version"`
	Kind         string                         `json:"kind"`
	Backend      string                         `json:"backend"`
	Command      string                         `json:"command"`
	CLIContract  string                         `json:"cli_contract"`
	NoPython     bool                           `json:"no_python"`
	ModelPath    string                         `json:"model_path"`
	Prompt       string                         `json:"prompt"`
	MaxCanvases  int                            `json:"max_canvases"`
	Steps        int                            `json:"steps"`
	Canvas       int                            `json:"canvas"`
	Entropy      float64                        `json:"entropy"`
	Seed         uint64                         `json:"seed"`
	ChatTemplate bool                           `json:"chat_template"`
	Trace        bool                           `json:"trace"`
	Model        inference.ModelIdentity        `json:"model"`
	Inspection   *inference.ModelPackInspection `json:"inspection,omitempty"`
	Native       diffuseNativeReport            `json:"native"`
	Labels       map[string]string              `json:"labels,omitempty"`
	Notes        []string                       `json:"notes,omitempty"`
}

type diffuseNativeReport struct {
	BlockDiffusion    bool   `json:"block_diffusion"`
	Runtime           string `json:"runtime"`
	Sampler           string `json:"sampler_runtime"`
	Trunk             string `json:"trunk_runtime"`
	Reference         string `json:"reference,omitempty"`
	ModelCanvasLength int    `json:"model_canvas_length,omitempty"`
	Ready             bool   `json:"ready"`
	Fallback          string `json:"fallback"`
	ExecutionStatus   string `json:"execution_status"`
}

func diffusePlanReportFromOptions(ctx context.Context, opts diffuseCommandOptions) (diffusePlanReport, error) {
	modelPath := strings.TrimSpace(opts.ModelPath)
	if modelPath == "" {
		return diffusePlanReport{}, fmt.Errorf("model path is required")
	}
	if opts.MaxCanvases <= 0 {
		return diffusePlanReport{}, fmt.Errorf("max canvases must be > 0")
	}
	if opts.Steps < 0 {
		return diffusePlanReport{}, fmt.Errorf("steps must be >= 0")
	}
	if opts.Canvas < 0 {
		return diffusePlanReport{}, fmt.Errorf("canvas must be >= 0")
	}
	if opts.Entropy < 0 {
		return diffusePlanReport{}, fmt.Errorf("entropy must be >= 0")
	}
	prompt := strings.TrimSpace(opts.Prompt)
	if prompt == "" {
		prompt = "Write a haiku about clockwork."
	}
	backend := normalizeBackendName(opts.Backend)
	if backend == "" {
		backend = defaultBackendName
	}
	opts.Backend = backend
	inspection, err := inspectModelPack(ctx, backend, modelPath)
	if err != nil {
		return diffusePlanReport{}, fmt.Errorf("inspect model pack: %w", err)
	}
	labels := diffusePlanLabels(opts, inspection)
	native := diffuseNativeReportFromLabels(labels)
	report := diffusePlanReport{
		Version:      1,
		Kind:         "native-diffusion-runtime-report",
		Backend:      backend,
		Command:      "diffuse",
		CLIContract:  cliContractName,
		NoPython:     true,
		ModelPath:    modelPath,
		Prompt:       prompt,
		MaxCanvases:  opts.MaxCanvases,
		Steps:        opts.Steps,
		Canvas:       opts.Canvas,
		Entropy:      opts.Entropy,
		Seed:         opts.Seed,
		ChatTemplate: opts.Chat,
		Trace:        opts.Trace,
		Model:        inspection.Model,
		Inspection:   inspection,
		Native:       native,
		Labels:       labels,
		Notes: []string{
			"ROCm recognises DiffusionGemma model-pack metadata and accepts the go-mlx diffuse CLI intake shape.",
			"Native ROCm canvas denoising sampler execution is not linked yet; this report refuses autoregressive fallback rather than silently changing decode mode.",
		},
	}
	if !native.BlockDiffusion {
		report.Notes = append(report.Notes, "The inspected model pack does not declare DiffusionGemma block-diffusion metadata.")
	}
	return report, nil
}

func diffusePlanLabels(opts diffuseCommandOptions, inspection *inference.ModelPackInspection) map[string]string {
	labels := map[string]string{
		"backend":                      defaultString(opts.Backend, defaultBackendName),
		"cli_contract":                 cliContractName,
		"diffusion_stage":              "native_diffusion_runtime_report",
		"reactive_diffusion_fallback":  "refused",
		"production_requires_env_gate": "false",
		"production_requires_cli_flag": "false",
		"no_python":                    "true",
		"max_canvases":                 strconv.Itoa(opts.MaxCanvases),
		"diffusion_steps":              strconv.Itoa(opts.Steps),
		"diffusion_canvas":             "auto",
		"diffusion_entropy":            strconv.FormatFloat(opts.Entropy, 'g', -1, 64),
		"chat_template":                strconv.FormatBool(opts.Chat),
		"trace":                        strconv.FormatBool(opts.Trace),
	}
	if opts.Canvas > 0 {
		labels["diffusion_canvas"] = strconv.Itoa(opts.Canvas)
	}
	if opts.Seed > 0 {
		labels["diffusion_seed"] = strconv.FormatUint(opts.Seed, 10)
	}
	if opts.CompatMaxTokens > 0 {
		labels["legacy_max_tokens"] = strconv.Itoa(opts.CompatMaxTokens)
	}
	if opts.CompatTemp != 0 {
		labels["legacy_temp"] = strconv.FormatFloat(opts.CompatTemp, 'g', -1, 64)
	}
	if inspection != nil {
		for key, value := range inspection.Labels {
			labels[key] = value
		}
		if inspection.Supported {
			labels["model_pack_supported"] = "true"
		} else {
			labels["model_pack_supported"] = "false"
		}
		if inspection.Model.Architecture == "diffusion_gemma" {
			labels["block_diffusion_model"] = "true"
		}
	}
	if labels["block_diffusion_model"] != "true" {
		labels["block_diffusion_model"] = "false"
	}
	if labels["diffusion_runtime"] == "" {
		labels["diffusion_runtime"] = "not_declared"
	}
	if labels["diffusion_sampler_runtime"] == "" {
		labels["diffusion_sampler_runtime"] = "not_declared"
	}
	if labels["diffusion_trunk_runtime"] == "" {
		labels["diffusion_trunk_runtime"] = "not_declared"
	}
	if labels["diffusion_fallback"] == "" {
		labels["diffusion_fallback"] = labels["reactive_diffusion_fallback"]
	}
	return labels
}

func diffuseNativeReportFromLabels(labels map[string]string) diffuseNativeReport {
	runtime := strings.TrimSpace(labels["diffusion_runtime"])
	sampler := strings.TrimSpace(labels["diffusion_sampler_runtime"])
	trunk := strings.TrimSpace(labels["diffusion_trunk_runtime"])
	blockDiffusion := labels["block_diffusion_model"] == "true"
	ready := blockDiffusion && runtime == "linked" && sampler == "linked"
	status := "not_linked"
	if !blockDiffusion {
		status = "not_block_diffusion_model"
	} else if ready {
		status = "ready"
	}
	fallback := labels["diffusion_fallback"]
	if fallback == "" {
		fallback = labels["reactive_diffusion_fallback"]
	}
	return diffuseNativeReport{
		BlockDiffusion:    blockDiffusion,
		Runtime:           runtime,
		Sampler:           sampler,
		Trunk:             trunk,
		Reference:         labels["diffusion_reference"],
		ModelCanvasLength: parseDiffusePositiveLabel(labels["diffusion_canvas_length"]),
		Ready:             ready,
		Fallback:          fallback,
		ExecutionStatus:   status,
	}
}

func diffuseReportExitCode(report diffusePlanReport) int {
	if report.Native.Ready {
		return 0
	}
	return 1
}

func printDiffusePlanSummary(stdout io.Writer, report diffusePlanReport) {
	fmt.Fprintln(stdout, "block-diffusion runtime report")
	fmt.Fprintf(stdout, "  model: %s (%s)\n", report.ModelPath, defaultString(report.Model.Architecture, "unknown"))
	fmt.Fprintf(stdout, "  prompt: %q\n", report.Prompt)
	fmt.Fprintf(stdout, "  canvases: %d canvas=%d steps=%d entropy=%g chat=%t trace=%t\n", report.MaxCanvases, report.Canvas, report.Steps, report.Entropy, report.ChatTemplate, report.Trace)
	fmt.Fprintf(stdout, "  native: block_diffusion=%t runtime=%s sampler=%s trunk=%s ready=%t fallback=%s\n",
		report.Native.BlockDiffusion,
		defaultString(report.Native.Runtime, "unknown"),
		defaultString(report.Native.Sampler, "unknown"),
		defaultString(report.Native.Trunk, "unknown"),
		report.Native.Ready,
		defaultString(report.Native.Fallback, "unknown"))
	fmt.Fprintf(stdout, "  status: %s\n", defaultString(report.Native.ExecutionStatus, "unknown"))
}

func parseDiffusePositiveLabel(value string) int {
	parsed, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || parsed <= 0 {
		return 0
	}
	return parsed
}
