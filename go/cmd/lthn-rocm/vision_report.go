// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"dappco.re/go/inference"
)

type visionCommandOptions struct {
	Backend     string
	ModelPath   string
	Images      string
	VideoFrames string
	FPS         float64
	Prompt      string
	MaxTokens   int
	Chat        bool
}

type visionPlanReport struct {
	Version             int                            `json:"version"`
	Kind                string                         `json:"kind"`
	Backend             string                         `json:"backend"`
	Command             string                         `json:"command"`
	CLIContract         string                         `json:"cli_contract"`
	NoPython            bool                           `json:"no_python"`
	ModelPath           string                         `json:"model_path"`
	Prompt              string                         `json:"prompt"`
	MaxTokens           int                            `json:"max_tokens"`
	ChatTemplate        bool                           `json:"chat_template"`
	Images              []visionAssetReport            `json:"images,omitempty"`
	VideoFrames         []visionAssetReport            `json:"video_frames,omitempty"`
	FPS                 float64                        `json:"fps,omitempty"`
	EstimatedSoftTokens int                            `json:"estimated_soft_tokens,omitempty"`
	SoftTokensPerImage  int                            `json:"soft_tokens_per_image,omitempty"`
	Model               inference.ModelIdentity        `json:"model"`
	Inspection          *inference.ModelPackInspection `json:"inspection,omitempty"`
	Native              visionNativeReport             `json:"native"`
	Labels              map[string]string              `json:"labels,omitempty"`
	Notes               []string                       `json:"notes,omitempty"`
}

type visionAssetReport struct {
	Path      string `json:"path"`
	Kind      string `json:"kind"`
	Extension string `json:"extension,omitempty"`
	Bytes     int64  `json:"bytes,omitempty"`
}

type visionNativeReport struct {
	Multimodal      bool   `json:"multimodal"`
	Runtime         string `json:"runtime"`
	Projector       string `json:"projector_runtime"`
	Reference       string `json:"reference,omitempty"`
	Ready           bool   `json:"ready"`
	Fallback        string `json:"fallback"`
	ExecutionStatus string `json:"execution_status"`
}

func visionPlanReportFromOptions(ctx context.Context, opts visionCommandOptions) (visionPlanReport, error) {
	modelPath := strings.TrimSpace(opts.ModelPath)
	if modelPath == "" {
		return visionPlanReport{}, fmt.Errorf("model path is required")
	}
	if opts.MaxTokens <= 0 {
		return visionPlanReport{}, fmt.Errorf("max tokens must be > 0")
	}
	if opts.FPS <= 0 {
		return visionPlanReport{}, fmt.Errorf("fps must be > 0")
	}
	images, err := visionAssetReports(splitVisionPathList(opts.Images), "image")
	if err != nil {
		return visionPlanReport{}, err
	}
	frames, err := visionAssetReports(splitVisionPathList(opts.VideoFrames), "video_frame")
	if err != nil {
		return visionPlanReport{}, err
	}
	if len(images) == 0 && len(frames) == 0 {
		return visionPlanReport{}, fmt.Errorf("at least one image or video frame is required")
	}
	prompt := strings.TrimSpace(opts.Prompt)
	if prompt == "" {
		prompt = "Describe what you see."
	}
	backend := normalizeBackendName(opts.Backend)
	if backend == "" {
		backend = defaultBackendName
	}
	inspection, err := inspectModelPack(ctx, backend, modelPath)
	if err != nil {
		return visionPlanReport{}, fmt.Errorf("inspect model pack: %w", err)
	}
	labels := visionPlanLabels(opts, inspection, len(images), len(frames))
	native := visionNativeReportFromLabels(labels)
	softTokens := parseVisionPositiveLabel(labels["vision_soft_tokens_per_image"])
	report := visionPlanReport{
		Version:             1,
		Kind:                "native-vision-runtime-report",
		Backend:             backend,
		Command:             "vision",
		CLIContract:         cliContractName,
		NoPython:            true,
		ModelPath:           modelPath,
		Prompt:              prompt,
		MaxTokens:           opts.MaxTokens,
		ChatTemplate:        opts.Chat,
		Images:              images,
		VideoFrames:         frames,
		FPS:                 opts.FPS,
		EstimatedSoftTokens: softTokens * (len(images) + len(frames)),
		SoftTokensPerImage:  softTokens,
		Model:               inspection.Model,
		Inspection:          inspection,
		Native:              native,
		Labels:              labels,
		Notes: []string{
			"ROCm recognises Gemma multimodal model-pack metadata and accepts the go-mlx vision CLI intake shape.",
			"Native ROCm vision tower/projector execution is not linked yet; this report refuses text-only fallback rather than silently dropping images.",
		},
	}
	if native.Multimodal {
		report.Notes = append(report.Notes, "Image/video placeholder accounting is planned against the model's declared soft-token budget.")
	} else {
		report.Notes = append(report.Notes, "The inspected model pack does not declare multimodal vision metadata.")
	}
	return report, nil
}

func visionPlanLabels(opts visionCommandOptions, inspection *inference.ModelPackInspection, imageCount, frameCount int) map[string]string {
	labels := map[string]string{
		"backend":                      defaultBackendName,
		"cli_contract":                 cliContractName,
		"vision_stage":                 "native_vision_runtime_report",
		"reactive_vision_fallback":     "refused",
		"production_requires_env_gate": "false",
		"production_requires_cli_flag": "false",
		"no_python":                    "true",
		"image_count":                  strconv.Itoa(imageCount),
		"video_frame_count":            strconv.Itoa(frameCount),
		"max_tokens":                   strconv.Itoa(opts.MaxTokens),
		"chat_template":                strconv.FormatBool(opts.Chat),
	}
	if opts.FPS > 0 {
		labels["video_frame_fps"] = strconv.FormatFloat(opts.FPS, 'g', -1, 64)
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
		if labels["multimodal_model"] != "true" {
			labels["multimodal_model"] = "false"
		}
	}
	if labels["vision_runtime"] == "" {
		labels["vision_runtime"] = "not_declared"
	}
	if labels["vision_projector_runtime"] == "" {
		labels["vision_projector_runtime"] = "not_declared"
	}
	return labels
}

func visionNativeReportFromLabels(labels map[string]string) visionNativeReport {
	runtime := strings.TrimSpace(labels["vision_runtime"])
	projector := strings.TrimSpace(labels["vision_projector_runtime"])
	ready := runtime == "linked" && projector == "linked"
	status := "not_linked"
	if labels["multimodal_model"] != "true" {
		status = "not_multimodal"
	} else if ready {
		status = "ready"
	}
	return visionNativeReport{
		Multimodal:      labels["multimodal_model"] == "true",
		Runtime:         runtime,
		Projector:       projector,
		Reference:       labels["vision_reference"],
		Ready:           ready,
		Fallback:        labels["reactive_vision_fallback"],
		ExecutionStatus: status,
	}
}

func splitVisionPathList(list string) []string {
	list = strings.TrimSpace(list)
	if list == "" {
		return nil
	}
	parts := strings.Split(list, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

func visionAssetReports(paths []string, kind string) ([]visionAssetReport, error) {
	reports := make([]visionAssetReport, 0, len(paths))
	for _, path := range paths {
		report, err := visionAssetReportForPath(path, kind)
		if err != nil {
			return nil, err
		}
		reports = append(reports, report)
	}
	return reports, nil
}

func visionAssetReportForPath(path, kind string) (visionAssetReport, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return visionAssetReport{}, fmt.Errorf("%s path is required", kind)
	}
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".png", ".jpg", ".jpeg":
	default:
		return visionAssetReport{}, fmt.Errorf("%s %s must be PNG or JPEG", kind, path)
	}
	info, err := os.Stat(path)
	if err != nil {
		return visionAssetReport{}, fmt.Errorf("%s %s: %w", kind, path, err)
	}
	if info.IsDir() {
		return visionAssetReport{}, fmt.Errorf("%s %s is a directory", kind, path)
	}
	return visionAssetReport{
		Path:      path,
		Kind:      kind,
		Extension: ext,
		Bytes:     info.Size(),
	}, nil
}

func parseVisionPositiveLabel(value string) int {
	parsed, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || parsed <= 0 {
		return 0
	}
	return parsed
}

func printVisionPlanReport(stdout io.Writer, report visionPlanReport) {
	fmt.Fprintln(stdout, "native vision runtime report")
	fmt.Fprintf(stdout, "  model: %s\n", report.ModelPath)
	fmt.Fprintf(stdout, "  input: %d image(s), %d video frame(s)", len(report.Images), len(report.VideoFrames))
	if len(report.VideoFrames) > 0 {
		fmt.Fprintf(stdout, " @ %.3g fps", report.FPS)
	}
	fmt.Fprintln(stdout)
	fmt.Fprintf(stdout, "  prompt: %s\n", report.Prompt)
	fmt.Fprintf(stdout, "  soft_tokens: %d", report.EstimatedSoftTokens)
	if report.SoftTokensPerImage > 0 {
		fmt.Fprintf(stdout, " (%d per image/frame)", report.SoftTokensPerImage)
	}
	fmt.Fprintln(stdout)
	fmt.Fprintf(stdout, "  model: architecture=%s supported=%t multimodal=%t\n",
		report.Model.Architecture,
		report.Inspection != nil && report.Inspection.Supported,
		report.Native.Multimodal,
	)
	fmt.Fprintf(stdout, "  native: vision=%s projector=%s ready=%t fallback=%s\n",
		report.Native.Runtime,
		report.Native.Projector,
		report.Native.Ready,
		report.Native.Fallback,
	)
}
