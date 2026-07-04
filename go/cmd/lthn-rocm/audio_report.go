// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"dappco.re/go/inference"
)

type audioCommandOptions struct {
	Backend   string
	ModelPath string
	AudioPath string
	Prompt    string
	MaxTokens int
	Chat      bool
}

type audioPlanReport struct {
	Version      int                            `json:"version"`
	Kind         string                         `json:"kind"`
	Backend      string                         `json:"backend"`
	Command      string                         `json:"command"`
	CLIContract  string                         `json:"cli_contract"`
	NoPython     bool                           `json:"no_python"`
	ModelPath    string                         `json:"model_path"`
	Prompt       string                         `json:"prompt"`
	MaxTokens    int                            `json:"max_tokens"`
	ChatTemplate bool                           `json:"chat_template"`
	Audio        audioAssetReport               `json:"audio"`
	Model        inference.ModelIdentity        `json:"model"`
	Inspection   *inference.ModelPackInspection `json:"inspection,omitempty"`
	Native       audioNativeReport              `json:"native"`
	Labels       map[string]string              `json:"labels,omitempty"`
	Notes        []string                       `json:"notes,omitempty"`
}

type audioAssetReport struct {
	Path          string  `json:"path"`
	Extension     string  `json:"extension,omitempty"`
	Bytes         int64   `json:"bytes,omitempty"`
	Format        uint16  `json:"format,omitempty"`
	Channels      uint16  `json:"channels,omitempty"`
	SampleRate    uint32  `json:"sample_rate,omitempty"`
	BitsPerSample uint16  `json:"bits_per_sample,omitempty"`
	Frames        int     `json:"frames,omitempty"`
	DurationSec   float64 `json:"duration_sec,omitempty"`
}

type audioNativeReport struct {
	Multimodal      bool   `json:"multimodal"`
	Runtime         string `json:"runtime"`
	Projector       string `json:"projector_runtime"`
	FrontEnd        string `json:"front_end_runtime"`
	Reference       string `json:"reference,omitempty"`
	Ready           bool   `json:"ready"`
	Fallback        string `json:"fallback"`
	ExecutionStatus string `json:"execution_status"`
}

func audioPlanReportFromOptions(ctx context.Context, opts audioCommandOptions) (audioPlanReport, error) {
	modelPath := strings.TrimSpace(opts.ModelPath)
	if modelPath == "" {
		return audioPlanReport{}, fmt.Errorf("model path is required")
	}
	if opts.MaxTokens <= 0 {
		return audioPlanReport{}, fmt.Errorf("max tokens must be > 0")
	}
	audio, err := audioAssetReportForPath(opts.AudioPath)
	if err != nil {
		return audioPlanReport{}, err
	}
	prompt := strings.TrimSpace(opts.Prompt)
	if prompt == "" {
		prompt = "What is said in this recording?"
	}
	backend := normalizeBackendName(opts.Backend)
	if backend == "" {
		backend = defaultBackendName
	}
	inspection, err := inspectModelPack(ctx, backend, modelPath)
	if err != nil {
		return audioPlanReport{}, fmt.Errorf("inspect model pack: %w", err)
	}
	labels := audioPlanLabels(opts, inspection, audio)
	native := audioNativeReportFromLabels(labels)
	return audioPlanReport{
		Version:      1,
		Kind:         "native-audio-runtime-report",
		Backend:      backend,
		Command:      "audio",
		CLIContract:  cliContractName,
		NoPython:     true,
		ModelPath:    modelPath,
		Prompt:       prompt,
		MaxTokens:    opts.MaxTokens,
		ChatTemplate: opts.Chat,
		Audio:        audio,
		Model:        inspection.Model,
		Inspection:   inspection,
		Native:       native,
		Labels:       labels,
		Notes: []string{
			"ROCm recognises Gemma audio model-pack metadata and accepts the go-mlx audio CLI intake shape.",
			"Native ROCm WAV front-end, audio tower, and projector execution are not linked yet; this report refuses text-only fallback rather than silently dropping audio.",
		},
	}, nil
}

func audioPlanLabels(opts audioCommandOptions, inspection *inference.ModelPackInspection, audio audioAssetReport) map[string]string {
	labels := map[string]string{
		"backend":                      defaultBackendName,
		"cli_contract":                 cliContractName,
		"audio_stage":                  "native_audio_runtime_report",
		"reactive_audio_fallback":      "refused",
		"audio_front_end_runtime":      "not_linked",
		"production_requires_env_gate": "false",
		"production_requires_cli_flag": "false",
		"no_python":                    "true",
		"audio_path":                   audio.Path,
		"audio_sample_rate":            strconv.FormatUint(uint64(audio.SampleRate), 10),
		"audio_channels":               strconv.FormatUint(uint64(audio.Channels), 10),
		"audio_bits_per_sample":        strconv.FormatUint(uint64(audio.BitsPerSample), 10),
		"max_tokens":                   strconv.Itoa(opts.MaxTokens),
		"chat_template":                strconv.FormatBool(opts.Chat),
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
	if labels["audio_runtime"] == "" {
		labels["audio_runtime"] = "not_declared"
	}
	if labels["audio_projector_runtime"] == "" {
		labels["audio_projector_runtime"] = "not_declared"
	}
	return labels
}

func audioNativeReportFromLabels(labels map[string]string) audioNativeReport {
	runtime := strings.TrimSpace(labels["audio_runtime"])
	projector := strings.TrimSpace(labels["audio_projector_runtime"])
	frontEnd := strings.TrimSpace(labels["audio_front_end_runtime"])
	ready := runtime == "linked" && projector == "linked" && frontEnd == "linked"
	status := "not_linked"
	if labels["multimodal_model"] != "true" || labels["audio_token_id"] == "" {
		status = "not_audio_multimodal"
	} else if ready {
		status = "ready"
	}
	return audioNativeReport{
		Multimodal:      labels["multimodal_model"] == "true" && labels["audio_token_id"] != "",
		Runtime:         runtime,
		Projector:       projector,
		FrontEnd:        frontEnd,
		Reference:       labels["audio_reference"],
		Ready:           ready,
		Fallback:        labels["reactive_audio_fallback"],
		ExecutionStatus: status,
	}
}

func audioAssetReportForPath(path string) (audioAssetReport, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return audioAssetReport{}, fmt.Errorf("audio path is required")
	}
	if strings.ToLower(filepath.Ext(path)) != ".wav" {
		return audioAssetReport{}, fmt.Errorf("audio %s must be a WAV file", path)
	}
	info, err := os.Stat(path)
	if err != nil {
		return audioAssetReport{}, fmt.Errorf("audio %s: %w", path, err)
	}
	if info.IsDir() {
		return audioAssetReport{}, fmt.Errorf("audio %s is a directory", path)
	}
	report, err := inspectWAVHeader(path)
	if err != nil {
		return audioAssetReport{}, err
	}
	report.Path = path
	report.Extension = ".wav"
	report.Bytes = info.Size()
	return report, nil
}

func inspectWAVHeader(path string) (audioAssetReport, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return audioAssetReport{}, fmt.Errorf("audio %s: %w", path, err)
	}
	if len(data) < 44 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return audioAssetReport{}, fmt.Errorf("audio %s is not a RIFF/WAVE file", path)
	}
	var report audioAssetReport
	le := binary.LittleEndian
	offset := 12
	var dataBytes int
	for offset+8 <= len(data) {
		chunkID := string(data[offset : offset+4])
		chunkLen := int(le.Uint32(data[offset+4 : offset+8]))
		body := offset + 8
		if chunkLen < 0 || body+chunkLen > len(data) {
			return audioAssetReport{}, fmt.Errorf("audio %s has a truncated WAV chunk", path)
		}
		switch chunkID {
		case "fmt ":
			if chunkLen < 16 {
				return audioAssetReport{}, fmt.Errorf("audio %s has a malformed WAV fmt chunk", path)
			}
			report.Format = le.Uint16(data[body : body+2])
			report.Channels = le.Uint16(data[body+2 : body+4])
			report.SampleRate = le.Uint32(data[body+4 : body+8])
			report.BitsPerSample = le.Uint16(data[body+14 : body+16])
		case "data":
			dataBytes = chunkLen
		}
		offset = body + chunkLen + (chunkLen & 1)
	}
	if report.Channels == 0 || report.SampleRate == 0 || report.BitsPerSample == 0 {
		return audioAssetReport{}, fmt.Errorf("audio %s has no usable WAV fmt chunk", path)
	}
	if report.Format != 1 && report.Format != 3 {
		return audioAssetReport{}, fmt.Errorf("audio %s unsupported WAV encoding: format %d", path, report.Format)
	}
	if dataBytes == 0 {
		return audioAssetReport{}, fmt.Errorf("audio %s has no WAV data chunk", path)
	}
	bytesPerFrame := int(report.Channels) * int(report.BitsPerSample) / 8
	if bytesPerFrame > 0 {
		report.Frames = dataBytes / bytesPerFrame
		report.DurationSec = float64(report.Frames) / float64(report.SampleRate)
	}
	return report, nil
}

func printAudioPlanReport(stdout io.Writer, report audioPlanReport) {
	fmt.Fprintln(stdout, "native audio runtime report")
	fmt.Fprintf(stdout, "  model: %s\n", report.ModelPath)
	fmt.Fprintf(stdout, "  audio: %s %.3gs %dHz %dch %dbit\n",
		report.Audio.Path,
		report.Audio.DurationSec,
		report.Audio.SampleRate,
		report.Audio.Channels,
		report.Audio.BitsPerSample,
	)
	fmt.Fprintf(stdout, "  prompt: %s\n", report.Prompt)
	fmt.Fprintf(stdout, "  model: architecture=%s supported=%t multimodal=%t\n",
		report.Model.Architecture,
		report.Inspection != nil && report.Inspection.Supported,
		report.Native.Multimodal,
	)
	fmt.Fprintf(stdout, "  native: audio=%s projector=%s front_end=%s ready=%t fallback=%s\n",
		report.Native.Runtime,
		report.Native.Projector,
		report.Native.FrontEnd,
		report.Native.Ready,
		report.Native.Fallback,
	)
}
