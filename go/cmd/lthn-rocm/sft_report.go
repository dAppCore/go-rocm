// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"dappco.re/go/inference"
	rocm "dappco.re/go/rocm"
)

type sftCommandOptions struct {
	BackendName             string
	ModelPath               string
	DataPath                string
	ValidationPath          string
	ValidationSamples       int
	ValidationEvery         int
	EvalPromptsPath         string
	EvalEvery               int
	EvalMaxTokens           int
	EvalProbes              int
	ScoreCascade            bool
	ScoreWindow             int
	RunID                   string
	MetricsLineProtocolPath string
	InfluxURL               string
	InfluxToken             string
	CapturePath             string
	Rank                    int
	Alpha                   float64
	LearningRate            float64
	Epochs                  int
	BatchSize               int
	GradientAccumulation    int
	MaxSequenceLength       int
	Packing                 bool
	CheckpointDir           string
	CheckpointEvery         int
	OutputAdapterPath       string
	ResumeAdapterPath       string
	MergeAfterTraining      bool
	ContextOverride         int
}

type sftPlanReport struct {
	Version            int                `json:"version"`
	Kind               string             `json:"kind"`
	Backend            string             `json:"backend"`
	Command            string             `json:"command"`
	CLIContract        string             `json:"cli_contract"`
	NoPython           bool               `json:"no_python"`
	ModelPath          string             `json:"model_path"`
	DataPath           string             `json:"data_path"`
	ValidationPath     string             `json:"validation_path,omitempty"`
	EvalPromptsPath    string             `json:"eval_prompts_path,omitempty"`
	OutputAdapterPath  string             `json:"output_adapter_path,omitempty"`
	ResumeAdapterPath  string             `json:"resume_adapter_path,omitempty"`
	CheckpointDir      string             `json:"checkpoint_dir,omitempty"`
	CapturePath        string             `json:"capture_path,omitempty"`
	RunID              string             `json:"run_id,omitempty"`
	Metrics            sftMetricsPlan     `json:"metrics,omitempty"`
	MergeAfterTraining bool               `json:"merge_after_training,omitempty"`
	Config             sftPlanConfig      `json:"config"`
	Dataset            sftDatasetSummary  `json:"dataset"`
	Validation         *sftDatasetSummary `json:"validation,omitempty"`
	Labels             map[string]string  `json:"labels,omitempty"`
	Notes              []string           `json:"notes,omitempty"`
}

type sftPlanConfig struct {
	LoRA                 sftLoRAPlanConfig `json:"lora"`
	LearningRate         float64           `json:"learning_rate"`
	Epochs               int               `json:"epochs"`
	BatchSize            int               `json:"batch_size"`
	GradientAccumulation int               `json:"gradient_accumulation"`
	MaxSequenceLength    int               `json:"max_sequence_length"`
	Packing              bool              `json:"packing,omitempty"`
	ContextOverride      int               `json:"context_override,omitempty"`
	ValidationSamples    int               `json:"validation_samples,omitempty"`
	ValidationEvery      int               `json:"validation_every,omitempty"`
	EvalEvery            int               `json:"eval_every,omitempty"`
	EvalMaxTokens        int               `json:"eval_max_tokens,omitempty"`
	EvalProbes           int               `json:"eval_probes,omitempty"`
	ScoreCascade         bool              `json:"score_cascade,omitempty"`
	ScoreWindow          int               `json:"score_window,omitempty"`
	CheckpointEvery      int               `json:"checkpoint_every,omitempty"`
}

type sftLoRAPlanConfig struct {
	Rank       int      `json:"rank"`
	Alpha      float64  `json:"alpha"`
	TargetKeys []string `json:"target_keys,omitempty"`
	BFloat16   bool     `json:"bfloat16,omitempty"`
}

type sftDatasetSummary struct {
	Path         string   `json:"path"`
	Samples      int      `json:"samples"`
	Formats      []string `json:"formats,omitempty"`
	PromptRows   int      `json:"prompt_rows,omitempty"`
	ResponseRows int      `json:"response_rows,omitempty"`
	MessageRows  int      `json:"message_rows,omitempty"`
	TextRows     int      `json:"text_rows,omitempty"`
	TargetRows   int      `json:"target_token_rows,omitempty"`
}

type sftMetricsPlan struct {
	LineProtocolPath string `json:"line_protocol_path,omitempty"`
	InfluxConfigured bool   `json:"influx_configured,omitempty"`
	InfluxTokenSet   bool   `json:"influx_token_set,omitempty"`
}

func sftPlanReportFromOptions(opts sftCommandOptions) (sftPlanReport, error) {
	opts = normalizeSFTCommandOptions(opts)
	if err := validateSFTCommandOptions(opts); err != nil {
		return sftPlanReport{}, err
	}
	backend := normalizeBackendName(opts.BackendName)
	dataset, err := summarizeSFTDataset(opts.DataPath)
	if err != nil {
		return sftPlanReport{}, fmt.Errorf("training dataset: %w", err)
	}
	var validation *sftDatasetSummary
	if strings.TrimSpace(opts.ValidationPath) != "" {
		summary, err := summarizeSFTDataset(opts.ValidationPath)
		if err != nil {
			return sftPlanReport{}, fmt.Errorf("validation dataset: %w", err)
		}
		validation = &summary
	}
	labels := sftPlanLabels(opts)
	report := sftPlanReport{
		Version:            1,
		Kind:               "native-lora-sft-plan",
		Backend:            backend,
		Command:            "sft",
		CLIContract:        cliContractName,
		NoPython:           true,
		ModelPath:          strings.TrimSpace(opts.ModelPath),
		DataPath:           strings.TrimSpace(opts.DataPath),
		ValidationPath:     strings.TrimSpace(opts.ValidationPath),
		EvalPromptsPath:    strings.TrimSpace(opts.EvalPromptsPath),
		OutputAdapterPath:  strings.TrimSpace(opts.OutputAdapterPath),
		ResumeAdapterPath:  strings.TrimSpace(opts.ResumeAdapterPath),
		CheckpointDir:      strings.TrimSpace(opts.CheckpointDir),
		CapturePath:        strings.TrimSpace(opts.CapturePath),
		RunID:              strings.TrimSpace(opts.RunID),
		Metrics:            sftMetricsPlanFromOptions(opts),
		MergeAfterTraining: opts.MergeAfterTraining,
		Config:             sftPlanConfigFromOptions(opts),
		Dataset:            dataset,
		Validation:         validation,
		Labels:             labels,
		Notes: []string{
			"The default CLI path loads the selected backend and calls inference.SFTTrainer when that backend exposes it.",
			"ROCm can parse the training JSONL and has native SFT loss plus AdamW update helpers linked into this binary.",
			"The ROCm backend model still fails closed until a full TrainSFT implementation is linked.",
		},
	}
	return report, nil
}

func normalizeSFTCommandOptions(opts sftCommandOptions) sftCommandOptions {
	opts.BackendName = normalizeBackendName(opts.BackendName)
	checkpointDir := strings.TrimSpace(opts.CheckpointDir)
	if checkpointDir != "" {
		if strings.TrimSpace(opts.OutputAdapterPath) == "" {
			opts.OutputAdapterPath = filepath.Join(checkpointDir, "adapter.safetensors")
		}
		if strings.TrimSpace(opts.MetricsLineProtocolPath) == "" {
			opts.MetricsLineProtocolPath = filepath.Join(checkpointDir, "metrics.lp")
		}
		if strings.TrimSpace(opts.CapturePath) == "" {
			opts.CapturePath = filepath.Join(checkpointDir, "captures.jsonl")
		}
	}
	if strings.EqualFold(strings.TrimSpace(opts.MetricsLineProtocolPath), "off") {
		opts.MetricsLineProtocolPath = ""
	}
	if strings.EqualFold(strings.TrimSpace(opts.CapturePath), "off") {
		opts.CapturePath = ""
	}
	return opts
}

func validateSFTCommandOptions(opts sftCommandOptions) error {
	switch {
	case strings.TrimSpace(opts.ModelPath) == "":
		return fmt.Errorf("model path is required")
	case strings.TrimSpace(opts.DataPath) == "":
		return fmt.Errorf("training data path is required")
	case opts.Rank <= 0:
		return fmt.Errorf("rank must be > 0")
	case opts.Alpha <= 0:
		return fmt.Errorf("alpha must be > 0")
	case opts.LearningRate <= 0:
		return fmt.Errorf("learning rate must be > 0")
	case opts.Epochs <= 0:
		return fmt.Errorf("epochs must be > 0")
	case opts.BatchSize <= 0:
		return fmt.Errorf("batch size must be > 0")
	case opts.GradientAccumulation <= 0:
		return fmt.Errorf("gradient accumulation must be > 0")
	case opts.MaxSequenceLength <= 0:
		return fmt.Errorf("max sequence length must be > 0")
	case opts.ValidationSamples < 0:
		return fmt.Errorf("validation samples must be >= 0")
	case opts.ValidationEvery < 0:
		return fmt.Errorf("validation cadence must be >= 0")
	case opts.EvalEvery < 0:
		return fmt.Errorf("eval cadence must be >= 0")
	case opts.EvalMaxTokens < 0:
		return fmt.Errorf("eval max tokens must be >= 0")
	case opts.EvalProbes < 0:
		return fmt.Errorf("eval probes must be >= 0")
	case opts.ScoreWindow < 0:
		return fmt.Errorf("score window must be >= 0")
	case opts.CheckpointEvery < 0:
		return fmt.Errorf("checkpoint cadence must be >= 0")
	case opts.ContextOverride < 0:
		return fmt.Errorf("context override must be >= 0")
	default:
		return nil
	}
}

func sftPlanConfigFromOptions(opts sftCommandOptions) sftPlanConfig {
	defaultLoRA := inference.DefaultLoRAConfig()
	return sftPlanConfig{
		LoRA: sftLoRAPlanConfig{
			Rank:       opts.Rank,
			Alpha:      opts.Alpha,
			TargetKeys: append([]string(nil), defaultLoRA.TargetKeys...),
			BFloat16:   defaultLoRA.BFloat16,
		},
		LearningRate:         opts.LearningRate,
		Epochs:               opts.Epochs,
		BatchSize:            opts.BatchSize,
		GradientAccumulation: opts.GradientAccumulation,
		MaxSequenceLength:    opts.MaxSequenceLength,
		Packing:              opts.Packing,
		ContextOverride:      opts.ContextOverride,
		ValidationSamples:    opts.ValidationSamples,
		ValidationEvery:      opts.ValidationEvery,
		EvalEvery:            opts.EvalEvery,
		EvalMaxTokens:        opts.EvalMaxTokens,
		EvalProbes:           opts.EvalProbes,
		ScoreCascade:         opts.ScoreCascade,
		ScoreWindow:          opts.ScoreWindow,
		CheckpointEvery:      opts.CheckpointEvery,
	}
}

func sftMetricsPlanFromOptions(opts sftCommandOptions) sftMetricsPlan {
	return sftMetricsPlan{
		LineProtocolPath: strings.TrimSpace(opts.MetricsLineProtocolPath),
		InfluxConfigured: strings.TrimSpace(opts.InfluxURL) != "",
		InfluxTokenSet:   strings.TrimSpace(opts.InfluxToken) != "",
	}
}

func sftPlanLabels(opts sftCommandOptions) map[string]string {
	backend := normalizeBackendName(opts.BackendName)
	labels := map[string]string{
		"backend":                             backend,
		"cli_contract":                        cliContractName,
		"training_stage":                      "native_lora_sft_plan",
		"adapter_format":                      "lora",
		"model_loader_interface":              "linked",
		"sft_trainer_interface":               "inference.SFTTrainer",
		"dataset_loader":                      "rocm.LoadJSONLDataset",
		"loss_helper":                         "RunNativeSFTLossPass",
		"lora_backward_helper":                "RunNativeLoRABackwardPass",
		"lora_update_helper":                  "RunNativeLoRAAdamWUpdatePass",
		"lora_track_helper":                   "RunNativeLoRAAdamWUpdateTrackPass",
		"lora_adapter_snapshot_helper":        "SaveNativeLoRAAdapterSnapshot",
		"lora_adapter_track_snapshot_helper":  "SaveNativeLoRAAdapterSnapshotTrackStep",
		"lora_adapter_track_latest_helper":    "SaveNativeLoRAAdapterSnapshotTrackLast",
		"optimizer_helper":                    "RunNativeSFTAdamWUpdatePass",
		"optimizer_track_helper":              "RunNativeSFTAdamWUpdateTrackPass",
		"trainer_interface":                   "runtime_checked",
		"native_trainer_interface":            "not_implemented",
		"training_interface":                  "lora_backward_plus_optimizer_update",
		"native_loss_pass":                    "linked",
		"native_lora_backward":                "reference",
		"native_lora_update_pass":             "linked",
		"adamw_update_pass":                   "linked",
		"production_requires_env_gate":        "false",
		"production_requires_cli_flag":        "false",
		"no_python":                           "true",
		"lora_rank":                           strconv.Itoa(opts.Rank),
		"lora_alpha":                          strconv.FormatFloat(opts.Alpha, 'f', -1, 64),
		"gradient_accumulation":               strconv.Itoa(opts.GradientAccumulation),
		"reactive_training_contract":          "live_when_backend_trainer",
		"reactive_adapter_save_path_required": strconv.FormatBool(strings.TrimSpace(opts.OutputAdapterPath) != ""),
	}
	if opts.MergeAfterTraining {
		labels["merge_after_training"] = "true"
	}
	if strings.TrimSpace(opts.ResumeAdapterPath) != "" {
		labels["resume_adapter"] = "true"
	}
	return labels
}

func summarizeSFTDataset(path string) (sftDatasetSummary, error) {
	path = strings.TrimSpace(path)
	file, err := os.Open(path)
	if err != nil {
		return sftDatasetSummary{}, err
	}
	defer file.Close()
	dataset, err := rocm.LoadJSONLDataset(file)
	if err != nil {
		return sftDatasetSummary{}, err
	}
	samples := dataset.Samples()
	summary := sftDatasetSummary{
		Path:    path,
		Samples: len(samples),
	}
	formatSet := make(map[string]struct{})
	for _, sample := range samples {
		if format := strings.TrimSpace(sample.Labels["format"]); format != "" {
			formatSet[format] = struct{}{}
		}
		if strings.TrimSpace(sample.Prompt) != "" {
			summary.PromptRows++
		}
		if strings.TrimSpace(sample.Response) != "" {
			summary.ResponseRows++
		}
		if len(sample.Messages) > 0 {
			summary.MessageRows++
		}
		if strings.TrimSpace(sample.Text) != "" {
			summary.TextRows++
		}
		if _, ok := sample.Labels["target_token_id"]; ok {
			summary.TargetRows++
		}
	}
	summary.Formats = sortedSFTDatasetFormats(formatSet)
	return summary, nil
}

func sortedSFTDatasetFormats(formatSet map[string]struct{}) []string {
	if len(formatSet) == 0 {
		return nil
	}
	formats := make([]string, 0, len(formatSet))
	for format := range formatSet {
		formats = append(formats, format)
	}
	sort.Strings(formats)
	return formats
}

func printSFTPlanReport(stdout io.Writer, report sftPlanReport) {
	fmt.Fprintln(stdout, "native LoRA SFT plan")
	fmt.Fprintf(stdout, "  model: %s\n", report.ModelPath)
	fmt.Fprintf(stdout, "  train: %d sample(s) from %s", report.Dataset.Samples, report.Dataset.Path)
	if len(report.Dataset.Formats) > 0 {
		fmt.Fprintf(stdout, " (%s)", strings.Join(report.Dataset.Formats, ","))
	}
	fmt.Fprintln(stdout)
	if report.Validation != nil {
		fmt.Fprintf(stdout, "  valid: %d sample(s) from %s\n", report.Validation.Samples, report.Validation.Path)
	}
	fmt.Fprintf(stdout, "  lora: rank=%d alpha=%g targets=%s\n", report.Config.LoRA.Rank, report.Config.LoRA.Alpha, strings.Join(report.Config.LoRA.TargetKeys, ","))
	fmt.Fprintf(stdout, "  optimizer: adamw lr=%g epochs=%d batch=%d grad_accum=%d max_seq=%d\n",
		report.Config.LearningRate,
		report.Config.Epochs,
		report.Config.BatchSize,
		report.Config.GradientAccumulation,
		report.Config.MaxSequenceLength,
	)
	fmt.Fprintf(stdout, "  status: trainer_interface=%s\n", report.Labels["trainer_interface"])
}
