// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"dappco.re/go/inference"
	rocm "dappco.re/go/rocm"
)

func runSFTExecute(ctx context.Context, opts sftCommandOptions, stdout, stderr io.Writer) int {
	dataFile, err := os.Open(strings.TrimSpace(opts.DataPath))
	if err != nil {
		fmt.Fprintf(stderr, "%s sft: training data unreadable: %v\n", cliName(), err)
		return 1
	}
	defer dataFile.Close()
	dataset, err := rocm.LoadJSONLDataset(dataFile)
	if err != nil {
		fmt.Fprintf(stderr, "%s sft: training data parse: %v\n", cliName(), err)
		return 1
	}

	loadOpts := make([]inference.LoadOption, 0, 2)
	if backend := normalizeBackendName(opts.BackendName); backend != "" && backend != "auto" {
		loadOpts = append(loadOpts, inference.WithBackend(backend))
	}
	if opts.ContextOverride > 0 {
		loadOpts = append(loadOpts, inference.WithContextLen(opts.ContextOverride))
	}
	result := inference.LoadModel(strings.TrimSpace(opts.ModelPath), loadOpts...)
	if !result.OK {
		fmt.Fprintf(stderr, "%s sft: load: %s\n", cliName(), result.Error())
		return 1
	}
	model, ok := result.Value.(inference.TextModel)
	if !ok || model == nil {
		fmt.Fprintf(stderr, "%s sft: load returned %T, not inference.TextModel\n", cliName(), result.Value)
		return 1
	}
	defer model.Close()

	trainer, ok := model.(inference.SFTTrainer)
	if !ok {
		fmt.Fprintf(stderr, "%s sft: backend model %T does not implement inference.SFTTrainer\n", cliName(), model)
		return 1
	}
	training, err := trainer.TrainSFT(ctx, dataset, sftTrainingConfigFromOptions(opts))
	if err != nil {
		fmt.Fprintf(stderr, "%s sft: training: %v\n", cliName(), err)
		return 1
	}
	printSFTTrainingResult(stdout, opts, training)
	return 0
}

func sftTrainingConfigFromOptions(opts sftCommandOptions) inference.TrainingConfig {
	defaultLoRA := inference.DefaultLoRAConfig()
	return inference.TrainingConfig{
		Epochs:               opts.Epochs,
		BatchSize:            opts.BatchSize,
		GradientAccumulation: opts.GradientAccumulation,
		LearningRate:         opts.LearningRate,
		LoRA: inference.LoRAConfig{
			Rank:       opts.Rank,
			Alpha:      float32(opts.Alpha),
			TargetKeys: append([]string(nil), defaultLoRA.TargetKeys...),
			BFloat16:   defaultLoRA.BFloat16,
		},
		Labels: sftTrainingLabelsFromOptions(opts),
	}
}

func sftTrainingLabelsFromOptions(opts sftCommandOptions) map[string]string {
	runID := strings.TrimSpace(opts.RunID)
	if runID == "" {
		runID = "sft-" + time.Now().Format("20060102-150405")
	}
	labels := map[string]string{
		"backend":                      normalizeBackendName(opts.BackendName),
		"cli_contract":                 cliContractName,
		"run_id":                       runID,
		"training_stage":               "native_lora_sft_execute",
		"trainer_interface":            "inference.SFTTrainer",
		"dataset_loader":               "rocm.LoadJSONLDataset",
		"adapter_format":               "lora",
		"max_sequence_length":          strconv.Itoa(opts.MaxSequenceLength),
		"validation_samples":           strconv.Itoa(opts.ValidationSamples),
		"validation_every":             strconv.Itoa(opts.ValidationEvery),
		"eval_every":                   strconv.Itoa(opts.EvalEvery),
		"eval_max_tokens":              strconv.Itoa(opts.EvalMaxTokens),
		"eval_probes":                  strconv.Itoa(opts.EvalProbes),
		"score_cascade":                strconv.FormatBool(opts.ScoreCascade),
		"score_window":                 strconv.Itoa(opts.ScoreWindow),
		"checkpoint_every":             strconv.Itoa(opts.CheckpointEvery),
		"merge_after_training":         strconv.FormatBool(opts.MergeAfterTraining),
		"production_requires_env_gate": "false",
		"production_requires_cli_flag": "false",
		"no_python":                    "true",
	}
	if value := strings.TrimSpace(opts.ValidationPath); value != "" {
		labels["validation_path"] = value
	}
	if value := strings.TrimSpace(opts.EvalPromptsPath); value != "" {
		labels["eval_prompts_path"] = value
	}
	if value := strings.TrimSpace(opts.CheckpointDir); value != "" {
		labels["checkpoint_dir"] = value
	}
	if value := strings.TrimSpace(opts.OutputAdapterPath); value != "" {
		labels["output_adapter_path"] = value
	}
	if value := strings.TrimSpace(opts.ResumeAdapterPath); value != "" {
		labels["resume_adapter_path"] = value
	}
	if value := strings.TrimSpace(opts.CapturePath); value != "" {
		labels["capture_path"] = value
	}
	if value := strings.TrimSpace(opts.MetricsLineProtocolPath); value != "" {
		labels["metrics_line_protocol_path"] = value
	}
	if strings.TrimSpace(opts.InfluxURL) != "" {
		labels["influx_configured"] = "true"
	}
	return labels
}

func printSFTTrainingResult(stdout io.Writer, opts sftCommandOptions, result *inference.TrainingResult) {
	if result == nil {
		fmt.Fprintln(stdout, "steps 0  epochs 0  samples 0  last-loss 0.0000")
		return
	}
	fmt.Fprintf(stdout, "steps %d  epochs %d  samples %d  last-loss %.4f\n",
		result.Metrics.Step,
		result.Metrics.Epoch,
		result.Metrics.Samples,
		result.Metrics.Loss,
	)
	if result.Adapter.Path != "" {
		fmt.Fprintf(stdout, "adapter %s\n", result.Adapter.Path)
	} else if result.Adapter.Format != "" {
		fmt.Fprintf(stdout, "adapter %s\n", result.Adapter.Format)
	} else if path := strings.TrimSpace(opts.OutputAdapterPath); path != "" {
		fmt.Fprintf(stdout, "adapter %s\n", path)
	}
	for _, checkpoint := range result.Checkpoints {
		if checkpoint.URI == "" {
			continue
		}
		if checkpoint.Kind != "" {
			fmt.Fprintf(stdout, "checkpoint %s  %s\n", checkpoint.Kind, checkpoint.URI)
			continue
		}
		fmt.Fprintf(stdout, "checkpoint %s\n", checkpoint.URI)
	}
}
