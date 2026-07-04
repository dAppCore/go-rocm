// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"

	"dappco.re/go/inference"
	rocm "dappco.re/go/rocm"
)

type ssdRecipesReport struct {
	Version      int                   `json:"version"`
	Kind         string                `json:"kind"`
	Backend      string                `json:"backend"`
	Command      string                `json:"command"`
	CLIContract  string                `json:"cli_contract"`
	NoPython     bool                  `json:"no_python"`
	TrainDefault ssdRecipeTrainConfig  `json:"train_default"`
	EvalDefault  ssdRecipeEvalConfig   `json:"eval_default"`
	Recipes      []ssdRecipeDescriptor `json:"recipes"`
	Notes        []string              `json:"notes,omitempty"`
}

type ssdRecipeDescriptor struct {
	Name          string               `json:"name"`
	Model         string               `json:"model"`
	Dataset       string               `json:"dataset,omitempty"`
	DatasetConfig string               `json:"dataset_config,omitempty"`
	DatasetSplit  string               `json:"dataset_split,omitempty"`
	Train         ssdRecipeTrainConfig `json:"train"`
	Eval          ssdRecipeEvalConfig  `json:"eval"`
	Notes         []string             `json:"notes,omitempty"`
}

type ssdRecipeTrainConfig struct {
	SampleMaxTokens       int     `json:"sample_max_tokens,omitempty"`
	SampleTemperature     float32 `json:"sample_temperature,omitempty"`
	SampleTopK            int     `json:"sample_top_k,omitempty"`
	SampleTopP            float32 `json:"sample_top_p,omitempty"`
	SampleMinP            float32 `json:"sample_min_p,omitempty"`
	RepetitionPenalty     float32 `json:"repetition_penalty,omitempty"`
	FilterShortestPercent float32 `json:"filter_shortest_percent,omitempty"`
	DecodeTemperature     float32 `json:"decode_temperature,omitempty"`
}

type ssdRecipeEvalConfig struct {
	Benchmark string                  `json:"benchmark,omitempty"`
	NRepeat   int                     `json:"n_repeat,omitempty"`
	Generate  ssdRecipeGenerateConfig `json:"generate"`
	Seeds     []uint64                `json:"seeds,omitempty"`
}

type ssdRecipeGenerateConfig struct {
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float32 `json:"temperature,omitempty"`
	TopP        float32 `json:"top_p,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	MinP        float32 `json:"min_p,omitempty"`
}

type ssdEvalPlanReport struct {
	Version       int                 `json:"version"`
	Kind          string              `json:"kind"`
	Backend       string              `json:"backend"`
	Command       string              `json:"command"`
	CLIContract   string              `json:"cli_contract"`
	NoPython      bool                `json:"no_python"`
	SamplePath    string              `json:"sample_path,omitempty"`
	OutputPath    string              `json:"output_path,omitempty"`
	LiveCodeBench bool                `json:"livecodebench_v6,omitempty"`
	Samples       int                 `json:"samples"`
	Config        ssdRecipeEvalConfig `json:"config"`
	Notes         []string            `json:"notes,omitempty"`
}

type ssdCommandOptions struct {
	BackendName             string
	ModelPath               string
	DataPath                string
	KernelPath              string
	SampleMaxTokens         int
	SampleTemperature       float32
	SampleTopK              int
	SampleTopP              float32
	SampleMinP              float32
	RepetitionPenalty       float32
	FilterShortestPercent   float32
	ScoreSamples            bool
	RunID                   string
	MetricsLineProtocolPath string
	InfluxURL               string
	InfluxToken             string
	CapturePath             string
	CheckpointDir           string
	ContextOverride         int
}

type ssdPlanReport struct {
	Version       int               `json:"version"`
	Kind          string            `json:"kind"`
	Backend       string            `json:"backend"`
	Command       string            `json:"command"`
	CLIContract   string            `json:"cli_contract"`
	NoPython      bool              `json:"no_python"`
	ModelPath     string            `json:"model_path"`
	DataPath      string            `json:"data_path"`
	KernelPath    string            `json:"kernel_path,omitempty"`
	CheckpointDir string            `json:"checkpoint_dir,omitempty"`
	CapturePath   string            `json:"capture_path,omitempty"`
	RunID         string            `json:"run_id,omitempty"`
	Metrics       sftMetricsPlan    `json:"metrics,omitempty"`
	Config        ssdPlanConfig     `json:"config"`
	Dataset       sftDatasetSummary `json:"dataset"`
	Labels        map[string]string `json:"labels,omitempty"`
	Notes         []string          `json:"notes,omitempty"`
}

type ssdPlanConfig struct {
	Sample       ssdRecipeTrainConfig `json:"sample"`
	ScoreSamples bool                 `json:"score_samples"`
}

func ssdRecipesReportFromROCmDefaults() ssdRecipesReport {
	return ssdRecipesReport{
		Version:      1,
		Kind:         "simple-self-distillation-recipes",
		Backend:      defaultBackendName,
		Command:      "ssd-recipes",
		CLIContract:  cliContractName,
		NoPython:     true,
		TrainDefault: ssdRecipeTrainConfigFromROCmConfig(rocm.DefaultSimpleSelfDistillationConfig()),
		EvalDefault:  ssdRecipeEvalConfigFromROCmConfig(rocm.DefaultSimpleSelfDistillationCodeBenchmarkConfig()),
		Recipes:      ssdRecipeDescriptorsFromROCmRecipes(rocm.SimpleSelfDistillationRecipes()),
		Notes: []string{
			"The go-rocm SSD command generates the frozen-model trace; SFT/AdamW refinement is a separate command/helper step.",
			"Use this report as the source manifest for docs/runtime SSD parity artifacts before heavyweight recipe runs are reproduced locally.",
		},
	}
}

func ssdPlanReportFromOptions(opts ssdCommandOptions) (ssdPlanReport, error) {
	if err := validateSSDCommandOptions(opts); err != nil {
		return ssdPlanReport{}, err
	}
	backend := normalizeBackendName(opts.BackendName)
	dataset, err := summarizeSFTDataset(opts.DataPath)
	if err != nil {
		return ssdPlanReport{}, fmt.Errorf("prompt dataset: %w", err)
	}
	report := ssdPlanReport{
		Version:       1,
		Kind:          "native-simple-self-distillation-trace-plan",
		Backend:       backend,
		Command:       "ssd",
		CLIContract:   cliContractName,
		NoPython:      true,
		ModelPath:     strings.TrimSpace(opts.ModelPath),
		DataPath:      strings.TrimSpace(opts.DataPath),
		KernelPath:    strings.TrimSpace(opts.KernelPath),
		CheckpointDir: strings.TrimSpace(opts.CheckpointDir),
		CapturePath:   strings.TrimSpace(opts.CapturePath),
		RunID:         strings.TrimSpace(opts.RunID),
		Metrics:       sftMetricsPlanFromOptions(sftCommandOptions{MetricsLineProtocolPath: opts.MetricsLineProtocolPath, InfluxURL: opts.InfluxURL, InfluxToken: opts.InfluxToken}),
		Config:        ssdPlanConfigFromOptions(opts),
		Dataset:       dataset,
		Labels:        ssdPlanLabels(opts),
		Notes: []string{
			"RunModelSimpleSelfDistillation owns the native ROCm frozen-model trace; do not run SFT inside SSD.",
			"Refine the captured trace in the lab, then run the separate sft command on the curated artifact.",
			"The default CLI path loads the model and writes the scored trace; -json prints this plan without executing.",
		},
	}
	return report, nil
}

func validateSSDCommandOptions(opts ssdCommandOptions) error {
	switch {
	case strings.TrimSpace(opts.ModelPath) == "":
		return fmt.Errorf("model path is required")
	case strings.TrimSpace(opts.DataPath) == "":
		return fmt.Errorf("prompt data path is required")
	case opts.SampleMaxTokens <= 0:
		return fmt.Errorf("sample max tokens must be > 0")
	case opts.SampleTemperature <= 0 || opts.SampleTemperature == 1 || math.IsNaN(float64(opts.SampleTemperature)) || math.IsInf(float64(opts.SampleTemperature), 0):
		return fmt.Errorf("sample temperature must be positive, finite, and non-unit")
	case opts.SampleTopK < 0:
		return fmt.Errorf("sample top-k must be >= 0")
	case opts.SampleTopP < 0 || opts.SampleTopP > 1 || math.IsNaN(float64(opts.SampleTopP)) || math.IsInf(float64(opts.SampleTopP), 0):
		return fmt.Errorf("sample top-p must be in [0, 1]")
	case opts.SampleMinP < 0 || opts.SampleMinP > 1 || math.IsNaN(float64(opts.SampleMinP)) || math.IsInf(float64(opts.SampleMinP), 0):
		return fmt.Errorf("sample min-p must be in [0, 1]")
	case opts.RepetitionPenalty < 0 || math.IsNaN(float64(opts.RepetitionPenalty)) || math.IsInf(float64(opts.RepetitionPenalty), 0):
		return fmt.Errorf("repetition penalty must be finite and non-negative")
	case opts.FilterShortestPercent < 0 || opts.FilterShortestPercent > 100 || math.IsNaN(float64(opts.FilterShortestPercent)) || math.IsInf(float64(opts.FilterShortestPercent), 0):
		return fmt.Errorf("filter shortest percent must be in [0, 100]")
	case opts.ContextOverride < 0:
		return fmt.Errorf("context override must be >= 0")
	default:
		return nil
	}
}

func ssdPlanConfigFromOptions(opts ssdCommandOptions) ssdPlanConfig {
	cfg := rocm.DefaultSimpleSelfDistillationConfig()
	cfg.SampleMaxTokens = opts.SampleMaxTokens
	cfg.SampleTemperature = opts.SampleTemperature
	cfg.SampleTopK = opts.SampleTopK
	cfg.SampleTopP = opts.SampleTopP
	cfg.SampleMinP = opts.SampleMinP
	cfg.RepetitionPenalty = opts.RepetitionPenalty
	cfg.FilterShortestPct = opts.FilterShortestPercent
	return ssdPlanConfig{
		Sample:       ssdRecipeTrainConfigFromROCmConfig(cfg),
		ScoreSamples: opts.ScoreSamples,
	}
}

func ssdPlanLabels(opts ssdCommandOptions) map[string]string {
	backend := normalizeBackendName(opts.BackendName)
	labels := map[string]string{
		"backend":                      backend,
		"cli_contract":                 cliContractName,
		"training_stage":               "native_simple_self_distillation_trace_plan",
		"dataset_loader":               "rocm.LoadJSONLDataset",
		"ssd_runner":                   "RunModelSimpleSelfDistillation",
		"ssd_stops_at":                 "scored_trace",
		"next_training_command":        "sft",
		"model_loader_interface":       "linked",
		"native_ssd_runner":            "linked",
		"score_samples":                strconv.FormatBool(opts.ScoreSamples),
		"sample_max_tokens":            strconv.Itoa(opts.SampleMaxTokens),
		"sample_temperature":           formatSSDFloat32(opts.SampleTemperature),
		"sample_top_k":                 strconv.Itoa(opts.SampleTopK),
		"sample_top_p":                 formatSSDFloat32(opts.SampleTopP),
		"sample_min_p":                 formatSSDFloat32(opts.SampleMinP),
		"repetition_penalty":           formatSSDFloat32(opts.RepetitionPenalty),
		"filter_shortest_percent":      formatSSDFloat32(opts.FilterShortestPercent),
		"production_requires_env_gate": "false",
		"production_requires_cli_flag": "false",
		"no_python":                    "true",
	}
	if strings.TrimSpace(opts.KernelPath) != "" {
		labels["kernel_prefix"] = strings.TrimSpace(opts.KernelPath)
	}
	if strings.TrimSpace(opts.CapturePath) != "" {
		labels["capture_path"] = strings.TrimSpace(opts.CapturePath)
	}
	return labels
}

func formatSSDFloat32(value float32) string {
	return strconv.FormatFloat(float64(value), 'f', -1, 32)
}

func ssdRecipeDescriptorsFromROCmRecipes(recipes []rocm.SimpleSelfDistillationRecipe) []ssdRecipeDescriptor {
	descriptors := make([]ssdRecipeDescriptor, 0, len(recipes))
	for _, recipe := range recipes {
		descriptors = append(descriptors, ssdRecipeDescriptor{
			Name:          recipe.Name,
			Model:         recipe.Model,
			Dataset:       recipe.Dataset,
			DatasetConfig: recipe.DatasetConfig,
			DatasetSplit:  recipe.DatasetSplit,
			Train:         ssdRecipeTrainConfigFromROCmConfig(recipe.Train),
			Eval:          ssdRecipeEvalConfigFromROCmConfig(recipe.Eval),
			Notes:         append([]string(nil), recipe.Notes...),
		})
	}
	return descriptors
}

func ssdRecipeTrainConfigFromROCmConfig(cfg rocm.SimpleSelfDistillationConfig) ssdRecipeTrainConfig {
	return ssdRecipeTrainConfig{
		SampleMaxTokens:       cfg.SampleMaxTokens,
		SampleTemperature:     cfg.SampleTemperature,
		SampleTopK:            cfg.SampleTopK,
		SampleTopP:            cfg.SampleTopP,
		SampleMinP:            cfg.SampleMinP,
		RepetitionPenalty:     cfg.RepetitionPenalty,
		FilterShortestPercent: cfg.FilterShortestPct,
		DecodeTemperature:     cfg.DecodeTemperature,
	}
}

func ssdRecipeEvalConfigFromROCmConfig(cfg rocm.SimpleSelfDistillationCodeBenchmarkConfig) ssdRecipeEvalConfig {
	return ssdRecipeEvalConfig{
		Benchmark: cfg.Benchmark,
		NRepeat:   cfg.NRepeat,
		Generate:  ssdRecipeGenerateConfigFromInference(cfg.Generate),
		Seeds:     append([]uint64(nil), cfg.Seeds...),
	}
}

func ssdRecipeGenerateConfigFromInference(cfg inference.GenerateConfig) ssdRecipeGenerateConfig {
	return ssdRecipeGenerateConfig{
		MaxTokens:   cfg.MaxTokens,
		Temperature: cfg.Temperature,
		TopP:        cfg.TopP,
		TopK:        cfg.TopK,
		MinP:        cfg.MinP,
	}
}

func printSSDPlanReport(stdout io.Writer, report ssdPlanReport) {
	fmt.Fprintln(stdout, "native simple self-distillation trace plan")
	fmt.Fprintf(stdout, "  model: %s\n", report.ModelPath)
	fmt.Fprintf(stdout, "  prompts: %d sample(s) from %s", report.Dataset.Samples, report.Dataset.Path)
	if len(report.Dataset.Formats) > 0 {
		fmt.Fprintf(stdout, " (%s)", strings.Join(report.Dataset.Formats, ","))
	}
	fmt.Fprintln(stdout)
	fmt.Fprintf(stdout, "  sample: max_tokens=%d temperature=%.1f top_p=%.2f top_k=%d min_p=%.2f repetition_penalty=%.2f filter_shortest_percent=%.0f score_samples=%t\n",
		report.Config.Sample.SampleMaxTokens,
		report.Config.Sample.SampleTemperature,
		report.Config.Sample.SampleTopP,
		report.Config.Sample.SampleTopK,
		report.Config.Sample.SampleMinP,
		report.Config.Sample.RepetitionPenalty,
		report.Config.Sample.FilterShortestPercent,
		report.Config.ScoreSamples,
	)
	fmt.Fprintln(stdout, "  next: refine trace in the lab, then run sft on the curated artifact")
	fmt.Fprintf(stdout, "  status: model_loader_interface=%s\n", report.Labels["model_loader_interface"])
}

func printSSDRecipesReport(stdout io.Writer, report ssdRecipesReport) {
	fmt.Fprintln(stdout, "simple self-distillation recipes")
	fmt.Fprintf(stdout, "  data-gen: max_tokens=%d temperature=%.1f top_p=%.1f top_k=%d repetition_penalty=%.1f filter_shortest_percent=%.0f\n",
		report.TrainDefault.SampleMaxTokens,
		report.TrainDefault.SampleTemperature,
		report.TrainDefault.SampleTopP,
		report.TrainDefault.SampleTopK,
		report.TrainDefault.RepetitionPenalty,
		report.TrainDefault.FilterShortestPercent,
	)
	fmt.Fprintf(stdout, "  eval: %s n_repeat=%d max_tokens=%d temperature=%.1f top_p=%.2f top_k=%d\n",
		report.EvalDefault.Benchmark,
		report.EvalDefault.NRepeat,
		report.EvalDefault.Generate.MaxTokens,
		report.EvalDefault.Generate.Temperature,
		report.EvalDefault.Generate.TopP,
		report.EvalDefault.Generate.TopK,
	)
	for _, recipe := range report.Recipes {
		fmt.Fprintf(stdout, "  %s: %s (%s/%s)\n", recipe.Name, recipe.Model, recipe.Dataset, recipe.DatasetConfig)
	}
}

func ssdEvalPlanReportFromFlags(samplesPath, outputPath string, liveCodeBenchV6 bool, maxSamples, nRepeat, maxTokens int, temperature, topP float64, topK int, minP float64, samplingParams string) (ssdEvalPlanReport, error) {
	samplesPath = strings.TrimSpace(samplesPath)
	if samplesPath == "" {
		return ssdEvalPlanReport{}, fmt.Errorf("samples path is required")
	}
	cfg := rocm.DefaultSimpleSelfDistillationCodeBenchmarkConfig()
	cfg.OutputPath = strings.TrimSpace(outputPath)
	if nRepeat > 0 {
		cfg.NRepeat = nRepeat
	}
	if err := applySSDEvalSamplingParams(&cfg, samplingParams); err != nil {
		return ssdEvalPlanReport{}, err
	}
	if maxTokens > 0 {
		cfg.Generate.MaxTokens = maxTokens
	}
	if temperature >= 0 {
		cfg.Generate.Temperature = float32(temperature)
	}
	if topP >= 0 {
		cfg.Generate.TopP = float32(topP)
	}
	if topK >= 0 {
		cfg.Generate.TopK = topK
	}
	if minP >= 0 {
		cfg.Generate.MinP = float32(minP)
	}
	samples, err := loadSSDEvalSamples(samplesPath, liveCodeBenchV6)
	if err != nil {
		return ssdEvalPlanReport{}, err
	}
	if maxSamples > 0 && len(samples) > maxSamples {
		samples = samples[:maxSamples]
	}
	return ssdEvalPlanReport{
		Version:       1,
		Kind:          "simple-self-distillation-eval-plan",
		Backend:       defaultBackendName,
		Command:       "ssd-eval",
		CLIContract:   cliContractName,
		NoPython:      true,
		SamplePath:    samplesPath,
		OutputPath:    cfg.OutputPath,
		LiveCodeBench: liveCodeBenchV6,
		Samples:       len(samples),
		Config:        ssdRecipeEvalConfigFromROCmConfig(cfg),
		Notes: []string{
			"SSD eval consumes trace/candidate samples separately from the SSD trace-generation command.",
			"LiveCodeBench code execution remains caller-supplied through the benchmark runner.",
		},
	}, nil
}

func loadSSDEvalSamples(path string, liveCodeBenchV6 bool) ([]rocm.SimpleSelfDistillationCodeBenchmarkSample, error) {
	if liveCodeBenchV6 {
		return rocm.LoadSimpleSelfDistillationLiveCodeBenchV6JSONLFile(path)
	}
	return rocm.LoadSimpleSelfDistillationCodeBenchmarkJSONLFile(path)
}

func applySSDEvalSamplingParams(cfg *rocm.SimpleSelfDistillationCodeBenchmarkConfig, raw string) error {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		key, value, ok := strings.Cut(part, "=")
		if !ok {
			return fmt.Errorf("invalid sampling param %q", part)
		}
		key = strings.ReplaceAll(strings.TrimSpace(key), "-", "_")
		value = strings.TrimSpace(value)
		switch key {
		case "temperature", "temp":
			parsed, err := parseSSDEvalFloat32(value)
			if err != nil {
				return fmt.Errorf("invalid temperature %q", value)
			}
			cfg.Generate.Temperature = parsed
		case "top_p":
			parsed, err := parseSSDEvalFloat32(value)
			if err != nil {
				return fmt.Errorf("invalid top_p %q", value)
			}
			cfg.Generate.TopP = parsed
		case "top_k":
			parsed, err := strconv.Atoi(value)
			if err != nil {
				return fmt.Errorf("invalid top_k %q", value)
			}
			cfg.Generate.TopK = parsed
		case "min_p":
			parsed, err := parseSSDEvalFloat32(value)
			if err != nil {
				return fmt.Errorf("invalid min_p %q", value)
			}
			cfg.Generate.MinP = parsed
		case "max_tokens":
			parsed, err := strconv.Atoi(value)
			if err != nil {
				return fmt.Errorf("invalid max_tokens %q", value)
			}
			cfg.Generate.MaxTokens = parsed
		default:
			return fmt.Errorf("unknown sampling param %q", key)
		}
	}
	return nil
}

func parseSSDEvalFloat32(value string) (float32, error) {
	parsed, err := strconv.ParseFloat(value, 32)
	if err != nil {
		return 0, err
	}
	return float32(parsed), nil
}

func printSSDEvalPlanReport(stdout io.Writer, report ssdEvalPlanReport) {
	fmt.Fprintln(stdout, "simple self-distillation eval plan")
	fmt.Fprintf(stdout, "  samples: %d\n", report.Samples)
	fmt.Fprintf(stdout, "  benchmark: %s n_repeat=%d max_tokens=%d temperature=%.3g top_p=%.3g top_k=%d\n",
		report.Config.Benchmark,
		report.Config.NRepeat,
		report.Config.Generate.MaxTokens,
		report.Config.Generate.Temperature,
		report.Config.Generate.TopP,
		report.Config.Generate.TopK,
	)
}
