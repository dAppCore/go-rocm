// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"strconv"
	"strings"

	modelgemma4 "dappco.re/go/rocm/model/gemma4"
)

type benchCommandOptions struct {
	ModelPath                string
	AssistantPath            string
	VLLMModelPath            string
	LlamaModelPath           string
	Backend                  string
	Prompt                   string
	StateName                string
	StateStorePath           string
	MaxTokens                int
	ContextLength            int
	TargetFloor              int
	TargetGoal               int
	EvidenceReleaseArtifacts bool
	EvidenceNativeMTP        bool
	EvidenceROCmTokS         float64
	EvidenceMTPTokS          float64
	EvidenceRetainedTokS     float64
	EvidenceReplayTokS       float64
	EvidenceOpenAITokS       float64
	EvidenceVLLMTokS         float64
	EvidenceLlamaTokS        float64
}

type benchPlanReport struct {
	Version     int                    `json:"version"`
	Kind        string                 `json:"kind"`
	Backend     string                 `json:"backend"`
	Command     string                 `json:"command"`
	CLIContract string                 `json:"cli_contract"`
	NoPython    bool                   `json:"no_python"`
	Model       benchModelReport       `json:"model"`
	State       benchStateReport       `json:"state"`
	Target      benchThroughputTarget  `json:"target"`
	Readiness   benchReadinessReport   `json:"readiness"`
	Evidence    *benchEvidenceReport   `json:"evidence,omitempty"`
	Commands    []benchRunCommand      `json:"commands"`
	Comparisons []benchComparisonRoute `json:"comparisons"`
	Labels      map[string]string      `json:"labels,omitempty"`
	Notes       []string               `json:"notes,omitempty"`
}

type benchModelReport struct {
	Path           string               `json:"path"`
	Assistant      string               `json:"assistant,omitempty"`
	QAT            *benchQATEntryReport `json:"qat,omitempty"`
	AssistantQAT   *benchQATEntryReport `json:"assistant_qat,omitempty"`
	Pack           *benchPackReport     `json:"pack,omitempty"`
	Supported      bool                 `json:"supported"`
	RunnableOnCard bool                 `json:"runnable_on_card"`
}

type benchQATEntryReport struct {
	CollectionID   string `json:"collection_id"`
	CollectionURL  string `json:"collection_url"`
	ModelID        string `json:"model_id"`
	Size           string `json:"size"`
	QuantMode      string `json:"quant_mode"`
	QuantSuffix    string `json:"quant_suffix"`
	Bits           int    `json:"bits"`
	QuantGroup     int    `json:"quant_group,omitempty"`
	Assistant      bool   `json:"assistant"`
	Runtime        string `json:"runtime"`
	GenerateStatus string `json:"generate_status"`
	RunnableOnCard bool   `json:"runnable_on_card"`
}

type benchPackReport struct {
	Name             string `json:"name"`
	Size             string `json:"size"`
	ModelID          string `json:"model_id"`
	LockedModelID    string `json:"locked_model_id,omitempty"`
	SourceCollection string `json:"source_collection,omitempty"`
	Bits             int    `json:"bits"`
	QuantMode        string `json:"quant_mode"`
	QuantGroup       int    `json:"quant_group,omitempty"`
	Runtime          string `json:"runtime"`
	GenerateStatus   string `json:"generate_status"`
	ProductRole      string `json:"product_role"`
	Supported        bool   `json:"supported"`
	RunnableOnCard   bool   `json:"runnable_on_card"`
	RequiresBench    bool   `json:"requires_bench,omitempty"`
	RequiresNative   bool   `json:"requires_native,omitempty"`
}

type benchStateReport struct {
	Name      string `json:"name"`
	StorePath string `json:"store_path"`
	Generate  bool   `json:"generate"`
	Server    bool   `json:"server"`
}

type benchThroughputTarget struct {
	FloorTokS int `json:"floor_tok_s"`
	GoalTokS  int `json:"goal_tok_s"`
	MaxTokens int `json:"max_tokens"`
	Context   int `json:"context,omitempty"`
}

type benchReadinessReport struct {
	ReleaseArtifacts      bool   `json:"release_artifacts"`
	StatefulGenerate      bool   `json:"stateful_generate"`
	OpenAIServer          bool   `json:"openai_server"`
	ROCmBenchmark         bool   `json:"rocm_benchmark"`
	VLLMComparison        bool   `json:"vllm_comparison"`
	LlamaCPPComparison    bool   `json:"llama_cpp_comparison"`
	ThroughputProof       bool   `json:"throughput_proof"`
	NativeAttachedDrafter string `json:"native_attached_drafter"`
	Done                  bool   `json:"done"`
}

type benchEvidenceReport struct {
	Submitted             bool     `json:"submitted"`
	ReleaseArtifacts      bool     `json:"release_artifacts"`
	NativeAttachedDrafter bool     `json:"native_attached_drafter"`
	ROCmDecodeTokS        float64  `json:"rocm_decode_tok_s,omitempty"`
	MTPAttachedTokS       float64  `json:"mtp_attached_tok_s,omitempty"`
	RetainedStateTokS     float64  `json:"retained_state_tok_s,omitempty"`
	ReplayBaselineTokS    float64  `json:"replay_baseline_tok_s,omitempty"`
	OpenAIServerTokS      float64  `json:"openai_server_tok_s,omitempty"`
	VLLMTokS              float64  `json:"vllm_tok_s,omitempty"`
	LlamaCPPTokS          float64  `json:"llama_cpp_tok_s,omitempty"`
	RetainedStateSpeedup  float64  `json:"retained_state_speedup,omitempty"`
	NativeMTPSpeedup      float64  `json:"native_mtp_speedup,omitempty"`
	ROCmVsVLLMRatio       float64  `json:"rocm_vs_vllm_ratio,omitempty"`
	ROCmVsLlamaCPPRatio   float64  `json:"rocm_vs_llama_cpp_ratio,omitempty"`
	Pass                  bool     `json:"pass"`
	Failures              []string `json:"failures,omitempty"`
}

type benchRunCommand struct {
	Name       string   `json:"name"`
	WorkingDir string   `json:"working_dir,omitempty"`
	Command    string   `json:"command"`
	Required   bool     `json:"required"`
	Runnable   bool     `json:"runnable"`
	Notes      []string `json:"notes,omitempty"`
}

type benchComparisonRoute struct {
	Name                 string   `json:"name"`
	ModelPath            string   `json:"model_path,omitempty"`
	RequiresExternalTool bool     `json:"requires_external_tool"`
	RequiresConversion   bool     `json:"requires_conversion"`
	CommandName          string   `json:"command_name"`
	Notes                []string `json:"notes,omitempty"`
}

func runBenchCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("bench"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON Gemma-4 QAT production bench plan")
	modelPath := fs.String("model", "", "Gemma-4 QAT model path or Hugging Face model ID")
	assistantPath := fs.String("assistant", "auto", "MTP-QAT assistant path, 'auto' derives it from the target, empty disables")
	vllmModelPath := fs.String("vllm-model", "", "vLLM-compatible model path; defaults to the target model")
	llamaModelPath := fs.String("llama-model", "", "llama.cpp GGUF model path for comparison")
	backendName := fs.String("backend", defaultCLIBackendName(), "registered backend name for ROCm commands")
	prompt := fs.String("prompt", defaultProductionMeasurementPrompt, "benchmark/smoke prompt")
	stateName := fs.String("state", "gemma4-qat-bench", "state session name for generate")
	stateStorePath := fs.String("state-store", "", "state store file used by generate and serve")
	maxTokens := fs.Int("max-tokens", 256, "tokens per decode measurement")
	contextLength := fs.Int("context", 48000, "context length for ROCm/vLLM/llama.cpp measurements")
	targetFloor := fs.Int("target-floor", 180, "minimum acceptable decode tokens/sec")
	targetGoal := fs.Int("target-goal", 200, "goal decode tokens/sec")
	evidenceReleaseArtifacts := fs.Bool("release-artifacts-ok", false, "evidence: release artifacts were built and dependency-checked")
	evidenceNativeMTP := fs.Bool("native-mtp-ok", false, "evidence: native attached MTP drafter execution was used")
	evidenceROCmTokS := fs.Float64("rocm-tok-s", 0, "evidence: ROCm decode throughput in tokens/sec")
	evidenceMTPTokS := fs.Float64("mtp-tok-s", 0, "evidence: native attached MTP throughput in tokens/sec")
	evidenceRetainedTokS := fs.Float64("retained-tok-s", 0, "evidence: retained-state multi-turn throughput in tokens/sec")
	evidenceReplayTokS := fs.Float64("replay-tok-s", 0, "evidence: prompt-replay baseline throughput in tokens/sec")
	evidenceOpenAITokS := fs.Float64("openai-tok-s", 0, "evidence: OpenAI server throughput in tokens/sec")
	evidenceVLLMTokS := fs.Float64("vllm-tok-s", 0, "evidence: vLLM comparison throughput in tokens/sec")
	evidenceLlamaTokS := fs.Float64("llama-tok-s", 0, "evidence: llama.cpp comparison throughput in tokens/sec")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: %s bench [flags] <model-path>\n\n", cliName())
		fmt.Fprintln(stderr, "Emit the Gemma-4 QAT/MTP-QAT production bench contract for ROCm, OpenAI server, vLLM, and llama.cpp.")
		fmt.Fprintln(stderr, "\nFlags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() > 1 {
		fmt.Fprintf(stderr, "%s bench: expected at most one model path\n", cliName())
		fs.Usage()
		return 2
	}
	if strings.TrimSpace(*modelPath) == "" && fs.NArg() == 1 {
		*modelPath = fs.Arg(0)
	}
	report, err := benchPlanReportFromOptions(benchCommandOptions{
		ModelPath:                *modelPath,
		AssistantPath:            *assistantPath,
		VLLMModelPath:            *vllmModelPath,
		LlamaModelPath:           *llamaModelPath,
		Backend:                  *backendName,
		Prompt:                   *prompt,
		StateName:                *stateName,
		StateStorePath:           *stateStorePath,
		MaxTokens:                *maxTokens,
		ContextLength:            *contextLength,
		TargetFloor:              *targetFloor,
		TargetGoal:               *targetGoal,
		EvidenceReleaseArtifacts: *evidenceReleaseArtifacts,
		EvidenceNativeMTP:        *evidenceNativeMTP,
		EvidenceROCmTokS:         *evidenceROCmTokS,
		EvidenceMTPTokS:          *evidenceMTPTokS,
		EvidenceRetainedTokS:     *evidenceRetainedTokS,
		EvidenceReplayTokS:       *evidenceReplayTokS,
		EvidenceOpenAITokS:       *evidenceOpenAITokS,
		EvidenceVLLMTokS:         *evidenceVLLMTokS,
		EvidenceLlamaTokS:        *evidenceLlamaTokS,
	})
	if err != nil {
		fmt.Fprintf(stderr, "%s bench: %v\n", cliName(), err)
		return 2
	}
	if *jsonOut {
		return writeJSON(stdout, stderr, report)
	}
	printBenchPlanReport(stdout, report)
	return 0
}

func benchPlanReportFromOptions(opts benchCommandOptions) (benchPlanReport, error) {
	modelPath := strings.TrimSpace(opts.ModelPath)
	if modelPath == "" {
		return benchPlanReport{}, fmt.Errorf("model path is required")
	}
	if opts.MaxTokens <= 0 {
		return benchPlanReport{}, fmt.Errorf("max tokens must be > 0")
	}
	if opts.ContextLength < 0 {
		return benchPlanReport{}, fmt.Errorf("context must be >= 0")
	}
	if opts.TargetFloor <= 0 || opts.TargetGoal <= 0 {
		return benchPlanReport{}, fmt.Errorf("throughput targets must be > 0")
	}
	if opts.TargetGoal < opts.TargetFloor {
		return benchPlanReport{}, fmt.Errorf("target goal must be >= target floor")
	}
	for name, value := range map[string]float64{
		"rocm-tok-s":     opts.EvidenceROCmTokS,
		"mtp-tok-s":      opts.EvidenceMTPTokS,
		"retained-tok-s": opts.EvidenceRetainedTokS,
		"replay-tok-s":   opts.EvidenceReplayTokS,
		"openai-tok-s":   opts.EvidenceOpenAITokS,
		"vllm-tok-s":     opts.EvidenceVLLMTokS,
		"llama-tok-s":    opts.EvidenceLlamaTokS,
	} {
		if value < 0 {
			return benchPlanReport{}, fmt.Errorf("%s must be >= 0", name)
		}
	}
	backend := normalizeBackendName(opts.Backend)
	if backend == "" {
		backend = defaultBackendName
	}
	prompt := strings.TrimSpace(opts.Prompt)
	if prompt == "" {
		prompt = defaultProductionMeasurementPrompt
	}
	stateName := strings.TrimSpace(opts.StateName)
	if stateName == "" {
		stateName = "gemma4-qat-bench"
	}
	stateStorePath := strings.TrimSpace(opts.StateStorePath)
	if stateStorePath == "" {
		if defaultPath, err := defaultROCmStateStorePath(); err == nil {
			stateStorePath = defaultPath
		} else {
			stateStorePath = "agent.kv"
		}
	}
	model := benchModelReport{Path: modelPath}
	qatEntry, qatOK := modelgemma4.QATCollectionEntryForModelID(modelPath)
	var targetQAT *modelgemma4.QATCollectionEntry
	if qatOK && !qatEntry.Assistant {
		targetQAT = &qatEntry
		model.QAT = benchQATEntryReportFromEntry(qatEntry)
	}
	var targetPack *modelgemma4.ProductionQuantizationPackSupport
	if pack, ok := modelgemma4.ProductionQuantizationPackAlias(modelPath); ok {
		targetPack = &pack
		model.Pack = benchPackReportFromSupport(pack)
		model.Supported = pack.Supported
		model.RunnableOnCard = pack.RunnableOnCard
		if model.QAT == nil {
			if entry, ok := benchQATEntryForProductionPack(pack); ok && !entry.Assistant {
				targetQAT = &entry
				model.QAT = benchQATEntryReportFromEntry(entry)
			}
		}
	} else if model.QAT != nil {
		model.Supported = true
		model.RunnableOnCard = model.QAT.RunnableOnCard
	}
	assistantPath := benchResolveAssistantPath(opts.AssistantPath, modelPath, targetQAT, targetPack)
	model.Assistant = assistantPath
	if assistantEntry, ok := modelgemma4.QATCollectionEntryForModelID(assistantPath); ok && assistantEntry.Assistant {
		model.AssistantQAT = benchQATEntryReportFromEntry(assistantEntry)
	}
	commands := benchCommandsForOptions(opts, backend, prompt, stateName, stateStorePath, model)
	comparisons := benchComparisonRoutes(opts, model)
	evidence := benchEvidenceFromOptions(opts, model)
	labels := benchPlanLabels(opts, backend, model, stateName, evidence)
	readiness := benchReadinessFromEvidence(evidence, opts)
	report := benchPlanReport{
		Version:     1,
		Kind:        "gemma4-qat-production-bench-plan",
		Backend:     backend,
		Command:     "bench",
		CLIContract: cliContractName,
		NoPython:    true,
		Model:       model,
		State: benchStateReport{
			Name:      stateName,
			StorePath: stateStorePath,
			Generate:  true,
			Server:    true,
		},
		Target: benchThroughputTarget{
			FloorTokS: opts.TargetFloor,
			GoalTokS:  opts.TargetGoal,
			MaxTokens: opts.MaxTokens,
			Context:   opts.ContextLength,
		},
		Readiness:   readiness,
		Evidence:    evidence,
		Commands:    commands,
		Comparisons: comparisons,
		Labels:      labels,
		Notes: []string{
			"This is the production bench contract for the Gemma-4 QAT and MTP-QAT lane; it does not mark the 180-200 tok/s goal complete until local measurements are attached.",
			"Stateful generate keeps the resolved MTP assistant on the command line and can run target-retained decode while native assistant verification is pending.",
			"The OpenAI server command keeps state conversations enabled and passes the same resolved assistant through the serve draft surface.",
			"llama.cpp comparison requires a GGUF-converted or equivalent model; MLX QAT safetensors are not a direct llama.cpp input.",
		},
	}
	if model.QAT == nil {
		report.Notes = append(report.Notes, "The target is not recognised as an mlx-community Gemma-4 QAT collection entry; QAT/MTP-QAT auto-pairing may be incomplete.")
	}
	if evidence != nil && evidence.Pass {
		report.Notes = append(report.Notes, "Attached evidence satisfies the Gemma-4 QAT production bench gate for release artifacts, stateful ROCm generation, OpenAI server, and comparison coverage.")
	} else if evidence != nil {
		report.Notes = append(report.Notes, "Attached evidence is incomplete or below threshold; keep this report as a failing production gate until the failures list is empty.")
	}
	return report, nil
}

func benchResolveAssistantPath(flagValue, modelPath string, qat *modelgemma4.QATCollectionEntry, pack *modelgemma4.ProductionQuantizationPackSupport) string {
	value := strings.TrimSpace(flagValue)
	switch value {
	case "":
		return ""
	case "auto":
	default:
		return value
	}
	if qat != nil {
		return modelgemma4.QATCollectionModelID(qat.Size, modelgemma4.DenormalizedQuantModeForCollection(qat.QuantMode), true)
	}
	if detection := resolveROCmDraft(modelPath, "auto", true); detection.Active() {
		return detection.DraftPath
	}
	if pack != nil && strings.TrimSpace(pack.Size) != "" {
		return modelgemma4.MTPAssistantPath(pack.Size, benchAssistantQuantModeForPack(*pack))
	}
	return ""
}

func benchAssistantQuantModeForPack(pack modelgemma4.ProductionQuantizationPackSupport) string {
	mode := strings.TrimSpace(pack.QuantMode)
	if strings.EqualFold(mode, "affine") && pack.Bits > 0 {
		return "q" + strconv.Itoa(pack.Bits)
	}
	return mode
}

func benchQATEntryForProductionPack(pack modelgemma4.ProductionQuantizationPackSupport) (modelgemma4.QATCollectionEntry, bool) {
	mode := benchAssistantQuantModeForPack(pack)
	if strings.TrimSpace(pack.Size) == "" || strings.TrimSpace(mode) == "" {
		return modelgemma4.QATCollectionEntry{}, false
	}
	return modelgemma4.QATCollectionEntryFor(pack.Size, mode, false)
}

func benchQATEntryReportFromEntry(entry modelgemma4.QATCollectionEntry) *benchQATEntryReport {
	return &benchQATEntryReport{
		CollectionID:   entry.CollectionID,
		CollectionURL:  entry.CollectionURL,
		ModelID:        entry.ModelID,
		Size:           entry.Size,
		QuantMode:      entry.QuantMode,
		QuantSuffix:    entry.QuantSuffix,
		Bits:           entry.Bits,
		QuantGroup:     entry.QuantGroup,
		Assistant:      entry.Assistant,
		Runtime:        entry.Runtime,
		GenerateStatus: entry.GenerateStatus,
		RunnableOnCard: entry.RunnableOnCard,
	}
}

func benchPackReportFromSupport(pack modelgemma4.ProductionQuantizationPackSupport) *benchPackReport {
	return &benchPackReport{
		Name:             pack.Name,
		Size:             pack.Size,
		ModelID:          pack.ModelID,
		LockedModelID:    pack.LockedModelID,
		SourceCollection: pack.SourceCollection,
		Bits:             pack.Bits,
		QuantMode:        pack.QuantMode,
		QuantGroup:       pack.QuantGroup,
		Runtime:          pack.Runtime,
		GenerateStatus:   pack.GenerateStatus,
		ProductRole:      pack.ProductRole,
		Supported:        pack.Supported,
		RunnableOnCard:   pack.RunnableOnCard,
		RequiresBench:    pack.RequiresBench,
		RequiresNative:   pack.RequiresNative,
	}
}

func benchEvidenceFromOptions(opts benchCommandOptions, model benchModelReport) *benchEvidenceReport {
	if !benchEvidenceSubmitted(opts) {
		return nil
	}
	evidence := &benchEvidenceReport{
		Submitted:             true,
		ReleaseArtifacts:      opts.EvidenceReleaseArtifacts,
		NativeAttachedDrafter: opts.EvidenceNativeMTP,
		ROCmDecodeTokS:        opts.EvidenceROCmTokS,
		MTPAttachedTokS:       opts.EvidenceMTPTokS,
		RetainedStateTokS:     opts.EvidenceRetainedTokS,
		ReplayBaselineTokS:    opts.EvidenceReplayTokS,
		OpenAIServerTokS:      opts.EvidenceOpenAITokS,
		VLLMTokS:              opts.EvidenceVLLMTokS,
		LlamaCPPTokS:          opts.EvidenceLlamaTokS,
	}
	failures := []string{}
	if model.QAT == nil {
		failures = append(failures, "target is not a recognized Gemma-4 QAT collection entry")
	}
	if model.AssistantQAT == nil {
		failures = append(failures, "matching Gemma-4 MTP-QAT assistant was not resolved")
	}
	if !model.Supported || !model.RunnableOnCard {
		failures = append(failures, "target model is not marked supported and runnable on this card")
	}
	if !evidence.ReleaseArtifacts {
		failures = append(failures, "release artifacts were not reported as built and dependency-checked")
	}
	if !evidence.NativeAttachedDrafter {
		failures = append(failures, "native attached MTP drafter execution was not reported as linked and measured")
	}
	failures = benchRequireTokS(failures, "ROCm decode", evidence.ROCmDecodeTokS, float64(opts.TargetFloor))
	failures = benchRequireTokS(failures, "native attached MTP decode", evidence.MTPAttachedTokS, float64(opts.TargetFloor))
	failures = benchRequireTokS(failures, "retained-state decode", evidence.RetainedStateTokS, float64(opts.TargetFloor))
	failures = benchRequireTokS(failures, "OpenAI server decode", evidence.OpenAIServerTokS, float64(opts.TargetFloor))
	if evidence.ReplayBaselineTokS <= 0 {
		failures = append(failures, "prompt-replay baseline tok/s is required to prove retained-state speedup")
	} else if evidence.RetainedStateTokS > 0 {
		evidence.RetainedStateSpeedup = evidence.RetainedStateTokS / evidence.ReplayBaselineTokS
		if evidence.RetainedStateSpeedup <= 1 {
			failures = append(failures, fmt.Sprintf("retained-state speedup %.3fx is not greater than prompt replay", evidence.RetainedStateSpeedup))
		}
	}
	if evidence.MTPAttachedTokS > 0 && evidence.RetainedStateTokS > 0 {
		evidence.NativeMTPSpeedup = evidence.MTPAttachedTokS / evidence.RetainedStateTokS
		if evidence.NativeMTPSpeedup <= 1 {
			failures = append(failures, fmt.Sprintf("native MTP speedup %.3fx is not greater than retained-state target decode", evidence.NativeMTPSpeedup))
		}
	}
	if evidence.VLLMTokS <= 0 {
		failures = append(failures, "vLLM comparison tok/s is required")
	} else if evidence.ROCmDecodeTokS > 0 {
		evidence.ROCmVsVLLMRatio = evidence.ROCmDecodeTokS / evidence.VLLMTokS
	}
	if evidence.LlamaCPPTokS <= 0 {
		failures = append(failures, "llama.cpp comparison tok/s is required")
	} else if evidence.ROCmDecodeTokS > 0 {
		evidence.ROCmVsLlamaCPPRatio = evidence.ROCmDecodeTokS / evidence.LlamaCPPTokS
	}
	evidence.Failures = failures
	evidence.Pass = len(failures) == 0
	return evidence
}

func benchEvidenceSubmitted(opts benchCommandOptions) bool {
	return opts.EvidenceReleaseArtifacts ||
		opts.EvidenceNativeMTP ||
		opts.EvidenceROCmTokS > 0 ||
		opts.EvidenceMTPTokS > 0 ||
		opts.EvidenceRetainedTokS > 0 ||
		opts.EvidenceReplayTokS > 0 ||
		opts.EvidenceOpenAITokS > 0 ||
		opts.EvidenceVLLMTokS > 0 ||
		opts.EvidenceLlamaTokS > 0
}

func benchRequireTokS(failures []string, name string, actual, floor float64) []string {
	switch {
	case actual <= 0:
		return append(failures, name+" tok/s is required")
	case actual < floor:
		return append(failures, fmt.Sprintf("%s %.3f tok/s below %.0f tok/s floor", name, actual, floor))
	default:
		return failures
	}
}

func benchReadinessFromEvidence(evidence *benchEvidenceReport, opts benchCommandOptions) benchReadinessReport {
	if evidence == nil {
		return benchReadinessReport{
			ReleaseArtifacts:      true,
			StatefulGenerate:      true,
			OpenAIServer:          true,
			ROCmBenchmark:         true,
			VLLMComparison:        true,
			LlamaCPPComparison:    true,
			ThroughputProof:       false,
			NativeAttachedDrafter: "pending",
			Done:                  false,
		}
	}
	native := "pending"
	if evidence.NativeAttachedDrafter {
		native = "linked"
	}
	return benchReadinessReport{
		ReleaseArtifacts:      evidence.ReleaseArtifacts,
		StatefulGenerate:      evidence.RetainedStateTokS >= float64(opts.TargetFloor) && evidence.ReplayBaselineTokS > 0 && evidence.RetainedStateSpeedup > 1,
		OpenAIServer:          evidence.OpenAIServerTokS >= float64(opts.TargetFloor),
		ROCmBenchmark:         evidence.ROCmDecodeTokS >= float64(opts.TargetFloor),
		VLLMComparison:        evidence.VLLMTokS > 0,
		LlamaCPPComparison:    evidence.LlamaCPPTokS > 0,
		ThroughputProof:       evidence.Pass,
		NativeAttachedDrafter: native,
		Done:                  evidence.Pass,
	}
}

func benchCommandsForOptions(opts benchCommandOptions, backend, prompt, stateName, stateStorePath string, model benchModelReport) []benchRunCommand {
	contextLength := opts.ContextLength
	maxTokens := opts.MaxTokens
	generateArgs := []string{
		cliName(), "generate",
		"-backend", backend,
		"-state", stateName,
		"-state-store", stateStorePath,
		"-draft", defaultString(model.Assistant, "auto"),
		"-kv-cache", "paged",
		"-max-tokens", strconv.Itoa(maxTokens),
		"-prompt", prompt,
	}
	if contextLength > 0 {
		generateArgs = append(generateArgs, "-context", strconv.Itoa(contextLength))
	}
	generateArgs = append(generateArgs, model.Path)
	serveArgs := []string{
		cliName(), "serve",
		"-backend", backend,
		"-model", model.Path,
		"-draft", defaultString(model.Assistant, "auto"),
		"-state-conversations=true",
		"-state-store", stateStorePath,
		"-addr", "127.0.0.1:36911",
	}
	if contextLength > 0 {
		serveArgs = append(serveArgs, "-context", strconv.Itoa(contextLength))
	}
	openAIWarmRequest := `{"model":"gemma4-qat","messages":[{"role":"user","content":` + strconv.Quote(prompt) + `}],"max_tokens":` + strconv.Itoa(maxTokens) + `}`
	openAIRetainedRequest := `{"model":"gemma4-qat","messages":[{"role":"user","content":` + strconv.Quote(prompt) + `},{"role":"assistant","content":"<assistant reply from openai-chat-warm-state>"},{"role":"user","content":"Continue with one concrete implementation note."}],"max_tokens":` + strconv.Itoa(maxTokens) + `}`
	commands := []benchRunCommand{
		{
			Name:       "release-artifacts",
			WorkingDir: ".",
			Command:    benchShellCommand("make", "release-artifacts"),
			Required:   true,
			Runnable:   true,
			Notes:      []string{"Builds lthn-amd, lthn-cuda, lthn-cpu-x86, and lthn-cpu-aarch64 release artifacts."},
		},
		{
			Name:       "rocm-stateful-generate",
			WorkingDir: ".",
			Command:    benchShellCommand(generateArgs...),
			Required:   true,
			Runnable:   true,
			Notes:      []string{"Uses the state wake/sleep path with the resolved MTP assistant; target-retained decode is the fast lane until native assistant verification is linked."},
		},
		{
			Name:       "openai-server-state",
			WorkingDir: ".",
			Command:    benchShellCommand(serveArgs...),
			Required:   true,
			Runnable:   true,
			Notes:      []string{"Starts the OpenAI-compatible server with state conversations enabled."},
		},
		{
			Name:       "openai-chat-warm-state",
			WorkingDir: ".",
			Command: benchShellCommand(
				"curl", "-sS", "http://127.0.0.1:36911/v1/chat/completions",
				"-H", "Content-Type: application/json",
				"-d", openAIWarmRequest,
			),
			Required: true,
			Runnable: true,
			Notes:    []string{"Run after openai-server-state is listening; this warms the conversation state store."},
		},
		{
			Name:       "openai-chat-retained-state",
			WorkingDir: ".",
			Command: benchShellCommand(
				"curl", "-sS", "http://127.0.0.1:36911/v1/chat/completions",
				"-H", "Content-Type: application/json",
				"-d", openAIRetainedRequest,
			),
			Required: true,
			Runnable: false,
			Notes:    []string{"Replace the placeholder assistant content with the exact warm-turn assistant message; this second turn should wake retained state instead of replaying the full prompt."},
		},
		{
			Name:       "rocm-decode-benchmark",
			WorkingDir: "go",
			Command: benchShellCommand(
				"env",
				"GO_ROCM_PRODUCTION_MODEL_PATH="+model.Path,
				"GO_ROCM_RUN_BENCHMARKS=1",
				"GO_ROCM_BENCH_TOKENS="+strconv.Itoa(maxTokens),
				benchContextEnv(contextLength),
				"go", "test", "./...", "-run", "^$", "-bench", "^BenchmarkInferenceGemma4Q4Generate$", "-benchmem", "-count=1",
			),
			Required: true,
			Runnable: true,
			Notes:    []string{"Primary decode tok/s proof for the 180-200+ tok/s target."},
		},
		{
			Name:       "rocm-retained-state-benchmark",
			WorkingDir: "go",
			Command: benchShellCommand(
				"env",
				"GO_ROCM_PRODUCTION_MODEL_PATH="+model.Path,
				"GO_ROCM_RUN_BENCHMARKS=1",
				"GO_ROCM_RUN_BOOK_BENCHMARKS=1",
				"GO_ROCM_RUN_RETAINED_BOOK_BENCHMARKS=1",
				benchContextEnv(contextLength),
				"go", "test", "./...", "-run", "^$", "-bench", "^BenchmarkInferenceGemma4Q4Book10Turn_RetainedState$", "-benchmem", "-count=1",
			),
			Required: true,
			Runnable: true,
			Notes:    []string{"Measures the retained-state multi-turn path that should improve per-turn speed."},
		},
	}
	if model.Assistant != "" {
		commands = append(commands, benchRunCommand{
			Name:       "mtp-assistant-boundary",
			WorkingDir: ".",
			Command: benchShellCommand(
				cliName(), "generate",
				"-backend", backend,
				"-draft", model.Assistant,
				"-max-tokens", strconv.Itoa(maxTokens),
				"-prompt", prompt,
				model.Path,
			),
			Required: true,
			Runnable: true,
			Notes:    []string{"Tracks the attached-drafter boundary; target-retained decode is runnable when the target path is linked, but the done gate still requires native ROCm MTP evidence."},
		})
	}
	commands = append(commands, benchComparisonCommands(opts, model, contextLength)...)
	return commands
}

func benchComparisonCommands(opts benchCommandOptions, model benchModelReport, contextLength int) []benchRunCommand {
	vllmModel := strings.TrimSpace(opts.VLLMModelPath)
	if vllmModel == "" {
		vllmModel = model.Path
	}
	llamaModel := strings.TrimSpace(opts.LlamaModelPath)
	if llamaModel == "" {
		llamaModel = "<gguf-equivalent>"
	}
	vllmArgs := []string{"vllm", "serve", vllmModel, "--host", "127.0.0.1", "--port", "8000", "--dtype", "auto"}
	if contextLength > 0 {
		vllmArgs = append(vllmArgs, "--max-model-len", strconv.Itoa(contextLength))
	}
	llamaArgs := []string{"llama-server", "-m", llamaModel, "--host", "127.0.0.1", "--port", "8080"}
	if contextLength > 0 {
		llamaArgs = append(llamaArgs, "-c", strconv.Itoa(contextLength))
	}
	return []benchRunCommand{
		{
			Name:       "vllm-server-comparison",
			WorkingDir: ".",
			Command:    benchShellCommand(vllmArgs...),
			Required:   true,
			Runnable:   true,
			Notes:      []string{"Use -vllm-model for a vLLM-loadable equivalent if the MLX QAT target is not accepted directly."},
		},
		{
			Name:       "llama-cpp-server-comparison",
			WorkingDir: ".",
			Command:    benchShellCommand(llamaArgs...),
			Required:   true,
			Runnable:   strings.TrimSpace(opts.LlamaModelPath) != "",
			Notes:      []string{"Requires -llama-model pointing at a GGUF-converted or equivalent model."},
		},
	}
}

func benchComparisonRoutes(opts benchCommandOptions, model benchModelReport) []benchComparisonRoute {
	vllmModel := strings.TrimSpace(opts.VLLMModelPath)
	if vllmModel == "" {
		vllmModel = model.Path
	}
	llamaModel := strings.TrimSpace(opts.LlamaModelPath)
	return []benchComparisonRoute{
		{
			Name:                 "vllm",
			ModelPath:            vllmModel,
			RequiresExternalTool: true,
			RequiresConversion:   strings.Contains(strings.ToLower(vllmModel), "mlx-community/"),
			CommandName:          "vllm",
			Notes:                []string{"Compare API-compatible serving latency and decode throughput against ROCm."},
		},
		{
			Name:                 "llama.cpp",
			ModelPath:            llamaModel,
			RequiresExternalTool: true,
			RequiresConversion:   true,
			CommandName:          "llama-server",
			Notes:                []string{"Use a GGUF-converted or equivalent model; MLX QAT safetensors are not direct llama.cpp input."},
		},
	}
}

func benchPlanLabels(opts benchCommandOptions, backend string, model benchModelReport, stateName string, evidence *benchEvidenceReport) map[string]string {
	current := currentCompileReport().Current
	labels := map[string]string{
		"backend":                       backend,
		"cli_contract":                  cliContractName,
		"active_backend":                defaultString(current.Labels["active_backend"], defaultBackendName),
		"binary_default_backend":        defaultCLIBackendName(),
		"runtime_dispatch_status":       defaultString(current.Labels["runtime_dispatch_status"], "unknown"),
		"runtime_lane":                  defaultString(current.Labels["runtime_lane"], defaultBackendName),
		"bench_stage":                   "gemma4_qat_production_bench_contract",
		"measurement_prompt_profile":    productionMeasurementPromptProfileFor(opts.Prompt),
		"target_decode_tok_s_floor":     strconv.Itoa(opts.TargetFloor),
		"target_decode_tok_s_goal":      strconv.Itoa(opts.TargetGoal),
		"stateful_generate_required":    "true",
		"openai_server_required":        "true",
		"comparison_vllm_required":      "true",
		"comparison_llama_cpp_required": "true",
		"release_artifacts_required":    "true",
		"production_requires_env_gate":  "false",
		"production_requires_cli_flag":  "false",
		"no_python":                     "true",
		"state_name":                    stateName,
		"native_attached_drafter":       "pending",
	}
	if model.QAT != nil {
		labels["qat_collection"] = model.QAT.CollectionID
		labels["qat_model"] = model.QAT.ModelID
		labels["qat_size"] = model.QAT.Size
		labels["qat_quant_mode"] = model.QAT.QuantMode
	}
	if model.AssistantQAT != nil {
		labels["mtp_qat_collection"] = model.AssistantQAT.CollectionID
		labels["mtp_qat_assistant"] = model.AssistantQAT.ModelID
	}
	if model.Pack != nil {
		labels["production_quant_pack"] = model.Pack.Name
		labels["production_quant_runtime"] = model.Pack.Runtime
		labels["production_quant_generate_status"] = model.Pack.GenerateStatus
	}
	if evidence != nil {
		labels["bench_evidence_submitted"] = strconv.FormatBool(evidence.Submitted)
		labels["bench_evidence_pass"] = strconv.FormatBool(evidence.Pass)
		labels["release_artifacts_ok"] = strconv.FormatBool(evidence.ReleaseArtifacts)
		labels["native_mtp_ok"] = strconv.FormatBool(evidence.NativeAttachedDrafter)
		labels["rocm_decode_tok_s"] = benchMetricLabel(evidence.ROCmDecodeTokS)
		labels["native_mtp_tok_s"] = benchMetricLabel(evidence.MTPAttachedTokS)
		labels["retained_state_tok_s"] = benchMetricLabel(evidence.RetainedStateTokS)
		labels["replay_baseline_tok_s"] = benchMetricLabel(evidence.ReplayBaselineTokS)
		labels["openai_server_tok_s"] = benchMetricLabel(evidence.OpenAIServerTokS)
		labels["vllm_tok_s"] = benchMetricLabel(evidence.VLLMTokS)
		labels["llama_cpp_tok_s"] = benchMetricLabel(evidence.LlamaCPPTokS)
		labels["retained_state_speedup"] = benchMetricLabel(evidence.RetainedStateSpeedup)
		labels["native_mtp_speedup"] = benchMetricLabel(evidence.NativeMTPSpeedup)
		labels["rocm_vs_vllm_ratio"] = benchMetricLabel(evidence.ROCmVsVLLMRatio)
		labels["rocm_vs_llama_cpp_ratio"] = benchMetricLabel(evidence.ROCmVsLlamaCPPRatio)
		if len(evidence.Failures) > 0 {
			labels["bench_evidence_failures"] = strings.Join(evidence.Failures, "; ")
		}
	}
	return labels
}

func printBenchPlanReport(stdout io.Writer, report benchPlanReport) {
	fmt.Fprintln(stdout, "Gemma-4 QAT production bench plan")
	fmt.Fprintf(stdout, "  model: %s\n", report.Model.Path)
	if report.Model.Assistant != "" {
		fmt.Fprintf(stdout, "  assistant: %s\n", report.Model.Assistant)
	}
	if report.Model.QAT != nil {
		fmt.Fprintf(stdout, "  qat: %s %s (%s)\n", report.Model.QAT.Size, report.Model.QAT.QuantMode, report.Model.QAT.CollectionID)
	}
	fmt.Fprintf(stdout, "  target: %d-%d tok/s, max_tokens=%d\n", report.Target.FloorTokS, report.Target.GoalTokS, report.Target.MaxTokens)
	fmt.Fprintf(stdout, "  state: %s (%s)\n", report.State.Name, report.State.StorePath)
	fmt.Fprintln(stdout, "  commands:")
	for _, command := range report.Commands {
		status := "required"
		if !command.Runnable {
			status = "pending"
		}
		if command.WorkingDir != "" {
			fmt.Fprintf(stdout, "    %s [%s] (%s): %s\n", command.Name, command.WorkingDir, status, command.Command)
		} else {
			fmt.Fprintf(stdout, "    %s (%s): %s\n", command.Name, status, command.Command)
		}
	}
	if report.Evidence != nil {
		fmt.Fprintf(stdout, "  evidence: pass=%t rocm=%.1f retained=%.1f openai=%.1f vllm=%.1f llama.cpp=%.1f\n",
			report.Evidence.Pass,
			report.Evidence.ROCmDecodeTokS,
			report.Evidence.RetainedStateTokS,
			report.Evidence.OpenAIServerTokS,
			report.Evidence.VLLMTokS,
			report.Evidence.LlamaCPPTokS,
		)
		if report.Evidence.RetainedStateSpeedup > 0 {
			fmt.Fprintf(stdout, "  state speedup: %.3fx\n", report.Evidence.RetainedStateSpeedup)
		}
		if report.Evidence.MTPAttachedTokS > 0 {
			fmt.Fprintf(stdout, "  native MTP: %.1f tok/s\n", report.Evidence.MTPAttachedTokS)
		}
		if report.Evidence.NativeMTPSpeedup > 0 {
			fmt.Fprintf(stdout, "  native MTP speedup: %.3fx\n", report.Evidence.NativeMTPSpeedup)
		}
		for _, failure := range report.Evidence.Failures {
			fmt.Fprintf(stdout, "  failure: %s\n", failure)
		}
	}
	fmt.Fprintf(stdout, "  done: %t (throughput proof pending=%t)\n", report.Readiness.Done, !report.Readiness.ThroughputProof)
}

func benchMetricLabel(value float64) string {
	if value <= 0 {
		return ""
	}
	return strconv.FormatFloat(value, 'f', 3, 64)
}

func benchContextEnv(contextLength int) string {
	if contextLength <= 0 {
		return "GO_ROCM_BENCH_CONTEXT_LEN=48000"
	}
	return "GO_ROCM_BENCH_CONTEXT_LEN=" + strconv.Itoa(contextLength)
}

func benchShellCommand(args ...string) string {
	parts := make([]string, 0, len(args))
	for _, arg := range args {
		if strings.TrimSpace(arg) == "" {
			parts = append(parts, "''")
			continue
		}
		parts = append(parts, benchShellQuote(arg))
	}
	return strings.Join(parts, " ")
}

func benchShellQuote(arg string) string {
	if arg == "" {
		return "''"
	}
	if strings.ContainsAny(arg, " \t\n'\"\\$`!*?[]{}()<>|&;") {
		return "'" + strings.ReplaceAll(arg, "'", "'\"'\"'") + "'"
	}
	return arg
}
