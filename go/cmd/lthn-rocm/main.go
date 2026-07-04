// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"hash/fnv"
	"io"
	"iter"
	"math"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
	rocm "dappco.re/go/rocm"
	"dappco.re/go/rocm/memorypretrain"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	commandName = filepath.Base(os.Args[0])
	os.Exit(runCommand(ctx, os.Args[1:], os.Stdout, os.Stderr))
}

const (
	defaultCLIName     = "lthn-rocm"
	defaultBackendName = "rocm"
	cliContractName    = "reactive-inference-v1"
)

var commandName = defaultCLIName

type statelessChatModel interface {
	ChatStateless(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token]
}

func cliName() string {
	if strings.TrimSpace(commandName) == "" {
		return defaultCLIName
	}
	return commandName
}

func cliCommandName(command string) string {
	if command == "" {
		return cliName()
	}
	return cliName() + " " + command
}

func cliFlagWasSet(fs *flag.FlagSet, name string) bool {
	seen := false
	fs.Visit(func(flag *flag.Flag) {
		if flag != nil && flag.Name == name {
			seen = true
		}
	})
	return seen
}

func runCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		printUsage(stdout)
		return 0
	}
	switch args[0] {
	case "menubar":
		return runMenubarCommand(args[1:], stdout, stderr)
	case "discover":
		return runDiscoverCommand(ctx, args[1:], stdout, stderr)
	case "pack":
		return runPackCommand(ctx, args[1:], stdout, stderr)
	case "reactive":
		return runReactiveCommand(ctx, args[1:], stdout, stderr)
	case "bench":
		return runBenchCommand(args[1:], stdout, stderr)
	case "ssd-recipes":
		return runSSDRecipesCommand(args[1:], stdout, stderr)
	case "ssd-eval":
		return runSSDEvalCommand(args[1:], stdout, stderr)
	case "memory-pretrain-build":
		return runMemoryPretrainBuildCommand(ctx, args[1:], stdout, stderr)
	case "serve":
		return runServeCommand(ctx, args[1:], stdout, stderr)
	case "daemon":
		return runDaemonCommand(ctx, args[1:], stdout, stderr)
	case "generate":
		return runGenerateCommand(ctx, args[1:], stdout, stderr)
	case "sft":
		return runSFTCommand(ctx, args[1:], stdout, stderr)
	case "fuse":
		return runFuseCommand(ctx, args[1:], stdout, stderr)
	case "ssd":
		return runSSDCommand(ctx, args[1:], stdout, stderr)
	case "tune":
		return runTuneCommand(ctx, args[1:], stdout, stderr)
	case "diffuse":
		return runDiffuseCommand(ctx, args[1:], stdout, stderr)
	case "audio":
		return runAudioCommand(ctx, args[1:], stdout, stderr)
	case "vision":
		return runVisionCommand(ctx, args[1:], stdout, stderr)
	case "ebook":
		return runEbookCommand(args[1:], stdout, stderr)
	case "slice":
		return runSliceCommand(ctx, args[1:], stdout, stderr)
	case "state-pack":
		return runStatePackCommand(ctx, args[1:], stdout, stderr)
	case "-h", "--help", "help":
		printUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "%s: unknown command %q\n", cliName(), args[0])
		printUsage(stderr)
		return 2
	}
}

type discoveryReport struct {
	Runtime         inference.RuntimeIdentity   `json:"runtime"`
	Available       bool                        `json:"available"`
	Compile         compileReport               `json:"compile"`
	DefaultBackend  string                      `json:"default_backend"`
	BackendNames    []string                    `json:"backend_names,omitempty"`
	ModelRegistry   modelRegistryReport         `json:"model_registry"`
	FastLane        rocm.ProductionFastLane     `json:"fast_lane"`
	Workloads       []inference.TuningWorkload  `json:"workloads,omitempty"`
	CapabilityCount int                         `json:"capability_count"`
	Capabilities    []inference.Capability      `json:"capabilities,omitempty"`
	CacheModes      []string                    `json:"cache_modes,omitempty"`
	Quantizations   []string                    `json:"quantizations,omitempty"`
	Models          []modelCandidate            `json:"models,omitempty"`
	Candidates      []inference.TuningCandidate `json:"candidates,omitempty"`
	Labels          map[string]string           `json:"labels,omitempty"`
	Warnings        []string                    `json:"warnings,omitempty"`
}

type modelRegistryReport = rocm.ROCmModelRegistrySnapshot

type modelCandidate struct {
	Path         string                      `json:"path"`
	Valid        bool                        `json:"valid"`
	Supported    bool                        `json:"supported,omitempty"`
	Format       string                      `json:"format,omitempty"`
	Model        inference.ModelIdentity     `json:"model,omitempty"`
	ModelProfile *rocm.ROCmModelProfile      `json:"model_profile,omitempty"`
	ModelRoutes  *rocm.ROCmModelRoutePlan    `json:"model_routes,omitempty"`
	TokenLoop    *rocm.ROCmTokenLoopStatus   `json:"token_loop,omitempty"`
	Tokenizer    inference.TokenizerIdentity `json:"tokenizer,omitempty"`
	Notes        []string                    `json:"notes,omitempty"`
	Labels       map[string]string           `json:"labels,omitempty"`
	Error        string                      `json:"error,omitempty"`
}

type packReport struct {
	*inference.ModelPackInspection
	Backend      string                    `json:"backend"`
	Command      string                    `json:"command"`
	CLIContract  string                    `json:"cli_contract"`
	ModelProfile *rocm.ROCmModelProfile    `json:"model_profile,omitempty"`
	ModelRoutes  *rocm.ROCmModelRoutePlan  `json:"model_routes,omitempty"`
	TokenLoop    *rocm.ROCmTokenLoopStatus `json:"token_loop,omitempty"`
}

type compileTarget struct {
	Name        string            `json:"name"`
	Artifact    string            `json:"artifact,omitempty"`
	Kind        string            `json:"kind"`
	Backend     string            `json:"backend"`
	OS          string            `json:"os,omitempty"`
	Arch        string            `json:"arch,omitempty"`
	Native      bool              `json:"native"`
	StubRuntime bool              `json:"stub_runtime,omitempty"`
	EnvGates    []string          `json:"env_gates,omitempty"`
	Sidecars    []string          `json:"sidecars,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`
}

type compileReport struct {
	Current       compileTarget   `json:"current"`
	BinaryTargets []compileTarget `json:"binary_targets,omitempty"`
	GoTargets     []compileTarget `json:"go_targets,omitempty"`
	KernelTargets []compileTarget `json:"kernel_targets,omitempty"`
}

type fuseReport struct {
	Backend     string                         `json:"backend"`
	Command     string                         `json:"command"`
	CLIContract string                         `json:"cli_contract"`
	Feature     string                         `json:"feature"`
	Status      string                         `json:"status"`
	Detail      string                         `json:"detail,omitempty"`
	Capability  inference.Capability           `json:"capability,omitempty"`
	Inputs      map[string]string              `json:"inputs,omitempty"`
	Base        *inference.ModelPackInspection `json:"base,omitempty"`
	Adapter     inference.AdapterIdentity      `json:"adapter,omitempty"`
	Result      *rocm.LoRAFuseResult           `json:"result,omitempty"`
}

func rocmModelRoutePlanForProfile(profile *rocm.ROCmModelProfile) *rocm.ROCmModelRoutePlan {
	if profile == nil || !profile.Matched() {
		return nil
	}
	plan := rocm.ROCmModelRoutePlanForProfile(*profile)
	if !plan.Matched() {
		return nil
	}
	return &plan
}

func rocmModelRoutePlanForProfileAndModel(profile *rocm.ROCmModelProfile, model inference.TextModel) *rocm.ROCmModelRoutePlan {
	if profile == nil || !profile.Matched() {
		return nil
	}
	plan := rocm.ROCmModelRoutePlanForProfileAndModel(*profile, model)
	if !plan.Matched() {
		return nil
	}
	return &plan
}

func rocmTokenLoopForProfile(profile *rocm.ROCmModelProfile) *rocm.ROCmTokenLoopStatus {
	if profile == nil || !profile.Matched() {
		return nil
	}
	identity := profile.Model
	if identity.Architecture == "" {
		identity.Architecture = firstROCmCLINonEmptyString(profile.Architecture, profile.ArchitectureProfile.ID)
	}
	status, ok := rocm.ROCmTokenLoopForIdentity(identity.Path, identity)
	if !ok || !status.Matched() {
		return nil
	}
	return &status
}

func rocmTokenLoopForProfileAndModel(profile *rocm.ROCmModelProfile, model inference.TextModel) *rocm.ROCmTokenLoopStatus {
	if model != nil {
		if status, ok := rocm.ROCmTokenLoopForModel(model); ok && status.Matched() {
			return &status
		}
	}
	return rocmTokenLoopForProfile(profile)
}

type memoryPretrainBuildReport struct {
	Version     int                                             `json:"version"`
	Kind        string                                          `json:"kind"`
	Backend     string                                          `json:"backend"`
	Command     string                                          `json:"command"`
	CLIContract string                                          `json:"cli_contract"`
	NoPython    bool                                            `json:"no_python"`
	Embedding   string                                          `json:"embedding"`
	Report      *memorypretrain.MemoryPretrainingArtifactReport `json:"report,omitempty"`
}

type peftAdapterConfig struct {
	PeftType            string          `json:"peft_type,omitempty"`
	BaseModelNameOrPath string          `json:"base_model_name_or_path,omitempty"`
	TaskType            string          `json:"task_type,omitempty"`
	Rank                int             `json:"rank,omitempty"`
	R                   int             `json:"r,omitempty"`
	Alpha               float64         `json:"alpha,omitempty"`
	LoRAAlpha           float64         `json:"lora_alpha,omitempty"`
	Scale               float64         `json:"scale,omitempty"`
	TargetModules       json.RawMessage `json:"target_modules,omitempty"`
	TargetParameters    json.RawMessage `json:"target_parameters,omitempty"`
}

func runDiscoverCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("discover"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	backendName := fs.String("backend", defaultCLIBackendName(), "registered inference backend to report: rocm, cuda, cpu, auto, or any registered backend")
	jsonOut := fs.Bool("json", false, "print JSON machine discovery report")
	modelDir := fs.String("model-dir", "", "model directory to scan without loading weights")
	includeModels := fs.Bool("include-models", false, "include discovered model packs")
	includeCandidates := fs.Bool("include-candidates", false, "include first-pass ROCm tuning candidates for discovered models")
	maxModels := fs.Int("max-models", 0, "maximum discovered models to report")
	probeDevice := fs.Bool("probe-device", false, "include registered ROCm capability details")
	workload := fs.String("workload", "", "workload hint retained for CLI contract compatibility")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: %s discover [flags]\n\n", cliName())
		fmt.Fprintln(stderr, "Report the registered ROCm runtime and optionally scan model packs without loading weights.")
		fmt.Fprintln(stderr, "\nFlags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s discover: unexpected positional arguments\n", cliName())
		fs.Usage()
		return 2
	}
	if *maxModels < 0 {
		fmt.Fprintf(stderr, "%s discover: -max-models must be >= 0\n", cliName())
		return 2
	}
	workloads, err := cliTuningWorkloads(*workload)
	if err != nil {
		fmt.Fprintf(stderr, "%s discover: %v\n", cliName(), err)
		return 2
	}

	backend := normalizeBackendName(*backendName)
	report := buildDiscoveryReport(backend)
	report.Workloads = cliTuningWorkloadsOrDefault(workloads)
	if report.Labels == nil {
		report.Labels = map[string]string{}
	}
	if strings.TrimSpace(*workload) != "" {
		report.Labels["workload"] = strings.TrimSpace(*workload)
	}
	report.Labels["include_candidates"] = strconv.FormatBool(*includeCandidates)
	if *includeModels || *includeCandidates || strings.TrimSpace(*modelDir) != "" {
		models, err := discoverModelCandidates(ctx, backend, *modelDir, *maxModels)
		if err != nil {
			fmt.Fprintf(stderr, "%s discover: %v\n", cliName(), err)
			return 1
		}
		report.Models = models
		if *includeCandidates {
			var warnings []string
			report.Candidates, warnings = buildROCmDiscoveryTuningCandidates(ctx, backend, report.Workloads, models)
			report.Warnings = append(report.Warnings, warnings...)
		}
	}
	if !*probeDevice {
		report.Capabilities = nil
	}
	if *jsonOut {
		return writeJSON(stdout, stderr, report)
	}
	printDiscoverySummary(stdout, report)
	return 0
}

func buildDiscoveryReport(backendName string) discoveryReport {
	if backendName == "" {
		backendName = defaultBackendName
	}
	report := inference.BackendCapabilities(nil)
	if backend, ok := inference.Get(backendName); ok {
		if caps, ok := inference.CapabilitiesOf(backend); ok {
			report = caps
		} else {
			report = inference.BackendCapabilities(backend)
		}
	} else {
		report.Runtime.Backend = backendName
	}
	labels := cloneStringMap(report.Labels)
	if labels == nil {
		labels = map[string]string{}
	}
	labels["cli"] = cliName()
	labels["cli_contract"] = cliContractName
	labels["active_backend"] = defaultString(currentCompileReport().Current.Labels["active_backend"], defaultBackendName)
	labels["binary_default_backend"] = defaultCLIBackendName()
	labels["runtime_dispatch_status"] = defaultString(currentCompileReport().Current.Labels["runtime_dispatch_status"], "unknown")
	labels["runtime_lane"] = defaultString(currentCompileReport().Current.Labels["runtime_lane"], defaultBackendName)
	labels["production_requires_env_gate"] = "false"
	labels["production_requires_cli_flag"] = "false"
	labels["reactive_draft_detection"] = "accepted"
	labels["reactive_state_continuity"] = "accepted"
	labels["reactive_admin_reload"] = "accepted"
	return discoveryReport{
		Runtime:         report.Runtime,
		Available:       report.Available,
		Compile:         currentCompileReport(),
		DefaultBackend:  backendName,
		BackendNames:    inference.List(),
		ModelRegistry:   currentModelRegistryReport(backendName),
		FastLane:        rocm.DefaultProductionFastLane(),
		CapabilityCount: len(report.Capabilities),
		Capabilities:    append([]inference.Capability(nil), report.Capabilities...),
		CacheModes:      append([]string(nil), report.CacheModes...),
		Quantizations:   append([]string(nil), report.Quantizations...),
		Labels:          labels,
	}
}

func currentModelRegistryReport(backendName string) modelRegistryReport {
	if backendName == "" {
		backendName = defaultBackendName
	}
	return rocm.DefaultROCmModelRegistrySnapshot(backendName)
}

func currentCompileReport() compileReport {
	return compileReport{
		Current:       currentCompileTargetForCLI(cliName()),
		BinaryTargets: releaseBinaryCompileTargets(),
		GoTargets: []compileTarget{
			{Name: "linux/amd64", Kind: "go", Backend: "rocm", OS: "linux", Arch: "amd64", Native: true},
			{Name: "linux/arm64", Kind: "go", Backend: "rocm", OS: "linux", Arch: "arm64", StubRuntime: true},
			{Name: "darwin/arm64", Kind: "go", Backend: "rocm", OS: "darwin", Arch: "arm64", StubRuntime: true},
			{Name: "windows/amd64", Kind: "go", Backend: "rocm", OS: "windows", Arch: "amd64", StubRuntime: true},
		},
		KernelTargets: []compileTarget{
			{
				Name:     "amd-hip-gfx1100",
				Kind:     "hip-kernel",
				Backend:  "rocm",
				Arch:     "gfx1100",
				Native:   true,
				EnvGates: []string{"GO_ROCM_RUN_AMD_HIP_COMPILE_TESTS", "GO_ROCM_AMD_HIP_ARCH", "GO_ROCM_AMD_HIP_STD"},
				Labels: map[string]string{
					"compiler": "hipcc",
					"output":   "hsaco",
				},
			},
			{
				Name:     "nvidia-hip-sm75",
				Kind:     "hip-kernel",
				Backend:  "cuda",
				Arch:     "sm_75",
				Native:   true,
				EnvGates: []string{"GO_ROCM_RUN_NVIDIA_HIP_COMPILE_TESTS", "GO_ROCM_NVIDIA_HIP_ARCH", "GO_ROCM_NVIDIA_HIP_STD", "CUDA_PATH", "CUDA_HOME"},
				Labels: map[string]string{
					"compiler":     "hipcc",
					"hip_platform": "nvidia",
					"output":       "object",
				},
			},
			{
				Name:     "zluda-cuda-sm75",
				Kind:     "cuda-smoke",
				Backend:  "cuda",
				Arch:     "sm_75",
				Native:   true,
				EnvGates: []string{"GO_ROCM_RUN_ZLUDA_CUDA_TESTS", "GO_ROCM_NVIDIA_CUDA_ARCH", "GO_ROCM_ZLUDA_DIR", "CUDA_PATH", "CUDA_HOME"},
				Labels: map[string]string{
					"compiler": "nvcc",
					"runtime":  "zluda",
				},
			},
			{
				Name:     "hip-cpu-x86_64",
				Kind:     "hip-cpu-kernel",
				Backend:  "cpu",
				Arch:     "x86_64",
				Native:   true,
				EnvGates: []string{"GO_ROCM_RUN_HIP_CPU_COMPILE_TESTS", "GO_ROCM_RUN_HIP_CPU_RUNTIME_TESTS", "GO_ROCM_RUN_HIP_CPU_KERNEL_RUNTIME_TESTS", "GO_ROCM_HIP_CPU_CXX", "GO_ROCM_HIP_CPU_INCLUDE", "GO_ROCM_HIP_CPU_ROOT"},
				Labels: map[string]string{
					"compiler": "g++",
				},
			},
			{
				Name:     "hip-cpu-aarch64",
				Kind:     "hip-cpu-kernel",
				Backend:  "cpu",
				Arch:     "aarch64",
				Native:   true,
				EnvGates: []string{"GO_ROCM_RUN_HIP_CPU_COMPILE_TESTS", "GO_ROCM_HIP_CPU_AARCH64_CXX", "GO_ROCM_HIP_CPU_INCLUDE", "GO_ROCM_HIP_CPU_ROOT"},
				Labels: map[string]string{
					"compiler": "aarch64-linux-gnu-g++",
				},
			},
		},
	}
}

func currentCompileTargetForCLI(name string) compileTarget {
	targetName := strings.ToLower(strings.TrimSpace(filepath.Base(name)))
	if lane, ok := rocm.RuntimeLaneForArtifact(targetName); ok {
		return annotateCurrentCompileTarget(compileTargetFromRuntimeLane(lane))
	}
	if lane, ok := rocm.RuntimeLaneForArtifact(name); ok {
		return annotateCurrentCompileTarget(compileTargetFromRuntimeLane(lane))
	}
	return annotateCurrentCompileTarget(compileTargetFromRuntimeLane(rocm.CurrentProcessRuntimeLane(name)))
}

func releaseBinaryCompileTargets() []compileTarget {
	lanes := rocm.DefaultRuntimeLanes()
	out := make([]compileTarget, 0, len(lanes))
	for _, lane := range lanes {
		if lane.Kind == "release-binary" {
			out = append(out, compileTargetFromRuntimeLane(lane))
		}
	}
	return out
}

func compileTargetFromRuntimeLane(lane rocm.RuntimeLaneStatus) compileTarget {
	return compileTarget{
		Name:        lane.Name,
		Artifact:    lane.Artifact,
		Kind:        lane.Kind,
		Backend:     lane.Backend,
		OS:          lane.OS,
		Arch:        lane.Arch,
		Native:      lane.Native,
		StubRuntime: lane.StubRuntime,
		Sidecars:    append([]string(nil), lane.Sidecars...),
		Labels:      cloneStringMap(lane.Labels),
	}
}

func annotateCurrentCompileTarget(target compileTarget) compileTarget {
	target.Labels = cloneStringMap(target.Labels)
	if target.Labels == nil {
		target.Labels = map[string]string{}
	}
	target.EnvGates = append([]string(nil), target.EnvGates...)
	target.Sidecars = append([]string(nil), target.Sidecars...)
	target.Labels["cli"] = cliName()
	target.Labels["current_goos"] = runtime.GOOS
	target.Labels["current_goarch"] = runtime.GOARCH
	target.Labels["module"] = "dappco.re/go/rocm"
	target.Labels["process_matches_artifact"] = strconv.FormatBool(target.OS == runtime.GOOS && target.Arch == runtime.GOARCH)
	return target
}

func discoverModelCandidates(ctx context.Context, backendName, root string, max int) ([]modelCandidate, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return nil, nil
	}
	info, err := os.Stat(root)
	if err != nil {
		return nil, err
	}
	var paths []string
	if !info.IsDir() {
		paths = append(paths, root)
	} else if isModelPackDir(root) {
		paths = append(paths, root)
	} else {
		err = filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if max > 0 && len(paths) >= max {
				if entry.IsDir() && path != root {
					return filepath.SkipDir
				}
				return nil
			}
			if !entry.IsDir() {
				return nil
			}
			if path != root && isModelPackDir(path) {
				paths = append(paths, path)
				return filepath.SkipDir
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	sort.Strings(paths)
	if max > 0 && len(paths) > max {
		paths = paths[:max]
	}
	out := make([]modelCandidate, 0, len(paths))
	for _, path := range paths {
		inspection, err := inspectModelPack(ctx, backendName, path)
		if err != nil {
			out = append(out, modelCandidate{Path: path, Error: err.Error()})
			continue
		}
		labels := cloneStringMap(inspection.Labels)
		labels = applyROCmDraftDetectionLabels(labels, resolveROCmDraft(path, "auto", true), 0)
		model := inspection.Model
		if len(model.Labels) == 0 && len(inspection.Labels) > 0 {
			model.Labels = cloneStringMap(inspection.Labels)
		}
		var modelProfile *rocm.ROCmModelProfile
		if profile, ok := rocm.ResolveROCmModelProfileForInspection(inspection); ok {
			modelProfile = &profile
		}
		modelRoutes := rocmModelRoutePlanForProfile(modelProfile)
		tokenLoop := rocmTokenLoopForProfile(modelProfile)
		labels = applyROCmTokenLoopLabels(labels, tokenLoop)
		out = append(out, modelCandidate{
			Path:         inspection.Path,
			Valid:        true,
			Supported:    inspection.Supported,
			Format:       inspection.Format,
			Model:        model,
			ModelProfile: modelProfile,
			ModelRoutes:  modelRoutes,
			TokenLoop:    tokenLoop,
			Tokenizer:    inspection.Tokenizer,
			Notes:        append([]string(nil), inspection.Notes...),
			Labels:       labels,
		})
	}
	return out, nil
}

func buildROCmDiscoveryTuningCandidates(ctx context.Context, backendName string, workloads []inference.TuningWorkload, models []modelCandidate) ([]inference.TuningCandidate, []string) {
	if len(models) == 0 {
		return nil, nil
	}
	workloads = cliTuningWorkloadsOrDefault(workloads)
	out := make([]inference.TuningCandidate, 0, len(models)*len(workloads))
	var warnings []string
	for _, discovered := range models {
		if !discovered.Valid || strings.TrimSpace(discovered.Path) == "" {
			continue
		}
		model := discovered.Model
		if model.Path == "" {
			model.Path = discovered.Path
		}
		plan, err := rocm.PlanLocalTuning(ctx, inference.TuningPlanRequest{
			Runtime: inference.RuntimeIdentity{
				Backend: backendName,
				Device:  defaultBackendName,
			},
			Model:     model,
			Workloads: workloads,
			Budget:    inference.TuningBudget{MaxCandidates: len(workloads)},
			Labels: map[string]string{
				"candidate_source": "lthn-rocm discover",
				"cli_contract":     cliContractName,
			},
		})
		if err != nil {
			warnings = append(warnings, err.Error())
			continue
		}
		out = append(out, plan.Candidates...)
	}
	return out, warnings
}

func isModelPackDir(path string) bool {
	names := []string{"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"}
	for _, name := range names {
		if _, err := os.Stat(filepath.Join(path, name)); err == nil {
			return true
		}
	}
	matches, _ := filepath.Glob(filepath.Join(path, "*.safetensors"))
	if len(matches) > 0 {
		return true
	}
	matches, _ = filepath.Glob(filepath.Join(path, "*.gguf"))
	return len(matches) > 0
}

func printDiscoverySummary(stdout io.Writer, report discoveryReport) {
	fmt.Fprintf(stdout, "runtime discovery: %s\n", defaultString(report.Runtime.Backend, "rocm"))
	fmt.Fprintf(stdout, "  compile: %s backend=%s active_backend=%s native=%t stub=%t\n",
		report.Compile.Current.Name,
		report.Compile.Current.Backend,
		defaultString(report.Compile.Current.Labels["active_backend"], report.DefaultBackend),
		report.Compile.Current.Native,
		report.Compile.Current.StubRuntime,
	)
	if len(report.Compile.Current.Sidecars) > 0 {
		fmt.Fprintf(stdout, "  sidecars: %s\n", strings.Join(report.Compile.Current.Sidecars, ", "))
	}
	fmt.Fprintf(stdout, "  kernel compile targets: %d\n", len(report.Compile.KernelTargets))
	fmt.Fprintf(stdout, "  available: %t\n", report.Available)
	fmt.Fprintf(stdout, "  registered backends: %s\n", strings.Join(report.BackendNames, ", "))
	fmt.Fprintf(stdout, "  capabilities: %d, cache modes: %d, quantizations: %d\n", report.CapabilityCount, len(report.CacheModes), len(report.Quantizations))
	fmt.Fprintf(stdout, "  model registry: %s profiles=%d algorithms=%d feature_routes=%d tokenizers=%d adapters=%d multimodal_processors=%d diffusion_samplers=%d state_contexts=%d attached_drafters=%d loaders=%d cache_modes=%d cache_routes=%d quant_schemes=%d quant_loaders=%d mixers=%d\n", report.ModelRegistry.Name, len(report.ModelRegistry.ArchitectureProfiles), len(report.ModelRegistry.AlgorithmProfiles), len(report.ModelRegistry.FeatureRoutes), len(report.ModelRegistry.TokenizerRoutes), len(report.ModelRegistry.LoRAAdapterRoutes), len(report.ModelRegistry.MultimodalProcessorRoutes), len(report.ModelRegistry.DiffusionSamplerRoutes), len(report.ModelRegistry.StateContextRoutes), len(report.ModelRegistry.AttachedDrafterRoutes), len(report.ModelRegistry.LoaderRoutes), len(report.ModelRegistry.CacheModeRoutes), len(report.ModelRegistry.CacheRoutes), len(report.ModelRegistry.QuantSchemes), len(report.ModelRegistry.QuantLoaderRoutes), len(report.ModelRegistry.MixerLoaderRoutes))
	fmt.Fprintf(stdout, "  fast lane: %s default=%t env_gate=%t cli_flag=%t\n", report.FastLane.Name, report.FastLane.EnabledByDefault, report.FastLane.RequiresEnvGate, report.FastLane.RequiresCLIFlag)
	fmt.Fprintf(stdout, "  models: %d, candidates: %d\n", len(report.Models), len(report.Candidates))
}

func runPackCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("pack"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	backendName := fs.String("backend", defaultBackendName, "registered backend inspector to use: rocm, cuda, cpu, or any registered backend")
	jsonOut := fs.Bool("json", false, "print JSON report")
	expectedQuant := fs.Int("quantization", 0, "required quantization bits")
	maxContext := fs.Int("max-context", 0, "maximum allowed context length")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: %s pack [flags] <model-path>\n\n", cliName())
		fmt.Fprintln(stderr, "Validate a model pack on disk without loading weights.")
		fmt.Fprintln(stderr, "\nFlags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintf(stderr, "%s pack: expected exactly one model path\n", cliName())
		fs.Usage()
		return 2
	}
	inspection, err := inspectModelPack(ctx, normalizeBackendName(*backendName), fs.Arg(0))
	if err != nil {
		fmt.Fprintf(stderr, "%s pack: %v\n", cliName(), err)
		return 1
	}
	if *expectedQuant > 0 && inspection.Model.QuantBits != *expectedQuant {
		inspection.Supported = false
		inspection.Notes = append(inspection.Notes, fmt.Sprintf("expected quantization=%d, got %d", *expectedQuant, inspection.Model.QuantBits))
	}
	if *maxContext > 0 && inspection.Model.ContextLength > *maxContext {
		inspection.Supported = false
		inspection.Notes = append(inspection.Notes, fmt.Sprintf("expected context<=%d, got %d", *maxContext, inspection.Model.ContextLength))
	}
	report := buildPackReport(normalizeBackendName(*backendName), inspection)
	if *jsonOut {
		code := writeJSON(stdout, stderr, report)
		if code != 0 {
			return code
		}
		if !inspection.Supported {
			return 1
		}
		return 0
	}
	if !inspection.Supported {
		fmt.Fprintf(stderr, "%s pack: unsupported model pack: %s\n", cliName(), fs.Arg(0))
		for _, note := range inspection.Notes {
			fmt.Fprintf(stderr, "  %s\n", note)
		}
		return 1
	}
	fmt.Fprintf(stdout, "valid model pack: %s (%s, %s, quant=%d, context=%d)\n",
		defaultString(inspection.Path, fs.Arg(0)),
		inspection.Model.Architecture,
		inspection.Format,
		inspection.Model.QuantBits,
		inspection.Model.ContextLength,
	)
	if report.ModelProfile != nil && report.ModelProfile.Matched() {
		fmt.Fprintf(stdout, "  model profile: %s (%s, load=%s)\n",
			report.ModelProfile.Name,
			defaultString(report.ModelProfile.Architecture, report.ModelProfile.Family),
			report.ModelProfile.LoadStatus.Status,
		)
	}
	return 0
}

func buildPackReport(backendName string, inspection *inference.ModelPackInspection) packReport {
	report := packReport{
		ModelPackInspection: inspection,
		Backend:             backendName,
		Command:             "pack",
		CLIContract:         cliContractName,
	}
	if inspection == nil {
		return report
	}
	if profile, ok := rocm.ResolveROCmModelProfileForInspection(inspection); ok {
		report.ModelProfile = &profile
		report.ModelRoutes = rocmModelRoutePlanForProfile(report.ModelProfile)
		report.TokenLoop = rocmTokenLoopForProfile(report.ModelProfile)
	}
	return report
}

func runReactiveCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("reactive"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON reactive sequence-mixer report")
	backendName := fs.String("backend", defaultBackendName, "accepted for CLI contract compatibility; ROCm planner is used")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: %s reactive [flags] <model-path>\n\n", cliName())
		fmt.Fprintln(stderr, "Report go-mlx config-composed sequence-mixer readiness for ROCm native load.")
		fmt.Fprintln(stderr, "\nFlags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintf(stderr, "%s reactive: expected exactly one model path\n", cliName())
		fs.Usage()
		return 2
	}
	_ = normalizeBackendName(*backendName)
	report, err := rocm.PlanReactiveSequenceMixer(ctx, fs.Arg(0))
	if err != nil {
		fmt.Fprintf(stderr, "%s reactive: %v\n", cliName(), err)
		return 1
	}
	if *jsonOut {
		if code := writeJSON(stdout, stderr, report); code != 0 {
			return code
		}
		return reactiveReportExitCode(report)
	}
	printReactiveSequenceMixerReport(stdout, report)
	return reactiveReportExitCode(report)
}

func reactiveReportExitCode(report *rocm.ReactiveSequenceMixerReport) int {
	if report == nil {
		return 1
	}
	switch report.Status {
	case "invalid", "incomplete", "unsupported_format", "native_unavailable":
		return 1
	default:
		return 0
	}
}

func printReactiveSequenceMixerReport(stdout io.Writer, report *rocm.ReactiveSequenceMixerReport) {
	if report == nil {
		return
	}
	fmt.Fprintln(stdout, "reactive sequence-mixer report")
	fmt.Fprintf(stdout, "  model: %s\n", report.ModelPath)
	fmt.Fprintf(stdout, "  status: %s (%s)\n", report.Status, report.ExecutionStatus)
	fmt.Fprintf(stdout, "  registry: %d mixer kind(s)\n", len(report.Registry))
	if report.Plan != nil {
		fmt.Fprintf(stdout, "  plan: %d layer(s), runtime=%s\n", len(report.Plan.Layers), report.Plan.Runtime)
		if subpaths := reactivePlanSubpaths(report.Plan); subpaths != "" {
			fmt.Fprintf(stdout, "  subpaths: %s\n", subpaths)
		}
		if holders := reactivePlanCacheHolders(report.Plan); holders != "" {
			fmt.Fprintf(stdout, "  cache holders: %s\n", holders)
			if modes := reactivePlanCacheModes(report.Plan); modes != "" {
				fmt.Fprintf(stdout, "  cache modes: %s\n", modes)
			}
		} else {
			fmt.Fprintf(stdout, "  cache: %d holder(s)\n", len(report.Plan.Cache.Layers))
		}
	}
	if len(report.MissingTensors) > 0 {
		fmt.Fprintf(stdout, "  missing tensors: %d\n", len(report.MissingTensors))
	}
	if report.Labels != nil && report.Labels["sequence_mixer_runner_status"] != "" {
		fmt.Fprintf(stdout, "  runner: %s\n", report.Labels["sequence_mixer_runner_status"])
	}
	fmt.Fprintf(stdout, "  ready: planning=%t tensor_binding=%t composed_stack=%t runner=%t\n",
		report.PlanningReady,
		report.TensorBindingReady,
		report.ComposedStackReady,
		report.RunnerReady,
	)
}

func reactivePlanSubpaths(plan *rocm.SequenceMixerLoadPlan) string {
	if plan == nil || len(plan.Layers) == 0 {
		return ""
	}
	parts := make([]string, 0, len(plan.Layers))
	for _, layer := range plan.Layers {
		subpath := strings.TrimSpace(layer.Subpath)
		if subpath == "" {
			subpath = "bare"
		}
		parts = append(parts, fmt.Sprintf("%d:%s", layer.Layer, subpath))
	}
	return strings.Join(parts, ",")
}

func reactivePlanCacheHolders(plan *rocm.SequenceMixerLoadPlan) string {
	if plan == nil || len(plan.Cache.Layers) == 0 {
		return ""
	}
	parts := make([]string, 0, len(plan.Cache.Layers))
	for _, layer := range plan.Cache.Layers {
		parts = append(parts, fmt.Sprintf("%d:%s:%s", layer.Layer, layer.Kind, layer.Holder))
	}
	return strings.Join(parts, ",")
}

func reactivePlanCacheModes(plan *rocm.SequenceMixerLoadPlan) string {
	if plan == nil || len(plan.Cache.Layers) == 0 {
		return ""
	}
	parts := make([]string, 0, len(plan.Cache.Layers))
	for _, layer := range plan.Cache.Layers {
		mode := strings.TrimSpace(layer.Mode)
		if mode == "" {
			mode = "unknown"
		}
		parts = append(parts, fmt.Sprintf("%d:%s:%s", layer.Layer, layer.Kind, mode))
	}
	return strings.Join(parts, ",")
}

func runGenerateCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("generate"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	backendName := fs.String("backend", defaultCLIBackendName(), "registered backend to load: rocm, cuda, cpu, auto, or any registered backend")
	prompt := fs.String("prompt", "Write a concise explanation of ROCm inference.", "user prompt")
	maxTokens := fs.Int("max-tokens", 0, "tokens to generate; 0 uses remaining model context/default")
	draftPath := fs.String("draft", "auto", "MTP drafter: 'auto' detects one beside a Gemma 4 target, a path forces it, '' disables")
	draftBlock := fs.Int("draft-block", 0, "MTP draft block; 0 = backend default")
	profileDir := fs.String("profile-dir", "", "tuned-profile directory")
	noAutoProfile := fs.Bool("no-auto-profile", false, "ignore tuned MTP draft-block profiles")
	temp := fs.Float64("temp", 0, "sampling temperature; 0 uses greedy production generation")
	think := fs.Bool("think", false, "enable thinking channel when the model supports it")
	contextLen := fs.Int("context", 0, "context length override")
	kvCacheMode := fs.String("kv-cache", "", rocmCLIKVCacheFlagUsage)
	pipeline := fs.Bool("pipeline", true, "native pipelined decode; false is recognized but not bound yet")
	kvStorage := fs.String("kv-storage", "", "retained KV storage dtype (fp16; bf16 recognized but not bound yet; empty = native default)")
	tracePhases := fs.Bool("trace", false, "print ROCm aggregate prefill/decode timing for greedy and sampled lanes")
	stateName := fs.String("state", "", "conversation state name: omitted uses the production default; explicit empty disables retained state")
	stateStore := fs.String("state-store", "", "state store file (default ~/Lethean/data/state/agent.kv)")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: %s generate [flags] <model-path>\n\n", cliName())
		fmt.Fprintln(stderr, "Load a ROCm model and stream one completion without the HTTP server path.")
		fmt.Fprintln(stderr, "\nFlags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintf(stderr, "%s generate: expected exactly one model path\n", cliName())
		fs.Usage()
		return 2
	}
	if *maxTokens < 0 {
		fmt.Fprintf(stderr, "%s generate: -max-tokens must be >= 0\n", cliName())
		return 2
	}
	if *draftBlock < 0 {
		fmt.Fprintf(stderr, "%s generate: -draft-block must be >= 0\n", cliName())
		return 2
	}
	kvCacheDecision := resolveROCmCLIKVCacheMode("generate", *kvCacheMode)
	if kvCacheDecision.Code != 0 {
		fmt.Fprint(stderr, kvCacheDecision.Message)
		return kvCacheDecision.Code
	}
	kvStorageDecision := resolveROCmCLIKVStorageMode("generate", *kvStorage)
	if kvStorageDecision.Code != 0 {
		fmt.Fprint(stderr, kvStorageDecision.Message)
		return kvStorageDecision.Code
	}
	kvLoadModeDecision := resolveROCmCLILoadModeForCacheAndStorage("generate", kvCacheDecision.Mode, kvStorageDecision.Mode)
	if kvLoadModeDecision.Code != 0 {
		fmt.Fprint(stderr, kvLoadModeDecision.Message)
		return kvLoadModeDecision.Code
	}
	pipelineDecision := resolveROCmCLIPipelineMode("generate", *pipeline)
	if pipelineDecision.Code != 0 {
		fmt.Fprint(stderr, pipelineDecision.Message)
		return pipelineDecision.Code
	}
	rocmLoadCfg := rocmCLILoadConfigForKVCache(kvLoadModeDecision.Mode)
	loadOpts := []inference.LoadOption{}
	backend := normalizeBackendName(*backendName)
	if err := validateCLIBackendDispatch("generate", backend); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if err := validateROCmCLILoadConfigBackend("generate", backend, rocmLoadCfg); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	if backend != "" && backend != "auto" {
		loadOpts = append(loadOpts, inference.WithBackend(backend))
	}
	if *contextLen > 0 {
		loadOpts = append(loadOpts, inference.WithContextLen(*contextLen))
	}
	draftDetection := resolveROCmDraft(fs.Arg(0), *draftPath, true)
	if *draftBlock > 0 && !draftDetection.Active() {
		fmt.Fprintf(stderr, "%s generate: -draft-block requires an explicit or auto-detected drafter; native ROCm drafter execution is pending\n", cliName())
		return 1
	}
	resolvedDraftBlock := *draftBlock
	draftBlockSource := ""
	if draftDetection.Active() {
		resolvedDraftBlock, draftBlockSource = resolveROCmServeDraftBlock(ctx, draftDetection, fs.Arg(0), *draftBlock, *noAutoProfile, *profileDir)
	}
	if *tracePhases {
		if draftDetection.Active() {
			return refuseGenerateReactiveDraftMode("trace", draftDetection, resolvedDraftBlock, stderr)
		}
		return runGenerateTrace(ctx, fs.Arg(0), *prompt, *maxTokens, float32(*temp), *think, rocmLoadCfg, loadOpts, stdout, stderr)
	}
	stateFlagSet := cliFlagWasSet(fs, "state")
	generateStateName := strings.TrimSpace(*stateName)
	if !stateFlagSet && rocmCLIBackendUsesDefaultState(backend) {
		generateStateName = defaultROCmGenerateStateName(fs.Arg(0))
	}
	if draftDetection.Active() && *temp != 0 {
		mode := "generate"
		if generateStateName != "" {
			mode = "state"
		}
		return refuseGenerateReactiveDraftSampling(mode, draftDetection, resolvedDraftBlock, *temp, stderr)
	}
	if strings.TrimSpace(*stateStore) != "" && stateFlagSet && generateStateName == "" {
		fmt.Fprintf(stderr, "%s generate: -state-store requires a non-empty -state when retained state is explicitly configured\n", cliName())
		return 2
	}
	if generateStateName != "" {
		return runGenerateState(ctx, fs.Arg(0), *prompt, generateStateName, *stateStore, *maxTokens, float32(*temp), *think, rocmLoadCfg, loadOpts, draftDetection, resolvedDraftBlock, draftBlockSource, stdout, stderr)
	}

	var model inference.TextModel
	if draftDetection.Active() {
		fmt.Fprintf(stderr, "%s generate: reactive MTP drafter resolved: %s (%s), block %s\n",
			cliName(), draftDetection.DraftPath, draftDetection.Note, resolvedROCmDraftBlockLabel(resolvedDraftBlock)+resolvedROCmDraftBlockSourceSuffix(draftBlockSource))
		loaded, err := rocm.LoadAttachedDrafterPairAsTextModelBlockWithConfig(fs.Arg(0), draftDetection.DraftPath, resolvedDraftBlock, rocmLoadCfg, loadOpts...)
		if err != nil {
			fmt.Fprintf(stderr, "%s generate: native ROCm drafter execution is pending: %v\n", cliName(), err)
			return 1
		}
		model = loaded
		reportGenerateDrafterStatus(stderr, model)
	} else {
		loaded, err := loadROCmCLITextModel(fs.Arg(0), rocmLoadCfg, loadOpts)
		if err != nil {
			fmt.Fprintf(stderr, "%s generate: load: %v\n", cliName(), err)
			return 1
		}
		model = loaded
	}
	defer model.Close()

	enableThinking := *think
	start := time.Now()
	count := 0
	messages := []inference.Message{{Role: "user", Content: *prompt}}
	generateOpts := rocmCLIGenerateOptions(*maxTokens, float32(*temp), enableThinking)
	var stream iter.Seq[inference.Token]
	if stateless, ok := model.(statelessChatModel); ok {
		stream = stateless.ChatStateless(ctx, messages, generateOpts...)
	} else {
		stream = model.Chat(ctx, messages, generateOpts...)
	}
	for tok := range stream {
		fmt.Fprint(stdout, tok.Text)
		count++
	}
	if err := model.Err(); err != nil {
		fmt.Fprintf(stderr, "%s generate: %v\n", cliName(), err)
		return 1
	}
	fmt.Fprintln(stdout)
	elapsed := time.Since(start)
	if elapsed > 0 {
		fmt.Fprintf(stdout, "\ndecode %.1f tok/s  (%d tok / %.3fs)\n", float64(count)/elapsed.Seconds(), count, elapsed.Seconds())
	}
	printGenerateAttachedDrafterMetrics(stdout, model)
	return 0
}

func printGenerateAttachedDrafterMetrics(stdout io.Writer, model inference.TextModel) {
	provider, ok := model.(interface {
		AttachedDrafterMetrics() *rocm.AttachedDrafterMetrics
	})
	if !ok || provider == nil {
		return
	}
	metrics := provider.AttachedDrafterMetrics()
	if metrics == nil || metrics.ProposedTokens <= 0 || metrics.VerifyCalls <= 0 {
		return
	}
	fmt.Fprintf(stdout, "mtp: %.0f%% acceptance (%d/%d drafted) over %d verify forwards\n",
		metrics.AcceptanceRate*100,
		metrics.AcceptedTokens,
		metrics.ProposedTokens,
		metrics.VerifyCalls,
	)
}

func reportGenerateDrafterStatus(stderr io.Writer, model inference.TextModel) {
	reporter, ok := model.(rocm.ROCmModelIdentityReporter)
	if !ok || reporter == nil {
		return
	}
	identity := reporter.ModelIdentity()
	labels := identity.Labels
	if len(labels) == 0 {
		return
	}
	attachment := strings.TrimSpace(labels["attached_drafter_native_attachment"])
	if attachment == "" {
		return
	}
	fmt.Fprintf(stderr, "%s generate: native MTP status attachment=%s handoff=%s route=%s assistant=%s preflight=%s plan=%s bridge=%s proposal=%s reason=%s\n",
		cliName(),
		attachment,
		firstCLIStatusLabel(labels, "attached_drafter_native_handoff"),
		firstCLIStatusLabel(labels, "attached_drafter_generation_route"),
		firstCLIStatusLabel(labels, "attached_drafter_assistant_verify", "attached_drafter_assistant_state_verify"),
		firstCLIStatusLabel(labels, "attached_drafter_assistant_verifier_preflight"),
		firstCLIStatusLabel(labels, "attached_drafter_assistant_verifier_plan"),
		firstCLIStatusLabel(labels, "attached_drafter_assistant_draft_step_input_bridge"),
		firstCLIStatusLabel(labels, "attached_drafter_assistant_draft_step_proposal_runtime"),
		firstCLIStatusLabel(labels,
			"attached_drafter_assistant_verifier_plan_reason",
			"attached_drafter_assistant_verifier_reason",
			"attached_drafter_assistant_draft_step_input_bridge_reason",
			"attached_drafter_assistant_draft_step_proposal_runtime_reason",
			"attached_drafter_assistant_layer_runtime_reason",
		),
	)
}

func firstCLIStatusLabel(labels map[string]string, keys ...string) string {
	for _, key := range keys {
		if value := strings.TrimSpace(labels[key]); value != "" {
			return value
		}
	}
	return "unknown"
}

func runServeCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("serve"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", ":36911", "listen address")
	backendName := fs.String("backend", defaultCLIBackendName(), "registered backend to load: rocm, cuda, cpu, auto, or any registered backend")
	modelPath := fs.String("model", "", "model path to load; empty starts model-less for admin reload")
	draftPath := fs.String("draft", "auto", "MTP drafter: 'auto' detects one beside a Gemma 4 target, a path forces it, '' disables")
	draftDetect := fs.Bool("draft-detect", true, "reactive drafter detection for Gemma 4 targets")
	draftBlock := fs.Int("draft-block", 0, "MTP draft block; 0 = backend default")
	noAutoProfile := fs.Bool("no-auto-profile", false, "ignore tuned profiles from tune")
	profileDir := fs.String("profile-dir", "", "tuned-profile directory")
	contextLen := fs.Int("context", 0, "override context length")
	kvCacheMode := fs.String("kv-cache", "", rocmCLIKVCacheFlagUsage)
	readTimeout := fs.Duration("read-timeout", 30*time.Second, "HTTP read header timeout")
	writeTimeout := fs.Duration("write-timeout", 5*time.Minute, "HTTP write timeout")
	shutdownTimeout := fs.Duration("shutdown-timeout", 10*time.Second, "graceful shutdown deadline")
	printAdminToken := fs.Bool("print-admin-token", false, "accepted for CLI contract compatibility")
	rotateAdminToken := fs.Bool("rotate-admin-token", false, "accepted for CLI contract compatibility")
	stateConversations := fs.Bool("state-conversations", true, "accepted for CLI contract compatibility")
	stateStorePath := fs.String("state-store", "", "accepted for CLI contract compatibility")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: %s serve [--model <path>] [flags]\n\n", cliName())
		fmt.Fprintln(stderr, "Host an OpenAI / Anthropic / Ollama-compatible HTTP API for ROCm-backed models.")
		fmt.Fprintln(stderr, "\nInference routes:")
		fmt.Fprintln(stderr, "  POST /v1/chat/completions")
		fmt.Fprintln(stderr, "  POST /v1/responses")
		fmt.Fprintln(stderr, "  POST /v1/messages")
		fmt.Fprintln(stderr, "  POST /api/chat")
		fmt.Fprintln(stderr, "  POST /api/generate")
		fmt.Fprintln(stderr, "  POST /v1/score")
		fmt.Fprintln(stderr, "\nAdmin routes require the generated bearer token.")
		fmt.Fprintln(stderr, "  GET  /v1/admin/machine")
		fmt.Fprintln(stderr, "  GET  /v1/admin/serve/status")
		fmt.Fprintln(stderr, "  POST /v1/admin/serve/reload")
		fmt.Fprintln(stderr, "  POST /v1/admin/models/download")
		fmt.Fprintln(stderr, "  GET  /v1/admin/models/download?job=<id>")
		fmt.Fprintln(stderr, "\nFlags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	kvCacheDecision := resolveROCmCLIKVCacheMode("serve", *kvCacheMode)
	if kvCacheDecision.Code != 0 {
		fmt.Fprint(stderr, kvCacheDecision.Message)
		return kvCacheDecision.Code
	}
	rocmLoadCfg := rocmCLILoadConfigForKVCache(kvCacheDecision.Mode)
	serveBackend := normalizeBackendName(*backendName)
	if strings.TrimSpace(*modelPath) != "" {
		if err := validateCLIBackendDispatch("serve", serveBackend); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	if err := validateROCmCLILoadConfigBackend("serve", serveBackend, rocmLoadCfg); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	_ = stdout
	tokenPath, err := defaultAdminTokenPath()
	if err != nil {
		fmt.Fprintf(stderr, "%s serve: admin token path: %v\n", cliName(), err)
		return 1
	}
	if *rotateAdminToken {
		token, err := rotateAdminTokenFile(tokenPath)
		if err != nil {
			fmt.Fprintf(stderr, "%s serve: token rotation failed: %v\n", cliName(), err)
			return 1
		}
		fmt.Fprintf(stderr, "%s admin token (rotated):\n  %s\n  saved to %s (mode 0600)\n  any running serve still holds the old token - restart to apply\n", cliName(), token, tokenPath)
		return 0
	}
	adminToken, generated, err := ensureAdminTokenFile(tokenPath)
	if err != nil {
		fmt.Fprintf(stderr, "%s serve: admin token init failed: %v\n", cliName(), err)
		return 1
	}
	if *printAdminToken {
		label := "loaded"
		if generated {
			label = "newly generated"
		}
		fmt.Fprintf(stderr, "%s admin token (%s):\n  %s\n  at %s (mode 0600)\n", cliName(), label, adminToken, tokenPath)
		return 0
	}
	loadOpts := []inference.LoadOption{}
	if *contextLen > 0 {
		loadOpts = append(loadOpts, inference.WithContextLen(*contextLen))
	}
	resolvedProfileDir := strings.TrimSpace(*profileDir)
	if resolvedProfileDir == "" {
		resolvedProfileDir = standardTuningProfileDir()
	}
	stateStorePathValue := strings.TrimSpace(*stateStorePath)
	var serveStateStore state.Store
	var serveStateStoreCloser io.Closer
	stateContinuityReason := ""
	if *stateConversations {
		if stateStorePathValue == "" {
			defaultStateStorePath, err := defaultROCmConversationStateStorePath()
			if err != nil {
				stateContinuityReason = "resolve default state store: " + err.Error()
			} else {
				stateStorePathValue = defaultStateStorePath
			}
		}
		if stateContinuityReason == "" {
			store, err := openOrCreateROCmStateStore(ctx, stateStorePathValue)
			if err != nil {
				stateContinuityReason = fmt.Sprintf("open state store %s: %v", stateStorePathValue, err)
			} else {
				serveStateStore = store
				serveStateStoreCloser = store
				defer serveStateStoreCloser.Close()
			}
		}
	}
	serveCfg := rocmServeConfig{
		Backend:               serveBackend,
		ModelPath:             *modelPath,
		ContextLen:            *contextLen,
		KVCacheMode:           kvCacheDecision.Mode,
		ROCmLoadConfig:        rocmLoadCfg,
		DraftPath:             *draftPath,
		DraftDetect:           *draftDetect,
		DraftBlock:            *draftBlock,
		DraftDetection:        resolveROCmDraft(*modelPath, *draftPath, *draftDetect),
		NoAutoProfile:         *noAutoProfile,
		ProfileDir:            resolvedProfileDir,
		StateConversations:    *stateConversations,
		StateStorePath:        stateStorePathValue,
		StateStore:            serveStateStore,
		StateContinuityReason: stateContinuityReason,
		LoadOptions:           loadOpts,
		AdminToken:            adminToken,
		AdminAudit:            stderr,
	}
	handler, resolver := newROCmServeHandler(serveCfg)
	if strings.TrimSpace(*modelPath) == "" {
		fmt.Fprintf(stderr, "%s serve: starting model-less; POST /v1/admin/serve/reload to set a model\n", cliName())
	}
	if serveCfg.DraftDetection.Active() {
		resolvedBlock, blockSource := resolveROCmServeDraftBlock(ctx, serveCfg.DraftDetection, *modelPath, *draftBlock, *noAutoProfile, serveCfg.ProfileDir)
		source := ""
		if strings.TrimSpace(blockSource) != "" {
			source = "; " + strings.TrimSpace(blockSource)
		}
		fmt.Fprintf(stderr, "%s serve: reactive MTP drafter resolved: %s (%s), block %s%s; native ROCm attached-drafter load will run on first request\n",
			cliName(), serveCfg.DraftDetection.DraftPath, serveCfg.DraftDetection.Note, resolvedROCmDraftBlockLabel(resolvedBlock), source)
	} else if explicitReactiveDraft(*draftPath, *draftBlock) || (*draftDetect && strings.TrimSpace(*modelPath) != "") {
		fmt.Fprintf(stderr, "%s serve: reactive MTP drafter surface accepted; no drafter resolved for this target yet\n", cliName())
	}
	if *stateConversations && serveStateStore != nil {
		fmt.Fprintf(stderr, "%s serve: conversation continuity store ready: %s\n", cliName(), stateStorePathValue)
	} else if *stateConversations {
		fmt.Fprintf(stderr, "%s serve: conversation continuity unavailable (%s); serving stateless while runtime wake/sleep admin routes remain available\n", cliName(), stateContinuityReason)
	}
	server := &http.Server{
		Addr:              *addr,
		Handler:           handler,
		ReadHeaderTimeout: *readTimeout,
		WriteTimeout:      *writeTimeout,
	}
	errCh := make(chan error, 1)
	go func() {
		errCh <- server.ListenAndServe()
	}()
	fmt.Fprintf(stderr, "%s serve: listening on %s with %s contract\n", cliName(), *addr, cliContractName)
	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), *shutdownTimeout)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			_ = resolver.Close()
			fmt.Fprintf(stderr, "%s serve: shutdown: %v\n", cliName(), err)
			return 1
		}
		_ = resolver.Close()
		return 0
	case err := <-errCh:
		_ = resolver.Close()
		if errors.Is(err, http.ErrServerClosed) {
			return 0
		}
		fmt.Fprintf(stderr, "%s serve: %v\n", cliName(), err)
		return 1
	}
}

func runMenubarCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("menubar"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	_ = fs.String("addr", ":36911", "accepted for CLI contract compatibility")
	_ = fs.String("model", "", "accepted for CLI contract compatibility")
	_ = fs.Bool("foreground", false, "accepted for CLI contract compatibility")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	return compatibilityStub("menubar", "menu-bar app", stdout, stderr)
}

func runDiffuseCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("diffuse"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	prompt := fs.String("prompt", "Write a haiku about clockwork.", "user prompt")
	maxCanvases := fs.Int("max-canvases", 8, "response length bound, in canvases")
	steps := fs.Int("steps", 0, "max denoising steps per canvas (0 = tuned default)")
	canvas := fs.Int("canvas", 0, "canvas length (0 = model/default)")
	entropy := fs.Float64("entropy", 0.3, "acceptance entropy budget per step")
	seed := fs.Uint64("seed", 0, "PRNG key chain root")
	chatTemplate := fs.Bool("chat", true, "format the prompt with the model chat template")
	trace := fs.Bool("trace", false, "include per-step trace request in the report")
	backendName := fs.String("backend", defaultBackendName, "backend name for model-pack inspection")
	jsonOut := fs.Bool("json", false, "print JSON diffusion runtime report")
	compatMaxTokens := fs.Int("max-tokens", 0, "accepted legacy alias for CLI compatibility")
	compatTemp := fs.Float64("temp", 0, "accepted legacy alias for CLI compatibility")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: %s diffuse [flags] <model-path>\n\n", cliName())
		fmt.Fprintln(stderr, "Inspect DiffusionGemma block-diffusion readiness using the go-mlx diffuse CLI contract.")
		fmt.Fprintln(stderr, "\nFlags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintf(stderr, "%s diffuse: expected exactly one model path\n", cliName())
		fs.Usage()
		return 2
	}
	report, err := diffusePlanReportFromOptions(ctx, diffuseCommandOptions{
		Backend:         *backendName,
		ModelPath:       fs.Arg(0),
		Prompt:          *prompt,
		MaxCanvases:     *maxCanvases,
		Steps:           *steps,
		Canvas:          *canvas,
		Entropy:         *entropy,
		Seed:            *seed,
		Chat:            *chatTemplate,
		Trace:           *trace,
		CompatMaxTokens: *compatMaxTokens,
		CompatTemp:      *compatTemp,
	})
	if err != nil {
		fmt.Fprintf(stderr, "%s diffuse: %v\n", cliName(), err)
		return 2
	}
	if *jsonOut {
		if code := writeJSON(stdout, stderr, report); code != 0 {
			return code
		}
		return diffuseReportExitCode(report)
	}
	printDiffusePlanSummary(stdout, report)
	return diffuseReportExitCode(report)
}

func runAudioCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("audio"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelPath := fs.String("model", "", "model path")
	backendName := fs.String("backend", defaultBackendName, "backend name for model-pack inspection")
	audioPath := fs.String("audio", "", "16 kHz mono WAV clip")
	prompt := fs.String("prompt", "What is said in this recording?", "prompt")
	maxTokens := fs.Int("max-tokens", 256, "tokens to generate")
	chatTemplate := fs.Bool("chat", true, "format with the model chat template")
	jsonOut := fs.Bool("json", false, "print JSON report")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: %s audio -audio clip.wav [flags] <model-path>\n", cliName())
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	resolvedModel := strings.TrimSpace(*modelPath)
	if resolvedModel == "" && fs.NArg() == 1 {
		resolvedModel = fs.Arg(0)
	} else if resolvedModel == "" && fs.NArg() > 1 {
		fmt.Fprintf(stderr, "%s audio: expected exactly one model path\n", cliName())
		fs.Usage()
		return 2
	} else if resolvedModel != "" && fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s audio: model path provided twice\n", cliName())
		fs.Usage()
		return 2
	}
	report, err := audioPlanReportFromOptions(ctx, audioCommandOptions{
		Backend:   *backendName,
		ModelPath: resolvedModel,
		AudioPath: *audioPath,
		Prompt:    *prompt,
		MaxTokens: *maxTokens,
		Chat:      *chatTemplate,
	})
	if err != nil {
		fmt.Fprintf(stderr, "%s audio: %v\n", cliName(), err)
		return 2
	}
	if *jsonOut {
		return writeJSON(stdout, stderr, report)
	}
	printAudioPlanReport(stdout, report)
	return 0
}

func runVisionCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("vision"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelPath := fs.String("model", "", "model path")
	backendName := fs.String("backend", defaultBackendName, "backend name for model-pack inspection")
	images := fs.String("images", "", "comma-separated PNG/JPEG image paths")
	videoFrames := fs.String("video-frames", "", "comma-separated PNG/JPEG frame paths")
	fps := fs.Float64("fps", 1, "frame rate for video frames")
	prompt := fs.String("prompt", "Describe what you see.", "prompt")
	maxTokens := fs.Int("max-tokens", 256, "tokens to generate")
	chatTemplate := fs.Bool("chat", true, "format with the model chat template")
	jsonOut := fs.Bool("json", false, "print JSON report")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: %s vision -images a.png[,b.jpg] [flags] <model-path>\n", cliName())
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	resolvedModel := strings.TrimSpace(*modelPath)
	if resolvedModel == "" && fs.NArg() == 1 {
		resolvedModel = fs.Arg(0)
	} else if resolvedModel == "" && fs.NArg() > 1 {
		fmt.Fprintf(stderr, "%s vision: expected exactly one model path\n", cliName())
		fs.Usage()
		return 2
	} else if resolvedModel != "" && fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s vision: model path provided twice\n", cliName())
		fs.Usage()
		return 2
	}
	report, err := visionPlanReportFromOptions(ctx, visionCommandOptions{
		Backend:     *backendName,
		ModelPath:   resolvedModel,
		Images:      *images,
		VideoFrames: *videoFrames,
		FPS:         *fps,
		Prompt:      *prompt,
		MaxTokens:   *maxTokens,
		Chat:        *chatTemplate,
	})
	if err != nil {
		fmt.Fprintf(stderr, "%s vision: %v\n", cliName(), err)
		return 2
	}
	if *jsonOut {
		return writeJSON(stdout, stderr, report)
	}
	printVisionPlanReport(stdout, report)
	return 0
}

func runEbookCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("ebook"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelDir := fs.String("model", "", "model directory to render")
	input := fs.String("input", "", "accepted for compatibility; aliases -model")
	out := fs.String("out", "", "output .epub path")
	output := fs.String("output", "", "accepted for compatibility; aliases -out")
	title := fs.String("title", "", "book title; defaults to model directory name")
	author := fs.String("author", "Lethean", "book author")
	foreword := fs.String("foreword", "", "foreword text file; defaults to <model>/README.md when present")
	weights := fs.Bool("weights", true, "include safetensors weights as base64 plates")
	chapterChars := fs.Int("chapter-chars", 0, "base64 characters per weight plate; 0 = default")
	format := fs.String("format", "epub", "output format; only epub is supported")
	jsonOut := fs.Bool("json", false, "print JSON report")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: %s ebook --model <dir> [--out book.epub] [flags]\n\n", cliName())
		fmt.Fprintln(stderr, "Render a safetensors model directory as an EPUB3 book without loading a model runtime.")
		fmt.Fprintln(stderr, "\nFlags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s ebook: unexpected positional arguments\n", cliName())
		fs.Usage()
		return 2
	}
	if strings.TrimSpace(*format) != "" && strings.ToLower(strings.TrimSpace(*format)) != "epub" {
		fmt.Fprintf(stderr, "%s ebook: unsupported -format %q; only epub is supported\n", cliName(), *format)
		return 2
	}
	resolvedModel := strings.TrimSpace(*modelDir)
	if resolvedModel == "" {
		resolvedModel = strings.TrimSpace(*input)
	}
	if resolvedModel == "" {
		fmt.Fprintf(stderr, "%s ebook: -model is required\n", cliName())
		fs.Usage()
		return 2
	}
	resolvedOutput := strings.TrimSpace(*out)
	if resolvedOutput == "" {
		resolvedOutput = strings.TrimSpace(*output)
	}
	report, err := ebookCommandReportFromOptions(ebookCommandOptions{
		ModelPath:      resolvedModel,
		OutputPath:     resolvedOutput,
		Title:          *title,
		Author:         *author,
		ForewordPath:   *foreword,
		IncludeWeights: *weights,
		ChapterChars:   *chapterChars,
	})
	if err != nil {
		fmt.Fprintf(stderr, "%s ebook: %v\n", cliName(), err)
		return 1
	}
	if *jsonOut {
		return writeJSON(stdout, stderr, report)
	}
	printEbookCommandReport(stdout, report)
	return 0
}

func runSliceCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("slice"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON slice plan")
	preset := fs.String("preset", string(inference.ModelSlicePresetClient), "slice preset: client, attention, embed, server, browse, router, expert_server, full")
	output := fs.String("output", "", "output directory for the materialised slice")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: %s slice [flags] <model-path>\n\n", cliName())
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintf(stderr, "%s slice: expected exactly one model path\n", cliName())
		fs.Usage()
		return 2
	}
	if strings.TrimSpace(*output) == "" {
		fmt.Fprintf(stderr, "%s slice: -output is required\n", cliName())
		fs.Usage()
		return 2
	}
	plan, err := rocm.SliceModel(ctx, inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePreset(*preset),
		Model:      inference.ModelIdentity{Path: fs.Arg(0)},
		OutputPath: *output,
		Labels: map[string]string{
			"cli":          cliName(),
			"cli_contract": cliContractName,
		},
	})
	if err != nil {
		fmt.Fprintf(stderr, "%s slice: %v\n", cliName(), err)
		return 1
	}
	if *jsonOut {
		return writeJSON(stdout, stderr, plan)
	}
	printSliceSummary(stdout, plan)
	return 0
}

func printSliceSummary(stdout io.Writer, plan *inference.ModelSlicePlan) {
	if plan == nil {
		return
	}
	fmt.Fprintf(stdout, "model slice: %s\n", plan.OutputPath)
	fmt.Fprintf(stdout, "  preset: %s, components: %d\n", plan.Preset, len(plan.Components))
	if plan.Labels != nil {
		fmt.Fprintf(stdout, "  tensors: %s, selected bytes: %s / %s\n", plan.Labels["tensor_count"], plan.Labels["selected_tensor_bytes"], plan.Labels["source_tensor_bytes"])
		if plan.Labels["retained_tensor_ratio"] != "" {
			fmt.Fprintf(stdout, "  retained tensor ratio: %s\n", plan.Labels["retained_tensor_ratio"])
		}
	}
}

func runSFTCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("sft"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON native SFT plan")
	backendName := fs.String("backend", defaultCLIBackendName(), "registered backend to load: rocm, cuda, cpu, auto, or any registered backend")
	modelPath := fs.String("model", "", "model path to fine-tune")
	dataPath := fs.String("data", "", "training JSONL")
	validationPath := fs.String("valid", "", "validation JSONL")
	validationSamples := fs.Int("val-samples", 32, "validation samples")
	validationEvery := fs.Int("val-every", 0, "validation cadence in optimizer steps")
	evalPromptsPath := fs.String("eval-prompts", "", "eval prompt file")
	evalEvery := fs.Int("eval-every", 25, "eval cadence in optimizer steps")
	evalMaxTokens := fs.Int("eval-max-tokens", 200, "tokens per eval generation")
	evalProbes := fs.Int("eval-probes", 4, "eval probes derived from validation data")
	scoreCascade := fs.Bool("score-cascade", true, "score eval generations and pick best checkpoint")
	scoreWindow := fs.Int("score-window", 3, "score-cascade window")
	runID := fs.String("run-id", "", "metrics run identity")
	metricsLineProtocolPath := fs.String("metrics-lp", "", "line protocol output")
	influxURL := fs.String("influx-url", "", "InfluxDB write URL")
	influxToken := fs.String("influx-token", "", "InfluxDB API token")
	capturePath := fs.String("capture", "", "eval generation capture JSONL")
	rank := fs.Int("rank", 16, "LoRA rank")
	alpha := fs.Float64("alpha", 32, "LoRA alpha")
	learningRate := fs.Float64("lr", 1e-4, "AdamW learning rate")
	epochs := fs.Int("epochs", 1, "training epochs")
	batchSize := fs.Int("batch", 1, "batch size")
	gradientAccumulation := fs.Int("grad-accum", 4, "gradient accumulation steps")
	maxSequenceLength := fs.Int("max-seq", 1024, "maximum sequence length")
	packing := fs.Bool("packing", false, "sequence packing")
	checkpointDir := fs.String("checkpoint-dir", "", "checkpoint directory")
	checkpointEvery := fs.Int("checkpoint-every", 50, "checkpoint cadence")
	outputAdapterPath := fs.String("save", "", "final adapter path")
	resumeAdapterPath := fs.String("resume", "", "resume adapter checkpoint")
	mergeAfterTraining := fs.Bool("merge", false, "merge adapter after training")
	contextOverride := fs.Int("context", 0, "model context override")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	opts := normalizeSFTCommandOptions(sftCommandOptions{
		BackendName:             normalizeBackendName(*backendName),
		ModelPath:               *modelPath,
		DataPath:                *dataPath,
		ValidationPath:          *validationPath,
		ValidationSamples:       *validationSamples,
		ValidationEvery:         *validationEvery,
		EvalPromptsPath:         *evalPromptsPath,
		EvalEvery:               *evalEvery,
		EvalMaxTokens:           *evalMaxTokens,
		EvalProbes:              *evalProbes,
		ScoreCascade:            *scoreCascade,
		ScoreWindow:             *scoreWindow,
		RunID:                   *runID,
		MetricsLineProtocolPath: *metricsLineProtocolPath,
		InfluxURL:               *influxURL,
		InfluxToken:             *influxToken,
		CapturePath:             *capturePath,
		Rank:                    *rank,
		Alpha:                   *alpha,
		LearningRate:            *learningRate,
		Epochs:                  *epochs,
		BatchSize:               *batchSize,
		GradientAccumulation:    *gradientAccumulation,
		MaxSequenceLength:       *maxSequenceLength,
		Packing:                 *packing,
		CheckpointDir:           *checkpointDir,
		CheckpointEvery:         *checkpointEvery,
		OutputAdapterPath:       *outputAdapterPath,
		ResumeAdapterPath:       *resumeAdapterPath,
		MergeAfterTraining:      *mergeAfterTraining,
		ContextOverride:         *contextOverride,
	})
	if !*jsonOut {
		if err := validateSFTCommandOptions(opts); err != nil {
			fmt.Fprintf(stderr, "%s sft: %v\n", cliName(), err)
			return 2
		}
		if err := validateCLIBackendDispatch("sft", opts.BackendName); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	report, err := sftPlanReportFromOptions(opts)
	if err != nil {
		fmt.Fprintf(stderr, "%s sft: %v\n", cliName(), err)
		return 2
	}
	if *jsonOut {
		return writeJSON(stdout, stderr, report)
	}
	return runSFTExecute(ctx, opts, stdout, stderr)
}

func runFuseCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	if ctx == nil {
		ctx = context.Background()
	}
	fs := flag.NewFlagSet(cliCommandName("fuse"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	backendName := fs.String("backend", defaultBackendName, "registered backend to inspect: rocm, cuda, cpu, auto, or any registered backend")
	jsonOut := fs.Bool("json", false, "print JSON fuse report")
	base := fs.String("base", "", "dense base model directory")
	adapter := fs.String("adapter", "", "trained LoRA adapter directory or adapter file")
	out := fs.String("out", "", "output directory for the fused model")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: %s fuse -base <dir> -adapter <dir> -out <dir>\n\n", cliName())
		fmt.Fprintln(stderr, "Fold a trained LoRA adapter into a local model pack and write a fused")
		fmt.Fprintln(stderr, "servable output directory. Unsupported tensor layouts fail with a JSON")
		fmt.Fprintln(stderr, "diagnostic instead of silently skipping a target.")
		fmt.Fprintln(stderr, "\nFlags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s fuse: unexpected positional arguments\n", cliName())
		fs.Usage()
		return 2
	}
	if strings.TrimSpace(*base) == "" || strings.TrimSpace(*adapter) == "" || strings.TrimSpace(*out) == "" {
		fmt.Fprintf(stderr, "%s fuse: -base, -adapter and -out are all required\n", cliName())
		fs.Usage()
		return 2
	}
	backend := normalizeBackendName(*backendName)
	baseInspection, err := inspectModelPack(ctx, backend, *base)
	if err != nil {
		fmt.Fprintf(stderr, "%s fuse: inspect base: %v\n", cliName(), err)
		return 1
	}
	baseArchitecture := ""
	if baseInspection != nil {
		baseArchitecture = baseInspection.Model.Architecture
	}
	adapterIdentity, err := inspectFuseAdapter(*adapter, baseArchitecture)
	if err != nil {
		fmt.Fprintf(stderr, "%s fuse: inspect adapter: %v\n", cliName(), err)
		return 1
	}
	mergeCapability := fuseMergeCapability(backend)
	result, fuseErr := rocm.FuseLoRAIntoModelPack(ctx, rocm.LoRAFuseOptions{
		BasePath:     *base,
		AdapterPath:  *adapter,
		OutputPath:   *out,
		Architecture: baseArchitecture,
		Adapter:      adapterIdentity,
		Labels: map[string]string{
			"cli_contract": cliContractName,
			"backend":      backend,
		},
	})
	status := "ok"
	detail := "ROCm dense safetensors LoRA fuse completed"
	if fuseErr != nil {
		status = "unsupported"
		detail = fuseErr.Error()
	}
	report := fuseReport{
		Backend:     backend,
		Command:     "fuse",
		CLIContract: cliContractName,
		Feature:     "LoRA on-disk fuse",
		Status:      status,
		Detail:      detail,
		Capability:  mergeCapability,
		Inputs: map[string]string{
			"base":    *base,
			"adapter": *adapter,
			"out":     *out,
		},
		Base:    baseInspection,
		Adapter: adapterIdentity,
		Result:  result,
	}
	if *jsonOut {
		if code := writeJSON(stdout, stderr, report); code != 0 {
			return code
		}
		if fuseErr != nil {
			return 1
		}
		return 0
	}
	if fuseErr != nil {
		fmt.Fprintf(stderr, "%s fuse: %s; capability %s is %s\n", cliName(), report.Detail, mergeCapability.ID, mergeCapability.Status)
		return 1
	}
	fusedLayers := result.FusedWeights
	if len(result.FusedLayers) > 0 {
		fusedLayers = len(result.FusedLayers)
	}
	fmt.Fprintf(stdout, "fused %d layer(s) into %s\n", fusedLayers, result.OutputPath)
	return 0
}

func runSSDCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("ssd"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	defaultSSD := rocm.DefaultSimpleSelfDistillationConfig()
	jsonOut := fs.Bool("json", false, "print JSON native SSD plan")
	backendName := fs.String("backend", defaultCLIBackendName(), "registered backend to load: rocm, cuda, cpu, auto, or any registered backend")
	modelPath := fs.String("model", "", "model path")
	dataPath := fs.String("data", "", "prompt JSONL")
	kernelPath := fs.String("kernel", "", "kernel prefix file")
	sampleMaxTokens := fs.Int("sample-max-tokens", defaultSSD.SampleMaxTokens, "tokens per self-generated sample")
	sampleTemperature := fs.Float64("sample-temp", float64(defaultSSD.SampleTemperature), "sampling temperature; must be non-unit")
	sampleTopK := fs.Int("sample-top-k", defaultSSD.SampleTopK, "sampling top-k")
	sampleTopP := fs.Float64("sample-top-p", float64(defaultSSD.SampleTopP), "sampling top-p")
	sampleMinP := fs.Float64("sample-min-p", float64(defaultSSD.SampleMinP), "sampling min-p")
	repetitionPenalty := fs.Float64("rep-penalty", float64(defaultSSD.RepetitionPenalty), "repetition penalty over self-samples")
	filterShortest := fs.Float64("filter-shortest", float64(defaultSSD.FilterShortestPct), "drop the shortest N percent of self-samples before downstream fine-tuning")
	scoreSamples := fs.Bool("score-samples", true, "score self-generated samples")
	runID := fs.String("run-id", "", "metrics run identity")
	metricsLineProtocolPath := fs.String("metrics-lp", "", "line protocol output")
	influxURL := fs.String("influx-url", "", "InfluxDB write URL")
	influxToken := fs.String("influx-token", "", "InfluxDB API token")
	capturePath := fs.String("capture", "", "trace capture JSONL")
	checkpointDir := fs.String("checkpoint-dir", "", "checkpoint directory")
	contextOverride := fs.Int("context", 0, "model context override")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	opts := ssdCommandOptions{
		BackendName:             normalizeBackendName(*backendName),
		ModelPath:               *modelPath,
		DataPath:                *dataPath,
		KernelPath:              *kernelPath,
		SampleMaxTokens:         *sampleMaxTokens,
		SampleTemperature:       float32(*sampleTemperature),
		SampleTopK:              *sampleTopK,
		SampleTopP:              float32(*sampleTopP),
		SampleMinP:              float32(*sampleMinP),
		RepetitionPenalty:       float32(*repetitionPenalty),
		FilterShortestPercent:   float32(*filterShortest),
		ScoreSamples:            *scoreSamples,
		RunID:                   *runID,
		MetricsLineProtocolPath: *metricsLineProtocolPath,
		InfluxURL:               *influxURL,
		InfluxToken:             *influxToken,
		CapturePath:             *capturePath,
		CheckpointDir:           *checkpointDir,
		ContextOverride:         *contextOverride,
	}
	if !*jsonOut {
		if err := validateSSDCommandOptions(opts); err != nil {
			fmt.Fprintf(stderr, "%s ssd: %v\n", cliName(), err)
			return 2
		}
		if err := validateCLIBackendDispatch("ssd", opts.BackendName); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	report, err := ssdPlanReportFromOptions(opts)
	if err != nil {
		fmt.Fprintf(stderr, "%s ssd: %v\n", cliName(), err)
		return 2
	}
	if *jsonOut {
		return writeJSON(stdout, stderr, report)
	}
	return runSSDExecute(ctx, opts, stdout, stderr)
}

func runTuneCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("tune"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelPath := fs.String("model", "", "Gemma 4 target model path")
	draftPath := fs.String("draft", "auto", "MTP drafter")
	depths := fs.String("depths", defaultTuneDepths, "comma-separated draft blocks to sweep")
	maxTokens := fs.Int("max-tokens", 256, "tokens per measurement run")
	prompt := fs.String("prompt", defaultProductionMeasurementPrompt, "measurement prompt")
	workload := fs.String("workload", "chat", "workload the profile is scored under")
	profileDir := fs.String("profile-dir", "", "tuned-profile directory")
	temp := fs.Float64("temp", 0, "sampling temperature; 0 uses greedy production tuning")
	topP := fs.Float64("top-p", 0, "sampling top-p; 0 disables nucleus sampling")
	topK := fs.Int("top-k", 0, "sampling top-k; 0 disables top-k sampling")
	minP := fs.Float64("min-p", 0, "sampling min-p; 0 disables min-p sampling")
	jsonOut := fs.Bool("json", false, "print JSON reactive MTP tuning plan")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	report, err := tunePlanReportFromOptions(tuneCommandOptions{
		ModelPath:   *modelPath,
		DraftPath:   *draftPath,
		Depths:      *depths,
		MaxTokens:   *maxTokens,
		Prompt:      *prompt,
		Workload:    *workload,
		ProfileDir:  *profileDir,
		Temperature: float32(*temp),
		TopP:        float32(*topP),
		TopK:        *topK,
		MinP:        float32(*minP),
	})
	if err != nil {
		fmt.Fprintf(stderr, "%s tune: %v\n", cliName(), err)
		return 2
	}
	if *jsonOut {
		return writeJSON(stdout, stderr, report)
	}
	return runTuneExecute(ctx, report, stdout, stderr)
}

func runSSDRecipesCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("ssd-recipes"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON report")
	_ = fs.String("workload", "chat", "accepted for CLI contract compatibility")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s ssd-recipes: expected no positional arguments\n", cliName())
		return 2
	}
	report := ssdRecipesReportFromROCmDefaults()
	if *jsonOut {
		if code := writeJSON(stdout, stderr, report); code != 0 {
			return code
		}
		return 0
	}
	printSSDRecipesReport(stdout, report)
	return 0
}

func runSSDEvalCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("ssd-eval"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON eval plan")
	samplesPath := fs.String("samples", "", "LiveCodeBench JSONL path")
	outputPath := fs.String("output", "", "output plan path")
	maxSamples := fs.Int("max-samples", 0, "maximum samples")
	liveCodeBenchV6 := fs.Bool("livecodebench-v6", true, "filter JSONL to the LiveCodeBench-v6 contest-date window")
	nRepeat := fs.Int("n-repeat", 0, "number of generated candidates per task")
	maxTokens := fs.Int("max-tokens", 0, "maximum generated tokens per candidate")
	temperature := fs.Float64("temperature", -1, "sampling temperature")
	topP := fs.Float64("top-p", -1, "sampling top-p")
	topK := fs.Int("top-k", -1, "sampling top-k")
	minP := fs.Float64("min-p", -1, "sampling min-p")
	samplingParams := fs.String("sampling-params", "", "comma-separated sampling params, e.g. temperature=0.9,top_p=0.8,top_k=20")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s ssd-eval: expected no positional arguments\n", cliName())
		return 2
	}
	report, err := ssdEvalPlanReportFromFlags(*samplesPath, *outputPath, *liveCodeBenchV6, *maxSamples, *nRepeat, *maxTokens, *temperature, *topP, *topK, *minP, *samplingParams)
	if err != nil {
		fmt.Fprintf(stderr, "%s ssd-eval: %v\n", cliName(), err)
		return 2
	}
	if *jsonOut {
		if code := writeJSON(stdout, stderr, report); code != 0 {
			return code
		}
		return 0
	}
	printSSDEvalPlanReport(stdout, report)
	return 0
}

func runMemoryPretrainBuildCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("memory-pretrain-build"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON report")
	corpusPath := fs.String("corpus", "", "input corpus JSONL with id, text, and optional string meta")
	routerPath := fs.String("router", "", "output hierarchical router bank JSON")
	ffnMemoryPath := fs.String("ffn-memory", "", "output FFN memory bank JSON")
	hiddenSize := fs.Int("hidden-size", 0, "anchor hidden size / embedding dimension")
	layers := fs.Int("layers", 0, "number of transformer layers to allocate FFN memory for")
	levels := fs.String("levels", "1,2,3,4", "comma-separated memory level names")
	tokens := fs.String("tokens", "8,16,32,64", "comma-separated FFN memory token counts per level")
	branching := fs.Int("branching", 8, "hierarchical KMeans branching factor")
	depth := fs.Int("depth", 3, "hierarchical KMeans max depth")
	minClusterSize := fs.Int("min-cluster-size", 8, "minimum cluster size before splitting")
	kmeansIters := fs.Int("kmeans-iters", 16, "KMeans iterations per split")
	clusterInput := fs.String("cluster-input", "", "optional task JSONL to enrich with cluster_ids")
	clusterOutput := fs.String("cluster-output", "", "output JSONL for -cluster-input")
	taskType := fs.String("task-type", memorypretrain.ClusterIDTaskLanguageModeling, "cluster task type: language_modeling, multiple_choice, generation_task_with_answers, or schema")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: %s memory-pretrain-build [flags]\n\n", cliName())
		fmt.Fprintln(stderr, "Build native hierarchical-memory pretraining artifacts from corpus JSONL.")
		fmt.Fprintln(stderr, "\nFlags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s memory-pretrain-build: expected no positional arguments\n", cliName())
		fs.Usage()
		return 2
	}
	levelNames := parseMemoryPretrainCSV(*levels)
	tokenCounts, err := parseMemoryPretrainInts(*tokens)
	if err != nil {
		fmt.Fprintf(stderr, "%s memory-pretrain-build: %v\n", cliName(), err)
		return 2
	}
	cfg := memorypretrain.MemoryPretrainingArtifactConfig{
		CorpusPath:          strings.TrimSpace(*corpusPath),
		RouterPath:          strings.TrimSpace(*routerPath),
		FFNMemoryPath:       strings.TrimSpace(*ffnMemoryPath),
		Build:               memorypretrain.BuildConfig{BranchingFactor: *branching, MaxDepth: *depth, MinClusterSize: *minClusterSize, KMeansIters: *kmeansIters},
		FFNMemory:           memorypretrain.FFNMemoryConfig{HiddenSize: *hiddenSize, Layers: *layers, MemoryLevels: levelNames, FFNMemoryTokens: tokenCounts},
		ClusterIDInputPath:  strings.TrimSpace(*clusterInput),
		ClusterIDOutputPath: strings.TrimSpace(*clusterOutput),
		ClusterIDJSONL:      memorypretrain.ClusterIDJSONLConfig{TaskType: strings.TrimSpace(*taskType)},
	}
	artifacts, err := memorypretrain.BuildMemoryPretrainingArtifactsFromFiles(ctx, memoryPretrainTextHashEmbedder(*hiddenSize), cfg)
	if err != nil {
		fmt.Fprintf(stderr, "%s memory-pretrain-build: %v\n", cliName(), err)
		return 2
	}
	report := memoryPretrainBuildReport{
		Version:     1,
		Kind:        "memory-pretraining-artifacts",
		Backend:     defaultBackendName,
		Command:     "memory-pretrain-build",
		CLIContract: cliContractName,
		NoPython:    true,
		Embedding:   "text-hash",
		Report:      artifacts.Report,
	}
	if *jsonOut {
		return writeJSON(stdout, stderr, report)
	}
	fmt.Fprintln(stdout, "memory pretraining artifacts")
	if report.Report != nil {
		fmt.Fprintf(stdout, "  corpus: %d records\n", report.Report.CorpusRecords)
		fmt.Fprintf(stdout, "  router: %d nodes -> %s\n", report.Report.RouterNodes, report.Report.RouterPath)
		fmt.Fprintf(stdout, "  ffn memory: %d layers -> %s\n", report.Report.FFNMemoryLayers, report.Report.FFNMemoryPath)
		if report.Report.ClusterIDReport != nil {
			fmt.Fprintf(stdout, "  cluster ids: %d learned row(s) -> %s\n", report.Report.ClusterIDReport.LearnedRows, report.Report.ClusterIDOutput)
		}
	}
	return 0
}

func parseMemoryPretrainCSV(raw string) []string {
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func parseMemoryPretrainInts(raw string) ([]int, error) {
	parts := parseMemoryPretrainCSV(raw)
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		value, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid integer %q", part)
		}
		out = append(out, value)
	}
	return out, nil
}

func memoryPretrainTextHashEmbedder(dim int) memorypretrain.Embedder {
	return memorypretrain.EmbedFunc(func(ctx context.Context, text string) ([]float32, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if dim <= 0 {
			return nil, core.NewError("memorypretrain: text-hash embedding dimension must be positive")
		}
		out := make([]float32, dim)
		h := fnv.New32a()
		var salt [2]byte
		for _, token := range strings.Fields(text) {
			tokenBytes := core.AsBytes(token)
			for i := range out {
				salt[0] = byte(i)
				salt[1] = byte(i >> 8)
				h.Reset()
				_, _ = h.Write(tokenBytes)
				_, _ = h.Write(salt[:])
				bucket := int(h.Sum32()%2001) - 1000
				out[i] += float32(bucket) / 1000
			}
		}
		var norm float64
		for _, value := range out {
			norm += float64(value * value)
		}
		if norm == 0 {
			out[0] = 1
			return out, nil
		}
		scale := float32(1 / math.Sqrt(norm))
		for i := range out {
			out[i] *= scale
		}
		return out, nil
	})
}

func writeJSON(stdout, stderr io.Writer, v any) int {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "%s: marshal JSON: %v\n", cliName(), err)
		return 1
	}
	fmt.Fprintln(stdout, string(data))
	return 0
}

func inspectModelPack(ctx context.Context, backendName, path string) (*inference.ModelPackInspection, error) {
	backendName = normalizeBackendName(backendName)
	if backendName == "" || backendName == defaultBackendName {
		return rocm.InspectModelPack(ctx, path)
	}
	backend, ok := inference.Get(backendName)
	if !ok {
		return nil, fmt.Errorf("backend %q is not registered", backendName)
	}
	inspector, ok := backend.(inference.ModelPackInspector)
	if !ok {
		return nil, fmt.Errorf("backend %q does not expose model-pack inspection", backendName)
	}
	return inspector.InspectModelPack(ctx, path)
}

func inspectFuseAdapter(path string, architecture string) (inference.AdapterIdentity, error) {
	path = strings.TrimSpace(path)
	info, err := os.Stat(path)
	if err != nil {
		return inference.AdapterIdentity{}, err
	}
	identity := inference.AdapterIdentity{
		Path:   path,
		Format: "lora",
		Labels: map[string]string{
			"adapter_path_kind": "file",
		},
	}
	if !info.IsDir() {
		identity.Labels["adapter_file"] = filepath.Base(path)
		if strings.EqualFold(filepath.Ext(path), ".safetensors") {
			configPath := filepath.Join(filepath.Dir(path), "adapter_config.json")
			if _, err := os.Stat(configPath); err == nil {
				if err := applyPEFTAdapterConfig(&identity, configPath, architecture); err != nil {
					return inference.AdapterIdentity{}, err
				}
			}
		}
		return identity, nil
	}
	identity.Labels["adapter_path_kind"] = "directory"
	configPath := filepath.Join(path, "adapter_config.json")
	if _, err := os.Stat(configPath); err == nil {
		identity.Labels["adapter_file"] = "adapter_config.json"
		if err := applyPEFTAdapterConfig(&identity, configPath, architecture); err != nil {
			return inference.AdapterIdentity{}, err
		}
		if _, err := os.Stat(filepath.Join(path, "adapter.safetensors")); err == nil {
			identity.Labels["adapter_weights_file"] = "adapter.safetensors"
		}
		return identity, nil
	}
	for _, candidate := range []struct {
		file   string
		format string
	}{
		{file: "adapter.safetensors", format: "lora"},
		{file: "rocm_lm_head_lora.json", format: "rocm-small-lm-head-lora"},
		{file: "rocm_tiny_lora.json", format: "rocm-tiny-lora"},
		{file: "rocm_classifier_lora.json", format: "rocm-classifier-lora"},
	} {
		candidatePath := filepath.Join(path, candidate.file)
		if _, err := os.Stat(candidatePath); err == nil {
			identity.Format = candidate.format
			identity.Labels["adapter_file"] = candidate.file
			return identity, nil
		}
	}
	return inference.AdapterIdentity{}, fmt.Errorf("%s does not contain adapter_config.json, adapter.safetensors, or a ROCm LoRA adapter JSON", path)
}

func applyPEFTAdapterConfig(identity *inference.AdapterIdentity, configPath string, architecture string) error {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("read adapter_config.json: %w", err)
	}
	var cfg peftAdapterConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("parse adapter_config.json: %w", err)
	}
	if identity.Labels == nil {
		identity.Labels = map[string]string{}
	}
	identity.Format = "lora"
	identity.Labels["adapter_config_file"] = filepath.Base(configPath)
	identity.Labels["adapter_config_format"] = "peft"
	if cfg.PeftType != "" {
		identity.Labels["adapter_peft_type"] = strings.TrimSpace(cfg.PeftType)
	}
	if cfg.TaskType != "" {
		identity.Labels["adapter_task_type"] = strings.TrimSpace(cfg.TaskType)
	}
	if cfg.BaseModelNameOrPath != "" {
		identity.Labels["adapter_base_model_name_or_path"] = strings.TrimSpace(cfg.BaseModelNameOrPath)
	}
	if rank := firstPositiveAdapterInt(cfg.R, cfg.Rank); rank > 0 {
		identity.Rank = rank
		identity.Labels["adapter_rank"] = strconv.Itoa(rank)
	}
	if alpha := firstPositiveFloat64(cfg.LoRAAlpha, cfg.Alpha); alpha > 0 {
		identity.Alpha = float32(alpha)
		identity.Labels["adapter_alpha"] = formatAdapterFloat(alpha)
	}
	scale := firstPositiveFloat64(cfg.Scale)
	if scale == 0 && identity.Rank > 0 && identity.Alpha == 0 {
		identity.Alpha = float32(identity.Rank) * 2
		identity.Labels["adapter_alpha"] = formatAdapterFloat(float64(identity.Alpha))
		identity.Labels["adapter_alpha_source"] = "default_rank_x2"
	}
	if scale == 0 && identity.Rank > 0 && identity.Alpha > 0 {
		scale = float64(identity.Alpha) / float64(identity.Rank)
	}
	if scale > 0 {
		identity.Labels["adapter_scale"] = formatAdapterFloat(scale)
	}

	targetModules, err := adapterConfigStringList(cfg.TargetModules)
	if err != nil {
		return fmt.Errorf("parse adapter_config.json target_modules: %w", err)
	}
	targetParameters, err := adapterConfigStringList(cfg.TargetParameters)
	if err != nil {
		return fmt.Errorf("parse adapter_config.json target_parameters: %w", err)
	}
	targets := appendUniqueAdapterStrings(nil, targetModules...)
	targets = appendUniqueAdapterStrings(targets, targetParameters...)
	if len(targets) > 0 {
		identity.TargetKeys = targets
		identity.Labels["adapter_target_count"] = strconv.Itoa(len(targets))
	}
	if len(targetModules) > 0 {
		identity.Labels["adapter_target_modules"] = strings.Join(targetModules, ",")
	}
	if len(targetParameters) > 0 {
		identity.Labels["adapter_target_parameters"] = strings.Join(targetParameters, ",")
	}
	applyGemma4FuseTargetLabels(identity, architecture)
	return nil
}

func adapterConfigStringList(raw json.RawMessage) ([]string, error) {
	raw = []byte(strings.TrimSpace(string(raw)))
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var values []string
	if err := json.Unmarshal(raw, &values); err == nil {
		return trimAdapterStrings(values), nil
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		return trimAdapterStrings([]string{single}), nil
	}
	var keyed map[string]bool
	if err := json.Unmarshal(raw, &keyed); err == nil {
		values := make([]string, 0, len(keyed))
		for key, enabled := range keyed {
			if enabled {
				values = append(values, key)
			}
		}
		sort.Strings(values)
		return trimAdapterStrings(values), nil
	}
	return nil, fmt.Errorf("expected string, string array, or string bool map")
}

func applyGemma4FuseTargetLabels(identity *inference.AdapterIdentity, architecture string) {
	policy, ok := rocm.ROCmLoRATargetPolicyForArchitecture(architecture)
	if !ok {
		return
	}
	if identity.Labels == nil {
		identity.Labels = map[string]string{}
	}
	identity.Labels["adapter_target_policy"] = "gemma4"
	identity.Labels["adapter_default_targets"] = strings.Join(policy.DefaultTargets, ",")
	identity.Labels["adapter_safe_targets"] = strings.Join(policy.SafeTargets, ",")
	identity.Labels["adapter_extended_targets"] = strings.Join(policy.ExtendedTargets, ",")
	if len(identity.TargetKeys) == 0 {
		identity.Labels["adapter_targets_defaultable"] = "true"
		return
	}
	canonicalTargets := make([]string, 0, len(identity.TargetKeys))
	safeTargets := make([]string, 0, len(identity.TargetKeys))
	extendedTargets := make([]string, 0)
	unknownTargets := make([]string, 0)
	for _, target := range identity.TargetKeys {
		canonical, safe, extended, ok := gemma4FuseCanonicalTarget(architecture, target)
		if !ok {
			unknownTargets = append(unknownTargets, target)
			continue
		}
		canonicalTargets = appendUniqueAdapterStrings(canonicalTargets, canonical)
		if safe {
			safeTargets = appendUniqueAdapterStrings(safeTargets, canonical)
		}
		if extended {
			extendedTargets = appendUniqueAdapterStrings(extendedTargets, canonical)
		}
	}
	if len(canonicalTargets) > 0 {
		identity.Labels["adapter_canonical_targets"] = strings.Join(canonicalTargets, ",")
		identity.Labels["adapter_canonical_target_count"] = strconv.Itoa(len(canonicalTargets))
	}
	if len(safeTargets) > 0 {
		identity.Labels["adapter_safe_targets_present"] = strings.Join(safeTargets, ",")
		identity.Labels["adapter_safe_target_count"] = strconv.Itoa(len(safeTargets))
	}
	if len(extendedTargets) > 0 {
		identity.Labels["adapter_extended_targets_present"] = strings.Join(extendedTargets, ",")
		identity.Labels["adapter_extended_target_count"] = strconv.Itoa(len(extendedTargets))
		identity.Labels["adapter_extended_targets_require_opt_in"] = "true"
	}
	if len(unknownTargets) > 0 {
		identity.Labels["adapter_unknown_targets"] = strings.Join(unknownTargets, ",")
		identity.Labels["adapter_unknown_target_count"] = strconv.Itoa(len(unknownTargets))
	}
	if len(extendedTargets) == 0 && len(unknownTargets) == 0 {
		identity.Labels["adapter_safe_for_default_fuse"] = "true"
	} else {
		identity.Labels["adapter_safe_for_default_fuse"] = "false"
	}
}

func gemma4FuseCanonicalTarget(architecture string, target string) (string, bool, bool, bool) {
	target = strings.TrimSpace(target)
	if target == "" {
		return "", false, false, false
	}
	parts := strings.Split(target, ".")
	if len(parts) >= 2 {
		short := parts[len(parts)-2] + "." + parts[len(parts)-1]
		if canonical, ok := rocm.ROCmLoRATargetPath(architecture, short); ok {
			return joinAdapterTarget(parts[:len(parts)-2], canonical), rocm.ROCmLoRASafeTarget(architecture, short), rocm.ROCmLoRAExtendedTarget(architecture, short), true
		}
		if len(parts) == 2 {
			return "", false, false, false
		}
	}
	short := parts[len(parts)-1]
	if canonical, ok := rocm.ROCmLoRATargetPath(architecture, short); ok {
		return joinAdapterTarget(parts[:len(parts)-1], canonical), rocm.ROCmLoRASafeTarget(architecture, short), rocm.ROCmLoRAExtendedTarget(architecture, short), true
	}
	return "", false, false, false
}

func joinAdapterTarget(prefix []string, canonical string) string {
	if len(prefix) == 0 {
		return canonical
	}
	parts := make([]string, 0, len(prefix)+strings.Count(canonical, ".")+1)
	parts = append(parts, prefix...)
	parts = append(parts, strings.Split(canonical, ".")...)
	return strings.Join(parts, ".")
}

func trimAdapterStrings(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			out = append(out, value)
		}
	}
	return out
}

func appendUniqueAdapterStrings(out []string, values ...string) []string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		exists := false
		for _, existing := range out {
			if existing == value {
				exists = true
				break
			}
		}
		if !exists {
			out = append(out, value)
		}
	}
	return out
}

func firstPositiveFloat64(values ...float64) float64 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func firstPositiveAdapterInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func formatAdapterFloat(value float64) string {
	return strconv.FormatFloat(value, 'g', -1, 64)
}

func fuseMergeCapability(backendName string) inference.Capability {
	report := inference.BackendCapabilities(nil)
	if backend, ok := inference.Get(backendName); ok {
		if caps, ok := inference.CapabilitiesOf(backend); ok {
			report = caps
		}
	}
	if capability, ok := report.Capability(inference.CapabilityModelMerge); ok {
		return capability
	}
	return inference.PlannedCapability(inference.CapabilityModelMerge, inference.CapabilityGroupRuntime, "model-pack merge is not implemented in the selected backend yet")
}

func normalizeBackendName(name string) string {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "", defaultBackendName:
		return defaultBackendName
	case "auto":
		return "auto"
	default:
		return strings.TrimSpace(name)
	}
}

func compatibilityStub(command, feature string, _ io.Writer, stderr io.Writer) int {
	fmt.Fprintf(stderr, "%s %s: %s is accepted at the %s CLI boundary but is not implemented for go-rocm yet\n", cliName(), command, feature, cliContractName)
	return 1
}

func explicitReactiveDraft(draftPath string, draftBlock int) bool {
	draftPath = strings.TrimSpace(draftPath)
	return draftBlock > 0 || (draftPath != "" && draftPath != "auto")
}

func rocmCLIGenerateOptions(maxTokens int, temp float32, enableThinking bool) []inference.GenerateOption {
	opts := []inference.GenerateOption{
		inference.WithTemperature(temp),
		inference.WithEnableThinking(&enableThinking),
	}
	if maxTokens > 0 {
		opts = append([]inference.GenerateOption{inference.WithMaxTokens(maxTokens)}, opts...)
	}
	return opts
}

func refuseGenerateReactiveDraftMode(mode string, detection rocm.DraftDetection, draftBlock int, stderr io.Writer) int {
	fmt.Fprintf(stderr, "%s generate: reactive MTP drafter resolved for %s: %s (%s), block %s\n",
		cliName(), mode, detection.DraftPath, detection.Note, resolvedROCmDraftBlockLabel(draftBlock))
	fmt.Fprintf(stderr, "%s generate: %s does not run native ROCm attached-drafter execution yet; use -draft '' to run target-only %s\n",
		cliName(), mode, mode)
	return 1
}

func refuseGenerateReactiveDraftSampling(mode string, detection rocm.DraftDetection, draftBlock int, temp float64, stderr io.Writer) int {
	fmt.Fprintf(stderr, "%s generate: reactive MTP drafter resolved for %s: %s (%s), block %s\n",
		cliName(), mode, detection.DraftPath, detection.Note, resolvedROCmDraftBlockLabel(draftBlock))
	fmt.Fprintf(stderr, "%s generate: native ROCm attached-drafter execution requires greedy generation (-temp 0); got -temp %g\n",
		cliName(), temp)
	return 2
}

func resolveROCmDraft(modelPath, draftFlag string, detect bool) rocm.DraftDetection {
	explicit := ""
	opts := rocm.DraftDetectOptions{Disabled: !detect}
	switch trimmed := strings.TrimSpace(draftFlag); trimmed {
	case "auto":
	case "":
		opts.Disabled = true
	default:
		explicit = trimmed
	}
	return rocm.DetectGemma4DraftPath(modelPath, explicit, opts)
}

func applyROCmDraftDetectionLabels(labels map[string]string, detection rocm.DraftDetection, draftBlock int) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if detection.Active() {
		labels["reactive_draft_detection"] = "active_pending_native_drafter"
		labels["reactive_draft_source"] = string(detection.Source)
		labels["reactive_draft_path"] = detection.DraftPath
		labels["reactive_draft_note"] = detection.Note
		labels["reactive_draft_block"] = resolvedROCmDraftBlockLabel(draftBlock)
		labels["reactive_draft_tokens"] = resolvedROCmDraftTokensLabel(draftBlock)
		labels["reactive_draft_fallback"] = "target_retained_decode"
		labels["native_mtp_attachment"] = "not_linked"
		return labels
	}
	if detection.Note == "drafter detection disabled" {
		labels["reactive_draft_detection"] = "disabled"
		return labels
	}
	labels["reactive_draft_detection"] = "standing_by"
	return labels
}

func resolvedROCmDraftBlockLabel(draftBlock int) string {
	if draftBlock > 0 {
		return strconv.Itoa(draftBlock)
	}
	return "backend_default"
}

func resolvedROCmDraftTokensLabel(draftBlock int) string {
	if draftBlock > 1 {
		return strconv.Itoa(draftBlock - 1)
	}
	return "backend_default"
}

func resolvedROCmDraftBlockSourceSuffix(source string) string {
	source = strings.TrimSpace(source)
	if source == "" {
		return ""
	}
	return " (" + source + ")"
}

func validatePositiveCSV(value, label string) error {
	for _, part := range strings.Split(value, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil || n <= 0 {
			return fmt.Errorf("invalid %s %q", label, part)
		}
	}
	return nil
}

func defaultString(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

func firstPositiveInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func maxPositiveInt(a, b int) int {
	if a <= 0 {
		return b
	}
	if b <= 0 {
		return a
	}
	if a > b {
		return a
	}
	return b
}

func cliTuningWorkloads(value string) ([]inference.TuningWorkload, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil, nil
	}
	workload := inference.TuningWorkload(value)
	if !cliValidTuningWorkload(workload) {
		return nil, fmt.Errorf("unsupported workload %q", value)
	}
	return []inference.TuningWorkload{workload}, nil
}

func cliTuningWorkloadsOrDefault(workloads []inference.TuningWorkload) []inference.TuningWorkload {
	if len(workloads) == 0 {
		return inference.DefaultTuningWorkloads()
	}
	return append([]inference.TuningWorkload(nil), workloads...)
}

func cliValidTuningWorkload(workload inference.TuningWorkload) bool {
	switch workload {
	case inference.TuningWorkloadChat,
		inference.TuningWorkloadCoding,
		inference.TuningWorkloadLongContext,
		inference.TuningWorkloadAgentState,
		inference.TuningWorkloadThroughput,
		inference.TuningWorkloadLowLatency:
		return true
	default:
		return false
	}
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func applyROCmTokenLoopLabels(labels map[string]string, status *rocm.ROCmTokenLoopStatus) map[string]string {
	if status == nil || !status.Matched() || len(status.Labels) == 0 {
		return labels
	}
	out := cloneStringMap(labels)
	if out == nil {
		out = map[string]string{}
	}
	for key, value := range status.Labels {
		out[key] = value
	}
	return out
}

func printUsage(w io.Writer) {
	name := cliName()
	fmt.Fprintf(w, "Usage: %s <command> [flags]\n\n", name)
	fmt.Fprintln(w, "Run inference")
	fmt.Fprintln(w, "  menubar             tray-only macOS app compatibility stub")
	fmt.Fprintln(w, "  serve               accept OpenAI/Anthropic/Ollama HTTP API flags")
	fmt.Fprintln(w, "  daemon              run local JSON-lines daemon over a Unix socket")
	fmt.Fprintln(w, "  generate            one-shot ROCm generation")
	fmt.Fprintln(w, "  diffuse             inspect DiffusionGemma block-diffusion readiness")
	fmt.Fprintln(w, "  audio               inspect Gemma audio runtime readiness")
	fmt.Fprintln(w, "  vision              inspect Gemma vision runtime readiness")
	fmt.Fprintln(w, "  ebook               render a safetensors model directory as EPUB3")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Inspect what is installed")
	fmt.Fprintln(w, "  discover            report local ROCm runtime + optional model packs")
	fmt.Fprintln(w, "  pack                validate a local native model pack")
	fmt.Fprintln(w, "  reactive            inspect config-composed sequence-mixer readiness")
	fmt.Fprintln(w, "  bench               emit Gemma-4 QAT/MTP-QAT production bench plan")
	fmt.Fprintln(w, "  ssd-recipes         Simple Self-Distillation recipe manifest")
	fmt.Fprintln(w, "  ssd-eval            Simple Self-Distillation LiveCodeBench eval plan")
	fmt.Fprintln(w, "  memory-pretrain-build  build hierarchical-memory pretraining artifacts")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Transform a model")
	fmt.Fprintln(w, "  slice               materialise a model-slice safetensors subset")
	fmt.Fprintln(w, "  fuse                fold a trained LoRA adapter into a model pack")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Train and tune")
	fmt.Fprintln(w, "  sft                 native LoRA SFT plan")
	fmt.Fprintln(w, "  ssd                 native Simple Self-Distillation trace plan")
	fmt.Fprintln(w, "  tune                reactive MTP tuning plan")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "State container ops")
	fmt.Fprintln(w, "  state-pack          pack a state marker and binary log into a portable .kv")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Examples")
	fmt.Fprintf(w, "  %s discover -json\n", name)
	fmt.Fprintf(w, "  %s pack ~/models/lemer-lite\n", name)
	fmt.Fprintf(w, "  %s generate -backend rocm -prompt 'hello' ~/models/lemer-lite\n", name)
	fmt.Fprintf(w, "\nRun \"%s <command> -h\" for command-specific flags.\n", name)
}
