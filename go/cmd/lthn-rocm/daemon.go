// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"strconv"
	"strings"

	"dappco.re/go/inference"
	rocm "dappco.re/go/rocm"
	"dappco.re/go/rocm/daemon"
)

type cliModelMapFlag map[string]string

type daemonModelRegistryBackend struct {
	Backend string
}

type daemonModelProfileBackend struct {
	Backend          string
	DefaultModelName string
	ModelPaths       map[string]string
}

func (backend daemonModelRegistryBackend) ModelRegistry(ctx context.Context, req daemon.ModelRegistryRequest) (any, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	backendName := firstDaemonNonEmptyString(req.Backend, backend.Backend, defaultBackendName)
	return currentModelRegistryReport(backendName), nil
}

func (backend daemonModelProfileBackend) ModelProfile(ctx context.Context, req daemon.ModelProfileRequest) (any, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	modelName := strings.TrimSpace(req.Model)
	if modelName == "" {
		modelName = firstDaemonNonEmptyString(backend.DefaultModelName, daemon.DefaultModelName)
	}
	modelPath := strings.TrimSpace(req.Path)
	if modelPath == "" && modelName != "" {
		modelPath = strings.TrimSpace(backend.ModelPaths[modelName])
	}
	if modelPath == "" {
		return nil, errors.New("model profile path or known model is required")
	}
	profile, ok := daemonResolveROCmModelProfile(modelPath, modelName, req.Labels)
	if !ok || !profile.Matched() {
		return nil, fmt.Errorf("model profile not found for %s", modelPath)
	}
	return profile, nil
}

func (backend daemonModelProfileBackend) EngineFeatures(ctx context.Context, req daemon.EngineFeaturesRequest) (any, error) {
	profile, err := backend.ModelProfile(ctx, daemon.ModelProfileRequest{
		Path:    req.Path,
		Model:   req.Model,
		Backend: req.Backend,
		Labels:  req.Labels,
	})
	if err != nil {
		return nil, err
	}
	rocmProfile, ok := profile.(rocm.ROCmModelProfile)
	if !ok || !rocmProfile.Matched() {
		return nil, errors.New("ROCm model profile is required for engine features")
	}
	return rocm.ROCmEngineFeaturesForProfile(rocmProfile), nil
}

func (backend daemonModelProfileBackend) ModelRoutes(ctx context.Context, req daemon.ModelRoutesRequest) (any, error) {
	profile, err := backend.ModelProfile(ctx, daemon.ModelProfileRequest{
		Path:    req.Path,
		Model:   req.Model,
		Backend: req.Backend,
		Labels:  req.Labels,
	})
	if err != nil {
		return nil, err
	}
	rocmProfile, ok := profile.(rocm.ROCmModelProfile)
	if !ok || !rocmProfile.Matched() {
		return nil, errors.New("ROCm model profile is required for route planning")
	}
	plan := rocm.ROCmModelRoutePlanForProfile(rocmProfile)
	if !plan.Matched() {
		return nil, errors.New("ROCm model routes not found")
	}
	return plan, nil
}

func daemonResolveROCmModelProfile(path, modelName string, labels map[string]string) (rocm.ROCmModelProfile, bool) {
	path = strings.TrimSpace(path)
	if path == "" {
		return rocm.ROCmModelProfile{}, false
	}
	identity := inference.ModelIdentity{
		ID:     strings.TrimSpace(modelName),
		Path:   path,
		Labels: cloneStringMap(labels),
	}
	if probe, err := rocm.ProbeROCmModelConfigFile(filepath.Join(path, "config.json")); err == nil && probe.ArchitectureResolution.Matched() {
		identity.Architecture = probe.ArchitectureResolution.Architecture
		identity.Labels = mergeDaemonStringLabels(identity.Labels, probe.Labels)
	}
	return rocm.ResolveROCmModelProfile(path, identity)
}

func mergeDaemonStringLabels(base, extra map[string]string) map[string]string {
	if len(base) == 0 && len(extra) == 0 {
		return nil
	}
	merged := cloneStringMap(base)
	if merged == nil {
		merged = map[string]string{}
	}
	for key, value := range extra {
		if strings.TrimSpace(key) != "" && value != "" {
			merged[key] = value
		}
	}
	return merged
}

func (flagValue *cliModelMapFlag) String() string {
	if flagValue == nil || len(*flagValue) == 0 {
		return ""
	}
	parts := make([]string, 0, len(*flagValue))
	for name, path := range *flagValue {
		parts = append(parts, name+"="+path)
	}
	return strings.Join(parts, ",")
}

func (flagValue *cliModelMapFlag) Set(value string) error {
	value = strings.TrimSpace(value)
	if value == "" {
		return errors.New("model entry is required")
	}
	name := daemon.DefaultModelName
	path := value
	if before, after, ok := strings.Cut(value, "="); ok {
		name = strings.TrimSpace(before)
		path = strings.TrimSpace(after)
	}
	if name == "" {
		return errors.New("model name is required")
	}
	if path == "" {
		return errors.New("model path is required")
	}
	if *flagValue == nil {
		*flagValue = map[string]string{}
	}
	(*flagValue)[name] = path
	return nil
}

func runDaemonCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("daemon"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	configPath := fs.String("config", "", "path to lthn.toml daemon config")
	socketPath := fs.String("socket", "", "Unix socket path (default: platform runtime cache path)")
	defaultModel := fs.String("model-name", daemon.DefaultModelName, "default model name used when requests omit model")
	defaultMaxTokens := fs.Int("max-tokens", 0, "default max tokens for daemon generate requests")
	contextLen := fs.Int("context", 0, "context length for loaded models")
	gpuLayers := fs.Int("gpu-layers", -1, "GPU layers for loaded models")
	parallelSlots := fs.Int("parallel", 0, "parallel slots for loaded models")
	kvCache := fs.String("kv-cache", "", "ROCm native KV cache mode")
	adapterPath := fs.String("adapter", "", "LoRA adapter path for loaded models")
	jsonOut := fs.Bool("json", false, "print daemon config and exit without serving")
	showVersion := fs.Bool("version", false, "print daemon version and exit")
	models := cliModelMapFlag{}
	fs.Var(&models, "model", "model path or name=path; repeat to expose multiple models")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: %s daemon [--model <path>|--model name=path] [flags]\n\n", cliName())
		fmt.Fprintln(stderr, "Run the local JSON-lines inference daemon over a Unix socket.")
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
		fmt.Fprintf(stderr, "%s daemon: unexpected positional arguments\n", cliName())
		fs.Usage()
		return 2
	}
	flagsSet := daemonFlagSet(fs)
	if *showVersion {
		fmt.Fprintf(stdout, "%s daemon %s\n", cliName(), daemon.DefaultVersion)
		return 0
	}

	runtimeCfg, err := loadDaemonRuntimeConfig(*configPath)
	if err != nil {
		fmt.Fprintf(stderr, "%s daemon: %v\n", cliName(), err)
		return 1
	}
	if flagsSet["socket"] {
		runtimeCfg.SocketPath = strings.TrimSpace(*socketPath)
	}
	if flagsSet["model-name"] {
		runtimeCfg.DefaultModelName = strings.TrimSpace(*defaultModel)
	}
	if flagsSet["max-tokens"] {
		runtimeCfg.DefaultMaxTokens = *defaultMaxTokens
	}
	if flagsSet["context"] {
		runtimeCfg.ContextLen = *contextLen
	}
	if flagsSet["gpu-layers"] {
		runtimeCfg.GPULayers = *gpuLayers
	}
	if flagsSet["parallel"] {
		runtimeCfg.ParallelSlots = *parallelSlots
	}
	if flagsSet["kv-cache"] {
		runtimeCfg.KVCache = strings.TrimSpace(*kvCache)
	}
	if flagsSet["adapter"] {
		runtimeCfg.AdapterPath = strings.TrimSpace(*adapterPath)
	}
	if runtimeCfg.Models == nil {
		runtimeCfg.Models = map[string]string{}
	}
	for name, path := range models {
		runtimeCfg.Models[name] = path
	}
	if strings.TrimSpace(runtimeCfg.DefaultModelName) == "" {
		runtimeCfg.DefaultModelName = daemon.DefaultModelName
	}

	if runtimeCfg.DefaultMaxTokens < 0 {
		fmt.Fprintf(stderr, "%s daemon: -max-tokens must be >= 0\n", cliName())
		return 2
	}
	if runtimeCfg.ContextLen < 0 {
		fmt.Fprintf(stderr, "%s daemon: -context must be >= 0\n", cliName())
		return 2
	}
	if runtimeCfg.ParallelSlots < 0 {
		fmt.Fprintf(stderr, "%s daemon: -parallel must be >= 0\n", cliName())
		return 2
	}

	cfg, err := newDaemonServerConfig(runtimeCfg)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	if *jsonOut {
		reportRegistry := daemon.DefaultRegistryForDaemon()
		if err := reportRegistry.RegisterEngineFeaturesBackend(daemonModelProfileBackend{
			Backend:          defaultBackendName,
			DefaultModelName: strings.TrimSpace(runtimeCfg.DefaultModelName),
			ModelPaths:       cloneStringMap(runtimeCfg.Models),
		}); err != nil {
			fmt.Fprintf(stderr, "%s daemon: %v\n", cliName(), err)
			return 1
		}
		if err := reportRegistry.RegisterModelRegistryBackend(daemonModelRegistryBackend{Backend: defaultBackendName}); err != nil {
			fmt.Fprintf(stderr, "%s daemon: %v\n", cliName(), err)
			return 1
		}
		if err := reportRegistry.RegisterModelProfileBackend(daemonModelProfileBackend{
			Backend:          defaultBackendName,
			DefaultModelName: strings.TrimSpace(runtimeCfg.DefaultModelName),
			ModelPaths:       cloneStringMap(runtimeCfg.Models),
		}); err != nil {
			fmt.Fprintf(stderr, "%s daemon: %v\n", cliName(), err)
			return 1
		}
		if err := reportRegistry.RegisterModelRoutesBackend(daemonModelProfileBackend{
			Backend:          defaultBackendName,
			DefaultModelName: strings.TrimSpace(runtimeCfg.DefaultModelName),
			ModelPaths:       cloneStringMap(runtimeCfg.Models),
		}); err != nil {
			fmt.Fprintf(stderr, "%s daemon: %v\n", cliName(), err)
			return 1
		}
		if err := reportRegistry.RegisterModelPackBackend(daemon.ROCmModelPackInspector{}); err != nil {
			fmt.Fprintf(stderr, "%s daemon: %v\n", cliName(), err)
			return 1
		}
		report := map[string]any{
			"backend":            defaultBackendName,
			"command":            "daemon",
			"cli_contract":       cliContractName,
			"daemon":             daemon.DaemonName,
			"socket":             cfg.SocketPath,
			"default_model":      firstDaemonNonEmptyString(cfg.NativeGenerate.DefaultModelName, daemon.DefaultModelName),
			"default_max_tokens": cfg.NativeGenerate.DefaultMaxTokens,
			"models":             cfg.ModelPaths,
			"model_count":        len(cfg.ModelPaths),
			"model_registry":     currentModelRegistryReport(defaultBackendName),
			"routes":             reportRegistry.Routes(),
			"load": map[string]any{
				"context":    runtimeCfg.ContextLen,
				"gpu_layers": runtimeCfg.GPULayers,
				"parallel":   runtimeCfg.ParallelSlots,
				"adapter":    strings.TrimSpace(runtimeCfg.AdapterPath),
				"kv_cache":   strings.TrimSpace(runtimeCfg.KVCache),
			},
			"labels": map[string]string{
				"daemon_contract":                       "json-lines-unix-v1",
				"daemon_actions":                        strings.Join(reportRegistry.Actions(), ","),
				"production_requires_env_gate":          strconv.FormatBool(false),
				"production_requires_cli_flag":          strconv.FormatBool(false),
				"reactive_daemon_embed_route":           "native_model_if_supported",
				"reactive_daemon_generate_route":        "accepted",
				"reactive_daemon_schedule_route":        "native_model_if_supported",
				"reactive_daemon_rerank_route":          "native_model_if_supported",
				"reactive_daemon_score_route":           "heuristic",
				"reactive_daemon_cache_route":           "native_model_if_supported",
				"reactive_daemon_cancel_route":          "native_model_if_supported",
				"reactive_daemon_health_route":          "system",
				"reactive_daemon_models_route":          "native_model_if_supported",
				"reactive_daemon_capabilities_route":    "native_model_if_supported",
				"reactive_daemon_engine_features_route": "native_metadata",
				"reactive_daemon_model_pack_route":      "native_metadata",
				"reactive_daemon_model_profile_route":   "native_metadata",
				"reactive_daemon_model_routes_route":    "native_metadata",
				"reactive_daemon_registry_route":        "native_metadata",
				"reactive_daemon_tokenizer_route":       "native_model_if_supported",
				"reactive_daemon_parser_route":          "native_model_if_supported",
				"reactive_daemon_backend":               defaultBackendName,
				"reactive_daemon_socket_protocol":       "jsonl",
			},
		}
		return writeJSON(stdout, stderr, report)
	}

	server := daemon.NewServer(cfg)
	if err := server.ListenAndServe(ctx); err != nil {
		fmt.Fprintf(stderr, "%s daemon: %v\n", cliName(), err)
		return 1
	}
	return 0
}

func newDaemonServerConfig(runtimeCfg daemonRuntimeConfig) (daemon.ServerConfig, error) {
	loadOpts := []inference.LoadOption{inference.WithBackend(defaultBackendName)}
	if runtimeCfg.ContextLen > 0 {
		loadOpts = append(loadOpts, inference.WithContextLen(runtimeCfg.ContextLen))
	}
	loadOpts = append(loadOpts, inference.WithGPULayers(runtimeCfg.GPULayers))
	if runtimeCfg.ParallelSlots > 0 {
		loadOpts = append(loadOpts, inference.WithParallelSlots(runtimeCfg.ParallelSlots))
	}
	if strings.TrimSpace(runtimeCfg.AdapterPath) != "" {
		loadOpts = append(loadOpts, inference.WithAdapterPath(strings.TrimSpace(runtimeCfg.AdapterPath)))
	}
	rocmLoadConfig := rocmCLILoadConfigForKVCache(runtimeCfg.KVCache)
	if err := validateROCmCLILoadConfigBackend("daemon", defaultBackendName, rocmLoadConfig); err != nil {
		return daemon.ServerConfig{}, err
	}
	return daemon.ServerConfig{
		SocketPath:           strings.TrimSpace(runtimeCfg.SocketPath),
		ModelPaths:           cloneStringMap(runtimeCfg.Models),
		ModelRegistryBackend: daemonModelRegistryBackend{Backend: defaultBackendName},
		ModelPackBackend:     daemon.ROCmModelPackInspector{},
		NativeGenerate: daemon.NativeGenerateConfig{
			ModelPaths:       cloneStringMap(runtimeCfg.Models),
			DefaultModelName: strings.TrimSpace(runtimeCfg.DefaultModelName),
			DefaultMaxTokens: runtimeCfg.DefaultMaxTokens,
			LoadOptions:      loadOpts,
			ROCmLoadConfig:   rocmLoadConfig,
		},
	}, nil
}

func firstDaemonNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
