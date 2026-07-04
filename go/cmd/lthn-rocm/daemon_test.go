// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	rocm "dappco.re/go/rocm"
	"dappco.re/go/rocm/daemon"
)

func TestDaemonJSONReportsLocalDaemonContract(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	var stdout, stderr bytes.Buffer

	code := runCommand(context.Background(), []string{
		"daemon",
		"-json",
		"-socket", "/tmp/lthn-test.sock",
		"-model", "main=/models/main",
		"-model", "/models/default",
		"-model-name", "main",
		"-max-tokens", "8",
		"-context", "4096",
		"-gpu-layers", "0",
		"-parallel", "2",
		"-kv-cache", "q8",
		"-adapter", "/adapters/domain",
	}, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("runCommand(daemon -json) = %d, stderr=%s", code, stderr.String())
	}
	var report struct {
		Backend          string              `json:"backend"`
		Command          string              `json:"command"`
		CLIContract      string              `json:"cli_contract"`
		Daemon           string              `json:"daemon"`
		Socket           string              `json:"socket"`
		DefaultModel     string              `json:"default_model"`
		DefaultMaxTokens int                 `json:"default_max_tokens"`
		ModelCount       int                 `json:"model_count"`
		Models           map[string]string   `json:"models"`
		ModelRegistry    modelRegistryReport `json:"model_registry"`
		Routes           []daemon.RouteInfo  `json:"routes"`
		Load             map[string]any      `json:"load"`
		Labels           map[string]string   `json:"labels"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode report: %v\n%s", err, stdout.String())
	}
	if report.Backend != defaultBackendName ||
		report.Command != "daemon" ||
		report.CLIContract != cliContractName ||
		report.Daemon != "lthn" ||
		report.Socket != "/tmp/lthn-test.sock" ||
		report.DefaultModel != "main" ||
		report.DefaultMaxTokens != 8 ||
		report.ModelCount != 2 {
		t.Fatalf("report = %+v, want daemon JSON contract", report)
	}
	if report.Models["main"] != "/models/main" || report.Models["default"] != "/models/default" {
		t.Fatalf("models = %+v, want named and default model entries", report.Models)
	}
	if report.Load["kv_cache"] != "q8" || report.Load["adapter"] != "/adapters/domain" {
		t.Fatalf("load = %+v, want ROCm load config surfaced", report.Load)
	}
	if report.ModelRegistry.Name == "" ||
		report.ModelRegistry.Backend != defaultBackendName ||
		report.ModelRegistry.Labels["engine_profile_reactive"] != "true" ||
		report.ModelRegistry.Labels["engine_algorithm_profile_contract"] != rocm.ROCmAlgorithmProfileRegistryContract ||
		report.ModelRegistry.Labels["algorithm_profile_count"] != strconv.Itoa(len(rocm.DefaultROCmAlgorithmProfiles())) ||
		len(report.ModelRegistry.AlgorithmProfiles) != len(rocm.DefaultROCmAlgorithmProfiles()) {
		t.Fatalf("model_registry = %+v, want reactive ROCm registry snapshot", report.ModelRegistry)
	}
	if !daemonReportHasRoute(report.Routes, "score", "heuristic") ||
		!daemonReportHasRoute(report.Routes, "generate", "stub") ||
		!daemonReportHasRoute(report.Routes, "schedule", "stub") ||
		!daemonReportHasRoute(report.Routes, "rerank", "stub") ||
		!daemonReportHasRoute(report.Routes, "cache_stats", "stub") ||
		!daemonReportHasRoute(report.Routes, "cache_warm", "stub") ||
		!daemonReportHasRoute(report.Routes, "cache_clear", "stub") ||
		!daemonReportHasRoute(report.Routes, "cache_entries", "stub") ||
		!daemonReportHasRoute(report.Routes, "cancel", "stub") ||
		!daemonReportHasRoute(report.Routes, "health", "system") ||
		!daemonReportHasRoute(report.Routes, "models", "stub") ||
		!daemonReportHasRoute(report.Routes, "model_info", "stub") ||
		!daemonReportHasRoute(report.Routes, "capabilities", "stub") ||
		!daemonReportHasRoute(report.Routes, "engine_features", "native_metadata") ||
		!daemonReportHasRoute(report.Routes, "inspect_model_pack", "native_metadata") ||
		!daemonReportHasRoute(report.Routes, "model_profile", "native_metadata") ||
		!daemonReportHasRoute(report.Routes, "model_routes", "native_metadata") ||
		!daemonReportHasRoute(report.Routes, "registry", "native_metadata") ||
		!daemonReportHasRoute(report.Routes, "tokenize", "stub") ||
		!daemonReportHasRoute(report.Routes, "detokenize", "stub") ||
		!daemonReportHasRoute(report.Routes, "chat_template", "stub") ||
		!daemonReportHasRoute(report.Routes, "parse_reasoning", "stub") ||
		!daemonReportHasRoute(report.Routes, "parse_tools", "stub") {
		t.Fatalf("routes = %+v, want daemon route metadata", report.Routes)
	}
	if report.Labels["daemon_contract"] != "json-lines-unix-v1" ||
		report.Labels["reactive_daemon_embed_route"] != "native_model_if_supported" ||
		report.Labels["reactive_daemon_generate_route"] != "accepted" ||
		report.Labels["reactive_daemon_schedule_route"] != "native_model_if_supported" ||
		report.Labels["reactive_daemon_rerank_route"] != "native_model_if_supported" ||
		report.Labels["reactive_daemon_score_route"] != "heuristic" ||
		report.Labels["reactive_daemon_cache_route"] != "native_model_if_supported" ||
		report.Labels["reactive_daemon_cancel_route"] != "native_model_if_supported" ||
		report.Labels["reactive_daemon_health_route"] != "system" ||
		report.Labels["reactive_daemon_models_route"] != "native_model_if_supported" ||
		report.Labels["reactive_daemon_capabilities_route"] != "native_model_if_supported" ||
		report.Labels["reactive_daemon_engine_features_route"] != "native_metadata" ||
		report.Labels["reactive_daemon_model_pack_route"] != "native_metadata" ||
		report.Labels["reactive_daemon_model_profile_route"] != "native_metadata" ||
		report.Labels["reactive_daemon_model_routes_route"] != "native_metadata" ||
		report.Labels["reactive_daemon_registry_route"] != "native_metadata" ||
		report.Labels["reactive_daemon_tokenizer_route"] != "native_model_if_supported" ||
		report.Labels["reactive_daemon_parser_route"] != "native_model_if_supported" ||
		!strings.Contains(report.Labels["daemon_actions"], "generate") {
		t.Fatalf("labels = %+v, want daemon contract labels", report.Labels)
	}
}

func TestDaemonServerConfig_Good_DefaultsReactiveMetadataToNativeRunner(t *testing.T) {
	cfg, err := newDaemonServerConfig(daemonRuntimeConfig{
		SocketPath:       "/tmp/lthn-test.sock",
		Models:           map[string]string{"main": "/models/main"},
		DefaultModelName: "main",
		DefaultMaxTokens: 8,
		ContextLen:       4096,
		GPULayers:        0,
		ParallelSlots:    2,
		KVCache:          "q8",
		AdapterPath:      "/adapters/domain",
	})
	if err != nil {
		t.Fatalf("newDaemonServerConfig() error = %v", err)
	}
	if cfg.EngineFeaturesBackend != nil ||
		cfg.ModelProfileBackend != nil ||
		cfg.ModelRoutesBackend != nil {
		t.Fatalf("metadata backends = features:%T profile:%T routes:%T, want NativeGenerateRunner defaults", cfg.EngineFeaturesBackend, cfg.ModelProfileBackend, cfg.ModelRoutesBackend)
	}
	if _, ok := cfg.ModelRegistryBackend.(daemonModelRegistryBackend); !ok {
		t.Fatalf("ModelRegistryBackend = %T, want static registry snapshot backend", cfg.ModelRegistryBackend)
	}
	if _, ok := cfg.ModelPackBackend.(daemon.ROCmModelPackInspector); !ok {
		t.Fatalf("ModelPackBackend = %T, want metadata inspector", cfg.ModelPackBackend)
	}

	server := daemon.NewServer(cfg)
	runner, ok := server.GenerateBackend.(*daemon.NativeGenerateRunner)
	if !ok {
		t.Fatalf("GenerateBackend = %T, want NativeGenerateRunner", server.GenerateBackend)
	}
	if features, ok := server.EngineFeaturesBackend.(*daemon.NativeGenerateRunner); !ok || features != runner {
		t.Fatalf("EngineFeaturesBackend = %T, want NativeGenerateRunner shared with generate", server.EngineFeaturesBackend)
	}
	if profile, ok := server.ModelProfileBackend.(*daemon.NativeGenerateRunner); !ok || profile != runner {
		t.Fatalf("ModelProfileBackend = %T, want NativeGenerateRunner shared with generate", server.ModelProfileBackend)
	}
	if routes, ok := server.ModelRoutesBackend.(*daemon.NativeGenerateRunner); !ok || routes != runner {
		t.Fatalf("ModelRoutesBackend = %T, want NativeGenerateRunner shared with generate", server.ModelRoutesBackend)
	}
	if _, static := server.ModelRoutesBackend.(daemonModelProfileBackend); static {
		t.Fatalf("ModelRoutesBackend = %T, want live-aware native runner instead of profile-only backend", server.ModelRoutesBackend)
	}
}

func TestDaemonModelProfileBackend_Good_ResolvesEngineFeatures(t *testing.T) {
	modelPath := filepath.Join(t.TempDir(), "gemma")
	writeCLIDraftDetectModelDir(t, modelPath, "gemma4_text")
	backend := daemonModelProfileBackend{
		Backend:          defaultBackendName,
		DefaultModelName: "main",
		ModelPaths:       map[string]string{"main": modelPath},
	}

	result, err := backend.EngineFeatures(context.Background(), daemon.EngineFeaturesRequest{
		Model:  " main ",
		Labels: map[string]string{"tenant": "test"},
	})

	if err != nil {
		t.Fatalf("EngineFeatures() error = %v", err)
	}
	features, ok := result.(rocm.ROCmEngineFeatures)
	if !ok {
		t.Fatalf("EngineFeatures() = %#v, want ROCmEngineFeatures", result)
	}
	if features.Contract == "" ||
		features.Architecture != "gemma4_text" ||
		features.Family != "gemma4" ||
		!features.ChatTemplate ||
		features.Labels["engine_feature_architecture"] != "gemma4_text" {
		t.Fatalf("features = %+v, want resolved reactive Gemma4 engine features", features)
	}
}

func TestDaemonModelProfileBackend_Good_ResolvesConfiguredModelProfile(t *testing.T) {
	modelPath := filepath.Join(t.TempDir(), "gemma")
	writeCLIDraftDetectModelDir(t, modelPath, "gemma4_text")
	backend := daemonModelProfileBackend{
		Backend:          defaultBackendName,
		DefaultModelName: "main",
		ModelPaths:       map[string]string{"main": modelPath},
	}

	result, err := backend.ModelProfile(context.Background(), daemon.ModelProfileRequest{
		Model:  " main ",
		Labels: map[string]string{"tenant": "test"},
	})

	if err != nil {
		t.Fatalf("ModelProfile() error = %v", err)
	}
	profile, ok := result.(rocm.ROCmModelProfile)
	if !ok {
		t.Fatalf("ModelProfile() = %#v, want ROCmModelProfile", result)
	}
	if !profile.Matched() ||
		profile.Registry == "" ||
		profile.Architecture != "gemma4_text" ||
		profile.Model.Path != modelPath ||
		profile.Model.Labels["tenant"] != "test" ||
		profile.Labels["engine_profile_reactive"] != "true" {
		t.Fatalf("profile = %+v, want resolved reactive Gemma4 profile", profile)
	}
}

func TestDaemonModelProfileBackend_Good_ResolvesModelRoutes(t *testing.T) {
	modelPath := filepath.Join(t.TempDir(), "gemma")
	writeCLIDraftDetectModelDir(t, modelPath, "gemma4_text")
	backend := daemonModelProfileBackend{
		Backend:          defaultBackendName,
		DefaultModelName: "main",
		ModelPaths:       map[string]string{"main": modelPath},
	}

	result, err := backend.ModelRoutes(context.Background(), daemon.ModelRoutesRequest{
		Model:  " main ",
		Labels: map[string]string{"tenant": "test"},
	})

	if err != nil {
		t.Fatalf("ModelRoutes() error = %v", err)
	}
	plan, ok := result.(rocm.ROCmModelRoutePlan)
	if !ok {
		t.Fatalf("ModelRoutes() = %#v, want ROCmModelRoutePlan", result)
	}
	if !plan.Matched() ||
		plan.Contract != rocm.ROCmModelRoutePlanContract ||
		plan.Architecture != "gemma4_text" ||
		plan.Family != "gemma4" ||
		!plan.FeatureRoute.Matched() ||
		!plan.TokenizerRoute.Matched() ||
		plan.LoadStatus.Status == "" ||
		plan.Labels["engine_route_plan_contract"] != rocm.ROCmModelRoutePlanContract ||
		plan.Labels["engine_route_plan_feature"] != "true" {
		t.Fatalf("plan = %+v, want resolved reactive Gemma4 route plan", plan)
	}
}

func TestDaemonModelProfileBackend_Bad_RequiresKnownModelOrPath(t *testing.T) {
	backend := daemonModelProfileBackend{DefaultModelName: "missing", ModelPaths: map[string]string{}}

	_, err := backend.ModelProfile(context.Background(), daemon.ModelProfileRequest{})

	if err == nil || !strings.Contains(err.Error(), "model profile path or known model") {
		t.Fatalf("ModelProfile(empty) = %v, want required path/model error", err)
	}
}

func daemonReportHasRoute(routes []daemon.RouteInfo, action, status string) bool {
	for _, route := range routes {
		if route.Action == action && route.Status == status {
			return true
		}
	}
	return false
}

func TestDaemonJSONLoadsConfigAndEnvFallbacks(t *testing.T) {
	t.Setenv("LTHN_MODEL_PATH", "/models/env-default")
	t.Setenv("LTHN_SCORE_MODEL_PATH", "/models/env-score")
	t.Setenv("VIOLET_GENERATE_MODEL_PATH", "/models/env-generate")
	path := filepath.Join(t.TempDir(), "lthn.toml")
	if err := os.WriteFile(path, []byte(`
socket = "/tmp/lthn-config.sock" # trailing comments are stripped
model = "/models/file-default"
default_model_name = "research"
max_tokens = 16
context = 8192
gpu_layers = 0
parallel = 3
kv_cache = "q8"
adapter = "/adapters/file"

[models]
generate = "/models/file-generate"
research = 'qwen#literal'
`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	var stdout, stderr bytes.Buffer

	code := runCommand(context.Background(), []string{"daemon", "-json", "-config", path}, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("runCommand(daemon config) = %d, stderr=%s", code, stderr.String())
	}
	var report struct {
		Socket           string            `json:"socket"`
		DefaultModel     string            `json:"default_model"`
		DefaultMaxTokens int               `json:"default_max_tokens"`
		ModelCount       int               `json:"model_count"`
		Models           map[string]string `json:"models"`
		Load             map[string]any    `json:"load"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode report: %v\n%s", err, stdout.String())
	}
	if report.Socket != "/tmp/lthn-config.sock" ||
		report.DefaultModel != "research" ||
		report.DefaultMaxTokens != 16 ||
		report.ModelCount != 4 {
		t.Fatalf("report = %+v, want config-driven daemon settings", report)
	}
	if report.Models["default"] != "/models/file-default" ||
		report.Models["score"] != "/models/env-score" ||
		report.Models["generate"] != "/models/file-generate" ||
		report.Models["research"] != "qwen#literal" {
		t.Fatalf("models = %+v, want file to beat env and env to fill gaps", report.Models)
	}
	if report.Load["context"] != float64(8192) ||
		report.Load["gpu_layers"] != float64(0) ||
		report.Load["parallel"] != float64(3) ||
		report.Load["kv_cache"] != "q8" ||
		report.Load["adapter"] != "/adapters/file" {
		t.Fatalf("load = %+v, want config load settings", report.Load)
	}
}

func TestDaemonJSONFlagsOverrideConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lthn.toml")
	if err := os.WriteFile(path, []byte(`
socket = "/tmp/lthn-config.sock"
model = "/models/file-default"
default_model_name = "default"
max_tokens = 16
context = 8192
parallel = 3

[models]
research = "/models/research"
`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	var stdout, stderr bytes.Buffer

	code := runCommand(context.Background(), []string{
		"daemon",
		"-json",
		"-config", path,
		"-socket", "/tmp/lthn-flag.sock",
		"-model", "research=/models/flag-research",
		"-model-name", "research",
		"-max-tokens", "4",
		"-context", "2048",
		"-parallel", "1",
	}, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("runCommand(daemon flags override) = %d, stderr=%s", code, stderr.String())
	}
	var report struct {
		Socket           string            `json:"socket"`
		DefaultModel     string            `json:"default_model"`
		DefaultMaxTokens int               `json:"default_max_tokens"`
		Models           map[string]string `json:"models"`
		Load             map[string]any    `json:"load"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode report: %v\n%s", err, stdout.String())
	}
	if report.Socket != "/tmp/lthn-flag.sock" ||
		report.DefaultModel != "research" ||
		report.DefaultMaxTokens != 4 ||
		report.Models["research"] != "/models/flag-research" ||
		report.Load["context"] != float64(2048) ||
		report.Load["parallel"] != float64(1) {
		t.Fatalf("report = %+v, want explicit daemon flags to override config", report)
	}
}

func TestDaemonConfigParserRejectsInvalidSyntax(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lthn.toml")
	if err := os.WriteFile(path, []byte("socket /tmp/lthn.sock\n"), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	_, err := loadDaemonRuntimeConfig(path)

	if err == nil || !strings.Contains(err.Error(), "expected key = value") {
		t.Fatalf("loadDaemonRuntimeConfig(invalid) = %v, want syntax error", err)
	}
}

func TestDaemonFlagsRejectNegativeDefaults(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	var stdout, stderr bytes.Buffer

	code := runCommand(context.Background(), []string{"daemon", "-json", "-max-tokens", "-1"}, &stdout, &stderr)

	if code != 2 {
		t.Fatalf("runCommand(daemon bad max tokens) = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "-max-tokens must be >= 0") {
		t.Fatalf("stderr = %q, want max-token validation", stderr.String())
	}
}
