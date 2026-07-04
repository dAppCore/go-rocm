// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"dappco.re/go/inference"
)

func TestROCmBackendImplementsDiscoveryPlanner(t *testing.T) {
	var _ inference.MachineDiscoverer = (*rocmBackend)(nil)
	var _ inference.TuningPlanner = (*rocmBackend)(nil)
}

func TestPlanLocalTuningReportsReactiveCandidate(t *testing.T) {
	plan, err := PlanLocalTuning(context.Background(), inference.TuningPlanRequest{
		Runtime: inference.RuntimeIdentity{Backend: "rocm"},
		Model: inference.ModelIdentity{
			Path:          "/models/qwen",
			Architecture:  "qwen3_6_moe",
			ContextLength: 8192,
			NumLayers:     1,
			HiddenSize:    1,
			VocabSize:     1,
			QuantBits:     4,
		},
		Workloads: []inference.TuningWorkload{
			inference.TuningWorkloadChat,
			inference.TuningWorkloadAgentState,
		},
		Labels: map[string]string{"caller": "unit-test"},
	})
	if err != nil {
		t.Fatalf("PlanLocalTuning returned error: %v", err)
	}
	if !slices.Equal(plan.Workloads, []inference.TuningWorkload{inference.TuningWorkloadChat, inference.TuningWorkloadAgentState}) {
		t.Fatalf("workloads = %+v, want requested workloads", plan.Workloads)
	}
	if len(plan.Candidates) != 2 {
		t.Fatalf("candidates = %d, want 2: %+v", len(plan.Candidates), plan.Candidates)
	}
	chat := plan.Candidates[0]
	if chat.Workload != inference.TuningWorkloadChat ||
		chat.CacheMode != "k-q8-v-q4" ||
		chat.Runtime.CacheMode != "k-q8-v-q4" ||
		chat.Model.Architecture != "qwen3_6_moe" ||
		chat.ContextLength != 8192 ||
		chat.BatchSize != 1 ||
		chat.ParallelSlots != 1 ||
		chat.PromptCache {
		t.Fatalf("chat candidate = %+v, want ROCm native-compatible chat candidate", chat)
	}
	if chat.Labels["engine_profile"] != "qwen" ||
		chat.Labels["engine_registry"] != DefaultROCmModelRegistryName() ||
		chat.Labels["engine_load_status"] != string(ROCmModelLoadStagedNative) ||
		chat.Labels["candidate_cache_mode_bound"] != "true" ||
		chat.Labels["reactive_registry_planning"] != "true" ||
		chat.Labels["production_requires_env_gate"] != "false" ||
		chat.Labels["production_requires_cli_flag"] != "false" ||
		chat.Labels["caller"] != "unit-test" {
		t.Fatalf("chat labels = %+v, want reactive registry and caller metadata", chat.Labels)
	}
	agent := plan.Candidates[1]
	if agent.Workload != inference.TuningWorkloadAgentState ||
		!agent.PromptCache ||
		agent.CachePolicy != "stateful" ||
		agent.Labels["state_restore"] != "candidate" ||
		agent.Labels["reactive_state_continuity"] != "candidate" {
		t.Fatalf("agent candidate = %+v, want stateful prompt-cache candidate", agent)
	}
	if plan.Recommended[inference.TuningWorkloadAgentState] != agent.ID {
		t.Fatalf("recommended agent_state = %q, want %q", plan.Recommended[inference.TuningWorkloadAgentState], agent.ID)
	}
	if len(plan.Warnings) != 0 {
		t.Fatalf("warnings = %+v, want none for registry-backed qwen", plan.Warnings)
	}
}

func TestDiscoverMachineIncludesCandidatesFromModelDir(t *testing.T) {
	model := t.TempDir()
	writeTuningTestModelDir(t, model, "qwen3_6_moe")

	report, err := (&rocmBackend{}).DiscoverMachine(context.Background(), inference.MachineDiscoveryRequest{
		ModelDirs:         []string{model},
		IncludeModels:     true,
		IncludeCandidates: true,
		MaxModels:         1,
		Workloads:         []inference.TuningWorkload{inference.TuningWorkloadChat},
		Labels:            map[string]string{"caller": "discover-test"},
	})
	if err != nil {
		t.Fatalf("DiscoverMachine returned error: %v", err)
	}
	if report.Runtime.Backend != "rocm" {
		t.Fatalf("runtime = %+v, want ROCm backend", report.Runtime)
	}
	if !slices.Equal(report.Workloads, []inference.TuningWorkload{inference.TuningWorkloadChat}) {
		t.Fatalf("workloads = %+v, want chat-only workload", report.Workloads)
	}
	if len(report.Models) != 1 || len(report.Candidates) != 1 {
		t.Fatalf("models=%d candidates=%d, want one discovered model and one candidate", len(report.Models), len(report.Candidates))
	}
	candidate := report.Candidates[0]
	if candidate.Workload != inference.TuningWorkloadChat ||
		!strings.HasPrefix(candidate.Model.Path, model) ||
		candidate.Model.Architecture != "qwen3_6_moe" ||
		candidate.Runtime.Backend != "rocm" ||
		candidate.Runtime.CacheMode != "k-q8-v-q4" ||
		candidate.CacheMode != "k-q8-v-q4" {
		t.Fatalf("candidate = %+v, want discovered ROCm tuning candidate", candidate)
	}
	if candidate.Labels["engine_profile"] != "qwen" ||
		candidate.Labels["candidate_contract"] != "go-inference.tuning-candidate" ||
		candidate.Labels["candidate_source"] != "go-rocm PlanTuning" ||
		candidate.Labels["caller"] != "discover-test" {
		t.Fatalf("candidate labels = %+v, want package planner labels", candidate.Labels)
	}
	if report.Labels["backend"] != "rocm" ||
		report.Labels["reactive_registry_planning"] != "true" ||
		report.Labels["production_requires_env_gate"] != "false" ||
		report.Labels["production_requires_cli_flag"] != "false" ||
		report.Labels["caller"] != "discover-test" {
		t.Fatalf("report labels = %+v, want discovery contract labels", report.Labels)
	}
}

func TestTuningCandidateLoadConfigAppliesCandidate(t *testing.T) {
	candidate := inference.TuningCandidate{
		Runtime: inference.RuntimeIdentity{
			Backend:   "rocm",
			CacheMode: "q8",
		},
		ContextLength: 8192,
		ParallelSlots: 2,
		CacheMode:     "kq8vq4",
		Adapter: inference.AdapterIdentity{
			Path: "/models/adapter",
		},
	}

	rocmCfg, opts := TuningCandidateLoadConfig(candidate)
	if rocmCfg.CacheMode != "k-q8-v-q4" || rocmCfg.DeviceKVMode != "k-q8-v-q4" {
		t.Fatalf("ROCm config = %+v, want normalized candidate cache mode", rocmCfg)
	}
	loadCfg := inference.ApplyLoadOpts(opts)
	if loadCfg.Backend != "rocm" ||
		loadCfg.ContextLen != 8192 ||
		loadCfg.ParallelSlots != 2 ||
		loadCfg.AdapterPath != "/models/adapter" {
		t.Fatalf("load config = %+v, want candidate-derived neutral load options", loadCfg)
	}
}

func TestTuningCandidateLoadConfigPreservesUnboundCacheMode(t *testing.T) {
	rocmCfg := TuningCandidateROCmLoadConfig(inference.TuningCandidate{
		Runtime: inference.RuntimeIdentity{CacheMode: "paged"},
	})
	if rocmCfg.CacheMode != "paged" || rocmCfg.DeviceKVMode != "paged" {
		t.Fatalf("ROCm config = %+v, want planned mode preserved for native validation", rocmCfg)
	}
}

func TestTuningCandidateLoadOptionsAllowCallerOverride(t *testing.T) {
	candidate := inference.TuningCandidate{
		Runtime:       inference.RuntimeIdentity{Backend: "rocm"},
		ContextLength: 8192,
		ParallelSlots: 2,
	}
	_, candidateOpts := TuningCandidateLoadConfig(candidate)
	opts := append(candidateOpts, inference.WithContextLen(4096), inference.WithParallelSlots(1))
	loadCfg := inference.ApplyLoadOpts(opts)
	if loadCfg.ContextLen != 4096 || loadCfg.ParallelSlots != 1 {
		t.Fatalf("load config = %+v, want later caller options to override candidate options", loadCfg)
	}
}

func TestSelectTuningResultAndLabels(t *testing.T) {
	results := []inference.TuningResult{
		{
			Candidate: inference.TuningCandidate{ID: "slow", Workload: inference.TuningWorkloadChat},
			Score:     inference.TuningScore{Score: 2},
		},
		{
			Candidate: inference.TuningCandidate{ID: "failed", Workload: inference.TuningWorkloadChat},
			Error:     "candidate failed",
		},
		{
			Candidate: inference.TuningCandidate{ID: "fast", Workload: inference.TuningWorkloadChat},
			Measurements: inference.TuningMeasurements{
				DecodeTokensPerSec:      42.5,
				FirstTokenMilliseconds:  12,
				PeakMemoryBytes:         1024,
				CorrectnessSmokeResult:  "pass",
				CorrectnessSmokeChecks:  3,
				KVRestoreMilliseconds:   4,
				StateBundleMilliseconds: 5,
				LoadMilliseconds:        6,
				PromptCacheHitRate:      0.75,
				GeneratedTokens:         16,
				PromptTokens:            8,
				PrefillTokensPerSec:     7,
				TotalMilliseconds:       20,
				ActiveMemoryBytes:       512,
			},
			Score: inference.TuningScore{Score: 5},
		},
	}
	selected, ok := SelectTuningResult(results)
	if !ok || selected.Candidate.ID != "fast" {
		t.Fatalf("SelectTuningResult = %+v ok=%t, want fast candidate", selected, ok)
	}
	labels := TuningSelectionLabels(results, selected)
	if labels["source"] != "go-rocm tune-run" ||
		labels["selection_policy"] != "highest_successful_score" ||
		labels["selected_candidate_id"] != "fast" ||
		labels["successful_candidates"] != "2" ||
		labels["failed_candidates"] != "1" ||
		labels["runner_up_candidate_id"] != "slow" ||
		labels["selection_score_delta"] != "3.000000" ||
		labels["selected_decode_tokens_per_sec"] != "42.500000" ||
		labels["selected_peak_memory_bytes"] != "1024" ||
		labels["selected_correctness_smoke_result"] != "pass" ||
		labels["selected_correctness_smoke_checks"] != "3" {
		t.Fatalf("selection labels = %+v, want measured sweep labels", labels)
	}
}

func TestBuildTuningProfileFillsPlanFallbacks(t *testing.T) {
	created := time.Unix(1234, 0)
	plan := inference.TuningPlan{
		Runtime: inference.RuntimeIdentity{Backend: "rocm", Device: "gfx1100", CacheMode: "q8"},
		Model: inference.ModelIdentity{
			Path:          "/models/qwen",
			Architecture:  "qwen3_6_moe",
			ContextLength: 8192,
		},
		Adapter: inference.AdapterIdentity{Path: "/models/adapter", Rank: 8},
	}
	result := inference.TuningResult{
		Candidate: inference.TuningCandidate{
			ID: "chat-kv",
		},
		Measurements: inference.TuningMeasurements{DecodeTokensPerSec: 12},
		Score:        inference.TuningScore{Score: 12},
	}
	profile := BuildTuningProfile(plan, "/fallback", "machine-a", inference.TuningWorkloadChat, result, map[string]string{"caller": "test"}, created)
	if profile.Key.MachineHash != "machine-a" ||
		profile.Key.Workload != inference.TuningWorkloadChat ||
		profile.Key.Runtime.Backend != "rocm" ||
		profile.Key.Model.Path != "/models/qwen" ||
		profile.Key.Adapter.Path != "/models/adapter" ||
		profile.Candidate.Model.Path != "/models/qwen" ||
		profile.Candidate.Runtime.CacheMode != "q8" ||
		profile.Candidate.Adapter.Rank != 8 ||
		profile.Candidate.Workload != inference.TuningWorkloadChat ||
		profile.Score.Workload != inference.TuningWorkloadChat ||
		profile.CreatedAtUnix != 1234 ||
		profile.Labels["source"] != "go-rocm tune-run" ||
		profile.Labels["caller"] != "test" {
		t.Fatalf("profile = %+v, want plan-backed durable profile", profile)
	}
}

func TestTuningProfilePathAndWrite(t *testing.T) {
	dir := t.TempDir()
	profile := inference.TuningProfile{
		Key: inference.TuningProfileKey{
			MachineHash: "sha256:abcdef0123456789",
			Model:       inference.ModelIdentity{Path: "/models/Model Name!"},
			Workload:    inference.TuningWorkloadAgentState,
		},
		Candidate:     inference.TuningCandidate{ID: "candidate/kq8vq4"},
		CreatedAtUnix: 123,
	}
	path := TuningProfilePath(dir, profile)
	if filepath.Base(path) != "agent-state-abcdef012345-model-name-candidate-kq8vq4.json" {
		t.Fatalf("profile path = %q, want sanitized conventional filename", path)
	}
	if err := WriteTuningProfile(path, profile); err != nil {
		t.Fatalf("WriteTuningProfile returned error: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("written profile stat failed: %v", err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("profile permissions = %v, want 0600", info.Mode().Perm())
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var roundtrip inference.TuningProfile
	if err := json.Unmarshal(data, &roundtrip); err != nil {
		t.Fatalf("profile JSON did not unmarshal: %v\n%s", err, string(data))
	}
	if roundtrip.Candidate.ID != profile.Candidate.ID || roundtrip.CreatedAtUnix != profile.CreatedAtUnix {
		t.Fatalf("roundtrip profile = %+v, want %+v", roundtrip, profile)
	}
}

func TestTuningProfileIdentityOverlaysCandidate(t *testing.T) {
	profile := inference.TuningProfile{
		Key: inference.TuningProfileKey{
			Runtime: inference.RuntimeIdentity{Backend: "rocm", Device: "old", CacheMode: "fp16"},
			Model:   inference.ModelIdentity{Path: "/old", Architecture: "old", QuantBits: 4},
			Adapter: inference.AdapterIdentity{Path: "/old-adapter", Rank: 4},
		},
		Candidate: inference.TuningCandidate{
			Runtime: inference.RuntimeIdentity{Device: "gfx1100", CacheMode: "k-q8-v-q4", Labels: map[string]string{"candidate": "runtime"}},
			Model:   inference.ModelIdentity{Path: "/new", Architecture: "qwen3_6_moe", ContextLength: 8192},
			Adapter: inference.AdapterIdentity{Path: "/new-adapter", Rank: 8, TargetKeys: []string{"q_proj"}, Labels: map[string]string{"candidate": "adapter"}},
		},
	}
	model := ModelIdentityFromTuningProfile(profile)
	runtime := RuntimeIdentityFromTuningProfile(profile)
	adapter := AdapterIdentityFromTuningProfile(profile)
	if model.Path != "/new" || model.Architecture != "qwen3_6_moe" || model.QuantBits != 4 || model.ContextLength != 8192 {
		t.Fatalf("model overlay = %+v, want candidate fields over key", model)
	}
	if runtime.Backend != "rocm" || runtime.Device != "gfx1100" || runtime.CacheMode != "k-q8-v-q4" || runtime.Labels["candidate"] != "runtime" {
		t.Fatalf("runtime overlay = %+v, want candidate runtime over key", runtime)
	}
	if adapter.Path != "/new-adapter" || adapter.Rank != 8 || !slices.Equal(adapter.TargetKeys, []string{"q_proj"}) || adapter.Labels["candidate"] != "adapter" {
		t.Fatalf("adapter overlay = %+v, want candidate adapter over key", adapter)
	}
}

func writeTuningTestModelDir(t *testing.T, dir, modelType string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	config := `{"model_type":"` + modelType + `","hidden_size":1,"num_hidden_layers":1,"vocab_size":1,"max_position_embeddings":8192,"quantization_config":{"bits":4,"group_size":64}}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	writeTuningTestSafetensors(t, filepath.Join(dir, "model.safetensors"))
}

func writeTuningTestSafetensors(t *testing.T, path string) {
	t.Helper()
	header := `{"model.embed_tokens.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, uint64(len(header))); err != nil {
		t.Fatal(err)
	}
	buf.WriteString(header)
	buf.Write([]byte{0, 0, 0, 0})
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}
