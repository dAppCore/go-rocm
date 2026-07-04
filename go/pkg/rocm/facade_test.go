// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"dappco.re/go/inference"
	root "dappco.re/go/rocm"
)

func TestFacade_Good_RuntimeLanesExposeReleaseArtifacts(t *testing.T) {
	lanes := DefaultRuntimeLanes()
	if len(lanes) != 4 {
		t.Fatalf("DefaultRuntimeLanes() = %d lanes, want amd/cuda/cpu-x86/cpu-aarch64", len(lanes))
	}
	amd, ok := RuntimeLaneForArtifact("/tmp/" + RuntimeLaneArtifactAMD)
	if !ok {
		t.Fatalf("RuntimeLaneForArtifact(%q) not found", RuntimeLaneArtifactAMD)
	}
	if amd.RuntimeDispatchStatus != RuntimeDispatchStatusActive || amd.RuntimeLane != RuntimeLaneAMD {
		t.Fatalf("amd lane = %+v, want active ROCm lane", amd)
	}
	cpu := RuntimeLanesForBackend("cpu")
	if len(cpu) != 2 {
		t.Fatalf("RuntimeLanesForBackend(cpu) = %d, want x86 and aarch64", len(cpu))
	}
	lanes[0].Labels["mutated"] = "true"
	next := DefaultRuntimeLanes()
	if next[0].Labels["mutated"] != "" {
		t.Fatalf("DefaultRuntimeLanes leaked caller label mutation")
	}
}

func TestFacade_Good_ProductionFastLaneIsDefaultPath(t *testing.T) {
	fast := DefaultProductionFastLane()
	if fast.Backend != "rocm" || fast.Library != "go-rocm" || fast.ReferenceBackend != "go-mlx" {
		t.Fatalf("DefaultProductionFastLane() = %+v, want go-rocm/go-mlx contract", fast)
	}
	if !fast.EnabledByDefault || fast.RequiresEnvGate || fast.RequiresCLIFlag {
		t.Fatalf("DefaultProductionFastLane() = %+v, want enabled production default without gates", fast)
	}
	if fast.MTPDefaultDraftTokens != ProductionMTPDefaultDraftTokens {
		t.Fatalf("MTPDefaultDraftTokens = %d, want %d", fast.MTPDefaultDraftTokens, ProductionMTPDefaultDraftTokens)
	}
}

func TestFacade_Good_ReactiveModelRoutePlan(t *testing.T) {
	identity := inference.ModelIdentity{
		Path:         "mlx-community/gemma-4-e2b-it-6bit",
		Architecture: "gemma4_text",
		QuantBits:    6,
		QuantGroup:   64,
		Labels: map[string]string{
			"gemma4_size":       "E2B",
			"gemma4_quant_mode": "6bit",
		},
	}
	profile, ok := ResolveROCmModelProfile(identity.Path, identity)
	if !ok || !profile.Matched() {
		t.Fatalf("ResolveROCmModelProfile() = %+v ok=%v, want Gemma4 profile", profile, ok)
	}
	features, ok := ROCmEngineFeaturesForIdentity(identity.Path, identity)
	if !ok || features.Architecture == "" || !features.ModelContextWindow {
		t.Fatalf("ROCmEngineFeaturesForIdentity() = %+v ok=%v, want reactive Gemma4 features", features, ok)
	}
	plan, ok := ROCmModelRoutePlanForIdentity(identity.Path, identity)
	if !ok || !plan.Matched() || plan.Contract != ROCmModelRoutePlanContract {
		t.Fatalf("ROCmModelRoutePlanForIdentity() = %+v ok=%v, want route plan", plan, ok)
	}
	if plan.EngineFeatures.Contract == "" || plan.LoadStatus.Status == "" {
		t.Fatalf("route plan = %+v, want engine features and load status", plan)
	}
	retained, ok := ROCmRetainedStateForIdentity(identity.Path, identity)
	if !ok || !retained.RuntimeOwnedDecodeReady() {
		t.Fatalf("ROCmRetainedStateForIdentity() = %+v ok=%v, want runtime-owned retained decode", retained, ok)
	}
	if retained.DefaultDeviceKVMode != "k-q8-v-q4" ||
		!retained.SleepState ||
		!retained.WakeState ||
		!retained.ModelContextWindow ||
		retained.Labels["engine_state_context_prompt_replay_refused"] != "true" {
		t.Fatalf("retained state = %+v, want Gemma4 state-context contract labels", retained)
	}
	retained.Labels["mutated"] = "true"
	nextRetained, ok := ROCmRetainedStateForIdentity(identity.Path, identity)
	if !ok || nextRetained.Labels["mutated"] != "" {
		t.Fatalf("ROCmRetainedStateForIdentity leaked caller label mutation: %+v", nextRetained.Labels)
	}
	tokenLoop, ok := ROCmTokenLoopForIdentity(identity.Path, identity)
	if !ok ||
		!tokenLoop.IncrementalDecodeReady() ||
		!tokenLoop.StepWithID ||
		!tokenLoop.PerLayerInputs ||
		tokenLoop.SessionState != "StateSession" ||
		tokenLoop.FastPath != "retained-state-session" ||
		tokenLoop.Labels["engine_token_loop_contract"] == "" {
		t.Fatalf("ROCmTokenLoopForIdentity() = %+v ok=%v, want retained token-loop session contract", tokenLoop, ok)
	}
}

func TestFacade_Good_SourceBoundaryDoesNotImportReferenceRepoOrCgo(t *testing.T) {
	dir := "."
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") {
			continue
		}
		if strings.HasSuffix(entry.Name(), "_test.go") {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		source := string(raw)
		forbidden := []string{
			`import "C"`,
			`dappco.re/go/mlx`,
			`github.com/dappcore/go-mlx`,
			`external/go-mlx`,
			`external/`,
		}
		for _, needle := range forbidden {
			if strings.Contains(source, needle) {
				t.Fatalf("%s contains forbidden boundary reference %q", path, needle)
			}
		}
	}
}

func TestFacade_Good_TypeAliasesRootContracts(t *testing.T) {
	var _ root.ROCmModelRoutePlan = ROCmModelRoutePlan{}
	var _ root.ProductionFastLane = ProductionFastLane{}
	var _ root.RuntimeLaneStatus = RuntimeLaneStatus{}
	var _ root.ROCmRetainedStateStatus = ROCmRetainedStateStatus{}
	var _ root.ROCmTokenLoopStatus = ROCmTokenLoopStatus{}
}
