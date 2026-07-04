// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"path/filepath"
	"testing"
)

func TestDefaultRuntimeLanesReportsReleaseArtifacts(t *testing.T) {
	lanes := DefaultRuntimeLanes()
	if len(lanes) != 4 {
		t.Fatalf("DefaultRuntimeLanes len=%d, want 4: %+v", len(lanes), lanes)
	}
	byName := map[string]RuntimeLaneStatus{}
	for _, lane := range lanes {
		byName[lane.Name] = lane
	}
	tests := []struct {
		name    string
		backend string
		lane    string
		status  string
		sidecar string
		active  bool
		pending bool
	}{
		{name: RuntimeLaneArtifactAMD, backend: "rocm", lane: RuntimeLaneAMD, status: RuntimeDispatchStatusActive, sidecar: RuntimeLaneSidecarAMD, active: true},
		{name: RuntimeLaneArtifactCUDA, backend: "cuda", lane: RuntimeLaneCUDA, status: RuntimeDispatchStatusCompileReadyPending, sidecar: RuntimeLaneSidecarCUDA, pending: true},
		{name: RuntimeLaneArtifactCPUX86, backend: "cpu", lane: RuntimeLaneCPUX86, status: RuntimeDispatchStatusCompileReadyPending, sidecar: RuntimeLaneSidecarCPUX86, pending: true},
		{name: RuntimeLaneArtifactCPUAArch64, backend: "cpu", lane: RuntimeLaneCPUAArch64, status: RuntimeDispatchStatusCompileReadyPending, sidecar: RuntimeLaneSidecarCPUAArch64, pending: true},
	}
	for _, tt := range tests {
		lane, ok := byName[tt.name]
		if !ok {
			t.Fatalf("lane %s missing from %+v", tt.name, lanes)
		}
		if lane.Backend != tt.backend ||
			lane.RuntimeLane != tt.lane ||
			lane.RuntimeDispatchStatus != tt.status ||
			!runtimeLaneStringSliceContains(lane.Sidecars, tt.sidecar) ||
			lane.Active() != tt.active ||
			lane.Pending() != tt.pending ||
			lane.Labels["runtime_lane"] != tt.lane ||
			lane.Labels["runtime_dispatch_status"] != tt.status {
			t.Fatalf("lane %s = %+v, want %s/%s/%s/%s active=%t pending=%t", tt.name, lane, tt.backend, tt.lane, tt.status, tt.sidecar, tt.active, tt.pending)
		}
	}
}

func TestRuntimeLaneForArtifactAcceptsPathAndClones(t *testing.T) {
	lane, ok := RuntimeLaneForArtifact(filepath.Join("/tmp", RuntimeLaneArtifactCUDA))
	if !ok {
		t.Fatalf("RuntimeLaneForArtifact did not resolve cuda artifact path")
	}
	lane.Labels["runtime_lane"] = "mutated"
	lane.Sidecars[0] = "mutated.o"

	again, ok := RuntimeLaneForArtifact(RuntimeLaneArtifactCUDA)
	if !ok {
		t.Fatalf("RuntimeLaneForArtifact did not resolve cuda artifact")
	}
	if again.Labels["runtime_lane"] != RuntimeLaneCUDA || again.Sidecars[0] != RuntimeLaneSidecarCUDA {
		t.Fatalf("runtime lane lookup returned mutable global state: %+v", again)
	}
}

func TestRuntimeLanesForBackendReportsCPUVariants(t *testing.T) {
	lanes := RuntimeLanesForBackend("cpu")
	if len(lanes) != 2 {
		t.Fatalf("RuntimeLanesForBackend(cpu) len=%d, want 2: %+v", len(lanes), lanes)
	}
	seen := map[string]bool{}
	for _, lane := range lanes {
		seen[lane.RuntimeLane] = true
	}
	if !seen[RuntimeLaneCPUX86] || !seen[RuntimeLaneCPUAArch64] {
		t.Fatalf("cpu runtime lanes = %+v, want x86 and aarch64", lanes)
	}
}

func TestCurrentProcessRuntimeLaneFallsBackToDevROCm(t *testing.T) {
	lane := CurrentProcessRuntimeLane("lthn-rocm-dev")
	if lane.Backend != "rocm" ||
		lane.RuntimeLane != RuntimeLaneAMD ||
		lane.RuntimeDispatchStatus != RuntimeDispatchStatusDevROCm ||
		lane.Labels["production_artifact"] != "false" ||
		lane.Labels["module"] != "dappco.re/go/rocm" {
		t.Fatalf("CurrentProcessRuntimeLane(dev) = %+v, want dev ROCm lane", lane)
	}
}

func runtimeLaneStringSliceContains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
