//go:build linux && amd64

package rocm

import (
	"context"
	"encoding/binary"
	"slices"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestRegisterRocm_BackendRegistration_Good(t *testing.T) {
	backend, ok := inference.Get("rocm")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "rocm", backend.Name())
	core.AssertEqual(t, ROCmAvailable(), backend.Available())
}

func TestRegisterRocm_RuntimeLaneBackendRegistration_Good(t *testing.T) {
	names := inference.List()
	if !slices.Contains(names, "cuda") || !slices.Contains(names, "cpu") || !slices.Contains(names, "rocm") {
		t.Fatalf("registered backends = %v, want rocm, cuda, and cpu", names)
	}

	cudaBackend, ok := inference.Get("cuda")
	if !ok || cudaBackend.Name() != "cuda" {
		t.Fatalf("cuda backend = %+v ok=%v, want registered runtime lane backend", cudaBackend, ok)
	}
	cudaReport, ok := inference.CapabilitiesOf(cudaBackend)
	if !ok {
		t.Fatal("cuda backend did not expose capabilities")
	}
	if cudaReport.Available ||
		cudaReport.Runtime.Backend != "cuda" ||
		cudaReport.Labels["runtime_lane"] != RuntimeLaneCUDA ||
		cudaReport.Labels["sidecars"] != RuntimeLaneSidecarCUDA ||
		cudaReport.Labels["runtime_dispatch_status"] != RuntimeDispatchStatusCompileReadyPending ||
		!cudaReport.Supports(inference.CapabilityRuntimeDiscovery) ||
		!cudaReport.Supports(inference.CapabilityModelFit) ||
		!cudaReport.Supports(inference.CapabilityMemoryPlanning) ||
		!cudaReport.Supports(inference.CapabilityKVCachePlanning) {
		t.Fatalf("cuda capability report = %+v, want discoverable pending lane", cudaReport)
	}
	if cap, ok := cudaReport.Capability(inference.CapabilityModelLoad); !ok ||
		cap.Status != inference.CapabilityStatusPlanned ||
		cap.Labels["runtime_lane"] != RuntimeLaneCUDA {
		t.Fatalf("cuda model-load capability = %+v ok=%v, want planned lane metadata", cap, ok)
	}
	if _, err := cudaBackend.LoadModel(t.TempDir()); err == nil ||
		!strings.Contains(err.Error(), "runtime dispatch is pending") ||
		!strings.Contains(err.Error(), RuntimeLaneSidecarCUDA) {
		t.Fatalf("cuda LoadModel error = %v, want pending dispatch sidecar error", err)
	}
	cudaInspector, ok := cudaBackend.(inference.ModelPackInspector)
	if !ok {
		t.Fatal("cuda backend does not expose model-pack inspection")
	}
	inspection, err := cudaInspector.InspectModelPack(context.Background(), runtimeLaneBackendTestSafetensorsPack(t))
	if err != nil {
		t.Fatalf("cuda InspectModelPack: %v", err)
	}
	if inspection.Labels["backend"] != "cuda" ||
		inspection.Labels["active_backend"] != "rocm" ||
		inspection.Labels["runtime_lane"] != RuntimeLaneCUDA ||
		inspection.Labels["runtime_dispatch_status"] != RuntimeDispatchStatusCompileReadyPending ||
		inspection.Model.Labels["runtime_lane"] != RuntimeLaneCUDA ||
		!runtimeLaneBackendTestNotesContain(inspection.Notes, runtimeLaneBackendPendingDetail) {
		t.Fatalf("cuda inspection labels/notes = %+v model=%+v notes=%+v, want lane-aware metadata inspection", inspection.Labels, inspection.Model.Labels, inspection.Notes)
	}
	cudaPlanner, ok := cudaBackend.(inference.ModelFitPlanner)
	if !ok {
		t.Fatal("cuda backend does not expose model-fit planning")
	}
	fit, err := cudaPlanner.PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture:  "gemma4_text",
		QuantBits:     6,
		QuantType:     "q6",
		ContextLength: 48000,
		NumLayers:     35,
		HiddenSize:    2304,
	}, 16<<30)
	if err != nil {
		t.Fatalf("cuda PlanModelFit: %v", err)
	}
	if fit.Model.Labels["runtime_lane"] != RuntimeLaneCUDA ||
		fit.MemoryPlan.Labels["backend"] != "cuda" ||
		fit.MemoryPlan.Labels["runtime_dispatch_status"] != RuntimeDispatchStatusCompileReadyPending ||
		!runtimeLaneBackendTestNotesContain(fit.Notes, runtimeLaneBackendPendingDetail) {
		t.Fatalf("cuda model fit = %+v, want lane-aware pending-dispatch plan", fit)
	}

	cpuBackend, ok := inference.Get("cpu")
	if !ok || cpuBackend.Name() != "cpu" {
		t.Fatalf("cpu backend = %+v ok=%v, want registered runtime lane backend", cpuBackend, ok)
	}
	cpuReport, ok := inference.CapabilitiesOf(cpuBackend)
	if !ok {
		t.Fatal("cpu backend did not expose capabilities")
	}
	if cpuReport.Available ||
		cpuReport.Runtime.Backend != "cpu" ||
		!strings.Contains(cpuReport.Labels["runtime_lanes"], RuntimeLaneCPUX86) ||
		!strings.Contains(cpuReport.Labels["runtime_lanes"], RuntimeLaneCPUAArch64) ||
		!strings.Contains(cpuReport.Labels["sidecars"], RuntimeLaneSidecarCPUX86) ||
		!strings.Contains(cpuReport.Labels["sidecars"], RuntimeLaneSidecarCPUAArch64) ||
		cpuReport.Labels["runtime_dispatch_status"] != RuntimeDispatchStatusCompileReadyPending {
		t.Fatalf("cpu capability report = %+v, want both pending CPU artifact lanes", cpuReport)
	}
	if _, err := cpuBackend.LoadModel(t.TempDir()); err == nil ||
		!strings.Contains(err.Error(), "runtime dispatch is pending") ||
		!strings.Contains(err.Error(), RuntimeLaneSidecarCPUX86) ||
		!strings.Contains(err.Error(), RuntimeLaneSidecarCPUAArch64) {
		t.Fatalf("cpu LoadModel error = %v, want pending dispatch sidecar error", err)
	}
	cpuPlanner, ok := cpuBackend.(inference.ModelFitPlanner)
	if !ok {
		t.Fatal("cpu backend does not expose model-fit planning")
	}
	cpuFit, err := cpuPlanner.PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture:  "gemma4_text",
		QuantBits:     6,
		QuantType:     "q6",
		ContextLength: 48000,
		NumLayers:     35,
		HiddenSize:    2304,
	}, 32<<30)
	if err != nil {
		t.Fatalf("cpu PlanModelFit: %v", err)
	}
	if !strings.Contains(cpuFit.Model.Labels["runtime_lanes"], RuntimeLaneCPUX86) ||
		!strings.Contains(cpuFit.Model.Labels["runtime_lanes"], RuntimeLaneCPUAArch64) ||
		cpuFit.MemoryPlan.Labels["backend"] != "cpu" ||
		cpuFit.MemoryPlan.Labels["runtime_dispatch_status"] != RuntimeDispatchStatusCompileReadyPending {
		t.Fatalf("cpu model fit = %+v, want both pending CPU lanes", cpuFit)
	}
}

func runtimeLaneBackendTestSafetensorsPack(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	config := []byte(`{"model_type":"gemma4_text","architectures":["Gemma4ForCausalLM"],"num_hidden_layers":35,"hidden_size":2304,"vocab_size":262144,"max_position_embeddings":48000}`)
	result := core.WriteFile(core.PathJoin(dir, "config.json"), config, 0o644)
	core.RequireTrue(t, result.OK)
	header := []byte(`{"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[0,16]},"__metadata__":{"format":"pt"}}`)
	buf := core.NewBuffer()
	core.RequireNoError(t, binary.Write(buf, binary.LittleEndian, uint64(len(header))))
	_, err := buf.Write(header)
	core.RequireNoError(t, err)
	_, err = buf.Write(make([]byte, 16))
	core.RequireNoError(t, err)
	result = core.WriteFile(core.PathJoin(dir, "model.safetensors"), buf.Bytes(), 0o644)
	core.RequireTrue(t, result.OK)
	return dir
}

func runtimeLaneBackendTestNotesContain(notes []string, needle string) bool {
	for _, note := range notes {
		if strings.Contains(note, needle) {
			return true
		}
	}
	return false
}

func TestRegisterRocm_ROCmAvailable_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	available := ROCmAvailable()
	core.AssertEqual(t, available, ROCmAvailable())
	core.AssertEqual(t, (&rocmBackend{}).Available(), available)
}

func TestRegisterRocm_ROCmAvailable_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	available := ROCmAvailable()
	core.AssertNotEqual(t, "", core.Sprintf("%v", available))
	core.AssertEqual(t, "linux", "linux")
}

func TestRegisterRocm_ROCmAvailable_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	first := ROCmAvailable()
	second := ROCmAvailable()
	core.AssertEqual(t, first, second)
}
