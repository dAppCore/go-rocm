// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestROCmRuntimeGate_Good_SetRestoreTypedGate(t *testing.T) {
	restoreStart := SetROCmRuntimeGate(ROCmGateNativeMLPMatVec, false)
	t.Cleanup(restoreStart)

	restore := SetROCmRuntimeGate(ROCmGateNativeMLPMatVec, true)
	if !ROCmRuntimeGateEnabled(ROCmGateNativeMLPMatVec) {
		t.Fatal("SetROCmRuntimeGate(true) did not enable native MLP matvec")
	}
	restore()
	if ROCmRuntimeGateEnabled(ROCmGateNativeMLPMatVec) {
		t.Fatal("SetROCmRuntimeGate restore did not restore native MLP matvec")
	}

	unknown := ROCmRuntimeGateID("unknown-gate")
	restoreUnknown := SetROCmRuntimeGate(unknown, true)
	restoreUnknown()
	if ROCmRuntimeGateEnabled(unknown) {
		t.Fatal("unknown runtime gate became enabled")
	}
}

func TestROCmEngineFeatures_Good_ApplyRuntimeGates(t *testing.T) {
	restoreDirect := SetROCmRuntimeGate(ROCmGateDirectGreedyToken, false)
	restoreMLP := SetROCmRuntimeGate(ROCmGateNativeMLPMatVec, false)
	restorePipeline := SetROCmRuntimeGate(ROCmGatePipelinedDecode, false)
	t.Cleanup(func() {
		restorePipeline()
		restoreMLP()
		restoreDirect()
	})

	features := ROCmEngineFeatures{
		DirectGreedyToken: true,
		NativeMLPMatVec:   true,
		PipelinedDecode:   true,
	}
	if got := features.EnabledRuntimeGates(); !slices.Equal(got, []ROCmRuntimeGateID{
		ROCmGateDirectGreedyToken,
		ROCmGateNativeMLPMatVec,
		ROCmGatePipelinedDecode,
	}) {
		t.Fatalf("EnabledRuntimeGates = %v, want direct/mlp/pipeline", got)
	}

	restore := features.ApplyRuntimeGates()
	if !ROCmRuntimeGateEnabled(ROCmGateDirectGreedyToken) ||
		!ROCmRuntimeGateEnabled(ROCmGateNativeMLPMatVec) ||
		!ROCmRuntimeGateEnabled(ROCmGatePipelinedDecode) {
		t.Fatalf("ApplyRuntimeGates did not enable declared gates: %+v", ROCmRuntimeGateSnapshot())
	}
	if ROCmRuntimeGateEnabled(ROCmGateNativeLinearMatVec) {
		t.Fatal("ApplyRuntimeGates enabled an undeclared gate")
	}
	restore()
	if ROCmRuntimeGateEnabled(ROCmGateDirectGreedyToken) ||
		ROCmRuntimeGateEnabled(ROCmGateNativeMLPMatVec) ||
		ROCmRuntimeGateEnabled(ROCmGatePipelinedDecode) {
		t.Fatalf("ApplyRuntimeGates restore leaked enabled gates: %+v", ROCmRuntimeGateSnapshot())
	}
}

func TestROCmRuntimeGatePlan_Good_AppliesReactiveRoutePlan(t *testing.T) {
	for _, gate := range []ROCmRuntimeGateID{
		ROCmGateDirectGreedyToken,
		ROCmGateNativeMLPMatVec,
		ROCmGateCompiledLayerDecode,
	} {
		restore := SetROCmRuntimeGate(gate, false)
		t.Cleanup(restore)
	}

	plan, ok := ROCmModelRoutePlanForIdentity("/models/gemma4-e2b-q6", inference.ModelIdentity{
		Architecture: "gemma4_text",
		Labels: map[string]string{
			"engine_feature_native_mlp_matvec":     "linked",
			"engine_feature_compiled_layer_decode": "ready",
		},
	})
	if !ok || !plan.RuntimeGatePlan.Matched() {
		t.Fatalf("ROCmModelRoutePlanForIdentity = %+v ok=%v, want runtime gate plan", plan, ok)
	}

	restore := ApplyROCmRuntimeGatePlan(plan.RuntimeGatePlan)
	if !ROCmRuntimeGateEnabled(ROCmGateDirectGreedyToken) ||
		!ROCmRuntimeGateEnabled(ROCmGateNativeMLPMatVec) ||
		!ROCmRuntimeGateEnabled(ROCmGateCompiledLayerDecode) {
		t.Fatalf("ApplyROCmRuntimeGatePlan did not enable route-plan gates: %+v", ROCmRuntimeGateSnapshot())
	}
	restore()
	if ROCmRuntimeGateEnabled(ROCmGateNativeMLPMatVec) ||
		ROCmRuntimeGateEnabled(ROCmGateCompiledLayerDecode) {
		t.Fatalf("ApplyROCmRuntimeGatePlan restore leaked route-plan gates: %+v", ROCmRuntimeGateSnapshot())
	}
}

type rocmRuntimeFeatureProviderTestModel struct {
	features ROCmEngineFeatures
	plan     ROCmModelRoutePlan
}

func (model rocmRuntimeFeatureProviderTestModel) ROCmEngineFeatures() ROCmEngineFeatures {
	return model.features
}

func (model rocmRuntimeFeatureProviderTestModel) ModelRoutePlan() ROCmModelRoutePlan {
	return model.plan
}

func TestROCmRuntimeFeatures_Good_AppliesLoadedModelDeclaration(t *testing.T) {
	for _, gate := range []ROCmRuntimeGateID{
		ROCmGateNativeMLPMatVec,
		ROCmGateCompiledLayerDecode,
	} {
		restore := SetROCmRuntimeGate(gate, false)
		t.Cleanup(restore)
	}

	model := rocmRuntimeFeatureProviderTestModel{
		features: ROCmEngineFeatures{
			Architecture:    "gemma4_text",
			NativeMLPMatVec: true,
		},
		plan: ROCmModelRoutePlan{
			RuntimeGatePlan: rocmmodel.RuntimeGatePlan{
				Contract:     rocmmodel.RuntimeGatePlanContract,
				Architecture: "gemma4_text",
				Gates: []rocmmodel.RuntimeGate{{
					ID:      rocmmodel.GateCompiledLayerDecode,
					Enabled: true,
					Source:  "test",
				}},
				GateIDs: []rocmmodel.RuntimeGateID{rocmmodel.GateCompiledLayerDecode},
			},
		},
	}

	features, ok := ROCmEngineFeaturesFor(model)
	if !ok || !features.NativeMLPMatVec || features.Architecture != "gemma4_text" {
		t.Fatalf("ROCmEngineFeaturesFor(provider) = %+v ok=%v, want model-owned features", features, ok)
	}

	restore := ApplyROCmRuntimeFeaturesForModel(model)
	if !ROCmRuntimeGateEnabled(ROCmGateNativeMLPMatVec) ||
		!ROCmRuntimeGateEnabled(ROCmGateCompiledLayerDecode) {
		t.Fatalf("ApplyROCmRuntimeFeaturesForModel did not enable loaded-model gates: %+v", ROCmRuntimeGateSnapshot())
	}
	restore()
	if ROCmRuntimeGateEnabled(ROCmGateNativeMLPMatVec) ||
		ROCmRuntimeGateEnabled(ROCmGateCompiledLayerDecode) {
		t.Fatalf("ApplyROCmRuntimeFeaturesForModel restore leaked gates: %+v", ROCmRuntimeGateSnapshot())
	}
}

func TestROCmRuntimeGate_Good_SnapshotAndIDsAreDefensive(t *testing.T) {
	ids := ROCmRuntimeGateIDs()
	if !slices.Contains(ids, ROCmGateGenerationStream) ||
		!slices.Contains(ids, ROCmGatePipelinedDecode) {
		t.Fatalf("ROCmRuntimeGateIDs = %v, want go-mlx-compatible gate IDs", ids)
	}
	ids[0] = ROCmRuntimeGateID("mutated")
	if next := ROCmRuntimeGateIDs(); next[0] == ROCmRuntimeGateID("mutated") {
		t.Fatalf("ROCmRuntimeGateIDs leaked mutable state: %v", next)
	}

	restore := SetROCmRuntimeGate(ROCmGateGenerationStream, false)
	t.Cleanup(restore)
	snapshot := ROCmRuntimeGateSnapshot()
	snapshot[ROCmGateGenerationStream] = true
	if ROCmRuntimeGateEnabled(ROCmGateGenerationStream) {
		t.Fatalf("ROCmRuntimeGateSnapshot leaked mutable state: %+v", snapshot)
	}

	emptyPlanRestore := ApplyROCmRuntimeGatePlan(rocmmodel.RuntimeGatePlan{})
	emptyPlanRestore()
}
