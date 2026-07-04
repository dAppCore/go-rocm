// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestGRPOAdamWUpdatePass_ComposesAdvantageAndOptimizer_Good(t *testing.T) {
	native := &fakeNativeModel{
		grpoKernelOK:  true,
		grpoKernelOut: []float64{-1.2247, 0, 1.2247},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "policy.lora", Shape: []int{2}, Values: []float32{1, 2}},
	}, NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true})
	core.RequireNoError(t, err)

	result, ok, err := RunNativeGRPOAdamWUpdatePass(context.Background(), model, &sliceInferenceSamples{samples: []inference.DatasetSample{
		{Labels: map[string]string{"reward": "1"}},
		{Labels: map[string]string{"reward": "2"}},
		{Labels: map[string]string{"reward": "3"}},
	}}, state, [][]float32{{0.5, -0.25}}, inference.GRPOConfig{
		TrainingConfig: inference.TrainingConfig{Labels: map[string]string{"run": "grpo-adamw"}},
		GroupSize:      3,
		KLWeight:       0.2,
	})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, 3, result.Metrics.Samples)
	core.AssertEqual(t, 1, result.Metrics.Step)
	core.AssertEqual(t, 0.1, result.Metrics.LearningRate)
	assertAdamWFloat32Near(t, 0.9, state.Parameters()[0], 0.0001)
	assertAdamWFloat32Near(t, 2.1, state.Parameters()[1], 0.0001)
	core.AssertEqual(t, "grpo-adamw", result.Labels["run"])
	core.AssertEqual(t, "grpo_advantage_adamw_update_pass", result.Labels["training_stage"])
	core.AssertEqual(t, "advantage_plus_optimizer_update", result.Labels["training_interface"])
	core.AssertEqual(t, "applied", result.Labels["training_update_status"])
	core.AssertEqual(t, "not_implemented", result.Labels["trainer_interface"])
	core.AssertEqual(t, "true", result.Labels["advantage_native_ready"])
	core.AssertEqual(t, "hip", result.Labels["advantage_backend"])
	core.AssertEqual(t, "-1.2247,0,1.2247", result.Labels["advantages"])
	core.AssertEqual(t, "adamw", result.Labels["optimizer"])
	core.AssertEqual(t, "reference", result.Labels["optimizer_backend"])
	core.AssertEqual(t, "1", result.Labels["optimizer_step"])
	core.AssertEqual(t, "2", result.Labels["optimizer_parameters"])
	core.AssertEqual(t, 1, native.grpoKernelCalls)
}

func TestGRPOAdamWUpdateTrackPass_AppendsOptimizerState_Good(t *testing.T) {
	native := &fakeNativeModel{
		grpoKernelOK:  true,
		grpoKernelOut: []float64{-1.2247, 0, 1.2247},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "policy.lora", Shape: []int{2}, Values: []float32{1, 2}},
	}, NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true})
	core.RequireNoError(t, err)
	trackPath := core.PathJoin(t.TempDir(), "training", "grpo-advantage-adamw.kv")

	result, record, ok, err := RunNativeGRPOAdamWUpdateTrackPass(context.Background(), model, &sliceInferenceSamples{samples: []inference.DatasetSample{
		{Labels: map[string]string{"reward": "1"}},
		{Labels: map[string]string{"reward": "2"}},
		{Labels: map[string]string{"reward": "3"}},
	}}, state, [][]float32{{0.5, -0.25}}, trackPath, inference.GRPOConfig{
		TrainingConfig: inference.TrainingConfig{Labels: map[string]string{"run": "grpo-advantage-track"}},
		GroupSize:      3,
	})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "grpo-advantage-track", result.Labels["run"])
	core.AssertEqual(t, "grpo_advantage_adamw_update_track_pass", result.Labels["training_stage"])
	core.AssertEqual(t, "append_only", result.Labels["optimizer_track"])
	core.AssertEqual(t, NativeAdamWTrackContainerKV, result.Labels["optimizer_track_container"])
	core.AssertEqual(t, "rocm_adamw_track_v1", result.Labels["optimizer_track_format"])
	core.AssertEqual(t, "0", result.Labels["optimizer_track_offset"])
	core.AssertEqual(t, "1", result.Labels["optimizer_track_step"])
	core.AssertEqual(t, "1", result.Labels["optimizer_track_frames"])
	core.AssertEqual(t, "LoadNativeAdamWStateTrackStep", result.Labels["optimizer_track_load_step_helper"])
	core.AssertEqual(t, trackPath, result.Labels["optimizer_track_path"])
	core.AssertEqual(t, int64(0), record.Offset)
	core.AssertEqual(t, 1, record.Step)

	loaded, loadedRecord, err := LoadLastNativeAdamWStateTrack(trackPath)
	core.RequireNoError(t, err)
	core.AssertEqual(t, record, loadedRecord)
	core.AssertEqual(t, state.Step, loaded.Step)
	core.AssertEqual(t, state.Parameters(), loaded.Parameters())
}

func TestGRPOAdamWUpdatePass_Bad(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{}}
	state, err := NewNativeAdamWState([]NativeAdamWParam{{Values: []float32{1, 2}}}, NativeAdamWConfig{})
	core.RequireNoError(t, err)

	_, _, err = RunNativeGRPOAdamWUpdatePass(context.Background(), model, &singleInferenceSample{}, nil, [][]float32{{1, 1}}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "state is nil")

	_, _, err = RunNativeGRPOAdamWUpdatePass(context.Background(), model, nil, state, [][]float32{{1, 1}}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dataset is nil")

	result, ok, err := RunNativeGRPOAdamWUpdatePass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"reward": "1"},
	}}, state, [][]float32{{1}}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertFalse(t, ok)
	core.AssertNotNil(t, result)
	core.AssertContains(t, err.Error(), "does not match")
	core.AssertEqual(t, "grpo_advantage_pass", result.Labels["training_stage"])

	_, _, _, err = RunNativeGRPOAdamWUpdateTrackPass(context.Background(), model, &singleInferenceSample{}, state, [][]float32{{1, 1}}, "", inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "track path is required")
}
