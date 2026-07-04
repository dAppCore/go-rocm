// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestDistillationAdamWUpdatePass_ComposesLossAndOptimizer_Good(t *testing.T) {
	native := &fakeNativeModel{
		distillKernelOK:  true,
		distillKernelOut: hipDistillationKLLossResult{KL: 0.0671},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "student.lora", Shape: []int{2}, Values: []float32{1, 2}},
	}, NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true})
	core.RequireNoError(t, err)

	result, ok, err := RunNativeDistillationAdamWUpdatePass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"student_logits": "1,0", "teacher_logits": "2,0"},
	}}, state, [][]float32{{0.5, -0.25}}, inference.DistillConfig{
		TrainingConfig: inference.TrainingConfig{Labels: map[string]string{"run": "distill-adamw"}},
		Temperature:    1,
	})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, 1, result.Metrics.Samples)
	core.AssertEqual(t, 1, result.Metrics.Step)
	core.AssertEqual(t, 0.1, result.Metrics.LearningRate)
	assertFloat64Near(t, 0.0671, result.Metrics.Loss, 0.0001)
	assertAdamWFloat32Near(t, 0.9, state.Parameters()[0], 0.0001)
	assertAdamWFloat32Near(t, 2.1, state.Parameters()[1], 0.0001)
	core.AssertEqual(t, "distill-adamw", result.Labels["run"])
	core.AssertEqual(t, "distillation_loss_adamw_update_pass", result.Labels["training_stage"])
	core.AssertEqual(t, "loss_plus_optimizer_update", result.Labels["training_interface"])
	core.AssertEqual(t, "applied", result.Labels["training_update_status"])
	core.AssertEqual(t, "not_implemented", result.Labels["trainer_interface"])
	core.AssertEqual(t, "true", result.Labels["loss_native_ready"])
	core.AssertEqual(t, "hip", result.Labels["loss_backend"])
	core.AssertEqual(t, hipKernelNameDistillKL, result.Labels["loss_kernel_name"])
	core.AssertEqual(t, "adamw", result.Labels["optimizer"])
	core.AssertEqual(t, "reference", result.Labels["optimizer_backend"])
	core.AssertEqual(t, "1", result.Labels["optimizer_step"])
	core.AssertEqual(t, "2", result.Labels["optimizer_parameters"])
	core.AssertEqual(t, 1, native.distillKernelCalls)
}

func TestDistillationAdamWUpdateTrackPass_AppendsOptimizerState_Good(t *testing.T) {
	native := &fakeNativeModel{
		distillKernelOK:  true,
		distillKernelOut: hipDistillationKLLossResult{KL: 0.0671},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "student.lora", Shape: []int{2}, Values: []float32{1, 2}},
	}, NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true})
	core.RequireNoError(t, err)
	trackPath := core.PathJoin(t.TempDir(), "training", "distillation-adamw.mp4")

	result, record, ok, err := RunNativeDistillationAdamWUpdateTrackPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"student_logits": "1,0", "teacher_logits": "2,0"},
	}}, state, [][]float32{{0.5, -0.25}}, trackPath, inference.DistillConfig{
		TrainingConfig: inference.TrainingConfig{Labels: map[string]string{"run": "distill-track"}},
		Temperature:    1,
	})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "distill-track", result.Labels["run"])
	core.AssertEqual(t, "distillation_loss_adamw_update_track_pass", result.Labels["training_stage"])
	core.AssertEqual(t, "append_only", result.Labels["optimizer_track"])
	core.AssertEqual(t, NativeAdamWTrackContainerMP4, result.Labels["optimizer_track_container"])
	core.AssertEqual(t, "rocm_adamw_track_v1", result.Labels["optimizer_track_format"])
	core.AssertEqual(t, "0", result.Labels["optimizer_track_offset"])
	core.AssertEqual(t, "1", result.Labels["optimizer_track_step"])
	core.AssertEqual(t, "1", result.Labels["optimizer_track_frames"])
	core.AssertEqual(t, "LoadNativeAdamWStateTrackStep", result.Labels["optimizer_track_load_step_helper"])
	core.AssertEqual(t, trackPath, result.Labels["optimizer_track_path"])
	core.AssertEqual(t, 0, int(record.Offset))
	core.AssertEqual(t, 1, record.Step)

	loaded, loadedRecord, err := LoadLastNativeAdamWStateTrack(trackPath)
	core.RequireNoError(t, err)
	core.AssertEqual(t, record, loadedRecord)
	core.AssertEqual(t, state.Step, loaded.Step)
	core.AssertEqual(t, state.Parameters(), loaded.Parameters())
}

func TestDistillationAdamWUpdatePass_Bad(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{}}
	state, err := NewNativeAdamWState([]NativeAdamWParam{{Values: []float32{1, 2}}}, NativeAdamWConfig{})
	core.RequireNoError(t, err)

	_, _, err = RunNativeDistillationAdamWUpdatePass(context.Background(), model, &singleInferenceSample{}, nil, [][]float32{{1, 1}}, inference.DistillConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "state is nil")

	_, _, err = RunNativeDistillationAdamWUpdatePass(context.Background(), model, nil, state, [][]float32{{1, 1}}, inference.DistillConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dataset is nil")

	result, ok, err := RunNativeDistillationAdamWUpdatePass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"student_logits": "1,0", "teacher_logits": "2,0"},
	}}, state, [][]float32{{1}}, inference.DistillConfig{})
	core.AssertError(t, err)
	core.AssertFalse(t, ok)
	core.AssertNotNil(t, result)
	core.AssertContains(t, err.Error(), "does not match")
	core.AssertEqual(t, "distillation_loss_pass", result.Labels["training_stage"])

	_, _, _, err = RunNativeDistillationAdamWUpdateTrackPass(context.Background(), model, &singleInferenceSample{}, state, [][]float32{{1, 1}}, "", inference.DistillConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "track path is required")
}
