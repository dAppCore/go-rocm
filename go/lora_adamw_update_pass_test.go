// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestLoRABackwardPass_ComputesABGradients_Good(t *testing.T) {
	gradients, err := RunNativeLoRABackwardPass(
		[]float32{2, 3},
		[]float32{1, 1},
		[]float32{2, -1},
		[]float32{4, -2},
		2,
		2,
		1,
		0.5,
	)

	core.RequireNoError(t, err)
	if len(gradients) != 2 {
		t.Fatalf("gradients = %+v, want A/B tensors", gradients)
	}
	assertFloat32SlicesNear(t, []float32{10, 15}, gradients[0], 0)
	assertFloat32SlicesNear(t, []float32{10, -5}, gradients[1], 0)
}

func TestLoRAAdamWUpdatePass_ComposesBackwardAndOptimizer_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}
	state, err := NewNativeLoRAAdamWState(
		[]float32{1, 1},
		[]float32{2, -1},
		2,
		2,
		1,
		NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true},
	)
	core.RequireNoError(t, err)

	result, err := RunNativeLoRAAdamWUpdatePass(context.Background(), model, state, []float32{2, 3}, []float32{4, -2}, 2, 2, 1, 0.5, inference.TrainingConfig{
		Labels: map[string]string{"run": "lora-adamw"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, result.Metrics.Step)
	core.AssertEqual(t, "lora-adamw", result.Labels["run"])
	core.AssertEqual(t, "lora_backward_adamw_update_pass", result.Labels["training_stage"])
	core.AssertEqual(t, "lora_backward_plus_optimizer_update", result.Labels["training_interface"])
	core.AssertEqual(t, "applied", result.Labels["training_update_status"])
	core.AssertEqual(t, "not_implemented", result.Labels["trainer_interface"])
	core.AssertEqual(t, "reference", result.Labels["lora_backward_backend"])
	core.AssertEqual(t, hipKernelStatusNotLinked, result.Labels["lora_backward_kernel"])
	core.AssertEqual(t, "lora_a,lora_b", result.Labels["lora_backward_parameters"])
	core.AssertEqual(t, "1", result.Labels["lora_backward_rank"])
	core.AssertEqual(t, "adamw", result.Labels["optimizer"])
	core.AssertEqual(t, "4", result.Labels["optimizer_parameters"])
	assertAdamWFloat32Near(t, 0.9, state.Parameters()[0], 0.0001)
	assertAdamWFloat32Near(t, 0.9, state.Parameters()[1], 0.0001)
	assertAdamWFloat32Near(t, 1.9, state.Parameters()[2], 0.0001)
	assertAdamWFloat32Near(t, -0.9, state.Parameters()[3], 0.0001)
}

func TestLoRAAdamWUpdateTrackPass_AppendsOptimizerState_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}
	state, err := NewNativeLoRAAdamWState(
		[]float32{1, 1},
		[]float32{2, -1},
		2,
		2,
		1,
		NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true},
	)
	core.RequireNoError(t, err)
	trackPath := core.PathJoin(t.TempDir(), "training", "lora-adamw.mp4")

	first, firstRecord, err := RunNativeLoRAAdamWUpdateTrackPass(context.Background(), model, state, []float32{2, 3}, []float32{4, -2}, 2, 2, 1, 0.5, trackPath, inference.TrainingConfig{
		Labels: map[string]string{"run": "lora-track"},
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int64(0), firstRecord.Offset)
	core.AssertEqual(t, 1, firstRecord.Step)
	core.AssertEqual(t, "lora_backward_adamw_update_track_pass", first.Labels["training_stage"])
	core.AssertEqual(t, "append_only", first.Labels["optimizer_track"])
	core.AssertEqual(t, NativeAdamWTrackContainerMP4, first.Labels["optimizer_track_container"])
	core.AssertEqual(t, "rocm_adamw_track_v1", first.Labels["optimizer_track_format"])
	core.AssertEqual(t, "0", first.Labels["optimizer_track_offset"])
	core.AssertEqual(t, "1", first.Labels["optimizer_track_step"])
	core.AssertEqual(t, "1", first.Labels["optimizer_track_frames"])
	core.AssertEqual(t, "ListNativeAdamWStateTrack", first.Labels["optimizer_track_list_helper"])
	core.AssertEqual(t, "FindNativeAdamWStateTrackStep", first.Labels["optimizer_track_find_helper"])
	core.AssertEqual(t, "LoadNativeAdamWStateTrackStep", first.Labels["optimizer_track_load_step_helper"])
	core.AssertEqual(t, trackPath, first.Labels["optimizer_track_path"])

	second, secondRecord, err := RunNativeLoRAAdamWUpdateTrackPass(context.Background(), model, state, []float32{2, 3}, []float32{4, -2}, 2, 2, 1, 0.5, trackPath, inference.TrainingConfig{})
	core.RequireNoError(t, err)
	if secondRecord.Offset <= firstRecord.Offset {
		t.Fatalf("second record offset = %d, first = %d, want append-only growth", secondRecord.Offset, firstRecord.Offset)
	}
	core.AssertEqual(t, 2, secondRecord.Step)
	core.AssertEqual(t, "2", second.Labels["optimizer_track_step"])
	core.AssertEqual(t, "2", second.Labels["optimizer_track_frames"])
	last, lastRecord, err := LoadLastNativeAdamWStateTrack(trackPath)
	core.RequireNoError(t, err)
	core.AssertEqual(t, secondRecord.Offset, lastRecord.Offset)
	core.AssertEqual(t, 2, last.Step)
	assertAdamWFloat32Near(t, state.Parameters()[0], last.Parameters()[0], 0)
}

func TestLoRAAdamWUpdatePass_Bad(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{}}
	state, err := NewNativeLoRAAdamWState([]float32{1, 1}, []float32{1, 1}, 2, 2, 1, NativeAdamWConfig{})
	core.RequireNoError(t, err)

	_, err = RunNativeLoRABackwardPass([]float32{1}, []float32{1, 1}, []float32{1, 1}, []float32{1, 1}, 2, 2, 1, 1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input length")

	_, err = RunNativeLoRAAdamWUpdatePass(context.Background(), model, state, []float32{1, 1}, []float32{1, 1}, 2, 2, 2, 1, inference.TrainingConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "layout does not match")

	_, err = RunNativeLoRAAdamWUpdatePass(context.Background(), nil, state, []float32{1, 1}, []float32{1, 1}, 2, 2, 1, 1, inference.TrainingConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model is nil")

	beforeStep := state.Step
	_, _, err = RunNativeLoRAAdamWUpdateTrackPass(context.Background(), model, state, []float32{1, 1}, []float32{1, 1}, 2, 2, 1, 1, "", inference.TrainingConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "track path is required")
	core.AssertEqual(t, beforeStep, state.Step)
}
