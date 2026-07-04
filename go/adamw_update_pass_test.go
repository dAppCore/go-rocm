// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestNativeAdamWUpdatePass_AppliesPackedState_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "w", Shape: []int{2}, Values: []float32{1, 2}},
	}, NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true})
	core.RequireNoError(t, err)

	result, err := RunNativeAdamWUpdatePass(context.Background(), model, state, [][]float32{{0.5, -0.25}}, inference.TrainingConfig{
		Labels: map[string]string{"run": "adamw-update"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, state.Step)
	core.AssertEqual(t, 1, result.Metrics.Step)
	core.AssertEqual(t, 0.1, result.Metrics.LearningRate)
	assertAdamWFloat32Near(t, 0.9, state.Parameters()[0], 0.0001)
	assertAdamWFloat32Near(t, 2.1, state.Parameters()[1], 0.0001)
	core.AssertEqual(t, "adamw-update", result.Labels["run"])
	core.AssertEqual(t, "adamw_update_pass", result.Labels["training_stage"])
	core.AssertEqual(t, "optimizer_update_only", result.Labels["training_interface"])
	core.AssertEqual(t, "applied", result.Labels["training_update_status"])
	core.AssertEqual(t, "not_implemented", result.Labels["trainer_interface"])
	core.AssertEqual(t, "adamw", result.Labels["optimizer"])
	core.AssertEqual(t, "reference", result.Labels["optimizer_backend"])
	core.AssertEqual(t, hipKernelStatusNotLinked, result.Labels["optimizer_kernel"])
	core.AssertEqual(t, nativeAdamWUpdateKernelName, result.Labels["optimizer_kernel_name"])
	core.AssertEqual(t, "hipAdamWUpdateLaunchArgs", result.Labels["optimizer_launch_args"])
	core.AssertEqual(t, "128", result.Labels["optimizer_launch_args_bytes"])
	core.AssertEqual(t, hipKernelStatusNotLinked, result.Labels["hip_optimizer_update"])
	core.AssertEqual(t, "packed_contiguous_parameters_m_v", result.Labels["optimizer_state_layout"])
	core.AssertEqual(t, "1", result.Labels["optimizer_tensors"])
	core.AssertEqual(t, "2", result.Labels["optimizer_parameters"])
	core.AssertEqual(t, "1", result.Labels["optimizer_step"])
	core.AssertEqual(t, "true", result.Labels["optimizer_packed"])
}

func TestNativeAdamWUpdatePass_TrainingConfigLearningRate_Good(t *testing.T) {
	model := &rocmModel{modelType: "tiny", modelInfo: inference.ModelInfo{Architecture: "tiny"}, native: &fakeNativeModel{}}
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "w", Values: []float32{1}},
	}, NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true})
	core.RequireNoError(t, err)

	result, err := RunNativeAdamWUpdatePass(context.Background(), model, state, [][]float32{{0.5}}, inference.TrainingConfig{LearningRate: 0.01})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 0.01, result.Metrics.LearningRate)
	core.AssertEqual(t, 0.01, state.Config.LearningRate)
	assertAdamWFloat32Near(t, 0.99, state.Parameters()[0], 0.0001)
}

func TestNativeAdamWUpdatePass_UsesOptimizerKernelStatus_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native: &fakeNativeModel{kernelStatus: hipKernelStatus{
			Optimizer: hipKernelStatusLinked,
		}},
	}
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "w", Values: []float32{1}},
	}, NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true})
	core.RequireNoError(t, err)

	result, err := RunNativeAdamWUpdatePass(context.Background(), model, state, [][]float32{{0.5}}, inference.TrainingConfig{})

	core.RequireNoError(t, err)
	core.AssertEqual(t, hipKernelStatusLinked, result.Labels["optimizer_kernel"])
	core.AssertEqual(t, hipKernelStatusLinked, result.Labels["hip_optimizer_update"])
	core.AssertEqual(t, nativeAdamWUpdateKernelName, result.Labels["optimizer_kernel_name"])
	core.AssertEqual(t, "reference", result.Labels["optimizer_backend"])
	core.AssertEqual(t, 1, state.Step)
}

func TestNativeAdamWUpdatePass_UsesLinkedOptimizerHook_Good(t *testing.T) {
	native := &fakeAdamWUpdateNativeModel{fakeNativeModel: fakeNativeModel{kernelStatus: hipKernelStatus{Optimizer: hipKernelStatusLinked}}}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny"},
		native:    native,
	}
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "w", Values: []float32{1, 2}},
	}, NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true})
	core.RequireNoError(t, err)

	result, err := RunNativeAdamWUpdatePass(context.Background(), model, state, [][]float32{{0.5, -0.25}}, inference.TrainingConfig{})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, native.calls)
	core.AssertEqual(t, "hip", result.Labels["optimizer_backend"])
	core.AssertEqual(t, hipKernelStatusLinked, result.Labels["optimizer_kernel"])
	core.AssertEqual(t, hipKernelStatusLinked, result.Labels["hip_optimizer_update"])
	core.AssertEqual(t, 1, state.Step)
	core.AssertEqual(t, 1, result.Metrics.Step)
	assertAdamWFloat32Near(t, 0.9, state.Parameters()[0], 0.0001)
	assertAdamWFloat32Near(t, 2.1, state.Parameters()[1], 0.0001)
}

func TestNativeAdamWUpdatePass_Bad(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{}}
	state, err := NewNativeAdamWState([]NativeAdamWParam{{Values: []float32{1, 2}}}, NativeAdamWConfig{})
	core.RequireNoError(t, err)

	_, err = RunNativeAdamWUpdatePass(context.Background(), nil, state, [][]float32{{1, 1}}, inference.TrainingConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model is nil")

	_, err = RunNativeAdamWUpdatePass(context.Background(), trainingKernelNonROCmModel{}, state, [][]float32{{1, 1}}, inference.TrainingConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "requires a ROCm model")

	_, err = RunNativeAdamWUpdatePass(context.Background(), model, nil, [][]float32{{1, 1}}, inference.TrainingConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "state is nil")

	_, err = RunNativeAdamWUpdatePass(context.Background(), model, state, [][]float32{{1}}, inference.TrainingConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "does not match")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = RunNativeAdamWUpdatePass(ctx, model, state, [][]float32{{1, 1}}, inference.TrainingConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "context canceled")
}

type fakeAdamWUpdateNativeModel struct {
	fakeNativeModel
	calls int
}

func (model *fakeAdamWUpdateNativeModel) RunAdamWUpdate(_ context.Context, state *NativeAdamWState, gradients [][]float32) (bool, error) {
	model.calls++
	if state == nil {
		return false, nil
	}
	return true, state.StepInPlace(gradients)
}
