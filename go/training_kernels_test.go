// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestTrainingKernels_RunNativeCrossEntropyLoss_Good(t *testing.T) {
	native := &fakeNativeModel{
		evalLossKernelOK:  true,
		evalLossKernelOut: hipCrossEntropyLossResult{Loss: 0.25, Perplexity: 1.284025},
	}
	result, ok, err := RunNativeCrossEntropyLoss(context.Background(), &rocmModel{native: native}, [][]float32{{1, 0}}, []int{0})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	assertFloat64Near(t, 0.25, result.Loss, 0.0001)
	assertFloat64Near(t, 1.284025, result.Perplexity, 0.0001)
	core.AssertEqual(t, 1, native.evalLossKernelCalls)
}

func TestTrainingKernels_RunNativeDistillationKLLoss_Good(t *testing.T) {
	native := &fakeNativeModel{
		distillKernelOK:  true,
		distillKernelOut: hipDistillationKLLossResult{KL: 0.0671},
	}
	result, ok, err := RunNativeDistillationKLLoss(context.Background(), &rocmModel{native: native}, [][]float32{{1, 0}}, [][]float32{{2, 0}}, 1)

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	assertFloat64Near(t, 0.0671, result.KL, 0.0001)
	core.AssertEqual(t, 1, native.distillKernelCalls)
}

func TestTrainingKernels_RunNativeGRPOAdvantage_Good(t *testing.T) {
	native := &fakeNativeModel{
		grpoKernelOK:  true,
		grpoKernelOut: []float64{-1.2247, 0, 1.2247},
	}
	result, ok, err := RunNativeGRPOAdvantage(context.Background(), &rocmModel{native: native}, []float64{1, 2, 3})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, []float64{-1.2247, 0, 1.2247}, result)
	core.AssertEqual(t, 1, native.grpoKernelCalls)
	result[0] = 99
	if native.grpoKernelOut[0] == 99 {
		t.Fatalf("GRPO result aliases native output: %+v", result)
	}
}

func TestTrainingKernels_RunNativeAdamWUpdate_Good(t *testing.T) {
	native := &fakeAdamWUpdateNativeModel{fakeNativeModel: fakeNativeModel{kernelStatus: hipKernelStatus{Optimizer: hipKernelStatusLinked}}}
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "w", Values: []float32{1, 2}},
	}, NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true})
	core.RequireNoError(t, err)

	ok, err := RunNativeAdamWUpdate(context.Background(), &rocmModel{native: native}, state, [][]float32{{0.5, -0.25}})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, 1, native.calls)
	core.AssertEqual(t, 1, state.Step)
	assertAdamWFloat32Near(t, 0.9, state.Parameters()[0], 0.0001)
	assertAdamWFloat32Near(t, 2.1, state.Parameters()[1], 0.0001)
}

func TestTrainingKernels_RunNativeUnavailable_Good(t *testing.T) {
	native := &fakeNativeModel{}

	_, crossEntropyOK, err := RunNativeCrossEntropyLoss(context.Background(), &rocmModel{native: native}, [][]float32{{1, 0}}, []int{0})
	core.RequireNoError(t, err)
	core.AssertFalse(t, crossEntropyOK)
	_, distillOK, err := RunNativeDistillationKLLoss(context.Background(), &rocmModel{native: native}, [][]float32{{1, 0}}, [][]float32{{2, 0}}, 1)
	core.RequireNoError(t, err)
	core.AssertFalse(t, distillOK)
	grpo, grpoOK, err := RunNativeGRPOAdvantage(context.Background(), &rocmModel{native: native}, []float64{1, 2, 3})
	core.RequireNoError(t, err)
	core.AssertFalse(t, grpoOK)
	core.AssertEqual(t, []float64(nil), grpo)
}

func TestTrainingKernels_RunNativeRejectsNonROCm_Bad(t *testing.T) {
	model := trainingKernelNonROCmModel{}

	_, _, err := RunNativeCrossEntropyLoss(context.Background(), model, [][]float32{{1, 0}}, []int{0})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "requires a ROCm model")
	_, _, err = RunNativeDistillationKLLoss(context.Background(), model, [][]float32{{1, 0}}, [][]float32{{2, 0}}, 1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "requires a ROCm model")
	_, _, err = RunNativeGRPOAdvantage(context.Background(), model, []float64{1, 2, 3})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "requires a ROCm model")
	_, err = RunNativeAdamWUpdate(context.Background(), model, nil, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "requires a ROCm model")
}

func TestTrainingKernels_RunNativeRejectsNil_Bad(t *testing.T) {
	_, _, err := RunNativeCrossEntropyLoss(context.Background(), nil, [][]float32{{1, 0}}, []int{0})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model is nil")
	_, err = RunNativeAdamWUpdate(context.Background(), nil, nil, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model is nil")
}

type trainingKernelNonROCmModel struct{}

func (trainingKernelNonROCmModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return emptyTokenSeq
}
func (trainingKernelNonROCmModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return emptyTokenSeq
}
func (trainingKernelNonROCmModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}
func (trainingKernelNonROCmModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}
func (trainingKernelNonROCmModel) ModelType() string         { return "other" }
func (trainingKernelNonROCmModel) Info() inference.ModelInfo { return inference.ModelInfo{} }
func (trainingKernelNonROCmModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{}
}
func (trainingKernelNonROCmModel) Err() error   { return nil }
func (trainingKernelNonROCmModel) Close() error { return nil }
