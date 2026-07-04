// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestSFTLossPass_UsesNativeCrossEntropyKernel_Good(t *testing.T) {
	native := &fakeNativeModel{
		classLogits:       [][]float32{{0, 3}},
		evalLossKernelOK:  true,
		evalLossKernelOut: hipCrossEntropyLossResult{Loss: 0.25, Perplexity: 1.284025},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}

	result, ok, err := RunNativeSFTLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Prompt: "hello",
		Labels: map[string]string{"target_token_id": "1"},
	}}, inference.TrainingConfig{BatchSize: 1, Labels: map[string]string{"run": "sft-loss", "ssd_eval_temperature": "0.25"}})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	assertFloat64Near(t, 0.25, result.Metrics.Loss, 0.0001)
	core.AssertEqual(t, 1, result.Metrics.Samples)
	core.AssertEqual(t, "sft-loss", result.Labels["run"])
	core.AssertEqual(t, "hip", result.Labels["loss_backend"])
	core.AssertEqual(t, hipKernelStatusLinked, result.Labels["loss_kernel"])
	core.AssertEqual(t, hipKernelNameCrossEntropy, result.Labels["loss_kernel_name"])
	core.AssertEqual(t, "sft_loss_pass", result.Labels["training_stage"])
	core.AssertEqual(t, "not_applied", result.Labels["training_update_status"])
	core.AssertEqual(t, "not_implemented", result.Labels["trainer_interface"])
	core.AssertEqual(t, "0.25", result.Labels["eval.temperature"])
	core.AssertEqual(t, "0.25", result.Labels["training_eval_temperature"])
	core.AssertEqual(t, "hip", result.Labels["eval.loss_backend"])
	core.AssertEqual(t, 1, native.evalLossKernelCalls)
}

func TestSFTLossPass_ReferenceLossIsNotNativeReady_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native: &fakeNativeModel{
			classLogits: [][]float32{{2, 0}},
		},
	}

	result, ok, err := RunNativeSFTLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Prompt: "hello",
		Labels: map[string]string{"target_token_id": "0"},
	}}, inference.TrainingConfig{})

	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
	assertFloat64Near(t, 0.1269, result.Metrics.Loss, 0.0001)
	core.AssertEqual(t, "reference", result.Labels["loss_backend"])
	core.AssertEqual(t, hipKernelStatusNotLinked, result.Labels["loss_kernel"])
	core.AssertEqual(t, "not_applied", result.Labels["training_update_status"])
}

func TestSFTLossPass_Bad(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{}}
	_, _, err := RunNativeSFTLossPass(context.Background(), nil, &singleInferenceSample{}, inference.TrainingConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model is nil")

	_, _, err = RunNativeSFTLossPass(context.Background(), trainingKernelNonROCmModel{}, &singleInferenceSample{}, inference.TrainingConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "requires a ROCm model")

	_, _, err = RunNativeSFTLossPass(context.Background(), model, nil, inference.TrainingConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dataset is nil")

	_, _, err = RunNativeSFTLossPass(context.Background(), model, &sliceInferenceSamples{}, inference.TrainingConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dataset produced no samples")

	_, _, err = RunNativeSFTLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Prompt: "hello",
		Labels: map[string]string{"target_token_id": "0"},
	}}, inference.TrainingConfig{Labels: map[string]string{"ssd_eval_temperature": "bad"}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "SSD eval temperature")
}
