// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestDistillationLossPass_UsesNativeKLKernel_Good(t *testing.T) {
	native := &fakeNativeModel{
		distillKernelOK:  true,
		distillKernelOut: hipDistillationKLLossResult{KL: 0.0671},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}

	result, ok, err := RunNativeDistillationLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"student_logits": "1,0", "teacher_logits": "2,0"},
	}}, inference.DistillConfig{
		TrainingConfig: inference.TrainingConfig{Labels: map[string]string{"run": "distill-loss"}},
		Temperature:    1,
	})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	assertFloat64Near(t, 0.0671, result.Metrics.Loss, 0.0001)
	core.AssertEqual(t, 1, result.Metrics.Samples)
	core.AssertEqual(t, "distill-loss", result.Labels["run"])
	core.AssertEqual(t, "hip", result.Labels["loss_backend"])
	core.AssertEqual(t, hipKernelStatusLinked, result.Labels["loss_kernel"])
	core.AssertEqual(t, hipKernelNameDistillKL, result.Labels["loss_kernel_name"])
	core.AssertEqual(t, "distillation_loss_pass", result.Labels["training_stage"])
	core.AssertEqual(t, "not_applied", result.Labels["training_update_status"])
	core.AssertEqual(t, "not_implemented", result.Labels["trainer_interface"])
	core.AssertEqual(t, "1", result.Labels["distillation_temperature"])
	core.AssertEqual(t, "2", result.Labels["distillation_vocab"])
	core.AssertEqual(t, 1, native.distillKernelCalls)
}

func TestDistillationLossPass_ReferenceLossIsNotNativeReady_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}

	result, ok, err := RunNativeDistillationLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"student_logits": "1,0", "teacher_logits": "2,0"},
	}}, inference.DistillConfig{})

	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
	assertFloat64Near(t, 0.0671, result.Metrics.Loss, 0.0001)
	core.AssertEqual(t, "reference", result.Labels["loss_backend"])
	core.AssertEqual(t, hipKernelStatusNotLinked, result.Labels["loss_kernel"])
	core.AssertEqual(t, "experimental", result.Labels["loss_status"])
	core.AssertEqual(t, "not_applied", result.Labels["training_update_status"])
}

func TestDistillationLossPass_Bad(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{}}

	_, _, err := RunNativeDistillationLossPass(context.Background(), nil, &singleInferenceSample{}, inference.DistillConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model is nil")

	_, _, err = RunNativeDistillationLossPass(context.Background(), trainingKernelNonROCmModel{}, &singleInferenceSample{}, inference.DistillConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "requires a ROCm model")

	_, _, err = RunNativeDistillationLossPass(context.Background(), model, nil, inference.DistillConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dataset is nil")

	_, _, err = RunNativeDistillationLossPass(context.Background(), model, &sliceInferenceSamples{}, inference.DistillConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "produced no labelled samples")

	_, _, err = RunNativeDistillationLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"student_logits": "1,0"},
	}}, inference.DistillConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "requires student_logits and teacher_logits")

	_, _, err = RunNativeDistillationLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"student_logits": "1,0", "teacher_logits": "2"},
	}}, inference.DistillConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "non-empty and equal length")

	_, _, err = RunNativeDistillationLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"student_logits": "nope", "teacher_logits": "2,0"},
	}}, inference.DistillConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "parse student_logits")
}
