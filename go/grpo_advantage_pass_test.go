// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestGRPOAdvantagePass_UsesNativeAdvantageKernel_Good(t *testing.T) {
	native := &fakeNativeModel{
		grpoKernelOK:  true,
		grpoKernelOut: []float64{-1.2247, 0, 1.2247},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}

	result, ok, err := RunNativeGRPOAdvantagePass(context.Background(), model, &sliceInferenceSamples{samples: []inference.DatasetSample{
		{Labels: map[string]string{"reward": "1"}},
		{Labels: map[string]string{"reward": "2"}},
		{Labels: map[string]string{"reward": "3"}},
	}}, inference.GRPOConfig{
		TrainingConfig: inference.TrainingConfig{Labels: map[string]string{"run": "grpo-advantage"}},
		GroupSize:      3,
		KLWeight:       0.2,
	})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, 3, result.Metrics.Samples)
	core.AssertEqual(t, 1, result.Metrics.Step)
	core.AssertEqual(t, "grpo-advantage", result.Labels["run"])
	core.AssertEqual(t, "hip", result.Labels["advantage_backend"])
	core.AssertEqual(t, hipKernelStatusLinked, result.Labels["advantage_kernel"])
	core.AssertEqual(t, hipKernelNameGRPOAdvantage, result.Labels["advantage_kernel_name"])
	core.AssertEqual(t, "-1.2247,0,1.2247", result.Labels["advantages"])
	core.AssertEqual(t, "grpo_advantage_pass", result.Labels["training_stage"])
	core.AssertEqual(t, "not_applied", result.Labels["training_update_status"])
	core.AssertEqual(t, "not_implemented", result.Labels["trainer_interface"])
	core.AssertEqual(t, "3", result.Labels["grpo_group_size"])
	core.AssertEqual(t, "0.2", result.Labels["grpo_kl_weight"])
	core.AssertEqual(t, 1, native.grpoKernelCalls)
}

func TestGRPOAdvantagePass_ReferenceAdvantageIsNotNativeReady_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}

	result, ok, err := RunNativeGRPOAdvantagePass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"rewards": "1,2,3"},
	}}, inference.GRPOConfig{})

	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
	core.AssertEqual(t, "reference", result.Labels["advantage_backend"])
	core.AssertEqual(t, hipKernelStatusNotLinked, result.Labels["advantage_kernel"])
	core.AssertEqual(t, "experimental", result.Labels["advantage_status"])
	core.AssertContains(t, result.Labels["advantages"], "-1.224")
	core.AssertContains(t, result.Labels["advantages"], "1.224")
}

func TestGRPOAdvantagePass_LoadsJSONLRewardLabels_Good(t *testing.T) {
	native := &fakeNativeModel{
		grpoKernelOK:  true,
		grpoKernelOut: []float64{-1, 1},
	}
	model := &rocmModel{modelType: "tiny", modelInfo: inference.ModelInfo{Architecture: "tiny"}, native: native}
	dataset, err := LoadJSONLDataset(strings.NewReader(`{"prompt":"p","response":"r","rewards":[1,3]}`))
	core.RequireNoError(t, err)

	result, ok, err := RunNativeGRPOAdvantagePass(context.Background(), model, dataset, inference.GRPOConfig{})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "2", result.Labels["grpo_rewards"])
	core.AssertEqual(t, "-1,1", result.Labels["advantages"])
}

var (
	rocmGRPOAdvantageBenchResult *inference.TrainingResult
	rocmGRPOAdvantageBenchOK     bool
	rocmGRPOAdvantageBenchErr    error
)

func BenchmarkGRPOAdvantagePass_RewardRows128(b *testing.B) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}
	samples := makeGRPORewardBenchSamples(128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rocmGRPOAdvantageBenchResult, rocmGRPOAdvantageBenchOK, rocmGRPOAdvantageBenchErr = RunNativeGRPOAdvantagePass(context.Background(), model, &sliceInferenceSamples{samples: samples}, inference.GRPOConfig{})
		if rocmGRPOAdvantageBenchErr != nil {
			b.Fatal(rocmGRPOAdvantageBenchErr)
		}
	}
}

func makeGRPORewardBenchSamples(n int) []inference.DatasetSample {
	samples := make([]inference.DatasetSample, n)
	for i := range samples {
		samples[i] = inference.DatasetSample{Labels: map[string]string{"rewards": "1,3"}}
	}
	return samples
}

func TestGRPOAdvantagePass_Bad(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{}}

	_, _, err := RunNativeGRPOAdvantagePass(context.Background(), nil, &singleInferenceSample{}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model is nil")

	_, _, err = RunNativeGRPOAdvantagePass(context.Background(), trainingKernelNonROCmModel{}, &singleInferenceSample{}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "requires a ROCm model")

	_, _, err = RunNativeGRPOAdvantagePass(context.Background(), model, nil, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dataset is nil")

	_, _, err = RunNativeGRPOAdvantagePass(context.Background(), model, &sliceInferenceSamples{}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "produced no rewards")

	_, _, err = RunNativeGRPOAdvantagePass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"reward": "nope"},
	}}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "parse reward")

	_, _, err = RunNativeGRPOAdvantagePass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"rewards": "1,"},
	}}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "parse rewards")
}
