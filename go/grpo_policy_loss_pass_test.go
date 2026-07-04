// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"math"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestGRPOPolicyLossPass_UsesNativeAdvantageAndReferencePolicyLoss_Good(t *testing.T) {
	native := &fakeNativeModel{
		grpoKernelOK:  true,
		grpoKernelOut: []float64{-1, 1},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}

	result, ok, err := RunNativeGRPOPolicyLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{
			"rewards":            "1,3",
			"logprobs":           "0,0",
			"old_logprobs":       "0,0",
			"reference_logprobs": "0,0",
		},
	}}, inference.GRPOConfig{
		TrainingConfig: inference.TrainingConfig{Labels: map[string]string{"run": "grpo-policy"}},
		GroupSize:      2,
		KLWeight:       0.2,
	})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "grpo-policy", result.Labels["run"])
	core.AssertEqual(t, "grpo_policy_loss_pass", result.Labels["training_stage"])
	core.AssertEqual(t, "policy_loss_only", result.Labels["training_interface"])
	core.AssertEqual(t, "not_applied", result.Labels["training_update_status"])
	core.AssertEqual(t, "not_implemented", result.Labels["trainer_interface"])
	core.AssertEqual(t, "reference", result.Labels["policy_loss_backend"])
	core.AssertEqual(t, hipKernelStatusNotLinked, result.Labels["policy_loss_kernel"])
	core.AssertEqual(t, "dataset", result.Labels["policy_reference_source"])
	core.AssertEqual(t, "2", result.Labels["grpo_policy_reference_terms"])
	core.AssertEqual(t, "0", result.Labels["grpo_policy_reference_fallback_terms"])
	core.AssertEqual(t, "true", result.Labels["advantage_native_ready"])
	core.AssertEqual(t, "reward_normalization", result.Labels["advantage_source"])
	core.AssertEqual(t, "-1,1", result.Labels["advantages"])
	core.AssertEqual(t, "2", result.Labels["grpo_policy_terms"])
	core.AssertEqual(t, "2", result.Labels["grpo_policy_active_terms"])
	core.AssertEqual(t, "2", result.Labels["grpo_group_size"])
	core.AssertEqual(t, "1", result.Labels["grpo_policy_groups"])
	core.AssertEqual(t, "0.2", result.Labels["grpo_kl_weight"])
	assertFloat64Near(t, 0, result.Metrics.Loss, 0.0001)
	assertFloat64Near(t, 1, parseFloat64LabelForTest(t, result.Labels["policy_ratio_mean"]), 0.0001)
	assertFloat64Near(t, 0, parseFloat64LabelForTest(t, result.Labels["policy_kl_mean"]), 0.0001)
	core.AssertEqual(t, 1, native.grpoKernelCalls)
}

func TestGRPOPolicyLossPass_LoadsJSONLPolicyLabels_Good(t *testing.T) {
	native := &fakeNativeModel{
		grpoKernelOK:  true,
		grpoKernelOut: []float64{-1, 1},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}
	dataset, err := LoadJSONLDataset(strings.NewReader(`{"prompt":"p","response":"r","rewards":[1,3],"current_logprobs":[0,0],"old_logprobs":[0,0],"reference_logprobs":[0,0]}`))
	core.RequireNoError(t, err)

	result, ok, err := RunNativeGRPOPolicyLossPass(context.Background(), model, dataset, inference.GRPOConfig{KLWeight: 0.1})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "2", result.Labels["grpo_policy_terms"])
	core.AssertEqual(t, "0.1", result.Labels["grpo_kl_weight"])
	core.AssertEqual(t, "reward_normalization", result.Labels["advantage_source"])
	core.AssertEqual(t, "-1,1", result.Labels["advantages"])
	assertFloat64Near(t, 0, result.Metrics.Loss, 0.0001)
}

func TestGRPOPolicyLossPass_LoadsJSONLPolicyAliases_Good(t *testing.T) {
	native := &fakeNativeModel{
		grpoKernelOK:  true,
		grpoKernelOut: []float64{-1, 1},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}
	dataset, err := LoadJSONLDataset(strings.NewReader(`{"prompt":"p","response":"r","rewards":[1,3],"policy_logprobs":[0,0],"old_policy_logprobs":[0,0],"ref_logprobs":[0,0]}`))
	core.RequireNoError(t, err)

	result, ok, err := RunNativeGRPOPolicyLossPass(context.Background(), model, dataset, inference.GRPOConfig{})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "reward_normalization", result.Labels["advantage_source"])
	core.AssertEqual(t, "2", result.Labels["grpo_policy_terms"])
	assertFloat64Near(t, 0, result.Metrics.Loss, 0.0001)
}

func TestGRPOPolicyLossPass_DirectPolicyAliases_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}

	result, ok, err := RunNativeGRPOPolicyLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{
			"advantages":              "-1,1",
			"current_policy_logprobs": "0,0",
			"old_policy_logprobs":     "0,0",
			"ref_logprobs":            "0,0",
		},
	}}, inference.GRPOConfig{})

	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
	core.AssertEqual(t, "dataset", result.Labels["advantage_source"])
	core.AssertEqual(t, "2", result.Labels["grpo_policy_terms"])
	assertFloat64Near(t, 0, result.Metrics.Loss, 0.0001)
}

func TestGRPOPolicyLossPass_LoadsJSONLAdvantages_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}
	dataset, err := LoadJSONLDataset(strings.NewReader(`{"prompt":"p","response":"r","advantages":[-1,1],"current_logprobs":[0,0],"old_logprobs":[0,0]}`))
	core.RequireNoError(t, err)

	result, ok, err := RunNativeGRPOPolicyLossPass(context.Background(), model, dataset, inference.GRPOConfig{})

	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
	core.AssertEqual(t, "dataset", result.Labels["advantage_source"])
	core.AssertEqual(t, "false", result.Labels["advantage_native_ready"])
	core.AssertEqual(t, "-1,1", result.Labels["advantages"])
	core.AssertEqual(t, "2", result.Labels["grpo_policy_terms"])
	core.AssertEqual(t, 2, result.Metrics.Samples)
	assertFloat64Near(t, 0, result.Metrics.Loss, 0.0001)
}

func TestGRPOPolicyLossPass_ReferenceAdvantage_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}

	result, ok, err := RunNativeGRPOPolicyLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{
			"rewards":      "1,2,3",
			"logprobs":     "0,0,0",
			"old_logprobs": "0,0,0",
		},
	}}, inference.GRPOConfig{})

	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
	core.AssertEqual(t, "false", result.Labels["advantage_native_ready"])
	core.AssertContains(t, result.Labels["advantages"], "-1.224")
	core.AssertContains(t, result.Labels["advantages"], "1.224")
	assertFloat64Near(t, 0, result.Metrics.Loss, 0.0001)
}

func TestGRPOPolicyLossPass_ClippedPolicyObjectiveLabels_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}

	result, ok, err := RunNativeGRPOPolicyLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{
			"advantages":         "1,-1",
			"logprobs":           formatFloat64CSVLabel([]float64{math.Log(2), math.Log(0.5)}),
			"old_logprobs":       "0,0",
			"reference_logprobs": "0,0",
		},
	}}, inference.GRPOConfig{TrainingConfig: inference.TrainingConfig{Labels: map[string]string{"clip_range": "0.2"}}})

	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
	core.AssertEqual(t, "0.2", result.Labels["policy_clip_range"])
	assertFloat64Near(t, 0.75, parseFloat64LabelForTest(t, result.Labels["policy_objective_mean"]), 0.0001)
	assertFloat64Near(t, 0.35, parseFloat64LabelForTest(t, result.Labels["policy_clipped_objective_mean"]), 0.0001)
	assertFloat64Near(t, -0.35, parseFloat64LabelForTest(t, result.Labels["policy_objective_loss"]), 0.0001)
	assertFloat64Near(t, 0, parseFloat64LabelForTest(t, result.Labels["policy_kl_loss"]), 0.0001)
	assertFloat64Near(t, 0.5, parseFloat64LabelForTest(t, result.Labels["policy_clip_fraction"]), 0.0001)
	assertFloat64Near(t, 0.5, parseFloat64LabelForTest(t, result.Labels["policy_clip_low_fraction"]), 0.0001)
	assertFloat64Near(t, 0.5, parseFloat64LabelForTest(t, result.Labels["policy_clip_high_fraction"]), 0.0001)
	assertFloat64Near(t, -0.35, result.Metrics.Loss, 0.0001)
}

func TestGRPOPolicyLossPass_UsesDatasetClipRange_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}
	dataset, err := LoadJSONLDataset(strings.NewReader(`{"prompt":"p","response":"r","advantages":[1,-1],"current_logprobs":[0.6931471805599453,-0.6931471805599453],"old_logprobs":[0,0],"clip_epsilon":0.2}`))
	core.RequireNoError(t, err)

	result, ok, err := RunNativeGRPOPolicyLossPass(context.Background(), model, dataset, inference.GRPOConfig{})

	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
	core.AssertEqual(t, "0.2", result.Labels["policy_clip_range"])
	assertFloat64Near(t, 0.35, parseFloat64LabelForTest(t, result.Labels["policy_clipped_objective_mean"]), 0.0001)
	assertFloat64Near(t, -0.35, result.Metrics.Loss, 0.0001)
}

func TestGRPOPolicyLossPass_UsesDatasetPolicyWeights_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}
	dataset, err := LoadJSONLDataset(strings.NewReader(`{"prompt":"p","response":"r","advantages":[1,3],"current_logprobs":[0.6931471805599453,-0.6931471805599453],"old_logprobs":[0,0],"response_mask":[1,0]}`))
	core.RequireNoError(t, err)

	result, ok, err := RunNativeGRPOPolicyLossPass(context.Background(), model, dataset, inference.GRPOConfig{})

	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
	core.AssertEqual(t, "dataset", result.Labels["policy_weight_source"])
	core.AssertEqual(t, "1", result.Labels["policy_weight_sum"])
	core.AssertEqual(t, "old_policy_fallback", result.Labels["policy_reference_source"])
	core.AssertEqual(t, "0", result.Labels["grpo_policy_reference_terms"])
	core.AssertEqual(t, "2", result.Labels["grpo_policy_reference_fallback_terms"])
	core.AssertEqual(t, "2", result.Labels["grpo_policy_terms"])
	core.AssertEqual(t, "1", result.Labels["grpo_policy_active_terms"])
	core.AssertEqual(t, 2, result.Metrics.Samples)
	assertFloat64Near(t, 2, parseFloat64LabelForTest(t, result.Labels["policy_ratio_mean"]), 0.0001)
	assertFloat64Near(t, 2, parseFloat64LabelForTest(t, result.Labels["policy_ratio_min"]), 0.0001)
	assertFloat64Near(t, 2, parseFloat64LabelForTest(t, result.Labels["policy_ratio_max"]), 0.0001)
	assertFloat64Near(t, 2, parseFloat64LabelForTest(t, result.Labels["policy_objective_mean"]), 0.0001)
	assertFloat64Near(t, -2, result.Metrics.Loss, 0.0001)
}

func TestGRPOPolicyLossPass_ReportsRolloutGroupIDs_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}
	dataset, err := LoadJSONLDataset(strings.NewReader(core.Join("\n",
		`{"prompt":"p1","response":"r1","group_id":"g1","prompt_id":"prompt-a","advantages":[1,-1],"current_logprobs":[0,0],"old_logprobs":[0,0]}`,
		`{"prompt":"p2","response":"r2","group_id":"g2","prompt_id":"prompt-b","advantages":[1,-1],"current_logprobs":[0,0],"old_logprobs":[0,0]}`,
		`{"prompt":"p3","response":"r3","group_id":"g1","prompt_id":"prompt-a","advantages":[1,-1],"current_logprobs":[0,0],"old_logprobs":[0,0]}`,
	)))
	core.RequireNoError(t, err)

	result, ok, err := RunNativeGRPOPolicyLossPass(context.Background(), model, dataset, inference.GRPOConfig{})

	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
	core.AssertEqual(t, "dataset", result.Labels["grpo_rollout_group_source"])
	core.AssertEqual(t, "2", result.Labels["grpo_rollout_groups"])
	core.AssertEqual(t, "dataset", result.Labels["grpo_rollout_prompt_source"])
	core.AssertEqual(t, "2", result.Labels["grpo_rollout_prompts"])
	core.AssertEqual(t, "6", result.Labels["grpo_policy_terms"])
}

func TestGRPOPolicyLossPass_ReportsRolloutQueryIDsAsPrompts_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}
	dataset, err := LoadJSONLDataset(strings.NewReader(core.Join("\n",
		`{"prompt":"p1","response":"r1","query_id":"query-a","advantages":[1,-1],"current_logprobs":[0,0],"old_logprobs":[0,0]}`,
		`{"prompt":"p2","response":"r2","query_id":"query-b","advantages":[1,-1],"current_logprobs":[0,0],"old_logprobs":[0,0]}`,
		`{"prompt":"p3","response":"r3","query_id":"query-a","advantages":[1,-1],"current_logprobs":[0,0],"old_logprobs":[0,0]}`,
	)))
	core.RequireNoError(t, err)

	result, ok, err := RunNativeGRPOPolicyLossPass(context.Background(), model, dataset, inference.GRPOConfig{})

	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
	core.AssertEqual(t, "dataset", result.Labels["grpo_rollout_prompt_source"])
	core.AssertEqual(t, "2", result.Labels["grpo_rollout_prompts"])
	core.AssertEqual(t, "6", result.Labels["grpo_policy_terms"])
}

func TestGRPOPolicyLossPass_ReportsRolloutIdentityIDs_Good(t *testing.T) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}
	dataset, err := LoadJSONLDataset(strings.NewReader(core.Join("\n",
		`{"prompt":"p1","response":"r1","rollout_id":"rollout-a","sample_id":"sample-a","trajectory_id":"trajectory-a","turn_id":1,"completion_id":"completion-a","episode_id":"episode-a","advantages":[1,-1],"current_logprobs":[0,0],"old_logprobs":[0,0]}`,
		`{"prompt":"p2","response":"r2","rollout_id":"rollout-a","sample_id":"sample-b","trajectory_id":"trajectory-a","turn_id":2,"completion_id":"completion-b","episode_id":"episode-a","advantages":[1,-1],"current_logprobs":[0,0],"old_logprobs":[0,0]}`,
		`{"prompt":"p3","response":"r3","rollout_id":"rollout-b","sample_id":"sample-c","trajectory_id":"trajectory-b","turn_id":2,"completion_id":"completion-c","episode_id":"episode-b","advantages":[1,-1],"current_logprobs":[0,0],"old_logprobs":[0,0]}`,
	)))
	core.RequireNoError(t, err)

	result, ok, err := RunNativeGRPOPolicyLossPass(context.Background(), model, dataset, inference.GRPOConfig{})

	core.RequireNoError(t, err)
	core.AssertFalse(t, ok)
	core.AssertEqual(t, "dataset", result.Labels["grpo_rollout_source"])
	core.AssertEqual(t, "2", result.Labels["grpo_rollouts"])
	core.AssertEqual(t, "dataset", result.Labels["grpo_rollout_sample_source"])
	core.AssertEqual(t, "3", result.Labels["grpo_rollout_samples"])
	core.AssertEqual(t, "dataset", result.Labels["grpo_rollout_trajectory_source"])
	core.AssertEqual(t, "2", result.Labels["grpo_rollout_trajectories"])
	core.AssertEqual(t, "dataset", result.Labels["grpo_rollout_turn_source"])
	core.AssertEqual(t, "2", result.Labels["grpo_rollout_turns"])
	core.AssertEqual(t, "dataset", result.Labels["grpo_rollout_completion_source"])
	core.AssertEqual(t, "3", result.Labels["grpo_rollout_completions"])
	core.AssertEqual(t, "dataset", result.Labels["grpo_rollout_episode_source"])
	core.AssertEqual(t, "2", result.Labels["grpo_rollout_episodes"])
	core.AssertEqual(t, "6", result.Labels["grpo_policy_terms"])
}

func TestGRPOPolicyAdamWUpdatePass_ComposesPolicyLossAndOptimizer_Good(t *testing.T) {
	native := &fakeNativeModel{
		grpoKernelOK:  true,
		grpoKernelOut: []float64{-1, 1},
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

	result, ok, err := RunNativeGRPOPolicyAdamWUpdatePass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{
			"rewards":      "1,3",
			"logprobs":     "0,0",
			"old_logprobs": "0,0",
		},
	}}, state, [][]float32{{0.5, -0.25}}, inference.GRPOConfig{
		TrainingConfig: inference.TrainingConfig{Labels: map[string]string{"run": "grpo-policy-adamw"}},
	})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "grpo-policy-adamw", result.Labels["run"])
	core.AssertEqual(t, "grpo_policy_loss_adamw_update_pass", result.Labels["training_stage"])
	core.AssertEqual(t, "policy_loss_plus_optimizer_update", result.Labels["training_interface"])
	core.AssertEqual(t, "applied", result.Labels["training_update_status"])
	core.AssertEqual(t, "adamw", result.Labels["optimizer"])
	core.AssertEqual(t, "1", result.Labels["optimizer_step"])
	core.AssertEqual(t, "2", result.Labels["optimizer_parameters"])
	core.AssertEqual(t, 1, result.Metrics.Step)
	core.AssertEqual(t, 0.1, result.Metrics.LearningRate)
	assertAdamWFloat32Near(t, 0.9, state.Parameters()[0], 0.0001)
	assertAdamWFloat32Near(t, 2.1, state.Parameters()[1], 0.0001)
}

func TestGRPOPolicyAdamWUpdateTrackPass_AppendsOptimizerState_Good(t *testing.T) {
	native := &fakeNativeModel{
		grpoKernelOK:  true,
		grpoKernelOut: []float64{-1, 1},
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
	trackPath := core.PathJoin(t.TempDir(), "training", "grpo-policy-adamw.mp4")

	first, firstRecord, ok, err := RunNativeGRPOPolicyAdamWUpdateTrackPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{
			"rewards":      "1,3",
			"logprobs":     "0,0",
			"old_logprobs": "0,0",
		},
	}}, state, [][]float32{{0.5, -0.25}}, trackPath, inference.GRPOConfig{
		TrainingConfig: inference.TrainingConfig{Labels: map[string]string{"run": "grpo-policy-track"}},
	})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "grpo-policy-track", first.Labels["run"])
	core.AssertEqual(t, "grpo_policy_loss_adamw_update_track_pass", first.Labels["training_stage"])
	core.AssertEqual(t, "append_only", first.Labels["optimizer_track"])
	core.AssertEqual(t, NativeAdamWTrackContainerMP4, first.Labels["optimizer_track_container"])
	core.AssertEqual(t, "rocm_adamw_track_v1", first.Labels["optimizer_track_format"])
	core.AssertEqual(t, "0", first.Labels["optimizer_track_offset"])
	core.AssertEqual(t, "1", first.Labels["optimizer_track_step"])
	core.AssertEqual(t, "1", first.Labels["optimizer_track_frames"])
	core.AssertEqual(t, "LoadNativeAdamWStateTrackStep", first.Labels["optimizer_track_load_step_helper"])
	core.AssertEqual(t, trackPath, first.Labels["optimizer_track_path"])
	core.AssertEqual(t, 0, int(firstRecord.Offset))
	core.AssertEqual(t, 1, firstRecord.Step)

	loaded, loadedRecord, err := LoadLastNativeAdamWStateTrack(trackPath)
	core.RequireNoError(t, err)
	core.AssertEqual(t, firstRecord, loadedRecord)
	core.AssertEqual(t, state.Step, loaded.Step)
	core.AssertEqual(t, state.Parameters(), loaded.Parameters())
}

var (
	rocmGRPOPolicyBenchResult *inference.TrainingResult
	rocmGRPOPolicyBenchOK     bool
	rocmGRPOPolicyBenchErr    error
)

func BenchmarkGRPOPolicyLossPass_RewardRows128(b *testing.B) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}
	samples := makeGRPOPolicyBenchSamples(128, false)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rocmGRPOPolicyBenchResult, rocmGRPOPolicyBenchOK, rocmGRPOPolicyBenchErr = RunNativeGRPOPolicyLossPass(context.Background(), model, &sliceInferenceSamples{samples: samples}, inference.GRPOConfig{KLWeight: 0.1})
		if rocmGRPOPolicyBenchErr != nil {
			b.Fatal(rocmGRPOPolicyBenchErr)
		}
	}
}

func (stream *singleInferenceSample) Remaining() int {
	if stream == nil || stream.done {
		return 0
	}
	return 1
}

func (stream *sliceInferenceSamples) Remaining() int {
	if stream == nil || stream.index >= len(stream.samples) {
		return 0
	}
	return len(stream.samples) - stream.index
}

func BenchmarkGRPOPolicyLossPass_DatasetAdvantages128(b *testing.B) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}
	samples := makeGRPOPolicyBenchSamples(128, true)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rocmGRPOPolicyBenchResult, rocmGRPOPolicyBenchOK, rocmGRPOPolicyBenchErr = RunNativeGRPOPolicyLossPass(context.Background(), model, &sliceInferenceSamples{samples: samples}, inference.GRPOConfig{KLWeight: 0.1})
		if rocmGRPOPolicyBenchErr != nil {
			b.Fatal(rocmGRPOPolicyBenchErr)
		}
	}
}

func BenchmarkGRPOPolicyLossPass_GroupedDatasetAdvantages128(b *testing.B) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}
	samples := makeGRPOPolicyBenchSamples(128, true)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rocmGRPOPolicyBenchResult, rocmGRPOPolicyBenchOK, rocmGRPOPolicyBenchErr = RunNativeGRPOPolicyLossPass(context.Background(), model, &sliceInferenceSamples{samples: samples}, inference.GRPOConfig{
			GroupSize: 2,
			KLWeight:  0.1,
		})
		if rocmGRPOPolicyBenchErr != nil {
			b.Fatal(rocmGRPOPolicyBenchErr)
		}
	}
}

func BenchmarkGRPOPolicyLossPass_ClippedDatasetAdvantages128(b *testing.B) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}
	samples := makeGRPOPolicyBenchSamples(128, true)
	for i := range samples {
		samples[i].Labels["policy_clip_range"] = "0.2"
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rocmGRPOPolicyBenchResult, rocmGRPOPolicyBenchOK, rocmGRPOPolicyBenchErr = RunNativeGRPOPolicyLossPass(context.Background(), model, &sliceInferenceSamples{samples: samples}, inference.GRPOConfig{KLWeight: 0.1})
		if rocmGRPOPolicyBenchErr != nil {
			b.Fatal(rocmGRPOPolicyBenchErr)
		}
	}
}

func BenchmarkGRPOPolicyLossPass_WeightedDatasetAdvantages128(b *testing.B) {
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    &fakeNativeModel{},
	}
	samples := makeGRPOPolicyBenchSamples(128, true)
	for i := range samples {
		samples[i].Labels["policy_weights"] = "1,0"
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rocmGRPOPolicyBenchResult, rocmGRPOPolicyBenchOK, rocmGRPOPolicyBenchErr = RunNativeGRPOPolicyLossPass(context.Background(), model, &sliceInferenceSamples{samples: samples}, inference.GRPOConfig{KLWeight: 0.1})
		if rocmGRPOPolicyBenchErr != nil {
			b.Fatal(rocmGRPOPolicyBenchErr)
		}
	}
}

func makeGRPOPolicyBenchSamples(n int, advantages bool) []inference.DatasetSample {
	samples := make([]inference.DatasetSample, n)
	for i := range samples {
		labels := map[string]string{
			"logprobs":           "0,0",
			"old_logprobs":       "0,0",
			"reference_logprobs": "0,0",
		}
		if advantages {
			labels["advantages"] = "-1,1"
		} else {
			labels["rewards"] = "1,3"
		}
		samples[i] = inference.DatasetSample{Labels: labels}
	}
	return samples
}

func TestGRPOPolicyLossPass_Bad(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{}}

	_, _, err := RunNativeGRPOPolicyLossPass(context.Background(), nil, &singleInferenceSample{}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model is nil")

	_, _, err = RunNativeGRPOPolicyLossPass(context.Background(), model, nil, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dataset is nil")

	_, _, err = RunNativeGRPOPolicyLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"reward": "1", "old_logprob": "0"},
	}}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "parse logprobs")

	_, _, err = RunNativeGRPOPolicyLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"rewards": "1,2", "logprobs": "0", "old_logprobs": "0,0"},
	}}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "count 1 does not match rewards 2")

	_, _, err = RunNativeGRPOPolicyLossPass(context.Background(), model, &sliceInferenceSamples{samples: []inference.DatasetSample{
		{Labels: map[string]string{"advantages": "-1,1", "logprobs": "0,0", "old_logprobs": "0,0"}},
		{Labels: map[string]string{"rewards": "1,3", "logprobs": "0,0", "old_logprobs": "0,0"}},
	}}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "cannot mix dataset advantages")

	_, _, err = RunNativeGRPOPolicyLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"advantages": "-1,1", "logprobs": "0,0", "old_logprobs": "0,0"},
	}}, inference.GRPOConfig{TrainingConfig: inference.TrainingConfig{Labels: map[string]string{"clip_range": "-0.1"}}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "clip range")

	_, _, err = RunNativeGRPOPolicyLossPass(context.Background(), model, &sliceInferenceSamples{samples: []inference.DatasetSample{
		{Labels: map[string]string{"advantages": "1,-1", "logprobs": "0,0", "old_logprobs": "0,0", "policy_clip_range": "0.2"}},
		{Labels: map[string]string{"advantages": "1,-1", "logprobs": "0,0", "old_logprobs": "0,0", "policy_clip_range": "0.3"}},
	}}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "clip range labels conflict")

	_, _, err = RunNativeGRPOPolicyLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"advantages": "1,-1", "logprobs": "0,0", "old_logprobs": "0,0", "policy_clip_range": "0.2"},
	}}, inference.GRPOConfig{TrainingConfig: inference.TrainingConfig{Labels: map[string]string{"clip_range": "0.3"}}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "clip range labels conflict")

	_, _, err = RunNativeGRPOPolicyLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"rewards": "1,2,3", "logprobs": "0,0,0", "old_logprobs": "0,0,0"},
	}}, inference.GRPOConfig{GroupSize: 2})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "divisible by group size")

	_, _, err = RunNativeGRPOPolicyLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"advantages": "1,-1", "logprobs": "0,0", "old_logprobs": "0,0", "policy_weights": "1,-1"},
	}}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "weights must be finite and non-negative")

	_, _, err = RunNativeGRPOPolicyLossPass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"advantages": "1,-1", "logprobs": "0,0", "old_logprobs": "0,0", "policy_weights": "0,0"},
	}}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "weight sum must be positive")

	_, _, err = RunNativeGRPOPolicyLossPass(context.Background(), model, &sliceInferenceSamples{samples: []inference.DatasetSample{
		{Labels: map[string]string{"advantages": "1,-1", "logprobs": "0,0", "old_logprobs": "0,0", "policy_weights": "1,0"}},
		{Labels: map[string]string{"advantages": "1,-1", "logprobs": "0,0", "old_logprobs": "0,0"}},
	}}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "cannot mix weighted and unweighted")

	state, stateErr := NewNativeAdamWState([]NativeAdamWParam{{Values: []float32{1, 2}}}, NativeAdamWConfig{})
	core.RequireNoError(t, stateErr)
	result, ok, err := RunNativeGRPOPolicyAdamWUpdatePass(context.Background(), model, &singleInferenceSample{sample: inference.DatasetSample{
		Labels: map[string]string{"reward": "1", "logprob": "0", "old_logprob": "0"},
	}}, state, [][]float32{{1}}, inference.GRPOConfig{})
	core.AssertError(t, err)
	core.AssertFalse(t, ok)
	core.AssertNotNil(t, result)
	core.AssertContains(t, err.Error(), "does not match")
	core.AssertEqual(t, "grpo_policy_loss_pass", result.Labels["training_stage"])
}

func parseFloat64LabelForTest(t *testing.T, raw string) float64 {
	t.Helper()
	values, err := parseFloat64CSVLabel(raw)
	core.RequireNoError(t, err)
	if len(values) != 1 {
		t.Fatalf("label %q parsed as %+v, want one value", raw, values)
	}
	return values[0]
}
