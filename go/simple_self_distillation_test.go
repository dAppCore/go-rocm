// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/rocm/memorypretrain"
)

func TestSimpleSelfDistillation_Good(t *testing.T) {
	source := &simpleSelfDistillationTestDataset{samples: []inference.DatasetSample{
		{Prompt: "alpha", Labels: map[string]string{"domain": "test"}},
		{Text: "beta"},
		{Prompt: ""},
	}}
	var generatedPrompts []string
	var generateConfigs []inference.GenerateConfig
	result, err := RunSimpleSelfDistillation(context.Background(), SimpleSelfDistillationRunner{
		Generate: func(_ context.Context, prompt string, cfg inference.GenerateConfig) (string, error) {
			generatedPrompts = append(generatedPrompts, prompt)
			generateConfigs = append(generateConfigs, cfg)
			return "out:" + prompt, nil
		},
	}, source, SimpleSelfDistillationConfig{
		SampleMaxTokens:   42,
		SampleTemperature: 0.8,
		SampleTopK:        32,
		SampleTopP:        0.9,
		SampleMinP:        0.05,
		DecodeTemperature: 0.25,
	})
	if err != nil {
		t.Fatalf("RunSimpleSelfDistillation: %v", err)
	}
	core.AssertEqual(t, []string{"alpha", "beta"}, generatedPrompts)
	if len(generateConfigs) != 2 ||
		generateConfigs[0].MaxTokens != 42 ||
		generateConfigs[0].Temperature != 0.8 ||
		generateConfigs[0].TopK != 32 ||
		generateConfigs[0].TopP != 0.9 ||
		generateConfigs[0].MinP != 0.05 {
		t.Fatalf("generate config = %+v, want SSD sample config", generateConfigs)
	}
	sampleCfg := result.SampleGenerateConfig()
	if sampleCfg.MaxTokens != 42 || sampleCfg.Temperature != 0.8 || sampleCfg.TopK != 32 || sampleCfg.TopP != 0.9 || sampleCfg.MinP != 0.05 {
		t.Fatalf("SampleGenerateConfig() = %+v, want SSD sample config", sampleCfg)
	}
	decodeCfg := result.DecodeGenerateConfig(2048)
	if decodeCfg.MaxTokens != 2048 || decodeCfg.Temperature != 0.25 || decodeCfg.TopK != 0 || decodeCfg.TopP != 0 || decodeCfg.MinP != 0 {
		t.Fatalf("DecodeGenerateConfig() = %+v, want decode temperature only", decodeCfg)
	}
	if result.SFT != nil {
		t.Fatalf("result.SFT = %+v, want nil because SSD stops at the trace", result.SFT)
	}
	if len(result.Samples) != 2 || result.Samples[0].Response != "out:alpha" || result.Samples[1].Response != "out:beta" {
		t.Fatalf("samples = %+v, want generated SSD samples", result.Samples)
	}
	if result.Samples[0].Labels["domain"] != "test" ||
		result.Samples[0].Labels["ssd"] != "simple_self_distillation" ||
		result.Samples[0].Labels["ssd_source_index"] != "0" ||
		result.Samples[0].Labels["ssd_sample_temperature"] != "0.8" ||
		result.Samples[1].Labels["ssd_source_index"] != "1" {
		t.Fatalf("trace samples = %+v, want labelled generated trace", result.Samples)
	}
}

func TestSimpleSelfDistillationResult_GenerateConfigs_Good(t *testing.T) {
	result := &SimpleSelfDistillationResult{
		SampleMaxTokens:   128,
		SampleTemperature: 0.6,
		SampleTopK:        48,
		SampleTopP:        0.92,
		SampleMinP:        0.04,
		DecodeTemperature: 0.15,
	}

	sample := result.SampleGenerateConfig()
	if sample.MaxTokens != 128 || sample.Temperature != 0.6 || sample.TopK != 48 || sample.TopP != 0.92 || sample.MinP != 0.04 {
		t.Fatalf("SampleGenerateConfig() = %+v", sample)
	}
	decode := result.DecodeGenerateConfig(2048)
	if decode.MaxTokens != 2048 || decode.Temperature != 0.15 || decode.TopK != 0 || decode.TopP != 0 || decode.MinP != 0 {
		t.Fatalf("DecodeGenerateConfig() = %+v", decode)
	}

	var nilResult *SimpleSelfDistillationResult
	if got := nilResult.SampleGenerateConfig(); got.MaxTokens != 0 || got.Temperature != 0 || got.TopK != 0 || got.TopP != 0 || got.MinP != 0 {
		t.Fatalf("nil SampleGenerateConfig() = %+v", got)
	}
	if got := nilResult.DecodeGenerateConfig(64); got.MaxTokens != 64 || got.Temperature != 0 {
		t.Fatalf("nil DecodeGenerateConfig() = %+v", got)
	}
}

func TestSimpleSelfDistillationEvalGenerateConfig_Good(t *testing.T) {
	cfg, ok, err := SimpleSelfDistillationEvalGenerateConfig(map[string]string{"ssd_eval_temperature": "0.35"}, 512)
	if err != nil || !ok || cfg.MaxTokens != 512 || cfg.Temperature != 0.35 {
		t.Fatalf("SimpleSelfDistillationEvalGenerateConfig(eval) = %+v ok=%v err=%v", cfg, ok, err)
	}
	cfg, ok, err = SimpleSelfDistillationEvalGenerateConfig(map[string]string{"ssd_decode_temperature": "0.2"}, 256)
	if err != nil || !ok || cfg.MaxTokens != 256 || cfg.Temperature != 0.2 {
		t.Fatalf("SimpleSelfDistillationEvalGenerateConfig(decode fallback) = %+v ok=%v err=%v", cfg, ok, err)
	}
	cfg, ok, err = SimpleSelfDistillationEvalGenerateConfig(nil, 128)
	if err != nil || ok || cfg.MaxTokens != 128 || cfg.Temperature != 0 {
		t.Fatalf("SimpleSelfDistillationEvalGenerateConfig(empty) = %+v ok=%v err=%v", cfg, ok, err)
	}
}

func TestSimpleSelfDistillationEvalGenerateConfig_Bad(t *testing.T) {
	_, _, err := SimpleSelfDistillationEvalGenerateConfig(map[string]string{"ssd_eval_temperature": "banana"}, 512)
	if err == nil || !core.Contains(err.Error(), "non-negative and finite") {
		t.Fatalf("SimpleSelfDistillationEvalGenerateConfig(malformed) error = %v", err)
	}
	_, _, err = SimpleSelfDistillationEvalGenerateConfig(map[string]string{"ssd_eval_temperature": "-0.1"}, 512)
	if err == nil || !core.Contains(err.Error(), "non-negative and finite") {
		t.Fatalf("SimpleSelfDistillationEvalGenerateConfig(negative) error = %v", err)
	}
}

func TestSimpleSelfDistillation_Defaults_Good(t *testing.T) {
	var gotCfg inference.GenerateConfig
	_, err := RunSimpleSelfDistillation(context.Background(), SimpleSelfDistillationRunner{
		Generate: func(_ context.Context, _ string, cfg inference.GenerateConfig) (string, error) {
			gotCfg = cfg
			return "answer", nil
		},
	}, &simpleSelfDistillationTestDataset{samples: []inference.DatasetSample{{Prompt: "p"}}}, SimpleSelfDistillationConfig{})
	if err != nil {
		t.Fatalf("RunSimpleSelfDistillation: %v", err)
	}
	if gotCfg.MaxTokens != defaultSimpleSelfDistillationMaxTokens ||
		gotCfg.Temperature != defaultSimpleSelfDistillationTemperature ||
		gotCfg.TopK != defaultSimpleSelfDistillationTopK ||
		gotCfg.TopP != defaultSimpleSelfDistillationTopP ||
		gotCfg.MinP != 0 ||
		gotCfg.RepeatPenalty != 0 {
		t.Fatalf("default generate config = %+v", gotCfg)
	}
}

func TestSimpleSelfDistillation_ManifestDefaultsAndLiveCodeBenchPlan_Good(t *testing.T) {
	train := DefaultSimpleSelfDistillationConfig()
	if train.SampleMaxTokens != 65536 ||
		train.SampleTemperature != 1.5 ||
		train.SampleTopK != 20 ||
		train.SampleTopP != 0.8 ||
		train.RepetitionPenalty != 1 ||
		train.FilterShortestPct != 10 {
		t.Fatalf("DefaultSimpleSelfDistillationConfig() = %+v, want ml-ssd defaults", train)
	}
	eval := DefaultSimpleSelfDistillationCodeBenchmarkConfig()
	if eval.Benchmark != "LiveCodeBench-v6" ||
		eval.NRepeat != 20 ||
		eval.Generate.MaxTokens != 32768 ||
		eval.Generate.Temperature != 0.6 ||
		eval.Generate.TopP != 0.95 ||
		eval.Generate.TopK != 20 ||
		len(eval.Seeds) != 4 {
		t.Fatalf("DefaultSimpleSelfDistillationCodeBenchmarkConfig() = %+v, want LiveCodeBench-v6 defaults", eval)
	}
	recipes := SimpleSelfDistillationRecipes()
	if len(recipes) != 3 {
		t.Fatalf("SimpleSelfDistillationRecipes() = %d, want released recipes", len(recipes))
	}
	recipe, ok := LookupSimpleSelfDistillationRecipe("apple/SimpleSD-4B-thinking")
	if !ok || recipe.Name != SimpleSelfDistillationRecipe4BThinking || recipe.Dataset != "microsoft/rStar-Coder" || recipe.DatasetConfig != "seed_sft" {
		t.Fatalf("LookupSimpleSelfDistillationRecipe() = %+v/%t", recipe, ok)
	}

	raw := []byte(`{"id":"old","prompt":"old","contest_date":"2025-01-01"}` + "\n" +
		`{"question_id":"new","question_content":"solve me","starter_code":"def f(): pass","entry_point":"f","is_stdin":false,"contest_date":"2025-03-02","public_test_cases":["assert f()==1"],"difficulty":"easy","platform":"lcb"}`)
	samples, err := LoadSimpleSelfDistillationLiveCodeBenchV6JSONL(raw)
	if err != nil {
		t.Fatalf("LoadSimpleSelfDistillationLiveCodeBenchV6JSONL: %v", err)
	}
	if len(samples) != 1 || samples[0].ID != "new" || !core.Contains(samples[0].Prompt, "starter code:") || samples[0].Meta["is_stdin"] != "false" || samples[0].Meta["entry_point"] != "f" {
		t.Fatalf("LiveCodeBench samples = %+v, want parsed v6 sample", samples)
	}
}

func TestSimpleSelfDistillation_TraceKeepsShortestSamples_Good(t *testing.T) {
	result, err := RunSimpleSelfDistillation(context.Background(), SimpleSelfDistillationRunner{
		Generate: func(_ context.Context, prompt string, _ inference.GenerateConfig) (string, error) {
			if prompt == "short" {
				return "x", nil
			}
			return "long answer", nil
		},
	}, &simpleSelfDistillationTestDataset{samples: []inference.DatasetSample{{Prompt: "short"}, {Prompt: "long"}}}, SimpleSelfDistillationConfig{
		SampleMaxTokens:   2,
		SampleTemperature: 0.7,
		FilterShortestPct: 50,
	})
	if err != nil {
		t.Fatalf("RunSimpleSelfDistillation: %v", err)
	}
	if len(result.Samples) != 2 || result.Samples[0].Prompt != "short" || result.Samples[1].Prompt != "long" {
		t.Fatalf("trace rows = %+v, want every sampled return", result.Samples)
	}
}

func TestSimpleSelfDistillation_ModelGemma4UsesRemainingSampleMaxTokens(t *testing.T) {
	native := &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "answer"}}}
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native:    native,
	}

	result, err := RunModelSimpleSelfDistillation(context.Background(), model, &simpleSelfDistillationTestDataset{
		samples: []inference.DatasetSample{{Prompt: "one two three"}},
	}, SimpleSelfDistillationConfig{})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, result.SampleMaxTokens)
	core.AssertEqual(t, defaultContextLengthCap-3, native.generateConfigs[0].MaxTokens)
}

func TestSimpleSelfDistillation_ModelNonGemmaKeepsDefaultSampleMaxTokens(t *testing.T) {
	native := &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "answer"}}}
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    native,
	}

	result, err := RunModelSimpleSelfDistillation(context.Background(), model, &simpleSelfDistillationTestDataset{
		samples: []inference.DatasetSample{{Prompt: "hello"}},
	}, SimpleSelfDistillationConfig{})

	core.RequireNoError(t, err)
	core.AssertEqual(t, defaultSimpleSelfDistillationMaxTokens, result.SampleMaxTokens)
	core.AssertEqual(t, defaultSimpleSelfDistillationMaxTokens, native.generateConfigs[0].MaxTokens)
}

func TestSimpleSelfDistillation_ModelGemma4RejectsNegativeSampleMaxTokens(t *testing.T) {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native:    &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "answer"}}},
	}

	_, err := RunModelSimpleSelfDistillation(context.Background(), model, &simpleSelfDistillationTestDataset{
		samples: []inference.DatasetSample{{Prompt: "hello"}},
	}, SimpleSelfDistillationConfig{SampleMaxTokens: -1})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "non-negative")
}

func TestSimpleSelfDistillation_NativeAdamWUpdatePass_Good(t *testing.T) {
	native := &fakeNativeModel{
		tokens:            []inference.Token{{ID: 1, Text: "answer"}},
		classLogits:       [][]float32{{0, 3}},
		evalLossKernelOK:  true,
		evalLossKernelOut: hipCrossEntropyLossResult{Loss: 0.25, Perplexity: 1.284025},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "lora.w", Shape: []int{2}, Values: []float32{1, 2}},
	}, NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true})
	core.RequireNoError(t, err)

	result, ok, err := RunModelNativeSimpleSelfDistillationAdamWUpdatePass(context.Background(), model, &simpleSelfDistillationTestDataset{
		samples: []inference.DatasetSample{{
			Prompt: "hello",
			Labels: map[string]string{"target_token_id": "1"},
		}},
	}, NativeSimpleSelfDistillationAdamWConfig{
		SSD: SimpleSelfDistillationConfig{
			SampleMaxTokens:   2,
			SampleTemperature: 0.7,
			SFT: inference.TrainingConfig{
				BatchSize: 1,
				Labels:    map[string]string{"run": "ssd-native-adamw"},
			},
		},
		State:     state,
		Gradients: [][]float32{{0.5, -0.25}},
	})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertNotNil(t, result)
	core.AssertNotNil(t, result.SFT)
	core.AssertEqual(t, 1, result.SFT.Metrics.Samples)
	core.AssertEqual(t, 1, result.SFT.Metrics.Step)
	assertFloat64Near(t, 0.25, result.SFT.Metrics.Loss, 0.0001)
	assertAdamWFloat32Near(t, 0.9, state.Parameters()[0], 0.0001)
	assertAdamWFloat32Near(t, 2.1, state.Parameters()[1], 0.0001)
	core.AssertEqual(t, "ssd-native-adamw", result.SFT.Labels["run"])
	core.AssertEqual(t, "sft_loss_adamw_update_pass", result.SFT.Labels["training_stage"])
	core.AssertEqual(t, "loss_plus_optimizer_update", result.SFT.Labels["training_interface"])
	core.AssertEqual(t, "applied", result.SFT.Labels["training_update_status"])
	core.AssertEqual(t, "not_implemented", result.SFT.Labels["trainer_interface"])
	core.AssertEqual(t, "true", result.SFT.Labels["loss_native_ready"])
	core.AssertEqual(t, "hip", result.SFT.Labels["loss_backend"])
	core.AssertEqual(t, "adamw", result.SFT.Labels["optimizer"])
	core.AssertEqual(t, "answer", result.Samples[0].Response)
	core.AssertEqual(t, "1", result.Samples[0].Labels["target_token_id"])
	core.AssertEqual(t, "simple_self_distillation", result.Samples[0].Labels["ssd"])
	core.AssertEqual(t, "hello", native.generatePrompts[0])
	core.AssertEqual(t, 2, native.generateConfigs[0].MaxTokens)
	core.AssertEqual(t, float32(0.7), native.generateConfigs[0].Temperature)
	core.AssertEqual(t, 1, native.evalLossKernelCalls)
}

func TestSimpleSelfDistillation_NativeAdamWUpdateTrackPass_Good(t *testing.T) {
	native := &fakeNativeModel{
		tokens:            []inference.Token{{ID: 1, Text: "answer"}},
		classLogits:       [][]float32{{0, 3}},
		evalLossKernelOK:  true,
		evalLossKernelOut: hipCrossEntropyLossResult{Loss: 0.25, Perplexity: 1.284025},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "lora.w", Shape: []int{2}, Values: []float32{1, 2}},
	}, NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true})
	core.RequireNoError(t, err)
	trackPath := core.PathJoin(t.TempDir(), "training", "ssd-sft-adamw.kv")

	result, ok, err := RunModelNativeSimpleSelfDistillationAdamWUpdatePass(context.Background(), model, &simpleSelfDistillationTestDataset{
		samples: []inference.DatasetSample{{
			Prompt: "hello",
			Labels: map[string]string{"target_token_id": "1"},
		}},
	}, NativeSimpleSelfDistillationAdamWConfig{
		SSD: SimpleSelfDistillationConfig{
			SampleMaxTokens:   2,
			SampleTemperature: 0.7,
			SFT: inference.TrainingConfig{
				BatchSize: 1,
				Labels:    map[string]string{"run": "ssd-native-track"},
			},
		},
		State:     state,
		Gradients: [][]float32{{0.5, -0.25}},
		TrackPath: trackPath,
	})

	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertNotNil(t, result)
	core.AssertNotNil(t, result.SFT)
	core.AssertEqual(t, "ssd-native-track", result.SFT.Labels["run"])
	core.AssertEqual(t, "sft_loss_adamw_update_track_pass", result.SFT.Labels["training_stage"])
	core.AssertEqual(t, "append_only", result.SFT.Labels["optimizer_track"])
	core.AssertEqual(t, NativeAdamWTrackContainerKV, result.SFT.Labels["optimizer_track_container"])
	core.AssertEqual(t, "rocm_adamw_track_v1", result.SFT.Labels["optimizer_track_format"])
	core.AssertEqual(t, "0", result.SFT.Labels["optimizer_track_offset"])
	core.AssertEqual(t, "1", result.SFT.Labels["optimizer_track_step"])
	core.AssertEqual(t, trackPath, result.SFT.Labels["optimizer_track_path"])
	core.AssertEqual(t, "answer", result.Samples[0].Response)

	loaded, record, err := LoadLastNativeAdamWStateTrack(trackPath)
	core.RequireNoError(t, err)
	core.AssertEqual(t, int64(0), record.Offset)
	core.AssertEqual(t, 1, record.Step)
	core.AssertEqual(t, state.Step, loaded.Step)
	core.AssertEqual(t, state.Parameters(), loaded.Parameters())
}

func TestSimpleSelfDistillation_NativeAdamWUpdatePass_Bad(t *testing.T) {
	state, err := NewNativeAdamWState([]NativeAdamWParam{{Values: []float32{1, 2}}}, NativeAdamWConfig{})
	core.RequireNoError(t, err)
	dataset := &simpleSelfDistillationTestDataset{samples: []inference.DatasetSample{{Prompt: "hello", Labels: map[string]string{"target_token_id": "0"}}}}

	_, _, err = RunModelNativeSimpleSelfDistillationAdamWUpdatePass(context.Background(), nil, dataset, NativeSimpleSelfDistillationAdamWConfig{State: state})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model is nil")

	model := &rocmModel{native: &fakeNativeModel{tokens: []inference.Token{{Text: "answer"}}}}
	_, _, err = RunModelNativeSimpleSelfDistillationAdamWUpdatePass(context.Background(), model, dataset, NativeSimpleSelfDistillationAdamWConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "state is nil")
}

func TestSimpleSelfDistillation_NativeAdamWUpdatePass_PreservesLossOnUpdateError(t *testing.T) {
	native := &fakeNativeModel{
		tokens:            []inference.Token{{ID: 1, Text: "answer"}},
		classLogits:       [][]float32{{3, 0}},
		evalLossKernelOK:  true,
		evalLossKernelOut: hipCrossEntropyLossResult{Loss: 0.1, Perplexity: 1.105171},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}
	state, err := NewNativeAdamWState([]NativeAdamWParam{{Values: []float32{1, 2}}}, NativeAdamWConfig{})
	core.RequireNoError(t, err)

	result, ok, err := RunModelNativeSimpleSelfDistillationAdamWUpdatePass(context.Background(), model, &simpleSelfDistillationTestDataset{
		samples: []inference.DatasetSample{{Prompt: "hello", Labels: map[string]string{"target_token_id": "0"}}},
	}, NativeSimpleSelfDistillationAdamWConfig{
		SSD:       SimpleSelfDistillationConfig{SampleMaxTokens: 2, SampleTemperature: 0.7},
		State:     state,
		Gradients: [][]float32{{1}},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "does not match")
	core.AssertTrue(t, ok)
	core.AssertNotNil(t, result)
	core.AssertNotNil(t, result.SFT)
	core.AssertEqual(t, "sft_loss_pass", result.SFT.Labels["training_stage"])
	core.AssertEqual(t, "not_applied", result.SFT.Labels["training_update_status"])
	core.AssertEqual(t, "answer", result.Samples[0].Response)
}

func TestSimpleSelfDistillation_MemoryPretraining_Good(t *testing.T) {
	native := &fakeNativeModel{
		tokens:            []inference.Token{{ID: 1, Text: "answer"}},
		classLogits:       [][]float32{{0, 3}},
		evalLossKernelOK:  true,
		evalLossKernelOut: hipCrossEntropyLossResult{Loss: 0.25, Perplexity: 1.284025},
	}
	model := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native:    native,
	}
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "lora.w", Shape: []int{2}, Values: []float32{1, 2}},
	}, NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true})
	core.RequireNoError(t, err)
	path := core.PathJoin(t.TempDir(), "ssd", "memory-bank.json")
	trackPath := core.PathJoin(t.TempDir(), "ssd", "memory-adamw.kv")
	var embedded []string

	result, err := RunModelNativeSimpleSelfDistillationMemoryPretraining(context.Background(), model, &simpleSelfDistillationTestDataset{
		samples: []inference.DatasetSample{{
			Prompt: "hello",
			Labels: map[string]string{"target_token_id": "1", "domain": "unit"},
		}},
	}, NativeSimpleSelfDistillationMemoryPretrainingConfig{
		SSDAdamW: NativeSimpleSelfDistillationAdamWConfig{
			SSD: SimpleSelfDistillationConfig{
				SampleMaxTokens:   2,
				SampleTemperature: 0.7,
				SFT: inference.TrainingConfig{
					BatchSize: 1,
					Labels:    map[string]string{"run": "ssd-memory-pretrain"},
				},
			},
			State:     state,
			Gradients: [][]float32{{0.5, -0.25}},
			TrackPath: trackPath,
		},
		Embedder: memorypretrain.EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
			embedded = append(embedded, text)
			return []float32{1, 0}, nil
		}),
		Bank:     memorypretrain.BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2},
		BankPath: path,
	})

	core.RequireNoError(t, err)
	core.AssertNotNil(t, result)
	core.AssertNotNil(t, result.SSD)
	core.AssertNotNil(t, result.Bank)
	core.AssertTrue(t, result.NativeLoss)
	core.AssertEqual(t, 1, result.Records)
	core.AssertEqual(t, []string{"hello\nanswer"}, embedded)
	core.AssertEqual(t, "ssd_sft_adamw_memory_bank_build", result.Labels["memory_pretraining_stage"])
	core.AssertEqual(t, "hierarchical", result.Labels["memory_pretraining"])
	core.AssertEqual(t, "1", result.Labels["memory_pretraining_bank_records"])
	core.AssertEqual(t, "2", result.Labels["memory_pretraining_bank_dimension"])
	core.AssertEqual(t, "pending", result.Labels["memory_pretraining_hip_injection"])
	core.AssertEqual(t, "not_implemented", result.Labels["trainer_interface"])
	core.AssertEqual(t, "true", result.Labels["ssd_native_loss_ready"])
	core.AssertEqual(t, path, result.Labels["memory_pretraining_bank_file"])
	core.AssertEqual(t, "append_only", result.Labels["memory_pretraining_optimizer_track"])
	core.AssertEqual(t, NativeAdamWTrackContainerKV, result.Labels["memory_pretraining_optimizer_track_container"])
	core.AssertEqual(t, "rocm_adamw_track_v1", result.Labels["memory_pretraining_optimizer_track_format"])
	core.AssertEqual(t, "0", result.Labels["memory_pretraining_optimizer_track_offset"])
	core.AssertEqual(t, "1", result.Labels["memory_pretraining_optimizer_track_step"])
	core.AssertEqual(t, "1", result.Labels["memory_pretraining_optimizer_track_frames"])
	core.AssertEqual(t, "ListNativeAdamWStateTrack", result.Labels["memory_pretraining_optimizer_track_list_helper"])
	core.AssertEqual(t, "FindNativeAdamWStateTrackStep", result.Labels["memory_pretraining_optimizer_track_find_helper"])
	core.AssertEqual(t, "LoadNativeAdamWStateTrackStep", result.Labels["memory_pretraining_optimizer_track_load_step_helper"])
	core.AssertEqual(t, trackPath, result.Labels["memory_pretraining_optimizer_track_path"])
	core.AssertEqual(t, "unit", result.Bank.Blocks[0].Meta["domain"])
	core.AssertEqual(t, "simple_self_distillation", result.Bank.Blocks[0].Meta["memory_pretraining_source"])
	core.AssertEqual(t, "0", result.Bank.Blocks[0].Meta["memory_pretraining_source_index"])

	loadedTrack, trackRecord, err := LoadLastNativeAdamWStateTrack(trackPath)
	core.RequireNoError(t, err)
	core.AssertEqual(t, int64(0), trackRecord.Offset)
	core.AssertEqual(t, 1, trackRecord.Step)
	core.AssertEqual(t, state.Parameters(), loadedTrack.Parameters())

	loaded, err := memorypretrain.LoadBank(path)
	core.RequireNoError(t, err)
	retrieved, err := loaded.Retrieve([]float32{1, 0}, 1)
	core.RequireNoError(t, err)
	if len(retrieved) != 1 || retrieved[0].BlockID != "ssd-0" || retrieved[0].Text != "hello\nanswer" {
		t.Fatalf("loaded Retrieve() = %+v, want saved SSD memory record", retrieved)
	}
}

func TestSimpleSelfDistillation_MemoryPretraining_Bad(t *testing.T) {
	state, err := NewNativeAdamWState([]NativeAdamWParam{{Values: []float32{1, 2}}}, NativeAdamWConfig{})
	core.RequireNoError(t, err)
	model := &rocmModel{native: &fakeNativeModel{tokens: []inference.Token{{Text: "answer"}}}}
	dataset := &simpleSelfDistillationTestDataset{samples: []inference.DatasetSample{{Prompt: "hello", Labels: map[string]string{"target_token_id": "0"}}}}

	_, err = RunModelNativeSimpleSelfDistillationMemoryPretraining(context.Background(), model, dataset, NativeSimpleSelfDistillationMemoryPretrainingConfig{
		SSDAdamW: NativeSimpleSelfDistillationAdamWConfig{State: state},
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "embedder is nil")

	result, err := RunModelNativeSimpleSelfDistillationMemoryPretraining(context.Background(), model, dataset, NativeSimpleSelfDistillationMemoryPretrainingConfig{
		SSDAdamW: NativeSimpleSelfDistillationAdamWConfig{
			SSD:       SimpleSelfDistillationConfig{SampleMaxTokens: 2, SampleTemperature: 0.7},
			State:     state,
			Gradients: [][]float32{{1}},
		},
		Embedder: memorypretrain.EmbedFunc(func(context.Context, string) ([]float32, error) {
			return []float32{1}, nil
		}),
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "does not match")
	core.AssertNotNil(t, result)
	core.AssertNotNil(t, result.SSD)
	core.AssertEqual(t, "sft_loss_pass", result.SSD.SFT.Labels["training_stage"])
}

func TestSimpleSelfDistillation_Bad(t *testing.T) {
	cases := []struct {
		name    string
		runner  SimpleSelfDistillationRunner
		dataset inference.DatasetStream
		cfg     SimpleSelfDistillationConfig
		want    string
	}{
		{name: "nil-dataset", runner: simpleSelfDistillationValidRunner(), want: "dataset is nil"},
		{name: "nil-generate", runner: SimpleSelfDistillationRunner{}, dataset: &simpleSelfDistillationTestDataset{}, want: "generate function is nil"},
		{name: "unit-temperature", runner: simpleSelfDistillationValidRunner(), dataset: &simpleSelfDistillationTestDataset{}, cfg: SimpleSelfDistillationConfig{SampleTemperature: 1}, want: "non-unit"},
		{name: "nan-temperature", runner: simpleSelfDistillationValidRunner(), dataset: &simpleSelfDistillationTestDataset{}, cfg: SimpleSelfDistillationConfig{SampleTemperature: float32(math.NaN())}, want: "positive and finite"},
		{name: "empty-prompts", runner: simpleSelfDistillationValidRunner(), dataset: &simpleSelfDistillationTestDataset{samples: []inference.DatasetSample{{}}}, want: "produced no prompts"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := RunSimpleSelfDistillation(context.Background(), tc.runner, tc.dataset, tc.cfg)
			if err == nil || !core.Contains(err.Error(), tc.want) {
				t.Fatalf("RunSimpleSelfDistillation error = %v, want %q", err, tc.want)
			}
		})
	}
}

func simpleSelfDistillationValidRunner() SimpleSelfDistillationRunner {
	return SimpleSelfDistillationRunner{
		Generate: func(context.Context, string, inference.GenerateConfig) (string, error) {
			return "ok", nil
		},
	}
}

type simpleSelfDistillationTestDataset struct {
	samples []inference.DatasetSample
	index   int
}

func (dataset *simpleSelfDistillationTestDataset) Next() (inference.DatasetSample, bool, error) {
	if dataset == nil || dataset.index >= len(dataset.samples) {
		return inference.DatasetSample{}, false, nil
	}
	sample := dataset.samples[dataset.index]
	dataset.index++
	return sample, true, nil
}
