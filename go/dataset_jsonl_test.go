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

func TestLoadJSONLDataset_RecognizesTrainingFormats_Good(t *testing.T) {
	input := core.Join("\n",
		`{"text":"plain corpus row","target_token_id":7}`,
		`{"prompt":"p","response":"r","labels":{"source":"pair"}}`,
		`{"instruction":"summarise","input":"lem notes","output":"short answer"}`,
		`{"messages":[{"role":"system","content":"steady"},{"role":"user","content":"ping"},{"role":"assistant","content":"pong"}],"student_logits":[1,0],"teacher_logits":[2,0]}`,
		`{"conversations":[{"from":"human","value":"hi"},{"from":"gpt","value":"there"}]}`,
		`{"problem":"2+2","thinking":"add the pair","solution":"4"}`,
	)
	dataset, err := LoadJSONLDataset(strings.NewReader(input))
	core.RequireNoError(t, err)

	samples := collectJSONLDatasetSamples(t, dataset)
	core.AssertEqual(t, 6, len(samples))
	core.AssertEqual(t, "plain corpus row", samples[0].Text)
	core.AssertEqual(t, "text", samples[0].Labels["format"])
	core.AssertEqual(t, "7", samples[0].Labels["target_token_id"])
	core.AssertEqual(t, "p", samples[1].Prompt)
	core.AssertEqual(t, "r", samples[1].Response)
	core.AssertEqual(t, "pair", samples[1].Labels["source"])
	core.AssertContains(t, samples[2].Prompt, "summarise")
	core.AssertContains(t, samples[2].Prompt, "lem notes")
	core.AssertEqual(t, "short answer", samples[2].Response)
	if len(samples[3].Messages) != 2 || samples[3].Messages[0].Role != "system" || samples[3].Messages[1].Role != "user" || samples[3].Response != "pong" {
		t.Fatalf("messages sample = %+v, want prompt messages plus assistant response", samples[3])
	}
	core.AssertEqual(t, "1,0", samples[3].Labels["student_logits"])
	core.AssertEqual(t, "2,0", samples[3].Labels["teacher_logits"])
	if len(samples[4].Messages) != 1 || samples[4].Messages[0].Role != "user" || samples[4].Response != "there" {
		t.Fatalf("sharegpt sample = %+v, want user prompt plus assistant response", samples[4])
	}
	core.AssertEqual(t, "2+2", samples[5].Prompt)
	core.AssertContains(t, samples[5].Response, "add the pair")
	core.AssertContains(t, samples[5].Response, "4")

	core.RequireNoError(t, dataset.Reset())
	again, ok, err := dataset.Next()
	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "plain corpus row", again.Text)
}

func TestJSONLDataset_ClonesSamples_Good(t *testing.T) {
	input := []inference.DatasetSample{{
		Text:     "a",
		Messages: []inference.Message{{Role: "user", Content: "hi"}},
		Labels:   map[string]string{"k": "v"},
	}}
	dataset := NewJSONLDataset(input)
	input[0].Text = "mutated"
	input[0].Messages[0].Content = "mutated"
	input[0].Labels["k"] = "mutated"

	got, ok, err := dataset.Next()
	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "a", got.Text)
	core.AssertEqual(t, "hi", got.Messages[0].Content)
	core.AssertEqual(t, "v", got.Labels["k"])

	got.Labels["k"] = "changed"
	got.Messages[0].Content = "changed"
	samples := dataset.Samples()
	core.AssertEqual(t, "v", samples[0].Labels["k"])
	core.AssertEqual(t, "hi", samples[0].Messages[0].Content)
}

func TestLoadJSONLDataset_PreservesRolloutMetadataLabels_Good(t *testing.T) {
	dataset, err := LoadJSONLDataset(strings.NewReader(`{"prompt":"p","response":"r","group_id":7,"prompt_id":"p1","query_id":"q1","rollout_id":"r1","sample_id":"s1","trajectory_id":"t1","turn_id":2,"completion_id":"c1","episode_id":"e1","rewards":[1,3],"policy_logprobs":[0,0],"old_policy_logprobs":[0,0],"clip_epsilon":0.2,"response_mask":[1,0]}`))
	core.RequireNoError(t, err)

	sample, ok, err := dataset.Next()
	core.RequireNoError(t, err)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "7", sample.Labels["group_id"])
	core.AssertEqual(t, "p1", sample.Labels["prompt_id"])
	core.AssertEqual(t, "q1", sample.Labels["query_id"])
	core.AssertEqual(t, "r1", sample.Labels["rollout_id"])
	core.AssertEqual(t, "s1", sample.Labels["sample_id"])
	core.AssertEqual(t, "t1", sample.Labels["trajectory_id"])
	core.AssertEqual(t, "2", sample.Labels["turn_id"])
	core.AssertEqual(t, "c1", sample.Labels["completion_id"])
	core.AssertEqual(t, "e1", sample.Labels["episode_id"])
	core.AssertEqual(t, "1,3", sample.Labels["rewards"])
	core.AssertEqual(t, "0,0", sample.Labels["logprobs"])
	core.AssertEqual(t, "0,0", sample.Labels["old_logprobs"])
	core.AssertEqual(t, "0.2", sample.Labels["policy_clip_range"])
	core.AssertEqual(t, "1,0", sample.Labels["policy_weight"])
}

func TestJSONLDataset_FeedsNativeLossPasses_Good(t *testing.T) {
	sftModel := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native: &fakeNativeModel{
			classLogits:       [][]float32{{0, 3}},
			evalLossKernelOK:  true,
			evalLossKernelOut: hipCrossEntropyLossResult{Loss: 0.25, Perplexity: 1.284025},
		},
	}
	sftDataset, err := LoadJSONLDataset(strings.NewReader(`{"prompt":"hello","response":"world","target_token_id":1}`))
	core.RequireNoError(t, err)
	sft, sftOK, err := RunNativeSFTLossPass(context.Background(), sftModel, sftDataset, inference.TrainingConfig{})
	core.RequireNoError(t, err)
	core.AssertTrue(t, sftOK)
	assertFloat64Near(t, 0.25, sft.Metrics.Loss, 0.0001)

	distillModel := &rocmModel{
		modelType: "tiny",
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 2, HiddenSize: 2},
		native: &fakeNativeModel{
			distillKernelOK:  true,
			distillKernelOut: hipDistillationKLLossResult{KL: 0.0671},
		},
	}
	distillDataset, err := LoadJSONLDataset(strings.NewReader(`{"text":"logit row","student_logits":[1,0],"teacher_logits":[2,0]}`))
	core.RequireNoError(t, err)
	distill, distillOK, err := RunNativeDistillationLossPass(context.Background(), distillModel, distillDataset, inference.DistillConfig{})
	core.RequireNoError(t, err)
	core.AssertTrue(t, distillOK)
	assertFloat64Near(t, 0.0671, distill.Metrics.Loss, 0.0001)
}

func TestLoadJSONLDataset_Bad(t *testing.T) {
	_, err := LoadJSONLDataset(nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "reader is nil")

	_, err = LoadJSONLDataset(strings.NewReader("{not-json}\n"))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "parse JSONL record 1")

	var dataset *JSONLDataset
	_, _, err = dataset.Next()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dataset is nil")
	err = dataset.Reset()
	core.AssertError(t, err)
}

func collectJSONLDatasetSamples(t *testing.T, dataset inference.DatasetStream) []inference.DatasetSample {
	t.Helper()
	var samples []inference.DatasetSample
	for {
		sample, ok, err := dataset.Next()
		core.RequireNoError(t, err)
		if !ok {
			return samples
		}
		samples = append(samples, sample)
	}
}
