// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	"dappco.re/go/inference"
)

type testModelInfoReporter struct {
	info inference.ModelInfo
}

func (reporter testModelInfoReporter) FillModelInfo(info *inference.ModelInfo) {
	*info = reporter.info
}

func TestResolveModelInfo_Good_DispatchesReporterCapability(t *testing.T) {
	report := ResolveModelInfo(ModelInfoRequest{
		Path:      "/models/gemma4",
		ModelType: "fallback",
		Info:      inference.ModelInfo{Architecture: "llama", VocabSize: 7},
		Labels: map[string]string{
			"caller":     "kept",
			"quant_type": "q6",
		},
		Reporter: testModelInfoReporter{info: inference.ModelInfo{
			Architecture: "Gemma4ForCausalLM",
			VocabSize:    42,
			NumLayers:    3,
			HiddenSize:   16,
			QuantBits:    6,
			QuantGroup:   32,
		}},
	})
	if !report.Matched() ||
		report.Contract != ModelInfoReporterContract ||
		report.Source != "model_info_reporter" ||
		report.Info.Architecture != "gemma4_text" ||
		report.Info.VocabSize != 42 ||
		report.Info.NumLayers != 3 ||
		report.Identity.Path != "/models/gemma4" ||
		report.Identity.QuantType != "q6" ||
		report.Labels["engine_model_info_source"] != "model_info_reporter" ||
		report.Labels["engine_model_info_reactive"] != "true" ||
		report.Labels["engine_model_info_architecture"] != "gemma4_text" ||
		report.Labels["caller"] != "kept" {
		t.Fatalf("ResolveModelInfo = %+v, want reporter-filled Gemma4 info", report)
	}
}

func TestResolveModelInfo_Good_ResolvedArchitectureLabelWins(t *testing.T) {
	report := ResolveModelInfo(ModelInfoRequest{
		Path: "/models/wrapped",
		Info: inference.ModelInfo{Architecture: "wrapped", VocabSize: 11},
		Identity: inference.ModelIdentity{
			Architecture: "raw",
			NumLayers:    5,
			Labels: map[string]string{
				"engine_architecture_resolved": "gpt_oss",
				"identity_label":               "wins",
			},
		},
		Labels: map[string]string{
			"engine_architecture_resolved": "qwen3",
			"identity_label":               "loses",
		},
	})
	if report.Info.Architecture != "gpt-oss" ||
		report.Identity.Architecture != "gpt-oss" ||
		report.Info.NumLayers != 5 ||
		report.Labels["identity_label"] != "wins" ||
		report.Labels["engine_model_info_architecture"] != "gpt-oss" {
		t.Fatalf("ResolveModelInfo = %+v, want resolved architecture from identity labels", report)
	}
}

func TestModelInfoFromIdentity_Good_NormalizesAndCopies(t *testing.T) {
	labels := map[string]string{"engine_architecture_resolved": "qwen36_moe"}
	identity := inference.ModelIdentity{
		Path:       "/models/qwen",
		VocabSize:  9,
		HiddenSize: 4,
		QuantBits:  8,
		Labels:     labels,
	}
	info := ModelInfoFromIdentity("/ignored", identity)
	if info.Architecture != "qwen3_6_moe" ||
		info.VocabSize != 9 ||
		info.HiddenSize != 4 ||
		info.QuantBits != 8 {
		t.Fatalf("ModelInfoFromIdentity = %+v, want normalized qwen3_6_moe info", info)
	}
	report := ResolveModelInfo(ModelInfoRequest{Path: "/models/qwen", Identity: identity})
	report.Labels["engine_architecture_resolved"] = "mutated"
	if labels["engine_architecture_resolved"] != "qwen36_moe" {
		t.Fatalf("ResolveModelInfo aliased caller labels: %+v", labels)
	}
}
