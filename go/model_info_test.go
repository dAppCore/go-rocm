// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"context"
	"iter"
	"testing"

	"dappco.re/go/inference"
)

type rootModelInfoTestReporter struct {
	info inference.ModelInfo
}

func (reporter rootModelInfoTestReporter) FillModelInfo(info *inference.ModelInfo) {
	if info == nil {
		return
	}
	*info = reporter.info
}

func TestResolveROCmModelInfo_Good_RootReporterContract(t *testing.T) {
	callerLabels := map[string]string{"caller_label": "kept"}
	identityLabels := map[string]string{
		"engine_architecture_resolved": "Gemma4TextForCausalLM",
		"quant_type":                   "q6",
	}
	report := ResolveROCmModelInfo(ROCmModelInfoRequest{
		Path:      "/models/gemma4",
		ModelType: "wrapped",
		Info: inference.ModelInfo{
			Architecture: "wrapped",
			VocabSize:    7,
		},
		Identity: inference.ModelIdentity{
			Labels: identityLabels,
		},
		Labels: callerLabels,
		Reporter: rootModelInfoTestReporter{info: inference.ModelInfo{
			Architecture: "reporter",
			VocabSize:    262144,
			NumLayers:    26,
			HiddenSize:   2304,
			QuantBits:    6,
			QuantGroup:   64,
		}},
	})
	if !report.Matched() ||
		report.Contract != ROCmModelInfoReporterContract ||
		report.Source != "model_info_reporter" ||
		report.Path != "/models/gemma4" ||
		report.Architecture != "gemma4_text" ||
		report.Info.Architecture != "gemma4_text" ||
		report.Info.VocabSize != 262144 ||
		report.Info.NumLayers != 26 ||
		report.Identity.Architecture != "gemma4_text" ||
		report.Identity.QuantType != "q6" ||
		report.Identity.Labels["engine_model_info_contract"] != ROCmModelInfoReporterContract ||
		report.Identity.Labels["engine_model_info_reactive"] != "true" ||
		report.Identity.Labels["caller_label"] != "kept" {
		t.Fatalf("ResolveROCmModelInfo = %+v, want root model-info reporter contract", report)
	}

	report.Identity.Labels["quant_type"] = "mutated"
	if identityLabels["quant_type"] != "q6" || callerLabels["caller_label"] != "kept" {
		t.Fatalf("ResolveROCmModelInfo aliased caller labels: identity=%+v caller=%+v", identityLabels, callerLabels)
	}
}

func TestROCmModelInfoReportForModel_Good_UsesModelOwnedIdentityAndReporter(t *testing.T) {
	model := &rootModelInfoTestModel{
		modelType: "wrapped",
		info: inference.ModelInfo{
			Architecture: "wrapped",
			VocabSize:    3,
		},
		identity: inference.ModelIdentity{
			Path: "/models/gpt-oss",
			Labels: map[string]string{
				"engine_architecture_resolved": "gpt_oss",
				"quant_type":                   "mxfp8",
				"loaded_label":                 "kept",
			},
		},
		reporter: inference.ModelInfo{
			Architecture: "reporter",
			VocabSize:    32000,
			NumLayers:    2,
			HiddenSize:   1024,
			QuantBits:    8,
		},
	}

	report, ok := ROCmModelInfoReportForModel(model)
	if !ok ||
		report.Contract != ROCmModelInfoReporterContract ||
		report.Source != "model_info_reporter" ||
		report.Path != "/models/gpt-oss" ||
		report.Architecture != "gpt-oss" ||
		report.Info.Architecture != "gpt-oss" ||
		report.Info.VocabSize != 32000 ||
		report.Info.HiddenSize != 1024 ||
		report.Identity.Path != "/models/gpt-oss" ||
		report.Identity.QuantType != "mxfp8" ||
		report.Identity.Labels["loaded_label"] != "kept" ||
		report.Labels["engine_model_info_source"] != "model_info_reporter" {
		t.Fatalf("ROCmModelInfoReportForModel = %+v ok=%v, want model-owned identity and reporter", report, ok)
	}

	report.Identity.Labels["loaded_label"] = "mutated"
	next, ok := ROCmModelInfoReportForModel(model)
	if !ok || next.Identity.Labels["loaded_label"] != "kept" {
		t.Fatalf("ROCmModelInfoReportForModel leaked mutable labels: %+v ok=%v", next, ok)
	}
	if _, ok := ROCmModelInfoReportForModel(nil); ok {
		t.Fatal("ROCmModelInfoReportForModel(nil) ok = true, want false")
	}
}

func TestROCmModelInfoIdentity_Good_RootIdentityHelpers(t *testing.T) {
	identity := ROCmModelInfoIdentity("/models/qwen", inference.ModelInfo{
		Architecture: "Qwen3_5MoeForConditionalGeneration",
		VocabSize:    151936,
		QuantBits:    4,
		QuantGroup:   64,
	}, map[string]string{"quant_type": "q4"})
	if identity.Path != "/models/qwen" ||
		identity.Architecture != "qwen3_6_moe" ||
		identity.VocabSize != 151936 ||
		identity.QuantType != "q4" ||
		identity.Labels["engine_model_info_reactive"] != "true" {
		t.Fatalf("ROCmModelInfoIdentity = %+v, want normalized root identity helper", identity)
	}
	info := ROCmModelInfoFromIdentity("", identity)
	if info.Architecture != "qwen3_6_moe" || info.VocabSize != 151936 || info.QuantBits != 4 {
		t.Fatalf("ROCmModelInfoFromIdentity = %+v, want normalized info helper", info)
	}
}

type rootModelInfoTestModel struct {
	modelType string
	info      inference.ModelInfo
	identity  inference.ModelIdentity
	reporter  inference.ModelInfo
}

func (model *rootModelInfoTestModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (model *rootModelInfoTestModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (model *rootModelInfoTestModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (model *rootModelInfoTestModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (model *rootModelInfoTestModel) ModelType() string { return model.modelType }

func (model *rootModelInfoTestModel) Info() inference.ModelInfo { return model.info }

func (model *rootModelInfoTestModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{}
}

func (model *rootModelInfoTestModel) Err() error { return nil }

func (model *rootModelInfoTestModel) Close() error { return nil }

func (model *rootModelInfoTestModel) ModelIdentity() inference.ModelIdentity {
	return model.identity
}

func (model *rootModelInfoTestModel) FillModelInfo(info *inference.ModelInfo) {
	if info == nil {
		return
	}
	*info = model.reporter
}
