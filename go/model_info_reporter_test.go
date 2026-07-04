// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestROCmModelInfo_Good_UsesFolderOwnedModelInfoReport(t *testing.T) {
	model := &rocmModel{
		modelPath: "/models/wrapped",
		modelType: "fallback",
		modelInfo: inference.ModelInfo{
			Architecture: "wrapped",
			VocabSize:    11,
			HiddenSize:   4,
			QuantBits:    8,
		},
		modelLabels: map[string]string{
			"engine_architecture_resolved": "gpt_oss",
			"quant_type":                   "mxfp8",
		},
		native: &hipLoadedModel{
			contextSize: 8192,
			modelLabels: map[string]string{
				"loaded_label": "kept",
			},
		},
	}

	info := model.Info()
	if info.Architecture != "gpt-oss" ||
		info.VocabSize != 11 ||
		info.HiddenSize != 4 ||
		info.QuantBits != 8 {
		t.Fatalf("Info = %+v, want resolved folder-owned model info", info)
	}

	identity := model.ModelIdentity()
	if identity.Architecture != "gpt-oss" ||
		identity.Path != "/models/wrapped" ||
		identity.ContextLength != 8192 ||
		identity.QuantType != "mxfp8" ||
		identity.Labels["loaded_label"] != "kept" ||
		identity.Labels["engine_model_info_contract"] != rocmmodel.ModelInfoReporterContract ||
		identity.Labels["engine_model_info_reactive"] != "true" {
		t.Fatalf("ModelIdentity = %+v, want info-report identity", identity)
	}

	identity.Labels["loaded_label"] = "mutated"
	if next := model.ModelIdentity(); next.Labels["loaded_label"] != "kept" {
		t.Fatalf("ModelIdentity leaked mutable labels: %+v", next)
	}
}
