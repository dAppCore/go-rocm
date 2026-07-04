// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"testing"

	"dappco.re/go/inference"
	modelgemma4 "dappco.re/go/rocm/model/gemma4"
)

func TestDefaultGemma4SizeQuantSupportUserMatrix(t *testing.T) {
	matrix := DefaultGemma4SizeQuantSupport()
	if len(matrix) != 5 {
		t.Fatalf("Gemma4 size matrix length = %d, want E2B/E4B/12B/26B-A4B/31B", len(matrix))
	}
	for _, size := range []string{"E2B", "E4B"} {
		entry, ok := Gemma4SizeQuantSupportBySize(size)
		if !ok || !entry.RunnableOnCard || !stringSliceContains(entry.QuantModes, "bf16") || !stringSliceContains(entry.QuantModes, "q8") || !stringSliceContains(entry.QuantModes, "q6") || !stringSliceContains(entry.QuantModes, "q4") {
			t.Fatalf("%s support = %+v ok=%v, want BF16/q8/q6/q4 runnable ladder", size, entry, ok)
		}
		if stringSliceContains(entry.QuantModes, "q5-bench") {
			t.Fatalf("%s support = %+v, want no benchmark-only q5 pseudo-mode", size, entry)
		}
		bf16, ok := Gemma4QuantModeSupportBySize(size, "bf16")
		if !ok || bf16.Runtime != Gemma4RuntimeBF16 || bf16.GenerateStatus != Gemma4GenerateLoadOnly {
			t.Fatalf("%s BF16 support = %+v ok=%v, want load-only correctness anchor", size, bf16, ok)
		}
		for _, mode := range []string{"q8", "q6", "q4"} {
			quant, ok := Gemma4QuantModeSupportBySize(size, mode)
			if !ok || quant.Runtime != Gemma4RuntimeMLXAffine || quant.GenerateStatus != Gemma4GenerateLinked {
				t.Fatalf("%s %s support = %+v ok=%v, want linked MLX-affine generation", size, mode, quant, ok)
			}
		}
	}
	entry, ok := Gemma4SizeQuantSupportBySize("12B")
	if !ok || !entry.RunnableOnCard || !stringSliceContains(entry.QuantModes, "q6") || !stringSliceContains(entry.QuantModes, "q4") {
		t.Fatalf("12B support = %+v ok=%v, want q6/q4 runnable targets", entry, ok)
	}
	for _, mode := range []string{"q6", "q4"} {
		quant, ok := Gemma4QuantModeSupportBySize("12B", mode)
		if !ok || quant.Runtime != Gemma4RuntimeMLXAffine || quant.GenerateStatus != Gemma4GenerateLinked {
			t.Fatalf("12B %s support = %+v ok=%v, want linked MLX-affine target", mode, quant, ok)
		}
	}
	for _, size := range []string{"26B-A4B", "31B"} {
		entry, ok := Gemma4SizeQuantSupportBySize(size)
		if !ok || entry.RunnableOnCard || entry.Runtime != Gemma4RuntimePlanned {
			t.Fatalf("%s support = %+v ok=%v, want planned/not-runnable status", size, entry, ok)
		}
		for _, mode := range []string{"q8-status", "q6-status", "q4-status"} {
			quant, ok := Gemma4QuantModeSupportBySize(size, mode)
			if !ok || quant.Runtime != Gemma4RuntimePlanned || quant.GenerateStatus != Gemma4GeneratePlannedOnly {
				t.Fatalf("%s %s support = %+v ok=%v, want planned status-only support", size, mode, quant, ok)
			}
		}
	}
}

func TestDefaultGemma4SizeQuantSupportDefensiveCopies(t *testing.T) {
	matrix := DefaultGemma4SizeQuantSupport()
	matrix[0].QuantModes[0] = "mutated"
	matrix[0].QuantModeSupport[0].GenerateStatus = "mutated"
	entry, ok := Gemma4SizeQuantSupportBySize("E2B")
	if !ok || entry.QuantModes[0] == "mutated" || entry.QuantModeSupport[0].GenerateStatus == "mutated" {
		t.Fatalf("Gemma4 size support leaked mutable quant modes: %+v ok=%v", entry, ok)
	}
}

func TestROCmGemma4ProductionQuantLabelsUseSizeSpecificPacks(t *testing.T) {
	tests := []struct {
		name            string
		path            string
		wantSize        string
		wantMode        string
		wantTier        string
		wantModel       string
		wantLocked      string
		wantTarget      string
		wantArchived    string
		wantActiveBytes bool
	}{
		{
			name:            "e2b_default_keeps_locked_lane",
			path:            "/models/lmstudio-community-gemma-4-e2b-it-6bit",
			wantSize:        "E2B",
			wantMode:        "q6",
			wantTier:        "default",
			wantModel:       ProductionLaneCurrentModelID,
			wantLocked:      ProductionLaneModelID,
			wantTarget:      ProductionLaneCurrentModelID,
			wantArchived:    ProductionLaneCurrentConstrainedModelID,
			wantActiveBytes: true,
		},
		{
			name:         "e4b_quality_uses_e4b_pack",
			path:         "/models/lmstudio-community-gemma-4-e4b-it-8bit",
			wantSize:     "E4B",
			wantMode:     "q8",
			wantTier:     "quality",
			wantModel:    "lmstudio-community/gemma-4-E4B-it-MLX-8bit",
			wantLocked:   "",
			wantTarget:   "lmstudio-community/gemma-4-E4B-it-MLX-6bit",
			wantArchived: "lmstudio-community/gemma-4-E4B-it-MLX-4bit",
		},
		{
			name:       "12b_target_uses_12b_pack",
			path:       "/models/lmstudio-community-gemma-4-12b-it-6bit",
			wantSize:   "12B",
			wantMode:   "q6",
			wantTier:   "largest-local-target",
			wantModel:  "mlx-community/gemma-4-12b-it-6bit",
			wantLocked: "",
			wantTarget: "mlx-community/gemma-4-12b-it-6bit",
		},
		{
			name:       "31b_status_only_uses_31b_pack",
			path:       "/models/lmstudio-community-gemma-4-31b-it-6bit",
			wantSize:   "31B",
			wantMode:   "q6-status",
			wantTier:   "status-only",
			wantModel:  "lmstudio-community/gemma-4-31B-it-MLX-6bit",
			wantLocked: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			labels := map[string]string{}
			rocmApplyGemma4ProductionQuantLabels(labels, inference.ModelIdentity{
				Architecture: "gemma4_text",
				Path:         tt.path,
			})

			if labels["production_quant_size"] != tt.wantSize ||
				labels["production_quant_mode"] != tt.wantMode ||
				labels["production_quant_tier"] != tt.wantTier ||
				labels["production_quant_model"] != tt.wantModel ||
				labels["production_quant_locked_model"] != tt.wantLocked ||
				labels["production_quant_target_model"] != tt.wantTarget ||
				labels["production_quant_archived_baseline"] != tt.wantArchived {
				t.Fatalf("production quant labels = %+v, want size=%s mode=%s tier=%s model=%s locked=%s target=%s archived=%s", labels, tt.wantSize, tt.wantMode, tt.wantTier, tt.wantModel, tt.wantLocked, tt.wantTarget, tt.wantArchived)
			}
			hasActiveBytes := labels["production_quant_active_weight_read_bytes_per_token"] != ""
			if hasActiveBytes != tt.wantActiveBytes {
				t.Fatalf("production quant labels = %+v, active bytes present=%t want %t", labels, hasActiveBytes, tt.wantActiveBytes)
			}
		})
	}
}

func TestROCmGemma4ProductionQuantLabelsRecognizeQATCollectionPaths(t *testing.T) {
	targetLabels := map[string]string{}
	rocmApplyGemma4ProductionQuantLabels(targetLabels, inference.ModelIdentity{
		Architecture: "gemma4_text",
		Path:         "/models/mlx-community/gemma-4-12B-it-qat-4bit",
	})
	if targetLabels["production_quant_collection"] != modelgemma4.QATCollectionID ||
		targetLabels["production_quant_size"] != "12B" ||
		targetLabels["production_quant_mode"] != "q4" ||
		targetLabels["production_quant_tier"] != "constrained" ||
		targetLabels["production_quant_model"] != "mlx-community/gemma-4-12B-it-qat-4bit" ||
		targetLabels["production_quant_generate_status"] != Gemma4GenerateLinked ||
		targetLabels["production_quant_runnable_on_card"] != "true" {
		t.Fatalf("target QAT production labels = %+v, want linked 12B q4 collection target", targetLabels)
	}

	assistantLabels := map[string]string{}
	rocmApplyGemma4ProductionQuantLabels(assistantLabels, inference.ModelIdentity{
		Architecture: officialGemma4E2BAssistantArchitecture,
		Path:         "/models/mlx-community/gemma-4-12B-it-qat-assistant-4bit",
	})
	if assistantLabels["production_quant_collection"] != modelgemma4.MTPQATCollectionID ||
		assistantLabels["production_quant_size"] != "12B" ||
		assistantLabels["production_quant_mode"] != "q4" ||
		assistantLabels["production_quant_pack"] != "12B:assistant-q4" ||
		assistantLabels["production_quant_pack_name"] != "12b-qat-assistant-4bit" ||
		assistantLabels["production_quant_model"] != "mlx-community/gemma-4-12B-it-qat-assistant-4bit" ||
		assistantLabels["production_quant_generate_status"] != Gemma4GenerateLoadOnly ||
		assistantLabels["production_quant_mtp_assistant"] != "true" {
		t.Fatalf("assistant QAT production labels = %+v, want load-only MTP QAT assistant", assistantLabels)
	}

	labelsOnlyAssistant := map[string]string{
		"gemma4_size":       "E4B",
		"gemma4_quant_mode": "q6",
	}
	rocmApplyGemma4SizeQuantSupportLabels(labelsOnlyAssistant, inference.ModelIdentity{
		Architecture: officialGemma4E2BAssistantArchitecture,
		Labels:       labelsOnlyAssistant,
	})
	rocmApplyGemma4ProductionQuantLabels(labelsOnlyAssistant, inference.ModelIdentity{
		Architecture: officialGemma4E2BAssistantArchitecture,
		Labels:       labelsOnlyAssistant,
	})
	if labelsOnlyAssistant["gemma4_pack_supported"] != "true" ||
		labelsOnlyAssistant["gemma4_runtime"] != Gemma4RuntimeMLXAffine ||
		labelsOnlyAssistant["gemma4_generate_status"] != Gemma4GenerateLoadOnly ||
		labelsOnlyAssistant["production_quant_pack"] != "E4B:assistant-q6" ||
		labelsOnlyAssistant["production_quant_model"] != rocmGemma4MTPAssistantPath("E4B", "q6") ||
		labelsOnlyAssistant["production_quant_mtp_assistant"] != "true" {
		t.Fatalf("labels-only assistant q6 labels = %+v, want reactive MTP-QAT assistant support", labelsOnlyAssistant)
	}
}

func TestROCmGemma4ProductionQuantLabelsRespectEffectiveLoadOnlySource(t *testing.T) {
	labels := map[string]string{}
	rocmApplyGemma4ProductionQuantLabels(labels, inference.ModelIdentity{
		Architecture: "gemma4_text",
		Path:         "/models/lmstudio-community-gemma-4-e2b-it-4bit",
		Labels: map[string]string{
			"gemma4_source_format": "gguf",
		},
	})

	if labels["production_quant_size"] != "E2B" ||
		labels["production_quant_mode"] != "q4" ||
		labels["production_quant_runtime"] != Gemma4RuntimeGGUF ||
		labels["production_quant_generate_status"] != Gemma4GenerateLoadOnly ||
		labels["production_quant_supported"] != "true" ||
		labels["production_quant_runnable_on_card"] != "true" {
		t.Fatalf("production quant labels = %+v, want GGUF effective load-only status over q4 pack defaults", labels)
	}
}

func TestROCmGemma4ProductionQuantLabelsInferGGUFLoadOnlyFromPath(t *testing.T) {
	labels := map[string]string{}
	rocmApplyGemma4ProductionQuantLabels(labels, inference.ModelIdentity{
		Architecture: "gemma4_text",
		Path:         "/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q6_K.gguf",
	})

	if labels["production_quant_size"] != "12B" ||
		labels["production_quant_mode"] != "q6" ||
		labels["production_quant_runtime"] != Gemma4RuntimeGGUF ||
		labels["production_quant_generate_status"] != Gemma4GenerateLoadOnly ||
		labels["production_quant_supported"] != "true" ||
		labels["production_quant_runnable_on_card"] != "true" {
		t.Fatalf("production quant labels = %+v, want path-inferred GGUF q6 load-only status", labels)
	}
}

func TestROCmGemma4CanonicalSizeQuantLabels(t *testing.T) {
	size := rocmGemma4CanonicalSize(" 31b ")
	mode := rocmGemma4CanonicalQuantMode(size, " Q6 ")
	if size != "31B" || mode != "q6-status" {
		t.Fatalf("canonical labels = %q/%q, want 31B/q6-status", size, mode)
	}
	mode = rocmGemma4CanonicalQuantMode(size, " Q4 ")
	if mode != "q4-status" {
		t.Fatalf("canonical mode = %q, want 31B/q4-status", mode)
	}
	mode = rocmGemma4CanonicalQuantMode("26B-A4B", " Q8 ")
	if mode != "q8-status" {
		t.Fatalf("canonical mode = %q, want 26B-A4B/q8-status", mode)
	}
	size = rocmGemma4CanonicalSize(" e4b ")
	mode = rocmGemma4CanonicalQuantMode(size, " Q8 ")
	if size != "E4B" || mode != "q8" {
		t.Fatalf("canonical labels = %q/%q, want E4B/q8", size, mode)
	}
}

func TestROCmGemma4SupportMatrixGenerateLinkedRespectsFailClosedLabels(t *testing.T) {
	model := inference.ModelIdentity{
		Architecture: "gemma4_text",
		QuantBits:    6,
		Labels: map[string]string{
			"gemma4_pack_supported": " FALSE ",
		},
	}
	if rocmGemma4SupportMatrixGenerateLinked(model) {
		t.Fatalf("Gemma4 pack-supported=false label must veto linked generation")
	}
	model.Labels = map[string]string{"gemma4_generate_status": " LOAD_ONLY "}
	if rocmGemma4SupportMatrixGenerateLinked(model) {
		t.Fatalf("Gemma4 load-only status label must veto linked generation")
	}
	model.Labels = map[string]string{"gemma4_generate_status": " Planned_Only "}
	if rocmGemma4SupportMatrixGenerateLinked(model) {
		t.Fatalf("Gemma4 planned-only status label must veto linked generation")
	}
	model.Labels = map[string]string{"gemma4_runnable_on_card": " FALSE "}
	if rocmGemma4SupportMatrixGenerateLinked(model) {
		t.Fatalf("Gemma4 runnable-on-card=false label must veto linked generation")
	}
	model.Labels = map[string]string{"format": " GGUF "}
	if rocmGemma4SupportMatrixGenerateLinked(model) {
		t.Fatalf("Gemma4 GGUF format label must veto linked generation")
	}
	model.Labels = nil
	model.Path = "/models/lmstudio-community/gemma-4-e2b-it-GGUF/gemma-4-E2B-it-Q6_K.gguf"
	if rocmGemma4SupportMatrixGenerateLinked(model) {
		t.Fatalf("Gemma4 GGUF path must veto linked generation")
	}
	model.Path = ""
	model.Labels = map[string]string{"gemma4_generate_status": Gemma4GenerateLinked}
	if rocmGemma4SupportMatrixGenerateLinked(model) {
		t.Fatalf("Gemma4 linked status without declared size/quant metadata must not use a bit-only fallback")
	}
	model.NumLayers = productionLaneGemma4E2BLayers
	model.HiddenSize = productionLaneGemma4E2BHiddenSize
	if rocmGemma4SupportMatrixGenerateLinked(model) {
		t.Fatalf("Gemma4 linked status with only hardcoded shape metadata must not expose linked generation")
	}
	model.Labels = map[string]string{
		"gemma4_generate_status": Gemma4GenerateLinked,
		"gemma4_size":            "E2B",
		"gemma4_quant_mode":      "q6",
	}
	if !rocmGemma4SupportMatrixGenerateLinked(model) {
		t.Fatalf("Gemma4 declared linked E2B/q6 metadata should enable linked generation")
	}
}

func TestROCmGemma4Q5PathFailsClosed(t *testing.T) {
	model := inference.ModelIdentity{
		Architecture: "gemma4_text",
		Path:         "/models/lmstudio-community-gemma-4-e2b-it-5bit",
	}
	mode := rocmGemma4ModelPackQuantModeForPath(model, model.Path)
	if mode != "q5" {
		t.Fatalf("Gemma4 q5 path mode = %q, want q5", mode)
	}
	labels := map[string]string{}
	rocmApplyGemma4SizeQuantSupportLabels(labels, model)
	if labels["gemma4_quant_mode"] != "q5" || labels["gemma4_pack_supported"] != "false" {
		t.Fatalf("Gemma4 q5 labels = %+v, want explicit unsupported q5", labels)
	}
	if rocmGemma4SupportMatrixGenerateLinked(model) {
		t.Fatalf("Gemma4 q5 path must not expose linked generation")
	}
}

func TestROCmGemma4MXFPPathModeSurvivesBitOnlyIdentity(t *testing.T) {
	for _, tt := range []struct {
		name      string
		path      string
		bits      int
		wantSize  string
		wantMode  string
		wantGroup int
	}{
		{name: "e2b_mxfp8", path: "/models/lmstudio-community-gemma-4-e2b-it-mxfp8", bits: 8, wantSize: "E2B", wantMode: "mxfp8", wantGroup: 32},
		{name: "e4b_mxfp4", path: "/models/lmstudio-community-gemma-4-e4b-it-mxfp4", bits: 4, wantSize: "E4B", wantMode: "mxfp4", wantGroup: 32},
	} {
		t.Run(tt.name, func(t *testing.T) {
			model := inference.ModelIdentity{
				Architecture: "gemma4_text",
				Path:         tt.path,
				QuantBits:    tt.bits,
			}
			mode := rocmGemma4ModelPackQuantModeForPath(model, model.Path)
			if mode != tt.wantMode {
				t.Fatalf("Gemma4 MXFP path mode = %q, want %s", mode, tt.wantMode)
			}
			identity := rocmGemma4ModelWithInferredPathQuant(model)
			if identity.QuantGroup != tt.wantGroup {
				t.Fatalf("Gemma4 MXFP identity = %+v, want quant group %d", identity, tt.wantGroup)
			}
			labels := map[string]string{}
			rocmApplyGemma4SizeQuantSupportLabels(labels, identity)
			if labels["gemma4_size"] != tt.wantSize ||
				labels["gemma4_quant_mode"] != tt.wantMode ||
				labels["gemma4_runtime"] != Gemma4RuntimePlanned ||
				labels["gemma4_generate_status"] != Gemma4GeneratePlannedOnly ||
				labels["gemma4_pack_supported"] != "true" ||
				labels["gemma4_runnable_on_card"] != "true" {
				t.Fatalf("Gemma4 MXFP labels = %+v, want %s/%s planned-only labels", labels, tt.wantSize, tt.wantMode)
			}
			if rocmGemma4SupportMatrixGenerateLinked(model) {
				t.Fatalf("%s must not expose linked generation", tt.path)
			}
		})
	}
}

func TestROCmGemma4LargePathVariantsRemainStatusOnly(t *testing.T) {
	tests := []struct {
		name     string
		path     string
		wantSize string
		wantMode string
	}{
		{name: "26b_q8", path: "/models/lmstudio-community-gemma-4-26b-a4b-it-8bit", wantSize: "26B-A4B", wantMode: "q8-status"},
		{name: "26b_q4", path: "/models/lmstudio-community-gemma-4-26b-a4b-it-4bit", wantSize: "26B-A4B", wantMode: "q4-status"},
		{name: "31b_q8", path: "/models/lmstudio-community-gemma-4-31b-it-8bit", wantSize: "31B", wantMode: "q8-status"},
		{name: "31b_q4", path: "/models/lmstudio-community-gemma-4-31b-it-4bit", wantSize: "31B", wantMode: "q4-status"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := inference.ModelIdentity{
				Architecture: "gemma4_text",
				Path:         tt.path,
			}
			labels := map[string]string{}
			rocmApplyGemma4SizeQuantSupportLabels(labels, model)
			if labels["gemma4_size"] != tt.wantSize ||
				labels["gemma4_quant_mode"] != tt.wantMode ||
				labels["gemma4_runtime"] != Gemma4RuntimePlanned ||
				labels["gemma4_generate_status"] != Gemma4GeneratePlannedOnly ||
				labels["gemma4_pack_supported"] != "true" ||
				labels["gemma4_runnable_on_card"] != "false" {
				t.Fatalf("Gemma4 large labels = %+v, want %s/%s planned status-only", labels, tt.wantSize, tt.wantMode)
			}
			if rocmGemma4SupportMatrixGenerateLinked(model) {
				t.Fatalf("%s must not expose linked generation", tt.path)
			}
		})
	}
}
