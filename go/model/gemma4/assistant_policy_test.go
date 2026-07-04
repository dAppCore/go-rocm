// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"strings"
	"testing"

	"dappco.re/go/inference"
)

func TestMTPAssistantPolicy_Good_SizeScopedBF16Lane(t *testing.T) {
	tests := []struct {
		size       string
		wantSize   string
		wantHidden int
	}{
		{size: "e2b", wantSize: "E2B", wantHidden: 1536},
		{size: "E4B", wantSize: "E4B", wantHidden: 2304},
		{size: "12b", wantSize: "12B", wantHidden: 3840},
		{size: "26B-A4B", wantSize: "26B-A4B", wantHidden: 4096},
		{size: "31b", wantSize: "31B", wantHidden: 4096},
		{size: "", wantSize: "E2B", wantHidden: 1536},
	}
	for _, tt := range tests {
		t.Run(tt.wantSize, func(t *testing.T) {
			path := MTPAssistantPath(tt.size, "bf16")
			if path != "google/gemma-4-"+tt.wantSize+"-it-assistant" {
				t.Fatalf("MTPAssistantPath(%q) = %q, want google/gemma-4-%s-it-assistant", tt.size, path, tt.wantSize)
			}
			if hidden := MTPAssistantHiddenSizeForTarget(tt.size, 0); hidden != tt.wantHidden {
				t.Fatalf("MTPAssistantHiddenSizeForTarget(%q, 0) = %d, want %d", tt.size, hidden, tt.wantHidden)
			}
			labels := MTPAssistantLabels(tt.size, map[string]string{"caller": "kept"})
			if labels["caller"] != "kept" ||
				labels["gemma4_size"] != tt.wantSize ||
				labels["gemma4_quant_mode"] != "bf16" ||
				labels["gemma4_runtime"] != RuntimeBF16 ||
				labels["gemma4_generate_status"] != GenerateLoadOnly ||
				labels["production_quant_pack"] != tt.wantSize+":assistant-bf16" ||
				labels["production_quant_pack_name"] != strings.ToLower(tt.wantSize)+"-assistant-bf16" ||
				labels["production_quant_model"] != path ||
				labels["production_quant_assistant_model"] != path ||
				labels["production_quant_mtp_assistant"] != "true" ||
				labels["production_quant_target_family"] != "gemma4" {
				t.Fatalf("MTPAssistantLabels(%q) = %+v, want BF16 assistant production labels", tt.size, labels)
			}
		})
	}
}

func TestMTPAssistantPolicy_Good_QATAssistantLane(t *testing.T) {
	path := MTPAssistantPath("12b", "q4")
	if path != "mlx-community/gemma-4-12B-it-qat-assistant-4bit" {
		t.Fatalf("MTPAssistantPath(12b, q4) = %q, want MTP-QAT assistant", path)
	}
	if q6 := MTPAssistantPath("E4B", "q6"); q6 != "mlx-community/gemma-4-E4B-it-qat-assistant-6bit" {
		t.Fatalf("MTPAssistantPath(E4B, q6) = %q, want MTP-QAT q6 assistant", q6)
	}

	labels := MTPAssistantLabelsForModel("12B", "q4", "", map[string]string{"caller": "kept"})
	if labels["caller"] != "kept" ||
		labels["gemma4_size"] != "12B" ||
		labels["gemma4_quant_mode"] != "q4" ||
		labels["gemma4_runtime"] != RuntimeMLXAffine ||
		labels["gemma4_generate_status"] != GenerateLoadOnly ||
		labels["production_quant_pack"] != "12B:assistant-q4" ||
		labels["production_quant_pack_name"] != "12b-assistant-4bit" ||
		labels["production_quant_model"] != path ||
		labels["production_quant_assistant_model"] != path ||
		labels["production_quant_collection"] != MTPQATCollectionID ||
		labels["production_quant_mtp_assistant"] != "true" {
		t.Fatalf("MTPAssistantLabelsForModel(12B, q4) = %+v, want MTP-QAT assistant labels", labels)
	}
}

func TestMTPAssistantPolicy_Good_DefensiveLabels(t *testing.T) {
	input := map[string]string{"gemma4_size": "mutated"}
	labels := MTPAssistantLabels("E4B", input)
	labels["caller"] = "changed"
	if input["caller"] != "" || input["gemma4_size"] != "mutated" {
		t.Fatalf("MTPAssistantLabels mutated caller map: input=%+v labels=%+v", input, labels)
	}
}

func TestAssistantConfigPolicy_Good_OfficialLayoutLabels(t *testing.T) {
	labels, contradicts := ApplyAssistantConfigLabels(map[string]string{"caller": "kept"}, AssistantConfig{
		BackboneHiddenSize:       1536,
		NumCentroids:             AssistantOrderedEmbeddingCentroids,
		CentroidIntermediateTopK: AssistantCentroidIntermediateTopK,
		UseOrderedEmbeddings:     true,
		UseOrderedEmbeddingsSet:  true,
		NumLayers:                AssistantLayerCount,
		VocabSize:                AssistantTokenOrderingVocabSize,
	})
	if contradicts ||
		labels["caller"] != "kept" ||
		labels["attached_drafter_assistant_backbone_hidden_size"] != "1536" ||
		labels["attached_drafter_assistant_centroids"] != "2048" ||
		labels["attached_drafter_assistant_centroid_intermediate_top_k"] != "32" ||
		labels["attached_drafter_assistant_ordered_embeddings"] != "true" ||
		labels["attached_drafter_assistant_layer_count"] != "4" ||
		labels["attached_drafter_assistant_four_layer_drafter"] != "true" ||
		labels["attached_drafter_assistant_token_ordering_shape"] != AssistantTokenOrderingShape {
		t.Fatalf("ApplyAssistantConfigLabels official = %+v contradicts=%v, want official layout labels", labels, contradicts)
	}
}

func TestAssistantConfigPolicy_Bad_ContradictsOfficial(t *testing.T) {
	tests := []struct {
		name string
		cfg  AssistantConfig
	}{
		{name: "centroids", cfg: AssistantConfig{NumCentroids: 1024}},
		{name: "topk", cfg: AssistantConfig{CentroidIntermediateTopK: 16}},
		{name: "ordered", cfg: AssistantConfig{UseOrderedEmbeddingsSet: true}},
		{name: "layers", cfg: AssistantConfig{NumLayers: 2}},
		{name: "shape", cfg: AssistantConfig{NumCentroids: AssistantOrderedEmbeddingCentroids, VocabSize: 131072}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			labels, contradicts := ApplyAssistantConfigLabels(nil, tt.cfg)
			if !contradicts {
				t.Fatalf("ApplyAssistantConfigLabels(%+v) labels=%+v contradicts=false, want true", tt.cfg, labels)
			}
		})
	}
}

func TestAssistantPairPolicy_Good_OfficialAndFamilyPairs(t *testing.T) {
	target := inference.ModelIdentity{
		QuantGroup: 64,
		Labels: map[string]string{
			"gemma4_size":            "E2B",
			"gemma4_quant_mode":      "q6",
			"gemma4_runtime":         RuntimeMLXAffine,
			"gemma4_generate_status": GenerateLinked,
		},
	}
	assistant := inference.ModelIdentity{
		Labels: map[string]string{
			"gemma4_size":            "e2b",
			"gemma4_quant_mode":      AssistantQuantMode,
			"gemma4_runtime":         RuntimeBF16,
			"gemma4_generate_status": GenerateLoadOnly,
		},
	}
	if !OfficialPairVerified(target, assistant) || !FamilyPairVerified(target, assistant) {
		t.Fatalf("official E2B pair did not verify")
	}
	labels := map[string]string{}
	ApplyOfficialPairLockLabels(labels, target, assistant, false)
	if labels["attached_drafter_official_pair_verified"] != "true" ||
		labels["attached_drafter_gemma4_family_pair_verified"] != "true" ||
		labels["attached_drafter_official_target_model_id"] != OfficialE2BTargetModelID ||
		labels["attached_drafter_official_assistant_model_id"] != OfficialE2BAssistantModelID {
		t.Fatalf("ApplyOfficialPairLockLabels = %+v, want locked official pair labels", labels)
	}
}

func TestAssistantPairPolicy_Good_FamilyOnly(t *testing.T) {
	target := inference.ModelIdentity{
		QuantGroup: 64,
		Labels: map[string]string{
			"gemma4_size":            "E4B",
			"gemma4_quant_mode":      "q6",
			"gemma4_runtime":         RuntimeMLXAffine,
			"gemma4_generate_status": GenerateLinked,
		},
	}
	assistant := inference.ModelIdentity{
		Labels: map[string]string{
			"gemma4_size":            "E4B",
			"gemma4_quant_mode":      AssistantQuantMode,
			"gemma4_runtime":         RuntimeBF16,
			"gemma4_generate_status": GenerateLoadOnly,
		},
	}
	if OfficialPairVerified(target, assistant) || !FamilyPairVerified(target, assistant) {
		t.Fatalf("E4B pair official=%v family=%v, want family-only", OfficialPairVerified(target, assistant), FamilyPairVerified(target, assistant))
	}
	labels := map[string]string{}
	ApplyOfficialPairLockLabels(labels, target, assistant, true)
	if labels["attached.drafter.official_pair_verified"] != "false" ||
		labels["attached.drafter.gemma4_family_pair_verified"] != "true" ||
		labels["attached.drafter.official_target_model_id"] != "" {
		t.Fatalf("ApplyOfficialPairLockLabels family-only = %+v, want no official lock ids", labels)
	}
}
