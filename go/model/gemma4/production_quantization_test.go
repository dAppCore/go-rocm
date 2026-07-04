// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	"dappco.re/go/inference"
)

func TestProductionQuantizationPack_Good_ModelFamilyOwnsPackMatrix(t *testing.T) {
	pack, ok := ProductionQuantizationPackByName("lmstudio-community/gemma-4-E4B-it-MLX-6bit")
	if !ok || pack.Size != "E4B" || pack.Bits != 6 || pack.GenerateStatus != GenerateLinked || !pack.RequiresBench {
		t.Fatalf("ProductionQuantizationPackByName(E4B q6) = %+v ok=%v, want runnable E4B q6 bench target", pack, ok)
	}

	gguf, ok := ProductionQuantizationPackByName("lmstudio-community/gemma-4-12B-it-GGUF:Q6_K")
	if !ok || gguf.Size != "12B" || gguf.Bits != 6 || gguf.Runtime != RuntimeGGUF || gguf.GenerateStatus != GenerateLoadOnly {
		t.Fatalf("ProductionQuantizationPackByName(12B GGUF Q6) = %+v ok=%v, want load-only GGUF q6 alias", gguf, ok)
	}

	assistant, ok := ProductionQuantizationPackByName("google/gemma-4-E4B-it-assistant")
	if !ok || assistant.Size != "E4B" || assistant.QuantMode != AssistantQuantMode || assistant.ProductRole != "mtp-assistant" {
		t.Fatalf("ProductionQuantizationPackByName(E4B assistant) = %+v ok=%v, want BF16 MTP assistant pack", assistant, ok)
	}

	qatAssistant, ok := ProductionQuantizationAssistantPackForModel(inference.ModelIdentity{
		Architecture: AssistantArchitecture,
		Labels: map[string]string{
			"gemma4_size":       "E4B",
			"gemma4_quant_mode": "q6",
		},
	})
	if !ok ||
		qatAssistant.Size != "E4B" ||
		qatAssistant.ModelID != "mlx-community/gemma-4-E4B-it-qat-assistant-6bit" ||
		ProductionQuantizationPackMode(qatAssistant) != "q6" ||
		qatAssistant.ProductRole != "mtp-assistant" ||
		qatAssistant.Runtime != RuntimeMLXAffine ||
		qatAssistant.GenerateStatus != GenerateLoadOnly {
		t.Fatalf("ProductionQuantizationAssistantPackForModel(E4B q6 labels) = %+v ok=%v, want MTP-QAT assistant pack", qatAssistant, ok)
	}
}

func TestProductionQuantizationTier_Good_ModelFamilySelectsMemoryLane(t *testing.T) {
	wide := inference.MachineDeviceInfo{MemorySize: 96 * productionQuantizationGiB, MaxRecommendedWorkingSetSize: 90 * productionQuantizationGiB}
	quality := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:        wide,
		ContextLength: ProductionLaneLongContextLength,
		QualityFirst:  true,
	})
	if quality.Tier.Bits != ProductionLaneQualityQuantBits || quality.Tier.ModelID != ProductionLaneCurrentQualityModelID || !quality.Fits {
		t.Fatalf("quality wide choice = %+v, want fitting q8", quality)
	}

	stepDown := SelectProductionQuantizationTier(ProductionQuantizationSelectionInput{
		Device:        inference.MachineDeviceInfo{MemorySize: 64 * productionQuantizationGiB, MaxRecommendedWorkingSetSize: 48 * productionQuantizationGiB},
		ContextLength: ProductionLaneLongContextLength,
		QualityFirst:  true,
	})
	if stepDown.Tier.Bits != ProductionLaneProductDefaultQuantBits ||
		stepDown.StepDownFromBits != ProductionLaneQualityQuantBits ||
		stepDown.StepDownRequiredWorkingSet != 64*productionQuantizationGiB {
		t.Fatalf("quality step-down choice = %+v, want q8 to q6 evidence at 64GiB requirement", stepDown)
	}
}

func TestProductionQuantization_Good_DefensiveCopies(t *testing.T) {
	packs := DefaultProductionQuantizationPackSupport()
	tiers := DefaultProductionQuantizationTiers()
	packs[0].Name = "mutated"
	tiers[0].Name = "mutated"

	nextPacks := DefaultProductionQuantizationPackSupport()
	nextTiers := DefaultProductionQuantizationTiers()
	if nextPacks[0].Name == "mutated" || nextTiers[0].Name == "mutated" {
		t.Fatalf("production quantization defaults leaked mutable state: packs=%+v tiers=%+v", nextPacks[0], nextTiers[0])
	}
}
