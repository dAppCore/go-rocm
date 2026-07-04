// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "testing"

func TestQATCollection_Good_TargetAndAssistantEntries(t *testing.T) {
	targets := DefaultQATTargetCollection()
	assistants := DefaultMTPQATCollection()
	if len(targets) != 40 || len(assistants) != 40 {
		t.Fatalf("collection sizes = targets:%d assistants:%d, want 40 target and 40 assistant entries", len(targets), len(assistants))
	}

	target, ok := QATCollectionEntryForModelID("mlx-community/gemma-4-12B-it-qat-4bit")
	if !ok ||
		target.CollectionID != QATCollectionID ||
		target.ModelID != "mlx-community/gemma-4-12B-it-qat-4bit" ||
		target.Size != "12B" ||
		target.QuantMode != "q4" ||
		target.Bits != 4 ||
		target.QuantGroup != 64 ||
		target.Assistant ||
		target.Runtime != RuntimeMLXAffine ||
		target.GenerateStatus != GenerateLinked ||
		!target.RunnableOnCard {
		t.Fatalf("12B q4 target entry = %+v ok=%v, want linked QAT target", target, ok)
	}

	assistant, ok := QATCollectionEntryForModelID("mlx-community/gemma-4-12B-it-qat-assistant-4bit")
	if !ok ||
		assistant.CollectionID != MTPQATCollectionID ||
		assistant.ModelID != "mlx-community/gemma-4-12B-it-qat-assistant-4bit" ||
		assistant.Size != "12B" ||
		assistant.QuantMode != "q4" ||
		assistant.Bits != 4 ||
		assistant.QuantGroup != 64 ||
		!assistant.Assistant ||
		assistant.Runtime != RuntimeMLXAffine ||
		assistant.GenerateStatus != GenerateLoadOnly ||
		!assistant.RunnableOnCard {
		t.Fatalf("12B q4 assistant entry = %+v ok=%v, want load-only QAT MTP assistant", assistant, ok)
	}

	nvfp4, ok := QATCollectionEntryForModelID("mlx-community/gemma-4-E4B-it-qat-nvfp4")
	if !ok || nvfp4.QuantMode != "nvfp4" || nvfp4.Runtime != RuntimePlanned || nvfp4.GenerateStatus != GeneratePlannedOnly {
		t.Fatalf("E4B nvfp4 entry = %+v ok=%v, want planned recognized pack", nvfp4, ok)
	}
}
