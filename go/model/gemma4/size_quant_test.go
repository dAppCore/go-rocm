// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "testing"

func TestSizeQuantSupport_Good_FamilyOwnsMatrix(t *testing.T) {
	matrix := DefaultSizeQuantSupport()
	if len(matrix) != 5 {
		t.Fatalf("DefaultSizeQuantSupport len = %d, want E2B/E4B/12B/26B-A4B/31B", len(matrix))
	}
	for _, size := range []string{"E2B", "E4B"} {
		entry, ok := SizeQuantSupportBySize(size)
		if !ok || !entry.RunnableOnCard || !containsString(entry.QuantModes, "bf16") || !containsString(entry.QuantModes, "q8") || !containsString(entry.QuantModes, "q6") || !containsString(entry.QuantModes, "q4") {
			t.Fatalf("%s support = %+v ok=%v, want BF16/q8/q6/q4 runnable ladder", size, entry, ok)
		}
		bf16, ok := QuantModeSupportBySize(size, "bf16")
		if !ok || bf16.Runtime != RuntimeBF16 || bf16.GenerateStatus != GenerateLoadOnly {
			t.Fatalf("%s BF16 support = %+v ok=%v, want load-only correctness anchor", size, bf16, ok)
		}
		for _, mode := range []string{"q8", "q6", "q4"} {
			quant, ok := QuantModeSupportBySize(size, mode)
			if !ok || quant.Runtime != RuntimeMLXAffine || quant.GenerateStatus != GenerateLinked {
				t.Fatalf("%s %s support = %+v ok=%v, want linked MLX-affine generation", size, mode, quant, ok)
			}
		}
	}

	entry, ok := SizeQuantSupportBySize("12B")
	if !ok || !entry.RunnableOnCard || !containsString(entry.QuantModes, "q6") || !containsString(entry.QuantModes, "q4") {
		t.Fatalf("12B support = %+v ok=%v, want q6/q4 runnable targets", entry, ok)
	}
	for _, mode := range []string{"q6", "q4"} {
		quant, ok := QuantModeSupportBySize("12B", mode)
		if !ok || quant.Runtime != RuntimeMLXAffine || quant.GenerateStatus != GenerateLinked {
			t.Fatalf("12B %s support = %+v ok=%v, want linked MLX-affine target", mode, quant, ok)
		}
	}

	for _, size := range []string{"26B-A4B", "31B"} {
		entry, ok := SizeQuantSupportBySize(size)
		if !ok || entry.RunnableOnCard || entry.Runtime != RuntimePlanned {
			t.Fatalf("%s support = %+v ok=%v, want planned/not-runnable status", size, entry, ok)
		}
		for _, mode := range []string{"q8-status", "q6-status", "q4-status"} {
			quant, ok := QuantModeSupportBySize(size, mode)
			if !ok || quant.Runtime != RuntimePlanned || quant.GenerateStatus != GeneratePlannedOnly {
				t.Fatalf("%s %s support = %+v ok=%v, want planned status-only support", size, mode, quant, ok)
			}
		}
	}
}

func TestSizeQuantSupport_Good_DefensiveCopies(t *testing.T) {
	matrix := DefaultSizeQuantSupport()
	matrix[0].QuantModes[0] = "mutated"
	matrix[0].QuantModeSupport[0].GenerateStatus = "mutated"

	entry, ok := SizeQuantSupportBySize("E2B")
	if !ok || entry.QuantModes[0] == "mutated" || entry.QuantModeSupport[0].GenerateStatus == "mutated" {
		t.Fatalf("SizeQuantSupport leaked mutable quant modes: %+v ok=%v", entry, ok)
	}
}

func containsString(values []string, needle string) bool {
	for _, value := range values {
		if value == needle {
			return true
		}
	}
	return false
}
