//go:build linux && amd64

package rocm

import (
	"strings"
	"testing"
)

func TestROCm_ROCmAvailable_Good(t *testing.T) {
	available := ROCmAvailable()
	if !available {
		t.Fatal("ROCmAvailable() = false, want true for linux/amd64 build")
	}
	if ROCmAvailable() != available {
		t.Fatal("ROCmAvailable() changed between calls")
	}
}

func TestROCm_ROCmAvailable_Bad(t *testing.T) {
	available := ROCmAvailable()
	info, err := GetVRAMInfo()
	if !available {
		t.Fatal("ROCmAvailable() = false, want compiled ROCm path")
	}
	if err == nil && info.Total == 0 {
		t.Fatal("GetVRAMInfo() succeeded with zero total VRAM")
	}
}

func TestROCm_ROCmAvailable_Ugly(t *testing.T) {
	first := ROCmAvailable()
	second := ROCmAvailable()
	if first != second {
		t.Fatalf("ROCmAvailable() changed from %v to %v", first, second)
	}
	if !first {
		t.Fatal("ROCmAvailable() = false, want stable true")
	}
}

func TestVRAM_GetVRAMInfo_Good(t *testing.T) {
	info, err := GetVRAMInfo()
	if err != nil {
		if info != (VRAMInfo{}) {
			t.Errorf("info = %+v, want zero when error is returned", info)
		}
		return
	}
	if info.Total == 0 {
		t.Fatal("info.Total = 0, want positive VRAM total")
	}
	if info.Free > info.Total {
		t.Fatalf("info.Free = %d, want <= total %d", info.Free, info.Total)
	}
}

func TestVRAM_GetVRAMInfo_Bad(t *testing.T) {
	info, err := GetVRAMInfo()
	if err != nil && !strings.Contains(err.Error(), "rocm.GetVRAMInfo") {
		t.Fatalf("err = %v, want rocm.GetVRAMInfo scope", err)
	}
	if err == nil && info.Used > info.Total && info.Free != 0 {
		t.Fatalf("info = %+v, want free clamped to zero when used exceeds total", info)
	}
}

func TestVRAM_GetVRAMInfo_Ugly(t *testing.T) {
	first, firstErr := GetVRAMInfo()
	second, secondErr := GetVRAMInfo()
	if (firstErr == nil) != (secondErr == nil) {
		t.Fatalf("error stability changed from %v to %v", firstErr, secondErr)
	}
	if firstErr == nil && (first.Total == 0 || second.Total == 0) {
		t.Fatalf("totals = %d, %d; want positive totals", first.Total, second.Total)
	}
}
