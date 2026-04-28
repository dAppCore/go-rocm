//go:build !linux || !amd64

package rocm

import (
	"strings"
	"testing"
)

func TestROCm_ROCmAvailable_Good(t *testing.T) {
	available := ROCmAvailable()
	if available {
		t.Fatal("ROCmAvailable() = true, want false on this platform")
	}
	if ROCmAvailable() != available {
		t.Fatal("ROCmAvailable() changed between calls")
	}
}

func TestROCm_ROCmAvailable_Bad(t *testing.T) {
	_, err := GetVRAMInfo()
	available := ROCmAvailable()
	if available {
		t.Fatal("ROCmAvailable() = true, want false when platform stub is active")
	}
	if err == nil {
		t.Fatal("GetVRAMInfo() error = nil, want platform error")
	}
}

func TestROCm_ROCmAvailable_Ugly(t *testing.T) {
	first := ROCmAvailable()
	second := ROCmAvailable()
	if first != second {
		t.Fatalf("ROCmAvailable() changed from %v to %v", first, second)
	}
	if first {
		t.Fatal("ROCmAvailable() = true, want false for repeated stub calls")
	}
}

func TestVRAM_GetVRAMInfo_Good(t *testing.T) {
	info, err := GetVRAMInfo()
	if err == nil {
		t.Fatal("GetVRAMInfo() error = nil, want unsupported-platform error")
	}
	if info != (VRAMInfo{}) {
		t.Errorf("info = %+v, want zero VRAMInfo", info)
	}
}

func TestVRAM_GetVRAMInfo_Bad(t *testing.T) {
	info, err := GetVRAMInfo()
	if err == nil {
		t.Fatal("GetVRAMInfo() error = nil, want error")
	}
	if !strings.Contains(err.Error(), "not available on this platform") {
		t.Errorf("err = %v, want platform unavailable message", err)
	}
	if info.Total != 0 || info.Used != 0 || info.Free != 0 {
		t.Errorf("info = %+v, want all fields zero", info)
	}
}

func TestVRAM_GetVRAMInfo_Ugly(t *testing.T) {
	first, firstErr := GetVRAMInfo()
	second, secondErr := GetVRAMInfo()
	if firstErr == nil || secondErr == nil {
		t.Fatalf("errors = %v, %v; want both non-nil", firstErr, secondErr)
	}
	if first != second {
		t.Fatalf("GetVRAMInfo() changed from %+v to %+v", first, second)
	}
}
