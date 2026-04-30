//go:build linux && amd64

package rocm

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadSysfsUint64(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_value")
	if err := os.WriteFile(path, []byte("17163091968\n"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	val, err := readSysfsUint64(path)
	if err != nil {
		t.Fatalf("readSysfsUint64: %v", err)
	}
	if val != uint64(17163091968) {
		t.Errorf("readSysfsUint64 = %d, want 17163091968", val)
	}
}

func TestReadSysfsUint64_NotFound(t *testing.T) {
	if _, err := readSysfsUint64("/nonexistent/path"); err == nil {
		t.Error("expected error for missing path, got nil")
	}
}

func TestReadSysfsUint64_InvalidContent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad_value")
	if err := os.WriteFile(path, []byte("not-a-number\n"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	if _, err := readSysfsUint64(path); err == nil {
		t.Error("expected error for non-numeric content, got nil")
	}
}

func TestReadSysfsUint64_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty_value")
	if err := os.WriteFile(path, []byte(""), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	if _, err := readSysfsUint64(path); err == nil {
		t.Error("expected error for empty file, got nil")
	}
}

func TestGetVRAMInfo(t *testing.T) {
	info, err := GetVRAMInfo()
	if err != nil {
		t.Skipf("no VRAM sysfs info available: %v", err)
	}

	// On this machine, the dGPU (RX 7800 XT) has ~16GB VRAM.
	if info.Total <= uint64(8*1024*1024*1024) {
		t.Errorf("Total = %d, expected dGPU with >8GB VRAM", info.Total)
	}
	if info.Used == 0 {
		t.Error("Used = 0, expected some VRAM in use")
	}
	if info.Total-info.Used != info.Free {
		t.Errorf("Free = %d, want Total-Used = %d", info.Free, info.Total-info.Used)
	}
}
