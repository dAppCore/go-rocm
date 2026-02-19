//go:build linux && amd64

package rocm

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReadSysfsUint64(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_value")
	require.NoError(t, os.WriteFile(path, []byte("17163091968\n"), 0644))

	val, err := readSysfsUint64(path)
	require.NoError(t, err)
	assert.Equal(t, uint64(17163091968), val)
}

func TestReadSysfsUint64_NotFound(t *testing.T) {
	_, err := readSysfsUint64("/nonexistent/path")
	assert.Error(t, err)
}

func TestGetVRAMInfo(t *testing.T) {
	info, err := GetVRAMInfo()
	if err != nil {
		t.Skipf("no VRAM sysfs info available: %v", err)
	}

	// On this machine, the dGPU (RX 7800 XT) has ~16GB VRAM.
	assert.Greater(t, info.Total, uint64(8*1024*1024*1024), "expected dGPU with >8GB VRAM")
	assert.Greater(t, info.Used, uint64(0), "expected some VRAM in use")
	assert.Equal(t, info.Total-info.Used, info.Free, "Free should equal Total-Used")
}
