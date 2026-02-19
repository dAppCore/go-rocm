//go:build linux && amd64

package rocm

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// GetVRAMInfo reads VRAM usage for the discrete GPU from sysfs.
// It identifies the dGPU by selecting the card with the largest VRAM total,
// which avoids hardcoding card numbers (e.g. card0=iGPU, card1=dGPU on Ryzen).
//
// Note: total and used are read non-atomically from sysfs; transient
// inconsistencies are possible under heavy allocation churn.
func GetVRAMInfo() (VRAMInfo, error) {
	cards, err := filepath.Glob("/sys/class/drm/card[0-9]*/device/mem_info_vram_total")
	if err != nil {
		return VRAMInfo{}, fmt.Errorf("rocm: glob vram sysfs: %w", err)
	}
	if len(cards) == 0 {
		return VRAMInfo{}, fmt.Errorf("rocm: no GPU VRAM info found in sysfs")
	}

	var bestDir string
	var bestTotal uint64

	for _, totalPath := range cards {
		total, err := readSysfsUint64(totalPath)
		if err != nil {
			continue
		}
		if total > bestTotal {
			bestTotal = total
			bestDir = filepath.Dir(totalPath)
		}
	}

	if bestDir == "" {
		return VRAMInfo{}, fmt.Errorf("rocm: no readable VRAM sysfs entries")
	}

	used, err := readSysfsUint64(filepath.Join(bestDir, "mem_info_vram_used"))
	if err != nil {
		return VRAMInfo{}, fmt.Errorf("rocm: read vram used: %w", err)
	}

	free := uint64(0)
	if bestTotal > used {
		free = bestTotal - used
	}

	return VRAMInfo{
		Total: bestTotal,
		Used:  used,
		Free:  free,
	}, nil
}

func readSysfsUint64(path string) (uint64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
}
