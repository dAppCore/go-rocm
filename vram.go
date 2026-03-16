//go:build linux && amd64

package rocm

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"

	coreerr "forge.lthn.ai/core/go-log"
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
		return VRAMInfo{}, coreerr.E("rocm.GetVRAMInfo", "glob vram sysfs", err)
	}
	if len(cards) == 0 {
		return VRAMInfo{}, coreerr.E("rocm.GetVRAMInfo", "no GPU VRAM info found in sysfs", nil)
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
		return VRAMInfo{}, coreerr.E("rocm.GetVRAMInfo", "no readable VRAM sysfs entries", nil)
	}

	used, err := readSysfsUint64(filepath.Join(bestDir, "mem_info_vram_used"))
	if err != nil {
		return VRAMInfo{}, coreerr.E("rocm.GetVRAMInfo", "read vram used", err)
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
