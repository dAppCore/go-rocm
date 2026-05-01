//go:build linux && amd64

package rocm

import (
	// Note: strconv: numeric parsing of sysfs values; no core.ParseInt
	"strconv"

	core "dappco.re/go"
)

//	info, err := GetVRAMInfo()
//	fmt.Printf("%d MiB free\n", info.Free>>20)
//
// GetVRAMInfo reads VRAM usage for the discrete GPU from sysfs. It identifies
// the dGPU by selecting the card with the largest VRAM total, which avoids
// hardcoding card numbers (e.g. card0=iGPU, card1=dGPU on Ryzen).
//
// Note: total and used are read non-atomically from sysfs; transient
// inconsistencies are possible under heavy allocation churn.
func GetVRAMInfo() (
	VRAMInfo,
	error,
) {
	cards := core.PathGlob("/sys/class/drm/card[0-9]*/device/mem_info_vram_total")
	if len(cards) == 0 {
		return VRAMInfo{}, core.E("rocm.GetVRAMInfo", "no GPU VRAM info found in sysfs", nil)
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
			bestDir = core.PathDir(totalPath)
		}
	}

	if bestDir == "" {
		return VRAMInfo{}, core.E("rocm.GetVRAMInfo", "no readable VRAM sysfs entries", nil)
	}

	used, err := readSysfsUint64(core.PathJoin(bestDir, "mem_info_vram_used"))
	if err != nil {
		return VRAMInfo{}, core.E("rocm.GetVRAMInfo", "read vram used", err)
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

func readSysfsUint64(path string) (
	uint64,
	error,
) {
	dataResult := core.ReadFile(path)
	if !dataResult.OK {
		return 0, dataResult.Value.(error)
	}
	return strconv.ParseUint(core.Trim(string(dataResult.Value.([]byte))), 10, 64)
}
