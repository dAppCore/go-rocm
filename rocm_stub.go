//go:build !linux || !amd64

package rocm

import coreerr "dappco.re/go/core/log"

// ROCmAvailable reports whether ROCm GPU inference is available.
// Returns false on non-Linux or non-amd64 platforms.
func ROCmAvailable() bool { return false }

// GetVRAMInfo is not available on non-Linux/non-amd64 platforms.
func GetVRAMInfo() (VRAMInfo, error) {
	return VRAMInfo{}, coreerr.E("rocm.GetVRAMInfo", "VRAM monitoring not available on this platform", nil)
}
