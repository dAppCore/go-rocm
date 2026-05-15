//go:build !linux || !amd64

package rocm

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func init() {
	inference.Register(unavailableROCmBackend{})
}

type unavailableROCmBackend struct{}

func (unavailableROCmBackend) Name() string { return "rocm" }
func (unavailableROCmBackend) Available() bool {
	return false
}
func (unavailableROCmBackend) LoadModel(string, ...inference.LoadOption) (inference.TextModel, error) {
	return nil, core.E("rocm.LoadModel", "native ROCm runtime is not available on this platform", nil)
}

//	if !ROCmAvailable() {
//	    fmt.Println("fall back to CPU or another backend")
//	}
//
// ROCmAvailable reports whether ROCm GPU inference is available.
// Returns false on non-Linux or non-amd64 platforms.
func ROCmAvailable() bool { return false }

//	_, err := GetVRAMInfo()
//	fmt.Println(err)
//
// GetVRAMInfo is not available on non-Linux/non-amd64 platforms.
func GetVRAMInfo() (
	VRAMInfo,
	error,
) {
	return VRAMInfo{}, core.E("rocm.GetVRAMInfo", "VRAM monitoring not available on this platform", nil)
}
