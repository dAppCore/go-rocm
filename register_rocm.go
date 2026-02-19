//go:build linux && amd64

package rocm

import "forge.lthn.ai/core/go-inference"

func init() {
	inference.Register(&rocmBackend{})
}

// ROCmAvailable reports whether ROCm GPU inference is available.
func ROCmAvailable() bool { return true }
