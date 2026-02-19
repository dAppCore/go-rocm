//go:build !linux || !amd64

package rocm

// ROCmAvailable reports whether ROCm GPU inference is available.
// Returns false on non-Linux or non-amd64 platforms.
func ROCmAvailable() bool { return false }
