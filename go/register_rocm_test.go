//go:build linux && amd64

package rocm

import (
	core "dappco.re/go"
	"testing"
)

func TestRegisterRocm_ROCmAvailable_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	available := ROCmAvailable()
	core.AssertTrue(t, available)
	core.AssertEqual(t, available, ROCmAvailable())
}

func TestRegisterRocm_ROCmAvailable_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	available := ROCmAvailable()
	core.AssertNotEqual(t, false, available)
	core.AssertEqual(t, "linux", "linux")
}

func TestRegisterRocm_ROCmAvailable_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	first := ROCmAvailable()
	second := ROCmAvailable()
	core.AssertEqual(t, first, second)
}
