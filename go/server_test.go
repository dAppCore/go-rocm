//go:build linux && amd64 && rocm_legacy_server

package rocm

import (
	core "dappco.re/go"
	"testing"
)

func TestServer_PortAllocator_NextAvailablePort_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	p, err := newDeterministicPortAllocator(41000, 2).NextAvailablePort()
	core.AssertNoError(t, err)
	core.AssertGreaterOrEqual(t, p, 41000)
}
func TestServer_PortAllocator_NextAvailablePort_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	p, err := newDeterministicPortAllocator(0, 2).NextAvailablePort()
	core.AssertError(t, err)
	core.AssertEqual(t, 0, p)
}
func TestServer_PortAllocator_NextAvailablePort_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	p, err := newDeterministicPortAllocator(65535, 1).NextAvailablePort()
	if err == nil {
		core.AssertEqual(t, 65535, p)
	}
}

func TestServer_OutputCapture_Write_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	c := newProcessOutputCapture(32)
	n, err := c.Write([]byte("ready"))
	core.AssertNoError(t, err)
	core.AssertEqual(t, 5, n)
}
func TestServer_OutputCapture_Write_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	c := newProcessOutputCapture(0)
	n, err := c.Write([]byte("ready"))
	core.AssertNoError(t, err)
	core.AssertEqual(t, 5, n)
}
func TestServer_OutputCapture_Write_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	c := newProcessOutputCapture(2)
	_, err := c.Write([]byte("ready"))
	core.AssertNoError(t, err)
	core.AssertEqual(t, "...dy", c.Summary())
}

func TestServer_OutputCapture_Summary_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	c := newProcessOutputCapture(32)
	c.Write([]byte(" a \n b "))
	core.AssertEqual(t, "a | b", c.Summary())
}
func TestServer_OutputCapture_Summary_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	c := newProcessOutputCapture(32)
	core.AssertEqual(t, "", c.Summary())
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestServer_OutputCapture_Summary_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	c := newProcessOutputCapture(3)
	c.Write([]byte("abcdef"))
	core.AssertContains(t, c.Summary(), "...")
}
