//go:build linux && amd64 && rocm_legacy_server

package rocm

import core "dappco.re/go"

func Example_legacyPortAllocatorNextAvailablePort() {
	p, err := newDeterministicPortAllocator(0, 1).NextAvailablePort()
	core.Println(p, err != nil) /* Output: 0 true */
}
func Example_legacyOutputCaptureWrite() {
	c := newProcessOutputCapture(8)
	n, _ := c.Write([]byte("ok"))
	core.Println(n) /* Output: 2 */
}
func Example_legacyOutputCaptureSummary() {
	c := newProcessOutputCapture(8)
	c.Write([]byte(" ok "))
	core.Println(c.Summary()) /* Output: ok */
}
