//go:build linux && amd64

package rocm

import core "dappco.re/go"

func ExamplePortAllocator_NextAvailablePort() {
	p, err := newDeterministicPortAllocator(0, 1).NextAvailablePort()
	core.Println(p, err != nil) /* Output: 0 true */
}
func ExampleOutputCapture_Write() {
	c := newProcessOutputCapture(8)
	n, _ := c.Write([]byte("ok"))
	core.Println(n) /* Output: 2 */
}
func ExampleOutputCapture_Summary() {
	c := newProcessOutputCapture(8)
	c.Write([]byte(" ok "))
	core.Println(c.Summary()) /* Output: ok */
}
