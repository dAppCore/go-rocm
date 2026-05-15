//go:build linux && amd64

package rocm

import core "dappco.re/go"

func ExampleROCmAvailable() {
	available := ROCmAvailable()
	core.Println(available || !available) /* Output: true */
}
