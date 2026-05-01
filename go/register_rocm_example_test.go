//go:build linux && amd64

package rocm

import core "dappco.re/go"

func ExampleROCmAvailable() { core.Println(ROCmAvailable()) /* Output: true */ }
