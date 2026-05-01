//go:build !linux || !amd64

package rocm

import core "dappco.re/go"

func ExampleROCmAvailable() { core.Println(ROCmAvailable()) /* Output: false */ }
func ExampleGetVRAMInfo()   { _, err := GetVRAMInfo(); core.Println(err != nil) /* Output: true */ }
