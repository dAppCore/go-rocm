//go:build rocm_legacy_server

package llamacpp

import core "dappco.re/go"

func ExampleNewClient() { core.Println(NewClient("http://example.test") != nil) /* Output: true */ }
func ExampleNewClientWithHTTPClient() {
	core.Println(NewClientWithHTTPClient("http://example.test", nil) != nil) /* Output: true */
}
func ExampleClient_Health() {
	err := NewClient("http://%zz").Health(nil)
	core.Println(err != nil) /* Output: true */
}
