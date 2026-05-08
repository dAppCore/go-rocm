//go:build rocm_legacy_server

package llamacpp

import core "dappco.re/go"

func ExampleClient_ChatComplete() {
	_, errFn := NewClient("http://%zz").ChatComplete(nil, ChatRequest{})
	core.Println(errFn() != nil) /* Output: true */
}
func ExampleClient_Complete() {
	_, errFn := NewClient("http://%zz").Complete(nil, CompletionRequest{})
	core.Println(errFn() != nil) /* Output: true */
}
