module dappco.re/go/rocm

go 1.26.0

require (
	dappco.re/go/core/log v0.1.0
	forge.lthn.ai/core/go-inference v0.1.7
)

require dappco.re/go/core v0.8.0-alpha.1 // indirect

replace dappco.re/go/core => ../go

replace dappco.re/go/core/log => ../go-log

replace forge.lthn.ai/core/go-inference => ../go-inference
