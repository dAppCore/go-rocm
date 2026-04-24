module dappco.re/go/rocm

go 1.26.0

require (
	dappco.re/go/log v0.8.0-alpha.1
	dappco.re/go/inference v0.8.0-alpha.1
)

require dappco.re/go/core v0.8.0-alpha.1 // indirect

replace dappco.re/go/core => ../go

replace dappco.re/go/log => ../go-log

replace dappco.re/go/inference => ../go-inference
