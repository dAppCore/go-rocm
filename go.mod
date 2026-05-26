module dappco.re/go/rocm/workspace

go 1.26.0

require dappco.re/go/rocm v0.9.0

require (
	dappco.re/go v0.10.3 // indirect
	dappco.re/go/cgo v0.11.1 // indirect
	dappco.re/go/inference v0.10.0 // indirect
)

replace dappco.re/go => ./external/go

replace dappco.re/go/cgo => ./external/go-cgo/go

replace dappco.re/go/inference => ./external/go-inference/go

replace dappco.re/go/rocm => ./go
