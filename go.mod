module dappco.re/go/rocm/workspace

go 1.26.0

require (
	dappco.re/go v0.9.0
	dappco.re/go/inference v0.9.0
	dappco.re/go/rocm v0.9.0
)

replace dappco.re/go => ./external/go

replace dappco.re/go/inference => ./external/go-inference/go

replace dappco.re/go/rocm => ./go
