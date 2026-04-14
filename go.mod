module dappco.re/go/rocm

go 1.26.0

require (
	dappco.re/go/core/log v0.1.0
	forge.lthn.ai/core/go-inference v0.1.7
	github.com/stretchr/testify v1.11.1
)

require (
	dappco.re/go/core v0.8.0-alpha.1 // indirect
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace dappco.re/go/core => ../go

replace dappco.re/go/core/log => ../go-log

replace forge.lthn.ai/core/go-inference => ../go-inference
