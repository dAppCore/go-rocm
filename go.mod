module forge.lthn.ai/core/go-rocm

go 1.25.5

require forge.lthn.ai/core/go-inference v0.0.0

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/stretchr/testify v1.11.1
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace forge.lthn.ai/core/go-inference => ../go-inference
