# go-rocm

Native AMD ROCm backend for the Core Go inference stack. The default package surface is now the ROCm sibling of `go-mlx`: it implements the shared `go-inference` contracts, exposes model-fit planning, probing, benchmarking, dataset eval hooks, tokenizer/adapter surfaces, GGUF metadata inspection, sysfs VRAM monitoring, and model discovery without routing inference through an OpenAI-compatible HTTP server.

The former managed `llama-server` implementation is retained only behind the explicit `rocm_legacy_server` build tag while the native HIP loader/kernels are filled in on ROCm hardware. Platform-restricted runtime code is `linux/amd64`; safe stubs compile everywhere else.

**Module**: `dappco.re/go/rocm`
**Licence**: EUPL-1.2
**Language**: Go 1.26

## Quick Start

```go
import (
    "dappco.re/go/inference"
    _ "dappco.re/go/rocm" // registers "rocm" backend via init()
)

model, err := inference.LoadModel("/path/to/model.gguf")
defer model.Close()

for tok := range model.Generate(ctx, "Hello", inference.WithMaxTokens(256)) {
    fmt.Print(tok.Text)
}
```

Until the HIP runtime loader lands, `Available()` stays false for the default native backend rather than silently falling back to a subprocess. Use `-tags rocm_legacy_server` only when intentionally testing the old server path.

## Documentation

- [Architecture](docs/architecture.md) — native backend direction, GGUF parser, VRAM monitoring, and legacy server notes
- [Development Guide](docs/development.md) — prerequisites, test commands, benchmarks
- [Project History](docs/history.md) — completed phases, commit hashes, known limitations

## Build & Test

```bash
go test ./...                             # unit tests, no GPU required
GOOS=linux GOARCH=amd64 go test ./...     # compile Linux native surface
go test -tags rocm_legacy_server ./...    # legacy subprocess path
```

## Licence

European Union Public Licence 1.2 — see [LICENCE](LICENCE) for details.
