# go-rocm

AMD ROCm GPU inference for Linux via a managed llama-server subprocess. Implements the `inference.Backend` and `inference.TextModel` interfaces from go-inference for AMD RDNA 3+ GPUs (validated on RX 7800 XT with ROCm 7.2). Uses llama-server's OpenAI-compatible streaming API rather than direct HIP CGO bindings, giving access to 50+ GGUF model architectures with GPU crash isolation. Includes a GGUF v2/v3 binary metadata parser, sysfs VRAM monitoring, and model discovery. Platform-restricted: `linux/amd64` only; a safe stub compiles everywhere else.

**Module**: `forge.lthn.ai/core/go-rocm`
**Licence**: EUPL-1.2
**Language**: Go 1.25

## Quick Start

```go
import (
    "forge.lthn.ai/core/go-inference"
    _ "forge.lthn.ai/core/go-rocm"  // registers "rocm" backend via init()
)

// Requires llama-server compiled with HIP/ROCm on PATH
model, err := inference.LoadModel("/path/to/model.gguf")
defer model.Close()

for tok := range model.Generate(ctx, "Hello", inference.WithMaxTokens(256)) {
    fmt.Print(tok.Text)
}
```

## Documentation

- [Architecture](docs/architecture.md) — subprocess design rationale, inference flow, server lifecycle, GGUF parser, VRAM monitoring
- [Development Guide](docs/development.md) — prerequisites (ROCm, llama-server build), test commands, benchmarks
- [Project History](docs/history.md) — completed phases, commit hashes, known limitations

## Build & Test

```bash
go test ./...                             # unit tests, no GPU required
go test -tags rocm ./...                  # integration tests and benchmarks (GPU required)
go build ./...
```

## Licence

European Union Public Licence 1.2 — see [LICENCE](LICENCE) for details.
