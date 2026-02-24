# Phase 1: Core Implementation Design

Approved 19 Feb 2026.

## Component Structure

4 files, layered bottom-up:

```
internal/llamacpp/client.go   <- HTTP + SSE (no inference types)
server.go                     <- process lifecycle (spawn, health, kill)
model.go                      <- inference.TextModel (wraps server + client)
backend.go                    <- inference.Backend (fill in stubs)
```

Each layer only knows about the one below it. `internal/llamacpp/` is a pure HTTP client with zero go-inference dependency — it speaks llama-server's OpenAI-compatible API and returns plain Go types.

## Data Flow

```
LoadModel("/path/to/model.gguf")
  -> backend.go: find llama-server binary, apply LoadConfig
  -> server.go:  spawn process (HIP_VISIBLE_DEVICES=0, random port)
                  poll GET /health until {"status":"ok"} (timeout 60s)
  -> model.go:   return &rocmModel{server, client}

Generate(ctx, "prompt")
  -> model.go:   build request body from GenerateConfig
  -> client.go:  POST /v1/completions, stream=true
                  parse SSE lines: data: {json}
                  yield Token{Text: chunk} via iter.Seq[Token]
  -> on ctx cancel: close HTTP response body (stops SSE)

Close()
  -> server.go:  SIGTERM -> wait 5s -> SIGKILL if needed
```

## Key Decisions

- **Server lifecycle**: Long-lived. Spawn on LoadModel(), reuse for all Generate/Chat calls, kill on Close().
- **Port selection**: `net.Listen(":0")` for kernel-assigned free port. Close listener, pass port to llama-server.
- **SSE parsing**: Hand-rolled with `bufio.Scanner`. llama-server's SSE format is simple (`data: {json}\n\n`), ~20 lines.
- **Health polling**: Simple loop, 100ms interval, 60s timeout. Returns error if exceeded.
- **Error handling**: `Err()` stores last error. Process death mid-stream detected via response body EOF + process state.
- **Thread safety**: `rocmModel` safe for concurrent calls (llama-server handles via slots). `lastErr` protected by mutex.
- **iGPU mitigation**: `HIP_VISIBLE_DEVICES=0` set in process env unconditionally.

## Testing

- **Unit tests** (no GPU): mock HTTP responses for SSE parsing in client.go
- **Integration test** (GPU + model): LoadModel -> Generate -> Close, `t.Skip()` when unavailable
- **Test model**: LEK-Gemma3-1B-layered-v2-Q5_K_M.gguf (smallest, fast load)
