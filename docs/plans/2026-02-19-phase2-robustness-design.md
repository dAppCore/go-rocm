# Phase 2: Robustness Design

Approved 19 Feb 2026.

## 1. Graceful Shutdown (context cancellation)

Already works in Phase 1. Context cancellation closes the HTTP response body, stops SSE streaming, but leaves llama-server alive. Generate/Chat can be called again with a new context.

Only change: add an integration test that cancels mid-stream then generates again on the same model to verify the server survives.

## 2. Port Conflict Handling

Retry loop in startServer(): if the process fails (port taken), call freePort() again and retry up to 3 attempts.

## 3. Server Crash Recovery

Add server.alive() method (non-blocking check on exited channel). Generate/Chat check alive() before making HTTP calls. If dead, return error immediately. No auto-restart — consumer must Close() + LoadModel() again.

## 4. VRAM Monitoring

Read sysfs directly (no subprocess spawn):
- `/sys/class/drm/cardN/device/mem_info_vram_total`
- `/sys/class/drm/cardN/device/mem_info_vram_used`

Find dGPU by picking the card with the largest VRAM total (avoids hardcoding card numbers). On this machine: card0 = iGPU (2GB), card1 = dGPU (16GB).

Expose via:
```go
type VRAMInfo struct {
    Total uint64
    Used  uint64
    Free  uint64
}

func GetVRAMInfo() (VRAMInfo, error)
```

## 5. Concurrent Requests

Integration test only. 3 goroutines calling Generate() on the same model simultaneously. Verify all get results. Document concurrency limits in FINDINGS.md.

## Testing

- Tasks 1-3: unit tests (mock servers, process helpers)
- Task 4: unit test (sysfs on real hardware)
- Task 5: integration test (GPU + model, //go:build rocm)
