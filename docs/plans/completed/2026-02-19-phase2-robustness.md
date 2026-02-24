# Phase 2: Robustness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make go-rocm resilient to server crashes, port conflicts, and add VRAM monitoring and concurrency verification.

**Architecture:** Five features layered onto existing server/model code: (1) `server.alive()` method + pre-flight check in Generate/Chat, (2) retry loop in `startServer()` with port re-selection, (3) VRAM monitoring via sysfs with dGPU auto-detection, (4-5) integration tests for graceful shutdown and concurrent requests.

**Tech Stack:** Go, sysfs (`/sys/class/drm/`), testify, `//go:build rocm` integration tests

---

### Task 1: Server Crash Detection

Add `alive()` method to `server` and pre-flight checks in `Generate`/`Chat`. If the server process has exited, return an error immediately rather than making HTTP calls that will fail.

**Files:**
- Modify: `server.go:20-26` (add alive method)
- Modify: `model.go:24-49,53-87` (add alive check to Generate and Chat)
- Modify: `server_test.go` (add alive tests)

**Step 1: Write the failing tests**

Add to `server_test.go`:

```go
func TestServerAlive_Running(t *testing.T) {
	s := &server{exited: make(chan struct{})}
	assert.True(t, s.alive())
}

func TestServerAlive_Exited(t *testing.T) {
	exited := make(chan struct{})
	close(exited)
	s := &server{exited: exited, exitErr: fmt.Errorf("process killed")}
	assert.False(t, s.alive())
}

func TestGenerate_ServerDead(t *testing.T) {
	exited := make(chan struct{})
	close(exited)
	s := &server{
		exited:  exited,
		exitErr: fmt.Errorf("process killed"),
	}
	m := &rocmModel{srv: s}

	var count int
	for range m.Generate(context.Background(), "hello") {
		count++
	}
	assert.Equal(t, 0, count)
	assert.ErrorContains(t, m.Err(), "server has exited")
}

func TestChat_ServerDead(t *testing.T) {
	exited := make(chan struct{})
	close(exited)
	s := &server{
		exited:  exited,
		exitErr: fmt.Errorf("process killed"),
	}
	m := &rocmModel{srv: s}

	msgs := []inference.Message{{Role: "user", Content: "hello"}}
	var count int
	for range m.Chat(context.Background(), msgs) {
		count++
	}
	assert.Equal(t, 0, count)
	assert.ErrorContains(t, m.Err(), "server has exited")
}
```

Add `"context"` and `"fmt"` to the imports in `server_test.go`. Add `"forge.lthn.ai/core/go-inference"` import for the Chat test.

**Step 2: Run tests to verify they fail**

Run: `go test -run "TestServerAlive|TestGenerate_ServerDead|TestChat_ServerDead" -v`
Expected: FAIL — `alive()` method doesn't exist, Generate/Chat don't check alive

**Step 3: Implement alive() method**

Add to `server.go` after the `server` struct definition (after line 26):

```go
// alive reports whether the llama-server process is still running.
func (s *server) alive() bool {
	select {
	case <-s.exited:
		return false
	default:
		return true
	}
}
```

**Step 4: Add alive check to Generate and Chat**

In `model.go`, add this check at the start of `Generate` (before the `cfg :=` line):

```go
if !m.srv.alive() {
	m.mu.Lock()
	m.lastErr = fmt.Errorf("rocm: server has exited: %w", m.srv.exitErr)
	m.mu.Unlock()
	return func(yield func(inference.Token) bool) {}
}
```

Add the identical check at the start of `Chat` (before the `cfg :=` line).

Add `"fmt"` to the imports in `model.go`.

**Step 5: Run tests to verify they pass**

Run: `go test -run "TestServerAlive|TestGenerate_ServerDead|TestChat_ServerDead" -v`
Expected: PASS (4 tests)

Run: `go test ./...`
Expected: All tests PASS (existing + new)

**Step 6: Commit**

```bash
git add server.go model.go server_test.go
git commit -m "feat: detect server crash before Generate/Chat calls"
```

---

### Task 2: Port Conflict Retry

Move `freePort()` into `startServer()` and add a retry loop. If the process exits during startup (e.g. port already taken), pick a new port and retry up to 3 times.

**Files:**
- Modify: `server.go:72-117` (restructure startServer with retry loop, remove port param)
- Modify: `backend.go:40-46` (remove freePort call, update startServer call)
- Modify: `server_test.go` (add retry test)

**Step 1: Write the failing test**

Add to `server_test.go`:

```go
func TestStartServer_RetriesOnProcessExit(t *testing.T) {
	// /bin/false starts successfully but exits immediately with code 1.
	// startServer should retry up to 3 times, then fail.
	_, err := startServer("/bin/false", "/nonexistent/model.gguf", 999, 0)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed after 3 attempts")
}
```

**Step 2: Run test to verify it fails**

Run: `go test -run TestStartServer_RetriesOnProcessExit -v`
Expected: FAIL — startServer has wrong signature (currently takes port param)

**Step 3: Restructure startServer with retry loop**

Replace the entire `startServer` function in `server.go` with:

```go
// startServer spawns llama-server and waits for it to become ready.
// It selects a free port automatically, retrying up to 3 times if the
// process exits during startup (e.g. port conflict).
func startServer(binary, modelPath string, gpuLayers, ctxSize int) (*server, error) {
	if gpuLayers < 0 {
		gpuLayers = 999
	}

	const maxAttempts = 3
	var lastErr error

	for attempt := range maxAttempts {
		port, err := freePort()
		if err != nil {
			return nil, fmt.Errorf("rocm: find free port: %w", err)
		}

		args := []string{
			"--model", modelPath,
			"--host", "127.0.0.1",
			"--port", strconv.Itoa(port),
			"--n-gpu-layers", strconv.Itoa(gpuLayers),
		}
		if ctxSize > 0 {
			args = append(args, "--ctx-size", strconv.Itoa(ctxSize))
		}

		cmd := exec.Command(binary, args...)
		cmd.Env = serverEnv()

		if err := cmd.Start(); err != nil {
			return nil, fmt.Errorf("start llama-server: %w", err)
		}

		s := &server{
			cmd:    cmd,
			port:   port,
			client: llamacpp.NewClient(fmt.Sprintf("http://127.0.0.1:%d", port)),
			exited: make(chan struct{}),
		}

		go func() {
			s.exitErr = cmd.Wait()
			close(s.exited)
		}()

		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		err = s.waitReady(ctx)
		cancel()
		if err == nil {
			return s, nil
		}

		_ = s.stop()
		lastErr = fmt.Errorf("attempt %d: %w", attempt+1, err)
	}

	return nil, fmt.Errorf("rocm: server failed after %d attempts: %w", maxAttempts, lastErr)
}
```

**Step 4: Update LoadModel in backend.go**

Replace `LoadModel` in `backend.go` — remove the `freePort()` call and the `port` parameter to `startServer`:

```go
func (b *rocmBackend) LoadModel(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
	cfg := inference.ApplyLoadOpts(opts)

	binary, err := findLlamaServer()
	if err != nil {
		return nil, err
	}

	srv, err := startServer(binary, path, cfg.GPULayers, cfg.ContextLen)
	if err != nil {
		return nil, err
	}

	return &rocmModel{
		srv:       srv,
		modelType: guessModelType(path),
	}, nil
}
```

Remove the `"fmt"` import from `backend.go` (no longer needed — the freePort error wrapping is gone). Keep `"os"`, `"path/filepath"`, `"strings"`, and `"forge.lthn.ai/core/go-inference"`.

**Step 5: Run tests to verify they pass**

Run: `go test -run TestStartServer_RetriesOnProcessExit -v -timeout 30s`
Expected: PASS — `/bin/false` exits immediately, waitReady detects exit via `<-s.exited`, retries 3 times, test completes in < 2 seconds.

Run: `go test ./...`
Expected: All tests PASS

Run: `go vet ./...`
Expected: Clean

**Step 6: Commit**

```bash
git add server.go backend.go server_test.go
git commit -m "feat: retry port selection in startServer on process failure"
```

---

### Task 3: VRAM Monitoring

Read AMD GPU VRAM usage from sysfs. Auto-detect the dGPU by selecting the card with the largest VRAM total (avoids hardcoding card numbers — card0 is the iGPU on this machine, card1 is the dGPU).

**Files:**
- Modify: `rocm.go` (add VRAMInfo type definition — no build tags)
- Create: `vram.go` (GetVRAMInfo implementation — linux && amd64)
- Modify: `rocm_stub.go` (add GetVRAMInfo stub)
- Create: `vram_test.go` (unit tests + real hardware test)

**Step 1: Write the failing tests**

Create `vram_test.go`:

```go
//go:build linux && amd64

package rocm

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReadSysfsUint64(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_value")
	require.NoError(t, os.WriteFile(path, []byte("17163091968\n"), 0644))

	val, err := readSysfsUint64(path)
	require.NoError(t, err)
	assert.Equal(t, uint64(17163091968), val)
}

func TestReadSysfsUint64_NotFound(t *testing.T) {
	_, err := readSysfsUint64("/nonexistent/path")
	assert.Error(t, err)
}

func TestGetVRAMInfo(t *testing.T) {
	info, err := GetVRAMInfo()
	if err != nil {
		t.Skipf("no VRAM sysfs info available: %v", err)
	}

	// On this machine, the dGPU (RX 7800 XT) has ~16GB VRAM.
	assert.Greater(t, info.Total, uint64(8*1024*1024*1024), "expected dGPU with >8GB VRAM")
	assert.Greater(t, info.Used, uint64(0), "expected some VRAM in use")
	assert.Equal(t, info.Total-info.Used, info.Free, "Free should equal Total-Used")
}
```

**Step 2: Run tests to verify they fail**

Run: `go test -run "TestReadSysfs|TestGetVRAMInfo" -v`
Expected: FAIL — `readSysfsUint64` and `GetVRAMInfo` don't exist

**Step 3: Add VRAMInfo type to rocm.go**

Append to `rocm.go` (after the package doc comment, before the closing line):

```go

// VRAMInfo reports GPU video memory usage in bytes.
type VRAMInfo struct {
	Total uint64
	Used  uint64
	Free  uint64
}
```

**Step 4: Create vram.go implementation**

Create `vram.go`:

```go
//go:build linux && amd64

package rocm

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// GetVRAMInfo reads VRAM usage for the discrete GPU from sysfs.
// It identifies the dGPU by selecting the card with the largest VRAM total,
// which avoids hardcoding card numbers (e.g. card0=iGPU, card1=dGPU on Ryzen).
func GetVRAMInfo() (VRAMInfo, error) {
	cards, err := filepath.Glob("/sys/class/drm/card[0-9]*/device/mem_info_vram_total")
	if err != nil {
		return VRAMInfo{}, fmt.Errorf("rocm: glob vram sysfs: %w", err)
	}
	if len(cards) == 0 {
		return VRAMInfo{}, fmt.Errorf("rocm: no GPU VRAM info found in sysfs")
	}

	var bestDir string
	var bestTotal uint64

	for _, totalPath := range cards {
		total, err := readSysfsUint64(totalPath)
		if err != nil {
			continue
		}
		if total > bestTotal {
			bestTotal = total
			bestDir = filepath.Dir(totalPath)
		}
	}

	if bestDir == "" {
		return VRAMInfo{}, fmt.Errorf("rocm: no readable VRAM sysfs entries")
	}

	used, err := readSysfsUint64(filepath.Join(bestDir, "mem_info_vram_used"))
	if err != nil {
		return VRAMInfo{}, fmt.Errorf("rocm: read vram used: %w", err)
	}

	return VRAMInfo{
		Total: bestTotal,
		Used:  used,
		Free:  bestTotal - used,
	}, nil
}

func readSysfsUint64(path string) (uint64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
}
```

**Step 5: Add GetVRAMInfo stub to rocm_stub.go**

Add to `rocm_stub.go`:

```go
// GetVRAMInfo is not available on non-Linux/non-amd64 platforms.
func GetVRAMInfo() (VRAMInfo, error) {
	return VRAMInfo{}, fmt.Errorf("rocm: VRAM monitoring not available on this platform")
}
```

Add `"fmt"` to imports in `rocm_stub.go`.

**Step 6: Run tests to verify they pass**

Run: `go test -run "TestReadSysfs|TestGetVRAMInfo" -v`
Expected: PASS (3 tests — 2 readSysfsUint64 unit tests + 1 GetVRAMInfo on real hardware)

Run: `go test ./...`
Expected: All tests PASS

Run: `go vet ./...`
Expected: Clean

**Step 7: Commit**

```bash
git add rocm.go vram.go vram_test.go rocm_stub.go
git commit -m "feat: VRAM monitoring via sysfs with dGPU auto-detection"
```

---

### Task 4: Integration Tests (Graceful Shutdown + Concurrent Requests)

Two new `//go:build rocm` integration tests:
1. Cancel mid-stream then generate again on the same model (server survives cancellation)
2. Three goroutines calling Generate() simultaneously (no panics, no deadlocks)

**Files:**
- Modify: `rocm_integration_test.go` (add 2 tests)

**Step 1: Add graceful shutdown test**

Add to `rocm_integration_test.go`:

```go
func TestROCm_GracefulShutdown(t *testing.T) {
	skipIfNoROCm(t)
	skipIfNoModel(t)

	b := &rocmBackend{}
	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	require.NoError(t, err)
	defer m.Close()

	// Cancel mid-stream.
	ctx1, cancel1 := context.WithCancel(context.Background())
	var count1 int
	for tok := range m.Generate(ctx1, "Write a long story about space exploration", inference.WithMaxTokens(256)) {
		_ = tok
		count1++
		if count1 >= 5 {
			cancel1()
		}
	}
	t.Logf("First generation: %d tokens before cancel", count1)

	// Generate again on the same model — server should still be alive.
	ctx2, cancel2 := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel2()

	var count2 int
	for tok := range m.Generate(ctx2, "The capital of France is", inference.WithMaxTokens(16)) {
		_ = tok
		count2++
	}

	require.NoError(t, m.Err())
	assert.Greater(t, count2, 0, "expected tokens from second generation after cancel")
	t.Logf("Second generation: %d tokens", count2)
}
```

**Step 2: Add concurrent requests test**

Add to `rocm_integration_test.go`:

```go
func TestROCm_ConcurrentRequests(t *testing.T) {
	skipIfNoROCm(t)
	skipIfNoModel(t)

	b := &rocmBackend{}
	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	require.NoError(t, err)
	defer m.Close()

	const numGoroutines = 3
	results := make([]string, numGoroutines)

	prompts := []string{
		"The capital of France is",
		"The capital of Germany is",
		"The capital of Italy is",
	}

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := range numGoroutines {
		go func(idx int) {
			defer wg.Done()

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			var sb strings.Builder
			for tok := range m.Generate(ctx, prompts[idx], inference.WithMaxTokens(16)) {
				sb.WriteString(tok.Text)
			}
			results[idx] = sb.String()
		}(i)
	}

	wg.Wait()

	for i, result := range results {
		t.Logf("Goroutine %d: %s", i, result)
		assert.NotEmpty(t, result, "goroutine %d produced no output", i)
	}
}
```

Add `"strings"` and `"sync"` to the import block in `rocm_integration_test.go`.

**Step 3: Run integration tests**

Run: `go test -tags rocm -run "TestROCm_GracefulShutdown|TestROCm_ConcurrentRequests" -v -timeout 120s`
Expected: PASS — both tests produce output and complete without panics.

Run: `go test -tags rocm -v -timeout 120s`
Expected: All 5 integration tests PASS (3 existing + 2 new).

**Step 4: Commit**

```bash
git add rocm_integration_test.go
git commit -m "test: graceful shutdown and concurrent request integration tests"
```

---

### Task 5: Update TODO.md and FINDINGS.md

Mark Phase 2 items as complete. Document concurrency findings.

**Files:**
- Modify: `TODO.md` (mark Phase 2 items `[x]`)
- Modify: `FINDINGS.md` (add concurrent requests findings)

**Step 1: Update TODO.md**

Mark all Phase 2 items as `[x]` with commit references from the tasks above.

**Step 2: Add concurrency findings to FINDINGS.md**

Add a new section to FINDINGS.md:

```markdown
## Concurrent Requests (Phase 2)

Tested 3 goroutines calling Generate() simultaneously on the same model (Gemma3-1B, llama-server with default settings). All 3 received output. llama-server handles concurrency via its slot system — default is 1 slot, so requests are serialised server-side.

For true parallel inference, use `--parallel N` flag in llama-server (not yet configurable via go-rocm). VRAM cost scales with number of slots.
```

**Step 3: Commit**

```bash
git add TODO.md FINDINGS.md
git commit -m "docs: Phase 2 complete — robustness features implemented"
```

---

## Summary

| Task | What | Files | Test Type |
|------|------|-------|-----------|
| 1 | Server crash detection | server.go, model.go, server_test.go | Unit |
| 2 | Port conflict retry | server.go, backend.go, server_test.go | Unit |
| 3 | VRAM monitoring | rocm.go, vram.go, vram_test.go, rocm_stub.go | Unit + hardware |
| 4 | Graceful shutdown + concurrent | rocm_integration_test.go | Integration (GPU) |
| 5 | Update docs | TODO.md, FINDINGS.md | N/A |
