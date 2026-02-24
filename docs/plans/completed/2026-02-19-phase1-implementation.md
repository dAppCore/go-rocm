# Phase 1: Core Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the full inference.Backend and inference.TextModel interfaces for AMD ROCm GPUs using llama-server as the subprocess engine.

**Architecture:** Bottom-up layered design. `internal/llamacpp/` is a pure HTTP+SSE client with no go-inference dependency. `server.go` manages the llama-server process lifecycle. `model.go` implements inference.TextModel by wrapping the server and client. `backend.go` implements inference.Backend by wiring everything together.

**Tech Stack:** Go 1.25, llama-server (llama.cpp with HIP/ROCm), OpenAI-compatible API, SSE streaming, testify for assertions, httptest for mocking.

**Critical environment note:** `HIP_VISIBLE_DEVICES=0` MUST be set when spawning llama-server. The Ryzen 9 9950X iGPU (Device 1) reports 100GB free memory (system RAM) and crashes llama-server when it tries to split tensors across devices.

---

### Task 1: Health Check Client

**Files:**
- Create: `internal/llamacpp/health.go`
- Create: `internal/llamacpp/health_test.go`

**Step 1: Create directory and write the failing test**

Create `internal/llamacpp/` directory, then write the test:

```go
// internal/llamacpp/health_test.go
package llamacpp

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHealth_OK(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/health", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"ok"}`))
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	err := c.Health(context.Background())
	require.NoError(t, err)
}

func TestHealth_NotReady(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"loading model"}`))
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	err := c.Health(context.Background())
	assert.ErrorContains(t, err, "not ready")
}

func TestHealth_ServerDown(t *testing.T) {
	c := NewClient("http://127.0.0.1:1") // nothing listening
	err := c.Health(context.Background())
	assert.Error(t, err)
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./internal/llamacpp/ -run TestHealth -v`
Expected: FAIL — `NewClient` and `Health` not defined.

**Step 3: Write minimal implementation**

```go
// internal/llamacpp/health.go
package llamacpp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// Client communicates with a llama-server instance.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a client for the llama-server at the given base URL.
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{},
	}
}

type healthResponse struct {
	Status string `json:"status"`
}

// Health checks whether the llama-server is ready to accept requests.
func (c *Client) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("llamacpp: health returned %d", resp.StatusCode)
	}
	var h healthResponse
	if err := json.NewDecoder(resp.Body).Decode(&h); err != nil {
		return fmt.Errorf("llamacpp: health decode: %w", err)
	}
	if h.Status != "ok" {
		return fmt.Errorf("llamacpp: server not ready (status: %s)", h.Status)
	}
	return nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./internal/llamacpp/ -run TestHealth -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add internal/llamacpp/health.go internal/llamacpp/health_test.go
git commit -m "feat: llamacpp health check client"
```

---

### Task 2: SSE Streaming + Chat Completions

**Files:**
- Create: `internal/llamacpp/client.go`
- Create: `internal/llamacpp/client_test.go`

**Step 1: Write the failing tests**

```go
// internal/llamacpp/client_test.go
package llamacpp

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatComplete_Streaming(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		flusher, ok := w.(http.Flusher)
		require.True(t, ok)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		chunks := []string{
			`{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}`,
			`{"choices":[{"delta":{"content":" world"},"finish_reason":null}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	chunks, errFn := c.ChatComplete(context.Background(), ChatRequest{
		Messages:  []ChatMessage{{Role: "user", Content: "Hi"}},
		MaxTokens: 32,
	})

	var result []string
	for text := range chunks {
		result = append(result, text)
	}

	require.NoError(t, errFn())
	assert.Equal(t, []string{"Hello", " world"}, result)
}

func TestChatComplete_EmptyResponse(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	chunks, errFn := c.ChatComplete(context.Background(), ChatRequest{
		Messages:  []ChatMessage{{Role: "user", Content: "Hi"}},
		MaxTokens: 1,
	})

	var result []string
	for text := range chunks {
		result = append(result, text)
	}

	require.NoError(t, errFn())
	assert.Empty(t, result)
}

func TestChatComplete_HTTPError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	chunks, errFn := c.ChatComplete(context.Background(), ChatRequest{
		Messages: []ChatMessage{{Role: "user", Content: "Hi"}},
	})

	for range chunks {
		t.Fatal("expected no chunks")
	}

	assert.ErrorContains(t, errFn(), "500")
}

func TestChatComplete_ContextCancelled(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, _ := w.(http.Flusher)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Send one chunk then block (context will cancel)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()

		// Block until client disconnects
		<-r.Context().Done()
	}))
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	c := NewClient(ts.URL)
	chunks, _ := c.ChatComplete(ctx, ChatRequest{
		Messages: []ChatMessage{{Role: "user", Content: "Hi"}},
	})

	var got int
	for range chunks {
		got++
		cancel() // Cancel after first chunk
	}
	assert.Equal(t, 1, got)
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./internal/llamacpp/ -run TestChat -v`
Expected: FAIL — `ChatComplete`, `ChatRequest`, `ChatMessage` not defined.

**Step 3: Write the implementation**

```go
// internal/llamacpp/client.go
package llamacpp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
)

// ChatMessage is a single message in a conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest is the request body for /v1/chat/completions.
type ChatRequest struct {
	Messages      []ChatMessage `json:"messages"`
	MaxTokens     int           `json:"max_tokens,omitempty"`
	Temperature   float32       `json:"temperature"`
	TopK          int           `json:"top_k,omitempty"`
	TopP          float32       `json:"top_p,omitempty"`
	RepeatPenalty float32       `json:"repeat_penalty,omitempty"`
	Stream        bool          `json:"stream"`
}

// CompletionRequest is the request body for /v1/completions.
type CompletionRequest struct {
	Prompt        string  `json:"prompt"`
	MaxTokens     int     `json:"max_tokens,omitempty"`
	Temperature   float32 `json:"temperature"`
	TopK          int     `json:"top_k,omitempty"`
	TopP          float32 `json:"top_p,omitempty"`
	RepeatPenalty float32 `json:"repeat_penalty,omitempty"`
	Stream        bool    `json:"stream"`
}

type chatChunkResponse struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

type completionChunkResponse struct {
	Choices []struct {
		Text         string  `json:"text"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

// ChatComplete sends a streaming chat completion request.
// Returns an iterator of text chunks and a function to retrieve any
// error that occurred during streaming. Call errFn after iteration.
func (c *Client) ChatComplete(ctx context.Context, req ChatRequest) (iter.Seq[string], func() error) {
	req.Stream = true
	body, err := json.Marshal(req)
	if err != nil {
		return noChunks, func() error { return fmt.Errorf("llamacpp: marshal: %w", err) }
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return noChunks, func() error { return err }
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return noChunks, func() error { return err }
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return noChunks, func() error {
			return fmt.Errorf("llamacpp: chat request returned %d", resp.StatusCode)
		}
	}

	var streamErr error
	seq := func(yield func(string) bool) {
		defer resp.Body.Close()
		for payload := range parseSSE(resp.Body, &streamErr) {
			var chunk chatChunkResponse
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				streamErr = fmt.Errorf("llamacpp: parse chunk: %w", err)
				return
			}
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				if !yield(chunk.Choices[0].Delta.Content) {
					return
				}
			}
		}
	}
	return seq, func() error { return streamErr }
}

// Complete sends a streaming text completion request.
// Returns an iterator of text chunks and a function to retrieve any
// error that occurred during streaming.
func (c *Client) Complete(ctx context.Context, req CompletionRequest) (iter.Seq[string], func() error) {
	req.Stream = true
	body, err := json.Marshal(req)
	if err != nil {
		return noChunks, func() error { return fmt.Errorf("llamacpp: marshal: %w", err) }
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/completions", bytes.NewReader(body))
	if err != nil {
		return noChunks, func() error { return err }
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return noChunks, func() error { return err }
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return noChunks, func() error {
			return fmt.Errorf("llamacpp: completion request returned %d", resp.StatusCode)
		}
	}

	var streamErr error
	seq := func(yield func(string) bool) {
		defer resp.Body.Close()
		for payload := range parseSSE(resp.Body, &streamErr) {
			var chunk completionChunkResponse
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				streamErr = fmt.Errorf("llamacpp: parse chunk: %w", err)
				return
			}
			if len(chunk.Choices) > 0 && chunk.Choices[0].Text != "" {
				if !yield(chunk.Choices[0].Text) {
					return
				}
			}
		}
	}
	return seq, func() error { return streamErr }
}

// parseSSE reads SSE data lines from r, yielding each data payload.
// Stops on "data: [DONE]" or reader exhaustion. Sets *errOut on read error.
func parseSSE(r io.Reader, errOut *error) iter.Seq[string] {
	return func(yield func(string) bool) {
		scanner := bufio.NewScanner(r)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			payload := strings.TrimPrefix(line, "data: ")
			if payload == "[DONE]" {
				return
			}
			if !yield(payload) {
				return
			}
		}
		if err := scanner.Err(); err != nil {
			*errOut = fmt.Errorf("llamacpp: read stream: %w", err)
		}
	}
}

// noChunks is an empty iterator for error cases.
func noChunks(func(string) bool) {}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./internal/llamacpp/ -run TestChat -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add internal/llamacpp/client.go internal/llamacpp/client_test.go
git commit -m "feat: llamacpp SSE streaming client for chat completions"
```

---

### Task 3: Text Completion Streaming Tests

**Files:**
- Modify: `internal/llamacpp/client_test.go` (add completion tests)

The `Complete` method was already implemented in Task 2. This task adds its tests.

**Step 1: Write the failing tests**

Append to `internal/llamacpp/client_test.go`:

```go
func TestComplete_Streaming(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/completions", r.URL.Path)

		flusher, ok := w.(http.Flusher)
		require.True(t, ok)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		chunks := []string{
			`{"choices":[{"text":"Once","finish_reason":null}]}`,
			`{"choices":[{"text":" upon","finish_reason":null}]}`,
			`{"choices":[{"text":" a time","finish_reason":null}]}`,
			`{"choices":[{"text":"","finish_reason":"stop"}]}`,
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	chunks, errFn := c.Complete(context.Background(), CompletionRequest{
		Prompt:    "Once",
		MaxTokens: 32,
	})

	var result []string
	for text := range chunks {
		result = append(result, text)
	}

	require.NoError(t, errFn())
	assert.Equal(t, []string{"Once", " upon", " a time"}, result)
}

func TestComplete_HTTPError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	chunks, errFn := c.Complete(context.Background(), CompletionRequest{
		Prompt: "Hello",
	})

	for range chunks {
		t.Fatal("expected no chunks")
	}

	assert.ErrorContains(t, errFn(), "400")
}
```

**Step 2: Run tests to verify they pass**

Run: `go test ./internal/llamacpp/ -run TestComplete -v`
Expected: PASS (both tests — implementation already exists from Task 2)

**Step 3: Commit**

```bash
git add internal/llamacpp/client_test.go
git commit -m "test: completion streaming tests for llamacpp client"
```

---

### Task 4: Server Helpers

**Files:**
- Create: `server.go`
- Create: `server_test.go`

These are the utility functions that don't require a GPU: finding the llama-server binary and getting a free port.

**Step 1: Write the failing tests**

```go
// server_test.go
//go:build linux && amd64

package rocm

import (
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFindLlamaServer_InPATH(t *testing.T) {
	// llama-server is installed at /usr/local/bin/llama-server on this machine
	path, err := findLlamaServer()
	require.NoError(t, err)
	assert.Contains(t, path, "llama-server")
}

func TestFindLlamaServer_EnvOverride(t *testing.T) {
	t.Setenv("ROCM_LLAMA_SERVER_PATH", "/usr/local/bin/llama-server")
	path, err := findLlamaServer()
	require.NoError(t, err)
	assert.Equal(t, "/usr/local/bin/llama-server", path)
}

func TestFindLlamaServer_EnvNotFound(t *testing.T) {
	t.Setenv("ROCM_LLAMA_SERVER_PATH", "/nonexistent/llama-server")
	_, err := findLlamaServer()
	assert.ErrorContains(t, err, "not found")
}

func TestFreePort(t *testing.T) {
	port, err := freePort()
	require.NoError(t, err)
	assert.Greater(t, port, 0)
	assert.Less(t, port, 65536)
}

func TestFreePort_UniquePerCall(t *testing.T) {
	p1, err := freePort()
	require.NoError(t, err)
	p2, err := freePort()
	require.NoError(t, err)
	// Not guaranteed but extremely likely with random kernel allocation
	_ = p1
	_ = p2
}

func TestServerEnv_HIPVisibleDevices(t *testing.T) {
	env := serverEnv()
	var hipVals []string
	for _, e := range env {
		if strings.HasPrefix(e, "HIP_VISIBLE_DEVICES=") {
			hipVals = append(hipVals, e)
		}
	}
	assert.Equal(t, []string{"HIP_VISIBLE_DEVICES=0"}, hipVals)
}

func TestServerEnv_FiltersExistingHIP(t *testing.T) {
	t.Setenv("HIP_VISIBLE_DEVICES", "1")
	env := serverEnv()
	var hipVals []string
	for _, e := range env {
		if strings.HasPrefix(e, "HIP_VISIBLE_DEVICES=") {
			hipVals = append(hipVals, e)
		}
	}
	// Must filter the old value and only have our override
	assert.Equal(t, []string{"HIP_VISIBLE_DEVICES=0"}, hipVals)
}
```

**Step 2: Run tests to verify they fail**

Run: `go test -run "TestFind|TestFree|TestServer" -v`
Expected: FAIL — `findLlamaServer`, `freePort`, `serverEnv` not defined.

**Step 3: Write minimal implementation**

```go
// server.go
//go:build linux && amd64

package rocm

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"

	"forge.lthn.ai/core/go-rocm/internal/llamacpp"
)

// server manages a llama-server subprocess.
type server struct {
	cmd     *exec.Cmd
	port    int
	client  *llamacpp.Client
	exited  chan struct{}
	exitErr error
}

// findLlamaServer locates the llama-server binary.
// Checks ROCM_LLAMA_SERVER_PATH first, then PATH.
func findLlamaServer() (string, error) {
	if p := os.Getenv("ROCM_LLAMA_SERVER_PATH"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
		return "", fmt.Errorf("rocm: ROCM_LLAMA_SERVER_PATH=%q not found", p)
	}
	p, err := exec.LookPath("llama-server")
	if err != nil {
		return "", fmt.Errorf("rocm: llama-server not in PATH: %w", err)
	}
	return p, nil
}

// freePort asks the kernel for a free TCP port on localhost.
func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	return port, nil
}

// serverEnv returns the environment for the llama-server subprocess.
// Filters any existing HIP_VISIBLE_DEVICES and sets it to 0 to mask the iGPU.
func serverEnv() []string {
	env := os.Environ()
	filtered := make([]string, 0, len(env)+1)
	for _, e := range env {
		if !strings.HasPrefix(e, "HIP_VISIBLE_DEVICES=") {
			filtered = append(filtered, e)
		}
	}
	return append(filtered, "HIP_VISIBLE_DEVICES=0")
}

// startServer spawns llama-server and waits for it to become ready.
func startServer(binary, modelPath string, port, gpuLayers, ctxSize int) (*server, error) {
	ngl := gpuLayers
	if ngl < 0 {
		ngl = 999 // all layers
	}

	args := []string{
		"--model", modelPath,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--n-gpu-layers", strconv.Itoa(ngl),
	}
	if ctxSize > 0 {
		args = append(args, "--ctx-size", strconv.Itoa(ctxSize))
	}

	cmd := exec.Command(binary, args...)
	cmd.Env = serverEnv()

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("rocm: start llama-server: %w", err)
	}

	client := llamacpp.NewClient(fmt.Sprintf("http://127.0.0.1:%d", port))
	s := &server{
		cmd:    cmd,
		port:   port,
		client: client,
		exited: make(chan struct{}),
	}

	go func() {
		s.exitErr = cmd.Wait()
		close(s.exited)
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := s.waitReady(ctx); err != nil {
		s.stop()
		return nil, err
	}

	return s, nil
}

// waitReady polls the health endpoint until the server is ready.
func (s *server) waitReady(ctx context.Context) error {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("rocm: llama-server health timeout: %w", ctx.Err())
		case <-s.exited:
			return fmt.Errorf("rocm: llama-server exited during startup: %v", s.exitErr)
		case <-ticker.C:
			if err := s.client.Health(ctx); err == nil {
				return nil
			}
		}
	}
}

// stop sends SIGTERM and waits up to 5s, then SIGKILL.
func (s *server) stop() error {
	if s.cmd.Process == nil {
		return nil
	}
	_ = s.cmd.Process.Signal(syscall.SIGTERM)
	select {
	case <-time.After(5 * time.Second):
		_ = s.cmd.Process.Kill()
		<-s.exited
	case <-s.exited:
	}
	return nil
}
```

**Step 4: Run tests to verify they pass**

Run: `go test -run "TestFind|TestFree|TestServer" -v`
Expected: PASS (all tests). Note: `TestFindLlamaServer_InPATH` requires llama-server installed.

**Step 5: Add testify dependency and commit**

```bash
go get github.com/stretchr/testify
go mod tidy
git add server.go server_test.go go.mod go.sum
git commit -m "feat: server lifecycle and helpers for llama-server subprocess"
```

---

### Task 5: TextModel Implementation

**Files:**
- Create: `model.go`

No unit test for this file — it's a thin wrapper that maps inference types to llamacpp types. Tested via integration test in Task 7.

**Step 1: Write the implementation**

```go
// model.go
//go:build linux && amd64

package rocm

import (
	"context"
	"iter"
	"sync"

	"forge.lthn.ai/core/go-inference"
	"forge.lthn.ai/core/go-rocm/internal/llamacpp"
)

// rocmModel implements inference.TextModel using a llama-server subprocess.
type rocmModel struct {
	srv       *server
	modelType string

	mu      sync.Mutex
	lastErr error
}

func (m *rocmModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)

	req := llamacpp.CompletionRequest{
		Prompt:        prompt,
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		RepeatPenalty: cfg.RepeatPenalty,
	}

	chunks, errFn := m.srv.client.Complete(ctx, req)
	return func(yield func(inference.Token) bool) {
		for text := range chunks {
			if !yield(inference.Token{Text: text}) {
				break
			}
		}
		if err := errFn(); err != nil {
			m.mu.Lock()
			m.lastErr = err
			m.mu.Unlock()
		}
	}
}

func (m *rocmModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)

	chatMsgs := make([]llamacpp.ChatMessage, len(messages))
	for i, msg := range messages {
		chatMsgs[i] = llamacpp.ChatMessage{Role: msg.Role, Content: msg.Content}
	}

	req := llamacpp.ChatRequest{
		Messages:      chatMsgs,
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		RepeatPenalty: cfg.RepeatPenalty,
	}

	chunks, errFn := m.srv.client.ChatComplete(ctx, req)
	return func(yield func(inference.Token) bool) {
		for text := range chunks {
			if !yield(inference.Token{Text: text}) {
				break
			}
		}
		if err := errFn(); err != nil {
			m.mu.Lock()
			m.lastErr = err
			m.mu.Unlock()
		}
	}
}

func (m *rocmModel) ModelType() string { return m.modelType }

func (m *rocmModel) Err() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastErr
}

func (m *rocmModel) Close() error {
	return m.srv.stop()
}
```

**Step 2: Verify it compiles**

Run: `go build ./...`
Expected: No errors.

**Step 3: Commit**

```bash
git add model.go
git commit -m "feat: TextModel implementation wrapping llama-server"
```

---

### Task 6: Backend Available() and LoadModel()

**Files:**
- Modify: `backend.go` (replace stubs with real implementation)

**Step 1: Update backend.go**

Replace the entire file:

```go
// backend.go
//go:build linux && amd64

package rocm

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"forge.lthn.ai/core/go-inference"
)

// rocmBackend implements inference.Backend for AMD ROCm GPUs.
type rocmBackend struct{}

func (b *rocmBackend) Name() string { return "rocm" }

func (b *rocmBackend) Available() bool {
	if _, err := os.Stat("/dev/kfd"); err != nil {
		return false
	}
	if _, err := findLlamaServer(); err != nil {
		return false
	}
	return true
}

func (b *rocmBackend) LoadModel(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
	cfg := inference.ApplyLoadOpts(opts)

	binary, err := findLlamaServer()
	if err != nil {
		return nil, err
	}

	port, err := freePort()
	if err != nil {
		return nil, fmt.Errorf("rocm: find free port: %w", err)
	}

	srv, err := startServer(binary, path, port, cfg.GPULayers, cfg.ContextLen)
	if err != nil {
		return nil, err
	}

	return &rocmModel{
		srv:       srv,
		modelType: guessModelType(path),
	}, nil
}

// guessModelType extracts a model type hint from the filename.
// e.g. "LEK-Gemma3-4B-Q4_K_M.gguf" -> "gemma3"
func guessModelType(path string) string {
	name := strings.ToLower(filepath.Base(path))
	for _, arch := range []string{"gemma3", "gemma2", "gemma", "qwen3", "qwen2", "qwen", "llama3", "llama", "mistral", "phi"} {
		if strings.Contains(name, arch) {
			return arch
		}
	}
	return "unknown"
}
```

**Step 2: Verify it compiles**

Run: `go build ./...`
Expected: No errors.

**Step 3: Write a quick test for Available() and guessModelType()**

Add to `server_test.go`:

```go
func TestAvailable(t *testing.T) {
	b := &rocmBackend{}
	// On this machine (snider-linux): /dev/kfd exists, llama-server in PATH
	// This test will fail on machines without ROCm
	if _, err := os.Stat("/dev/kfd"); err != nil {
		t.Skip("no ROCm hardware")
	}
	assert.True(t, b.Available())
}

func TestGuessModelType(t *testing.T) {
	tests := []struct {
		path     string
		expected string
	}{
		{"/data/lem/gguf/LEK-Gemma3-4B-Q4_K_M.gguf", "gemma3"},
		{"/data/models/Qwen3-8B-Q4_K_M.gguf", "qwen3"},
		{"/data/models/Llama-3.1-8B-Q4_K_M.gguf", "llama3"},
		{"/data/models/Mistral-7B-v0.3-Q4_K_M.gguf", "mistral"},
		{"/data/models/random-model.gguf", "unknown"},
	}
	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			assert.Equal(t, tt.expected, guessModelType(tt.path))
		})
	}
}
```

**Step 4: Run tests**

Run: `go test -run "TestAvailable|TestGuessModel" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend.go server_test.go
git commit -m "feat: Backend Available() and LoadModel() with GPU detection"
```

---

### Task 7: Integration Test

**Files:**
- Create: `rocm_integration_test.go`

This test loads a real model on the GPU, generates tokens, and verifies the full pipeline.

**Step 1: Write the integration test**

```go
// rocm_integration_test.go
//go:build rocm

package rocm

import (
	"context"
	"os"
	"testing"
	"time"

	"forge.lthn.ai/core/go-inference"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testModel = "/data/lem/gguf/LEK-Gemma3-1B-layered-v2-Q5_K_M.gguf"

func skipIfNoModel(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(testModel); os.IsNotExist(err) {
		t.Skip("test model not available (SMB mount down?)")
	}
}

func skipIfNoROCm(t *testing.T) {
	t.Helper()
	if _, err := os.Stat("/dev/kfd"); err != nil {
		t.Skip("no ROCm hardware")
	}
	if _, err := findLlamaServer(); err != nil {
		t.Skip("llama-server not found")
	}
}

func TestROCm_LoadAndGenerate(t *testing.T) {
	skipIfNoROCm(t)
	skipIfNoModel(t)

	b := &rocmBackend{}
	require.True(t, b.Available())

	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	require.NoError(t, err)
	defer m.Close()

	assert.Equal(t, "gemma3", m.ModelType())

	// Generate with a simple prompt
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var tokens []string
	for tok := range m.Generate(ctx, "The capital of France is", inference.WithMaxTokens(16)) {
		tokens = append(tokens, tok.Text)
	}

	require.NoError(t, m.Err())
	require.NotEmpty(t, tokens, "expected at least one token")

	full := ""
	for _, tok := range tokens {
		full += tok
	}
	t.Logf("Generated: %s", full)
}

func TestROCm_Chat(t *testing.T) {
	skipIfNoROCm(t)
	skipIfNoModel(t)

	b := &rocmBackend{}
	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	require.NoError(t, err)
	defer m.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	messages := []inference.Message{
		{Role: "user", Content: "Say hello in exactly three words."},
	}

	var tokens []string
	for tok := range m.Chat(ctx, messages, inference.WithMaxTokens(32)) {
		tokens = append(tokens, tok.Text)
	}

	require.NoError(t, m.Err())
	require.NotEmpty(t, tokens, "expected at least one token")

	full := ""
	for _, tok := range tokens {
		full += tok
	}
	t.Logf("Chat response: %s", full)
}

func TestROCm_ContextCancellation(t *testing.T) {
	skipIfNoROCm(t)
	skipIfNoModel(t)

	b := &rocmBackend{}
	m, err := b.LoadModel(testModel, inference.WithContextLen(2048))
	require.NoError(t, err)
	defer m.Close()

	ctx, cancel := context.WithCancel(context.Background())

	var count int
	for tok := range m.Generate(ctx, "Write a very long story about dragons", inference.WithMaxTokens(256)) {
		_ = tok
		count++
		if count >= 3 {
			cancel()
		}
	}

	t.Logf("Got %d tokens before cancel", count)
	assert.GreaterOrEqual(t, count, 3)
}
```

**Step 2: Run integration tests**

Run: `go test -tags rocm -run TestROCm -v -timeout 120s`
Expected: PASS (all 3 tests). Each test loads the model (~2-3s for 1B), generates tokens, and shuts down.

**Step 3: Commit**

```bash
git add rocm_integration_test.go
git commit -m "test: integration tests for full ROCm inference pipeline"
```

---

### Task 8: Update TODO.md

**Files:**
- Modify: `TODO.md`

**Step 1: Mark Phase 1 tasks complete**

Mark all 5 Phase 1 items as `[x]` with commit references and date.

**Step 2: Commit**

```bash
git add TODO.md
git commit -m "docs: mark Phase 1 tasks complete"
```

---

## Build Order Summary

```
Task 1: health.go + test          (unit, no GPU)
Task 2: client.go + test          (unit, no GPU)
Task 3: client_test.go additions  (unit, no GPU)
Task 4: server.go + test          (unit, no GPU — helpers only)
Task 5: model.go                  (compiles)
Task 6: backend.go + test         (unit, partial)
Task 7: integration test          (GPU + model required)
Task 8: docs update
```

Tasks 1-6 can be verified without GPU hardware. Task 7 requires the RX 7800 XT and test model on SMB mount.
