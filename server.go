//go:build linux && amd64

package rocm

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	coreerr "dappco.re/go/core/log"
	"dappco.re/go/rocm/internal/llamacpp"
)

var (
	serverStartupTimeout    = 60 * time.Second
	serverReadyPollInterval = 100 * time.Millisecond
	serverPortAllocator     = newDeterministicPortAllocator(serverPortRangeStart, serverPortRangeCount)
	// listenLocalTCP lets tests stub port probing without opening real sockets.
	listenLocalTCP = net.Listen
)

const (
	serverProcessOutputLimit       = 32 << 10
	serverProcessOutputSummarySize = 1024
	serverPortRangeStart           = 38080
	serverPortRangeCount           = 256
)

// server manages a llama-server subprocess.
type server struct {
	cmd           *exec.Cmd
	port          int
	client        *llamacpp.Client
	exited        chan struct{}
	exitErr       error // safe to read only after <-exited
	processOutput *processOutputCapture
}

// serverStartConfig keeps llama-server startup settings named instead of positional.
type serverStartConfig struct {
	BinaryPath        string
	ModelPath         string
	GPULayerCount     int
	ContextSize       int
	ParallelSlotCount int
}

// alive reports whether the llama-server process is still running.
func (s *server) alive() bool {
	select {
	case <-s.exited:
		return false
	default:
		return true
	}
}

// findLlamaServer locates the llama-server binary.
// Checks ROCM_LLAMA_SERVER_PATH first, then PATH.
func findLlamaServer() (string, error) {
	if p := os.Getenv("ROCM_LLAMA_SERVER_PATH"); p != "" {
		return validateLlamaServerPath(p)
	}
	p, err := exec.LookPath("llama-server")
	if err != nil {
		return "", coreerr.E("rocm.findLlamaServer", "llama-server not found in PATH", err)
	}
	return p, nil
}

func validateLlamaServerPath(path string) (string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return "", coreerr.E("rocm.findLlamaServer", "llama-server not found at ROCM_LLAMA_SERVER_PATH="+path, err)
	}
	if info.IsDir() {
		return "", coreerr.E("rocm.findLlamaServer", "ROCM_LLAMA_SERVER_PATH must point to a file", nil)
	}
	if info.Mode().Perm()&0o111 == 0 {
		return "", coreerr.E("rocm.findLlamaServer", "llama-server is not executable at ROCM_LLAMA_SERVER_PATH="+path, nil)
	}
	return path, nil
}

// freePort walks a deterministic localhost port range and returns the first
// currently-bindable port.
func freePort() (int, error) {
	return serverPortAllocator.NextAvailablePort()
}

// serverEnv returns the environment for the llama-server subprocess.
// Filters any existing HIP_* settings and sets HIP_VISIBLE_DEVICES=0 to mask
// the iGPU. This is critical — the Ryzen 9 iGPU crashes llama-server if not
// masked, and inherited HIP variables can re-expose multi-GPU state.
func serverEnv() []string {
	environ := os.Environ()
	env := make([]string, 0, len(environ)+1)
	for _, e := range environ {
		if strings.HasPrefix(e, "HIP_") {
			continue
		}
		env = append(env, e)
	}
	env = append(env, "HIP_VISIBLE_DEVICES=0")
	return env
}

// startServer spawns llama-server and waits for it to become ready.
// It selects a free port automatically, retrying up to 3 times if startup
// fails before the health endpoint becomes ready.
func startServer(startConfig serverStartConfig) (*server, error) {
	gpuLayerCount := startConfig.GPULayerCount
	if gpuLayerCount < 0 {
		gpuLayerCount = 999
	}

	const maxAttempts = 3
	var lastErr error

	for attempt := 0; attempt < maxAttempts; attempt++ {
		port, err := freePort()
		if err != nil {
			return nil, coreerr.E("rocm.startServer", "find free port", err)
		}

		args := []string{
			"--model", startConfig.ModelPath,
			"--host", "127.0.0.1",
			"--port", strconv.Itoa(port),
			"--n-gpu-layers", strconv.Itoa(gpuLayerCount),
		}
		if startConfig.ContextSize > 0 {
			args = append(args, "--ctx-size", strconv.Itoa(startConfig.ContextSize))
		}
		if startConfig.ParallelSlotCount > 0 {
			args = append(args, "--parallel", strconv.Itoa(startConfig.ParallelSlotCount))
		}

		processOutput := newProcessOutputCapture(serverProcessOutputLimit)
		cmd := exec.Command(startConfig.BinaryPath, args...)
		cmd.Env = serverEnv()
		cmd.Stdout = processOutput
		cmd.Stderr = processOutput

		if err := cmd.Start(); err != nil {
			return nil, coreerr.E("rocm.startServer", "start llama-server", err)
		}

		s := &server{
			cmd:           cmd,
			port:          port,
			client:        llamacpp.NewClient(fmt.Sprintf("http://127.0.0.1:%d", port)),
			exited:        make(chan struct{}),
			processOutput: processOutput,
		}

		go func() {
			s.exitErr = cmd.Wait()
			close(s.exited)
		}()

		ctx, cancel := context.WithTimeout(context.Background(), serverStartupTimeout)
		err = s.waitReady(ctx)
		cancel()
		if err == nil {
			return s, nil
		}

		if stopErr := s.stop(); stopErr != nil {
			coreerr.Warn("llama-server cleanup after failed startup returned error", "attempt", attempt+1, "err", stopErr)
		}
		lastErr = coreerr.E("rocm.startServer", fmt.Sprintf("attempt %d", attempt+1), err)
		if attempt < maxAttempts-1 {
			coreerr.Warn("llama-server startup failed; retrying", "attempt", attempt+1, "max_attempts", maxAttempts, "err", lastErr)
		}
	}

	return nil, coreerr.E("rocm.startServer", fmt.Sprintf("server failed after %d attempts", maxAttempts), lastErr)
}

// waitReady polls the health endpoint until the server is ready.
func (s *server) waitReady(ctx context.Context) error {
	ticker := time.NewTicker(serverReadyPollInterval)
	defer ticker.Stop()

	var lastHealthErr error

	for {
		select {
		case <-ctx.Done():
			if lastHealthErr != nil {
				return coreerr.E("server.waitReady", s.messageWithProcessOutput("timeout waiting for llama-server"), lastHealthErr)
			}
			return coreerr.E("server.waitReady", s.messageWithProcessOutput("timeout waiting for llama-server"), ctx.Err())
		case <-s.exited:
			return s.wrapProcessError("server.waitReady", "llama-server exited before becoming ready", s.exitErr)
		case <-ticker.C:
			if err := s.client.Health(ctx); err == nil {
				return nil
			} else {
				lastHealthErr = err
			}
		}
	}
}

// stop sends SIGTERM and waits up to 5s, then SIGKILL. Exit caused by those
// signals is treated as a successful caller-initiated shutdown.
func (s *server) stop() error {
	if s.cmd.Process == nil {
		return nil
	}

	// Already exited?
	select {
	case <-s.exited:
		if isExpectedStopExitErr(s.exitErr) {
			return nil
		}
		return s.wrapProcessError("server.stop", "llama-server already exited", s.exitErr)
	default:
	}

	// Send SIGTERM for graceful shutdown.
	if err := s.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		return coreerr.E("server.stop", "sigterm llama-server", err)
	}

	// Wait up to 5 seconds for clean exit.
	select {
	case <-s.exited:
		if isExpectedStopExitErr(s.exitErr) {
			return nil
		}
		return s.wrapProcessError("server.stop", "llama-server exited after sigterm", s.exitErr)
	case <-time.After(5 * time.Second):
		// Force kill.
		if err := s.cmd.Process.Kill(); err != nil {
			return coreerr.E("server.stop", "kill llama-server", err)
		}
		<-s.exited
		if isExpectedStopExitErr(s.exitErr) {
			return nil
		}
		return s.wrapProcessError("server.stop", "llama-server exited after sigkill", s.exitErr)
	}
}

func isExpectedStopExitErr(err error) bool {
	if err == nil {
		return false
	}

	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		return false
	}

	status, ok := exitErr.ProcessState.Sys().(syscall.WaitStatus)
	if !ok || !status.Signaled() {
		return false
	}

	switch status.Signal() {
	case syscall.SIGTERM, syscall.SIGKILL:
		return true
	default:
		return false
	}
}

func (s *server) messageWithProcessOutput(message string) string {
	if s == nil || s.processOutput == nil {
		return message
	}
	output := s.processOutput.Summary()
	if output == "" {
		return message
	}
	return message + " (llama-server output: " + output + ")"
}

func (s *server) wrapProcessError(op, message string, err error) error {
	if err == nil {
		return nil
	}
	return coreerr.E(op, s.messageWithProcessOutput(message), err)
}

type deterministicPortAllocator struct {
	basePort  int
	portCount int
	nextPort  atomic.Uint64
}

func newDeterministicPortAllocator(basePort, portCount int) *deterministicPortAllocator {
	return &deterministicPortAllocator{
		basePort:  basePort,
		portCount: portCount,
	}
}

func (allocator *deterministicPortAllocator) NextAvailablePort() (int, error) {
	if allocator == nil || allocator.portCount <= 0 {
		return 0, coreerr.E("rocm.freePort", "port allocator is not configured", nil)
	}

	lastPort := allocator.basePort + allocator.portCount - 1
	if allocator.basePort <= 0 || lastPort > 65535 {
		return 0, coreerr.E("rocm.freePort", fmt.Sprintf("invalid port range %d-%d", allocator.basePort, lastPort), nil)
	}

	startIndex := allocator.nextPort.Add(1) - 1
	for scanned := 0; scanned < allocator.portCount; scanned++ {
		portIndex := int((startIndex + uint64(scanned)) % uint64(allocator.portCount))
		port := allocator.basePort + portIndex
		address := net.JoinHostPort("127.0.0.1", strconv.Itoa(port))

		listener, err := listenLocalTCP("tcp", address)
		if err != nil {
			continue
		}
		listener.Close()

		allocator.advancePast(startIndex + uint64(scanned) + 1)
		return port, nil
	}

	return 0, coreerr.E("rocm.freePort", fmt.Sprintf("no free port in deterministic range %d-%d", allocator.basePort, lastPort), nil)
}

func (allocator *deterministicPortAllocator) advancePast(candidate uint64) {
	for {
		current := allocator.nextPort.Load()
		if current >= candidate {
			return
		}
		if allocator.nextPort.CompareAndSwap(current, candidate) {
			return
		}
	}
}

type processOutputCapture struct {
	maxBytes int

	mu        sync.Mutex
	buffer    []byte
	truncated bool
}

func newProcessOutputCapture(maxBytes int) *processOutputCapture {
	return &processOutputCapture{maxBytes: maxBytes}
}

func (c *processOutputCapture) Write(p []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	written := len(p)
	if c.maxBytes <= 0 || written == 0 {
		return written, nil
	}

	c.buffer = append(c.buffer, p...)
	if len(c.buffer) > c.maxBytes {
		c.buffer = append([]byte(nil), c.buffer[len(c.buffer)-c.maxBytes:]...)
		c.truncated = true
	}

	return written, nil
}

func (c *processOutputCapture) Summary() string {
	c.mu.Lock()
	defer c.mu.Unlock()

	output := strings.TrimSpace(string(c.buffer))
	if output == "" {
		return ""
	}

	lines := strings.Split(output, "\n")
	parts := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts = append(parts, line)
	}

	output = strings.Join(parts, " | ")
	if len(output) > serverProcessOutputSummarySize {
		output = output[:serverProcessOutputSummarySize] + "..."
	}
	if c.truncated {
		return "..." + output
	}
	return output
}
