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

	coreerr "forge.lthn.ai/core/go-log"
	"forge.lthn.ai/core/go-rocm/internal/llamacpp"
)

// server manages a llama-server subprocess.
type server struct {
	cmd     *exec.Cmd
	port    int
	client  *llamacpp.Client
	exited  chan struct{}
	exitErr error // safe to read only after <-exited
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
		if _, err := os.Stat(p); err != nil {
			return "", coreerr.E("rocm.findLlamaServer", "llama-server not found at ROCM_LLAMA_SERVER_PATH="+p, err)
		}
		return p, nil
	}
	p, err := exec.LookPath("llama-server")
	if err != nil {
		return "", coreerr.E("rocm.findLlamaServer", "llama-server not found in PATH", err)
	}
	return p, nil
}

// freePort asks the kernel for a free TCP port on localhost.
func freePort() (int, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, coreerr.E("rocm.freePort", "listen for free port", err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()
	return port, nil
}

// serverEnv returns the environment for the llama-server subprocess.
// Filters any existing HIP_VISIBLE_DEVICES and sets it to 0 to mask the iGPU.
// This is critical — the Ryzen 9 iGPU crashes llama-server if not masked.
func serverEnv() []string {
	environ := os.Environ()
	env := make([]string, 0, len(environ)+1)
	for _, e := range environ {
		if strings.HasPrefix(e, "HIP_VISIBLE_DEVICES=") {
			continue
		}
		env = append(env, e)
	}
	env = append(env, "HIP_VISIBLE_DEVICES=0")
	return env
}

// startServer spawns llama-server and waits for it to become ready.
// It selects a free port automatically, retrying up to 3 times if the
// process exits during startup (e.g. port conflict).
func startServer(binary, modelPath string, gpuLayers, ctxSize, parallelSlots int) (*server, error) {
	if gpuLayers < 0 {
		gpuLayers = 999
	}

	const maxAttempts = 3
	var lastErr error

	for attempt := range maxAttempts {
		port, err := freePort()
		if err != nil {
			return nil, coreerr.E("rocm.startServer", "find free port", err)
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
		if parallelSlots > 0 {
			args = append(args, "--parallel", strconv.Itoa(parallelSlots))
		}

		cmd := exec.Command(binary, args...)
		cmd.Env = serverEnv()

		if err := cmd.Start(); err != nil {
			return nil, coreerr.E("rocm.startServer", "start llama-server", err)
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

		// Only retry if the process actually exited (e.g. port conflict).
		// A timeout means the server is stuck, not a port issue.
		select {
		case <-s.exited:
			_ = s.stop()
			lastErr = coreerr.E("rocm.startServer", fmt.Sprintf("attempt %d", attempt+1), err)
			continue
		default:
			_ = s.stop()
			return nil, coreerr.E("rocm.startServer", "llama-server not ready", err)
		}
	}

	return nil, coreerr.E("rocm.startServer", fmt.Sprintf("server failed after %d attempts", maxAttempts), lastErr)
}

// waitReady polls the health endpoint until the server is ready.
func (s *server) waitReady(ctx context.Context) error {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return coreerr.E("server.waitReady", "timeout waiting for llama-server", ctx.Err())
		case <-s.exited:
			return coreerr.E("server.waitReady", "llama-server exited before becoming ready", s.exitErr)
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

	// Already exited?
	select {
	case <-s.exited:
		return s.exitErr
	default:
	}

	// Send SIGTERM for graceful shutdown.
	if err := s.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		return coreerr.E("server.stop", "sigterm llama-server", err)
	}

	// Wait up to 5 seconds for clean exit.
	select {
	case <-s.exited:
		return s.exitErr
	case <-time.After(5 * time.Second):
		// Force kill.
		if err := s.cmd.Process.Kill(); err != nil {
			return coreerr.E("server.stop", "kill llama-server", err)
		}
		<-s.exited
		return s.exitErr
	}
}
