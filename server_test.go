//go:build linux && amd64

package rocm

import (
	"context"
	"errors"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	coreerr "dappco.re/go/core/log"
	"forge.lthn.ai/core/go-inference"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFindLlamaServer_InPATH(t *testing.T) {
	// llama-server is at /usr/local/bin/llama-server on this machine.
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

func TestFindLlamaServer_EnvNotExecutable(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "llama-server")
	require.NoError(t, os.WriteFile(path, []byte("#!/bin/sh\nexit 0\n"), 0644))

	t.Setenv("ROCM_LLAMA_SERVER_PATH", path)

	_, err := findLlamaServer()
	require.Error(t, err)
	assert.ErrorContains(t, err, "not executable")
}

func TestFreePort(t *testing.T) {
	restoreListen := stubListenLocalTCP(t, func(network, address string) (net.Listener, error) {
		return fakeTCPListener{address: address}, nil
	})
	defer restoreListen()

	port, err := freePort()
	require.NoError(t, err)
	assert.Greater(t, port, 0)
	assert.Less(t, port, 65536)
}

func TestFreePort_UniquePerCall(t *testing.T) {
	restoreListen := stubListenLocalTCP(t, func(network, address string) (net.Listener, error) {
		return fakeTCPListener{address: address}, nil
	})
	defer restoreListen()

	p1, err := freePort()
	require.NoError(t, err)
	p2, err := freePort()
	require.NoError(t, err)
	assert.NotEqual(t, p1, p2)
}

func TestDeterministicPortAllocator_AdvancesAcrossCalls(t *testing.T) {
	restoreListen := stubListenLocalTCP(t, func(network, address string) (net.Listener, error) {
		return fakeTCPListener{address: address}, nil
	})
	defer restoreListen()

	allocator := newDeterministicPortAllocator(41000, 3)

	firstPort, err := allocator.NextAvailablePort()
	require.NoError(t, err)

	secondPort, err := allocator.NextAvailablePort()
	require.NoError(t, err)

	assert.Equal(t, 41000, firstPort)
	assert.Equal(t, 41001, secondPort)
}

func TestDeterministicPortAllocator_SkipsOccupiedPort(t *testing.T) {
	restoreListen := stubListenLocalTCP(t, func(network, address string) (net.Listener, error) {
		if address == "127.0.0.1:42000" {
			return nil, errors.New("port already in use")
		}
		return fakeTCPListener{address: address}, nil
	})
	defer restoreListen()

	allocator := newDeterministicPortAllocator(42000, 3)
	port, err := allocator.NextAvailablePort()
	require.NoError(t, err)
	assert.Equal(t, 42001, port)
}

func TestDeterministicPortAllocator_ReturnsErrorWhenRangeIsExhausted(t *testing.T) {
	restoreListen := stubListenLocalTCP(t, func(network, address string) (net.Listener, error) {
		return nil, errors.New("port already in use")
	})
	defer restoreListen()

	allocator := newDeterministicPortAllocator(43000, 2)
	_, err := allocator.NextAvailablePort()
	require.Error(t, err)
	assert.ErrorContains(t, err, "no free port in deterministic range")
}

func TestLlamaServerArguments(t *testing.T) {
	args := llamaServerArguments(serverStartConfig{
		ModelPath:         "/models/gemma3.gguf",
		ContextSize:       2048,
		ParallelSlotCount: 4,
	}, 38123, 999)

	assert.Equal(t, []string{
		"--model", "/models/gemma3.gguf",
		"--host", "127.0.0.1",
		"--port", "38123",
		"--n-gpu-layers", "999",
		"--ctx-size", "2048",
		"--parallel", "4",
	}, args)
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
	t.Setenv("HIP_DEVICE_ORDER", "PCI_BUS_ID")
	t.Setenv("HIP_TRACE_API", "1")
	env := serverEnv()
	var hipVals []string
	for _, e := range env {
		if strings.HasPrefix(e, "HIP_") {
			hipVals = append(hipVals, e)
		}
	}
	assert.Equal(t, []string{"HIP_VISIBLE_DEVICES=0"}, hipVals)
}

func TestAvailable(t *testing.T) {
	b := &rocmBackend{}
	if _, err := os.Stat("/dev/kfd"); err != nil {
		t.Skip("no ROCm hardware")
	}
	assert.True(t, b.Available())
}

func TestServerAlive_Running(t *testing.T) {
	s := &server{processExited: make(chan struct{})}
	assert.True(t, s.alive())
}

func TestServerAlive_Exited(t *testing.T) {
	processExited := make(chan struct{})
	close(processExited)
	s := &server{processExited: processExited, processExitError: coreerr.E("test", "process killed", nil)}
	assert.False(t, s.alive())
}

func TestGenerate_ServerDead(t *testing.T) {
	processOutput := newProcessOutputCapture(serverProcessOutputLimit)
	_, _ = processOutput.Write([]byte("fatal: HIP launch failure\n"))
	processExited := make(chan struct{})
	close(processExited)
	s := &server{
		processExited:    processExited,
		processExitError: coreerr.E("test", "process killed", nil),
		processOutput:    processOutput,
	}
	m := &rocmModel{server: s}

	var count int
	for range m.Generate(context.Background(), "hello") {
		count++
	}
	assert.Equal(t, 0, count)
	assert.ErrorContains(t, m.Err(), "server has exited")
	assert.ErrorContains(t, m.Err(), "HIP launch failure")
}

func TestStartServer_RetriesOnProcessExit(t *testing.T) {
	restoreListen := stubListenLocalTCP(t, func(network, address string) (net.Listener, error) {
		return fakeTCPListener{address: address}, nil
	})
	defer restoreListen()

	// /bin/false starts successfully but exits immediately with code 1.
	// startServer should retry up to 3 times, then fail.
	_, err := startServer(serverStartConfig{
		BinaryPath:    "/bin/false",
		ModelPath:     "/nonexistent/model.gguf",
		GPULayerCount: 999,
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed after 3 attempts")
}

func TestStartServer_RetriesOnStartupTimeout(t *testing.T) {
	restoreListen := stubListenLocalTCP(t, func(network, address string) (net.Listener, error) {
		return fakeTCPListener{address: address}, nil
	})
	defer restoreListen()

	dir := t.TempDir()
	binary := filepath.Join(dir, "fake-llama-server")
	require.NoError(t, os.WriteFile(binary, []byte("#!/bin/sh\nsleep 1\n"), 0755))

	oldTimeout := serverStartupTimeout
	oldInterval := serverReadyPollInterval
	serverStartupTimeout = 50 * time.Millisecond
	serverReadyPollInterval = 10 * time.Millisecond
	t.Cleanup(func() {
		serverStartupTimeout = oldTimeout
		serverReadyPollInterval = oldInterval
	})

	_, err := startServer(serverStartConfig{
		BinaryPath:    binary,
		ModelPath:     "/nonexistent/model.gguf",
		GPULayerCount: 999,
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed after 3 attempts")
	assert.Contains(t, err.Error(), "timeout waiting for llama-server")
}

func TestServerStop_GracefulSignalReturnsNil(t *testing.T) {
	processCommand := exec.Command("/bin/sleep", "60")
	require.NoError(t, processCommand.Start())

	s := &server{
		processCommand: processCommand,
		processExited:  make(chan struct{}),
	}
	go func() {
		s.processExitError = processCommand.Wait()
		close(s.processExited)
	}()

	require.NoError(t, s.stop())
	require.NoError(t, s.stop(), "stop should remain idempotent after graceful shutdown")
}

func TestServerWrapProcessError_IncludesProcessOutput(t *testing.T) {
	processOutput := newProcessOutputCapture(serverProcessOutputLimit)
	_, _ = processOutput.Write([]byte("HIP runtime exploded\nsecondary detail\n"))

	s := &server{processOutput: processOutput}

	err := s.wrapProcessError("server.waitReady", "llama-server exited before becoming ready", coreerr.E("test", "exit 1", nil))
	require.Error(t, err)
	assert.ErrorContains(t, err, "HIP runtime exploded")
	assert.ErrorContains(t, err, "secondary detail")
}

func TestChat_ServerDead(t *testing.T) {
	processExited := make(chan struct{})
	close(processExited)
	s := &server{
		processExited:    processExited,
		processExitError: coreerr.E("test", "process killed", nil),
	}
	m := &rocmModel{server: s}

	msgs := []inference.Message{{Role: "user", Content: "hello"}}
	var count int
	for range m.Chat(context.Background(), msgs) {
		count++
	}
	assert.Equal(t, 0, count)
	assert.ErrorContains(t, m.Err(), "server has exited")
}

func stubListenLocalTCP(t *testing.T, stub func(network, address string) (net.Listener, error)) func() {
	t.Helper()

	original := listenLocalTCP
	listenLocalTCP = stub
	return func() {
		listenLocalTCP = original
	}
}

type fakeTCPListener struct {
	address string
}

func (listener fakeTCPListener) Accept() (net.Conn, error) {
	return nil, errors.New("not implemented in test listener")
}

func (listener fakeTCPListener) Close() error { return nil }

func (listener fakeTCPListener) Addr() net.Addr {
	host, portText, err := net.SplitHostPort(listener.address)
	if err != nil {
		return &net.TCPAddr{}
	}

	port, err := strconv.Atoi(portText)
	if err != nil {
		return &net.TCPAddr{}
	}

	return &net.TCPAddr{
		IP:   net.ParseIP(host),
		Port: port,
	}
}
