//go:build linux && amd64

package rocm

import (
	"context"
	"errors"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"dappco.re/go/inference"
	coreerr "dappco.re/go/log"
)

func TestFindLlamaServer_InPATH(t *testing.T) {
	// llama-server is at /usr/local/bin/llama-server on this machine.
	path, err := findLlamaServer()
	if err != nil {
		t.Fatalf("findLlamaServer: %v", err)
	}
	if !strings.Contains(path, "llama-server") {
		t.Errorf("path = %q, want contains %q", path, "llama-server")
	}
}

func TestFindLlamaServer_EnvOverride(t *testing.T) {
	t.Setenv("ROCM_LLAMA_SERVER_PATH", "/usr/local/bin/llama-server")
	path, err := findLlamaServer()
	if err != nil {
		t.Fatalf("findLlamaServer: %v", err)
	}
	if path != "/usr/local/bin/llama-server" {
		t.Errorf("path = %q, want %q", path, "/usr/local/bin/llama-server")
	}
}

func TestFindLlamaServer_EnvNotFound(t *testing.T) {
	t.Setenv("ROCM_LLAMA_SERVER_PATH", "/nonexistent/llama-server")
	_, err := findLlamaServer()
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("err = %v, want contains %q", err, "not found")
	}
}

func TestFindLlamaServer_EnvNotExecutable(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "llama-server")
	if err := os.WriteFile(path, []byte("#!/bin/sh\nexit 0\n"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	t.Setenv("ROCM_LLAMA_SERVER_PATH", path)

	_, err := findLlamaServer()
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "not executable") {
		t.Errorf("err = %v, want contains %q", err, "not executable")
	}
}

func TestFreePort(t *testing.T) {
	restoreListen := stubListenLocalTCP(t, func(network, address string) (net.Listener, error) {
		return fakeTCPListener{address: address}, nil
	})
	defer restoreListen()

	port, err := freePort()
	if err != nil {
		t.Fatalf("freePort: %v", err)
	}
	if port <= 0 {
		t.Errorf("port = %d, want > 0", port)
	}
	if port >= 65536 {
		t.Errorf("port = %d, want < 65536", port)
	}
}

func TestFreePort_UniquePerCall(t *testing.T) {
	restoreListen := stubListenLocalTCP(t, func(network, address string) (net.Listener, error) {
		return fakeTCPListener{address: address}, nil
	})
	defer restoreListen()

	p1, err := freePort()
	if err != nil {
		t.Fatalf("freePort: %v", err)
	}
	p2, err := freePort()
	if err != nil {
		t.Fatalf("freePort: %v", err)
	}
	if p1 == p2 {
		t.Errorf("freePort returned identical ports: %d", p1)
	}
}

func TestDeterministicPortAllocator_AdvancesAcrossCalls(t *testing.T) {
	restoreListen := stubListenLocalTCP(t, func(network, address string) (net.Listener, error) {
		return fakeTCPListener{address: address}, nil
	})
	defer restoreListen()

	allocator := newDeterministicPortAllocator(41000, 3)

	firstPort, err := allocator.NextAvailablePort()
	if err != nil {
		t.Fatalf("NextAvailablePort #1: %v", err)
	}

	secondPort, err := allocator.NextAvailablePort()
	if err != nil {
		t.Fatalf("NextAvailablePort #2: %v", err)
	}

	if firstPort != 41000 {
		t.Errorf("firstPort = %d, want 41000", firstPort)
	}
	if secondPort != 41001 {
		t.Errorf("secondPort = %d, want 41001", secondPort)
	}
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
	if err != nil {
		t.Fatalf("NextAvailablePort: %v", err)
	}
	if port != 42001 {
		t.Errorf("port = %d, want 42001", port)
	}
}

func TestDeterministicPortAllocator_ReturnsErrorWhenRangeIsExhausted(t *testing.T) {
	restoreListen := stubListenLocalTCP(t, func(network, address string) (net.Listener, error) {
		return nil, errors.New("port already in use")
	})
	defer restoreListen()

	allocator := newDeterministicPortAllocator(43000, 2)
	_, err := allocator.NextAvailablePort()
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "no free port in deterministic range") {
		t.Errorf("err = %v, want contains %q", err, "no free port in deterministic range")
	}
}

func TestLlamaServerArguments(t *testing.T) {
	args := llamaServerArguments(serverStartConfig{
		ModelPath:         "/models/gemma3.gguf",
		ContextSize:       2048,
		ParallelSlotCount: 4,
	}, 38123, 999)

	want := []string{
		"--model", "/models/gemma3.gguf",
		"--host", "127.0.0.1",
		"--port", "38123",
		"--n-gpu-layers", "999",
		"--ctx-size", "2048",
		"--parallel", "4",
	}
	if !reflect.DeepEqual(args, want) {
		t.Errorf("args = %v, want %v", args, want)
	}
}

func TestServerEnv_HIPVisibleDevices(t *testing.T) {
	env := serverEnv()
	var hipVals []string
	for _, e := range env {
		if strings.HasPrefix(e, "HIP_VISIBLE_DEVICES=") {
			hipVals = append(hipVals, e)
		}
	}
	want := []string{"HIP_VISIBLE_DEVICES=0"}
	if !reflect.DeepEqual(hipVals, want) {
		t.Errorf("hipVals = %v, want %v", hipVals, want)
	}
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
	want := []string{"HIP_VISIBLE_DEVICES=0"}
	if !reflect.DeepEqual(hipVals, want) {
		t.Errorf("hipVals = %v, want %v", hipVals, want)
	}
}

func TestAvailable(t *testing.T) {
	b := &rocmBackend{}
	if _, err := os.Stat("/dev/kfd"); err != nil {
		t.Skip("no ROCm hardware")
	}
	if !b.Available() {
		t.Error("b.Available() = false, want true")
	}
}

func TestServerAlive_Running(t *testing.T) {
	s := &server{processExited: make(chan struct{})}
	if !s.alive() {
		t.Error("s.alive() = false, want true")
	}
}

func TestServerAlive_Exited(t *testing.T) {
	processExited := make(chan struct{})
	close(processExited)
	s := &server{processExited: processExited, processExitError: coreerr.E("test", "process killed", nil)}
	if s.alive() {
		t.Error("s.alive() = true, want false")
	}
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
	if count != 0 {
		t.Errorf("count = %d, want 0", count)
	}
	err := m.Err()
	if err == nil {
		t.Fatal("m.Err() = nil, want error")
	}
	if !strings.Contains(err.Error(), "server has exited") {
		t.Errorf("err = %v, want contains %q", err, "server has exited")
	}
	if !strings.Contains(err.Error(), "HIP launch failure") {
		t.Errorf("err = %v, want contains %q", err, "HIP launch failure")
	}
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
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "failed after 3 attempts") {
		t.Errorf("err = %v, want contains %q", err, "failed after 3 attempts")
	}
}

func TestStartServer_RetriesOnStartupTimeout(t *testing.T) {
	restoreListen := stubListenLocalTCP(t, func(network, address string) (net.Listener, error) {
		return fakeTCPListener{address: address}, nil
	})
	defer restoreListen()

	dir := t.TempDir()
	binary := filepath.Join(dir, "fake-llama-server")
	if err := os.WriteFile(binary, []byte("#!/bin/sh\nsleep 1\n"), 0755); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

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
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "failed after 3 attempts") {
		t.Errorf("err = %v, want contains %q", err, "failed after 3 attempts")
	}
	if !strings.Contains(err.Error(), "timeout waiting for llama-server") {
		t.Errorf("err = %v, want contains %q", err, "timeout waiting for llama-server")
	}
}

func TestServerStop_GracefulSignalReturnsNil(t *testing.T) {
	processCommand := exec.Command("/bin/sleep", "60")
	if err := processCommand.Start(); err != nil {
		t.Fatalf("processCommand.Start: %v", err)
	}

	s := &server{
		processCommand: processCommand,
		processExited:  make(chan struct{}),
	}
	go func() {
		s.processExitError = processCommand.Wait()
		close(s.processExited)
	}()

	if err := s.stop(); err != nil {
		t.Fatalf("s.stop(): %v", err)
	}
	if err := s.stop(); err != nil {
		t.Fatalf("s.stop() second call should remain idempotent: %v", err)
	}
}

func TestServerWrapProcessError_IncludesProcessOutput(t *testing.T) {
	processOutput := newProcessOutputCapture(serverProcessOutputLimit)
	_, _ = processOutput.Write([]byte("HIP runtime exploded\nsecondary detail\n"))

	s := &server{processOutput: processOutput}

	err := s.wrapProcessError("server.waitReady", "llama-server exited before becoming ready", coreerr.E("test", "exit 1", nil))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "HIP runtime exploded") {
		t.Errorf("err = %v, want contains %q", err, "HIP runtime exploded")
	}
	if !strings.Contains(err.Error(), "secondary detail") {
		t.Errorf("err = %v, want contains %q", err, "secondary detail")
	}
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
	if count != 0 {
		t.Errorf("count = %d, want 0", count)
	}
	err := m.Err()
	if err == nil {
		t.Fatal("m.Err() = nil, want error")
	}
	if !strings.Contains(err.Error(), "server has exited") {
		t.Errorf("err = %v, want contains %q", err, "server has exited")
	}
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
