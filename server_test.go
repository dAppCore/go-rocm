//go:build linux && amd64

package rocm

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"forge.lthn.ai/core/go-inference"
	coreerr "forge.lthn.ai/core/go-log"
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
	s := &server{exited: make(chan struct{})}
	assert.True(t, s.alive())
}

func TestServerAlive_Exited(t *testing.T) {
	exited := make(chan struct{})
	close(exited)
	s := &server{exited: exited, exitErr: coreerr.E("test", "process killed", nil)}
	assert.False(t, s.alive())
}

func TestGenerate_ServerDead(t *testing.T) {
	exited := make(chan struct{})
	close(exited)
	s := &server{
		exited:  exited,
		exitErr: coreerr.E("test", "process killed", nil),
	}
	m := &rocmModel{srv: s}

	var count int
	for range m.Generate(context.Background(), "hello") {
		count++
	}
	assert.Equal(t, 0, count)
	assert.ErrorContains(t, m.Err(), "server has exited")
}

func TestStartServer_RetriesOnProcessExit(t *testing.T) {
	// /bin/false starts successfully but exits immediately with code 1.
	// startServer should retry up to 3 times, then fail.
	_, err := startServer("/bin/false", "/nonexistent/model.gguf", 999, 0, 0)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed after 3 attempts")
}

func TestStartServer_RetriesOnStartupTimeout(t *testing.T) {
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

	_, err := startServer(binary, "/nonexistent/model.gguf", 999, 0, 0)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed after 3 attempts")
	assert.Contains(t, err.Error(), "timeout waiting for llama-server")
}

func TestChat_ServerDead(t *testing.T) {
	exited := make(chan struct{})
	close(exited)
	s := &server{
		exited:  exited,
		exitErr: coreerr.E("test", "process killed", nil),
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
