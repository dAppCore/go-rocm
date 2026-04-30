//go:build linux && amd64

package rocm

import (
	"errors"
	"net"
	"strings"
	"testing"
)

func TestServer_OutputCapture_Write_Good(t *testing.T) {
	capture := newProcessOutputCapture(32)
	n, err := capture.Write([]byte("llama ready\n"))
	if err != nil {
		t.Fatalf("Write() = %v, want nil", err)
	}
	if n != len("llama ready\n") {
		t.Fatalf("Write() n = %d, want %d", n, len("llama ready\n"))
	}
	if capture.Summary() != "llama ready" {
		t.Fatalf("Summary() = %q, want llama ready", capture.Summary())
	}
}

func TestServer_OutputCapture_Write_Bad(t *testing.T) {
	capture := newProcessOutputCapture(0)
	n, err := capture.Write([]byte("ignored"))
	if err != nil {
		t.Fatalf("Write() = %v, want nil", err)
	}
	if n != len("ignored") {
		t.Fatalf("Write() n = %d, want %d", n, len("ignored"))
	}
	if capture.Summary() != "" {
		t.Fatalf("Summary() = %q, want empty", capture.Summary())
	}
}

func TestServer_OutputCapture_Write_Ugly(t *testing.T) {
	capture := newProcessOutputCapture(5)
	n, err := capture.Write([]byte("abcdef"))
	if err != nil {
		t.Fatalf("Write() = %v, want nil", err)
	}
	if n != len("abcdef") {
		t.Fatalf("Write() n = %d, want %d", n, len("abcdef"))
	}
	if capture.Summary() != "...bcdef" {
		t.Fatalf("Summary() = %q, want truncated tail", capture.Summary())
	}
}

func TestServer_OutputCapture_Summary_Good(t *testing.T) {
	capture := newProcessOutputCapture(128)
	_, err := capture.Write([]byte(" first line \n\n second line \n"))
	if err != nil {
		t.Fatalf("Write() = %v, want nil", err)
	}
	if capture.Summary() != "first line | second line" {
		t.Fatalf("Summary() = %q, want joined trimmed lines", capture.Summary())
	}
}

func TestServer_OutputCapture_Summary_Bad(t *testing.T) {
	capture := newProcessOutputCapture(128)
	summary := capture.Summary()
	if summary != "" {
		t.Fatalf("Summary() = %q, want empty", summary)
	}
	if capture.truncated {
		t.Fatal("new capture should not be marked truncated")
	}
}

func TestServer_OutputCapture_Summary_Ugly(t *testing.T) {
	capture := newProcessOutputCapture(serverProcessOutputSummarySize + 128)
	_, err := capture.Write([]byte(strings.Repeat("x", serverProcessOutputSummarySize+32)))
	if err != nil {
		t.Fatalf("Write() = %v, want nil", err)
	}
	summary := capture.Summary()
	if !strings.HasSuffix(summary, "...") {
		t.Fatalf("Summary() = %q, want ellipsis suffix", summary)
	}
}

func TestServer_PortAllocator_NextAvailablePort_Good(t *testing.T) {
	restoreListen := stubListenLocalTCP(t, func(network, address string) (net.Listener, error) {
		return fakeTCPListener{address: address}, nil
	})
	defer restoreListen()

	allocator := newDeterministicPortAllocator(41000, 2)
	port, err := allocator.NextAvailablePort()
	if err != nil {
		t.Fatalf("NextAvailablePort() = %v, want nil", err)
	}
	if port != 41000 {
		t.Fatalf("port = %d, want 41000", port)
	}
}

func TestServer_PortAllocator_NextAvailablePort_Bad(t *testing.T) {
	allocator := newDeterministicPortAllocator(0, 2)
	port, err := allocator.NextAvailablePort()
	if err == nil {
		t.Fatal("NextAvailablePort() error = nil, want invalid range error")
	}
	if port != 0 {
		t.Fatalf("port = %d, want 0", port)
	}
}

func TestServer_PortAllocator_NextAvailablePort_Ugly(t *testing.T) {
	restoreListen := stubListenLocalTCP(t, func(network, address string) (net.Listener, error) {
		if address == "127.0.0.1:42000" {
			return nil, errors.New("occupied")
		}
		return fakeTCPListener{address: address}, nil
	})
	defer restoreListen()

	allocator := newDeterministicPortAllocator(42000, 2)
	port, err := allocator.NextAvailablePort()
	if err != nil {
		t.Fatalf("NextAvailablePort() = %v, want nil", err)
	}
	if port != 42001 {
		t.Fatalf("port = %d, want skipped occupied port 42001", port)
	}
}
