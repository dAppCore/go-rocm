//go:build linux && amd64

package rocm

import (
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"dappco.re/go/inference"
)

func TestMain(m *testing.M) {
	if os.Getenv("ROCM_FAKE_LLAMA_SERVER") == "1" {
		runFakeLlamaServer()
		return
	}
	os.Exit(m.Run())
}

func runFakeLlamaServer() {
	port := ""
	for i, arg := range os.Args {
		if arg == "--port" && i+1 < len(os.Args) {
			port = os.Args[i+1]
		}
	}
	if port == "" {
		os.Exit(2)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, _ *http.Request) {
		writeSSEEvent(w, `{"choices":[{"text":"ok","finish_reason":null}]}`)
		writeSSEEvent(w, "[DONE]")
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, _ *http.Request) {
		writeSSEEvent(w, `{"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}`)
		writeSSEEvent(w, "[DONE]")
	})

	if err := http.ListenAndServe("127.0.0.1:"+port, mux); err != nil {
		os.Exit(3)
	}
}

func useFakeLlamaServer(t *testing.T) {
	t.Helper()
	t.Setenv("ROCM_FAKE_LLAMA_SERVER", "1")
	t.Setenv("ROCM_LLAMA_SERVER_PATH", os.Args[0])
}

func TestBackend_Backend_Name_Good(t *testing.T) {
	name := (&rocmBackend{}).Name()
	if name != "rocm" {
		t.Fatalf("Name() = %q, want rocm", name)
	}
	if len(name) == 0 {
		t.Fatal("Name() returned an empty backend name")
	}
}

func TestBackend_Backend_Name_Bad(t *testing.T) {
	var backend *rocmBackend
	name := backend.Name()
	if name != "rocm" {
		t.Fatalf("nil receiver Name() = %q, want rocm", name)
	}
	if name == "cpu" {
		t.Fatal("Name() returned the wrong backend family")
	}
}

func TestBackend_Backend_Name_Ugly(t *testing.T) {
	backend := &rocmBackend{}
	first := backend.Name()
	second := backend.Name()
	if first != second {
		t.Fatalf("Name() changed from %q to %q", first, second)
	}
}

func TestBackend_Backend_Available_Good(t *testing.T) {
	backend := &rocmBackend{}
	available := backend.Available()
	if available {
		if _, err := os.Stat("/dev/kfd"); err != nil {
			t.Fatalf("Available() = true but /dev/kfd stat failed: %v", err)
		}
	}
	if available != backend.Available() {
		t.Fatal("Available() changed between adjacent calls")
	}
}

func TestBackend_Backend_Available_Bad(t *testing.T) {
	t.Setenv("ROCM_LLAMA_SERVER_PATH", filepath.Join(t.TempDir(), "missing-llama-server"))
	available := (&rocmBackend{}).Available()
	if available {
		t.Fatal("Available() = true with missing llama-server override")
	}
	if ROCmAvailable() != true {
		t.Fatal("ROCmAvailable() should still report compiled support")
	}
}

func TestBackend_Backend_Available_Ugly(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ROCM_LLAMA_SERVER_PATH", dir)
	available := (&rocmBackend{}).Available()
	if available {
		t.Fatal("Available() = true with directory llama-server override")
	}
	if _, err := validateLlamaServerPath(dir); err == nil {
		t.Fatal("validateLlamaServerPath(directory) error = nil, want error")
	}
}

func TestBackend_Backend_LoadModel_Good(t *testing.T) {
	useFakeLlamaServer(t)
	dir := t.TempDir()
	modelPath := writeDiscoverTestGGUF(t, dir, "model.gguf", [][2]any{
		{"general.architecture", "llama"},
		{"general.name", "AX Model"},
		{"general.file_type", uint32(15)},
		{"llama.context_length", uint32(4096)},
		{"llama.block_count", uint32(32)},
	})

	model, err := (&rocmBackend{}).LoadModel(modelPath, inference.WithContextLen(1024))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer model.Close()
	if model.ModelType() != "llama" {
		t.Fatalf("ModelType() = %q, want llama", model.ModelType())
	}
	if model.Info().NumLayers != 32 {
		t.Fatalf("Info().NumLayers = %d, want 32", model.Info().NumLayers)
	}
}

func TestBackend_Backend_LoadModel_Bad(t *testing.T) {
	t.Setenv("ROCM_LLAMA_SERVER_PATH", filepath.Join(t.TempDir(), "missing-llama-server"))
	model, err := (&rocmBackend{}).LoadModel("/missing/model.gguf")
	if err == nil {
		t.Fatal("LoadModel() error = nil, want missing llama-server error")
	}
	if model != nil {
		t.Fatalf("model = %v, want nil", model)
	}
}

func TestBackend_Backend_LoadModel_Ugly(t *testing.T) {
	useFakeLlamaServer(t)
	path := filepath.Join(t.TempDir(), "corrupt.gguf")
	if err := os.WriteFile(path, []byte("not gguf"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	model, err := (&rocmBackend{}).LoadModel(path)
	if err == nil {
		t.Fatal("LoadModel() error = nil, want metadata error")
	}
	if model != nil {
		t.Fatalf("model = %v, want nil", model)
	}
}
