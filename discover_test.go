package rocm

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeDiscoverTestGGUF creates a minimal GGUF v3 file in dir with the given
// filename and metadata KV pairs. Returns the full path to the created file.
func writeDiscoverTestGGUF(t *testing.T, dir, filename string, kvs [][2]any) string {
	t.Helper()

	path := filepath.Join(dir, filename)

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("os.Create: %v", err)
	}
	defer f.Close()

	// Magic: "GGUF" in little-endian
	if err := binary.Write(f, binary.LittleEndian, uint32(0x46554747)); err != nil {
		t.Fatalf("write magic: %v", err)
	}
	// Version 3
	if err := binary.Write(f, binary.LittleEndian, uint32(3)); err != nil {
		t.Fatalf("write version: %v", err)
	}
	// Tensor count (uint64): 0
	if err := binary.Write(f, binary.LittleEndian, uint64(0)); err != nil {
		t.Fatalf("write tensor count: %v", err)
	}
	// KV count (uint64)
	if err := binary.Write(f, binary.LittleEndian, uint64(len(kvs))); err != nil {
		t.Fatalf("write kv count: %v", err)
	}

	for _, kv := range kvs {
		key := kv[0].(string)
		writeDiscoverKV(t, f, key, kv[1])
	}

	return path
}

func writeDiscoverKV(t *testing.T, f *os.File, key string, val any) {
	t.Helper()

	// Key: uint64 length + bytes
	if err := binary.Write(f, binary.LittleEndian, uint64(len(key))); err != nil {
		t.Fatalf("write key len: %v", err)
	}
	if _, err := f.Write([]byte(key)); err != nil {
		t.Fatalf("write key bytes: %v", err)
	}

	switch v := val.(type) {
	case string:
		// Type: 8 (string)
		if err := binary.Write(f, binary.LittleEndian, uint32(8)); err != nil {
			t.Fatalf("write string type: %v", err)
		}
		// String value: uint64 length + bytes
		if err := binary.Write(f, binary.LittleEndian, uint64(len(v))); err != nil {
			t.Fatalf("write string len: %v", err)
		}
		if _, err := f.Write([]byte(v)); err != nil {
			t.Fatalf("write string bytes: %v", err)
		}
	case uint32:
		// Type: 4 (uint32)
		if err := binary.Write(f, binary.LittleEndian, uint32(4)); err != nil {
			t.Fatalf("write uint32 type: %v", err)
		}
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			t.Fatalf("write uint32 val: %v", err)
		}
	default:
		t.Fatalf("writeDiscoverKV: unsupported value type %T", val)
	}
}

func TestDiscoverModels(t *testing.T) {
	dir := t.TempDir()

	// Create two valid GGUF model files.
	writeDiscoverTestGGUF(t, dir, "gemma3-4b-q4km.gguf", [][2]any{
		{"general.architecture", "gemma3"},
		{"general.name", "Gemma 3 4B Instruct"},
		{"general.file_type", uint32(15)},
		{"general.size_label", "4B"},
		{"gemma3.context_length", uint32(32768)},
		{"gemma3.block_count", uint32(34)},
	})

	writeDiscoverTestGGUF(t, dir, "llama-3.1-8b-q4km.gguf", [][2]any{
		{"general.architecture", "llama"},
		{"general.name", "Llama 3.1 8B Instruct"},
		{"general.file_type", uint32(15)},
		{"general.size_label", "8B"},
		{"llama.context_length", uint32(131072)},
		{"llama.block_count", uint32(32)},
	})

	// Create a non-GGUF file that should be ignored (no .gguf extension).
	if err := os.WriteFile(filepath.Join(dir, "README.txt"), []byte("not a model"), 0644); err != nil {
		t.Fatalf("WriteFile README: %v", err)
	}

	models, err := DiscoverModels(dir)
	if err != nil {
		t.Fatalf("DiscoverModels: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("len(models) = %d, want 2", len(models))
	}

	// Sort order from Glob is lexicographic, so gemma3 comes first.
	gemma := models[0]
	if got, want := gemma.Path, filepath.Join(dir, "gemma3-4b-q4km.gguf"); got != want {
		t.Errorf("gemma.Path = %q, want %q", got, want)
	}
	if gemma.Architecture != "gemma3" {
		t.Errorf("gemma.Architecture = %q, want %q", gemma.Architecture, "gemma3")
	}
	if gemma.Name != "Gemma 3 4B Instruct" {
		t.Errorf("gemma.Name = %q, want %q", gemma.Name, "Gemma 3 4B Instruct")
	}
	if gemma.Quantisation != "Q4_K_M" {
		t.Errorf("gemma.Quantisation = %q, want %q", gemma.Quantisation, "Q4_K_M")
	}
	if gemma.Parameters != "4B" {
		t.Errorf("gemma.Parameters = %q, want %q", gemma.Parameters, "4B")
	}
	if gemma.ContextLen != uint32(32768) {
		t.Errorf("gemma.ContextLen = %d, want 32768", gemma.ContextLen)
	}
	if gemma.FileSize <= 0 {
		t.Errorf("gemma.FileSize = %d, want > 0", gemma.FileSize)
	}

	llama := models[1]
	if got, want := llama.Path, filepath.Join(dir, "llama-3.1-8b-q4km.gguf"); got != want {
		t.Errorf("llama.Path = %q, want %q", got, want)
	}
	if llama.Architecture != "llama" {
		t.Errorf("llama.Architecture = %q, want %q", llama.Architecture, "llama")
	}
	if llama.Name != "Llama 3.1 8B Instruct" {
		t.Errorf("llama.Name = %q, want %q", llama.Name, "Llama 3.1 8B Instruct")
	}
	if llama.Quantisation != "Q4_K_M" {
		t.Errorf("llama.Quantisation = %q, want %q", llama.Quantisation, "Q4_K_M")
	}
	if llama.Parameters != "8B" {
		t.Errorf("llama.Parameters = %q, want %q", llama.Parameters, "8B")
	}
	if llama.ContextLen != uint32(131072) {
		t.Errorf("llama.ContextLen = %d, want 131072", llama.ContextLen)
	}
	if llama.FileSize <= 0 {
		t.Errorf("llama.FileSize = %d, want > 0", llama.FileSize)
	}
}

func TestDiscoverModels_RelativeDirReturnsAbsolutePaths(t *testing.T) {
	parent := t.TempDir()
	dir := filepath.Join(parent, "models")
	if err := os.Mkdir(dir, 0755); err != nil {
		t.Fatalf("Mkdir: %v", err)
	}

	path := writeDiscoverTestGGUF(t, dir, "model.gguf", [][2]any{
		{"general.architecture", "llama"},
		{"general.name", "Relative Model"},
		{"general.file_type", uint32(15)},
	})

	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if err := os.Chdir(parent); err != nil {
		t.Fatalf("Chdir parent: %v", err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(wd); err != nil {
			t.Fatalf("Chdir restore: %v", err)
		}
	})

	models, err := DiscoverModels("models")
	if err != nil {
		t.Fatalf("DiscoverModels: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("len(models) = %d, want 1", len(models))
	}
	gotPath, err := filepath.EvalSymlinks(models[0].Path)
	if err != nil {
		t.Fatalf("EvalSymlinks got path: %v", err)
	}
	wantPath, err := filepath.EvalSymlinks(path)
	if err != nil {
		t.Fatalf("EvalSymlinks want path: %v", err)
	}
	if gotPath != wantPath {
		t.Errorf("models[0].Path = %q, want %q", gotPath, wantPath)
	}
}

func TestDiscoverModels_EmptyDir(t *testing.T) {
	dir := t.TempDir()

	models, err := DiscoverModels(dir)
	if err != nil {
		t.Fatalf("DiscoverModels: %v", err)
	}
	if len(models) != 0 {
		t.Errorf("len(models) = %d, want 0", len(models))
	}
}

func TestDiscoverModels_NotFound(t *testing.T) {
	// filepath.Glob returns nil, nil for a pattern matching no files,
	// even when the directory does not exist.
	models, err := DiscoverModels("/nonexistent/dir")
	if err != nil {
		t.Fatalf("DiscoverModels: %v", err)
	}
	if len(models) != 0 {
		t.Errorf("len(models) = %d, want 0", len(models))
	}
}

func TestDiscoverModels_BadPattern(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "bad[")

	_, err := DiscoverModels(dir)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "glob gguf files") {
		t.Errorf("err = %v, want contains %q", err, "glob gguf files")
	}
}

func TestDiscoverModels_SkipsCorruptFile(t *testing.T) {
	dir := t.TempDir()

	// Create a valid GGUF file.
	writeDiscoverTestGGUF(t, dir, "valid.gguf", [][2]any{
		{"general.architecture", "llama"},
		{"general.name", "Valid Model"},
		{"general.file_type", uint32(15)},
	})

	// Create a corrupt .gguf file (not valid GGUF binary).
	if err := os.WriteFile(filepath.Join(dir, "corrupt.gguf"), []byte("not gguf data"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	models, err := DiscoverModels(dir)
	if err != nil {
		t.Fatalf("DiscoverModels: %v", err)
	}
	// Only the valid model should be returned; corrupt one is silently skipped.
	if len(models) != 1 {
		t.Fatalf("len(models) = %d, want 1", len(models))
	}
	if models[0].Name != "Valid Model" {
		t.Errorf("models[0].Name = %q, want %q", models[0].Name, "Valid Model")
	}
}
