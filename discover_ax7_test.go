package rocm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDiscover_DiscoverModels_Good(t *testing.T) {
	dir := t.TempDir()
	modelPath := writeDiscoverTestGGUF(t, dir, "agent.gguf", [][2]any{
		{"general.architecture", "llama"},
		{"general.name", "Agent Model"},
		{"general.file_type", uint32(15)},
		{"general.size_label", "8B"},
		{"llama.context_length", uint32(8192)},
	})

	models, err := DiscoverModels(dir)
	if err != nil {
		t.Fatalf("DiscoverModels: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("len(models) = %d, want 1", len(models))
	}
	if models[0].Path != modelPath {
		t.Errorf("models[0].Path = %q, want %q", models[0].Path, modelPath)
	}
	if models[0].Architecture != "llama" {
		t.Errorf("models[0].Architecture = %q, want llama", models[0].Architecture)
	}
}

func TestDiscover_DiscoverModels_Bad(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "corrupt.gguf"), []byte("not a gguf header"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	models, err := DiscoverModels(dir)
	if err != nil {
		t.Fatalf("DiscoverModels: %v", err)
	}
	if len(models) != 0 {
		t.Errorf("len(models) = %d, want 0", len(models))
	}
}

func TestDiscover_DiscoverModels_Ugly(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "bad[")
	models, err := DiscoverModels(dir)
	if err == nil {
		t.Fatal("expected glob error, got nil")
	}
	if models != nil {
		t.Errorf("models = %v, want nil", models)
	}
	if !strings.Contains(err.Error(), "glob gguf files") {
		t.Errorf("err = %v, want contains glob gguf files", err)
	}
}
