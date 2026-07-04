// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestROCmModelFiles_Good_RootAPIResolvesFilePath(t *testing.T) {
	dir := t.TempDir()
	weight := filepath.Join(dir, "model.safetensors")
	writeROCmModelFileTestFile(t, weight)
	writeROCmModelFileTestFile(t, filepath.Join(dir, "config.json"))
	writeROCmModelFileTestFile(t, filepath.Join(dir, "tokenizer.json"))

	manifest, err := InspectROCmModelPackFiles(weight)
	if err != nil {
		t.Fatalf("InspectROCmModelPackFiles(file): %v", err)
	}
	if manifest.Contract != ROCmModelPackFileManifestContract ||
		manifest.SourcePath != weight ||
		manifest.Root != dir ||
		manifest.SourceIsDir ||
		manifest.Format != ROCmModelPackFormatSafetensors ||
		manifest.Status != ROCmModelPackFilesStatusReady ||
		len(manifest.LoadWeightFiles) != 1 ||
		manifest.LoadWeightFiles[0].Path != weight ||
		manifest.ConfigPath == "" ||
		manifest.TokenizerPath == "" {
		t.Fatalf("manifest = %+v, want root ROCm file manifest matching go-mlx file-root contract", manifest)
	}

	root, err := ResolveROCmModelRoot(weight)
	if err != nil || root != dir {
		t.Fatalf("ResolveROCmModelRoot(%q) = %q err=%v, want %q", weight, root, err, dir)
	}
	paths, err := ROCmModelLoadWeightPaths(weight)
	if err != nil || !slices.Equal(paths, []string{weight}) {
		t.Fatalf("ROCmModelLoadWeightPaths(%q) = %v err=%v, want single safetensors file", weight, paths, err)
	}
}

func TestROCmModelFiles_Good_RootAPIPrefersSafetensorsAndFlagsAmbiguousGGUF(t *testing.T) {
	dir := t.TempDir()
	gguf := filepath.Join(dir, "model.gguf")
	safetensors := filepath.Join(dir, "model.safetensors")
	shard := filepath.Join(dir, "shard.safetensors")
	writeROCmModelFileTestFile(t, gguf)
	writeROCmModelFileTestFile(t, safetensors)
	writeROCmModelFileTestFile(t, shard)

	loadFiles, err := ROCmModelLoadWeightFiles(dir)
	if err != nil {
		t.Fatalf("ROCmModelLoadWeightFiles(mixed): %v", err)
	}
	if len(loadFiles) != 2 ||
		loadFiles[0].Path != safetensors ||
		loadFiles[1].Path != shard {
		t.Fatalf("ROCmModelLoadWeightFiles(mixed) = %+v, want safetensors load preference", loadFiles)
	}
	loadFiles[0].Path = "mutated"
	next, err := ROCmModelLoadWeightFiles(dir)
	if err != nil || next[0].Path != safetensors {
		t.Fatalf("ROCmModelLoadWeightFiles leaked mutable state: %+v err=%v", next, err)
	}

	ambiguous := t.TempDir()
	writeROCmModelFileTestFile(t, filepath.Join(ambiguous, "a.gguf"))
	writeROCmModelFileTestFile(t, filepath.Join(ambiguous, "b.gguf"))
	manifest, err := InspectROCmModelPackFiles(ambiguous)
	if err != nil {
		t.Fatalf("InspectROCmModelPackFiles(ambiguous): %v", err)
	}
	if manifest.Status != ROCmModelPackFilesStatusAmbiguousGGUF ||
		manifest.Format != ROCmModelPackFormatGGUF ||
		len(manifest.LoadWeightFiles) != 0 ||
		!manifest.AmbiguousGGUF {
		t.Fatalf("ambiguous manifest = %+v, want no default load file for multiple GGUFs", manifest)
	}
}

func writeROCmModelFileTestFile(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte("{}"), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
