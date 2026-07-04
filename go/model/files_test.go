// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestInspectModelPackFiles_Good_FilePathResolvesRoot(t *testing.T) {
	dir := t.TempDir()
	weight := filepath.Join(dir, "model.safetensors")
	writeModelPackFile(t, weight)
	writeModelPackFile(t, filepath.Join(dir, "config.json"))
	writeModelPackFile(t, filepath.Join(dir, "tokenizer.json"))

	manifest, err := InspectModelPackFiles(weight)
	if err != nil {
		t.Fatalf("InspectModelPackFiles(file): %v", err)
	}
	if manifest.Contract != ModelPackFileManifestContract ||
		manifest.SourcePath != weight ||
		manifest.Root != dir ||
		manifest.SourceIsDir ||
		manifest.Format != ModelPackFormatSafetensors ||
		manifest.Status != ModelPackFilesStatusReady ||
		len(manifest.WeightFiles) != 1 ||
		len(manifest.LoadWeightFiles) != 1 ||
		manifest.WeightFiles[0].Path != weight ||
		manifest.ConfigPath == "" ||
		manifest.TokenizerPath == "" ||
		manifest.Labels["model_pack_file_manifest_contract"] != ModelPackFileManifestContract ||
		manifest.Labels["model_pack_source_is_dir"] != "false" ||
		manifest.Labels["model_pack_load_weight_file_names"] != "model.safetensors" {
		t.Fatalf("manifest = %+v, want single-file safetensors root manifest", manifest)
	}

	root, err := ResolveModelPackRoot(weight)
	if err != nil || root != dir {
		t.Fatalf("ResolveModelPackRoot(%q) = %q err=%v, want %q", weight, root, err, dir)
	}
}

func TestInspectModelPackFiles_Good_DirectoryPrefersSafetensorsForLoad(t *testing.T) {
	dir := t.TempDir()
	gguf := filepath.Join(dir, "model.gguf")
	safetensors := filepath.Join(dir, "model.safetensors")
	nested := filepath.Join(dir, "nested", "shard.safetensors")
	writeModelPackFile(t, gguf)
	writeModelPackFile(t, safetensors)
	writeModelPackFile(t, nested)
	writeModelPackFile(t, filepath.Join(dir, ".cache", "ignored.safetensors"))
	writeModelPackFile(t, filepath.Join(dir, "model.safetensors.index.json"))

	manifest, err := InspectModelPackFiles(dir)
	if err != nil {
		t.Fatalf("InspectModelPackFiles(dir): %v", err)
	}
	if !manifest.SourceIsDir ||
		manifest.Format != ModelPackFormatMixed ||
		manifest.Status != ModelPackFilesStatusReady ||
		manifest.GGUFCount != 1 ||
		manifest.SafetensorsCount != 2 ||
		len(manifest.WeightFiles) != 3 ||
		len(manifest.LoadWeightFiles) != 2 ||
		manifest.MixedWeights != true ||
		manifest.SafetensorsIndexPath == "" ||
		!slices.Equal(manifest.LoadWeightPaths(), []string{safetensors, nested}) ||
		manifest.Labels["model_pack_mixed_weights"] != "true" ||
		manifest.Labels["model_pack_safetensors_index"] != "true" {
		t.Fatalf("manifest = %+v, want mixed diagnostic manifest with safetensors load preference", manifest)
	}
}

func TestInspectModelPackFiles_Good_MultipleGGUFIsAmbiguousForLoad(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, filepath.Join(dir, "a.gguf"))
	writeModelPackFile(t, filepath.Join(dir, "b.gguf"))

	manifest, err := InspectModelPackFiles(dir)
	if err != nil {
		t.Fatalf("InspectModelPackFiles(dir): %v", err)
	}
	if manifest.Format != ModelPackFormatGGUF ||
		manifest.Status != ModelPackFilesStatusAmbiguousGGUF ||
		!manifest.AmbiguousGGUF ||
		len(manifest.WeightFiles) != 2 ||
		len(manifest.LoadWeightFiles) != 0 ||
		manifest.Labels["model_pack_ambiguous_gguf"] != "true" {
		t.Fatalf("manifest = %+v, want go-mlx-style ambiguous GGUF load status", manifest)
	}
}

func TestInspectModelPackFiles_Good_MissingWeights(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, filepath.Join(dir, "config.json"))

	manifest, err := InspectModelPackFiles(dir)
	if err != nil {
		t.Fatalf("InspectModelPackFiles(empty): %v", err)
	}
	if manifest.Format != ModelPackFormatMissing ||
		manifest.Status != ModelPackFilesStatusMissing ||
		!manifest.MissingWeights ||
		len(manifest.WeightFiles) != 0 ||
		manifest.Labels["model_pack_missing_weights"] != "true" ||
		manifest.Labels["model_pack_config"] != "true" {
		t.Fatalf("manifest = %+v, want missing-weight manifest with config sidecar", manifest)
	}
}

func TestInspectModelPackFiles_Good_CopySafe(t *testing.T) {
	dir := t.TempDir()
	weight := filepath.Join(dir, "model.safetensors")
	writeModelPackFile(t, weight)

	manifest, err := InspectModelPackFiles(dir)
	if err != nil {
		t.Fatalf("InspectModelPackFiles(dir): %v", err)
	}
	manifest.WeightFiles[0].Path = "mutated"
	manifest.LoadWeightFiles[0].Path = "mutated"
	manifest.Labels["model_pack_root"] = "mutated"

	next, err := InspectModelPackFiles(dir)
	if err != nil {
		t.Fatalf("InspectModelPackFiles(dir): %v", err)
	}
	if next.WeightFiles[0].Path != weight ||
		next.LoadWeightFiles[0].Path != weight ||
		next.Labels["model_pack_root"] != dir {
		t.Fatalf("manifest leaked mutable state: %+v", next)
	}
}

func writeModelPackFile(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte("{}"), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
