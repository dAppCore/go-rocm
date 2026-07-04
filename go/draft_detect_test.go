// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDraftDetect_AssistantDir_Good(t *testing.T) {
	model := t.TempDir()
	writeDraftDetectModelDir(t, model, "gemma4_text")
	writeDraftDetectModelDir(t, filepath.Join(model, "assistant"), "gemma4_assistant")

	detection := DetectGemma4DraftPath(model, "", DraftDetectOptions{})

	if !detection.Active() || detection.Source != DraftSourceAssistantDir {
		t.Fatalf("detection = %+v, want assistant-dir", detection)
	}
	if detection.DraftPath != filepath.Join(model, "assistant") {
		t.Fatalf("draft path = %q, want assistant dir", detection.DraftPath)
	}
}

func TestDraftDetect_PairBundleSibling_Good(t *testing.T) {
	bundle := t.TempDir()
	target := filepath.Join(bundle, "target")
	writeDraftDetectModelDir(t, target, "gemma4_text")
	writeDraftDetectModelDir(t, filepath.Join(bundle, "assistant"), "gemma4_assistant")
	if err := os.WriteFile(filepath.Join(bundle, "mtplx_pair.json"), []byte(`{"layout":"ignored"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	detection := DetectGemma4DraftPath(target, "", DraftDetectOptions{})

	if !detection.Active() || detection.Source != DraftSourceSiblingAssistant {
		t.Fatalf("detection = %+v, want sibling assistant", detection)
	}
	if detection.DraftPath != filepath.Join(bundle, "assistant") {
		t.Fatalf("draft path = %q, want bundle assistant", detection.DraftPath)
	}
}

func TestDraftDetect_MTPLayouts_Good(t *testing.T) {
	model := t.TempDir()
	writeDraftDetectModelDir(t, model, "gemma4_text")
	if err := os.Mkdir(filepath.Join(model, "MTP"), 0o755); err != nil {
		t.Fatal(err)
	}
	mtp := filepath.Join(model, "MTP", "gemma-4-31B-it-Q8_0-MTP.gguf")
	if err := os.WriteFile(mtp, []byte("stub"), 0o644); err != nil {
		t.Fatal(err)
	}
	detection := DetectGemma4DraftPath(model, "", DraftDetectOptions{})
	if !detection.Active() || detection.Source != DraftSourceMTPDir || detection.DraftPath != mtp {
		t.Fatalf("detection = %+v, want MTP/ gguf", detection)
	}
	if detection.Note != "auto-detected MTP/ drafter (unsloth GGUF convention)" {
		t.Fatalf("MTP/ note = %q, want go-mlx-compatible operator note", detection.Note)
	}

	siblingModel := t.TempDir()
	writeDraftDetectModelDir(t, siblingModel, "gemma4_text")
	sibling := filepath.Join(siblingModel, "mtp-gemma-4-31B.gguf")
	if err := os.WriteFile(sibling, []byte("stub"), 0o644); err != nil {
		t.Fatal(err)
	}
	detection = DetectGemma4DraftPath(siblingModel, "", DraftDetectOptions{})
	if !detection.Active() || detection.Source != DraftSourceMTPSibling || detection.DraftPath != sibling {
		t.Fatalf("detection = %+v, want sibling mtp-*.gguf", detection)
	}
}

func TestDraftDetect_PrecedenceAndDisable_Good(t *testing.T) {
	model := t.TempDir()
	writeDraftDetectModelDir(t, model, "gemma4_text")
	writeDraftDetectModelDir(t, filepath.Join(model, "assistant"), "gemma4_assistant")
	if err := os.Mkdir(filepath.Join(model, "MTP"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(model, "MTP", "x-MTP.gguf"), []byte("stub"), 0o644); err != nil {
		t.Fatal(err)
	}

	detection := DetectGemma4DraftPath(model, "/explicit/drafter", DraftDetectOptions{})
	if detection.Source != DraftSourceFlag || detection.DraftPath != "/explicit/drafter" {
		t.Fatalf("detection = %+v, want explicit flag", detection)
	}
	detection = DetectGemma4DraftPath(model, "", DraftDetectOptions{})
	if detection.Source != DraftSourceAssistantDir {
		t.Fatalf("detection = %+v, want assistant before MTP", detection)
	}
	detection = DetectGemma4DraftPath(model, "", DraftDetectOptions{Disabled: true})
	if detection.Active() || detection.Note != "drafter detection disabled" {
		t.Fatalf("detection = %+v, want disabled inactive note", detection)
	}
}

func TestDraftDetect_DegenerateShapes_Bad(t *testing.T) {
	model := t.TempDir()
	writeDraftDetectModelDir(t, model, "qwen3")
	writeDraftDetectModelDir(t, filepath.Join(model, "assistant"), "gemma4_assistant")
	if detection := DetectGemma4DraftPath(model, "", DraftDetectOptions{}); detection.Active() {
		t.Fatalf("detection = %+v, want non-Gemma target inactive", detection)
	}

	gemma := t.TempDir()
	writeDraftDetectModelDir(t, gemma, "gemma4_text")
	if err := os.Mkdir(filepath.Join(gemma, "assistant"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(gemma, "assistant", "config.json"), []byte(`{"model_type":"gemma4_assistant"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if detection := DetectGemma4DraftPath(gemma, "", DraftDetectOptions{}); detection.Active() {
		t.Fatalf("detection = %+v, want weightless assistant inactive", detection)
	}
	if err := os.Mkdir(filepath.Join(gemma, "MTP"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"a-MTP.gguf", "b-MTP.gguf"} {
		if err := os.WriteFile(filepath.Join(gemma, "MTP", name), []byte("stub"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if detection := DetectGemma4DraftPath(gemma, "", DraftDetectOptions{}); detection.Active() {
		t.Fatalf("detection = %+v, want ambiguous MTP inactive", detection)
	}
	if detection := DetectGemma4DraftPath("", "", DraftDetectOptions{}); detection.Active() {
		t.Fatalf("detection = %+v, want blank path inactive", detection)
	}
}

func writeDraftDetectModelDir(t *testing.T, dir, modelType string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"`+modelType+`"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("stub"), 0o644); err != nil {
		t.Fatal(err)
	}
}
