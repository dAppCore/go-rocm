package gguf

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGGUF_FileTypeName_Good(t *testing.T) {
	name := FileTypeName(15)
	if name != "Q4_K_M" {
		t.Fatalf("FileTypeName(15) = %q, want Q4_K_M", name)
	}
	if FileTypeName(17) != "Q5_K_M" {
		t.Fatalf("FileTypeName(17) = %q, want Q5_K_M", FileTypeName(17))
	}
}

func TestGGUF_FileTypeName_Bad(t *testing.T) {
	name := FileTypeName(999)
	if name != "type_999" {
		t.Fatalf("FileTypeName(999) = %q, want type_999", name)
	}
	if !strings.HasPrefix(name, "type_") {
		t.Fatalf("unknown file type name = %q, want type_ prefix", name)
	}
}

func TestGGUF_FileTypeName_Ugly(t *testing.T) {
	zero := FileTypeName(0)
	large := FileTypeName(^uint32(0))
	if zero != "F32" {
		t.Fatalf("FileTypeName(0) = %q, want F32", zero)
	}
	if large != "type_4294967295" {
		t.Fatalf("FileTypeName(MaxUint32) = %q, want generated name", large)
	}
}

func TestGGUF_ReadMetadata_Good(t *testing.T) {
	path := writeTestGGUFOrdered(t, [][2]any{
		{"general.architecture", "llama"},
		{"general.name", "AX Llama"},
		{"general.file_type", uint32(15)},
		{"general.size_label", "8B"},
		{"llama.context_length", uint32(4096)},
		{"llama.block_count", uint32(32)},
	})

	metadata, err := ReadMetadata(path)
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}
	if metadata.Architecture != "llama" || metadata.Name != "AX Llama" {
		t.Fatalf("metadata = %+v, want llama metadata", metadata)
	}
}

func TestGGUF_ReadMetadata_Bad(t *testing.T) {
	metadata, err := ReadMetadata(filepath.Join(t.TempDir(), "missing.gguf"))
	if err == nil {
		t.Fatal("ReadMetadata() error = nil, want open error")
	}
	if metadata != (Metadata{}) {
		t.Fatalf("metadata = %+v, want zero value", metadata)
	}
}

func TestGGUF_ReadMetadata_Ugly(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "truncated.gguf")
	if err := os.WriteFile(path, []byte("GGUF"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	metadata, err := ReadMetadata(path)
	if err == nil {
		t.Fatal("ReadMetadata() error = nil, want truncated header error")
	}
	if metadata != (Metadata{}) {
		t.Fatalf("metadata = %+v, want zero value", metadata)
	}
}
