package gguf

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeTestGGUFOrdered creates a synthetic GGUF v3 file with KV pairs in the
// exact order specified. Each element is a [2]any{key string, value any}.
func writeTestGGUFOrdered(t *testing.T, kvs [][2]any) string {
	t.Helper()

	dir := t.TempDir()
	path := filepath.Join(dir, "test.gguf")

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("os.Create: %v", err)
	}
	defer f.Close()

	// Magic
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
		writeKV(t, f, key, kv[1])
	}

	return path
}

func writeKV(t *testing.T, f *os.File, key string, val any) {
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
	case uint64:
		// Type: 10 (uint64)
		if err := binary.Write(f, binary.LittleEndian, uint32(10)); err != nil {
			t.Fatalf("write uint64 type: %v", err)
		}
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			t.Fatalf("write uint64 val: %v", err)
		}
	default:
		t.Fatalf("writeKV: unsupported value type %T", val)
	}
}

// writeRawKV writes a key with a specific GGUF type and raw byte payload.
// Used to test skipValue for types not used in interesting keys.
func writeRawKV(t *testing.T, f *os.File, key string, valType uint32, rawVal []byte) {
	t.Helper()

	if err := binary.Write(f, binary.LittleEndian, uint64(len(key))); err != nil {
		t.Fatalf("write key len: %v", err)
	}
	if _, err := f.Write([]byte(key)); err != nil {
		t.Fatalf("write key bytes: %v", err)
	}
	if err := binary.Write(f, binary.LittleEndian, valType); err != nil {
		t.Fatalf("write val type: %v", err)
	}
	if _, err := f.Write(rawVal); err != nil {
		t.Fatalf("write raw val: %v", err)
	}
}

// writeTestGGUFV2 creates a synthetic GGUF v2 file (uint32 tensor/kv counts).
func writeTestGGUFV2(t *testing.T, kvs [][2]any) string {
	t.Helper()

	dir := t.TempDir()
	path := filepath.Join(dir, "test_v2.gguf")

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("os.Create: %v", err)
	}
	defer f.Close()

	// Magic
	if err := binary.Write(f, binary.LittleEndian, uint32(0x46554747)); err != nil {
		t.Fatalf("write magic: %v", err)
	}
	// Version 2
	if err := binary.Write(f, binary.LittleEndian, uint32(2)); err != nil {
		t.Fatalf("write version: %v", err)
	}
	// Tensor count (uint32 for v2): 0
	if err := binary.Write(f, binary.LittleEndian, uint32(0)); err != nil {
		t.Fatalf("write tensor count: %v", err)
	}
	// KV count (uint32 for v2)
	if err := binary.Write(f, binary.LittleEndian, uint32(len(kvs))); err != nil {
		t.Fatalf("write kv count: %v", err)
	}

	for _, kv := range kvs {
		key := kv[0].(string)
		writeKV(t, f, key, kv[1])
	}

	return path
}

func TestReadMetadata_Gemma3(t *testing.T) {
	path := writeTestGGUFOrdered(t, [][2]any{
		{"general.architecture", "gemma3"},
		{"general.name", "Test Gemma3 1B"},
		{"general.file_type", uint32(17)},
		{"general.size_label", "1B"},
		{"gemma3.context_length", uint32(32768)},
		{"gemma3.block_count", uint32(26)},
	})

	m, err := ReadMetadata(path)
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}

	if m.Architecture != "gemma3" {
		t.Errorf("Architecture = %q, want %q", m.Architecture, "gemma3")
	}
	if m.Name != "Test Gemma3 1B" {
		t.Errorf("Name = %q, want %q", m.Name, "Test Gemma3 1B")
	}
	if m.FileType != uint32(17) {
		t.Errorf("FileType = %d, want 17", m.FileType)
	}
	if m.SizeLabel != "1B" {
		t.Errorf("SizeLabel = %q, want %q", m.SizeLabel, "1B")
	}
	if m.ContextLength != uint32(32768) {
		t.Errorf("ContextLength = %d, want 32768", m.ContextLength)
	}
	if m.BlockCount != uint32(26) {
		t.Errorf("BlockCount = %d, want 26", m.BlockCount)
	}
	if m.FileSize <= 0 {
		t.Errorf("FileSize = %d, want > 0", m.FileSize)
	}
}

func TestReadMetadata_Llama(t *testing.T) {
	path := writeTestGGUFOrdered(t, [][2]any{
		{"general.architecture", "llama"},
		{"general.name", "Test Llama 8B"},
		{"general.file_type", uint32(15)},
		{"general.size_label", "8B"},
		{"llama.context_length", uint32(131072)},
		{"llama.block_count", uint32(32)},
	})

	m, err := ReadMetadata(path)
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}

	if m.Architecture != "llama" {
		t.Errorf("Architecture = %q, want %q", m.Architecture, "llama")
	}
	if m.Name != "Test Llama 8B" {
		t.Errorf("Name = %q, want %q", m.Name, "Test Llama 8B")
	}
	if m.FileType != uint32(15) {
		t.Errorf("FileType = %d, want 15", m.FileType)
	}
	if m.SizeLabel != "8B" {
		t.Errorf("SizeLabel = %q, want %q", m.SizeLabel, "8B")
	}
	if m.ContextLength != uint32(131072) {
		t.Errorf("ContextLength = %d, want 131072", m.ContextLength)
	}
	if m.BlockCount != uint32(32) {
		t.Errorf("BlockCount = %d, want 32", m.BlockCount)
	}
	if m.FileSize <= 0 {
		t.Errorf("FileSize = %d, want > 0", m.FileSize)
	}
}

func TestReadMetadata_ArchAfterContextLength(t *testing.T) {
	// Architecture key comes AFTER the arch-specific keys.
	// The parser must handle deferred resolution of arch-prefixed keys.
	path := writeTestGGUFOrdered(t, [][2]any{
		{"general.name", "Out-of-Order Model"},
		{"general.file_type", uint32(15)},
		{"general.size_label", "8B"},
		{"llama.context_length", uint32(4096)},
		{"llama.block_count", uint32(32)},
		{"general.architecture", "llama"},
	})

	m, err := ReadMetadata(path)
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}

	if m.Architecture != "llama" {
		t.Errorf("Architecture = %q, want %q", m.Architecture, "llama")
	}
	if m.Name != "Out-of-Order Model" {
		t.Errorf("Name = %q, want %q", m.Name, "Out-of-Order Model")
	}
	if m.ContextLength != uint32(4096) {
		t.Errorf("ContextLength = %d, want 4096", m.ContextLength)
	}
	if m.BlockCount != uint32(32) {
		t.Errorf("BlockCount = %d, want 32", m.BlockCount)
	}
}

func TestReadMetadata_InvalidMagic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "notgguf.bin")

	if err := os.WriteFile(path, []byte("this is not a GGUF file at all"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	_, err := ReadMetadata(path)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "invalid magic") {
		t.Errorf("err = %v, want contains %q", err, "invalid magic")
	}
}

func TestReadMetadata_FileNotFound(t *testing.T) {
	_, err := ReadMetadata("/nonexistent/path/model.gguf")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "open file") {
		t.Errorf("err = %v, want contains %q", err, "open file")
	}
}

func TestFileTypeName(t *testing.T) {
	cases := []struct {
		ft   uint32
		want string
	}{
		{15, "Q4_K_M"},
		{17, "Q5_K_M"},
		{7, "Q8_0"},
		{1, "F16"},
		{999, "type_999"},
	}
	for _, c := range cases {
		if got := FileTypeName(c.ft); got != c.want {
			t.Errorf("FileTypeName(%d) = %q, want %q", c.ft, got, c.want)
		}
	}
}

func TestReadMetadata_V2(t *testing.T) {
	// GGUF v2 uses uint32 for tensor and KV counts (instead of uint64 in v3).
	path := writeTestGGUFV2(t, [][2]any{
		{"general.architecture", "llama"},
		{"general.name", "V2 Model"},
		{"general.file_type", uint32(15)},
		{"llama.context_length", uint32(2048)},
		{"llama.block_count", uint32(16)},
	})

	m, err := ReadMetadata(path)
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}

	if m.Architecture != "llama" {
		t.Errorf("Architecture = %q, want %q", m.Architecture, "llama")
	}
	if m.Name != "V2 Model" {
		t.Errorf("Name = %q, want %q", m.Name, "V2 Model")
	}
	if m.FileType != uint32(15) {
		t.Errorf("FileType = %d, want 15", m.FileType)
	}
	if m.ContextLength != uint32(2048) {
		t.Errorf("ContextLength = %d, want 2048", m.ContextLength)
	}
	if m.BlockCount != uint32(16) {
		t.Errorf("BlockCount = %d, want 16", m.BlockCount)
	}
}

func TestReadMetadata_UnsupportedVersion(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad_version.gguf")

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("os.Create: %v", err)
	}

	if err := binary.Write(f, binary.LittleEndian, uint32(0x46554747)); err != nil {
		t.Fatalf("write magic: %v", err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(99)); err != nil {
		t.Fatalf("write version: %v", err)
	}
	f.Close()

	_, err = ReadMetadata(path)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "unsupported GGUF version") {
		t.Errorf("err = %v, want contains %q", err, "unsupported GGUF version")
	}
}

func TestReadMetadata_SkipsUnknownValueTypes(t *testing.T) {
	// Tests skipValue for uint8, int16, float32, uint64, bool, and array types.
	// These are stored under uninteresting keys so ReadMetadata skips them.
	dir := t.TempDir()
	path := filepath.Join(dir, "skip_types.gguf")

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("os.Create: %v", err)
	}

	// Header: magic, v3, 0 tensors
	if err := binary.Write(f, binary.LittleEndian, uint32(0x46554747)); err != nil {
		t.Fatalf("write magic: %v", err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(3)); err != nil {
		t.Fatalf("write version: %v", err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(0)); err != nil {
		t.Fatalf("write tensor count: %v", err)
	}
	// 8 KV pairs: 6 skip types + 2 interesting keys
	if err := binary.Write(f, binary.LittleEndian, uint64(8)); err != nil {
		t.Fatalf("write kv count: %v", err)
	}

	// 1. uint8 (type 0) — 1 byte
	raw := make([]byte, 1)
	raw[0] = 42
	writeRawKV(t, f, "custom.uint8_val", 0, raw)

	// 2. bool (type 7) — 1 byte
	raw = []byte{1}
	writeRawKV(t, f, "custom.bool_val", 7, raw)

	// 3. int16 (type 3) — 2 bytes
	raw = make([]byte, 2)
	binary.LittleEndian.PutUint16(raw, 1234)
	writeRawKV(t, f, "custom.int16_val", 3, raw)

	// 4. float32 (type 6) — 4 bytes
	raw = make([]byte, 4)
	binary.LittleEndian.PutUint32(raw, 0x3F800000) // 1.0
	writeRawKV(t, f, "custom.float32_val", 6, raw)

	// 5. uint64 (type 10) — 8 bytes
	raw = make([]byte, 8)
	binary.LittleEndian.PutUint64(raw, 9999)
	writeRawKV(t, f, "custom.uint64_val", 10, raw)

	// 6. array of uint8 (type 9, element type 0, count 3)
	var arrBuf []byte
	b4 := make([]byte, 4)
	binary.LittleEndian.PutUint32(b4, 0) // element type: uint8
	arrBuf = append(arrBuf, b4...)
	b8 := make([]byte, 8)
	binary.LittleEndian.PutUint64(b8, 3) // count: 3
	arrBuf = append(arrBuf, b8...)
	arrBuf = append(arrBuf, 10, 20, 30) // 3 uint8 values
	writeRawKV(t, f, "custom.array_val", 9, arrBuf)

	// 7-8. Interesting keys to verify parsing continued correctly.
	writeKV(t, f, "general.architecture", "llama")
	writeKV(t, f, "general.name", "Skip Test Model")

	f.Close()

	m, err := ReadMetadata(path)
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}

	if m.Architecture != "llama" {
		t.Errorf("Architecture = %q, want %q", m.Architecture, "llama")
	}
	if m.Name != "Skip Test Model" {
		t.Errorf("Name = %q, want %q", m.Name, "Skip Test Model")
	}
}

func TestReadMetadata_Uint64ContextLength(t *testing.T) {
	// context_length stored as uint64 that fits in uint32 — readTypedValue
	// should downcast it to uint32.
	path := writeTestGGUFOrdered(t, [][2]any{
		{"general.architecture", "llama"},
		{"llama.context_length", uint64(8192)},
		{"llama.block_count", uint64(32)},
	})

	m, err := ReadMetadata(path)
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}

	if m.ContextLength != uint32(8192) {
		t.Errorf("ContextLength = %d, want 8192", m.ContextLength)
	}
	if m.BlockCount != uint32(32) {
		t.Errorf("BlockCount = %d, want 32", m.BlockCount)
	}
}

func TestReadMetadata_TruncatedFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "truncated.gguf")

	// Write only the magic — no version or counts.
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("os.Create: %v", err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(0x46554747)); err != nil {
		t.Fatalf("write magic: %v", err)
	}
	f.Close()

	_, err = ReadMetadata(path)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "reading version") {
		t.Errorf("err = %v, want contains %q", err, "reading version")
	}
}

func TestReadMetadata_SkipsStringValue(t *testing.T) {
	// Tests skipValue for string type (type 8) on an uninteresting key.
	dir := t.TempDir()
	path := filepath.Join(dir, "skip_string.gguf")

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("os.Create: %v", err)
	}

	if err := binary.Write(f, binary.LittleEndian, uint32(0x46554747)); err != nil {
		t.Fatalf("write magic: %v", err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(3)); err != nil {
		t.Fatalf("write version: %v", err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(0)); err != nil {
		t.Fatalf("write tensor count: %v", err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(2)); err != nil {
		t.Fatalf("write kv count: %v", err)
	}

	// Uninteresting string key (exercises skipValue for typeString).
	writeKV(t, f, "custom.description", "a long description value")
	// Interesting key to confirm parsing continued.
	writeKV(t, f, "general.architecture", "gemma3")

	f.Close()

	m, err := ReadMetadata(path)
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}
	if m.Architecture != "gemma3" {
		t.Errorf("Architecture = %q, want %q", m.Architecture, "gemma3")
	}
}
