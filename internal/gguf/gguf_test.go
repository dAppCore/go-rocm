package gguf

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// writeTestGGUFOrdered creates a synthetic GGUF v3 file with KV pairs in the
// exact order specified. Each element is a [2]any{key string, value any}.
func writeTestGGUFOrdered(t *testing.T, kvs [][2]any) string {
	t.Helper()

	dir := t.TempDir()
	path := filepath.Join(dir, "test.gguf")

	f, err := os.Create(path)
	require.NoError(t, err)
	defer f.Close()

	// Magic
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(0x46554747)))
	// Version 3
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(3)))
	// Tensor count (uint64): 0
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint64(0)))
	// KV count (uint64)
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint64(len(kvs))))

	for _, kv := range kvs {
		key := kv[0].(string)
		writeKV(t, f, key, kv[1])
	}

	return path
}

func writeKV(t *testing.T, f *os.File, key string, val any) {
	t.Helper()

	// Key: uint64 length + bytes
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint64(len(key))))
	_, err := f.Write([]byte(key))
	require.NoError(t, err)

	switch v := val.(type) {
	case string:
		// Type: 8 (string)
		require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(8)))
		// String value: uint64 length + bytes
		require.NoError(t, binary.Write(f, binary.LittleEndian, uint64(len(v))))
		_, err := f.Write([]byte(v))
		require.NoError(t, err)
	case uint32:
		// Type: 4 (uint32)
		require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(4)))
		require.NoError(t, binary.Write(f, binary.LittleEndian, v))
	case uint64:
		// Type: 10 (uint64)
		require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(10)))
		require.NoError(t, binary.Write(f, binary.LittleEndian, v))
	default:
		t.Fatalf("writeKV: unsupported value type %T", val)
	}
}

// writeRawKV writes a key with a specific GGUF type and raw byte payload.
// Used to test skipValue for types not used in interesting keys.
func writeRawKV(t *testing.T, f *os.File, key string, valType uint32, rawVal []byte) {
	t.Helper()

	require.NoError(t, binary.Write(f, binary.LittleEndian, uint64(len(key))))
	_, err := f.Write([]byte(key))
	require.NoError(t, err)
	require.NoError(t, binary.Write(f, binary.LittleEndian, valType))
	_, err = f.Write(rawVal)
	require.NoError(t, err)
}

// writeTestGGUFV2 creates a synthetic GGUF v2 file (uint32 tensor/kv counts).
func writeTestGGUFV2(t *testing.T, kvs [][2]any) string {
	t.Helper()

	dir := t.TempDir()
	path := filepath.Join(dir, "test_v2.gguf")

	f, err := os.Create(path)
	require.NoError(t, err)
	defer f.Close()

	// Magic
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(0x46554747)))
	// Version 2
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(2)))
	// Tensor count (uint32 for v2): 0
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(0)))
	// KV count (uint32 for v2)
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(len(kvs))))

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
	require.NoError(t, err)

	assert.Equal(t, "gemma3", m.Architecture)
	assert.Equal(t, "Test Gemma3 1B", m.Name)
	assert.Equal(t, uint32(17), m.FileType)
	assert.Equal(t, "1B", m.SizeLabel)
	assert.Equal(t, uint32(32768), m.ContextLength)
	assert.Equal(t, uint32(26), m.BlockCount)
	assert.Greater(t, m.FileSize, int64(0))
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
	require.NoError(t, err)

	assert.Equal(t, "llama", m.Architecture)
	assert.Equal(t, "Test Llama 8B", m.Name)
	assert.Equal(t, uint32(15), m.FileType)
	assert.Equal(t, "8B", m.SizeLabel)
	assert.Equal(t, uint32(131072), m.ContextLength)
	assert.Equal(t, uint32(32), m.BlockCount)
	assert.Greater(t, m.FileSize, int64(0))
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
	require.NoError(t, err)

	assert.Equal(t, "llama", m.Architecture)
	assert.Equal(t, "Out-of-Order Model", m.Name)
	assert.Equal(t, uint32(4096), m.ContextLength)
	assert.Equal(t, uint32(32), m.BlockCount)
}

func TestReadMetadata_InvalidMagic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "notgguf.bin")

	err := os.WriteFile(path, []byte("this is not a GGUF file at all"), 0644)
	require.NoError(t, err)

	_, err = ReadMetadata(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid magic")
}

func TestReadMetadata_FileNotFound(t *testing.T) {
	_, err := ReadMetadata("/nonexistent/path/model.gguf")
	require.Error(t, err)
	assert.ErrorContains(t, err, "open file")
}

func TestFileTypeName(t *testing.T) {
	assert.Equal(t, "Q4_K_M", FileTypeName(15))
	assert.Equal(t, "Q5_K_M", FileTypeName(17))
	assert.Equal(t, "Q8_0", FileTypeName(7))
	assert.Equal(t, "F16", FileTypeName(1))
	assert.Equal(t, "type_999", FileTypeName(999))
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
	require.NoError(t, err)

	assert.Equal(t, "llama", m.Architecture)
	assert.Equal(t, "V2 Model", m.Name)
	assert.Equal(t, uint32(15), m.FileType)
	assert.Equal(t, uint32(2048), m.ContextLength)
	assert.Equal(t, uint32(16), m.BlockCount)
}

func TestReadMetadata_UnsupportedVersion(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad_version.gguf")

	f, err := os.Create(path)
	require.NoError(t, err)

	require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(0x46554747))) // magic
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(99)))         // invalid version
	f.Close()

	_, err = ReadMetadata(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported GGUF version")
}

func TestReadMetadata_SkipsUnknownValueTypes(t *testing.T) {
	// Tests skipValue for uint8, int16, float32, uint64, bool, and array types.
	// These are stored under uninteresting keys so ReadMetadata skips them.
	dir := t.TempDir()
	path := filepath.Join(dir, "skip_types.gguf")

	f, err := os.Create(path)
	require.NoError(t, err)

	// Header: magic, v3, 0 tensors
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(0x46554747)))
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(3)))
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint64(0)))
	// 8 KV pairs: 6 skip types + 2 interesting keys
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint64(8)))

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
	require.NoError(t, err)

	assert.Equal(t, "llama", m.Architecture)
	assert.Equal(t, "Skip Test Model", m.Name)
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
	require.NoError(t, err)

	assert.Equal(t, uint32(8192), m.ContextLength)
	assert.Equal(t, uint32(32), m.BlockCount)
}

func TestReadMetadata_TruncatedFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "truncated.gguf")

	// Write only the magic — no version or counts.
	f, err := os.Create(path)
	require.NoError(t, err)
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(0x46554747)))
	f.Close()

	_, err = ReadMetadata(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "reading version")
}

func TestReadMetadata_SkipsStringValue(t *testing.T) {
	// Tests skipValue for string type (type 8) on an uninteresting key.
	dir := t.TempDir()
	path := filepath.Join(dir, "skip_string.gguf")

	f, err := os.Create(path)
	require.NoError(t, err)

	require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(0x46554747)))
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(3)))
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint64(0)))
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint64(2)))

	// Uninteresting string key (exercises skipValue for typeString).
	writeKV(t, f, "custom.description", "a long description value")
	// Interesting key to confirm parsing continued.
	writeKV(t, f, "general.architecture", "gemma3")

	f.Close()

	m, err := ReadMetadata(path)
	require.NoError(t, err)
	assert.Equal(t, "gemma3", m.Architecture)
}
