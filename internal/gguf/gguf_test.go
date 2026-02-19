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
	default:
		t.Fatalf("writeKV: unsupported value type %T", val)
	}
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
}

func TestFileTypeName(t *testing.T) {
	assert.Equal(t, "Q4_K_M", FileTypeName(15))
	assert.Equal(t, "Q5_K_M", FileTypeName(17))
	assert.Equal(t, "Q8_0", FileTypeName(7))
	assert.Equal(t, "F16", FileTypeName(1))
	assert.Equal(t, "type_999", FileTypeName(999))
}
