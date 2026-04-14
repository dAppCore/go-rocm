package rocm

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// writeDiscoverTestGGUF creates a minimal GGUF v3 file in dir with the given
// filename and metadata KV pairs. Returns the full path to the created file.
func writeDiscoverTestGGUF(t *testing.T, dir, filename string, kvs [][2]any) string {
	t.Helper()

	path := filepath.Join(dir, filename)

	f, err := os.Create(path)
	require.NoError(t, err)
	defer f.Close()

	// Magic: "GGUF" in little-endian
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(0x46554747)))
	// Version 3
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint32(3)))
	// Tensor count (uint64): 0
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint64(0)))
	// KV count (uint64)
	require.NoError(t, binary.Write(f, binary.LittleEndian, uint64(len(kvs))))

	for _, kv := range kvs {
		key := kv[0].(string)
		writeDiscoverKV(t, f, key, kv[1])
	}

	return path
}

func writeDiscoverKV(t *testing.T, f *os.File, key string, val any) {
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
	require.NoError(t, os.WriteFile(filepath.Join(dir, "README.txt"), []byte("not a model"), 0644))

	models, err := DiscoverModels(dir)
	require.NoError(t, err)
	require.Len(t, models, 2)

	// Sort order from Glob is lexicographic, so gemma3 comes first.
	gemma := models[0]
	assert.Equal(t, filepath.Join(dir, "gemma3-4b-q4km.gguf"), gemma.Path)
	assert.Equal(t, "gemma3", gemma.Architecture)
	assert.Equal(t, "Gemma 3 4B Instruct", gemma.Name)
	assert.Equal(t, "Q4_K_M", gemma.Quantisation)
	assert.Equal(t, "4B", gemma.Parameters)
	assert.Equal(t, uint32(32768), gemma.ContextLen)
	assert.Greater(t, gemma.FileSize, int64(0))

	llama := models[1]
	assert.Equal(t, filepath.Join(dir, "llama-3.1-8b-q4km.gguf"), llama.Path)
	assert.Equal(t, "llama", llama.Architecture)
	assert.Equal(t, "Llama 3.1 8B Instruct", llama.Name)
	assert.Equal(t, "Q4_K_M", llama.Quantisation)
	assert.Equal(t, "8B", llama.Parameters)
	assert.Equal(t, uint32(131072), llama.ContextLen)
	assert.Greater(t, llama.FileSize, int64(0))
}

func TestDiscoverModels_RelativeDirReturnsAbsolutePaths(t *testing.T) {
	parent := t.TempDir()
	dir := filepath.Join(parent, "models")
	require.NoError(t, os.Mkdir(dir, 0755))

	path := writeDiscoverTestGGUF(t, dir, "model.gguf", [][2]any{
		{"general.architecture", "llama"},
		{"general.name", "Relative Model"},
		{"general.file_type", uint32(15)},
	})

	wd, err := os.Getwd()
	require.NoError(t, err)
	require.NoError(t, os.Chdir(parent))
	t.Cleanup(func() {
		require.NoError(t, os.Chdir(wd))
	})

	models, err := DiscoverModels("models")
	require.NoError(t, err)
	require.Len(t, models, 1)
	assert.Equal(t, path, models[0].Path)
}

func TestDiscoverModels_EmptyDir(t *testing.T) {
	dir := t.TempDir()

	models, err := DiscoverModels(dir)
	require.NoError(t, err)
	assert.Empty(t, models)
}

func TestDiscoverModels_NotFound(t *testing.T) {
	// filepath.Glob returns nil, nil for a pattern matching no files,
	// even when the directory does not exist.
	models, err := DiscoverModels("/nonexistent/dir")
	require.NoError(t, err)
	assert.Empty(t, models)
}

func TestDiscoverModels_BadPattern(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "bad[")

	_, err := DiscoverModels(dir)
	require.Error(t, err)
	assert.ErrorContains(t, err, "glob gguf files")
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
	require.NoError(t, os.WriteFile(filepath.Join(dir, "corrupt.gguf"), []byte("not gguf data"), 0644))

	models, err := DiscoverModels(dir)
	require.NoError(t, err)
	// Only the valid model should be returned; corrupt one is silently skipped.
	require.Len(t, models, 1)
	assert.Equal(t, "Valid Model", models[0].Name)
}
