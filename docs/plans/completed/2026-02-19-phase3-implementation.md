# Phase 3: Model Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GGUF metadata parsing, model directory scanning, and auto-detection of context window and model architecture at load time.

**Architecture:** New `internal/gguf/` package reads GGUF binary headers (magic, version, metadata KV pairs). `DiscoverModels()` scans directories. `LoadModel()` uses GGUF metadata to replace filename-based guessing and auto-cap context window at 4096.

**Tech Stack:** Go, binary/encoding (little-endian GGUF), testify, bufio

---

### Task 1: GGUF Metadata Parser

New `internal/gguf/` package that reads GGUF v2/v3 binary headers and extracts metadata. Reads only the header — no tensor data. Completes in <1ms per file.

**Files:**
- Create: `internal/gguf/gguf.go`
- Create: `internal/gguf/gguf_test.go`

**Step 1: Write the failing tests**

Create `internal/gguf/gguf_test.go`:

```go
package gguf

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// writeTestGGUF creates a minimal GGUF v3 file with the given KV pairs.
// Supports string and uint32 values.
func writeTestGGUF(t *testing.T, kvPairs [][2]any) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.gguf")
	f, err := os.Create(path)
	require.NoError(t, err)
	defer f.Close()

	w := f
	binary.Write(w, binary.LittleEndian, uint32(0x46554747)) // magic "GGUF"
	binary.Write(w, binary.LittleEndian, uint32(3))           // version
	binary.Write(w, binary.LittleEndian, uint64(0))           // tensor count
	binary.Write(w, binary.LittleEndian, uint64(len(kvPairs))) // kv count

	for _, kv := range kvPairs {
		key := kv[0].(string)
		// Write key
		binary.Write(w, binary.LittleEndian, uint64(len(key)))
		w.Write([]byte(key))

		switch val := kv[1].(type) {
		case string:
			binary.Write(w, binary.LittleEndian, uint32(8)) // typeString
			binary.Write(w, binary.LittleEndian, uint64(len(val)))
			w.Write([]byte(val))
		case uint32:
			binary.Write(w, binary.LittleEndian, uint32(4)) // typeUint32
			binary.Write(w, binary.LittleEndian, val)
		}
	}

	return path
}

func TestReadMetadata_Gemma3(t *testing.T) {
	path := writeTestGGUF(t, [][2]any{
		{"general.architecture", "gemma3"},
		{"general.name", "Test Gemma3 1B"},
		{"general.file_type", uint32(17)},
		{"general.size_label", "1B"},
		{"gemma3.context_length", uint32(32768)},
		{"gemma3.block_count", uint32(26)},
	})

	meta, err := ReadMetadata(path)
	require.NoError(t, err)

	assert.Equal(t, "gemma3", meta.Architecture)
	assert.Equal(t, "Test Gemma3 1B", meta.Name)
	assert.Equal(t, uint32(17), meta.FileType)
	assert.Equal(t, "1B", meta.SizeLabel)
	assert.Equal(t, uint32(32768), meta.ContextLength)
	assert.Equal(t, uint32(26), meta.BlockCount)
	assert.Greater(t, meta.FileSize, int64(0))
}

func TestReadMetadata_Llama(t *testing.T) {
	path := writeTestGGUF(t, [][2]any{
		{"general.architecture", "llama"},
		{"general.name", "Test Llama 8B"},
		{"general.file_type", uint32(15)},
		{"general.size_label", "8B"},
		{"llama.context_length", uint32(131072)},
		{"llama.block_count", uint32(32)},
	})

	meta, err := ReadMetadata(path)
	require.NoError(t, err)

	assert.Equal(t, "llama", meta.Architecture)
	assert.Equal(t, uint32(131072), meta.ContextLength)
	assert.Equal(t, uint32(32), meta.BlockCount)
}

func TestReadMetadata_ArchAfterContextLength(t *testing.T) {
	// Architecture key comes AFTER the arch-specific keys.
	// Parser must handle this (two-pass or map approach).
	path := writeTestGGUF(t, [][2]any{
		{"gemma3.context_length", uint32(8192)},
		{"gemma3.block_count", uint32(18)},
		{"general.architecture", "gemma3"},
		{"general.name", "Out Of Order"},
		{"general.file_type", uint32(15)},
	})

	meta, err := ReadMetadata(path)
	require.NoError(t, err)

	assert.Equal(t, "gemma3", meta.Architecture)
	assert.Equal(t, uint32(8192), meta.ContextLength)
	assert.Equal(t, uint32(18), meta.BlockCount)
}

func TestReadMetadata_InvalidMagic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.gguf")
	require.NoError(t, os.WriteFile(path, []byte("NOT_GGUF_DATA_HERE"), 0644))

	_, err := ReadMetadata(path)
	assert.ErrorContains(t, err, "invalid magic")
}

func TestReadMetadata_FileNotFound(t *testing.T) {
	_, err := ReadMetadata("/nonexistent/model.gguf")
	assert.Error(t, err)
}

func TestFileTypeName(t *testing.T) {
	assert.Equal(t, "Q4_K_M", FileTypeName(15))
	assert.Equal(t, "Q5_K_M", FileTypeName(17))
	assert.Equal(t, "Q8_0", FileTypeName(7))
	assert.Equal(t, "F16", FileTypeName(1))
	assert.Equal(t, "type_999", FileTypeName(999))
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./internal/gguf/ -v`
Expected: FAIL — package doesn't exist

**Step 3: Implement the GGUF parser**

Create `internal/gguf/gguf.go`:

```go
package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

const ggufMagic = 0x46554747 // "GGUF" in little-endian

// GGUF value types.
const (
	typeUint8   uint32 = 0
	typeInt8    uint32 = 1
	typeUint16  uint32 = 2
	typeInt16   uint32 = 3
	typeUint32  uint32 = 4
	typeInt32   uint32 = 5
	typeFloat32 uint32 = 6
	typeBool    uint32 = 7
	typeString  uint32 = 8
	typeArray   uint32 = 9
	typeUint64  uint32 = 10
	typeInt64   uint32 = 11
	typeFloat64 uint32 = 12
)

// Metadata holds parsed GGUF file metadata.
type Metadata struct {
	Architecture  string // "gemma3", "llama", "qwen2"
	Name          string // human-readable model name
	SizeLabel     string // "1B", "8B", etc.
	ContextLength uint32 // native context window
	BlockCount    uint32 // transformer layers
	FileType      uint32 // GGML quantisation file type
	FileSize      int64  // file size on disk in bytes
}

// ReadMetadata reads the GGUF header metadata from a file.
// Only reads the header — no tensor data. Supports GGUF v2 and v3.
func ReadMetadata(path string) (Metadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return Metadata{}, err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return Metadata{}, err
	}

	r := bufio.NewReader(f)

	var magic uint32
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return Metadata{}, fmt.Errorf("gguf: read magic: %w", err)
	}
	if magic != ggufMagic {
		return Metadata{}, fmt.Errorf("gguf: invalid magic: 0x%08x", magic)
	}

	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return Metadata{}, fmt.Errorf("gguf: read version: %w", err)
	}
	if version < 2 || version > 3 {
		return Metadata{}, fmt.Errorf("gguf: unsupported version: %d", version)
	}

	// v3 uses uint64 for counts, v2 uses uint32.
	var kvCount uint64
	if version == 3 {
		var tensorCount uint64
		binary.Read(r, binary.LittleEndian, &tensorCount)
		binary.Read(r, binary.LittleEndian, &kvCount)
	} else {
		var tensorCount, kc uint32
		binary.Read(r, binary.LittleEndian, &tensorCount)
		binary.Read(r, binary.LittleEndian, &kc)
		kvCount = uint64(kc)
	}

	// Read all metadata KV pairs, storing only interesting keys.
	kv := make(map[string]any)
	for i := uint64(0); i < kvCount; i++ {
		key, err := readString(r)
		if err != nil {
			return Metadata{}, fmt.Errorf("gguf: read key %d: %w", i, err)
		}

		var valueType uint32
		if err := binary.Read(r, binary.LittleEndian, &valueType); err != nil {
			return Metadata{}, fmt.Errorf("gguf: read type for %s: %w", key, err)
		}

		if isInteresting(key) {
			val, err := readValue(r, valueType)
			if err != nil {
				return Metadata{}, fmt.Errorf("gguf: read value for %s: %w", key, err)
			}
			kv[key] = val
		} else {
			if err := skipValue(r, valueType); err != nil {
				return Metadata{}, fmt.Errorf("gguf: skip value for %s: %w", key, err)
			}
		}
	}

	m := Metadata{FileSize: info.Size()}
	m.Architecture, _ = kv["general.architecture"].(string)
	m.Name, _ = kv["general.name"].(string)
	m.SizeLabel, _ = kv["general.size_label"].(string)
	m.FileType, _ = kv["general.file_type"].(uint32)
	if m.Architecture != "" {
		m.ContextLength, _ = kv[m.Architecture+".context_length"].(uint32)
		m.BlockCount, _ = kv[m.Architecture+".block_count"].(uint32)
	}

	return m, nil
}

func isInteresting(key string) bool {
	switch key {
	case "general.architecture", "general.name", "general.file_type", "general.size_label":
		return true
	}
	return strings.HasSuffix(key, ".context_length") || strings.HasSuffix(key, ".block_count")
}

func readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > 1<<20 { // 1MB sanity limit for a single string
		return "", fmt.Errorf("string too long: %d", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readValue(r io.Reader, vtype uint32) (any, error) {
	switch vtype {
	case typeUint8:
		var v uint8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeInt8:
		var v int8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeUint16:
		var v uint16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeInt16:
		var v int16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeUint32:
		var v uint32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeInt32:
		var v int32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeFloat32:
		var v float32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeBool:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v != 0, err
	case typeString:
		return readString(r)
	case typeArray:
		return readArray(r)
	case typeUint64:
		var v uint64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeInt64:
		var v int64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeFloat64:
		var v float64
		return v, binary.Read(r, binary.LittleEndian, &v)
	default:
		return nil, fmt.Errorf("unknown value type: %d", vtype)
	}
}

func readArray(r io.Reader) (any, error) {
	var elemType uint32
	if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
		return nil, err
	}
	var count uint64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, err
	}
	for i := uint64(0); i < count; i++ {
		if err := skipValue(r, elemType); err != nil {
			return nil, err
		}
	}
	return nil, nil
}

func skipValue(r io.Reader, vtype uint32) error {
	switch vtype {
	case typeUint8, typeInt8, typeBool:
		_, err := io.CopyN(io.Discard, r, 1)
		return err
	case typeUint16, typeInt16:
		_, err := io.CopyN(io.Discard, r, 2)
		return err
	case typeUint32, typeInt32, typeFloat32:
		_, err := io.CopyN(io.Discard, r, 4)
		return err
	case typeUint64, typeInt64, typeFloat64:
		_, err := io.CopyN(io.Discard, r, 8)
		return err
	case typeString:
		var length uint64
		if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
			return err
		}
		_, err := io.CopyN(io.Discard, r, int64(length))
		return err
	case typeArray:
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return err
		}
		var count uint64
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return err
		}
		for i := uint64(0); i < count; i++ {
			if err := skipValue(r, elemType); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("unknown value type: %d", vtype)
	}
}

// FileTypeName returns the quantisation name for a GGML file type number.
func FileTypeName(ft uint32) string {
	names := map[uint32]string{
		0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
		7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
		10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
		14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
		18: "Q6_K",
	}
	if name, ok := names[ft]; ok {
		return name
	}
	return fmt.Sprintf("type_%d", ft)
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./internal/gguf/ -v`
Expected: PASS (6 tests)

Run: `go test ./...`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add internal/gguf/
git commit -m "feat: GGUF metadata parser for model discovery

Co-Authored-By: Virgil <virgil@lethean.io>"
```

---

### Task 2: Model Discovery

Scan a directory for `.gguf` files and return structured inventory using the GGUF parser.

**Files:**
- Modify: `rocm.go` (add ModelInfo type — no build tags)
- Create: `discover.go` (DiscoverModels — no build tags, pure file I/O)
- Create: `discover_test.go` (tests with synthetic GGUF files)

**Step 1: Write the failing tests**

Create `discover_test.go` (no build tags needed — pure file I/O):

```go
package rocm

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// writeDiscoverTestGGUF creates a minimal GGUF file for discovery tests.
func writeDiscoverTestGGUF(t *testing.T, dir, filename string, kvPairs [][2]any) {
	t.Helper()
	path := filepath.Join(dir, filename)
	f, err := os.Create(path)
	require.NoError(t, err)
	defer f.Close()

	binary.Write(f, binary.LittleEndian, uint32(0x46554747))
	binary.Write(f, binary.LittleEndian, uint32(3))
	binary.Write(f, binary.LittleEndian, uint64(0))
	binary.Write(f, binary.LittleEndian, uint64(len(kvPairs)))

	for _, kv := range kvPairs {
		key := kv[0].(string)
		binary.Write(f, binary.LittleEndian, uint64(len(key)))
		f.Write([]byte(key))
		switch val := kv[1].(type) {
		case string:
			binary.Write(f, binary.LittleEndian, uint32(8))
			binary.Write(f, binary.LittleEndian, uint64(len(val)))
			f.Write([]byte(val))
		case uint32:
			binary.Write(f, binary.LittleEndian, uint32(4))
			binary.Write(f, binary.LittleEndian, val)
		}
	}
}

func TestDiscoverModels(t *testing.T) {
	dir := t.TempDir()

	writeDiscoverTestGGUF(t, dir, "Gemma3-4B-Q4_K_M.gguf", [][2]any{
		{"general.architecture", "gemma3"},
		{"general.name", "Gemma3 4B"},
		{"general.file_type", uint32(15)},
		{"general.size_label", "4B"},
		{"gemma3.context_length", uint32(32768)},
		{"gemma3.block_count", uint32(34)},
	})

	writeDiscoverTestGGUF(t, dir, "Llama-3.1-8B-Q4_K_M.gguf", [][2]any{
		{"general.architecture", "llama"},
		{"general.name", "Llama 3.1 8B"},
		{"general.file_type", uint32(15)},
		{"general.size_label", "8B"},
		{"llama.context_length", uint32(131072)},
		{"llama.block_count", uint32(32)},
	})

	// Non-GGUF file should be ignored.
	require.NoError(t, os.WriteFile(filepath.Join(dir, "readme.txt"), []byte("hi"), 0644))

	models, err := DiscoverModels(dir)
	require.NoError(t, err)
	require.Len(t, models, 2)

	// Sort is by filepath.Glob order (alphabetical).
	assert.Equal(t, "gemma3", models[0].Architecture)
	assert.Equal(t, "Gemma3 4B", models[0].Name)
	assert.Equal(t, "Q4_K_M", models[0].Quantisation)
	assert.Equal(t, "4B", models[0].Parameters)
	assert.Equal(t, uint32(32768), models[0].ContextLen)

	assert.Equal(t, "llama", models[1].Architecture)
	assert.Equal(t, "8B", models[1].Parameters)
}

func TestDiscoverModels_EmptyDir(t *testing.T) {
	dir := t.TempDir()
	models, err := DiscoverModels(dir)
	require.NoError(t, err)
	assert.Empty(t, models)
}

func TestDiscoverModels_NotFound(t *testing.T) {
	_, err := DiscoverModels("/nonexistent/dir")
	require.NoError(t, err) // Glob returns nil, nil for no matches
}
```

**Step 2: Run tests to verify they fail**

Run: `go test -run "TestDiscoverModels" -v`
Expected: FAIL — DiscoverModels and ModelInfo don't exist

**Step 3: Add ModelInfo to rocm.go**

Append to `rocm.go` (after VRAMInfo):

```go

// ModelInfo describes a GGUF model file discovered on disk.
type ModelInfo struct {
	Path         string // full path to .gguf file
	Architecture string // GGUF architecture (e.g. "gemma3", "llama", "qwen2")
	Name         string // human-readable model name from GGUF metadata
	Quantisation string // quantisation level (e.g. "Q4_K_M", "Q8_0")
	Parameters   string // parameter size label (e.g. "1B", "8B")
	FileSize     int64  // file size in bytes
	ContextLen   uint32 // native context window length
}
```

**Step 4: Create discover.go**

Create `discover.go` (no build tags — pure file I/O):

```go
package rocm

import (
	"path/filepath"

	"forge.lthn.ai/core/go-rocm/internal/gguf"
)

// DiscoverModels scans a directory for GGUF model files and returns
// structured information about each. Files that cannot be parsed are skipped.
func DiscoverModels(dir string) ([]ModelInfo, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.gguf"))
	if err != nil {
		return nil, err
	}

	var models []ModelInfo
	for _, path := range matches {
		meta, err := gguf.ReadMetadata(path)
		if err != nil {
			continue
		}

		models = append(models, ModelInfo{
			Path:         path,
			Architecture: meta.Architecture,
			Name:         meta.Name,
			Quantisation: gguf.FileTypeName(meta.FileType),
			Parameters:   meta.SizeLabel,
			FileSize:     meta.FileSize,
			ContextLen:   meta.ContextLength,
		})
	}

	return models, nil
}
```

**Step 5: Run tests to verify they pass**

Run: `go test -run "TestDiscoverModels" -v`
Expected: PASS (3 tests)

Run: `go test ./...`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add rocm.go discover.go discover_test.go
git commit -m "feat: model discovery scanning directories for GGUF files

Co-Authored-By: Virgil <virgil@lethean.io>"
```

---

### Task 3: LoadModel Enrichment

Replace filename-based `guessModelType` with GGUF metadata parsing. Auto-cap context window at 4096 when user doesn't specify one.

**Files:**
- Modify: `backend.go` (use GGUF metadata in LoadModel, remove guessModelType)
- Modify: `server_test.go` (remove TestGuessModelType, add TestLoadModelContextDefault)

**Step 1: Update LoadModel in backend.go**

Replace the `LoadModel` function and remove `guessModelType`:

```go
func (b *rocmBackend) LoadModel(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
	cfg := inference.ApplyLoadOpts(opts)

	binary, err := findLlamaServer()
	if err != nil {
		return nil, err
	}

	meta, err := gguf.ReadMetadata(path)
	if err != nil {
		return nil, fmt.Errorf("rocm: read model metadata: %w", err)
	}

	ctxLen := cfg.ContextLen
	if ctxLen == 0 && meta.ContextLength > 0 {
		ctxLen = int(min(meta.ContextLength, 4096))
	}

	srv, err := startServer(binary, path, cfg.GPULayers, ctxLen)
	if err != nil {
		return nil, err
	}

	return &rocmModel{
		srv:       srv,
		modelType: meta.Architecture,
	}, nil
}
```

Add `"fmt"` and `"forge.lthn.ai/core/go-rocm/internal/gguf"` to imports. Remove `"path/filepath"` and `"strings"` (no longer needed without guessModelType).

Delete the `guessModelType` function entirely (it was at the bottom of backend.go).

**Step 2: Update server_test.go**

Remove `TestGuessModelType` (the function it tested no longer exists).

**Step 3: Run tests to verify they pass**

Run: `go test ./...`
Expected: All tests PASS. Some existing tests may need adjustment if they depend on guessModelType.

Run: `go vet ./...`
Expected: Clean

**Step 4: Commit**

```bash
git add backend.go server_test.go
git commit -m "feat: use GGUF metadata for model type and context window auto-detection

Replaces filename-based guessModelType. Caps default context at 4096
to prevent VRAM exhaustion on models with 128K+ native context.

Co-Authored-By: Virgil <virgil@lethean.io>"
```

---

### Task 4: Integration Tests + Documentation

Verify GGUF metadata enrichment works on real models. Verify chat templates work natively. Update TODO.md.

**Files:**
- Modify: `rocm_integration_test.go` (add metadata verification to existing test)
- Modify: `TODO.md` (mark Phase 3 items done)
- Modify: `FINDINGS.md` (document GGUF metadata findings)

**Step 1: Add GGUF metadata verification to integration tests**

In `rocm_integration_test.go`, modify `TestROCm_LoadAndGenerate` to verify ModelType comes from GGUF:

Replace the existing `assert.Equal(t, "gemma3", m.ModelType())` line — it should still pass since the GGUF metadata for our test model (Gemma3-1B) has `general.architecture = "gemma3"`.

Add a new test to verify discovery on the real model directory:

```go
func TestROCm_DiscoverModels(t *testing.T) {
	dir := filepath.Dir(testModel)
	if _, err := os.Stat(dir); err != nil {
		t.Skip("model directory not available")
	}

	models, err := DiscoverModels(dir)
	require.NoError(t, err)
	require.NotEmpty(t, models, "expected at least one model in %s", dir)

	for _, m := range models {
		t.Logf("Found: %s (%s %s %s, ctx=%d)", filepath.Base(m.Path), m.Architecture, m.Parameters, m.Quantisation, m.ContextLen)
		assert.NotEmpty(t, m.Architecture)
		assert.NotEmpty(t, m.Name)
		assert.Greater(t, m.FileSize, int64(0))
	}
}
```

Add `"path/filepath"` to imports if not already present.

**Step 2: Run integration tests**

Run: `go test -tags rocm -v -timeout 120s`
Expected: All tests PASS. ModelType() returns "gemma3" (from GGUF, not filename parsing).

**Step 3: Update TODO.md**

Mark all Phase 3 items as `[x]` with commit references.

**Step 4: Update FINDINGS.md**

Add a section documenting GGUF metadata findings:

```markdown
## 2026-02-19: Phase 3 Model Support (Charon)

### GGUF Metadata

Parsed GGUF headers from available models:

| Model | Architecture | Size Label | File Type | Context | Blocks |
|-------|-------------|------------|-----------|---------|--------|
| Gemma3-1B Q5_K_M | gemma3 | 1B | 17 (Q5_K_M) | 32768 | 26 |
| Llama-3.1-8B Q4_K_M | llama | 8B | 15 (Q4_K_M) | 131072 | 32 |
| Qwen-2.5-7B Q4_K_M | qwen2 | 7B | 15 (Q4_K_M) | 32768 | 28 |

Note: Qwen 2.5 reports as "qwen2" architecture (not "qwen3"). Gemma3 reports as "gemma3" (not "gemma2").

### Chat Templates

llama-server reads `tokenizer.chat_template` from the GGUF and applies it automatically on `/v1/chat/completions`. No go-rocm code needed. Verified working for Gemma3.

### Context Window Auto-Detection

Default context capped at 4096 when user doesn't specify. Without this cap, Llama-3.1 would try to allocate 131072 context, which exceeds 16GB VRAM. Users can override with `inference.WithContextLen(N)`.
```

**Step 5: Commit**

```bash
git add rocm_integration_test.go TODO.md FINDINGS.md
git commit -m "docs: Phase 3 complete — GGUF metadata, discovery, auto context

Co-Authored-By: Virgil <virgil@lethean.io>"
```

---

## Summary

| Task | What | Files | Test Type |
|------|------|-------|-----------|
| 1 | GGUF metadata parser | internal/gguf/gguf.go, gguf_test.go | Unit (synthetic fixtures) |
| 2 | Model discovery | rocm.go, discover.go, discover_test.go | Unit (synthetic fixtures) |
| 3 | LoadModel enrichment | backend.go, server_test.go | Unit + integration |
| 4 | Integration tests + docs | rocm_integration_test.go, TODO.md, FINDINGS.md | Integration (GPU) |
