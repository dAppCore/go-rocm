// Package gguf provides a GGUF binary metadata parser for reading model headers.
//
// GGUF (GGML Universal File) is the file format used by llama.cpp and other
// GGML-based inference engines. This package reads the metadata key-value pairs
// from the file header without loading tensor data, enabling fast model discovery.
//
// Supports GGUF v2 (uint32 counts) and v3 (uint64 counts).
package gguf

import (
	"bufio"
	"encoding/binary"
	"io"
	"math"

	core "dappco.re/go"
)

// ggufMagic is the GGUF file magic number: "GGUF" in little-endian.
const ggufMagic = 0x46554747

// GGUF value type codes.
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

// Metadata holds the interesting fields extracted from a GGUF file header.
type Metadata struct {
	Architecture  string // "gemma3", "llama", "qwen2"
	Name          string // human-readable model name
	SizeLabel     string // "1B", "8B", etc.
	ContextLength uint32 // native context window
	BlockCount    uint32 // transformer layers
	FileType      uint32 // GGML quantisation file type
	FileSize      int64  // file size on disk in bytes
}

// fileTypeNames maps GGML quantisation file type numbers to human-readable names.
var fileTypeNames = map[uint32]string{
	0:  "F32",
	1:  "F16",
	2:  "Q4_0",
	3:  "Q4_1",
	7:  "Q8_0",
	8:  "Q5_0",
	9:  "Q5_1",
	10: "Q2_K",
	11: "Q3_K_S",
	12: "Q3_K_M",
	13: "Q3_K_L",
	14: "Q4_K_S",
	15: "Q4_K_M",
	16: "Q5_K_S",
	17: "Q5_K_M",
	18: "Q6_K",
}

//	name := FileTypeName(15) // "Q4_K_M"
//
// FileTypeName returns a human-readable name for a GGML quantisation file
// type. Unknown types return "type_N" where N is the numeric value.
func FileTypeName(ft uint32) string {
	if name, ok := fileTypeNames[ft]; ok {
		return name
	}
	return core.Sprintf("type_%d", ft)
}

//	metadata, err := ReadMetadata("/models/gemma3-4b.gguf")
//
// ReadMetadata reads the GGUF header from the file at path and returns the
// extracted metadata. Only metadata KV pairs are read; tensor data is not
// loaded.
func ReadMetadata(path string) (
	Metadata,
	error,
) {
	fileResult := core.Open(path)
	if !fileResult.OK {
		return Metadata{}, core.E("gguf.ReadMetadata", "open file", fileResult.Value.(error))
	}
	file := fileResult.Value.(*core.OSFile)
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		return Metadata{}, core.E("gguf.ReadMetadata", "stat file", err)
	}

	reader := bufio.NewReader(file)

	// Read and validate magic number.
	var magic uint32
	if err := binary.Read(reader, binary.LittleEndian, &magic); err != nil {
		return Metadata{}, core.E("gguf.ReadMetadata", "reading magic", err)
	}
	if magic != ggufMagic {
		return Metadata{}, core.E("gguf.ReadMetadata", core.Sprintf("invalid magic: 0x%08X (expected 0x%08X)", magic, ggufMagic), nil)
	}

	// Read version.
	var version uint32
	if err := binary.Read(reader, binary.LittleEndian, &version); err != nil {
		return Metadata{}, core.E("gguf.ReadMetadata", "reading version", err)
	}
	if version < 2 || version > 3 {
		return Metadata{}, core.E("gguf.ReadMetadata", core.Sprintf("unsupported GGUF version: %d", version), nil)
	}

	// Read tensor count and KV count. v3 uses uint64, v2 uses uint32.
	var tensorCount, kvCount uint64
	if version == 3 {
		if err := binary.Read(reader, binary.LittleEndian, &tensorCount); err != nil {
			return Metadata{}, core.E("gguf.ReadMetadata", "reading tensor count", err)
		}
		if err := binary.Read(reader, binary.LittleEndian, &kvCount); err != nil {
			return Metadata{}, core.E("gguf.ReadMetadata", "reading kv count", err)
		}
	} else {
		var tensorCount32, kvCount32 uint32
		if err := binary.Read(reader, binary.LittleEndian, &tensorCount32); err != nil {
			return Metadata{}, core.E("gguf.ReadMetadata", "reading tensor count", err)
		}
		if err := binary.Read(reader, binary.LittleEndian, &kvCount32); err != nil {
			return Metadata{}, core.E("gguf.ReadMetadata", "reading kv count", err)
		}
		tensorCount = uint64(tensorCount32)
		kvCount = uint64(kvCount32)
	}
	_ = tensorCount // we only read metadata KVs

	// Read all KV pairs. We store interesting keys and skip the rest.
	// Architecture-specific keys (e.g. llama.context_length) may appear before
	// the general.architecture key, so we collect all candidates and resolve after.
	var meta Metadata
	meta.FileSize = fileInfo.Size()

	// candidateContextLength and candidateBlockCount store values keyed by
	// their full key name (e.g. "llama.context_length") so we can match them
	// against the architecture once it is known.
	candidateContextLength := make(map[string]uint32)
	candidateBlockCount := make(map[string]uint32)

	for i := uint64(0); i < kvCount; i++ {
		key, err := readString(reader)
		if err != nil {
			return Metadata{}, core.E("gguf.ReadMetadata", core.Sprintf("reading key %d", i), err)
		}

		var valType uint32
		if err := binary.Read(reader, binary.LittleEndian, &valType); err != nil {
			return Metadata{}, core.E("gguf.ReadMetadata", core.Sprintf("reading value type for key %q", key), err)
		}

		// Check whether this is an interesting key before reading the value.
		switch {
		case key == "general.architecture":
			value, err := readTypedValue(reader, valType)
			if err != nil {
				return Metadata{}, core.E("gguf.ReadMetadata", core.Sprintf("reading value for key %q", key), err)
			}
			if s, ok := value.(string); ok {
				meta.Architecture = s
			}

		case key == "general.name":
			value, err := readTypedValue(reader, valType)
			if err != nil {
				return Metadata{}, core.E("gguf.ReadMetadata", core.Sprintf("reading value for key %q", key), err)
			}
			if s, ok := value.(string); ok {
				meta.Name = s
			}

		case key == "general.file_type":
			value, err := readTypedValue(reader, valType)
			if err != nil {
				return Metadata{}, core.E("gguf.ReadMetadata", core.Sprintf("reading value for key %q", key), err)
			}
			if u, ok := value.(uint32); ok {
				meta.FileType = u
			}

		case key == "general.size_label":
			value, err := readTypedValue(reader, valType)
			if err != nil {
				return Metadata{}, core.E("gguf.ReadMetadata", core.Sprintf("reading value for key %q", key), err)
			}
			if s, ok := value.(string); ok {
				meta.SizeLabel = s
			}

		case core.HasSuffix(key, ".context_length"):
			value, err := readTypedValue(reader, valType)
			if err != nil {
				return Metadata{}, core.E("gguf.ReadMetadata", core.Sprintf("reading value for key %q", key), err)
			}
			if u, ok := value.(uint32); ok {
				candidateContextLength[key] = u
			}

		case core.HasSuffix(key, ".block_count"):
			value, err := readTypedValue(reader, valType)
			if err != nil {
				return Metadata{}, core.E("gguf.ReadMetadata", core.Sprintf("reading value for key %q", key), err)
			}
			if u, ok := value.(uint32); ok {
				candidateBlockCount[key] = u
			}

		default:
			// Skip uninteresting value.
			if err := skipValue(reader, valType); err != nil {
				return Metadata{}, core.E("gguf.ReadMetadata", core.Sprintf("skipping value for key %q", key), err)
			}
		}
	}

	// Resolve architecture-specific keys.
	if meta.Architecture != "" {
		prefix := meta.Architecture + "."
		if v, ok := candidateContextLength[prefix+"context_length"]; ok {
			meta.ContextLength = v
		}
		if v, ok := candidateBlockCount[prefix+"block_count"]; ok {
			meta.BlockCount = v
		}
	}

	return meta, nil
}

// maxStringLength is a sanity limit for GGUF string values. No metadata string
// should ever approach 1 MiB; this prevents memory exhaustion from malformed files.
const maxStringLength = 1 << 20

type ggufFailure interface {
	Error() string
}

// readString reads a GGUF string: uint64 length followed by that many bytes.
func readString(r io.Reader) (
	string,
	error,
) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > maxStringLength {
		return "", core.E("gguf.readString", core.Sprintf("string length %d exceeds maximum %d", length, maxStringLength), nil)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// readTypedValue reads a value of the given GGUF type and returns it as a Go
// value. String, uint32, and uint64 types return typed values (uint64 is
// downcast to uint32 when it fits). All others are read and discarded.
func readTypedValue(r io.Reader, valType uint32) (
	any,
	error,
) {
	switch valType {
	case typeString:
		return readString(r)
	case typeUint32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case typeUint64:
		var v uint64
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		if v <= math.MaxUint32 {
			return uint32(v), nil
		}
		return v, nil
	default:
		// Read and discard the value, returning nil.
		err := skipValue(r, valType)
		return nil, err
	}
}

// skipValue reads and discards a GGUF value of the given type from r.
func skipValue(r io.Reader, valType uint32) ggufFailure {
	switch valType {
	case typeUint8, typeInt8, typeBool:
		_, err := discardBytes(r, 1)
		return err
	case typeUint16, typeInt16:
		_, err := discardBytes(r, 2)
		return err
	case typeUint32, typeInt32, typeFloat32:
		_, err := discardBytes(r, 4)
		return err
	case typeUint64, typeInt64, typeFloat64:
		_, err := discardBytes(r, 8)
		return err
	case typeString:
		var length uint64
		if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
			return err
		}
		if length > maxStringLength {
			return core.E("gguf.skipValue", core.Sprintf("string length %d exceeds maximum %d", length, maxStringLength), nil)
		}
		_, err := discardBytes(r, int64(length))
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
		return core.E("gguf.skipValue", core.Sprintf("unknown GGUF value type: %d", valType), nil)
	}
}

// discardBytes reads and discards exactly n bytes from r.
func discardBytes(r io.Reader, n int64) (
	int64,
	error,
) {
	return io.CopyN(io.Discard, r, n)
}
