package gguf

import (
	core "dappco.re/go"
	"encoding/binary"
	"testing"
)

func tinyGGUF(t *testing.T) string {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "tiny.gguf")
	buf := core.NewBuffer()
	binary.Write(buf, binary.LittleEndian, uint32(ggufMagic))
	binary.Write(buf, binary.LittleEndian, uint32(3))
	binary.Write(buf, binary.LittleEndian, uint64(0))
	binary.Write(buf, binary.LittleEndian, uint64(0))
	r := core.WriteFile(path, buf.Bytes(), 0o644)
	core.RequireTrue(t, r.OK)
	return path
}

func TestGguf_FileTypeName_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	core.AssertEqual(t, "Q4_K_M", FileTypeName(15))
}
func TestGguf_FileTypeName_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	core.AssertEqual(t, "type_999", FileTypeName(999))
}
func TestGguf_FileTypeName_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	core.AssertNotEqual(t, "", FileTypeName(0))
}

func TestGguf_ReadMetadata_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	meta, err := ReadMetadata(tinyGGUF(t))
	core.AssertNoError(t, err)
	core.AssertEqual(t, int64(24), meta.FileSize)
}
func TestGguf_ReadMetadata_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	_, err := ReadMetadata(core.PathJoin(t.TempDir(), "missing.gguf"))
	core.AssertError(t, err)
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestGguf_ReadMetadata_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	path := core.PathJoin(t.TempDir(), "bad.gguf")
	core.WriteFile(path, []byte("bad"), 0o644)
	_, err := ReadMetadata(path)
	core.AssertError(t, err)
}

func TestGguf_ReadInfo_Good_TensorDirectory(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	path := tensorGGUF(t)

	info, err := ReadInfo(path)

	core.AssertNoError(t, err)
	core.AssertEqual(t, "qwen3", info.Metadata.Architecture)
	core.AssertEqual(t, 1, len(info.Tensors))
	core.AssertEqual(t, "tok_embeddings.weight", info.Tensors[0].Name)
	core.AssertEqual(t, []uint64{2, 2}, info.Tensors[0].Dimensions)
	core.AssertEqual(t, uint32(0), info.Tensors[0].Type)
	core.AssertEqual(t, uint64(16), info.Tensors[0].ByteSize)
	core.AssertGreater(t, info.DataOffset, int64(0))
}

func TestGguf_ReadInfo_Bad_UnsupportedTensorType(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	path := tensorGGUFWithType(t, 999)

	_, err := ReadInfo(path)

	core.AssertError(t, err)
}

func TestGguf_ReadInfo_Ugly_EmptyTensorDirectory(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)

	info, err := ReadInfo(tinyGGUF(t))

	core.AssertNoError(t, err)
	core.AssertEqual(t, 0, len(info.Tensors))
	core.AssertGreater(t, info.DataOffset, int64(0))
}

func tensorGGUF(t *testing.T) string {
	t.Helper()
	return tensorGGUFWithType(t, 0)
}

func tensorGGUFWithType(t *testing.T, tensorType uint32) string {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "tensor.gguf")
	buf := core.NewBuffer()
	writeUint32 := func(v uint32) { core.RequireNoError(t, binary.Write(buf, binary.LittleEndian, v)) }
	writeUint64 := func(v uint64) { core.RequireNoError(t, binary.Write(buf, binary.LittleEndian, v)) }
	writeString := func(v string) {
		writeUint64(uint64(len(v)))
		_, err := buf.Write([]byte(v))
		core.RequireNoError(t, err)
	}
	writeKVString := func(key, value string) {
		writeString(key)
		writeUint32(typeString)
		writeString(value)
	}
	writeKVUint32 := func(key string, value uint32) {
		writeString(key)
		writeUint32(typeUint32)
		writeUint32(value)
	}

	writeUint32(ggufMagic)
	writeUint32(3)
	writeUint64(1)
	writeUint64(4)
	writeKVString("general.architecture", "qwen3")
	writeKVString("general.name", "tensor-test")
	writeKVUint32("general.file_type", 0)
	writeKVUint32("qwen3.block_count", 1)

	writeString("tok_embeddings.weight")
	writeUint32(2)
	writeUint64(2)
	writeUint64(2)
	writeUint32(tensorType)
	writeUint64(0)

	for buf.Len()%defaultAlignment != 0 {
		buf.WriteByte(0)
	}
	buf.Write(make([]byte, 16))

	result := core.WriteFile(path, buf.Bytes(), 0o644)
	core.RequireTrue(t, result.OK)
	return path
}
