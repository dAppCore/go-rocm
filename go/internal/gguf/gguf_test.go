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
