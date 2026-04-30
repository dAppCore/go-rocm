// AX-10 CLI driver for go-rocm. It exercises the public model discovery and
// platform availability surface without requiring ROCm hardware or llama-server.
//
//	task -d tests/cli/rocm test
//	go run ./tests/cli/rocm
package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"dappco.re/go/rocm"
)

type ggufKV struct {
	key   string
	value any
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	if err := verifyDiscoverModels(); err != nil {
		return fmt.Errorf("discover models: %w", err)
	}
	if err := verifyPlatformSurface(); err != nil {
		return fmt.Errorf("platform surface: %w", err)
	}
	return nil
}

func verifyDiscoverModels() error {
	dir, err := os.MkdirTemp("", "go-rocm-ax10-models-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)

	gemmaPath, err := writeGGUF(dir, "gemma3-1b-q5km.gguf", []ggufKV{
		{key: "general.architecture", value: "gemma3"},
		{key: "general.name", value: "AX-10 Gemma3"},
		{key: "general.file_type", value: uint32(17)},
		{key: "general.size_label", value: "1B"},
		{key: "gemma3.context_length", value: uint32(32768)},
		{key: "gemma3.block_count", value: uint32(26)},
	})
	if err != nil {
		return err
	}

	llamaPath, err := writeGGUF(dir, "llama-8b-q4km.gguf", []ggufKV{
		{key: "general.architecture", value: "llama"},
		{key: "general.name", value: "AX-10 Llama"},
		{key: "general.file_type", value: uint32(15)},
		{key: "general.size_label", value: "8B"},
		{key: "llama.context_length", value: uint32(131072)},
		{key: "llama.block_count", value: uint32(32)},
	})
	if err != nil {
		return err
	}

	if err := os.WriteFile(filepath.Join(dir, "corrupt.gguf"), []byte("not gguf"), 0644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "README.txt"), []byte("not a model"), 0644); err != nil {
		return err
	}

	models, err := rocm.DiscoverModels(dir)
	if err != nil {
		return err
	}
	if len(models) != 2 {
		return fmt.Errorf("len(models) = %d, want 2", len(models))
	}

	if err := expectModel(models[0], rocm.ModelInfo{
		Path:         gemmaPath,
		Architecture: "gemma3",
		Name:         "AX-10 Gemma3",
		Quantisation: "Q5_K_M",
		Parameters:   "1B",
		ContextLen:   32768,
	}); err != nil {
		return fmt.Errorf("gemma model: %w", err)
	}

	if err := expectModel(models[1], rocm.ModelInfo{
		Path:         llamaPath,
		Architecture: "llama",
		Name:         "AX-10 Llama",
		Quantisation: "Q4_K_M",
		Parameters:   "8B",
		ContextLen:   131072,
	}); err != nil {
		return fmt.Errorf("llama model: %w", err)
	}

	emptyModels, err := rocm.DiscoverModels(filepath.Join(dir, "missing"))
	if err != nil {
		return err
	}
	if len(emptyModels) != 0 {
		return fmt.Errorf("missing directory returned %d models, want 0", len(emptyModels))
	}

	_, err = rocm.DiscoverModels(filepath.Join(dir, "bad["))
	if err == nil {
		return errors.New("bad glob pattern returned nil error")
	}
	if !strings.Contains(err.Error(), "glob gguf files") {
		return fmt.Errorf("bad glob error = %v", err)
	}

	return nil
}

func expectModel(got rocm.ModelInfo, want rocm.ModelInfo) error {
	if got.Path != want.Path {
		return fmt.Errorf("Path = %q, want %q", got.Path, want.Path)
	}
	if got.Architecture != want.Architecture {
		return fmt.Errorf("Architecture = %q, want %q", got.Architecture, want.Architecture)
	}
	if got.Name != want.Name {
		return fmt.Errorf("Name = %q, want %q", got.Name, want.Name)
	}
	if got.Quantisation != want.Quantisation {
		return fmt.Errorf("Quantisation = %q, want %q", got.Quantisation, want.Quantisation)
	}
	if got.Parameters != want.Parameters {
		return fmt.Errorf("Parameters = %q, want %q", got.Parameters, want.Parameters)
	}
	if got.ContextLen != want.ContextLen {
		return fmt.Errorf("ContextLen = %d, want %d", got.ContextLen, want.ContextLen)
	}
	if got.FileSize <= 0 {
		return fmt.Errorf("FileSize = %d, want > 0", got.FileSize)
	}
	return nil
}

func verifyPlatformSurface() error {
	compiledForROCm := runtime.GOOS == "linux" && runtime.GOARCH == "amd64"
	if got := rocm.ROCmAvailable(); got != compiledForROCm {
		return fmt.Errorf("ROCmAvailable() = %v, want %v", got, compiledForROCm)
	}

	info, err := rocm.GetVRAMInfo()
	if !compiledForROCm {
		if err == nil {
			return errors.New("GetVRAMInfo() on non-ROCm platform returned nil error")
		}
		if info != (rocm.VRAMInfo{}) {
			return fmt.Errorf("GetVRAMInfo() on non-ROCm platform = %+v, want zero value", info)
		}
		return nil
	}

	if err != nil {
		return nil
	}
	if info.Total == 0 {
		return fmt.Errorf("VRAM Total = %d, want > 0", info.Total)
	}
	if info.Used > info.Total {
		return fmt.Errorf("VRAM Used = %d, want <= Total %d", info.Used, info.Total)
	}
	if info.Free > info.Total {
		return fmt.Errorf("VRAM Free = %d, want <= Total %d", info.Free, info.Total)
	}
	return nil
}

func writeGGUF(dir, filename string, kvs []ggufKV) (string, error) {
	path := filepath.Join(dir, filename)

	file, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer file.Close()

	if err := binary.Write(file, binary.LittleEndian, uint32(0x46554747)); err != nil {
		return "", err
	}
	if err := binary.Write(file, binary.LittleEndian, uint32(3)); err != nil {
		return "", err
	}
	if err := binary.Write(file, binary.LittleEndian, uint64(0)); err != nil {
		return "", err
	}
	if err := binary.Write(file, binary.LittleEndian, uint64(len(kvs))); err != nil {
		return "", err
	}

	for _, kv := range kvs {
		if err := writeKV(file, kv); err != nil {
			return "", err
		}
	}

	return path, nil
}

func writeKV(file *os.File, kv ggufKV) error {
	if err := binary.Write(file, binary.LittleEndian, uint64(len(kv.key))); err != nil {
		return err
	}
	if _, err := file.Write([]byte(kv.key)); err != nil {
		return err
	}

	switch value := kv.value.(type) {
	case string:
		if err := binary.Write(file, binary.LittleEndian, uint32(8)); err != nil {
			return err
		}
		if err := binary.Write(file, binary.LittleEndian, uint64(len(value))); err != nil {
			return err
		}
		_, err := file.Write([]byte(value))
		return err
	case uint32:
		if err := binary.Write(file, binary.LittleEndian, uint32(4)); err != nil {
			return err
		}
		return binary.Write(file, binary.LittleEndian, value)
	default:
		return fmt.Errorf("unsupported GGUF value type %T for %q", kv.value, kv.key)
	}
}
