// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestHIPRuntime_LoadModelAllocatesAndCopiesGGUFTensors_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	driver := &fakeHIPDriver{
		available: true,
		device:    nativeDeviceInfo{Name: "gfx1100", MemoryBytes: 16 * memoryGiB, FreeBytes: 12 * memoryGiB, Driver: "fake"},
	}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3", NumLayers: 1, QuantBits: 32},
		DataOffset: dataOffset,
		Tensors: []nativeTensorInfo{{
			Name:     "tok_embeddings.weight",
			Type:     0,
			Offset:   0,
			ByteSize: 16,
		}},
	})

	core.AssertNoError(t, err)
	core.AssertNotNil(t, model)
	core.AssertEqual(t, []uint64{16}, driver.allocations)
	core.AssertEqual(t, []uint64{16}, driver.copies)
	stream, errFn := model.Generate(context.Background(), "hello", inference.DefaultGenerateConfig())
	for range stream {
	}
	core.AssertError(t, errFn())
	core.AssertNoError(t, model.Close())
	core.AssertEqual(t, 1, len(driver.frees))
}

func TestHIPRuntime_LoadModelBadFreeOnCopyFailure_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	driver := &fakeHIPDriver{available: true, copyErr: core.NewError("copy failed")}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3"},
		DataOffset: dataOffset,
		Tensors:    []nativeTensorInfo{{Name: "bad.weight", ByteSize: 16}},
	})

	core.AssertError(t, err)
	core.AssertNil(t, model)
	core.AssertEqual(t, 1, len(driver.frees))
}

func TestHIPRuntime_LoadModelUglyEmptyTensorMap_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	driver := &fakeHIPDriver{available: true}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3"},
		DataOffset: dataOffset,
	})

	core.AssertNoError(t, err)
	core.AssertNotNil(t, model)
	core.AssertEqual(t, 0, len(driver.allocations))
	core.AssertNoError(t, model.Close())
}

type fakeHIPDriver struct {
	available   bool
	device      nativeDeviceInfo
	nextPointer nativeDevicePointer
	allocations []uint64
	copies      []uint64
	frees       []nativeDevicePointer
	copyErr     error
}

func (driver *fakeHIPDriver) Available() bool { return driver.available }
func (driver *fakeHIPDriver) DeviceInfo() nativeDeviceInfo {
	return driver.device
}
func (driver *fakeHIPDriver) Malloc(size uint64) (nativeDevicePointer, error) {
	driver.allocations = append(driver.allocations, size)
	driver.nextPointer++
	return driver.nextPointer, nil
}
func (driver *fakeHIPDriver) Free(pointer nativeDevicePointer) error {
	driver.frees = append(driver.frees, pointer)
	return nil
}
func (driver *fakeHIPDriver) CopyHostToDevice(_ nativeDevicePointer, data []byte) error {
	driver.copies = append(driver.copies, uint64(len(data)))
	return driver.copyErr
}

func nativeHIPTensorGGUF(t *testing.T) (string, int64) {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "weights.gguf")
	result := core.WriteFile(path, []byte("0123456789abcdef"), 0o644)
	core.RequireTrue(t, result.OK)
	return path, 0
}
