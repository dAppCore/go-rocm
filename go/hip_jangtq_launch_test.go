// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
)

func TestHIPJANGTQProjectionLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipJANGTQProjectionRequest{
		Input:         []float32{2, 4},
		PackedWeights: []byte{0x8d},
		Descriptor:    rocmJANGTQDescriptor{WeightFormat: "mxtq", Bits: 2, GroupSize: 2},
		Rows:          2,
		Cols:          2,
		Scale:         0.5,
		Bias:          []float32{0, 1},
	}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.RequireNoError(t, err)
	payload, err := launch.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipJANGTQLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipJANGTQLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipJANGTQLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(buffers.Input.Pointer()), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(buffers.Packed.Pointer()), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(buffers.Bias.Pointer()), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(payload[32:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[40:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[44:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[48:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[52:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[56:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[60:]))
	core.AssertEqual(t, uint32(1), binary.LittleEndian.Uint32(payload[64:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[68:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[72:]))
	core.AssertEqual(t, hipJANGTQLaunchFlagBias, binary.LittleEndian.Uint32(payload[80:]))
}

func TestHIPJANGTQProjectionLaunch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipJANGTQProjectionRequest{
		Input:         []float32{2, 4},
		PackedWeights: []byte{0x8d},
		Descriptor:    rocmJANGTQDescriptor{WeightFormat: "mxtq", Bits: 2, GroupSize: 2},
		Rows:          2,
		Cols:          2,
		Scale:         0.5,
		Bias:          []float32{0, 1},
	}
	want, err := rocmReferenceJANGTQProjection(req.Input, req.PackedWeights, req.Descriptor, req.Rows, req.Cols, req.Scale, req.Bias)
	core.RequireNoError(t, err)

	got, err := hipRunJANGTQProjectionKernel(context.Background(), driver, req)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameJANGTQ, driver.launches[0].Name)
	core.AssertEqual(t, hipJANGTQLaunchArgsBytes, len(driver.launches[0].Args))
	assertFloat32SlicesNear(t, want, got, 0)
}

func TestHIPJANGTQProjectionLaunch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	_, err := hipRunJANGTQProjectionKernel(context.Background(), driver, hipJANGTQProjectionRequest{
		Input:         []float32{1},
		PackedWeights: []byte{0},
		Descriptor:    rocmJANGTQDescriptor{WeightFormat: "mxtq", Bits: 3, GroupSize: 64},
		Rows:          1,
		Cols:          1,
		Scale:         1,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported bit layout")

	_, err = hipRunJANGTQProjectionKernel(context.Background(), driver, hipJANGTQProjectionRequest{
		Input:         []float32{1, 2},
		PackedWeights: nil,
		Descriptor:    rocmJANGTQDescriptor{WeightFormat: "mxtq", Bits: 2, GroupSize: 2},
		Rows:          2,
		Cols:          2,
		Scale:         1,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "packed weights need")

	_, err = hipRunJANGTQProjectionKernel(context.Background(), driver, hipJANGTQProjectionRequest{
		Input:         []float32{float32(math.Inf(1))},
		PackedWeights: []byte{0},
		Descriptor:    rocmJANGTQDescriptor{WeightFormat: "mxtq", Bits: 2, GroupSize: 1},
		Rows:          1,
		Cols:          1,
		Scale:         1,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = (hipJANGTQLaunchArgs{
		InputPointer:  1,
		PackedPointer: 2,
		OutputPointer: 3,
		InputCount:    2,
		Rows:          2,
		Cols:          2,
		Bits:          2,
		GroupSize:     2,
		InputBytes:    4,
		PackedBytes:   1,
		OutputBytes:   8,
		Scale:         1,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input byte count")
}

func TestHIPJANGTQProjectionLaunchBufferValidation_Bad(t *testing.T) {
	req := hipJANGTQProjectionRequest{
		Input:         []float32{2, 4},
		PackedWeights: []byte{0x8d},
		Descriptor:    rocmJANGTQDescriptor{WeightFormat: "mxtq", Bits: 2, GroupSize: 2},
		Rows:          2,
		Cols:          2,
		Scale:         0.5,
		Bias:          []float32{0, 1},
	}
	_, err := req.launchArgs(nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "JANGTQ device buffers are required")

	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.Output.count++
	_, err = req.launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "JANGTQ device buffer shape mismatch")

	_, err = (hipJANGTQLaunchArgs{
		InputPointer:  1,
		PackedPointer: 2,
		OutputPointer: 3,
		InputCount:    2,
		Rows:          2,
		Cols:          2,
		Bits:          2,
		GroupSize:     2,
		InputBytes:    8,
		PackedBytes:   1,
		BiasBytes:     8,
		OutputBytes:   8,
		Scale:         1,
		Flags:         hipJANGTQLaunchFlagBias,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "bias pointer is nil")

	_, err = (hipJANGTQLaunchArgs{
		InputPointer:  1,
		PackedPointer: 2,
		BiasPointer:   4,
		OutputPointer: 3,
		InputCount:    2,
		Rows:          2,
		Cols:          2,
		Bits:          2,
		GroupSize:     2,
		InputBytes:    8,
		PackedBytes:   1,
		BiasBytes:     8,
		OutputBytes:   8,
		Scale:         1,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "bias metadata supplied without bias flag")
}

func TestHIPJANGTQProjectionReadOutputValidation_Bad(t *testing.T) {
	_, err := (*hipJANGTQDeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "JANGTQ output buffer is required")

	driver := &fakeHIPDriver{available: true}
	req := hipJANGTQProjectionRequest{
		Input:         []float32{2, 4},
		PackedWeights: []byte{0x8d},
		Descriptor:    rocmJANGTQDescriptor{WeightFormat: "mxtq", Bits: 2, GroupSize: 2},
		Rows:          2,
		Cols:          2,
		Scale:         0.5,
	}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.Output.sizeBytes++
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "JANGTQ output byte count mismatch")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	payload, err := hipFloat32Payload([]float32{0, float32(math.NaN())})
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(buffers.Output.Pointer(), payload))
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	driver.copyErr = core.NewError("copy failed")

	_, err = buffers.ReadOutput()

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy JANGTQ output")
}
