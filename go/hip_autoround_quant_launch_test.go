// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
)

func TestHIPAutoRoundQuantizeLaunchArgs_Good(t *testing.T) {
	profile, ok := ProductionAutoRoundQuantizationProfileByName("mxfp4")
	core.AssertEqual(t, true, ok)
	plan := DefaultProductionAutoRoundCalibrationPlan(profile)
	req := hipAutoRoundQuantizeRequest{
		Weights: autoroundLaunchWeights(2 * 128),
		Plan:    plan,
		Rows:    2,
		Cols:    128,
	}
	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.RequireNoError(t, err)
	payload, err := launch.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipAutoRoundQuantizeLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipAutoRoundQuantizeLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipAutoRoundQuantizeLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(buffers.Weights.Pointer()), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(buffers.PackedOutput.Pointer()), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(buffers.ScaleOutput.Pointer()), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[32:]))
	core.AssertEqual(t, uint32(128), binary.LittleEndian.Uint32(payload[36:]))
	core.AssertEqual(t, hipAutoRoundFormatMXFP4, binary.LittleEndian.Uint32(payload[40:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(payload[44:]))
	core.AssertEqual(t, uint32(128), binary.LittleEndian.Uint32(payload[48:]))
	core.AssertEqual(t, uint32(1), binary.LittleEndian.Uint32(payload[52:]))
	core.AssertEqual(t, uint32(1024), binary.LittleEndian.Uint32(payload[56:]))
	core.AssertEqual(t, uint32(128), binary.LittleEndian.Uint32(payload[60:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(payload[64:]))
	core.AssertEqual(t, uint32(plan.NSamples), binary.LittleEndian.Uint32(payload[68:]))
	core.AssertEqual(t, uint32(plan.SeqLen), binary.LittleEndian.Uint32(payload[72:]))
	core.AssertEqual(t, uint32(plan.Iters), binary.LittleEndian.Uint32(payload[76:]))
}

func TestHIPAutoRoundQuantizeLaunchDispatch_Good(t *testing.T) {
	profile, ok := ProductionAutoRoundQuantizationProfileByName("w8a16-mxfp8-g64")
	core.AssertEqual(t, true, ok)
	driver := &fakeHIPDriver{available: true}
	req := hipAutoRoundQuantizeRequest{
		Weights: autoroundLaunchWeights(64),
		Plan:    DefaultProductionAutoRoundCalibrationPlan(profile),
		Rows:    1,
		Cols:    64,
	}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	core.RequireNoError(t, err)
	packet, err := launch.Binary()
	core.RequireNoError(t, err)
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameAutoRoundQuantize, packet, req.Rows*buffers.GroupsPerRow)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(driver, config))

	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameAutoRoundQuantize, driver.launches[0].Name)
	core.AssertEqual(t, hipAutoRoundQuantizeLaunchArgsBytes, len(driver.launches[0].Args))
}

func TestHIPAutoRoundQuantizeLaunch_Bad(t *testing.T) {
	profile, ok := ProductionAutoRoundQuantizationProfileByName("mxfp4")
	core.AssertEqual(t, true, ok)
	plan := DefaultProductionAutoRoundCalibrationPlan(profile)
	driver := &fakeHIPDriver{available: true}

	_, err := (hipAutoRoundQuantizeRequest{
		Weights: autoroundLaunchWeights(127),
		Plan:    plan,
		Rows:    1,
		Cols:    127,
	}).deviceBuffers(driver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "group size")

	badFormat := plan
	badFormat.FloatFormat = "bad"
	_, err = (hipAutoRoundQuantizeRequest{
		Weights: autoroundLaunchWeights(128),
		Plan:    badFormat,
		Rows:    1,
		Cols:    128,
	}).deviceBuffers(driver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported AutoRound format")

	linked := plan
	linked.HIPKernel = hipKernelStatusLinked
	_, err = (hipAutoRoundQuantizeRequest{
		Weights: autoroundLaunchWeights(128),
		Plan:    linked,
		Rows:    1,
		Cols:    128,
	}).deviceBuffers(driver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "not_linked")

	_, err = (hipAutoRoundQuantizeRequest{
		Weights: []float32{float32(math.NaN())},
		Plan:    plan,
		Rows:    1,
		Cols:    1,
	}).deviceBuffers(driver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = (hipAutoRoundQuantizeLaunchArgs{
		WeightPointer: 1,
		PackedPointer: 2,
		ScalePointer:  3,
		Rows:          1,
		Cols:          128,
		FormatCode:    hipAutoRoundFormatMXFP4,
		Bits:          4,
		GroupSize:     128,
		GroupsPerRow:  1,
		WeightBytes:   512,
		PackedBytes:   1,
		ScaleBytes:    4,
		NSamples:      512,
		SeqLen:        2048,
		Iters:         200,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "packed output byte count")
}

func TestHIPAutoRoundQuantizeLaunchBufferValidation_Bad(t *testing.T) {
	profile, ok := ProductionAutoRoundQuantizationProfileByName("int2")
	core.AssertEqual(t, true, ok)
	req := hipAutoRoundQuantizeRequest{
		Weights: autoroundLaunchWeights(128),
		Plan:    DefaultProductionAutoRoundCalibrationPlan(profile),
		Rows:    1,
		Cols:    128,
	}
	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	buffers.ScaleOutput.count++
	_, err = req.launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	_, err = (hipAutoRoundQuantizeLaunchArgs{
		WeightPointer: 1,
		PackedPointer: 2,
		ScalePointer:  3,
		Rows:          1,
		Cols:          128,
		FormatCode:    hipAutoRoundFormatMXFP4,
		Bits:          8,
		GroupSize:     128,
		GroupsPerRow:  1,
		WeightBytes:   512,
		PackedBytes:   128,
		ScaleBytes:    4,
		NSamples:      512,
		SeqLen:        2048,
		Iters:         200,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "format code")
}

var hipAutoRoundQuantizePacketSink []byte

func BenchmarkHIPAutoRoundQuantizeLaunchArgsBinaryInto_MXFP4(b *testing.B) {
	args := hipAutoRoundQuantizeLaunchArgs{
		WeightPointer: 1,
		PackedPointer: 2,
		ScalePointer:  3,
		Rows:          128,
		Cols:          128,
		FormatCode:    hipAutoRoundFormatMXFP4,
		Bits:          4,
		GroupSize:     128,
		GroupsPerRow:  1,
		WeightBytes:   128 * 128 * 4,
		PackedBytes:   128 * 128 / 2,
		ScaleBytes:    128 * 4,
		NSamples:      512,
		SeqLen:        2048,
		Iters:         200,
	}
	packet := make([]byte, hipAutoRoundQuantizeLaunchArgsBytes)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		payload, err := args.BinaryInto(packet)
		if err != nil {
			b.Fatal(err)
		}
		hipAutoRoundQuantizePacketSink = payload
	}
}

func autoroundLaunchWeights(count int) []float32 {
	weights := make([]float32, count)
	for index := range weights {
		weights[index] = float32(index%17) * 0.125
	}
	return weights
}
