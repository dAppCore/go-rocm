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

var (
	hipAdamWUpdateLaunchBenchPayload []byte
	hipAdamWUpdateLaunchBenchErr     error
)

func TestHIPAdamWUpdateLaunchArgs_Good(t *testing.T) {
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "a", Values: []float32{1, 2}},
		{Name: "b", Values: []float32{3}},
	}, NativeAdamWConfig{LearningRate: 0.01, WeightDecay: 0.1, WeightDecaySet: true})
	core.RequireNoError(t, err)
	req := hipAdamWUpdateRequest{
		State:     state,
		Gradients: [][]float32{{0.5, -0.25}, {0.125}},
	}
	driver := &fakeHIPDriver{available: true}

	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	core.RequireNoError(t, err)
	payload, err := launch.Binary()
	core.RequireNoError(t, err)

	core.AssertEqual(t, hipAdamWUpdateLaunchArgsBytes, len(payload))
	core.AssertEqual(t, hipAdamWUpdateLaunchArgsVersion, binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(hipAdamWUpdateLaunchArgsBytes), binary.LittleEndian.Uint32(payload[4:]))
	core.AssertEqual(t, uint64(buffers.Parameters.Pointer()), binary.LittleEndian.Uint64(payload[8:]))
	core.AssertEqual(t, uint64(buffers.MomentM.Pointer()), binary.LittleEndian.Uint64(payload[16:]))
	core.AssertEqual(t, uint64(buffers.MomentV.Pointer()), binary.LittleEndian.Uint64(payload[24:]))
	core.AssertEqual(t, uint64(buffers.Gradients.Pointer()), binary.LittleEndian.Uint64(payload[32:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(payload[40:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(payload[44:]))
	core.AssertEqual(t, uint32(1), binary.LittleEndian.Uint32(payload[48:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(payload[52:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(payload[56:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(payload[60:]))
	core.AssertEqual(t, state.Config.LearningRate, math.Float64frombits(binary.LittleEndian.Uint64(payload[64:])))
	core.AssertEqual(t, state.Config.Beta1, math.Float64frombits(binary.LittleEndian.Uint64(payload[72:])))
	core.AssertEqual(t, state.Config.Beta2, math.Float64frombits(binary.LittleEndian.Uint64(payload[80:])))
	core.AssertEqual(t, state.Config.Eps, math.Float64frombits(binary.LittleEndian.Uint64(payload[88:])))
	core.AssertEqual(t, state.Config.WeightDecay, math.Float64frombits(binary.LittleEndian.Uint64(payload[96:])))

	expected, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "a", Values: []float32{1, 2}},
		{Name: "b", Values: []float32{3}},
	}, NativeAdamWConfig{LearningRate: 0.01, WeightDecay: 0.1, WeightDecaySet: true})
	core.RequireNoError(t, err)
	core.RequireNoError(t, expected.StepInPlace(req.Gradients))
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameAdamWUpdate, payload, launch.ParamCount)
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipLaunchKernel(driver, config))
	gotParams := hipAdamWReadDeviceFloat32(t, driver, buffers.Parameters.Pointer(), int(buffers.Parameters.SizeBytes()))
	gotMomentM := hipAdamWReadDeviceFloat32(t, driver, buffers.MomentM.Pointer(), int(buffers.MomentM.SizeBytes()))
	gotMomentV := hipAdamWReadDeviceFloat32(t, driver, buffers.MomentV.Pointer(), int(buffers.MomentV.SizeBytes()))
	for index, want := range expected.Parameters() {
		assertAdamWFloat32Near(t, want, gotParams[index], 0.0001)
	}
	for index, want := range expected.FirstMoment() {
		assertAdamWFloat32Near(t, want, gotMomentM[index], 0.0001)
	}
	for index, want := range expected.SecondMoment() {
		assertAdamWFloat32Near(t, want, gotMomentV[index], 0.00001)
	}

	driver = &fakeHIPDriver{available: true}
	err = hipRunAdamWUpdateKernel(context.Background(), driver, req)
	core.RequireNoError(t, err)
	core.AssertEqual(t, expected.Step, state.Step)
	for index, want := range expected.Parameters() {
		assertAdamWFloat32Near(t, want, state.Parameters()[index], 0.0001)
	}
	for index, want := range expected.FirstMoment() {
		assertAdamWFloat32Near(t, want, state.FirstMoment()[index], 0.0001)
	}
	for index, want := range expected.SecondMoment() {
		assertAdamWFloat32Near(t, want, state.SecondMoment()[index], 0.00001)
	}
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameAdamWUpdate, driver.launches[0].Name)
	core.AssertEqual(t, hipAdamWUpdateLaunchArgsBytes, len(driver.launches[0].Args))
}

func TestHIPAdamWUpdateLaunchArgs_Bad(t *testing.T) {
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "w", Values: []float32{1, 2}},
	}, NativeAdamWConfig{})
	core.RequireNoError(t, err)

	core.AssertError(t, (hipAdamWUpdateRequest{}).validate())
	core.AssertError(t, (hipAdamWUpdateRequest{State: state}).validate())
	core.AssertError(t, (hipAdamWUpdateRequest{State: state, Gradients: [][]float32{{1}}}).validate())
	core.AssertError(t, (hipAdamWUpdateRequest{State: state, Gradients: [][]float32{{float32(math.NaN()), 1}}}).validate())

	_, err = (hipAdamWUpdateLaunchArgs{}).Binary()
	core.AssertError(t, err)

	_, err = (hipAdamWUpdateLaunchArgs{
		ParameterPointer: 1,
		MomentMPointer:   2,
		MomentVPointer:   3,
		GradientPointer:  4,
		ParamCount:       2,
		TensorCount:      1,
		Step:             1,
		ParameterBytes:   8,
		MomentBytes:      8,
		GradientBytes:    4,
		LearningRate:     0.01,
		Beta1:            0.9,
		Beta2:            0.999,
		Eps:              1e-8,
		WeightDecay:      0,
	}).Binary()
	core.AssertError(t, err)

	_, err = (hipAdamWUpdateLaunchArgs{
		ParameterPointer: 1,
		MomentMPointer:   2,
		MomentVPointer:   3,
		GradientPointer:  4,
		ParamCount:       2,
		TensorCount:      1,
		Step:             1,
		ParameterBytes:   8,
		MomentBytes:      8,
		GradientBytes:    8,
		LearningRate:     math.Inf(1),
		Beta1:            0.9,
		Beta2:            0.999,
		Eps:              1e-8,
		WeightDecay:      0,
	}).Binary()
	core.AssertError(t, err)
}

func TestHIPAdamWUpdateLaunchArgs_DeviceBufferShapeMismatch_Bad(t *testing.T) {
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "w", Values: []float32{1, 2}},
	}, NativeAdamWConfig{})
	core.RequireNoError(t, err)
	req := hipAdamWUpdateRequest{State: state, Gradients: [][]float32{{0.1, 0.2}}}
	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()

	buffers.ParamCount = 1
	_, err = req.launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")
}

func TestHIPAdamWUpdateRunKernel_Bad(t *testing.T) {
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "w", Values: []float32{1, 2}},
	}, NativeAdamWConfig{})
	core.RequireNoError(t, err)
	req := hipAdamWUpdateRequest{State: state, Gradients: [][]float32{{0.1, 0.2}}}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err = hipRunAdamWUpdateKernel(ctx, &fakeHIPDriver{available: true}, req)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "context canceled")

	err = hipRunAdamWUpdateKernel(context.Background(), &fakeHIPDriver{available: true, launchErr: core.NewError("launch failed")}, req)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "launch failed")
}

func BenchmarkHIPAdamWUpdateLaunchArgsBinaryInto_Hot(b *testing.B) {
	args := hipAdamWUpdateLaunchArgs{
		ParameterPointer: 0x1000,
		MomentMPointer:   0x2000,
		MomentVPointer:   0x3000,
		GradientPointer:  0x4000,
		ParamCount:       65536,
		TensorCount:      2,
		Step:             128,
		ParameterBytes:   65536 * 4,
		MomentBytes:      65536 * 4,
		GradientBytes:    65536 * 4,
		LearningRate:     1e-4,
		Beta1:            0.9,
		Beta2:            0.999,
		Eps:              1e-8,
		WeightDecay:      0.01,
	}
	var scratch [hipAdamWUpdateLaunchArgsBytes]byte
	payload, err := args.BinaryInto(scratch[:])
	if err != nil {
		b.Fatal(err)
	}
	if len(payload) != hipAdamWUpdateLaunchArgsBytes {
		b.Fatalf("packet len = %d, want %d", len(payload), hipAdamWUpdateLaunchArgsBytes)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hipAdamWUpdateLaunchBenchPayload, hipAdamWUpdateLaunchBenchErr = args.BinaryInto(scratch[:])
		if hipAdamWUpdateLaunchBenchErr != nil {
			b.Fatal(hipAdamWUpdateLaunchBenchErr)
		}
	}
}

func BenchmarkHIPAdamWUpdateFakeLaunch_LoRA4096Rank8(b *testing.B) {
	state := nativeAdamWBenchState(b)
	req := hipAdamWUpdateRequest{
		State: state,
		Gradients: [][]float32{
			make([]float32, 8*4096),
			make([]float32, 4096*8),
		},
	}
	driver := &fakeHIPDriver{available: true, skipLaunchRecording: true, skipDriverRecording: true}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		b.Fatal(err)
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		b.Fatal(err)
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		b.Fatal(err)
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameAdamWUpdate, launchBytes, launch.ParamCount)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hipAdamWUpdateLaunchBenchErr = hipLaunchKernel(driver, config)
		if hipAdamWUpdateLaunchBenchErr != nil {
			b.Fatal(hipAdamWUpdateLaunchBenchErr)
		}
	}
}

func BenchmarkHIPAdamWUpdateRunKernel_LoRA4096Rank8(b *testing.B) {
	state := nativeAdamWBenchState(b)
	req := hipAdamWUpdateRequest{
		State: state,
		Gradients: [][]float32{
			make([]float32, 8*4096),
			make([]float32, 4096*8),
		},
	}
	driver := &fakeHIPDriver{available: true, skipLaunchRecording: true, skipDriverRecording: true}
	hipPrewarmAdamWUpdateBuffers(driver, stateTotalLen(state), 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hipAdamWUpdateLaunchBenchErr = hipRunAdamWUpdateKernel(context.Background(), driver, req)
		if hipAdamWUpdateLaunchBenchErr != nil {
			b.Fatal(hipAdamWUpdateLaunchBenchErr)
		}
	}
}

func hipAdamWReadDeviceFloat32(t *testing.T, driver *fakeHIPDriver, pointer nativeDevicePointer, sizeBytes int) []float32 {
	t.Helper()
	data := make([]byte, sizeBytes)
	core.RequireNoError(t, driver.CopyDeviceToHost(pointer, data))
	values, err := hipFloat32PayloadValues(data)
	core.RequireNoError(t, err)
	return values
}
