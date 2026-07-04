// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"os"
	"testing"

	core "dappco.re/go"
)

func TestHIPHardwareAutoRoundQuantizeKernel_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware smoke tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled kernel bundle containing rocm_autoround_quantize")
	}
	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatalf("native ROCm runtime is not available")
	}
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || hipRuntime.driver == nil {
		t.Fatalf("runtime = %T, want hipRuntime with driver", runtime)
	}
	profile, ok := ProductionAutoRoundQuantizationProfileByName("int2")
	core.AssertEqual(t, true, ok)
	weights := make([]float32, 128)
	for index := range weights {
		switch index % 3 {
		case 0:
			weights[index] = -1
		case 1:
			weights[index] = 0
		default:
			weights[index] = 1
		}
	}
	result, err := hipRunAutoRoundQuantizeKernel(context.Background(), hipRuntime.driver, hipAutoRoundQuantizeRequest{
		Weights: weights,
		Plan:    DefaultProductionAutoRoundCalibrationPlan(profile),
		Rows:    1,
		Cols:    128,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 32, len(result.Packed))
	core.AssertEqual(t, 1, len(result.Scales))
	assertFloat32Near(t, 1, result.Scales[0])
	if !hipAutoRoundPackedOutputHasNonZeroByte(result.Packed) {
		t.Fatalf("packed AutoRound INT2 output is all zero: %v", result.Packed)
	}
}

var hipAutoRoundQuantizeResultSink hipAutoRoundQuantizeResult

func BenchmarkHIPHardwareAutoRoundQuantize_MXFP4Rows128Cols128(b *testing.B) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		b.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware benchmarks")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		b.Skip("set GO_ROCM_KERNEL_HSACO to a compiled kernel bundle containing rocm_autoround_quantize")
	}
	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		b.Fatalf("native ROCm runtime is not available")
	}
	hipRuntime, ok := runtime.(*hipRuntime)
	if !ok || hipRuntime.driver == nil {
		b.Fatalf("runtime = %T, want hipRuntime with driver", runtime)
	}
	profile, ok := ProductionAutoRoundQuantizationProfileByName("mxfp4")
	if !ok {
		b.Fatal("missing MXFP4 AutoRound profile")
	}
	req := hipAutoRoundQuantizeRequest{
		Weights: autoroundLaunchWeights(128 * 128),
		Plan:    DefaultProductionAutoRoundCalibrationPlan(profile),
		Rows:    128,
		Cols:    128,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		result, err := hipRunAutoRoundQuantizeKernel(context.Background(), hipRuntime.driver, req)
		if err != nil {
			b.Fatal(err)
		}
		if len(result.Packed) != 128*128/2 || len(result.Scales) != 128 {
			b.Fatalf("result shape = packed:%d scales:%d, want packed:%d scales:%d", len(result.Packed), len(result.Scales), 128*128/2, 128)
		}
		hipAutoRoundQuantizeResultSink = result
	}
}

func hipAutoRoundPackedOutputHasNonZeroByte(packed []byte) bool {
	for _, value := range packed {
		if value != 0 {
			return true
		}
	}
	return false
}
