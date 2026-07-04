// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import "testing"

var (
	adamWBenchState *NativeAdamWState
	adamWBenchErr   error
)

func BenchmarkNativeAdamWState_InitLoRA_4096xRank8(b *testing.B) {
	a := make([]float32, 8*4096)
	loraB := make([]float32, 4096*8)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		adamWBenchState, adamWBenchErr = NewNativeLoRAAdamWState(a, loraB, 4096, 4096, 8, NativeAdamWConfig{})
	}
}

func BenchmarkNativeAdamWState_StepLoRA_4096xRank8(b *testing.B) {
	a := make([]float32, 8*4096)
	loraB := make([]float32, 4096*8)
	gradA := make([]float32, len(a))
	gradB := make([]float32, len(loraB))
	for i := range gradA {
		gradA[i] = 0.001
	}
	for i := range gradB {
		gradB[i] = -0.001
	}
	state, err := NewNativeLoRAAdamWState(a, loraB, 4096, 4096, 8, NativeAdamWConfig{LearningRate: 1e-4})
	if err != nil {
		b.Fatal(err)
	}
	gradients := [][]float32{gradA, gradB}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adamWBenchErr = state.StepInPlace(gradients)
	}
}
