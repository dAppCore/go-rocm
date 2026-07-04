// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"testing"

	core "dappco.re/go"
)

var (
	adamWFileBenchPayload []byte
	adamWFileBenchState   *NativeAdamWState
	adamWFileBenchErr     error
)

func BenchmarkNativeAdamWState_MarshalLoRA_4096xRank8(b *testing.B) {
	state := nativeAdamWBenchState(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		adamWFileBenchPayload, adamWFileBenchErr = MarshalNativeAdamWState(state)
	}
}

func BenchmarkNativeAdamWState_UnmarshalLoRA_4096xRank8(b *testing.B) {
	state := nativeAdamWBenchState(b)
	payload, err := MarshalNativeAdamWState(state)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adamWFileBenchState, adamWFileBenchErr = UnmarshalNativeAdamWState(payload)
	}
}

func BenchmarkNativeAdamWStateTrack_ListLoRA_4Frames(b *testing.B) {
	path := core.PathJoin(b.TempDir(), "adamw.track")
	state := nativeAdamWBenchState(b)
	gradients := [][]float32{
		make([]float32, 8*4096),
		make([]float32, 4096*8),
	}
	for i := 0; i < 4; i++ {
		if i != 0 {
			if err := state.StepInPlace(gradients); err != nil {
				b.Fatal(err)
			}
		}
		if _, err := AppendNativeAdamWStateTrack(path, state); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, adamWFileBenchErr = ListNativeAdamWStateTrack(path)
	}
}

func nativeAdamWBenchState(b *testing.B) *NativeAdamWState {
	b.Helper()
	a := make([]float32, 8*4096)
	loraB := make([]float32, 4096*8)
	state, err := NewNativeLoRAAdamWState(a, loraB, 4096, 4096, 8, NativeAdamWConfig{LearningRate: 1e-4})
	if err != nil {
		b.Fatal(err)
	}
	return state
}
