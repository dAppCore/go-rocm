// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import "testing"

var moeTextRuntimeSummaryBenchSink MoETextRuntimeSummary

func BenchmarkMoETextLayersRuntimeAvailable_Dense64(b *testing.B) {
	layers := make([]moeRuntimeTestLayer, 64)
	for i := range layers {
		layers[i] = moeRuntimeTestLayer{dense: true}
	}
	b.ReportAllocs()
	for b.Loop() {
		moeTextRuntimeSummaryBenchSink = SummarizeMoETextLayersRuntime(layers, moeRuntimeTestParts)
	}
}

func BenchmarkMoETextLayersRuntimeAvailable_Sparse64EveryOther(b *testing.B) {
	layers := make([]moeRuntimeTestLayer, 64)
	for i := range layers {
		layers[i] = moeRuntimeTestLayer{dense: true}
		if i%2 == 1 {
			layers[i].sparse = true
			layers[i].router = true
			layers[i].experts = true
		}
	}
	b.ReportAllocs()
	for b.Loop() {
		moeTextRuntimeSummaryBenchSink = SummarizeMoETextLayersRuntime(layers, moeRuntimeTestParts)
	}
}
