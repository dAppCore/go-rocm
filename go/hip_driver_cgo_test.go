// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && cgo && !rocm_legacy_server

package rocm

import "testing"

var benchmarkCGOHIPLaunchArgModeSink cgoHIPLaunchArgMode

func BenchmarkCGOHIPLaunchArgModeConfig_Hot(b *testing.B) {
	_ = cgoHIPLaunchArgModeConfig()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchmarkCGOHIPLaunchArgModeSink = cgoHIPLaunchArgModeConfig()
	}
}
