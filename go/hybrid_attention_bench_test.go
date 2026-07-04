// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import "testing"

func BenchmarkBuildHybridAttentionCachePlan_Qwen36_64Layers(b *testing.B) {
	for b.Loop() {
		plan, err := BuildHybridAttentionCachePlan(64, []string{"linear_attention", "full_attention"}, 512)
		if err != nil {
			b.Fatal(err)
		}
		if plan.GlobalLayers != 32 {
			b.Fatalf("GlobalLayers = %d, want 32", plan.GlobalLayers)
		}
	}
}
