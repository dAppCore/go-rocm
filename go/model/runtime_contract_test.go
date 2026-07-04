// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestRuntimeContractRouteForArchitecture_Good_GoMLXOptionalContracts(t *testing.T) {
	gemma4, ok := RuntimeContractRouteForArchitecture("Gemma4ForConditionalGeneration")
	if !ok ||
		!gemma4.Matched() ||
		gemma4.Architecture != "gemma4" ||
		gemma4.Family != "gemma4" ||
		!gemma4.NativeRuntime ||
		!gemma4.TextGenerate ||
		!gemma4.LastTokenLogits ||
		!gemma4.GreedyToken ||
		!gemma4.SuppressedGreedyToken ||
		!gemma4.QueryHeads ||
		!gemma4.LoRALinearResolver ||
		!gemma4.DenseSplitParts ||
		!gemma4.CacheTopology ||
		!gemma4.AttentionCacheLayout ||
		!gemma4.FixedSlidingPrefillLimit ||
		!gemma4.FixedSlidingCache ||
		!gemma4.ThoughtChannelSuppressor ||
		!gemma4.ModelInfoReporter ||
		!slices.Contains(gemma4.ContractIDs, RuntimeContractLastTokenLogits) ||
		!slices.Contains(gemma4.ContractIDs, RuntimeContractThoughtChannelSuppressor) ||
		gemma4.Labels["engine_runtime_contract_fixed_sliding_cache"] != "true" {
		t.Fatalf("RuntimeContractRouteForArchitecture(gemma4) = %+v ok=%v, want Gemma4 optional runtime hooks", gemma4, ok)
	}

	mamba, ok := RuntimeContractRouteForArchitecture("Mamba2ForCausalLM")
	if !ok ||
		!mamba.Matched() ||
		mamba.Architecture != "mamba2" ||
		mamba.NativeRuntime ||
		!mamba.MetadataOnly ||
		!mamba.ModelInfoReporter ||
		!mamba.DecodeUnavailableReporter ||
		mamba.GreedyToken ||
		!slices.Equal(mamba.ContractIDs, []RuntimeContractID{RuntimeContractModelInfoReporter, RuntimeContractDecodeUnavailableReport}) {
		t.Fatalf("RuntimeContractRouteForArchitecture(mamba2) = %+v ok=%v, want metadata-only reporting contracts", mamba, ok)
	}

	qwen36MoE, ok := RuntimeContractRouteForArchitecture("Qwen3.6MoeForConditionalGeneration")
	if !ok ||
		qwen36MoE.Architecture != "qwen3_6_moe" ||
		!qwen36MoE.NativeRuntime ||
		qwen36MoE.TextGenerate ||
		!qwen36MoE.MoETextRuntimeReporter ||
		!qwen36MoE.HybridAttentionCachePlanner ||
		!qwen36MoE.DecodeUnavailableReporter ||
		!slices.Contains(qwen36MoE.ContractIDs, RuntimeContractMoETextRuntimeReporter) ||
		!slices.Contains(qwen36MoE.ContractIDs, RuntimeContractHybridAttentionCachePlan) {
		t.Fatalf("RuntimeContractRouteForArchitecture(qwen3_6_moe) = %+v ok=%v, want staged hybrid/MoE contracts", qwen36MoE, ok)
	}
}

func TestRuntimeContractRouteForIdentity_Good_ResolvedLabelsWin(t *testing.T) {
	route, ok := RuntimeContractRouteForIdentity("/models/kimi", inference.ModelIdentity{
		Architecture: "Qwen3ForCausalLM",
		Labels: map[string]string{
			"engine_architecture_resolved": "kimi",
		},
	})
	if !ok ||
		route.Architecture != "kimi" ||
		!route.MoETextRuntimeReporter ||
		!route.DecodeUnavailableReporter ||
		route.Labels["engine_runtime_contract_architecture"] != "kimi" {
		t.Fatalf("RuntimeContractRouteForIdentity = %+v ok=%v, want label-refined kimi contracts", route, ok)
	}
}

func TestRegisterRuntimeContractRoute_Good_ExtendsAndIsCopySafe(t *testing.T) {
	restoreRegisteredRuntimeContractsForTest(t)

	RegisterRuntimeContractRoute(RuntimeContractRoute{})
	RegisterRuntimeContractRoute(RuntimeContractRoute{
		Architecture:              "fake-contract",
		Family:                    "fake",
		LastTokenLogits:           true,
		ModelInfoReporter:         true,
		RuntimeStatus:             inference.FeatureRuntimeNative,
		NativeRuntime:             true,
		ContractIDs:               []RuntimeContractID{RuntimeContractDecodeUnavailableReport},
		DecodeUnavailableReporter: true,
	})

	if got := RegisteredRuntimeContractArchitectures(); !slices.Equal(got, []string{"fake_contract"}) {
		t.Fatalf("RegisteredRuntimeContractArchitectures = %v, want fake_contract", got)
	}
	route, ok := RuntimeContractRouteForArchitecture("fake-contract")
	if !ok ||
		route.Contract != RuntimeContractRegistryContract ||
		route.Name != RuntimeContractRouteName ||
		route.Architecture != "fake_contract" ||
		route.Family != "fake" ||
		!route.LastTokenLogits ||
		!route.ModelInfoReporter ||
		!route.DecodeUnavailableReporter ||
		!slices.Equal(route.ContractIDs, []RuntimeContractID{RuntimeContractLastTokenLogits, RuntimeContractModelInfoReporter, RuntimeContractDecodeUnavailableReport}) {
		t.Fatalf("RuntimeContractRouteForArchitecture(fake-contract) = %+v ok=%v, want normalized registered contract route", route, ok)
	}
	route.Labels["engine_runtime_contract_architecture"] = "mutated"
	next, ok := RuntimeContractRouteForArchitecture("fake-contract")
	if !ok || next.Labels["engine_runtime_contract_architecture"] != "fake_contract" {
		t.Fatalf("RuntimeContractRouteForArchitecture leaked mutable labels: %+v ok=%v", next, ok)
	}
}

func restoreRegisteredRuntimeContractsForTest(t *testing.T) {
	t.Helper()
	order, routes := registeredRuntimeContracts.Snapshot()
	for architecture, route := range routes {
		routes[architecture] = route.Clone()
	}
	t.Cleanup(func() {
		restoreRoutes := make(map[string]RuntimeContractRoute, len(routes))
		for architecture, route := range routes {
			restoreRoutes[architecture] = route.Clone()
		}
		registeredRuntimeContracts.Restore(order, restoreRoutes)
	})
}
