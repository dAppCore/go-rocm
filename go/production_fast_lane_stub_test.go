// SPDX-Licence-Identifier: EUPL-1.2

//go:build !linux || !amd64 || rocm_legacy_server

package rocm

import "testing"

func TestProductionFastLane_PortableContract_Good(t *testing.T) {
	fast := DefaultProductionFastLane()

	if fast.Name != ProductionFastLaneName ||
		fast.Backend != "rocm" ||
		fast.Library != "go-rocm" ||
		fast.ReferenceBackend != "go-mlx" {
		t.Fatalf("portable fast lane identity = %+v, want ROCm production contract", fast)
	}
	if fast.ModelID != portableProductionLaneCurrentModelID ||
		fast.LockedModelID != portableProductionLaneModelID ||
		fast.OfficialTargetModelID != portableOfficialGemma4E2BTargetModelID ||
		fast.AssistantModelID != portableOfficialGemma4E2BAssistantModelID {
		t.Fatalf("portable fast lane model IDs = %+v, want Gemma4 E2B target/assistant contract", fast)
	}
	if fast.Architecture != portableProductionLaneArchitecture ||
		fast.ChatTemplate != portableProductionLaneChatTemplate ||
		fast.QuantBits != portableProductionLaneProductDefaultQuantBits ||
		fast.QuantMode != portableProductionFastLaneQuantMode ||
		fast.QuantGroup != portableProductionFastLaneQuantGroup ||
		fast.CacheMode != portableProductionFastLaneCacheMode {
		t.Fatalf("portable fast lane defaults = %+v, want q6 affine turboquant Gemma4 defaults", fast)
	}
	if !fast.EnabledByDefault || fast.RequiresEnvGate || fast.RequiresCLIFlag {
		t.Fatalf("portable fast lane gates = default:%t env:%t cli:%t, want production default with no gates", fast.EnabledByDefault, fast.RequiresEnvGate, fast.RequiresCLIFlag)
	}
	for _, metric := range []string{"load_duration", "retained_workflow", "candidate_cache_mode", "mtp_draft_calls", "turboquant_candidate_cache_mode"} {
		if !portableFastLaneMetricContains(fast.RequiredMetrics, metric) {
			t.Fatalf("portable fast lane metrics = %v, missing %q", fast.RequiredMetrics, metric)
		}
	}
	if fast.Labels["production_fast_lane"] != "true" ||
		fast.Labels["production_default"] != "true" ||
		fast.Labels["production_requires_env_gate"] != "false" ||
		fast.Labels["production_requires_cli_flag"] != "false" ||
		fast.Labels["production_build"] != "portable" ||
		fast.Labels["production_cache_mode"] != portableProductionFastLaneCacheMode ||
		fast.Labels["production_mtp_assistant_model"] != portableOfficialGemma4E2BAssistantModelID {
		t.Fatalf("portable fast lane labels = %+v, want production-shaped portable API contract", fast.Labels)
	}

	fast.RequiredMetrics[0] = "mutated"
	fast.Labels["production_fast_lane"] = "mutated"
	next := DefaultProductionFastLane()
	if next.RequiredMetrics[0] == "mutated" || next.Labels["production_fast_lane"] == "mutated" {
		t.Fatalf("DefaultProductionFastLane aliases caller mutation in portable build: %+v", next)
	}
}

func TestProductionTurboQuantPolicy_PortableDefaults_Good(t *testing.T) {
	policy := DefaultProductionTurboQuantPolicy()

	if policy.TargetModelID != portableProductionLaneCurrentModelID ||
		policy.CacheMode != portableProductionTurboQuantKVMode ||
		policy.Mode != portableProductionTurboQuantKVMode ||
		policy.TargetEffectiveBitsMilli != 3500 ||
		policy.RequiredLayoutVersion != ProductionTurboQuantKVLayoutVersion ||
		policy.RequiredKeyAlgorithm != ProductionTurboQuantKeyAlgorithm ||
		policy.RequiredValueAlgorithm != ProductionTurboQuantValueAlgorithm ||
		policy.RequiredOutlierPolicy != ProductionTurboQuantOutlierPolicy {
		t.Fatalf("portable TurboQuant policy = %+v, want native production defaults", policy)
	}
	if !policy.EnabledByDefault ||
		policy.RequiresExplicitOptIn ||
		!policy.RequiresRetainedWorkflow ||
		!policy.RequiresQualityParity ||
		!policy.RequiresSideBySideBenchmark ||
		!policy.RequiresNormalContextValidation ||
		!policy.RequiresStressContextValidation ||
		policy.MinimumRetainedTurns != portableProductionRetainedTurns ||
		policy.NormalContextLength != portableProductionLongContextLength ||
		policy.StressContextLength != portableProductionHyperLongContextLength {
		t.Fatalf("portable TurboQuant evidence gates = %+v, want production fast-lane policy", policy)
	}
	for _, mode := range []string{"fp16", portableProductionTurboQuantCacheModePaged, "q8", "k-q8-v-q4"} {
		if !portableFastLaneMetricContains(policy.CompareAgainstCacheModes, mode) {
			t.Fatalf("portable TurboQuant compare modes = %v, missing %q", policy.CompareAgainstCacheModes, mode)
		}
	}
	for _, metric := range []string{"retained_workflow", "candidate_cache_mode", "candidate_qjl_residual", "normal_context_validated", "stress_context_validated"} {
		if !portableFastLaneMetricContains(policy.RequiredMetrics, metric) {
			t.Fatalf("portable TurboQuant metrics = %v, missing %q", policy.RequiredMetrics, metric)
		}
	}

	policy.CompareAgainstCacheModes[0] = "mutated"
	policy.RequiredMetrics[0] = "mutated"
	next := DefaultProductionTurboQuantPolicy()
	if next.CompareAgainstCacheModes[0] == "mutated" || next.RequiredMetrics[0] == "mutated" {
		t.Fatalf("DefaultProductionTurboQuantPolicy aliases caller mutation in portable build: %+v", next)
	}
}

func TestProductionCombinedMTPAndTurboQuantPolicy_PortableDefaults_Good(t *testing.T) {
	policy := DefaultProductionCombinedMTPAndTurboQuantPolicy()

	if policy.TargetModelID != portableOfficialGemma4E2BTargetModelID ||
		policy.AssistantModelID != portableOfficialGemma4E2BAssistantModelID ||
		policy.Mode != ProductionCombinedMTPAndTurboQuantMode ||
		policy.CacheMode != portableProductionTurboQuantKVMode {
		t.Fatalf("portable combined policy = %+v, want Gemma4 MTP+TurboQuant defaults", policy)
	}
	if !policy.EnabledByDefault ||
		policy.RequiresExplicitOptIn ||
		!policy.RequiresRetainedWorkflow ||
		!policy.RequiresGreedyParity ||
		!policy.RequiresTurboQuantQualityParity ||
		!policy.RequiresMTPPromotion ||
		!policy.RequiresTurboQuantPromotion ||
		policy.MinimumRetainedTurns != portableProductionRetainedTurns {
		t.Fatalf("portable combined policy gates = %+v, want production fast-lane policy", policy)
	}
	for _, metric := range []string{"retained_workflow", "mtp_draft_calls", "turboquant_candidate_cache_mode", "turboquant_candidate_qjl_residual", "gemma4_family_pair_verified"} {
		if !portableFastLaneMetricContains(policy.RequiredMetrics, metric) {
			t.Fatalf("portable combined metrics = %v, missing %q", policy.RequiredMetrics, metric)
		}
	}

	policy.RequiredMetrics[0] = "mutated"
	next := DefaultProductionCombinedMTPAndTurboQuantPolicy()
	if next.RequiredMetrics[0] == "mutated" {
		t.Fatalf("DefaultProductionCombinedMTPAndTurboQuantPolicy aliases caller mutation in portable build: %+v", next)
	}
}

func portableFastLaneMetricContains(metrics []string, want string) bool {
	for _, metric := range metrics {
		if metric == want {
			return true
		}
	}
	return false
}
