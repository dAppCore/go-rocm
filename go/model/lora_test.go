// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestDefaultLoRAAdapterRoutes_Good_ArchitectureCatalogue(t *testing.T) {
	routes := DefaultLoRAAdapterRoutes()
	if len(routes) == 0 {
		t.Fatal("DefaultLoRAAdapterRoutes returned no routes")
	}
	byArchitecture := map[string]LoRAAdapterRoute{}
	for _, route := range routes {
		byArchitecture[route.Architecture] = route
	}
	gemma4 := byArchitecture["gemma4_text"]
	if !gemma4.Matched() ||
		gemma4.Contract != LoRAAdapterRegistryContract ||
		gemma4.Name != LoRAAdapterRouteName ||
		gemma4.Loader != LoRAAdapterLoaderLinear ||
		gemma4.Runtime != LoRAAdapterRuntimeHIP ||
		gemma4.Status != LoRAAdapterRouteExperimentalNative ||
		gemma4.TargetPolicy != "gemma4" ||
		!slices.Contains(gemma4.DefaultTargets, "q_proj") ||
		gemma4.TargetPaths["q_proj"] != "self_attn.q_proj" ||
		!gemma4.Registered ||
		!gemma4.NativeRuntime ||
		!gemma4.ApplySupported ||
		!gemma4.LoadSupported ||
		!gemma4.FuseSupported ||
		!gemma4.TrainingSupported ||
		!gemma4.RequiresExtendedOptIn ||
		!slices.Contains(gemma4.Capabilities, inference.CapabilityLoRAInference) ||
		gemma4.Labels["engine_lora_target_policy"] != "gemma4" {
		t.Fatalf("gemma4_text LoRA route = %+v, want native Gemma4 adapter route", gemma4)
	}
	qwen := byArchitecture["qwen3"]
	if !qwen.Matched() ||
		qwen.TargetPolicy != "decoder" ||
		!slices.Equal(qwen.DefaultTargets, []string{"q_proj", "v_proj"}) ||
		qwen.TargetPaths["gate_proj"] != "mlp.gate_proj" ||
		!qwen.Registered ||
		!qwen.NativeRuntime {
		t.Fatalf("qwen3 LoRA route = %+v, want decoder target policy", qwen)
	}
	if assistant := byArchitecture["gemma4_assistant"]; assistant.Matched() {
		t.Fatalf("gemma4_assistant LoRA route = %+v, want no attached-only adapter route", assistant)
	}
}

func TestLoRAAdapterRouteForArchitecture_Good_NormalizesAliases(t *testing.T) {
	route, ok := LoRAAdapterRouteForArchitecture("Qwen3ForCausalLM")
	if !ok ||
		route.Architecture != "qwen3" ||
		route.Family != "qwen" ||
		route.TargetPolicy != "decoder" ||
		route.TargetPaths["q_proj"] != "self_attn.q_proj" {
		t.Fatalf("LoRAAdapterRouteForArchitecture(Qwen3ForCausalLM) = %+v ok=%v, want qwen3 decoder route", route, ok)
	}
	if _, ok := LoRAAdapterRouteForArchitecture("BertForSequenceClassification"); ok {
		t.Fatal("LoRAAdapterRouteForArchitecture(BERT rerank) ok = true, want false")
	}
}

func TestLoRATargetPolicy_Good_PackageOwnedHelpers(t *testing.T) {
	policy, ok := LoRATargetPolicyForArchitecture("qwen3")
	if !ok ||
		!slices.Equal(policy.DefaultTargets, []string{"q_proj", "v_proj"}) ||
		policy.TargetPaths["gate_proj"] != "mlp.gate_proj" {
		t.Fatalf("LoRATargetPolicyForArchitecture(qwen3) = %+v ok=%v, want decoder policy", policy, ok)
	}
	policy.TargetPaths["gate_proj"] = "mutated"
	policy, _ = LoRATargetPolicyForArchitecture("qwen3")
	if policy.TargetPaths["gate_proj"] != "mlp.gate_proj" {
		t.Fatalf("LoRATargetPolicyForArchitecture leaked mutable target paths: %+v", policy.TargetPaths)
	}
	if path, ok := LoRATargetPath("qwen3", "q_proj"); !ok || path != "self_attn.q_proj" {
		t.Fatalf("LoRATargetPath(qwen3, q_proj) = %q ok=%v, want self_attn.q_proj true", path, ok)
	}
	if canonical, ok := LoRACanonicalTarget("qwen3", "model.layers.0.q_proj"); !ok || canonical != "model.layers.0.self_attn.q_proj" {
		t.Fatalf("LoRACanonicalTarget(qwen3) = %q ok=%v, want model.layers.0.self_attn.q_proj true", canonical, ok)
	}
	if !LoRASafeTarget("qwen3", "q_proj") || LoRAExtendedTarget("qwen3", "q_proj") {
		t.Fatalf("qwen3 LoRA target safety mismatch")
	}
}

func TestRegisterLoRAAdapterRoute_Good_ExtendsReactiveCatalogue(t *testing.T) {
	restoreRegisteredLoRAAdaptersForTest(t)

	RegisterLoRAAdapterRoute(LoRAAdapterRoute{})
	RegisterLoRAAdapterRoute(LoRAAdapterRoute{
		Architecture:   "fake-loader",
		TargetPolicy:   "first-policy",
		DefaultTargets: []string{"old_proj"},
		TargetPaths:    map[string]string{"old_proj": "old.proj"},
	})
	RegisterLoRAAdapterRoute(LoRAAdapterRoute{
		Architecture:      "fake-loader",
		Family:            "fake",
		RuntimeStatus:     inference.FeatureRuntimeNative,
		TargetPolicy:      "fake-policy",
		DefaultTargets:    []string{"q_proj", "v_proj", "q_proj"},
		SafeTargets:       []string{"q_proj", "v_proj"},
		ExtendedTargets:   []string{"router.proj"},
		TargetPaths:       map[string]string{"q_proj": "blocks.attn.q", "v_proj": "blocks.attn.v", "router.proj": "router.proj"},
		NativeRuntime:     true,
		TrainingSupported: true,
	})

	if got := RegisteredLoRAAdapterArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredLoRAAdapterArchitectures = %v, want normalized replacement", got)
	}
	registeredRoutes := RegisteredLoRAAdapterRoutes()
	if len(registeredRoutes) != 1 ||
		registeredRoutes[0].Architecture != "fake_loader" ||
		registeredRoutes[0].TargetPolicy != "fake-policy" {
		t.Fatalf("RegisteredLoRAAdapterRoutes = %+v, want one replacement route", registeredRoutes)
	}
	routesForReplace := RegisteredLoRAAdapterRoutes()
	registeredRoute, ok := RegisteredLoRAAdapterRouteForArchitecture("fake-loader")
	if !ok || registeredRoute.TargetPolicy != "fake-policy" {
		t.Fatalf("RegisteredLoRAAdapterRouteForArchitecture(fake-loader) = %+v ok=%v, want registered route", registeredRoute, ok)
	}
	registeredRoutes[0].Labels["engine_lora_target_policy"] = "mutated"
	registeredRoutes[0].TargetPaths["q_proj"] = "mutated"
	nextRegisteredRoute, ok := RegisteredLoRAAdapterRouteForArchitecture("fake-loader")
	if !ok ||
		nextRegisteredRoute.Labels["engine_lora_target_policy"] != "fake-policy" ||
		nextRegisteredRoute.TargetPaths["q_proj"] != "blocks.attn.q" {
		t.Fatalf("RegisteredLoRAAdapterRoutes leaked mutable state: %+v ok=%v", nextRegisteredRoute, ok)
	}
	ReplaceRegisteredLoRAAdapterRoutes(routesForReplace)
	replaced, ok := RegisteredLoRAAdapterRouteForArchitecture("fake-loader")
	if !ok ||
		replaced.Labels["engine_lora_target_policy"] != "fake-policy" ||
		replaced.TargetPaths["q_proj"] != "blocks.attn.q" {
		t.Fatalf("ReplaceRegisteredLoRAAdapterRoutes = %+v ok=%v, want copy-safe restored route", replaced, ok)
	}

	route, ok := LoRAAdapterRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != LoRAAdapterRegistryContract ||
		route.Name != LoRAAdapterRouteName ||
		route.Architecture != "fake_loader" ||
		route.Family != "fake" ||
		route.Loader != LoRAAdapterLoaderLinear ||
		route.Runtime != LoRAAdapterRuntimeHIP ||
		route.RuntimeStatus != inference.FeatureRuntimeNative ||
		route.Status != LoRAAdapterRouteExperimentalNative ||
		route.TargetPolicy != "fake-policy" ||
		!slices.Equal(route.DefaultTargets, []string{"q_proj", "v_proj"}) ||
		!slices.Equal(route.SafeTargets, []string{"q_proj", "v_proj"}) ||
		!slices.Equal(route.ExtendedTargets, []string{"router.proj"}) ||
		route.TargetPaths["q_proj"] != "blocks.attn.q" ||
		route.TargetPaths["router.proj"] != "router.proj" ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.ApplySupported ||
		!route.LoadSupported ||
		!route.FuseSupported ||
		!route.TrainingSupported ||
		route.Staged ||
		route.Planned ||
		!route.RequiresExtendedOptIn ||
		!slices.Equal(route.Capabilities, []inference.CapabilityID{inference.CapabilityLoRAInference, inference.CapabilityLoRATraining, inference.CapabilityModelMerge}) ||
		route.Labels["engine_lora_target_policy"] != "fake-policy" ||
		route.Labels["engine_lora_default_targets"] != "q_proj,v_proj" ||
		route.Labels["engine_lora_capabilities"] != "lora.inference,lora.training,model.merge" {
		t.Fatalf("LoRAAdapterRouteForArchitecture(fake-loader) = %+v ok=%v, want registered adapter route", route, ok)
	}
	route.TargetPaths["q_proj"] = "mutated"
	route.DefaultTargets[0] = "mutated"
	next, ok := LoRAAdapterRouteForArchitecture("fake-loader")
	if !ok ||
		next.TargetPaths["q_proj"] != "blocks.attn.q" ||
		!slices.Equal(next.DefaultTargets, []string{"q_proj", "v_proj"}) {
		t.Fatalf("LoRAAdapterRouteForArchitecture leaked mutable state: %+v ok=%v", next, ok)
	}
	if !slices.ContainsFunc(DefaultLoRAAdapterRoutes(), func(route LoRAAdapterRoute) bool {
		return route.Architecture == "fake_loader" && route.TargetPolicy == "fake-policy"
	}) {
		t.Fatalf("DefaultLoRAAdapterRoutes missing fake_loader registration")
	}
	policy, ok := LoRATargetPolicyForArchitecture("fake-loader")
	if !ok ||
		!slices.Equal(policy.DefaultTargets, []string{"q_proj", "v_proj"}) ||
		policy.TargetPaths["q_proj"] != "blocks.attn.q" {
		t.Fatalf("LoRATargetPolicyForArchitecture(fake-loader) = %+v ok=%v, want registered target policy", policy, ok)
	}
	if canonical, ok := LoRACanonicalTarget("fake-loader", "model.layers.0.q_proj"); !ok || canonical != "model.layers.0.blocks.attn.q" {
		t.Fatalf("LoRACanonicalTarget(fake-loader) = %q ok=%v, want registered canonical path", canonical, ok)
	}
}

func restoreRegisteredLoRAAdaptersForTest(t *testing.T) {
	t.Helper()
	order, routes := registeredLoRAAdapters.Snapshot()
	for architecture, route := range routes {
		routes[architecture] = route.Clone()
	}
	t.Cleanup(func() {
		restoreRoutes := make(map[string]LoRAAdapterRoute, len(routes))
		for architecture, route := range routes {
			restoreRoutes[architecture] = route.Clone()
		}
		registeredLoRAAdapters.Restore(order, restoreRoutes)
	})
}
