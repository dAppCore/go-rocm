// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestRegisterROCmLoRAAdapterRoute_Good_ExtendsReactiveAdapterRegistry(t *testing.T) {
	restoreRegisteredROCmLoRAAdapterRoutesForTest(t)

	RegisterROCmLoRAAdapterRoute(ROCmLoRAAdapterRoute{})
	RegisterROCmLoRAAdapterRoute(ROCmLoRAAdapterRoute{
		Architecture:   "fake-loader",
		TargetPolicy:   "first-policy",
		DefaultTargets: []string{"old_proj"},
		TargetPaths:    map[string]string{"old_proj": "old.proj"},
	})
	RegisterROCmLoRAAdapterRoute(ROCmLoRAAdapterRoute{
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

	if got := RegisteredROCmLoRAAdapterRouteArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmLoRAAdapterRouteArchitectures = %v, want normalized replacement under one architecture", got)
	}
	registered := RegisteredROCmLoRAAdapterRouteArchitectures()
	registered[0] = "mutated"
	if next := RegisteredROCmLoRAAdapterRouteArchitectures(); !slices.Equal(next, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmLoRAAdapterRouteArchitectures returned mutable state: %v", next)
	}

	route, ok := ROCmLoRAAdapterRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != ROCmLoRAAdapterRegistryContract ||
		route.Name != rocmLoRAAdapterRegistryRouteName ||
		route.Architecture != "fake_loader" ||
		route.Family != "fake" ||
		route.Loader != rocmLoRAAdapterLoaderLinear ||
		route.Runtime != rocmLoRAAdapterRuntimeHIP ||
		route.RuntimeStatus != inference.FeatureRuntimeNative ||
		route.Status != ROCmLoRAAdapterRouteExperimentalNative ||
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
		t.Fatalf("ROCmLoRAAdapterRouteForArchitecture(fake-loader) = %+v ok=%v, want registered adapter route", route, ok)
	}

	route.TargetPaths["q_proj"] = "mutated"
	route.DefaultTargets[0] = "mutated"
	nextRoute, ok := ROCmLoRAAdapterRouteForArchitecture("fake-loader")
	if !ok ||
		nextRoute.TargetPaths["q_proj"] != "blocks.attn.q" ||
		!slices.Equal(nextRoute.DefaultTargets, []string{"q_proj", "v_proj"}) {
		t.Fatalf("ROCmLoRAAdapterRouteForArchitecture leaked mutable state: %+v ok=%v", nextRoute, ok)
	}
	modelRoute, ok := rocmmodel.RegisteredLoRAAdapterRouteForArchitecture("fake-loader")
	if !ok ||
		modelRoute.TargetPolicy != "fake-policy" ||
		modelRoute.TargetPaths["q_proj"] != "blocks.attn.q" ||
		!slices.Equal(modelRoute.DefaultTargets, []string{"q_proj", "v_proj"}) ||
		modelRoute.Labels["engine_lora_target_policy"] != "fake-policy" {
		t.Fatalf("model.RegisteredLoRAAdapterRouteForArchitecture(fake-loader) = %+v ok=%v, want mirrored route", modelRoute, ok)
	}

	defaults := DefaultROCmLoRAAdapterRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmLoRAAdapterRoute) bool {
		return route.Architecture == "fake_loader" && route.TargetPolicy == "fake-policy"
	}) {
		t.Fatalf("DefaultROCmLoRAAdapterRoutes missing registered adapter route: %+v", defaults)
	}

	policy, ok := ROCmLoRATargetPolicyForArchitecture("fake-loader")
	if !ok ||
		!slices.Equal(policy.DefaultTargets, []string{"q_proj", "v_proj"}) ||
		policy.TargetPaths["q_proj"] != "blocks.attn.q" {
		t.Fatalf("ROCmLoRATargetPolicyForArchitecture(fake-loader) = %+v ok=%v, want registered target policy", policy, ok)
	}
	policy.TargetPaths["q_proj"] = "mutated"
	policy, _ = ROCmLoRATargetPolicyForArchitecture("fake-loader")
	if policy.TargetPaths["q_proj"] != "blocks.attn.q" {
		t.Fatalf("ROCmLoRATargetPolicyForArchitecture leaked mutable target paths: %+v", policy.TargetPaths)
	}
	if path, ok := ROCmLoRATargetPath("fake-loader", "q_proj"); !ok || path != "blocks.attn.q" {
		t.Fatalf("ROCmLoRATargetPath(fake-loader, q_proj) = %q ok=%v, want registered path", path, ok)
	}
	if !ROCmLoRASafeTarget("fake-loader", "q_proj") ||
		ROCmLoRASafeTarget("fake-loader", "router.proj") ||
		!ROCmLoRAExtendedTarget("fake-loader", "router.proj") {
		t.Fatalf("registered fake LoRA target safety mismatch")
	}
	if canonical, ok := ROCmLoRACanonicalTarget("fake-loader", "model.layers.0.q_proj"); !ok || canonical != "model.layers.0.blocks.attn.q" {
		t.Fatalf("ROCmLoRACanonicalTarget(fake-loader) = %q ok=%v, want registered canonical path", canonical, ok)
	}

	profile := ROCmModelProfile{
		Name:         "fake",
		Family:       "fake",
		Architecture: "fake_loader",
		ArchitectureProfile: ROCmArchitectureProfile{
			ID:            "fake_loader",
			Family:        "fake",
			RuntimeStatus: inference.FeatureRuntimeNative,
			NativeRuntime: true,
			Generation:    true,
		},
		FeatureRoute: ROCmModelFeatureRoute{
			Contract:      ROCmModelFeatureRegistryContract,
			Name:          rocmModelFeatureRegistryRouteName,
			Architecture:  "fake_loader",
			Family:        "fake",
			NativeRuntime: true,
			TextGenerate:  true,
		},
	}
	profileRoute := ROCmLoRAAdapterRouteForProfile(profile)
	if profileRoute.TargetPolicy != "fake-policy" ||
		profileRoute.TargetPaths["q_proj"] != "blocks.attn.q" ||
		!profileRoute.Registered ||
		!profileRoute.NativeRuntime ||
		profileRoute.Labels["engine_lora_target_policy"] != "fake-policy" {
		t.Fatalf("ROCmLoRAAdapterRouteForProfile registered route = %+v, want profile to use registered LoRA route", profileRoute)
	}
}

func TestRegisterROCmLoRAAdapterRoute_Good_SeesModelPackageRegistration(t *testing.T) {
	restoreRegisteredROCmLoRAAdapterRoutesForTest(t)

	rocmmodel.RegisterLoRAAdapterRoute(rocmmodel.LoRAAdapterRoute{
		Architecture:      "folder-lora",
		Family:            "folder",
		RuntimeStatus:     inference.FeatureRuntimeNative,
		TargetPolicy:      "folder-policy",
		DefaultTargets:    []string{"q_proj", "v_proj", "q_proj"},
		SafeTargets:       []string{"q_proj", "v_proj"},
		ExtendedTargets:   []string{"router.proj"},
		TargetPaths:       map[string]string{"q_proj": "blocks.attn.q", "v_proj": "blocks.attn.v", "router.proj": "router.proj"},
		NativeRuntime:     true,
		TrainingSupported: true,
	})

	route, ok := ROCmLoRAAdapterRouteForArchitecture("folder-lora")
	if !ok ||
		route.Contract != ROCmLoRAAdapterRegistryContract ||
		route.Name != rocmLoRAAdapterRegistryRouteName ||
		route.Architecture != "folder_lora" ||
		route.Family != "folder" ||
		route.Loader != rocmLoRAAdapterLoaderLinear ||
		route.Runtime != rocmLoRAAdapterRuntimeHIP ||
		route.RuntimeStatus != inference.FeatureRuntimeNative ||
		route.Status != ROCmLoRAAdapterRouteExperimentalNative ||
		route.TargetPolicy != "folder-policy" ||
		!slices.Equal(route.DefaultTargets, []string{"q_proj", "v_proj"}) ||
		!slices.Equal(route.SafeTargets, []string{"q_proj", "v_proj"}) ||
		!slices.Equal(route.ExtendedTargets, []string{"router.proj"}) ||
		route.TargetPaths["q_proj"] != "blocks.attn.q" ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.ApplySupported ||
		!route.LoadSupported ||
		!route.FuseSupported ||
		!route.TrainingSupported ||
		!route.RequiresExtendedOptIn ||
		!slices.Contains(route.Capabilities, inference.CapabilityLoRAInference) ||
		route.Labels["engine_lora_target_policy"] != "folder-policy" {
		t.Fatalf("ROCmLoRAAdapterRouteForArchitecture(folder-lora) = %+v ok=%v, want model package route", route, ok)
	}
	defaults := DefaultROCmLoRAAdapterRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmLoRAAdapterRoute) bool {
		return route.Architecture == "folder_lora" && route.TargetPolicy == "folder-policy"
	}) {
		t.Fatalf("DefaultROCmLoRAAdapterRoutes missing model package route: %+v", defaults)
	}
	policy, ok := ROCmLoRATargetPolicyForArchitecture("folder-lora")
	if !ok ||
		!slices.Equal(policy.DefaultTargets, []string{"q_proj", "v_proj"}) ||
		policy.TargetPaths["q_proj"] != "blocks.attn.q" {
		t.Fatalf("ROCmLoRATargetPolicyForArchitecture(folder-lora) = %+v ok=%v, want model package target policy", policy, ok)
	}
	if canonical, ok := ROCmLoRACanonicalTarget("folder-lora", "model.layers.0.q_proj"); !ok || canonical != "model.layers.0.blocks.attn.q" {
		t.Fatalf("ROCmLoRACanonicalTarget(folder-lora) = %q ok=%v, want registered canonical path", canonical, ok)
	}
	profileRoute := ROCmLoRAAdapterRouteForProfile(ROCmModelProfile{
		Name:         "folder",
		Family:       "folder",
		Architecture: "folder_lora",
	})
	if profileRoute.TargetPolicy != "folder-policy" ||
		profileRoute.TargetPaths["q_proj"] != "blocks.attn.q" ||
		!profileRoute.NativeRuntime ||
		!profileRoute.ApplySupported {
		t.Fatalf("ROCmLoRAAdapterRouteForProfile(folder_lora) = %+v, want model package LoRA route", profileRoute)
	}
}

func TestRegisterROCmLoRAAdapterRoute_Good_OverridesBuiltinAdapterRoute(t *testing.T) {
	restoreRegisteredROCmLoRAAdapterRoutesForTest(t)

	RegisterROCmLoRAAdapterRoute(ROCmLoRAAdapterRoute{
		Architecture:      "qwen3",
		Family:            "qwen",
		TargetPolicy:      "registered-qwen",
		DefaultTargets:    []string{"gate_proj"},
		SafeTargets:       []string{"gate_proj"},
		TargetPaths:       map[string]string{"gate_proj": "mlp.gate_proj"},
		NativeRuntime:     true,
		ApplySupported:    true,
		LoadSupported:     true,
		FuseSupported:     true,
		TrainingSupported: true,
	})

	route, ok := ROCmLoRAAdapterRouteForArchitecture("Qwen3ForCausalLM")
	if !ok ||
		route.Architecture != "qwen3" ||
		route.Family != "qwen" ||
		route.TargetPolicy != "registered-qwen" ||
		!slices.Equal(route.DefaultTargets, []string{"gate_proj"}) ||
		route.TargetPaths["gate_proj"] != "mlp.gate_proj" ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.ApplySupported ||
		!route.LoadSupported ||
		!route.FuseSupported ||
		!route.TrainingSupported ||
		route.Labels["engine_lora_target_policy"] != "registered-qwen" {
		t.Fatalf("ROCmLoRAAdapterRouteForArchitecture(qwen3 override) = %+v ok=%v, want registered override", route, ok)
	}
	if path, ok := ROCmLoRATargetPath("Qwen3ForCausalLM", "gate_proj"); !ok || path != "mlp.gate_proj" {
		t.Fatalf("ROCmLoRATargetPath(qwen3 registered override) = %q ok=%v, want registered target path", path, ok)
	}
	if path, ok := ROCmLoRATargetPath("Qwen3ForCausalLM", "q_proj"); ok || path != "" {
		t.Fatalf("ROCmLoRATargetPath(qwen3 q_proj) = %q ok=%v, want registered override to replace defaults", path, ok)
	}

	profile, ok := ResolveROCmModelProfile("/models/qwen", inference.ModelIdentity{Architecture: "Qwen3ForCausalLM"})
	if !ok ||
		profile.LoRAAdapterRoute.TargetPolicy != "registered-qwen" ||
		!slices.Equal(profile.LoRAAdapterRoute.DefaultTargets, []string{"gate_proj"}) ||
		profile.LoRAAdapterRoute.TargetPaths["gate_proj"] != "mlp.gate_proj" {
		t.Fatalf("ResolveROCmModelProfile(qwen3 registered LoRA override) = %+v ok=%v, want profile to expose registered adapter route", profile, ok)
	}
}

func restoreRegisteredROCmLoRAAdapterRoutesForTest(t *testing.T) {
	t.Helper()
	modelRoutes := rocmmodel.RegisteredLoRAAdapterRoutes()

	t.Cleanup(func() {
		restoreRoutes := make([]rocmmodel.LoRAAdapterRoute, 0, len(modelRoutes))
		for _, route := range modelRoutes {
			restoreRoutes = append(restoreRoutes, route.Clone())
		}
		rocmmodel.ReplaceRegisteredLoRAAdapterRoutes(restoreRoutes)
	})
}
