// SPDX-Licence-Identifier: EUPL-1.2

package architecture

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/rocm/model"
)

func TestProfileFactory_Good_ResolvesArchitectureProfile(t *testing.T) {
	callerLabels := map[string]string{"architecture_resolved": "Qwen3_5MoeForConditionalGeneration"}
	profile, ok := (ProfileFactory{}).BuildModelProfile(model.ProfileRequest{
		Path: "/models/qwen",
		Model: inference.ModelIdentity{
			Architecture: "unknown",
			Labels:       callerLabels,
		},
	})
	if !ok ||
		profile.Name != "qwen" ||
		profile.Family != "qwen" ||
		profile.Architecture != "qwen3_6_moe" ||
		profile.Registry != model.ProfileRegistryName ||
		profile.Model.Path != "/models/qwen" ||
		profile.Model.Architecture != "qwen3_6_moe" ||
		profile.RouteSet.Contract != model.RouteSetContract ||
		profile.RouteSet.FeatureRoute.Architecture != "qwen3_6_moe" ||
		profile.RouteSet.LoaderRoute.Architecture != "qwen3_6_moe" ||
		profile.Labels["engine_profile_source"] != "architecture_profile" ||
		profile.Labels["engine_profile_reactive"] != "true" {
		t.Fatalf("BuildModelProfile(qwen) = %+v ok=%v, want architecture-profile-backed model profile", profile, ok)
	}

	profile.Model.Labels["architecture_resolved"] = "mutated"
	if callerLabels["architecture_resolved"] != "Qwen3_5MoeForConditionalGeneration" {
		t.Fatalf("BuildModelProfile aliased caller labels: %+v", callerLabels)
	}
	profile.RouteSet.Labels["engine_route_set_contract"] = "mutated"
	next, ok := (ProfileFactory{}).BuildModelProfile(model.ProfileRequest{
		Path: "/models/qwen",
		Model: inference.ModelIdentity{
			Architecture: "unknown",
			Labels:       callerLabels,
		},
	})
	if !ok || next.RouteSet.Labels["engine_route_set_contract"] != model.RouteSetContract {
		t.Fatalf("BuildModelProfile leaked mutable route set: %+v ok=%v", next, ok)
	}
}

func TestProfileFactory_Good_IgnoresUnknownArchitecture(t *testing.T) {
	if _, ok := (ProfileFactory{}).BuildModelProfile(model.ProfileRequest{
		Model: inference.ModelIdentity{Architecture: "totally_unknown"},
	}); ok {
		t.Fatal("BuildModelProfile(unknown) ok = true, want false")
	}
}
