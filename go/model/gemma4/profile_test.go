// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/rocm/model"
)

func TestProfileFactory_Good_SelfRegistersGemma4Profile(t *testing.T) {
	if names := model.RegisteredProfileFactoryNames(); !slices.Contains(names, "gemma4") {
		t.Fatalf("RegisteredProfileFactoryNames = %v, want Gemma4 model package self-registration", names)
	}

	callerLabels := map[string]string{"engine_architecture_resolved": "gemma4_text"}
	profile, ok := model.ResolveRegisteredProfile("/models/gemma4-e2b-q6", inference.ModelIdentity{
		Architecture: "Gemma4ForCausalLM",
		QuantBits:    6,
		QuantGroup:   64,
		Labels:       callerLabels,
	})
	if !ok ||
		profile.Name != "gemma4" ||
		profile.Family != "gemma4" ||
		profile.Architecture != "gemma4_text" ||
		profile.Registry != model.ProfileRegistryName ||
		profile.Model.Path != "/models/gemma4-e2b-q6" ||
		profile.Model.Architecture != "gemma4_text" ||
		profile.RouteSet.Contract != model.RouteSetContract ||
		profile.RouteSet.FeatureRoute.Architecture != "gemma4_text" ||
		profile.RouteSet.LoaderRoute.Loader != "gemma4_text" ||
		profile.RouteSet.TokenizerRoute.Loader != "hf-tokenizer-json" ||
		profile.Labels["engine_profile_source"] != "model_config" ||
		profile.Labels["engine_profile_reactive"] != "true" {
		t.Fatalf("ResolveRegisteredProfile(Gemma4) = %+v ok=%v, want model-owned Gemma4 profile", profile, ok)
	}

	profile.Model.Labels["engine_architecture_resolved"] = "mutated"
	if callerLabels["engine_architecture_resolved"] != "gemma4_text" {
		t.Fatalf("ResolveRegisteredProfile aliased caller labels: %+v", callerLabels)
	}
	profile.RouteSet.Labels["engine_route_set_contract"] = "mutated"
	next, ok := model.ResolveRegisteredProfile("/models/gemma4-e2b-q6", inference.ModelIdentity{
		Architecture: "Gemma4ForCausalLM",
		QuantBits:    6,
		QuantGroup:   64,
		Labels:       callerLabels,
	})
	if !ok || next.RouteSet.Labels["engine_route_set_contract"] != model.RouteSetContract {
		t.Fatalf("ResolveRegisteredProfile leaked mutable Gemma4 route set: %+v ok=%v", next, ok)
	}
}

func TestTokenizerRoute_Good_SelfRegistersThoughtChannel(t *testing.T) {
	route, ok := model.TokenizerRouteForArchitecture("Gemma4TextForCausalLM")
	if !ok ||
		route.Architecture != "gemma4_text" ||
		!route.ThinkingChannel ||
		route.ThinkingChannelOpen != ThinkingChannelOpenMarker ||
		route.ThinkingChannelClose != ThinkingChannelCloseMarker ||
		route.Labels["engine_tokenizer_thinking_channel"] != "true" ||
		route.Labels["engine_tokenizer_thinking_channel_open"] != ThinkingChannelOpenMarker ||
		route.Labels["engine_tokenizer_thinking_channel_close"] != ThinkingChannelCloseMarker {
		t.Fatalf("TokenizerRouteForArchitecture(Gemma4Text) = %+v ok=%v, want Gemma4 thought-channel metadata", route, ok)
	}
}

func TestProfileFactory_Good_IgnoresNonGemma4Identity(t *testing.T) {
	if _, ok := (ProfileFactory{}).BuildModelProfile(model.ProfileRequest{
		Model: inference.ModelIdentity{Architecture: "Qwen3ForCausalLM"},
	}); ok {
		t.Fatal("BuildModelProfile(qwen3) ok = true, want Gemma4-only factory")
	}
}
