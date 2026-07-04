// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

type profileFactoryTestFactory struct {
	name    string
	payload string
}

func (factory profileFactoryTestFactory) Name() string { return factory.name }

func (factory profileFactoryTestFactory) BuildModelProfile(req ProfileRequest) (Profile, bool) {
	if req.Model.Labels["profile_factory_test"] != "true" {
		return Profile{}, false
	}
	model := req.Model
	model.Path = firstNonEmpty(model.Path, req.Path)
	model.Architecture = "qwen3"
	return Profile{
		Name:         "registered-qwen",
		Family:       "qwen",
		Architecture: "qwen3",
		Model:        model,
		Labels: map[string]string{
			"factory_payload": factory.payload,
		},
	}, true
}

type blankProfileFactoryTestFactory struct{}

func (blankProfileFactoryTestFactory) Name() string { return " " }
func (blankProfileFactoryTestFactory) BuildModelProfile(ProfileRequest) (Profile, bool) {
	return Profile{Name: "blank"}, true
}

func TestRegisterProfileFactory_Good_ExtendsReactiveModelFactoryRegistry(t *testing.T) {
	restoreRegisteredProfileFactoriesForTest(t)

	RegisterProfileFactory(nil)
	RegisterProfileFactory(blankProfileFactoryTestFactory{})
	RegisterProfileFactory(profileFactoryTestFactory{name: "test-profile-factory", payload: "first"})
	RegisterProfileFactory(profileFactoryTestFactory{name: "test-profile-factory", payload: "second"})

	if got := RegisteredProfileFactoryNames(); !slices.Equal(got, []string{"test-profile-factory"}) {
		t.Fatalf("RegisteredProfileFactoryNames = %v, want replacement under one name", got)
	}
	names := RegisteredProfileFactoryNames()
	names[0] = "mutated"
	if next := RegisteredProfileFactoryNames(); !slices.Equal(next, []string{"test-profile-factory"}) {
		t.Fatalf("RegisteredProfileFactoryNames returned mutable state: %v", next)
	}

	callerLabels := map[string]string{"profile_factory_test": "true"}
	profile, ok := ResolveRegisteredProfile("/models/qwen", inference.ModelIdentity{
		Architecture: "qwen3",
		Labels:       callerLabels,
	})
	if !ok ||
		!profile.Matched() ||
		profile.Contract != ProfileFactoryRegistryContract ||
		profile.Registry != ProfileRegistryName ||
		profile.Name != "registered-qwen" ||
		profile.Family != "qwen" ||
		profile.Architecture != "qwen3" ||
		profile.Model.Path != "/models/qwen" ||
		profile.Model.Architecture != "qwen3" ||
		profile.RouteSet.Contract != RouteSetContract ||
		profile.RouteSet.Architecture != "qwen3" ||
		profile.RouteSet.Model.Path != "/models/qwen" ||
		profile.RouteSet.Labels["engine_route_set_contract"] != RouteSetContract ||
		profile.Labels["factory_payload"] != "second" ||
		profile.Labels["engine_profile_source"] != "registered_factory" ||
		profile.Labels["engine_profile_factory"] != "test-profile-factory" ||
		profile.Labels["engine_profile_reactive"] != "true" ||
		profile.Labels["engine_profile_architecture"] != "qwen3" {
		t.Fatalf("ResolveRegisteredProfile = %+v ok=%v, want registered qwen profile", profile, ok)
	}

	profile.Model.Labels["profile_factory_test"] = "mutated"
	if callerLabels["profile_factory_test"] != "true" {
		t.Fatalf("ResolveRegisteredProfile aliased caller labels: %+v", callerLabels)
	}
	profile.Labels["factory_payload"] = "mutated"
	profile.RouteSet.Labels["engine_route_set_contract"] = "mutated"
	next, ok := ResolveRegisteredProfile("/models/qwen", inference.ModelIdentity{
		Architecture: "qwen3",
		Labels:       callerLabels,
	})
	if !ok ||
		next.Labels["factory_payload"] != "second" ||
		next.RouteSet.Labels["engine_route_set_contract"] != RouteSetContract {
		t.Fatalf("ResolveRegisteredProfile leaked mutable profile labels: %+v ok=%v", next, ok)
	}
}

func restoreRegisteredProfileFactoriesForTest(t *testing.T) {
	t.Helper()
	factories := RegisteredProfileFactories()
	t.Cleanup(func() {
		ReplaceRegisteredProfileFactories(factories)
	})
}
