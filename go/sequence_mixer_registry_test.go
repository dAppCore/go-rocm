// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"testing"

	rocmmodel "dappco.re/go/rocm/model"
)

func TestRegisterSequenceMixerFamily_Good_ExtendsReactiveMixerRegistry(t *testing.T) {
	restoreRegisteredSequenceMixerFamiliesForTest(t)

	RegisterSequenceMixerFamily(SequenceMixerFamily{}, nil)
	RegisterSequenceMixerFamily(SequenceMixerFamily{Kind: "bad", State: "unknown"}, []string{"x.weight"})
	RegisterSequenceMixerFamily(SequenceMixerFamily{
		Kind:       "fake-recurrent",
		State:      SequenceMixerStateRecurrent,
		StateSlots: []string{"memory_state"},
		Source:     "test",
	}, []string{"in_proj.weight", "out_proj.weight", "in_proj.weight"})
	RegisterSequenceMixerFamily(SequenceMixerFamily{
		Kind:       "fake-recurrent",
		State:      SequenceMixerStateRecurrent,
		StateSlots: []string{"memory_state", "gate_state"},
		Source:     "test_override",
	}, []string{"in_proj.weight", "out_proj.weight", "gate.weight"})

	if got := RegisteredSequenceMixerFamilyKinds(); !slices.Equal(got, []string{"fake_recurrent"}) {
		t.Fatalf("RegisteredSequenceMixerFamilyKinds = %v, want normalized replacement under one kind", got)
	}
	registered := RegisteredSequenceMixerFamilyKinds()
	registered[0] = "mutated"
	if next := RegisteredSequenceMixerFamilyKinds(); !slices.Equal(next, []string{"fake_recurrent"}) {
		t.Fatalf("RegisteredSequenceMixerFamilyKinds returned mutable state: %v", next)
	}

	family, ok := SequenceMixerFamilyByKind("fake-recurrent")
	if !ok ||
		family.Kind != "fake_recurrent" ||
		family.State != SequenceMixerStateRecurrent ||
		family.CacheMode != SequenceMixerCacheModeRecurrent ||
		family.Source != "test_override" ||
		family.Runtime != SequenceMixerRuntimePlannedHIP ||
		!slices.Equal(family.StateSlots, []string{"memory_state", "gate_state"}) {
		t.Fatalf("SequenceMixerFamilyByKind(fake-recurrent) = %+v ok=%v, want registered recurrent family", family, ok)
	}
	leaves, ok := SequenceMixerRequiredLeaves("fake-recurrent")
	if !ok || !slices.Equal(leaves, []string{"in_proj.weight", "out_proj.weight", "gate.weight"}) {
		t.Fatalf("SequenceMixerRequiredLeaves(fake-recurrent) = %v ok=%v, want de-duped registered leaves", leaves, ok)
	}
	leaves[0] = "mutated"
	nextLeaves, _ := SequenceMixerRequiredLeaves("fake-recurrent")
	if nextLeaves[0] == "mutated" {
		t.Fatalf("SequenceMixerRequiredLeaves leaked mutable leaf slice: %v", nextLeaves)
	}

	route, ok := ROCmSequenceMixerLoaderRouteForKind("fake-recurrent")
	if !ok ||
		route.Contract != ROCmSequenceMixerLoaderRegistryContract ||
		route.Kind != "fake_recurrent" ||
		route.Loader != "fake_recurrent" ||
		route.CacheMode != SequenceMixerCacheModeRecurrent ||
		route.Source != "test_override" ||
		!route.Planned ||
		route.Labels["engine_mixer_loader_source"] != "test_override" ||
		route.Labels["engine_mixer_loader_state_slots"] != "memory_state,gate_state" ||
		route.Labels["engine_mixer_loader_required_leaves"] != "in_proj.weight,out_proj.weight,gate.weight" {
		t.Fatalf("ROCmSequenceMixerLoaderRouteForKind(fake-recurrent) = %+v ok=%v, want registered mixer route", route, ok)
	}

	families := DefaultSequenceMixerFamilies()
	if !slices.ContainsFunc(families, func(family SequenceMixerFamily) bool {
		return family.Kind == "fake_recurrent"
	}) {
		t.Fatalf("DefaultSequenceMixerFamilies missing registered family: %+v", families)
	}
	families[len(families)-1].StateSlots[0] = "mutated"
	nextFamily, ok := SequenceMixerFamilyByKind("fake-recurrent")
	if !ok || nextFamily.StateSlots[0] == "mutated" {
		t.Fatalf("DefaultSequenceMixerFamilies leaked mutable family state: %+v ok=%v", nextFamily, ok)
	}

	plan, err := BuildSequenceMixerLoadPlan(
		[]string{"fake-recurrent"},
		[]string{
			"model.layers.0.mixer.in_proj.weight",
			"model.layers.0.mixer.out_proj.weight",
			"model.layers.0.mixer.gate.weight",
		},
		1,
	)
	if err != nil {
		t.Fatalf("BuildSequenceMixerLoadPlan registered family: %v", err)
	}
	if len(plan.Layers) != 1 ||
		plan.Layers[0].Kind != "fake_recurrent" ||
		plan.Layers[0].Subpath != "mixer" ||
		!slices.Equal(plan.Layers[0].StateSlots, []string{"memory_state", "gate_state"}) ||
		len(plan.Cache.Layers) != 1 ||
		plan.Cache.Layers[0].Mode != SequenceMixerCacheModeRecurrent ||
		!slices.Equal(plan.Cache.Layers[0].StateSlots, []string{"memory_state", "gate_state"}) {
		t.Fatalf("BuildSequenceMixerLoadPlan registered family = %+v, want recurrent registered plan", plan)
	}
}

func restoreRegisteredSequenceMixerFamiliesForTest(t *testing.T) {
	t.Helper()
	registrations := rocmmodel.RegisteredSequenceMixerFamilies()

	t.Cleanup(func() {
		restore := make([]rocmmodel.SequenceMixerRegistration, len(registrations))
		for i, registration := range registrations {
			restore[i] = registration.Clone()
		}
		rocmmodel.ReplaceRegisteredSequenceMixerFamilies(restore)
	})
}
