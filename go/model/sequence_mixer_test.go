// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	core "dappco.re/go"
	rocmscheme "dappco.re/go/rocm/scheme"
)

type testSequenceMixerSchemeInfo struct {
	kind      string
	state     rocmscheme.StateKind
	cacheMode string
}

func (mixer testSequenceMixerSchemeInfo) Kind() string { return mixer.kind }
func (mixer testSequenceMixerSchemeInfo) State() rocmscheme.StateKind {
	return mixer.state
}
func (mixer testSequenceMixerSchemeInfo) CacheMode() string { return mixer.cacheMode }

func TestSequenceMixerFamilies_MatchReactiveRegistry_Good(t *testing.T) {
	got := SequenceMixerFamilyKinds()
	want := []string{
		"full_attention",
		"mamba2", "rwkv7",
		"gla", "retnet", "deltanet",
		"gsa", "nsa", "moba", "mla",
	}
	core.AssertEqual(t, want, got)

	for _, kind := range want {
		family, ok := SequenceMixerFamilyByKind(kind)
		if !ok {
			t.Fatalf("missing sequence mixer family %s", kind)
		}
		core.AssertEqual(t, kind, family.Kind)
		core.AssertEqual(t, SequenceMixerRuntimePlannedHIP, family.Runtime)
		if family.CacheMode == "" {
			t.Fatalf("SequenceMixerFamilyByKind(%q).CacheMode is empty", kind)
		}
		leaves, ok := SequenceMixerRequiredLeaves(kind)
		if !ok || len(leaves) == 0 {
			t.Fatalf("SequenceMixerRequiredLeaves(%q) = %v, %v; want required leaves", kind, leaves, ok)
		}
	}

	slots, ok := SequenceMixerStateSlotsForKind("gsa")
	if !ok || !slices.Equal(slots, []string{"slot_key_state", "slot_value_state"}) {
		t.Fatalf("SequenceMixerStateSlotsForKind(gsa) = %v ok=%v", slots, ok)
	}
	slots[0] = "mutated"
	nextSlots, _ := SequenceMixerStateSlotsForKind("gsa")
	if nextSlots[0] == "mutated" {
		t.Fatalf("SequenceMixerStateSlotsForKind leaked mutable slot slice: %v", nextSlots)
	}
}

func TestRegisterSequenceMixerFamily_Good_ExtendsModelRegistry(t *testing.T) {
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
	schemeMixer, ok := rocmscheme.MixerFor("fake_recurrent")
	if !ok ||
		schemeMixer.Kind() != "fake_recurrent" ||
		schemeMixer.State() != rocmscheme.StateRecurrent ||
		rocmscheme.CacheModeForMixer(schemeMixer) != rocmscheme.CacheModeRecurrent {
		t.Fatalf("scheme.MixerFor(fake_recurrent) = %+v ok=%v, want registered recurrent scheme", schemeMixer, ok)
	}
	if cache, ok := rocmscheme.CacheForMixer(schemeMixer); !ok || cache.Mode() != rocmscheme.CacheModeRecurrent {
		t.Fatalf("scheme.CacheForMixer(fake_recurrent) = %+v ok=%v, want recurrent cache scheme", cache, ok)
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
		t.Fatalf("SequenceMixerFamilyByKind(fake-recurrent) = %+v ok=%v", family, ok)
	}
	leaves, ok := SequenceMixerRequiredLeaves("fake-recurrent")
	if !ok || !slices.Equal(leaves, []string{"in_proj.weight", "out_proj.weight", "gate.weight"}) {
		t.Fatalf("SequenceMixerRequiredLeaves(fake-recurrent) = %v ok=%v", leaves, ok)
	}
	leaves[0] = "mutated"
	nextLeaves, _ := SequenceMixerRequiredLeaves("fake-recurrent")
	if nextLeaves[0] == "mutated" {
		t.Fatalf("SequenceMixerRequiredLeaves leaked mutable leaf slice: %v", nextLeaves)
	}

	route, ok := SequenceMixerLoaderRouteForKind("fake-recurrent")
	if !ok ||
		route.Contract != SequenceMixerRegistryContract ||
		route.Kind != "fake_recurrent" ||
		route.Loader != "fake_recurrent" ||
		route.CacheMode != SequenceMixerCacheModeRecurrent ||
		route.Source != "test_override" ||
		!route.Planned ||
		route.Labels["engine_mixer_loader_source"] != "test_override" ||
		route.Labels["engine_mixer_loader_state_slots"] != "memory_state,gate_state" ||
		route.Labels["engine_mixer_loader_required_leaves"] != "in_proj.weight,out_proj.weight,gate.weight" {
		t.Fatalf("SequenceMixerLoaderRouteForKind(fake-recurrent) = %+v ok=%v", route, ok)
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
		t.Fatalf("BuildSequenceMixerLoadPlan registered family = %+v", plan)
	}
}

func TestSequenceMixerCacheFactoryAPI_MatchesModes_Good(t *testing.T) {
	modes := DefaultSequenceMixerCacheFactoryModes()
	core.AssertEqual(t, []string{
		SequenceMixerCacheModeDefault,
		"fp16",
		"q8",
		"k-q8-v-q4",
		"paged",
		"fixed",
		"turboquant",
		SequenceMixerCacheModeMLALatent,
		SequenceMixerCacheModeCompaction,
		SequenceMixerCacheModeCompactionFull,
		SequenceMixerCacheModeRecurrent,
	}, modes)
	modes[0] = "mutated"
	core.AssertEqual(t, SequenceMixerCacheModeDefault, DefaultSequenceMixerCacheFactoryModes()[0])

	cache, err := BuildSequenceMixerCachePlan([]SequenceMixerLayerPlan{
		{Layer: 0, Kind: "full_attention", State: SequenceMixerStateKVCache},
		{Layer: 1, Kind: "mla", State: SequenceMixerStateKVCache},
		{Layer: 2, Kind: "mamba2", State: SequenceMixerStateRecurrent},
	})
	if err != nil {
		t.Fatalf("BuildSequenceMixerCachePlan: %v", err)
	}
	core.AssertEqual(t, SequenceMixerCachePlanContract, cache.Contract)
	core.AssertEqual(t, []SequenceMixerCacheLayerPlan{
		{Layer: 0, Kind: "full_attention", State: SequenceMixerStateKVCache, Holder: SequenceMixerStateKVCache, Mode: SequenceMixerCacheModeDefault},
		{Layer: 1, Kind: "mla", State: SequenceMixerStateKVCache, Holder: SequenceMixerStateKVCache, Mode: SequenceMixerCacheModeMLALatent},
		{Layer: 2, Kind: "mamba2", State: SequenceMixerStateRecurrent, Holder: SequenceMixerStateRecurrent, Mode: SequenceMixerCacheModeRecurrent, StateSlots: []string{"conv_state", "ssm_state"}},
	}, cache.Layers)
}

func TestSequenceMixerCacheFactoryAPI_Good_UsesSchemeRegistryOverride(t *testing.T) {
	restoreRegisteredSequenceMixerFamiliesForTest(t)

	RegisterSequenceMixerFamily(SequenceMixerFamily{
		Kind:   "fake-kv",
		State:  SequenceMixerStateKVCache,
		Source: "test",
	}, []string{"q_proj.weight"})
	rocmscheme.RegisterMixer(testSequenceMixerSchemeInfo{
		kind:      "fake_kv",
		state:     rocmscheme.StateKVCache,
		cacheMode: "q8",
	})

	family, ok := SequenceMixerFamilyByKind("fake-kv")
	if !ok || family.CacheMode != "q8" {
		t.Fatalf("SequenceMixerFamilyByKind(fake-kv) = %+v ok=%v, want q8 from scheme registry", family, ok)
	}
	mode, ok := SequenceMixerCacheModeForKind("fake-kv")
	if !ok || mode != "q8" {
		t.Fatalf("SequenceMixerCacheModeForKind(fake-kv) = %q ok=%v, want q8 from scheme registry", mode, ok)
	}
	if modes := DefaultSequenceMixerCacheFactoryModes(); !slices.Contains(modes, "q8") {
		t.Fatalf("DefaultSequenceMixerCacheFactoryModes = %v, want q8 from registered scheme override", modes)
	}

	cache, err := BuildSequenceMixerCachePlan([]SequenceMixerLayerPlan{
		{Layer: 0, Kind: "fake_kv", State: SequenceMixerStateKVCache},
	})
	if err != nil {
		t.Fatalf("BuildSequenceMixerCachePlan(fake_kv): %v", err)
	}
	core.AssertEqual(t, []SequenceMixerCacheLayerPlan{
		{Layer: 0, Kind: "fake_kv", State: SequenceMixerStateKVCache, Holder: SequenceMixerStateKVCache, Mode: "q8"},
	}, cache.Layers)
}

func TestBuildSequenceMixerLoadPlan_ReactiveConfigComposed_Good(t *testing.T) {
	plan, err := BuildSequenceMixerLoadPlan(
		[]string{"full_attention", "mamba2"},
		[]string{
			"model.layers.0.self_attn.q_proj.weight",
			"model.layers.0.self_attn.k_proj.weight",
			"model.layers.0.self_attn.v_proj.weight",
			"model.layers.0.self_attn.o_proj.weight",
			"model.layers.0.mlp.down_proj.weight",
			"model.layers.1.mixer.in_proj.weight",
			"model.layers.1.mixer.out_proj.weight",
			"model.layers.1.mixer.conv1d.weight",
			"model.layers.1.mixer.A_log",
			"model.layers.1.mlp.down_proj.weight",
		},
		2,
	)
	if err != nil {
		t.Fatalf("BuildSequenceMixerLoadPlan: %v", err)
	}
	core.AssertEqual(t, SequenceMixerRegistryContract, plan.Contract)
	core.AssertEqual(t, SequenceMixerRuntimePlannedHIP, plan.Runtime)
	core.AssertEqual(t, SequenceMixerLayerPlan{Layer: 0, Kind: "full_attention", State: SequenceMixerStateKVCache, Source: "generic_softmax", Runtime: SequenceMixerRuntimePlannedHIP, Subpath: "self_attn"}, plan.Layers[0])
	core.AssertEqual(t, SequenceMixerLayerPlan{Layer: 1, Kind: "mamba2", State: SequenceMixerStateRecurrent, StateSlots: []string{"conv_state", "ssm_state"}, Source: "fla", Runtime: SequenceMixerRuntimePlannedHIP, Subpath: "mixer"}, plan.Layers[1])
	core.AssertEqual(t, "1:mamba2:conv_state|ssm_state", SequenceMixerCachePlanSlotCSV(plan.Cache.Layers))
}

func TestSequenceMixerLoadPlanClone_Good_DeepCopiesModelContract(t *testing.T) {
	plan := SequenceMixerLoadPlan{
		Contract: SequenceMixerRegistryContract,
		Runtime:  SequenceMixerRuntimePlannedHIP,
		Layers: []SequenceMixerLayerPlan{{
			Layer:      1,
			Kind:       "mamba2",
			State:      SequenceMixerStateRecurrent,
			StateSlots: []string{"conv_state", "ssm_state"},
			Source:     "fla",
			Runtime:    SequenceMixerRuntimePlannedHIP,
			Subpath:    "mixer",
		}},
		Subpaths: SequenceMixerSubpathPlan{
			LayerCount: 2,
			Subpaths:   map[int]string{1: "mixer"},
			Ambiguous:  map[int][]string{0: {"self_attn", "attention"}},
		},
		Cache: SequenceMixerCachePlan{
			Contract: SequenceMixerCachePlanContract,
			Layers: []SequenceMixerCacheLayerPlan{{
				Layer:      1,
				Kind:       "mamba2",
				State:      SequenceMixerStateRecurrent,
				Holder:     SequenceMixerStateRecurrent,
				Mode:       SequenceMixerCacheModeRecurrent,
				StateSlots: []string{"conv_state", "ssm_state"},
			}},
		},
	}
	cloned := plan.Clone()
	cloned.Layers[0].StateSlots[0] = "mutated_layer"
	cloned.Subpaths.Subpaths[1] = "mutated_subpath"
	cloned.Subpaths.Ambiguous[0][0] = "mutated_ambiguous"
	cloned.Cache.Layers[0].StateSlots[0] = "mutated_cache"

	if plan.Layers[0].StateSlots[0] != "conv_state" ||
		plan.Subpaths.Subpaths[1] != "mixer" ||
		plan.Subpaths.Ambiguous[0][0] != "self_attn" ||
		plan.Cache.Layers[0].StateSlots[0] != "conv_state" {
		t.Fatalf("SequenceMixerLoadPlan.Clone leaked mutable contract state: %+v", plan)
	}

	ptr := CloneSequenceMixerLoadPlan(&plan)
	ptr.Layers[0].StateSlots[1] = "mutated_ptr"
	if plan.Layers[0].StateSlots[1] != "ssm_state" {
		t.Fatalf("CloneSequenceMixerLoadPlan leaked mutable layer slots: %+v", plan)
	}
	if CloneSequenceMixerLoadPlan(nil) != nil {
		t.Fatal("CloneSequenceMixerLoadPlan(nil) returned non-nil")
	}
}

func TestProbeSequenceMixerConfig_Good_ComposedHybridContract(t *testing.T) {
	probe := ProbeSequenceMixerConfig(SequenceMixerConfigInput{
		ModelType:       "composed",
		NumHiddenLayers: 2,
		LayerTypes:      []string{"full-attention", "mamba2"},
	})
	if !probe.Composed ||
		probe.LayerSource != "layer_types" ||
		probe.PlanStatus != "valid" ||
		len(probe.Layers) != 2 ||
		probe.Layers[0].Kind != "full_attention" ||
		probe.Layers[1].Kind != "mamba2" ||
		probe.Cache.Layers[0].Mode != SequenceMixerCacheModeDefault ||
		probe.Cache.Layers[1].Mode != SequenceMixerCacheModeRecurrent {
		t.Fatalf("ProbeSequenceMixerConfig(composed) = %+v, want valid composed mixer/cache plan", probe)
	}
	probe.LayerTypes[0] = "mutated"
	next := ProbeSequenceMixerConfig(SequenceMixerConfigInput{
		ModelType:       "composed",
		NumHiddenLayers: 2,
		LayerTypes:      []string{"full-attention", "mamba2"},
	})
	if next.LayerTypes[0] != "full_attention" {
		t.Fatalf("ProbeSequenceMixerConfig leaked mutable layer types: %+v", next)
	}

	uniform := ProbeSequenceMixerConfig(SequenceMixerConfigInput{
		ModelType: "mamba2",
		NumLayers: 2,
	})
	if !uniform.Composed ||
		uniform.LayerSource != "model_type" ||
		uniform.PlanStatus != "valid" ||
		len(uniform.Layers) != 2 ||
		uniform.Layers[0].Kind != "mamba2" ||
		uniform.Cache.Layers[0].Mode != SequenceMixerCacheModeRecurrent {
		t.Fatalf("ProbeSequenceMixerConfig(mamba2) = %+v, want uniform registered mixer stack", uniform)
	}

	hybrid := ProbeSequenceMixerConfig(SequenceMixerConfigInput{
		ModelType:       "hybrid",
		NumHiddenLayers: 2,
	})
	if !hybrid.Composed ||
		hybrid.LayerSource != "model_type" ||
		hybrid.PlanStatus != "invalid" ||
		hybrid.PlanError != "needs per-layer layer_types or a mixer model_type" {
		t.Fatalf("ProbeSequenceMixerConfig(hybrid) = %+v, want loud missing-layer-types refusal", hybrid)
	}
}

func restoreRegisteredSequenceMixerFamiliesForTest(t *testing.T) {
	t.Helper()
	registrations := RegisteredSequenceMixerFamilies()
	t.Cleanup(func() {
		restore := make([]SequenceMixerRegistration, len(registrations))
		for i, registration := range registrations {
			restore[i] = registration.Clone()
		}
		ReplaceRegisteredSequenceMixerFamilies(restore)
	})
}
