// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"testing"

	core "dappco.re/go"
)

func TestSequenceMixerFamilies_MatchGoMLXReactiveRegistry_Good(t *testing.T) {
	got := sequenceMixerRegisteredKinds()
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
			t.Fatalf("SequenceMixerFamilyByKind(%q).CacheMode is empty, want go-mlx cache factory mode", kind)
		}
		leaves, ok := SequenceMixerRequiredLeaves(kind)
		if !ok || len(leaves) == 0 {
			t.Fatalf("SequenceMixerRequiredLeaves(%q) = %v, %v; want go-mlx loader required leaves", kind, leaves, ok)
		}
	}
	expectedStates := map[string]string{
		"full_attention": SequenceMixerStateKVCache,
		"mamba2":         SequenceMixerStateRecurrent,
		"rwkv7":          SequenceMixerStateRecurrent,
		"gla":            SequenceMixerStateRecurrent,
		"retnet":         SequenceMixerStateRecurrent,
		"deltanet":       SequenceMixerStateRecurrent,
		"gsa":            SequenceMixerStateRecurrent,
		"nsa":            SequenceMixerStateKVCache,
		"moba":           SequenceMixerStateKVCache,
		"mla":            SequenceMixerStateKVCache,
	}
	for kind, state := range expectedStates {
		family, ok := SequenceMixerFamilyByKind(kind)
		if !ok {
			t.Fatalf("missing sequence mixer family %s", kind)
		}
		core.AssertEqual(t, state, family.State)
	}
	expectedCacheModes := map[string]string{
		"full_attention": SequenceMixerCacheModeDefault,
		"mamba2":         SequenceMixerCacheModeRecurrent,
		"rwkv7":          SequenceMixerCacheModeRecurrent,
		"gla":            SequenceMixerCacheModeRecurrent,
		"retnet":         SequenceMixerCacheModeRecurrent,
		"deltanet":       SequenceMixerCacheModeRecurrent,
		"gsa":            SequenceMixerCacheModeRecurrent,
		"nsa":            SequenceMixerCacheModeDefault,
		"moba":           SequenceMixerCacheModeDefault,
		"mla":            SequenceMixerCacheModeMLALatent,
	}
	for kind, mode := range expectedCacheModes {
		family, ok := SequenceMixerFamilyByKind(kind)
		if !ok {
			t.Fatalf("missing sequence mixer family %s", kind)
		}
		core.AssertEqual(t, mode, family.CacheMode)
	}
	expectedStateSlots := map[string][]string{
		"full_attention": nil,
		"mamba2":         {"conv_state", "ssm_state"},
		"rwkv7":          {"wkv_state"},
		"gla":            {"gated_linear_state"},
		"retnet":         {"retention_state"},
		"deltanet":       {"value_memory_state"},
		"gsa":            {"slot_key_state", "slot_value_state"},
		"nsa":            nil,
		"moba":           nil,
		"mla":            nil,
	}
	for kind, slots := range expectedStateSlots {
		got, ok := SequenceMixerStateSlotsForKind(kind)
		if !ok {
			t.Fatalf("SequenceMixerStateSlotsForKind(%q) ok = false", kind)
		}
		core.AssertEqual(t, slots, got)
		if len(got) > 0 {
			got[0] = "mutated"
			next, _ := SequenceMixerStateSlotsForKind(kind)
			if next[0] == "mutated" {
				t.Fatalf("SequenceMixerStateSlotsForKind(%q) leaked mutable slot slice: %v", kind, next)
			}
		}
	}
	if _, ok := SequenceMixerFamilyByKind("unknown_mixer"); ok {
		t.Fatal("SequenceMixerFamilyByKind(unknown_mixer) ok = true")
	}
}

func TestSequenceMixerCacheFactoryAPI_MatchesGoMLXModes_Good(t *testing.T) {
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

	mode, ok := SequenceMixerCacheModeForKind("mla")
	core.AssertEqual(t, true, ok)
	core.AssertEqual(t, SequenceMixerCacheModeMLALatent, mode)
	mode, ok = SequenceMixerCacheModeForKind("full-attention")
	core.AssertEqual(t, true, ok)
	core.AssertEqual(t, SequenceMixerCacheModeDefault, mode)
	mode, ok = SequenceMixerCacheModeForKind("mamba2")
	core.AssertEqual(t, true, ok)
	core.AssertEqual(t, SequenceMixerCacheModeRecurrent, mode)
	if mode, ok = SequenceMixerCacheModeForKind("no_such_mixer"); ok || mode != "" {
		t.Fatalf("SequenceMixerCacheModeForKind(no_such_mixer) = %q, %v; want empty false", mode, ok)
	}

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

func TestDiscoverSequenceMixerSubpaths_RawModelKeysOnly_Good(t *testing.T) {
	plan := DiscoverSequenceMixerSubpaths([]string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
		"model.layers.1.mixer.in_proj.weight",
		"model.layers.1.mlp.down_proj.weight",
	}, 2)

	core.AssertEqual(t, 2, plan.LayerCount)
	core.AssertEqual(t, map[int]string{0: "self_attn", 1: "mixer"}, plan.Subpaths)
	core.AssertEqual(t, 0, len(plan.Ambiguous))
}

func TestDiscoverSequenceMixerSubpaths_AliasKeysMissRawScan_Good(t *testing.T) {
	plan := DiscoverSequenceMixerSubpaths([]string{
		"language_model.model.layers.0.self_attn.q_proj.weight",
		"language_model.model.layers.0.self_attn.k_proj.weight",
		"language_model.model.layers.0.mlp.down_proj.weight",
	}, 1)

	core.AssertEqual(t, 1, plan.LayerCount)
	core.AssertEqual(t, 0, len(plan.Subpaths))
	core.AssertEqual(t, 0, len(plan.Ambiguous))
}

func TestDiscoverSequenceMixerSubpaths_Ambiguous_Bad(t *testing.T) {
	plan := DiscoverSequenceMixerSubpaths([]string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.mixer.in_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
	}, 1)

	core.AssertEqual(t, []string{"mixer", "self_attn"}, plan.Ambiguous[0])
	if plan.Subpaths[0] != "" {
		t.Fatalf("Subpaths[0] = %q, want no random winner for ambiguous layer", plan.Subpaths[0])
	}
}

func TestDiscoverSequenceMixerSubpaths_IgnoresFeedForwardOwners_Good(t *testing.T) {
	plan := DiscoverSequenceMixerSubpaths([]string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.block_sparse_moe.gate.weight",
		"model.layers.0.block_sparse_moe.experts.0.up_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
	}, 1)

	core.AssertEqual(t, map[int]string{0: "self_attn"}, plan.Subpaths)
	core.AssertEqual(t, 0, len(plan.Ambiguous))
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
	core.AssertEqual(t, 2, len(plan.Layers))
	core.AssertEqual(t, SequenceMixerLayerPlan{Layer: 0, Kind: "full_attention", State: SequenceMixerStateKVCache, Source: "generic_softmax", Runtime: SequenceMixerRuntimePlannedHIP, Subpath: "self_attn"}, plan.Layers[0])
	core.AssertEqual(t, SequenceMixerLayerPlan{Layer: 1, Kind: "mamba2", State: SequenceMixerStateRecurrent, StateSlots: []string{"conv_state", "ssm_state"}, Source: "fla", Runtime: SequenceMixerRuntimePlannedHIP, Subpath: "mixer"}, plan.Layers[1])
	core.AssertEqual(t, SequenceMixerCachePlanContract, plan.Cache.Contract)
	core.AssertEqual(t, []SequenceMixerCacheLayerPlan{
		{Layer: 0, Kind: "full_attention", State: SequenceMixerStateKVCache, Holder: SequenceMixerStateKVCache, Mode: SequenceMixerCacheModeDefault},
		{Layer: 1, Kind: "mamba2", State: SequenceMixerStateRecurrent, Holder: SequenceMixerStateRecurrent, Mode: SequenceMixerCacheModeRecurrent, StateSlots: []string{"conv_state", "ssm_state"}},
	}, plan.Cache.Layers)
	core.AssertEqual(t, "1:mamba2:conv_state|ssm_state", sequenceMixerCachePlanSlotCSV(plan.Cache.Layers))
}

func TestBuildSequenceMixerLoadPlan_RecurrentStateSlots_Good(t *testing.T) {
	plan, err := BuildSequenceMixerLoadPlan(
		[]string{"rwkv7", "gsa"},
		[]string{
			"model.layers.0.mixer.receptance.weight",
			"model.layers.0.mixer.key.weight",
			"model.layers.0.mixer.value.weight",
			"model.layers.0.mixer.output.weight",
			"model.layers.0.mixer.decay.weight",
			"model.layers.0.mixer.a_proj.weight",
			"model.layers.0.mixer.b_proj.weight",
			"model.layers.1.mixer.q_proj.weight",
			"model.layers.1.mixer.k_proj.weight",
			"model.layers.1.mixer.v_proj.weight",
			"model.layers.1.mixer.f_proj.weight",
			"model.layers.1.mixer.g_proj.weight",
			"model.layers.1.mixer.o_proj.weight",
		},
		2,
	)
	if err != nil {
		t.Fatalf("BuildSequenceMixerLoadPlan(rwkv7,gsa): %v", err)
	}
	core.AssertEqual(t, []string{"wkv_state"}, plan.Layers[0].StateSlots)
	core.AssertEqual(t, []string{"slot_key_state", "slot_value_state"}, plan.Layers[1].StateSlots)
	core.AssertEqual(t, []string{"wkv_state"}, plan.Cache.Layers[0].StateSlots)
	core.AssertEqual(t, []string{"slot_key_state", "slot_value_state"}, plan.Cache.Layers[1].StateSlots)
	core.AssertEqual(t, "0:rwkv7:wkv_state,1:gsa:slot_key_state|slot_value_state", sequenceMixerCachePlanSlotCSV(plan.Cache.Layers))
}

func TestBuildSequenceMixerLoadPlan_MLALatentCacheFactoryMode_Good(t *testing.T) {
	plan, err := BuildSequenceMixerLoadPlan(
		[]string{"mla"},
		[]string{
			"model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
			"model.layers.0.self_attn.kv_b_proj.weight",
			"model.layers.0.self_attn.q_a_proj.weight",
			"model.layers.0.self_attn.q_b_proj.weight",
			"model.layers.0.self_attn.o_proj.weight",
		},
		1,
	)
	if err != nil {
		t.Fatalf("BuildSequenceMixerLoadPlan(mla): %v", err)
	}
	core.AssertEqual(t, 1, len(plan.Cache.Layers))
	core.AssertEqual(t, SequenceMixerStateKVCache, plan.Cache.Layers[0].Holder)
	core.AssertEqual(t, SequenceMixerCacheModeMLALatent, plan.Cache.Layers[0].Mode)
	core.AssertEqual(t, "0:mla:kv-cache:mla-latent", sequenceMixerCachePlanCSV(plan.Cache.Layers))
}

func TestBuildSequenceMixerLoadPlan_MismatchAndUnregistered_Bad(t *testing.T) {
	if _, err := BuildSequenceMixerLoadPlan([]string{"mamba2"}, nil, 2); err == nil {
		t.Fatal("expected layer_types length mismatch error")
	}
	if _, err := BuildSequenceMixerLoadPlan([]string{"full_attention", "no_such_mixer"}, nil, 2); err == nil {
		t.Fatal("expected unregistered mixer kind error")
	}
}

func TestBuildSequenceMixerLoadPlan_MissingRequiredLeaves_Bad(t *testing.T) {
	_, err := BuildSequenceMixerLoadPlan(
		[]string{"full_attention", "mamba2"},
		[]string{
			"model.layers.0.self_attn.q_proj.weight",
			"model.layers.0.self_attn.k_proj.weight",
			"model.layers.0.self_attn.v_proj.weight",
			"model.layers.0.self_attn.o_proj.weight",
			"model.layers.1.mixer.in_proj.weight",
			"model.layers.1.mixer.out_proj.weight",
			"model.layers.1.mixer.conv1d.weight",
		},
		2,
	)
	if err == nil {
		t.Fatal("expected missing required mamba2 leaf error")
	}
	core.AssertContains(t, err.Error(), "layer 1 mamba2 missing required mixer tensors A_log")
}
