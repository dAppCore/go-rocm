// SPDX-Licence-Identifier: EUPL-1.2

package profile

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestRegisterArchitectureProfile_Good_ExtendsReactiveProfileRegistry(t *testing.T) {
	restoreRegisteredArchitectureProfilesForTest(t)

	RegisterArchitectureProfile(ArchitectureProfile{})
	RegisterArchitectureProfile(ArchitectureProfile{
		ID:                   "fake_reactive",
		Family:               "fake",
		ParserID:             "fake-parser",
		ToolParserID:         "fake-tools",
		RuntimeStatus:        inference.FeatureRuntimeNative,
		NativeRuntime:        true,
		Generation:           true,
		Chat:                 true,
		RequiresChatTemplate: true,
		ChatTemplate:         "fake-chat",
		MoE:                  true,
		LoRATargets:          []string{"fake_q", "fake_v", "fake_router"},
		LoRADefaultTargets:   []string{"fake_q"},
		LoRATargetPaths:      map[string]string{"fake_q": "fake_attn.q", "fake_v": "fake_attn.v", "fake_router": "fake_router"},
		LoRAExtendedTargets:  []string{"fake_router"},
		QuantizationHints:    []string{"fake-q4"},
		CacheHints:           []string{"fake-state"},
		Notes:                []string{"registered extension"},
		Aliases:              []string{"FakeReactiveForCausalLM"},
	})

	if got := RegisteredArchitectureProfileIDs(); !slices.Equal(got, []string{"fake_reactive"}) {
		t.Fatalf("RegisteredArchitectureProfileIDs = %v, want fake_reactive", got)
	}
	profile, ok := LookupArchitectureProfile("FakeReactiveForCausalLM")
	if !ok || profile.ID != "fake_reactive" || profile.ParserID != "fake-parser" || profile.ToolParserID != "fake-tools" {
		t.Fatalf("LookupArchitectureProfile(alias) = %+v ok=%v, want registered profile", profile, ok)
	}
	if !KnownArchitectureProfileID("fake_reactive") ||
		!SupportedNativeArchitecture("FakeReactiveForCausalLM") ||
		!IsMoEArchitecture("fake_reactive") ||
		ArchitectureProfileFamily("fake_reactive") != "fake" ||
		ArchitectureProfileParser("fake_reactive") != "fake-parser" ||
		!ArchitectureProfileGeneration("fake_reactive") ||
		!ArchitectureProfileChat("fake_reactive") ||
		ArchitectureProfileChatTemplate("fake_reactive") != "fake-chat" {
		t.Fatalf("registered helper mismatch: %+v", profile)
	}
	if !slices.Equal(ArchitectureProfileQuantizationHints("fake_reactive"), []string{"fake-q4"}) ||
		!slices.Equal(ArchitectureProfileCacheHints("fake_reactive"), []string{"fake-state"}) ||
		!slices.Equal(ArchitectureProfileAliases("fake_reactive"), []string{"FakeReactiveForCausalLM"}) ||
		!slices.Equal(ArchitectureProfileNotes("fake_reactive"), []string{"registered extension"}) {
		t.Fatalf("registered slice helper mismatch: %+v", profile)
	}
	if !slices.Equal(DefaultLoRATargets("FakeReactiveForCausalLM"), []string{"fake_q"}) {
		t.Fatalf("DefaultLoRATargets(alias) = %v, want fake_q", DefaultLoRATargets("FakeReactiveForCausalLM"))
	}
	if path, ok := LoRATargetPath("fake_reactive", "fake_v"); !ok || path != "fake_attn.v" {
		t.Fatalf("LoRATargetPath(fake_v) = %q ok=%v, want fake_attn.v true", path, ok)
	}
	if !SafeLoRATarget("fake_reactive", "fake_q") || SafeLoRATarget("fake_reactive", "fake_router") || !LoRAExtendedTarget("fake_reactive", "fake_router") {
		t.Fatalf("registered LoRA safety helpers did not use profile policy")
	}
	if canonical, ok := LoRACanonicalTarget("fake_reactive", "layers.0.fake_q"); !ok || canonical != "layers.0.fake_attn.q" {
		t.Fatalf("LoRACanonicalTarget = %q ok=%v, want layers.0.fake_attn.q true", canonical, ok)
	}

	profiles := ArchitectureProfiles()
	if !slices.ContainsFunc(profiles, func(profile ArchitectureProfile) bool { return profile.ID == "fake_reactive" }) {
		t.Fatalf("ArchitectureProfiles missing registered profile: %+v", profiles)
	}
	profiles[len(profiles)-1].Aliases = []string{"mutated"}
	profiles[len(profiles)-1].LoRATargetPaths["fake_q"] = "mutated"
	next, ok := LookupArchitectureProfile("fake_reactive")
	if !ok || slices.Contains(next.Aliases, "mutated") || next.LoRATargetPaths["fake_q"] == "mutated" {
		t.Fatalf("LookupArchitectureProfile returned mutable registered profile: %+v ok=%v", next, ok)
	}
}

func TestRegisterArchitectureProfile_Good_ReplacesProfileAndAliasIndex(t *testing.T) {
	restoreRegisteredArchitectureProfilesForTest(t)

	RegisterArchitectureProfile(ArchitectureProfile{
		ID:      "fake_replace",
		Family:  "old",
		Aliases: []string{"OldFakeForCausalLM"},
	})
	RegisterArchitectureProfile(ArchitectureProfile{
		ID:            "fake_replace",
		Family:        "new",
		ParserID:      "new-parser",
		NativeRuntime: true,
		Aliases:       []string{"NewFakeForCausalLM"},
	})

	if got := RegisteredArchitectureProfileIDs(); !slices.Equal(got, []string{"fake_replace"}) {
		t.Fatalf("RegisteredArchitectureProfileIDs = %v, want one replacement", got)
	}
	if _, ok := LookupArchitectureProfile("OldFakeForCausalLM"); ok {
		t.Fatal("LookupArchitectureProfile(old alias) ok = true, want replacement to drop stale alias")
	}
	profile, ok := LookupArchitectureProfile("NewFakeForCausalLM")
	if !ok || profile.Family != "new" || profile.ParserID != "new-parser" || profile.ToolParserID != "new-parser" {
		t.Fatalf("LookupArchitectureProfile(new alias) = %+v ok=%v, want replacement", profile, ok)
	}
}

func TestRegisterArchitectureProfile_Good_OverridesBuiltinProfile(t *testing.T) {
	restoreRegisteredArchitectureProfilesForTest(t)

	RegisterArchitectureProfile(ArchitectureProfile{
		ID:                "qwen3",
		Family:            "qwen",
		ParserID:          "qwen-custom",
		NativeRuntime:     true,
		Generation:        true,
		Chat:              true,
		ChatTemplate:      "qwen-custom-template",
		Aliases:           []string{"Qwen3ForCausalLM"},
		CacheHints:        []string{"custom-cache"},
		QuantizationHints: []string{"custom-quant"},
	})

	profile, ok := LookupArchitectureProfile("Qwen3ForCausalLM")
	if !ok || profile.ID != "qwen3" || profile.ParserID != "qwen-custom" || profile.ChatTemplate != "qwen-custom-template" {
		t.Fatalf("LookupArchitectureProfile(qwen3 alias) = %+v ok=%v, want registered override", profile, ok)
	}
	profiles := ArchitectureProfiles()
	count := 0
	for _, profile := range profiles {
		if profile.ID == "qwen3" {
			count++
			if profile.ParserID != "qwen-custom" {
				t.Fatalf("ArchitectureProfiles qwen3 = %+v, want override", profile)
			}
		}
	}
	if count != 1 {
		t.Fatalf("ArchitectureProfiles qwen3 count = %d, want one override", count)
	}
}

func restoreRegisteredArchitectureProfilesForTest(t *testing.T) {
	t.Helper()
	order, values := registeredArchitectureProfiles.Snapshot()
	aliasOrder, aliases := registeredArchitectureProfileAliases.Snapshot()
	t.Cleanup(func() {
		registeredArchitectureProfiles.Restore(order, values)
		registeredArchitectureProfileAliases.Restore(aliasOrder, aliases)
	})
}
