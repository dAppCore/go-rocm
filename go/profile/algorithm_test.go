// SPDX-Licence-Identifier: EUPL-1.2

package profile_test

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/rocm/profile"
)

func TestAlgorithmProfile_BuiltinStatuses_Good(t *testing.T) {
	cases := []struct {
		id      inference.CapabilityID
		runtime profile.AlgorithmRuntimeStatus
		status  inference.CapabilityStatus
	}{
		{id: inference.CapabilityScheduler, runtime: profile.AlgorithmRuntimeNative, status: inference.CapabilityStatusSupported},
		{id: inference.CapabilityCacheBlocks, runtime: profile.AlgorithmRuntimeNative, status: inference.CapabilityStatusSupported},
		{id: inference.CapabilityReasoningParse, runtime: profile.AlgorithmRuntimeNative, status: inference.CapabilityStatusSupported},
		{id: inference.CapabilityJANGTQ, runtime: profile.AlgorithmRuntimeMetadataOnly, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilityCodebookVQ, runtime: profile.AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilityQuantization, runtime: profile.AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilityEmbeddings, runtime: profile.AlgorithmRuntimeMetadataOnly, status: inference.CapabilityStatusPlanned},
		{id: inference.CapabilityMoERouting, runtime: profile.AlgorithmRuntimeMetadataOnly, status: inference.CapabilityStatusPlanned},
		{id: inference.CapabilityMoELazyExperts, runtime: profile.AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilitySpeculativeDecode, runtime: profile.AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilityPromptLookupDecode, runtime: profile.AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
	}

	for _, tc := range cases {
		t.Run(string(tc.id), func(t *testing.T) {
			p, ok := profile.LookupAlgorithmProfile(tc.id)
			if !ok {
				t.Fatalf("LookupAlgorithmProfile(%q) ok = false", tc.id)
			}
			if p.RuntimeStatus != tc.runtime || p.CapabilityStatus != tc.status {
				t.Fatalf("profile = %+v, want runtime/status %q/%q", p, tc.runtime, tc.status)
			}
			if p.Group == "" || p.Detail == "" {
				t.Fatalf("profile = %+v, want group and detail", p)
			}
		})
	}
}

func TestAlgorithmProfile_AutoRoundQuantization_Good(t *testing.T) {
	p, ok := profile.LookupAlgorithmProfile(inference.CapabilityQuantization)
	if !ok {
		t.Fatal("missing quantization profile")
	}
	if p.Algorithm != "auto-round" || p.RuntimeStatus != profile.AlgorithmRuntimeExperimental {
		t.Fatalf("quantization profile = %+v, want auto-round experimental", p)
	}
	for _, want := range []string{
		"quantization.profile.auto-round",
		"quantization.profile.auto-round-best",
		"quantization.profile.auto-round-light",
		"weight_rounding.signround",
		"model_pack.inspect_autoround",
		"autoround.calibration.plan",
		"autoround.calibration.evidence",
		"autoround.calibration.decision",
		"hip.autoround_quantize.launch_args",
		"hip.autoround_quantize.kernel",
	} {
		if !slices.Contains(p.Provides, want) {
			t.Fatalf("quantization provides = %+v, want %q", p.Provides, want)
		}
	}
}

func TestAlgorithmProfile_CapabilityListHasNoDuplicateIDs_Good(t *testing.T) {
	capabilities := profile.AlgorithmCapabilities()
	seen := map[inference.CapabilityID]bool{}
	for _, capability := range capabilities {
		if seen[capability.ID] {
			t.Fatalf("duplicate algorithm capability %q", capability.ID)
		}
		seen[capability.ID] = true
		if capability.Labels["runtime_status"] == "" {
			t.Fatalf("capability = %+v, want runtime_status label", capability)
		}
	}
	for _, id := range []inference.CapabilityID{
		inference.CapabilitySpeculativeDecode,
		inference.CapabilityPromptLookupDecode,
		inference.CapabilityEmbeddings,
		inference.CapabilityRerank,
		inference.CapabilityMoERouting,
		inference.CapabilityMoELazyExperts,
		inference.CapabilityCodebookVQ,
		inference.CapabilityQuantization,
	} {
		if !seen[id] {
			t.Fatalf("missing algorithm capability %q", id)
		}
	}
}

func TestAlgorithmProfile_BuiltinProfilesAreCloned_Good(t *testing.T) {
	profiles := profile.BuiltinAlgorithmProfiles()
	if len(profiles) == 0 {
		t.Fatal("BuiltinAlgorithmProfiles() returned no profiles")
	}
	for index, p := range profiles {
		if p.ID == inference.CapabilityQuantization {
			profiles[index].Provides[0] = "mutated"
			profiles[index].Notes[0] = "mutated"
			break
		}
	}
	again, ok := profile.LookupAlgorithmProfile(inference.CapabilityQuantization)
	if !ok {
		t.Fatal("LookupAlgorithmProfile(CapabilityQuantization) ok = false")
	}
	if again.Provides[0] == "mutated" || again.Notes[0] == "mutated" {
		t.Fatalf("algorithm profiles alias caller mutation: %+v", again)
	}
	again.Provides[0] = "mutated"
	again, ok = profile.LookupAlgorithmProfile(inference.CapabilityQuantization)
	if !ok || again.Provides[0] == "mutated" {
		t.Fatalf("LookupAlgorithmProfile returned mutable backing slices: %+v ok=%v", again, ok)
	}
	if _, ok := profile.LookupAlgorithmProfile("missing-capability"); ok {
		t.Fatal("LookupAlgorithmProfile(missing) ok = true")
	}
}
