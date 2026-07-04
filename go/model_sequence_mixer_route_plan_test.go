// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"testing"

	"dappco.re/go/inference"
)

func TestROCmModelRoutePlan_Good_CarriesSequenceMixerRoutes(t *testing.T) {
	plan, ok := ROCmModelRoutePlanForIdentity("/models/mamba", inference.ModelIdentity{
		Architecture: "Mamba2ForCausalLM",
	})
	if !ok ||
		!plan.Matched() ||
		len(plan.SequenceMixerRoutes) != 1 ||
		plan.SequenceMixerRoutes[0].Kind != "mamba2" ||
		plan.SequenceMixerRoutes[0].State != SequenceMixerStateRecurrent ||
		plan.SequenceMixerRoutes[0].CacheMode != SequenceMixerCacheModeRecurrent ||
		plan.Labels["engine_route_plan_sequence_mixer"] != "true" ||
		plan.Labels["engine_route_plan_sequence_mixer_count"] != "1" ||
		plan.Labels["engine_route_plan_sequence_mixer_kinds"] != "mamba2" ||
		plan.Labels["engine_route_plan_sequence_mixer_cache_modes"] != SequenceMixerCacheModeRecurrent ||
		plan.Labels["engine_route_plan_sequence_mixer_states"] != SequenceMixerStateRecurrent ||
		plan.Labels["engine_route_plan_sequence_mixer_planned"] != "true" ||
		plan.Labels["engine_mixer_loader_kind"] != "mamba2" ||
		plan.Labels["engine_mixer_loader_state_slots"] != "conv_state,ssm_state" {
		t.Fatalf("ROCmModelRoutePlanForIdentity(mamba2) = %+v ok=%v, want sequence mixer route", plan, ok)
	}

	plan.SequenceMixerRoutes[0].Kind = "mutated"
	plan.SequenceMixerRoutes[0].StateSlots[0] = "mutated"
	plan.SequenceMixerRoutes[0].Labels["engine_mixer_loader_kind"] = "mutated"
	next, ok := ROCmModelRoutePlanForIdentity("/models/mamba", inference.ModelIdentity{
		Architecture: "Mamba2ForCausalLM",
	})
	if !ok ||
		len(next.SequenceMixerRoutes) != 1 ||
		next.SequenceMixerRoutes[0].Kind != "mamba2" ||
		next.SequenceMixerRoutes[0].StateSlots[0] != "conv_state" ||
		next.SequenceMixerRoutes[0].Labels["engine_mixer_loader_kind"] != "mamba2" {
		t.Fatalf("ROCmModelRoutePlanForIdentity leaked mutable sequence mixer route: %+v ok=%v", next.SequenceMixerRoutes, ok)
	}
}
