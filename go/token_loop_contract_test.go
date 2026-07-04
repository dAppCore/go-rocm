// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"testing"

	"dappco.re/go/inference"
)

func TestROCmTokenLoopForIdentity_Good_Gemma4QATUsesRetainedIncrementalSession(t *testing.T) {
	identity := inference.ModelIdentity{
		Path:         "/data/ai/models/mlx-community/gemma-4-e2b-it-6bit",
		Architecture: "Gemma4ForCausalLM",
		QuantBits:    6,
		QuantGroup:   64,
		Labels: map[string]string{
			"engine_architecture_resolved": "gemma4_text",
			"gemma4_size":                  "E2B",
			"gemma4_quant_mode":            "q6",
		},
	}
	status, ok := ROCmTokenLoopForIdentity(identity.Path, identity)
	if !ok ||
		!status.Matched() ||
		status.Contract != ROCmTokenLoopContract ||
		status.Architecture != "gemma4_text" ||
		status.Family != "gemma4" ||
		!status.TextModel ||
		!status.TokenLoop ||
		!status.EmbedBookend ||
		!status.DecodeForward ||
		!status.LMHeadBookend ||
		!status.SharedGenerateLoop ||
		!status.IncrementalSession ||
		status.SessionState != "StateSession" ||
		!status.CloseSession ||
		!status.StepWithID ||
		!status.PerLayerInputs ||
		!status.RuntimeOwnedKV ||
		!status.DeviceKVState ||
		!status.PromptReplayRefused ||
		!status.RetainedStateRequired ||
		!status.IncrementalDecodeReady() ||
		status.FastPath != "retained-state-session" ||
		status.FallbackPath != "text-model-generate" ||
		status.Reference != "go_mlx_session_model" {
		t.Fatalf("ROCmTokenLoopForIdentity = %+v ok=%v, want retained Gemma4 token loop session contract", status, ok)
	}
	if status.Labels["engine_token_loop_contract"] != ROCmTokenLoopContract ||
		status.Labels["engine_token_loop_incremental_ready"] != "true" ||
		status.Labels["engine_token_loop_step_with_id"] != "true" ||
		status.Labels["engine_token_loop_fast_path"] != "retained-state-session" ||
		status.Labels["engine_state_context_prompt_replay_refused"] != "true" {
		t.Fatalf("token-loop labels = %+v, want token/state contract labels", status.Labels)
	}

	status.Labels["engine_token_loop_contract"] = "mutated"
	next, ok := ROCmTokenLoopForIdentity(identity.Path, identity)
	if !ok || next.Labels["engine_token_loop_contract"] != ROCmTokenLoopContract {
		t.Fatalf("ROCmTokenLoopForIdentity leaked mutable labels: %+v ok=%v", next.Labels, ok)
	}
}

func TestROCmTokenLoopForIdentity_Good_UnknownModelHasNoContract(t *testing.T) {
	if status, ok := ROCmTokenLoopForIdentity("/models/unknown", inference.ModelIdentity{Architecture: "unknown"}); ok {
		t.Fatalf("ROCmTokenLoopForIdentity unknown = %+v ok=true, want false", status)
	}
}
