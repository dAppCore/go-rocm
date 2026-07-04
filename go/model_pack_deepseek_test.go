// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"
)

func TestModelPackDeepSeek_MLAPlan_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, `{
		"model_type":"DeepseekV3ForCausalLM",
		"hidden_size":1024,
		"num_hidden_layers":2,
		"num_attention_heads":8,
		"num_key_value_heads":2,
		"q_lora_rank":1536,
		"kv_lora_rank":512,
		"qk_nope_head_dim":128,
		"qk_rope_head_dim":64,
		"v_head_dim":128
	}`))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "deepseek" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported DeepSeek", inspection, inspection.Labels)
	}
	if inspection.Labels["deepseek_mla"] != "true" ||
		inspection.Labels["deepseek_mla_valid"] != "true" ||
		inspection.Labels["deepseek_q_lora_rank"] != "1536" ||
		inspection.Labels["deepseek_kv_lora_rank"] != "512" ||
		inspection.Labels["deepseek_qk_nope_head_dim"] != "128" ||
		inspection.Labels["deepseek_qk_rope_head_dim"] != "64" ||
		inspection.Labels["deepseek_qk_head_dim"] != "192" ||
		inspection.Labels["deepseek_v_head_dim"] != "128" {
		t.Fatalf("labels = %+v, want validated DeepSeek MLA plan", inspection.Labels)
	}
}

func TestModelPackDeepSeek_MLAPlanInvalid_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, `{
		"model_type":"DeepseekV3ForCausalLM",
		"hidden_size":1024,
		"num_hidden_layers":2,
		"num_attention_heads":8,
		"num_key_value_heads":2,
		"kv_lora_rank":512,
		"qk_nope_head_dim":128,
		"qk_rope_head_dim":64,
		"qk_head_dim":256,
		"v_head_dim":128
	}`))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "deepseek" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported DeepSeek with invalid MLA metadata labelled", inspection, inspection.Labels)
	}
	if inspection.Labels["deepseek_mla"] != "true" ||
		inspection.Labels["deepseek_mla_valid"] != "false" ||
		inspection.Labels["deepseek_qk_head_dim"] != "256" {
		t.Fatalf("labels = %+v, want invalid DeepSeek MLA plan marker", inspection.Labels)
	}
}
