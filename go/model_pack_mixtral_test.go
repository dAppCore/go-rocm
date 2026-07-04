// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"
)

func TestModelPackMixtral_MoESparsePlanLabels_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, `{
		"model_type":"MixtralForCausalLM",
		"hidden_size":4096,
		"num_hidden_layers":32,
		"num_attention_heads":32,
		"num_key_value_heads":8,
		"num_local_experts":8,
		"num_experts_per_tok":2,
		"decoder_sparse_step":2,
		"max_position_embeddings":32768,
		"quantization_config":{"bits":4,"group_size":64}
	}`))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "mixtral" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported Mixtral", inspection, inspection.Labels)
	}
	labels := inspection.Labels
	if labels["mixtral_sparse_plan"] != "metadata" ||
		labels["mixtral_local_experts"] != "8" ||
		labels["mixtral_experts_per_token"] != "2" ||
		labels["mixtral_sparse_step"] != "2" ||
		labels["mixtral_required_router_tensor"] != "model.layers.0.block_sparse_moe.gate.weight" ||
		labels["mixtral_required_expert_tensors"] != "w1,w2,w3" ||
		labels["moe_experts"] != "8" ||
		labels["moe_top_k"] != "2" ||
		labels["moe_sparse_step"] != "2" ||
		labels["moe_text_runtime"] != hipKernelStatusNotLinked ||
		labels["moe_text_decode_family"] != "mixtral" ||
		labels["moe_selected_expert_dispatch"] != hipKernelStatusNotLinked {
		t.Fatalf("labels = %+v model=%+v, want Mixtral sparse-plan/runtime labels", labels, inspection.Model)
	}
}

func TestModelPackMixtral_MoEDefaults_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, `{
		"model_type":"MixtralForCausalLM",
		"hidden_size":4096,
		"num_hidden_layers":2,
		"num_attention_heads":32,
		"num_key_value_heads":8
	}`))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "mixtral" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported Mixtral defaults", inspection, inspection.Labels)
	}
	labels := inspection.Labels
	if labels["mixtral_local_experts"] != "8" ||
		labels["mixtral_experts_per_token"] != "2" ||
		labels["mixtral_sparse_step"] != "all" ||
		labels["moe_experts"] != "8" ||
		labels["moe_top_k"] != "2" {
		t.Fatalf("labels = %+v, want Mixtral go-mlx sparse defaults", labels)
	}
}
