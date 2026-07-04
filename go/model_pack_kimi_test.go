// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"
)

func TestModelPackKimi_MoESparseStepLabels_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, `{
		"architectures":["KimiK2ForCausalLM"],
		"hidden_size":4096,
		"num_hidden_layers":32,
		"num_attention_heads":32,
		"num_key_value_heads":8,
		"num_local_experts":384,
		"n_routed_experts":384,
		"num_experts_per_tok":8,
		"decoder_sparse_step":2,
		"max_position_embeddings":131072,
		"quantization_config":{"format":"nvfp4"}
	}`))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "kimi" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported Kimi", inspection, inspection.Labels)
	}
	if inspection.Model.QuantType != "nvfp4" ||
		inspection.Labels["moe_experts"] != "384" ||
		inspection.Labels["moe_top_k"] != "8" ||
		inspection.Labels["moe_sparse_step"] != "2" ||
		inspection.Labels["moe_text_runtime"] != hipKernelStatusNotLinked ||
		inspection.Labels["moe_text_decode_family"] != "kimi" ||
		inspection.Labels["moe_selected_expert_dispatch"] != hipKernelStatusNotLinked {
		t.Fatalf("labels = %+v model=%+v, want Kimi MoE sparse-step/runtime labels", inspection.Labels, inspection.Model)
	}
}
