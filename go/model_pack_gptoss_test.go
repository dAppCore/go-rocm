// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"
)

func TestModelPackGPTOSS_MoESparseStepLabels_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, `{
		"architectures":["GptOssForCausalLM"],
		"hidden_size":2880,
		"num_hidden_layers":24,
		"num_attention_heads":64,
		"num_key_value_heads":8,
		"num_local_experts":32,
		"num_experts_per_tok":4,
		"decoder_sparse_step":3,
		"max_position_embeddings":131072,
		"quantization_config":{"quant_method":"mxfp4"}
	}`))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "gpt-oss" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported GPT-OSS", inspection, inspection.Labels)
	}
	if inspection.Model.QuantType != "mxfp4" ||
		inspection.Labels["moe_experts"] != "32" ||
		inspection.Labels["moe_top_k"] != "4" ||
		inspection.Labels["moe_sparse_step"] != "3" ||
		inspection.Labels["moe_text_runtime"] != hipKernelStatusNotLinked ||
		inspection.Labels["moe_text_decode_family"] != "gpt_oss" ||
		inspection.Labels["moe_selected_expert_dispatch"] != hipKernelStatusNotLinked {
		t.Fatalf("labels = %+v model=%+v, want GPT-OSS MoE sparse-step/runtime labels", inspection.Labels, inspection.Model)
	}
}
