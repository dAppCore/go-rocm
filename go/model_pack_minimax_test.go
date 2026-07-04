// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestModelPackMiniMaxM2_ReadsJANGTQSidecars_Good(t *testing.T) {
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type":"MiniMaxM2ForCausalLM",
		"hidden_size":2048,
		"intermediate_size":8192,
		"num_hidden_layers":24,
		"vocab_size":32000,
		"max_position_embeddings":32768,
		"num_local_experts":32,
		"num_experts_per_tok":2,
		"use_routing_bias":true,
		"quantization_config":{"quant_method":"jangtq","bits":2,"group_size":64,"weight_format":"mxtq"}
	}`)
	writeModelPackMiniMaxM2Layer0Safetensors(t, core.PathJoin(dir, "model.safetensors"))
	writeNativeContractFile(t, core.PathJoin(dir, "jang_config.json"), `{
		"version":1,
		"weight_format":"mxtq",
		"profile":"JANGTQ",
		"source_model":{"name":"MiniMax-M2.7","org":"dealignai","architecture":"MiniMaxM2ForCausalLM"},
		"mxtq_bits":{"attention":8,"shared_expert":4,"routed_expert":2,"embed_tokens":8,"lm_head":8},
		"quantization":{"method":"affine+mxtq","group_size":64,"bits_default":2},
		"capabilities":{"reasoning_parser":"minimax","tool_parser":"json","supports_tools":true,"supports_thinking":true,"cache_type":"block-prefix"}
	}`)
	writeNativeContractFile(t, core.PathJoin(dir, "codebook_config.json"), `{
		"type":"codebook",
		"format":"vq",
		"codebook_size":16,
		"code_dim":2,
		"index_bits":8,
		"tensors":[{"name":"model.layers.0.mlp.down_proj.weight","shape":[2,4],"codes":"codes","codebook":"table"}]
	}`)

	backend := newROCmBackendWithRuntime(&fakeNativeRuntime{device: nativeDeviceInfo{MemoryBytes: 16 * memoryGiB, Name: "gfx1100"}})
	inspection, err := backend.InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !inspection.Supported || inspection.Format != "safetensors" || inspection.Model.Architecture != "minimax_m2" || inspection.Model.QuantType != "jangtq" {
		t.Fatalf("inspection = %+v, want supported MiniMax/JANGTQ safetensors pack", inspection)
	}
	if inspection.Labels["memory_plan_machine_class"] != "rocm-16gb" ||
		inspection.Labels["memory_plan_moe_lazy_experts"] != "true" ||
		inspection.Labels["codebook_format"] != "vq" {
		t.Fatalf("inspection labels = %+v, want memory fit and codebook metadata", inspection.Labels)
	}
	if inspection.Labels["minimax_m2_sparse_plan"] != "staged_metadata" ||
		inspection.Labels["minimax_m2_intermediate_size"] != "8192" ||
		inspection.Labels["minimax_m2_local_experts"] != "32" ||
		inspection.Labels["minimax_m2_experts_per_token"] != "2" ||
		inspection.Labels["minimax_m2_routing_bias"] != "true" ||
		inspection.Labels["minimax_m2_layer0_skeleton"] != "present" ||
		inspection.Labels["minimax_m2_layer0_required_tensor_count"] != "9" ||
		inspection.Labels["minimax_m2_required_router_tensor"] != "model.layers.0.block_sparse_moe.gate.weight" ||
		inspection.Labels["minimax_m2_required_router_bias_tensor"] != "model.layers.0.block_sparse_moe.e_score_correction_bias" ||
		inspection.Labels["minimax_m2_required_expert_tensors"] != "gate_proj,up_proj,down_proj" {
		t.Fatalf("inspection labels = %+v, want MiniMax staged sparse-plan labels", inspection.Labels)
	}
	if inspection.Labels["jang_source_name"] != "MiniMax-M2.7" ||
		inspection.Labels["jang_source_org"] != "dealignai" ||
		inspection.Labels["jang_source_architecture"] != "minimax_m2" ||
		inspection.Labels["jang_attention_bits"] != "8" ||
		inspection.Labels["jang_shared_expert_bits"] != "4" ||
		inspection.Labels["jang_routed_expert_bits"] != "2" ||
		inspection.Labels["jang_embed_tokens_bits"] != "8" ||
		inspection.Labels["jang_lm_head_bits"] != "8" {
		t.Fatalf("inspection labels = %+v, want MiniMax/JANGTQ source and role-bit labels", inspection.Labels)
	}
	metadataFixtures := nativeContractMetadataFixtureKernels()
	for _, id := range []inference.CapabilityID{inference.CapabilityJANGTQ, inference.CapabilityCodebookVQ, inference.CapabilityMoERouting, inference.CapabilityMoELazyExperts} {
		capability, ok := nativeInspectionCapability(inspection, id)
		if !ok || capability.Status != inference.CapabilityStatusExperimental ||
			capability.Labels["runtime_status"] != string(inference.FeatureRuntimeExperimental) ||
			capability.Labels["fixture_kernel"] != hipKernelStatusLinked ||
			capability.Labels["fixture_kernel_name"] != metadataFixtures[id] ||
			capability.Labels["required_integration"] == "" ||
			capability.Labels["production_integration"] != "pending" {
			t.Fatalf("inspection capability %s = %+v ok=%v, want linked fixture experimental", id, capability, ok)
		}
	}
}

func writeModelPackMiniMaxM2Layer0Safetensors(t *testing.T, path string) {
	t.Helper()
	names := []string{
		"model.layers.0.self_attn.qkv_proj.weight",
		"model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.block_sparse_moe.gate.weight",
		"model.layers.0.block_sparse_moe.e_score_correction_bias",
		"model.layers.0.block_sparse_moe.experts.0.gate_proj.weight",
		"model.layers.0.block_sparse_moe.experts.0.up_proj.weight",
		"model.layers.0.block_sparse_moe.experts.0.down_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
	}
	header := "{"
	offset := 0
	for i, name := range names {
		if i > 0 {
			header += ","
		}
		next := offset + 2
		header += core.Sprintf("%q:{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[%d,%d]}", name, offset, next)
		offset = next
	}
	header += `,"__metadata__":{"format":"pt"}}`
	writeNativeContractSafetensorsHeaderWithPayload(t, path, header, offset)
}
