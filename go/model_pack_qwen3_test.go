// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"

	core "dappco.re/go"
)

func TestModelPackQwen3_QKNormSafetensorsPlan_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractQwen3SafetensorsPack(t, `{
		"model_type":"qwen3",
		"hidden_size":16,
		"num_hidden_layers":1,
		"num_attention_heads":4,
		"num_key_value_heads":2,
		"head_dim":4,
		"vocab_size":128,
		"max_position_embeddings":32768
	}`, `{
		"model.layers.0.self_attn.q_norm.weight":{"dtype":"F16","shape":[4],"data_offsets":[0,8]},
		"model.layers.0.self_attn.k_norm.weight":{"dtype":"F16","shape":[4],"data_offsets":[8,16]},
		"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[16,32]},
		"__metadata__":{"format":"pt"}
	}`, 32))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "qwen3" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported qwen3", inspection, inspection.Labels)
	}
	if inspection.Labels["qwen3_attention_qk_norm"] != "true" ||
		inspection.Labels["qwen3_qk_norm_skeleton"] != "present" ||
		inspection.Labels["qwen3_qk_norm_required_tensor_count"] != "2" ||
		inspection.Labels["qwen3_q_norm_tensor"] != "model.layers.0.self_attn.q_norm.weight" ||
		inspection.Labels["qwen3_k_norm_tensor"] != "model.layers.0.self_attn.k_norm.weight" {
		t.Fatalf("labels = %+v, want Qwen3 Q/K norm skeleton", inspection.Labels)
	}
}

func TestModelPackQwen3_QKNormSafetensorsPlan_MissingPair(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractQwen3SafetensorsPack(t, `{
		"model_type":"qwen3",
		"hidden_size":16,
		"num_hidden_layers":1,
		"num_attention_heads":4,
		"num_key_value_heads":2,
		"head_dim":4,
		"vocab_size":128,
		"max_position_embeddings":32768
	}`, `{
		"model.layers.0.self_attn.q_norm.weight":{"dtype":"F16","shape":[4],"data_offsets":[0,8]},
		"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[8,24]},
		"__metadata__":{"format":"pt"}
	}`, 24))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "qwen3" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported qwen3", inspection, inspection.Labels)
	}
	if inspection.Labels["qwen3_attention_qk_norm"] != "true" ||
		inspection.Labels["qwen3_qk_norm_skeleton"] != "missing" ||
		inspection.Labels["qwen3_qk_norm_missing_tensors"] != "model.layers.0.self_attn.k_norm.weight" {
		t.Fatalf("labels = %+v, want missing K norm tensor", inspection.Labels)
	}
}

func TestModelPackQwen3_QKNormSafetensorsPlan_InferArchitectureFromAliases(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractQwen3SafetensorsPack(t, `{
		"hidden_size":16,
		"num_hidden_layers":1,
		"num_attention_heads":4,
		"num_key_value_heads":2,
		"head_dim":4,
		"vocab_size":128,
		"max_position_embeddings":32768
	}`, `{
		"language_model.model.layers.0.self_attn.q_norm.weight":{"dtype":"F16","shape":[4],"data_offsets":[0,8]},
		"language_model.model.layers.0.self_attn.k_norm.weight":{"dtype":"F16","shape":[4],"data_offsets":[8,16]},
		"language_model.model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[16,32]},
		"__metadata__":{"format":"pt"}
	}`, 32))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "qwen3" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported inferred qwen3", inspection, inspection.Labels)
	}
	if inspection.Labels["architecture_inferred_from_weights"] != "true" ||
		inspection.Labels["architecture_inference_source"] != "dense_weight_names" ||
		inspection.Labels["qwen3_attention_qk_norm"] != "true" ||
		inspection.Labels["qwen3_qk_norm_skeleton"] != "present" {
		t.Fatalf("labels = %+v, want inferred architecture and aliased Q/K norm skeleton", inspection.Labels)
	}
	if inspection.Labels["dense_route_candidate"] != "true" ||
		inspection.Labels["dense_route_backend"] != "hip_small_decode" {
		t.Fatalf("labels = %+v, want inferred qwen3 dense route candidate", inspection.Labels)
	}
}

func nativeContractQwen3SafetensorsPack(t *testing.T, config, header string, payloadBytes int) string {
	t.Helper()
	dir := t.TempDir()
	writeNativeContractFile(t, core.PathJoin(dir, "config.json"), config)
	writeNativeContractSafetensorsHeaderWithPayload(t, core.PathJoin(dir, "model.safetensors"), header, payloadBytes)
	return dir
}
