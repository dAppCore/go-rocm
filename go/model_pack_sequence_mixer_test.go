// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"strings"
	"testing"
)

func TestModelPackSequenceMixer_FLAPlanAndSubpathDiscovery_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractQwen3SafetensorsPack(t, `{
		"architectures":["Qwen3NextForCausalLM"],
		"hidden_size":4,
		"num_hidden_layers":2,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":4,
		"vocab_size":16,
		"max_position_embeddings":32768,
		"layer_types":["full_attention","mamba2"]
	}`, `{
		"model.layers.0.self_attn.q_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[0,16]},
		"model.layers.0.self_attn.k_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[16,32]},
		"model.layers.0.self_attn.v_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[32,48]},
		"model.layers.0.self_attn.o_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[48,64]},
		"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[64,80]},
		"model.layers.1.mixer.in_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[80,112]},
		"model.layers.1.mixer.out_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[112,144]},
		"model.layers.1.mixer.conv1d.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[144,176]},
		"model.layers.1.mixer.A_log":{"dtype":"F16","shape":[4,4],"data_offsets":[176,208]},
		"model.layers.1.mlp.down_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[208,224]},
		"__metadata__":{"format":"pt"}
	}`, 224))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "qwen3_next" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported qwen3_next sequence mixer plan", inspection, inspection.Labels)
	}
	if inspection.Labels["sequence_mixer_registry_contract"] != SequenceMixerRegistryContract ||
		inspection.Labels["sequence_mixer_registry_kinds"] != "full_attention,mamba2,rwkv7,gla,retnet,deltanet,gsa,nsa,moba,mla" ||
		inspection.Labels["sequence_mixer_state_contract"] != SequenceMixerStateContract ||
		inspection.Labels["sequence_mixer_registered_states"] != "full_attention:kv-cache,mamba2:recurrent,rwkv7:recurrent,gla:recurrent,retnet:recurrent,deltanet:recurrent,gsa:recurrent,nsa:kv-cache,moba:kv-cache,mla:kv-cache" ||
		inspection.Labels["sequence_mixer_state_slots_contract"] != SequenceMixerStateSlotsContract ||
		inspection.Labels["sequence_mixer_registered_state_slots"] != "mamba2:conv_state|ssm_state,rwkv7:wkv_state,gla:gated_linear_state,retnet:retention_state,deltanet:value_memory_state,gsa:slot_key_state|slot_value_state" ||
		inspection.Labels["sequence_mixer_state_slot_counts"] != "mamba2:2,rwkv7:1,gla:1,retnet:1,deltanet:1,gsa:2" ||
		inspection.Labels["sequence_mixer_cache_factory_contract"] != SequenceMixerCacheFactoryContract ||
		inspection.Labels["sequence_mixer_cache_factory_modes"] != "default,fp16,q8,k-q8-v-q4,paged,fixed,turboquant,mla-latent,compaction,compaction-full,recurrent" ||
		inspection.Labels["sequence_mixer_registered_cache_modes"] != "full_attention:default,mamba2:recurrent,rwkv7:recurrent,gla:recurrent,retnet:recurrent,deltanet:recurrent,gsa:recurrent,nsa:default,moba:default,mla:mla-latent" ||
		inspection.Labels["sequence_mixer_declared_kinds"] != "full_attention,mamba2" ||
		inspection.Labels["sequence_mixer_layer_types_source"] != "layer_types" ||
		inspection.Labels["sequence_mixer_load_plan_candidate"] != "true" ||
		inspection.Labels["sequence_mixer_registered_declared_kinds"] != "full_attention,mamba2" ||
		inspection.Labels["sequence_mixer_fla"] != "true" ||
		inspection.Labels["sequence_mixer_fla_kinds"] != "mamba2" ||
		inspection.Labels["sequence_mixer_fla_layers"] != "1" ||
		inspection.Labels["sequence_mixer_full_attention_layers"] != "1" ||
		inspection.Labels["sequence_mixer_runtime"] != SequenceMixerRuntimePlannedHIP {
		t.Fatalf("labels = %+v, want go-mlx FLA sequence mixer registry parity labels", inspection.Labels)
	}
	if inspection.Labels["sequence_mixer_subpath_discovery"] != "safetensors" ||
		inspection.Labels["sequence_mixer_subpath_status"] != "ok" ||
		inspection.Labels["sequence_mixer_subpath_count"] != "2" ||
		inspection.Labels["sequence_mixer_subpaths"] != "0:self_attn,1:mixer" {
		t.Fatalf("labels = %+v, want checkpoint-derived sequence mixer subpaths", inspection.Labels)
	}
	if inspection.Labels["sequence_mixer_load_plan"] != SequenceMixerRuntimePlannedHIP ||
		inspection.Labels["sequence_mixer_load_plan_contract"] != SequenceMixerRegistryContract ||
		inspection.Labels["sequence_mixer_load_plan_status"] != "valid" ||
		inspection.Labels["sequence_mixer_load_plan_layers"] != "2" ||
		inspection.Labels["sequence_mixer_load_plan_entries"] != "0:full_attention:kv-cache:self_attn:planned_hip,1:mamba2:recurrent:mixer:planned_hip" ||
		inspection.Labels["sequence_mixer_cache_plan_contract"] != SequenceMixerCachePlanContract ||
		inspection.Labels["sequence_mixer_cache_plan_layers"] != "2" ||
		inspection.Labels["sequence_mixer_cache_plan_entries"] != "0:full_attention:kv-cache:default,1:mamba2:recurrent:recurrent" ||
		inspection.Labels["sequence_mixer_cache_plan_state_slots"] != "1:mamba2:conv_state|ssm_state" {
		t.Fatalf("labels = %+v, want validated reactive sequence mixer load plan", inspection.Labels)
	}
}

func TestModelPackSequenceMixer_MissingRequiredLeaf_Bad(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractQwen3SafetensorsPack(t, `{
		"architectures":["Qwen3NextForCausalLM"],
		"hidden_size":4,
		"num_hidden_layers":1,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":4,
		"vocab_size":16,
		"max_position_embeddings":32768,
		"layer_types":["mamba2"]
	}`, `{
		"model.layers.0.mixer.in_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[0,32]},
		"model.layers.0.mixer.out_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[32,64]},
		"model.layers.0.mixer.conv1d.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[64,96]},
		"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[96,128]},
		"__metadata__":{"format":"pt"}
	}`, 128))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, mamba2 missing A_log should fail support", inspection, inspection.Labels)
	}
	if inspection.Labels["sequence_mixer_load_plan_status"] != "invalid" ||
		!strings.Contains(inspection.Labels["sequence_mixer_load_plan_error"], "layer 0 mamba2 missing required mixer tensors A_log") {
		t.Fatalf("labels = %+v, want missing required mamba2 leaf diagnostics", inspection.Labels)
	}
	if !nativeContractHasNoteContaining(inspection.Notes, "sequence mixer safetensors plan could not be validated") ||
		!strings.Contains(strings.Join(inspection.Notes, "\n"), "missing required mixer tensors A_log") {
		t.Fatalf("notes = %+v, want loud missing required mixer leaf note", inspection.Notes)
	}
}

func TestModelPackSequenceMixer_AliasOnlyNestedSubpathFailsRawDiscovery_Bad(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractQwen3SafetensorsPack(t, `{
		"architectures":["Qwen3NextForCausalLM"],
		"hidden_size":4,
		"num_hidden_layers":1,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":4,
		"vocab_size":16,
		"max_position_embeddings":32768,
		"layer_types":["full_attention"]
	}`, `{
		"language_model.model.layers.0.self_attn.q_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[0,16]},
		"language_model.model.layers.0.self_attn.k_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[16,32]},
		"language_model.model.layers.0.self_attn.v_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[32,48]},
		"language_model.model.layers.0.self_attn.o_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[48,64]},
		"language_model.model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[64,80]},
		"__metadata__":{"format":"pt"}
	}`, 80))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, alias-only nested subpath should fail support", inspection, inspection.Labels)
	}
	if inspection.Labels["sequence_mixer_subpath_status"] != "bare" ||
		inspection.Labels["sequence_mixer_subpath_count"] != "0" ||
		inspection.Labels["sequence_mixer_subpaths"] != "" {
		t.Fatalf("labels = %+v, want raw-key discovery to miss language_model alias subpath", inspection.Labels)
	}
	if inspection.Labels["sequence_mixer_load_plan_status"] != "invalid" ||
		!strings.Contains(inspection.Labels["sequence_mixer_load_plan_error"], "layer 0 full_attention missing required mixer tensors") {
		t.Fatalf("labels = %+v, want alias-only nested subpath to fail required leaf validation", inspection.Labels)
	}
}

func TestModelPackSequenceMixer_UniformModelTypePlan_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractQwen3SafetensorsPack(t, `{
		"model_type":"mamba2",
		"hidden_size":4,
		"num_hidden_layers":2,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":4,
		"vocab_size":16,
		"max_position_embeddings":32768
	}`, `{
		"model.layers.0.mixer.in_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[0,32]},
		"model.layers.0.mixer.out_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[32,64]},
		"model.layers.0.mixer.conv1d.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[64,96]},
		"model.layers.0.mixer.A_log":{"dtype":"F16","shape":[4,4],"data_offsets":[96,128]},
		"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[128,160]},
		"model.layers.1.mixer.in_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[160,192]},
		"model.layers.1.mixer.out_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[192,224]},
		"model.layers.1.mixer.conv1d.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[224,256]},
		"model.layers.1.mixer.A_log":{"dtype":"F16","shape":[4,4],"data_offsets":[256,288]},
		"model.layers.1.mlp.down_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[288,320]},
		"__metadata__":{"format":"pt"}
	}`, 320))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Labels["attention_layer_types"] != "mamba2,mamba2" ||
		inspection.Labels["sequence_mixer_declared_kinds"] != "mamba2" ||
		inspection.Labels["sequence_mixer_layer_types_source"] != "model_type" ||
		inspection.Labels["sequence_mixer_load_plan_candidate"] != "true" ||
		inspection.Labels["sequence_mixer_subpaths"] != "0:mixer,1:mixer" ||
		inspection.Labels["sequence_mixer_load_plan_status"] != "valid" ||
		inspection.Labels["sequence_mixer_load_plan_entries"] != "0:mamba2:recurrent:mixer:planned_hip,1:mamba2:recurrent:mixer:planned_hip" ||
		inspection.Labels["sequence_mixer_cache_plan_contract"] != SequenceMixerCachePlanContract ||
		inspection.Labels["sequence_mixer_cache_plan_entries"] != "0:mamba2:recurrent:recurrent,1:mamba2:recurrent:recurrent" ||
		inspection.Labels["sequence_mixer_cache_plan_state_slots"] != "0:mamba2:conv_state|ssm_state,1:mamba2:conv_state|ssm_state" {
		t.Fatalf("labels = %+v, want go-mlx uniform model_type sequence mixer plan", inspection.Labels)
	}
}

func TestModelPackSequenceMixer_UniformFullAttentionPlan_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractQwen3SafetensorsPack(t, `{
		"model_type":"full_attention",
		"hidden_size":4,
		"num_hidden_layers":1,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":4,
		"vocab_size":16,
		"max_position_embeddings":32768
	}`, `{
		"model.layers.0.self_attn.q_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[0,32]},
		"model.layers.0.self_attn.k_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[32,64]},
		"model.layers.0.self_attn.v_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[64,96]},
		"model.layers.0.self_attn.o_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[96,128]},
		"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[128,160]},
		"__metadata__":{"format":"pt"}
	}`, 160))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Labels["attention_layer_types"] != "full_attention" ||
		inspection.Labels["sequence_mixer_declared_kinds"] != "full_attention" ||
		inspection.Labels["sequence_mixer_layer_types_source"] != "model_type" ||
		inspection.Labels["sequence_mixer_load_plan_candidate"] != "true" ||
		inspection.Labels["sequence_mixer_fla"] != "" ||
		inspection.Labels["sequence_mixer_subpaths"] != "0:self_attn" ||
		inspection.Labels["sequence_mixer_load_plan_status"] != "valid" ||
		inspection.Labels["sequence_mixer_load_plan_entries"] != "0:full_attention:kv-cache:self_attn:planned_hip" ||
		inspection.Labels["sequence_mixer_cache_plan_contract"] != SequenceMixerCachePlanContract ||
		inspection.Labels["sequence_mixer_cache_plan_entries"] != "0:full_attention:kv-cache:default" {
		t.Fatalf("labels = %+v, want go-mlx uniform full_attention plan", inspection.Labels)
	}
}

func TestModelPackSequenceMixer_IgnoresFeedForwardSubpathDuringDiscovery_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractQwen3SafetensorsPack(t, `{
		"model_type":"composed",
		"hidden_size":4,
		"num_hidden_layers":1,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":4,
		"vocab_size":16,
		"max_position_embeddings":32768,
		"layer_types":["full_attention"]
	}`, `{
		"model.layers.0.self_attn.q_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[0,32]},
		"model.layers.0.self_attn.k_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[32,64]},
		"model.layers.0.self_attn.v_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[64,96]},
		"model.layers.0.self_attn.o_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[96,128]},
		"model.layers.0.block_sparse_moe.gate.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[128,160]},
		"model.layers.0.block_sparse_moe.experts.0.gate_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[160,192]},
		"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[192,224]},
		"__metadata__":{"format":"pt"}
	}`, 224))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want feed-forward subpath ignored for sequence mixer discovery", inspection, inspection.Labels)
	}
	if inspection.Labels["sequence_mixer_subpath_status"] != "ok" ||
		inspection.Labels["sequence_mixer_subpaths"] != "0:self_attn" ||
		inspection.Labels["sequence_mixer_load_plan_status"] != "valid" ||
		inspection.Labels["sequence_mixer_load_plan_entries"] != "0:full_attention:kv-cache:self_attn:planned_hip" {
		t.Fatalf("labels = %+v, want self_attn selected without block_sparse_moe ambiguity", inspection.Labels)
	}
}

func TestModelPackSequenceMixer_ComposedModelTypePlan_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractQwen3SafetensorsPack(t, `{
		"model_type":"composed",
		"hidden_size":4,
		"num_hidden_layers":2,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":4,
		"vocab_size":16,
		"max_position_embeddings":32768,
		"layer_types":["full_attention","mamba2"]
	}`, `{
		"model.layers.0.self_attn.q_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[0,32]},
		"model.layers.0.self_attn.k_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[32,64]},
		"model.layers.0.self_attn.v_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[64,96]},
		"model.layers.0.self_attn.o_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[96,128]},
		"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[128,160]},
		"model.layers.1.mixer.in_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[160,192]},
		"model.layers.1.mixer.out_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[192,224]},
		"model.layers.1.mixer.conv1d.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[224,256]},
		"model.layers.1.mixer.A_log":{"dtype":"F16","shape":[4,4],"data_offsets":[256,288]},
		"model.layers.1.mlp.down_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[288,320]},
		"__metadata__":{"format":"pt"}
	}`, 320))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "composed" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported composed sequence mixer plan", inspection, inspection.Labels)
	}
	if inspection.Labels["architecture_supported"] != "true" ||
		inspection.Labels["attention_layer_types"] != "full_attention,mamba2" ||
		inspection.Labels["sequence_mixer_declared_kinds"] != "full_attention,mamba2" ||
		inspection.Labels["sequence_mixer_layer_types_source"] != "layer_types" ||
		inspection.Labels["sequence_mixer_load_plan_status"] != "valid" ||
		inspection.Labels["sequence_mixer_load_plan_entries"] != "0:full_attention:kv-cache:self_attn:planned_hip,1:mamba2:recurrent:mixer:planned_hip" ||
		inspection.Labels["sequence_mixer_cache_plan_entries"] != "0:full_attention:kv-cache:default,1:mamba2:recurrent:recurrent" {
		t.Fatalf("labels = %+v, want go-mlx composed model_type reactive plan", inspection.Labels)
	}
}

func TestModelPackSequenceMixer_ComposedModelTypeMissingLayerTypes_Bad(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractQwen3SafetensorsPack(t, `{
		"model_type":"hybrid",
		"hidden_size":4,
		"num_hidden_layers":2,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":4,
		"vocab_size":16,
		"max_position_embeddings":32768
	}`, `{
		"model.layers.0.self_attn.q_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[0,32]},
		"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[32,64]},
		"__metadata__":{"format":"pt"}
	}`, 64))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, hybrid without layer_types should fail support", inspection, inspection.Labels)
	}
	if inspection.Model.Architecture != "hybrid" ||
		inspection.Labels["architecture_supported"] != "true" ||
		inspection.Labels["attention_layer_types"] != "" ||
		inspection.Labels["sequence_mixer_layer_types_source"] != "model_type" ||
		inspection.Labels["sequence_mixer_load_plan_status"] != "invalid" ||
		!strings.Contains(inspection.Labels["sequence_mixer_load_plan_error"], "needs per-layer layer_types or a mixer model_type") {
		t.Fatalf("labels = %+v, want go-mlx composed/hybrid missing layer_types refusal", inspection.Labels)
	}
	if !nativeContractHasNoteContaining(inspection.Notes, "sequence mixer safetensors plan could not be validated") ||
		!strings.Contains(strings.Join(inspection.Notes, "\n"), "needs per-layer layer_types or a mixer model_type") {
		t.Fatalf("notes = %+v, want loud composed/hybrid missing layer_types note", inspection.Notes)
	}
}

func TestModelPackSequenceMixer_LoadModelCarriesReactivePlanToNativeRuntime_Good(t *testing.T) {
	dir := nativeContractQwen3SafetensorsPack(t, `{
		"architectures":["Qwen3NextForCausalLM"],
		"hidden_size":4,
		"num_hidden_layers":2,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":4,
		"vocab_size":16,
		"max_position_embeddings":32768,
		"layer_types":["full_attention","mamba2"]
	}`, `{
		"model.embed_tokens.weight":{"dtype":"F16","shape":[16,4],"data_offsets":[0,128]},
		"lm_head.weight":{"dtype":"F16","shape":[16,4],"data_offsets":[128,256]},
		"model.layers.0.self_attn.q_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[256,288]},
		"model.layers.0.self_attn.k_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[288,320]},
		"model.layers.0.self_attn.v_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[320,352]},
		"model.layers.0.self_attn.o_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[352,384]},
		"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[384,416]},
		"model.layers.1.mixer.in_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[416,448]},
		"model.layers.1.mixer.out_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[448,480]},
		"model.layers.1.mixer.conv1d.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[480,512]},
		"model.layers.1.mixer.A_log":{"dtype":"F16","shape":[4,4],"data_offsets":[512,544]},
		"model.layers.1.mlp.down_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[544,576]},
		"__metadata__":{"format":"pt"}
	}`, 576)
	runtime := &fakeNativeRuntime{available: true, model: &fakeNativeModel{}}

	model, err := newROCmBackendWithRuntime(runtime).LoadModel(dir)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer model.Close()

	plan := runtime.loadConfig.SequenceMixerPlan
	if plan == nil {
		t.Fatal("SequenceMixerPlan is nil, want reactive plan carried into native runtime")
	}
	if plan.Contract != SequenceMixerRegistryContract ||
		plan.Runtime != SequenceMixerRuntimePlannedHIP ||
		len(plan.Layers) != 2 ||
		plan.Layers[0].Kind != "full_attention" ||
		plan.Layers[0].State != SequenceMixerStateKVCache ||
		plan.Layers[0].Subpath != "self_attn" ||
		plan.Layers[1].Kind != "mamba2" ||
		plan.Layers[1].State != SequenceMixerStateRecurrent ||
		plan.Layers[1].Subpath != "mixer" ||
		plan.Cache.Contract != SequenceMixerCachePlanContract ||
		len(plan.Cache.Layers) != 2 ||
		plan.Cache.Layers[0].Holder != SequenceMixerStateKVCache ||
		plan.Cache.Layers[0].Mode != SequenceMixerCacheModeDefault ||
		plan.Cache.Layers[1].Holder != SequenceMixerStateRecurrent ||
		plan.Cache.Layers[1].Mode != SequenceMixerCacheModeRecurrent {
		t.Fatalf("SequenceMixerPlan = %+v, want validated reactive composed plan", plan)
	}
	if runtime.loadConfig.ModelLabels["sequence_mixer_load_plan_status"] != "valid" {
		t.Fatalf("ModelLabels = %+v, want valid load-plan labels beside structured plan", runtime.loadConfig.ModelLabels)
	}
}

func TestModelPackSequenceMixer_AmbiguousSubpath_Bad(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractQwen3SafetensorsPack(t, `{
		"architectures":["Qwen3NextForCausalLM"],
		"hidden_size":4,
		"num_hidden_layers":1,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":4,
		"vocab_size":16,
		"max_position_embeddings":32768,
		"layer_types":["mamba2"]
	}`, `{
		"model.layers.0.self_attn.q_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[0,16]},
		"model.layers.0.mixer.in_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[16,48]},
		"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[48,64]},
		"__metadata__":{"format":"pt"}
	}`, 64))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, ambiguous sequence mixer subpath should fail support", inspection, inspection.Labels)
	}
	if inspection.Labels["sequence_mixer_subpath_status"] != "ambiguous" ||
		inspection.Labels["sequence_mixer_subpath_ambiguous_layers"] != "0:mixer|self_attn" {
		t.Fatalf("labels = %+v, want ambiguous subpath diagnostics", inspection.Labels)
	}
	if inspection.Labels["sequence_mixer_load_plan_status"] != "invalid" ||
		!strings.Contains(inspection.Labels["sequence_mixer_load_plan_error"], "0:mixer|self_attn") {
		t.Fatalf("labels = %+v, want invalid load-plan diagnostics", inspection.Labels)
	}
	if !nativeContractHasNoteContaining(inspection.Notes, "sequence mixer safetensors plan could not be validated") ||
		!strings.Contains(strings.Join(inspection.Notes, "\n"), "0:mixer|self_attn") {
		t.Fatalf("notes = %+v, want loud sequence mixer ambiguity note", inspection.Notes)
	}
}

func TestModelPackSequenceMixer_LayerTypesMismatch_Bad(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractQwen3SafetensorsPack(t, `{
		"architectures":["Qwen3NextForCausalLM"],
		"hidden_size":4,
		"num_hidden_layers":2,
		"num_attention_heads":1,
		"num_key_value_heads":1,
		"head_dim":4,
		"vocab_size":16,
		"max_position_embeddings":32768,
		"layer_types":["mamba2"]
	}`, `{
		"model.layers.0.mixer.in_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[0,32]},
		"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[32,48]},
		"__metadata__":{"format":"pt"}
	}`, 48))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Supported || inspection.Labels["weight_metadata_valid"] != "false" {
		t.Fatalf("inspection = %+v labels=%+v, mismatched layer_types should fail support", inspection, inspection.Labels)
	}
	if inspection.Labels["sequence_mixer_load_plan_status"] != "invalid" ||
		!strings.Contains(inspection.Labels["sequence_mixer_load_plan_error"], "layer_types length 1 != num_hidden_layers 2") {
		t.Fatalf("labels = %+v, want layer_types mismatch diagnostics", inspection.Labels)
	}
}
