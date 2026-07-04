// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"

	"dappco.re/go/inference"
)

func TestModelPackQwen36_HybridAttentionPlan_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, `{
		"architectures":["Qwen3_5ForConditionalGeneration"],
		"num_hidden_layers":6,
		"max_position_embeddings":262144,
		"sliding_window":512,
		"layer_types":["linear_attention","full_attention"]
	}`))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "qwen3_6" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported qwen3_6", inspection, inspection.Labels)
	}
	if inspection.Labels["qwen36_hybrid_attention"] != "true" ||
		inspection.Labels["attention_linear_layers"] != "3" ||
		inspection.Labels["attention_full_layers"] != "3" ||
		inspection.Labels["attention_cacheless_layers"] != "3" ||
		inspection.Labels["qwen36_cacheless_layers"] != "3" ||
		inspection.Labels["qwen36_hybrid_cache_plan"] != "metadata" ||
		inspection.Labels["qwen36_kv_cache_count"] != "3" ||
		inspection.Labels["qwen36_cache_index_by_layer"] != "-1,0,-1,1,-1,2" ||
		inspection.Labels["qwen36_local_window"] != "512" ||
		inspection.Labels["engine_profile"] != "qwen" ||
		inspection.Labels["engine_profile_source"] != "architecture_profile" ||
		inspection.Labels["engine_feature_architecture"] != "qwen3_6" ||
		inspection.Labels["engine_feature_chat_template_id"] != "qwen" ||
		inspection.Labels["engine_feature_reasoning_parser"] != "qwen" ||
		inspection.Labels["engine_feature_tool_parser"] != "qwen" ||
		inspection.Labels["engine_feature_text_generate"] != "false" ||
		inspection.Labels["engine_load_status"] != string(ROCmModelLoadStagedNative) ||
		inspection.Labels["engine_load_target"] != "standalone" ||
		inspection.Labels["engine_load_staged"] != "true" ||
		inspection.Labels["attention_layer_types"] != "linear_attention,full_attention,linear_attention,full_attention,linear_attention,full_attention" {
		t.Fatalf("labels = %+v, want repeated Qwen3.6 hybrid attention plan", inspection.Labels)
	}
	if load, ok := nativeInspectionCapability(inspection, inference.CapabilityModelLoad); !ok ||
		load.Status != inference.CapabilityStatusExperimental ||
		load.Labels["engine_load_status"] != string(ROCmModelLoadStagedNative) ||
		load.Labels["engine_load_architecture"] != "qwen3_6" ||
		load.Labels["engine_load_text_generate"] != "false" {
		t.Fatalf("model.load capability = %+v ok=%v, want staged Qwen3.6 load-status labels", load, ok)
	}
	if !nativeInspectionHasCapability(inspection, inference.CapabilityReasoningParse) ||
		!nativeInspectionHasCapability(inspection, inference.CapabilityToolParse) ||
		!nativeInspectionHasCapability(inspection, inference.CapabilityChatTemplate) {
		t.Fatalf("capabilities = %+v, want registry-derived Qwen parser/template capabilities", inspection.Capabilities)
	}
}

func TestModelPackQwen36_MoEHybridCachelessPlan_Good(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractSafetensorsPack(t, `{
		"architectures":["Qwen3_5MoeForConditionalGeneration"],
		"num_hidden_layers":4,
		"num_local_experts":128,
		"num_experts_per_tok":8,
		"layer_types":["linear_attention","full_attention"]
	}`))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if inspection.Model.Architecture != "qwen3_6_moe" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported qwen3_6_moe", inspection, inspection.Labels)
	}
	if inspection.Labels["qwen36_hybrid_attention"] != "true" ||
		inspection.Labels["attention_linear_layers"] != "2" ||
		inspection.Labels["attention_full_layers"] != "2" ||
		inspection.Labels["attention_cacheless_layers"] != "2" ||
		inspection.Labels["qwen36_cacheless_layers"] != "2" ||
		inspection.Labels["qwen36_hybrid_cache_plan"] != "metadata" ||
		inspection.Labels["qwen36_kv_cache_count"] != "2" ||
		inspection.Labels["qwen36_cache_index_by_layer"] != "-1,0,-1,1" ||
		inspection.Labels["engine_profile"] != "qwen" ||
		inspection.Labels["engine_feature_architecture"] != "qwen3_6_moe" ||
		inspection.Labels["engine_feature_moe"] != "true" ||
		inspection.Labels["engine_feature_chat_template_id"] != "qwen" ||
		inspection.Labels["engine_feature_capabilities"] != "chat.template,reasoning.parse,tool.parse" ||
		inspection.Labels["engine_load_status"] != string(ROCmModelLoadStagedNative) ||
		inspection.Labels["engine_load_target"] != "standalone" ||
		inspection.Labels["engine_load_staged"] != "true" {
		t.Fatalf("labels = %+v, want Qwen3.6 MoE cacheless hybrid attention plan", inspection.Labels)
	}
	if load, ok := nativeInspectionCapability(inspection, inference.CapabilityModelLoad); !ok ||
		load.Status != inference.CapabilityStatusExperimental ||
		load.Labels["engine_load_status"] != string(ROCmModelLoadStagedNative) ||
		load.Labels["engine_load_architecture"] != "qwen3_6_moe" ||
		load.Labels["engine_load_text_generate"] != "false" {
		t.Fatalf("model.load capability = %+v ok=%v, want staged Qwen3.6 MoE load-status labels", load, ok)
	}
	if !nativeInspectionHasCapability(inspection, inference.CapabilityReasoningParse) ||
		!nativeInspectionHasCapability(inspection, inference.CapabilityToolParse) ||
		!nativeInspectionHasCapability(inspection, inference.CapabilityChatTemplate) {
		t.Fatalf("capabilities = %+v, want registry-derived Qwen MoE parser/template capabilities", inspection.Capabilities)
	}
}
