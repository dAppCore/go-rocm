// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestRouteSetForIdentity_Good_CollectsReactiveModelRoutes(t *testing.T) {
	set, ok := RouteSetForIdentityWithOptions("/models/gemma4", inference.ModelIdentity{
		Architecture: "Gemma4ForConditionalGeneration",
		QuantBits:    6,
		Labels: map[string]string{
			"engine_architecture_resolved": "gemma4_text",
			"gemma4_quant_mode":            "q6",
		},
	}, RouteSetOptions{QuantLoaderPacks: []QuantLoaderPack{{
		Name:           "gemma-4-e4b-q6",
		Size:           "E4B",
		Bits:           6,
		QuantMode:      "affine",
		QuantGroup:     64,
		Runtime:        QuantRuntimeMLXAffine,
		GenerateStatus: QuantGenerateLinked,
		ProductRole:    "target",
		Supported:      true,
		RunnableOnCard: true,
	}}})
	if !ok ||
		!set.Matched() ||
		set.Contract != RouteSetContract ||
		set.Architecture != "gemma4_text" ||
		set.Family != "gemma4" ||
		!set.FeatureRoute.Matched() ||
		!set.LoaderRoute.Matched() ||
		!set.TokenizerRoute.Matched() ||
		!set.LoRAAdapterRoute.Matched() ||
		!set.StateContextRoute.Matched() ||
		!set.AttachedDrafterRoute.Matched() ||
		!set.QuantLoaderRoute.Matched() ||
		!set.RuntimeContractRoute.Matched() ||
		!set.RuntimeContractRoute.FixedSlidingCache ||
		set.QuantLoaderRoute.Mode != "q6" ||
		set.Labels["engine_route_set_contract"] != RouteSetContract ||
		set.Labels["engine_route_set_architecture"] != "gemma4_text" ||
		set.Labels["engine_route_set_feature"] != "true" ||
		set.Labels["engine_route_set_loader"] != "true" ||
		set.Labels["engine_route_set_tokenizer"] != "true" ||
		set.Labels["engine_route_set_lora_adapter"] != "true" ||
		set.Labels["engine_route_set_state_context"] != "true" ||
		set.Labels["engine_route_set_drafter"] != "true" ||
		set.Labels["engine_route_set_quant_loader"] != "true" ||
		set.Labels["engine_route_set_runtime_contract"] != "true" ||
		set.Labels["engine_route_set_runtime_contract_count"] == "" ||
		set.Labels["engine_route_set_quant_mode"] != "q6" {
		t.Fatalf("RouteSetForIdentityWithOptions = %+v ok=%v, want Gemma route set with q6 quant route", set, ok)
	}

	set.Model.Labels["engine_architecture_resolved"] = "mutated"
	set.FeatureRoute.Labels["engine_feature_route_architecture"] = "mutated"
	set.QuantLoaderRoute.Labels["engine_quant_loader_mode"] = "mutated"
	set.RuntimeContractRoute.Labels["engine_runtime_contract_architecture"] = "mutated"
	next, ok := RouteSetForIdentityWithOptions("/models/gemma4", inference.ModelIdentity{
		Architecture: "Gemma4ForConditionalGeneration",
		QuantBits:    6,
		Labels: map[string]string{
			"engine_architecture_resolved": "gemma4_text",
			"gemma4_quant_mode":            "q6",
		},
	}, RouteSetOptions{QuantLoaderPacks: []QuantLoaderPack{{
		Name:           "gemma-4-e4b-q6",
		Size:           "E4B",
		Bits:           6,
		QuantMode:      "affine",
		QuantGroup:     64,
		Runtime:        QuantRuntimeMLXAffine,
		GenerateStatus: QuantGenerateLinked,
		ProductRole:    "target",
		Supported:      true,
		RunnableOnCard: true,
	}}})
	if !ok ||
		next.Model.Labels["engine_architecture_resolved"] != "gemma4_text" ||
		next.FeatureRoute.Labels["engine_feature_route_architecture"] != "gemma4_text" ||
		next.QuantLoaderRoute.Labels["engine_quant_loader_mode"] != "q6" ||
		next.RuntimeContractRoute.Labels["engine_runtime_contract_architecture"] != "gemma4_text" {
		t.Fatalf("RouteSetForIdentityWithOptions leaked mutable state: %+v ok=%v", next, ok)
	}
}

func TestRouteSetForInspection_Good_MergesInspectionLabels(t *testing.T) {
	inspectionLabels := map[string]string{"architecture_resolved": "qwen3"}
	modelLabels := map[string]string{"engine_architecture_resolved": "bert_rerank"}
	inspection := &inference.ModelPackInspection{
		Path:   "/models/rerank",
		Labels: inspectionLabels,
		Model: inference.ModelIdentity{
			Architecture: "Qwen3ForCausalLM",
			Labels:       modelLabels,
		},
	}

	set, ok := RouteSetForInspection(inspection, RouteSetOptions{})
	if !ok ||
		!set.Matched() ||
		set.Architecture != "bert_rerank" ||
		set.Family != "bert" ||
		!set.FeatureRoute.Rerank ||
		set.FeatureRoute.TextGenerate ||
		!slices.Contains(set.FeatureRoute.Capabilities, inference.CapabilityRerank) {
		t.Fatalf("RouteSetForInspection = %+v ok=%v, want model-label-refined BERT rerank set", set, ok)
	}

	inspectionLabels["architecture_resolved"] = "mutated"
	modelLabels["engine_architecture_resolved"] = "mutated"
	if set.Architecture != "bert_rerank" {
		t.Fatalf("previous RouteSet mutated after caller label changes: %+v", set)
	}
}

func TestRouteSetForIdentity_Good_CollectsDirectSequenceMixerRoute(t *testing.T) {
	set, ok := RouteSetForIdentity("/models/mamba", inference.ModelIdentity{
		Architecture: "Mamba2ForCausalLM",
	})
	if !ok ||
		!set.Matched() ||
		set.Architecture != "mamba2" ||
		len(set.SequenceMixerRoutes) != 1 ||
		set.SequenceMixerRoutes[0].Kind != "mamba2" ||
		set.SequenceMixerRoutes[0].CacheMode != SequenceMixerCacheModeRecurrent ||
		!set.RuntimeContractRoute.DecodeUnavailableReporter ||
		set.Labels["engine_route_set_sequence_mixer"] != "true" ||
		set.Labels["engine_route_set_sequence_mixer_kinds"] != "mamba2" ||
		set.Labels["engine_route_set_sequence_mixer_cache_modes"] != SequenceMixerCacheModeRecurrent {
		t.Fatalf("RouteSetForIdentity(mamba2) = %+v ok=%v, want direct sequence mixer route", set, ok)
	}

	set.SequenceMixerRoutes[0].Labels["engine_mixer_loader_kind"] = "mutated"
	next, ok := RouteSetForIdentity("/models/mamba", inference.ModelIdentity{
		Architecture: "Mamba2ForCausalLM",
	})
	if !ok || next.SequenceMixerRoutes[0].Labels["engine_mixer_loader_kind"] != "mamba2" {
		t.Fatalf("RouteSetForIdentity sequence mixer route leaked mutable labels: %+v ok=%v", next, ok)
	}
}

func TestRouteSetForIdentity_Good_CollectsComposedLayerMixerRoutes(t *testing.T) {
	set, ok := RouteSetForIdentity("/models/hybrid", inference.ModelIdentity{
		Architecture: "hybrid",
		Labels: map[string]string{
			"layer_types": "full_attention,mamba2,mla,mamba2",
		},
	})
	if !ok ||
		!set.Matched() ||
		set.Architecture != "hybrid" ||
		len(set.SequenceMixerRoutes) != 3 ||
		set.SequenceMixerRoutes[0].Kind != "full_attention" ||
		set.SequenceMixerRoutes[1].Kind != "mamba2" ||
		set.SequenceMixerRoutes[2].Kind != "mla" ||
		set.Labels["engine_route_set_sequence_mixer_kinds"] != "full_attention,mamba2,mla" ||
		set.Labels["engine_route_set_sequence_mixer_cache_modes"] != "default,recurrent,mla-latent" {
		t.Fatalf("RouteSetForIdentity(hybrid layer_types) = %+v ok=%v, want unique composed mixer routes", set, ok)
	}
}

func TestRouteSetForIdentity_Good_Gemma4LayerTypesDoNotBecomeComposedRoutes(t *testing.T) {
	set, ok := RouteSetForIdentity("/models/gemma4", inference.ModelIdentity{
		Architecture: "Gemma4ForCausalLM",
		Labels: map[string]string{
			"layer_types": "sliding_attention,full_attention",
		},
	})
	if !ok ||
		set.Architecture != "gemma4_text" ||
		len(set.SequenceMixerRoutes) != 0 ||
		set.Labels["engine_route_set_sequence_mixer"] != "false" {
		t.Fatalf("RouteSetForIdentity(gemma4 layer_types) = %+v ok=%v, want no generic composed route", set, ok)
	}
}
