// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestROCmModelRouteSet_Good_RootIdentityUsesProductionQuantDefaults(t *testing.T) {
	set, ok := ROCmModelRouteSetForIdentity("/models/gemma4-e2b-q6", inference.ModelIdentity{
		Architecture: "Gemma4ForConditionalGeneration",
		QuantBits:    6,
		Labels: map[string]string{
			"engine_architecture_resolved": "gemma4_text",
			"gemma4_quant_mode":            "q6",
			"gemma4_size":                  "E2B",
		},
	})
	if !ok ||
		!set.Matched() ||
		set.Contract != ROCmModelRouteSetContract ||
		set.Architecture != "gemma4_text" ||
		set.Family != "gemma4" ||
		!set.FeatureRoute.Matched() ||
		!set.CacheRoute.Matched() ||
		!set.LoaderRoute.Matched() ||
		!set.TokenizerRoute.Matched() ||
		!set.LoRAAdapterRoute.Matched() ||
		!set.StateContextRoute.Matched() ||
		!set.AttachedDrafterRoute.Matched() ||
		!set.QuantLoaderRoute.Matched() ||
		set.QuantLoaderRoute.Mode != "q6" ||
		!set.RuntimeContractRoute.Matched() ||
		!set.RuntimeGatePlan.Matched() ||
		!set.RuntimeAuthorPlan.Matched() ||
		set.Labels["engine_route_set_contract"] != ROCmModelRouteSetContract ||
		set.Labels["engine_route_set_quant_loader"] != "true" ||
		set.Labels["engine_route_set_quant_mode"] != "q6" {
		t.Fatalf("ROCmModelRouteSetForIdentity = %+v ok=%v, want production-aware Gemma4 route set", set, ok)
	}

	labels := ApplyROCmModelRouteSetLabels(map[string]string{"caller": "kept"}, set)
	if labels["caller"] != "kept" ||
		labels["engine_route_set_contract"] != ROCmModelRouteSetContract ||
		labels["engine_route_set_quant_mode"] != "q6" {
		t.Fatalf("ApplyROCmModelRouteSetLabels = %+v, want route-set labels merged", labels)
	}
	labels["engine_route_set_quant_mode"] = "mutated"
	set.Labels["engine_route_set_quant_mode"] = "mutated"
	next, ok := ROCmModelRouteSetForIdentity("/models/gemma4-e2b-q6", inference.ModelIdentity{
		Architecture: "Gemma4ForConditionalGeneration",
		QuantBits:    6,
		Labels: map[string]string{
			"engine_architecture_resolved": "gemma4_text",
			"gemma4_quant_mode":            "q6",
			"gemma4_size":                  "E2B",
		},
	})
	if !ok || next.Labels["engine_route_set_quant_mode"] != "q6" {
		t.Fatalf("ROCmModelRouteSetForIdentity leaked mutable labels: %+v ok=%v", next, ok)
	}
}

func TestROCmModelRouteSet_Good_InfoInspectionAndProfileRoots(t *testing.T) {
	infoSet, ok := ROCmModelRouteSetForInfo("/models/qwen", inference.ModelInfo{
		Architecture: "Qwen3ForCausalLM",
		VocabSize:    151936,
	}, nil)
	if !ok ||
		infoSet.Architecture != "qwen3" ||
		infoSet.Family != "qwen" ||
		!infoSet.FeatureRoute.Matched() ||
		!infoSet.RuntimeGatePlan.Matched() ||
		infoSet.Labels["engine_route_set_runtime_gate"] != "true" {
		t.Fatalf("ROCmModelRouteSetForInfo = %+v ok=%v, want qwen route set", infoSet, ok)
	}

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
	inspectionSet, ok := ROCmModelRouteSetForInspection(inspection)
	if !ok ||
		inspectionSet.Architecture != "bert_rerank" ||
		inspectionSet.Family != "bert" ||
		!inspectionSet.FeatureRoute.Rerank ||
		inspectionSet.FeatureRoute.TextGenerate ||
		!slices.Contains(inspectionSet.FeatureRoute.Capabilities, inference.CapabilityRerank) {
		t.Fatalf("ROCmModelRouteSetForInspection = %+v ok=%v, want model-label-refined BERT rerank set", inspectionSet, ok)
	}
	inspectionLabels["architecture_resolved"] = "mutated"
	modelLabels["engine_architecture_resolved"] = "mutated"
	if inspectionSet.Architecture != "bert_rerank" {
		t.Fatalf("inspection route set mutated after caller label changes: %+v", inspectionSet)
	}

	profile, ok := ResolveROCmModelProfile("/models/gemma4-e2b-q6", inference.ModelIdentity{
		Architecture: "gemma4_text",
		QuantBits:    6,
		Labels: map[string]string{
			"gemma4_quant_mode": "q6",
			"gemma4_size":       "E2B",
		},
	})
	if !ok {
		t.Fatal("ResolveROCmModelProfile(gemma4_text) ok=false, want profile")
	}
	profileSet, ok := ROCmModelRouteSetForProfile(profile)
	if !ok ||
		profileSet.Architecture != "gemma4_text" ||
		profileSet.QuantLoaderRoute.Mode != "q6" ||
		profileSet.Labels["engine_route_set_quant_mode"] != "q6" {
		t.Fatalf("ROCmModelRouteSetForProfile = %+v ok=%v, want profile route set with production q6 route", profileSet, ok)
	}
}

func TestROCmModelRouteSet_Good_CustomOptionsCanBeExact(t *testing.T) {
	set, ok := ROCmModelRouteSetForIdentityWithOptions("/models/gemma4-e2b-q6", inference.ModelIdentity{
		Architecture: "gemma4_text",
		QuantBits:    6,
		Labels: map[string]string{
			"gemma4_quant_mode": "q6",
		},
	}, ROCmModelRouteSetOptions{})
	if !ok || !set.Matched() {
		t.Fatalf("ROCmModelRouteSetForIdentityWithOptions(empty) = %+v ok=%v, want non-quant route set", set, ok)
	}
	if set.QuantLoaderRoute.Matched() {
		t.Fatalf("ROCmModelRouteSetForIdentityWithOptions(empty) quant route = %+v, want caller-owned options to be exact", set.QuantLoaderRoute)
	}
}
