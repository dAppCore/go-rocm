// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"testing"
)

func TestModelPackDiffusionGemma_Good_LabelsBlockDiffusionBoundary(t *testing.T) {
	inspection, err := newROCmBackendWithRuntime(&fakeNativeRuntime{}).InspectModelPack(context.Background(), nativeContractQwen3SafetensorsPack(t, `{
		"architectures":["DiffusionGemmaForBlockDiffusion"],
		"model_type":"diffusion_gemma",
		"canvas_length":256,
		"hidden_size":16,
		"num_hidden_layers":1,
		"num_attention_heads":4,
		"num_key_value_heads":2,
		"head_dim":4,
		"vocab_size":128,
		"max_position_embeddings":32768
	}`, `{
		"model.decoder.embed_tokens.weight":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},
		"__metadata__":{"format":"pt"}
	}`, 2))
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	labels := inspection.Labels
	if inspection.Model.Architecture != "diffusion_gemma" || !inspection.Supported {
		t.Fatalf("inspection = %+v, want supported diffusion_gemma", inspection)
	}
	if labels["block_diffusion_model"] != "true" ||
		labels["diffusion_runtime"] != hipKernelStatusNotLinked ||
		labels["diffusion_sampler_runtime"] != hipKernelStatusNotLinked ||
		labels["diffusion_trunk_runtime"] != "model_pack_metadata" ||
		labels["diffusion_reference"] != "go_mlx_diffusion_gemma" ||
		labels["diffusion_fallback"] != "refused" ||
		labels["diffusion_canvas_length"] != "256" ||
		labels["diffusion_default_canvas_length"] != "64" ||
		labels["diffusion_reference_canvas_length"] != "256" ||
		labels["diffusion_default_max_steps"] != "16" ||
		labels["diffusion_reference_max_steps"] != "48" ||
		labels["diffusion_stability_threshold"] != "1" ||
		labels["diffusion_confidence_threshold"] != "0.005" ||
		labels["diffusion_entropy_bound"] != "0.3" ||
		labels["diffusion_reference_entropy_bound"] != "0.1" ||
		labels["diffusion_max_temperature"] != "0.8" ||
		labels["diffusion_min_temperature"] != "0.4" ||
		labels["diffusion_temperature_exponent"] != "1" ||
		labels["diffusion_text_vocab_size"] != "128" ||
		labels["gemma4_diffusion_default_canvas_length"] != "64" ||
		labels["gemma4_diffusion_reference_canvas_length"] != "256" ||
		labels["gemma4_diffusion_entropy_bound"] != "0.3" ||
		labels["engine_diffusion_sampler_route_contract"] != ROCmDiffusionSamplerRegistryContract ||
		labels["engine_diffusion_sampler_architecture"] != "diffusion_gemma" ||
		labels["engine_diffusion_sampler_reference"] != "go_mlx_diffusion_gemma" ||
		labels["engine_diffusion_sampler_diffusion_runtime"] != hipKernelStatusNotLinked ||
		labels["engine_diffusion_sampler_sampler_runtime"] != hipKernelStatusNotLinked ||
		labels["engine_diffusion_sampler_trunk_runtime"] != "model_pack_metadata" ||
		labels["engine_diffusion_sampler_canvas_length"] != "256" ||
		labels["engine_diffusion_sampler_default_canvas_length"] != "64" ||
		labels["engine_diffusion_sampler_default_max_steps"] != "16" ||
		labels["engine_diffusion_sampler_fallback_refused"] != "true" {
		t.Fatalf("labels = %+v, want DiffusionGemma block-diffusion boundary", labels)
	}
	route, ok := ROCmDiffusionSamplerRouteForInspection(inspection)
	if !ok ||
		route.Contract != ROCmDiffusionSamplerRegistryContract ||
		route.Reference != "go_mlx_diffusion_gemma" ||
		route.CanvasLength != 256 ||
		route.DefaultCanvasLength != 64 ||
		route.DefaultMaxSteps != 16 ||
		route.NativeRuntime ||
		!route.BlockDiffusion ||
		!route.Sampler ||
		!route.Trunk ||
		!route.FallbackRefused {
		t.Fatalf("DiffusionGemma sampler route = %+v ok=%t, want model-derived not-linked route", route, ok)
	}
}
