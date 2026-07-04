// SPDX-Licence-Identifier: EUPL-1.2

//go:build !linux || !amd64 || rocm_legacy_server

package rocm

import (
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"dappco.re/go/inference"
)

func TestPortableModelPackGemma4ReactiveRegistryLabels(t *testing.T) {
	tests := []struct {
		name          string
		pathName      string
		config        string
		wantSupported bool
		wantGenerate  bool
		wantSize      string
		wantMode      string
		wantRuntime   string
		wantStatus    string
		wantRunnable  string
		wantTier      string
	}{
		{
			name:          "e4b_q6_linked",
			pathName:      "lmstudio-community/gemma-4-E4B-it-MLX-6bit",
			wantSupported: true,
			wantGenerate:  true,
			wantSize:      "E4B",
			wantMode:      "q6",
			wantRuntime:   Gemma4RuntimeMLXAffine,
			wantStatus:    Gemma4GenerateLinked,
			wantRunnable:  "true",
			wantTier:      "default",
			config: `{
				"model_type":"gemma4_text",
				"hidden_size":2304,
				"num_hidden_layers":26,
				"vocab_size":262144,
				"max_position_embeddings":131072,
				"sliding_window":1024,
				"sliding_window_pattern":6,
				"num_kv_shared_layers":4,
				"enable_moe_block":true,
				"num_experts":16,
				"top_k_experts":2,
				"quantization_config":{"bits":6,"group_size":64,"quant_method":"mlx"}
			}`,
		},
		{
			name:          "thirty_one_b_q6_status_only",
			pathName:      "lmstudio-community/gemma-4-31B-it-MLX-6bit",
			wantSupported: false,
			wantGenerate:  false,
			wantSize:      "31B",
			wantMode:      "q6-status",
			wantRuntime:   Gemma4RuntimePlanned,
			wantStatus:    Gemma4GeneratePlannedOnly,
			wantRunnable:  "false",
			wantTier:      "status-only",
			config: `{
				"model_type":"gemma4_text",
				"hidden_size":4096,
				"num_hidden_layers":64,
				"vocab_size":262144,
				"max_position_embeddings":131072,
				"sliding_window":1024,
				"sliding_window_pattern":6,
				"num_kv_shared_layers":4,
				"quantization_config":{"bits":6,"group_size":64,"quant_method":"mlx"}
			}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := writePortableGemma4ModelPack(t, tt.pathName, tt.config)
			inspection, err := InspectModelPack(context.Background(), dir)
			if err != nil {
				t.Fatalf("InspectModelPack: %v", err)
			}
			if inspection.Supported != tt.wantSupported {
				t.Fatalf("inspection.Supported = %t, want %t labels=%+v", inspection.Supported, tt.wantSupported, inspection.Labels)
			}
			labels := inspection.Labels
			if inspection.Model.Architecture != "gemma4_text" ||
				labels["engine_registry"] != rocmModelRegistryName ||
				labels["engine_profile"] != "gemma4" ||
				labels["engine_profile_reactive"] != "true" ||
				labels["gemma4_size"] != tt.wantSize ||
				labels["gemma4_quant_mode"] != tt.wantMode ||
				labels["gemma4_runtime"] != tt.wantRuntime ||
				labels["gemma4_generate_status"] != tt.wantStatus ||
				labels["gemma4_runnable_on_card"] != tt.wantRunnable ||
				labels["production_quant_policy"] != "gemma4_mlx_affine" ||
				labels["production_quant_pack_sizes"] != "E2B,E4B,12B,26B-A4B,31B" ||
				labels["production_quant_size"] != tt.wantSize ||
				labels["production_quant_mode"] != tt.wantMode ||
				labels["production_quant_tier"] != tt.wantTier ||
				labels["production_quant_generate_status"] != tt.wantStatus ||
				labels["production_quant_runnable_on_card"] != tt.wantRunnable ||
				labels["architecture_resolved"] != "gemma4_text" ||
				labels["architecture_resolution_source"] != "model_type" ||
				labels["engine_architecture_resolved_family"] != "gemma4" ||
				labels["engine_architecture_resolved_parser"] != "gemma" ||
				labels["engine_architecture_resolved_chat_template"] != "gemma4_hf_turn" ||
				labels["engine_architecture_profile"] != "gemma4_text" ||
				labels["engine_architecture_runtime_status"] != string(inference.FeatureRuntimeNative) ||
				labels["engine_architecture_reasoning_parser"] != "gemma" ||
				labels["engine_architecture_tool_parser"] != "gemma" ||
				labels["engine_architecture_quantization_hints"] != "bf16,q8,q6,q4,mxfp8,mxfp4" ||
				labels["engine_architecture_cache_hints"] != "q8,paged,k-q8-v-q4,retained-state" ||
				labels["engine_chat_template"] != "gemma4_hf_turn" ||
				labels["chat_template"] != "gemma4_hf_turn" ||
				labels["engine_default_thinking"] != "true" ||
				labels["engine_requires_chat_template"] != "true" ||
				labels["gemma4_weight_policy"] != "model_registry" ||
				labels["engine_lora_policy_source"] != "model_registry" ||
				labels["gemma4_lora_default_targets"] != "q_proj,v_proj,o_proj" ||
				labels["gemma4_lora_safe_targets"] != "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" ||
				labels["gemma4_lora_extended_targets"] != "router.proj,per_layer_input_gate,per_layer_projection" ||
				labels["gemma4_lora_extended_targets_require_opt_in"] != "true" {
				t.Fatalf("labels = %+v, want portable Gemma4 registry/matrix labels for %s/%s", labels, tt.wantSize, tt.wantMode)
			}
			wantLinkedFastPath := "false"
			if tt.wantGenerate {
				wantLinkedFastPath = "true"
			}
			if labels["gemma4_attention_sliding_window"] != "1024" ||
				labels["gemma4_attention_sliding_pattern"] != "6" ||
				labels["gemma4_attention_kv_shared_layers"] != "4" ||
				labels["gemma4_attention_layer_count"] == "" ||
				labels["attention_cache_shared_layers"] != "4" ||
				labels["gemma4_fixed_sliding_prefill_chunk_limit"] != "1024" ||
				labels["attention_window_policy"] != "sliding_causal" ||
				labels["gemma4_attention_mask_fixed_single_token"] != "true" ||
				labels["gemma4_speculative_verify_proposal_window_limit"] != "1023" ||
				labels["engine_fixed_sliding_cache"] != "true" ||
				labels["engine_fixed_sliding_cache_bound"] != "true" ||
				labels["engine_native_mlp_matvec"] != wantLinkedFastPath ||
				labels["engine_native_linear_matvec"] != wantLinkedFastPath ||
				labels["engine_native_q6_bitstream_matvec"] != wantLinkedFastPath ||
				labels["engine_native_attention_o_matvec"] != wantLinkedFastPath ||
				labels["engine_native_fixed_sliding_attention"] != wantLinkedFastPath ||
				labels["engine_generation_stream"] != wantLinkedFastPath ||
				labels["engine_async_decode_prefetch"] != wantLinkedFastPath ||
				labels["engine_compiled_layer_decode"] != "false" ||
				labels["engine_pipelined_decode"] != "false" {
				t.Fatalf("labels = %+v, want Gemma4 config-driven attention feature labels", labels)
			}
			if tt.name == "e4b_q6_linked" && (labels["gemma4_mixture"] != "true" || labels["gemma4_num_experts"] != "16" || labels["gemma4_top_k_experts"] != "2") {
				t.Fatalf("labels = %+v, want Gemma4 config-driven MoE labels", labels)
			}
			profile, ok := defaultROCmModelProfileRegistry().Resolve(rocmModelProfileRequest{
				Path:  inspection.Path,
				Model: inspection.Model,
			})
			if !ok || !profile.Matched() || profile.Registry != rocmModelRegistryName || profile.Name != "gemma4" || profile.Family != "gemma4" {
				t.Fatalf("profile = %+v ok=%t, want portable Gemma4 registry/factory match", profile, ok)
			}
			if len(profile.Gemma4LoRATargetPolicy.DefaultTargets) != 3 ||
				profile.Gemma4LoRATargetPolicy.DefaultTargets[0] != "q_proj" ||
				profile.Gemma4LoRATargetPolicy.TargetPaths["router.proj"] != "router.proj" {
				t.Fatalf("profile LoRA policy = %+v, want portable Gemma4 registry policy", profile.Gemma4LoRATargetPolicy)
			}
			if profile.Gemma4Settings.ID != "gemma4_text" ||
				profile.Gemma4Settings.ChatTemplate != "gemma4_hf_turn" ||
				!profile.Gemma4Settings.DefaultThinking ||
				profile.Gemma4Settings.GenerationRole != "model" {
				t.Fatalf("profile settings = %+v, want portable Gemma4 registry architecture settings", profile.Gemma4Settings)
			}
			if profile.Gemma4EngineFeatures.GenerateLinked() != tt.wantGenerate ||
				profile.Gemma4DeclaredFeatures.Attention.SlidingWindow != 1024 ||
				profile.Gemma4DeclaredFeatures.Attention.SlidingPattern != 6 ||
				profile.Gemma4DeclaredFeatures.Attention.SharedKVLayers != 4 {
				t.Fatalf("profile = %+v, want registry-derived Gemma4 engine/declaration features", profile)
			}
			generate, hasGenerate := portableInspectionCapability(inspection, inference.CapabilityGenerate)
			if hasGenerate != tt.wantGenerate {
				t.Fatalf("generate capability present = %t, want %t capabilities=%+v", hasGenerate, tt.wantGenerate, inspection.Capabilities)
			}
			if hasGenerate && (generate.Labels["engine_lora_policy_source"] != "model_registry" ||
				generate.Labels["gemma4_lora_default_targets"] != "q_proj,v_proj,o_proj" ||
				generate.Labels["engine_architecture_runtime_status"] != string(inference.FeatureRuntimeNative) ||
				generate.Labels["engine_architecture_reasoning_parser"] != "gemma" ||
				generate.Labels["engine_chat_template"] != "gemma4_hf_turn" ||
				generate.Labels["gemma4_weight_policy"] != "model_registry") {
				t.Fatalf("generate capability = %+v, want registry-owned Gemma4 settings and LoRA policy labels", generate)
			}
			if !tt.wantGenerate {
				load, ok := portableInspectionCapability(inspection, inference.CapabilityModelLoad)
				if !ok || load.Labels["gemma4_generate_status"] != Gemma4GeneratePlannedOnly ||
					load.Labels["engine_lora_policy_source"] != "model_registry" ||
					load.Labels["gemma4_lora_default_targets"] != "q_proj,v_proj,o_proj" ||
					load.Labels["engine_architecture_runtime_status"] != string(inference.FeatureRuntimeNative) ||
					load.Labels["engine_architecture_reasoning_parser"] != "gemma" ||
					load.Labels["engine_chat_template"] != "gemma4_hf_turn" ||
					load.Labels["gemma4_weight_policy"] != "model_registry" {
					t.Fatalf("model.load capability = %+v ok=%t, want planned Gemma4 status-only load capability", load, ok)
				}
			}
		})
	}
}

func TestPortableModelPackGemma4ProcessorConfig_MetadataLabels_Good(t *testing.T) {
	dir := writePortableGemma4ModelPack(t, "gemma-4-e2b-it", `{
		"model_type":"gemma4",
		"text_config":{
			"model_type":"gemma4_text",
			"hidden_size":2304,
			"num_hidden_layers":26,
			"vocab_size":262144,
			"max_position_embeddings":131072
		}
	}`)
	if err := os.WriteFile(filepath.Join(dir, "processor_config.json"), []byte(`{
		"audio_ms_per_token":40,
		"audio_seq_length":750,
		"image_processor":{"max_soft_tokens":280,"do_resize":true},
		"video_processor":{"max_soft_tokens":70,"do_resize":true,"num_frames":32},
		"feature_extractor":{"sampling_rate":16000,"num_mel_filters":128,"fft_length":512,"hop_length":160}
	}`), 0o644); err != nil {
		t.Fatalf("write processor_config.json: %v", err)
	}

	inspection, err := InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	labels := inspection.Labels
	for key, want := range map[string]string{
		"processor_config":                 "true",
		"multimodal_model":                 "true",
		"gemma4_multimodal":                "true",
		"vision_processor_config":          "true",
		"audio_processor_config":           "true",
		"image_processor_max_soft_tokens":  "280",
		"video_processor_num_frames":       "32",
		"audio_feature_extractor":          "true",
		"audio_feature_frame_length":       "320",
		"audio_feature_fft_length":         "512",
		"audio_feature_max_length_samples": "480000",
		"vision_runtime":                   hipKernelStatusNotLinked,
		"audio_frontend_runtime":           hipKernelStatusNotLinked,
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok || !profile.Gemma4DeclaredFeatures.Vision || !profile.Gemma4DeclaredFeatures.Audio {
		t.Fatalf("ROCm model profile = %+v ok=%v labels=%+v, want processor config to declare Gemma4 multimodal features", profile, ok, labels)
	}
}

func TestPortableModelPackGemma4AssistantRegistryProfileAttachedOnly(t *testing.T) {
	dir := writePortableGemma4ModelPack(t, "google/gemma-4-E4B-it-assistant", `{
		"model_type":"gemma4_assistant",
		"hidden_size":2304,
		"num_hidden_layers":4,
		"vocab_size":262144,
		"max_position_embeddings":131072
	}`)
	inspection, err := InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !inspection.Supported || inspection.Model.Architecture != "gemma4_assistant" {
		t.Fatalf("inspection = %+v labels=%+v, want supported portable Gemma4 assistant metadata pack", inspection, inspection.Labels)
	}
	labels := inspection.Labels
	if labels["engine_registry"] != rocmModelRegistryName ||
		labels["engine_profile"] != "gemma4" ||
		labels["engine_profile_reactive"] != "true" ||
		labels["engine_profile_architecture"] != "gemma4_assistant" ||
		labels["engine_architecture_profile"] != "gemma4_assistant" ||
		labels["engine_architecture_attached_only"] != "true" ||
		labels["engine_architecture_generation"] != "false" ||
		labels["engine_architecture_cache_hints"] != "retained-state,attached-drafter" ||
		labels["engine_text_generate"] != "false" ||
		labels["gemma4_lora_default_targets"] != "" ||
		labels["gemma4_weight_policy"] != "" {
		t.Fatalf("labels = %+v, want portable Gemma4 attached-only assistant registry labels", labels)
	}
	if labels["gemma4_size"] != "E4B" ||
		labels["gemma4_quant_mode"] != "bf16" ||
		labels["gemma4_runtime"] != Gemma4RuntimeBF16 ||
		labels["gemma4_generate_status"] != Gemma4GenerateLoadOnly ||
		labels["gemma4_pack_supported"] != "true" ||
		labels["gemma4_runnable_on_card"] != "true" ||
		labels["production_quant_pack"] != "E4B:assistant-bf16" ||
		labels["production_quant_pack_name"] != "e4b-assistant-bf16" ||
		labels["production_quant_tier"] != "mtp-assistant" ||
		labels["production_quant_model"] != dir ||
		labels["production_quant_assistant_model"] != dir ||
		labels["production_quant_mtp_assistant"] != "true" ||
		labels["production_quant_target_family"] != "gemma4" {
		t.Fatalf("labels = %+v, want portable Gemma4 assistant production quant labels", labels)
	}
	profile, ok := ResolveROCmModelProfileForInspection(inspection)
	if !ok || !profile.Matched() || profile.Architecture != "gemma4_assistant" {
		t.Fatalf("profile = %+v ok=%v, want portable Gemma4 assistant registry profile", profile, ok)
	}
	if !profile.Gemma4Settings.AttachedOnly ||
		profile.Gemma4Settings.Generation ||
		profile.Gemma4Settings.Chat ||
		profile.Model.QuantBits != 16 ||
		profile.Model.QuantType != "bf16" ||
		profile.Model.Labels["production_quant_pack"] != "E4B:assistant-bf16" ||
		len(profile.Gemma4LoRATargetPolicy.DefaultTargets) != 0 ||
		profile.Gemma4EngineFeatures.GenerateLinked() {
		t.Fatalf("profile = %+v, want portable attached-only assistant model profile with no standalone generation", profile)
	}
	if _, ok := portableInspectionCapability(inspection, inference.CapabilityGenerate); ok {
		t.Fatalf("capabilities = %+v, assistant pack must not expose standalone generation", inspection.Capabilities)
	}
	load, ok := portableInspectionCapability(inspection, inference.CapabilityModelLoad)
	if !ok ||
		load.Labels["production_quant_pack"] != "E4B:assistant-bf16" ||
		load.Labels["production_quant_mtp_assistant"] != "true" ||
		load.Labels["engine_architecture_attached_only"] != "true" ||
		load.Labels["engine_text_generate"] != "false" {
		t.Fatalf("model.load capability = %+v ok=%v, want attached-only assistant load labels", load, ok)
	}
}

func TestPortableModelPackGenericReactiveRegistryLabels(t *testing.T) {
	dir := writePortableGemma4ModelPack(t, "qwen/qwen3-6-moe", `{
		"architectures":["Qwen3_5MoeForConditionalGeneration"],
		"num_hidden_layers":4,
		"hidden_size":2048,
		"vocab_size":151936,
		"num_local_experts":128,
		"num_experts_per_tok":8,
		"quantization_config":{"bits":4,"group_size":64,"quant_method":"gptq"}
	}`)
	inspection, err := InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	if !inspection.Supported || inspection.Model.Architecture != "qwen3_6_moe" {
		t.Fatalf("inspection = %+v labels=%+v, want supported portable qwen3_6_moe profile", inspection, inspection.Labels)
	}
	labels := inspection.Labels
	if labels["engine_registry"] != rocmModelRegistryName ||
		labels["engine_profile"] != "qwen" ||
		labels["engine_profile_source"] != "architecture_profile" ||
		labels["engine_profile_reactive"] != "true" ||
		labels["architecture_resolved"] != "qwen3_6_moe" ||
		labels["engine_architecture_profile"] != "qwen3_6_moe" ||
		labels["engine_architecture_reasoning_parser"] != "qwen" ||
		labels["engine_architecture_tool_parser"] != "qwen" ||
		labels["engine_architecture_moe"] != "true" ||
		labels["engine_feature_architecture"] != "qwen3_6_moe" ||
		labels["engine_feature_family"] != "qwen" ||
		labels["engine_feature_text_generate"] != "false" ||
		labels["engine_feature_chat_template"] != "true" ||
		labels["engine_feature_chat_template_id"] != "qwen" ||
		labels["engine_feature_reasoning_parser"] != "qwen" ||
		labels["engine_feature_tool_parser"] != "qwen" ||
		labels["engine_feature_moe"] != "true" ||
		labels["engine_feature_capabilities"] != "chat.template,reasoning.parse,tool.parse" ||
		labels["engine_load_status"] != string(ROCmModelLoadStagedNative) ||
		labels["engine_load_target"] != "standalone" ||
		labels["engine_load_staged"] != "true" ||
		labels["engine_load_text_generate"] != "false" {
		t.Fatalf("labels = %+v, want generic architecture-profile registry and engine feature labels", labels)
	}
	if _, ok := portableInspectionCapability(inspection, inference.CapabilityGenerate); ok {
		t.Fatalf("capabilities = %+v, staged Qwen profile must not claim generate", inspection.Capabilities)
	}
	if load, ok := portableInspectionCapability(inspection, inference.CapabilityModelLoad); !ok ||
		load.Status != inference.CapabilityStatusExperimental ||
		load.Labels["engine_load_status"] != string(ROCmModelLoadStagedNative) ||
		load.Labels["engine_load_architecture"] != "qwen3_6_moe" ||
		load.Labels["engine_load_text_generate"] != "false" {
		t.Fatalf("model.load capability = %+v ok=%v, want staged Qwen load-status labels", load, ok)
	}
	for _, id := range []inference.CapabilityID{inference.CapabilityChatTemplate, inference.CapabilityReasoningParse, inference.CapabilityToolParse} {
		if capability, ok := portableInspectionCapability(inspection, id); !ok ||
			capability.Labels["engine_feature_chat_template_id"] != "qwen" ||
			capability.Labels["engine_feature_reasoning_parser"] != "qwen" {
			t.Fatalf("capability %s = %+v ok=%v, want registry-derived parser/template labels", id, capability, ok)
		}
	}
}

func TestPortableModelPackDiffusionGemmaPolicyLabels_Good(t *testing.T) {
	dir := writePortableGemma4ModelPack(t, "google/diffusion-gemma", `{
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
	}`)
	inspection, err := InspectModelPack(context.Background(), dir)
	if err != nil {
		t.Fatalf("InspectModelPack: %v", err)
	}
	labels := inspection.Labels
	if inspection.Model.Architecture != "diffusion_gemma" || !inspection.Supported {
		t.Fatalf("inspection = %+v labels=%+v, want supported DiffusionGemma metadata pack", inspection, labels)
	}
	for key, want := range map[string]string{
		"block_diffusion_model":                  "true",
		"diffusion_runtime":                      hipKernelStatusNotLinked,
		"diffusion_sampler_runtime":              hipKernelStatusNotLinked,
		"diffusion_trunk_runtime":                "model_pack_metadata",
		"diffusion_reference":                    "go_mlx_diffusion_gemma",
		"diffusion_fallback":                     "refused",
		"diffusion_canvas_length":                "256",
		"diffusion_default_canvas_length":        "64",
		"diffusion_reference_canvas_length":      "256",
		"diffusion_default_max_steps":            "16",
		"diffusion_reference_max_steps":          "48",
		"diffusion_stability_threshold":          "1",
		"diffusion_confidence_threshold":         "0.005",
		"diffusion_entropy_bound":                "0.3",
		"diffusion_reference_entropy_bound":      "0.1",
		"diffusion_max_temperature":              "0.8",
		"diffusion_min_temperature":              "0.4",
		"diffusion_temperature_exponent":         "1",
		"diffusion_text_vocab_size":              "128",
		"gemma4_diffusion_default_canvas_length": "64",
		"gemma4_diffusion_entropy_bound":         "0.3",
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}
	route, ok := ROCmDiffusionSamplerRouteForInspection(inspection)
	if !ok ||
		route.CanvasLength != 256 ||
		route.DefaultCanvasLength != 64 ||
		route.ReferenceCanvasLength != 256 ||
		route.DefaultMaxSteps != 16 ||
		route.ReferenceMaxSteps != 48 ||
		route.EntropyBound != 0.3 ||
		route.MaxTemperature != 0.8 ||
		route.MinTemperature != 0.4 ||
		route.TemperatureExponent != 1.0 ||
		!route.FallbackRefused {
		t.Fatalf("portable DiffusionGemma route = %+v ok=%t, want policy-derived route", route, ok)
	}
}

func writePortableGemma4ModelPack(t *testing.T, name, config string) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), filepath.FromSlash(name))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir model pack: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	writePortableSafetensors(t, filepath.Join(dir, "model.safetensors"))
	return dir
}

func writePortableSafetensors(t *testing.T, path string) {
	t.Helper()
	header := []byte(`{"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[0,16]},"__metadata__":{"format":"pt"}}`)
	buf := make([]byte, 8+len(header)+16)
	binary.LittleEndian.PutUint64(buf[:8], uint64(len(header)))
	copy(buf[8:], header)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("write safetensors: %v", err)
	}
}

func portableInspectionCapability(inspection *inference.ModelPackInspection, id inference.CapabilityID) (inference.Capability, bool) {
	for _, capability := range inspection.Capabilities {
		if capability.ID == id {
			return capability, true
		}
	}
	return inference.Capability{}, false
}
