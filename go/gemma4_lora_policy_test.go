// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"strconv"
	"strings"
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
	rocmscheme "dappco.re/go/rocm/scheme"
)

func TestGemma4LoRATargetPolicy_Good_MatchesReactiveContract(t *testing.T) {
	wantDefault := []string{"q_proj", "v_proj", "o_proj"}
	for _, architecture := range []string{
		"gemma4",
		"gemma4_text",
		"gemma4_unified",
		"Gemma4ForConditionalGeneration",
		"Gemma4UnifiedForConditionalGeneration",
	} {
		t.Run(architecture, func(t *testing.T) {
			policy, ok := Gemma4LoRATargetPolicyForArchitecture(architecture)
			if !ok {
				t.Fatalf("Gemma4LoRATargetPolicyForArchitecture(%q) ok = false", architecture)
			}
			if !slices.Equal(policy.DefaultTargets, wantDefault) {
				t.Fatalf("DefaultTargets = %v, want %v", policy.DefaultTargets, wantDefault)
			}
			cases := []struct {
				target   string
				wantPath string
				wantSafe bool
			}{
				{"q_proj", "self_attn.q_proj", true},
				{"self_attn.q_proj", "self_attn.q_proj", true},
				{"gate_proj", "mlp.gate_proj", true},
				{"mlp.up_proj", "mlp.up_proj", true},
				{"router.proj", "router.proj", false},
				{"per_layer_input_gate", "per_layer_input_gate", false},
			}
			for _, tc := range cases {
				path, ok := Gemma4LoRATargetPath(architecture, tc.target)
				if !ok || path != tc.wantPath {
					t.Fatalf("Gemma4LoRATargetPath(%q, %q) = %q, %v; want %q, true", architecture, tc.target, path, ok, tc.wantPath)
				}
				canonical, ok := Gemma4LoRACanonicalTarget(architecture, "model.layers.0."+tc.target)
				if !ok || canonical != "model.layers.0."+tc.wantPath {
					t.Fatalf("Gemma4LoRACanonicalTarget(%q, %q) = %q, %v; want %q, true", architecture, tc.target, canonical, ok, "model.layers.0."+tc.wantPath)
				}
				if safe := Gemma4LoRASafeTarget(architecture, tc.target); safe != tc.wantSafe {
					t.Fatalf("Gemma4LoRASafeTarget(%q, %q) = %v, want %v", architecture, tc.target, safe, tc.wantSafe)
				}
			}
			if _, ok := Gemma4LoRATargetPath(architecture, "vision_tower.q_proj"); ok {
				t.Fatalf("Gemma4LoRATargetPath(%q, vision_tower.q_proj) ok = true, want false", architecture)
			}
		})
	}
}

func TestGemma4LoRATargetPolicy_Good_DefensiveCopies(t *testing.T) {
	policy, ok := Gemma4LoRATargetPolicyForArchitecture("gemma4_text")
	if !ok {
		t.Fatal("Gemma4LoRATargetPolicyForArchitecture ok = false")
	}
	policy.DefaultTargets[0] = "mutated"
	policy.TargetPaths["q_proj"] = "mutated"

	policy, ok = Gemma4LoRATargetPolicyForArchitecture("gemma4_text")
	if !ok {
		t.Fatal("Gemma4LoRATargetPolicyForArchitecture ok = false")
	}
	if policy.DefaultTargets[0] != "q_proj" || policy.TargetPaths["q_proj"] != "self_attn.q_proj" {
		t.Fatalf("policy was mutated through returned copy: %+v", policy)
	}
}

func TestGemma4LoRATargetPolicy_Good_NoAssistantTargets(t *testing.T) {
	if targets := Gemma4LoRADefaultTargets("gemma4_assistant"); targets != nil {
		t.Fatalf("Gemma4LoRADefaultTargets(gemma4_assistant) = %v, want nil", targets)
	}
	if _, ok := Gemma4LoRATargetPolicyForArchitecture("Gemma4AssistantForCausalLM"); ok {
		t.Fatal("Gemma4LoRATargetPolicyForArchitecture(assistant) ok = true, want false")
	}
}

func TestDefaultROCmModelRegistrySnapshot_Good_CopySafeReactiveContract(t *testing.T) {
	snapshot := DefaultROCmModelRegistrySnapshot(" cuda ")
	wantProfiles := len(DefaultROCmArchitectureProfiles())
	wantFeatureRoutes := len(DefaultROCmModelFeatureRoutes())
	wantTokenizerRoutes := len(DefaultROCmModelTokenizerRoutes())
	wantLoRARoutes := len(DefaultROCmLoRAAdapterRoutes())
	wantMultimodalRoutes := len(DefaultROCmMultimodalProcessorRoutes())
	wantDiffusionRoutes := len(DefaultROCmDiffusionSamplerRoutes())
	wantStateContextRoutes := len(DefaultROCmStateContextRoutes())
	wantAttachedDrafterRoutes := len(DefaultROCmAttachedDrafterRoutes())
	wantRoutes := len(DefaultROCmModelLoaderRoutes())
	wantCacheModeRoutes := len(DefaultROCmCacheModeRoutes())
	wantCacheRoutes := len(defaultROCmModelRegistryCacheRoutes(DefaultROCmArchitectureProfiles()))
	wantQuantSchemes := len(DefaultROCmQuantSchemes())
	wantQuantRoutes := len(DefaultROCmQuantLoaderRoutes())
	wantMixerRoutes := len(DefaultROCmSequenceMixerLoaderRoutes())
	wantAlgorithmProfiles := len(DefaultROCmAlgorithmProfiles())
	wantSchemeMixers := len(rocmscheme.MixerKinds())
	wantSchemeCaches := len(rocmscheme.CacheModes())
	wantSchemeQuants := len(rocmscheme.QuantKinds())
	wantModelLoaders := len(rocmmodel.DefaultLoaderRoutes())
	if snapshot.Name != rocmModelRegistryName ||
		snapshot.Backend != "cuda" ||
		snapshot.DefaultFamily != "gemma4" ||
		len(snapshot.Factories) != 2 ||
		snapshot.Factories[0] != "gemma4" ||
		snapshot.Factories[1] != "architecture-profile" ||
		len(snapshot.ArchitectureProfiles) != wantProfiles ||
		len(snapshot.FeatureRoutes) != wantFeatureRoutes ||
		len(snapshot.TokenizerRoutes) != wantTokenizerRoutes ||
		len(snapshot.LoRAAdapterRoutes) != wantLoRARoutes ||
		len(snapshot.MultimodalProcessorRoutes) != wantMultimodalRoutes ||
		len(snapshot.DiffusionSamplerRoutes) != wantDiffusionRoutes ||
		len(snapshot.StateContextRoutes) != wantStateContextRoutes ||
		len(snapshot.AttachedDrafterRoutes) != wantAttachedDrafterRoutes ||
		len(snapshot.LoaderRoutes) != wantRoutes ||
		len(snapshot.CacheModeRoutes) != wantCacheModeRoutes ||
		len(snapshot.CacheRoutes) != wantCacheRoutes ||
		len(snapshot.QuantSchemes) != wantQuantSchemes ||
		len(snapshot.QuantLoaderRoutes) != wantQuantRoutes ||
		len(snapshot.MixerLoaderRoutes) != wantMixerRoutes ||
		len(snapshot.AlgorithmProfiles) != wantAlgorithmProfiles ||
		snapshot.Labels["engine_registry"] != rocmModelRegistryName ||
		snapshot.Labels["engine_algorithm_profile_contract"] != ROCmAlgorithmProfileRegistryContract ||
		snapshot.Labels["engine_config_probe_contract"] != ROCmModelConfigProbeContract ||
		snapshot.Labels["engine_feature_route_contract"] != ROCmModelFeatureRegistryContract ||
		snapshot.Labels["engine_tokenizer_route_contract"] != ROCmModelTokenizerRegistryContract ||
		snapshot.Labels["engine_lora_route_contract"] != ROCmLoRAAdapterRegistryContract ||
		snapshot.Labels["engine_model_loader_contract"] != rocmmodel.LoaderRegistryContract ||
		!strings.Contains(snapshot.Labels["engine_model_loader_architectures"], "gemma4") ||
		snapshot.Labels["engine_loader_contract"] != ROCmModelLoaderRegistryContract ||
		snapshot.Labels["engine_cache_factory_contract"] != ROCmCacheFactoryRouteContract ||
		snapshot.Labels["engine_mixer_loader_contract"] != ROCmSequenceMixerLoaderRegistryContract ||
		snapshot.Labels["engine_multimodal_processor_route_contract"] != ROCmMultimodalProcessorRegistryContract ||
		snapshot.Labels["engine_diffusion_sampler_route_contract"] != ROCmDiffusionSamplerRegistryContract ||
		snapshot.Labels["engine_state_context_route_contract"] != ROCmStateContextRegistryContract ||
		snapshot.Labels["engine_attached_drafter_route_contract"] != ROCmAttachedDrafterRegistryContract ||
		snapshot.Labels["engine_scheme_contract"] != rocmscheme.RegistryContract ||
		!strings.Contains(snapshot.Labels["engine_scheme_mixer_kinds"], "mamba2") ||
		!strings.Contains(snapshot.Labels["engine_scheme_cache_modes"], "turboquant") ||
		!strings.Contains(snapshot.Labels["engine_scheme_quant_kinds"], "q4_0") ||
		snapshot.Labels["engine_quant_scheme_contract"] != ROCmQuantSchemeRegistryContract ||
		snapshot.Labels["engine_quant_scheme_kinds"] != "affine,bf16,mxfp4,mxfp8,nvfp4,q4_0,jangtq" ||
		snapshot.Labels["engine_quant_loader_contract"] != ROCmQuantLoaderRegistryContract ||
		snapshot.Labels["architecture_resolution_contract"] != ROCmArchitectureResolutionContract ||
		snapshot.Labels["engine_profile_reactive"] != "true" ||
		snapshot.Labels["engine_registry_scope"] != "architecture_profiles" ||
		snapshot.Labels["algorithm_profile_count"] != strconv.Itoa(wantAlgorithmProfiles) ||
		snapshot.Labels["cache_mode_route_count"] != strconv.Itoa(wantCacheModeRoutes) ||
		snapshot.Labels["cache_route_count"] != strconv.Itoa(wantCacheRoutes) ||
		snapshot.Labels["feature_route_count"] != strconv.Itoa(wantFeatureRoutes) ||
		snapshot.Labels["loader_route_count"] != strconv.Itoa(wantRoutes) ||
		snapshot.Labels["diffusion_sampler_route_count"] != strconv.Itoa(wantDiffusionRoutes) ||
		snapshot.Labels["lora_adapter_route_count"] != strconv.Itoa(wantLoRARoutes) ||
		snapshot.Labels["model_loader_count"] != strconv.Itoa(wantModelLoaders) ||
		snapshot.Labels["mixer_loader_route_count"] != strconv.Itoa(wantMixerRoutes) ||
		snapshot.Labels["multimodal_processor_route_count"] != strconv.Itoa(wantMultimodalRoutes) ||
		snapshot.Labels["quant_scheme_count"] != strconv.Itoa(wantQuantSchemes) ||
		snapshot.Labels["quant_loader_route_count"] != strconv.Itoa(wantQuantRoutes) ||
		snapshot.Labels["scheme_cache_count"] != strconv.Itoa(wantSchemeCaches) ||
		snapshot.Labels["scheme_mixer_count"] != strconv.Itoa(wantSchemeMixers) ||
		snapshot.Labels["scheme_quant_count"] != strconv.Itoa(wantSchemeQuants) ||
		snapshot.Labels["state_context_route_count"] != strconv.Itoa(wantStateContextRoutes) ||
		snapshot.Labels["tokenizer_route_count"] != strconv.Itoa(wantTokenizerRoutes) ||
		snapshot.Labels["profile_count"] != strconv.Itoa(wantProfiles) ||
		snapshot.Labels["production_contract"] != "reactive-inference-v1" {
		t.Fatalf("DefaultROCmModelRegistrySnapshot = %+v, want reactive registry contract", snapshot)
	}
	if snapshot.Labels["attached_drafter_route_count"] != strconv.Itoa(wantAttachedDrafterRoutes) {
		t.Fatalf("attached drafter route count label = %q, want %d", snapshot.Labels["attached_drafter_route_count"], wantAttachedDrafterRoutes)
	}
	algorithms := map[inference.CapabilityID]ROCmAlgorithmProfile{}
	for _, profile := range snapshot.AlgorithmProfiles {
		algorithms[profile.ID] = profile
	}
	if algorithms[inference.CapabilityQuantization].Algorithm != "auto-round" ||
		algorithms[inference.CapabilityQuantization].RuntimeStatus != inference.FeatureRuntimeExperimental ||
		algorithms[inference.CapabilitySpeculativeDecode].Algorithm != "speculative-decode" ||
		!slices.Contains(algorithms[inference.CapabilitySpeculativeDecode].Provides, "mtp.attached_drafter.plan") ||
		algorithms[inference.CapabilityCacheDisk].RuntimeStatus != inference.FeatureRuntimePlanned {
		t.Fatalf("algorithm profiles = %+v, want go-mlx-style reactive algorithm matrix", snapshot.AlgorithmProfiles)
	}
	profiles := map[string]ROCmArchitectureProfile{}
	for _, profile := range snapshot.ArchitectureProfiles {
		profiles[profile.ID] = profile
	}
	if profiles["gemma4_text"].ChatTemplate != "gemma4_hf_turn" ||
		profiles["composed"].Family != "composed" ||
		profiles["composed"].Generation ||
		!slices.Contains(profiles["composed"].CacheHints, SequenceMixerCacheModeRecurrent) ||
		profiles["hybrid"].Family != "hybrid" ||
		profiles["hybrid"].Generation ||
		profiles["qwen3_6_moe"].Family != "qwen" ||
		!profiles["qwen3_6_moe"].MoE ||
		!profiles["bert_rerank"].Rerank ||
		profiles["bert_rerank"].Chat {
		t.Fatalf("snapshot architecture profiles = %+v, want broad reactive registry entries", snapshot.ArchitectureProfiles)
	}
	featureRoutes := map[string]ROCmModelFeatureRoute{}
	for _, route := range snapshot.FeatureRoutes {
		featureRoutes[route.Architecture] = route
	}
	gemmaFeatures := featureRoutes["gemma4_text"]
	if gemmaFeatures.Contract != ROCmModelFeatureRegistryContract ||
		gemmaFeatures.Family != "gemma4" ||
		gemmaFeatures.ReasoningParserID != "gemma" ||
		gemmaFeatures.ToolParserID != "gemma" ||
		gemmaFeatures.ChatTemplateID != "gemma4_hf_turn" ||
		gemmaFeatures.GenerationRole != "model" ||
		!gemmaFeatures.Registered ||
		!gemmaFeatures.NativeRuntime ||
		!gemmaFeatures.Generation ||
		!gemmaFeatures.Chat ||
		!gemmaFeatures.ChatTemplate ||
		!gemmaFeatures.DefaultThinking ||
		!gemmaFeatures.RequiresChatTemplate ||
		gemmaFeatures.TextGenerate ||
		!slices.Contains(gemmaFeatures.Capabilities, inference.CapabilityChatTemplate) ||
		gemmaFeatures.Labels["engine_feature_route_contract"] != ROCmModelFeatureRegistryContract {
		t.Fatalf("snapshot Gemma4 feature route = %+v, want parser/template route with model-owned generation metadata", gemmaFeatures)
	}
	bertRerankFeatures := featureRoutes["bert_rerank"]
	if bertRerankFeatures.Family != "bert" ||
		!bertRerankFeatures.Rerank ||
		bertRerankFeatures.Chat ||
		bertRerankFeatures.ChatTemplate ||
		!slices.Contains(bertRerankFeatures.Capabilities, inference.CapabilityRerank) {
		t.Fatalf("snapshot BERT rerank feature route = %+v, want rerank-only feature route", bertRerankFeatures)
	}
	qwenMoEFeatures := featureRoutes["qwen3_6_moe"]
	if qwenMoEFeatures.Family != "qwen" ||
		qwenMoEFeatures.ReasoningParserID != "qwen" ||
		qwenMoEFeatures.ToolParserID != "qwen" ||
		qwenMoEFeatures.ChatTemplateID != "qwen" ||
		!qwenMoEFeatures.MoE ||
		qwenMoEFeatures.Generation ||
		qwenMoEFeatures.TextGenerate ||
		!qwenMoEFeatures.ChatTemplate {
		t.Fatalf("snapshot Qwen MoE feature route = %+v, want staged parser/template/MoE route", qwenMoEFeatures)
	}
	composedFeatures := featureRoutes["composed"]
	if composedFeatures.Generation ||
		composedFeatures.Chat ||
		composedFeatures.ChatTemplate ||
		!composedFeatures.SequenceMixer ||
		composedFeatures.ReasoningParserID != "composed" ||
		composedFeatures.ToolParserID != "composed" ||
		composedFeatures.Labels["engine_feature_route_sequence_mixer"] != "true" {
		t.Fatalf("snapshot composed feature route = %+v, want non-generating composed parser route", composedFeatures)
	}
	tokenizerRoutes := map[string]ROCmModelTokenizerRoute{}
	for _, route := range snapshot.TokenizerRoutes {
		tokenizerRoutes[route.Architecture] = route
	}
	gemmaTokenizer := tokenizerRoutes["gemma4_text"]
	if gemmaTokenizer.Contract != ROCmModelTokenizerRegistryContract ||
		gemmaTokenizer.Loader != "hf-tokenizer-json" ||
		gemmaTokenizer.Runtime != "host" ||
		gemmaTokenizer.TokenizerKind != "GemmaTokenizer" ||
		gemmaTokenizer.ChatTemplateID != "gemma4_hf_turn" ||
		gemmaTokenizer.ChatTemplateSource != "registry" ||
		!gemmaTokenizer.Registered ||
		!gemmaTokenizer.NativeRuntime ||
		!gemmaTokenizer.ChatTemplate ||
		!gemmaTokenizer.RequiresChatTemplate ||
		!gemmaTokenizer.ModelOwnedTemplate ||
		gemmaTokenizer.SidecarTemplate ||
		!slices.Contains(gemmaTokenizer.RequiredFiles, "tokenizer.json") ||
		!slices.Contains(gemmaTokenizer.OptionalFiles, "tokenizer_config.json") ||
		!slices.Contains(gemmaTokenizer.Capabilities, inference.CapabilityTokenizer) ||
		!slices.Contains(gemmaTokenizer.Capabilities, inference.CapabilityChatTemplate) ||
		gemmaTokenizer.Labels["engine_tokenizer_route_contract"] != ROCmModelTokenizerRegistryContract {
		t.Fatalf("snapshot Gemma4 tokenizer route = %+v, want HF tokenizer route with registry chat template", gemmaTokenizer)
	}
	qwenTokenizer := tokenizerRoutes["qwen3_6_moe"]
	if qwenTokenizer.TokenizerKind != "Qwen2Tokenizer" ||
		qwenTokenizer.ChatTemplateID != "qwen" ||
		!qwenTokenizer.ChatTemplate ||
		qwenTokenizer.Generation ||
		qwenTokenizer.Chat {
		t.Fatalf("snapshot Qwen tokenizer route = %+v, want staged Qwen tokenizer/template route", qwenTokenizer)
	}
	bertTokenizer := tokenizerRoutes["bert_rerank"]
	if bertTokenizer.TokenizerKind != "BertTokenizer" ||
		bertTokenizer.ChatTemplate ||
		bertTokenizer.RequiresChatTemplate ||
		!slices.Contains(bertTokenizer.Capabilities, inference.CapabilityTokenizer) ||
		slices.Contains(bertTokenizer.Capabilities, inference.CapabilityChatTemplate) {
		t.Fatalf("snapshot BERT tokenizer route = %+v, want tokenizer-only rerank route", bertTokenizer)
	}
	loraRoutes := map[string]ROCmLoRAAdapterRoute{}
	for _, route := range snapshot.LoRAAdapterRoutes {
		loraRoutes[route.Architecture] = route
	}
	gemmaLoRA := loraRoutes["gemma4_text"]
	if gemmaLoRA.Contract != ROCmLoRAAdapterRegistryContract ||
		gemmaLoRA.TargetPolicy != "gemma4" ||
		!slices.Equal(gemmaLoRA.DefaultTargets, []string{"q_proj", "v_proj", "o_proj"}) ||
		!slices.Contains(gemmaLoRA.SafeTargets, "gate_proj") ||
		!slices.Contains(gemmaLoRA.ExtendedTargets, "router.proj") ||
		gemmaLoRA.TargetPaths["q_proj"] != "self_attn.q_proj" ||
		!gemmaLoRA.Registered ||
		!gemmaLoRA.NativeRuntime ||
		!gemmaLoRA.ApplySupported ||
		!gemmaLoRA.LoadSupported ||
		!gemmaLoRA.FuseSupported ||
		!gemmaLoRA.TrainingSupported ||
		!gemmaLoRA.Staged ||
		gemmaLoRA.Planned ||
		!gemmaLoRA.RequiresExtendedOptIn ||
		!slices.Contains(gemmaLoRA.Capabilities, inference.CapabilityLoRAInference) ||
		!slices.Contains(gemmaLoRA.Capabilities, inference.CapabilityLoRATraining) ||
		!slices.Contains(gemmaLoRA.Capabilities, inference.CapabilityModelMerge) ||
		gemmaLoRA.Labels["engine_lora_route_contract"] != ROCmLoRAAdapterRegistryContract {
		t.Fatalf("snapshot Gemma4 LoRA adapter route = %+v, want Gemma4 model-owned adapter target route", gemmaLoRA)
	}
	qwenLoRA := loraRoutes["qwen3_6_moe"]
	if qwenLoRA.TargetPolicy != "decoder" ||
		qwenLoRA.Family != "qwen" ||
		qwenLoRA.TargetPaths["q_proj"] != "self_attn.q_proj" ||
		qwenLoRA.TargetPaths["gate_proj"] != "mlp.gate_proj" ||
		!slices.Equal(qwenLoRA.DefaultTargets, []string{"q_proj", "v_proj"}) ||
		!qwenLoRA.Staged ||
		qwenLoRA.Planned ||
		!slices.Contains(qwenLoRA.Capabilities, inference.CapabilityLoRAInference) {
		t.Fatalf("snapshot Qwen LoRA adapter route = %+v, want staged decoder adapter route", qwenLoRA)
	}
	composedLoRA := loraRoutes["composed"]
	if composedLoRA.TargetPolicy != "composed_mlp" ||
		!slices.Equal(composedLoRA.DefaultTargets, []string{"gate_proj", "up_proj", "down_proj"}) ||
		composedLoRA.TargetPaths["gate_proj"] != "mlp.gate_proj" ||
		slices.Contains(composedLoRA.SafeTargets, "q_proj") ||
		!composedLoRA.Staged {
		t.Fatalf("snapshot composed LoRA adapter route = %+v, want model-owned MLP-only adapter route", composedLoRA)
	}
	if _, ok := loraRoutes["bert_rerank"]; ok {
		t.Fatalf("snapshot LoRA adapter routes included bert_rerank, want decoder/composed routes only: %+v", loraRoutes["bert_rerank"])
	}
	multimodalRoutes := map[string]ROCmMultimodalProcessorRoute{}
	for _, route := range snapshot.MultimodalProcessorRoutes {
		multimodalRoutes[route.Architecture] = route
	}
	gemma4Vision := multimodalRoutes["gemma4"]
	if gemma4Vision.Contract != ROCmMultimodalProcessorRegistryContract ||
		gemma4Vision.Family != "gemma4" ||
		gemma4Vision.Reference != "go_mlx_gemma4_vision" ||
		gemma4Vision.VisionRuntime != hipKernelStatusNotLinked ||
		gemma4Vision.VisionProjectorRuntime != hipKernelStatusNotLinked ||
		!gemma4Vision.Registered ||
		gemma4Vision.NativeRuntime ||
		!gemma4Vision.Multimodal ||
		!gemma4Vision.Vision ||
		!gemma4Vision.Video ||
		gemma4Vision.Audio ||
		!gemma4Vision.Projector ||
		!gemma4Vision.Planned ||
		!gemma4Vision.Staged ||
		gemma4Vision.Labels["engine_multimodal_processor_route_contract"] != ROCmMultimodalProcessorRegistryContract {
		t.Fatalf("snapshot Gemma4 multimodal processor route = %+v, want planned go-mlx vision route", gemma4Vision)
	}
	gemma4Audio := multimodalRoutes["gemma4_unified"]
	if gemma4Audio.Reference != "go_mlx_gemma4_audio" ||
		gemma4Audio.Vision ||
		!gemma4Audio.Audio ||
		gemma4Audio.AudioRuntime != hipKernelStatusNotLinked ||
		gemma4Audio.AudioProjectorRuntime != hipKernelStatusNotLinked ||
		gemma4Audio.AudioFrontEndRuntime != hipKernelStatusNotLinked {
		t.Fatalf("snapshot Gemma4 unified multimodal processor route = %+v, want planned audio route", gemma4Audio)
	}
	gemma3Vision := multimodalRoutes["gemma3"]
	if gemma3Vision.Reference != "go_mlx_gemma3_multimodal_wrapper" ||
		!gemma3Vision.Vision ||
		gemma3Vision.Audio ||
		gemma3Vision.Video {
		t.Fatalf("snapshot Gemma3 multimodal processor route = %+v, want vision wrapper route", gemma3Vision)
	}
	diffusionRoutes := map[string]ROCmDiffusionSamplerRoute{}
	for _, route := range snapshot.DiffusionSamplerRoutes {
		diffusionRoutes[route.Architecture] = route
	}
	diffusion := diffusionRoutes["diffusion_gemma"]
	if diffusion.Contract != ROCmDiffusionSamplerRegistryContract ||
		diffusion.Family != "gemma" ||
		diffusion.Reference != "go_mlx_diffusion_gemma" ||
		diffusion.DiffusionRuntime != hipKernelStatusNotLinked ||
		diffusion.SamplerRuntime != hipKernelStatusNotLinked ||
		diffusion.TrunkRuntime != "model_pack_metadata" ||
		!diffusion.Registered ||
		diffusion.NativeRuntime ||
		!diffusion.BlockDiffusion ||
		!diffusion.Sampler ||
		!diffusion.Trunk ||
		!diffusion.Generation ||
		!diffusion.SelfConditioning ||
		!diffusion.EncoderLayerScalars ||
		!diffusion.GlobalCanvasMask ||
		!diffusion.BlockLocalCanvasMask ||
		!diffusion.KVCacheRollback ||
		!diffusion.Streaming ||
		!diffusion.Planned ||
		!diffusion.Staged ||
		!diffusion.FallbackRefused ||
		diffusion.DefaultCanvasLength != 64 ||
		diffusion.ReferenceCanvasLength != 256 ||
		diffusion.DefaultMaxSteps != 16 ||
		diffusion.ReferenceMaxSteps != 48 ||
		diffusion.EntropyBound != 0.3 ||
		diffusion.MaxTemperature != 0.8 ||
		diffusion.MinTemperature != 0.4 ||
		diffusion.Labels["engine_diffusion_sampler_route_contract"] != ROCmDiffusionSamplerRegistryContract {
		t.Fatalf("snapshot DiffusionGemma sampler route = %+v, want planned go-mlx block diffusion sampler route", diffusion)
	}
	stateRoutes := map[string]ROCmStateContextRoute{}
	for _, route := range snapshot.StateContextRoutes {
		stateRoutes[route.Architecture] = route
	}
	gemmaState := stateRoutes["gemma4_text"]
	if gemmaState.Contract != ROCmStateContextRegistryContract ||
		gemmaState.Family != "gemma4" ||
		gemmaState.Reference != "go_mlx_gemma4_retained_state" ||
		!gemmaState.Registered ||
		!gemmaState.NativeRuntime ||
		!gemmaState.StateSession ||
		!gemmaState.SleepState ||
		!gemmaState.WakeState ||
		!gemmaState.ForkState ||
		!gemmaState.CaptureState ||
		!gemmaState.RestoreState ||
		!gemmaState.ResetState ||
		!gemmaState.RuntimeOwnedKV ||
		!gemmaState.PromptReplayRefused ||
		!gemmaState.RemainingContextDefault ||
		!gemmaState.ModelContextWindow ||
		!gemmaState.DeviceKVState ||
		!gemmaState.HIPDeviceMirror ||
		!gemmaState.PackageLocalKV ||
		!gemmaState.BlockBundleRefs ||
		!gemmaState.PortableRefs ||
		!gemmaState.RetainedStateRequired ||
		gemmaState.DefaultDeviceKVMode != "k-q8-v-q4" ||
		gemmaState.DefaultStateBlockSize != 128 ||
		!slices.Contains(gemmaState.CacheModes, "retained-state") ||
		!slices.Contains(gemmaState.Capabilities, inference.CapabilityStateWake) ||
		!slices.Contains(gemmaState.Capabilities, inference.CapabilityStateSleep) ||
		!slices.Contains(gemmaState.Capabilities, inference.CapabilityStateFork) ||
		gemmaState.Labels["engine_state_context_route_contract"] != ROCmStateContextRegistryContract ||
		gemmaState.Labels["engine_state_context_prompt_replay_refused"] != "true" {
		t.Fatalf("snapshot Gemma4 state context route = %+v, want retained-state runtime route", gemmaState)
	}
	assistantState := stateRoutes["gemma4_assistant"]
	if assistantState.Contract != ROCmStateContextRegistryContract ||
		!assistantState.AttachedOnly ||
		!assistantState.AttachedDrafterState ||
		assistantState.Reference != "go_mlx_gemma4_attached_drafter_retained_state" ||
		!slices.Contains(assistantState.CacheModes, "attached-drafter") {
		t.Fatalf("snapshot Gemma4 assistant state context route = %+v, want attached-drafter retained route", assistantState)
	}
	attachedRoutes := map[string]ROCmAttachedDrafterRoute{}
	for _, route := range snapshot.AttachedDrafterRoutes {
		attachedRoutes[route.Architecture] = route
	}
	attached := attachedRoutes["gemma4_text"]
	if attached.Contract != ROCmAttachedDrafterRegistryContract ||
		attached.Family != "gemma4" ||
		attached.Reference != "go_mlx_gemma4_assistant_pair" ||
		attached.Mode != "mtp_attached_drafter" ||
		attached.Role != "target" ||
		attached.TargetArchitecture != "gemma4_text" ||
		attached.AssistantArchitecture != "gemma4_assistant" ||
		attached.NativeAttachment != hipKernelStatusNotLinked ||
		attached.ExecutionStatus != hipKernelStatusNotLinked ||
		attached.Fallback != "refused" ||
		!attached.Registered ||
		attached.NativeRuntime ||
		!attached.Target ||
		attached.Assistant ||
		attached.AttachedOnly ||
		!attached.PairValidation ||
		!attached.FamilyPairRequired ||
		!attached.OfficialPairKnown ||
		!attached.OfficialPairLocked ||
		!attached.SameSizeRequired ||
		!attached.SameTokenizerRequired ||
		attached.HiddenSizeMatchRequired ||
		!attached.VocabMatchRequired ||
		!attached.LayerTypeMatchRequired ||
		!attached.RetainedStateRequired ||
		!attached.RuntimeOwnedKV ||
		!attached.PromptReplayRefused ||
		!attached.DraftDetection ||
		!attached.AutoDetectAssistantDir ||
		!attached.AutoDetectSiblingPair ||
		!attached.AutoDetectMTPDir ||
		!attached.AutoDetectMTPSiblingGGUF ||
		!attached.TuneProfile ||
		!attached.FourLayerDrafter ||
		!attached.OrderedEmbeddings ||
		!attached.CentroidRouting ||
		!attached.BorrowTargetKV ||
		!attached.VerifyForward ||
		attached.NativeGeneration ||
		attached.NativeStateGeneration ||
		!attached.FallbackRefused ||
		!attached.Staged ||
		!attached.Planned ||
		attached.DefaultDraftTokens != ProductionMTPDefaultDraftTokens ||
		attached.DefaultDraftBlock != 5 ||
		attached.MinimumRetainedTurns <= 0 ||
		attached.AssistantCentroids != 2048 ||
		attached.AssistantCentroidIntermediateTopK != 32 ||
		!slices.Equal(attached.AssistantTokenOrderingShape, []int{2048, 128}) ||
		!slices.Contains(attached.TargetSizes, "26B-A4B") ||
		!slices.Contains(attached.TargetSizes, "31B") ||
		!slices.Contains(attached.TargetQuantModes, "q6") ||
		!slices.Contains(attached.TargetQuantModes, "mxfp4") ||
		!slices.Contains(attached.AssistantQuantModes, "bf16") ||
		!slices.Contains(attached.AssistantQuantModes, "q4") ||
		!slices.Contains(attached.AssistantModelIDs, "google/gemma-4-12B-it-assistant") ||
		!slices.Contains(attached.AssistantModelIDs, "mlx-community/gemma-4-12B-it-qat-assistant-4bit") ||
		!slices.Contains(attached.DetectionSources, string(DraftSourceAssistantDir)) ||
		!slices.Contains(attached.RequiredDraftTokenSweeps, 4) ||
		!slices.Contains(attached.TunableDraftBlocks, 5) ||
		!slices.Contains(attached.RequiredMetrics, "attached_drafter_retained_state_required") ||
		!slices.Contains(attached.Capabilities, inference.CapabilitySpeculativeDecode) ||
		!slices.Contains(attached.Capabilities, inference.CapabilityStateWake) ||
		attached.Labels["engine_attached_drafter_route_contract"] != ROCmAttachedDrafterRegistryContract ||
		attached.Labels["engine_attached_drafter_native_attachment"] != hipKernelStatusNotLinked ||
		attached.Labels["engine_attached_drafter_hidden_size_match_required"] != "false" ||
		attached.Labels["engine_attached_drafter_prompt_replay_refused"] != "true" {
		t.Fatalf("snapshot Gemma4 attached drafter route = %+v, want reactive native-pending MTP route", attached)
	}
	assistantDrafter := attachedRoutes["gemma4_assistant"]
	if assistantDrafter.Contract != ROCmAttachedDrafterRegistryContract ||
		assistantDrafter.Role != "assistant" ||
		!assistantDrafter.Assistant ||
		!assistantDrafter.AttachedOnly ||
		assistantDrafter.Target ||
		assistantDrafter.Status != ROCmAttachedDrafterRouteAttachedOnly ||
		assistantDrafter.AssistantArchitecture != "gemma4_assistant" ||
		assistantDrafter.TargetArchitecture != "gemma4_text" ||
		assistantDrafter.NativeAttachment != hipKernelStatusNotLinked ||
		assistantDrafter.StandaloneGeneration {
		t.Fatalf("snapshot Gemma4 assistant drafter route = %+v, want attach-only MTP assistant route", assistantDrafter)
	}
	routes := map[string]ROCmModelLoaderRoute{}
	for _, route := range snapshot.LoaderRoutes {
		routes[route.Architecture] = route
	}
	if routes["gemma4_text"].Contract != ROCmModelLoaderRegistryContract ||
		routes["gemma4_text"].Loader != "gemma4_text" ||
		routes["gemma4_text"].Runtime != "hip" ||
		routes["gemma4_text"].Status != ROCmModelLoadStandaloneNative ||
		!routes["gemma4_text"].Registered ||
		routes["gemma4_text"].Staged ||
		!routes["gemma4_text"].Standalone ||
		!routes["gemma4_text"].TextGenerate ||
		routes["gemma4_assistant"].Status != ROCmModelLoadAttachedOnly ||
		routes["gemma4_assistant"].Target != "attached" ||
		!routes["gemma4_assistant"].AttachedOnly ||
		routes["composed"].Status != ROCmModelLoadStagedNative ||
		routes["composed"].Runtime != "hip" ||
		!routes["composed"].Registered ||
		!routes["composed"].Staged ||
		routes["composed"].TextGenerate ||
		routes["hybrid"].Status != ROCmModelLoadStagedNative ||
		routes["qwen3_6_moe"].Family != "qwen" ||
		routes["bert_rerank"].Family != "bert" {
		t.Fatalf("snapshot loader routes = %+v, want architecture-keyed loader route registry", snapshot.LoaderRoutes)
	}
	cacheModeRoutes := map[string]ROCmCacheModeRoute{}
	for _, route := range snapshot.CacheModeRoutes {
		cacheModeRoutes[route.Mode] = route
	}
	if cacheModeRoutes[ROCmCacheModeQ8].Runtime != ROCmCacheRuntimeHIP ||
		!cacheModeRoutes[ROCmCacheModeQ8].NativeKV ||
		!cacheModeRoutes[ROCmCacheModeQ8].DeviceKV ||
		!cacheModeRoutes[ROCmCacheModeQ8].Quantized ||
		cacheModeRoutes[ROCmCacheModeRecurrent].Runtime != ROCmCacheRuntimeMetadata ||
		!cacheModeRoutes[ROCmCacheModeRecurrent].Recurrent ||
		cacheModeRoutes[ROCmCacheModeQ8].Labels["engine_cache_mode"] != ROCmCacheModeQ8 {
		t.Fatalf("snapshot cache mode routes = %+v, want cache factory mode registry", snapshot.CacheModeRoutes)
	}
	cacheRoutes := map[string]ROCmCacheRoute{}
	for _, route := range snapshot.CacheRoutes {
		cacheRoutes[route.Architecture] = route
	}
	if !cacheRoutes["gemma4_text"].Matched() ||
		cacheRoutes["gemma4_text"].Contract != ROCmCacheFactoryRouteContract ||
		cacheRoutes["gemma4_text"].Name != ROCmCacheFactoryRouteName ||
		cacheRoutes["gemma4_text"].Family != "gemma4" ||
		!cacheRoutes["gemma4_text"].SupportsKV ||
		!cacheRoutes["gemma4_text"].SupportsDevice ||
		cacheRoutes["gemma4_text"].Labels["engine_cache_factory_contract"] != ROCmCacheFactoryRouteContract ||
		!slices.Contains(cacheRoutes["composed"].CacheHints, ROCmCacheModeRecurrent) ||
		!cacheRoutes["composed"].SupportsRecurrent {
		t.Fatalf("snapshot cache routes = %+v, want architecture-keyed cache factory routes", snapshot.CacheRoutes)
	}
	quantSchemes := map[string]ROCmQuantScheme{}
	for _, scheme := range snapshot.QuantSchemes {
		quantSchemes[scheme.Kind] = scheme
	}
	if len(quantSchemes) != len(DefaultROCmQuantSchemes()) ||
		quantSchemes["affine"].Contract != ROCmQuantSchemeRegistryContract ||
		quantSchemes["affine"].Bits != 0 ||
		quantSchemes["affine"].Loader != "gemma4_affine" ||
		quantSchemes["affine"].Runtime != Gemma4RuntimeMLXAffine ||
		!quantSchemes["affine"].Registered ||
		!quantSchemes["affine"].NativeRuntime ||
		quantSchemes["affine"].Labels["engine_quant_scheme_contract"] != ROCmQuantSchemeRegistryContract ||
		quantSchemes["mxfp4"].RuntimeStatus != inference.FeatureRuntimePlanned ||
		!quantSchemes["mxfp4"].Planned ||
		quantSchemes["q4_0"].RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		!quantSchemes["q4_0"].MetadataOnly {
		t.Fatalf("snapshot quant schemes = %+v, want go-mlx-style quant scheme catalogue", snapshot.QuantSchemes)
	}
	if alias, ok := ROCmQuantSchemeForKind("mlx_q6"); !ok || alias.Kind != "affine" {
		t.Fatalf("ROCmQuantSchemeForKind(mlx_q6) = %+v ok=%v, want affine alias", alias, ok)
	}
	if alias, ok := ROCmQuantSchemeForKind("mxtq"); !ok || alias.Kind != "jangtq" {
		t.Fatalf("ROCmQuantSchemeForKind(mxtq) = %+v ok=%v, want jangtq alias", alias, ok)
	}
	quantRoutes := map[string]ROCmQuantLoaderRoute{}
	for _, route := range snapshot.QuantLoaderRoutes {
		quantRoutes[route.Size+":"+route.Mode] = route
	}
	q6 := quantRoutes["E2B:q6"]
	if q6.Contract != ROCmQuantLoaderRegistryContract ||
		q6.Loader != "gemma4_affine" ||
		q6.Runtime != Gemma4RuntimeMLXAffine ||
		q6.GenerateStatus != Gemma4GenerateLinked ||
		q6.Target != "generate" ||
		!q6.Registered ||
		!q6.NativeRuntime ||
		!q6.RunnableOnCard ||
		q6.Planned ||
		q6.LoadOnly ||
		q6.Labels["engine_quant_loader_contract"] != ROCmQuantLoaderRegistryContract {
		t.Fatalf("snapshot q6 quant loader route = %+v, want linked affine quant loader", q6)
	}
	bf16 := quantRoutes["E2B:bf16"]
	if bf16.Loader != "gemma4_bf16" ||
		bf16.Runtime != Gemma4RuntimeBF16 ||
		bf16.GenerateStatus != Gemma4GenerateLoadOnly ||
		bf16.Target != "load" ||
		!bf16.LoadOnly ||
		!bf16.Staged ||
		!bf16.NativeRuntime ||
		!bf16.RequiresNative {
		t.Fatalf("snapshot bf16 quant loader route = %+v, want native load-only quant loader", bf16)
	}
	mxfp4 := quantRoutes["E2B:mxfp4"]
	if mxfp4.Loader != "gemma4_mxfp4" ||
		mxfp4.Runtime != Gemma4RuntimePlanned ||
		mxfp4.GenerateStatus != Gemma4GeneratePlannedOnly ||
		mxfp4.Target != "planned" ||
		!mxfp4.Planned ||
		!mxfp4.Staged ||
		mxfp4.NativeRuntime ||
		!mxfp4.RequiresBench {
		t.Fatalf("snapshot mxfp4 quant loader route = %+v, want planned quant loader", mxfp4)
	}
	statusOnly := quantRoutes["31B:q4-status"]
	if statusOnly.Loader != "gemma4_status" ||
		statusOnly.Target != "metadata" ||
		statusOnly.RunnableOnCard ||
		!statusOnly.Planned {
		t.Fatalf("snapshot 31B status quant loader route = %+v, want metadata-only status route", statusOnly)
	}
	mixerRoutes := map[string]ROCmSequenceMixerLoaderRoute{}
	for _, route := range snapshot.MixerLoaderRoutes {
		mixerRoutes[route.Kind] = route
	}
	full := mixerRoutes["full_attention"]
	if full.Contract != ROCmSequenceMixerLoaderRegistryContract ||
		full.Loader != "full_attention" ||
		full.State != SequenceMixerStateKVCache ||
		full.CacheMode != SequenceMixerCacheModeDefault ||
		full.Source != "generic_softmax" ||
		full.Runtime != SequenceMixerRuntimePlannedHIP ||
		!full.Registered ||
		full.NativeRuntime ||
		!full.Planned ||
		!slices.Contains(full.RequiredLeaves, "q_proj.weight") ||
		full.Labels["engine_mixer_loader_contract"] != ROCmSequenceMixerLoaderRegistryContract {
		t.Fatalf("snapshot full_attention mixer route = %+v, want go-mlx softmax mixer loader route", full)
	}
	mamba := mixerRoutes["mamba2"]
	if mamba.State != SequenceMixerStateRecurrent ||
		mamba.CacheMode != SequenceMixerCacheModeRecurrent ||
		mamba.Source != "fla" ||
		!slices.Equal(mamba.StateSlots, []string{"conv_state", "ssm_state"}) ||
		!slices.Contains(mamba.RequiredLeaves, "conv1d.weight") ||
		mamba.Labels["engine_mixer_loader_state_slots"] != "conv_state,ssm_state" ||
		mamba.Labels["engine_mixer_loader_state_slot_count"] != "2" {
		t.Fatalf("snapshot mamba2 mixer route = %+v, want recurrent FLA mixer loader route", mamba)
	}
	gsa := mixerRoutes["gsa"]
	if gsa.State != SequenceMixerStateRecurrent ||
		gsa.CacheMode != SequenceMixerCacheModeRecurrent ||
		!slices.Equal(gsa.StateSlots, []string{"slot_key_state", "slot_value_state"}) ||
		!slices.Contains(gsa.RequiredLeaves, "f_proj.weight") {
		t.Fatalf("snapshot GSA mixer route = %+v, want recurrent slot-memory mixer loader route", gsa)
	}
	rwkv7 := mixerRoutes["rwkv7"]
	if rwkv7.State != SequenceMixerStateRecurrent ||
		!slices.Equal(rwkv7.StateSlots, []string{"wkv_state"}) ||
		!slices.Contains(rwkv7.RequiredLeaves, "decay.weight") {
		t.Fatalf("snapshot RWKV7 mixer route = %+v, want recurrent WKV state mixer loader route", rwkv7)
	}
	mla := mixerRoutes["mla"]
	if mla.State != SequenceMixerStateKVCache ||
		mla.CacheMode != SequenceMixerCacheModeMLALatent ||
		mla.Labels["engine_mixer_loader_cache_mode"] != SequenceMixerCacheModeMLALatent {
		t.Fatalf("snapshot MLA mixer route = %+v, want latent-cache mixer loader route", mla)
	}

	snapshot.Factories[0] = "mutated"
	snapshot.ArchitectureProfiles[0].QuantizationHints[0] = "mutated"
	snapshot.FeatureRoutes[0].Labels["engine_feature_route"] = "mutated"
	snapshot.FeatureRoutes[0].Capabilities[0] = inference.CapabilityGenerate
	snapshot.TokenizerRoutes[0].Labels["engine_tokenizer_loader"] = "mutated"
	snapshot.TokenizerRoutes[0].RequiredFiles[0] = "mutated"
	snapshot.TokenizerRoutes[0].Capabilities[0] = inference.CapabilityGenerate
	snapshot.TokenizerRoutes[0].Tokenizer.Labels = map[string]string{"mutated": "true"}
	snapshot.LoRAAdapterRoutes[0].Labels["engine_lora_loader"] = "mutated"
	snapshot.LoRAAdapterRoutes[0].DefaultTargets[0] = "mutated"
	snapshot.LoRAAdapterRoutes[0].TargetPaths["q_proj"] = "mutated"
	snapshot.LoRAAdapterRoutes[0].Capabilities[0] = inference.CapabilityGenerate
	snapshot.MultimodalProcessorRoutes[0].Labels["engine_multimodal_processor_route"] = "mutated"
	snapshot.MultimodalProcessorRoutes[0].RequiredFiles[0] = "mutated"
	snapshot.MultimodalProcessorRoutes[0].OptionalFiles[0] = "mutated"
	snapshot.DiffusionSamplerRoutes[0].Labels["engine_diffusion_sampler_route"] = "mutated"
	snapshot.DiffusionSamplerRoutes[0].RequiredFiles[0] = "mutated"
	snapshot.DiffusionSamplerRoutes[0].OptionalFiles[0] = "mutated"
	snapshot.DiffusionSamplerRoutes[0].RequiredWeightLeaves[0] = "mutated"
	snapshot.DiffusionSamplerRoutes[0].OptionalWeightPrefixes[0] = "mutated"
	snapshot.StateContextRoutes[0].Labels["engine_state_context_route"] = "mutated"
	snapshot.StateContextRoutes[0].CacheModes[0] = "mutated"
	snapshot.StateContextRoutes[0].StateBackends[0] = "mutated"
	snapshot.StateContextRoutes[0].Capabilities[0] = inference.CapabilityGenerate
	snapshot.AttachedDrafterRoutes[0].Labels["engine_attached_drafter_route"] = "mutated"
	snapshot.AttachedDrafterRoutes[0].AssistantTokenOrderingShape[0] = 1
	snapshot.AttachedDrafterRoutes[0].TargetSizes[0] = "mutated"
	snapshot.AttachedDrafterRoutes[0].AssistantModelIDs[0] = "mutated"
	snapshot.AttachedDrafterRoutes[0].DetectionSources[0] = "mutated"
	snapshot.AttachedDrafterRoutes[0].RequiredDraftTokenSweeps[0] = 99
	snapshot.AttachedDrafterRoutes[0].TunableDraftBlocks[0] = 99
	snapshot.AttachedDrafterRoutes[0].RequiredMetrics[0] = "mutated"
	snapshot.AttachedDrafterRoutes[0].Capabilities[0] = inference.CapabilityGenerate
	snapshot.LoaderRoutes[0].Labels["engine_loader"] = "mutated"
	snapshot.CacheModeRoutes[0].Labels["engine_cache_mode"] = "mutated"
	snapshot.CacheRoutes[0].Labels["engine_cache_factory_route"] = "mutated"
	snapshot.CacheRoutes[0].Modes[0].Labels["engine_cache_mode"] = "mutated"
	snapshot.QuantSchemes[0].Labels["engine_quant_scheme"] = "mutated"
	snapshot.QuantLoaderRoutes[0].Labels["engine_quant_loader"] = "mutated"
	snapshot.MixerLoaderRoutes[0].Labels["engine_mixer_loader"] = "mutated"
	snapshot.MixerLoaderRoutes[0].RequiredLeaves[0] = "mutated"
	if len(snapshot.MixerLoaderRoutes[1].StateSlots) > 0 {
		snapshot.MixerLoaderRoutes[1].StateSlots[0] = "mutated"
	}
	snapshot.Labels["engine_registry"] = "mutated"
	next := DefaultROCmModelRegistrySnapshot("")
	if next.Backend != "rocm" ||
		next.Factories[0] == "mutated" ||
		next.Factories[1] != "architecture-profile" ||
		next.ArchitectureProfiles[0].QuantizationHints[0] == "mutated" ||
		next.FeatureRoutes[0].Labels["engine_feature_route"] == "mutated" ||
		next.FeatureRoutes[0].Capabilities[0] == inference.CapabilityGenerate ||
		next.TokenizerRoutes[0].Labels["engine_tokenizer_loader"] == "mutated" ||
		next.TokenizerRoutes[0].RequiredFiles[0] == "mutated" ||
		next.TokenizerRoutes[0].Capabilities[0] == inference.CapabilityGenerate ||
		next.TokenizerRoutes[0].Tokenizer.Labels["mutated"] == "true" ||
		next.LoRAAdapterRoutes[0].Labels["engine_lora_loader"] == "mutated" ||
		next.LoRAAdapterRoutes[0].DefaultTargets[0] == "mutated" ||
		next.LoRAAdapterRoutes[0].TargetPaths["q_proj"] == "mutated" ||
		next.LoRAAdapterRoutes[0].Capabilities[0] == inference.CapabilityGenerate ||
		next.MultimodalProcessorRoutes[0].Labels["engine_multimodal_processor_route"] == "mutated" ||
		next.MultimodalProcessorRoutes[0].RequiredFiles[0] == "mutated" ||
		next.MultimodalProcessorRoutes[0].OptionalFiles[0] == "mutated" ||
		next.DiffusionSamplerRoutes[0].Labels["engine_diffusion_sampler_route"] == "mutated" ||
		next.DiffusionSamplerRoutes[0].RequiredFiles[0] == "mutated" ||
		next.DiffusionSamplerRoutes[0].OptionalFiles[0] == "mutated" ||
		next.DiffusionSamplerRoutes[0].RequiredWeightLeaves[0] == "mutated" ||
		next.DiffusionSamplerRoutes[0].OptionalWeightPrefixes[0] == "mutated" ||
		next.StateContextRoutes[0].Labels["engine_state_context_route"] == "mutated" ||
		next.StateContextRoutes[0].CacheModes[0] == "mutated" ||
		next.StateContextRoutes[0].StateBackends[0] == "mutated" ||
		next.StateContextRoutes[0].Capabilities[0] == inference.CapabilityGenerate ||
		next.AttachedDrafterRoutes[0].Labels["engine_attached_drafter_route"] == "mutated" ||
		next.AttachedDrafterRoutes[0].AssistantTokenOrderingShape[0] == 1 ||
		next.AttachedDrafterRoutes[0].TargetSizes[0] == "mutated" ||
		next.AttachedDrafterRoutes[0].AssistantModelIDs[0] == "mutated" ||
		next.AttachedDrafterRoutes[0].DetectionSources[0] == "mutated" ||
		next.AttachedDrafterRoutes[0].RequiredDraftTokenSweeps[0] == 99 ||
		next.AttachedDrafterRoutes[0].TunableDraftBlocks[0] == 99 ||
		next.AttachedDrafterRoutes[0].RequiredMetrics[0] == "mutated" ||
		next.AttachedDrafterRoutes[0].Capabilities[0] == inference.CapabilityGenerate ||
		next.LoaderRoutes[0].Labels["engine_loader"] == "mutated" ||
		next.CacheModeRoutes[0].Labels["engine_cache_mode"] == "mutated" ||
		next.CacheRoutes[0].Labels["engine_cache_factory_route"] == "mutated" ||
		next.CacheRoutes[0].Modes[0].Labels["engine_cache_mode"] == "mutated" ||
		next.QuantSchemes[0].Labels["engine_quant_scheme"] == "mutated" ||
		next.QuantLoaderRoutes[0].Labels["engine_quant_loader"] == "mutated" ||
		next.MixerLoaderRoutes[0].Labels["engine_mixer_loader"] == "mutated" ||
		next.MixerLoaderRoutes[0].RequiredLeaves[0] == "mutated" ||
		next.MixerLoaderRoutes[1].StateSlots[0] == "mutated" ||
		next.Labels["engine_registry"] == "mutated" {
		t.Fatalf("DefaultROCmModelRegistrySnapshot returned mutable defaults: %+v", next)
	}
}

func TestROCmSequenceMixerLoaderRouteForKind_Good_MatchesFactoryContract(t *testing.T) {
	route, ok := ROCmSequenceMixerLoaderRouteForKind("mla")
	if !ok ||
		route.Contract != ROCmSequenceMixerLoaderRegistryContract ||
		route.Kind != "mla" ||
		route.Loader != "mla" ||
		route.State != SequenceMixerStateKVCache ||
		route.CacheMode != SequenceMixerCacheModeMLALatent ||
		route.Runtime != SequenceMixerRuntimePlannedHIP ||
		!route.Registered ||
		route.NativeRuntime ||
		!route.Planned ||
		!slices.Contains(route.RequiredLeaves, "kv_a_proj_with_mqa.weight") {
		t.Fatalf("ROCmSequenceMixerLoaderRouteForKind(mla) = %+v ok=%v, want latent-cache mixer route", route, ok)
	}
	gsa, ok := ROCmSequenceMixerLoaderRouteForKind("gsa")
	if !ok ||
		gsa.State != SequenceMixerStateRecurrent ||
		gsa.CacheMode != SequenceMixerCacheModeRecurrent ||
		!slices.Equal(gsa.StateSlots, []string{"slot_key_state", "slot_value_state"}) ||
		gsa.Labels["engine_mixer_state_slots_contract"] != SequenceMixerStateSlotsContract ||
		gsa.Labels["engine_mixer_loader_state_slots"] != "slot_key_state,slot_value_state" {
		t.Fatalf("ROCmSequenceMixerLoaderRouteForKind(gsa) = %+v ok=%v, want recurrent state-slot route", gsa, ok)
	}
	route.RequiredLeaves[0] = "mutated"
	gsa.StateSlots[0] = "mutated"
	next, ok := ROCmSequenceMixerLoaderRouteForKind("mla")
	if !ok || next.RequiredLeaves[0] == "mutated" {
		t.Fatalf("ROCmSequenceMixerLoaderRouteForKind leaked mutable required leaves: %+v", next)
	}
	nextGSA, ok := ROCmSequenceMixerLoaderRouteForKind("gsa")
	if !ok || nextGSA.StateSlots[0] == "mutated" {
		t.Fatalf("ROCmSequenceMixerLoaderRouteForKind leaked mutable state slots: %+v", nextGSA)
	}
	if route, ok := ROCmSequenceMixerLoaderRouteForKind("nope"); ok || route.Matched() {
		t.Fatalf("ROCmSequenceMixerLoaderRouteForKind(nope) = %+v ok=%v, want empty false", route, ok)
	}
}

func TestProbeROCmModelConfig_Good_ModelLoaderDispatchContract(t *testing.T) {
	probe, err := ProbeROCmModelConfig([]byte(`{
		"model_type":"gemma4",
		"architectures":["Gemma4ForConditionalGeneration"],
		"text_config":{"model_type":"gemma4_text"}
	}`))
	if err != nil {
		t.Fatalf("ProbeROCmModelConfig(gemma4) error = %v", err)
	}
	if probe.Contract != ROCmModelConfigProbeContract ||
		probe.ArchitectureResolution.Contract != ROCmArchitectureResolutionContract ||
		probe.ArchitectureResolution.Architecture != "gemma4_text" ||
		probe.ArchitectureResolution.Source != "model_type_text_tower" ||
		probe.LoaderRoute.Contract != ROCmModelLoaderRegistryContract ||
		probe.LoaderRoute.Loader != "gemma4_text" ||
		probe.LoaderRoute.Runtime != "hip" ||
		probe.LoaderRoute.Status != ROCmModelLoadStandaloneNative ||
		probe.RuntimeContractRoute.Contract != ROCmModelRuntimeContractRegistryContract ||
		!probe.RuntimeContractRoute.FixedSlidingCache ||
		!probe.RuntimeContractRoute.ModelInfoReporter ||
		!probe.Registered ||
		!probe.Standalone ||
		!probe.TextGenerate ||
		probe.AttachedOnly ||
		probe.Labels["engine_config_probe_contract"] != ROCmModelConfigProbeContract ||
		probe.Labels["engine_config_architecture_resolved"] != "gemma4_text" ||
		probe.Labels["engine_config_loader"] != "gemma4_text" ||
		probe.Labels["engine_config_runtime_contract"] != "true" ||
		probe.Labels["engine_runtime_contract_fixed_sliding_cache"] != "true" {
		t.Fatalf("ProbeROCmModelConfig(gemma4) = %+v, want go-mlx-style config probe to loader/runtime routes", probe)
	}

	assistant, err := ProbeROCmModelConfig([]byte(`{"model_type":"gemma4_assistant","architectures":["Gemma4AssistantForCausalLM"]}`))
	if err != nil {
		t.Fatalf("ProbeROCmModelConfig(assistant) error = %v", err)
	}
	if assistant.ArchitectureResolution.Architecture != "gemma4_assistant" ||
		assistant.LoaderRoute.Status != ROCmModelLoadAttachedOnly ||
		!assistant.RuntimeContractRoute.DecodeUnavailableReporter ||
		!assistant.AttachedOnly ||
		assistant.Standalone {
		t.Fatalf("assistant probe = %+v, want attached-only loader route", assistant)
	}

	qwen36MoE, err := ProbeROCmModelConfig([]byte(`{"model_type":"qwen3_6_moe","architectures":["Qwen3.6MoeForConditionalGeneration"]}`))
	if err != nil {
		t.Fatalf("ProbeROCmModelConfig(qwen3_6_moe) error = %v", err)
	}
	if qwen36MoE.ArchitectureResolution.Architecture != "qwen3_6_moe" ||
		!qwen36MoE.RuntimeContractRoute.MoETextRuntimeReporter ||
		!qwen36MoE.RuntimeContractRoute.HybridAttentionCachePlanner ||
		qwen36MoE.RuntimeContractRoute.TextGenerate ||
		qwen36MoE.Labels["engine_runtime_contract_hybrid_attention_cache_planner"] != "true" {
		t.Fatalf("qwen3_6_moe probe = %+v, want staged hybrid/MoE runtime contract route", qwen36MoE)
	}
}

func TestProbeROCmModelConfig_Good_ComposedSequenceMixerContract(t *testing.T) {
	probe, err := ProbeROCmModelConfig([]byte(`{
		"model_type":"composed",
		"num_hidden_layers":2,
		"hidden_size":4,
		"layer_types":["full_attention","mamba2"]
	}`))
	if err != nil {
		t.Fatalf("ProbeROCmModelConfig(composed) error = %v", err)
	}
	if probe.ArchitectureResolution.Architecture != "composed" ||
		probe.LoaderRoute.Status != ROCmModelLoadStagedNative ||
		!probe.Registered ||
		!probe.Staged ||
		probe.TextGenerate ||
		!probe.ConfigComposed ||
		probe.SequenceMixerPlanStatus != "valid" ||
		len(probe.SequenceMixerLayers) != 2 ||
		probe.SequenceMixerLayers[0].Kind != "full_attention" ||
		probe.SequenceMixerLayers[1].Kind != "mamba2" ||
		probe.SequenceMixerCache.Layers[0].Mode != SequenceMixerCacheModeDefault ||
		probe.SequenceMixerCache.Layers[1].Mode != SequenceMixerCacheModeRecurrent ||
		!probe.RuntimeContractRoute.DecodeUnavailableReporter ||
		probe.Labels["sequence_mixer_config_plan_status"] != "valid" ||
		probe.Labels["sequence_mixer_cache_plan_layers"] != "2" ||
		probe.Labels["engine_config_runtime_contract"] != "true" {
		t.Fatalf("composed probe = %+v, want registered composed loader and cache factory plan", probe)
	}

	hybrid, err := ProbeROCmModelConfig([]byte(`{"model_type":"hybrid","num_hidden_layers":2}`))
	if err != nil {
		t.Fatalf("ProbeROCmModelConfig(hybrid) error = %v", err)
	}
	if hybrid.ArchitectureResolution.Architecture != "hybrid" ||
		hybrid.SequenceMixerPlanStatus != "invalid" ||
		hybrid.SequenceMixerPlanError == "" ||
		hybrid.Labels["sequence_mixer_config_plan_status"] != "invalid" {
		t.Fatalf("hybrid probe = %+v, want loud missing layer_types refusal", hybrid)
	}
}

func TestROCmArchitectureProfileAccessors_Good_GenericRegistryContract(t *testing.T) {
	wantDefaults := []string{"q_proj", "v_proj", "o_proj"}
	defaults := ROCmLoRADefaultTargets("Gemma4ForConditionalGeneration")
	if !slices.Equal(defaults, wantDefaults) {
		t.Fatalf("ROCmLoRADefaultTargets = %v, want %v", defaults, wantDefaults)
	}
	defaults[0] = "mutated"
	if next := ROCmLoRADefaultTargets("gemma4_text"); !slices.Equal(next, wantDefaults) {
		t.Fatalf("ROCmLoRADefaultTargets returned mutable defaults: %v", next)
	}

	policy, ok := ROCmLoRATargetPolicyForArchitecture("gemma4_unified")
	if !ok ||
		!slices.Equal(policy.DefaultTargets, wantDefaults) ||
		!slices.Contains(policy.SafeTargets, "gate_proj") ||
		!slices.Contains(policy.ExtendedTargets, "router.proj") ||
		policy.TargetPaths["q_proj"] != "self_attn.q_proj" {
		t.Fatalf("ROCmLoRATargetPolicyForArchitecture = %+v ok=%v, want registry-backed Gemma4 LoRA policy", policy, ok)
	}
	policy.TargetPaths["q_proj"] = "mutated"
	policy, _ = ROCmLoRATargetPolicyForArchitecture("gemma4_unified")
	if policy.TargetPaths["q_proj"] != "self_attn.q_proj" {
		t.Fatalf("ROCmLoRATargetPolicyForArchitecture returned mutable path map: %+v", policy.TargetPaths)
	}

	if path, ok := ROCmLoRATargetPath("gemma4_text", "mlp.up_proj"); !ok || path != "mlp.up_proj" {
		t.Fatalf("ROCmLoRATargetPath = %q ok=%v, want mlp.up_proj true", path, ok)
	}
	if !ROCmLoRASafeTarget("gemma4_text", "gate_proj") ||
		ROCmLoRASafeTarget("gemma4_text", "router.proj") ||
		!ROCmLoRAExtendedTarget("gemma4_text", "router.proj") {
		t.Fatalf("ROCm LoRA target safety mismatch")
	}
	canonical, ok := ROCmLoRACanonicalTarget("gemma4_text", "model.layers.0.q_proj")
	if !ok || canonical != "model.layers.0.self_attn.q_proj" {
		t.Fatalf("ROCmLoRACanonicalTarget = %q ok=%v, want model.layers.0.self_attn.q_proj true", canonical, ok)
	}

	trimmed, ok := ROCmTrimWeightWrapperPrefix("gemma4_text", "model.language_model.model.layers.0.self_attn.q_proj.weight")
	if !ok || trimmed != "layers.0.self_attn.q_proj.weight" {
		t.Fatalf("ROCmTrimWeightWrapperPrefix = %q ok=%v, want layers.0.self_attn.q_proj.weight true", trimmed, ok)
	}
	weight, ok := ROCmCanonicalWeightName("gemma4_text", "language_model.model.layers.0.self_attn.q_proj.weight")
	if !ok || weight != "model.layers.0.self_attn.q_proj.weight" {
		t.Fatalf("ROCmCanonicalWeightName = %q ok=%v, want model.layers.0.self_attn.q_proj.weight true", weight, ok)
	}
	if skipped, ok := ROCmCanonicalWeightName("gemma4_text", "vision_tower.encoder.weight"); ok || skipped != "" {
		t.Fatalf("ROCmCanonicalWeightName vision = %q ok=%v, want skipped", skipped, ok)
	}

	unknown := "language_model.model.layers.0.self_attn.q_proj.weight"
	if got, ok := ROCmCanonicalWeightName("qwen3", unknown); !ok || got != unknown {
		t.Fatalf("ROCmCanonicalWeightName unknown = %q ok=%v, want passthrough", got, ok)
	}
	if targets := ROCmLoRADefaultTargets("gemma4_assistant"); targets != nil {
		t.Fatalf("ROCmLoRADefaultTargets(gemma4_assistant) = %v, want nil", targets)
	}
	qwenPolicy, ok := ROCmLoRATargetPolicyForArchitecture("qwen3")
	if !ok ||
		!slices.Equal(qwenPolicy.DefaultTargets, []string{"q_proj", "v_proj"}) ||
		qwenPolicy.TargetPaths["q_proj"] != "self_attn.q_proj" ||
		qwenPolicy.TargetPaths["gate_proj"] != "mlp.gate_proj" {
		t.Fatalf("ROCmLoRATargetPolicyForArchitecture(qwen3) = %+v ok=%v, want decoder target policy", qwenPolicy, ok)
	}
	composedPolicy, ok := ROCmLoRATargetPolicyForArchitecture("composed")
	if !ok ||
		!slices.Equal(composedPolicy.DefaultTargets, []string{"gate_proj", "up_proj", "down_proj"}) ||
		slices.Contains(composedPolicy.SafeTargets, "q_proj") ||
		composedPolicy.TargetPaths["gate_proj"] != "mlp.gate_proj" {
		t.Fatalf("ROCmLoRATargetPolicyForArchitecture(composed) = %+v ok=%v, want MLP-only target policy", composedPolicy, ok)
	}
}

func TestResolveROCmModelProfile_Good_PublicReactiveResolver(t *testing.T) {
	inputLabels := map[string]string{
		"gemma4_size":                       "E4B",
		"gemma4_quant_mode":                 "q6",
		"gemma4_sliding_window":             "1024",
		"gemma4_sliding_window_pattern":     "6",
		"gemma4_attention_kv_shared_layers": "4",
		"gemma4_enable_moe_block":           "true",
		"gemma4_num_experts":                "16",
		"gemma4_top_k_experts":              "2",
	}
	profile, ok := ResolveROCmModelProfile("", inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		Architecture: "gemma4_text",
		QuantBits:    6,
		NumLayers:    26,
		HiddenSize:   2304,
		Labels:       inputLabels,
	})
	if !ok ||
		!profile.Matched() ||
		profile.Name != "gemma4" ||
		profile.Registry != rocmModelRegistryName ||
		profile.Architecture != "gemma4_text" ||
		profile.Model.Path == "" ||
		!profile.Gemma4EngineFeatures.GenerateLinked() ||
		!profile.Gemma4EngineFeatures.FixedSlidingCache ||
		!profile.Gemma4DeclaredFeatures.Mixture ||
		profile.Gemma4DeclaredFeatures.NumExperts != 16 ||
		profile.Gemma4DeclaredFeatures.TopKExperts != 2 ||
		profile.Gemma4DeclaredFeatures.Attention.SlidingWindow != 1024 ||
		profile.Gemma4DeclaredFeatures.Attention.SlidingPattern != 6 ||
		profile.Gemma4DeclaredFeatures.Attention.SharedKVLayers != 4 ||
		!profile.CacheRoute.Matched() ||
		profile.CacheRoute.Architecture != "gemma4_text" ||
		profile.CacheRoute.Labels["engine_cache_factory_architecture"] != "gemma4_text" {
		t.Fatalf("ResolveROCmModelProfile = %+v ok=%v, want reactive Gemma4 profile", profile, ok)
	}
	inputLabels["gemma4_size"] = "mutated"
	if profile.Model.Labels["gemma4_size"] != "E4B" {
		t.Fatalf("ResolveROCmModelProfile aliased caller labels: %+v", profile.Model.Labels)
	}

	applied := ApplyROCmModelProfileLabels(map[string]string{"caller": "kept"}, profile)
	if inputLabels["engine_profile"] != "" ||
		applied["caller"] != "kept" ||
		applied["engine_profile"] != "gemma4" ||
		applied["engine_cache_factory_architecture"] != "gemma4_text" ||
		applied["engine_fixed_sliding_cache"] != "true" ||
		applied["gemma4_lora_default_targets"] != "q_proj,v_proj,o_proj" {
		t.Fatalf("ApplyROCmModelProfileLabels = %+v input=%+v, want copy-safe registry labels", applied, inputLabels)
	}
}

func TestResolveROCmModelProfileForInfo_Good_UsesPathAndLabels(t *testing.T) {
	labels := map[string]string{
		"sliding_window":             "512",
		"sliding_window_pattern":     "5",
		"attention_kv_shared_layers": "2",
	}
	profile, ok := ResolveROCmModelProfileForInfo("/models/lmstudio-community-gemma-4-E2B-it-MLX-6bit", inference.ModelInfo{
		Architecture: "gemma4_text",
		VocabSize:    262144,
		NumLayers:    35,
		HiddenSize:   1536,
		QuantBits:    6,
		QuantGroup:   64,
	}, labels)
	if !ok ||
		profile.Model.Labels["gemma4_size"] != "E2B" ||
		profile.Model.Labels["gemma4_quant_mode"] != "q6" ||
		!profile.Gemma4EngineFeatures.GenerateLinked() ||
		!profile.Gemma4EngineFeatures.FixedSlidingCache ||
		profile.Gemma4DeclaredFeatures.Attention.SlidingWindow != 512 ||
		profile.Gemma4DeclaredFeatures.Attention.SlidingPattern != 5 ||
		profile.Gemma4DeclaredFeatures.Attention.SharedKVLayers != 2 {
		t.Fatalf("ResolveROCmModelProfileForInfo = %+v ok=%v, want path-inferred Gemma4 E2B q6 profile", profile, ok)
	}
	labels["sliding_window"] = "mutated"
	if profile.Model.Labels["sliding_window"] != "512" {
		t.Fatalf("ResolveROCmModelProfileForInfo aliased caller labels: %+v", profile.Model.Labels)
	}
	qwen, ok := ResolveROCmModelProfileForInfo("/models/qwen", inference.ModelInfo{Architecture: "Qwen3_5MoeForConditionalGeneration", QuantBits: 4}, nil)
	if !ok ||
		qwen.Name != "qwen" ||
		qwen.Family != "qwen" ||
		qwen.Architecture != "qwen3_6_moe" ||
		qwen.Model.Architecture != "qwen3_6_moe" ||
		qwen.ArchitectureProfile.ID != "qwen3_6_moe" ||
		qwen.ArchitectureProfile.ParserID != "qwen" ||
		!qwen.ArchitectureProfile.MoE ||
		qwen.ArchitectureProfile.Generation ||
		qwen.ArchitectureProfile.Chat ||
		qwen.Labels["engine_profile"] != "qwen" ||
		qwen.Labels["engine_profile_source"] != "architecture_profile" {
		t.Fatalf("ResolveROCmModelProfileForInfo(qwen) = %+v ok=%v, want generic architecture-profile model profile", qwen, ok)
	}
}

func TestResolveROCmModelProfile_Good_GenericArchitectureProfile(t *testing.T) {
	labels := map[string]string{"architecture_resolved": "BertForSequenceClassification"}
	profile, ok := ResolveROCmModelProfile("/models/rerank", inference.ModelIdentity{
		Architecture: "bert",
		Labels:       labels,
	})
	if !ok ||
		profile.Name != "bert" ||
		profile.Family != "bert" ||
		profile.Architecture != "bert_rerank" ||
		profile.Model.Path != "/models/rerank" ||
		profile.Model.Architecture != "bert_rerank" ||
		profile.ArchitectureProfile.ID != "bert_rerank" ||
		!profile.ArchitectureProfile.Rerank ||
		profile.ArchitectureProfile.Chat ||
		profile.ArchitectureProfile.Generation ||
		profile.ArchitectureProfile.ParserID != "generic" {
		t.Fatalf("ResolveROCmModelProfile(rerank) = %+v ok=%v, want generic architecture-profile rerank model profile", profile, ok)
	}
	labels["architecture_resolved"] = "mutated"
	if profile.Model.Labels["architecture_resolved"] != "BertForSequenceClassification" {
		t.Fatalf("ResolveROCmModelProfile aliased generic labels: %+v", profile.Model.Labels)
	}
	profile.ArchitectureProfile.CacheHints = append(profile.ArchitectureProfile.CacheHints, "mutated")
	next, ok := ResolveROCmArchitectureProfileForIdentity("", inference.ModelIdentity{Architecture: "bert_rerank"})
	if !ok || len(next.CacheHints) != 0 {
		t.Fatalf("ResolveROCmArchitectureProfileForIdentity = %+v ok=%v, want copy-safe rerank profile without mutated cache hints", next, ok)
	}
	applied := ApplyROCmModelProfileLabels(map[string]string{"caller": "kept"}, profile)
	if applied["caller"] != "kept" ||
		applied["engine_profile"] != "bert" ||
		applied["engine_profile_source"] != "architecture_profile" ||
		applied["engine_architecture_profile"] != "bert_rerank" ||
		applied["engine_architecture_rerank"] != "true" ||
		applied["engine_architecture_generation"] != "false" ||
		applied["engine_architecture_chat"] != "false" ||
		applied["engine_architecture_reasoning_parser"] != "generic" ||
		applied["reasoning_parser"] != "generic" {
		t.Fatalf("ApplyROCmModelProfileLabels(generic) = %+v, want architecture profile labels", applied)
	}
	if _, ok := ResolveROCmModelProfile("/models/unknown", inference.ModelIdentity{Architecture: "totally_unknown"}); ok {
		t.Fatal("ResolveROCmModelProfile(unknown) ok = true, want false")
	}
}

func TestGemma4LoRATargetPolicy_Good_RegistryLabels(t *testing.T) {
	labels := rocmApplyGemma4LoRAPolicyLabels(nil, "gemma4_text", Gemma4LoRATargetPolicy{})
	if labels["engine_lora_policy"] != "gemma4" ||
		labels["engine_lora_policy_source"] != "model_registry" ||
		labels["engine_lora_default_targets"] != "q_proj,v_proj,o_proj" ||
		labels["engine_lora_safe_targets"] != "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" ||
		labels["engine_lora_extended_targets"] != "router.proj,per_layer_input_gate,per_layer_projection" ||
		labels["engine_lora_extended_targets_require_opt_in"] != "true" ||
		labels["gemma4_lora_targets"] != "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,router.proj,per_layer_input_gate,per_layer_projection" {
		t.Fatalf("labels = %+v, want Gemma4 LoRA policy registry labels", labels)
	}

	assistantLabels := rocmApplyGemma4LoRAPolicyLabels(nil, "gemma4_assistant", Gemma4LoRATargetPolicy{})
	if assistantLabels["engine_lora_policy"] != "" || assistantLabels["gemma4_lora_default_targets"] != "" {
		t.Fatalf("assistant labels = %+v, want no LoRA policy labels", assistantLabels)
	}
}
