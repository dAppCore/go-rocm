// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

func TestProbeConfig_Good_ModelLoaderDispatchContract(t *testing.T) {
	probe := ProbeConfig(ConfigProbeInput{
		ModelType:          "gemma4",
		TextTowerModelType: "gemma4_text",
		Architectures:      []string{"Gemma4ForConditionalGeneration"},
	})
	if probe.Contract != ConfigProbeContract ||
		probe.ArchitectureResolution.Architecture != "gemma4_text" ||
		probe.ArchitectureResolution.Source != "model_type_text_tower" ||
		probe.LoaderRoute.Contract != LoaderRegistryContract ||
		probe.LoaderRoute.Loader != "gemma4_text" ||
		probe.LoaderRoute.Runtime != RuntimeHIP ||
		probe.LoaderRoute.Status != StatusStandaloneNative ||
		probe.RuntimeContractRoute.Contract != RuntimeContractRegistryContract ||
		!probe.RuntimeContractRoute.FixedSlidingCache ||
		!probe.RuntimeContractRoute.ModelInfoReporter ||
		!probe.Registered ||
		!probe.Standalone ||
		!probe.TextGenerate ||
		probe.AttachedOnly ||
		probe.Labels["engine_config_probe_contract"] != ConfigProbeContract ||
		probe.Labels["architecture_resolution_contract"] != ArchitectureResolutionContract ||
		probe.Labels["engine_config_architecture_resolved"] != "gemma4_text" ||
		probe.Labels["engine_config_loader"] != "gemma4_text" ||
		probe.Labels["engine_config_runtime_contract"] != "true" ||
		probe.Labels["engine_runtime_contract_fixed_sliding_cache"] != "true" {
		t.Fatalf("ProbeConfig(gemma4) = %+v, want model-owned config probe to loader/runtime routes", probe)
	}
	probe.Labels["engine_config_loader"] = "mutated"
	next := ProbeConfig(ConfigProbeInput{
		ModelType:          "gemma4",
		TextTowerModelType: "gemma4_text",
		Architectures:      []string{"Gemma4ForConditionalGeneration"},
	})
	if next.Labels["engine_config_loader"] != "gemma4_text" {
		t.Fatalf("ProbeConfig leaked mutable labels: %+v", next)
	}

	assistant := ProbeConfig(ConfigProbeInput{
		ModelType:     "gemma4_assistant",
		Architectures: []string{"Gemma4AssistantForCausalLM"},
	})
	if assistant.ArchitectureResolution.Architecture != "gemma4_assistant" ||
		assistant.LoaderRoute.Status != StatusAttachedOnly ||
		!assistant.RuntimeContractRoute.DecodeUnavailableReporter ||
		!assistant.AttachedOnly ||
		assistant.Standalone {
		t.Fatalf("ProbeConfig(assistant) = %+v, want attached-only loader route", assistant)
	}
}

func TestProbeConfig_Good_ComposedSequenceMixerContract(t *testing.T) {
	probe := ProbeConfig(ConfigProbeInput{
		ModelType:       "composed",
		NumHiddenLayers: 2,
		LayerTypes:      []string{"full_attention", "mamba2"},
	})
	if probe.ArchitectureResolution.Architecture != "composed" ||
		probe.LoaderRoute.Status != StatusStagedNative ||
		!probe.Registered ||
		!probe.Staged ||
		probe.TextGenerate ||
		!probe.SequenceMixer.Composed ||
		probe.SequenceMixer.PlanStatus != "valid" ||
		len(probe.SequenceMixer.Layers) != 2 ||
		probe.SequenceMixer.Layers[0].Kind != "full_attention" ||
		probe.SequenceMixer.Layers[1].Kind != "mamba2" ||
		probe.SequenceMixer.Cache.Layers[0].Mode != SequenceMixerCacheModeDefault ||
		probe.SequenceMixer.Cache.Layers[1].Mode != SequenceMixerCacheModeRecurrent ||
		!probe.RuntimeContractRoute.DecodeUnavailableReporter ||
		probe.Labels["sequence_mixer_config_plan_status"] != "valid" ||
		probe.Labels["sequence_mixer_cache_plan_layers"] != "2" ||
		probe.Labels["engine_config_runtime_contract"] != "true" {
		t.Fatalf("ProbeConfig(composed) = %+v, want registered composed loader and cache factory plan", probe)
	}

	hybrid := ProbeConfig(ConfigProbeInput{
		ModelType:       "hybrid",
		NumHiddenLayers: 2,
	})
	if hybrid.ArchitectureResolution.Architecture != "hybrid" ||
		hybrid.SequenceMixer.PlanStatus != "invalid" ||
		hybrid.SequenceMixer.PlanError == "" ||
		hybrid.Labels["sequence_mixer_config_plan_status"] != "invalid" {
		t.Fatalf("ProbeConfig(hybrid) = %+v, want loud missing layer_types refusal", hybrid)
	}
}
