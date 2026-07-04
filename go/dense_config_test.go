// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"math"
	"testing"
)

func TestParseDenseConfig_Defaults_Good(t *testing.T) {
	cfg, err := ParseDenseConfig([]byte(`{
		"hidden_size": 1024,
		"num_hidden_layers": 8,
		"num_attention_heads": 4,
		"num_key_value_heads": 2
	}`))
	if err != nil {
		t.Fatalf("ParseDenseConfig() error = %v", err)
	}
	if cfg.HeadDim != 256 {
		t.Fatalf("HeadDim = %d, want 256", cfg.HeadDim)
	}
	if cfg.Scale != float32(1.0/math.Sqrt(256)) {
		t.Fatalf("Scale = %f, want attention scale", cfg.Scale)
	}
	if cfg.RopeTheta != 1000000 || cfg.RMSNormEps != 1e-6 || cfg.VocabSize != 151936 {
		t.Fatalf("defaults = vocab:%d rope:%f eps:%f", cfg.VocabSize, cfg.RopeTheta, cfg.RMSNormEps)
	}
}

func TestParseDenseConfig_TextConfigAndQuantization_Good(t *testing.T) {
	cfg, err := ParseDenseConfig([]byte(`{
		"model_type": "qwen3",
		"hidden_size": 4096,
		"num_attention_heads": 32,
		"num_experts_per_tok": 4,
		"layer_types": ["linear-attention", "full.attention"],
		"quantization": {"format":"q6"},
		"text_config": {
			"hidden_size": 2048,
			"num_attention_heads": 16,
			"num_experts": 8,
			"quantization_config": {"format":"q8"}
		}
	}`))
	if err != nil {
		t.Fatalf("ParseDenseConfig() error = %v", err)
	}
	if cfg.ModelType != "qwen3" || cfg.HiddenSize != 2048 || cfg.NumAttentionHeads != 16 || cfg.NumExperts != 8 || cfg.NumExpertsPerTok != 4 {
		t.Fatalf("cfg = %+v, want merged text config with top-level sparse metadata", cfg)
	}
	if cfg.Quantization == nil || cfg.Quantization.Format != "q6" {
		t.Fatalf("quantization = %+v, want top-level quantization first", cfg.Quantization)
	}
	if len(cfg.LayerTypes) != 2 || NormalizeDenseLayerType(cfg.LayerTypes[0]) != "linear_attention" || NormalizeDenseLayerType(cfg.LayerTypes[1]) != "full_attention" {
		t.Fatalf("layer types = %+v, want top-level copied into text config", cfg.LayerTypes)
	}
	if !cfg.IsMoE() {
		t.Fatalf("IsMoE() = false, want true")
	}
}

func TestParseDenseConfig_AutoRoundQuantization_Good(t *testing.T) {
	sym := true
	cfg, err := ParseDenseConfig([]byte(`{
		"model_type": "qwen3",
		"quantization_config": {
			"quant_method": "auto-round-best",
			"format": "mxfp4",
			"weight_format": "nvfp4",
			"scheme": "W4A16",
			"bits": 4,
			"group_size": 128,
			"iters": 200,
			"nsamples": 512,
			"seqlen": 2048,
			"sym": true
		}
	}`))
	if err != nil {
		t.Fatalf("ParseDenseConfig() error = %v", err)
	}
	if cfg.Quantization == nil ||
		!cfg.Quantization.IsAutoRound() ||
		cfg.Quantization.Method() != "auto_round_best" ||
		cfg.Quantization.Format != "mxfp4" ||
		cfg.Quantization.WeightFormat != "nvfp4" ||
		cfg.Quantization.Scheme != "W4A16" ||
		cfg.Quantization.Bits != 4 ||
		cfg.Quantization.GroupSize != 128 ||
		cfg.Quantization.Iters != 200 ||
		cfg.Quantization.NSamples != 512 ||
		cfg.Quantization.SeqLen != 2048 ||
		cfg.Quantization.Sym == nil ||
		*cfg.Quantization.Sym != sym {
		t.Fatalf("quantization = %+v, want AutoRound FP4 config metadata", cfg.Quantization)
	}
	profile, ok := cfg.Quantization.AutoRoundProfile()
	if !ok || profile.Name != "w4a16-nvfp4-g128" || profile.FloatFormat != "nvfp4" || profile.ProductRole != "rocm-fp4-planning" || profile.HIPKernel != hipKernelStatusNotLinked {
		t.Fatalf("AutoRoundProfile() = %+v ok=%v, want matched ROCm NVFP4 profile", profile, ok)
	}
	plan, ok := cfg.Quantization.AutoRoundCalibrationPlan()
	if !ok ||
		plan.ProfileName != "w4a16-nvfp4-g128" ||
		plan.FloatFormat != "nvfp4" ||
		plan.NSamples != 512 ||
		plan.SeqLen != 2048 ||
		plan.Iters != 200 ||
		plan.Runtime != "planned_hip" ||
		plan.HIPKernel != hipKernelStatusNotLinked ||
		!plan.RequiresCalibration ||
		!plan.RequiresBench {
		t.Fatalf("AutoRoundCalibrationPlan() = %+v ok=%v, want planned ROCm NVFP4 calibration plan", plan, ok)
	}
	override := *cfg.Quantization
	override.NSamples = 768
	override.SeqLen = 4096
	override.Iters = 240
	plan, ok = override.AutoRoundCalibrationPlan()
	if !ok || plan.NSamples != 768 || plan.SeqLen != 4096 || plan.Iters != 240 || plan.ProfileName != "w4a16-nvfp4-g128" {
		t.Fatalf("override AutoRoundCalibrationPlan() = %+v ok=%v, want config calibration knobs over profile defaults", plan, ok)
	}
	if plan.NSamplesLabel != "768" || plan.SeqLenLabel != "4096" || plan.ItersLabel != "240" {
		t.Fatalf("override calibration labels = nsamples:%q seqlen:%q iters:%q, want refreshed config override labels", plan.NSamplesLabel, plan.SeqLenLabel, plan.ItersLabel)
	}
	if !(&DenseQuantizationConfig{QuantMethod: "autoround"}).IsAutoRound() {
		t.Fatal("DenseQuantizationConfig{QuantMethod: autoround}.IsAutoRound() = false")
	}
	mxfp8 := DenseQuantizationConfig{
		QuantMethod:  "auto-round",
		WeightFormat: "mxfp8",
		Scheme:       "W8A16",
		Bits:         8,
		GroupSize:    64,
	}
	profile, ok = mxfp8.AutoRoundProfile()
	if !ok || profile.Name != "w8a16-mxfp8-g64" || profile.FloatFormat != "mxfp8" || profile.ProductRole != "rocm-fp8-planning" {
		t.Fatalf("MXFP8 AutoRoundProfile() = %+v ok=%v, want planned ROCm MXFP8 profile", profile, ok)
	}
	plan, ok = mxfp8.AutoRoundCalibrationPlan()
	if !ok || plan.ProfileName != "w8a16-mxfp8-g64" || plan.FloatFormat != "mxfp8" || plan.GroupSize != 64 || plan.HIPKernel != hipKernelStatusNotLinked {
		t.Fatalf("MXFP8 AutoRoundCalibrationPlan() = %+v ok=%v, want planned ROCm MXFP8 calibration plan", plan, ok)
	}
	int2 := DenseQuantizationConfig{
		QuantMethod:  "auto-round",
		WeightFormat: "int2",
		Scheme:       "W2A16",
		Bits:         2,
		GroupSize:    128,
	}
	profile, ok = int2.AutoRoundProfile()
	if !ok || profile.Name != "w2a16-int2-g128" || profile.FloatFormat != "int2" || profile.ProductRole != "rocm-int2-planning" {
		t.Fatalf("INT2 AutoRoundProfile() = %+v ok=%v, want planned ROCm W2A16 INT2 profile", profile, ok)
	}
	plan, ok = int2.AutoRoundCalibrationPlan()
	if !ok || plan.ProfileName != "w2a16-int2-g128" || plan.Bits != 2 || plan.GroupSize != 128 || plan.HIPKernel != hipKernelStatusNotLinked {
		t.Fatalf("INT2 AutoRoundCalibrationPlan() = %+v ok=%v, want planned ROCm W2A16 INT2 calibration plan", plan, ok)
	}
	q2Alias := DenseQuantizationConfig{
		QuantMethod:  "auto-round",
		WeightFormat: "q2",
		Scheme:       "W2A16",
		Bits:         2,
		GroupSize:    128,
	}
	profile, ok = q2Alias.AutoRoundProfile()
	if !ok || profile.Name != "w2a16-int2-g128" || profile.FloatFormat != "int2" {
		t.Fatalf("q2 alias AutoRoundProfile() = %+v ok=%v, want planned ROCm W2A16 INT2 profile", profile, ok)
	}
}

func TestParseDenseConfig_ArchitectureFallbackAndHybrid_Good(t *testing.T) {
	cfg, err := ParseDenseConfig([]byte(`{
		"architectures": ["Qwen3_5ForConditionalGeneration"],
		"partial_rotary_factor": 0.5
	}`))
	if err != nil {
		t.Fatalf("ParseDenseConfig() error = %v", err)
	}
	if cfg.ModelType != "qwen3_6" || !cfg.IsQwen36Hybrid() {
		t.Fatalf("cfg = %+v, want qwen3_6 hybrid", cfg)
	}
	cfg, err = ParseDenseConfig([]byte(`{"model_type":"qwen3","layer_types":["linear.attention","full_attention"]}`))
	if err != nil {
		t.Fatalf("ParseDenseConfig(layer types) error = %v", err)
	}
	if !cfg.IsQwen36Hybrid() {
		t.Fatalf("IsQwen36Hybrid() = false, want linear_attention layer metadata to qualify")
	}
}

func TestParseDenseConfig_Bad(t *testing.T) {
	if _, err := ParseDenseConfig([]byte("{broken")); err == nil {
		t.Fatal("ParseDenseConfig(invalid JSON) error = nil")
	}
}

func BenchmarkDenseQuantizationConfigAutoRoundProfile_NVFP4(b *testing.B) {
	cfg := &DenseQuantizationConfig{
		QuantMethod:  "auto-round-best",
		Format:       "mxfp4",
		WeightFormat: "nvfp4",
		Scheme:       "W4A16",
		Bits:         4,
		GroupSize:    128,
	}
	for i := 0; i < b.N; i++ {
		profile, ok := cfg.AutoRoundProfile()
		if !ok {
			b.Fatal("missing nvfp4 profile")
		}
		productionAutoRoundProfileSink = profile
	}
}

var productionAutoRoundCalibrationPlanSink ProductionAutoRoundCalibrationPlan

func BenchmarkDenseQuantizationConfigAutoRoundCalibrationPlan_NVFP4(b *testing.B) {
	cfg := &DenseQuantizationConfig{
		QuantMethod:  "auto-round-best",
		Format:       "mxfp4",
		WeightFormat: "nvfp4",
		Scheme:       "W4A16",
		Bits:         4,
		GroupSize:    128,
		NSamples:     768,
		SeqLen:       4096,
		Iters:        240,
	}
	for i := 0; i < b.N; i++ {
		plan, ok := cfg.AutoRoundCalibrationPlan()
		if !ok {
			b.Fatal("missing nvfp4 calibration plan")
		}
		productionAutoRoundCalibrationPlanSink = plan
	}
}

func TestDetectDenseModelType_Good(t *testing.T) {
	tests := []struct {
		name   string
		config string
		names  map[string]bool
		want   string
	}{
		{
			name:   "llama architecture metadata",
			config: `{"architectures":["LlamaForCausalLM"]}`,
			want:   "llama",
		},
		{
			name:   "qwen3 next architecture metadata",
			config: `{"architectures":["Qwen3NextForCausalLM"]}`,
			want:   "qwen3_next",
		},
		{
			name:   "qwen3 moe architecture metadata",
			config: `{"architectures":["Qwen3MoeForCausalLM"]}`,
			want:   "qwen3_moe",
		},
		{
			name:   "qwen3 q norm alias",
			config: `{}`,
			names: map[string]bool{
				"language_model.model.layers.0.self_attn.q_norm.weight": true,
			},
			want: "qwen3",
		},
		{
			name:   "qwen2 default",
			config: `{}`,
			names:  map[string]bool{},
			want:   "qwen2",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DetectDenseModelType([]byte(tt.config), tt.names); got != tt.want {
				t.Fatalf("DetectDenseModelType() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDenseWeightNameCandidates_Good(t *testing.T) {
	names := map[string]bool{
		"language_model.model.layers.0.self_attn.q_norm.weight": true,
	}
	if !HasResolvedDenseWeightName(names, "model.layers.0.self_attn.q_norm.weight") {
		t.Fatal("HasResolvedDenseWeightName() did not resolve language_model model alias")
	}
	candidates := DenseWeightNameCandidates("layers.0.self_attn.q_norm.weight")
	if len(candidates) != 6 ||
		candidates[0] != "layers.0.self_attn.q_norm.weight" ||
		candidates[1] != "model.layers.0.self_attn.q_norm.weight" ||
		candidates[3] != "language_model.model.layers.0.self_attn.q_norm.weight" {
		t.Fatalf("DenseWeightNameCandidates() = %+v, want standard aliases", candidates)
	}
}

func TestDenseHelpers_Good(t *testing.T) {
	if NormalizeDenseLayerType("global-attention") != "global_attention" {
		t.Fatalf("NormalizeDenseLayerType(global-attention) = %q", NormalizeDenseLayerType("global-attention"))
	}
	if Qwen36NativeGuardMessage("qwen3_6_moe") == Qwen36NativeGuardMessage("qwen3_6") {
		t.Fatal("Qwen36NativeGuardMessage() did not distinguish MoE")
	}
	q8 := &DenseQuantizationConfig{Format: "q8"}
	q6 := &DenseQuantizationConfig{Format: "q6"}
	if FirstDenseQuantization(nil, q8, q6) != q8 {
		t.Fatal("FirstDenseQuantization() did not return first non-nil config")
	}
}
