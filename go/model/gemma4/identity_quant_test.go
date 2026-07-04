// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	"dappco.re/go/inference"
)

func TestIdentityQuant_Good_SizeInference(t *testing.T) {
	tests := []struct {
		name  string
		model inference.ModelIdentity
		path  string
		want  string
	}{
		{name: "label", model: inference.ModelIdentity{Labels: map[string]string{"gemma4_size": " 31b "}}, want: "31B"},
		{name: "26b-a4b path", path: "/models/gemma-4-26B-A4B-it-MLX-6bit", want: "26B-A4B"},
		{name: "31b path", path: "/models/gemma-4-31B-it-MLX-6bit", want: "31B"},
		{name: "12b path", path: "/models/gemma-4-12B-it-6bit", want: "12B"},
		{name: "e4b path", path: "/models/gemma-4-E4B-it-MLX-8bit", want: "E4B"},
		{name: "e2b path", path: "/models/gemma-4-E2B-it-6bit", want: "E2B"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ModelPackSize(tt.model, tt.path); got != tt.want {
				t.Fatalf("ModelPackSize(%+v, %q) = %q, want %q", tt.model, tt.path, got, tt.want)
			}
		})
	}
}

func TestIdentityQuant_Good_GeometrySizeInference(t *testing.T) {
	tests := []struct {
		name  string
		model inference.ModelIdentity
		want  string
	}{
		{name: "31b geometry", model: inference.ModelIdentity{NumLayers: 64, HiddenSize: 4096}, want: "31B"},
		{name: "12b geometry", model: inference.ModelIdentity{NumLayers: 48, HiddenSize: 3840}, want: "12B"},
		{name: "e4b geometry", model: inference.ModelIdentity{NumLayers: 26, HiddenSize: 2304}, want: "E4B"},
		{name: "e2b geometry", model: inference.ModelIdentity{NumLayers: 35, HiddenSize: 1536}, want: "E2B"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ModelPackSize(tt.model, ""); got != "" {
				t.Fatalf("ModelPackSize(%+v, empty) = %q, want fail-closed empty without geometry", tt.model, got)
			}
			if got := ModelPackSizeWithGeometry(tt.model, ""); got != tt.want {
				t.Fatalf("ModelPackSizeWithGeometry(%+v, empty) = %q, want %q", tt.model, got, tt.want)
			}
		})
	}
}

func TestIdentityQuant_Good_QuantModeInference(t *testing.T) {
	tests := []struct {
		name  string
		model inference.ModelIdentity
		path  string
		want  string
	}{
		{name: "label canonicalizes large", model: inference.ModelIdentity{Path: "/models/gemma-4-31B-it", Labels: map[string]string{"gemma4_quant_mode": " Q6 "}}, want: "q6-status"},
		{name: "mxfp8 path wins", model: inference.ModelIdentity{QuantBits: 6}, path: "/models/gemma-4-E2B-it-mxfp8", want: "mxfp8"},
		{name: "mxfp4 path wins", model: inference.ModelIdentity{QuantBits: 6}, path: "/models/gemma-4-E2B-it-mxfp4", want: "mxfp4"},
		{name: "quant type", model: inference.ModelIdentity{QuantType: "bfloat16"}, want: "bf16"},
		{name: "quant bits", model: inference.ModelIdentity{QuantBits: 5}, want: "q5"},
		{name: "path mode", path: "/models/gemma-4-E2B-it-Q4_K_M.gguf", want: "q4"},
		{name: "assistant default", model: inference.ModelIdentity{Architecture: "Gemma4AssistantForCausalLM"}, path: "/models/gemma-4-E2B-it-assistant", want: "bf16"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.path == "" {
				tt.path = tt.model.Path
			}
			if got := ModelPackQuantModeForPath(tt.model, tt.path); got != tt.want {
				t.Fatalf("ModelPackQuantModeForPath(%+v, %q) = %q, want %q", tt.model, tt.path, got, tt.want)
			}
		})
	}
}

func TestIdentityQuant_Good_InferredQuantModeMetadata(t *testing.T) {
	tests := []struct {
		mode      string
		wantType  string
		wantBits  int
		wantGroup int
	}{
		{mode: "bf16", wantType: "bf16", wantBits: 16},
		{mode: "mxfp8", wantType: "mxfp8", wantBits: 8, wantGroup: 32},
		{mode: "mxfp4", wantType: "mxfp4", wantBits: 4, wantGroup: 32},
		{mode: "nvfp4", wantType: "nvfp4", wantBits: 4, wantGroup: 32},
		{mode: "q8-status", wantType: "q8", wantBits: 8},
		{mode: "q6", wantType: "q6", wantBits: 6, wantGroup: 64},
		{mode: "q5", wantType: "q5", wantBits: 5, wantGroup: 64},
		{mode: "q4-status", wantType: "q4", wantBits: 4},
	}
	for _, tt := range tests {
		t.Run(tt.mode, func(t *testing.T) {
			model := ModelWithInferredQuantMode(inference.ModelIdentity{}, tt.mode)
			if model.QuantType != tt.wantType || model.QuantBits != tt.wantBits || model.QuantGroup != tt.wantGroup {
				t.Fatalf("ModelWithInferredQuantMode(%q) = type=%q bits=%d group=%d, want type=%q bits=%d group=%d", tt.mode, model.QuantType, model.QuantBits, model.QuantGroup, tt.wantType, tt.wantBits, tt.wantGroup)
			}
		})
	}
}

func TestIdentityQuant_Good_CanonicalQuantMode(t *testing.T) {
	if got := CanonicalQuantMode("31B", " Q6 "); got != "q6-status" {
		t.Fatalf("CanonicalQuantMode(31B, Q6) = %q, want q6-status", got)
	}
	if got := CanonicalQuantMode("E4B", " Q8 "); got != "q8" {
		t.Fatalf("CanonicalQuantMode(E4B, Q8) = %q, want q8", got)
	}
	if got := NormalizeSizeQuantMode("E4B", " Q8 "); got != " Q8 " {
		t.Fatalf("NormalizeSizeQuantMode preserves non-status spelling = %q, want original", got)
	}
}
