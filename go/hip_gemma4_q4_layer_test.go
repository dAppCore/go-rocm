// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"testing"

	"dappco.re/go/inference"
)

func TestHIPGemma4Q4ProjectionConfigInfersMixedAffineBits(t *testing.T) {
	const (
		groupSize = 64
		rows      = 15360
		cols      = 3840
		bits      = 8
	)
	groups := cols / groupSize
	packedCols, err := hipMLXAffinePackedCols(cols, bits)
	if err != nil {
		t.Fatal(err)
	}
	model := &hipLoadedModel{
		modelInfo: inference.ModelInfo{QuantBits: 4},
		tensors: map[string]hipTensor{
			"language_model.model.layers.0.mlp.gate_proj.weight": {
				info:    nativeTensorInfo{TypeName: "U32", Dimensions: []uint64{rows, uint64(packedCols)}, ByteSize: uint64(rows * packedCols * 4)},
				pointer: nativeDevicePointer(0x1000),
			},
			"language_model.model.layers.0.mlp.gate_proj.scales": {
				info:    nativeTensorInfo{TypeName: "BF16", Dimensions: []uint64{rows, uint64(groups)}, ByteSize: uint64(rows * groups * 2)},
				pointer: nativeDevicePointer(0x2000),
			},
			"language_model.model.layers.0.mlp.gate_proj.biases": {
				info:    nativeTensorInfo{TypeName: "BF16", Dimensions: []uint64{rows, uint64(groups)}, ByteSize: uint64(rows * groups * 2)},
				pointer: nativeDevicePointer(0x3000),
			},
		},
	}

	cfg, gotRows, gotCols, err := model.loadedGemma4Q4ProjectionConfig("language_model.model.layers.0.mlp.gate_proj", "mlp.gate_proj", groupSize)
	if err != nil {
		t.Fatal(err)
	}
	if gotRows != rows {
		t.Fatalf("rows = %d, want %d", gotRows, rows)
	}
	if gotCols != cols {
		t.Fatalf("cols = %d, want %d", gotCols, cols)
	}
	if cfg.Bits != bits {
		t.Fatalf("cfg.Bits = %d, want %d", cfg.Bits, bits)
	}
}

func TestHIPGemma4Q4LayerHeadDimResolvesGemma412BGQA(t *testing.T) {
	model := &hipLoadedModel{
		gemma4TextConfig: nativeGemma4TextConfig{
			HeadDim:       256,
			GlobalHeadDim: 512,
		},
	}

	slidingHeadDim := model.loadedGemma4Q4LayerHeadDim("sliding_attention", 4096, 2048)
	if slidingHeadDim != 256 {
		t.Fatalf("sliding head dim = %d, want 256", slidingHeadDim)
	}
	fullHeadDim := model.loadedGemma4Q4LayerHeadDim("full_attention", 8192, 512)
	if fullHeadDim != 512 {
		t.Fatalf("full head dim = %d, want 512", fullHeadDim)
	}
}
