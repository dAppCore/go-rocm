// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"dappco.re/go/inference"
)

func TestLoRAFuseDenseSafetensors_Good_MergesF32AndWritesProvenance(t *testing.T) {
	baseDir := t.TempDir()
	adapterDir := t.TempDir()
	outDir := filepath.Join(t.TempDir(), "fused")

	baseWeight := "model.layers.0.self_attn.q_proj.weight"
	if err := rocmWriteFuseSafetensors(filepath.Join(baseDir, "model.safetensors"), []rocmFuseWriteTensor{
		{Name: baseWeight, DType: "F32", Shape: []uint64{2, 3}, Data: encodeTestF32s(1, 2, 3, 4, 5, 6)},
		{Name: "model.norm.weight", DType: "F32", Shape: []uint64{3}, Data: encodeTestF32s(7, 8, 9)},
	}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(baseDir, "config.json"), []byte(`{"model_type":"gemma4_text"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := rocmWriteFuseSafetensors(filepath.Join(adapterDir, "adapter.safetensors"), []rocmFuseWriteTensor{
		{Name: "model.layers.0.q_proj.lora_A.weight", DType: "F32", Shape: []uint64{1, 3}, Data: encodeTestF32s(0.1, 0.2, 0.3)},
		{Name: "model.layers.0.q_proj.lora_B.weight", DType: "F32", Shape: []uint64{2, 1}, Data: encodeTestF32s(2, 3)},
	}); err != nil {
		t.Fatal(err)
	}

	result, err := FuseLoRAIntoModelPack(context.Background(), LoRAFuseOptions{
		BasePath:     baseDir,
		AdapterPath:  adapterDir,
		OutputPath:   outDir,
		Architecture: "gemma4_text",
		Adapter: inference.AdapterIdentity{
			Path:       adapterDir,
			Format:     "lora",
			Rank:       1,
			Alpha:      2,
			TargetKeys: []string{"q_proj"},
		},
	})
	if err != nil {
		t.Fatalf("FuseLoRAIntoModelPack: %v", err)
	}
	if result.FusedWeights != 1 || len(result.FusedWeightKeys) != 1 || result.FusedWeightKeys[0] != baseWeight ||
		len(result.FusedLayers) != 1 || result.FusedLayers[0] != strings.TrimSuffix(baseWeight, ".weight") {
		t.Fatalf("result = %+v, want one fused q_proj weight", result)
	}
	if result.Labels["fuse_runtime"] != "dense_f32_cpu" || result.Labels["fuse_safetensors"] != "linked" {
		t.Fatalf("labels = %+v, want dense F32 linked fuse", result.Labels)
	}
	if result.Labels["fuse_layer_count"] != "1" {
		t.Fatalf("labels = %+v, want one fused layer", result.Labels)
	}
	if _, err := os.Stat(filepath.Join(outDir, "config.json")); err != nil {
		t.Fatalf("config.json was not copied: %v", err)
	}
	provenanceRaw, err := os.ReadFile(filepath.Join(outDir, LoRAFuseProvenanceFile))
	if err != nil {
		t.Fatalf("provenance was not written: %v", err)
	}
	var provenance LoRAFuseProvenance
	if err := json.Unmarshal(provenanceRaw, &provenance); err != nil {
		t.Fatalf("provenance JSON: %v", err)
	}
	if len(provenance.FusedLayers) != 1 || provenance.FusedLayers[0] != result.FusedLayers[0] {
		t.Fatalf("provenance fused layers = %+v, want %+v", provenance.FusedLayers, result.FusedLayers)
	}

	index, err := rocmReadFuseSafetensorsIndex(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	got, err := rocmReadFuseTensorF32(index[baseWeight])
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1.4, 2.8, 4.2, 4.6, 6.2, 7.8}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("fused[%d] = %.4f, want %.4f (all=%v)", i, got[i], want[i], got)
		}
	}
	norm, err := rocmReadFuseTensorF32(index["model.norm.weight"])
	if err != nil {
		t.Fatal(err)
	}
	if len(norm) != 3 || norm[0] != 7 || norm[1] != 8 || norm[2] != 9 {
		t.Fatalf("untargeted tensor = %v, want carry-through", norm)
	}
}

func TestLoRAFuseDenseSafetensors_Good_DefaultsRankOnlyAdapterScale(t *testing.T) {
	baseDir := t.TempDir()
	adapterDir := t.TempDir()
	outDir := filepath.Join(t.TempDir(), "fused")

	baseWeight := "model.layers.0.self_attn.q_proj.weight"
	if err := rocmWriteFuseSafetensors(filepath.Join(baseDir, "model.safetensors"), []rocmFuseWriteTensor{
		{Name: baseWeight, DType: "F32", Shape: []uint64{2, 3}, Data: encodeTestF32s(1, 2, 3, 4, 5, 6)},
	}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(baseDir, "config.json"), []byte(`{"model_type":"gemma4_text"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := rocmWriteFuseSafetensors(filepath.Join(adapterDir, "adapter.safetensors"), []rocmFuseWriteTensor{
		{Name: "model.layers.0.q_proj.lora_A.weight", DType: "F32", Shape: []uint64{1, 3}, Data: encodeTestF32s(0.1, 0.2, 0.3)},
		{Name: "model.layers.0.q_proj.lora_B.weight", DType: "F32", Shape: []uint64{2, 1}, Data: encodeTestF32s(2, 3)},
	}); err != nil {
		t.Fatal(err)
	}

	result, err := FuseLoRAIntoModelPack(context.Background(), LoRAFuseOptions{
		BasePath:     baseDir,
		AdapterPath:  adapterDir,
		OutputPath:   outDir,
		Architecture: "gemma4_text",
		Adapter: inference.AdapterIdentity{
			Path:       adapterDir,
			Format:     "lora",
			Rank:       1,
			TargetKeys: []string{"q_proj"},
		},
	})
	if err != nil {
		t.Fatalf("FuseLoRAIntoModelPack: %v", err)
	}
	if result.FusedWeights != 1 || result.FusedWeightKeys[0] != baseWeight {
		t.Fatalf("result = %+v, want one rank-only fused q_proj weight", result)
	}

	index, err := rocmReadFuseSafetensorsIndex(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	got, err := rocmReadFuseTensorF32(index[baseWeight])
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1.4, 2.8, 4.2, 4.6, 6.2, 7.8}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("rank-only fused[%d] = %.4f, want %.4f (all=%v)", i, got[i], want[i], got)
		}
	}
}

func TestLoRAFuseDenseSafetensors_Good_DequantizesMLXAffineQ6AndDropsSidecars(t *testing.T) {
	baseDir := t.TempDir()
	adapterDir := t.TempDir()
	outDir := filepath.Join(t.TempDir(), "fused")

	const cols = 64
	values := make([]uint32, cols)
	aValues := make([]float32, cols)
	for i := range values {
		values[i] = uint32(i % cols)
		aValues[i] = 1
	}
	packed := hipPackMLXAffineValuesForTest(values, cols, 6)

	basePrefix := "language_model.model.layers.0.self_attn.q_proj"
	baseWeight := basePrefix + ".weight"
	if err := rocmWriteFuseSafetensors(filepath.Join(baseDir, "model.safetensors"), []rocmFuseWriteTensor{
		{Name: baseWeight, DType: "U32", Shape: []uint64{1, uint64(len(packed))}, Data: encodeTestU32s(packed...)},
		{Name: basePrefix + ".scales", DType: "BF16", Shape: []uint64{1, 1}, Data: encodeTestBF16s(0x3f80)},
		{Name: basePrefix + ".biases", DType: "BF16", Shape: []uint64{1, 1}, Data: encodeTestBF16s(0)},
		{Name: "model.norm.weight", DType: "F32", Shape: []uint64{1}, Data: encodeTestF32s(7)},
	}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(baseDir, "config.json"), []byte(`{"model_type":"gemma4_text"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := rocmWriteFuseSafetensors(filepath.Join(adapterDir, "adapter.safetensors"), []rocmFuseWriteTensor{
		{Name: "model.layers.0.q_proj.lora_A.weight", DType: "F32", Shape: []uint64{1, cols}, Data: encodeTestF32Slice(aValues)},
		{Name: "model.layers.0.q_proj.lora_B.weight", DType: "F32", Shape: []uint64{1, 1}, Data: encodeTestF32s(1)},
	}); err != nil {
		t.Fatal(err)
	}

	result, err := FuseLoRAIntoModelPack(context.Background(), LoRAFuseOptions{
		BasePath:     baseDir,
		AdapterPath:  adapterDir,
		OutputPath:   outDir,
		Architecture: "gemma4_text",
		Adapter: inference.AdapterIdentity{
			Path:       adapterDir,
			Format:     "lora",
			Rank:       1,
			Alpha:      1,
			TargetKeys: []string{"q_proj"},
		},
	})
	if err != nil {
		t.Fatalf("FuseLoRAIntoModelPack: %v", err)
	}
	if result.FusedWeights != 1 || result.FusedWeightKeys[0] != baseWeight ||
		len(result.FusedLayers) != 1 || result.FusedLayers[0] != strings.TrimSuffix(baseWeight, ".weight") {
		t.Fatalf("result = %+v, want wrapped q_proj fused", result)
	}
	if result.Labels["fuse_quantized_base"] != "dequantized_dense" || result.Labels["fuse_dequantized_targets"] != "1" {
		t.Fatalf("labels = %+v, want quantized base dequantized", result.Labels)
	}

	index, err := rocmReadFuseSafetensorsIndex(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := index[basePrefix+".scales"]; ok {
		t.Fatalf("quantized scale sidecar was retained in fused output")
	}
	if _, ok := index[basePrefix+".biases"]; ok {
		t.Fatalf("quantized bias sidecar was retained in fused output")
	}
	fused := index[baseWeight]
	if fused.DType != "F32" || !sameUint64Shape(fused.Shape, []uint64{1, cols}) {
		t.Fatalf("fused tensor = %+v, want dense F32 [1,%d]", fused, cols)
	}
	got, err := rocmReadFuseTensorF32(fused)
	if err != nil {
		t.Fatal(err)
	}
	for i, wantBase := range values {
		want := float32(wantBase) + 1
		if math.Abs(float64(got[i]-want)) > 1e-5 {
			t.Fatalf("fused[%d] = %.4f, want %.4f (all=%v)", i, got[i], want, got)
		}
	}
}

func TestLoRAFuseDenseSafetensors_Bad_RejectsNonF32Base(t *testing.T) {
	baseDir := t.TempDir()
	adapterDir := t.TempDir()
	if err := rocmWriteFuseSafetensors(filepath.Join(baseDir, "model.safetensors"), []rocmFuseWriteTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", DType: "F16", Shape: []uint64{2, 3}, Data: make([]byte, 12)},
	}); err != nil {
		t.Fatal(err)
	}
	if err := rocmWriteFuseSafetensors(filepath.Join(adapterDir, "adapter.safetensors"), []rocmFuseWriteTensor{
		{Name: "model.layers.0.self_attn.q_proj.lora_A.weight", DType: "F32", Shape: []uint64{1, 3}, Data: encodeTestF32s(0.1, 0.2, 0.3)},
		{Name: "model.layers.0.self_attn.q_proj.lora_B.weight", DType: "F32", Shape: []uint64{2, 1}, Data: encodeTestF32s(2, 3)},
	}); err != nil {
		t.Fatal(err)
	}

	_, err := FuseLoRAIntoModelPack(context.Background(), LoRAFuseOptions{
		BasePath:     baseDir,
		AdapterPath:  adapterDir,
		OutputPath:   filepath.Join(t.TempDir(), "fused"),
		Architecture: "gemma4_text",
		Adapter:      inference.AdapterIdentity{Rank: 1, Alpha: 2},
	})
	if err == nil || !strings.Contains(err.Error(), "F32") {
		t.Fatalf("FuseLoRAIntoModelPack err = %v, want F32 refusal", err)
	}
}

func encodeTestF32s(values ...float32) []byte {
	return encodeTestF32Slice(values)
}

func encodeTestF32Slice(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(value))
	}
	return out
}

func encodeTestU32s(values ...uint32) []byte {
	out := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(out[i*4:], value)
	}
	return out
}

func encodeTestBF16s(values ...uint16) []byte {
	out := make([]byte, len(values)*2)
	for i, value := range values {
		binary.LittleEndian.PutUint16(out[i*2:], value)
	}
	return out
}
