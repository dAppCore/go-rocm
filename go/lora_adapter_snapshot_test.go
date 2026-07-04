// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"

	core "dappco.re/go"
)

var (
	loraAdapterSnapshotBenchIdentity inferenceAdapterIdentityShim
	loraAdapterSnapshotBenchRecord   NativeAdamWTrackRecord
	loraAdapterSnapshotBenchErr      error
)

func TestNativeLoRAAdapterSnapshot_TinyRoundTrip_Good(t *testing.T) {
	state, err := NewNativeLoRAAdamWState(
		[]float32{1, 2},
		[]float32{3, 4, 5},
		3,
		2,
		1,
		NativeAdamWConfig{},
	)
	core.RequireNoError(t, err)
	path := core.PathJoin(t.TempDir(), "adapter", "rocm_tiny_lora.json")

	identity, err := SaveNativeLoRAAdapterSnapshot(path, state, NativeLoRAAdapterSnapshotConfig{
		Format: rocmTinyLoRAFormat,
		Name:   "trained-tiny",
		Rows:   3,
		Cols:   2,
		Rank:   1,
		Alpha:  2,
		Bias:   []float32{0.1, 0.2, 0.3},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmTinyLoRAFormat, identity.Format)
	core.AssertEqual(t, 1, identity.Rank)
	core.AssertEqual(t, float32(2), identity.Alpha)
	core.AssertEqual(t, "2", identity.Labels["adapter_alpha"])
	core.AssertEqual(t, rocmTinyLoRAFormat, identity.Labels["adapter_format"])
	core.AssertEqual(t, "trained-tiny", identity.Labels["adapter_name"])
	core.AssertEqual(t, "1", identity.Labels["adapter_rank"])
	core.AssertEqual(t, "lora_adamw_state", identity.Labels["adapter_snapshot"])
	core.AssertEqual(t, "output.weight", identity.Labels["adapter_target"])
	core.AssertEqual(t, "2", identity.Labels["adapter_target_cols"])
	core.AssertEqual(t, "3", identity.Labels["adapter_target_rows"])
	read := core.ReadFile(path)
	if !read.OK {
		t.Fatalf("ReadFile snapshot: %v", read.Value)
	}
	sum := sha256.Sum256(read.Value.([]byte))
	core.AssertEqual(t, hex.EncodeToString(sum[:]), identity.Hash)
	core.AssertEqual(t, identity.Hash, identity.Labels["adapter_hash"])
	var file hipTinyLoRAAdapterFile
	if result := core.JSONUnmarshal(read.Value.([]byte), &file); !result.OK {
		t.Fatalf("JSONUnmarshal snapshot: %v", result.Value)
	}
	adapter, err := validateTinyLoRAAdapterFile(file, hipLoadedTinyLMConfig{HiddenSize: 2, VocabSize: 3})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{1, 2}, adapter.a)
	core.AssertEqual(t, []float32{3, 4, 5}, adapter.b)
	core.AssertEqual(t, []float32{0.1, 0.2, 0.3}, adapter.bias)
}

func TestNativeLoRAAdapterSnapshot_ClassifierRoundTrip_Good(t *testing.T) {
	state, err := NewNativeLoRAAdamWState(
		[]float32{1, 0},
		[]float32{0, 4},
		2,
		2,
		1,
		NativeAdamWConfig{},
	)
	core.RequireNoError(t, err)
	path := core.PathJoin(t.TempDir(), "classifier_lora.json")

	identity, err := SaveNativeLoRAAdapterSnapshot(path, state, NativeLoRAAdapterSnapshotConfig{
		Format: rocmClassifierLoRAFormat,
		Rows:   2,
		Cols:   2,
		Rank:   1,
		Alpha:  1,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmClassifierLoRAFormat, identity.Format)
	core.AssertEqual(t, []string{"classifier.weight"}, identity.TargetKeys)
	read := core.ReadFile(path)
	if !read.OK {
		t.Fatalf("ReadFile classifier snapshot: %v", read.Value)
	}
	var file hipClassifierLoRAAdapterFile
	if result := core.JSONUnmarshal(read.Value.([]byte), &file); !result.OK {
		t.Fatalf("JSONUnmarshal classifier snapshot: %v", result.Value)
	}
	adapter, err := validateClassifierLoRAAdapterFile(file, hipLoadedSequenceClassifierConfig{HiddenSize: 2, NumLabels: 2})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{1, 0}, adapter.a)
	core.AssertEqual(t, []float32{0, 4}, adapter.b)
}

func TestNativeLoRAAdapterSnapshot_FromTrackStep_Good(t *testing.T) {
	state, err := NewNativeLoRAAdamWState(
		[]float32{1, 2},
		[]float32{3, 4},
		2,
		2,
		1,
		NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true},
	)
	core.RequireNoError(t, err)
	trackPath := core.PathJoin(t.TempDir(), "training", "lora-adapter.mp4")
	_, err = AppendNativeAdamWStateTrack(trackPath, state)
	core.RequireNoError(t, err)
	core.RequireNoError(t, state.StepInPlace([][]float32{{0.5, -0.25}, {0.1, -0.2}}))
	record, err := AppendNativeAdamWStateTrack(trackPath, state)
	core.RequireNoError(t, err)
	path := core.PathJoin(t.TempDir(), "adapter", "step-1.json")

	identity, loadedRecord, err := SaveNativeLoRAAdapterSnapshotTrackStep(trackPath, 1, path, NativeLoRAAdapterSnapshotConfig{
		Format: rocmTinyLoRAFormat,
		Name:   "step-1",
		Rows:   2,
		Cols:   2,
		Rank:   1,
		Alpha:  1,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, record, loadedRecord)
	core.AssertEqual(t, NativeAdamWTrackContainerMP4, identity.Labels["adapter_track_container"])
	core.AssertEqual(t, rocmTinyLoRAFormat, identity.Labels["adapter_format"])
	core.AssertEqual(t, "1", identity.Labels["adapter_rank"])
	core.AssertEqual(t, "1", identity.Labels["adapter_alpha"])
	core.AssertEqual(t, "output.weight", identity.Labels["adapter_target"])
	core.AssertEqual(t, "2", identity.Labels["adapter_target_cols"])
	core.AssertEqual(t, "2", identity.Labels["adapter_target_rows"])
	core.AssertEqual(t, "adamw_append_only", identity.Labels["adapter_track_source"])
	core.AssertEqual(t, "rocm_adamw_track_v1", identity.Labels["adapter_track_format"])
	core.AssertEqual(t, "1", identity.Labels["adapter_track_step"])
	core.AssertEqual(t, "2", identity.Labels["adapter_track_frames"])
	core.AssertEqual(t, "LoadNativeAdamWStateTrackStep", identity.Labels["adapter_track_load_helper"])
	core.AssertEqual(t, "LoadNativeAdamWStateTrackStep", identity.Labels["adapter_track_load_step_helper"])
	var file hipTinyLoRAAdapterFile
	read := core.ReadFile(path)
	if !read.OK {
		t.Fatalf("ReadFile track snapshot: %v", read.Value)
	}
	core.AssertEqual(t, identity.Hash, identity.Labels["adapter_hash"])
	if result := core.JSONUnmarshal(read.Value.([]byte), &file); !result.OK {
		t.Fatalf("JSONUnmarshal track snapshot: %v", result.Value)
	}
	adapter, err := validateTinyLoRAAdapterFile(file, hipLoadedTinyLMConfig{HiddenSize: 2, VocabSize: 2})
	core.RequireNoError(t, err)
	core.AssertEqual(t, state.Parameters()[:2], adapter.a)
	core.AssertEqual(t, state.Parameters()[2:], adapter.b)
}

func TestNativeLoRAAdapterSnapshot_FromTrackLast_Good(t *testing.T) {
	state, err := NewNativeLoRAAdamWState(
		[]float32{1, 2},
		[]float32{3, 4},
		2,
		2,
		1,
		NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true},
	)
	core.RequireNoError(t, err)
	trackPath := core.PathJoin(t.TempDir(), "training", "lora-adapter.kv")
	_, err = AppendNativeAdamWStateTrack(trackPath, state)
	core.RequireNoError(t, err)
	core.RequireNoError(t, state.StepInPlace([][]float32{{0.5, -0.25}, {0.1, -0.2}}))
	_, err = AppendNativeAdamWStateTrack(trackPath, state)
	core.RequireNoError(t, err)
	core.RequireNoError(t, state.StepInPlace([][]float32{{0.25, -0.5}, {-0.1, 0.2}}))
	record, err := AppendNativeAdamWStateTrack(trackPath, state)
	core.RequireNoError(t, err)
	path := core.PathJoin(t.TempDir(), "adapter", "latest.json")

	identity, loadedRecord, err := SaveNativeLoRAAdapterSnapshotTrackLast(trackPath, path, NativeLoRAAdapterSnapshotConfig{
		Format: rocmTinyLoRAFormat,
		Name:   "latest",
		Rows:   2,
		Cols:   2,
		Rank:   1,
		Alpha:  1,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, record, loadedRecord)
	core.AssertEqual(t, NativeAdamWTrackContainerKV, identity.Labels["adapter_track_container"])
	core.AssertEqual(t, rocmTinyLoRAFormat, identity.Labels["adapter_format"])
	core.AssertEqual(t, "1", identity.Labels["adapter_rank"])
	core.AssertEqual(t, "1", identity.Labels["adapter_alpha"])
	core.AssertEqual(t, "output.weight", identity.Labels["adapter_target"])
	core.AssertEqual(t, "2", identity.Labels["adapter_target_cols"])
	core.AssertEqual(t, "2", identity.Labels["adapter_target_rows"])
	core.AssertEqual(t, "rocm_adamw_track_v1", identity.Labels["adapter_track_format"])
	core.AssertEqual(t, "2", identity.Labels["adapter_track_step"])
	core.AssertEqual(t, "3", identity.Labels["adapter_track_frames"])
	core.AssertEqual(t, "LoadLastNativeAdamWStateTrack", identity.Labels["adapter_track_load_helper"])
	core.AssertEqual(t, "", identity.Labels["adapter_track_load_step_helper"])
	var file hipTinyLoRAAdapterFile
	read := core.ReadFile(path)
	if !read.OK {
		t.Fatalf("ReadFile latest snapshot: %v", read.Value)
	}
	core.AssertEqual(t, identity.Hash, identity.Labels["adapter_hash"])
	if result := core.JSONUnmarshal(read.Value.([]byte), &file); !result.OK {
		t.Fatalf("JSONUnmarshal latest snapshot: %v", result.Value)
	}
	adapter, err := validateTinyLoRAAdapterFile(file, hipLoadedTinyLMConfig{HiddenSize: 2, VocabSize: 2})
	core.RequireNoError(t, err)
	core.AssertEqual(t, state.Parameters()[:2], adapter.a)
	core.AssertEqual(t, state.Parameters()[2:], adapter.b)
}

func TestNativeLoRAAdapterSnapshot_Bad(t *testing.T) {
	state, err := NewNativeLoRAAdamWState([]float32{1, 2}, []float32{3, 4}, 2, 2, 1, NativeAdamWConfig{})
	core.RequireNoError(t, err)

	_, err = SaveNativeLoRAAdapterSnapshot("", state, NativeLoRAAdapterSnapshotConfig{Rows: 2, Cols: 2, Rank: 1, Alpha: 1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "path is required")

	_, err = SaveNativeLoRAAdapterSnapshot(core.PathJoin(t.TempDir(), "adapter.json"), state, NativeLoRAAdapterSnapshotConfig{Format: "other", Rows: 2, Cols: 2, Rank: 1, Alpha: 1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "format is unsupported")

	_, err = SaveNativeLoRAAdapterSnapshot(core.PathJoin(t.TempDir(), "adapter.json"), state, NativeLoRAAdapterSnapshotConfig{Rows: 3, Cols: 2, Rank: 1, Alpha: 1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "layout does not match")
}

func BenchmarkNativeLoRAAdapterSnapshot_DirectLoRA4096Rank8(b *testing.B) {
	state := nativeAdamWBenchState(b)
	path := core.PathJoin(b.TempDir(), "adapter.json")
	cfg := NativeLoRAAdapterSnapshotConfig{
		Format: rocmTinyLoRAFormat,
		Name:   "bench-direct",
		Rows:   4096,
		Cols:   4096,
		Rank:   8,
		Alpha:  8,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identity, err := SaveNativeLoRAAdapterSnapshot(path, state, cfg)
		loraAdapterSnapshotBenchIdentity = inferenceAdapterIdentityShim{
			Path: identity.Path,
			Hash: identity.Hash,
		}
		loraAdapterSnapshotBenchErr = err
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeLoRAAdapterSnapshot_TrackLastLoRA4096Rank8(b *testing.B) {
	state := nativeAdamWBenchState(b)
	trackPath := core.PathJoin(b.TempDir(), "adapter.kv")
	gradients := [][]float32{
		make([]float32, 8*4096),
		make([]float32, 4096*8),
	}
	for i := 0; i < 4; i++ {
		if i != 0 {
			if err := state.StepInPlace(gradients); err != nil {
				b.Fatal(err)
			}
		}
		if _, err := AppendNativeAdamWStateTrack(trackPath, state); err != nil {
			b.Fatal(err)
		}
	}
	path := core.PathJoin(b.TempDir(), "adapter.json")
	cfg := NativeLoRAAdapterSnapshotConfig{
		Format: rocmTinyLoRAFormat,
		Name:   "bench-latest",
		Rows:   4096,
		Cols:   4096,
		Rank:   8,
		Alpha:  8,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identity, record, err := SaveNativeLoRAAdapterSnapshotTrackLast(trackPath, path, cfg)
		loraAdapterSnapshotBenchIdentity = inferenceAdapterIdentityShim{
			Path: identity.Path,
			Hash: identity.Hash,
		}
		loraAdapterSnapshotBenchRecord = record
		loraAdapterSnapshotBenchErr = err
		if err != nil {
			b.Fatal(err)
		}
	}
}

type inferenceAdapterIdentityShim struct {
	Path string
	Hash string
}
