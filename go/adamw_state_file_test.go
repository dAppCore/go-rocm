// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"testing"

	core "dappco.re/go"
)

func TestNativeAdamWStateFile_SaveLoadRoundTrip_Good(t *testing.T) {
	state := nativeAdamWStateFileFixture(t)
	core.RequireNoError(t, state.StepInPlace([][]float32{{0.5, -0.25}, {0.1, 0.2, 0.3}}))
	path := core.PathJoin(t.TempDir(), "nested", "adamw.bin")

	core.RequireNoError(t, SaveNativeAdamWState(path, state))
	loaded, err := LoadNativeAdamWState(path)

	core.RequireNoError(t, err)
	core.AssertEqual(t, state.Step, loaded.Step)
	core.AssertEqual(t, state.Config.LearningRate, loaded.Config.LearningRate)
	core.AssertEqual(t, state.Layout[0].Name, loaded.Layout[0].Name)
	core.AssertEqual(t, state.Layout[1].Shape, loaded.Layout[1].Shape)
	core.AssertEqual(t, len(state.Slab), len(loaded.Slab))
	assertAdamWFloat32Near(t, state.Parameters()[0], loaded.Parameters()[0], 0)
	assertAdamWFloat32Near(t, state.FirstMoment()[0], loaded.FirstMoment()[0], 0)
	assertAdamWFloat32Near(t, state.SecondMoment()[2], loaded.SecondMoment()[2], 0)
}

func TestNativeAdamWStateTrack_AppendAndLoadLast_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "adamw.track")
	state := nativeAdamWStateFileFixture(t)
	first, err := AppendNativeAdamWStateTrack(path, state)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, int(first.Offset))
	core.AssertEqual(t, 0, first.Step)

	core.RequireNoError(t, state.StepInPlace([][]float32{{0.5, -0.25}, {0.1, 0.2, 0.3}}))
	second, err := AppendNativeAdamWStateTrack(path, state)
	core.RequireNoError(t, err)
	if second.Offset <= first.Offset {
		t.Fatalf("second offset = %d, first = %d, want append-only growth", second.Offset, first.Offset)
	}
	core.AssertEqual(t, 1, second.Step)

	firstState, err := LoadNativeAdamWStateTrackAt(path, first.Offset)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, firstState.Step)
	lastState, last, err := LoadLastNativeAdamWStateTrack(path)
	core.RequireNoError(t, err)
	core.AssertEqual(t, second.Offset, last.Offset)
	core.AssertEqual(t, 1, last.Step)
	core.AssertEqual(t, 1, lastState.Step)
	assertAdamWFloat32Near(t, state.Parameters()[0], lastState.Parameters()[0], 0)

	records, err := ListNativeAdamWStateTrack(path)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []NativeAdamWTrackRecord{first, second}, records)
	stepRecord, err := FindNativeAdamWStateTrackStep(path, 1)
	core.RequireNoError(t, err)
	core.AssertEqual(t, second, stepRecord)
	stepState, stepLoaded, err := LoadNativeAdamWStateTrackStep(path, 1)
	core.RequireNoError(t, err)
	core.AssertEqual(t, second, stepLoaded)
	core.AssertEqual(t, 1, stepState.Step)
	assertAdamWFloat32Near(t, state.Parameters()[0], stepState.Parameters()[0], 0)
	for _, record := range records {
		frame, err := LoadNativeAdamWStateTrackAt(path, record.Offset)
		core.RequireNoError(t, err)
		core.AssertEqual(t, record.Step, frame.Step)
	}
}

func TestNativeAdamWStateTrack_KVAndMP4Containers_Good(t *testing.T) {
	cases := []struct {
		name      string
		file      string
		container string
	}{
		{name: "kv", file: "optimizer.kv", container: NativeAdamWTrackContainerKV},
		{name: "mp4", file: "optimizer.mp4", container: NativeAdamWTrackContainerMP4},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			path := core.PathJoin(t.TempDir(), tt.file)
			state := nativeAdamWStateFileFixture(t)
			core.RequireNoError(t, state.StepInPlace([][]float32{{0.5, -0.25}, {0.1, 0.2, 0.3}}))
			record, err := AppendNativeAdamWStateTrack(path, state)
			core.RequireNoError(t, err)
			core.AssertEqual(t, tt.container, NativeAdamWTrackContainer(path))

			records, err := ListNativeAdamWStateTrack(path)
			core.RequireNoError(t, err)
			core.AssertEqual(t, []NativeAdamWTrackRecord{record}, records)
			loaded, loadedRecord, err := LoadNativeAdamWStateTrackStep(path, state.Step)
			core.RequireNoError(t, err)
			core.AssertEqual(t, record, loadedRecord)
			core.AssertEqual(t, state.Step, loaded.Step)
			assertAdamWFloat32Near(t, state.Parameters()[0], loaded.Parameters()[0], 0)

			labels := map[string]string{}
			core.RequireNoError(t, addNativeAdamWTrackLabels(labels, path, record))
			core.AssertEqual(t, tt.container, labels["optimizer_track_container"])
			core.AssertEqual(t, "rocm_adamw_track_v1", labels["optimizer_track_format"])
		})
	}
}

func TestNativeAdamWStateFile_Bad(t *testing.T) {
	if err := SaveNativeAdamWState("", nativeAdamWStateFileFixture(t)); err == nil {
		t.Fatal("SaveNativeAdamWState(empty path) error = nil")
	}
	if _, err := LoadNativeAdamWState(""); err == nil {
		t.Fatal("LoadNativeAdamWState(empty path) error = nil")
	}
	if _, err := MarshalNativeAdamWState(nil); err == nil {
		t.Fatal("MarshalNativeAdamWState(nil) error = nil")
	}
	if _, err := UnmarshalNativeAdamWState([]byte("bad")); err == nil {
		t.Fatal("UnmarshalNativeAdamWState(bad magic) error = nil")
	}
	state := nativeAdamWStateFileFixture(t)
	payload, err := MarshalNativeAdamWState(state)
	core.RequireNoError(t, err)
	payload[0] = 'X'
	if _, err := UnmarshalNativeAdamWState(payload); err == nil {
		t.Fatal("UnmarshalNativeAdamWState(corrupt magic) error = nil")
	}
	if _, _, err := LoadLastNativeAdamWStateTrack(core.PathJoin(t.TempDir(), "missing.track")); err == nil {
		t.Fatal("LoadLastNativeAdamWStateTrack(missing) error = nil")
	}
	path := core.PathJoin(t.TempDir(), "bad.track")
	if result := core.WriteFile(path, []byte(nativeAdamWTrackMagic), 0o644); !result.OK {
		t.Fatalf("WriteFile bad track: %v", result.Value)
	}
	if _, _, err := LoadLastNativeAdamWStateTrack(path); err == nil {
		t.Fatal("LoadLastNativeAdamWStateTrack(truncated) error = nil")
	}
	if _, err := LoadNativeAdamWStateTrackAt(path, -1); err == nil {
		t.Fatal("LoadNativeAdamWStateTrackAt(negative) error = nil")
	}
	if _, err := ListNativeAdamWStateTrack(""); err == nil {
		t.Fatal("ListNativeAdamWStateTrack(empty path) error = nil")
	}
	if _, err := ListNativeAdamWStateTrack(path); err == nil {
		t.Fatal("ListNativeAdamWStateTrack(truncated) error = nil")
	}
	if _, err := FindNativeAdamWStateTrackStep(path, -1); err == nil {
		t.Fatal("FindNativeAdamWStateTrackStep(negative) error = nil")
	}
	goodPath := core.PathJoin(t.TempDir(), "good.track")
	goodState := nativeAdamWStateFileFixture(t)
	_, err = AppendNativeAdamWStateTrack(goodPath, goodState)
	core.RequireNoError(t, err)
	if _, err := FindNativeAdamWStateTrackStep(goodPath, 7); err == nil {
		t.Fatal("FindNativeAdamWStateTrackStep(missing) error = nil")
	}
	if _, _, err := LoadNativeAdamWStateTrackStep(goodPath, 7); err == nil {
		t.Fatal("LoadNativeAdamWStateTrackStep(missing) error = nil")
	}
}

func nativeAdamWStateFileFixture(t *testing.T) *NativeAdamWState {
	t.Helper()
	state, err := NewNativeAdamWState([]NativeAdamWParam{
		{Name: "lora_a", Shape: []int{1, 2}, Values: []float32{1, 2}},
		{Name: "lora_b", Shape: []int{3, 1}, Values: []float32{3, 4, 5}},
	}, NativeAdamWConfig{LearningRate: 0.1, WeightDecay: 0, WeightDecaySet: true})
	core.RequireNoError(t, err)
	return state
}
