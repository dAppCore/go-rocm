// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "testing"

func TestImageFeatureGeometryOf_Good_MatchesGemma4ProcessorAspectMath(t *testing.T) {
	cases := []struct {
		height int
		width  int
		max    int
		wantH  int
		wantW  int
		tokens int
	}{
		{height: 480, width: 640, max: 2520, wantH: 672, wantW: 912, tokens: 266},
		{height: 64, width: 64, max: 2520, wantH: 768, wantW: 768, tokens: 256},
		{height: 100, width: 1200, max: 2520, wantH: 192, wantW: 2736, tokens: 228},
		{height: 480, width: 640, max: 630, wantH: 336, wantW: 432, tokens: 63},
	}
	for _, tc := range cases {
		got, err := ImageFeatureGeometryOf(tc.height, tc.width, ImageFeatureConfig{
			MaxSoftTokens: tc.max / 9,
			DoResize:      true,
		})
		if err != nil {
			t.Fatalf("ImageFeatureGeometryOf(%dx%d): %v", tc.height, tc.width, err)
		}
		if got.TargetHeight != tc.wantH || got.TargetWidth != tc.wantW || got.SoftTokens != tc.tokens {
			t.Fatalf("ImageFeatureGeometryOf(%dx%d) = %+v, want target %dx%d tokens=%d", tc.height, tc.width, got, tc.wantH, tc.wantW, tc.tokens)
		}
	}
	if _, err := ImageFeatureGeometryOf(0, 100, ImageFeatureConfig{}); err == nil {
		t.Fatal("zero-height image geometry succeeded")
	}
}

func TestNormalizeImageFeatureConfig_Good_FillsHFDefaults(t *testing.T) {
	cfg := NormalizeImageFeatureConfig(ImageFeatureConfig{})
	if cfg.PatchSize != 16 ||
		cfg.MaxSoftTokens != 280 ||
		cfg.PoolingKernelSize != 3 ||
		cfg.RescaleFactor != 1.0/255.0 {
		t.Fatalf("NormalizeImageFeatureConfig(empty) = %+v, want Gemma4 image processor defaults", cfg)
	}
}

func TestAudioFeaturePlanOf_Good_FillsPartialConfigDefaults(t *testing.T) {
	plan, err := AudioFeaturePlanOf(AudioFeatureConfig{
		SamplingRate:  16000,
		NumMelFilters: 128,
		FFTLength:     512,
		HopLength:     160,
	})
	if err != nil {
		t.Fatalf("AudioFeaturePlanOf(partial): %v", err)
	}
	if plan.Config.FeatureSize != 128 ||
		plan.Config.FrameLength != 320 ||
		plan.Config.HopLength != 160 ||
		plan.Config.MaxFrequency != 8000 ||
		plan.Config.MelFloor != 1e-3 ||
		plan.FFTLength != 512 ||
		plan.MaxLengthSamples != 480000 ||
		plan.PadToMultiple != 128 {
		t.Fatalf("AudioFeaturePlanOf(partial) = %+v, want HF constructor defaults", plan)
	}
	frames, err := AudioFrameCount(1600, plan)
	if err != nil || frames != 10 {
		t.Fatalf("AudioFrameCount(1600) = %d, %v; want 10, nil", frames, err)
	}
	if tokens := AudioSoftTokens(frames); tokens != 3 {
		t.Fatalf("AudioSoftTokens(%d) = %d, want 3", frames, tokens)
	}
}

func TestAudioFeaturePlanOf_Bad_FailsLoud(t *testing.T) {
	if _, err := AudioFeaturePlanOf(AudioFeatureConfig{
		FrameLength:  320,
		HopLength:    160,
		FFTLength:    300,
		MaxFrequency: 8000,
	}); err == nil {
		t.Fatal("non-power-of-two FFT length succeeded")
	}
	if _, err := AudioFeaturePlanOf(AudioFeatureConfig{
		FrameLength:  320,
		HopLength:    160,
		FFTLength:    512,
		MinFrequency: 9000,
		MaxFrequency: 8000,
	}); err == nil {
		t.Fatal("empty mel band succeeded")
	}
}

func TestParseProcessorConfig_Good_NormalizesSectionsAndLabels(t *testing.T) {
	cfg, err := ParseProcessorConfig([]byte(`{
		"audio_ms_per_token": 40,
		"audio_seq_length": 750,
		"image_processor": {"max_soft_tokens": 280, "do_resize": true},
		"video_processor": {"max_soft_tokens": 70, "do_resize": true, "num_frames": 32},
		"feature_extractor": {"sampling_rate": 16000, "num_mel_filters": 128, "fft_length": 512, "hop_length": 160}
	}`))
	if err != nil {
		t.Fatalf("ParseProcessorConfig: %v", err)
	}
	if cfg.ImageProcessor == nil || cfg.VideoProcessor == nil || cfg.FeatureExtractor == nil {
		t.Fatalf("ParseProcessorConfig = %+v, want all processor sections", cfg)
	}
	labels := ApplyProcessorConfigLabels(nil, cfg)
	for key, want := range map[string]string{
		"processor_audio_ms_per_token":        "40",
		"processor_audio_seq_length":          "750",
		"image_processor":                     "true",
		"image_processor_patch_size":          "16",
		"image_processor_max_soft_tokens":     "280",
		"image_processor_pooling_kernel_size": "3",
		"image_processor_do_resize":           "true",
		"video_processor":                     "true",
		"video_processor_max_soft_tokens":     "70",
		"video_processor_num_frames":          "32",
		"audio_feature_extractor":             "true",
		"audio_feature_size":                  "128",
		"audio_feature_frame_length":          "320",
		"audio_feature_hop_length":            "160",
		"audio_feature_fft_length":            "512",
		"audio_feature_max_length_samples":    "480000",
		"audio_feature_pad_to_multiple":       "128",
	} {
		if labels[key] != want {
			t.Fatalf("labels[%q] = %q, want %q in %+v", key, labels[key], want, labels)
		}
	}
}
