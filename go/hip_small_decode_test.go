// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestHIPSmallDecode_Good_QwenGemmaSmoke(t *testing.T) {
	for _, architecture := range []string{"qwen3", "gemma3"} {
		t.Run(architecture, func(t *testing.T) {
			req := hipSmallDecodeFixture(architecture)
			want, err := hipReferenceSmallDecode(req)
			core.RequireNoError(t, err)

			driver := &fakeHIPDriver{available: true}
			got, err := hipRunSmallDecode(context.Background(), driver, req)
			core.RequireNoError(t, err)

			core.AssertEqual(t, want.TokenID, got.TokenID)
			assertFloat32Near(t, want.Score, got.Score)
			assertFloat32SlicesNear(t, want.Logits, got.Logits, 0.0001)
			assertFloat32SlicesNear(t, want.Attention, got.Attention, 0.0001)
			assertFloat32SlicesNear(t, want.UpdatedKeys, got.UpdatedKeys, 0.0001)
			assertFloat32SlicesNear(t, want.UpdatedValues, got.UpdatedValues, 0.0001)
			core.AssertEqual(t, architecture, got.Labels["decode_architecture"])
			core.AssertEqual(t, "rms_norm,projection,rope,attention,greedy", got.Labels["decode_primitives"])

			var launchNames []string
			for _, launch := range driver.launches {
				launchNames = append(launchNames, launch.Name)
			}
			joined := core.Join(",", launchNames...)
			core.AssertContains(t, joined, hipKernelNameRMSNorm)
			core.AssertContains(t, joined, hipKernelNameProjection)
			core.AssertContains(t, joined, hipKernelNameRoPE)
			core.AssertContains(t, joined, hipKernelNameAttention)
			core.AssertContains(t, joined, hipKernelNameGreedy)
		})
	}
}

func TestHIPGemma4Q4Layer0_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	got, err := hipRunGemma4Q4Layer0(context.Background(), driver, cfg, hipGemma4Q4Layer0Request{
		TokenID:  1,
		Position: 1,
		RoPEBase: 10000,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)

	core.AssertEqual(t, cfg.HiddenSize, len(got.Embedding))
	core.AssertEqual(t, cfg.HiddenSize, len(got.LayerInput))
	core.AssertEqual(t, cfg.QueryHeads*cfg.HeadDim, len(got.AttentionOutput))
	core.AssertEqual(t, cfg.HiddenSize, len(got.FinalHidden))
	core.AssertEqual(t, cfg.VocabSize, len(got.Logits))
	core.AssertEqual(t, 0, got.Greedy.TokenID)
	assertFloat32Near(t, 0, got.Greedy.Score)
	core.AssertEqual(t, hipKernelStatusLinked, got.Labels["gemma4_q4_layer0_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, got.Labels["production_decode"])
	core.AssertEqual(t, "0", got.Labels["decode_layer"])
	core.AssertContains(t, got.Labels["decode_primitives"], "mlx_q4_projection")

	var launchNames []string
	for _, launch := range driver.launches {
		launchNames = append(launchNames, launch.Name)
	}
	joined := core.Join(",", launchNames...)
	core.AssertContains(t, joined, hipKernelNameEmbedLookup)
	core.AssertContains(t, joined, hipKernelNameVectorScale)
	core.AssertContains(t, joined, hipKernelNameRMSNorm)
	core.AssertContains(t, joined, hipKernelNameMLXQ4Proj)
	core.AssertContains(t, joined, hipKernelNameRoPE)
	core.AssertContains(t, joined, hipKernelNameAttention)
	core.AssertContains(t, joined, hipKernelNameVectorAdd)
	core.AssertContains(t, joined, hipKernelNameGreedy)
	core.AssertContains(t, got.Labels["decode_primitives"], "gelu_tanh_mlp")
	attentionScales := 0
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameAttention {
			attentionScales++
			assertFloat32Near(t, 1, math.Float32frombits(binary.LittleEndian.Uint32(launch.Args[80:])))
		}
	}
	if attentionScales == 0 {
		t.Fatalf("Gemma4 q4 layer did not launch attention")
	}

	layerOnly, err := hipRunGemma4Q4DecoderLayer(context.Background(), driver, cfg, got.ScaledEmbedding, hipGemma4Q4DecoderLayerRequest{
		Position: 1,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, got.FinalHidden, layerOnly.FinalHidden, 0.0001)

	nonZeroInput := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	residualLayer, err := hipRunGemma4Q4DecoderLayer(context.Background(), driver, cfg, nonZeroInput, hipGemma4Q4DecoderLayerRequest{
		Position: 1,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, nonZeroInput, residualLayer.AttentionResidual, 0.0001)
	assertFloat32SlicesNear(t, nonZeroInput, residualLayer.FinalHidden, 0.0001)

	scaledCfg := cfg
	scaledCfg.LayerScalar = 0.5
	scaledLayer, err := hipRunGemma4Q4DecoderLayer(context.Background(), driver, scaledCfg, nonZeroInput, hipGemma4Q4DecoderLayerRequest{
		Position: 1,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4}, scaledLayer.FinalHidden, 0.0001)

	gelu, err := hipGemma4Q4HostGELU([]float32{-1, 0, 1})
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{-0.1588, 0, 0.8412}, gelu, 0.0001)

	partialRoPEStart := len(driver.launches)
	partialRoPECfg := cfg
	partialRoPECfg.RoPERotaryDim = cfg.HeadDim / 2
	_, err = hipRunGemma4Q4DecoderLayer(context.Background(), driver, partialRoPECfg, nonZeroInput, hipGemma4Q4DecoderLayerRequest{
		Position: 1,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	partialRoPELaunches := 0
	for _, launch := range driver.launches[partialRoPEStart:] {
		if launch.Name == hipKernelNameRoPE {
			partialRoPELaunches++
			core.AssertEqual(t, uint32(cfg.HeadDim), binary.LittleEndian.Uint32(launch.Args[44:]))
		}
	}
	if partialRoPELaunches == 0 {
		t.Fatalf("partial Gemma4 q4 RoPE did not launch")
	}

	perLayerStart := len(driver.launches)
	perLayerLayer, err := hipRunGemma4Q4DecoderLayer(context.Background(), driver, cfg, nonZeroInput, hipGemma4Q4DecoderLayerRequest{
		Position:      1,
		Epsilon:       1e-6,
		PerLayerInput: []float32{0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2},
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, cfg.HiddenSize, len(perLayerLayer.FinalHidden))
	perLayerQ4Launches := 0
	for _, launch := range driver.launches[perLayerStart:] {
		if launch.Name == hipKernelNameMLXQ4Proj {
			perLayerQ4Launches++
		}
	}
	core.AssertEqual(t, 9, perLayerQ4Launches)

	variable, variableCleanup := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 16)
	variable.RoPEBase = 1000000
	variable.RoPERotaryDim = 2
	variable.SlidingWindow = 0
	defer variableCleanup()
	variableLayer, err := hipRunGemma4Q4DecoderLayer(context.Background(), driver, variable, got.ScaledEmbedding, hipGemma4Q4DecoderLayerRequest{
		Position: 1,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, cfg.HiddenSize, len(variableLayer.FinalHidden))
	core.AssertEqual(t, variable.QueryHeads*variable.HeadDim, len(variableLayer.AttentionOutput))
	core.AssertEqual(t, variable.IntermediateSize, variable.GateProjection.Rows)

	forward, err := hipRunGemma4Q4SingleTokenForward(context.Background(), driver, hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg, variable}}, hipGemma4Q4ForwardRequest{
		TokenID:  1,
		Position: 1,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(forward.LayerResults))
	core.AssertEqual(t, cfg.HiddenSize, len(forward.FinalHidden))
	core.AssertEqual(t, cfg.VocabSize, len(forward.Logits))
	core.AssertEqual(t, "2", forward.Labels["decode_layers"])
	core.AssertEqual(t, hipKernelStatusNotLinked, forward.Labels["production_decode"])

	sliding := cfg
	sliding.SlidingWindow = 2
	decodeCfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{sliding, variable}}
	decodeLaunchStart := len(driver.launches)
	decode, err := hipRunGemma4Q4GreedyDecode(context.Background(), driver, decodeCfg, hipGemma4Q4GreedyDecodeRequest{
		PromptTokenIDs:    []int32{1, 0},
		MaxNewTokens:      2,
		Position:          1,
		Epsilon:           1e-6,
		MirrorDeviceKV:    true,
		DeviceKVAttention: true,
		DeviceKVMode:      rocmKVCacheModeKQ8VQ4,
	})
	core.RequireNoError(t, err)
	defer decode.DeviceState.Close()
	core.AssertEqual(t, 2, len(decode.Generated))
	core.AssertEqual(t, 3, len(decode.StepResults))
	core.AssertEqual(t, 2, len(decode.State.Layers))
	core.AssertEqual(t, cfg.HeadDim*2, len(decode.State.Layers[0].Keys))
	core.AssertEqual(t, variable.HeadDim*3, len(decode.State.Layers[1].Keys))
	core.AssertEqual(t, "2", decode.Labels["decode_prompt_tokens"])
	core.AssertEqual(t, "2", decode.Labels["decode_generated_tokens"])
	core.AssertEqual(t, "3", decode.Labels["decode_forward_steps"])
	core.AssertEqual(t, "2", decode.Labels["decode_state_tokens"])
	core.AssertEqual(t, hipKernelStatusNotLinked, decode.Labels["production_decode"])
	core.AssertEqual(t, hipKernelStatusNotLinked, decode.Labels["production_kv_cache_backing"])
	core.AssertEqual(t, "hip_device_mirror", decode.Labels["gemma4_q4_device_kv_backing"])
	core.AssertEqual(t, "2", decode.Labels["gemma4_q4_device_kv_layers"])
	core.AssertEqual(t, "2", decode.Labels["gemma4_q4_device_kv_min_tokens"])
	core.AssertEqual(t, "3", decode.Labels["gemma4_q4_device_kv_max_tokens"])
	core.AssertEqual(t, "hip_device_descriptor", decode.StepResults[0].Labels["attention_kv_backing"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, decode.StepResults[0].Labels["attention_kv_mode"])
	core.AssertEqual(t, "returned", decode.StepResults[0].Labels["gemma4_q4_forward_device_state"])
	core.AssertEqual(t, "0", decode.StepResults[0].Labels["attention_kv_append_layers"])
	core.AssertEqual(t, "2", decode.StepResults[0].Labels["attention_kv_remirror_layers"])
	core.AssertEqual(t, "2", decode.StepResults[1].Labels["attention_kv_append_layers"])
	core.AssertEqual(t, "0", decode.StepResults[1].Labels["attention_kv_remirror_layers"])
	core.AssertEqual(t, "1", decode.StepResults[2].Labels["attention_kv_append_layers"])
	core.AssertEqual(t, "1", decode.StepResults[2].Labels["attention_kv_remirror_layers"])
	core.AssertEqual(t, "1", decode.StepResults[2].Labels["gemma4_q4_device_kv_append_layers"])
	core.AssertEqual(t, "1", decode.StepResults[2].Labels["gemma4_q4_device_kv_remirror_layers"])
	if countDeviceAttentionLaunches(driver.launches[decodeLaunchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 decode launched no descriptor-backed attention kernels")
	}

	deviceState := decode.DeviceState
	if deviceState == nil {
		t.Fatalf("Gemma4 q4 decode device state is nil, want carried HIP mirror")
	}
	core.AssertEqual(t, 2, deviceState.LayerCount())
	core.AssertEqual(t, []int{2, 3}, deviceState.LayerTokenCounts())
	deviceLabels := deviceState.Labels()
	core.AssertEqual(t, "hip_device_mirror", deviceLabels["gemma4_q4_device_kv_backing"])
	core.AssertEqual(t, "2", deviceLabels["gemma4_q4_device_kv_layers"])
	core.AssertEqual(t, "2", deviceLabels["gemma4_q4_device_kv_min_tokens"])
	core.AssertEqual(t, "3", deviceLabels["gemma4_q4_device_kv_max_tokens"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, deviceLabels["gemma4_q4_device_kv_mode"])
	core.AssertEqual(t, "1", deviceLabels["gemma4_q4_device_kv_append_layers"])
	core.AssertEqual(t, "1", deviceLabels["gemma4_q4_device_kv_remirror_layers"])
	core.AssertEqual(t, hipKernelStatusNotLinked, deviceLabels["production_kv_cache_backing"])
	restoredState, err := deviceState.HostState()
	core.RequireNoError(t, err)
	assertGemma4Q4DeviceStateMatchesQuantizedHost(t, decodeCfg, decode.State, restoredState, deviceState, rocmKVCacheModeKQ8VQ4)
	freeStart := len(driver.frees)
	core.RequireNoError(t, deviceState.Close())
	if len(driver.frees)-freeStart <= 0 {
		t.Fatalf("device state close freed %d allocations, want at least one", len(driver.frees)-freeStart)
	}
	_, err = deviceState.HostState()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "closed")

	quantizedForwardStart := len(driver.launches)
	quantizedForward, err := hipRunGemma4Q4SingleTokenForward(context.Background(), driver, hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg}}, hipGemma4Q4ForwardRequest{
		TokenID:           1,
		Position:          1,
		Epsilon:           1e-6,
		DeviceKVAttention: true,
		DeviceKVMode:      rocmKVCacheModeKQ8VQ4,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "hip_device_descriptor", quantizedForward.Labels["attention_kv_backing"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, quantizedForward.Labels["attention_kv_mode"])
	core.AssertEqual(t, "0", quantizedForward.Labels["attention_kv_append_layers"])
	core.AssertEqual(t, "1", quantizedForward.Labels["attention_kv_remirror_layers"])
	core.AssertEqual(t, hipKernelStatusNotLinked, quantizedForward.Labels["production_kv_cache_backing"])
	if countDeviceAttentionLaunches(driver.launches[quantizedForwardStart:]) == 0 {
		t.Fatalf("Gemma4 q4 k-q8-v-q4 forward launched no descriptor-backed attention kernels")
	}

	partialRoPE, err := hipRunGemma4Q4RoPEVector(context.Background(), driver, []float32{1, 0, 3, 4}, 1, 1, 2)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{float32(math.Cos(1)), float32(math.Sin(1)), 3, 4}, partialRoPE, 0.0001)

	softcapped, err := hipGemma4Q4SoftcapLogits([]float32{0, 30, -30}, 30)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0, float32(math.Tanh(1) * 30), -float32(math.Tanh(1) * 30)}, softcapped, 0.0001)

	t.Setenv("GO_ROCM_GEMMA4_Q4_FORWARD_LAYERS", "2")
	layerCount, ok := gemma4Q4ForwardLayerCountFromEnv(t, 2)
	core.AssertEqual(t, true, ok)
	core.AssertEqual(t, 2, layerCount)

	t.Setenv("GO_ROCM_GEMMA4_Q4_DECODE_PROMPT_TOKENS", "1, 0")
	promptTokens := gemma4Q4DecodePromptTokensEnv(t, cfg.VocabSize)
	core.AssertEqual(t, []int32{1, 0}, promptTokens)

	parsedTokens, tokenPrompt, err := hipGemma4Q4TokenPromptIDs("tokens:1, 0", cfg.VocabSize)
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, tokenPrompt)
	core.AssertEqual(t, []int32{1, 0}, parsedTokens)
	_, tokenPrompt, err = hipGemma4Q4TokenPromptIDs("hello", cfg.VocabSize)
	core.RequireNoError(t, err)
	core.AssertEqual(t, false, tokenPrompt)

	countEmbeddingLaunches := func(start int) int {
		t.Helper()
		var count int
		for _, launch := range driver.launches[start:] {
			if launch.Name == hipKernelNameEmbedLookup {
				count++
			}
		}
		return count
	}

	launchStart := len(driver.launches)
	stream, streamErr := hipGemma4Q4GenerateTokenSeq(context.Background(), &hipLoadedModel{driver: driver}, decodeCfg, []int32{1, 0}, inference.GenerateConfig{MaxTokens: 2})
	var generated []inference.Token
	for token := range stream {
		generated = append(generated, token)
	}
	core.RequireNoError(t, streamErr())
	core.AssertEqual(t, 2, len(generated))
	core.AssertEqual(t, 3, countEmbeddingLaunches(launchStart))
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 public generate launched no descriptor-backed attention kernels")
	}
	for _, token := range generated {
		core.AssertEqual(t, int32(0), token.ID)
		core.AssertEqual(t, "<token:0>", token.Text)
	}

	tokenText := &hipTokenTextDecoder{
		vocab: map[string]int32{
			"h":                  10,
			"e":                  11,
			"he":                 12,
			"\u2581":             13,
			"z":                  14,
			"\u2581z":            15,
			"<0xE2>":             16,
			"<0x82>":             17,
			"<0xAC>":             18,
			"<unk>":              19,
			"\u2581zero":         0,
			"<eos>":              1,
			"<0xE2><0x82><0xAC>": 2,
		},
		pieces: map[int32]string{
			0:  "\u2581zero",
			1:  "<eos>",
			2:  "<0xE2><0x82><0xAC>",
			10: "h",
			11: "e",
			12: "he",
			13: "\u2581",
			14: "z",
			15: "\u2581z",
			16: "<0xE2>",
			17: "<0x82>",
			18: "<0xAC>",
			19: "<unk>",
		},
		mergeRanks:  map[string]int{"h e": 0, "\u2581 z": 1},
		special:     map[int32]bool{1: true},
		specialText: map[string]int32{"<eos>": 1},
		unknownID:   19,
		hasUnknown:  true,
		}
		core.AssertEqual(t, []int32{12, 15, 1}, tokenText.Encode("he z<eos>"))
		bosTokenText := &hipTokenTextDecoder{
			vocab: map[string]int32{
				"<bos>": 2,
				"h":     10,
				"e":     11,
				"he":    12,
			},
			pieces:      map[int32]string{2: "<bos>", 10: "h", 11: "e", 12: "he"},
			mergeRanks:  map[string]int{"h e": 0},
			special:     map[int32]bool{2: true},
			specialText: map[string]int32{"<bos>": 2},
			bosID:       2,
			hasBOS:      true,
		}
		core.AssertEqual(t, []int32{2, 12}, bosTokenText.Encode("he"))
		core.AssertEqual(t, []int32{2, 12}, bosTokenText.Encode("<bos>he"))
		textPromptTokens, textPrompt, err := hipGemma4Q4TextPromptIDs("text:he z", &hipLoadedModel{tokenText: tokenText})
		core.RequireNoError(t, err)
		core.AssertEqual(t, true, textPrompt)
	core.AssertEqual(t, []int32{12, 15}, textPromptTokens)
	_, textPrompt, err = hipGemma4Q4TextPromptIDs("he z", &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4", QuantBits: 4},
		tokenText: tokenText,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, false, textPrompt)
	t.Setenv("GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE", "1")
	textPromptTokens, textPrompt, err = hipGemma4Q4TextPromptIDs("he z", &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4", QuantBits: 4},
		tokenText: tokenText,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, textPrompt)
	core.AssertEqual(t, []int32{12, 15}, textPromptTokens)
	textPromptTokens, textPrompt, err = hipGemma4Q4TextPromptIDs(" z", &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4", QuantBits: 4},
		tokenText: tokenText,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, textPrompt)
	core.AssertEqual(t, []int32{15}, textPromptTokens)
	textPromptTokens, textPrompt, err = hipGemma4Q4TextPromptIDs("he", &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 4},
		tokenText: tokenText,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, textPrompt)
	core.AssertEqual(t, []int32{12}, textPromptTokens)
	core.AssertEqual(t, " zero", tokenText.DecodeToken(0))
	core.AssertEqual(t, "", tokenText.DecodeToken(1))
	core.AssertEqual(t, "\xe2\x82\xac", tokenText.DecodeToken(2))

	stream, streamErr = hipGemma4Q4GenerateTokenSeq(context.Background(), &hipLoadedModel{driver: driver, tokenText: tokenText}, decodeCfg, []int32{1, 0}, inference.GenerateConfig{MaxTokens: 1})
	generated = nil
	for token := range stream {
		generated = append(generated, token)
	}
	core.RequireNoError(t, streamErr())
	core.AssertEqual(t, 1, len(generated))
	core.AssertEqual(t, " zero", generated[0].Text)

	launchStart = len(driver.launches)
	freeStart = len(driver.frees)
	stream, streamErr = hipGemma4Q4GenerateTokenSeq(context.Background(), &hipLoadedModel{driver: driver, tokenText: tokenText}, decodeCfg, []int32{1, 0}, inference.GenerateConfig{MaxTokens: 2})
	generated = nil
	for token := range stream {
		generated = append(generated, token)
		break
	}
	core.RequireNoError(t, streamErr())
	core.AssertEqual(t, 1, len(generated))
	core.AssertEqual(t, 2, countEmbeddingLaunches(launchStart))
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 early-stopped public generate launched no descriptor-backed attention kernels")
	}
	if len(driver.frees) == freeStart {
		t.Fatalf("Gemma4 q4 early-stopped public generate freed no device KV allocations")
	}
}

func TestHIPGemma4Q4PerLayerInputPrecompute_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 8, 1, 8)
	defer cleanup1()
	layers, cleanupPerLayer := hipGemma4Q4GlobalPerLayerInputFixture(t, driver, []hipGemma4Q4Layer0Config{layer0, layer1})
	defer cleanupPerLayer()

	start := len(driver.launches)
	forward, err := hipRunGemma4Q4SingleTokenForward(context.Background(), driver, hipGemma4Q4ForwardConfig{Layers: layers}, hipGemma4Q4ForwardRequest{
		TokenID:  1,
		Position: 0,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipKernelStatusLinked, forward.Labels["gemma4_per_layer_inputs"])
	core.AssertEqual(t, "8", forward.Labels["gemma4_per_layer_input_size"])
	core.AssertContains(t, forward.Labels["decode_primitives"], "gemma4_per_layer_input")

	embeddingLaunches := 0
	projectionLaunches := 0
	for _, launch := range driver.launches[start:] {
		switch launch.Name {
		case hipKernelNameEmbedLookup:
			embeddingLaunches++
		case hipKernelNameProjection:
			projectionLaunches++
		}
	}
	core.AssertEqual(t, 2, embeddingLaunches)
	core.AssertEqual(t, 1, projectionLaunches)
}

func TestHIPGemma4Q4SharedKV_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	layer2, cleanup2 := hipGemma4Q4FixtureConfig(t, driver, 2, 8, 1, 8)
	defer cleanup2()
	layer3, cleanup3 := hipGemma4Q4FixtureConfig(t, driver, 3, 4, 2, 8)
	defer cleanup3()
	layer0.LayerType = "sliding_attention"
	layer1.LayerType = "full_attention"
	layer2.LayerType = "sliding_attention"
	layer3.LayerType = "full_attention"
	cfg := hipGemma4Q4ForwardConfig{
		Layers:         []hipGemma4Q4Layer0Config{layer0, layer1, layer2, layer3},
		KVSharedLayers: 2,
	}
	sources := hipGemma4Q4SharedKVSourceByLayer(cfg)
	core.AssertEqual(t, []int{0, 1, 0, 1}, sources)

	start := len(driver.launches)
	forward, err := hipRunGemma4Q4SingleTokenForward(context.Background(), driver, cfg, hipGemma4Q4ForwardRequest{
		TokenID:  1,
		Position: 0,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "2", forward.Labels["gemma4_q4_kv_shared_layers"])
	core.AssertEqual(t, "2", forward.Labels["gemma4_q4_kv_shared_runtime_layers"])
	core.AssertContains(t, forward.Labels["decode_primitives"], "gemma4_shared_kv")
	core.AssertEqual(t, layer0.HeadDim, len(forward.LayerResults[2].UpdatedKeys))
	core.AssertEqual(t, layer1.HeadDim, len(forward.LayerResults[3].UpdatedKeys))

	q4Launches := 0
	for _, launch := range driver.launches[start:] {
		if launch.Name == hipKernelNameMLXQ4Proj {
			q4Launches++
		}
	}
	core.AssertEqual(t, 25, q4Launches)
}

func TestHIPGemma4Q4PackagePrefillDecode_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 16)
	defer cleanup1()
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}
	model := &hipLoadedModel{driver: driver}

	launchStart := len(driver.launches)
	prefill, err := hipRunGemma4Q4PackagePrefill(context.Background(), model, cfg, hipPrefillRequest{
		TokenIDs: []int32{1, 0},
	})
	core.RequireNoError(t, err)
	defer prefill.Gemma4Q4DeviceState.Close()

	core.AssertEqual(t, 2, prefill.PromptTokens)
	core.AssertEqual(t, layer0.VocabSize, len(prefill.Logits))
	core.AssertEqual(t, 2, len(prefill.Gemma4Q4State.Layers))
	core.AssertEqual(t, []int{2, 2}, prefill.Gemma4Q4DeviceState.LayerTokenCounts())
	core.AssertEqual(t, "loaded_gemma4_q4_experimental_prefill", prefill.Labels["kernel_scope"])
	core.AssertEqual(t, hipKernelStatusLinked, prefill.Labels["gemma4_q4_prefill_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, prefill.Labels["prefill_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, prefill.Labels["production_prefill"])
	core.AssertEqual(t, hipKernelStatusNotLinked, prefill.Labels["production_decode"])
	core.AssertEqual(t, hipKernelStatusNotLinked, prefill.Labels["production_kv_cache_backing"])
	core.AssertEqual(t, "hip_device_descriptor", prefill.Labels["attention_kv_backing"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, prefill.Labels["attention_kv_mode"])
	core.AssertEqual(t, "forward_returned_device_state", prefill.Labels["gemma4_q4_device_kv_state"])
	core.AssertEqual(t, "2", prefill.Labels["prefill_prompt_tokens"])
	core.AssertEqual(t, "1", prefill.Labels["decode_position"])
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 package prefill launched no descriptor-backed attention kernels")
	}

	prefillDeviceState := prefill.Gemma4Q4DeviceState
	launchStart = len(driver.launches)
	decode, err := hipRunGemma4Q4PackageDecode(context.Background(), model, cfg, hipDecodeRequest{
		TokenID:             int32(0),
		Gemma4Q4State:       prefill.Gemma4Q4State,
		Gemma4Q4DeviceState: prefillDeviceState,
	})
	core.RequireNoError(t, err)
	defer decode.Gemma4Q4DeviceState.Close()

	core.AssertEqual(t, int32(0), decode.Token.ID)
	core.AssertEqual(t, "<token:0>", decode.Token.Text)
	core.AssertEqual(t, layer0.VocabSize, len(decode.Logits))
	core.AssertEqual(t, 2, len(decode.Gemma4Q4State.Layers))
	core.AssertEqual(t, []int{3, 3}, decode.Gemma4Q4DeviceState.LayerTokenCounts())
	core.AssertEqual(t, true, prefillDeviceState.closed)
	core.AssertEqual(t, "loaded_gemma4_q4_experimental_decode", decode.Labels["kernel_scope"])
	core.AssertEqual(t, hipKernelStatusLinked, decode.Labels["gemma4_q4_decode_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, decode.Labels["decode_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, decode.Labels["production_prefill"])
	core.AssertEqual(t, hipKernelStatusNotLinked, decode.Labels["production_decode"])
	core.AssertEqual(t, hipKernelStatusNotLinked, decode.Labels["production_kv_cache_backing"])
	core.AssertEqual(t, "hip_device_descriptor", decode.Labels["attention_kv_backing"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, decode.Labels["attention_kv_mode"])
	core.AssertEqual(t, "forward_returned_device_state", decode.Labels["gemma4_q4_device_kv_state"])
	core.AssertEqual(t, "3", decode.Labels["decode_state_tokens"])
	core.AssertEqual(t, "2", decode.Labels["decode_position"])
	core.AssertEqual(t, "2", decode.Labels["attention_kv_append_layers"])
	core.AssertEqual(t, "0", decode.Labels["attention_kv_remirror_layers"])
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 package decode launched no descriptor-backed attention kernels")
	}

	launchStart = len(driver.launches)
	batch := hipGemma4Q4BatchGenerate(context.Background(), model, cfg, []string{"tokens:1,0", "plain"}, inference.GenerateConfig{MaxTokens: 1})
	core.AssertEqual(t, 2, len(batch))
	core.AssertEqual(t, 1, len(batch[0].Tokens))
	core.AssertEqual(t, int32(0), batch[0].Tokens[0].ID)
	core.RequireNoError(t, batch[0].Err)
	core.AssertError(t, batch[1].Err)
	core.AssertContains(t, batch[1].Err.Error(), "native decode kernels are not linked yet")
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 batch generate launched no descriptor-backed attention kernels")
	}

	model.modelInfo = inference.ModelInfo{Architecture: "gemma4", QuantBits: 4, VocabSize: layer0.VocabSize}
	model.tokenText = &hipTokenTextDecoder{
		vocab: map[string]int32{
			"a": 1,
			"b": 0,
		},
		pieces: map[int32]string{
			0: "b",
			1: "a",
		},
	}
	t.Setenv("GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE", "1")
	launchStart = len(driver.launches)
	textBatch := hipGemma4Q4BatchGenerate(context.Background(), model, cfg, []string{"text:a", "a"}, inference.GenerateConfig{MaxTokens: 1})
	core.AssertEqual(t, 2, len(textBatch))
	for index, result := range textBatch {
		core.RequireNoError(t, result.Err)
		if len(result.Tokens) != 1 || result.Tokens[0].ID < 0 || int(result.Tokens[0].ID) >= layer0.VocabSize {
			t.Fatalf("Gemma4 q4 text batch result[%d] = %+v, want one in-vocab token", index, result)
		}
		core.AssertEqual(t, "b", result.Tokens[0].Text)
	}
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 text batch generate launched no descriptor-backed attention kernels")
	}

	launchStart = len(driver.launches)
	classify, err := hipGemma4Q4Classify(context.Background(), model, cfg, []string{"tokens:1,0"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(classify))
	core.AssertEqual(t, int32(0), classify[0].Token.ID)
	core.AssertEqual(t, layer0.VocabSize, len(classify[0].Logits))
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 classify launched no descriptor-backed attention kernels")
	}
}

func assertFloat32SlicesNearRelative(t *testing.T, want, got []float32, absoluteTolerance, relativeTolerance float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("slice len = %d, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		tolerance := absoluteTolerance
		if scaled := float32(math.Abs(float64(want[i]))) * relativeTolerance; scaled > tolerance {
			tolerance = scaled
		}
		if math.Abs(float64(want[i]-got[i])) > float64(tolerance) {
			t.Fatalf("slice[%d] = %f, want %f within abs=%f rel=%f; got %+v", i, got[i], want[i], absoluteTolerance, relativeTolerance, got)
		}
	}
}

func countDeviceAttentionLaunches(launches []hipKernelLaunchConfig) int {
	var count int
	for _, launch := range launches {
		if launch.Name == hipKernelNameAttention &&
			len(launch.Args) >= hipAttentionLaunchArgsBytes &&
			binary.LittleEndian.Uint32(launch.Args[76:]) == hipAttentionKVSourceDevice {
			count++
		}
	}
	return count
}

func assertGemma4Q4DeviceStateMatchesQuantizedHost(t *testing.T, cfg hipGemma4Q4ForwardConfig, hostState, restoredState hipGemma4Q4DecodeState, deviceState *hipGemma4Q4DeviceDecodeState, mode string) {
	t.Helper()
	core.AssertEqual(t, len(hostState.Layers), len(restoredState.Layers))
	if deviceState == nil {
		t.Fatalf("device state is nil")
	}
	core.AssertEqual(t, len(hostState.Layers), len(deviceState.layers))
	for index := range hostState.Layers {
		cache, err := newROCmKVCache(mode, defaultROCmKVBlockSize)
		core.RequireNoError(t, err)
		layerCfg := cfg.Layers[index]
		for _, page := range deviceState.layers[index].cache.pages {
			keyStart := page.tokenStart * layerCfg.HeadDim
			keyEnd := keyStart + page.tokenCount*layerCfg.HeadDim
			if keyStart < 0 || keyEnd > len(hostState.Layers[index].Keys) {
				t.Fatalf("device layer %d page token range [%d,%d) exceeds host key length %d", index, keyStart, keyEnd, len(hostState.Layers[index].Keys))
			}
			valueStart := page.tokenStart * layerCfg.HeadDim
			valueEnd := valueStart + page.tokenCount*layerCfg.HeadDim
			if valueStart < 0 || valueEnd > len(hostState.Layers[index].Values) {
				t.Fatalf("device layer %d page token range [%d,%d) exceeds host value length %d", index, valueStart, valueEnd, len(hostState.Layers[index].Values))
			}
			core.RequireNoError(t, cache.AppendVectors(page.tokenStart, layerCfg.HeadDim, layerCfg.HeadDim, hostState.Layers[index].Keys[keyStart:keyEnd], hostState.Layers[index].Values[valueStart:valueEnd]))
		}
		wantKeys, wantValues, err := cache.Restore(0, cache.TokenCount())
		core.RequireNoError(t, err)
		assertFloat32SlicesNearRelative(t, wantKeys, restoredState.Layers[index].Keys, 0.0001, 0.0001)
		assertFloat32SlicesNearRelative(t, wantValues, restoredState.Layers[index].Values, 0.0001, 0.0001)
	}
}

func TestHIPGemma4Q4Layer0_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	_, err := hipRunGemma4Q4Layer0(context.Background(), driver, cfg, hipGemma4Q4Layer0Request{
		TokenID:  int32(cfg.VocabSize),
		Position: 1,
		RoPEBase: 10000,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside vocabulary")

	_, err = hipRunGemma4Q4Layer0(context.Background(), driver, cfg, hipGemma4Q4Layer0Request{
		TokenID:  1,
		Position: 1,
		RoPEBase: float32(math.NaN()),
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	badCfg := cfg
	badCfg.QueryProjection.WeightPointer = 0
	_, err = hipRunGemma4Q4Layer0(context.Background(), driver, badCfg, hipGemma4Q4Layer0Request{
		TokenID:  1,
		Position: 1,
		RoPEBase: 10000,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "q_proj config")

	badCfg = cfg
	badCfg.Layer = -1
	_, err = hipRunGemma4Q4Layer0(context.Background(), driver, badCfg, hipGemma4Q4Layer0Request{
		TokenID:  1,
		Position: 1,
		RoPEBase: 10000,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "layer index")

	decodeCfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg}}
	validState := hipGemma4Q4DecodeState{Layers: []hipGemma4Q4LayerKVState{{
		Keys:   make([]float32, cfg.HeadDim),
		Values: make([]float32, cfg.HeadDim),
	}}}
	_, err = hipMirrorGemma4Q4DecodeState(nil, decodeCfg, validState, "")
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "HIP driver is nil")

	_, err = hipMirrorGemma4Q4DecodeState(&fakeHIPDriver{available: false}, decodeCfg, validState, "")
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "HIP driver is not available")

	_, err = hipMirrorGemma4Q4DecodeState(driver, decodeCfg, hipGemma4Q4DecodeState{}, "")
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "decode state has no layers")

	_, err = hipMirrorGemma4Q4DecodeState(driver, decodeCfg, hipGemma4Q4DecodeState{Layers: []hipGemma4Q4LayerKVState{{}}}, "")
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "no KV tokens")

	_, err = hipMirrorGemma4Q4DecodeState(driver, decodeCfg, validState, "bad")
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported cache mode")

	deviceState, err := hipMirrorGemma4Q4DecodeState(driver, decodeCfg, validState, "")
	core.RequireNoError(t, err)
	defer deviceState.Close()
	_, err = hipRunGemma4Q4DecoderLayer(context.Background(), driver, cfg, make([]float32, cfg.HiddenSize), hipGemma4Q4DecoderLayerRequest{
		Position:          1,
		Epsilon:           1e-6,
		PriorKeys:         validState.Layers[0].Keys,
		PriorValues:       validState.Layers[0].Values,
		DeviceKVAttention: true,
		DeviceKVMode:      rocmKVCacheModeQ8,
		PriorDeviceKV:     deviceState.layerCache(0),
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prior device KV mode mismatch")

	_, err = hipRunGemma4Q4GreedyDecode(context.Background(), driver, decodeCfg, hipGemma4Q4GreedyDecodeRequest{
		PromptTokenIDs: []int32{1},
		MaxNewTokens:   1,
		MirrorDeviceKV: true,
		DeviceKVMode:   "bad",
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported device KV cache mode")

	_, _, err = hipRunGemma4Q4SingleTokenForwardWithState(context.Background(), driver, decodeCfg, validState, hipGemma4Q4ForwardRequest{
		TokenID:          1,
		Position:         1,
		Epsilon:          1e-6,
		PriorDeviceState: &hipGemma4Q4DeviceDecodeState{},
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prior device state requires device KV attention")

	_, _, err = hipRunGemma4Q4SingleTokenForwardWithState(context.Background(), driver, decodeCfg, validState, hipGemma4Q4ForwardRequest{
		TokenID:           1,
		Position:          1,
		Epsilon:           1e-6,
		ReturnDeviceState: true,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "returning device state requires device KV attention")

	badCfg = cfg
	badCfg.RoPEBase = -1
	_, err = hipRunGemma4Q4Layer0(context.Background(), driver, badCfg, hipGemma4Q4Layer0Request{
		TokenID:  1,
		Position: 1,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "layer RoPE base")

	badCfg = cfg
	badCfg.RoPERotaryDim = 3
	_, err = hipRunGemma4Q4Layer0(context.Background(), driver, badCfg, hipGemma4Q4Layer0Request{
		TokenID:  1,
		Position: 1,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rotary dimension")

	badCfg = cfg
	badCfg.FinalLogitSoftcap = float32(math.Inf(1))
	_, err = hipRunGemma4Q4Layer0(context.Background(), driver, badCfg, hipGemma4Q4Layer0Request{
		TokenID:  1,
		Position: 1,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "softcap")

	_, err = hipRunGemma4Q4DecoderLayer(context.Background(), driver, cfg, []float32{1}, hipGemma4Q4DecoderLayerRequest{
		Position: 1,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input length")

	_, err = hipRunGemma4Q4DecoderLayer(context.Background(), driver, cfg, make([]float32, cfg.HiddenSize), hipGemma4Q4DecoderLayerRequest{
		Position:    1,
		Epsilon:     1e-6,
		PriorKeys:   []float32{1},
		PriorValues: []float32{1},
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prior key/value")

	_, err = hipRunGemma4Q4SingleTokenForward(context.Background(), driver, hipGemma4Q4ForwardConfig{}, hipGemma4Q4ForwardRequest{
		TokenID:  1,
		Position: 1,
		RoPEBase: 10000,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "at least one")

	_, err = hipRunGemma4Q4GreedyDecode(context.Background(), driver, hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg}}, hipGemma4Q4GreedyDecodeRequest{
		MaxNewTokens: 1,
		Epsilon:      1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompt token")

	_, err = hipRunGemma4Q4GreedyDecode(context.Background(), driver, hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg}}, hipGemma4Q4GreedyDecodeRequest{
		PromptTokenIDs: []int32{1},
		Epsilon:        1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "max new tokens")

	_, tokenPrompt, err := hipGemma4Q4TokenPromptIDs("tokens:", cfg.VocabSize)
	core.AssertEqual(t, true, tokenPrompt)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "at least one")

	_, tokenPrompt, err = hipGemma4Q4TokenPromptIDs("tokens:999", cfg.VocabSize)
	core.AssertEqual(t, true, tokenPrompt)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside vocabulary")

	_, textPrompt, err := hipGemma4Q4TextPromptIDs("text:", &hipLoadedModel{})
	core.AssertEqual(t, true, textPrompt)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompt text")
}

func TestHIPGemma4Q4PackagePrefillDecode_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()
	forwardCfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg}}
	model := &hipLoadedModel{driver: driver}

	_, err := hipRunGemma4Q4PackagePrefill(context.Background(), model, forwardCfg, hipPrefillRequest{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompt or token IDs are required")

	_, err = hipRunGemma4Q4PackagePrefill(context.Background(), model, forwardCfg, hipPrefillRequest{
		TokenIDs:  []int32{1},
		CacheMode: "bad",
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported cache mode")

	_, err = hipRunGemma4Q4PackagePrefill(context.Background(), model, forwardCfg, hipPrefillRequest{
		TokenIDs: []int32{1},
		KeyWidth: cfg.HeadDim + 1,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV widths")

	_, err = hipRunGemma4Q4PackageDecode(context.Background(), model, forwardCfg, hipDecodeRequest{TokenID: 1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "Gemma4 q4 decode state is required")

	validPrefill, err := hipRunGemma4Q4PackagePrefill(context.Background(), model, forwardCfg, hipPrefillRequest{TokenIDs: []int32{1}})
	core.RequireNoError(t, err)
	defer validPrefill.Gemma4Q4DeviceState.Close()

	_, err = hipRunGemma4Q4PackageDecode(context.Background(), model, forwardCfg, hipDecodeRequest{
		TokenID:       1,
		DeviceKVMode:  "bad",
		Gemma4Q4State: validPrefill.Gemma4Q4State,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported device KV cache mode")

	_, err = hipRunGemma4Q4PackageDecode(context.Background(), model, forwardCfg, hipDecodeRequest{
		TokenID:       1,
		Position:      -1,
		Gemma4Q4State: validPrefill.Gemma4Q4State,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "decode position")

	core.RequireNoError(t, validPrefill.Gemma4Q4DeviceState.Close())
	_, err = hipRunGemma4Q4PackageDecode(context.Background(), model, forwardCfg, hipDecodeRequest{
		TokenID:             1,
		Gemma4Q4State:       validPrefill.Gemma4Q4State,
		Gemma4Q4DeviceState: validPrefill.Gemma4Q4DeviceState,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "device decode state is closed")
}

func TestHIPSmallDecode_Bad(t *testing.T) {
	_, err := hipReferenceSmallDecode(hipSmallDecodeFixture("llama"))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "Qwen or Gemma")

	req := hipSmallDecodeFixture("qwen3")
	req.Position = 99
	_, err = hipReferenceSmallDecode(req)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "decode position")

	req = hipSmallDecodeFixture("qwen3")
	req.Epsilon = float32(math.NaN())
	_, err = hipReferenceSmallDecode(req)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	req = hipSmallDecodeFixture("qwen3")
	req.RoPEBase = float32(math.Inf(1))
	_, err = hipReferenceSmallDecode(req)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	req = hipSmallDecodeFixture("qwen3")
	req.QueryFP16 = req.QueryFP16[:1]
	_, err = hipRunSmallDecode(context.Background(), &fakeHIPDriver{available: true}, req)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "query projection weight length")

	_, err = hipRunSmallDecode(context.Background(), &fakeHIPDriver{}, hipSmallDecodeFixture("qwen3"))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "HIP driver is not available")
}

func TestHIPRuntime_LoadedSmallDecodeRequestFiniteValidation_Bad(t *testing.T) {
	loaded, _ := hipLoadedSmallDecodeFixture(t, "qwen3")
	defer loaded.Close()
	cfg, err := loaded.loadedSmallDecodeConfig()
	core.RequireNoError(t, err)
	smoke := hipSmallDecodeFixture("qwen3")

	_, err = hipRunLoadedSmallDecode(context.Background(), loaded.driver, cfg, hipLoadedSmallDecodeRequest{
		Input:       smoke.Input,
		PriorKeys:   smoke.PriorKeys,
		PriorValues: smoke.PriorValues,
		Position:    smoke.Position,
		RoPEBase:    smoke.RoPEBase,
		Epsilon:     float32(math.NaN()),
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = hipRunLoadedSmallDecode(context.Background(), loaded.driver, cfg, hipLoadedSmallDecodeRequest{
		Input:       smoke.Input,
		PriorKeys:   smoke.PriorKeys,
		PriorValues: smoke.PriorValues,
		Position:    smoke.Position,
		RoPEBase:    float32(math.Inf(1)),
		Epsilon:     smoke.Epsilon,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")
}

func TestHIPRuntime_LoadedSmallDecodeEmbeddingReadFiniteValidation_Bad(t *testing.T) {
	loaded, driver := hipLoadedSmallDecodeFixture(t, "qwen3")
	defer loaded.Close()
	cfg, err := loaded.loadedSmallDecodeConfig()
	core.RequireNoError(t, err)

	payload, err := hipFloat32Payload([]float32{1, float32(math.NaN())})
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(cfg.EmbeddingPointer, payload))

	_, err = hipReadLoadedSmallEmbedding(context.Background(), driver, cfg, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "embedding row values must be finite")
}

func TestHIPRuntime_LoadedEmbeddingTableFiniteValidation_Bad(t *testing.T) {
	loaded, driver := hipLoadedSmallDecodeFixture(t, "qwen3")
	defer loaded.Close()
	cfg, err := loaded.loadedEmbeddingConfig()
	core.RequireNoError(t, err)

	payload, err := hipFloat32Payload([]float32{1, float32(math.Inf(1))})
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(cfg.EmbeddingPointer, payload))

	_, err = loaded.loadedEmbeddingTable(cfg)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "embedding table values must be finite")
}

func TestHIPRuntime_LoadModelRunsSmallDecodeSmokeWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-small-decode.hsaco")
	loaded, driver := hipLoadedSmallDecodeFixture(t, "qwen3")
	defer loaded.Close()

	cfg, err := loaded.loadedSmallDecodeConfig()
	core.RequireNoError(t, err)
	smoke := hipSmallDecodeFixture("qwen3")
	want, err := hipReferenceSmallDecode(smoke)
	core.RequireNoError(t, err)
	got, err := hipRunLoadedSmallDecode(context.Background(), loaded.driver, cfg, hipLoadedSmallDecodeRequest{
		Input:       smoke.Input,
		PriorKeys:   smoke.PriorKeys,
		PriorValues: smoke.PriorValues,
		Position:    smoke.Position,
		RoPEBase:    smoke.RoPEBase,
		Epsilon:     smoke.Epsilon,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, want.TokenID, got.TokenID)
	assertFloat32Near(t, want.Score, got.Score)
	assertFloat32SlicesNear(t, want.Logits, got.Logits, 0.0001)
	assertFloat32SlicesNear(t, want.Attention, got.Attention, 0.0001)
	assertFloat32SlicesNear(t, want.UpdatedKeys, got.UpdatedKeys, 0.0001)
	assertFloat32SlicesNear(t, want.UpdatedValues, got.UpdatedValues, 0.0001)
	core.AssertEqual(t, "loaded_device", got.Labels["decode_tensor_backing"])

	cache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, smoke.HiddenSize, smoke.HiddenSize, smoke.PriorKeys, smoke.PriorValues))
	decoded, err := loaded.DecodeToken(context.Background(), hipDecodeRequest{TokenID: 2, KV: cache})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(want.TokenID), decoded.Token.ID)
	core.AssertEqual(t, 3, decoded.KV.TokenCount())
	if decoded.KV != cache {
		t.Fatalf("decoded KV cache = %p, want original cache %p", decoded.KV, cache)
	}
	decodedKeys, decodedValues, err := decoded.KV.Restore(0, decoded.KV.TokenCount())
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want.Logits, decoded.Logits, 0.0001)
	assertFloat32SlicesNear(t, want.UpdatedKeys, decodedKeys, 0.0005)
	assertFloat32SlicesNear(t, want.UpdatedValues, decodedValues, 0.0005)
	core.AssertEqual(t, "loaded_device", decoded.Labels["decode_tensor_backing"])
	core.AssertEqual(t, "2", decoded.Labels["decode_launch_token"])

	deviceCache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, deviceCache.AppendVectors(0, smoke.HiddenSize, smoke.HiddenSize, smoke.PriorKeys, smoke.PriorValues))
	deviceKV, table, err := hipMirrorTinyKV(driver, deviceCache, map[string]string{})
	core.RequireNoError(t, err)
	defer deviceKV.Close()
	defer table.Close()
	decodedWithDevice, err := loaded.DecodeToken(context.Background(), hipDecodeRequest{
		TokenID:         2,
		KV:              deviceCache,
		DeviceKV:        deviceKV,
		DescriptorTable: table,
	})
	core.RequireNoError(t, err)
	defer decodedWithDevice.DeviceKV.Close()
	defer decodedWithDevice.DescriptorTable.Close()
	if decodedWithDevice.KV == deviceCache {
		t.Fatalf("decoded device KV cache = original cache %p, want cloned host cache", deviceCache)
	}
	core.AssertEqual(t, 2, deviceCache.TokenCount())
	core.AssertEqual(t, 3, decodedWithDevice.KV.TokenCount())
	core.AssertEqual(t, 3, decodedWithDevice.DeviceKV.TokenCount())
	if !deviceKV.closed || !table.closed {
		t.Fatalf("original device resources should be closed after successful small decode device append")
	}
	deviceDecodedKeys, deviceDecodedValues, err := decodedWithDevice.KV.Restore(0, decodedWithDevice.KV.TokenCount())
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want.Logits, decodedWithDevice.Logits, 0.0001)
	assertFloat32SlicesNear(t, want.UpdatedKeys, deviceDecodedKeys, 0.0005)
	assertFloat32SlicesNear(t, want.UpdatedValues, deviceDecodedValues, 0.0005)
	core.AssertEqual(t, "loaded_device", decodedWithDevice.Labels["decode_tensor_backing"])
	core.AssertEqual(t, "hip_device", decodedWithDevice.Labels["kv_descriptor_table"])
	core.AssertEqual(t, "append_token", decodedWithDevice.Labels["kv_device_update"])
	core.AssertEqual(t, "1", decodedWithDevice.Labels["kv_device_update_pages"])
	core.AssertEqual(t, "1", decodedWithDevice.Labels["kv_device_update_from_pages"])
	core.AssertEqual(t, "2", decodedWithDevice.Labels["kv_device_update_from_tokens"])
	core.AssertEqual(t, "2", decodedWithDevice.Labels["kv_device_update_to_pages"])
	core.AssertEqual(t, "3", decodedWithDevice.Labels["kv_device_update_to_tokens"])
	core.AssertEqual(t, "success", decodedWithDevice.Labels["kv_device_update_descriptor_refresh"])
	core.AssertEqual(t, "3", decodedWithDevice.Labels["kv_tokens"])

	for _, tt := range []struct {
		mode           string
		keyTolerance   float32
		valueTolerance float32
	}{
		{mode: rocmKVCacheModeQ8, keyTolerance: 0.01, valueTolerance: 0.03},
		{mode: rocmKVCacheModeKQ8VQ4, keyTolerance: 0.01, valueTolerance: 0.15},
	} {
		t.Run("typed-"+tt.mode, func(t *testing.T) {
			modeCache, err := newROCmKVCache(tt.mode, defaultROCmKVBlockSize)
			core.RequireNoError(t, err)
			core.RequireNoError(t, modeCache.AppendVectors(0, smoke.HiddenSize, smoke.HiddenSize, smoke.PriorKeys, smoke.PriorValues))
			modeDecoded, err := loaded.DecodeToken(context.Background(), hipDecodeRequest{TokenID: 2, KV: modeCache})
			core.RequireNoError(t, err)
			core.AssertEqual(t, int32(want.TokenID), modeDecoded.Token.ID)
			core.AssertEqual(t, 3, modeDecoded.KV.TokenCount())
			core.AssertEqual(t, tt.mode, modeDecoded.KV.Stats().CacheMode)
			modeKeys, modeValues, err := modeDecoded.KV.Restore(0, modeDecoded.KV.TokenCount())
			core.RequireNoError(t, err)
			assertFloat32SlicesNear(t, want.Logits, modeDecoded.Logits, 0.0001)
			assertFloat32SlicesNear(t, want.UpdatedKeys, modeKeys, tt.keyTolerance)
			assertFloat32SlicesNear(t, want.UpdatedValues, modeValues, tt.valueTolerance)

			modeDeviceCache, err := newROCmKVCache(tt.mode, defaultROCmKVBlockSize)
			core.RequireNoError(t, err)
			core.RequireNoError(t, modeDeviceCache.AppendVectors(0, smoke.HiddenSize, smoke.HiddenSize, smoke.PriorKeys, smoke.PriorValues))
			modeDeviceKV, modeTable, err := hipMirrorTinyKV(driver, modeDeviceCache, map[string]string{})
			core.RequireNoError(t, err)
			defer modeDeviceKV.Close()
			defer modeTable.Close()
			modeDecodedWithDevice, err := loaded.DecodeToken(context.Background(), hipDecodeRequest{
				TokenID:         2,
				KV:              modeDeviceCache,
				DeviceKV:        modeDeviceKV,
				DescriptorTable: modeTable,
			})
			core.RequireNoError(t, err)
			defer modeDecodedWithDevice.DeviceKV.Close()
			defer modeDecodedWithDevice.DescriptorTable.Close()
			core.AssertEqual(t, int32(want.TokenID), modeDecodedWithDevice.Token.ID)
			core.AssertEqual(t, 2, modeDeviceCache.TokenCount())
			core.AssertEqual(t, 3, modeDecodedWithDevice.KV.TokenCount())
			core.AssertEqual(t, 3, modeDecodedWithDevice.DeviceKV.TokenCount())
			core.AssertEqual(t, tt.mode, modeDecodedWithDevice.KV.Stats().CacheMode)
			core.AssertEqual(t, tt.mode, modeDecodedWithDevice.DeviceKV.Stats().CacheMode)
			if !modeDeviceKV.closed || !modeTable.closed {
				t.Fatalf("original %s device resources should be closed after successful small decode device append", tt.mode)
			}
			modeDeviceKeys, modeDeviceValues, err := modeDecodedWithDevice.KV.Restore(0, modeDecodedWithDevice.KV.TokenCount())
			core.RequireNoError(t, err)
			assertFloat32SlicesNear(t, want.Logits, modeDecodedWithDevice.Logits, 0.0001)
			assertFloat32SlicesNear(t, want.UpdatedKeys, modeDeviceKeys, tt.keyTolerance)
			assertFloat32SlicesNear(t, want.UpdatedValues, modeDeviceValues, tt.valueTolerance)
			core.AssertEqual(t, "hip_device", modeDecodedWithDevice.Labels["kv_descriptor_table"])
			core.AssertEqual(t, "append_token", modeDecodedWithDevice.Labels["kv_device_update"])
			core.AssertEqual(t, "2", modeDecodedWithDevice.Labels["kv_device_update_to_pages"])
			core.AssertEqual(t, "3", modeDecodedWithDevice.Labels["kv_device_update_to_tokens"])
			core.AssertEqual(t, "success", modeDecodedWithDevice.Labels["kv_device_update_descriptor_refresh"])
			core.AssertEqual(t, "3", modeDecodedWithDevice.Labels["kv_tokens"])
		})
	}

	rmsPointer := loaded.tensors["model.layers.0.input_layernorm.weight"].pointer
	queryPointer := loaded.tensors["model.layers.0.self_attn.q_proj.weight"].pointer
	lmHeadPointer := loaded.tensors["output.weight"].pointer
	var sawRMSWeight, sawQueryWeight, sawLMHead bool
	for _, launch := range driver.launches {
		switch launch.Name {
		case hipKernelNameRMSNorm:
			if nativeDevicePointer(binary.LittleEndian.Uint64(launch.Args[16:])) == rmsPointer {
				sawRMSWeight = true
			}
		case hipKernelNameProjection:
			weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(launch.Args[24:]))
			if weightPointer == queryPointer {
				sawQueryWeight = true
			}
			if weightPointer == lmHeadPointer {
				sawLMHead = true
			}
		}
	}
	core.AssertTrue(t, sawRMSWeight)
	core.AssertTrue(t, sawQueryWeight)
	core.AssertTrue(t, sawLMHead)
}

func TestHIPRuntime_LoadModelRunsSmallDecodeLoRAAdapterWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-small-decode-lora.hsaco")
	loaded, driver := hipLoadedSmallDecodeFixture(t, "qwen3")
	defer loaded.Close()

	status := loaded.KernelStatus()
	core.AssertEqual(t, hipKernelStatusLinked, status.LoRA)
	adapterPath := core.PathJoin(t.TempDir(), "rocm_lm_head_lora.json")
	writeTinyLoRAAdapterFile(t, adapterPath, `{
		"format":"rocm-small-lm-head-lora",
		"name":"boost-zero",
		"target":"lm_head.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"vocab_size":3,
		"lora_a":[0,0],
		"lora_b":[0,0,0],
		"bias":[10,0,0]
	}`)
	identity, err := loaded.LoadAdapter(adapterPath)
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmSmallLoRAFormat, identity.Format)
	core.AssertEqual(t, "hip_small_lm_head", identity.Labels["adapter_runtime"])
	core.AssertEqual(t, hipKernelNameLoRA, identity.Labels["lora_kernel_name"])

	smoke := hipSmallDecodeFixture("qwen3")
	want, err := hipReferenceSmallDecode(smoke)
	core.RequireNoError(t, err)
	want.Logits[0] += 10
	cache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, smoke.HiddenSize, smoke.HiddenSize, smoke.PriorKeys, smoke.PriorValues))
	decoded, err := loaded.DecodeToken(context.Background(), hipDecodeRequest{TokenID: 2, KV: cache})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(0), decoded.Token.ID)
	core.AssertEqual(t, identity.Hash, decoded.Labels["adapter_hash"])
	core.AssertEqual(t, "hip_small_lm_head", decoded.Labels["adapter_runtime"])
	core.AssertEqual(t, hipKernelNameLoRA, decoded.Labels["lora_kernel_name"])
	core.AssertEqual(t, "experimental_qwen_gemma_small_decode", decoded.Labels["lora_model_status"])
	assertFloat32SlicesNear(t, want.Logits, decoded.Logits, 0.0001)

	var sawLoRA bool
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameLoRA {
			sawLoRA = true
		}
	}
	core.AssertTrue(t, sawLoRA)
}

func TestHIPRuntime_LoadedSmallDecodeConfig_Bad(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-small-decode.hsaco")

	loaded, _ := hipLoadedSmallDecodeFixture(t, "llama")
	defer loaded.Close()
	_, err := loaded.loadedSmallDecodeConfig()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "Qwen or Gemma")

	loaded, _ = hipLoadedSmallDecodeFixture(t, "qwen3")
	defer loaded.Close()
	tensor := loaded.tensors["model.layers.0.self_attn.q_proj.weight"]
	tensor.info.Type = 0
	tensor.info.TypeName = "f32"
	loaded.tensors["model.layers.0.self_attn.q_proj.weight"] = tensor
	_, err = loaded.loadedSmallDecodeConfig()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "query weight must be f16")

	typedLoaded, _ := hipLoadedSmallDecodeFixture(t, "qwen3")
	defer typedLoaded.Close()
	cache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 1, 1, []float32{1, 0}, []float32{1, 0}))
	_, err = typedLoaded.DecodeToken(context.Background(), hipDecodeRequest{TokenID: 2, KV: cache})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV widths must match hidden size")

	failingLoaded, failingDriver := hipLoadedSmallDecodeFixture(t, "qwen3")
	defer failingLoaded.Close()
	smoke := hipSmallDecodeFixture("qwen3")
	deviceCache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, deviceCache.AppendVectors(0, smoke.HiddenSize, smoke.HiddenSize, smoke.PriorKeys, smoke.PriorValues))
	deviceKV, table, err := hipMirrorTinyKV(failingDriver, deviceCache, map[string]string{})
	core.RequireNoError(t, err)
	defer deviceKV.Close()
	defer table.Close()
	failingDriver.copyErr = core.NewError("append copy failed")
	const smallDecodePrimitiveLaunches = 10
	failingDriver.copyHostErrAfterLaunches = len(failingDriver.launches) + smallDecodePrimitiveLaunches
	decoded, err := failingLoaded.DecodeToken(context.Background(), hipDecodeRequest{
		TokenID:         2,
		KV:              deviceCache,
		DeviceKV:        deviceKV,
		DescriptorTable: table,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "append copy failed")
	core.AssertNil(t, decoded.KV)
	core.AssertEqual(t, 2, deviceCache.TokenCount())
	core.AssertEqual(t, 2, deviceKV.TokenCount())
	if deviceKV.closed || table.closed {
		t.Fatalf("original device resources were closed after failed small decode device append")
	}
}

func hipSmallDecodeFixture(architecture string) hipSmallDecodeRequest {
	identity := []uint16{
		0x3c00, 0,
		0, 0x3c00,
	}
	lmHead := []uint16{
		0x3c00, 0,
		0, 0x3c00,
		0x3c00, 0x3c00,
	}
	return hipSmallDecodeRequest{
		Architecture: architecture,
		Input:        []float32{1, 1},
		RMSWeight:    []float32{1, 1},
		Epsilon:      0,
		QueryFP16:    append([]uint16(nil), identity...),
		KeyFP16:      append([]uint16(nil), identity...),
		ValueFP16:    append([]uint16(nil), identity...),
		OutputFP16:   append([]uint16(nil), identity...),
		LMHeadFP16:   lmHead,
		PriorKeys: []float32{
			1, 0,
			0, 1,
		},
		PriorValues: []float32{
			1, 0,
			0, 1,
		},
		Position:   2,
		RoPEBase:   10000,
		VocabSize:  3,
		HiddenSize: 2,
	}
}

func hipLoadedSmallDecodeFixture(t *testing.T, architecture string) (*hipLoadedModel, *fakeHIPDriver) {
	t.Helper()
	payload, tensors := hipSmallDecodeModelPayload(t, architecture)
	modelPath := core.PathJoin(t.TempDir(), "small-decode.bin")
	write := core.WriteFile(modelPath, payload, 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: architecture, VocabSize: 3, HiddenSize: 2, NumLayers: 1, QuantBits: 16},
		Tensors:   tensors,
	})
	core.RequireNoError(t, err)
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	return loaded, driver
}

func hipSmallDecodeModelPayload(t *testing.T, architecture string) ([]byte, []nativeTensorInfo) {
	t.Helper()
	smoke := hipSmallDecodeFixture(architecture)
	embeddingPayload, err := hipFloat32Payload(hipReferenceTinyLMFixture().EmbeddingTable)
	core.RequireNoError(t, err)
	rmsPayload, err := hipFloat32Payload(smoke.RMSWeight)
	core.RequireNoError(t, err)
	queryPayload, err := hipUint16Payload(smoke.QueryFP16)
	core.RequireNoError(t, err)
	keyPayload, err := hipUint16Payload(smoke.KeyFP16)
	core.RequireNoError(t, err)
	valuePayload, err := hipUint16Payload(smoke.ValueFP16)
	core.RequireNoError(t, err)
	outputPayload, err := hipUint16Payload(smoke.OutputFP16)
	core.RequireNoError(t, err)
	lmHeadPayload, err := hipUint16Payload(smoke.LMHeadFP16)
	core.RequireNoError(t, err)

	var payload []byte
	var tensors []nativeTensorInfo
	appendTensor := func(name string, tensorType uint32, dimensions []uint64, tensorPayload []byte) {
		tensors = append(tensors, nativeTensorInfo{
			Name:       name,
			Type:       tensorType,
			Dimensions: dimensions,
			Offset:     uint64(len(payload)),
			ByteSize:   uint64(len(tensorPayload)),
		})
		payload = append(payload, tensorPayload...)
	}
	appendTensor("tok_embeddings.weight", 0, []uint64{3, 2}, embeddingPayload)
	appendTensor("model.layers.0.input_layernorm.weight", 0, []uint64{2}, rmsPayload)
	appendTensor("model.layers.0.self_attn.q_proj.weight", 1, []uint64{2, 2}, queryPayload)
	appendTensor("model.layers.0.self_attn.k_proj.weight", 1, []uint64{2, 2}, keyPayload)
	appendTensor("model.layers.0.self_attn.v_proj.weight", 1, []uint64{2, 2}, valuePayload)
	appendTensor("model.layers.0.self_attn.o_proj.weight", 1, []uint64{2, 2}, outputPayload)
	appendTensor("output.weight", 1, []uint64{3, 2}, lmHeadPayload)
	return payload, tensors
}

func hipGemma4Q4Layer0FixtureConfig(t *testing.T, driver nativeHIPDriver) (hipGemma4Q4Layer0Config, func()) {
	t.Helper()
	return hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
}

func hipGemma4Q4GlobalPerLayerInputFixture(t *testing.T, driver nativeHIPDriver, layers []hipGemma4Q4Layer0Config) ([]hipGemma4Q4Layer0Config, func()) {
	t.Helper()
	if len(layers) == 0 {
		t.Fatalf("per-layer input fixture requires layers")
	}
	hidden := layers[0].HiddenSize
	vocab := layers[0].VocabSize
	groupSize := layers[0].GroupSize
	totalHidden := hidden * len(layers)
	if hidden <= 0 || vocab <= 0 || groupSize <= 0 || totalHidden%8 != 0 || totalHidden%groupSize != 0 {
		t.Fatalf("invalid per-layer input fixture geometry hidden=%d vocab=%d group=%d layers=%d", hidden, vocab, groupSize, len(layers))
	}
	var buffers []*hipDeviceByteBuffer
	uploadU16 := func(label string, count int) *hipDeviceByteBuffer {
		t.Helper()
		payload, err := hipUint16Payload(make([]uint16, count))
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, count)
		core.RequireNoError(t, err)
		buffers = append(buffers, buffer)
		return buffer
	}
	uploadU32 := func(label string, count int) *hipDeviceByteBuffer {
		t.Helper()
		payload, err := hipUint32Payload(make([]uint32, count))
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, count)
		core.RequireNoError(t, err)
		buffers = append(buffers, buffer)
		return buffer
	}
	norm := func(label string, count int) hipRMSNormDeviceWeightConfig {
		t.Helper()
		buffer := uploadU16(label, count)
		return hipRMSNormDeviceWeightConfig{
			WeightPointer:  buffer.Pointer(),
			WeightBytes:    buffer.SizeBytes(),
			Count:          count,
			WeightEncoding: hipRMSNormWeightEncodingBF16,
		}
	}

	embeddingWeights := uploadU32("embed_tokens_per_layer weights", vocab*(totalHidden/8))
	embeddingScales := uploadU16("embed_tokens_per_layer scales", vocab*(totalHidden/groupSize))
	embeddingBiases := uploadU16("embed_tokens_per_layer biases", vocab*(totalHidden/groupSize))
	modelProjectionWeights := uploadU16("per_layer_model_projection weights", totalHidden*hidden)
	projectionNorm := norm("per_layer_projection_norm", hidden)
	output := append([]hipGemma4Q4Layer0Config(nil), layers...)
	for index := range output {
		perLayer := output[index].PerLayerInput
		perLayer.InputSize = hidden
		perLayer.Embedding = hipDeviceEmbeddingLookupConfig{
			EmbeddingPointer: embeddingWeights.Pointer(),
			EmbeddingBytes:   embeddingWeights.SizeBytes(),
			TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
			VocabSize:        vocab,
			HiddenSize:       totalHidden,
			GroupSize:        groupSize,
			ScalePointer:     embeddingScales.Pointer(),
			BiasPointer:      embeddingBiases.Pointer(),
			ScaleBytes:       embeddingScales.SizeBytes(),
			BiasBytes:        embeddingBiases.SizeBytes(),
		}
		perLayer.ModelProjection = hipBF16DeviceWeightConfig{
			WeightPointer: modelProjectionWeights.Pointer(),
			WeightBytes:   modelProjectionWeights.SizeBytes(),
			Rows:          totalHidden,
			Cols:          hidden,
		}
		perLayer.ProjectionNorm = projectionNorm
		output[index].PerLayerInput = perLayer
	}
	cleanup := func() {
		for index := len(buffers) - 1; index >= 0; index-- {
			_ = buffers[index].Close()
		}
	}
	return output, cleanup
}

func hipGemma4Q4FixtureConfig(t *testing.T, driver nativeHIPDriver, layer, headDim, queryHeads, intermediate int) (hipGemma4Q4Layer0Config, func()) {
	t.Helper()
	const (
		hidden    = 8
		vocab     = 2
		groupSize = 8
	)
	var buffers []*hipDeviceByteBuffer
	uploadU16 := func(label string, count int) *hipDeviceByteBuffer {
		t.Helper()
		payload, err := hipUint16Payload(make([]uint16, count))
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, count)
		core.RequireNoError(t, err)
		buffers = append(buffers, buffer)
		return buffer
	}
	uploadU32 := func(label string, count int) *hipDeviceByteBuffer {
		t.Helper()
		payload, err := hipUint32Payload(make([]uint32, count))
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, count)
		core.RequireNoError(t, err)
		buffers = append(buffers, buffer)
		return buffer
	}
	norm := func(label string, count int) hipRMSNormDeviceWeightConfig {
		buffer := uploadU16(label, count)
		return hipRMSNormDeviceWeightConfig{
			WeightPointer:  buffer.Pointer(),
			WeightBytes:    buffer.SizeBytes(),
			Count:          count,
			WeightEncoding: hipRMSNormWeightEncodingBF16,
		}
	}
	q4Projection := func(label string, rows, cols int) hipMLXQ4DeviceWeightConfig {
		t.Helper()
		weights := uploadU32(label+" weights", rows*(cols/8))
		scales := uploadU16(label+" scales", rows*(cols/groupSize))
		biases := uploadU16(label+" biases", rows*(cols/groupSize))
		return hipMLXQ4DeviceWeightConfig{
			WeightPointer: weights.Pointer(),
			ScalePointer:  scales.Pointer(),
			BiasPointer:   biases.Pointer(),
			WeightBytes:   weights.SizeBytes(),
			ScaleBytes:    scales.SizeBytes(),
			BiasBytes:     biases.SizeBytes(),
			Rows:          rows,
			Cols:          cols,
			GroupSize:     groupSize,
		}
	}

	embeddingWeights := uploadU32("embed_tokens weights", vocab*(hidden/8))
	embeddingScales := uploadU16("embed_tokens scales", vocab*(hidden/groupSize))
	embeddingBiases := uploadU16("embed_tokens biases", vocab*(hidden/groupSize))
	cleanup := func() {
		for index := len(buffers) - 1; index >= 0; index-- {
			_ = buffers[index].Close()
		}
	}
	return hipGemma4Q4Layer0Config{
		Layer:     layer,
		LayerType: hipGemma4Q4LayerTypeFromHeadDim(headDim),
		Embedding: hipDeviceEmbeddingLookupConfig{
			EmbeddingPointer: embeddingWeights.Pointer(),
			EmbeddingBytes:   embeddingWeights.SizeBytes(),
			TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
			VocabSize:        vocab,
			HiddenSize:       hidden,
			GroupSize:        groupSize,
			ScalePointer:     embeddingScales.Pointer(),
			BiasPointer:      embeddingBiases.Pointer(),
			ScaleBytes:       embeddingScales.SizeBytes(),
			BiasBytes:        embeddingBiases.SizeBytes(),
		},
		HiddenSize:        hidden,
		VocabSize:         vocab,
		GroupSize:         groupSize,
		HeadDim:           headDim,
		QueryHeads:        queryHeads,
		IntermediateSize:  intermediate,
		RoPEBase:          10000,
		RoPERotaryDim:     headDim,
		SlidingWindow:     512,
		FinalLogitSoftcap: 30,
		LayerScalar:       1,
		PerLayerInput: hipGemma4Q4PerLayerInputConfig{
			InputSize:     hidden,
			InputGate:     q4Projection("per_layer_input_gate", hidden, hidden),
			Projection:    q4Projection("per_layer_projection", hidden, hidden),
			PostInputNorm: norm("post_per_layer_input_norm", hidden),
		},
		InputNorm:           norm("input_layernorm", hidden),
		QueryNorm:           norm("q_norm", headDim),
		KeyNorm:             norm("k_norm", headDim),
		PostAttentionNorm:   norm("post_attention_layernorm", hidden),
		PreFeedForwardNorm:  norm("pre_feedforward_layernorm", hidden),
		PostFeedForwardNorm: norm("post_feedforward_layernorm", hidden),
		FinalNorm:           norm("final_norm", hidden),
		QueryProjection:     q4Projection("q_proj", queryHeads*headDim, hidden),
		KeyProjection:       q4Projection("k_proj", headDim, hidden),
		ValueProjection:     q4Projection("v_proj", headDim, hidden),
		OutputProjection:    q4Projection("o_proj", hidden, queryHeads*headDim),
		GateProjection:      q4Projection("mlp.gate_proj", intermediate, hidden),
		UpProjection:        q4Projection("mlp.up_proj", intermediate, hidden),
		DownProjection:      q4Projection("mlp.down_proj", hidden, intermediate),
		LMHeadProjection:    q4Projection("embed_tokens_lm_head", vocab, hidden),
	}, cleanup
}
