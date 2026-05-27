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
	core.AssertContains(t, joined, hipKernelNameRMSNormRoPEHeads)
	core.AssertContains(t, joined, hipKernelNameAttentionHeads)
	core.AssertContains(t, joined, hipKernelNameRMSNormResidualAdd)
	core.AssertContains(t, joined, hipKernelNameMLXQ4GELUTanhMul)
	core.AssertContains(t, joined, hipKernelNameGreedy)
	core.AssertContains(t, got.Labels["decode_primitives"], "gelu_tanh_mlp")
	core.AssertEqual(t, "device_gelu_tanh_multiply", got.Labels["gemma4_mlp_activation"])
	attentionScales := 0
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameAttentionHeads {
			attentionScales++
			tokenCount := binary.LittleEndian.Uint32(launch.Args[52:])
			core.AssertEqual(t, uint64(0), binary.LittleEndian.Uint64(launch.Args[40:]))
			core.AssertEqual(t, uint32(0), binary.LittleEndian.Uint32(launch.Args[76:]))
			core.AssertEqual(t, tokenCount*4, launch.SharedMemBytes)
			assertFloat32Near(t, 1, math.Float32frombits(binary.LittleEndian.Uint32(launch.Args[84:])))
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
		if launch.Name == hipKernelNameRMSNormRoPEHeads {
			partialRoPELaunches++
			core.AssertEqual(t, uint32(cfg.HeadDim), binary.LittleEndian.Uint32(launch.Args[72:]))
			core.AssertEqual(t, uint32(cfg.HeadDim/2), binary.LittleEndian.Uint32(launch.Args[76:]))
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
	perLayerQ4Ops := 0
	perLayerTripleQ4Launches := 0
	for _, launch := range driver.launches[perLayerStart:] {
		switch launch.Name {
		case hipKernelNameMLXQ4Proj:
			perLayerQ4Ops++
		case hipKernelNameMLXQ4TripleProj:
			perLayerQ4Ops += 3
			perLayerTripleQ4Launches++
		}
	}
	core.AssertEqual(t, 6, perLayerQ4Ops)
	core.AssertEqual(t, 1, perLayerTripleQ4Launches)

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
			if launch.Name == hipKernelNameEmbedLookup || launch.Name == hipKernelNameEmbedLookupGreedyToken {
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
	core.AssertEqual(t, 6, countKVEncodeTokenLaunches(driver.launches[launchStart:]))
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
	core.AssertEqual(t, 4, countKVEncodeTokenLaunches(driver.launches[launchStart:]))
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 early-stopped public generate launched no descriptor-backed attention kernels")
	}
	if len(driver.frees) == freeStart {
		t.Fatalf("Gemma4 q4 early-stopped public generate freed no device KV allocations")
	}
}

func TestHIPGemma4Q4GenerateTokenSeq_UsesBatchedPrefill_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	embeddingWeightsPayload, err := hipUint32Payload(make([]uint32, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize)))
	core.RequireNoError(t, err)
	embeddingWeights, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "generate batched prefill embedding weights", embeddingWeightsPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingWeights.Close()
	embeddingScalesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingScales, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "generate batched prefill embedding scales", embeddingScalesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingScales.Close()
	embeddingBiasesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingBiases, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "generate batched prefill embedding biases", embeddingBiasesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingBiases.Close()
	layer0.Embedding = hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: embeddingWeights.Pointer(),
		EmbeddingBytes:   embeddingWeights.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		VocabSize:        layer0.VocabSize,
		HiddenSize:       layer0.HiddenSize,
		GroupSize:        layer0.GroupSize,
		ScalePointer:     embeddingScales.Pointer(),
		BiasPointer:      embeddingBiases.Pointer(),
		ScaleBytes:       embeddingScales.SizeBytes(),
		BiasBytes:        embeddingBiases.SizeBytes(),
	}
	layers, cleanupPerLayer := hipGemma4Q4GlobalPerLayerInputFixture(t, driver, []hipGemma4Q4Layer0Config{layer0, layer1})
	defer cleanupPerLayer()
	cfg := hipGemma4Q4ForwardConfig{Layers: layers}
	core.AssertEqual(t, true, hipGemma4Q4CanUseBatchedGeneratePrefill(cfg))
	t.Setenv(hipGemma4Q4PrefillUBatchEnv, "2")

	start := len(driver.launches)
	stream, streamErr := hipGemma4Q4GenerateTokenSeq(context.Background(), &hipLoadedModel{driver: driver}, cfg, []int32{0, 1, 0}, inference.GenerateConfig{MaxTokens: 1})
	var generated []inference.Token
	for token := range stream {
		generated = append(generated, token)
	}

	core.RequireNoError(t, streamErr())
	core.AssertEqual(t, 1, len(generated))
	launches := driver.launches[start:]
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameEmbedLookupGreedyToken))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4Proj))
	batchProjectionLaunches := countLaunchName(launches, hipKernelNameMLXQ4ProjBatch)
	batchAttentionLaunches := countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal)
	finalGreedyLaunches := countLaunchName(launches, hipKernelNameMLXQ4ProjGreedy)
	if batchProjectionLaunches == 0 || batchAttentionLaunches == 0 || finalGreedyLaunches == 0 {
		t.Fatalf("Gemma4 q4 generate batched prefill launches projection_batch=%d attention_batch=%d final_greedy=%d, want all nonzero", batchProjectionLaunches, batchAttentionLaunches, finalGreedyLaunches)
	}
}

func TestHIPGemma4Q4EffectiveSlidingWindow_Good(t *testing.T) {
	core.AssertEqual(t, 512, hipGemma4Q4EffectiveSlidingWindow(256, 0))
	core.AssertEqual(t, 128, hipGemma4Q4EffectiveSlidingWindow(256, 128))
	core.AssertEqual(t, 512, hipGemma4Q4EffectiveSlidingWindow(256, 2048))
	core.AssertEqual(t, 0, hipGemma4Q4EffectiveSlidingWindow(512, 128))
}

func TestHIPGemma4Q4ChunkedAttentionEnabled_Good(t *testing.T) {
	t.Setenv("GO_ROCM_GEMMA4_Q4_CHUNKED_ATTENTION", "")
	core.AssertEqual(t, true, hipGemma4Q4ChunkedAttentionEnabled(1))
	core.AssertEqual(t, true, hipGemma4Q4ChunkedAttentionEnabled(4000))

	t.Setenv("GO_ROCM_GEMMA4_Q4_CHUNKED_ATTENTION", "auto")
	core.AssertEqual(t, false, hipGemma4Q4ChunkedAttentionEnabled(2000))
	core.AssertEqual(t, true, hipGemma4Q4ChunkedAttentionEnabled(4000))

	t.Setenv("GO_ROCM_GEMMA4_Q4_CHUNKED_ATTENTION", "0")
	core.AssertEqual(t, false, hipGemma4Q4ChunkedAttentionEnabled(48000))

	t.Setenv("GO_ROCM_GEMMA4_Q4_CHUNKED_ATTENTION", "1")
	core.AssertEqual(t, true, hipGemma4Q4ChunkedAttentionEnabled(1))
}

func TestHIPGemma4Q4DeviceKVBlockSize_Good(t *testing.T) {
	t.Setenv("GO_ROCM_GEMMA4_Q4_DEVICE_KV_BLOCK_SIZE", "")
	core.AssertEqual(t, rocmGemma4Q4DeviceKVBlockSize, hipGemma4Q4DeviceKVBlockSize())

	t.Setenv("GO_ROCM_GEMMA4_Q4_DEVICE_KV_BLOCK_SIZE", "16")
	core.AssertEqual(t, 16, hipGemma4Q4DeviceKVBlockSize())

	t.Setenv("GO_ROCM_GEMMA4_Q4_DEVICE_KV_BLOCK_SIZE", "bad")
	core.AssertEqual(t, rocmGemma4Q4DeviceKVBlockSize, hipGemma4Q4DeviceKVBlockSize())
}

func TestHIPGemma4Q4PrefillPlan_Good(t *testing.T) {
	t.Setenv(hipGemma4Q4PrefillUBatchEnv, "")
	ubatchTokens, err := hipGemma4Q4PrefillUBatchTokens()
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipGemma4Q4PrefillDefaultUBatchTokens, ubatchTokens)

	t.Setenv(hipGemma4Q4PrefillUBatchEnv, "2")
	ubatchTokens, err = hipGemma4Q4PrefillUBatchTokens()
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, ubatchTokens)

	plan, err := hipGemma4Q4PlanPromptPrefill([]int32{2, 10979, 2, 10979, 2}, 7, ubatchTokens)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 5, plan.PromptTokens)
	core.AssertEqual(t, 7, plan.StartPos)
	core.AssertEqual(t, 2, plan.UBatchTokens)
	core.AssertEqual(t, 1, plan.OutputTokens)
	core.AssertEqual(t, 12, plan.NextPosition())
	core.AssertEqual(t, 3, len(plan.Batches))
	core.AssertEqual(t, []int32{2, 10979}, plan.Batches[0].Tokens)
	core.AssertEqual(t, 0, len(plan.Batches[0].OutputTokens))
	core.AssertEqual(t, false, plan.Batches[0].OutputToken(0))
	core.AssertEqual(t, 0, plan.Batches[0].Start)
	core.AssertEqual(t, 2, plan.Batches[0].End)
	core.AssertEqual(t, 7, plan.Batches[0].Position)
	core.AssertEqual(t, []int32{2}, plan.Batches[2].Tokens)
	core.AssertEqual(t, []bool{true}, plan.Batches[2].OutputTokens)
	core.AssertEqual(t, true, plan.Batches[2].OutputToken(0))
	core.AssertEqual(t, 4, plan.Batches[2].Start)
	core.AssertEqual(t, 5, plan.Batches[2].End)
	core.AssertEqual(t, 11, plan.Batches[2].Position)
}

func BenchmarkHIPGemma4Q4PlanPromptPrefill_29K(b *testing.B) {
	tokens := make([]int32, 29000)
	for index := range tokens {
		tokens[index] = int32(index%32000 + 1)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		plan, err := hipGemma4Q4PlanPromptPrefill(tokens, 0, hipGemma4Q4PrefillDefaultUBatchTokens)
		if err != nil {
			b.Fatalf("hipGemma4Q4PlanPromptPrefill: %v", err)
		}
		if plan.PromptTokens != len(tokens) || len(plan.Batches) != 1813 {
			b.Fatalf("plan = tokens %d batches %d, want 29000/1813", plan.PromptTokens, len(plan.Batches))
		}
	}
}

func BenchmarkHIPGemma4Q4DeviceLayerCaches_Reused(b *testing.B) {
	state := &hipGemma4Q4DeviceDecodeState{layers: make([]hipGemma4Q4DeviceLayerKVState, 35)}
	for index := range state.layers {
		state.layers[index].cache = &rocmDeviceKVCache{}
	}
	scratch := make([]*rocmDeviceKVCache, 0, len(state.layers))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		scratch = hipGemma4Q4DeviceLayerCaches(state, scratch, len(state.layers))
		if len(scratch) != len(state.layers) || scratch[0] == nil {
			b.Fatalf("layer cache scratch len=%d first=%v", len(scratch), scratch[0])
		}
	}
}

func TestHIPGemma4Q4PrefillPlan_Bad(t *testing.T) {
	t.Setenv(hipGemma4Q4PrefillUBatchEnv, "nope")
	if _, err := hipGemma4Q4PrefillUBatchTokens(); err == nil {
		t.Fatalf("hipGemma4Q4PrefillUBatchTokens succeeded, want invalid env error")
	}
	if _, err := hipGemma4Q4PlanPromptPrefill(nil, 0, 512); err == nil {
		t.Fatalf("hipGemma4Q4PlanPromptPrefill succeeded with empty prompt")
	}
	if _, err := hipGemma4Q4PlanPromptPrefill([]int32{1}, -1, 512); err == nil {
		t.Fatalf("hipGemma4Q4PlanPromptPrefill succeeded with negative start position")
	}
	if _, err := hipGemma4Q4PlanPromptPrefill([]int32{1}, 0, 0); err == nil {
		t.Fatalf("hipGemma4Q4PlanPromptPrefill succeeded with zero ubatch size")
	}
}

func TestHIPGemma4Q4PrefillEmbeddingBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	tokens := []int32{1, 0, 1}
	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillEmbeddingBatch(context.Background(), driver, cfg, tokens)
	core.RequireNoError(t, err)
	defer output.Close()

	wantCount := len(tokens) * cfg.HiddenSize
	core.AssertEqual(t, wantCount, output.Count())
	core.AssertEqual(t, uint64(wantCount*4), output.SizeBytes())

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameEmbedLookup))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameVectorScale))
	for _, launch := range launches {
		switch launch.Name {
		case hipKernelNameEmbedLookup:
			core.AssertEqual(t, uint32(len(tokens)), binary.LittleEndian.Uint32(launch.Args[32:]))
			core.AssertEqual(t, uint32(cfg.HiddenSize), binary.LittleEndian.Uint32(launch.Args[40:]))
			core.AssertEqual(t, uint64(wantCount*4), binary.LittleEndian.Uint64(launch.Args[56:]))
		case hipKernelNameVectorScale:
			core.AssertEqual(t, uint32(wantCount), binary.LittleEndian.Uint32(launch.Args[24:]))
			core.AssertEqual(t, uint32(wantCount*4), binary.LittleEndian.Uint32(launch.Args[28:]))
			core.AssertEqual(t, uint32(wantCount*4), binary.LittleEndian.Uint32(launch.Args[32:]))
			assertFloat32Near(t, float32(math.Sqrt(float64(cfg.HiddenSize))), math.Float32frombits(binary.LittleEndian.Uint32(launch.Args[36:])))
		}
	}
}

func TestHIPGemma4Q4PrefillEmbeddingBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	if _, err := hipRunGemma4Q4PrefillEmbeddingBatch(context.Background(), driver, cfg, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillEmbeddingBatch succeeded with empty tokens")
	}
	core.AssertEqual(t, 0, len(driver.launches))

	unavailable := &fakeHIPDriver{available: false}
	if _, err := hipRunGemma4Q4PrefillEmbeddingBatch(context.Background(), unavailable, cfg, []int32{1}); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillEmbeddingBatch succeeded with unavailable driver")
	}
	core.AssertEqual(t, 0, len(unavailable.launches))
}

func TestHIPGemma4Q4PrefillInputNormBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	tokenCount := 3
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill input norm fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillInputNormBatch(context.Background(), driver, cfg, input, tokenCount)
	core.RequireNoError(t, err)
	defer output.Close()

	wantCount := tokenCount * cfg.HiddenSize
	core.AssertEqual(t, wantCount, output.Count())
	core.AssertEqual(t, uint64(wantCount*4), output.SizeBytes())

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, len(launches))
	launch := launches[0]
	core.AssertEqual(t, hipKernelNameRMSNormHeads, launch.Name)
	core.AssertEqual(t, uint32(tokenCount), launch.GridX)
	core.AssertEqual(t, uint32(cfg.HiddenSize), binary.LittleEndian.Uint32(launch.Args[32:]))
	core.AssertEqual(t, uint32(tokenCount), binary.LittleEndian.Uint32(launch.Args[36:]))
	core.AssertEqual(t, uint32(wantCount*4), binary.LittleEndian.Uint32(launch.Args[40:]))
	core.AssertEqual(t, uint32(wantCount*4), binary.LittleEndian.Uint32(launch.Args[48:]))
	assertFloat32Near(t, cfg.InputNorm.Epsilon, math.Float32frombits(binary.LittleEndian.Uint32(launch.Args[52:])))
	core.AssertEqual(t, hipRMSNormWeightEncodingBF16, binary.LittleEndian.Uint32(launch.Args[56:]))
}

func TestHIPGemma4Q4PrefillInputNormBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	inputValues := make([]float32, cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill input norm bad fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillInputNormBatch(context.Background(), driver, cfg, input, 0); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillInputNormBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillInputNormBatch(context.Background(), driver, cfg, input, 2); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillInputNormBatch succeeded with mismatched token count")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillQKVProjectionBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	tokenCount := 2
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill QKV fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	start := len(driver.launches)
	qkv, err := hipRunGemma4Q4PrefillQKVProjectionBatch(context.Background(), driver, cfg, input, tokenCount)
	core.RequireNoError(t, err)
	defer qkv.Close()

	core.AssertEqual(t, tokenCount*cfg.QueryProjection.Rows, qkv.Query.Count())
	core.AssertEqual(t, tokenCount*cfg.KeyProjection.Rows, qkv.Key.Count())
	core.AssertEqual(t, tokenCount*cfg.ValueProjection.Rows, qkv.Value.Count())
	launches := driver.launches[start:]
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	wantRows := []int{cfg.QueryProjection.Rows, cfg.KeyProjection.Rows, cfg.ValueProjection.Rows}
	for index, launch := range launches {
		core.AssertEqual(t, hipKernelNameMLXQ4ProjBatch, launch.Name)
		core.AssertEqual(t, uint32((tokenCount+hipMLXQ4ProjectionBatchTokensPerBlock-1)/hipMLXQ4ProjectionBatchTokensPerBlock), launch.GridY)
		core.AssertEqual(t, uint32(wantRows[index]), binary.LittleEndian.Uint32(launch.Args[48:]))
		core.AssertEqual(t, uint32(cfg.HiddenSize), binary.LittleEndian.Uint32(launch.Args[52:]))
		core.AssertEqual(t, uint32(tokenCount), binary.LittleEndian.Uint32(launch.Args[56:]))
	}
}

func TestHIPGemma4Q4PrefillQKVProjectionBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	inputValues := make([]float32, cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill QKV bad fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillQKVProjectionBatch(context.Background(), driver, cfg, input, 0); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillQKVProjectionBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillQKVProjectionBatch(context.Background(), driver, cfg, input, 2); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillQKVProjectionBatch succeeded with mismatched token count")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillQKNormRoPEBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.RoPERotaryDim = 2

	tokenCount := 2
	queryValues := make([]float32, tokenCount*cfg.QueryHeads*cfg.HeadDim)
	for index := range queryValues {
		queryValues[index] = float32(index%cfg.HeadDim + 1)
	}
	keyValues := make([]float32, tokenCount*cfg.HeadDim)
	for index := range keyValues {
		keyValues[index] = float32(index%cfg.HeadDim + 1)
	}
	queryPayload, err := hipFloat32Payload(queryValues)
	core.RequireNoError(t, err)
	query, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill Q/K RoPE query fixture", queryPayload, len(queryValues))
	core.RequireNoError(t, err)
	defer query.Close()
	keyPayload, err := hipFloat32Payload(keyValues)
	core.RequireNoError(t, err)
	key, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill Q/K RoPE key fixture", keyPayload, len(keyValues))
	core.RequireNoError(t, err)
	defer key.Close()
	qkv := &hipGemma4Q4PrefillQKVBatch{Query: query, Key: key}

	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillQKNormRoPEBatch(context.Background(), driver, cfg, qkv, tokenCount, 5, 1e-6)
	core.RequireNoError(t, err)
	defer output.Close()

	core.AssertEqual(t, tokenCount*cfg.QueryHeads*cfg.HeadDim, output.Query.Count())
	core.AssertEqual(t, tokenCount*cfg.HeadDim, output.Key.Count())
	launches := driver.launches[start:]
	core.AssertEqual(t, 2, len(launches))
	core.AssertEqual(t, hipKernelNameRMSNormRoPEHeadsBatch, launches[0].Name)
	core.AssertEqual(t, hipKernelNameRMSNormRoPEHeadsBatch, launches[1].Name)
	core.AssertEqual(t, uint32(cfg.QueryHeads), launches[0].GridX)
	core.AssertEqual(t, uint32(tokenCount), launches[0].GridY)
	core.AssertEqual(t, uint32(1), launches[1].GridX)
	core.AssertEqual(t, uint32(tokenCount), launches[1].GridY)
	for index, launch := range launches {
		wantHeads := cfg.QueryHeads
		if index == 1 {
			wantHeads = 1
		}
		core.AssertEqual(t, uint32(cfg.HeadDim), binary.LittleEndian.Uint32(launch.Args[32:]))
		core.AssertEqual(t, uint32(wantHeads), binary.LittleEndian.Uint32(launch.Args[36:]))
		core.AssertEqual(t, uint32(tokenCount), binary.LittleEndian.Uint32(launch.Args[40:]))
		assertFloat32Near(t, 1e-6, math.Float32frombits(binary.LittleEndian.Uint32(launch.Args[56:])))
		core.AssertEqual(t, hipRMSNormWeightEncodingBF16, binary.LittleEndian.Uint32(launch.Args[60:]))
		core.AssertEqual(t, uint32(5), binary.LittleEndian.Uint32(launch.Args[68:]))
		core.AssertEqual(t, uint32(cfg.HeadDim), binary.LittleEndian.Uint32(launch.Args[76:]))
		core.AssertEqual(t, uint32(cfg.RoPERotaryDim), binary.LittleEndian.Uint32(launch.Args[80:]))
	}
}

func TestHIPGemma4Q4PrefillQKNormRoPEBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	tokenCount := 1
	queryPayload, err := hipFloat32Payload(make([]float32, tokenCount*cfg.QueryHeads*cfg.HeadDim))
	core.RequireNoError(t, err)
	query, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill Q/K RoPE bad query fixture", queryPayload, tokenCount*cfg.QueryHeads*cfg.HeadDim)
	core.RequireNoError(t, err)
	defer query.Close()
	keyPayload, err := hipFloat32Payload(make([]float32, tokenCount*cfg.HeadDim))
	core.RequireNoError(t, err)
	key, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill Q/K RoPE bad key fixture", keyPayload, tokenCount*cfg.HeadDim)
	core.RequireNoError(t, err)
	defer key.Close()
	qkv := &hipGemma4Q4PrefillQKVBatch{Query: query, Key: key}
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillQKNormRoPEBatch(context.Background(), driver, cfg, qkv, 0, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillQKNormRoPEBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillQKNormRoPEBatch(context.Background(), driver, cfg, qkv, 2, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillQKNormRoPEBatch succeeded with mismatched token count")
	}
	cfg.RoPERotaryDim = 3
	if _, err := hipRunGemma4Q4PrefillQKNormRoPEBatch(context.Background(), driver, cfg, qkv, tokenCount, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillQKNormRoPEBatch succeeded with odd rotary dimension")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillValueNormBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	tokenCount := 2
	valueValues := []float32{
		1, 0, 3, 4,
		0, 2, 5, 12,
	}
	valuePayload, err := hipFloat32Payload(valueValues)
	core.RequireNoError(t, err)
	value, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill value norm fixture", valuePayload, len(valueValues))
	core.RequireNoError(t, err)
	defer value.Close()
	qkv := &hipGemma4Q4PrefillQKVBatch{Value: value}

	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillValueNormBatch(context.Background(), driver, cfg, qkv, tokenCount, 1e-6)
	core.RequireNoError(t, err)
	defer output.Close()
	values, err := hipReadFloat32DeviceOutput(output, hipGemma4Q4Layer0Operation, "prefill value norm output", len(valueValues))
	core.RequireNoError(t, err)

	var want []float32
	unitWeight := []float32{1, 1, 1, 1}
	for token := 0; token < tokenCount; token++ {
		offset := token * cfg.HeadDim
		normalized, err := hipReferenceRMSNorm(valueValues[offset:offset+cfg.HeadDim], unitWeight, 1e-6)
		core.RequireNoError(t, err)
		want = append(want, normalized...)
	}
	assertFloat32SlicesNear(t, want, values, 0.0001)

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, len(launches))
	core.AssertEqual(t, hipKernelNameRMSNormHeads, launches[0].Name)
	core.AssertEqual(t, uint32(tokenCount), launches[0].GridX)
	core.AssertEqual(t, uint32(cfg.HeadDim), binary.LittleEndian.Uint32(launches[0].Args[32:]))
	core.AssertEqual(t, uint32(tokenCount), binary.LittleEndian.Uint32(launches[0].Args[36:]))
	core.AssertEqual(t, hipRMSNormWeightEncodingNone, binary.LittleEndian.Uint32(launches[0].Args[56:]))
}

func TestHIPGemma4Q4PrefillValueNormBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	valuePayload, err := hipFloat32Payload(make([]float32, cfg.HeadDim))
	core.RequireNoError(t, err)
	value, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill value norm bad fixture", valuePayload, cfg.HeadDim)
	core.RequireNoError(t, err)
	defer value.Close()
	qkv := &hipGemma4Q4PrefillQKVBatch{Value: value}
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillValueNormBatch(context.Background(), driver, cfg, qkv, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillValueNormBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillValueNormBatch(context.Background(), driver, cfg, qkv, 2, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillValueNormBatch succeeded with mismatched token count")
	}
	if _, err := hipRunGemma4Q4PrefillValueNormBatch(context.Background(), driver, cfg, &hipGemma4Q4PrefillQKVBatch{}, 1, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillValueNormBatch succeeded with missing value buffer")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillDeviceKVBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	tokenCount := 3
	keyRows := []float32{
		1, 0, 0, 1,
		0, 1, 1, 0,
		-1, 1, 0.5, -0.5,
	}
	valueRows := []float32{
		2, 0, 0, 2,
		0, 2, 2, 0,
		3, -3, 1, -1,
	}
	keyPayload, err := hipFloat32Payload(keyRows)
	core.RequireNoError(t, err)
	key, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill device KV key fixture", keyPayload, len(keyRows))
	core.RequireNoError(t, err)
	defer key.Close()
	valuePayload, err := hipFloat32Payload(valueRows)
	core.RequireNoError(t, err)
	value, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill device KV value fixture", valuePayload, len(valueRows))
	core.RequireNoError(t, err)
	defer value.Close()
	qk := &hipGemma4Q4PrefillRoPEQKBatch{Key: key}

	start := len(driver.launches)
	deviceKV, err := hipRunGemma4Q4PrefillDeviceKVBatch(context.Background(), driver, cfg, qk, value, tokenCount, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer deviceKV.Close()

	wantPages := gemma4Q4DeviceKVPagesForTokens(tokenCount)
	core.AssertEqual(t, wantPages, countLaunchName(driver.launches[start:], hipKernelNameKVEncodeToken))
	core.AssertEqual(t, tokenCount, deviceKV.Cache.TokenCount())
	core.AssertEqual(t, wantPages, deviceKV.Cache.PageCount())
	core.AssertEqual(t, min(tokenCount, hipGemma4Q4DeviceKVBlockSize()), deviceKV.Cache.pages[0].tokenCount)
	core.AssertEqual(t, cfg.HeadDim, deviceKV.Launch.KeyWidth)
	core.AssertEqual(t, cfg.HeadDim, deviceKV.Launch.ValueWidth)
	core.AssertEqual(t, tokenCount, deviceKV.Launch.TokenCount)
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, deviceKV.Launch.Mode)

	descriptorPayload := make([]byte, deviceKV.DescriptorTable.SizeBytes())
	core.RequireNoError(t, driver.CopyDeviceToHost(deviceKV.DescriptorTable.Pointer(), descriptorPayload))
	core.AssertEqual(t, uint64(tokenCount), binary.LittleEndian.Uint64(descriptorPayload[24:]))
	pageOffset := rocmDeviceKVDescriptorHeaderBytes
	core.AssertEqual(t, uint64(0), binary.LittleEndian.Uint64(descriptorPayload[pageOffset:]))
	core.AssertEqual(t, uint64(min(tokenCount, hipGemma4Q4DeviceKVBlockSize())), binary.LittleEndian.Uint64(descriptorPayload[pageOffset+8:]))
	core.AssertEqual(t, uint32(cfg.HeadDim), binary.LittleEndian.Uint32(descriptorPayload[pageOffset+16:]))
	core.AssertEqual(t, uint32(cfg.HeadDim), binary.LittleEndian.Uint32(descriptorPayload[pageOffset+20:]))
}

func TestHIPGemma4Q4PrefillDeviceKVBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	keyPayload, err := hipFloat32Payload(make([]float32, cfg.HeadDim))
	core.RequireNoError(t, err)
	key, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill device KV bad key fixture", keyPayload, cfg.HeadDim)
	core.RequireNoError(t, err)
	defer key.Close()
	valuePayload, err := hipFloat32Payload(make([]float32, cfg.HeadDim))
	core.RequireNoError(t, err)
	value, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill device KV bad value fixture", valuePayload, cfg.HeadDim)
	core.RequireNoError(t, err)
	defer value.Close()
	qk := &hipGemma4Q4PrefillRoPEQKBatch{Key: key}
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillDeviceKVBatch(context.Background(), driver, cfg, qk, value, 0, rocmKVCacheModeKQ8VQ4); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillDeviceKVBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillDeviceKVBatch(context.Background(), driver, cfg, qk, value, 2, rocmKVCacheModeKQ8VQ4); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillDeviceKVBatch succeeded with mismatched key/value rows")
	}
	if _, err := hipRunGemma4Q4PrefillDeviceKVBatch(context.Background(), driver, cfg, &hipGemma4Q4PrefillRoPEQKBatch{}, value, 1, rocmKVCacheModeKQ8VQ4); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillDeviceKVBatch succeeded with missing key buffer")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillLayerKVBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.RoPERotaryDim = 2

	tokenCount := 3
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer KV input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	start := len(driver.launches)
	layer, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, tokenCount, 7, 1e-6, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer layer.Close()

	core.AssertEqual(t, tokenCount*cfg.HiddenSize, layer.InputNorm.Count())
	core.AssertEqual(t, tokenCount*cfg.QueryProjection.Rows, layer.QKV.Query.Count())
	core.AssertEqual(t, tokenCount*cfg.HeadDim, layer.QK.Key.Count())
	core.AssertEqual(t, tokenCount*cfg.HeadDim, layer.Value.Count())
	core.AssertEqual(t, tokenCount, layer.DeviceKV.Cache.TokenCount())
	core.AssertEqual(t, gemma4Q4DeviceKVPagesForTokens(tokenCount), layer.DeviceKV.Cache.PageCount())

	launches := driver.launches[start:]
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameRMSNormHeads))
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameRMSNormRoPEHeadsBatch))
	core.AssertEqual(t, gemma4Q4DeviceKVPagesForTokens(tokenCount), countLaunchName(launches, hipKernelNameKVEncodeToken))
	core.AssertEqual(t, tokenCount, layer.DeviceKV.Launch.TokenCount)
	core.AssertEqual(t, cfg.HeadDim, layer.DeviceKV.Launch.KeyWidth)
	core.AssertEqual(t, cfg.HeadDim, layer.DeviceKV.Launch.ValueWidth)
}

func TestHIPGemma4Q4PrefillAttentionBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.RoPERotaryDim = 2

	tokenCount := 3
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill attention input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	layer, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, tokenCount, 0, 1e-6, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer layer.Close()
	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillAttentionBatch(context.Background(), driver, cfg, layer, tokenCount, 0)
	core.RequireNoError(t, err)
	defer output.Close()

	core.AssertEqual(t, tokenCount*cfg.QueryHeads*cfg.HeadDim, output.Count())
	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal))
	core.AssertEqual(t, uint32(cfg.QueryHeads), launches[0].GridX)
	core.AssertEqual(t, uint32(tokenCount), launches[0].GridY)
	core.AssertEqual(t, hipAttentionHeadsBatchCausalLaunchArgsBytes, len(launches[0].Args))
	core.AssertEqual(t, uint32(tokenCount), binary.LittleEndian.Uint32(launches[0].Args[52:]))
	core.AssertEqual(t, uint32(cfg.QueryHeads), binary.LittleEndian.Uint32(launches[0].Args[56:]))
	core.AssertEqual(t, uint32(tokenCount), binary.LittleEndian.Uint32(launches[0].Args[60:]))
	core.AssertEqual(t, uint32(0), binary.LittleEndian.Uint32(launches[0].Args[64:]))
	core.AssertEqual(t, hipAttentionKVSourceContiguous, binary.LittleEndian.Uint32(launches[0].Args[88:]))

	sharedLayer := &hipGemma4Q4PrefillLayerKVBatch{
		QK:        &hipGemma4Q4PrefillRoPEQKBatch{Query: layer.QK.Query},
		DeviceKV:  layer.DeviceKV,
		SharedKey: layer.QK.Key,
		SharedVal: layer.Value,
	}
	start = len(driver.launches)
	sharedOutput, err := hipRunGemma4Q4PrefillAttentionBatch(context.Background(), driver, cfg, sharedLayer, tokenCount, 0)
	core.RequireNoError(t, err)
	defer sharedOutput.Close()
	sharedLaunches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(sharedLaunches, hipKernelNameAttentionHeadsBatchCausal))
	core.AssertEqual(t, hipAttentionKVSourceContiguous, binary.LittleEndian.Uint32(sharedLaunches[0].Args[88:]))
}

func TestHIPGemma4Q4PrefillAttentionBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillAttentionBatch(context.Background(), driver, cfg, nil, 0, 0); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillAttentionBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillAttentionBatch(context.Background(), driver, cfg, nil, 1, -1); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillAttentionBatch succeeded with negative query start")
	}
	if _, err := hipRunGemma4Q4PrefillAttentionBatch(context.Background(), driver, cfg, nil, 1, 0); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillAttentionBatch succeeded with missing layer")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillLayerBodyBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.RoPERotaryDim = 2

	tokenCount := 3
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer body input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	layer, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, tokenCount, 0, 1e-6, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer layer.Close()
	start := len(driver.launches)
	body, err := hipRunGemma4Q4PrefillLayerBodyBatch(context.Background(), driver, cfg, input, layer, tokenCount, 0, 1e-6)
	core.RequireNoError(t, err)
	defer body.Close()

	core.AssertEqual(t, tokenCount*cfg.QueryHeads*cfg.HeadDim, body.AttentionOutput.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.AttentionProjection.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.AttentionResidual.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.PreFeedForward.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.MLPOutput.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.FinalHidden.Count())
	finalHidden, err := hipReadFloat32DeviceOutput(body.FinalHidden, hipGemma4Q4Layer0Operation, "prefill layer body final hidden", len(inputValues))
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, inputValues, finalHidden, 0.0001)

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameRMSNormHeads))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameVectorAdd))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMulBatch))
}

func TestHIPGemma4Q4PrefillLayerBodyBatchWithPerLayerInput_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.RoPERotaryDim = 2

	tokenCount := 3
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer body per-layer input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	perLayerValues := make([]float32, tokenCount*cfg.PerLayerInput.InputSize)
	for index := range perLayerValues {
		perLayerValues[index] = float32(index%cfg.PerLayerInput.InputSize + 1)
	}
	perLayerPayload, err := hipFloat32Payload(perLayerValues)
	core.RequireNoError(t, err)
	perLayerInput, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer body per-layer input multiplier", perLayerPayload, len(perLayerValues))
	core.RequireNoError(t, err)
	defer perLayerInput.Close()

	layer, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, tokenCount, 0, 1e-6, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer layer.Close()
	start := len(driver.launches)
	body, err := hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInput(context.Background(), driver, cfg, input, layer, perLayerInput, tokenCount, 0, 1e-6)
	core.RequireNoError(t, err)
	defer body.Close()

	core.AssertEqual(t, tokenCount*cfg.QueryHeads*cfg.HeadDim, body.AttentionOutput.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.PostFeedForward.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.PerLayerProjection.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.FinalHidden.Count())
	finalHidden, err := hipReadFloat32DeviceOutput(body.FinalHidden, hipGemma4Q4Layer0Operation, "prefill layer body per-layer final hidden", len(inputValues))
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, inputValues, finalHidden, 0.0001)

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal))
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	core.AssertEqual(t, 4, countLaunchName(launches, hipKernelNameRMSNormHeads))
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameVectorAdd))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMulBatch))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhProjBatch))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameVectorScale))
}

func TestHIPGemma4Q4PrefillFinalGreedyForRow_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	tokenCount := 3
	hiddenValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range hiddenValues {
		hiddenValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(hiddenValues)
	core.RequireNoError(t, err)
	hidden, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill final greedy hidden fixture", payload, len(hiddenValues))
	core.RequireNoError(t, err)
	defer hidden.Close()
	start := len(driver.launches)

	greedy, err := hipRunGemma4Q4PrefillFinalGreedyForRow(context.Background(), driver, cfg, hidden, tokenCount, tokenCount-1, 1e-6, nil)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, greedy.TokenID)

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameRMSNorm))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4ProjGreedy))
}

func TestHIPGemma4Q4PrefillForwardBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	embeddingWeightsPayload, err := hipUint32Payload(make([]uint32, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize)))
	core.RequireNoError(t, err)
	embeddingWeights, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill forward nonzero embedding weights", embeddingWeightsPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingWeights.Close()
	embeddingScalesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingScales, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill forward nonzero embedding scales", embeddingScalesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingScales.Close()
	embeddingBiasesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingBiases, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill forward nonzero embedding biases", embeddingBiasesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingBiases.Close()
	layer0.Embedding = hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: embeddingWeights.Pointer(),
		EmbeddingBytes:   embeddingWeights.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		VocabSize:        layer0.VocabSize,
		HiddenSize:       layer0.HiddenSize,
		GroupSize:        layer0.GroupSize,
		ScalePointer:     embeddingScales.Pointer(),
		BiasPointer:      embeddingBiases.Pointer(),
		ScaleBytes:       embeddingScales.SizeBytes(),
		BiasBytes:        embeddingBiases.SizeBytes(),
	}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}
	tokens := []int32{0, 1, 0}
	perLayerInputs := make([]*hipDeviceByteBuffer, len(cfg.Layers))
	for layerIndex, layer := range cfg.Layers {
		values := make([]float32, len(tokens)*layer.PerLayerInput.InputSize)
		for index := range values {
			values[index] = float32(layerIndex + 1)
		}
		payload, err := hipFloat32Payload(values)
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill forward per-layer input fixture", payload, len(values))
		core.RequireNoError(t, err)
		defer buffer.Close()
		perLayerInputs[layerIndex] = buffer
	}
	start := len(driver.launches)
	forward, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, cfg, tokens, 0, 1e-6, rocmKVCacheModeKQ8VQ4, perLayerInputs, nil, nil)
	core.RequireNoError(t, err)
	defer forward.Close()

	core.AssertEqual(t, len(tokens)*layer0.HiddenSize, forward.Embedding.Count())
	core.AssertEqual(t, 2, len(forward.Layers))
	core.AssertEqual(t, len(tokens)*layer0.HiddenSize, forward.FinalHidden.Count())
	core.AssertEqual(t, 0, len(forward.Greedy))
	finalHidden, err := hipReadFloat32DeviceOutput(forward.FinalHidden, hipGemma4Q4Layer0Operation, "prefill forward final hidden", len(tokens)*layer0.HiddenSize)
	core.RequireNoError(t, err)
	expectedHidden := make([]float32, len(tokens)*layer0.HiddenSize)
	for index := range expectedHidden {
		expectedHidden[index] = float32(math.Sqrt(float64(layer0.HiddenSize)))
	}
	assertFloat32SlicesNear(t, expectedHidden, finalHidden, 0.0001)

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameEmbedLookup))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameVectorScale))
	core.AssertEqual(t, 12, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	core.AssertEqual(t, 12, countLaunchName(launches, hipKernelNameRMSNormHeads))
	core.AssertEqual(t, 4, countLaunchName(launches, hipKernelNameRMSNormRoPEHeadsBatch))
	core.AssertEqual(t, len(cfg.Layers)*gemma4Q4DeviceKVPagesForTokens(len(tokens)), countLaunchName(launches, hipKernelNameKVEncodeToken))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMulBatch))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhProjBatch))
	core.AssertEqual(t, 6, countLaunchName(launches, hipKernelNameVectorAdd))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameRMSNorm))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4ProjGreedy))
}

func TestHIPGemma4Q4PrefillForwardBatchWithPrior_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	embeddingWeightsPayload, err := hipUint32Payload(make([]uint32, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize)))
	core.RequireNoError(t, err)
	embeddingWeights, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill forward prior nonzero embedding weights", embeddingWeightsPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingWeights.Close()
	embeddingScalesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingScales, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill forward prior nonzero embedding scales", embeddingScalesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingScales.Close()
	embeddingBiasesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingBiases, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill forward prior nonzero embedding biases", embeddingBiasesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingBiases.Close()
	layer0.Embedding = hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: embeddingWeights.Pointer(),
		EmbeddingBytes:   embeddingWeights.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		VocabSize:        layer0.VocabSize,
		HiddenSize:       layer0.HiddenSize,
		GroupSize:        layer0.GroupSize,
		ScalePointer:     embeddingScales.Pointer(),
		BiasPointer:      embeddingBiases.Pointer(),
		ScaleBytes:       embeddingScales.SizeBytes(),
		BiasBytes:        embeddingBiases.SizeBytes(),
	}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}
	tokens := []int32{0, 1}
	makePerLayerInputs := func(label string) []*hipDeviceByteBuffer {
		t.Helper()
		perLayerInputs := make([]*hipDeviceByteBuffer, len(cfg.Layers))
		for layerIndex, layer := range cfg.Layers {
			values := make([]float32, len(tokens)*layer.PerLayerInput.InputSize)
			for index := range values {
				values[index] = float32(layerIndex + 1)
			}
			payload, err := hipFloat32Payload(values)
			core.RequireNoError(t, err)
			buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, len(values))
			core.RequireNoError(t, err)
			buf := buffer
			t.Cleanup(func() {
				_ = buf.Close()
			})
			perLayerInputs[layerIndex] = buffer
		}
		return perLayerInputs
	}

	first, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, cfg, tokens, 0, 1e-6, rocmKVCacheModeKQ8VQ4, makePerLayerInputs("prefill forward prior first per-layer input"), nil, nil)
	core.RequireNoError(t, err)
	defer first.Close()
	prior := []*rocmDeviceKVCache{
		first.Layers[0].KV.DeviceKV.Cache,
		first.Layers[1].KV.DeviceKV.Cache,
	}

	start := len(driver.launches)
	second, err := hipRunGemma4Q4PrefillForwardBatchWithPrior(context.Background(), driver, cfg, tokens, len(tokens), 1e-6, rocmKVCacheModeKQ8VQ4, prior, makePerLayerInputs("prefill forward prior second per-layer input"), nil, nil)
	core.RequireNoError(t, err)
	defer second.Close()

	core.AssertEqual(t, 2, len(second.Layers))
	for index := range second.Layers {
		core.AssertEqual(t, len(tokens)*2, second.Layers[index].KV.DeviceKV.Cache.TokenCount())
	}

	launches := driver.launches[start:]
	var attentionLaunches []hipKernelLaunchConfig
	for _, launch := range launches {
		if launch.Name == hipKernelNameAttentionHeadsBatchCausal {
			attentionLaunches = append(attentionLaunches, launch)
		}
	}
	core.AssertEqual(t, 2, len(attentionLaunches))
	for _, launch := range attentionLaunches {
		core.AssertEqual(t, uint32(len(tokens)*2), binary.LittleEndian.Uint32(launch.Args[52:]))
		core.AssertEqual(t, uint32(len(tokens)), binary.LittleEndian.Uint32(launch.Args[60:]))
		core.AssertEqual(t, uint32(len(tokens)), binary.LittleEndian.Uint32(launch.Args[64:]))
	}
	core.AssertEqual(t, len(cfg.Layers)*gemma4Q4DeviceKVPagesForTokens(len(tokens)), countLaunchName(launches, hipKernelNameKVEncodeToken))
}

func TestHIPGemma4Q4PrefillForwardBatchWithGeneratedPerLayerInput_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	embeddingWeightsPayload, err := hipUint32Payload(make([]uint32, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize)))
	core.RequireNoError(t, err)
	embeddingWeights, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated input nonzero embedding weights", embeddingWeightsPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingWeights.Close()
	embeddingScalesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingScales, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated input nonzero embedding scales", embeddingScalesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingScales.Close()
	embeddingBiasesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingBiases, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated input nonzero embedding biases", embeddingBiasesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingBiases.Close()
	layer0.Embedding = hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: embeddingWeights.Pointer(),
		EmbeddingBytes:   embeddingWeights.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		VocabSize:        layer0.VocabSize,
		HiddenSize:       layer0.HiddenSize,
		GroupSize:        layer0.GroupSize,
		ScalePointer:     embeddingScales.Pointer(),
		BiasPointer:      embeddingBiases.Pointer(),
		ScaleBytes:       embeddingScales.SizeBytes(),
		BiasBytes:        embeddingBiases.SizeBytes(),
	}

	layerCount := 2
	inputSize := layer0.PerLayerInput.InputSize
	globalRows := layerCount * inputSize
	globalWeightsPayload, err := hipUint32Payload(make([]uint32, layer0.VocabSize*(globalRows/layer0.GroupSize)))
	core.RequireNoError(t, err)
	globalWeights, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated per-layer embedding weights", globalWeightsPayload, layer0.VocabSize*(globalRows/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer globalWeights.Close()
	globalScalesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80, 0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	globalScales, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated per-layer embedding scales", globalScalesPayload, layer0.VocabSize*(globalRows/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer globalScales.Close()
	globalBiasesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80, 0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	globalBiases, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated per-layer embedding biases", globalBiasesPayload, layer0.VocabSize*(globalRows/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer globalBiases.Close()
	modelProjectionPayload, err := hipUint16Payload(repeatUint16(0x3f80, globalRows*layer0.HiddenSize))
	core.RequireNoError(t, err)
	modelProjection, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated model projection", modelProjectionPayload, globalRows*layer0.HiddenSize)
	core.RequireNoError(t, err)
	defer modelProjection.Close()
	projectionNormPayload, err := hipUint16Payload(repeatUint16(0x3f80, inputSize))
	core.RequireNoError(t, err)
	projectionNorm, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated projection norm", projectionNormPayload, inputSize)
	core.RequireNoError(t, err)
	defer projectionNorm.Close()
	layer0.PerLayerInput.Embedding = hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: globalWeights.Pointer(),
		EmbeddingBytes:   globalWeights.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		VocabSize:        layer0.VocabSize,
		HiddenSize:       globalRows,
		GroupSize:        layer0.GroupSize,
		ScalePointer:     globalScales.Pointer(),
		BiasPointer:      globalBiases.Pointer(),
		ScaleBytes:       globalScales.SizeBytes(),
		BiasBytes:        globalBiases.SizeBytes(),
	}
	layer0.PerLayerInput.ModelProjection = hipBF16DeviceWeightConfig{
		WeightPointer: modelProjection.Pointer(),
		WeightBytes:   modelProjection.SizeBytes(),
		Rows:          globalRows,
		Cols:          layer0.HiddenSize,
	}
	layer0.PerLayerInput.ProjectionNorm = hipRMSNormDeviceWeightConfig{
		WeightPointer:  projectionNorm.Pointer(),
		WeightBytes:    projectionNorm.SizeBytes(),
		Count:          inputSize,
		WeightEncoding: hipRMSNormWeightEncodingBF16,
	}

	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}
	tokens := []int32{0, 1}
	start := len(driver.launches)
	forward, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, cfg, tokens, 0, 1e-6, rocmKVCacheModeKQ8VQ4, nil, nil, nil)
	core.RequireNoError(t, err)
	defer forward.Close()

	core.AssertEqual(t, len(tokens)*layer0.HiddenSize, forward.FinalHidden.Count())
	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameProjectionBatch))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNamePerLayerInputTranspose))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhProjBatch))
}

func TestHIPGemma4Q4PrefillLayerBodyBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	inputValues := make([]float32, cfg.HiddenSize)
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer body bad input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillLayerBodyBatch(context.Background(), driver, cfg, input, nil, 0, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerBodyBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillLayerBodyBatch(context.Background(), driver, cfg, input, nil, 1, -1, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerBodyBatch succeeded with negative query start")
	}
	if _, err := hipRunGemma4Q4PrefillLayerBodyBatch(context.Background(), driver, cfg, input, nil, 2, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerBodyBatch succeeded with mismatched input shape")
	}
	if _, err := hipRunGemma4Q4PrefillLayerBodyBatch(context.Background(), driver, cfg, input, nil, 1, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerBodyBatch succeeded with missing layer")
	}
	badPerLayerPayload, err := hipFloat32Payload([]float32{1})
	core.RequireNoError(t, err)
	badPerLayer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer body bad per-layer input fixture", badPerLayerPayload, 1)
	core.RequireNoError(t, err)
	defer badPerLayer.Close()
	if _, err := hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInput(context.Background(), driver, cfg, input, nil, badPerLayer, 1, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInput succeeded with mismatched per-layer input")
	}
	if _, err := hipRunGemma4Q4PrefillFinalGreedyForRow(context.Background(), driver, cfg, input, 1, 1, 1e-6, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillFinalGreedyForRow succeeded with row outside token batch")
	}
	if _, err := hipRunGemma4Q4PrefillFinalGreedyForRow(context.Background(), driver, cfg, input, 2, 0, 1e-6, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillFinalGreedyForRow succeeded with mismatched hidden batch shape")
	}
	forwardCfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg}}
	if _, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, forwardCfg, []int32{0}, 1, 1e-6, rocmKVCacheModeKQ8VQ4, nil, nil, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillForwardBatch succeeded with nonzero start position")
	}
	prior := &rocmDeviceKVCache{driver: driver, mode: rocmKVCacheModeKQ8VQ4, blockSize: defaultROCmKVBlockSize, tokenCount: 2}
	if _, err := hipRunGemma4Q4PrefillForwardBatchWithPrior(context.Background(), driver, forwardCfg, []int32{0}, 0, 1e-6, rocmKVCacheModeKQ8VQ4, []*rocmDeviceKVCache{prior}, nil, nil, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillForwardBatchWithPrior succeeded with prior at start position 0")
	}
	if _, err := hipRunGemma4Q4PrefillForwardBatchWithPrior(context.Background(), driver, forwardCfg, []int32{0}, 1, 1e-6, rocmKVCacheModeKQ8VQ4, []*rocmDeviceKVCache{prior}, nil, nil, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillForwardBatchWithPrior succeeded with mismatched prior token count")
	}
	if _, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, forwardCfg, []int32{0}, 0, 1e-6, rocmKVCacheModeKQ8VQ4, nil, nil, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillForwardBatch succeeded without required per-layer inputs")
	}
	if _, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, forwardCfg, []int32{0}, 0, 1e-6, rocmKVCacheModeKQ8VQ4, []*hipDeviceByteBuffer{badPerLayer}, []bool{true, false}, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillForwardBatch succeeded with mismatched output mask")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillLayerKVBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	inputValues := make([]float32, cfg.HiddenSize)
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer KV bad input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, 0, 0, 1e-6, rocmKVCacheModeKQ8VQ4); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerKVBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, 2, 0, 1e-6, rocmKVCacheModeKQ8VQ4); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerKVBatch succeeded with mismatched input shape")
	}
	if _, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, 1, -1, 1e-6, rocmKVCacheModeKQ8VQ4); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerKVBatch succeeded with negative start position")
	}
	prior := &rocmDeviceKVCache{driver: driver, mode: rocmKVCacheModeKQ8VQ4, blockSize: defaultROCmKVBlockSize, tokenCount: 2}
	if _, err := hipRunGemma4Q4PrefillLayerKVBatchWithPrior(context.Background(), driver, cfg, input, prior, 1, 1, 1e-6, rocmKVCacheModeKQ8VQ4); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerKVBatchWithPrior succeeded with mismatched prior token count")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillMLPBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	tokenCount := 2
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill MLP fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillMLPBatch(context.Background(), driver, cfg, input, tokenCount)
	core.RequireNoError(t, err)
	defer output.Close()

	core.AssertEqual(t, tokenCount*cfg.DownProjection.Rows, output.Count())
	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMulBatch))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	for _, launch := range launches {
		core.AssertEqual(t, uint32((tokenCount+hipMLXQ4ProjectionBatchTokensPerBlock-1)/hipMLXQ4ProjectionBatchTokensPerBlock), launch.GridY)
	}
}

func TestHIPGemma4Q4PrefillMLPBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	inputValues := make([]float32, cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill MLP bad fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillMLPBatch(context.Background(), driver, cfg, input, 0); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillMLPBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillMLPBatch(context.Background(), driver, cfg, input, 2); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillMLPBatch succeeded with mismatched token count")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4GenerateTokenSeq_BadPrefillUBatchEnv(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	t.Setenv(hipGemma4Q4PrefillUBatchEnv, "nope")
	stream, streamErr := hipGemma4Q4GenerateTokenSeq(context.Background(), &hipLoadedModel{driver: driver}, hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg}}, []int32{1}, inference.GenerateConfig{MaxTokens: 1})
	for range stream {
		t.Fatalf("hipGemma4Q4GenerateTokenSeq yielded token, want prefill ubatch env error")
	}
	err := streamErr()
	if err == nil {
		t.Fatalf("hipGemma4Q4GenerateTokenSeq succeeded, want prefill ubatch env error")
	}
	core.AssertContains(t, err.Error(), hipGemma4Q4PrefillUBatchEnv)
	core.AssertEqual(t, 0, len(driver.launches))
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

	q4Ops := 0
	tripleQ4Launches := 0
	for _, launch := range driver.launches[start:] {
		switch launch.Name {
		case hipKernelNameMLXQ4Proj:
			q4Ops++
		case hipKernelNameMLXQ4TripleProj:
			q4Ops += 3
			tripleQ4Launches++
		}
	}
	core.AssertEqual(t, 17, q4Ops)
	core.AssertEqual(t, 2, tripleQ4Launches)
}

func TestHIPGemma4Q4SharedDeviceKV_Good(t *testing.T) {
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
	layer0.SlidingWindow = 1
	layer1.LayerType = "full_attention"
	layer1.SlidingWindow = 0
	layer2.LayerType = "sliding_attention"
	layer2.SlidingWindow = 1
	layer3.LayerType = "full_attention"
	layer3.SlidingWindow = 0
	cfg := hipGemma4Q4ForwardConfig{
		Layers:         []hipGemma4Q4Layer0Config{layer0, layer1, layer2, layer3},
		KVSharedLayers: 2,
	}

	launchStart := len(driver.launches)
	first, firstState, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:           1,
		Position:          0,
		Epsilon:           1e-6,
		DeviceKVAttention: true,
		DeviceKVMode:      rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState: true,
		DeviceFinalSample: true,
		OmitDebugTensors:  true,
	}, false)
	core.RequireNoError(t, err)
	if first.DeviceState == nil {
		t.Fatal("first forward device state is nil")
	}
	core.AssertEqual(t, []int{1, 1, 1, 1}, first.DeviceState.LayerTokenCounts())
	core.AssertEqual(t, "0", first.Labels["attention_kv_remirror_layers"])
	core.AssertEqual(t, "2", first.Labels["attention_kv_shared_device_layers"])
	core.AssertEqual(t, "2", first.Labels["gemma4_q4_device_kv_shared_layers"])
	core.AssertEqual(t, 2, countKVEncodeTokenLaunches(driver.launches[launchStart:]))
	for index, layer := range firstState.Layers {
		if len(layer.Keys) != 0 || len(layer.Values) != 0 {
			t.Fatalf("first host state layer %d retained host KV in device-only generation path", index)
		}
	}

	priorDeviceState := first.DeviceState
	first.DeviceState = nil
	second, secondState, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, firstState, hipGemma4Q4ForwardRequest{
		TokenID:           int32(first.Greedy.TokenID),
		Position:          1,
		Epsilon:           1e-6,
		DeviceKVAttention: true,
		DeviceKVMode:      rocmKVCacheModeKQ8VQ4,
		PriorDeviceState:  priorDeviceState,
		ReturnDeviceState: true,
		DeviceFinalSample: true,
		OmitDebugTensors:  true,
	}, false)
	if err != nil {
		_ = priorDeviceState.Close()
	}
	core.RequireNoError(t, err)
	defer second.DeviceState.Close()
	core.AssertEqual(t, true, priorDeviceState.closed)
	core.AssertEqual(t, []int{1, 2, 1, 2}, second.DeviceState.LayerTokenCounts())
	core.AssertEqual(t, "2", second.Labels["attention_kv_append_layers"])
	core.AssertEqual(t, "0", second.Labels["attention_kv_remirror_layers"])
	core.AssertEqual(t, "2", second.Labels["attention_kv_shared_device_layers"])
	core.AssertEqual(t, "2", second.Labels["gemma4_q4_device_kv_shared_layers"])
	for index, layer := range secondState.Layers {
		if len(layer.Keys) != 0 || len(layer.Values) != 0 {
			t.Fatalf("second host state layer %d retained host KV in device-only generation path", index)
		}
	}
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 shared-device forward launched no descriptor-backed attention kernels")
	}
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

func TestHIPGemma4Q4SkipFinalSample_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}

	launchStart := len(driver.launches)
	forward, state, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:           1,
		Position:          0,
		Epsilon:           1e-6,
		DeviceKVAttention: true,
		DeviceKVMode:      rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState: true,
		SkipFinalSample:   true,
		OmitDebugTensors:  true,
	}, false)
	core.RequireNoError(t, err)
	defer forward.DeviceState.Close()

	core.AssertEqual(t, 0, len(forward.Logits))
	core.AssertEqual(t, 0, forward.Greedy.TokenID)
	assertFloat32Near(t, 0, forward.Greedy.Score)
	core.AssertEqual(t, "skipped", forward.Labels["gemma4_q4_final_sample"])
	core.AssertEqual(t, []int{1, 1}, forward.DeviceState.LayerTokenCounts())
	for index, layer := range state.Layers {
		if len(layer.Keys) != 0 || len(layer.Values) != 0 {
			t.Fatalf("skip-final forward host state layer %d retained host KV in device-only path", index)
		}
	}
	launches := driver.launches[launchStart:]
	if countLaunchName(launches, hipKernelNameMLXQ4ProjGreedy) != 0 {
		t.Fatalf("skip-final forward launched fused LM-head greedy projection")
	}
	if countLaunchName(launches, hipKernelNameGreedy) != 0 {
		t.Fatalf("skip-final forward launched host greedy sampling")
	}
	if countLaunchName(launches, hipKernelNameMLXQ4Proj) == 0 {
		t.Fatalf("skip-final forward did not run decoder q4 projections")
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
		if launch.Name == hipKernelNameAttentionHeads &&
			len(launch.Args) >= hipAttentionHeadsLaunchArgsBytes &&
			binary.LittleEndian.Uint32(launch.Args[80:]) == hipAttentionKVSourceDevice {
			count++
		}
	}
	return count
}

func countLaunchName(launches []hipKernelLaunchConfig, name string) int {
	var count int
	for _, launch := range launches {
		if launch.Name == name {
			count++
		}
	}
	return count
}

func gemma4Q4DeviceKVPagesForTokens(tokens int) int {
	if tokens <= 0 {
		return 0
	}
	blockSize := hipGemma4Q4DeviceKVBlockSize()
	if blockSize <= 0 {
		return tokens
	}
	return (tokens + blockSize - 1) / blockSize
}

func countKVEncodeTokenLaunches(launches []hipKernelLaunchConfig) int {
	var count int
	for _, launch := range launches {
		if launch.Name == hipKernelNameKVEncodeToken {
			count++
		}
	}
	return count
}

func repeatUint16(value uint16, count int) []uint16 {
	values := make([]uint16, count)
	for index := range values {
		values[index] = value
	}
	return values
}

func TestHIPAttentionHeadsBlockSize_Good(t *testing.T) {
	core.AssertEqual(t, uint32(256), hipAttentionHeadsBlockSize(1))
	core.AssertEqual(t, uint32(256), hipAttentionHeadsBlockSize(15))
	core.AssertEqual(t, uint32(512), hipAttentionHeadsBlockSize(16))
	core.AssertEqual(t, uint32(512), hipAttentionHeadsBlockSize(511))
	core.AssertEqual(t, uint32(512), hipAttentionHeadsBlockSize(512))
	core.AssertEqual(t, uint32(512), hipAttentionHeadsBlockSize(1023))
	core.AssertEqual(t, uint32(512), hipAttentionHeadsBlockSize(1024))
	core.AssertEqual(t, uint32(512), hipAttentionHeadsBlockSize(2000))
}

func TestHIPAttentionHeadsSharedMemBytes_Good(t *testing.T) {
	plain, err := hipAttentionHeadsSharedMemBytes(2000, false)
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint32(8000), plain)

	shortDevice, err := hipAttentionHeadsSharedMemBytes(511, true)
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint32(8180), shortDevice)

	longDevice, err := hipAttentionHeadsSharedMemBytes(2000, true)
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint32(32000), longDevice)
}

func TestHIPAttentionHeadsChunkedSharedMemBytes_Good(t *testing.T) {
	dim256, err := hipAttentionHeadsChunkedSharedMemBytes(128, 256)
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint32(3072), dim256)

	dim512, err := hipAttentionHeadsChunkedSharedMemBytes(128, 512)
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint32(4096), dim512)

	_, err = hipAttentionHeadsChunkedSharedMemBytes(0, 512)
	core.AssertNotEqual(t, nil, err)
}

func BenchmarkHIPDeviceByteBufferPool_ReusedSize(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	const sizeBytes uint64 = 4096
	const pointer nativeDevicePointer = 42
	hipDeviceByteBufferPool.Lock()
	hipDeviceByteBufferPool.entries = make(map[uint64][]hipDeviceByteBufferPoolEntry)
	hipDeviceByteBufferPool.bytes = 0
	hipDeviceByteBufferPool.Unlock()
	if !hipDeviceByteBufferPoolPut(driver, pointer, sizeBytes) {
		b.Fatal("seed device buffer pool")
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got, ok := hipDeviceByteBufferPoolTake(driver, sizeBytes)
		if !ok || got != pointer {
			b.Fatalf("take = %d, %v; want %d, true", got, ok, pointer)
		}
		if !hipDeviceByteBufferPoolPut(driver, got, sizeBytes) {
			b.Fatal("return device buffer to pool")
		}
	}
}

func BenchmarkHIPLaunchPacketPool_ReusedSize(b *testing.B) {
	hipLaunchPacketPools.Range(func(key, _ any) bool {
		hipLaunchPacketPools.Delete(key)
		return true
	})
	packet := hipBorrowLaunchPacket(hipMLXQ4TripleProjLaunchArgsBytes)
	hipReleaseLaunchPacket(packet)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet = hipBorrowLaunchPacket(hipMLXQ4TripleProjLaunchArgsBytes)
		if len(packet) != hipMLXQ4TripleProjLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4TripleProjLaunchArgsBytes)
		}
		hipReleaseLaunchPacket(packet)
	}
}

func BenchmarkHIPKernelLaunchConfigValidate_Hot(b *testing.B) {
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4Proj,
		Args:   []byte{1},
		GridX:  1,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := config.Validate(); err != nil {
			b.Fatal(err)
		}
	}
}

type fakeHIPUint64Reader struct {
	fakeHIPDriver
	value uint64
}

func (driver *fakeHIPUint64Reader) CopyDeviceToHostUint64(nativeDevicePointer) (uint64, error) {
	return driver.value, nil
}

func BenchmarkHIPReadDeviceUint64_DirectReader(b *testing.B) {
	driver := &fakeHIPUint64Reader{
		fakeHIPDriver: fakeHIPDriver{available: true},
		value:         0x400921fb54442d18,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		value, err := hipReadDeviceUint64(driver, 42)
		if err != nil {
			b.Fatal(err)
		}
		if value != driver.value {
			b.Fatalf("value = %#x, want %#x", value, driver.value)
		}
	}
}

func BenchmarkHIPGemma4Q4DeviceDecodeStatePool_Reused(b *testing.B) {
	state := hipNewGemma4Q4DeviceDecodeState(rocmKVCacheModeKQ8VQ4, 35)
	hipReleaseGemma4Q4DeviceLayerStates(state.layers)
	state.layers = nil
	state.closed = true
	hipReleaseClosedGemma4Q4DeviceDecodeState(state)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		state = hipNewGemma4Q4DeviceDecodeState(rocmKVCacheModeKQ8VQ4, 35)
		if state == nil || state.mode != rocmKVCacheModeKQ8VQ4 || len(state.layers) != 0 || cap(state.layers) < 35 {
			b.Fatal("decode state pool returned invalid state")
		}
		hipReleaseGemma4Q4DeviceLayerStates(state.layers)
		state.layers = nil
		state.closed = true
		hipReleaseClosedGemma4Q4DeviceDecodeState(state)
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_AttentionOutputReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureAttentionOutput(driver, 8, 256)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 2048 || output.SizeBytes() != 8192 {
		b.Fatalf("attention output shape = %d/%d, want 2048/8192", output.Count(), output.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureAttentionOutput(driver, 8, 256)
		if err != nil {
			b.Fatal(err)
		}
		if output.Count() != 2048 || output.SizeBytes() != 8192 {
			b.Fatalf("attention output shape = %d/%d, want 2048/8192", output.Count(), output.SizeBytes())
		}
	}
}

func TestHIPAttentionHeadsBatchCausalWorkspaceCap_Good(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	const (
		dim        = 1
		tokenCount = hipAttentionHeadsSharedMaxTokens + 1
		queryCount = 1
	)
	keyValues := make([]float32, tokenCount*dim)
	valueValues := make([]float32, tokenCount*dim)
	for index := range keyValues {
		keyValues[index] = 1
		valueValues[index] = float32(index + 1)
	}
	keyPayload, err := hipFloat32Payload(keyValues)
	core.RequireNoError(t, err)
	valuePayload, err := hipFloat32Payload(valueValues)
	core.RequireNoError(t, err)

	for _, tc := range []struct {
		name          string
		headCount     int
		wantWorkspace bool
	}{
		{name: "under_cap", headCount: 1, wantWorkspace: true},
		{name: "over_cap", headCount: 33, wantWorkspace: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			driver := &fakeHIPDriver{available: true}
			workspace := &hipAttentionHeadsChunkedWorkspace{}
			defer workspace.Close()
			queryValues := make([]float32, queryCount*tc.headCount*dim)
			for index := range queryValues {
				queryValues[index] = 1
			}
			queryPayload, err := hipFloat32Payload(queryValues)
			core.RequireNoError(t, err)
			query, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch query", queryPayload, len(queryValues))
			core.RequireNoError(t, err)
			defer query.Close()
			keys, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch keys", keyPayload, len(keyValues))
			core.RequireNoError(t, err)
			defer keys.Close()
			values, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch values", valuePayload, len(valueValues))
			core.RequireNoError(t, err)
			defer values.Close()
			output, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch output", uint64(len(queryValues)*4), len(queryValues))
			core.RequireNoError(t, err)
			defer output.Close()

			err = hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernelWorkspace(context.Background(), driver, hipAttentionHeadsBatchCausalDeviceRequest{
				Key:             keys,
				Value:           values,
				Dim:             dim,
				TokenCount:      tokenCount,
				HeadCount:       tc.headCount,
				QueryCount:      queryCount,
				QueryStartToken: tokenCount - 1,
				Scale:           1,
			}, query, output, workspace)
			core.RequireNoError(t, err)
			launch := driver.launches[len(driver.launches)-1]
			weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(launch.Args[40:]))
			core.AssertEqual(t, uint32(queryCount*tc.headCount*tokenCount*4), binary.LittleEndian.Uint32(launch.Args[84:]))
			if tc.wantWorkspace {
				if workspace.BatchAttentionWeight == nil || workspace.BatchAttentionWeight.Pointer() != weightPointer {
					t.Fatalf("workspace weight pointer = %#v, launch pointer %x", workspace.BatchAttentionWeight, weightPointer)
				}
				core.AssertEqual(t, 0, len(driver.frees))
				return
			}
			if workspace.BatchAttentionWeight != nil {
				t.Fatalf("workspace retained over-cap attention weights")
			}
			foundFree := false
			for _, freed := range driver.frees {
				if freed == weightPointer {
					foundFree = true
					break
				}
			}
			if !foundFree {
				t.Fatalf("over-cap attention weights %x were not released", weightPointer)
			}
		})
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_BatchAttentionWeightsReused_Good(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	first, err := workspace.EnsureBatchAttentionWeights(driver, 4096)
	core.RequireNoError(t, err)
	if first == nil || first.Count() != 4096 || first.SizeBytes() != 16384 {
		t.Fatalf("batch attention weights = %#v, want 4096/16384", first)
	}
	firstPointer := first.Pointer()
	core.AssertEqual(t, 1, len(driver.allocations))

	smaller, err := workspace.EnsureBatchAttentionWeights(driver, 2048)
	core.RequireNoError(t, err)
	if smaller.Pointer() != firstPointer || smaller.Count() != 4096 {
		t.Fatalf("smaller weights reused pointer/count = %#v, want pointer %x count 4096", smaller, firstPointer)
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	core.AssertEqual(t, 0, len(driver.frees))

	larger, err := workspace.EnsureBatchAttentionWeights(driver, 8192)
	core.RequireNoError(t, err)
	if larger == nil || larger.Pointer() == firstPointer || larger.Count() != 8192 || larger.SizeBytes() != 32768 {
		t.Fatalf("larger weights = %#v, want fresh 8192/32768 buffer", larger)
	}
	core.AssertEqual(t, 2, len(driver.allocations))
	core.AssertEqual(t, 1, len(driver.frees))
	core.AssertEqual(t, firstPointer, driver.frees[0])
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_BatchAttentionWeightsReused(b *testing.B) {
	b.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureBatchAttentionWeights(driver, 4096)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 4096 || output.SizeBytes() != 16384 {
		b.Fatalf("batch attention weight shape = %d/%d, want 4096/16384", output.Count(), output.SizeBytes())
	}
	pointer := output.Pointer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureBatchAttentionWeights(driver, 2048)
		if err != nil {
			b.Fatal(err)
		}
		if output.Pointer() != pointer || output.Count() != 4096 || output.SizeBytes() != 16384 {
			b.Fatalf("batch attention weight shape = %x %d/%d, want %x 4096/16384", output.Pointer(), output.Count(), output.SizeBytes(), pointer)
		}
	}
}

func BenchmarkHIPAttentionHeadsBatchChunkedLaunchArgs_FullWindow(b *testing.B) {
	const (
		dim        = 256
		tokenCount = 4096
		headCount  = 8
		queryCount = 16
		chunkSize  = hipAttentionHeadsChunkSize
	)
	chunkCount := (tokenCount + chunkSize - 1) / chunkSize
	queryElements := dim * headCount * queryCount
	args := hipAttentionHeadsBatchChunkedLaunchArgs{
		QueryPointer:      1,
		DescriptorPointer: 2,
		PartialPointer:    3,
		StatsPointer:      4,
		OutputPointer:     5,
		Dim:               dim,
		TokenCount:        tokenCount,
		HeadCount:         headCount,
		QueryCount:        queryCount,
		QueryStartToken:   tokenCount - queryCount,
		ChunkSize:         chunkSize,
		ChunkCount:        chunkCount,
		QueryBytes:        uint64(queryElements * 4),
		DescriptorBytes:   uint64(rocmDeviceKVDescriptorHeaderBytes + tokenCount*rocmDeviceKVDescriptorPageBytes),
		PartialBytes:      uint64(queryElements * chunkCount * 4),
		StatsBytes:        uint64(queryCount * headCount * chunkCount * 2 * 4),
		OutputBytes:       uint64(queryElements * 4),
		Scale:             1,
	}
	packet, err := args.Binary()
	if err != nil {
		b.Fatal(err)
	}
	hipReleaseLaunchPacket(packet)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.Binary()
		if err != nil {
			b.Fatal(err)
		}
		hipReleaseLaunchPacket(packet)
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_ProjectionOutputReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureProjectionOutput(driver, 2304)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 2304 || output.SizeBytes() != 9216 {
		b.Fatalf("projection output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureProjectionOutput(driver, 2304)
		if err != nil {
			b.Fatal(err)
		}
		if output.Count() != 2304 || output.SizeBytes() != 9216 {
			b.Fatalf("projection output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
		}
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_ActivationOutputReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureActivationOutput(driver, 9216)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 9216 || output.SizeBytes() != 36864 {
		b.Fatalf("activation output shape = %d/%d, want 9216/36864", output.Count(), output.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureActivationOutput(driver, 9216)
		if err != nil {
			b.Fatal(err)
		}
		if output.Count() != 9216 || output.SizeBytes() != 36864 {
			b.Fatalf("activation output shape = %d/%d, want 9216/36864", output.Count(), output.SizeBytes())
		}
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_RMSOutputsReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	residualOutput, err := workspace.EnsureRMSResidualOutput(driver, 2304)
	if err != nil {
		b.Fatal(err)
	}
	normOutput, err := workspace.EnsureRMSNormOutput(driver, 2304)
	if err != nil {
		b.Fatal(err)
	}
	if residualOutput.Count() != 2304 || residualOutput.SizeBytes() != 9216 {
		b.Fatalf("RMS residual output shape = %d/%d, want 2304/9216", residualOutput.Count(), residualOutput.SizeBytes())
	}
	if normOutput.Count() != 2304 || normOutput.SizeBytes() != 9216 {
		b.Fatalf("RMS norm output shape = %d/%d, want 2304/9216", normOutput.Count(), normOutput.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		residualOutput, err = workspace.EnsureRMSResidualOutput(driver, 2304)
		if err != nil {
			b.Fatal(err)
		}
		normOutput, err = workspace.EnsureRMSNormOutput(driver, 2304)
		if err != nil {
			b.Fatal(err)
		}
		if residualOutput.Count() != 2304 || residualOutput.SizeBytes() != 9216 {
			b.Fatalf("RMS residual output shape = %d/%d, want 2304/9216", residualOutput.Count(), residualOutput.SizeBytes())
		}
		if normOutput.Count() != 2304 || normOutput.SizeBytes() != 9216 {
			b.Fatalf("RMS norm output shape = %d/%d, want 2304/9216", normOutput.Count(), normOutput.SizeBytes())
		}
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_RMSRoPEOutputsReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	queryOutput, err := workspace.EnsureRMSRoPEOutput(driver, 2048)
	if err != nil {
		b.Fatal(err)
	}
	keyOutput, err := workspace.EnsureRMSRoPEOutput(driver, 256)
	if err != nil {
		b.Fatal(err)
	}
	noScaleOutput, err := workspace.EnsureRMSNoScaleOutput(driver, 256)
	if err != nil {
		b.Fatal(err)
	}
	if queryOutput.Count() != 2048 || queryOutput.SizeBytes() != 8192 {
		b.Fatalf("RMS RoPE query output shape = %d/%d, want 2048/8192", queryOutput.Count(), queryOutput.SizeBytes())
	}
	if keyOutput.Count() != 256 || keyOutput.SizeBytes() != 1024 {
		b.Fatalf("RMS RoPE key output shape = %d/%d, want 256/1024", keyOutput.Count(), keyOutput.SizeBytes())
	}
	if noScaleOutput.Count() != 256 || noScaleOutput.SizeBytes() != 1024 {
		b.Fatalf("RMS no-scale output shape = %d/%d, want 256/1024", noScaleOutput.Count(), noScaleOutput.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		queryOutput, err = workspace.EnsureRMSRoPEOutput(driver, 2048)
		if err != nil {
			b.Fatal(err)
		}
		keyOutput, err = workspace.EnsureRMSRoPEOutput(driver, 256)
		if err != nil {
			b.Fatal(err)
		}
		noScaleOutput, err = workspace.EnsureRMSNoScaleOutput(driver, 256)
		if err != nil {
			b.Fatal(err)
		}
		if queryOutput.Count() != 2048 || queryOutput.SizeBytes() != 8192 {
			b.Fatalf("RMS RoPE query output shape = %d/%d, want 2048/8192", queryOutput.Count(), queryOutput.SizeBytes())
		}
		if keyOutput.Count() != 256 || keyOutput.SizeBytes() != 1024 {
			b.Fatalf("RMS RoPE key output shape = %d/%d, want 256/1024", keyOutput.Count(), keyOutput.SizeBytes())
		}
		if noScaleOutput.Count() != 256 || noScaleOutput.SizeBytes() != 1024 {
			b.Fatalf("RMS no-scale output shape = %d/%d, want 256/1024", noScaleOutput.Count(), noScaleOutput.SizeBytes())
		}
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_IntermediateOutputReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureIntermediateOutput(driver, 2304)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 2304 || output.SizeBytes() != 9216 {
		b.Fatalf("intermediate output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureIntermediateOutput(driver, 2304)
		if err != nil {
			b.Fatal(err)
		}
		if output.Count() != 2304 || output.SizeBytes() != 9216 {
			b.Fatalf("intermediate output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
		}
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_QKVOutputReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureQKVOutput(driver, 2560)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 2560 || output.SizeBytes() != 10240 {
		b.Fatalf("QKV output shape = %d/%d, want 2560/10240", output.Count(), output.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureQKVOutput(driver, 2560)
		if err != nil {
			b.Fatal(err)
		}
		if output.Count() != 2560 || output.SizeBytes() != 10240 {
			b.Fatalf("QKV output shape = %d/%d, want 2560/10240", output.Count(), output.SizeBytes())
		}
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_FinalHiddenOutputReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureFinalHiddenOutput(driver, 2304, 0)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 2304 || output.SizeBytes() != 9216 {
		b.Fatalf("final hidden output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureFinalHiddenOutput(driver, 2304, i&1)
		if err != nil {
			b.Fatal(err)
		}
		if output.Count() != 2304 || output.SizeBytes() != 9216 {
			b.Fatalf("final hidden output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
		}
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_NextInputOutputReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureNextInputOutput(driver, 2304, 0)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 2304 || output.SizeBytes() != 9216 {
		b.Fatalf("next input output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureNextInputOutput(driver, 2304, i&1)
		if err != nil {
			b.Fatal(err)
		}
		if output.Count() != 2304 || output.SizeBytes() != 9216 {
			b.Fatalf("next input output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
		}
	}
}

func BenchmarkHIPGemma4Q4PerLayerInputDeviceSetLayer_View(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	const (
		layerCount = 32
		inputSize  = 2304
	)
	set := &hipGemma4Q4PerLayerInputDeviceSet{
		driver:           driver,
		layerCount:       layerCount,
		layerStrideBytes: uint64(inputSize * 4),
		layerValueCount:  inputSize,
		viewLabel:        "per-layer input slice",
		Backing: []*hipDeviceByteBuffer{{
			driver:    driver,
			pointer:   0x100000,
			count:     layerCount * inputSize,
			sizeBytes: uint64(layerCount * inputSize * 4),
		}},
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		layer := set.Layer(i % layerCount)
		if layer == nil || layer.Pointer() == 0 || layer.Count() != inputSize {
			b.Fatalf("layer view = %#v", layer)
		}
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_PerLayerInputDeviceSetReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	const (
		layerCount = 32
		inputSize  = 2304
	)
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	backing := &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   0x100000,
		count:     layerCount * inputSize,
		sizeBytes: uint64(layerCount * inputSize * 4),
	}
	set, err := workspace.BorrowPerLayerInputDeviceSet(driver, layerCount, inputSize, backing)
	if err != nil {
		b.Fatal(err)
	}
	if set.LayerCount() != layerCount || set.Layer(0) == nil {
		b.Fatalf("per-layer input set = %#v, want reusable device views", set)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		set, err = workspace.BorrowPerLayerInputDeviceSet(driver, layerCount, inputSize, backing)
		if err != nil {
			b.Fatal(err)
		}
		layer := set.Layer(i % layerCount)
		if layer == nil || layer.Pointer() == 0 || layer.Count() != inputSize {
			b.Fatalf("layer view = %#v", layer)
		}
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_SuppressTokenBufferReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	tokens := []int32{0, 2, 105, 106, 107, 200}
	buffer, err := workspace.EnsureSuppressTokenBuffer(driver, tokens)
	if err != nil {
		b.Fatal(err)
	}
	if buffer == nil || buffer.Count() != len(tokens) {
		b.Fatalf("suppress token buffer = %#v, want %d tokens", buffer, len(tokens))
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		buffer, err = workspace.EnsureSuppressTokenBuffer(driver, tokens)
		if err != nil {
			b.Fatal(err)
		}
		if buffer == nil || buffer.Count() != len(tokens) {
			b.Fatalf("suppress token buffer = %#v, want %d tokens", buffer, len(tokens))
		}
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_TokenIDValueCached_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	buffer, err := workspace.EnsureTokenIDValue(driver, 42, 128)
	core.RequireNoError(t, err)
	if buffer == nil || buffer.Count() != 1 {
		t.Fatalf("token buffer = %#v, want one token", buffer)
	}
	copiesAfterFirst := len(driver.copies)
	_, err = workspace.EnsureTokenIDValue(driver, 42, 128)
	core.RequireNoError(t, err)
	core.AssertEqual(t, copiesAfterFirst, len(driver.copies))
	_, err = workspace.EnsureTokenIDValue(driver, 43, 128)
	core.RequireNoError(t, err)
	core.AssertEqual(t, copiesAfterFirst+1, len(driver.copies))
	_, err = workspace.EnsureTokenIDValue(driver, 128, 128)
	core.AssertError(t, err)
	core.AssertEqual(t, copiesAfterFirst+1, len(driver.copies))
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_TokenIDValueCached(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	buffer, err := workspace.EnsureTokenIDValue(driver, 42, 128)
	if err != nil {
		b.Fatal(err)
	}
	if buffer == nil || buffer.Count() != 1 {
		b.Fatalf("token buffer = %#v, want one token", buffer)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		buffer, err = workspace.EnsureTokenIDValue(driver, 42, 128)
		if err != nil {
			b.Fatal(err)
		}
		if buffer == nil || buffer.Count() != 1 {
			b.Fatalf("token buffer = %#v, want one token", buffer)
		}
	}
}

func BenchmarkHIPGemma4Q4SharedKVSourceByLayer_Cached(b *testing.B) {
	const layerCount = 32
	layers := make([]hipGemma4Q4Layer0Config, layerCount)
	for index := range layers {
		if index%6 == 5 {
			layers[index].LayerType = "full_attention"
			layers[index].HeadDim = 256
		} else {
			layers[index].LayerType = "sliding_attention"
			layers[index].HeadDim = 256
		}
	}
	cfg := hipGemma4Q4ForwardConfig{
		Layers:         layers,
		KVSharedLayers: hipGemma4Q4DefaultKVSharedLayers(layerCount),
	}
	cfg.SharedKVSources = hipGemma4Q4BuildSharedKVSourceByLayer(cfg)
	if got := len(hipGemma4Q4SharedKVSourceByLayer(cfg)); got != layerCount {
		b.Fatalf("shared KV source count = %d, want %d", got, layerCount)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sources := hipGemma4Q4SharedKVSourceByLayer(cfg)
		if len(sources) != layerCount || sources[layerCount-1] < 0 {
			b.Fatalf("shared KV sources = %#v", sources)
		}
	}
}

func BenchmarkHIPGemma4Q4DecoderLayerRequest_NextInputNormValue(b *testing.B) {
	req := hipGemma4Q4DecoderLayerRequest{
		NextInputNormValue: hipRMSNormDeviceWeightConfig{
			WeightPointer:  0x1000,
			WeightBytes:    4608,
			Count:          2304,
			WeightEncoding: hipRMSNormWeightEncodingBF16,
			Flags:          hipRMSNormLaunchFlagAddUnitWeight,
			Epsilon:        1e-6,
		},
		HasNextInputNorm: true,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		cfg, ok := req.nextInputNormConfig()
		if !ok || cfg.Count != 2304 || cfg.WeightPointer == 0 {
			b.Fatalf("next input norm = %#v, %v", cfg, ok)
		}
	}
}

func BenchmarkHIPGemma4Q4DeviceLayerKVStateValueHandoff(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	cache := &rocmDeviceKVCache{
		driver:     driver,
		mode:       rocmKVCacheModeKQ8VQ4,
		blockSize:  1,
		tokenCount: 1,
		pages: []rocmDeviceKVPage{{
			tokenStart: 0,
			tokenCount: 1,
			keyWidth:   4,
			valueWidth: 4,
			key:        rocmDeviceKVTensor{pointer: 0x1000, sizeBytes: 4, encoding: rocmKVEncodingQ8},
			value:      rocmDeviceKVTensor{pointer: 0x2000, sizeBytes: 2, encoding: rocmKVEncodingQ4},
			owned:      true,
		}},
	}
	table := &rocmDeviceKVDescriptorTable{
		driver:    driver,
		pointer:   0x3000,
		sizeBytes: rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVDescriptorPageBytes,
		version:   rocmDeviceKVDescriptorVersion,
		pageCount: 1,
	}
	next := &hipGemma4Q4DeviceDecodeState{layers: make([]hipGemma4Q4DeviceLayerKVState, 0, 1)}
	result := hipGemma4Q4DecoderLayerResult{
		DeviceLayer: hipGemma4Q4DeviceLayerKVState{
			cache:                   cache,
			descriptorTable:         table,
			borrowedCache:           true,
			borrowedDescriptorTable: true,
		},
		DeviceLayerValid: true,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		next.layers = next.layers[:0]
		if !result.DeviceLayerValid {
			b.Fatal("device layer was not returned")
		}
		next.layers = append(next.layers, result.DeviceLayer)
		if len(next.layers) != 1 || next.layers[0].cache != cache {
			b.Fatalf("handoff layers = %#v", next.layers)
		}
	}
}

func BenchmarkHIPGemma4Q4DeviceLayerStatePool_Reused(b *testing.B) {
	layers := hipBorrowGemma4Q4DeviceLayerStates(32)
	hipReleaseGemma4Q4DeviceLayerStates(layers)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		layers = hipBorrowGemma4Q4DeviceLayerStates(32)
		if len(layers) != 0 || cap(layers) < 32 {
			b.Fatalf("layer state slice len/cap = %d/%d, want 0/>=32", len(layers), cap(layers))
		}
		hipReleaseGemma4Q4DeviceLayerStates(layers)
	}
}

func BenchmarkHIPGemma4Q4DeviceLayerKVStateClose_Borrowed(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	cache := &rocmDeviceKVCache{
		driver:     driver,
		mode:       rocmKVCacheModeKQ8VQ4,
		blockSize:  1,
		tokenCount: 1,
		pages: []rocmDeviceKVPage{{
			tokenStart: 0,
			tokenCount: 1,
			keyWidth:   4,
			valueWidth: 4,
			key:        rocmDeviceKVTensor{pointer: 0x1000, sizeBytes: 4, encoding: rocmKVEncodingQ8},
			value:      rocmDeviceKVTensor{pointer: 0x2000, sizeBytes: 2, encoding: rocmKVEncodingQ4},
			owned:      true,
		}},
	}
	table := &rocmDeviceKVDescriptorTable{
		driver:    driver,
		pointer:   0x3000,
		sizeBytes: rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVDescriptorPageBytes,
		version:   rocmDeviceKVDescriptorVersion,
		pageCount: 1,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		layer := hipGemma4Q4DeviceLayerKVState{
			cache:                   cache,
			descriptorTable:         table,
			borrowedCache:           true,
			borrowedDescriptorTable: true,
		}
		if err := layer.Close(); err != nil {
			b.Fatal(err)
		}
		if cache.closed || table.closed {
			b.Fatal("borrowed layer close closed source owner")
		}
	}
}

func BenchmarkROCmDeviceKVCacheBorrowRelease_Hot(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	cache := rocmBorrowDeviceKVCache(driver, rocmKVCacheModeKQ8VQ4, 1, 0, nil, false)
	rocmReleaseDeviceKVCache(cache)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		cache = rocmBorrowDeviceKVCache(driver, rocmKVCacheModeKQ8VQ4, 1, 128, nil, false)
		if cache.driver != driver || cache.mode != rocmKVCacheModeKQ8VQ4 || cache.TokenCount() != 128 {
			b.Fatalf("cache = %#v", cache)
		}
		rocmReleaseDeviceKVCache(cache)
	}
}

func BenchmarkHIPMLXQ4DeviceWeightConfigValidateInputCount_Hot(b *testing.B) {
	cfg := hipMLXQ4DeviceWeightConfig{
		WeightPointer: 0x1000,
		ScalePointer:  0x2000,
		BiasPointer:   0x3000,
		WeightBytes:   2304 * 288 * 4,
		ScaleBytes:    2304 * 36 * 2,
		BiasBytes:     2304 * 36 * 2,
		Rows:          2304,
		Cols:          2304,
		GroupSize:     64,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := cfg.validateInputCount(2304); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkROCmModelEmitTokenProbe_NoSink(b *testing.B) {
	model := &rocmModel{}
	token := inference.Token{ID: 42, Text: "hello"}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		model.emitTokenProbe(token, 2, i+1)
	}
}

func BenchmarkHIPMLXQ4TripleProjLaunchArgsBinary_Hot(b *testing.B) {
	args := hipMLXQ4TripleProjLaunchArgs{
		InputPointer:        0x1000,
		OutputPointer:       0x2000,
		FirstWeightPointer:  0x3000,
		FirstScalePointer:   0x4000,
		FirstBiasPointer:    0x5000,
		SecondWeightPointer: 0x6000,
		SecondScalePointer:  0x7000,
		SecondBiasPointer:   0x8000,
		ThirdWeightPointer:  0x9000,
		ThirdScalePointer:   0xa000,
		ThirdBiasPointer:    0xb000,
		FirstRows:           16,
		SecondRows:          4,
		ThirdRows:           4,
		Cols:                16,
		GroupSize:           8,
		Bits:                hipMLXQ4ProjectionBits,
		InputBytes:          64,
		OutputBytes:         96,
		FirstWeightBytes:    128,
		FirstScaleBytes:     64,
		FirstBiasBytes:      64,
		SecondWeightBytes:   32,
		SecondScaleBytes:    16,
		SecondBiasBytes:     16,
		ThirdWeightBytes:    32,
		ThirdScaleBytes:     16,
		ThirdBiasBytes:      16,
	}
	packet, err := args.Binary()
	if err != nil {
		b.Fatal(err)
	}
	hipReleaseLaunchPacket(packet)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.Binary()
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipMLXQ4TripleProjLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4TripleProjLaunchArgsBytes)
		}
		hipReleaseLaunchPacket(packet)
	}
}

func BenchmarkHIPPackedTopKLaunchArgsBinary_Hot(b *testing.B) {
	inputCount := 256000
	topK := 64
	chunkCount := (inputCount + hipPackedTopKChunkSize - 1) / hipPackedTopKChunkSize
	outputCount := chunkCount * topK
	args := hipPackedTopKLaunchArgs{
		InputPointer:  0x1000,
		OutputPointer: 0x2000,
		InputCount:    inputCount,
		OutputCount:   outputCount,
		TopK:          topK,
		ChunkSize:     hipPackedTopKChunkSize,
		InputBytes:    uint64(inputCount * hipMLXQ4ProjectionBestBytes),
		OutputBytes:   uint64(outputCount * hipMLXQ4ProjectionBestBytes),
	}
	packet, err := args.Binary()
	if err != nil {
		b.Fatal(err)
	}
	hipReleaseLaunchPacket(packet)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.Binary()
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipPackedTopKLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipPackedTopKLaunchArgsBytes)
		}
		hipReleaseLaunchPacket(packet)
	}
}

func BenchmarkHIPMLXQ4GELUTanhMultiplyLaunchArgsBinary_Hot(b *testing.B) {
	args := hipMLXQ4GELUTanhMulLaunchArgs{
		InputPointer:      0x1000,
		GateWeightPointer: 0x2000,
		GateScalePointer:  0x3000,
		GateBiasPointer:   0x4000,
		UpWeightPointer:   0x5000,
		UpScalePointer:    0x6000,
		UpBiasPointer:     0x7000,
		OutputPointer:     0x8000,
		Rows:              32,
		Cols:              16,
		GroupSize:         8,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        64,
		GateWeightBytes:   256,
		GateScaleBytes:    128,
		GateBiasBytes:     128,
		UpWeightBytes:     256,
		UpScaleBytes:      128,
		UpBiasBytes:       128,
		OutputBytes:       128,
	}
	packet, err := args.Binary()
	if err != nil {
		b.Fatal(err)
	}
	hipReleaseLaunchPacket(packet)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.Binary()
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipMLXQ4GELUTanhMulLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4GELUTanhMulLaunchArgsBytes)
		}
		hipReleaseLaunchPacket(packet)
	}
}

func BenchmarkROCmDeviceKVPageSlicePool_ReusedCapacity(b *testing.B) {
	rocmDeviceKVPageSlicePools.Range(func(key, _ any) bool {
		rocmDeviceKVPageSlicePools.Delete(key)
		return true
	})
	pages := rocmDeviceKVBorrowPageSlice(32, rocmDeviceKVHotPageCapacity)
	rocmDeviceKVReleasePageSlice(pages)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pages = rocmDeviceKVBorrowPageSlice(32, rocmDeviceKVHotPageCapacity)
		if len(pages) != 32 || cap(pages) != rocmDeviceKVHotPageCapacity {
			b.Fatalf("page slice len/cap = %d/%d, want 32/%d", len(pages), cap(pages), rocmDeviceKVHotPageCapacity)
		}
		rocmDeviceKVReleasePageSlice(pages)
	}
}

func BenchmarkROCmDeviceKVTransferSharedPages_TrimmedSuffix(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	sourcePages := make([]rocmDeviceKVPage, rocmDeviceKVHotPageCapacity+1, rocmDeviceKVPagePoolMaxCapacity+1)
	targetPages := make([]rocmDeviceKVPage, rocmDeviceKVHotPageCapacity, rocmDeviceKVPagePoolMaxCapacity+1)
	for index := range sourcePages {
		pointerBase := nativeDevicePointer(0x100000 + index*0x100)
		sourcePages[index] = rocmDeviceKVPage{
			tokenStart: index,
			tokenCount: 1,
			keyWidth:   256,
			valueWidth: 256,
			key:        rocmDeviceKVTensor{pointer: pointerBase + 1, sizeBytes: 260, encoding: rocmKVEncodingQ8},
			value:      rocmDeviceKVTensor{pointer: pointerBase + 2, sizeBytes: 132, encoding: rocmKVEncodingQ4},
		}
	}
	for index := range targetPages {
		targetPages[index] = sourcePages[index+1]
		targetPages[index].tokenStart = index
	}
	var source rocmDeviceKVCache
	var target rocmDeviceKVCache
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		source = rocmDeviceKVCache{driver: driver, mode: rocmKVCacheModeKQ8VQ4, blockSize: 1, tokenCount: len(sourcePages), pages: sourcePages}
		target = rocmDeviceKVCache{driver: driver, mode: rocmKVCacheModeKQ8VQ4, blockSize: 1, tokenCount: len(targetPages), pages: targetPages}
		if err := source.transferSharedPagesTo(&target); err != nil {
			b.Fatal(err)
		}
	}
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
