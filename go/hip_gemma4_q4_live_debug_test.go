// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"os"
	"testing"

	"dappco.re/go/inference"
)

func TestDebugLiveGemma412BQATConfig(t *testing.T) {
	if os.Getenv("GO_ROCM_DEBUG_LIVE_12B_CONFIG") != "1" {
		t.Skip("set GO_ROCM_DEBUG_LIVE_12B_CONFIG=1 to inspect a local Gemma4 12B QAT pack")
	}
	path := os.Getenv("GO_ROCM_PRODUCTION_MODEL_PATH")
	if path == "" {
		path = os.Getenv("GO_ROCM_MODEL_PATH")
	}
	if path == "" {
		t.Skip("set GO_ROCM_PRODUCTION_MODEL_PATH or GO_ROCM_MODEL_PATH")
	}
	model, err := newROCmBackendWithRuntime(newSystemNativeRuntime()).LoadModel(path, inference.WithContextLen(4096))
	if err != nil {
		t.Fatalf("LoadModel(%q): %v", path, err)
	}
	defer model.Close()
	rocmLoaded, ok := model.(*rocmModel)
	if !ok {
		t.Fatalf("model type = %T, want *rocmModel", model)
	}
	loaded, ok := rocmLoaded.native.(*hipLoadedModel)
	if !ok {
		t.Fatalf("native model type = %T, want *hipLoadedModel", rocmLoaded.native)
	}
	cfg, ok, err := loaded.loadedGemma4Q4PackageForwardConfig()
	if err != nil {
		t.Fatalf("loadedGemma4Q4PackageForwardConfig: %v", err)
	}
	if !ok {
		t.Fatalf("Gemma4 q4 package config unavailable")
	}
	t.Logf("modelInfo=%+v context=%d tied=%v labels=%v", loaded.modelInfo, loaded.contextSize, loaded.modelLabels["tied_word_embeddings"], loaded.modelLabels)
	t.Logf("textConfig hidden=%d layers=%d headDim=%d globalHeadDim=%d sliding=%d attentionKEqV=%v finalSoftcap=%g kvShared=%d/%v rope=%+v",
		loaded.modelInfo.HiddenSize,
		loaded.gemma4TextConfig.NumLayers,
		loaded.gemma4TextConfig.HeadDim,
		loaded.gemma4TextConfig.GlobalHeadDim,
		loaded.gemma4TextConfig.SlidingWindow,
		loaded.gemma4TextConfig.AttentionKEqV,
		loaded.gemma4TextConfig.FinalLogitSoftcap,
		cfg.KVSharedLayers,
		cfg.SharedKVSources,
		loaded.gemma4TextConfig.RoPEParameters)
	for _, index := range []int{0, 1, 4, 5, 11, 47} {
		if index < 0 || index >= len(cfg.Layers) {
			continue
		}
		layer := cfg.Layers[index]
		t.Logf("layer[%d] type=%s hidden=%d heads q=%d kv=%d headDim=%d kvDim=%d ropeBase=%g rotary=%d freqScale=%g window=%d kEqV=%v scalar=%g finalSoftcap=%g",
			index,
			layer.LayerType,
			layer.HiddenSize,
			layer.QueryHeads,
			layer.KeyHeads,
			layer.HeadDim,
			layer.keyValueDim(),
			layer.RoPEBase,
			layer.RoPERotaryDim,
			layer.RoPEFrequencyScale,
			layer.SlidingWindow,
			layer.AttentionKEqV,
			layer.LayerScalar,
			layer.FinalLogitSoftcap)
		t.Logf("layer[%d] bits embed=%d q=%d k=%d v=%d o=%d gate=%d up=%d down=%d lm=%d rows q/k/v/o=%d/%d/%d/%d mlp=%d/%d/%d lm=%d",
			index,
			layer.Embedding.QuantBits,
			layer.QueryProjection.Bits,
			layer.KeyProjection.Bits,
			layer.ValueProjection.Bits,
			layer.OutputProjection.Bits,
			layer.GateProjection.Bits,
			layer.UpProjection.Bits,
			layer.DownProjection.Bits,
			layer.LMHeadProjection.Bits,
			layer.QueryProjection.Rows,
			layer.KeyProjection.Rows,
			layer.ValueProjection.Rows,
			layer.OutputProjection.Rows,
			layer.GateProjection.Rows,
			layer.UpProjection.Rows,
			layer.DownProjection.Rows,
			layer.LMHeadProjection.Rows)
	}
	layer0 := cfg.Layers[0]
	input := make([]float32, layer0.HiddenSize)
	secondInput := make([]float32, layer0.HiddenSize)
	for index := range input {
		input[index] = float32((index%17)-8) * 0.03125
		secondInput[index] = float32((index%13)-6) * 0.046875
	}
	proj, err := hipRunMLXQ4ProjectionKernelWithDeviceWeightConfig(context.Background(), loaded.driver, input, layer0.GateProjection)
	if err != nil {
		t.Fatalf("layer0 gate q%d projection: %v", layer0.GateProjection.Bits, err)
	}
	assertDebugLiveProjectionPrefix(t, loaded, "language_model.model.layers.0.mlp.gate_proj", "layer0 gate q8 single", layer0.GateProjection, input, proj)

	batchInput := append(append([]float32(nil), input...), secondInput...)
	batchPayload, err := hipFloat32Payload(batchInput)
	if err != nil {
		t.Fatalf("batch payload: %v", err)
	}
	batchInputBuffer, err := hipUploadByteBuffer(loaded.driver, "rocm.hip.MLXQ4ProjectionBatchLaunch", "debug layer0 gate q8 batch input", batchPayload, len(batchInput))
	if err != nil {
		t.Fatalf("upload batch input: %v", err)
	}
	defer batchInputBuffer.Close()
	batchOutputBuffer, err := hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(context.Background(), loaded.driver, batchInputBuffer, layer0.GateProjection, 2)
	if err != nil {
		t.Fatalf("layer0 gate q%d batch projection: %v", layer0.GateProjection.Bits, err)
	}
	defer batchOutputBuffer.Close()
	batchProj, err := hipReadFloat32DeviceOutput(batchOutputBuffer, "rocm.hip.MLXQ4ProjectionBatchLaunch", "debug layer0 gate q8 batch output", layer0.GateProjection.Rows*2)
	if err != nil {
		t.Fatalf("read batch projection: %v", err)
	}
	assertDebugLiveProjectionPrefix(t, loaded, "language_model.model.layers.0.mlp.gate_proj", "layer0 gate q8 batch first", layer0.GateProjection, input, batchProj[:layer0.GateProjection.Rows])
	assertDebugLiveProjectionPrefix(t, loaded, "language_model.model.layers.0.mlp.gate_proj", "layer0 gate q8 batch second", layer0.GateProjection, secondInput, batchProj[layer0.GateProjection.Rows:])

	debugForward, _, err := hipRunGemma4Q4SingleTokenForwardWithState(context.Background(), loaded.driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:  2,
		Position: 0,
		Epsilon:  1e-6,
	})
	if err != nil {
		t.Fatalf("debug single-token forward: %v", err)
	}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	optimizedForward, _, err := hipRunGemma4Q4SingleTokenForwardWithState(context.Background(), loaded.driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:            2,
		Position:           0,
		Epsilon:            1e-6,
		DeviceFinalSample:  true,
		OmitDebugTensors:   true,
		AttentionWorkspace: workspace,
	})
	if err != nil {
		t.Fatalf("optimized single-token forward: %v", err)
	}
	if debugForward.Greedy.TokenID != optimizedForward.Greedy.TokenID {
		t.Fatalf("optimized greedy = %+v, debug = %+v", optimizedForward.Greedy, debugForward.Greedy)
	}
	t.Logf("single-token greedy debug=%+v optimized=%+v", debugForward.Greedy, optimizedForward.Greedy)

	suppressTokens := hipGemma4Q4GenerationSuppressTokenIDs(loaded, nil)
	for _, id := range suppressTokens {
		t.Logf("suppress token id=%d text=%q", id, hipGeneratedTokenText(loaded, id))
	}
	prompt := formatGemma4ChatTemplateWithConfig([]inference.Message{{Role: "user", Content: "Write one short sentence about apples."}}, loaded.gemma4ChatTemplateConfig(inference.GenerateConfig{}, false))
	promptTokens, ok, err := hipGemma4Q4TextPromptIDs("text:"+prompt, loaded)
	if err != nil || !ok {
		t.Fatalf("prompt tokenization ok=%v err=%v", ok, err)
	}
	t.Logf("prompt=%q tokens=%v", prompt, promptTokens)
	mismatches := 0
	for _, prefixLen := range []int{1, len(promptTokens)} {
		if prefixLen > len(promptTokens) {
			continue
		}
		prefix := promptTokens[:prefixLen]
		host := debugLiveHostPromptGreedy(t, loaded, cfg, prefix, suppressTokens)
		stepwiseFP16 := debugLiveStepwisePromptGreedyMode(t, loaded, cfg, prefix, suppressTokens, rocmKVCacheModeFP16)
		stepwiseKQ8VQ4 := debugLiveStepwisePromptGreedyMode(t, loaded, cfg, prefix, suppressTokens, rocmKVCacheModeKQ8VQ4)
		batchedKQ8VQ4 := debugLiveBatchedPromptGreedyMode(t, loaded, cfg, prefix, suppressTokens, rocmKVCacheModeKQ8VQ4)
		t.Logf("prompt prefix=%d final greedy host=%+v text=%q stepwise_fp16=%+v text=%q stepwise_kq8vq4=%+v text=%q batched_kq8vq4=%+v text=%q",
			prefixLen,
			host, hipGeneratedTokenText(loaded, int32(host.TokenID)),
			stepwiseFP16, hipGeneratedTokenText(loaded, int32(stepwiseFP16.TokenID)),
			stepwiseKQ8VQ4, hipGeneratedTokenText(loaded, int32(stepwiseKQ8VQ4.TokenID)),
			batchedKQ8VQ4, hipGeneratedTokenText(loaded, int32(batchedKQ8VQ4.TokenID)))
		if host.TokenID != stepwiseFP16.TokenID || host.TokenID != stepwiseKQ8VQ4.TokenID || host.TokenID != batchedKQ8VQ4.TokenID {
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Fatalf("prompt final greedy mismatches=%d", mismatches)
	}
}

func debugLiveStepwisePromptGreedy(t *testing.T, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, tokens []int32, suppressTokens []int32) hipGreedySampleResult {
	return debugLiveStepwisePromptGreedyMode(t, model, cfg, tokens, suppressTokens, rocmKVCacheModeKQ8VQ4)
}

func debugLiveHostPromptGreedy(t *testing.T, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, tokens []int32, suppressTokens []int32) hipGreedySampleResult {
	t.Helper()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(model.driver, workspace, cfg, len(tokens)+2); err != nil {
		t.Fatalf("host workspace decode capacity: %v", err)
	}
	workspace.EnsureProjectionGreedyBestCapacity(2)
	best, err := workspace.BorrowProjectionGreedyBest(model.driver)
	if err != nil {
		t.Fatalf("host greedy best: %v", err)
	}
	state := hipGemma4Q4DecodeState{}
	current := hipGemma4Q4ForwardResult{}
	for index, tokenID := range tokens {
		outputToken := index == len(tokens)-1
		current, state, err = hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), model.driver, cfg, state, hipGemma4Q4ForwardRequest{
			TokenID:            tokenID,
			Position:           index,
			Epsilon:            1e-6,
			DeviceFinalSample:  outputToken,
			SkipFinalSample:    !outputToken,
			FinalGreedyBuffer:  best,
			SuppressTokens:     suppressTokens,
			AttentionWorkspace: workspace,
			OmitDebugTensors:   true,
			OmitLabels:         true,
		}, false)
		if err != nil {
			t.Fatalf("host prompt token %d forward: %v", index, err)
		}
	}
	return current.Greedy
}

func debugLiveStepwisePromptGreedyMode(t *testing.T, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, tokens []int32, suppressTokens []int32, mode string) hipGreedySampleResult {
	t.Helper()
	engineConfig := defaultHIPGemma4Q4EngineConfig()
	engineConfig.DeviceKVMode = mode
	mode, err := engineConfig.deviceKVMode()
	if err != nil {
		t.Fatalf("device KV mode: %v", err)
	}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(model.driver, workspace, cfg, len(tokens)+2); err != nil {
		t.Fatalf("stepwise workspace decode capacity: %v", err)
	}
	workspace.EnsureProjectionGreedyBestCapacity(2)
	best, err := workspace.BorrowProjectionGreedyBest(model.driver)
	if err != nil {
		t.Fatalf("stepwise greedy best: %v", err)
	}
	state := hipGemma4Q4DecodeState{}
	var deviceState *hipGemma4Q4DeviceDecodeState
	defer func() {
		if deviceState != nil {
			_ = deviceState.Close()
		}
	}()
	current := hipGemma4Q4ForwardResult{}
	for index, tokenID := range tokens {
		outputToken := index == len(tokens)-1
		current, state, err = hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), model.driver, cfg, state, hipGemma4Q4ForwardRequest{
			TokenID:            tokenID,
			Position:           index,
			Epsilon:            1e-6,
			DeviceKVAttention:  true,
			DeviceKVMode:       mode,
			EngineConfig:       engineConfig,
			PriorDeviceState:   deviceState,
			ReturnDeviceState:  true,
			DeviceFinalSample:  outputToken,
			SkipFinalSample:    !outputToken,
			FinalGreedyBuffer:  best,
			SuppressTokens:     suppressTokens,
			AttentionWorkspace: workspace,
			OmitDebugTensors:   true,
			OmitLabels:         true,
			OmitHostState:      true,
		}, false)
		if err != nil {
			t.Fatalf("stepwise prompt token %d forward: %v", index, err)
		}
		if current.DeviceState == nil {
			t.Fatalf("stepwise prompt token %d did not return device state", index)
		}
		previousDeviceState := deviceState
		deviceState = current.DeviceState
		current.DeviceState = nil
		hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
	}
	return current.Greedy
}

func debugLiveBatchedPromptGreedy(t *testing.T, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, tokens []int32, suppressTokens []int32) hipGreedySampleResult {
	return debugLiveBatchedPromptGreedyMode(t, model, cfg, tokens, suppressTokens, rocmKVCacheModeKQ8VQ4)
}

func debugLiveBatchedPromptGreedyMode(t *testing.T, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, tokens []int32, suppressTokens []int32, mode string) hipGreedySampleResult {
	t.Helper()
	engineConfig := defaultHIPGemma4Q4EngineConfig()
	engineConfig.DeviceKVMode = mode
	mode, err := engineConfig.deviceKVMode()
	if err != nil {
		t.Fatalf("device KV mode: %v", err)
	}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	planBatches := hipBorrowGemma4Q4PrefillUBatches(1)
	defer hipReleaseGemma4Q4PrefillUBatches(planBatches)
	plan, planBatches, err := hipGemma4Q4PlanPromptPrefillInto(tokens, 0, len(tokens), planBatches)
	if err != nil {
		t.Fatalf("batched prefill plan: %v", err)
	}
	if err := hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(model.driver, workspace, cfg, len(tokens)+2); err != nil {
		t.Fatalf("batched workspace decode capacity: %v", err)
	}
	if err := hipGemma4Q4EnsureAttentionWorkspacePrefillCapacity(model.driver, workspace, cfg, plan, true); err != nil {
		t.Fatalf("batched workspace prefill capacity: %v", err)
	}
	workspace.EnsureProjectionGreedyBestCapacity(2)
	best, err := workspace.BorrowProjectionGreedyBest(model.driver)
	if err != nil {
		t.Fatalf("batched greedy best: %v", err)
	}
	forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(context.Background(), model.driver, cfg, tokens, 0, 1e-6, mode, nil, nil, nil, nil, len(tokens)-1, best, workspace, engineConfig)
	if err != nil {
		t.Fatalf("batched prefill forward: %v", err)
	}
	defer forward.Close()
	if len(forward.Greedy) == 0 {
		t.Fatalf("batched prefill did not return greedy")
	}
	greedy := forward.Greedy[len(forward.Greedy)-1].Greedy
	if hipTokenIsSuppressed(int32(greedy.TokenID), suppressTokens) {
		last := cfg.Layers[len(cfg.Layers)-1]
		greedy, err = hipRunGemma4Q4PrefillFinalGreedyForRowSuppressWorkspace(context.Background(), model.driver, last, forward.FinalHidden, len(tokens), len(tokens)-1, 1e-6, best, suppressTokens, workspace)
		if err != nil {
			t.Fatalf("batched suppress rerun: %v", err)
		}
	}
	return greedy
}

func assertDebugLiveProjectionPrefix(t *testing.T, model *hipLoadedModel, baseName, label string, cfg hipMLXQ4DeviceWeightConfig, input, got []float32) {
	t.Helper()
	const compareRows = 8
	weight, err := model.requiredHIPTensor(baseName+".weight", label+" weight")
	if err != nil {
		t.Fatalf("%s: %v", label, err)
	}
	scales, err := model.requiredHIPTensor(baseName+".scales", label+" scales")
	if err != nil {
		t.Fatalf("%s: %v", label, err)
	}
	biases, err := model.requiredHIPTensor(baseName+".biases", label+" biases")
	if err != nil {
		t.Fatalf("%s: %v", label, err)
	}
	packedPerRow, err := hipMLXAffinePackedCols(cfg.Cols, cfg.Bits)
	if err != nil {
		t.Fatalf("%s packed cols: %v", label, err)
	}
	groups := cfg.Cols / cfg.GroupSize
	wantWeights := readLoadedUint32TensorRows(t, weight, compareRows, packedPerRow)
	wantScales := readLoadedBF16TensorRows(t, scales, compareRows, groups)
	wantBiases := readLoadedBF16TensorRows(t, biases, compareRows, groups)
	want, err := hipReferenceMLXAffineProjection(input, wantWeights, wantScales, wantBiases, compareRows, cfg.Cols, cfg.GroupSize, cfg.Bits)
	if err != nil {
		t.Fatalf("%s reference: %v", label, err)
	}
	if len(got) < compareRows {
		t.Fatalf("%s output length = %d, want at least %d", label, len(got), compareRows)
	}
	assertFloat32SlicesNear(t, want, got[:compareRows], 0.05)
}
