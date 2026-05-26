// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"math"

	core "dappco.re/go"
)

type hipSmallDecodeRequest struct {
	Architecture string
	Input        []float32
	RMSWeight    []float32
	Epsilon      float32
	QueryFP16    []uint16
	KeyFP16      []uint16
	ValueFP16    []uint16
	OutputFP16   []uint16
	LMHeadFP16   []uint16
	PriorKeys    []float32
	PriorValues  []float32
	Position     int
	RoPEBase     float32
	VocabSize    int
	HiddenSize   int
}

type hipSmallDecodeResult struct {
	Logits        []float32
	Attention     []float32
	UpdatedKeys   []float32
	UpdatedValues []float32
	Projected     []float32
	TokenID       int
	Score         float32
	Labels        map[string]string
}

type hipLoadedSmallDecodeConfig struct {
	Architecture        string
	EmbeddingPointer    nativeDevicePointer
	EmbeddingBytes      uint64
	RMSWeightPointer    nativeDevicePointer
	RMSWeightBytes      uint64
	QueryWeightPointer  nativeDevicePointer
	QueryWeightBytes    uint64
	KeyWeightPointer    nativeDevicePointer
	KeyWeightBytes      uint64
	ValueWeightPointer  nativeDevicePointer
	ValueWeightBytes    uint64
	OutputWeightPointer nativeDevicePointer
	OutputWeightBytes   uint64
	LMHeadPointer       nativeDevicePointer
	LMHeadBytes         uint64
	VocabSize           int
	HiddenSize          int
}

type hipLoadedSmallDecodeRequest struct {
	Input       []float32
	PriorKeys   []float32
	PriorValues []float32
	Position    int
	RoPEBase    float32
	Epsilon     float32
}

type hipRMSNormDeviceWeightConfig struct {
	WeightPointer  nativeDevicePointer
	WeightBytes    uint64
	Count          int
	Epsilon        float32
	WeightEncoding uint32
	Flags          uint32
}

func (req hipSmallDecodeRequest) validate() error {
	architecture := normalizeROCmArchitecture(req.Architecture)
	switch architecture {
	case "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "gemma4":
	default:
		return core.E("rocm.hip.SmallDecode", "small decode smoke supports only Qwen or Gemma architectures", nil)
	}
	if req.HiddenSize <= 0 || req.HiddenSize%2 != 0 || req.VocabSize <= 0 {
		return core.E("rocm.hip.SmallDecode", "hidden size must be positive and even and vocab size must be positive", nil)
	}
	if len(req.Input) != req.HiddenSize {
		return core.E("rocm.hip.SmallDecode", "input length must match hidden size", nil)
	}
	if len(req.RMSWeight) != req.HiddenSize {
		return core.E("rocm.hip.SmallDecode", "RMS weight length must match hidden size", nil)
	}
	if req.Epsilon < 0 || math.IsNaN(float64(req.Epsilon)) || math.IsInf(float64(req.Epsilon), 0) {
		return core.E("rocm.hip.SmallDecode", "epsilon must be non-negative and finite", nil)
	}
	if req.Position < 0 {
		return core.E("rocm.hip.SmallDecode", "position must be non-negative", nil)
	}
	if req.RoPEBase <= 0 || math.IsNaN(float64(req.RoPEBase)) || math.IsInf(float64(req.RoPEBase), 0) {
		return core.E("rocm.hip.SmallDecode", "RoPE base must be positive and finite", nil)
	}
	projectionWeights := req.HiddenSize * req.HiddenSize
	for name, weights := range map[string][]uint16{
		"query":  req.QueryFP16,
		"key":    req.KeyFP16,
		"value":  req.ValueFP16,
		"output": req.OutputFP16,
	} {
		if len(weights) != projectionWeights {
			return core.E("rocm.hip.SmallDecode", name+" projection weight length must match hidden*hidden", nil)
		}
	}
	if len(req.LMHeadFP16) != req.VocabSize*req.HiddenSize {
		return core.E("rocm.hip.SmallDecode", "LM head weight length must match vocab*hidden", nil)
	}
	if len(req.PriorKeys) == 0 || len(req.PriorValues) == 0 {
		return core.E("rocm.hip.SmallDecode", "prior key/value tensors are required", nil)
	}
	if len(req.PriorKeys) != len(req.PriorValues) || len(req.PriorKeys)%req.HiddenSize != 0 {
		return core.E("rocm.hip.SmallDecode", "prior key/value tensors must align with hidden size", nil)
	}
	if req.Position != len(req.PriorKeys)/req.HiddenSize {
		return core.E("rocm.hip.SmallDecode", "decode position must equal prior KV token count", nil)
	}
	return nil
}

func (model *hipLoadedModel) loadedSmallDecodeConfig() (hipLoadedSmallDecodeConfig, error) {
	if model == nil {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "loaded model is required", nil)
	}
	if model.driver == nil || !model.driver.Available() {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "HIP driver is not available", nil)
	}
	architecture := normalizeROCmArchitecture(model.modelInfo.Architecture)
	switch architecture {
	case "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "gemma4", "gemma4_text":
	default:
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "small decode smoke supports only Qwen or Gemma architectures", nil)
	}
	hiddenSize := model.modelInfo.HiddenSize
	vocabSize := model.modelInfo.VocabSize
	if hiddenSize <= 0 || hiddenSize%2 != 0 || vocabSize <= 0 {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "model hidden size must be positive and even and vocab size must be positive", nil)
	}
	embedding, ok := model.findHIPTensor(isHIPEmbeddingTensor)
	if !ok {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "embedding tensor is required", nil)
	}
	rms, ok := model.findHIPTensor(hipSmallDecodeRMSWeightTensor)
	if !ok {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "input RMSNorm weight tensor is required", nil)
	}
	query, ok := model.findHIPTensor(hipSmallDecodeProjectionTensor("q_proj"))
	if !ok {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "query projection tensor is required", nil)
	}
	key, ok := model.findHIPTensor(hipSmallDecodeProjectionTensor("k_proj"))
	if !ok {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "key projection tensor is required", nil)
	}
	value, ok := model.findHIPTensor(hipSmallDecodeProjectionTensor("v_proj"))
	if !ok {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "value projection tensor is required", nil)
	}
	output, ok := model.findHIPTensor(hipSmallDecodeProjectionTensor("o_proj"))
	if !ok {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "output projection tensor is required", nil)
	}
	lmHead, ok := model.findHIPTensor(isHIPOutputTensor)
	if !ok {
		return hipLoadedSmallDecodeConfig{}, core.E("rocm.hip.SmallDecode", "LM head tensor is required", nil)
	}
	if err := hipLoadedSmallDecodeRMSWeight(rms, hiddenSize); err != nil {
		return hipLoadedSmallDecodeConfig{}, err
	}
	if err := hipLoadedSmallDecodeEmbedding(embedding, vocabSize, hiddenSize); err != nil {
		return hipLoadedSmallDecodeConfig{}, err
	}
	for label, tensor := range map[string]hipTensor{
		"query":  query,
		"key":    key,
		"value":  value,
		"output": output,
	} {
		if err := hipLoadedSmallDecodeFP16Matrix(label, tensor, hiddenSize, hiddenSize); err != nil {
			return hipLoadedSmallDecodeConfig{}, err
		}
	}
	if err := hipLoadedSmallDecodeFP16Matrix("LM head", lmHead, vocabSize, hiddenSize); err != nil {
		return hipLoadedSmallDecodeConfig{}, err
	}
	return hipLoadedSmallDecodeConfig{
		Architecture:        architecture,
		EmbeddingPointer:    embedding.pointer,
		EmbeddingBytes:      embedding.info.ByteSize,
		RMSWeightPointer:    rms.pointer,
		RMSWeightBytes:      rms.info.ByteSize,
		QueryWeightPointer:  query.pointer,
		QueryWeightBytes:    query.info.ByteSize,
		KeyWeightPointer:    key.pointer,
		KeyWeightBytes:      key.info.ByteSize,
		ValueWeightPointer:  value.pointer,
		ValueWeightBytes:    value.info.ByteSize,
		OutputWeightPointer: output.pointer,
		OutputWeightBytes:   output.info.ByteSize,
		LMHeadPointer:       lmHead.pointer,
		LMHeadBytes:         lmHead.info.ByteSize,
		VocabSize:           vocabSize,
		HiddenSize:          hiddenSize,
	}, nil
}

func hipLoadedSmallDecodeEmbedding(tensor hipTensor, vocabSize, hiddenSize int) error {
	if tensor.pointer == 0 {
		return core.E("rocm.hip.SmallDecode", "embedding pointer is required", nil)
	}
	if !hipTinyTensorIsFP32(tensor.info) {
		return core.E("rocm.hip.SmallDecode", "embedding tensor must be f32", nil)
	}
	if len(tensor.info.Dimensions) != 2 || tensor.info.Dimensions[0] != uint64(vocabSize) || tensor.info.Dimensions[1] != uint64(hiddenSize) {
		return core.E("rocm.hip.SmallDecode", "embedding tensor shape must be vocab-major vocab*hidden", nil)
	}
	if _, err := hipExactUint32Bytes("embedding", tensor.info.ByteSize, uint64(vocabSize)*uint64(hiddenSize)*4); err != nil {
		return core.E("rocm.hip.SmallDecode", "embedding byte count", err)
	}
	return nil
}

func hipSmallDecodeRMSWeightTensor(name string) bool {
	return core.Contains(name, "layers.0") &&
		core.Contains(name, "weight") &&
		(core.Contains(name, "input_layernorm") || core.Contains(name, "attention_norm"))
}

func hipSmallDecodeProjectionTensor(kind string) func(string) bool {
	return func(name string) bool {
		return core.Contains(name, "layers.0") &&
			core.Contains(name, kind) &&
			core.Contains(name, "weight")
	}
}

func hipLoadedSmallDecodeRMSWeight(tensor hipTensor, hiddenSize int) error {
	if tensor.pointer == 0 {
		return core.E("rocm.hip.SmallDecode", "RMSNorm weight pointer is required", nil)
	}
	if !hipTinyTensorIsFP32(tensor.info) {
		return core.E("rocm.hip.SmallDecode", "RMSNorm weight must be f32", nil)
	}
	if len(tensor.info.Dimensions) != 1 || tensor.info.Dimensions[0] != uint64(hiddenSize) {
		return core.E("rocm.hip.SmallDecode", "RMSNorm weight shape must match hidden size", nil)
	}
	if _, err := hipExactUint32Bytes("RMSNorm weight", tensor.info.ByteSize, uint64(hiddenSize)*4); err != nil {
		return core.E("rocm.hip.SmallDecode", "RMSNorm weight byte count", err)
	}
	return nil
}

func hipLoadedSmallDecodeFP16Matrix(label string, tensor hipTensor, rows, cols int) error {
	if tensor.pointer == 0 {
		return core.E("rocm.hip.SmallDecode", label+" weight pointer is required", nil)
	}
	if !hipTinyTensorIsFP16(tensor.info) {
		return core.E("rocm.hip.SmallDecode", label+" weight must be f16", nil)
	}
	if len(tensor.info.Dimensions) != 2 || tensor.info.Dimensions[0] != uint64(rows) || tensor.info.Dimensions[1] != uint64(cols) {
		return core.E("rocm.hip.SmallDecode", label+" weight shape must be row-major rows*cols", nil)
	}
	if _, err := hipExactUint32Bytes(label+" weight", tensor.info.ByteSize, uint64(rows)*uint64(cols)*2); err != nil {
		return core.E("rocm.hip.SmallDecode", label+" weight byte count", err)
	}
	return nil
}

func hipReferenceSmallDecode(req hipSmallDecodeRequest) (hipSmallDecodeResult, error) {
	if err := req.validate(); err != nil {
		return hipSmallDecodeResult{}, err
	}
	normalized, err := hipReferenceRMSNorm(req.Input, req.RMSWeight, req.Epsilon)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	query, err := hipReferenceFP16Projection(normalized, req.QueryFP16, req.HiddenSize, req.HiddenSize, nil)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	key, err := hipReferenceFP16Projection(normalized, req.KeyFP16, req.HiddenSize, req.HiddenSize, nil)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	value, err := hipReferenceFP16Projection(normalized, req.ValueFP16, req.HiddenSize, req.HiddenSize, nil)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	ropeQuery, err := hipReferenceRoPE(query, req.Position, float64(req.RoPEBase))
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	ropeKey, err := hipReferenceRoPE(key, req.Position, float64(req.RoPEBase))
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	priorKeys, err := splitHIPReferenceVectors(req.PriorKeys, req.HiddenSize)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	priorValues, err := splitHIPReferenceVectors(req.PriorValues, req.HiddenSize)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	attentionOutput, attention, updatedKeys, updatedValues, err := hipReferenceDecodeWithKV(ropeQuery, ropeKey, value, priorKeys, priorValues)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	projected, err := hipReferenceFP16Projection(attentionOutput, req.OutputFP16, req.HiddenSize, req.HiddenSize, nil)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	logits, err := hipReferenceFP16Projection(projected, req.LMHeadFP16, req.VocabSize, req.HiddenSize, nil)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	tokenID, score, err := hipReferenceGreedySample(logits)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	return hipSmallDecodeResult{
		Logits:        logits,
		Attention:     attention,
		UpdatedKeys:   flattenHIPReferenceMatrix(updatedKeys),
		UpdatedValues: flattenHIPReferenceMatrix(updatedValues),
		Projected:     projected,
		TokenID:       tokenID,
		Score:         score,
		Labels:        hipSmallDecodeLabels(req),
	}, nil
}

func hipRunLoadedSmallDecode(ctx context.Context, driver nativeHIPDriver, cfg hipLoadedSmallDecodeConfig, req hipLoadedSmallDecodeRequest) (hipSmallDecodeResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipSmallDecodeResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipSmallDecodeResult{}, core.E("rocm.hip.SmallDecode", "HIP driver is not available", nil)
	}
	if err := cfg.validate(); err != nil {
		return hipSmallDecodeResult{}, err
	}
	if err := req.validate(cfg); err != nil {
		return hipSmallDecodeResult{}, err
	}
	normalized, err := hipRunRMSNormKernelWithDeviceWeight(ctx, driver, req.Input, cfg.RMSWeightPointer, cfg.RMSWeightBytes, cfg.HiddenSize, req.Epsilon)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	query, err := hipRunProjectionKernelWithDeviceWeight(ctx, driver, normalized, cfg.QueryWeightPointer, cfg.QueryWeightBytes, cfg.HiddenSize, cfg.HiddenSize)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	key, err := hipRunProjectionKernelWithDeviceWeight(ctx, driver, normalized, cfg.KeyWeightPointer, cfg.KeyWeightBytes, cfg.HiddenSize, cfg.HiddenSize)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	value, err := hipRunProjectionKernelWithDeviceWeight(ctx, driver, normalized, cfg.ValueWeightPointer, cfg.ValueWeightBytes, cfg.HiddenSize, cfg.HiddenSize)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	ropeQuery, err := hipRunRoPEKernel(ctx, driver, hipRoPERequest{Input: query, Position: req.Position, Base: req.RoPEBase})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	ropeKey, err := hipRunRoPEKernel(ctx, driver, hipRoPERequest{Input: key, Position: req.Position, Base: req.RoPEBase})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	updatedKeys := append(append([]float32(nil), req.PriorKeys...), ropeKey...)
	updatedValues := append(append([]float32(nil), req.PriorValues...), value...)
	attention, err := hipRunAttentionKernel(ctx, driver, hipAttentionRequest{Query: ropeQuery, Keys: updatedKeys, Values: updatedValues})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	projected, err := hipRunProjectionKernelWithDeviceWeight(ctx, driver, attention.Output, cfg.OutputWeightPointer, cfg.OutputWeightBytes, cfg.HiddenSize, cfg.HiddenSize)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	logits, err := hipRunProjectionKernelWithDeviceWeight(ctx, driver, projected, cfg.LMHeadPointer, cfg.LMHeadBytes, cfg.VocabSize, cfg.HiddenSize)
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	greedy, err := hipRunGreedyKernel(ctx, driver, hipGreedySampleRequest{Logits: logits})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	return hipSmallDecodeResult{
		Logits:        logits,
		Attention:     attention.Weights,
		UpdatedKeys:   updatedKeys,
		UpdatedValues: updatedValues,
		Projected:     projected,
		TokenID:       greedy.TokenID,
		Score:         greedy.Score,
		Labels:        hipLoadedSmallDecodeLabels(cfg, req),
	}, nil
}

func hipRunLoadedSmallDecodeToken(ctx context.Context, model *hipLoadedModel, cfg hipLoadedSmallDecodeConfig, req hipDecodeRequest) (hipDecodeResult, error) {
	if model == nil {
		return hipDecodeResult{}, core.E("rocm.hip.SmallDecode", "loaded model is required", nil)
	}
	if err := req.validate(); err != nil {
		return hipDecodeResult{}, err
	}
	if int(req.TokenID) >= cfg.VocabSize {
		return hipDecodeResult{}, core.E("rocm.hip.SmallDecode", "token ID is outside vocabulary", nil)
	}
	keyWidth, valueWidth, err := req.kvVectorWidths()
	if err != nil {
		return hipDecodeResult{}, err
	}
	if keyWidth != cfg.HiddenSize || valueWidth != cfg.HiddenSize {
		return hipDecodeResult{}, core.E("rocm.hip.SmallDecode", "KV widths must match hidden size", nil)
	}
	priorKeys, priorValues, err := req.KV.Restore(0, req.KV.TokenCount())
	if err != nil {
		return hipDecodeResult{}, err
	}
	input, err := hipReadLoadedSmallEmbedding(ctx, model.driver, cfg, req.TokenID)
	if err != nil {
		return hipDecodeResult{}, err
	}
	output, err := hipRunLoadedSmallDecode(ctx, model.driver, cfg, hipLoadedSmallDecodeRequest{
		Input:       input,
		PriorKeys:   priorKeys,
		PriorValues: priorValues,
		Position:    req.KV.TokenCount(),
		RoPEBase:    10000,
		Epsilon:     0,
	})
	if err != nil {
		return hipDecodeResult{}, err
	}
	if model.smallLoRA != nil {
		logits, tokenID, score, err := model.runSmallLoRAProjection(ctx, cfg, output.Projected)
		if err != nil {
			return hipDecodeResult{}, err
		}
		output.Logits = logits
		output.TokenID = tokenID
		output.Score = score
		model.addSmallLoRALabels(output.Labels)
	}
	targetKV := req.KV
	if req.DeviceKV != nil {
		cloned, err := req.KV.Clone()
		if err != nil {
			return hipDecodeResult{}, err
		}
		targetKV = cloned
	}
	keyStart := len(output.UpdatedKeys) - cfg.HiddenSize
	valueStart := len(output.UpdatedValues) - cfg.HiddenSize
	if err := targetKV.AppendToken(targetKV.TokenCount(), output.UpdatedKeys[keyStart:], output.UpdatedValues[valueStart:]); err != nil {
		return hipDecodeResult{}, err
	}
	labels := output.Labels
	labels["decode_launch_token"] = core.Sprintf("%d", req.TokenID)
	var deviceKV *rocmDeviceKVCache
	var descriptorTable *rocmDeviceKVDescriptorTable
	if req.DeviceKV != nil {
		device, table, err := hipAppendDecodeDeviceKV(req, output.UpdatedKeys[keyStart:], output.UpdatedValues[valueStart:], labels)
		if err != nil {
			return hipDecodeResult{}, err
		}
		deviceKV = device
		descriptorTable = table
	}
	return hipDecodeResult{
		Token:           hipTinyToken(model, int32(output.TokenID)),
		Logits:          output.Logits,
		KV:              targetKV,
		DeviceKV:        deviceKV,
		DescriptorTable: descriptorTable,
		Labels:          labels,
	}, nil
}

func hipReadLoadedSmallEmbedding(ctx context.Context, driver nativeHIPDriver, cfg hipLoadedSmallDecodeConfig, tokenID int32) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.SmallDecode", "HIP driver is not available", nil)
	}
	if tokenID < 0 || int(tokenID) >= cfg.VocabSize {
		return nil, core.E("rocm.hip.SmallDecode", "token ID is outside vocabulary", nil)
	}
	rowBytes := uint64(cfg.HiddenSize * 4)
	offset := uint64(tokenID) * rowBytes
	if offset+rowBytes > cfg.EmbeddingBytes {
		return nil, core.E("rocm.hip.SmallDecode", "embedding row exceeds tensor byte size", nil)
	}
	payload := make([]byte, rowBytes)
	pointer := nativeDevicePointer(uintptr(cfg.EmbeddingPointer) + uintptr(offset))
	if err := driver.CopyDeviceToHost(pointer, payload); err != nil {
		return nil, core.E("rocm.hip.SmallDecode", "copy embedding row", err)
	}
	values, err := hipFloat32PayloadValues(payload)
	if err != nil {
		return nil, err
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E("rocm.hip.SmallDecode", "embedding row values must be finite", nil)
	}
	return values, nil
}

func (cfg hipLoadedSmallDecodeConfig) validate() error {
	switch normalizeROCmArchitecture(cfg.Architecture) {
	case "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "gemma4":
	default:
		return core.E("rocm.hip.SmallDecode", "small decode smoke supports only Qwen or Gemma architectures", nil)
	}
	if cfg.HiddenSize <= 0 || cfg.HiddenSize%2 != 0 || cfg.VocabSize <= 0 {
		return core.E("rocm.hip.SmallDecode", "hidden size must be positive and even and vocab size must be positive", nil)
	}
	if cfg.EmbeddingPointer == 0 || cfg.RMSWeightPointer == 0 || cfg.QueryWeightPointer == 0 || cfg.KeyWeightPointer == 0 ||
		cfg.ValueWeightPointer == 0 || cfg.OutputWeightPointer == 0 || cfg.LMHeadPointer == 0 {
		return core.E("rocm.hip.SmallDecode", "loaded weight pointers are required", nil)
	}
	if _, err := hipExactUint32Bytes("embedding", cfg.EmbeddingBytes, uint64(cfg.VocabSize)*uint64(cfg.HiddenSize)*4); err != nil {
		return core.E("rocm.hip.SmallDecode", "embedding byte count", err)
	}
	if _, err := hipExactUint32Bytes("RMSNorm weight", cfg.RMSWeightBytes, uint64(cfg.HiddenSize)*4); err != nil {
		return core.E("rocm.hip.SmallDecode", "RMSNorm weight byte count", err)
	}
	for label, bytes := range map[string]uint64{
		"query":  cfg.QueryWeightBytes,
		"key":    cfg.KeyWeightBytes,
		"value":  cfg.ValueWeightBytes,
		"output": cfg.OutputWeightBytes,
	} {
		if _, err := hipExactUint32Bytes(label+" weight", bytes, uint64(cfg.HiddenSize)*uint64(cfg.HiddenSize)*2); err != nil {
			return core.E("rocm.hip.SmallDecode", label+" weight byte count", err)
		}
	}
	if _, err := hipExactUint32Bytes("LM head weight", cfg.LMHeadBytes, uint64(cfg.VocabSize)*uint64(cfg.HiddenSize)*2); err != nil {
		return core.E("rocm.hip.SmallDecode", "LM head weight byte count", err)
	}
	return nil
}

func (req hipLoadedSmallDecodeRequest) validate(cfg hipLoadedSmallDecodeConfig) error {
	if len(req.Input) != cfg.HiddenSize {
		return core.E("rocm.hip.SmallDecode", "input length must match hidden size", nil)
	}
	if req.Epsilon < 0 || math.IsNaN(float64(req.Epsilon)) || math.IsInf(float64(req.Epsilon), 0) {
		return core.E("rocm.hip.SmallDecode", "epsilon must be non-negative and finite", nil)
	}
	if req.Position < 0 {
		return core.E("rocm.hip.SmallDecode", "position must be non-negative", nil)
	}
	if req.RoPEBase <= 0 || math.IsNaN(float64(req.RoPEBase)) || math.IsInf(float64(req.RoPEBase), 0) {
		return core.E("rocm.hip.SmallDecode", "RoPE base must be positive and finite", nil)
	}
	if len(req.PriorKeys) == 0 || len(req.PriorValues) == 0 {
		return core.E("rocm.hip.SmallDecode", "prior key/value tensors are required", nil)
	}
	if len(req.PriorKeys) != len(req.PriorValues) || len(req.PriorKeys)%cfg.HiddenSize != 0 {
		return core.E("rocm.hip.SmallDecode", "prior key/value tensors must align with hidden size", nil)
	}
	if req.Position != len(req.PriorKeys)/cfg.HiddenSize {
		return core.E("rocm.hip.SmallDecode", "decode position must equal prior KV token count", nil)
	}
	return nil
}

func hipRunSmallDecode(ctx context.Context, driver nativeHIPDriver, req hipSmallDecodeRequest) (hipSmallDecodeResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipSmallDecodeResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipSmallDecodeResult{}, core.E("rocm.hip.SmallDecode", "HIP driver is not available", nil)
	}
	if err := req.validate(); err != nil {
		return hipSmallDecodeResult{}, err
	}
	normalized, err := hipRunRMSNormKernel(ctx, driver, hipRMSNormRequest{Input: req.Input, Weight: req.RMSWeight, Epsilon: req.Epsilon})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	query, err := hipRunProjectionKernel(ctx, driver, hipProjectionRequest{Input: normalized, FP16: req.QueryFP16, Rows: req.HiddenSize, Cols: req.HiddenSize})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	key, err := hipRunProjectionKernel(ctx, driver, hipProjectionRequest{Input: normalized, FP16: req.KeyFP16, Rows: req.HiddenSize, Cols: req.HiddenSize})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	value, err := hipRunProjectionKernel(ctx, driver, hipProjectionRequest{Input: normalized, FP16: req.ValueFP16, Rows: req.HiddenSize, Cols: req.HiddenSize})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	ropeQuery, err := hipRunRoPEKernel(ctx, driver, hipRoPERequest{Input: query, Position: req.Position, Base: req.RoPEBase})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	ropeKey, err := hipRunRoPEKernel(ctx, driver, hipRoPERequest{Input: key, Position: req.Position, Base: req.RoPEBase})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	updatedKeys := append(append([]float32(nil), req.PriorKeys...), ropeKey...)
	updatedValues := append(append([]float32(nil), req.PriorValues...), value...)
	attention, err := hipRunAttentionKernel(ctx, driver, hipAttentionRequest{Query: ropeQuery, Keys: updatedKeys, Values: updatedValues})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	projected, err := hipRunProjectionKernel(ctx, driver, hipProjectionRequest{Input: attention.Output, FP16: req.OutputFP16, Rows: req.HiddenSize, Cols: req.HiddenSize})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	logits, err := hipRunProjectionKernel(ctx, driver, hipProjectionRequest{Input: projected, FP16: req.LMHeadFP16, Rows: req.VocabSize, Cols: req.HiddenSize})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	greedy, err := hipRunGreedyKernel(ctx, driver, hipGreedySampleRequest{Logits: logits})
	if err != nil {
		return hipSmallDecodeResult{}, err
	}
	return hipSmallDecodeResult{
		Logits:        logits,
		Attention:     attention.Weights,
		UpdatedKeys:   updatedKeys,
		UpdatedValues: updatedValues,
		Projected:     projected,
		TokenID:       greedy.TokenID,
		Score:         greedy.Score,
		Labels:        hipSmallDecodeLabels(req),
	}, nil
}

func hipSmallDecodeLabels(req hipSmallDecodeRequest) map[string]string {
	return map[string]string{
		"decode_kernel":       hipKernelStatusLinked,
		"decode_kernel_name":  "rocm_small_decode_smoke",
		"decode_architecture": normalizeROCmArchitecture(req.Architecture),
		"decode_position":     core.Sprintf("%d", req.Position),
		"decode_vocab_size":   core.Sprintf("%d", req.VocabSize),
		"decode_hidden_size":  core.Sprintf("%d", req.HiddenSize),
		"decode_primitives":   "rms_norm,projection,rope,attention,greedy",
	}
}

func hipLoadedSmallDecodeLabels(cfg hipLoadedSmallDecodeConfig, req hipLoadedSmallDecodeRequest) map[string]string {
	labels := map[string]string{
		"decode_tensor_backing": "loaded_device",
		"decode_position":       core.Sprintf("%d", req.Position),
		"decode_vocab_size":     core.Sprintf("%d", cfg.VocabSize),
		"decode_hidden_size":    core.Sprintf("%d", cfg.HiddenSize),
	}
	for key, value := range hipSmallDecodeLabels(hipSmallDecodeRequest{Architecture: cfg.Architecture, Position: req.Position, VocabSize: cfg.VocabSize, HiddenSize: cfg.HiddenSize}) {
		labels[key] = value
	}
	return labels
}

func hipRunProjectionKernel(ctx context.Context, driver nativeHIPDriver, req hipProjectionRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.projectionDeviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.projectionLaunchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameProjection, launchBytes, req.Rows)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipRunProjectionKernelWithDeviceWeight(ctx context.Context, driver nativeHIPDriver, input []float32, weightPointer nativeDevicePointer, weightBytes uint64, rows, cols int) ([]float32, error) {
	return hipRunProjectionKernelWithDeviceWeightEncoding(ctx, driver, input, weightPointer, weightBytes, rows, cols, hipProjectionWeightEncodingFP16)
}

func hipRunProjectionKernelWithDeviceWeightEncoding(ctx context.Context, driver nativeHIPDriver, input []float32, weightPointer nativeDevicePointer, weightBytes uint64, rows, cols int, encoding uint32) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	weightElements, err := hipProjectionDeviceWeightElementCount(weightBytes, encoding)
	if err != nil {
		return nil, err
	}
	if err := validateHIPProjectionShape(len(input), weightElements, 0, rows, cols); err != nil {
		return nil, err
	}
	inputPayload, err := hipFloat32Payload(input)
	if err != nil {
		return nil, core.E("rocm.hip.ProjectionLaunch", "encode input", err)
	}
	inputBuffer, err := hipUploadByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection input", inputPayload, len(input))
	if err != nil {
		return nil, err
	}
	defer inputBuffer.Close()
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection output", uint64(rows*4), rows)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	launchBytes, err := (hipProjectionLaunchArgs{
		InputPointer:   inputBuffer.Pointer(),
		InputCount:     len(input),
		InputBytes:     inputBuffer.SizeBytes(),
		WeightPointer:  weightPointer,
		WeightBytes:    weightBytes,
		OutputPointer:  output.Pointer(),
		OutputBytes:    output.SizeBytes(),
		Rows:           rows,
		Cols:           cols,
		WeightEncoding: encoding,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameProjection, launchBytes, rows)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	return hipReadFloat32DeviceOutput(output, "rocm.hip.ProjectionLaunch", "projection output", rows)
}

func hipRunProjectionKernelWithDeviceInputWeightEncoding(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, weightPointer nativeDevicePointer, weightBytes uint64, rows, cols int, encoding uint32) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "projection device input is required", nil)
	}
	weightElements, err := hipProjectionDeviceWeightElementCount(weightBytes, encoding)
	if err != nil {
		return nil, err
	}
	if err := validateHIPProjectionShape(input.Count(), weightElements, 0, rows, cols); err != nil {
		return nil, err
	}
	if input.SizeBytes() != uint64(cols*4) {
		return nil, core.E("rocm.hip.ProjectionLaunch", "projection device input byte count mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection output", uint64(rows*4), rows)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipProjectionLaunchArgs{
		InputPointer:   input.Pointer(),
		InputCount:     input.Count(),
		InputBytes:     input.SizeBytes(),
		WeightPointer:  weightPointer,
		WeightBytes:    weightBytes,
		OutputPointer:  output.Pointer(),
		OutputBytes:    output.SizeBytes(),
		Rows:           rows,
		Cols:           cols,
		WeightEncoding: encoding,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameProjection, launchBytes, rows)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunProjectionBatchKernelWithDeviceInputWeightEncoding(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, weightPointer nativeDevicePointer, weightBytes uint64, rows, cols int, encoding uint32, batch int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "projection batch device input is required", nil)
	}
	if batch <= 0 {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "projection batch size must be positive", nil)
	}
	weightElements, err := hipProjectionDeviceWeightElementCount(weightBytes, encoding)
	if err != nil {
		return nil, err
	}
	if err := validateHIPProjectionShape(cols, weightElements, 0, rows, cols); err != nil {
		return nil, err
	}
	if input.Count() != cols*batch || input.SizeBytes() != uint64(cols*batch*4) {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "projection batch device input shape mismatch", nil)
	}
	outputCount := rows * batch
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.ProjectionBatchLaunch", "projection batch output", uint64(outputCount*4), outputCount)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipProjectionBatchLaunchArgs{
		InputPointer:   input.Pointer(),
		WeightPointer:  weightPointer,
		WeightBytes:    weightBytes,
		OutputPointer:  output.Pointer(),
		InputBytes:     input.SizeBytes(),
		OutputBytes:    output.SizeBytes(),
		Rows:           rows,
		Cols:           cols,
		Batch:          batch,
		WeightEncoding: encoding,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipProjectionBatchLaunchConfig(launchBytes, rows, batch)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipProjectionDeviceWeightElementCount(weightBytes uint64, encoding uint32) (int, error) {
	if weightBytes == 0 {
		return 0, core.E("rocm.hip.ProjectionLaunch", "projection weight bytes are required", nil)
	}
	var bytesPerElement uint64
	switch encoding {
	case hipProjectionWeightEncodingFP16, hipProjectionWeightEncodingBF16:
		bytesPerElement = 2
	case hipProjectionWeightEncodingQ8:
		bytesPerElement = 1
	case hipProjectionWeightEncodingF32:
		bytesPerElement = 4
	default:
		return 0, core.E("rocm.hip.ProjectionLaunch", core.Sprintf("unsupported projection weight encoding %d", encoding), nil)
	}
	if weightBytes%bytesPerElement != 0 {
		return 0, core.E("rocm.hip.ProjectionLaunch", "projection weight byte count must be element-aligned", nil)
	}
	elements := weightBytes / bytesPerElement
	if elements > uint64(int(^uint(0)>>1)) {
		return 0, core.E("rocm.hip.ProjectionLaunch", "projection weight element count is out of int range", nil)
	}
	return int(elements), nil
}

func hipRunRMSNormKernel(ctx context.Context, driver nativeHIPDriver, req hipRMSNormRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipSingleBlockLaunchConfig(hipKernelNameRMSNorm, launchBytes, 256)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipRunRMSNormKernelWithDeviceWeight(ctx context.Context, driver nativeHIPDriver, input []float32, weightPointer nativeDevicePointer, weightBytes uint64, count int, epsilon float32) ([]float32, error) {
	return hipRunRMSNormKernelWithDeviceWeightConfig(ctx, driver, input, hipRMSNormDeviceWeightConfig{
		WeightPointer:  weightPointer,
		WeightBytes:    weightBytes,
		Count:          count,
		Epsilon:        epsilon,
		WeightEncoding: hipRMSNormWeightEncodingF32,
	})
}

func hipRunRMSNormKernelWithDeviceWeightConfig(ctx context.Context, driver nativeHIPDriver, input []float32, cfg hipRMSNormDeviceWeightConfig) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.RMSNormLaunch", "HIP driver is not available", nil)
	}
	if cfg.WeightEncoding == hipRMSNormWeightEncodingNone {
		if cfg.Flags != 0 {
			return nil, core.E("rocm.hip.RMSNormLaunch", "unit RMSNorm weight does not support flags", nil)
		}
		if cfg.WeightPointer != 0 || cfg.WeightBytes != 0 {
			return nil, core.E("rocm.hip.RMSNormLaunch", "unit RMSNorm weight must not provide a weight pointer", nil)
		}
	} else if cfg.WeightPointer == 0 {
		return nil, core.E("rocm.hip.RMSNormLaunch", "RMSNorm weight pointer is required", nil)
	}
	if cfg.Count <= 0 {
		return nil, core.E("rocm.hip.RMSNormLaunch", "count must be positive", nil)
	}
	if len(input) != cfg.Count {
		return nil, core.E("rocm.hip.RMSNormLaunch", "input length must match count", nil)
	}
	inputPayload, err := hipFloat32Payload(input)
	if err != nil {
		return nil, core.E("rocm.hip.RMSNormLaunch", "encode input", err)
	}
	inputBuffer, err := hipUploadByteBuffer(driver, "rocm.hip.RMSNormLaunch", "rms norm input", inputPayload, len(input))
	if err != nil {
		return nil, err
	}
	defer inputBuffer.Close()
	output, err := hipRunRMSNormKernelWithDeviceInputWeightConfig(ctx, driver, inputBuffer, cfg)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	return hipReadFloat32DeviceOutput(output, "rocm.hip.RMSNormLaunch", "rms norm output", cfg.Count)
}

func hipRunRMSNormKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.RMSNormLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.RMSNormLaunch", "RMSNorm input device buffer is required", nil)
	}
	if cfg.WeightEncoding == hipRMSNormWeightEncodingNone {
		if cfg.Flags != 0 {
			return nil, core.E("rocm.hip.RMSNormLaunch", "unit RMSNorm weight does not support flags", nil)
		}
		if cfg.WeightPointer != 0 || cfg.WeightBytes != 0 {
			return nil, core.E("rocm.hip.RMSNormLaunch", "unit RMSNorm weight must not provide a weight pointer", nil)
		}
	} else if cfg.WeightPointer == 0 {
		return nil, core.E("rocm.hip.RMSNormLaunch", "RMSNorm weight pointer is required", nil)
	}
	if cfg.Count <= 0 {
		return nil, core.E("rocm.hip.RMSNormLaunch", "count must be positive", nil)
	}
	if input.Count() != cfg.Count || input.SizeBytes() != uint64(cfg.Count*4) {
		return nil, core.E("rocm.hip.RMSNormLaunch", "RMSNorm input device buffer shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormLaunch", "rms norm output", uint64(cfg.Count*4), cfg.Count)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipRMSNormLaunchArgs{
		InputPointer:   input.Pointer(),
		WeightPointer:  cfg.WeightPointer,
		OutputPointer:  output.Pointer(),
		Count:          cfg.Count,
		InputBytes:     input.SizeBytes(),
		WeightBytes:    cfg.WeightBytes,
		OutputBytes:    output.SizeBytes(),
		Epsilon:        cfg.Epsilon,
		WeightEncoding: cfg.WeightEncoding,
		Flags:          cfg.Flags,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipSingleBlockLaunchConfig(hipKernelNameRMSNorm, launchBytes, 256)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunRMSNormResidualAddKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig) (*hipDeviceByteBuffer, error) {
	return hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfig(ctx, driver, input, residual, cfg, 1)
}

func hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, outputScale float32) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "RMSNorm input device buffer is required", nil)
	}
	if residual == nil || residual.Pointer() == 0 {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "residual device buffer is required", nil)
	}
	if cfg.WeightEncoding == hipRMSNormWeightEncodingNone {
		if cfg.Flags != 0 {
			return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "unit RMSNorm weight does not support flags", nil)
		}
		if cfg.WeightPointer != 0 || cfg.WeightBytes != 0 {
			return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "unit RMSNorm weight must not provide a weight pointer", nil)
		}
	} else if cfg.WeightPointer == 0 {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "RMSNorm weight pointer is required", nil)
	}
	if cfg.Count <= 0 {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "count must be positive", nil)
	}
	if math.IsNaN(float64(outputScale)) || math.IsInf(float64(outputScale), 0) {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "output scale must be finite", nil)
	}
	if input.Count() != cfg.Count || residual.Count() != cfg.Count || input.SizeBytes() != uint64(cfg.Count*4) || residual.SizeBytes() != uint64(cfg.Count*4) {
		return nil, core.E("rocm.hip.RMSNormResidualAddLaunch", "RMSNorm residual-add device buffer shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormResidualAddLaunch", "rms norm residual-add output", uint64(cfg.Count*4), cfg.Count)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipRMSNormResidualAddLaunchArgs{
		InputPointer:    input.Pointer(),
		WeightPointer:   cfg.WeightPointer,
		ResidualPointer: residual.Pointer(),
		OutputPointer:   output.Pointer(),
		Count:           cfg.Count,
		InputBytes:      input.SizeBytes(),
		WeightBytes:     cfg.WeightBytes,
		ResidualBytes:   residual.SizeBytes(),
		OutputBytes:     output.SizeBytes(),
		Epsilon:         cfg.Epsilon,
		WeightEncoding:  cfg.WeightEncoding,
		Flags:           cfg.Flags,
		OutputScale:     outputScale,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipSingleBlockLaunchConfig(hipKernelNameRMSNormResidualAdd, launchBytes, 256)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunRMSNormResidualAddNormKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, residualCfg, normCfg hipRMSNormDeviceWeightConfig) (*hipDeviceByteBuffer, *hipDeviceByteBuffer, error) {
	return hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfig(ctx, driver, input, residual, residualCfg, normCfg, 1)
}

func hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, residualCfg, normCfg hipRMSNormDeviceWeightConfig, outputScale float32) (*hipDeviceByteBuffer, *hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "RMSNorm input device buffer is required", nil)
	}
	if residual == nil || residual.Pointer() == 0 {
		return nil, nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "residual device buffer is required", nil)
	}
	if err := hipValidateRMSNormDeviceWeightConfig("RMSNormResidualAddNormLaunch", residualCfg); err != nil {
		return nil, nil, err
	}
	if err := hipValidateRMSNormDeviceWeightConfig("RMSNormResidualAddNormLaunch", normCfg); err != nil {
		return nil, nil, err
	}
	if residualCfg.Count <= 0 || residualCfg.Count != normCfg.Count {
		return nil, nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "RMSNorm counts must be positive and equal", nil)
	}
	if input.Count() != residualCfg.Count || residual.Count() != residualCfg.Count ||
		input.SizeBytes() != uint64(residualCfg.Count*4) ||
		residual.SizeBytes() != uint64(residualCfg.Count*4) {
		return nil, nil, core.E("rocm.hip.RMSNormResidualAddNormLaunch", "RMSNorm residual-add-norm device buffer shape mismatch", nil)
	}
	residualOutput, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormResidualAddNormLaunch", "rms norm residual-add output", uint64(residualCfg.Count*4), residualCfg.Count)
	if err != nil {
		return nil, nil, err
	}
	normOutput, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormResidualAddNormLaunch", "rms norm residual-add norm output", uint64(normCfg.Count*4), normCfg.Count)
	if err != nil {
		_ = residualOutput.Close()
		return nil, nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = normOutput.Close()
			_ = residualOutput.Close()
		}
	}()
	launchBytes, err := (hipRMSNormResidualAddNormLaunchArgs{
		InputPointer:          input.Pointer(),
		WeightPointer:         residualCfg.WeightPointer,
		ResidualPointer:       residual.Pointer(),
		ResidualOutputPointer: residualOutput.Pointer(),
		NormWeightPointer:     normCfg.WeightPointer,
		NormOutputPointer:     normOutput.Pointer(),
		Count:                 residualCfg.Count,
		InputBytes:            input.SizeBytes(),
		WeightBytes:           residualCfg.WeightBytes,
		ResidualBytes:         residual.SizeBytes(),
		ResidualOutputBytes:   residualOutput.SizeBytes(),
		NormWeightBytes:       normCfg.WeightBytes,
		NormOutputBytes:       normOutput.SizeBytes(),
		Epsilon:               residualCfg.Epsilon,
		WeightEncoding:        residualCfg.WeightEncoding,
		Flags:                 residualCfg.Flags,
		NormEpsilon:           normCfg.Epsilon,
		NormWeightEncoding:    normCfg.WeightEncoding,
		NormFlags:             normCfg.Flags,
		OutputScale:           outputScale,
	}).Binary()
	if err != nil {
		return nil, nil, err
	}
	config, err := hipSingleBlockLaunchConfig(hipKernelNameRMSNormResAddNorm, launchBytes, 256)
	if err != nil {
		return nil, nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, nil, err
	}
	success = true
	return residualOutput, normOutput, nil
}

func hipValidateRMSNormDeviceWeightConfig(operation string, cfg hipRMSNormDeviceWeightConfig) error {
	if cfg.WeightEncoding == hipRMSNormWeightEncodingNone {
		if cfg.Flags != 0 {
			return core.E("rocm.hip."+operation, "unit RMSNorm weight does not support flags", nil)
		}
		if cfg.WeightPointer != 0 || cfg.WeightBytes != 0 {
			return core.E("rocm.hip."+operation, "unit RMSNorm weight must not provide a weight pointer", nil)
		}
		return nil
	}
	if cfg.WeightPointer == 0 {
		return core.E("rocm.hip."+operation, "RMSNorm weight pointer is required", nil)
	}
	return nil
}

func hipRunRMSNormDeviceToDeviceKernel(ctx context.Context, driver nativeHIPDriver, inputPointer nativeDevicePointer, inputBytes uint64, outputPointer nativeDevicePointer, outputBytes uint64, cfg hipRMSNormDeviceWeightConfig) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if cfg.WeightEncoding == hipRMSNormWeightEncodingNone {
		if cfg.Flags != 0 {
			return core.E("rocm.hip.RMSNormLaunch", "unit RMSNorm weight does not support flags", nil)
		}
		if cfg.WeightPointer != 0 || cfg.WeightBytes != 0 {
			return core.E("rocm.hip.RMSNormLaunch", "unit RMSNorm weight must not provide a weight pointer", nil)
		}
	} else if cfg.WeightPointer == 0 {
		return core.E("rocm.hip.RMSNormLaunch", "RMSNorm weight pointer is required", nil)
	}
	launchBytes, err := (hipRMSNormLaunchArgs{
		InputPointer:   inputPointer,
		WeightPointer:  cfg.WeightPointer,
		OutputPointer:  outputPointer,
		Count:          cfg.Count,
		InputBytes:     inputBytes,
		WeightBytes:    cfg.WeightBytes,
		OutputBytes:    outputBytes,
		Epsilon:        cfg.Epsilon,
		WeightEncoding: cfg.WeightEncoding,
		Flags:          cfg.Flags,
	}).Binary()
	if err != nil {
		return err
	}
	config, err := hipSingleBlockLaunchConfig(hipKernelNameRMSNorm, launchBytes, 256)
	if err != nil {
		return err
	}
	return hipLaunchKernel(driver, config)
}

func hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, headCount int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "RMSNorm heads input device buffer is required", nil)
	}
	if cfg.Count <= 0 || headCount <= 0 {
		return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "head dim and head count must be positive", nil)
	}
	if input.Count() != cfg.Count*headCount || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E("rocm.hip.RMSNormHeadsLaunch", "RMSNorm heads input device buffer shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormHeadsLaunch", "rms norm heads output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipRMSNormHeadsLaunchArgs{
		InputPointer:   input.Pointer(),
		WeightPointer:  cfg.WeightPointer,
		OutputPointer:  output.Pointer(),
		HeadDim:        cfg.Count,
		HeadCount:      headCount,
		InputBytes:     input.SizeBytes(),
		WeightBytes:    cfg.WeightBytes,
		OutputBytes:    output.SizeBytes(),
		Epsilon:        cfg.Epsilon,
		WeightEncoding: cfg.WeightEncoding,
		Flags:          cfg.Flags,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameRMSNormHeads,
		Args:   launchBytes,
		GridX:  uint32(headCount),
		GridY:  1,
		GridZ:  1,
		BlockX: 256,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, headCount int, position int, base float32, frequencyDim int, rotaryCount int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.RMSNormRoPEHeadsLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.RMSNormRoPEHeadsLaunch", "RMSNorm RoPE heads input device buffer is required", nil)
	}
	if cfg.Count <= 0 || cfg.Count%2 != 0 || headCount <= 0 {
		return nil, core.E("rocm.hip.RMSNormRoPEHeadsLaunch", "head dim must be positive/even and head count must be positive", nil)
	}
	if input.Count() != cfg.Count*headCount || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E("rocm.hip.RMSNormRoPEHeadsLaunch", "RMSNorm RoPE heads input device buffer shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormRoPEHeadsLaunch", "rms norm rope heads output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipRMSNormRoPEHeadsLaunchArgs{
		InputPointer:   input.Pointer(),
		WeightPointer:  cfg.WeightPointer,
		OutputPointer:  output.Pointer(),
		HeadDim:        cfg.Count,
		HeadCount:      headCount,
		InputBytes:     input.SizeBytes(),
		WeightBytes:    cfg.WeightBytes,
		OutputBytes:    output.SizeBytes(),
		Epsilon:        cfg.Epsilon,
		WeightEncoding: cfg.WeightEncoding,
		Flags:          cfg.Flags,
		Position:       position,
		Base:           base,
		FrequencyDim:   frequencyDim,
		RotaryCount:    rotaryCount,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameRMSNormRoPEHeads,
		Args:   launchBytes,
		GridX:  uint32(headCount),
		GridY:  1,
		GridZ:  1,
		BlockX: 256,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfig(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, headCount int, batch int, startPosition int, base float32, frequencyDim int, rotaryCount int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.RMSNormRoPEHeadsBatchLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.RMSNormRoPEHeadsBatchLaunch", "RMSNorm RoPE heads batch input device buffer is required", nil)
	}
	if cfg.Count <= 0 || cfg.Count%2 != 0 || headCount <= 0 || batch <= 0 {
		return nil, core.E("rocm.hip.RMSNormRoPEHeadsBatchLaunch", "head dim must be positive/even and head count/batch must be positive", nil)
	}
	if input.Count() != cfg.Count*headCount*batch || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E("rocm.hip.RMSNormRoPEHeadsBatchLaunch", "RMSNorm RoPE heads batch input device buffer shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormRoPEHeadsBatchLaunch", "rms norm rope heads batch output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipRMSNormRoPEHeadsBatchLaunchArgs{
		InputPointer:   input.Pointer(),
		WeightPointer:  cfg.WeightPointer,
		OutputPointer:  output.Pointer(),
		HeadDim:        cfg.Count,
		HeadCount:      headCount,
		Batch:          batch,
		InputBytes:     input.SizeBytes(),
		WeightBytes:    cfg.WeightBytes,
		OutputBytes:    output.SizeBytes(),
		Epsilon:        cfg.Epsilon,
		WeightEncoding: cfg.WeightEncoding,
		Flags:          cfg.Flags,
		StartPosition:  startPosition,
		Base:           base,
		FrequencyDim:   frequencyDim,
		RotaryCount:    rotaryCount,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameRMSNormRoPEHeadsBatch,
		Args:   launchBytes,
		GridX:  uint32(headCount),
		GridY:  uint32(batch),
		GridZ:  1,
		BlockX: 256,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunRoPEKernel(ctx context.Context, driver nativeHIPDriver, req hipRoPERequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameRoPE, launchBytes, buffers.Count/2)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipRunRoPEDeviceKernel(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, position int, base float32, frequencyDim int) (*hipDeviceByteBuffer, error) {
	return hipRunRoPEDeviceKernelWithRotaryCount(ctx, driver, input, position, base, frequencyDim, 0)
}

func hipRunRoPEDeviceKernelWithRotaryCount(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, position int, base float32, frequencyDim int, rotaryCount int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.RoPELaunch", "rope device input is required", nil)
	}
	if input.Count() <= 0 || input.Count()%2 != 0 || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E("rocm.hip.RoPELaunch", "rope device input shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RoPELaunch", "rope output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunRoPEDeviceToDeviceKernelWithRotaryCount(ctx, driver, input.Pointer(), input.SizeBytes(), output.Pointer(), output.SizeBytes(), input.Count(), position, base, frequencyDim, rotaryCount); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunRoPEDeviceToDeviceKernel(ctx context.Context, driver nativeHIPDriver, inputPointer nativeDevicePointer, inputBytes uint64, outputPointer nativeDevicePointer, outputBytes uint64, count int, position int, base float32, frequencyDim int) error {
	return hipRunRoPEDeviceToDeviceKernelWithRotaryCount(ctx, driver, inputPointer, inputBytes, outputPointer, outputBytes, count, position, base, frequencyDim, 0)
}

func hipRunRoPEDeviceToDeviceKernelWithRotaryCount(ctx context.Context, driver nativeHIPDriver, inputPointer nativeDevicePointer, inputBytes uint64, outputPointer nativeDevicePointer, outputBytes uint64, count int, position int, base float32, frequencyDim int, rotaryCount int) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	launchBytes, err := (hipRoPELaunchArgs{
		InputPointer:  inputPointer,
		OutputPointer: outputPointer,
		Count:         count,
		InputBytes:    inputBytes,
		OutputBytes:   outputBytes,
		Position:      position,
		Base:          base,
		FrequencyDim:  frequencyDim,
		RotaryCount:   rotaryCount,
	}).Binary()
	if err != nil {
		return err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameRoPE, launchBytes, count/2)
	if err != nil {
		return err
	}
	return hipLaunchKernel(driver, config)
}

func hipRunRoPEHeadsDeviceKernelWithRotaryCount(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, headDim, headCount int, position int, base float32, frequencyDim int, rotaryCount int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "rope heads device input is required", nil)
	}
	if headDim <= 0 || headDim%2 != 0 || headCount <= 0 {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "head dim must be positive/even and head count must be positive", nil)
	}
	if input.Count() != headDim*headCount || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E("rocm.hip.RoPEHeadsLaunch", "rope heads device input shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RoPEHeadsLaunch", "rope heads output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipRoPEHeadsLaunchArgs{
		InputPointer:  input.Pointer(),
		OutputPointer: output.Pointer(),
		HeadDim:       headDim,
		HeadCount:     headCount,
		InputBytes:    input.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		Position:      position,
		Base:          base,
		FrequencyDim:  frequencyDim,
		RotaryCount:   rotaryCount,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameRoPEHeads, launchBytes, headCount*(headDim/2))
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunAttentionKernel(ctx context.Context, driver nativeHIPDriver, req hipAttentionRequest) (hipAttentionResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipAttentionResult{}, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return hipAttentionResult{}, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return hipAttentionResult{}, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return hipAttentionResult{}, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameAttention, launchBytes, buffers.TokenCount)
	if err != nil {
		return hipAttentionResult{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipAttentionResult{}, err
	}
	return buffers.ReadOutput()
}

func hipRunAttentionOutputKernel(ctx context.Context, driver nativeHIPDriver, req hipAttentionRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameAttention, launchBytes, buffers.TokenCount)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutputOnly()
}

func hipRunAttentionOutputToDeviceKernel(ctx context.Context, driver nativeHIPDriver, req hipAttentionRequest, output *hipDeviceByteBuffer, outputElementOffset int) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	queryPayload, err := hipFloat32Payload(req.Query)
	if err != nil {
		return core.E("rocm.hip.AttentionLaunch", "encode query", err)
	}
	query, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionLaunch", "attention query", queryPayload, len(req.Query))
	if err != nil {
		return err
	}
	defer query.Close()
	return hipRunAttentionOutputFromDeviceQueryToDeviceKernel(ctx, driver, req, query, 0, output, outputElementOffset)
}

func hipRunAttentionOutputFromDeviceQueryToDeviceKernel(ctx context.Context, driver nativeHIPDriver, req hipAttentionRequest, query *hipDeviceByteBuffer, queryElementOffset int, output *hipDeviceByteBuffer, outputElementOffset int) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	dim, tokenCount, err := req.shape()
	if err != nil {
		return err
	}
	if query == nil || query.Pointer() == 0 {
		return core.E("rocm.hip.AttentionLaunch", "attention query device buffer is required", nil)
	}
	if queryElementOffset < 0 || query.Count() < queryElementOffset+dim || query.SizeBytes() < uint64(queryElementOffset+dim)*4 {
		return core.E("rocm.hip.AttentionLaunch", "attention query device buffer shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 {
		return core.E("rocm.hip.AttentionLaunch", "attention destination output buffer is required", nil)
	}
	if outputElementOffset < 0 || output.Count() < outputElementOffset+dim || output.SizeBytes() < uint64(outputElementOffset+dim)*4 {
		return core.E("rocm.hip.AttentionLaunch", "attention destination output buffer shape mismatch", nil)
	}
	weights, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionLaunch", "attention weights", uint64(tokenCount*4), tokenCount)
	if err != nil {
		return err
	}
	defer weights.Close()

	launch := hipAttentionLaunchArgs{
		QueryPointer:  nativeDevicePointer(uintptr(query.Pointer()) + uintptr(queryElementOffset*4)),
		OutputPointer: nativeDevicePointer(uintptr(output.Pointer()) + uintptr(outputElementOffset*4)),
		WeightPointer: weights.Pointer(),
		Dim:           dim,
		TokenCount:    tokenCount,
		QueryBytes:    uint64(dim * 4),
		OutputBytes:   uint64(dim * 4),
		WeightBytes:   weights.SizeBytes(),
		Scale:         req.Scale,
	}
	if req.DeviceKV == nil {
		keyPayload, err := hipFloat32Payload(req.Keys)
		if err != nil {
			return core.E("rocm.hip.AttentionLaunch", "encode keys", err)
		}
		keys, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionLaunch", "attention keys", keyPayload, len(req.Keys))
		if err != nil {
			return err
		}
		defer keys.Close()
		valuePayload, err := hipFloat32Payload(req.Values)
		if err != nil {
			return core.E("rocm.hip.AttentionLaunch", "encode values", err)
		}
		values, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionLaunch", "attention values", valuePayload, len(req.Values))
		if err != nil {
			return err
		}
		defer values.Close()
		launch.KVSource = hipAttentionKVSourceContiguous
		launch.KeyPointer = keys.Pointer()
		launch.ValuePointer = values.Pointer()
		launch.KeyBytes = keys.SizeBytes()
		launch.ValueBytes = values.SizeBytes()
	} else {
		launch.KVSource = hipAttentionKVSourceDevice
		launch.DescriptorPointer = req.DescriptorTable.Pointer()
		launch.DescriptorBytes = req.DescriptorTable.SizeBytes()
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameAttention, launchBytes, tokenCount)
	if err != nil {
		return err
	}
	return hipLaunchKernel(driver, config)
}

func hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernel(ctx context.Context, driver nativeHIPDriver, req hipAttentionRequest, query *hipDeviceByteBuffer, headCount int, output *hipDeviceByteBuffer) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	dim, tokenCount, err := req.shape()
	if err != nil {
		return err
	}
	if headCount <= 0 {
		return core.E("rocm.hip.AttentionHeadsLaunch", "head count must be positive", nil)
	}
	if query == nil || query.Pointer() == 0 || query.Count() != headCount*dim || query.SizeBytes() != uint64(headCount*dim*4) {
		return core.E("rocm.hip.AttentionHeadsLaunch", "attention query device buffer shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != headCount*dim || output.SizeBytes() != uint64(headCount*dim*4) {
		return core.E("rocm.hip.AttentionHeadsLaunch", "attention output device buffer shape mismatch", nil)
	}
	useSharedWeights := tokenCount <= hipAttentionHeadsSharedMaxTokens
	var weights *hipDeviceByteBuffer
	if !useSharedWeights {
		weights, err = hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsLaunch", "attention head weights", uint64(headCount*tokenCount*4), headCount*tokenCount)
		if err != nil {
			return err
		}
		defer weights.Close()
	}
	launch := hipAttentionHeadsLaunchArgs{
		QueryPointer:  query.Pointer(),
		OutputPointer: output.Pointer(),
		Dim:           dim,
		TokenCount:    tokenCount,
		HeadCount:     headCount,
		QueryBytes:    query.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		Scale:         req.Scale,
	}
	var sharedMemBytes uint32
	if useSharedWeights {
		sharedMemBytes, err = hipAttentionHeadsSharedMemBytes(tokenCount, req.DeviceKV != nil)
		if err != nil {
			return err
		}
		launch.SharedMemBytes = uint64(sharedMemBytes)
	} else {
		launch.WeightPointer = weights.Pointer()
		launch.WeightBytes = weights.SizeBytes()
	}
	if req.DeviceKV == nil {
		launch.KVSource = hipAttentionKVSourceContiguous
		keyPayload, err := hipFloat32Payload(req.Keys)
		if err != nil {
			return core.E("rocm.hip.AttentionHeadsLaunch", "encode keys", err)
		}
		keys, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionHeadsLaunch", "attention keys", keyPayload, len(req.Keys))
		if err != nil {
			return err
		}
		defer keys.Close()
		valuePayload, err := hipFloat32Payload(req.Values)
		if err != nil {
			return core.E("rocm.hip.AttentionHeadsLaunch", "encode values", err)
		}
		values, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionHeadsLaunch", "attention values", valuePayload, len(req.Values))
		if err != nil {
			return err
		}
		defer values.Close()
		launch.KeyPointer = keys.Pointer()
		launch.ValuePointer = values.Pointer()
		launch.KeyBytes = keys.SizeBytes()
		launch.ValueBytes = values.SizeBytes()
	} else {
		launch.KVSource = hipAttentionKVSourceDevice
		launch.DescriptorPointer = req.DescriptorTable.Pointer()
		launch.DescriptorBytes = req.DescriptorTable.SizeBytes()
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return err
	}
	config := hipKernelLaunchConfig{
		Name:           hipKernelNameAttentionHeads,
		Args:           launchBytes,
		GridX:          uint32(headCount),
		GridY:          1,
		GridZ:          1,
		BlockX:         hipAttentionHeadsBlockSize(tokenCount),
		BlockY:         1,
		BlockZ:         1,
		SharedMemBytes: sharedMemBytes,
	}
	if err := config.Validate(); err != nil {
		return err
	}
	return hipLaunchKernel(driver, config)
}

type hipAttentionHeadsBatchCausalDeviceRequest struct {
	Key             *hipDeviceByteBuffer
	Value           *hipDeviceByteBuffer
	DeviceKV        *rocmDeviceKVCache
	DescriptorTable *rocmDeviceKVDescriptorTable
	Dim             int
	TokenCount      int
	HeadCount       int
	QueryCount      int
	QueryStartToken int
	Scale           float32
}

func hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernel(ctx context.Context, driver nativeHIPDriver, req hipAttentionHeadsBatchCausalDeviceRequest, query *hipDeviceByteBuffer, output *hipDeviceByteBuffer) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if req.Dim <= 0 || req.TokenCount <= 0 || req.HeadCount <= 0 || req.QueryCount <= 0 {
		return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch dimensions must be positive", nil)
	}
	if req.QueryStartToken < 0 || uint64(req.QueryStartToken)+uint64(req.QueryCount) > uint64(req.TokenCount) {
		return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "causal query window exceeds token count", nil)
	}
	if req.Scale < 0 || math.IsNaN(float64(req.Scale)) || math.IsInf(float64(req.Scale), 0) {
		return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "scale must be non-negative and finite", nil)
	}
	queryCount := req.QueryCount * req.HeadCount * req.Dim
	if query == nil || query.Pointer() == 0 || query.Count() != queryCount || query.SizeBytes() != uint64(queryCount*4) {
		return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "attention query device buffer shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != queryCount || output.SizeBytes() != uint64(queryCount*4) {
		return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "attention output device buffer shape mismatch", nil)
	}
	launch := hipAttentionHeadsBatchCausalLaunchArgs{
		QueryPointer:    query.Pointer(),
		OutputPointer:   output.Pointer(),
		Dim:             req.Dim,
		TokenCount:      req.TokenCount,
		HeadCount:       req.HeadCount,
		QueryCount:      req.QueryCount,
		QueryStartToken: req.QueryStartToken,
		QueryBytes:      query.SizeBytes(),
		OutputBytes:     output.SizeBytes(),
		Scale:           req.Scale,
	}
	if req.DeviceKV == nil {
		if req.DescriptorTable != nil {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "descriptor table requires device KV cache", nil)
		}
		if req.Key == nil || req.Key.Pointer() == 0 || req.Value == nil || req.Value.Pointer() == 0 {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "attention key and value device buffers are required", nil)
		}
		kvCount := req.TokenCount * req.Dim
		if req.Key.Count() != kvCount || req.Value.Count() != kvCount ||
			req.Key.SizeBytes() != uint64(kvCount*4) || req.Value.SizeBytes() != uint64(kvCount*4) {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "attention key/value device buffer shape mismatch", nil)
		}
		launch.KVSource = hipAttentionKVSourceContiguous
		launch.KeyPointer = req.Key.Pointer()
		launch.ValuePointer = req.Value.Pointer()
		launch.KeyBytes = req.Key.SizeBytes()
		launch.ValueBytes = req.Value.SizeBytes()
	} else {
		if req.Key != nil || req.Value != nil {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "device KV attention must not set contiguous KV buffers", nil)
		}
		if req.DescriptorTable == nil {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "device KV attention requires descriptor table", nil)
		}
		if err := req.DescriptorTable.CompatibleWith(req.DeviceKV); err != nil {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "descriptor table does not match device KV cache", err)
		}
		keyWidth, valueWidth, ok := req.DeviceKV.LastVectorWidths()
		if !ok {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "device KV cache has no pages", nil)
		}
		if keyWidth != req.Dim || valueWidth != req.Dim {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "device KV widths must match attention dimension", nil)
		}
		if req.DeviceKV.TokenCount() != req.TokenCount {
			return core.E("rocm.hip.AttentionHeadsBatchCausalLaunch", "device KV token count mismatch", nil)
		}
		launch.KVSource = hipAttentionKVSourceDevice
		launch.DescriptorPointer = req.DescriptorTable.Pointer()
		launch.DescriptorBytes = req.DescriptorTable.SizeBytes()
	}
	useSharedWeights := req.TokenCount <= hipAttentionHeadsSharedMaxTokens
	var sharedMemBytes uint32
	var weights *hipDeviceByteBuffer
	var err error
	if useSharedWeights {
		sharedMemBytes, err = hipAttentionHeadsSharedMemBytes(req.TokenCount, req.DeviceKV != nil)
		if err != nil {
			return err
		}
		launch.SharedMemBytes = uint64(sharedMemBytes)
	} else {
		weightCount := req.QueryCount * req.HeadCount * req.TokenCount
		weights, err = hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch head weights", uint64(weightCount)*4, weightCount)
		if err != nil {
			return err
		}
		defer weights.Close()
		launch.WeightPointer = weights.Pointer()
		launch.WeightBytes = weights.SizeBytes()
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return err
	}
	config := hipKernelLaunchConfig{
		Name:           hipKernelNameAttentionHeadsBatchCausal,
		Args:           launchBytes,
		GridX:          uint32(req.HeadCount),
		GridY:          uint32(req.QueryCount),
		GridZ:          1,
		BlockX:         hipAttentionHeadsBlockSize(req.TokenCount),
		BlockY:         1,
		BlockZ:         1,
		SharedMemBytes: sharedMemBytes,
	}
	if err := config.Validate(); err != nil {
		return err
	}
	return hipLaunchKernel(driver, config)
}

type hipAttentionHeadsChunkedWorkspace struct {
	Partial          *hipDeviceByteBuffer
	Stats            *hipDeviceByteBuffer
	AttentionOutputs map[int]*hipDeviceByteBuffer
	partialCap       int
	statsCap         int
}

func (workspace *hipAttentionHeadsChunkedWorkspace) Ensure(driver nativeHIPDriver, headCount, dim, tokenCount, chunkSize int) error {
	if workspace == nil {
		return core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if headCount <= 0 || dim <= 0 || tokenCount <= 0 || chunkSize <= 0 {
		return core.E("rocm.hip.AttentionHeadsChunkedLaunch", "workspace dimensions must be positive", nil)
	}
	chunkCount := (tokenCount + chunkSize - 1) / chunkSize
	partialCount := headCount * chunkCount * dim
	statsCount := headCount * chunkCount * 2
	if workspace.Partial == nil || workspace.Partial.Pointer() == 0 || workspace.partialCap < partialCount {
		if err := workspace.Partial.Close(); err != nil {
			return err
		}
		partial, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsChunkedLaunch", "attention chunked partials", uint64(partialCount*4), partialCount)
		if err != nil {
			return err
		}
		workspace.Partial = partial
		workspace.partialCap = partialCount
	}
	if workspace.Stats == nil || workspace.Stats.Pointer() == 0 || workspace.statsCap < statsCount {
		if err := workspace.Stats.Close(); err != nil {
			return err
		}
		stats, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsChunkedLaunch", "attention chunked stats", uint64(statsCount*4), statsCount)
		if err != nil {
			return err
		}
		workspace.Stats = stats
		workspace.statsCap = statsCount
	}
	return nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) EnsureAttentionOutput(driver nativeHIPDriver, headCount, dim int) (*hipDeviceByteBuffer, error) {
	if workspace == nil {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention workspace is required", nil)
	}
	if headCount <= 0 || dim <= 0 {
		return nil, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention output dimensions must be positive", nil)
	}
	count := headCount * dim
	if workspace.AttentionOutputs == nil {
		workspace.AttentionOutputs = make(map[int]*hipDeviceByteBuffer, 2)
	}
	if output := workspace.AttentionOutputs[count]; output != nil && output.Pointer() != 0 && output.Count() == count && output.SizeBytes() == uint64(count*4) {
		return output, nil
	}
	if err := workspace.AttentionOutputs[count].Close(); err != nil {
		return nil, err
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsChunkedLaunch", "attention concat output", uint64(count*4), count)
	if err != nil {
		return nil, err
	}
	workspace.AttentionOutputs[count] = output
	return output, nil
}

func (workspace *hipAttentionHeadsChunkedWorkspace) Close() error {
	if workspace == nil {
		return nil
	}
	var lastErr error
	if err := workspace.Partial.Close(); err != nil {
		lastErr = err
	}
	if err := workspace.Stats.Close(); err != nil {
		lastErr = err
	}
	for _, output := range workspace.AttentionOutputs {
		if err := output.Close(); err != nil {
			lastErr = err
		}
	}
	workspace.AttentionOutputs = nil
	return lastErr
}

func hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernelWithWorkspace(ctx context.Context, driver nativeHIPDriver, req hipAttentionRequest, query *hipDeviceByteBuffer, headCount int, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if workspace == nil {
		return hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernel(ctx, driver, req, query, headCount, output)
	}
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	dim, tokenCount, err := req.shape()
	if err != nil {
		return err
	}
	if !hipAttentionHeadsChunkedEligible(req, dim, tokenCount) {
		return hipRunAttentionHeadsOutputFromDeviceQueryToDeviceKernel(ctx, driver, req, query, headCount, output)
	}
	if headCount <= 0 {
		return core.E("rocm.hip.AttentionHeadsChunkedLaunch", "head count must be positive", nil)
	}
	if query == nil || query.Pointer() == 0 || query.Count() != headCount*dim || query.SizeBytes() != uint64(headCount*dim*4) {
		return core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention query device buffer shape mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != headCount*dim || output.SizeBytes() != uint64(headCount*dim*4) {
		return core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention output device buffer shape mismatch", nil)
	}
	return hipRunAttentionHeadsChunked(ctx, driver, req, query, headCount, dim, tokenCount, output, workspace)
}

func hipAttentionHeadsChunkedEligible(req hipAttentionRequest, dim, tokenCount int) bool {
	if dim <= 0 || dim > hipAttentionHeadsChunkedBlockSize || tokenCount < hipAttentionHeadsChunkSize {
		return false
	}
	if req.DeviceKV == nil || req.DescriptorTable == nil {
		return false
	}
	if req.DeviceKV.mode != rocmKVCacheModeKQ8VQ4 {
		return false
	}
	return req.DeviceKV.TokenCount() == tokenCount && req.DeviceKV.PageCount() == tokenCount
}

func hipRunAttentionHeadsChunked(ctx context.Context, driver nativeHIPDriver, req hipAttentionRequest, query *hipDeviceByteBuffer, headCount, dim, tokenCount int, output *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	chunkSize := hipAttentionHeadsChunkSize
	chunkCount := (tokenCount + chunkSize - 1) / chunkSize
	if err := workspace.Ensure(driver, headCount, dim, tokenCount, chunkSize); err != nil {
		return err
	}
	launch := hipAttentionHeadsChunkedLaunchArgs{
		QueryPointer:      query.Pointer(),
		DescriptorPointer: req.DescriptorTable.Pointer(),
		PartialPointer:    workspace.Partial.Pointer(),
		StatsPointer:      workspace.Stats.Pointer(),
		OutputPointer:     output.Pointer(),
		Dim:               dim,
		TokenCount:        tokenCount,
		HeadCount:         headCount,
		ChunkSize:         chunkSize,
		ChunkCount:        chunkCount,
		QueryBytes:        query.SizeBytes(),
		DescriptorBytes:   req.DescriptorTable.SizeBytes(),
		PartialBytes:      uint64(headCount * chunkCount * dim * 4),
		StatsBytes:        uint64(headCount * chunkCount * 2 * 4),
		OutputBytes:       output.SizeBytes(),
		Scale:             req.Scale,
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return err
	}
	stage2LaunchBytes := hipBorrowLaunchPacket(len(launchBytes))
	copy(stage2LaunchBytes, launchBytes)
	sharedMemBytes, err := hipAttentionHeadsChunkedSharedMemBytes(chunkSize, dim)
	if err != nil {
		hipReleaseLaunchPacket(stage2LaunchBytes)
		return err
	}
	gridX, err := rocmDeviceKVPositiveUint32("attention chunked stage1 blocks", headCount*chunkCount)
	if err != nil {
		hipReleaseLaunchPacket(stage2LaunchBytes)
		return err
	}
	stage1 := hipKernelLaunchConfig{
		Name:           hipKernelNameAttentionHeadsChunkedStage1,
		Args:           launchBytes,
		GridX:          gridX,
		GridY:          1,
		GridZ:          1,
		BlockX:         hipAttentionHeadsChunkedBlockSize,
		BlockY:         1,
		BlockZ:         1,
		SharedMemBytes: sharedMemBytes,
	}
	if err := stage1.Validate(); err != nil {
		return err
	}
	if err := hipLaunchKernel(driver, stage1); err != nil {
		return err
	}
	stage2 := hipKernelLaunchConfig{
		Name:           hipKernelNameAttentionHeadsChunkedStage2,
		Args:           stage2LaunchBytes,
		GridX:          uint32(headCount),
		GridY:          1,
		GridZ:          1,
		BlockX:         hipAttentionHeadsChunkedBlockSize,
		BlockY:         1,
		BlockZ:         1,
		SharedMemBytes: 0,
	}
	if err := stage2.Validate(); err != nil {
		hipReleaseLaunchPacket(stage2LaunchBytes)
		return err
	}
	return hipLaunchKernel(driver, stage2)
}

func hipAttentionHeadsChunkedSharedMemBytes(chunkSize, dim int) (uint32, error) {
	chunk, err := rocmDeviceKVPositiveUint32("attention chunked chunk size", chunkSize)
	if err != nil {
		return 0, err
	}
	width, err := rocmDeviceKVPositiveUint32("attention chunked query dim", dim)
	if err != nil {
		return 0, err
	}
	bytes := uint64(chunk) * 4
	bytes = hipAttentionHeadsAlignSharedBytes(bytes, 8)
	bytes += uint64(chunk) * 8
	bytes = hipAttentionHeadsAlignSharedBytes(bytes, 4)
	bytes += uint64(chunk) * 4
	bytes = hipAttentionHeadsAlignSharedBytes(bytes, 4)
	bytes += uint64(width) * 4
	if bytes > math.MaxUint32 {
		return 0, core.E("rocm.hip.AttentionHeadsChunkedLaunch", "attention chunked shared memory byte count is out of uint32 range", nil)
	}
	return uint32(bytes), nil
}

func hipAttentionHeadsSharedMemBytes(tokenCount int, deviceKV bool) (uint32, error) {
	tokens, err := rocmDeviceKVPositiveUint32("attention token count", tokenCount)
	if err != nil {
		return 0, err
	}
	bytes := uint64(tokens) * 4
	if deviceKV && tokenCount >= 16 {
		bytes = hipAttentionHeadsAlignSharedBytes(bytes, 8)
		bytes += uint64(tokens) * 8
		bytes = hipAttentionHeadsAlignSharedBytes(bytes, 4)
		bytes += uint64(tokens) * 4
	}
	if bytes > math.MaxUint32 {
		return 0, core.E("rocm.hip.AttentionHeadsLaunch", "attention shared memory byte count is out of uint32 range", nil)
	}
	return uint32(bytes), nil
}

func hipAttentionHeadsAlignSharedBytes(value, alignment uint64) uint64 {
	if alignment <= 1 {
		return value
	}
	remainder := value % alignment
	if remainder == 0 {
		return value
	}
	return value + alignment - remainder
}

func hipAttentionHeadsBlockSize(tokenCount int) uint32 {
	if tokenCount >= 16 {
		return 512
	}
	return 256
}

func hipRunVectorAddKernel(ctx context.Context, driver nativeHIPDriver, req hipVectorAddRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameVectorAdd, launchBytes, buffers.Count)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipRunVectorAddDeviceKernel(ctx context.Context, driver nativeHIPDriver, left, right *hipDeviceByteBuffer) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if left == nil || right == nil || left.Pointer() == 0 || right.Pointer() == 0 {
		return nil, core.E("rocm.hip.VectorAddLaunch", "vector add device inputs are required", nil)
	}
	if left.Count() <= 0 || right.Count() != left.Count() ||
		left.SizeBytes() != uint64(left.Count()*4) ||
		right.SizeBytes() != uint64(right.Count()*4) {
		return nil, core.E("rocm.hip.VectorAddLaunch", "vector add device input shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.VectorAddLaunch", "vector add output", left.SizeBytes(), left.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipVectorAddLaunchArgs{
		LeftPointer:   left.Pointer(),
		RightPointer:  right.Pointer(),
		OutputPointer: output.Pointer(),
		Count:         left.Count(),
		LeftBytes:     left.SizeBytes(),
		RightBytes:    right.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameVectorAdd, launchBytes, left.Count())
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunVectorScaleKernel(ctx context.Context, driver nativeHIPDriver, req hipVectorScaleRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameVectorScale, launchBytes, buffers.Count)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipRunVectorScaleDeviceKernel(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, scale float32) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.VectorScaleLaunch", "vector scale device input is required", nil)
	}
	if input.Count() <= 0 || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E("rocm.hip.VectorScaleLaunch", "vector scale device input shape mismatch", nil)
	}
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return nil, core.E("rocm.hip.VectorScaleLaunch", "scale must be finite", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.VectorScaleLaunch", "vector scale output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipVectorScaleLaunchArgs{
		InputPointer:  input.Pointer(),
		OutputPointer: output.Pointer(),
		Count:         input.Count(),
		InputBytes:    input.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		Scale:         scale,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameVectorScale, launchBytes, input.Count())
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunSwiGLUKernel(ctx context.Context, driver nativeHIPDriver, req hipSwiGLURequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameSwiGLU, launchBytes, buffers.Count)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipRunGELUTanhMultiplyKernel(ctx context.Context, driver nativeHIPDriver, req hipGELUTanhMultiplyRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	defer buffers.Close()
	if err := hipLaunchGELUTanhMultiplyDeviceBuffers(driver, buffers); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
}

func hipRunGELUTanhMultiplyDeviceKernel(ctx context.Context, driver nativeHIPDriver, gate, up *hipDeviceByteBuffer) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if gate == nil || up == nil || gate.Pointer() == 0 || up.Pointer() == 0 {
		return nil, core.E("rocm.hip.GELUTanhMultiplyLaunch", "gate and up device buffers are required", nil)
	}
	if gate.Count() <= 0 || up.Count() != gate.Count() ||
		gate.SizeBytes() != uint64(gate.Count()*4) ||
		up.SizeBytes() != uint64(up.Count()*4) {
		return nil, core.E("rocm.hip.GELUTanhMultiplyLaunch", "gate and up device buffer shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.GELUTanhMultiplyLaunch", "GELU tanh multiply output", gate.SizeBytes(), gate.Count())
	if err != nil {
		return nil, err
	}
	buffers := &hipGELUTanhMultiplyDeviceBuffers{Gate: gate, Up: up, Output: output, Count: gate.Count()}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipLaunchGELUTanhMultiplyDeviceBuffers(driver, buffers); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipLaunchGELUTanhMultiplyDeviceBuffers(driver nativeHIPDriver, buffers *hipGELUTanhMultiplyDeviceBuffers) error {
	launch, err := hipGELUTanhMultiplyLaunchArgsForDeviceBuffers(buffers)
	if err != nil {
		return err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameGELUTanhMul, launchBytes, buffers.Count)
	if err != nil {
		return err
	}
	return hipLaunchKernel(driver, config)
}

func hipRunGreedyKernel(ctx context.Context, driver nativeHIPDriver, req hipGreedySampleRequest) (hipGreedySampleResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipGreedySampleResult{}, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameGreedy, launchBytes, buffers.Count)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipGreedySampleResult{}, err
	}
	return buffers.ReadOutput()
}

func hipRunSoftcapGreedyKernelWithDeviceLogits(ctx context.Context, driver nativeHIPDriver, logits *hipDeviceByteBuffer, softcap float32) (hipGreedySampleResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipGreedySampleResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipGreedySampleResult{}, core.E("rocm.hip.SoftcapGreedyLaunch", "HIP driver is not available", nil)
	}
	if logits == nil || logits.Pointer() == 0 {
		return hipGreedySampleResult{}, core.E("rocm.hip.SoftcapGreedyLaunch", "logits device buffer is required", nil)
	}
	if logits.Count() <= 0 || logits.SizeBytes() != uint64(logits.Count()*4) {
		return hipGreedySampleResult{}, core.E("rocm.hip.SoftcapGreedyLaunch", "logits device buffer shape mismatch", nil)
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return hipGreedySampleResult{}, core.E("rocm.hip.SoftcapGreedyLaunch", "softcap must be non-negative and finite", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.SoftcapGreedyLaunch", "softcap greedy output", hipGreedyResultBytes, 1)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	defer output.Close()
	launchBytes, err := (hipSoftcapGreedySampleLaunchArgs{
		LogitsPointer: logits.Pointer(),
		OutputPointer: output.Pointer(),
		Count:         logits.Count(),
		LogitsBytes:   logits.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		Softcap:       softcap,
	}).Binary()
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameSoftcapGreedy,
		Args:   launchBytes,
		GridX:  1,
		GridY:  1,
		GridZ:  1,
		BlockX: 256,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := config.Validate(); err != nil {
		return hipGreedySampleResult{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipGreedySampleResult{}, err
	}
	return hipReadGreedyResult(output, "rocm.hip.SoftcapGreedyLaunch", "softcap greedy output", logits.Count())
}
