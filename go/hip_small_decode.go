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
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameRMSNorm, launchBytes, buffers.Count)
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
	if cfg.WeightPointer == 0 {
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
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormLaunch", "rms norm output", uint64(cfg.Count*4), cfg.Count)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	launchBytes, err := (hipRMSNormLaunchArgs{
		InputPointer:   inputBuffer.Pointer(),
		WeightPointer:  cfg.WeightPointer,
		OutputPointer:  output.Pointer(),
		Count:          cfg.Count,
		InputBytes:     inputBuffer.SizeBytes(),
		WeightBytes:    cfg.WeightBytes,
		OutputBytes:    output.SizeBytes(),
		Epsilon:        cfg.Epsilon,
		WeightEncoding: cfg.WeightEncoding,
		Flags:          cfg.Flags,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameRMSNorm, launchBytes, cfg.Count)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	return hipReadFloat32DeviceOutput(output, "rocm.hip.RMSNormLaunch", "rms norm output", cfg.Count)
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
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameRoPE, launchBytes, buffers.Count)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	return buffers.ReadOutput()
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
