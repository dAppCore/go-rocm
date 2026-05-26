// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"iter"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

type hipLoadedTinyLMConfig struct {
	EmbeddingPointer       nativeDevicePointer
	EmbeddingBytes         uint64
	OutputWeightPointer    nativeDevicePointer
	OutputWeightBytes      uint64
	OutputWeightEncoding   uint32
	Q8Scale                float32
	OutputJANGTQDescriptor rocmJANGTQDescriptor
	OutputJANGTQScale      float32
	OutputCodebookPointer  nativeDevicePointer
	OutputCodebookBytes    uint64
	OutputCodebookCount    int
	OutputCodebookDim      int
	VocabSize              int
	HiddenSize             int
}

func (model *hipLoadedModel) loadedTinyLMConfig() (hipLoadedTinyLMConfig, error) {
	if model == nil {
		return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "loaded model is required", nil)
	}
	if model.driver == nil || !model.driver.Available() {
		return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "HIP driver is not available", nil)
	}
	architecture := normalizeROCmArchitecture(model.modelInfo.Architecture)
	if architecture != "" && architecture != "tiny" {
		return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "tiny loaded path supports only tiny architecture fixtures", nil)
	}
	embedding, ok := model.findHIPTensor(isHIPEmbeddingTensor)
	if !ok {
		return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "embedding tensor is required", nil)
	}
	output, ok := model.findHIPTensor(isHIPOutputTensor)
	if !ok {
		return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "output tensor is required", nil)
	}
	if !hipTinyTensorIsFP32(embedding.info) {
		return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "tiny loaded path requires f32 embeddings", nil)
	}
	vocabSize, hiddenSize, err := hipTinyTensorVocabHiddenShape(model.modelInfo, embedding.info)
	if err != nil {
		return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "embedding shape", err)
	}
	outputVocabSize, outputHiddenSize, err := hipTinyTensorVocabHiddenShape(model.modelInfo, output.info)
	if err != nil {
		return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "output shape", err)
	}
	if outputVocabSize != vocabSize || outputHiddenSize != hiddenSize {
		return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "embedding and output tensor shapes must match", nil)
	}
	encoding, q8Scale, jangtqDescriptor, jangtqScale, err := hipTinyLoadedOutputEncoding(output.info)
	if err != nil {
		return hipLoadedTinyLMConfig{}, err
	}
	codebookDim, _, err := hipTinyLoadedCodebookOutput(output.info.TypeName)
	if err != nil {
		return hipLoadedTinyLMConfig{}, err
	}
	tableCount := uint64(vocabSize) * uint64(hiddenSize)
	if _, err := hipExactUint32Bytes("embedding", embedding.info.ByteSize, tableCount*4); err != nil {
		return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "embedding byte count", err)
	}
	var codebook hipTensor
	var codebookCount int
	if encoding == hipTinyOutputWeightEncodingJANGTQ {
		tableCountInt, err := hipTinyUint64ToInt("JANGTQ output weight count", tableCount)
		if err != nil {
			return hipLoadedTinyLMConfig{}, err
		}
		if _, err := hipExactUint32Bytes("JANGTQ output weight", output.info.ByteSize, uint64(packedROCmJANGTQBytes(jangtqDescriptor.Bits, tableCountInt))); err != nil {
			return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "output byte count", err)
		}
	} else if encoding == hipTinyOutputWeightEncodingCodebook {
		if codebookDim != 1 {
			return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "codebook output code dimension must be 1 for scalar output weights", nil)
		}
		if _, err := hipExactUint32Bytes("codebook output codes", output.info.ByteSize, tableCount); err != nil {
			return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "output byte count", err)
		}
		var ok bool
		codebook, ok = model.findHIPTensor(isHIPOutputCodebookTensor)
		if !ok {
			return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "codebook output table tensor is required", nil)
		}
		if !hipTinyTensorIsFP32(codebook.info) {
			return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "codebook output table must be f32", nil)
		}
		codebookCount, err = hipTinyCodebookTensorShape(codebook.info, codebookDim)
		if err != nil {
			return hipLoadedTinyLMConfig{}, err
		}
		if _, err := hipExactUint32Bytes("codebook output table", codebook.info.ByteSize, uint64(codebookCount*codebookDim)*4); err != nil {
			return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "codebook table byte count", err)
		}
	} else if _, err := hipTinyOutputWeightByteCount(encoding, output.info.ByteSize, tableCount, q8Scale); err != nil {
		return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "output byte count", err)
	}
	if embedding.pointer == 0 || output.pointer == 0 {
		return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "embedding and output tensor pointers are required", nil)
	}
	if encoding == hipTinyOutputWeightEncodingCodebook && codebook.pointer == 0 {
		return hipLoadedTinyLMConfig{}, core.E("rocm.hip.TinyLoadedModel", "codebook output table tensor pointer is required", nil)
	}
	return hipLoadedTinyLMConfig{
		EmbeddingPointer:       embedding.pointer,
		EmbeddingBytes:         embedding.info.ByteSize,
		OutputWeightPointer:    output.pointer,
		OutputWeightBytes:      output.info.ByteSize,
		OutputWeightEncoding:   encoding,
		Q8Scale:                q8Scale,
		OutputJANGTQDescriptor: jangtqDescriptor,
		OutputJANGTQScale:      jangtqScale,
		OutputCodebookPointer:  codebook.pointer,
		OutputCodebookBytes:    codebook.info.ByteSize,
		OutputCodebookCount:    codebookCount,
		OutputCodebookDim:      codebookDim,
		VocabSize:              vocabSize,
		HiddenSize:             hiddenSize,
	}, nil
}

func (model *hipLoadedModel) findHIPTensor(match func(string) bool) (hipTensor, bool) {
	if model == nil || match == nil {
		return hipTensor{}, false
	}
	for _, tensor := range model.tensors {
		if match(core.Lower(tensor.info.Name)) {
			return tensor, true
		}
	}
	return hipTensor{}, false
}

func (model *hipLoadedModel) tinyLoadedKernelStatus(status hipKernelStatus) hipKernelStatus {
	status = normalizeHIPKernelStatus(status)
	if model == nil {
		return status
	}
	if _, ok := model.kernelSet().(hipNativeProjectionKernelSet); !ok {
		return status
	}
	if _, err := model.loadedTinyLMConfig(); err != nil {
		if _, hasClassifier, classifierErr := model.loadedSequenceClassifierConfig(); classifierErr == nil && hasClassifier {
			status.LoRA = hipKernelStatusLinked
			status.Reason = "native classifier LoRA projection kernel is linked for loaded BERT sequence-classifier rerank; production adapter application remains limited"
		}
		if _, smallErr := model.loadedSmallDecodeConfig(); smallErr == nil {
			status.LoRA = hipKernelStatusLinked
			status.Reason = "native small-decode LM-head LoRA projection kernel is linked for loaded Qwen/Gemma decode smoke; production adapter application remains limited"
		}
		return status
	}
	status.Decode = hipKernelStatusLinked
	status.Prefill = hipKernelStatusLinked
	status.LoRA = hipKernelStatusLinked
	status.Reason = "native tiny loaded-model prefill/decode kernels are linked for f32 toy models with f32/f16/q8/JANGTQ/codebook output heads; production generation remains limited"
	return status
}

func hipTinyTensorVocabHiddenShape(info inference.ModelInfo, tensor nativeTensorInfo) (int, int, error) {
	if len(tensor.Dimensions) != 2 {
		return 0, 0, core.E("rocm.hip.TinyLoadedModel", "tiny loaded path requires rank-2 vocab-major tensors", nil)
	}
	vocabSize, err := hipTinyUint64ToInt("vocab size", tensor.Dimensions[0])
	if err != nil {
		return 0, 0, err
	}
	hiddenSize, err := hipTinyUint64ToInt("hidden size", tensor.Dimensions[1])
	if err != nil {
		return 0, 0, err
	}
	if info.VocabSize > 0 && vocabSize != info.VocabSize {
		return 0, 0, core.E("rocm.hip.TinyLoadedModel", core.Sprintf("vocab-major tensor first dimension %d does not match vocab size %d", vocabSize, info.VocabSize), nil)
	}
	if info.HiddenSize > 0 && hiddenSize != info.HiddenSize {
		return 0, 0, core.E("rocm.hip.TinyLoadedModel", core.Sprintf("vocab-major tensor second dimension %d does not match hidden size %d", hiddenSize, info.HiddenSize), nil)
	}
	return vocabSize, hiddenSize, nil
}

func hipTinyUint64ToInt(label string, value uint64) (int, error) {
	maxInt := uint64(^uint(0) >> 1)
	if value == 0 {
		return 0, core.E("rocm.hip.TinyLoadedModel", label+" must be positive", nil)
	}
	if value > maxInt {
		return 0, core.E("rocm.hip.TinyLoadedModel", label+" exceeds int range", nil)
	}
	return int(value), nil
}

func hipTinyTensorIsFP32(tensor nativeTensorInfo) bool {
	name := core.Lower(tensor.TypeName)
	return tensor.Type == 0 || name == "f32" || name == "float32"
}

func hipTinyTensorIsFP16(tensor nativeTensorInfo) bool {
	name := core.Lower(tensor.TypeName)
	return tensor.Type == 1 || name == "f16" || name == "float16"
}

func hipTinyTensorIsRawQ8(tensor nativeTensorInfo) bool {
	name := core.Lower(tensor.TypeName)
	return tensor.Type == 24 || name == "q8" || name == "i8" || core.HasPrefix(name, "q8:") || core.HasPrefix(name, "i8:")
}

func hipTinyLoadedOutputEncoding(tensor nativeTensorInfo) (uint32, float32, rocmJANGTQDescriptor, float32, error) {
	if desc, scale, ok, err := hipTinyLoadedJANGTQOutput(tensor.TypeName); ok || err != nil {
		return hipTinyOutputWeightEncodingJANGTQ, 0, desc, scale, err
	}
	if _, ok, err := hipTinyLoadedCodebookOutput(tensor.TypeName); ok || err != nil {
		return hipTinyOutputWeightEncodingCodebook, 0, rocmJANGTQDescriptor{}, 0, err
	}
	switch {
	case hipTinyTensorIsFP32(tensor):
		return hipTinyOutputWeightEncodingFP32, 0, rocmJANGTQDescriptor{}, 0, nil
	case hipTinyTensorIsFP16(tensor):
		return hipTinyOutputWeightEncodingFP16, 0, rocmJANGTQDescriptor{}, 0, nil
	case hipTinyTensorIsRawQ8(tensor):
		scale, err := hipTinyLoadedQ8Scale(tensor.TypeName)
		if err != nil {
			return 0, 0, rocmJANGTQDescriptor{}, 0, err
		}
		return hipTinyOutputWeightEncodingQ8, scale, rocmJANGTQDescriptor{}, 0, nil
	default:
		return 0, 0, rocmJANGTQDescriptor{}, 0, core.E("rocm.hip.TinyLoadedModel", "tiny loaded path supports only f32, f16, raw q8, JANGTQ, or codebook output tensors", nil)
	}
}

func hipTinyLoadedJANGTQOutput(typeName string) (rocmJANGTQDescriptor, float32, bool, error) {
	name := core.Lower(core.Trim(typeName))
	if !core.Contains(name, "jangtq") && !core.Contains(name, "mxtq") {
		return rocmJANGTQDescriptor{}, 0, false, nil
	}
	desc := rocmJANGTQDescriptor{WeightFormat: "mxtq", Bits: 2, GroupSize: 64}
	scale := float32(1)
	fields := strings.FieldsFunc(name, func(r rune) bool {
		return r == ':' || r == ',' || r == ';' || r == ' '
	})
	for _, field := range fields {
		key, value, ok := strings.Cut(field, "=")
		if !ok {
			continue
		}
		key = core.Trim(key)
		value = core.Trim(value)
		switch key {
		case "bits", "bit":
			parsed, err := strconv.Atoi(value)
			if err != nil {
				return rocmJANGTQDescriptor{}, 0, true, core.E("rocm.hip.TinyLoadedModel", "parse JANGTQ bits", err)
			}
			desc.Bits = parsed
		case "group", "group_size", "groupsize":
			parsed, err := strconv.Atoi(value)
			if err != nil {
				return rocmJANGTQDescriptor{}, 0, true, core.E("rocm.hip.TinyLoadedModel", "parse JANGTQ group size", err)
			}
			desc.GroupSize = parsed
		case "scale":
			parsed, err := strconv.ParseFloat(value, 32)
			if err != nil {
				return rocmJANGTQDescriptor{}, 0, true, core.E("rocm.hip.TinyLoadedModel", "parse JANGTQ scale", err)
			}
			scale = float32(parsed)
		}
	}
	if err := validateROCmJANGTQDescriptor(desc); err != nil {
		return rocmJANGTQDescriptor{}, 0, true, err
	}
	if !hipQ8ScaleIsPositiveFinite(scale) {
		return rocmJANGTQDescriptor{}, 0, true, core.E("rocm.hip.TinyLoadedModel", "JANGTQ scale must be positive and finite", nil)
	}
	return desc, scale, true, nil
}

func hipTinyLoadedCodebookOutput(typeName string) (int, bool, error) {
	name := core.Lower(core.Trim(typeName))
	if !core.Contains(name, "codebook") && !core.Contains(name, "vq") {
		return 0, false, nil
	}
	codeDim := 1
	fields := strings.FieldsFunc(name, func(r rune) bool {
		return r == ':' || r == ',' || r == ';' || r == ' '
	})
	for _, field := range fields {
		key, value, ok := strings.Cut(field, "=")
		if !ok {
			continue
		}
		key = core.Trim(key)
		value = core.Trim(value)
		switch key {
		case "dim", "code_dim", "codedim":
			parsed, err := strconv.Atoi(value)
			if err != nil {
				return 0, true, core.E("rocm.hip.TinyLoadedModel", "parse codebook dimension", err)
			}
			codeDim = parsed
		}
	}
	if codeDim <= 0 {
		return 0, true, core.E("rocm.hip.TinyLoadedModel", "codebook dimension must be positive", nil)
	}
	return codeDim, true, nil
}

func isHIPOutputCodebookTensor(name string) bool {
	name = core.Lower(name)
	return name == "output.codebook" ||
		name == "lm_head.codebook" ||
		core.HasSuffix(name, ".output.codebook") ||
		core.HasSuffix(name, ".lm_head.codebook")
}

func hipTinyCodebookTensorShape(tensor nativeTensorInfo, codeDim int) (int, error) {
	if len(tensor.Dimensions) != 2 {
		return 0, core.E("rocm.hip.TinyLoadedModel", "codebook output table tensor must be rank 2", nil)
	}
	codebookCount, err := hipTinyUint64ToInt("codebook entry count", tensor.Dimensions[0])
	if err != nil {
		return 0, err
	}
	tableCodeDim, err := hipTinyUint64ToInt("codebook dimension", tensor.Dimensions[1])
	if err != nil {
		return 0, err
	}
	if tableCodeDim != codeDim {
		return 0, core.E("rocm.hip.TinyLoadedModel", "codebook output table dimension mismatch", nil)
	}
	return codebookCount, nil
}

func hipTinyLoadedQ8Scale(typeName string) (float32, error) {
	name := core.Lower(core.Trim(typeName))
	if name == "" || name == "q8" || name == "i8" {
		return 1, nil
	}
	_, rawScale, ok := strings.Cut(name, ":")
	if !ok {
		return 1, nil
	}
	value, err := strconv.ParseFloat(core.Trim(rawScale), 32)
	if err != nil {
		return 0, core.E("rocm.hip.TinyLoadedModel", "parse q8 output scale", err)
	}
	scale := float32(value)
	if !hipQ8ScaleIsPositiveFinite(scale) {
		return 0, core.E("rocm.hip.TinyLoadedModel", "q8 output scale must be positive and finite", nil)
	}
	return scale, nil
}

func hipTinyKernelOutputWeight(cfg hipLoadedTinyLMConfig) (nativeDevicePointer, uint64, uint32, float32) {
	if cfg.OutputWeightEncoding == hipTinyOutputWeightEncodingJANGTQ || cfg.OutputWeightEncoding == hipTinyOutputWeightEncodingCodebook {
		return cfg.EmbeddingPointer, cfg.EmbeddingBytes, hipTinyOutputWeightEncodingFP32, 0
	}
	return cfg.OutputWeightPointer, cfg.OutputWeightBytes, cfg.OutputWeightEncoding, cfg.Q8Scale
}

func hipTinyUsesJANGTQOutput(cfg hipLoadedTinyLMConfig) bool {
	return cfg.OutputWeightEncoding == hipTinyOutputWeightEncodingJANGTQ
}

func hipTinyUsesCodebookOutput(cfg hipLoadedTinyLMConfig) bool {
	return cfg.OutputWeightEncoding == hipTinyOutputWeightEncodingCodebook
}

func hipRunLoadedTinyPrefill(ctx context.Context, driver nativeHIPDriver, cfg hipLoadedTinyLMConfig, tokenIDs []int32) (hipTinyPrefillResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipTinyPrefillResult{}, err
	}
	if err := hipValidateTinyTokenIDs(tokenIDs, cfg.VocabSize); err != nil {
		return hipTinyPrefillResult{}, err
	}
	tokenBuffer, err := hipUploadTokenIDs(driver, tokenIDs)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	defer tokenBuffer.Close()

	logits, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyLoadedPrefill", "tiny loaded prefill logits", uint64(cfg.VocabSize*4), cfg.VocabSize)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	stateCount := len(tokenIDs) * cfg.HiddenSize
	buffers := &hipTinyPrefillDeviceBuffers{Logits: logits, TokenCount: len(tokenIDs), VocabSize: cfg.VocabSize, HiddenSize: cfg.HiddenSize}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()
	attention, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyLoadedPrefill", "tiny loaded prefill attention", uint64(len(tokenIDs)*4), len(tokenIDs))
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	buffers.Attention = attention
	keys, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyLoadedPrefill", "tiny loaded prefill keys", uint64(stateCount*4), stateCount)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	buffers.Keys = keys
	values, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyLoadedPrefill", "tiny loaded prefill values", uint64(stateCount*4), stateCount)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	buffers.Values = values
	result, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyLoadedPrefill", "tiny loaded prefill result", hipGreedyResultBytes, 1)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	buffers.Result = result
	outputWeightPointer, outputWeightBytes, outputWeightEncoding, outputScale := hipTinyKernelOutputWeight(cfg)

	launchBytes, err := (hipTinyPrefillLaunchArgs{
		TokenPointer:         tokenBuffer.Pointer(),
		EmbeddingPointer:     cfg.EmbeddingPointer,
		OutputWeightPointer:  outputWeightPointer,
		LogitPointer:         buffers.Logits.Pointer(),
		AttentionPointer:     buffers.Attention.Pointer(),
		ResultPointer:        buffers.Result.Pointer(),
		KeyPointer:           buffers.Keys.Pointer(),
		ValuePointer:         buffers.Values.Pointer(),
		TokenCount:           len(tokenIDs),
		VocabSize:            cfg.VocabSize,
		HiddenSize:           cfg.HiddenSize,
		TokenBytes:           tokenBuffer.SizeBytes(),
		EmbeddingBytes:       cfg.EmbeddingBytes,
		OutputWeightBytes:    outputWeightBytes,
		LogitBytes:           buffers.Logits.SizeBytes(),
		AttentionBytes:       buffers.Attention.SizeBytes(),
		ResultBytes:          buffers.Result.SizeBytes(),
		KeyBytes:             buffers.Keys.SizeBytes(),
		ValueBytes:           buffers.Values.SizeBytes(),
		OutputWeightEncoding: outputWeightEncoding,
		Q8Scale:              outputScale,
	}).Binary()
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameTinyPrefill, launchBytes, 1)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipTinyPrefillResult{}, err
	}
	output, err := buffers.ReadOutput()
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	success = true
	if err := buffers.Close(); err != nil {
		return hipTinyPrefillResult{}, err
	}
	return output, nil
}

func hipRunLoadedTinyDecode(ctx context.Context, driver nativeHIPDriver, cfg hipLoadedTinyLMConfig, tokenID int32, priorKeys, priorValues []float32) (hipTinyDecodeResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipTinyDecodeResult{}, err
	}
	if err := hipValidateTinyTokenIDs([]int32{tokenID}, cfg.VocabSize); err != nil {
		return hipTinyDecodeResult{}, err
	}
	if len(priorKeys) == 0 || len(priorValues) == 0 || len(priorKeys) != len(priorValues) || len(priorKeys)%cfg.HiddenSize != 0 {
		return hipTinyDecodeResult{}, core.E("rocm.hip.TinyLoadedDecode", "prior key/value tensors must align with hidden size", nil)
	}
	priorTokenCount := len(priorKeys) / cfg.HiddenSize
	keyPayload, err := hipFloat32Payload(priorKeys)
	if err != nil {
		return hipTinyDecodeResult{}, core.E("rocm.hip.TinyLoadedDecode", "encode prior keys", err)
	}
	keys, err := hipUploadByteBuffer(driver, "rocm.hip.TinyLoadedDecode", "tiny loaded decode prior keys", keyPayload, len(priorKeys))
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	buffers := &hipTinyDecodeDeviceBuffers{PriorKeys: keys, PriorTokenCount: priorTokenCount, VocabSize: cfg.VocabSize, HiddenSize: cfg.HiddenSize}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()
	valuePayload, err := hipFloat32Payload(priorValues)
	if err != nil {
		return hipTinyDecodeResult{}, core.E("rocm.hip.TinyLoadedDecode", "encode prior values", err)
	}
	values, err := hipUploadByteBuffer(driver, "rocm.hip.TinyLoadedDecode", "tiny loaded decode prior values", valuePayload, len(priorValues))
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	buffers.PriorValues = values
	logits, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyLoadedDecode", "tiny loaded decode logits", uint64(cfg.VocabSize*4), cfg.VocabSize)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	buffers.Logits = logits
	attention, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyLoadedDecode", "tiny loaded decode attention", uint64((priorTokenCount+1)*4), priorTokenCount+1)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	buffers.Attention = attention
	updatedCount := (priorTokenCount + 1) * cfg.HiddenSize
	updatedKeys, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyLoadedDecode", "tiny loaded decode updated keys", uint64(updatedCount*4), updatedCount)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	buffers.UpdatedKeys = updatedKeys
	updatedValues, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyLoadedDecode", "tiny loaded decode updated values", uint64(updatedCount*4), updatedCount)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	buffers.UpdatedValues = updatedValues
	result, err := hipAllocateByteBuffer(driver, "rocm.hip.TinyLoadedDecode", "tiny loaded decode result", hipGreedyResultBytes, 1)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	buffers.Result = result
	outputWeightPointer, outputWeightBytes, outputWeightEncoding, outputScale := hipTinyKernelOutputWeight(cfg)

	launchBytes, err := (hipTinyDecodeLaunchArgs{
		PriorKeyPointer:      buffers.PriorKeys.Pointer(),
		PriorValuePointer:    buffers.PriorValues.Pointer(),
		EmbeddingPointer:     cfg.EmbeddingPointer,
		OutputWeightPointer:  outputWeightPointer,
		LogitPointer:         buffers.Logits.Pointer(),
		AttentionPointer:     buffers.Attention.Pointer(),
		UpdatedKeyPointer:    buffers.UpdatedKeys.Pointer(),
		UpdatedValuePointer:  buffers.UpdatedValues.Pointer(),
		ResultPointer:        buffers.Result.Pointer(),
		TokenID:              tokenID,
		PriorTokenCount:      priorTokenCount,
		VocabSize:            cfg.VocabSize,
		HiddenSize:           cfg.HiddenSize,
		PriorKeyBytes:        buffers.PriorKeys.SizeBytes(),
		PriorValueBytes:      buffers.PriorValues.SizeBytes(),
		EmbeddingBytes:       cfg.EmbeddingBytes,
		OutputWeightBytes:    outputWeightBytes,
		LogitBytes:           buffers.Logits.SizeBytes(),
		AttentionBytes:       buffers.Attention.SizeBytes(),
		UpdatedKeyBytes:      buffers.UpdatedKeys.SizeBytes(),
		UpdatedValueBytes:    buffers.UpdatedValues.SizeBytes(),
		ResultBytes:          buffers.Result.SizeBytes(),
		OutputWeightEncoding: outputWeightEncoding,
		Q8Scale:              outputScale,
	}).Binary()
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameTinyDecode, launchBytes, 1)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipTinyDecodeResult{}, err
	}
	output, err := buffers.ReadOutput()
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	success = true
	if err := buffers.Close(); err != nil {
		return hipTinyDecodeResult{}, err
	}
	return output, nil
}

func (kernels hipNativeProjectionKernelSet) Generate(ctx context.Context, model *hipLoadedModel, prompt string, cfg inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	if err := hipContextErr(ctx); err != nil {
		return emptyTokenSeq, func() error { return err }
	}
	promptTokens, tokenPrompt, tokenPromptErr := hipGemma4Q4PromptTokenIDs(prompt, model)
	if tokenPromptErr != nil {
		return emptyTokenSeq, func() error { return tokenPromptErr }
	}
	if tokenPrompt {
		if model == nil {
			return emptyTokenSeq, func() error { return core.E(hipGemma4Q4Layer0Operation, "loaded model is required", nil) }
		}
		if model.modelInfo.NumLayers <= 0 {
			return emptyTokenSeq, func() error {
				return core.E(hipGemma4Q4Layer0Operation, "loaded Gemma4 q4 layer count is required", nil)
			}
		}
		q4Cfg, err := model.cachedGemma4Q4ForwardConfig(model.modelInfo.NumLayers)
		if err != nil {
			return emptyTokenSeq, func() error { return err }
		}
		return hipGemma4Q4GenerateTokenSeq(ctx, model, q4Cfg, promptTokens, cfg)
	}
	tinyCfg, err := model.loadedTinyLMConfig()
	if err != nil {
		return kernels.hipKernelStub.Generate(ctx, model, prompt, cfg)
	}
	return hipTinyGenerateSeq(ctx, model, tinyCfg, prompt, cfg)
}

func hipGemma4Q4PromptTokenIDs(prompt string, model *hipLoadedModel) ([]int32, bool, error) {
	promptTokens, tokenPrompt, err := hipGemma4Q4TokenPromptIDs(prompt, modelVocabSize(model))
	if err != nil || tokenPrompt {
		return promptTokens, tokenPrompt, err
	}
	return hipGemma4Q4TextPromptIDs(prompt, model)
}

func (kernels hipNativeProjectionKernelSet) Chat(ctx context.Context, model *hipLoadedModel, messages []inference.Message, cfg inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	if err := hipContextErr(ctx); err != nil {
		return emptyTokenSeq, func() error { return err }
	}
	if err := validateROCmChatMessages("rocm.hip.Chat", messages); err != nil {
		return emptyTokenSeq, func() error { return err }
	}
	prompt, err := model.ApplyChatTemplate(messages)
	if err != nil {
		return emptyTokenSeq, func() error { return err }
	}
	if _, ok, q4Err := model.loadedGemma4Q4PackageForwardConfig(); ok {
		if q4Err != nil {
			return emptyTokenSeq, func() error { return q4Err }
		}
		return kernels.Generate(ctx, model, "text:"+prompt, cfg)
	}
	return kernels.Generate(ctx, model, prompt, cfg)
}

func (kernels hipNativeProjectionKernelSet) Classify(ctx context.Context, model *hipLoadedModel, prompts []string, cfg inference.GenerateConfig) ([]inference.ClassifyResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if err := validateROCmPromptBatch("rocm.hip.Classify", prompts); err != nil {
		return nil, err
	}
	tinyCfg, err := model.loadedTinyLMConfig()
	if err != nil {
		classifier, hasClassifier, classifierErr := model.loadedSequenceClassifierConfig()
		if classifierErr != nil {
			return nil, classifierErr
		}
		if hasClassifier {
			return model.classifyWithSequenceClassifier(ctx, prompts, cfg, classifier)
		}
		if q4Cfg, ok, q4Err := model.loadedGemma4Q4PackageForwardConfig(); ok {
			if q4Err != nil {
				return nil, q4Err
			}
			return hipGemma4Q4Classify(ctx, model, q4Cfg, prompts, cfg)
		}
		return kernels.hipKernelStub.Classify(ctx, model, prompts, cfg)
	}
	results := make([]inference.ClassifyResult, len(prompts))
	for index, prompt := range prompts {
		tokens := model.Encode(prompt)
		output, err := hipRunLoadedTinyPrefill(ctx, model.driver, tinyCfg, tokens)
		if err != nil {
			return nil, err
		}
		output, err = model.applyTinyJANGTQOutputToPrefill(ctx, tinyCfg, output)
		if err != nil {
			return nil, err
		}
		output, err = model.applyTinyCodebookOutputToPrefill(ctx, tinyCfg, output)
		if err != nil {
			return nil, err
		}
		output, err = model.applyTinyLoRAToPrefill(ctx, tinyCfg, output)
		if err != nil {
			return nil, err
		}
		results[index] = inference.ClassifyResult{Token: hipTinyToken(model, int32(output.NextTokenID))}
		if cfg.ReturnLogits {
			results[index].Logits = output.Logits
		}
	}
	return results, nil
}

func hipGemma4Q4Classify(ctx context.Context, model *hipLoadedModel, q4Cfg hipGemma4Q4ForwardConfig, prompts []string, cfg inference.GenerateConfig) ([]inference.ClassifyResult, error) {
	results := make([]inference.ClassifyResult, len(prompts))
	for index, prompt := range prompts {
		tokens, err := hipGemma4Q4ClassifyPromptTokenIDs(prompt, model)
		if err != nil {
			return nil, err
		}
		prefill, err := hipRunGemma4Q4PackagePrefill(ctx, model, q4Cfg, hipPrefillRequest{TokenIDs: tokens})
		if err != nil {
			return nil, err
		}
		if err := prefill.Gemma4Q4DeviceState.Close(); err != nil {
			return nil, err
		}
		nextID, _, err := hipReferenceGreedySample(prefill.Logits)
		if err != nil {
			return nil, err
		}
		tokenID := int32(nextID)
		results[index] = inference.ClassifyResult{Token: inference.Token{ID: tokenID, Text: hipGeneratedTokenText(model, tokenID)}}
		if cfg.ReturnLogits {
			results[index].Logits = prefill.Logits
		}
	}
	return results, nil
}

func hipGemma4Q4ClassifyPromptTokenIDs(prompt string, model *hipLoadedModel) ([]int32, error) {
	tokens, tokenPrompt, err := hipGemma4Q4PromptTokenIDs(prompt, model)
	if err != nil {
		return nil, err
	}
	if tokenPrompt {
		return tokens, nil
	}
	return hipGemma4Q4TextPromptIDsRequired("text:"+prompt, model)
}

func hipGemma4Q4TextPromptIDsRequired(prompt string, model *hipLoadedModel) ([]int32, error) {
	tokens, ok, err := hipGemma4Q4TextPromptIDs(prompt, model)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, core.E(hipGemma4Q4Layer0Operation, "Gemma4 q4 text prompt is required", nil)
	}
	return tokens, nil
}

func (kernels hipNativeProjectionKernelSet) BatchGenerate(ctx context.Context, model *hipLoadedModel, prompts []string, cfg inference.GenerateConfig) ([]inference.BatchResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if err := validateROCmPromptBatch("rocm.hip.BatchGenerate", prompts); err != nil {
		return nil, err
	}
	if q4Cfg, ok, q4Err := model.loadedGemma4Q4PackageForwardConfig(); ok {
		if q4Err != nil {
			return nil, q4Err
		}
		return hipGemma4Q4BatchGenerate(ctx, model, q4Cfg, prompts, cfg), nil
	}
	tinyCfg, err := model.loadedTinyLMConfig()
	if err != nil {
		return kernels.hipKernelStub.BatchGenerate(ctx, model, prompts, cfg)
	}
	results := make([]inference.BatchResult, len(prompts))
	for index, prompt := range prompts {
		stream, streamErr := hipTinyGenerateSeq(ctx, model, tinyCfg, prompt, cfg)
		for token := range stream {
			results[index].Tokens = append(results[index].Tokens, token)
		}
		results[index].Err = streamErr()
	}
	return results, nil
}

func hipGemma4Q4BatchGenerate(ctx context.Context, model *hipLoadedModel, q4Cfg hipGemma4Q4ForwardConfig, prompts []string, cfg inference.GenerateConfig) []inference.BatchResult {
	results := make([]inference.BatchResult, len(prompts))
	for index, prompt := range prompts {
		promptTokens, tokenPrompt, err := hipGemma4Q4PromptTokenIDs(prompt, model)
		if err != nil {
			results[index].Err = err
			continue
		}
		if !tokenPrompt {
			results[index].Err = hipKernelNotLinkedError("rocm.hip.BatchGenerate", hipKernelDecode, model.kernelSet().Status())
			continue
		}
		stream, streamErr := hipGemma4Q4GenerateTokenSeq(ctx, model, q4Cfg, promptTokens, cfg)
		for token := range stream {
			results[index].Tokens = append(results[index].Tokens, token)
		}
		results[index].Err = streamErr()
	}
	return results
}

func (kernels hipNativeProjectionKernelSet) Prefill(ctx context.Context, model *hipLoadedModel, req hipPrefillRequest) (hipPrefillResult, error) {
	tinyCfg, err := model.loadedTinyLMConfig()
	if err != nil {
		if q4Cfg, ok, q4Err := model.loadedGemma4Q4PackageForwardConfig(); ok {
			if q4Err != nil {
				return hipPrefillResult{}, q4Err
			}
			return hipRunGemma4Q4PackagePrefill(ctx, model, q4Cfg, req)
		}
		return kernels.hipKernelStub.Prefill(ctx, model, req)
	}
	if err := req.validate(); err != nil {
		return hipPrefillResult{}, err
	}
	tokens, err := req.resolvedTokenIDs(model)
	if err != nil {
		return hipPrefillResult{}, err
	}
	mode, keyWidth, valueWidth, err := hipTinyKVConfig(req, tinyCfg.HiddenSize)
	if err != nil {
		return hipPrefillResult{}, err
	}
	output, err := hipRunLoadedTinyPrefill(ctx, model.driver, tinyCfg, tokens)
	if err != nil {
		return hipPrefillResult{}, err
	}
	output, err = model.applyTinyJANGTQOutputToPrefill(ctx, tinyCfg, output)
	if err != nil {
		return hipPrefillResult{}, err
	}
	output, err = model.applyTinyCodebookOutputToPrefill(ctx, tinyCfg, output)
	if err != nil {
		return hipPrefillResult{}, err
	}
	output, err = model.applyTinyLoRAToPrefill(ctx, tinyCfg, output)
	if err != nil {
		return hipPrefillResult{}, err
	}
	cache, err := newROCmKVCache(mode, defaultROCmKVBlockSize)
	if err != nil {
		return hipPrefillResult{}, err
	}
	if err := cache.AppendVectors(0, keyWidth, valueWidth, output.StateKeys, output.StateValues); err != nil {
		return hipPrefillResult{}, err
	}
	labels := hipTinyPrefillLabels(mode, keyWidth, valueWidth, len(tokens))
	hipAddTinyJANGTQOutputLabels(labels, tinyCfg)
	hipAddTinyCodebookOutputLabels(labels, tinyCfg)
	model.addTinyLoRALabels(labels)
	deviceKV, descriptorTable, err := hipMirrorTinyKV(model.driver, cache, labels)
	if err != nil {
		return hipPrefillResult{}, err
	}
	return hipPrefillResult{
		Logits:          output.Logits,
		PromptTokens:    len(tokens),
		KV:              cache,
		DeviceKV:        deviceKV,
		DescriptorTable: descriptorTable,
		Labels:          labels,
	}, nil
}

func (kernels hipNativeProjectionKernelSet) Decode(ctx context.Context, model *hipLoadedModel, req hipDecodeRequest) (hipDecodeResult, error) {
	if q4Cfg, ok, q4Err := model.loadedGemma4Q4PackageForwardConfig(); ok {
		if q4Err != nil {
			return hipDecodeResult{}, q4Err
		}
		return hipRunGemma4Q4PackageDecode(ctx, model, q4Cfg, req)
	}
	if smallCfg, err := model.loadedSmallDecodeConfig(); err == nil {
		return hipRunLoadedSmallDecodeToken(ctx, model, smallCfg, req)
	}
	tinyCfg, err := model.loadedTinyLMConfig()
	if err != nil {
		return kernels.hipKernelStub.Decode(ctx, model, req)
	}
	if err := req.validate(); err != nil {
		return hipDecodeResult{}, err
	}
	priorKeys, priorValues, err := req.KV.Restore(0, req.KV.TokenCount())
	if err != nil {
		return hipDecodeResult{}, err
	}
	output, err := hipRunLoadedTinyDecode(ctx, model.driver, tinyCfg, req.TokenID, priorKeys, priorValues)
	if err != nil {
		return hipDecodeResult{}, err
	}
	output, err = model.applyTinyJANGTQOutputToDecode(ctx, tinyCfg, output)
	if err != nil {
		return hipDecodeResult{}, err
	}
	output, err = model.applyTinyCodebookOutputToDecode(ctx, tinyCfg, output)
	if err != nil {
		return hipDecodeResult{}, err
	}
	output, err = model.applyTinyLoRAToDecode(ctx, tinyCfg, output)
	if err != nil {
		return hipDecodeResult{}, err
	}
	targetKV := req.KV
	if req.DeviceKV != nil {
		cloned, err := req.KV.Clone()
		if err != nil {
			return hipDecodeResult{}, err
		}
		targetKV = cloned
	}
	keyStart := len(output.UpdatedKeys) - tinyCfg.HiddenSize
	valueStart := len(output.UpdatedValues) - tinyCfg.HiddenSize
	if err := targetKV.AppendToken(targetKV.TokenCount(), output.UpdatedKeys[keyStart:], output.UpdatedValues[valueStart:]); err != nil {
		return hipDecodeResult{}, err
	}
	labels := map[string]string{
		"decode_kernel":            hipKernelStatusLinked,
		"decode_kernel_name":       hipKernelNameTinyDecode,
		"decode_launch_args_bytes": core.Sprintf("%d", hipTinyDecodeLaunchArgsBytes),
		"decode_launch_token":      core.Sprintf("%d", req.TokenID),
	}
	hipAddTinyJANGTQOutputLabels(labels, tinyCfg)
	hipAddTinyCodebookOutputLabels(labels, tinyCfg)
	model.addTinyLoRALabels(labels)
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
		Token:           hipTinyToken(model, int32(output.NextTokenID)),
		Logits:          output.Logits,
		KV:              targetKV,
		DeviceKV:        deviceKV,
		DescriptorTable: descriptorTable,
		Labels:          labels,
	}, nil
}

func hipTinyGenerateSeq(ctx context.Context, model *hipLoadedModel, cfg hipLoadedTinyLMConfig, prompt string, generate inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	var runErr error
	return func(yield func(inference.Token) bool) {
		if err := hipContextErr(ctx); err != nil {
			runErr = err
			return
		}
		if generate.MaxTokens <= 0 {
			return
		}
		tokens := model.Encode(prompt)
		output, err := hipRunLoadedTinyPrefill(ctx, model.driver, cfg, tokens)
		if err != nil {
			runErr = err
			return
		}
		output, err = model.applyTinyJANGTQOutputToPrefill(ctx, cfg, output)
		if err != nil {
			runErr = err
			return
		}
		output, err = model.applyTinyCodebookOutputToPrefill(ctx, cfg, output)
		if err != nil {
			runErr = err
			return
		}
		output, err = model.applyTinyLoRAToPrefill(ctx, cfg, output)
		if err != nil {
			runErr = err
			return
		}
		nextID := int32(output.NextTokenID)
		keys := output.StateKeys
		values := output.StateValues
		for generated := 0; generated < generate.MaxTokens; generated++ {
			if err := hipContextErr(ctx); err != nil {
				runErr = err
				return
			}
			token := hipTinyToken(model, nextID)
			if !yield(token) {
				return
			}
			if hipTokenIsStop(nextID, generate.StopTokens) {
				return
			}
			if generated == generate.MaxTokens-1 {
				return
			}
			decoded, err := hipRunLoadedTinyDecode(ctx, model.driver, cfg, nextID, keys, values)
			if err != nil {
				runErr = err
				return
			}
			decoded, err = model.applyTinyJANGTQOutputToDecode(ctx, cfg, decoded)
			if err != nil {
				runErr = err
				return
			}
			decoded, err = model.applyTinyCodebookOutputToDecode(ctx, cfg, decoded)
			if err != nil {
				runErr = err
				return
			}
			decoded, err = model.applyTinyLoRAToDecode(ctx, cfg, decoded)
			if err != nil {
				runErr = err
				return
			}
			nextID = int32(decoded.NextTokenID)
			keys = decoded.UpdatedKeys
			values = decoded.UpdatedValues
		}
	}, func() error { return runErr }
}

func hipGemma4Q4GenerateTokenSeq(ctx context.Context, model *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, promptTokens []int32, generate inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	var runErr error
	return func(yield func(inference.Token) bool) {
		if err := hipContextErr(ctx); err != nil {
			runErr = err
			return
		}
		if generate.MaxTokens <= 0 {
			return
		}
		if model == nil {
			runErr = core.E(hipGemma4Q4Layer0Operation, "loaded model is required", nil)
			return
		}
		req := hipGemma4Q4GreedyDecodeRequest{
			PromptTokenIDs: promptTokens,
			MaxNewTokens:   generate.MaxTokens,
			Position:       0,
			Epsilon:        1e-6,
		}
		if err := cfg.validate(); err != nil {
			runErr = err
			return
		}
		suppressTokens := hipGemma4Q4GenerationSuppressTokenIDs(model, generate.StopTokens)
		hostSampling := hipGemma4Q4HostSamplingRequested(generate)
		if err := req.validate(cfg); err != nil {
			runErr = err
			return
		}
		ubatchTokens, err := hipGemma4Q4PrefillUBatchTokens()
		if err != nil {
			runErr = err
			return
		}
		prefillPlan, err := hipGemma4Q4PlanPromptPrefill(promptTokens, req.Position, ubatchTokens)
		if err != nil {
			runErr = err
			return
		}
		deviceKVMode, err := hipGemma4Q4GenerateDeviceKVMode()
		if err != nil {
			runErr = err
			return
		}
		finalGreedyBuffer, err := hipAllocateByteBuffer(model.driver, "rocm.hip.Gemma4Q4Generate", "Gemma4 q4 final greedy result", hipMLXQ4ProjectionBestBytes, 1)
		if err != nil {
			runErr = err
			return
		}
		defer func() {
			if err := finalGreedyBuffer.Close(); err != nil && runErr == nil {
				runErr = err
			}
		}()
		var attentionWorkspace *hipAttentionHeadsChunkedWorkspace
		if hipGemma4Q4ChunkedAttentionEnabled(len(promptTokens)) {
			attentionWorkspace = &hipAttentionHeadsChunkedWorkspace{}
			defer func() {
				if err := attentionWorkspace.Close(); err != nil && runErr == nil {
					runErr = err
				}
			}()
		}
		state := hipGemma4Q4DecodeState{}
		var deviceState *hipGemma4Q4DeviceDecodeState
		defer func() {
			if err := deviceState.Close(); err != nil && runErr == nil {
				runErr = err
			}
		}()
		position := req.Position
		var current hipGemma4Q4ForwardResult
		haveCurrent := false
		var history []int32
		if hostSampling {
			history = make([]int32, 0, generate.MaxTokens)
		}
		useBatchedPrefill := hipGemma4Q4CanUseBatchedGeneratePrefill(cfg) && !hostSampling
		for _, ubatch := range prefillPlan.Batches {
			if !useBatchedPrefill {
				for index, promptToken := range ubatch.Tokens {
					if err := hipContextErr(ctx); err != nil {
						runErr = err
						return
					}
					outputToken := ubatch.OutputTokens[index]
					var err error
					current, state, err = hipRunGemma4Q4SingleTokenForwardWithStateInternal(ctx, model.driver, cfg, state, hipGemma4Q4ForwardRequest{
						TokenID:            promptToken,
						Position:           ubatch.Position + index,
						Epsilon:            req.Epsilon,
						DeviceKVAttention:  true,
						DeviceKVMode:       deviceKVMode,
						PriorDeviceState:   deviceState,
						ReturnDeviceState:  true,
						DeviceFinalSample:  outputToken && !hostSampling,
						SkipFinalSample:    !outputToken,
						FinalGreedyBuffer:  finalGreedyBuffer,
						SuppressTokens:     suppressTokens,
						AttentionWorkspace: attentionWorkspace,
						OmitDebugTensors:   true,
						OmitLabels:         true,
						OmitHostState:      true,
					}, false)
					if err != nil {
						runErr = err
						return
					}
					if current.DeviceState == nil {
						runErr = core.E(hipGemma4Q4Layer0Operation, "forward did not return device KV state", nil)
						return
					}
					previousDeviceState := deviceState
					deviceState = current.DeviceState
					current.DeviceState = nil
					hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
					if outputToken {
						if hostSampling {
							current.Greedy, err = hipGemma4Q4HostSampleResult(current.Logits, generate, suppressTokens, history, rand.Float64())
							if err != nil {
								runErr = err
								return
							}
						}
						haveCurrent = true
					}
				}
				continue
			}
			if err := hipContextErr(ctx); err != nil {
				runErr = err
				return
			}
			var priorLayerKV []*rocmDeviceKVCache
			if deviceState != nil {
				priorLayerKV = make([]*rocmDeviceKVCache, len(cfg.Layers))
				for index := range priorLayerKV {
					priorLayerKV[index] = deviceState.layerCache(index)
				}
			}
			forward, err := hipRunGemma4Q4PrefillForwardBatchWithPrior(ctx, model.driver, cfg, ubatch.Tokens, ubatch.Position, req.Epsilon, deviceKVMode, priorLayerKV, nil, ubatch.OutputTokens, finalGreedyBuffer)
			if err != nil {
				runErr = err
				return
			}
			if len(forward.Greedy) > 0 {
				greedyOut := forward.Greedy[len(forward.Greedy)-1]
				current.Greedy = greedyOut.Greedy
				if hipTokenIsSuppressed(int32(current.Greedy.TokenID), suppressTokens) {
					last := cfg.Layers[len(cfg.Layers)-1]
					current.Greedy, err = hipRunGemma4Q4PrefillFinalGreedyForRowSuppress(ctx, model.driver, last, forward.FinalHidden, len(ubatch.Tokens), greedyOut.Row, req.Epsilon, finalGreedyBuffer, suppressTokens)
					if err != nil {
						_ = forward.Close()
						runErr = err
						return
					}
				}
				haveCurrent = true
			}
			nextDeviceState, err := hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward, deviceKVMode)
			closeErr := forward.Close()
			if err != nil {
				runErr = err
				return
			}
			if closeErr != nil {
				_ = nextDeviceState.Close()
				runErr = closeErr
				return
			}
			previousDeviceState := deviceState
			if err := hipFinalizeGemma4Q4ForwardDeviceState(previousDeviceState, nextDeviceState); err != nil {
				_ = nextDeviceState.Close()
				runErr = err
				return
			}
			deviceState = nextDeviceState
			hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
		}
		if !haveCurrent {
			runErr = core.E(hipGemma4Q4Layer0Operation, "prefill did not produce a final greedy token", nil)
			return
		}
		position = prefillPlan.NextPosition()
		for generated := 0; generated < generate.MaxTokens; generated++ {
			if err := hipContextErr(ctx); err != nil {
				runErr = err
				return
			}
			tokenID := int32(current.Greedy.TokenID)
			if hipTokenIsStop(tokenID, generate.StopTokens) {
				return
			}
			token := inference.Token{
				ID:   tokenID,
				Text: hipGeneratedTokenText(model, tokenID),
			}
			if !yield(token) {
				return
			}
			if hostSampling {
				history = append(history, tokenID)
			}
			if generated == generate.MaxTokens-1 {
				return
			}
			var tokenIDDeviceBuffer *hipDeviceByteBuffer
			if !hostSampling {
				tokenIDDeviceBuffer = finalGreedyBuffer
			}
			var err error
			current, state, err = hipRunGemma4Q4SingleTokenForwardWithStateInternal(ctx, model.driver, cfg, state, hipGemma4Q4ForwardRequest{
				TokenID:             tokenID,
				Position:            position,
				Epsilon:             req.Epsilon,
				DeviceKVAttention:   true,
				DeviceKVMode:        deviceKVMode,
				PriorDeviceState:    deviceState,
				ReturnDeviceState:   true,
				DeviceFinalSample:   !hostSampling,
				FinalGreedyBuffer:   finalGreedyBuffer,
				TokenIDDeviceBuffer: tokenIDDeviceBuffer,
				SuppressTokens:      suppressTokens,
				AttentionWorkspace:  attentionWorkspace,
				OmitDebugTensors:    true,
				OmitLabels:          true,
				OmitHostState:       true,
			}, false)
			if err != nil {
				runErr = err
				return
			}
			if current.DeviceState == nil {
				runErr = core.E(hipGemma4Q4Layer0Operation, "forward did not return device KV state", nil)
				return
			}
			previousDeviceState := deviceState
			deviceState = current.DeviceState
			current.DeviceState = nil
			hipReleaseClosedGemma4Q4DeviceDecodeState(previousDeviceState)
			if hostSampling {
				current.Greedy, err = hipGemma4Q4HostSampleResult(current.Logits, generate, suppressTokens, history, rand.Float64())
				if err != nil {
					runErr = err
					return
				}
			}
			position++
		}
	}, func() error { return runErr }
}

func hipGemma4Q4GenerateDeviceKVMode() (string, error) {
	raw := strings.ToLower(strings.TrimSpace(os.Getenv("GO_ROCM_GEMMA4_Q4_DEVICE_KV_MODE")))
	if raw == "" {
		return rocmKVCacheModeKQ8VQ4, nil
	}
	if !isROCmKVCacheMode(raw) {
		return "", core.E(hipGemma4Q4Layer0Operation, core.Sprintf("unsupported Gemma4 q4 device KV cache mode %q", raw), nil)
	}
	return raw, nil
}

const hipGemma4Q4ChunkedAttentionAutoPromptTokens = 4000

func hipGemma4Q4ChunkedAttentionEnabled(promptTokens int) bool {
	raw := strings.ToLower(strings.TrimSpace(os.Getenv("GO_ROCM_GEMMA4_Q4_CHUNKED_ATTENTION")))
	switch raw {
	case "1", "true", "yes", "on":
		return true
	case "auto":
		return promptTokens >= hipGemma4Q4ChunkedAttentionAutoPromptTokens
	case "0", "false", "no", "off":
		return false
	case "":
		return true
	default:
		return false
	}
}

func hipGemma4Q4TokenPromptIDs(prompt string, vocabSize int) ([]int32, bool, error) {
	const prefix = "tokens:"
	trimmed := strings.TrimSpace(prompt)
	if !strings.HasPrefix(strings.ToLower(trimmed), prefix) {
		return nil, false, nil
	}
	body := strings.TrimSpace(trimmed[len(prefix):])
	if body == "" {
		return nil, true, core.E(hipGemma4Q4Layer0Operation, "token prompt must contain at least one token ID", nil)
	}
	parts := strings.Split(body, ",")
	tokens := make([]int32, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil, true, core.E(hipGemma4Q4Layer0Operation, "token prompt contains an empty token ID", nil)
		}
		value, err := strconv.Atoi(part)
		if err != nil || value < 0 || (vocabSize > 0 && value >= vocabSize) {
			return nil, true, core.E(hipGemma4Q4Layer0Operation, core.Sprintf("token prompt ID %q is outside vocabulary", part), nil)
		}
		tokens = append(tokens, int32(value))
	}
	return tokens, true, nil
}

func hipGemma4Q4TextPromptIDs(prompt string, model *hipLoadedModel) ([]int32, bool, error) {
	const prefix = "text:"
	leftTrimmed := strings.TrimLeft(prompt, " \t\r\n\v\f")
	prefixed := strings.HasPrefix(strings.ToLower(leftTrimmed), prefix)
	if !prefixed && !hipGemma4Q4ExperimentalTextGenerateEnabled(model) {
		return nil, false, nil
	}
	body := prompt
	if prefixed {
		body = leftTrimmed[len(prefix):]
	}
	if strings.TrimSpace(body) == "" {
		return nil, true, core.E(hipGemma4Q4Layer0Operation, "text prompt must contain prompt text", nil)
	}
	if model == nil {
		return nil, true, core.E(hipGemma4Q4Layer0Operation, "loaded model is required", nil)
	}
	tokens := model.Encode(body)
	if len(tokens) == 0 {
		return nil, true, core.E(hipGemma4Q4Layer0Operation, "text prompt produced no token IDs", nil)
	}
	return tokens, true, nil
}

func hipGemma4Q4ExperimentalTextGenerateEnabled(model *hipLoadedModel) bool {
	if model == nil ||
		!isROCmGemma4Architecture(model.modelInfo.Architecture) ||
		model.modelInfo.QuantBits != 4 {
		return false
	}
	raw := strings.ToLower(strings.TrimSpace(os.Getenv("GO_ROCM_GEMMA4_Q4_EXPERIMENTAL_TEXT_GENERATE")))
	return raw == "1" || raw == "true" || raw == "yes"
}

func modelVocabSize(model *hipLoadedModel) int {
	if model == nil {
		return 0
	}
	return model.modelInfo.VocabSize
}

func (model *hipLoadedModel) applyTinyJANGTQOutputToPrefill(ctx context.Context, cfg hipLoadedTinyLMConfig, output hipTinyPrefillResult) (hipTinyPrefillResult, error) {
	if !hipTinyUsesJANGTQOutput(cfg) {
		return output, nil
	}
	hidden, err := hipTinyAttentionWeightedOutput(output.StateValues, output.Attention, cfg.HiddenSize)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	logits, next, score, err := model.runTinyJANGTQOutputProjection(ctx, cfg, hidden)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	output.Logits = logits
	output.NextTokenID = next
	output.NextScore = score
	return output, nil
}

func (model *hipLoadedModel) applyTinyJANGTQOutputToDecode(ctx context.Context, cfg hipLoadedTinyLMConfig, output hipTinyDecodeResult) (hipTinyDecodeResult, error) {
	if !hipTinyUsesJANGTQOutput(cfg) {
		return output, nil
	}
	hidden, err := hipTinyAttentionWeightedOutput(output.UpdatedValues, output.Attention, cfg.HiddenSize)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	logits, next, score, err := model.runTinyJANGTQOutputProjection(ctx, cfg, hidden)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	output.Logits = logits
	output.NextTokenID = next
	output.NextScore = score
	return output, nil
}

func (model *hipLoadedModel) applyTinyCodebookOutputToPrefill(ctx context.Context, cfg hipLoadedTinyLMConfig, output hipTinyPrefillResult) (hipTinyPrefillResult, error) {
	if !hipTinyUsesCodebookOutput(cfg) {
		return output, nil
	}
	hidden, err := hipTinyAttentionWeightedOutput(output.StateValues, output.Attention, cfg.HiddenSize)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	logits, next, score, err := model.runTinyCodebookOutputProjection(ctx, cfg, hidden)
	if err != nil {
		return hipTinyPrefillResult{}, err
	}
	output.Logits = logits
	output.NextTokenID = next
	output.NextScore = score
	return output, nil
}

func (model *hipLoadedModel) applyTinyCodebookOutputToDecode(ctx context.Context, cfg hipLoadedTinyLMConfig, output hipTinyDecodeResult) (hipTinyDecodeResult, error) {
	if !hipTinyUsesCodebookOutput(cfg) {
		return output, nil
	}
	hidden, err := hipTinyAttentionWeightedOutput(output.UpdatedValues, output.Attention, cfg.HiddenSize)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	logits, next, score, err := model.runTinyCodebookOutputProjection(ctx, cfg, hidden)
	if err != nil {
		return hipTinyDecodeResult{}, err
	}
	output.Logits = logits
	output.NextTokenID = next
	output.NextScore = score
	return output, nil
}

func (model *hipLoadedModel) runTinyJANGTQOutputProjection(ctx context.Context, cfg hipLoadedTinyLMConfig, hidden []float32) ([]float32, int, float32, error) {
	if model == nil {
		return nil, 0, 0, core.E("rocm.hip.TinyJANGTQ", "loaded model is required", nil)
	}
	packed, err := model.loadedTensorBytes("rocm.hip.TinyJANGTQ", "JANGTQ output weights", cfg.OutputWeightPointer, cfg.OutputWeightBytes)
	if err != nil {
		return nil, 0, 0, err
	}
	logits, err := hipRunJANGTQProjectionKernel(ctx, model.driver, hipJANGTQProjectionRequest{
		Input:         hidden,
		PackedWeights: packed,
		Descriptor:    cfg.OutputJANGTQDescriptor,
		Rows:          cfg.VocabSize,
		Cols:          cfg.HiddenSize,
		Scale:         cfg.OutputJANGTQScale,
	})
	if err != nil {
		return nil, 0, 0, err
	}
	next, score, err := hipReferenceGreedySample(logits)
	if err != nil {
		return nil, 0, 0, err
	}
	return logits, next, score, nil
}

func (model *hipLoadedModel) runTinyCodebookOutputProjection(ctx context.Context, cfg hipLoadedTinyLMConfig, hidden []float32) ([]float32, int, float32, error) {
	if model == nil {
		return nil, 0, 0, core.E("rocm.hip.TinyCodebook", "loaded model is required", nil)
	}
	codes, err := model.loadedTensorBytes("rocm.hip.TinyCodebook", "codebook output codes", cfg.OutputWeightPointer, cfg.OutputWeightBytes)
	if err != nil {
		return nil, 0, 0, err
	}
	codebook, err := model.loadedF32TensorPayload("rocm.hip.TinyCodebook", "codebook output table", cfg.OutputCodebookPointer, cfg.OutputCodebookBytes, cfg.OutputCodebookCount*cfg.OutputCodebookDim)
	if err != nil {
		return nil, 0, 0, err
	}
	expanded, err := hipRunCodebookLookupKernel(ctx, model.driver, hipCodebookLookupRequest{
		Codes:    codes,
		Codebook: codebook,
		CodeDim:  cfg.OutputCodebookDim,
	})
	if err != nil {
		return nil, 0, 0, err
	}
	logits, err := hipRunProjectionKernel(ctx, model.driver, hipProjectionRequest{
		Input: hidden,
		F32:   expanded,
		Rows:  cfg.VocabSize,
		Cols:  cfg.HiddenSize,
	})
	if err != nil {
		return nil, 0, 0, err
	}
	next, score, err := hipReferenceGreedySample(logits)
	if err != nil {
		return nil, 0, 0, err
	}
	return logits, next, score, nil
}

func hipAddTinyJANGTQOutputLabels(labels map[string]string, cfg hipLoadedTinyLMConfig) {
	if labels == nil || !hipTinyUsesJANGTQOutput(cfg) {
		return
	}
	labels["output_projection_kernel"] = hipKernelStatusLinked
	labels["output_projection_kernel_name"] = hipKernelNameJANGTQ
	labels["output_weight_encoding"] = "jangtq"
	labels["output_jangtq_bits"] = core.Sprintf("%d", cfg.OutputJANGTQDescriptor.Bits)
	labels["output_jangtq_group_size"] = core.Sprintf("%d", cfg.OutputJANGTQDescriptor.GroupSize)
	labels["output_jangtq_scale"] = core.Sprintf("%.6g", cfg.OutputJANGTQScale)
}

func hipAddTinyCodebookOutputLabels(labels map[string]string, cfg hipLoadedTinyLMConfig) {
	if labels == nil || !hipTinyUsesCodebookOutput(cfg) {
		return
	}
	labels["output_lookup_kernel"] = hipKernelStatusLinked
	labels["output_lookup_kernel_name"] = hipKernelNameCodebook
	labels["output_projection_kernel"] = hipKernelStatusLinked
	labels["output_projection_kernel_name"] = hipKernelNameProjection
	labels["output_weight_encoding"] = "codebook"
	labels["output_codebook_entries"] = core.Sprintf("%d", cfg.OutputCodebookCount)
	labels["output_codebook_dim"] = core.Sprintf("%d", cfg.OutputCodebookDim)
}

func hipTinyKVConfig(req hipPrefillRequest, hiddenSize int) (string, int, int, error) {
	mode := firstNonEmptyString(req.CacheMode, rocmKVCacheModeFP16)
	if !isROCmKVCacheMode(mode) {
		return "", 0, 0, core.E("rocm.hip.TinyLoadedModel", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
	keyWidth, valueWidth := hiddenSize, hiddenSize
	if req.KeyWidth > 0 || req.ValueWidth > 0 {
		var err error
		keyWidth, valueWidth, err = hipKVVectorWidths(req.KeyWidth, req.ValueWidth)
		if err != nil {
			return "", 0, 0, err
		}
		if keyWidth != hiddenSize || valueWidth != hiddenSize {
			return "", 0, 0, core.E("rocm.hip.TinyLoadedModel", "tiny loaded path requires KV widths to match hidden size", nil)
		}
	}
	return mode, keyWidth, valueWidth, nil
}

func hipTinyPrefillLabels(mode string, keyWidth, valueWidth, tokenCount int) map[string]string {
	return map[string]string{
		"kv_cache_mode":             mode,
		"kv_key_width":              core.Sprintf("%d", keyWidth),
		"kv_value_width":            core.Sprintf("%d", valueWidth),
		"prefill_kernel":            hipKernelStatusLinked,
		"prefill_kernel_name":       hipKernelNameTinyPrefill,
		"prefill_launch_args_bytes": core.Sprintf("%d", hipTinyPrefillLaunchArgsBytes),
		"prefill_launch_tokens":     core.Sprintf("%d", tokenCount),
	}
}

func hipGeneratedTokenText(model *hipLoadedModel, tokenID int32) string {
	if model != nil && model.tokenText != nil {
		if text := model.tokenText.DecodeToken(tokenID); text != "" {
			return text
		}
	}
	return core.Sprintf("<token:%d>", tokenID)
}

func hipMirrorTinyKV(driver nativeHIPDriver, cache *rocmKVCache, labels map[string]string) (*rocmDeviceKVCache, *rocmDeviceKVDescriptorTable, error) {
	device, err := cache.MirrorToDevice(driver)
	if err != nil {
		return nil, nil, err
	}
	table, err := device.KernelDescriptorTable()
	if err != nil {
		_ = device.Close()
		return nil, nil, err
	}
	for key, value := range device.Stats().Labels {
		labels[key] = value
	}
	hipAddDescriptorTableLabels(labels, table)
	return device, table, nil
}

func hipAppendDecodeDeviceKV(req hipDecodeRequest, key, value []float32, labels map[string]string) (*rocmDeviceKVCache, *rocmDeviceKVDescriptorTable, error) {
	if req.DeviceKV == nil {
		return nil, nil, nil
	}
	sourcePageCount := req.DeviceKV.PageCount()
	sourceTokenCount := req.DeviceKV.TokenCount()
	device, err := req.DeviceKV.withAppendedToken(key, value)
	if err != nil {
		return nil, nil, err
	}
	table, err := device.KernelDescriptorTable()
	if err != nil {
		_ = device.closePagesFrom(sourcePageCount)
		return nil, nil, err
	}
	if err := req.DeviceKV.transferPagesTo(device); err != nil {
		_ = table.Close()
		_ = device.closePagesFrom(sourcePageCount)
		return nil, nil, err
	}
	if req.DescriptorTable != nil {
		_ = req.DescriptorTable.Close()
	}
	for key, value := range device.Stats().Labels {
		labels[key] = value
	}
	hipAddDescriptorTableLabels(labels, table)
	labels["kv_device_update"] = "append_token"
	labels["kv_device_update_pages"] = "1"
	labels["kv_device_update_from_pages"] = core.Sprintf("%d", sourcePageCount)
	labels["kv_device_update_from_tokens"] = core.Sprintf("%d", sourceTokenCount)
	labels["kv_device_update_to_pages"] = core.Sprintf("%d", device.PageCount())
	labels["kv_device_update_to_tokens"] = core.Sprintf("%d", device.TokenCount())
	labels["kv_device_update_descriptor_refresh"] = "success"
	return device, table, nil
}

func hipAddDescriptorTableLabels(labels map[string]string, table *rocmDeviceKVDescriptorTable) {
	if labels == nil || table == nil {
		return
	}
	labels["kv_descriptor_bytes"] = core.Sprintf("%d", table.SizeBytes())
	labels["kv_descriptor_pages"] = core.Sprintf("%d", table.pageCount)
	labels["kv_descriptor_table"] = "hip_device"
	labels["kv_descriptor_version"] = core.Sprintf("%d", table.version)
}

func hipValidateTinyTokenIDs(tokenIDs []int32, vocabSize int) error {
	if len(tokenIDs) == 0 {
		return core.E("rocm.hip.TinyLoadedModel", "token IDs are required", nil)
	}
	for _, id := range tokenIDs {
		if id < 0 || int(id) >= vocabSize {
			return core.E("rocm.hip.TinyLoadedModel", "token ID is outside vocabulary", nil)
		}
	}
	return nil
}

func hipTinyToken(model *hipLoadedModel, id int32) inference.Token {
	text := core.Sprintf("%d", id)
	if model != nil {
		if decoded := model.Decode([]int32{id}); decoded != "" {
			text = decoded
		}
	}
	return inference.Token{ID: id, Text: text}
}

func hipTokenIsStop(id int32, stopTokens []int32) bool {
	for _, stop := range stopTokens {
		if id == stop {
			return true
		}
	}
	return false
}

func hipTokenIsSuppressed(id int32, suppressTokens []int32) bool {
	for _, suppressed := range suppressTokens {
		if id == suppressed {
			return true
		}
	}
	return false
}

func hipGemma4Q4DefaultSuppressTokenIDs(model *hipLoadedModel) []int32 {
	if model == nil || model.tokenText == nil || !isROCmGemma4Architecture(model.modelInfo.Architecture) {
		return nil
	}
	model.q4ConfigMu.Lock()
	defer model.q4ConfigMu.Unlock()
	if len(model.q4Suppress) == 0 {
		model.q4Suppress = hipTokenTextIDs(model.tokenText, []string{
			"<pad>",
			"<bos>",
			"<unk>",
			"<mask>",
			"<|tool>",
			"<tool|>",
			"<|tool_call>",
			"<tool_call|>",
			"<|tool_response>",
			"<tool_response|>",
			`<|"|>`,
			"<|think|>",
			"<|channel>",
			"<channel|>",
			"<|turn>",
			"<|image>",
			"<|audio>",
			"<|image|>",
			"<|audio|>",
			"<image|>",
			"<audio|>",
			"<|video|>",
		})
	}
	return model.q4Suppress[:len(model.q4Suppress):len(model.q4Suppress)]
}

func hipGemma4Q4DefaultStopTokenIDs(model *hipLoadedModel) []int32 {
	if model == nil || model.tokenText == nil || !isROCmGemma4Architecture(model.modelInfo.Architecture) {
		return nil
	}
	model.q4ConfigMu.Lock()
	defer model.q4ConfigMu.Unlock()
	if len(model.q4Stop) == 0 {
		model.q4Stop = hipTokenTextIDs(model.tokenText, []string{
			"<eos>",
			"<turn|>",
			"<|tool_response>",
		})
	}
	return model.q4Stop[:len(model.q4Stop):len(model.q4Stop)]
}

func hipGemma4Q4GenerationSuppressTokenIDs(model *hipLoadedModel, stopTokens []int32) []int32 {
	suppressTokens := hipGemma4Q4DefaultSuppressTokenIDs(model)
	if len(stopTokens) > 0 {
		return suppressTokens
	}
	for _, id := range hipGemma4Q4DefaultStopTokenIDs(model) {
		if !hipTokenIsSuppressed(id, suppressTokens) {
			suppressTokens = append(suppressTokens, id)
		}
	}
	return suppressTokens
}

func hipGemma4Q4HostSamplingRequested(generate inference.GenerateConfig) bool {
	return generate.Temperature > 0 ||
		generate.TopK > 0 ||
		generate.TopP > 0 ||
		generate.RepeatPenalty > 1
}

func hipGemma4Q4HostSampleResult(logits []float32, generate inference.GenerateConfig, suppressTokens []int32, history []int32, draw float64) (hipGreedySampleResult, error) {
	if len(logits) == 0 {
		return hipGreedySampleResult{}, core.E("rocm.hip.Gemma4Q4HostSampler", "logits are required", nil)
	}
	working := append([]float32(nil), logits...)
	for _, id := range suppressTokens {
		if id >= 0 && int(id) < len(working) {
			working[id] = float32(math.Inf(-1))
		}
	}
	if generate.RepeatPenalty > 1 {
		if math.IsNaN(float64(generate.RepeatPenalty)) || math.IsInf(float64(generate.RepeatPenalty), 0) {
			return hipGreedySampleResult{}, core.E("rocm.hip.Gemma4Q4HostSampler", "repeat penalty must be finite", nil)
		}
		for _, id := range history {
			if id < 0 || int(id) >= len(working) {
				continue
			}
			if working[id] < 0 {
				working[id] *= generate.RepeatPenalty
			} else {
				working[id] /= generate.RepeatPenalty
			}
		}
	}
	if generate.Temperature <= 0 && generate.TopK <= 0 && generate.TopP <= 0 {
		tokenID, score, err := hipReferenceGreedySampleSuppress(working, nil)
		if err != nil {
			return hipGreedySampleResult{}, err
		}
		return hipGreedySampleResult{TokenID: tokenID, Score: score}, nil
	}
	temperature := generate.Temperature
	if temperature == 0 {
		temperature = 1
	}
	if temperature <= 0 || math.IsNaN(float64(temperature)) || math.IsInf(float64(temperature), 0) {
		return hipGreedySampleResult{}, core.E("rocm.hip.Gemma4Q4HostSampler", "temperature must be positive and finite", nil)
	}
	topP := generate.TopP
	if topP == 0 {
		topP = 1
	}
	if topP <= 0 || topP > 1 || math.IsNaN(float64(topP)) || math.IsInf(float64(topP), 0) {
		return hipGreedySampleResult{}, core.E("rocm.hip.Gemma4Q4HostSampler", "top-p must be in (0, 1]", nil)
	}
	candidates := make([]hipReferenceCandidate, 0, len(working))
	for index, value := range working {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			continue
		}
		candidates = append(candidates, hipReferenceCandidate{index: index, value: value})
	}
	if len(candidates) == 0 {
		return hipGreedySampleResult{}, core.E("rocm.hip.Gemma4Q4HostSampler", "all logits are suppressed", nil)
	}
	sortHIPReferenceCandidates(candidates)
	topK := generate.TopK
	if topK < 0 || topK > len(working) {
		return hipGreedySampleResult{}, core.E("rocm.hip.Gemma4Q4HostSampler", "top-k must be within vocabulary size", nil)
	}
	if topK == 0 || topK > len(candidates) {
		topK = len(candidates)
	}
	candidates = candidates[:topK]
	maxValue := float64(candidates[0].value) / float64(temperature)
	weights := make([]float64, len(candidates))
	total := 0.0
	for index, candidate := range candidates {
		weight := math.Exp(float64(candidate.value)/float64(temperature) - maxValue)
		weights[index] = weight
		total += weight
	}
	if total <= 0 || math.IsNaN(total) || math.IsInf(total, 0) {
		return hipGreedySampleResult{}, core.E("rocm.hip.Gemma4Q4HostSampler", "sampling distribution is invalid", nil)
	}
	limit := len(candidates)
	if topP < 1 {
		cumulative := 0.0
		for index, weight := range weights {
			cumulative += weight
			if cumulative/total >= float64(topP) {
				limit = index + 1
				break
			}
		}
	}
	selectedTotal := 0.0
	for _, weight := range weights[:limit] {
		selectedTotal += weight
	}
	if draw < 0 {
		draw = 0
	}
	if draw >= 1 {
		draw = math.Nextafter(1, 0)
	}
	target := draw * selectedTotal
	cumulative := 0.0
	for index, weight := range weights[:limit] {
		cumulative += weight
		if target <= cumulative {
			candidate := candidates[index]
			return hipGreedySampleResult{TokenID: candidate.index, Score: candidate.value}, nil
		}
	}
	candidate := candidates[limit-1]
	return hipGreedySampleResult{TokenID: candidate.index, Score: candidate.value}, nil
}

func hipTokenTextIDs(decoder *hipTokenTextDecoder, texts []string) []int32 {
	if decoder == nil || len(texts) == 0 {
		return nil
	}
	ids := make([]int32, 0, len(texts))
	for _, text := range texts {
		id, ok := decoder.specialText[text]
		if !ok || hipTokenIsSuppressed(id, ids) {
			continue
		}
		ids = append(ids, id)
	}
	return ids
}

func hipContextErr(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}
