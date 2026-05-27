// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"strconv"
	"strings"

	core "dappco.re/go"
)

const (
	hipGemma4Q4PrefillDefaultUBatchTokens = 16
	hipGemma4Q4PrefillUBatchEnv           = "GO_ROCM_GEMMA4_Q4_PREFILL_UBATCH_TOKENS"
)

const (
	hipPerLayerInputTransposeLaunchArgsVersion uint32 = 1
	hipPerLayerInputTransposeLaunchArgsBytes          = 56
)

type hipPerLayerInputTransposeLaunchArgs struct {
	InputPointer  nativeDevicePointer
	OutputPointer nativeDevicePointer
	InputBytes    uint64
	OutputBytes   uint64
	Batch         int
	LayerCount    int
	InputSize     int
}

type hipGemma4Q4PrefillPlan struct {
	PromptTokens int
	StartPos     int
	UBatchTokens int
	OutputTokens int
	Batches      []hipGemma4Q4PrefillUBatch
}

func (plan hipGemma4Q4PrefillPlan) NextPosition() int {
	return plan.StartPos + plan.PromptTokens
}

type hipGemma4Q4PrefillUBatch struct {
	Start        int
	End          int
	Position     int
	Tokens       []int32
	OutputTokens []bool
}

func (batch hipGemma4Q4PrefillUBatch) OutputToken(index int) bool {
	return index >= 0 && index < len(batch.OutputTokens) && batch.OutputTokens[index]
}

type hipGemma4Q4PrefillQKVBatch struct {
	Query *hipDeviceByteBuffer
	Key   *hipDeviceByteBuffer
	Value *hipDeviceByteBuffer
}

func (batch *hipGemma4Q4PrefillQKVBatch) Close() error {
	if batch == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{batch.Value, batch.Key, batch.Query} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

type hipGemma4Q4PrefillRoPEQKBatch struct {
	Query *hipDeviceByteBuffer
	Key   *hipDeviceByteBuffer
}

func (batch *hipGemma4Q4PrefillRoPEQKBatch) Close() error {
	if batch == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{batch.Key, batch.Query} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

type hipGemma4Q4PrefillDeviceKVBatch struct {
	Cache           *rocmDeviceKVCache
	DescriptorTable *rocmDeviceKVDescriptorTable
	Launch          rocmDeviceKVLaunchDescriptor
}

func (batch *hipGemma4Q4PrefillDeviceKVBatch) Close() error {
	if batch == nil {
		return nil
	}
	var lastErr error
	if err := batch.DescriptorTable.Close(); err != nil {
		lastErr = err
	}
	if err := batch.Cache.Close(); err != nil {
		lastErr = err
	}
	return lastErr
}

type hipGemma4Q4PrefillLayerKVBatch struct {
	InputNorm *hipDeviceByteBuffer
	QKV       *hipGemma4Q4PrefillQKVBatch
	QK        *hipGemma4Q4PrefillRoPEQKBatch
	Value     *hipDeviceByteBuffer
	DeviceKV  *hipGemma4Q4PrefillDeviceKVBatch
	SharedKey *hipDeviceByteBuffer
	SharedVal *hipDeviceByteBuffer
}

func (batch *hipGemma4Q4PrefillLayerKVBatch) Close() error {
	if batch == nil {
		return nil
	}
	var lastErr error
	if err := batch.DeviceKV.Close(); err != nil {
		lastErr = err
	}
	for _, buffer := range []*hipDeviceByteBuffer{batch.Value, batch.InputNorm} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	if err := batch.QK.Close(); err != nil {
		lastErr = err
	}
	if err := batch.QKV.Close(); err != nil {
		lastErr = err
	}
	return lastErr
}

type hipGemma4Q4PrefillLayerBodyBatch struct {
	AttentionOutput     *hipDeviceByteBuffer
	AttentionProjection *hipDeviceByteBuffer
	AttentionResidual   *hipDeviceByteBuffer
	PreFeedForward      *hipDeviceByteBuffer
	MLPOutput           *hipDeviceByteBuffer
	PostFeedForward     *hipDeviceByteBuffer
	PerLayerProjection  *hipDeviceByteBuffer
	FinalHidden         *hipDeviceByteBuffer
}

type hipGemma4Q4PrefillForwardLayerBatch struct {
	KV   *hipGemma4Q4PrefillLayerKVBatch
	Body *hipGemma4Q4PrefillLayerBodyBatch
}

func (batch *hipGemma4Q4PrefillForwardLayerBatch) Close() error {
	if batch == nil {
		return nil
	}
	var lastErr error
	if err := batch.Body.Close(); err != nil {
		lastErr = err
	}
	if err := batch.KV.Close(); err != nil {
		lastErr = err
	}
	return lastErr
}

type hipGemma4Q4PrefillGreedyBatchOutput struct {
	Row    int
	Greedy hipGreedySampleResult
}

type hipGemma4Q4PrefillForwardBatch struct {
	Embedding   *hipDeviceByteBuffer
	Layers      []hipGemma4Q4PrefillForwardLayerBatch
	FinalHidden *hipDeviceByteBuffer
	Greedy      []hipGemma4Q4PrefillGreedyBatchOutput
}

func (batch *hipGemma4Q4PrefillForwardBatch) Close() error {
	if batch == nil {
		return nil
	}
	var lastErr error
	for index := len(batch.Layers) - 1; index >= 0; index-- {
		if err := batch.Layers[index].Close(); err != nil {
			lastErr = err
		}
	}
	if err := batch.Embedding.Close(); err != nil {
		lastErr = err
	}
	return lastErr
}

func (batch *hipGemma4Q4PrefillLayerBodyBatch) Close() error {
	if batch == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{
		batch.FinalHidden,
		batch.PerLayerProjection,
		batch.PostFeedForward,
		batch.MLPOutput,
		batch.PreFeedForward,
		batch.AttentionResidual,
		batch.AttentionProjection,
		batch.AttentionOutput,
	} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func hipGemma4Q4PrefillUBatchTokens() (int, error) {
	raw := strings.TrimSpace(os.Getenv(hipGemma4Q4PrefillUBatchEnv))
	if raw == "" {
		return hipGemma4Q4PrefillDefaultUBatchTokens, nil
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 {
		return 0, core.E(hipGemma4Q4Layer0Operation, core.Sprintf("%s must be a positive integer", hipGemma4Q4PrefillUBatchEnv), nil)
	}
	return value, nil
}

func (args hipPerLayerInputTransposeLaunchArgs) Binary() ([]byte, error) {
	if args.InputPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input transpose pointers are required", nil)
	}
	batch, err := rocmDeviceKVPositiveUint32("per-layer input transpose batch", args.Batch)
	if err != nil {
		return nil, err
	}
	layerCount, err := rocmDeviceKVPositiveUint32("per-layer input transpose layer count", args.LayerCount)
	if err != nil {
		return nil, err
	}
	inputSize, err := rocmDeviceKVPositiveUint32("per-layer input transpose input size", args.InputSize)
	if err != nil {
		return nil, err
	}
	wantBytes := uint64(batch) * uint64(layerCount) * uint64(inputSize) * 4
	if args.InputBytes != wantBytes || args.OutputBytes != wantBytes {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input transpose byte count mismatch", nil)
	}
	payload := make([]byte, hipPerLayerInputTransposeLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipPerLayerInputTransposeLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[24:], args.InputBytes)
	binary.LittleEndian.PutUint64(payload[32:], args.OutputBytes)
	binary.LittleEndian.PutUint32(payload[40:], batch)
	binary.LittleEndian.PutUint32(payload[44:], layerCount)
	binary.LittleEndian.PutUint32(payload[48:], inputSize)
	return payload, nil
}

func hipGemma4Q4PlanPromptPrefill(promptTokens []int32, startPos int, ubatchTokens int) (hipGemma4Q4PrefillPlan, error) {
	if len(promptTokens) == 0 {
		return hipGemma4Q4PrefillPlan{}, core.E(hipGemma4Q4Layer0Operation, "prompt prefill requires at least one token", nil)
	}
	if startPos < 0 {
		return hipGemma4Q4PrefillPlan{}, core.E(hipGemma4Q4Layer0Operation, "prompt prefill start position must be non-negative", nil)
	}
	if ubatchTokens <= 0 {
		return hipGemma4Q4PrefillPlan{}, core.E(hipGemma4Q4Layer0Operation, "prompt prefill ubatch size must be positive", nil)
	}
	plan := hipGemma4Q4PrefillPlan{
		PromptTokens: len(promptTokens),
		StartPos:     startPos,
		UBatchTokens: ubatchTokens,
		OutputTokens: 1,
		Batches:      make([]hipGemma4Q4PrefillUBatch, 0, (len(promptTokens)+ubatchTokens-1)/ubatchTokens),
	}
	for start := 0; start < len(promptTokens); start += ubatchTokens {
		end := start + ubatchTokens
		if end > len(promptTokens) {
			end = len(promptTokens)
		}
		tokens := promptTokens[start:end]
		var outputs []bool
		if end == len(promptTokens) {
			outputs = make([]bool, len(tokens))
			outputs[len(outputs)-1] = true
		}
		plan.Batches = append(plan.Batches, hipGemma4Q4PrefillUBatch{
			Start:        start,
			End:          end,
			Position:     startPos + start,
			Tokens:       tokens,
			OutputTokens: outputs,
		})
	}
	return plan, nil
}

func hipRunGemma4Q4PrefillEmbeddingBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, tokens []int32) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if cfg.HiddenSize <= 0 || cfg.Embedding.HiddenSize != cfg.HiddenSize {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill embedding hidden size mismatch", nil)
	}
	if err := cfg.Embedding.validate(tokens); err != nil {
		return nil, err
	}
	embedding, err := hipRunEmbeddingLookupKernelWithDeviceTableBuffer(ctx, driver, tokens, cfg.Embedding)
	if err != nil {
		return nil, err
	}
	defer embedding.Close()
	scaled, err := hipRunVectorScaleDeviceKernel(ctx, driver, embedding, float32(math.Sqrt(float64(cfg.HiddenSize))))
	if err != nil {
		return nil, err
	}
	return scaled, nil
}

func hipRunPerLayerInputTransposeKernel(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, batch, layerCount, inputSize int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input transpose input buffer is required", nil)
	}
	if batch <= 0 || layerCount <= 0 || inputSize <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input transpose shape must be positive", nil)
	}
	count := batch * layerCount * inputSize
	if input.Count() != count || input.SizeBytes() != uint64(count*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input transpose input shape mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, hipGemma4Q4Layer0Operation, "Gemma4 q4 per-layer input transpose output", input.SizeBytes(), input.Count())
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipPerLayerInputTransposeLaunchArgs{
		InputPointer:  input.Pointer(),
		OutputPointer: output.Pointer(),
		InputBytes:    input.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		Batch:         batch,
		LayerCount:    layerCount,
		InputSize:     inputSize,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNamePerLayerInputTranspose, launchBytes, count)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, hidden *hipDeviceByteBuffer, epsilon float32) (*hipGemma4Q4PerLayerInputDeviceSet, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if len(cfg.Layers) == 0 || !cfg.Layers[0].PerLayerInput.hasGlobalPrecompute() {
		return nil, nil
	}
	if hidden == nil || hidden.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input hidden batch is required", nil)
	}
	if len(tokens) == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input tokens are required", nil)
	}
	perLayer := cfg.Layers[0].PerLayerInput
	if !perLayer.hasLayerApply() {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input precompute requires per-layer gate/projection tensors", nil)
	}
	if perLayer.InputSize <= 0 || perLayer.ModelProjection.Rows%perLayer.InputSize != 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "per-layer input rows must align with input size", nil)
	}
	layerCount := perLayer.ModelProjection.Rows / perLayer.InputSize
	if layerCount < len(cfg.Layers) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "computed per-layer input count is smaller than forward layer count", nil)
	}
	if hidden.Count() != len(tokens)*perLayer.ModelProjection.Cols || hidden.SizeBytes() != uint64(hidden.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input hidden batch shape mismatch", nil)
	}
	perLayerEmbedding, err := hipRunEmbeddingLookupKernelWithDeviceTableBuffer(ctx, driver, tokens, perLayer.Embedding)
	if err != nil {
		return nil, err
	}
	defer perLayerEmbedding.Close()
	perLayerEmbeddingScaled, err := hipRunVectorScaleDeviceKernel(ctx, driver, perLayerEmbedding, float32(math.Sqrt(float64(perLayer.InputSize))))
	if err != nil {
		return nil, err
	}
	defer perLayerEmbeddingScaled.Close()
	projected, err := hipRunProjectionBatchKernelWithDeviceInputWeightEncoding(
		ctx,
		driver,
		hidden,
		perLayer.ModelProjection.WeightPointer,
		perLayer.ModelProjection.WeightBytes,
		perLayer.ModelProjection.Rows,
		perLayer.ModelProjection.Cols,
		hipProjectionWeightEncodingBF16,
		len(tokens),
	)
	if err != nil {
		return nil, err
	}
	defer projected.Close()
	projectedScaled, err := hipRunVectorScaleDeviceKernel(ctx, driver, projected, float32(math.Pow(float64(perLayer.ModelProjection.Cols), -0.5)))
	if err != nil {
		return nil, err
	}
	defer projectedScaled.Close()
	normCfg := perLayer.ProjectionNorm
	normCfg.Epsilon = epsilon
	normCfg.Count = perLayer.InputSize
	projectedNorm, err := hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, driver, projectedScaled, normCfg, layerCount*len(tokens))
	if err != nil {
		return nil, err
	}
	defer projectedNorm.Close()
	combined, err := hipRunVectorAddDeviceKernel(ctx, driver, projectedNorm, perLayerEmbeddingScaled)
	if err != nil {
		return nil, err
	}
	defer combined.Close()
	scaled, err := hipRunVectorScaleDeviceKernel(ctx, driver, combined, float32(math.Sqrt(0.5)))
	if err != nil {
		return nil, err
	}
	defer scaled.Close()
	transposed, err := hipRunPerLayerInputTransposeKernel(ctx, driver, scaled, len(tokens), layerCount, perLayer.InputSize)
	if err != nil {
		return nil, err
	}
	outputs := &hipGemma4Q4PerLayerInputDeviceSet{
		driver:           driver,
		layerCount:       layerCount,
		layerStrideBytes: uint64(len(tokens) * perLayer.InputSize * 4),
		layerValueCount:  len(tokens) * perLayer.InputSize,
		viewLabel:        "per-layer input batch slice",
		Backing:          []*hipDeviceByteBuffer{transposed},
	}
	success := false
	defer func() {
		if !success {
			_ = outputs.Close()
		}
	}()
	success = true
	return outputs, nil
}

func hipRunGemma4Q4PrefillInputNormBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill input-norm token count must be positive", nil)
	}
	if cfg.HiddenSize <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill input-norm hidden size must be positive", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill input-norm input buffer is required", nil)
	}
	if input.Count() != tokenCount*cfg.HiddenSize || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill input-norm input buffer shape mismatch", nil)
	}
	if err := hipValidateGemma4Q4NormConfig("Gemma4Q4PrefillInputNorm", cfg.InputNorm, cfg.HiddenSize); err != nil {
		return nil, err
	}
	return hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, driver, input, cfg.InputNorm, tokenCount)
}

func hipRunGemma4Q4PrefillQKVProjectionBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int) (*hipGemma4Q4PrefillQKVBatch, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill QKV token count must be positive", nil)
	}
	if cfg.HiddenSize <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill QKV hidden size must be positive", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill QKV input buffer is required", nil)
	}
	if input.Count() != tokenCount*cfg.HiddenSize || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill QKV input buffer shape mismatch", nil)
	}
	out := &hipGemma4Q4PrefillQKVBatch{}
	success := false
	defer func() {
		if !success {
			_ = out.Close()
		}
	}()
	var err error
	out.Query, err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx, driver, input, cfg.QueryProjection, tokenCount)
	if err != nil {
		return nil, err
	}
	out.Key, err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx, driver, input, cfg.KeyProjection, tokenCount)
	if err != nil {
		return nil, err
	}
	out.Value, err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx, driver, input, cfg.ValueProjection, tokenCount)
	if err != nil {
		return nil, err
	}
	success = true
	return out, nil
}

func hipRunGemma4Q4PrefillQKNormRoPEBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, qkv *hipGemma4Q4PrefillQKVBatch, tokenCount int, startPosition int, epsilon float32) (*hipGemma4Q4PrefillRoPEQKBatch, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE token count must be positive", nil)
	}
	if startPosition < 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE start position must be non-negative", nil)
	}
	if cfg.HeadDim <= 0 || cfg.HeadDim%2 != 0 || cfg.QueryHeads <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE layer geometry mismatch", nil)
	}
	if cfg.RoPEBase <= 0 || math.IsNaN(float64(cfg.RoPEBase)) || math.IsInf(float64(cfg.RoPEBase), 0) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE base must be positive and finite", nil)
	}
	if cfg.RoPERotaryDim <= 0 || cfg.RoPERotaryDim > cfg.HeadDim || cfg.RoPERotaryDim%2 != 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE rotary dimension mismatch", nil)
	}
	if qkv == nil || qkv.Query == nil || qkv.Query.Pointer() == 0 || qkv.Key == nil || qkv.Key.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE QKV buffers are required", nil)
	}
	queryRows := cfg.QueryHeads * cfg.HeadDim
	keyRows := cfg.HeadDim
	if qkv.Query.Count() != tokenCount*queryRows || qkv.Query.SizeBytes() != uint64(qkv.Query.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE query buffer shape mismatch", nil)
	}
	if qkv.Key.Count() != tokenCount*keyRows || qkv.Key.SizeBytes() != uint64(qkv.Key.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill Q/K RoPE key buffer shape mismatch", nil)
	}
	if err := hipValidateGemma4Q4NormConfig("Gemma4Q4PrefillQueryNorm", cfg.QueryNorm, cfg.HeadDim); err != nil {
		return nil, err
	}
	if err := hipValidateGemma4Q4NormConfig("Gemma4Q4PrefillKeyNorm", cfg.KeyNorm, cfg.HeadDim); err != nil {
		return nil, err
	}
	queryNormCfg := hipGemma4Q4RoPENormConfig(cfg.QueryNorm, epsilon, cfg.HeadDim)
	keyNormCfg := hipGemma4Q4RoPENormConfig(cfg.KeyNorm, epsilon, cfg.HeadDim)
	ropeFrequencyDim, ropeRotaryCount := hipGemma4Q4RoPEKernelDims(cfg)
	out := &hipGemma4Q4PrefillRoPEQKBatch{}
	success := false
	defer func() {
		if !success {
			_ = out.Close()
		}
	}()
	var err error
	out.Query, err = hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfig(ctx, driver, qkv.Query, queryNormCfg, cfg.QueryHeads, tokenCount, startPosition, cfg.RoPEBase, ropeFrequencyDim, ropeRotaryCount)
	if err != nil {
		return nil, err
	}
	out.Key, err = hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfig(ctx, driver, qkv.Key, keyNormCfg, 1, tokenCount, startPosition, cfg.RoPEBase, ropeFrequencyDim, ropeRotaryCount)
	if err != nil {
		return nil, err
	}
	success = true
	return out, nil
}

func hipRunGemma4Q4PrefillValueNormBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, qkv *hipGemma4Q4PrefillQKVBatch, tokenCount int, epsilon float32) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill value norm token count must be positive", nil)
	}
	if cfg.HeadDim <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill value norm head dim must be positive", nil)
	}
	if qkv == nil || qkv.Value == nil || qkv.Value.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill value norm value buffer is required", nil)
	}
	if qkv.Value.Count() != tokenCount*cfg.HeadDim || qkv.Value.SizeBytes() != uint64(qkv.Value.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill value norm value buffer shape mismatch", nil)
	}
	return hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, driver, qkv.Value, hipRMSNormDeviceWeightConfig{
		Count:          cfg.HeadDim,
		Epsilon:        epsilon,
		WeightEncoding: hipRMSNormWeightEncodingNone,
	}, tokenCount)
}

func hipRunGemma4Q4PrefillDeviceKVBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, qk *hipGemma4Q4PrefillRoPEQKBatch, value *hipDeviceByteBuffer, tokenCount int, mode string) (*hipGemma4Q4PrefillDeviceKVBatch, error) {
	return hipRunGemma4Q4PrefillDeviceKVBatchWithPrior(ctx, driver, cfg, nil, qk, value, tokenCount, mode)
}

func hipRunGemma4Q4PrefillDeviceKVBatchWithPrior(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, prior *rocmDeviceKVCache, qk *hipGemma4Q4PrefillRoPEQKBatch, value *hipDeviceByteBuffer, tokenCount int, mode string) (*hipGemma4Q4PrefillDeviceKVBatch, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill device KV token count must be positive", nil)
	}
	if cfg.HeadDim <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill device KV head dim must be positive", nil)
	}
	if qk == nil || qk.Key == nil || qk.Key.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill device KV key buffer is required", nil)
	}
	if value == nil || value.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill device KV value buffer is required", nil)
	}
	if qk.Key.Count() != tokenCount*cfg.HeadDim || qk.Key.SizeBytes() != uint64(qk.Key.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill device KV key buffer shape mismatch", nil)
	}
	if value.Count() != tokenCount*cfg.HeadDim || value.SizeBytes() != uint64(value.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill device KV value buffer shape mismatch", nil)
	}
	var cache *rocmDeviceKVCache
	var err error
	if prior != nil {
		if prior.closed {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill prior device KV cache is closed", nil)
		}
		if prior.TokenCount() <= 0 {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill prior device KV cache is empty", nil)
		}
		if mode != "" && prior.mode != "" && prior.mode != mode {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill prior device KV mode mismatch", nil)
		}
		window := 0
		if cfg.SlidingWindow > 0 {
			window = cfg.SlidingWindow + tokenCount
		}
		cache, err = prior.withAppendedDeviceRowsWindow(ctx, qk.Key, value, cfg.HeadDim, cfg.HeadDim, tokenCount, window)
	} else {
		cache, err = newROCmDeviceKVCacheFromDeviceRows(ctx, driver, firstNonEmptyString(mode, rocmKVCacheModeFP16), hipGemma4Q4DeviceKVBlockSize(), qk.Key, value, cfg.HeadDim, cfg.HeadDim, tokenCount, 0)
	}
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = cache.Close()
		}
	}()
	table, err := cache.KernelDescriptorTable()
	if err != nil {
		return nil, err
	}
	defer func() {
		if !success {
			_ = table.Close()
		}
	}()
	launch, err := cache.KernelLaunchDescriptor(table)
	if err != nil {
		return nil, err
	}
	success = true
	return &hipGemma4Q4PrefillDeviceKVBatch{
		Cache:           cache,
		DescriptorTable: table,
		Launch:          launch,
	}, nil
}

func hipRunGemma4Q4PrefillLayerKVBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int, startPosition int, epsilon float32, mode string) (*hipGemma4Q4PrefillLayerKVBatch, error) {
	return hipRunGemma4Q4PrefillLayerKVBatchWithPrior(ctx, driver, cfg, input, nil, tokenCount, startPosition, epsilon, mode)
}

func hipRunGemma4Q4PrefillLayerKVBatchWithPrior(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, prior *rocmDeviceKVCache, tokenCount int, startPosition int, epsilon float32, mode string) (*hipGemma4Q4PrefillLayerKVBatch, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer KV token count must be positive", nil)
	}
	if startPosition < 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer KV start position must be non-negative", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer KV input buffer is required", nil)
	}
	if prior != nil {
		priorTokens := prior.TokenCount()
		if priorTokens != startPosition && (cfg.SlidingWindow <= 0 || priorTokens <= 0 || priorTokens > startPosition) {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill prior device KV token count must match start position or a retained sliding window", nil)
		}
	}
	out := &hipGemma4Q4PrefillLayerKVBatch{}
	success := false
	defer func() {
		if !success {
			_ = out.Close()
		}
	}()
	var err error
	out.InputNorm, err = hipRunGemma4Q4PrefillInputNormBatch(ctx, driver, cfg, input, tokenCount)
	if err != nil {
		return nil, err
	}
	out.QKV, err = hipRunGemma4Q4PrefillQKVProjectionBatch(ctx, driver, cfg, out.InputNorm, tokenCount)
	if err != nil {
		return nil, err
	}
	out.QK, err = hipRunGemma4Q4PrefillQKNormRoPEBatch(ctx, driver, cfg, out.QKV, tokenCount, startPosition, epsilon)
	if err != nil {
		return nil, err
	}
	out.Value, err = hipRunGemma4Q4PrefillValueNormBatch(ctx, driver, cfg, out.QKV, tokenCount, epsilon)
	if err != nil {
		return nil, err
	}
	out.DeviceKV, err = hipRunGemma4Q4PrefillDeviceKVBatchWithPrior(ctx, driver, cfg, prior, out.QK, out.Value, tokenCount, mode)
	if err != nil {
		return nil, err
	}
	success = true
	return out, nil
}

func hipRunGemma4Q4PrefillLayerQueryBatchWithSharedKV(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, sharedSource *hipGemma4Q4PrefillLayerKVBatch, tokenCount int, startPosition int, epsilon float32) (*hipGemma4Q4PrefillLayerKVBatch, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill shared layer query token count must be positive", nil)
	}
	if startPosition < 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill shared layer query start position must be non-negative", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill shared layer query input buffer is required", nil)
	}
	if sharedSource == nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill shared layer source KV is required", nil)
	}
	shared := sharedSource.DeviceKV
	if shared == nil || shared.Cache == nil || shared.DescriptorTable == nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill shared layer device KV is required", nil)
	}
	out := &hipGemma4Q4PrefillLayerKVBatch{}
	success := false
	defer func() {
		if !success {
			_ = out.Close()
		}
	}()
	var err error
	out.InputNorm, err = hipRunGemma4Q4PrefillInputNormBatch(ctx, driver, cfg, input, tokenCount)
	if err != nil {
		return nil, err
	}
	query, err := hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx, driver, out.InputNorm, cfg.QueryProjection, tokenCount)
	if err != nil {
		return nil, err
	}
	out.QKV = &hipGemma4Q4PrefillQKVBatch{Query: query}
	queryNormCfg := hipGemma4Q4RoPENormConfig(cfg.QueryNorm, epsilon, cfg.HeadDim)
	ropeFrequencyDim, ropeRotaryCount := hipGemma4Q4RoPEKernelDims(cfg)
	ropeQuery, err := hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfig(ctx, driver, query, queryNormCfg, cfg.QueryHeads, tokenCount, startPosition, cfg.RoPEBase, ropeFrequencyDim, ropeRotaryCount)
	if err != nil {
		return nil, err
	}
	out.QK = &hipGemma4Q4PrefillRoPEQKBatch{Query: ropeQuery}
	if sharedSource.QK != nil && sharedSource.QK.Key != nil && sharedSource.Value != nil {
		out.SharedKey = sharedSource.QK.Key
		out.SharedVal = sharedSource.Value
	}
	cache, err := shared.Cache.borrowedAlias()
	if err != nil {
		return nil, err
	}
	table, err := shared.DescriptorTable.borrowedAlias()
	if err != nil {
		_ = cache.Close()
		return nil, err
	}
	launch, err := cache.KernelLaunchDescriptor(table)
	if err != nil {
		_ = table.Close()
		_ = cache.Close()
		return nil, err
	}
	out.DeviceKV = &hipGemma4Q4PrefillDeviceKVBatch{
		Cache:           cache,
		DescriptorTable: table,
		Launch:          launch,
	}
	success = true
	return out, nil
}

func hipRunGemma4Q4PrefillAttentionBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, layer *hipGemma4Q4PrefillLayerKVBatch, tokenCount int, queryStartToken int) (*hipDeviceByteBuffer, error) {
	return hipRunGemma4Q4PrefillAttentionBatchWorkspace(ctx, driver, cfg, layer, tokenCount, queryStartToken, nil)
}

func hipRunGemma4Q4PrefillAttentionBatchWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, layer *hipGemma4Q4PrefillLayerKVBatch, tokenCount int, queryStartToken int, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill attention token count must be positive", nil)
	}
	if queryStartToken < 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill attention query start token must be non-negative", nil)
	}
	if cfg.HeadDim <= 0 || cfg.QueryHeads <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill attention layer geometry mismatch", nil)
	}
	if layer == nil || layer.QK == nil || layer.QK.Query == nil || layer.QK.Query.Pointer() == 0 ||
		layer.DeviceKV == nil || layer.DeviceKV.Cache == nil || layer.DeviceKV.DescriptorTable == nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill attention Q/K/V device buffers are required", nil)
	}
	queryCount := tokenCount * cfg.QueryHeads * cfg.HeadDim
	if layer.QK.Query.Count() != queryCount || layer.QK.Query.SizeBytes() != uint64(queryCount*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill attention query buffer shape mismatch", nil)
	}
	if uint64(queryStartToken)+uint64(tokenCount) > uint64(layer.DeviceKV.Cache.TokenCount()) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill attention causal window exceeds device KV token count", nil)
	}
	output, err := hipAllocateByteBuffer(driver, hipGemma4Q4Layer0Operation, "Gemma4 q4 prefill attention batch output", uint64(queryCount*4), queryCount)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	attentionReq := hipAttentionHeadsBatchCausalDeviceRequest{
		Dim:             cfg.HeadDim,
		TokenCount:      layer.DeviceKV.Cache.TokenCount(),
		HeadCount:       cfg.QueryHeads,
		QueryCount:      tokenCount,
		QueryStartToken: queryStartToken,
		Scale:           hipGemma4Q4AttentionScale(cfg.HeadDim),
	}
	contiguousKey := layer.QK.Key
	contiguousValue := layer.Value
	if contiguousKey == nil || contiguousValue == nil {
		contiguousKey = layer.SharedKey
		contiguousValue = layer.SharedVal
	}
	if queryStartToken == 0 && layer.DeviceKV.Cache.TokenCount() == tokenCount && contiguousKey != nil && contiguousValue != nil {
		attentionReq.Key = contiguousKey
		attentionReq.Value = contiguousValue
	} else {
		attentionReq.DeviceKV = layer.DeviceKV.Cache
		attentionReq.DescriptorTable = layer.DeviceKV.DescriptorTable
	}
	err = hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernelWorkspace(ctx, driver, attentionReq, layer.QK.Query, output, workspace)
	if err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunGemma4Q4PrefillResidualAddNormBatch(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, residualCfg, normCfg hipRMSNormDeviceWeightConfig, tokenCount int, outputScale float32) (*hipDeviceByteBuffer, *hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, nil, err
	}
	if tokenCount <= 0 {
		return nil, nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add-norm token count must be positive", nil)
	}
	if residualCfg.Count <= 0 || residualCfg.Count != normCfg.Count {
		return nil, nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add-norm dimensions must be positive and equal", nil)
	}
	if input == nil || input.Pointer() == 0 || residual == nil || residual.Pointer() == 0 {
		return nil, nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add-norm input buffers are required", nil)
	}
	wantCount := tokenCount * residualCfg.Count
	if input.Count() != wantCount || residual.Count() != wantCount ||
		input.SizeBytes() != uint64(wantCount*4) || residual.SizeBytes() != uint64(wantCount*4) {
		return nil, nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add-norm buffer shape mismatch", nil)
	}
	if math.IsNaN(float64(outputScale)) || math.IsInf(float64(outputScale), 0) {
		return nil, nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add-norm output scale must be finite", nil)
	}
	normalizedInput, err := hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, driver, input, residualCfg, tokenCount)
	if err != nil {
		return nil, nil, err
	}
	defer normalizedInput.Close()
	residualOutput, err := hipRunVectorAddDeviceKernel(ctx, driver, normalizedInput, residual)
	if err != nil {
		return nil, nil, err
	}
	if outputScale != 1 {
		scaled, err := hipRunVectorScaleDeviceKernel(ctx, driver, residualOutput, outputScale)
		if err != nil {
			_ = residualOutput.Close()
			return nil, nil, err
		}
		_ = residualOutput.Close()
		residualOutput = scaled
	}
	success := false
	defer func() {
		if !success {
			_ = residualOutput.Close()
		}
	}()
	normOutput, err := hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, driver, residualOutput, normCfg, tokenCount)
	if err != nil {
		return nil, nil, err
	}
	success = true
	return residualOutput, normOutput, nil
}

func hipRunGemma4Q4PrefillResidualAddBatch(ctx context.Context, driver nativeHIPDriver, input, residual *hipDeviceByteBuffer, cfg hipRMSNormDeviceWeightConfig, tokenCount int, outputScale float32) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add token count must be positive", nil)
	}
	if cfg.Count <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add dimension must be positive", nil)
	}
	if input == nil || input.Pointer() == 0 || residual == nil || residual.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add input buffers are required", nil)
	}
	wantCount := tokenCount * cfg.Count
	if input.Count() != wantCount || residual.Count() != wantCount ||
		input.SizeBytes() != uint64(wantCount*4) || residual.SizeBytes() != uint64(wantCount*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add buffer shape mismatch", nil)
	}
	if math.IsNaN(float64(outputScale)) || math.IsInf(float64(outputScale), 0) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill residual-add output scale must be finite", nil)
	}
	normalizedInput, err := hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(ctx, driver, input, cfg, tokenCount)
	if err != nil {
		return nil, err
	}
	defer normalizedInput.Close()
	residualOutput, err := hipRunVectorAddDeviceKernel(ctx, driver, normalizedInput, residual)
	if err != nil {
		return nil, err
	}
	if outputScale == 1 {
		return residualOutput, nil
	}
	scaled, err := hipRunVectorScaleDeviceKernel(ctx, driver, residualOutput, outputScale)
	if err != nil {
		_ = residualOutput.Close()
		return nil, err
	}
	_ = residualOutput.Close()
	return scaled, nil
}

func hipRunGemma4Q4PrefillMLPBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, tokenCount int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill MLP token count must be positive", nil)
	}
	if cfg.HiddenSize <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill MLP hidden size must be positive", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill MLP input buffer is required", nil)
	}
	if input.Count() != tokenCount*cfg.HiddenSize || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill MLP input buffer shape mismatch", nil)
	}
	activated, err := hipRunMLXQ4GELUTanhMultiplyBatchKernelWithDeviceInput(ctx, driver, input, cfg.GateProjection, cfg.UpProjection, tokenCount)
	if err != nil {
		return nil, err
	}
	defer activated.Close()
	return hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx, driver, activated, cfg.DownProjection, tokenCount)
}

func hipRunGemma4Q4PrefillLayerBodyBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, layer *hipGemma4Q4PrefillLayerKVBatch, tokenCount int, queryStartToken int, epsilon float32) (*hipGemma4Q4PrefillLayerBodyBatch, error) {
	return hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInput(ctx, driver, cfg, input, layer, nil, tokenCount, queryStartToken, epsilon)
}

func hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInputWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, layer *hipGemma4Q4PrefillLayerKVBatch, perLayerInput *hipDeviceByteBuffer, tokenCount int, queryStartToken int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipGemma4Q4PrefillLayerBodyBatch, error) {
	return hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInputInternal(ctx, driver, cfg, input, layer, perLayerInput, tokenCount, queryStartToken, epsilon, workspace)
}

func hipValidateGemma4Q4PrefillForwardBatch(cfg hipGemma4Q4ForwardConfig, tokens []int32, startPosition int, priorLayerKV []*rocmDeviceKVCache, perLayerInputs []*hipDeviceByteBuffer, outputRows []bool) error {
	if len(tokens) == 0 {
		return core.E(hipGemma4Q4Layer0Operation, "prefill forward token span is required", nil)
	}
	if startPosition < 0 {
		return core.E(hipGemma4Q4Layer0Operation, "prefill forward start position must be non-negative", nil)
	}
	if err := cfg.validate(); err != nil {
		return err
	}
	if startPosition == 0 && len(priorLayerKV) != 0 {
		return core.E(hipGemma4Q4Layer0Operation, "prefill forward prior layer KV requires nonzero start position", nil)
	}
	if startPosition > 0 && len(priorLayerKV) != len(cfg.Layers) {
		return core.E(hipGemma4Q4Layer0Operation, "prefill forward prior layer KV count mismatch", nil)
	}
	if len(priorLayerKV) != 0 && len(priorLayerKV) != len(cfg.Layers) {
		return core.E(hipGemma4Q4Layer0Operation, "prefill forward prior layer KV count mismatch", nil)
	}
	for index, prior := range priorLayerKV {
		if prior == nil {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("prefill forward layer %d prior device KV is required", index), nil)
		}
		if prior.closed {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("prefill forward layer %d prior device KV is closed", index), nil)
		}
		priorTokens := prior.TokenCount()
		if priorTokens != startPosition && (cfg.Layers[index].SlidingWindow <= 0 || priorTokens <= 0 || priorTokens > startPosition) {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("prefill forward layer %d prior device KV token count must match start position or a retained sliding window", index), nil)
		}
	}
	if len(outputRows) != 0 && len(outputRows) != len(tokens) {
		return core.E(hipGemma4Q4Layer0Operation, "prefill forward output mask length mismatch", nil)
	}
	if len(perLayerInputs) != 0 && len(perLayerInputs) != len(cfg.Layers) {
		return core.E(hipGemma4Q4Layer0Operation, "prefill forward per-layer input count mismatch", nil)
	}
	generatePerLayerInputs := len(perLayerInputs) == 0 && len(cfg.Layers) > 0 && cfg.Layers[0].PerLayerInput.hasGlobalPrecompute()
	for index, layer := range cfg.Layers {
		if !layer.PerLayerInput.hasLayerApply() {
			continue
		}
		if generatePerLayerInputs {
			continue
		}
		if len(perLayerInputs) == 0 || perLayerInputs[index] == nil {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("prefill forward layer %d per-layer input batch is required", index), nil)
		}
		input := perLayerInputs[index]
		wantCount := len(tokens) * layer.PerLayerInput.InputSize
		if input.Pointer() == 0 || input.Count() != wantCount || input.SizeBytes() != uint64(wantCount*4) {
			return core.E(hipGemma4Q4Layer0Operation, core.Sprintf("prefill forward layer %d per-layer input batch shape mismatch", index), nil)
		}
	}
	return nil
}

func hipGemma4Q4CanUseBatchedGeneratePrefill(cfg hipGemma4Q4ForwardConfig) bool {
	if len(cfg.Layers) == 0 {
		return false
	}
	for _, layer := range cfg.Layers {
		if layer.PerLayerInput.hasLayerApply() && !cfg.Layers[0].PerLayerInput.hasGlobalPrecompute() {
			return false
		}
	}
	return true
}

func hipRunGemma4Q4PrefillForwardBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, startPosition int, epsilon float32, mode string, perLayerInputs []*hipDeviceByteBuffer, outputRows []bool, best *hipDeviceByteBuffer) (*hipGemma4Q4PrefillForwardBatch, error) {
	return hipRunGemma4Q4PrefillForwardBatchWithPrior(ctx, driver, cfg, tokens, startPosition, epsilon, mode, nil, perLayerInputs, outputRows, best)
}

func hipRunGemma4Q4PrefillForwardBatchWithPrior(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, startPosition int, epsilon float32, mode string, priorLayerKV []*rocmDeviceKVCache, perLayerInputs []*hipDeviceByteBuffer, outputRows []bool, best *hipDeviceByteBuffer) (*hipGemma4Q4PrefillForwardBatch, error) {
	return hipRunGemma4Q4PrefillForwardBatchWithPriorWorkspace(ctx, driver, cfg, tokens, startPosition, epsilon, mode, priorLayerKV, perLayerInputs, outputRows, best, nil)
}

func hipRunGemma4Q4PrefillForwardBatchWithPriorWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, startPosition int, epsilon float32, mode string, priorLayerKV []*rocmDeviceKVCache, perLayerInputs []*hipDeviceByteBuffer, outputRows []bool, best *hipDeviceByteBuffer, workspace *hipAttentionHeadsChunkedWorkspace) (*hipGemma4Q4PrefillForwardBatch, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if err := hipValidateGemma4Q4PrefillForwardBatch(cfg, tokens, startPosition, priorLayerKV, perLayerInputs, outputRows); err != nil {
		return nil, err
	}
	out := &hipGemma4Q4PrefillForwardBatch{
		Layers: make([]hipGemma4Q4PrefillForwardLayerBatch, 0, len(cfg.Layers)),
	}
	success := false
	defer func() {
		if !success {
			_ = out.Close()
		}
	}()
	hidden, err := hipRunGemma4Q4PrefillEmbeddingBatch(ctx, driver, cfg.Layers[0], tokens)
	if err != nil {
		return nil, err
	}
	out.Embedding = hidden
	var generatedPerLayerInputs *hipGemma4Q4PerLayerInputDeviceSet
	if len(perLayerInputs) == 0 && cfg.Layers[0].PerLayerInput.hasGlobalPrecompute() {
		generatedPerLayerInputs, err = hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatch(ctx, driver, cfg, tokens, hidden, epsilon)
		if err != nil {
			return nil, err
		}
		defer generatedPerLayerInputs.Close()
	}
	tokenCount := len(tokens)
	sharedSources := hipGemma4Q4SharedKVSourceByLayer(cfg)
	for index, layerCfg := range cfg.Layers {
		layerInput := hidden
		prior := (*rocmDeviceKVCache)(nil)
		if len(priorLayerKV) > index {
			prior = priorLayerKV[index]
		}
		var layerKV *hipGemma4Q4PrefillLayerKVBatch
		if len(sharedSources) > index && sharedSources[index] != index {
			source := sharedSources[index]
			if source < 0 || source >= len(out.Layers) || out.Layers[source].KV == nil || out.Layers[source].KV.DeviceKV == nil {
				return nil, core.E(hipGemma4Q4Layer0Operation, "prefill shared KV source layer is unavailable", nil)
			}
			layerKV, err = hipRunGemma4Q4PrefillLayerQueryBatchWithSharedKV(ctx, driver, layerCfg, layerInput, out.Layers[source].KV, tokenCount, startPosition, epsilon)
			if err != nil {
				return nil, err
			}
		} else {
			layerKV, err = hipRunGemma4Q4PrefillLayerKVBatchWithPrior(ctx, driver, layerCfg, layerInput, prior, tokenCount, startPosition, epsilon, mode)
			if err != nil {
				return nil, err
			}
		}
		layerOut := hipGemma4Q4PrefillForwardLayerBatch{KV: layerKV}
		out.Layers = append(out.Layers, layerOut)
		perLayerInput := (*hipDeviceByteBuffer)(nil)
		if generatedPerLayerInputs != nil {
			perLayerInput = generatedPerLayerInputs.Layer(index)
		} else if len(perLayerInputs) > index {
			perLayerInput = perLayerInputs[index]
		}
		queryStartToken := layerKV.DeviceKV.Cache.TokenCount() - tokenCount
		if queryStartToken < 0 {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer device KV token count is smaller than query batch", nil)
		}
		body, err := hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInputWorkspace(ctx, driver, layerCfg, layerInput, layerKV, perLayerInput, tokenCount, queryStartToken, epsilon, workspace)
		if err != nil {
			return nil, err
		}
		out.Layers[index].Body = body
		hidden = body.FinalHidden
		out.FinalHidden = hidden
	}
	if len(outputRows) > 0 {
		last := cfg.Layers[len(cfg.Layers)-1]
		for row, selected := range outputRows {
			if !selected {
				continue
			}
			greedy, err := hipRunGemma4Q4PrefillFinalGreedyForRow(ctx, driver, last, out.FinalHidden, tokenCount, row, epsilon, best)
			if err != nil {
				return nil, err
			}
			out.Greedy = append(out.Greedy, hipGemma4Q4PrefillGreedyBatchOutput{
				Row:    row,
				Greedy: greedy,
			})
		}
	}
	success = true
	return out, nil
}

func hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward *hipGemma4Q4PrefillForwardBatch, mode string) (*hipGemma4Q4DeviceDecodeState, error) {
	if forward == nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill forward output is required", nil)
	}
	state := hipNewGemma4Q4DeviceDecodeState(firstNonEmptyString(mode, rocmKVCacheModeFP16), len(forward.Layers))
	state.appendLayers = len(forward.Layers)
	success := false
	defer func() {
		if !success {
			_ = state.Close()
		}
	}()
	for index := range forward.Layers {
		deviceKV := (*hipGemma4Q4PrefillDeviceKVBatch)(nil)
		if forward.Layers[index].KV != nil {
			deviceKV = forward.Layers[index].KV.DeviceKV
		}
		if deviceKV == nil || deviceKV.Cache == nil || deviceKV.DescriptorTable == nil {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill forward layer device KV is required", nil)
		}
		state.layers = append(state.layers, hipGemma4Q4DeviceLayerKVState{
			cache:           deviceKV.Cache,
			descriptorTable: deviceKV.DescriptorTable,
			launch:          deviceKV.Launch,
		})
		deviceKV.Cache = nil
		deviceKV.DescriptorTable = nil
		deviceKV.Launch = rocmDeviceKVLaunchDescriptor{}
	}
	success = true
	return state, nil
}

func hipRunGemma4Q4PrefillPerLayerInputProjectionBatch(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input, perLayerInput *hipDeviceByteBuffer, tokenCount int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if err := cfg.validatePerLayerInput(); err != nil {
		return nil, err
	}
	if !cfg.PerLayerInput.hasLayerApply() {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input tensors are not configured", nil)
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input token count must be positive", nil)
	}
	if cfg.HiddenSize <= 0 || cfg.PerLayerInput.InputSize <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input geometry mismatch", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input hidden buffer is required", nil)
	}
	if input.Count() != tokenCount*cfg.HiddenSize || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input hidden buffer shape mismatch", nil)
	}
	if perLayerInput == nil || perLayerInput.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input multiplier buffer is required", nil)
	}
	if perLayerInput.Count() != tokenCount*cfg.PerLayerInput.InputSize ||
		perLayerInput.SizeBytes() != uint64(perLayerInput.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input multiplier buffer shape mismatch", nil)
	}
	activated, err := hipRunMLXQ4GELUTanhProjectionBatchKernelWithDeviceMultiplier(ctx, driver, input, perLayerInput, cfg.PerLayerInput.InputGate, tokenCount)
	if err != nil {
		return nil, err
	}
	defer activated.Close()
	return hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx, driver, activated, cfg.PerLayerInput.Projection, tokenCount)
}

func hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInput(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, layer *hipGemma4Q4PrefillLayerKVBatch, perLayerInput *hipDeviceByteBuffer, tokenCount int, queryStartToken int, epsilon float32) (*hipGemma4Q4PrefillLayerBodyBatch, error) {
	return hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInputInternal(ctx, driver, cfg, input, layer, perLayerInput, tokenCount, queryStartToken, epsilon, nil)
}

func hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInputInternal(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, input *hipDeviceByteBuffer, layer *hipGemma4Q4PrefillLayerKVBatch, perLayerInput *hipDeviceByteBuffer, tokenCount int, queryStartToken int, epsilon float32, workspace *hipAttentionHeadsChunkedWorkspace) (*hipGemma4Q4PrefillLayerBodyBatch, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if tokenCount <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer body token count must be positive", nil)
	}
	if queryStartToken < 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer body query start token must be non-negative", nil)
	}
	if cfg.HiddenSize <= 0 || cfg.QueryHeads <= 0 || cfg.HeadDim <= 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer body geometry mismatch", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer body residual input buffer is required", nil)
	}
	if input.Count() != tokenCount*cfg.HiddenSize || input.SizeBytes() != uint64(input.Count()*4) {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer body residual input buffer shape mismatch", nil)
	}
	if perLayerInput != nil {
		if err := cfg.validatePerLayerInput(); err != nil {
			return nil, err
		}
		if !cfg.PerLayerInput.hasLayerApply() {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input tensors are not configured", nil)
		}
		if perLayerInput.Pointer() == 0 ||
			perLayerInput.Count() != tokenCount*cfg.PerLayerInput.InputSize ||
			perLayerInput.SizeBytes() != uint64(perLayerInput.Count()*4) {
			return nil, core.E(hipGemma4Q4Layer0Operation, "prefill per-layer input multiplier buffer shape mismatch", nil)
		}
	}
	if layer == nil {
		return nil, core.E(hipGemma4Q4Layer0Operation, "prefill layer body KV setup is required", nil)
	}
	out := &hipGemma4Q4PrefillLayerBodyBatch{}
	success := false
	defer func() {
		if !success {
			_ = out.Close()
		}
	}()
	var err error
	out.AttentionOutput, err = hipRunGemma4Q4PrefillAttentionBatchWorkspace(ctx, driver, cfg, layer, tokenCount, queryStartToken, workspace)
	if err != nil {
		return nil, err
	}
	out.AttentionProjection, err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx, driver, out.AttentionOutput, cfg.OutputProjection, tokenCount)
	if err != nil {
		return nil, err
	}
	postAttentionNormCfg := cfg.PostAttentionNorm
	postAttentionNormCfg.Epsilon = epsilon
	postAttentionNormCfg.Count = cfg.HiddenSize
	preFeedForwardNormCfg := cfg.PreFeedForwardNorm
	preFeedForwardNormCfg.Epsilon = epsilon
	preFeedForwardNormCfg.Count = cfg.HiddenSize
	out.AttentionResidual, out.PreFeedForward, err = hipRunGemma4Q4PrefillResidualAddNormBatch(ctx, driver, out.AttentionProjection, input, postAttentionNormCfg, preFeedForwardNormCfg, tokenCount, 1)
	if err != nil {
		return nil, err
	}
	out.MLPOutput, err = hipRunGemma4Q4PrefillMLPBatch(ctx, driver, cfg, out.PreFeedForward, tokenCount)
	if err != nil {
		return nil, err
	}
	postFeedForwardNormCfg := cfg.PostFeedForwardNorm
	postFeedForwardNormCfg.Epsilon = epsilon
	postFeedForwardNormCfg.Count = cfg.HiddenSize
	postFeedForwardScale := float32(1)
	if perLayerInput == nil {
		postFeedForwardScale = cfg.effectiveLayerScalar()
	}
	out.PostFeedForward, err = hipRunGemma4Q4PrefillResidualAddBatch(ctx, driver, out.MLPOutput, out.AttentionResidual, postFeedForwardNormCfg, tokenCount, postFeedForwardScale)
	if err != nil {
		return nil, err
	}
	if perLayerInput == nil {
		out.FinalHidden = out.PostFeedForward
	} else {
		out.PerLayerProjection, err = hipRunGemma4Q4PrefillPerLayerInputProjectionBatch(ctx, driver, cfg, out.PostFeedForward, perLayerInput, tokenCount)
		if err != nil {
			return nil, err
		}
		perLayerNormCfg := cfg.PerLayerInput.PostInputNorm
		perLayerNormCfg.Epsilon = epsilon
		perLayerNormCfg.Count = cfg.HiddenSize
		out.FinalHidden, err = hipRunGemma4Q4PrefillResidualAddBatch(ctx, driver, out.PerLayerProjection, out.PostFeedForward, perLayerNormCfg, tokenCount, cfg.effectiveLayerScalar())
		if err != nil {
			return nil, err
		}
	}
	success = true
	return out, nil
}

func hipRunGemma4Q4PrefillFinalGreedyForRow(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, hidden *hipDeviceByteBuffer, tokenCount int, row int, epsilon float32, best *hipDeviceByteBuffer) (hipGreedySampleResult, error) {
	return hipRunGemma4Q4PrefillFinalGreedyForRowSuppress(ctx, driver, cfg, hidden, tokenCount, row, epsilon, best, nil)
}

func hipRunGemma4Q4PrefillFinalGreedyForRowSuppress(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, hidden *hipDeviceByteBuffer, tokenCount int, row int, epsilon float32, best *hipDeviceByteBuffer, suppressTokens []int32) (hipGreedySampleResult, error) {
	return hipRunGemma4Q4PrefillFinalGreedyForRowSuppressWorkspace(ctx, driver, cfg, hidden, tokenCount, row, epsilon, best, suppressTokens, nil)
}

func hipRunGemma4Q4PrefillFinalGreedyForRowSuppressWorkspace(ctx context.Context, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, hidden *hipDeviceByteBuffer, tokenCount int, row int, epsilon float32, best *hipDeviceByteBuffer, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (hipGreedySampleResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipGreedySampleResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "HIP driver is not available", nil)
	}
	if tokenCount <= 0 {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy token count must be positive", nil)
	}
	if row < 0 || row >= tokenCount {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy row is outside token batch", nil)
	}
	if cfg.HiddenSize <= 0 {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy hidden size must be positive", nil)
	}
	if hidden == nil || hidden.Pointer() == 0 {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy hidden batch is required", nil)
	}
	if hidden.Count() != tokenCount*cfg.HiddenSize || hidden.SizeBytes() != uint64(hidden.Count()*4) {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy hidden batch shape mismatch", nil)
	}
	finalNormCfg := cfg.FinalNorm
	finalNormCfg.Epsilon = epsilon
	finalNormCfg.Count = cfg.HiddenSize
	if err := hipValidateGemma4Q4NormConfig("prefill_final_norm", finalNormCfg, cfg.HiddenSize); err != nil {
		return hipGreedySampleResult{}, err
	}
	if cfg.LMHeadProjection.Rows != cfg.VocabSize {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy LM head shape mismatch", nil)
	}
	if err := cfg.LMHeadProjection.validateInputCount(cfg.HiddenSize); err != nil {
		return hipGreedySampleResult{}, core.E(hipGemma4Q4Layer0Operation, "prefill final greedy LM head config", err)
	}
	rowOffset := nativeDevicePointer(row * cfg.HiddenSize * 4)
	hiddenRow := hipBorrowDeviceByteBuffer(driver, "Gemma4 q4 prefill selected final hidden row", hidden.Pointer()+rowOffset, uint64(cfg.HiddenSize*4), cfg.HiddenSize)
	finalNorm, err := hipRunRMSNormKernelWithDeviceInputWeightConfig(ctx, driver, hiddenRow, finalNormCfg)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	defer finalNorm.Close()
	return hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppress(ctx, driver, finalNorm, cfg.LMHeadProjection, cfg.FinalLogitSoftcap, best, suppressTokens, workspace)
}
