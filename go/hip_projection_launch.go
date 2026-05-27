// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"sync"

	core "dappco.re/go"
)

const (
	hipProjectionLaunchArgsVersion             uint32 = 1
	hipProjectionLaunchArgsBytes                      = 96
	hipProjectionBatchLaunchArgsVersion        uint32 = 1
	hipProjectionBatchLaunchArgsBytes                 = 104
	hipMLXQ4ProjectionLaunchArgsVersion        uint32 = 1
	hipMLXQ4ProjectionLaunchArgsBytes                 = 96
	hipMLXQ4ProjectionBatchLaunchArgsVersion   uint32 = 1
	hipMLXQ4ProjectionBatchLaunchArgsBytes            = 96
	hipMLXQ4TripleProjLaunchArgsVersion        uint32 = 1
	hipMLXQ4TripleProjLaunchArgsBytes                 = 168
	hipMLXQ4GELUTanhMulLaunchArgsVersion       uint32 = 1
	hipMLXQ4GELUTanhMulLaunchArgsBytes                = 128
	hipMLXQ4GELUTanhMulBatchLaunchArgsVersion  uint32 = 1
	hipMLXQ4GELUTanhMulBatchLaunchArgsBytes           = 128
	hipMLXQ4GELUTanhProjLaunchArgsVersion      uint32 = 1
	hipMLXQ4GELUTanhProjLaunchArgsBytes               = 96
	hipMLXQ4GELUTanhProjBatchLaunchArgsVersion uint32 = 1
	hipMLXQ4GELUTanhProjBatchLaunchArgsBytes          = 104
	hipPackedTopKLaunchArgsVersion             uint32 = 1
	hipPackedTopKLaunchArgsBytes                      = 48
	hipMLXQ4ProjectionBits                            = 4
	hipMLXQ4ProjectionBlockSize                uint32 = 256
	hipMLXQ4ProjectionRowsPerBlock                    = 8
	hipMLXQ4ProjectionBatchTokensPerBlock             = 8
	hipMLXQ4ProjectionGreedyRowsPerBlock              = 32
	hipMLXQ4ProjectionBestBytes                       = 8
	hipPackedTopKMaxK                                 = 128
	hipPackedTopKBlockSize                     uint32 = 256
	hipPackedTopKChunkSize                            = 512
)

const (
	hipProjectionWeightEncodingFP16 uint32 = 1
	hipProjectionWeightEncodingQ8   uint32 = 2
	hipProjectionWeightEncodingF32  uint32 = 3
	hipProjectionWeightEncodingBF16 uint32 = 4
)

const hipProjectionLaunchFlagBias uint32 = 1

type hipDeviceByteBuffer struct {
	driver    nativeHIPDriver
	pointer   nativeDevicePointer
	count     int
	sizeBytes uint64
	closed    bool
	borrowed  bool
	pooled    bool
	label     string
}

type hipDeviceByteBufferPoolEntry struct {
	driver  nativeHIPDriver
	pointer nativeDevicePointer
}

var hipDeviceByteBufferPool = struct {
	sync.Mutex
	entries map[uint64][]hipDeviceByteBufferPoolEntry
	bytes   uint64
}{
	entries: make(map[uint64][]hipDeviceByteBufferPoolEntry),
}

const (
	hipDeviceByteBufferPoolMaxBytes   = 768 << 20
	hipDeviceByteBufferPoolMaxPerSize = 512
)

func hipProjectionUint32Bytes(operation, label string, value uint64) error {
	if value > uint64(^uint32(0)) {
		return core.E(operation, label+" are out of uint32 range", nil)
	}
	return nil
}

type hipProjectionDeviceBuffers struct {
	Input    *hipDeviceByteBuffer
	Weights  *hipDeviceByteBuffer
	Bias     *hipDeviceByteBuffer
	Output   *hipDeviceByteBuffer
	Encoding uint32
	Q8Scale  float32
	Rows     int
	Cols     int
}

type hipProjectionLaunchArgs struct {
	InputPointer   nativeDevicePointer
	InputCount     int
	InputBytes     uint64
	WeightPointer  nativeDevicePointer
	WeightBytes    uint64
	BiasPointer    nativeDevicePointer
	BiasBytes      uint64
	OutputPointer  nativeDevicePointer
	OutputBytes    uint64
	Rows           int
	Cols           int
	WeightEncoding uint32
	Flags          uint32
	Q8Scale        float32
}

type hipProjectionBatchLaunchArgs struct {
	InputPointer   nativeDevicePointer
	WeightPointer  nativeDevicePointer
	WeightBytes    uint64
	BiasPointer    nativeDevicePointer
	BiasBytes      uint64
	OutputPointer  nativeDevicePointer
	InputBytes     uint64
	OutputBytes    uint64
	Rows           int
	Cols           int
	Batch          int
	WeightEncoding uint32
	Flags          uint32
	Q8Scale        float32
}

type hipMLXQ4ProjectionRequest struct {
	Input     []float32
	Weight    []uint32
	Scales    []uint16
	Biases    []uint16
	Rows      int
	Cols      int
	GroupSize int
}

type hipMLXQ4ProjectionDeviceBuffers struct {
	Input     *hipDeviceByteBuffer
	Weight    *hipDeviceByteBuffer
	Scales    *hipDeviceByteBuffer
	Biases    *hipDeviceByteBuffer
	Output    *hipDeviceByteBuffer
	Rows      int
	Cols      int
	GroupSize int
}

type hipMLXQ4DeviceWeightConfig struct {
	WeightPointer nativeDevicePointer
	ScalePointer  nativeDevicePointer
	BiasPointer   nativeDevicePointer
	WeightBytes   uint64
	ScaleBytes    uint64
	BiasBytes     uint64
	Rows          int
	Cols          int
	GroupSize     int
}

type hipMLXQ4ProjectionLaunchArgs struct {
	InputPointer    nativeDevicePointer
	WeightPointer   nativeDevicePointer
	ScalePointer    nativeDevicePointer
	BiasPointer     nativeDevicePointer
	OutputPointer   nativeDevicePointer
	SuppressPointer nativeDevicePointer
	Rows            int
	Cols            int
	GroupSize       int
	Bits            int
	SuppressCount   int
	InputBytes      uint64
	WeightBytes     uint64
	ScaleBytes      uint64
	BiasBytes       uint64
	OutputBytes     uint64
}

type hipPackedTopKLaunchArgs struct {
	InputPointer  nativeDevicePointer
	OutputPointer nativeDevicePointer
	InputCount    int
	OutputCount   int
	TopK          int
	ChunkSize     int
	InputBytes    uint64
	OutputBytes   uint64
}

type hipMLXQ4ProjectionBatchLaunchArgs struct {
	InputPointer  nativeDevicePointer
	WeightPointer nativeDevicePointer
	ScalePointer  nativeDevicePointer
	BiasPointer   nativeDevicePointer
	OutputPointer nativeDevicePointer
	Rows          int
	Cols          int
	Batch         int
	GroupSize     int
	Bits          int
	InputBytes    uint64
	WeightBytes   uint64
	ScaleBytes    uint64
	BiasBytes     uint64
	OutputBytes   uint64
}

type hipMLXQ4GELUTanhMulLaunchArgs struct {
	InputPointer      nativeDevicePointer
	GateWeightPointer nativeDevicePointer
	GateScalePointer  nativeDevicePointer
	GateBiasPointer   nativeDevicePointer
	UpWeightPointer   nativeDevicePointer
	UpScalePointer    nativeDevicePointer
	UpBiasPointer     nativeDevicePointer
	OutputPointer     nativeDevicePointer
	Rows              int
	Cols              int
	GroupSize         int
	Bits              int
	InputBytes        uint64
	GateWeightBytes   uint64
	GateScaleBytes    uint64
	GateBiasBytes     uint64
	UpWeightBytes     uint64
	UpScaleBytes      uint64
	UpBiasBytes       uint64
	OutputBytes       uint64
}

type hipMLXQ4GELUTanhMulBatchLaunchArgs struct {
	InputPointer      nativeDevicePointer
	GateWeightPointer nativeDevicePointer
	GateScalePointer  nativeDevicePointer
	GateBiasPointer   nativeDevicePointer
	UpWeightPointer   nativeDevicePointer
	UpScalePointer    nativeDevicePointer
	UpBiasPointer     nativeDevicePointer
	OutputPointer     nativeDevicePointer
	Rows              int
	Cols              int
	GroupSize         int
	Bits              int
	InputBytes        uint64
	GateWeightBytes   uint64
	GateScaleBytes    uint64
	GateBiasBytes     uint64
	UpWeightBytes     uint64
	UpScaleBytes      uint64
	UpBiasBytes       uint64
	OutputBytes       uint64
	Batch             int
}

type hipMLXQ4TripleProjLaunchArgs struct {
	InputPointer        nativeDevicePointer
	OutputPointer       nativeDevicePointer
	FirstWeightPointer  nativeDevicePointer
	FirstScalePointer   nativeDevicePointer
	FirstBiasPointer    nativeDevicePointer
	SecondWeightPointer nativeDevicePointer
	SecondScalePointer  nativeDevicePointer
	SecondBiasPointer   nativeDevicePointer
	ThirdWeightPointer  nativeDevicePointer
	ThirdScalePointer   nativeDevicePointer
	ThirdBiasPointer    nativeDevicePointer
	FirstRows           int
	SecondRows          int
	ThirdRows           int
	Cols                int
	GroupSize           int
	Bits                int
	InputBytes          uint64
	OutputBytes         uint64
	FirstWeightBytes    uint64
	FirstScaleBytes     uint64
	FirstBiasBytes      uint64
	SecondWeightBytes   uint64
	SecondScaleBytes    uint64
	SecondBiasBytes     uint64
	ThirdWeightBytes    uint64
	ThirdScaleBytes     uint64
	ThirdBiasBytes      uint64
}

type hipMLXQ4GELUTanhProjLaunchArgs struct {
	InputPointer      nativeDevicePointer
	WeightPointer     nativeDevicePointer
	ScalePointer      nativeDevicePointer
	BiasPointer       nativeDevicePointer
	MultiplierPointer nativeDevicePointer
	OutputPointer     nativeDevicePointer
	Rows              int
	Cols              int
	GroupSize         int
	Bits              int
	InputBytes        uint64
	WeightBytes       uint64
	ScaleBytes        uint64
	BiasBytes         uint64
	MultiplierBytes   uint64
	OutputBytes       uint64
}

type hipMLXQ4GELUTanhProjBatchLaunchArgs struct {
	InputPointer      nativeDevicePointer
	WeightPointer     nativeDevicePointer
	ScalePointer      nativeDevicePointer
	BiasPointer       nativeDevicePointer
	MultiplierPointer nativeDevicePointer
	OutputPointer     nativeDevicePointer
	Rows              int
	Cols              int
	Batch             int
	GroupSize         int
	Bits              int
	InputBytes        uint64
	WeightBytes       uint64
	ScaleBytes        uint64
	BiasBytes         uint64
	MultiplierBytes   uint64
	OutputBytes       uint64
}

func (req hipProjectionRequest) projectionDeviceBuffers(driver nativeHIPDriver) (*hipProjectionDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	inputPayload, err := hipFloat32Payload(req.Input)
	if err != nil {
		return nil, core.E("rocm.hip.ProjectionLaunch", "encode input", err)
	}
	input, err := hipUploadByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection input", inputPayload, len(req.Input))
	if err != nil {
		return nil, err
	}
	buffers := &hipProjectionDeviceBuffers{Input: input, Rows: req.Rows, Cols: req.Cols}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	switch {
	case len(req.F32) > 0:
		weightsPayload, err := hipFloat32Payload(req.F32)
		if err != nil {
			return nil, core.E("rocm.hip.ProjectionLaunch", "encode f32 weights", err)
		}
		weights, err := hipUploadByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection f32 weights", weightsPayload, len(req.F32))
		if err != nil {
			return nil, err
		}
		buffers.Weights = weights
		buffers.Encoding = hipProjectionWeightEncodingF32
	case len(req.FP16) > 0:
		weightsPayload, err := hipUint16Payload(req.FP16)
		if err != nil {
			return nil, core.E("rocm.hip.ProjectionLaunch", "encode fp16 weights", err)
		}
		weights, err := hipUploadByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection fp16 weights", weightsPayload, len(req.FP16))
		if err != nil {
			return nil, err
		}
		buffers.Weights = weights
		buffers.Encoding = hipProjectionWeightEncodingFP16
	case len(req.BF16) > 0:
		weightsPayload, err := hipUint16Payload(req.BF16)
		if err != nil {
			return nil, core.E("rocm.hip.ProjectionLaunch", "encode bf16 weights", err)
		}
		weights, err := hipUploadByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection bf16 weights", weightsPayload, len(req.BF16))
		if err != nil {
			return nil, err
		}
		buffers.Weights = weights
		buffers.Encoding = hipProjectionWeightEncodingBF16
	case len(req.Q8) > 0:
		weightsPayload := hipInt8Payload(req.Q8)
		weights, err := hipUploadByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection q8 weights", weightsPayload, len(req.Q8))
		if err != nil {
			return nil, err
		}
		buffers.Weights = weights
		buffers.Encoding = hipProjectionWeightEncodingQ8
		buffers.Q8Scale = req.Q8Scale
	default:
		return nil, core.E("rocm.hip.ProjectionLaunch", "projection weights are required", nil)
	}

	if len(req.Bias) > 0 {
		biasPayload, err := hipFloat32Payload(req.Bias)
		if err != nil {
			return nil, core.E("rocm.hip.ProjectionLaunch", "encode bias", err)
		}
		bias, err := hipUploadByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection bias", biasPayload, len(req.Bias))
		if err != nil {
			return nil, err
		}
		buffers.Bias = bias
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.ProjectionLaunch", "projection output", uint64(req.Rows*4), req.Rows)
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipMLXQ4ProjectionRequest) validate() error {
	return validateHIPMLXQ4ProjectionShape(len(req.Input), len(req.Weight), len(req.Scales), len(req.Biases), req.Rows, req.Cols, req.GroupSize)
}

func (cfg hipMLXQ4DeviceWeightConfig) validate(input []float32) error {
	return cfg.validateInputCount(len(input))
}

func (cfg hipMLXQ4DeviceWeightConfig) validateInputCount(inputCount int) error {
	if cfg.WeightPointer == 0 || cfg.ScalePointer == 0 || cfg.BiasPointer == 0 {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device weight, scale, and bias pointers are required", nil)
	}
	if cfg.WeightBytes == 0 || cfg.ScaleBytes == 0 || cfg.BiasBytes == 0 {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device weight, scale, and bias byte counts are required", nil)
	}
	if cfg.WeightBytes%4 != 0 || cfg.ScaleBytes%2 != 0 || cfg.BiasBytes%2 != 0 {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device byte counts must be element-aligned", nil)
	}
	if cfg.WeightBytes/4 > uint64(int(^uint(0)>>1)) ||
		cfg.ScaleBytes/2 > uint64(int(^uint(0)>>1)) ||
		cfg.BiasBytes/2 > uint64(int(^uint(0)>>1)) {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device element counts are out of int range", nil)
	}
	return validateHIPMLXQ4ProjectionShape(inputCount, int(cfg.WeightBytes/4), int(cfg.ScaleBytes/2), int(cfg.BiasBytes/2), cfg.Rows, cfg.Cols, cfg.GroupSize)
}

func (cfg hipMLXQ4DeviceWeightConfig) validateBatchInputCount(inputCount int, batch int) error {
	if batch <= 0 {
		return core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch size must be positive", nil)
	}
	if inputCount != cfg.Cols*batch {
		return core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch input count mismatch", nil)
	}
	if cfg.WeightPointer == 0 || cfg.ScalePointer == 0 || cfg.BiasPointer == 0 {
		return core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection device weight, scale, and bias pointers are required", nil)
	}
	if cfg.WeightBytes == 0 || cfg.ScaleBytes == 0 || cfg.BiasBytes == 0 {
		return core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection device weight, scale, and bias byte counts are required", nil)
	}
	if cfg.WeightBytes%4 != 0 || cfg.ScaleBytes%2 != 0 || cfg.BiasBytes%2 != 0 {
		return core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection device byte counts must be element-aligned", nil)
	}
	if cfg.WeightBytes/4 > uint64(int(^uint(0)>>1)) ||
		cfg.ScaleBytes/2 > uint64(int(^uint(0)>>1)) ||
		cfg.BiasBytes/2 > uint64(int(^uint(0)>>1)) {
		return core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection device element counts are out of int range", nil)
	}
	return validateHIPMLXQ4ProjectionShape(cfg.Cols, int(cfg.WeightBytes/4), int(cfg.ScaleBytes/2), int(cfg.BiasBytes/2), cfg.Rows, cfg.Cols, cfg.GroupSize)
}

func (req hipMLXQ4ProjectionRequest) deviceBuffers(driver nativeHIPDriver) (*hipMLXQ4ProjectionDeviceBuffers, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	inputPayload, err := hipFloat32Payload(req.Input)
	if err != nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "encode input", err)
	}
	input, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection input", inputPayload, len(req.Input))
	if err != nil {
		return nil, err
	}
	buffers := &hipMLXQ4ProjectionDeviceBuffers{Input: input, Rows: req.Rows, Cols: req.Cols, GroupSize: req.GroupSize}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()

	weightPayload, err := hipUint32Payload(req.Weight)
	if err != nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "encode packed weights", err)
	}
	weights, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection packed weights", weightPayload, len(req.Weight))
	if err != nil {
		return nil, err
	}
	buffers.Weight = weights

	scalePayload, err := hipUint16Payload(req.Scales)
	if err != nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "encode scales", err)
	}
	scales, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection scales", scalePayload, len(req.Scales))
	if err != nil {
		return nil, err
	}
	buffers.Scales = scales

	biasPayload, err := hipUint16Payload(req.Biases)
	if err != nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "encode biases", err)
	}
	biases, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection biases", biasPayload, len(req.Biases))
	if err != nil {
		return nil, err
	}
	buffers.Biases = biases

	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection output", uint64(req.Rows*4), req.Rows)
	if err != nil {
		return nil, err
	}
	buffers.Output = output
	success = true
	return buffers, nil
}

func (req hipMLXQ4ProjectionRequest) launchArgs(buffers *hipMLXQ4ProjectionDeviceBuffers) (hipMLXQ4ProjectionLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipMLXQ4ProjectionLaunchArgs{}, err
	}
	if buffers == nil || buffers.Input == nil || buffers.Weight == nil || buffers.Scales == nil || buffers.Biases == nil || buffers.Output == nil {
		return hipMLXQ4ProjectionLaunchArgs{}, core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device buffers are required", nil)
	}
	packedPerRow := req.Cols / 8
	groupsPerRow := req.Cols / req.GroupSize
	if buffers.Input.Count() != req.Cols ||
		buffers.Weight.Count() != req.Rows*packedPerRow ||
		buffers.Scales.Count() != req.Rows*groupsPerRow ||
		buffers.Biases.Count() != req.Rows*groupsPerRow ||
		buffers.Output.Count() != req.Rows ||
		buffers.Rows != req.Rows ||
		buffers.Cols != req.Cols ||
		buffers.GroupSize != req.GroupSize {
		return hipMLXQ4ProjectionLaunchArgs{}, core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device buffer shape mismatch", nil)
	}
	return hipMLXQ4ProjectionLaunchArgs{
		InputPointer:  buffers.Input.Pointer(),
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		OutputPointer: buffers.Output.Pointer(),
		Rows:          req.Rows,
		Cols:          req.Cols,
		GroupSize:     req.GroupSize,
		Bits:          hipMLXQ4ProjectionBits,
		InputBytes:    buffers.Input.SizeBytes(),
		WeightBytes:   buffers.Weight.SizeBytes(),
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		OutputBytes:   buffers.Output.SizeBytes(),
	}, nil
}

func (req hipProjectionRequest) projectionLaunchArgs(buffers *hipProjectionDeviceBuffers) (hipProjectionLaunchArgs, error) {
	if err := req.validate(); err != nil {
		return hipProjectionLaunchArgs{}, err
	}
	if buffers == nil || buffers.Input == nil || buffers.Weights == nil || buffers.Output == nil {
		return hipProjectionLaunchArgs{}, core.E("rocm.hip.ProjectionLaunch", "projection device buffers are required", nil)
	}
	if buffers.Input.Count() != req.Cols || buffers.Weights.Count() != req.Rows*req.Cols || buffers.Output.Count() != req.Rows {
		return hipProjectionLaunchArgs{}, core.E("rocm.hip.ProjectionLaunch", "projection device buffer shape mismatch", nil)
	}
	var biasPointer nativeDevicePointer
	var biasBytes uint64
	var flags uint32
	if len(req.Bias) > 0 {
		if buffers.Bias == nil || buffers.Bias.Count() != req.Rows {
			return hipProjectionLaunchArgs{}, core.E("rocm.hip.ProjectionLaunch", "projection bias buffer shape mismatch", nil)
		}
		biasPointer = buffers.Bias.Pointer()
		biasBytes = buffers.Bias.SizeBytes()
		flags |= hipProjectionLaunchFlagBias
	}
	encoding, err := hipProjectionWeightEncodingCode(req)
	if err != nil {
		return hipProjectionLaunchArgs{}, err
	}
	if buffers.Encoding != encoding {
		return hipProjectionLaunchArgs{}, core.E("rocm.hip.ProjectionLaunch", "projection weight encoding mismatch", nil)
	}
	return hipProjectionLaunchArgs{
		InputPointer:   buffers.Input.Pointer(),
		InputCount:     buffers.Input.Count(),
		InputBytes:     buffers.Input.SizeBytes(),
		WeightPointer:  buffers.Weights.Pointer(),
		WeightBytes:    buffers.Weights.SizeBytes(),
		BiasPointer:    biasPointer,
		BiasBytes:      biasBytes,
		OutputPointer:  buffers.Output.Pointer(),
		OutputBytes:    buffers.Output.SizeBytes(),
		Rows:           req.Rows,
		Cols:           req.Cols,
		WeightEncoding: encoding,
		Flags:          flags,
		Q8Scale:        req.Q8Scale,
	}, nil
}

func (args hipProjectionLaunchArgs) Binary() ([]byte, error) {
	if args.InputPointer == 0 || args.WeightPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "input, weight, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	inputCount, err := rocmDeviceKVPositiveUint32("input count", args.InputCount)
	if err != nil {
		return nil, err
	}
	if inputCount != cols {
		return nil, core.E("rocm.hip.ProjectionLaunch", "input count must match cols", nil)
	}
	if args.InputBytes != uint64(cols)*4 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "input byte count mismatch", nil)
	}
	if args.InputBytes > uint64(^uint32(0)) {
		return nil, core.E("rocm.hip.ProjectionLaunch", "input bytes are out of uint32 range", nil)
	}
	if args.OutputBytes != uint64(rows)*4 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "output byte count mismatch", nil)
	}
	switch args.WeightEncoding {
	case hipProjectionWeightEncodingFP16, hipProjectionWeightEncodingBF16:
		if args.WeightBytes != uint64(rows)*uint64(cols)*2 {
			return nil, core.E("rocm.hip.ProjectionLaunch", "fp16/bf16 weight byte count mismatch", nil)
		}
	case hipProjectionWeightEncodingQ8:
		if args.WeightBytes != uint64(rows)*uint64(cols) {
			return nil, core.E("rocm.hip.ProjectionLaunch", "q8 weight byte count mismatch", nil)
		}
		if !hipQ8ScaleIsPositiveFinite(args.Q8Scale) {
			return nil, core.E("rocm.hip.ProjectionLaunch", "q8 scale must be positive and finite", nil)
		}
	case hipProjectionWeightEncodingF32:
		if args.WeightBytes != uint64(rows)*uint64(cols)*4 {
			return nil, core.E("rocm.hip.ProjectionLaunch", "f32 weight byte count mismatch", nil)
		}
	default:
		return nil, core.E("rocm.hip.ProjectionLaunch", core.Sprintf("unsupported projection weight encoding %d", args.WeightEncoding), nil)
	}
	if args.Flags&hipProjectionLaunchFlagBias != 0 {
		if args.BiasPointer == 0 {
			return nil, core.E("rocm.hip.ProjectionLaunch", "bias pointer is nil", nil)
		}
		if args.BiasBytes != uint64(rows)*4 {
			return nil, core.E("rocm.hip.ProjectionLaunch", "bias byte count mismatch", nil)
		}
	} else if args.BiasPointer != 0 || args.BiasBytes != 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "bias metadata supplied without bias flag", nil)
	}
	payload := hipBorrowLaunchPacket(hipProjectionLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipProjectionLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint32(payload[16:], inputCount)
	binary.LittleEndian.PutUint32(payload[20:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[32:], args.WeightBytes)
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[48:], args.BiasBytes)
	binary.LittleEndian.PutUint64(payload[56:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[64:], args.OutputBytes)
	binary.LittleEndian.PutUint32(payload[72:], rows)
	binary.LittleEndian.PutUint32(payload[76:], cols)
	binary.LittleEndian.PutUint32(payload[80:], args.WeightEncoding)
	binary.LittleEndian.PutUint32(payload[84:], args.Flags)
	binary.LittleEndian.PutUint32(payload[88:], math.Float32bits(args.Q8Scale))
	return payload, nil
}

func (args hipProjectionBatchLaunchArgs) Binary() ([]byte, error) {
	if args.InputPointer == 0 || args.WeightPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "input, weight, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	batch, err := rocmDeviceKVPositiveUint32("batch", args.Batch)
	if err != nil {
		return nil, err
	}
	if args.InputBytes != uint64(cols)*uint64(batch)*4 {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "input byte count mismatch", nil)
	}
	if args.OutputBytes != uint64(rows)*uint64(batch)*4 {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "output byte count mismatch", nil)
	}
	switch args.WeightEncoding {
	case hipProjectionWeightEncodingFP16, hipProjectionWeightEncodingBF16:
		if args.WeightBytes != uint64(rows)*uint64(cols)*2 {
			return nil, core.E("rocm.hip.ProjectionBatchLaunch", "fp16/bf16 weight byte count mismatch", nil)
		}
	case hipProjectionWeightEncodingQ8:
		if args.WeightBytes != uint64(rows)*uint64(cols) {
			return nil, core.E("rocm.hip.ProjectionBatchLaunch", "q8 weight byte count mismatch", nil)
		}
		if !hipQ8ScaleIsPositiveFinite(args.Q8Scale) {
			return nil, core.E("rocm.hip.ProjectionBatchLaunch", "q8 scale must be positive and finite", nil)
		}
	case hipProjectionWeightEncodingF32:
		if args.WeightBytes != uint64(rows)*uint64(cols)*4 {
			return nil, core.E("rocm.hip.ProjectionBatchLaunch", "f32 weight byte count mismatch", nil)
		}
	default:
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", core.Sprintf("unsupported projection weight encoding %d", args.WeightEncoding), nil)
	}
	if args.Flags&hipProjectionLaunchFlagBias != 0 {
		if args.BiasPointer == 0 {
			return nil, core.E("rocm.hip.ProjectionBatchLaunch", "bias pointer is nil", nil)
		}
		if args.BiasBytes != uint64(rows)*4 {
			return nil, core.E("rocm.hip.ProjectionBatchLaunch", "bias byte count mismatch", nil)
		}
	} else if args.BiasPointer != 0 || args.BiasBytes != 0 {
		return nil, core.E("rocm.hip.ProjectionBatchLaunch", "bias metadata supplied without bias flag", nil)
	}
	payload := hipBorrowLaunchPacket(hipProjectionBatchLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipProjectionBatchLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], args.WeightBytes)
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], args.BiasBytes)
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[56:], args.OutputBytes)
	binary.LittleEndian.PutUint32(payload[64:], rows)
	binary.LittleEndian.PutUint32(payload[68:], cols)
	binary.LittleEndian.PutUint32(payload[72:], batch)
	binary.LittleEndian.PutUint32(payload[76:], args.WeightEncoding)
	binary.LittleEndian.PutUint32(payload[80:], args.Flags)
	binary.LittleEndian.PutUint32(payload[84:], math.Float32bits(args.Q8Scale))
	binary.LittleEndian.PutUint64(payload[88:], args.InputBytes)
	return payload, nil
}

func (args hipMLXQ4ProjectionLaunchArgs) Binary() ([]byte, error) {
	return args.binary(hipMLXQ4ProjectionOutputFull)
}

func (args hipMLXQ4ProjectionLaunchArgs) GreedyBinary() ([]byte, error) {
	return args.binary(hipMLXQ4ProjectionOutputBest)
}

func (args hipMLXQ4ProjectionLaunchArgs) ScoresBinary() ([]byte, error) {
	return args.binary(hipMLXQ4ProjectionOutputScores)
}

func (args hipPackedTopKLaunchArgs) Binary() ([]byte, error) {
	if args.InputPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "input and output pointers are required", nil)
	}
	inputCount, err := rocmDeviceKVPositiveUint32("packed top-k input count", args.InputCount)
	if err != nil {
		return nil, err
	}
	outputCount, err := rocmDeviceKVPositiveUint32("packed top-k output count", args.OutputCount)
	if err != nil {
		return nil, err
	}
	topK, err := rocmDeviceKVPositiveUint32("packed top-k", args.TopK)
	if err != nil {
		return nil, err
	}
	if topK > hipPackedTopKMaxK {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "top-k exceeds kernel maximum", nil)
	}
	chunkSize, err := rocmDeviceKVPositiveUint32("packed top-k chunk size", args.ChunkSize)
	if err != nil {
		return nil, err
	}
	if args.ChunkSize != hipPackedTopKChunkSize {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "chunk size mismatch", nil)
	}
	chunkCount := (args.InputCount + args.ChunkSize - 1) / args.ChunkSize
	if args.OutputCount != chunkCount*args.TopK {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "output count mismatch", nil)
	}
	if args.InputBytes != uint64(args.InputCount*hipMLXQ4ProjectionBestBytes) {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "input byte count mismatch", nil)
	}
	if args.OutputBytes != uint64(args.OutputCount*hipMLXQ4ProjectionBestBytes) {
		return nil, core.E("rocm.hip.PackedTopKLaunch", "output byte count mismatch", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.PackedTopKLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.PackedTopKLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	payload := hipBorrowLaunchPacket(hipPackedTopKLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipPackedTopKLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[24:], inputCount)
	binary.LittleEndian.PutUint32(payload[28:], outputCount)
	binary.LittleEndian.PutUint32(payload[32:], topK)
	binary.LittleEndian.PutUint32(payload[36:], chunkSize)
	binary.LittleEndian.PutUint32(payload[40:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[44:], uint32(args.OutputBytes))
	return payload, nil
}

const (
	hipMLXQ4ProjectionOutputFull = iota
	hipMLXQ4ProjectionOutputBest
	hipMLXQ4ProjectionOutputScores
)

func (args hipMLXQ4ProjectionLaunchArgs) binary(outputKind int) ([]byte, error) {
	if args.InputPointer == 0 || args.WeightPointer == 0 || args.ScalePointer == 0 || args.BiasPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "input, weight, scale, bias, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	if bits != hipMLXQ4ProjectionBits {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "only 4-bit MLX affine projection is supported", nil)
	}
	if cols%8 != 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "cols must be divisible by 8 for q4 packing", nil)
	}
	if cols%groupSize != 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "cols must be divisible by group size", nil)
	}
	packedPerRow := uint64(cols / 8)
	groupsPerRow := uint64(cols / groupSize)
	if args.InputBytes != uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "input byte count mismatch", nil)
	}
	if args.WeightBytes != uint64(rows)*packedPerRow*4 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "packed weight byte count mismatch", nil)
	}
	if args.ScaleBytes != uint64(rows)*groupsPerRow*2 || args.BiasBytes != uint64(rows)*groupsPerRow*2 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "scale/bias byte count mismatch", nil)
	}
	wantOutputBytes := uint64(rows) * 4
	switch outputKind {
	case hipMLXQ4ProjectionOutputFull:
	case hipMLXQ4ProjectionOutputBest:
		wantOutputBytes = hipMLXQ4ProjectionBestBytes
	case hipMLXQ4ProjectionOutputScores:
		wantOutputBytes = uint64(rows) * hipMLXQ4ProjectionBestBytes
	default:
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "unsupported q4 projection output kind", nil)
	}
	if args.OutputBytes != wantOutputBytes {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "output byte count mismatch", nil)
	}
	suppressCount := uint32(0)
	if args.SuppressCount > 0 {
		if outputKind != hipMLXQ4ProjectionOutputBest && outputKind != hipMLXQ4ProjectionOutputScores {
			return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "suppress tokens require greedy or score output", nil)
		}
		if args.SuppressPointer == 0 {
			return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "suppress token pointer is required", nil)
		}
		value, err := rocmDeviceKVPositiveUint32("suppress token count", args.SuppressCount)
		if err != nil {
			return nil, err
		}
		suppressCount = value
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionLaunch", "weight bytes", args.WeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionLaunch", "scale bytes", args.ScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionLaunch", "bias bytes", args.BiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	payload := hipBorrowLaunchPacket(hipMLXQ4ProjectionLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4ProjectionLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ScalePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[48:], rows)
	binary.LittleEndian.PutUint32(payload[52:], cols)
	binary.LittleEndian.PutUint32(payload[56:], groupSize)
	binary.LittleEndian.PutUint32(payload[60:], bits)
	binary.LittleEndian.PutUint32(payload[64:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[68:], uint32(args.WeightBytes))
	binary.LittleEndian.PutUint32(payload[72:], uint32(args.ScaleBytes))
	binary.LittleEndian.PutUint32(payload[76:], uint32(args.BiasBytes))
	binary.LittleEndian.PutUint32(payload[80:], uint32(args.OutputBytes))
	binary.LittleEndian.PutUint32(payload[84:], suppressCount)
	binary.LittleEndian.PutUint64(payload[88:], uint64(args.SuppressPointer))
	return payload, nil
}

func (args hipMLXQ4ProjectionBatchLaunchArgs) Binary() ([]byte, error) {
	if args.InputPointer == 0 || args.WeightPointer == 0 || args.ScalePointer == 0 || args.BiasPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "input, weight, scale, bias, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	batch, err := rocmDeviceKVPositiveUint32("batch", args.Batch)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	if bits != hipMLXQ4ProjectionBits {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "only 4-bit MLX affine projection is supported", nil)
	}
	if cols%8 != 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "cols must be divisible by 8 for q4 packing", nil)
	}
	if cols%groupSize != 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "cols must be divisible by group size", nil)
	}
	packedPerRow := uint64(cols / 8)
	groupsPerRow := uint64(cols / groupSize)
	if args.InputBytes != uint64(batch)*uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "input byte count mismatch", nil)
	}
	if args.WeightBytes != uint64(rows)*packedPerRow*4 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "packed weight byte count mismatch", nil)
	}
	if args.ScaleBytes != uint64(rows)*groupsPerRow*2 || args.BiasBytes != uint64(rows)*groupsPerRow*2 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "scale/bias byte count mismatch", nil)
	}
	if args.OutputBytes != uint64(batch)*uint64(rows)*4 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "output byte count mismatch", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionBatchLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionBatchLaunch", "weight bytes", args.WeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionBatchLaunch", "scale bytes", args.ScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionBatchLaunch", "bias bytes", args.BiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4ProjectionBatchLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	payload := hipBorrowLaunchPacket(hipMLXQ4ProjectionBatchLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4ProjectionBatchLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ScalePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[48:], rows)
	binary.LittleEndian.PutUint32(payload[52:], cols)
	binary.LittleEndian.PutUint32(payload[56:], batch)
	binary.LittleEndian.PutUint32(payload[60:], groupSize)
	binary.LittleEndian.PutUint32(payload[64:], bits)
	binary.LittleEndian.PutUint32(payload[68:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[72:], uint32(args.WeightBytes))
	binary.LittleEndian.PutUint32(payload[76:], uint32(args.ScaleBytes))
	binary.LittleEndian.PutUint32(payload[80:], uint32(args.BiasBytes))
	binary.LittleEndian.PutUint32(payload[84:], uint32(args.OutputBytes))
	return payload, nil
}

func (args hipMLXQ4TripleProjLaunchArgs) Binary() ([]byte, error) {
	if args.InputPointer == 0 || args.OutputPointer == 0 ||
		args.FirstWeightPointer == 0 || args.FirstScalePointer == 0 || args.FirstBiasPointer == 0 ||
		args.SecondWeightPointer == 0 || args.SecondScalePointer == 0 || args.SecondBiasPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "input, output, and q4 weight pointers are required", nil)
	}
	firstRows, err := rocmDeviceKVPositiveUint32("first rows", args.FirstRows)
	if err != nil {
		return nil, err
	}
	secondRows, err := rocmDeviceKVPositiveUint32("second rows", args.SecondRows)
	if err != nil {
		return nil, err
	}
	thirdRows, err := rocmDeviceKVUint32("third rows", args.ThirdRows)
	if err != nil {
		return nil, err
	}
	if thirdRows > 0 && (args.ThirdWeightPointer == 0 || args.ThirdScalePointer == 0 || args.ThirdBiasPointer == 0) {
		return nil, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "third q4 weight pointers are required when third rows are non-zero", nil)
	}
	if thirdRows == 0 && (args.ThirdWeightBytes != 0 || args.ThirdScaleBytes != 0 || args.ThirdBiasBytes != 0) {
		return nil, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "third q4 byte counts must be zero when third rows are zero", nil)
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	if bits != hipMLXQ4ProjectionBits {
		return nil, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "only 4-bit MLX affine projection is supported", nil)
	}
	if cols%8 != 0 {
		return nil, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "cols must be divisible by 8 for q4 packing", nil)
	}
	if cols%groupSize != 0 {
		return nil, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "cols must be divisible by group size", nil)
	}
	packedPerRow := uint64(cols / 8)
	groupsPerRow := uint64(cols / groupSize)
	totalRows := uint64(firstRows) + uint64(secondRows) + uint64(thirdRows)
	if args.InputBytes != uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "input byte count mismatch", nil)
	}
	if args.OutputBytes != totalRows*4 {
		return nil, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "output byte count mismatch", nil)
	}
	checkPart := func(label string, rows uint32, weightBytes, scaleBytes, biasBytes uint64) error {
		if weightBytes != uint64(rows)*packedPerRow*4 {
			return core.E("rocm.hip.MLXQ4TripleProjectionLaunch", label+" packed weight byte count mismatch", nil)
		}
		wantScaleBiasBytes := uint64(rows) * groupsPerRow * 2
		if scaleBytes != wantScaleBiasBytes || biasBytes != wantScaleBiasBytes {
			return core.E("rocm.hip.MLXQ4TripleProjectionLaunch", label+" scale/bias byte count mismatch", nil)
		}
		return nil
	}
	if err := checkPart("first", firstRows, args.FirstWeightBytes, args.FirstScaleBytes, args.FirstBiasBytes); err != nil {
		return nil, err
	}
	if err := checkPart("second", secondRows, args.SecondWeightBytes, args.SecondScaleBytes, args.SecondBiasBytes); err != nil {
		return nil, err
	}
	if thirdRows > 0 {
		if err := checkPart("third", thirdRows, args.ThirdWeightBytes, args.ThirdScaleBytes, args.ThirdBiasBytes); err != nil {
			return nil, err
		}
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "first weight bytes", args.FirstWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "first scale bytes", args.FirstScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "first bias bytes", args.FirstBiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "second weight bytes", args.SecondWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "second scale bytes", args.SecondScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "second bias bytes", args.SecondBiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "third weight bytes", args.ThirdWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "third scale bytes", args.ThirdScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4TripleProjectionLaunch", "third bias bytes", args.ThirdBiasBytes); err != nil {
		return nil, err
	}
	payload := hipBorrowLaunchPacket(hipMLXQ4TripleProjLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4TripleProjLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.FirstWeightPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.FirstScalePointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.FirstBiasPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.SecondWeightPointer))
	binary.LittleEndian.PutUint64(payload[56:], uint64(args.SecondScalePointer))
	binary.LittleEndian.PutUint64(payload[64:], uint64(args.SecondBiasPointer))
	binary.LittleEndian.PutUint64(payload[72:], uint64(args.ThirdWeightPointer))
	binary.LittleEndian.PutUint64(payload[80:], uint64(args.ThirdScalePointer))
	binary.LittleEndian.PutUint64(payload[88:], uint64(args.ThirdBiasPointer))
	binary.LittleEndian.PutUint32(payload[96:], firstRows)
	binary.LittleEndian.PutUint32(payload[100:], secondRows)
	binary.LittleEndian.PutUint32(payload[104:], thirdRows)
	binary.LittleEndian.PutUint32(payload[108:], cols)
	binary.LittleEndian.PutUint32(payload[112:], groupSize)
	binary.LittleEndian.PutUint32(payload[116:], bits)
	binary.LittleEndian.PutUint32(payload[120:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[124:], uint32(args.OutputBytes))
	binary.LittleEndian.PutUint32(payload[128:], uint32(args.FirstWeightBytes))
	binary.LittleEndian.PutUint32(payload[132:], uint32(args.FirstScaleBytes))
	binary.LittleEndian.PutUint32(payload[136:], uint32(args.FirstBiasBytes))
	binary.LittleEndian.PutUint32(payload[140:], uint32(args.SecondWeightBytes))
	binary.LittleEndian.PutUint32(payload[144:], uint32(args.SecondScaleBytes))
	binary.LittleEndian.PutUint32(payload[148:], uint32(args.SecondBiasBytes))
	binary.LittleEndian.PutUint32(payload[152:], uint32(args.ThirdWeightBytes))
	binary.LittleEndian.PutUint32(payload[156:], uint32(args.ThirdScaleBytes))
	binary.LittleEndian.PutUint32(payload[160:], uint32(args.ThirdBiasBytes))
	return payload, nil
}

func (args hipMLXQ4GELUTanhMulLaunchArgs) Binary() ([]byte, error) {
	if args.InputPointer == 0 || args.GateWeightPointer == 0 || args.GateScalePointer == 0 ||
		args.GateBiasPointer == 0 || args.UpWeightPointer == 0 || args.UpScalePointer == 0 ||
		args.UpBiasPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "input, gate/up weights, scale/bias, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	if bits != hipMLXQ4ProjectionBits {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "only 4-bit MLX affine projection is supported", nil)
	}
	if cols%8 != 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "cols must be divisible by 8 for q4 packing", nil)
	}
	if cols%groupSize != 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "cols must be divisible by group size", nil)
	}
	packedPerRow := uint64(cols / 8)
	groupsPerRow := uint64(cols / groupSize)
	if args.InputBytes != uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "input byte count mismatch", nil)
	}
	wantWeightBytes := uint64(rows) * packedPerRow * 4
	if args.GateWeightBytes != wantWeightBytes || args.UpWeightBytes != wantWeightBytes {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "packed weight byte count mismatch", nil)
	}
	wantScaleBiasBytes := uint64(rows) * groupsPerRow * 2
	if args.GateScaleBytes != wantScaleBiasBytes || args.GateBiasBytes != wantScaleBiasBytes ||
		args.UpScaleBytes != wantScaleBiasBytes || args.UpBiasBytes != wantScaleBiasBytes {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "scale/bias byte count mismatch", nil)
	}
	if args.OutputBytes != uint64(rows)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "output byte count mismatch", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "gate weight bytes", args.GateWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "gate scale bytes", args.GateScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "gate bias bytes", args.GateBiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "up weight bytes", args.UpWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "up scale bytes", args.UpScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "up bias bytes", args.UpBiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	payload := hipBorrowLaunchPacket(hipMLXQ4GELUTanhMulLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4GELUTanhMulLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.GateWeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.GateScalePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.GateBiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.UpWeightPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.UpScalePointer))
	binary.LittleEndian.PutUint64(payload[56:], uint64(args.UpBiasPointer))
	binary.LittleEndian.PutUint64(payload[64:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[72:], rows)
	binary.LittleEndian.PutUint32(payload[76:], cols)
	binary.LittleEndian.PutUint32(payload[80:], groupSize)
	binary.LittleEndian.PutUint32(payload[84:], bits)
	binary.LittleEndian.PutUint32(payload[88:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[92:], uint32(args.GateWeightBytes))
	binary.LittleEndian.PutUint32(payload[96:], uint32(args.GateScaleBytes))
	binary.LittleEndian.PutUint32(payload[100:], uint32(args.GateBiasBytes))
	binary.LittleEndian.PutUint32(payload[104:], uint32(args.UpWeightBytes))
	binary.LittleEndian.PutUint32(payload[108:], uint32(args.UpScaleBytes))
	binary.LittleEndian.PutUint32(payload[112:], uint32(args.UpBiasBytes))
	binary.LittleEndian.PutUint32(payload[116:], uint32(args.OutputBytes))
	return payload, nil
}

func (args hipMLXQ4GELUTanhMulBatchLaunchArgs) Binary() ([]byte, error) {
	if args.InputPointer == 0 || args.GateWeightPointer == 0 || args.GateScalePointer == 0 ||
		args.GateBiasPointer == 0 || args.UpWeightPointer == 0 || args.UpScalePointer == 0 ||
		args.UpBiasPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "input, gate/up weights, scale/bias, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	batch, err := rocmDeviceKVPositiveUint32("batch", args.Batch)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	if bits != hipMLXQ4ProjectionBits {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "only 4-bit MLX affine projection is supported", nil)
	}
	if cols%8 != 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "cols must be divisible by 8 for q4 packing", nil)
	}
	if cols%groupSize != 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "cols must be divisible by group size", nil)
	}
	packedPerRow := uint64(cols / 8)
	groupsPerRow := uint64(cols / groupSize)
	if args.InputBytes != uint64(batch)*uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "input byte count mismatch", nil)
	}
	wantWeightBytes := uint64(rows) * packedPerRow * 4
	if args.GateWeightBytes != wantWeightBytes || args.UpWeightBytes != wantWeightBytes {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "packed weight byte count mismatch", nil)
	}
	wantScaleBiasBytes := uint64(rows) * groupsPerRow * 2
	if args.GateScaleBytes != wantScaleBiasBytes || args.GateBiasBytes != wantScaleBiasBytes ||
		args.UpScaleBytes != wantScaleBiasBytes || args.UpBiasBytes != wantScaleBiasBytes {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "scale/bias byte count mismatch", nil)
	}
	if args.OutputBytes != uint64(batch)*uint64(rows)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "output byte count mismatch", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "gate weight bytes", args.GateWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "gate scale bytes", args.GateScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "gate bias bytes", args.GateBiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "up weight bytes", args.UpWeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "up scale bytes", args.UpScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "up bias bytes", args.UpBiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	payload := hipBorrowLaunchPacket(hipMLXQ4GELUTanhMulBatchLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4GELUTanhMulBatchLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.GateWeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.GateScalePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.GateBiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.UpWeightPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.UpScalePointer))
	binary.LittleEndian.PutUint64(payload[56:], uint64(args.UpBiasPointer))
	binary.LittleEndian.PutUint64(payload[64:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[72:], rows)
	binary.LittleEndian.PutUint32(payload[76:], cols)
	binary.LittleEndian.PutUint32(payload[80:], groupSize)
	binary.LittleEndian.PutUint32(payload[84:], bits)
	binary.LittleEndian.PutUint32(payload[88:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[92:], uint32(args.GateWeightBytes))
	binary.LittleEndian.PutUint32(payload[96:], uint32(args.GateScaleBytes))
	binary.LittleEndian.PutUint32(payload[100:], uint32(args.GateBiasBytes))
	binary.LittleEndian.PutUint32(payload[104:], uint32(args.UpWeightBytes))
	binary.LittleEndian.PutUint32(payload[108:], uint32(args.UpScaleBytes))
	binary.LittleEndian.PutUint32(payload[112:], uint32(args.UpBiasBytes))
	binary.LittleEndian.PutUint32(payload[116:], uint32(args.OutputBytes))
	binary.LittleEndian.PutUint32(payload[120:], batch)
	return payload, nil
}

func (args hipMLXQ4GELUTanhProjLaunchArgs) Binary() ([]byte, error) {
	if args.InputPointer == 0 || args.WeightPointer == 0 || args.ScalePointer == 0 ||
		args.BiasPointer == 0 || args.MultiplierPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "input, weight, scale, bias, multiplier, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	if bits != hipMLXQ4ProjectionBits {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "only 4-bit MLX affine projection is supported", nil)
	}
	if cols%8 != 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "cols must be divisible by 8 for q4 packing", nil)
	}
	if cols%groupSize != 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "cols must be divisible by group size", nil)
	}
	packedPerRow := uint64(cols / 8)
	groupsPerRow := uint64(cols / groupSize)
	if args.InputBytes != uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "input byte count mismatch", nil)
	}
	if args.WeightBytes != uint64(rows)*packedPerRow*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "packed weight byte count mismatch", nil)
	}
	if args.ScaleBytes != uint64(rows)*groupsPerRow*2 || args.BiasBytes != uint64(rows)*groupsPerRow*2 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "scale/bias byte count mismatch", nil)
	}
	if args.MultiplierBytes != uint64(rows)*4 || args.OutputBytes != uint64(rows)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "multiplier/output byte count mismatch", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "weight bytes", args.WeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "scale bytes", args.ScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "bias bytes", args.BiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "multiplier bytes", args.MultiplierBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	payload := hipBorrowLaunchPacket(hipMLXQ4GELUTanhProjLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4GELUTanhProjLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ScalePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.MultiplierPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[56:], rows)
	binary.LittleEndian.PutUint32(payload[60:], cols)
	binary.LittleEndian.PutUint32(payload[64:], groupSize)
	binary.LittleEndian.PutUint32(payload[68:], bits)
	binary.LittleEndian.PutUint32(payload[72:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[76:], uint32(args.WeightBytes))
	binary.LittleEndian.PutUint32(payload[80:], uint32(args.ScaleBytes))
	binary.LittleEndian.PutUint32(payload[84:], uint32(args.BiasBytes))
	binary.LittleEndian.PutUint32(payload[88:], uint32(args.MultiplierBytes))
	binary.LittleEndian.PutUint32(payload[92:], uint32(args.OutputBytes))
	return payload, nil
}

func (args hipMLXQ4GELUTanhProjBatchLaunchArgs) Binary() ([]byte, error) {
	if args.InputPointer == 0 || args.WeightPointer == 0 || args.ScalePointer == 0 ||
		args.BiasPointer == 0 || args.MultiplierPointer == 0 || args.OutputPointer == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "input, weight, scale, bias, multiplier, and output pointers are required", nil)
	}
	rows, err := rocmDeviceKVPositiveUint32("rows", args.Rows)
	if err != nil {
		return nil, err
	}
	cols, err := rocmDeviceKVPositiveUint32("cols", args.Cols)
	if err != nil {
		return nil, err
	}
	batch, err := rocmDeviceKVPositiveUint32("batch", args.Batch)
	if err != nil {
		return nil, err
	}
	groupSize, err := rocmDeviceKVPositiveUint32("group size", args.GroupSize)
	if err != nil {
		return nil, err
	}
	bits, err := rocmDeviceKVPositiveUint32("bits", args.Bits)
	if err != nil {
		return nil, err
	}
	if bits != hipMLXQ4ProjectionBits {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "only 4-bit MLX affine projection is supported", nil)
	}
	if cols%8 != 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "cols must be divisible by 8 for q4 packing", nil)
	}
	if cols%groupSize != 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "cols must be divisible by group size", nil)
	}
	packedPerRow := uint64(cols / 8)
	groupsPerRow := uint64(cols / groupSize)
	if args.InputBytes != uint64(batch)*uint64(cols)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "input byte count mismatch", nil)
	}
	if args.WeightBytes != uint64(rows)*packedPerRow*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "packed weight byte count mismatch", nil)
	}
	if args.ScaleBytes != uint64(rows)*groupsPerRow*2 || args.BiasBytes != uint64(rows)*groupsPerRow*2 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "scale/bias byte count mismatch", nil)
	}
	if args.MultiplierBytes != uint64(batch)*uint64(rows)*4 || args.OutputBytes != uint64(batch)*uint64(rows)*4 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "multiplier/output byte count mismatch", nil)
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "input bytes", args.InputBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "weight bytes", args.WeightBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "scale bytes", args.ScaleBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "bias bytes", args.BiasBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "multiplier bytes", args.MultiplierBytes); err != nil {
		return nil, err
	}
	if err := hipProjectionUint32Bytes("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "output bytes", args.OutputBytes); err != nil {
		return nil, err
	}
	payload := hipBorrowLaunchPacket(hipMLXQ4GELUTanhProjBatchLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipMLXQ4GELUTanhProjBatchLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.InputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.WeightPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.ScalePointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.BiasPointer))
	binary.LittleEndian.PutUint64(payload[40:], uint64(args.MultiplierPointer))
	binary.LittleEndian.PutUint64(payload[48:], uint64(args.OutputPointer))
	binary.LittleEndian.PutUint32(payload[56:], rows)
	binary.LittleEndian.PutUint32(payload[60:], cols)
	binary.LittleEndian.PutUint32(payload[64:], batch)
	binary.LittleEndian.PutUint32(payload[68:], groupSize)
	binary.LittleEndian.PutUint32(payload[72:], bits)
	binary.LittleEndian.PutUint32(payload[76:], uint32(args.InputBytes))
	binary.LittleEndian.PutUint32(payload[80:], uint32(args.WeightBytes))
	binary.LittleEndian.PutUint32(payload[84:], uint32(args.ScaleBytes))
	binary.LittleEndian.PutUint32(payload[88:], uint32(args.BiasBytes))
	binary.LittleEndian.PutUint32(payload[92:], uint32(args.MultiplierBytes))
	binary.LittleEndian.PutUint32(payload[96:], uint32(args.OutputBytes))
	return payload, nil
}

func hipOrderedFloat32Key(value float32) uint32 {
	bits := math.Float32bits(value)
	if bits&0x80000000 != 0 {
		return ^bits
	}
	return bits ^ 0x80000000
}

func hipFloat32FromOrderedKey(key uint32) float32 {
	if key&0x80000000 != 0 {
		return math.Float32frombits(key ^ 0x80000000)
	}
	return math.Float32frombits(^key)
}

func hipPackGreedyBest(score float32, tokenID int) uint64 {
	return uint64(hipOrderedFloat32Key(score))<<32 | uint64(^uint32(tokenID))
}

func hipUnpackGreedyBest(packed uint64, softcap float32, vocabSize int) (hipGreedySampleResult, error) {
	if vocabSize <= 0 {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "vocab size must be positive", nil)
	}
	if packed == 0 {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "greedy projection did not produce a result", nil)
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "softcap must be non-negative and finite", nil)
	}
	tokenID := int(^uint32(packed))
	if tokenID < 0 || tokenID >= vocabSize {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "greedy projection token is out of range", nil)
	}
	score := hipFloat32FromOrderedKey(uint32(packed >> 32))
	if softcap > 0 {
		score = float32(math.Tanh(float64(score/softcap))) * softcap
	}
	return hipGreedySampleResult{TokenID: tokenID, Score: score}, nil
}

func hipProjectionWeightEncodingCode(req hipProjectionRequest) (uint32, error) {
	switch {
	case len(req.F32) > 0 && len(req.FP16) == 0 && len(req.BF16) == 0 && len(req.Q8) == 0:
		return hipProjectionWeightEncodingF32, nil
	case len(req.FP16) > 0 && len(req.F32) == 0 && len(req.BF16) == 0 && len(req.Q8) == 0:
		return hipProjectionWeightEncodingFP16, nil
	case len(req.BF16) > 0 && len(req.F32) == 0 && len(req.FP16) == 0 && len(req.Q8) == 0:
		return hipProjectionWeightEncodingBF16, nil
	case len(req.Q8) > 0 && len(req.F32) == 0 && len(req.FP16) == 0 && len(req.BF16) == 0:
		return hipProjectionWeightEncodingQ8, nil
	default:
		return 0, core.E("rocm.hip.ProjectionLaunch", "exactly one projection weight encoding is required", nil)
	}
}

func hipRunMLXQ4ProjectionKernel(ctx context.Context, driver nativeHIPDriver, req hipMLXQ4ProjectionRequest) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	buffers, err := req.deviceBuffers(driver)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()
	launch, err := req.launchArgs(buffers)
	if err != nil {
		return nil, err
	}
	launchBytes, err := launch.Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipMLXQ4ProjectionLaunchConfig(launchBytes, req.Rows)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	output, err := buffers.ReadOutput()
	if err != nil {
		return nil, err
	}
	success = true
	if err := buffers.Close(); err != nil {
		return nil, err
	}
	return output, nil
}

func hipRunMLXQ4ProjectionKernelWithDeviceWeights(ctx context.Context, driver nativeHIPDriver, input []float32, weightPointer, scalePointer, biasPointer nativeDevicePointer, weightBytes, scaleBytes, biasBytes uint64, rows, cols, groupSize int) ([]float32, error) {
	return hipRunMLXQ4ProjectionKernelWithDeviceWeightConfig(ctx, driver, input, hipMLXQ4DeviceWeightConfig{
		WeightPointer: weightPointer,
		ScalePointer:  scalePointer,
		BiasPointer:   biasPointer,
		WeightBytes:   weightBytes,
		ScaleBytes:    scaleBytes,
		BiasBytes:     biasBytes,
		Rows:          rows,
		Cols:          cols,
		GroupSize:     groupSize,
	})
}

func hipRunMLXQ4ProjectionKernelWithDeviceWeightConfig(ctx context.Context, driver nativeHIPDriver, input []float32, cfg hipMLXQ4DeviceWeightConfig) ([]float32, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if err := cfg.validate(input); err != nil {
		return nil, err
	}
	inputPayload, err := hipFloat32Payload(input)
	if err != nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "encode input", err)
	}
	inputBuffer, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection input", inputPayload, len(input))
	if err != nil {
		return nil, err
	}
	defer inputBuffer.Close()
	output, err := hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, inputBuffer, cfg)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	return hipReadFloat32DeviceOutput(output, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection output", cfg.Rows)
}

func hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig) (*hipDeviceByteBuffer, error) {
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device input is required", nil)
	}
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return nil, err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device input byte count mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection output", uint64(cfg.Rows*4), cfg.Rows)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunMLXQ4ProjectionKernelWithDeviceInputOutput(ctx, driver, input, cfg, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunMLXQ4ProjectionKernelWithDeviceInputOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device input is required", nil)
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection device input byte count mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != cfg.Rows || output.SizeBytes() != uint64(cfg.Rows*4) {
		return core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection output shape mismatch", nil)
	}
	launchBytes, err := (hipMLXQ4ProjectionLaunchArgs{
		InputPointer:  input.Pointer(),
		WeightPointer: cfg.WeightPointer,
		ScalePointer:  cfg.ScalePointer,
		BiasPointer:   cfg.BiasPointer,
		OutputPointer: output.Pointer(),
		Rows:          cfg.Rows,
		Cols:          cfg.Cols,
		GroupSize:     cfg.GroupSize,
		Bits:          hipMLXQ4ProjectionBits,
		InputBytes:    input.SizeBytes(),
		WeightBytes:   cfg.WeightBytes,
		ScaleBytes:    cfg.ScaleBytes,
		BiasBytes:     cfg.BiasBytes,
		OutputBytes:   output.SizeBytes(),
	}).Binary()
	if err != nil {
		return err
	}
	config, err := hipMLXQ4ProjectionLaunchConfig(launchBytes, cfg.Rows)
	if err != nil {
		return err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return err
	}
	return nil
}

func hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, batch int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch device input is required", nil)
	}
	if err := cfg.validateBatchInputCount(input.Count(), batch); err != nil {
		return nil, err
	}
	if input.SizeBytes() != uint64(batch*cfg.Cols*4) {
		return nil, core.E("rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch device input byte count mismatch", nil)
	}
	outputCount := batch * cfg.Rows
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch output", uint64(outputCount*4), outputCount)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipMLXQ4ProjectionBatchLaunchArgs{
		InputPointer:  input.Pointer(),
		WeightPointer: cfg.WeightPointer,
		ScalePointer:  cfg.ScalePointer,
		BiasPointer:   cfg.BiasPointer,
		OutputPointer: output.Pointer(),
		Rows:          cfg.Rows,
		Cols:          cfg.Cols,
		Batch:         batch,
		GroupSize:     cfg.GroupSize,
		Bits:          hipMLXQ4ProjectionBits,
		InputBytes:    input.SizeBytes(),
		WeightBytes:   cfg.WeightBytes,
		ScaleBytes:    cfg.ScaleBytes,
		BiasBytes:     cfg.BiasBytes,
		OutputBytes:   output.SizeBytes(),
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipMLXQ4ProjectionBatchLaunchConfig(launchBytes, cfg.Rows, batch)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunMLXQ4TripleProjectionKernelWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, firstCfg, secondCfg, thirdCfg hipMLXQ4DeviceWeightConfig) (*hipDeviceByteBuffer, *hipDeviceByteBuffer, *hipDeviceByteBuffer, *hipDeviceByteBuffer, error) {
	output, firstView, secondView, thirdView, err := hipRunMLXQ4TripleProjectionKernelWithDeviceInputViews(ctx, driver, input, firstCfg, secondCfg, thirdCfg)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	first := firstView
	second := secondView
	third := thirdView
	return output, &first, &second, &third, nil
}

func hipRunMLXQ4TripleProjectionKernelWithDeviceInputViews(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, firstCfg, secondCfg, thirdCfg hipMLXQ4DeviceWeightConfig) (*hipDeviceByteBuffer, hipDeviceByteBuffer, hipDeviceByteBuffer, hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input == nil || input.Pointer() == 0 {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection device input is required", nil)
	}
	if firstCfg.Cols != secondCfg.Cols || firstCfg.Cols != thirdCfg.Cols ||
		firstCfg.GroupSize != secondCfg.GroupSize || firstCfg.GroupSize != thirdCfg.GroupSize {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "triple projection input shapes must match", nil)
	}
	for _, cfg := range []hipMLXQ4DeviceWeightConfig{firstCfg, secondCfg, thirdCfg} {
		if err := cfg.validateInputCount(input.Count()); err != nil {
			return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
		}
		if input.SizeBytes() != uint64(cfg.Cols*4) {
			return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection device input byte count mismatch", nil)
		}
	}
	totalRows := firstCfg.Rows + secondCfg.Rows + thirdCfg.Rows
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection output", uint64(totalRows*4), totalRows)
	if err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	first, second, third, err := hipRunMLXQ4TripleProjectionKernelWithDeviceInputViewsOutput(ctx, driver, input, firstCfg, secondCfg, thirdCfg, output)
	if err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	success = true
	return output, first, second, third, nil
}

func hipRunMLXQ4TripleProjectionKernelWithDeviceInputViewsOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, firstCfg, secondCfg, thirdCfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer) (hipDeviceByteBuffer, hipDeviceByteBuffer, hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input == nil || input.Pointer() == 0 {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection device input is required", nil)
	}
	if firstCfg.Cols != secondCfg.Cols || firstCfg.Cols != thirdCfg.Cols ||
		firstCfg.GroupSize != secondCfg.GroupSize || firstCfg.GroupSize != thirdCfg.GroupSize {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "triple projection input shapes must match", nil)
	}
	for _, cfg := range []hipMLXQ4DeviceWeightConfig{firstCfg, secondCfg, thirdCfg} {
		if err := cfg.validateInputCount(input.Count()); err != nil {
			return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
		}
		if input.SizeBytes() != uint64(cfg.Cols*4) {
			return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection device input byte count mismatch", nil)
		}
	}
	totalRows := firstCfg.Rows + secondCfg.Rows + thirdCfg.Rows
	if output == nil || output.Pointer() == 0 || output.Count() != totalRows || output.SizeBytes() != uint64(totalRows*4) {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4TripleProjectionLaunch", "MLX q4 triple projection output shape mismatch", nil)
	}
	launchBytes, err := (hipMLXQ4TripleProjLaunchArgs{
		InputPointer:        input.Pointer(),
		OutputPointer:       output.Pointer(),
		FirstWeightPointer:  firstCfg.WeightPointer,
		FirstScalePointer:   firstCfg.ScalePointer,
		FirstBiasPointer:    firstCfg.BiasPointer,
		SecondWeightPointer: secondCfg.WeightPointer,
		SecondScalePointer:  secondCfg.ScalePointer,
		SecondBiasPointer:   secondCfg.BiasPointer,
		ThirdWeightPointer:  thirdCfg.WeightPointer,
		ThirdScalePointer:   thirdCfg.ScalePointer,
		ThirdBiasPointer:    thirdCfg.BiasPointer,
		FirstRows:           firstCfg.Rows,
		SecondRows:          secondCfg.Rows,
		ThirdRows:           thirdCfg.Rows,
		Cols:                firstCfg.Cols,
		GroupSize:           firstCfg.GroupSize,
		Bits:                hipMLXQ4ProjectionBits,
		InputBytes:          input.SizeBytes(),
		OutputBytes:         output.SizeBytes(),
		FirstWeightBytes:    firstCfg.WeightBytes,
		FirstScaleBytes:     firstCfg.ScaleBytes,
		FirstBiasBytes:      firstCfg.BiasBytes,
		SecondWeightBytes:   secondCfg.WeightBytes,
		SecondScaleBytes:    secondCfg.ScaleBytes,
		SecondBiasBytes:     secondCfg.BiasBytes,
		ThirdWeightBytes:    thirdCfg.WeightBytes,
		ThirdScaleBytes:     thirdCfg.ScaleBytes,
		ThirdBiasBytes:      thirdCfg.BiasBytes,
	}).Binary()
	if err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	config, err := hipMLXQ4TripleProjectionLaunchConfig(launchBytes, totalRows)
	if err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	first := hipDeviceByteBuffer{
		driver:    driver,
		pointer:   output.Pointer(),
		count:     firstCfg.Rows,
		sizeBytes: uint64(firstCfg.Rows * 4),
		borrowed:  true,
		label:     "MLX q4 triple projection first output",
	}
	secondOffset := nativeDevicePointer(firstCfg.Rows * 4)
	second := hipDeviceByteBuffer{
		driver:    driver,
		pointer:   output.Pointer() + secondOffset,
		count:     secondCfg.Rows,
		sizeBytes: uint64(secondCfg.Rows * 4),
		borrowed:  true,
		label:     "MLX q4 triple projection second output",
	}
	thirdOffset := nativeDevicePointer((firstCfg.Rows + secondCfg.Rows) * 4)
	third := hipDeviceByteBuffer{
		driver:    driver,
		pointer:   output.Pointer() + thirdOffset,
		count:     thirdCfg.Rows,
		sizeBytes: uint64(thirdCfg.Rows * 4),
		borrowed:  true,
		label:     "MLX q4 triple projection third output",
	}
	return first, second, third, nil
}

func hipRunMLXQ4PairProjectionKernelWithDeviceInputViews(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, firstCfg, secondCfg hipMLXQ4DeviceWeightConfig) (*hipDeviceByteBuffer, hipDeviceByteBuffer, hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input == nil || input.Pointer() == 0 {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection device input is required", nil)
	}
	if firstCfg.Cols != secondCfg.Cols || firstCfg.GroupSize != secondCfg.GroupSize {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "pair projection input shapes must match", nil)
	}
	if err := firstCfg.validateInputCount(input.Count()); err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input.SizeBytes() != uint64(firstCfg.Cols*4) {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection device input byte count mismatch", nil)
	}
	if err := secondCfg.validateInputCount(input.Count()); err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input.SizeBytes() != uint64(secondCfg.Cols*4) {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection device input byte count mismatch", nil)
	}
	totalRows := firstCfg.Rows + secondCfg.Rows
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection output", uint64(totalRows*4), totalRows)
	if err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	first, second, err := hipRunMLXQ4PairProjectionKernelWithDeviceInputViewsOutput(ctx, driver, input, firstCfg, secondCfg, output)
	if err != nil {
		return nil, hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	success = true
	return output, first, second, nil
}

func hipRunMLXQ4PairProjectionKernelWithDeviceInputViewsOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, firstCfg, secondCfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer) (hipDeviceByteBuffer, hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input == nil || input.Pointer() == 0 {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection device input is required", nil)
	}
	if firstCfg.Cols != secondCfg.Cols || firstCfg.GroupSize != secondCfg.GroupSize {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "pair projection input shapes must match", nil)
	}
	if err := firstCfg.validateInputCount(input.Count()); err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input.SizeBytes() != uint64(firstCfg.Cols*4) {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection device input byte count mismatch", nil)
	}
	if err := secondCfg.validateInputCount(input.Count()); err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if input.SizeBytes() != uint64(secondCfg.Cols*4) {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection device input byte count mismatch", nil)
	}
	totalRows := firstCfg.Rows + secondCfg.Rows
	if output == nil || output.Pointer() == 0 || output.Count() != totalRows || output.SizeBytes() != uint64(totalRows*4) {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, core.E("rocm.hip.MLXQ4PairProjectionLaunch", "MLX q4 pair projection output shape mismatch", nil)
	}
	launchBytes, err := (hipMLXQ4TripleProjLaunchArgs{
		InputPointer:        input.Pointer(),
		OutputPointer:       output.Pointer(),
		FirstWeightPointer:  firstCfg.WeightPointer,
		FirstScalePointer:   firstCfg.ScalePointer,
		FirstBiasPointer:    firstCfg.BiasPointer,
		SecondWeightPointer: secondCfg.WeightPointer,
		SecondScalePointer:  secondCfg.ScalePointer,
		SecondBiasPointer:   secondCfg.BiasPointer,
		FirstRows:           firstCfg.Rows,
		SecondRows:          secondCfg.Rows,
		Cols:                firstCfg.Cols,
		GroupSize:           firstCfg.GroupSize,
		Bits:                hipMLXQ4ProjectionBits,
		InputBytes:          input.SizeBytes(),
		OutputBytes:         output.SizeBytes(),
		FirstWeightBytes:    firstCfg.WeightBytes,
		FirstScaleBytes:     firstCfg.ScaleBytes,
		FirstBiasBytes:      firstCfg.BiasBytes,
		SecondWeightBytes:   secondCfg.WeightBytes,
		SecondScaleBytes:    secondCfg.ScaleBytes,
		SecondBiasBytes:     secondCfg.BiasBytes,
	}).Binary()
	if err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	config, err := hipMLXQ4PairProjectionLaunchConfig(launchBytes, totalRows)
	if err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipDeviceByteBuffer{}, hipDeviceByteBuffer{}, err
	}
	first := hipDeviceByteBuffer{
		driver:    driver,
		pointer:   output.Pointer(),
		count:     firstCfg.Rows,
		sizeBytes: uint64(firstCfg.Rows * 4),
		borrowed:  true,
		label:     "MLX q4 pair projection first output",
	}
	secondOffset := nativeDevicePointer(firstCfg.Rows * 4)
	second := hipDeviceByteBuffer{
		driver:    driver,
		pointer:   output.Pointer() + secondOffset,
		count:     secondCfg.Rows,
		sizeBytes: uint64(secondCfg.Rows * 4),
		borrowed:  true,
		label:     "MLX q4 pair projection second output",
	}
	return first, second, nil
}

func hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, gateCfg, upCfg hipMLXQ4DeviceWeightConfig) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "MLX q4 GELU tanh multiply device input is required", nil)
	}
	if gateCfg.Rows != upCfg.Rows || gateCfg.Cols != upCfg.Cols || gateCfg.GroupSize != upCfg.GroupSize {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "gate and up q4 projection shapes must match", nil)
	}
	if err := gateCfg.validateInputCount(input.Count()); err != nil {
		return nil, err
	}
	if err := upCfg.validateInputCount(input.Count()); err != nil {
		return nil, err
	}
	if input.SizeBytes() != uint64(gateCfg.Cols*4) {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "MLX q4 GELU tanh multiply device input byte count mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "MLX q4 GELU tanh multiply output", uint64(gateCfg.Rows*4), gateCfg.Rows)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInputOutput(ctx, driver, input, gateCfg, upCfg, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInputOutput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, gateCfg, upCfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "MLX q4 GELU tanh multiply device input is required", nil)
	}
	if gateCfg.Rows != upCfg.Rows || gateCfg.Cols != upCfg.Cols || gateCfg.GroupSize != upCfg.GroupSize {
		return core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "gate and up q4 projection shapes must match", nil)
	}
	if err := gateCfg.validateInputCount(input.Count()); err != nil {
		return err
	}
	if err := upCfg.validateInputCount(input.Count()); err != nil {
		return err
	}
	if input.SizeBytes() != uint64(gateCfg.Cols*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "MLX q4 GELU tanh multiply device input byte count mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != gateCfg.Rows || output.SizeBytes() != uint64(gateCfg.Rows*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "MLX q4 GELU tanh multiply output shape mismatch", nil)
	}
	launchBytes, err := (hipMLXQ4GELUTanhMulLaunchArgs{
		InputPointer:      input.Pointer(),
		GateWeightPointer: gateCfg.WeightPointer,
		GateScalePointer:  gateCfg.ScalePointer,
		GateBiasPointer:   gateCfg.BiasPointer,
		UpWeightPointer:   upCfg.WeightPointer,
		UpScalePointer:    upCfg.ScalePointer,
		UpBiasPointer:     upCfg.BiasPointer,
		OutputPointer:     output.Pointer(),
		Rows:              gateCfg.Rows,
		Cols:              gateCfg.Cols,
		GroupSize:         gateCfg.GroupSize,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        input.SizeBytes(),
		GateWeightBytes:   gateCfg.WeightBytes,
		GateScaleBytes:    gateCfg.ScaleBytes,
		GateBiasBytes:     gateCfg.BiasBytes,
		UpWeightBytes:     upCfg.WeightBytes,
		UpScaleBytes:      upCfg.ScaleBytes,
		UpBiasBytes:       upCfg.BiasBytes,
		OutputBytes:       output.SizeBytes(),
	}).Binary()
	if err != nil {
		return err
	}
	config, err := hipMLXQ4GELUTanhMultiplyLaunchConfig(launchBytes, gateCfg.Rows)
	if err != nil {
		return err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return err
	}
	return nil
}

func hipRunMLXQ4GELUTanhMultiplyBatchKernelWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, gateCfg, upCfg hipMLXQ4DeviceWeightConfig, batch int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "MLX q4 GELU tanh multiply batch device input is required", nil)
	}
	if gateCfg.Rows != upCfg.Rows || gateCfg.Cols != upCfg.Cols || gateCfg.GroupSize != upCfg.GroupSize {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "gate and up q4 projection shapes must match", nil)
	}
	if err := gateCfg.validateBatchInputCount(input.Count(), batch); err != nil {
		return nil, err
	}
	if err := upCfg.validateBatchInputCount(input.Count(), batch); err != nil {
		return nil, err
	}
	if input.SizeBytes() != uint64(batch*gateCfg.Cols*4) {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "MLX q4 GELU tanh multiply batch device input byte count mismatch", nil)
	}
	outputCount := batch * gateCfg.Rows
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "MLX q4 GELU tanh multiply batch output", uint64(outputCount*4), outputCount)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipMLXQ4GELUTanhMulBatchLaunchArgs{
		InputPointer:      input.Pointer(),
		GateWeightPointer: gateCfg.WeightPointer,
		GateScalePointer:  gateCfg.ScalePointer,
		GateBiasPointer:   gateCfg.BiasPointer,
		UpWeightPointer:   upCfg.WeightPointer,
		UpScalePointer:    upCfg.ScalePointer,
		UpBiasPointer:     upCfg.BiasPointer,
		OutputPointer:     output.Pointer(),
		Rows:              gateCfg.Rows,
		Cols:              gateCfg.Cols,
		GroupSize:         gateCfg.GroupSize,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        input.SizeBytes(),
		GateWeightBytes:   gateCfg.WeightBytes,
		GateScaleBytes:    gateCfg.ScaleBytes,
		GateBiasBytes:     gateCfg.BiasBytes,
		UpWeightBytes:     upCfg.WeightBytes,
		UpScaleBytes:      upCfg.ScaleBytes,
		UpBiasBytes:       upCfg.BiasBytes,
		OutputBytes:       output.SizeBytes(),
		Batch:             batch,
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipMLXQ4GELUTanhMultiplyBatchLaunchConfig(launchBytes, gateCfg.Rows, batch)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplier(ctx context.Context, driver nativeHIPDriver, input, multiplier *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection device input is required", nil)
	}
	if multiplier == nil || multiplier.Pointer() == 0 || multiplier.Count() != cfg.Rows || multiplier.SizeBytes() != uint64(cfg.Rows*4) {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection multiplier device buffer shape mismatch", nil)
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return nil, err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection device input byte count mismatch", nil)
	}
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection output", uint64(cfg.Rows*4), cfg.Rows)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	if err := hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplierOutput(ctx, driver, input, multiplier, cfg, output); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplierOutput(ctx context.Context, driver nativeHIPDriver, input, multiplier *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, output *hipDeviceByteBuffer) error {
	if err := hipContextErr(ctx); err != nil {
		return err
	}
	if driver == nil || !driver.Available() {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection device input is required", nil)
	}
	if multiplier == nil || multiplier.Pointer() == 0 || multiplier.Count() != cfg.Rows || multiplier.SizeBytes() != uint64(cfg.Rows*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection multiplier device buffer shape mismatch", nil)
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection device input byte count mismatch", nil)
	}
	if output == nil || output.Pointer() == 0 || output.Count() != cfg.Rows || output.SizeBytes() != uint64(cfg.Rows*4) {
		return core.E("rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection output shape mismatch", nil)
	}
	launchBytes, err := (hipMLXQ4GELUTanhProjLaunchArgs{
		InputPointer:      input.Pointer(),
		WeightPointer:     cfg.WeightPointer,
		ScalePointer:      cfg.ScalePointer,
		BiasPointer:       cfg.BiasPointer,
		MultiplierPointer: multiplier.Pointer(),
		OutputPointer:     output.Pointer(),
		Rows:              cfg.Rows,
		Cols:              cfg.Cols,
		GroupSize:         cfg.GroupSize,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        input.SizeBytes(),
		WeightBytes:       cfg.WeightBytes,
		ScaleBytes:        cfg.ScaleBytes,
		BiasBytes:         cfg.BiasBytes,
		MultiplierBytes:   multiplier.SizeBytes(),
		OutputBytes:       output.SizeBytes(),
	}).Binary()
	if err != nil {
		return err
	}
	config, err := hipMLXQ4GELUTanhProjectionLaunchConfig(launchBytes, cfg.Rows)
	if err != nil {
		return err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return err
	}
	return nil
}

func hipRunMLXQ4GELUTanhProjectionBatchKernelWithDeviceMultiplier(ctx context.Context, driver nativeHIPDriver, input, multiplier *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, batch int) (*hipDeviceByteBuffer, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "HIP driver is not available", nil)
	}
	if batch <= 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch size must be positive", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch device input is required", nil)
	}
	if multiplier == nil || multiplier.Pointer() == 0 || multiplier.Count() != batch*cfg.Rows || multiplier.SizeBytes() != uint64(batch*cfg.Rows*4) {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch multiplier device buffer shape mismatch", nil)
	}
	if err := cfg.validateInputCount(input.Count() / batch); err != nil {
		return nil, err
	}
	if input.Count() != batch*cfg.Cols || input.SizeBytes() != uint64(batch*cfg.Cols*4) {
		return nil, core.E("rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch device input byte count mismatch", nil)
	}
	outputCount := batch * cfg.Rows
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch output", uint64(outputCount*4), outputCount)
	if err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = output.Close()
		}
	}()
	launchBytes, err := (hipMLXQ4GELUTanhProjBatchLaunchArgs{
		InputPointer:      input.Pointer(),
		WeightPointer:     cfg.WeightPointer,
		ScalePointer:      cfg.ScalePointer,
		BiasPointer:       cfg.BiasPointer,
		MultiplierPointer: multiplier.Pointer(),
		OutputPointer:     output.Pointer(),
		Rows:              cfg.Rows,
		Cols:              cfg.Cols,
		Batch:             batch,
		GroupSize:         cfg.GroupSize,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        input.SizeBytes(),
		WeightBytes:       cfg.WeightBytes,
		ScaleBytes:        cfg.ScaleBytes,
		BiasBytes:         cfg.BiasBytes,
		MultiplierBytes:   multiplier.SizeBytes(),
		OutputBytes:       output.SizeBytes(),
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipMLXQ4GELUTanhProjectionBatchLaunchConfig(launchBytes, cfg.Rows, batch)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	success = true
	return output, nil
}

func hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInput(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32) (hipGreedySampleResult, error) {
	return hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBuffer(ctx, driver, input, cfg, softcap, nil)
}

func hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBuffer(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, best *hipDeviceByteBuffer) (hipGreedySampleResult, error) {
	return hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressBuffer(ctx, driver, input, cfg, softcap, best, nil)
}

func hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressBuffer(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, best *hipDeviceByteBuffer, suppress *hipDeviceTokenBuffer) (hipGreedySampleResult, error) {
	if err := hipContextErr(ctx); err != nil {
		return hipGreedySampleResult{}, err
	}
	if driver == nil || !driver.Available() {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "HIP driver is not available", nil)
	}
	if input == nil || input.Pointer() == 0 {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 projection device input is required", nil)
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return hipGreedySampleResult{}, err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 projection device input byte count mismatch", nil)
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "softcap must be non-negative and finite", nil)
	}
	ownsBest := false
	if best == nil {
		var err error
		best, err = hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 projection greedy best", hipMLXQ4ProjectionBestBytes, 1)
		if err != nil {
			return hipGreedySampleResult{}, err
		}
		ownsBest = true
	} else if best.Pointer() == 0 || best.Count() != 1 || best.SizeBytes() != hipMLXQ4ProjectionBestBytes {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 projection greedy best buffer shape mismatch", nil)
	}
	if suppress != nil && (suppress.Pointer() == 0 || suppress.Count() <= 0 || suppress.SizeBytes() != uint64(suppress.Count()*4)) {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 suppress token buffer shape mismatch", nil)
	}
	if ownsBest {
		defer best.Close()
	}
	if err := hipMemsetDevice(driver, best.Pointer(), 0, best.SizeBytes()); err != nil {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "initialize greedy best", err)
	}
	launchBytes, err := (hipMLXQ4ProjectionLaunchArgs{
		InputPointer:  input.Pointer(),
		WeightPointer: cfg.WeightPointer,
		ScalePointer:  cfg.ScalePointer,
		BiasPointer:   cfg.BiasPointer,
		OutputPointer: best.Pointer(),
		Rows:          cfg.Rows,
		Cols:          cfg.Cols,
		GroupSize:     cfg.GroupSize,
		Bits:          hipMLXQ4ProjectionBits,
		InputBytes:    input.SizeBytes(),
		WeightBytes:   cfg.WeightBytes,
		ScaleBytes:    cfg.ScaleBytes,
		BiasBytes:     cfg.BiasBytes,
		OutputBytes:   best.SizeBytes(),
	}).GreedyBinary()
	if suppress != nil {
		launchBytes, err = (hipMLXQ4ProjectionLaunchArgs{
			InputPointer:    input.Pointer(),
			WeightPointer:   cfg.WeightPointer,
			ScalePointer:    cfg.ScalePointer,
			BiasPointer:     cfg.BiasPointer,
			OutputPointer:   best.Pointer(),
			SuppressPointer: suppress.Pointer(),
			Rows:            cfg.Rows,
			Cols:            cfg.Cols,
			GroupSize:       cfg.GroupSize,
			Bits:            hipMLXQ4ProjectionBits,
			SuppressCount:   suppress.Count(),
			InputBytes:      input.SizeBytes(),
			WeightBytes:     cfg.WeightBytes,
			ScaleBytes:      cfg.ScaleBytes,
			BiasBytes:       cfg.BiasBytes,
			OutputBytes:     best.SizeBytes(),
		}).GreedyBinary()
	}
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	config, err := hipMLXQ4ProjectionGreedyLaunchConfig(launchBytes, cfg.Rows)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return hipGreedySampleResult{}, err
	}
	packed, err := hipReadDeviceUint64(driver, best.Pointer())
	if err != nil {
		return hipGreedySampleResult{}, core.E("rocm.hip.MLXQ4ProjectionGreedyLaunch", "copy greedy best", err)
	}
	return hipUnpackGreedyBest(packed, softcap, cfg.Rows)
}

type nativeHIPDeviceUint64Reader interface {
	CopyDeviceToHostUint64(pointer nativeDevicePointer) (uint64, error)
}

func hipReadDeviceUint64(driver nativeHIPDriver, pointer nativeDevicePointer) (uint64, error) {
	if reader, ok := driver.(nativeHIPDeviceUint64Reader); ok {
		return reader.CopyDeviceToHostUint64(pointer)
	}
	var payload [8]byte
	if err := driver.CopyDeviceToHost(pointer, payload[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(payload[:]), nil
}

func hipReadUint64DeviceOutput(buffer *hipDeviceByteBuffer, operation, label string, count int) ([]uint64, error) {
	if buffer == nil || buffer.Pointer() == 0 {
		return nil, core.E(operation, label+" device buffer is required", nil)
	}
	if count <= 0 {
		return nil, core.E(operation, label+" count must be positive", nil)
	}
	if buffer.Count() != count || buffer.SizeBytes() != uint64(count*8) {
		return nil, core.E(operation, label+" byte count mismatch", nil)
	}
	payload := make([]byte, count*8)
	if err := buffer.driver.CopyDeviceToHost(buffer.Pointer(), payload); err != nil {
		return nil, core.E(operation, "copy "+label, err)
	}
	values := make([]uint64, count)
	for index := range values {
		values[index] = binary.LittleEndian.Uint64(payload[index*8:])
	}
	return values, nil
}

func hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppress(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, best *hipDeviceByteBuffer, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) (hipGreedySampleResult, error) {
	greedy, err := hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBuffer(ctx, driver, input, cfg, softcap, best)
	if err != nil || !hipTokenIsSuppressed(int32(greedy.TokenID), suppressTokens) {
		return greedy, err
	}
	if workspace != nil {
		suppress, err := workspace.EnsureSuppressTokenBuffer(driver, suppressTokens)
		if err != nil {
			return hipGreedySampleResult{}, err
		}
		return hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInputBufferSuppressBuffer(ctx, driver, input, cfg, softcap, best, suppress)
	}
	logitsBuffer, err := hipRunMLXQ4ProjectionKernelWithDeviceInput(ctx, driver, input, cfg)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	defer logitsBuffer.Close()
	logits, err := hipReadFloat32DeviceOutput(logitsBuffer, "rocm.hip.MLXQ4ProjectionGreedyLaunch", "MLX q4 suppressed projection logits", cfg.Rows)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	logits, err = hipGemma4Q4SoftcapLogits(logits, softcap)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	tokenID, score, err := hipReferenceGreedySampleSuppress(logits, suppressTokens)
	if err != nil {
		return hipGreedySampleResult{}, err
	}
	return hipGreedySampleResult{TokenID: tokenID, Score: score}, nil
}

func hipRunPackedTopKKernelWithWorkspace(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, inputCount, topK int, workspace *hipAttentionHeadsChunkedWorkspace) (*hipDeviceByteBuffer, int, error) {
	if input == nil || input.Pointer() == 0 {
		return nil, 0, core.E("rocm.hip.PackedTopKLaunch", "packed score input is required", nil)
	}
	if inputCount <= 0 || input.Count() != inputCount || input.SizeBytes() != uint64(inputCount*hipMLXQ4ProjectionBestBytes) {
		return nil, 0, core.E("rocm.hip.PackedTopKLaunch", "packed score input shape mismatch", nil)
	}
	if topK <= 0 || topK > hipPackedTopKMaxK {
		return nil, 0, core.E("rocm.hip.PackedTopKLaunch", "top-k must be within kernel maximum", nil)
	}
	if workspace == nil {
		return nil, 0, core.E("rocm.hip.PackedTopKLaunch", "attention workspace is required", nil)
	}
	chunkCount := (inputCount + hipPackedTopKChunkSize - 1) / hipPackedTopKChunkSize
	outputCount := chunkCount * topK
	output, err := workspace.EnsureProjectionTopKOutput(driver, outputCount)
	if err != nil {
		return nil, 0, err
	}
	launchBytes, err := (hipPackedTopKLaunchArgs{
		InputPointer:  input.Pointer(),
		OutputPointer: output.Pointer(),
		InputCount:    inputCount,
		OutputCount:   outputCount,
		TopK:          topK,
		ChunkSize:     hipPackedTopKChunkSize,
		InputBytes:    input.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
	}).Binary()
	if err != nil {
		return nil, 0, err
	}
	config, err := hipPackedTopKLaunchConfig(launchBytes, chunkCount)
	if err != nil {
		return nil, 0, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, 0, err
	}
	return output, outputCount, nil
}

func hipRunMLXQ4ProjectionSoftcapScoreKernelWithDeviceInputBufferSuppress(ctx context.Context, driver nativeHIPDriver, input *hipDeviceByteBuffer, cfg hipMLXQ4DeviceWeightConfig, softcap float32, topK int, suppressTokens []int32, workspace *hipAttentionHeadsChunkedWorkspace) ([]hipGreedySampleResult, error) {
	if input == nil || input.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "MLX q4 projection device input is required", nil)
	}
	if err := cfg.validateInputCount(input.Count()); err != nil {
		return nil, err
	}
	if input.SizeBytes() != uint64(cfg.Cols*4) {
		return nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "MLX q4 projection device input byte count mismatch", nil)
	}
	if topK <= 0 || topK > cfg.Rows {
		return nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "top-k must be within vocabulary size", nil)
	}
	if softcap < 0 || math.IsNaN(float64(softcap)) || math.IsInf(float64(softcap), 0) {
		return nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "softcap must be non-negative and finite", nil)
	}
	var suppress *hipDeviceTokenBuffer
	var err error
	if len(suppressTokens) > 0 {
		if workspace != nil {
			suppress, err = workspace.EnsureSuppressTokenBuffer(driver, suppressTokens)
		} else {
			suppress, err = hipUploadTokenIDs(driver, suppressTokens)
		}
		if err != nil {
			return nil, err
		}
		if workspace == nil {
			defer suppress.Close()
		}
	}
	var scores *hipDeviceByteBuffer
	if workspace != nil {
		scores, err = workspace.EnsureProjectionScoreOutput(driver, cfg.Rows)
		if err != nil {
			return nil, err
		}
	} else {
		scores, err = hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4ProjectionScoresLaunch", "MLX q4 projection packed scores", uint64(cfg.Rows*hipMLXQ4ProjectionBestBytes), cfg.Rows)
		if err != nil {
			return nil, err
		}
		defer scores.Close()
	}
	launchArgs := hipMLXQ4ProjectionLaunchArgs{
		InputPointer:  input.Pointer(),
		WeightPointer: cfg.WeightPointer,
		ScalePointer:  cfg.ScalePointer,
		BiasPointer:   cfg.BiasPointer,
		OutputPointer: scores.Pointer(),
		Rows:          cfg.Rows,
		Cols:          cfg.Cols,
		GroupSize:     cfg.GroupSize,
		Bits:          hipMLXQ4ProjectionBits,
		InputBytes:    input.SizeBytes(),
		WeightBytes:   cfg.WeightBytes,
		ScaleBytes:    cfg.ScaleBytes,
		BiasBytes:     cfg.BiasBytes,
		OutputBytes:   scores.SizeBytes(),
	}
	if suppress != nil {
		launchArgs.SuppressPointer = suppress.Pointer()
		launchArgs.SuppressCount = suppress.Count()
	}
	launchBytes, err := launchArgs.ScoresBinary()
	if err != nil {
		return nil, err
	}
	config, err := hipMLXQ4ProjectionScoresLaunchConfig(launchBytes, cfg.Rows)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	var top []uint64
	if workspace != nil {
		partial, partialCount, err := hipRunPackedTopKKernelWithWorkspace(ctx, driver, scores, cfg.Rows, topK, workspace)
		if err != nil {
			return nil, err
		}
		payload, err := workspace.ProjectionTopKPayload(partialCount)
		if err != nil {
			return nil, err
		}
		if err := driver.CopyDeviceToHost(partial.Pointer(), payload); err != nil {
			return nil, core.E("rocm.hip.PackedTopKLaunch", "copy packed top-k partial scores", err)
		}
		top = hipTopPackedScoresBytesInto(payload, topK, workspace.ProjectionTopPacked)
		workspace.ProjectionTopPacked = top
	} else {
		packed, err := hipReadUint64DeviceOutput(scores, "rocm.hip.MLXQ4ProjectionScoresLaunch", "MLX q4 projection packed scores", cfg.Rows)
		if err != nil {
			return nil, err
		}
		top = hipTopPackedScores(packed, topK)
	}
	var candidates []hipGreedySampleResult
	if workspace != nil {
		candidates = workspace.ProjectionCandidates[:0]
		if cap(candidates) < len(top) {
			candidates = make([]hipGreedySampleResult, 0, len(top))
		}
	} else {
		candidates = make([]hipGreedySampleResult, 0, len(top))
	}
	for _, value := range top {
		candidate, err := hipUnpackGreedyBest(value, softcap, cfg.Rows)
		if err != nil {
			return nil, err
		}
		candidates = append(candidates, candidate)
	}
	if len(candidates) == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionScoresLaunch", "score projection did not produce candidates", nil)
	}
	if workspace != nil {
		workspace.ProjectionCandidates = candidates
	}
	return candidates, nil
}

func hipTopPackedScores(values []uint64, topK int) []uint64 {
	if topK <= 0 || len(values) == 0 {
		return nil
	}
	top := make([]uint64, 0, min(topK, len(values)))
	for _, value := range values {
		if value == 0 {
			continue
		}
		insert := len(top)
		for insert > 0 && value > top[insert-1] {
			insert--
		}
		if insert >= topK {
			continue
		}
		if len(top) < topK {
			top = append(top, 0)
			copy(top[insert+1:], top[insert:])
		} else {
			copy(top[insert+1:], top[insert:len(top)-1])
		}
		top[insert] = value
	}
	return top
}

func hipTopPackedScoresBytes(payload []byte, topK int) []uint64 {
	return hipTopPackedScoresBytesInto(payload, topK, nil)
}

func hipTopPackedScoresBytesInto(payload []byte, topK int, top []uint64) []uint64 {
	if topK <= 0 || len(payload) == 0 {
		return nil
	}
	top = top[:0]
	if cap(top) < min(topK, len(payload)/hipMLXQ4ProjectionBestBytes) {
		top = make([]uint64, 0, min(topK, len(payload)/hipMLXQ4ProjectionBestBytes))
	}
	for offset := 0; offset+hipMLXQ4ProjectionBestBytes <= len(payload); offset += hipMLXQ4ProjectionBestBytes {
		value := binary.LittleEndian.Uint64(payload[offset:])
		if value == 0 {
			continue
		}
		insert := len(top)
		for insert > 0 && value > top[insert-1] {
			insert--
		}
		if insert >= topK {
			continue
		}
		if len(top) < topK {
			top = append(top, 0)
			copy(top[insert+1:], top[insert:])
		} else {
			copy(top[insert+1:], top[insert:len(top)-1])
		}
		top[insert] = value
	}
	return top
}

func hipMLXQ4ProjectionLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 projection row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4Proj,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4ProjectionScoresLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 projection score row blocks", (rows+hipMLXQ4ProjectionGreedyRowsPerBlock-1)/hipMLXQ4ProjectionGreedyRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4ProjScores,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipPackedTopKLaunchConfig(args []byte, chunkCount int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("packed top-k chunks", chunkCount)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNamePackedTopK,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipPackedTopKBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipProjectionBatchLaunchConfig(args []byte, rows, batch int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("projection batch row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	gridY, err := rocmDeviceKVPositiveUint32("projection batch token blocks", (batch+hipMLXQ4ProjectionBatchTokensPerBlock-1)/hipMLXQ4ProjectionBatchTokensPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameProjectionBatch,
		Args:   args,
		GridX:  gridX,
		GridY:  gridY,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4ProjectionBatchLaunchConfig(args []byte, rows, batch int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 projection batch row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	gridY, err := rocmDeviceKVPositiveUint32("MLX q4 projection batch token blocks", (batch+hipMLXQ4ProjectionBatchTokensPerBlock-1)/hipMLXQ4ProjectionBatchTokensPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4ProjBatch,
		Args:   args,
		GridX:  gridX,
		GridY:  gridY,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4ProjectionGreedyLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 projection row blocks", (rows+hipMLXQ4ProjectionGreedyRowsPerBlock-1)/hipMLXQ4ProjectionGreedyRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4ProjGreedy,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4TripleProjectionLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 triple projection row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4TripleProj,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4PairProjectionLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 pair projection row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4PairProj,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4GELUTanhMultiplyLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 GELU tanh multiply row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4GELUTanhMul,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4GELUTanhMultiplyBatchLaunchConfig(args []byte, rows, batch int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 GELU tanh multiply batch row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	gridY, err := rocmDeviceKVPositiveUint32("MLX q4 GELU tanh multiply batch token blocks", (batch+hipMLXQ4ProjectionBatchTokensPerBlock-1)/hipMLXQ4ProjectionBatchTokensPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4GELUTanhMulBatch,
		Args:   args,
		GridX:  gridX,
		GridY:  gridY,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4GELUTanhProjectionLaunchConfig(args []byte, rows int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 GELU tanh projection row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4GELUTanhProj,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipMLXQ4GELUTanhProjectionBatchLaunchConfig(args []byte, rows, batch int) (hipKernelLaunchConfig, error) {
	gridX, err := rocmDeviceKVPositiveUint32("MLX q4 GELU tanh projection batch row blocks", (rows+hipMLXQ4ProjectionRowsPerBlock-1)/hipMLXQ4ProjectionRowsPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	gridY, err := rocmDeviceKVPositiveUint32("MLX q4 GELU tanh projection batch token blocks", (batch+hipMLXQ4ProjectionBatchTokensPerBlock-1)/hipMLXQ4ProjectionBatchTokensPerBlock)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4GELUTanhProjBatch,
		Args:   args,
		GridX:  gridX,
		GridY:  gridY,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipUploadByteBuffer(driver nativeHIPDriver, operation, label string, payload []byte, count int) (*hipDeviceByteBuffer, error) {
	if len(payload) == 0 {
		return nil, core.E(operation, label+" payload is empty", nil)
	}
	buffer, err := hipAllocateByteBuffer(driver, operation, label, uint64(len(payload)), count)
	if err != nil {
		return nil, err
	}
	if err := hipCopyHostToDevice(driver, buffer.pointer, payload); err != nil {
		_ = buffer.Close()
		return nil, core.E(operation, "copy "+label, err)
	}
	return buffer, nil
}

func hipAllocateByteBuffer(driver nativeHIPDriver, operation, label string, sizeBytes uint64, count int) (*hipDeviceByteBuffer, error) {
	if driver == nil {
		return nil, core.E(operation, "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return nil, core.E(operation, "HIP driver is not available", nil)
	}
	if sizeBytes == 0 || count <= 0 {
		return nil, core.E(operation, label+" size must be positive", nil)
	}
	if pointer, ok := hipDeviceByteBufferPoolTake(driver, sizeBytes); ok {
		return &hipDeviceByteBuffer{
			driver:    driver,
			pointer:   pointer,
			count:     count,
			sizeBytes: sizeBytes,
			pooled:    true,
			label:     label,
		}, nil
	}
	pointer, err := driver.Malloc(sizeBytes)
	if err != nil {
		return nil, core.E(operation, "allocate "+label, err)
	}
	return &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   pointer,
		count:     count,
		sizeBytes: sizeBytes,
		pooled:    hipDeviceByteBufferPoolEnabled(),
		label:     label,
	}, nil
}

func hipDeviceByteBufferPoolEnabled() bool {
	return os.Getenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL") != "1"
}

func hipDeviceByteBufferPoolTake(driver nativeHIPDriver, sizeBytes uint64) (nativeDevicePointer, bool) {
	if !hipDeviceByteBufferPoolEnabled() {
		return 0, false
	}
	hipDeviceByteBufferPool.Lock()
	defer hipDeviceByteBufferPool.Unlock()
	entries := hipDeviceByteBufferPool.entries[sizeBytes]
	for index := len(entries) - 1; index >= 0; index-- {
		entry := entries[index]
		if entry.driver != driver {
			continue
		}
		pointer := entry.pointer
		entries[index] = entries[len(entries)-1]
		entries[len(entries)-1] = hipDeviceByteBufferPoolEntry{}
		entries = entries[:len(entries)-1]
		if hipDeviceByteBufferPool.bytes >= sizeBytes {
			hipDeviceByteBufferPool.bytes -= sizeBytes
		} else {
			hipDeviceByteBufferPool.bytes = 0
		}
		hipDeviceByteBufferPool.entries[sizeBytes] = entries
		return pointer, true
	}
	return 0, false
}

func hipDeviceByteBufferPoolPut(driver nativeHIPDriver, pointer nativeDevicePointer, sizeBytes uint64) bool {
	if !hipDeviceByteBufferPoolEnabled() || driver == nil || pointer == 0 || sizeBytes == 0 {
		return false
	}
	hipDeviceByteBufferPool.Lock()
	defer hipDeviceByteBufferPool.Unlock()
	entries := hipDeviceByteBufferPool.entries[sizeBytes]
	if len(entries) >= hipDeviceByteBufferPoolMaxPerSize || hipDeviceByteBufferPool.bytes+sizeBytes > hipDeviceByteBufferPoolMaxBytes {
		return false
	}
	hipDeviceByteBufferPool.entries[sizeBytes] = append(entries, hipDeviceByteBufferPoolEntry{driver: driver, pointer: pointer})
	hipDeviceByteBufferPool.bytes += sizeBytes
	return true
}

func hipBorrowDeviceByteBuffer(driver nativeHIPDriver, label string, pointer nativeDevicePointer, sizeBytes uint64, count int) *hipDeviceByteBuffer {
	return &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   pointer,
		count:     count,
		sizeBytes: sizeBytes,
		borrowed:  true,
		label:     label,
	}
}

func (buffer *hipDeviceByteBuffer) Pointer() nativeDevicePointer {
	if buffer == nil || buffer.closed {
		return 0
	}
	return buffer.pointer
}

func (buffer *hipDeviceByteBuffer) Count() int {
	if buffer == nil || buffer.closed {
		return 0
	}
	return buffer.count
}

func (buffer *hipDeviceByteBuffer) SizeBytes() uint64 {
	if buffer == nil || buffer.closed {
		return 0
	}
	return buffer.sizeBytes
}

func (buffer *hipDeviceByteBuffer) Close() error {
	if buffer == nil || buffer.closed {
		return nil
	}
	if buffer.pointer != 0 && !buffer.borrowed {
		if buffer.driver == nil {
			return core.E("rocm.hip.ProjectionLaunch", "HIP driver is nil", nil)
		}
		if buffer.pooled && hipDeviceByteBufferPoolPut(buffer.driver, buffer.pointer, buffer.sizeBytes) {
			buffer.pointer = 0
			buffer.closed = true
			return nil
		}
		if err := buffer.driver.Free(buffer.pointer); err != nil {
			return core.E("rocm.hip.ProjectionLaunch", "free "+firstNonEmptyString(buffer.label, "device buffer"), err)
		}
		buffer.pointer = 0
	}
	buffer.closed = true
	return nil
}

func (buffers *hipProjectionDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Bias, buffers.Weights, buffers.Input} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipProjectionDeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "projection output buffer is required", nil)
	}
	if buffers.Rows <= 0 || buffers.Output.Count() != buffers.Rows || buffers.Output.SizeBytes() != uint64(buffers.Rows*4) {
		return nil, core.E("rocm.hip.ProjectionLaunch", "projection output byte count mismatch", nil)
	}
	payload := make([]byte, buffers.Output.SizeBytes())
	if err := buffers.Output.driver.CopyDeviceToHost(buffers.Output.Pointer(), payload); err != nil {
		return nil, core.E("rocm.hip.ProjectionLaunch", "copy projection output", err)
	}
	values, err := hipFloat32PayloadValues(payload)
	if err != nil {
		return nil, err
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E("rocm.hip.ProjectionLaunch", "projection output values must be finite", nil)
	}
	return values, nil
}

func hipFloat32Payload(values []float32) ([]byte, error) {
	if len(values) == 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "float32 payload is empty", nil)
	}
	payload := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(payload[index*4:], math.Float32bits(value))
	}
	return payload, nil
}

func hipUint16Payload(values []uint16) ([]byte, error) {
	if len(values) == 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "uint16 payload is empty", nil)
	}
	payload := make([]byte, len(values)*2)
	for index, value := range values {
		binary.LittleEndian.PutUint16(payload[index*2:], value)
	}
	return payload, nil
}

func hipUint32Payload(values []uint32) ([]byte, error) {
	if len(values) == 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "uint32 payload is empty", nil)
	}
	payload := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(payload[index*4:], value)
	}
	return payload, nil
}

func hipInt8Payload(values []int8) []byte {
	payload := make([]byte, len(values))
	for index, value := range values {
		payload[index] = byte(value)
	}
	return payload
}

func (buffers *hipMLXQ4ProjectionDeviceBuffers) Close() error {
	if buffers == nil {
		return nil
	}
	var lastErr error
	for _, buffer := range []*hipDeviceByteBuffer{buffers.Output, buffers.Biases, buffers.Scales, buffers.Weight, buffers.Input} {
		if err := buffer.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

func (buffers *hipMLXQ4ProjectionDeviceBuffers) ReadOutput() ([]float32, error) {
	if buffers == nil || buffers.Output == nil || buffers.Output.Pointer() == 0 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection output buffer is required", nil)
	}
	if buffers.Rows <= 0 || buffers.Output.Count() != buffers.Rows || buffers.Output.SizeBytes() != uint64(buffers.Rows*4) {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection output byte count mismatch", nil)
	}
	payload := make([]byte, buffers.Output.SizeBytes())
	if err := buffers.Output.driver.CopyDeviceToHost(buffers.Output.Pointer(), payload); err != nil {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "copy MLX q4 projection output", err)
	}
	values, err := hipFloat32PayloadValues(payload)
	if err != nil {
		return nil, err
	}
	if !rocmFloat32SliceFinite(values) {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection output values must be finite", nil)
	}
	return values, nil
}

func hipFloat32PayloadValues(payload []byte) ([]float32, error) {
	if len(payload) == 0 || len(payload)%4 != 0 {
		return nil, core.E("rocm.hip.ProjectionLaunch", "float32 payload byte length must be positive and aligned", nil)
	}
	values := make([]float32, len(payload)/4)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(payload[index*4:]))
	}
	return values, nil
}
