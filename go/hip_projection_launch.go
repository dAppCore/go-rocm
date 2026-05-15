// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"encoding/binary"
	"math"

	core "dappco.re/go"
)

const (
	hipProjectionLaunchArgsVersion      uint32 = 1
	hipProjectionLaunchArgsBytes               = 96
	hipMLXQ4ProjectionLaunchArgsVersion uint32 = 1
	hipMLXQ4ProjectionLaunchArgsBytes          = 96
	hipMLXQ4ProjectionBits                     = 4
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
	label     string
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
	InputPointer  nativeDevicePointer
	WeightPointer nativeDevicePointer
	ScalePointer  nativeDevicePointer
	BiasPointer   nativeDevicePointer
	OutputPointer nativeDevicePointer
	Rows          int
	Cols          int
	GroupSize     int
	Bits          int
	InputBytes    uint64
	WeightBytes   uint64
	ScaleBytes    uint64
	BiasBytes     uint64
	OutputBytes   uint64
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
	return validateHIPMLXQ4ProjectionShape(len(input), int(cfg.WeightBytes/4), int(cfg.ScaleBytes/2), int(cfg.BiasBytes/2), cfg.Rows, cfg.Cols, cfg.GroupSize)
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
	payload := make([]byte, hipProjectionLaunchArgsBytes)
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

func (args hipMLXQ4ProjectionLaunchArgs) Binary() ([]byte, error) {
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
	if args.OutputBytes != uint64(rows)*4 {
		return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", "output byte count mismatch", nil)
	}
	for label, value := range map[string]uint64{
		"input bytes":  args.InputBytes,
		"weight bytes": args.WeightBytes,
		"scale bytes":  args.ScaleBytes,
		"bias bytes":   args.BiasBytes,
		"output bytes": args.OutputBytes,
	} {
		if value > uint64(^uint32(0)) {
			return nil, core.E("rocm.hip.MLXQ4ProjectionLaunch", label+" are out of uint32 range", nil)
		}
	}
	payload := make([]byte, hipMLXQ4ProjectionLaunchArgsBytes)
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
	return payload, nil
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
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameMLXQ4Proj, launchBytes, req.Rows)
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
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4ProjectionLaunch", "MLX q4 projection output", uint64(cfg.Rows*4), cfg.Rows)
	if err != nil {
		_ = inputBuffer.Close()
		return nil, err
	}
	buffers := &hipMLXQ4ProjectionDeviceBuffers{Input: inputBuffer, Output: output, Rows: cfg.Rows, Cols: cfg.Cols, GroupSize: cfg.GroupSize}
	success := false
	defer func() {
		if !success {
			_ = buffers.Close()
		}
	}()
	launchBytes, err := (hipMLXQ4ProjectionLaunchArgs{
		InputPointer:  inputBuffer.Pointer(),
		WeightPointer: cfg.WeightPointer,
		ScalePointer:  cfg.ScalePointer,
		BiasPointer:   cfg.BiasPointer,
		OutputPointer: output.Pointer(),
		Rows:          cfg.Rows,
		Cols:          cfg.Cols,
		GroupSize:     cfg.GroupSize,
		Bits:          hipMLXQ4ProjectionBits,
		InputBytes:    inputBuffer.SizeBytes(),
		WeightBytes:   cfg.WeightBytes,
		ScaleBytes:    cfg.ScaleBytes,
		BiasBytes:     cfg.BiasBytes,
		OutputBytes:   output.SizeBytes(),
	}).Binary()
	if err != nil {
		return nil, err
	}
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameMLXQ4Proj, launchBytes, cfg.Rows)
	if err != nil {
		return nil, err
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		return nil, err
	}
	values, err := buffers.ReadOutput()
	if err != nil {
		return nil, err
	}
	success = true
	if err := buffers.Close(); err != nil {
		return nil, err
	}
	return values, nil
}

func hipUploadByteBuffer(driver nativeHIPDriver, operation, label string, payload []byte, count int) (*hipDeviceByteBuffer, error) {
	if len(payload) == 0 {
		return nil, core.E(operation, label+" payload is empty", nil)
	}
	buffer, err := hipAllocateByteBuffer(driver, operation, label, uint64(len(payload)), count)
	if err != nil {
		return nil, err
	}
	if err := driver.CopyHostToDevice(buffer.pointer, payload); err != nil {
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
	pointer, err := driver.Malloc(sizeBytes)
	if err != nil {
		return nil, core.E(operation, "allocate "+label, err)
	}
	return &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   pointer,
		count:     count,
		sizeBytes: sizeBytes,
		label:     label,
	}, nil
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
	if buffer.pointer != 0 {
		if buffer.driver == nil {
			return core.E("rocm.hip.ProjectionLaunch", "HIP driver is nil", nil)
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
