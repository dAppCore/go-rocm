// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"io"
	"iter"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const nativeTensorCopyChunkBytes = 16 << 20

type nativeDevicePointer uintptr

type nativeHIPDriver interface {
	Available() bool
	DeviceInfo() nativeDeviceInfo
	Malloc(size uint64) (nativeDevicePointer, error)
	Free(pointer nativeDevicePointer) error
	CopyHostToDevice(pointer nativeDevicePointer, data []byte) error
}

type hipRuntime struct {
	driver nativeHIPDriver
}

func newSystemNativeRuntime() nativeRuntime {
	return newHIPRuntime(newSystemHIPDriver())
}

func newHIPRuntime(driver nativeHIPDriver) *hipRuntime {
	return &hipRuntime{driver: driver}
}

func (runtime *hipRuntime) Available() bool {
	return runtime != nil && runtime.driver != nil && runtime.driver.Available()
}

func (runtime *hipRuntime) DeviceInfo() nativeDeviceInfo {
	if runtime == nil || runtime.driver == nil {
		return nativeDeviceInfo{}
	}
	return runtime.driver.DeviceInfo()
}

func (runtime *hipRuntime) LoadModel(path string, cfg nativeLoadConfig) (nativeModel, error) {
	if runtime == nil || runtime.driver == nil {
		return nil, core.E("rocm.hip.LoadModel", "HIP driver is nil", nil)
	}
	if !runtime.driver.Available() {
		return nil, core.E("rocm.hip.LoadModel", "HIP driver is not available", nil)
	}
	model := &hipLoadedModel{
		driver:    runtime.driver,
		modelInfo: cfg.ModelInfo,
		tensors:   make(map[string]hipTensor, len(cfg.Tensors)),
		createdAt: time.Now(),
	}
	for _, tensor := range cfg.Tensors {
		if tensor.ByteSize == 0 {
			continue
		}
		pointer, err := runtime.driver.Malloc(tensor.ByteSize)
		if err != nil {
			model.Close()
			return nil, core.E("rocm.hip.LoadModel", "allocate tensor "+tensor.Name, err)
		}
		loaded := hipTensor{info: tensor, pointer: pointer}
		model.tensors[tensor.Name] = loaded
		if err := copyTensorToDevice(runtime.driver, path, cfg.DataOffset, loaded); err != nil {
			model.Close()
			return nil, core.E("rocm.hip.LoadModel", "copy tensor "+tensor.Name, err)
		}
	}
	return model, nil
}

type hipTensor struct {
	info    nativeTensorInfo
	pointer nativeDevicePointer
}

type hipLoadedModel struct {
	driver    nativeHIPDriver
	modelInfo inference.ModelInfo
	tensors   map[string]hipTensor
	adapter   inference.AdapterIdentity
	createdAt time.Time
	closed    bool
}

func (model *hipLoadedModel) Generate(context.Context, string, inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	return emptyTokenSeq, func() error {
		return core.E("rocm.hip.Generate", "native decode kernels are not linked yet", nil)
	}
}

func (model *hipLoadedModel) Chat(context.Context, []inference.Message, inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	return emptyTokenSeq, func() error {
		return core.E("rocm.hip.Chat", "native decode kernels are not linked yet", nil)
	}
}

func (model *hipLoadedModel) Classify(context.Context, []string, inference.GenerateConfig) ([]inference.ClassifyResult, error) {
	return nil, core.E("rocm.hip.Classify", "native prefill kernels are not linked yet", nil)
}

func (model *hipLoadedModel) BatchGenerate(context.Context, []string, inference.GenerateConfig) ([]inference.BatchResult, error) {
	return nil, core.E("rocm.hip.BatchGenerate", "native decode kernels are not linked yet", nil)
}

func (model *hipLoadedModel) Encode(text string) []int32 {
	return approximateTokenIDs(text)
}

func (model *hipLoadedModel) Decode(ids []int32) string {
	if len(ids) == 0 {
		return ""
	}
	return core.Sprintf("%d tokens", len(ids))
}

func (model *hipLoadedModel) ApplyChatTemplate(messages []inference.Message) (string, error) {
	return formatFallbackChatTemplate(messages), nil
}

func (model *hipLoadedModel) LoadAdapter(path string) (inference.AdapterIdentity, error) {
	return inference.AdapterIdentity{}, core.E("rocm.hip.LoadAdapter", "native LoRA adapter application is not linked yet: "+path, nil)
}

func (model *hipLoadedModel) UnloadAdapter() error {
	model.adapter = inference.AdapterIdentity{}
	return nil
}

func (model *hipLoadedModel) ActiveAdapter() inference.AdapterIdentity {
	if model == nil {
		return inference.AdapterIdentity{}
	}
	return model.adapter
}

func (model *hipLoadedModel) Metrics() inference.GenerateMetrics {
	if model == nil {
		return inference.GenerateMetrics{}
	}
	metrics := inference.GenerateMetrics{ActiveMemoryBytes: model.deviceBytes()}
	metrics.PeakMemoryBytes = metrics.ActiveMemoryBytes
	return metrics
}

func (model *hipLoadedModel) Close() error {
	if model == nil || model.closed {
		return nil
	}
	var lastErr error
	for name, tensor := range model.tensors {
		if err := model.driver.Free(tensor.pointer); err != nil {
			lastErr = core.E("rocm.hip.Close", "free tensor "+name, err)
		}
		delete(model.tensors, name)
	}
	model.closed = true
	return lastErr
}

func (model *hipLoadedModel) deviceBytes() uint64 {
	var total uint64
	for _, tensor := range model.tensors {
		total += tensor.info.ByteSize
	}
	return total
}

func copyTensorToDevice(driver nativeHIPDriver, path string, dataOffset int64, tensor hipTensor) error {
	fileResult := core.Open(path)
	if !fileResult.OK {
		return fileResult.Value.(error)
	}
	file := fileResult.Value.(*core.OSFile)
	defer file.Close()

	start := dataOffset + int64(tensor.info.Offset)
	if _, err := file.Seek(start, io.SeekStart); err != nil {
		return err
	}

	remaining := tensor.info.ByteSize
	buffer := make([]byte, min(uint64(nativeTensorCopyChunkBytes), remaining))
	var copied uint64
	for remaining > 0 {
		chunk := int(min(uint64(len(buffer)), remaining))
		if _, err := io.ReadFull(file, buffer[:chunk]); err != nil {
			return err
		}
		if err := driver.CopyHostToDevice(tensor.pointer+nativeDevicePointer(copied), buffer[:chunk]); err != nil {
			return err
		}
		copied += uint64(chunk)
		remaining -= uint64(chunk)
	}
	return nil
}
