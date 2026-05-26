// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"encoding/binary"
	"errors"
	"iter"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestHIPKernels_StatusLabels_Good(t *testing.T) {
	status := defaultHIPKernelStatus()
	labels := status.Labels()

	core.AssertEqual(t, hipKernelStatusNotLinked, status.Overall())
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["kernel_status"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["cross_entropy_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["decode_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["distillation_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["embedding_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["grpo_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["lora_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["prefill_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["projection_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["rerank_kernel"])
	core.AssertEqual(t, hipKernelStatusPlanned, labels["kv_cache_kernel"])
	core.AssertContains(t, labels["kernel_detail"], "not linked")
}

func TestHIPKernels_NotLinkedErrors_Bad(t *testing.T) {
	kernels := newDefaultHIPKernelSet()
	model := &hipLoadedModel{kernels: kernels}

	stream, streamErr := kernels.Generate(context.Background(), model, "hello", inference.DefaultGenerateConfig())
	for range stream {
	}
	core.AssertError(t, streamErr())
	core.AssertContains(t, streamErr().Error(), "native decode kernels are not linked yet")

	chat, chatErr := kernels.Chat(context.Background(), model, []inference.Message{{Role: "user", Content: "hello"}}, inference.DefaultGenerateConfig())
	for range chat {
	}
	core.AssertError(t, chatErr())
	core.AssertContains(t, chatErr().Error(), "native decode kernels are not linked yet")

	_, classifyErr := kernels.Classify(context.Background(), model, []string{"hello"}, inference.DefaultGenerateConfig())
	core.AssertError(t, classifyErr)
	core.AssertContains(t, classifyErr.Error(), "native prefill kernels are not linked yet")

	_, batchErr := kernels.BatchGenerate(context.Background(), model, []string{"hello"}, inference.DefaultGenerateConfig())
	core.AssertError(t, batchErr)
	core.AssertContains(t, batchErr.Error(), "native decode kernels are not linked yet")

	_, projectErr := kernels.Project(context.Background(), model, hipProjectionRequest{
		Input: []float32{1},
		FP16:  []uint16{0x3c00},
		Rows:  1,
		Cols:  1,
	})
	core.AssertError(t, projectErr)
	core.AssertContains(t, projectErr.Error(), "native projection kernels are not linked yet")

	_, prefillErr := kernels.Prefill(context.Background(), model, hipPrefillRequest{TokenIDs: []int32{1, 2}})
	core.AssertError(t, prefillErr)
	core.AssertContains(t, prefillErr.Error(), "native prefill kernels are not linked yet")

	_, decodeErr := kernels.Decode(context.Background(), model, hipDecodeRequest{TokenID: 2})
	core.AssertError(t, decodeErr)
	core.AssertContains(t, decodeErr.Error(), "native decode kernels are not linked yet")
}

func TestHIPKernels_NotLinkedChatPreflightsMessages_Bad(t *testing.T) {
	kernels := newDefaultHIPKernelSet()

	chat, chatErr := kernels.Chat(context.Background(), &hipLoadedModel{}, nil, inference.DefaultGenerateConfig())
	for range chat {
		t.Fatal("Chat(nil) yielded token, want empty stream")
	}
	core.AssertError(t, chatErr())
	core.AssertContains(t, chatErr().Error(), "messages are required")

	chat, chatErr = kernels.Chat(context.Background(), &hipLoadedModel{}, []inference.Message{{Role: "moderator", Content: "hello"}}, inference.DefaultGenerateConfig())
	for range chat {
		t.Fatal("Chat(invalid role) yielded token, want empty stream")
	}
	core.AssertError(t, chatErr())
	core.AssertContains(t, chatErr().Error(), "message 0 role")

	chat, chatErr = kernels.Chat(context.Background(), &hipLoadedModel{}, []inference.Message{{Role: "user", Content: " "}}, inference.DefaultGenerateConfig())
	for range chat {
		t.Fatal("Chat(empty content) yielded token, want empty stream")
	}
	core.AssertError(t, chatErr())
	core.AssertContains(t, chatErr().Error(), "at least one message must contain content")

	chat, chatErr = kernels.Chat(context.Background(), &hipLoadedModel{}, []inference.Message{{Role: "user", Content: "hello"}}, inference.DefaultGenerateConfig())
	for range chat {
	}
	core.AssertError(t, chatErr())
	core.AssertContains(t, chatErr().Error(), "native decode kernels are not linked yet")
}

func TestHIPKernels_NotLinkedDecodePreflightsDeviceKV_Bad(t *testing.T) {
	kernels := newDefaultHIPKernelSet()
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	device, err := cache.MirrorToDevice(&fakeHIPDriver{available: true})
	core.RequireNoError(t, err)
	defer device.Close()

	_, err = kernels.Decode(context.Background(), &hipLoadedModel{}, hipDecodeRequest{TokenID: 3, KV: cache, DeviceKV: device})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "descriptor table")

	table, err := device.KernelDescriptorTable()
	core.RequireNoError(t, err)
	defer table.Close()
	_, err = kernels.Decode(context.Background(), &hipLoadedModel{}, hipDecodeRequest{TokenID: 3, KV: cache, DeviceKV: device, DescriptorTable: table})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native decode kernels are not linked yet")

	core.RequireNoError(t, table.Close())
	_, err = kernels.Decode(context.Background(), &hipLoadedModel{}, hipDecodeRequest{TokenID: 3, KV: cache, DeviceKV: device, DescriptorTable: table})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "descriptor table")
}

func TestHIPKernels_NotLinkedPrefillPreflightsRequest_Bad(t *testing.T) {
	kernels := newDefaultHIPKernelSet()

	_, err := kernels.Prefill(context.Background(), &hipLoadedModel{}, hipPrefillRequest{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompt or token IDs are required")

	_, err = kernels.Prefill(context.Background(), &hipLoadedModel{}, hipPrefillRequest{TokenIDs: []int32{-1}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token IDs")

	_, err = kernels.Prefill(context.Background(), &hipLoadedModel{}, hipPrefillRequest{TokenIDs: []int32{1}, CacheMode: "not-a-cache-mode"})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported cache mode")

	_, err = kernels.Prefill(context.Background(), &hipLoadedModel{}, hipPrefillRequest{TokenIDs: []int32{1}, KeyWidth: -1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV vector widths")

	_, err = kernels.Prefill(context.Background(), &hipLoadedModel{}, hipPrefillRequest{TokenIDs: []int32{1}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native prefill kernels are not linked yet")
}

func TestHIPKernels_NotLinkedProjectPreflightsRequest_Bad(t *testing.T) {
	kernels := newDefaultHIPKernelSet()

	_, err := kernels.Project(context.Background(), &hipLoadedModel{}, hipProjectionRequest{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "projection weights are required")

	_, err = kernels.Project(context.Background(), &hipLoadedModel{}, hipProjectionRequest{
		Input: []float32{1},
		FP16:  []uint16{0x3c00},
		Q8:    []int8{1},
		Rows:  1,
		Cols:  1,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "only one projection weight encoding")

	_, err = kernels.Project(context.Background(), &hipLoadedModel{}, hipProjectionRequest{
		Input:   []float32{1},
		Q8:      []int8{1},
		Q8Scale: 0,
		Rows:    1,
		Cols:    1,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "q8 scale")

	_, err = kernels.Project(context.Background(), &hipLoadedModel{}, hipProjectionRequest{
		Input: []float32{1},
		FP16:  []uint16{0x3c00},
		Rows:  1,
		Cols:  1,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native projection kernels are not linked yet")
}

func TestHIPKernels_NotLinkedClassifyPreflightsPrompts_Bad(t *testing.T) {
	kernels := newDefaultHIPKernelSet()

	_, err := kernels.Classify(context.Background(), &hipLoadedModel{}, nil, inference.DefaultGenerateConfig())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompts are required")

	_, err = kernels.Classify(context.Background(), &hipLoadedModel{}, []string{"ok", "   "}, inference.DefaultGenerateConfig())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompt 1 is empty")

	_, err = kernels.Classify(context.Background(), &hipLoadedModel{}, []string{"ok"}, inference.DefaultGenerateConfig())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native prefill kernels are not linked yet")
}

func TestHIPKernels_NotLinkedBatchGeneratePreflightsPrompts_Bad(t *testing.T) {
	kernels := newDefaultHIPKernelSet()

	_, err := kernels.BatchGenerate(context.Background(), &hipLoadedModel{}, nil, inference.DefaultGenerateConfig())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompts are required")

	_, err = kernels.BatchGenerate(context.Background(), &hipLoadedModel{}, []string{"ok", ""}, inference.DefaultGenerateConfig())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompt 1 is empty")

	_, err = kernels.BatchGenerate(context.Background(), &hipLoadedModel{}, []string{"ok"}, inference.DefaultGenerateConfig())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native decode kernels are not linked yet")
}

func TestHIPKernels_CancelledContext_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	kernels := newDefaultHIPKernelSet()

	stream, streamErr := kernels.Generate(ctx, &hipLoadedModel{}, "hello", inference.DefaultGenerateConfig())
	for range stream {
	}

	if !errors.Is(streamErr(), context.Canceled) {
		t.Fatalf("stream error = %v, want context.Canceled", streamErr())
	}
	_, err := kernels.Classify(ctx, &hipLoadedModel{}, []string{"hello"}, inference.DefaultGenerateConfig())
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("classify error = %v, want context.Canceled", err)
	}
	_, err = kernels.Project(ctx, &hipLoadedModel{}, hipProjectionRequest{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("project error = %v, want context.Canceled", err)
	}
	_, err = kernels.Prefill(ctx, &hipLoadedModel{}, hipPrefillRequest{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("prefill error = %v, want context.Canceled", err)
	}
	_, err = kernels.Decode(ctx, &hipLoadedModel{}, hipDecodeRequest{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("decode error = %v, want context.Canceled", err)
	}
}

func TestHIPKernels_DeviceTokenBuffer_Good(t *testing.T) {
	payload, err := hipTokenIDsPayload([]int32{7, 513})
	core.AssertNoError(t, err)
	core.AssertEqual(t, 8, len(payload))
	core.AssertEqual(t, uint32(7), binary.LittleEndian.Uint32(payload[0:]))
	core.AssertEqual(t, uint32(513), binary.LittleEndian.Uint32(payload[4:]))

	driver := &fakeHIPDriver{available: true}
	buffer, err := hipUploadTokenIDs(driver, []int32{7, 513})
	core.AssertNoError(t, err)
	core.AssertNotNil(t, buffer)
	core.AssertEqual(t, 2, buffer.Count())
	core.AssertEqual(t, uint64(8), buffer.SizeBytes())
	core.AssertEqual(t, []uint64{8}, driver.allocations)
	core.AssertEqual(t, []uint64{8}, driver.copies)
	launch, err := (hipPrefillRequest{
		TokenIDs:   []int32{7, 513},
		CacheMode:  rocmKVCacheModeQ8,
		KeyWidth:   2,
		ValueWidth: 3,
	}).prefillLaunchArgs(buffer)
	core.AssertNoError(t, err)
	launchBytes, err := launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipPrefillLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipPrefillLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipPrefillLaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(buffer.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(2), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint64(8), binary.LittleEndian.Uint64(launchBytes[24:]))
	core.AssertEqual(t, rocmDeviceKVDescriptorModeQ8, binary.LittleEndian.Uint32(launchBytes[32:]))
	core.AssertEqual(t, uint32(defaultROCmKVBlockSize), binary.LittleEndian.Uint32(launchBytes[36:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[40:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(launchBytes[44:]))
	statusLaunch := launch
	statusLaunch.StatusPointer = 1234
	statusLaunchBytes, err := statusLaunch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, uint64(1234), binary.LittleEndian.Uint64(statusLaunchBytes[48:]))
	core.AssertEqual(t, hipPrefillLaunchStatusOK, binary.LittleEndian.Uint32(statusLaunchBytes[56:]))
	core.AssertNoError(t, buffer.Close())
	core.AssertNoError(t, buffer.Close())
	core.AssertEqual(t, 1, len(driver.frees))
	core.AssertEqual(t, nativeDevicePointer(0), buffer.Pointer())
}

func TestHIPKernels_DeviceTokenBuffer_Bad(t *testing.T) {
	_, err := hipUploadTokenIDs(nil, []int32{1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "HIP driver is nil")

	driver := &fakeHIPDriver{available: false}
	_, err = hipUploadTokenIDs(driver, []int32{1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "HIP driver is not available")
	core.AssertEqual(t, 0, len(driver.allocations))

	driver = &fakeHIPDriver{available: true}
	_, err = hipUploadTokenIDs(driver, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token IDs are required")
	core.AssertEqual(t, 0, len(driver.allocations))

	_, err = hipTokenIDsPayload([]int32{1, -1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token IDs")

	driver = &fakeHIPDriver{available: true, copyErr: core.NewError("copy failed")}
	_, err = hipUploadTokenIDs(driver, []int32{9})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy token buffer")
	core.AssertEqual(t, []uint64{4}, driver.allocations)
	core.AssertEqual(t, []uint64{4}, driver.copies)
	core.AssertEqual(t, 1, len(driver.frees))

	_, err = (hipPrefillRequest{TokenIDs: []int32{1}}).prefillLaunchArgs(nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token buffer")

	driver = &fakeHIPDriver{available: true}
	buffer, err := hipUploadTokenIDs(driver, []int32{1})
	core.AssertNoError(t, err)
	defer buffer.Close()
	_, err = (hipPrefillRequest{TokenIDs: []int32{1, 2}}).prefillLaunchArgs(buffer)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token buffer count")

	badLaunch := hipPrefillLaunchArgs{
		TokenPointer: 1,
		TokenCount:   1,
		TokenBytes:   8,
		CacheMode:    rocmKVCacheModeQ8,
		ModeCode:     rocmDeviceKVDescriptorModeQ8,
		BlockSize:    defaultROCmKVBlockSize,
		KeyWidth:     1,
		ValueWidth:   1,
	}
	_, err = badLaunch.Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token byte count")

	badLaunch.TokenBytes = 4
	badLaunch.ModeCode = rocmDeviceKVDescriptorModeFP16
	_, err = badLaunch.Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "mode code")
}

func TestHIPKernels_ProjectionLaunchArgs_Good(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	driver := &fakeHIPDriver{available: true}
	req := hipProjectionRequest{
		Input: []float32{1, 2},
		FP16:  []uint16{0x3c00, 0x4000},
		Bias:  []float32{0.5},
		Rows:  1,
		Cols:  2,
	}
	buffers, err := req.projectionDeviceBuffers(driver)
	core.AssertNoError(t, err)
	core.AssertNotNil(t, buffers)
	launch, err := req.projectionLaunchArgs(buffers)
	core.AssertNoError(t, err)
	launchBytes, err := launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipProjectionLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipProjectionLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipProjectionLaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(buffers.Input.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[16:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[20:]))
	core.AssertEqual(t, uint64(buffers.Weights.Pointer()), binary.LittleEndian.Uint64(launchBytes[24:]))
	core.AssertEqual(t, uint64(4), binary.LittleEndian.Uint64(launchBytes[32:]))
	core.AssertEqual(t, uint64(buffers.Bias.Pointer()), binary.LittleEndian.Uint64(launchBytes[40:]))
	core.AssertEqual(t, uint64(4), binary.LittleEndian.Uint64(launchBytes[48:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(launchBytes[56:]))
	core.AssertEqual(t, uint64(4), binary.LittleEndian.Uint64(launchBytes[64:]))
	core.AssertEqual(t, uint32(1), binary.LittleEndian.Uint32(launchBytes[72:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[76:]))
	core.AssertEqual(t, hipProjectionWeightEncodingFP16, binary.LittleEndian.Uint32(launchBytes[80:]))
	core.AssertEqual(t, hipProjectionLaunchFlagBias, binary.LittleEndian.Uint32(launchBytes[84:]))
	core.AssertNoError(t, buffers.Close())
	core.AssertNoError(t, buffers.Close())
	core.AssertEqual(t, []uint64{8, 4, 4, 4}, driver.allocations)
	core.AssertEqual(t, []uint64{8, 4, 4}, driver.copies)
	core.AssertEqual(t, 4, len(driver.frees))

	q8Req := hipProjectionRequest{
		Input:   []float32{3},
		Q8:      []int8{2},
		Q8Scale: 0.25,
		Rows:    1,
		Cols:    1,
	}
	q8Buffers, err := q8Req.projectionDeviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer q8Buffers.Close()
	q8Launch, err := q8Req.projectionLaunchArgs(q8Buffers)
	core.AssertNoError(t, err)
	q8LaunchBytes, err := q8Launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipProjectionWeightEncodingQ8, binary.LittleEndian.Uint32(q8LaunchBytes[80:]))
	core.AssertEqual(t, uint32(0), binary.LittleEndian.Uint32(q8LaunchBytes[84:]))
	core.AssertEqual(t, math.Float32bits(0.25), binary.LittleEndian.Uint32(q8LaunchBytes[88:]))

	bf16Req := hipProjectionRequest{
		Input: []float32{1, 2},
		BF16:  []uint16{0x3f80, 0x4000},
		Rows:  1,
		Cols:  2,
	}
	bf16Buffers, err := bf16Req.projectionDeviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer bf16Buffers.Close()
	bf16Launch, err := bf16Req.projectionLaunchArgs(bf16Buffers)
	core.AssertNoError(t, err)
	bf16LaunchBytes, err := bf16Launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipProjectionWeightEncodingBF16, binary.LittleEndian.Uint32(bf16LaunchBytes[80:]))
	core.AssertEqual(t, uint64(4), binary.LittleEndian.Uint64(bf16LaunchBytes[32:]))

	f32Req := hipProjectionRequest{
		Input: []float32{1, 2},
		F32:   []float32{1, 0.5},
		Rows:  1,
		Cols:  2,
	}
	f32Buffers, err := f32Req.projectionDeviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer f32Buffers.Close()
	f32Launch, err := f32Req.projectionLaunchArgs(f32Buffers)
	core.AssertNoError(t, err)
	f32LaunchBytes, err := f32Launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipProjectionWeightEncodingF32, binary.LittleEndian.Uint32(f32LaunchBytes[80:]))
	core.AssertEqual(t, uint64(8), binary.LittleEndian.Uint64(f32LaunchBytes[32:]))
}

func TestHIPKernels_ProjectionLaunchArgs_Bad(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	req := hipProjectionRequest{Input: []float32{1}, FP16: []uint16{0x3c00}, Rows: 1, Cols: 1}
	_, err := req.projectionDeviceBuffers(nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "HIP driver is nil")

	driver := &fakeHIPDriver{available: true, copyErr: core.NewError("copy failed"), copyErrAt: 2}
	_, err = req.projectionDeviceBuffers(driver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy projection fp16 weights")
	core.AssertEqual(t, []uint64{4, 2}, driver.allocations)
	core.AssertEqual(t, []uint64{4, 2}, driver.copies)
	core.AssertEqual(t, 2, len(driver.frees))

	buffers, err := req.projectionDeviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer buffers.Close()
	_, err = (hipProjectionRequest{Input: []float32{1, 2}, FP16: []uint16{0x3c00, 0x4000}, Rows: 1, Cols: 2}).projectionLaunchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	badLaunch := hipProjectionLaunchArgs{
		InputPointer:   1,
		InputCount:     1,
		InputBytes:     4,
		WeightPointer:  2,
		WeightBytes:    2,
		OutputPointer:  3,
		OutputBytes:    8,
		Rows:           1,
		Cols:           1,
		WeightEncoding: hipProjectionWeightEncodingFP16,
	}
	_, err = badLaunch.Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "output byte count")

	badLaunch.OutputBytes = 4
	badLaunch.WeightEncoding = 99
	_, err = badLaunch.Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported projection weight encoding")

	_, err = (hipProjectionRequest{Input: []float32{1}, Q8: []int8{1}, Q8Scale: float32(math.NaN()), Rows: 1, Cols: 1}).projectionDeviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "q8 scale must be positive and finite")

	_, err = (hipProjectionLaunchArgs{
		InputPointer:   1,
		InputCount:     1,
		InputBytes:     4,
		WeightPointer:  2,
		WeightBytes:    1,
		OutputPointer:  3,
		OutputBytes:    4,
		Rows:           1,
		Cols:           1,
		WeightEncoding: hipProjectionWeightEncodingQ8,
		Q8Scale:        float32(math.Inf(1)),
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "q8 scale must be positive and finite")
}

func TestHIPKernels_MLXQ4ProjectionLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipMLXQ4ProjectionRequest{
		Input:     []float32{1, 1, 1, 1, 1, 1, 1, 1},
		Weight:    []uint32{0x76543210, 0xfedcba98},
		Scales:    []uint16{0x3f80, 0x3f00},
		Biases:    []uint16{0x0000, 0xbf80},
		Rows:      2,
		Cols:      8,
		GroupSize: 8,
	}
	buffers, err := req.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer buffers.Close()
	launch, err := req.launchArgs(buffers)
	core.AssertNoError(t, err)
	launchBytes, err := launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipMLXQ4ProjectionLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipMLXQ4ProjectionLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipMLXQ4ProjectionLaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(buffers.Input.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(buffers.Weight.Pointer()), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint64(buffers.Scales.Pointer()), binary.LittleEndian.Uint64(launchBytes[24:]))
	core.AssertEqual(t, uint64(buffers.Biases.Pointer()), binary.LittleEndian.Uint64(launchBytes[32:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(launchBytes[40:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[48:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[52:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[56:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[60:]))
	core.AssertEqual(t, uint32(32), binary.LittleEndian.Uint32(launchBytes[64:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[68:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[72:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[76:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[80:]))
	config, err := hipOneDimensionalLaunchConfig(hipKernelNameMLXQ4Proj, launchBytes, req.Rows)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	output, err := buffers.ReadOutput()
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{28, 38}, output, 0.0001)

	runnerOutput, err := hipRunMLXQ4ProjectionKernelWithDeviceWeightConfig(context.Background(), driver, req.Input, hipMLXQ4DeviceWeightConfig{
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		WeightBytes:   buffers.Weight.SizeBytes(),
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		Rows:          req.Rows,
		Cols:          req.Cols,
		GroupSize:     req.GroupSize,
	})
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{28, 38}, runnerOutput, 0.0001)

	batchInputPayload, err := hipFloat32Payload([]float32{
		1, 1, 1, 1, 1, 1, 1, 1,
		2, 2, 2, 2, 2, 2, 2, 2,
	})
	core.AssertNoError(t, err)
	batchInput, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch input", batchInputPayload, req.Cols*2)
	core.AssertNoError(t, err)
	defer batchInput.Close()
	batchOutput, err := hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(context.Background(), driver, batchInput, hipMLXQ4DeviceWeightConfig{
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		WeightBytes:   buffers.Weight.SizeBytes(),
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		Rows:          req.Rows,
		Cols:          req.Cols,
		GroupSize:     req.GroupSize,
	}, 2)
	core.AssertNoError(t, err)
	defer batchOutput.Close()
	batchValues, err := hipReadFloat32DeviceOutput(batchOutput, "rocm.hip.MLXQ4ProjectionBatchLaunch", "MLX q4 projection batch output", req.Rows*2)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{28, 38, 56, 76}, batchValues, 0.0001)
	batchLaunch := driver.launches[len(driver.launches)-1]
	core.AssertEqual(t, hipKernelNameMLXQ4ProjBatch, batchLaunch.Name)
	core.AssertEqual(t, uint32(1), batchLaunch.GridX)
	core.AssertEqual(t, uint32(1), batchLaunch.GridY)
	core.AssertEqual(t, hipMLXQ4ProjectionBatchLaunchArgsBytes, len(batchLaunch.Args))
	core.AssertEqual(t, hipMLXQ4ProjectionBatchLaunchArgsVersion, binary.LittleEndian.Uint32(batchLaunch.Args[0:]))
	core.AssertEqual(t, uint32(req.Rows), binary.LittleEndian.Uint32(batchLaunch.Args[48:]))
	core.AssertEqual(t, uint32(req.Cols), binary.LittleEndian.Uint32(batchLaunch.Args[52:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(batchLaunch.Args[56:]))
	core.AssertEqual(t, uint32(req.GroupSize), binary.LittleEndian.Uint32(batchLaunch.Args[60:]))
	core.AssertEqual(t, uint32(req.Cols*2*4), binary.LittleEndian.Uint32(batchLaunch.Args[68:]))
	core.AssertEqual(t, uint32(req.Rows*2*4), binary.LittleEndian.Uint32(batchLaunch.Args[84:]))

	greedy, err := hipRunMLXQ4ProjectionSoftcapGreedyKernelWithDeviceInput(context.Background(), driver, buffers.Input, hipMLXQ4DeviceWeightConfig{
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		WeightBytes:   buffers.Weight.SizeBytes(),
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		Rows:          req.Rows,
		Cols:          req.Cols,
		GroupSize:     req.GroupSize,
	}, 0)
	core.AssertNoError(t, err)
	core.AssertEqual(t, 1, greedy.TokenID)
	assertFloat32Near(t, 38, greedy.Score)
	core.AssertEqual(t, []uint64{hipMLXQ4ProjectionBestBytes}, driver.memsets)
	core.AssertEqual(t, hipKernelNameMLXQ4ProjGreedy, driver.launches[len(driver.launches)-1].Name)
}

func TestHIPKernels_MLXQ4TripleProjectionLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	firstReq := hipMLXQ4ProjectionRequest{
		Input:     []float32{1, 1, 1, 1, 1, 1, 1, 1},
		Weight:    []uint32{0x76543210, 0xfedcba98},
		Scales:    []uint16{0x3f80, 0x3f00},
		Biases:    []uint16{0x0000, 0xbf80},
		Rows:      2,
		Cols:      8,
		GroupSize: 8,
	}
	secondReq := hipMLXQ4ProjectionRequest{
		Input:     firstReq.Input,
		Weight:    []uint32{0x11111111},
		Scales:    []uint16{0x3f80},
		Biases:    []uint16{0x0000},
		Rows:      1,
		Cols:      8,
		GroupSize: 8,
	}
	thirdReq := hipMLXQ4ProjectionRequest{
		Input:     firstReq.Input,
		Weight:    []uint32{0x22222222},
		Scales:    []uint16{0x3f80},
		Biases:    []uint16{0x0000},
		Rows:      1,
		Cols:      8,
		GroupSize: 8,
	}
	firstBuffers, err := firstReq.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer firstBuffers.Close()
	secondBuffers, err := secondReq.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer secondBuffers.Close()
	thirdBuffers, err := thirdReq.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer thirdBuffers.Close()
	launchBytes, err := (hipMLXQ4TripleProjLaunchArgs{
		InputPointer:        firstBuffers.Input.Pointer(),
		OutputPointer:       nativeDevicePointer(99),
		FirstWeightPointer:  firstBuffers.Weight.Pointer(),
		FirstScalePointer:   firstBuffers.Scales.Pointer(),
		FirstBiasPointer:    firstBuffers.Biases.Pointer(),
		SecondWeightPointer: secondBuffers.Weight.Pointer(),
		SecondScalePointer:  secondBuffers.Scales.Pointer(),
		SecondBiasPointer:   secondBuffers.Biases.Pointer(),
		ThirdWeightPointer:  thirdBuffers.Weight.Pointer(),
		ThirdScalePointer:   thirdBuffers.Scales.Pointer(),
		ThirdBiasPointer:    thirdBuffers.Biases.Pointer(),
		FirstRows:           firstReq.Rows,
		SecondRows:          secondReq.Rows,
		ThirdRows:           thirdReq.Rows,
		Cols:                firstReq.Cols,
		GroupSize:           firstReq.GroupSize,
		Bits:                hipMLXQ4ProjectionBits,
		InputBytes:          firstBuffers.Input.SizeBytes(),
		OutputBytes:         uint64((firstReq.Rows + secondReq.Rows + thirdReq.Rows) * 4),
		FirstWeightBytes:    firstBuffers.Weight.SizeBytes(),
		FirstScaleBytes:     firstBuffers.Scales.SizeBytes(),
		FirstBiasBytes:      firstBuffers.Biases.SizeBytes(),
		SecondWeightBytes:   secondBuffers.Weight.SizeBytes(),
		SecondScaleBytes:    secondBuffers.Scales.SizeBytes(),
		SecondBiasBytes:     secondBuffers.Biases.SizeBytes(),
		ThirdWeightBytes:    thirdBuffers.Weight.SizeBytes(),
		ThirdScaleBytes:     thirdBuffers.Scales.SizeBytes(),
		ThirdBiasBytes:      thirdBuffers.Biases.SizeBytes(),
	}).Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipMLXQ4TripleProjLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipMLXQ4TripleProjLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipMLXQ4TripleProjLaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(firstBuffers.Input.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(99), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[96:]))
	core.AssertEqual(t, uint32(1), binary.LittleEndian.Uint32(launchBytes[100:]))
	core.AssertEqual(t, uint32(1), binary.LittleEndian.Uint32(launchBytes[104:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[108:]))
	output, first, second, third, err := hipRunMLXQ4TripleProjectionKernelWithDeviceInput(context.Background(), driver, firstBuffers.Input,
		hipMLXQ4DeviceWeightConfig{
			WeightPointer: firstBuffers.Weight.Pointer(),
			ScalePointer:  firstBuffers.Scales.Pointer(),
			BiasPointer:   firstBuffers.Biases.Pointer(),
			WeightBytes:   firstBuffers.Weight.SizeBytes(),
			ScaleBytes:    firstBuffers.Scales.SizeBytes(),
			BiasBytes:     firstBuffers.Biases.SizeBytes(),
			Rows:          firstReq.Rows,
			Cols:          firstReq.Cols,
			GroupSize:     firstReq.GroupSize,
		},
		hipMLXQ4DeviceWeightConfig{
			WeightPointer: secondBuffers.Weight.Pointer(),
			ScalePointer:  secondBuffers.Scales.Pointer(),
			BiasPointer:   secondBuffers.Biases.Pointer(),
			WeightBytes:   secondBuffers.Weight.SizeBytes(),
			ScaleBytes:    secondBuffers.Scales.SizeBytes(),
			BiasBytes:     secondBuffers.Biases.SizeBytes(),
			Rows:          secondReq.Rows,
			Cols:          secondReq.Cols,
			GroupSize:     secondReq.GroupSize,
		},
		hipMLXQ4DeviceWeightConfig{
			WeightPointer: thirdBuffers.Weight.Pointer(),
			ScalePointer:  thirdBuffers.Scales.Pointer(),
			BiasPointer:   thirdBuffers.Biases.Pointer(),
			WeightBytes:   thirdBuffers.Weight.SizeBytes(),
			ScaleBytes:    thirdBuffers.Scales.SizeBytes(),
			BiasBytes:     thirdBuffers.Biases.SizeBytes(),
			Rows:          thirdReq.Rows,
			Cols:          thirdReq.Cols,
			GroupSize:     thirdReq.GroupSize,
		})
	core.AssertNoError(t, err)
	defer output.Close()
	core.AssertEqual(t, hipKernelNameMLXQ4TripleProj, driver.launches[len(driver.launches)-1].Name)
	core.AssertEqual(t, output.Pointer(), first.Pointer())
	core.AssertEqual(t, output.Pointer()+nativeDevicePointer(firstReq.Rows*4), second.Pointer())
	core.AssertEqual(t, output.Pointer()+nativeDevicePointer((firstReq.Rows+secondReq.Rows)*4), third.Pointer())
	firstValues, err := hipReadFloat32DeviceOutput(first, "rocm.hip.MLXQ4TripleProjectionLaunch", "first output", firstReq.Rows)
	core.AssertNoError(t, err)
	secondValues, err := hipReadFloat32DeviceOutput(second, "rocm.hip.MLXQ4TripleProjectionLaunch", "second output", secondReq.Rows)
	core.AssertNoError(t, err)
	thirdValues, err := hipReadFloat32DeviceOutput(third, "rocm.hip.MLXQ4TripleProjectionLaunch", "third output", thirdReq.Rows)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{28, 38}, firstValues, 0.0001)
	assertFloat32SlicesNear(t, []float32{8}, secondValues, 0.0001)
	assertFloat32SlicesNear(t, []float32{16}, thirdValues, 0.0001)

	reusedOutput, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4TripleProjectionLaunch", "reused triple projection output", uint64((firstReq.Rows+secondReq.Rows+thirdReq.Rows)*4), firstReq.Rows+secondReq.Rows+thirdReq.Rows)
	core.AssertNoError(t, err)
	defer reusedOutput.Close()
	reusedFirst, reusedSecond, reusedThird, err := hipRunMLXQ4TripleProjectionKernelWithDeviceInputViewsOutput(context.Background(), driver, firstBuffers.Input,
		hipMLXQ4DeviceWeightConfig{
			WeightPointer: firstBuffers.Weight.Pointer(),
			ScalePointer:  firstBuffers.Scales.Pointer(),
			BiasPointer:   firstBuffers.Biases.Pointer(),
			WeightBytes:   firstBuffers.Weight.SizeBytes(),
			ScaleBytes:    firstBuffers.Scales.SizeBytes(),
			BiasBytes:     firstBuffers.Biases.SizeBytes(),
			Rows:          firstReq.Rows,
			Cols:          firstReq.Cols,
			GroupSize:     firstReq.GroupSize,
		},
		hipMLXQ4DeviceWeightConfig{
			WeightPointer: secondBuffers.Weight.Pointer(),
			ScalePointer:  secondBuffers.Scales.Pointer(),
			BiasPointer:   secondBuffers.Biases.Pointer(),
			WeightBytes:   secondBuffers.Weight.SizeBytes(),
			ScaleBytes:    secondBuffers.Scales.SizeBytes(),
			BiasBytes:     secondBuffers.Biases.SizeBytes(),
			Rows:          secondReq.Rows,
			Cols:          secondReq.Cols,
			GroupSize:     secondReq.GroupSize,
		},
		hipMLXQ4DeviceWeightConfig{
			WeightPointer: thirdBuffers.Weight.Pointer(),
			ScalePointer:  thirdBuffers.Scales.Pointer(),
			BiasPointer:   thirdBuffers.Biases.Pointer(),
			WeightBytes:   thirdBuffers.Weight.SizeBytes(),
			ScaleBytes:    thirdBuffers.Scales.SizeBytes(),
			BiasBytes:     thirdBuffers.Biases.SizeBytes(),
			Rows:          thirdReq.Rows,
			Cols:          thirdReq.Cols,
			GroupSize:     thirdReq.GroupSize,
		}, reusedOutput)
	core.AssertNoError(t, err)
	reusedFirstValues, err := hipReadFloat32DeviceOutput(&reusedFirst, "rocm.hip.MLXQ4TripleProjectionLaunch", "reused first output", firstReq.Rows)
	core.AssertNoError(t, err)
	reusedSecondValues, err := hipReadFloat32DeviceOutput(&reusedSecond, "rocm.hip.MLXQ4TripleProjectionLaunch", "reused second output", secondReq.Rows)
	core.AssertNoError(t, err)
	reusedThirdValues, err := hipReadFloat32DeviceOutput(&reusedThird, "rocm.hip.MLXQ4TripleProjectionLaunch", "reused third output", thirdReq.Rows)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{28, 38}, reusedFirstValues, 0.0001)
	assertFloat32SlicesNear(t, []float32{8}, reusedSecondValues, 0.0001)
	assertFloat32SlicesNear(t, []float32{16}, reusedThirdValues, 0.0001)
}

func TestHIPKernels_MLXQ4GELUTanhMultiplyLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	gateReq := hipMLXQ4ProjectionRequest{
		Input:     []float32{1, 1, 1, 1, 1, 1, 1, 1},
		Weight:    []uint32{0x76543210, 0xfedcba98},
		Scales:    []uint16{0x3f80, 0x3f00},
		Biases:    []uint16{0x0000, 0xbf80},
		Rows:      2,
		Cols:      8,
		GroupSize: 8,
	}
	upReq := hipMLXQ4ProjectionRequest{
		Input:     gateReq.Input,
		Weight:    []uint32{0x11111111, 0x22222222},
		Scales:    []uint16{0x3f80, 0x3f80},
		Biases:    []uint16{0x0000, 0x0000},
		Rows:      gateReq.Rows,
		Cols:      gateReq.Cols,
		GroupSize: gateReq.GroupSize,
	}
	gateBuffers, err := gateReq.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer gateBuffers.Close()
	upBuffers, err := upReq.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer upBuffers.Close()
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "MLX q4 GELU tanh multiply output", uint64(gateReq.Rows*4), gateReq.Rows)
	core.AssertNoError(t, err)
	defer output.Close()

	launchBytes, err := (hipMLXQ4GELUTanhMulLaunchArgs{
		InputPointer:      gateBuffers.Input.Pointer(),
		GateWeightPointer: gateBuffers.Weight.Pointer(),
		GateScalePointer:  gateBuffers.Scales.Pointer(),
		GateBiasPointer:   gateBuffers.Biases.Pointer(),
		UpWeightPointer:   upBuffers.Weight.Pointer(),
		UpScalePointer:    upBuffers.Scales.Pointer(),
		UpBiasPointer:     upBuffers.Biases.Pointer(),
		OutputPointer:     output.Pointer(),
		Rows:              gateReq.Rows,
		Cols:              gateReq.Cols,
		GroupSize:         gateReq.GroupSize,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        gateBuffers.Input.SizeBytes(),
		GateWeightBytes:   gateBuffers.Weight.SizeBytes(),
		GateScaleBytes:    gateBuffers.Scales.SizeBytes(),
		GateBiasBytes:     gateBuffers.Biases.SizeBytes(),
		UpWeightBytes:     upBuffers.Weight.SizeBytes(),
		UpScaleBytes:      upBuffers.Scales.SizeBytes(),
		UpBiasBytes:       upBuffers.Biases.SizeBytes(),
		OutputBytes:       output.SizeBytes(),
	}).Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipMLXQ4GELUTanhMulLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipMLXQ4GELUTanhMulLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipMLXQ4GELUTanhMulLaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(gateBuffers.Input.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(gateBuffers.Weight.Pointer()), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint64(gateBuffers.Scales.Pointer()), binary.LittleEndian.Uint64(launchBytes[24:]))
	core.AssertEqual(t, uint64(gateBuffers.Biases.Pointer()), binary.LittleEndian.Uint64(launchBytes[32:]))
	core.AssertEqual(t, uint64(upBuffers.Weight.Pointer()), binary.LittleEndian.Uint64(launchBytes[40:]))
	core.AssertEqual(t, uint64(upBuffers.Scales.Pointer()), binary.LittleEndian.Uint64(launchBytes[48:]))
	core.AssertEqual(t, uint64(upBuffers.Biases.Pointer()), binary.LittleEndian.Uint64(launchBytes[56:]))
	core.AssertEqual(t, uint64(output.Pointer()), binary.LittleEndian.Uint64(launchBytes[64:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[72:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[76:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[80:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[84:]))
	core.AssertEqual(t, uint32(32), binary.LittleEndian.Uint32(launchBytes[88:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[92:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[96:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[100:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[104:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[108:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[112:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[116:]))

	config, err := hipMLXQ4GELUTanhMultiplyLaunchConfig(launchBytes, gateReq.Rows)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	outputValues, err := hipReadFloat32DeviceOutput(output, "rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "MLX q4 GELU tanh multiply output", gateReq.Rows)
	core.AssertNoError(t, err)
	want := expectedGELUTanhMultiplyFromQ4(t, gateReq, upReq)
	assertFloat32SlicesNear(t, want, outputValues, 0.0001)

	gateCfg := hipMLXQ4DeviceWeightConfig{
		WeightPointer: gateBuffers.Weight.Pointer(),
		ScalePointer:  gateBuffers.Scales.Pointer(),
		BiasPointer:   gateBuffers.Biases.Pointer(),
		WeightBytes:   gateBuffers.Weight.SizeBytes(),
		ScaleBytes:    gateBuffers.Scales.SizeBytes(),
		BiasBytes:     gateBuffers.Biases.SizeBytes(),
		Rows:          gateReq.Rows,
		Cols:          gateReq.Cols,
		GroupSize:     gateReq.GroupSize,
	}
	upCfg := hipMLXQ4DeviceWeightConfig{
		WeightPointer: upBuffers.Weight.Pointer(),
		ScalePointer:  upBuffers.Scales.Pointer(),
		BiasPointer:   upBuffers.Biases.Pointer(),
		WeightBytes:   upBuffers.Weight.SizeBytes(),
		ScaleBytes:    upBuffers.Scales.SizeBytes(),
		BiasBytes:     upBuffers.Biases.SizeBytes(),
		Rows:          upReq.Rows,
		Cols:          upReq.Cols,
		GroupSize:     upReq.GroupSize,
	}
	activated, err := hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInput(context.Background(), driver, gateBuffers.Input, gateCfg, upCfg)
	core.AssertNoError(t, err)
	defer activated.Close()
	activatedValues, err := hipReadFloat32DeviceOutput(activated, "rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "MLX q4 GELU tanh multiply output", gateReq.Rows)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, want, activatedValues, 0.0001)
	core.AssertEqual(t, hipKernelNameMLXQ4GELUTanhMul, driver.launches[len(driver.launches)-1].Name)

	reusedActivated, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "reused MLX q4 GELU tanh multiply output", uint64(gateReq.Rows*4), gateReq.Rows)
	core.AssertNoError(t, err)
	defer reusedActivated.Close()
	core.AssertNoError(t, hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInputOutput(context.Background(), driver, gateBuffers.Input, gateCfg, upCfg, reusedActivated))
	reusedActivatedValues, err := hipReadFloat32DeviceOutput(reusedActivated, "rocm.hip.MLXQ4GELUTanhMultiplyLaunch", "reused MLX q4 GELU tanh multiply output", gateReq.Rows)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, want, reusedActivatedValues, 0.0001)
	core.AssertEqual(t, hipKernelNameMLXQ4GELUTanhMul, driver.launches[len(driver.launches)-1].Name)

	batchInputPayload, err := hipFloat32Payload([]float32{
		1, 1, 1, 1, 1, 1, 1, 1,
		2, 2, 2, 2, 2, 2, 2, 2,
	})
	core.AssertNoError(t, err)
	batchInput, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "MLX q4 GELU tanh multiply batch input", batchInputPayload, gateReq.Cols*2)
	core.AssertNoError(t, err)
	defer batchInput.Close()
	batchActivated, err := hipRunMLXQ4GELUTanhMultiplyBatchKernelWithDeviceInput(context.Background(), driver, batchInput, hipMLXQ4DeviceWeightConfig{
		WeightPointer: gateBuffers.Weight.Pointer(),
		ScalePointer:  gateBuffers.Scales.Pointer(),
		BiasPointer:   gateBuffers.Biases.Pointer(),
		WeightBytes:   gateBuffers.Weight.SizeBytes(),
		ScaleBytes:    gateBuffers.Scales.SizeBytes(),
		BiasBytes:     gateBuffers.Biases.SizeBytes(),
		Rows:          gateReq.Rows,
		Cols:          gateReq.Cols,
		GroupSize:     gateReq.GroupSize,
	}, hipMLXQ4DeviceWeightConfig{
		WeightPointer: upBuffers.Weight.Pointer(),
		ScalePointer:  upBuffers.Scales.Pointer(),
		BiasPointer:   upBuffers.Biases.Pointer(),
		WeightBytes:   upBuffers.Weight.SizeBytes(),
		ScaleBytes:    upBuffers.Scales.SizeBytes(),
		BiasBytes:     upBuffers.Biases.SizeBytes(),
		Rows:          upReq.Rows,
		Cols:          upReq.Cols,
		GroupSize:     upReq.GroupSize,
	}, 2)
	core.AssertNoError(t, err)
	defer batchActivated.Close()
	batchValues, err := hipReadFloat32DeviceOutput(batchActivated, "rocm.hip.MLXQ4GELUTanhMultiplyBatchLaunch", "MLX q4 GELU tanh multiply batch output", gateReq.Rows*2)
	core.AssertNoError(t, err)
	secondGateReq := gateReq
	secondGateReq.Input = []float32{2, 2, 2, 2, 2, 2, 2, 2}
	secondUpReq := upReq
	secondUpReq.Input = secondGateReq.Input
	secondWant := expectedGELUTanhMultiplyFromQ4(t, secondGateReq, secondUpReq)
	assertFloat32SlicesNear(t, append(append([]float32(nil), want...), secondWant...), batchValues, 0.0001)
	batchLaunch := driver.launches[len(driver.launches)-1]
	core.AssertEqual(t, hipKernelNameMLXQ4GELUTanhMulBatch, batchLaunch.Name)
	core.AssertEqual(t, uint32(1), batchLaunch.GridY)
	core.AssertEqual(t, hipMLXQ4GELUTanhMulBatchLaunchArgsBytes, len(batchLaunch.Args))
	core.AssertEqual(t, hipMLXQ4GELUTanhMulBatchLaunchArgsVersion, binary.LittleEndian.Uint32(batchLaunch.Args[0:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(batchLaunch.Args[120:]))
}

func TestHIPKernels_MLXQ4GELUTanhMultiplyLaunchArgs_Bad(t *testing.T) {
	_, err := (hipMLXQ4GELUTanhMulLaunchArgs{
		InputPointer:      1,
		GateWeightPointer: 2,
		GateScalePointer:  3,
		GateBiasPointer:   4,
		UpWeightPointer:   5,
		UpScalePointer:    6,
		UpBiasPointer:     7,
		OutputPointer:     8,
		Rows:              1,
		Cols:              8,
		GroupSize:         8,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        32,
		GateWeightBytes:   8,
		GateScaleBytes:    2,
		GateBiasBytes:     2,
		UpWeightBytes:     4,
		UpScaleBytes:      2,
		UpBiasBytes:       2,
		OutputBytes:       4,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "packed weight byte count")

	_, err = (hipMLXQ4GELUTanhMulBatchLaunchArgs{
		InputPointer:      1,
		GateWeightPointer: 2,
		GateScalePointer:  3,
		GateBiasPointer:   4,
		UpWeightPointer:   5,
		UpScalePointer:    6,
		UpBiasPointer:     7,
		OutputPointer:     8,
		Rows:              1,
		Cols:              8,
		GroupSize:         8,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        32,
		GateWeightBytes:   4,
		GateScaleBytes:    2,
		GateBiasBytes:     2,
		UpWeightBytes:     4,
		UpScaleBytes:      2,
		UpBiasBytes:       2,
		OutputBytes:       8,
		Batch:             2,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input byte count")

	driver := &fakeHIPDriver{available: true}
	req := hipMLXQ4ProjectionRequest{
		Input:     []float32{1, 1, 1, 1, 1, 1, 1, 1},
		Weight:    []uint32{0x76543210},
		Scales:    []uint16{0x3f80},
		Biases:    []uint16{0x0000},
		Rows:      1,
		Cols:      8,
		GroupSize: 8,
	}
	buffers, err := req.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer buffers.Close()
	_, err = hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInput(context.Background(), driver, buffers.Input, hipMLXQ4DeviceWeightConfig{
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		WeightBytes:   buffers.Weight.SizeBytes(),
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		Rows:          1,
		Cols:          8,
		GroupSize:     8,
	}, hipMLXQ4DeviceWeightConfig{
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		WeightBytes:   buffers.Weight.SizeBytes(),
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		Rows:          2,
		Cols:          8,
		GroupSize:     8,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shapes must match")

	_, err = hipRunMLXQ4GELUTanhMultiplyBatchKernelWithDeviceInput(context.Background(), driver, buffers.Input, hipMLXQ4DeviceWeightConfig{
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		WeightBytes:   buffers.Weight.SizeBytes(),
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		Rows:          req.Rows,
		Cols:          req.Cols,
		GroupSize:     req.GroupSize,
	}, hipMLXQ4DeviceWeightConfig{
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		WeightBytes:   buffers.Weight.SizeBytes(),
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		Rows:          req.Rows,
		Cols:          req.Cols,
		GroupSize:     req.GroupSize,
	}, 2)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "batch input count mismatch")

}

func TestHIPKernels_MLXQ4GELUTanhProjectionLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipMLXQ4ProjectionRequest{
		Input:     []float32{1, 1, 1, 1, 1, 1, 1, 1},
		Weight:    []uint32{0x76543210, 0xfedcba98},
		Scales:    []uint16{0x3f80, 0x3f00},
		Biases:    []uint16{0x0000, 0xbf80},
		Rows:      2,
		Cols:      8,
		GroupSize: 8,
	}
	buffers, err := req.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer buffers.Close()
	multiplierPayload, err := hipFloat32Payload([]float32{2, 3})
	core.AssertNoError(t, err)
	multiplier, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection multiplier", multiplierPayload, req.Rows)
	core.AssertNoError(t, err)
	defer multiplier.Close()
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection output", uint64(req.Rows*4), req.Rows)
	core.AssertNoError(t, err)
	defer output.Close()

	launchBytes, err := (hipMLXQ4GELUTanhProjLaunchArgs{
		InputPointer:      buffers.Input.Pointer(),
		WeightPointer:     buffers.Weight.Pointer(),
		ScalePointer:      buffers.Scales.Pointer(),
		BiasPointer:       buffers.Biases.Pointer(),
		MultiplierPointer: multiplier.Pointer(),
		OutputPointer:     output.Pointer(),
		Rows:              req.Rows,
		Cols:              req.Cols,
		GroupSize:         req.GroupSize,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        buffers.Input.SizeBytes(),
		WeightBytes:       buffers.Weight.SizeBytes(),
		ScaleBytes:        buffers.Scales.SizeBytes(),
		BiasBytes:         buffers.Biases.SizeBytes(),
		MultiplierBytes:   multiplier.SizeBytes(),
		OutputBytes:       output.SizeBytes(),
	}).Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipMLXQ4GELUTanhProjLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipMLXQ4GELUTanhProjLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipMLXQ4GELUTanhProjLaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(buffers.Input.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(buffers.Weight.Pointer()), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint64(buffers.Scales.Pointer()), binary.LittleEndian.Uint64(launchBytes[24:]))
	core.AssertEqual(t, uint64(buffers.Biases.Pointer()), binary.LittleEndian.Uint64(launchBytes[32:]))
	core.AssertEqual(t, uint64(multiplier.Pointer()), binary.LittleEndian.Uint64(launchBytes[40:]))
	core.AssertEqual(t, uint64(output.Pointer()), binary.LittleEndian.Uint64(launchBytes[48:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[56:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[60:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[64:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[68:]))
	core.AssertEqual(t, uint32(32), binary.LittleEndian.Uint32(launchBytes[72:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[76:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[80:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[84:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[88:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[92:]))

	config, err := hipMLXQ4GELUTanhProjectionLaunchConfig(launchBytes, req.Rows)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	outputValues, err := hipReadFloat32DeviceOutput(output, "rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection output", req.Rows)
	core.AssertNoError(t, err)
	want := expectedGELUTanhProjectionFromQ4(t, req, []float32{2, 3})
	assertFloat32SlicesNear(t, want, outputValues, 0.0001)

	cfg := hipMLXQ4DeviceWeightConfig{
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		WeightBytes:   buffers.Weight.SizeBytes(),
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		Rows:          req.Rows,
		Cols:          req.Cols,
		GroupSize:     req.GroupSize,
	}
	activated, err := hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplier(context.Background(), driver, buffers.Input, multiplier, cfg)
	core.AssertNoError(t, err)
	defer activated.Close()
	activatedValues, err := hipReadFloat32DeviceOutput(activated, "rocm.hip.MLXQ4GELUTanhProjectionLaunch", "MLX q4 GELU tanh projection output", req.Rows)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, want, activatedValues, 0.0001)
	core.AssertEqual(t, hipKernelNameMLXQ4GELUTanhProj, driver.launches[len(driver.launches)-1].Name)

	reusedActivated, err := hipAllocateByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhProjectionLaunch", "reused MLX q4 GELU tanh projection output", uint64(req.Rows*4), req.Rows)
	core.AssertNoError(t, err)
	defer reusedActivated.Close()
	core.AssertNoError(t, hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplierOutput(context.Background(), driver, buffers.Input, multiplier, cfg, reusedActivated))
	reusedActivatedValues, err := hipReadFloat32DeviceOutput(reusedActivated, "rocm.hip.MLXQ4GELUTanhProjectionLaunch", "reused MLX q4 GELU tanh projection output", req.Rows)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, want, reusedActivatedValues, 0.0001)
	core.AssertEqual(t, hipKernelNameMLXQ4GELUTanhProj, driver.launches[len(driver.launches)-1].Name)

	secondReq := req
	secondReq.Input = []float32{2, 2, 2, 2, 2, 2, 2, 2}
	batchInputValues := append(append([]float32(nil), req.Input...), secondReq.Input...)
	batchInputPayload, err := hipFloat32Payload(batchInputValues)
	core.AssertNoError(t, err)
	batchInput, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch input", batchInputPayload, len(batchInputValues))
	core.AssertNoError(t, err)
	defer batchInput.Close()
	batchMultiplierValues := []float32{2, 3, 4, 5}
	batchMultiplierPayload, err := hipFloat32Payload(batchMultiplierValues)
	core.AssertNoError(t, err)
	batchMultiplier, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch multiplier", batchMultiplierPayload, len(batchMultiplierValues))
	core.AssertNoError(t, err)
	defer batchMultiplier.Close()
	batchActivated, err := hipRunMLXQ4GELUTanhProjectionBatchKernelWithDeviceMultiplier(context.Background(), driver, batchInput, batchMultiplier, hipMLXQ4DeviceWeightConfig{
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		WeightBytes:   buffers.Weight.SizeBytes(),
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		Rows:          req.Rows,
		Cols:          req.Cols,
		GroupSize:     req.GroupSize,
	}, 2)
	core.AssertNoError(t, err)
	defer batchActivated.Close()
	batchValues, err := hipReadFloat32DeviceOutput(batchActivated, "rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch output", req.Rows*2)
	core.AssertNoError(t, err)
	batchWant := append(
		expectedGELUTanhProjectionFromQ4(t, req, []float32{2, 3}),
		expectedGELUTanhProjectionFromQ4(t, secondReq, []float32{4, 5})...,
	)
	assertFloat32SlicesNear(t, batchWant, batchValues, 0.0001)
	batchLaunch := driver.launches[len(driver.launches)-1]
	core.AssertEqual(t, hipKernelNameMLXQ4GELUTanhProjBatch, batchLaunch.Name)
	core.AssertEqual(t, uint32(1), batchLaunch.GridY)
	core.AssertEqual(t, hipMLXQ4GELUTanhProjBatchLaunchArgsBytes, len(batchLaunch.Args))
	core.AssertEqual(t, hipMLXQ4GELUTanhProjBatchLaunchArgsVersion, binary.LittleEndian.Uint32(batchLaunch.Args[0:]))
	core.AssertEqual(t, uint32(hipMLXQ4GELUTanhProjBatchLaunchArgsBytes), binary.LittleEndian.Uint32(batchLaunch.Args[4:]))
	core.AssertEqual(t, uint32(req.Rows), binary.LittleEndian.Uint32(batchLaunch.Args[56:]))
	core.AssertEqual(t, uint32(req.Cols), binary.LittleEndian.Uint32(batchLaunch.Args[60:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(batchLaunch.Args[64:]))
}

func TestHIPKernels_MLXQ4GELUTanhProjectionLaunchArgs_Bad(t *testing.T) {
	_, err := (hipMLXQ4GELUTanhProjLaunchArgs{
		InputPointer:      1,
		WeightPointer:     2,
		ScalePointer:      3,
		BiasPointer:       4,
		MultiplierPointer: 5,
		OutputPointer:     6,
		Rows:              1,
		Cols:              8,
		GroupSize:         8,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        32,
		WeightBytes:       4,
		ScaleBytes:        2,
		BiasBytes:         2,
		MultiplierBytes:   8,
		OutputBytes:       4,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "multiplier/output byte count")

	_, err = (hipMLXQ4GELUTanhProjBatchLaunchArgs{
		InputPointer:      1,
		WeightPointer:     2,
		ScalePointer:      3,
		BiasPointer:       4,
		MultiplierPointer: 5,
		OutputPointer:     6,
		Rows:              1,
		Cols:              8,
		Batch:             0,
		GroupSize:         8,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        32,
		WeightBytes:       4,
		ScaleBytes:        2,
		BiasBytes:         2,
		MultiplierBytes:   4,
		OutputBytes:       4,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "batch")

	req := hipMLXQ4ProjectionRequest{
		Input:     []float32{1, 1, 1, 1, 1, 1, 1, 1},
		Weight:    []uint32{0x76543210},
		Scales:    []uint16{0x3f80},
		Biases:    []uint16{0x0000},
		Rows:      1,
		Cols:      8,
		GroupSize: 8,
	}
	driver := &fakeHIPDriver{available: true}
	buffers, err := req.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer buffers.Close()
	multiplierPayload, err := hipFloat32Payload([]float32{1})
	core.AssertNoError(t, err)
	multiplier, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch multiplier", multiplierPayload, 1)
	core.AssertNoError(t, err)
	defer multiplier.Close()
	_, err = hipRunMLXQ4GELUTanhProjectionBatchKernelWithDeviceMultiplier(context.Background(), driver, buffers.Input, multiplier, hipMLXQ4DeviceWeightConfig{
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		WeightBytes:   buffers.Weight.SizeBytes(),
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		Rows:          req.Rows,
		Cols:          req.Cols,
		GroupSize:     req.GroupSize,
	}, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "batch size")

	batchInputPayload, err := hipFloat32Payload(append(append([]float32(nil), req.Input...), req.Input...))
	core.AssertNoError(t, err)
	batchInput, err := hipUploadByteBuffer(driver, "rocm.hip.MLXQ4GELUTanhProjectionBatchLaunch", "MLX q4 GELU tanh projection batch input", batchInputPayload, req.Cols*2)
	core.AssertNoError(t, err)
	defer batchInput.Close()
	_, err = hipRunMLXQ4GELUTanhProjectionBatchKernelWithDeviceMultiplier(context.Background(), driver, batchInput, multiplier, hipMLXQ4DeviceWeightConfig{
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		WeightBytes:   buffers.Weight.SizeBytes(),
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		Rows:          req.Rows,
		Cols:          req.Cols,
		GroupSize:     req.GroupSize,
	}, 2)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "multiplier device buffer shape mismatch")
}

func TestHIPKernels_MLXQ4ProjectionLaunchArgs_Bad(t *testing.T) {
	req := hipMLXQ4ProjectionRequest{
		Input:     []float32{1, 1, 1, 1, 1, 1, 1, 1},
		Weight:    []uint32{0x76543210},
		Scales:    []uint16{0x3f80},
		Biases:    []uint16{0x0000},
		Rows:      1,
		Cols:      8,
		GroupSize: 8,
	}
	_, err := req.deviceBuffers(nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "HIP driver is nil")

	driver := &fakeHIPDriver{available: true, copyErr: core.NewError("copy failed"), copyErrAt: 2}
	_, err = req.deviceBuffers(driver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy MLX q4 projection packed weights")

	buffers, err := req.deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer buffers.Close()
	_, err = (hipMLXQ4ProjectionRequest{
		Input:     req.Input,
		Weight:    req.Weight,
		Scales:    []uint16{0x3f80, 0x3f80},
		Biases:    []uint16{0, 0},
		Rows:      1,
		Cols:      8,
		GroupSize: 4,
	}).launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	_, err = (hipMLXQ4ProjectionLaunchArgs{
		InputPointer:  1,
		WeightPointer: 2,
		ScalePointer:  3,
		BiasPointer:   4,
		OutputPointer: 5,
		Rows:          1,
		Cols:          8,
		GroupSize:     8,
		Bits:          3,
		InputBytes:    32,
		WeightBytes:   4,
		ScaleBytes:    2,
		BiasBytes:     2,
		OutputBytes:   4,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "4-bit")

	_, err = (hipMLXQ4ProjectionLaunchArgs{
		InputPointer:  1,
		WeightPointer: 2,
		ScalePointer:  3,
		BiasPointer:   4,
		OutputPointer: 5,
		Rows:          1,
		Cols:          8,
		GroupSize:     8,
		Bits:          hipMLXQ4ProjectionBits,
		InputBytes:    32,
		WeightBytes:   8,
		ScaleBytes:    2,
		BiasBytes:     2,
		OutputBytes:   4,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "packed weight byte count")

	_, err = (hipMLXQ4ProjectionBatchLaunchArgs{
		InputPointer:  1,
		WeightPointer: 2,
		ScalePointer:  3,
		BiasPointer:   4,
		OutputPointer: 5,
		Rows:          1,
		Cols:          8,
		Batch:         2,
		GroupSize:     8,
		Bits:          hipMLXQ4ProjectionBits,
		InputBytes:    32,
		WeightBytes:   4,
		ScaleBytes:    2,
		BiasBytes:     2,
		OutputBytes:   8,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input byte count")

	_, err = hipRunMLXQ4ProjectionKernelWithDeviceWeightConfig(context.Background(), &fakeHIPDriver{available: true}, req.Input, hipMLXQ4DeviceWeightConfig{
		ScalePointer: buffers.Scales.Pointer(),
		BiasPointer:  buffers.Biases.Pointer(),
		WeightBytes:  buffers.Weight.SizeBytes(),
		ScaleBytes:   buffers.Scales.SizeBytes(),
		BiasBytes:    buffers.Biases.SizeBytes(),
		Rows:         req.Rows,
		Cols:         req.Cols,
		GroupSize:    req.GroupSize,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "pointers are required")

	_, err = hipRunMLXQ4ProjectionKernelWithDeviceWeightConfig(context.Background(), &fakeHIPDriver{available: true}, req.Input, hipMLXQ4DeviceWeightConfig{
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		WeightBytes:   buffers.Weight.SizeBytes() + 1,
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		Rows:          req.Rows,
		Cols:          req.Cols,
		GroupSize:     req.GroupSize,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "element-aligned")

	_, err = hipRunMLXQ4ProjectionBatchKernelWithDeviceInput(context.Background(), &fakeHIPDriver{available: true}, buffers.Input, hipMLXQ4DeviceWeightConfig{
		WeightPointer: buffers.Weight.Pointer(),
		ScalePointer:  buffers.Scales.Pointer(),
		BiasPointer:   buffers.Biases.Pointer(),
		WeightBytes:   buffers.Weight.SizeBytes(),
		ScaleBytes:    buffers.Scales.SizeBytes(),
		BiasBytes:     buffers.Biases.SizeBytes(),
		Rows:          req.Rows,
		Cols:          req.Cols,
		GroupSize:     req.GroupSize,
	}, 2)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "batch input count mismatch")
}

func TestHIPKernels_ProjectionReadOutputValidation_Bad(t *testing.T) {
	_, err := (*hipProjectionDeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "projection output buffer is required")

	req := hipProjectionRequest{Input: []float32{1}, F32: []float32{1}, Rows: 1, Cols: 1}
	driver := &fakeHIPDriver{available: true}
	buffers, err := req.projectionDeviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	buffers.Output.sizeBytes++
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "projection output byte count mismatch")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.projectionDeviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	payload, err := hipFloat32Payload([]float32{float32(math.NaN())})
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(buffers.Output.Pointer(), payload))
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	driver = &fakeHIPDriver{available: true}
	buffers, err = req.projectionDeviceBuffers(driver)
	core.RequireNoError(t, err)
	defer buffers.Close()
	driver.copyErr = core.NewError("copy failed")
	_, err = buffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy projection output")
}

func TestHIPKernels_RMSNormLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipRMSNormRequest{Input: []float32{3, 4}, Weight: []float32{1, 0.5}}
	buffers, err := req.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.AssertNoError(t, err)
	launchBytes, err := launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipRMSNormLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipRMSNormLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipRMSNormLaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(buffers.Input.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(buffers.Weight.Pointer()), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(launchBytes[24:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[32:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[36:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[40:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[44:]))
	core.AssertEqual(t, hipRMSNormWeightEncodingF32, binary.LittleEndian.Uint32(launchBytes[52:]))

	config, err := hipOneDimensionalLaunchConfig(hipKernelNameRMSNorm, launchBytes, buffers.Count)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	output, err := buffers.ReadOutput()
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0.8485, 0.5657}, output, 0.0001)

	bf16Req := hipRMSNormRequest{Input: []float32{3, 4}, WeightBF16: []uint16{0x3f80, 0x3f00}}
	bf16Buffers, err := bf16Req.deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer bf16Buffers.Close()
	bf16Launch, err := bf16Req.launchArgs(bf16Buffers)
	core.AssertNoError(t, err)
	bf16LaunchBytes, err := bf16Launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(bf16LaunchBytes[40:]))
	core.AssertEqual(t, hipRMSNormWeightEncodingBF16, binary.LittleEndian.Uint32(bf16LaunchBytes[52:]))
	config, err = hipOneDimensionalLaunchConfig(hipKernelNameRMSNorm, bf16LaunchBytes, bf16Buffers.Count)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(bf16Buffers.Input.driver, config))
	bf16Output, err := bf16Buffers.ReadOutput()
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0.8485, 0.5657}, bf16Output, 0.0001)

	gemmaReq := hipRMSNormRequest{Input: []float32{3, 4}, WeightBF16: []uint16{0x0000, 0xbf00}, AddUnitWeight: true}
	gemmaBuffers, err := gemmaReq.deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer gemmaBuffers.Close()
	gemmaLaunch, err := gemmaReq.launchArgs(gemmaBuffers)
	core.AssertNoError(t, err)
	gemmaLaunchBytes, err := gemmaLaunch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipRMSNormLaunchFlagAddUnitWeight, binary.LittleEndian.Uint32(gemmaLaunchBytes[56:]))
	config, err = hipOneDimensionalLaunchConfig(hipKernelNameRMSNorm, gemmaLaunchBytes, gemmaBuffers.Count)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(gemmaBuffers.Input.driver, config))
	gemmaOutput, err := gemmaBuffers.ReadOutput()
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0.8485, 0.5657}, gemmaOutput, 0.0001)

	gemmaRunnerOutput, err := hipRunRMSNormKernelWithDeviceWeightConfig(context.Background(), gemmaBuffers.Input.driver, gemmaReq.Input, hipRMSNormDeviceWeightConfig{
		WeightPointer:  gemmaBuffers.Weight.Pointer(),
		WeightBytes:    gemmaBuffers.Weight.SizeBytes(),
		Count:          len(gemmaReq.Input),
		WeightEncoding: hipRMSNormWeightEncodingBF16,
		Flags:          hipRMSNormLaunchFlagAddUnitWeight,
	})
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0.8485, 0.5657}, gemmaRunnerOutput, 0.0001)

	unitOutput, err := hipRunRMSNormKernelWithDeviceInputWeightConfig(context.Background(), driver, buffers.Input, hipRMSNormDeviceWeightConfig{
		Count:          len(req.Input),
		WeightEncoding: hipRMSNormWeightEncodingNone,
	})
	core.AssertNoError(t, err)
	defer unitOutput.Close()
	unitValues, err := hipReadFloat32DeviceOutput(unitOutput, "rocm.hip.RMSNormLaunch", "unit rms norm output", len(req.Input))
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0.8485, 1.1314}, unitValues, 0.0001)
}

func TestHIPKernels_RMSNormLaunchArgs_Bad(t *testing.T) {
	_, err := (hipRMSNormRequest{Input: []float32{1}, Weight: []float32{1}, Epsilon: -1}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "epsilon")

	_, err = (hipRMSNormRequest{Input: []float32{1}, Weight: []float32{1}, Epsilon: float32(math.NaN())}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = (hipRMSNormRequest{Input: []float32{1}, Weight: []float32{1}, WeightBF16: []uint16{0x3f80}}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "exactly one")

	buffers, err := (hipRMSNormRequest{Input: []float32{1}, Weight: []float32{1}}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer buffers.Close()
	_, err = (hipRMSNormRequest{Input: []float32{1, 2}, Weight: []float32{1, 1}}).launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	_, err = (hipRMSNormLaunchArgs{
		InputPointer:  1,
		WeightPointer: 2,
		OutputPointer: 3,
		Count:         2,
		InputBytes:    4,
		WeightBytes:   8,
		OutputBytes:   8,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input byte count")

	_, err = (hipRMSNormLaunchArgs{
		InputPointer:  1,
		WeightPointer: 2,
		OutputPointer: 3,
		Count:         1,
		InputBytes:    4,
		WeightBytes:   4,
		OutputBytes:   4,
		Epsilon:       float32(math.Inf(1)),
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = hipRunRMSNormKernelWithDeviceWeightConfig(context.Background(), &fakeHIPDriver{available: true}, []float32{1}, hipRMSNormDeviceWeightConfig{
		WeightPointer:  1,
		WeightBytes:    4,
		Count:          1,
		WeightEncoding: 999,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported")
}

func TestHIPKernels_RMSNormResidualAddLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	inputPayload, err := hipFloat32Payload([]float32{3, 4})
	core.AssertNoError(t, err)
	input, err := hipUploadByteBuffer(driver, "rocm.hip.RMSNormResidualAddLaunch", "input", inputPayload, 2)
	core.AssertNoError(t, err)
	defer input.Close()
	residualPayload, err := hipFloat32Payload([]float32{10, -1})
	core.AssertNoError(t, err)
	residual, err := hipUploadByteBuffer(driver, "rocm.hip.RMSNormResidualAddLaunch", "residual", residualPayload, 2)
	core.AssertNoError(t, err)
	defer residual.Close()
	weightPayload, err := hipFloat32Payload([]float32{1, 0.5})
	core.AssertNoError(t, err)
	weight, err := hipUploadByteBuffer(driver, "rocm.hip.RMSNormResidualAddLaunch", "weight", weightPayload, 2)
	core.AssertNoError(t, err)
	defer weight.Close()
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormResidualAddLaunch", "output", 8, 2)
	core.AssertNoError(t, err)
	defer output.Close()

	launchBytes, err := (hipRMSNormResidualAddLaunchArgs{
		InputPointer:    input.Pointer(),
		WeightPointer:   weight.Pointer(),
		ResidualPointer: residual.Pointer(),
		OutputPointer:   output.Pointer(),
		Count:           2,
		InputBytes:      input.SizeBytes(),
		WeightBytes:     weight.SizeBytes(),
		ResidualBytes:   residual.SizeBytes(),
		OutputBytes:     output.SizeBytes(),
		WeightEncoding:  hipRMSNormWeightEncodingF32,
		OutputScale:     0.5,
	}).Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipRMSNormResidualAddArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipRMSNormResidualAddArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipRMSNormResidualAddArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(input.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(weight.Pointer()), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint64(residual.Pointer()), binary.LittleEndian.Uint64(launchBytes[24:]))
	core.AssertEqual(t, uint64(output.Pointer()), binary.LittleEndian.Uint64(launchBytes[32:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[40:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[44:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[48:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[52:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[56:]))
	core.AssertEqual(t, hipRMSNormWeightEncodingF32, binary.LittleEndian.Uint32(launchBytes[64:]))
	core.AssertEqual(t, math.Float32bits(0.5), binary.LittleEndian.Uint32(launchBytes[72:]))

	config, err := hipSingleBlockLaunchConfig(hipKernelNameRMSNormResidualAdd, launchBytes, 256)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	values, err := hipReadFloat32DeviceOutput(output, "rocm.hip.RMSNormResidualAddLaunch", "output", 2)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{5.4243, -0.2172}, values, 0.0001)

	unitOutput, err := hipRunRMSNormResidualAddKernelWithDeviceInputWeightConfig(context.Background(), driver, input, residual, hipRMSNormDeviceWeightConfig{
		Count:          2,
		WeightEncoding: hipRMSNormWeightEncodingNone,
	})
	core.AssertNoError(t, err)
	defer unitOutput.Close()
	unitValues, err := hipReadFloat32DeviceOutput(unitOutput, "rocm.hip.RMSNormResidualAddLaunch", "unit output", 2)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{10.8485, 0.1314}, unitValues, 0.0001)

	scaledUnitOutput, err := hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfig(context.Background(), driver, input, residual, hipRMSNormDeviceWeightConfig{
		Count:          2,
		WeightEncoding: hipRMSNormWeightEncodingNone,
	}, 0.5)
	core.AssertNoError(t, err)
	defer scaledUnitOutput.Close()
	scaledUnitValues, err := hipReadFloat32DeviceOutput(scaledUnitOutput, "rocm.hip.RMSNormResidualAddLaunch", "scaled unit output", 2)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{5.4243, 0.0657}, scaledUnitValues, 0.0001)

	reusedOutput, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormResidualAddLaunch", "reused output", 8, 2)
	core.AssertNoError(t, err)
	defer reusedOutput.Close()
	core.AssertNoError(t, hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfigOutput(context.Background(), driver, input, residual, hipRMSNormDeviceWeightConfig{
		Count:          2,
		WeightEncoding: hipRMSNormWeightEncodingNone,
	}, reusedOutput, 0.5))
	reusedValues, err := hipReadFloat32DeviceOutput(reusedOutput, "rocm.hip.RMSNormResidualAddLaunch", "reused output", 2)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{5.4243, 0.0657}, reusedValues, 0.0001)
}

func TestHIPKernels_RMSNormResidualAddLaunchArgs_Bad(t *testing.T) {
	_, err := (hipRMSNormResidualAddLaunchArgs{
		InputPointer:    1,
		WeightPointer:   2,
		ResidualPointer: 3,
		OutputPointer:   4,
		Count:           2,
		InputBytes:      8,
		WeightBytes:     8,
		ResidualBytes:   4,
		OutputBytes:     8,
		WeightEncoding:  hipRMSNormWeightEncodingF32,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "residual byte count")

	driver := &fakeHIPDriver{available: true}
	inputPayload, err := hipFloat32Payload([]float32{1, 2})
	core.AssertNoError(t, err)
	input, err := hipUploadByteBuffer(driver, "rocm.hip.RMSNormResidualAddLaunch", "input", inputPayload, 2)
	core.AssertNoError(t, err)
	defer input.Close()
	residualPayload, err := hipFloat32Payload([]float32{1})
	core.AssertNoError(t, err)
	residual, err := hipUploadByteBuffer(driver, "rocm.hip.RMSNormResidualAddLaunch", "residual", residualPayload, 1)
	core.AssertNoError(t, err)
	defer residual.Close()
	_, err = hipRunRMSNormResidualAddKernelWithDeviceInputWeightConfig(context.Background(), driver, input, residual, hipRMSNormDeviceWeightConfig{
		Count:          2,
		WeightEncoding: hipRMSNormWeightEncodingNone,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")
}

func TestHIPKernels_RMSNormHeadsLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	inputPayload, err := hipFloat32Payload([]float32{3, 4, 6, 8})
	core.AssertNoError(t, err)
	input, err := hipUploadByteBuffer(driver, "rocm.hip.RMSNormHeadsLaunch", "rms norm heads input", inputPayload, 4)
	core.AssertNoError(t, err)
	defer input.Close()
	weightPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f00})
	core.AssertNoError(t, err)
	weight, err := hipUploadByteBuffer(driver, "rocm.hip.RMSNormHeadsLaunch", "rms norm heads bf16 weight", weightPayload, 2)
	core.AssertNoError(t, err)
	defer weight.Close()

	cfg := hipRMSNormDeviceWeightConfig{
		WeightPointer:  weight.Pointer(),
		WeightBytes:    weight.SizeBytes(),
		Count:          2,
		WeightEncoding: hipRMSNormWeightEncodingBF16,
	}
	output, err := hipRunRMSNormHeadsKernelWithDeviceInputWeightConfig(context.Background(), driver, input, cfg, 2)
	core.AssertNoError(t, err)
	defer output.Close()
	values, err := hipReadFloat32DeviceOutput(output, "rocm.hip.RMSNormHeadsLaunch", "rms norm heads output", 4)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0.8485, 0.5657, 0.8485, 0.5657}, values, 0.0001)

	launchBytes, err := (hipRMSNormHeadsLaunchArgs{
		InputPointer:   input.Pointer(),
		WeightPointer:  weight.Pointer(),
		OutputPointer:  output.Pointer(),
		HeadDim:        2,
		HeadCount:      2,
		InputBytes:     input.SizeBytes(),
		WeightBytes:    weight.SizeBytes(),
		OutputBytes:    output.SizeBytes(),
		WeightEncoding: hipRMSNormWeightEncodingBF16,
	}).Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipRMSNormHeadsLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipRMSNormHeadsLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[32:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[36:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[44:]))
}

func TestHIPKernels_RMSNormRoPEHeadsLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	inputValues := []float32{1, 0, 3, 4, 0, 2, 5, 12}
	inputPayload, err := hipFloat32Payload(inputValues)
	core.AssertNoError(t, err)
	input, err := hipUploadByteBuffer(driver, "rocm.hip.RMSNormRoPEHeadsLaunch", "rms norm rope heads input", inputPayload, len(inputValues))
	core.AssertNoError(t, err)
	defer input.Close()

	cfg := hipRMSNormDeviceWeightConfig{
		Count:          4,
		WeightEncoding: hipRMSNormWeightEncodingNone,
	}
	output, err := hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfig(context.Background(), driver, input, cfg, 2, 1, 1, 4, 2)
	core.AssertNoError(t, err)
	defer output.Close()
	values, err := hipReadFloat32DeviceOutput(output, "rocm.hip.RMSNormRoPEHeadsLaunch", "rms norm rope heads output", len(inputValues))
	core.AssertNoError(t, err)

	var want []float32
	unitWeight := []float32{1, 1, 1, 1}
	for head := 0; head < 2; head++ {
		start := head * 4
		normalized, err := hipReferenceRMSNorm(inputValues[start:start+4], unitWeight, 0)
		core.AssertNoError(t, err)
		rotated, err := hipReferenceRoPEWithFrequencyDim(normalized[:2], 1, 1, 4)
		core.AssertNoError(t, err)
		normalized[0] = rotated[0]
		normalized[1] = rotated[1]
		want = append(want, normalized...)
	}
	assertFloat32SlicesNear(t, want, values, 0.0001)

	reusedOutput, err := hipAllocateByteBuffer(driver, "rocm.hip.RMSNormRoPEHeadsLaunch", "reused rms norm rope heads output", input.SizeBytes(), input.Count())
	core.AssertNoError(t, err)
	defer reusedOutput.Close()
	core.AssertNoError(t, hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutput(context.Background(), driver, input, cfg, 2, 1, 1, 4, 2, reusedOutput))
	reusedValues, err := hipReadFloat32DeviceOutput(reusedOutput, "rocm.hip.RMSNormRoPEHeadsLaunch", "reused rms norm rope heads output", len(inputValues))
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, want, reusedValues, 0.0001)

	neoxCfg := cfg
	neoxCfg.Flags = hipRMSNormLaunchFlagRoPENeoX
	neoxOutput, err := hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfig(context.Background(), driver, input, neoxCfg, 2, 1, 1, 4, 2)
	core.AssertNoError(t, err)
	defer neoxOutput.Close()
	neoxValues, err := hipReadFloat32DeviceOutput(neoxOutput, "rocm.hip.RMSNormRoPEHeadsLaunch", "rms norm rope heads neox output", len(inputValues))
	core.AssertNoError(t, err)
	want = want[:0]
	for head := 0; head < 2; head++ {
		start := head * 4
		normalized, err := hipReferenceRMSNorm(inputValues[start:start+4], unitWeight, 0)
		core.AssertNoError(t, err)
		rotated, err := hipReferenceRoPENeoXWithFrequencyDim(normalized, 1, 1, 4, 2)
		core.AssertNoError(t, err)
		want = append(want, rotated...)
	}
	assertFloat32SlicesNear(t, want, neoxValues, 0.0001)

	launchBytes, err := (hipRMSNormRoPEHeadsLaunchArgs{
		InputPointer:   input.Pointer(),
		OutputPointer:  output.Pointer(),
		HeadDim:        4,
		HeadCount:      2,
		InputBytes:     input.SizeBytes(),
		OutputBytes:    output.SizeBytes(),
		WeightEncoding: hipRMSNormWeightEncodingNone,
		Flags:          hipRMSNormLaunchFlagRoPENeoX,
		Position:       1,
		Base:           1,
		FrequencyDim:   4,
		RotaryCount:    2,
	}).Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipRMSNormRoPEHeadsLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipRMSNormRoPEHeadsLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[32:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[36:]))
	core.AssertEqual(t, hipRMSNormLaunchFlagRoPENeoX, binary.LittleEndian.Uint32(launchBytes[60:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[72:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[76:]))
}

func TestHIPKernels_RMSNormRoPEHeadsBatchLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	inputValues := []float32{
		1, 0, 3, 4,
		0, 2, 5, 12,
		2, 0, 1, 1,
		0, 3, 4, 3,
	}
	inputPayload, err := hipFloat32Payload(inputValues)
	core.AssertNoError(t, err)
	input, err := hipUploadByteBuffer(driver, "rocm.hip.RMSNormRoPEHeadsBatchLaunch", "rms norm rope heads batch input", inputPayload, len(inputValues))
	core.AssertNoError(t, err)
	defer input.Close()

	cfg := hipRMSNormDeviceWeightConfig{
		Count:          4,
		WeightEncoding: hipRMSNormWeightEncodingNone,
	}
	output, err := hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfig(context.Background(), driver, input, cfg, 2, 2, 3, 1, 4, 2)
	core.AssertNoError(t, err)
	defer output.Close()
	values, err := hipReadFloat32DeviceOutput(output, "rocm.hip.RMSNormRoPEHeadsBatchLaunch", "rms norm rope heads batch output", len(inputValues))
	core.AssertNoError(t, err)

	var want []float32
	unitWeight := []float32{1, 1, 1, 1}
	for batch := 0; batch < 2; batch++ {
		for head := 0; head < 2; head++ {
			start := (batch*2 + head) * 4
			normalized, err := hipReferenceRMSNorm(inputValues[start:start+4], unitWeight, 0)
			core.AssertNoError(t, err)
			rotated, err := hipReferenceRoPEWithFrequencyDim(normalized[:2], 3+batch, 1, 4)
			core.AssertNoError(t, err)
			normalized[0] = rotated[0]
			normalized[1] = rotated[1]
			want = append(want, normalized...)
		}
	}
	assertFloat32SlicesNear(t, want, values, 0.0001)

	launches := driver.launches
	core.AssertEqual(t, 1, len(launches))
	core.AssertEqual(t, hipKernelNameRMSNormRoPEHeadsBatch, launches[0].Name)
	core.AssertEqual(t, uint32(2), launches[0].GridX)
	core.AssertEqual(t, uint32(2), launches[0].GridY)

	launchBytes, err := (hipRMSNormRoPEHeadsBatchLaunchArgs{
		InputPointer:   input.Pointer(),
		OutputPointer:  output.Pointer(),
		HeadDim:        4,
		HeadCount:      2,
		Batch:          2,
		InputBytes:     input.SizeBytes(),
		OutputBytes:    output.SizeBytes(),
		WeightEncoding: hipRMSNormWeightEncodingNone,
		StartPosition:  3,
		Base:           1,
		FrequencyDim:   4,
		RotaryCount:    2,
	}).Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipRMSNormRoPEHeadsBatchLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipRMSNormRoPEHeadsBatchLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[32:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[36:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[40:]))
	core.AssertEqual(t, uint32(len(inputValues)*4), binary.LittleEndian.Uint32(launchBytes[44:]))
	core.AssertEqual(t, uint32(len(inputValues)*4), binary.LittleEndian.Uint32(launchBytes[52:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(launchBytes[68:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[76:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[80:]))
}

func TestHIPKernels_RMSNormRoPEHeadsBatchLaunchArgs_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	inputPayload, err := hipFloat32Payload([]float32{1, 0, 3, 4})
	core.AssertNoError(t, err)
	input, err := hipUploadByteBuffer(driver, "rocm.hip.RMSNormRoPEHeadsBatchLaunch", "rms norm rope heads batch bad input", inputPayload, 4)
	core.AssertNoError(t, err)
	defer input.Close()

	cfg := hipRMSNormDeviceWeightConfig{
		Count:          4,
		WeightEncoding: hipRMSNormWeightEncodingNone,
	}
	if _, err := hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfig(context.Background(), driver, input, cfg, 2, 1, 0, 1, 0, 0); err == nil {
		t.Fatalf("hipRunRMSNormRoPEHeadsBatchKernelWithDeviceInputWeightConfig succeeded with mismatched input count")
	}
	if _, err := (hipRMSNormRoPEHeadsBatchLaunchArgs{
		InputPointer:   input.Pointer(),
		OutputPointer:  input.Pointer(),
		HeadDim:        4,
		HeadCount:      1,
		Batch:          1,
		InputBytes:     input.SizeBytes(),
		OutputBytes:    input.SizeBytes(),
		WeightEncoding: hipRMSNormWeightEncodingNone,
		StartPosition:  -1,
		Base:           1,
	}).Binary(); err == nil {
		t.Fatalf("hipRMSNormRoPEHeadsBatchLaunchArgs.Binary succeeded with negative start position")
	}
	core.AssertEqual(t, 0, len(driver.launches))
}

func TestHIPKernels_RoPELaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipRoPERequest{Input: []float32{1, 0}, Position: 1, Base: 1}
	buffers, err := req.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.AssertNoError(t, err)
	launchBytes, err := launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipRoPELaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipRoPELaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipRoPELaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(buffers.Input.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[24:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[28:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[32:]))
	core.AssertEqual(t, uint32(1), binary.LittleEndian.Uint32(launchBytes[36:]))
	core.AssertEqual(t, math.Float32bits(1), binary.LittleEndian.Uint32(launchBytes[40:]))
	core.AssertEqual(t, uint32(0), binary.LittleEndian.Uint32(launchBytes[44:]))

	config, err := hipOneDimensionalLaunchConfig(hipKernelNameRoPE, launchBytes, buffers.Count)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	output, err := buffers.ReadOutput()
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{float32(math.Cos(1)), float32(math.Sin(1))}, output, 0.0001)

	runnerOutput, err := hipRunRoPEKernel(context.Background(), &fakeHIPDriver{available: true}, req)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{float32(math.Cos(1)), float32(math.Sin(1))}, runnerOutput, 0.0001)

	frequencyReq := hipRoPERequest{Input: []float32{1, 0, 1, 0}, Position: 1, Base: 10000, FrequencyDim: 8}
	frequencyBuffers, err := frequencyReq.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer frequencyBuffers.Close()
	frequencyLaunch, err := frequencyReq.launchArgs(frequencyBuffers)
	core.AssertNoError(t, err)
	frequencyLaunchBytes, err := frequencyLaunch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(frequencyLaunchBytes[44:]))
	frequencyOutput, err := hipRunRoPEKernel(context.Background(), &fakeHIPDriver{available: true}, frequencyReq)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{
		float32(math.Cos(1)),
		float32(math.Sin(1)),
		float32(math.Cos(0.1)),
		float32(math.Sin(0.1)),
	}, frequencyOutput, 0.0001)
}

func TestHIPKernels_RoPEHeadsLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	inputPayload, err := hipFloat32Payload([]float32{1, 0, 1, 0, 1, 0, 1, 0})
	core.AssertNoError(t, err)
	input, err := hipUploadByteBuffer(driver, "rocm.hip.RoPEHeadsLaunch", "rope heads input", inputPayload, 8)
	core.AssertNoError(t, err)
	defer input.Close()

	output, err := hipRunRoPEHeadsDeviceKernelWithRotaryCount(context.Background(), driver, input, 4, 2, 1, 10000, 8, 0)
	core.AssertNoError(t, err)
	defer output.Close()
	values, err := hipReadFloat32DeviceOutput(output, "rocm.hip.RoPEHeadsLaunch", "rope heads output", 8)
	core.AssertNoError(t, err)
	want := []float32{
		float32(math.Cos(1)),
		float32(math.Sin(1)),
		float32(math.Cos(0.1)),
		float32(math.Sin(0.1)),
		float32(math.Cos(1)),
		float32(math.Sin(1)),
		float32(math.Cos(0.1)),
		float32(math.Sin(0.1)),
	}
	assertFloat32SlicesNear(t, want, values, 0.0001)

	launchBytes, err := (hipRoPEHeadsLaunchArgs{
		InputPointer:  input.Pointer(),
		OutputPointer: output.Pointer(),
		HeadDim:       4,
		HeadCount:     2,
		InputBytes:    input.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		Position:      1,
		Base:          10000,
		FrequencyDim:  8,
	}).Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipRoPEHeadsLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipRoPEHeadsLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(4), binary.LittleEndian.Uint32(launchBytes[24:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[28:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[48:]))
}

func TestHIPKernels_RoPELaunchArgs_Bad(t *testing.T) {
	_, err := (hipRoPERequest{Input: []float32{1}, Position: 0, Base: 1}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "positive and even")

	_, err = (hipRoPERequest{Input: []float32{1, 0}, Position: 0, Base: float32(math.NaN())}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = (hipRoPERequest{Input: []float32{1, 0, 0, 1}, Position: 0, Base: 1, FrequencyDim: 2}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "frequency dimension")

	buffers, err := (hipRoPERequest{Input: []float32{1, 0}, Position: 0, Base: 1}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer buffers.Close()
	_, err = (hipRoPERequest{Input: []float32{1, 0, 0, 1}, Position: 0, Base: 1}).launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	_, err = (hipRoPELaunchArgs{
		InputPointer:  1,
		OutputPointer: 2,
		Count:         3,
		InputBytes:    12,
		OutputBytes:   12,
		Base:          1,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "count must be even")

	_, err = (hipRoPELaunchArgs{
		InputPointer:  1,
		OutputPointer: 2,
		Count:         2,
		InputBytes:    8,
		OutputBytes:   8,
		Base:          float32(math.Inf(1)),
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")
}

func TestHIPKernels_GreedySampleLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipGreedySampleRequest{Logits: []float32{-1, 0.25, 0.2}}
	buffers, err := req.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.AssertNoError(t, err)
	launchBytes, err := launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipGreedyLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipGreedyLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipGreedyLaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(buffers.Logits.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(launchBytes[24:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[28:]))
	core.AssertEqual(t, uint32(hipGreedyResultBytes), binary.LittleEndian.Uint32(launchBytes[32:]))

	config, err := hipOneDimensionalLaunchConfig(hipKernelNameGreedy, launchBytes, 1)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	output, err := buffers.ReadOutput()
	core.AssertNoError(t, err)
	core.AssertEqual(t, 1, output.TokenID)
	assertFloat32Near(t, 0.25, output.Score)

	runnerOutput, err := hipRunGreedyKernel(context.Background(), &fakeHIPDriver{available: true}, req)
	core.AssertNoError(t, err)
	core.AssertEqual(t, 1, runnerOutput.TokenID)
	assertFloat32Near(t, 0.25, runnerOutput.Score)
}

func TestHIPKernels_GreedySampleLaunchArgs_Bad(t *testing.T) {
	_, err := (hipGreedySampleRequest{}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "logits")

	buffers, err := (hipGreedySampleRequest{Logits: []float32{1}}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer buffers.Close()
	_, err = (hipGreedySampleRequest{Logits: []float32{1, 2}}).launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	_, err = (hipGreedySampleLaunchArgs{
		LogitsPointer: 1,
		OutputPointer: 2,
		Count:         2,
		LogitsBytes:   4,
		OutputBytes:   hipGreedyResultBytes,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "logits byte count")
}

func TestHIPKernels_SoftcapGreedySampleLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	payload, err := hipFloat32Payload([]float32{-1, 30, 29})
	core.AssertNoError(t, err)
	logits, err := hipUploadByteBuffer(driver, "rocm.hip.SoftcapGreedyLaunch", "softcap greedy logits", payload, 3)
	core.AssertNoError(t, err)
	defer logits.Close()
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.SoftcapGreedyLaunch", "softcap greedy output", hipGreedyResultBytes, 1)
	core.AssertNoError(t, err)
	defer output.Close()

	launchBytes, err := (hipSoftcapGreedySampleLaunchArgs{
		LogitsPointer: logits.Pointer(),
		OutputPointer: output.Pointer(),
		Count:         3,
		LogitsBytes:   logits.SizeBytes(),
		OutputBytes:   output.SizeBytes(),
		Softcap:       30,
	}).Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipSoftcapGreedyLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipSoftcapGreedyLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipSoftcapGreedyLaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(logits.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(output.Pointer()), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(launchBytes[24:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[28:]))
	core.AssertEqual(t, uint32(hipGreedyResultBytes), binary.LittleEndian.Uint32(launchBytes[32:]))
	assertFloat32Near(t, 30, math.Float32frombits(binary.LittleEndian.Uint32(launchBytes[36:])))

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
	core.AssertNoError(t, config.Validate())
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	got, err := hipReadGreedyResult(output, "rocm.hip.SoftcapGreedyLaunch", "softcap greedy output", logits.Count())
	core.AssertNoError(t, err)
	core.AssertEqual(t, 1, got.TokenID)
	assertFloat32Near(t, float32(math.Tanh(1))*30, got.Score)

	runnerOutput, err := hipRunSoftcapGreedyKernelWithDeviceLogits(context.Background(), driver, logits, 30)
	core.AssertNoError(t, err)
	core.AssertEqual(t, 1, runnerOutput.TokenID)
	assertFloat32Near(t, float32(math.Tanh(1))*30, runnerOutput.Score)
}

func TestHIPKernels_SoftcapGreedySampleLaunchArgs_Bad(t *testing.T) {
	_, err := (hipSoftcapGreedySampleLaunchArgs{
		LogitsPointer: 1,
		OutputPointer: 2,
		Count:         2,
		LogitsBytes:   4,
		OutputBytes:   hipGreedyResultBytes,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "logits byte count")

	_, err = (hipSoftcapGreedySampleLaunchArgs{
		LogitsPointer: 1,
		OutputPointer: 2,
		Count:         2,
		LogitsBytes:   8,
		OutputBytes:   hipGreedyResultBytes,
		Softcap:       float32(math.Inf(1)),
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "softcap")

	_, err = hipRunSoftcapGreedyKernelWithDeviceLogits(context.Background(), &fakeHIPDriver{available: true}, nil, 30)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "logits")
}

func TestHIPKernels_AttentionLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipAttentionRequest{
		Query:  []float32{1, 0},
		Keys:   []float32{1, 0, 0, 1},
		Values: []float32{2, 0, 0, 4},
	}
	buffers, err := req.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.AssertNoError(t, err)
	launchBytes, err := launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipAttentionLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipAttentionLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipAttentionLaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(buffers.Query.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(buffers.Keys.Pointer()), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint64(buffers.Values.Pointer()), binary.LittleEndian.Uint64(launchBytes[24:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(launchBytes[32:]))
	core.AssertEqual(t, uint64(buffers.Weights.Pointer()), binary.LittleEndian.Uint64(launchBytes[40:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[48:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[52:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[56:]))
	core.AssertEqual(t, uint32(16), binary.LittleEndian.Uint32(launchBytes[60:]))
	core.AssertEqual(t, uint32(16), binary.LittleEndian.Uint32(launchBytes[64:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[68:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[72:]))
	core.AssertEqual(t, hipAttentionKVSourceContiguous, binary.LittleEndian.Uint32(launchBytes[76:]))
	core.AssertEqual(t, uint32(0), binary.LittleEndian.Uint32(launchBytes[80:]))

	config, err := hipOneDimensionalLaunchConfig(hipKernelNameAttention, launchBytes, 1)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	output, err := buffers.ReadOutput()
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1.3395, 1.3210}, output.Output, 0.0001)
	assertFloat32SlicesNear(t, []float32{0.6698, 0.3302}, output.Weights, 0.0001)

	runnerOutput, err := hipRunAttentionKernel(context.Background(), &fakeHIPDriver{available: true}, req)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1.3395, 1.3210}, runnerOutput.Output, 0.0001)
	assertFloat32SlicesNear(t, []float32{0.6698, 0.3302}, runnerOutput.Weights, 0.0001)

	scaledReq := req
	scaledReq.Scale = 1
	scaledOutput, err := hipRunAttentionKernel(context.Background(), &fakeHIPDriver{available: true}, scaledReq)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1.4621, 1.0758}, scaledOutput.Output, 0.0001)
	assertFloat32SlicesNear(t, []float32{0.7311, 0.2689}, scaledOutput.Weights, 0.0001)

	deviceDriver := &fakeHIPDriver{available: true}
	cache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.AssertNoError(t, err)
	core.AssertNoError(t, cache.AppendVectors(0, 2, 2, req.Keys, req.Values))
	deviceKV, err := cache.MirrorToDevice(deviceDriver)
	core.AssertNoError(t, err)
	defer deviceKV.Close()
	table, err := deviceKV.KernelDescriptorTable()
	core.AssertNoError(t, err)
	defer table.Close()
	deviceReq := hipAttentionRequest{Query: req.Query, DeviceKV: deviceKV, DescriptorTable: table}
	deviceBuffers, err := deviceReq.deviceBuffers(deviceDriver)
	core.AssertNoError(t, err)
	defer deviceBuffers.Close()
	deviceLaunch, err := deviceReq.launchArgs(deviceBuffers)
	core.AssertNoError(t, err)
	deviceLaunchBytes, err := deviceLaunch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipAttentionLaunchArgsBytes, len(deviceLaunchBytes))
	core.AssertEqual(t, uint64(deviceBuffers.Query.Pointer()), binary.LittleEndian.Uint64(deviceLaunchBytes[8:]))
	core.AssertEqual(t, uint64(0), binary.LittleEndian.Uint64(deviceLaunchBytes[16:]))
	core.AssertEqual(t, uint64(0), binary.LittleEndian.Uint64(deviceLaunchBytes[24:]))
	core.AssertEqual(t, uint32(0), binary.LittleEndian.Uint32(deviceLaunchBytes[60:]))
	core.AssertEqual(t, uint32(0), binary.LittleEndian.Uint32(deviceLaunchBytes[64:]))
	core.AssertEqual(t, hipAttentionKVSourceDevice, binary.LittleEndian.Uint32(deviceLaunchBytes[76:]))
	core.AssertEqual(t, uint32(0), binary.LittleEndian.Uint32(deviceLaunchBytes[80:]))
	core.AssertEqual(t, uint64(table.Pointer()), binary.LittleEndian.Uint64(deviceLaunchBytes[88:]))
	core.AssertEqual(t, table.SizeBytes(), binary.LittleEndian.Uint64(deviceLaunchBytes[96:]))
	deviceConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameAttention, deviceLaunchBytes, 1)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(deviceDriver, deviceConfig))
	deviceOutput, err := deviceBuffers.ReadOutput()
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1.3395, 1.3210}, deviceOutput.Output, 0.0001)
	assertFloat32SlicesNear(t, []float32{0.6698, 0.3302}, deviceOutput.Weights, 0.0001)

	deviceRunnerOutput, err := hipRunAttentionKernel(context.Background(), deviceDriver, deviceReq)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1.3395, 1.3210}, deviceRunnerOutput.Output, 0.0001)
	assertFloat32SlicesNear(t, []float32{0.6698, 0.3302}, deviceRunnerOutput.Weights, 0.0001)

	for _, mode := range []string{rocmKVCacheModeQ8, rocmKVCacheModeKQ8VQ4} {
		modeDriver := &fakeHIPDriver{available: true}
		modeCache, err := newROCmKVCache(mode, defaultROCmKVBlockSize)
		core.AssertNoError(t, err)
		core.AssertNoError(t, modeCache.AppendVectors(0, 2, 2, req.Keys, req.Values))
		modeDeviceKV, err := modeCache.MirrorToDevice(modeDriver)
		core.AssertNoError(t, err)
		defer modeDeviceKV.Close()
		modeTable, err := modeDeviceKV.KernelDescriptorTable()
		core.AssertNoError(t, err)
		defer modeTable.Close()
		modeOutput, err := hipRunAttentionKernel(context.Background(), modeDriver, hipAttentionRequest{
			Query:           req.Query,
			DeviceKV:        modeDeviceKV,
			DescriptorTable: modeTable,
		})
		core.AssertNoError(t, err)
		restoredKeys, restoredValues, err := modeCache.Restore(0, modeCache.TokenCount())
		core.AssertNoError(t, err)
		wantKeys, err := splitHIPReferenceVectors(restoredKeys, 2)
		core.AssertNoError(t, err)
		wantValues, err := splitHIPReferenceVectors(restoredValues, 2)
		core.AssertNoError(t, err)
		wantOutput, wantWeights, err := hipReferenceSingleHeadAttention(req.Query, wantKeys, wantValues)
		core.AssertNoError(t, err)
		assertFloat32SlicesNear(t, wantOutput, modeOutput.Output, 0.0001)
		assertFloat32SlicesNear(t, wantWeights, modeOutput.Weights, 0.0001)
	}
}

func TestHIPKernels_AttentionHeadsBatchCausalLaunchArgs_Good(t *testing.T) {
	const (
		dim             = 2
		tokenCount      = 3
		headCount       = 2
		queryCount      = 2
		queryStartToken = 1
	)
	queryValues := []float32{
		1, 0,
		0, 1,
		0, 1,
		1, 1,
	}
	keyValues := []float32{
		1, 0,
		0, 1,
		1, 1,
	}
	valueValues := []float32{
		2, 0,
		0, 4,
		4, 4,
	}
	wantOutput := func(t *testing.T) []float32 {
		t.Helper()
		keys, err := splitHIPReferenceVectors(keyValues, dim)
		core.RequireNoError(t, err)
		values, err := splitHIPReferenceVectors(valueValues, dim)
		core.RequireNoError(t, err)
		out := make([]float32, 0, queryCount*headCount*dim)
		for queryIndex := 0; queryIndex < queryCount; queryIndex++ {
			visibleTokens := queryStartToken + queryIndex + 1
			for head := 0; head < headCount; head++ {
				queryBase := (queryIndex*headCount + head) * dim
				headOutput, _, err := hipReferenceSingleHeadAttentionWithScale(queryValues[queryBase:queryBase+dim], keys[:visibleTokens], values[:visibleTokens], 1)
				core.RequireNoError(t, err)
				out = append(out, headOutput...)
			}
		}
		return out
	}

	driver := &fakeHIPDriver{available: true}
	queryPayload, err := hipFloat32Payload(queryValues)
	core.RequireNoError(t, err)
	query, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch query", queryPayload, len(queryValues))
	core.RequireNoError(t, err)
	defer query.Close()
	keyPayload, err := hipFloat32Payload(keyValues)
	core.RequireNoError(t, err)
	keys, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch keys", keyPayload, len(keyValues))
	core.RequireNoError(t, err)
	defer keys.Close()
	valuePayload, err := hipFloat32Payload(valueValues)
	core.RequireNoError(t, err)
	values, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch values", valuePayload, len(valueValues))
	core.RequireNoError(t, err)
	defer values.Close()
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch output", uint64(len(queryValues)*4), len(queryValues))
	core.RequireNoError(t, err)
	defer output.Close()

	start := len(driver.launches)
	err = hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernel(context.Background(), driver, hipAttentionHeadsBatchCausalDeviceRequest{
		Key:             keys,
		Value:           values,
		Dim:             dim,
		TokenCount:      tokenCount,
		HeadCount:       headCount,
		QueryCount:      queryCount,
		QueryStartToken: queryStartToken,
		Scale:           1,
	}, query, output)
	core.RequireNoError(t, err)
	got, err := hipReadFloat32DeviceOutput(output, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch output", len(queryValues))
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, wantOutput(t), got, 0.0001)
	launches := driver.launches[start:]
	core.AssertEqual(t, 1, len(launches))
	launch := launches[0]
	core.AssertEqual(t, hipKernelNameAttentionHeadsBatchCausal, launch.Name)
	core.AssertEqual(t, uint32(headCount), launch.GridX)
	core.AssertEqual(t, uint32(queryCount), launch.GridY)
	core.AssertEqual(t, hipAttentionHeadsBlockSize(tokenCount), launch.BlockX)
	core.AssertEqual(t, hipAttentionHeadsBatchCausalLaunchArgsBytes, len(launch.Args))
	core.AssertEqual(t, hipAttentionHeadsBatchCausalLaunchArgsVersion, binary.LittleEndian.Uint32(launch.Args[0:]))
	core.AssertEqual(t, uint32(hipAttentionHeadsBatchCausalLaunchArgsBytes), binary.LittleEndian.Uint32(launch.Args[4:]))
	core.AssertEqual(t, uint64(query.Pointer()), binary.LittleEndian.Uint64(launch.Args[8:]))
	core.AssertEqual(t, uint64(keys.Pointer()), binary.LittleEndian.Uint64(launch.Args[16:]))
	core.AssertEqual(t, uint64(values.Pointer()), binary.LittleEndian.Uint64(launch.Args[24:]))
	core.AssertEqual(t, uint64(output.Pointer()), binary.LittleEndian.Uint64(launch.Args[32:]))
	core.AssertEqual(t, uint32(dim), binary.LittleEndian.Uint32(launch.Args[48:]))
	core.AssertEqual(t, uint32(tokenCount), binary.LittleEndian.Uint32(launch.Args[52:]))
	core.AssertEqual(t, uint32(headCount), binary.LittleEndian.Uint32(launch.Args[56:]))
	core.AssertEqual(t, uint32(queryCount), binary.LittleEndian.Uint32(launch.Args[60:]))
	core.AssertEqual(t, uint32(queryStartToken), binary.LittleEndian.Uint32(launch.Args[64:]))
	core.AssertEqual(t, uint32(len(queryValues)*4), binary.LittleEndian.Uint32(launch.Args[68:]))
	core.AssertEqual(t, uint32(len(keyValues)*4), binary.LittleEndian.Uint32(launch.Args[72:]))
	core.AssertEqual(t, uint32(len(valueValues)*4), binary.LittleEndian.Uint32(launch.Args[76:]))
	core.AssertEqual(t, uint32(len(queryValues)*4), binary.LittleEndian.Uint32(launch.Args[80:]))
	core.AssertEqual(t, uint32(0), binary.LittleEndian.Uint32(launch.Args[84:]))
	core.AssertEqual(t, hipAttentionKVSourceContiguous, binary.LittleEndian.Uint32(launch.Args[88:]))
	core.AssertEqual(t, math.Float32bits(1), binary.LittleEndian.Uint32(launch.Args[92:]))

	deviceDriver := &fakeHIPDriver{available: true}
	deviceQuery, err := hipUploadByteBuffer(deviceDriver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch device-KV query", queryPayload, len(queryValues))
	core.RequireNoError(t, err)
	defer deviceQuery.Close()
	deviceOutput, err := hipAllocateByteBuffer(deviceDriver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch device-KV output", uint64(len(queryValues)*4), len(queryValues))
	core.RequireNoError(t, err)
	defer deviceOutput.Close()
	cache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, dim, dim, keyValues, valueValues))
	deviceKV, err := cache.MirrorToDevice(deviceDriver)
	core.RequireNoError(t, err)
	defer deviceKV.Close()
	table, err := deviceKV.KernelDescriptorTable()
	core.RequireNoError(t, err)
	defer table.Close()
	err = hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernel(context.Background(), deviceDriver, hipAttentionHeadsBatchCausalDeviceRequest{
		DeviceKV:        deviceKV,
		DescriptorTable: table,
		Dim:             dim,
		TokenCount:      tokenCount,
		HeadCount:       headCount,
		QueryCount:      queryCount,
		QueryStartToken: queryStartToken,
		Scale:           1,
	}, deviceQuery, deviceOutput)
	core.RequireNoError(t, err)
	deviceGot, err := hipReadFloat32DeviceOutput(deviceOutput, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch device-KV output", len(queryValues))
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, wantOutput(t), deviceGot, 0.0001)
}

func TestHIPKernels_AttentionHeadsBatchCausalLaunchArgs_Bad(t *testing.T) {
	_, err := (hipAttentionHeadsBatchCausalLaunchArgs{
		QueryPointer:    1,
		KeyPointer:      2,
		ValuePointer:    3,
		OutputPointer:   4,
		Dim:             2,
		TokenCount:      3,
		HeadCount:       2,
		QueryCount:      2,
		QueryStartToken: 2,
		QueryBytes:      16,
		KeyBytes:        24,
		ValueBytes:      24,
		OutputBytes:     16,
		KVSource:        hipAttentionKVSourceContiguous,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "causal query window")

	driver := &fakeHIPDriver{available: true}
	payload, err := hipFloat32Payload([]float32{1, 0, 0, 1})
	core.RequireNoError(t, err)
	query, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "bad attention batch query", payload, 4)
	core.RequireNoError(t, err)
	defer query.Close()
	output, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "bad attention batch output", 4, 1)
	core.RequireNoError(t, err)
	defer output.Close()
	start := len(driver.launches)
	err = hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernel(context.Background(), driver, hipAttentionHeadsBatchCausalDeviceRequest{
		Dim:             2,
		TokenCount:      1,
		HeadCount:       1,
		QueryCount:      2,
		QueryStartToken: 0,
	}, query, output)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "causal query window")
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPKernels_AttentionLaunchArgs_Bad(t *testing.T) {
	_, err := (hipAttentionRequest{Query: []float32{1}, Keys: []float32{1, 2}, Values: []float32{1}}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "same token count")

	buffers, err := (hipAttentionRequest{Query: []float32{1}, Keys: []float32{1}, Values: []float32{1}}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer buffers.Close()
	_, err = (hipAttentionRequest{Query: []float32{1, 0}, Keys: []float32{1, 0}, Values: []float32{1, 0}}).launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	_, err = (hipAttentionRequest{Query: []float32{1}, Keys: []float32{1}, Values: []float32{1}, Scale: float32(math.NaN())}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "scale")

	deviceDriver := &fakeHIPDriver{available: true}
	cache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.AssertNoError(t, err)
	core.AssertNoError(t, cache.AppendVectors(0, 1, 1, []float32{1}, []float32{1}))
	deviceKV, err := cache.MirrorToDevice(deviceDriver)
	core.AssertNoError(t, err)
	defer deviceKV.Close()
	table, err := deviceKV.KernelDescriptorTable()
	core.AssertNoError(t, err)
	defer table.Close()
	_, err = (hipAttentionRequest{Query: []float32{1}, DeviceKV: deviceKV}).deviceBuffers(deviceDriver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "descriptor table")
	_, err = (hipAttentionRequest{Query: []float32{1}, DescriptorTable: table}).deviceBuffers(deviceDriver)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "descriptor table requires device KV cache")

	_, err = (hipAttentionLaunchArgs{
		QueryPointer:  1,
		KeyPointer:    2,
		ValuePointer:  3,
		OutputPointer: 4,
		WeightPointer: 5,
		Dim:           2,
		TokenCount:    1,
		QueryBytes:    4,
		KeyBytes:      8,
		ValueBytes:    8,
		OutputBytes:   8,
		WeightBytes:   4,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "query byte count")

	_, err = (hipAttentionLaunchArgs{
		QueryPointer:      1,
		OutputPointer:     4,
		WeightPointer:     5,
		Dim:               2,
		TokenCount:        1,
		QueryBytes:        8,
		OutputBytes:       8,
		WeightBytes:       4,
		KVSource:          hipAttentionKVSourceDevice,
		DescriptorPointer: 0,
		DescriptorBytes:   rocmDeviceKVDescriptorHeaderBytes,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "device KV descriptor")
}

func TestHIPKernels_VectorAddLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipVectorAddRequest{Left: []float32{1, -2, 0.5}, Right: []float32{4, 3, -0.25}}
	buffers, err := req.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.AssertNoError(t, err)
	launchBytes, err := launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipVectorAddLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipVectorAddLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipVectorAddLaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(buffers.Left.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(buffers.Right.Pointer()), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(launchBytes[24:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(launchBytes[32:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[36:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[40:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[44:]))

	config, err := hipOneDimensionalLaunchConfig(hipKernelNameVectorAdd, launchBytes, buffers.Count)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	output, err := buffers.ReadOutput()
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{5, 1, 0.25}, output, 0.0001)

	runnerOutput, err := hipRunVectorAddKernel(context.Background(), &fakeHIPDriver{available: true}, req)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{5, 1, 0.25}, runnerOutput, 0.0001)
}

func TestHIPKernels_VectorAddLaunchArgs_Bad(t *testing.T) {
	_, err := (hipVectorAddRequest{Left: []float32{1}, Right: []float32{1, 2}}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "length")

	buffers, err := (hipVectorAddRequest{Left: []float32{1}, Right: []float32{2}}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer buffers.Close()
	_, err = (hipVectorAddRequest{Left: []float32{1, 2}, Right: []float32{3, 4}}).launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	_, err = (hipVectorAddLaunchArgs{
		LeftPointer:   1,
		RightPointer:  2,
		OutputPointer: 3,
		Count:         2,
		LeftBytes:     4,
		RightBytes:    8,
		OutputBytes:   8,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "left byte count")
}

func TestHIPKernels_VectorScaleLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipVectorScaleRequest{Input: []float32{1, -2, 0.5}, Scale: 4}
	buffers, err := req.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.AssertNoError(t, err)
	launchBytes, err := launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipVectorScaleLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipVectorScaleLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipVectorScaleLaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(buffers.Input.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(launchBytes[24:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[28:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[32:]))
	core.AssertEqual(t, math.Float32bits(4), binary.LittleEndian.Uint32(launchBytes[36:]))

	config, err := hipOneDimensionalLaunchConfig(hipKernelNameVectorScale, launchBytes, buffers.Count)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	output, err := buffers.ReadOutput()
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{4, -8, 2}, output, 0.0001)

	runnerOutput, err := hipRunVectorScaleKernel(context.Background(), &fakeHIPDriver{available: true}, req)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{4, -8, 2}, runnerOutput, 0.0001)
}

func TestHIPKernels_VectorScaleLaunchArgs_Bad(t *testing.T) {
	_, err := (hipVectorScaleRequest{}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input")

	_, err = (hipVectorScaleRequest{Input: []float32{1}, Scale: float32(math.Inf(1))}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	buffers, err := (hipVectorScaleRequest{Input: []float32{1}, Scale: 2}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer buffers.Close()
	_, err = (hipVectorScaleRequest{Input: []float32{1, 2}, Scale: 2}).launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	_, err = (hipVectorScaleLaunchArgs{
		InputPointer:  1,
		OutputPointer: 2,
		Count:         2,
		InputBytes:    4,
		OutputBytes:   8,
		Scale:         1,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input byte count")
}

func TestHIPKernels_SwiGLULaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipSwiGLURequest{Gate: []float32{0, 1, -1}, Up: []float32{2, 4, 8}}
	buffers, err := req.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.AssertNoError(t, err)
	launchBytes, err := launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipSwiGLULaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipSwiGLULaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipSwiGLULaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(buffers.Gate.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(buffers.Up.Pointer()), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(launchBytes[24:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(launchBytes[32:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[36:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[40:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[44:]))

	config, err := hipOneDimensionalLaunchConfig(hipKernelNameSwiGLU, launchBytes, buffers.Count)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	output, err := buffers.ReadOutput()
	core.AssertNoError(t, err)
	want := []float32{
		0,
		1 / (1 + float32(math.Exp(-1))) * 4,
		-1 / (1 + float32(math.Exp(1))) * 8,
	}
	assertFloat32SlicesNear(t, want, output, 0.0001)

	runnerOutput, err := hipRunSwiGLUKernel(context.Background(), &fakeHIPDriver{available: true}, req)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, want, runnerOutput, 0.0001)
}

func TestHIPKernels_SwiGLULaunchArgs_Bad(t *testing.T) {
	_, err := (hipSwiGLURequest{Gate: []float32{1}, Up: []float32{1, 2}}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "length")

	buffers, err := (hipSwiGLURequest{Gate: []float32{1}, Up: []float32{2}}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer buffers.Close()
	_, err = (hipSwiGLURequest{Gate: []float32{1, 2}, Up: []float32{3, 4}}).launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	_, err = (hipSwiGLULaunchArgs{
		GatePointer:   1,
		UpPointer:     2,
		OutputPointer: 3,
		Count:         2,
		GateBytes:     4,
		UpBytes:       8,
		OutputBytes:   8,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "gate byte count")
}

func TestHIPKernels_GELUTanhMultiplyLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	req := hipGELUTanhMultiplyRequest{Gate: []float32{-1, 0, 1}, Up: []float32{2, 4, 8}}
	buffers, err := req.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.AssertNoError(t, err)
	launchBytes, err := launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipGELUTanhMulLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipGELUTanhMulLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipGELUTanhMulLaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(buffers.Gate.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(buffers.Up.Pointer()), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint64(buffers.Output.Pointer()), binary.LittleEndian.Uint64(launchBytes[24:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(launchBytes[32:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[36:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[40:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[44:]))

	config, err := hipOneDimensionalLaunchConfig(hipKernelNameGELUTanhMul, launchBytes, buffers.Count)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	output, err := buffers.ReadOutput()
	core.AssertNoError(t, err)
	want := []float32{-0.1588 * 2, 0, 0.8412 * 8}
	assertFloat32SlicesNear(t, want, output, 0.0005)

	runnerOutput, err := hipRunGELUTanhMultiplyKernel(context.Background(), &fakeHIPDriver{available: true}, req)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, want, runnerOutput, 0.0005)
}

func TestHIPKernels_GELUTanhMultiplyLaunchArgs_Bad(t *testing.T) {
	_, err := (hipGELUTanhMultiplyRequest{Gate: []float32{1}, Up: []float32{1, 2}}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "length")

	buffers, err := (hipGELUTanhMultiplyRequest{Gate: []float32{1}, Up: []float32{2}}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer buffers.Close()
	_, err = (hipGELUTanhMultiplyRequest{Gate: []float32{1, 2}, Up: []float32{3, 4}}).launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	_, err = (hipGELUTanhMultiplyLaunchArgs{
		GatePointer:   1,
		UpPointer:     2,
		OutputPointer: 3,
		Count:         2,
		GateBytes:     4,
		UpBytes:       8,
		OutputBytes:   8,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "gate byte count")
}

func TestHIPKernels_TransformerPrimitiveReadOutputValidation_Bad(t *testing.T) {
	_, err := (*hipRMSNormDeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rms norm output buffer is required")

	rmsReq := hipRMSNormRequest{Input: []float32{3, 4}, Weight: []float32{1, 0.5}}
	driver := &fakeHIPDriver{available: true}
	rmsBuffers, err := rmsReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer rmsBuffers.Close()
	rmsBuffers.Output.sizeBytes++
	_, err = rmsBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rms norm output byte count mismatch")

	driver = &fakeHIPDriver{available: true}
	rmsBuffers, err = rmsReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer rmsBuffers.Close()
	payload, err := hipFloat32Payload([]float32{1, float32(math.NaN())})
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(rmsBuffers.Output.Pointer(), payload))
	_, err = rmsBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	ropeReq := hipRoPERequest{Input: []float32{1, 0}, Position: 1, Base: 1}
	driver = &fakeHIPDriver{available: true}
	ropeBuffers, err := ropeReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer ropeBuffers.Close()
	ropeBuffers.Output.sizeBytes++
	_, err = ropeBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rope output byte count mismatch")

	greedyReq := hipGreedySampleRequest{Logits: []float32{1, 2}}
	driver = &fakeHIPDriver{available: true}
	greedyBuffers, err := greedyReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer greedyBuffers.Close()
	core.RequireNoError(t, driver.CopyHostToDevice(greedyBuffers.Output.Pointer(), hipGreedyResultPayloadForTest(2, 2)))
	_, err = greedyBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token ID out of range")

	driver = &fakeHIPDriver{available: true}
	greedyBuffers, err = greedyReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer greedyBuffers.Close()
	core.RequireNoError(t, driver.CopyHostToDevice(greedyBuffers.Output.Pointer(), hipGreedyResultPayloadForTest(1, float32(math.Inf(1)))))
	_, err = greedyBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "score must be finite")

	attentionReq := hipAttentionRequest{
		Query:  []float32{1, 0},
		Keys:   []float32{1, 0, 0, 1},
		Values: []float32{2, 0, 0, 4},
	}
	driver = &fakeHIPDriver{available: true}
	attentionBuffers, err := attentionReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer attentionBuffers.Close()
	attentionBuffers.Output.sizeBytes++
	_, err = attentionBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "attention output byte count mismatch")

	driver = &fakeHIPDriver{available: true}
	attentionBuffers, err = attentionReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer attentionBuffers.Close()
	payload, err = hipFloat32Payload([]float32{1.25, -0.25})
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(attentionBuffers.Weights.Pointer(), payload))
	_, err = attentionBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "attention weights must be probabilities")

	driver = &fakeHIPDriver{available: true}
	attentionBuffers, err = attentionReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer attentionBuffers.Close()
	driver.copyErr = core.NewError("copy failed")
	_, err = attentionBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy attention output")
}

func hipGreedyResultPayloadForTest(tokenID int32, score float32) []byte {
	payload := make([]byte, hipGreedyResultBytes)
	binary.LittleEndian.PutUint32(payload[0:], uint32(tokenID))
	binary.LittleEndian.PutUint32(payload[4:], math.Float32bits(score))
	return payload
}

func TestHIPKernels_TinyPrefillLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	fixture := hipReferenceTinyLMFixture()
	req := hipTinyPrefillRequest{
		TokenIDs:       []int32{0, 1},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}
	buffers, err := req.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.AssertNoError(t, err)
	launchBytes, err := launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipTinyPrefillLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipTinyPrefillLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipTinyPrefillLaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(buffers.Tokens.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(buffers.EmbeddingTable.Pointer()), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint64(buffers.OutputWeights.Pointer()), binary.LittleEndian.Uint64(launchBytes[24:]))
	core.AssertEqual(t, uint64(buffers.Logits.Pointer()), binary.LittleEndian.Uint64(launchBytes[32:]))
	core.AssertEqual(t, uint64(buffers.Attention.Pointer()), binary.LittleEndian.Uint64(launchBytes[40:]))
	core.AssertEqual(t, uint64(buffers.Result.Pointer()), binary.LittleEndian.Uint64(launchBytes[48:]))
	core.AssertEqual(t, uint64(buffers.Keys.Pointer()), binary.LittleEndian.Uint64(launchBytes[56:]))
	core.AssertEqual(t, uint64(buffers.Values.Pointer()), binary.LittleEndian.Uint64(launchBytes[64:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[72:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(launchBytes[76:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[80:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[84:]))
	core.AssertEqual(t, uint32(24), binary.LittleEndian.Uint32(launchBytes[88:]))
	core.AssertEqual(t, uint32(24), binary.LittleEndian.Uint32(launchBytes[92:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[96:]))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(launchBytes[100:]))
	core.AssertEqual(t, uint32(hipGreedyResultBytes), binary.LittleEndian.Uint32(launchBytes[104:]))
	core.AssertEqual(t, uint32(16), binary.LittleEndian.Uint32(launchBytes[108:]))
	core.AssertEqual(t, uint32(16), binary.LittleEndian.Uint32(launchBytes[112:]))
	core.AssertEqual(t, hipTinyOutputWeightEncodingFP32, binary.LittleEndian.Uint32(launchBytes[116:]))

	config, err := hipOneDimensionalLaunchConfig(hipKernelNameTinyPrefill, launchBytes, 1)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	output, err := buffers.ReadOutput()
	core.AssertNoError(t, err)
	core.AssertEqual(t, 2, output.NextTokenID)
	assertFloat32Near(t, 1, output.NextScore)
	assertFloat32SlicesNear(t, []float32{0.3302, 0.6698, 1}, output.Logits, 0.0001)
	assertFloat32SlicesNear(t, []float32{0.3302, 0.6698}, output.Attention, 0.0001)
	assertFloat32SlicesNear(t, []float32{1, 0, 0, 1}, output.StateKeys, 0.0001)
	assertFloat32SlicesNear(t, []float32{1, 0, 0, 1}, output.StateValues, 0.0001)

	for _, tt := range []struct {
		name       string
		fp16       []uint16
		q8         []int8
		q8Scale    float32
		encoding   uint32
		weightByte uint32
	}{{
		name:       "fp16",
		fp16:       hipTinyOutputWeightsFP16Fixture(),
		encoding:   hipTinyOutputWeightEncodingFP16,
		weightByte: 12,
	}, {
		name:       "q8",
		q8:         hipTinyOutputWeightsQ8Fixture(),
		q8Scale:    0.5,
		encoding:   hipTinyOutputWeightEncodingQ8,
		weightByte: 6,
	}} {
		t.Run(tt.name, func(t *testing.T) {
			variantReq := hipTinyPrefillRequest{
				TokenIDs:       []int32{0, 1},
				EmbeddingTable: fixture.EmbeddingTable,
				OutputFP16:     tt.fp16,
				OutputQ8:       tt.q8,
				Q8Scale:        tt.q8Scale,
				VocabSize:      fixture.VocabSize,
				HiddenSize:     fixture.HiddenSize,
			}
			variantDriver := &fakeHIPDriver{available: true}
			variantBuffers, err := variantReq.deviceBuffers(variantDriver)
			core.RequireNoError(t, err)
			defer variantBuffers.Close()
			variantLaunch, err := variantReq.launchArgs(variantBuffers)
			core.RequireNoError(t, err)
			variantLaunchBytes, err := variantLaunch.Binary()
			core.RequireNoError(t, err)
			core.AssertEqual(t, tt.weightByte, binary.LittleEndian.Uint32(variantLaunchBytes[92:]))
			core.AssertEqual(t, tt.encoding, binary.LittleEndian.Uint32(variantLaunchBytes[116:]))
			core.AssertEqual(t, math.Float32bits(tt.q8Scale), binary.LittleEndian.Uint32(variantLaunchBytes[120:]))
			variantConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameTinyPrefill, variantLaunchBytes, 1)
			core.RequireNoError(t, err)
			core.RequireNoError(t, hipLaunchKernel(variantDriver, variantConfig))
			variantOutput, err := variantBuffers.ReadOutput()
			core.RequireNoError(t, err)
			core.AssertEqual(t, 2, variantOutput.NextTokenID)
			assertFloat32Near(t, 1, variantOutput.NextScore)
			assertFloat32SlicesNear(t, []float32{0.3302, 0.6698, 1}, variantOutput.Logits, 0.0001)
			assertFloat32SlicesNear(t, []float32{0.3302, 0.6698}, variantOutput.Attention, 0.0001)
			assertFloat32SlicesNear(t, []float32{1, 0, 0, 1}, variantOutput.StateKeys, 0.0001)
			assertFloat32SlicesNear(t, []float32{1, 0, 0, 1}, variantOutput.StateValues, 0.0001)
		})
	}
}

func TestHIPKernels_TinyPrefillLaunchArgs_Bad(t *testing.T) {
	fixture := hipReferenceTinyLMFixture()
	_, err := (hipTinyPrefillRequest{
		TokenIDs:       []int32{0},
		EmbeddingTable: fixture.EmbeddingTable,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "exactly one output weight encoding")

	_, err = (hipTinyPrefillRequest{
		TokenIDs:       []int32{0},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		OutputFP16:     hipTinyOutputWeightsFP16Fixture(),
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "exactly one output weight encoding")

	_, err = (hipTinyPrefillRequest{
		TokenIDs:       []int32{99},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside vocabulary")

	_, err = (hipTinyPrefillRequest{
		TokenIDs:       []int32{0},
		EmbeddingTable: []float32{1},
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "embedding table length")

	_, err = (hipTinyPrefillRequest{
		TokenIDs:       []int32{0},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights[:len(fixture.OutputWeights)-1],
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "output weight length")

	_, err = (hipTinyPrefillRequest{
		TokenIDs:       []int32{0},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputQ8:       hipTinyOutputWeightsQ8Fixture(),
		Q8Scale:        float32(math.Inf(1)),
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "q8 scale must be positive and finite")

	req := hipTinyPrefillRequest{
		TokenIDs:       []int32{0},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}
	buffers, err := req.deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer buffers.Close()
	_, err = (hipTinyPrefillRequest{
		TokenIDs:       []int32{0, 1},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}).launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	_, err = (hipTinyPrefillLaunchArgs{
		TokenPointer:         1,
		EmbeddingPointer:     2,
		OutputWeightPointer:  3,
		LogitPointer:         4,
		AttentionPointer:     5,
		ResultPointer:        6,
		KeyPointer:           7,
		ValuePointer:         8,
		TokenCount:           2,
		VocabSize:            3,
		HiddenSize:           2,
		TokenBytes:           4,
		EmbeddingBytes:       24,
		OutputWeightBytes:    24,
		LogitBytes:           12,
		AttentionBytes:       8,
		ResultBytes:          hipGreedyResultBytes,
		KeyBytes:             16,
		ValueBytes:           16,
		OutputWeightEncoding: hipTinyOutputWeightEncodingFP32,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token byte count")

	_, err = (hipTinyPrefillLaunchArgs{
		TokenPointer:         1,
		EmbeddingPointer:     2,
		OutputWeightPointer:  3,
		LogitPointer:         4,
		AttentionPointer:     5,
		ResultPointer:        6,
		KeyPointer:           7,
		ValuePointer:         8,
		TokenCount:           1,
		VocabSize:            3,
		HiddenSize:           2,
		TokenBytes:           4,
		EmbeddingBytes:       24,
		OutputWeightBytes:    24,
		LogitBytes:           12,
		AttentionBytes:       4,
		ResultBytes:          hipGreedyResultBytes,
		KeyBytes:             8,
		ValueBytes:           8,
		OutputWeightEncoding: hipTinyOutputWeightEncodingJANGTQ,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported output weight encoding")

	_, err = (hipTinyPrefillLaunchArgs{
		TokenPointer:         1,
		EmbeddingPointer:     2,
		OutputWeightPointer:  3,
		LogitPointer:         4,
		AttentionPointer:     5,
		ResultPointer:        6,
		KeyPointer:           7,
		ValuePointer:         8,
		TokenCount:           1,
		VocabSize:            3,
		HiddenSize:           2,
		TokenBytes:           4,
		EmbeddingBytes:       24,
		OutputWeightBytes:    6,
		LogitBytes:           12,
		AttentionBytes:       4,
		ResultBytes:          hipGreedyResultBytes,
		KeyBytes:             8,
		ValueBytes:           8,
		OutputWeightEncoding: hipTinyOutputWeightEncodingQ8,
		Q8Scale:              -1,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "q8 scale must be positive and finite")
}

func TestHIPKernels_TinyDecodeLaunchArgs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	fixture := hipReferenceTinyLMFixture()
	prefill, err := hipReferenceTinyPrefill(fixture, []int32{0, 1})
	core.RequireNoError(t, err)
	req := hipTinyDecodeRequest{
		TokenID:        2,
		PriorKeys:      flattenHIPReferenceMatrix(prefill.State.Keys),
		PriorValues:    flattenHIPReferenceMatrix(prefill.State.Values),
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}
	buffers, err := req.deviceBuffers(driver)
	core.AssertNoError(t, err)
	defer buffers.Close()

	launch, err := req.launchArgs(buffers)
	core.AssertNoError(t, err)
	launchBytes, err := launch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipTinyDecodeLaunchArgsBytes, len(launchBytes))
	core.AssertEqual(t, hipTinyDecodeLaunchArgsVersion, binary.LittleEndian.Uint32(launchBytes[0:]))
	core.AssertEqual(t, uint32(hipTinyDecodeLaunchArgsBytes), binary.LittleEndian.Uint32(launchBytes[4:]))
	core.AssertEqual(t, uint64(buffers.PriorKeys.Pointer()), binary.LittleEndian.Uint64(launchBytes[8:]))
	core.AssertEqual(t, uint64(buffers.PriorValues.Pointer()), binary.LittleEndian.Uint64(launchBytes[16:]))
	core.AssertEqual(t, uint64(buffers.EmbeddingTable.Pointer()), binary.LittleEndian.Uint64(launchBytes[24:]))
	core.AssertEqual(t, uint64(buffers.OutputWeights.Pointer()), binary.LittleEndian.Uint64(launchBytes[32:]))
	core.AssertEqual(t, uint64(buffers.Logits.Pointer()), binary.LittleEndian.Uint64(launchBytes[40:]))
	core.AssertEqual(t, uint64(buffers.Attention.Pointer()), binary.LittleEndian.Uint64(launchBytes[48:]))
	core.AssertEqual(t, uint64(buffers.UpdatedKeys.Pointer()), binary.LittleEndian.Uint64(launchBytes[56:]))
	core.AssertEqual(t, uint64(buffers.UpdatedValues.Pointer()), binary.LittleEndian.Uint64(launchBytes[64:]))
	core.AssertEqual(t, uint64(buffers.Result.Pointer()), binary.LittleEndian.Uint64(launchBytes[72:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[80:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[84:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(launchBytes[88:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(launchBytes[92:]))
	core.AssertEqual(t, uint32(16), binary.LittleEndian.Uint32(launchBytes[96:]))
	core.AssertEqual(t, uint32(16), binary.LittleEndian.Uint32(launchBytes[100:]))
	core.AssertEqual(t, uint32(24), binary.LittleEndian.Uint32(launchBytes[104:]))
	core.AssertEqual(t, uint32(24), binary.LittleEndian.Uint32(launchBytes[108:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[112:]))
	core.AssertEqual(t, uint32(12), binary.LittleEndian.Uint32(launchBytes[116:]))
	core.AssertEqual(t, uint32(24), binary.LittleEndian.Uint32(launchBytes[120:]))
	core.AssertEqual(t, uint32(24), binary.LittleEndian.Uint32(launchBytes[124:]))
	core.AssertEqual(t, uint32(hipGreedyResultBytes), binary.LittleEndian.Uint32(launchBytes[128:]))
	core.AssertEqual(t, hipTinyOutputWeightEncodingFP32, binary.LittleEndian.Uint32(launchBytes[132:]))

	config, err := hipOneDimensionalLaunchConfig(hipKernelNameTinyDecode, launchBytes, 1)
	core.AssertNoError(t, err)
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	output, err := buffers.ReadOutput()
	core.AssertNoError(t, err)
	core.AssertEqual(t, 2, output.NextTokenID)
	assertFloat32Near(t, 1.5035, output.NextScore)
	assertFloat32SlicesNear(t, []float32{0.7517, 0.7517, 1.5035}, output.Logits, 0.0001)
	assertFloat32SlicesNear(t, []float32{0.2483, 0.2483, 0.5035}, output.Attention, 0.0001)
	assertFloat32SlicesNear(t, []float32{1, 0, 0, 1, 1, 1}, output.UpdatedKeys, 0.0001)
	assertFloat32SlicesNear(t, []float32{1, 0, 0, 1, 1, 1}, output.UpdatedValues, 0.0001)

	for _, tt := range []struct {
		name       string
		fp16       []uint16
		q8         []int8
		q8Scale    float32
		encoding   uint32
		weightByte uint32
	}{{
		name:       "fp16",
		fp16:       hipTinyOutputWeightsFP16Fixture(),
		encoding:   hipTinyOutputWeightEncodingFP16,
		weightByte: 12,
	}, {
		name:       "q8",
		q8:         hipTinyOutputWeightsQ8Fixture(),
		q8Scale:    0.5,
		encoding:   hipTinyOutputWeightEncodingQ8,
		weightByte: 6,
	}} {
		t.Run(tt.name, func(t *testing.T) {
			variantReq := hipTinyDecodeRequest{
				TokenID:        2,
				PriorKeys:      flattenHIPReferenceMatrix(prefill.State.Keys),
				PriorValues:    flattenHIPReferenceMatrix(prefill.State.Values),
				EmbeddingTable: fixture.EmbeddingTable,
				OutputFP16:     tt.fp16,
				OutputQ8:       tt.q8,
				Q8Scale:        tt.q8Scale,
				VocabSize:      fixture.VocabSize,
				HiddenSize:     fixture.HiddenSize,
			}
			variantDriver := &fakeHIPDriver{available: true}
			variantBuffers, err := variantReq.deviceBuffers(variantDriver)
			core.RequireNoError(t, err)
			defer variantBuffers.Close()
			variantLaunch, err := variantReq.launchArgs(variantBuffers)
			core.RequireNoError(t, err)
			variantLaunchBytes, err := variantLaunch.Binary()
			core.RequireNoError(t, err)
			core.AssertEqual(t, tt.weightByte, binary.LittleEndian.Uint32(variantLaunchBytes[108:]))
			core.AssertEqual(t, tt.encoding, binary.LittleEndian.Uint32(variantLaunchBytes[132:]))
			core.AssertEqual(t, math.Float32bits(tt.q8Scale), binary.LittleEndian.Uint32(variantLaunchBytes[136:]))
			variantConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameTinyDecode, variantLaunchBytes, 1)
			core.RequireNoError(t, err)
			core.RequireNoError(t, hipLaunchKernel(variantDriver, variantConfig))
			variantOutput, err := variantBuffers.ReadOutput()
			core.RequireNoError(t, err)
			core.AssertEqual(t, 2, variantOutput.NextTokenID)
			assertFloat32Near(t, 1.5035, variantOutput.NextScore)
			assertFloat32SlicesNear(t, []float32{0.7517, 0.7517, 1.5035}, variantOutput.Logits, 0.0001)
			assertFloat32SlicesNear(t, []float32{0.2483, 0.2483, 0.5035}, variantOutput.Attention, 0.0001)
			assertFloat32SlicesNear(t, []float32{1, 0, 0, 1, 1, 1}, variantOutput.UpdatedKeys, 0.0001)
			assertFloat32SlicesNear(t, []float32{1, 0, 0, 1, 1, 1}, variantOutput.UpdatedValues, 0.0001)
		})
	}
}

func TestHIPKernels_TinyDecodeLaunchArgs_Bad(t *testing.T) {
	fixture := hipReferenceTinyLMFixture()
	_, err := (hipTinyDecodeRequest{
		TokenID:        -1,
		PriorKeys:      []float32{1, 0},
		PriorValues:    []float32{1, 0},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token ID must be non-negative")

	_, err = (hipTinyDecodeRequest{
		TokenID:        3,
		PriorKeys:      []float32{1, 0},
		PriorValues:    []float32{1, 0},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside vocabulary")

	_, err = (hipTinyDecodeRequest{
		TokenID:        0,
		PriorKeys:      []float32{1, 0},
		PriorValues:    []float32{1, 0, 1, 0},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "lengths must match")

	_, err = (hipTinyDecodeRequest{
		TokenID:        0,
		PriorKeys:      []float32{1, 0, 1},
		PriorValues:    []float32{1, 0, 1},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "align with hidden size")

	_, err = (hipTinyDecodeRequest{
		TokenID:        0,
		PriorKeys:      []float32{1, 0},
		PriorValues:    []float32{1, 0},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputQ8:       hipTinyOutputWeightsQ8Fixture(),
		Q8Scale:        float32(math.NaN()),
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}).deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "q8 scale must be positive and finite")

	req := hipTinyDecodeRequest{
		TokenID:        0,
		PriorKeys:      []float32{1, 0},
		PriorValues:    []float32{1, 0},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}
	buffers, err := req.deviceBuffers(&fakeHIPDriver{available: true})
	core.AssertNoError(t, err)
	defer buffers.Close()
	_, err = (hipTinyDecodeRequest{
		TokenID:        0,
		PriorKeys:      []float32{1, 0, 0, 1},
		PriorValues:    []float32{1, 0, 0, 1},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}).launchArgs(buffers)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")

	_, err = (hipTinyDecodeLaunchArgs{
		PriorKeyPointer:     1,
		PriorValuePointer:   2,
		EmbeddingPointer:    3,
		OutputWeightPointer: 4,
		LogitPointer:        5,
		AttentionPointer:    6,
		UpdatedKeyPointer:   7,
		UpdatedValuePointer: 8,
		ResultPointer:       9,
		TokenID:             0,
		PriorTokenCount:     1,
		VocabSize:           3,
		HiddenSize:          2,
		PriorKeyBytes:       4,
		PriorValueBytes:     8,
		EmbeddingBytes:      24,
		OutputWeightBytes:   24,
		LogitBytes:          12,
		AttentionBytes:      8,
		UpdatedKeyBytes:     16,
		UpdatedValueBytes:   16,
		ResultBytes:         hipGreedyResultBytes,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prior key byte count")

	_, err = (hipTinyDecodeLaunchArgs{
		PriorKeyPointer:      1,
		PriorValuePointer:    2,
		EmbeddingPointer:     3,
		OutputWeightPointer:  4,
		LogitPointer:         5,
		AttentionPointer:     6,
		UpdatedKeyPointer:    7,
		UpdatedValuePointer:  8,
		ResultPointer:        9,
		TokenID:              0,
		PriorTokenCount:      1,
		VocabSize:            3,
		HiddenSize:           2,
		PriorKeyBytes:        8,
		PriorValueBytes:      8,
		EmbeddingBytes:       24,
		OutputWeightBytes:    24,
		LogitBytes:           12,
		AttentionBytes:       8,
		UpdatedKeyBytes:      16,
		UpdatedValueBytes:    16,
		ResultBytes:          hipGreedyResultBytes,
		OutputWeightEncoding: hipTinyOutputWeightEncodingCodebook,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported output weight encoding")

	_, err = (hipTinyDecodeLaunchArgs{
		PriorKeyPointer:      1,
		PriorValuePointer:    2,
		EmbeddingPointer:     3,
		OutputWeightPointer:  4,
		LogitPointer:         5,
		AttentionPointer:     6,
		UpdatedKeyPointer:    7,
		UpdatedValuePointer:  8,
		ResultPointer:        9,
		TokenID:              0,
		PriorTokenCount:      1,
		VocabSize:            3,
		HiddenSize:           2,
		PriorKeyBytes:        8,
		PriorValueBytes:      8,
		EmbeddingBytes:       24,
		OutputWeightBytes:    6,
		LogitBytes:           12,
		AttentionBytes:       8,
		UpdatedKeyBytes:      16,
		UpdatedValueBytes:    16,
		ResultBytes:          hipGreedyResultBytes,
		OutputWeightEncoding: hipTinyOutputWeightEncodingQ8,
		Q8Scale:              0,
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "q8 scale must be positive and finite")
}

func TestHIPKernels_TinyReadOutputValidation_Bad(t *testing.T) {
	_, err := (*hipTinyPrefillDeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tiny prefill output buffers are required")

	fixture := hipReferenceTinyLMFixture()
	prefillReq := hipTinyPrefillRequest{
		TokenIDs:       []int32{0, 1},
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}
	driver := &fakeHIPDriver{available: true}
	prefillBuffers, err := prefillReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer prefillBuffers.Close()
	prefillBuffers.Logits.sizeBytes++
	_, err = prefillBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tiny prefill logits byte count mismatch")

	driver = &fakeHIPDriver{available: true}
	prefillBuffers, err = prefillReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer prefillBuffers.Close()
	payload, err := hipFloat32Payload([]float32{0, float32(math.NaN()), 1})
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(prefillBuffers.Logits.Pointer(), payload))
	_, err = prefillBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tiny prefill logits values must be finite")

	driver = &fakeHIPDriver{available: true}
	prefillBuffers, err = prefillReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer prefillBuffers.Close()
	payload, err = hipFloat32Payload([]float32{0.5, 1.5})
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(prefillBuffers.Attention.Pointer(), payload))
	_, err = prefillBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tiny prefill attention must be probabilities")

	driver = &fakeHIPDriver{available: true}
	prefillBuffers, err = prefillReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer prefillBuffers.Close()
	core.RequireNoError(t, driver.CopyHostToDevice(prefillBuffers.Result.Pointer(), hipGreedyResultPayloadForTest(int32(fixture.VocabSize), 1)))
	_, err = prefillBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tiny prefill result token ID out of range")

	_, err = (*hipTinyDecodeDeviceBuffers)(nil).ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tiny decode output buffers are required")

	prefill, err := hipReferenceTinyPrefill(fixture, []int32{0, 1})
	core.RequireNoError(t, err)
	decodeReq := hipTinyDecodeRequest{
		TokenID:        2,
		PriorKeys:      flattenHIPReferenceMatrix(prefill.State.Keys),
		PriorValues:    flattenHIPReferenceMatrix(prefill.State.Values),
		EmbeddingTable: fixture.EmbeddingTable,
		OutputWeights:  fixture.OutputWeights,
		VocabSize:      fixture.VocabSize,
		HiddenSize:     fixture.HiddenSize,
	}
	driver = &fakeHIPDriver{available: true}
	decodeBuffers, err := decodeReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer decodeBuffers.Close()
	decodeBuffers.UpdatedValues.sizeBytes++
	_, err = decodeBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tiny decode updated values byte count mismatch")

	driver = &fakeHIPDriver{available: true}
	decodeBuffers, err = decodeReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer decodeBuffers.Close()
	payload, err = hipFloat32Payload([]float32{0.25, 0.25, 1.25})
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(decodeBuffers.Attention.Pointer(), payload))
	_, err = decodeBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tiny decode attention must be probabilities")

	driver = &fakeHIPDriver{available: true}
	decodeBuffers, err = decodeReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer decodeBuffers.Close()
	core.RequireNoError(t, driver.CopyHostToDevice(decodeBuffers.Result.Pointer(), hipGreedyResultPayloadForTest(1, float32(math.Inf(1)))))
	_, err = decodeBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tiny decode result score must be finite")

	driver = &fakeHIPDriver{available: true}
	decodeBuffers, err = decodeReq.deviceBuffers(driver)
	core.RequireNoError(t, err)
	defer decodeBuffers.Close()
	driver.copyErr = core.NewError("copy failed")
	_, err = decodeBuffers.ReadOutput()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy tiny decode logits")
}

func TestHIPKernels_TinyOutputWeightValues_Bad(t *testing.T) {
	_, err := hipTinyOutputWeightValues([]byte{0x00}, hipTinyOutputWeightEncodingFP16, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "fp16 payload byte length")

	_, err = hipTinyOutputWeightValues([]byte{0x00, 0x7e}, hipTinyOutputWeightEncodingFP16, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "output weight values must be finite")

	payload, err := hipFloat32Payload([]float32{float32(math.NaN())})
	core.RequireNoError(t, err)
	_, err = hipTinyOutputWeightValues(payload, hipTinyOutputWeightEncodingFP32, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "output weight values must be finite")

	_, err = hipTinyOutputWeightValues(nil, hipTinyOutputWeightEncodingQ8, 1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "q8 payload is empty")

	_, err = hipTinyOutputWeightValues([]byte{1}, hipTinyOutputWeightEncodingQ8, float32(math.NaN()))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "q8 scale must be positive and finite")

	_, err = hipTinyOutputWeightValues([]byte{1}, hipTinyOutputWeightEncodingJANGTQ, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported output weight encoding")
}

func TestHIPKernels_KernelLaunchConfig_GoodBad(t *testing.T) {
	config, err := hipOneDimensionalLaunchConfig("test_kernel", []byte{1, 2, 3}, 65)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "test_kernel", config.Name)
	core.AssertEqual(t, uint32(2), config.GridX)
	core.AssertEqual(t, uint32(64), config.BlockX)

	driver := &fakeHIPDriver{available: true}
	core.AssertNoError(t, hipLaunchKernel(driver, config))
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, "test_kernel", driver.launches[0].Name)
	core.AssertEqual(t, 3, len(driver.launches[0].Args))

	_, err = hipOneDimensionalLaunchConfig("", []byte{1}, 1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "kernel name")

	_, err = hipOneDimensionalLaunchConfig(hipKernelNameProjection, nil, 1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "launch args")

	_, err = hipOneDimensionalLaunchConfig(hipKernelNameProjection, []byte{1}, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "work items")

	err = hipLaunchKernel(&failingHIPDriver{available: true}, config)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "not linked")

	driver.launchErr = core.NewError("launch failed")
	err = hipLaunchKernel(driver, config)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "launch failed")

	prefillArgs, err := (hipPrefillLaunchArgs{
		TokenPointer: 999,
		TokenCount:   1,
		TokenBytes:   4,
		CacheMode:    rocmKVCacheModeFP16,
		ModeCode:     rocmDeviceKVDescriptorModeFP16,
		BlockSize:    defaultROCmKVBlockSize,
		KeyWidth:     1,
		ValueWidth:   1,
	}).Binary()
	core.AssertNoError(t, err)
	prefillConfig, err := hipOneDimensionalLaunchConfig(hipKernelNamePrefill, prefillArgs, 1)
	core.AssertNoError(t, err)
	err = hipLaunchKernel(&fakeHIPDriver{available: true}, prefillConfig)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prefill token buffer")

	decodeArgs, err := (hipDecodeLaunchArgs{
		TokenID:  1,
		Position: 1,
		KV: rocmDeviceKVLaunchDescriptor{
			DescriptorPointer: 999,
			DescriptorBytes:   uint64(rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVDescriptorPageBytes),
			DescriptorVersion: rocmDeviceKVDescriptorVersion,
			Mode:              rocmKVCacheModeFP16,
			ModeCode:          rocmDeviceKVDescriptorModeFP16,
			BlockSize:         defaultROCmKVBlockSize,
			PageCount:         1,
			TokenCount:        1,
			KeyWidth:          1,
			ValueWidth:        1,
		},
	}).Binary()
	core.AssertNoError(t, err)
	decodeConfig, err := hipOneDimensionalLaunchConfig(hipKernelNameDecode, decodeArgs, 1)
	core.AssertNoError(t, err)
	err = hipLaunchKernel(&fakeHIPDriver{available: true}, decodeConfig)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "decode descriptor table")
}

func TestHIPKernels_LoadedModelDispatchesKernelSet_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	model := &hipLoadedModel{driver: driver, kernels: fakeLinkedHIPKernelSet{tokens: []inference.Token{{ID: 7, Text: "ok"}}}}

	stream, streamErr := model.Generate(context.Background(), "hello", inference.DefaultGenerateConfig())
	var got []inference.Token
	for token := range stream {
		got = append(got, token)
	}

	core.AssertNoError(t, streamErr())
	core.AssertEqual(t, 1, len(got))
	core.AssertEqual(t, int32(7), got[0].ID)
	core.AssertEqual(t, hipKernelStatusLinked, model.KernelStatus().Decode)

	projected, err := model.Project(context.Background(), hipProjectionRequest{
		Input: []float32{1, 2},
		FP16:  []uint16{0x3c00, 0x4000},
		Rows:  1,
		Cols:  2,
	})
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{5}, projected, 0)
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameProjection, driver.launches[0].Name)
	core.AssertEqual(t, hipProjectionLaunchArgsBytes, len(driver.launches[0].Args))

	prefill, err := model.Prefill(context.Background(), hipPrefillRequest{
		TokenIDs:   []int32{1, 2, 3},
		CacheMode:  rocmKVCacheModeKQ8VQ4,
		KeyWidth:   2,
		ValueWidth: 3,
	})
	core.AssertNoError(t, err)
	core.AssertEqual(t, 3, prefill.PromptTokens)
	core.AssertEqual(t, "linked", prefill.Labels["prefill_kernel"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, prefill.Labels["kv_cache_mode"])
	core.AssertEqual(t, "2", prefill.Labels["kv_key_width"])
	core.AssertEqual(t, "3", prefill.Labels["kv_value_width"])
	core.AssertNotNil(t, prefill.DeviceKV)
	core.AssertNotNil(t, prefill.DescriptorTable)
	core.AssertEqual(t, "hip_device_mirror", prefill.Labels["kv_backing"])
	core.AssertEqual(t, "mirrored", prefill.Labels["kv_device_backing"])
	core.AssertEqual(t, "hip_device", prefill.Labels["kv_descriptor_table"])
	core.AssertEqual(t, "96", prefill.Labels["kv_descriptor_bytes"])
	core.AssertEqual(t, "64", prefill.Labels["prefill_launch_args_bytes"])
	core.AssertEqual(t, "12", prefill.Labels["prefill_token_bytes"])
	core.AssertEqual(t, "3", prefill.Labels["prefill_launch_tokens"])
	core.AssertEqual(t, 2, len(driver.launches))
	core.AssertEqual(t, hipKernelNamePrefill, driver.launches[1].Name)
	core.AssertEqual(t, hipPrefillLaunchArgsBytes, len(driver.launches[1].Args))
	prefillLaunch, err := (hipDecodeRequest{
		TokenID:         7,
		KV:              prefill.KV,
		DeviceKV:        prefill.DeviceKV,
		DescriptorTable: prefill.DescriptorTable,
	}).kvLaunchDescriptor()
	core.AssertNoError(t, err)
	core.AssertEqual(t, prefill.DescriptorTable.Pointer(), prefillLaunch.DescriptorPointer)
	core.AssertEqual(t, uint64(96), prefillLaunch.DescriptorBytes)
	core.AssertEqual(t, rocmDeviceKVDescriptorModeKQ8VQ4, prefillLaunch.ModeCode)
	core.AssertEqual(t, 3, prefillLaunch.TokenCount)
	core.AssertEqual(t, 1, prefillLaunch.PageCount)
	core.AssertEqual(t, 2, prefillLaunch.KeyWidth)
	core.AssertEqual(t, 3, prefillLaunch.ValueWidth)
	prefillLaunchBytes, err := prefillLaunch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, rocmDeviceKVLaunchDescriptorBytes, len(prefillLaunchBytes))
	core.AssertEqual(t, uint64(prefill.DescriptorTable.Pointer()), binary.LittleEndian.Uint64(prefillLaunchBytes[0:]))
	core.AssertEqual(t, uint64(96), binary.LittleEndian.Uint64(prefillLaunchBytes[8:]))
	core.AssertEqual(t, rocmDeviceKVDescriptorVersion, binary.LittleEndian.Uint32(prefillLaunchBytes[16:]))
	core.AssertEqual(t, rocmDeviceKVDescriptorModeKQ8VQ4, binary.LittleEndian.Uint32(prefillLaunchBytes[20:]))
	statusLaunch := prefillLaunch
	statusLaunch.StatusPointer = 4321
	statusLaunchBytes, err := statusLaunch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, uint64(4321), binary.LittleEndian.Uint64(statusLaunchBytes[48:]))
	core.AssertEqual(t, hipDecodeLaunchStatusOK, binary.LittleEndian.Uint32(statusLaunchBytes[56:]))
	prefillDecodeLaunchBytes, err := (hipDecodeRequest{
		TokenID:         7,
		KV:              prefill.KV,
		DeviceKV:        prefill.DeviceKV,
		DescriptorTable: prefill.DescriptorTable,
	}).decodeLaunchArgsBytes()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipDecodeLaunchArgsBytes, len(prefillDecodeLaunchBytes))
	core.AssertEqual(t, hipDecodeLaunchArgsVersion, binary.LittleEndian.Uint32(prefillDecodeLaunchBytes[0:]))
	core.AssertEqual(t, uint32(hipDecodeLaunchArgsHeaderBytes), binary.LittleEndian.Uint32(prefillDecodeLaunchBytes[4:]))
	core.AssertEqual(t, uint32(hipDecodeLaunchArgsBytes), binary.LittleEndian.Uint32(prefillDecodeLaunchBytes[8:]))
	core.AssertEqual(t, uint32(7), binary.LittleEndian.Uint32(prefillDecodeLaunchBytes[12:]))
	core.AssertEqual(t, uint64(3), binary.LittleEndian.Uint64(prefillDecodeLaunchBytes[16:]))
	core.AssertEqual(t, uint32(rocmDeviceKVLaunchDescriptorBytes), binary.LittleEndian.Uint32(prefillDecodeLaunchBytes[24:]))
	core.AssertEqual(t, uint64(prefill.DescriptorTable.Pointer()), binary.LittleEndian.Uint64(prefillDecodeLaunchBytes[hipDecodeLaunchArgsHeaderBytes:]))
	core.AssertEqual(t, uint64(96), binary.LittleEndian.Uint64(prefillDecodeLaunchBytes[hipDecodeLaunchArgsHeaderBytes+8:]))
	core.AssertEqual(t, uint64(3), binary.LittleEndian.Uint64(prefillDecodeLaunchBytes[hipDecodeLaunchArgsHeaderBytes+32:]))

	decoded, err := model.DecodeToken(context.Background(), hipDecodeRequest{
		TokenID:         7,
		KV:              prefill.KV,
		DeviceKV:        prefill.DeviceKV,
		DescriptorTable: prefill.DescriptorTable,
	})
	core.AssertNoError(t, err)
	if decoded.DeviceKV != nil {
		defer decoded.DeviceKV.Close()
	}
	if decoded.DescriptorTable != nil {
		defer decoded.DescriptorTable.Close()
	}
	core.AssertEqual(t, int32(7), decoded.Token.ID)
	core.AssertEqual(t, 4, decoded.KV.TokenCount())
	core.AssertNotNil(t, decoded.DeviceKV)
	core.AssertNotNil(t, decoded.DescriptorTable)
	core.AssertEqual(t, 4, decoded.DeviceKV.TokenCount())
	core.AssertEqual(t, "hip_device", decoded.Labels["kv_descriptor_table"])
	core.AssertEqual(t, "160", decoded.Labels["kv_descriptor_bytes"])
	core.AssertEqual(t, "ready", decoded.Labels["kv_launch_descriptor"])
	core.AssertEqual(t, "160", decoded.Labels["kv_launch_descriptor_bytes"])
	core.AssertEqual(t, "64", decoded.Labels["kv_launch_args_bytes"])
	core.AssertEqual(t, "4", decoded.Labels["kv_launch_tokens"])
	core.AssertEqual(t, "96", decoded.Labels["decode_launch_args_bytes"])
	core.AssertEqual(t, "7", decoded.Labels["decode_launch_token"])
	core.AssertEqual(t, "3", decoded.Labels["decode_launch_position"])
	core.AssertEqual(t, 3, len(driver.launches))
	core.AssertEqual(t, hipKernelNameDecode, driver.launches[2].Name)
	core.AssertEqual(t, hipDecodeLaunchArgsBytes, len(driver.launches[2].Args))
	decodedLaunch, err := (hipDecodeRequest{
		TokenID:         8,
		KV:              decoded.KV,
		DeviceKV:        decoded.DeviceKV,
		DescriptorTable: decoded.DescriptorTable,
	}).kvLaunchDescriptor()
	core.AssertNoError(t, err)
	core.AssertEqual(t, decoded.DescriptorTable.Pointer(), decodedLaunch.DescriptorPointer)
	core.AssertEqual(t, uint64(160), decodedLaunch.DescriptorBytes)
	core.AssertEqual(t, 4, decodedLaunch.TokenCount)
	core.AssertEqual(t, 2, decodedLaunch.PageCount)
	decodedLaunchBytes, err := (hipDecodeRequest{
		TokenID:         8,
		KV:              decoded.KV,
		DeviceKV:        decoded.DeviceKV,
		DescriptorTable: decoded.DescriptorTable,
	}).kvLaunchDescriptorBytes()
	core.AssertNoError(t, err)
	core.AssertEqual(t, rocmDeviceKVLaunchDescriptorBytes, len(decodedLaunchBytes))
	core.AssertEqual(t, uint64(decoded.DescriptorTable.Pointer()), binary.LittleEndian.Uint64(decodedLaunchBytes[0:]))
	core.AssertEqual(t, uint64(160), binary.LittleEndian.Uint64(decodedLaunchBytes[8:]))
	core.AssertEqual(t, uint64(4), binary.LittleEndian.Uint64(decodedLaunchBytes[32:]))
	decodedLaunchArgsBytes, err := (hipDecodeRequest{
		TokenID:         8,
		KV:              decoded.KV,
		DeviceKV:        decoded.DeviceKV,
		DescriptorTable: decoded.DescriptorTable,
	}).decodeLaunchArgsBytes()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipDecodeLaunchArgsBytes, len(decodedLaunchArgsBytes))
	core.AssertEqual(t, uint32(8), binary.LittleEndian.Uint32(decodedLaunchArgsBytes[12:]))
	core.AssertEqual(t, uint64(4), binary.LittleEndian.Uint64(decodedLaunchArgsBytes[16:]))
	core.AssertEqual(t, uint64(decoded.DescriptorTable.Pointer()), binary.LittleEndian.Uint64(decodedLaunchArgsBytes[hipDecodeLaunchArgsHeaderBytes:]))
	core.AssertEqual(t, uint64(160), binary.LittleEndian.Uint64(decodedLaunchArgsBytes[hipDecodeLaunchArgsHeaderBytes+8:]))
	keys, values, err := decoded.KV.Restore(3, 1)
	core.AssertNoError(t, err)
	core.AssertEqual(t, 2, len(keys))
	core.AssertEqual(t, 3, len(values))
	core.AssertEqual(t, "linked", decoded.Labels["decode_kernel"])
	core.AssertEqual(t, "mirrored", decoded.Labels["kv_device_backing"])
	descriptor, err := decoded.DeviceKV.KernelDescriptor()
	core.RequireNoError(t, err)
	core.AssertEqual(t, 4, descriptor.TokenCount)
	core.AssertEqual(t, 2, descriptor.Pages[len(descriptor.Pages)-1].KeyWidth)
	core.AssertEqual(t, 3, descriptor.Pages[len(descriptor.Pages)-1].ValueWidth)
	if !prefill.DeviceKV.closed || !prefill.DescriptorTable.closed {
		t.Fatalf("prefill device resources were not closed after successful decode")
	}
	if len(driver.allocations) < 6 || len(driver.frees) < 3 {
		t.Fatalf("driver allocations=%+v frees=%+v, want prefill mirror/table and decode remirror/table", driver.allocations, driver.frees)
	}
}

func TestHIPKernels_RequestValidation_Bad(t *testing.T) {
	kernels := fakeLinkedHIPKernelSet{}

	_, err := kernels.Project(context.Background(), &hipLoadedModel{}, hipProjectionRequest{
		Input: []float32{1},
		FP16:  []uint16{0x3c00},
		Q8:    []int8{1},
		Rows:  1,
		Cols:  1,
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "only one projection weight encoding")

	_, err = kernels.Prefill(context.Background(), &hipLoadedModel{}, hipPrefillRequest{TokenIDs: []int32{-1}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token IDs")

	_, err = kernels.Prefill(context.Background(), &hipLoadedModel{}, hipPrefillRequest{TokenIDs: []int32{1}, CacheMode: "not-a-mode"})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported cache mode")

	_, err = kernels.Prefill(context.Background(), &hipLoadedModel{}, hipPrefillRequest{TokenIDs: []int32{1}, KeyWidth: -1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV vector widths")

	_, err = kernels.Decode(context.Background(), &hipLoadedModel{}, hipDecodeRequest{TokenID: -1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token ID")

	_, err = kernels.Decode(context.Background(), &hipLoadedModel{}, hipDecodeRequest{TokenID: 1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prefill KV cache is required")

	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.Append(0, []float32{1, 2}, []float32{2, 1}))
	device, err := cache.MirrorToDevice(&fakeHIPDriver{available: true})
	core.RequireNoError(t, err)
	defer device.Close()
	table, err := device.KernelDescriptorTable()
	core.RequireNoError(t, err)
	defer table.Close()
	mismatched, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, mismatched.Append(0, []float32{1}, []float32{1}))
	_, err = kernels.Decode(context.Background(), &hipLoadedModel{}, hipDecodeRequest{TokenID: 1, KV: mismatched, DeviceKV: device})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "device KV cache")

	_, err = kernels.Decode(context.Background(), &hipLoadedModel{}, hipDecodeRequest{TokenID: 1, KV: cache, DeviceKV: device})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "descriptor table")

	_, err = kernels.Decode(context.Background(), &hipLoadedModel{}, hipDecodeRequest{TokenID: 1, KV: cache, DescriptorTable: table})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "descriptor table")

	_, err = (hipDecodeRequest{TokenID: 1, KV: cache}).kvLaunchDescriptorBytes()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "device KV cache")

	_, err = (hipDecodeRequest{TokenID: 1, KV: cache}).decodeLaunchArgsBytes()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "device KV cache")

	validDecodeLaunch, err := (hipDecodeRequest{
		TokenID:         1,
		KV:              cache,
		DeviceKV:        device,
		DescriptorTable: table,
	}).decodeLaunchArgs()
	core.AssertNoError(t, err)
	validDecodeLaunchBytes, err := validDecodeLaunch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, hipDecodeLaunchArgsBytes, len(validDecodeLaunchBytes))
	validDecodeLaunch.KV.StatusPointer = 2468
	validDecodeLaunchBytes, err = validDecodeLaunch.Binary()
	core.AssertNoError(t, err)
	core.AssertEqual(t, uint64(2468), binary.LittleEndian.Uint64(validDecodeLaunchBytes[hipDecodeLaunchArgsHeaderBytes+48:]))
	core.AssertEqual(t, hipDecodeLaunchStatusOK, binary.LittleEndian.Uint32(validDecodeLaunchBytes[hipDecodeLaunchArgsHeaderBytes+56:]))

	badDecodeLaunch := validDecodeLaunch
	badDecodeLaunch.TokenID = -1
	_, err = badDecodeLaunch.Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token ID")

	badDecodeLaunch = validDecodeLaunch
	badDecodeLaunch.Position++
	_, err = badDecodeLaunch.Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "decode position")

	core.RequireNoError(t, table.Close())
	_, err = kernels.Decode(context.Background(), &hipLoadedModel{}, hipDecodeRequest{TokenID: 1, KV: cache, DeviceKV: device, DescriptorTable: table})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "descriptor table")
}

func TestHIPKernels_BadDecodeDeviceMirrorFailureKeepsOriginalKV(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	model := &hipLoadedModel{driver: driver, kernels: fakeLinkedHIPKernelSet{}}
	prefill, err := model.Prefill(context.Background(), hipPrefillRequest{
		TokenIDs:   []int32{1, 2},
		CacheMode:  rocmKVCacheModeQ8,
		KeyWidth:   2,
		ValueWidth: 2,
	})
	core.RequireNoError(t, err)
	defer prefill.DeviceKV.Close()
	defer prefill.DescriptorTable.Close()
	failAt := len(driver.copies) + 2
	driver.copyErr = core.NewError("copy failed")
	driver.copyErrAt = failAt
	freesBeforeDecode := len(driver.frees)

	decoded, err := model.DecodeToken(context.Background(), hipDecodeRequest{
		TokenID:         9,
		KV:              prefill.KV,
		DeviceKV:        prefill.DeviceKV,
		DescriptorTable: prefill.DescriptorTable,
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy KV value page")
	core.AssertNil(t, decoded.KV)
	core.AssertEqual(t, 2, prefill.KV.TokenCount())
	core.AssertEqual(t, 2, prefill.DeviceKV.TokenCount())
	if prefill.DeviceKV.closed || prefill.DescriptorTable.closed {
		t.Fatalf("prefill device resources were closed after failed remirror")
	}
	if got := len(driver.frees) - freesBeforeDecode; got != 2 {
		t.Fatalf("decode frees = %d (%+v), want only failed remirror allocations cleaned up", got, driver.frees[freesBeforeDecode:])
	}
}

func TestHIPKernels_BadDecodeDescriptorTableFailureKeepsOriginalKV(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	model := &hipLoadedModel{driver: driver, kernels: fakeLinkedHIPKernelSet{}}
	prefill, err := model.Prefill(context.Background(), hipPrefillRequest{
		TokenIDs:   []int32{1, 2},
		CacheMode:  rocmKVCacheModeQ8,
		KeyWidth:   2,
		ValueWidth: 2,
	})
	core.RequireNoError(t, err)
	defer prefill.DeviceKV.Close()
	defer prefill.DescriptorTable.Close()
	driver.copyErr = core.NewError("descriptor copy failed")
	driver.copyErrAt = len(driver.copies) + 2*(prefill.KV.PageCount()+1) + 1
	freesBeforeDecode := len(driver.frees)

	decoded, err := model.DecodeToken(context.Background(), hipDecodeRequest{
		TokenID:         9,
		KV:              prefill.KV,
		DeviceKV:        prefill.DeviceKV,
		DescriptorTable: prefill.DescriptorTable,
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy descriptor table")
	core.AssertNil(t, decoded.KV)
	core.AssertEqual(t, 2, prefill.KV.TokenCount())
	core.AssertEqual(t, 2, prefill.DeviceKV.TokenCount())
	if prefill.DeviceKV.closed || prefill.DescriptorTable.closed {
		t.Fatalf("prefill device resources were closed after failed descriptor table update")
	}
	if got := len(driver.frees) - freesBeforeDecode; got != 5 {
		t.Fatalf("decode frees = %d (%+v), want failed descriptor table and updated mirror cleaned up", got, driver.frees[freesBeforeDecode:])
	}
}

type fakeLinkedHIPKernelSet struct {
	tokens []inference.Token
}

func (fakeLinkedHIPKernelSet) Status() hipKernelStatus {
	return hipKernelStatus{
		CrossEntropy: hipKernelStatusLinked,
		Decode:       hipKernelStatusLinked,
		Distillation: hipKernelStatusLinked,
		GRPO:         hipKernelStatusLinked,
		Prefill:      hipKernelStatusLinked,
		Projection:   hipKernelStatusLinked,
		KVCache:      hipKernelStatusPlanned,
		Reason:       "fake linked test kernel",
	}
}

func (kernels fakeLinkedHIPKernelSet) Generate(_ context.Context, _ *hipLoadedModel, _ string, _ inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	return func(yield func(inference.Token) bool) {
		for _, token := range kernels.tokens {
			if !yield(token) {
				return
			}
		}
	}, func() error { return nil }
}

func (kernels fakeLinkedHIPKernelSet) Chat(ctx context.Context, model *hipLoadedModel, _ []inference.Message, cfg inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	return kernels.Generate(ctx, model, "", cfg)
}

func (kernels fakeLinkedHIPKernelSet) Classify(_ context.Context, _ *hipLoadedModel, prompts []string, _ inference.GenerateConfig) ([]inference.ClassifyResult, error) {
	results := make([]inference.ClassifyResult, len(prompts))
	for i := range results {
		results[i] = inference.ClassifyResult{Token: inference.Token{ID: int32(i + 1), Text: "ok"}}
	}
	return results, nil
}

func (kernels fakeLinkedHIPKernelSet) BatchGenerate(_ context.Context, _ *hipLoadedModel, prompts []string, _ inference.GenerateConfig) ([]inference.BatchResult, error) {
	results := make([]inference.BatchResult, len(prompts))
	for i := range results {
		results[i] = inference.BatchResult{Tokens: append([]inference.Token(nil), kernels.tokens...)}
	}
	return results, nil
}

func (kernels fakeLinkedHIPKernelSet) Project(ctx context.Context, model *hipLoadedModel, req hipProjectionRequest) ([]float32, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	if err := req.validate(); err != nil {
		return nil, err
	}
	if model != nil && model.driver != nil && model.driver.Available() {
		buffers, err := req.projectionDeviceBuffers(model.driver)
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
		if err := hipLaunchKernel(model.driver, config); err != nil {
			return nil, err
		}
		return buffers.ReadOutput()
	}
	if len(req.F32) > 0 {
		return hipReferenceF32Projection(req.Input, req.F32, req.Rows, req.Cols, req.Bias)
	}
	if len(req.FP16) > 0 {
		return hipReferenceFP16Projection(req.Input, req.FP16, req.Rows, req.Cols, req.Bias)
	}
	if len(req.BF16) > 0 {
		return hipReferenceBF16Projection(req.Input, req.BF16, req.Rows, req.Cols, req.Bias)
	}
	return hipReferenceQ8Projection(req.Input, req.Q8, req.Q8Scale, req.Rows, req.Cols, req.Bias)
}

func (kernels fakeLinkedHIPKernelSet) Prefill(ctx context.Context, model *hipLoadedModel, req hipPrefillRequest) (hipPrefillResult, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return hipPrefillResult{}, err
		}
	}
	if err := req.validate(); err != nil {
		return hipPrefillResult{}, err
	}
	tokens, err := req.resolvedTokenIDs(model)
	if err != nil {
		return hipPrefillResult{}, err
	}
	mode, keyWidth, valueWidth, err := req.kvConfig()
	if err != nil {
		return hipPrefillResult{}, err
	}
	cache, err := newROCmKVCache(mode, defaultROCmKVBlockSize)
	if err != nil {
		return hipPrefillResult{}, err
	}
	keys, values := fakeHIPKVTensors(tokens, keyWidth, valueWidth)
	if err := cache.AppendVectors(0, keyWidth, valueWidth, keys, values); err != nil {
		return hipPrefillResult{}, err
	}
	labels := map[string]string{
		"kv_cache_mode":  mode,
		"kv_key_width":   core.Sprintf("%d", keyWidth),
		"kv_value_width": core.Sprintf("%d", valueWidth),
		"prefill_kernel": hipKernelStatusLinked,
	}
	var deviceKV *rocmDeviceKVCache
	var descriptorTable *rocmDeviceKVDescriptorTable
	if model != nil && model.driver != nil && model.driver.Available() {
		tokenBuffer, err := hipUploadTokenIDs(model.driver, tokens)
		if err != nil {
			return hipPrefillResult{}, err
		}
		defer tokenBuffer.Close()
		launch, err := req.prefillLaunchArgs(tokenBuffer)
		if err != nil {
			return hipPrefillResult{}, err
		}
		launchBytes, err := launch.Binary()
		if err != nil {
			return hipPrefillResult{}, err
		}
		config, err := hipOneDimensionalLaunchConfig(hipKernelNamePrefill, launchBytes, len(tokens))
		if err != nil {
			return hipPrefillResult{}, err
		}
		if err := hipLaunchKernel(model.driver, config); err != nil {
			return hipPrefillResult{}, err
		}
		addFakeHIPPrefillLaunchArgsLabels(labels, launch, len(launchBytes))
		device, err := cache.MirrorToDevice(model.driver)
		if err != nil {
			return hipPrefillResult{}, err
		}
		table, err := device.KernelDescriptorTable()
		if err != nil {
			_ = device.Close()
			return hipPrefillResult{}, err
		}
		deviceKV = device
		descriptorTable = table
		for key, value := range device.Stats().Labels {
			labels[key] = value
		}
		addFakeHIPDescriptorTableLabels(labels, table)
	}
	return hipPrefillResult{
		Logits:          []float32{float32(len(tokens))},
		PromptTokens:    len(tokens),
		KV:              cache,
		DeviceKV:        deviceKV,
		DescriptorTable: descriptorTable,
		Labels:          labels,
	}, nil
}

func (kernels fakeLinkedHIPKernelSet) Decode(ctx context.Context, _ *hipLoadedModel, req hipDecodeRequest) (hipDecodeResult, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return hipDecodeResult{}, err
		}
	}
	if err := req.validate(); err != nil {
		return hipDecodeResult{}, err
	}
	var decodeLaunch hipDecodeLaunchArgs
	var decodeLaunchBytes []byte
	if req.DeviceKV != nil {
		args, err := req.decodeLaunchArgs()
		if err != nil {
			return hipDecodeResult{}, err
		}
		payload, err := args.Binary()
		if err != nil {
			return hipDecodeResult{}, err
		}
		config, err := hipOneDimensionalLaunchConfig(hipKernelNameDecode, payload, 1)
		if err != nil {
			return hipDecodeResult{}, err
		}
		if err := hipLaunchKernel(req.DeviceKV.driver, config); err != nil {
			return hipDecodeResult{}, err
		}
		decodeLaunch = args
		decodeLaunchBytes = payload
	}
	keyWidth, valueWidth, err := req.kvVectorWidths()
	if err != nil {
		return hipDecodeResult{}, err
	}
	keys, values := fakeHIPKVTensors([]int32{req.TokenID}, keyWidth, valueWidth)
	targetKV := req.KV
	if req.DeviceKV != nil {
		cloned, err := req.KV.Clone()
		if err != nil {
			return hipDecodeResult{}, err
		}
		targetKV = cloned
	}
	if err := targetKV.AppendToken(targetKV.TokenCount(), keys, values); err != nil {
		return hipDecodeResult{}, err
	}
	labels := map[string]string{"decode_kernel": hipKernelStatusLinked}
	var deviceKV *rocmDeviceKVCache
	var descriptorTable *rocmDeviceKVDescriptorTable
	if req.DeviceKV != nil {
		updated, err := targetKV.MirrorToDevice(req.DeviceKV.driver)
		if err != nil {
			return hipDecodeResult{}, err
		}
		table, err := updated.KernelDescriptorTable()
		if err != nil {
			_ = updated.Close()
			return hipDecodeResult{}, err
		}
		if req.DescriptorTable != nil {
			_ = req.DescriptorTable.Close()
		}
		_ = req.DeviceKV.Close()
		deviceKV = updated
		descriptorTable = table
		for key, value := range deviceKV.Stats().Labels {
			labels[key] = value
		}
		addFakeHIPDescriptorTableLabels(labels, table)
		launch, err := updated.KernelLaunchDescriptor(table)
		if err != nil {
			_ = table.Close()
			_ = updated.Close()
			return hipDecodeResult{}, err
		}
		launchArgs, err := launch.Binary()
		if err != nil {
			_ = table.Close()
			_ = updated.Close()
			return hipDecodeResult{}, err
		}
		addFakeHIPLaunchDescriptorLabels(labels, launch)
		labels["kv_launch_args_bytes"] = core.Sprintf("%d", len(launchArgs))
		addFakeHIPDecodeLaunchArgsLabels(labels, decodeLaunch, len(decodeLaunchBytes))
	}
	return hipDecodeResult{
		Token:           inference.Token{ID: req.TokenID, Text: "ok"},
		Logits:          []float32{float32(req.TokenID)},
		KV:              targetKV,
		DeviceKV:        deviceKV,
		DescriptorTable: descriptorTable,
		Labels:          labels,
	}, nil
}

func addFakeHIPDescriptorTableLabels(labels map[string]string, table *rocmDeviceKVDescriptorTable) {
	if labels == nil || table == nil {
		return
	}
	labels["kv_descriptor_bytes"] = core.Sprintf("%d", table.SizeBytes())
	labels["kv_descriptor_pages"] = core.Sprintf("%d", table.pageCount)
	labels["kv_descriptor_table"] = "hip_device"
	labels["kv_descriptor_version"] = core.Sprintf("%d", table.version)
}

func addFakeHIPLaunchDescriptorLabels(labels map[string]string, launch rocmDeviceKVLaunchDescriptor) {
	if labels == nil {
		return
	}
	labels["kv_launch_block_size"] = core.Sprintf("%d", launch.BlockSize)
	labels["kv_launch_descriptor"] = "ready"
	labels["kv_launch_descriptor_bytes"] = core.Sprintf("%d", launch.DescriptorBytes)
	labels["kv_launch_mode"] = launch.Mode
	labels["kv_launch_pages"] = core.Sprintf("%d", launch.PageCount)
	labels["kv_launch_tokens"] = core.Sprintf("%d", launch.TokenCount)
}

func expectedGELUTanhMultiplyFromQ4(t *testing.T, gateReq, upReq hipMLXQ4ProjectionRequest) []float32 {
	t.Helper()
	gate, err := hipReferenceMLXQ4Projection(gateReq.Input, gateReq.Weight, gateReq.Scales, gateReq.Biases, gateReq.Rows, gateReq.Cols, gateReq.GroupSize)
	core.RequireNoError(t, err)
	up, err := hipReferenceMLXQ4Projection(upReq.Input, upReq.Weight, upReq.Scales, upReq.Biases, upReq.Rows, upReq.Cols, upReq.GroupSize)
	core.RequireNoError(t, err)
	return expectedGELUTanhMultiply(gate, up)
}

func expectedGELUTanhProjectionFromQ4(t *testing.T, req hipMLXQ4ProjectionRequest, multiplier []float32) []float32 {
	t.Helper()
	projected, err := hipReferenceMLXQ4Projection(req.Input, req.Weight, req.Scales, req.Biases, req.Rows, req.Cols, req.GroupSize)
	core.RequireNoError(t, err)
	return expectedGELUTanhMultiply(projected, multiplier)
}

func expectedGELUTanhMultiply(gate, up []float32) []float32 {
	out := make([]float32, len(gate))
	const sqrt2OverPi = 0.7978845608028654
	const coeff = 0.044715
	for index := range out {
		value := float64(gate[index])
		gelu := 0.5 * value * (1 + math.Tanh(sqrt2OverPi*(value+coeff*value*value*value)))
		out[index] = float32(gelu) * up[index]
	}
	return out
}

func addFakeHIPPrefillLaunchArgsLabels(labels map[string]string, launch hipPrefillLaunchArgs, size int) {
	if labels == nil {
		return
	}
	labels["prefill_launch_args_bytes"] = core.Sprintf("%d", size)
	labels["prefill_launch_mode"] = launch.CacheMode
	labels["prefill_launch_tokens"] = core.Sprintf("%d", launch.TokenCount)
	labels["prefill_token_bytes"] = core.Sprintf("%d", launch.TokenBytes)
}

func addFakeHIPDecodeLaunchArgsLabels(labels map[string]string, args hipDecodeLaunchArgs, size int) {
	if labels == nil {
		return
	}
	labels["decode_launch_args_bytes"] = core.Sprintf("%d", size)
	labels["decode_launch_position"] = core.Sprintf("%d", args.Position)
	labels["decode_launch_token"] = core.Sprintf("%d", args.TokenID)
}

func fakeHIPKVTensors(tokens []int32, keyWidth, valueWidth int) ([]float32, []float32) {
	keys := make([]float32, len(tokens)*keyWidth)
	values := make([]float32, len(tokens)*valueWidth)
	for i, token := range tokens {
		for j := 0; j < keyWidth; j++ {
			keys[i*keyWidth+j] = float32(token) + float32(j)/100
		}
		for j := 0; j < valueWidth; j++ {
			values[i*valueWidth+j] = float32(token) - float32(j)/100
		}
	}
	return keys, values
}

func hipTinyOutputWeightsFP16Fixture() []uint16 {
	return []uint16{
		0x3c00, 0,
		0, 0x3c00,
		0x3c00, 0x3c00,
	}
}

func hipTinyOutputWeightsQ8Fixture() []int8 {
	return []int8{
		2, 0,
		0, 2,
		2, 2,
	}
}
