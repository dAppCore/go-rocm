// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
)

func TestKVCache_Good_FP16RoundTripsFakeBlocks(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeFP16, 2)
	core.RequireNoError(t, err)
	err = cache.Append(0, []float32{1, 0.5, -2, 4}, []float32{0, 2, 3, 0.25})
	core.RequireNoError(t, err)

	keys, values, err := cache.Restore(0, 4)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1, 0.5, -2, 4}, keys, 0)
	assertFloat32SlicesNear(t, []float32{0, 2, 3, 0.25}, values, 0)
}

func TestKVCache_Good_Q8RoundTripsWithinTolerance(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 4)
	core.RequireNoError(t, err)
	err = cache.Append(0, []float32{-1, -0.25, 0.5, 1}, []float32{0.75, -0.5, 0.25, -1})
	core.RequireNoError(t, err)

	keys, values, err := cache.Restore(0, 4)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{-1, -0.25, 0.5, 1}, keys, 0.01)
	assertFloat32SlicesNear(t, []float32{0.75, -0.5, 0.25, -1}, values, 0.01)
}

func TestKVCache_Good_KQ8VQ4UsesLessMemory(t *testing.T) {
	keys := []float32{-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5}
	values := []float32{1, 0.8, 0.6, 0.4, 0.2, 0, -0.2, -0.4, -0.6, -0.8, -1, -0.8, -0.6, -0.4, -0.2, 0}
	q8, err := newROCmKVCache(rocmKVCacheModeQ8, 16)
	core.RequireNoError(t, err)
	compact, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 16)
	core.RequireNoError(t, err)
	core.RequireNoError(t, q8.Append(0, keys, values))
	core.RequireNoError(t, compact.Append(0, keys, values))

	restoredKeys, restoredValues, err := compact.Restore(0, len(keys))

	core.RequireNoError(t, err)
	if compact.MemoryBytes() >= q8.MemoryBytes() {
		t.Fatalf("compact memory = %d, q8 memory = %d, want k-q8-v-q4 lower byte count", compact.MemoryBytes(), q8.MemoryBytes())
	}
	assertFloat32SlicesNear(t, keys, restoredKeys, 0.01)
	assertFloat32SlicesNear(t, values, restoredValues, 0.15)
}

func TestKVCache_Good_PagedAppendAvoidsFullConcatenation(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)

	err = cache.Append(0, []float32{1, 2, 3, 4, 5}, []float32{5, 4, 3, 2, 1})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 3, cache.PageCount())
	for _, block := range cache.blocks {
		if block.tokenCount > 2 {
			t.Fatalf("block = %+v, want paged blocks no larger than configured block size", block)
		}
	}
}

func TestKVCache_Good_RestoresOutOfOrderNonOverlappingPages(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeFP16, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.Append(2, []float32{3, 4}, []float32{7, 8}))
	core.RequireNoError(t, cache.Append(0, []float32{1, 2}, []float32{5, 6}))

	keys, values, err := cache.Restore(0, 4)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1, 2, 3, 4}, keys, 0)
	assertFloat32SlicesNear(t, []float32{5, 6, 7, 8}, values, 0)
	if cache.blocks[0].tokenStart != 0 || cache.blocks[1].tokenStart != 2 {
		t.Fatalf("blocks = %+v, want deterministic token order", cache.blocks)
	}
}

func TestKVCache_Good_RoundTripsPagedTokenVectors(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeFP16, 2)
	core.RequireNoError(t, err)
	err = cache.AppendVectors(
		10,
		2,
		3,
		[]float32{1, 0, 0.5, -0.5, -1, 1},
		[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
	)
	core.RequireNoError(t, err)

	keys, values, err := cache.Restore(11, 2)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, cache.PageCount())
	core.AssertEqual(t, 13, cache.TokenCount())
	assertFloat32SlicesNear(t, []float32{0.5, -0.5, -1, 1}, keys, 0)
	assertFloat32SlicesNear(t, []float32{4, 5, 6, 7, 8, 9}, values, 0)
}

func TestKVCache_Good_AppendsSingleDecodeTokenVector(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))

	err = cache.AppendToken(cache.TokenCount(), []float32{-1, 1}, []float32{3, -3})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 3, cache.TokenCount())
	keys, values, err := cache.Restore(2, 1)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{-1, 1}, keys, 0.01)
	assertFloat32SlicesNear(t, []float32{3, -3}, values, 0.03)
}

func TestKVCache_Good_StatsHitRateRestoreTime(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.Append(0, []float32{1, 2}, []float32{2, 1}))
	_, _, err = cache.Restore(0, 2)
	core.RequireNoError(t, err)
	_, _, err = cache.Restore(4, 1)
	core.AssertError(t, err)

	stats := cache.Stats()

	core.AssertEqual(t, 1, stats.Blocks)
	core.AssertEqual(t, uint64(1), stats.Hits)
	core.AssertEqual(t, uint64(1), stats.Misses)
	assertFloat32Near(t, 0.5, float32(stats.HitRate))
	if stats.RestoreMillis <= 0 {
		t.Fatalf("restore millis = %f, want positive restore timing", stats.RestoreMillis)
	}
	core.AssertEqual(t, rocmKVCacheModeQ8, stats.CacheMode)
	core.AssertEqual(t, "package_local", stats.Labels["kv_backing"])
	core.AssertEqual(t, "planned", stats.Labels["kv_device_backing"])
	core.AssertEqual(t, "2", stats.Labels["kv_block_size"])
	core.AssertEqual(t, "1", stats.Labels["kv_key_width"])
	core.AssertEqual(t, "1", stats.Labels["kv_value_width"])
	core.AssertEqual(t, "1", stats.Labels["kv_pages"])
	core.AssertEqual(t, "2", stats.Labels["kv_tokens"])
}

func TestKVCache_Good_SnapshotRoundTripsRuntimeOwnedPages(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(
		0,
		2,
		3,
		[]float32{1, 0.5, -1, 0},
		[]float32{0.75, -0.5, 0.25, 1, -1, 0.5},
	))

	payload, err := cache.Snapshot()
	core.RequireNoError(t, err)
	restored, err := newROCmKVCacheFromSnapshot(payload)
	core.RequireNoError(t, err)
	keys, values, err := restored.Restore(0, 2)

	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, restored.Stats().CacheMode)
	core.AssertEqual(t, 1, restored.PageCount())
	core.AssertEqual(t, 2, restored.TokenCount())
	assertFloat32SlicesNear(t, []float32{1, 0.5, -1, 0}, keys, 0.01)
	assertFloat32SlicesNear(t, []float32{0.75, -0.5, 0.25, 1, -1, 0.5}, values, 0.15)
}

func TestKVCache_Good_CloneDoesNotAliasRuntimeOwnedPages(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))

	clone, err := cache.Clone()
	core.RequireNoError(t, err)
	core.RequireNoError(t, clone.AppendToken(2, []float32{3, 4}, []float32{5, 6}))

	core.AssertEqual(t, 2, cache.TokenCount())
	core.AssertEqual(t, 3, clone.TokenCount())
	core.AssertEqual(t, rocmKVCacheModeQ8, clone.Stats().CacheMode)
	core.AssertEqual(t, "2", clone.Stats().Labels["kv_key_width"])
	core.AssertEqual(t, "2", clone.Stats().Labels["kv_value_width"])
}

func TestKVCache_Good_MirrorsPagesToHIPDevice(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(
		0,
		2,
		3,
		[]float32{1, 0.5, -1, 0},
		[]float32{0.75, -0.5, 0.25, 1, -1, 0.5},
	))
	driver := &fakeHIPDriver{available: true}

	device, err := cache.MirrorToDevice(driver)
	core.RequireNoError(t, err)
	defer device.Close()

	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, device.mode)
	core.AssertEqual(t, 1, device.PageCount())
	core.AssertEqual(t, 2, device.TokenCount())
	core.AssertEqual(t, uint64(15), device.MemoryBytes())
	core.AssertEqual(t, []uint64{8, 7}, driver.allocations)
	core.AssertEqual(t, []uint64{8, 7}, driver.copies)
	core.AssertEqual(t, 2, driver.pinnedCopies)
	stats := device.Stats()
	core.AssertEqual(t, 1, stats.Blocks)
	core.AssertEqual(t, uint64(15), stats.MemoryBytes)
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, stats.CacheMode)
	core.AssertEqual(t, "hip_device_mirror", stats.Labels["kv_backing"])
	core.AssertEqual(t, "mirrored", stats.Labels["kv_device_backing"])
	core.AssertEqual(t, "2", stats.Labels["kv_key_width"])
	core.AssertEqual(t, "3", stats.Labels["kv_value_width"])
	core.AssertEqual(t, "1", stats.Labels["kv_pages"])
	core.AssertEqual(t, "2", stats.Labels["kv_tokens"])
	descriptor, err := device.KernelDescriptor()
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, descriptor.Mode)
	core.AssertEqual(t, 2, descriptor.BlockSize)
	core.AssertEqual(t, 2, descriptor.TokenCount)
	core.AssertEqual(t, 1, len(descriptor.Pages))
	core.AssertTrue(t, descriptor.Pages[0].KeyPointer != 0)
	core.AssertTrue(t, descriptor.Pages[0].ValuePointer != 0)
	core.AssertEqual(t, rocmKVEncodingQ8, descriptor.Pages[0].KeyEncoding)
	core.AssertEqual(t, rocmKVEncodingQ4, descriptor.Pages[0].ValueEncoding)
	core.AssertEqual(t, uint64(8), descriptor.Pages[0].KeyBytes)
	core.AssertEqual(t, uint64(7), descriptor.Pages[0].ValueBytes)
	descriptorBytes, err := device.KernelDescriptorBytes()
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmDeviceKVDescriptorHeaderBytes+rocmDeviceKVDescriptorPageBytes, len(descriptorBytes))
	core.AssertEqual(t, rocmDeviceKVDescriptorVersion, binary.LittleEndian.Uint32(descriptorBytes[0:]))
	core.AssertEqual(t, uint32(rocmDeviceKVDescriptorHeaderBytes), binary.LittleEndian.Uint32(descriptorBytes[4:]))
	core.AssertEqual(t, uint32(rocmDeviceKVDescriptorPageBytes), binary.LittleEndian.Uint32(descriptorBytes[8:]))
	core.AssertEqual(t, rocmDeviceKVDescriptorModeKQ8VQ4, binary.LittleEndian.Uint32(descriptorBytes[12:]))
	core.AssertEqual(t, uint32(1), binary.LittleEndian.Uint32(descriptorBytes[16:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(descriptorBytes[20:]))
	core.AssertEqual(t, uint64(2), binary.LittleEndian.Uint64(descriptorBytes[24:]))
	pageBytes := descriptorBytes[rocmDeviceKVDescriptorHeaderBytes:]
	core.AssertEqual(t, uint64(0), binary.LittleEndian.Uint64(pageBytes[0:]))
	core.AssertEqual(t, uint64(2), binary.LittleEndian.Uint64(pageBytes[8:]))
	core.AssertEqual(t, uint32(2), binary.LittleEndian.Uint32(pageBytes[16:]))
	core.AssertEqual(t, uint32(3), binary.LittleEndian.Uint32(pageBytes[20:]))
	core.AssertEqual(t, rocmDeviceKVDescriptorEncodingQ8, binary.LittleEndian.Uint32(pageBytes[24:]))
	core.AssertEqual(t, rocmDeviceKVDescriptorEncodingQ4, binary.LittleEndian.Uint32(pageBytes[28:]))
	core.AssertEqual(t, uint64(descriptor.Pages[0].KeyPointer), binary.LittleEndian.Uint64(pageBytes[32:]))
	core.AssertEqual(t, uint64(descriptor.Pages[0].ValuePointer), binary.LittleEndian.Uint64(pageBytes[40:]))
	core.AssertEqual(t, uint64(8), binary.LittleEndian.Uint64(pageBytes[48:]))
	core.AssertEqual(t, uint64(7), binary.LittleEndian.Uint64(pageBytes[56:]))
	table, err := device.KernelDescriptorTable()
	core.RequireNoError(t, err)
	core.AssertTrue(t, table.Pointer() != 0)
	core.AssertEqual(t, uint64(len(descriptorBytes)), table.SizeBytes())
	core.AssertEqual(t, rocmDeviceKVDescriptorVersion, table.version)
	core.AssertEqual(t, 1, table.pageCount)
	core.AssertEqual(t, []uint64{8, 7, uint64(len(descriptorBytes))}, driver.allocations)
	core.AssertEqual(t, []uint64{8, 7, uint64(len(descriptorBytes))}, driver.copies)
	core.AssertEqual(t, 3, driver.pinnedCopies)
	core.RequireNoError(t, table.Close())
	core.AssertEqual(t, nativeDevicePointer(0), table.Pointer())
	core.AssertEqual(t, uint64(0), table.SizeBytes())
	core.RequireNoError(t, table.Close())

	core.RequireNoError(t, device.Close())
	_, err = device.KernelDescriptor()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "closed")
	_, err = device.KernelDescriptorBytes()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "closed")
	core.RequireNoError(t, device.Close())
	core.AssertEqual(t, 3, len(driver.frees))
}

func TestKVCache_Good_DeviceBorrowedAliasDoesNotOwnSourcePages(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(
		0,
		2,
		3,
		[]float32{1, 0.5, -1, 0},
		[]float32{0.75, -0.5, 0.25, 1, -1, 0.5},
	))
	driver := &fakeHIPDriver{available: true}
	device, err := cache.MirrorToDevice(driver)
	core.RequireNoError(t, err)
	defer device.Close()

	alias, err := device.borrowedAlias()
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, alias.borrowed)
	core.AssertEqual(t, device.PageCount(), alias.PageCount())
	core.AssertEqual(t, device.TokenCount(), alias.TokenCount())
	core.AssertEqual(t, device.pages[0].key.pointer, alias.pages[0].key.pointer)
	core.AssertEqual(t, device.pages[0].value.pointer, alias.pages[0].value.pointer)
	core.AssertEqual(t, false, alias.ownsAnyPages())
	core.AssertEqual(t, true, alias.borrowsPagesFrom(device))

	core.RequireNoError(t, alias.Close())
	core.AssertEqual(t, 0, len(driver.frees))
	_, err = device.KernelDescriptor()
	core.RequireNoError(t, err)
	core.RequireNoError(t, device.Close())
	core.AssertEqual(t, 2, len(driver.frees))
}

func TestKVCache_Good_DeviceMirrorSnapshotsFromHIPMemory(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(
		0,
		2,
		3,
		[]float32{1, 0.5, -1, 0},
		[]float32{0.75, -0.5, 0.25, 1, -1, 0.5},
	))
	device, err := cache.MirrorToDevice(&fakeHIPDriver{available: true})
	core.RequireNoError(t, err)
	defer device.Close()

	payload, err := device.Snapshot()
	core.RequireNoError(t, err)
	restored, err := newROCmKVCacheFromSnapshot(payload)
	core.RequireNoError(t, err)
	keys, values, err := restored.Restore(0, 2)

	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, restored.Stats().CacheMode)
	core.AssertEqual(t, 1, restored.PageCount())
	core.AssertEqual(t, 2, restored.TokenCount())
	assertFloat32SlicesNear(t, []float32{1, 0.5, -1, 0}, keys, 0.01)
	assertFloat32SlicesNear(t, []float32{0.75, -0.5, 0.25, 1, -1, 0.5}, values, 0.15)
}

func TestKVCache_Good_DeviceMirrorAppendsDecodeTokenIncrementally(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	driver := &fakeHIPDriver{available: true}
	device, err := cache.MirrorToDevice(driver)
	core.RequireNoError(t, err)
	sourcePageCount := device.PageCount()

	next, err := device.withAppendedToken([]float32{-1, 1}, []float32{3, -3})
	core.RequireNoError(t, err)
	table, err := next.KernelDescriptorTable()
	core.RequireNoError(t, err)
	core.RequireNoError(t, device.transferPagesTo(next))
	core.RequireNoError(t, device.Close())
	core.AssertEqual(t, true, device.closed)
	core.AssertEqual(t, 3, next.TokenCount())
	core.AssertEqual(t, 2, next.PageCount())
	core.AssertEqual(t, uint64(rocmDeviceKVDescriptorHeaderBytes+2*rocmDeviceKVDescriptorPageBytes), table.SizeBytes())
	core.AssertEqual(t, 2, table.pageCount)
	core.AssertEqual(t, []uint64{8, 8, 6, 6, uint64(rocmDeviceKVDescriptorHeaderBytes + 2*rocmDeviceKVDescriptorPageBytes)}, driver.allocations)
	core.AssertEqual(t, []uint64{8, 8, 6, 6, uint64(rocmDeviceKVDescriptorHeaderBytes + 2*rocmDeviceKVDescriptorPageBytes)}, driver.copies)
	payload, err := next.Snapshot()
	core.RequireNoError(t, err)
	restored, err := newROCmKVCacheFromSnapshot(payload)
	core.RequireNoError(t, err)
	keys, values, err := restored.Restore(0, 3)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1, 0, 0, 1, -1, 1}, keys, 0.01)
	assertFloat32SlicesNear(t, []float32{2, 0, 0, 2, 3, -3}, values, 0.03)
	core.RequireNoError(t, table.Close())
	core.RequireNoError(t, next.Close())
	core.AssertEqual(t, 1, sourcePageCount)
	core.AssertEqual(t, 5, len(driver.frees))
}

func TestKVCache_Good_KVEncodeTokenKernelEncodesDeviceToken(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	keyInput, err := hipUploadByteBuffer(driver, "rocm.KVCache.Test", "key token", mustHIPFloat32Payload(t, []float32{1, -0.5, 0.25}), 3)
	core.RequireNoError(t, err)
	defer keyInput.Close()
	valueInput, err := hipUploadByteBuffer(driver, "rocm.KVCache.Test", "value token", mustHIPFloat32Payload(t, []float32{0.75, -0.75, 0.25}), 3)
	core.RequireNoError(t, err)
	defer valueInput.Close()

	key, value, err := hipRunKVEncodeTokenKernel(context.Background(), driver, keyInput, valueInput, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer driver.Free(key.pointer)
	defer driver.Free(value.pointer)

	core.AssertEqual(t, rocmKVEncodingQ8, key.encoding)
	core.AssertEqual(t, rocmKVEncodingQ4, value.encoding)
	core.AssertEqual(t, uint64(7), key.sizeBytes)
	core.AssertEqual(t, uint64(6), value.sizeBytes)
	keyDecoded, err := copyROCmDeviceKVTensorToHost(driver, key, 3)
	core.RequireNoError(t, err)
	valueDecoded, err := copyROCmDeviceKVTensorToHost(driver, value, 3)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1, -0.5, 0.25}, keyDecoded.decode(), 0.01)
	assertFloat32SlicesNear(t, []float32{0.75, -0.75, 0.25}, valueDecoded.decode(), 0.12)
	core.AssertEqual(t, hipKernelNameKVEncodeToken, driver.launches[len(driver.launches)-1].Name)
}

func TestKVCache_Good_RowScaledTensorEncoding(t *testing.T) {
	keyTensor, err := encodeROCmKVTensorRows(rocmKVEncodingQ8Rows, []float32{100, -100, 0.5, -0.5}, 2, 2)
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint64(12), keyTensor.sizeBytes)
	assertFloat32SlicesNear(t, []float32{100, -100, 0.5, -0.5}, keyTensor.decodeRows(2), 0.01)

	valueTensor, err := encodeROCmKVTensorRows(rocmKVEncodingQ4Rows, []float32{7, -7, 0.25, -0.25}, 2, 2)
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint64(10), valueTensor.sizeBytes)
	assertFloat32SlicesNear(t, []float32{7, -7, 0.25, -0.25}, valueTensor.decodeRows(2), 0.02)

	payload, err := valueTensor.deviceBytes()
	core.RequireNoError(t, err)
	restored, err := rocmKVTensorFromDeviceBytesRows(rocmKVEncodingQ4Rows, valueTensor.length, 2, payload)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{7, -7, 0.25, -0.25}, restored.decodeRows(2), 0.02)
}

func TestKVCache_Good_DeviceMirrorAppendsDeviceRowsWindow(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	keyRows := []float32{
		100, -100,
		0.5, -0.5,
		-1, 1,
	}
	valueRows := []float32{
		7, -7,
		0.25, -0.25,
		3, -3,
	}
	keyInput, err := hipUploadByteBuffer(driver, "rocm.KVCache.Test", "key rows", mustHIPFloat32Payload(t, keyRows), len(keyRows))
	core.RequireNoError(t, err)
	defer keyInput.Close()
	valueInput, err := hipUploadByteBuffer(driver, "rocm.KVCache.Test", "value rows", mustHIPFloat32Payload(t, valueRows), len(valueRows))
	core.RequireNoError(t, err)
	defer valueInput.Close()

	cache := &rocmDeviceKVCache{driver: driver, mode: rocmKVCacheModeKQ8VQ4, blockSize: 2}
	next, err := cache.withAppendedDeviceRowsWindow(context.Background(), keyInput, valueInput, 2, 2, 3, 0)
	core.RequireNoError(t, err)
	defer next.Close()

	core.AssertEqual(t, 3, next.TokenCount())
	core.AssertEqual(t, 2, next.PageCount())
	core.AssertEqual(t, 2, next.pages[0].tokenCount)
	core.AssertEqual(t, 1, next.pages[1].tokenCount)
	core.AssertEqual(t, 0, next.pages[0].tokenStart)
	core.AssertEqual(t, 2, next.pages[1].tokenStart)
	core.AssertEqual(t, rocmKVEncodingQ8Rows, next.pages[0].key.encoding)
	core.AssertEqual(t, rocmKVEncodingQ4Rows, next.pages[0].value.encoding)
	core.AssertEqual(t, uint64(12), next.pages[0].key.sizeBytes)
	core.AssertEqual(t, uint64(10), next.pages[0].value.sizeBytes)
	core.AssertEqual(t, rocmKVEncodingQ8, next.pages[1].key.encoding)
	core.AssertEqual(t, rocmKVEncodingQ4, next.pages[1].value.encoding)
	core.AssertEqual(t, 2, countLaunchName(driver.launches, hipKernelNameKVEncodeToken))

	payload, err := next.Snapshot()
	core.RequireNoError(t, err)
	restored, err := newROCmKVCacheFromSnapshot(payload)
	core.RequireNoError(t, err)
	keys, values, err := restored.Restore(0, 3)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, keyRows, keys, 0.02)
	assertFloat32SlicesNear(t, valueRows, values, 0.06)

	descriptor, err := next.KernelDescriptor()
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(descriptor.Pages))
	core.AssertEqual(t, 2, descriptor.Pages[0].TokenCount)
	core.AssertEqual(t, 1, descriptor.Pages[1].TokenCount)
}

func TestKVCache_Bad_DeviceMirrorAppendsDeviceRowsWindow(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	keyInput, err := hipUploadByteBuffer(driver, "rocm.KVCache.Test", "bad key rows", mustHIPFloat32Payload(t, []float32{1, 0}), 2)
	core.RequireNoError(t, err)
	defer keyInput.Close()
	valueInput, err := hipUploadByteBuffer(driver, "rocm.KVCache.Test", "bad value rows", mustHIPFloat32Payload(t, []float32{1, 0}), 2)
	core.RequireNoError(t, err)
	defer valueInput.Close()
	cache := &rocmDeviceKVCache{driver: driver, mode: rocmKVCacheModeKQ8VQ4, blockSize: 2}

	if _, err := cache.withAppendedDeviceRowsWindow(context.Background(), keyInput, valueInput, 2, 2, 0, 0); err == nil {
		t.Fatalf("withAppendedDeviceRowsWindow succeeded with zero token count")
	}
	if _, err := cache.withAppendedDeviceRowsWindow(context.Background(), keyInput, valueInput, 2, 2, 2, 0); err == nil {
		t.Fatalf("withAppendedDeviceRowsWindow succeeded with mismatched row shape")
	}
	if _, err := newROCmDeviceKVCacheFromDeviceRows(context.Background(), &fakeHIPDriver{available: false}, rocmKVCacheModeKQ8VQ4, 2, keyInput, valueInput, 2, 2, 1, 0); err == nil {
		t.Fatalf("newROCmDeviceKVCacheFromDeviceRows succeeded with unavailable driver")
	}
	core.AssertEqual(t, 0, len(driver.launches))
}

func TestKVCache_Good_DeviceMirrorAppendsDeviceTokenWindow(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 1)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2,
		[]float32{1, 0, 0, 1},
		[]float32{2, 0, 0, 2},
	))
	driver := &fakeHIPDriver{available: true}
	device, err := cache.MirrorToDevice(driver)
	core.RequireNoError(t, err)
	keyInput, err := hipUploadByteBuffer(driver, "rocm.KVCache.Test", "key token", mustHIPFloat32Payload(t, []float32{-1, 1}), 2)
	core.RequireNoError(t, err)
	defer keyInput.Close()
	valueInput, err := hipUploadByteBuffer(driver, "rocm.KVCache.Test", "value token", mustHIPFloat32Payload(t, []float32{3, -3}), 2)
	core.RequireNoError(t, err)
	defer valueInput.Close()

	next, err := device.withAppendedDeviceTokenWindow(context.Background(), keyInput, valueInput, 2)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 2, next.TokenCount())
	core.AssertEqual(t, 2, next.PageCount())
	core.AssertEqual(t, rocmKVEncodingQ8, next.pages[1].key.encoding)
	core.AssertEqual(t, rocmKVEncodingQ4, next.pages[1].value.encoding)
	payload, err := next.Snapshot()
	core.RequireNoError(t, err)
	restored, err := newROCmKVCacheFromSnapshot(payload)
	core.RequireNoError(t, err)
	keys, values, err := restored.Restore(0, 2)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0, 1, -1, 1}, keys, 0.01)
	assertFloat32SlicesNear(t, []float32{0, 2, 3, -3}, values, 0.15)
	core.RequireNoError(t, device.transferSharedPagesTo(next))
	core.RequireNoError(t, next.Close())
	core.AssertEqual(t, true, device.closed)
}

func TestKVCache_Good_DeviceDescriptorAppendBuildsTableOnDevice(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 1)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2,
		[]float32{1, 0, 0, 1},
		[]float32{2, 0, 0, 2},
	))
	driver := &fakeHIPDriver{available: true}
	device, err := cache.MirrorToDevice(driver)
	core.RequireNoError(t, err)
	previousTable, err := device.KernelDescriptorTable()
	core.RequireNoError(t, err)
	defer previousTable.Close()
	keyInput, err := hipUploadByteBuffer(driver, "rocm.KVCache.Test", "key token", mustHIPFloat32Payload(t, []float32{-1, 1}), 2)
	core.RequireNoError(t, err)
	defer keyInput.Close()
	valueInput, err := hipUploadByteBuffer(driver, "rocm.KVCache.Test", "value token", mustHIPFloat32Payload(t, []float32{3, -3}), 2)
	core.RequireNoError(t, err)
	defer valueInput.Close()
	next, err := device.withAppendedDeviceTokenWindow(context.Background(), keyInput, valueInput, 2)
	core.RequireNoError(t, err)

	table, err := next.KernelDescriptorTableFromAppendedToken(context.Background(), device, previousTable)
	core.RequireNoError(t, err)
	defer table.Close()

	core.RequireNoError(t, table.CompatibleWith(next))
	got := make([]byte, table.SizeBytes())
	core.RequireNoError(t, driver.CopyDeviceToHost(table.Pointer(), got))
	want, err := next.KernelDescriptorBytes()
	core.RequireNoError(t, err)
	if !bytes.Equal(got, want) {
		t.Fatalf("device-built descriptor = %v, want %v", got, want)
	}
	core.AssertEqual(t, hipKernelNameKVDescriptorAppend, driver.launches[len(driver.launches)-1].Name)
	core.RequireNoError(t, device.transferSharedPagesTo(next))
	core.RequireNoError(t, next.Close())
	core.AssertEqual(t, true, device.closed)
}

func TestKVCache_Good_DeviceDescriptorAppendReusesCapacityInPlace(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 1)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2,
		[]float32{1, 0},
		[]float32{2, 0},
	))
	driver := &fakeHIPDriver{available: true}
	device, err := cache.MirrorToDevice(driver)
	core.RequireNoError(t, err)
	defer device.Close()
	payload, err := device.KernelDescriptorBytes()
	core.RequireNoError(t, err)
	pointer, allocationBytes, err := rocmDeviceKVDescriptorTableMalloc(driver, uint64(len(payload)))
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipCopyHostToDevice(driver, pointer, payload))
	previousTable := rocmBorrowDeviceKVDescriptorTableAllocated(driver, pointer, uint64(len(payload)), allocationBytes, rocmDeviceKVDescriptorVersion, device.PageCount(), false, true)
	keyInput, err := hipUploadByteBuffer(driver, "rocm.KVCache.Test", "key token", mustHIPFloat32Payload(t, []float32{-1, 1}), 2)
	core.RequireNoError(t, err)
	defer keyInput.Close()
	valueInput, err := hipUploadByteBuffer(driver, "rocm.KVCache.Test", "value token", mustHIPFloat32Payload(t, []float32{3, -3}), 2)
	core.RequireNoError(t, err)
	defer valueInput.Close()
	next, err := device.withAppendedDeviceTokenWindow(context.Background(), keyInput, valueInput, 4)
	core.RequireNoError(t, err)
	defer next.closePagesFrom(device.PageCount())
	allocationCount := len(driver.allocations)

	table, err := next.KernelDescriptorTableFromAppendedToken(context.Background(), device, previousTable)
	core.RequireNoError(t, err)
	defer table.Close()

	core.AssertEqual(t, previousTable, table)
	core.AssertEqual(t, allocationCount, len(driver.allocations))
	core.AssertEqual(t, uint64(rocmDeviceKVDescriptorHeaderBytes+2*rocmDeviceKVDescriptorPageBytes), table.SizeBytes())
	core.AssertEqual(t, allocationBytes, table.AllocationBytes())
	got := make([]byte, table.SizeBytes())
	core.RequireNoError(t, driver.CopyDeviceToHost(table.Pointer(), got))
	want, err := next.KernelDescriptorBytes()
	core.RequireNoError(t, err)
	if !bytes.Equal(got, want) {
		t.Fatalf("in-place device-built descriptor = %v, want %v", got, want)
	}
}

func TestKVCache_Good_DeviceDescriptorAppendReusesCapacityAcrossTrim(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 1)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2,
		[]float32{1, 0, 0, 1},
		[]float32{2, 0, 0, 2},
	))
	driver := &fakeHIPDriver{available: true}
	device, err := cache.MirrorToDevice(driver)
	core.RequireNoError(t, err)
	defer device.Close()
	payload, err := device.KernelDescriptorBytes()
	core.RequireNoError(t, err)
	pointer, allocationBytes, err := rocmDeviceKVDescriptorTableMalloc(driver, uint64(len(payload)))
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipCopyHostToDevice(driver, pointer, payload))
	previousTable := rocmBorrowDeviceKVDescriptorTableAllocated(driver, pointer, uint64(len(payload)), allocationBytes, rocmDeviceKVDescriptorVersion, device.PageCount(), false, true)
	keyInput, err := hipUploadByteBuffer(driver, "rocm.KVCache.Test", "key token", mustHIPFloat32Payload(t, []float32{-1, 1}), 2)
	core.RequireNoError(t, err)
	defer keyInput.Close()
	valueInput, err := hipUploadByteBuffer(driver, "rocm.KVCache.Test", "value token", mustHIPFloat32Payload(t, []float32{3, -3}), 2)
	core.RequireNoError(t, err)
	defer valueInput.Close()
	next, err := device.withAppendedDeviceTokenWindow(context.Background(), keyInput, valueInput, 2)
	core.RequireNoError(t, err)
	defer next.closePagesFrom(device.PageCount())
	allocationCount := len(driver.allocations)

	table, err := next.KernelDescriptorTableFromAppendedToken(context.Background(), device, previousTable)
	core.RequireNoError(t, err)
	defer table.Close()

	core.AssertNotEqual(t, previousTable, table)
	core.AssertEqual(t, allocationCount+1, len(driver.allocations))
	core.AssertEqual(t, 2, table.pageCount)
	got := make([]byte, table.SizeBytes())
	core.RequireNoError(t, driver.CopyDeviceToHost(table.Pointer(), got))
	want, err := next.KernelDescriptorBytes()
	core.RequireNoError(t, err)
	if !bytes.Equal(got, want) {
		t.Fatalf("in-place trimmed device-built descriptor = %v, want %v", got, want)
	}
}

func TestKVCache_Good_DeviceFinalizeTransfersInPlaceDescriptorTable(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	sourcePages := []rocmDeviceKVPage{{
		tokenStart: 0,
		tokenCount: 1,
		key:        rocmDeviceKVTensor{pointer: 0x1001, sizeBytes: 4, encoding: rocmKVEncodingQ8},
		value:      rocmDeviceKVTensor{pointer: 0x1002, sizeBytes: 4, encoding: rocmKVEncodingQ4},
		owned:      true,
	}}
	targetPages := []rocmDeviceKVPage{
		sourcePages[0],
		{
			tokenStart: 1,
			tokenCount: 1,
			key:        rocmDeviceKVTensor{pointer: 0x2001, sizeBytes: 4, encoding: rocmKVEncodingQ8},
			value:      rocmDeviceKVTensor{pointer: 0x2002, sizeBytes: 4, encoding: rocmKVEncodingQ4},
			owned:      true,
		},
	}
	targetPages[0].owned = false
	table := rocmBorrowDeviceKVDescriptorTableAllocated(
		driver,
		0x5000,
		uint64(rocmDeviceKVDescriptorHeaderBytes+2*rocmDeviceKVDescriptorPageBytes),
		rocmDeviceKVDescriptorTableAllocationBytes(uint64(rocmDeviceKVDescriptorHeaderBytes+2*rocmDeviceKVDescriptorPageBytes)),
		rocmDeviceKVDescriptorVersion,
		2,
		false,
		true,
	)
	previous := &hipGemma4Q4DeviceDecodeState{layers: []hipGemma4Q4DeviceLayerKVState{{
		cache:           rocmBorrowDeviceKVCache(driver, rocmKVCacheModeKQ8VQ4, 1, 1, sourcePages, false),
		descriptorTable: table,
	}}}
	next := &hipGemma4Q4DeviceDecodeState{layers: []hipGemma4Q4DeviceLayerKVState{{
		cache:           rocmBorrowDeviceKVCache(driver, rocmKVCacheModeKQ8VQ4, 1, 2, targetPages, false),
		descriptorTable: table,
	}}}

	core.RequireNoError(t, hipFinalizeGemma4Q4ForwardDeviceState(previous, next))

	core.AssertEqual(t, false, table.closed)
	core.AssertEqual(t, table, next.layers[0].descriptorTable)
	core.AssertEqual(t, true, next.layers[0].cache.pages[0].owned)
	core.RequireNoError(t, next.Close())
}

func TestKVCache_Good_DeviceMirrorWindowAppendTrimsAndTransfersPages(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 1)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2,
		[]float32{1, 0, 0, 1},
		[]float32{2, 0, 0, 2},
	))
	driver := &fakeHIPDriver{available: true}
	device, err := cache.MirrorToDevice(driver)
	core.RequireNoError(t, err)

	next, err := device.withAppendedTokenWindow([]float32{-1, 1}, []float32{3, -3}, 2)
	core.RequireNoError(t, err)
	table, err := next.KernelDescriptorTable()
	core.RequireNoError(t, err)
	defer table.Close()

	core.AssertEqual(t, 2, next.TokenCount())
	core.AssertEqual(t, 2, next.PageCount())
	core.AssertEqual(t, 0, next.pages[0].tokenStart)
	core.AssertEqual(t, 1, next.pages[1].tokenStart)
	core.RequireNoError(t, device.transferSharedPagesTo(next))
	core.AssertEqual(t, true, device.closed)

	payload, err := next.Snapshot()
	core.RequireNoError(t, err)
	restored, err := newROCmKVCacheFromSnapshot(payload)
	core.RequireNoError(t, err)
	keys, values, err := restored.Restore(0, 2)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0, 1, -1, 1}, keys, 0.01)
	assertFloat32SlicesNear(t, []float32{0, 2, 3, -3}, values, 0.03)
	core.RequireNoError(t, next.Close())
}

func TestKVCache_Good_DeviceTransferSharedPagesTrimmedSuffix(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	sourcePages := []rocmDeviceKVPage{
		{
			tokenStart: 0,
			tokenCount: 1,
			key:        rocmDeviceKVTensor{pointer: 0x1001, sizeBytes: 4, encoding: rocmKVEncodingQ8},
			value:      rocmDeviceKVTensor{pointer: 0x1002, sizeBytes: 4, encoding: rocmKVEncodingQ4},
			owned:      true,
		},
		{
			tokenStart: 1,
			tokenCount: 1,
			key:        rocmDeviceKVTensor{pointer: 0x2001, sizeBytes: 4, encoding: rocmKVEncodingQ8},
			value:      rocmDeviceKVTensor{pointer: 0x2002, sizeBytes: 4, encoding: rocmKVEncodingQ4},
			owned:      true,
		},
		{
			tokenStart: 2,
			tokenCount: 1,
			key:        rocmDeviceKVTensor{pointer: 0x3001, sizeBytes: 4, encoding: rocmKVEncodingQ8},
			value:      rocmDeviceKVTensor{pointer: 0x3002, sizeBytes: 4, encoding: rocmKVEncodingQ4},
			owned:      true,
		},
	}
	targetPages := []rocmDeviceKVPage{
		sourcePages[1],
		sourcePages[2],
	}
	for index := range targetPages {
		targetPages[index].tokenStart = index
		targetPages[index].owned = false
	}
	source := rocmBorrowDeviceKVCache(driver, rocmKVCacheModeKQ8VQ4, 1, len(sourcePages), sourcePages, false)
	target := rocmBorrowDeviceKVCache(driver, rocmKVCacheModeKQ8VQ4, 1, len(targetPages), targetPages, false)

	core.RequireNoError(t, source.transferSharedPagesTo(target))

	core.AssertEqual(t, true, source.closed)
	core.AssertEqual(t, 0, len(source.pages))
	core.AssertEqual(t, []nativeDevicePointer{0x1001, 0x1002}, driver.frees)
	core.AssertEqual(t, true, target.pages[0].owned)
	core.AssertEqual(t, true, target.pages[1].owned)
	core.AssertEqual(t, nativeDevicePointer(0x2001), target.pages[0].key.pointer)
	core.AssertEqual(t, nativeDevicePointer(0x3002), target.pages[1].value.pointer)
	core.RequireNoError(t, target.Close())
}

func TestKVCache_Bad_DeviceMirrorAppendRollbackOnDescriptorFailure(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	driver := &fakeHIPDriver{available: true}
	device, table, err := hipMirrorTinyKV(driver, cache, map[string]string{})
	core.RequireNoError(t, err)
	defer device.Close()
	defer table.Close()
	driver.copyErr = core.NewError("descriptor copy failed")
	driver.copyErrAt = len(driver.copies) + 3

	next, nextTable, err := hipAppendDecodeDeviceKV(hipDecodeRequest{
		KV:              cache,
		DeviceKV:        device,
		DescriptorTable: table,
	}, []float32{-1, 1}, []float32{3, -3}, map[string]string{})

	core.AssertNil(t, next)
	core.AssertNil(t, nextTable)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy descriptor table")
	core.AssertEqual(t, 2, device.TokenCount())
	core.AssertEqual(t, false, device.closed)
	core.AssertEqual(t, false, table.closed)
	core.AssertEqual(t, 3, len(driver.frees))
	payload, err := device.Snapshot()
	core.RequireNoError(t, err)
	restored, err := newROCmKVCacheFromSnapshot(payload)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, restored.TokenCount())
}

func TestKVCache_Bad_DeviceMirrorAppendScratchCloseDoesNotFreeSourcePages(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	driver := &fakeHIPDriver{available: true}
	device, err := cache.MirrorToDevice(driver)
	core.RequireNoError(t, err)
	defer device.Close()
	next, err := device.withAppendedToken([]float32{-1, 1}, []float32{3, -3})
	core.RequireNoError(t, err)

	core.RequireNoError(t, next.Close())

	core.AssertEqual(t, false, device.closed)
	core.AssertEqual(t, 2, device.TokenCount())
	core.AssertEqual(t, 2, len(driver.frees))
	payload, err := device.Snapshot()
	core.RequireNoError(t, err)
	restored, err := newROCmKVCacheFromSnapshot(payload)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, restored.TokenCount())
}

func TestKVCache_Bad_RejectsInvalidModeRangeAndSnapshot(t *testing.T) {
	cache, err := newROCmKVCache("not-a-mode", 0)
	core.AssertNil(t, cache)
	core.AssertError(t, err)

	cache, err = newROCmKVCache(rocmKVCacheModeFP16, 2)
	core.RequireNoError(t, err)
	err = cache.Append(0, []float32{1}, []float32{})
	core.AssertError(t, err)
	err = cache.AppendVectors(0, 2, 1, []float32{1, 2, 3}, []float32{1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "vector widths")
	_, _, err = cache.Restore(0, 1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "cache block range")

	err = cache.Append(0, []float32{1, 2}, []float32{2, 1})
	core.RequireNoError(t, err)
	err = cache.Append(1, []float32{3, 4}, []float32{4, 3})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "overlap")
	core.AssertEqual(t, 2, cache.TokenCount())

	err = cache.AppendToken(2, []float32{3, 4}, []float32{4, 3})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV vector widths")
	core.AssertEqual(t, 2, cache.TokenCount())

	_, err = newROCmKVCacheFromSnapshot([]byte(`{"version":1,"mode":"q8","block_size":2,"blocks":[{"token_start":0,"token_count":2,"key":{"encoding":"q8","length":2,"scale":0,"q8":[1,2]},"value":{"encoding":"q8","length":2,"scale":1,"q8":[1,2]}}]}`))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "q8 scale")

	_, err = newROCmKVCacheFromSnapshot([]byte(`{"version":1,"mode":"fp16","block_size":2,"blocks":[{"token_start":0,"token_count":2,"key":{"encoding":"fp16","length":2,"f16":[15360,16384]},"value":{"encoding":"fp16","length":2,"f16":[15360,16384]}},{"token_start":1,"token_count":1,"key":{"encoding":"fp16","length":1,"f16":[16896]},"value":{"encoding":"fp16","length":1,"f16":[16896]}}]}`))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "overlap")

	_, err = newROCmKVCacheFromSnapshot([]byte(`{"version":1,"mode":"fp16","block_size":2,"blocks":[{"token_start":0,"token_count":1,"key_width":1,"value_width":1,"key":{"encoding":"fp16","length":1,"f16":[15360]},"value":{"encoding":"fp16","length":1,"f16":[15360]}},{"token_start":1,"token_count":1,"key_width":2,"value_width":1,"key":{"encoding":"fp16","length":2,"f16":[15360,16384]},"value":{"encoding":"fp16","length":1,"f16":[16896]}}]}`))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV vector widths")
}

func TestKVCache_Bad_DeviceMirrorRollbackOnCopyFailure(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.Append(0, []float32{1, 2}, []float32{3, 4}))
	driver := &fakeHIPDriver{available: true, copyErr: core.NewError("copy failed"), copyErrAt: 2}

	device, err := cache.MirrorToDevice(driver)

	core.AssertNil(t, device)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy KV value page")
	core.AssertEqual(t, []uint64{6, 6}, driver.allocations)
	core.AssertEqual(t, []uint64{6, 6}, driver.copies)
	core.AssertEqual(t, 2, len(driver.frees))
	core.AssertEqual(t, 2, cache.TokenCount())

	device, err = cache.MirrorToDevice(nil)
	core.AssertNil(t, device)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "HIP driver is nil")
}

func TestKVCache_Bad_DeviceMirrorSnapshotRejectsClosedAndCopyFailure(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.Append(0, []float32{1, 2}, []float32{3, 4}))
	driver := &fakeHIPDriver{available: true}
	device, err := cache.MirrorToDevice(driver)
	core.RequireNoError(t, err)
	driver.copyErr = core.NewError("device read failed")
	driver.copyErrAt = len(driver.copies) + 1

	payload, err := device.Snapshot()

	core.AssertNil(t, payload)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy KV key page")

	driver.copyErr = nil
	driver.copyErrAt = 0
	core.RequireNoError(t, device.Close())
	payload, err = device.Snapshot()
	core.AssertNil(t, payload)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "closed")
}

func TestKVCache_Bad_DeviceDescriptorTableRollbackOnCopyFailure(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.Append(0, []float32{1, 2}, []float32{3, 4}))
	driver := &fakeHIPDriver{available: true}
	device, err := cache.MirrorToDevice(driver)
	core.RequireNoError(t, err)
	defer device.Close()
	driver.copyErr = core.NewError("descriptor copy failed")
	driver.copyErrAt = len(driver.copies) + 1

	table, err := device.KernelDescriptorTable()

	core.AssertNil(t, table)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "copy descriptor table")
	core.AssertEqual(t, []uint64{6, 6, uint64(rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVDescriptorPageBytes)}, driver.allocations)
	core.AssertEqual(t, []uint64{6, 6, uint64(rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVDescriptorPageBytes)}, driver.copies)
	core.AssertEqual(t, 1, len(driver.frees))

	core.RequireNoError(t, device.Close())
	_, err = device.KernelDescriptorTable()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "closed")

	noDriver := &rocmDeviceKVCache{mode: rocmKVCacheModeQ8, blockSize: 1}
	table, err = noDriver.KernelDescriptorTable()
	core.AssertNil(t, table)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "HIP driver is nil")
}

func TestKVCache_DevicePageSliceCapacity_Good(t *testing.T) {
	core.AssertEqual(t, 0, rocmDeviceKVPageSliceCapacity(0))
	core.AssertEqual(t, rocmDeviceKVHotPageCapacity, rocmDeviceKVPageSliceCapacity(1))
	core.AssertEqual(t, rocmDeviceKVHotPageCapacity, rocmDeviceKVPageSliceCapacity(rocmDeviceKVHotPageCapacity))
	core.AssertEqual(t, rocmDeviceKVHotPageCapacity*2, rocmDeviceKVPageSliceCapacity(rocmDeviceKVHotPageCapacity+1))
	core.AssertEqual(t, rocmDeviceKVPagePoolMaxCapacity, rocmDeviceKVPageSliceCapacity(rocmDeviceKVPagePoolMaxCapacity-1))
	core.AssertEqual(t, rocmDeviceKVPagePoolMaxCapacity+1, rocmDeviceKVPageSliceCapacity(rocmDeviceKVPagePoolMaxCapacity+1))
}

func TestKVCache_DeviceDescriptorTableAllocationBytes_Good(t *testing.T) {
	descriptorBytes := func(pages int) uint64 {
		return uint64(rocmDeviceKVDescriptorHeaderBytes + pages*rocmDeviceKVDescriptorPageBytes)
	}
	core.AssertEqual(t, uint64(rocmDeviceKVDescriptorHeaderBytes), rocmDeviceKVDescriptorTableAllocationBytes(uint64(rocmDeviceKVDescriptorHeaderBytes)))
	core.AssertEqual(t, rocmDeviceKVDescriptorHotTableBytes(), rocmDeviceKVDescriptorTableAllocationBytes(descriptorBytes(1)))
	core.AssertEqual(t, rocmDeviceKVDescriptorHotTableBytes(), rocmDeviceKVDescriptorTableAllocationBytes(descriptorBytes(rocmDeviceKVHotPageCapacity)))
	core.AssertEqual(t, descriptorBytes(rocmDeviceKVHotPageCapacity*2), rocmDeviceKVDescriptorTableAllocationBytes(descriptorBytes(rocmDeviceKVHotPageCapacity+1)))
	core.AssertEqual(t, descriptorBytes(rocmDeviceKVPagePoolMaxCapacity), rocmDeviceKVDescriptorTableAllocationBytes(descriptorBytes(rocmDeviceKVPagePoolMaxCapacity-1)))
	core.AssertEqual(t, descriptorBytes(rocmDeviceKVPagePoolMaxCapacity+1), rocmDeviceKVDescriptorTableAllocationBytes(descriptorBytes(rocmDeviceKVPagePoolMaxCapacity+1)))
}

func TestKVCache_DeviceDescriptorTableLogicalAndAllocationBytes_Good(t *testing.T) {
	logicalBytes := uint64(rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVDescriptorPageBytes)
	allocationBytes := rocmDeviceKVDescriptorTableAllocationBytes(logicalBytes)
	table := rocmBorrowDeviceKVDescriptorTableAllocated(&fakeHIPDriver{available: true}, 4096, logicalBytes, allocationBytes, rocmDeviceKVDescriptorVersion, 1, false, true)
	core.AssertEqual(t, logicalBytes, table.SizeBytes())
	core.AssertEqual(t, allocationBytes, table.AllocationBytes())
	rocmReleaseDeviceKVDescriptorTable(table)
}

func TestKVCache_Bad_DeviceDescriptorBytesRejectUnsupportedABIValues(t *testing.T) {
	validPage := rocmDeviceKVPageDescriptor{
		TokenStart:    0,
		TokenCount:    1,
		KeyWidth:      2,
		ValueWidth:    2,
		KeyPointer:    1,
		ValuePointer:  2,
		KeyBytes:      8,
		ValueBytes:    8,
		KeyEncoding:   rocmKVEncodingQ8,
		ValueEncoding: rocmKVEncodingQ8,
	}
	_, err := (rocmDeviceKVDescriptor{
		Mode:       "not-a-mode",
		BlockSize:  1,
		TokenCount: 1,
		Pages:      []rocmDeviceKVPageDescriptor{validPage},
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported cache mode")

	badEncoding := validPage
	badEncoding.ValueEncoding = "packed"
	_, err = (rocmDeviceKVDescriptor{
		Mode:       rocmKVCacheModeQ8,
		BlockSize:  1,
		TokenCount: 1,
		Pages:      []rocmDeviceKVPageDescriptor{badEncoding},
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported tensor encoding")

	nilPointer := validPage
	nilPointer.KeyPointer = 0
	_, err = (rocmDeviceKVDescriptor{
		Mode:       rocmKVCacheModeQ8,
		BlockSize:  1,
		TokenCount: 1,
		Pages:      []rocmDeviceKVPageDescriptor{nilPointer},
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "nil pointer")

	_, err = (rocmDeviceKVDescriptor{
		Mode:       rocmKVCacheModeQ8,
		BlockSize:  -1,
		TokenCount: 1,
		Pages:      []rocmDeviceKVPageDescriptor{validPage},
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "block size")

	zeroWidth := validPage
	zeroWidth.KeyWidth = 0
	_, err = (rocmDeviceKVDescriptor{
		Mode:       rocmKVCacheModeQ8,
		BlockSize:  1,
		TokenCount: 1,
		Pages:      []rocmDeviceKVPageDescriptor{zeroWidth},
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "key width")

	outOfRange := validPage
	outOfRange.TokenStart = 1
	_, err = (rocmDeviceKVDescriptor{
		Mode:       rocmKVCacheModeQ8,
		BlockSize:  1,
		TokenCount: 1,
		Pages:      []rocmDeviceKVPageDescriptor{outOfRange},
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token range")

	overlap := validPage
	overlap.TokenStart = 0
	_, err = (rocmDeviceKVDescriptor{
		Mode:       rocmKVCacheModeQ8,
		BlockSize:  1,
		TokenCount: 2,
		Pages:      []rocmDeviceKVPageDescriptor{validPage, overlap},
	}).Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "non-overlap")

	validLaunch := rocmDeviceKVLaunchDescriptor{
		DescriptorPointer: 1,
		DescriptorBytes:   uint64(rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVDescriptorPageBytes),
		DescriptorVersion: rocmDeviceKVDescriptorVersion,
		Mode:              rocmKVCacheModeQ8,
		ModeCode:          rocmDeviceKVDescriptorModeQ8,
		BlockSize:         2,
		PageCount:         1,
		TokenCount:        1,
		KeyWidth:          2,
		ValueWidth:        2,
	}
	badLaunch := validLaunch
	badLaunch.DescriptorPointer = 0
	_, err = badLaunch.Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "descriptor pointer")

	badLaunch = validLaunch
	badLaunch.ModeCode = 99
	_, err = badLaunch.Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "mode code")

	badLaunch = validLaunch
	badLaunch.ModeCode = rocmDeviceKVDescriptorModeFP16
	_, err = badLaunch.Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "mode code mismatch")

	badLaunch = validLaunch
	badLaunch.KeyWidth = 0
	_, err = badLaunch.Binary()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "key width")
}

func BenchmarkROCmKVCacheBlockFromRawPayload_KQ8VQ4Page(b *testing.B) {
	payload := benchmarkROCmKVRawPayload(b)
	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		block, err := rocmKVCacheBlockFromRawPayload(payload)
		if err != nil {
			b.Fatalf("decode raw KV block: %v", err)
		}
		if block.tokenCount != 512 || block.keyWidth != 128 || block.valueWidth != 128 {
			b.Fatalf("decoded block metadata = tokens:%d key:%d value:%d", block.tokenCount, block.keyWidth, block.valueWidth)
		}
	}
}

func BenchmarkROCmDeviceKVPageFromRawPayload_KQ8VQ4PinnedCopy(b *testing.B) {
	payload := benchmarkROCmKVRawPayload(b)
	driver := &fakeHIPDriver{available: true}
	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		page, err := rocmDeviceKVPageFromRawPayload(driver, payload)
		if err != nil {
			b.Fatalf("restore raw KV block to device: %v", err)
		}
		if page.tokenCount != 512 || page.keyWidth != 128 || page.valueWidth != 128 {
			b.Fatalf("device page metadata = tokens:%d key:%d value:%d", page.tokenCount, page.keyWidth, page.valueWidth)
		}
		cache := &rocmDeviceKVCache{
			driver:     driver,
			mode:       rocmKVCacheModeKQ8VQ4,
			blockSize:  page.tokenCount,
			pages:      []rocmDeviceKVPage{page},
			tokenCount: page.tokenCount,
		}
		if err := cache.Close(); err != nil {
			b.Fatalf("close restored device page: %v", err)
		}
	}
}

func BenchmarkROCmDeviceKVDescriptorTablePool_Reused(b *testing.B) {
	rocmDeviceKVDescriptorTablePool.Lock()
	rocmDeviceKVDescriptorTablePool.entries = nil
	rocmDeviceKVDescriptorTablePool.Unlock()
	driver := &fakeHIPDriver{available: true}
	table := rocmBorrowDeviceKVDescriptorTable(driver, 4096, rocmDeviceKVDescriptorHeaderBytes+rocmDeviceKVDescriptorPageBytes, rocmDeviceKVDescriptorVersion, 1, false, true)
	rocmReleaseDeviceKVDescriptorTable(table)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		table = rocmBorrowDeviceKVDescriptorTable(driver, 4096, rocmDeviceKVDescriptorHeaderBytes+rocmDeviceKVDescriptorPageBytes, rocmDeviceKVDescriptorVersion, 1, false, true)
		if table.Pointer() != 4096 || table.SizeBytes() != rocmDeviceKVDescriptorHeaderBytes+rocmDeviceKVDescriptorPageBytes || table.pageCount != 1 {
			b.Fatalf("descriptor table = ptr:%d bytes:%d pages:%d", table.Pointer(), table.SizeBytes(), table.pageCount)
		}
		rocmReleaseDeviceKVDescriptorTable(table)
	}
}

func BenchmarkROCmDeviceKVDescriptorPointerPool_HotWindow(b *testing.B) {
	rocmDeviceKVDescriptorPointerPool.Lock()
	rocmDeviceKVDescriptorPointerPool.entries = make(map[uint64][]rocmDeviceKVDescriptorPointerPoolEntry)
	rocmDeviceKVDescriptorPointerPool.bytes = 0
	rocmDeviceKVDescriptorPointerPool.Unlock()
	driver := &fakeHIPDriver{available: true}
	sizeBytes := rocmDeviceKVDescriptorHotTableBytes()
	pointer, allocationBytes, err := rocmDeviceKVDescriptorTableMalloc(driver, sizeBytes)
	if err != nil {
		b.Fatalf("descriptor malloc: %v", err)
	}
	if allocationBytes != sizeBytes {
		b.Fatalf("descriptor allocation bytes = %d, want %d", allocationBytes, sizeBytes)
	}
	if err := rocmDeviceKVDescriptorTableFree(driver, pointer, allocationBytes); err != nil {
		b.Fatalf("descriptor free: %v", err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pointer, allocationBytes, err = rocmDeviceKVDescriptorTableMalloc(driver, sizeBytes)
		if err != nil {
			b.Fatalf("descriptor malloc: %v", err)
		}
		if pointer == 0 {
			b.Fatalf("descriptor pointer is nil")
		}
		if allocationBytes != sizeBytes {
			b.Fatalf("descriptor allocation bytes = %d, want %d", allocationBytes, sizeBytes)
		}
		if err := rocmDeviceKVDescriptorTableFree(driver, pointer, allocationBytes); err != nil {
			b.Fatalf("descriptor free: %v", err)
		}
	}
}

func BenchmarkROCmDeviceKVCacheKernelDescriptorBytes_HotWindow(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	const (
		keyWidth   = 128
		valueWidth = 128
	)
	keyBytes, err := rocmKVTensorDeviceByteCount(rocmKVEncodingQ8, keyWidth)
	if err != nil {
		b.Fatalf("key bytes: %v", err)
	}
	valueBytes, err := rocmKVTensorDeviceByteCount(rocmKVEncodingQ4, valueWidth)
	if err != nil {
		b.Fatalf("value bytes: %v", err)
	}
	pages := rocmDeviceKVBorrowPageSlice(0, rocmDeviceKVHotPageCapacity)
	for token := 0; token < rocmDeviceKVHotPageCapacity; token++ {
		pages = append(pages, rocmDeviceKVPage{
			tokenStart: token,
			tokenCount: 1,
			keyWidth:   keyWidth,
			valueWidth: valueWidth,
			key:        rocmDeviceKVTensor{pointer: nativeDevicePointer(0x100000 + token*0x1000), sizeBytes: keyBytes, encoding: rocmKVEncodingQ8},
			value:      rocmDeviceKVTensor{pointer: nativeDevicePointer(0x200000 + token*0x1000), sizeBytes: valueBytes, encoding: rocmKVEncodingQ4},
		})
	}
	cache := rocmBorrowDeviceKVCache(driver, rocmKVCacheModeKQ8VQ4, rocmGemma4Q4DeviceKVBlockSize, rocmDeviceKVHotPageCapacity, pages, false)
	b.Cleanup(func() {
		rocmDeviceKVReleasePageSlice(cache.pages)
		cache.pages = nil
		rocmReleaseDeviceKVCache(cache)
	})
	wantBytes := rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVHotPageCapacity*rocmDeviceKVDescriptorPageBytes
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		payload, err := cache.KernelDescriptorBytes()
		if err != nil {
			b.Fatalf("descriptor bytes: %v", err)
		}
		if len(payload) != wantBytes {
			b.Fatalf("descriptor bytes len = %d, want %d", len(payload), wantBytes)
		}
	}
}

func BenchmarkROCmDeviceKVCacheKernelDescriptorTable_HotWindowPooled(b *testing.B) {
	rocmDeviceKVDescriptorPointerPool.Lock()
	rocmDeviceKVDescriptorPointerPool.entries = make(map[uint64][]rocmDeviceKVDescriptorPointerPoolEntry)
	rocmDeviceKVDescriptorPointerPool.bytes = 0
	rocmDeviceKVDescriptorPointerPool.Unlock()
	rocmDeviceKVDescriptorBytePools.Range(func(key, _ any) bool {
		rocmDeviceKVDescriptorBytePools.Delete(key)
		return true
	})
	driver := &fakeHIPDriver{available: true}
	const (
		keyWidth   = 128
		valueWidth = 128
	)
	keyBytes, err := rocmKVTensorDeviceByteCount(rocmKVEncodingQ8, keyWidth)
	if err != nil {
		b.Fatalf("key bytes: %v", err)
	}
	valueBytes, err := rocmKVTensorDeviceByteCount(rocmKVEncodingQ4, valueWidth)
	if err != nil {
		b.Fatalf("value bytes: %v", err)
	}
	pages := rocmDeviceKVBorrowPageSlice(0, rocmDeviceKVHotPageCapacity)
	for token := 0; token < rocmDeviceKVHotPageCapacity; token++ {
		pages = append(pages, rocmDeviceKVPage{
			tokenStart: token,
			tokenCount: 1,
			keyWidth:   keyWidth,
			valueWidth: valueWidth,
			key:        rocmDeviceKVTensor{pointer: nativeDevicePointer(0x100000 + token*0x1000), sizeBytes: keyBytes, encoding: rocmKVEncodingQ8},
			value:      rocmDeviceKVTensor{pointer: nativeDevicePointer(0x200000 + token*0x1000), sizeBytes: valueBytes, encoding: rocmKVEncodingQ4},
		})
	}
	cache := rocmBorrowDeviceKVCache(driver, rocmKVCacheModeKQ8VQ4, rocmGemma4Q4DeviceKVBlockSize, rocmDeviceKVHotPageCapacity, pages, false)
	warm, err := cache.KernelDescriptorTable()
	if err != nil {
		b.Fatalf("warm descriptor table: %v", err)
	}
	if err := warm.Close(); err != nil {
		b.Fatalf("close warm descriptor table: %v", err)
	}
	allocationsAfterWarm := len(driver.allocations)
	b.Cleanup(func() {
		rocmDeviceKVReleasePageSlice(cache.pages)
		cache.pages = nil
		rocmReleaseDeviceKVCache(cache)
	})
	wantBytes := uint64(rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVHotPageCapacity*rocmDeviceKVDescriptorPageBytes)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		table, err := cache.KernelDescriptorTable()
		if err != nil {
			b.Fatalf("descriptor table: %v", err)
		}
		if table.SizeBytes() != wantBytes || table.pageCount != rocmDeviceKVHotPageCapacity {
			b.Fatalf("descriptor table shape = %d/%d, want %d/%d", table.SizeBytes(), table.pageCount, wantBytes, rocmDeviceKVHotPageCapacity)
		}
		if err := table.Close(); err != nil {
			b.Fatalf("close descriptor table: %v", err)
		}
		if len(driver.allocations) != allocationsAfterWarm {
			b.Fatalf("descriptor table used fresh device allocation: got %d allocations, want %d", len(driver.allocations), allocationsAfterWarm)
		}
	}
}

func BenchmarkROCmDeviceKVDescriptorAppendInPlace_HotWindow(b *testing.B) {
	driver := &fakeHIPDriver{available: true, skipLaunchRecording: true, releaseLaunchPackets: true}
	const (
		keyWidth   = 128
		valueWidth = 128
	)
	keyBytes, err := rocmKVTensorDeviceByteCount(rocmKVEncodingQ8, keyWidth)
	if err != nil {
		b.Fatalf("key bytes: %v", err)
	}
	valueBytes, err := rocmKVTensorDeviceByteCount(rocmKVEncodingQ4, valueWidth)
	if err != nil {
		b.Fatalf("value bytes: %v", err)
	}
	pages := rocmDeviceKVBorrowPageSlice(0, rocmDeviceKVHotPageCapacity-1)
	for token := 0; token < rocmDeviceKVHotPageCapacity-1; token++ {
		pages = append(pages, rocmDeviceKVPage{
			tokenStart: token,
			tokenCount: 1,
			keyWidth:   keyWidth,
			valueWidth: valueWidth,
			key:        rocmDeviceKVTensor{pointer: nativeDevicePointer(0x100000 + token*0x1000), sizeBytes: keyBytes, encoding: rocmKVEncodingQ8},
			value:      rocmDeviceKVTensor{pointer: nativeDevicePointer(0x200000 + token*0x1000), sizeBytes: valueBytes, encoding: rocmKVEncodingQ4},
		})
	}
	previous := rocmBorrowDeviceKVCache(driver, rocmKVCacheModeKQ8VQ4, rocmGemma4Q4DeviceKVBlockSize, rocmDeviceKVHotPageCapacity-1, pages, false)
	nextPages := rocmDeviceKVCopyPagesWithExtra(previous.pages, 1)
	nextPages = append(nextPages, rocmDeviceKVPage{
		tokenStart: rocmDeviceKVHotPageCapacity - 1,
		tokenCount: 1,
		keyWidth:   keyWidth,
		valueWidth: valueWidth,
		key:        rocmDeviceKVTensor{pointer: 0x300000, sizeBytes: keyBytes, encoding: rocmKVEncodingQ8},
		value:      rocmDeviceKVTensor{pointer: 0x400000, sizeBytes: valueBytes, encoding: rocmKVEncodingQ4},
		owned:      true,
	})
	next := rocmBorrowDeviceKVCache(driver, rocmKVCacheModeKQ8VQ4, rocmGemma4Q4DeviceKVBlockSize, rocmDeviceKVHotPageCapacity, nextPages, false)
	payload, err := previous.KernelDescriptorBytes()
	if err != nil {
		b.Fatalf("descriptor bytes: %v", err)
	}
	pointer, allocationBytes, err := rocmDeviceKVDescriptorTableMalloc(driver, uint64(len(payload)))
	if err != nil {
		b.Fatalf("descriptor malloc: %v", err)
	}
	if err := hipCopyHostToDevice(driver, pointer, payload); err != nil {
		b.Fatalf("copy descriptor: %v", err)
	}
	table := rocmBorrowDeviceKVDescriptorTableAllocated(driver, pointer, uint64(len(payload)), allocationBytes, rocmDeviceKVDescriptorVersion, previous.PageCount(), false, true)
	b.Cleanup(func() {
		_ = table.Close()
		rocmDeviceKVReleasePageSlice(next.pages)
		next.pages = nil
		rocmReleaseDeviceKVCache(next)
		rocmDeviceKVReleasePageSlice(previous.pages)
		previous.pages = nil
		rocmReleaseDeviceKVCache(previous)
	})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		table.sizeBytes = uint64(len(payload))
		table.pageCount = previous.PageCount()
		target, offset, ok := driver.memoryForPointer(pointer, len(payload))
		if !ok {
			b.Fatalf("descriptor pointer is missing")
		}
		copy(target[offset:], payload)
		out, err := next.KernelDescriptorTableFromAppendedToken(context.Background(), previous, table)
		if err != nil {
			b.Fatalf("append descriptor in place: %v", err)
		}
		if out != table {
			b.Fatalf("descriptor table was not reused in place")
		}
		if table.pageCount != next.PageCount() || table.SizeBytes() != uint64(rocmDeviceKVDescriptorHeaderBytes+next.PageCount()*rocmDeviceKVDescriptorPageBytes) {
			b.Fatalf("descriptor shape = pages:%d bytes:%d", table.pageCount, table.SizeBytes())
		}
	}
}

func BenchmarkROCmDeviceKVAppendEncodedTokenWindow_Hot(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	const (
		keyWidth   = 128
		valueWidth = 128
	)
	keyBytes, err := rocmKVTensorDeviceByteCount(rocmKVEncodingQ8, keyWidth)
	if err != nil {
		b.Fatalf("key bytes: %v", err)
	}
	valueBytes, err := rocmKVTensorDeviceByteCount(rocmKVEncodingQ4, valueWidth)
	if err != nil {
		b.Fatalf("value bytes: %v", err)
	}
	pages := rocmDeviceKVBorrowPageSlice(0, rocmDeviceKVHotPageCapacity)
	for token := 0; token < rocmDeviceKVHotPageCapacity; token++ {
		pages = append(pages, rocmDeviceKVPage{
			tokenStart: token,
			tokenCount: 1,
			keyWidth:   keyWidth,
			valueWidth: valueWidth,
			key:        rocmDeviceKVTensor{pointer: nativeDevicePointer(0x100000 + token*0x1000), sizeBytes: keyBytes, encoding: rocmKVEncodingQ8},
			value:      rocmDeviceKVTensor{pointer: nativeDevicePointer(0x200000 + token*0x1000), sizeBytes: valueBytes, encoding: rocmKVEncodingQ4},
		})
	}
	cache := rocmBorrowDeviceKVCache(driver, rocmKVCacheModeKQ8VQ4, rocmGemma4Q4DeviceKVBlockSize, rocmDeviceKVHotPageCapacity, pages, false)
	key := rocmDeviceKVTensor{pointer: 0x300000, sizeBytes: keyBytes, encoding: rocmKVEncodingQ8}
	value := rocmDeviceKVTensor{pointer: 0x400000, sizeBytes: valueBytes, encoding: rocmKVEncodingQ4}
	b.Cleanup(func() {
		rocmDeviceKVReleasePageSlice(cache.pages)
		cache.pages = nil
		rocmReleaseDeviceKVCache(cache)
	})
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		next, err := cache.withAppendedEncodedTokenWindow(key, value, keyWidth, valueWidth, rocmDeviceKVHotPageCapacity)
		if err != nil {
			b.Fatalf("append token window: %v", err)
		}
		if next.TokenCount() != rocmDeviceKVHotPageCapacity || len(next.pages) != rocmDeviceKVHotPageCapacity || cap(next.pages) != rocmDeviceKVHotPageCapacity {
			b.Fatalf("next cache tokens/pages/cap = %d/%d/%d", next.TokenCount(), len(next.pages), cap(next.pages))
		}
		rocmDeviceKVReleasePageSlice(next.pages)
		next.pages = nil
		rocmReleaseDeviceKVCache(next)
	}
}

func benchmarkROCmKVRawPayload(tb testing.TB) []byte {
	tb.Helper()
	const (
		tokens     = 512
		keyWidth   = 128
		valueWidth = 128
	)
	keys := make([]float32, tokens*keyWidth)
	values := make([]float32, tokens*valueWidth)
	for index := range keys {
		keys[index] = float32((index%251)-125) / 125.0
	}
	for index := range values {
		values[index] = float32((index%197)-98) / 98.0
	}
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, tokens)
	if err != nil {
		tb.Fatalf("create KV cache: %v", err)
	}
	if err := cache.AppendVectors(0, keyWidth, valueWidth, keys, values); err != nil {
		tb.Fatalf("append KV vectors: %v", err)
	}
	payload, err := cache.rawBlock(cache.blocks[0])
	if err != nil {
		tb.Fatalf("encode raw KV block: %v", err)
	}
	return payload
}

func assertFloat32SlicesNear(t *testing.T, want, got []float32, tolerance float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("slice len = %d, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if math.Abs(float64(want[i]-got[i])) > float64(tolerance) {
			t.Fatalf("slice[%d] = %f, want %f within %f; got %+v", i, got[i], want[i], tolerance, got)
		}
	}
}

func mustHIPFloat32Payload(t *testing.T, values []float32) []byte {
	t.Helper()
	payload, err := hipFloat32Payload(values)
	core.RequireNoError(t, err)
	return payload
}
