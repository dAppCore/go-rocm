// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"strconv"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const (
	rocmDeviceKVDescriptorVersion          uint32 = 1
	rocmDeviceKVDescriptorHeaderBytes             = 32
	rocmDeviceKVDescriptorPageBytes               = 64
	rocmDeviceKVLaunchDescriptorBytes             = 64
	hipKVEncodeTokenLaunchArgsVersion      uint32 = 1
	hipKVEncodeTokenLaunchArgsBytes               = 96
	hipKVEncodeTokenBlockSize              uint32 = 256
	hipKVDescriptorAppendLaunchArgsVersion uint32 = 1
	hipKVDescriptorAppendLaunchArgsBytes          = 128
	hipKVDescriptorAppendBlockSize         uint32 = 64
)

const (
	rocmDeviceKVDescriptorModeFP16   uint32 = 1
	rocmDeviceKVDescriptorModeQ8     uint32 = 2
	rocmDeviceKVDescriptorModeKQ8VQ4 uint32 = 3
)

const (
	rocmDeviceKVHotPageCapacity        = 512
	rocmDeviceKVPagePoolMaxCapacity    = 128 * 1024
	rocmDeviceKVDescriptorTablePoolMax = 4096
	rocmGemma4Q4DeviceKVBlockSize      = 1
)

const (
	rocmDeviceKVDescriptorPointerPoolMaxBytes   = 32 << 20
	rocmDeviceKVDescriptorPointerPoolMaxPerSize = 4096
)

const (
	rocmDeviceKVDescriptorEncodingFP16 uint32 = 1
	rocmDeviceKVDescriptorEncodingQ8   uint32 = 2
	rocmDeviceKVDescriptorEncodingQ4   uint32 = 3
)

type rocmDeviceKVCache struct {
	driver     nativeHIPDriver
	mode       string
	blockSize  int
	pages      []rocmDeviceKVPage
	tokenCount int
	closed     bool
	borrowed   bool
}

var rocmDeviceKVCachePool = sync.Pool{
	New: func() any { return &rocmDeviceKVCache{} },
}

func rocmBorrowDeviceKVCache(driver nativeHIPDriver, mode string, blockSize, tokenCount int, pages []rocmDeviceKVPage, borrowed bool) *rocmDeviceKVCache {
	cache := rocmDeviceKVCachePool.Get().(*rocmDeviceKVCache)
	*cache = rocmDeviceKVCache{
		driver:     driver,
		mode:       mode,
		blockSize:  blockSize,
		pages:      pages,
		tokenCount: tokenCount,
		borrowed:   borrowed,
	}
	return cache
}

func rocmReleaseDeviceKVCache(cache *rocmDeviceKVCache) {
	if cache == nil {
		return
	}
	*cache = rocmDeviceKVCache{}
	rocmDeviceKVCachePool.Put(cache)
}

type rocmDeviceKVPage struct {
	tokenStart int
	tokenCount int
	keyWidth   int
	valueWidth int
	key        rocmDeviceKVTensor
	value      rocmDeviceKVTensor
	owned      bool
}

type rocmDeviceKVTensor struct {
	pointer   nativeDevicePointer
	sizeBytes uint64
	encoding  string
}

type rocmDeviceKVDescriptorTable struct {
	driver          nativeHIPDriver
	pointer         nativeDevicePointer
	sizeBytes       uint64
	allocationBytes uint64
	version         uint32
	pageCount       int
	closed          bool
	borrowed        bool
	poolable        bool
}

var rocmDeviceKVDescriptorTablePool = struct {
	sync.Mutex
	entries []*rocmDeviceKVDescriptorTable
}{}

type rocmDeviceKVDescriptorPointerPoolEntry struct {
	driver  nativeHIPDriver
	pointer nativeDevicePointer
}

var rocmDeviceKVDescriptorPointerPool = struct {
	sync.Mutex
	entries map[uint64][]rocmDeviceKVDescriptorPointerPoolEntry
	bytes   uint64
}{
	entries: make(map[uint64][]rocmDeviceKVDescriptorPointerPoolEntry),
}

type rocmDeviceKVLaunchDescriptor struct {
	DescriptorPointer nativeDevicePointer
	DescriptorBytes   uint64
	DescriptorVersion uint32
	Mode              string
	ModeCode          uint32
	BlockSize         int
	PageCount         int
	TokenCount        int
	KeyWidth          int
	ValueWidth        int
	StatusPointer     nativeDevicePointer
	StatusValue       uint32
}

func rocmBorrowDeviceKVDescriptorTable(driver nativeHIPDriver, pointer nativeDevicePointer, sizeBytes uint64, version uint32, pageCount int, borrowed, poolable bool) *rocmDeviceKVDescriptorTable {
	return rocmBorrowDeviceKVDescriptorTableAllocated(driver, pointer, sizeBytes, sizeBytes, version, pageCount, borrowed, poolable)
}

func rocmBorrowDeviceKVDescriptorTableAllocated(driver nativeHIPDriver, pointer nativeDevicePointer, sizeBytes, allocationBytes uint64, version uint32, pageCount int, borrowed, poolable bool) *rocmDeviceKVDescriptorTable {
	var table *rocmDeviceKVDescriptorTable
	if poolable {
		rocmDeviceKVDescriptorTablePool.Lock()
		count := len(rocmDeviceKVDescriptorTablePool.entries)
		if count > 0 {
			table = rocmDeviceKVDescriptorTablePool.entries[count-1]
			rocmDeviceKVDescriptorTablePool.entries[count-1] = nil
			rocmDeviceKVDescriptorTablePool.entries = rocmDeviceKVDescriptorTablePool.entries[:count-1]
		}
		rocmDeviceKVDescriptorTablePool.Unlock()
	}
	if table == nil {
		table = &rocmDeviceKVDescriptorTable{}
	}
	if allocationBytes == 0 {
		allocationBytes = sizeBytes
	}
	*table = rocmDeviceKVDescriptorTable{
		driver:          driver,
		pointer:         pointer,
		sizeBytes:       sizeBytes,
		allocationBytes: allocationBytes,
		version:         version,
		pageCount:       pageCount,
		borrowed:        borrowed,
		poolable:        poolable,
	}
	return table
}

func rocmReleaseDeviceKVDescriptorTable(table *rocmDeviceKVDescriptorTable) {
	if table == nil {
		return
	}
	*table = rocmDeviceKVDescriptorTable{}
	rocmDeviceKVDescriptorTablePool.Lock()
	if len(rocmDeviceKVDescriptorTablePool.entries) < rocmDeviceKVDescriptorTablePoolMax {
		rocmDeviceKVDescriptorTablePool.entries = append(rocmDeviceKVDescriptorTablePool.entries, table)
	}
	rocmDeviceKVDescriptorTablePool.Unlock()
}

func rocmDeviceKVDescriptorHotTableBytes() uint64 {
	return uint64(rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVHotPageCapacity*rocmDeviceKVDescriptorPageBytes)
}

func rocmDeviceKVDescriptorTableAllocationBytes(sizeBytes uint64) uint64 {
	if sizeBytes <= uint64(rocmDeviceKVDescriptorHeaderBytes) {
		return sizeBytes
	}
	pageBytes := uint64(rocmDeviceKVDescriptorPageBytes)
	pageCount := int((sizeBytes - uint64(rocmDeviceKVDescriptorHeaderBytes) + pageBytes - 1) / pageBytes)
	pageCapacity := rocmDeviceKVPageSliceCapacity(pageCount)
	if pageCapacity > rocmDeviceKVPagePoolMaxCapacity {
		return sizeBytes
	}
	return uint64(rocmDeviceKVDescriptorHeaderBytes + pageCapacity*rocmDeviceKVDescriptorPageBytes)
}

func rocmDeviceKVDescriptorPointerPoolable(sizeBytes uint64) bool {
	return sizeBytes >= rocmDeviceKVDescriptorHotTableBytes() &&
		sizeBytes <= uint64(rocmDeviceKVDescriptorHeaderBytes+rocmDeviceKVPagePoolMaxCapacity*rocmDeviceKVDescriptorPageBytes)
}

func rocmDeviceKVDescriptorTableMalloc(driver nativeHIPDriver, sizeBytes uint64) (nativeDevicePointer, uint64, error) {
	if driver == nil {
		return 0, 0, core.E("rocm.KVCache.DeviceDescriptor", "HIP driver is nil", nil)
	}
	allocationBytes := rocmDeviceKVDescriptorTableAllocationBytes(sizeBytes)
	if rocmDeviceKVDescriptorPointerPoolable(allocationBytes) {
		rocmDeviceKVDescriptorPointerPool.Lock()
		entries := rocmDeviceKVDescriptorPointerPool.entries[allocationBytes]
		for index := len(entries) - 1; index >= 0; index-- {
			entry := entries[index]
			if entry.driver != driver {
				continue
			}
			entries[index] = entries[len(entries)-1]
			entries[len(entries)-1] = rocmDeviceKVDescriptorPointerPoolEntry{}
			entries = entries[:len(entries)-1]
			rocmDeviceKVDescriptorPointerPool.entries[allocationBytes] = entries
			rocmDeviceKVDescriptorPointerPool.bytes -= allocationBytes
			rocmDeviceKVDescriptorPointerPool.Unlock()
			return entry.pointer, allocationBytes, nil
		}
		rocmDeviceKVDescriptorPointerPool.Unlock()
	}
	pointer, err := driver.Malloc(allocationBytes)
	return pointer, allocationBytes, err
}

func rocmDeviceKVDescriptorTableFree(driver nativeHIPDriver, pointer nativeDevicePointer, sizeBytes uint64) error {
	if pointer == 0 {
		return nil
	}
	if driver == nil {
		return core.E("rocm.KVCache.DeviceDescriptor", "HIP driver is nil", nil)
	}
	if rocmDeviceKVDescriptorPointerPoolable(sizeBytes) {
		rocmDeviceKVDescriptorPointerPool.Lock()
		entries := rocmDeviceKVDescriptorPointerPool.entries[sizeBytes]
		if rocmDeviceKVDescriptorPointerPool.bytes+sizeBytes <= rocmDeviceKVDescriptorPointerPoolMaxBytes &&
			len(entries) < rocmDeviceKVDescriptorPointerPoolMaxPerSize {
			rocmDeviceKVDescriptorPointerPool.entries[sizeBytes] = append(entries, rocmDeviceKVDescriptorPointerPoolEntry{
				driver:  driver,
				pointer: pointer,
			})
			rocmDeviceKVDescriptorPointerPool.bytes += sizeBytes
			rocmDeviceKVDescriptorPointerPool.Unlock()
			return nil
		}
		rocmDeviceKVDescriptorPointerPool.Unlock()
	}
	return driver.Free(pointer)
}

type hipKVEncodeTokenLaunchArgs struct {
	KeyInputPointer    nativeDevicePointer
	ValueInputPointer  nativeDevicePointer
	KeyOutputPointer   nativeDevicePointer
	ValueOutputPointer nativeDevicePointer
	KeyCount           int
	ValueCount         int
	KeyInputBytes      uint64
	ValueInputBytes    uint64
	KeyOutputBytes     uint64
	ValueOutputBytes   uint64
	KeyEncoding        uint32
	ValueEncoding      uint32
}

type hipKVDescriptorAppendLaunchArgs struct {
	PreviousDescriptorPointer nativeDevicePointer
	OutputDescriptorPointer   nativeDevicePointer
	NewKeyPointer             nativeDevicePointer
	NewValuePointer           nativeDevicePointer
	PreviousDescriptorBytes   uint64
	OutputDescriptorBytes     uint64
	NewKeyBytes               uint64
	NewValueBytes             uint64
	ModeCode                  uint32
	BlockSize                 int
	OutputPageCount           int
	OutputTokenCount          int
	KeyWidth                  int
	ValueWidth                int
	NewKeyEncoding            uint32
	NewValueEncoding          uint32
	TrimStart                 int
}

type rocmDeviceKVDescriptor struct {
	Mode       string
	BlockSize  int
	TokenCount int
	Pages      []rocmDeviceKVPageDescriptor
}

type rocmDeviceKVPageDescriptor struct {
	TokenStart    int
	TokenCount    int
	KeyWidth      int
	ValueWidth    int
	KeyPointer    nativeDevicePointer
	ValuePointer  nativeDevicePointer
	KeyBytes      uint64
	ValueBytes    uint64
	KeyEncoding   string
	ValueEncoding string
}

type rocmDeviceKVPageSlicePool struct {
	sync.Mutex
	pages [][]rocmDeviceKVPage
}

var rocmDeviceKVPageSlicePools sync.Map

const rocmDeviceKVPageSlicePoolMaxPerCapacity = 512

type rocmDeviceKVTensorPoolEntry struct {
	driver  nativeHIPDriver
	pointer nativeDevicePointer
}

var rocmDeviceKVTensorPool = struct {
	sync.Mutex
	entries map[uint64][]rocmDeviceKVTensorPoolEntry
}{
	entries: make(map[uint64][]rocmDeviceKVTensorPoolEntry),
}

func hipGemma4Q4DeviceKVBlockSize() int {
	raw := os.Getenv("GO_ROCM_GEMMA4_Q4_DEVICE_KV_BLOCK_SIZE")
	if raw == "" {
		return rocmGemma4Q4DeviceKVBlockSize
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 {
		return rocmGemma4Q4DeviceKVBlockSize
	}
	return value
}

func rocmDeviceKVBorrowPageSlice(length, minCapacity int) []rocmDeviceKVPage {
	if minCapacity < length {
		minCapacity = length
	}
	minCapacity = rocmDeviceKVPageSliceCapacity(minCapacity)
	if minCapacity >= rocmDeviceKVHotPageCapacity && minCapacity <= rocmDeviceKVPagePoolMaxCapacity {
		poolValue, ok := rocmDeviceKVPageSlicePools.Load(minCapacity)
		if !ok {
			pool := &rocmDeviceKVPageSlicePool{}
			poolValue, _ = rocmDeviceKVPageSlicePools.LoadOrStore(minCapacity, pool)
		}
		pool := poolValue.(*rocmDeviceKVPageSlicePool)
		pool.Lock()
		if index := len(pool.pages) - 1; index >= 0 {
			pages := pool.pages[index]
			pool.pages[index] = nil
			pool.pages = pool.pages[:index]
			pool.Unlock()
			return pages[:length]
		}
		pool.Unlock()
	}
	return make([]rocmDeviceKVPage, length, minCapacity)
}

func rocmDeviceKVPageSliceCapacity(minCapacity int) int {
	if minCapacity <= 0 {
		return 0
	}
	capacity := rocmDeviceKVHotPageCapacity
	for capacity < minCapacity && capacity < rocmDeviceKVPagePoolMaxCapacity {
		capacity *= 2
	}
	if capacity < minCapacity {
		return minCapacity
	}
	return capacity
}

func rocmDeviceKVCopyPagesWithExtra(pages []rocmDeviceKVPage, extra int) []rocmDeviceKVPage {
	out := rocmDeviceKVBorrowPageSlice(len(pages), len(pages)+extra)
	copy(out, pages)
	return out
}

func rocmDeviceKVReleasePageSlice(pages []rocmDeviceKVPage) {
	if cap(pages) < rocmDeviceKVHotPageCapacity || cap(pages) > rocmDeviceKVPagePoolMaxCapacity {
		return
	}
	full := pages[:cap(pages)]
	for index := range full {
		full[index] = rocmDeviceKVPage{}
	}
	poolValue, ok := rocmDeviceKVPageSlicePools.Load(cap(full))
	if !ok {
		pool := &rocmDeviceKVPageSlicePool{}
		poolValue, _ = rocmDeviceKVPageSlicePools.LoadOrStore(cap(full), pool)
	}
	pool := poolValue.(*rocmDeviceKVPageSlicePool)
	pool.Lock()
	if len(pool.pages) < rocmDeviceKVPageSlicePoolMaxPerCapacity {
		pool.pages = append(pool.pages, full[:0])
	}
	pool.Unlock()
}

func rocmDeviceKVTensorPoolEnabled() bool {
	return os.Getenv("GO_ROCM_ENABLE_KV_TENSOR_POOL") == "1"
}

func rocmDeviceKVTensorMalloc(driver nativeHIPDriver, sizeBytes uint64) (nativeDevicePointer, error) {
	if !rocmDeviceKVTensorPoolEnabled() {
		return driver.Malloc(sizeBytes)
	}
	rocmDeviceKVTensorPool.Lock()
	entries := rocmDeviceKVTensorPool.entries[sizeBytes]
	for index := len(entries) - 1; index >= 0; index-- {
		entry := entries[index]
		if entry.driver != driver {
			continue
		}
		pointer := entry.pointer
		entries[index] = entries[len(entries)-1]
		entries = entries[:len(entries)-1]
		if len(entries) == 0 {
			delete(rocmDeviceKVTensorPool.entries, sizeBytes)
		} else {
			rocmDeviceKVTensorPool.entries[sizeBytes] = entries
		}
		rocmDeviceKVTensorPool.Unlock()
		return pointer, nil
	}
	rocmDeviceKVTensorPool.Unlock()
	return driver.Malloc(sizeBytes)
}

func rocmDeviceKVTensorFree(driver nativeHIPDriver, pointer nativeDevicePointer, sizeBytes uint64) error {
	if pointer == 0 {
		return nil
	}
	if rocmDeviceKVTensorPoolEnabled() && driver != nil && sizeBytes > 0 {
		rocmDeviceKVTensorPool.Lock()
		entries := rocmDeviceKVTensorPool.entries[sizeBytes]
		const maxPooledPerSize = 4096
		if len(entries) < maxPooledPerSize {
			rocmDeviceKVTensorPool.entries[sizeBytes] = append(entries, rocmDeviceKVTensorPoolEntry{driver: driver, pointer: pointer})
			rocmDeviceKVTensorPool.Unlock()
			return nil
		}
		rocmDeviceKVTensorPool.Unlock()
	}
	return driver.Free(pointer)
}

func (cache *rocmKVCache) MirrorToDevice(driver nativeHIPDriver) (*rocmDeviceKVCache, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceMirror", "cache is nil", nil)
	}
	if driver == nil {
		return nil, core.E("rocm.KVCache.DeviceMirror", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceMirror", "HIP driver is not available", nil)
	}
	if len(cache.blocks) == 0 {
		return nil, core.E("rocm.KVCache.DeviceMirror", "cache has no pages", nil)
	}
	device := &rocmDeviceKVCache{
		driver:     driver,
		mode:       cache.mode,
		blockSize:  cache.blockSize,
		tokenCount: cache.TokenCount(),
		pages:      make([]rocmDeviceKVPage, 0, len(cache.blocks)),
	}
	for _, block := range cache.blocks {
		page := rocmDeviceKVPage{
			tokenStart: block.tokenStart,
			tokenCount: block.tokenCount,
			keyWidth:   block.keyWidth,
			valueWidth: block.valueWidth,
			owned:      true,
		}
		key, err := mirrorROCmKVTensorToDevice(driver, block.key)
		if err != nil {
			_ = device.Close()
			return nil, core.E("rocm.KVCache.DeviceMirror", "copy KV key page", err)
		}
		page.key = key
		value, err := mirrorROCmKVTensorToDevice(driver, block.value)
		if err != nil {
			_ = rocmDeviceKVTensorFree(driver, key.pointer, key.sizeBytes)
			_ = device.Close()
			return nil, core.E("rocm.KVCache.DeviceMirror", "copy KV value page", err)
		}
		page.value = value
		device.pages = append(device.pages, page)
	}
	return device, nil
}

func mirrorROCmKVTensorToDevice(driver nativeHIPDriver, tensor rocmKVEncodedTensor) (rocmDeviceKVTensor, error) {
	payload, err := tensor.deviceBytes()
	if err != nil {
		return rocmDeviceKVTensor{}, err
	}
	pointer, err := rocmDeviceKVTensorMalloc(driver, uint64(len(payload)))
	if err != nil {
		return rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceMirror", "allocate KV tensor", err)
	}
	if err := hipCopyPinnedHostToDevice(driver, pointer, payload); err != nil {
		_ = rocmDeviceKVTensorFree(driver, pointer, uint64(len(payload)))
		return rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceMirror", "copy KV tensor", err)
	}
	return rocmDeviceKVTensor{pointer: pointer, sizeBytes: uint64(len(payload)), encoding: tensor.encoding}, nil
}

func rocmDeviceKVPageFromRawPayload(driver nativeHIPDriver, payload []byte) (rocmDeviceKVPage, error) {
	if driver == nil {
		return rocmDeviceKVPage{}, core.E("rocm.KVCache.DeviceRestore", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return rocmDeviceKVPage{}, core.E("rocm.KVCache.DeviceRestore", "HIP driver is not available", nil)
	}
	meta, keyPayload, valuePayload, err := rocmKVBlockRawPayloadParts(payload)
	if err != nil {
		return rocmDeviceKVPage{}, err
	}
	keyPointer, err := rocmDeviceKVTensorMalloc(driver, uint64(len(keyPayload)))
	if err != nil {
		return rocmDeviceKVPage{}, core.E("rocm.KVCache.DeviceRestore", "allocate KV key page", err)
	}
	if err := hipCopyPinnedHostToDevice(driver, keyPointer, keyPayload); err != nil {
		_ = rocmDeviceKVTensorFree(driver, keyPointer, uint64(len(keyPayload)))
		return rocmDeviceKVPage{}, core.E("rocm.KVCache.DeviceRestore", "copy KV key page", err)
	}
	valuePointer, err := rocmDeviceKVTensorMalloc(driver, uint64(len(valuePayload)))
	if err != nil {
		_ = rocmDeviceKVTensorFree(driver, keyPointer, uint64(len(keyPayload)))
		return rocmDeviceKVPage{}, core.E("rocm.KVCache.DeviceRestore", "allocate KV value page", err)
	}
	if err := hipCopyPinnedHostToDevice(driver, valuePointer, valuePayload); err != nil {
		_ = rocmDeviceKVTensorFree(driver, valuePointer, uint64(len(valuePayload)))
		_ = rocmDeviceKVTensorFree(driver, keyPointer, uint64(len(keyPayload)))
		return rocmDeviceKVPage{}, core.E("rocm.KVCache.DeviceRestore", "copy KV value page", err)
	}
	return rocmDeviceKVPage{
		tokenStart: meta.tokenStart,
		tokenCount: meta.tokenCount,
		keyWidth:   meta.keyWidth,
		valueWidth: meta.valueWidth,
		key: rocmDeviceKVTensor{
			pointer:   keyPointer,
			sizeBytes: uint64(len(keyPayload)),
			encoding:  meta.keyEncoding,
		},
		value: rocmDeviceKVTensor{
			pointer:   valuePointer,
			sizeBytes: uint64(len(valuePayload)),
			encoding:  meta.valueEncoding,
		},
		owned: true,
	}, nil
}

func (cache *rocmDeviceKVCache) withAppendedToken(key, value []float32) (*rocmDeviceKVCache, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache is nil", nil)
	}
	if cache.closed {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache is closed", nil)
	}
	if cache.driver == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !cache.driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	keyWidth, valueWidth, ok := cache.LastVectorWidths()
	if !ok {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache has no pages", nil)
	}
	if len(key) != keyWidth || len(value) != valueWidth {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV vector widths must match device cache shape", nil)
	}
	keyEncoding, valueEncoding := rocmKVEncodingsForMode(cache.mode)
	keyTensor, err := encodeROCmKVTensor(keyEncoding, key)
	if err != nil {
		return nil, err
	}
	valueTensor, err := encodeROCmKVTensor(valueEncoding, value)
	if err != nil {
		return nil, err
	}
	deviceKey, err := mirrorROCmKVTensorToDevice(cache.driver, keyTensor)
	if err != nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "copy KV key page", err)
	}
	deviceValue, err := mirrorROCmKVTensorToDevice(cache.driver, valueTensor)
	if err != nil {
		_ = rocmDeviceKVTensorFree(cache.driver, deviceKey.pointer, deviceKey.sizeBytes)
		return nil, core.E("rocm.KVCache.DeviceAppend", "copy KV value page", err)
	}
	tokenStart := cache.TokenCount()
	next := rocmBorrowDeviceKVCache(cache.driver, cache.mode, cache.blockSize, tokenStart+1, rocmDeviceKVCopyPagesWithExtra(cache.pages, 1), false)
	for index := range next.pages {
		next.pages[index].owned = false
	}
	next.pages = append(next.pages, rocmDeviceKVPage{
		tokenStart: tokenStart,
		tokenCount: 1,
		keyWidth:   keyWidth,
		valueWidth: valueWidth,
		key:        deviceKey,
		value:      deviceValue,
		owned:      true,
	})
	return next, nil
}

func (cache *rocmDeviceKVCache) withAppendedTokenWindow(key, value []float32, window int) (*rocmDeviceKVCache, error) {
	next, err := cache.withAppendedToken(key, value)
	if err != nil {
		return nil, err
	}
	if window <= 0 || next.TokenCount() <= window {
		return next, nil
	}
	oldTokenCount := next.TokenCount()
	trimStart := oldTokenCount - window
	trimmed := rocmDeviceKVBorrowPageSlice(0, len(next.pages))
	for _, page := range next.pages {
		pageEnd := page.tokenStart + page.tokenCount
		if pageEnd <= trimStart {
			continue
		}
		if page.tokenStart < trimStart {
			// The hot Gemma4 generation path appends one-token pages. If a
			// multi-token page straddles the window boundary, keep the untrimmed
			// cache rather than making a descriptor that points into the middle of
			// an encoded page we cannot slice safely.
			rocmDeviceKVReleasePageSlice(trimmed)
			return next, nil
		}
		page.tokenStart -= trimStart
		trimmed = append(trimmed, page)
	}
	if len(trimmed) == 0 {
		rocmDeviceKVReleasePageSlice(trimmed)
		return next, nil
	}
	pages := next.pages
	next.pages = trimmed
	next.tokenCount = oldTokenCount - trimStart
	rocmDeviceKVReleasePageSlice(pages)
	return next, nil
}

func (cache *rocmDeviceKVCache) withAppendedDeviceTokenWindow(ctx context.Context, key, value *hipDeviceByteBuffer, window int) (*rocmDeviceKVCache, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache is nil", nil)
	}
	if cache.closed {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache is closed", nil)
	}
	keyWidth, valueWidth, ok := cache.LastVectorWidths()
	if !ok {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache has no pages", nil)
	}
	encodedKey, encodedValue, err := hipRunKVEncodeTokenKernel(ctx, cache.driver, key, value, cache.mode)
	if err != nil {
		return nil, err
	}
	next, err := cache.withAppendedEncodedTokenWindow(encodedKey, encodedValue, keyWidth, valueWidth, window)
	if err != nil {
		_ = rocmDeviceKVTensorFree(cache.driver, encodedKey.pointer, encodedKey.sizeBytes)
		_ = rocmDeviceKVTensorFree(cache.driver, encodedValue.pointer, encodedValue.sizeBytes)
		return nil, err
	}
	return next, nil
}

func (cache *rocmDeviceKVCache) withAppendedDeviceRowsWindow(ctx context.Context, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, tokenCount, window int) (*rocmDeviceKVCache, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache is nil", nil)
	}
	if cache.closed {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV cache is closed", nil)
	}
	if cache.driver == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !cache.driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	if key == nil || key.Pointer() == 0 || value == nil || value.Pointer() == 0 {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV row buffers are required", nil)
	}
	if keyWidth <= 0 || valueWidth <= 0 || tokenCount <= 0 {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV row widths and token count must be positive", nil)
	}
	if key.Count() != keyWidth*tokenCount || value.Count() != valueWidth*tokenCount ||
		key.SizeBytes() != uint64(key.Count()*4) || value.SizeBytes() != uint64(value.Count()*4) {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV row buffer shape mismatch", nil)
	}
	if priorKeyWidth, priorValueWidth, ok := cache.LastVectorWidths(); ok && (priorKeyWidth != keyWidth || priorValueWidth != valueWidth) {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV row widths must match device cache shape", nil)
	}
	mode := firstNonEmptyString(cache.mode, rocmKVCacheModeFP16)
	if !isROCmKVCacheMode(mode) {
		return nil, core.E("rocm.KVCache.DeviceAppend", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
	blockSize := cache.blockSize
	if blockSize <= 0 {
		blockSize = defaultROCmKVBlockSize
	}
	pageCount := (tokenCount + blockSize - 1) / blockSize
	tokenStart := cache.TokenCount()
	next := rocmBorrowDeviceKVCache(cache.driver, mode, blockSize, tokenStart+tokenCount, rocmDeviceKVCopyPagesWithExtra(cache.pages, pageCount), false)
	for index := range next.pages {
		next.pages[index].owned = false
	}
	basePageCount := len(cache.pages)
	success := false
	defer func() {
		if !success {
			_ = next.closePagesFrom(basePageCount)
		}
	}()
	for tokenOffset := 0; tokenOffset < tokenCount; tokenOffset += blockSize {
		tokenEnd := tokenOffset + blockSize
		if tokenEnd > tokenCount {
			tokenEnd = tokenCount
		}
		pageTokens := tokenEnd - tokenOffset
		keyCount := pageTokens * keyWidth
		valueCount := pageTokens * valueWidth
		keyByteOffset := nativeDevicePointer(tokenOffset * keyWidth * 4)
		valueByteOffset := nativeDevicePointer(tokenOffset * valueWidth * 4)
		keyPage := hipBorrowDeviceByteBuffer(cache.driver, "device KV key row page", key.Pointer()+keyByteOffset, uint64(keyCount*4), keyCount)
		valuePage := hipBorrowDeviceByteBuffer(cache.driver, "device KV value row page", value.Pointer()+valueByteOffset, uint64(valueCount*4), valueCount)
		encodedKey, encodedValue, err := hipRunKVEncodeTokenKernel(ctx, cache.driver, keyPage, valuePage, mode)
		if err != nil {
			return nil, err
		}
		next.pages = append(next.pages, rocmDeviceKVPage{
			tokenStart: tokenStart + tokenOffset,
			tokenCount: pageTokens,
			keyWidth:   keyWidth,
			valueWidth: valueWidth,
			key:        encodedKey,
			value:      encodedValue,
			owned:      true,
		})
	}
	success = true
	if window > 0 && next.TokenCount() > window {
		return next.trimDeviceTokenWindow(window), nil
	}
	return next, nil
}

func newROCmDeviceKVCacheFromDeviceToken(ctx context.Context, driver nativeHIPDriver, mode string, blockSize int, key, value *hipDeviceByteBuffer, window int) (*rocmDeviceKVCache, error) {
	if driver == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	if key == nil || value == nil || key.Pointer() == 0 || value.Pointer() == 0 {
		return nil, core.E("rocm.KVCache.DeviceAppend", "device KV token buffers are required", nil)
	}
	mode = firstNonEmptyString(mode, rocmKVCacheModeFP16)
	if !isROCmKVCacheMode(mode) {
		return nil, core.E("rocm.KVCache.DeviceAppend", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
	if blockSize <= 0 {
		blockSize = defaultROCmKVBlockSize
	}
	encodedKey, encodedValue, err := hipRunKVEncodeTokenKernel(ctx, driver, key, value, mode)
	if err != nil {
		return nil, err
	}
	cache := rocmBorrowDeviceKVCache(driver, mode, blockSize, 0, nil, false)
	next, err := cache.withAppendedEncodedToken(encodedKey, encodedValue, key.Count(), value.Count())
	rocmReleaseDeviceKVCache(cache)
	if err != nil {
		_ = rocmDeviceKVTensorFree(driver, encodedKey.pointer, encodedKey.sizeBytes)
		_ = rocmDeviceKVTensorFree(driver, encodedValue.pointer, encodedValue.sizeBytes)
		return nil, err
	}
	if window > 0 && next.TokenCount() > window {
		return next.trimDeviceTokenWindow(window), nil
	}
	return next, nil
}

func newROCmDeviceKVCacheFromDeviceRows(ctx context.Context, driver nativeHIPDriver, mode string, blockSize int, key, value *hipDeviceByteBuffer, keyWidth, valueWidth, tokenCount, window int) (*rocmDeviceKVCache, error) {
	if driver == nil {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	mode = firstNonEmptyString(mode, rocmKVCacheModeFP16)
	if !isROCmKVCacheMode(mode) {
		return nil, core.E("rocm.KVCache.DeviceAppend", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
	if blockSize <= 0 {
		blockSize = defaultROCmKVBlockSize
	}
	cache := rocmBorrowDeviceKVCache(driver, mode, blockSize, 0, nil, false)
	next, err := cache.withAppendedDeviceRowsWindow(ctx, key, value, keyWidth, valueWidth, tokenCount, window)
	rocmReleaseDeviceKVCache(cache)
	return next, err
}

func (cache *rocmDeviceKVCache) withAppendedEncodedToken(key, value rocmDeviceKVTensor, keyWidth, valueWidth int) (*rocmDeviceKVCache, error) {
	return cache.withAppendedEncodedRows(key, value, keyWidth, valueWidth, 1)
}

func (cache *rocmDeviceKVCache) withAppendedEncodedTokenWindow(key, value rocmDeviceKVTensor, keyWidth, valueWidth, window int) (*rocmDeviceKVCache, error) {
	if window <= 0 || cache == nil || cache.TokenCount()+1 <= window {
		return cache.withAppendedEncodedToken(key, value, keyWidth, valueWidth)
	}
	next, ok, err := cache.withAppendedEncodedTokenTrimmed(key, value, keyWidth, valueWidth, window)
	if err != nil {
		return nil, err
	}
	if ok {
		return next, nil
	}
	next, err = cache.withAppendedEncodedToken(key, value, keyWidth, valueWidth)
	if err != nil {
		return nil, err
	}
	return next.trimDeviceTokenWindow(window), nil
}

func (cache *rocmDeviceKVCache) withAppendedEncodedTokenTrimmed(key, value rocmDeviceKVTensor, keyWidth, valueWidth, window int) (*rocmDeviceKVCache, bool, error) {
	if err := cache.validateAppendedEncodedRows(key, value, keyWidth, valueWidth, 1); err != nil {
		return nil, false, err
	}
	trimStart := cache.TokenCount() + 1 - window
	pages := rocmDeviceKVBorrowPageSlice(0, window)
	for _, page := range cache.pages {
		pageEnd := page.tokenStart + page.tokenCount
		if pageEnd <= trimStart {
			continue
		}
		if page.tokenStart < trimStart {
			rocmDeviceKVReleasePageSlice(pages)
			return nil, false, nil
		}
		page.tokenStart -= trimStart
		page.owned = false
		pages = append(pages, page)
	}
	tokenStart := cache.TokenCount() - trimStart
	pages = append(pages, rocmDeviceKVPage{
		tokenStart: tokenStart,
		tokenCount: 1,
		keyWidth:   keyWidth,
		valueWidth: valueWidth,
		key:        key,
		value:      value,
		owned:      true,
	})
	return rocmBorrowDeviceKVCache(cache.driver, cache.mode, cache.blockSize, window, pages, false), true, nil
}

func (cache *rocmDeviceKVCache) withAppendedEncodedRows(key, value rocmDeviceKVTensor, keyWidth, valueWidth, tokenCount int) (*rocmDeviceKVCache, error) {
	if err := cache.validateAppendedEncodedRows(key, value, keyWidth, valueWidth, tokenCount); err != nil {
		return nil, err
	}
	tokenStart := cache.TokenCount()
	next := rocmBorrowDeviceKVCache(cache.driver, cache.mode, cache.blockSize, tokenStart+tokenCount, rocmDeviceKVCopyPagesWithExtra(cache.pages, 1), false)
	for index := range next.pages {
		next.pages[index].owned = false
	}
	next.pages = append(next.pages, rocmDeviceKVPage{
		tokenStart: tokenStart,
		tokenCount: tokenCount,
		keyWidth:   keyWidth,
		valueWidth: valueWidth,
		key:        key,
		value:      value,
		owned:      true,
	})
	return next, nil
}

func (cache *rocmDeviceKVCache) validateAppendedEncodedRows(key, value rocmDeviceKVTensor, keyWidth, valueWidth, tokenCount int) error {
	if cache == nil {
		return core.E("rocm.KVCache.DeviceAppend", "device KV cache is nil", nil)
	}
	if cache.closed {
		return core.E("rocm.KVCache.DeviceAppend", "device KV cache is closed", nil)
	}
	if cache.driver == nil {
		return core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !cache.driver.Available() {
		return core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	if key.pointer == 0 || value.pointer == 0 || key.sizeBytes == 0 || value.sizeBytes == 0 {
		return core.E("rocm.KVCache.DeviceAppend", "encoded device KV row tensors are required", nil)
	}
	if keyWidth <= 0 || valueWidth <= 0 || tokenCount <= 0 {
		return core.E("rocm.KVCache.DeviceAppend", "KV row widths and token count must be positive", nil)
	}
	if priorKeyWidth, priorValueWidth, ok := cache.LastVectorWidths(); ok && (priorKeyWidth != keyWidth || priorValueWidth != valueWidth) {
		return core.E("rocm.KVCache.DeviceAppend", "KV row widths must match device cache shape", nil)
	}
	expectedKeyEncoding, expectedValueEncoding := rocmKVEncodingsForMode(cache.mode)
	if key.encoding != expectedKeyEncoding || value.encoding != expectedValueEncoding {
		return core.E("rocm.KVCache.DeviceAppend", "encoded device KV row encodings do not match cache mode", nil)
	}
	expectedKeyBytes, err := rocmKVTensorDeviceByteCount(key.encoding, keyWidth*tokenCount)
	if err != nil {
		return err
	}
	expectedValueBytes, err := rocmKVTensorDeviceByteCount(value.encoding, valueWidth*tokenCount)
	if err != nil {
		return err
	}
	if key.sizeBytes != expectedKeyBytes || value.sizeBytes != expectedValueBytes {
		return core.E("rocm.KVCache.DeviceAppend", "encoded device KV row byte count mismatch", nil)
	}
	return nil
}

func (cache *rocmDeviceKVCache) trimDeviceTokenWindow(window int) *rocmDeviceKVCache {
	if cache == nil || window <= 0 || cache.TokenCount() <= window {
		return cache
	}
	oldTokenCount := cache.TokenCount()
	trimStart := oldTokenCount - window
	trimmed := rocmDeviceKVBorrowPageSlice(0, len(cache.pages))
	for _, page := range cache.pages {
		pageEnd := page.tokenStart + page.tokenCount
		if pageEnd <= trimStart {
			continue
		}
		if page.tokenStart < trimStart {
			rocmDeviceKVReleasePageSlice(trimmed)
			return cache
		}
		page.tokenStart -= trimStart
		trimmed = append(trimmed, page)
	}
	if len(trimmed) == 0 {
		rocmDeviceKVReleasePageSlice(trimmed)
		return cache
	}
	pages := cache.pages
	cache.pages = trimmed
	cache.tokenCount = oldTokenCount - trimStart
	rocmDeviceKVReleasePageSlice(pages)
	return cache
}

func hipRunKVEncodeTokenKernel(ctx context.Context, driver nativeHIPDriver, key, value *hipDeviceByteBuffer, mode string) (rocmDeviceKVTensor, rocmDeviceKVTensor, error) {
	if err := hipContextErr(ctx); err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	if driver == nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "HIP driver is not available", nil)
	}
	if key == nil || key.Pointer() == 0 || key.Count() <= 0 || key.SizeBytes() != uint64(key.Count())*4 {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "device KV key token buffer is required", nil)
	}
	if value == nil || value.Pointer() == 0 || value.Count() <= 0 || value.SizeBytes() != uint64(value.Count())*4 {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "device KV value token buffer is required", nil)
	}
	mode = firstNonEmptyString(mode, rocmKVCacheModeFP16)
	if !isROCmKVCacheMode(mode) {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
	keyEncoding, valueEncoding := rocmKVEncodingsForMode(mode)
	keyEncodingCode, err := rocmDeviceKVEncodingCode(keyEncoding)
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	valueEncodingCode, err := rocmDeviceKVEncodingCode(valueEncoding)
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	keyBytes, err := rocmKVTensorDeviceByteCount(keyEncoding, key.Count())
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	valueBytes, err := rocmKVTensorDeviceByteCount(valueEncoding, value.Count())
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	keyPointer, err := rocmDeviceKVTensorMalloc(driver, keyBytes)
	if err != nil {
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "allocate encoded KV key token", err)
	}
	valuePointer, err := rocmDeviceKVTensorMalloc(driver, valueBytes)
	if err != nil {
		_ = rocmDeviceKVTensorFree(driver, keyPointer, keyBytes)
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceAppend", "allocate encoded KV value token", err)
	}
	args, err := (hipKVEncodeTokenLaunchArgs{
		KeyInputPointer:    key.Pointer(),
		ValueInputPointer:  value.Pointer(),
		KeyOutputPointer:   keyPointer,
		ValueOutputPointer: valuePointer,
		KeyCount:           key.Count(),
		ValueCount:         value.Count(),
		KeyInputBytes:      key.SizeBytes(),
		ValueInputBytes:    value.SizeBytes(),
		KeyOutputBytes:     keyBytes,
		ValueOutputBytes:   valueBytes,
		KeyEncoding:        keyEncodingCode,
		ValueEncoding:      valueEncodingCode,
	}).Binary()
	if err != nil {
		_ = rocmDeviceKVTensorFree(driver, keyPointer, keyBytes)
		_ = rocmDeviceKVTensorFree(driver, valuePointer, valueBytes)
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameKVEncodeToken,
		Args:   args,
		GridX:  2,
		GridY:  1,
		GridZ:  1,
		BlockX: hipKVEncodeTokenBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := hipLaunchKernel(driver, config); err != nil {
		_ = rocmDeviceKVTensorFree(driver, keyPointer, keyBytes)
		_ = rocmDeviceKVTensorFree(driver, valuePointer, valueBytes)
		return rocmDeviceKVTensor{}, rocmDeviceKVTensor{}, err
	}
	return rocmDeviceKVTensor{pointer: keyPointer, sizeBytes: keyBytes, encoding: keyEncoding},
		rocmDeviceKVTensor{pointer: valuePointer, sizeBytes: valueBytes, encoding: valueEncoding}, nil
}

func (cache *rocmDeviceKVCache) borrowedAlias() (*rocmDeviceKVCache, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceAlias", "device KV cache is nil", nil)
	}
	if cache.closed {
		return nil, core.E("rocm.KVCache.DeviceAlias", "device KV cache is closed", nil)
	}
	if cache.driver == nil {
		return nil, core.E("rocm.KVCache.DeviceAlias", "HIP driver is nil", nil)
	}
	if len(cache.pages) == 0 {
		return nil, core.E("rocm.KVCache.DeviceAlias", "device KV cache has no pages", nil)
	}
	alias := &rocmDeviceKVCache{
		driver:     cache.driver,
		mode:       cache.mode,
		blockSize:  cache.blockSize,
		pages:      cache.pages,
		tokenCount: cache.TokenCount(),
		borrowed:   true,
	}
	return alias, nil
}

func (cache *rocmDeviceKVCache) closePagesFrom(index int) error {
	if cache == nil {
		return nil
	}
	if cache.borrowed {
		cache.pages = nil
		cache.tokenCount = 0
		cache.closed = true
		return nil
	}
	if index < 0 {
		index = 0
	}
	if index > len(cache.pages) {
		index = len(cache.pages)
	}
	var lastErr error
	pages := cache.pages
	for pageIndex := index; pageIndex < len(cache.pages); pageIndex++ {
		page := &cache.pages[pageIndex]
		if !page.owned {
			continue
		}
		if page.key.pointer != 0 {
			if err := rocmDeviceKVTensorFree(cache.driver, page.key.pointer, page.key.sizeBytes); err != nil {
				lastErr = core.E("rocm.KVCache.DeviceAppend", "free appended KV key page", err)
			}
			page.key.pointer = 0
		}
		if page.value.pointer != 0 {
			if err := rocmDeviceKVTensorFree(cache.driver, page.value.pointer, page.value.sizeBytes); err != nil {
				lastErr = core.E("rocm.KVCache.DeviceAppend", "free appended KV value page", err)
			}
			page.value.pointer = 0
		}
		page.owned = false
	}
	cache.pages = nil
	cache.tokenCount = 0
	cache.closed = true
	rocmDeviceKVReleasePageSlice(pages)
	return lastErr
}

func (cache *rocmDeviceKVCache) transferPagesTo(next *rocmDeviceKVCache) error {
	if cache == nil {
		return nil
	}
	if next == nil {
		return core.E("rocm.KVCache.DeviceAppend", "next device KV cache is nil", nil)
	}
	if cache.driver != next.driver || cache.mode != next.mode || cache.blockSize != next.blockSize {
		return core.E("rocm.KVCache.DeviceAppend", "device KV cache ownership target does not match", nil)
	}
	if len(next.pages) < len(cache.pages) {
		return core.E("rocm.KVCache.DeviceAppend", "device KV cache ownership target is missing source pages", nil)
	}
	for index := range cache.pages {
		if cache.pages[index].key.pointer != next.pages[index].key.pointer || cache.pages[index].value.pointer != next.pages[index].value.pointer {
			return core.E("rocm.KVCache.DeviceAppend", "device KV cache ownership target page pointers do not match source", nil)
		}
		if cache.pages[index].owned {
			next.pages[index].owned = true
			cache.pages[index].key.pointer = 0
			cache.pages[index].value.pointer = 0
		}
		cache.pages[index].owned = false
	}
	pages := cache.pages
	cache.pages = nil
	cache.tokenCount = 0
	cache.closed = true
	rocmDeviceKVReleasePageSlice(pages)
	return nil
}

func (cache *rocmDeviceKVCache) transferSharedPagesTo(next *rocmDeviceKVCache) error {
	if cache == nil {
		return nil
	}
	if next == nil {
		return core.E("rocm.KVCache.DeviceAppend", "next device KV cache is nil", nil)
	}
	if cache.driver != next.driver || cache.mode != next.mode || cache.blockSize != next.blockSize {
		return core.E("rocm.KVCache.DeviceAppend", "device KV cache ownership target does not match", nil)
	}
	var lastErr error
	sourcePages := cache.pages
	targetPages := next.pages
	if len(sourcePages) > 0 && len(targetPages) >= len(sourcePages) &&
		rocmDeviceKVPagePointersEqual(&sourcePages[0], &targetPages[0]) &&
		rocmDeviceKVPagePointersEqual(&sourcePages[len(sourcePages)-1], &targetPages[len(sourcePages)-1]) {
		for index := range sourcePages {
			rocmDeviceKVTransferPageOwnership(&sourcePages[index], &targetPages[index])
		}
		cache.finishTransferSharedPages()
		return nil
	}
	if suffixOffset := len(sourcePages) - len(targetPages); suffixOffset > 0 && len(targetPages) > 0 &&
		rocmDeviceKVPagePointersEqual(&sourcePages[suffixOffset], &targetPages[0]) &&
		rocmDeviceKVPagePointersEqual(&sourcePages[len(sourcePages)-1], &targetPages[len(targetPages)-1]) {
		for sourceIndex := 0; sourceIndex < suffixOffset; sourceIndex++ {
			rocmDeviceKVFreeOwnedPage(cache.driver, &sourcePages[sourceIndex], &lastErr)
		}
		for targetIndex := range targetPages {
			rocmDeviceKVTransferPageOwnership(&sourcePages[targetIndex+suffixOffset], &targetPages[targetIndex])
		}
		cache.finishTransferSharedPages()
		return lastErr
	}
	for sourceIndex := range sourcePages {
		source := &sourcePages[sourceIndex]
		matched := false
		for targetIndex := range targetPages {
			target := &targetPages[targetIndex]
			if rocmDeviceKVPagePointersEqual(source, target) {
				rocmDeviceKVTransferPageOwnership(source, target)
				matched = true
				break
			}
		}
		if matched || !source.owned {
			continue
		}
		rocmDeviceKVFreeOwnedPage(cache.driver, source, &lastErr)
	}
	cache.finishTransferSharedPages()
	return lastErr
}

func (cache *rocmDeviceKVCache) finishTransferSharedPages() {
	pages := cache.pages
	cache.pages = nil
	cache.tokenCount = 0
	cache.closed = true
	rocmDeviceKVReleasePageSlice(pages)
}

func rocmDeviceKVPagePointersEqual(source, target *rocmDeviceKVPage) bool {
	return source != nil && target != nil &&
		source.key.pointer == target.key.pointer &&
		source.value.pointer == target.value.pointer
}

func rocmDeviceKVTransferPageOwnership(source, target *rocmDeviceKVPage) {
	if source.owned {
		target.owned = true
		source.key.pointer = 0
		source.value.pointer = 0
	}
	source.owned = false
}

func rocmDeviceKVFreeOwnedPage(driver nativeHIPDriver, page *rocmDeviceKVPage, lastErr *error) {
	if page == nil || !page.owned {
		return
	}
	if page.key.pointer != 0 {
		if err := rocmDeviceKVTensorFree(driver, page.key.pointer, page.key.sizeBytes); err != nil && lastErr != nil {
			*lastErr = core.E("rocm.KVCache.DeviceAppend", "free trimmed KV key page", err)
		}
		page.key.pointer = 0
	}
	if page.value.pointer != 0 {
		if err := rocmDeviceKVTensorFree(driver, page.value.pointer, page.value.sizeBytes); err != nil && lastErr != nil {
			*lastErr = core.E("rocm.KVCache.DeviceAppend", "free trimmed KV value page", err)
		}
		page.value.pointer = 0
	}
	page.owned = false
}

func (cache *rocmDeviceKVCache) borrowsPagesFrom(source *rocmDeviceKVCache) bool {
	if cache == nil || source == nil {
		return false
	}
	if cache.driver != source.driver || cache.mode != source.mode || cache.blockSize != source.blockSize {
		return false
	}
	if len(cache.pages) < len(source.pages) {
		return false
	}
	for index := range source.pages {
		if cache.pages[index].key.pointer != source.pages[index].key.pointer ||
			cache.pages[index].value.pointer != source.pages[index].value.pointer {
			return false
		}
	}
	return true
}

func (cache *rocmDeviceKVCache) sharesPagesFrom(source *rocmDeviceKVCache) bool {
	if cache == nil || source == nil {
		return false
	}
	if cache.driver != source.driver || cache.mode != source.mode || cache.blockSize != source.blockSize {
		return false
	}
	for sourceIndex := range source.pages {
		for targetIndex := range cache.pages {
			if source.pages[sourceIndex].key.pointer == cache.pages[targetIndex].key.pointer &&
				source.pages[sourceIndex].value.pointer == cache.pages[targetIndex].value.pointer {
				return true
			}
		}
	}
	return false
}

func (cache *rocmDeviceKVCache) ownsAnyPages() bool {
	if cache == nil || cache.borrowed {
		return false
	}
	for _, page := range cache.pages {
		if page.owned {
			return true
		}
	}
	return false
}

func (cache *rocmDeviceKVCache) Close() error {
	if cache == nil || cache.closed {
		return nil
	}
	if cache.borrowed {
		cache.pages = nil
		cache.tokenCount = 0
		cache.closed = true
		return nil
	}
	var lastErr error
	pages := cache.pages
	for index := range cache.pages {
		page := &cache.pages[index]
		if !page.owned {
			continue
		}
		if page.key.pointer != 0 {
			if err := rocmDeviceKVTensorFree(cache.driver, page.key.pointer, page.key.sizeBytes); err != nil {
				lastErr = core.E("rocm.KVCache.DeviceMirror", "free KV key page", err)
			}
			page.key.pointer = 0
		}
		if page.value.pointer != 0 {
			if err := rocmDeviceKVTensorFree(cache.driver, page.value.pointer, page.value.sizeBytes); err != nil {
				lastErr = core.E("rocm.KVCache.DeviceMirror", "free KV value page", err)
			}
			page.value.pointer = 0
		}
		page.owned = false
	}
	cache.pages = nil
	cache.tokenCount = 0
	cache.closed = true
	rocmDeviceKVReleasePageSlice(pages)
	return lastErr
}

func (cache *rocmDeviceKVCache) PageCount() int {
	if cache == nil {
		return 0
	}
	return len(cache.pages)
}

func (cache *rocmDeviceKVCache) TokenCount() int {
	if cache == nil {
		return 0
	}
	if cache.tokenCount > 0 || len(cache.pages) == 0 {
		return cache.tokenCount
	}
	return rocmDeviceKVPagesTokenCount(cache.pages)
}

func rocmDeviceKVPagesTokenCount(pages []rocmDeviceKVPage) int {
	var maxEnd int
	for _, page := range pages {
		if end := page.tokenStart + page.tokenCount; end > maxEnd {
			maxEnd = end
		}
	}
	return maxEnd
}

func (cache *rocmDeviceKVCache) MemoryBytes() uint64 {
	if cache == nil {
		return 0
	}
	var total uint64
	for _, page := range cache.pages {
		total += page.key.sizeBytes + page.value.sizeBytes
	}
	return total
}

func (cache *rocmDeviceKVCache) Stats() inference.CacheStats {
	if cache == nil {
		return inference.CacheStats{}
	}
	labels := map[string]string{
		"kv_backing":          "hip_device_mirror",
		"kv_block_size":       core.Sprintf("%d", cache.blockSize),
		"kv_cache_block_size": core.Sprintf("%d", cache.blockSize),
		"kv_device_backing":   "mirrored",
		"kv_pages":            core.Sprintf("%d", cache.PageCount()),
		"kv_tokens":           core.Sprintf("%d", cache.TokenCount()),
	}
	if keyWidth, valueWidth, ok := cache.LastVectorWidths(); ok {
		labels["kv_key_width"] = core.Sprintf("%d", keyWidth)
		labels["kv_value_width"] = core.Sprintf("%d", valueWidth)
	}
	return inference.CacheStats{
		Blocks:      len(cache.pages),
		MemoryBytes: cache.MemoryBytes(),
		CacheMode:   cache.mode,
		Labels:      labels,
	}
}

func (cache *rocmDeviceKVCache) Snapshot() ([]byte, error) {
	host, err := cache.hostCache()
	if err != nil {
		return nil, err
	}
	return host.Snapshot()
}

func (cache *rocmDeviceKVCache) hostCache() (*rocmKVCache, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceSnapshot", "device KV cache is nil", nil)
	}
	if cache.closed {
		return nil, core.E("rocm.KVCache.DeviceSnapshot", "device KV cache is closed", nil)
	}
	if cache.driver == nil {
		return nil, core.E("rocm.KVCache.DeviceSnapshot", "HIP driver is nil", nil)
	}
	if !cache.driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceSnapshot", "HIP driver is not available", nil)
	}
	if len(cache.pages) == 0 {
		return nil, core.E("rocm.KVCache.DeviceSnapshot", "device KV cache has no pages", nil)
	}
	host, err := newROCmKVCache(cache.mode, cache.blockSize)
	if err != nil {
		return nil, err
	}
	for _, page := range cache.pages {
		key, err := copyROCmDeviceKVTensorToHost(cache.driver, page.key, page.tokenCount*page.keyWidth)
		if err != nil {
			return nil, core.E("rocm.KVCache.DeviceSnapshot", "copy KV key page", err)
		}
		value, err := copyROCmDeviceKVTensorToHost(cache.driver, page.value, page.tokenCount*page.valueWidth)
		if err != nil {
			return nil, core.E("rocm.KVCache.DeviceSnapshot", "copy KV value page", err)
		}
		if err := host.validateVectorShape(page.keyWidth, page.valueWidth); err != nil {
			return nil, err
		}
		block := rocmKVCacheBlock{
			tokenStart: page.tokenStart,
			tokenCount: page.tokenCount,
			keyWidth:   page.keyWidth,
			valueWidth: page.valueWidth,
			key:        key,
			value:      value,
		}
		host.blocks, err = insertROCmKVCacheBlock(host.blocks, block)
		if err != nil {
			return nil, err
		}
		host.setVectorShape(page.keyWidth, page.valueWidth)
	}
	return host, nil
}

func (cache *rocmDeviceKVCache) LastVectorWidths() (int, int, bool) {
	if cache == nil || len(cache.pages) == 0 {
		return 0, 0, false
	}
	page := cache.pages[len(cache.pages)-1]
	return page.keyWidth, page.valueWidth, true
}

func (cache *rocmDeviceKVCache) CompatibleWith(host *rocmKVCache) error {
	if cache == nil {
		return nil
	}
	if cache.closed {
		return core.E("rocm.KVCache.DeviceMirror", "device KV cache is closed", nil)
	}
	if host == nil {
		return core.E("rocm.KVCache.DeviceMirror", "package KV cache is nil", nil)
	}
	if cache.mode != host.mode {
		return core.E("rocm.KVCache.DeviceMirror", "cache mode mismatch", nil)
	}
	if cache.blockSize != host.blockSize {
		return core.E("rocm.KVCache.DeviceMirror", "cache block size mismatch", nil)
	}
	if cache.PageCount() != host.PageCount() {
		return core.E("rocm.KVCache.DeviceMirror", "page count mismatch", nil)
	}
	if cache.TokenCount() != host.TokenCount() {
		return core.E("rocm.KVCache.DeviceMirror", "token count mismatch", nil)
	}
	keyWidth, valueWidth, ok := cache.LastVectorWidths()
	hostKeyWidth, hostValueWidth, hostOK := host.LastVectorWidths()
	if ok != hostOK || keyWidth != hostKeyWidth || valueWidth != hostValueWidth {
		return core.E("rocm.KVCache.DeviceMirror", "KV vector width mismatch", nil)
	}
	return nil
}

func (cache *rocmDeviceKVCache) KernelDescriptor() (rocmDeviceKVDescriptor, error) {
	if cache == nil {
		return rocmDeviceKVDescriptor{}, core.E("rocm.KVCache.DeviceMirror", "device KV cache is nil", nil)
	}
	if cache.closed {
		return rocmDeviceKVDescriptor{}, core.E("rocm.KVCache.DeviceMirror", "device KV cache is closed", nil)
	}
	if len(cache.pages) == 0 {
		return rocmDeviceKVDescriptor{}, core.E("rocm.KVCache.DeviceMirror", "device KV cache has no pages", nil)
	}
	descriptor := rocmDeviceKVDescriptor{
		Mode:       cache.mode,
		BlockSize:  cache.blockSize,
		TokenCount: cache.TokenCount(),
		Pages:      make([]rocmDeviceKVPageDescriptor, 0, len(cache.pages)),
	}
	for _, page := range cache.pages {
		if page.key.pointer == 0 || page.value.pointer == 0 {
			return rocmDeviceKVDescriptor{}, core.E("rocm.KVCache.DeviceMirror", "device KV page has nil pointer", nil)
		}
		descriptor.Pages = append(descriptor.Pages, rocmDeviceKVPageDescriptor{
			TokenStart:    page.tokenStart,
			TokenCount:    page.tokenCount,
			KeyWidth:      page.keyWidth,
			ValueWidth:    page.valueWidth,
			KeyPointer:    page.key.pointer,
			ValuePointer:  page.value.pointer,
			KeyBytes:      page.key.sizeBytes,
			ValueBytes:    page.value.sizeBytes,
			KeyEncoding:   page.key.encoding,
			ValueEncoding: page.value.encoding,
		})
	}
	return descriptor, nil
}

func (cache *rocmDeviceKVCache) KernelDescriptorBytes() ([]byte, error) {
	descriptor, err := cache.KernelDescriptor()
	if err != nil {
		return nil, err
	}
	return descriptor.Binary()
}

func (cache *rocmDeviceKVCache) KernelDescriptorTable() (*rocmDeviceKVDescriptorTable, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV cache is nil", nil)
	}
	if cache.driver == nil {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "HIP driver is nil", nil)
	}
	if !cache.driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "HIP driver is not available", nil)
	}
	payload, err := cache.KernelDescriptorBytes()
	if err != nil {
		return nil, err
	}
	pointer, err := cache.driver.Malloc(uint64(len(payload)))
	if err != nil {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "allocate descriptor table", err)
	}
	if err := hipCopyHostToDevice(cache.driver, pointer, payload); err != nil {
		_ = cache.driver.Free(pointer)
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "copy descriptor table", err)
	}
	return rocmBorrowDeviceKVDescriptorTable(cache.driver, pointer, uint64(len(payload)), rocmDeviceKVDescriptorVersion, cache.PageCount(), false, false), nil
}

func (cache *rocmDeviceKVCache) KernelDescriptorTableFromAppendedToken(ctx context.Context, previous *rocmDeviceKVCache, previousTable *rocmDeviceKVDescriptorTable) (*rocmDeviceKVDescriptorTable, error) {
	if err := hipContextErr(ctx); err != nil {
		return nil, err
	}
	if cache == nil || previous == nil {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV append descriptor caches are required", nil)
	}
	if cache.closed || previous.closed {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV append descriptor cache is closed", nil)
	}
	if cache.driver == nil || !cache.driver.Available() {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "HIP driver is not available", nil)
	}
	if cache.driver != previous.driver || cache.mode != previous.mode || cache.blockSize != previous.blockSize {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV append descriptor cache shape mismatch", nil)
	}
	if previousTable == nil {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "previous descriptor table is required", nil)
	}
	if err := previousTable.CompatibleWith(previous); err != nil {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "previous descriptor table does not match device KV cache", err)
	}
	trimStart, copiedPages, err := rocmDeviceKVAppendDescriptorShape(previous, cache)
	if err != nil {
		return nil, err
	}
	if copiedPages+1 != cache.PageCount() {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV append descriptor page count mismatch", nil)
	}
	lastPage := cache.pages[len(cache.pages)-1]
	modeCode, err := rocmDeviceKVModeCode(cache.mode)
	if err != nil {
		return nil, err
	}
	keyEncoding, err := rocmDeviceKVEncodingCode(lastPage.key.encoding)
	if err != nil {
		return nil, err
	}
	valueEncoding, err := rocmDeviceKVEncodingCode(lastPage.value.encoding)
	if err != nil {
		return nil, err
	}
	outputBytes := uint64(rocmDeviceKVDescriptorHeaderBytes + cache.PageCount()*rocmDeviceKVDescriptorPageBytes)
	pointer, allocationBytes, err := rocmDeviceKVDescriptorTableMalloc(cache.driver, outputBytes)
	if err != nil {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "allocate appended descriptor table", err)
	}
	args, err := (hipKVDescriptorAppendLaunchArgs{
		PreviousDescriptorPointer: previousTable.Pointer(),
		OutputDescriptorPointer:   pointer,
		NewKeyPointer:             lastPage.key.pointer,
		NewValuePointer:           lastPage.value.pointer,
		PreviousDescriptorBytes:   previousTable.SizeBytes(),
		OutputDescriptorBytes:     outputBytes,
		NewKeyBytes:               lastPage.key.sizeBytes,
		NewValueBytes:             lastPage.value.sizeBytes,
		ModeCode:                  modeCode,
		BlockSize:                 cache.blockSize,
		OutputPageCount:           cache.PageCount(),
		OutputTokenCount:          cache.TokenCount(),
		KeyWidth:                  lastPage.keyWidth,
		ValueWidth:                lastPage.valueWidth,
		NewKeyEncoding:            keyEncoding,
		NewValueEncoding:          valueEncoding,
		TrimStart:                 trimStart,
	}).Binary()
	if err != nil {
		_ = rocmDeviceKVDescriptorTableFree(cache.driver, pointer, allocationBytes)
		return nil, err
	}
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameKVDescriptorAppend,
		Args:   args,
		GridX:  1,
		GridY:  1,
		GridZ:  1,
		BlockX: hipKVDescriptorAppendBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	if err := hipLaunchKernel(cache.driver, config); err != nil {
		_ = rocmDeviceKVDescriptorTableFree(cache.driver, pointer, allocationBytes)
		return nil, err
	}
	return rocmBorrowDeviceKVDescriptorTableAllocated(cache.driver, pointer, outputBytes, allocationBytes, rocmDeviceKVDescriptorVersion, cache.PageCount(), false, true), nil
}

func rocmDeviceKVAppendDescriptorShape(previous, next *rocmDeviceKVCache) (int, int, error) {
	if previous == nil || next == nil || len(previous.pages) == 0 || len(next.pages) == 0 {
		return 0, 0, core.E("rocm.KVCache.DeviceDescriptor", "device KV append descriptor caches must have pages", nil)
	}
	lastPage := next.pages[len(next.pages)-1]
	if lastPage.tokenCount != 1 || lastPage.tokenStart != next.TokenCount()-1 {
		return 0, 0, core.E("rocm.KVCache.DeviceDescriptor", "device KV append descriptor last page must be the appended token", nil)
	}
	if previous.TokenCount()+1 < next.TokenCount() {
		return 0, 0, core.E("rocm.KVCache.DeviceDescriptor", "device KV append descriptor token range mismatch", nil)
	}
	trimStart := previous.TokenCount() + 1 - next.TokenCount()
	copiedPages := 0
	for _, page := range previous.pages {
		pageEnd := page.tokenStart + page.tokenCount
		if pageEnd <= trimStart {
			continue
		}
		if page.tokenStart < trimStart {
			return 0, 0, core.E("rocm.KVCache.DeviceDescriptor", "device KV append descriptor cannot trim inside a page", nil)
		}
		if copiedPages >= len(next.pages)-1 {
			return 0, 0, core.E("rocm.KVCache.DeviceDescriptor", "device KV append descriptor has too many retained pages", nil)
		}
		nextPage := next.pages[copiedPages]
		if nextPage.tokenStart != page.tokenStart-trimStart ||
			nextPage.tokenCount != page.tokenCount ||
			nextPage.keyWidth != page.keyWidth ||
			nextPage.valueWidth != page.valueWidth ||
			nextPage.key.pointer != page.key.pointer ||
			nextPage.value.pointer != page.value.pointer ||
			nextPage.key.sizeBytes != page.key.sizeBytes ||
			nextPage.value.sizeBytes != page.value.sizeBytes ||
			nextPage.key.encoding != page.key.encoding ||
			nextPage.value.encoding != page.value.encoding {
			return 0, 0, core.E("rocm.KVCache.DeviceDescriptor", "device KV append descriptor retained page mismatch", nil)
		}
		copiedPages++
	}
	if copiedPages != len(next.pages)-1 {
		return 0, 0, core.E("rocm.KVCache.DeviceDescriptor", "device KV append descriptor retained page count mismatch", nil)
	}
	return trimStart, copiedPages, nil
}

func (table *rocmDeviceKVDescriptorTable) Pointer() nativeDevicePointer {
	if table == nil || table.closed {
		return 0
	}
	return table.pointer
}

func (table *rocmDeviceKVDescriptorTable) SizeBytes() uint64 {
	if table == nil || table.closed {
		return 0
	}
	return table.sizeBytes
}

func (table *rocmDeviceKVDescriptorTable) AllocationBytes() uint64 {
	if table == nil || table.closed {
		return 0
	}
	if table.allocationBytes != 0 {
		return table.allocationBytes
	}
	return table.sizeBytes
}

func (table *rocmDeviceKVDescriptorTable) CompatibleWith(cache *rocmDeviceKVCache) error {
	if table == nil {
		return nil
	}
	if table.closed || table.pointer == 0 {
		return core.E("rocm.KVCache.DeviceDescriptor", "descriptor table is closed", nil)
	}
	if cache == nil {
		return core.E("rocm.KVCache.DeviceDescriptor", "device KV cache is nil", nil)
	}
	if cache.closed {
		return core.E("rocm.KVCache.DeviceDescriptor", "device KV cache is closed", nil)
	}
	if table.version != rocmDeviceKVDescriptorVersion {
		return core.E("rocm.KVCache.DeviceDescriptor", "descriptor table version mismatch", nil)
	}
	if table.pageCount != cache.PageCount() {
		return core.E("rocm.KVCache.DeviceDescriptor", "descriptor table page count mismatch", nil)
	}
	expectedBytes := uint64(rocmDeviceKVDescriptorHeaderBytes + table.pageCount*rocmDeviceKVDescriptorPageBytes)
	if table.sizeBytes != expectedBytes {
		return core.E("rocm.KVCache.DeviceDescriptor", "descriptor table size mismatch", nil)
	}
	return nil
}

func (cache *rocmDeviceKVCache) KernelLaunchDescriptor(table *rocmDeviceKVDescriptorTable) (rocmDeviceKVLaunchDescriptor, error) {
	if cache == nil {
		return rocmDeviceKVLaunchDescriptor{}, core.E("rocm.KVCache.DeviceLaunch", "device KV cache is nil", nil)
	}
	if cache.closed {
		return rocmDeviceKVLaunchDescriptor{}, core.E("rocm.KVCache.DeviceLaunch", "device KV cache is closed", nil)
	}
	if table == nil {
		return rocmDeviceKVLaunchDescriptor{}, core.E("rocm.KVCache.DeviceLaunch", "descriptor table is required", nil)
	}
	if err := table.CompatibleWith(cache); err != nil {
		return rocmDeviceKVLaunchDescriptor{}, core.E("rocm.KVCache.DeviceLaunch", "descriptor table does not match device KV cache", err)
	}
	modeCode, err := rocmDeviceKVModeCode(cache.mode)
	if err != nil {
		return rocmDeviceKVLaunchDescriptor{}, err
	}
	keyWidth, valueWidth, ok := cache.LastVectorWidths()
	if !ok {
		return rocmDeviceKVLaunchDescriptor{}, core.E("rocm.KVCache.DeviceLaunch", "device KV cache has no pages", nil)
	}
	return rocmDeviceKVLaunchDescriptor{
		DescriptorPointer: table.Pointer(),
		DescriptorBytes:   table.SizeBytes(),
		DescriptorVersion: table.version,
		Mode:              cache.mode,
		ModeCode:          modeCode,
		BlockSize:         cache.blockSize,
		PageCount:         cache.PageCount(),
		TokenCount:        cache.TokenCount(),
		KeyWidth:          keyWidth,
		ValueWidth:        valueWidth,
	}, nil
}

func (launch rocmDeviceKVLaunchDescriptor) Binary() ([]byte, error) {
	if launch.DescriptorPointer == 0 {
		return nil, core.E("rocm.KVCache.DeviceLaunch", "descriptor pointer is nil", nil)
	}
	if launch.DescriptorBytes == 0 {
		return nil, core.E("rocm.KVCache.DeviceLaunch", "descriptor bytes must be positive", nil)
	}
	if launch.DescriptorVersion != rocmDeviceKVDescriptorVersion {
		return nil, core.E("rocm.KVCache.DeviceLaunch", "descriptor version mismatch", nil)
	}
	if err := rocmDeviceKVValidateModeCode(launch.ModeCode); err != nil {
		return nil, err
	}
	if launch.Mode != "" {
		modeCode, err := rocmDeviceKVModeCode(launch.Mode)
		if err != nil {
			return nil, err
		}
		if modeCode != launch.ModeCode {
			return nil, core.E("rocm.KVCache.DeviceLaunch", "mode code mismatch", nil)
		}
	}
	blockSize, err := rocmDeviceKVPositiveUint32("block size", launch.BlockSize)
	if err != nil {
		return nil, err
	}
	pageCount, err := rocmDeviceKVPositiveUint32("page count", launch.PageCount)
	if err != nil {
		return nil, err
	}
	tokenCount, err := rocmDeviceKVUint64("token count", launch.TokenCount)
	if err != nil {
		return nil, err
	}
	if tokenCount == 0 {
		return nil, core.E("rocm.KVCache.DeviceLaunch", "token count must be positive", nil)
	}
	keyWidth, err := rocmDeviceKVPositiveUint32("key width", launch.KeyWidth)
	if err != nil {
		return nil, err
	}
	valueWidth, err := rocmDeviceKVPositiveUint32("value width", launch.ValueWidth)
	if err != nil {
		return nil, err
	}
	payload := make([]byte, rocmDeviceKVLaunchDescriptorBytes)
	statusValue := launch.StatusValue
	if launch.StatusPointer != 0 && statusValue == 0 {
		statusValue = hipDecodeLaunchStatusOK
	}
	binary.LittleEndian.PutUint64(payload[0:], uint64(launch.DescriptorPointer))
	binary.LittleEndian.PutUint64(payload[8:], launch.DescriptorBytes)
	binary.LittleEndian.PutUint32(payload[16:], launch.DescriptorVersion)
	binary.LittleEndian.PutUint32(payload[20:], launch.ModeCode)
	binary.LittleEndian.PutUint32(payload[24:], blockSize)
	binary.LittleEndian.PutUint32(payload[28:], pageCount)
	binary.LittleEndian.PutUint64(payload[32:], tokenCount)
	binary.LittleEndian.PutUint32(payload[40:], keyWidth)
	binary.LittleEndian.PutUint32(payload[44:], valueWidth)
	binary.LittleEndian.PutUint64(payload[48:], uint64(launch.StatusPointer))
	binary.LittleEndian.PutUint32(payload[56:], statusValue)
	return payload, nil
}

func (table *rocmDeviceKVDescriptorTable) Close() error {
	if table == nil || table.closed {
		return nil
	}
	if table.borrowed {
		table.closed = true
		return nil
	}
	if table.pointer != 0 {
		if err := rocmDeviceKVDescriptorTableFree(table.driver, table.pointer, table.AllocationBytes()); err != nil {
			return core.E("rocm.KVCache.DeviceDescriptor", "free descriptor table", err)
		}
		table.pointer = 0
	}
	if table.poolable {
		rocmReleaseDeviceKVDescriptorTable(table)
		return nil
	}
	table.closed = true
	return nil
}

func (table *rocmDeviceKVDescriptorTable) borrowedAlias() (*rocmDeviceKVDescriptorTable, error) {
	if table == nil {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "descriptor table is nil", nil)
	}
	if table.closed || table.pointer == 0 {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "descriptor table is closed", nil)
	}
	return rocmBorrowDeviceKVDescriptorTable(table.driver, table.pointer, table.sizeBytes, table.version, table.pageCount, true, false), nil
}

func (descriptor rocmDeviceKVDescriptor) Binary() ([]byte, error) {
	if len(descriptor.Pages) == 0 {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV descriptor has no pages", nil)
	}
	modeCode, err := rocmDeviceKVModeCode(descriptor.Mode)
	if err != nil {
		return nil, err
	}
	pageCount, err := rocmDeviceKVUint32("page count", len(descriptor.Pages))
	if err != nil {
		return nil, err
	}
	blockSize, err := rocmDeviceKVPositiveUint32("block size", descriptor.BlockSize)
	if err != nil {
		return nil, err
	}
	tokenCount, err := rocmDeviceKVUint64("token count", descriptor.TokenCount)
	if err != nil {
		return nil, err
	}
	if tokenCount == 0 {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "token count must be positive", nil)
	}
	payload := make([]byte, rocmDeviceKVDescriptorHeaderBytes+len(descriptor.Pages)*rocmDeviceKVDescriptorPageBytes)
	binary.LittleEndian.PutUint32(payload[0:], rocmDeviceKVDescriptorVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(rocmDeviceKVDescriptorHeaderBytes))
	binary.LittleEndian.PutUint32(payload[8:], uint32(rocmDeviceKVDescriptorPageBytes))
	binary.LittleEndian.PutUint32(payload[12:], modeCode)
	binary.LittleEndian.PutUint32(payload[16:], pageCount)
	binary.LittleEndian.PutUint32(payload[20:], blockSize)
	binary.LittleEndian.PutUint64(payload[24:], tokenCount)

	var lastPageEnd uint64
	for index, page := range descriptor.Pages {
		offset := rocmDeviceKVDescriptorHeaderBytes + index*rocmDeviceKVDescriptorPageBytes
		tokenStart, err := rocmDeviceKVUint64("page token start", page.TokenStart)
		if err != nil {
			return nil, err
		}
		pageTokenCount, err := rocmDeviceKVUint64("page token count", page.TokenCount)
		if err != nil {
			return nil, err
		}
		if pageTokenCount == 0 {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "page token count must be positive", nil)
		}
		pageEnd := tokenStart + pageTokenCount
		if pageEnd > tokenCount {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "page token range exceeds descriptor token count", nil)
		}
		if index > 0 && tokenStart < lastPageEnd {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV descriptor pages must be sorted and non-overlapping", nil)
		}
		lastPageEnd = pageEnd
		keyWidth, err := rocmDeviceKVPositiveUint32("page key width", page.KeyWidth)
		if err != nil {
			return nil, err
		}
		valueWidth, err := rocmDeviceKVPositiveUint32("page value width", page.ValueWidth)
		if err != nil {
			return nil, err
		}
		keyEncoding, err := rocmDeviceKVEncodingCode(page.KeyEncoding)
		if err != nil {
			return nil, err
		}
		valueEncoding, err := rocmDeviceKVEncodingCode(page.ValueEncoding)
		if err != nil {
			return nil, err
		}
		if page.KeyPointer == 0 || page.ValuePointer == 0 {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV descriptor page has nil pointer", nil)
		}
		if page.KeyBytes == 0 || page.ValueBytes == 0 {
			return nil, core.E("rocm.KVCache.DeviceDescriptor", "device KV descriptor page has empty tensor bytes", nil)
		}
		binary.LittleEndian.PutUint64(payload[offset:], tokenStart)
		binary.LittleEndian.PutUint64(payload[offset+8:], pageTokenCount)
		binary.LittleEndian.PutUint32(payload[offset+16:], keyWidth)
		binary.LittleEndian.PutUint32(payload[offset+20:], valueWidth)
		binary.LittleEndian.PutUint32(payload[offset+24:], keyEncoding)
		binary.LittleEndian.PutUint32(payload[offset+28:], valueEncoding)
		binary.LittleEndian.PutUint64(payload[offset+32:], uint64(page.KeyPointer))
		binary.LittleEndian.PutUint64(payload[offset+40:], uint64(page.ValuePointer))
		binary.LittleEndian.PutUint64(payload[offset+48:], page.KeyBytes)
		binary.LittleEndian.PutUint64(payload[offset+56:], page.ValueBytes)
	}
	return payload, nil
}

func rocmDeviceKVModeCode(mode string) (uint32, error) {
	switch mode {
	case rocmKVCacheModeFP16:
		return rocmDeviceKVDescriptorModeFP16, nil
	case rocmKVCacheModeQ8:
		return rocmDeviceKVDescriptorModeQ8, nil
	case rocmKVCacheModeKQ8VQ4:
		return rocmDeviceKVDescriptorModeKQ8VQ4, nil
	default:
		return 0, core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
}

func rocmDeviceKVValidateModeCode(code uint32) error {
	switch code {
	case rocmDeviceKVDescriptorModeFP16, rocmDeviceKVDescriptorModeQ8, rocmDeviceKVDescriptorModeKQ8VQ4:
		return nil
	default:
		return core.E("rocm.KVCache.DeviceLaunch", core.Sprintf("unsupported cache mode code %d", code), nil)
	}
}

func rocmDeviceKVEncodingCode(encoding string) (uint32, error) {
	switch encoding {
	case rocmKVEncodingFP16:
		return rocmDeviceKVDescriptorEncodingFP16, nil
	case rocmKVEncodingQ8:
		return rocmDeviceKVDescriptorEncodingQ8, nil
	case rocmKVEncodingQ4:
		return rocmDeviceKVDescriptorEncodingQ4, nil
	default:
		return 0, core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("unsupported tensor encoding %q", encoding), nil)
	}
}

func rocmDeviceKVValidateEncodingCode(code uint32) error {
	switch code {
	case rocmDeviceKVDescriptorEncodingFP16, rocmDeviceKVDescriptorEncodingQ8, rocmDeviceKVDescriptorEncodingQ4:
		return nil
	default:
		return core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("unsupported tensor encoding code %d", code), nil)
	}
}

func rocmDeviceKVEncodingName(code uint32) (string, error) {
	switch code {
	case rocmDeviceKVDescriptorEncodingFP16:
		return rocmKVEncodingFP16, nil
	case rocmDeviceKVDescriptorEncodingQ8:
		return rocmKVEncodingQ8, nil
	case rocmDeviceKVDescriptorEncodingQ4:
		return rocmKVEncodingQ4, nil
	default:
		return "", core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("unsupported tensor encoding code %d", code), nil)
	}
}

func rocmKVTensorDeviceByteCount(encoding string, length int) (uint64, error) {
	if length <= 0 {
		return 0, core.E("rocm.KVCache.DeviceDescriptor", "tensor length must be positive", nil)
	}
	switch encoding {
	case rocmKVEncodingFP16:
		return uint64(length) * 2, nil
	case rocmKVEncodingQ8:
		return uint64(length) + 4, nil
	case rocmKVEncodingQ4:
		return uint64((length+1)/2) + 4, nil
	default:
		return 0, core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("unsupported tensor encoding %q", encoding), nil)
	}
}

func rocmDeviceKVUint32Bytes(field string, value uint64) (uint32, error) {
	if value > uint64(^uint32(0)) {
		return 0, core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("%s are out of uint32 range", field), nil)
	}
	return uint32(value), nil
}

func (args hipKVEncodeTokenLaunchArgs) Binary() ([]byte, error) {
	if args.KeyInputPointer == 0 || args.ValueInputPointer == 0 || args.KeyOutputPointer == 0 || args.ValueOutputPointer == 0 {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV encode token pointers are required", nil)
	}
	keyCount, err := rocmDeviceKVPositiveUint32("key count", args.KeyCount)
	if err != nil {
		return nil, err
	}
	valueCount, err := rocmDeviceKVPositiveUint32("value count", args.ValueCount)
	if err != nil {
		return nil, err
	}
	if args.KeyInputBytes != uint64(keyCount)*4 || args.ValueInputBytes != uint64(valueCount)*4 {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV encode token input byte count mismatch", nil)
	}
	if err := rocmDeviceKVValidateEncodingCode(args.KeyEncoding); err != nil {
		return nil, err
	}
	if err := rocmDeviceKVValidateEncodingCode(args.ValueEncoding); err != nil {
		return nil, err
	}
	keyEncoding, err := rocmDeviceKVEncodingName(args.KeyEncoding)
	if err != nil {
		return nil, err
	}
	valueEncoding, err := rocmDeviceKVEncodingName(args.ValueEncoding)
	if err != nil {
		return nil, err
	}
	expectedKeyBytes, err := rocmKVTensorDeviceByteCount(keyEncoding, int(keyCount))
	if err != nil {
		return nil, err
	}
	expectedValueBytes, err := rocmKVTensorDeviceByteCount(valueEncoding, int(valueCount))
	if err != nil {
		return nil, err
	}
	if args.KeyOutputBytes != expectedKeyBytes || args.ValueOutputBytes != expectedValueBytes {
		return nil, core.E("rocm.KVCache.DeviceAppend", "KV encode token output byte count mismatch", nil)
	}
	keyInputBytes, err := rocmDeviceKVUint32Bytes("key input bytes", args.KeyInputBytes)
	if err != nil {
		return nil, err
	}
	valueInputBytes, err := rocmDeviceKVUint32Bytes("value input bytes", args.ValueInputBytes)
	if err != nil {
		return nil, err
	}
	keyOutputBytes, err := rocmDeviceKVUint32Bytes("key output bytes", args.KeyOutputBytes)
	if err != nil {
		return nil, err
	}
	valueOutputBytes, err := rocmDeviceKVUint32Bytes("value output bytes", args.ValueOutputBytes)
	if err != nil {
		return nil, err
	}
	payload := hipBorrowLaunchPacket(hipKVEncodeTokenLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipKVEncodeTokenLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(hipKVEncodeTokenLaunchArgsBytes))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.KeyInputPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.ValueInputPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.KeyOutputPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.ValueOutputPointer))
	binary.LittleEndian.PutUint32(payload[40:], keyCount)
	binary.LittleEndian.PutUint32(payload[44:], valueCount)
	binary.LittleEndian.PutUint32(payload[48:], keyInputBytes)
	binary.LittleEndian.PutUint32(payload[52:], valueInputBytes)
	binary.LittleEndian.PutUint32(payload[56:], keyOutputBytes)
	binary.LittleEndian.PutUint32(payload[60:], valueOutputBytes)
	binary.LittleEndian.PutUint32(payload[64:], args.KeyEncoding)
	binary.LittleEndian.PutUint32(payload[68:], args.ValueEncoding)
	return payload, nil
}

func (args hipKVDescriptorAppendLaunchArgs) Binary() ([]byte, error) {
	if args.PreviousDescriptorPointer == 0 || args.OutputDescriptorPointer == 0 || args.NewKeyPointer == 0 || args.NewValuePointer == 0 {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "KV descriptor append pointers are required", nil)
	}
	if args.PreviousDescriptorBytes < rocmDeviceKVDescriptorHeaderBytes || args.OutputDescriptorBytes < rocmDeviceKVDescriptorHeaderBytes {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "KV descriptor append byte counts must include headers", nil)
	}
	if err := rocmDeviceKVValidateModeCode(args.ModeCode); err != nil {
		return nil, err
	}
	blockSize, err := rocmDeviceKVPositiveUint32("block size", args.BlockSize)
	if err != nil {
		return nil, err
	}
	outputPageCount, err := rocmDeviceKVPositiveUint32("output page count", args.OutputPageCount)
	if err != nil {
		return nil, err
	}
	outputTokenCount, err := rocmDeviceKVPositiveUint32("output token count", args.OutputTokenCount)
	if err != nil {
		return nil, err
	}
	keyWidth, err := rocmDeviceKVPositiveUint32("key width", args.KeyWidth)
	if err != nil {
		return nil, err
	}
	valueWidth, err := rocmDeviceKVPositiveUint32("value width", args.ValueWidth)
	if err != nil {
		return nil, err
	}
	if err := rocmDeviceKVValidateEncodingCode(args.NewKeyEncoding); err != nil {
		return nil, err
	}
	if err := rocmDeviceKVValidateEncodingCode(args.NewValueEncoding); err != nil {
		return nil, err
	}
	keyEncoding, err := rocmDeviceKVEncodingName(args.NewKeyEncoding)
	if err != nil {
		return nil, err
	}
	valueEncoding, err := rocmDeviceKVEncodingName(args.NewValueEncoding)
	if err != nil {
		return nil, err
	}
	expectedKeyBytes, err := rocmKVTensorDeviceByteCount(keyEncoding, int(keyWidth))
	if err != nil {
		return nil, err
	}
	expectedValueBytes, err := rocmKVTensorDeviceByteCount(valueEncoding, int(valueWidth))
	if err != nil {
		return nil, err
	}
	if args.NewKeyBytes != expectedKeyBytes || args.NewValueBytes != expectedValueBytes {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "KV descriptor append new page byte count mismatch", nil)
	}
	expectedOutputBytes := uint64(rocmDeviceKVDescriptorHeaderBytes) + uint64(outputPageCount)*uint64(rocmDeviceKVDescriptorPageBytes)
	if args.OutputDescriptorBytes != expectedOutputBytes {
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "KV descriptor append output byte count mismatch", nil)
	}
	trimStart, err := rocmDeviceKVUint64("trim start", args.TrimStart)
	if err != nil {
		return nil, err
	}
	payload := hipBorrowLaunchPacket(hipKVDescriptorAppendLaunchArgsBytes)
	binary.LittleEndian.PutUint32(payload[0:], hipKVDescriptorAppendLaunchArgsVersion)
	binary.LittleEndian.PutUint32(payload[4:], uint32(hipKVDescriptorAppendLaunchArgsBytes))
	binary.LittleEndian.PutUint64(payload[8:], uint64(args.PreviousDescriptorPointer))
	binary.LittleEndian.PutUint64(payload[16:], uint64(args.OutputDescriptorPointer))
	binary.LittleEndian.PutUint64(payload[24:], uint64(args.NewKeyPointer))
	binary.LittleEndian.PutUint64(payload[32:], uint64(args.NewValuePointer))
	binary.LittleEndian.PutUint64(payload[40:], args.PreviousDescriptorBytes)
	binary.LittleEndian.PutUint64(payload[48:], args.OutputDescriptorBytes)
	binary.LittleEndian.PutUint64(payload[56:], args.NewKeyBytes)
	binary.LittleEndian.PutUint64(payload[64:], args.NewValueBytes)
	binary.LittleEndian.PutUint32(payload[72:], args.ModeCode)
	binary.LittleEndian.PutUint32(payload[76:], blockSize)
	binary.LittleEndian.PutUint32(payload[80:], outputPageCount)
	binary.LittleEndian.PutUint32(payload[84:], outputTokenCount)
	binary.LittleEndian.PutUint32(payload[88:], keyWidth)
	binary.LittleEndian.PutUint32(payload[92:], valueWidth)
	binary.LittleEndian.PutUint32(payload[96:], args.NewKeyEncoding)
	binary.LittleEndian.PutUint32(payload[100:], args.NewValueEncoding)
	binary.LittleEndian.PutUint64(payload[104:], trimStart)
	return payload, nil
}

func rocmDeviceKVUint32(field string, value int) (uint32, error) {
	if value < 0 || value > int(^uint32(0)) {
		return 0, core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("%s is out of uint32 range", field), nil)
	}
	return uint32(value), nil
}

func rocmDeviceKVPositiveUint32(field string, value int) (uint32, error) {
	out, err := rocmDeviceKVUint32(field, value)
	if err != nil {
		return 0, err
	}
	if out == 0 {
		return 0, core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("%s must be positive", field), nil)
	}
	return out, nil
}

func rocmDeviceKVUint64(field string, value int) (uint64, error) {
	if value < 0 {
		return 0, core.E("rocm.KVCache.DeviceDescriptor", core.Sprintf("%s is out of uint64 range", field), nil)
	}
	return uint64(value), nil
}

func (tensor rocmKVEncodedTensor) deviceBytes() ([]byte, error) {
	switch tensor.encoding {
	case rocmKVEncodingFP16:
		payload := make([]byte, len(tensor.f16)*2)
		for i, value := range tensor.f16 {
			binary.LittleEndian.PutUint16(payload[i*2:], value)
		}
		return payload, nil
	case rocmKVEncodingQ8:
		payload := make([]byte, 4+len(tensor.q8))
		binary.LittleEndian.PutUint32(payload, math.Float32bits(tensor.scale))
		for i, value := range tensor.q8 {
			payload[4+i] = byte(value)
		}
		return payload, nil
	case rocmKVEncodingQ4:
		payload := make([]byte, 4+len(tensor.packedQ4))
		binary.LittleEndian.PutUint32(payload, math.Float32bits(tensor.scale))
		copy(payload[4:], tensor.packedQ4)
		return payload, nil
	default:
		return nil, core.E("rocm.KVCache.DeviceMirror", core.Sprintf("unsupported tensor encoding %q", tensor.encoding), nil)
	}
}

func copyROCmDeviceKVTensorToHost(driver nativeHIPDriver, tensor rocmDeviceKVTensor, length int) (rocmKVEncodedTensor, error) {
	if driver == nil {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "HIP driver is nil", nil)
	}
	if tensor.pointer == 0 {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "device tensor pointer is nil", nil)
	}
	if tensor.sizeBytes == 0 {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "device tensor byte count is zero", nil)
	}
	maxInt := uint64(int(^uint(0) >> 1))
	if tensor.sizeBytes > maxInt {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "device tensor byte count exceeds addressable memory", nil)
	}
	payload := make([]byte, int(tensor.sizeBytes))
	if err := driver.CopyDeviceToHost(tensor.pointer, payload); err != nil {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "copy device tensor", err)
	}
	return rocmKVTensorFromDeviceBytes(tensor.encoding, length, payload)
}

func rocmKVTensorFromDeviceBytes(encoding string, length int, payload []byte) (rocmKVEncodedTensor, error) {
	if length <= 0 {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "tensor length must be positive", nil)
	}
	tensor := rocmKVEncodedTensor{encoding: encoding, length: length, sizeBytes: uint64(len(payload))}
	switch encoding {
	case rocmKVEncodingFP16:
		if len(payload) != length*2 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "fp16 tensor byte count mismatch", nil)
		}
		tensor.f16 = make([]uint16, length)
		for index := range tensor.f16 {
			tensor.f16[index] = binary.LittleEndian.Uint16(payload[index*2:])
		}
	case rocmKVEncodingQ8:
		if len(payload) != length+4 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q8 tensor byte count mismatch", nil)
		}
		tensor.scale = math.Float32frombits(binary.LittleEndian.Uint32(payload[0:]))
		if tensor.scale <= 0 || math.IsNaN(float64(tensor.scale)) || math.IsInf(float64(tensor.scale), 0) {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q8 scale must be positive and finite", nil)
		}
		tensor.q8 = make([]int8, length)
		for index, value := range payload[4:] {
			tensor.q8[index] = int8(value)
		}
	case rocmKVEncodingQ4:
		packedLength := (length + 1) / 2
		if len(payload) != packedLength+4 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q4 tensor byte count mismatch", nil)
		}
		tensor.scale = math.Float32frombits(binary.LittleEndian.Uint32(payload[0:]))
		if tensor.scale <= 0 || math.IsNaN(float64(tensor.scale)) || math.IsInf(float64(tensor.scale), 0) {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", "q4 scale must be positive and finite", nil)
		}
		tensor.packedQ4 = append([]byte(nil), payload[4:]...)
	default:
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.DeviceSnapshot", core.Sprintf("unsupported tensor encoding %q", encoding), nil)
	}
	return tensor, nil
}
