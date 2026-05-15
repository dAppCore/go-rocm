// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const (
	rocmDeviceKVDescriptorVersion     uint32 = 1
	rocmDeviceKVDescriptorHeaderBytes        = 32
	rocmDeviceKVDescriptorPageBytes          = 64
	rocmDeviceKVLaunchDescriptorBytes        = 64
)

const (
	rocmDeviceKVDescriptorModeFP16   uint32 = 1
	rocmDeviceKVDescriptorModeQ8     uint32 = 2
	rocmDeviceKVDescriptorModeKQ8VQ4 uint32 = 3
)

const (
	rocmDeviceKVDescriptorEncodingFP16 uint32 = 1
	rocmDeviceKVDescriptorEncodingQ8   uint32 = 2
	rocmDeviceKVDescriptorEncodingQ4   uint32 = 3
)

type rocmDeviceKVCache struct {
	driver    nativeHIPDriver
	mode      string
	blockSize int
	pages     []rocmDeviceKVPage
	closed    bool
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
	driver    nativeHIPDriver
	pointer   nativeDevicePointer
	sizeBytes uint64
	version   uint32
	pageCount int
	closed    bool
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
		driver:    driver,
		mode:      cache.mode,
		blockSize: cache.blockSize,
		pages:     make([]rocmDeviceKVPage, 0, len(cache.blocks)),
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
			_ = driver.Free(key.pointer)
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
	pointer, err := driver.Malloc(uint64(len(payload)))
	if err != nil {
		return rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceMirror", "allocate KV tensor", err)
	}
	if err := driver.CopyHostToDevice(pointer, payload); err != nil {
		_ = driver.Free(pointer)
		return rocmDeviceKVTensor{}, core.E("rocm.KVCache.DeviceMirror", "copy KV tensor", err)
	}
	return rocmDeviceKVTensor{pointer: pointer, sizeBytes: uint64(len(payload)), encoding: tensor.encoding}, nil
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
		_ = cache.driver.Free(deviceKey.pointer)
		return nil, core.E("rocm.KVCache.DeviceAppend", "copy KV value page", err)
	}
	next := &rocmDeviceKVCache{
		driver:    cache.driver,
		mode:      cache.mode,
		blockSize: cache.blockSize,
		pages:     append([]rocmDeviceKVPage(nil), cache.pages...),
	}
	for index := range next.pages {
		next.pages[index].owned = false
	}
	next.pages = append(next.pages, rocmDeviceKVPage{
		tokenStart: cache.TokenCount(),
		tokenCount: 1,
		keyWidth:   keyWidth,
		valueWidth: valueWidth,
		key:        deviceKey,
		value:      deviceValue,
		owned:      true,
	})
	return next, nil
}

func (cache *rocmDeviceKVCache) closePagesFrom(index int) error {
	if cache == nil {
		return nil
	}
	if index < 0 {
		index = 0
	}
	if index > len(cache.pages) {
		index = len(cache.pages)
	}
	var lastErr error
	for pageIndex := index; pageIndex < len(cache.pages); pageIndex++ {
		page := &cache.pages[pageIndex]
		if !page.owned {
			continue
		}
		if page.key.pointer != 0 {
			if err := cache.driver.Free(page.key.pointer); err != nil {
				lastErr = core.E("rocm.KVCache.DeviceAppend", "free appended KV key page", err)
			}
			page.key.pointer = 0
		}
		if page.value.pointer != 0 {
			if err := cache.driver.Free(page.value.pointer); err != nil {
				lastErr = core.E("rocm.KVCache.DeviceAppend", "free appended KV value page", err)
			}
			page.value.pointer = 0
		}
		page.owned = false
	}
	cache.closed = true
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
		next.pages[index].owned = true
		cache.pages[index].key.pointer = 0
		cache.pages[index].value.pointer = 0
		cache.pages[index].owned = false
	}
	cache.pages = nil
	cache.closed = true
	return nil
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

func (cache *rocmDeviceKVCache) Close() error {
	if cache == nil || cache.closed {
		return nil
	}
	var lastErr error
	for index := range cache.pages {
		page := &cache.pages[index]
		if !page.owned {
			continue
		}
		if page.key.pointer != 0 {
			if err := cache.driver.Free(page.key.pointer); err != nil {
				lastErr = core.E("rocm.KVCache.DeviceMirror", "free KV key page", err)
			}
			page.key.pointer = 0
		}
		if page.value.pointer != 0 {
			if err := cache.driver.Free(page.value.pointer); err != nil {
				lastErr = core.E("rocm.KVCache.DeviceMirror", "free KV value page", err)
			}
			page.value.pointer = 0
		}
		page.owned = false
	}
	cache.closed = true
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
	var maxEnd int
	for _, page := range cache.pages {
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
	if err := cache.driver.CopyHostToDevice(pointer, payload); err != nil {
		_ = cache.driver.Free(pointer)
		return nil, core.E("rocm.KVCache.DeviceDescriptor", "copy descriptor table", err)
	}
	return &rocmDeviceKVDescriptorTable{
		driver:    cache.driver,
		pointer:   pointer,
		sizeBytes: uint64(len(payload)),
		version:   rocmDeviceKVDescriptorVersion,
		pageCount: cache.PageCount(),
	}, nil
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
	if table.pointer != 0 {
		if err := table.driver.Free(table.pointer); err != nil {
			return core.E("rocm.KVCache.DeviceDescriptor", "free descriptor table", err)
		}
		table.pointer = 0
	}
	table.closed = true
	return nil
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
