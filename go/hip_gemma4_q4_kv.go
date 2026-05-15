// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import core "dappco.re/go"

type hipGemma4Q4DeviceDecodeState struct {
	mode           string
	layers         []hipGemma4Q4DeviceLayerKVState
	appendLayers   int
	remirrorLayers int
	closed         bool
}

type hipGemma4Q4DeviceLayerKVState struct {
	cache           *rocmDeviceKVCache
	descriptorTable *rocmDeviceKVDescriptorTable
	launch          rocmDeviceKVLaunchDescriptor
}

func (layer *hipGemma4Q4DeviceLayerKVState) Close() error {
	if layer == nil {
		return nil
	}
	var lastErr error
	if err := layer.descriptorTable.Close(); err != nil {
		lastErr = core.E("rocm.hip.Gemma4Q4DeviceKV", "free descriptor table", err)
	}
	if err := layer.cache.Close(); err != nil {
		lastErr = core.E("rocm.hip.Gemma4Q4DeviceKV", "free device KV layer", err)
	}
	return lastErr
}

func hipMirrorGemma4Q4DecodeState(driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, state hipGemma4Q4DecodeState, mode string) (*hipGemma4Q4DeviceDecodeState, error) {
	if driver == nil {
		return nil, core.E("rocm.hip.Gemma4Q4DeviceKV", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return nil, core.E("rocm.hip.Gemma4Q4DeviceKV", "HIP driver is not available", nil)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if err := state.validate(cfg); err != nil {
		return nil, err
	}
	if len(state.Layers) == 0 {
		return nil, core.E("rocm.hip.Gemma4Q4DeviceKV", "decode state has no layers", nil)
	}
	if mode == "" {
		mode = rocmKVCacheModeFP16
	}
	deviceState := &hipGemma4Q4DeviceDecodeState{mode: mode, remirrorLayers: len(state.Layers), layers: make([]hipGemma4Q4DeviceLayerKVState, 0, len(state.Layers))}
	for index, layerState := range state.Layers {
		layer, err := hipMirrorGemma4Q4LayerDecodeState(driver, cfg.Layers[index], layerState, mode)
		if err != nil {
			_ = deviceState.Close()
			return nil, err
		}
		deviceState.layers = append(deviceState.layers, layer)
	}
	return deviceState, nil
}

func hipUpdateGemma4Q4DeviceDecodeState(driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, previousHost, nextHost hipGemma4Q4DecodeState, previousDevice *hipGemma4Q4DeviceDecodeState, mode string) (*hipGemma4Q4DeviceDecodeState, error) {
	if previousDevice == nil {
		return hipMirrorGemma4Q4DecodeState(driver, cfg, nextHost, mode)
	}
	if previousDevice.closed {
		return nil, core.E("rocm.hip.Gemma4Q4DeviceKV", "previous device decode state is closed", nil)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if err := previousHost.validate(cfg); err != nil {
		return nil, err
	}
	if err := nextHost.validate(cfg); err != nil {
		return nil, err
	}
	if len(previousHost.Layers) == 0 {
		nextDevice, err := hipMirrorGemma4Q4DecodeState(driver, cfg, nextHost, firstNonEmptyString(mode, previousDevice.mode))
		if err != nil {
			return nil, err
		}
		if err := previousDevice.Close(); err != nil {
			_ = nextDevice.Close()
			return nil, err
		}
		return nextDevice, nil
	}
	if len(previousHost.Layers) != len(nextHost.Layers) || len(previousDevice.layers) != len(nextHost.Layers) {
		return nil, core.E("rocm.hip.Gemma4Q4DeviceKV", "decode state layer counts must match for device update", nil)
	}
	mode = firstNonEmptyString(mode, previousDevice.mode)
	if mode == "" {
		mode = rocmKVCacheModeFP16
	}
	if previousDevice.mode != "" && mode != previousDevice.mode {
		return nil, core.E("rocm.hip.Gemma4Q4DeviceKV", "device KV mode mismatch", nil)
	}
	nextDevice := &hipGemma4Q4DeviceDecodeState{mode: mode, layers: make([]hipGemma4Q4DeviceLayerKVState, 0, len(nextHost.Layers))}
	type ownershipAction struct {
		oldLayer *hipGemma4Q4DeviceLayerKVState
		newCache *rocmDeviceKVCache
		append   bool
	}
	actions := make([]ownershipAction, 0, len(nextHost.Layers))
	success := false
	defer func() {
		if !success {
			_ = nextDevice.Close()
		}
	}()
	for index := range nextHost.Layers {
		oldLayer := &previousDevice.layers[index]
		layerCfg := cfg.Layers[index]
		if hipGemma4Q4LayerStateCanAppendDeviceKV(layerCfg, previousHost.Layers[index], nextHost.Layers[index]) {
			keyStart := len(nextHost.Layers[index].Keys) - layerCfg.HeadDim
			valueStart := len(nextHost.Layers[index].Values) - layerCfg.HeadDim
			nextCache, err := oldLayer.cache.withAppendedToken(nextHost.Layers[index].Keys[keyStart:], nextHost.Layers[index].Values[valueStart:])
			if err != nil {
				return nil, err
			}
			table, err := nextCache.KernelDescriptorTable()
			if err != nil {
				_ = nextCache.closePagesFrom(oldLayer.cache.PageCount())
				return nil, err
			}
			launch, err := nextCache.KernelLaunchDescriptor(table)
			if err != nil {
				_ = table.Close()
				_ = nextCache.closePagesFrom(oldLayer.cache.PageCount())
				return nil, err
			}
			nextDevice.layers = append(nextDevice.layers, hipGemma4Q4DeviceLayerKVState{cache: nextCache, descriptorTable: table, launch: launch})
			nextDevice.appendLayers++
			actions = append(actions, ownershipAction{oldLayer: oldLayer, newCache: nextCache, append: true})
			continue
		}
		layer, err := hipMirrorGemma4Q4LayerDecodeState(driver, layerCfg, nextHost.Layers[index], mode)
		if err != nil {
			return nil, err
		}
		nextDevice.layers = append(nextDevice.layers, layer)
		nextDevice.remirrorLayers++
		actions = append(actions, ownershipAction{oldLayer: oldLayer})
	}
	for _, action := range actions {
		if action.append {
			if err := action.oldLayer.cache.transferPagesTo(action.newCache); err != nil {
				return nil, err
			}
		} else if err := action.oldLayer.cache.Close(); err != nil {
			return nil, err
		}
		if err := action.oldLayer.descriptorTable.Close(); err != nil {
			return nil, err
		}
	}
	previousDevice.layers = nil
	previousDevice.closed = true
	success = true
	return nextDevice, nil
}

func hipFinalizeGemma4Q4ForwardDeviceState(previous, next *hipGemma4Q4DeviceDecodeState) error {
	if next == nil || previous == nil {
		return nil
	}
	if previous.closed {
		return core.E("rocm.hip.Gemma4Q4DeviceKV", "previous device decode state is closed", nil)
	}
	if len(previous.layers) != len(next.layers) {
		return core.E("rocm.hip.Gemma4Q4DeviceKV", "device state layer counts must match for forward transfer", nil)
	}
	for index := range next.layers {
		oldLayer := &previous.layers[index]
		newLayer := &next.layers[index]
		if newLayer.cache.borrowsPagesFrom(oldLayer.cache) {
			if err := oldLayer.cache.transferPagesTo(newLayer.cache); err != nil {
				return err
			}
		} else if err := oldLayer.cache.Close(); err != nil {
			return err
		}
		if err := oldLayer.descriptorTable.Close(); err != nil {
			return err
		}
	}
	previous.layers = nil
	previous.closed = true
	return nil
}

func hipMirrorGemma4Q4LayerDecodeState(driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, layerState hipGemma4Q4LayerKVState, mode string) (hipGemma4Q4DeviceLayerKVState, error) {
	if len(layerState.Keys) == 0 || len(layerState.Values) == 0 {
		return hipGemma4Q4DeviceLayerKVState{}, core.E("rocm.hip.Gemma4Q4DeviceKV", "decode state layer has no KV tokens", nil)
	}
	host, err := newROCmKVCache(mode, defaultROCmKVBlockSize)
	if err != nil {
		return hipGemma4Q4DeviceLayerKVState{}, err
	}
	if err := host.AppendVectors(0, cfg.HeadDim, cfg.HeadDim, layerState.Keys, layerState.Values); err != nil {
		return hipGemma4Q4DeviceLayerKVState{}, err
	}
	device, err := host.MirrorToDevice(driver)
	if err != nil {
		return hipGemma4Q4DeviceLayerKVState{}, err
	}
	table, err := device.KernelDescriptorTable()
	if err != nil {
		_ = device.Close()
		return hipGemma4Q4DeviceLayerKVState{}, err
	}
	launch, err := device.KernelLaunchDescriptor(table)
	if err != nil {
		_ = table.Close()
		_ = device.Close()
		return hipGemma4Q4DeviceLayerKVState{}, err
	}
	return hipGemma4Q4DeviceLayerKVState{cache: device, descriptorTable: table, launch: launch}, nil
}

func hipGemma4Q4LayerStateCanAppendDeviceKV(cfg hipGemma4Q4Layer0Config, previous, next hipGemma4Q4LayerKVState) bool {
	if cfg.HeadDim <= 0 || len(previous.Keys) == 0 || len(previous.Values) == 0 {
		return false
	}
	if len(next.Keys) != len(previous.Keys)+cfg.HeadDim || len(next.Values) != len(previous.Values)+cfg.HeadDim {
		return false
	}
	return hipFloat32SlicesEqual(previous.Keys, next.Keys[:len(previous.Keys)]) && hipFloat32SlicesEqual(previous.Values, next.Values[:len(previous.Values)])
}

func hipFloat32SlicesEqual(left, right []float32) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func (state *hipGemma4Q4DeviceDecodeState) Close() error {
	if state == nil || state.closed {
		return nil
	}
	var lastErr error
	for index := range state.layers {
		if err := state.layers[index].Close(); err != nil {
			lastErr = err
		}
	}
	state.closed = true
	return lastErr
}

func (state *hipGemma4Q4DeviceDecodeState) LayerCount() int {
	if state == nil {
		return 0
	}
	return len(state.layers)
}

func (state *hipGemma4Q4DeviceDecodeState) layerCache(index int) *rocmDeviceKVCache {
	if state == nil || index < 0 || index >= len(state.layers) {
		return nil
	}
	return state.layers[index].cache
}

func (state *hipGemma4Q4DeviceDecodeState) LayerTokenCounts() []int {
	if state == nil {
		return nil
	}
	counts := make([]int, 0, len(state.layers))
	for _, layer := range state.layers {
		counts = append(counts, layer.cache.TokenCount())
	}
	return counts
}

func (state *hipGemma4Q4DeviceDecodeState) MemoryBytes() uint64 {
	if state == nil {
		return 0
	}
	var total uint64
	for _, layer := range state.layers {
		total += layer.cache.MemoryBytes()
		if layer.descriptorTable != nil {
			total += layer.descriptorTable.SizeBytes()
		}
	}
	return total
}

func (state *hipGemma4Q4DeviceDecodeState) CompatibleWithHostState(cfg hipGemma4Q4ForwardConfig, host hipGemma4Q4DecodeState, mode string) error {
	if state == nil {
		return nil
	}
	if state.closed {
		return core.E("rocm.hip.Gemma4Q4DeviceKV", "device decode state is closed", nil)
	}
	if err := cfg.validate(); err != nil {
		return err
	}
	if err := host.validate(cfg); err != nil {
		return err
	}
	if len(host.Layers) == 0 {
		return core.E("rocm.hip.Gemma4Q4DeviceKV", "prior device state requires host KV state", nil)
	}
	if len(state.layers) != len(host.Layers) || len(state.layers) != len(cfg.Layers) {
		return core.E("rocm.hip.Gemma4Q4DeviceKV", "device state layer count must match host state", nil)
	}
	mode = firstNonEmptyString(mode, state.mode)
	if mode == "" {
		mode = rocmKVCacheModeFP16
	}
	if state.mode != "" && state.mode != mode {
		return core.E("rocm.hip.Gemma4Q4DeviceKV", "device KV mode mismatch", nil)
	}
	for index, layer := range state.layers {
		if layer.cache == nil {
			return core.E("rocm.hip.Gemma4Q4DeviceKV", core.Sprintf("device layer %d cache is nil", index), nil)
		}
		if layer.cache.closed {
			return core.E("rocm.hip.Gemma4Q4DeviceKV", core.Sprintf("device layer %d cache is closed", index), nil)
		}
		if layer.cache.mode != mode {
			return core.E("rocm.hip.Gemma4Q4DeviceKV", core.Sprintf("device layer %d cache mode mismatch", index), nil)
		}
		layerCfg := cfg.Layers[index]
		hostTokens := len(host.Layers[index].Keys) / layerCfg.HeadDim
		if layer.cache.TokenCount() != hostTokens {
			return core.E("rocm.hip.Gemma4Q4DeviceKV", core.Sprintf("device layer %d token count mismatch", index), nil)
		}
		keyWidth, valueWidth, ok := layer.cache.LastVectorWidths()
		if !ok || keyWidth != layerCfg.HeadDim || valueWidth != layerCfg.HeadDim {
			return core.E("rocm.hip.Gemma4Q4DeviceKV", core.Sprintf("device layer %d KV width mismatch", index), nil)
		}
	}
	return nil
}

func (state *hipGemma4Q4DeviceDecodeState) HostState() (hipGemma4Q4DecodeState, error) {
	if state == nil {
		return hipGemma4Q4DecodeState{}, core.E("rocm.hip.Gemma4Q4DeviceKV", "device decode state is nil", nil)
	}
	if state.closed {
		return hipGemma4Q4DecodeState{}, core.E("rocm.hip.Gemma4Q4DeviceKV", "device decode state is closed", nil)
	}
	hostState := hipGemma4Q4DecodeState{Layers: make([]hipGemma4Q4LayerKVState, 0, len(state.layers))}
	for index, layer := range state.layers {
		hostCache, err := layer.cache.hostCache()
		if err != nil {
			return hipGemma4Q4DecodeState{}, core.E("rocm.hip.Gemma4Q4DeviceKV", core.Sprintf("copy layer %d", index), err)
		}
		keys, values, err := hostCache.Restore(0, hostCache.TokenCount())
		if err != nil {
			return hipGemma4Q4DecodeState{}, core.E("rocm.hip.Gemma4Q4DeviceKV", core.Sprintf("restore layer %d", index), err)
		}
		hostState.Layers = append(hostState.Layers, hipGemma4Q4LayerKVState{Keys: keys, Values: values})
	}
	return hostState, nil
}

func (state *hipGemma4Q4DeviceDecodeState) Labels() map[string]string {
	labels := map[string]string{
		"gemma4_q4_device_kv_backing": "hip_device_mirror",
		"gemma4_q4_device_kv_layers":  core.Sprintf("%d", state.LayerCount()),
		"production_kv_cache_backing": hipKernelStatusNotLinked,
	}
	if state == nil {
		return labels
	}
	labels["gemma4_q4_device_kv_mode"] = state.mode
	labels["gemma4_q4_device_kv_bytes"] = core.Sprintf("%d", state.MemoryBytes())
	labels["gemma4_q4_device_kv_append_layers"] = core.Sprintf("%d", state.appendLayers)
	labels["gemma4_q4_device_kv_remirror_layers"] = core.Sprintf("%d", state.remirrorLayers)
	counts := state.LayerTokenCounts()
	if len(counts) > 0 {
		minTokens := counts[0]
		maxTokens := counts[0]
		for _, count := range counts[1:] {
			if count < minTokens {
				minTokens = count
			}
			if count > maxTokens {
				maxTokens = count
			}
		}
		labels["gemma4_q4_device_kv_min_tokens"] = core.Sprintf("%d", minTokens)
		labels["gemma4_q4_device_kv_max_tokens"] = core.Sprintf("%d", maxTokens)
	}
	return labels
}
