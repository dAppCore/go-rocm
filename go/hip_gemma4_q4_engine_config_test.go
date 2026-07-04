// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestHIPGemma4Q4EngineConfig_GoodDefaults(t *testing.T) {
	cfg := defaultHIPGemma4Q4EngineConfig()

	mode, err := cfg.deviceKVMode()
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, mode)
	core.AssertEqual(t, true, cfg.chunkedAttentionEnabled(1))
	core.AssertEqual(t, true, cfg.chunkedAttentionEnabled(4000))
	core.AssertEqual(t, rocmGemma4Q4DeviceKVBlockSize, cfg.deviceKVBlockSize())
	core.AssertEqual(t, rocmGemma4Q4GlobalDeviceKVBlockSize, cfg.globalDeviceKVBlockSize())
	core.AssertEqual(t, rocmGemma4Q4DeviceKVBlockSize, cfg.deviceKVBlockSizeForSlidingWindow(512))
	core.AssertEqual(t, rocmGemma4Q4GlobalDeviceKVBlockSize, cfg.deviceKVBlockSizeForSlidingWindow(0))
	core.AssertEqual(t, true, cfg.interleavedRowPagesEnabled())
	core.AssertEqual(t, false, cfg.pageAlignedLocalKVEnabled())
	ubatchTokens, err := cfg.prefillUBatchTokens()
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipGemma4Q4PrefillDefaultUBatchTokens, ubatchTokens)
	core.AssertEqual(t, hipGemma4Q4DefaultPrefillAttentionQueryChunkTokens, cfg.prefillAttentionQueryChunkTokens())
	core.AssertEqual(t, true, cfg.attentionWorkspaceNeeded(128, inference.GenerateConfig{}))
}

func TestHIPGemma4Q4EngineConfig_BadDeviceKVMode(t *testing.T) {
	_, err := (hipGemma4Q4EngineConfig{DeviceKVMode: "bad"}).deviceKVMode()
	if err == nil {
		t.Fatal("expected invalid Gemma4 q4 device KV mode error")
	}
}

func TestHIPGemma4Q4PackageKVModeDefaultsToLoadedEngineConfig(t *testing.T) {
	model := &hipLoadedModel{gemma4Q4Config: hipGemma4Q4EngineConfig{DeviceKVMode: rocmKVCacheModeQ8}}
	mode, err := hipGemma4Q4PackagePrefillKVMode(model, hipGemma4Q4ForwardConfig{}, hipPrefillRequest{})
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmKVCacheModeQ8, mode)

	mode, err = hipGemma4Q4PackageDecodeKVMode(model, hipDecodeRequest{})
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmKVCacheModeQ8, mode)

	mode, err = hipGemma4Q4PackagePrefillKVMode(model, hipGemma4Q4ForwardConfig{}, hipPrefillRequest{CacheMode: rocmKVCacheModeFP16})
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmKVCacheModeFP16, mode)
}

func TestHIPGemma4Q4EngineConfig_GoodKVLayoutOverrides(t *testing.T) {
	cfg := defaultHIPGemma4Q4EngineConfig()
	cfg.DeviceKVBlockSize = 16
	cfg.GlobalDeviceKVBlockSize = 256

	core.AssertEqual(t, 16, cfg.deviceKVBlockSize())
	core.AssertEqual(t, 256, cfg.globalDeviceKVBlockSize())
	core.AssertEqual(t, 16, cfg.deviceKVBlockSizeForSlidingWindow(512))
	core.AssertEqual(t, 256, cfg.deviceKVBlockSizeForSlidingWindow(0))

	cfg.DisableInterleavedRowPages = true
	core.AssertEqual(t, false, cfg.interleavedRowPagesEnabled())
	core.AssertEqual(t, rocmGemma4Q4DeviceKVBlockSize, cfg.deviceKVBlockSizeForSlidingWindow(512))

	cfg.PageAlignedLocalKV = true
	core.AssertEqual(t, true, cfg.pageAlignedLocalKVEnabled())
}

func TestHIPGemma4Q4EngineConfig_BadPrefillUBatchTokens(t *testing.T) {
	cfg := defaultHIPGemma4Q4EngineConfig()
	cfg.PrefillUBatchTokens = 0

	_, err := cfg.prefillUBatchTokens()
	if err == nil {
		t.Fatal("expected invalid Gemma4 q4 prefill ubatch error")
	}
}

func TestHIPGemma4Q4EngineConfig_GoodNegativePrefillChunkFallsBack(t *testing.T) {
	cfg := defaultHIPGemma4Q4EngineConfig()
	cfg.PrefillAttentionQueryChunkTokens = -1

	core.AssertEqual(t, hipGemma4Q4DefaultPrefillAttentionQueryChunkTokens, cfg.prefillAttentionQueryChunkTokens())
}
