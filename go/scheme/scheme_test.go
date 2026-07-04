// SPDX-Licence-Identifier: EUPL-1.2

package scheme

import (
	"slices"
	"testing"
)

type cacheModeMixerInfo struct {
	mixerInfo
	mode string
}

func (mixer cacheModeMixerInfo) CacheMode() string { return mixer.mode }

func TestBuiltinsRegistered_Good(t *testing.T) {
	for _, kind := range []string{"full_attention", "softmax-hybrid", "mamba2", "rwkv7", "gla", "retnet", "deltanet", "gsa", "nsa", "moba", "mla"} {
		if _, ok := MixerFor(kind); !ok {
			t.Fatalf("MixerFor(%q) ok = false", kind)
		}
	}
	for _, mode := range []string{"default", "fp16", "q8", "k-q8-v-q4", "paged", "fixed", "turboquant", "mla-latent", "compaction", "compaction-full", "recurrent"} {
		if _, ok := CacheFor(mode); !ok {
			t.Fatalf("CacheFor(%q) ok = false", mode)
		}
	}
	for _, kind := range []string{"affine", "bf16", "mxfp4", "mxfp8", "nvfp4", "q4_0", "jangtq"} {
		if _, ok := QuantFor(kind); !ok {
			t.Fatalf("QuantFor(%q) ok = false", kind)
		}
	}
}

func TestCacheModeForMixer_Good_MatchesGoMLXFactoryContract(t *testing.T) {
	softmax, _ := MixerFor("softmax-hybrid")
	recurrentMixer, _ := MixerFor("mamba2")
	mla, _ := MixerFor("mla")

	if got := CacheModeForMixer(softmax); got != CacheModeDefault {
		t.Fatalf("CacheModeForMixer(softmax) = %q, want %q", got, CacheModeDefault)
	}
	if got := CacheModeForMixer(recurrentMixer); got != CacheModeRecurrent {
		t.Fatalf("CacheModeForMixer(recurrent) = %q, want %q", got, CacheModeRecurrent)
	}
	if got := CacheModeForMixer(mla); got != CacheModeMLALatent {
		t.Fatalf("CacheModeForMixer(mla) = %q, want %q", got, CacheModeMLALatent)
	}
	if got := CacheModeForMixer(cacheModeMixerInfo{
		mixerInfo: mixerInfo{kind: "override", state: StateKVCache},
		mode:      " MLA-Latent ",
	}); got != "mla-latent" {
		t.Fatalf("CacheModeForMixer(override) = %q, want normalized mla-latent", got)
	}
	if got := CacheModeForMixer(nil); got != "" {
		t.Fatalf("CacheModeForMixer(nil) = %q, want empty", got)
	}
}

func TestCacheForMixer_Good_ResolvesRegisteredCacheScheme(t *testing.T) {
	softmax, _ := MixerFor("softmax-hybrid")
	recurrentMixer, _ := MixerFor("mamba2")
	mla, _ := MixerFor("mla")

	if cache, ok := CacheForMixer(softmax); !ok || cache.Mode() != CacheModeDefault || cache.Serves() != StateKVCache {
		t.Fatalf("CacheForMixer(softmax) = %+v ok=%v, want default KV cache", cache, ok)
	}
	if cache, ok := CacheForMixer(recurrentMixer); !ok || cache.Mode() != CacheModeRecurrent || cache.Serves() != StateRecurrent {
		t.Fatalf("CacheForMixer(recurrent) = %+v ok=%v, want recurrent cache", cache, ok)
	}
	if cache, ok := CacheForMixer(mla); !ok || cache.Mode() != CacheModeMLALatent || cache.Serves() != StateKVCache {
		t.Fatalf("CacheForMixer(mla) = %+v ok=%v, want MLA latent cache", cache, ok)
	}
	if cache, ok := CacheForMixer(cacheModeMixerInfo{
		mixerInfo: mixerInfo{kind: "unknown", state: StateKVCache},
		mode:      "not-registered",
	}); ok || cache != nil {
		t.Fatalf("CacheForMixer(unregistered) = %+v ok=%v, want nil false", cache, ok)
	}
	if cache, ok := CacheForMixer(nil); ok || cache != nil {
		t.Fatalf("CacheForMixer(nil) = %+v ok=%v, want nil false", cache, ok)
	}
}

func TestStateCompatibility_Good(t *testing.T) {
	softmax, _ := MixerFor("softmax-hybrid")
	recurrentMixer, _ := MixerFor("mamba2")
	kv, _ := CacheFor("q8")
	recurrent, _ := CacheFor("recurrent")

	if !Compatible(softmax, kv) {
		t.Fatal("softmax-hybrid mixer should be compatible with q8 KV cache")
	}
	if Compatible(softmax, recurrent) {
		t.Fatal("softmax-hybrid mixer should not be compatible with recurrent holder")
	}
	if Compatible(recurrentMixer, kv) {
		t.Fatal("mamba2 mixer should not be compatible with KV cache")
	}
	if !Compatible(recurrentMixer, recurrent) {
		t.Fatal("mamba2 mixer should be compatible with recurrent holder")
	}
}

func TestRegisterAndResolve_Good(t *testing.T) {
	RegisterMixer(mixerInfo{kind: "fake_mixer", state: StateRecurrent})
	RegisterCache(cacheInfo{mode: "fake-cache", serves: StateRecurrent})
	RegisterQuant(quantInfo{kind: "fake-fp3", bits: 3})

	if mixer, ok := MixerFor("fake_mixer"); !ok || mixer.Kind() != "fake_mixer" || mixer.State() != StateRecurrent {
		t.Fatalf("MixerFor(fake_mixer) = %+v ok=%v, want registered mixer", mixer, ok)
	}
	if cache, ok := CacheFor("fake-cache"); !ok || cache.Mode() != "fake-cache" || cache.Serves() != StateRecurrent {
		t.Fatalf("CacheFor(fake-cache) = %+v ok=%v, want registered cache", cache, ok)
	}
	if quant, ok := QuantFor("fake-fp3"); !ok || quant.Kind() != "fake-fp3" || quant.Bits() != 3 {
		t.Fatalf("QuantFor(fake-fp3) = %+v ok=%v, want registered quant", quant, ok)
	}
	if !slices.Contains(MixerKinds(), "fake_mixer") ||
		!slices.Contains(CacheModes(), "fake-cache") ||
		!slices.Contains(QuantKinds(), "fake-fp3") {
		t.Fatalf("registry names missing registered entries: mixers=%v caches=%v quants=%v", MixerKinds(), CacheModes(), QuantKinds())
	}
}

func TestUnknownKind_Bad(t *testing.T) {
	if _, ok := MixerFor("does-not-exist"); ok {
		t.Fatal("unknown mixer resolved")
	}
	if _, ok := CacheFor("does-not-exist"); ok {
		t.Fatal("unknown cache resolved")
	}
	if _, ok := QuantFor("does-not-exist"); ok {
		t.Fatal("unknown quant resolved")
	}
	if Compatible(nil, nil) {
		t.Fatal("nil mixer/cache should be incompatible")
	}
}
