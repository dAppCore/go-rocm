// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"errors"
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestROCmRuntimeAuthorNative_Good_RocmModelExposesConcreteSurface(t *testing.T) {
	native := &fakeNativeModel{
		encodeByText:       map[string][]int32{"hello": {4, 5}},
		chatTemplateResult: "<chat>",
	}
	model := &rocmModel{
		native:    native,
		modelPath: "/models/lmstudio-community-gemma-4-e2b-it-6bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			VocabSize:    262144,
			NumLayers:    26,
			HiddenSize:   2304,
			QuantBits:    6,
			QuantGroup:   64,
		},
		modelLabels: map[string]string{
			"gemma4_size":       "E2B",
			"gemma4_quant_mode": "q6",
		},
		adapter: inference.AdapterIdentity{Hash: "adapter-hash", Path: "/adapters/a"},
	}

	var author ROCmRuntimeAuthorModel = model
	if author.UnderlyingModel() != native {
		t.Fatalf("UnderlyingModel() = %#v, want native handle", author.UnderlyingModel())
	}
	tokenizer := author.RuntimeTokenizer()
	if tokenizer == nil {
		t.Fatal("RuntimeTokenizer() = nil")
	}
	tokens := tokenizer.Encode("hello")
	if len(tokens) != 2 || tokens[0] != 4 || tokens[1] != 5 {
		t.Fatalf("RuntimeTokenizer().Encode = %v, want fake native tokens", tokens)
	}
	if text, err := tokenizer.ApplyChatTemplate([]inference.Message{{Role: "user", Content: "hi"}}); err != nil || text != "<chat>" {
		t.Fatalf("RuntimeTokenizer().ApplyChatTemplate = %q err=%v, want fake template", text, err)
	}

	if err := author.RequireTextRuntime("test"); err != nil {
		t.Fatalf("RequireTextRuntime() error = %v", err)
	}
	release, err := author.AcquireSlot(context.Background())
	if err != nil {
		t.Fatalf("AcquireSlot() error = %v", err)
	}
	release()
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := author.AcquireSlot(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatalf("AcquireSlot(cancelled) error = %v, want context.Canceled", err)
	}
	var called bool
	if err := author.WithDevice(func() { called = true }); err != nil || !called {
		t.Fatalf("WithDevice() called=%v err=%v, want callback", called, err)
	}
	author.AcquirePromptCache()()

	if author.RuntimeCacheService() == nil {
		t.Fatal("RuntimeCacheService() = nil")
	}
	if _, err := author.RuntimeCacheService().WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}}); err != nil {
		t.Fatalf("WarmCache() error = %v", err)
	}
	blockEntry, blockPrefix := author.PromptCacheMatchWithHidden([]int32{1, 2, 3, 4})
	if blockEntry == nil || blockPrefix != 3 || len(blockEntry.CacheBlocks) != 1 {
		t.Fatalf("PromptCacheMatchWithHidden(block cache) = %+v prefix=%d, want warmed block entry", blockEntry, blockPrefix)
	}
	cacheProfile, err := author.RuntimeCacheProfile(context.Background())
	if err != nil || !cacheProfile.Matched() || cacheProfile.MaxCacheTokens != 3 {
		t.Fatalf("RuntimeCacheProfile() = %+v err=%v, want live cache profile", cacheProfile, err)
	}
	requestCache := author.NewCachesWithRequestFixedSize(9)
	if requestCache == nil {
		t.Fatal("NewCachesWithRequestFixedSize() = nil")
	}
	if warmed, err := requestCache.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{4, 5}, Labels: map[string]string{"source": "runtime_author"}}); err != nil ||
		warmed.Labels["request_fixed_size"] != "9" ||
		warmed.Labels["runtime_author"] != "true" {
		t.Fatalf("request cache warm = %+v err=%v, want request-sized runtime-author labels", warmed, err)
	}
	if author.RuntimeStateSession() == nil {
		t.Fatal("RuntimeStateSession() = nil")
	}

	routePlan := author.RuntimeModelRoutePlan()
	if !routePlan.Matched() ||
		!routePlan.RuntimeAuthorPlan.Matched() ||
		!routePlan.RuntimeAuthorPlan.HasCapability(rocmmodel.RuntimeAuthorUnderlyingModel) ||
		!routePlan.RuntimeAuthorPlan.HasCapability(rocmmodel.RuntimeAuthorRuntimeTokenizer) ||
		!routePlan.RuntimeAuthorPlan.HasCapability(rocmmodel.RuntimeAuthorCacheProfile) {
		t.Fatalf("RuntimeModelRoutePlan() = %+v, want runtime-author route plan", routePlan)
	}
	authorPlan := author.RuntimeAuthorPlan()
	if !authorPlan.Matched() || authorPlan.Labels["engine_runtime_author_plan_contract"] != rocmmodel.RuntimeAuthorPlanContract {
		t.Fatalf("RuntimeAuthorPlan() = %+v, want cloned plan labels", authorPlan)
	}
	authorPlan.Labels["engine_runtime_author_plan_contract"] = "mutated"
	nextAuthorPlan := author.RuntimeAuthorPlan()
	if nextAuthorPlan.Labels["engine_runtime_author_plan_contract"] != rocmmodel.RuntimeAuthorPlanContract {
		t.Fatalf("RuntimeAuthorPlan() leaked mutable labels: %+v", nextAuthorPlan.Labels)
	}
	resolvedPlan, ok := RuntimeAuthorPlanForModel(model)
	if !ok || !resolvedPlan.Matched() || resolvedPlan.Architecture != "gemma4_text" {
		t.Fatalf("RuntimeAuthorPlanForModel() = %+v ok=%v, want gemma4 runtime plan", resolvedPlan, ok)
	}
	if !resolvedPlan.HasCapability(rocmmodel.RuntimeAuthorNewCachesWithRequestFixedSize) ||
		!resolvedPlan.HasCapability(rocmmodel.RuntimeAuthorGenerationFixedCacheSize) ||
		!resolvedPlan.HasCapability(rocmmodel.RuntimeAuthorStorePromptCacheEntry) ||
		!resolvedPlan.HasCapability(rocmmodel.RuntimeAuthorRestoreCaches) {
		t.Fatalf("RuntimeAuthorPlanForModel() capabilities = %v, want concrete cache author methods", resolvedPlan.CapabilityIDs)
	}
	if got := author.GenerationFixedSlidingCacheSize(3, 4); got != 7 {
		t.Fatalf("GenerationFixedSlidingCacheSize(3,4) = %d, want 7", got)
	}
	if got := author.GenerationFixedSlidingCacheSize(3, 0); got != 0 {
		t.Fatalf("GenerationFixedSlidingCacheSize(3,0) = %d, want grow-as-needed zero", got)
	}

	entry := NewROCmPromptCacheEntry(
		[]int32{4, 5},
		nil,
		[]inference.StateRef{{Kind: "hidden", URI: "state://hidden"}},
		[]inference.StateRef{{Kind: "logits", URI: "state://logits"}},
		map[string]string{"cache_mode": "block-prefix", "source": "stored"},
	)
	author.StorePromptCacheEntry(entry)
	entry.Tokens[0] = 99
	entry.HiddenRefs[0].URI = "mutated"
	stored, storedPrefix := author.PromptCacheMatchWithHidden([]int32{4, 5, 6})
	if stored == nil || storedPrefix != 2 ||
		len(stored.Hidden()) != 1 ||
		stored.Hidden()[0].URI != "state://hidden" ||
		len(stored.Logits()) != 1 ||
		stored.Logits()[0].URI != "state://logits" {
		t.Fatalf("PromptCacheMatchWithHidden(stored) = %+v prefix=%d, want cloned hidden/logit refs", stored, storedPrefix)
	}
	hidden := stored.Hidden()
	hidden[0].URI = "mutated-again"
	nextStored, _ := author.PromptCacheMatchWithHidden([]int32{4, 5, 6})
	if nextStored.Hidden()[0].URI != "state://hidden" {
		t.Fatalf("PromptCacheMatchWithHidden leaked hidden refs: %+v", nextStored.Hidden())
	}
	restoredCache, err := stored.RestoreCaches(context.Background(), storedPrefix, 11)
	if err != nil {
		t.Fatalf("RestoreCaches() error = %v", err)
	}
	restoredStats, err := restoredCache.CacheStats(context.Background())
	if err != nil || restoredStats.Blocks != 1 || restoredStats.CacheMode != "block-prefix" {
		t.Fatalf("restored cache stats = %+v err=%v, want one block-prefix cache", restoredStats, err)
	}
	model.adapter = inference.AdapterIdentity{Hash: "other-adapter"}
	if mismatched, prefix := author.PromptCacheMatchWithHidden([]int32{4, 5, 6}); mismatched != nil || prefix != 0 {
		t.Fatalf("PromptCacheMatchWithHidden(after adapter swap) = %+v prefix=%d, want adapter-key miss", mismatched, prefix)
	}
	model.adapter = inference.AdapterIdentity{Hash: "adapter-hash", Path: "/adapters/a"}

	boom := errors.New("boom")
	author.SetLastErr(boom)
	if model.Err() != boom {
		t.Fatalf("SetLastErr() did not update Err(): %v", model.Err())
	}
	metrics := inference.GenerateMetrics{GeneratedTokens: 7}
	author.SetLastMetrics(metrics)
	if model.Metrics().GeneratedTokens != 7 {
		t.Fatalf("SetLastMetrics() metrics = %+v, want generated token count", model.Metrics())
	}
	if key := author.AdapterCacheKey(); key != "adapter-hash" {
		t.Fatalf("AdapterCacheKey() = %q, want adapter hash", key)
	}
	if !author.PromptCacheEnabled() || !author.RuntimeCachesSnapshotSafe() || author.PromptCacheMinimum() != 1 {
		t.Fatalf("runtime cache flags enabled=%v snapshot=%v min=%d, want active cache surface",
			author.PromptCacheEnabled(), author.RuntimeCachesSnapshotSafe(), author.PromptCacheMinimum())
	}
}

func TestROCmRuntimeAuthorNative_Bad_NilModelHasNoRuntime(t *testing.T) {
	var model *rocmModel
	if model.UnderlyingModel() != nil || model.RuntimeTokenizer() != nil || model.RuntimeCacheService() != nil || model.RuntimeStateSession() != nil {
		t.Fatalf("nil model exposed runtime state")
	}
	if err := model.RequireTextRuntime("nil-test"); err == nil {
		t.Fatal("RequireTextRuntime(nil) error = nil, want failure")
	}
	if _, ok := RuntimeAuthorPlanForModel(nil); ok {
		t.Fatal("RuntimeAuthorPlanForModel(nil) ok = true, want false")
	}
}
