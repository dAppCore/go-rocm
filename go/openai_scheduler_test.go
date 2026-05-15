// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/openai"
)

func TestOpenAI_NewOpenAIServiceMux_Good_CacheEndpointsUseROCmModel(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen3"}}
	mux := NewOpenAIServiceMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))

	warmReq := httptest.NewRequest(http.MethodPost, openaicompat.DefaultCacheWarmPath, strings.NewReader(`{"model":"qwen","tokens":[1,2,3],"labels":{"tenant":"a"}}`))
	warmRec := httptest.NewRecorder()
	mux.ServeHTTP(warmRec, warmReq)
	if warmRec.Code != http.StatusOK || !strings.Contains(warmRec.Body.String(), `"token_count":3`) {
		t.Fatalf("cache warm status = %d body=%s, want warm result", warmRec.Code, warmRec.Body.String())
	}

	statsReq := httptest.NewRequest(http.MethodGet, openaicompat.DefaultCacheStatsPath+"?model=qwen", nil)
	statsRec := httptest.NewRecorder()
	mux.ServeHTTP(statsRec, statsReq)
	if statsRec.Code != http.StatusOK || !strings.Contains(statsRec.Body.String(), `"blocks":1`) || !strings.Contains(statsRec.Body.String(), `"cached_tokens":"3"`) {
		t.Fatalf("cache stats status = %d body=%s, want warmed stats", statsRec.Code, statsRec.Body.String())
	}

	clearReq := httptest.NewRequest(http.MethodPost, openaicompat.DefaultCacheClearPath, strings.NewReader(`{"model":"qwen","labels":{"tenant":"a"}}`))
	clearRec := httptest.NewRecorder()
	mux.ServeHTTP(clearRec, clearReq)
	if clearRec.Code != http.StatusOK || !strings.Contains(clearRec.Body.String(), `"evictions":1`) {
		t.Fatalf("cache clear status = %d body=%s, want cleared stats", clearRec.Code, clearRec.Body.String())
	}
}

func TestOpenAI_NewOpenAIServiceMux_Bad_CacheWarmRejectsEmptyInput(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen3"}}
	mux := NewOpenAIServiceMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultCacheWarmPath, strings.NewReader(`{"model":"qwen"}`))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "prompt or tokens are required") {
		t.Fatalf("cache warm status = %d body=%s, want bad request before cache service", rec.Code, rec.Body.String())
	}
}

func TestOpenAI_NewOpenAIServiceMux_Bad_ROCmModelEmbeddingReportsKernelNotLinked(t *testing.T) {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "bert"},
		native: &fakeNativeModel{kernelStatus: hipKernelStatus{
			Embedding: hipKernelStatusLinked,
		}},
	}
	mux := NewOpenAIServiceMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"bert": model}))
	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultEmbeddingsPath, strings.NewReader(`{"model":"bert","input":"hello"}`))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError || !strings.Contains(rec.Body.String(), "native embedding kernels are not linked yet") {
		t.Fatalf("embedding status = %d body=%s, want kernel-not-linked error", rec.Code, rec.Body.String())
	}
}

func TestOpenAI_NewOpenAIServiceMux_Bad_ROCmModelRerankReportsKernelNotLinked(t *testing.T) {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "bert"},
		native: &fakeNativeModel{kernelStatus: hipKernelStatus{
			Rerank: hipKernelStatusLinked,
		}},
	}
	mux := NewOpenAIServiceMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"bert": model}))
	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultRerankPath, strings.NewReader(`{"model":"bert","query":"core","documents":["a"]}`))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError || !strings.Contains(rec.Body.String(), "native rerank kernels are not linked yet") {
		t.Fatalf("rerank status = %d body=%s, want kernel-not-linked error", rec.Code, rec.Body.String())
	}
}

func TestOpenAI_NewOpenAIServiceMux_Good_CancelEndpointUsesScheduledModel(t *testing.T) {
	scheduled, err := NewScheduledModel(&openAITestModel{tokens: []inference.Token{{Text: "ok"}}}, SchedulerConfig{QueueSize: 1})
	if err != nil {
		t.Fatalf("NewScheduledModel: %v", err)
	}
	defer scheduled.Close()
	mux := NewOpenAIServiceMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": scheduled}))
	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultCancelPath, strings.NewReader(`{"model":"qwen","id":"external"}`))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"id":"external"`) {
		t.Fatalf("cancel status = %d body=%s, want cancel response", rec.Code, rec.Body.String())
	}
}
