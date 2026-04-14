//go:build linux && amd64

package rocm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"dappco.re/go/rocm/internal/llamacpp"
	"forge.lthn.ai/core/go-inference"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newHTTPBackedModel(ts *httptest.Server) *rocmModel {
	return &rocmModel{
		server: &server{
			client: llamacpp.NewClient(ts.URL),
			exited: make(chan struct{}),
		},
	}
}

func writeSSEEvent(w http.ResponseWriter, payload string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Write([]byte("data: " + payload + "\n\n"))
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
}

func TestGenerate_MetricsSplitPrefillAndDecode(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/completions", r.URL.Path)

		time.Sleep(25 * time.Millisecond)
		writeSSEEvent(w, `{"choices":[{"text":"Hello","finish_reason":null}]}`)
		time.Sleep(25 * time.Millisecond)
		writeSSEEvent(w, `{"choices":[{"text":" world","finish_reason":null}]}`)
		writeSSEEvent(w, "[DONE]")
	}))
	defer ts.Close()

	m := newHTTPBackedModel(ts)

	var got []string
	for tok := range m.Generate(context.Background(), "hello world", inference.WithMaxTokens(2)) {
		got = append(got, tok.Text)
	}

	require.NoError(t, m.Err())
	assert.Equal(t, []string{"Hello", " world"}, got)

	met := m.Metrics()
	assert.Equal(t, 2, met.PromptTokens)
	assert.Equal(t, 2, met.GeneratedTokens)
	assert.GreaterOrEqual(t, met.PrefillDuration, 20*time.Millisecond)
	assert.GreaterOrEqual(t, met.DecodeDuration, 20*time.Millisecond)
	assert.GreaterOrEqual(t, met.TotalDuration, 45*time.Millisecond)
	assert.Greater(t, met.PrefillTokensPerSec, 0.0)
	assert.Greater(t, met.DecodeTokensPerSec, 0.0)
}

func TestClassify_AppliesGenerateOptions(t *testing.T) {
	var (
		mu       sync.Mutex
		requests []llamacpp.CompletionRequest
	)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/completions", r.URL.Path)

		var req llamacpp.CompletionRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		mu.Lock()
		requests = append(requests, req)
		mu.Unlock()

		writeSSEEvent(w, `{"choices":[{"text":"label","finish_reason":null}]}`)
		writeSSEEvent(w, "[DONE]")
	}))
	defer ts.Close()

	m := newHTTPBackedModel(ts)

	results, err := m.Classify(
		context.Background(),
		[]string{"hello world"},
		inference.WithTemperature(0.7),
		inference.WithTopK(42),
		inference.WithTopP(0.91),
		inference.WithRepeatPenalty(1.3),
	)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, "label", results[0].Token.Text)

	mu.Lock()
	require.Len(t, requests, 1)
	req := requests[0]
	mu.Unlock()

	assert.Equal(t, "hello world", req.Prompt)
	assert.Equal(t, 1, req.MaxTokens)
	assert.InDelta(t, 0.7, req.Temperature, 0.001)
	assert.Equal(t, 42, req.TopK)
	assert.InDelta(t, 0.91, req.TopP, 0.001)
	assert.InDelta(t, 1.3, req.RepeatPenalty, 0.001)
}

func TestBatchGenerate_MetricsAggregatePrefillAndDecode(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/completions", r.URL.Path)

		time.Sleep(15 * time.Millisecond)
		writeSSEEvent(w, `{"choices":[{"text":"A","finish_reason":null}]}`)
		time.Sleep(15 * time.Millisecond)
		writeSSEEvent(w, `{"choices":[{"text":"B","finish_reason":null}]}`)
		writeSSEEvent(w, "[DONE]")
	}))
	defer ts.Close()

	m := newHTTPBackedModel(ts)

	results, err := m.BatchGenerate(context.Background(), []string{"alpha beta", "gamma delta"}, inference.WithMaxTokens(2))
	require.NoError(t, err)
	require.Len(t, results, 2)
	require.Len(t, results[0].Tokens, 2)
	require.Len(t, results[1].Tokens, 2)

	met := m.Metrics()
	assert.Equal(t, 4, met.PromptTokens)
	assert.Equal(t, 4, met.GeneratedTokens)
	assert.GreaterOrEqual(t, met.PrefillDuration, 20*time.Millisecond)
	assert.GreaterOrEqual(t, met.DecodeDuration, 20*time.Millisecond)
	assert.GreaterOrEqual(t, met.TotalDuration, 50*time.Millisecond)
	assert.Greater(t, met.PrefillTokensPerSec, 0.0)
	assert.Greater(t, met.DecodeTokensPerSec, 0.0)
}

func TestClassify_ContextCancelledRecordsMetricsAndWrapsError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var requestCount int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		writeSSEEvent(w, `{"choices":[{"text":"label","finish_reason":null}]}`)
		writeSSEEvent(w, "[DONE]")
		if requestCount == 1 {
			cancel()
		}
	}))
	defer ts.Close()

	m := newHTTPBackedModel(ts)

	results, err := m.Classify(ctx, []string{"hello world", "goodbye world"})
	require.Error(t, err)
	assert.Nil(t, results)
	assert.Equal(t, 1, requestCount)
	assert.ErrorContains(t, err, "rocm.Classify")
	assert.ErrorContains(t, err, "cancelled before prompt 1")

	metrics := m.Metrics()
	assert.Equal(t, 2, metrics.PromptTokens)
	assert.Equal(t, 1, metrics.GeneratedTokens)
}

func TestBatchGenerate_ContextCancelledWrapsPerPromptError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var requestCount int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		writeSSEEvent(w, `{"choices":[{"text":"token","finish_reason":null}]}`)
		writeSSEEvent(w, "[DONE]")
		if requestCount == 1 {
			cancel()
		}
	}))
	defer ts.Close()

	m := newHTTPBackedModel(ts)

	results, err := m.BatchGenerate(ctx, []string{"hello world", "goodbye world"}, inference.WithMaxTokens(1))
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.Equal(t, 1, requestCount)
	require.Len(t, results[0].Tokens, 1)
	require.Error(t, results[1].Err)
	assert.ErrorContains(t, results[1].Err, "rocm.BatchGenerate")
	assert.ErrorContains(t, results[1].Err, "cancelled before start")

	metrics := m.Metrics()
	assert.Equal(t, 2, metrics.PromptTokens)
	assert.Equal(t, 1, metrics.GeneratedTokens)
}
