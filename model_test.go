//go:build linux && amd64

package rocm

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"dappco.re/go/rocm/internal/llamacpp"
	"forge.lthn.ai/core/go-inference"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return fn(r)
}

func newClientBackedModel(client *llamacpp.Client) *rocmModel {
	return &rocmModel{
		server: &server{
			llamaClient:   client,
			processExited: make(chan struct{}),
		},
	}
}

func newHTTPBackedModel(ts *httptest.Server) *rocmModel {
	return newClientBackedModel(llamacpp.NewClient(ts.URL))
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
		if r.URL.Path != "/v1/completions" {
			t.Errorf("r.URL.Path = %q, want %q", r.URL.Path, "/v1/completions")
		}

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

	if err := m.Err(); err != nil {
		t.Fatalf("m.Err(): %v", err)
	}
	want := []string{"Hello", " world"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("tokens = %v, want %v", got, want)
	}

	met := m.Metrics()
	if met.PromptTokens != 2 {
		t.Errorf("PromptTokens = %d, want 2", met.PromptTokens)
	}
	if met.GeneratedTokens != 2 {
		t.Errorf("GeneratedTokens = %d, want 2", met.GeneratedTokens)
	}
	if met.PrefillDuration < 20*time.Millisecond {
		t.Errorf("PrefillDuration = %s, want >= 20ms", met.PrefillDuration)
	}
	if met.DecodeDuration < 20*time.Millisecond {
		t.Errorf("DecodeDuration = %s, want >= 20ms", met.DecodeDuration)
	}
	if met.TotalDuration < 45*time.Millisecond {
		t.Errorf("TotalDuration = %s, want >= 45ms", met.TotalDuration)
	}
	if met.PrefillTokensPerSec <= 0 {
		t.Errorf("PrefillTokensPerSec = %v, want > 0", met.PrefillTokensPerSec)
	}
	if met.DecodeTokensPerSec <= 0 {
		t.Errorf("DecodeTokensPerSec = %v, want > 0", met.DecodeTokensPerSec)
	}
}

func TestClassify_AppliesGenerateOptions(t *testing.T) {
	var (
		mu       sync.Mutex
		requests []llamacpp.CompletionRequest
	)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			t.Errorf("r.URL.Path = %q, want %q", r.URL.Path, "/v1/completions")
		}

		var req llamacpp.CompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
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
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}
	if results[0].Token.Text != "label" {
		t.Errorf("results[0].Token.Text = %q, want %q", results[0].Token.Text, "label")
	}

	mu.Lock()
	if len(requests) != 1 {
		mu.Unlock()
		t.Fatalf("len(requests) = %d, want 1", len(requests))
	}
	req := requests[0]
	mu.Unlock()

	if req.Prompt != "hello world" {
		t.Errorf("req.Prompt = %q, want %q", req.Prompt, "hello world")
	}
	if req.MaxTokens != 1 {
		t.Errorf("req.MaxTokens = %d, want 1", req.MaxTokens)
	}
	if math.Abs(req.Temperature-0.7) > 0.001 {
		t.Errorf("req.Temperature = %v, want ~0.7", req.Temperature)
	}
	if req.TopK != 42 {
		t.Errorf("req.TopK = %d, want 42", req.TopK)
	}
	if math.Abs(req.TopP-0.91) > 0.001 {
		t.Errorf("req.TopP = %v, want ~0.91", req.TopP)
	}
	if math.Abs(req.RepeatPenalty-1.3) > 0.001 {
		t.Errorf("req.RepeatPenalty = %v, want ~1.3", req.RepeatPenalty)
	}
}

func TestBatchGenerate_MetricsAggregatePrefillAndDecode(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			t.Errorf("r.URL.Path = %q, want %q", r.URL.Path, "/v1/completions")
		}

		time.Sleep(15 * time.Millisecond)
		writeSSEEvent(w, `{"choices":[{"text":"A","finish_reason":null}]}`)
		time.Sleep(15 * time.Millisecond)
		writeSSEEvent(w, `{"choices":[{"text":"B","finish_reason":null}]}`)
		writeSSEEvent(w, "[DONE]")
	}))
	defer ts.Close()

	m := newHTTPBackedModel(ts)

	results, err := m.BatchGenerate(context.Background(), []string{"alpha beta", "gamma delta"}, inference.WithMaxTokens(2))
	if err != nil {
		t.Fatalf("BatchGenerate: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("len(results) = %d, want 2", len(results))
	}
	if len(results[0].Tokens) != 2 {
		t.Fatalf("len(results[0].Tokens) = %d, want 2", len(results[0].Tokens))
	}
	if len(results[1].Tokens) != 2 {
		t.Fatalf("len(results[1].Tokens) = %d, want 2", len(results[1].Tokens))
	}

	met := m.Metrics()
	if met.PromptTokens != 4 {
		t.Errorf("PromptTokens = %d, want 4", met.PromptTokens)
	}
	if met.GeneratedTokens != 4 {
		t.Errorf("GeneratedTokens = %d, want 4", met.GeneratedTokens)
	}
	if met.PrefillDuration < 20*time.Millisecond {
		t.Errorf("PrefillDuration = %s, want >= 20ms", met.PrefillDuration)
	}
	if met.DecodeDuration < 20*time.Millisecond {
		t.Errorf("DecodeDuration = %s, want >= 20ms", met.DecodeDuration)
	}
	if met.TotalDuration < 50*time.Millisecond {
		t.Errorf("TotalDuration = %s, want >= 50ms", met.TotalDuration)
	}
	if met.PrefillTokensPerSec <= 0 {
		t.Errorf("PrefillTokensPerSec = %v, want > 0", met.PrefillTokensPerSec)
	}
	if met.DecodeTokensPerSec <= 0 {
		t.Errorf("DecodeTokensPerSec = %v, want > 0", met.DecodeTokensPerSec)
	}
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
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if results != nil {
		t.Errorf("results = %v, want nil", results)
	}
	if requestCount != 1 {
		t.Errorf("requestCount = %d, want 1", requestCount)
	}
	if !strings.Contains(err.Error(), "rocm.Classify") {
		t.Errorf("err = %v, want contains %q", err, "rocm.Classify")
	}
	if !strings.Contains(err.Error(), "cancelled before prompt 1") {
		t.Errorf("err = %v, want contains %q", err, "cancelled before prompt 1")
	}

	metrics := m.Metrics()
	if metrics.PromptTokens != 2 {
		t.Errorf("PromptTokens = %d, want 2", metrics.PromptTokens)
	}
	if metrics.GeneratedTokens != 1 {
		t.Errorf("GeneratedTokens = %d, want 1", metrics.GeneratedTokens)
	}
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
	if err != nil {
		t.Fatalf("BatchGenerate: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("len(results) = %d, want 2", len(results))
	}
	if requestCount != 1 {
		t.Errorf("requestCount = %d, want 1", requestCount)
	}
	if len(results[0].Tokens) != 1 {
		t.Fatalf("len(results[0].Tokens) = %d, want 1", len(results[0].Tokens))
	}
	if results[1].Err == nil {
		t.Fatal("results[1].Err = nil, want error")
	}
	if !strings.Contains(results[1].Err.Error(), "rocm.BatchGenerate") {
		t.Errorf("results[1].Err = %v, want contains %q", results[1].Err, "rocm.BatchGenerate")
	}
	if !strings.Contains(results[1].Err.Error(), "cancelled before start") {
		t.Errorf("results[1].Err = %v, want contains %q", results[1].Err, "cancelled before start")
	}

	metrics := m.Metrics()
	if metrics.PromptTokens != 2 {
		t.Errorf("PromptTokens = %d, want 2", metrics.PromptTokens)
	}
	if metrics.GeneratedTokens != 1 {
		t.Errorf("GeneratedTokens = %d, want 1", metrics.GeneratedTokens)
	}
}

func TestGenerate_TruncatedStreamSetsLastError(t *testing.T) {
	client := llamacpp.NewClientWithHTTPClient("http://llama.test", &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			if r.URL.Path != "/v1/completions" {
				t.Errorf("r.URL.Path = %q, want %q", r.URL.Path, "/v1/completions")
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body: io.NopCloser(strings.NewReader(
					"data: " + `{"choices":[{"text":"partial","finish_reason":null}]}` + "\n\n",
				)),
				Request: r,
			}, nil
		}),
	})

	m := newClientBackedModel(client)

	var got []string
	for tok := range m.Generate(context.Background(), "hello", inference.WithMaxTokens(1)) {
		got = append(got, tok.Text)
	}

	want := []string{"partial"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("tokens = %v, want %v", got, want)
	}
	if err := m.Err(); err == nil {
		t.Fatal("m.Err() = nil, want error")
	} else if !strings.Contains(err.Error(), "stream ended before [DONE]") {
		t.Errorf("m.Err() = %v, want contains %q", err, "stream ended before [DONE]")
	}
}
