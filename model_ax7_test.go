//go:build linux && amd64

package rocm

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"reflect"
	"strings"
	"testing"
	"time"

	"dappco.re/go/inference"
	"dappco.re/go/rocm/internal/llamacpp"
)

func TestModel_Model_Generate_Good(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			t.Fatalf("r.URL.Path = %q, want /v1/completions", r.URL.Path)
		}
		writeSSEEvent(w, `{"choices":[{"text":"Hello","finish_reason":null}]}`)
		writeSSEEvent(w, `{"choices":[{"text":" ROCm","finish_reason":null}]}`)
		writeSSEEvent(w, "[DONE]")
	}))
	defer ts.Close()

	model := newHTTPBackedModel(ts)
	var got []string
	for token := range model.Generate(context.Background(), "hello rocm", inference.WithMaxTokens(2)) {
		got = append(got, token.Text)
	}
	if err := model.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil", err)
	}
	if !reflect.DeepEqual(got, []string{"Hello", " ROCm"}) {
		t.Fatalf("tokens = %v, want Hello ROCm chunks", got)
	}
}

func TestModel_Model_Generate_Bad(t *testing.T) {
	processExited := make(chan struct{})
	close(processExited)
	model := &rocmModel{server: &server{processExited: processExited}}

	var count int
	for range model.Generate(context.Background(), "hello") {
		count++
	}
	if count != 0 {
		t.Fatalf("token count = %d, want 0", count)
	}
	if err := model.Err(); err == nil {
		t.Fatal("Err() = nil, want server exit error")
	}
}

func TestModel_Model_Generate_Ugly(t *testing.T) {
	client := llamacpp.NewClientWithHTTPClient("http://llama.test", &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader("data: {\"choices\":[{\"text\":\"partial\"}]}\n\n")),
				Request:    r,
			}, nil
		}),
	})
	model := newClientBackedModel(client)

	var got []string
	for token := range model.Generate(context.Background(), "hello") {
		got = append(got, token.Text)
	}
	if !reflect.DeepEqual(got, []string{"partial"}) {
		t.Fatalf("tokens = %v, want partial token", got)
	}
	if err := model.Err(); err == nil || !strings.Contains(err.Error(), "stream ended before [DONE]") {
		t.Fatalf("Err() = %v, want truncated stream error", err)
	}
}

func TestModel_Model_Chat_Good(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("r.URL.Path = %q, want /v1/chat/completions", r.URL.Path)
		}
		writeSSEEvent(w, `{"choices":[{"delta":{"content":"Ready"},"finish_reason":null}]}`)
		writeSSEEvent(w, "[DONE]")
	}))
	defer ts.Close()

	model := newHTTPBackedModel(ts)
	var got []string
	for token := range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "status"}}) {
		got = append(got, token.Text)
	}
	if err := model.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil", err)
	}
	if !reflect.DeepEqual(got, []string{"Ready"}) {
		t.Fatalf("tokens = %v, want Ready", got)
	}
}

func TestModel_Model_Chat_Bad(t *testing.T) {
	processExited := make(chan struct{})
	close(processExited)
	model := &rocmModel{server: &server{processExited: processExited}}

	var count int
	for range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hello"}}) {
		count++
	}
	if count != 0 {
		t.Fatalf("token count = %d, want 0", count)
	}
	if err := model.Err(); err == nil {
		t.Fatal("Err() = nil, want server exit error")
	}
}

func TestModel_Model_Chat_Ugly(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "chat failed", http.StatusInternalServerError)
	}))
	defer ts.Close()

	model := newHTTPBackedModel(ts)
	var got []string
	for token := range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hello"}}) {
		got = append(got, token.Text)
	}
	if len(got) != 0 {
		t.Fatalf("tokens = %v, want empty", got)
	}
	if err := model.Err(); err == nil || !strings.Contains(err.Error(), "chat returned 500") {
		t.Fatalf("Err() = %v, want chat returned 500", err)
	}
}

func TestModel_Model_Classify_Good(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			t.Fatalf("r.URL.Path = %q, want /v1/completions", r.URL.Path)
		}
		writeSSEEvent(w, `{"choices":[{"text":"positive","finish_reason":null}]}`)
		writeSSEEvent(w, "[DONE]")
	}))
	defer ts.Close()

	model := newHTTPBackedModel(ts)
	results, err := model.Classify(context.Background(), []string{"good"})
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if len(results) != 1 || results[0].Token.Text != "positive" {
		t.Fatalf("results = %+v, want positive label", results)
	}
}

func TestModel_Model_Classify_Bad(t *testing.T) {
	processExited := make(chan struct{})
	close(processExited)
	model := &rocmModel{server: &server{processExited: processExited}}

	results, err := model.Classify(context.Background(), []string{"bad"})
	if err == nil {
		t.Fatal("Classify error = nil, want server exit error")
	}
	if results != nil {
		t.Fatalf("results = %+v, want nil", results)
	}
}

func TestModel_Model_Classify_Ugly(t *testing.T) {
	client := llamacpp.NewClientWithHTTPClient("http://llama.test", &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader("data: {\"choices\":[{\"text\":\"partial\"}]}\n\n")),
				Request:    r,
			}, nil
		}),
	})
	model := newClientBackedModel(client)

	results, err := model.Classify(context.Background(), []string{"edge"})
	if err == nil {
		t.Fatal("Classify error = nil, want truncated stream error")
	}
	if results != nil {
		t.Fatalf("results = %+v, want nil", results)
	}
}

func TestModel_Model_BatchGenerate_Good(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeSSEEvent(w, `{"choices":[{"text":"A","finish_reason":null}]}`)
		writeSSEEvent(w, "[DONE]")
	}))
	defer ts.Close()

	model := newHTTPBackedModel(ts)
	results, err := model.BatchGenerate(context.Background(), []string{"one", "two"}, inference.WithMaxTokens(1))
	if err != nil {
		t.Fatalf("BatchGenerate: %v", err)
	}
	if len(results) != 2 || len(results[0].Tokens) != 1 || len(results[1].Tokens) != 1 {
		t.Fatalf("results = %+v, want one token per prompt", results)
	}
}

func TestModel_Model_BatchGenerate_Bad(t *testing.T) {
	processExited := make(chan struct{})
	close(processExited)
	model := &rocmModel{server: &server{processExited: processExited}}

	results, err := model.BatchGenerate(context.Background(), []string{"bad"})
	if err == nil {
		t.Fatal("BatchGenerate error = nil, want server exit error")
	}
	if results != nil {
		t.Fatalf("results = %+v, want nil", results)
	}
}

func TestModel_Model_BatchGenerate_Ugly(t *testing.T) {
	model := newClientBackedModel(llamacpp.NewClient("http://127.0.0.1:1"))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	results, err := model.BatchGenerate(ctx, []string{"cancelled"})
	if err != nil {
		t.Fatalf("BatchGenerate: %v", err)
	}
	if len(results) != 1 || results[0].Err == nil {
		t.Fatalf("results = %+v, want per-prompt cancellation error", results)
	}
}

func TestModel_Model_ModelType_Good(t *testing.T) {
	model := &rocmModel{modelType: "gemma3"}
	got := model.ModelType()
	if got != "gemma3" {
		t.Fatalf("ModelType() = %q, want gemma3", got)
	}
}

func TestModel_Model_ModelType_Bad(t *testing.T) {
	model := &rocmModel{}
	got := model.ModelType()
	if got != "" {
		t.Fatalf("ModelType() = %q, want empty string", got)
	}
}

func TestModel_Model_ModelType_Ugly(t *testing.T) {
	model := &rocmModel{modelType: "vendor.experimental"}
	first := model.ModelType()
	second := model.ModelType()
	if first != second {
		t.Fatalf("ModelType() changed from %q to %q", first, second)
	}
}

func TestModel_Model_Info_Good(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "llama", NumLayers: 32, QuantBits: 4}}
	info := model.Info()
	if info.Architecture != "llama" {
		t.Fatalf("Info().Architecture = %q, want llama", info.Architecture)
	}
	if info.NumLayers != 32 || info.QuantBits != 4 {
		t.Fatalf("Info() = %+v, want layer and quant metadata", info)
	}
}

func TestModel_Model_Info_Bad(t *testing.T) {
	model := &rocmModel{}
	info := model.Info()
	if info != (inference.ModelInfo{}) {
		t.Fatalf("Info() = %+v, want zero value", info)
	}
}

func TestModel_Model_Info_Ugly(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen2", NumLayers: 28}}
	info := model.Info()
	info.Architecture = "mutated"
	if model.Info().Architecture != "qwen2" {
		t.Fatalf("Info() returned mutable internal state: %+v", model.Info())
	}
}

func TestModel_Model_Metrics_Good(t *testing.T) {
	model := &rocmModel{}
	model.recordMetricsDurations(2, 3, 10*time.Millisecond, 20*time.Millisecond)
	metrics := model.Metrics()
	if metrics.PromptTokens != 2 || metrics.GeneratedTokens != 3 {
		t.Fatalf("Metrics() = %+v, want prompt and generated token counts", metrics)
	}
	if metrics.TotalDuration != 30*time.Millisecond {
		t.Fatalf("TotalDuration = %s, want 30ms", metrics.TotalDuration)
	}
}

func TestModel_Model_Metrics_Bad(t *testing.T) {
	model := &rocmModel{}
	metrics := model.Metrics()
	if metrics != (inference.GenerateMetrics{}) {
		t.Fatalf("Metrics() = %+v, want zero value", metrics)
	}
}

func TestModel_Model_Metrics_Ugly(t *testing.T) {
	model := &rocmModel{}
	model.recordMetricsDurations(1, 1, -time.Second, -time.Second)
	metrics := model.Metrics()
	if metrics.TotalDuration != 0 {
		t.Fatalf("TotalDuration = %s, want clamped zero", metrics.TotalDuration)
	}
}

func TestModel_Model_Err_Good(t *testing.T) {
	model := &rocmModel{}
	sentinel := errors.New("decode failed")
	model.setLastError(sentinel)
	if !errors.Is(model.Err(), sentinel) {
		t.Fatalf("Err() = %v, want sentinel", model.Err())
	}
}

func TestModel_Model_Err_Bad(t *testing.T) {
	model := &rocmModel{}
	model.setLastError(errors.New("temporary"))
	model.clearLastError()
	if model.Err() != nil {
		t.Fatalf("Err() = %v, want nil", model.Err())
	}
}

func TestModel_Model_Err_Ugly(t *testing.T) {
	model := &rocmModel{server: &server{processOutput: newProcessOutputCapture(serverProcessOutputLimit)}}
	model.setServerExitErr()
	if model.Err() == nil {
		t.Fatal("Err() = nil, want server exit error")
	}
}

func TestModel_Model_Close_Good(t *testing.T) {
	model := &rocmModel{server: &server{processCommand: &exec.Cmd{}}}
	err := model.Close()
	if err != nil {
		t.Fatalf("Close() = %v, want nil", err)
	}
}

func TestModel_Model_Close_Bad(t *testing.T) {
	model := &rocmModel{}
	defer func() {
		if recover() == nil {
			t.Fatal("Close() panic = nil, want panic for missing server")
		}
	}()
	_ = model.Close()
}

func TestModel_Model_Close_Ugly(t *testing.T) {
	model := &rocmModel{server: &server{processCommand: &exec.Cmd{}}}
	first := model.Close()
	second := model.Close()
	if first != nil || second != nil {
		t.Fatalf("Close() errors = %v, %v; want nil", first, second)
	}
}
