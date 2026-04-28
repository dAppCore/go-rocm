package llamacpp

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestClient_NewClient_Good(t *testing.T) {
	client := NewClient("http://127.0.0.1:38080")
	if client.baseURL != "http://127.0.0.1:38080" {
		t.Fatalf("baseURL = %q, want trimmed base URL", client.baseURL)
	}
	if client.httpClient == nil {
		t.Fatal("httpClient = nil, want default client")
	}
}

func TestClient_NewClient_Bad(t *testing.T) {
	client := NewClient("")
	if client.baseURL != "" {
		t.Fatalf("baseURL = %q, want empty base URL preserved", client.baseURL)
	}
	if client.httpClient == nil {
		t.Fatal("httpClient = nil, want default client")
	}
}

func TestClient_NewClient_Ugly(t *testing.T) {
	client := NewClient("http://127.0.0.1:38080///")
	if client.baseURL != "http://127.0.0.1:38080" {
		t.Fatalf("baseURL = %q, want all trailing slashes trimmed", client.baseURL)
	}
	if client.httpClient == nil {
		t.Fatal("httpClient = nil, want default client")
	}
}

func TestClient_NewClientWithHTTPClient_Good(t *testing.T) {
	httpClient := &http.Client{}
	client := NewClientWithHTTPClient("http://llama.test/", httpClient)
	if client.httpClient != httpClient {
		t.Fatal("httpClient was not preserved")
	}
	if client.baseURL != "http://llama.test" {
		t.Fatalf("baseURL = %q, want trimmed base URL", client.baseURL)
	}
}

func TestClient_NewClientWithHTTPClient_Bad(t *testing.T) {
	client := NewClientWithHTTPClient("http://llama.test", nil)
	if client.httpClient == nil {
		t.Fatal("httpClient = nil, want default client")
	}
	if client.baseURL != "http://llama.test" {
		t.Fatalf("baseURL = %q, want original base URL", client.baseURL)
	}
}

func TestClient_NewClientWithHTTPClient_Ugly(t *testing.T) {
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return nil, fmt.Errorf("unused")
	})}
	client := NewClientWithHTTPClient("http://llama.test////", httpClient)
	if client.baseURL != "http://llama.test" {
		t.Fatalf("baseURL = %q, want deeply trimmed base URL", client.baseURL)
	}
	if client.httpClient.Transport == nil {
		t.Fatal("transport = nil, want injected transport")
	}
}

func TestClient_Client_Health_Good(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Fatalf("path = %q, want /health", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	}))
	defer ts.Close()

	client := NewClient(ts.URL)
	if err := client.Health(context.Background()); err != nil {
		t.Fatalf("Health() = %v, want nil", err)
	}
}

func TestClient_Client_Health_Bad(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "loading", http.StatusServiceUnavailable)
	}))
	defer ts.Close()

	client := NewClient(ts.URL)
	err := client.Health(context.Background())
	if err == nil {
		t.Fatal("Health() error = nil, want non-200 error")
	}
	if !strings.Contains(err.Error(), "503") {
		t.Fatalf("Health() = %v, want 503", err)
	}
}

func TestClient_Client_Health_Ugly(t *testing.T) {
	client := NewClient("http://%zz")
	err := client.Health(context.Background())
	if err == nil {
		t.Fatal("Health() error = nil, want request creation error")
	}
	if !strings.Contains(err.Error(), "create health request") {
		t.Fatalf("Health() = %v, want create health request", err)
	}
}

func TestClient_Client_Complete_Good(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			t.Fatalf("path = %q, want /v1/completions", r.URL.Path)
		}
		sseLines(w, []string{
			`{"choices":[{"text":"A","finish_reason":null}]}`,
			`{"choices":[{"text":"B","finish_reason":null}]}`,
			"[DONE]",
		})
	}))
	defer ts.Close()

	client := NewClient(ts.URL)
	tokens, errFn := client.Complete(context.Background(), CompletionRequest{Prompt: "go"})
	got := collectStringSeq(tokens)
	if err := errFn(); err != nil {
		t.Fatalf("errFn() = %v, want nil", err)
	}
	if !reflect.DeepEqual(got, []string{"A", "B"}) {
		t.Fatalf("tokens = %v, want A/B", got)
	}
}

func TestClient_Client_Complete_Bad(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer ts.Close()

	client := NewClient(ts.URL)
	tokens, errFn := client.Complete(context.Background(), CompletionRequest{Prompt: "go"})
	got := collectStringSeq(tokens)
	if len(got) != 0 {
		t.Fatalf("tokens = %v, want empty", got)
	}
	if err := errFn(); err == nil || !strings.Contains(err.Error(), "400") {
		t.Fatalf("errFn() = %v, want 400", err)
	}
}

func TestClient_Client_Complete_Ugly(t *testing.T) {
	client := NewClientWithHTTPClient("http://llama.test", &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader("data: {\"choices\":[{\"text\":\"partial\"}]}\n\n")),
				Request:    r,
			}, nil
		}),
	})

	tokens, errFn := client.Complete(context.Background(), CompletionRequest{Prompt: "go"})
	got := collectStringSeq(tokens)
	if !reflect.DeepEqual(got, []string{"partial"}) {
		t.Fatalf("tokens = %v, want partial", got)
	}
	if err := errFn(); err == nil || !strings.Contains(err.Error(), "stream ended before [DONE]") {
		t.Fatalf("errFn() = %v, want truncated stream", err)
	}
}

func TestClient_Client_ChatComplete_Good(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %q, want /v1/chat/completions", r.URL.Path)
		}
		sseLines(w, []string{
			`{"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}`,
			`{"choices":[{"delta":{"content":" there"},"finish_reason":null}]}`,
			"[DONE]",
		})
	}))
	defer ts.Close()

	client := NewClient(ts.URL)
	tokens, errFn := client.ChatComplete(context.Background(), ChatRequest{Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	got := collectStringSeq(tokens)
	if err := errFn(); err != nil {
		t.Fatalf("errFn() = %v, want nil", err)
	}
	if !reflect.DeepEqual(got, []string{"hi", " there"}) {
		t.Fatalf("tokens = %v, want chat chunks", got)
	}
}

func TestClient_Client_ChatComplete_Bad(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "chat failed", http.StatusInternalServerError)
	}))
	defer ts.Close()

	client := NewClient(ts.URL)
	tokens, errFn := client.ChatComplete(context.Background(), ChatRequest{Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	got := collectStringSeq(tokens)
	if len(got) != 0 {
		t.Fatalf("tokens = %v, want empty", got)
	}
	if err := errFn(); err == nil || !strings.Contains(err.Error(), "500") {
		t.Fatalf("errFn() = %v, want 500", err)
	}
}

func TestClient_Client_ChatComplete_Ugly(t *testing.T) {
	client := NewClientWithHTTPClient("http://llama.test", &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n")),
				Request:    r,
			}, nil
		}),
	})

	tokens, errFn := client.ChatComplete(context.Background(), ChatRequest{Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	got := collectStringSeq(tokens)
	if !reflect.DeepEqual(got, []string{"partial"}) {
		t.Fatalf("tokens = %v, want partial", got)
	}
	if err := errFn(); err == nil || !strings.Contains(err.Error(), "stream ended before [DONE]") {
		t.Fatalf("errFn() = %v, want truncated stream", err)
	}
}

func collectStringSeq(seq func(func(string) bool)) []string {
	var got []string
	for token := range seq {
		got = append(got, token)
	}
	return got
}
