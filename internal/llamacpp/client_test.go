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

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return fn(r)
}

// sseLines writes SSE-formatted lines to a flushing response writer.
func sseLines(w http.ResponseWriter, lines []string) {
	f, ok := w.(http.Flusher)
	if !ok {
		panic("ResponseWriter does not implement Flusher")
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)

	for _, line := range lines {
		fmt.Fprintf(w, "data: %s\n\n", line)
		f.Flush()
	}
}

func TestChatComplete_Streaming(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("r.URL.Path = %q, want %q", r.URL.Path, "/v1/chat/completions")
		}
		if r.Method != "POST" {
			t.Errorf("r.Method = %q, want %q", r.Method, "POST")
		}
		sseLines(w, []string{
			`{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}`,
			`{"choices":[{"delta":{"content":" world"},"finish_reason":null}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			"[DONE]",
		})
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	tokens, errFn := c.ChatComplete(context.Background(), ChatRequest{
		Messages:    []ChatMessage{{Role: "user", Content: "Hi"}},
		MaxTokens:   64,
		Temperature: 0.0,
		Stream:      true,
	})

	var got []string
	for tok := range tokens {
		got = append(got, tok)
	}
	if err := errFn(); err != nil {
		t.Fatalf("errFn: %v", err)
	}
	want := []string{"Hello", " world"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("tokens = %v, want %v", got, want)
	}
}

func TestChatComplete_EmptyResponse(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sseLines(w, []string{"[DONE]"})
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	tokens, errFn := c.ChatComplete(context.Background(), ChatRequest{
		Messages:    []ChatMessage{{Role: "user", Content: "Hi"}},
		Temperature: 0.7,
		Stream:      true,
	})

	var got []string
	for tok := range tokens {
		got = append(got, tok)
	}
	if err := errFn(); err != nil {
		t.Fatalf("errFn: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got = %v, want empty", got)
	}
}

func TestChatComplete_HTTPError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal server error", http.StatusInternalServerError)
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	tokens, errFn := c.ChatComplete(context.Background(), ChatRequest{
		Messages:    []ChatMessage{{Role: "user", Content: "Hi"}},
		Temperature: 0.7,
		Stream:      true,
	})

	var got []string
	for tok := range tokens {
		got = append(got, tok)
	}
	if len(got) != 0 {
		t.Errorf("got = %v, want empty", got)
	}
	err := errFn()
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("err = %v, want contains %q", err, "500")
	}
}

func TestChatComplete_ContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f, ok := w.(http.Flusher)
		if !ok {
			panic("ResponseWriter does not implement Flusher")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)

		// Send first chunk.
		fmt.Fprintf(w, "data: %s\n\n", `{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}`)
		f.Flush()

		// Wait for context cancellation before sending more.
		<-r.Context().Done()
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	tokens, errFn := c.ChatComplete(ctx, ChatRequest{
		Messages:    []ChatMessage{{Role: "user", Content: "Hi"}},
		Temperature: 0.7,
		Stream:      true,
	})

	var got []string
	for tok := range tokens {
		got = append(got, tok)
		cancel() // Cancel after receiving the first token.
	}
	// The error may or may not be nil depending on timing;
	// the important thing is we got exactly 1 token.
	_ = errFn()
	want := []string{"Hello"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("tokens = %v, want %v", got, want)
	}
}

func TestComplete_Streaming(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			t.Errorf("r.URL.Path = %q, want %q", r.URL.Path, "/v1/completions")
		}
		if r.Method != "POST" {
			t.Errorf("r.Method = %q, want %q", r.Method, "POST")
		}
		sseLines(w, []string{
			`{"choices":[{"text":"Once","finish_reason":null}]}`,
			`{"choices":[{"text":" upon","finish_reason":null}]}`,
			`{"choices":[{"text":" a time","finish_reason":null}]}`,
			`{"choices":[{"text":"","finish_reason":"stop"}]}`,
			"[DONE]",
		})
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	tokens, errFn := c.Complete(context.Background(), CompletionRequest{
		Prompt:      "Once",
		MaxTokens:   64,
		Temperature: 0.0,
		Stream:      true,
	})

	var got []string
	for tok := range tokens {
		got = append(got, tok)
	}
	if err := errFn(); err != nil {
		t.Fatalf("errFn: %v", err)
	}
	want := []string{"Once", " upon", " a time"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("tokens = %v, want %v", got, want)
	}
}

func TestComplete_HTTPError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	tokens, errFn := c.Complete(context.Background(), CompletionRequest{
		Prompt:      "Hello",
		Temperature: 0.7,
		Stream:      true,
	})

	var got []string
	for tok := range tokens {
		got = append(got, tok)
	}
	if len(got) != 0 {
		t.Errorf("got = %v, want empty", got)
	}
	err := errFn()
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "400") {
		t.Errorf("err = %v, want contains %q", err, "400")
	}
}

func TestComplete_TruncatedStreamReturnsError(t *testing.T) {
	c := NewClientWithHTTPClient("http://llama.test", &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			if r.URL.Path != "/v1/completions" {
				return nil, fmt.Errorf("unexpected path %q", r.URL.Path)
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
	tokens, errFn := c.Complete(context.Background(), CompletionRequest{
		Prompt:      "Hello",
		Temperature: 0.0,
		Stream:      true,
	})

	var got []string
	for tok := range tokens {
		got = append(got, tok)
	}

	want := []string{"partial"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("tokens = %v, want %v", got, want)
	}
	err := errFn()
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "stream ended before [DONE]") {
		t.Errorf("err = %v, want contains %q", err, "stream ended before [DONE]")
	}
}
