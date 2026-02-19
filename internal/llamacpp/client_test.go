package llamacpp

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		assert.Equal(t, "POST", r.Method)
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
	require.NoError(t, errFn())
	assert.Equal(t, []string{"Hello", " world"}, got)
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
	require.NoError(t, errFn())
	assert.Empty(t, got)
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
	assert.Empty(t, got)
	err := errFn()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "500")
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
	assert.Equal(t, []string{"Hello"}, got)
}

func TestComplete_Streaming(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/completions", r.URL.Path)
		assert.Equal(t, "POST", r.Method)
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
	require.NoError(t, errFn())
	assert.Equal(t, []string{"Once", " upon", " a time"}, got)
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
	assert.Empty(t, got)
	err := errFn()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "400")
}
