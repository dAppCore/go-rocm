package llamacpp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
)

// ChatMessage is a single message in a conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest is the request body for /v1/chat/completions.
type ChatRequest struct {
	Messages      []ChatMessage `json:"messages"`
	MaxTokens     int           `json:"max_tokens,omitempty"`
	Temperature   float32       `json:"temperature"`
	TopK          int           `json:"top_k,omitempty"`
	TopP          float32       `json:"top_p,omitempty"`
	RepeatPenalty float32       `json:"repeat_penalty,omitempty"`
	Stream        bool          `json:"stream"`
}

// CompletionRequest is the request body for /v1/completions.
type CompletionRequest struct {
	Prompt        string  `json:"prompt"`
	MaxTokens     int     `json:"max_tokens,omitempty"`
	Temperature   float32 `json:"temperature"`
	TopK          int     `json:"top_k,omitempty"`
	TopP          float32 `json:"top_p,omitempty"`
	RepeatPenalty float32 `json:"repeat_penalty,omitempty"`
	Stream        bool    `json:"stream"`
}

type chatChunkResponse struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

type completionChunkResponse struct {
	Choices []struct {
		Text         string  `json:"text"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

// ChatComplete sends a streaming chat completion request to /v1/chat/completions.
// It returns an iterator over text chunks and a function that returns any error
// that occurred during the request or while reading the stream.
func (c *Client) ChatComplete(ctx context.Context, req ChatRequest) (iter.Seq[string], func() error) {
	req.Stream = true

	body, err := json.Marshal(req)
	if err != nil {
		return noChunks, func() error { return fmt.Errorf("llamacpp: marshal chat request: %w", err) }
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return noChunks, func() error { return fmt.Errorf("llamacpp: create chat request: %w", err) }
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return noChunks, func() error { return fmt.Errorf("llamacpp: chat request: %w", err) }
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		return noChunks, func() error {
			return fmt.Errorf("llamacpp: chat returned %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
		}
	}

	var streamErr error
	sseData := parseSSE(resp.Body, &streamErr)

	tokens := func(yield func(string) bool) {
		defer resp.Body.Close()
		for raw := range sseData {
			var chunk chatChunkResponse
			if err := json.Unmarshal([]byte(raw), &chunk); err != nil {
				streamErr = fmt.Errorf("llamacpp: decode chat chunk: %w", err)
				return
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			text := chunk.Choices[0].Delta.Content
			if text == "" {
				continue
			}
			if !yield(text) {
				return
			}
		}
	}

	return tokens, func() error { return streamErr }
}

// Complete sends a streaming completion request to /v1/completions.
// It returns an iterator over text chunks and a function that returns any error
// that occurred during the request or while reading the stream.
func (c *Client) Complete(ctx context.Context, req CompletionRequest) (iter.Seq[string], func() error) {
	req.Stream = true

	body, err := json.Marshal(req)
	if err != nil {
		return noChunks, func() error { return fmt.Errorf("llamacpp: marshal completion request: %w", err) }
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/completions", bytes.NewReader(body))
	if err != nil {
		return noChunks, func() error { return fmt.Errorf("llamacpp: create completion request: %w", err) }
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return noChunks, func() error { return fmt.Errorf("llamacpp: completion request: %w", err) }
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		return noChunks, func() error {
			return fmt.Errorf("llamacpp: completion returned %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
		}
	}

	var streamErr error
	sseData := parseSSE(resp.Body, &streamErr)

	tokens := func(yield func(string) bool) {
		defer resp.Body.Close()
		for raw := range sseData {
			var chunk completionChunkResponse
			if err := json.Unmarshal([]byte(raw), &chunk); err != nil {
				streamErr = fmt.Errorf("llamacpp: decode completion chunk: %w", err)
				return
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			text := chunk.Choices[0].Text
			if text == "" {
				continue
			}
			if !yield(text) {
				return
			}
		}
	}

	return tokens, func() error { return streamErr }
}

// parseSSE reads SSE-formatted lines from r and yields the payload of each
// "data: " line. It stops when it encounters "[DONE]" or an I/O error.
// Any read error (other than EOF) is stored via errOut.
func parseSSE(r io.Reader, errOut *error) iter.Seq[string] {
	return func(yield func(string) bool) {
		scanner := bufio.NewScanner(r)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			payload := strings.TrimPrefix(line, "data: ")
			if payload == "[DONE]" {
				return
			}
			if !yield(payload) {
				return
			}
		}
		if err := scanner.Err(); err != nil {
			*errOut = fmt.Errorf("llamacpp: read SSE stream: %w", err)
		}
	}
}

// noChunks is an empty iterator returned when an error occurs before streaming begins.
func noChunks(func(string) bool) {}
