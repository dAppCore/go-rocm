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
	"sync"

	coreerr "dappco.re/go/core/log"
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

type chatStreamChunkResponse struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

type completionStreamChunkResponse struct {
	Choices []struct {
		Text         string  `json:"text"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

//	chunks, streamError := client.ChatComplete(ctx, ChatRequest{
//		Messages: []ChatMessage{{Role: "user", Content: "Hi"}},
//	})
//
// ChatComplete sends a streaming chat completion request to
// /v1/chat/completions. It returns an iterator over text chunks and a function
// that returns any error that occurred during the request or while reading the
// stream.
func (c *Client) ChatComplete(ctx context.Context, req ChatRequest) (iter.Seq[string], func() error) {
	req.Stream = true

	requestBody, err := json.Marshal(req)
	if err != nil {
		return noStreamChunks, func() error { return coreerr.E("llamacpp.ChatComplete", "marshal chat request", err) }
	}

	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(requestBody))
	if err != nil {
		return noStreamChunks, func() error { return coreerr.E("llamacpp.ChatComplete", "create chat request", err) }
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Accept", "text/event-stream")

	response, err := c.httpClient.Do(httpRequest)
	if err != nil {
		return noStreamChunks, func() error { return coreerr.E("llamacpp.ChatComplete", "chat request", err) }
	}

	if response.StatusCode != http.StatusOK {
		defer response.Body.Close()
		responseBody, _ := io.ReadAll(io.LimitReader(response.Body, 256))
		return noStreamChunks, func() error {
			return coreerr.E("llamacpp.ChatComplete", fmt.Sprintf("chat returned %d: %s", response.StatusCode, strings.TrimSpace(string(responseBody))), nil)
		}
	}

	var (
		streamErr error
		closeOnce sync.Once
		closeBody = func() { closeOnce.Do(func() { response.Body.Close() }) }
	)
	eventDataStream := streamSSEData(response.Body, &streamErr)

	tokenStream := func(yield func(string) bool) {
		defer closeBody()
		for rawChunk := range eventDataStream {
			var chunk chatStreamChunkResponse
			if err := json.Unmarshal([]byte(rawChunk), &chunk); err != nil {
				streamErr = coreerr.E("llamacpp.ChatComplete", "decode chat chunk", err)
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

	return tokenStream, func() error {
		closeBody()
		return streamErr
	}
}

//	chunks, streamError := client.Complete(ctx, CompletionRequest{
//		Prompt: "Hello",
//	})
//
// Complete sends a streaming completion request to /v1/completions. It
// returns an iterator over text chunks and a function that returns any error
// that occurred during the request or while reading the stream.
func (c *Client) Complete(ctx context.Context, req CompletionRequest) (iter.Seq[string], func() error) {
	req.Stream = true

	requestBody, err := json.Marshal(req)
	if err != nil {
		return noStreamChunks, func() error { return coreerr.E("llamacpp.Complete", "marshal completion request", err) }
	}

	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/completions", bytes.NewReader(requestBody))
	if err != nil {
		return noStreamChunks, func() error { return coreerr.E("llamacpp.Complete", "create completion request", err) }
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Accept", "text/event-stream")

	response, err := c.httpClient.Do(httpRequest)
	if err != nil {
		return noStreamChunks, func() error { return coreerr.E("llamacpp.Complete", "completion request", err) }
	}

	if response.StatusCode != http.StatusOK {
		defer response.Body.Close()
		responseBody, _ := io.ReadAll(io.LimitReader(response.Body, 256))
		return noStreamChunks, func() error {
			return coreerr.E("llamacpp.Complete", fmt.Sprintf("completion returned %d: %s", response.StatusCode, strings.TrimSpace(string(responseBody))), nil)
		}
	}

	var (
		streamErr error
		closeOnce sync.Once
		closeBody = func() { closeOnce.Do(func() { response.Body.Close() }) }
	)
	eventDataStream := streamSSEData(response.Body, &streamErr)

	tokenStream := func(yield func(string) bool) {
		defer closeBody()
		for rawChunk := range eventDataStream {
			var chunk completionStreamChunkResponse
			if err := json.Unmarshal([]byte(rawChunk), &chunk); err != nil {
				streamErr = coreerr.E("llamacpp.Complete", "decode completion chunk", err)
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

	return tokenStream, func() error {
		closeBody()
		return streamErr
	}
}

// streamSSEData reads SSE-formatted lines from r and yields the payload of
// each "data: " line. It stops when it encounters "[DONE]" or an I/O error.
// Any read error (other than EOF) is stored via errOut.
func streamSSEData(r io.Reader, errOut *error) iter.Seq[string] {
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
			*errOut = coreerr.E("llamacpp.streamSSEData", "read SSE stream", err)
		}
	}
}

// noStreamChunks is an empty iterator returned when an error occurs before
// streaming begins.
func noStreamChunks(func(string) bool) {}
