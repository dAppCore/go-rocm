package llamacpp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Client communicates with a llama-server instance.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a client for the llama-server at the given base URL.
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{},
	}
}

type healthResponse struct {
	Status string `json:"status"`
}

// Health checks whether the llama-server is ready to accept requests.
func (c *Client) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		return fmt.Errorf("llamacpp: health returned %d: %s", resp.StatusCode, string(body))
	}
	var h healthResponse
	if err := json.NewDecoder(resp.Body).Decode(&h); err != nil {
		return fmt.Errorf("llamacpp: health decode: %w", err)
	}
	if h.Status != "ok" {
		return fmt.Errorf("llamacpp: server not ready (status: %s)", h.Status)
	}
	return nil
}
