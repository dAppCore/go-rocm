package llamacpp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	coreerr "dappco.re/go/core/log"
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
		return coreerr.E("llamacpp.Health", "create health request", err)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return coreerr.E("llamacpp.Health", "health request", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		return coreerr.E("llamacpp.Health", fmt.Sprintf("health returned %d: %s", resp.StatusCode, string(body)), nil)
	}
	var h healthResponse
	if err := json.NewDecoder(resp.Body).Decode(&h); err != nil {
		return coreerr.E("llamacpp.Health", "health decode", err)
	}
	if h.Status != "ok" {
		return coreerr.E("llamacpp.Health", fmt.Sprintf("server not ready (status: %s)", h.Status), nil)
	}
	return nil
}
