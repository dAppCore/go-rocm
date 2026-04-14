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

//	client := NewClient("http://127.0.0.1:38080")
//
// NewClient creates a client for the llama-server at the given base URL.
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{},
	}
}

type healthStatusResponse struct {
	Status string `json:"status"`
}

//	err := client.Health(ctx)
//	fmt.Println(err == nil)
//
// Health checks whether the llama-server is ready to accept requests.
func (c *Client) Health(ctx context.Context) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return coreerr.E("llamacpp.Health", "create health request", err)
	}
	response, err := c.httpClient.Do(request)
	if err != nil {
		return coreerr.E("llamacpp.Health", "health request", err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		responseBody, _ := io.ReadAll(io.LimitReader(response.Body, 256))
		return coreerr.E("llamacpp.Health", fmt.Sprintf("health returned %d: %s", response.StatusCode, string(responseBody)), nil)
	}
	var healthStatus healthStatusResponse
	if err := json.NewDecoder(response.Body).Decode(&healthStatus); err != nil {
		return coreerr.E("llamacpp.Health", "health decode", err)
	}
	if healthStatus.Status != "ok" {
		return coreerr.E("llamacpp.Health", fmt.Sprintf("server not ready (status: %s)", healthStatus.Status), nil)
	}
	return nil
}
