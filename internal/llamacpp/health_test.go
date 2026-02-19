package llamacpp

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHealth_OK(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/health", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"ok"}`))
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	err := c.Health(context.Background())
	require.NoError(t, err)
}

func TestHealth_NotReady(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"loading model"}`))
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	err := c.Health(context.Background())
	assert.ErrorContains(t, err, "not ready")
}

func TestHealth_Loading(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte(`{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}`))
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	err := c.Health(context.Background())
	assert.ErrorContains(t, err, "503")
}

func TestHealth_ServerDown(t *testing.T) {
	c := NewClient("http://127.0.0.1:1") // nothing listening
	err := c.Health(context.Background())
	assert.Error(t, err)
}
