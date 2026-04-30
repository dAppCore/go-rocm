package llamacpp

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHealth_OK(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Errorf("r.URL.Path = %q, want %q", r.URL.Path, "/health")
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"ok"}`))
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	if err := c.Health(context.Background()); err != nil {
		t.Fatalf("Health: %v", err)
	}
}

func TestHealth_NotReady(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"loading model"}`))
	}))
	defer ts.Close()

	c := NewClient(ts.URL)
	err := c.Health(context.Background())
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "not ready") {
		t.Errorf("err = %v, want contains %q", err, "not ready")
	}
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
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "503") {
		t.Errorf("err = %v, want contains %q", err, "503")
	}
}

func TestHealth_ServerDown(t *testing.T) {
	c := NewClient("http://127.0.0.1:1") // nothing listening
	err := c.Health(context.Background())
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "health request") {
		t.Errorf("err = %v, want contains %q", err, "health request")
	}
}

func TestHealth_InvalidBaseURL(t *testing.T) {
	c := NewClient("http://%zz")
	err := c.Health(context.Background())
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "create health request") {
		t.Errorf("err = %v, want contains %q", err, "create health request")
	}
}
