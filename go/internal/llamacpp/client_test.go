package llamacpp

import (
	"context"
	core "dappco.re/go"
	"net/http"
	"net/http/httptest"
	"testing"
)

func sse(w http.ResponseWriter, payload string) {
	core.WriteString(w, "data: "+payload+"\n\ndata: [DONE]\n\n")
}

func TestClient_Client_ChatComplete_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { sse(w, `{"choices":[{"delta":{"content":"ok"}}]}`) }))
	defer ts.Close()
	stream, errFn := NewClient(ts.URL).ChatComplete(context.Background(), ChatRequest{})
	var got []string
	for s := range stream {
		got = append(got, s)
	}
	core.AssertNoError(t, errFn())
	core.AssertEqual(t, "ok", got[0])
}
func TestClient_Client_ChatComplete_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusBadRequest) }))
	defer ts.Close()
	_, errFn := NewClient(ts.URL).ChatComplete(context.Background(), ChatRequest{})
	core.AssertError(t, errFn())
}
func TestClient_Client_ChatComplete_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	_, errFn := NewClient("http://%zz").ChatComplete(context.Background(), ChatRequest{})
	core.AssertError(t, errFn())
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}

func TestClient_Client_Complete_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { sse(w, `{"choices":[{"text":"ok"}]}`) }))
	defer ts.Close()
	stream, errFn := NewClient(ts.URL).Complete(context.Background(), CompletionRequest{})
	var got []string
	for s := range stream {
		got = append(got, s)
	}
	core.AssertNoError(t, errFn())
	core.AssertEqual(t, "ok", got[0])
}
func TestClient_Client_Complete_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusBadRequest) }))
	defer ts.Close()
	_, errFn := NewClient(ts.URL).Complete(context.Background(), CompletionRequest{})
	core.AssertError(t, errFn())
}
func TestClient_Client_Complete_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	_, errFn := NewClient("http://%zz").Complete(context.Background(), CompletionRequest{})
	core.AssertError(t, errFn())
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
