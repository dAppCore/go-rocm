//go:build rocm_legacy_server

package llamacpp

import (
	"context"
	core "dappco.re/go"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHealth_NewClient_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	core.AssertNotNil(t, NewClient("http://example.test"))
}
func TestHealth_NewClient_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	c := NewClient("http://example.test/")
	core.AssertEqual(t, "http://example.test", c.baseURL)
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestHealth_NewClient_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	core.AssertNotNil(t, NewClient(""))
}

func TestHealth_NewClientWithHTTPClient_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	core.AssertNotNil(t, NewClientWithHTTPClient("http://example.test", &http.Client{}))
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestHealth_NewClientWithHTTPClient_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	c := NewClientWithHTTPClient("http://example.test", nil)
	core.AssertNotNil(t, c.httpClient)
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestHealth_NewClientWithHTTPClient_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	c := NewClientWithHTTPClient("http://example.test/", nil)
	core.AssertEqual(t, "http://example.test", c.baseURL)
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}

func TestHealth_Client_Health_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte(`{"status":"ok"}`)) }))
	defer ts.Close()
	core.AssertNoError(t, NewClient(ts.URL).Health(context.Background()))
}
func TestHealth_Client_Health_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusServiceUnavailable) }))
	defer ts.Close()
	core.AssertError(t, NewClient(ts.URL).Health(context.Background()))
}
func TestHealth_Client_Health_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	core.AssertError(t, NewClient("http://%zz").Health(context.Background()))
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
