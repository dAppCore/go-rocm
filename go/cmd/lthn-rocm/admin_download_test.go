// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

type serveDownloadFakeTree struct {
	mu      sync.Mutex
	entries []rocmHFFileEntry
	err     error
	calls   int
}

type serveDownloadFixtureRoundTripper struct {
	status int
	body   string
	paths  []string
}

func (rt *serveDownloadFixtureRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.paths = append(rt.paths, req.URL.Path)
	return &http.Response{
		StatusCode: rt.status,
		Body:       io.NopCloser(strings.NewReader(rt.body)),
		Header:     http.Header{},
	}, nil
}

func (tree *serveDownloadFakeTree) ResolveTree(context.Context, string, string) ([]rocmHFFileEntry, error) {
	tree.mu.Lock()
	defer tree.mu.Unlock()
	tree.calls++
	if tree.err != nil {
		return nil, tree.err
	}
	return append([]rocmHFFileEntry(nil), tree.entries...), nil
}

func (tree *serveDownloadFakeTree) Calls() int {
	tree.mu.Lock()
	defer tree.mu.Unlock()
	return tree.calls
}

const rocmHFTreeFixture = `[
  {"type":"file","oid":"52373fe2","size":1570,"path":".gitattributes"},
  {"type":"file","path":"model.safetensors","size":4,"lfs":{"sha256":"ABC123","size":806000000}},
  {"type":"file","path":"config.json","size":910},
  {"type":"directory","path":"assets"},
  {"type":"file","path":"../escape.bin","size":9}
]`

func TestROCmHFTreeClientResolveTreeRealDecodePath(t *testing.T) {
	rt := &serveDownloadFixtureRoundTripper{status: http.StatusOK, body: rocmHFTreeFixture}
	client := &rocmHFTreeClient{httpClient: &http.Client{Transport: rt}}

	entries, err := client.ResolveTree(context.Background(), "ok/repo", "")
	if err != nil {
		t.Fatalf("ResolveTree() error=%v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("entries=%d, want 2: %+v", len(entries), entries)
	}
	if entries[0].Path != "model.safetensors" ||
		entries[0].Digest != "abc123" ||
		entries[0].Size != 4 &&
			entries[0].Size != 806000000 {
		t.Fatalf("entry[0]=%+v, want LFS safetensors metadata", entries[0])
	}
	if entries[1].Path != "config.json" || entries[1].Size != 910 {
		t.Fatalf("entry[1]=%+v, want config.json/910", entries[1])
	}
	for _, entry := range entries {
		if strings.Contains(entry.Path, "..") || strings.HasPrefix(entry.Path, ".") {
			t.Fatalf("unsafe path survived: %+v", entry)
		}
	}
	if len(rt.paths) != 1 || !strings.Contains(rt.paths[0], "/tree/main") {
		t.Fatalf("request paths=%v, want empty revision defaulted to main", rt.paths)
	}
}

func TestROCmHFTreeClientResolveTreeAuthAndBadJSON(t *testing.T) {
	for _, status := range []int{http.StatusUnauthorized, http.StatusForbidden} {
		client := &rocmHFTreeClient{httpClient: &http.Client{Transport: &serveDownloadFixtureRoundTripper{status: status, body: "denied"}}}
		_, err := client.ResolveTree(context.Background(), "ok/repo", "main")
		if err == nil || !strings.Contains(err.Error(), "private repo or token") {
			t.Fatalf("status %d err=%v, want private repo/token hint", status, err)
		}
	}

	client := &rocmHFTreeClient{httpClient: &http.Client{Transport: &serveDownloadFixtureRoundTripper{status: http.StatusOK, body: ""}}}
	if _, err := client.ResolveTree(context.Background(), "ok/repo", "main"); err == nil {
		t.Fatal("empty tree response decoded, want JSON error")
	}
}

func TestServeAdminDownloadRequiresAuthAndAllowlist(t *testing.T) {
	dir := t.TempDir()
	allowedPath := filepath.Join(dir, "allowed-models.json")
	if err := os.WriteFile(allowedPath, []byte(`{"repos":["ok/repo"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	tree := &serveDownloadFakeTree{}
	handler, resolver := newROCmServeHandler(rocmServeConfig{
		Backend:                  "rocm",
		AdminToken:               "test-token",
		AdminDownloadAllowedPath: allowedPath,
		AdminDownloadModelDir:    filepath.Join(dir, "models"),
		AdminDownloadTreeAPI:     tree,
		AdminDownloadFetch:       serveDownloadNoopFetch,
	})
	defer resolver.Close()

	unauthorized := httptest.NewRecorder()
	handler.ServeHTTP(unauthorized, httptest.NewRequest(http.MethodPost, rocmServeAdminDownloadPath, strings.NewReader(`{"repo":"ok/repo"}`)))
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("download without token status=%d body=%s, want 401", unauthorized.Code, unauthorized.Body.String())
	}

	req := httptest.NewRequest(http.MethodPost, rocmServeAdminDownloadPath, strings.NewReader(`{"repo":"blocked/repo"}`))
	req.Header.Set("Authorization", "Bearer test-token")
	forbidden := httptest.NewRecorder()
	handler.ServeHTTP(forbidden, req)
	if forbidden.Code != http.StatusForbidden || !strings.Contains(forbidden.Body.String(), "repo not in allowlist") {
		t.Fatalf("blocked repo status=%d body=%s, want allowlist 403", forbidden.Code, forbidden.Body.String())
	}
	if tree.Calls() != 0 {
		t.Fatalf("tree API calls=%d, want 0 for allowlist rejection", tree.Calls())
	}
}

func TestServeAdminDownloadStartsAndPollsJob(t *testing.T) {
	dir := t.TempDir()
	allowedPath := filepath.Join(dir, "allowed-models.json")
	if err := os.WriteFile(allowedPath, []byte(`{"repos":["ok/repo"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	payload := []byte(`{"model":"ok"}`)
	sum := sha256.Sum256(payload)
	digest := hex.EncodeToString(sum[:])
	modelDir := filepath.Join(dir, "models")
	tree := &serveDownloadFakeTree{entries: []rocmHFFileEntry{{
		Path:   "config.json",
		URL:    rocmHFHostResolve + "ok/repo/resolve/main/config.json",
		Size:   int64(len(payload)),
		Digest: digest,
	}}}
	handler, resolver := newROCmServeHandler(rocmServeConfig{
		Backend:                  "rocm",
		AdminToken:               "test-token",
		AdminDownloadAllowedPath: allowedPath,
		AdminDownloadModelDir:    modelDir,
		AdminDownloadTreeAPI:     tree,
		AdminDownloadFetch: serveDownloadPayloadFetch(map[string][]byte{
			rocmHFHostResolve + "ok/repo/resolve/main/config.json": payload,
		}),
	})
	defer resolver.Close()

	req := httptest.NewRequest(http.MethodPost, rocmServeAdminDownloadPath, strings.NewReader(`{"repo":"ok/repo","revision":"main"}`))
	req.Header.Set("Authorization", "Bearer test-token")
	start := httptest.NewRecorder()
	handler.ServeHTTP(start, req)
	if start.Code != http.StatusAccepted {
		t.Fatalf("download start status=%d body=%s, want 202", start.Code, start.Body.String())
	}
	var started rocmAdminDownloadJob
	if err := json.Unmarshal(start.Body.Bytes(), &started); err != nil {
		t.Fatal(err)
	}
	if started.ID == "" || started.Status != "pending" || started.DestPath == "" {
		t.Fatalf("unexpected start job: %+v", started)
	}

	done := waitServeDownloadJob(t, handler, "test-token", started.ID)
	if done.Status != "done" || done.BytesDone != int64(len(payload)) || done.FileCount != 1 {
		t.Fatalf("final job=%+v, want done with downloaded bytes", done)
	}
	finalFile := filepath.Join(modelDir, "ok__repo", "main", "config.json")
	if data, err := os.ReadFile(finalFile); err != nil || string(data) != string(payload) {
		t.Fatalf("downloaded file data=%q err=%v, want payload", string(data), err)
	}
	manifest, err := os.ReadFile(filepath.Join(modelDir, "ok__repo", "main", ".sha256"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(manifest), digest+"  config.json") {
		t.Fatalf("manifest=%q, want config digest", string(manifest))
	}
}

func TestServeAdminDownloadRejectsBadRevisionAndMissingJob(t *testing.T) {
	dir := t.TempDir()
	allowedPath := filepath.Join(dir, "allowed-models.json")
	if err := os.WriteFile(allowedPath, []byte(`{"repos":["ok/repo"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	handler, resolver := newROCmServeHandler(rocmServeConfig{
		Backend:                  "rocm",
		AdminToken:               "test-token",
		AdminDownloadAllowedPath: allowedPath,
		AdminDownloadModelDir:    filepath.Join(dir, "models"),
		AdminDownloadTreeAPI:     &serveDownloadFakeTree{},
		AdminDownloadFetch:       serveDownloadNoopFetch,
	})
	defer resolver.Close()

	badRevision := httptest.NewRequest(http.MethodPost, rocmServeAdminDownloadPath, strings.NewReader(`{"repo":"ok/repo","revision":"branch/name"}`))
	badRevision.Header.Set("Authorization", "Bearer test-token")
	bad := httptest.NewRecorder()
	handler.ServeHTTP(bad, badRevision)
	if bad.Code != http.StatusBadRequest || !strings.Contains(bad.Body.String(), "revision contains disallowed character") {
		t.Fatalf("bad revision status=%d body=%s, want 400", bad.Code, bad.Body.String())
	}

	missingJob := httptest.NewRequest(http.MethodGet, rocmServeAdminDownloadPath, nil)
	missingJob.Header.Set("Authorization", "Bearer test-token")
	missing := httptest.NewRecorder()
	handler.ServeHTTP(missing, missingJob)
	if missing.Code != http.StatusBadRequest {
		t.Fatalf("missing job status=%d body=%s, want 400", missing.Code, missing.Body.String())
	}

	unknownJob := httptest.NewRequest(http.MethodGet, rocmServeAdminDownloadPath+"?job=download-missing", nil)
	unknownJob.Header.Set("Authorization", "Bearer test-token")
	unknown := httptest.NewRecorder()
	handler.ServeHTTP(unknown, unknownJob)
	if unknown.Code != http.StatusNotFound {
		t.Fatalf("unknown job status=%d body=%s, want 404", unknown.Code, unknown.Body.String())
	}
}

func TestAdminDownloadValidationHelpers(t *testing.T) {
	for _, revision := range []string{"../etc", "branch/name", "bad revision"} {
		if err := validateROCmRevision(revision); err == nil {
			t.Fatalf("validateROCmRevision(%q) succeeded, want error", revision)
		}
	}
	for _, path := range []string{"../config.json", ".gitattributes", "/config.json", "nested/../config.json", "bad\x00name"} {
		if isSafeROCmHFEntryPath(path) {
			t.Fatalf("isSafeROCmHFEntryPath(%q)=true, want false", path)
		}
	}
	if !isSafeROCmHFEntryPath("nested/config.json") {
		t.Fatal("nested/config.json rejected, want safe")
	}
	entries := []rocmHFFileEntry{{Path: "config.json"}, {Path: "weights.safetensors"}}
	filtered := filterROCmDownloadEntries(entries, []string{"weights.safetensors"})
	if len(filtered) != 1 || filtered[0].Path != "weights.safetensors" {
		t.Fatalf("filtered=%+v, want weights only", filtered)
	}
}

func waitServeDownloadJob(t *testing.T, handler http.Handler, token, id string) rocmAdminDownloadJob {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	var job rocmAdminDownloadJob
	for time.Now().Before(deadline) {
		req := httptest.NewRequest(http.MethodGet, rocmServeAdminDownloadPath+"?job="+id, nil)
		req.Header.Set("Authorization", "Bearer "+token)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("poll status=%d body=%s, want 200", rec.Code, rec.Body.String())
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &job); err != nil {
			t.Fatal(err)
		}
		if job.Status == "done" || job.Status == "failed" {
			return job
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for job %s; last=%+v", id, job)
	return job
}

func serveDownloadPayloadFetch(payloads map[string][]byte) rocmAdminFetchFunc {
	return func(_ context.Context, sourceURL, destPath, expectedDigest string, _ int64) (int64, string, error) {
		payload, ok := payloads[sourceURL]
		if !ok {
			return 0, "", fmt.Errorf("missing payload for %s", sourceURL)
		}
		if err := os.WriteFile(destPath, payload, 0o600); err != nil {
			return 0, "", err
		}
		sum := sha256.Sum256(payload)
		digest := hex.EncodeToString(sum[:])
		if expectedDigest != "" && digest != strings.ToLower(expectedDigest) {
			return 0, "", fmt.Errorf("sha256 mismatch: got=%s want=%s", digest, expectedDigest)
		}
		return int64(len(payload)), digest, nil
	}
}

func serveDownloadNoopFetch(context.Context, string, string, string, int64) (int64, string, error) {
	return 0, "", nil
}
