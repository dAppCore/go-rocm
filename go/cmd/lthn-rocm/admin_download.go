// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"
)

const (
	rocmHFHostTreeAPI                 = "https://huggingface.co/api/models/"
	rocmHFHostResolve                 = "https://huggingface.co/"
	rocmHFTreeResponseCap       int64 = 4 << 20
	rocmHFFileCap               int64 = 256 << 30
	maxROCmDownloadJobsRetained       = 32
)

type rocmAdminDownloadRequest struct {
	Repo     string   `json:"repo"`
	Revision string   `json:"revision"`
	Files    []string `json:"files,omitempty"`
}

type rocmAdminDownloadJob struct {
	ID         string     `json:"id"`
	Status     string     `json:"status"`
	Repo       string     `json:"repo"`
	Revision   string     `json:"revision"`
	StartedAt  time.Time  `json:"started_at"`
	FinishedAt *time.Time `json:"finished_at,omitempty"`
	DestPath   string     `json:"dest_path,omitempty"`
	BytesTotal int64      `json:"bytes_total,omitempty"`
	BytesDone  int64      `json:"bytes_done,omitempty"`
	FileCount  int        `json:"file_count,omitempty"`
	Error      string     `json:"error,omitempty"`
}

type rocmAdminDownloadConfig struct {
	AllowedModelsPath string
	ModelDir          string
	TreeAPI           rocmHFTreeAPI
	Fetch             rocmAdminFetchFunc
}

type rocmAdminFetchFunc func(ctx context.Context, url, destPath, expectedDigest string, expectedSize int64) (int64, string, error)

type rocmAdminDownloadRegistry struct {
	mu          sync.Mutex
	jobs        map[string]*rocmAdminDownloadJob
	activeSlots chan struct{}
	ctx         context.Context
	cfg         rocmAdminDownloadConfig
}

func newROCmAdminDownloadRegistry(ctx context.Context, cfg rocmAdminDownloadConfig) *rocmAdminDownloadRegistry {
	if ctx == nil {
		ctx = context.Background()
	}
	if cfg.TreeAPI == nil {
		cfg.TreeAPI = &rocmHFTreeClient{httpClient: &http.Client{}}
	}
	if cfg.Fetch == nil {
		cfg.Fetch = rocmFetchAndVerify
	}
	return &rocmAdminDownloadRegistry{
		jobs:        map[string]*rocmAdminDownloadJob{},
		activeSlots: make(chan struct{}, 1),
		ctx:         ctx,
		cfg:         cfg,
	}
}

func rocmAdminDownloadHandler(registry *rocmAdminDownloadRegistry) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if registry == nil {
			writeROCmServeError(w, http.StatusServiceUnavailable, "admin download registry is not configured", "download")
			return
		}
		switch r.Method {
		case http.MethodGet:
			rocmAdminDownloadGet(registry, w, r)
		case http.MethodPost:
			rocmAdminDownloadPost(registry, w, r)
		default:
			w.Header().Set("Allow", http.MethodGet+", "+http.MethodPost)
			writeROCmServeError(w, http.StatusMethodNotAllowed, "method not allowed", "method")
		}
	}
}

func rocmAdminDownloadGet(registry *rocmAdminDownloadRegistry, w http.ResponseWriter, r *http.Request) {
	jobID := strings.TrimSpace(r.URL.Query().Get("job"))
	if jobID == "" {
		writeROCmServeError(w, http.StatusBadRequest, "missing job id; use GET ?job=<id>", "job")
		return
	}
	registry.mu.Lock()
	job, ok := registry.jobs[jobID]
	var snapshot rocmAdminDownloadJob
	if ok {
		snapshot = *job
	}
	registry.mu.Unlock()
	if !ok {
		writeROCmServeError(w, http.StatusNotFound, "job not found", "job")
		return
	}
	writeROCmServeJSON(w, http.StatusOK, snapshot)
}

func rocmAdminDownloadPost(registry *rocmAdminDownloadRegistry, w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	var req rocmAdminDownloadRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64*1024)).Decode(&req); err != nil {
		writeROCmServeError(w, http.StatusBadRequest, "invalid request body", "body")
		return
	}
	req.Repo = strings.TrimSpace(req.Repo)
	req.Revision = strings.TrimSpace(req.Revision)
	if req.Revision == "" {
		req.Revision = "main"
	}
	if err := validateROCmDownloadRequest(req); err != nil {
		writeROCmServeError(w, http.StatusBadRequest, err.Error(), "request")
		return
	}
	allowed, err := loadROCmAllowedModels(rocmAdminDownloadAllowedPath(registry.cfg))
	if err != nil {
		writeROCmServeError(w, http.StatusInternalServerError, "allowlist parse: "+err.Error(), "allowlist")
		return
	}
	if !isROCmRepoAllowed(allowed, req.Repo) {
		writeROCmServeError(w, http.StatusForbidden, "repo not in allowlist", "repo")
		return
	}
	if !registry.tryAcquire() {
		writeROCmServeError(w, http.StatusTooManyRequests, "download busy; another job is in flight", "download")
		return
	}

	jobID := rocmDownloadJobID()
	destRoot := filepath.Join(rocmAdminDownloadModelDir(registry.cfg), canonicaliseROCmRepoName(req.Repo), req.Revision)
	job := &rocmAdminDownloadJob{
		ID:        jobID,
		Status:    "pending",
		Repo:      req.Repo,
		Revision:  req.Revision,
		StartedAt: time.Now().UTC(),
		DestPath:  destRoot,
	}
	registry.mu.Lock()
	registry.jobs[jobID] = job
	registry.evictOldJobsLocked()
	snapshot := *job
	registry.mu.Unlock()

	writeROCmServeJSON(w, http.StatusAccepted, snapshot)
	go func() {
		defer registry.release()
		registry.run(job, req)
	}()
}

func validateROCmDownloadRequest(req rocmAdminDownloadRequest) error {
	if err := validateROCmRepoID(req.Repo); err != nil {
		return err
	}
	if err := validateROCmRevision(req.Revision); err != nil {
		return err
	}
	for _, file := range req.Files {
		if !isSafeROCmHFEntryPath(file) {
			return fmt.Errorf("unsafe file path %q", file)
		}
	}
	return nil
}

func (registry *rocmAdminDownloadRegistry) run(job *rocmAdminDownloadJob, req rocmAdminDownloadRequest) {
	defer func() {
		registry.mu.Lock()
		finished := time.Now().UTC()
		job.FinishedAt = &finished
		registry.mu.Unlock()
	}()

	registry.mu.Lock()
	job.Status = "running"
	registry.mu.Unlock()

	entries, err := registry.cfg.TreeAPI.ResolveTree(registry.ctx, req.Repo, req.Revision)
	if err != nil {
		registry.fail(job, "resolve tree: "+err.Error())
		return
	}
	wanted := filterROCmDownloadEntries(entries, req.Files)
	if len(wanted) == 0 {
		registry.fail(job, "no files matched")
		return
	}
	var total int64
	for _, entry := range wanted {
		total += entry.Size
	}
	registry.mu.Lock()
	job.BytesTotal = total
	job.FileCount = len(wanted)
	registry.mu.Unlock()

	parent := filepath.Dir(job.DestPath)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		registry.fail(job, "mkdir model parent: "+err.Error())
		return
	}
	free := rocmDiskFreeBytes(parent)
	if free > 0 && total > 0 && free < uint64(total*2) {
		registry.fail(job, fmt.Sprintf("disk-space: free=%d need=%d", free, total*2))
		return
	}

	finalDir := job.DestPath
	quarantineDir := finalDir + ".quarantine"
	_ = os.RemoveAll(quarantineDir)
	if err := os.MkdirAll(quarantineDir, 0o755); err != nil {
		registry.fail(job, "mkdir quarantine: "+err.Error())
		return
	}

	digests := make(map[string]string, len(wanted))
	var done int64
	for _, entry := range wanted {
		if err := registry.ctx.Err(); err != nil {
			registry.fail(job, "cancelled: "+err.Error())
			return
		}
		destFile := filepath.Join(quarantineDir, filepath.FromSlash(entry.Path))
		if err := os.MkdirAll(filepath.Dir(destFile), 0o755); err != nil {
			registry.fail(job, "mkdir file dir: "+err.Error())
			return
		}
		written, sha, err := registry.cfg.Fetch(registry.ctx, entry.URL, destFile, entry.Digest, entry.Size)
		if err != nil {
			registry.fail(job, "fetch "+entry.Path+": "+err.Error())
			return
		}
		digests[entry.Path] = sha
		done += written
		registry.mu.Lock()
		job.BytesDone = done
		registry.mu.Unlock()
	}

	if err := writeROCmDownloadManifest(quarantineDir, digests); err != nil {
		registry.fail(job, "write manifest: "+err.Error())
		return
	}
	if err := os.RemoveAll(finalDir); err != nil {
		registry.fail(job, "remove old: "+err.Error())
		return
	}
	if err := os.Rename(quarantineDir, finalDir); err != nil {
		registry.fail(job, "promote: "+err.Error())
		return
	}

	registry.mu.Lock()
	job.Status = "done"
	registry.mu.Unlock()
}

func filterROCmDownloadEntries(entries []rocmHFFileEntry, files []string) []rocmHFFileEntry {
	if len(files) == 0 {
		return append([]rocmHFFileEntry(nil), entries...)
	}
	want := map[string]struct{}{}
	for _, file := range files {
		want[file] = struct{}{}
	}
	filtered := make([]rocmHFFileEntry, 0, len(files))
	for _, entry := range entries {
		if _, ok := want[entry.Path]; ok {
			filtered = append(filtered, entry)
		}
	}
	return filtered
}

func (registry *rocmAdminDownloadRegistry) fail(job *rocmAdminDownloadJob, reason string) {
	registry.mu.Lock()
	job.Status = "failed"
	job.Error = reason
	registry.mu.Unlock()
}

func (registry *rocmAdminDownloadRegistry) tryAcquire() bool {
	select {
	case registry.activeSlots <- struct{}{}:
		return true
	default:
		return false
	}
}

func (registry *rocmAdminDownloadRegistry) release() {
	<-registry.activeSlots
}

func (registry *rocmAdminDownloadRegistry) evictOldJobsLocked() {
	for len(registry.jobs) > maxROCmDownloadJobsRetained {
		var oldestID string
		var oldest time.Time
		for id, job := range registry.jobs {
			if job.Status != "done" && job.Status != "failed" {
				continue
			}
			if oldestID == "" || job.StartedAt.Before(oldest) {
				oldestID = id
				oldest = job.StartedAt
			}
		}
		if oldestID == "" {
			return
		}
		delete(registry.jobs, oldestID)
	}
}

type rocmAllowedModelsFile struct {
	Repos []string `json:"repos"`
}

func loadROCmAllowedModels(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, nil
	}
	var file rocmAllowedModelsFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, err
	}
	return file.Repos, nil
}

func isROCmRepoAllowed(allowed []string, repo string) bool {
	for _, value := range allowed {
		if value == repo {
			return true
		}
	}
	return false
}

func validateROCmRepoID(repo string) error {
	if repo == "" {
		return errors.New("repo required")
	}
	if len(repo) > 256 {
		return errors.New("repo too long")
	}
	if strings.HasPrefix(repo, "/") || strings.HasSuffix(repo, "/") || strings.Contains(repo, "//") {
		return errors.New("repo must be a HuggingFace repo id")
	}
	for _, segment := range strings.Split(repo, "/") {
		if segment == "" || segment == "." || segment == ".." || strings.HasPrefix(segment, ".") {
			return errors.New("repo contains unsafe path segment")
		}
		for _, c := range segment {
			ok := c >= 'a' && c <= 'z' ||
				c >= 'A' && c <= 'Z' ||
				c >= '0' && c <= '9' ||
				c == '-' || c == '_' || c == '.'
			if !ok {
				return errors.New("repo contains disallowed character")
			}
		}
	}
	return nil
}

func validateROCmRevision(revision string) error {
	if revision == "" {
		return errors.New("revision required")
	}
	if len(revision) > 64 {
		return errors.New("revision too long")
	}
	for _, c := range revision {
		ok := c >= 'a' && c <= 'z' ||
			c >= 'A' && c <= 'Z' ||
			c >= '0' && c <= '9' ||
			c == '-' || c == '_' || c == '.'
		if !ok {
			return errors.New("revision contains disallowed character")
		}
	}
	return nil
}

func canonicaliseROCmRepoName(repo string) string {
	return strings.ReplaceAll(repo, "/", "__")
}

func isSafeROCmHFEntryPath(path string) bool {
	if path == "" || strings.HasPrefix(path, "/") || strings.Contains(path, "\x00") {
		return false
	}
	for _, segment := range strings.Split(path, "/") {
		if segment == "" || segment == "." || segment == ".." || strings.HasPrefix(segment, ".") {
			return false
		}
	}
	return true
}

func rocmAdminDownloadAllowedPath(cfg rocmAdminDownloadConfig) string {
	if strings.TrimSpace(cfg.AllowedModelsPath) != "" {
		return cfg.AllowedModelsPath
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "allowed-models.json"
	}
	return filepath.Join(home, "Lethean", "data", "allowed-models.json")
}

func rocmAdminDownloadModelDir(cfg rocmAdminDownloadConfig) string {
	if strings.TrimSpace(cfg.ModelDir) != "" {
		return cfg.ModelDir
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "models"
	}
	return filepath.Join(home, "Lethean", "data", "models")
}

func rocmDownloadJobID() string {
	return fmt.Sprintf("download-%d", time.Now().UTC().UnixNano())
}

func rocmDiskFreeBytes(path string) uint64 {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return 0
	}
	return uint64(stat.Bavail) * uint64(stat.Bsize)
}

func writeROCmDownloadManifest(dir string, digests map[string]string) error {
	paths := make([]string, 0, len(digests))
	for path := range digests {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	var builder strings.Builder
	for _, path := range paths {
		builder.WriteString(digests[path])
		builder.WriteString("  ")
		builder.WriteString(path)
		builder.WriteByte('\n')
	}
	return os.WriteFile(filepath.Join(dir, ".sha256"), []byte(builder.String()), 0o600)
}

func rocmFetchAndVerify(ctx context.Context, sourceURL, destPath, expectedDigest string, expectedSize int64) (int64, string, error) {
	if !strings.HasPrefix(sourceURL, rocmHFHostResolve) {
		return 0, "", errors.New("disallowed source: " + sourceURL)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, sourceURL, nil)
	if err != nil {
		return 0, "", err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return 0, "", fmt.Errorf("HTTP %d from %s", resp.StatusCode, sourceURL)
	}
	if resp.ContentLength > rocmHFFileCap {
		return 0, "", fmt.Errorf("Content-Length %d exceeds cap %d", resp.ContentLength, rocmHFFileCap)
	}
	flags := os.O_CREATE | os.O_EXCL | os.O_WRONLY | syscall.O_NOFOLLOW
	file, err := os.OpenFile(destPath, flags, 0o600)
	if err != nil {
		if errors.Is(err, syscall.ELOOP) {
			return 0, "", fmt.Errorf("quarantine_symlink_refused: %s: %w", destPath, err)
		}
		if errors.Is(err, os.ErrExist) {
			return 0, "", fmt.Errorf("quarantine_exists: %s: %w", destPath, err)
		}
		return 0, "", err
	}
	hasher := sha256.New()
	written, copyErr := io.Copy(io.MultiWriter(file, hasher), io.LimitReader(resp.Body, rocmHFFileCap+1))
	closeErr := file.Close()
	if copyErr != nil {
		_ = os.Remove(destPath)
		return 0, "", copyErr
	}
	if closeErr != nil {
		_ = os.Remove(destPath)
		return 0, "", closeErr
	}
	if written > rocmHFFileCap {
		_ = os.Remove(destPath)
		return 0, "", fmt.Errorf("download exceeded %d byte cap", rocmHFFileCap)
	}
	computed := hex.EncodeToString(hasher.Sum(nil))
	if expectedDigest != "" && computed != strings.ToLower(expectedDigest) {
		_ = os.Remove(destPath)
		return 0, "", fmt.Errorf("sha256 mismatch: got=%s want=%s", computed, expectedDigest)
	}
	if expectedSize > 0 && written != expectedSize {
		return written, computed, nil
	}
	return written, computed, nil
}

type rocmHFFileEntry struct {
	Path   string
	URL    string
	Size   int64
	Digest string
}

type rocmHFTreeAPI interface {
	ResolveTree(ctx context.Context, repo, revision string) ([]rocmHFFileEntry, error)
}

type rocmHFTreeClient struct {
	httpClient *http.Client
}

type rocmHFTreeEntryRaw struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size int64  `json:"size"`
	LFS  *struct {
		SHA256 string `json:"sha256"`
		Size   int64  `json:"size"`
	} `json:"lfs"`
}

func (client *rocmHFTreeClient) ResolveTree(ctx context.Context, repo, revision string) ([]rocmHFFileEntry, error) {
	if client == nil || client.httpClient == nil {
		client = &rocmHFTreeClient{httpClient: http.DefaultClient}
	}
	revision = strings.TrimSpace(revision)
	if revision == "" {
		revision = "main"
	}
	if err := validateROCmRepoID(repo); err != nil {
		return nil, err
	}
	if err := validateROCmRevision(revision); err != nil {
		return nil, err
	}
	apiURL := rocmHFHostTreeAPI + escapeROCmHFPath(repo) + "/tree/" + url.PathEscape(revision) + "?expand=true"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, apiURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return nil, fmt.Errorf("HTTP %d; private repo or token required", resp.StatusCode)
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("HTTP %d from tree API", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, rocmHFTreeResponseCap+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > rocmHFTreeResponseCap {
		return nil, fmt.Errorf("tree response exceeds %d byte cap", rocmHFTreeResponseCap)
	}
	var raw []rocmHFTreeEntryRaw
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, err
	}
	entries := make([]rocmHFFileEntry, 0, len(raw))
	for _, entry := range raw {
		if entry.Type != "file" || !isSafeROCmHFEntryPath(entry.Path) {
			continue
		}
		item := rocmHFFileEntry{
			Path: entry.Path,
			URL:  rocmHFHostResolve + escapeROCmHFPath(repo) + "/resolve/" + url.PathEscape(revision) + "/" + escapeROCmHFPath(entry.Path),
			Size: entry.Size,
		}
		if entry.LFS != nil {
			item.Digest = strings.ToLower(strings.TrimSpace(entry.LFS.SHA256))
			if item.Size == 0 && entry.LFS.Size > 0 {
				item.Size = entry.LFS.Size
			}
		}
		entries = append(entries, item)
	}
	return entries, nil
}

func escapeROCmHFPath(path string) string {
	segments := strings.Split(path, "/")
	for i, segment := range segments {
		segments[i] = url.PathEscape(segment)
	}
	return strings.Join(segments, "/")
}
