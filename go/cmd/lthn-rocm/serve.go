// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"dappco.re/go/inference"
	anthropiccompat "dappco.re/go/inference/anthropic"
	ollamacompat "dappco.re/go/inference/ollama"
	openaicompat "dappco.re/go/inference/openai"
	state "dappco.re/go/inference/state"
	rocm "dappco.re/go/rocm"
)

const (
	rocmServeHealthPath           = "/v1/health"
	rocmServeModelsPath           = "/v1/models"
	rocmServeRuntimeWakePath      = "/v1/runtime/wake"
	rocmServeRuntimeSleepPath     = "/v1/runtime/sleep"
	rocmServeCacheEntriesPath     = "/v1/cache/entries"
	rocmServeAdminMachinePath     = "/v1/admin/machine"
	rocmServeAdminStatusPath      = "/v1/admin/serve/status"
	rocmServeAdminReloadPath      = "/v1/admin/serve/reload"
	rocmServeAdminDownloadPath    = "/v1/admin/models/download"
	rocmServeAdminSFTStartPath    = "/v1/admin/sft/start"
	rocmServeAdminSFTStatusPath   = "/v1/admin/sft/status"
	rocmServeAdminSFTStopPath     = "/v1/admin/sft/stop"
	rocmServeAdminSFTAdaptersPath = "/v1/admin/sft/adapters"

	rocmServeSHAManifestFilename = ".sha256"
	rocmAdminTokenPrefix         = "lthn-rocm_"
	rocmAdminAuthRealm           = `Bearer realm="lthn-rocm-admin"`
)

type rocmServeConfig struct {
	Backend                  string
	ModelPath                string
	ContextLen               int
	KVCacheMode              string
	ROCmLoadConfig           rocm.ROCmLoadConfig
	DraftPath                string
	DraftDetect              bool
	DraftBlock               int
	DraftDetection           rocm.DraftDetection
	NoAutoProfile            bool
	ProfileDir               string
	StateConversations       bool
	StateStorePath           string
	StateStore               state.Store
	StateContinuityReason    string
	LoadOptions              []inference.LoadOption
	AdminToken               string
	AdminDownloadAllowedPath string
	AdminDownloadModelDir    string
	AdminDownloadTreeAPI     rocmHFTreeAPI
	AdminDownloadFetch       rocmAdminFetchFunc
	AdminDownloadContext     context.Context
	AdminSFTAdapterRoot      string
	AdminAudit               io.Writer
}

type rocmServeResolver struct {
	mu              sync.Mutex
	backend         string
	modelPath       string
	contextLen      int
	kvCacheMode     string
	rocmLoadCfg     rocm.ROCmLoadConfig
	draftPath       string
	draftDetect     bool
	draftBlock      int
	noAutoProfile   bool
	profileDir      string
	profilePath     string
	adapterPath     string
	statusConfig    rocmServeStatusConfig
	loadOptions     []inference.LoadOption
	model           inference.TextModel
	loadedAt        time.Time
	loadedSelection rocmServeDraftSelection
	stateContinuity *rocmServeConversationContinuityManager
}

func newROCmServeResolver(cfg rocmServeConfig) *rocmServeResolver {
	return &rocmServeResolver{
		backend:         firstServeNonEmptyString(normalizeBackendName(cfg.Backend), defaultBackendName),
		modelPath:       strings.TrimSpace(cfg.ModelPath),
		contextLen:      cfg.ContextLen,
		kvCacheMode:     strings.TrimSpace(cfg.KVCacheMode),
		rocmLoadCfg:     cfg.ROCmLoadConfig,
		draftPath:       strings.TrimSpace(cfg.DraftPath),
		draftDetect:     cfg.DraftDetect,
		draftBlock:      cfg.DraftBlock,
		noAutoProfile:   cfg.NoAutoProfile,
		profileDir:      strings.TrimSpace(cfg.ProfileDir),
		loadOptions:     append([]inference.LoadOption(nil), cfg.LoadOptions...),
		statusConfig:    rocmServeStatusConfigFromLoad(cfg.ROCmLoadConfig, cfg.LoadOptions, cfg.ContextLen, ""),
		stateContinuity: newROCmServeConversationContinuityManager(cfg),
	}
}

func (resolver *rocmServeResolver) ResolveModel(ctx context.Context, _ string) (inference.TextModel, error) {
	if resolver == nil {
		return nil, errors.New("serve resolver is nil")
	}
	if err := ctxErr(ctx); err != nil {
		return nil, err
	}
	resolver.mu.Lock()
	defer resolver.mu.Unlock()
	if resolver.model != nil {
		return resolver.model, nil
	}
	if resolver.modelPath == "" {
		return nil, errors.New("no model loaded; POST /v1/admin/serve/reload or restart with --model")
	}
	opts := append([]inference.LoadOption(nil), resolver.loadOptions...)
	if resolver.backend != "" && resolver.backend != "auto" {
		opts = append(opts, inference.WithBackend(resolver.backend))
	}
	selection := resolver.resolveDraftLocked(ctx)
	if selection.Detection.Active() {
		model, err := rocm.LoadAttachedDrafterPairAsTextModelBlockWithConfig(
			resolver.modelPath,
			selection.Detection.DraftPath,
			selection.DraftBlock,
			resolver.rocmLoadCfg,
			opts...,
		)
		if err != nil {
			stateContinuity, _, _ := resolver.stateContinuity.Status()
			return nil, rocmServeNativeDrafterLoadError(selection.Detection, selection.DraftBlock, selection.DraftBlockSource, stateContinuity, err)
		}
		resolver.model = model
		resolver.loadedAt = time.Now()
		resolver.loadedSelection = selection
		return model, nil
	}
	model, err := loadROCmCLITextModel(resolver.modelPath, resolver.rocmLoadCfg, opts)
	if err != nil {
		return nil, err
	}
	resolver.model = model
	resolver.loadedAt = time.Now()
	resolver.loadedSelection = selection
	return model, nil
}

type rocmServeReloadOptions struct {
	Path           string
	Backend        string
	ContextLen     int
	ProfilePath    string
	AdapterPath    string
	ROCmLoadConfig rocm.ROCmLoadConfig
	LoadOptions    []inference.LoadOption
	StatusConfig   rocmServeStatusConfig
}

func (resolver *rocmServeResolver) Reload(path, backend string, contextLen int) {
	resolver.ReloadWithOptions(rocmServeReloadOptions{Path: path, Backend: backend, ContextLen: contextLen})
}

func (resolver *rocmServeResolver) ReloadWithOptions(reload rocmServeReloadOptions) {
	if resolver == nil {
		return
	}
	path := strings.TrimSpace(reload.Path)
	backend := normalizeBackendName(reload.Backend)
	if backend == "" {
		backend = defaultBackendName
	}
	contextLen := reload.ContextLen
	profilePath := strings.TrimSpace(reload.ProfilePath)
	adapterPath := strings.TrimSpace(reload.AdapterPath)
	opts := append([]inference.LoadOption(nil), reload.LoadOptions...)
	rocmLoadCfg := reload.ROCmLoadConfig
	resolver.mu.Lock()
	if len(opts) == 0 {
		opts = append(opts, resolver.loadOptions...)
	}
	if rocmServeLoadConfigEmpty(rocmLoadCfg) {
		rocmLoadCfg = resolver.rocmLoadCfg
	}
	resolver.mu.Unlock()
	loadConfig := inference.ApplyLoadOpts(opts)
	if contextLen > 0 && loadConfig.ContextLen != contextLen {
		opts = append(opts, inference.WithContextLen(contextLen))
	}
	if adapterPath != "" && loadConfig.AdapterPath != adapterPath {
		opts = append(opts, inference.WithAdapterPath(adapterPath))
	}
	statusConfig := reload.StatusConfig
	if statusConfig == (rocmServeStatusConfig{}) {
		statusConfig = rocmServeStatusConfigFromLoad(rocmLoadCfg, opts, contextLen, adapterPath)
	}
	resolver.mu.Lock()
	old := resolver.model
	resolver.model = nil
	resolver.loadedAt = time.Time{}
	resolver.loadedSelection = rocmServeDraftSelection{}
	resolver.modelPath = path
	resolver.backend = backend
	resolver.contextLen = contextLen
	resolver.kvCacheMode = firstServeNonEmptyString(rocmLoadCfg.CacheMode, rocmLoadCfg.DeviceKVMode)
	resolver.profilePath = profilePath
	resolver.adapterPath = adapterPath
	resolver.rocmLoadCfg = rocmLoadCfg
	resolver.statusConfig = statusConfig
	resolver.loadOptions = opts
	resolver.mu.Unlock()
	if old != nil {
		_ = old.Close()
	}
}

func (resolver *rocmServeResolver) resolveDraftLocked(ctx context.Context) rocmServeDraftSelection {
	if resolver == nil {
		return rocmServeDraftSelection{}
	}
	return resolveROCmServeDraftSelection(ctx,
		resolver.modelPath,
		resolver.draftPath,
		resolver.draftDetect,
		resolver.draftBlock,
		resolver.noAutoProfile,
		resolver.profileDir,
	)
}

func (resolver *rocmServeResolver) Wake(ctx context.Context) error {
	if resolver == nil {
		return errors.New("serve resolver is nil")
	}
	status := resolver.Status()
	if strings.TrimSpace(status.ModelPath) == "" {
		return nil
	}
	_, err := resolver.ResolveModel(ctx, status.ModelName)
	return err
}

func (resolver *rocmServeResolver) Sleep() error {
	if resolver == nil {
		return nil
	}
	resolver.mu.Lock()
	model := resolver.model
	resolver.model = nil
	resolver.loadedAt = time.Time{}
	resolver.loadedSelection = rocmServeDraftSelection{}
	resolver.mu.Unlock()
	if model != nil {
		return model.Close()
	}
	return nil
}

func (resolver *rocmServeResolver) Close() error {
	if resolver == nil {
		return nil
	}
	return resolver.Sleep()
}

func (resolver *rocmServeResolver) OllamaModelNames(context.Context) ([]string, error) {
	snapshot := resolver.Status()
	if snapshot.ModelName == "" {
		return nil, nil
	}
	return []string{snapshot.ModelName}, nil
}

func (resolver *rocmServeResolver) Status() rocmServeResolverStatus {
	if resolver == nil {
		return rocmServeResolverStatus{}
	}
	resolver.mu.Lock()
	defer resolver.mu.Unlock()
	selection := resolver.resolveDraftLocked(context.Background())
	if resolver.model != nil && resolver.loadedSelection.Detection.Active() {
		selection = resolver.loadedSelection
	}
	memory := rocmServeStatusMemory{}
	modelProfile := rocmServeResolveModelProfile(resolver.model, resolver.modelPath)
	modelRoutes := rocmModelRoutePlanForProfileAndModel(modelProfile, resolver.model)
	tokenLoop := rocmTokenLoopForProfileAndModel(modelProfile, resolver.model)
	if resolver.model != nil {
		metrics := resolver.model.Metrics()
		memory.ActiveBytes = metrics.ActiveMemoryBytes
		memory.PeakBytes = metrics.PeakMemoryBytes
	}
	stateContinuity, stateReason, stateStats := resolver.stateContinuity.Status()
	nativeMTPAttachment := rocmServeNativeMTPAttachmentStatus(resolver.model)
	return rocmServeResolverStatus{
		Backend:             resolver.backend,
		ModelPath:           resolver.modelPath,
		ModelName:           rocmServeModelName(resolver.modelPath),
		ContextLen:          resolver.contextLen,
		KVCacheMode:         resolver.kvCacheMode,
		ProfilePath:         resolver.profilePath,
		AdapterPath:         resolver.adapterPath,
		Loaded:              resolver.model != nil,
		LoadedAt:            resolver.loadedAt,
		ModelProfile:        modelProfile,
		ModelRoutes:         modelRoutes,
		TokenLoop:           tokenLoop,
		DraftDetection:      selection.Detection,
		DraftBlock:          selection.DraftBlock,
		DraftBlockSource:    selection.DraftBlockSource,
		Memory:              memory,
		statusConfig:        resolver.statusConfig,
		StateContinuity:     stateContinuity,
		StateReason:         stateReason,
		StateStats:          stateStats,
		NativeMTPAttachment: nativeMTPAttachment,
	}
}

type rocmServeResolverStatus struct {
	Backend             string                    `json:"backend,omitempty"`
	ModelPath           string                    `json:"model_path,omitempty"`
	ModelName           string                    `json:"model_name,omitempty"`
	ContextLen          int                       `json:"context,omitempty"`
	KVCacheMode         string                    `json:"kv_cache,omitempty"`
	ProfilePath         string                    `json:"profile_path,omitempty"`
	AdapterPath         string                    `json:"adapter_path,omitempty"`
	Loaded              bool                      `json:"loaded"`
	LoadedAt            time.Time                 `json:"loaded_at,omitempty"`
	ModelProfile        *rocm.ROCmModelProfile    `json:"model_profile,omitempty"`
	ModelRoutes         *rocm.ROCmModelRoutePlan  `json:"model_routes,omitempty"`
	TokenLoop           *rocm.ROCmTokenLoopStatus `json:"token_loop,omitempty"`
	DraftDetection      rocm.DraftDetection       `json:"draft_detection,omitempty"`
	DraftBlock          int                       `json:"draft_block,omitempty"`
	DraftBlockSource    string                    `json:"draft_block_source,omitempty"`
	Memory              rocmServeStatusMemory     `json:"memory"`
	statusConfig        rocmServeStatusConfig
	StateContinuity     string                               `json:"state_continuity,omitempty"`
	StateReason         string                               `json:"state_reason,omitempty"`
	StateStats          rocmServeConversationContinuityStats `json:"state_stats,omitempty"`
	NativeMTPAttachment string                               `json:"native_mtp_attachment,omitempty"`
}

type rocmServeAdminStatusResponse struct {
	Status           string                    `json:"status"`
	Runtime          string                    `json:"runtime"`
	CLIContract      string                    `json:"cli_contract"`
	ModelPath        string                    `json:"model_path,omitempty"`
	ProfilePath      string                    `json:"profile_path,omitempty"`
	AdapterPath      string                    `json:"adapter_path,omitempty"`
	LoadedAtUnix     int64                     `json:"loaded_at_unix,omitempty"`
	ModelRegistry    modelRegistryReport       `json:"model_registry"`
	ModelProfile     *rocm.ROCmModelProfile    `json:"model_profile,omitempty"`
	ModelRoutes      *rocm.ROCmModelRoutePlan  `json:"model_routes,omitempty"`
	TokenLoop        *rocm.ROCmTokenLoopStatus `json:"token_loop,omitempty"`
	Config           rocmServeStatusConfig     `json:"config"`
	Memory           rocmServeStatusMemory     `json:"memory"`
	Resolver         rocmServeResolverStatus   `json:"resolver"`
	DraftDetection   rocm.DraftDetection       `json:"draft_detection,omitempty"`
	DraftBlock       int                       `json:"draft_block,omitempty"`
	DraftBlockSource string                    `json:"draft_block_source,omitempty"`
	Labels           map[string]string         `json:"labels,omitempty"`
}

type rocmServeStatusConfig struct {
	ContextLength        int    `json:"context_length,omitempty"`
	ParallelSlots        int    `json:"parallel_slots,omitempty"`
	PromptCache          bool   `json:"prompt_cache"`
	PromptCacheMinTokens int    `json:"prompt_cache_min_tokens,omitempty"`
	CachePolicy          string `json:"cache_policy,omitempty"`
	KVCache              string `json:"kv_cache,omitempty"`
	CacheMode            string `json:"cache_mode,omitempty"`
	DeviceKVMode         string `json:"device_kv_mode,omitempty"`
	BatchSize            int    `json:"batch_size,omitempty"`
	PrefillChunkSize     int    `json:"prefill_chunk_size,omitempty"`
	ExpectedQuantization int    `json:"expected_quantization,omitempty"`
	MemoryLimitBytes     uint64 `json:"memory_limit_bytes,omitempty"`
	CacheLimitBytes      uint64 `json:"cache_limit_bytes,omitempty"`
	WiredLimitBytes      uint64 `json:"wired_limit_bytes,omitempty"`
	AdapterPath          string `json:"adapter_path,omitempty"`
	Draft                string `json:"draft,omitempty"`
	DraftDetect          bool   `json:"draft_detect"`
	DraftBlock           int    `json:"draft_block,omitempty"`
	NoAutoProfile        bool   `json:"no_auto_profile"`
	ProfileDir           string `json:"profile_dir,omitempty"`
	StateConversations   bool   `json:"state_conversations"`
	StateStore           string `json:"state_store,omitempty"`
	StateContinuity      string `json:"state_continuity,omitempty"`
	StateReason          string `json:"state_reason,omitempty"`
}

type rocmServeStatusMemory struct {
	ActiveBytes uint64 `json:"active_bytes"`
	CacheBytes  uint64 `json:"cache_bytes"`
	PeakBytes   uint64 `json:"peak_bytes"`
}

type rocmServeAdminMachineResponse struct {
	Hash        string            `json:"hash"`
	Hostname    string            `json:"hostname,omitempty"`
	Runtime     string            `json:"runtime"`
	GoVersion   string            `json:"go_version,omitempty"`
	OS          string            `json:"os,omitempty"`
	Arch        string            `json:"arch,omitempty"`
	Time        int64             `json:"time"`
	CLIContract string            `json:"cli_contract"`
	Compile     compileReport     `json:"compile"`
	Labels      map[string]string `json:"labels,omitempty"`
}

func rocmServeNativeDrafterPendingError(detection rocm.DraftDetection, draftBlock int, draftBlockSource, stateContinuity string) error {
	source := ""
	if strings.TrimSpace(draftBlockSource) != "" {
		source = "; " + strings.TrimSpace(draftBlockSource)
	}
	stateRoute := ""
	if rocmServeStateContinuitySupportsRetainedMTP(stateContinuity) {
		stateRoute = "; retained-state OpenAI/server route will use rocm_state_session_runtime_kv once native attachment is linked"
	}
	return fmt.Errorf("reactive MTP drafter resolved: %s (%s), block %s%s; native ROCm drafter execution is pending%s; refusing autoregressive fallback",
		detection.DraftPath, detection.Note, resolvedROCmDraftBlockLabel(draftBlock), source, stateRoute)
}

func rocmServeNativeDrafterLoadError(detection rocm.DraftDetection, draftBlock int, draftBlockSource, stateContinuity string, err error) error {
	base := rocmServeNativeDrafterPendingError(detection, draftBlock, draftBlockSource, stateContinuity)
	if err == nil {
		return base
	}
	return fmt.Errorf("%w: %v", base, err)
}

func newROCmServeHandler(cfg rocmServeConfig) (http.Handler, *rocmServeResolver) {
	resolver := newROCmServeResolver(cfg)
	generationResolver := newROCmServeGenerationResolver(resolver)
	downloadCtx := cfg.AdminDownloadContext
	if downloadCtx == nil {
		downloadCtx = context.Background()
	}
	downloadRegistry := newROCmAdminDownloadRegistry(downloadCtx, rocmAdminDownloadConfig{
		AllowedModelsPath: cfg.AdminDownloadAllowedPath,
		ModelDir:          cfg.AdminDownloadModelDir,
		TreeAPI:           cfg.AdminDownloadTreeAPI,
		Fetch:             cfg.AdminDownloadFetch,
	})
	sftRegistry := newROCmAdminSFTRegistry()
	mux := http.NewServeMux()
	mux.Handle(openaicompat.DefaultChatCompletionsPath, openaicompat.NewHandler(generationResolver))
	mux.Handle(openaicompat.DefaultResponsesPath, rocm.NewOpenAIResponsesHandler(generationResolver))
	mux.Handle(openaicompat.DefaultEmbeddingsPath, openaicompat.NewEmbeddingsHandler(resolver))
	mux.Handle(openaicompat.DefaultRerankPath, openaicompat.NewRerankHandler(resolver))
	mux.Handle(openaicompat.DefaultCapabilitiesPath, openaicompat.NewCapabilityHandler(resolver))
	mux.Handle(openaicompat.DefaultCacheStatsPath, openaicompat.NewCacheStatsHandler(resolver))
	mux.Handle(openaicompat.DefaultCacheWarmPath, openaicompat.NewCacheWarmHandler(resolver))
	mux.Handle(openaicompat.DefaultCacheClearPath, openaicompat.NewCacheClearHandler(resolver))
	mux.Handle(openaicompat.DefaultCancelPath, openaicompat.NewCancelHandler(resolver))
	mux.HandleFunc(rocmServeScorePath, handleROCmScorePair)
	mux.Handle(anthropiccompat.DefaultMessagesPath, rocm.NewAnthropicMessagesHandler(generationResolver))
	ollamaMux := rocm.NewOllamaHandler(generationResolver)
	mux.Handle(ollamacompat.DefaultChatPath, ollamaMux)
	mux.Handle(ollamacompat.DefaultGeneratePath, ollamaMux)
	mux.Handle(ollamacompat.DefaultTagsPath, ollamaMux)
	mux.Handle(ollamacompat.DefaultShowPath, ollamaMux)
	mux.HandleFunc(rocmServeHealthPath, func(w http.ResponseWriter, r *http.Request) {
		if !rocmServeRequireMethod(w, r, http.MethodGet) {
			return
		}
		status := resolver.Status()
		models := []string(nil)
		if status.ModelName != "" {
			models = []string{status.ModelName}
		}
		writeROCmServeJSON(w, http.StatusOK, map[string]any{
			"status":  "ok",
			"runtime": defaultBackendName,
			"models":  models,
			"time":    time.Now().Unix(),
			"labels":  rocmServeLabelsForStatus(cfg, status),
		})
	})
	mux.HandleFunc(rocmServeModelsPath, func(w http.ResponseWriter, r *http.Request) {
		if !rocmServeRequireMethod(w, r, http.MethodGet) {
			return
		}
		writeROCmServeJSON(w, http.StatusOK, rocmServeModelsResponse(cfg, resolver.Status()))
	})
	mux.Handle(rocmServeRuntimeWakePath, rocmServeAdminAuth(cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !rocmServeRequireMethod(w, r, http.MethodPost) {
			return
		}
		if err := resolver.Wake(r.Context()); err != nil {
			writeROCmServeError(w, http.StatusInternalServerError, err.Error(), "wake")
			return
		}
		writeROCmServeJSON(w, http.StatusOK, rocmServeRuntimeActionResponse("wake", cfg, resolver.Status()))
	})))
	mux.Handle(rocmServeRuntimeSleepPath, rocmServeAdminAuth(cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !rocmServeRequireMethod(w, r, http.MethodPost) {
			return
		}
		if err := resolver.Sleep(); err != nil {
			writeROCmServeError(w, http.StatusInternalServerError, err.Error(), "sleep")
			return
		}
		writeROCmServeJSON(w, http.StatusOK, rocmServeRuntimeActionResponse("sleep", cfg, resolver.Status()))
	})))
	mux.Handle(rocmServeCacheEntriesPath, rocmServeAdminAuth(cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !rocmServeRequireMethod(w, r, http.MethodGet) {
			return
		}
		modelName := strings.TrimSpace(r.URL.Query().Get("model"))
		model, err := resolver.ResolveModel(r.Context(), modelName)
		if err != nil {
			writeROCmServeError(w, http.StatusNotFound, err.Error(), "model")
			return
		}
		lister, ok := model.(rocmServeCacheEntryLister)
		if !ok {
			writeROCmServeError(w, http.StatusNotImplemented, "model does not support cache entry listing", "model")
			return
		}
		entries, err := lister.CacheEntries(r.Context(), rocmServeCacheEntryLabels(r))
		if err != nil {
			writeROCmServeError(w, http.StatusInternalServerError, err.Error(), "cache")
			return
		}
		response := rocmServeCacheEntriesResponse{
			Object:  "list",
			Model:   firstServeNonEmptyString(modelName, resolver.Status().ModelName),
			Entries: entries,
		}
		if service, ok := model.(inference.CacheService); ok {
			stats, err := service.CacheStats(r.Context())
			if err != nil {
				writeROCmServeError(w, http.StatusInternalServerError, err.Error(), "cache")
				return
			}
			response.Stats = &stats
		}
		writeROCmServeJSON(w, http.StatusOK, response)
	})))
	mux.Handle(rocmServeAdminMachinePath, rocmServeAdminAuth(cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !rocmServeRequireMethod(w, r, http.MethodGet) {
			return
		}
		machine, err := buildROCmServeAdminMachineResponse(r.Context(), cfg)
		if err != nil {
			writeROCmServeError(w, http.StatusInternalServerError, "machine hash unavailable: "+err.Error(), "machine")
			return
		}
		writeROCmServeJSON(w, http.StatusOK, machine)
	})))
	mux.Handle(rocmServeAdminStatusPath, rocmServeAdminAuth(cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !rocmServeRequireMethod(w, r, http.MethodGet) {
			return
		}
		writeROCmServeJSON(w, http.StatusOK, rocmServeStatusResponse(cfg, resolver.Status()))
	})))
	mux.Handle(rocmServeAdminReloadPath, rocmServeAdminAuth(cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !rocmServeRequireMethod(w, r, http.MethodPost) {
			return
		}
		var req rocmServeReloadRequest
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64*1024)).Decode(&req); err != nil {
			writeROCmServeError(w, http.StatusBadRequest, "invalid request body", "body")
			return
		}
		fromPath := resolver.Status().ModelPath
		auditTarget := firstServeNonEmptyString(req.ModelPath, req.Path, req.Model)
		auditROCmServeReload(cfg.AdminAudit, "attempt", r, fromPath, auditTarget, req.AdapterPath, "")
		if strings.TrimSpace(auditTarget) == "" {
			auditROCmServeReload(cfg.AdminAudit, "deny", r, fromPath, auditTarget, req.AdapterPath, "model or model_path required")
			writeROCmServeError(w, http.StatusBadRequest, "model or model_path required", "model")
			return
		}
		confirmation := strings.TrimSpace(firstServeNonEmptyString(req.ConfirmMachine, req.Confirmation))
		if confirmation == "" {
			auditROCmServeReload(cfg.AdminAudit, "deny", r, fromPath, auditTarget, req.AdapterPath, "confirm_machine required")
			writeROCmServeError(w, http.StatusBadRequest, "confirm_machine required (machine_hash from /v1/admin/machine)", "confirm_machine")
			return
		}
		expected, err := rocm.CurrentMachineProfileHash(r.Context())
		if err != nil {
			auditROCmServeReload(cfg.AdminAudit, "fail", r, fromPath, auditTarget, req.AdapterPath, "machine hash unavailable: "+err.Error())
			writeROCmServeError(w, http.StatusInternalServerError, "machine hash unavailable: "+err.Error(), "machine")
			return
		}
		if confirmation != expected {
			auditROCmServeReload(cfg.AdminAudit, "deny", r, fromPath, auditTarget, req.AdapterPath, "confirm_machine mismatch")
			writeROCmServeError(w, http.StatusBadRequest, "confirm_machine mismatch", "confirm_machine")
			return
		}
		modelPath, err := resolveROCmServeReloadModelPath(cfg, req)
		if err != nil {
			auditROCmServeReload(cfg.AdminAudit, "deny", r, fromPath, auditTarget, req.AdapterPath, err.Error())
			writeROCmServeError(w, http.StatusBadRequest, err.Error(), "model")
			return
		}
		contextOverride := firstServePositiveInt(req.ContextLength, req.ContextLen)
		contextLen := contextOverride
		if contextLen == 0 {
			contextLen = cfg.ContextLen
		}
		profilePath := strings.TrimSpace(req.ProfilePath)
		adapterPath := strings.TrimSpace(req.AdapterPath)
		rocmLoadCfg := cfg.ROCmLoadConfig
		loadOptions := append([]inference.LoadOption(nil), cfg.LoadOptions...)
		statusConfig := rocmServeStatusConfig{}
		if profilePath != "" {
			profile, found, err := loadROCmServeReloadProfile(profilePath)
			if err != nil {
				auditROCmServeReload(cfg.AdminAudit, "deny", r, fromPath, auditTarget, adapterPath, err.Error())
				writeROCmServeError(w, http.StatusBadRequest, err.Error(), "profile_path")
				return
			}
			if found {
				rocmLoadCfg, loadOptions = rocm.TuningCandidateLoadConfig(profile.Candidate)
				statusConfig = rocmServeStatusConfigFromTuningCandidate(profile.Candidate, contextOverride, adapterPath)
				if contextLen == 0 {
					contextLen = profile.Candidate.ContextLength
				}
				adapterPath = firstServeNonEmptyString(adapterPath, profile.Candidate.Adapter.Path)
			}
		}
		if contextLen > 0 {
			loadOptions = append(loadOptions, inference.WithContextLen(contextLen))
		}
		if adapterPath != "" {
			loadOptions = append(loadOptions, inference.WithAdapterPath(adapterPath))
		}
		if statusConfig == (rocmServeStatusConfig{}) {
			statusConfig = rocmServeStatusConfigFromLoad(rocmLoadCfg, loadOptions, contextLen, adapterPath)
		} else {
			statusConfig.ContextLength = firstServePositiveInt(contextLen, statusConfig.ContextLength)
			statusConfig.AdapterPath = firstServeNonEmptyString(adapterPath, statusConfig.AdapterPath)
		}
		resolver.ReloadWithOptions(rocmServeReloadOptions{
			Path:           modelPath,
			Backend:        firstServeNonEmptyString(req.Backend, cfg.Backend),
			ContextLen:     contextLen,
			ProfilePath:    profilePath,
			AdapterPath:    adapterPath,
			ROCmLoadConfig: rocmLoadCfg,
			LoadOptions:    loadOptions,
			StatusConfig:   statusConfig,
		})
		auditROCmServeReload(cfg.AdminAudit, "success", r, fromPath, modelPath, adapterPath, "")
		writeROCmServeJSON(w, http.StatusOK, rocmServeStatusResponse(cfg, resolver.Status()))
	})))
	mux.Handle(rocmServeAdminDownloadPath, rocmServeAdminAuth(cfg, rocmAdminDownloadHandler(downloadRegistry)))
	mux.Handle(rocmServeAdminSFTStartPath, rocmServeAdminAuth(cfg, rocmAdminSFTStartHandler(sftRegistry, cfg)))
	mux.Handle(rocmServeAdminSFTStatusPath, rocmServeAdminAuth(cfg, rocmAdminSFTStatusHandler(sftRegistry)))
	mux.Handle(rocmServeAdminSFTStopPath, rocmServeAdminAuth(cfg, rocmAdminSFTStopHandler(sftRegistry)))
	mux.Handle(rocmServeAdminSFTAdaptersPath, rocmServeAdminAuth(cfg, rocmAdminSFTAdaptersHandler(cfg.AdminSFTAdapterRoot)))
	return mux, resolver
}

type rocmServeReloadRequest struct {
	Model          string `json:"model,omitempty"`
	ModelPath      string `json:"model_path,omitempty"`
	Path           string `json:"path,omitempty"`
	Backend        string `json:"backend,omitempty"`
	ContextLen     int    `json:"context,omitempty"`
	ContextLength  int    `json:"context_length,omitempty"`
	Confirmation   string `json:"confirmation,omitempty"`
	ConfirmMachine string `json:"confirm_machine,omitempty"`
	ProfilePath    string `json:"profile_path,omitempty"`
	AdapterPath    string `json:"adapter_path,omitempty"`
}

func resolveROCmServeReloadModelPath(cfg rocmServeConfig, req rocmServeReloadRequest) (string, error) {
	root := rocmAdminDownloadModelDir(rocmAdminDownloadConfig{ModelDir: cfg.AdminDownloadModelDir})
	modelPath := strings.TrimSpace(firstServeNonEmptyString(req.ModelPath, req.Path))
	if modelPath != "" {
		return bindROCmServeReloadModelPath(root, modelPath)
	}
	return resolveROCmServeReloadModelName(root, strings.TrimSpace(req.Model))
}

func resolveROCmServeReloadModelName(root, name string) (string, error) {
	if name == "" {
		return "", errors.New("model name required")
	}
	if strings.Contains(name, "/") || strings.Contains(name, `\`) || strings.Contains(name, "..") || strings.HasPrefix(name, ".") {
		return "", errors.New("model name must be a basename (no /, no .., no leading .)")
	}
	return bindROCmServeReloadModelPath(root, filepath.Join(root, name))
}

func bindROCmServeReloadModelPath(root, modelPath string) (string, error) {
	if strings.TrimSpace(modelPath) == "" {
		return "", errors.New("model_path required")
	}
	if !filepath.IsAbs(modelPath) {
		return "", errors.New("model_path must be absolute")
	}
	rootResolved, err := rocmServeEvalDir(root)
	if err != nil {
		return "", fmt.Errorf("models dir not found: %s", root)
	}
	resolved, err := rocmServeEvalDir(modelPath)
	if err != nil {
		return "", fmt.Errorf("model dir not found: %s", modelPath)
	}
	if !rocmServePathWithinDir(rootResolved, resolved) {
		return "", errors.New("model path escapes models dir")
	}
	manifestPath := filepath.Join(resolved, rocmServeSHAManifestFilename)
	info, err := os.Stat(manifestPath)
	if err != nil || info.IsDir() {
		return "", fmt.Errorf("model has no sha manifest: %s", modelPath)
	}
	return resolved, nil
}

func rocmServeEvalDir(path string) (string, error) {
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		return "", fmt.Errorf("%s is not a directory", path)
	}
	return filepath.Clean(resolved), nil
}

func rocmServePathWithinDir(rootResolved, resolved string) bool {
	if rootResolved == "" || resolved == "" {
		return false
	}
	if resolved == rootResolved {
		return true
	}
	rel, err := filepath.Rel(rootResolved, resolved)
	if err != nil {
		return false
	}
	if rel == "" || rel == "." {
		return true
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) || filepath.IsAbs(rel) {
		return false
	}
	return true
}

func auditROCmServeReload(w io.Writer, outcome string, r *http.Request, fromPath, toPath, adapterPath, reason string) {
	if w == nil {
		return
	}
	remote := ""
	if r != nil {
		remote = r.RemoteAddr
	}
	if strings.TrimSpace(reason) != "" {
		fmt.Fprintf(w, "%s admin: serve_reload %s requester=%s from=%s to=%s adapter=%s reason=%s\n",
			cliName(), outcome, remote, fromPath, toPath, adapterPath, reason)
		return
	}
	fmt.Fprintf(w, "%s admin: serve_reload %s requester=%s from=%s to=%s adapter=%s\n",
		cliName(), outcome, remote, fromPath, toPath, adapterPath)
}

type rocmServeCacheEntryLister interface {
	CacheEntries(ctx context.Context, labels map[string]string) ([]inference.CacheBlockRef, error)
}

type rocmServeCacheEntriesResponse struct {
	Object  string                    `json:"object"`
	Model   string                    `json:"model,omitempty"`
	Entries []inference.CacheBlockRef `json:"entries"`
	Stats   *inference.CacheStats     `json:"stats,omitempty"`
}

func rocmServeRuntimeActionResponse(action string, cfg rocmServeConfig, status rocmServeResolverStatus) map[string]any {
	return map[string]any{
		"action":   action,
		"status":   "ok",
		"runtime":  defaultBackendName,
		"resolver": status,
		"labels":   rocmServeLabelsForStatus(cfg, status),
	}
}

func buildROCmServeAdminMachineResponse(ctx context.Context, cfg rocmServeConfig) (rocmServeAdminMachineResponse, error) {
	hash, err := rocm.CurrentMachineProfileHash(ctx)
	if err != nil {
		return rocmServeAdminMachineResponse{}, err
	}
	hostname, _ := os.Hostname()
	return rocmServeAdminMachineResponse{
		Hash:        hash,
		Hostname:    hostname,
		Runtime:     defaultBackendName,
		GoVersion:   runtime.Version(),
		OS:          runtime.GOOS,
		Arch:        runtime.GOARCH,
		Time:        time.Now().Unix(),
		CLIContract: cliContractName,
		Compile:     currentCompileReport(),
		Labels:      rocmServeLabels(cfg),
	}, nil
}

func rocmServeCacheEntryLabels(r *http.Request) map[string]string {
	labels := map[string]string{}
	if r == nil || r.URL == nil {
		return labels
	}
	for key, values := range r.URL.Query() {
		if key == "model" || len(values) == 0 {
			continue
		}
		value := strings.TrimSpace(values[0])
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}

func rocmServeStatusResponse(cfg rocmServeConfig, status rocmServeResolverStatus) rocmServeAdminStatusResponse {
	detection := status.DraftDetection
	if !detection.Active() {
		detection = resolveROCmDraft(firstServeNonEmptyString(status.ModelPath, cfg.ModelPath), cfg.DraftPath, cfg.DraftDetect)
	}
	modelPath := firstServeNonEmptyString(status.ModelPath, cfg.ModelPath)
	loadedAtUnix := int64(0)
	if !status.LoadedAt.IsZero() {
		loadedAtUnix = status.LoadedAt.Unix()
	}
	statusConfig := rocmServeHydrateStatusConfig(cfg, status)
	modelProfile := status.ModelProfile
	if modelProfile == nil {
		modelProfile = rocmServeResolveModelProfile(nil, modelPath)
	}
	modelRoutes := status.ModelRoutes
	if modelRoutes == nil {
		modelRoutes = rocmModelRoutePlanForProfile(modelProfile)
	}
	tokenLoop := status.TokenLoop
	if tokenLoop == nil {
		tokenLoop = rocmTokenLoopForProfile(modelProfile)
	}
	return rocmServeAdminStatusResponse{
		Status:           "ok",
		Runtime:          defaultBackendName,
		CLIContract:      cliContractName,
		ModelPath:        modelPath,
		ProfilePath:      status.ProfilePath,
		AdapterPath:      status.AdapterPath,
		LoadedAtUnix:     loadedAtUnix,
		ModelRegistry:    currentModelRegistryReport(firstServeNonEmptyString(status.Backend, cfg.Backend, defaultBackendName)),
		ModelProfile:     modelProfile,
		ModelRoutes:      modelRoutes,
		TokenLoop:        tokenLoop,
		Config:           statusConfig,
		Memory:           status.Memory,
		Resolver:         status,
		DraftDetection:   detection,
		DraftBlock:       status.DraftBlock,
		DraftBlockSource: status.DraftBlockSource,
		Labels:           rocmServeLabelsForStatus(cfg, status),
	}
}

func rocmServeHydrateStatusConfig(cfg rocmServeConfig, status rocmServeResolverStatus) rocmServeStatusConfig {
	statusConfig := status.statusConfig
	if statusConfig == (rocmServeStatusConfig{}) {
		statusConfig = rocmServeStatusConfigFromLoad(cfg.ROCmLoadConfig, cfg.LoadOptions, firstServePositiveInt(status.ContextLen, cfg.ContextLen), status.AdapterPath)
	}
	statusConfig.ContextLength = firstServePositiveInt(status.ContextLen, statusConfig.ContextLength, cfg.ContextLen)
	statusConfig.KVCache = firstServeNonEmptyString(statusConfig.KVCache, status.KVCacheMode, cfg.KVCacheMode)
	statusConfig.CacheMode = firstServeNonEmptyString(statusConfig.CacheMode, status.KVCacheMode, cfg.ROCmLoadConfig.CacheMode, cfg.KVCacheMode)
	statusConfig.DeviceKVMode = firstServeNonEmptyString(statusConfig.DeviceKVMode, cfg.ROCmLoadConfig.DeviceKVMode)
	statusConfig.AdapterPath = firstServeNonEmptyString(status.AdapterPath, statusConfig.AdapterPath)
	statusConfig.Draft = cfg.DraftPath
	statusConfig.DraftDetect = cfg.DraftDetect
	statusConfig.DraftBlock = cfg.DraftBlock
	statusConfig.NoAutoProfile = cfg.NoAutoProfile
	statusConfig.ProfileDir = cfg.ProfileDir
	statusConfig.StateConversations = cfg.StateConversations
	statusConfig.StateStore = cfg.StateStorePath
	statusConfig.StateContinuity = rocmServeStateContinuityStatus(cfg, status)
	statusConfig.StateReason = firstServeNonEmptyString(status.StateReason, cfg.StateContinuityReason)
	return statusConfig
}

func rocmServeStatusConfigFromLoad(rocmCfg rocm.ROCmLoadConfig, opts []inference.LoadOption, contextLen int, adapterPath string) rocmServeStatusConfig {
	loadCfg := inference.ApplyLoadOpts(opts)
	cacheMode := firstServeNonEmptyString(rocmCfg.CacheMode, rocmCfg.DeviceKVMode)
	return rocmServeStatusConfig{
		ContextLength: firstServePositiveInt(contextLen, loadCfg.ContextLen),
		ParallelSlots: loadCfg.ParallelSlots,
		KVCache:       cacheMode,
		CacheMode:     cacheMode,
		DeviceKVMode:  firstServeNonEmptyString(rocmCfg.DeviceKVMode, rocmCfg.CacheMode),
		AdapterPath:   firstServeNonEmptyString(adapterPath, loadCfg.AdapterPath),
	}
}

func rocmServeLoadConfigEmpty(cfg rocm.ROCmLoadConfig) bool {
	return strings.TrimSpace(cfg.CacheMode) == "" && strings.TrimSpace(cfg.DeviceKVMode) == ""
}

func rocmServeStatusConfigFromTuningCandidate(candidate inference.TuningCandidate, contextOverride int, adapterOverride string) rocmServeStatusConfig {
	contextLength := candidate.ContextLength
	if contextOverride > 0 {
		contextLength = contextOverride
	}
	adapterPath := firstServeNonEmptyString(adapterOverride, candidate.Adapter.Path)
	cacheMode := firstServeNonEmptyString(candidate.CacheMode, candidate.Runtime.CacheMode)
	return rocmServeStatusConfig{
		ContextLength:        contextLength,
		ParallelSlots:        candidate.ParallelSlots,
		PromptCache:          candidate.PromptCache,
		PromptCacheMinTokens: candidate.PromptCacheMinTokens,
		CachePolicy:          candidate.CachePolicy,
		KVCache:              cacheMode,
		CacheMode:            cacheMode,
		DeviceKVMode:         cacheMode,
		BatchSize:            candidate.BatchSize,
		PrefillChunkSize:     candidate.PrefillChunkSize,
		ExpectedQuantization: candidate.ExpectedQuantization,
		MemoryLimitBytes:     candidate.MemoryLimitBytes,
		CacheLimitBytes:      candidate.CacheLimitBytes,
		WiredLimitBytes:      candidate.WiredLimitBytes,
		AdapterPath:          adapterPath,
	}
}

func rocmServeLabels(cfg rocmServeConfig) map[string]string {
	return rocmServeLabelsForModel(cfg, cfg.ModelPath)
}

func rocmServeLabelsForStatus(cfg rocmServeConfig, status rocmServeResolverStatus) map[string]string {
	labels := rocmServeLabelsForModel(cfg, firstServeNonEmptyString(status.ModelPath, cfg.ModelPath))
	profile := status.ModelProfile
	if profile == nil {
		profile = rocmServeResolveModelProfile(nil, firstServeNonEmptyString(status.ModelPath, cfg.ModelPath))
	}
	if profile != nil && profile.Matched() {
		labels = rocm.ApplyROCmModelProfileLabels(labels, *profile)
	}
	labels = rocmServeApplyModelRoutePlanLabels(labels, profile, status.ModelRoutes)
	labels = applyROCmTokenLoopLabels(labels, rocmServeTokenLoopStatus(status))
	detection := status.DraftDetection
	if !detection.Active() {
		detection = resolveROCmDraft(firstServeNonEmptyString(status.ModelPath, cfg.ModelPath), cfg.DraftPath, cfg.DraftDetect)
	}
	labels = rocmServeApplyOpenAIStateMTPLabels(labels, cfg, status, detection)
	return labels
}

func rocmServeApplyModelRoutePlanLabels(labels map[string]string, profile *rocm.ROCmModelProfile, routes *rocm.ROCmModelRoutePlan) map[string]string {
	if routes != nil && routes.Matched() {
		return rocm.ApplyROCmModelRoutePlanLabels(labels, *routes)
	}
	if profile != nil && profile.Matched() {
		return rocm.ApplyROCmModelRoutePlanLabels(labels, rocm.ROCmModelRoutePlanForProfile(*profile))
	}
	return labels
}

func rocmServeResolveModelProfile(model inference.TextModel, path string) *rocm.ROCmModelProfile {
	if model != nil {
		if profile, ok := rocm.ResolveROCmModelProfileForModel(model); ok && profile.Matched() {
			return &profile
		}
	}
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}
	identity := inference.ModelIdentity{Path: path}
	if probe, err := rocm.ProbeROCmModelConfigFile(filepath.Join(path, "config.json")); err == nil && probe.ArchitectureResolution.Matched() {
		identity.Architecture = probe.ArchitectureResolution.Architecture
		identity.Labels = probe.Labels
	}
	if profile, ok := rocm.ResolveROCmModelProfile(path, identity); ok && profile.Matched() {
		return &profile
	}
	return nil
}

func rocmServeLabelsForModel(cfg rocmServeConfig, modelPath string) map[string]string {
	labels := map[string]string{
		"backend":                      defaultBackendName,
		"cli_contract":                 cliContractName,
		"openai_chat_completions":      "ready",
		"openai_responses":             "streaming",
		"anthropic_messages":           "ready",
		"ollama_chat_generate":         "streaming",
		"lem_score_pair":               "ready",
		"admin_auth":                   "bearer_constant_time",
		"admin_auth_audit":             "deny",
		"admin_token_prefix":           rocmAdminTokenPrefix,
		"admin_reload":                 "confirmation_sha_manifest_gated",
		"admin_reload_confirm_machine": "accepted",
		"admin_reload_adapter_path":    "load_option",
		"admin_reload_profile_path":    "candidate_load_config_optional",
		"admin_model_download":         "ready",
		"admin_sft":                    "inference_sft_trainer",
		"cache_entries":                "ready",
		"reactive_state_continuity":    rocmServeStateContinuityLabel(rocmServeStateContinuityStatus(cfg, rocmServeResolverStatus{})),
		"production_requires_env_gate": "false",
		"production_requires_cli_flag": "false",
	}
	detection := cfg.DraftDetection
	if strings.TrimSpace(modelPath) != strings.TrimSpace(cfg.ModelPath) || !detection.Active() {
		detection = resolveROCmDraft(modelPath, cfg.DraftPath, cfg.DraftDetect)
	}
	draftBlock, draftBlockSource := resolveROCmServeDraftBlock(context.Background(), detection, modelPath, cfg.DraftBlock, cfg.NoAutoProfile, cfg.ProfileDir)
	labels = applyROCmDraftDetectionLabels(labels, detection, draftBlock)
	labels = rocmServeApplyOpenAIStateMTPLabels(labels, cfg, rocmServeResolverStatus{}, detection)
	if strings.TrimSpace(draftBlockSource) != "" {
		labels["reactive_draft_block_source"] = draftBlockSource
	}
	if strings.TrimSpace(cfg.KVCacheMode) != "" {
		labels["kv_cache"] = strings.TrimSpace(cfg.KVCacheMode)
	}
	return labels
}

func rocmServeApplyOpenAIStateMTPLabels(labels map[string]string, cfg rocmServeConfig, status rocmServeResolverStatus, detection rocm.DraftDetection) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	stateStatus := rocmServeStateContinuityStatus(cfg, status)
	nativeAttachment := strings.TrimSpace(status.NativeMTPAttachment)
	labels["reactive_state_continuity"] = rocmServeStateContinuityLabel(stateStatus)
	labels["openai_chat_state_continuity"] = rocmServeStateContinuityLabel(stateStatus)
	labels["openai_chat_retained_state_mtp"] = rocmServeOpenAIStateMTPStatus(stateStatus, detection.Active(), nativeAttachment)
	if nativeAttachment != "" {
		labels["native_mtp_attachment"] = nativeAttachment
		labels["openai_chat_retained_state_mtp_attachment"] = nativeAttachment
	}
	if !detection.Active() {
		return labels
	}
	labels["openai_chat_retained_state_mtp_required"] = "true"
	labels["openai_chat_retained_state_source"] = "rocm_serve_conversation_store"
	labels["openai_chat_retained_state_runtime_state"] = "rocm_state_session_runtime_kv"
	labels["openai_chat_retained_state_entrypoint"] = "attached_drafter_textmodel_generate_native_from_state"
	labels["openai_chat_retained_state_prompt_replay_fallback"] = "forbidden"
	return labels
}

func rocmServeNativeMTPAttachmentStatus(model inference.TextModel) string {
	if model == nil {
		return ""
	}
	reporter, ok := model.(interface {
		ModelIdentity() inference.ModelIdentity
	})
	if !ok || reporter == nil {
		return ""
	}
	labels := reporter.ModelIdentity().Labels
	for _, key := range []string{
		"attached_drafter_native_attachment",
		"engine_attached_drafter_native_attachment",
		"native_mtp_attachment",
	} {
		if value := strings.TrimSpace(labels[key]); value != "" {
			return value
		}
	}
	return ""
}

func rocmServeModelsResponse(cfg rocmServeConfig, status rocmServeResolverStatus) map[string]any {
	data := []map[string]any{}
	if status.ModelName != "" {
		created := time.Now().Unix()
		if !status.LoadedAt.IsZero() {
			created = status.LoadedAt.Unix()
		}
		model := map[string]any{
			"id":       status.ModelName,
			"object":   "model",
			"created":  created,
			"owned_by": defaultBackendName,
			"metadata": rocmServeModelMetadata(status),
			"labels":   rocmServeLabelsForStatus(cfg, status),
		}
		if status.ModelProfile != nil && status.ModelProfile.Matched() {
			model["model_profile"] = status.ModelProfile
		}
		if status.ModelRoutes != nil && status.ModelRoutes.Matched() {
			model["model_routes"] = status.ModelRoutes
		}
		if tokenLoop := rocmServeTokenLoopStatus(status); tokenLoop != nil {
			model["token_loop"] = tokenLoop
		}
		if retained := rocmServeRetainedStateStatus(status); retained != nil {
			model["retained_state"] = retained
		}
		data = append(data, model)
	}
	return map[string]any{"object": "list", "data": data}
}

func rocmServeModelMetadata(status rocmServeResolverStatus) map[string]any {
	metadata := map[string]any{
		"backend":          firstServeNonEmptyString(status.Backend, defaultBackendName),
		"model_path":       status.ModelPath,
		"loaded":           status.Loaded,
		"state_continuity": rocmServeStateContinuityLabel(status.StateContinuity),
	}
	if status.ContextLen > 0 {
		metadata["context"] = status.ContextLen
	}
	if strings.TrimSpace(status.KVCacheMode) != "" {
		metadata["kv_cache"] = status.KVCacheMode
	}
	if status.DraftDetection.Active() {
		metadata["draft_detection"] = status.DraftDetection
	}
	if status.DraftBlock > 0 {
		metadata["draft_block"] = status.DraftBlock
	}
	if strings.TrimSpace(status.DraftBlockSource) != "" {
		metadata["draft_block_source"] = status.DraftBlockSource
	}
	if strings.TrimSpace(status.NativeMTPAttachment) != "" {
		metadata["native_mtp_attachment"] = status.NativeMTPAttachment
	}
	return metadata
}

func rocmServeRetainedStateStatus(status rocmServeResolverStatus) *rocm.ROCmRetainedStateStatus {
	if status.ModelProfile == nil || !status.ModelProfile.Matched() {
		return nil
	}
	identity := status.ModelProfile.Model
	if identity.Path == "" {
		identity.Path = firstServeNonEmptyString(status.ModelPath, identity.Path)
	}
	if identity.Architecture == "" {
		identity.Architecture = firstServeNonEmptyString(status.ModelProfile.Architecture, status.ModelProfile.ArchitectureProfile.ID)
	}
	retained, ok := rocm.ROCmRetainedStateForIdentity(identity.Path, identity)
	if !ok {
		return nil
	}
	return &retained
}

func rocmServeTokenLoopStatus(status rocmServeResolverStatus) *rocm.ROCmTokenLoopStatus {
	if status.TokenLoop != nil && status.TokenLoop.Matched() {
		tokenLoop := status.TokenLoop.Clone()
		return &tokenLoop
	}
	if status.ModelProfile == nil || !status.ModelProfile.Matched() {
		return nil
	}
	return rocmTokenLoopForProfile(status.ModelProfile)
}

func rocmServeRequireMethod(w http.ResponseWriter, r *http.Request, method string) bool {
	if r == nil {
		writeROCmServeError(w, http.StatusBadRequest, "request is nil", "request")
		return false
	}
	if r.Method != method {
		w.Header().Set("Allow", method)
		writeROCmServeError(w, http.StatusMethodNotAllowed, "method not allowed", "method")
		return false
	}
	return true
}

func rocmServeAdminAuth(cfg rocmServeConfig, next http.Handler) http.Handler {
	token := strings.TrimSpace(cfg.AdminToken)
	expected := []byte("Bearer " + token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if token != "" {
			got := []byte(r.Header.Get("Authorization"))
			if len(got) != len(expected) || subtle.ConstantTimeCompare(got, expected) != 1 {
				auditROCmServeAdminAuthDeny(cfg.AdminAudit, r)
				w.Header().Set("WWW-Authenticate", rocmAdminAuthRealm)
				writeROCmServeError(w, http.StatusUnauthorized, "admin endpoint requires Authorization: Bearer <token>; missing or invalid admin bearer token", "authorization")
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

func auditROCmServeAdminAuthDeny(w io.Writer, r *http.Request) {
	if w == nil {
		return
	}
	path := ""
	remote := ""
	if r != nil {
		remote = r.RemoteAddr
		if r.URL != nil {
			path = r.URL.Path
		}
	}
	fmt.Fprintf(w, "%s admin: auth deny path=%s remote=%s\n", cliName(), path, remote)
}

func writeROCmServeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeROCmServeError(w http.ResponseWriter, status int, message, param string) {
	writeROCmServeJSON(w, status, map[string]any{
		"error": map[string]string{
			"message": message,
			"type":    "invalid_request_error",
			"param":   param,
		},
	})
}

func defaultAdminTokenPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, "Lethean", "data", "admin.token"), nil
}

func ensureAdminTokenFile(path string) (string, bool, error) {
	data, err := os.ReadFile(path)
	if err == nil {
		token := strings.TrimSpace(string(data))
		if token != "" {
			return token, false, nil
		}
	}
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", false, err
	}
	token, err := generateAdminToken()
	if err != nil {
		return "", false, err
	}
	if err := writeAdminTokenFile(path, token); err != nil {
		return "", false, err
	}
	after, err := os.ReadFile(path)
	if err == nil {
		afterToken := strings.TrimSpace(string(after))
		if afterToken != "" && afterToken != token {
			return afterToken, false, nil
		}
	}
	return token, true, nil
}

func rotateAdminTokenFile(path string) (string, error) {
	token, err := generateAdminToken()
	if err != nil {
		return "", err
	}
	return token, writeAdminTokenFile(path, token)
}

func writeAdminTokenFile(path, token string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(token+"\n"), 0o600)
}

func generateAdminToken() (string, error) {
	var raw [32]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return "", err
	}
	return rocmAdminTokenPrefix + base64.RawURLEncoding.EncodeToString(raw[:]), nil
}

func rocmServeModelName(path string) string {
	name := strings.TrimSpace(filepath.Base(strings.TrimSpace(path)))
	if name == "" || name == "." || name == string(filepath.Separator) {
		return ""
	}
	ext := filepath.Ext(name)
	if ext != "" {
		name = strings.TrimSuffix(name, ext)
	}
	return name
}

func ctxErr(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return nil
	}
}

func firstServeNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func firstServePositiveInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}
