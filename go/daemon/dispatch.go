// SPDX-Licence-Identifier: EUPL-1.2

// Package daemon exposes the local JSON-lines daemon contract used by Core Go
// inference backends. It is intentionally small: clients can discover actions,
// call generation, and share the same request shape across ROCm, MLX, CUDA, and
// CPU binaries.
package daemon

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/rocm/score"
)

const (
	DaemonName     = "lthn"
	DefaultVersion = "dev"
)

var (
	errRegistryNil              = errors.New("registry is nil")
	errActionRequired           = errors.New("action is required")
	errCacheBackendNil          = errors.New("cache backend is nil")
	errCacheEntryBackendNil     = errors.New("cache entry backend is nil")
	errCancelBackendNil         = errors.New("cancel backend is nil")
	errCancelIDRequired         = errors.New("cancel id is required")
	errEmbedBackendNil          = errors.New("embed backend is nil")
	errEmbedInputRequired       = errors.New("embed input text is required")
	errEngineFeaturesBackendNil = errors.New("engine features backend is nil")
	errGenerateBackendNil       = errors.New("generate backend is nil")
	errIntrospectionBackendNil  = errors.New("introspection backend is nil")
	errModelRegistryBackendNil  = errors.New("model registry backend is nil")
	errModelRoutesBackendNil    = errors.New("model routes backend is nil")
	errModelProfileBackendNil   = errors.New("model profile backend is nil")
	errModelPackBackendNil      = errors.New("model pack backend is nil")
	errModelPackPathRequired    = errors.New("model pack path is required")
	errParserBackendNil         = errors.New("parser backend is nil")
	errParserTextRequired       = errors.New("parser text or tokens are required")
	errRerankBackendNil         = errors.New("rerank backend is nil")
	errRerankQueryRequired      = errors.New("rerank query is required")
	errRerankDocumentsRequired  = errors.New("rerank documents are required")
	errScheduleBackendNil       = errors.New("schedule backend is nil")
	errScheduleInputRequired    = errors.New("schedule prompt or messages are required")
	errScheduleStreamNil        = errors.New("schedule token stream is nil")
	errScoreBackendNil          = errors.New("score backend is nil")
	errScoreTextRequired        = errors.New("score text, prompt, or response is required")
	errTokenizerBackendNil      = errors.New("tokenizer backend is nil")
	errTokenizerTextRequired    = errors.New("tokenizer text is required")
	errTokenizerTokensRequired  = errors.New("tokenizer tokens are required")
	errChatMessagesRequired     = errors.New("chat template messages are required")
)

// Request is one JSON-line frame from a local daemon client.
type Request struct {
	Action         string            `json:"action"`
	Text           string            `json:"text,omitempty"`
	Input          []string          `json:"input,omitempty"`
	Path           string            `json:"path,omitempty"`
	Prompt         string            `json:"prompt,omitempty"`
	Query          string            `json:"query,omitempty"`
	Response       string            `json:"response,omitempty"`
	Documents      []string          `json:"documents,omitempty"`
	ID             string            `json:"id,omitempty"`
	Model          string            `json:"model,omitempty"`
	Backend        string            `json:"backend,omitempty"`
	Tokens         []int32           `json:"tokens,omitempty"`
	Mode           string            `json:"mode,omitempty"`
	Normalize      bool              `json:"normalize,omitempty"`
	TopN           int               `json:"top_n,omitempty"`
	Labels         map[string]string `json:"labels,omitempty"`
	Messages       []Message         `json:"messages,omitempty"`
	MaxTokens      int               `json:"max_tokens,omitempty"`
	Temperature    float64           `json:"temperature,omitempty"`
	TopK           int               `json:"top_k,omitempty"`
	TopP           float64           `json:"top_p,omitempty"`
	MinP           float64           `json:"min_p,omitempty"`
	StopTokens     []int32           `json:"stop_tokens,omitempty"`
	StopSequences  []string          `json:"stop_sequences,omitempty"`
	RepeatPenalty  float64           `json:"repeat_penalty,omitempty"`
	ReturnLogits   bool              `json:"return_logits,omitempty"`
	EnableThinking *bool             `json:"enable_thinking,omitempty"`
}

// Response is encoded as one complete JSON-line frame.
type Response map[string]any

// RouteInfo is the discoverable status for one daemon action.
type RouteInfo struct {
	Action  string `json:"action"`
	Status  string `json:"status"`
	Backend string `json:"backend,omitempty"`
}

// Message is a chat message sent to the native generate backend.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// GenerateRequest is the normalized input passed to a generate backend.
type GenerateRequest struct {
	Prompt         string
	Model          string
	Messages       []Message
	MaxTokens      int
	Temperature    float64
	TopK           int
	TopP           float64
	MinP           float64
	StopTokens     []int32
	RepeatPenalty  float64
	ReturnLogits   bool
	EnableThinking *bool
}

// GenerateResult is returned by native generation backends.
type GenerateResult struct {
	Text    string
	Model   string
	Metrics GenerateMetrics
}

// GenerateMetrics are JSON-friendly counters from a backend generation call.
type GenerateMetrics struct {
	PromptTokens        int     `json:"prompt_tokens"`
	GeneratedTokens     int     `json:"generated_tokens"`
	PrefillSeconds      float64 `json:"prefill_seconds,omitempty"`
	DecodeSeconds       float64 `json:"decode_seconds,omitempty"`
	TotalSeconds        float64 `json:"total_seconds,omitempty"`
	PrefillTokensPerSec float64 `json:"prefill_tokens_per_sec,omitempty"`
	DecodeTokensPerSec  float64 `json:"decode_tokens_per_sec,omitempty"`
	PeakMemoryBytes     uint64  `json:"peak_memory_bytes,omitempty"`
	ActiveMemoryBytes   uint64  `json:"active_memory_bytes,omitempty"`
}

// ScoreRequest is the normalized input passed to the daemon score route.
type ScoreRequest struct {
	Text     string
	Prompt   string
	Response string
	Model    string
}

// CacheRequest is the normalized control request passed to daemon cache routes.
type CacheRequest struct {
	Model  string
	Labels map[string]string
}

// CacheEntriesResult is returned by daemon cache entry backends.
type CacheEntriesResult struct {
	Entries []inference.CacheBlockRef `json:"entries"`
	Stats   *inference.CacheStats     `json:"stats,omitempty"`
}

// CancelRequest is the normalized control request passed to daemon cancel routes.
type CancelRequest struct {
	Model string
	ID    string
}

// ModelRequest is the normalized model selection request passed to daemon
// model introspection routes.
type ModelRequest struct {
	Model string
}

// EngineFeaturesRequest is the normalized input passed to daemon engine-feature routes.
type EngineFeaturesRequest struct {
	Path    string
	Model   string
	Backend string
	Labels  map[string]string
}

// ModelRecord describes one configured model alias in the local daemon.
type ModelRecord struct {
	ID      string `json:"id"`
	Path    string `json:"path,omitempty"`
	Default bool   `json:"default,omitempty"`
	Loaded  bool   `json:"loaded,omitempty"`
}

// ModelPackRequest is the normalized input passed to daemon model-pack routes.
type ModelPackRequest struct {
	Path string
}

// ModelProfileRequest is the normalized input passed to daemon model-profile routes.
type ModelProfileRequest struct {
	Path    string
	Model   string
	Backend string
	Labels  map[string]string
}

// ModelRoutesRequest is the normalized input passed to daemon model-route plan routes.
type ModelRoutesRequest struct {
	Path    string
	Model   string
	Backend string
	Labels  map[string]string
}

// ModelRegistryRequest is the normalized input passed to daemon model-registry routes.
type ModelRegistryRequest struct {
	Backend string
	Labels  map[string]string
}

// TokenizeRequest is the normalized input passed to daemon tokenization routes.
type TokenizeRequest struct {
	Model string
	Text  string
}

// TokenizeResult is returned by daemon tokenization backends.
type TokenizeResult struct {
	Model  string  `json:"model,omitempty"`
	Tokens []int32 `json:"tokens"`
}

// DetokenizeRequest is the normalized input passed to daemon detokenization routes.
type DetokenizeRequest struct {
	Model  string
	Tokens []int32
}

// DetokenizeResult is returned by daemon detokenization backends.
type DetokenizeResult struct {
	Model string `json:"model,omitempty"`
	Text  string `json:"text"`
}

// ChatTemplateRequest is the normalized input passed to daemon chat-template routes.
type ChatTemplateRequest struct {
	Model    string
	Messages []Message
}

// ChatTemplateResult is returned by daemon chat-template backends.
type ChatTemplateResult struct {
	Model string `json:"model,omitempty"`
	Text  string `json:"text"`
}

// ParseRequest is the normalized input passed to daemon parser routes.
type ParseRequest struct {
	Model  string
	Text   string
	Tokens []int32
}

// ScoreResult is returned by daemon score backends.
type ScoreResult struct {
	Kind  string             `json:"kind"`
	Score *score.ScoreResult `json:"score,omitempty"`
	Pair  *score.DiffResult  `json:"pair,omitempty"`
}

// Handler processes one daemon action request.
type Handler func(context.Context, Request) (Response, error)

// CacheBackend handles native non-HTTP cache controls.
type CacheBackend interface {
	CacheStats(context.Context, CacheRequest) (inference.CacheStats, error)
	WarmCache(context.Context, inference.CacheWarmRequest) (inference.CacheWarmResult, error)
	ClearCache(context.Context, CacheRequest) (inference.CacheStats, error)
}

// CacheEntryBackend handles native non-HTTP cache entry listing.
type CacheEntryBackend interface {
	CacheEntries(context.Context, CacheRequest) (CacheEntriesResult, error)
}

// CacheEntryLister is implemented by loaded models that can list cache entries.
type CacheEntryLister interface {
	CacheEntries(context.Context, map[string]string) ([]inference.CacheBlockRef, error)
}

// CancelBackend handles native non-HTTP request cancellation.
type CancelBackend interface {
	Cancel(context.Context, CancelRequest) (inference.RequestCancelResult, error)
}

// EmbedBackend handles native non-HTTP embedding requests.
type EmbedBackend interface {
	Embed(context.Context, inference.EmbeddingRequest) (*inference.EmbeddingResult, error)
}

// EngineFeaturesBackend handles native non-HTTP engine-feature metadata lookup.
type EngineFeaturesBackend interface {
	EngineFeatures(context.Context, EngineFeaturesRequest) (any, error)
}

// GenerateBackend handles native non-HTTP generation requests.
type GenerateBackend interface {
	Generate(context.Context, GenerateRequest) (GenerateResult, error)
}

// IntrospectionBackend handles local model and capability discovery.
type IntrospectionBackend interface {
	Models(context.Context) ([]ModelRecord, error)
	ModelInfo(context.Context, ModelRequest) (inference.ModelInfo, error)
	Capabilities(context.Context, ModelRequest) (inference.CapabilityReport, error)
}

// ModelPackBackend handles native non-HTTP model-pack metadata inspection.
type ModelPackBackend interface {
	InspectModelPack(context.Context, ModelPackRequest) (*inference.ModelPackInspection, error)
}

// ModelProfileBackend handles native non-HTTP model-profile lookup.
type ModelProfileBackend interface {
	ModelProfile(context.Context, ModelProfileRequest) (any, error)
}

// ModelRoutesBackend handles native non-HTTP model-route plan lookup.
type ModelRoutesBackend interface {
	ModelRoutes(context.Context, ModelRoutesRequest) (any, error)
}

// ModelRegistryBackend handles native non-HTTP model-registry metadata.
type ModelRegistryBackend interface {
	ModelRegistry(context.Context, ModelRegistryRequest) (any, error)
}

// TokenizerBackend handles native non-HTTP tokenization and chat templates.
type TokenizerBackend interface {
	Tokenize(context.Context, TokenizeRequest) (TokenizeResult, error)
	Detokenize(context.Context, DetokenizeRequest) (DetokenizeResult, error)
	ApplyChatTemplate(context.Context, ChatTemplateRequest) (ChatTemplateResult, error)
}

// ParserBackend handles native non-HTTP reasoning and tool-call parsing.
type ParserBackend interface {
	ParseReasoning(context.Context, ParseRequest) (inference.ReasoningParseResult, error)
	ParseTools(context.Context, ParseRequest) (inference.ToolParseResult, error)
}

// RerankBackend handles native non-HTTP reranking requests.
type RerankBackend interface {
	Rerank(context.Context, inference.RerankRequest) (*inference.RerankResult, error)
}

// ScheduleBackend handles queue-aware generation requests.
type ScheduleBackend interface {
	Schedule(context.Context, inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error)
}

// ScoreBackend handles local content scoring requests.
type ScoreBackend interface {
	Score(context.Context, ScoreRequest) (ScoreResult, error)
}

// HeuristicScoreBackend exposes the ROCm lexical scorer through the daemon.
type HeuristicScoreBackend struct{}

func (HeuristicScoreBackend) Score(ctx context.Context, req ScoreRequest) (ScoreResult, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return ScoreResult{}, err
		}
	}
	prompt := strings.TrimSpace(req.Prompt)
	response := strings.TrimSpace(req.Response)
	if prompt != "" && response != "" {
		pair := score.ScorePair(req.Prompt, req.Response)
		return ScoreResult{Kind: "pair", Pair: &pair}, nil
	}
	text := firstScoreText(req.Text, req.Response, req.Prompt)
	if text == "" {
		return ScoreResult{}, errScoreTextRequired
	}
	result := score.Score(text)
	return ScoreResult{Kind: "text", Score: &result}, nil
}

// Registry maps daemon actions to handlers. It preserves registration order so
// the info response is stable and human-readable.
type Registry struct {
	name         string
	version      string
	handlers     map[string]Handler
	order        []string
	routes       map[string]RouteInfo
	infoResponse Response
}

func NewRegistry(name, version string) *Registry {
	if name == "" {
		name = DaemonName
	}
	if version == "" {
		version = DefaultVersion
	}
	r := &Registry{
		name:     name,
		version:  version,
		handlers: make(map[string]Handler, 4),
		order:    make([]string, 0, 4),
		routes:   make(map[string]RouteInfo, 4),
	}
	mustRegisterRoute(r, "embed", stubHandler("embed"), "stub", "")
	mustRegisterRoute(r, "score", scoreHandler(HeuristicScoreBackend{}), "heuristic", "rocm/score")
	mustRegisterRoute(r, "generate", stubHandler("generate"), "stub", "")
	mustRegisterRoute(r, "schedule", stubHandler("schedule"), "stub", "")
	mustRegisterRoute(r, "rerank", stubHandler("rerank"), "stub", "")
	mustRegisterRoute(r, "cache_stats", stubHandler("cache_stats"), "stub", "")
	mustRegisterRoute(r, "cache_warm", stubHandler("cache_warm"), "stub", "")
	mustRegisterRoute(r, "cache_clear", stubHandler("cache_clear"), "stub", "")
	mustRegisterRoute(r, "cache_entries", stubHandler("cache_entries"), "stub", "")
	mustRegisterRoute(r, "cancel", stubHandler("cancel"), "stub", "")
	mustRegisterRoute(r, "health", healthHandler(r), "system", "daemon")
	mustRegisterRoute(r, "models", stubHandler("models"), "stub", "")
	mustRegisterRoute(r, "model_info", stubHandler("model_info"), "stub", "")
	mustRegisterRoute(r, "capabilities", stubHandler("capabilities"), "stub", "")
	mustRegisterRoute(r, "engine_features", stubHandler("engine_features"), "stub", "")
	mustRegisterRoute(r, "inspect_model_pack", stubHandler("inspect_model_pack"), "stub", "")
	mustRegisterRoute(r, "model_profile", stubHandler("model_profile"), "stub", "")
	mustRegisterRoute(r, "model_routes", stubHandler("model_routes"), "stub", "")
	mustRegisterRoute(r, "registry", stubHandler("registry"), "stub", "")
	mustRegisterRoute(r, "tokenize", stubHandler("tokenize"), "stub", "")
	mustRegisterRoute(r, "detokenize", stubHandler("detokenize"), "stub", "")
	mustRegisterRoute(r, "chat_template", stubHandler("chat_template"), "stub", "")
	mustRegisterRoute(r, "parse_reasoning", stubHandler("parse_reasoning"), "stub", "")
	mustRegisterRoute(r, "parse_tools", stubHandler("parse_tools"), "stub", "")
	mustRegisterRoute(r, "info", func(context.Context, Request) (Response, error) {
		if r.infoResponse == nil {
			r.infoResponse = Response{
				"name":    r.name,
				"version": r.version,
				"actions": r.order,
				"routes":  r.Routes(),
			}
		}
		return r.infoResponse, nil
	}, "system", "daemon")
	return r
}

func DefaultRegistryForDaemon() *Registry {
	return NewRegistry(DaemonName, DefaultVersion)
}

func (r *Registry) Register(action string, handler Handler) error {
	return r.register(action, handler, "custom", "")
}

func (r *Registry) register(action string, handler Handler, status, backend string) error {
	action = normalizeAction(action)
	if action == "" {
		return errActionRequired
	}
	if handler == nil {
		return fmt.Errorf("handler for action %q is nil", action)
	}
	if r.handlers == nil {
		r.handlers = make(map[string]Handler)
	}
	if r.routes == nil {
		r.routes = make(map[string]RouteInfo)
	}
	if _, exists := r.handlers[action]; !exists {
		r.order = append(r.order, action)
	}
	r.handlers[action] = handler
	r.routes[action] = RouteInfo{Action: action, Status: status, Backend: backend}
	r.infoResponse = nil
	return nil
}

// RegisterCacheBackend replaces the default cache stubs with a native backend.
func (r *Registry) RegisterCacheBackend(backend CacheBackend) error {
	if backend == nil {
		return errCacheBackendNil
	}
	if err := r.register("cache_stats", cacheStatsHandler(backend), "native_model", "inference.CacheService"); err != nil {
		return err
	}
	if err := r.register("cache_warm", cacheWarmHandler(backend), "native_model", "inference.CacheService"); err != nil {
		return err
	}
	return r.register("cache_clear", cacheClearHandler(backend), "native_model", "inference.CacheService")
}

// RegisterCacheEntryBackend replaces the default cache entry stub with a native backend.
func (r *Registry) RegisterCacheEntryBackend(backend CacheEntryBackend) error {
	if backend == nil {
		return errCacheEntryBackendNil
	}
	return r.register("cache_entries", cacheEntriesHandler(backend), "native_model", "daemon.CacheEntryBackend")
}

// RegisterCancelBackend replaces the default cancel stub with a native backend.
func (r *Registry) RegisterCancelBackend(backend CancelBackend) error {
	if backend == nil {
		return errCancelBackendNil
	}
	return r.register("cancel", cancelHandler(backend), "native_model", "inference.CancellableModel")
}

// RegisterEmbedBackend replaces the default embed stub with a native backend.
func (r *Registry) RegisterEmbedBackend(backend EmbedBackend) error {
	if backend == nil {
		return errEmbedBackendNil
	}
	return r.register("embed", embedHandler(backend), "native_model", "inference.EmbeddingModel")
}

// RegisterEngineFeaturesBackend replaces the default engine-feature metadata stub.
func (r *Registry) RegisterEngineFeaturesBackend(backend EngineFeaturesBackend) error {
	if backend == nil {
		return errEngineFeaturesBackendNil
	}
	return r.register("engine_features", engineFeaturesHandler(backend), "native_metadata", "daemon.EngineFeaturesBackend")
}

// RegisterGenerateBackend replaces the default generate stub with a native backend.
func (r *Registry) RegisterGenerateBackend(backend GenerateBackend) error {
	if backend == nil {
		return errGenerateBackendNil
	}
	return r.register("generate", func(ctx context.Context, req Request) (Response, error) {
		result, err := backend.Generate(ctx, generateRequestFromRequest(req))
		if err != nil {
			return nil, err
		}
		return generateResponseFromResult(result), nil
	}, "native_model", "daemon.GenerateBackend")
}

// RegisterIntrospectionBackend replaces model discovery stubs with a native backend.
func (r *Registry) RegisterIntrospectionBackend(backend IntrospectionBackend) error {
	if backend == nil {
		return errIntrospectionBackendNil
	}
	if err := r.register("models", modelsHandler(backend), "native_model", "daemon.IntrospectionBackend"); err != nil {
		return err
	}
	if err := r.register("model_info", modelInfoHandler(backend), "native_model", "daemon.IntrospectionBackend"); err != nil {
		return err
	}
	return r.register("capabilities", capabilitiesHandler(backend), "native_model", "daemon.IntrospectionBackend")
}

// RegisterModelPackBackend replaces the default model-pack inspection stub.
func (r *Registry) RegisterModelPackBackend(backend ModelPackBackend) error {
	if backend == nil {
		return errModelPackBackendNil
	}
	return r.register("inspect_model_pack", inspectModelPackHandler(backend), "native_metadata", "inference.ModelPackInspector")
}

// RegisterModelProfileBackend replaces the default model-profile metadata stub.
func (r *Registry) RegisterModelProfileBackend(backend ModelProfileBackend) error {
	if backend == nil {
		return errModelProfileBackendNil
	}
	return r.register("model_profile", modelProfileHandler(backend), "native_metadata", "daemon.ModelProfileBackend")
}

// RegisterModelRoutesBackend replaces the default model-route plan metadata stub.
func (r *Registry) RegisterModelRoutesBackend(backend ModelRoutesBackend) error {
	if backend == nil {
		return errModelRoutesBackendNil
	}
	return r.register("model_routes", modelRoutesHandler(backend), "native_metadata", "daemon.ModelRoutesBackend")
}

// RegisterModelRegistryBackend replaces the default model-registry metadata stub.
func (r *Registry) RegisterModelRegistryBackend(backend ModelRegistryBackend) error {
	if backend == nil {
		return errModelRegistryBackendNil
	}
	return r.register("registry", modelRegistryHandler(backend), "native_metadata", "daemon.ModelRegistryBackend")
}

// RegisterRerankBackend replaces the default rerank stub with a native backend.
func (r *Registry) RegisterRerankBackend(backend RerankBackend) error {
	if backend == nil {
		return errRerankBackendNil
	}
	return r.register("rerank", rerankHandler(backend), "native_model", "inference.RerankModel")
}

// RegisterScheduleBackend replaces the default schedule stub with a native backend.
func (r *Registry) RegisterScheduleBackend(backend ScheduleBackend) error {
	if backend == nil {
		return errScheduleBackendNil
	}
	return r.register("schedule", scheduleHandler(backend), "native_model", "inference.SchedulerModel")
}

// RegisterTokenizerBackend replaces the default tokenizer stubs with a native backend.
func (r *Registry) RegisterTokenizerBackend(backend TokenizerBackend) error {
	if backend == nil {
		return errTokenizerBackendNil
	}
	if err := r.register("tokenize", tokenizeHandler(backend), "native_model", "inference.TokenizerModel"); err != nil {
		return err
	}
	if err := r.register("detokenize", detokenizeHandler(backend), "native_model", "inference.TokenizerModel"); err != nil {
		return err
	}
	return r.register("chat_template", chatTemplateHandler(backend), "native_model", "inference.TokenizerModel")
}

// RegisterParserBackend replaces the default parser stubs with a native backend.
func (r *Registry) RegisterParserBackend(backend ParserBackend) error {
	if backend == nil {
		return errParserBackendNil
	}
	if err := r.register("parse_reasoning", parseReasoningHandler(backend), "native_model", "inference.ReasoningParser"); err != nil {
		return err
	}
	return r.register("parse_tools", parseToolsHandler(backend), "native_model", "inference.ToolParser")
}

// RegisterScoreBackend replaces the default heuristic score route.
func (r *Registry) RegisterScoreBackend(backend ScoreBackend) error {
	if backend == nil {
		return errScoreBackendNil
	}
	return r.register("score", scoreHandler(backend), "custom", "daemon.ScoreBackend")
}

func (r *Registry) Dispatch(ctx context.Context, req Request) (Response, error) {
	if r == nil {
		return nil, errRegistryNil
	}
	action := normalizeAction(req.Action)
	if action == "" {
		return nil, errActionRequired
	}
	handler, ok := r.handlers[action]
	if !ok {
		return nil, fmt.Errorf("unsupported action %q", action)
	}
	req.Action = action
	return handler(ctx, req)
}

func (r *Registry) Actions() []string {
	if r == nil {
		return nil
	}
	actions := make([]string, len(r.order))
	copy(actions, r.order)
	return actions
}

func (r *Registry) Routes() []RouteInfo {
	if r == nil {
		return nil
	}
	routes := make([]RouteInfo, 0, len(r.order))
	for _, action := range r.order {
		route := r.routes[action]
		if route.Action == "" {
			route.Action = action
		}
		routes = append(routes, route)
	}
	return routes
}

func healthHandler(r *Registry) Handler {
	return func(context.Context, Request) (Response, error) {
		return Response{
			"status":  "ok",
			"action":  "health",
			"name":    r.name,
			"version": r.version,
			"actions": r.Actions(),
			"routes":  r.Routes(),
		}, nil
	}
}

func cacheStatsHandler(backend CacheBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		stats, err := backend.CacheStats(ctx, cacheRequestFromRequest(req))
		if err != nil {
			return nil, err
		}
		return Response{"status": "ok", "action": "cache_stats", "stats": stats}, nil
	}
}

func cacheWarmHandler(backend CacheBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		result, err := backend.WarmCache(ctx, cacheWarmRequestFromRequest(req))
		if err != nil {
			return nil, err
		}
		return Response{"status": "ok", "action": "cache_warm", "result": result}, nil
	}
}

func cacheClearHandler(backend CacheBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		stats, err := backend.ClearCache(ctx, cacheRequestFromRequest(req))
		if err != nil {
			return nil, err
		}
		return Response{"status": "ok", "action": "cache_clear", "stats": stats}, nil
	}
}

func cacheEntriesHandler(backend CacheEntryBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		result, err := backend.CacheEntries(ctx, cacheRequestFromRequest(req))
		if err != nil {
			return nil, err
		}
		resp := Response{"status": "ok", "action": "cache_entries", "entries": result.Entries}
		if result.Stats != nil {
			resp["stats"] = *result.Stats
		}
		return resp, nil
	}
}

func cancelHandler(backend CancelBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		cancelReq, err := cancelRequestFromRequest(req)
		if err != nil {
			return nil, err
		}
		result, err := backend.Cancel(ctx, cancelReq)
		if err != nil {
			return nil, err
		}
		return Response{"status": "ok", "action": "cancel", "result": result}, nil
	}
}

func embedHandler(backend EmbedBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		embedReq, err := embedRequestFromRequest(req)
		if err != nil {
			return nil, err
		}
		result, err := backend.Embed(ctx, embedReq)
		if err != nil {
			return nil, err
		}
		return embedResponseFromResult(result)
	}
}

func engineFeaturesHandler(backend EngineFeaturesBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		features, err := backend.EngineFeatures(ctx, engineFeaturesRequestFromRequest(req))
		if err != nil {
			return nil, err
		}
		if features == nil {
			return nil, errors.New("engine features are nil")
		}
		resp := Response{"status": "ok", "action": "engine_features", "features": features}
		if path := strings.TrimSpace(req.Path); path != "" {
			resp["path"] = path
		}
		if model := strings.TrimSpace(req.Model); model != "" {
			resp["model"] = model
		}
		if backendName := strings.TrimSpace(req.Backend); backendName != "" {
			resp["backend"] = backendName
		}
		return resp, nil
	}
}

func modelsHandler(backend IntrospectionBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		records, err := backend.Models(ctx)
		if err != nil {
			return nil, err
		}
		return Response{"status": "ok", "action": "models", "object": "list", "models": records}, nil
	}
}

func modelInfoHandler(backend IntrospectionBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		info, err := backend.ModelInfo(ctx, modelRequestFromRequest(req))
		if err != nil {
			return nil, err
		}
		resp := Response{"status": "ok", "action": "model_info", "info": info}
		if strings.TrimSpace(req.Model) != "" {
			resp["model"] = strings.TrimSpace(req.Model)
		}
		return resp, nil
	}
}

func inspectModelPackHandler(backend ModelPackBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		modelPackReq, err := modelPackRequestFromRequest(req)
		if err != nil {
			return nil, err
		}
		inspection, err := backend.InspectModelPack(ctx, modelPackReq)
		if err != nil {
			return nil, err
		}
		if inspection == nil {
			return nil, errors.New("model pack inspection is nil")
		}
		resp := Response{"status": "ok", "action": "inspect_model_pack", "inspection": inspection}
		if inspection.Path != "" {
			resp["path"] = inspection.Path
		} else {
			resp["path"] = modelPackReq.Path
		}
		return resp, nil
	}
}

func capabilitiesHandler(backend IntrospectionBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		report, err := backend.Capabilities(ctx, modelRequestFromRequest(req))
		if err != nil {
			return nil, err
		}
		return Response{"status": "ok", "action": "capabilities", "report": report}, nil
	}
}

func modelRegistryHandler(backend ModelRegistryBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		registry, err := backend.ModelRegistry(ctx, modelRegistryRequestFromRequest(req))
		if err != nil {
			return nil, err
		}
		if registry == nil {
			return nil, errors.New("model registry is nil")
		}
		resp := Response{"status": "ok", "action": "registry", "registry": registry}
		if backendName := strings.TrimSpace(req.Backend); backendName != "" {
			resp["backend"] = backendName
		}
		return resp, nil
	}
}

func modelProfileHandler(backend ModelProfileBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		profile, err := backend.ModelProfile(ctx, modelProfileRequestFromRequest(req))
		if err != nil {
			return nil, err
		}
		if profile == nil {
			return nil, errors.New("model profile is nil")
		}
		resp := Response{"status": "ok", "action": "model_profile", "profile": profile}
		if path := strings.TrimSpace(req.Path); path != "" {
			resp["path"] = path
		}
		if model := strings.TrimSpace(req.Model); model != "" {
			resp["model"] = model
		}
		if backendName := strings.TrimSpace(req.Backend); backendName != "" {
			resp["backend"] = backendName
		}
		return resp, nil
	}
}

func modelRoutesHandler(backend ModelRoutesBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		routes, err := backend.ModelRoutes(ctx, modelRoutesRequestFromRequest(req))
		if err != nil {
			return nil, err
		}
		if routes == nil {
			return nil, errors.New("model routes are nil")
		}
		resp := Response{"status": "ok", "action": "model_routes", "routes": routes}
		if path := strings.TrimSpace(req.Path); path != "" {
			resp["path"] = path
		}
		if model := strings.TrimSpace(req.Model); model != "" {
			resp["model"] = model
		}
		if backendName := strings.TrimSpace(req.Backend); backendName != "" {
			resp["backend"] = backendName
		}
		return resp, nil
	}
}

func rerankHandler(backend RerankBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		rerankReq, err := rerankRequestFromRequest(req)
		if err != nil {
			return nil, err
		}
		result, err := backend.Rerank(ctx, rerankReq)
		if err != nil {
			return nil, err
		}
		return rerankResponseFromResult(result)
	}
}

func scheduleHandler(backend ScheduleBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		scheduleReq, err := scheduleRequestFromRequest(req)
		if err != nil {
			return nil, err
		}
		handle, stream, err := backend.Schedule(ctx, scheduleReq)
		if err != nil {
			return nil, err
		}
		if stream == nil {
			return nil, errScheduleStreamNil
		}
		tokens := []inference.ScheduledToken{}
		var builder strings.Builder
		for token := range stream {
			cloned := cloneScheduledToken(token)
			tokens = append(tokens, cloned)
			builder.WriteString(cloned.Token.Text)
		}
		return scheduleResponseFromResult(handle, tokens, builder.String()), nil
	}
}

func scoreHandler(backend ScoreBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		result, err := backend.Score(ctx, scoreRequestFromRequest(req))
		if err != nil {
			return nil, err
		}
		return scoreResponseFromResult(result), nil
	}
}

func tokenizeHandler(backend TokenizerBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		tokenizeReq, err := tokenizeRequestFromRequest(req)
		if err != nil {
			return nil, err
		}
		result, err := backend.Tokenize(ctx, tokenizeReq)
		if err != nil {
			return nil, err
		}
		return tokenizeResponseFromResult(result), nil
	}
}

func detokenizeHandler(backend TokenizerBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		detokenizeReq, err := detokenizeRequestFromRequest(req)
		if err != nil {
			return nil, err
		}
		result, err := backend.Detokenize(ctx, detokenizeReq)
		if err != nil {
			return nil, err
		}
		return detokenizeResponseFromResult(result), nil
	}
}

func chatTemplateHandler(backend TokenizerBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		templateReq, err := chatTemplateRequestFromRequest(req)
		if err != nil {
			return nil, err
		}
		result, err := backend.ApplyChatTemplate(ctx, templateReq)
		if err != nil {
			return nil, err
		}
		return chatTemplateResponseFromResult(result), nil
	}
}

func parseReasoningHandler(backend ParserBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		parseReq, err := parseRequestFromRequest(req)
		if err != nil {
			return nil, err
		}
		result, err := backend.ParseReasoning(ctx, parseReq)
		if err != nil {
			return nil, err
		}
		return reasoningResponseFromResult(parseReq.Model, result), nil
	}
}

func parseToolsHandler(backend ParserBackend) Handler {
	return func(ctx context.Context, req Request) (Response, error) {
		parseReq, err := parseRequestFromRequest(req)
		if err != nil {
			return nil, err
		}
		result, err := backend.ParseTools(ctx, parseReq)
		if err != nil {
			return nil, err
		}
		return toolResponseFromResult(parseReq.Model, result), nil
	}
}

func embedRequestFromRequest(req Request) (inference.EmbeddingRequest, error) {
	input := append([]string(nil), req.Input...)
	if len(input) == 0 {
		if text := firstScoreText(req.Text, req.Prompt); text != "" {
			input = []string{text}
		}
	}
	if len(input) == 0 {
		return inference.EmbeddingRequest{}, errEmbedInputRequired
	}
	return inference.EmbeddingRequest{
		Model:     req.Model,
		Input:     input,
		Normalize: req.Normalize,
		Labels:    cloneLabels(req.Labels),
	}, nil
}

func rerankRequestFromRequest(req Request) (inference.RerankRequest, error) {
	if strings.TrimSpace(req.Query) == "" {
		return inference.RerankRequest{}, errRerankQueryRequired
	}
	documents := append([]string(nil), req.Documents...)
	if len(documents) == 0 {
		return inference.RerankRequest{}, errRerankDocumentsRequired
	}
	return inference.RerankRequest{
		Model:     req.Model,
		Query:     req.Query,
		Documents: documents,
		TopN:      req.TopN,
		Labels:    cloneLabels(req.Labels),
	}, nil
}

func cacheRequestFromRequest(req Request) CacheRequest {
	return CacheRequest{Model: req.Model, Labels: cloneLabels(req.Labels)}
}

func cacheWarmRequestFromRequest(req Request) inference.CacheWarmRequest {
	prompt := req.Prompt
	if prompt == "" {
		prompt = req.Text
	}
	return inference.CacheWarmRequest{
		Model:  inference.ModelIdentity{ID: req.Model},
		Prompt: prompt,
		Tokens: append([]int32(nil), req.Tokens...),
		Mode:   req.Mode,
		Labels: cloneLabels(req.Labels),
	}
}

func cancelRequestFromRequest(req Request) (CancelRequest, error) {
	id := strings.TrimSpace(req.ID)
	if id == "" {
		return CancelRequest{}, errCancelIDRequired
	}
	return CancelRequest{Model: req.Model, ID: id}, nil
}

func modelRequestFromRequest(req Request) ModelRequest {
	return ModelRequest{Model: req.Model}
}

func engineFeaturesRequestFromRequest(req Request) EngineFeaturesRequest {
	return EngineFeaturesRequest{
		Path:    strings.TrimSpace(req.Path),
		Model:   strings.TrimSpace(req.Model),
		Backend: strings.TrimSpace(req.Backend),
		Labels:  cloneLabels(req.Labels),
	}
}

func modelPackRequestFromRequest(req Request) (ModelPackRequest, error) {
	path := strings.TrimSpace(req.Path)
	if path == "" {
		return ModelPackRequest{}, errModelPackPathRequired
	}
	return ModelPackRequest{Path: path}, nil
}

func modelRegistryRequestFromRequest(req Request) ModelRegistryRequest {
	return ModelRegistryRequest{Backend: strings.TrimSpace(req.Backend), Labels: cloneLabels(req.Labels)}
}

func modelProfileRequestFromRequest(req Request) ModelProfileRequest {
	return ModelProfileRequest{
		Path:    strings.TrimSpace(req.Path),
		Model:   strings.TrimSpace(req.Model),
		Backend: strings.TrimSpace(req.Backend),
		Labels:  cloneLabels(req.Labels),
	}
}

func modelRoutesRequestFromRequest(req Request) ModelRoutesRequest {
	return ModelRoutesRequest{
		Path:    strings.TrimSpace(req.Path),
		Model:   strings.TrimSpace(req.Model),
		Backend: strings.TrimSpace(req.Backend),
		Labels:  cloneLabels(req.Labels),
	}
}

func scheduleRequestFromRequest(req Request) (inference.ScheduledRequest, error) {
	prompt := req.Prompt
	if prompt == "" {
		prompt = req.Text
	}
	messages := toInferenceMessages(req.Messages)
	if strings.TrimSpace(prompt) == "" && len(messages) == 0 {
		return inference.ScheduledRequest{}, errScheduleInputRequired
	}
	return inference.ScheduledRequest{
		ID:       strings.TrimSpace(req.ID),
		Model:    req.Model,
		Prompt:   prompt,
		Messages: messages,
		Sampler: inference.SamplerConfig{
			MaxTokens:     req.MaxTokens,
			Temperature:   float32(req.Temperature),
			TopK:          req.TopK,
			TopP:          float32(req.TopP),
			MinP:          float32(req.MinP),
			RepeatPenalty: float32(req.RepeatPenalty),
			StopTokens:    append([]int32(nil), req.StopTokens...),
			StopSequences: append([]string(nil), req.StopSequences...),
			ReturnLogits:  req.ReturnLogits,
		},
		Labels: cloneLabels(req.Labels),
	}, nil
}

func tokenizeRequestFromRequest(req Request) (TokenizeRequest, error) {
	text := firstScoreText(req.Text, req.Prompt, req.Response)
	if strings.TrimSpace(text) == "" {
		return TokenizeRequest{}, errTokenizerTextRequired
	}
	return TokenizeRequest{Model: req.Model, Text: text}, nil
}

func detokenizeRequestFromRequest(req Request) (DetokenizeRequest, error) {
	if len(req.Tokens) == 0 {
		return DetokenizeRequest{}, errTokenizerTokensRequired
	}
	return DetokenizeRequest{Model: req.Model, Tokens: append([]int32(nil), req.Tokens...)}, nil
}

func chatTemplateRequestFromRequest(req Request) (ChatTemplateRequest, error) {
	if len(req.Messages) == 0 {
		return ChatTemplateRequest{}, errChatMessagesRequired
	}
	return ChatTemplateRequest{Model: req.Model, Messages: cloneMessages(req.Messages)}, nil
}

func parseRequestFromRequest(req Request) (ParseRequest, error) {
	text := firstScoreText(req.Text, req.Response, req.Prompt)
	tokens := append([]int32(nil), req.Tokens...)
	if strings.TrimSpace(text) == "" && len(tokens) == 0 {
		return ParseRequest{}, errParserTextRequired
	}
	return ParseRequest{Model: req.Model, Text: text, Tokens: tokens}, nil
}

func generateRequestFromRequest(req Request) GenerateRequest {
	prompt := req.Prompt
	if prompt == "" {
		prompt = req.Text
	}
	return GenerateRequest{
		Prompt:         prompt,
		Model:          req.Model,
		Messages:       req.Messages,
		MaxTokens:      req.MaxTokens,
		Temperature:    req.Temperature,
		TopK:           req.TopK,
		TopP:           req.TopP,
		MinP:           req.MinP,
		StopTokens:     append([]int32(nil), req.StopTokens...),
		RepeatPenalty:  req.RepeatPenalty,
		ReturnLogits:   req.ReturnLogits,
		EnableThinking: req.EnableThinking,
	}
}

func scoreRequestFromRequest(req Request) ScoreRequest {
	return ScoreRequest{
		Text:     req.Text,
		Prompt:   req.Prompt,
		Response: req.Response,
		Model:    req.Model,
	}
}

func embedResponseFromResult(result *inference.EmbeddingResult) (Response, error) {
	if result == nil {
		return nil, errors.New("embed result is nil")
	}
	resp := Response{
		"status":  "ok",
		"action":  "embed",
		"vectors": result.Vectors,
	}
	if hasModelIdentity(result.Model) {
		resp["model"] = result.Model
	}
	if result.Usage != (inference.EmbeddingUsage{}) {
		resp["usage"] = result.Usage
	}
	if len(result.Labels) > 0 {
		resp["labels"] = result.Labels
	}
	return resp, nil
}

func rerankResponseFromResult(result *inference.RerankResult) (Response, error) {
	if result == nil {
		return nil, errors.New("rerank result is nil")
	}
	resp := Response{
		"status":  "ok",
		"action":  "rerank",
		"results": result.Results,
	}
	if hasModelIdentity(result.Model) {
		resp["model"] = result.Model
	}
	if len(result.Labels) > 0 {
		resp["labels"] = result.Labels
	}
	return resp, nil
}

func generateResponseFromResult(result GenerateResult) Response {
	resp := Response{
		"status": "ok",
		"action": "generate",
		"text":   result.Text,
	}
	if result.Model != "" {
		resp["model"] = result.Model
	}
	if hasGenerateMetrics(result.Metrics) {
		resp["metrics"] = result.Metrics
	}
	return resp
}

func scoreResponseFromResult(result ScoreResult) Response {
	resp := Response{
		"status": "ok",
		"action": "score",
		"kind":   result.Kind,
	}
	if result.Score != nil {
		resp["score"] = result.Score
	}
	if result.Pair != nil {
		resp["pair"] = result.Pair
	}
	return resp
}

func scheduleResponseFromResult(handle inference.RequestHandle, tokens []inference.ScheduledToken, text string) Response {
	resp := Response{
		"status": "ok",
		"action": "schedule",
		"handle": handle,
		"text":   text,
		"tokens": tokens,
	}
	if handle.ID != "" {
		resp["id"] = handle.ID
	}
	if hasModelIdentity(handle.Model) {
		resp["model"] = handle.Model
	}
	if len(handle.Labels) > 0 {
		resp["labels"] = handle.Labels
	}
	return resp
}

func tokenizeResponseFromResult(result TokenizeResult) Response {
	resp := Response{
		"status": "ok",
		"action": "tokenize",
		"tokens": append([]int32(nil), result.Tokens...),
	}
	if result.Model != "" {
		resp["model"] = result.Model
	}
	return resp
}

func detokenizeResponseFromResult(result DetokenizeResult) Response {
	resp := Response{
		"status": "ok",
		"action": "detokenize",
		"text":   result.Text,
	}
	if result.Model != "" {
		resp["model"] = result.Model
	}
	return resp
}

func chatTemplateResponseFromResult(result ChatTemplateResult) Response {
	resp := Response{
		"status": "ok",
		"action": "chat_template",
		"text":   result.Text,
	}
	if result.Model != "" {
		resp["model"] = result.Model
	}
	return resp
}

func reasoningResponseFromResult(model string, result inference.ReasoningParseResult) Response {
	resp := Response{
		"status":       "ok",
		"action":       "parse_reasoning",
		"result":       result,
		"visible_text": result.VisibleText,
		"reasoning":    result.Reasoning,
	}
	if model != "" {
		resp["model"] = model
	}
	if len(result.Labels) > 0 {
		resp["labels"] = result.Labels
	}
	return resp
}

func toolResponseFromResult(model string, result inference.ToolParseResult) Response {
	resp := Response{
		"status":       "ok",
		"action":       "parse_tools",
		"result":       result,
		"visible_text": result.VisibleText,
		"calls":        result.Calls,
	}
	if model != "" {
		resp["model"] = model
	}
	if len(result.Labels) > 0 {
		resp["labels"] = result.Labels
	}
	return resp
}

func hasGenerateMetrics(metrics GenerateMetrics) bool {
	return metrics.PromptTokens != 0 ||
		metrics.GeneratedTokens != 0 ||
		metrics.PrefillSeconds != 0 ||
		metrics.DecodeSeconds != 0 ||
		metrics.TotalSeconds != 0 ||
		metrics.PrefillTokensPerSec != 0 ||
		metrics.DecodeTokensPerSec != 0 ||
		metrics.PeakMemoryBytes != 0 ||
		metrics.ActiveMemoryBytes != 0
}

func firstScoreText(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func cloneMessages(messages []Message) []Message {
	if len(messages) == 0 {
		return nil
	}
	out := make([]Message, len(messages))
	copy(out, messages)
	return out
}

func cloneScheduledToken(token inference.ScheduledToken) inference.ScheduledToken {
	token.Labels = cloneLabels(token.Labels)
	return token
}

func cloneLabels(labels map[string]string) map[string]string {
	if len(labels) == 0 {
		return nil
	}
	out := make(map[string]string, len(labels))
	for key, value := range labels {
		out[key] = value
	}
	return out
}

func hasModelIdentity(identity inference.ModelIdentity) bool {
	return identity.ID != "" ||
		identity.Path != "" ||
		identity.Architecture != "" ||
		identity.Revision != "" ||
		identity.Hash != "" ||
		identity.QuantBits != 0 ||
		identity.QuantGroup != 0 ||
		identity.QuantType != "" ||
		identity.ContextLength != 0 ||
		identity.NumLayers != 0 ||
		identity.HiddenSize != 0 ||
		identity.VocabSize != 0 ||
		len(identity.Labels) != 0
}

func normalizeAction(action string) string {
	return strings.ToLower(strings.TrimSpace(action))
}

func mustRegister(r *Registry, action string, handler Handler) {
	if err := r.Register(action, handler); err != nil {
		panic(err)
	}
}

func mustRegisterRoute(r *Registry, action string, handler Handler, status, backend string) {
	if err := r.register(action, handler, status, backend); err != nil {
		panic(err)
	}
}

var (
	stubEmbedResponse    = Response{"status": "stub", "action": "embed"}
	stubScoreResponse    = Response{"status": "stub", "action": "score"}
	stubGenerateResponse = Response{"status": "stub", "action": "generate"}

	stubEmbedHandler    Handler = func(context.Context, Request) (Response, error) { return stubEmbedResponse, nil }
	stubScoreHandler    Handler = func(context.Context, Request) (Response, error) { return stubScoreResponse, nil }
	stubGenerateHandler Handler = func(context.Context, Request) (Response, error) { return stubGenerateResponse, nil }
)

func stubHandler(action string) Handler {
	switch action {
	case "embed":
		return stubEmbedHandler
	case "score":
		return stubScoreHandler
	case "generate":
		return stubGenerateHandler
	case "rerank", "schedule":
		return func(context.Context, Request) (Response, error) {
			return Response{"status": "stub", "action": action}, nil
		}
	case "cache_stats", "cache_warm", "cache_clear", "cache_entries", "cancel", "models", "model_info", "capabilities", "inspect_model_pack", "tokenize", "detokenize", "chat_template", "parse_reasoning", "parse_tools":
		return func(context.Context, Request) (Response, error) {
			return Response{"status": "stub", "action": action}, nil
		}
	default:
		return func(context.Context, Request) (Response, error) {
			return Response{"status": "stub", "action": action}, nil
		}
	}
}

func toDaemonMetrics(metrics inference.GenerateMetrics) GenerateMetrics {
	return GenerateMetrics{
		PromptTokens:        metrics.PromptTokens,
		GeneratedTokens:     metrics.GeneratedTokens,
		PrefillSeconds:      metrics.PrefillDuration.Seconds(),
		DecodeSeconds:       metrics.DecodeDuration.Seconds(),
		TotalSeconds:        metrics.TotalDuration.Seconds(),
		PrefillTokensPerSec: metrics.PrefillTokensPerSec,
		DecodeTokensPerSec:  metrics.DecodeTokensPerSec,
		PeakMemoryBytes:     metrics.PeakMemoryBytes,
		ActiveMemoryBytes:   metrics.ActiveMemoryBytes,
	}
}
