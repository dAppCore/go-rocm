// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"sync/atomic"

	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
)

const (
	rocmServeStateContinuityDisabled                = "disabled"
	rocmServeStateContinuityStoreReady              = "store_ready"
	rocmServeStateContinuityConversationStore       = "conversation_store"
	rocmServeStateContinuityStoreUnavailable        = "store_unavailable"
	rocmServeStateContinuityModelStateless          = "store_unavailable_model_stateless"
	rocmServeStateContinuityWakeSleepRoutesFallback = "wake_sleep_routes"

	rocmServeOpenAIStateMTPNotApplicable = "not_applicable"
	rocmServeOpenAIStateMTPLinked        = "linked"
	rocmServeOpenAIStateMTPPendingNative = "pending_native_drafter"
	rocmServeOpenAIStateMTPDisabled      = "disabled"
	rocmServeOpenAIStateMTPUnavailable   = "state_unavailable"
)

type rocmServeConversationContinuityStats struct {
	FreshConversations int64 `json:"fresh_conversations"`
	StoreWakes         int64 `json:"store_wakes"`
	Sleeps             int64 `json:"sleeps"`
	StatelessFallbacks int64 `json:"stateless_fallbacks"`
	WakeFailures       int64 `json:"wake_failures"`
	SleepFailures      int64 `json:"sleep_failures"`
}

type rocmServeConversationContinuityManager struct {
	enabled bool
	store   state.Store

	mu     sync.Mutex
	status string
	reason string

	freshConversations atomic.Int64
	storeWakes         atomic.Int64
	sleeps             atomic.Int64
	statelessFallbacks atomic.Int64
	wakeFailures       atomic.Int64
	sleepFailures      atomic.Int64
}

func newROCmServeConversationContinuityManager(cfg rocmServeConfig) *rocmServeConversationContinuityManager {
	manager := &rocmServeConversationContinuityManager{
		enabled: cfg.StateConversations,
		store:   cfg.StateStore,
		status:  rocmServeStateContinuityDisabled,
	}
	if !cfg.StateConversations {
		return manager
	}
	manager.status = rocmServeStateContinuityStoreReady
	if cfg.StateStore == nil {
		manager.status = rocmServeStateContinuityStoreUnavailable
		manager.reason = firstServeNonEmptyString(cfg.StateContinuityReason, "state store is not configured")
		return manager
	}
	if _, ok := cfg.StateStore.(state.URIResolver); !ok {
		manager.status = rocmServeStateContinuityStoreUnavailable
		manager.reason = "state store does not resolve URI-backed conversation indexes"
		return manager
	}
	if _, ok := cfg.StateStore.(state.Writer); !ok {
		manager.status = rocmServeStateContinuityStoreUnavailable
		manager.reason = "state store does not support conversation state writes"
	}
	return manager
}

func (manager *rocmServeConversationContinuityManager) Wrap(model inference.TextModel) inference.TextModel {
	if manager == nil || model == nil || !manager.enabled {
		return model
	}
	if manager.store == nil {
		manager.setStatus(rocmServeStateContinuityStoreUnavailable, "state store is not configured")
		return model
	}
	session, ok := model.(inference.AgentMemorySession)
	if !ok || session == nil {
		manager.setStatus(rocmServeStateContinuityModelStateless, fmt.Sprintf("model %T does not implement AgentMemorySession", model))
		return model
	}
	manager.setStatus(rocmServeStateContinuityConversationStore, "")
	return &rocmServeConversationContinuity{
		model:   model,
		session: session,
		store:   manager.store,
		manager: manager,
	}
}

func (manager *rocmServeConversationContinuityManager) Status() (string, string, rocmServeConversationContinuityStats) {
	if manager == nil {
		return rocmServeStateContinuityDisabled, "", rocmServeConversationContinuityStats{}
	}
	manager.mu.Lock()
	status := manager.status
	reason := manager.reason
	manager.mu.Unlock()
	if status == "" {
		status = rocmServeStateContinuityDisabled
	}
	return status, reason, manager.stats()
}

func (manager *rocmServeConversationContinuityManager) setStatus(status, reason string) {
	if manager == nil {
		return
	}
	manager.mu.Lock()
	manager.status = status
	manager.reason = reason
	manager.mu.Unlock()
}

func (manager *rocmServeConversationContinuityManager) stats() rocmServeConversationContinuityStats {
	if manager == nil {
		return rocmServeConversationContinuityStats{}
	}
	return rocmServeConversationContinuityStats{
		FreshConversations: manager.freshConversations.Load(),
		StoreWakes:         manager.storeWakes.Load(),
		Sleeps:             manager.sleeps.Load(),
		StatelessFallbacks: manager.statelessFallbacks.Load(),
		WakeFailures:       manager.wakeFailures.Load(),
		SleepFailures:      manager.sleepFailures.Load(),
	}
}

type rocmServeGenerationResolver struct {
	base    *rocmServeResolver
	manager *rocmServeConversationContinuityManager
}

func newROCmServeGenerationResolver(base *rocmServeResolver) *rocmServeGenerationResolver {
	if base == nil {
		return &rocmServeGenerationResolver{}
	}
	return &rocmServeGenerationResolver{base: base, manager: base.stateContinuity}
}

func (resolver *rocmServeGenerationResolver) ResolveModel(ctx context.Context, name string) (inference.TextModel, error) {
	if resolver == nil || resolver.base == nil {
		return nil, errors.New("serve resolver is nil")
	}
	model, err := resolver.base.ResolveModel(ctx, name)
	if err != nil {
		return nil, err
	}
	if resolver.manager == nil {
		return model, nil
	}
	return resolver.manager.Wrap(model), nil
}

func (resolver *rocmServeGenerationResolver) OllamaModelNames(ctx context.Context) ([]string, error) {
	if resolver == nil || resolver.base == nil {
		return nil, nil
	}
	return resolver.base.OllamaModelNames(ctx)
}

type rocmServeConversationContinuity struct {
	model   inference.TextModel
	session inference.AgentMemorySession
	store   state.Store
	manager *rocmServeConversationContinuityManager
}

func (continuity *rocmServeConversationContinuity) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return continuity.model.Generate(ctx, prompt, opts...)
}

func (continuity *rocmServeConversationContinuity) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		if continuity == nil || continuity.model == nil || continuity.session == nil || continuity.store == nil {
			return
		}
		chatMessages, wake, ok := continuity.prepare(ctx, messages)
		if !ok {
			for token := range continuity.model.Chat(ctx, messages, opts...) {
				if !yield(token) {
					return
				}
			}
			return
		}
		var reply strings.Builder
		completed := true
		for token := range continuity.model.Chat(ctx, chatMessages, opts...) {
			reply.WriteString(token.Text)
			if !yield(token) {
				completed = false
				break
			}
		}
		if !completed || continuity.model.Err() != nil || ctxErr(ctx) != nil {
			return
		}
		continuity.sleep(ctx, messages, reply.String(), wake)
	}
}

func (continuity *rocmServeConversationContinuity) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return continuity.model.Classify(ctx, prompts, opts...)
}

func (continuity *rocmServeConversationContinuity) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return continuity.model.BatchGenerate(ctx, prompts, opts...)
}

func (continuity *rocmServeConversationContinuity) ModelType() string {
	return continuity.model.ModelType()
}

func (continuity *rocmServeConversationContinuity) Info() inference.ModelInfo {
	return continuity.model.Info()
}

func (continuity *rocmServeConversationContinuity) Metrics() inference.GenerateMetrics {
	return continuity.model.Metrics()
}

func (continuity *rocmServeConversationContinuity) Err() error {
	return continuity.model.Err()
}

func (continuity *rocmServeConversationContinuity) Close() error {
	return continuity.model.Close()
}

func (continuity *rocmServeConversationContinuity) WakeState(ctx context.Context, req inference.AgentMemoryWakeRequest) (*inference.AgentMemoryWakeResult, error) {
	return continuity.session.WakeState(ctx, req)
}

func (continuity *rocmServeConversationContinuity) SleepState(ctx context.Context, req inference.AgentMemorySleepRequest) (*inference.AgentMemorySleepResult, error) {
	return continuity.session.SleepState(ctx, req)
}

func (continuity *rocmServeConversationContinuity) prepare(ctx context.Context, messages []inference.Message) ([]inference.Message, *inference.AgentMemoryWakeResult, bool) {
	if len(messages) == 0 {
		continuity.manager.statelessFallbacks.Add(1)
		return nil, nil, false
	}
	tailStart := rocmServeConversationTailStart(messages)
	if tailStart >= len(messages) {
		continuity.manager.statelessFallbacks.Add(1)
		return nil, nil, false
	}
	if tailStart == 0 {
		continuity.manager.freshConversations.Add(1)
		return messages, nil, true
	}
	entryURI, indexURI := rocmServeConversationStateURIs(messages[:tailStart])
	found, err := rocmServeStateIndexAvailable(ctx, continuity.store, indexURI)
	if err != nil {
		continuity.manager.statelessFallbacks.Add(1)
		return nil, nil, false
	}
	if !found {
		continuity.manager.freshConversations.Add(1)
		return messages, nil, true
	}
	wake, err := continuity.session.WakeState(ctx, inference.AgentMemoryWakeRequest{
		Store:    continuity.store,
		EntryURI: entryURI,
		IndexURI: indexURI,
		Labels: map[string]string{
			"cli":      cliName(),
			"contract": cliContractName,
			"runtime":  defaultBackendName,
			"surface":  "serve",
		},
	})
	if err != nil {
		continuity.manager.wakeFailures.Add(1)
		continuity.manager.statelessFallbacks.Add(1)
		return nil, nil, false
	}
	continuity.manager.storeWakes.Add(1)
	return messages[tailStart:], wake, true
}

func (continuity *rocmServeConversationContinuity) sleep(ctx context.Context, messages []inference.Message, reply string, wake *inference.AgentMemoryWakeResult) {
	full := make([]inference.Message, 0, len(messages)+1)
	full = append(full, messages...)
	full = append(full, inference.Message{Role: "assistant", Content: reply})
	entryURI, indexURI := rocmServeConversationStateURIs(full)
	_, err := continuity.session.SleepState(ctx, inference.AgentMemorySleepRequest{
		Store:             continuity.store,
		EntryURI:          entryURI,
		IndexURI:          indexURI,
		ParentEntryURI:    parentEntryURI(wake),
		ParentBundleURI:   parentBundleURI(wake),
		ParentIndexURI:    parentIndexURI(wake),
		Title:             "serve conversation " + strings.TrimPrefix(entryURI, "rocm://serve/conversation/")[:12],
		ReuseParentPrefix: wake != nil,
		Labels: map[string]string{
			"cli":      cliName(),
			"contract": cliContractName,
			"runtime":  defaultBackendName,
			"surface":  "serve",
		},
	})
	if err != nil {
		continuity.manager.sleepFailures.Add(1)
		return
	}
	continuity.manager.sleeps.Add(1)
}

func rocmServeConversationTailStart(messages []inference.Message) int {
	for index := len(messages) - 1; index >= 0; index-- {
		switch strings.ToLower(strings.TrimSpace(messages[index].Role)) {
		case "user", "tool":
			continue
		default:
			return index + 1
		}
	}
	return 0
}

func rocmServeConversationStateURIs(messages []inference.Message) (string, string) {
	key := rocmServeConversationStateKey(messages)
	entryURI := "rocm://serve/conversation/" + key
	return entryURI, entryURI + "/index"
}

func rocmServeConversationStateKey(messages []inference.Message) string {
	hash := sha256.New()
	for _, message := range messages {
		hash.Write([]byte(strings.ToLower(strings.TrimSpace(message.Role))))
		hash.Write([]byte{0})
		hash.Write([]byte(message.Content))
		hash.Write([]byte{1})
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func rocmServeStateIndexAvailable(ctx context.Context, store state.Store, indexURI string) (bool, error) {
	_, err := state.ResolveURI(ctx, store, indexURI)
	if err == nil {
		return true, nil
	}
	var notFound *state.URIChunkNotFoundError
	if errors.As(err, &notFound) || errors.Is(err, state.ErrChunkNotFound) {
		return false, nil
	}
	return false, err
}

func rocmServeStateContinuityStatus(cfg rocmServeConfig, status rocmServeResolverStatus) string {
	if strings.TrimSpace(status.StateContinuity) != "" {
		return status.StateContinuity
	}
	if !cfg.StateConversations {
		return rocmServeStateContinuityDisabled
	}
	if cfg.StateStore == nil {
		return rocmServeStateContinuityStoreUnavailable
	}
	return rocmServeStateContinuityStoreReady
}

func rocmServeStateContinuityLabel(status string) string {
	switch strings.TrimSpace(status) {
	case rocmServeStateContinuityConversationStore:
		return rocmServeStateContinuityConversationStore
	case rocmServeStateContinuityStoreReady:
		return rocmServeStateContinuityStoreReady
	case rocmServeStateContinuityStoreUnavailable, rocmServeStateContinuityModelStateless:
		return "unavailable_stateless"
	case rocmServeStateContinuityDisabled:
		return rocmServeStateContinuityDisabled
	default:
		return rocmServeStateContinuityWakeSleepRoutesFallback
	}
}

func rocmServeStateContinuitySupportsRetainedMTP(status string) bool {
	switch strings.TrimSpace(status) {
	case rocmServeStateContinuityConversationStore, rocmServeStateContinuityStoreReady:
		return true
	default:
		return false
	}
}

func rocmServeOpenAIStateMTPStatus(stateStatus string, draftActive bool, nativeAttachment string) string {
	if !draftActive {
		return rocmServeOpenAIStateMTPNotApplicable
	}
	if rocmServeStateContinuitySupportsRetainedMTP(stateStatus) {
		if strings.TrimSpace(nativeAttachment) == rocmServeOpenAIStateMTPLinked {
			return rocmServeOpenAIStateMTPLinked
		}
		return rocmServeOpenAIStateMTPPendingNative
	}
	if strings.TrimSpace(stateStatus) == rocmServeStateContinuityDisabled {
		return rocmServeOpenAIStateMTPDisabled
	}
	return rocmServeOpenAIStateMTPUnavailable
}
