// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/state"
)

const defaultROCmStateBlockSize = 128

// StateSession owns ROCm state lifecycle metadata. Runtime handles remain
// package-local and are not embedded in portable state refs.
type StateSession struct {
	model     inference.ModelIdentity
	tokenizer inference.TokenizerIdentity
	labels    map[string]string
	runtime   any
}

// NewStateSession creates a ROCm state lifecycle wrapper.
func NewStateSession(model inference.ModelIdentity, tokenizer inference.TokenizerIdentity, labels map[string]string) *StateSession {
	return &StateSession{
		model:     cloneModelIdentity(model),
		tokenizer: cloneTokenizerIdentity(tokenizer),
		labels:    mergeStringMaps(map[string]string{"backend": "rocm"}, labels),
	}
}

func newStateSessionWithRuntime(model inference.ModelIdentity, tokenizer inference.TokenizerIdentity, labels map[string]string, runtime any) *StateSession {
	session := NewStateSession(model, tokenizer, labels)
	session.runtime = runtime
	return session
}

func (session *StateSession) Close() error {
	if session == nil {
		return nil
	}
	runtime := session.runtime
	if err := closeROCmStateRuntime(runtime); err != nil {
		return err
	}
	session.runtime = nil
	return nil
}

func (session *StateSession) replaceRuntime(runtime any) error {
	if session == nil {
		return closeROCmStateRuntime(runtime)
	}
	if session.runtime == runtime {
		return nil
	}
	previous := session.runtime
	if err := closeROCmStateRuntime(previous); err != nil {
		return err
	}
	session.runtime = runtime
	return nil
}

func (session *StateSession) WakeState(ctx context.Context, req inference.AgentMemoryWakeRequest) (*inference.AgentMemoryWakeResult, error) {
	if session == nil {
		return nil, core.E("rocm.WakeState", "state session is nil", nil)
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := session.checkWakeCompatibility(req); err != nil {
		return nil, err
	}
	store, ok := req.Store.(state.Store)
	if !ok || store == nil {
		return nil, core.E("rocm.WakeState", "state store is missing", nil)
	}
	uri := firstNonEmptyString(req.EntryURI, req.IndexURI)
	if uri == "" {
		return nil, core.E("rocm.WakeState", "entry or index URI is required", nil)
	}
	chunk, err := state.ResolveURI(ctx, store, uri)
	if err != nil {
		return nil, core.E("rocm.WakeState", "resolve state URI", err)
	}
	labels := mergeStringMaps(session.labels, req.Labels)
	if cache, ok, err := wakeKVCacheFromChunk(chunk); err != nil {
		return nil, err
	} else if ok {
		if err := session.replaceRuntime(cache); err != nil {
			return nil, core.E("rocm.WakeState", "close previous state runtime", err)
		}
		tokens := cache.TokenCount()
		blockSize := cache.blockSize
		blocks := cache.PageCount()
		for key, value := range cache.Stats().Labels {
			labels[key] = value
		}
		labels["kv_restore"] = "runtime_owned"
		labels["kv_device_backing"] = "planned"
		labels["cache_mode"] = cache.mode
		return &inference.AgentMemoryWakeResult{
			Entry:        inference.AgentMemoryRef{URI: uri, IndexURI: req.IndexURI, Kind: "prefix", TokenCount: tokens, Labels: cloneStringMap(labels)},
			Bundle:       inference.StateRef{Kind: "kv", URI: firstNonEmptyString(req.EntryURI, uri), SizeBytes: uint64(len(chunk.Data)), Encoding: rocmKVSnapshotEncoding, Labels: cloneStringMap(labels)},
			Index:        inference.StateRef{Kind: "index", URI: req.IndexURI},
			PrefixTokens: tokens,
			BundleTokens: tokens,
			BlockSize:    blockSize,
			BlocksRead:   blocks,
			Labels:       cloneStringMap(labels),
		}, nil
	}
	tokens := len(approximateTokenIDs(firstNonEmptyString(chunk.Text, string(chunk.Data))))
	if tokens == 0 {
		tokens = req.Model.ContextLength
	}
	blockSize := defaultROCmStateBlockSize
	blocks := blocksForTokens(tokens, blockSize)
	labels["kv_restore"] = "planned"
	return &inference.AgentMemoryWakeResult{
		Entry:        inference.AgentMemoryRef{URI: uri, IndexURI: req.IndexURI, Kind: "prefix", TokenCount: tokens, Labels: cloneStringMap(labels)},
		Bundle:       inference.StateRef{Kind: "kv", URI: firstNonEmptyString(req.EntryURI, uri), Encoding: "rocm/planned", Labels: cloneStringMap(labels)},
		Index:        inference.StateRef{Kind: "index", URI: req.IndexURI},
		PrefixTokens: tokens,
		BundleTokens: tokens,
		BlockSize:    blockSize,
		BlocksRead:   blocks,
		Labels:       cloneStringMap(labels),
	}, nil
}

func (session *StateSession) SleepState(ctx context.Context, req inference.AgentMemorySleepRequest) (*inference.AgentMemorySleepResult, error) {
	if session == nil {
		return nil, core.E("rocm.SleepState", "state session is nil", nil)
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := session.checkSleepCompatibility(req); err != nil {
		return nil, err
	}
	if req.Store == nil {
		return nil, core.E("rocm.SleepState", "state store is missing", nil)
	}
	entryURI := firstNonEmptyString(req.EntryURI, "rocm://state/entry")
	blockSize := req.BlockSize
	if blockSize <= 0 {
		blockSize = defaultROCmStateBlockSize
	}
	labels := mergeStringMaps(session.labels, req.Labels)
	ref, encoding, sizeBytes, tokens, blocks, err := session.sleepStatePayload(ctx, req, entryURI, blockSize, labels)
	if err != nil {
		return nil, err
	}
	refLabels := cloneStringMap(labels)
	if refLabels == nil {
		refLabels = map[string]string{}
	}
	refLabels["chunk_id"] = core.Sprintf("%d", ref.ChunkID)
	return &inference.AgentMemorySleepResult{
		Entry: inference.AgentMemoryRef{
			URI:        entryURI,
			BundleURI:  req.BundleURI,
			IndexURI:   req.IndexURI,
			Title:      req.Title,
			Kind:       "prefix",
			TokenCount: tokens,
			StateRefs:  []inference.StateRef{{Kind: "kv", URI: entryURI, SizeBytes: sizeBytes, Encoding: encoding, Labels: cloneStringMap(refLabels)}},
			Labels:     cloneStringMap(labels),
		},
		Parent:        inference.AgentMemoryRef{URI: req.ParentEntryURI, BundleURI: req.ParentBundleURI, IndexURI: req.ParentIndexURI},
		Bundle:        inference.StateRef{Kind: "bundle", URI: entryURI, SizeBytes: sizeBytes, Encoding: encoding, Labels: cloneStringMap(refLabels)},
		Index:         inference.StateRef{Kind: "index", URI: req.IndexURI},
		TokenCount:    tokens,
		BlockSize:     blockSize,
		BlocksWritten: blocks,
		Encoding:      encoding,
		Labels:        cloneStringMap(labels),
	}, nil
}

func (session *StateSession) ForkState(ctx context.Context, req inference.AgentMemoryWakeRequest) (inference.AgentMemorySession, *inference.AgentMemoryWakeResult, error) {
	if session == nil {
		return nil, nil, core.E("rocm.ForkState", "state session is nil", nil)
	}
	fork := &StateSession{
		model:     cloneModelIdentity(session.model),
		tokenizer: cloneTokenizerIdentity(session.tokenizer),
		labels:    mergeStringMaps(session.labels, map[string]string{"fork": "true"}),
		runtime:   nil,
	}
	wake, err := fork.WakeState(ctx, req)
	if err != nil {
		return nil, nil, core.E("rocm.ForkState", "wake forked state", err)
	}
	return fork, wake, nil
}

func cloneModelIdentity(identity inference.ModelIdentity) inference.ModelIdentity {
	identity.Labels = cloneStringMap(identity.Labels)
	return identity
}

func cloneTokenizerIdentity(identity inference.TokenizerIdentity) inference.TokenizerIdentity {
	identity.Labels = cloneStringMap(identity.Labels)
	return identity
}

func (session *StateSession) checkWakeCompatibility(req inference.AgentMemoryWakeRequest) error {
	if req.SkipCompatibilityCheck {
		return nil
	}
	if err := checkROCmStateModelCompatibility("rocm.WakeState", session.model, req.Model); err != nil {
		return err
	}
	if err := checkROCmStateTokenizerCompatibility("rocm.WakeState", session.tokenizer, req.Tokenizer); err != nil {
		return err
	}
	return nil
}

func (session *StateSession) checkSleepCompatibility(req inference.AgentMemorySleepRequest) error {
	if err := checkROCmStateModelCompatibility("rocm.SleepState", session.model, req.Model); err != nil {
		return err
	}
	if err := checkROCmStateTokenizerCompatibility("rocm.SleepState", session.tokenizer, req.Tokenizer); err != nil {
		return err
	}
	return nil
}

func checkROCmStateModelCompatibility(operation string, sessionModel, reqModel inference.ModelIdentity) error {
	if sessionModel.Hash != "" && reqModel.Hash != "" && sessionModel.Hash != reqModel.Hash {
		return core.E(operation, "model hash mismatch", nil)
	}
	if sessionModel.Architecture != "" && reqModel.Architecture != "" && normalizeROCmArchitecture(sessionModel.Architecture) != normalizeROCmArchitecture(reqModel.Architecture) {
		return core.E(operation, "model architecture mismatch", nil)
	}
	return nil
}

func checkROCmStateTokenizerCompatibility(operation string, sessionTokenizer, reqTokenizer inference.TokenizerIdentity) error {
	if sessionTokenizer.Hash != "" && reqTokenizer.Hash != "" && sessionTokenizer.Hash != reqTokenizer.Hash {
		return core.E(operation, "tokenizer hash mismatch", nil)
	}
	if sessionTokenizer.Kind != "" && reqTokenizer.Kind != "" && sessionTokenizer.Kind != reqTokenizer.Kind {
		return core.E(operation, "tokenizer kind mismatch", nil)
	}
	return nil
}

func (m *rocmModel) WakeState(ctx context.Context, req inference.AgentMemoryWakeRequest) (wake *inference.AgentMemoryWakeResult, err error) {
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	session := m.stateSession()
	wake, err = session.WakeState(ctx, req)
	if err != nil {
		return nil, err
	}
	m.remirrorWakeStateKV(session, wake)
	return wake, nil
}

func (m *rocmModel) SleepState(ctx context.Context, req inference.AgentMemorySleepRequest) (sleep *inference.AgentMemorySleepResult, err error) {
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	return m.stateSession().SleepState(ctx, req)
}

func (m *rocmModel) ForkState(ctx context.Context, req inference.AgentMemoryWakeRequest) (forked inference.AgentMemorySession, wake *inference.AgentMemoryWakeResult, err error) {
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	forked, wake, err = m.stateSession().ForkState(ctx, req)
	if err != nil {
		return nil, nil, err
	}
	if session, ok := forked.(*StateSession); ok {
		m.remirrorWakeStateKV(session, wake)
	}
	return forked, wake, nil
}

func (m *rocmModel) stateSession() *StateSession {
	if m == nil {
		return NewStateSession(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil)
	}
	m.stateMutex.Lock()
	defer m.stateMutex.Unlock()
	if m.state == nil {
		m.state = NewStateSession(m.modelIdentity(), inference.TokenizerIdentity{}, map[string]string{"native_runtime": "hip"})
	}
	return m.state
}

func (m *rocmModel) remirrorWakeStateKV(session *StateSession, wake *inference.AgentMemoryWakeResult) {
	if m == nil || session == nil || wake == nil {
		return
	}
	cache, ok := session.runtime.(*rocmKVCache)
	if !ok || cache == nil || cache.PageCount() == 0 {
		return
	}
	driver := m.wakeStateHIPDriver()
	if driver == nil || !driver.Available() {
		return
	}
	device, err := cache.MirrorToDevice(driver)
	if err != nil {
		rocmAnnotateWakeKVLabels(wake, map[string]string{
			"kv_device_restore":       "failed",
			"kv_device_restore_error": err.Error(),
		})
		return
	}
	if err := session.replaceRuntime(device); err != nil {
		_ = device.Close()
		rocmAnnotateWakeKVLabels(wake, map[string]string{
			"kv_device_restore":       "failed",
			"kv_device_restore_error": err.Error(),
		})
		return
	}
	labels := device.Stats().Labels
	labels["cache_mode"] = device.mode
	labels["kv_restore"] = "device_mirror"
	labels["kv_device_restore"] = "mirrored"
	rocmAnnotateWakeKVLabels(wake, labels)
}

func (m *rocmModel) wakeStateHIPDriver() nativeHIPDriver {
	if m == nil {
		return nil
	}
	m.stateMutex.Lock()
	native := m.native
	m.stateMutex.Unlock()
	loaded, ok := native.(*hipLoadedModel)
	if !ok || loaded == nil || loaded.closed {
		return nil
	}
	return loaded.driver
}

func rocmAnnotateWakeKVLabels(wake *inference.AgentMemoryWakeResult, labels map[string]string) {
	if wake == nil || len(labels) == 0 {
		return
	}
	wake.Labels = mergeStringMaps(wake.Labels, labels)
	wake.Entry.Labels = mergeStringMaps(wake.Entry.Labels, labels)
	wake.Bundle.Labels = mergeStringMaps(wake.Bundle.Labels, labels)
}

func closeROCmStateRuntime(runtime any) error {
	closer, ok := runtime.(interface{ Close() error })
	if !ok || closer == nil {
		return nil
	}
	return closer.Close()
}

func blocksForTokens(tokens, blockSize int) int {
	if tokens <= 0 {
		return 0
	}
	if blockSize <= 0 {
		blockSize = defaultROCmStateBlockSize
	}
	return (tokens + blockSize - 1) / blockSize
}

func wakeKVCacheFromChunk(chunk state.Chunk) (*rocmKVCache, bool, error) {
	if len(chunk.Data) == 0 {
		return nil, false, nil
	}
	cache, err := newROCmKVCacheFromSnapshot(chunk.Data)
	if err != nil {
		return nil, false, core.E("rocm.WakeState", "restore KV cache snapshot", err)
	}
	return cache, true, nil
}

func (session *StateSession) sleepStatePayload(ctx context.Context, req inference.AgentMemorySleepRequest, entryURI string, blockSize int, labels map[string]string) (state.ChunkRef, string, uint64, int, int, error) {
	if cache, ok := session.runtime.(*rocmDeviceKVCache); ok && cache != nil && cache.PageCount() > 0 {
		writer, ok := req.Store.(state.BinaryWriter)
		if !ok || writer == nil {
			return state.ChunkRef{}, "", 0, 0, 0, core.E("rocm.SleepState", "binary state store is missing", nil)
		}
		payload, err := cache.Snapshot()
		if err != nil {
			return state.ChunkRef{}, "", 0, 0, 0, core.E("rocm.SleepState", "snapshot HIP device KV cache", err)
		}
		for key, value := range cache.Stats().Labels {
			labels[key] = value
		}
		labels["kv_serialize"] = "device_mirror"
		labels["cache_mode"] = cache.mode
		ref, err := writer.PutBytes(ctx, payload, state.PutOptions{
			URI:   entryURI,
			Title: req.Title,
			Kind:  "rocm-hip-kv-state",
			Track: cache.mode,
			Tags:  mergeStringMaps(req.Metadata, labels),
		})
		if err != nil {
			return state.ChunkRef{}, "", 0, 0, 0, core.E("rocm.SleepState", "write HIP device KV state ref", err)
		}
		return ref, rocmKVSnapshotEncoding, uint64(len(payload)), cache.TokenCount(), cache.PageCount(), nil
	}
	if cache, ok := session.runtime.(*rocmKVCache); ok && cache != nil && cache.PageCount() > 0 {
		writer, ok := req.Store.(state.BinaryWriter)
		if !ok || writer == nil {
			return state.ChunkRef{}, "", 0, 0, 0, core.E("rocm.SleepState", "binary state store is missing", nil)
		}
		payload, err := cache.Snapshot()
		if err != nil {
			return state.ChunkRef{}, "", 0, 0, 0, core.E("rocm.SleepState", "snapshot KV cache", err)
		}
		for key, value := range cache.Stats().Labels {
			labels[key] = value
		}
		labels["kv_serialize"] = "runtime_owned"
		labels["kv_device_backing"] = "planned"
		labels["cache_mode"] = cache.mode
		ref, err := writer.PutBytes(ctx, payload, state.PutOptions{
			URI:   entryURI,
			Title: req.Title,
			Kind:  "rocm-kv-state",
			Track: cache.mode,
			Tags:  mergeStringMaps(req.Metadata, labels),
		})
		if err != nil {
			return state.ChunkRef{}, "", 0, 0, 0, core.E("rocm.SleepState", "write KV state ref", err)
		}
		return ref, rocmKVSnapshotEncoding, uint64(len(payload)), cache.TokenCount(), cache.PageCount(), nil
	}

	writer, ok := req.Store.(state.Writer)
	if !ok || writer == nil {
		return state.ChunkRef{}, "", 0, 0, 0, core.E("rocm.SleepState", "state store is missing", nil)
	}
	labels["kv_serialize"] = "planned"
	ref, err := writer.Put(ctx, "rocm state placeholder: native KV pages are not serialised yet", state.PutOptions{
		URI:   entryURI,
		Title: req.Title,
		Kind:  "rocm-state",
		Tags:  mergeStringMaps(req.Metadata, labels),
	})
	if err != nil {
		return state.ChunkRef{}, "", 0, 0, 0, core.E("rocm.SleepState", "write state ref", err)
	}
	tokens := firstPositiveInt(req.Model.ContextLength, session.model.ContextLength, blockSize)
	return ref, firstNonEmptyString(req.Encoding, state.CodecMemory), 0, tokens, blocksForTokens(tokens, blockSize), nil
}
