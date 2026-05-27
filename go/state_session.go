// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/state"
)

const defaultROCmStateBlockSize = 128

const (
	rocmKVStateIndexKind     = "rocm-kv-state-block-bundle-index"
	rocmKVStateIndexEncoding = "rocm/kv-cache-block-bundle-index+json"
)

type rocmKVStateIndex struct {
	Version    int                         `json:"version"`
	Kind       string                      `json:"kind"`
	BundleURI  string                      `json:"bundle_uri,omitempty"`
	TokenCount int                         `json:"token_count,omitempty"`
	BlockSize  int                         `json:"block_size,omitempty"`
	Model      inference.ModelIdentity     `json:"model,omitempty"`
	Tokenizer  inference.TokenizerIdentity `json:"tokenizer,omitempty"`
	Entries    []rocmKVStateIndexEntry     `json:"entries,omitempty"`
	Hash       string                      `json:"hash,omitempty"`
}

type rocmKVStateIndexEntry struct {
	URI        string            `json:"uri"`
	BundleURI  string            `json:"bundle_uri,omitempty"`
	Title      string            `json:"title,omitempty"`
	TokenStart int               `json:"token_start"`
	TokenCount int               `json:"token_count"`
	Labels     map[string]string `json:"labels,omitempty"`
	Meta       map[string]string `json:"meta,omitempty"`
}

func (entry rocmKVStateIndexEntry) PrefixTokens() int {
	return entry.TokenStart + entry.TokenCount
}

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

func cloneStateRefs(refs []inference.StateRef) []inference.StateRef {
	if len(refs) == 0 {
		return nil
	}
	out := make([]inference.StateRef, len(refs))
	for i, ref := range refs {
		out[i] = ref
		out[i].Labels = cloneStringMap(ref.Labels)
	}
	return out
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
	if req.EntryURI == "" && req.IndexURI == "" {
		return nil, core.E("rocm.WakeState", "entry or index URI is required", nil)
	}
	labels := mergeStringMaps(session.labels, req.Labels)
	if req.IndexURI != "" {
		return session.wakeStateFromIndex(ctx, store, req, labels)
	}
	uri := req.EntryURI
	chunk, err := state.ResolveURI(ctx, store, uri)
	if err != nil {
		indexReq := req
		indexReq.IndexURI = uri + "/index"
		if wake, indexErr := session.wakeStateFromIndex(ctx, store, indexReq, cloneStringMap(labels)); indexErr == nil {
			return wake, nil
		}
		return nil, core.E("rocm.WakeState", "resolve state URI", err)
	}
	if cache, ok, restoreLabels, err := wakeKVCacheFromChunk(ctx, store, chunk); err != nil {
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
		for key, value := range restoreLabels {
			labels[key] = value
		}
		labels["kv_restore"] = "runtime_owned"
		labels["kv_device_backing"] = "planned"
		labels["cache_mode"] = cache.mode
		bundleEncoding := rocmKVSnapshotEncoding
		if restoreLabels["kv_restore_path"] == "block_stream" {
			bundleEncoding = rocmKVBlockBundleEncoding
		}
		return &inference.AgentMemoryWakeResult{
			Entry:        inference.AgentMemoryRef{URI: uri, IndexURI: req.IndexURI, Kind: "prefix", TokenCount: tokens, Labels: cloneStringMap(labels)},
			Bundle:       inference.StateRef{Kind: "kv", URI: firstNonEmptyString(req.EntryURI, uri), SizeBytes: uint64(len(chunk.Data)), Encoding: bundleEncoding, Labels: cloneStringMap(labels)},
			Index:        inference.StateRef{Kind: "index", URI: req.IndexURI},
			PrefixTokens: tokens,
			BundleTokens: tokens,
			BlockSize:    blockSize,
			BlocksRead:   blocks,
			Labels:       cloneStringMap(labels),
		}, nil
	}
	return nil, core.E("rocm.WakeState", "KV state is required; refusing to rebuild retained state from prompt text", nil)
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
	bundleURI := firstNonEmptyString(req.BundleURI, entryURI+"/bundle")
	indexURI := firstNonEmptyString(req.IndexURI, entryURI+"/index")
	encoding := req.Encoding
	if encoding == "" {
		encoding = rocmKVBlockBundleEncoding
	}
	req.Encoding = encoding
	labels := mergeStringMaps(session.labels, req.Labels)
	ref, stateRefs, encoding, sizeBytes, tokens, blocks, err := session.sleepStatePayload(ctx, req, bundleURI, blockSize, labels)
	if err != nil {
		return nil, err
	}
	if parsedBlockSize, parseErr := strconv.Atoi(labels["kv_cache_block_size"]); parseErr == nil && parsedBlockSize > 0 {
		blockSize = parsedBlockSize
	}
	_, indexBytes, err := sleepROCmKVStateIndex(ctx, req, entryURI, bundleURI, indexURI, tokens, blockSize, labels, session.model, session.tokenizer)
	if err != nil {
		return nil, err
	}
	refLabels := cloneStringMap(labels)
	if refLabels == nil {
		refLabels = map[string]string{}
	}
	refLabels["chunk_id"] = core.Sprintf("%d", ref.ChunkID)
	if len(stateRefs) == 0 {
		stateRefs = []inference.StateRef{{Kind: "kv", URI: bundleURI, SizeBytes: sizeBytes, Encoding: encoding, Labels: cloneStringMap(refLabels)}}
	}
	return &inference.AgentMemorySleepResult{
		Entry: inference.AgentMemoryRef{
			URI:        entryURI,
			BundleURI:  bundleURI,
			IndexURI:   indexURI,
			Title:      req.Title,
			Kind:       "prefix",
			TokenCount: tokens,
			StateRefs:  cloneStateRefs(stateRefs),
			Labels:     cloneStringMap(labels),
		},
		Parent:        inference.AgentMemoryRef{URI: req.ParentEntryURI, BundleURI: req.ParentBundleURI, IndexURI: req.ParentIndexURI},
		Bundle:        inference.StateRef{Kind: "bundle", URI: bundleURI, SizeBytes: sizeBytes, Encoding: encoding, Labels: cloneStringMap(refLabels)},
		Index:         inference.StateRef{Kind: "index", URI: indexURI, SizeBytes: uint64(indexBytes), Encoding: rocmKVStateIndexEncoding, Labels: cloneStringMap(labels)},
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

func modelIdentityIsZero(identity inference.ModelIdentity) bool {
	return identity.ID == "" &&
		identity.Path == "" &&
		identity.Architecture == "" &&
		identity.Revision == "" &&
		identity.Hash == "" &&
		identity.QuantBits == 0 &&
		identity.QuantGroup == 0 &&
		identity.QuantType == "" &&
		identity.ContextLength == 0 &&
		identity.NumLayers == 0 &&
		identity.HiddenSize == 0 &&
		identity.VocabSize == 0 &&
		len(identity.Labels) == 0
}

func tokenizerIdentityIsZero(identity inference.TokenizerIdentity) bool {
	return identity.Kind == "" &&
		identity.Path == "" &&
		identity.Hash == "" &&
		identity.ChatTemplate == "" &&
		identity.BOSID == 0 &&
		identity.EOSID == 0 &&
		identity.PADID == 0 &&
		len(identity.Labels) == 0
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
	if m.restoreWakeStateDeviceKVBlocks(ctx, session, req, wake) {
		return wake, nil
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

func (m *rocmModel) restoreWakeStateDeviceKVBlocks(ctx context.Context, session *StateSession, req inference.AgentMemoryWakeRequest, wake *inference.AgentMemoryWakeResult) bool {
	if m == nil || session == nil || wake == nil || wake.Labels["kv_restore_path"] != "block_stream" {
		return false
	}
	store, ok := req.Store.(state.Store)
	if !ok || store == nil {
		return false
	}
	driver := m.wakeStateHIPDriver()
	if driver == nil || !driver.Available() {
		return false
	}
	uri := wake.Bundle.URI
	if uri == "" && req.IndexURI != "" {
		if index, err := loadROCmKVStateIndex(ctx, store, req.IndexURI); err == nil {
			if entry, ok := selectROCmKVStateIndexEntry(index, req.EntryURI); ok {
				uri = firstNonEmptyString(entry.BundleURI, index.BundleURI)
			}
		}
	}
	if uri == "" {
		uri = firstNonEmptyString(req.EntryURI, req.IndexURI)
	}
	if uri == "" {
		return false
	}
	chunk, err := state.ResolveURI(ctx, store, uri)
	if err != nil {
		rocmAnnotateWakeKVLabels(wake, map[string]string{
			"kv_device_restore":       "failed",
			"kv_device_restore_error": err.Error(),
		})
		return false
	}
	device, ok, err := wakeDeviceKVCacheBlockBundleFromChunk(ctx, store, driver, chunk)
	if !ok {
		return false
	}
	if err != nil {
		rocmAnnotateWakeKVLabels(wake, map[string]string{
			"kv_device_restore":       "failed",
			"kv_device_restore_error": err.Error(),
		})
		return false
	}
	if err := session.replaceRuntime(device); err != nil {
		_ = device.Close()
		rocmAnnotateWakeKVLabels(wake, map[string]string{
			"kv_device_restore":       "failed",
			"kv_device_restore_error": err.Error(),
		})
		return false
	}
	labels := device.Stats().Labels
	labels["cache_mode"] = device.mode
	labels["kv_restore"] = "hip_device_block_stream"
	labels["kv_device_restore"] = "block_stream"
	labels["kv_device_restore_path"] = "borrow_ref_pinned"
	rocmAnnotateWakeKVLabels(wake, labels)
	return true
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

func (session *StateSession) wakeStateFromIndex(ctx context.Context, store state.Store, req inference.AgentMemoryWakeRequest, labels map[string]string) (*inference.AgentMemoryWakeResult, error) {
	index, err := loadROCmKVStateIndex(ctx, store, req.IndexURI)
	if err != nil {
		return nil, err
	}
	if !req.SkipCompatibilityCheck {
		if err := checkROCmStateModelCompatibility("rocm.WakeState", session.model, index.Model); err != nil {
			return nil, err
		}
		if err := checkROCmStateTokenizerCompatibility("rocm.WakeState", session.tokenizer, index.Tokenizer); err != nil {
			return nil, err
		}
		if err := checkROCmStateModelCompatibility("rocm.WakeState", req.Model, index.Model); err != nil {
			return nil, err
		}
		if err := checkROCmStateTokenizerCompatibility("rocm.WakeState", req.Tokenizer, index.Tokenizer); err != nil {
			return nil, err
		}
	}
	entry, ok := selectROCmKVStateIndexEntry(index, req.EntryURI)
	if !ok {
		return nil, core.E("rocm.WakeState", "state index entry not found", nil)
	}
	bundleURI := firstNonEmptyString(entry.BundleURI, index.BundleURI)
	if bundleURI == "" {
		return nil, core.E("rocm.WakeState", "state index bundle URI is required", nil)
	}
	chunk, err := state.ResolveURI(ctx, store, bundleURI)
	if err != nil {
		return nil, core.E("rocm.WakeState", "resolve state bundle URI", err)
	}
	prefixTokens := entry.PrefixTokens()
	cache, ok, restoreLabels, err := wakeKVCacheFromChunkWithPrefix(ctx, store, chunk, prefixTokens)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, core.E("rocm.WakeState", "KV state is required; refusing to rebuild retained state from prompt text", nil)
	}
	if err := session.replaceRuntime(cache); err != nil {
		return nil, core.E("rocm.WakeState", "close previous state runtime", err)
	}
	for key, value := range cache.Stats().Labels {
		labels[key] = value
	}
	for key, value := range restoreLabels {
		labels[key] = value
	}
	for key, value := range entry.Labels {
		if core.HasPrefix(key, "kv_") || key == "cache_mode" {
			continue
		}
		labels[key] = value
	}
	labels["kv_restore"] = "runtime_owned"
	labels["kv_device_backing"] = "planned"
	labels["cache_mode"] = cache.mode
	labels["kv_index_restore"] = "state_index"
	bundleEncoding := rocmKVSnapshotEncoding
	if restoreLabels["kv_restore_path"] == "block_stream" {
		bundleEncoding = rocmKVBlockBundleEncoding
	}
	return &inference.AgentMemoryWakeResult{
		Entry:        inference.AgentMemoryRef{URI: entry.URI, BundleURI: bundleURI, IndexURI: req.IndexURI, Title: entry.Title, Kind: "prefix", TokenCount: prefixTokens, Labels: cloneStringMap(labels)},
		Bundle:       inference.StateRef{Kind: "kv", URI: bundleURI, SizeBytes: uint64(len(chunk.Data)), Encoding: bundleEncoding, Labels: cloneStringMap(labels)},
		Index:        inference.StateRef{Kind: "index", URI: req.IndexURI, Encoding: rocmKVStateIndexEncoding},
		PrefixTokens: prefixTokens,
		BundleTokens: index.TokenCount,
		BlockSize:    firstPositiveInt(index.BlockSize, cache.blockSize),
		BlocksRead:   blocksForTokens(prefixTokens, firstPositiveInt(index.BlockSize, cache.blockSize)),
		Labels:       cloneStringMap(labels),
	}, nil
}

func sleepROCmKVStateIndex(ctx context.Context, req inference.AgentMemorySleepRequest, entryURI, bundleURI, indexURI string, tokens, blockSize int, labels map[string]string, model inference.ModelIdentity, tokenizer inference.TokenizerIdentity) (state.ChunkRef, int, error) {
	writer, ok := req.Store.(state.Writer)
	if !ok || writer == nil {
		return state.ChunkRef{}, 0, core.E("rocm.SleepState", "state index store is missing", nil)
	}
	if tokens <= 0 {
		return state.ChunkRef{}, 0, core.E("rocm.SleepState", "KV token count is empty", nil)
	}
	if blockSize <= 0 {
		blockSize = defaultROCmStateBlockSize
	}
	if modelIdentityIsZero(model) {
		model = cloneModelIdentity(req.Model)
	} else {
		model = cloneModelIdentity(model)
	}
	if tokenizerIdentityIsZero(tokenizer) {
		tokenizer = cloneTokenizerIdentity(req.Tokenizer)
	} else {
		tokenizer = cloneTokenizerIdentity(tokenizer)
	}
	entryLabels := cloneStringMap(labels)
	entryMeta := map[string]string{}
	if req.ParentEntryURI != "" {
		entryMeta["parent_entry_uri"] = req.ParentEntryURI
	}
	if req.ParentBundleURI != "" {
		entryMeta["parent_bundle_uri"] = req.ParentBundleURI
	}
	if req.ParentIndexURI != "" {
		entryMeta["parent_index_uri"] = req.ParentIndexURI
	}
	index := rocmKVStateIndex{
		Version:    1,
		Kind:       rocmKVStateIndexKind,
		BundleURI:  bundleURI,
		TokenCount: tokens,
		BlockSize:  blockSize,
		Model:      model,
		Tokenizer:  tokenizer,
		Entries: []rocmKVStateIndexEntry{{
			URI:        entryURI,
			BundleURI:  bundleURI,
			Title:      req.Title,
			TokenStart: 0,
			TokenCount: tokens,
			Labels:     entryLabels,
			Meta:       entryMeta,
		}},
	}
	index.Hash = rocmKVStateIndexHash(index)
	payload, err := json.Marshal(index)
	if err != nil {
		return state.ChunkRef{}, 0, core.E("rocm.SleepState", "encode KV state index", err)
	}
	ref, err := writer.Put(ctx, string(payload), state.PutOptions{
		URI:   indexURI,
		Title: firstNonEmptyString(req.Title, "ROCm KV state index"),
		Kind:  rocmKVStateIndexKind,
		Track: rocmKVStateIndexEncoding,
		Tags:  mergeStringMaps(req.Metadata, labels),
	})
	if err != nil {
		return state.ChunkRef{}, 0, core.E("rocm.SleepState", "write KV state index", err)
	}
	return ref, len(payload), nil
}

func loadROCmKVStateIndex(ctx context.Context, store state.Store, uri string) (*rocmKVStateIndex, error) {
	if uri == "" {
		return nil, core.E("rocm.WakeState", "state index URI is required", nil)
	}
	chunk, err := state.ResolveURI(ctx, store, uri)
	if err != nil {
		return nil, core.E("rocm.WakeState", "resolve state index URI", err)
	}
	data := chunk.Data
	if len(data) == 0 && chunk.Text != "" {
		data = []byte(chunk.Text)
	}
	var index rocmKVStateIndex
	if err := json.Unmarshal(data, &index); err != nil {
		return nil, core.E("rocm.WakeState", "parse KV state index", err)
	}
	if err := validateROCmKVStateIndex(index); err != nil {
		return nil, err
	}
	return &index, nil
}

func validateROCmKVStateIndex(index rocmKVStateIndex) error {
	if index.Version != 1 {
		return core.E("rocm.WakeState", "unsupported KV state index version", nil)
	}
	if index.Kind != rocmKVStateIndexKind {
		return core.E("rocm.WakeState", "invalid KV state index kind", nil)
	}
	if index.TokenCount <= 0 {
		return core.E("rocm.WakeState", "KV state index token count is empty", nil)
	}
	if len(index.Entries) == 0 {
		return core.E("rocm.WakeState", "KV state index has no entries", nil)
	}
	if index.Hash != "" && rocmKVStateIndexHash(index) != index.Hash {
		return core.E("rocm.WakeState", "KV state index hash mismatch", nil)
	}
	for _, entry := range index.Entries {
		if entry.URI == "" {
			return core.E("rocm.WakeState", "KV state index entry URI is required", nil)
		}
		if firstNonEmptyString(entry.BundleURI, index.BundleURI) == "" {
			return core.E("rocm.WakeState", "KV state index entry bundle URI is required", nil)
		}
		if entry.TokenStart < 0 || entry.TokenCount <= 0 || entry.TokenStart+entry.TokenCount > index.TokenCount {
			return core.E("rocm.WakeState", "KV state index entry token range is invalid", nil)
		}
	}
	return nil
}

func selectROCmKVStateIndexEntry(index *rocmKVStateIndex, uri string) (rocmKVStateIndexEntry, bool) {
	if index == nil || len(index.Entries) == 0 {
		return rocmKVStateIndexEntry{}, false
	}
	if uri == "" {
		return cloneROCmKVStateIndexEntry(index.Entries[0]), true
	}
	for _, entry := range index.Entries {
		if entry.URI == uri {
			return cloneROCmKVStateIndexEntry(entry), true
		}
	}
	return rocmKVStateIndexEntry{}, false
}

func cloneROCmKVStateIndexEntry(entry rocmKVStateIndexEntry) rocmKVStateIndexEntry {
	entry.Labels = cloneStringMap(entry.Labels)
	entry.Meta = cloneStringMap(entry.Meta)
	return entry
}

func rocmKVStateIndexHash(index rocmKVStateIndex) string {
	index.Hash = ""
	payload, _ := json.Marshal(index)
	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:])
}

func wakeKVCacheFromChunk(ctx context.Context, store state.Store, chunk state.Chunk) (*rocmKVCache, bool, map[string]string, error) {
	return wakeKVCacheFromChunkWithPrefix(ctx, store, chunk, 0)
}

func wakeKVCacheFromChunkWithPrefix(ctx context.Context, store state.Store, chunk state.Chunk, prefixTokens int) (*rocmKVCache, bool, map[string]string, error) {
	data := chunk.Data
	textFallback := false
	if len(data) == 0 && chunk.Text != "" {
		data = []byte(chunk.Text)
		textFallback = true
	}
	if len(data) == 0 {
		return nil, false, nil, nil
	}
	chunk.Data = data
	if cache, ok, err := wakeKVCacheBlockBundleFromChunk(ctx, store, chunk, prefixTokens); ok || err != nil {
		labels := map[string]string{"kv_restore_path": "block_stream"}
		return cache, ok, labels, err
	}
	cache, err := newROCmKVCacheFromSnapshot(data)
	if err != nil {
		if textFallback {
			return nil, false, nil, nil
		}
		return nil, false, nil, core.E("rocm.WakeState", "restore KV cache snapshot", err)
	}
	if prefixTokens > 0 && prefixTokens < cache.TokenCount() {
		keys, values, err := cache.Restore(0, prefixTokens)
		if err != nil {
			return nil, false, nil, err
		}
		prefix, err := newROCmKVCache(cache.mode, cache.blockSize)
		if err != nil {
			return nil, false, nil, err
		}
		keyWidth, valueWidth, ok := cache.LastVectorWidths()
		if !ok {
			return nil, false, nil, core.E("rocm.WakeState", "KV snapshot vector shape is missing", nil)
		}
		if err := prefix.AppendVectors(0, keyWidth, valueWidth, keys, values); err != nil {
			return nil, false, nil, err
		}
		cache = prefix
	}
	return cache, true, nil, nil
}

func wakeKVCacheBlockBundleFromChunk(ctx context.Context, store state.Store, chunk state.Chunk, prefixTokens int) (*rocmKVCache, bool, error) {
	var bundle rocmKVBlockBundleSnapshot
	if err := json.Unmarshal(chunk.Data, &bundle); err != nil || bundle.Kind != rocmKVBlockBundleKind {
		return nil, false, nil
	}
	targetTokens := bundle.TokenCount
	if prefixTokens > 0 {
		if prefixTokens > bundle.TokenCount {
			return nil, true, core.E("rocm.WakeState", "KV block prefix exceeds bundle token count", nil)
		}
		targetTokens = prefixTokens
	}
	cache, err := newROCmKVCache(bundle.Mode, bundle.BlockSize)
	if err != nil {
		return nil, true, err
	}
	nextStart := 0
	for _, blockRef := range bundle.Blocks {
		if blockRef.TokenStart >= targetTokens {
			break
		}
		blockData, release, err := borrowROCmKVBlockBundleRefBytes(ctx, store, blockRef)
		if err != nil {
			return nil, true, err
		}
		block, err := rocmKVCacheBlockFromBundlePayload(blockRef, blockData)
		if release != nil {
			release()
		}
		if err != nil {
			return nil, true, err
		}
		if block.tokenStart != blockRef.TokenStart || block.tokenCount != blockRef.TokenCount {
			return nil, true, core.E("rocm.WakeState", "KV block token range mismatch", nil)
		}
		if block.tokenStart != nextStart {
			return nil, true, core.E("rocm.WakeState", "KV block token range mismatch", nil)
		}
		if err := cache.validateVectorShape(block.keyWidth, block.valueWidth); err != nil {
			return nil, true, err
		}
		blockEnd := block.tokenStart + block.tokenCount
		if blockEnd > targetTokens {
			keys := block.key.decode()
			values := block.value.decode()
			keepTokens := targetTokens - block.tokenStart
			err = cache.AppendVectors(block.tokenStart, block.keyWidth, block.valueWidth, keys[:keepTokens*block.keyWidth], values[:keepTokens*block.valueWidth])
		} else {
			cache.blocks, err = insertROCmKVCacheBlock(cache.blocks, block)
			cache.setVectorShape(block.keyWidth, block.valueWidth)
		}
		if err != nil {
			return nil, true, err
		}
		nextStart = min(blockEnd, targetTokens)
		if nextStart == targetTokens {
			break
		}
	}
	if cache.TokenCount() != targetTokens {
		return nil, true, core.E("rocm.WakeState", "KV block bundle token count mismatch", nil)
	}
	cache.restoreMillis += float64(cache.TokenCount()) * rocmKVRestoreMillisUnit
	return cache, true, nil
}

func borrowROCmKVBlockBundleRefBytes(ctx context.Context, store state.Store, ref rocmKVBlockBundleRef) ([]byte, func(), error) {
	chunkRef := ref.State
	if chunkRef.ChunkID == 0 && ref.ChunkID != 0 {
		chunkRef.ChunkID = ref.ChunkID
	}
	if chunkRef.ChunkID != 0 || chunkRef.HasFrameOffset || chunkRef.Segment != "" || chunkRef.Codec != "" {
		borrowed, err := state.BorrowRefBytes(ctx, store, chunkRef)
		if err != nil {
			return nil, nil, core.E("rocm.WakeState", "borrow KV block ref", err)
		}
		return borrowed.Data, borrowed.Release, nil
	}
	if ref.URI == "" {
		return nil, nil, core.E("rocm.WakeState", "KV block URI is required", nil)
	}
	chunk, err := state.ResolveURI(ctx, store, ref.URI)
	if err != nil {
		return nil, nil, core.E("rocm.WakeState", "resolve KV block URI", err)
	}
	return chunk.Data, nil, nil
}

func rocmKVCacheBlockFromBundlePayload(ref rocmKVBlockBundleRef, payload []byte) (rocmKVCacheBlock, error) {
	switch firstNonEmptyString(ref.Encoding, rocmKVSnapshotEncoding) {
	case rocmKVBlockRawEncoding:
		return rocmKVCacheBlockFromRawPayload(payload)
	case rocmKVSnapshotEncoding:
		blockCache, err := newROCmKVCacheFromSnapshot(payload)
		if err != nil {
			return rocmKVCacheBlock{}, core.E("rocm.WakeState", "restore KV block snapshot", err)
		}
		if len(blockCache.blocks) != 1 {
			return rocmKVCacheBlock{}, core.E("rocm.WakeState", "KV block metadata mismatch", nil)
		}
		return blockCache.blocks[0], nil
	default:
		return rocmKVCacheBlock{}, core.E("rocm.WakeState", "unsupported KV block encoding", nil)
	}
}

func wakeDeviceKVCacheBlockBundleFromChunk(ctx context.Context, store state.Store, driver nativeHIPDriver, chunk state.Chunk) (*rocmDeviceKVCache, bool, error) {
	data := chunk.Data
	if len(data) == 0 && chunk.Text != "" {
		data = []byte(chunk.Text)
	}
	if len(data) == 0 {
		return nil, false, nil
	}
	var bundle rocmKVBlockBundleSnapshot
	if err := json.Unmarshal(data, &bundle); err != nil || bundle.Kind != rocmKVBlockBundleKind {
		return nil, false, nil
	}
	for _, ref := range bundle.Blocks {
		if firstNonEmptyString(ref.Encoding, rocmKVSnapshotEncoding) != rocmKVBlockRawEncoding {
			return nil, false, nil
		}
	}
	device := &rocmDeviceKVCache{
		driver:     driver,
		mode:       bundle.Mode,
		blockSize:  bundle.BlockSize,
		tokenCount: bundle.TokenCount,
		pages:      make([]rocmDeviceKVPage, 0, len(bundle.Blocks)),
	}
	success := false
	defer func() {
		if !success {
			_ = device.Close()
		}
	}()
	nextStart := 0
	for _, blockRef := range bundle.Blocks {
		blockData, release, err := borrowROCmKVBlockBundleRefBytes(ctx, store, blockRef)
		if err != nil {
			return nil, true, err
		}
		page, err := rocmDeviceKVPageFromRawPayload(driver, blockData)
		if release != nil {
			release()
		}
		if err != nil {
			return nil, true, err
		}
		if page.tokenStart != blockRef.TokenStart || page.tokenCount != blockRef.TokenCount || page.keyWidth != blockRef.KeyWidth || page.valueWidth != blockRef.ValueWidth {
			_ = rocmDeviceKVTensorFree(driver, page.key.pointer, page.key.sizeBytes)
			_ = rocmDeviceKVTensorFree(driver, page.value.pointer, page.value.sizeBytes)
			return nil, true, core.E("rocm.WakeState", "KV device block metadata mismatch", nil)
		}
		if page.tokenStart != nextStart || page.tokenCount <= 0 {
			_ = rocmDeviceKVTensorFree(driver, page.key.pointer, page.key.sizeBytes)
			_ = rocmDeviceKVTensorFree(driver, page.value.pointer, page.value.sizeBytes)
			return nil, true, core.E("rocm.WakeState", "KV device block token range mismatch", nil)
		}
		nextStart += page.tokenCount
		device.pages = append(device.pages, page)
	}
	if bundle.TokenCount > 0 && nextStart != bundle.TokenCount {
		return nil, true, core.E("rocm.WakeState", "KV device block bundle token count mismatch", nil)
	}
	success = true
	return device, true, nil
}

func (session *StateSession) sleepStatePayload(ctx context.Context, req inference.AgentMemorySleepRequest, entryURI string, blockSize int, labels map[string]string) (state.ChunkRef, []inference.StateRef, string, uint64, int, int, error) {
	if cache, ok := session.runtime.(*rocmDeviceKVCache); ok && cache != nil && cache.PageCount() > 0 {
		writer, ok := req.Store.(state.BinaryWriter)
		if !ok || writer == nil {
			return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", "binary state store is missing", nil)
		}
		payload, err := cache.Snapshot()
		if err != nil {
			return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", "snapshot HIP device KV cache", err)
		}
		if req.Encoding == rocmKVBlockBundleEncoding {
			hostCache, err := newROCmKVCacheFromSnapshot(payload)
			if err != nil {
				return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", "decode HIP device KV snapshot", err)
			}
			return sleepKVCacheBlockBundle(ctx, req, writer, entryURI, labels, hostCache, "device_mirror_blocks")
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
			return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", "write HIP device KV state ref", err)
		}
		return ref, nil, rocmKVSnapshotEncoding, uint64(len(payload)), cache.TokenCount(), cache.PageCount(), nil
	}
	if cache, ok := session.runtime.(*rocmKVCache); ok && cache != nil && cache.PageCount() > 0 {
		writer, ok := req.Store.(state.BinaryWriter)
		if !ok || writer == nil {
			return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", "binary state store is missing", nil)
		}
		if req.Encoding == rocmKVBlockBundleEncoding {
			return sleepKVCacheBlockBundle(ctx, req, writer, entryURI, labels, cache, "runtime_owned_blocks")
		}
		payload, err := cache.Snapshot()
		if err != nil {
			return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", "snapshot KV cache", err)
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
			return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", "write KV state ref", err)
		}
		return ref, nil, rocmKVSnapshotEncoding, uint64(len(payload)), cache.TokenCount(), cache.PageCount(), nil
	}

	return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", "KV runtime is required; refusing to write prompt placeholder state", nil)
}

func sleepKVCacheBlockBundle(ctx context.Context, req inference.AgentMemorySleepRequest, writer state.BinaryWriter, entryURI string, labels map[string]string, cache *rocmKVCache, serializeMode string) (state.ChunkRef, []inference.StateRef, string, uint64, int, int, error) {
	if cache == nil {
		return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", "KV cache is nil", nil)
	}
	for key, value := range cache.Stats().Labels {
		labels[key] = value
	}
	labels["kv_serialize"] = serializeMode
	labels["kv_block_bundle"] = "state_refs"
	labels["kv_restore_path"] = "block_stream"
	labels["cache_mode"] = cache.mode
	refs := make([]inference.StateRef, 0, len(cache.blocks))
	bundleRefs := make([]rocmKVBlockBundleRef, 0, len(cache.blocks))
	var totalBytes uint64
	for index, block := range cache.blocks {
		payload, err := cache.rawBlock(block)
		if err != nil {
			return state.ChunkRef{}, nil, "", 0, 0, 0, err
		}
		blockURI := core.Sprintf("%s/block/%06d", entryURI, index)
		blockLabels := mergeStringMaps(labels, map[string]string{
			"kv_block_index":       core.Sprintf("%d", index),
			"kv_block_token_start": core.Sprintf("%d", block.tokenStart),
			"kv_block_token_count": core.Sprintf("%d", block.tokenCount),
		})
		ref, err := writer.PutBytes(ctx, payload, state.PutOptions{
			URI:   blockURI,
			Title: req.Title,
			Kind:  rocmKVBlockKind,
			Track: rocmKVBlockRawEncoding,
			Tags:  mergeStringMaps(req.Metadata, blockLabels),
		})
		if err != nil {
			return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", "write KV state block", err)
		}
		sizeBytes := uint64(len(payload))
		totalBytes += sizeBytes
		stateRef := inference.StateRef{
			Kind:      "kv-block",
			URI:       blockURI,
			SizeBytes: sizeBytes,
			Encoding:  rocmKVBlockRawEncoding,
			Labels:    cloneStringMap(blockLabels),
		}
		refs = append(refs, stateRef)
		bundleRefs = append(bundleRefs, rocmKVBlockBundleRef{
			Index:      index,
			URI:        blockURI,
			ChunkID:    ref.ChunkID,
			State:      ref,
			TokenStart: block.tokenStart,
			TokenCount: block.tokenCount,
			KeyWidth:   block.keyWidth,
			ValueWidth: block.valueWidth,
			SizeBytes:  sizeBytes,
			Encoding:   rocmKVBlockRawEncoding,
			Labels:     cloneStringMap(blockLabels),
		})
	}
	labels["kv_block_bundle_blocks"] = core.Sprintf("%d", len(refs))
	labels["kv_block_bundle_block_bytes"] = core.Sprintf("%d", totalBytes)
	bundle := rocmKVBlockBundleSnapshot{
		Version:     1,
		Kind:        rocmKVBlockBundleKind,
		Mode:        cache.mode,
		BlockSize:   cache.blockSize,
		TokenCount:  cache.TokenCount(),
		MemoryBytes: cache.MemoryBytes(),
		Labels:      cloneStringMap(labels),
		Blocks:      bundleRefs,
	}
	payload, err := json.Marshal(bundle)
	if err != nil {
		return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", "encode KV block bundle", err)
	}
	ref, err := writer.PutBytes(ctx, payload, state.PutOptions{
		URI:   entryURI,
		Title: req.Title,
		Kind:  rocmKVBlockBundleKind,
		Track: rocmKVBlockBundleEncoding,
		Tags:  mergeStringMaps(req.Metadata, labels),
	})
	if err != nil {
		return state.ChunkRef{}, nil, "", 0, 0, 0, core.E("rocm.SleepState", "write KV block bundle", err)
	}
	totalBytes += uint64(len(payload))
	labels["kv_block_bundle_bytes"] = core.Sprintf("%d", totalBytes)
	return ref, refs, rocmKVBlockBundleEncoding, uint64(len(payload)), cache.TokenCount(), len(cache.blocks), nil
}
