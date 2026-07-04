// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"dappco.re/go/inference"
)

func TestServer_Listen_Good(t *testing.T) {
	socketPath, cancel, done := startTestServer(t)
	defer stopTestServer(t, cancel, done)

	info, err := os.Lstat(socketPath)
	if err != nil {
		t.Fatalf("stat socket: %v", err)
	}
	if info.Mode()&os.ModeSocket == 0 {
		t.Fatalf("socket path mode = %v, want socket", info.Mode())
	}
	if got := info.Mode().Perm(); got != socketFileMode {
		t.Fatalf("socket mode = %v, want %v", got, socketFileMode)
	}

	resp := sendFrame(t, socketPath, `{"action":"info"}`)
	if resp["name"] != DaemonName || resp["version"] != "test" {
		t.Fatalf("info response = %+v, want daemon test identity", resp)
	}
	if !containsAction(resp["actions"], "info") {
		t.Fatalf("actions = %v, want info", resp["actions"])
	}
	if !containsRouteStatus(resp["routes"], "score", "heuristic") {
		t.Fatalf("routes = %v, want heuristic score route", resp["routes"])
	}
}

func TestServer_Listen_Bad_InvalidJSON(t *testing.T) {
	socketPath, cancel, done := startTestServer(t)
	defer stopTestServer(t, cancel, done)

	resp := sendFrame(t, socketPath, `{`)
	if resp["status"] != "error" || resp["error"] != "invalid_json" {
		t.Fatalf("response = %+v, want invalid_json error", resp)
	}
}

func TestServer_Listen_Good_GenerateBackend(t *testing.T) {
	backend := &fakeGenerateBackend{result: GenerateResult{Text: "native ok", Model: "default"}}
	socketPath, cancel, done := startTestServerWithConfig(t, ServerConfig{GenerateBackend: backend})
	defer stopTestServer(t, cancel, done)

	resp := sendFrame(t, socketPath, `{"action":"generate","prompt":"hello","model":"default","max_tokens":8}`)

	if resp["status"] != "ok" || resp["text"] != "native ok" {
		t.Fatalf("response = %+v, want native ok", resp)
	}
	if backend.request.Prompt != "hello" || backend.request.MaxTokens != 8 {
		t.Fatalf("backend request = %+v, want prompt/max tokens", backend.request)
	}
}

func TestServer_Listen_Good_ScheduleBackend(t *testing.T) {
	backend := &fakeScheduleBackend{
		handle: inference.RequestHandle{
			ID:    "req-1",
			Model: inference.ModelIdentity{ID: "main"},
		},
		tokens: []inference.ScheduledToken{
			{RequestID: "req-1", Token: inference.Token{ID: 7, Text: "hel"}},
			{RequestID: "req-1", Token: inference.Token{ID: 8, Text: "lo"}},
		},
	}
	socketPath, cancel, done := startTestServerWithConfig(t, ServerConfig{ScheduleBackend: backend})
	defer stopTestServer(t, cancel, done)

	resp := sendFrame(t, socketPath, `{"action":"schedule","id":"req-1","model":"main","prompt":"hello","max_tokens":2,"stop_tokens":[2],"labels":{"tenant":"test"}}`)

	if resp["status"] != "ok" || resp["action"] != "schedule" || resp["id"] != "req-1" || resp["text"] != "hello" {
		t.Fatalf("response = %+v, want schedule response", resp)
	}
	tokens, ok := resp["tokens"].([]any)
	if !ok || len(tokens) != 2 {
		t.Fatalf("tokens = %#v, want two scheduled tokens", resp["tokens"])
	}
	handle, ok := resp["handle"].(map[string]any)
	if !ok || handle["id"] != "req-1" {
		t.Fatalf("handle = %#v, want JSON request handle", resp["handle"])
	}
	if backend.request.ID != "req-1" ||
		backend.request.Model != "main" ||
		backend.request.Prompt != "hello" ||
		backend.request.Sampler.MaxTokens != 2 ||
		len(backend.request.Sampler.StopTokens) != 1 ||
		backend.request.Sampler.StopTokens[0] != 2 ||
		backend.request.Labels["tenant"] != "test" {
		t.Fatalf("backend request = %+v, want normalized schedule request", backend.request)
	}
}

func TestServer_Listen_Good_EmbedBackend(t *testing.T) {
	backend := &fakeEmbedBackend{
		result: &inference.EmbeddingResult{
			Model:   inference.ModelIdentity{ID: "embedder", Architecture: "bert"},
			Vectors: [][]float32{{0.25, 0.75}},
			Usage:   inference.EmbeddingUsage{PromptTokens: 2, TotalTokens: 2},
		},
	}
	socketPath, cancel, done := startTestServerWithConfig(t, ServerConfig{EmbedBackend: backend})
	defer stopTestServer(t, cancel, done)

	resp := sendFrame(t, socketPath, `{"action":"embed","input":["hello"],"model":"embedder","normalize":true}`)

	if resp["status"] != "ok" || resp["action"] != "embed" {
		t.Fatalf("response = %+v, want embed response", resp)
	}
	vectors, ok := resp["vectors"].([]any)
	if !ok || len(vectors) != 1 {
		t.Fatalf("vectors = %#v, want one JSON vector", resp["vectors"])
	}
	if backend.request.Model != "embedder" || len(backend.request.Input) != 1 || !backend.request.Normalize {
		t.Fatalf("backend request = %+v, want model/input/normalize", backend.request)
	}
}

func TestServer_Listen_Good_EngineFeaturesBackend(t *testing.T) {
	backend := &fakeEngineFeaturesBackend{
		result: map[string]any{
			"architecture":  "gemma4_text",
			"text_generate": true,
		},
	}
	socketPath, cancel, done := startTestServerWithConfig(t, ServerConfig{EngineFeaturesBackend: backend})
	defer stopTestServer(t, cancel, done)

	resp := sendFrame(t, socketPath, `{"action":"engine_features","path":"/models/main","model":"main","backend":"rocm","labels":{"tenant":"test"}}`)

	if resp["status"] != "ok" ||
		resp["action"] != "engine_features" ||
		resp["path"] != "/models/main" ||
		resp["model"] != "main" ||
		resp["backend"] != "rocm" {
		t.Fatalf("engine_features = %+v, want JSON feature response", resp)
	}
	features, ok := resp["features"].(map[string]any)
	if !ok || features["architecture"] != "gemma4_text" {
		t.Fatalf("features = %#v, want JSON engine features", resp["features"])
	}
	if backend.request.Path != "/models/main" ||
		backend.request.Model != "main" ||
		backend.request.Backend != "rocm" ||
		backend.request.Labels["tenant"] != "test" {
		t.Fatalf("backend request = %+v, want path/model/backend/labels", backend.request)
	}
}

func TestServer_Listen_Good_RerankBackend(t *testing.T) {
	backend := &fakeRerankBackend{
		result: &inference.RerankResult{
			Model: inference.ModelIdentity{ID: "reranker", Architecture: "bert"},
			Results: []inference.RerankScore{
				{Index: 0, Score: 0.75, Text: "hello"},
			},
			Labels: map[string]string{"backend": "fake"},
		},
	}
	socketPath, cancel, done := startTestServerWithConfig(t, ServerConfig{RerankBackend: backend})
	defer stopTestServer(t, cancel, done)

	resp := sendFrame(t, socketPath, `{"action":"rerank","query":"hi","documents":["hello"],"model":"reranker","top_n":1}`)

	if resp["status"] != "ok" || resp["action"] != "rerank" {
		t.Fatalf("response = %+v, want rerank response", resp)
	}
	results, ok := resp["results"].([]any)
	if !ok || len(results) != 1 {
		t.Fatalf("results = %#v, want one JSON rerank result", resp["results"])
	}
	if backend.request.Model != "reranker" ||
		backend.request.Query != "hi" ||
		len(backend.request.Documents) != 1 ||
		backend.request.TopN != 1 {
		t.Fatalf("backend request = %+v, want model/query/documents/top_n", backend.request)
	}
}

func TestServer_Listen_Good_ScoreRoute(t *testing.T) {
	socketPath, cancel, done := startTestServer(t)
	defer stopTestServer(t, cancel, done)

	resp := sendFrame(t, socketPath, `{"action":"score","prompt":"Check this answer?","response":"Absolutely, you are right."}`)

	if resp["status"] != "ok" || resp["action"] != "score" || resp["kind"] != "pair" {
		t.Fatalf("response = %+v, want pair score response", resp)
	}
	pair, ok := resp["pair"].(map[string]any)
	if !ok || pair["prompt"] == nil || pair["response"] == nil {
		t.Fatalf("pair = %#v, want JSON score pair payload", resp["pair"])
	}
}

func TestServer_Listen_Good_CacheBackend(t *testing.T) {
	backend := &fakeCacheBackend{
		statsResult: inference.CacheStats{Blocks: 3, CacheMode: "block-q8"},
		warmResult: inference.CacheWarmResult{
			Blocks: []inference.CacheBlockRef{{ID: "blk", TokenCount: 2}},
			Stats:  inference.CacheStats{Blocks: 1, CacheMode: "block-q8"},
		},
		clearResult: inference.CacheStats{CacheMode: "block-q8"},
		entriesResult: CacheEntriesResult{
			Entries: []inference.CacheBlockRef{{ID: "blk-a", TokenCount: 2}},
			Stats:   &inference.CacheStats{Blocks: 1, CacheMode: "block-q8"},
		},
	}
	socketPath, cancel, done := startTestServerWithConfig(t, ServerConfig{CacheBackend: backend, CacheEntryBackend: backend})
	defer stopTestServer(t, cancel, done)

	statsResp := sendFrame(t, socketPath, `{"action":"cache_stats","model":"main","labels":{"tenant":"test"}}`)
	if statsResp["status"] != "ok" || statsResp["action"] != "cache_stats" {
		t.Fatalf("stats response = %+v, want cache_stats response", statsResp)
	}
	stats, ok := statsResp["stats"].(map[string]any)
	if !ok || stats["cache_mode"] != "block-q8" {
		t.Fatalf("stats = %#v, want JSON cache stats", statsResp["stats"])
	}
	if backend.statsRequest.Model != "main" || backend.statsRequest.Labels["tenant"] != "test" {
		t.Fatalf("stats request = %+v, want model/labels", backend.statsRequest)
	}

	warmResp := sendFrame(t, socketPath, `{"action":"cache_warm","model":"main","tokens":[1,2],"mode":"block-q8"}`)
	if warmResp["status"] != "ok" || warmResp["action"] != "cache_warm" {
		t.Fatalf("warm response = %+v, want cache_warm response", warmResp)
	}
	result, ok := warmResp["result"].(map[string]any)
	if !ok || result["blocks"] == nil {
		t.Fatalf("warm result = %#v, want JSON cache warm result", warmResp["result"])
	}
	if backend.warmRequest.Model.ID != "main" || len(backend.warmRequest.Tokens) != 2 || backend.warmRequest.Mode != "block-q8" {
		t.Fatalf("warm request = %+v, want model/tokens/mode", backend.warmRequest)
	}

	clearResp := sendFrame(t, socketPath, `{"action":"cache_clear","model":"main","labels":{"tenant":"test"}}`)
	if clearResp["status"] != "ok" || clearResp["action"] != "cache_clear" {
		t.Fatalf("clear response = %+v, want cache_clear response", clearResp)
	}
	if backend.clearRequest.Model != "main" || backend.clearRequest.Labels["tenant"] != "test" {
		t.Fatalf("clear request = %+v, want model/labels", backend.clearRequest)
	}

	entriesResp := sendFrame(t, socketPath, `{"action":"cache_entries","model":"main","labels":{"tenant":"test"}}`)
	if entriesResp["status"] != "ok" || entriesResp["action"] != "cache_entries" {
		t.Fatalf("entries response = %+v, want cache_entries response", entriesResp)
	}
	entries, ok := entriesResp["entries"].([]any)
	if !ok || len(entries) != 1 {
		t.Fatalf("entries = %#v, want JSON cache entries", entriesResp["entries"])
	}
	if _, ok := entriesResp["stats"].(map[string]any); !ok {
		t.Fatalf("stats = %#v, want JSON cache stats alongside entries", entriesResp["stats"])
	}
	if backend.entriesRequest.Model != "main" || backend.entriesRequest.Labels["tenant"] != "test" {
		t.Fatalf("entries request = %+v, want model/labels", backend.entriesRequest)
	}
}

func TestServer_Listen_Good_CancelBackend(t *testing.T) {
	backend := &fakeCancelBackend{result: inference.RequestCancelResult{ID: "req-1", Cancelled: true}}
	socketPath, cancel, done := startTestServerWithConfig(t, ServerConfig{CancelBackend: backend})
	defer stopTestServer(t, cancel, done)

	resp := sendFrame(t, socketPath, `{"action":"cancel","model":"main","id":"req-1"}`)

	if resp["status"] != "ok" || resp["action"] != "cancel" {
		t.Fatalf("response = %+v, want cancel response", resp)
	}
	result, ok := resp["result"].(map[string]any)
	if !ok || result["cancelled"] != true || result["id"] != "req-1" {
		t.Fatalf("result = %#v, want JSON cancellation result", resp["result"])
	}
	if backend.request.Model != "main" || backend.request.ID != "req-1" {
		t.Fatalf("backend request = %+v, want model/id", backend.request)
	}
}

func TestServer_Listen_Good_IntrospectionBackend(t *testing.T) {
	backend := &fakeIntrospectionBackend{
		modelsResult: []ModelRecord{{ID: "main", Path: "/models/main", Default: true}},
		infoResult:   inference.ModelInfo{Architecture: "gemma4", VocabSize: 42},
		capabilitiesResult: inference.CapabilityReport{
			Runtime:   inference.RuntimeIdentity{Backend: "rocm"},
			Available: true,
			Capabilities: []inference.Capability{
				inference.SupportedCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel),
			},
		},
	}
	socketPath, cancel, done := startTestServerWithConfig(t, ServerConfig{IntrospectionBackend: backend})
	defer stopTestServer(t, cancel, done)

	health := sendFrame(t, socketPath, `{"action":"health"}`)
	if health["status"] != "ok" || health["action"] != "health" || health["name"] != DaemonName {
		t.Fatalf("health = %+v, want daemon health response", health)
	}

	models := sendFrame(t, socketPath, `{"action":"models"}`)
	if models["status"] != "ok" || models["action"] != "models" {
		t.Fatalf("models = %+v, want models response", models)
	}
	modelList, ok := models["models"].([]any)
	if !ok || len(modelList) != 1 {
		t.Fatalf("models payload = %#v, want one JSON model record", models["models"])
	}

	info := sendFrame(t, socketPath, `{"action":"model_info","model":"main"}`)
	if info["status"] != "ok" || info["action"] != "model_info" || info["model"] != "main" {
		t.Fatalf("model_info = %+v, want model info response", info)
	}
	infoPayload, ok := info["info"].(map[string]any)
	if !ok || infoPayload["Architecture"] != "gemma4" {
		t.Fatalf("info payload = %#v, want JSON model info", info["info"])
	}
	if backend.infoRequest.Model != "main" {
		t.Fatalf("info request = %+v, want model main", backend.infoRequest)
	}

	capabilities := sendFrame(t, socketPath, `{"action":"capabilities","model":"main"}`)
	if capabilities["status"] != "ok" || capabilities["action"] != "capabilities" {
		t.Fatalf("capabilities = %+v, want capabilities response", capabilities)
	}
	report, ok := capabilities["report"].(map[string]any)
	if !ok || report["available"] != true {
		t.Fatalf("capability report = %#v, want available JSON report", capabilities["report"])
	}
	if backend.capabilitiesRequest.Model != "main" {
		t.Fatalf("capabilities request = %+v, want model main", backend.capabilitiesRequest)
	}
}

func TestServer_Listen_Good_ModelPackBackend(t *testing.T) {
	backend := &fakeModelPackBackend{
		result: &inference.ModelPackInspection{
			Path:      "/models/main",
			Format:    "safetensors",
			Supported: true,
			Model:     inference.ModelIdentity{Architecture: "gemma4_text", QuantBits: 6},
		},
	}
	socketPath, cancel, done := startTestServerWithConfig(t, ServerConfig{ModelPackBackend: backend})
	defer stopTestServer(t, cancel, done)

	resp := sendFrame(t, socketPath, `{"action":"inspect_model_pack","path":"/models/main"}`)

	if resp["status"] != "ok" || resp["action"] != "inspect_model_pack" || resp["path"] != "/models/main" {
		t.Fatalf("inspect_model_pack = %+v, want JSON inspection response", resp)
	}
	inspection, ok := resp["inspection"].(map[string]any)
	if !ok || inspection["path"] != "/models/main" || inspection["format"] != "safetensors" {
		t.Fatalf("inspection = %#v, want JSON model-pack inspection", resp["inspection"])
	}
	if backend.request.Path != "/models/main" {
		t.Fatalf("backend request = %+v, want model pack path", backend.request)
	}
}

func TestServer_Listen_Good_ModelProfileBackend(t *testing.T) {
	backend := &fakeModelProfileBackend{
		result: map[string]any{
			"architecture": "gemma4_text",
			"labels":       map[string]string{"engine_profile_reactive": "true"},
		},
	}
	socketPath, cancel, done := startTestServerWithConfig(t, ServerConfig{ModelProfileBackend: backend})
	defer stopTestServer(t, cancel, done)

	resp := sendFrame(t, socketPath, `{"action":"model_profile","path":"/models/main","model":"main","backend":"rocm","labels":{"tenant":"test"}}`)

	if resp["status"] != "ok" ||
		resp["action"] != "model_profile" ||
		resp["path"] != "/models/main" ||
		resp["model"] != "main" ||
		resp["backend"] != "rocm" {
		t.Fatalf("model_profile = %+v, want JSON profile response", resp)
	}
	profile, ok := resp["profile"].(map[string]any)
	if !ok || profile["architecture"] != "gemma4_text" {
		t.Fatalf("profile = %#v, want JSON model profile", resp["profile"])
	}
	if backend.request.Path != "/models/main" ||
		backend.request.Model != "main" ||
		backend.request.Backend != "rocm" ||
		backend.request.Labels["tenant"] != "test" {
		t.Fatalf("backend request = %+v, want path/model/backend/labels", backend.request)
	}
}

func TestServer_Listen_Good_ModelRoutesBackend(t *testing.T) {
	backend := &fakeModelRoutesBackend{
		result: map[string]any{
			"contract":     "rocm-model-route-plan-v1",
			"architecture": "gemma4_text",
			"labels":       map[string]string{"engine_route_plan_feature": "true"},
		},
	}
	socketPath, cancel, done := startTestServerWithConfig(t, ServerConfig{ModelRoutesBackend: backend})
	defer stopTestServer(t, cancel, done)

	resp := sendFrame(t, socketPath, `{"action":"model_routes","path":"/models/main","model":"main","backend":"rocm","labels":{"tenant":"test"}}`)

	if resp["status"] != "ok" ||
		resp["action"] != "model_routes" ||
		resp["path"] != "/models/main" ||
		resp["model"] != "main" ||
		resp["backend"] != "rocm" {
		t.Fatalf("model_routes = %+v, want JSON route-plan response", resp)
	}
	routes, ok := resp["routes"].(map[string]any)
	if !ok || routes["contract"] != "rocm-model-route-plan-v1" {
		t.Fatalf("routes = %#v, want JSON model route plan", resp["routes"])
	}
	if backend.request.Path != "/models/main" ||
		backend.request.Model != "main" ||
		backend.request.Backend != "rocm" ||
		backend.request.Labels["tenant"] != "test" {
		t.Fatalf("backend request = %+v, want path/model/backend/labels", backend.request)
	}
}

func TestServer_Listen_Good_ModelRegistryBackend(t *testing.T) {
	backend := &fakeModelRegistryBackend{
		result: map[string]any{
			"name":    "rocm-model-registry",
			"backend": "cuda",
			"labels":  map[string]string{"engine_profile_reactive": "true"},
		},
	}
	socketPath, cancel, done := startTestServerWithConfig(t, ServerConfig{ModelRegistryBackend: backend})
	defer stopTestServer(t, cancel, done)

	resp := sendFrame(t, socketPath, `{"action":"registry","backend":"cuda","labels":{"tenant":"test"}}`)

	if resp["status"] != "ok" || resp["action"] != "registry" || resp["backend"] != "cuda" {
		t.Fatalf("registry = %+v, want JSON registry response", resp)
	}
	registry, ok := resp["registry"].(map[string]any)
	if !ok || registry["name"] != "rocm-model-registry" {
		t.Fatalf("registry payload = %#v, want model registry metadata", resp["registry"])
	}
	if backend.request.Backend != "cuda" || backend.request.Labels["tenant"] != "test" {
		t.Fatalf("backend request = %+v, want backend/labels", backend.request)
	}
}

func TestServer_Listen_Good_TokenizerAndParserBackends(t *testing.T) {
	tokenizer := &fakeTokenizerBackend{
		tokenizeResult:     TokenizeResult{Model: "main", Tokens: []int32{11, 12}},
		detokenizeResult:   DetokenizeResult{Model: "main", Text: "hi"},
		chatTemplateResult: ChatTemplateResult{Model: "main", Text: "<user>hi</user>"},
	}
	parser := &fakeParserBackend{
		reasoningResult: inference.ReasoningParseResult{
			VisibleText: "answer",
			Reasoning:   []inference.ReasoningSegment{{Text: "plan"}},
		},
		toolsResult: inference.ToolParseResult{
			VisibleText: "done",
			Calls:       []inference.ToolCall{{Name: "lookup", ArgumentsJSON: `{"id":7}`}},
		},
	}
	socketPath, cancel, done := startTestServerWithConfig(t, ServerConfig{TokenizerBackend: tokenizer, ParserBackend: parser})
	defer stopTestServer(t, cancel, done)

	tokenResp := sendFrame(t, socketPath, `{"action":"tokenize","model":"main","text":"hi"}`)
	if tokenResp["status"] != "ok" || tokenResp["action"] != "tokenize" || tokenResp["model"] != "main" {
		t.Fatalf("tokenize = %+v, want JSON tokenization response", tokenResp)
	}
	tokens, ok := tokenResp["tokens"].([]any)
	if !ok || len(tokens) != 2 {
		t.Fatalf("tokens = %#v, want two JSON token ids", tokenResp["tokens"])
	}
	if tokenizer.tokenizeRequest.Model != "main" || tokenizer.tokenizeRequest.Text != "hi" {
		t.Fatalf("tokenize request = %+v, want model/text", tokenizer.tokenizeRequest)
	}

	detokenResp := sendFrame(t, socketPath, `{"action":"detokenize","model":"main","tokens":[11,12]}`)
	if detokenResp["status"] != "ok" || detokenResp["action"] != "detokenize" || detokenResp["text"] != "hi" {
		t.Fatalf("detokenize = %+v, want JSON detokenize response", detokenResp)
	}
	if tokenizer.detokenizeRequest.Model != "main" || len(tokenizer.detokenizeRequest.Tokens) != 2 {
		t.Fatalf("detokenize request = %+v, want model/tokens", tokenizer.detokenizeRequest)
	}

	templateResp := sendFrame(t, socketPath, `{"action":"chat_template","model":"main","messages":[{"role":"user","content":"hi"}]}`)
	if templateResp["status"] != "ok" || templateResp["action"] != "chat_template" || templateResp["text"] != "<user>hi</user>" {
		t.Fatalf("chat_template = %+v, want JSON template response", templateResp)
	}
	if tokenizer.chatTemplateRequest.Model != "main" || len(tokenizer.chatTemplateRequest.Messages) != 1 {
		t.Fatalf("chat template request = %+v, want model/messages", tokenizer.chatTemplateRequest)
	}

	reasonResp := sendFrame(t, socketPath, `{"action":"parse_reasoning","model":"main","text":"<think>plan</think>answer"}`)
	if reasonResp["status"] != "ok" || reasonResp["action"] != "parse_reasoning" || reasonResp["visible_text"] != "answer" {
		t.Fatalf("parse_reasoning = %+v, want JSON reasoning response", reasonResp)
	}
	if _, ok := reasonResp["reasoning"].([]any); !ok {
		t.Fatalf("reasoning = %#v, want JSON reasoning segments", reasonResp["reasoning"])
	}
	if parser.reasoningRequest.Model != "main" || parser.reasoningRequest.Text == "" {
		t.Fatalf("reasoning request = %+v, want model/text", parser.reasoningRequest)
	}

	toolsResp := sendFrame(t, socketPath, `{"action":"parse_tools","model":"main","response":"<tool_call>{\"name\":\"lookup\",\"arguments\":{\"id\":7}}</tool_call>"}`)
	if toolsResp["status"] != "ok" || toolsResp["action"] != "parse_tools" || toolsResp["visible_text"] != "done" {
		t.Fatalf("parse_tools = %+v, want JSON tools response", toolsResp)
	}
	if calls, ok := toolsResp["calls"].([]any); !ok || len(calls) != 1 {
		t.Fatalf("calls = %#v, want JSON tool call", toolsResp["calls"])
	}
	if parser.toolsRequest.Model != "main" || parser.toolsRequest.Text == "" {
		t.Fatalf("tools request = %+v, want model/text", parser.toolsRequest)
	}
}

func TestServer_Listen_Ugly_ExistingNonSocket(t *testing.T) {
	socketPath := filepath.Join(t.TempDir(), "lthn.sock")
	if err := os.WriteFile(socketPath, []byte("not a socket"), 0o600); err != nil {
		t.Fatalf("write existing file: %v", err)
	}

	err := NewServer(ServerConfig{SocketPath: socketPath}).ListenAndServe(context.Background())

	if err == nil || !strings.Contains(err.Error(), "refusing to replace non-socket") {
		t.Fatalf("ListenAndServe() error = %v, want non-socket refusal", err)
	}
	if _, err := os.Stat(socketPath); err != nil {
		t.Fatalf("existing non-socket was removed: %v", err)
	}
}

func TestNewServer_ConfigCloneAndDefaults_Good(t *testing.T) {
	modelPaths := map[string]string{"default": "/models/qwen"}
	server := NewServer(ServerConfig{ModelPaths: modelPaths})
	modelPaths["default"] = "/mutated"

	if server.Registry == nil {
		t.Fatal("Registry = nil, want default registry")
	}
	if server.ModelPaths["default"] != "/models/qwen" {
		t.Fatalf("ModelPaths was not cloned: %+v", server.ModelPaths)
	}
	if server.GenerateBackend == nil {
		t.Fatal("GenerateBackend = nil, want native backend for configured model paths")
	}
	if server.ScheduleBackend == nil {
		t.Fatal("ScheduleBackend = nil, want native scheduler backend for configured model paths")
	}
	if server.IntrospectionBackend == nil {
		t.Fatal("IntrospectionBackend = nil, want native model introspection backend for configured model paths")
	}
	if server.ModelPackBackend == nil {
		t.Fatal("ModelPackBackend = nil, want native model-pack inspection backend for configured model paths")
	}
	if server.ModelProfileBackend == nil {
		t.Fatal("ModelProfileBackend = nil, want native metadata profile backend for configured model paths")
	}
	if server.ModelRoutesBackend == nil {
		t.Fatal("ModelRoutesBackend = nil, want native metadata route-plan backend for configured model paths")
	}
	if server.CacheBackend == nil {
		t.Fatal("CacheBackend = nil, want native cache backend for configured model paths")
	}
	if server.CacheEntryBackend == nil {
		t.Fatal("CacheEntryBackend = nil, want native cache entry backend for configured model paths")
	}
	if server.CancelBackend == nil {
		t.Fatal("CancelBackend = nil, want native cancel backend for configured model paths")
	}
	if server.EmbedBackend == nil {
		t.Fatal("EmbedBackend = nil, want native embed backend for configured model paths")
	}
	if server.EngineFeaturesBackend == nil {
		t.Fatal("EngineFeaturesBackend = nil, want native metadata feature backend for configured model paths")
	}
	if server.RerankBackend == nil {
		t.Fatal("RerankBackend = nil, want native rerank backend for configured model paths")
	}
	if server.TokenizerBackend == nil {
		t.Fatal("TokenizerBackend = nil, want native tokenizer backend for configured model paths")
	}
	if server.ParserBackend == nil {
		t.Fatal("ParserBackend = nil, want native parser backend for configured model paths")
	}

	explicit := NewServer(ServerConfig{Registry: NewRegistry("lthn-test", "1"), GenerateBackend: &fakeGenerateBackend{}})
	if explicit.GenerateBackend == nil {
		t.Fatal("explicit GenerateBackend was not retained")
	}
}

type closeableGenerateBackend struct {
	fakeGenerateBackend
	closeErr error
	closed   bool
}

func (backend *closeableGenerateBackend) Close() error {
	backend.closed = true
	return backend.closeErr
}

func TestCloseGenerateBackend_Good(t *testing.T) {
	if err := closeGenerateBackend(nil); err != nil {
		t.Fatalf("closeGenerateBackend(nil) = %v", err)
	}
	if err := closeGenerateBackend(&fakeGenerateBackend{}); err != nil {
		t.Fatalf("closeGenerateBackend(non-closer) = %v", err)
	}

	wantErr := errors.New("close failed")
	backend := &closeableGenerateBackend{closeErr: wantErr}
	if err := closeGenerateBackend(backend); !errors.Is(err, wantErr) {
		t.Fatalf("closeGenerateBackend(closer) = %v, want %v", err, wantErr)
	}
	if !backend.closed {
		t.Fatal("closeable backend was not closed")
	}
}

func TestServer_ResolvedSocketPathAndDefaults_Good(t *testing.T) {
	socketPath := filepath.Join(t.TempDir(), "lthn.sock")
	server := &Server{SocketPath: socketPath}
	got, err := server.resolvedSocketPath()
	if err != nil {
		t.Fatalf("resolvedSocketPath(explicit): %v", err)
	}
	if got != socketPath {
		t.Fatalf("resolvedSocketPath(explicit) = %q, want %q", got, socketPath)
	}

	if runtime.GOOS != "darwin" {
		runtimeDir := t.TempDir()
		t.Setenv("XDG_RUNTIME_DIR", runtimeDir)
		defaultPath, err := DefaultSocketPath()
		if err != nil {
			t.Fatalf("DefaultSocketPath(): %v", err)
		}
		if defaultPath != filepath.Join(runtimeDir, "ofm", "lthn.sock") {
			t.Fatalf("DefaultSocketPath() = %q, want runtime dir lthn.sock", defaultPath)
		}
	}
}

func TestPrepareSocketPath_Validation_Bad(t *testing.T) {
	if err := prepareSocketPath(""); err == nil {
		t.Fatal("expected empty socket path error")
	}

	socketPath := filepath.Join(t.TempDir(), "stale.sock")
	ln, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("listen stale socket: %v", err)
	}
	if err := ln.Close(); err != nil {
		t.Fatalf("close stale listener: %v", err)
	}
	if err := prepareSocketPath(socketPath); err != nil {
		t.Fatalf("prepareSocketPath(stale socket): %v", err)
	}
	if _, err := os.Lstat(socketPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("stale socket path err = %v, want missing", err)
	}
}

func TestWriteJSONLineAndRemovePath_Bad(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	if err := writeJSONLine(buf, map[string]string{"status": "ok"}); err != nil {
		t.Fatalf("writeJSONLine(valid): %v", err)
	}
	if got := buf.String(); got != "{\"status\":\"ok\"}\n" {
		t.Fatalf("writeJSONLine output = %q", got)
	}
	if err := writeJSONLine(bytes.NewBuffer(nil), map[string]any{"bad": make(chan int)}); err == nil {
		t.Fatal("expected JSON marshal error")
	}

	path := filepath.Join(t.TempDir(), "delete-me")
	if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}
	if err := removePath(path); err != nil {
		t.Fatalf("removePath(existing): %v", err)
	}
	if err := removePath(path); err == nil {
		t.Fatal("expected removePath missing file error")
	}
}

func startTestServer(t *testing.T) (string, context.CancelFunc, <-chan error) {
	t.Helper()
	return startTestServerWithConfig(t, ServerConfig{
		Registry: NewRegistry(DaemonName, "test"),
	})
}

func startTestServerWithConfig(t *testing.T, cfg ServerConfig) (string, context.CancelFunc, <-chan error) {
	t.Helper()
	tmpDir := t.TempDir()
	socketPath := filepath.Join(tmpDir, "ofm", "lthn.sock")
	ctx, cancel := context.WithCancel(context.Background())
	cfg.SocketPath = socketPath
	if cfg.Registry == nil {
		cfg.Registry = NewRegistry(DaemonName, "test")
	}
	srv := NewServer(cfg)

	done := make(chan error, 1)
	go func() {
		done <- srv.ListenAndServe(ctx)
	}()

	waitForSocket(t, socketPath)
	return socketPath, cancel, done
}

func stopTestServer(t *testing.T, cancel context.CancelFunc, done <-chan error) {
	t.Helper()
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("server shutdown: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("server did not shut down")
	}
}

func waitForSocket(t *testing.T, socketPath string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		info, err := os.Lstat(socketPath)
		if err == nil && info.Mode()&os.ModeSocket != 0 {
			conn, err := net.DialTimeout("unix", socketPath, 50*time.Millisecond)
			if err == nil {
				_ = conn.Close()
				return
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("socket %s was not created", socketPath)
}

func sendFrame(t *testing.T, socketPath, frame string) map[string]any {
	t.Helper()
	conn, err := net.DialTimeout("unix", socketPath, time.Second)
	if err != nil {
		t.Fatalf("dial socket: %v", err)
	}
	defer conn.Close()

	if _, err := conn.Write([]byte(frame + "\n")); err != nil {
		t.Fatalf("write frame: %v", err)
	}
	line, err := bufio.NewReader(conn).ReadBytes('\n')
	if err != nil {
		t.Fatalf("read response: %v", err)
	}

	var resp map[string]any
	if err := json.Unmarshal(line, &resp); err != nil {
		t.Fatalf("decode response %q: %v", string(line), err)
	}
	return resp
}

func containsAction(raw any, action string) bool {
	actions, ok := raw.([]any)
	if !ok {
		return false
	}
	for _, got := range actions {
		if got == action {
			return true
		}
	}
	return false
}

func containsRouteStatus(raw any, action, status string) bool {
	routes, ok := raw.([]any)
	if !ok {
		return false
	}
	for _, item := range routes {
		route, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if route["action"] == action && route["status"] == status {
			return true
		}
	}
	return false
}
