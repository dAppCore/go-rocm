// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"strings"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/rocm/internal/gguf"
)

func TestResolveContextLengthUsesModelWindow(t *testing.T) {
	meta := gguf.Metadata{ContextLength: 131072}
	if got := resolveContextLength(0, meta); got != 131072 {
		t.Fatalf("GGUF context length = %d, want model window", got)
	}
	if got := resolveModelContextLength(0, 131072); got != 131072 {
		t.Fatalf("safetensors context length = %d, want model window", got)
	}
	if got := resolveContextLength(8192, meta); got != 8192 {
		t.Fatalf("explicit GGUF context length = %d, want requested value", got)
	}
	if got := resolveModelContextLength(8192, 131072); got != 8192 {
		t.Fatalf("explicit safetensors context length = %d, want requested value", got)
	}
}

func TestResolveContextLengthFallsBackWhenMetadataIsMissing(t *testing.T) {
	if got := resolveContextLength(0, gguf.Metadata{}); got != defaultContextLengthCap {
		t.Fatalf("missing GGUF context length = %d, want fallback %d", got, defaultContextLengthCap)
	}
	if got := resolveModelContextLength(0, 0); got != defaultContextLengthCap {
		t.Fatalf("missing safetensors context length = %d, want fallback %d", got, defaultContextLengthCap)
	}
}

func TestHIPGemma4Q4ResolveGenerateContextUsesRemainingWindow(t *testing.T) {
	model := &hipLoadedModel{contextSize: 8}
	cfg, err := hipGemma4Q4ResolveGenerateContext(model, []int32{1, 2, 3}, inference.GenerateConfig{})
	if err != nil {
		t.Fatalf("resolve generate context failed: %v", err)
	}
	if cfg.MaxTokens != 5 {
		t.Fatalf("max tokens = %d, want remaining context", cfg.MaxTokens)
	}
}

func TestHIPGemma4Q4ResolveGenerateContextKeepsExplicitLimit(t *testing.T) {
	cfg, err := hipGemma4Q4ResolveGenerateContext(&hipLoadedModel{contextSize: 8}, []int32{1, 2, 3}, inference.GenerateConfig{MaxTokens: 2})
	if err != nil {
		t.Fatalf("resolve generate context failed: %v", err)
	}
	if cfg.MaxTokens != 2 {
		t.Fatalf("max tokens = %d, want explicit limit", cfg.MaxTokens)
	}
}

func TestHIPGemma4Q4ResolveGenerateContextRejectsExplicitLimitPastWindow(t *testing.T) {
	_, err := hipGemma4Q4ResolveGenerateContext(&hipLoadedModel{contextSize: 8}, []int32{1, 2, 3}, inference.GenerateConfig{MaxTokens: 6})
	if err == nil {
		t.Fatalf("resolve generate context succeeded, want max-tokens context-window error")
	}
}

func TestHIPGemma4Q4GenerateTokenSeqRejectsExplicitLimitPastWindow(t *testing.T) {
	stream, streamErr := hipGemma4Q4GenerateTokenSeq(context.Background(), &hipLoadedModel{contextSize: 8}, hipGemma4Q4ForwardConfig{}, []int32{1, 2, 3}, inference.GenerateConfig{MaxTokens: 6})
	for token := range stream {
		t.Fatalf("generated token %+v, want context-window error before decode", token)
	}
	err := streamErr()
	if err == nil || !strings.Contains(err.Error(), "remaining model context window") {
		t.Fatalf("stream error = %v, want remaining context-window error", err)
	}
}

func TestHIPGemma4Q4BatchGenerateRejectsExplicitLimitPastWindow(t *testing.T) {
	batch := hipGemma4Q4BatchGenerate(context.Background(), &hipLoadedModel{contextSize: 8}, hipGemma4Q4ForwardConfig{}, []string{"tokens:1,2,3"}, inference.GenerateConfig{MaxTokens: 6})
	if len(batch) != 1 {
		t.Fatalf("batch len = %d, want one result", len(batch))
	}
	if batch[0].Err == nil || !strings.Contains(batch[0].Err.Error(), "remaining model context window") {
		t.Fatalf("batch error = %v, want remaining context-window error", batch[0].Err)
	}
	if len(batch[0].Tokens) != 0 {
		t.Fatalf("batch tokens = %+v, want no tokens before context-window error", batch[0].Tokens)
	}
}

func TestROCmBatchGenerateRejectsExplicitLimitPastGemma4Window(t *testing.T) {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native: &hipLoadedModel{
			modelInfo:   inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6},
			contextSize: 8,
		},
	}

	results, err := model.BatchGenerate(context.Background(), []string{"tokens:1,2,3"}, inference.WithMaxTokens(6))

	if err == nil || !strings.Contains(err.Error(), "remaining model context window") {
		t.Fatalf("BatchGenerate err = %v results=%+v, want remaining context-window error", err, results)
	}
	if model.Err() == nil || !strings.Contains(model.Err().Error(), "remaining model context window") {
		t.Fatalf("model Err() = %v, want remaining context-window error", model.Err())
	}
}

func TestROCmBatchGenerateRejectsPromptAtGemma4Window(t *testing.T) {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native: &hipLoadedModel{
			modelInfo:   inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6},
			contextSize: 3,
		},
	}

	results, err := model.BatchGenerate(context.Background(), []string{"tokens:1,2,3"})

	if err == nil || !strings.Contains(err.Error(), "prompt reaches model context window") {
		t.Fatalf("BatchGenerate err = %v results=%+v, want prompt context-window error", err, results)
	}
}

func TestHIPGemma4Q4ChatRejectsExplicitLimitPastWindow(t *testing.T) {
	model := &hipLoadedModel{
		contextSize: 8,
		modelInfo:   inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 4, NumLayers: 1},
		modelLabels: linkedGemma4TestLabels("E2B", "q4"),
		q4ConfigOK:  true,
		q4Layers:    1,
	}
	stream, streamErr := (hipNativeProjectionKernelSet{}).Chat(context.Background(), model, []inference.Message{{Role: "user", Content: "hello"}}, inference.GenerateConfig{MaxTokens: 8})
	for token := range stream {
		t.Fatalf("generated token %+v, want context-window error before decode", token)
	}
	err := streamErr()
	if err == nil || !strings.Contains(err.Error(), "remaining model context window") {
		t.Fatalf("stream error = %v, want remaining context-window error", err)
	}
}

func TestHIPGemma4Q4ResolveGenerateContextRejectsFullPrompt(t *testing.T) {
	_, err := hipGemma4Q4ResolveGenerateContext(&hipLoadedModel{contextSize: 3}, []int32{1, 2, 3}, inference.GenerateConfig{})
	if err == nil {
		t.Fatalf("resolve generate context succeeded, want context-window error")
	}
}

func TestROCmBenchmarkMaxTokensUsesGemma4RemainingWindow(t *testing.T) {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native: &hipLoadedModel{
			modelInfo:   inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6},
			contextSize: 8,
		},
	}
	got, err := model.benchmarkMaxTokens([]string{"tokens:1,2,3", "tokens:1,2"}, 0)
	if err != nil {
		t.Fatalf("benchmarkMaxTokens: %v", err)
	}
	if got != 5 {
		t.Fatalf("benchmark max tokens = %d, want remaining Gemma4 context", got)
	}
	got, err = model.benchmarkMaxTokens([]string{"tokens:1,2,3"}, 2)
	if err != nil || got != 2 {
		t.Fatalf("explicit benchmark max tokens = %d err=%v, want explicit limit", got, err)
	}
	_, err = model.benchmarkMaxTokens([]string{"tokens:1,2,3"}, 6)
	if err == nil || !strings.Contains(err.Error(), "remaining model context window") {
		t.Fatalf("explicit benchmark error = %v, want remaining context-window error", err)
	}
	_, err = model.benchmarkMaxTokens([]string{"tokens:1,2,3,4,5,6,7,8"}, 0)
	if err == nil || !strings.Contains(err.Error(), "model context window") {
		t.Fatalf("error = %v, want full context prompt rejection", err)
	}
}

func TestROCmBenchmarkMaxTokensKeepsNonGemmaDefault(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen3"}}
	got, err := model.benchmarkMaxTokens([]string{"hello"}, 0)
	if err != nil || got != 32 {
		t.Fatalf("non-Gemma benchmark max tokens = %d err=%v, want legacy default", got, err)
	}
}

func TestROCmQualityProbeMaxTokensUsesGemma4RemainingWindow(t *testing.T) {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native: &hipLoadedModel{
			modelInfo:   inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6},
			contextSize: 8,
		},
	}
	got, err := model.qualityProbeMaxTokens([]inference.QualityProbe{
		{Name: "short", Prompt: "tokens:1,2"},
		{Name: "long", Prompt: "tokens:1,2,3,4"},
	}, 0)
	if err != nil {
		t.Fatalf("qualityProbeMaxTokens: %v", err)
	}
	if got != 4 {
		t.Fatalf("quality probe max tokens = %d, want remaining Gemma4 context", got)
	}
	got, err = model.qualityProbeMaxTokens([]inference.QualityProbe{{Prompt: "tokens:1,2,3"}}, 2)
	if err != nil || got != 2 {
		t.Fatalf("explicit quality probe max tokens = %d err=%v, want explicit limit", got, err)
	}
	_, err = model.qualityProbeMaxTokens([]inference.QualityProbe{{Prompt: "tokens:1,2,3"}}, 6)
	if err == nil || !strings.Contains(err.Error(), "remaining model context window") {
		t.Fatalf("explicit quality probe error = %v, want remaining context-window error", err)
	}
	_, err = model.qualityProbeMaxTokens([]inference.QualityProbe{{Prompt: "tokens:1,2,3,4,5,6,7,8"}}, 0)
	if err == nil || !strings.Contains(err.Error(), "model context window") {
		t.Fatalf("error = %v, want full context probe rejection", err)
	}
}

func TestROCmQualityProbeMaxTokensKeepsNonGemmaDefault(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen3"}}
	got, err := model.qualityProbeMaxTokens([]inference.QualityProbe{{Prompt: "hello"}}, 0)
	if err != nil || got != 32 {
		t.Fatalf("non-Gemma quality probe max tokens = %d err=%v, want legacy default", got, err)
	}
}

func TestROCmGenerateUsesRemainingGemma4MaxTokens(t *testing.T) {
	native := &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "ok"}}}
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native:    native,
	}

	_ = collectTokenText(model.Generate(context.Background(), "one two three"))
	_ = collectTokenText(model.Generate(context.Background(), "one two three", inference.WithTemperature(0.7)))
	_ = collectTokenText(model.Generate(context.Background(), "one two three", inference.WithMaxTokens(7)))
	_ = collectTokenText(model.Generate(context.Background(), "one two three", inference.WithMaxTokens(-1)))

	if len(native.generateConfigs) != 4 {
		t.Fatalf("generate config count = %d, want 4", len(native.generateConfigs))
	}
	remaining := defaultContextLengthCap - 3
	if native.generateConfigs[0].MaxTokens != remaining ||
		native.generateConfigs[1].MaxTokens != remaining ||
		native.generateConfigs[3].MaxTokens != remaining {
		t.Fatalf("Gemma4 unset generate configs = %+v, want remaining context %d", native.generateConfigs, remaining)
	}
	if native.generateConfigs[2].MaxTokens != 7 {
		t.Fatalf("Gemma4 explicit generate config = %+v, want explicit MaxTokens", native.generateConfigs[2])
	}
}

func TestROCmGenerateRejectsExplicitLimitPastGemma4WindowBeforeNative(t *testing.T) {
	native := &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "ok"}}}
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native:    native,
	}
	prompt := strings.Repeat("x ", defaultContextLengthCap-1)

	_ = collectTokenText(model.Generate(context.Background(), prompt, inference.WithMaxTokens(2)))

	if len(native.generateConfigs) != 0 {
		t.Fatalf("native generate configs = %+v, want context rejection before native Generate", native.generateConfigs)
	}
	if model.Err() == nil || !strings.Contains(model.Err().Error(), "remaining model context window") {
		t.Fatalf("model Err() = %v, want remaining context-window error", model.Err())
	}
}

func TestROCmGenerateKeepsNonGemmaDefaultMaxTokens(t *testing.T) {
	native := &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "ok"}}}
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    native,
	}

	_ = collectTokenText(model.Generate(context.Background(), "hello"))

	if len(native.generateConfigs) != 1 || native.generateConfigs[0].MaxTokens != inference.DefaultGenerateConfig().MaxTokens {
		t.Fatalf("non-Gemma generate config = %+v, want inference default max tokens", native.generateConfigs)
	}
}

func TestROCmChatAndBatchUseRemainingGemma4MaxTokens(t *testing.T) {
	native := &fakeNativeModel{
		tokens:             []inference.Token{{ID: 1, Text: "ok"}},
		chatTemplateResult: "one two three four",
	}
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native:    native,
	}

	messages := []inference.Message{{Role: "user", Content: "hello"}}
	chatPromptTokens := model.chatPromptTokenCount(messages)
	_ = collectTokenText(model.Chat(context.Background(), messages))
	_, err := model.BatchGenerate(context.Background(), []string{"one two three", "one two three four five"})
	if err != nil {
		t.Fatalf("BatchGenerate: %v", err)
	}
	_, err = model.BatchGenerate(context.Background(), []string{"one two three", "one two three four five"}, inference.WithMaxTokens(-1))
	if err != nil {
		t.Fatalf("BatchGenerate negative max tokens: %v", err)
	}
	if len(native.generateConfigs) != 3 {
		t.Fatalf("generate config count = %d, want chat and batch configs", len(native.generateConfigs))
	}
	if native.generateConfigs[0].MaxTokens != defaultContextLengthCap-chatPromptTokens {
		t.Fatalf("Gemma4 chat config = %+v, want remaining context %d", native.generateConfigs[0], defaultContextLengthCap-chatPromptTokens)
	}
	if native.generateConfigs[1].MaxTokens != defaultContextLengthCap-5 {
		t.Fatalf("Gemma4 batch config = %+v, want shortest remaining context %d", native.generateConfigs[1], defaultContextLengthCap-5)
	}
	if native.generateConfigs[2].MaxTokens != defaultContextLengthCap-5 {
		t.Fatalf("Gemma4 negative batch config = %+v, want shortest remaining context %d", native.generateConfigs[2], defaultContextLengthCap-5)
	}
}

func TestROCmChatRejectsExplicitLimitPastGemma4WindowBeforeNative(t *testing.T) {
	native := &fakeNativeModel{
		tokens:             []inference.Token{{ID: 1, Text: "ok"}},
		chatTemplateResult: "ignored by Gemma4 registry formatter",
	}
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native:    native,
	}

	_ = collectTokenText(model.Chat(context.Background(), []inference.Message{{Role: "user", Content: strings.Repeat("x ", defaultContextLengthCap-1)}}, inference.WithMaxTokens(2)))

	if len(native.generateConfigs) != 0 {
		t.Fatalf("native generate configs = %+v, want context rejection before native Chat", native.generateConfigs)
	}
	if model.Err() == nil || !strings.Contains(model.Err().Error(), "remaining model context window") {
		t.Fatalf("model Err() = %v, want remaining context-window error", model.Err())
	}
}
