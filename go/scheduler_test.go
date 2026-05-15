// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"iter"
	"sync"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestScheduler_Good_StreamsQueuedRequest(t *testing.T) {
	var _ inference.SchedulerModel = (*ScheduledModel)(nil)
	var _ inference.CancellableModel = (*ScheduledModel)(nil)
	var _ inference.ProbeableModel = (*ScheduledModel)(nil)

	fake := &schedulerFakeTextModel{tokens: []inference.Token{{Text: "a"}, {Text: "b"}}}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()

	handle, stream, err := model.Schedule(context.Background(), inference.ScheduledRequest{ID: "req-1", Prompt: "hello", Sampler: inference.SamplerConfig{MaxTokens: 2, StopSequences: []string{"END"}}})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "req-1", handle.ID)
	core.AssertEqual(t, []string{"a", "b"}, collectScheduledTokenText(stream))
	core.AssertEqual(t, []string{"END"}, fake.lastConfig.StopSequences)
}

func TestScheduler_Good_NormalizesBlankRequestID(t *testing.T) {
	model, err := NewScheduledModel(&schedulerFakeTextModel{tokens: []inference.Token{{Text: "ok"}}}, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()

	handle, stream, err := model.Schedule(context.Background(), inference.ScheduledRequest{ID: "   ", Prompt: "hello"})

	core.RequireNoError(t, err)
	core.AssertContains(t, handle.ID, "rocm-")
	core.AssertEqual(t, []string{"ok"}, collectScheduledTokenText(stream))
}

func TestScheduler_Good_ClonesRequestLabels(t *testing.T) {
	model, err := NewScheduledModel(&schedulerFakeTextModel{tokens: []inference.Token{{Text: "ok"}}}, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()
	labels := map[string]string{"tenant": "a"}

	handle, stream, err := model.Schedule(context.Background(), inference.ScheduledRequest{ID: "labels", Prompt: "hello", Labels: labels})
	core.RequireNoError(t, err)
	handle.Labels["tenant"] = "handle-mutated"
	labels["tenant"] = "caller-mutated"
	token := <-stream
	_ = collectScheduledTokenText(stream)

	core.AssertEqual(t, "a", token.Labels["tenant"])
}

func TestScheduler_Good_ClonesQueuedRequestMessagesAndSampler(t *testing.T) {
	release := make(chan struct{})
	fake := &schedulerFakeTextModel{
		tokens:  []inference.Token{{Text: "ok"}},
		wait:    release,
		started: make(chan string, 2),
	}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 2})
	core.RequireNoError(t, err)
	defer model.Close()

	_, first, err := model.Schedule(context.Background(), inference.ScheduledRequest{ID: "first", Prompt: "hold"})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "hold", <-fake.started)
	req := inference.ScheduledRequest{
		ID:       "queued",
		Messages: []inference.Message{{Role: "user", Content: "original"}},
		Sampler:  inference.SamplerConfig{MaxTokens: 1, StopSequences: []string{"END"}},
	}
	_, second, err := model.Schedule(context.Background(), req)
	core.RequireNoError(t, err)
	req.Messages[0].Content = "mutated"
	req.Sampler.StopSequences[0] = "changed"

	close(release)
	core.AssertEqual(t, []string{"ok"}, collectScheduledTokenText(first))
	core.AssertEqual(t, "original", <-fake.started)
	core.AssertEqual(t, []string{"ok"}, collectScheduledTokenText(second))
	core.AssertEqual(t, []string{"END"}, fake.lastConfig.StopSequences)
}

func TestScheduler_Good_GenerateUsesSchedulerQueue(t *testing.T) {
	release := make(chan struct{})
	fake := &schedulerFakeTextModel{
		tokens:  []inference.Token{{Text: "one"}},
		wait:    release,
		started: make(chan string, 2),
	}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 2})
	core.RequireNoError(t, err)
	defer model.Close()

	firstDone := make(chan []string, 1)
	go func() {
		firstDone <- collectTokenText(model.Generate(context.Background(), "first"))
	}()
	core.AssertEqual(t, "first", <-fake.started)
	secondDone := make(chan []string, 1)
	go func() {
		secondDone <- collectTokenText(model.Generate(context.Background(), "second"))
	}()
	select {
	case started := <-fake.started:
		t.Fatalf("second request started before first released: %q", started)
	case <-time.After(10 * time.Millisecond):
	}

	close(release)
	core.AssertEqual(t, []string{"one"}, <-firstDone)
	core.AssertEqual(t, "second", <-fake.started)
	core.AssertEqual(t, []string{"one"}, <-secondDone)
}

func TestScheduler_Good_ChatUsesSchedulerQueue(t *testing.T) {
	fake := &schedulerFakeTextModel{tokens: []inference.Token{{Text: "reply"}}, started: make(chan string, 1)}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()

	tokens := collectTokenText(model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hello"}}))

	core.AssertEqual(t, "hello", <-fake.started)
	core.AssertEqual(t, []string{"reply"}, tokens)
}

func TestScheduler_Bad_GenerateClosedSchedulerSetsErr(t *testing.T) {
	model, err := NewScheduledModel(&schedulerFakeTextModel{tokens: []inference.Token{{Text: "ok"}}}, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	core.RequireNoError(t, model.Close())

	tokens := collectTokenText(model.Generate(context.Background(), "closed"))

	core.AssertEqual(t, []string{}, tokens)
	core.AssertError(t, model.Err())
	core.AssertContains(t, model.Err().Error(), "scheduler is closed")
}

func TestScheduler_Good_NonStreamingDelegatesClearSchedulerErr(t *testing.T) {
	fake := &schedulerFakeTextModel{
		tokens:          []inference.Token{{Text: "ok"}},
		classifyResults: []inference.ClassifyResult{{Token: inference.Token{Text: "yes"}}},
		batchResults:    []inference.BatchResult{{Tokens: []inference.Token{{Text: "batch"}}}},
	}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	core.AssertEqual(t, []string{}, collectTokenText(model.Generate(ctx, "cancelled")))
	core.AssertError(t, model.Err())

	classified, err := model.Classify(context.Background(), []string{"prompt"})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "yes", classified[0].Token.Text)
	core.AssertNil(t, model.Err())

	ctx, cancel = context.WithCancel(context.Background())
	cancel()
	core.AssertEqual(t, []string{}, collectTokenText(model.Generate(ctx, "cancelled-again")))
	core.AssertError(t, model.Err())

	batches, err := model.BatchGenerate(context.Background(), []string{"prompt"})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "batch", batches[0].Tokens[0].Text)
	core.AssertNil(t, model.Err())
}

func TestScheduler_Good_NonStreamingDelegateResultsCloned(t *testing.T) {
	fake := &schedulerFakeTextModel{
		classifyResults: []inference.ClassifyResult{{
			Token:  inference.Token{Text: "yes"},
			Logits: []float32{1, 2},
		}},
		batchResults: []inference.BatchResult{{
			Tokens: []inference.Token{{Text: "batch"}},
		}},
	}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()

	classified, err := model.Classify(context.Background(), []string{"prompt"}, inference.WithLogits())
	core.RequireNoError(t, err)
	batches, err := model.BatchGenerate(context.Background(), []string{"prompt"})
	core.RequireNoError(t, err)

	classified[0].Logits[0] = 99
	batches[0].Tokens[0].Text = "mutated"

	core.AssertEqual(t, float32(1), fake.classifyResults[0].Logits[0])
	core.AssertEqual(t, "batch", fake.batchResults[0].Tokens[0].Text)
}

func TestScheduler_Good_NonStreamingDelegateInputsCloned(t *testing.T) {
	fake := &schedulerFakeTextModel{
		classifyResults:    []inference.ClassifyResult{{Token: inference.Token{Text: "class"}}},
		batchResults:       []inference.BatchResult{{Tokens: []inference.Token{{Text: "batch"}}}},
		mutatePromptInputs: true,
	}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()
	prompts := []string{"prompt"}

	_, err = model.Classify(context.Background(), prompts)
	core.RequireNoError(t, err)
	core.AssertEqual(t, "prompt", prompts[0])

	_, err = model.BatchGenerate(context.Background(), prompts)
	core.RequireNoError(t, err)
	core.AssertEqual(t, "prompt", prompts[0])
}

func TestScheduler_Bad_NonStreamingDelegatesRecordErr(t *testing.T) {
	fake := &schedulerFakeTextModel{
		classifyErr: core.NewError("classify failed"),
		batchErr:    core.NewError("batch failed"),
	}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()

	_, err = model.Classify(context.Background(), []string{"prompt"})
	core.AssertError(t, err)
	core.AssertContains(t, model.Err().Error(), "classify failed")

	_, err = model.BatchGenerate(context.Background(), []string{"prompt"})
	core.AssertError(t, err)
	core.AssertContains(t, model.Err().Error(), "batch failed")
}

func TestScheduler_Bad_NonStreamingDelegatesPreferCancelledContext(t *testing.T) {
	fake := &schedulerFakeTextModel{
		classifyResults: []inference.ClassifyResult{{Token: inference.Token{Text: "class"}}},
		batchResults:    []inference.BatchResult{{Tokens: []inference.Token{{Text: "batch"}}}},
	}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	classify, err := model.Classify(ctx, []string{"prompt"})

	core.AssertNil(t, classify)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "context canceled")
	core.AssertContains(t, model.Err().Error(), "context canceled")

	batch, err := model.BatchGenerate(ctx, []string{"prompt"})

	core.AssertNil(t, batch)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "context canceled")
	core.AssertContains(t, model.Err().Error(), "context canceled")
}

func TestScheduler_Bad_BatchGenerateRecordsPerPromptErr(t *testing.T) {
	fake := &schedulerFakeTextModel{
		batchResults: []inference.BatchResult{
			{Tokens: []inference.Token{{Text: "ok"}}},
			{Err: core.NewError("prompt failed")},
		},
	}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()

	results, err := model.BatchGenerate(context.Background(), []string{"ok", "bad"})

	core.RequireNoError(t, err)
	if len(results) != 2 || results[1].Err == nil {
		t.Fatalf("BatchGenerate = %+v, want per-prompt error", results)
	}
	core.AssertContains(t, model.Err().Error(), "prompt failed")
}

func TestScheduler_Good_CancelsBeforeStart(t *testing.T) {
	release := make(chan struct{})
	fake := &schedulerFakeTextModel{tokens: []inference.Token{{Text: "first"}}, wait: release, started: make(chan string, 2)}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 2})
	core.RequireNoError(t, err)
	defer model.Close()

	_, first, err := model.Schedule(context.Background(), inference.ScheduledRequest{ID: "first", Prompt: "hold"})
	core.RequireNoError(t, err)
	<-fake.started
	_, second, err := model.Schedule(context.Background(), inference.ScheduledRequest{ID: "second", Prompt: "cancel"})
	core.RequireNoError(t, err)
	model.setErr(core.NewError("stale scheduler failure"))

	cancelled, err := model.CancelRequest(context.Background(), "second")
	core.RequireNoError(t, err)
	core.AssertTrue(t, cancelled.Cancelled)
	core.AssertNil(t, model.Err())
	close(release)

	core.AssertEqual(t, []string{"first"}, collectScheduledTokenText(first))
	core.AssertEqual(t, []string{}, collectScheduledTokenText(second))
}

func TestScheduler_Good_CancelsDuringDecode(t *testing.T) {
	fake := &schedulerFakeTextModel{tokens: []inference.Token{{Text: "a"}, {Text: "b"}, {Text: "c"}}, perTokenDelay: 5 * time.Millisecond}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()

	_, stream, err := model.Schedule(context.Background(), inference.ScheduledRequest{ID: "decode", Prompt: "x"})
	core.RequireNoError(t, err)
	first := <-stream
	core.AssertEqual(t, "a", first.Token.Text)
	model.setErr(core.NewError("stale scheduler failure"))

	cancelled, err := model.CancelRequest(context.Background(), "decode")
	core.RequireNoError(t, err)
	core.AssertTrue(t, cancelled.Cancelled)
	core.AssertNil(t, model.Err())
	remaining := collectScheduledTokenText(stream)
	if len(remaining) >= 2 {
		t.Fatalf("remaining tokens = %+v, want cancellation before full decode", remaining)
	}
}

func TestScheduler_Bad_RejectsNilModel(t *testing.T) {
	model, err := NewScheduledModel(nil, SchedulerConfig{})

	core.AssertNil(t, model)
	core.AssertError(t, err)
}

func TestScheduler_Bad_NilWrappedModelRecordsErr(t *testing.T) {
	model := &ScheduledModel{}

	_, stream, err := model.Schedule(context.Background(), inference.ScheduledRequest{Prompt: "prompt"})
	core.AssertError(t, err)
	core.AssertNil(t, stream)
	core.AssertContains(t, model.Err().Error(), "scheduled model is nil")

	_, err = model.CancelRequest(context.Background(), "prompt")
	core.AssertError(t, err)
	core.AssertContains(t, model.Err().Error(), "scheduled model is nil")

	core.AssertEqual(t, []string{}, collectTokenText(model.Generate(context.Background(), "prompt")))
	core.AssertContains(t, model.Err().Error(), "scheduled model is nil")

	core.AssertEqual(t, []string{}, collectTokenText(model.Chat(context.Background(), nil)))
	core.AssertContains(t, model.Err().Error(), "scheduled model is nil")

	_, err = model.Classify(context.Background(), []string{"prompt"})
	core.AssertError(t, err)
	core.AssertContains(t, model.Err().Error(), "scheduled model is nil")

	_, err = model.BatchGenerate(context.Background(), []string{"prompt"})
	core.AssertError(t, err)
	core.AssertContains(t, model.Err().Error(), "scheduled model is nil")

	core.RequireNoError(t, model.Close())
}

func TestScheduler_Bad_RejectsClosedScheduler(t *testing.T) {
	model, err := NewScheduledModel(&schedulerFakeTextModel{tokens: []inference.Token{{Text: "ok"}}}, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	core.RequireNoError(t, model.Close())

	_, stream, err := model.Schedule(context.Background(), inference.ScheduledRequest{ID: "closed", Prompt: "x"})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "scheduler is closed")
	core.AssertContains(t, model.Err().Error(), "scheduler is closed")
	if stream != nil {
		t.Fatalf("closed scheduler stream = %v, want nil", stream)
	}
}

func TestScheduler_Good_CloseIsIdempotent(t *testing.T) {
	fake := &schedulerFakeTextModel{tokens: []inference.Token{{Text: "ok"}}}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)

	core.RequireNoError(t, model.Close())
	core.RequireNoError(t, model.Close())

	core.AssertEqual(t, 1, fake.closeCalls)
}

func TestScheduler_Bad_RejectsCancelledContextBeforeEnqueue(t *testing.T) {
	model, err := NewScheduledModel(&schedulerFakeTextModel{tokens: []inference.Token{{Text: "ok"}}}, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, stream, err := model.Schedule(ctx, inference.ScheduledRequest{ID: "cancelled", Prompt: "x"})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "enqueue request")
	core.AssertContains(t, model.Err().Error(), "enqueue request")
	if stream != nil {
		t.Fatalf("cancelled scheduler stream = %v, want nil", stream)
	}
}

func TestScheduler_Bad_RejectsBlankCancelID(t *testing.T) {
	model, err := NewScheduledModel(&schedulerFakeTextModel{tokens: []inference.Token{{Text: "ok"}}}, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()

	_, err = model.CancelRequest(context.Background(), "   ")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "request id is empty")
	core.AssertContains(t, model.Err().Error(), "request id is empty")
}

func TestScheduler_Bad_RejectsDuplicateInFlightRequestID(t *testing.T) {
	release := make(chan struct{})
	fake := &schedulerFakeTextModel{tokens: []inference.Token{{Text: "first"}}, wait: release, started: make(chan string, 1)}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()

	_, first, err := model.Schedule(context.Background(), inference.ScheduledRequest{ID: "same", Prompt: "hold"})
	core.RequireNoError(t, err)
	<-fake.started

	_, second, err := model.Schedule(context.Background(), inference.ScheduledRequest{ID: "same", Prompt: "duplicate"})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "duplicate request id")
	core.AssertContains(t, model.Err().Error(), "duplicate request id")
	if second != nil {
		t.Fatalf("duplicate stream = %v, want nil", second)
	}

	close(release)
	core.AssertEqual(t, []string{"first"}, collectScheduledTokenText(first))
}

func TestScheduler_Bad_RejectsFullQueueWithoutBlocking(t *testing.T) {
	release := make(chan struct{})
	fake := &schedulerFakeTextModel{tokens: []inference.Token{{Text: "ok"}}, wait: release, started: make(chan string, 1)}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()

	_, first, err := model.Schedule(context.Background(), inference.ScheduledRequest{ID: "first", Prompt: "hold"})
	core.RequireNoError(t, err)
	<-fake.started
	_, second, err := model.Schedule(context.Background(), inference.ScheduledRequest{ID: "second", Prompt: "queued"})
	core.RequireNoError(t, err)

	_, third, err := model.Schedule(context.Background(), inference.ScheduledRequest{ID: "third", Prompt: "full"})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "queue is full")
	core.AssertContains(t, model.Err().Error(), "queue is full")
	if third != nil {
		t.Fatalf("full queue stream = %v, want nil", third)
	}
	close(release)
	core.AssertEqual(t, []string{"ok"}, collectScheduledTokenText(first))
	core.AssertEqual(t, []string{"ok"}, collectScheduledTokenText(second))
}

func TestScheduler_Good_DelegatesUnknownCancelToBaseModel(t *testing.T) {
	fake := &schedulerFakeTextModel{tokens: []inference.Token{{Text: "ok"}}}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()
	model.setErr(core.NewError("stale scheduler failure"))

	cancelled, err := model.CancelRequest(context.Background(), "external")

	core.RequireNoError(t, err)
	core.AssertTrue(t, cancelled.Cancelled)
	core.AssertEqual(t, "external", fake.cancelledID)
	core.AssertEqual(t, "base_cancelled", cancelled.Reason)
	core.AssertNil(t, model.Err())
}

func TestScheduler_Bad_CancelRequestRecordsErr(t *testing.T) {
	fake := &schedulerFakeTextModel{
		tokens:    []inference.Token{{Text: "ok"}},
		cancelErr: core.NewError("cancel failed"),
	}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = model.CancelRequest(ctx, "external")
	core.AssertError(t, err)
	core.AssertContains(t, model.Err().Error(), "context canceled")

	_, err = model.CancelRequest(context.Background(), "external")
	core.AssertError(t, err)
	core.AssertContains(t, model.Err().Error(), "cancel failed")
}

func TestScheduler_Ugly_SlowConsumerDoesNotDeadlock(t *testing.T) {
	fake := &schedulerFakeTextModel{tokens: []inference.Token{{Text: "a"}, {Text: "b"}, {Text: "c"}}}
	model, err := NewScheduledModel(fake, SchedulerConfig{QueueSize: 1, OutputBuffer: 1})
	core.RequireNoError(t, err)
	defer model.Close()

	_, stream, err := model.Schedule(context.Background(), inference.ScheduledRequest{ID: "slow", Prompt: "x"})
	core.RequireNoError(t, err)
	time.Sleep(5 * time.Millisecond)

	core.AssertEqual(t, []string{"a", "b", "c"}, collectScheduledTokenText(stream))
}

func TestScheduler_Good_EmitsProbeEvents(t *testing.T) {
	model, err := NewScheduledModel(&schedulerFakeTextModel{tokens: []inference.Token{{Text: "a"}}}, SchedulerConfig{QueueSize: 1})
	core.RequireNoError(t, err)
	defer model.Close()
	var events []inference.ProbeEvent
	model.SetProbeSink(inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		events = append(events, event)
	}))

	_, stream, err := model.Schedule(context.Background(), inference.ScheduledRequest{ID: "probe", Prompt: "x"})
	core.RequireNoError(t, err)
	_ = collectScheduledTokenText(stream)

	event, ok := schedulerEvent(events, "probe", "first_token")
	if !ok {
		t.Fatalf("events = %+v, want scheduler first_token event", events)
	}
	if event.Labels["queue_latency_ms"] == "" || event.Labels["first_token_latency_ms"] == "" || event.Labels["cancelled"] != "false" {
		t.Fatalf("first token event labels = %+v, want queue/first-token latency and cancellation labels", event.Labels)
	}
	if event.Scheduler == nil || event.Scheduler.RequestID != "probe" || event.Scheduler.Event != "first_token" || event.Scheduler.QueueLatencyMillis < 0 || event.Scheduler.FirstTokenLatencyMillis < 0 {
		t.Fatalf("scheduler payload = %+v, want typed scheduler latency payload", event.Scheduler)
	}
}

type schedulerFakeTextModel struct {
	tokens             []inference.Token
	wait               <-chan struct{}
	started            chan string
	perTokenDelay      time.Duration
	err                error
	mu                 sync.Mutex
	lastMetrics        inference.GenerateMetrics
	lastError          error
	lastConfig         inference.GenerateConfig
	cancelledID        string
	cancelErr          error
	closeCalls         int
	classifyResults    []inference.ClassifyResult
	classifyErr        error
	batchResults       []inference.BatchResult
	batchErr           error
	mutatePromptInputs bool
}

func (m *schedulerFakeTextModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.stream(ctx, prompt, opts...)
}

func (m *schedulerFakeTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	prompt := ""
	if len(messages) > 0 {
		prompt = messages[len(messages)-1].Content
	}
	return m.stream(ctx, prompt, opts...)
}

func (m *schedulerFakeTextModel) stream(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)
	m.mu.Lock()
	m.lastConfig = cfg
	m.mu.Unlock()
	return func(yield func(inference.Token) bool) {
		if m.started != nil {
			m.started <- prompt
		}
		if m.wait != nil {
			select {
			case <-m.wait:
			case <-ctx.Done():
				m.setErr(ctx.Err())
				return
			}
		}
		limit := len(m.tokens)
		if cfg.MaxTokens > 0 && cfg.MaxTokens < limit {
			limit = cfg.MaxTokens
		}
		for i := 0; i < limit; i++ {
			if m.perTokenDelay > 0 {
				select {
				case <-time.After(m.perTokenDelay):
				case <-ctx.Done():
					m.setErr(ctx.Err())
					return
				}
			}
			select {
			case <-ctx.Done():
				m.setErr(ctx.Err())
				return
			default:
			}
			if !yield(m.tokens[i]) {
				return
			}
		}
		m.mu.Lock()
		m.lastMetrics = inference.GenerateMetrics{GeneratedTokens: limit}
		m.lastError = m.err
		m.mu.Unlock()
	}
}

func (m *schedulerFakeTextModel) Classify(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	if m.mutatePromptInputs && len(prompts) > 0 {
		prompts[0] = "mutated"
	}
	return m.classifyResults, m.classifyErr
}
func (m *schedulerFakeTextModel) BatchGenerate(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.BatchResult, error) {
	if m.mutatePromptInputs && len(prompts) > 0 {
		prompts[0] = "mutated"
	}
	return m.batchResults, m.batchErr
}
func (m *schedulerFakeTextModel) ModelType() string { return "fake" }
func (m *schedulerFakeTextModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "fake"}
}
func (m *schedulerFakeTextModel) Metrics() inference.GenerateMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastMetrics
}
func (m *schedulerFakeTextModel) Err() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastError
}
func (m *schedulerFakeTextModel) Close() error {
	m.mu.Lock()
	m.closeCalls++
	m.mu.Unlock()
	return nil
}

func (m *schedulerFakeTextModel) CancelRequest(_ context.Context, id string) (inference.RequestCancelResult, error) {
	m.mu.Lock()
	m.cancelledID = id
	m.mu.Unlock()
	return inference.RequestCancelResult{ID: id, Cancelled: id != "", Reason: "base_cancelled"}, m.cancelErr
}

func (m *schedulerFakeTextModel) setErr(err error) {
	m.mu.Lock()
	m.lastError = err
	m.mu.Unlock()
}

func collectScheduledTokenText(stream <-chan inference.ScheduledToken) []string {
	out := []string{}
	for token := range stream {
		out = append(out, token.Token.Text)
	}
	return out
}

func collectTokenText(stream iter.Seq[inference.Token]) []string {
	out := []string{}
	for token := range stream {
		out = append(out, token.Text)
	}
	return out
}

func schedulerEventsContain(events []inference.ProbeEvent, requestID, eventName string) bool {
	_, ok := schedulerEvent(events, requestID, eventName)
	return ok
}

func schedulerEvent(events []inference.ProbeEvent, requestID, eventName string) (inference.ProbeEvent, bool) {
	for _, event := range events {
		if event.Kind == inference.ProbeEventScheduler && event.Labels["request_id"] == requestID && event.Labels["event"] == eventName {
			return event, true
		}
	}
	return inference.ProbeEvent{}, false
}
