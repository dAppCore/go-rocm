// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"iter"
	"sync"
	"sync/atomic"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// SchedulerConfig controls the package-first ROCm scheduler wrapper.
type SchedulerConfig struct {
	QueueSize    int
	OutputBuffer int
}

// ScheduledModel wraps a TextModel with bounded queueing and request
// cancellation. It does not add kernels; it owns request lifecycle only.
type ScheduledModel struct {
	model        inference.TextModel
	queue        chan *scheduledWork
	outputBuffer int
	nextID       atomic.Uint64

	mu       sync.Mutex
	cancel   map[string]context.CancelFunc
	sink     inference.ProbeSink
	closed   bool
	closeOne sync.Once
	closeErr error
	lastErr  error
}

type scheduledWork struct {
	id       string
	req      inference.ScheduledRequest
	ctx      context.Context
	cancel   context.CancelFunc
	out      chan inference.ScheduledToken
	enqueued time.Time
}

// NewScheduledModel wraps model with a bounded single-worker scheduler.
func NewScheduledModel(model inference.TextModel, cfg SchedulerConfig) (*ScheduledModel, error) {
	if model == nil {
		return nil, core.E("rocm.NewScheduledModel", "model is nil", nil)
	}
	if cfg.QueueSize <= 0 {
		cfg.QueueSize = 1
	}
	if cfg.OutputBuffer <= 0 {
		cfg.OutputBuffer = 1
	}
	scheduled := &ScheduledModel{
		model:        model,
		queue:        make(chan *scheduledWork, cfg.QueueSize),
		outputBuffer: cfg.OutputBuffer,
		cancel:       map[string]context.CancelFunc{},
	}
	go scheduled.run()
	return scheduled, nil
}

func (m *ScheduledModel) Schedule(ctx context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	if m == nil {
		return inference.RequestHandle{}, nil, core.E("rocm.Schedule", "scheduler is nil", nil)
	}
	if m.model == nil {
		err := core.E("rocm.Schedule", "scheduled model is nil", nil)
		m.setErr(err)
		return inference.RequestHandle{}, nil, err
	}
	if m.queue == nil || m.cancel == nil {
		err := core.E("rocm.Schedule", "scheduler is not initialized", nil)
		m.setErr(err)
		return inference.RequestHandle{}, nil, err
	}
	m.setErr(nil)
	if ctx == nil {
		ctx = context.Background()
	}
	req.ID = core.Trim(req.ID)
	if req.ID == "" {
		req.ID = core.Sprintf("rocm-%d", m.nextID.Add(1))
	}
	req.Messages = append([]inference.Message(nil), req.Messages...)
	req.Sampler = cloneSamplerConfig(req.Sampler)
	req.Labels = cloneStringMap(req.Labels)
	if err := ctx.Err(); err != nil {
		err = core.E("rocm.Schedule", "enqueue request", err)
		m.setErr(err)
		return inference.RequestHandle{}, nil, err
	}
	reqCtx, cancel := context.WithCancel(ctx)
	work := &scheduledWork{
		id:       req.ID,
		req:      req,
		ctx:      reqCtx,
		cancel:   cancel,
		out:      make(chan inference.ScheduledToken, m.outputBuffer),
		enqueued: time.Now(),
	}

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		cancel()
		close(work.out)
		err := core.E("rocm.Schedule", "scheduler is closed", nil)
		m.setErr(err)
		return inference.RequestHandle{}, nil, err
	}
	if _, exists := m.cancel[work.id]; exists {
		m.mu.Unlock()
		cancel()
		close(work.out)
		err := core.E("rocm.Schedule", "duplicate request id "+work.id, nil)
		m.setErr(err)
		return inference.RequestHandle{}, nil, err
	}
	m.cancel[work.id] = cancel
	select {
	case m.queue <- work:
		m.mu.Unlock()
		m.emitSchedulerProbe(work.id, "queued", inference.ProbePhaseQueue, 0, 0, 0, false)
		return inference.RequestHandle{ID: work.id, Labels: cloneStringMap(req.Labels)}, work.out, nil
	default:
		delete(m.cancel, work.id)
		m.mu.Unlock()
		cancel()
		close(work.out)
		err := core.E("rocm.Schedule", "queue is full", nil)
		m.setErr(err)
		return inference.RequestHandle{}, nil, err
	}
}

func (m *ScheduledModel) CancelRequest(ctx context.Context, id string) (inference.RequestCancelResult, error) {
	if m == nil {
		return inference.RequestCancelResult{}, core.E("rocm.CancelRequest", "scheduler is nil", nil)
	}
	if m.model == nil {
		err := core.E("rocm.CancelRequest", "scheduled model is nil", nil)
		m.setErr(err)
		return inference.RequestCancelResult{}, err
	}
	if m.cancel == nil {
		err := core.E("rocm.CancelRequest", "scheduler is not initialized", nil)
		m.setErr(err)
		return inference.RequestCancelResult{}, err
	}
	id = core.Trim(id)
	if id == "" {
		err := core.E("rocm.CancelRequest", "request id is empty", nil)
		m.setErr(err)
		return inference.RequestCancelResult{}, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		m.setErr(err)
		return inference.RequestCancelResult{}, err
	}
	m.setErr(nil)
	m.mu.Lock()
	cancel := m.cancel[id]
	m.mu.Unlock()
	if cancel == nil {
		if cancellable, ok := m.model.(inference.CancellableModel); ok {
			result, err := cancellable.CancelRequest(ctx, id)
			if err != nil {
				m.setErr(err)
			}
			return result, err
		}
		return inference.RequestCancelResult{ID: id, Cancelled: false, Reason: "request not found"}, nil
	}
	cancel()
	m.emitSchedulerProbe(id, "cancelled", inference.ProbePhaseQueue, 0, 0, 0, true)
	return inference.RequestCancelResult{ID: id, Cancelled: true}, nil
}

func (m *ScheduledModel) SetProbeSink(sink inference.ProbeSink) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.sink = sink
	m.mu.Unlock()
	if probeable, ok := m.model.(inference.ProbeableModel); ok {
		probeable.SetProbeSink(sink)
	}
}

func (m *ScheduledModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		if m == nil || m.model == nil {
			if m != nil {
				m.setErr(core.E("rocm.Generate", "scheduled model is nil", nil))
			}
			return
		}
		m.setErr(nil)
		req := inference.ScheduledRequest{Prompt: prompt, Sampler: inference.SamplerConfigFromGenerateConfig(inference.ApplyGenerateOpts(opts))}
		_, stream, err := m.Schedule(ctx, req)
		if err != nil {
			m.setErr(err)
			return
		}
		for scheduled := range stream {
			if !yield(scheduled.Token) {
				_, _ = m.CancelRequest(ctx, scheduled.RequestID)
				return
			}
		}
	}
}

func (m *ScheduledModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		if m == nil || m.model == nil {
			if m != nil {
				m.setErr(core.E("rocm.Chat", "scheduled model is nil", nil))
			}
			return
		}
		m.setErr(nil)
		req := inference.ScheduledRequest{Messages: append([]inference.Message(nil), messages...), Sampler: inference.SamplerConfigFromGenerateConfig(inference.ApplyGenerateOpts(opts))}
		_, stream, err := m.Schedule(ctx, req)
		if err != nil {
			m.setErr(err)
			return
		}
		for scheduled := range stream {
			if !yield(scheduled.Token) {
				_, _ = m.CancelRequest(ctx, scheduled.RequestID)
				return
			}
		}
	}
}

func (m *ScheduledModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	if m == nil || m.model == nil {
		err := core.E("rocm.Classify", "scheduled model is nil", nil)
		if m != nil {
			m.setErr(err)
		}
		return nil, err
	}
	m.setErr(nil)
	if err := rocmContextErr(ctx); err != nil {
		m.setErr(err)
		return nil, err
	}
	results, err := m.model.Classify(ctx, append([]string(nil), prompts...), opts...)
	results = cloneClassifyResults(results)
	if err != nil {
		m.setErr(err)
	}
	return results, err
}

func (m *ScheduledModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.BatchResult, error) {
	if m == nil || m.model == nil {
		err := core.E("rocm.BatchGenerate", "scheduled model is nil", nil)
		if m != nil {
			m.setErr(err)
		}
		return nil, err
	}
	m.setErr(nil)
	if err := rocmContextErr(ctx); err != nil {
		m.setErr(err)
		return nil, err
	}
	results, err := m.model.BatchGenerate(ctx, append([]string(nil), prompts...), opts...)
	results = cloneBatchResults(results)
	if err != nil {
		m.setErr(err)
	} else if resultErr := firstBatchResultError(results); resultErr != nil {
		m.setErr(resultErr)
	}
	return results, err
}

func (m *ScheduledModel) ModelType() string {
	if m == nil || m.model == nil {
		return ""
	}
	return m.model.ModelType()
}

func (m *ScheduledModel) Info() inference.ModelInfo {
	if m == nil || m.model == nil {
		return inference.ModelInfo{}
	}
	return m.model.Info()
}

func (m *ScheduledModel) Metrics() inference.GenerateMetrics {
	if m == nil || m.model == nil {
		return inference.GenerateMetrics{}
	}
	return m.model.Metrics()
}

func (m *ScheduledModel) Err() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	err := m.lastErr
	m.mu.Unlock()
	if err != nil {
		return err
	}
	if m.model == nil {
		return nil
	}
	return m.model.Err()
}

func (m *ScheduledModel) Close() error {
	if m == nil {
		return nil
	}
	m.closeOne.Do(func() {
		m.mu.Lock()
		m.closed = true
		for _, cancel := range m.cancel {
			cancel()
		}
		queue := m.queue
		model := m.model
		m.mu.Unlock()
		if queue != nil {
			close(queue)
		}
		if model != nil {
			m.closeErr = model.Close()
		}
	})
	return m.closeErr
}

func (m *ScheduledModel) run() {
	for work := range m.queue {
		m.process(work)
	}
}

func (m *ScheduledModel) process(work *scheduledWork) {
	defer func() {
		m.forget(work.id)
		close(work.out)
	}()

	queueLatency := time.Since(work.enqueued)
	if err := work.ctx.Err(); err != nil {
		m.emitSchedulerProbe(work.id, "cancelled_before_start", inference.ProbePhaseQueue, queueLatency, 0, time.Since(work.enqueued), true)
		return
	}
	m.emitSchedulerProbe(work.id, "started", inference.ProbePhasePrefill, queueLatency, 0, queueLatency, false)

	opts := generateOptionsFromSampler(work.req.Sampler)
	var stream iter.Seq[inference.Token]
	if len(work.req.Messages) > 0 {
		stream = m.model.Chat(work.ctx, append([]inference.Message(nil), work.req.Messages...), opts...)
	} else {
		stream = m.model.Generate(work.ctx, work.req.Prompt, opts...)
	}

	start := time.Now()
	var firstTokenLatency time.Duration
	var count int
	cancelled := false
streamLoop:
	for token := range stream {
		if count == 0 {
			firstTokenLatency = time.Since(start)
			m.emitSchedulerProbe(work.id, "first_token", inference.ProbePhaseDecode, queueLatency, firstTokenLatency, time.Since(work.enqueued), false)
		}
		count++
		select {
		case work.out <- inference.ScheduledToken{
			RequestID: work.id,
			Token:     token,
			Metrics:   m.model.Metrics(),
			Labels:    cloneStringMap(work.req.Labels),
		}:
		case <-work.ctx.Done():
			cancelled = true
			break streamLoop
		}
	}
	if work.ctx.Err() != nil {
		cancelled = true
	}
	event := "completed"
	if cancelled {
		event = "cancelled_during_decode"
	}
	m.emitSchedulerProbe(work.id, event, inference.ProbePhaseDecode, queueLatency, firstTokenLatency, time.Since(work.enqueued), cancelled)
}

func (m *ScheduledModel) forget(id string) {
	m.mu.Lock()
	delete(m.cancel, id)
	m.mu.Unlock()
}

func (m *ScheduledModel) emitSchedulerProbe(id, event string, phase inference.ProbePhase, queueLatency, firstTokenLatency, totalLatency time.Duration, cancelled bool) {
	if m == nil {
		return
	}
	m.mu.Lock()
	sink := m.sink
	queueDepth := len(m.queue)
	m.mu.Unlock()
	if sink == nil {
		return
	}
	sink.EmitProbe(inference.ProbeEvent{
		Kind:  inference.ProbeEventScheduler,
		Phase: phase,
		Labels: map[string]string{
			"request_id":             id,
			"event":                  event,
			"cancelled":              core.Sprintf("%t", cancelled),
			"queue_latency_ms":       core.Sprintf("%d", queueLatency.Milliseconds()),
			"first_token_latency_ms": core.Sprintf("%d", firstTokenLatency.Milliseconds()),
		},
		Scheduler: &inference.ProbeScheduler{
			RequestID:               id,
			Event:                   event,
			QueueDepth:              queueDepth,
			QueueLatencyMillis:      durationMilliseconds(queueLatency),
			FirstTokenLatencyMillis: durationMilliseconds(firstTokenLatency),
			TotalLatencyMillis:      durationMilliseconds(totalLatency),
			Cancelled:               cancelled,
		},
	})
}

func (m *ScheduledModel) setErr(err error) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.lastErr = err
	m.mu.Unlock()
}

func durationMilliseconds(duration time.Duration) float64 {
	return float64(duration) / float64(time.Millisecond)
}

func generateOptionsFromSampler(cfg inference.SamplerConfig) []inference.GenerateOption {
	opts := []inference.GenerateOption{}
	if cfg.MaxTokens > 0 {
		opts = append(opts, inference.WithMaxTokens(cfg.MaxTokens))
	}
	if cfg.Temperature != 0 {
		opts = append(opts, inference.WithTemperature(cfg.Temperature))
	}
	if cfg.TopK != 0 {
		opts = append(opts, inference.WithTopK(cfg.TopK))
	}
	if cfg.TopP != 0 {
		opts = append(opts, inference.WithTopP(cfg.TopP))
	}
	if cfg.RepeatPenalty != 0 {
		opts = append(opts, inference.WithRepeatPenalty(cfg.RepeatPenalty))
	}
	if len(cfg.StopTokens) > 0 {
		opts = append(opts, inference.WithStopTokens(cfg.StopTokens...))
	}
	if len(cfg.StopSequences) > 0 {
		opts = append(opts, inference.WithStopSequences(cfg.StopSequences...))
	}
	if cfg.ReturnLogits {
		opts = append(opts, inference.WithLogits())
	}
	return opts
}

func cloneSamplerConfig(cfg inference.SamplerConfig) inference.SamplerConfig {
	cfg.StopTokens = append([]int32(nil), cfg.StopTokens...)
	cfg.StopSequences = append([]string(nil), cfg.StopSequences...)
	return cfg
}
