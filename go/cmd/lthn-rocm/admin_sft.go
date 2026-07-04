// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"dappco.re/go/inference"
	rocm "dappco.re/go/rocm"
)

const (
	rocmAdminSFTLossRingSize = 512

	rocmAdminSFTDefaultEpochs    = 3
	rocmAdminSFTDefaultBatchSize = 8
	rocmAdminSFTDefaultLR        = 1e-4
	rocmAdminSFTDefaultLoRARank  = 16
	rocmAdminSFTDefaultLoRAAlpha = 32
)

type rocmAdminSFTRequest struct {
	ModelPath     string  `json:"model_path"`
	DatasetPath   string  `json:"dataset_path"`
	AdapterName   string  `json:"adapter_name,omitempty"`
	Backend       string  `json:"backend,omitempty"`
	BatchSize     int     `json:"batch_size,omitempty"`
	Epochs        int     `json:"epochs,omitempty"`
	LearningRate  float64 `json:"learning_rate,omitempty"`
	LoRARank      int     `json:"lora_rank,omitempty"`
	LoRAAlpha     int     `json:"lora_alpha,omitempty"`
	LoRADropout   float64 `json:"lora_dropout,omitempty"`
	MaxSeqLen     int     `json:"max_seq_len,omitempty"`
	ContextLength int     `json:"context_length,omitempty"`
}

type rocmAdminSFTJobState string

const (
	rocmAdminSFTStatePending rocmAdminSFTJobState = "pending"
	rocmAdminSFTStateRunning rocmAdminSFTJobState = "running"
	rocmAdminSFTStateDone    rocmAdminSFTJobState = "done"
	rocmAdminSFTStateFailed  rocmAdminSFTJobState = "failed"
	rocmAdminSFTStateStopped rocmAdminSFTJobState = "stopped"
)

type rocmAdminSFTLossSample struct {
	Step  int     `json:"step"`
	Epoch int     `json:"epoch"`
	Loss  float64 `json:"loss"`
	TS    int64   `json:"ts_unix"`
}

type rocmAdminSFTJob struct {
	JobID       string                    `json:"job_id"`
	State       rocmAdminSFTJobState      `json:"state"`
	ModelPath   string                    `json:"model_path"`
	DatasetPath string                    `json:"dataset_path"`
	AdapterDir  string                    `json:"adapter_dir"`
	Backend     string                    `json:"backend,omitempty"`
	StartedUnix int64                     `json:"started_unix"`
	UpdatedUnix int64                     `json:"updated_unix"`
	EndedUnix   int64                     `json:"ended_unix,omitempty"`
	Step        int                       `json:"step"`
	Epoch       int                       `json:"epoch"`
	LastLoss    float64                   `json:"last_loss"`
	Samples     int                       `json:"samples"`
	Error       string                    `json:"error,omitempty"`
	Loss        []rocmAdminSFTLossSample  `json:"loss,omitempty"`
	Adapter     inference.AdapterIdentity `json:"adapter,omitempty"`
	Metrics     inference.TrainingMetrics `json:"metrics,omitempty"`
	Checkpoints []inference.StateRef      `json:"checkpoints,omitempty"`
	Labels      map[string]string         `json:"labels,omitempty"`

	cancel context.CancelFunc `json:"-"`
}

type rocmAdminSFTRegistry struct {
	mu     sync.RWMutex
	active *rocmAdminSFTJob
	last   *rocmAdminSFTJob
}

func newROCmAdminSFTRegistry() *rocmAdminSFTRegistry {
	return &rocmAdminSFTRegistry{}
}

func rocmAdminSFTStartHandler(registry *rocmAdminSFTRegistry, cfg rocmServeConfig) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !rocmServeRequireMethod(w, r, http.MethodPost) {
			return
		}
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 1<<16))
		if err != nil {
			writeROCmServeError(w, http.StatusBadRequest, "read body: "+err.Error(), "body")
			return
		}
		var req rocmAdminSFTRequest
		if err := json.Unmarshal(body, &req); err != nil {
			writeROCmServeError(w, http.StatusBadRequest, "decode body: "+err.Error(), "body")
			return
		}
		if strings.TrimSpace(req.ModelPath) == "" {
			writeROCmServeError(w, http.StatusBadRequest, "model_path required", "model_path")
			return
		}
		if strings.TrimSpace(req.DatasetPath) == "" {
			writeROCmServeError(w, http.StatusBadRequest, "dataset_path required", "dataset_path")
			return
		}
		if _, err := os.Stat(req.DatasetPath); err != nil {
			writeROCmServeError(w, http.StatusBadRequest, "dataset_path not found: "+err.Error(), "dataset_path")
			return
		}
		adapterRoot, err := rocmAdminSFTAdapterRoot(cfg.AdminSFTAdapterRoot)
		if err != nil {
			writeROCmServeError(w, http.StatusInternalServerError, "adapter root: "+err.Error(), "adapter_root")
			return
		}
		adapterName := strings.TrimSpace(req.AdapterName)
		if adapterName == "" {
			adapterName = rocmAdminSFTDeriveAdapterName(req.ModelPath)
		}
		if strings.Contains(adapterName, "/") || strings.Contains(adapterName, string(filepath.Separator)) || adapterName == "." || adapterName == ".." {
			writeROCmServeError(w, http.StatusBadRequest, "adapter_name must be a directory name", "adapter_name")
			return
		}
		adapterDir := filepath.Join(adapterRoot, adapterName)
		if err := os.MkdirAll(adapterDir, 0o755); err != nil {
			writeROCmServeError(w, http.StatusInternalServerError, "create adapter dir: "+err.Error(), "adapter_dir")
			return
		}
		if registry == nil {
			writeROCmServeError(w, http.StatusInternalServerError, "SFT registry is nil", "sft")
			return
		}
		ctx, cancel := context.WithCancel(context.Background())
		job := &rocmAdminSFTJob{
			JobID:       rocmAdminSFTNewJobID(),
			State:       rocmAdminSFTStatePending,
			ModelPath:   strings.TrimSpace(req.ModelPath),
			DatasetPath: strings.TrimSpace(req.DatasetPath),
			AdapterDir:  adapterDir,
			Backend:     rocmAdminSFTBackend(req, cfg),
			StartedUnix: time.Now().Unix(),
			UpdatedUnix: time.Now().Unix(),
			cancel:      cancel,
		}
		registry.mu.Lock()
		if registry.active != nil {
			registry.mu.Unlock()
			cancel()
			writeROCmServeError(w, http.StatusConflict, "another SFT job is already running", "sft")
			return
		}
		registry.active = job
		registry.mu.Unlock()

		go runROCmAdminSFTJob(ctx, registry, job, req, cfg)
		writeROCmServeJSON(w, http.StatusAccepted, cloneROCmAdminSFTJob(job))
	})
}

func rocmAdminSFTStatusHandler(registry *rocmAdminSFTRegistry) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !rocmServeRequireMethod(w, r, http.MethodGet) {
			return
		}
		jobID := strings.TrimSpace(r.URL.Query().Get("job"))
		snap := registry.snapshot(jobID)
		if snap == nil {
			writeROCmServeError(w, http.StatusNotFound, "no SFT job", "job")
			return
		}
		writeROCmServeJSON(w, http.StatusOK, snap)
	})
}

func rocmAdminSFTStopHandler(registry *rocmAdminSFTRegistry) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !rocmServeRequireMethod(w, r, http.MethodPost) {
			return
		}
		jobID := strings.TrimSpace(r.URL.Query().Get("job"))
		registry.mu.Lock()
		if registry.active == nil || (jobID != "" && registry.active.JobID != jobID) {
			registry.mu.Unlock()
			writeROCmServeError(w, http.StatusNotFound, "no active SFT job for that id", "job")
			return
		}
		if registry.active.cancel != nil {
			registry.active.cancel()
		}
		snap := cloneROCmAdminSFTJob(registry.active)
		registry.mu.Unlock()
		writeROCmServeJSON(w, http.StatusOK, snap)
	})
}

func rocmAdminSFTAdaptersHandler(rootOverride string) http.Handler {
	type adapterEntry struct {
		Name       string `json:"name"`
		Path       string `json:"path"`
		SizeBytes  int64  `json:"size_bytes"`
		ModifiedAt int64  `json:"modified_unix"`
	}
	type adapterList struct {
		Dir      string         `json:"dir"`
		Adapters []adapterEntry `json:"adapters"`
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !rocmServeRequireMethod(w, r, http.MethodGet) {
			return
		}
		root, err := rocmAdminSFTAdapterRoot(rootOverride)
		if err != nil {
			writeROCmServeError(w, http.StatusInternalServerError, "adapter root: "+err.Error(), "adapter_root")
			return
		}
		out := adapterList{Dir: root, Adapters: []adapterEntry{}}
		entries, err := os.ReadDir(root)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				writeROCmServeJSON(w, http.StatusOK, out)
				return
			}
			writeROCmServeError(w, http.StatusInternalServerError, "read adapters: "+err.Error(), "adapters")
			return
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			info, err := entry.Info()
			if err != nil {
				continue
			}
			path := filepath.Join(root, entry.Name())
			out.Adapters = append(out.Adapters, adapterEntry{
				Name:       entry.Name(),
				Path:       path,
				SizeBytes:  rocmAdminSFTDirSizeBytes(path),
				ModifiedAt: info.ModTime().Unix(),
			})
		}
		writeROCmServeJSON(w, http.StatusOK, out)
	})
}

func runROCmAdminSFTJob(ctx context.Context, registry *rocmAdminSFTRegistry, job *rocmAdminSFTJob, req rocmAdminSFTRequest, cfg rocmServeConfig) {
	defer func() {
		registry.mu.Lock()
		registry.last = registry.active
		registry.active = nil
		registry.mu.Unlock()
	}()
	if err := ctx.Err(); err != nil {
		registry.markStopped(job)
		return
	}
	dataFile, err := os.Open(strings.TrimSpace(req.DatasetPath))
	if err != nil {
		registry.failJob(job, "open dataset: "+err.Error())
		return
	}
	defer dataFile.Close()
	dataset, err := rocm.LoadJSONLDataset(dataFile)
	if err != nil {
		registry.failJob(job, "parse dataset: "+err.Error())
		return
	}
	loadOpts := make([]inference.LoadOption, 0, 2)
	backend := rocmAdminSFTBackend(req, cfg)
	if backend != "" && backend != "auto" {
		loadOpts = append(loadOpts, inference.WithBackend(backend))
	}
	if req.ContextLength > 0 {
		loadOpts = append(loadOpts, inference.WithContextLen(req.ContextLength))
	}
	result := inference.LoadModel(strings.TrimSpace(req.ModelPath), loadOpts...)
	if !result.OK {
		registry.failJob(job, "load model: "+result.Error())
		return
	}
	model, ok := result.Value.(inference.TextModel)
	if !ok || model == nil {
		registry.failJob(job, fmt.Sprintf("load returned %T, not inference.TextModel", result.Value))
		return
	}
	defer model.Close()
	trainer, ok := model.(inference.SFTTrainer)
	if !ok {
		registry.failJob(job, fmt.Sprintf("backend model %T does not implement inference.SFTTrainer", model))
		return
	}
	registry.markRunning(job)
	training, err := trainer.TrainSFT(ctx, dataset, rocmAdminSFTTrainingConfig(req, job, backend))
	if err != nil {
		if ctx.Err() != nil {
			registry.markStopped(job)
			return
		}
		registry.failJob(job, err.Error())
		return
	}
	if ctx.Err() != nil {
		registry.markStopped(job)
		return
	}
	registry.markDone(job, training)
}

func rocmAdminSFTTrainingConfig(req rocmAdminSFTRequest, job *rocmAdminSFTJob, backend string) inference.TrainingConfig {
	defaultLoRA := inference.DefaultLoRAConfig()
	rank := pickPositiveInt(req.LoRARank, rocmAdminSFTDefaultLoRARank)
	alpha := pickPositiveInt(req.LoRAAlpha, rocmAdminSFTDefaultLoRAAlpha)
	adapterPath := filepath.Join(job.AdapterDir, "adapter.safetensors")
	labels := map[string]string{
		"backend":                      backend,
		"cli_contract":                 cliContractName,
		"run_id":                       job.JobID,
		"training_stage":               "admin_native_lora_sft_execute",
		"trainer_interface":            "inference.SFTTrainer",
		"dataset_loader":               "rocm.LoadJSONLDataset",
		"adapter_format":               "lora",
		"adapter_dir":                  job.AdapterDir,
		"checkpoint_dir":               job.AdapterDir,
		"output_adapter_path":          adapterPath,
		"metrics_line_protocol_path":   filepath.Join(job.AdapterDir, "metrics.lp"),
		"capture_path":                 filepath.Join(job.AdapterDir, "captures.jsonl"),
		"max_sequence_length":          strconv.Itoa(req.MaxSeqLen),
		"production_requires_env_gate": "false",
		"production_requires_cli_flag": "false",
		"no_python":                    "true",
	}
	if req.LoRADropout > 0 {
		labels["lora_dropout_requested"] = strconv.FormatFloat(req.LoRADropout, 'f', -1, 64)
	}
	return inference.TrainingConfig{
		Epochs:       pickPositiveInt(req.Epochs, rocmAdminSFTDefaultEpochs),
		BatchSize:    pickPositiveInt(req.BatchSize, rocmAdminSFTDefaultBatchSize),
		LearningRate: pickPositiveFloat(req.LearningRate, rocmAdminSFTDefaultLR),
		LoRA: inference.LoRAConfig{
			Rank:       rank,
			Alpha:      float32(alpha),
			TargetKeys: append([]string(nil), defaultLoRA.TargetKeys...),
			BFloat16:   defaultLoRA.BFloat16,
		},
		Labels: labels,
	}
}

func rocmAdminSFTBackend(req rocmAdminSFTRequest, cfg rocmServeConfig) string {
	if strings.TrimSpace(req.Backend) != "" {
		return normalizeBackendName(req.Backend)
	}
	if strings.TrimSpace(cfg.Backend) != "" {
		return normalizeBackendName(cfg.Backend)
	}
	return defaultBackendName
}

func (registry *rocmAdminSFTRegistry) snapshot(jobID string) *rocmAdminSFTJob {
	if registry == nil {
		return nil
	}
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	for _, job := range []*rocmAdminSFTJob{registry.active, registry.last} {
		if job == nil {
			continue
		}
		if jobID == "" || job.JobID == jobID {
			return cloneROCmAdminSFTJob(job)
		}
	}
	return nil
}

func (registry *rocmAdminSFTRegistry) markRunning(job *rocmAdminSFTJob) {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if registry.active != nil && registry.active.JobID == job.JobID {
		registry.active.State = rocmAdminSFTStateRunning
		registry.active.UpdatedUnix = time.Now().Unix()
	}
}

func (registry *rocmAdminSFTRegistry) markDone(job *rocmAdminSFTJob, result *inference.TrainingResult) {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if registry.active == nil || registry.active.JobID != job.JobID {
		return
	}
	now := time.Now().Unix()
	registry.active.State = rocmAdminSFTStateDone
	registry.active.EndedUnix = now
	registry.active.UpdatedUnix = now
	if result != nil {
		registry.active.Metrics = result.Metrics
		registry.active.Adapter = result.Adapter
		registry.active.Checkpoints = append([]inference.StateRef(nil), result.Checkpoints...)
		registry.active.Labels = cloneStringMap(result.Labels)
		registry.active.Step = result.Metrics.Step
		registry.active.Epoch = result.Metrics.Epoch
		registry.active.Samples = result.Metrics.Samples
		registry.active.LastLoss = result.Metrics.Loss
		registry.active.appendLossLocked(result.Metrics)
	}
}

func (registry *rocmAdminSFTRegistry) markStopped(job *rocmAdminSFTJob) {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if registry.active != nil && registry.active.JobID == job.JobID {
		now := time.Now().Unix()
		registry.active.State = rocmAdminSFTStateStopped
		registry.active.EndedUnix = now
		registry.active.UpdatedUnix = now
	}
}

func (registry *rocmAdminSFTRegistry) failJob(job *rocmAdminSFTJob, reason string) {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if registry.active != nil && registry.active.JobID == job.JobID {
		now := time.Now().Unix()
		registry.active.State = rocmAdminSFTStateFailed
		registry.active.Error = reason
		registry.active.EndedUnix = now
		registry.active.UpdatedUnix = now
	}
}

func (job *rocmAdminSFTJob) appendLossLocked(metrics inference.TrainingMetrics) {
	if job == nil || metrics.Loss == 0 {
		return
	}
	sample := rocmAdminSFTLossSample{
		Step:  metrics.Step,
		Epoch: metrics.Epoch,
		Loss:  metrics.Loss,
		TS:    time.Now().Unix(),
	}
	if len(job.Loss) >= rocmAdminSFTLossRingSize {
		job.Loss = append(job.Loss[1:], sample)
		return
	}
	job.Loss = append(job.Loss, sample)
}

func cloneROCmAdminSFTJob(src *rocmAdminSFTJob) *rocmAdminSFTJob {
	if src == nil {
		return nil
	}
	out := *src
	out.cancel = nil
	if len(src.Loss) > 0 {
		out.Loss = append([]rocmAdminSFTLossSample(nil), src.Loss...)
	}
	if len(src.Checkpoints) > 0 {
		out.Checkpoints = append([]inference.StateRef(nil), src.Checkpoints...)
	}
	out.Labels = cloneStringMap(src.Labels)
	if len(src.Adapter.TargetKeys) > 0 {
		out.Adapter.TargetKeys = append([]string(nil), src.Adapter.TargetKeys...)
	}
	out.Adapter.Labels = cloneStringMap(src.Adapter.Labels)
	return &out
}

func rocmAdminSFTAdapterRoot(rootOverride string) (string, error) {
	if strings.TrimSpace(rootOverride) != "" {
		return filepath.Clean(rootOverride), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(home) == "" {
		return "", errors.New("home directory is empty")
	}
	return filepath.Join(home, "Lethean", "data", "adapters"), nil
}

func rocmAdminSFTDeriveAdapterName(modelPath string) string {
	base := filepath.Base(filepath.Clean(strings.TrimSpace(modelPath)))
	if base == "" || base == "." || base == string(filepath.Separator) {
		base = "adapter"
	}
	return base + "-" + strconv.FormatInt(time.Now().Unix(), 10)
}

func rocmAdminSFTNewJobID() string {
	return "sft-" + strconv.FormatInt(time.Now().UnixNano(), 36)
}

func rocmAdminSFTDirSizeBytes(dir string) int64 {
	var total int64
	_ = filepath.Walk(dir, func(_ string, info os.FileInfo, err error) error {
		if err != nil || info == nil || info.IsDir() {
			return nil
		}
		total += info.Size()
		return nil
	})
	return total
}

func pickPositiveInt(value, fallback int) int {
	if value > 0 {
		return value
	}
	return fallback
}

func pickPositiveFloat(value, fallback float64) float64 {
	if value > 0 {
		return value
	}
	return fallback
}
