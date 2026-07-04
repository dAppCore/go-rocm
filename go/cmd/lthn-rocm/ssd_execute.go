// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"dappco.re/go/inference"
	rocm "dappco.re/go/rocm"
	"dappco.re/go/rocm/score"
)

type ssdTraceJSONLRecord struct {
	Step     int               `json:"step"`
	Prompt   string            `json:"prompt"`
	Response string            `json:"response"`
	Labels   map[string]string `json:"labels,omitempty"`
}

type ssdScoreJSONLRecord struct {
	Step     int              `json:"step"`
	Prompt   string           `json:"prompt"`
	Response string           `json:"response"`
	LEK      float64          `json:"lek,omitempty"`
	Score    score.DiffResult `json:"score"`
}

func runSSDExecute(ctx context.Context, opts ssdCommandOptions, stdout, stderr io.Writer) int {
	runID := strings.TrimSpace(opts.RunID)
	if runID == "" {
		runID = "ssd-" + time.Now().Format("20060102-150405")
	}
	checkpointDir := strings.TrimSpace(opts.CheckpointDir)
	if checkpointDir == "" {
		checkpointDir = runID
	}
	capturePath := strings.TrimSpace(opts.CapturePath)
	if capturePath == "" {
		capturePath = filepath.Join(checkpointDir, "ssd-captures.jsonl")
	}
	scorePath := ""
	if opts.ScoreSamples {
		scorePath = filepath.Join(filepath.Dir(capturePath), "ssd-samples-score.jsonl")
	}

	dataFile, err := os.Open(strings.TrimSpace(opts.DataPath))
	if err != nil {
		fmt.Fprintf(stderr, "%s ssd: prompt data unreadable: %v\n", cliName(), err)
		return 1
	}
	defer dataFile.Close()
	dataset, err := rocm.LoadJSONLDataset(dataFile)
	if err != nil {
		fmt.Fprintf(stderr, "%s ssd: prompt data parse: %v\n", cliName(), err)
		return 1
	}

	loadOpts := make([]inference.LoadOption, 0, 2)
	if backend := normalizeBackendName(opts.BackendName); backend != "" && backend != "auto" {
		loadOpts = append(loadOpts, inference.WithBackend(backend))
	}
	if opts.ContextOverride > 0 {
		loadOpts = append(loadOpts, inference.WithContextLen(opts.ContextOverride))
	}
	result := inference.LoadModel(strings.TrimSpace(opts.ModelPath), loadOpts...)
	if !result.OK {
		fmt.Fprintf(stderr, "%s ssd: load: %s\n", cliName(), result.Error())
		return 1
	}
	model, ok := result.Value.(inference.TextModel)
	if !ok || model == nil {
		fmt.Fprintf(stderr, "%s ssd: load returned %T, not inference.TextModel\n", cliName(), result.Value)
		return 1
	}
	defer model.Close()

	cfg := rocm.DefaultSimpleSelfDistillationConfig()
	cfg.SampleMaxTokens = opts.SampleMaxTokens
	cfg.SampleTemperature = opts.SampleTemperature
	cfg.SampleTopK = opts.SampleTopK
	cfg.SampleTopP = opts.SampleTopP
	cfg.SampleMinP = opts.SampleMinP
	cfg.RepetitionPenalty = opts.RepetitionPenalty
	cfg.FilterShortestPct = opts.FilterShortestPercent
	ssd, err := rocm.RunModelSimpleSelfDistillation(ctx, model, dataset, cfg)
	if err != nil {
		fmt.Fprintf(stderr, "%s ssd: self-distillation: %v\n", cliName(), err)
		return 1
	}
	if err := writeSSDTraceSidecar(capturePath, ssd.Samples); err != nil {
		fmt.Fprintf(stderr, "%s ssd: write trace: %v\n", cliName(), err)
		return 1
	}

	var scoreMean float64
	if opts.ScoreSamples {
		scoreMean, err = writeSSDScoreSidecar(scorePath, ssd.Samples)
		if err != nil {
			fmt.Fprintf(stderr, "%s ssd: write sample scores: %v\n", cliName(), err)
			return 1
		}
	}

	fmt.Fprintf(stdout, "self-samples %d  sample-temp %.2f  top-k %d  top-p %.2f  min-p %.2f\n",
		len(ssd.Samples), ssd.SampleTemperature, ssd.SampleTopK, ssd.SampleTopP, ssd.SampleMinP)
	fmt.Fprintf(stdout, "ssd trace %s  (the lab picks steps from this", capturePath)
	if opts.ScoreSamples {
		fmt.Fprintf(stdout, " + %s", scorePath)
	}
	fmt.Fprintln(stdout, ")")
	if opts.ScoreSamples {
		fmt.Fprintf(stdout, "sample-score mean %.2f over %d scored\n", scoreMean, len(ssd.Samples))
	}
	fmt.Fprintf(stdout, "next: refine the trace in the lab, then  %s sft --data <artifact> --model %s\n", cliName(), strings.TrimSpace(opts.ModelPath))
	return 0
}

func writeSSDTraceSidecar(path string, samples []rocm.SimpleSelfDistillationSample) error {
	file, err := createSSDJSONLSidecar(path)
	if err != nil {
		return err
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	for index, sample := range samples {
		if err := encoder.Encode(ssdTraceJSONLRecord{
			Step:     index,
			Prompt:   sample.Prompt,
			Response: sample.Response,
			Labels:   sample.Labels,
		}); err != nil {
			return err
		}
	}
	return nil
}

func writeSSDScoreSidecar(path string, samples []rocm.SimpleSelfDistillationSample) (float64, error) {
	file, err := createSSDJSONLSidecar(path)
	if err != nil {
		return 0, err
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	sum := 0.0
	for index, sample := range samples {
		scored := score.ScorePair(sample.Prompt, sample.Response)
		lek := 0.0
		if scored.Response.LEK != nil {
			lek = scored.Response.LEK.LEKScore
		}
		sum += lek
		if err := encoder.Encode(ssdScoreJSONLRecord{
			Step:     index,
			Prompt:   sample.Prompt,
			Response: sample.Response,
			LEK:      lek,
			Score:    scored,
		}); err != nil {
			return 0, err
		}
	}
	if len(samples) == 0 {
		return 0, nil
	}
	return sum / float64(len(samples)), nil
}

func createSSDJSONLSidecar(path string) (*os.File, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, fmt.Errorf("sidecar path is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	return os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
}
