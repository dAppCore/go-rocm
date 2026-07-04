// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"dappco.re/go/inference"
	rocm "dappco.re/go/rocm"
)

const (
	// tuneDraftBlockLabel carries the winning MTP block inside persisted
	// inference.TuningProfile candidates. go-mlx tune writes this label;
	// lthn-rocm generate and serve read the same contract.
	tuneDraftBlockLabel = "mtp_draft_block"
)

type rocmServeDraftSelection struct {
	Detection        rocm.DraftDetection
	DraftBlock       int
	DraftBlockSource string
}

func resolveROCmServeDraftSelection(ctx context.Context, modelPath, draftPath string, detect bool, flagBlock int, noAutoProfile bool, profileDir string) rocmServeDraftSelection {
	detection := resolveROCmDraft(modelPath, draftPath, detect)
	block, source := resolveROCmServeDraftBlock(ctx, detection, modelPath, flagBlock, noAutoProfile, profileDir)
	return rocmServeDraftSelection{
		Detection:        detection,
		DraftBlock:       block,
		DraftBlockSource: source,
	}
}

func resolveROCmServeDraftBlock(ctx context.Context, detection rocm.DraftDetection, modelPath string, flagBlock int, noAutoProfile bool, profileDir string) (int, string) {
	if !detection.Active() || flagBlock > 0 || noAutoProfile {
		return flagBlock, ""
	}
	dir := strings.TrimSpace(profileDir)
	if dir == "" {
		dir = standardTuningProfileDir()
	}
	if strings.TrimSpace(dir) == "" {
		return 0, ""
	}
	machineHash, err := currentROCmMachineProfileHash(ctx)
	if err != nil {
		machineHash = ""
	}
	block, path := loadTunedDraftBlock(dir, modelPath, machineHash)
	if block <= 0 {
		return 0, ""
	}
	return block, "tuned: " + path
}

func standardTuningProfileDir() string {
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		home = os.Getenv("HOME")
	}
	if strings.TrimSpace(home) == "" {
		return ""
	}
	return filepath.Join(home, "Lethean", "data", "tuning")
}

func currentROCmMachineProfileHash(ctx context.Context) (string, error) {
	return rocm.CurrentMachineProfileHash(ctx)
}

func loadTunedDraftBlock(dir, modelPath, machineHash string) (int, string) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return 0, ""
	}
	bestBlock, bestPath := 0, ""
	var bestCreated int64 = -1
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var profile inference.TuningProfile
		if err := json.Unmarshal(data, &profile); err != nil {
			continue
		}
		if !tunedModelPathMatches(rocm.ModelIdentityFromTuningProfile(profile).Path, modelPath) {
			continue
		}
		if profile.Key.MachineHash != "" {
			if machineHash == "" || profile.Key.MachineHash != machineHash {
				continue
			}
		}
		block, err := strconv.Atoi(strings.TrimSpace(profile.Candidate.Labels[tuneDraftBlockLabel]))
		if err != nil || block < 2 || block > 8 {
			continue
		}
		if profile.CreatedAtUnix > bestCreated {
			bestCreated = profile.CreatedAtUnix
			bestBlock = block
			bestPath = path
		}
	}
	return bestBlock, bestPath
}

func tunedModelPathMatches(profilePath, modelPath string) bool {
	profilePath = strings.TrimSpace(profilePath)
	modelPath = strings.TrimSpace(modelPath)
	if profilePath == "" || modelPath == "" {
		return false
	}
	profilePath = filepath.Clean(profilePath)
	modelPath = filepath.Clean(modelPath)
	if profilePath == modelPath {
		return true
	}
	if filepath.Ext(profilePath) != "" && filepath.Dir(profilePath) == modelPath {
		return true
	}
	if filepath.Ext(modelPath) != "" && filepath.Dir(modelPath) == profilePath {
		return true
	}
	return false
}

func loadROCmServeReloadProfile(path string) (inference.TuningProfile, bool, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return inference.TuningProfile{}, false, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return inference.TuningProfile{}, false, nil
		}
		return inference.TuningProfile{}, false, err
	}
	var profile inference.TuningProfile
	if err := json.Unmarshal(data, &profile); err != nil {
		return inference.TuningProfile{}, false, err
	}
	return profile, true, nil
}

func writeROCmTuningProfile(path string, profile inference.TuningProfile) error {
	return rocm.WriteTuningProfile(path, profile)
}
