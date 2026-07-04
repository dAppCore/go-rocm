// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
	"dappco.re/go/inference/state/filestore"
	rocm "dappco.re/go/rocm"
)

func runGenerateState(ctx context.Context, modelPath, prompt, name, storePath string, maxTokens int, temp float32, enableThinking bool, rocmLoadCfg rocm.ROCmLoadConfig, loadOpts []inference.LoadOption, draftDetection rocm.DraftDetection, draftBlock int, draftBlockSource string, stdout, stderr io.Writer) int {
	name = strings.TrimSpace(name)
	if name == "" {
		fmt.Fprintf(stderr, "%s generate: -state requires a non-empty name\n", cliName())
		return 2
	}
	if storePath == "" {
		defaultPath, err := defaultROCmStateStorePath()
		if err != nil {
			fmt.Fprintf(stderr, "%s generate: resolve default -state-store: %v\n", cliName(), err)
			return 1
		}
		storePath = defaultPath
	}
	store, err := openOrCreateROCmStateStore(ctx, storePath)
	if err != nil {
		fmt.Fprintf(stderr, "%s generate: state store %s: %v\n", cliName(), storePath, err)
		return 1
	}
	defer store.Close()

	var model inference.TextModel
	if draftDetection.Active() {
		fmt.Fprintf(stderr, "%s generate: reactive MTP drafter resolved for state: %s (%s), block %s\n",
			cliName(), draftDetection.DraftPath, draftDetection.Note, resolvedROCmDraftBlockLabel(draftBlock)+resolvedROCmDraftBlockSourceSuffix(draftBlockSource))
		loaded, err := rocm.LoadAttachedDrafterPairAsTextModelBlockWithConfig(modelPath, draftDetection.DraftPath, draftBlock, rocmLoadCfg, loadOpts...)
		if err != nil {
			fmt.Fprintf(stderr, "%s generate: native ROCm retained-state drafter execution is pending: %v\n", cliName(), err)
			return 1
		}
		model = loaded
		reportGenerateDrafterStatus(stderr, model)
	} else {
		loaded, err := loadROCmCLITextModel(modelPath, rocmLoadCfg, loadOpts)
		if err != nil {
			fmt.Fprintf(stderr, "%s generate: load: %v\n", cliName(), err)
			return 1
		}
		model = loaded
	}
	defer model.Close()

	session, ok := model.(inference.AgentMemorySession)
	if !ok || session == nil {
		fmt.Fprintf(stderr, "%s generate: backend model %T does not implement AgentMemorySession for -state\n", cliName(), model)
		return 1
	}

	entryURI := "rocm://agent/" + name
	indexURI := entryURI + "/index"
	var wake *inference.AgentMemoryWakeResult
	var wakeDur time.Duration
	if _, idxErr := state.ResolveURI(ctx, store, indexURI); idxErr == nil {
		start := time.Now()
		wake, err = session.WakeState(ctx, inference.AgentMemoryWakeRequest{
			Store:    store,
			EntryURI: entryURI,
			IndexURI: indexURI,
		})
		if err != nil {
			fmt.Fprintf(stderr, "%s generate: wake %s: %v\n", cliName(), name, err)
			return 1
		}
		wakeDur = time.Since(start)
	} else {
		var notFound *state.URIChunkNotFoundError
		if !errors.As(idxErr, &notFound) {
			fmt.Fprintf(stderr, "%s generate: state index %s: %v\n", cliName(), indexURI, idxErr)
			return 1
		}
	}

	start := time.Now()
	count := 0
	for tok := range model.Chat(ctx, []inference.Message{{Role: "user", Content: prompt}},
		rocmCLIGenerateOptions(maxTokens, temp, enableThinking)...,
	) {
		fmt.Fprint(stdout, tok.Text)
		count++
	}
	decodeDur := time.Since(start)
	if err := model.Err(); err != nil {
		fmt.Fprintf(stderr, "%s generate: %v\n", cliName(), err)
		return 1
	}

	sleepStart := time.Now()
	sleep, err := session.SleepState(ctx, inference.AgentMemorySleepRequest{
		Store:             store,
		EntryURI:          entryURI,
		IndexURI:          indexURI,
		ParentEntryURI:    parentEntryURI(wake),
		ParentIndexURI:    parentIndexURI(wake),
		Title:             name,
		ReuseParentPrefix: wake != nil,
		Labels: map[string]string{
			"cli":      cliName(),
			"contract": cliContractName,
			"runtime":  "rocm",
		},
	})
	if err != nil {
		fmt.Fprintf(stderr, "%s generate: sleep %s: %v\n", cliName(), name, err)
		return 1
	}
	sleepDur := time.Since(sleepStart)

	fmt.Fprintln(stdout)
	fmt.Fprintln(stdout)
	if wake != nil {
		fmt.Fprintf(stdout, "turn: woke %d prefix tokens in %dms (durable KV restore)\n", wake.PrefixTokens, wakeDur.Milliseconds())
	} else {
		fmt.Fprintln(stdout, "turn: fresh state")
	}
	if decodeDur > 0 && count > 0 {
		fmt.Fprintf(stdout, "decode %.1f tok/s  (%d tok / %.3fs)\n", float64(count)/decodeDur.Seconds(), count, decodeDur.Seconds())
	}
	printGenerateAttachedDrafterMetrics(stdout, model)
	fmt.Fprintf(stdout, "slept %d tokens -> %d blocks in %dms\n", sleep.TokenCount, sleep.BlocksWritten, sleepDur.Milliseconds())
	fmt.Fprintf(stdout, "state: %s (%s)\n", name, storePath)
	return 0
}

func defaultROCmStateStorePath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(home) == "" {
		return "", errors.New("home directory is empty")
	}
	return filepath.Join(home, "Lethean", "data", "state", "agent.kv"), nil
}

func defaultROCmGenerateStateName(modelPath string) string {
	clean := strings.TrimSpace(modelPath)
	if clean == "" {
		clean = "model"
	}
	base := filepath.Base(filepath.Clean(clean))
	base = sanitizeROCmGenerateStateName(base)
	hash := fnv.New64a()
	_, _ = hash.Write([]byte(clean))
	return "generate-" + base + "-" + strconv.FormatUint(hash.Sum64(), 16)
}

func sanitizeROCmGenerateStateName(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	var builder strings.Builder
	lastDash := false
	for i := 0; i < len(value); i++ {
		c := value[i]
		allowed := (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.'
		if allowed {
			builder.WriteByte(c)
			lastDash = false
			continue
		}
		if !lastDash {
			builder.WriteByte('-')
			lastDash = true
		}
	}
	name := strings.Trim(builder.String(), "-.")
	if name == "" {
		name = "model"
	}
	if len(name) > 48 {
		name = strings.Trim(name[:48], "-.")
		if name == "" {
			name = "model"
		}
	}
	return name
}

func rocmCLIBackendUsesDefaultState(backend string) bool {
	switch normalizeBackendName(backend) {
	case defaultBackendName, "auto":
		return true
	default:
		return false
	}
}

func defaultROCmConversationStateStorePath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(home) == "" {
		return "", errors.New("home directory is empty")
	}
	return filepath.Join(home, "Lethean", "data", "state", "conversations.kv"), nil
}

func openOrCreateROCmStateStore(ctx context.Context, path string) (*filestore.Store, error) {
	if strings.TrimSpace(path) == "" {
		return nil, errors.New("path is required")
	}
	if _, err := os.Stat(path); err == nil {
		return filestore.Open(ctx, path)
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}
	return filestore.Create(ctx, path)
}

func parentEntryURI(wake *inference.AgentMemoryWakeResult) string {
	if wake == nil {
		return ""
	}
	return wake.Entry.URI
}

func parentIndexURI(wake *inference.AgentMemoryWakeResult) string {
	if wake == nil {
		return ""
	}
	return wake.Entry.IndexURI
}

func parentBundleURI(wake *inference.AgentMemoryWakeResult) string {
	if wake == nil {
		return ""
	}
	return wake.Entry.BundleURI
}
