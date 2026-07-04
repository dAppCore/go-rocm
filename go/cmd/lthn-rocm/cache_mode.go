// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"fmt"
	"strings"
)

const (
	rocmCLIKVCacheFlagUsage = "KV cache mode (fp16, q8, kq8vq4/k-q8-v-q4, paged, turboquant, disk-l2; empty = native default)"
	rocmCLIKVCacheModes     = "fp16, q8, kq8vq4, k-q8-v-q4, paged, turboquant, disk-l2"
	rocmCLIKVStorageModes   = "fp16, bf16"
)

type rocmCLIKVCacheModeDecision struct {
	Mode    string
	Code    int
	Message string
}

func resolveROCmCLIKVCacheMode(command, raw string) rocmCLIKVCacheModeDecision {
	mode, ok := normalizeROCmCLIKVCacheMode(raw)
	if strings.TrimSpace(raw) == "" {
		return rocmCLIKVCacheModeDecision{}
	}
	if !ok {
		return rocmCLIKVCacheModeDecision{
			Code:    2,
			Message: fmt.Sprintf("%s %s: unsupported -kv-cache %q (supported: %s)\n", cliName(), command, raw, rocmCLIKVCacheModes),
		}
	}
	if !rocmCLIKVCacheModeBound(mode) {
		return rocmCLIKVCacheModeDecision{
			Mode: mode,
			Code: 1,
			Message: fmt.Sprintf("%s %s: -kv-cache %s is recognized (%s) but native CLI load cannot bind this planned runtime mode yet; refusing backend-default fallback\n",
				cliName(), command, mode, rocmCLIKVCacheModeStatus(mode)),
		}
	}
	return rocmCLIKVCacheModeDecision{Mode: mode}
}

func validateROCmCLIKVCacheMode(command, raw string) (int, string) {
	decision := resolveROCmCLIKVCacheMode(command, raw)
	return decision.Code, decision.Message
}

func resolveROCmCLIKVStorageMode(command, raw string) rocmCLIKVCacheModeDecision {
	mode, ok := normalizeROCmCLIKVStorageMode(raw)
	if strings.TrimSpace(raw) == "" {
		return rocmCLIKVCacheModeDecision{}
	}
	if !ok {
		return rocmCLIKVCacheModeDecision{
			Code:    2,
			Message: fmt.Sprintf("%s %s: unsupported -kv-storage %q (supported: %s)\n", cliName(), command, raw, rocmCLIKVStorageModes),
		}
	}
	if !rocmCLIKVStorageModeBound(mode) {
		return rocmCLIKVCacheModeDecision{
			Mode: mode,
			Code: 1,
			Message: fmt.Sprintf("%s %s: -kv-storage %s is recognized but native CLI load cannot bind this retained KV storage dtype yet; refusing backend-default fallback\n",
				cliName(), command, mode),
		}
	}
	return rocmCLIKVCacheModeDecision{Mode: mode}
}

func resolveROCmCLILoadModeForCacheAndStorage(command, cacheMode, storageMode string) rocmCLIKVCacheModeDecision {
	cacheMode = strings.TrimSpace(cacheMode)
	storageMode = strings.TrimSpace(storageMode)
	switch {
	case cacheMode == "":
		return rocmCLIKVCacheModeDecision{Mode: storageMode}
	case storageMode == "":
		return rocmCLIKVCacheModeDecision{Mode: cacheMode}
	case cacheMode == storageMode:
		return rocmCLIKVCacheModeDecision{Mode: cacheMode}
	default:
		return rocmCLIKVCacheModeDecision{
			Code: 2,
			Message: fmt.Sprintf("%s %s: -kv-storage %s conflicts with -kv-cache %s; pass matching native values or omit -kv-storage\n",
				cliName(), command, storageMode, cacheMode),
		}
	}
}

func resolveROCmCLIPipelineMode(command string, pipeline bool) rocmCLIKVCacheModeDecision {
	if pipeline {
		return rocmCLIKVCacheModeDecision{}
	}
	return rocmCLIKVCacheModeDecision{
		Code: 1,
		Message: fmt.Sprintf("%s %s: -pipeline=false is recognized but native ROCm CLI cannot force serial decode yet; refusing backend-default fallback\n",
			cliName(), command),
	}
}

func normalizeROCmCLIKVCacheMode(raw string) (string, bool) {
	mode := strings.ToLower(strings.TrimSpace(raw))
	mode = strings.ReplaceAll(mode, "_", "-")
	switch mode {
	case "fp16", "q8", "paged", "turboquant", "disk-l2":
		return mode, true
	case "kq8vq4", "k-q8-v-q4":
		return "k-q8-v-q4", true
	default:
		return "", false
	}
}

func normalizeROCmCLIKVStorageMode(raw string) (string, bool) {
	mode := strings.ToLower(strings.TrimSpace(raw))
	mode = strings.ReplaceAll(mode, "_", "-")
	switch mode {
	case "fp16", "f16":
		return "fp16", true
	case "bf16", "bfloat16":
		return "bf16", true
	default:
		return "", false
	}
}

func rocmCLIKVCacheModeStatus(mode string) string {
	switch mode {
	case "fp16", "q8", "k-q8-v-q4":
		return "native_device_kv_bound"
	case "paged", "turboquant", "disk-l2":
		return "planned_runtime_mode"
	default:
		return "unknown"
	}
}

func rocmCLIKVStorageModeBound(mode string) bool {
	return mode == "fp16"
}

func rocmCLIKVCacheModeBound(mode string) bool {
	switch mode {
	case "fp16", "q8", "k-q8-v-q4":
		return true
	default:
		return false
	}
}
