// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"fmt"
	"strings"

	"dappco.re/go/inference"
	rocm "dappco.re/go/rocm"
)

func rocmCLILoadConfigForKVCache(mode string) rocm.ROCmLoadConfig {
	mode = strings.TrimSpace(mode)
	if mode == "" {
		return rocm.ROCmLoadConfig{}
	}
	return rocm.ROCmLoadConfig{CacheMode: mode, DeviceKVMode: mode}
}

func rocmCLILoadConfigActive(cfg rocm.ROCmLoadConfig) bool {
	return strings.TrimSpace(cfg.CacheMode) != "" || strings.TrimSpace(cfg.DeviceKVMode) != ""
}

func validateROCmCLILoadConfigBackend(command, backend string, cfg rocm.ROCmLoadConfig) error {
	if !rocmCLILoadConfigActive(cfg) {
		return nil
	}
	backend = normalizeBackendName(backend)
	if backend == "" || backend == "auto" || backend == defaultBackendName {
		return nil
	}
	return fmt.Errorf("%s %s: ROCm-native KV load config cannot be used with -backend %s", cliName(), command, backend)
}

func loadROCmCLITextModel(path string, cfg rocm.ROCmLoadConfig, loadOpts []inference.LoadOption) (inference.TextModel, error) {
	if err := rocmCLIPreflightTextGeneration(path, loadOpts); err != nil {
		return nil, err
	}
	if rocmCLILoadConfigActive(cfg) {
		return rocm.LoadModelWithConfig(path, cfg, loadOpts...)
	}
	result := inference.LoadModel(path, loadOpts...)
	if !result.OK {
		return nil, fmt.Errorf("%s", result.Error())
	}
	loaded, ok := result.Value.(inference.TextModel)
	if !ok || loaded == nil {
		return nil, fmt.Errorf("load returned %T, not inference.TextModel", result.Value)
	}
	return loaded, nil
}

func rocmCLIPreflightTextGeneration(path string, loadOpts []inference.LoadOption) error {
	loadCfg := inference.ApplyLoadOpts(loadOpts)
	backend := normalizeBackendName(loadCfg.Backend)
	if backend != defaultBackendName && backend != "auto" {
		return nil
	}
	inspection, err := inspectModelPack(context.Background(), defaultBackendName, path)
	if err != nil {
		return nil
	}
	status, ok := rocm.ROCmModelLoadStatusForInspection(inspection)
	if !ok || status.Status == "" {
		return nil
	}
	if status.Status == rocm.ROCmModelLoadStandaloneNative && status.TextGenerate {
		return nil
	}
	architecture := firstROCmCLINonEmptyString(status.Architecture, inspection.Model.Architecture, "unknown")
	family := firstROCmCLINonEmptyString(status.Family, architecture)
	target := firstROCmCLINonEmptyString(status.Target, "unknown")
	reason := strings.TrimSpace(status.Reason)
	if reason == "" {
		reason = "registry profile does not advertise standalone text generation"
	}
	return fmt.Errorf("registry load status blocks standalone text generation for %s/%s: engine_load_status=%s engine_load_target=%s engine_load_text_generate=%t (%s)", family, architecture, status.Status, target, status.TextGenerate, reason)
}

func firstROCmCLINonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
