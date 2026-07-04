// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"dappco.re/go/rocm/daemon"
)

type daemonRuntimeConfig struct {
	SocketPath       string
	Models           map[string]string
	DefaultModelName string
	DefaultMaxTokens int
	ContextLen       int
	GPULayers        int
	ParallelSlots    int
	KVCache          string
	AdapterPath      string
}

func loadDaemonRuntimeConfig(explicitPath string) (daemonRuntimeConfig, error) {
	cfg := daemonRuntimeConfig{
		Models:    map[string]string{},
		GPULayers: -1,
	}
	configPath := strings.TrimSpace(explicitPath)
	if configPath == "" {
		configPath = defaultDaemonConfigPath()
	}
	if configPath != "" {
		if err := readDaemonConfigFile(configPath, &cfg); err != nil {
			if explicitPath != "" || !os.IsNotExist(err) {
				return cfg, err
			}
		}
	}
	if err := applyDaemonEnvFallbacks(&cfg); err != nil {
		return cfg, err
	}
	if strings.TrimSpace(cfg.DefaultModelName) == "" {
		cfg.DefaultModelName = daemon.DefaultModelName
	}
	return cfg, nil
}

func defaultDaemonConfigPath() string {
	if configHome := os.Getenv("XDG_CONFIG_HOME"); strings.TrimSpace(configHome) != "" {
		return filepath.Join(configHome, "ofm", "lthn.toml")
	}
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		return ""
	}
	return filepath.Join(home, ".config", "ofm", "lthn.toml")
}

func readDaemonConfigFile(path string, cfg *daemonRuntimeConfig) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	section := ""
	scanner := bufio.NewScanner(file)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSpace(stripDaemonConfigComment(scanner.Text()))
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			section = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(line, "["), "]"))
			continue
		}

		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			return fmt.Errorf("%s:%d: expected key = value", path, lineNumber)
		}
		key := strings.TrimSpace(parts[0])
		value, err := parseDaemonConfigValue(strings.TrimSpace(parts[1]))
		if err != nil {
			return fmt.Errorf("%s:%d: %w", path, lineNumber, err)
		}

		switch section {
		case "models":
			if key != "" && value != "" {
				if cfg.Models == nil {
					cfg.Models = map[string]string{}
				}
				cfg.Models[key] = value
			}
		case "":
			if err := applyDaemonConfigRootKey(cfg, key, value, path, lineNumber); err != nil {
				return err
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return nil
}

func applyDaemonConfigRootKey(cfg *daemonRuntimeConfig, key, value, path string, lineNumber int) error {
	switch strings.TrimSpace(key) {
	case "socket", "socket_path":
		cfg.SocketPath = value
	case "model", "model_path", "default_model":
		if value != "" {
			if cfg.Models == nil {
				cfg.Models = map[string]string{}
			}
			cfg.Models[daemon.DefaultModelName] = value
		}
	case "model_name", "default_model_name":
		cfg.DefaultModelName = value
	case "max_tokens", "default_max_tokens":
		n, err := parseDaemonConfigNonNegativeInt(value)
		if err != nil {
			return fmt.Errorf("%s:%d: %s: %w", path, lineNumber, key, err)
		}
		cfg.DefaultMaxTokens = n
	case "context", "context_len", "context_length":
		n, err := parseDaemonConfigNonNegativeInt(value)
		if err != nil {
			return fmt.Errorf("%s:%d: %s: %w", path, lineNumber, key, err)
		}
		cfg.ContextLen = n
	case "gpu_layers", "gpu-layers":
		n, err := strconv.Atoi(strings.TrimSpace(value))
		if err != nil {
			return fmt.Errorf("%s:%d: %s: %w", path, lineNumber, key, err)
		}
		cfg.GPULayers = n
	case "parallel", "parallel_slots":
		n, err := parseDaemonConfigNonNegativeInt(value)
		if err != nil {
			return fmt.Errorf("%s:%d: %s: %w", path, lineNumber, key, err)
		}
		cfg.ParallelSlots = n
	case "kv_cache", "cache_mode", "device_kv_mode":
		cfg.KVCache = value
	case "adapter", "adapter_path":
		cfg.AdapterPath = value
	}
	return nil
}

func parseDaemonConfigNonNegativeInt(value string) (int, error) {
	n, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil {
		return 0, err
	}
	if n < 0 {
		return 0, fmt.Errorf("must be >= 0")
	}
	return n, nil
}

func parseDaemonConfigValue(raw string) (string, error) {
	if raw == "" {
		return "", nil
	}
	if strings.HasPrefix(raw, "\"") {
		return strconv.Unquote(raw)
	}
	if strings.HasPrefix(raw, "'") && strings.HasSuffix(raw, "'") {
		return strings.TrimSuffix(strings.TrimPrefix(raw, "'"), "'"), nil
	}
	if strings.HasPrefix(raw, "'") {
		return "", fmt.Errorf("unterminated single-quoted value")
	}
	return strings.TrimSpace(raw), nil
}

func stripDaemonConfigComment(line string) string {
	var quote rune
	escaped := false
	for i, r := range line {
		if escaped {
			escaped = false
			continue
		}
		if quote == '"' && r == '\\' {
			escaped = true
			continue
		}
		if quote != 0 {
			if r == quote {
				quote = 0
			}
			continue
		}
		if r == '"' || r == '\'' {
			quote = r
			continue
		}
		if r == '#' {
			return line[:i]
		}
	}
	return line
}

func applyDaemonEnvFallbacks(cfg *daemonRuntimeConfig) error {
	if cfg.Models == nil {
		cfg.Models = map[string]string{}
	}
	if cfg.SocketPath == "" {
		cfg.SocketPath = firstDaemonEnv("LTHN_SOCKET_PATH", "VIOLET_SOCKET_PATH")
	}
	if cfg.DefaultModelName == "" {
		cfg.DefaultModelName = firstDaemonEnv("LTHN_DEFAULT_MODEL_NAME", "VIOLET_DEFAULT_MODEL_NAME")
	}
	applyDaemonModelEnv(cfg, daemon.DefaultModelName, "LTHN_MODEL_PATH", "LTHN_DEFAULT_MODEL_PATH", "VIOLET_MODEL_PATH")
	applyDaemonModelEnv(cfg, "embed", "LTHN_EMBED_MODEL_PATH", "VIOLET_EMBED_MODEL_PATH")
	applyDaemonModelEnv(cfg, "score", "LTHN_SCORE_MODEL_PATH", "VIOLET_SCORE_MODEL_PATH")
	applyDaemonModelEnv(cfg, "generate", "LTHN_GENERATE_MODEL_PATH", "VIOLET_GENERATE_MODEL_PATH")
	if cfg.DefaultMaxTokens == 0 {
		if raw := firstDaemonEnv("LTHN_DEFAULT_MAX_TOKENS", "LTHN_MAX_TOKENS"); raw != "" {
			n, err := parseDaemonConfigNonNegativeInt(raw)
			if err != nil {
				return fmt.Errorf("daemon env max tokens: %w", err)
			}
			cfg.DefaultMaxTokens = n
		}
	}
	return nil
}

func applyDaemonModelEnv(cfg *daemonRuntimeConfig, name string, envNames ...string) {
	if cfg.Models[name] != "" {
		return
	}
	if value := firstDaemonEnv(envNames...); value != "" {
		cfg.Models[name] = value
	}
}

func firstDaemonEnv(names ...string) string {
	for _, name := range names {
		if value := strings.TrimSpace(os.Getenv(name)); value != "" {
			return value
		}
	}
	return ""
}

func daemonFlagSet(fs *flag.FlagSet) map[string]bool {
	out := map[string]bool{}
	fs.Visit(func(f *flag.Flag) {
		out[f.Name] = true
	})
	return out
}
