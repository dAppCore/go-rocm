// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"sync"
)

const (
	socketFileMode os.FileMode = 0o600
	socketDirMode  os.FileMode = 0o700
	maxFrameBytes              = 16 * 1024 * 1024
)

var (
	errSocketPathRequired = errors.New("socket path is required")
	errXDGRuntimeDirUnset = errors.New("XDG_RUNTIME_DIR is not set")
)

type ServerConfig struct {
	SocketPath string
	Registry   *Registry

	ModelPaths            map[string]string
	CacheBackend          CacheBackend
	CacheEntryBackend     CacheEntryBackend
	CancelBackend         CancelBackend
	EmbedBackend          EmbedBackend
	EngineFeaturesBackend EngineFeaturesBackend
	GenerateBackend       GenerateBackend
	IntrospectionBackend  IntrospectionBackend
	ModelRegistryBackend  ModelRegistryBackend
	ModelRoutesBackend    ModelRoutesBackend
	ModelProfileBackend   ModelProfileBackend
	ModelPackBackend      ModelPackBackend
	ScheduleBackend       ScheduleBackend
	TokenizerBackend      TokenizerBackend
	ParserBackend         ParserBackend
	RerankBackend         RerankBackend
	NativeGenerate        NativeGenerateConfig
}

type Server struct {
	SocketPath            string
	Registry              *Registry
	ModelPaths            map[string]string
	CacheBackend          CacheBackend
	CacheEntryBackend     CacheEntryBackend
	CancelBackend         CancelBackend
	EmbedBackend          EmbedBackend
	EngineFeaturesBackend EngineFeaturesBackend
	GenerateBackend       GenerateBackend
	IntrospectionBackend  IntrospectionBackend
	ModelRegistryBackend  ModelRegistryBackend
	ModelRoutesBackend    ModelRoutesBackend
	ModelProfileBackend   ModelProfileBackend
	ModelPackBackend      ModelPackBackend
	ScheduleBackend       ScheduleBackend
	TokenizerBackend      TokenizerBackend
	ParserBackend         ParserBackend
	RerankBackend         RerankBackend
	NativeGenerate        NativeGenerateConfig
}

type errorResponse struct {
	Status  string `json:"status"`
	Error   string `json:"error"`
	Message string `json:"message,omitempty"`
}

func NewServer(cfg ServerConfig) *Server {
	modelPaths := make(map[string]string, len(cfg.ModelPaths))
	maps.Copy(modelPaths, cfg.ModelPaths)
	if len(cfg.NativeGenerate.ModelPaths) == 0 && len(modelPaths) > 0 {
		cfg.NativeGenerate.ModelPaths = modelPaths
	}
	if cfg.Registry == nil {
		cfg.Registry = DefaultRegistryForDaemon()
	}
	if cfg.GenerateBackend == nil && len(cfg.NativeGenerate.ModelPaths) > 0 {
		cfg.GenerateBackend = NewNativeGenerateRunner(cfg.NativeGenerate)
	}
	if cfg.CacheBackend == nil {
		if backend, ok := cfg.GenerateBackend.(CacheBackend); ok {
			cfg.CacheBackend = backend
		}
	}
	if cfg.CacheEntryBackend == nil {
		if backend, ok := cfg.GenerateBackend.(CacheEntryBackend); ok {
			cfg.CacheEntryBackend = backend
		}
	}
	if cfg.CancelBackend == nil {
		if backend, ok := cfg.GenerateBackend.(CancelBackend); ok {
			cfg.CancelBackend = backend
		}
	}
	if cfg.EmbedBackend == nil {
		if backend, ok := cfg.GenerateBackend.(EmbedBackend); ok {
			cfg.EmbedBackend = backend
		}
	}
	if cfg.EngineFeaturesBackend == nil {
		if backend, ok := cfg.GenerateBackend.(EngineFeaturesBackend); ok {
			cfg.EngineFeaturesBackend = backend
		}
	}
	if cfg.IntrospectionBackend == nil {
		if backend, ok := cfg.GenerateBackend.(IntrospectionBackend); ok {
			cfg.IntrospectionBackend = backend
		}
	}
	if cfg.ModelPackBackend == nil {
		if backend, ok := cfg.GenerateBackend.(ModelPackBackend); ok {
			cfg.ModelPackBackend = backend
		}
	}
	if cfg.ModelProfileBackend == nil {
		if backend, ok := cfg.GenerateBackend.(ModelProfileBackend); ok {
			cfg.ModelProfileBackend = backend
		}
	}
	if cfg.ModelRoutesBackend == nil {
		if backend, ok := cfg.GenerateBackend.(ModelRoutesBackend); ok {
			cfg.ModelRoutesBackend = backend
		}
	}
	if cfg.ScheduleBackend == nil {
		if backend, ok := cfg.GenerateBackend.(ScheduleBackend); ok {
			cfg.ScheduleBackend = backend
		}
	}
	if cfg.TokenizerBackend == nil {
		if backend, ok := cfg.GenerateBackend.(TokenizerBackend); ok {
			cfg.TokenizerBackend = backend
		}
	}
	if cfg.ParserBackend == nil {
		if backend, ok := cfg.GenerateBackend.(ParserBackend); ok {
			cfg.ParserBackend = backend
		}
	}
	if cfg.RerankBackend == nil {
		if backend, ok := cfg.GenerateBackend.(RerankBackend); ok {
			cfg.RerankBackend = backend
		}
	}
	if cfg.EmbedBackend != nil {
		if err := cfg.Registry.RegisterEmbedBackend(cfg.EmbedBackend); err != nil {
			panic(err)
		}
	}
	if cfg.EngineFeaturesBackend != nil {
		if err := cfg.Registry.RegisterEngineFeaturesBackend(cfg.EngineFeaturesBackend); err != nil {
			panic(err)
		}
	}
	if cfg.RerankBackend != nil {
		if err := cfg.Registry.RegisterRerankBackend(cfg.RerankBackend); err != nil {
			panic(err)
		}
	}
	if cfg.CacheBackend != nil {
		if err := cfg.Registry.RegisterCacheBackend(cfg.CacheBackend); err != nil {
			panic(err)
		}
	}
	if cfg.CacheEntryBackend != nil {
		if err := cfg.Registry.RegisterCacheEntryBackend(cfg.CacheEntryBackend); err != nil {
			panic(err)
		}
	}
	if cfg.CancelBackend != nil {
		if err := cfg.Registry.RegisterCancelBackend(cfg.CancelBackend); err != nil {
			panic(err)
		}
	}
	if cfg.GenerateBackend != nil {
		if err := cfg.Registry.RegisterGenerateBackend(cfg.GenerateBackend); err != nil {
			panic(err)
		}
	}
	if cfg.ScheduleBackend != nil {
		if err := cfg.Registry.RegisterScheduleBackend(cfg.ScheduleBackend); err != nil {
			panic(err)
		}
	}
	if cfg.IntrospectionBackend != nil {
		if err := cfg.Registry.RegisterIntrospectionBackend(cfg.IntrospectionBackend); err != nil {
			panic(err)
		}
	}
	if cfg.ModelRegistryBackend != nil {
		if err := cfg.Registry.RegisterModelRegistryBackend(cfg.ModelRegistryBackend); err != nil {
			panic(err)
		}
	}
	if cfg.ModelPackBackend != nil {
		if err := cfg.Registry.RegisterModelPackBackend(cfg.ModelPackBackend); err != nil {
			panic(err)
		}
	}
	if cfg.ModelProfileBackend != nil {
		if err := cfg.Registry.RegisterModelProfileBackend(cfg.ModelProfileBackend); err != nil {
			panic(err)
		}
	}
	if cfg.ModelRoutesBackend != nil {
		if err := cfg.Registry.RegisterModelRoutesBackend(cfg.ModelRoutesBackend); err != nil {
			panic(err)
		}
	}
	if cfg.TokenizerBackend != nil {
		if err := cfg.Registry.RegisterTokenizerBackend(cfg.TokenizerBackend); err != nil {
			panic(err)
		}
	}
	if cfg.ParserBackend != nil {
		if err := cfg.Registry.RegisterParserBackend(cfg.ParserBackend); err != nil {
			panic(err)
		}
	}
	return &Server{
		SocketPath:            cfg.SocketPath,
		Registry:              cfg.Registry,
		ModelPaths:            modelPaths,
		CacheBackend:          cfg.CacheBackend,
		CacheEntryBackend:     cfg.CacheEntryBackend,
		CancelBackend:         cfg.CancelBackend,
		EmbedBackend:          cfg.EmbedBackend,
		EngineFeaturesBackend: cfg.EngineFeaturesBackend,
		GenerateBackend:       cfg.GenerateBackend,
		IntrospectionBackend:  cfg.IntrospectionBackend,
		ModelRegistryBackend:  cfg.ModelRegistryBackend,
		ModelRoutesBackend:    cfg.ModelRoutesBackend,
		ModelProfileBackend:   cfg.ModelProfileBackend,
		ModelPackBackend:      cfg.ModelPackBackend,
		ScheduleBackend:       cfg.ScheduleBackend,
		TokenizerBackend:      cfg.TokenizerBackend,
		ParserBackend:         cfg.ParserBackend,
		RerankBackend:         cfg.RerankBackend,
		NativeGenerate:        cfg.NativeGenerate,
	}
}

func (s *Server) ListenAndServe(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if s.Registry == nil {
		s.Registry = DefaultRegistryForDaemon()
	}
	socketPath, err := s.resolvedSocketPath()
	if err != nil {
		return err
	}
	if err := prepareSocketPath(socketPath); err != nil {
		return err
	}

	ln, err := net.Listen("unix", socketPath)
	if err != nil {
		return fmt.Errorf("listen unix %s: %w", socketPath, err)
	}
	if err := os.Chmod(socketPath, socketFileMode); err != nil {
		_ = ln.Close()
		_ = removePath(socketPath)
		return fmt.Errorf("chmod socket %s: %w", socketPath, err)
	}

	s.SocketPath = socketPath
	defer func() {
		_ = closeGenerateBackend(s.GenerateBackend)
		_ = ln.Close()
		_ = removePath(socketPath)
	}()
	return s.serve(ctx, ln)
}

func closeGenerateBackend(backend GenerateBackend) error {
	closer, ok := backend.(interface{ Close() error })
	if !ok || closer == nil {
		return nil
	}
	return closer.Close()
}

func (s *Server) serve(ctx context.Context, ln net.Listener) error {
	var wg sync.WaitGroup
	var conns sync.Map
	done := make(chan struct{})

	go func() {
		select {
		case <-ctx.Done():
			_ = ln.Close()
			conns.Range(func(key, _ any) bool {
				_ = key.(net.Conn).Close()
				return true
			})
		case <-done:
		}
	}()

	defer func() {
		close(done)
		wg.Wait()
	}()

	for {
		conn, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil || errors.Is(err, net.ErrClosed) {
				return nil
			}
			return fmt.Errorf("accept unix connection: %w", err)
		}
		conns.Store(conn, struct{}{})
		wg.Add(1)
		go func(conn net.Conn) {
			defer wg.Done()
			defer conns.Delete(conn)
			_ = s.handleConn(ctx, conn)
		}(conn)
	}
}

func (s *Server) handleConn(ctx context.Context, conn net.Conn) error {
	defer conn.Close()

	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 0, 64*1024), maxFrameBytes)
	var req Request
	for scanner.Scan() {
		if ctx.Err() != nil {
			return nil
		}
		trimmed := bytes.TrimSpace(scanner.Bytes())
		if len(trimmed) == 0 {
			continue
		}
		req = Request{}
		if err := json.Unmarshal(trimmed, &req); err != nil {
			if encodeErr := writeJSONLine(conn, errorResponse{
				Status:  "error",
				Error:   "invalid_json",
				Message: err.Error(),
			}); encodeErr != nil {
				return encodeErr
			}
			continue
		}
		resp, err := s.Registry.Dispatch(ctx, req)
		if err != nil {
			if encodeErr := writeJSONLine(conn, errorResponse{
				Status:  "error",
				Error:   "dispatch_error",
				Message: err.Error(),
			}); encodeErr != nil {
				return encodeErr
			}
			continue
		}
		if err := writeJSONLine(conn, resp); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil && ctx.Err() == nil {
		return err
	}
	return nil
}

func (s *Server) resolvedSocketPath() (string, error) {
	if s != nil && s.SocketPath != "" {
		return s.SocketPath, nil
	}
	return DefaultSocketPath()
}

func DefaultSocketPath() (string, error) {
	if runtime.GOOS == "darwin" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve home directory: %w", err)
		}
		return filepath.Join(home, "Library", "Caches", "ofm", "lthn.sock"), nil
	}
	runtimeDir := os.Getenv("XDG_RUNTIME_DIR")
	if runtimeDir == "" {
		return "", errXDGRuntimeDirUnset
	}
	return filepath.Join(runtimeDir, "ofm", "lthn.sock"), nil
}

func prepareSocketPath(socketPath string) error {
	if socketPath == "" {
		return errSocketPathRequired
	}
	if err := os.MkdirAll(filepath.Dir(socketPath), socketDirMode); err != nil {
		return fmt.Errorf("create socket directory: %w", err)
	}
	info, err := os.Lstat(socketPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("stat socket path: %w", err)
	}
	if info.Mode()&os.ModeSocket == 0 {
		return fmt.Errorf("refusing to replace non-socket path %s", socketPath)
	}
	if err := removePath(socketPath); err != nil {
		return fmt.Errorf("remove stale socket %s: %w", socketPath, err)
	}
	return nil
}

func writeJSONLine(w io.Writer, value any) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	encoded = append(encoded, '\n')
	_, err = w.Write(encoded)
	return err
}

func removePath(path string) error {
	return os.Remove(path)
}
