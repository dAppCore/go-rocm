// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	stateKVContainerMagic          = "KVST"
	stateKVContainerVersion        = 2
	stateKVContainerMaxHeaderBytes = 16 * 1024 * 1024

	stateKVContainerContentType      = "application/vnd.go-rocm.state-log"
	goMLXStateKVContainerContentType = "application/vnd.go-mlx.state-log"
	stateKVContainerKind             = "go-rocm/state-kv"
	goMLXStateKVContainerKind        = "go-mlx/state-kv"
)

type stateRampFoldMarker struct {
	StorePath  string `json:"store_path,omitempty"`
	IndexURI   string `json:"index_uri,omitempty"`
	EntryURI   string `json:"entry_uri,omitempty"`
	BundleURI  string `json:"bundle_uri,omitempty"`
	TokenCount int    `json:"token_count,omitempty"`
}

type statePackOptions struct {
	MarkerFile     string
	StateStorePath string
	OutputPath     string
}

type statePackReport struct {
	Version        int                 `json:"version"`
	Magic          string              `json:"magic"`
	TrixVersion    int                 `json:"trix_version"`
	Backend        string              `json:"backend"`
	Command        string              `json:"command"`
	CLIContract    string              `json:"cli_contract"`
	MarkerFile     string              `json:"marker_file"`
	StateStorePath string              `json:"state_store_path"`
	OutputPath     string              `json:"output_path"`
	PayloadBytes   int64               `json:"payload_bytes"`
	ContainerBytes int64               `json:"container_bytes,omitempty"`
	Marker         stateRampFoldMarker `json:"marker"`
	Header         map[string]any      `json:"header,omitempty"`
}

type stateWakeProfileMarkerFile struct {
	StorePath string                      `json:"store_path,omitempty"`
	IndexURI  string                      `json:"index_uri,omitempty"`
	EntryURI  string                      `json:"entry_uri,omitempty"`
	BundleURI string                      `json:"bundle_uri,omitempty"`
	Fold      *stateWakeProfileMarkerFold `json:"fold,omitempty"`
}

type stateWakeProfileMarkerFold struct {
	StorePath     string                        `json:"store_path,omitempty"`
	CompactMarker *stateRampFoldMarker          `json:"compact_marker,omitempty"`
	Folded        *stateWakeProfileFoldedReport `json:"folded,omitempty"`
}

type stateWakeProfileFoldedReport struct {
	IndexURI   string `json:"index_uri,omitempty"`
	EntryURI   string `json:"entry_uri,omitempty"`
	BundleURI  string `json:"bundle_uri,omitempty"`
	TokenCount int    `json:"token_count,omitempty"`
}

type stateKVContainerHeaderInfo struct {
	Header        map[string]any
	PayloadOffset int64
	PayloadBytes  int64
}

func runStatePackCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("state-pack"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON report")
	markerFile := fs.String("marker-file", "", "state-ramp-profile report or compact marker JSON")
	stateStorePath := fs.String("state-store", "", "state .mvlog path; defaults to the marker store_path")
	logPath := fs.String("log", "", "binary state log path; compatibility alias for -state-store")
	outputPath := fs.String("output", "", "output .kv container path")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: %s state-pack -marker-file <path> -output <path.kv> [flags]\n\n", cliName())
		fmt.Fprintln(stderr, "Pack a state marker and its binary payload into a portable KVST container.")
		fmt.Fprintln(stderr, "\nFlags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s state-pack: expected no positional arguments\n", cliName())
		fs.Usage()
		return 2
	}
	storePath := strings.TrimSpace(*stateStorePath)
	if storePath == "" {
		storePath = strings.TrimSpace(*logPath)
	}
	if strings.TrimSpace(*markerFile) == "" {
		fmt.Fprintf(stderr, "%s state-pack: marker file is required\n", cliName())
		return 2
	}
	if strings.TrimSpace(*outputPath) == "" {
		fmt.Fprintf(stderr, "%s state-pack: output path is required\n", cliName())
		return 2
	}
	report, err := runStatePack(ctx, statePackOptions{
		MarkerFile:     *markerFile,
		StateStorePath: storePath,
		OutputPath:     *outputPath,
	})
	if err != nil {
		fmt.Fprintf(stderr, "%s state-pack: %v\n", cliName(), err)
		return 1
	}
	if *jsonOut {
		return writeJSON(stdout, stderr, report)
	}
	fmt.Fprintf(stdout, "packed %s (%d payload bytes) into %s\n", report.StateStorePath, report.PayloadBytes, report.OutputPath)
	return 0
}

var runStatePack = defaultRunStatePack

func defaultRunStatePack(ctx context.Context, opts statePackOptions) (*statePackReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	opts.MarkerFile = strings.TrimSpace(opts.MarkerFile)
	opts.StateStorePath = strings.TrimSpace(opts.StateStorePath)
	opts.OutputPath = strings.TrimSpace(opts.OutputPath)
	marker, err := stateWakeProfileCompactMarkerFromFile(opts.MarkerFile)
	if err != nil {
		return nil, err
	}
	if opts.StateStorePath == "" {
		opts.StateStorePath = marker.StorePath
	}
	if opts.StateStorePath == "" {
		return nil, errors.New("state store path is required")
	}
	info, err := os.Stat(opts.StateStorePath)
	if err != nil {
		return nil, err
	}
	header := stateKVContainerHeader(opts, marker, info.Size())
	written, err := stateKVContainerEncode(ctx, opts.OutputPath, header, opts.StateStorePath)
	if err != nil {
		return nil, err
	}
	report := &statePackReport{
		Version:        1,
		Magic:          stateKVContainerMagic,
		TrixVersion:    stateKVContainerVersion,
		Backend:        defaultBackendName,
		Command:        "state-pack",
		CLIContract:    cliContractName,
		MarkerFile:     opts.MarkerFile,
		StateStorePath: opts.StateStorePath,
		OutputPath:     opts.OutputPath,
		PayloadBytes:   written,
		Marker:         marker,
		Header:         header,
	}
	if stat, err := os.Stat(opts.OutputPath); err == nil {
		report.ContainerBytes = stat.Size()
	}
	return report, nil
}

func stateWakeProfileCompactMarkerFromFile(path string) (stateRampFoldMarker, error) {
	data, err := os.ReadFile(strings.TrimSpace(path))
	if err != nil {
		return stateRampFoldMarker{}, err
	}
	var payload stateWakeProfileMarkerFile
	if err := json.Unmarshal(data, &payload); err != nil {
		return stateRampFoldMarker{}, err
	}
	if marker := stateWakeProfileCompactMarkerFromPayload(payload); marker.IndexURI != "" {
		return marker, nil
	}
	return stateRampFoldMarker{}, errors.New("state compact marker missing store_path or index_uri")
}

func stateWakeProfileCompactMarkerFromPayload(payload stateWakeProfileMarkerFile) stateRampFoldMarker {
	if payload.IndexURI != "" {
		return stateRampFoldMarker{
			StorePath:  payload.StorePath,
			IndexURI:   payload.IndexURI,
			EntryURI:   payload.EntryURI,
			BundleURI:  payload.BundleURI,
			TokenCount: 0,
		}
	}
	if payload.Fold == nil {
		return stateRampFoldMarker{}
	}
	if marker := payload.Fold.CompactMarker; marker != nil && marker.IndexURI != "" {
		return *marker
	}
	if payload.Fold.Folded == nil || payload.Fold.Folded.IndexURI == "" {
		return stateRampFoldMarker{}
	}
	return stateRampFoldMarker{
		StorePath:  payload.Fold.StorePath,
		IndexURI:   payload.Fold.Folded.IndexURI,
		EntryURI:   payload.Fold.Folded.EntryURI,
		BundleURI:  payload.Fold.Folded.BundleURI,
		TokenCount: payload.Fold.Folded.TokenCount,
	}
}

func stateKVContainerHeader(opts statePackOptions, marker stateRampFoldMarker, payloadBytes int64) map[string]any {
	return map[string]any{
		"kind":                 stateKVContainerKind,
		"content_type":         stateKVContainerContentType,
		"payload_file":         filepath.Base(opts.StateStorePath),
		"payload_bytes":        payloadBytes,
		"marker_file":          opts.MarkerFile,
		"state_store_path":     opts.StateStorePath,
		"index_uri":            marker.IndexURI,
		"entry_uri":            marker.EntryURI,
		"bundle_uri":           marker.BundleURI,
		"token_count":          marker.TokenCount,
		"created_at_unix_nano": time.Now().UTC().UnixNano(),
		"backend":              defaultBackendName,
		"cli_contract":         cliContractName,
	}
}

func stateKVContainerEncode(ctx context.Context, outputPath string, header map[string]any, payloadPath string) (int64, error) {
	if len(stateKVContainerMagic) != 4 {
		return 0, errors.New("state KV container magic must be 4 bytes")
	}
	outputPath = strings.TrimSpace(outputPath)
	if dir := filepath.Dir(outputPath); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return 0, fmt.Errorf("create output directory: %w", err)
		}
	}
	headerBytes, err := json.Marshal(header)
	if err != nil {
		return 0, err
	}
	if len(headerBytes) > stateKVContainerMaxHeaderBytes {
		return 0, fmt.Errorf("state KV container header exceeds %d bytes", stateKVContainerMaxHeaderBytes)
	}
	payload, err := os.Open(payloadPath)
	if err != nil {
		return 0, err
	}
	defer payload.Close()
	out, err := os.OpenFile(outputPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return 0, err
	}
	defer out.Close()
	if _, err := io.WriteString(out, stateKVContainerMagic); err != nil {
		return 0, err
	}
	if _, err := out.Write([]byte{byte(stateKVContainerVersion)}); err != nil {
		return 0, err
	}
	if err := binary.Write(out, binary.BigEndian, uint32(len(headerBytes))); err != nil {
		return 0, err
	}
	if _, err := out.Write(headerBytes); err != nil {
		return 0, err
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	written, err := io.Copy(out, payload)
	if err != nil {
		return written, err
	}
	return written, ctx.Err()
}

func stateKVContainerReadHeaderInfo(path string) (stateKVContainerHeaderInfo, error) {
	file, err := os.Open(path)
	if err != nil {
		return stateKVContainerHeaderInfo{}, err
	}
	defer file.Close()
	magic := make([]byte, 4)
	if _, err := io.ReadFull(file, magic); err != nil {
		return stateKVContainerHeaderInfo{}, err
	}
	if string(magic) != stateKVContainerMagic {
		return stateKVContainerHeaderInfo{}, fmt.Errorf("state KV magic = %q, want %q", string(magic), stateKVContainerMagic)
	}
	version := make([]byte, 1)
	if _, err := io.ReadFull(file, version); err != nil {
		return stateKVContainerHeaderInfo{}, err
	}
	if version[0] != stateKVContainerVersion {
		return stateKVContainerHeaderInfo{}, fmt.Errorf("state KV version = %d, want %d", version[0], stateKVContainerVersion)
	}
	var headerLen uint32
	if err := binary.Read(file, binary.BigEndian, &headerLen); err != nil {
		return stateKVContainerHeaderInfo{}, err
	}
	if headerLen > stateKVContainerMaxHeaderBytes {
		return stateKVContainerHeaderInfo{}, fmt.Errorf("state KV header exceeds %d bytes", stateKVContainerMaxHeaderBytes)
	}
	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(file, headerBytes); err != nil {
		return stateKVContainerHeaderInfo{}, err
	}
	var header map[string]any
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return stateKVContainerHeaderInfo{}, err
	}
	offset, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return stateKVContainerHeaderInfo{}, err
	}
	stat, err := file.Stat()
	if err != nil {
		return stateKVContainerHeaderInfo{}, err
	}
	return stateKVContainerHeaderInfo{Header: header, PayloadOffset: offset, PayloadBytes: stat.Size() - offset}, nil
}

func stateKVContainerPayload(path string) ([]byte, map[string]any, error) {
	info, err := stateKVContainerReadHeaderInfo(path)
	if err != nil {
		return nil, nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()
	if _, err := file.Seek(info.PayloadOffset, io.SeekStart); err != nil {
		return nil, nil, err
	}
	payload, err := io.ReadAll(file)
	if err != nil {
		return nil, nil, err
	}
	return payload, info.Header, nil
}

func stateKVContainerMarkerFromHeader(header map[string]any, actualPayloadBytes int64) (stateRampFoldMarker, error) {
	if kind := stateKVHeaderString(header, "kind"); !stateKVContainerKindCompatible(kind) {
		return stateRampFoldMarker{}, fmt.Errorf("state KV container kind = %q", kind)
	}
	if contentType := stateKVHeaderString(header, "content_type"); !stateKVContentTypeCompatible(contentType) {
		return stateRampFoldMarker{}, fmt.Errorf("state KV content type = %q", contentType)
	}
	if expectedPayloadBytes := stateKVHeaderInt64(header, "payload_bytes"); expectedPayloadBytes > 0 && expectedPayloadBytes != actualPayloadBytes {
		return stateRampFoldMarker{}, fmt.Errorf("state KV payload bytes = %d, want %d", actualPayloadBytes, expectedPayloadBytes)
	}
	marker := stateRampFoldMarker{
		StorePath:  stateKVHeaderString(header, "state_store_path"),
		IndexURI:   stateKVHeaderString(header, "index_uri"),
		EntryURI:   stateKVHeaderString(header, "entry_uri"),
		BundleURI:  stateKVHeaderString(header, "bundle_uri"),
		TokenCount: int(stateKVHeaderInt64(header, "token_count")),
	}
	if marker.IndexURI == "" {
		return stateRampFoldMarker{}, errors.New("state KV container missing index_uri")
	}
	return marker, nil
}

func stateKVContainerKindCompatible(kind string) bool {
	return kind == stateKVContainerKind || kind == goMLXStateKVContainerKind
}

func stateKVContentTypeCompatible(contentType string) bool {
	return contentType == stateKVContainerContentType || contentType == goMLXStateKVContainerContentType
}

func stateKVHeaderString(header map[string]any, key string) string {
	value, ok := header[key]
	if !ok {
		return ""
	}
	text, ok := value.(string)
	if !ok {
		return ""
	}
	return text
}

func stateKVHeaderInt64(header map[string]any, key string) int64 {
	value, ok := header[key]
	if !ok {
		return 0
	}
	switch n := value.(type) {
	case int:
		return int64(n)
	case int64:
		return n
	case float64:
		return int64(n)
	default:
		return 0
	}
}
