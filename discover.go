package rocm

import (
	"path/filepath"

	coreerr "dappco.re/go/core/log"
	"dappco.re/go/rocm/internal/gguf"
)

// DiscoverModels scans a directory for GGUF model files and returns
// structured information about each. Files that cannot be parsed are skipped.
func DiscoverModels(dir string) ([]ModelInfo, error) {
	root, err := filepath.Abs(dir)
	if err != nil {
		return nil, coreerr.E("rocm.DiscoverModels", "resolve model directory", err)
	}

	matches, err := filepath.Glob(filepath.Join(root, "*.gguf"))
	if err != nil {
		return nil, coreerr.E("rocm.DiscoverModels", "glob gguf files", err)
	}

	var models []ModelInfo
	for _, path := range matches {
		meta, err := gguf.ReadMetadata(path)
		if err != nil {
			continue
		}

		models = append(models, ModelInfo{
			Path:         path,
			Architecture: meta.Architecture,
			Name:         meta.Name,
			Quantisation: gguf.FileTypeName(meta.FileType),
			Parameters:   meta.SizeLabel,
			FileSize:     meta.FileSize,
			ContextLen:   meta.ContextLength,
		})
	}

	return models, nil
}
