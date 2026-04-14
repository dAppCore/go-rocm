package rocm

import (
	"path/filepath"

	"dappco.re/go/core/rocm/internal/gguf"
)

// DiscoverModels scans a directory for GGUF model files and returns
// structured information about each. Files that cannot be parsed are skipped.
func DiscoverModels(dir string) ([]ModelInfo, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.gguf"))
	if err != nil {
		return nil, err
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
