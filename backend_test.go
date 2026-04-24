//go:build linux && amd64

package rocm

import (
	"testing"

	"dappco.re/go/rocm/internal/gguf"
)

func TestBackend_ResolveContextLength_Good(t *testing.T) {
	if got := resolveContextLength(2048, gguf.Metadata{ContextLength: 32768}); got != 2048 {
		t.Errorf("resolveContextLength(2048, 32768) = %d, want 2048", got)
	}
	if got := resolveContextLength(0, gguf.Metadata{ContextLength: 1024}); got != 1024 {
		t.Errorf("resolveContextLength(0, 1024) = %d, want 1024", got)
	}
	if got := resolveContextLength(0, gguf.Metadata{ContextLength: 131072}); got != defaultContextLengthCap {
		t.Errorf("resolveContextLength(0, 131072) = %d, want %d", got, defaultContextLengthCap)
	}
}

func TestBackend_ResolveContextLength_Ugly(t *testing.T) {
	if got := resolveContextLength(0, gguf.Metadata{}); got != defaultContextLengthCap {
		t.Errorf("resolveContextLength(0, empty) = %d, want %d", got, defaultContextLengthCap)
	}
}

func TestBackend_QuantisationFromFileType_Good(t *testing.T) {
	testCases := []struct {
		name              string
		fileType          uint32
		expectedBits      int
		expectedGroupSize int
	}{
		{name: "q4", fileType: 15, expectedBits: 4, expectedGroupSize: 32},
		{name: "q5", fileType: 17, expectedBits: 5, expectedGroupSize: 32},
		{name: "q8", fileType: 7, expectedBits: 8, expectedGroupSize: 32},
		{name: "q2", fileType: 10, expectedBits: 2, expectedGroupSize: 16},
		{name: "q6", fileType: 18, expectedBits: 6, expectedGroupSize: 64},
		{name: "f16", fileType: 1, expectedBits: 16, expectedGroupSize: 0},
		{name: "f32", fileType: 0, expectedBits: 32, expectedGroupSize: 0},
		{name: "unknown", fileType: 999, expectedBits: 0, expectedGroupSize: 0},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			bits, groupSize := quantisationFromFileType(testCase.fileType)
			if bits != testCase.expectedBits {
				t.Errorf("bits = %d, want %d", bits, testCase.expectedBits)
			}
			if groupSize != testCase.expectedGroupSize {
				t.Errorf("groupSize = %d, want %d", groupSize, testCase.expectedGroupSize)
			}
		})
	}
}

func TestBackend_ModelInfoFromMetadata_Good(t *testing.T) {
	modelInfo := modelInfoFromMetadata(gguf.Metadata{
		Architecture: "gemma3",
		BlockCount:   34,
		FileType:     15,
	})

	if modelInfo.Architecture != "gemma3" {
		t.Errorf("Architecture = %q, want %q", modelInfo.Architecture, "gemma3")
	}
	if modelInfo.NumLayers != 34 {
		t.Errorf("NumLayers = %d, want 34", modelInfo.NumLayers)
	}
	if modelInfo.QuantBits != 4 {
		t.Errorf("QuantBits = %d, want 4", modelInfo.QuantBits)
	}
	if modelInfo.QuantGroup != 32 {
		t.Errorf("QuantGroup = %d, want 32", modelInfo.QuantGroup)
	}
}
