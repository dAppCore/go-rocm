//go:build linux && amd64

package rocm

import (
	"testing"

	"dappco.re/go/rocm/internal/gguf"
	"github.com/stretchr/testify/assert"
)

func TestBackend_ResolveContextLength_Good(t *testing.T) {
	assert.Equal(t, 2048, resolveContextLength(2048, gguf.Metadata{ContextLength: 32768}))
	assert.Equal(t, 1024, resolveContextLength(0, gguf.Metadata{ContextLength: 1024}))
	assert.Equal(t, defaultContextLengthCap, resolveContextLength(0, gguf.Metadata{ContextLength: 131072}))
}

func TestBackend_ResolveContextLength_Ugly(t *testing.T) {
	assert.Equal(t, defaultContextLengthCap, resolveContextLength(0, gguf.Metadata{}))
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
			assert.Equal(t, testCase.expectedBits, bits)
			assert.Equal(t, testCase.expectedGroupSize, groupSize)
		})
	}
}

func TestBackend_ModelInfoFromMetadata_Good(t *testing.T) {
	modelInfo := modelInfoFromMetadata(gguf.Metadata{
		Architecture: "gemma3",
		BlockCount:   34,
		FileType:     15,
	})

	assert.Equal(t, "gemma3", modelInfo.Architecture)
	assert.Equal(t, 34, modelInfo.NumLayers)
	assert.Equal(t, 4, modelInfo.QuantBits)
	assert.Equal(t, 32, modelInfo.QuantGroup)
}
