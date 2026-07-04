// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"strings"
	"testing"

	"dappco.re/go/inference"
)

var (
	rocmJSONLBenchDataset *JSONLDataset
	rocmJSONLBenchSample  inference.DatasetSample
	rocmJSONLBenchOK      bool
	rocmJSONLBenchErr     error
)

func BenchmarkJSONLDataset_LoadPromptResponse_1000Rows(b *testing.B) {
	input := rocmJSONLBenchRows(1000, `{"prompt":"p","response":"r","target_token_id":7}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rocmJSONLBenchDataset, rocmJSONLBenchErr = LoadJSONLDataset(strings.NewReader(input))
	}
}

func BenchmarkJSONLDataset_LoadMessages_1000Rows(b *testing.B) {
	input := rocmJSONLBenchRows(1000, `{"messages":[{"role":"system","content":"steady"},{"role":"user","content":"ping"},{"role":"assistant","content":"pong"}],"student_logits":[1,0],"teacher_logits":[2,0]}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rocmJSONLBenchDataset, rocmJSONLBenchErr = LoadJSONLDataset(strings.NewReader(input))
	}
}

func BenchmarkJSONLDataset_StreamNext_1000Rows(b *testing.B) {
	samples := make([]inference.DatasetSample, 1000)
	for i := range samples {
		samples[i] = inference.DatasetSample{Prompt: "p", Response: "r", Labels: map[string]string{"target_token_id": "7"}}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dataset := NewJSONLDataset(samples)
		for {
			rocmJSONLBenchSample, rocmJSONLBenchOK, rocmJSONLBenchErr = dataset.Next()
			if !rocmJSONLBenchOK || rocmJSONLBenchErr != nil {
				break
			}
		}
	}
}

func rocmJSONLBenchRows(n int, row string) string {
	var builder strings.Builder
	builder.Grow((len(row) + 1) * n)
	for i := 0; i < n; i++ {
		builder.WriteString(row)
		builder.WriteByte('\n')
	}
	return builder.String()
}
