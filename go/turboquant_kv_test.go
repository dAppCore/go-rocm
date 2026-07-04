// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"math"
	"testing"

	core "dappco.re/go"
)

func TestTurboQuantKV_Good_ThreePointFiveBitRoundTrip(t *testing.T) {
	values := turboQuantKVFixture(128)
	desc := defaultROCmTurboQuantKVDescriptor()
	desc.GroupSize = 32

	encoded, err := encodeROCmTurboQuantKV(values, desc)
	core.RequireNoError(t, err)
	decoded, err := encoded.decode()
	core.RequireNoError(t, err)

	q8, err := encodeROCmKVTensor(rocmKVEncodingQ8, values)
	core.RequireNoError(t, err)
	if encoded.SizeBytes >= q8.sizeBytes {
		t.Fatalf("turboquant bytes = %d, q8 bytes = %d, want smaller KV payload", encoded.SizeBytes, q8.sizeBytes)
	}
	core.AssertEqual(t, 56, len(encoded.Packed))
	core.AssertEqual(t, 4, len(encoded.Scales))
	core.AssertEqual(t, 4, len(encoded.Residuals))
	if maxFloat32AbsDiff(values, decoded) > 0.16 {
		t.Fatalf("max round-trip diff = %.4f, want bounded research-codec error", maxFloat32AbsDiff(values, decoded))
	}
}

func TestTurboQuantKV_Good_ResidualCorrectionReducesBias(t *testing.T) {
	values := turboQuantKVFixture(96)
	desc := defaultROCmTurboQuantKVDescriptor()
	desc.GroupSize = 32
	withResidual, err := encodeROCmTurboQuantKV(values, desc)
	core.RequireNoError(t, err)
	withDecoded, err := withResidual.decode()
	core.RequireNoError(t, err)

	desc.ResidualCorrection = false
	withoutResidual, err := encodeROCmTurboQuantKV(values, desc)
	core.RequireNoError(t, err)
	withoutDecoded, err := withoutResidual.decode()
	core.RequireNoError(t, err)

	if math.Abs(meanFloat32Error(values, withDecoded)) > math.Abs(meanFloat32Error(values, withoutDecoded)) {
		t.Fatalf("residual mean error = %.6f, no residual = %.6f, want group bias correction",
			meanFloat32Error(values, withDecoded), meanFloat32Error(values, withoutDecoded))
	}
}

func TestTurboQuantKV_Good_WorkspaceReuseMatchesConveniencePath(t *testing.T) {
	values := turboQuantKVFixture(256)
	desc := defaultROCmTurboQuantKVDescriptor()
	var workspace rocmTurboQuantKVWorkspace

	want, err := encodeROCmTurboQuantKV(values, desc)
	core.RequireNoError(t, err)
	wantDecoded, err := want.decode()
	core.RequireNoError(t, err)

	for i := 0; i < 3; i++ {
		got, err := encodeROCmTurboQuantKVInto(values, desc, &workspace)
		core.RequireNoError(t, err)
		gotDecoded, err := got.decodeInto(&workspace)
		core.RequireNoError(t, err)
		core.AssertEqual(t, want.SizeBytes, got.SizeBytes)
		assertFloat32SlicesNear(t, wantDecoded, gotDecoded, 0)
	}
}

func TestTurboQuantKV_Bad_Validation(t *testing.T) {
	_, err := encodeROCmTurboQuantKV(nil, defaultROCmTurboQuantKVDescriptor())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "values are required")

	desc := defaultROCmTurboQuantKVDescriptor()
	desc.BitsNumerator = 1
	_, err = encodeROCmTurboQuantKV([]float32{1}, desc)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "average bits")

	desc = defaultROCmTurboQuantKVDescriptor()
	desc.GroupSize = 3
	_, err = encodeROCmTurboQuantKV([]float32{1}, desc)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "power of two")

	desc = defaultROCmTurboQuantKVDescriptor()
	encoded, err := encodeROCmTurboQuantKV([]float32{1, 2, 3, 4}, desc)
	core.RequireNoError(t, err)
	encoded.Packed = encoded.Packed[:0]
	_, err = encoded.decode()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "packed values need")
}

func BenchmarkTurboQuantKV_EncodeDecode_ThreePointFiveBitPage(b *testing.B) {
	values := turboQuantKVFixture(512 * 128)
	desc := defaultROCmTurboQuantKVDescriptor()
	b.SetBytes(int64(len(values) * 4))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		encoded, err := encodeROCmTurboQuantKV(values, desc)
		if err != nil {
			b.Fatalf("encode TurboQuant KV: %v", err)
		}
		decoded, err := encoded.decode()
		if err != nil {
			b.Fatalf("decode TurboQuant KV: %v", err)
		}
		if len(decoded) != len(values) {
			b.Fatalf("decoded length = %d, want %d", len(decoded), len(values))
		}
	}
}

func BenchmarkTurboQuantKV_EncodeDecodeWorkspace_ThreePointFiveBitPage(b *testing.B) {
	values := turboQuantKVFixture(512 * 128)
	desc := defaultROCmTurboQuantKVDescriptor()
	var workspace rocmTurboQuantKVWorkspace
	encoded, err := encodeROCmTurboQuantKVInto(values, desc, &workspace)
	if err != nil {
		b.Fatalf("prewarm TurboQuant KV encode: %v", err)
	}
	if _, err := encoded.decodeInto(&workspace); err != nil {
		b.Fatalf("prewarm TurboQuant KV decode: %v", err)
	}

	b.SetBytes(int64(len(values) * 4))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		encoded, err := encodeROCmTurboQuantKVInto(values, desc, &workspace)
		if err != nil {
			b.Fatalf("encode TurboQuant KV: %v", err)
		}
		decoded, err := encoded.decodeInto(&workspace)
		if err != nil {
			b.Fatalf("decode TurboQuant KV: %v", err)
		}
		if len(decoded) != len(values) {
			b.Fatalf("decoded length = %d, want %d", len(decoded), len(values))
		}
	}
}

func turboQuantKVFixture(count int) []float32 {
	values := make([]float32, count)
	for i := range values {
		x := float64(i)
		values[i] = float32(math.Sin(x*0.031)*0.7 + math.Cos(x*0.007)*0.2 + math.Sin(x*0.113)*0.05)
	}
	return values
}

func maxFloat32AbsDiff(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}
	maxDiff := float32(0)
	for i := range a {
		diff := float32(math.Abs(float64(a[i] - b[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	return maxDiff
}

func meanFloat32AbsDiff(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return math.Inf(1)
	}
	sum := float64(0)
	for i := range a {
		sum += math.Abs(float64(a[i] - b[i]))
	}
	return sum / float64(len(a))
}

func meanFloat32Error(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return math.Inf(1)
	}
	sum := float64(0)
	for i := range a {
		sum += float64(a[i] - b[i])
	}
	return sum / float64(len(a))
}
