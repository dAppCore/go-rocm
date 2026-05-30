// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"iter"
	"testing"

	"dappco.re/go/inference"
)

// benchWireTokenSeq returns an iter.Seq[inference.Token] yielding count tokens,
// each carrying piece as its text. Models a streamed chat/generate response.
func benchWireTokenSeq(count int, piece string) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for index := 0; index < count; index++ {
			if !yield(inference.Token{Text: piece}) {
				return
			}
		}
	}
}

// collectROCmWireTokenTextNaive reproduces the pre-AX11 `text += token.Text`
// accumulation so the benchmark can demonstrate the alloc regression the
// builder-backed collectROCmWireTokenText removes.
func collectROCmWireTokenTextNaive(tokens iter.Seq[inference.Token]) string {
	text := ""
	for token := range tokens {
		text += token.Text
	}
	return text
}

func BenchmarkCollectROCmWireTokenText_512Tokens(b *testing.B) {
	const count = 512
	const piece = "token "
	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		_ = collectROCmWireTokenText(benchWireTokenSeq(count, piece))
	}
}

func BenchmarkCollectROCmWireTokenTextNaive_512Tokens(b *testing.B) {
	const count = 512
	const piece = "token "
	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		_ = collectROCmWireTokenTextNaive(benchWireTokenSeq(count, piece))
	}
}
