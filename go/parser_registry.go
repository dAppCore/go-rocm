// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"dappco.re/go/inference"
	outputparser "dappco.re/go/inference/parser"
)

// ParserRegistry provides architecture-aware reasoning and tool parsing.
type ParserRegistry struct {
	architecture string
	parser       outputparser.OutputParser
}

// NewParserRegistry creates a parser registry for one model family.
func NewParserRegistry(architecture string) ParserRegistry {
	architecture = normalizeROCmArchitecture(architecture)
	return ParserRegistry{
		architecture: architecture,
		parser:       outputparser.ForHint(outputparser.Hint{Architecture: architecture}),
	}
}

func (registry ParserRegistry) ParseReasoning(tokens []inference.Token, text string) (inference.ReasoningParseResult, error) {
	return registry.outputParser().ParseReasoning(tokens, text)
}

func (registry ParserRegistry) ParseTools(tokens []inference.Token, text string) (inference.ToolParseResult, error) {
	return registry.outputParser().ParseTools(tokens, text)
}

func (registry ParserRegistry) outputParser() outputparser.OutputParser {
	if registry.parser != nil {
		return registry.parser
	}
	return outputparser.ForHint(outputparser.Hint{Architecture: registry.architecture})
}

func (m *rocmModel) ParseReasoning(tokens []inference.Token, text string) (result inference.ReasoningParseResult, err error) {
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	architecture := ""
	if m != nil {
		architecture = firstNonEmptyString(m.modelInfo.Architecture, m.modelType)
	}
	return NewParserRegistry(architecture).ParseReasoning(tokens, text)
}

func (m *rocmModel) ParseTools(tokens []inference.Token, text string) (result inference.ToolParseResult, err error) {
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	architecture := ""
	if m != nil {
		architecture = firstNonEmptyString(m.modelInfo.Architecture, m.modelType)
	}
	return NewParserRegistry(architecture).ParseTools(tokens, text)
}
