// SPDX-Licence-Identifier: EUPL-1.2

package main

import "strings"

const productionMeasurementPromptProfile = "long_mtp_warmup"

const defaultProductionMeasurementPrompt = "You are reviewing a ROCm inference runtime for a production beta. Summarize how a retained-state Gemma-4 QAT server should load model packs, wake and sleep KV state, verify attached MTP draft tokens, serve OpenAI-compatible chat turns, package AMD CUDA and CPU artifacts, compare against vLLM and llama.cpp, and refuse unsafe fallbacks. Include concrete implementation details, name the evidence a release manager should inspect, and finish with a compact checklist for the next engineering handoff."

func productionMeasurementPromptProfileFor(prompt string) string {
	if strings.TrimSpace(prompt) == "" || strings.TrimSpace(prompt) == defaultProductionMeasurementPrompt {
		return productionMeasurementPromptProfile
	}
	return "custom"
}
