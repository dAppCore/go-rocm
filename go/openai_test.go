// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import "testing"

func TestOpenAI_NewOpenAIResolver_Good_UsesROCmBackend(t *testing.T) {
	resolver := NewOpenAIResolver("/models/qwen3.gguf")
	if resolver == nil {
		t.Fatal("NewOpenAIResolver() returned nil")
	}
	if resolver.BackendName != "rocm" {
		t.Fatalf("BackendName = %q, want rocm", resolver.BackendName)
	}
	if resolver.ModelPath != "/models/qwen3.gguf" {
		t.Fatalf("ModelPath = %q", resolver.ModelPath)
	}
}

func TestOpenAI_NewOpenAIHandler_Good_ReturnsHTTPHandler(t *testing.T) {
	handler := NewOpenAIHandler("/models/qwen3.gguf")
	if handler == nil {
		t.Fatal("NewOpenAIHandler() returned nil")
	}
}
