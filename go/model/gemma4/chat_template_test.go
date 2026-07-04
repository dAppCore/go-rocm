// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	"dappco.re/go/inference"
)

func TestFormatChatTemplate_Good_DefaultAndSystemTurns(t *testing.T) {
	if got, want := FormatChatTemplate([]inference.Message{{Role: "user", Content: " hello "}}),
		"<bos><|turn>user\nhello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"; got != want {
		t.Fatalf("FormatChatTemplate(user) = %q, want %q", got, want)
	}

	got := FormatChatTemplate([]inference.Message{
		{Role: "developer", Content: " be concise "},
		{Role: "user", Content: "hello"},
	})
	want := "<bos><|turn>system\nbe concise<turn|>\n<|turn>user\nhello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
	if got != want {
		t.Fatalf("FormatChatTemplate(system) = %q, want %q", got, want)
	}
}

func TestFormatChatTemplateWithConfig_Good_ThinkingContinuationAndSuppressor(t *testing.T) {
	got := FormatChatTemplateWithConfig([]inference.Message{{Role: "user", Content: "hello"}}, ChatTemplateConfig{EnableThinking: true})
	want := "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhello<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("thinking template = %q, want %q", got, want)
	}

	got = FormatChatTemplateWithConfig([]inference.Message{{Role: "user", Content: "next"}}, ChatTemplateConfig{Continuation: true})
	want = "<turn|>\n<|turn>user\nnext<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
	if got != want {
		t.Fatalf("continuation template = %q, want %q", got, want)
	}
}

func TestFormatChatTemplateWithConfig_Good_StripsAssistantThoughtHistory(t *testing.T) {
	got := FormatChatTemplateWithConfig([]inference.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "<|channel>thought\nprivate<channel|>visible"},
	}, ChatTemplateConfig{NoGenerationPrompt: true})
	want := "<bos><|turn>user\nhi<turn|>\n<|turn>model\nvisible<turn|>\n"
	if got != want {
		t.Fatalf("assistant thought strip = %q, want %q", got, want)
	}
}

func TestFormatChatTemplateWithConfig_Good_ContinuesAssistantRuns(t *testing.T) {
	got := FormatChatTemplateWithConfig([]inference.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "one"},
		{Role: "assistant", Content: "two"},
	}, ChatTemplateConfig{})
	want := "<bos><|turn>user\nhi<turn|>\n<|turn>model\none<turn|>\ntwo<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
	if got != want {
		t.Fatalf("assistant continuation = %q, want %q", got, want)
	}
}

func TestStripThinkingChannels_Good_DropsPrivateSpans(t *testing.T) {
	for _, tc := range []struct {
		name string
		text string
		want string
	}{
		{name: "visible", text: "visible", want: "visible"},
		{name: "middle", text: "visible<|channel>hidden<channel|>", want: "visible"},
		{name: "prefix", text: "<|channel>hidden<channel|>visible", want: "visible"},
		{name: "multi", text: "a<|channel>x<channel|>b<|channel>y<channel|>c", want: "abc"},
		{name: "unclosed", text: "visible<|channel>hidden", want: "visible"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := StripThinkingChannels(tc.text); got != tc.want {
				t.Fatalf("StripThinkingChannels(%q) = %q, want %q", tc.text, got, tc.want)
			}
		})
	}
}
