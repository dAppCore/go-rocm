// SPDX-Licence-Identifier: EUPL-1.2

package registry

import (
	"slices"
	"testing"
)

func TestOrdered_Good_ReplacesValuesWithoutChangingOrder(t *testing.T) {
	registry := NewOrdered[string, string]()
	registry.Put("gemma4", "first")
	registry.Put("qwen3", "second")
	registry.Put("gemma4", "replacement")

	if keys := registry.Keys(); !slices.Equal(keys, []string{"gemma4", "qwen3"}) {
		t.Fatalf("Keys = %v, want first-registration order", keys)
	}
	if value, ok := registry.Get("gemma4"); !ok || value != "replacement" {
		t.Fatalf("Get(gemma4) = %q ok=%v, want replacement", value, ok)
	}
	if values := registry.Values(); !slices.Equal(values, []string{"replacement", "second"}) {
		t.Fatalf("Values = %v, want replacement in original order", values)
	}
}

func TestOrdered_Good_SnapshotAndRestoreAreCopySafe(t *testing.T) {
	registry := NewOrdered[string, string]()
	registry.Put("gemma4", "first")

	order, values := registry.Snapshot()
	order[0] = "mutated"
	values["gemma4"] = "mutated"
	if keys := registry.Keys(); !slices.Equal(keys, []string{"gemma4"}) {
		t.Fatalf("Keys after mutated snapshot = %v, want registry state unchanged", keys)
	}
	if got := registry.Values(); !slices.Equal(got, []string{"first"}) {
		t.Fatalf("Values after mutated snapshot = %v, want registry state unchanged", got)
	}

	registry.Put("qwen3", "second")
	registry.Restore([]string{"restored"}, map[string]string{"restored": "value"})
	if keys := registry.Keys(); !slices.Equal(keys, []string{"restored"}) {
		t.Fatalf("Keys after restore = %v, want restored order", keys)
	}
	if got := registry.Values(); !slices.Equal(got, []string{"value"}) {
		t.Fatalf("Values after restore = %v, want restored value", got)
	}
}
