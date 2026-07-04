// SPDX-Licence-Identifier: EUPL-1.2

package builtin

import (
	"slices"
	"testing"

	"dappco.re/go/rocm/model"
)

func TestBuiltinProfileFactories_Good_RegisterSpecificBeforeGeneric(t *testing.T) {
	names := model.RegisteredProfileFactoryNames()
	if !slices.Equal(names, []string{"gemma4", "architecture-profile"}) {
		t.Fatalf("RegisteredProfileFactoryNames = %v, want Gemma4 before generic architecture-profile fallback", names)
	}
}
