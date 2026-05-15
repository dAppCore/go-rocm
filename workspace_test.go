// SPDX-Licence-Identifier: EUPL-1.2

//go:build cgo && !rocm_legacy_server

package workspace

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestWorkspaceDefaultModuleGates(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	runWorkspaceGoTest(ctx, t, "go", "test", "./go/...", "-count=1")
	runWorkspaceGoTest(ctx, t, "go", "test", "./external/go-inference/go/...", "-count=1")
}

func runWorkspaceGoTest(ctx context.Context, t *testing.T, name string, args ...string) {
	t.Helper()

	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Env = workspaceGoEnv()
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	if err := cmd.Run(); err != nil {
		t.Fatalf("%s %v: %v\n%s", name, args, err, output.String())
	}
}

func workspaceGoEnv() []string {
	env := os.Environ()
	out := make([]string, 0, len(env))
	for _, value := range env {
		if strings.HasPrefix(value, "GOWORK=") {
			continue
		}
		out = append(out, value)
	}
	return out
}
