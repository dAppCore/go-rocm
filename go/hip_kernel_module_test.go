// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	core "dappco.re/go"
)

func TestHIPKernelModuleResolverEnvWins_Good(t *testing.T) {
	t.Setenv(hipKernelModuleEnv, " /tmp/explicit.hsaco ")
	oldExecutable := hipKernelModuleExecutable
	hipKernelModuleExecutable = func() (string, error) {
		return filepath.Join(t.TempDir(), "lthn-amd"), nil
	}
	t.Cleanup(func() { hipKernelModuleExecutable = oldExecutable })

	resolution := resolveHIPKernelModule()
	core.AssertEqual(t, "/tmp/explicit.hsaco", resolution.Path)
	core.AssertEqual(t, "env", resolution.Source)
}

func TestHIPKernelModuleResolverExecutableSidecarLinksKernelSet_Good(t *testing.T) {
	t.Setenv(hipKernelModuleEnv, "")
	dir := t.TempDir()
	sidecar := filepath.Join(dir, "rocm_kernels_gfx1100.hsaco")
	core.RequireNoError(t, os.WriteFile(sidecar, []byte("hsaco"), 0o644))
	oldExecutable := hipKernelModuleExecutable
	hipKernelModuleExecutable = func() (string, error) {
		return filepath.Join(dir, "lthn-amd"), nil
	}
	t.Cleanup(func() { hipKernelModuleExecutable = oldExecutable })

	resolution := resolveHIPKernelModule()
	core.AssertEqual(t, sidecar, resolution.Path)
	core.AssertEqual(t, "sidecar", resolution.Source)

	kernels, ok := newHIPRuntimeKernelSet(&fakeHIPDriver{available: true}).(hipNativeProjectionKernelSet)
	if !ok {
		t.Fatalf("newHIPRuntimeKernelSet() = %T, want linked kernel set from sidecar", newHIPRuntimeKernelSet(&fakeHIPDriver{available: true}))
	}
	status := kernels.Status()
	core.AssertEqual(t, hipKernelStatusLinked, status.Projection)
	if !strings.Contains(status.Reason, "packaged HSACO sidecar") {
		t.Fatalf("status reason = %q, want sidecar source", status.Reason)
	}
}

func TestHIPKernelModuleResolverBuildBinSiblingKernels_Good(t *testing.T) {
	t.Setenv(hipKernelModuleEnv, "")
	root := t.TempDir()
	binDir := filepath.Join(root, "build", "bin")
	kernelDir := filepath.Join(root, "build", "kernels")
	core.RequireNoError(t, os.MkdirAll(binDir, 0o755))
	core.RequireNoError(t, os.MkdirAll(kernelDir, 0o755))
	sidecar := filepath.Join(kernelDir, "rocm_kernels_gfx1100.hsaco")
	core.RequireNoError(t, os.WriteFile(sidecar, []byte("hsaco"), 0o644))
	oldExecutable := hipKernelModuleExecutable
	hipKernelModuleExecutable = func() (string, error) {
		return filepath.Join(binDir, "lthn-amd"), nil
	}
	t.Cleanup(func() { hipKernelModuleExecutable = oldExecutable })

	resolution := resolveHIPKernelModule()
	core.AssertEqual(t, filepath.Clean(sidecar), resolution.Path)
	core.AssertEqual(t, "sidecar", resolution.Source)
}
