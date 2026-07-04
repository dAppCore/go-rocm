// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"fmt"
	"strings"
)

func defaultCLIBackendName() string {
	current := currentCompileTargetForCLI(cliName())
	if current.Kind == "release-binary" && strings.TrimSpace(current.Backend) != "" {
		return normalizeBackendName(current.Backend)
	}
	return defaultBackendName
}

func validateCLIBackendDispatch(command, backend string) error {
	backend = normalizeBackendName(backend)
	if backend == "" || backend == "auto" {
		return nil
	}
	current := currentCompileTargetForCLI(cliName())
	if current.Kind != "release-binary" || normalizeBackendName(current.Backend) != backend {
		return nil
	}
	status := strings.TrimSpace(current.Labels["runtime_dispatch_status"])
	if status == "" || status == "active" {
		return nil
	}
	sidecars := strings.Join(current.Sidecars, ",")
	if sidecars == "" {
		sidecars = "none"
	}
	lane := strings.TrimSpace(current.Labels["runtime_lane"])
	if lane == "" {
		lane = backend
	}
	return fmt.Errorf("%s %s: %s runtime lane is compiled and packaged for %s (%s), but runtime dispatch is pending (runtime_dispatch_status=%s); use lthn-amd or pass -backend rocm on this machine until the %s backend is registered", cliName(), command, backend, lane, sidecars, status, backend)
}
