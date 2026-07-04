// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestRegisterROCmStateContextRoute_Good_ExtendsReactiveStateRegistry(t *testing.T) {
	restoreRegisteredROCmStateContextRoutesForTest(t)

	RegisterROCmStateContextRoute(ROCmStateContextRoute{})
	RegisterROCmStateContextRoute(ROCmStateContextRoute{
		Architecture:        "fake-loader",
		Reference:           "first_state_context",
		StateSession:        true,
		DefaultDeviceKVMode: "q8",
	})
	RegisterROCmStateContextRoute(ROCmStateContextRoute{
		Architecture:            "fake-loader",
		Family:                  "fake",
		Reference:               "fake_retained_state",
		NativeRuntime:           true,
		StateSession:            true,
		SleepState:              true,
		WakeState:               true,
		ForkState:               true,
		CaptureState:            true,
		RestoreState:            true,
		ResetState:              true,
		RuntimeOwnedKV:          true,
		PromptReplayRefused:     true,
		RemainingContextDefault: true,
		ModelContextWindow:      true,
		DeviceKVState:           true,
		HIPDeviceMirror:         true,
		PackageLocalKV:          true,
		BlockBundleRefs:         true,
		PortableRefs:            true,
		RetainedStateRequired:   true,
		ContextWindow:           8192,
		DefaultContextWindow:    4096,
		DefaultStateBlockSize:   256,
		DefaultDeviceKVMode:     "k-q8-v-q4",
		CacheModes:              []string{"retained-state", "q8"},
		StateBackends:           []string{"package-local-kv", "hip-device-mirror"},
	})

	if got := RegisteredROCmStateContextRouteArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmStateContextRouteArchitectures = %v, want normalized replacement under one architecture", got)
	}
	registered := RegisteredROCmStateContextRouteArchitectures()
	registered[0] = "mutated"
	if next := RegisteredROCmStateContextRouteArchitectures(); !slices.Equal(next, []string{"fake_loader"}) {
		t.Fatalf("RegisteredROCmStateContextRouteArchitectures returned mutable state: %v", next)
	}

	route, ok := ROCmStateContextRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != ROCmStateContextRegistryContract ||
		route.Name != rocmStateContextRegistryRouteName ||
		route.Architecture != "fake_loader" ||
		route.Family != "fake" ||
		route.Reference != "fake_retained_state" ||
		route.Runtime != rocmStateContextRuntimeAPI ||
		route.RuntimeStatus != inference.FeatureRuntimeExperimental ||
		route.Status != ROCmStateContextRouteExperimentalRuntime ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.StateSession ||
		!route.SleepState ||
		!route.WakeState ||
		!route.ForkState ||
		!route.CaptureState ||
		!route.RestoreState ||
		!route.ResetState ||
		!route.RuntimeOwnedKV ||
		!route.PromptReplayRefused ||
		!route.RemainingContextDefault ||
		!route.ModelContextWindow ||
		!route.DeviceKVState ||
		!route.HIPDeviceMirror ||
		!route.PackageLocalKV ||
		!route.BlockBundleRefs ||
		!route.PortableRefs ||
		!route.RetainedStateRequired ||
		route.Staged ||
		route.Planned ||
		route.ContextWindow != 8192 ||
		route.DefaultContextWindow != 4096 ||
		route.DefaultStateBlockSize != 256 ||
		route.DefaultDeviceKVMode != "k-q8-v-q4" ||
		!slices.Equal(route.CacheModes, []string{"retained-state", "q8"}) ||
		!slices.Equal(route.StateBackends, []string{"package-local-kv", "hip-device-mirror"}) ||
		!slices.Equal(route.Capabilities, []inference.CapabilityID{
			inference.CapabilityStateBundle,
			inference.CapabilityStateWake,
			inference.CapabilityStateSleep,
			inference.CapabilityStateFork,
		}) ||
		route.Labels["engine_state_context_reference"] != "fake_retained_state" ||
		route.Labels["engine_state_context_window"] != "8192" ||
		route.Labels["engine_state_context_runtime_owned_kv"] != "true" ||
		route.Labels["engine_state_context_capabilities"] != "state.bundle,state.wake,state.sleep,state.fork" {
		t.Fatalf("ROCmStateContextRouteForArchitecture(fake-loader) = %+v ok=%v, want registered state route", route, ok)
	}

	route.Labels["engine_state_context_reference"] = "mutated"
	route.CacheModes[0] = "mutated"
	route.Capabilities[0] = inference.CapabilityGenerate
	nextRoute, ok := ROCmStateContextRouteForArchitecture("fake-loader")
	if !ok ||
		nextRoute.Labels["engine_state_context_reference"] != "fake_retained_state" ||
		!slices.Equal(nextRoute.CacheModes, []string{"retained-state", "q8"}) ||
		nextRoute.Capabilities[0] != inference.CapabilityStateBundle {
		t.Fatalf("ROCmStateContextRouteForArchitecture leaked mutable state: %+v ok=%v", nextRoute, ok)
	}
	modelRoute, ok := rocmmodel.RegisteredStateContextRouteForArchitecture("fake-loader")
	if !ok ||
		modelRoute.Reference != "fake_retained_state" ||
		!slices.Equal(modelRoute.CacheModes, []string{"retained-state", "q8"}) ||
		!slices.Equal(modelRoute.StateBackends, []string{"package-local-kv", "hip-device-mirror"}) ||
		modelRoute.Labels["engine_state_context_reference"] != "fake_retained_state" {
		t.Fatalf("model.RegisteredStateContextRouteForArchitecture(fake-loader) = %+v ok=%v, want mirrored route", modelRoute, ok)
	}

	defaults := DefaultROCmStateContextRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmStateContextRoute) bool {
		return route.Architecture == "fake_loader" && route.Reference == "fake_retained_state"
	}) {
		t.Fatalf("DefaultROCmStateContextRoutes missing registered route: %+v", defaults)
	}

	profileRoute := ROCmStateContextRouteForProfile(ROCmModelProfile{
		Name:         "fake",
		Family:       "fake",
		Architecture: "fake_loader",
		ArchitectureProfile: ROCmArchitectureProfile{
			ID:     "fake_loader",
			Family: "fake",
		},
	})
	if profileRoute.Reference != "fake_retained_state" ||
		!profileRoute.StateSession ||
		!profileRoute.RuntimeOwnedKV ||
		profileRoute.ContextWindow != 8192 ||
		profileRoute.Labels["engine_state_context_reference"] != "fake_retained_state" {
		t.Fatalf("ROCmStateContextRouteForProfile registered route = %+v, want profile to use registered state route", profileRoute)
	}
}

func TestRegisterROCmStateContextRoute_Good_SeesModelPackageRegistration(t *testing.T) {
	restoreRegisteredROCmStateContextRoutesForTest(t)

	rocmmodel.RegisterStateContextRoute(rocmmodel.StateContextRoute{
		Architecture:            "folder-state",
		Family:                  "folder",
		Reference:               "folder_retained_state",
		NativeRuntime:           true,
		StateSession:            true,
		SleepState:              true,
		WakeState:               true,
		ForkState:               true,
		CaptureState:            true,
		RestoreState:            true,
		ResetState:              true,
		RuntimeOwnedKV:          true,
		PromptReplayRefused:     true,
		RemainingContextDefault: true,
		ModelContextWindow:      true,
		DeviceKVState:           true,
		HIPDeviceMirror:         true,
		PackageLocalKV:          true,
		BlockBundleRefs:         true,
		PortableRefs:            true,
		RetainedStateRequired:   true,
		ContextWindow:           8192,
		DefaultContextWindow:    4096,
		DefaultStateBlockSize:   256,
		DefaultDeviceKVMode:     "k-q8-v-q4",
		CacheModes:              []string{"retained-state", "q8"},
		StateBackends:           []string{"package-local-kv", "hip-device-mirror"},
	})

	route, ok := ROCmStateContextRouteForArchitecture("folder-state")
	if !ok ||
		route.Contract != ROCmStateContextRegistryContract ||
		route.Name != rocmStateContextRegistryRouteName ||
		route.Architecture != "folder_state" ||
		route.Family != "folder" ||
		route.Reference != "folder_retained_state" ||
		route.Runtime != rocmStateContextRuntimeAPI ||
		route.RuntimeStatus != inference.FeatureRuntimeExperimental ||
		route.Status != ROCmStateContextRouteExperimentalRuntime ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.StateSession ||
		!route.SleepState ||
		!route.WakeState ||
		!route.ForkState ||
		!route.CaptureState ||
		!route.RestoreState ||
		!route.ResetState ||
		!route.RuntimeOwnedKV ||
		!route.PromptReplayRefused ||
		!route.RemainingContextDefault ||
		!route.ModelContextWindow ||
		!route.DeviceKVState ||
		!route.HIPDeviceMirror ||
		!route.PackageLocalKV ||
		!route.BlockBundleRefs ||
		!route.PortableRefs ||
		!route.RetainedStateRequired ||
		route.Staged ||
		route.Planned ||
		route.ContextWindow != 8192 ||
		route.DefaultContextWindow != 4096 ||
		route.DefaultStateBlockSize != 256 ||
		route.DefaultDeviceKVMode != "k-q8-v-q4" ||
		!slices.Equal(route.CacheModes, []string{"retained-state", "q8"}) ||
		!slices.Equal(route.StateBackends, []string{"package-local-kv", "hip-device-mirror"}) ||
		route.Labels["engine_state_context_reference"] != "folder_retained_state" {
		t.Fatalf("ROCmStateContextRouteForArchitecture(folder-state) = %+v ok=%v, want model package route", route, ok)
	}
	defaults := DefaultROCmStateContextRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmStateContextRoute) bool {
		return route.Architecture == "folder_state" && route.Reference == "folder_retained_state"
	}) {
		t.Fatalf("DefaultROCmStateContextRoutes missing model package route: %+v", defaults)
	}
	profileRoute := ROCmStateContextRouteForProfile(ROCmModelProfile{
		Name:         "folder",
		Family:       "folder",
		Architecture: "folder_state",
	})
	if profileRoute.Reference != "folder_retained_state" ||
		!profileRoute.StateSession ||
		!profileRoute.RuntimeOwnedKV ||
		profileRoute.ContextWindow != 8192 ||
		profileRoute.Labels["engine_state_context_reference"] != "folder_retained_state" {
		t.Fatalf("ROCmStateContextRouteForProfile(folder_state) = %+v, want model package state route", profileRoute)
	}
}

func TestRegisterROCmStateContextRoute_Good_OverridesBuiltinStateRoute(t *testing.T) {
	restoreRegisteredROCmStateContextRoutesForTest(t)

	RegisterROCmStateContextRoute(ROCmStateContextRoute{
		Architecture:            "gemma4_text",
		Family:                  "gemma4",
		Reference:               "registered_gemma4_retained_state",
		NativeRuntime:           true,
		StateSession:            true,
		SleepState:              true,
		WakeState:               true,
		ForkState:               true,
		CaptureState:            true,
		RestoreState:            true,
		ResetState:              true,
		RuntimeOwnedKV:          true,
		PromptReplayRefused:     true,
		RemainingContextDefault: true,
		ModelContextWindow:      true,
		DeviceKVState:           true,
		HIPDeviceMirror:         true,
		PackageLocalKV:          true,
		BlockBundleRefs:         true,
		PortableRefs:            true,
		RetainedStateRequired:   true,
		ContextWindow:           65536,
		DefaultStateBlockSize:   256,
		DefaultDeviceKVMode:     "q8",
		CacheModes:              []string{"retained-state", "q8"},
	})

	route, ok := ROCmStateContextRouteForArchitecture("Gemma4ForCausalLM")
	if !ok ||
		route.Architecture != "gemma4_text" ||
		route.Reference != "registered_gemma4_retained_state" ||
		route.Runtime != rocmStateContextRuntimeAPI ||
		route.Status != ROCmStateContextRouteExperimentalRuntime ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.StateSession ||
		!route.RuntimeOwnedKV ||
		!route.PromptReplayRefused ||
		!route.RemainingContextDefault ||
		!route.DeviceKVState ||
		!route.HIPDeviceMirror ||
		!route.PackageLocalKV ||
		!route.BlockBundleRefs ||
		route.Staged ||
		route.Planned ||
		route.ContextWindow != 65536 ||
		route.DefaultStateBlockSize != 256 ||
		route.DefaultDeviceKVMode != "q8" ||
		!slices.Equal(route.CacheModes, []string{"retained-state", "q8"}) ||
		route.Labels["engine_state_context_reference"] != "registered_gemma4_retained_state" ||
		route.Labels["engine_state_context_default_device_kv_mode"] != "q8" {
		t.Fatalf("ROCmStateContextRouteForArchitecture(gemma4_text override) = %+v ok=%v, want registered state route", route, ok)
	}

	profile, ok := ResolveROCmModelProfile("/models/gemma4", inference.ModelIdentity{Architecture: "Gemma4ForCausalLM"})
	if !ok ||
		profile.StateContextRoute.Reference != "registered_gemma4_retained_state" ||
		!profile.StateContextRoute.NativeRuntime ||
		!profile.StateContextRoute.RuntimeOwnedKV ||
		profile.StateContextRoute.ContextWindow != 65536 ||
		profile.StateContextRoute.DefaultDeviceKVMode != "q8" ||
		profile.StateContextRoute.Labels["engine_state_context_reference"] != "registered_gemma4_retained_state" {
		t.Fatalf("ResolveROCmModelProfile(gemma4 registered state override) = %+v ok=%v, want profile to expose registered state route", profile, ok)
	}
}

func restoreRegisteredROCmStateContextRoutesForTest(t *testing.T) {
	t.Helper()
	modelRoutes := rocmmodel.RegisteredStateContextRoutes()

	t.Cleanup(func() {
		restoreRoutes := make([]rocmmodel.StateContextRoute, 0, len(modelRoutes))
		for _, route := range modelRoutes {
			restoreRoutes = append(restoreRoutes, route.Clone())
		}
		rocmmodel.ReplaceRegisteredStateContextRoutes(restoreRoutes)
	})
}
