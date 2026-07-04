// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestDefaultStateContextRoutes_Good_StaticCatalogue(t *testing.T) {
	routes := DefaultStateContextRoutes()
	if len(routes) == 0 {
		t.Fatal("DefaultStateContextRoutes returned no routes")
	}
	byArchitecture := map[string]StateContextRoute{}
	for _, route := range routes {
		byArchitecture[route.Architecture] = route
	}
	text := byArchitecture["gemma4_text"]
	if !text.Matched() ||
		text.Contract != StateContextRegistryContract ||
		text.Name != StateContextRouteName ||
		text.Reference != "go_mlx_gemma4_retained_state" ||
		text.Runtime != StateContextRuntimeAPI ||
		text.RuntimeStatus != inference.FeatureRuntimeMetadataOnly ||
		text.Status != StateContextRouteExperimentalRuntime ||
		!text.Registered ||
		!text.NativeRuntime ||
		text.AttachedOnly ||
		!text.StateSession ||
		!text.SleepState ||
		!text.WakeState ||
		!text.ForkState ||
		!text.CaptureState ||
		!text.RestoreState ||
		!text.ResetState ||
		!text.RuntimeOwnedKV ||
		!text.PromptReplayRefused ||
		!text.RemainingContextDefault ||
		!text.ModelContextWindow ||
		!text.DeviceKVState ||
		!text.HIPDeviceMirror ||
		!text.PackageLocalKV ||
		!text.BlockBundleRefs ||
		!text.PortableRefs ||
		!text.RetainedStateRequired ||
		text.Staged ||
		text.Planned ||
		text.ContextWindow != 4096 ||
		text.DefaultStateBlockSize != 128 ||
		text.DefaultDeviceKVMode != "k-q8-v-q4" ||
		!slices.Contains(text.CacheModes, "retained-state") ||
		!slices.Equal(text.Capabilities, []inference.CapabilityID{
			inference.CapabilityStateBundle,
			inference.CapabilityStateWake,
			inference.CapabilityStateSleep,
			inference.CapabilityStateFork,
		}) ||
		text.Labels["engine_state_context_reference"] != "go_mlx_gemma4_retained_state" ||
		text.Labels["engine_state_context_capabilities"] != "state.bundle,state.wake,state.sleep,state.fork" {
		t.Fatalf("gemma4_text state context route = %+v, want native retained-state route", text)
	}
	assistant := byArchitecture["gemma4_assistant"]
	if !assistant.Matched() ||
		assistant.Reference != "go_mlx_gemma4_attached_drafter_retained_state" ||
		assistant.Status != StateContextRouteAttachedRuntime ||
		!assistant.AttachedOnly ||
		!assistant.AttachedDrafterState ||
		assistant.CaptureState ||
		assistant.RestoreState ||
		assistant.ResetState {
		t.Fatalf("gemma4_assistant state context route = %+v, want attached drafter state route", assistant)
	}
}

func TestStateContextRouteForInspection_Good_UsesLabels(t *testing.T) {
	inspectionLabels := map[string]string{
		"engine_architecture_profile":                 "gemma4_text",
		"engine_state_context_window":                 "16384",
		"engine_architecture_cache_hints":             "retained-state,q8",
		"device_kv_mode":                              "q8",
		"gemma4_size":                                 "27b",
		"production_quant_mode":                       "q6",
		"engine_model_context_window":                 "true",
		"engine_device_kv_state":                      "true",
		"attached_drafter_retained_state_required":    "true",
		"engine_state_context_attached_drafter_state": "ignored_engine_label",
	}
	inspection := &inference.ModelPackInspection{
		Path:   "/models/gemma4",
		Labels: inspectionLabels,
		Model: inference.ModelIdentity{
			Architecture:  "Gemma4ForCausalLM",
			ContextLength: 32768,
			Labels: map[string]string{
				"engine_architecture_resolved": "gemma4_text",
			},
		},
	}
	route, ok := StateContextRouteForInspection(inspection)
	if !ok ||
		route.Architecture != "gemma4_text" ||
		route.ContextWindow != 16384 ||
		route.DefaultDeviceKVMode != "q8" ||
		route.Gemma4Size != "27b" ||
		route.Gemma4QuantMode != "q6" ||
		!slices.Equal(route.CacheModes, []string{"retained-state", "q8"}) ||
		!route.ModelContextWindow ||
		!route.DeviceKVState ||
		!route.RetainedStateRequired ||
		!route.AttachedDrafterState ||
		route.Labels["engine_state_context_window"] != "16384" ||
		route.Labels["engine_state_context_default_device_kv_mode"] != "q8" ||
		route.Labels["engine_state_context_gemma4_quant_mode"] != "q6" {
		t.Fatalf("StateContextRouteForInspection = %+v ok=%v, want label-derived retained state route", route, ok)
	}
	inspectionLabels["engine_architecture_cache_hints"] = "mutated"
	route.CacheModes[0] = "mutated"
	next, ok := StateContextRouteForArchitecture("gemma4_text")
	if !ok ||
		!slices.Contains(next.CacheModes, "retained-state") ||
		slices.Contains(next.CacheModes, "mutated") {
		t.Fatalf("StateContextRouteForInspection leaked mutable state: %+v ok=%v", next, ok)
	}
}

func TestRegisterStateContextRoute_Good_ExtendsReactiveCatalogue(t *testing.T) {
	restoreRegisteredStateContextsForTest(t)

	RegisterStateContextRoute(StateContextRoute{})
	RegisterStateContextRoute(StateContextRoute{
		Architecture:        "fake-loader",
		Reference:           "first_state_context",
		StateSession:        true,
		DefaultDeviceKVMode: "q8",
	})
	RegisterStateContextRoute(StateContextRoute{
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

	if got := RegisteredStateContextArchitectures(); !slices.Equal(got, []string{"fake_loader"}) {
		t.Fatalf("RegisteredStateContextArchitectures = %v, want normalized replacement", got)
	}
	registeredRoutes := RegisteredStateContextRoutes()
	if len(registeredRoutes) != 1 ||
		registeredRoutes[0].Architecture != "fake_loader" ||
		registeredRoutes[0].Reference != "fake_retained_state" {
		t.Fatalf("RegisteredStateContextRoutes = %+v, want one replacement route", registeredRoutes)
	}
	routesForReplace := RegisteredStateContextRoutes()
	registeredRoutes[0].Labels["engine_state_context_reference"] = "mutated"
	registeredRoutes[0].CacheModes[0] = "mutated"
	nextRegistered, ok := RegisteredStateContextRouteForArchitecture("fake-loader")
	if !ok ||
		nextRegistered.Labels["engine_state_context_reference"] != "fake_retained_state" ||
		!slices.Equal(nextRegistered.CacheModes, []string{"retained-state", "q8"}) {
		t.Fatalf("RegisteredStateContextRoutes leaked mutable state: %+v ok=%v", nextRegistered, ok)
	}
	ReplaceRegisteredStateContextRoutes(routesForReplace)

	route, ok := StateContextRouteForArchitecture("fake-loader")
	if !ok ||
		route.Contract != StateContextRegistryContract ||
		route.Name != StateContextRouteName ||
		route.Architecture != "fake_loader" ||
		route.Family != "fake" ||
		route.Reference != "fake_retained_state" ||
		route.Runtime != StateContextRuntimeAPI ||
		route.RuntimeStatus != inference.FeatureRuntimeExperimental ||
		route.Status != StateContextRouteExperimentalRuntime ||
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
		t.Fatalf("StateContextRouteForArchitecture(fake-loader) = %+v ok=%v, want registered route", route, ok)
	}
	route.Labels["engine_state_context_reference"] = "mutated"
	route.CacheModes[0] = "mutated"
	route.Capabilities[0] = inference.CapabilityGenerate
	next, ok := StateContextRouteForArchitecture("fake-loader")
	if !ok ||
		next.Labels["engine_state_context_reference"] != "fake_retained_state" ||
		!slices.Equal(next.CacheModes, []string{"retained-state", "q8"}) ||
		next.Capabilities[0] != inference.CapabilityStateBundle {
		t.Fatalf("StateContextRouteForArchitecture leaked mutable state: %+v ok=%v", next, ok)
	}
	if !slices.ContainsFunc(DefaultStateContextRoutes(), func(route StateContextRoute) bool {
		return route.Architecture == "fake_loader" && route.Reference == "fake_retained_state"
	}) {
		t.Fatalf("DefaultStateContextRoutes missing fake_loader registration")
	}
}

func restoreRegisteredStateContextsForTest(t *testing.T) {
	t.Helper()
	routes := RegisteredStateContextRoutes()
	t.Cleanup(func() {
		ReplaceRegisteredStateContextRoutes(routes)
	})
}
