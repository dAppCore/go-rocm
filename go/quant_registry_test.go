// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/rocm/model"
)

func TestRegisterROCmQuantScheme_Good_ExtendsReactiveQuantRegistry(t *testing.T) {
	restoreRegisteredROCmQuantRegistriesForTest(t)

	RegisterROCmQuantScheme(ROCmQuantScheme{})
	RegisterROCmQuantScheme(ROCmQuantScheme{
		Kind:    "fake-fp3",
		Bits:    3,
		Loader:  "fake_fp3_first",
		Planned: true,
	})
	RegisterROCmQuantScheme(ROCmQuantScheme{
		Kind:    "fake-fp3",
		Bits:    3,
		Loader:  "fake_fp3_loader",
		Source:  "test_override",
		Planned: true,
	})

	if got := RegisteredROCmQuantSchemeKinds(); !slices.Equal(got, []string{"fake_fp3"}) {
		t.Fatalf("RegisteredROCmQuantSchemeKinds = %v, want normalized replacement under one kind", got)
	}
	kinds := RegisteredROCmQuantSchemeKinds()
	kinds[0] = "mutated"
	if next := RegisteredROCmQuantSchemeKinds(); !slices.Equal(next, []string{"fake_fp3"}) {
		t.Fatalf("RegisteredROCmQuantSchemeKinds returned mutable state: %v", next)
	}

	scheme, ok := ROCmQuantSchemeForKind("fake-fp3")
	if !ok ||
		scheme.Contract != ROCmQuantSchemeRegistryContract ||
		scheme.Kind != "fake_fp3" ||
		scheme.Bits != 3 ||
		scheme.Loader != "fake_fp3_loader" ||
		scheme.Source != "test_override" ||
		scheme.Runtime != rocmQuantSchemeRuntimePlannedHIP ||
		scheme.RuntimeStatus != inference.FeatureRuntimePlanned ||
		!scheme.Registered ||
		scheme.NativeRuntime ||
		!scheme.Planned ||
		scheme.Labels["engine_quant_scheme_kind"] != "fake_fp3" ||
		scheme.Labels["engine_quant_scheme_loader"] != "fake_fp3_loader" {
		t.Fatalf("ROCmQuantSchemeForKind(fake-fp3) = %+v ok=%v, want registered planned quant scheme", scheme, ok)
	}
	scheme.Labels["engine_quant_scheme_loader"] = "mutated"
	nextScheme, ok := ROCmQuantSchemeForKind("fake-fp3")
	if !ok || nextScheme.Labels["engine_quant_scheme_loader"] != "fake_fp3_loader" {
		t.Fatalf("ROCmQuantSchemeForKind leaked mutable labels: %+v ok=%v", nextScheme, ok)
	}
	defaults := DefaultROCmQuantSchemes()
	if !slices.ContainsFunc(defaults, func(scheme ROCmQuantScheme) bool {
		return scheme.Kind == "fake_fp3" && scheme.Loader == "fake_fp3_loader"
	}) {
		t.Fatalf("DefaultROCmQuantSchemes missing registered scheme: %+v", defaults)
	}
}

func TestRegisterROCmQuantLoaderRoute_Good_ExtendsReactiveQuantLoaderRegistry(t *testing.T) {
	restoreRegisteredROCmQuantRegistriesForTest(t)

	RegisterROCmQuantLoaderRoute(ROCmQuantLoaderRoute{})
	RegisterROCmQuantLoaderRoute(ROCmQuantLoaderRoute{
		Size:           "Fake",
		Mode:           "fake-fp3",
		Loader:         "fake_fp3_first",
		GenerateStatus: Gemma4GeneratePlannedOnly,
		RunnableOnCard: true,
	})
	RegisterROCmQuantLoaderRoute(ROCmQuantLoaderRoute{
		Family:         "fake",
		Architecture:   "qwen3",
		Size:           "Fake",
		Mode:           "fake-fp3",
		Bits:           3,
		Group:          96,
		Loader:         "fake_fp3_loader",
		Runtime:        rocmQuantSchemeRuntimePlannedHIP,
		GenerateStatus: Gemma4GeneratePlannedOnly,
		RunnableOnCard: true,
		RequiresBench:  true,
	})

	if got := RegisteredROCmQuantLoaderRoutePacks(); !slices.Equal(got, []string{"Fake:fake_fp3"}) {
		t.Fatalf("RegisteredROCmQuantLoaderRoutePacks = %v, want normalized replacement under one pack", got)
	}
	packs := RegisteredROCmQuantLoaderRoutePacks()
	packs[0] = "mutated"
	if next := RegisteredROCmQuantLoaderRoutePacks(); !slices.Equal(next, []string{"Fake:fake_fp3"}) {
		t.Fatalf("RegisteredROCmQuantLoaderRoutePacks returned mutable state: %v", next)
	}

	route, ok := ROCmQuantLoaderRouteForMode("fake-fp3")
	if !ok ||
		route.Contract != ROCmQuantLoaderRegistryContract ||
		route.Family != "fake" ||
		route.Architecture != "qwen3" ||
		route.Pack != "Fake:fake_fp3" ||
		route.Mode != "fake_fp3" ||
		route.Bits != 3 ||
		route.Group != 96 ||
		route.Loader != "fake_fp3_loader" ||
		route.Target != "planned" ||
		!route.Registered ||
		route.NativeRuntime ||
		!route.Planned ||
		!route.Staged ||
		!route.RequiresBench ||
		route.Labels["engine_quant_loader_mode"] != "fake_fp3" ||
		route.Labels["engine_quant_loader_target"] != "planned" {
		t.Fatalf("ROCmQuantLoaderRouteForMode(fake-fp3) = %+v ok=%v, want registered planned quant loader", route, ok)
	}
	route.Labels["engine_quant_loader"] = "mutated"
	nextRoute, ok := ROCmQuantLoaderRouteForMode("Fake:fake_fp3")
	if !ok || nextRoute.Labels["engine_quant_loader"] != "fake_fp3_loader" {
		t.Fatalf("ROCmQuantLoaderRouteForMode leaked mutable labels: %+v ok=%v", nextRoute, ok)
	}
	defaults := DefaultROCmQuantLoaderRoutes()
	if !slices.ContainsFunc(defaults, func(route ROCmQuantLoaderRoute) bool {
		return route.Pack == "Fake:fake_fp3" && route.Loader == "fake_fp3_loader"
	}) {
		t.Fatalf("DefaultROCmQuantLoaderRoutes missing registered route: %+v", defaults)
	}

	identity := inference.ModelIdentity{
		Path:         "/models/fake",
		Architecture: "qwen3",
		Labels: map[string]string{
			"engine_quant_loader_mode": "fake-fp3",
		},
	}
	byIdentity, ok := ROCmQuantLoaderRouteForIdentity("", identity)
	if !ok || byIdentity.Pack != "Fake:fake_fp3" || byIdentity.Loader != "fake_fp3_loader" {
		t.Fatalf("ROCmQuantLoaderRouteForIdentity registered route = %+v ok=%v, want label-selected quant loader", byIdentity, ok)
	}
}

func TestRegisterROCmQuantLoaderRoute_Good_OverridesBuiltinQuantRoute(t *testing.T) {
	restoreRegisteredROCmQuantRegistriesForTest(t)

	RegisterROCmQuantLoaderRoute(ROCmQuantLoaderRoute{
		Pack:           "E2B:q6",
		Family:         "gemma4",
		Architecture:   "gemma4_text",
		Size:           "E2B",
		Mode:           "q6",
		Bits:           6,
		Group:          64,
		Loader:         "registered_q6_loader",
		Runtime:        Gemma4RuntimeMLXAffine,
		GenerateStatus: Gemma4GenerateLinked,
		RunnableOnCard: true,
	})

	route, ok := ROCmQuantLoaderRouteForMode("E2B:q6")
	if !ok ||
		route.Pack != "E2B:q6" ||
		route.Loader != "registered_q6_loader" ||
		route.Target != "generate" ||
		!route.Registered ||
		!route.NativeRuntime ||
		!route.RunnableOnCard ||
		route.Planned ||
		route.LoadOnly {
		t.Fatalf("ROCmQuantLoaderRouteForMode(E2B:q6 override) = %+v ok=%v, want registered override", route, ok)
	}
}

func restoreRegisteredROCmQuantRegistriesForTest(t *testing.T) {
	t.Helper()
	schemes := rocmmodel.RegisteredQuantSchemes()
	routes := rocmmodel.RegisteredQuantLoaderRoutes()

	t.Cleanup(func() {
		restoreSchemes := make([]rocmmodel.QuantScheme, len(schemes))
		for i, scheme := range schemes {
			restoreSchemes[i] = scheme.Clone()
		}
		rocmmodel.ReplaceRegisteredQuantSchemes(restoreSchemes)

		restoreRoutes := make([]rocmmodel.QuantLoaderRoute, len(routes))
		for i, route := range routes {
			restoreRoutes[i] = route.Clone()
		}
		rocmmodel.ReplaceRegisteredQuantLoaderRoutes(restoreRoutes)
	})
}
