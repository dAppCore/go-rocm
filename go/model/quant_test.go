// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func TestRegisterQuantScheme_Good_ExtendsModelQuantRegistry(t *testing.T) {
	restoreQuantRegistriesForTest(t)

	RegisterQuantScheme(QuantScheme{})
	RegisterQuantScheme(QuantScheme{
		Kind:    "fake-fp3",
		Bits:    3,
		Loader:  "fake_fp3_first",
		Planned: true,
	})
	RegisterQuantScheme(QuantScheme{
		Kind:    "fake-fp3",
		Bits:    3,
		Loader:  "fake_fp3_loader",
		Source:  "test_override",
		Planned: true,
	})

	if got := RegisteredQuantSchemeKinds(); !slices.Equal(got, []string{"fake_fp3"}) {
		t.Fatalf("RegisteredQuantSchemeKinds = %v, want normalized replacement under one kind", got)
	}
	kinds := RegisteredQuantSchemeKinds()
	kinds[0] = "mutated"
	if next := RegisteredQuantSchemeKinds(); !slices.Equal(next, []string{"fake_fp3"}) {
		t.Fatalf("RegisteredQuantSchemeKinds returned mutable state: %v", next)
	}

	scheme, ok := QuantSchemeForKind("fake-fp3")
	if !ok ||
		scheme.Contract != QuantSchemeRegistryContract ||
		scheme.Kind != "fake_fp3" ||
		scheme.Bits != 3 ||
		scheme.Loader != "fake_fp3_loader" ||
		scheme.Source != "test_override" ||
		scheme.Runtime != QuantSchemeRuntimePlannedHIP ||
		scheme.RuntimeStatus != inference.FeatureRuntimePlanned ||
		!scheme.Registered ||
		scheme.NativeRuntime ||
		!scheme.Planned ||
		scheme.Labels["engine_quant_scheme_kind"] != "fake_fp3" ||
		scheme.Labels["engine_quant_scheme_loader"] != "fake_fp3_loader" {
		t.Fatalf("QuantSchemeForKind(fake-fp3) = %+v ok=%v", scheme, ok)
	}
	scheme.Labels["engine_quant_scheme_loader"] = "mutated"
	nextScheme, ok := QuantSchemeForKind("fake-fp3")
	if !ok || nextScheme.Labels["engine_quant_scheme_loader"] != "fake_fp3_loader" {
		t.Fatalf("QuantSchemeForKind leaked mutable labels: %+v ok=%v", nextScheme, ok)
	}
}

func TestRegisterQuantLoaderRoute_Good_ExtendsModelQuantLoaderRegistry(t *testing.T) {
	restoreQuantRegistriesForTest(t)

	RegisterQuantLoaderRoute(QuantLoaderRoute{})
	RegisterQuantLoaderRoute(QuantLoaderRoute{
		Size:           "Fake",
		Mode:           "fake-fp3",
		Loader:         "fake_fp3_first",
		GenerateStatus: QuantGeneratePlannedOnly,
		RunnableOnCard: true,
	})
	RegisterQuantLoaderRoute(QuantLoaderRoute{
		Family:         "fake",
		Architecture:   "qwen3",
		Size:           "Fake",
		Mode:           "fake-fp3",
		Bits:           3,
		Group:          96,
		Loader:         "fake_fp3_loader",
		Runtime:        QuantSchemeRuntimePlannedHIP,
		GenerateStatus: QuantGeneratePlannedOnly,
		RunnableOnCard: true,
		RequiresBench:  true,
	})

	if got := RegisteredQuantLoaderRoutePacks(); !slices.Equal(got, []string{"Fake:fake_fp3"}) {
		t.Fatalf("RegisteredQuantLoaderRoutePacks = %v, want normalized replacement under one pack", got)
	}
	route, ok := RegisteredQuantLoaderRouteForToken("fake-fp3")
	if !ok ||
		route.Contract != QuantLoaderRegistryContract ||
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
		t.Fatalf("RegisteredQuantLoaderRouteForToken(fake-fp3) = %+v ok=%v", route, ok)
	}
	route.Labels["engine_quant_loader"] = "mutated"
	nextRoute, ok := RegisteredQuantLoaderRouteForToken("Fake:fake_fp3")
	if !ok || nextRoute.Labels["engine_quant_loader"] != "fake_fp3_loader" {
		t.Fatalf("RegisteredQuantLoaderRouteForToken leaked mutable labels: %+v ok=%v", nextRoute, ok)
	}
}

func TestQuantLoaderRouteForPack_Good_MatchesProductionPackContract(t *testing.T) {
	route := QuantLoaderRouteForPack(QuantLoaderPack{
		Name:           "e2b-6bit",
		Size:           "E2B",
		ModelID:        "lmstudio-community/gemma-4-E2B-it-MLX-6bit",
		LockedModelID:  "mlx-community/gemma-4-e2b-it-6bit",
		Bits:           6,
		QuantMode:      "affine",
		QuantGroup:     64,
		Runtime:        QuantRuntimeMLXAffine,
		GenerateStatus: QuantGenerateLinked,
		ProductRole:    "default",
		Supported:      true,
		RunnableOnCard: true,
	})

	if route.Pack != "E2B:q6" ||
		route.PackName != "e2b-6bit" ||
		route.Mode != "q6" ||
		route.Loader != "gemma4_affine" ||
		route.Target != "generate" ||
		!route.Registered ||
		!route.NativeRuntime ||
		route.Staged ||
		route.Labels["engine_quant_loader_pack"] != "E2B:q6" ||
		route.Labels["engine_quant_loader_mode"] != "q6" {
		t.Fatalf("QuantLoaderRouteForPack = %+v, want linked q6 affine route", route)
	}
}

func restoreQuantRegistriesForTest(t *testing.T) {
	t.Helper()
	schemes := RegisteredQuantSchemes()
	loaders := RegisteredQuantLoaderRoutes()
	t.Cleanup(func() {
		restoreSchemes := make([]QuantScheme, len(schemes))
		for i, scheme := range schemes {
			restoreSchemes[i] = scheme.Clone()
		}
		ReplaceRegisteredQuantSchemes(restoreSchemes)

		restoreLoaders := make([]QuantLoaderRoute, len(loaders))
		for i, route := range loaders {
			restoreLoaders[i] = route.Clone()
		}
		ReplaceRegisteredQuantLoaderRoutes(restoreLoaders)
	})
}
