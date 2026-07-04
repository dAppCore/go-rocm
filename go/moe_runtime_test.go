// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import "testing"

func TestMoETextLayersRuntimeAvailable_Good(t *testing.T) {
	layers := []moeRuntimeTestLayer{
		{dense: true},
		{dense: true, sparse: true, router: true, experts: true},
		{dense: true},
	}
	summary := SummarizeMoETextLayersRuntime(layers, moeRuntimeTestParts)
	if !summary.Available || summary.Layers != 3 || summary.DenseLayers != 2 || summary.SparseLayers != 1 {
		t.Fatalf("summary = %+v, want available 2 dense 1 sparse", summary)
	}
	if !MoETextLayersRuntimeAvailable(layers, moeRuntimeTestParts) {
		t.Fatal("MoETextLayersRuntimeAvailable() = false, want true")
	}
}

func TestMoETextLayersRuntimeAvailable_Bad(t *testing.T) {
	readyDense := moeRuntimeTestLayer{dense: true}
	readySparse := moeRuntimeTestLayer{dense: true, sparse: true, router: true, experts: true}
	cases := []struct {
		name   string
		layers []moeRuntimeTestLayer
		parts  func(moeRuntimeTestLayer) MoETextLayerParts
	}{
		{name: "empty"},
		{name: "nil-parts", layers: []moeRuntimeTestLayer{readyDense}},
		{name: "not-ok", layers: []moeRuntimeTestLayer{{dense: true, ok: false}}, parts: moeRuntimeTestPartsExplicitOK},
		{name: "missing-dense", layers: []moeRuntimeTestLayer{{}}, parts: moeRuntimeTestParts},
		{name: "moe-missing-router", layers: []moeRuntimeTestLayer{{dense: true, sparse: true, experts: true}}, parts: moeRuntimeTestParts},
		{name: "moe-missing-experts", layers: []moeRuntimeTestLayer{{dense: true, sparse: true, router: true}}, parts: moeRuntimeTestParts},
		{name: "second-layer-bad", layers: []moeRuntimeTestLayer{readySparse, {}}, parts: moeRuntimeTestParts},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if MoETextLayersRuntimeAvailable(tc.layers, tc.parts) {
				t.Fatal("MoETextLayersRuntimeAvailable() = true, want false")
			}
			if summary := SummarizeMoETextLayersRuntime(tc.layers, tc.parts); summary.Available {
				t.Fatalf("summary = %+v, want unavailable", summary)
			}
		})
	}
}

func TestMoETextLayerRuntimeReady_Good(t *testing.T) {
	if !MoETextLayerRuntimeReady(MoETextLayerParts{DenseReady: true, OK: true}) {
		t.Fatal("dense layer readiness = false, want true")
	}
	if !MoETextLayerRuntimeReady(MoETextLayerParts{DenseReady: true, IsMoE: true, RouterReady: true, ExpertsReady: true, OK: true}) {
		t.Fatal("sparse layer readiness = false, want true")
	}
	if MoETextLayerRuntimeReady(MoETextLayerParts{DenseReady: true, IsMoE: true, RouterReady: true, OK: true}) {
		t.Fatal("sparse layer without experts = true, want false")
	}
}

type moeRuntimeTestLayer struct {
	dense   bool
	sparse  bool
	router  bool
	experts bool
	ok      bool
}

func moeRuntimeTestParts(layer moeRuntimeTestLayer) MoETextLayerParts {
	return MoETextLayerParts{
		DenseReady:   layer.dense,
		IsMoE:        layer.sparse,
		RouterReady:  layer.router,
		ExpertsReady: layer.experts,
		OK:           true,
	}
}

func moeRuntimeTestPartsExplicitOK(layer moeRuntimeTestLayer) MoETextLayerParts {
	parts := moeRuntimeTestParts(layer)
	parts.OK = layer.ok
	return parts
}
