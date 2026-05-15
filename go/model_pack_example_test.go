// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"encoding/binary"
	"os"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func Example_inspectModelPack() {
	dir, cleanup, err := exampleGemma4ModelPack()
	if err != nil {
		panic(err)
	}
	defer cleanup()

	backend := newROCmBackendWithRuntime(&fakeNativeRuntime{device: nativeDeviceInfo{MemoryBytes: 17163091968, Name: "gfx1100"}})
	inspection, err := backend.InspectModelPack(context.Background(), dir)
	if err != nil {
		panic(err)
	}

	core.Println(
		inspection.Model.Architecture,
		inspection.Format,
		inspection.Supported,
		inspection.Labels["memory_plan_machine_class"],
		inspection.Labels["memory_plan_cache_mode"],
	)
	// Output: gemma4_text safetensors true rocm-16gb k-q8-v-q4
}

func Example_planModelFit() {
	report, err := (&rocmBackend{}).PlanModelFit(context.Background(), inference.ModelIdentity{
		Architecture:  "gemma4_text",
		QuantBits:     4,
		QuantGroup:    64,
		ContextLength: 131072,
		NumLayers:     35,
		HiddenSize:    1536,
		Labels: map[string]string{
			"attention_full_layers":    "7",
			"attention_sliding_layers": "28",
			"sliding_window":           "512",
		},
	}, 17163091968)
	if err != nil {
		panic(err)
	}

	core.Println(report.ArchitectureOK, report.QuantizationOK, report.MemoryPlan.MachineClass, report.MemoryPlan.CacheMode)
	// Output: true true rocm-16gb k-q8-v-q4
}

func exampleGemma4ModelPack() (string, func(), error) {
	dir, err := os.MkdirTemp("", "go-rocm-gemma4-example-*")
	if err != nil {
		return "", nil, err
	}
	cleanup := func() { _ = os.RemoveAll(dir) }
	config := `{
		"model_type":"Gemma4ForCausalLM",
		"text_config":{
			"hidden_size":1536,
			"num_hidden_layers":35,
			"vocab_size":256000,
			"max_position_embeddings":131072,
			"num_attention_heads":8,
			"num_key_value_heads":4,
			"head_dim":256,
			"sliding_window":512,
			"layer_types":["full_attention","sliding_attention"]
		},
		"quantization_config":{"bits":4,"group_size":64,"weight_format":"mlx_q4"}
	}`
	if result := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(config), 0o644); !result.OK {
		cleanup()
		return "", nil, result.Value.(error)
	}
	if err := writeExampleSafetensors(core.PathJoin(dir, "model.safetensors")); err != nil {
		cleanup()
		return "", nil, err
	}
	return dir, cleanup, nil
}

func writeExampleSafetensors(path string) error {
	header := []byte(`{"language_model.model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[2,4],"data_offsets":[0,16]},"__metadata__":{"format":"pt"}}`)
	buf := core.NewBuffer()
	if err := binary.Write(buf, binary.LittleEndian, uint64(len(header))); err != nil {
		return err
	}
	if _, err := buf.Write(header); err != nil {
		return err
	}
	if _, err := buf.Write(make([]byte, 16)); err != nil {
		return err
	}
	result := core.WriteFile(path, buf.Bytes(), 0o644)
	if !result.OK {
		return result.Value.(error)
	}
	return nil
}
