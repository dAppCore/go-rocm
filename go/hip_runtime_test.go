// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestHIPRuntime_LoadModelAllocatesAndCopiesGGUFTensors_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	driver := &fakeHIPDriver{
		available: true,
		device:    nativeDeviceInfo{Name: "gfx1100", MemoryBytes: 16 * memoryGiB, FreeBytes: 12 * memoryGiB, Driver: "fake"},
	}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3", NumLayers: 1, QuantBits: 32},
		DataOffset: dataOffset,
		Tensors: []nativeTensorInfo{{
			Name:     "tok_embeddings.weight",
			Type:     0,
			Offset:   0,
			ByteSize: 16,
		}, {
			Name:     "output.weight",
			Type:     0,
			Offset:   16,
			ByteSize: 16,
		}},
	})

	core.AssertNoError(t, err)
	core.AssertNotNil(t, model)
	core.AssertEqual(t, []uint64{16, 16}, driver.allocations)
	core.AssertEqual(t, []uint64{16, 16}, driver.copies)
	stream, errFn := model.Generate(context.Background(), "hello", inference.DefaultGenerateConfig())
	for range stream {
	}
	core.AssertError(t, errFn())
	core.AssertNoError(t, model.Close())
	core.AssertEqual(t, 2, len(driver.frees))
}

func TestHIPRuntime_LoadModelLinksProjectionKernelWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-projection.hsaco")
	driver := &fakeHIPDriver{
		available: true,
		device:    nativeDeviceInfo{Name: "gfx1100", MemoryBytes: 16 * memoryGiB, FreeBytes: 12 * memoryGiB, Driver: "fake"},
	}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3", NumLayers: 1, QuantBits: 32},
		DataOffset: dataOffset,
		Tensors: []nativeTensorInfo{{
			Name:     "tok_embeddings.weight",
			Type:     0,
			Offset:   0,
			ByteSize: 16,
		}, {
			Name:     "output.weight",
			Type:     0,
			Offset:   16,
			ByteSize: 16,
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	status := loaded.KernelStatus()
	core.AssertEqual(t, hipKernelStatusNotLinked, status.Decode)
	core.AssertEqual(t, hipKernelStatusNotLinked, status.Prefill)
	core.AssertEqual(t, hipKernelStatusLinked, status.Projection)
	core.AssertEqual(t, hipKernelStatusLinked, status.CrossEntropy)
	core.AssertEqual(t, hipKernelStatusLinked, status.Distillation)
	core.AssertEqual(t, hipKernelStatusLinked, status.GRPO)

	projected, err := loaded.Project(context.Background(), hipProjectionRequest{
		Input: []float32{1, 2},
		FP16:  []uint16{0x3c00, 0x4000},
		Bias:  []float32{0.5},
		Rows:  1,
		Cols:  2,
	})
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{5.5}, projected, 0)
	core.AssertEqual(t, 1, len(driver.launches))
	core.AssertEqual(t, hipKernelNameProjection, driver.launches[0].Name)

	stream, streamErr := loaded.Generate(context.Background(), "hello", inference.DefaultGenerateConfig())
	for range stream {
	}
	core.AssertError(t, streamErr())
	core.AssertContains(t, streamErr().Error(), "native decode kernels are not linked yet")
}

func TestHIPRuntime_LoadModelRunsTinyPrefillDecodeWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-tiny.hsaco")
	fixture := hipReferenceTinyLMFixture()
	for _, tt := range []struct {
		name             string
		outputType       uint32
		outputTypeName   string
		outputPayload    []byte
		codebookPayload  []byte
		codebookValues   []float32
		outputEncoding   uint32
		outputScale      float32
		outputWeightByte uint32
		wantJANGTQ       bool
		wantCodebook     bool
	}{{
		name:             "f32-output",
		outputType:       0,
		outputEncoding:   hipTinyOutputWeightEncodingFP32,
		outputWeightByte: 24,
	}, {
		name:             "f16-output",
		outputType:       1,
		outputEncoding:   hipTinyOutputWeightEncodingFP16,
		outputWeightByte: 12,
	}, {
		name:             "q8-output",
		outputType:       24,
		outputTypeName:   "q8:0.5",
		outputPayload:    hipInt8Payload(hipTinyOutputWeightsQ8Fixture()),
		outputEncoding:   hipTinyOutputWeightEncodingQ8,
		outputScale:      0.5,
		outputWeightByte: 6,
	}, {
		name:             "jangtq-output",
		outputType:       999,
		outputTypeName:   "jangtq:bits=2:group=2:scale=1",
		outputPayload:    []byte{0x41, 0x05},
		outputEncoding:   hipTinyOutputWeightEncodingFP32,
		outputWeightByte: 24,
		wantJANGTQ:       true,
	}, {
		name:             "codebook-output",
		outputType:       1000,
		outputTypeName:   "codebook:vq:dim=1",
		outputPayload:    []byte{1, 0, 0, 1, 1, 1},
		codebookValues:   []float32{0, 1},
		outputEncoding:   hipTinyOutputWeightEncodingFP32,
		outputWeightByte: 24,
		wantCodebook:     true,
	}} {
		t.Run(tt.name, func(t *testing.T) {
			embeddingPayload, err := hipFloat32Payload(fixture.EmbeddingTable)
			core.RequireNoError(t, err)
			outputPayload := tt.outputPayload
			if len(outputPayload) == 0 {
				switch tt.outputType {
				case 0:
					outputPayload, err = hipFloat32Payload(fixture.OutputWeights)
				case 1:
					outputPayload, err = hipUint16Payload(hipTinyOutputWeightsFP16Fixture())
				case 24:
					outputPayload = hipInt8Payload(hipTinyOutputWeightsQ8Fixture())
				default:
					t.Fatalf("unsupported output type %d", tt.outputType)
				}
				core.RequireNoError(t, err)
			}
			codebookPayload := tt.codebookPayload
			if len(codebookPayload) == 0 && len(tt.codebookValues) > 0 {
				codebookPayload, err = hipFloat32Payload(tt.codebookValues)
				core.RequireNoError(t, err)
			}
			modelPath := core.PathJoin(t.TempDir(), "tiny.bin")
			payload := append(append([]byte(nil), embeddingPayload...), outputPayload...)
			payload = append(payload, codebookPayload...)
			write := core.WriteFile(modelPath, payload, 0o644)
			core.RequireTrue(t, write.OK)
			driver := &fakeHIPDriver{available: true}
			runtime := newHIPRuntime(driver)
			tensors := []nativeTensorInfo{{
				Name:       "tok_embeddings.weight",
				Type:       0,
				Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
				Offset:     0,
				ByteSize:   uint64(len(embeddingPayload)),
			}, {
				Name:       "output.weight",
				Type:       tt.outputType,
				TypeName:   tt.outputTypeName,
				Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
				Offset:     uint64(len(embeddingPayload)),
				ByteSize:   uint64(len(outputPayload)),
			}}
			if len(codebookPayload) > 0 {
				tensors = append(tensors, nativeTensorInfo{
					Name:       "output.codebook",
					Type:       0,
					Dimensions: []uint64{uint64(len(tt.codebookValues)), 1},
					Offset:     uint64(len(embeddingPayload) + len(outputPayload)),
					ByteSize:   uint64(len(codebookPayload)),
				})
			}
			model, err := runtime.LoadModel(modelPath, nativeLoadConfig{
				ModelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: fixture.VocabSize, HiddenSize: fixture.HiddenSize, QuantBits: 32},
				Tensors:   tensors,
			})
			core.RequireNoError(t, err)
			defer model.Close()
			loaded, ok := model.(*hipLoadedModel)
			core.RequireTrue(t, ok)
			status := loaded.KernelStatus()
			core.AssertEqual(t, hipKernelStatusLinked, status.Prefill)
			core.AssertEqual(t, hipKernelStatusLinked, status.Decode)
			core.AssertEqual(t, hipKernelStatusLinked, status.Projection)

			prefill, err := loaded.Prefill(context.Background(), hipPrefillRequest{TokenIDs: []int32{0, 1}})
			core.RequireNoError(t, err)
			core.AssertEqual(t, 2, prefill.PromptTokens)
			core.AssertEqual(t, hipKernelNameTinyPrefill, prefill.Labels["prefill_kernel_name"])
			if tt.wantJANGTQ {
				core.AssertEqual(t, hipKernelNameJANGTQ, prefill.Labels["output_projection_kernel_name"])
				core.AssertEqual(t, "2", prefill.Labels["output_jangtq_bits"])
				core.AssertEqual(t, "2", prefill.Labels["output_jangtq_group_size"])
			}
			if tt.wantCodebook {
				core.AssertEqual(t, hipKernelNameCodebook, prefill.Labels["output_lookup_kernel_name"])
				core.AssertEqual(t, hipKernelNameProjection, prefill.Labels["output_projection_kernel_name"])
				core.AssertEqual(t, "2", prefill.Labels["output_codebook_entries"])
				core.AssertEqual(t, "1", prefill.Labels["output_codebook_dim"])
			}
			assertFloat32SlicesNear(t, []float32{0.3302, 0.6698, 1}, prefill.Logits, 0.0001)
			keys, values, err := prefill.KV.Restore(0, 2)
			core.RequireNoError(t, err)
			assertFloat32SlicesNear(t, []float32{1, 0, 0, 1}, keys, 0.0001)
			assertFloat32SlicesNear(t, []float32{1, 0, 0, 1}, values, 0.0001)
			core.AssertNotNil(t, prefill.DeviceKV)
			core.AssertNotNil(t, prefill.DescriptorTable)

			decoded, err := loaded.DecodeToken(context.Background(), hipDecodeRequest{
				TokenID:         2,
				KV:              prefill.KV,
				DeviceKV:        prefill.DeviceKV,
				DescriptorTable: prefill.DescriptorTable,
			})
			core.RequireNoError(t, err)
			defer decoded.DeviceKV.Close()
			defer decoded.DescriptorTable.Close()
			core.AssertEqual(t, int32(2), decoded.Token.ID)
			core.AssertEqual(t, hipKernelNameTinyDecode, decoded.Labels["decode_kernel_name"])
			if tt.wantJANGTQ {
				core.AssertEqual(t, hipKernelNameJANGTQ, decoded.Labels["output_projection_kernel_name"])
			}
			if tt.wantCodebook {
				core.AssertEqual(t, hipKernelNameCodebook, decoded.Labels["output_lookup_kernel_name"])
				core.AssertEqual(t, hipKernelNameProjection, decoded.Labels["output_projection_kernel_name"])
				core.AssertEqual(t, "2", decoded.Labels["output_codebook_entries"])
				core.AssertEqual(t, "1", decoded.Labels["output_codebook_dim"])
			}
			assertFloat32SlicesNear(t, []float32{0.7517, 0.7517, 1.5035}, decoded.Logits, 0.0001)
			core.AssertEqual(t, 3, decoded.KV.TokenCount())
			core.AssertEqual(t, "append_token", decoded.Labels["kv_device_update"])
			core.AssertEqual(t, "1", decoded.Labels["kv_device_update_pages"])
			core.AssertEqual(t, "1", decoded.Labels["kv_device_update_from_pages"])
			core.AssertEqual(t, "2", decoded.Labels["kv_device_update_from_tokens"])
			core.AssertEqual(t, "2", decoded.Labels["kv_device_update_to_pages"])
			core.AssertEqual(t, "3", decoded.Labels["kv_device_update_to_tokens"])
			core.AssertEqual(t, "success", decoded.Labels["kv_device_update_descriptor_refresh"])
			if !prefill.DeviceKV.closed || !prefill.DescriptorTable.closed {
				t.Fatalf("prefill device resources should be closed after successful tiny decode")
			}

			stream, streamErr := loaded.Generate(context.Background(), "hello", inference.GenerateConfig{MaxTokens: 2})
			var generated []int32
			for token := range stream {
				generated = append(generated, token.ID)
			}
			core.RequireNoError(t, streamErr())
			core.AssertEqual(t, []int32{1, 1}, generated)

			classified, err := loaded.Classify(context.Background(), []string{"hello"}, inference.GenerateConfig{ReturnLogits: true})
			core.RequireNoError(t, err)
			core.AssertEqual(t, int32(1), classified[0].Token.ID)
			assertFloat32SlicesNear(t, []float32{0, 1, 1}, classified[0].Logits, 0.0001)
			classifiedNoLogits, err := loaded.Classify(context.Background(), []string{"hello"}, inference.DefaultGenerateConfig())
			core.RequireNoError(t, err)
			core.AssertEqual(t, 0, len(classifiedNoLogits[0].Logits))

			launchNames := make([]string, len(driver.launches))
			for index, launch := range driver.launches {
				launchNames[index] = launch.Name
			}
			core.AssertContains(t, core.Join(",", launchNames...), hipKernelNameTinyPrefill)
			core.AssertContains(t, core.Join(",", launchNames...), hipKernelNameTinyDecode)
			if tt.wantJANGTQ {
				core.AssertContains(t, core.Join(",", launchNames...), hipKernelNameJANGTQ)
			}
			if tt.wantCodebook {
				core.AssertContains(t, core.Join(",", launchNames...), hipKernelNameCodebook)
				core.AssertContains(t, core.Join(",", launchNames...), hipKernelNameProjection)
			}
			var checkedPrefillLaunch bool
			for _, launch := range driver.launches {
				if launch.Name != hipKernelNameTinyPrefill || len(launch.Args) != hipTinyPrefillLaunchArgsBytes {
					continue
				}
				core.AssertEqual(t, tt.outputWeightByte, binary.LittleEndian.Uint32(launch.Args[92:]))
				core.AssertEqual(t, tt.outputEncoding, binary.LittleEndian.Uint32(launch.Args[116:]))
				core.AssertEqual(t, math.Float32bits(tt.outputScale), binary.LittleEndian.Uint32(launch.Args[120:]))
				checkedPrefillLaunch = true
				break
			}
			core.AssertTrue(t, checkedPrefillLaunch)
		})
	}
}

func TestHIPRuntime_LoadedTinyTextPathsPreflightRequests_Bad(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-tiny-preflight.hsaco")
	loaded, _ := loadHIPTinyF32FixtureModel(t, &fakeHIPDriver{available: true})

	stream, streamErr := loaded.Chat(context.Background(), nil, inference.DefaultGenerateConfig())
	for range stream {
		t.Fatal("Chat(nil) yielded token, want empty stream")
	}
	core.AssertError(t, streamErr())
	core.AssertContains(t, streamErr().Error(), "messages are required")

	stream, streamErr = loaded.Chat(context.Background(), []inference.Message{{Role: "moderator", Content: "hello"}}, inference.DefaultGenerateConfig())
	for range stream {
		t.Fatal("Chat(invalid role) yielded token, want empty stream")
	}
	core.AssertError(t, streamErr())
	core.AssertContains(t, streamErr().Error(), "message 0 role")

	_, err := loaded.Classify(context.Background(), nil, inference.DefaultGenerateConfig())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompts are required")

	_, err = loaded.Classify(context.Background(), []string{"hello", ""}, inference.DefaultGenerateConfig())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompt 1 is empty")

	_, err = loaded.BatchGenerate(context.Background(), nil, inference.DefaultGenerateConfig())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompts are required")

	_, err = loaded.BatchGenerate(context.Background(), []string{"hello", " "}, inference.DefaultGenerateConfig())
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompt 1 is empty")

	results, err := loaded.BatchGenerate(context.Background(), []string{"hello"}, inference.GenerateConfig{MaxTokens: 1})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(results))
	core.AssertEqual(t, 1, len(results[0].Tokens))
}

func TestHIPRuntime_LoadedTinyRequestValidation_Bad(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-tiny-request-validation.hsaco")
	loaded, _ := loadHIPTinyF32FixtureModel(t, &fakeHIPDriver{available: true})

	_, err := loaded.Prefill(context.Background(), hipPrefillRequest{TokenIDs: []int32{99}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token ID is outside vocabulary")

	_, err = loaded.Prefill(context.Background(), hipPrefillRequest{TokenIDs: []int32{0}, KeyWidth: 1, ValueWidth: 1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV widths to match hidden size")

	cache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{1, 0, 0, 1}))

	_, err = loaded.DecodeToken(context.Background(), hipDecodeRequest{TokenID: 99, KV: cache})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token ID is outside vocabulary")
}

func TestHIPRuntime_LoadedTinyTextPathsPreferCancelledContext_Ugly(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-tiny-cancel.hsaco")
	loaded, _ := loadHIPTinyF32FixtureModel(t, &fakeHIPDriver{available: true})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	stream, streamErr := loaded.Generate(ctx, "hello", inference.GenerateConfig{MaxTokens: 0})
	for range stream {
		t.Fatal("Generate(cancelled) yielded token, want empty stream")
	}
	if !errors.Is(streamErr(), context.Canceled) {
		t.Fatalf("Generate error = %v, want context.Canceled", streamErr())
	}

	stream, streamErr = loaded.Chat(ctx, nil, inference.DefaultGenerateConfig())
	for range stream {
		t.Fatal("Chat(cancelled) yielded token, want empty stream")
	}
	if !errors.Is(streamErr(), context.Canceled) {
		t.Fatalf("Chat error = %v, want context.Canceled", streamErr())
	}

	_, err := loaded.Classify(ctx, nil, inference.DefaultGenerateConfig())
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Classify error = %v, want context.Canceled", err)
	}

	_, err = loaded.BatchGenerate(ctx, nil, inference.DefaultGenerateConfig())
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("BatchGenerate error = %v, want context.Canceled", err)
	}
}

func TestHIPRuntime_LoadedTinyLMConfigShapeValidation_Bad(t *testing.T) {
	baseModel := func() *hipLoadedModel {
		return &hipLoadedModel{
			driver:    &fakeHIPDriver{available: true},
			modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 3, HiddenSize: 2, QuantBits: 32},
			tensors: map[string]hipTensor{
				"tok_embeddings.weight": {
					info: nativeTensorInfo{
						Name:       "tok_embeddings.weight",
						Type:       0,
						Dimensions: []uint64{3, 2},
						ByteSize:   24,
					},
					pointer: 1,
				},
				"output.weight": {
					info: nativeTensorInfo{
						Name:       "output.weight",
						Type:       0,
						Dimensions: []uint64{3, 2},
						ByteSize:   24,
					},
					pointer: 2,
				},
			},
		}
	}

	for _, tt := range []struct {
		name   string
		mutate func(*hipLoadedModel)
		want   string
	}{{
		name: "embedding-rank",
		mutate: func(model *hipLoadedModel) {
			tensor := model.tensors["tok_embeddings.weight"]
			tensor.info.Dimensions = []uint64{6}
			model.tensors["tok_embeddings.weight"] = tensor
		},
		want: "embedding shape",
	}, {
		name: "output-dimension-mismatch",
		mutate: func(model *hipLoadedModel) {
			tensor := model.tensors["output.weight"]
			tensor.info.Dimensions = []uint64{3, 3}
			tensor.info.ByteSize = 36
			model.tensors["output.weight"] = tensor
		},
		want: "output shape",
	}, {
		name: "output-shape-mismatch",
		mutate: func(model *hipLoadedModel) {
			model.modelInfo = inference.ModelInfo{Architecture: "tiny", QuantBits: 32}
			tensor := model.tensors["output.weight"]
			tensor.info.Dimensions = []uint64{4, 2}
			tensor.info.ByteSize = 32
			model.tensors["output.weight"] = tensor
		},
		want: "embedding and output tensor shapes must match",
	}, {
		name: "embedding-byte-count",
		mutate: func(model *hipLoadedModel) {
			tensor := model.tensors["tok_embeddings.weight"]
			tensor.info.ByteSize = 20
			model.tensors["tok_embeddings.weight"] = tensor
		},
		want: "embedding byte count",
	}, {
		name: "output-byte-count",
		mutate: func(model *hipLoadedModel) {
			tensor := model.tensors["output.weight"]
			tensor.info.ByteSize = 20
			model.tensors["output.weight"] = tensor
		},
		want: "output byte count",
	}, {
		name: "zero-pointer",
		mutate: func(model *hipLoadedModel) {
			tensor := model.tensors["output.weight"]
			tensor.pointer = 0
			model.tensors["output.weight"] = tensor
		},
		want: "embedding and output tensor pointers",
	}} {
		t.Run(tt.name, func(t *testing.T) {
			model := baseModel()
			tt.mutate(model)
			_, err := model.loadedTinyLMConfig()
			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}
}

func TestHIPRuntime_LoadModelRunsTinyEmbedAndRerankWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-embedding.hsaco")
	fixture := hipReferenceTinyLMFixture()
	embeddingPayload, err := hipFloat32Payload(fixture.EmbeddingTable)
	core.RequireNoError(t, err)
	outputPayload, err := hipFloat32Payload(fixture.OutputWeights)
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "tiny-embedding.bin")
	payload := append(append([]byte(nil), embeddingPayload...), outputPayload...)
	write := core.WriteFile(modelPath, payload, 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: fixture.VocabSize, HiddenSize: fixture.HiddenSize, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "tok_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "output.weight",
			Type:       0,
			Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(outputPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)

	status := loaded.KernelStatus()
	core.AssertEqual(t, hipKernelStatusLinked, status.Embedding)
	core.AssertEqual(t, hipKernelStatusLinked, status.Rerank)
	embedded, err := loaded.Embed(context.Background(), inference.EmbeddingRequest{
		Input:     []string{"hello", "hello world"},
		Normalize: true,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(embedded.Vectors))
	core.AssertEqual(t, hipKernelNameEmbedMean, embedded.Labels["embedding_kernel_name"])
	assertFloat32SlicesNear(t, []float32{0, 1}, embedded.Vectors[0], 0.0001)
	assertFloat32SlicesNear(t, []float32{0.4472136, 0.8944272}, embedded.Vectors[1], 0.0001)
	core.AssertEqual(t, 3, embedded.Usage.PromptTokens)

	reranked, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(reranked.Results))
	core.AssertEqual(t, 1, reranked.Results[0].Index)
	core.AssertEqual(t, "hello", reranked.Results[0].Text)
	core.AssertEqual(t, hipKernelNameRerank, reranked.Labels["rerank_kernel_name"])

	var sawEmbedding, sawRerank bool
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameEmbedMean {
			sawEmbedding = true
		}
		if launch.Name == hipKernelNameRerank {
			sawRerank = true
		}
	}
	core.AssertTrue(t, sawEmbedding)
	core.AssertTrue(t, sawRerank)
}

func TestHIPRuntime_LoadModelRunsBERTEmbedAndRerankWithoutOutputHeadWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-bert-embedding.hsaco")
	embeddingTable := []float32{
		1, 0,
		0, 1,
		1, 1,
	}
	embeddingPayload, err := hipFloat32Payload(embeddingTable)
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "bert-embedding.bin")
	write := core.WriteFile(modelPath, embeddingPayload, 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 3, HiddenSize: 2, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "embeddings.word_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{3, 2},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)

	status := loaded.KernelStatus()
	core.AssertEqual(t, hipKernelStatusLinked, status.Embedding)
	core.AssertEqual(t, hipKernelStatusLinked, status.Rerank)
	core.AssertEqual(t, hipKernelStatusNotLinked, status.Prefill)
	core.AssertEqual(t, hipKernelStatusNotLinked, status.Decode)
	embedded, err := loaded.Embed(context.Background(), inference.EmbeddingRequest{
		Input:     []string{"hello", "hello world"},
		Normalize: true,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(embedded.Vectors))
	core.AssertEqual(t, "bert", embedded.Labels["embedding_model_family"])
	core.AssertEqual(t, "experimental_loaded_f32_table", embedded.Labels["embedding_model_status"])
	assertFloat32SlicesNear(t, []float32{0, 1}, embedded.Vectors[0], 0.0001)
	assertFloat32SlicesNear(t, []float32{0.4472136, 0.8944272}, embedded.Vectors[1], 0.0001)

	reranked, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(reranked.Results))
	core.AssertEqual(t, 1, reranked.Results[0].Index)
	core.AssertEqual(t, "experimental_embedding_cosine", reranked.Labels["rerank_model_status"])

	stream, streamErr := loaded.Generate(context.Background(), "hello", inference.GenerateConfig{MaxTokens: 1})
	for range stream {
	}
	core.AssertError(t, streamErr())
	core.AssertContains(t, streamErr().Error(), "native decode kernels are not linked yet")
}

func TestHIPRuntime_LoadModelRunsBERTSequenceClassifierRerankWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-bert-sequence-classifier.hsaco")
	embeddingTable := []float32{
		0, 0,
		0, 1,
		0, 0,
		0, 1,
		1, 0,
	}
	classifierWeights := []float32{
		0, 0,
		1, 0,
	}
	classifierBias := []float32{0, 0}
	embeddingPayload, err := hipFloat32Payload(embeddingTable)
	core.RequireNoError(t, err)
	classifierPayload, err := hipFloat32Payload(classifierWeights)
	core.RequireNoError(t, err)
	biasPayload, err := hipFloat32Payload(classifierBias)
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "bert-sequence-classifier.bin")
	payload := append(append(append([]byte(nil), embeddingPayload...), classifierPayload...), biasPayload...)
	write := core.WriteFile(modelPath, payload, 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 5, HiddenSize: 2, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "embeddings.word_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{5, 2},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "classifier.weight",
			Type:       0,
			Dimensions: []uint64{2, 2},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(classifierPayload)),
		}, {
			Name:       "classifier.bias",
			Type:       0,
			Dimensions: []uint64{2},
			Offset:     uint64(len(embeddingPayload) + len(classifierPayload)),
			ByteSize:   uint64(len(biasPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)

	wrapper := &rocmModel{
		modelType: "bert",
		modelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 5, HiddenSize: 2, QuantBits: 32},
		native:    loaded,
	}
	report := wrapper.Capabilities()
	classifyCapability, ok := report.Capability(inference.CapabilityClassify)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, inference.CapabilityStatusExperimental, classifyCapability.Status)
	core.AssertEqual(t, "bert_sequence_classifier", classifyCapability.Labels["classify_path"])
	evalCapability, ok := report.Capability(inference.CapabilityEvaluation)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, "bert_sequence_classifier", evalCapability.Labels["classify_path"])

	noTargetEval, err := wrapper.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{Text: "hello world"}}, inference.EvalConfig{MaxSamples: 1})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "not_requested", noTargetEval.Labels["loss_status"])
	core.AssertEqual(t, "not_requested", noTargetEval.Labels["perplexity_status"])
	core.AssertEqual(t, "bert_sequence_classifier", noTargetEval.Labels["classify_path"])
	core.AssertEqual(t, hipKernelStatusLinked, noTargetEval.Labels["loss_kernel"])
	core.AssertEqual(t, hipKernelNameCrossEntropy, noTargetEval.Labels["loss_kernel_name"])

	var evalEvents []inference.ProbeEvent
	wrapper.SetProbeSink(inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		evalEvents = append(evalEvents, event)
	}))
	lossEval, err := wrapper.Evaluate(context.Background(), &singleInferenceSample{sample: inference.DatasetSample{
		Prompt: "hello world again now",
		Labels: map[string]string{"target_token_id": "1"},
	}}, inference.EvalConfig{MaxSamples: 1})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "experimental", lossEval.Labels["loss_status"])
	core.AssertEqual(t, "experimental", lossEval.Labels["perplexity_status"])
	core.AssertEqual(t, "1", lossEval.Labels["eval.loss_tokens"])
	core.AssertEqual(t, "bert_sequence_classifier", lossEval.Labels["classify_path"])
	core.AssertEqual(t, "hip", lossEval.Labels["loss_backend"])
	core.AssertEqual(t, hipKernelStatusLinked, lossEval.Labels["loss_kernel"])
	core.AssertEqual(t, hipKernelNameCrossEntropy, lossEval.Labels["loss_kernel_name"])
	logitEvent, ok := nativeContractProbeEvent(evalEvents, inference.ProbeEventLogits)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, "classification", logitEvent.Labels["source"])
	core.AssertEqual(t, "0", logitEvent.Labels["classify_prompt_index"])
	entropyEvent, ok := nativeContractProbeEvent(evalEvents, inference.ProbeEventEntropy)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, "classification", entropyEvent.Labels["source"])

	classified, err := loaded.Classify(context.Background(), []string{"hello world again now"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(1), classified[0].Token.ID)
	core.AssertEqual(t, "label_1", classified[0].Token.Text)
	assertFloat32SlicesNear(t, []float32{0, 0.25}, classified[0].Logits, 0.0001)
	classifiedNoLogits, err := loaded.Classify(context.Background(), []string{"hello world again now"}, inference.DefaultGenerateConfig())
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, len(classifiedNoLogits[0].Logits))

	reranked, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(reranked.Results))
	core.AssertEqual(t, 0, reranked.Results[0].Index)
	core.AssertEqual(t, "hello world", reranked.Results[0].Text)
	core.AssertEqual(t, "experimental_bert_sequence_classifier", reranked.Labels["rerank_model_status"])
	core.AssertEqual(t, "classifier_positive_logit", reranked.Labels["rerank_score_source"])
	core.AssertEqual(t, hipKernelNameProjection, reranked.Labels["projection_kernel_name"])
	core.AssertEqual(t, "1", reranked.Results[0].Labels["rerank_classifier_index"])

	var sawEmbedding, sawProjection, sawCosine bool
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameEmbedMean {
			sawEmbedding = true
		}
		if launch.Name == hipKernelNameProjection {
			sawProjection = true
			core.AssertEqual(t, hipProjectionWeightEncodingF32, binary.LittleEndian.Uint32(launch.Args[80:]))
		}
		if launch.Name == hipKernelNameRerank {
			sawCosine = true
		}
	}
	core.AssertTrue(t, sawEmbedding)
	core.AssertTrue(t, sawProjection)
	core.AssertFalse(t, sawCosine)
}

func TestHIPRuntime_LoadModelRunsBERTSequenceClassifierRerankWithF16Head_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-bert-sequence-classifier-f16.hsaco")
	embeddingPayload, err := hipFloat32Payload([]float32{
		0, 0,
		0, 1,
		0, 0,
		0, 1,
		1, 0,
	})
	core.RequireNoError(t, err)
	classifierPayload, err := hipUint16Payload([]uint16{0, 0, 0x3c00, 0})
	core.RequireNoError(t, err)
	biasPayload, err := hipUint16Payload([]uint16{0, 0})
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "bert-sequence-classifier-f16.bin")
	write := core.WriteFile(modelPath, append(append(append([]byte(nil), embeddingPayload...), classifierPayload...), biasPayload...), 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 5, HiddenSize: 2, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "embeddings.word_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{5, 2},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "classifier.weight",
			Type:       1,
			Dimensions: []uint64{2, 2},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(classifierPayload)),
		}, {
			Name:       "classifier.bias",
			Type:       1,
			Dimensions: []uint64{2},
			Offset:     uint64(len(embeddingPayload) + len(classifierPayload)),
			ByteSize:   uint64(len(biasPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)

	classified, err := loaded.Classify(context.Background(), []string{"hello world again now"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(1), classified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 0.25}, classified[0].Logits, 0.0001)

	reranked, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, reranked.Results[0].Index)
	core.AssertEqual(t, "fp16", reranked.Labels["rerank_classifier_encoding"])
	core.AssertEqual(t, "fp16", reranked.Labels["rerank_classifier_bias_encoding"])
	var sawProjection bool
	for _, launch := range driver.launches {
		if launch.Name != hipKernelNameProjection {
			continue
		}
		sawProjection = true
		core.AssertEqual(t, hipProjectionWeightEncodingFP16, binary.LittleEndian.Uint32(launch.Args[80:]))
		core.AssertEqual(t, hipProjectionLaunchFlagBias, binary.LittleEndian.Uint32(launch.Args[84:]))
	}
	core.AssertTrue(t, sawProjection)
}

func TestHIPRuntime_LoadModelRunsBERTSequenceClassifierLoRAAdapterWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-bert-classifier-lora.hsaco")
	embeddingPayload, err := hipFloat32Payload([]float32{
		0, 0,
		0, 1,
		0, 0,
		0, 1,
		1, 0,
	})
	core.RequireNoError(t, err)
	classifierPayload, err := hipFloat32Payload([]float32{
		0, 0,
		0, 1,
	})
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "bert-sequence-classifier-lora.bin")
	write := core.WriteFile(modelPath, append(append([]byte(nil), embeddingPayload...), classifierPayload...), 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 5, HiddenSize: 2, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "embeddings.word_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{5, 2},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "classifier.weight",
			Type:       0,
			Dimensions: []uint64{2, 2},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(classifierPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, hipKernelStatusLinked, loaded.KernelStatus().LoRA)

	base, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, base.Results[0].Index)

	baseClassified, err := loaded.Classify(context.Background(), []string{"hello world again now"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(1), baseClassified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 0.5}, baseClassified[0].Logits, 0.0001)

	adapterPath := core.PathJoin(t.TempDir(), "classifier_lora.json")
	write = core.WriteFile(adapterPath, []byte(`{
		"format":"rocm-classifier-lora",
		"name":"bert-rerank-domain",
		"target":"classifier.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"num_labels":2,
		"lora_a":[1,0],
		"lora_b":[0,4]
	}`), 0o644)
	core.RequireTrue(t, write.OK)
	identity, err := loaded.LoadAdapter(adapterPath)
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmClassifierLoRAFormat, identity.Format)
	core.AssertEqual(t, "hip_bert_classifier", identity.Labels["adapter_runtime"])
	core.AssertEqual(t, adapterPath, loaded.ActiveAdapter().Path)

	classified, err := loaded.Classify(context.Background(), []string{"hello world again now"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(1), classified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 1.5}, classified[0].Logits, 0.0001)

	reranked, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, reranked.Results[0].Index)
	core.AssertEqual(t, hipKernelNameLoRA, reranked.Labels["projection_kernel_name"])
	core.AssertEqual(t, hipKernelNameLoRA, reranked.Labels["lora_kernel_name"])
	core.AssertEqual(t, "hip_bert_classifier", reranked.Labels["adapter_runtime"])

	var sawLoRA, sawCosine bool
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameLoRA {
			sawLoRA = true
		}
		if launch.Name == hipKernelNameRerank {
			sawCosine = true
		}
	}
	core.AssertTrue(t, sawLoRA)
	core.AssertFalse(t, sawCosine)
	core.AssertNoError(t, loaded.UnloadAdapter())
	if !adapterIdentityIsZero(loaded.ActiveAdapter()) {
		t.Fatalf("active adapter = %+v, want zero after unload", loaded.ActiveAdapter())
	}
}

func TestHIPRuntime_LoadModelRunsBERTScoreTensorLoRAAdapterWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-bert-score-lora.hsaco")
	embeddingPayload, err := hipFloat32Payload([]float32{
		0, 0,
		0, 1,
		0, 0,
		0, 1,
		1, 0,
	})
	core.RequireNoError(t, err)
	scorePayload, err := hipFloat32Payload([]float32{
		0, 0,
		0, 1,
	})
	core.RequireNoError(t, err)
	biasPayload, err := hipFloat32Payload([]float32{0, 0})
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "bert-score-lora.bin")
	write := core.WriteFile(modelPath, append(append(append([]byte(nil), embeddingPayload...), scorePayload...), biasPayload...), 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 5, HiddenSize: 2, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "embeddings.word_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{5, 2},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "score.weight",
			Type:       0,
			Dimensions: []uint64{2, 2},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(scorePayload)),
		}, {
			Name:       "score.bias",
			Type:       0,
			Dimensions: []uint64{2},
			Offset:     uint64(len(embeddingPayload) + len(scorePayload)),
			ByteSize:   uint64(len(biasPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)

	base, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, base.Results[0].Index)
	core.AssertEqual(t, "score.weight", base.Labels["rerank_classifier_tensor"])
	core.AssertEqual(t, "score.bias", base.Labels["rerank_classifier_bias"])

	baseClassified, err := loaded.Classify(context.Background(), []string{"hello world again now"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(1), baseClassified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 0.5}, baseClassified[0].Logits, 0.0001)

	adapterPath := core.PathJoin(t.TempDir(), "score_lora.json")
	write = core.WriteFile(adapterPath, []byte(`{
		"format":"rocm-classifier-lora",
		"name":"bert-score-domain",
		"target":"score.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"num_labels":2,
		"lora_a":[1,0],
		"lora_b":[0,4]
	}`), 0o644)
	core.RequireTrue(t, write.OK)
	identity, err := loaded.LoadAdapter(adapterPath)
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmClassifierLoRAFormat, identity.Format)
	core.AssertEqual(t, "score.weight", identity.TargetKeys[0])
	core.AssertEqual(t, "score.weight", identity.Labels["target"])
	core.AssertEqual(t, "score.weight", identity.Labels["classifier_tensor"])

	classified, err := loaded.Classify(context.Background(), []string{"hello world again now"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(1), classified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 1.5}, classified[0].Logits, 0.0001)

	reranked, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, reranked.Results[0].Index)
	core.AssertEqual(t, hipKernelNameLoRA, reranked.Labels["projection_kernel_name"])
	core.AssertEqual(t, hipKernelNameLoRA, reranked.Labels["lora_kernel_name"])
	core.AssertEqual(t, "score.weight", reranked.Labels["rerank_classifier_tensor"])
	core.AssertEqual(t, "score.bias", reranked.Labels["rerank_classifier_bias"])

	var sawLoRA bool
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameLoRA {
			sawLoRA = true
		}
	}
	core.AssertTrue(t, sawLoRA)
}

func TestHIPRuntime_LoadedSequenceClassifierConfigPairsCanonicalHeadBias_Good(t *testing.T) {
	model := &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "bert", HiddenSize: 2},
		tensors: map[string]hipTensor{
			"score.bias": {
				info:    nativeTensorInfo{Name: "score.bias", Type: 0, Dimensions: []uint64{2}, ByteSize: 8},
				pointer: 11,
			},
			"score.weight": {
				info:    nativeTensorInfo{Name: "score.weight", Type: 0, Dimensions: []uint64{2, 2}, ByteSize: 16},
				pointer: 12,
			},
			"classifier.bias": {
				info:    nativeTensorInfo{Name: "classifier.bias", Type: 0, Dimensions: []uint64{2}, ByteSize: 8},
				pointer: 13,
			},
			"classifier.weight": {
				info:    nativeTensorInfo{Name: "classifier.weight", Type: 0, Dimensions: []uint64{2, 2}, ByteSize: 16},
				pointer: 14,
			},
		},
	}

	cfg, hasClassifier, err := model.loadedSequenceClassifierConfig()

	core.RequireNoError(t, err)
	core.RequireTrue(t, hasClassifier)
	core.AssertEqual(t, "classifier.weight", cfg.WeightTensor)
	core.AssertEqual(t, "classifier.bias", cfg.BiasTensor)
	core.AssertEqual(t, nativeDevicePointer(14), cfg.WeightPointer)
	core.AssertEqual(t, nativeDevicePointer(13), cfg.BiasPointer)
}

func TestHIPRuntime_LoadedSequenceClassifierConfigDoesNotPairForeignBias_Good(t *testing.T) {
	model := &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "bert", HiddenSize: 2},
		tensors: map[string]hipTensor{
			"score.weight": {
				info:    nativeTensorInfo{Name: "score.weight", Type: 0, Dimensions: []uint64{2, 2}, ByteSize: 16},
				pointer: 21,
			},
			"classifier.bias": {
				info:    nativeTensorInfo{Name: "classifier.bias", Type: 0, Dimensions: []uint64{2}, ByteSize: 8},
				pointer: 22,
			},
		},
	}

	cfg, hasClassifier, err := model.loadedSequenceClassifierConfig()

	core.RequireNoError(t, err)
	core.RequireTrue(t, hasClassifier)
	core.AssertEqual(t, "score.weight", cfg.WeightTensor)
	core.AssertEqual(t, "", cfg.BiasTensor)
	core.AssertEqual(t, nativeDevicePointer(21), cfg.WeightPointer)
	core.AssertEqual(t, nativeDevicePointer(0), cfg.BiasPointer)
}

func TestHIPRuntime_LoadModelBERTSequenceClassifierLoRAAdapterRejectsBadShape_Bad(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-bert-classifier-lora.hsaco")
	embeddingPayload, err := hipFloat32Payload([]float32{
		0, 0,
		0, 1,
		0, 0,
	})
	core.RequireNoError(t, err)
	classifierPayload, err := hipFloat32Payload([]float32{0, 0, 0, 1})
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "bert-bad-classifier-lora.bin")
	write := core.WriteFile(modelPath, append(append([]byte(nil), embeddingPayload...), classifierPayload...), 0o644)
	core.RequireTrue(t, write.OK)
	model, err := newHIPRuntime(&fakeHIPDriver{available: true}).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 3, HiddenSize: 2, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "embeddings.word_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{3, 2},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "classifier.weight",
			Type:       0,
			Dimensions: []uint64{2, 2},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(classifierPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	adapterPath := core.PathJoin(t.TempDir(), "bad_classifier_lora.json")
	write = core.WriteFile(adapterPath, []byte(`{
		"format":"rocm-classifier-lora",
		"target":"classifier.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"num_labels":2,
		"lora_a":[1,0],
		"lora_b":[4]
	}`), 0o644)
	core.RequireTrue(t, write.OK)

	_, err = loaded.LoadAdapter(adapterPath)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "LoRA B length")
	if !adapterIdentityIsZero(loaded.ActiveAdapter()) {
		t.Fatalf("active adapter = %+v, want zero after failed load", loaded.ActiveAdapter())
	}
}

func TestHIPRuntime_LoadModelBERTSequenceClassifierRerankRejectsBadHead_Bad(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-bert-sequence-classifier.hsaco")
	embeddingPayload, err := hipFloat32Payload([]float32{
		0, 0,
		0, 1,
		0, 0,
		0, 1,
	})
	core.RequireNoError(t, err)
	classifierPayload, err := hipFloat32Payload([]float32{1, 0, 0, 1, 1, 1})
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "bad-bert-sequence-classifier.bin")
	write := core.WriteFile(modelPath, append(append([]byte(nil), embeddingPayload...), classifierPayload...), 0o644)
	core.RequireTrue(t, write.OK)
	model, err := newHIPRuntime(&fakeHIPDriver{available: true}).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 4, HiddenSize: 2, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "embeddings.word_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{4, 2},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "classifier.weight",
			Type:       0,
			Dimensions: []uint64{2, 3},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(classifierPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)

	_, err = loaded.Rerank(context.Background(), inference.RerankRequest{Query: "hello", Documents: []string{"hello"}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "classifier hidden size")
}

func TestHIPRuntime_LoadModelRunsTinyLoRAAdapterWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-lora.hsaco")
	driver := &fakeHIPDriver{available: true}
	loaded, _ := loadHIPTinyF32FixtureModel(t, driver)

	status := loaded.KernelStatus()
	core.AssertEqual(t, hipKernelStatusLinked, status.LoRA)

	adapterDir := t.TempDir()
	adapterPath := core.PathJoin(adapterDir, "rocm_tiny_lora.json")
	writeTinyLoRAAdapterFile(t, adapterPath, `{
		"format":"rocm-tiny-lora",
		"name":"boost-two",
		"target":"output.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"vocab_size":3,
		"lora_a":[0,1],
		"lora_b":[0,0,2]
	}`)

	identity, err := loaded.LoadAdapter(adapterDir)
	core.RequireNoError(t, err)
	core.AssertEqual(t, adapterDir, identity.Path)
	core.AssertEqual(t, rocmTinyLoRAFormat, identity.Format)
	core.AssertEqual(t, 1, identity.Rank)
	core.AssertEqual(t, float32(1), identity.Alpha)
	core.AssertNotEmpty(t, identity.Hash)
	core.AssertEqual(t, hipKernelStatusLinked, identity.Labels["lora_kernel"])
	core.AssertEqual(t, hipKernelNameLoRA, identity.Labels["lora_kernel_name"])
	core.AssertEqual(t, adapterPath, identity.Labels["adapter_file"])
	core.AssertEqual(t, identity.Hash, loaded.ActiveAdapter().Hash)
	identity.TargetKeys[0] = "mutated"
	identity.Labels["lora_kernel"] = "mutated"
	active := loaded.ActiveAdapter()
	core.AssertEqual(t, "output.weight", active.TargetKeys[0])
	core.AssertEqual(t, hipKernelStatusLinked, active.Labels["lora_kernel"])

	classified, err := loaded.Classify(context.Background(), []string{"hello"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(2), classified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 1, 3}, classified[0].Logits, 0.0001)

	prefill, err := loaded.Prefill(context.Background(), hipPrefillRequest{TokenIDs: []int32{1}})
	core.RequireNoError(t, err)
	defer prefill.DeviceKV.Close()
	defer prefill.DescriptorTable.Close()
	core.AssertEqual(t, identity.Hash, prefill.Labels["adapter_hash"])
	core.AssertEqual(t, hipKernelStatusLinked, prefill.Labels["lora_kernel"])
	core.AssertEqual(t, hipKernelNameLoRA, prefill.Labels["lora_kernel_name"])
	assertFloat32SlicesNear(t, []float32{0, 1, 3}, prefill.Logits, 0.0001)

	stream, streamErr := loaded.Generate(context.Background(), "hello", inference.GenerateConfig{MaxTokens: 1})
	var generated []int32
	for token := range stream {
		generated = append(generated, token.ID)
	}
	core.RequireNoError(t, streamErr())
	core.AssertEqual(t, []int32{2}, generated)

	var sawLoRA bool
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameLoRA {
			sawLoRA = true
		}
	}
	core.AssertTrue(t, sawLoRA)

	core.RequireNoError(t, loaded.UnloadAdapter())
	if !adapterIdentityIsZero(loaded.ActiveAdapter()) {
		t.Fatalf("active adapter after unload = %+v, want zero", loaded.ActiveAdapter())
	}
	classified, err = loaded.Classify(context.Background(), []string{"hello"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(1), classified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 1, 1}, classified[0].Logits, 0.0001)
}

func TestHIPRuntime_LoadModelRunsTinyLoRAAdapterWithCodebookOutputWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-codebook-lora.hsaco")
	fixture := hipReferenceTinyLMFixture()
	embeddingPayload, err := hipFloat32Payload(fixture.EmbeddingTable)
	core.RequireNoError(t, err)
	codePayload := []byte{1, 0, 0, 1, 1, 1}
	codebookPayload, err := hipFloat32Payload([]float32{0, 1})
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "tiny-codebook-lora.bin")
	payload := append(append(append([]byte(nil), embeddingPayload...), codePayload...), codebookPayload...)
	write := core.WriteFile(modelPath, payload, 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: fixture.VocabSize, HiddenSize: fixture.HiddenSize, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "tok_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "output.weight",
			Type:       1000,
			TypeName:   "codebook:vq:dim=1",
			Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(codePayload)),
		}, {
			Name:       "output.codebook",
			Type:       0,
			Dimensions: []uint64{2, 1},
			Offset:     uint64(len(embeddingPayload) + len(codePayload)),
			ByteSize:   uint64(len(codebookPayload)),
		}},
	})
	core.RequireNoError(t, err)
	defer model.Close()
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)

	adapterPath := core.PathJoin(t.TempDir(), "rocm_tiny_lora.json")
	writeTinyLoRAAdapterFile(t, adapterPath, `{
		"format":"rocm-tiny-lora",
		"target":"output.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"vocab_size":3,
		"lora_a":[0,1],
		"lora_b":[0,0,2]
	}`)
	identity, err := loaded.LoadAdapter(adapterPath)
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmTinyLoRAFormat, identity.Format)

	classified, err := loaded.Classify(context.Background(), []string{"hello"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(2), classified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 1, 3}, classified[0].Logits, 0.0001)

	prefill, err := loaded.Prefill(context.Background(), hipPrefillRequest{TokenIDs: []int32{1}})
	core.RequireNoError(t, err)
	defer prefill.DeviceKV.Close()
	defer prefill.DescriptorTable.Close()
	core.AssertEqual(t, hipKernelNameCodebook, prefill.Labels["output_lookup_kernel_name"])
	core.AssertEqual(t, hipKernelNameLoRA, prefill.Labels["lora_kernel_name"])
	assertFloat32SlicesNear(t, []float32{0, 1, 3}, prefill.Logits, 0.0001)

	var sawCodebook, sawLoRA bool
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameCodebook {
			sawCodebook = true
		}
		if launch.Name == hipKernelNameLoRA {
			sawLoRA = true
		}
	}
	core.AssertTrue(t, sawCodebook)
	core.AssertTrue(t, sawLoRA)
}

func TestHIPRuntime_LoadTinyLoRAAdapterBadValidationKeepsActiveAdapter_Bad(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-lora.hsaco")
	loaded, _ := loadHIPTinyF32FixtureModel(t, &fakeHIPDriver{available: true})
	validPath := core.PathJoin(t.TempDir(), "valid-lora.json")
	writeTinyLoRAAdapterFile(t, validPath, `{
		"format":"rocm-tiny-lora",
		"target":"output.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"vocab_size":3,
		"lora_a":[0,1],
		"lora_b":[0,0,2]
	}`)
	previous, err := loaded.LoadAdapter(validPath)
	core.RequireNoError(t, err)

	invalidPath := core.PathJoin(t.TempDir(), "invalid-lora.json")
	writeTinyLoRAAdapterFile(t, invalidPath, `{
		"format":"rocm-tiny-lora",
		"target":"output.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"vocab_size":3,
		"lora_a":[0,1],
		"lora_b":[0,2]
	}`)

	identity, err := loaded.LoadAdapter(invalidPath)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "adapter LoRA B length must match vocab*rank")
	if !adapterIdentityIsZero(identity) {
		t.Fatalf("identity = %+v, want zero", identity)
	}
	active := loaded.ActiveAdapter()
	core.AssertEqual(t, previous.Path, active.Path)
	core.AssertEqual(t, previous.Hash, active.Hash)

	classified, err := loaded.Classify(context.Background(), []string{"hello"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(2), classified[0].Token.ID)
	assertFloat32SlicesNear(t, []float32{0, 1, 3}, classified[0].Logits, 0.0001)
}

func TestHIPRuntime_LoadedTinyEmbedAndRerankNotLinked_Bad(t *testing.T) {
	loaded := &hipLoadedModel{kernels: newDefaultHIPKernelSet()}

	_, err := loaded.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"hello"}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native embedding kernels are not linked yet")
	_, err = loaded.Rerank(context.Background(), inference.RerankRequest{Query: "hello", Documents: []string{"doc"}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native rerank kernels are not linked yet")
}

func TestHIPRuntime_LoadedTinyEmbedAndRerankPreflightBeforeNotLinked_Bad(t *testing.T) {
	loaded := &hipLoadedModel{kernels: newDefaultHIPKernelSet()}

	_, err := loaded.Embed(context.Background(), inference.EmbeddingRequest{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input text is required")

	_, err = loaded.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"ok", "   "}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input 1 is empty")

	_, err = loaded.Rerank(context.Background(), inference.RerankRequest{Documents: []string{"doc"}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "query is required")

	_, err = loaded.Rerank(context.Background(), inference.RerankRequest{Query: "hello"})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "documents are required")

	_, err = loaded.Rerank(context.Background(), inference.RerankRequest{Query: "hello", Documents: []string{"doc", ""}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "document 1 is empty")
}

func TestHIPRuntime_LoadedTinyQ8ScaleValidation_Bad(t *testing.T) {
	for _, tt := range []struct {
		name     string
		typeName string
		want     string
	}{{
		name:     "empty-scale",
		typeName: "q8:",
		want:     "parse q8 output scale",
	}, {
		name:     "zero-scale",
		typeName: "q8:0",
		want:     "q8 output scale must be positive and finite",
	}, {
		name:     "negative-scale",
		typeName: "q8:-0.5",
		want:     "q8 output scale must be positive and finite",
	}, {
		name:     "nan-scale",
		typeName: "q8:NaN",
		want:     "q8 output scale must be positive and finite",
	}, {
		name:     "inf-scale",
		typeName: "q8:+Inf",
		want:     "q8 output scale must be positive and finite",
	}} {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, _, err := hipTinyLoadedOutputEncoding(nativeTensorInfo{Type: 24, TypeName: tt.typeName})
			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}
}

func TestHIPRuntime_LoadedTinyJANGTQOutputValidation_Bad(t *testing.T) {
	for _, tt := range []struct {
		name     string
		typeName string
		want     string
	}{{
		name:     "bad-bits",
		typeName: "jangtq:bits=3:group=2:scale=1",
		want:     "unsupported bit layout",
	}, {
		name:     "bad-group",
		typeName: "jangtq:bits=2:group=3:scale=1",
		want:     "group size must be a positive power of two",
	}, {
		name:     "bad-scale",
		typeName: "mxtq:bits=2:group=2:scale=0",
		want:     "JANGTQ scale must be positive and finite",
	}} {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, _, err := hipTinyLoadedOutputEncoding(nativeTensorInfo{Type: 999, TypeName: tt.typeName})
			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}

	model := &hipLoadedModel{
		driver:    &fakeHIPDriver{available: true},
		modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: 3, HiddenSize: 2, QuantBits: 2},
		tensors: map[string]hipTensor{
			"tok_embeddings.weight": {
				info:    nativeTensorInfo{Name: "tok_embeddings.weight", Type: 0, Dimensions: []uint64{3, 2}, ByteSize: 24},
				pointer: 1,
			},
			"output.weight": {
				info:    nativeTensorInfo{Name: "output.weight", Type: 999, TypeName: "jangtq:bits=2:group=2:scale=1", Dimensions: []uint64{3, 2}, ByteSize: 1},
				pointer: 2,
			},
		},
	}
	_, err := model.loadedTinyLMConfig()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "output byte count")
}

func TestHIPRuntime_LoadedTinyCodebookOutputValidation_Bad(t *testing.T) {
	_, _, _, _, err := hipTinyLoadedOutputEncoding(nativeTensorInfo{Type: 1000, TypeName: "codebook:vq:dim=0"})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "codebook dimension must be positive")

	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-tiny.hsaco")
	fixture := hipReferenceTinyLMFixture()
	embeddingPayload, err := hipFloat32Payload(fixture.EmbeddingTable)
	core.RequireNoError(t, err)
	codePayload := []byte{1, 0, 0, 1, 1, 1}
	codebookPayload, err := hipFloat32Payload([]float32{0, 1})
	core.RequireNoError(t, err)
	codebookFP16Payload := []byte{0, 0, 0, 0}

	for _, tt := range []struct {
		name           string
		outputTypeName string
		outputByteSize uint64
		tensors        []nativeTensorInfo
		payload        []byte
		want           string
	}{{
		name:           "missing-table",
		outputTypeName: "codebook:vq:dim=1",
		tensors:        nil,
		payload:        append(append([]byte(nil), embeddingPayload...), codePayload...),
		want:           "codebook output table tensor is required",
	}, {
		name:           "vector-codes",
		outputTypeName: "codebook:vq:dim=2",
		tensors:        nil,
		payload:        append(append([]byte(nil), embeddingPayload...), codePayload...),
		want:           "codebook output code dimension must be 1",
	}, {
		name:           "output-code-byte-count",
		outputTypeName: "codebook:vq:dim=1",
		outputByteSize: uint64(len(codePayload) - 1),
		tensors:        nil,
		payload:        append(append([]byte(nil), embeddingPayload...), codePayload...),
		want:           "output byte count",
	}, {
		name:           "table-not-f32",
		outputTypeName: "codebook:vq:dim=1",
		tensors: []nativeTensorInfo{{
			Name:       "output.codebook",
			Type:       1,
			TypeName:   "f16",
			Dimensions: []uint64{2, 1},
			Offset:     uint64(len(embeddingPayload) + len(codePayload)),
			ByteSize:   uint64(len(codebookFP16Payload)),
		}},
		payload: append(append(append([]byte(nil), embeddingPayload...), codePayload...), codebookFP16Payload...),
		want:    "codebook output table must be f32",
	}, {
		name:           "table-rank",
		outputTypeName: "codebook:vq:dim=1",
		tensors: []nativeTensorInfo{{
			Name:       "output.codebook",
			Type:       0,
			Dimensions: []uint64{2},
			Offset:     uint64(len(embeddingPayload) + len(codePayload)),
			ByteSize:   uint64(len(codebookPayload)),
		}},
		payload: append(append(append([]byte(nil), embeddingPayload...), codePayload...), codebookPayload...),
		want:    "codebook output table tensor must be rank 2",
	}, {
		name:           "table-dimension-mismatch",
		outputTypeName: "codebook:vq:dim=1",
		tensors: []nativeTensorInfo{{
			Name:       "output.codebook",
			Type:       0,
			Dimensions: []uint64{1, 2},
			Offset:     uint64(len(embeddingPayload) + len(codePayload)),
			ByteSize:   uint64(len(codebookPayload)),
		}},
		payload: append(append(append([]byte(nil), embeddingPayload...), codePayload...), codebookPayload...),
		want:    "codebook output table dimension mismatch",
	}} {
		t.Run(tt.name, func(t *testing.T) {
			modelPath := core.PathJoin(t.TempDir(), "tiny-codebook-bad.bin")
			write := core.WriteFile(modelPath, tt.payload, 0o644)
			core.RequireTrue(t, write.OK)
			driver := &fakeHIPDriver{available: true}
			runtime := newHIPRuntime(driver)
			tensors := []nativeTensorInfo{{
				Name:       "tok_embeddings.weight",
				Type:       0,
				Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
				Offset:     0,
				ByteSize:   uint64(len(embeddingPayload)),
			}, {
				Name:       "output.weight",
				Type:       1000,
				TypeName:   tt.outputTypeName,
				Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
				Offset:     uint64(len(embeddingPayload)),
				ByteSize:   uint64(len(codePayload)),
			}}
			if tt.outputByteSize > 0 {
				tensors[1].ByteSize = tt.outputByteSize
			}
			tensors = append(tensors, tt.tensors...)
			model, err := runtime.LoadModel(modelPath, nativeLoadConfig{
				ModelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: fixture.VocabSize, HiddenSize: fixture.HiddenSize, QuantBits: 32},
				Tensors:   tensors,
			})
			core.RequireNoError(t, err)
			defer model.Close()
			loaded, ok := model.(*hipLoadedModel)
			core.RequireTrue(t, ok)

			_, err = loaded.loadedTinyLMConfig()
			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}

	for _, tt := range []struct {
		name     string
		byteSize uint64
		pointer  nativeDevicePointer
		want     string
	}{{
		name:     "table-byte-count",
		byteSize: 4,
		pointer:  3,
		want:     "codebook table byte count",
	}, {
		name:     "table-pointer",
		byteSize: uint64(len(codebookPayload)),
		pointer:  0,
		want:     "codebook output table tensor pointer",
	}} {
		t.Run(tt.name, func(t *testing.T) {
			model := &hipLoadedModel{
				driver:    &fakeHIPDriver{available: true},
				modelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: fixture.VocabSize, HiddenSize: fixture.HiddenSize, QuantBits: 8},
				tensors: map[string]hipTensor{
					"tok_embeddings.weight": {
						info:    nativeTensorInfo{Name: "tok_embeddings.weight", Type: 0, Dimensions: []uint64{3, 2}, ByteSize: uint64(len(embeddingPayload))},
						pointer: 1,
					},
					"output.weight": {
						info:    nativeTensorInfo{Name: "output.weight", Type: 1000, TypeName: "codebook:vq:dim=1", Dimensions: []uint64{3, 2}, ByteSize: uint64(len(codePayload))},
						pointer: 2,
					},
					"output.codebook": {
						info:    nativeTensorInfo{Name: "output.codebook", Type: 0, Dimensions: []uint64{2, 1}, ByteSize: tt.byteSize},
						pointer: tt.pointer,
					},
				},
			}
			_, err = model.loadedTinyLMConfig()
			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}
}

func TestHIPRuntime_LoadModelBadFreeOnCopyFailure_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	driver := &fakeHIPDriver{available: true, copyErr: core.NewError("copy failed")}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3"},
		DataOffset: dataOffset,
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", ByteSize: 16},
			{Name: "output.weight", Offset: 16, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertNil(t, model)
	core.AssertEqual(t, 1, len(driver.frees))
}

func TestHIPRuntime_LoadModelBadFreesAllTensorsOnSecondCopyFailure_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true, copyErr: core.NewError("copy failed"), copyErrAt: 2}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3"},
		DataOffset: dataOffset,
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", ByteSize: 16},
			{Name: "output.weight", Offset: 16, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertNil(t, model)
	core.AssertEqual(t, []uint64{16, 16}, driver.allocations)
	core.AssertEqual(t, []uint64{16, 16}, driver.copies)
	core.AssertEqual(t, 2, len(driver.frees))
}

func TestHIPRuntime_LoadModelBadShortTensorReadRejectedBeforeAllocation_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3"},
		DataOffset: dataOffset,
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", ByteSize: 16},
			{Name: "output.weight", Offset: 24, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tensor byte range exceeds file size")
	core.AssertNil(t, model)
	core.AssertEqual(t, 0, len(driver.allocations))
	core.AssertEqual(t, 0, len(driver.copies))
	core.AssertEqual(t, 0, len(driver.frees))
}

func TestHIPRuntime_LoadModelUglyEmptyTensorMap_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	driver := &fakeHIPDriver{available: true}
	runtime := newHIPRuntime(driver)
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := runtime.LoadModel(path, nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3"},
		DataOffset: dataOffset,
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "missing token embedding tensor")
	core.AssertNil(t, model)
	core.AssertEqual(t, 0, len(driver.allocations))
}

func TestHIPRuntime_Validate_BadMissingOutputHead(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors:   []nativeTensorInfo{{Name: "tok_embeddings.weight", Type: 0, ByteSize: 16}},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "missing output head tensor")
}

func TestHIPRuntime_Validate_BadRequiredTensorHasZeroBytes(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, ByteSize: 0},
			{Name: "output.weight", Type: 0, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "zero byte size")
}

func TestHIPRuntime_Validate_BadMismatchedLayerCount(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3", NumLayers: 2},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, ByteSize: 16},
			{Name: "model.layers.0.attn.weight", Type: 0, ByteSize: 16},
			{Name: "output.weight", Type: 0, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "mismatched layer count")
}

func TestHIPRuntime_Validate_GoodKnownNumericQuantizedDTypeWithoutName(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 15, ByteSize: 16},
			{Name: "output.weight", Type: 15, ByteSize: 16},
		},
	})

	core.AssertNoError(t, err)
}

func TestHIPRuntime_Validate_GoodKnownGGUFQuantizedTypeName(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 15, TypeName: "Q8_K", ByteSize: 16},
			{Name: "output.weight", Type: 15, TypeName: "Q8_K", ByteSize: 16},
		},
	})

	core.AssertNoError(t, err)
}

func TestHIPRuntime_Validate_GoodGGUFTokenEmbeddingAlias(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "gemma3"},
		Tensors: []nativeTensorInfo{
			{Name: "token_embd.weight", Type: 15, TypeName: "Q4_K", ByteSize: 16},
			{Name: "output.weight", Type: 15, TypeName: "Q4_K", ByteSize: 16},
		},
	})

	core.AssertNoError(t, err)
}

func TestHIPRuntime_Validate_GoodGemma4TiedSafetensorsEmbedding(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo:          inference.ModelInfo{Architecture: "gemma4", VocabSize: 8, HiddenSize: 16, NumLayers: 1, QuantBits: 4, QuantGroup: 64},
		TiedWordEmbeddings: true,
		Tensors: []nativeTensorInfo{
			{Name: "language_model.model.embed_tokens.weight", Dimensions: []uint64{8, 2}, Type: 26, TypeName: "U32", ByteSize: 64},
			{Name: "language_model.model.embed_tokens.biases", Dimensions: []uint64{8, 2}, Type: 30, TypeName: "BF16", ByteSize: 32},
			{Name: "language_model.model.layers.0.input_layernorm.weight", Dimensions: []uint64{16}, Type: 30, TypeName: "BF16", ByteSize: 32},
		},
	})

	core.AssertNoError(t, err)
}

func TestHIPRuntime_LoadModelCopiesShardedSafetensorsSources_Good(t *testing.T) {
	dir := t.TempDir()
	shardA := core.PathJoin(dir, "model-00001-of-00002.safetensors")
	shardB := core.PathJoin(dir, "model-00002-of-00002.safetensors")
	writeNativeContractFile(t, shardA, string(make([]byte, 8+64)))
	writeNativeContractFile(t, shardB, string(make([]byte, 8+32)))
	driver := &fakeHIPDriver{available: true}
	runtime := newHIPRuntime(driver)

	model, err := runtime.LoadModel(dir, nativeLoadConfig{
		ModelInfo:          inference.ModelInfo{Architecture: "gemma4", VocabSize: 8, HiddenSize: 16, NumLayers: 1, QuantBits: 4, QuantGroup: 64},
		TiedWordEmbeddings: true,
		Tensors: []nativeTensorInfo{
			{Name: "language_model.model.embed_tokens.weight", SourcePath: shardA, DataOffset: 8, Dimensions: []uint64{8, 2}, Type: 26, TypeName: "U32", ByteSize: 64},
			{Name: "language_model.model.layers.0.input_layernorm.weight", SourcePath: shardB, DataOffset: 8, Dimensions: []uint64{16}, Type: 30, TypeName: "BF16", ByteSize: 32},
		},
	})

	core.AssertNoError(t, err)
	core.AssertNotNil(t, model)
	core.AssertEqual(t, []uint64{64, 32}, driver.allocations)
	core.AssertEqual(t, []uint64{64, 32}, driver.copies)
	core.AssertNoError(t, model.Close())
	core.AssertEqual(t, 2, len(driver.frees))
}

func TestHIPRuntime_Validate_BadUnsupportedDType(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 999, TypeName: "q9", ByteSize: 16},
			{Name: "output.weight", Type: 0, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported tensor dtype")
}

func TestHIPRuntime_Validate_BadEmptyTensorName(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "", Type: 0, ByteSize: 16},
			{Name: "output.weight", Type: 0, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tensor name is required")
}

func TestHIPRuntime_Validate_BadDuplicateTensorName(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, ByteSize: 16},
			{Name: "TOK_EMBEDDINGS.WEIGHT", Type: 0, ByteSize: 16},
			{Name: "output.weight", Type: 0, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "duplicate tensor name")
}

func TestHIPRuntime_Validate_BadUnsupportedQuantization(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3", QuantBits: 12},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, ByteSize: 16},
			{Name: "output.weight", Type: 0, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported quantization")
}

func TestHIPRuntime_Validate_BadNegativeDataOffset(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3"},
		DataOffset: -1,
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, ByteSize: 16},
			{Name: "output.weight", Type: 0, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "data offset")
}

func TestHIPRuntime_Validate_BadTensorDataOffsetOverflow(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3"},
		DataOffset: 1,
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, ByteSize: 16},
			{Name: "output.weight", Type: 0, Offset: 1 << 63, ByteSize: 16},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "offset overflows")
}

func TestHIPRuntime_Validate_BadTensorFileRangeOverflow(t *testing.T) {
	_, err := hipTensorFileEnd(1, 1<<63)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "overflows")
}

func TestHIPRuntime_Validate_GoodProjectionShapes(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3", VocabSize: 4, HiddenSize: 2},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, Dimensions: []uint64{2, 4}, ByteSize: 32},
			{Name: "output.weight", Type: 1, Dimensions: []uint64{4, 2}, ByteSize: 16},
		},
	})

	core.AssertNoError(t, err)
}

func TestHIPRuntime_Validate_BadProjectionRank(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3", VocabSize: 4, HiddenSize: 2},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, Dimensions: []uint64{8}, ByteSize: 32},
			{Name: "output.weight", Type: 0, Dimensions: []uint64{2, 4}, ByteSize: 32},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "projection tensor must be rank 2")
}

func TestHIPRuntime_Validate_BadProjectionIdentityMismatch(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3", VocabSize: 32000, HiddenSize: 4096},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 1, Dimensions: []uint64{128, 4096}, ByteSize: 1048576},
			{Name: "output.weight", Type: 1, Dimensions: []uint64{4096, 128}, ByteSize: 1048576},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "missing vocab size")
}

func TestHIPRuntime_Validate_BadByteSizeMismatch(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, Dimensions: []uint64{2, 4}, ByteSize: 16},
			{Name: "output.weight", Type: 0, Dimensions: []uint64{2, 4}, ByteSize: 32},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tensor byte size mismatch")
}

func TestHIPRuntime_Validate_UglyZeroDimension(t *testing.T) {
	err := validateHIPLoadConfig(nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "qwen3"},
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, Dimensions: []uint64{0, 4}, ByteSize: 0},
			{Name: "output.weight", Type: 0, Dimensions: []uint64{2, 4}, ByteSize: 32},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "zero dimension")
}

func TestHIPRuntime_DecodeKernelsNotLinked_Bad(t *testing.T) {
	model := &hipLoadedModel{}

	stream, streamErr := model.Generate(context.Background(), "hello", inference.DefaultGenerateConfig())
	for range stream {
	}
	err := streamErr()

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native decode kernels are not linked yet")
}

func TestHIPRuntime_CloseGoodIdempotentClearsRuntimeState(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	model := &hipLoadedModel{
		driver:  driver,
		tensors: map[string]hipTensor{"tok_embeddings.weight": {info: nativeTensorInfo{ByteSize: 16}, pointer: 7}},
		adapter: inference.AdapterIdentity{Path: "domain.safetensors", Format: "lora"},
	}

	core.AssertNoError(t, model.Close())
	core.AssertNoError(t, model.Close())

	core.AssertEqual(t, 1, len(driver.frees))
	if !adapterIdentityIsZero(model.ActiveAdapter()) {
		t.Fatalf("active adapter = %+v, want zero after close", model.ActiveAdapter())
	}
	core.AssertEqual(t, uint64(0), model.Metrics().ActiveMemoryBytes)
}

func TestHIPRuntime_LoadAdapterBadEmptyPath_Bad(t *testing.T) {
	model := &hipLoadedModel{}

	identity, err := model.LoadAdapter(" \t")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "adapter path is required")
	if !adapterIdentityIsZero(identity) {
		t.Fatalf("identity = %+v, want zero", identity)
	}
	if !adapterIdentityIsZero(model.ActiveAdapter()) {
		t.Fatalf("active adapter = %+v, want zero", model.ActiveAdapter())
	}
}

func TestHIPRuntime_LoadAdapterBadNotLinkedKeepsActiveAdapter_Bad(t *testing.T) {
	model := &hipLoadedModel{adapter: inference.AdapterIdentity{Path: "previous.safetensors", Format: "lora"}}

	identity, err := model.LoadAdapter("domain.safetensors")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native LoRA adapter application is not linked yet")
	core.AssertContains(t, err.Error(), "domain.safetensors")
	if !adapterIdentityIsZero(identity) {
		t.Fatalf("identity = %+v, want zero", identity)
	}
	if got := model.ActiveAdapter(); got.Path != "previous.safetensors" || got.Format != "lora" {
		t.Fatalf("active adapter = %+v, want previous adapter", got)
	}
}

func loadHIPTinyF32FixtureModel(t *testing.T, driver *fakeHIPDriver) (*hipLoadedModel, hipReferenceTinyLMConfig) {
	t.Helper()
	fixture := hipReferenceTinyLMFixture()
	embeddingPayload, err := hipFloat32Payload(fixture.EmbeddingTable)
	core.RequireNoError(t, err)
	outputPayload, err := hipFloat32Payload(fixture.OutputWeights)
	core.RequireNoError(t, err)
	modelPath := core.PathJoin(t.TempDir(), "tiny.bin")
	payload := append(append([]byte(nil), embeddingPayload...), outputPayload...)
	write := core.WriteFile(modelPath, payload, 0o644)
	core.RequireTrue(t, write.OK)
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "tiny", VocabSize: fixture.VocabSize, HiddenSize: fixture.HiddenSize, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "tok_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "output.weight",
			Type:       0,
			Dimensions: []uint64{uint64(fixture.VocabSize), uint64(fixture.HiddenSize)},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(outputPayload)),
		}},
	})
	core.RequireNoError(t, err)
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	t.Cleanup(func() {
		core.AssertNoError(t, loaded.Close())
	})
	return loaded, fixture
}

func writeTinyLoRAAdapterFile(t *testing.T, path, payload string) {
	t.Helper()
	write := core.WriteFile(path, []byte(payload), 0o644)
	core.RequireTrue(t, write.OK)
}

type fakeHIPDriver struct {
	available                bool
	device                   nativeDeviceInfo
	nextPointer              nativeDevicePointer
	allocations              []uint64
	copies                   []uint64
	frees                    []nativeDevicePointer
	launches                 []hipKernelLaunchConfig
	memory                   map[nativeDevicePointer][]byte
	copyErr                  error
	copyErrAt                int
	copyHostErrAfterLaunches int
	launchErr                error
}

func (driver *fakeHIPDriver) Available() bool { return driver.available }
func (driver *fakeHIPDriver) DeviceInfo() nativeDeviceInfo {
	return driver.device
}
func (driver *fakeHIPDriver) Malloc(size uint64) (nativeDevicePointer, error) {
	driver.allocations = append(driver.allocations, size)
	if driver.nextPointer == 0 {
		driver.nextPointer = 0x1000
	}
	pointer := driver.nextPointer
	driver.nextPointer += nativeDevicePointer(size) + 0x1000
	if driver.memory == nil {
		driver.memory = map[nativeDevicePointer][]byte{}
	}
	driver.memory[pointer] = make([]byte, int(size))
	return pointer, nil
}
func (driver *fakeHIPDriver) Free(pointer nativeDevicePointer) error {
	driver.frees = append(driver.frees, pointer)
	delete(driver.memory, pointer)
	return nil
}
func (driver *fakeHIPDriver) CopyHostToDevice(pointer nativeDevicePointer, data []byte) error {
	driver.copies = append(driver.copies, uint64(len(data)))
	if driver.shouldFailCopy(true) {
		return driver.copyErr
	}
	if target, offset, ok := driver.memoryForPointer(pointer, len(data)); ok {
		copy(target[offset:], data)
	}
	return nil
}
func (driver *fakeHIPDriver) CopyDeviceToHost(pointer nativeDevicePointer, data []byte) error {
	driver.copies = append(driver.copies, uint64(len(data)))
	if driver.shouldFailCopy(false) {
		return driver.copyErr
	}
	if source, offset, ok := driver.memoryForPointer(pointer, len(data)); ok {
		copy(data, source[offset:offset+len(data)])
	}
	return nil
}
func (driver *fakeHIPDriver) shouldFailCopy(hostToDevice bool) bool {
	if driver.copyErr == nil {
		return false
	}
	if driver.copyErrAt > 0 && len(driver.copies) == driver.copyErrAt {
		return true
	}
	if hostToDevice && driver.copyHostErrAfterLaunches > 0 && len(driver.launches) >= driver.copyHostErrAfterLaunches {
		return true
	}
	return driver.copyErrAt == 0 && driver.copyHostErrAfterLaunches == 0
}
func (driver *fakeHIPDriver) LaunchKernel(config hipKernelLaunchConfig) error {
	copied := config
	copied.Args = append([]byte(nil), config.Args...)
	driver.launches = append(driver.launches, copied)
	if driver.launchErr != nil {
		return driver.launchErr
	}
	switch config.Name {
	case hipKernelNamePrefill:
		return driver.launchPrefill(config.Args)
	case hipKernelNameDecode:
		return driver.launchDecode(config.Args)
	case hipKernelNameProjection:
		return driver.launchProjection(config.Args)
	case hipKernelNameMLXQ4Proj:
		return driver.launchMLXQ4Projection(config.Args)
	case hipKernelNameRMSNorm:
		return driver.launchRMSNorm(config.Args)
	case hipKernelNameRoPE:
		return driver.launchRoPE(config.Args)
	case hipKernelNameGreedy:
		return driver.launchGreedySample(config.Args)
	case hipKernelNameAttention:
		return driver.launchAttention(config.Args)
	case hipKernelNameVectorAdd:
		return driver.launchVectorAdd(config.Args)
	case hipKernelNameVectorScale:
		return driver.launchVectorScale(config.Args)
	case hipKernelNameSwiGLU:
		return driver.launchSwiGLU(config.Args)
	case hipKernelNameMoERouter:
		return driver.launchMoERouter(config.Args)
	case hipKernelNameMoELazy:
		return driver.launchMoELazyExperts(config.Args)
	case hipKernelNameJANGTQ:
		return driver.launchJANGTQProjection(config.Args)
	case hipKernelNameCodebook:
		return driver.launchCodebookLookup(config.Args)
	case hipKernelNameLoRA:
		return driver.launchLoRAProjection(config.Args)
	case hipKernelNameEmbedLookup:
		return driver.launchEmbeddingLookup(config.Args)
	case hipKernelNameEmbedMean:
		return driver.launchEmbeddingMeanPool(config.Args)
	case hipKernelNameRerank:
		return driver.launchRerankCosine(config.Args)
	case hipKernelNameTinyPrefill:
		return driver.launchTinyPrefill(config.Args)
	case hipKernelNameTinyDecode:
		return driver.launchTinyDecode(config.Args)
	case hipKernelNameCrossEntropy:
		return driver.launchCrossEntropyLoss(config.Args)
	case hipKernelNameDistillKL:
		return driver.launchDistillationKLLoss(config.Args)
	case hipKernelNameGRPOAdvantage:
		return driver.launchGRPOAdvantage(config.Args)
	}
	return nil
}

func (driver *fakeHIPDriver) launchCrossEntropyLoss(args []byte) error {
	if len(args) != hipCrossEntropyLossLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "cross entropy launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipCrossEntropyLossLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipCrossEntropyLossLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "cross entropy launch header mismatch", nil)
	}
	logitPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	targetPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	batch := int(binary.LittleEndian.Uint32(args[32:]))
	vocab := int(binary.LittleEndian.Uint32(args[36:]))
	logitBytes := int(binary.LittleEndian.Uint32(args[40:]))
	targetBytes := int(binary.LittleEndian.Uint32(args[44:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[48:]))
	if batch <= 0 || vocab <= 0 || logitBytes != batch*vocab*4 || targetBytes != batch*4 || outputBytes != hipCrossEntropyLossOutputBytes {
		return core.E("rocm.hip.FakeLaunch", "cross entropy shape metadata mismatch", nil)
	}
	logitData, logitOffset, ok := driver.memoryForPointer(logitPointer, logitBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "cross entropy logits buffer is missing", nil)
	}
	targetData, targetOffset, ok := driver.memoryForPointer(targetPointer, targetBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "cross entropy target buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "cross entropy output buffer is missing", nil)
	}
	logits, err := hipFloat32PayloadValues(logitData[logitOffset : logitOffset+logitBytes])
	if err != nil {
		return err
	}
	targets := make([]int, batch)
	for index := range targets {
		targets[index] = int(int32(binary.LittleEndian.Uint32(targetData[targetOffset+index*4:])))
	}
	loss, perplexity, err := rocmReferenceCrossEntropyLoss(splitFloat32Vectors(logits, vocab), targets)
	if err != nil {
		return err
	}
	binary.LittleEndian.PutUint64(outputData[outputOffset:], math.Float64bits(loss))
	binary.LittleEndian.PutUint64(outputData[outputOffset+8:], math.Float64bits(perplexity))
	return nil
}

func (driver *fakeHIPDriver) launchDistillationKLLoss(args []byte) error {
	if len(args) != hipDistillationKLLossLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "distillation KL launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipDistillationKLLossLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipDistillationKLLossLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "distillation KL launch header mismatch", nil)
	}
	studentPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	teacherPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	batch := int(binary.LittleEndian.Uint32(args[32:]))
	vocab := int(binary.LittleEndian.Uint32(args[36:]))
	studentBytes := int(binary.LittleEndian.Uint32(args[40:]))
	teacherBytes := int(binary.LittleEndian.Uint32(args[44:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[48:]))
	temperature := math.Float64frombits(binary.LittleEndian.Uint64(args[56:]))
	if batch <= 0 || vocab <= 0 || studentBytes != batch*vocab*4 || teacherBytes != batch*vocab*4 ||
		outputBytes != hipDistillationKLLossOutputBytes || temperature <= 0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		return core.E("rocm.hip.FakeLaunch", "distillation KL shape metadata mismatch", nil)
	}
	studentData, studentOffset, ok := driver.memoryForPointer(studentPointer, studentBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "distillation student buffer is missing", nil)
	}
	teacherData, teacherOffset, ok := driver.memoryForPointer(teacherPointer, teacherBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "distillation teacher buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "distillation output buffer is missing", nil)
	}
	students, err := hipFloat32PayloadValues(studentData[studentOffset : studentOffset+studentBytes])
	if err != nil {
		return err
	}
	teachers, err := hipFloat32PayloadValues(teacherData[teacherOffset : teacherOffset+teacherBytes])
	if err != nil {
		return err
	}
	kl, err := rocmReferenceDistillationKL(splitFloat32Vectors(students, vocab), splitFloat32Vectors(teachers, vocab), temperature)
	if err != nil {
		return err
	}
	binary.LittleEndian.PutUint64(outputData[outputOffset:], math.Float64bits(kl))
	return nil
}

func (driver *fakeHIPDriver) launchGRPOAdvantage(args []byte) error {
	if len(args) != hipGRPOAdvantageLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "GRPO advantage launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipGRPOAdvantageLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipGRPOAdvantageLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "GRPO advantage launch header mismatch", nil)
	}
	rewardPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	count := int(binary.LittleEndian.Uint32(args[24:]))
	rewardBytes := int(binary.LittleEndian.Uint32(args[28:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[32:]))
	if count <= 0 || rewardBytes != count*8 || outputBytes != count*8 {
		return core.E("rocm.hip.FakeLaunch", "GRPO advantage shape metadata mismatch", nil)
	}
	rewardData, rewardOffset, ok := driver.memoryForPointer(rewardPointer, rewardBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "GRPO reward buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "GRPO advantage output buffer is missing", nil)
	}
	rewards, err := hipFloat64PayloadValues(rewardData[rewardOffset : rewardOffset+rewardBytes])
	if err != nil {
		return err
	}
	advantages, err := rocmReferenceNormalizeAdvantages(rewards)
	if err != nil {
		return err
	}
	payload, err := hipFloat64Payload(advantages)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchMoERouter(args []byte) error {
	if len(args) != hipMoERouterLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MoE router launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMoERouterLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMoERouterLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MoE router launch header mismatch", nil)
	}
	logitPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	idPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	probPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	expertCount := int(binary.LittleEndian.Uint32(args[32:]))
	topK := int(binary.LittleEndian.Uint32(args[36:]))
	logitBytes := int(binary.LittleEndian.Uint32(args[40:]))
	idBytes := int(binary.LittleEndian.Uint32(args[44:]))
	probBytes := int(binary.LittleEndian.Uint32(args[48:]))
	layer := int(binary.LittleEndian.Uint32(args[52:]))
	statusPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[56:]))
	if expertCount <= 0 || topK <= 0 || topK > expertCount || logitBytes != expertCount*4 || idBytes != topK*4 || probBytes != topK*4 {
		return core.E("rocm.hip.FakeLaunch", "MoE router shape metadata mismatch", nil)
	}
	logitData, logitOffset, ok := driver.memoryForPointer(logitPointer, logitBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MoE router logits buffer is missing", nil)
	}
	idData, idOffset, ok := driver.memoryForPointer(idPointer, idBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MoE router id output buffer is missing", nil)
	}
	probData, probOffset, ok := driver.memoryForPointer(probPointer, probBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MoE router probability output buffer is missing", nil)
	}
	logits, err := hipFloat32PayloadValues(logitData[logitOffset : logitOffset+logitBytes])
	if err != nil {
		return err
	}
	routes, err := rocmReferenceRouteExperts(logits, topK, layer, nil)
	if err != nil {
		return err
	}
	for index, route := range routes {
		binary.LittleEndian.PutUint32(idData[idOffset+index*4:], uint32(int32(route.ID)))
		binary.LittleEndian.PutUint32(probData[probOffset+index*4:], math.Float32bits(route.Prob))
	}
	if statusPointer != 0 {
		status, offset, ok := driver.memoryForPointer(statusPointer, 4)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "MoE router status buffer is missing", nil)
		}
		binary.LittleEndian.PutUint32(status[offset:], hipMoERouterLaunchStatusOK)
	}
	return nil
}

func (driver *fakeHIPDriver) launchMoELazyExperts(args []byte) error {
	if len(args) != hipMoELazyLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MoE lazy expert launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMoELazyLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMoELazyLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MoE lazy expert launch header mismatch", nil)
	}
	idPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	residentPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	selected := int(binary.LittleEndian.Uint32(args[24:]))
	total := int(binary.LittleEndian.Uint32(args[28:]))
	idBytes := int(binary.LittleEndian.Uint32(args[32:]))
	residentBytes := int(binary.LittleEndian.Uint32(args[36:]))
	if selected <= 0 || total <= 0 || idBytes != selected*4 || residentBytes != total {
		return core.E("rocm.hip.FakeLaunch", "MoE lazy expert shape metadata mismatch", nil)
	}
	idData, idOffset, ok := driver.memoryForPointer(idPointer, idBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MoE lazy expert ID buffer is missing", nil)
	}
	residentData, residentOffset, ok := driver.memoryForPointer(residentPointer, residentBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MoE lazy expert output buffer is missing", nil)
	}
	routes := make([]rocmExpertRoute, selected)
	for index := range routes {
		routes[index] = rocmExpertRoute{ID: int(int32(binary.LittleEndian.Uint32(idData[idOffset+index*4:])))}
	}
	resident, err := rocmReferenceLazyExpertResidency(routes, total)
	if err != nil {
		return err
	}
	for index, value := range resident {
		if value {
			residentData[residentOffset+index] = 1
		} else {
			residentData[residentOffset+index] = 0
		}
	}
	return nil
}
func (driver *fakeHIPDriver) memoryForPointer(pointer nativeDevicePointer, size int) ([]byte, int, bool) {
	if driver.memory == nil || pointer == 0 || size < 0 {
		return nil, 0, false
	}
	if data, ok := driver.memory[pointer]; ok && len(data) >= size {
		return data, 0, true
	}
	for base, data := range driver.memory {
		if pointer < base {
			continue
		}
		offset := int(pointer - base)
		if offset >= 0 && offset+size <= len(data) {
			return data, offset, true
		}
	}
	return nil, 0, false
}
func (driver *fakeHIPDriver) launchPrefill(args []byte) error {
	if len(args) != hipPrefillLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "prefill launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipPrefillLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipPrefillLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "prefill launch header mismatch", nil)
	}
	tokenPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	tokenCount := binary.LittleEndian.Uint64(args[16:])
	tokenBytes := binary.LittleEndian.Uint64(args[24:])
	modeCode := binary.LittleEndian.Uint32(args[32:])
	blockSize := binary.LittleEndian.Uint32(args[36:])
	keyWidth := binary.LittleEndian.Uint32(args[40:])
	valueWidth := binary.LittleEndian.Uint32(args[44:])
	statusPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	statusValue := binary.LittleEndian.Uint32(args[56:])
	if tokenCount == 0 || tokenBytes != tokenCount*4 {
		return core.E("rocm.hip.FakeLaunch", "prefill token metadata mismatch", nil)
	}
	if blockSize == 0 || keyWidth == 0 || valueWidth == 0 {
		return core.E("rocm.hip.FakeLaunch", "prefill KV shape metadata is invalid", nil)
	}
	if err := rocmDeviceKVValidateModeCode(modeCode); err != nil {
		return err
	}
	if _, _, ok := driver.memoryForPointer(tokenPointer, int(tokenBytes)); !ok {
		return core.E("rocm.hip.FakeLaunch", "prefill token buffer is missing", nil)
	}
	if statusPointer != 0 {
		if statusValue == 0 {
			statusValue = hipPrefillLaunchStatusOK
		}
		status, offset, ok := driver.memoryForPointer(statusPointer, 4)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "prefill status buffer is missing", nil)
		}
		binary.LittleEndian.PutUint32(status[offset:], statusValue)
	}
	return nil
}
func (driver *fakeHIPDriver) launchDecode(args []byte) error {
	if len(args) != hipDecodeLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "decode launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipDecodeLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipDecodeLaunchArgsHeaderBytes) ||
		binary.LittleEndian.Uint32(args[8:]) != uint32(hipDecodeLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "decode launch header mismatch", nil)
	}
	position := binary.LittleEndian.Uint64(args[16:])
	kvBytes := binary.LittleEndian.Uint32(args[24:])
	if kvBytes != rocmDeviceKVLaunchDescriptorBytes {
		return core.E("rocm.hip.FakeLaunch", "decode KV launch descriptor size mismatch", nil)
	}
	kv := args[hipDecodeLaunchArgsHeaderBytes:]
	descriptorPointer := nativeDevicePointer(binary.LittleEndian.Uint64(kv[0:]))
	descriptorBytes := binary.LittleEndian.Uint64(kv[8:])
	descriptorVersion := binary.LittleEndian.Uint32(kv[16:])
	modeCode := binary.LittleEndian.Uint32(kv[20:])
	pageCount := binary.LittleEndian.Uint32(kv[28:])
	tokenCount := binary.LittleEndian.Uint64(kv[32:])
	keyWidth := binary.LittleEndian.Uint32(kv[40:])
	valueWidth := binary.LittleEndian.Uint32(kv[44:])
	statusPointer := nativeDevicePointer(binary.LittleEndian.Uint64(kv[48:]))
	statusValue := binary.LittleEndian.Uint32(kv[56:])
	if descriptorVersion != rocmDeviceKVDescriptorVersion {
		return core.E("rocm.hip.FakeLaunch", "decode descriptor version mismatch", nil)
	}
	if err := rocmDeviceKVValidateModeCode(modeCode); err != nil {
		return err
	}
	if position != tokenCount || tokenCount == 0 || pageCount == 0 || keyWidth == 0 || valueWidth == 0 {
		return core.E("rocm.hip.FakeLaunch", "decode KV metadata mismatch", nil)
	}
	descriptor, offset, ok := driver.memoryForPointer(descriptorPointer, int(descriptorBytes))
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "decode descriptor table is missing", nil)
	}
	table := descriptor[offset : offset+int(descriptorBytes)]
	if len(table) < rocmDeviceKVDescriptorHeaderBytes ||
		binary.LittleEndian.Uint32(table[0:]) != rocmDeviceKVDescriptorVersion ||
		binary.LittleEndian.Uint32(table[12:]) != modeCode ||
		binary.LittleEndian.Uint32(table[16:]) != pageCount ||
		binary.LittleEndian.Uint64(table[24:]) != tokenCount {
		return core.E("rocm.hip.FakeLaunch", "decode descriptor table header mismatch", nil)
	}
	if statusPointer != 0 {
		if statusValue == 0 {
			statusValue = hipDecodeLaunchStatusOK
		}
		status, offset, ok := driver.memoryForPointer(statusPointer, 4)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "decode status buffer is missing", nil)
		}
		binary.LittleEndian.PutUint32(status[offset:], statusValue)
	}
	return nil
}
func (driver *fakeHIPDriver) launchProjection(args []byte) error {
	if len(args) != hipProjectionLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "projection launch args size mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	inputCount := int(binary.LittleEndian.Uint32(args[16:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[20:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	weightBytes := int(binary.LittleEndian.Uint64(args[32:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	biasBytes := int(binary.LittleEndian.Uint64(args[48:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[56:]))
	outputBytes := int(binary.LittleEndian.Uint64(args[64:]))
	rows := int(binary.LittleEndian.Uint32(args[72:]))
	cols := int(binary.LittleEndian.Uint32(args[76:]))
	encoding := binary.LittleEndian.Uint32(args[80:])
	flags := binary.LittleEndian.Uint32(args[84:])
	q8Scale := math.Float32frombits(binary.LittleEndian.Uint32(args[88:]))
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "projection input buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "projection weight buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "projection output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	var bias []float32
	if flags&hipProjectionLaunchFlagBias != 0 {
		biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "projection bias buffer is missing", nil)
		}
		bias, err = hipFloat32PayloadValues(biasData[biasOffset : biasOffset+biasBytes])
		if err != nil {
			return err
		}
	}
	var output []float32
	switch encoding {
	case hipProjectionWeightEncodingFP16:
		weights := make([]uint16, weightBytes/2)
		for index := range weights {
			weights[index] = binary.LittleEndian.Uint16(weightData[weightOffset+index*2:])
		}
		output, err = hipReferenceFP16Projection(input[:inputCount], weights, rows, cols, bias)
	case hipProjectionWeightEncodingBF16:
		weights := make([]uint16, weightBytes/2)
		for index := range weights {
			weights[index] = binary.LittleEndian.Uint16(weightData[weightOffset+index*2:])
		}
		output, err = hipReferenceBF16Projection(input[:inputCount], weights, rows, cols, bias)
	case hipProjectionWeightEncodingQ8:
		weights := make([]int8, weightBytes)
		for index := range weights {
			weights[index] = int8(weightData[weightOffset+index])
		}
		output, err = hipReferenceQ8Projection(input[:inputCount], weights, q8Scale, rows, cols, bias)
	case hipProjectionWeightEncodingF32:
		weights, decodeErr := hipFloat32PayloadValues(weightData[weightOffset : weightOffset+weightBytes])
		if decodeErr != nil {
			return decodeErr
		}
		output, err = hipReferenceF32Projection(input[:inputCount], weights, rows, cols, bias)
	default:
		err = core.E("rocm.hip.FakeLaunch", "unsupported projection encoding", nil)
	}
	if err != nil {
		return err
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchMLXQ4Projection(args []byte) error {
	if len(args) != hipMLXQ4ProjectionLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipMLXQ4ProjectionLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipMLXQ4ProjectionLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	scalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	rows := int(binary.LittleEndian.Uint32(args[48:]))
	cols := int(binary.LittleEndian.Uint32(args[52:]))
	groupSize := int(binary.LittleEndian.Uint32(args[56:]))
	bits := int(binary.LittleEndian.Uint32(args[60:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[64:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[68:]))
	scaleBytes := int(binary.LittleEndian.Uint32(args[72:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[76:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[80:]))
	if bits != hipMLXQ4ProjectionBits ||
		validateHIPMLXQ4ProjectionShape(cols, weightBytes/4, scaleBytes/2, biasBytes/2, rows, cols, groupSize) != nil ||
		inputBytes != cols*4 ||
		outputBytes != rows*4 {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection input buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection packed weight buffer is missing", nil)
	}
	scaleData, scaleOffset, ok := driver.memoryForPointer(scalePointer, scaleBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection scale buffer is missing", nil)
	}
	biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection bias buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "MLX q4 projection output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	weights := make([]uint32, weightBytes/4)
	for index := range weights {
		weights[index] = binary.LittleEndian.Uint32(weightData[weightOffset+index*4:])
	}
	scales := make([]uint16, scaleBytes/2)
	for index := range scales {
		scales[index] = binary.LittleEndian.Uint16(scaleData[scaleOffset+index*2:])
	}
	biases := make([]uint16, biasBytes/2)
	for index := range biases {
		biases[index] = binary.LittleEndian.Uint16(biasData[biasOffset+index*2:])
	}
	output, err := hipReferenceMLXQ4Projection(input, weights, scales, biases, rows, cols, groupSize)
	if err != nil {
		return err
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchJANGTQProjection(args []byte) error {
	if len(args) != hipJANGTQLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "JANGTQ launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipJANGTQLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipJANGTQLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "JANGTQ launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	packedPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	inputCount := int(binary.LittleEndian.Uint32(args[40:]))
	rows := int(binary.LittleEndian.Uint32(args[44:]))
	cols := int(binary.LittleEndian.Uint32(args[48:]))
	bits := int(binary.LittleEndian.Uint32(args[52:]))
	groupSize := int(binary.LittleEndian.Uint32(args[56:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[60:]))
	packedBytes := int(binary.LittleEndian.Uint32(args[64:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[68:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[72:]))
	scale := math.Float32frombits(binary.LittleEndian.Uint32(args[76:]))
	flags := binary.LittleEndian.Uint32(args[80:])
	if inputCount != cols || rows <= 0 || cols <= 0 || inputBytes != cols*4 || outputBytes != rows*4 || packedBytes < packedROCmJANGTQBytes(bits, rows*cols) {
		return core.E("rocm.hip.FakeLaunch", "JANGTQ shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "JANGTQ input buffer is missing", nil)
	}
	packedData, packedOffset, ok := driver.memoryForPointer(packedPointer, packedBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "JANGTQ packed weight buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "JANGTQ output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	var bias []float32
	if flags&hipJANGTQLaunchFlagBias != 0 {
		biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "JANGTQ bias buffer is missing", nil)
		}
		bias, err = hipFloat32PayloadValues(biasData[biasOffset : biasOffset+biasBytes])
		if err != nil {
			return err
		}
	}
	output, err := rocmReferenceJANGTQProjection(
		input[:inputCount],
		packedData[packedOffset:packedOffset+packedBytes],
		rocmJANGTQDescriptor{WeightFormat: "mxtq", Bits: bits, GroupSize: groupSize},
		rows,
		cols,
		scale,
		bias,
	)
	if err != nil {
		return err
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchCodebookLookup(args []byte) error {
	if len(args) != hipCodebookLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "codebook launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipCodebookLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipCodebookLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "codebook launch header mismatch", nil)
	}
	codePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	codebookPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	codeCount := int(binary.LittleEndian.Uint32(args[32:]))
	codebookCount := int(binary.LittleEndian.Uint32(args[36:]))
	codeDim := int(binary.LittleEndian.Uint32(args[40:]))
	codeBytes := int(binary.LittleEndian.Uint32(args[44:]))
	codebookBytes := int(binary.LittleEndian.Uint32(args[48:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[52:]))
	if codeCount <= 0 || codebookCount <= 0 || codeDim <= 0 || codeBytes != codeCount || codebookBytes != codebookCount*codeDim*4 || outputBytes != codeCount*codeDim*4 {
		return core.E("rocm.hip.FakeLaunch", "codebook shape metadata mismatch", nil)
	}
	codeData, codeOffset, ok := driver.memoryForPointer(codePointer, codeBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "codebook code buffer is missing", nil)
	}
	codebookData, codebookOffset, ok := driver.memoryForPointer(codebookPointer, codebookBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "codebook table buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "codebook output buffer is missing", nil)
	}
	codebook, err := hipFloat32PayloadValues(codebookData[codebookOffset : codebookOffset+codebookBytes])
	if err != nil {
		return err
	}
	output, err := rocmReferenceCodebookLookup(codeData[codeOffset:codeOffset+codeBytes], codebook, codeDim)
	if err != nil {
		return err
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchLoRAProjection(args []byte) error {
	if len(args) != hipLoRALaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "LoRA launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipLoRALaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipLoRALaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "LoRA launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	basePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	aPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	bPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	inputCount := int(binary.LittleEndian.Uint32(args[56:]))
	rows := int(binary.LittleEndian.Uint32(args[60:]))
	cols := int(binary.LittleEndian.Uint32(args[64:]))
	rank := int(binary.LittleEndian.Uint32(args[68:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[72:]))
	baseBytes := int(binary.LittleEndian.Uint32(args[76:]))
	aBytes := int(binary.LittleEndian.Uint32(args[80:]))
	bBytes := int(binary.LittleEndian.Uint32(args[84:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[88:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[92:]))
	alpha := math.Float32frombits(binary.LittleEndian.Uint32(args[96:]))
	flags := binary.LittleEndian.Uint32(args[100:])
	if inputCount != cols || rows <= 0 || cols <= 0 || rank <= 0 || inputBytes != cols*4 ||
		baseBytes != rows*cols*4 || aBytes != rank*cols*4 || bBytes != rows*rank*4 ||
		outputBytes != rows*4 || !hipQ8ScaleIsPositiveFinite(alpha) {
		return core.E("rocm.hip.FakeLaunch", "LoRA shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "LoRA input buffer is missing", nil)
	}
	baseData, baseOffset, ok := driver.memoryForPointer(basePointer, baseBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "LoRA base weight buffer is missing", nil)
	}
	aData, aOffset, ok := driver.memoryForPointer(aPointer, aBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "LoRA A buffer is missing", nil)
	}
	bData, bOffset, ok := driver.memoryForPointer(bPointer, bBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "LoRA B buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "LoRA output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	base, err := hipFloat32PayloadValues(baseData[baseOffset : baseOffset+baseBytes])
	if err != nil {
		return err
	}
	loraA, err := hipFloat32PayloadValues(aData[aOffset : aOffset+aBytes])
	if err != nil {
		return err
	}
	loraB, err := hipFloat32PayloadValues(bData[bOffset : bOffset+bBytes])
	if err != nil {
		return err
	}
	var bias []float32
	if flags&hipLoRALaunchFlagBias != 0 {
		biasData, biasOffset, ok := driver.memoryForPointer(biasPointer, biasBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "LoRA bias buffer is missing", nil)
		}
		bias, err = hipFloat32PayloadValues(biasData[biasOffset : biasOffset+biasBytes])
		if err != nil {
			return err
		}
	}
	output, err := rocmReferenceLoRAProjection(input[:inputCount], base, loraA, loraB, rows, cols, rank, alpha, bias)
	if err != nil {
		return err
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchEmbeddingLookup(args []byte) error {
	if len(args) != hipEmbeddingLookupLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "embedding lookup launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipEmbeddingLookupLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipEmbeddingLookupLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "embedding lookup launch header mismatch", nil)
	}
	tokenPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	embeddingPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	tokenCount := int(binary.LittleEndian.Uint32(args[32:]))
	vocabSize := int(binary.LittleEndian.Uint32(args[36:]))
	hiddenSize := int(binary.LittleEndian.Uint32(args[40:]))
	tokenBytes := int(binary.LittleEndian.Uint32(args[44:]))
	embeddingBytes := int(binary.LittleEndian.Uint64(args[48:]))
	outputBytes := int(binary.LittleEndian.Uint64(args[56:]))
	encoding := binary.LittleEndian.Uint32(args[64:])
	groupSize := int(binary.LittleEndian.Uint32(args[68:]))
	scalePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[72:]))
	biasPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[80:]))
	scaleBytes := int(binary.LittleEndian.Uint32(args[88:]))
	biasBytes := int(binary.LittleEndian.Uint32(args[92:]))
	if tokenCount <= 0 || vocabSize <= 0 || hiddenSize <= 0 || tokenBytes != tokenCount*4 || outputBytes != tokenCount*hiddenSize*4 {
		return core.E("rocm.hip.FakeLaunch", "embedding lookup shape metadata mismatch", nil)
	}
	tokenData, tokenOffset, ok := driver.memoryForPointer(tokenPointer, tokenBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "embedding lookup token buffer is missing", nil)
	}
	embeddingData, embeddingOffset, ok := driver.memoryForPointer(embeddingPointer, embeddingBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "embedding lookup table buffer is missing", nil)
	}
	var scaleData []byte
	var scaleOffset int
	var biasData []byte
	var biasOffset int
	if encoding == hipEmbeddingTableEncodingMLXQ4 {
		if groupSize <= 0 || hiddenSize%8 != 0 || hiddenSize%groupSize != 0 {
			return core.E("rocm.hip.FakeLaunch", "embedding lookup q4 shape metadata mismatch", nil)
		}
		if scaleBytes != vocabSize*(hiddenSize/groupSize)*2 || biasBytes != scaleBytes {
			return core.E("rocm.hip.FakeLaunch", "embedding lookup q4 scale/bias byte count mismatch", nil)
		}
		var ok bool
		scaleData, scaleOffset, ok = driver.memoryForPointer(scalePointer, scaleBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "embedding lookup q4 scale buffer is missing", nil)
		}
		biasData, biasOffset, ok = driver.memoryForPointer(biasPointer, biasBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "embedding lookup q4 bias buffer is missing", nil)
		}
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "embedding lookup output buffer is missing", nil)
	}
	output := make([]float32, tokenCount*hiddenSize)
	for tokenIndex := 0; tokenIndex < tokenCount; tokenIndex++ {
		id := int(int32(binary.LittleEndian.Uint32(tokenData[tokenOffset+tokenIndex*4:])))
		if id < 0 || id >= vocabSize {
			return core.E("rocm.hip.FakeLaunch", "embedding lookup token ID is outside vocabulary", nil)
		}
		for dim := 0; dim < hiddenSize; dim++ {
			tableIndex := id*hiddenSize + dim
			switch encoding {
			case hipEmbeddingTableEncodingF32:
				if embeddingBytes != vocabSize*hiddenSize*4 {
					return core.E("rocm.hip.FakeLaunch", "embedding lookup f32 byte count mismatch", nil)
				}
				output[tokenIndex*hiddenSize+dim] = math.Float32frombits(binary.LittleEndian.Uint32(embeddingData[embeddingOffset+tableIndex*4:]))
			case hipEmbeddingTableEncodingBF16:
				if embeddingBytes != vocabSize*hiddenSize*2 {
					return core.E("rocm.hip.FakeLaunch", "embedding lookup bf16 byte count mismatch", nil)
				}
				output[tokenIndex*hiddenSize+dim] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(embeddingData[embeddingOffset+tableIndex*2:]))
			case hipEmbeddingTableEncodingMLXQ4:
				packedPerRow := hiddenSize / 8
				groupsPerRow := hiddenSize / groupSize
				if embeddingBytes != vocabSize*packedPerRow*4 {
					return core.E("rocm.hip.FakeLaunch", "embedding lookup q4 byte count mismatch", nil)
				}
				wordIndex := id*packedPerRow + dim/8
				word := binary.LittleEndian.Uint32(embeddingData[embeddingOffset+wordIndex*4:])
				quantized := float32((word >> uint((dim%8)*4)) & 0x0f)
				group := id*groupsPerRow + dim/groupSize
				scale := hipBFloat16ToFloat32(binary.LittleEndian.Uint16(scaleData[scaleOffset+group*2:]))
				bias := hipBFloat16ToFloat32(binary.LittleEndian.Uint16(biasData[biasOffset+group*2:]))
				output[tokenIndex*hiddenSize+dim] = quantized*scale + bias
			default:
				return core.E("rocm.hip.FakeLaunch", "unsupported embedding lookup encoding", nil)
			}
		}
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchEmbeddingMeanPool(args []byte) error {
	if len(args) != hipEmbeddingMeanPoolLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "embedding mean-pool launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipEmbeddingMeanPoolLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipEmbeddingMeanPoolLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "embedding mean-pool launch header mismatch", nil)
	}
	tokenPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	tokenCount := int(binary.LittleEndian.Uint32(args[24:]))
	dim := int(binary.LittleEndian.Uint32(args[28:]))
	tokenBytes := int(binary.LittleEndian.Uint32(args[32:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[36:]))
	flags := binary.LittleEndian.Uint32(args[40:])
	if tokenCount <= 0 || dim <= 0 || tokenBytes != tokenCount*dim*4 || outputBytes != dim*4 {
		return core.E("rocm.hip.FakeLaunch", "embedding mean-pool shape metadata mismatch", nil)
	}
	tokenData, tokenOffset, ok := driver.memoryForPointer(tokenPointer, tokenBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "embedding token buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "embedding output buffer is missing", nil)
	}
	tokens, err := hipFloat32PayloadValues(tokenData[tokenOffset : tokenOffset+tokenBytes])
	if err != nil {
		return err
	}
	output, err := rocmReferenceMeanPoolEmbedding(splitFloat32Vectors(tokens, dim), flags&hipEmbeddingMeanPoolLaunchFlagNormalize != 0)
	if err != nil {
		return err
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchRerankCosine(args []byte) error {
	if len(args) != hipRerankCosineLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "rerank cosine launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipRerankCosineLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipRerankCosineLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "rerank cosine launch header mismatch", nil)
	}
	queryPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	documentPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	documentCount := int(binary.LittleEndian.Uint32(args[32:]))
	dim := int(binary.LittleEndian.Uint32(args[36:]))
	queryBytes := int(binary.LittleEndian.Uint32(args[40:]))
	documentBytes := int(binary.LittleEndian.Uint32(args[44:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[48:]))
	if documentCount <= 0 || dim <= 0 || queryBytes != dim*4 || documentBytes != documentCount*dim*4 || outputBytes != documentCount*4 {
		return core.E("rocm.hip.FakeLaunch", "rerank cosine shape metadata mismatch", nil)
	}
	queryData, queryOffset, ok := driver.memoryForPointer(queryPointer, queryBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rerank query buffer is missing", nil)
	}
	documentData, documentOffset, ok := driver.memoryForPointer(documentPointer, documentBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rerank document buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rerank output buffer is missing", nil)
	}
	query, err := hipFloat32PayloadValues(queryData[queryOffset : queryOffset+queryBytes])
	if err != nil {
		return err
	}
	documents, err := hipFloat32PayloadValues(documentData[documentOffset : documentOffset+documentBytes])
	if err != nil {
		return err
	}
	scores := make([]float32, documentCount)
	for index := 0; index < documentCount; index++ {
		start := index * dim
		score, err := rocmReferenceCosineSimilarity(query, documents[start:start+dim])
		if err != nil {
			return err
		}
		scores[index] = float32(score)
	}
	payload, err := hipFloat32Payload(scores)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchRMSNorm(args []byte) error {
	if len(args) != hipRMSNormLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "rms norm launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipRMSNormLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipRMSNormLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "rms norm launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	count := int(binary.LittleEndian.Uint32(args[32:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[36:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[40:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[44:]))
	epsilon := math.Float32frombits(binary.LittleEndian.Uint32(args[48:]))
	encoding := binary.LittleEndian.Uint32(args[52:])
	flags := binary.LittleEndian.Uint32(args[56:])
	if encoding == 0 {
		encoding = hipRMSNormWeightEncodingF32
	}
	if count <= 0 || inputBytes != count*4 || outputBytes != count*4 {
		return core.E("rocm.hip.FakeLaunch", "rms norm shape metadata mismatch", nil)
	}
	if flags&^hipRMSNormLaunchFlagAddUnitWeight != 0 {
		return core.E("rocm.hip.FakeLaunch", "unsupported rms norm flags", nil)
	}
	switch encoding {
	case hipRMSNormWeightEncodingF32:
		if weightBytes != count*4 {
			return core.E("rocm.hip.FakeLaunch", "rms norm f32 weight byte count mismatch", nil)
		}
	case hipRMSNormWeightEncodingBF16:
		if weightBytes != count*2 {
			return core.E("rocm.hip.FakeLaunch", "rms norm bf16 weight byte count mismatch", nil)
		}
	default:
		return core.E("rocm.hip.FakeLaunch", "unsupported rms norm weight encoding", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm input buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm weight buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rms norm output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	weight := make([]float32, count)
	switch encoding {
	case hipRMSNormWeightEncodingF32:
		weight, err = hipFloat32PayloadValues(weightData[weightOffset : weightOffset+weightBytes])
		if err != nil {
			return err
		}
	case hipRMSNormWeightEncodingBF16:
		for index := range weight {
			weight[index] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(weightData[weightOffset+index*2:]))
		}
	}
	if flags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
		for index := range weight {
			weight[index] += 1
		}
	}
	output, err := hipReferenceRMSNorm(input, weight, epsilon)
	if err != nil {
		return err
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchRoPE(args []byte) error {
	if len(args) != hipRoPELaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "rope launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipRoPELaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipRoPELaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "rope launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	count := int(binary.LittleEndian.Uint32(args[24:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[28:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[32:]))
	position := int(binary.LittleEndian.Uint32(args[36:]))
	base := math.Float32frombits(binary.LittleEndian.Uint32(args[40:]))
	frequencyDim := int(binary.LittleEndian.Uint32(args[44:]))
	if count <= 0 || count%2 != 0 || inputBytes != count*4 || outputBytes != count*4 {
		return core.E("rocm.hip.FakeLaunch", "rope shape metadata mismatch", nil)
	}
	if frequencyDim > 0 && frequencyDim < count {
		return core.E("rocm.hip.FakeLaunch", "rope frequency dimension mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rope input buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "rope output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	output, err := hipReferenceRoPEWithFrequencyDim(input, position, float64(base), frequencyDim)
	if err != nil {
		return err
	}
	payload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchGreedySample(args []byte) error {
	if len(args) != hipGreedyLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "greedy launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipGreedyLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipGreedyLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "greedy launch header mismatch", nil)
	}
	logitsPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	count := int(binary.LittleEndian.Uint32(args[24:]))
	logitsBytes := int(binary.LittleEndian.Uint32(args[28:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[32:]))
	if count <= 0 || logitsBytes != count*4 || outputBytes != hipGreedyResultBytes {
		return core.E("rocm.hip.FakeLaunch", "greedy shape metadata mismatch", nil)
	}
	logitsData, logitsOffset, ok := driver.memoryForPointer(logitsPointer, logitsBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "greedy logits buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "greedy output buffer is missing", nil)
	}
	logits, err := hipFloat32PayloadValues(logitsData[logitsOffset : logitsOffset+logitsBytes])
	if err != nil {
		return err
	}
	index, score, err := hipReferenceGreedySample(logits)
	if err != nil {
		return err
	}
	binary.LittleEndian.PutUint32(outputData[outputOffset:], uint32(int32(index)))
	binary.LittleEndian.PutUint32(outputData[outputOffset+4:], math.Float32bits(score))
	return nil
}

func (driver *fakeHIPDriver) launchAttention(args []byte) error {
	if len(args) != hipAttentionLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "attention launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipAttentionLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipAttentionLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "attention launch header mismatch", nil)
	}
	queryPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	keyPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	valuePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	dim := int(binary.LittleEndian.Uint32(args[48:]))
	tokenCount := int(binary.LittleEndian.Uint32(args[52:]))
	queryBytes := int(binary.LittleEndian.Uint32(args[56:]))
	keyBytes := int(binary.LittleEndian.Uint32(args[60:]))
	valueBytes := int(binary.LittleEndian.Uint32(args[64:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[68:]))
	weightBytes := int(binary.LittleEndian.Uint32(args[72:]))
	kvSource := binary.LittleEndian.Uint32(args[76:])
	scale := math.Float32frombits(binary.LittleEndian.Uint32(args[80:]))
	descriptorPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[88:]))
	descriptorBytes := int(binary.LittleEndian.Uint64(args[96:]))
	if dim <= 0 || tokenCount <= 0 || queryBytes != dim*4 || outputBytes != dim*4 || weightBytes != tokenCount*4 {
		return core.E("rocm.hip.FakeLaunch", "attention shape metadata mismatch", nil)
	}
	if scale < 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return core.E("rocm.hip.FakeLaunch", "attention scale is invalid", nil)
	}
	queryData, queryOffset, ok := driver.memoryForPointer(queryPointer, queryBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention query buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention output buffer is missing", nil)
	}
	weightData, weightOffset, ok := driver.memoryForPointer(weightPointer, weightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "attention weight buffer is missing", nil)
	}
	query, err := hipFloat32PayloadValues(queryData[queryOffset : queryOffset+queryBytes])
	if err != nil {
		return err
	}
	var keyFlat []float32
	var valueFlat []float32
	switch kvSource {
	case hipAttentionKVSourceContiguous:
		if keyBytes != dim*tokenCount*4 || valueBytes != dim*tokenCount*4 {
			return core.E("rocm.hip.FakeLaunch", "attention shape metadata mismatch", nil)
		}
		keyData, keyOffset, ok := driver.memoryForPointer(keyPointer, keyBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention key buffer is missing", nil)
		}
		valueData, valueOffset, ok := driver.memoryForPointer(valuePointer, valueBytes)
		if !ok {
			return core.E("rocm.hip.FakeLaunch", "attention value buffer is missing", nil)
		}
		keyFlat, err = hipFloat32PayloadValues(keyData[keyOffset : keyOffset+keyBytes])
		if err != nil {
			return err
		}
		valueFlat, err = hipFloat32PayloadValues(valueData[valueOffset : valueOffset+valueBytes])
		if err != nil {
			return err
		}
	case hipAttentionKVSourceDevice:
		if keyPointer != 0 || valuePointer != 0 || keyBytes != 0 || valueBytes != 0 {
			return core.E("rocm.hip.FakeLaunch", "attention device KV source must not include contiguous KV buffers", nil)
		}
		keyFlat, valueFlat, err = driver.readDeviceKVDescriptorForAttention(descriptorPointer, descriptorBytes, tokenCount, dim)
		if err != nil {
			return err
		}
	default:
		return core.E("rocm.hip.FakeLaunch", "attention KV source is unsupported", nil)
	}
	keys, err := splitHIPReferenceVectors(keyFlat, dim)
	if err != nil {
		return err
	}
	values, err := splitHIPReferenceVectors(valueFlat, dim)
	if err != nil {
		return err
	}
	output, weights, err := hipReferenceSingleHeadAttentionWithScale(query, keys, values, scale)
	if err != nil {
		return err
	}
	outputPayload, err := hipFloat32Payload(output)
	if err != nil {
		return err
	}
	weightPayload, err := hipFloat32Payload(weights)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], outputPayload)
	copy(weightData[weightOffset:weightOffset+weightBytes], weightPayload)
	return nil
}

func (driver *fakeHIPDriver) readDeviceKVDescriptorForAttention(pointer nativeDevicePointer, sizeBytes, tokenCount, dim int) ([]float32, []float32, error) {
	if pointer == 0 || sizeBytes < rocmDeviceKVDescriptorHeaderBytes {
		return nil, nil, core.E("rocm.hip.FakeLaunch", "attention device KV descriptor is missing", nil)
	}
	data, offset, ok := driver.memoryForPointer(pointer, sizeBytes)
	if !ok {
		return nil, nil, core.E("rocm.hip.FakeLaunch", "attention device KV descriptor buffer is missing", nil)
	}
	descriptor := data[offset : offset+sizeBytes]
	if binary.LittleEndian.Uint32(descriptor[0:]) != rocmDeviceKVDescriptorVersion ||
		int(binary.LittleEndian.Uint32(descriptor[4:])) != rocmDeviceKVDescriptorHeaderBytes ||
		int(binary.LittleEndian.Uint32(descriptor[8:])) != rocmDeviceKVDescriptorPageBytes ||
		int(binary.LittleEndian.Uint64(descriptor[24:])) != tokenCount {
		return nil, nil, core.E("rocm.hip.FakeLaunch", "attention device KV descriptor header mismatch", nil)
	}
	pageCount := int(binary.LittleEndian.Uint32(descriptor[16:]))
	if sizeBytes != rocmDeviceKVDescriptorHeaderBytes+pageCount*rocmDeviceKVDescriptorPageBytes {
		return nil, nil, core.E("rocm.hip.FakeLaunch", "attention device KV descriptor size mismatch", nil)
	}
	keys := make([]float32, tokenCount*dim)
	values := make([]float32, tokenCount*dim)
	for pageIndex := 0; pageIndex < pageCount; pageIndex++ {
		pageOffset := rocmDeviceKVDescriptorHeaderBytes + pageIndex*rocmDeviceKVDescriptorPageBytes
		page := descriptor[pageOffset : pageOffset+rocmDeviceKVDescriptorPageBytes]
		tokenStart := int(binary.LittleEndian.Uint64(page[0:]))
		pageTokens := int(binary.LittleEndian.Uint64(page[8:]))
		keyWidth := int(binary.LittleEndian.Uint32(page[16:]))
		valueWidth := int(binary.LittleEndian.Uint32(page[20:]))
		keyEncoding := fakeROCmKVEncoding(binary.LittleEndian.Uint32(page[24:]))
		valueEncoding := fakeROCmKVEncoding(binary.LittleEndian.Uint32(page[28:]))
		keyPointer := nativeDevicePointer(binary.LittleEndian.Uint64(page[32:]))
		valuePointer := nativeDevicePointer(binary.LittleEndian.Uint64(page[40:]))
		keyBytes := int(binary.LittleEndian.Uint64(page[48:]))
		valueBytes := int(binary.LittleEndian.Uint64(page[56:]))
		if tokenStart < 0 || pageTokens <= 0 || tokenStart+pageTokens > tokenCount || keyWidth != dim || valueWidth != dim || keyEncoding == "" || valueEncoding == "" {
			return nil, nil, core.E("rocm.hip.FakeLaunch", "attention device KV descriptor page shape mismatch", nil)
		}
		pageKeys, err := driver.readDeviceKVTensor(keyPointer, keyBytes, keyEncoding, pageTokens*keyWidth)
		if err != nil {
			return nil, nil, err
		}
		pageValues, err := driver.readDeviceKVTensor(valuePointer, valueBytes, valueEncoding, pageTokens*valueWidth)
		if err != nil {
			return nil, nil, err
		}
		copy(keys[tokenStart*dim:(tokenStart+pageTokens)*dim], pageKeys)
		copy(values[tokenStart*dim:(tokenStart+pageTokens)*dim], pageValues)
	}
	return keys, values, nil
}

func (driver *fakeHIPDriver) readDeviceKVTensor(pointer nativeDevicePointer, sizeBytes int, encoding string, length int) ([]float32, error) {
	data, offset, ok := driver.memoryForPointer(pointer, sizeBytes)
	if !ok {
		return nil, core.E("rocm.hip.FakeLaunch", "attention device KV tensor buffer is missing", nil)
	}
	tensor, err := rocmKVTensorFromDeviceBytes(encoding, length, append([]byte(nil), data[offset:offset+sizeBytes]...))
	if err != nil {
		return nil, err
	}
	return tensor.decode(), nil
}

func fakeROCmKVEncoding(code uint32) string {
	switch code {
	case rocmDeviceKVDescriptorEncodingFP16:
		return rocmKVEncodingFP16
	case rocmDeviceKVDescriptorEncodingQ8:
		return rocmKVEncodingQ8
	case rocmDeviceKVDescriptorEncodingQ4:
		return rocmKVEncodingQ4
	default:
		return ""
	}
}

func (driver *fakeHIPDriver) launchVectorAdd(args []byte) error {
	if len(args) != hipVectorAddLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "vector add launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipVectorAddLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipVectorAddLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "vector add launch header mismatch", nil)
	}
	leftPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	rightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	count := int(binary.LittleEndian.Uint32(args[32:]))
	leftBytes := int(binary.LittleEndian.Uint32(args[36:]))
	rightBytes := int(binary.LittleEndian.Uint32(args[40:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[44:]))
	if count <= 0 || leftBytes != count*4 || rightBytes != count*4 || outputBytes != count*4 {
		return core.E("rocm.hip.FakeLaunch", "vector add shape metadata mismatch", nil)
	}
	leftData, leftOffset, ok := driver.memoryForPointer(leftPointer, leftBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "vector add left buffer is missing", nil)
	}
	rightData, rightOffset, ok := driver.memoryForPointer(rightPointer, rightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "vector add right buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "vector add output buffer is missing", nil)
	}
	left, err := hipFloat32PayloadValues(leftData[leftOffset : leftOffset+leftBytes])
	if err != nil {
		return err
	}
	right, err := hipFloat32PayloadValues(rightData[rightOffset : rightOffset+rightBytes])
	if err != nil {
		return err
	}
	out := make([]float32, count)
	for index := range out {
		out[index] = left[index] + right[index]
	}
	payload, err := hipFloat32Payload(out)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchVectorScale(args []byte) error {
	if len(args) != hipVectorScaleLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "vector scale launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipVectorScaleLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipVectorScaleLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "vector scale launch header mismatch", nil)
	}
	inputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	count := int(binary.LittleEndian.Uint32(args[24:]))
	inputBytes := int(binary.LittleEndian.Uint32(args[28:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[32:]))
	scale := math.Float32frombits(binary.LittleEndian.Uint32(args[36:]))
	if count <= 0 || inputBytes != count*4 || outputBytes != count*4 ||
		math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return core.E("rocm.hip.FakeLaunch", "vector scale shape metadata mismatch", nil)
	}
	inputData, inputOffset, ok := driver.memoryForPointer(inputPointer, inputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "vector scale input buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "vector scale output buffer is missing", nil)
	}
	input, err := hipFloat32PayloadValues(inputData[inputOffset : inputOffset+inputBytes])
	if err != nil {
		return err
	}
	out := make([]float32, count)
	for index := range out {
		out[index] = input[index] * scale
	}
	payload, err := hipFloat32Payload(out)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchSwiGLU(args []byte) error {
	if len(args) != hipSwiGLULaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "swiglu launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipSwiGLULaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipSwiGLULaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "swiglu launch header mismatch", nil)
	}
	gatePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	upPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	count := int(binary.LittleEndian.Uint32(args[32:]))
	gateBytes := int(binary.LittleEndian.Uint32(args[36:]))
	upBytes := int(binary.LittleEndian.Uint32(args[40:]))
	outputBytes := int(binary.LittleEndian.Uint32(args[44:]))
	if count <= 0 || gateBytes != count*4 || upBytes != count*4 || outputBytes != count*4 {
		return core.E("rocm.hip.FakeLaunch", "swiglu shape metadata mismatch", nil)
	}
	gateData, gateOffset, ok := driver.memoryForPointer(gatePointer, gateBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "swiglu gate buffer is missing", nil)
	}
	upData, upOffset, ok := driver.memoryForPointer(upPointer, upBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "swiglu up buffer is missing", nil)
	}
	outputData, outputOffset, ok := driver.memoryForPointer(outputPointer, outputBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "swiglu output buffer is missing", nil)
	}
	gate, err := hipFloat32PayloadValues(gateData[gateOffset : gateOffset+gateBytes])
	if err != nil {
		return err
	}
	up, err := hipFloat32PayloadValues(upData[upOffset : upOffset+upBytes])
	if err != nil {
		return err
	}
	out := make([]float32, count)
	for index := range out {
		out[index] = gate[index] / (1 + float32(math.Exp(float64(-gate[index])))) * up[index]
	}
	payload, err := hipFloat32Payload(out)
	if err != nil {
		return err
	}
	copy(outputData[outputOffset:outputOffset+outputBytes], payload)
	return nil
}

func (driver *fakeHIPDriver) launchTinyPrefill(args []byte) error {
	if len(args) != hipTinyPrefillLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipTinyPrefillLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipTinyPrefillLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill launch header mismatch", nil)
	}
	tokenPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	embeddingPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	outputWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	logitPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	attentionPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	resultPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	keyPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[56:]))
	valuePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[64:]))
	tokenCount := int(binary.LittleEndian.Uint32(args[72:]))
	vocabSize := int(binary.LittleEndian.Uint32(args[76:]))
	hiddenSize := int(binary.LittleEndian.Uint32(args[80:]))
	tokenBytes := int(binary.LittleEndian.Uint32(args[84:]))
	embeddingBytes := int(binary.LittleEndian.Uint32(args[88:]))
	outputWeightBytes := int(binary.LittleEndian.Uint32(args[92:]))
	logitBytes := int(binary.LittleEndian.Uint32(args[96:]))
	attentionBytes := int(binary.LittleEndian.Uint32(args[100:]))
	resultBytes := int(binary.LittleEndian.Uint32(args[104:]))
	keyBytes := int(binary.LittleEndian.Uint32(args[108:]))
	valueBytes := int(binary.LittleEndian.Uint32(args[112:]))
	outputWeightEncoding := binary.LittleEndian.Uint32(args[116:])
	q8Scale := math.Float32frombits(binary.LittleEndian.Uint32(args[120:]))
	expectedOutputWeightBytes, err := hipTinyOutputWeightByteCount(outputWeightEncoding, uint64(outputWeightBytes), uint64(vocabSize*hiddenSize), q8Scale)
	if err != nil {
		return err
	}
	stateBytes := tokenCount * hiddenSize * 4
	if tokenCount <= 0 || vocabSize <= 0 || hiddenSize <= 0 ||
		tokenBytes != tokenCount*4 ||
		embeddingBytes != vocabSize*hiddenSize*4 ||
		outputWeightBytes != int(expectedOutputWeightBytes) ||
		logitBytes != vocabSize*4 ||
		attentionBytes != tokenCount*4 ||
		keyBytes != stateBytes ||
		valueBytes != stateBytes ||
		resultBytes != hipGreedyResultBytes {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill shape metadata mismatch", nil)
	}
	tokenData, tokenOffset, ok := driver.memoryForPointer(tokenPointer, tokenBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill token buffer is missing", nil)
	}
	embeddingData, embeddingOffset, ok := driver.memoryForPointer(embeddingPointer, embeddingBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill embedding buffer is missing", nil)
	}
	outputWeightData, outputWeightOffset, ok := driver.memoryForPointer(outputWeightPointer, outputWeightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill output weight buffer is missing", nil)
	}
	logitData, logitOffset, ok := driver.memoryForPointer(logitPointer, logitBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill logit buffer is missing", nil)
	}
	attentionData, attentionOffset, ok := driver.memoryForPointer(attentionPointer, attentionBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill attention buffer is missing", nil)
	}
	keyData, keyOffset, ok := driver.memoryForPointer(keyPointer, keyBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill key buffer is missing", nil)
	}
	valueData, valueOffset, ok := driver.memoryForPointer(valuePointer, valueBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill value buffer is missing", nil)
	}
	resultData, resultOffset, ok := driver.memoryForPointer(resultPointer, resultBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny prefill result buffer is missing", nil)
	}
	tokens := make([]int32, tokenCount)
	for index := range tokens {
		tokens[index] = int32(binary.LittleEndian.Uint32(tokenData[tokenOffset+index*4:]))
	}
	embedding, err := hipFloat32PayloadValues(embeddingData[embeddingOffset : embeddingOffset+embeddingBytes])
	if err != nil {
		return err
	}
	outputWeights, err := hipTinyOutputWeightValues(outputWeightData[outputWeightOffset:outputWeightOffset+outputWeightBytes], outputWeightEncoding, q8Scale)
	if err != nil {
		return err
	}
	result, err := hipReferenceTinyPrefill(hipReferenceTinyLMConfig{
		EmbeddingTable: embedding,
		OutputWeights:  outputWeights,
		VocabSize:      vocabSize,
		HiddenSize:     hiddenSize,
	}, tokens)
	if err != nil {
		return err
	}
	logitPayload, err := hipFloat32Payload(result.Logits)
	if err != nil {
		return err
	}
	attentionPayload, err := hipFloat32Payload(result.Attention)
	if err != nil {
		return err
	}
	keyPayload, err := hipFloat32Payload(flattenHIPReferenceMatrix(result.State.Keys))
	if err != nil {
		return err
	}
	valuePayload, err := hipFloat32Payload(flattenHIPReferenceMatrix(result.State.Values))
	if err != nil {
		return err
	}
	copy(logitData[logitOffset:logitOffset+logitBytes], logitPayload)
	copy(attentionData[attentionOffset:attentionOffset+attentionBytes], attentionPayload)
	copy(keyData[keyOffset:keyOffset+keyBytes], keyPayload)
	copy(valueData[valueOffset:valueOffset+valueBytes], valuePayload)
	binary.LittleEndian.PutUint32(resultData[resultOffset:], uint32(int32(result.NextTokenID)))
	binary.LittleEndian.PutUint32(resultData[resultOffset+4:], math.Float32bits(result.NextScore))
	return nil
}

func (driver *fakeHIPDriver) launchTinyDecode(args []byte) error {
	if len(args) != hipTinyDecodeLaunchArgsBytes {
		return core.E("rocm.hip.FakeLaunch", "tiny decode launch args size mismatch", nil)
	}
	if binary.LittleEndian.Uint32(args[0:]) != hipTinyDecodeLaunchArgsVersion ||
		binary.LittleEndian.Uint32(args[4:]) != uint32(hipTinyDecodeLaunchArgsBytes) {
		return core.E("rocm.hip.FakeLaunch", "tiny decode launch header mismatch", nil)
	}
	priorKeyPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[8:]))
	priorValuePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[16:]))
	embeddingPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[24:]))
	outputWeightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[32:]))
	logitPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[40:]))
	attentionPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[48:]))
	updatedKeyPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[56:]))
	updatedValuePointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[64:]))
	resultPointer := nativeDevicePointer(binary.LittleEndian.Uint64(args[72:]))
	tokenID := int32(binary.LittleEndian.Uint32(args[80:]))
	priorTokenCount := int(binary.LittleEndian.Uint32(args[84:]))
	vocabSize := int(binary.LittleEndian.Uint32(args[88:]))
	hiddenSize := int(binary.LittleEndian.Uint32(args[92:]))
	priorKeyBytes := int(binary.LittleEndian.Uint32(args[96:]))
	priorValueBytes := int(binary.LittleEndian.Uint32(args[100:]))
	embeddingBytes := int(binary.LittleEndian.Uint32(args[104:]))
	outputWeightBytes := int(binary.LittleEndian.Uint32(args[108:]))
	logitBytes := int(binary.LittleEndian.Uint32(args[112:]))
	attentionBytes := int(binary.LittleEndian.Uint32(args[116:]))
	updatedKeyBytes := int(binary.LittleEndian.Uint32(args[120:]))
	updatedValueBytes := int(binary.LittleEndian.Uint32(args[124:]))
	resultBytes := int(binary.LittleEndian.Uint32(args[128:]))
	outputWeightEncoding := binary.LittleEndian.Uint32(args[132:])
	q8Scale := math.Float32frombits(binary.LittleEndian.Uint32(args[136:]))
	expectedOutputWeightBytes, err := hipTinyOutputWeightByteCount(outputWeightEncoding, uint64(outputWeightBytes), uint64(vocabSize*hiddenSize), q8Scale)
	if err != nil {
		return err
	}
	if tokenID < 0 || priorTokenCount <= 0 || vocabSize <= 0 || hiddenSize <= 0 ||
		int(tokenID) >= vocabSize ||
		priorKeyBytes != priorTokenCount*hiddenSize*4 ||
		priorValueBytes != priorTokenCount*hiddenSize*4 ||
		embeddingBytes != vocabSize*hiddenSize*4 ||
		outputWeightBytes != int(expectedOutputWeightBytes) ||
		logitBytes != vocabSize*4 ||
		attentionBytes != (priorTokenCount+1)*4 ||
		updatedKeyBytes != (priorTokenCount+1)*hiddenSize*4 ||
		updatedValueBytes != (priorTokenCount+1)*hiddenSize*4 ||
		resultBytes != hipGreedyResultBytes {
		return core.E("rocm.hip.FakeLaunch", "tiny decode shape metadata mismatch", nil)
	}
	priorKeyData, priorKeyOffset, ok := driver.memoryForPointer(priorKeyPointer, priorKeyBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode prior key buffer is missing", nil)
	}
	priorValueData, priorValueOffset, ok := driver.memoryForPointer(priorValuePointer, priorValueBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode prior value buffer is missing", nil)
	}
	embeddingData, embeddingOffset, ok := driver.memoryForPointer(embeddingPointer, embeddingBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode embedding buffer is missing", nil)
	}
	outputWeightData, outputWeightOffset, ok := driver.memoryForPointer(outputWeightPointer, outputWeightBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode output weight buffer is missing", nil)
	}
	logitData, logitOffset, ok := driver.memoryForPointer(logitPointer, logitBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode logit buffer is missing", nil)
	}
	attentionData, attentionOffset, ok := driver.memoryForPointer(attentionPointer, attentionBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode attention buffer is missing", nil)
	}
	updatedKeyData, updatedKeyOffset, ok := driver.memoryForPointer(updatedKeyPointer, updatedKeyBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode updated key buffer is missing", nil)
	}
	updatedValueData, updatedValueOffset, ok := driver.memoryForPointer(updatedValuePointer, updatedValueBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode updated value buffer is missing", nil)
	}
	resultData, resultOffset, ok := driver.memoryForPointer(resultPointer, resultBytes)
	if !ok {
		return core.E("rocm.hip.FakeLaunch", "tiny decode result buffer is missing", nil)
	}
	priorKeysFlat, err := hipFloat32PayloadValues(priorKeyData[priorKeyOffset : priorKeyOffset+priorKeyBytes])
	if err != nil {
		return err
	}
	priorValuesFlat, err := hipFloat32PayloadValues(priorValueData[priorValueOffset : priorValueOffset+priorValueBytes])
	if err != nil {
		return err
	}
	priorKeys, err := splitHIPReferenceVectors(priorKeysFlat, hiddenSize)
	if err != nil {
		return err
	}
	priorValues, err := splitHIPReferenceVectors(priorValuesFlat, hiddenSize)
	if err != nil {
		return err
	}
	embedding, err := hipFloat32PayloadValues(embeddingData[embeddingOffset : embeddingOffset+embeddingBytes])
	if err != nil {
		return err
	}
	outputWeights, err := hipTinyOutputWeightValues(outputWeightData[outputWeightOffset:outputWeightOffset+outputWeightBytes], outputWeightEncoding, q8Scale)
	if err != nil {
		return err
	}
	result, err := hipReferenceTinyDecode(hipReferenceTinyLMConfig{
		EmbeddingTable: embedding,
		OutputWeights:  outputWeights,
		VocabSize:      vocabSize,
		HiddenSize:     hiddenSize,
	}, hipReferenceTinyLMState{Keys: priorKeys, Values: priorValues}, tokenID)
	if err != nil {
		return err
	}
	logitPayload, err := hipFloat32Payload(result.Logits)
	if err != nil {
		return err
	}
	attentionPayload, err := hipFloat32Payload(result.Attention)
	if err != nil {
		return err
	}
	updatedKeysPayload, err := hipFloat32Payload(flattenHIPReferenceMatrix(result.State.Keys))
	if err != nil {
		return err
	}
	updatedValuesPayload, err := hipFloat32Payload(flattenHIPReferenceMatrix(result.State.Values))
	if err != nil {
		return err
	}
	copy(logitData[logitOffset:logitOffset+logitBytes], logitPayload)
	copy(attentionData[attentionOffset:attentionOffset+attentionBytes], attentionPayload)
	copy(updatedKeyData[updatedKeyOffset:updatedKeyOffset+updatedKeyBytes], updatedKeysPayload)
	copy(updatedValueData[updatedValueOffset:updatedValueOffset+updatedValueBytes], updatedValuesPayload)
	binary.LittleEndian.PutUint32(resultData[resultOffset:], uint32(int32(result.NextTokenID)))
	binary.LittleEndian.PutUint32(resultData[resultOffset+4:], math.Float32bits(result.NextScore))
	return nil
}

func nativeHIPTensorGGUF(t *testing.T) (string, int64) {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "weights.gguf")
	result := core.WriteFile(path, []byte("0123456789abcdef0123456789abcdef"), 0o644)
	core.RequireTrue(t, result.OK)
	return path, 0
}
