<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-rocm HIP Kernels

`rocm_kernels.hip` contains the first native kernel source for the launch ABI used by `go/hip_launch.go`.

Build a gfx1100 HSACO on a ROCm machine:

```bash
mkdir -p build
hipcc --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o build/rocm_kernels_gfx1100.hsaco
GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=$PWD/build/rocm_kernels_gfx1100.hsaco go test ./go -run 'TestHIPHardware.*KernelSource' -count=1 -v
```

The exported symbols must stay in sync with the Go launcher names:

- `rocm_prefill`
- `rocm_decode`
- `rocm_projection`
- `rocm_mlx_q4_projection`
- `rocm_rms_norm`
- `rocm_rope`
- `rocm_greedy_sample`
- `rocm_attention`
- `rocm_vector_add`
- `rocm_vector_scale`
- `rocm_swiglu`
- `rocm_embedding_lookup`
- `rocm_embedding_mean_pool`
- `rocm_rerank_cosine`
- `rocm_tiny_prefill`
- `rocm_tiny_decode`

The prefill and decode kernels currently validate and consume their launch packets, referenced device memory, and optional status-output pointers in the reserved packet fields; the hardware smoke covers fp16, q8, and k-q8-v-q4 cache-mode descriptors. The projection kernels perform the toy fp16/q8/BF16 row projections and MLX affine q4 packed row projection used by the Go fake-driver fixtures and loaded-model projection smoke. `rocm_embedding_lookup` supports f32, BF16, and MLX affine q4 embedding tables, including loaded Gemma4 q4 packed U32 weights with BF16 scales/biases. The RMSNorm, RoPE, greedy sampler, single-head attention, vector-add, vector-scale, and SwiGLU kernels execute deterministic transformer primitive fixtures. `rocm_tiny_prefill` is a toy embedding-attention-output fixture that writes toy KV buffers, logits, final-token attention weights, and a greedy result buffer. `rocm_tiny_decode` consumes those toy prior KV vectors, appends the decoded token embedding, and writes updated KV, logits, attention, and greedy result buffers. The tiny kernels accept fp32, fp16, or q8 output-head weights. These tiny kernels are not yet the production loaded-model generation path.
