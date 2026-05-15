// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"os"
	"strings"
	"testing"

	core "dappco.re/go"
)

func TestHIPKernelSource_ExportsLaunchABI_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile("../kernels/rocm_kernels.hip")
	core.RequireNoError(t, err)
	source := string(sourceBytes)

	for _, symbol := range []string{
		`extern "C" __global__ void rocm_prefill`,
		`extern "C" __global__ void rocm_decode`,
		`extern "C" __global__ void rocm_projection`,
		`extern "C" __global__ void rocm_mlx_q4_projection`,
		`extern "C" __global__ void rocm_rms_norm`,
		`extern "C" __global__ void rocm_rope`,
		`extern "C" __global__ void rocm_greedy_sample`,
		`extern "C" __global__ void rocm_attention`,
		`extern "C" __global__ void rocm_vector_add`,
		`extern "C" __global__ void rocm_vector_scale`,
		`extern "C" __global__ void rocm_swiglu`,
		`extern "C" __global__ void rocm_moe_router`,
		`extern "C" __global__ void rocm_moe_lazy_experts`,
		`extern "C" __global__ void rocm_jangtq_projection`,
		`extern "C" __global__ void rocm_codebook_lookup`,
		`extern "C" __global__ void rocm_lora_projection`,
		`extern "C" __global__ void rocm_embedding_lookup`,
		`extern "C" __global__ void rocm_embedding_mean_pool`,
		`extern "C" __global__ void rocm_rerank_cosine`,
		`extern "C" __global__ void rocm_tiny_prefill`,
		`extern "C" __global__ void rocm_tiny_decode`,
		`extern "C" __global__ void rocm_cross_entropy_loss`,
		`extern "C" __global__ void rocm_distillation_kl_loss`,
		`extern "C" __global__ void rocm_grpo_advantage`,
	} {
		core.AssertTrue(t, strings.Contains(source, symbol), symbol)
	}

	for _, abi := range []string{
		core.Sprintf("ROCM_PREFILL_LAUNCH_ARGS_VERSION = %d", hipPrefillLaunchArgsVersion),
		core.Sprintf("ROCM_PREFILL_LAUNCH_ARGS_BYTES = %d", hipPrefillLaunchArgsBytes),
		core.Sprintf("ROCM_DECODE_LAUNCH_ARGS_VERSION = %d", hipDecodeLaunchArgsVersion),
		core.Sprintf("ROCM_DECODE_LAUNCH_ARGS_HEADER_BYTES = %d", hipDecodeLaunchArgsHeaderBytes),
		core.Sprintf("ROCM_DECODE_LAUNCH_ARGS_BYTES = %d", hipDecodeLaunchArgsBytes),
		core.Sprintf("ROCM_DEVICE_KV_LAUNCH_DESCRIPTOR_BYTES = %d", rocmDeviceKVLaunchDescriptorBytes),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_VERSION = %d", rocmDeviceKVDescriptorVersion),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES = %d", rocmDeviceKVDescriptorHeaderBytes),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES = %d", rocmDeviceKVDescriptorPageBytes),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16 = %d", rocmDeviceKVDescriptorEncodingFP16),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8 = %d", rocmDeviceKVDescriptorEncodingQ8),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4 = %d", rocmDeviceKVDescriptorEncodingQ4),
		core.Sprintf("ROCM_PROJECTION_LAUNCH_ARGS_VERSION = %d", hipProjectionLaunchArgsVersion),
		core.Sprintf("ROCM_PROJECTION_LAUNCH_ARGS_BYTES = %d", hipProjectionLaunchArgsBytes),
		core.Sprintf("ROCM_PROJECTION_WEIGHT_ENCODING_FP16 = %d", hipProjectionWeightEncodingFP16),
		core.Sprintf("ROCM_PROJECTION_WEIGHT_ENCODING_Q8 = %d", hipProjectionWeightEncodingQ8),
		core.Sprintf("ROCM_PROJECTION_WEIGHT_ENCODING_F32 = %d", hipProjectionWeightEncodingF32),
		core.Sprintf("ROCM_PROJECTION_WEIGHT_ENCODING_BF16 = %d", hipProjectionWeightEncodingBF16),
		core.Sprintf("ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_VERSION = %d", hipMLXQ4ProjectionLaunchArgsVersion),
		core.Sprintf("ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_BYTES = %d", hipMLXQ4ProjectionLaunchArgsBytes),
		core.Sprintf("ROCM_MLX_Q4_PROJECTION_BITS = %d", hipMLXQ4ProjectionBits),
		core.Sprintf("ROCM_RMS_NORM_LAUNCH_ARGS_VERSION = %d", hipRMSNormLaunchArgsVersion),
		core.Sprintf("ROCM_RMS_NORM_LAUNCH_ARGS_BYTES = %d", hipRMSNormLaunchArgsBytes),
		core.Sprintf("ROCM_RMS_NORM_WEIGHT_ENCODING_F32 = %d", hipRMSNormWeightEncodingF32),
		core.Sprintf("ROCM_RMS_NORM_WEIGHT_ENCODING_BF16 = %d", hipRMSNormWeightEncodingBF16),
		core.Sprintf("ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT = %d", hipRMSNormLaunchFlagAddUnitWeight),
		core.Sprintf("ROCM_ROPE_LAUNCH_ARGS_VERSION = %d", hipRoPELaunchArgsVersion),
		core.Sprintf("ROCM_ROPE_LAUNCH_ARGS_BYTES = %d", hipRoPELaunchArgsBytes),
		core.Sprintf("ROCM_GREEDY_LAUNCH_ARGS_VERSION = %d", hipGreedyLaunchArgsVersion),
		core.Sprintf("ROCM_GREEDY_LAUNCH_ARGS_BYTES = %d", hipGreedyLaunchArgsBytes),
		core.Sprintf("ROCM_GREEDY_RESULT_BYTES = %d", hipGreedyResultBytes),
		core.Sprintf("ROCM_ATTENTION_LAUNCH_ARGS_VERSION = %d", hipAttentionLaunchArgsVersion),
		core.Sprintf("ROCM_ATTENTION_LAUNCH_ARGS_BYTES = %d", hipAttentionLaunchArgsBytes),
		core.Sprintf("ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS = %d", hipAttentionKVSourceContiguous),
		core.Sprintf("ROCM_ATTENTION_KV_SOURCE_DEVICE = %d", hipAttentionKVSourceDevice),
		core.Sprintf("ROCM_VECTOR_ADD_LAUNCH_ARGS_VERSION = %d", hipVectorAddLaunchArgsVersion),
		core.Sprintf("ROCM_VECTOR_ADD_LAUNCH_ARGS_BYTES = %d", hipVectorAddLaunchArgsBytes),
		core.Sprintf("ROCM_VECTOR_SCALE_LAUNCH_ARGS_VERSION = %d", hipVectorScaleLaunchArgsVersion),
		core.Sprintf("ROCM_VECTOR_SCALE_LAUNCH_ARGS_BYTES = %d", hipVectorScaleLaunchArgsBytes),
		core.Sprintf("ROCM_SWIGLU_LAUNCH_ARGS_VERSION = %d", hipSwiGLULaunchArgsVersion),
		core.Sprintf("ROCM_SWIGLU_LAUNCH_ARGS_BYTES = %d", hipSwiGLULaunchArgsBytes),
		core.Sprintf("ROCM_MOE_ROUTER_LAUNCH_ARGS_VERSION = %d", hipMoERouterLaunchArgsVersion),
		core.Sprintf("ROCM_MOE_ROUTER_LAUNCH_ARGS_BYTES = %d", hipMoERouterLaunchArgsBytes),
		core.Sprintf("ROCM_MOE_LAZY_LAUNCH_ARGS_VERSION = %d", hipMoELazyLaunchArgsVersion),
		core.Sprintf("ROCM_MOE_LAZY_LAUNCH_ARGS_BYTES = %d", hipMoELazyLaunchArgsBytes),
		core.Sprintf("ROCM_JANGTQ_LAUNCH_ARGS_VERSION = %d", hipJANGTQLaunchArgsVersion),
		core.Sprintf("ROCM_JANGTQ_LAUNCH_ARGS_BYTES = %d", hipJANGTQLaunchArgsBytes),
		core.Sprintf("ROCM_CODEBOOK_LAUNCH_ARGS_VERSION = %d", hipCodebookLaunchArgsVersion),
		core.Sprintf("ROCM_CODEBOOK_LAUNCH_ARGS_BYTES = %d", hipCodebookLaunchArgsBytes),
		core.Sprintf("ROCM_LORA_LAUNCH_ARGS_VERSION = %d", hipLoRALaunchArgsVersion),
		core.Sprintf("ROCM_LORA_LAUNCH_ARGS_BYTES = %d", hipLoRALaunchArgsBytes),
		core.Sprintf("ROCM_EMBEDDING_LOOKUP_LAUNCH_ARGS_VERSION = %d", hipEmbeddingLookupLaunchArgsVersion),
		core.Sprintf("ROCM_EMBEDDING_LOOKUP_LAUNCH_ARGS_BYTES = %d", hipEmbeddingLookupLaunchArgsBytes),
		core.Sprintf("ROCM_EMBEDDING_TABLE_ENCODING_F32 = %d", hipEmbeddingTableEncodingF32),
		core.Sprintf("ROCM_EMBEDDING_TABLE_ENCODING_BF16 = %d", hipEmbeddingTableEncodingBF16),
		core.Sprintf("ROCM_EMBEDDING_TABLE_ENCODING_MLX_Q4 = %d", hipEmbeddingTableEncodingMLXQ4),
		core.Sprintf("ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_VERSION = %d", hipEmbeddingMeanPoolLaunchArgsVersion),
		core.Sprintf("ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_BYTES = %d", hipEmbeddingMeanPoolLaunchArgsBytes),
		core.Sprintf("ROCM_RERANK_COSINE_LAUNCH_ARGS_VERSION = %d", hipRerankCosineLaunchArgsVersion),
		core.Sprintf("ROCM_RERANK_COSINE_LAUNCH_ARGS_BYTES = %d", hipRerankCosineLaunchArgsBytes),
		core.Sprintf("ROCM_TINY_PREFILL_LAUNCH_ARGS_VERSION = %d", hipTinyPrefillLaunchArgsVersion),
		core.Sprintf("ROCM_TINY_PREFILL_LAUNCH_ARGS_BYTES = %d", hipTinyPrefillLaunchArgsBytes),
		core.Sprintf("ROCM_TINY_DECODE_LAUNCH_ARGS_VERSION = %d", hipTinyDecodeLaunchArgsVersion),
		core.Sprintf("ROCM_TINY_DECODE_LAUNCH_ARGS_BYTES = %d", hipTinyDecodeLaunchArgsBytes),
		core.Sprintf("ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP32 = %d", hipTinyOutputWeightEncodingFP32),
		core.Sprintf("ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP16 = %d", hipTinyOutputWeightEncodingFP16),
		core.Sprintf("ROCM_TINY_OUTPUT_WEIGHT_ENCODING_Q8 = %d", hipTinyOutputWeightEncodingQ8),
		core.Sprintf("ROCM_CROSS_ENTROPY_LOSS_LAUNCH_ARGS_VERSION = %d", hipCrossEntropyLossLaunchArgsVersion),
		core.Sprintf("ROCM_CROSS_ENTROPY_LOSS_LAUNCH_ARGS_BYTES = %d", hipCrossEntropyLossLaunchArgsBytes),
		core.Sprintf("ROCM_CROSS_ENTROPY_LOSS_OUTPUT_BYTES = %d", hipCrossEntropyLossOutputBytes),
		core.Sprintf("ROCM_DISTILLATION_KL_LOSS_LAUNCH_ARGS_VERSION = %d", hipDistillationKLLossLaunchArgsVersion),
		core.Sprintf("ROCM_DISTILLATION_KL_LOSS_LAUNCH_ARGS_BYTES = %d", hipDistillationKLLossLaunchArgsBytes),
		core.Sprintf("ROCM_DISTILLATION_KL_LOSS_OUTPUT_BYTES = %d", hipDistillationKLLossOutputBytes),
		core.Sprintf("ROCM_GRPO_ADVANTAGE_LAUNCH_ARGS_VERSION = %d", hipGRPOAdvantageLaunchArgsVersion),
		core.Sprintf("ROCM_GRPO_ADVANTAGE_LAUNCH_ARGS_BYTES = %d", hipGRPOAdvantageLaunchArgsBytes),
		"ROCM_PREFILL_LAUNCH_STATUS_OK",
		"ROCM_DECODE_LAUNCH_STATUS_OK",
		"ROCM_MOE_ROUTER_LAUNCH_STATUS_OK",
	} {
		core.AssertTrue(t, strings.Contains(source, abi), abi)
	}
}
