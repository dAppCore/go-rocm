// SPDX-Licence-Identifier: EUPL-1.2

#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr uint32_t ROCM_PREFILL_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_PREFILL_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_PREFILL_LAUNCH_STATUS_OK = 0x5052464cu;
constexpr uint32_t ROCM_DECODE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_DECODE_LAUNCH_ARGS_HEADER_BYTES = 32;
constexpr uint32_t ROCM_DECODE_LAUNCH_ARGS_BYTES = 96;
constexpr uint32_t ROCM_DEVICE_KV_LAUNCH_DESCRIPTOR_BYTES = 64;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_VERSION = 1;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES = 32;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES = 64;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16 = 1;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8 = 2;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4 = 3;
constexpr uint32_t ROCM_DECODE_LAUNCH_STATUS_OK = 0x4445434fu;
constexpr uint32_t ROCM_PROJECTION_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_PROJECTION_LAUNCH_ARGS_BYTES = 96;
constexpr uint32_t ROCM_PROJECTION_WEIGHT_ENCODING_FP16 = 1;
constexpr uint32_t ROCM_PROJECTION_WEIGHT_ENCODING_Q8 = 2;
constexpr uint32_t ROCM_PROJECTION_WEIGHT_ENCODING_F32 = 3;
constexpr uint32_t ROCM_PROJECTION_WEIGHT_ENCODING_BF16 = 4;
constexpr uint32_t ROCM_PROJECTION_LAUNCH_FLAG_BIAS = 1;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_BYTES = 96;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_BITS = 4;
constexpr uint32_t ROCM_RMS_NORM_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_RMS_NORM_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_RMS_NORM_WEIGHT_ENCODING_F32 = 1;
constexpr uint32_t ROCM_RMS_NORM_WEIGHT_ENCODING_BF16 = 2;
constexpr uint32_t ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT = 1;
constexpr uint32_t ROCM_ROPE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_ROPE_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_GREEDY_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_GREEDY_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_GREEDY_RESULT_BYTES = 8;
constexpr uint32_t ROCM_ATTENTION_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_ATTENTION_LAUNCH_ARGS_BYTES = 104;
constexpr uint32_t ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS = 0;
constexpr uint32_t ROCM_ATTENTION_KV_SOURCE_DEVICE = 1;
constexpr uint32_t ROCM_VECTOR_ADD_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_VECTOR_ADD_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_VECTOR_SCALE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_VECTOR_SCALE_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_SWIGLU_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_SWIGLU_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_MOE_ROUTER_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MOE_ROUTER_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_MOE_ROUTER_LAUNCH_STATUS_OK = 0x4d4f4552u;
constexpr uint32_t ROCM_MOE_LAZY_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MOE_LAZY_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_JANGTQ_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_JANGTQ_LAUNCH_ARGS_BYTES = 96;
constexpr uint32_t ROCM_JANGTQ_LAUNCH_FLAG_BIAS = 1;
constexpr uint32_t ROCM_CODEBOOK_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_CODEBOOK_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_LORA_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_LORA_LAUNCH_ARGS_BYTES = 128;
constexpr uint32_t ROCM_LORA_LAUNCH_FLAG_BIAS = 1;
constexpr uint32_t ROCM_EMBEDDING_LOOKUP_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_EMBEDDING_LOOKUP_LAUNCH_ARGS_BYTES = 96;
constexpr uint32_t ROCM_EMBEDDING_TABLE_ENCODING_F32 = 1;
constexpr uint32_t ROCM_EMBEDDING_TABLE_ENCODING_BF16 = 2;
constexpr uint32_t ROCM_EMBEDDING_TABLE_ENCODING_MLX_Q4 = 3;
constexpr uint32_t ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_EMBEDDING_MEAN_POOL_LAUNCH_FLAG_NORMALIZE = 1;
constexpr uint32_t ROCM_RERANK_COSINE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_RERANK_COSINE_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_TINY_PREFILL_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_TINY_PREFILL_LAUNCH_ARGS_BYTES = 160;
constexpr uint32_t ROCM_TINY_DECODE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_TINY_DECODE_LAUNCH_ARGS_BYTES = 160;
constexpr uint32_t ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP32 = 1;
constexpr uint32_t ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP16 = 2;
constexpr uint32_t ROCM_TINY_OUTPUT_WEIGHT_ENCODING_Q8 = 3;
constexpr uint32_t ROCM_CROSS_ENTROPY_LOSS_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_CROSS_ENTROPY_LOSS_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_CROSS_ENTROPY_LOSS_OUTPUT_BYTES = 16;
constexpr uint32_t ROCM_DISTILLATION_KL_LOSS_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_DISTILLATION_KL_LOSS_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_DISTILLATION_KL_LOSS_OUTPUT_BYTES = 8;
constexpr uint32_t ROCM_GRPO_ADVANTAGE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_GRPO_ADVANTAGE_LAUNCH_ARGS_BYTES = 64;

struct rocm_prefill_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t token_pointer;
	uint64_t token_count;
	uint64_t token_bytes;
	uint32_t mode_code;
	uint32_t block_size;
	uint32_t key_width;
	uint32_t value_width;
	uint64_t status_pointer;
	uint32_t status_value;
	uint32_t reserved;
};

struct rocm_device_kv_launch_descriptor {
	uint64_t descriptor_pointer;
	uint64_t descriptor_bytes;
	uint32_t descriptor_version;
	uint32_t mode_code;
	uint32_t block_size;
	uint32_t page_count;
	uint64_t token_count;
	uint32_t key_width;
	uint32_t value_width;
	uint64_t status_pointer;
	uint32_t status_value;
	uint32_t reserved;
};

struct rocm_device_kv_descriptor_header {
	uint32_t version;
	uint32_t header_bytes;
	uint32_t page_bytes;
	uint32_t mode_code;
	uint32_t page_count;
	uint32_t block_size;
	uint64_t token_count;
};

struct rocm_device_kv_page_descriptor {
	uint64_t token_start;
	uint64_t token_count;
	uint32_t key_width;
	uint32_t value_width;
	uint32_t key_encoding;
	uint32_t value_encoding;
	uint64_t key_pointer;
	uint64_t value_pointer;
	uint64_t key_bytes;
	uint64_t value_bytes;
};

struct rocm_decode_launch_args {
	uint32_t version;
	uint32_t header_bytes;
	uint32_t total_bytes;
	uint32_t token_id;
	uint64_t position;
	uint32_t kv_descriptor_bytes;
	uint32_t reserved;
	rocm_device_kv_launch_descriptor kv;
};

struct rocm_projection_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint32_t input_count;
	uint32_t input_bytes;
	uint64_t weight_pointer;
	uint64_t weight_bytes;
	uint64_t bias_pointer;
	uint64_t bias_bytes;
	uint64_t output_pointer;
	uint64_t output_bytes;
	uint32_t rows;
	uint32_t cols;
	uint32_t weight_encoding;
	uint32_t flags;
	uint32_t q8_scale_bits;
	uint32_t reserved;
};

struct rocm_mlx_q4_projection_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t weight_pointer;
	uint64_t scale_pointer;
	uint64_t bias_pointer;
	uint64_t output_pointer;
	uint32_t rows;
	uint32_t cols;
	uint32_t group_size;
	uint32_t bits;
	uint32_t input_bytes;
	uint32_t weight_bytes;
	uint32_t scale_bytes;
	uint32_t bias_bytes;
	uint32_t output_bytes;
	uint32_t reserved0;
	uint64_t reserved1;
};

struct rocm_lora_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t base_weight_pointer;
	uint64_t lora_a_pointer;
	uint64_t lora_b_pointer;
	uint64_t bias_pointer;
	uint64_t output_pointer;
	uint32_t input_count;
	uint32_t rows;
	uint32_t cols;
	uint32_t rank;
	uint32_t input_bytes;
	uint32_t base_weight_bytes;
	uint32_t lora_a_bytes;
	uint32_t lora_b_bytes;
	uint32_t bias_bytes;
	uint32_t output_bytes;
	uint32_t alpha_bits;
	uint32_t flags;
	uint64_t reserved1;
	uint64_t reserved2;
	uint64_t reserved3;
};

struct rocm_embedding_lookup_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t token_pointer;
	uint64_t embedding_pointer;
	uint64_t output_pointer;
	uint32_t token_count;
	uint32_t vocab_size;
	uint32_t hidden_size;
	uint32_t token_bytes;
	uint64_t embedding_bytes;
	uint64_t output_bytes;
	uint32_t table_encoding;
	uint32_t group_size;
	uint64_t scale_pointer;
	uint64_t bias_pointer;
	uint32_t scale_bytes;
	uint32_t bias_bytes;
};

struct rocm_embedding_mean_pool_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t token_pointer;
	uint64_t output_pointer;
	uint32_t token_count;
	uint32_t dim;
	uint32_t token_bytes;
	uint32_t output_bytes;
	uint32_t flags;
	uint32_t reserved;
	uint64_t reserved2;
	uint64_t reserved3;
};

struct rocm_rerank_cosine_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t query_pointer;
	uint64_t document_pointer;
	uint64_t output_pointer;
	uint32_t document_count;
	uint32_t dim;
	uint32_t query_bytes;
	uint32_t document_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
	uint64_t reserved2;
};

struct rocm_rms_norm_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t weight_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t input_bytes;
	uint32_t weight_bytes;
	uint32_t output_bytes;
	uint32_t epsilon_bits;
	uint32_t weight_encoding;
	uint32_t flags;
	uint32_t reserved;
};

struct rocm_rope_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t input_bytes;
	uint32_t output_bytes;
	uint32_t position;
	uint32_t base_bits;
	uint32_t frequency_dim;
	uint64_t reserved2;
	uint64_t reserved3;
};

struct rocm_greedy_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t logits_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t logits_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
	uint64_t reserved2;
	uint64_t reserved3;
	uint64_t reserved4;
};

struct rocm_attention_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t query_pointer;
	uint64_t key_pointer;
	uint64_t value_pointer;
	uint64_t output_pointer;
	uint64_t weight_pointer;
	uint32_t dim;
	uint32_t token_count;
	uint32_t query_bytes;
	uint32_t key_bytes;
	uint32_t value_bytes;
	uint32_t output_bytes;
	uint32_t weight_bytes;
	uint32_t kv_source;
	uint32_t scale_bits;
	uint64_t descriptor_pointer;
	uint64_t descriptor_bytes;
};

struct rocm_vector_add_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t left_pointer;
	uint64_t right_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t left_bytes;
	uint32_t right_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
	uint64_t reserved2;
};

struct rocm_vector_scale_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t input_bytes;
	uint32_t output_bytes;
	uint32_t scale_bits;
	uint64_t reserved2;
	uint64_t reserved3;
	uint64_t reserved4;
};

struct rocm_swiglu_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t gate_pointer;
	uint64_t up_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t gate_bytes;
	uint32_t up_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
	uint64_t reserved2;
};

struct rocm_moe_router_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t logit_pointer;
	uint64_t id_pointer;
	uint64_t prob_pointer;
	uint32_t expert_count;
	uint32_t top_k;
	uint32_t logit_bytes;
	uint32_t id_bytes;
	uint32_t prob_bytes;
	uint32_t layer;
	uint64_t status_pointer;
};

struct rocm_moe_lazy_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t id_pointer;
	uint64_t resident_pointer;
	uint32_t selected_count;
	uint32_t expert_count;
	uint32_t id_bytes;
	uint32_t resident_bytes;
	uint64_t reserved;
	uint64_t reserved2;
	uint64_t reserved3;
};

struct rocm_jangtq_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t packed_pointer;
	uint64_t bias_pointer;
	uint64_t output_pointer;
	uint32_t input_count;
	uint32_t rows;
	uint32_t cols;
	uint32_t bits;
	uint32_t group_size;
	uint32_t input_bytes;
	uint32_t packed_bytes;
	uint32_t bias_bytes;
	uint32_t output_bytes;
	uint32_t scale_bits;
	uint32_t flags;
	uint64_t reserved;
};

struct rocm_codebook_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t code_pointer;
	uint64_t codebook_pointer;
	uint64_t output_pointer;
	uint32_t code_count;
	uint32_t codebook_count;
	uint32_t code_dim;
	uint32_t code_bytes;
	uint32_t codebook_bytes;
	uint32_t output_bytes;
	uint64_t reserved;
};

struct rocm_tiny_prefill_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t token_pointer;
	uint64_t embedding_pointer;
	uint64_t output_weight_pointer;
	uint64_t logit_pointer;
	uint64_t attention_pointer;
	uint64_t result_pointer;
	uint64_t key_pointer;
	uint64_t value_pointer;
	uint32_t token_count;
	uint32_t vocab_size;
	uint32_t hidden_size;
	uint32_t token_bytes;
	uint32_t embedding_bytes;
	uint32_t output_weight_bytes;
	uint32_t logit_bytes;
	uint32_t attention_bytes;
	uint32_t result_bytes;
	uint32_t key_bytes;
	uint32_t value_bytes;
	uint32_t output_weight_encoding;
	uint32_t q8_scale_bits;
	uint32_t reserved;
	uint64_t reserved2;
	uint64_t reserved3;
	uint64_t reserved4;
	uint64_t reserved5;
};

struct rocm_tiny_decode_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t prior_key_pointer;
	uint64_t prior_value_pointer;
	uint64_t embedding_pointer;
	uint64_t output_weight_pointer;
	uint64_t logit_pointer;
	uint64_t attention_pointer;
	uint64_t updated_key_pointer;
	uint64_t updated_value_pointer;
	uint64_t result_pointer;
	uint32_t token_id;
	uint32_t prior_token_count;
	uint32_t vocab_size;
	uint32_t hidden_size;
	uint32_t prior_key_bytes;
	uint32_t prior_value_bytes;
	uint32_t embedding_bytes;
	uint32_t output_weight_bytes;
	uint32_t logit_bytes;
	uint32_t attention_bytes;
	uint32_t updated_key_bytes;
	uint32_t updated_value_bytes;
	uint32_t result_bytes;
	uint32_t output_weight_encoding;
	uint32_t q8_scale_bits;
	uint32_t reserved;
	uint64_t reserved2;
	uint64_t reserved3;
};

struct rocm_cross_entropy_loss_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t logit_pointer;
	uint64_t target_pointer;
	uint64_t output_pointer;
	uint32_t batch;
	uint32_t vocab;
	uint32_t logit_bytes;
	uint32_t target_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
	uint64_t reserved2;
};

struct rocm_distillation_kl_loss_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t student_pointer;
	uint64_t teacher_pointer;
	uint64_t output_pointer;
	uint32_t batch;
	uint32_t vocab;
	uint32_t student_bytes;
	uint32_t teacher_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
	uint64_t temperature_bits;
};

struct rocm_grpo_advantage_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t reward_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t reward_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
	uint64_t reserved2;
	uint64_t reserved3;
	uint64_t reserved4;
};

static_assert(sizeof(rocm_prefill_launch_args) == ROCM_PREFILL_LAUNCH_ARGS_BYTES, "prefill launch ABI drift");
static_assert(sizeof(rocm_device_kv_launch_descriptor) == ROCM_DEVICE_KV_LAUNCH_DESCRIPTOR_BYTES, "KV launch ABI drift");
static_assert(sizeof(rocm_device_kv_descriptor_header) == ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES, "KV descriptor header ABI drift");
static_assert(sizeof(rocm_device_kv_page_descriptor) == ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES, "KV descriptor page ABI drift");
static_assert(sizeof(rocm_decode_launch_args) == ROCM_DECODE_LAUNCH_ARGS_BYTES, "decode launch ABI drift");
static_assert(sizeof(rocm_projection_launch_args) == ROCM_PROJECTION_LAUNCH_ARGS_BYTES, "projection launch ABI drift");
static_assert(sizeof(rocm_mlx_q4_projection_launch_args) == ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_BYTES, "MLX q4 projection launch ABI drift");
static_assert(sizeof(rocm_rms_norm_launch_args) == ROCM_RMS_NORM_LAUNCH_ARGS_BYTES, "RMSNorm launch ABI drift");
static_assert(sizeof(rocm_rope_launch_args) == ROCM_ROPE_LAUNCH_ARGS_BYTES, "RoPE launch ABI drift");
static_assert(sizeof(rocm_greedy_launch_args) == ROCM_GREEDY_LAUNCH_ARGS_BYTES, "greedy launch ABI drift");
static_assert(sizeof(rocm_attention_launch_args) == ROCM_ATTENTION_LAUNCH_ARGS_BYTES, "attention launch ABI drift");
static_assert(sizeof(rocm_vector_add_launch_args) == ROCM_VECTOR_ADD_LAUNCH_ARGS_BYTES, "vector add launch ABI drift");
static_assert(sizeof(rocm_vector_scale_launch_args) == ROCM_VECTOR_SCALE_LAUNCH_ARGS_BYTES, "vector scale launch ABI drift");
static_assert(sizeof(rocm_swiglu_launch_args) == ROCM_SWIGLU_LAUNCH_ARGS_BYTES, "SwiGLU launch ABI drift");
static_assert(sizeof(rocm_moe_router_launch_args) == ROCM_MOE_ROUTER_LAUNCH_ARGS_BYTES, "MoE router launch ABI drift");
static_assert(sizeof(rocm_moe_lazy_launch_args) == ROCM_MOE_LAZY_LAUNCH_ARGS_BYTES, "MoE lazy expert launch ABI drift");
static_assert(sizeof(rocm_jangtq_launch_args) == ROCM_JANGTQ_LAUNCH_ARGS_BYTES, "JANGTQ launch ABI drift");
static_assert(sizeof(rocm_codebook_launch_args) == ROCM_CODEBOOK_LAUNCH_ARGS_BYTES, "codebook launch ABI drift");
static_assert(sizeof(rocm_lora_launch_args) == ROCM_LORA_LAUNCH_ARGS_BYTES, "LoRA launch ABI drift");
static_assert(sizeof(rocm_embedding_lookup_launch_args) == ROCM_EMBEDDING_LOOKUP_LAUNCH_ARGS_BYTES, "embedding lookup launch ABI drift");
static_assert(sizeof(rocm_embedding_mean_pool_launch_args) == ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_BYTES, "embedding mean-pool launch ABI drift");
static_assert(sizeof(rocm_rerank_cosine_launch_args) == ROCM_RERANK_COSINE_LAUNCH_ARGS_BYTES, "rerank cosine launch ABI drift");
static_assert(sizeof(rocm_tiny_prefill_launch_args) == ROCM_TINY_PREFILL_LAUNCH_ARGS_BYTES, "tiny prefill launch ABI drift");
static_assert(sizeof(rocm_tiny_decode_launch_args) == ROCM_TINY_DECODE_LAUNCH_ARGS_BYTES, "tiny decode launch ABI drift");
static_assert(sizeof(rocm_cross_entropy_loss_launch_args) == ROCM_CROSS_ENTROPY_LOSS_LAUNCH_ARGS_BYTES, "cross entropy loss launch ABI drift");
static_assert(sizeof(rocm_distillation_kl_loss_launch_args) == ROCM_DISTILLATION_KL_LOSS_LAUNCH_ARGS_BYTES, "distillation KL loss launch ABI drift");
static_assert(sizeof(rocm_grpo_advantage_launch_args) == ROCM_GRPO_ADVANTAGE_LAUNCH_ARGS_BYTES, "GRPO advantage launch ABI drift");

__device__ float rocm_float_from_bits(uint32_t bits)
{
	union {
		uint32_t u;
		float f;
	} value;
	value.u = bits;
	return value.f;
}

__device__ double rocm_double_from_bits(uint64_t bits)
{
	union {
		uint64_t u;
		double f;
	} value;
	value.u = bits;
	return value.f;
}

__device__ float rocm_half_to_float(uint16_t half)
{
	const uint32_t sign = (static_cast<uint32_t>(half & 0x8000u)) << 16;
	uint32_t exponent = (half >> 10) & 0x1fu;
	uint32_t mantissa = half & 0x03ffu;

	if (exponent == 0) {
		if (mantissa == 0) {
			return rocm_float_from_bits(sign);
		}
		while ((mantissa & 0x0400u) == 0) {
			mantissa <<= 1;
			--exponent;
		}
		++exponent;
		mantissa &= 0x03ffu;
	} else if (exponent == 31) {
		return rocm_float_from_bits(sign | 0x7f800000u | (mantissa << 13));
	}

	exponent = exponent + (127u - 15u);
	return rocm_float_from_bits(sign | (exponent << 23) | (mantissa << 13));
}

__device__ float rocm_bfloat16_to_float(uint16_t value)
{
	return rocm_float_from_bits(static_cast<uint32_t>(value) << 16);
}

__device__ bool rocm_supported_kv_mode(uint32_t mode_code)
{
	return mode_code >= 1 && mode_code <= 3;
}

__device__ bool rocm_positive_finite(float value)
{
	return value > 0.0f && isfinite(value);
}

__device__ bool rocm_valid_prefill_args(const rocm_prefill_launch_args &args)
{
	return args.version == ROCM_PREFILL_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_PREFILL_LAUNCH_ARGS_BYTES &&
		args.token_pointer != 0 &&
		args.token_count > 0 &&
		args.token_bytes == args.token_count * sizeof(int32_t) &&
		rocm_supported_kv_mode(args.mode_code) &&
		args.block_size > 0 &&
		args.key_width > 0 &&
		args.value_width > 0;
}

__device__ bool rocm_valid_decode_args(const rocm_decode_launch_args &args)
{
	return args.version == ROCM_DECODE_LAUNCH_ARGS_VERSION &&
		args.header_bytes == ROCM_DECODE_LAUNCH_ARGS_HEADER_BYTES &&
		args.total_bytes == ROCM_DECODE_LAUNCH_ARGS_BYTES &&
		args.kv_descriptor_bytes == ROCM_DEVICE_KV_LAUNCH_DESCRIPTOR_BYTES &&
		args.kv.descriptor_pointer != 0 &&
		args.kv.descriptor_bytes >= 32 &&
		args.kv.descriptor_version == ROCM_DEVICE_KV_DESCRIPTOR_VERSION &&
		rocm_supported_kv_mode(args.kv.mode_code) &&
		args.kv.block_size > 0 &&
		args.kv.page_count > 0 &&
		args.kv.token_count > 0 &&
		args.position == args.kv.token_count &&
		args.kv.key_width > 0 &&
		args.kv.value_width > 0;
}

__device__ bool rocm_valid_projection_args(const rocm_projection_launch_args &args)
{
	if (args.version != ROCM_PROJECTION_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_PROJECTION_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.input_count != args.cols ||
		args.input_bytes != args.cols * sizeof(float) ||
		args.output_bytes != args.rows * sizeof(float)) {
		return false;
	}
	if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_FP16 ||
		args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_BF16) {
		return args.weight_bytes == static_cast<uint64_t>(args.rows) * args.cols * sizeof(uint16_t);
	}
	if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_Q8) {
		const float scale = rocm_float_from_bits(args.q8_scale_bits);
		return args.weight_bytes == static_cast<uint64_t>(args.rows) * args.cols &&
			rocm_positive_finite(scale);
	}
	if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_F32) {
		return args.weight_bytes == static_cast<uint64_t>(args.rows) * args.cols * sizeof(float);
	}
	return false;
}

__device__ bool rocm_valid_mlx_q4_projection_args(const rocm_mlx_q4_projection_launch_args &args)
{
	if (args.version != ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.scale_pointer == 0 ||
		args.bias_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.group_size == 0 ||
		args.bits != ROCM_MLX_Q4_PROJECTION_BITS ||
		(args.cols % 8) != 0 ||
		(args.cols % args.group_size) != 0 ||
		args.input_bytes != args.cols * sizeof(float) ||
		args.output_bytes != args.rows * sizeof(float)) {
		return false;
	}
	const uint32_t packed_per_row = args.cols / 8;
	const uint32_t groups_per_row = args.cols / args.group_size;
	return args.weight_bytes == args.rows * packed_per_row * sizeof(uint32_t) &&
		args.scale_bytes == args.rows * groups_per_row * sizeof(uint16_t) &&
		args.bias_bytes == args.rows * groups_per_row * sizeof(uint16_t);
}

__device__ bool rocm_valid_lora_args(const rocm_lora_launch_args &args)
{
	const bool has_bias = (args.flags & ROCM_LORA_LAUNCH_FLAG_BIAS) != 0;
	bool valid_bias = args.bias_pointer == 0 && args.bias_bytes == 0;
	if (has_bias) {
		valid_bias = args.bias_pointer != 0 && args.bias_bytes == args.rows * sizeof(float);
	}
	return args.version == ROCM_LORA_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_LORA_LAUNCH_ARGS_BYTES &&
		args.input_pointer != 0 &&
		args.base_weight_pointer != 0 &&
		args.lora_a_pointer != 0 &&
		args.lora_b_pointer != 0 &&
		args.output_pointer != 0 &&
		args.rows > 0 &&
		args.cols > 0 &&
		args.rank > 0 &&
		args.input_count == args.cols &&
		args.input_bytes == args.cols * sizeof(float) &&
		args.base_weight_bytes == args.rows * args.cols * sizeof(float) &&
		args.lora_a_bytes == args.rank * args.cols * sizeof(float) &&
		args.lora_b_bytes == args.rows * args.rank * sizeof(float) &&
		args.output_bytes == args.rows * sizeof(float) &&
		rocm_positive_finite(rocm_float_from_bits(args.alpha_bits)) &&
		valid_bias;
}

__device__ bool rocm_valid_embedding_mean_pool_args(const rocm_embedding_mean_pool_launch_args &args)
{
	return args.version == ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_BYTES &&
		args.token_pointer != 0 &&
		args.output_pointer != 0 &&
		args.token_count > 0 &&
		args.dim > 0 &&
		args.token_bytes == args.token_count * args.dim * sizeof(float) &&
		args.output_bytes == args.dim * sizeof(float);
}

__device__ bool rocm_valid_embedding_lookup_args(const rocm_embedding_lookup_launch_args &args)
{
	if (args.version != ROCM_EMBEDDING_LOOKUP_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_EMBEDDING_LOOKUP_LAUNCH_ARGS_BYTES ||
		args.token_pointer == 0 ||
		args.embedding_pointer == 0 ||
		args.output_pointer == 0 ||
		args.token_count == 0 ||
		args.vocab_size == 0 ||
		args.hidden_size == 0 ||
		args.token_bytes != args.token_count * sizeof(int32_t) ||
		args.output_bytes != static_cast<uint64_t>(args.token_count) * args.hidden_size * sizeof(float)) {
		return false;
	}
	if (args.table_encoding == ROCM_EMBEDDING_TABLE_ENCODING_F32) {
		return args.embedding_bytes == static_cast<uint64_t>(args.vocab_size) * args.hidden_size * sizeof(float);
	}
	if (args.table_encoding == ROCM_EMBEDDING_TABLE_ENCODING_BF16) {
		return args.embedding_bytes == static_cast<uint64_t>(args.vocab_size) * args.hidden_size * sizeof(uint16_t);
	}
	if (args.table_encoding == ROCM_EMBEDDING_TABLE_ENCODING_MLX_Q4) {
		return args.scale_pointer != 0 &&
			args.bias_pointer != 0 &&
			args.group_size > 0 &&
			(args.hidden_size % 8u) == 0 &&
			(args.hidden_size % args.group_size) == 0 &&
			args.embedding_bytes == static_cast<uint64_t>(args.vocab_size) * (args.hidden_size / 8u) * sizeof(uint32_t) &&
			args.scale_bytes == static_cast<uint64_t>(args.vocab_size) * (args.hidden_size / args.group_size) * sizeof(uint16_t) &&
			args.bias_bytes == static_cast<uint64_t>(args.vocab_size) * (args.hidden_size / args.group_size) * sizeof(uint16_t);
	}
	return false;
}

__device__ bool rocm_valid_rerank_cosine_args(const rocm_rerank_cosine_launch_args &args)
{
	return args.version == ROCM_RERANK_COSINE_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_RERANK_COSINE_LAUNCH_ARGS_BYTES &&
		args.query_pointer != 0 &&
		args.document_pointer != 0 &&
		args.output_pointer != 0 &&
		args.document_count > 0 &&
		args.dim > 0 &&
		args.query_bytes == args.dim * sizeof(float) &&
		args.document_bytes == args.document_count * args.dim * sizeof(float) &&
		args.output_bytes == args.document_count * sizeof(float);
}

__device__ bool rocm_valid_rms_norm_args(const rocm_rms_norm_launch_args &args)
{
	const float epsilon = rocm_float_from_bits(args.epsilon_bits);
	if (args.version != ROCM_RMS_NORM_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_RMS_NORM_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.output_pointer == 0 ||
		args.count == 0 ||
		args.input_bytes != args.count * sizeof(float) ||
		args.output_bytes != args.count * sizeof(float) ||
		!(epsilon >= 0.0f) ||
		(args.flags & ~ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) != 0) {
		return false;
	}
	if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_F32) {
		return args.weight_bytes == args.count * sizeof(float);
	}
	if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_BF16) {
		return args.weight_bytes == args.count * sizeof(uint16_t);
	}
	return false;
}

__device__ bool rocm_valid_rope_args(const rocm_rope_launch_args &args)
{
	return args.version == ROCM_ROPE_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_ROPE_LAUNCH_ARGS_BYTES &&
		args.input_pointer != 0 &&
		args.output_pointer != 0 &&
		args.count > 0 &&
		(args.count % 2) == 0 &&
		args.input_bytes == args.count * sizeof(float) &&
		args.output_bytes == args.count * sizeof(float) &&
		rocm_float_from_bits(args.base_bits) > 0.0f &&
		(args.frequency_dim == 0 || args.frequency_dim >= args.count);
}

__device__ bool rocm_valid_greedy_args(const rocm_greedy_launch_args &args)
{
	return args.version == ROCM_GREEDY_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_GREEDY_LAUNCH_ARGS_BYTES &&
		args.logits_pointer != 0 &&
		args.output_pointer != 0 &&
		args.count > 0 &&
		args.logits_bytes == args.count * sizeof(float) &&
		args.output_bytes == ROCM_GREEDY_RESULT_BYTES;
}

__device__ uint64_t rocm_device_kv_tensor_bytes(uint32_t encoding, uint64_t length);

__device__ bool rocm_valid_device_kv_attention_descriptor(const rocm_attention_launch_args &args)
{
	if (args.kv_source != ROCM_ATTENTION_KV_SOURCE_DEVICE ||
		args.descriptor_pointer == 0 ||
		args.descriptor_bytes < ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES ||
		args.key_pointer != 0 ||
		args.value_pointer != 0 ||
		args.key_bytes != 0 ||
		args.value_bytes != 0) {
		return false;
	}
	const rocm_device_kv_descriptor_header *header = reinterpret_cast<const rocm_device_kv_descriptor_header *>(static_cast<uintptr_t>(args.descriptor_pointer));
	if (header->version != ROCM_DEVICE_KV_DESCRIPTOR_VERSION ||
		header->header_bytes != ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES ||
		header->page_bytes != ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES ||
		header->page_count == 0 ||
		header->token_count != args.token_count ||
		args.descriptor_bytes != static_cast<uint64_t>(ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES) + static_cast<uint64_t>(header->page_count) * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES) {
		return false;
	}
	const unsigned char *base = reinterpret_cast<const unsigned char *>(header);
	for (uint32_t page_index = 0; page_index < header->page_count; ++page_index) {
		const rocm_device_kv_page_descriptor *page = reinterpret_cast<const rocm_device_kv_page_descriptor *>(base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + page_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
		if (page->token_count == 0 ||
			page->token_start + page->token_count > args.token_count ||
			page->key_width != args.dim ||
			page->value_width != args.dim ||
			page->key_pointer == 0 ||
			page->value_pointer == 0 ||
			rocm_device_kv_tensor_bytes(page->key_encoding, static_cast<uint64_t>(page->token_count) * page->key_width) == 0 ||
			rocm_device_kv_tensor_bytes(page->value_encoding, static_cast<uint64_t>(page->token_count) * page->value_width) == 0 ||
			page->key_bytes != rocm_device_kv_tensor_bytes(page->key_encoding, static_cast<uint64_t>(page->token_count) * page->key_width) ||
			page->value_bytes != rocm_device_kv_tensor_bytes(page->value_encoding, static_cast<uint64_t>(page->token_count) * page->value_width)) {
			return false;
		}
	}
	return true;
}

__device__ uint64_t rocm_device_kv_tensor_bytes(uint32_t encoding, uint64_t length)
{
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16) {
		return length * sizeof(uint16_t);
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8) {
		return sizeof(uint32_t) + length * sizeof(int8_t);
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4) {
		return sizeof(uint32_t) + ((length + 1u) / 2u);
	}
	return 0;
}

__device__ bool rocm_valid_attention_args(const rocm_attention_launch_args &args)
{
	const float scale = rocm_float_from_bits(args.scale_bits);
	if (args.version != ROCM_ATTENTION_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_ATTENTION_LAUNCH_ARGS_BYTES ||
		args.query_pointer == 0 ||
		args.output_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.dim == 0 ||
		args.token_count == 0 ||
		args.query_bytes != args.dim * sizeof(float) ||
		args.output_bytes != args.dim * sizeof(float) ||
		args.weight_bytes != args.token_count * sizeof(float) ||
		!(scale >= 0.0f) ||
		!isfinite(scale)) {
		return false;
	}
	if (args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS) {
		return args.key_pointer != 0 &&
			args.value_pointer != 0 &&
			args.key_bytes == args.dim * args.token_count * sizeof(float) &&
			args.value_bytes == args.dim * args.token_count * sizeof(float) &&
			args.descriptor_pointer == 0 &&
			args.descriptor_bytes == 0;
	}
	return rocm_valid_device_kv_attention_descriptor(args);
}

__device__ bool rocm_valid_vector_add_args(const rocm_vector_add_launch_args &args)
{
	return args.version == ROCM_VECTOR_ADD_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_VECTOR_ADD_LAUNCH_ARGS_BYTES &&
		args.left_pointer != 0 &&
		args.right_pointer != 0 &&
		args.output_pointer != 0 &&
		args.count > 0 &&
		args.left_bytes == args.count * sizeof(float) &&
		args.right_bytes == args.count * sizeof(float) &&
		args.output_bytes == args.count * sizeof(float);
}

__device__ bool rocm_valid_vector_scale_args(const rocm_vector_scale_launch_args &args)
{
	return args.version == ROCM_VECTOR_SCALE_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_VECTOR_SCALE_LAUNCH_ARGS_BYTES &&
		args.input_pointer != 0 &&
		args.output_pointer != 0 &&
		args.count > 0 &&
		args.input_bytes == args.count * sizeof(float) &&
		args.output_bytes == args.count * sizeof(float) &&
		isfinite(rocm_float_from_bits(args.scale_bits));
}

__device__ bool rocm_valid_swiglu_args(const rocm_swiglu_launch_args &args)
{
	return args.version == ROCM_SWIGLU_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_SWIGLU_LAUNCH_ARGS_BYTES &&
		args.gate_pointer != 0 &&
		args.up_pointer != 0 &&
		args.output_pointer != 0 &&
		args.count > 0 &&
		args.gate_bytes == args.count * sizeof(float) &&
		args.up_bytes == args.count * sizeof(float) &&
		args.output_bytes == args.count * sizeof(float);
}

__device__ bool rocm_valid_moe_router_args(const rocm_moe_router_launch_args &args)
{
	return args.version == ROCM_MOE_ROUTER_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_MOE_ROUTER_LAUNCH_ARGS_BYTES &&
		args.logit_pointer != 0 &&
		args.id_pointer != 0 &&
		args.prob_pointer != 0 &&
		args.expert_count > 0 &&
		args.top_k > 0 &&
		args.top_k <= args.expert_count &&
		args.logit_bytes == args.expert_count * sizeof(float) &&
		args.id_bytes == args.top_k * sizeof(int32_t) &&
		args.prob_bytes == args.top_k * sizeof(float);
}

__device__ bool rocm_valid_moe_lazy_args(const rocm_moe_lazy_launch_args &args)
{
	return args.version == ROCM_MOE_LAZY_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_MOE_LAZY_LAUNCH_ARGS_BYTES &&
		args.id_pointer != 0 &&
		args.resident_pointer != 0 &&
		args.selected_count > 0 &&
		args.expert_count > 0 &&
		args.id_bytes == args.selected_count * sizeof(int32_t) &&
		args.resident_bytes == args.expert_count * sizeof(uint8_t);
}

__device__ bool rocm_valid_jangtq_args(const rocm_jangtq_launch_args &args)
{
	bool valid_bits = args.bits == 2 || args.bits == 4 || args.bits == 8;
	const uint64_t required_packed = (static_cast<uint64_t>(args.bits) * args.rows * args.cols + 7u) / 8u;
	const bool has_bias = (args.flags & ROCM_JANGTQ_LAUNCH_FLAG_BIAS) != 0;
	bool valid_bias = args.bias_pointer == 0 && args.bias_bytes == 0;
	if (has_bias) {
		valid_bias = args.bias_pointer != 0 && args.bias_bytes == args.rows * sizeof(float);
	}
	return args.version == ROCM_JANGTQ_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_JANGTQ_LAUNCH_ARGS_BYTES &&
		args.input_pointer != 0 &&
		args.packed_pointer != 0 &&
		args.output_pointer != 0 &&
		args.input_count == args.cols &&
		args.rows > 0 &&
		args.cols > 0 &&
		valid_bits &&
		args.group_size > 0 &&
		(args.group_size & (args.group_size - 1u)) == 0 &&
		args.input_bytes == args.cols * sizeof(float) &&
		args.packed_bytes >= required_packed &&
		args.output_bytes == args.rows * sizeof(float) &&
		rocm_positive_finite(rocm_float_from_bits(args.scale_bits)) &&
		valid_bias;
}

__device__ int8_t rocm_unpack_signed_bits(const uint8_t *packed, uint32_t bits, uint32_t index)
{
	const uint32_t bit_offset = index * bits;
	const uint32_t byte_index = bit_offset / 8u;
	const uint32_t shift = bit_offset % 8u;
	uint32_t raw = static_cast<uint32_t>(packed[byte_index] >> shift);
	if (shift + bits > 8u) {
		raw |= static_cast<uint32_t>(packed[byte_index + 1u]) << (8u - shift);
	}
	const uint32_t mask = (1u << bits) - 1u;
	const uint32_t sign_bit = 1u << (bits - 1u);
	raw &= mask;
	int32_t value = static_cast<int32_t>(raw);
	if ((raw & sign_bit) != 0) {
		value -= static_cast<int32_t>(1u << bits);
	}
	return static_cast<int8_t>(value);
}

__device__ bool rocm_valid_codebook_args(const rocm_codebook_launch_args &args)
{
	return args.version == ROCM_CODEBOOK_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_CODEBOOK_LAUNCH_ARGS_BYTES &&
		args.code_pointer != 0 &&
		args.codebook_pointer != 0 &&
		args.output_pointer != 0 &&
		args.code_count > 0 &&
		args.codebook_count > 0 &&
		args.code_dim > 0 &&
		args.code_bytes == args.code_count * sizeof(uint8_t) &&
		args.codebook_bytes == args.codebook_count * args.code_dim * sizeof(float) &&
		args.output_bytes == args.code_count * args.code_dim * sizeof(float);
}

__device__ bool rocm_valid_tiny_prefill_args(const rocm_tiny_prefill_launch_args &args)
{
	const uint32_t table_count = args.vocab_size * args.hidden_size;
	bool valid_output_weights = false;
	if (args.output_weight_encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP32) {
		valid_output_weights = args.output_weight_bytes == table_count * sizeof(float);
	} else if (args.output_weight_encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP16) {
		valid_output_weights = args.output_weight_bytes == table_count * sizeof(uint16_t);
	} else if (args.output_weight_encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_Q8) {
		const float scale = rocm_float_from_bits(args.q8_scale_bits);
		valid_output_weights = args.output_weight_bytes == table_count &&
			rocm_positive_finite(scale);
	}
	return args.version == ROCM_TINY_PREFILL_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_TINY_PREFILL_LAUNCH_ARGS_BYTES &&
		args.token_pointer != 0 &&
		args.embedding_pointer != 0 &&
		args.output_weight_pointer != 0 &&
		args.logit_pointer != 0 &&
		args.attention_pointer != 0 &&
		args.result_pointer != 0 &&
		args.key_pointer != 0 &&
		args.value_pointer != 0 &&
		args.token_count > 0 &&
		args.vocab_size > 0 &&
		args.hidden_size > 0 &&
		args.token_bytes == args.token_count * sizeof(int32_t) &&
		args.embedding_bytes == args.vocab_size * args.hidden_size * sizeof(float) &&
		valid_output_weights &&
		args.logit_bytes == args.vocab_size * sizeof(float) &&
		args.attention_bytes == args.token_count * sizeof(float) &&
		args.result_bytes == ROCM_GREEDY_RESULT_BYTES &&
		args.key_bytes == args.token_count * args.hidden_size * sizeof(float) &&
		args.value_bytes == args.token_count * args.hidden_size * sizeof(float);
}

__device__ bool rocm_valid_tiny_decode_args(const rocm_tiny_decode_launch_args &args)
{
	const uint32_t table_count = args.vocab_size * args.hidden_size;
	bool valid_output_weights = false;
	if (args.output_weight_encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP32) {
		valid_output_weights = args.output_weight_bytes == table_count * sizeof(float);
	} else if (args.output_weight_encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP16) {
		valid_output_weights = args.output_weight_bytes == table_count * sizeof(uint16_t);
	} else if (args.output_weight_encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_Q8) {
		const float scale = rocm_float_from_bits(args.q8_scale_bits);
		valid_output_weights = args.output_weight_bytes == table_count &&
			rocm_positive_finite(scale);
	}
	return args.version == ROCM_TINY_DECODE_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_TINY_DECODE_LAUNCH_ARGS_BYTES &&
		args.prior_key_pointer != 0 &&
		args.prior_value_pointer != 0 &&
		args.embedding_pointer != 0 &&
		args.output_weight_pointer != 0 &&
		args.logit_pointer != 0 &&
		args.attention_pointer != 0 &&
		args.updated_key_pointer != 0 &&
		args.updated_value_pointer != 0 &&
		args.result_pointer != 0 &&
		args.prior_token_count > 0 &&
		args.vocab_size > 0 &&
		args.hidden_size > 0 &&
		args.token_id < args.vocab_size &&
		args.prior_key_bytes == args.prior_token_count * args.hidden_size * sizeof(float) &&
		args.prior_value_bytes == args.prior_token_count * args.hidden_size * sizeof(float) &&
		args.embedding_bytes == args.vocab_size * args.hidden_size * sizeof(float) &&
		valid_output_weights &&
		args.logit_bytes == args.vocab_size * sizeof(float) &&
		args.attention_bytes == (args.prior_token_count + 1u) * sizeof(float) &&
		args.updated_key_bytes == (args.prior_token_count + 1u) * args.hidden_size * sizeof(float) &&
		args.updated_value_bytes == (args.prior_token_count + 1u) * args.hidden_size * sizeof(float) &&
		args.result_bytes == ROCM_GREEDY_RESULT_BYTES;
}

__device__ bool rocm_valid_cross_entropy_loss_args(const rocm_cross_entropy_loss_launch_args &args)
{
	return args.version == ROCM_CROSS_ENTROPY_LOSS_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_CROSS_ENTROPY_LOSS_LAUNCH_ARGS_BYTES &&
		args.logit_pointer != 0 &&
		args.target_pointer != 0 &&
		args.output_pointer != 0 &&
		args.batch > 0 &&
		args.vocab > 0 &&
		args.logit_bytes == args.batch * args.vocab * sizeof(float) &&
		args.target_bytes == args.batch * sizeof(int32_t) &&
		args.output_bytes == ROCM_CROSS_ENTROPY_LOSS_OUTPUT_BYTES;
}

__device__ bool rocm_valid_distillation_kl_loss_args(const rocm_distillation_kl_loss_launch_args &args)
{
	const double temperature = rocm_double_from_bits(args.temperature_bits);
	return args.version == ROCM_DISTILLATION_KL_LOSS_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_DISTILLATION_KL_LOSS_LAUNCH_ARGS_BYTES &&
		args.student_pointer != 0 &&
		args.teacher_pointer != 0 &&
		args.output_pointer != 0 &&
		args.batch > 0 &&
		args.vocab > 0 &&
		args.student_bytes == args.batch * args.vocab * sizeof(float) &&
		args.teacher_bytes == args.batch * args.vocab * sizeof(float) &&
		args.output_bytes == ROCM_DISTILLATION_KL_LOSS_OUTPUT_BYTES &&
		temperature > 0.0 &&
		isfinite(temperature);
}

__device__ bool rocm_valid_grpo_advantage_args(const rocm_grpo_advantage_launch_args &args)
{
	return args.version == ROCM_GRPO_ADVANTAGE_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_GRPO_ADVANTAGE_LAUNCH_ARGS_BYTES &&
		args.reward_pointer != 0 &&
		args.output_pointer != 0 &&
		args.count > 0 &&
		args.reward_bytes == args.count * sizeof(double) &&
		args.output_bytes == args.count * sizeof(double);
}

__device__ float rocm_tiny_output_weight_value(uint64_t pointer, uint32_t encoding, uint32_t index, float q8_scale)
{
	if (encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP32) {
		const float *weights = reinterpret_cast<const float *>(static_cast<uintptr_t>(pointer));
		return weights[index];
	}
	if (encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP16) {
		const uint16_t *weights = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(pointer));
		return rocm_half_to_float(weights[index]);
	}
	const int8_t *weights = reinterpret_cast<const int8_t *>(static_cast<uintptr_t>(pointer));
	return static_cast<float>(weights[index]) * q8_scale;
}

}

extern "C" __global__ void rocm_prefill(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_prefill_launch_args &args = *reinterpret_cast<const rocm_prefill_launch_args *>(packet);
	if (!rocm_valid_prefill_args(args)) {
		return;
	}
	const int32_t *tokens = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.token_pointer));
	volatile int32_t first_token = tokens[0];
	(void)first_token;
	if (args.status_pointer != 0) {
		uint32_t *status = reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(args.status_pointer));
		*status = args.status_value == 0 ? ROCM_PREFILL_LAUNCH_STATUS_OK : args.status_value;
	}
}

extern "C" __global__ void rocm_decode(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_decode_launch_args &args = *reinterpret_cast<const rocm_decode_launch_args *>(packet);
	if (!rocm_valid_decode_args(args)) {
		return;
	}
	const uint32_t *descriptor = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.kv.descriptor_pointer));
	volatile uint32_t descriptor_version = descriptor[0];
	volatile uint32_t descriptor_mode = descriptor[3];
	(void)descriptor_version;
	(void)descriptor_mode;
	if (args.kv.status_pointer != 0) {
		uint32_t *status = reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(args.kv.status_pointer));
		*status = args.kv.status_value == 0 ? ROCM_DECODE_LAUNCH_STATUS_OK : args.kv.status_value;
	}
}

extern "C" __global__ void rocm_projection(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_projection_launch_args &args = *reinterpret_cast<const rocm_projection_launch_args *>(packet);
	if (!rocm_valid_projection_args(args)) {
		return;
	}

	const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= args.rows) {
		return;
	}

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = 0.0f;
	if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_FP16) {
		const uint16_t *weights = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.weight_pointer));
		for (uint32_t col = 0; col < args.cols; ++col) {
			sum += input[col] * rocm_half_to_float(weights[row * args.cols + col]);
		}
	} else if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_BF16) {
		const uint16_t *weights = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.weight_pointer));
		for (uint32_t col = 0; col < args.cols; ++col) {
			sum += input[col] * rocm_bfloat16_to_float(weights[row * args.cols + col]);
		}
	} else if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_Q8) {
		const int8_t *weights = reinterpret_cast<const int8_t *>(static_cast<uintptr_t>(args.weight_pointer));
		const float scale = rocm_float_from_bits(args.q8_scale_bits);
		for (uint32_t col = 0; col < args.cols; ++col) {
			sum += input[col] * static_cast<float>(weights[row * args.cols + col]) * scale;
		}
	} else {
		const float *weights = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.weight_pointer));
		for (uint32_t col = 0; col < args.cols; ++col) {
			sum += input[col] * weights[row * args.cols + col];
		}
	}
	if ((args.flags & ROCM_PROJECTION_LAUNCH_FLAG_BIAS) != 0 && args.bias_pointer != 0) {
		const float *bias = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.bias_pointer));
		sum += bias[row];
	}
	output[row] = sum;
}

extern "C" __global__ void rocm_mlx_q4_projection(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_args(args)) {
		return;
	}

	const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= args.rows) {
		return;
	}

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint32_t packed_per_row = args.cols / 8;
	const uint32_t groups_per_row = args.cols / args.group_size;
	float sum = 0.0f;
	for (uint32_t col = 0; col < args.cols; ++col) {
		const uint32_t word = weights[row * packed_per_row + col / 8];
		const uint32_t q = (word >> ((col & 7u) * 4u)) & 0x0fu;
		const uint32_t group = row * groups_per_row + col / args.group_size;
		const float scale = rocm_bfloat16_to_float(scales[group]);
		const float bias = rocm_bfloat16_to_float(biases[group]);
		sum += input[col] * (static_cast<float>(q) * scale + bias);
	}
	output[row] = sum;
}

extern "C" __global__ void rocm_lora_projection(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_lora_launch_args &args = *reinterpret_cast<const rocm_lora_launch_args *>(packet);
	if (!rocm_valid_lora_args(args)) {
		return;
	}

	const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= args.rows) {
		return;
	}

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const float *base = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.base_weight_pointer));
	const float *lora_a = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.lora_a_pointer));
	const float *lora_b = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.lora_b_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));

	float sum = 0.0f;
	for (uint32_t col = 0; col < args.cols; ++col) {
		sum += input[col] * base[row * args.cols + col];
	}
	if ((args.flags & ROCM_LORA_LAUNCH_FLAG_BIAS) != 0 && args.bias_pointer != 0) {
		const float *bias = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.bias_pointer));
		sum += bias[row];
	}
	float delta = 0.0f;
	for (uint32_t r = 0; r < args.rank; ++r) {
		float down = 0.0f;
		for (uint32_t col = 0; col < args.cols; ++col) {
			down += lora_a[r * args.cols + col] * input[col];
		}
		delta += lora_b[row * args.rank + r] * down;
	}
	const float scale = rocm_float_from_bits(args.alpha_bits) / static_cast<float>(args.rank);
	output[row] = sum + scale * delta;
}

extern "C" __global__ void rocm_embedding_lookup(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_embedding_lookup_launch_args &args = *reinterpret_cast<const rocm_embedding_lookup_launch_args *>(packet);
	if (!rocm_valid_embedding_lookup_args(args)) {
		return;
	}
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t total = static_cast<uint64_t>(args.token_count) * args.hidden_size;
	if (index >= total) {
		return;
	}
	const int32_t *tokens = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.token_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint32_t token_index = index / args.hidden_size;
	const uint32_t dim = index - token_index * args.hidden_size;
	const int32_t token_id = tokens[token_index];
	if (token_id < 0 || static_cast<uint32_t>(token_id) >= args.vocab_size) {
		return;
	}
	if (args.table_encoding == ROCM_EMBEDDING_TABLE_ENCODING_MLX_Q4) {
		const uint32_t *embedding = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.embedding_pointer));
		const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
		const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
		const uint32_t packed_per_row = args.hidden_size / 8u;
		const uint32_t groups_per_row = args.hidden_size / args.group_size;
		const uint32_t token_u = static_cast<uint32_t>(token_id);
		const uint32_t word = embedding[static_cast<uint64_t>(token_u) * packed_per_row + dim / 8u];
		const uint32_t quantized = (word >> ((dim & 7u) * 4u)) & 0x0fu;
		const uint64_t group = static_cast<uint64_t>(token_u) * groups_per_row + dim / args.group_size;
		output[index] = static_cast<float>(quantized) * rocm_bfloat16_to_float(scales[group]) + rocm_bfloat16_to_float(biases[group]);
		return;
	}
	const uint64_t table_index = static_cast<uint64_t>(token_id) * args.hidden_size + dim;
	if (args.table_encoding == ROCM_EMBEDDING_TABLE_ENCODING_BF16) {
		const uint16_t *embedding = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.embedding_pointer));
		output[index] = rocm_bfloat16_to_float(embedding[table_index]);
	} else {
		const float *embedding = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.embedding_pointer));
		output[index] = embedding[table_index];
	}
}

extern "C" __global__ void rocm_embedding_mean_pool(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_embedding_mean_pool_launch_args &args = *reinterpret_cast<const rocm_embedding_mean_pool_launch_args *>(packet);
	if (!rocm_valid_embedding_mean_pool_args(args)) {
		return;
	}
	const float *tokens = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.token_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	for (uint32_t dim = 0; dim < args.dim; ++dim) {
		float sum = 0.0f;
		for (uint32_t token = 0; token < args.token_count; ++token) {
			sum += tokens[token * args.dim + dim];
		}
		output[dim] = sum / static_cast<float>(args.token_count);
	}
	if ((args.flags & ROCM_EMBEDDING_MEAN_POOL_LAUNCH_FLAG_NORMALIZE) != 0) {
		float norm = 0.0f;
		for (uint32_t dim = 0; dim < args.dim; ++dim) {
			norm += output[dim] * output[dim];
		}
		if (norm > 0.0f) {
			const float scale = rsqrtf(norm);
			for (uint32_t dim = 0; dim < args.dim; ++dim) {
				output[dim] *= scale;
			}
		}
	}
}

extern "C" __global__ void rocm_rerank_cosine(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_rerank_cosine_launch_args &args = *reinterpret_cast<const rocm_rerank_cosine_launch_args *>(packet);
	if (!rocm_valid_rerank_cosine_args(args)) {
		return;
	}
	const uint32_t doc = blockIdx.x * blockDim.x + threadIdx.x;
	if (doc >= args.document_count) {
		return;
	}
	const float *query = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.query_pointer));
	const float *documents = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.document_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float dot = 0.0f;
	float query_norm = 0.0f;
	float doc_norm = 0.0f;
	const float *document = documents + doc * args.dim;
	for (uint32_t dim = 0; dim < args.dim; ++dim) {
		const float q = query[dim];
		const float d = document[dim];
		dot += q * d;
		query_norm += q * q;
		doc_norm += d * d;
	}
	if (query_norm == 0.0f || doc_norm == 0.0f) {
		output[doc] = 0.0f;
		return;
	}
	output[doc] = dot * rsqrtf(query_norm) * rsqrtf(doc_norm);
}

extern "C" __global__ void rocm_rms_norm(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_rms_norm_launch_args &args = *reinterpret_cast<const rocm_rms_norm_launch_args *>(packet);
	if (!rocm_valid_rms_norm_args(args)) {
		return;
	}
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum_squares = 0.0f;
	for (uint32_t i = 0; i < args.count; ++i) {
		sum_squares += input[i] * input[i];
	}
	const float rms = sqrtf(sum_squares / static_cast<float>(args.count) + rocm_float_from_bits(args.epsilon_bits));
	if (rms == 0.0f) {
		return;
	}
	if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_BF16) {
		const uint16_t *weight = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.weight_pointer));
		for (uint32_t i = 0; i < args.count; ++i) {
			float w = rocm_bfloat16_to_float(weight[i]);
			if ((args.flags & ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) != 0) {
				w += 1.0f;
			}
			output[i] = input[i] / rms * w;
		}
	} else {
		const float *weight = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.weight_pointer));
		for (uint32_t i = 0; i < args.count; ++i) {
			float w = weight[i];
			if ((args.flags & ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) != 0) {
				w += 1.0f;
			}
			output[i] = input[i] / rms * w;
		}
	}
}

extern "C" __global__ void rocm_rope(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_rope_launch_args &args = *reinterpret_cast<const rocm_rope_launch_args *>(packet);
	if (!rocm_valid_rope_args(args)) {
		return;
	}
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const float base = rocm_float_from_bits(args.base_bits);
	const uint32_t frequency_dim = args.frequency_dim == 0 ? args.count : args.frequency_dim;
	const float dim = static_cast<float>(frequency_dim);
	for (uint32_t i = 0; i < args.count; i += 2) {
		const float frequency = 1.0f / powf(base, static_cast<float>(i) / dim);
		const float angle = static_cast<float>(args.position) * frequency;
		const float cosine = cosf(angle);
		const float sine = sinf(angle);
		const float x = input[i];
		const float y = input[i + 1];
		output[i] = x * cosine - y * sine;
		output[i + 1] = x * sine + y * cosine;
	}
}

extern "C" __global__ void rocm_greedy_sample(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_greedy_launch_args &args = *reinterpret_cast<const rocm_greedy_launch_args *>(packet);
	if (!rocm_valid_greedy_args(args)) {
		return;
	}
	const float *logits = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.logits_pointer));
	int32_t best_index = 0;
	float best_score = logits[0];
	for (uint32_t i = 1; i < args.count; ++i) {
		if (logits[i] > best_score) {
			best_index = static_cast<int32_t>(i);
			best_score = logits[i];
		}
	}
	const uintptr_t output_pointer = static_cast<uintptr_t>(args.output_pointer);
	int32_t *output_index = reinterpret_cast<int32_t *>(output_pointer);
	float *output_score = reinterpret_cast<float *>(output_pointer + sizeof(int32_t));
	*output_index = best_index;
	*output_score = best_score;
}

extern "C" __global__ void rocm_cross_entropy_loss(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_cross_entropy_loss_launch_args &args = *reinterpret_cast<const rocm_cross_entropy_loss_launch_args *>(packet);
	if (!rocm_valid_cross_entropy_loss_args(args)) {
		return;
	}
	const float *logits = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.logit_pointer));
	const int32_t *targets = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.target_pointer));
	double total = 0.0;
	for (uint32_t batch = 0; batch < args.batch; ++batch) {
		const float *row = logits + batch * args.vocab;
		const int32_t target = targets[batch];
		if (target < 0 || static_cast<uint32_t>(target) >= args.vocab) {
			return;
		}
		float max_value = row[0];
		for (uint32_t col = 1; col < args.vocab; ++col) {
			max_value = fmaxf(max_value, row[col]);
		}
		double sum = 0.0;
		for (uint32_t col = 0; col < args.vocab; ++col) {
			sum += exp(static_cast<double>(row[col] - max_value));
		}
		total += static_cast<double>(max_value) + log(sum) - static_cast<double>(row[target]);
	}
	const double loss = total / static_cast<double>(args.batch);
	double *output = reinterpret_cast<double *>(static_cast<uintptr_t>(args.output_pointer));
	output[0] = loss;
	output[1] = exp(loss);
}

extern "C" __global__ void rocm_distillation_kl_loss(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_distillation_kl_loss_launch_args &args = *reinterpret_cast<const rocm_distillation_kl_loss_launch_args *>(packet);
	if (!rocm_valid_distillation_kl_loss_args(args)) {
		return;
	}
	const float *student = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.student_pointer));
	const float *teacher = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.teacher_pointer));
	const double temperature = rocm_double_from_bits(args.temperature_bits);
	double total = 0.0;
	for (uint32_t batch = 0; batch < args.batch; ++batch) {
		const float *student_row = student + batch * args.vocab;
		const float *teacher_row = teacher + batch * args.vocab;
		double student_max = static_cast<double>(student_row[0]) / temperature;
		double teacher_max = static_cast<double>(teacher_row[0]) / temperature;
		for (uint32_t col = 1; col < args.vocab; ++col) {
			student_max = fmax(student_max, static_cast<double>(student_row[col]) / temperature);
			teacher_max = fmax(teacher_max, static_cast<double>(teacher_row[col]) / temperature);
		}
		double student_sum = 0.0;
		double teacher_sum = 0.0;
		for (uint32_t col = 0; col < args.vocab; ++col) {
			student_sum += exp(static_cast<double>(student_row[col]) / temperature - student_max);
			teacher_sum += exp(static_cast<double>(teacher_row[col]) / temperature - teacher_max);
		}
		const double student_norm = student_max + log(student_sum);
		const double teacher_norm = teacher_max + log(teacher_sum);
		for (uint32_t col = 0; col < args.vocab; ++col) {
			const double student_log_prob = static_cast<double>(student_row[col]) / temperature - student_norm;
			const double teacher_log_prob = static_cast<double>(teacher_row[col]) / temperature - teacher_norm;
			const double teacher_prob = exp(teacher_log_prob);
			total += teacher_prob * (teacher_log_prob - student_log_prob);
		}
	}
	double *output = reinterpret_cast<double *>(static_cast<uintptr_t>(args.output_pointer));
	output[0] = total * temperature * temperature / static_cast<double>(args.batch);
}

extern "C" __global__ void rocm_grpo_advantage(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_grpo_advantage_launch_args &args = *reinterpret_cast<const rocm_grpo_advantage_launch_args *>(packet);
	if (!rocm_valid_grpo_advantage_args(args)) {
		return;
	}
	const double *rewards = reinterpret_cast<const double *>(static_cast<uintptr_t>(args.reward_pointer));
	double *output = reinterpret_cast<double *>(static_cast<uintptr_t>(args.output_pointer));
	double mean = 0.0;
	for (uint32_t index = 0; index < args.count; ++index) {
		const double reward = rewards[index];
		if (!isfinite(reward)) {
			return;
		}
		mean += reward;
	}
	mean /= static_cast<double>(args.count);
	double variance = 0.0;
	for (uint32_t index = 0; index < args.count; ++index) {
		const double diff = rewards[index] - mean;
		variance += diff * diff;
	}
	variance /= static_cast<double>(args.count);
	if (variance == 0.0) {
		for (uint32_t index = 0; index < args.count; ++index) {
			output[index] = 0.0;
		}
		return;
	}
	const double stddev = sqrt(variance);
	for (uint32_t index = 0; index < args.count; ++index) {
		output[index] = (rewards[index] - mean) / stddev;
	}
}

__device__ const rocm_device_kv_page_descriptor *rocm_attention_device_kv_page(const rocm_attention_launch_args &args, uint32_t token)
{
	const rocm_device_kv_descriptor_header *header = reinterpret_cast<const rocm_device_kv_descriptor_header *>(static_cast<uintptr_t>(args.descriptor_pointer));
	const unsigned char *base = reinterpret_cast<const unsigned char *>(header);
	for (uint32_t page_index = 0; page_index < header->page_count; ++page_index) {
		const rocm_device_kv_page_descriptor *page = reinterpret_cast<const rocm_device_kv_page_descriptor *>(base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + page_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
		if (token >= page->token_start && token < page->token_start + page->token_count) {
			return page;
		}
	}
	return nullptr;
}

__device__ float rocm_attention_device_kv_value(const rocm_attention_launch_args &args, bool key, uint32_t token, uint32_t dim)
{
	const rocm_device_kv_page_descriptor *page = rocm_attention_device_kv_page(args, token);
	if (page == nullptr) {
		return 0.0f;
	}
	const uint32_t width = key ? page->key_width : page->value_width;
	const uint64_t pointer = key ? page->key_pointer : page->value_pointer;
	const uint32_t encoding = key ? page->key_encoding : page->value_encoding;
	const uint64_t local_token = static_cast<uint64_t>(token) - page->token_start;
	const uint64_t index = local_token * width + dim;
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16) {
		const uint16_t *values = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(pointer));
		return rocm_half_to_float(values[index]);
	}
	const unsigned char *bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(pointer));
	const uint32_t scale_bits = *reinterpret_cast<const uint32_t *>(bytes);
	const float scale = rocm_float_from_bits(scale_bits);
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8) {
		const int8_t *values = reinterpret_cast<const int8_t *>(bytes + sizeof(uint32_t));
		return static_cast<float>(values[index]) * scale;
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4) {
		const unsigned char *values = bytes + sizeof(uint32_t);
		unsigned char packed = values[index / 2u];
		if ((index & 1u) != 0u) {
			packed >>= 4;
		}
		int quantized = static_cast<int>(packed & 0x0fu);
		if (quantized >= 8) {
			quantized -= 16;
		}
		return static_cast<float>(quantized) * scale;
	}
	return 0.0f;
}

extern "C" __global__ void rocm_attention(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_attention_launch_args &args = *reinterpret_cast<const rocm_attention_launch_args *>(packet);
	if (!rocm_valid_attention_args(args)) {
		return;
	}
	const float *query = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.query_pointer));
	const float *keys = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.key_pointer));
	const float *values = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.value_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float *weights = reinterpret_cast<float *>(static_cast<uintptr_t>(args.weight_pointer));
	const float requested_scale = rocm_float_from_bits(args.scale_bits);
	const float scale = requested_scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(args.dim)) : requested_scale;
	const bool device_kv = args.kv_source == ROCM_ATTENTION_KV_SOURCE_DEVICE;

	float max_score = -3.4028234663852886e38f;
	for (uint32_t token = 0; token < args.token_count; ++token) {
		float score = 0.0f;
		const uint32_t base = token * args.dim;
		for (uint32_t dim = 0; dim < args.dim; ++dim) {
			const float key = device_kv ? rocm_attention_device_kv_value(args, true, token, dim) : keys[base + dim];
			score += query[dim] * key;
		}
		score *= scale;
		weights[token] = score;
		if (score > max_score) {
			max_score = score;
		}
	}

	float sum = 0.0f;
	for (uint32_t token = 0; token < args.token_count; ++token) {
		const float value = expf(weights[token] - max_score);
		weights[token] = value;
		sum += value;
	}
	if (sum == 0.0f) {
		return;
	}
	for (uint32_t dim = 0; dim < args.dim; ++dim) {
		output[dim] = 0.0f;
	}
	for (uint32_t token = 0; token < args.token_count; ++token) {
		const float weight = weights[token] / sum;
		weights[token] = weight;
		const uint32_t base = token * args.dim;
		for (uint32_t dim = 0; dim < args.dim; ++dim) {
			const float value = device_kv ? rocm_attention_device_kv_value(args, false, token, dim) : values[base + dim];
			output[dim] += weight * value;
		}
	}
}

extern "C" __global__ void rocm_vector_add(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_vector_add_launch_args &args = *reinterpret_cast<const rocm_vector_add_launch_args *>(packet);
	if (!rocm_valid_vector_add_args(args)) {
		return;
	}
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= args.count) {
		return;
	}
	const float *left = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.left_pointer));
	const float *right = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.right_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	output[index] = left[index] + right[index];
}

extern "C" __global__ void rocm_vector_scale(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_vector_scale_launch_args &args = *reinterpret_cast<const rocm_vector_scale_launch_args *>(packet);
	if (!rocm_valid_vector_scale_args(args)) {
		return;
	}
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= args.count) {
		return;
	}
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	output[index] = input[index] * rocm_float_from_bits(args.scale_bits);
}

extern "C" __global__ void rocm_swiglu(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_swiglu_launch_args &args = *reinterpret_cast<const rocm_swiglu_launch_args *>(packet);
	if (!rocm_valid_swiglu_args(args)) {
		return;
	}
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= args.count) {
		return;
	}
	const float *gate = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.gate_pointer));
	const float *up = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.up_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const float value = gate[index];
	output[index] = value / (1.0f + expf(-value)) * up[index];
}

extern "C" __global__ void rocm_moe_router(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_moe_router_launch_args &args = *reinterpret_cast<const rocm_moe_router_launch_args *>(packet);
	if (!rocm_valid_moe_router_args(args)) {
		return;
	}
	const float *logits = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.logit_pointer));
	int32_t *ids = reinterpret_cast<int32_t *>(static_cast<uintptr_t>(args.id_pointer));
	float *probs = reinterpret_cast<float *>(static_cast<uintptr_t>(args.prob_pointer));

	float max_logit = logits[0];
	for (uint32_t expert = 1; expert < args.expert_count; ++expert) {
		if (logits[expert] > max_logit) {
			max_logit = logits[expert];
		}
	}
	float sum = 0.0f;
	for (uint32_t expert = 0; expert < args.expert_count; ++expert) {
		sum += expf(logits[expert] - max_logit);
	}
	if (sum == 0.0f) {
		return;
	}
	for (uint32_t rank = 0; rank < args.top_k; ++rank) {
		int32_t best = -1;
		float best_score = -FLT_MAX;
		for (uint32_t expert = 0; expert < args.expert_count; ++expert) {
			bool used = false;
			for (uint32_t previous = 0; previous < rank; ++previous) {
				if (ids[previous] == static_cast<int32_t>(expert)) {
					used = true;
				}
			}
			if (used) {
				continue;
			}
			const float score = logits[expert];
			if (best < 0 || score > best_score || (score == best_score && expert < static_cast<uint32_t>(best))) {
				best = static_cast<int32_t>(expert);
				best_score = score;
			}
		}
		if (best < 0) {
			return;
		}
		ids[rank] = best;
		probs[rank] = expf(best_score - max_logit) / sum;
	}
	if (args.status_pointer != 0) {
		uint32_t *status = reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(args.status_pointer));
		*status = ROCM_MOE_ROUTER_LAUNCH_STATUS_OK;
	}
}

extern "C" __global__ void rocm_moe_lazy_experts(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_moe_lazy_launch_args &args = *reinterpret_cast<const rocm_moe_lazy_launch_args *>(packet);
	if (!rocm_valid_moe_lazy_args(args)) {
		return;
	}
	const uint32_t expert = blockIdx.x * blockDim.x + threadIdx.x;
	if (expert >= args.expert_count) {
		return;
	}
	const int32_t *ids = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.id_pointer));
	uint8_t *resident = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(args.resident_pointer));
	uint8_t value = 0;
	for (uint32_t index = 0; index < args.selected_count; ++index) {
		if (ids[index] == static_cast<int32_t>(expert)) {
			value = 1;
		}
	}
	resident[expert] = value;
}

extern "C" __global__ void rocm_jangtq_projection(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_jangtq_launch_args &args = *reinterpret_cast<const rocm_jangtq_launch_args *>(packet);
	if (!rocm_valid_jangtq_args(args)) {
		return;
	}
	const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= args.rows) {
		return;
	}
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint8_t *packed = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(args.packed_pointer));
	const float *bias = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const float scale = rocm_float_from_bits(args.scale_bits);
	float sum = 0.0f;
	if ((args.flags & ROCM_JANGTQ_LAUNCH_FLAG_BIAS) != 0) {
		sum = bias[row];
	}
	for (uint32_t col = 0; col < args.cols; ++col) {
		const uint32_t index = row * args.cols + col;
		const int8_t quantized = rocm_unpack_signed_bits(packed, args.bits, index);
		sum += input[col] * static_cast<float>(quantized) * scale;
	}
	output[row] = sum;
}

extern "C" __global__ void rocm_codebook_lookup(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_codebook_launch_args &args = *reinterpret_cast<const rocm_codebook_launch_args *>(packet);
	if (!rocm_valid_codebook_args(args)) {
		return;
	}
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t output_count = args.code_count * args.code_dim;
	if (index >= output_count) {
		return;
	}
	const uint8_t *codes = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(args.code_pointer));
	const float *codebook = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.codebook_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint32_t code_index = index / args.code_dim;
	const uint32_t dim = index % args.code_dim;
	const uint32_t code = static_cast<uint32_t>(codes[code_index]);
	if (code >= args.codebook_count) {
		return;
	}
	output[index] = codebook[code * args.code_dim + dim];
}

extern "C" __global__ void rocm_tiny_prefill(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_tiny_prefill_launch_args &args = *reinterpret_cast<const rocm_tiny_prefill_launch_args *>(packet);
	if (!rocm_valid_tiny_prefill_args(args)) {
		return;
	}
	const int32_t *tokens = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.token_pointer));
	const float *embedding = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.embedding_pointer));
	float *logits = reinterpret_cast<float *>(static_cast<uintptr_t>(args.logit_pointer));
	float *attention = reinterpret_cast<float *>(static_cast<uintptr_t>(args.attention_pointer));
	float *state_keys = reinterpret_cast<float *>(static_cast<uintptr_t>(args.key_pointer));
	float *state_values = reinterpret_cast<float *>(static_cast<uintptr_t>(args.value_pointer));
	const float q8_scale = rocm_float_from_bits(args.q8_scale_bits);

	const int32_t query_token = tokens[args.token_count - 1];
	if (query_token < 0 || static_cast<uint32_t>(query_token) >= args.vocab_size) {
		return;
	}
	const uint32_t query_base = static_cast<uint32_t>(query_token) * args.hidden_size;
	const float scale = 1.0f / sqrtf(static_cast<float>(args.hidden_size));
	float max_score = -3.4028234663852886e38f;
	for (uint32_t token = 0; token < args.token_count; ++token) {
		const int32_t token_id = tokens[token];
		if (token_id < 0 || static_cast<uint32_t>(token_id) >= args.vocab_size) {
			return;
		}
		const uint32_t key_base = static_cast<uint32_t>(token_id) * args.hidden_size;
		const uint32_t state_base = token * args.hidden_size;
		float score = 0.0f;
		for (uint32_t dim = 0; dim < args.hidden_size; ++dim) {
			const float value = embedding[key_base + dim];
			state_keys[state_base + dim] = value;
			state_values[state_base + dim] = value;
			score += embedding[query_base + dim] * value;
		}
		score *= scale;
		attention[token] = score;
		if (score > max_score) {
			max_score = score;
		}
	}

	float attention_sum = 0.0f;
	for (uint32_t token = 0; token < args.token_count; ++token) {
		const float value = expf(attention[token] - max_score);
		attention[token] = value;
		attention_sum += value;
	}
	if (attention_sum == 0.0f) {
		return;
	}
	for (uint32_t token = 0; token < args.token_count; ++token) {
		attention[token] = attention[token] / attention_sum;
	}

	int32_t best_index = 0;
	float best_score = 0.0f;
	for (uint32_t vocab = 0; vocab < args.vocab_size; ++vocab) {
		float logit = 0.0f;
		const uint32_t weight_base = vocab * args.hidden_size;
		for (uint32_t dim = 0; dim < args.hidden_size; ++dim) {
			float context_value = 0.0f;
			for (uint32_t token = 0; token < args.token_count; ++token) {
				const uint32_t value_base = static_cast<uint32_t>(tokens[token]) * args.hidden_size;
				context_value += attention[token] * embedding[value_base + dim];
			}
			logit += context_value * rocm_tiny_output_weight_value(args.output_weight_pointer, args.output_weight_encoding, weight_base + dim, q8_scale);
		}
		logits[vocab] = logit;
		if (vocab == 0 || logit > best_score) {
			best_index = static_cast<int32_t>(vocab);
			best_score = logit;
		}
	}

	const uintptr_t result_pointer = static_cast<uintptr_t>(args.result_pointer);
	int32_t *result_index = reinterpret_cast<int32_t *>(result_pointer);
	float *result_score = reinterpret_cast<float *>(result_pointer + sizeof(int32_t));
	*result_index = best_index;
	*result_score = best_score;
}

extern "C" __global__ void rocm_tiny_decode(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_tiny_decode_launch_args &args = *reinterpret_cast<const rocm_tiny_decode_launch_args *>(packet);
	if (!rocm_valid_tiny_decode_args(args)) {
		return;
	}
	const float *prior_keys = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.prior_key_pointer));
	const float *prior_values = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.prior_value_pointer));
	const float *embedding = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.embedding_pointer));
	float *logits = reinterpret_cast<float *>(static_cast<uintptr_t>(args.logit_pointer));
	float *attention = reinterpret_cast<float *>(static_cast<uintptr_t>(args.attention_pointer));
	float *updated_keys = reinterpret_cast<float *>(static_cast<uintptr_t>(args.updated_key_pointer));
	float *updated_values = reinterpret_cast<float *>(static_cast<uintptr_t>(args.updated_value_pointer));
	const float q8_scale = rocm_float_from_bits(args.q8_scale_bits);

	const uint32_t token_base = args.token_id * args.hidden_size;
	for (uint32_t index = 0; index < args.prior_token_count * args.hidden_size; ++index) {
		updated_keys[index] = prior_keys[index];
		updated_values[index] = prior_values[index];
	}
	const uint32_t updated_base = args.prior_token_count * args.hidden_size;
	for (uint32_t dim = 0; dim < args.hidden_size; ++dim) {
		updated_keys[updated_base + dim] = embedding[token_base + dim];
		updated_values[updated_base + dim] = embedding[token_base + dim];
	}

	const float scale = 1.0f / sqrtf(static_cast<float>(args.hidden_size));
	float max_score = -3.4028234663852886e38f;
	const uint32_t total_tokens = args.prior_token_count + 1u;
	for (uint32_t token = 0; token < total_tokens; ++token) {
		const uint32_t key_base = token * args.hidden_size;
		float score = 0.0f;
		for (uint32_t dim = 0; dim < args.hidden_size; ++dim) {
			score += embedding[token_base + dim] * updated_keys[key_base + dim];
		}
		score *= scale;
		attention[token] = score;
		if (score > max_score) {
			max_score = score;
		}
	}

	float attention_sum = 0.0f;
	for (uint32_t token = 0; token < total_tokens; ++token) {
		const float value = expf(attention[token] - max_score);
		attention[token] = value;
		attention_sum += value;
	}
	if (attention_sum == 0.0f) {
		return;
	}
	for (uint32_t token = 0; token < total_tokens; ++token) {
		attention[token] = attention[token] / attention_sum;
	}

	int32_t best_index = 0;
	float best_score = 0.0f;
	for (uint32_t vocab = 0; vocab < args.vocab_size; ++vocab) {
		float logit = 0.0f;
		const uint32_t weight_base = vocab * args.hidden_size;
		for (uint32_t dim = 0; dim < args.hidden_size; ++dim) {
			float context_value = 0.0f;
			for (uint32_t token = 0; token < total_tokens; ++token) {
				context_value += attention[token] * updated_values[token * args.hidden_size + dim];
			}
			logit += context_value * rocm_tiny_output_weight_value(args.output_weight_pointer, args.output_weight_encoding, weight_base + dim, q8_scale);
		}
		logits[vocab] = logit;
		if (vocab == 0 || logit > best_score) {
			best_index = static_cast<int32_t>(vocab);
			best_score = logit;
		}
	}

	const uintptr_t result_pointer = static_cast<uintptr_t>(args.result_pointer);
	int32_t *result_index = reinterpret_cast<int32_t *>(result_pointer);
	float *result_score = reinterpret_cast<float *>(result_pointer + sizeof(int32_t));
	*result_index = best_index;
	*result_score = best_score;
}
