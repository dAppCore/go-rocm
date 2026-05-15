// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import core "dappco.re/go"

const (
	hipKernelNamePrefill       = "rocm_prefill"
	hipKernelNameDecode        = "rocm_decode"
	hipKernelNameProjection    = "rocm_projection"
	hipKernelNameMLXQ4Proj     = "rocm_mlx_q4_projection"
	hipKernelNameRMSNorm       = "rocm_rms_norm"
	hipKernelNameRoPE          = "rocm_rope"
	hipKernelNameGreedy        = "rocm_greedy_sample"
	hipKernelNameAttention     = "rocm_attention"
	hipKernelNameVectorAdd     = "rocm_vector_add"
	hipKernelNameVectorScale   = "rocm_vector_scale"
	hipKernelNameSwiGLU        = "rocm_swiglu"
	hipKernelNameMoERouter     = "rocm_moe_router"
	hipKernelNameMoELazy       = "rocm_moe_lazy_experts"
	hipKernelNameJANGTQ        = "rocm_jangtq_projection"
	hipKernelNameCodebook      = "rocm_codebook_lookup"
	hipKernelNameLoRA          = "rocm_lora_projection"
	hipKernelNameEmbedLookup   = "rocm_embedding_lookup"
	hipKernelNameEmbedMean     = "rocm_embedding_mean_pool"
	hipKernelNameRerank        = "rocm_rerank_cosine"
	hipKernelNameTinyPrefill   = "rocm_tiny_prefill"
	hipKernelNameTinyDecode    = "rocm_tiny_decode"
	hipKernelNameCrossEntropy  = "rocm_cross_entropy_loss"
	hipKernelNameDistillKL     = "rocm_distillation_kl_loss"
	hipKernelNameGRPOAdvantage = "rocm_grpo_advantage"
)

type hipKernelLaunchConfig struct {
	Name           string
	Args           []byte
	GridX          uint32
	GridY          uint32
	GridZ          uint32
	BlockX         uint32
	BlockY         uint32
	BlockZ         uint32
	SharedMemBytes uint32
}

type nativeHIPKernelLauncher interface {
	LaunchKernel(config hipKernelLaunchConfig) error
}

func hipLaunchKernel(driver nativeHIPDriver, config hipKernelLaunchConfig) error {
	if err := config.Validate(); err != nil {
		return err
	}
	if driver == nil {
		return core.E("rocm.hip.LaunchKernel", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return core.E("rocm.hip.LaunchKernel", "HIP driver is not available", nil)
	}
	launcher, ok := driver.(nativeHIPKernelLauncher)
	if !ok {
		return core.E("rocm.hip.LaunchKernel", "native HIP kernel launcher is not linked yet", nil)
	}
	return launcher.LaunchKernel(config)
}

func (config hipKernelLaunchConfig) Validate() error {
	if core.Trim(config.Name) == "" {
		return core.E("rocm.hip.LaunchKernel", "kernel name is required", nil)
	}
	if len(config.Args) == 0 {
		return core.E("rocm.hip.LaunchKernel", "kernel launch args are required", nil)
	}
	if config.GridX == 0 || config.GridY == 0 || config.GridZ == 0 {
		return core.E("rocm.hip.LaunchKernel", "kernel grid dimensions must be positive", nil)
	}
	if config.BlockX == 0 || config.BlockY == 0 || config.BlockZ == 0 {
		return core.E("rocm.hip.LaunchKernel", "kernel block dimensions must be positive", nil)
	}
	return nil
}

func hipOneDimensionalLaunchConfig(name string, args []byte, workItems int) (hipKernelLaunchConfig, error) {
	work, err := rocmDeviceKVPositiveUint32("work items", workItems)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	const blockSize uint32 = 64
	gridX := (work + blockSize - 1) / blockSize
	config := hipKernelLaunchConfig{
		Name:   name,
		Args:   append([]byte(nil), args...),
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: blockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}
