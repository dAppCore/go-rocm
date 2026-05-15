// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	"context"
	"os"

	core "dappco.re/go"
)

type hipNativeProjectionKernelSet struct {
	hipKernelStub
}

func newHIPRuntimeKernelSet(driver nativeHIPDriver) hipKernelSet {
	if driver == nil {
		return newDefaultHIPKernelSet()
	}
	if _, ok := driver.(nativeHIPKernelLauncher); !ok {
		return newDefaultHIPKernelSet()
	}
	if core.Trim(os.Getenv("GO_ROCM_KERNEL_HSACO")) == "" {
		return newDefaultHIPKernelSet()
	}
	return hipNativeProjectionKernelSet{}
}

func (hipNativeProjectionKernelSet) Status() hipKernelStatus {
	return hipKernelStatus{
		CrossEntropy: hipKernelStatusLinked,
		Decode:       hipKernelStatusNotLinked,
		Distillation: hipKernelStatusLinked,
		Embedding:    hipKernelStatusLinked,
		GRPO:         hipKernelStatusLinked,
		Prefill:      hipKernelStatusNotLinked,
		Projection:   hipKernelStatusLinked,
		Rerank:       hipKernelStatusLinked,
		KVCache:      hipKernelStatusPlanned,
		Reason:       "native projection, embedding, rerank, and toy loss kernels configured by GO_ROCM_KERNEL_HSACO; prefill/decode kernels are not linked yet",
	}
}

func (kernels hipNativeProjectionKernelSet) Project(ctx context.Context, model *hipLoadedModel, req hipProjectionRequest) ([]float32, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	if model == nil || model.driver == nil {
		return nil, core.E("rocm.hip.Project", "HIP driver is nil", nil)
	}
	return hipRunProjectionKernel(ctx, model.driver, req)
}
