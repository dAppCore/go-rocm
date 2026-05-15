// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package rocm

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func Example_evaluationLossCapabilityReport() {
	defaultReport := (&rocmBackend{}).Capabilities()
	linkedReport := rocmCapabilityReport(nativeDeviceInfo{}, inference.ModelIdentity{}, inference.AdapterIdentity{}, true, hipKernelStatus{
		CrossEntropy: hipKernelStatusLinked,
	})

	for _, report := range []inference.CapabilityReport{defaultReport, linkedReport} {
		capability, _ := report.Capability(inference.CapabilityEvaluation)
		core.Println(
			capability.ID,
			capability.Status,
			capability.Labels["loss_kernel"],
			capability.Labels["loss_kernel_name"],
			capability.Labels["loss_scope"],
		)
	}
	// Output:
	// evaluation experimental not_linked rocm_cross_entropy_loss toy_cross_entropy
	// evaluation experimental linked rocm_cross_entropy_loss toy_cross_entropy
}

func Example_trainingCapabilityReport() {
	report := (&rocmBackend{}).Capabilities()
	for _, id := range []inference.CapabilityID{
		inference.CapabilityLoRATraining,
		inference.CapabilityDistillation,
		inference.CapabilityGRPO,
	} {
		capability, _ := report.Capability(id)
		core.Println(
			capability.ID,
			capability.Status,
			capability.Labels["runtime_status"],
			capability.Labels["training_kernel"],
			capability.Labels["training_interface"],
			capability.Labels["required_kernel"],
		)
	}
	// Output:
	// lora.training planned planned not_linked not_implemented lora_backward
	// distillation planned planned not_linked not_implemented distillation_forward_loss
	// grpo planned planned not_linked not_implemented grpo_rollout_policy
}

func Example_metadataOnlyFixtureCapabilities() {
	report := (&rocmBackend{}).Capabilities()
	for _, id := range []inference.CapabilityID{
		inference.CapabilityMoERouting,
		inference.CapabilityMoELazyExperts,
		inference.CapabilityJANGTQ,
		inference.CapabilityCodebookVQ,
	} {
		capability, _ := report.Capability(id)
		core.Println(
			capability.ID,
			capability.Labels["runtime_status"],
			capability.Labels["kernel_status"],
			capability.Labels["fixture_kernel_name"],
			capability.Labels["production_integration"],
			capability.Labels["required_integration"],
		)
	}
	// Output:
	// moe.routing metadata_only planned rocm_moe_router pending model_router_forward
	// moe.lazy_experts metadata_only planned rocm_moe_lazy_experts pending expert_paging
	// jangtq metadata_only planned rocm_jangtq_projection pending packed_weight_model_integration
	// codebook.vq metadata_only planned rocm_codebook_lookup pending codebook_weight_model_integration
}
