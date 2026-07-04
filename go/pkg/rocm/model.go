// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import (
	"dappco.re/go/inference"
	root "dappco.re/go/rocm"
)

const (
	ROCmModelRoutePlanContract = root.ROCmModelRoutePlanContract
	ROCmTokenLoopContract      = root.ROCmTokenLoopContract
)

type (
	ROCmArchitectureProfile           = root.ROCmArchitectureProfile
	Gemma4ArchitectureSettings        = root.Gemma4ArchitectureSettings
	Gemma4DeclaredFeatures            = root.Gemma4DeclaredFeatures
	Gemma4EngineFeatures              = root.Gemma4EngineFeatures
	Gemma4LoRATargetPolicy            = root.Gemma4LoRATargetPolicy
	ROCmModelProfile                  = root.ROCmModelProfile
	ROCmModelIdentityReporter         = root.ROCmModelIdentityReporter
	ROCmModelInfoReporter             = root.ROCmModelInfoReporter
	ROCmModelInfoRequest              = root.ROCmModelInfoRequest
	ROCmModelInfoReport               = root.ROCmModelInfoReport
	ROCmModelProfileReporter          = root.ROCmModelProfileReporter
	ROCmModelRoutePlanReporter        = root.ROCmModelRoutePlanReporter
	ROCmModelProfileRequest           = root.ROCmModelProfileRequest
	ROCmModelProfileFactory           = root.ROCmModelProfileFactory
	ROCmEngineFeatures                = root.ROCmEngineFeatures
	ROCmEngineFeaturesReporter        = root.ROCmEngineFeaturesReporter
	ROCmModelRoutePlan                = root.ROCmModelRoutePlan
	ROCmModelFeatureRoute             = root.ROCmModelFeatureRoute
	ROCmModelTokenizerRoute           = root.ROCmModelTokenizerRoute
	ROCmLoRAAdapterRoute              = root.ROCmLoRAAdapterRoute
	ROCmMultimodalProcessorRoute      = root.ROCmMultimodalProcessorRoute
	ROCmDiffusionSamplerRoute         = root.ROCmDiffusionSamplerRoute
	ROCmStateContextRoute             = root.ROCmStateContextRoute
	ROCmAttachedDrafterRoute          = root.ROCmAttachedDrafterRoute
	ROCmModelLoadStatus               = root.ROCmModelLoadStatus
	ROCmModelLoaderRoute              = root.ROCmModelLoaderRoute
	ROCmQuantLoaderRoute              = root.ROCmQuantLoaderRoute
	ROCmSequenceMixerLoaderRoute      = root.ROCmSequenceMixerLoaderRoute
	ROCmModelRuntimeContractRoute     = root.ROCmModelRuntimeContractRoute
	ROCmRetainedStateStatus           = root.ROCmRetainedStateStatus
	ROCmTokenLoopStatus               = root.ROCmTokenLoopStatus
	ROCmCacheProfileReporter          = root.ROCmCacheProfileReporter
	ROCmNativeModelLoaderRegistration = root.ROCmNativeModelLoaderRegistration
	ProductionQuantizationPackSupport = root.ProductionQuantizationPackSupport
)

func ROCmArchitectureID(architecture string) string {
	return root.ROCmArchitectureID(architecture)
}

func DefaultROCmArchitectureProfiles() []ROCmArchitectureProfile {
	return root.DefaultROCmArchitectureProfiles()
}

func RegisteredROCmArchitectureProfileIDs() []string {
	return root.RegisteredROCmArchitectureProfileIDs()
}

func RegisteredROCmArchitectureProfiles() []ROCmArchitectureProfile {
	return root.RegisteredROCmArchitectureProfiles()
}

func ROCmArchitectureProfileForArchitecture(architecture string) (ROCmArchitectureProfile, bool) {
	return root.ROCmArchitectureProfileForArchitecture(architecture)
}

func Gemma4ArchitectureSettingsForArchitecture(architecture string) (Gemma4ArchitectureSettings, bool) {
	return root.Gemma4ArchitectureSettingsForArchitecture(architecture)
}

func RegisterROCmModelProfileFactory(factory ROCmModelProfileFactory) {
	root.RegisterROCmModelProfileFactory(factory)
}

func RegisteredROCmModelProfileFactoryNames() []string {
	return root.RegisteredROCmModelProfileFactoryNames()
}

func ResolveROCmModelInfo(req ROCmModelInfoRequest) ROCmModelInfoReport {
	return root.ResolveROCmModelInfo(req)
}

func ROCmModelInfoFromIdentity(path string, identity inference.ModelIdentity) inference.ModelInfo {
	return root.ROCmModelInfoFromIdentity(path, identity)
}

func ROCmModelInfoIdentity(path string, info inference.ModelInfo, labels map[string]string) inference.ModelIdentity {
	return root.ROCmModelInfoIdentity(path, info, labels)
}

func ROCmModelInfoReportForModel(model inference.TextModel) (ROCmModelInfoReport, bool) {
	return root.ROCmModelInfoReportForModel(model)
}

func ResolveROCmModelProfile(path string, model inference.ModelIdentity) (ROCmModelProfile, bool) {
	return root.ResolveROCmModelProfile(path, model)
}

func ResolveROCmModelProfileForInspection(inspection *inference.ModelPackInspection) (ROCmModelProfile, bool) {
	return root.ResolveROCmModelProfileForInspection(inspection)
}

func ResolveROCmModelProfileForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmModelProfile, bool) {
	return root.ResolveROCmModelProfileForInfo(path, info, labels)
}

func ResolveROCmModelProfileForModel(model inference.TextModel) (ROCmModelProfile, bool) {
	return root.ResolveROCmModelProfileForModel(model)
}

func ApplyROCmModelProfileLabels(labels map[string]string, profile ROCmModelProfile) map[string]string {
	return root.ApplyROCmModelProfileLabels(labels, profile)
}

func ROCmEngineFeaturesFor(model any) (ROCmEngineFeatures, bool) {
	return root.ROCmEngineFeaturesFor(model)
}

func ROCmEngineFeaturesForIdentity(path string, model inference.ModelIdentity) (ROCmEngineFeatures, bool) {
	return root.ROCmEngineFeaturesForIdentity(path, model)
}

func ROCmEngineFeaturesForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmEngineFeatures, bool) {
	return root.ROCmEngineFeaturesForInfo(path, info, labels)
}

func ROCmEngineFeaturesForInspection(inspection *inference.ModelPackInspection) (ROCmEngineFeatures, bool) {
	return root.ROCmEngineFeaturesForInspection(inspection)
}

func ROCmEngineFeaturesForModel(model inference.TextModel) (ROCmEngineFeatures, bool) {
	return root.ROCmEngineFeaturesForModel(model)
}

func ROCmEngineFeaturesForProfile(profile ROCmModelProfile) ROCmEngineFeatures {
	return root.ROCmEngineFeaturesForProfile(profile)
}

func ROCmModelRoutePlanForIdentity(path string, model inference.ModelIdentity) (ROCmModelRoutePlan, bool) {
	return root.ROCmModelRoutePlanForIdentity(path, model)
}

func ROCmModelRoutePlanForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmModelRoutePlan, bool) {
	return root.ROCmModelRoutePlanForInfo(path, info, labels)
}

func ROCmModelRoutePlanForInspection(inspection *inference.ModelPackInspection) (ROCmModelRoutePlan, bool) {
	return root.ROCmModelRoutePlanForInspection(inspection)
}

func ROCmModelRoutePlanForModel(model inference.TextModel) (ROCmModelRoutePlan, bool) {
	return root.ROCmModelRoutePlanForModel(model)
}

func ROCmRetainedStateForIdentity(path string, model inference.ModelIdentity) (ROCmRetainedStateStatus, bool) {
	return root.ROCmRetainedStateForIdentity(path, model)
}

func ROCmRetainedStateForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmRetainedStateStatus, bool) {
	return root.ROCmRetainedStateForInfo(path, info, labels)
}

func ROCmRetainedStateForInspection(inspection *inference.ModelPackInspection) (ROCmRetainedStateStatus, bool) {
	return root.ROCmRetainedStateForInspection(inspection)
}

func ROCmRetainedStateForModel(model inference.TextModel) (ROCmRetainedStateStatus, bool) {
	return root.ROCmRetainedStateForModel(model)
}

func ROCmTokenLoopForIdentity(path string, model inference.ModelIdentity) (ROCmTokenLoopStatus, bool) {
	return root.ROCmTokenLoopForIdentity(path, model)
}

func ROCmTokenLoopForInfo(path string, info inference.ModelInfo, labels map[string]string) (ROCmTokenLoopStatus, bool) {
	return root.ROCmTokenLoopForInfo(path, info, labels)
}

func ROCmTokenLoopForInspection(inspection *inference.ModelPackInspection) (ROCmTokenLoopStatus, bool) {
	return root.ROCmTokenLoopForInspection(inspection)
}

func ROCmTokenLoopForModel(model inference.TextModel) (ROCmTokenLoopStatus, bool) {
	return root.ROCmTokenLoopForModel(model)
}

func ROCmModelRoutePlanForProfileAndModel(profile ROCmModelProfile, model inference.TextModel) ROCmModelRoutePlan {
	return root.ROCmModelRoutePlanForProfileAndModel(profile, model)
}

func ROCmModelRoutePlanForProfile(profile ROCmModelProfile) ROCmModelRoutePlan {
	return root.ROCmModelRoutePlanForProfile(profile)
}

func ApplyROCmModelRoutePlanLabels(labels map[string]string, plan ROCmModelRoutePlan) map[string]string {
	return root.ApplyROCmModelRoutePlanLabels(labels, plan)
}

func RegisteredROCmNativeModelLoaderRegistrations() []ROCmNativeModelLoaderRegistration {
	return root.RegisteredROCmNativeModelLoaderRegistrations()
}

func ROCmNativeModelLoaderRegistrationForArchitecture(architecture string) (ROCmNativeModelLoaderRegistration, bool) {
	return root.ROCmNativeModelLoaderRegistrationForArchitecture(architecture)
}

func DefaultProductionQuantizationPackSupport() []ProductionQuantizationPackSupport {
	return root.DefaultProductionQuantizationPackSupport()
}

func ProductionQuantizationPacksBySize(size string) []ProductionQuantizationPackSupport {
	return root.ProductionQuantizationPacksBySize(size)
}

func ProductionQuantizationPackByName(name string) (ProductionQuantizationPackSupport, bool) {
	return root.ProductionQuantizationPackByName(name)
}

func ApplyProductionQuantizationPackSupportLabels(labels map[string]string) {
	root.ApplyProductionQuantizationPackSupportLabels(labels)
}
