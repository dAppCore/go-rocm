// SPDX-Licence-Identifier: EUPL-1.2

package rocm

import root "dappco.re/go/rocm"

const (
	RuntimeDispatchStatusActive                    = root.RuntimeDispatchStatusActive
	RuntimeDispatchStatusDevROCm                   = root.RuntimeDispatchStatusDevROCm
	RuntimeDispatchStatusCompileReadyPending       = root.RuntimeDispatchStatusCompileReadyPending
	RuntimeLaneAMD                                 = root.RuntimeLaneAMD
	RuntimeLaneCUDA                                = root.RuntimeLaneCUDA
	RuntimeLaneCPUX86                              = root.RuntimeLaneCPUX86
	RuntimeLaneCPUAArch64                          = root.RuntimeLaneCPUAArch64
	RuntimeLaneArtifactAMD                         = root.RuntimeLaneArtifactAMD
	RuntimeLaneArtifactCUDA                        = root.RuntimeLaneArtifactCUDA
	RuntimeLaneArtifactCPUX86                      = root.RuntimeLaneArtifactCPUX86
	RuntimeLaneArtifactCPUAArch64                  = root.RuntimeLaneArtifactCPUAArch64
	RuntimeLaneSidecarAMD                          = root.RuntimeLaneSidecarAMD
	RuntimeLaneSidecarCUDA                         = root.RuntimeLaneSidecarCUDA
	RuntimeLaneSidecarCPUX86                       = root.RuntimeLaneSidecarCPUX86
	RuntimeLaneSidecarCPUAArch64                   = root.RuntimeLaneSidecarCPUAArch64
	RuntimeLaneDispatchNextWorkCUDA                = root.RuntimeLaneDispatchNextWorkCUDA
	RuntimeLaneDispatchNextWorkCPU                 = root.RuntimeLaneDispatchNextWorkCPU
	RuntimeLaneDispatchNextWorkStatefulGenerate    = root.RuntimeLaneDispatchNextWorkStatefulGenerate
	RuntimeLaneDispatchNextWorkOpenAIServer        = root.RuntimeLaneDispatchNextWorkOpenAIServer
	RuntimeLaneDispatchNextWorkThroughputBenchmark = root.RuntimeLaneDispatchNextWorkThroughputBenchmark
)

type RuntimeLaneStatus = root.RuntimeLaneStatus

func DefaultRuntimeLanes() []RuntimeLaneStatus {
	return root.DefaultRuntimeLanes()
}

func RuntimeLaneForArtifact(name string) (RuntimeLaneStatus, bool) {
	return root.RuntimeLaneForArtifact(name)
}

func RuntimeLanesForBackend(backend string) []RuntimeLaneStatus {
	return root.RuntimeLanesForBackend(backend)
}

func CurrentProcessRuntimeLane(name string) RuntimeLaneStatus {
	return root.CurrentProcessRuntimeLane(name)
}

func AnnotateRuntimeLaneForCurrentProcess(lane RuntimeLaneStatus) RuntimeLaneStatus {
	return root.AnnotateRuntimeLaneForCurrentProcess(lane)
}
