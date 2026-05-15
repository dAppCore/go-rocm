// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && cgo && !rocm_legacy_server

package rocm

/*
#cgo linux LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>

typedef int (*hipGetDeviceCount_t)(int*);
typedef int (*hipSetDevice_t)(int);
typedef int (*hipMemGetInfo_t)(size_t*, size_t*);
typedef int (*hipRuntimeGetVersion_t)(int*);
typedef int (*hipMalloc_t)(void**, size_t);
typedef int (*hipFree_t)(void*);
typedef int (*hipMemcpy_t)(void*, const void*, size_t, int);
typedef int (*hipModuleLoadData_t)(void**, const void*);
typedef int (*hipModuleUnload_t)(void*);
typedef int (*hipModuleGetFunction_t)(void**, void*, const char*);
typedef int (*hipModuleLaunchKernel_t)(void*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, void*, void**, void**);

static void* core_rocm_hip_lib = NULL;

static void* core_rocm_open_hip() {
	if (core_rocm_hip_lib != NULL) {
		return core_rocm_hip_lib;
	}
	const char* names[] = {
		"libamdhip64.so",
		"libamdhip64.so.7",
		"libamdhip64.so.6",
		"libamdhip64.so.5",
		NULL,
	};
	for (int i = 0; names[i] != NULL; i++) {
		core_rocm_hip_lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
		if (core_rocm_hip_lib != NULL) {
			return core_rocm_hip_lib;
		}
	}
	return NULL;
}

static void* core_rocm_hip_symbol(const char* name) {
	void* lib = core_rocm_open_hip();
	if (lib == NULL) {
		return NULL;
	}
	return dlsym(lib, name);
}

static int core_rocm_hip_device_count(int* count) {
	hipGetDeviceCount_t fn = (hipGetDeviceCount_t)core_rocm_hip_symbol("hipGetDeviceCount");
	if (fn == NULL) {
		return -100001;
	}
	return fn(count);
}

static int core_rocm_hip_set_device(int device) {
	hipSetDevice_t fn = (hipSetDevice_t)core_rocm_hip_symbol("hipSetDevice");
	if (fn == NULL) {
		return -100002;
	}
	return fn(device);
}

static int core_rocm_hip_mem_info(size_t* free_bytes, size_t* total_bytes) {
	hipMemGetInfo_t fn = (hipMemGetInfo_t)core_rocm_hip_symbol("hipMemGetInfo");
	if (fn == NULL) {
		return -100003;
	}
	return fn(free_bytes, total_bytes);
}

static int core_rocm_hip_runtime_version(int* version) {
	hipRuntimeGetVersion_t fn = (hipRuntimeGetVersion_t)core_rocm_hip_symbol("hipRuntimeGetVersion");
	if (fn == NULL) {
		return -100004;
	}
	return fn(version);
}

static int core_rocm_hip_malloc(uintptr_t* out, size_t size) {
	hipMalloc_t fn = (hipMalloc_t)core_rocm_hip_symbol("hipMalloc");
	if (fn == NULL) {
		return -100005;
	}
	void* ptr = NULL;
	int rc = fn(&ptr, size);
	*out = (uintptr_t)ptr;
	return rc;
}

static int core_rocm_hip_free(uintptr_t ptr) {
	hipFree_t fn = (hipFree_t)core_rocm_hip_symbol("hipFree");
	if (fn == NULL) {
		return -100006;
	}
	return fn((void*)ptr);
}

static int core_rocm_hip_memcpy_htod(uintptr_t dst, void* src, size_t size) {
	hipMemcpy_t fn = (hipMemcpy_t)core_rocm_hip_symbol("hipMemcpy");
	if (fn == NULL) {
		return -100007;
	}
	return fn((void*)dst, src, size, 1);
}

static int core_rocm_hip_memcpy_dtoh(void* dst, uintptr_t src, size_t size) {
	hipMemcpy_t fn = (hipMemcpy_t)core_rocm_hip_symbol("hipMemcpy");
	if (fn == NULL) {
		return -100012;
	}
	return fn(dst, (void*)src, size, 2);
}

static int core_rocm_hip_module_load_data(uintptr_t* out, void* image) {
	hipModuleLoadData_t fn = (hipModuleLoadData_t)core_rocm_hip_symbol("hipModuleLoadData");
	if (fn == NULL) {
		return -100008;
	}
	void* module = NULL;
	int rc = fn(&module, image);
	*out = (uintptr_t)module;
	return rc;
}

static int core_rocm_hip_module_unload(uintptr_t module) {
	hipModuleUnload_t fn = (hipModuleUnload_t)core_rocm_hip_symbol("hipModuleUnload");
	if (fn == NULL) {
		return -100009;
	}
	return fn((void*)module);
}

static int core_rocm_hip_module_get_function(uintptr_t* out, uintptr_t module, const char* name) {
	hipModuleGetFunction_t fn = (hipModuleGetFunction_t)core_rocm_hip_symbol("hipModuleGetFunction");
	if (fn == NULL) {
		return -100010;
	}
	void* function = NULL;
	int rc = fn(&function, (void*)module, name);
	*out = (uintptr_t)function;
	return rc;
}

static int core_rocm_hip_module_launch_kernel(
	uintptr_t function,
	unsigned int grid_x,
	unsigned int grid_y,
	unsigned int grid_z,
	unsigned int block_x,
	unsigned int block_y,
	unsigned int block_z,
	unsigned int shared_mem_bytes,
	uintptr_t args
) {
	hipModuleLaunchKernel_t fn = (hipModuleLaunchKernel_t)core_rocm_hip_symbol("hipModuleLaunchKernel");
	if (fn == NULL) {
		return -100011;
	}
	uintptr_t arg_ptr = args;
	void* kernel_params[] = { &arg_ptr };
	return fn((void*)function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem_bytes, NULL, kernel_params, NULL);
}
*/
import "C"

import (
	"os"
	"unsafe"

	core "dappco.re/go"
)

type cgoHIPDriver struct{}

func newSystemHIPDriver() nativeHIPDriver {
	return cgoHIPDriver{}
}

func (cgoHIPDriver) Available() bool {
	var count C.int
	if rc := C.core_rocm_hip_device_count(&count); rc != 0 {
		return false
	}
	return count > 0
}

func (driver cgoHIPDriver) DeviceInfo() nativeDeviceInfo {
	var freeBytes C.size_t
	var totalBytes C.size_t
	if driver.Available() {
		_ = C.core_rocm_hip_set_device(0)
	}
	if rc := C.core_rocm_hip_mem_info(&freeBytes, &totalBytes); rc != 0 {
		if info, err := GetVRAMInfo(); err == nil {
			return nativeDeviceInfo{Name: "rocm", MemoryBytes: info.Total, FreeBytes: info.Free, Driver: "hip"}
		}
		return nativeDeviceInfo{Driver: "hip"}
	}
	var version C.int
	_ = C.core_rocm_hip_runtime_version(&version)
	return nativeDeviceInfo{
		Name:        "rocm",
		MemoryBytes: uint64(totalBytes),
		FreeBytes:   uint64(freeBytes),
		Driver:      core.Sprintf("hip:%d", int(version)),
	}
}

func (cgoHIPDriver) Malloc(size uint64) (nativeDevicePointer, error) {
	var ptr C.uintptr_t
	if rc := C.core_rocm_hip_malloc(&ptr, C.size_t(size)); rc != 0 {
		return 0, hipReturnError("hipMalloc", int(rc))
	}
	return nativeDevicePointer(ptr), nil
}

func (cgoHIPDriver) Free(pointer nativeDevicePointer) error {
	if pointer == 0 {
		return nil
	}
	if rc := C.core_rocm_hip_free(C.uintptr_t(pointer)); rc != 0 {
		return hipReturnError("hipFree", int(rc))
	}
	return nil
}

func (cgoHIPDriver) CopyHostToDevice(pointer nativeDevicePointer, data []byte) error {
	if len(data) == 0 {
		return nil
	}
	if rc := C.core_rocm_hip_memcpy_htod(C.uintptr_t(pointer), unsafe.Pointer(&data[0]), C.size_t(len(data))); rc != 0 {
		return hipReturnError("hipMemcpyHostToDevice", int(rc))
	}
	return nil
}

func (cgoHIPDriver) CopyDeviceToHost(pointer nativeDevicePointer, data []byte) error {
	if len(data) == 0 {
		return nil
	}
	if rc := C.core_rocm_hip_memcpy_dtoh(unsafe.Pointer(&data[0]), C.uintptr_t(pointer), C.size_t(len(data))); rc != 0 {
		return hipReturnError("hipMemcpyDeviceToHost", int(rc))
	}
	return nil
}

func (driver cgoHIPDriver) LaunchKernel(config hipKernelLaunchConfig) error {
	if err := config.Validate(); err != nil {
		return err
	}
	if !driver.Available() {
		return core.E("rocm.hip.LaunchKernel", "HIP driver is not available", nil)
	}
	modulePath := core.Trim(os.Getenv("GO_ROCM_KERNEL_HSACO"))
	if modulePath == "" {
		return core.E("rocm.hip.LaunchKernel", "GO_ROCM_KERNEL_HSACO is not set; native HIP kernels are not linked yet", nil)
	}
	image, err := os.ReadFile(modulePath)
	if err != nil {
		return core.E("rocm.hip.LaunchKernel", "read kernel module "+modulePath, err)
	}
	if len(image) == 0 {
		return core.E("rocm.hip.LaunchKernel", "kernel module is empty "+modulePath, nil)
	}
	cImage := C.CBytes(image)
	defer C.free(cImage)

	var module C.uintptr_t
	if rc := C.core_rocm_hip_module_load_data(&module, cImage); rc != 0 {
		return hipReturnError("hipModuleLoadData", int(rc))
	}
	defer C.core_rocm_hip_module_unload(module)

	cName := C.CString(config.Name)
	defer C.free(unsafe.Pointer(cName))
	var function C.uintptr_t
	if rc := C.core_rocm_hip_module_get_function(&function, module, cName); rc != 0 {
		return hipReturnError("hipModuleGetFunction", int(rc))
	}

	args, err := driver.Malloc(uint64(len(config.Args)))
	if err != nil {
		return core.E("rocm.hip.LaunchKernel", "allocate kernel argument packet", err)
	}
	defer driver.Free(args)
	if err := driver.CopyHostToDevice(args, config.Args); err != nil {
		return core.E("rocm.hip.LaunchKernel", "copy kernel argument packet", err)
	}
	if rc := C.core_rocm_hip_module_launch_kernel(
		function,
		C.uint(config.GridX),
		C.uint(config.GridY),
		C.uint(config.GridZ),
		C.uint(config.BlockX),
		C.uint(config.BlockY),
		C.uint(config.BlockZ),
		C.uint(config.SharedMemBytes),
		C.uintptr_t(args),
	); rc != 0 {
		return hipReturnError("hipModuleLaunchKernel", int(rc))
	}
	return nil
}

func hipReturnError(op string, code int) error {
	return core.E("rocm.hip."+op, core.Sprintf("HIP returned %d", code), nil)
}
