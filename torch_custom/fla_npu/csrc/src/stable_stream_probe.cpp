// `_stream_probe`: reports the current stream two ways so Python can compare
// them against torch_npu's raw accessor.  Debug-only; the launcher's ABI
// helpers it needs are looked up at run time, so a probe-enabled build still
// loads on torch 2.7.1/2.8.

#include <torch/csrc/stable/library.h>
#ifndef FLA_STABLE_NO_DEBUG_PROBE
#include <torch/csrc/stable/accelerator.h>
#endif
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>

#include "stable/acl_meta.h"
// For detail::enqueue_launch: the submission the macro applies for free.
#include "stable/exec.h"

#include <dlfcn.h>

#include <cstdint>
#include <optional>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::AclTensorView;
using fla_npu_stable::stable::LaunchFn;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::aclOpExecutor;
using fla_npu_stable::stable::aclTensor;
using fla_npu_stable::stable::allocate_bytes;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::meta_of_handle;
using fla_npu_stable::stable::meta_optional_handle;
using fla_npu_stable::stable::kAclFormatNd;
using fla_npu_stable::stable::note_launch_stream;
using fla_npu_stable::stable::detail::check_async_failure;
using fla_npu_stable::stable::detail::enqueue_launch;

// Prefixed per op: everything lives in one TU (see stable_ops.cpp), so shared
// local names would collide.
using GdrGetWorkspaceFn = int (*)(const aclTensor*, const aclTensor*,
                                  const aclTensor*, const aclTensor*,
                                  aclTensor*, const aclTensor*,
                                  const aclTensor*, const aclTensor*,
                                  const aclTensor*, const aclTensor*, float,
                                  aclTensor*, uint64_t*, aclOpExecutor**);

#ifndef FLA_STABLE_NO_DEBUG_PROBE
// Reports the current stream for `device_index` two ways so Python can compare
// them against torch_npu's raw accessor (both returned 0 on torch_npu
// 2.9.0.post2, i.e. the stable stream API does not map to the NPU stream yet).
//
// The stream shims this needs -- aoti_torch_get_current_stream,
// aoti_torch_stream_id and aoti_torch_delete_stream -- only exist from torch
// 2.9 on, and torch::stable::accelerator::getCurrentStream() resolves the same
// three.  They are looked up at run time instead of being left to the loader,
// because a probe-enabled build would otherwise refuse to load on torch
// 2.7.1/2.8 with "undefined symbol: aoti_torch_stream_id": the debug flag would
// silently raise the library's torch floor above the >= 2.7.1 the wheel
// metadata declares.  Where the shims are missing the probe reports -3 and
// every operator keeps working.
using GetCurrentStreamFn = int32_t (*)(int32_t, StreamHandle*);
using StreamIdFn = int32_t (*)(StreamHandle, int64_t*);
using DeleteStreamFn = int32_t (*)(StreamHandle);

void* stable_runtime_symbol(const char* name) {
  // The launcher links libtorch_cpu/libc10/libtorch, so those objects are
  // already mapped; RTLD_NOLOAD reaches them without depending on whether
  // torch.ops.load_library() opened the launcher into the global scope.
  static void* const libs[] = {
      dlopen("libtorch_cpu.so", RTLD_NOLOAD | RTLD_LAZY),
      dlopen("libtorch.so", RTLD_NOLOAD | RTLD_LAZY),
      dlopen("libc10.so", RTLD_NOLOAD | RTLD_LAZY),
  };
  for (void* lib : libs) {
    if (lib != nullptr) {
      if (void* symbol = dlsym(lib, name)) {
        return symbol;
      }
    }
  }
  return dlsym(RTLD_DEFAULT, name);
}

GetCurrentStreamFn stable_get_current_stream() {
  static const auto fn = reinterpret_cast<GetCurrentStreamFn>(
      stable_runtime_symbol("aoti_torch_get_current_stream"));
  return fn;
}

StreamIdFn stable_stream_id() {
  static const auto fn = reinterpret_cast<StreamIdFn>(
      stable_runtime_symbol("aoti_torch_stream_id"));
  return fn;
}

DeleteStreamFn stable_delete_stream() {
  static const auto fn = reinterpret_cast<DeleteStreamFn>(
      stable_runtime_symbol("aoti_torch_delete_stream"));
  return fn;
}

void boxed_stream_probe(StableIValue* stack, uint64_t num_inputs,
                        uint64_t num_outputs) {
  (void)num_inputs;
  (void)num_outputs;
  const int64_t device_index = to<int64_t>(stack[0]);
  const auto get_stream = stable_get_current_stream();
  const auto stream_id_of = stable_stream_id();
  const auto drop_stream = stable_delete_stream();
  const bool have_shims = get_stream != nullptr && stream_id_of != nullptr &&
                          drop_stream != nullptr;
  int64_t shim_id = have_shims ? -1 : -3;
  StreamHandle handle = nullptr;
  if (have_shims &&
      get_stream(static_cast<int32_t>(device_index), &handle) == 0 &&
      handle != nullptr) {
    if (stream_id_of(handle, &shim_id) != 0) {
      shim_id = -2;
    }
    drop_stream(handle);
  }
  // Second read: the same shims the header's getCurrentStream() wrapper uses.
  // Two independent reads keep the (raw, stable) shape the Python side expects
  // and would expose a shim that answers differently the second time.
  int64_t stream_id = have_shims ? -1 : -3;
  StreamHandle stable_handle = nullptr;
  if (have_shims &&
      get_stream(static_cast<int32_t>(device_index), &stable_handle) == 0 &&
      stable_handle != nullptr) {
    if (stream_id_of(stable_handle, &stream_id) != 0) {
      stream_id = -2;
    }
    drop_stream(stable_handle);
  }
  stack[0] = from(shim_id);
  stack[1] = from(stream_id);
}
#endif  // FLA_STABLE_NO_DEBUG_PROBE

}  // namespace
