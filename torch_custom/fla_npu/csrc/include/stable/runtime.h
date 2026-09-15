// Runtime symbol resolution for the launcher (no torch_npu headers).
//
// `dlopen`s the FLA custom `op_api` library and CANN's libopapi, and resolves
// `aclnn*` entry points with a per-name cache.  It lives with the Stable-ABI
// launcher because that is the one that ships.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace fla_npu_stable {

class Runtime {
 public:
  static Runtime& instance();

  // dlopen the FLA custom op_api library (FLA_NPU_OP_API_LIB or explicit path)
  // and CANN libopapi.so. Safe to call multiple times.
  void init(const std::string& custom_lib_path = "");

  // Custom-first symbol lookup with per-name cache. Throws std::runtime_error
  // when a symbol cannot be resolved.
  void* symbol(const std::string& name);

  // Raw aclrtStream of the calling thread on `device_index`.
  //
  // Resolved through torch_npu's AOTI shim, which hands back the very pointer
  // torch_npu's own `_npu_getCurrentRawStream` returns; the device-generic
  // `aoti_torch_get_current_stream` does not, because an NPU `c10::Stream::id()`
  // is not the underlying handle.  The lookup is lazy and every call really
  // queries the runtime: caching the pointer in a process-global is what sent
  // kernels to another thread's stream in the vLLM run.
  //
  // Throws when the shim is not reachable; `has_stream_resolver()` tells the
  // callers (the Python glue, once at load) whether they may use this at all.
  int64_t current_stream(int32_t device_index);

  // Whether current_stream() can do its job.  Deliberately does not `init()`,
  // so it stays callable before FLA_NPU_OP_API_LIB is set.
  bool has_stream_resolver();

 private:
  Runtime() = default;
  void* open_custom_library(const std::string& path);
  void* stream_resolver();

  void* custom_handle_ = nullptr;
  void* cann_handle_ = nullptr;
  bool initialized_ = false;
  std::unordered_map<std::string, void*> cache_;

  void* stream_resolver_ = nullptr;
  bool stream_resolver_looked_up_ = false;
};

}  // namespace fla_npu_stable
