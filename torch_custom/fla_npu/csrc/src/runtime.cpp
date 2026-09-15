#include "stable/runtime.h"

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace fla_npu_stable {

namespace {

// torch_npu's AOTI shim for "the stream this thread is currently on", the NPU
// twin of aoti_torch_get_current_cuda_stream.  It lives in libtorch_npu.so,
// which is not on the launcher's link line, so it is resolved by name.
constexpr const char* kStreamResolver = "aoti_torch_get_current_npu_stream";

void* dlopen_required(const std::string& path, const char* what) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    throw std::runtime_error(
        std::string(what) + ": dlopen failed for " + path + ": " + dlerror());
  }
  return handle;
}

// torch_npu is loaded by the time an operator runs (fla_npu imports it first),
// so the global namespace normally has the shim.  The dlopen fallback covers a
// host that loaded the extension with RTLD_LOCAL instead.
void* find_stream_resolver() {
  if (void* address = dlsym(RTLD_DEFAULT, kStreamResolver)) {
    return address;
  }
  static const char* const kLibraries[] = {"libtorch_npu.so",
                                           "libtorch_npu.so.2",
                                           "libtorch_npu.so.1"};
  for (const char* name : kLibraries) {
    void* handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      continue;
    }
    if (void* address = dlsym(handle, kStreamResolver)) {
      return address;
    }
  }
  return nullptr;
}

}  // namespace

Runtime& Runtime::instance() {
  static Runtime runtime;
  return runtime;
}

void Runtime::init(const std::string& custom_lib_path) {
  if (initialized_) {
    return;
  }
  std::string cust_path = custom_lib_path;
  if (cust_path.empty()) {
    const char* env = std::getenv("FLA_NPU_OP_API_LIB");
    if (env != nullptr && *env != '\0') {
      cust_path = env;
    }
  }
  if (cust_path.empty()) {
    throw std::runtime_error(
        "stable launcher: FLA_NPU_OP_API_LIB is not set; call "
        "fla_npu.load_ascendc_opapi_libraries() before using the stable launcher");
  }
  custom_handle_ = open_custom_library(cust_path);
  cann_handle_ = dlopen_required("libopapi.so", "CANN opapi");
  initialized_ = true;
}

void* Runtime::open_custom_library(const std::string& path) {
  return dlopen_required(path, "FLA custom opapi");
}

void* Runtime::symbol(const std::string& name) {
  init();
  auto it = cache_.find(name);
  if (it != cache_.end()) {
    return it->second;
  }
  void* addr = nullptr;
  if (custom_handle_ != nullptr) {
    addr = dlsym(custom_handle_, name.c_str());
  }
  if (addr == nullptr && cann_handle_ != nullptr) {
    addr = dlsym(cann_handle_, name.c_str());
  }
  if (addr == nullptr) {
    throw std::runtime_error("stable launcher: symbol not found: " + name);
  }
  cache_[name] = addr;
  return addr;
}

void* Runtime::stream_resolver() {
  if (!stream_resolver_looked_up_) {
    stream_resolver_ = find_stream_resolver();
    stream_resolver_looked_up_ = true;
  }
  return stream_resolver_;
}

bool Runtime::has_stream_resolver() { return stream_resolver() != nullptr; }

int64_t Runtime::current_stream(int32_t device_index) {
  using CurrentStreamFn = int32_t (*)(int32_t, void**);
  auto fn = reinterpret_cast<CurrentStreamFn>(stream_resolver());
  if (fn == nullptr) {
    throw std::runtime_error(
        "stable launcher: torch_npu's aoti_torch_get_current_npu_stream is not "
        "available, so the launcher cannot resolve the NPU stream itself; "
        "import torch_npu (fla_npu does this for you) or pass the stream "
        "explicitly");
  }
  void* stream = nullptr;
  const int32_t status = fn(device_index, &stream);
  if (status != 0) {
    throw std::runtime_error(
        "stable launcher: aoti_torch_get_current_npu_stream failed: " +
        std::to_string(status));
  }
  return reinterpret_cast<int64_t>(stream);
}

}  // namespace fla_npu_stable
