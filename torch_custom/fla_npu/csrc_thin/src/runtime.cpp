#include "thin_launcher/runtime.h"

#include <dlfcn.h>

#include <cstdlib>
#include <stdexcept>

namespace fla_npu_thin {

namespace {

void* dlopen_required(const std::string& path, const char* what) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    throw std::runtime_error(
        std::string(what) + ": dlopen failed for " + path + ": " + dlerror());
  }
  return handle;
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
        "thin launcher: FLA_NPU_OP_API_LIB is not set; call "
        "fla_npu.load_ascendc_opapi_libraries() before using the thin launcher");
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
    throw std::runtime_error("thin launcher: symbol not found: " + name);
  }
  cache_[name] = addr;
  return addr;
}

}  // namespace fla_npu_thin
