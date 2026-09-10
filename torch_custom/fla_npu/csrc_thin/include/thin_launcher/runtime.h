// fla_npu C++ thin launcher: runtime symbol resolution (no torch_npu headers).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace fla_npu_thin {

class Runtime {
 public:
  static Runtime& instance();

  // dlopen the FLA custom op_api library (FLA_NPU_OP_API_LIB or explicit path)
  // and CANN libopapi.so. Safe to call multiple times.
  void init(const std::string& custom_lib_path = "");

  // Custom-first symbol lookup with per-name cache. Throws std::runtime_error
  // when a symbol cannot be resolved.
  void* symbol(const std::string& name);

 private:
  Runtime() = default;
  void* open_custom_library(const std::string& path);

  void* custom_handle_ = nullptr;
  void* cann_handle_ = nullptr;
  bool initialized_ = false;
  std::unordered_map<std::string, void*> cache_;
};

}  // namespace fla_npu_thin
