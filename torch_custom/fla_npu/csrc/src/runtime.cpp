#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "stable/runtime.h"

#include <dlfcn.h>
#include <link.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace fla_npu_stable {

namespace {

// torch_npu's entry point for "run this callable on the task queue", the same
// one its own inductor and mstx paths use.  Two spellings because the newer
// overload takes the callable by const reference and keeps the full operator
// name, while the older one takes it by value.
//
// This is what makes a submission that does not wait for the queue to drain
// ordering-safe: the callable runs on the queue's consumer thread, in queue
// order, so it is the queue -- not a host-side barrier -- that keeps our
// kernel behind everything the host enqueued before us.
constexpr const char* kQueueEnqueueV2 =
    "_ZN6at_npu6native9OpCommand10RunOpApiV2ERKNSt7__cxx1112basic_"
    "stringIcSt11char_traitsIcESaIcEEERKSt8functionIFivEEb";
constexpr const char* kQueueEnqueueV1 =
    "_ZN6at_npu6native9OpCommand8RunOpApiERKNSt7__cxx1112basic_"
    "stringIcSt11char_traitsIcESaIcEEESt8functionIFivEEb";

void* dlopen_required(const std::string& path, const char* what) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    throw std::runtime_error(
        std::string(what) + ": dlopen failed for " + path + ": " + dlerror());
  }
  return handle;
}

link_map* symbol_owner(void* address) {
  Dl_info info{};
  void* extra = nullptr;
  if (dladdr1(address, &info, &extra, RTLD_DL_LINKMAP) == 0 || extra == nullptr) {
    throw std::runtime_error("stable launcher: cannot identify ACLNN symbol owner");
  }
  return static_cast<link_map*>(extra);
}

void* own_symbol(void* handle, const std::string& name) {
  void* address = dlsym(handle, name.c_str());
  if (address == nullptr) {
    return nullptr;
  }
  link_map* library = nullptr;
  if (dlinfo(handle, RTLD_DI_LINKMAP, &library) != 0 || library == nullptr) {
    throw std::runtime_error("stable launcher: cannot identify custom opapi library");
  }
  // dlsym searches DT_NEEDED dependencies too. Their ACLNN definitions must not
  // jump ahead of the next FLA vendor. Compare loaded objects, not path strings.
  return symbol_owner(address) == library ? address : nullptr;
}

// torch_npu is loaded by the time an operator runs (fla_npu imports it first),
// so the global namespace normally has the symbol.  The dlopen fallback covers
// a host that loaded the extension with RTLD_LOCAL instead.
void* find_exported(const char* name) {
  if (void* address = dlsym(RTLD_DEFAULT, name)) {
    return address;
  }
  static const char* const kLibraries[] = {"libtorch_npu.so",
                                           "libtorch_npu.so.2",
                                           "libtorch_npu.so.1"};
  for (const char* library : kLibraries) {
    void* handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      continue;
    }
    if (void* address = dlsym(handle, name)) {
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

namespace {

// `os.pathsep`-separated path list, highest priority first.
std::vector<std::string> split_path_list(const char* value) {
  std::vector<std::string> paths;
  std::string current;
  for (const char* p = value;; ++p) {
    if (*p == ':' || *p == '\0') {
      if (!current.empty()) {
        paths.push_back(current);
      }
      current.clear();
      if (*p == '\0') {
        break;
      }
    } else {
      current.push_back(*p);
    }
  }
  return paths;
}

}  // namespace

bool Runtime::set_verified_base(const std::string& path) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (path.empty()) {
    return false;
  }
  if (path == verified_base_path_) {
    return true;
  }
  if (initialized_ || !verified_base_path_.empty()) {
    return false;
  }
  verified_base_path_ = path;
  return true;
}

void Runtime::init(const std::string& custom_lib_path) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (initialized_) {
    return;
  }
  std::vector<std::string> custom_paths;
  if (!custom_lib_path.empty()) {
    custom_paths.push_back(custom_lib_path);
  } else {
    // FLA_NPU_OP_API_LIBS (plural) is an os.pathsep-separated list, highest
    // priority first: overlay vendors first, the complete base library last.
    // FLA_NPU_OP_API_LIB (singular) stays supported for single-library setups.
    const char* plural = std::getenv("FLA_NPU_OP_API_LIBS");
    if (plural != nullptr && *plural != '\0') {
      custom_paths = split_path_list(plural);
    }
    if (custom_paths.empty()) {
      const char* env = std::getenv("FLA_NPU_OP_API_LIB");
      if (env != nullptr && *env != '\0') {
        custom_paths.push_back(env);
      }
    }
  }
  if (custom_paths.empty()) {
    throw std::runtime_error(
        "stable launcher: FLA_NPU_OP_API_LIB is not set; call "
        "fla_npu.load_ascendc_opapi_libraries() before using the stable launcher");
  }
  for (const std::string& path : custom_paths) {
    custom_handles_.push_back(open_custom_library(path));
  }
  // Only the explicitly verified, last vendor may use the original base/CANN
  // lookup. A lone arbitrary library is not evidence of a validated base.
  if (!verified_base_path_.empty() && custom_paths.back() == verified_base_path_ &&
      std::find(custom_handles_.begin(), custom_handles_.end() - 1,
                custom_handles_.back()) == custom_handles_.end() - 1) {
    verified_base_handle_ = custom_handles_.back();
  }
  cann_handle_ = dlopen_required("libopapi.so", "CANN opapi");
  initialized_ = true;
}

void* Runtime::open_custom_library(const std::string& path) {
  return dlopen_required(path, "FLA custom opapi");
}

void* Runtime::symbol(const std::string& name) {
  init();
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = cache_.find(name);
  if (it != cache_.end()) {
    return it->second;
  }
  void* addr = nullptr;
  // Ordered search: the first custom library that exports the symbol wins, so an
  // overlay vendor overrides the base vendor regardless of the backend choice.
  for (void* handle : custom_handles_) {
    if (handle == nullptr) {
      continue;
    }
    addr = name.compare(0, 5, "aclnn") == 0 && handle != verified_base_handle_
               ? own_symbol(handle, name)
               : dlsym(handle, name.c_str());
    if (addr != nullptr) {
      break;
    }
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

OpApi Runtime::op_api(const std::string& name) {
  init();
  std::lock_guard<std::mutex> guard(mutex_);
  auto found = op_cache_.find(name);
  if (found != op_cache_.end()) {
    return found->second;
  }
  const std::string workspace = name + "GetWorkspaceSize";
  for (void* handle : custom_handles_) {
    // A validated base contains complete pairs. Its dependencies are already
    // the terminal CANN fallback, so no per-interface ownership scan is needed.
    OpApi api = handle == verified_base_handle_
        ? OpApi{dlsym(handle, workspace.c_str()), dlsym(handle, name.c_str())}
        : OpApi{own_symbol(handle, workspace), own_symbol(handle, name)};
    if (api.get_workspace == nullptr && api.launch == nullptr) {
      continue;
    }
    if (api.get_workspace == nullptr || api.launch == nullptr) {
      throw std::runtime_error("stable launcher: incomplete custom ACLNN pair: " + name);
    }
    op_cache_[name] = api;
    return api;
  }
  OpApi api{dlsym(cann_handle_, workspace.c_str()), dlsym(cann_handle_, name.c_str())};
  if (api.get_workspace == nullptr || api.launch == nullptr) {
    throw std::runtime_error("stable launcher: missing CANN ACLNN pair: " + name);
  }
  if (symbol_owner(api.get_workspace) != symbol_owner(api.launch)) {
    throw std::runtime_error("stable launcher: CANN ACLNN phases have different providers: " + name);
  }
  op_cache_[name] = api;
  return api;
}

std::string Runtime::recent_error_message() {
  using GetError = const char* (*)();
  GetError get_error = nullptr;
  try {
    get_error = reinterpret_cast<GetError>(symbol("aclGetRecentErrMsg"));
  } catch (const std::exception&) {
    static void* acl = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
    if (acl != nullptr) {
      get_error = reinterpret_cast<GetError>(dlsym(acl, "aclGetRecentErrMsg"));
    }
  }
  if (get_error == nullptr) {
    return {};
  }
  const char* message = get_error();
  return message == nullptr ? std::string() : std::string(message);
}

void* Runtime::queue_enqueue() {
  if (!queue_enqueue_looked_up_) {
    queue_enqueue_ = find_exported(kQueueEnqueueV2);
    if (queue_enqueue_ != nullptr) {
      queue_enqueue_takes_value_ = false;
    } else {
      queue_enqueue_ = find_exported(kQueueEnqueueV1);
      queue_enqueue_takes_value_ = queue_enqueue_ != nullptr;
    }
    queue_enqueue_looked_up_ = true;
  }
  return queue_enqueue_;
}

bool Runtime::enqueue_enabled() {
  // Read once per process: the answer decides which stream accessor the Python
  // glue uses, so it must not change between two calls of the same shape.
  static const bool enabled = []() -> bool {
    const char* requested = std::getenv("FLA_NPU_STABLE_LAUNCH");
    if (requested != nullptr && std::string(requested) == "inline") {
      return false;
    }
    return Runtime::instance().queue_enqueue() != nullptr;
  }();
  return enabled;
}

void Runtime::enqueue(const std::string& name, const std::function<int()>& fn) {
  void* entry = queue_enqueue();
  if (entry == nullptr) {
    throw std::runtime_error(
        "stable launcher: torch_npu's task queue entry point is not available");
  }
  if (queue_enqueue_takes_value_) {
    using EnqueueByValueFn = void (*)(const std::string&, std::function<int()>,
                                      bool);
    std::function<int()> copy = fn;  // the callee moves out of this one
    reinterpret_cast<EnqueueByValueFn>(entry)(name, std::move(copy), false);
    return;
  }
  using EnqueueFn = void (*)(const std::string&, const std::function<int()>&,
                             bool);
  reinterpret_cast<EnqueueFn>(entry)(name, fn, false);
}

}  // namespace fla_npu_stable

// Version 1 guarantees overlay -> base -> CANN lookup in this launcher.
extern "C" __attribute__((visibility("default")))
std::int32_t fla_npu_stable_overlay_order_version() {
  return 1;
}

// This is a loader handshake, not an environment switch. Unregistered native
// callers retain strict ownership checks even when only one library is loaded.
extern "C" __attribute__((visibility("default")))
std::int32_t fla_npu_stable_set_verified_base(const char* path) noexcept {
  try {
    return path != nullptr && fla_npu_stable::Runtime::instance().set_verified_base(path) ? 0 : 1;
  } catch (...) {
    return 1;
  }
}
