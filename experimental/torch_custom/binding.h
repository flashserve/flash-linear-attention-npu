// Bind owned ACLNN symbols to the explicitly selected OPP libraries once.
#pragma once
#include "stable/runtime.h"
#include <cstdlib>
#include <dlfcn.h>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

inline std::string canonical_library(const char* path) {
  char* resolved = realpath(path, nullptr);
  if (!resolved) throw std::runtime_error("Cannot resolve OPP library path");
  std::string result(resolved);
  free(resolved);
  return result;
}

inline void bind_experimental_symbols(const std::string& paths,
                                     std::initializer_list<const char*> symbols) {
  std::unordered_set<std::string> allowed;
  std::istringstream stream(paths);
  std::string path;
  while (std::getline(stream, path, ':')) {
    if (!path.empty()) allowed.insert(canonical_library(path.c_str()));
  }
  if (allowed.empty()) throw std::runtime_error("Explicit OPP paths are required");
  auto& runtime = fla_npu_stable::Runtime::instance();
  runtime.init(paths);
  for (const char* symbol : symbols) {
    void* address = runtime.symbol(symbol);
    Dl_info origin{};
    if (!dladdr(address, &origin) || !origin.dli_fname ||
        !allowed.count(canonical_library(origin.dli_fname))) {
      throw std::runtime_error(std::string("Selected OPP is missing owned symbol: ") + symbol);
    }
  }
}
