/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file device_op_impl_registry_impl.h
 * \brief
 */

#ifndef OP_TILING_DEVICE_OP_IMPL_REGISTRY_IMPL_H
#define OP_TILING_DEVICE_OP_IMPL_REGISTRY_IMPL_H

#include <string>
#include <map>
#include "register/device_op_impl_registry.h"

namespace optiling {
class DeviceOpImplRegistry {
 public:
  static DeviceOpImplRegistry& GetSingleton();
  void RegisterSinkTiling(std::string &opType, SinkTilingFunc& func);
  SinkTilingFunc GetSinkTilingFunc(std::string &opType);

 private:
  DeviceOpImplRegistry() = default;
  ~DeviceOpImplRegistry() = default;

 private:
  std::map<std::string, SinkTilingFunc> sinkTilingFuncsMap_;
};

class DeviceOpImplRegisterImpl {
 public:
  DeviceOpImplRegisterImpl() = default;
  ~DeviceOpImplRegisterImpl();
  std::string& GetOpType();

 private:
  std::string opType_ = "";
};
}  // namespace optiling

#endif