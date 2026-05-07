/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 * The code snippet comes from Huawei's open-source Ascend project.
 * Copyright 2021 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#include "add_example_aicpu.h"

#include <cmath>
#include <string>
#include "log.h"

namespace {
const char* const kAddExample = "AddExample";
const uint32_t kFirstInputIndex = 0;
const uint32_t kSecondInputIndex = 1;
const uint32_t kFirstOutputIndex = 0;
const uint32_t kSuccess = 0;
const uint32_t kParamInvalid = 1;
const uint32_t kError = 2;
}  // namespace

namespace aicpu {
uint32_t AddExampleCpuKernel::Compute(CpuKernelContext& ctx) {
  Tensor* input0 = ctx.Input(kFirstInputIndex);
  Tensor* input1 = ctx.Input(kSecondInputIndex);
  Tensor* output = ctx.Output(0);

  if (input0 == nullptr || input1 == nullptr || output == nullptr) {
    KERNEL_LOG_ERROR("Invalid argument");
    return kParamInvalid;
  }

  if (input0->GetDataSize() == 0 || input1->GetDataSize() == 0) {
    return kSuccess;
  }

  auto data_type = static_cast<DataType>(input0->GetDataType());
  switch (data_type) {
    case DT_FLOAT:
      return AddCompute<float>(ctx);
    case DT_INT32:
      return AddCompute<int32_t>(ctx);
    default:
      return kParamInvalid;
  }
  return kSuccess;
}

template <typename T>
uint32_t AddExampleCpuKernel::AddCompute(CpuKernelContext& ctx) {
  Tensor* input0 = ctx.Input(kFirstInputIndex);
  Tensor* input1 = ctx.Input(kSecondInputIndex);
  Tensor* output = ctx.Output(kFirstOutputIndex);

  T* x0 = reinterpret_cast<T*>(input0->GetData());
  if (x0 == nullptr) {
    return kParamInvalid;
  }
  T* x1 = reinterpret_cast<T*>(input1->GetData());
  if (x1 == nullptr) {
    return kParamInvalid;
  }
  T* y = reinterpret_cast<T*>(output->GetData());
  if (y == nullptr) {
    return kParamInvalid;
  }

  int64_t num_elements = input0->NumElements();
  KERNEL_LOG_INFO("Num of elements: %ld", num_elements);
  for (int64_t i = 0; i < num_elements; i++) {
    y[i] = x0[i] + x1[i];
  }
  return kSuccess;
}

REGISTER_CPU_KERNEL(kAddExample, AddExampleCpuKernel);

}  // namespace aicpu