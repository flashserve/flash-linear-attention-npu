/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef AICPU_ADD_EXAMPLE_CPU_KERNELS_H_
#define AICPU_ADD_EXAMPLE_CPU_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class AddExampleCpuKernel : public CpuKernel {
 public:
  ~AddExampleCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;
  template<typename T>
  uint32_t AddCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
