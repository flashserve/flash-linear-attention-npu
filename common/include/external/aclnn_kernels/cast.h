/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef COMMON_INC_EXTERNAL_ACLNN_KERNELS_CAST_H
#define COMMON_INC_EXTERNAL_ACLNN_KERNELS_CAST_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
const aclTensor* Cast(const aclTensor* self, op::DataType dstDtype, aclOpExecutor* executor);

// 专攻卷积反向定制
const aclTensor* CastOnlyForConvBackward(const aclTensor* self, op::DataType dstDtype, aclOpExecutor* executor);
} // namespace l0op

#endif // COMMON_INC_EXTERNAL_ACLNN_KERNELS_CAST_H
