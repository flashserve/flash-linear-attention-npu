/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef OP_API_INC_LEVEL0_OP_GATHER_V2_H_
#define OP_API_INC_LEVEL0_OP_GATHER_V2_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor *GatherV2(const aclTensor *self, int64_t axis, const aclTensor *indices, aclOpExecutor *executor,
                          int batchDims = 0, bool negativeIndexSupport = false);

const aclTensor *GatherV2WithImplMode(const aclTensor *self, int64_t axis, const aclTensor *indices, int64_t implMode,
                                      aclOpExecutor *executor, int batchDims = 0, bool negativeIndexSupport = false);

}

#endif // OP_API_INC_LEVEL0_OP_GATHER_V2_H_
