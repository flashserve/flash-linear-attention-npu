/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef OP_API_INC_LEVEL0_OP_PRE_PROCESS_FWD_KERNEL_MERGED_OP_H
#define OP_API_INC_LEVEL0_OP_PRE_PROCESS_FWD_KERNEL_MERGED_OP_H

#include "opdev/op_executor.h"

namespace l0op {
// 参数顺序与 aclnn / ctypes ARGTYPES 逐参一致；hmOut 为 [Nseq,Hv,K,V+K] FP32。
const std::array<const aclTensor *, 1> PreProcessFwdKernelMerged(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *g,
    const aclTensor *gk,
    const aclTensor *bg,
    const aclTensor *v,
    const aclIntArray *cuSeqlensOptional,
    int64_t chunkSize,
    const aclTensor *hmOut,
    aclOpExecutor *executor);
} // namespace l0op

#endif
