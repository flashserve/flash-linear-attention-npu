/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef OP_API_INC_LEVEL0_OP_CHUNK_KDA_FWD_FUSED_OP_H
#define OP_API_INC_LEVEL0_OP_CHUNK_KDA_FWD_FUSED_OP_H

#include "opdev/op_executor.h"

namespace l0op {

// Ascend950 chunk32 单 kernel 大融合；epsilon 在 tiling 中固定为 1e-6。
// 产出 attn_out 与可选 final_state，不导出中间量。
const aclTensor *KdaChunkForwardFused(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g,
    const aclTensor *beta, const aclTensor *aLog, const aclTensor *dtBias,
    const aclTensor *initialState, double scale, double lowerBound,
    const aclTensor *o, const aclTensor *finalState, const aclIntArray *cuSeqlens,
    aclOpExecutor *executor, const char *layout = "TND");

} // namespace l0op

#endif
