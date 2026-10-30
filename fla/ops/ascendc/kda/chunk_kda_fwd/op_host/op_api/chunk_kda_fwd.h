/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef OP_API_INC_LEVEL0_OP_CHUNK_KDA_FWD_OP_H
#define OP_API_INC_LEVEL0_OP_CHUNK_KDA_FWD_OP_H

#include <array>
#include "opdev/op_executor.h"

namespace l0op {
using KdaCoreOutputs = std::array<const aclTensor *, 13>;

KdaCoreOutputs KdaChunkForward(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g, const aclTensor *beta,
    const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, double scale, int64_t chunkSize,
    bool safeGate, bool inputSequenceMajor, bool useGateInKernel, double lowerBound,
    const aclTensor *attnOut,
    const aclTensor *finalStateOut, const aclTensor *gkOut, const aclTensor *aqkOut,
    const aclTensor *akkOut, const aclTensor *wOut, const aclTensor *uOut, const aclTensor *qgOut,
    const aclTensor *kgOut, const aclTensor *vNewOut, const aclTensor *hOut,
    const aclTensor *qgScaledOut, const aclTensor *uSeedOut, int64_t stage,
    aclOpExecutor *executor);

// Ascend950 chunk32 分支；epsilon 在 tiling 中固定为 1e-6。
const aclTensor *KdaChunkForward32(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g,
    const aclTensor *beta, const aclTensor *aLog, const aclTensor *dtBias,
    const aclTensor *initialState, double scale, double lowerBound,
    const aclTensor *o, const aclTensor *finalState, const aclIntArray *cuSeqlens,
    aclOpExecutor *executor, const char *layout = "TND");
} // namespace l0op

#endif
