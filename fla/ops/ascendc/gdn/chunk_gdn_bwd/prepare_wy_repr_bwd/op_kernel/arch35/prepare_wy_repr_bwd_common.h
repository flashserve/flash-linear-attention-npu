/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_common.h
 * \brief Common helpers for fused prepare_wy_repr_bwd A5 kernel.
 */

#ifndef PREPARE_WY_REPR_BWD_COMMON_H
#define PREPARE_WY_REPR_BWD_COMMON_H

#include "prepare_wy_repr_bwd_struct.h"

constexpr uint64_t PREPARE_WY_REPR_BWD_VEC_TO_CUBE_FLAG_READY = 2;
constexpr uint64_t PREPARE_WY_REPR_BWD_CUBE_TO_VEC_FLAG_READY = 4;
constexpr uint64_t PRONE_BLOCK_BYTES_32 = 32;
constexpr uint32_t K_DIM = 128;
constexpr uint32_t UB_BYTES_16K = 16 * 1024;
constexpr uint32_t BUFFER_COUNT_2 = 2;
constexpr uint32_t BUFFER_COUNT_4 = 4;
constexpr uint32_t PREPARE_WY_REPR_BWD_UB_PING_PONG_COUNT = 2;
constexpr uint32_t PREPARE_WY_REPR_BWD_FP32_PER_REPEAT = 64;
constexpr uint32_t PREPARE_WY_REPR_BWD_MASK_BITS_PER_BYTE = 8;

struct PrepareWyReprBwdTaskInfo {
    uint32_t valueBos = 0;
    uint32_t valueEos = 0;
    uint32_t keyBos = 0;
    uint32_t curChunkSize = 0;
};

__aicore__ inline uint64_t PrepareWyReprBwdCeilDiv(uint64_t dividend, uint64_t divisor)
{
    if (unlikely(divisor == 0)) {
        return 0;
    }
    return (dividend + divisor - 1) / divisor;
}

__aicore__ inline uint64_t Align32(uint64_t bytes)
{
    return (bytes + PRONE_BLOCK_BYTES_32 - 1) / PRONE_BLOCK_BYTES_32 *
           PRONE_BLOCK_BYTES_32;
}

__aicore__ inline void PrepareWyReprBwdGetTaskInfo(GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                                   const GDN::PrepareWyReprBwdTilingData &tiling, uint32_t taskIdx,
                                                   PrepareWyReprBwdTaskInfo &task)
{
    if (cuSeqlens == nullptr) {
        uint32_t chunkNumPerB = static_cast<uint32_t>(tiling.chunkNumPerB);
        if (chunkNumPerB == 0) {
            chunkNumPerB = static_cast<uint32_t>(PrepareWyReprBwdCeilDiv(tiling.T, tiling.chunkSize));
        }
        uint32_t chunkIdx = taskIdx % chunkNumPerB;
        uint32_t batchIdx = taskIdx / chunkNumPerB;
        uint32_t timeBos = chunkIdx * static_cast<uint32_t>(tiling.chunkSize);
        uint32_t timeEos = timeBos + static_cast<uint32_t>(tiling.chunkSize) > tiling.T ?
                               static_cast<uint32_t>(tiling.T) :
                               timeBos + static_cast<uint32_t>(tiling.chunkSize);
        task.valueBos = batchIdx * static_cast<uint32_t>(tiling.HV * tiling.T) + timeBos;
        task.valueEos = batchIdx * static_cast<uint32_t>(tiling.HV * tiling.T) + timeEos;
        task.keyBos = batchIdx * static_cast<uint32_t>(tiling.HK * tiling.T) + timeBos;
    } else {
        AscendC::GlobalTensor<uint64_t> cuSeqlensTensor;
        AscendC::GlobalTensor<uint64_t> chunkIndicesTensor;
        cuSeqlensTensor.SetGlobalBuffer((__gm__ uint64_t *)cuSeqlens);
        chunkIndicesTensor.SetGlobalBuffer((__gm__ uint64_t *)chunkIndices);
        uint32_t seqIdx = chunkIndicesTensor.GetValue(2 * taskIdx);
        uint32_t chunkIdx = chunkIndicesTensor.GetValue(2 * taskIdx + 1);
        uint32_t seqBegin = cuSeqlensTensor.GetValue(seqIdx);
        uint32_t seqEnd = cuSeqlensTensor.GetValue(seqIdx + 1);
        uint32_t timeBos = seqBegin + chunkIdx * static_cast<uint32_t>(tiling.chunkSize);
        uint32_t timeEos = timeBos + static_cast<uint32_t>(tiling.chunkSize) > seqEnd ?
                               seqEnd :
                               timeBos + static_cast<uint32_t>(tiling.chunkSize);
        task.valueBos = timeBos;
        task.valueEos = timeEos;
        task.keyBos = timeBos;
    }
    task.curChunkSize = task.valueEos - task.valueBos;
}

__aicore__ inline GM_ADDR PrepareWyReprBwdGetSlotBase(GM_ADDR workspace, uint32_t coreIdx, uint32_t curSlot,
                                                      const GDN::PrepareWyReprBwdTilingData &tiling)
{
    uint64_t workspaceCoreSize = tiling.workspaceCoreSize;
    if (workspaceCoreSize == 0) {
        workspaceCoreSize =
            tiling.workspaceBufferCount * tiling.workspaceSlotSize + tiling.workspaceBufferCount * tiling.mBytes;
    }
    return workspace + coreIdx * workspaceCoreSize + curSlot * tiling.workspaceSlotSize;
}

__aicore__ inline GM_ADDR PrepareWyReprBwdGetKktBase(GM_ADDR workspace, uint32_t coreIdx, uint32_t kktSlot,
                                                     const GDN::PrepareWyReprBwdTilingData &tiling)
{
    uint64_t workspaceCoreSize = tiling.workspaceCoreSize;
    if (workspaceCoreSize == 0) {
        workspaceCoreSize =
            tiling.workspaceBufferCount * tiling.workspaceSlotSize + tiling.workspaceBufferCount * tiling.mBytes;
    }
    return workspace + coreIdx * workspaceCoreSize + tiling.kktOffset + kktSlot * tiling.mBytes;
}

__aicore__ inline uint64_t PrepareWyReprBwdGetGroupSize(const GDN::PrepareWyReprBwdTilingData &tiling)
{
    if (tiling.groupSize > 0) {
        return tiling.groupSize;
    }
    return tiling.HV / tiling.HK;
}

#endif // PREPARE_WY_REPR_BWD_COMMON_H
