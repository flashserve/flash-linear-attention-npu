/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_fwd_h_common.h
 * \brief Shared constants, helpers, and schedulers for chunk_gated_delta_rule_fwd_h
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_H_COMMON_H
#define CHUNK_GATED_DELTA_RULE_FWD_H_COMMON_H

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"

constexpr uint32_t PING_PONG_STAGES = 2;

template <typename T>
CATLASS_DEVICE T AlignUp(T a, T b) {
    return (b == 0) ? 0 : (a + b - 1) / b * b;
}

template <typename T>
CATLASS_DEVICE T Min(T a, T b) {
    return (a > b) ? b : a;
}

template <typename T>
CATLASS_DEVICE T Max(T a, T b) {
    return (a > b) ? a : b;
}

struct GDNFwdHOffsets {
    uint32_t hSrcOffset;
    uint32_t hDstOffset;
    uint32_t uvOffset;
    uint32_t wkOffset;
    uint32_t wOffset;
    uint32_t gOffset;
    uint32_t hWorkOffset;
    uint32_t vWorkOffset;
    bool isFinalState;
    uint32_t blockTokens;
    bool isDummyHead;
    uint32_t batchIdx;
    uint32_t headIdx;
    uint32_t chunkIdx;
};

struct BlockSchedulerGdnFwdH {
    uint32_t batch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    uint32_t vBlockSize{128};
    uint32_t isVariedLen;
    uint32_t shapeBatch;
    uint32_t tokenBatch;

    uint32_t taskIdx;
    uint32_t taskLoops;
    uint32_t cubeCoreIdx;
    uint32_t cubeCoreNum;
    uint32_t vLoops;
    uint32_t taskNum;
    uint32_t headGroups;
    uint32_t totalChunks;
    uint32_t totalTokens;
    uint32_t headInnerLoop;

    bool hasDummyHead;
    bool isRunning;
    bool processNewTask {true};
    bool firstLoop {true};
    bool lastLoop {false};
    GDNFwdHOffsets offsets[PING_PONG_STAGES];
    int32_t currStage{PING_PONG_STAGES - 1};

    uint32_t vIdx;
    uint32_t batchIdx;
    uint32_t baseHeadIdx;
    uint32_t chunkIdx;
    uint32_t headInnerIdx;
    uint32_t vHeadIdx;
    uint32_t kHeadIdx;
    uint32_t shapeBatchIdx;
    uint32_t tokenBatchIdx;

    uint32_t chunkOffset;
    uint32_t tokenOffset;
    uint32_t batchChunks;
    uint32_t batchTokens;

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumChunks;

    Catlass::Arch::CrossCoreFlag cube1Done{0};
    Catlass::Arch::CrossCoreFlag vec1Done{1};
    Catlass::Arch::CrossCoreFlag cube2Done{2};
    Catlass::Arch::CrossCoreFlag vec2Done{3};

    CATLASS_DEVICE
    BlockSchedulerGdnFwdH() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling, uint32_t coreIdx, uint32_t coreNum) {
        __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

        batch = gdnFwdHTilingData->batch;
        seqlen = gdnFwdHTilingData->seqlen;
        kNumHead = gdnFwdHTilingData->kNumHead;
        vNumHead = gdnFwdHTilingData->vNumHead;
        kHeadDim = gdnFwdHTilingData->kHeadDim;
        vHeadDim = gdnFwdHTilingData->vHeadDim;
        chunkSize = gdnFwdHTilingData->chunkSize;
        isVariedLen = gdnFwdHTilingData->isVariedLen;
        shapeBatch = gdnFwdHTilingData->shapeBatch;
        tokenBatch = gdnFwdHTilingData->tokenBatch;

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)chunk_indices);

        cubeCoreIdx = coreIdx;
        cubeCoreNum = coreNum;
        vLoops = vHeadDim / vBlockSize;
        taskNum = vLoops * batch * vNumHead;
        headGroups = vNumHead / kNumHead;
        hasDummyHead = taskNum % (PING_PONG_STAGES * cubeCoreNum) <= cubeCoreNum;
        taskLoops = (taskNum + cubeCoreNum * PING_PONG_STAGES - 1) / (cubeCoreNum * PING_PONG_STAGES);
        headInnerLoop = taskNum > cubeCoreNum ? PING_PONG_STAGES : 1;
        taskIdx = cubeCoreIdx * headInnerLoop;
        isRunning = taskIdx < taskNum;

        if (isVariedLen) {
            for (uint32_t b = 1; b <= tokenBatch; b++) {
                int64_t batchChunk = (gmSeqlen.GetValue(b) - gmSeqlen.GetValue(b - 1) + chunkSize - 1) / chunkSize;
                gmNumChunks.SetValue(b, gmNumChunks.GetValue(b - 1) + batchChunk);
            }
            totalChunks = gmNumChunks.GetValue(tokenBatch);
            totalTokens = gmSeqlen.GetValue(tokenBatch);
        } else {
            totalChunks = (seqlen + chunkSize - 1) / chunkSize;
            totalTokens = seqlen;
        }
    }

    CATLASS_DEVICE
    void InitTask() {
        if (unlikely(processNewTask)) {
            vIdx = taskIdx / (batch * vNumHead);
            batchIdx = (taskIdx - vIdx * batch * vNumHead) / vNumHead;
            baseHeadIdx = taskIdx % vNumHead;
            shapeBatchIdx = isVariedLen ? 0 : batchIdx;
            tokenBatchIdx = isVariedLen ? batchIdx : 0;
            chunkOffset = isVariedLen ? gmNumChunks.GetValue(tokenBatchIdx) : 0;
            batchChunks = isVariedLen ? (gmNumChunks.GetValue(tokenBatchIdx + 1) - chunkOffset) : totalChunks;
            tokenOffset = isVariedLen ? gmSeqlen.GetValue(tokenBatchIdx) : 0;
            batchTokens = isVariedLen ? (gmSeqlen.GetValue(tokenBatchIdx + 1) - tokenOffset) : totalTokens;
            chunkIdx = 0;
            headInnerIdx = 0;
        } else {
            chunkIdx = headInnerIdx == PING_PONG_STAGES - 1 ? chunkIdx + 1 : chunkIdx;
            headInnerIdx = (headInnerIdx + 1) % PING_PONG_STAGES;
        }

        vHeadIdx = baseHeadIdx + headInnerIdx;
        kHeadIdx = vHeadIdx / headGroups;
        offsets[currStage].hSrcOffset = (shapeBatchIdx * vNumHead * totalChunks + vHeadIdx * totalChunks + chunkOffset + chunkIdx) * kHeadDim * vHeadDim;
        offsets[currStage].hDstOffset = offsets[currStage].hSrcOffset + kHeadDim * vHeadDim;
        offsets[currStage].uvOffset = (shapeBatchIdx * vNumHead * totalTokens + vHeadIdx * totalTokens + tokenOffset + chunkIdx * chunkSize) * vHeadDim;
        offsets[currStage].wkOffset = (shapeBatchIdx * kNumHead * totalTokens + kHeadIdx * totalTokens + tokenOffset + chunkIdx * chunkSize) * kHeadDim;
        offsets[currStage].wOffset = (shapeBatchIdx * vNumHead * totalTokens + vHeadIdx * totalTokens + tokenOffset + chunkIdx * chunkSize) * kHeadDim;
        offsets[currStage].gOffset = shapeBatchIdx * vNumHead * totalTokens + vHeadIdx * totalTokens + tokenOffset + chunkIdx * chunkSize;
        offsets[currStage].hWorkOffset = (cubeCoreIdx * PING_PONG_STAGES + currStage) * kHeadDim * vHeadDim;
        offsets[currStage].vWorkOffset = (cubeCoreIdx * PING_PONG_STAGES + currStage) * chunkSize * vHeadDim;
        offsets[currStage].isFinalState = chunkIdx == (batchChunks - 1);
        offsets[currStage].blockTokens = offsets[currStage].isFinalState ? (batchTokens - chunkIdx * chunkSize) : chunkSize;
        offsets[currStage].isDummyHead = headInnerLoop < PING_PONG_STAGES && headInnerIdx >= headInnerLoop;
        offsets[currStage].batchIdx = batchIdx;
        offsets[currStage].headIdx = vHeadIdx;
        offsets[currStage].chunkIdx = chunkIdx;

        processNewTask = chunkIdx == batchChunks - 1 && headInnerIdx == PING_PONG_STAGES - 1;
        if (unlikely(processNewTask)) {
            uint32_t currLoopIdx = taskIdx / (PING_PONG_STAGES * cubeCoreNum);
            headInnerLoop = ((currLoopIdx + 2 == taskLoops) && hasDummyHead) ? 1 : PING_PONG_STAGES;
            taskIdx = (currLoopIdx + 1) * PING_PONG_STAGES * cubeCoreNum + headInnerLoop * cubeCoreIdx;
            if (unlikely(taskIdx >= taskNum)) {
                isRunning = false;
            }
        }

        currStage = (currStage + 1) % PING_PONG_STAGES;
    }
};

struct BlockSchedulerGdnFwdHCube : public BlockSchedulerGdnFwdH {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdHCube() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling) {
        BlockSchedulerGdnFwdH::Init(cu_seqlens, chunk_indices, tiling, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    }

    CATLASS_DEVICE
    bool NeedProcessCube1() {
        return true;
    }

    CATLASS_DEVICE
    GDNFwdHOffsets& GetCube1Offsets() {
        return offsets[(currStage - 1) % PING_PONG_STAGES];
    }

    CATLASS_DEVICE
    bool NeedProcessCube2() {
        if (unlikely(firstLoop)) {
            firstLoop = false;
            return false;
        }
        return true;
    }

    CATLASS_DEVICE
    GDNFwdHOffsets& GetCube2Offsets() {
        return offsets[(currStage - 2) % PING_PONG_STAGES];
    }
};

struct BlockSchedulerGdnFwdHVec : public BlockSchedulerGdnFwdH {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdHVec() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling) {
        BlockSchedulerGdnFwdH::Init(cu_seqlens, chunk_indices, tiling, AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(), AscendC::GetBlockNum());
    }

    CATLASS_DEVICE
    bool NeedProcessVec1() {
        return isRunning;
    }

    CATLASS_DEVICE
    bool NeedProcessVec2() {
        if (unlikely(firstLoop)) {
            firstLoop = false;
            return false;
        }
        return true;
    }

    CATLASS_DEVICE
    GDNFwdHOffsets& GetVec1Offsets() {
        return offsets[(currStage - 1) % PING_PONG_STAGES];
    }

    CATLASS_DEVICE
    GDNFwdHOffsets& GetVec2Offsets() {
        return offsets[(currStage - 2) % PING_PONG_STAGES];
    }
};

#endif // CHUNK_GATED_DELTA_RULE_FWD_H_COMMON_H
