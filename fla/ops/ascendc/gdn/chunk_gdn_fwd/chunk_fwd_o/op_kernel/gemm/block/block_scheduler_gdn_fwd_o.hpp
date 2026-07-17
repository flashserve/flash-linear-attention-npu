/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_GEMM_SCHEDULER_GDN_FWD_O_HPP
#define CATLASS_GEMM_SCHEDULER_GDN_FWD_O_HPP

#include "../../chunk_fwd_o_struct.h"

// constexpr uint32_t PING_PONG_STAGES = 1;
constexpr uint32_t PING_PONG_STAGES = 2;
constexpr uint32_t BYTE_SIZE_16_BIT = 2;

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

namespace Catlass::Gemm::Block {


struct GDNFwdOOffsets {
    int64_t qkOffset;
    int64_t ovOffset;
    int64_t hOffset;
    int64_t gOffset;
    int64_t attnWorkOffset;
    int64_t hvWorkOffset;
    uint32_t vBlockOffset;
    uint32_t vBlockDim;
    bool isFinalState;
    uint32_t blockTokens;
    // for debug
    uint32_t batchIdx;
    uint32_t headIdx;
    uint32_t chunkIdx;

};

struct BlockSchedulerGdnFwdO {
    uint32_t shapeBatch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    uint32_t isVariedLen;
    uint32_t tokenBatch;
    uint32_t numChunks{0};
    uint32_t vBlockSize{128};

    uint32_t taskIdx;
    uint32_t cubeCoreIdx;
    uint32_t cubeCoreNum;
    uint32_t taskNum;
    uint32_t headGroups;

    bool isRunning;
    bool processNewTask {true};
    bool firstLoop {true};
    bool lastLoop {false};
    GDNFwdOOffsets offsets[PING_PONG_STAGES];
    int32_t currStage{PING_PONG_STAGES - 1};

    uint32_t baseTaskIdx;
    uint32_t chunkIdx;
    uint32_t headInnerIdx;
    uint32_t vHeadIdx;
    uint32_t kHeadIdx;
    uint32_t shapeBatchIdx;
    uint32_t tokenBatchIdx;

    uint32_t batchChunkIdx;
    uint32_t batchChunkStartIdx;
    uint32_t tokenOffset;
    uint32_t batchChunks;
    uint32_t batchTokens;

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmChunkOffsets;

    Arch::CrossCoreFlag cube1Done[PING_PONG_STAGES] = {0, 1};
    Arch::CrossCoreFlag vec1Done[PING_PONG_STAGES] = {2, 3};
    Arch::CrossCoreFlag cube3Done[PING_PONG_STAGES] = {4, 5};
    Arch::CrossCoreFlag vec2Done[PING_PONG_STAGES] = {6, 7};

    CATLASS_DEVICE
    BlockSchedulerGdnFwdO() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, const GDN::ChunkFwdOTilingData *tilingData,
              uint32_t coreIdx, uint32_t coreNum) {
        shapeBatch = tilingData->shapeBatch;
        seqlen = tilingData->seqlen;
        kNumHead = tilingData->kNumHead;
        vNumHead = tilingData->vNumHead;
        kHeadDim = tilingData->kHeadDim;
        vHeadDim = tilingData->vHeadDim;
        chunkSize = tilingData->chunkSize;
        isVariedLen = tilingData->isVariedLen;
        tokenBatch = tilingData->tokenBatch;

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmChunkOffsets.SetGlobalBuffer((__gm__ int64_t *)chunk_offsets);

        if (isVariedLen) {
            for (uint32_t b = 1; b <= tokenBatch; b++) {
                numChunks += (gmSeqlen.GetValue(b) - gmSeqlen.GetValue(b - 1) + chunkSize - 1) / chunkSize;
            }
        } else {
            numChunks = (seqlen + chunkSize - 1) / chunkSize;
        }

        cubeCoreIdx = coreIdx;
        cubeCoreNum = coreNum;
        vBlockSize = vHeadDim;
        taskNum = shapeBatch * numChunks * vNumHead;
        headGroups = vNumHead / kNumHead;
        taskIdx = cubeCoreIdx * PING_PONG_STAGES;
        isRunning = taskIdx < taskNum;

    }

    CATLASS_DEVICE
    void InitTask() {
        if (processNewTask) {
            headInnerIdx = 0;
            baseTaskIdx = taskIdx;
        } else {
            headInnerIdx = (headInnerIdx + 1) % PING_PONG_STAGES;
        }

        uint32_t curTaskIdx = baseTaskIdx + headInnerIdx;
        if (unlikely(curTaskIdx >= taskNum)) {
            isRunning = false;
            processNewTask = true;
            currStage = (currStage + 1) % PING_PONG_STAGES;
            return;
        }

        shapeBatchIdx = curTaskIdx / (numChunks * vNumHead);
        chunkIdx = (curTaskIdx - shapeBatchIdx * numChunks * vNumHead) / vNumHead;
        vHeadIdx = curTaskIdx % vNumHead;
        kHeadIdx = vHeadIdx / headGroups;
        tokenBatchIdx = isVariedLen ? gmChunkOffsets.GetValue(2 * chunkIdx) : 0;
        batchChunkIdx = isVariedLen ? gmChunkOffsets.GetValue(2 * chunkIdx + 1) : chunkIdx;
        batchChunkStartIdx = chunkIdx - batchChunkIdx;
        tokenOffset = isVariedLen ? gmSeqlen.GetValue(tokenBatchIdx) : 0;
        batchTokens = isVariedLen ? (gmSeqlen.GetValue(tokenBatchIdx + 1) - tokenOffset) : seqlen;
        uint32_t vBlockOffset = 0;
        uint32_t vBlockDim = vBlockSize;
        const int64_t tokenStart = static_cast<int64_t>(tokenOffset) +
                                   static_cast<int64_t>(batchChunkIdx) * chunkSize;
        const int64_t qkRowOffset = (static_cast<int64_t>(shapeBatchIdx) * kNumHead + kHeadIdx) * seqlen +
                                    tokenStart;
        const int64_t ovRowOffset = (static_cast<int64_t>(shapeBatchIdx) * vNumHead + vHeadIdx) * seqlen +
                                    tokenStart;
        const int64_t hBlockOffset = (static_cast<int64_t>(shapeBatchIdx) * vNumHead * numChunks +
                                      static_cast<int64_t>(vHeadIdx) * numChunks + chunkIdx) *
                                     kHeadDim;
        const int64_t workStageOffset = static_cast<int64_t>(cubeCoreIdx) * PING_PONG_STAGES + currStage;
        offsets[currStage].qkOffset = qkRowOffset * kHeadDim;
        offsets[currStage].ovOffset = ovRowOffset * vHeadDim + vBlockOffset;
        offsets[currStage].hOffset = hBlockOffset * vHeadDim + vBlockOffset;
        offsets[currStage].gOffset = ovRowOffset;
        offsets[currStage].attnWorkOffset = workStageOffset * chunkSize * chunkSize;
        offsets[currStage].hvWorkOffset = workStageOffset * chunkSize * vBlockSize;
        offsets[currStage].vBlockOffset = vBlockOffset;
        offsets[currStage].vBlockDim = vBlockDim;
        offsets[currStage].isFinalState = chunkIdx == (numChunks - 1) || (isVariedLen && gmChunkOffsets.GetValue(2 * chunkIdx + 3) == 0);
        offsets[currStage].blockTokens = offsets[currStage].isFinalState ? (batchTokens - batchChunkIdx * chunkSize) : chunkSize;
        offsets[currStage].batchIdx = shapeBatchIdx;
        offsets[currStage].headIdx = vHeadIdx;
        offsets[currStage].chunkIdx = chunkIdx;

        processNewTask = headInnerIdx == PING_PONG_STAGES - 1;
        if (processNewTask) {
            taskIdx += PING_PONG_STAGES * cubeCoreNum;
        }

        currStage = (currStage + 1) % PING_PONG_STAGES;
    }

    CATLASS_DEVICE
    uint32_t GetCurStageId() const {
        return (currStage + PING_PONG_STAGES - 1) % PING_PONG_STAGES;
    }

    CATLASS_DEVICE
    uint32_t GetPrevStageId() const {
        return (currStage + PING_PONG_STAGES - 2) % PING_PONG_STAGES;
    }


};

struct BlockSchedulerGdnFwdOCube : public BlockSchedulerGdnFwdO {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdOCube() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, const GDN::ChunkFwdOTilingData *tilingData) {
        BlockSchedulerGdnFwdO::Init(cu_seqlens, chunk_offsets, tilingData, AscendC::GetBlockIdx(),
                                    AscendC::GetBlockNum());
    }

    CATLASS_DEVICE
    bool NeedProcessCube1() {
        return true;
    }

    CATLASS_DEVICE
    GDNFwdOOffsets& GetCube1Offsets() {
        return offsets[GetCurStageId()];
    }

    CATLASS_DEVICE
    GemmCoord GetCube1Shape() {
        GDNFwdOOffsets& cube1Offsets = GetCube1Offsets();
        return GemmCoord{cube1Offsets.blockTokens, cube1Offsets.blockTokens, kHeadDim};
    }

    CATLASS_DEVICE
    bool NeedProcessCube23() {
        if (unlikely(firstLoop)) {
            firstLoop = false;
            return false;
        }
        return true;
    }

    CATLASS_DEVICE
    GDNFwdOOffsets& GetCube23Offsets() {
        return offsets[GetPrevStageId()];
    }

    CATLASS_DEVICE
    GemmCoord GetCube2Shape() {
        GDNFwdOOffsets& cube2Offsets = GetCube23Offsets();
        return GemmCoord{cube2Offsets.blockTokens, cube2Offsets.vBlockDim, kHeadDim};
    }

    CATLASS_DEVICE
    GemmCoord GetCube3Shape() {
        GDNFwdOOffsets& cube2Offsets = GetCube23Offsets();
        return GemmCoord{cube2Offsets.blockTokens, cube2Offsets.vBlockDim, cube2Offsets.blockTokens};
    }

};

struct BlockSchedulerGdnFwdOVec : public BlockSchedulerGdnFwdO {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdOVec() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, const GDN::ChunkFwdOTilingData *tilingData) {
        BlockSchedulerGdnFwdO::Init(cu_seqlens, chunk_offsets, tilingData,
                                    AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(), AscendC::GetBlockNum());
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
    GDNFwdOOffsets& GetVec1Offsets() {
        return offsets[GetCurStageId()];
    }

    CATLASS_DEVICE
    GDNFwdOOffsets& GetVec2Offsets() {
        return offsets[GetPrevStageId()];
    }

};

}  // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_SCHEDULER_GDN_FWD_O_HPP
