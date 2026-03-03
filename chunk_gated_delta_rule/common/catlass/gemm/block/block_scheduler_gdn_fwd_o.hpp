/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_SCHEDULER_GDN_FWD_O_HPP
#define CATLASS_GEMM_SCHEDULER_GDN_FWD_O_HPP

constexpr uint32_t PING_PONG_STAGES = 1;
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
    uint32_t qkOffset;
    uint32_t ovOffset;
    uint32_t hOffset;
    uint32_t gOffset;
    uint32_t attnWorkOffset;
    uint32_t hvWorkOffset;
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
    uint32_t numChunks;
    uint32_t vBlockSize{128};

    uint32_t taskIdx;
    uint32_t cubeCoreIdx;
    uint32_t cubeCoreNum;
    uint32_t vLoops;
    uint32_t taskNum;
    uint32_t headGroups;

    bool isRunning;
    bool processNewTask {true};
    bool firstLoop {true};
    bool lastLoop {false};
    GDNFwdOOffsets offsets[PING_PONG_STAGES];
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
    AscendC::GlobalTensor<int64_t> gmChunklen;

    Arch::CrossCoreFlag cube1Done{3};
    Arch::CrossCoreFlag vec1Done{4};
    Arch::CrossCoreFlag cube2Done{5};
    Arch::CrossCoreFlag cube3Done{6};
    Arch::CrossCoreFlag vec2Done{7};

    CATLASS_DEVICE
    BlockSchedulerGdnFwdO() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR tiling, uint32_t coreIdx, uint32_t coreNum) {
        __gm__ ChunkFwdOTilingData *__restrict gdnFwdOTilingData = reinterpret_cast<__gm__ ChunkFwdOTilingData *__restrict>(tiling);

        shapeBatch = gdnFwdOTilingData->shapeBatch;
        seqlen = gdnFwdOTilingData->seqlen;
        kNumHead = gdnFwdOTilingData->kNumHead;
        vNumHead = gdnFwdOTilingData->vNumHead;
        kHeadDim = gdnFwdOTilingData->kHeadDim;
        vHeadDim = gdnFwdOTilingData->vHeadDim;
        chunkSize = gdnFwdOTilingData->chunkSize;
        isVariedLen = gdnFwdOTilingData->isVariedLen;
        tokenBatch = gdnFwdOTilingData->tokenBatch;

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmChunklen.SetGlobalBuffer((__gm__ int64_t *)chunk_offsets);

        cubeCoreIdx = coreIdx;
        cubeCoreNum = coreNum;
        vLoops = vHeadDim / vBlockSize;
        taskNum = vLoops * shapeBatch * tokenBatch * vNumHead;
        headGroups = vNumHead / kNumHead;
        taskIdx = cubeCoreIdx * PING_PONG_STAGES;
        isRunning = taskIdx < taskNum;

        if (isVariedLen) {
            for (uint32_t b = 1; b <= tokenBatch; b++) {
                int64_t batchChunk = (gmSeqlen.GetValue(b) - gmSeqlen.GetValue(b - 1) + chunkSize - 1) / chunkSize;
                gmChunklen.SetValue(b, gmChunklen.GetValue(b - 1) + batchChunk);
            }
            numChunks = gmChunklen.GetValue(tokenBatch);
        }
    }

    CATLASS_DEVICE
    void InitTask() {
        if (unlikely(processNewTask)) {
            vIdx = taskIdx / (shapeBatch * tokenBatch * vNumHead);
            batchIdx = (taskIdx - vIdx * shapeBatch * tokenBatch * vNumHead) / vNumHead;
            baseHeadIdx = taskIdx % vNumHead;
            shapeBatchIdx = isVariedLen ? 0 : batchIdx;
            tokenBatchIdx = isVariedLen ? batchIdx : 0;
            chunkOffset = isVariedLen ? gmChunklen.GetValue(tokenBatchIdx) : 0;
            batchChunks = isVariedLen ? (gmChunklen.GetValue(tokenBatchIdx + 1) - chunkOffset) : numChunks;
            tokenOffset = isVariedLen ? gmSeqlen.GetValue(tokenBatchIdx) : 0;
            batchTokens = isVariedLen ? (gmSeqlen.GetValue(tokenBatchIdx + 1) - tokenOffset) : seqlen;
            chunkIdx = 0;
            headInnerIdx = 0;
        } else {
            chunkIdx = headInnerIdx == PING_PONG_STAGES - 1 ? chunkIdx + 1 : chunkIdx;
            headInnerIdx = (headInnerIdx + 1) % PING_PONG_STAGES;
        }
        
        vHeadIdx = baseHeadIdx + headInnerIdx;
        kHeadIdx = vHeadIdx / headGroups;
        offsets[currStage].qkOffset = (shapeBatchIdx * kNumHead * seqlen + kHeadIdx * seqlen + tokenOffset + chunkIdx * chunkSize) * kHeadDim;
        offsets[currStage].ovOffset = (shapeBatchIdx * vNumHead * seqlen + vHeadIdx * seqlen + tokenOffset + chunkIdx * chunkSize) * vHeadDim;
        offsets[currStage].hOffset = (shapeBatchIdx * vNumHead * numChunks + vHeadIdx * numChunks + chunkOffset + chunkIdx) * kHeadDim * vHeadDim;
        offsets[currStage].gOffset = shapeBatchIdx * vNumHead * seqlen + vHeadIdx * seqlen + tokenOffset + chunkIdx * chunkSize;
        offsets[currStage].attnWorkOffset = (cubeCoreIdx * PING_PONG_STAGES + currStage) * chunkSize * chunkSize;
        offsets[currStage].hvWorkOffset = (cubeCoreIdx * PING_PONG_STAGES + currStage) * chunkSize * vHeadDim;
        offsets[currStage].blockTokens = chunkIdx == (batchChunks - 1) ? (batchTokens - chunkIdx * chunkSize) : chunkSize;
        offsets[currStage].batchIdx = batchIdx; 
        offsets[currStage].headIdx = vHeadIdx; 
        offsets[currStage].chunkIdx = chunkIdx; 

        processNewTask = chunkIdx == batchChunks - 1 && headInnerIdx == PING_PONG_STAGES - 1;
        if (unlikely(processNewTask)){
            taskIdx += PING_PONG_STAGES * cubeCoreNum;
            if (unlikely(taskIdx >= taskNum)) {
                isRunning = false;
            }
        }
        
        currStage = (currStage + 1) % PING_PONG_STAGES;
    }
};

struct BlockSchedulerGdnFwdOCube : public BlockSchedulerGdnFwdO {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdOCube() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR tiling) {
        BlockSchedulerGdnFwdO::Init(cu_seqlens, chunk_offsets, tiling, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    }

    CATLASS_DEVICE
    bool NeedProcessCube1() {
        return true;
    }

    CATLASS_DEVICE
    GDNFwdOOffsets& GetCube1Offsets() {
        return offsets[(currStage - 1) % PING_PONG_STAGES];
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
        return offsets[(currStage - 2) % PING_PONG_STAGES];
    }

    CATLASS_DEVICE
    GemmCoord GetCube2Shape() {
        GDNFwdOOffsets& cube2Offsets = GetCube23Offsets();
        return GemmCoord{kHeadDim, vHeadDim, cube2Offsets.blockTokens};
    }

    CATLASS_DEVICE
    GemmCoord GetCube3Shape() {
        GDNFwdOOffsets& cube2Offsets = GetCube23Offsets();
        return GemmCoord{kHeadDim, vHeadDim, cube2Offsets.blockTokens};
    }
};

struct BlockSchedulerGdnFwdOVec : public BlockSchedulerGdnFwdO {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdOVec() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR tiling) {
        BlockSchedulerGdnFwdO::Init(cu_seqlens, chunk_offsets, tiling, AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(), AscendC::GetBlockNum());
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
        return offsets[(currStage - 1) % PING_PONG_STAGES];
    }
    
    CATLASS_DEVICE
    GDNFwdOOffsets& GetVec2Offsets() {
        return offsets[(currStage - 2) % PING_PONG_STAGES];
    }

};

}  // namespace Catlass::Gemm::Block
#endif // CATLASS_GEMM_SCHEDULER_GDN_FWD_O_HPP
