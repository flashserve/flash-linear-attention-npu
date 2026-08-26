/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "catlass/gemm_coord.hpp"
using namespace Catlass;

#ifndef CHUNK_GATED_DELTA_RULE_FWD_H_SCHEDULER_H
#define CHUNK_GATED_DELTA_RULE_FWD_H_SCHEDULER_H

constexpr uint32_t LOCAL_PING_PONG_STAGES = 2;
// One core can own more heads than fit in a stage round. Each round uses at
// most four heads and the same two-bank, eight-slot ready/free protocol.
constexpr uint32_t HEADS_PER_ROUND = 4;
constexpr uint32_t WORKSPACE_WINDOW_COUNT = 2;
constexpr uint32_t WORKSPACE_BUFFER_COUNT = HEADS_PER_ROUND * WORKSPACE_WINDOW_COUNT;
constexpr uint32_t BYTE_SIZE_16_BIT = 2;
constexpr uint32_t BYTES_PER_C0 = 32;
constexpr uint32_t BYTE_SIZE_PER_REPEAT = 256;
constexpr uint32_t SIZE_16_NUM_PER_C0 = BYTES_PER_C0 / BYTE_SIZE_16_BIT;
constexpr uint32_t FLOAT_NUM_PER_REPEAT = BYTE_SIZE_PER_REPEAT / sizeof(float);
constexpr uint32_t NZ_BLOCK_SIZE = 16;

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

struct GDNFwdHOffsets {
    uint32_t hSrcOffset;
    uint32_t hDstOffset;
    uint32_t uvOffset;
    uint32_t wkOffset;
    uint32_t wOffset;
    uint32_t gOffset;
    uint32_t gkOffset;
    uint32_t hWorkOffset;
    uint32_t vWorkOffset;
    uint32_t vBlockOffset;
    uint32_t vBlockDim;
    uint32_t initialStateOffset;
    uint32_t finalStateOffset;
    bool isInitialState;
    bool isFinalState;
    uint32_t blockTokens;
    uint32_t headOffset;
    uint32_t windowId;
    uint32_t workspaceSlot;
    // for debug
    uint32_t batchIdx;
    uint32_t headIdx;
    uint32_t chunkIdx;

};

struct GDNFwdHHeadTask {
    uint32_t batchIdx;
    uint32_t chunkIdx{0};
    uint32_t vHeadIdx;
    uint32_t kHeadIdx;
    uint32_t shapeBatchIdx;
    uint32_t tokenBatchIdx;

    uint32_t chunkOffset;
    uint32_t tokenOffset;
    uint32_t batchChunks{0};
    uint32_t batchTokens;
    bool active{false};

    GDNFwdHOffsets offset;
};

struct GDNFwdHHeadWindow {
    GDNFwdHHeadTask headTasks[HEADS_PER_ROUND];
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
    uint32_t inputTokenBatch;
    bool useInitialState;
    bool storeFinalState;
    uint32_t numSeqWorkspaceOffset;
    uint32_t numChunksWorkspaceOffset;

    uint32_t taskStride;
    uint32_t cubeCoreIdx;
    uint32_t cubeCoreNum;
    uint32_t taskNum;
    uint32_t headWindowNum;
    uint32_t coreHeadBase;
    uint32_t coreHeadCount;
    uint32_t headRoundNum;
    uint32_t currentHeadsInRound;
    uint32_t headGroups;
    uint32_t totalChunks;
    uint32_t totalTokens;

    GDNFwdHHeadWindow headWindow;

    bool isRunning;
    bool windowActive;
    uint32_t currentTaskIdx;
    uint32_t currentTaskRound;
    uint32_t currentChunkIdx;
    uint32_t currentHeadRoundIdx;
    uint32_t currentBatchChunks;
    uint32_t currentBatchIdx;
    uint32_t currentChunkOffset;
    uint32_t currentTokenOffset;
    uint32_t currentBatchTokens;
    uint32_t nextTaskIdx;

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumSeq;
    AscendC::GlobalTensor<int64_t> gmNumChunks;

    Arch::CrossCoreFlag cube1Done[WORKSPACE_WINDOW_COUNT] = {0, 1};
    Arch::CrossCoreFlag vec1Done[WORKSPACE_WINDOW_COUNT] = {2, 3};
    Arch::CrossCoreFlag cube2Done[WORKSPACE_WINDOW_COUNT] = {4, 5};
    Arch::CrossCoreFlag vec2Done[WORKSPACE_WINDOW_COUNT] = {6, 7};

    CATLASS_DEVICE
    BlockSchedulerGdnFwdH() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling, GM_ADDR user, uint32_t coreIdx, uint32_t coreNum) {
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
        useInitialState = gdnFwdHTilingData->useInitialState;
        storeFinalState = gdnFwdHTilingData->storeFinalState;
        numSeqWorkspaceOffset = gdnFwdHTilingData->numSeqWorkspaceOffset;
        numChunksWorkspaceOffset = gdnFwdHTilingData->numChunksWorkspaceOffset;

        InitRuntime(cu_seqlens, chunk_indices, user, coreIdx, coreNum);
    }

    template <typename TilingData>
    CATLASS_DEVICE
    void InitFromData(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, const TilingData& tilingData,
                      GM_ADDR user, uint32_t coreIdx, uint32_t coreNum) {
        batch = tilingData.batch;
        seqlen = tilingData.seqlen;
        kNumHead = tilingData.kNumHead;
        vNumHead = tilingData.vNumHead;
        kHeadDim = tilingData.kHeadDim;
        vHeadDim = tilingData.vHeadDim;
        chunkSize = tilingData.chunkSize;
        isVariedLen = tilingData.isVariedLen;
        shapeBatch = tilingData.shapeBatch;
        tokenBatch = tilingData.tokenBatch;
        useInitialState = tilingData.useInitialState;
        storeFinalState = tilingData.storeFinalState;
        numSeqWorkspaceOffset = tilingData.numSeqWorkspaceOffset;
        numChunksWorkspaceOffset = tilingData.numChunksWorkspaceOffset;

        InitRuntime(cu_seqlens, chunk_indices, user, coreIdx, coreNum);
    }

    CATLASS_DEVICE
    void InitRuntime(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR user,
                     uint32_t coreIdx, uint32_t coreNum) {

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        if (isVariedLen) {
            inputTokenBatch = tokenBatch;
            uint32_t actualBatch = 0;
            int64_t chunkPrefix = 0;
            int64_t prevSeq = 0, currSeq;
            for (uint32_t b = 1; b <= inputTokenBatch; b++) {
                currSeq = gmSeqlen.GetValue(b);
                int64_t batchSeqLen = currSeq - prevSeq;
                if (batchSeqLen > 0) {
                    actualBatch++;
                    int64_t batchChunk = (batchSeqLen + chunkSize - 1) / chunkSize;
                    chunkPrefix += batchChunk;
                }
                prevSeq = currSeq;
            }
            tokenBatch = actualBatch;
            batch = actualBatch;
            totalChunks = chunkPrefix;
            totalTokens = prevSeq;
        } else {
            totalChunks = (seqlen + chunkSize - 1) / chunkSize;
            totalTokens = seqlen;
        }

        cubeCoreIdx = coreIdx;
        cubeCoreNum = coreNum;
        vBlockSize = vHeadDim;
        // The host already selected the minimum active core count for the
        // minimum worst-case load. Split the remainder over the first cores
        // so every core owns one balanced, contiguous head range.
        uint32_t baseHeadsPerCore = vNumHead / cubeCoreNum;
        uint32_t remainderHeads = vNumHead % cubeCoreNum;
        coreHeadCount = baseHeadsPerCore + (cubeCoreIdx < remainderHeads ? 1 : 0);
        coreHeadBase = cubeCoreIdx * baseHeadsPerCore + Min(cubeCoreIdx, remainderHeads);
        headRoundNum = (coreHeadCount + HEADS_PER_ROUND - 1) / HEADS_PER_ROUND;
        currentHeadsInRound = 0;
        headWindowNum = cubeCoreNum;
        taskNum = batch * headWindowNum;
        headGroups = vNumHead / kNumHead;
        taskStride = cubeCoreNum;
        nextTaskIdx = cubeCoreIdx;
        currentTaskIdx = taskNum;
        currentTaskRound = 0;
        currentChunkIdx = 0;
        currentHeadRoundIdx = 0;
        currentBatchChunks = 0;
        currentBatchIdx = 0;
        currentChunkOffset = 0;
        currentTokenOffset = 0;
        currentBatchTokens = 0;
        windowActive = false;
        for (uint32_t headOffset = 0; headOffset < HEADS_PER_ROUND; ++headOffset) {
            auto& headTask = headWindow.headTasks[headOffset];
            headTask.chunkIdx = 0;
            headTask.batchChunks = 0;
            headTask.active = false;
        }
        isRunning = nextTaskIdx < taskNum;

    }

    CATLASS_DEVICE
    void ResolveVarlenSequence(uint32_t compactBatchIdx, GDNFwdHHeadTask& headTask) {
        uint32_t actualBatch = 0;
        int64_t chunkPrefix = 0;
        int64_t prevSeq = 0;
        for (uint32_t b = 1; b <= inputTokenBatch; ++b) {
            int64_t currSeq = gmSeqlen.GetValue(b);
            int64_t batchTokens = currSeq - prevSeq;
            if (batchTokens > 0) {
                int64_t batchChunks = (batchTokens + chunkSize - 1) / chunkSize;
                if (actualBatch == compactBatchIdx) {
                    headTask.chunkOffset = static_cast<uint32_t>(chunkPrefix);
                    headTask.batchChunks = static_cast<uint32_t>(batchChunks);
                    headTask.tokenOffset = static_cast<uint32_t>(prevSeq);
                    headTask.batchTokens = static_cast<uint32_t>(batchTokens);
                    return;
                }
                ++actualBatch;
                chunkPrefix += batchChunks;
            }
            prevSeq = currSeq;
        }
        headTask.chunkOffset = 0;
        headTask.batchChunks = 0;
        headTask.tokenOffset = 0;
        headTask.batchTokens = 0;
    }

    CATLASS_DEVICE
    uint32_t GetVarlenChunkOffset(uint32_t compactBatchIdx) {
        GDNFwdHHeadTask headTask;
        ResolveVarlenSequence(compactBatchIdx, headTask);
        return headTask.chunkOffset;
    }

    CATLASS_DEVICE
    void InitNewHeadTask(GDNFwdHHeadTask& newHeadTask, uint32_t vHeadIdx) {
        newHeadTask.batchIdx = currentBatchIdx;
        newHeadTask.vHeadIdx = vHeadIdx;
        newHeadTask.kHeadIdx = newHeadTask.vHeadIdx / headGroups;
        newHeadTask.shapeBatchIdx = isVariedLen ? 0 : newHeadTask.batchIdx;
        newHeadTask.tokenBatchIdx = isVariedLen ? newHeadTask.batchIdx : 0;
        newHeadTask.chunkOffset = currentChunkOffset;
        newHeadTask.batchChunks = currentBatchChunks;
        newHeadTask.tokenOffset = currentTokenOffset;
        newHeadTask.batchTokens = currentBatchTokens;
        newHeadTask.chunkIdx = currentChunkIdx;
    }

    CATLASS_DEVICE
    void PrepareCurrentHeadRound() {
        uint32_t roundHeadBase = currentHeadRoundIdx * HEADS_PER_ROUND;
        currentHeadsInRound = Min(HEADS_PER_ROUND, coreHeadCount - roundHeadBase);
        for (uint32_t headOffset = 0; headOffset < HEADS_PER_ROUND; ++headOffset) {
            auto& headTask = headWindow.headTasks[headOffset];
            headTask.active = headOffset < currentHeadsInRound;
            if (!headTask.active) {
                headTask.batchChunks = 0;
                continue;
            }
            InitNewHeadTask(headTask, coreHeadBase + roundHeadBase + headOffset);
            UpdateTask(headOffset);
        }
    }

    CATLASS_DEVICE
    void AssignNextWindow() {
        while (nextTaskIdx < taskNum) {
            currentTaskIdx = nextTaskIdx;
            nextTaskIdx += taskStride;
            currentTaskRound = (currentTaskIdx - cubeCoreIdx) / taskStride;
            currentBatchIdx = currentTaskIdx / headWindowNum;
            currentChunkIdx = 0;
            currentHeadRoundIdx = 0;

            GDNFwdHHeadTask sequenceTask;
            if (isVariedLen) {
                ResolveVarlenSequence(currentBatchIdx, sequenceTask);
                currentChunkOffset = sequenceTask.chunkOffset;
                currentBatchChunks = sequenceTask.batchChunks;
                currentTokenOffset = sequenceTask.tokenOffset;
                currentBatchTokens = sequenceTask.batchTokens;
            } else {
                currentChunkOffset = 0;
                currentBatchChunks = totalChunks;
                currentTokenOffset = 0;
                currentBatchTokens = totalTokens;
            }
            windowActive = currentBatchChunks > 0 && coreHeadCount > 0;
            if (windowActive) {
                PrepareCurrentHeadRound();
                return;
            }
        }
        windowActive = false;
        currentHeadsInRound = 0;
        for (uint32_t headOffset = 0; headOffset < HEADS_PER_ROUND; ++headOffset) {
            headWindow.headTasks[headOffset].active = false;
        }
    }

    CATLASS_DEVICE
    void UpdateTask(uint32_t headOffset) {
        auto& headTask = headWindow.headTasks[headOffset];
        auto& offset = headTask.offset;

        offset.isInitialState = headTask.chunkIdx == 0;
        offset.isFinalState = headTask.chunkIdx == (headTask.batchChunks - 1);
        uint32_t vBlockOffset = 0;
        uint32_t vBlockDim = vBlockSize;
        offset.initialStateOffset = (headTask.batchIdx * vNumHead + headTask.vHeadIdx) * kHeadDim * vHeadDim + vBlockOffset;
        offset.finalStateOffset = (headTask.batchIdx * vNumHead + headTask.vHeadIdx) * kHeadDim * vHeadDim + vBlockOffset;
        offset.hSrcOffset = (headTask.shapeBatchIdx * vNumHead * totalChunks + headTask.vHeadIdx * totalChunks + headTask.chunkOffset + headTask.chunkIdx) * kHeadDim * vHeadDim + vBlockOffset;
        offset.hDstOffset = offset.hSrcOffset + kHeadDim * vHeadDim;
        if (storeFinalState && offset.isFinalState) {
            offset.hDstOffset = offset.hSrcOffset;
        }
        offset.uvOffset = (headTask.shapeBatchIdx * vNumHead * totalTokens + headTask.vHeadIdx * totalTokens + headTask.tokenOffset + headTask.chunkIdx * chunkSize) * vHeadDim + vBlockOffset;
        offset.wkOffset = (headTask.shapeBatchIdx * kNumHead * totalTokens + headTask.kHeadIdx * totalTokens + headTask.tokenOffset + headTask.chunkIdx * chunkSize) * kHeadDim;
        offset.wOffset = (headTask.shapeBatchIdx * vNumHead * totalTokens + headTask.vHeadIdx * totalTokens + headTask.tokenOffset + headTask.chunkIdx * chunkSize) * kHeadDim;
        offset.gOffset = headTask.shapeBatchIdx * vNumHead * totalTokens + headTask.vHeadIdx * totalTokens + headTask.tokenOffset + headTask.chunkIdx * chunkSize;
        offset.gkOffset = (headTask.shapeBatchIdx * vNumHead * totalTokens + headTask.vHeadIdx * totalTokens + headTask.tokenOffset + headTask.chunkIdx * chunkSize) * kHeadDim;
        // A head round stays on the same bank across chunks, so Stage0 of the
        // next chunk waits for that round's Stage3 state write to complete.
        uint32_t windowId =
            (currentTaskRound + currentHeadRoundIdx) % WORKSPACE_WINDOW_COUNT;
        uint32_t workspaceSlot = windowId * HEADS_PER_ROUND + headOffset;
        offset.hWorkOffset = (cubeCoreIdx * WORKSPACE_BUFFER_COUNT + workspaceSlot) * kHeadDim * vBlockSize;
        offset.vWorkOffset = (cubeCoreIdx * WORKSPACE_BUFFER_COUNT + workspaceSlot) * chunkSize * vBlockSize;
        offset.vBlockOffset = vBlockOffset;
        offset.vBlockDim = vBlockDim;
        offset.blockTokens = offset.isFinalState ? (headTask.batchTokens - headTask.chunkIdx * chunkSize) : chunkSize;
        offset.headOffset = headOffset;
        offset.windowId = windowId;
        offset.workspaceSlot = workspaceSlot;
        offset.batchIdx = headTask.batchIdx;
        offset.headIdx = headTask.vHeadIdx;
        offset.chunkIdx = headTask.chunkIdx;
    }

    CATLASS_DEVICE
    void InitTasks() {
        if (!windowActive) {
            AssignNextWindow();
        } else if (currentHeadRoundIdx + 1 < headRoundNum) {
            ++currentHeadRoundIdx;
            PrepareCurrentHeadRound();
        } else if (currentChunkIdx + 1 < currentBatchChunks) {
            ++currentChunkIdx;
            currentHeadRoundIdx = 0;
            PrepareCurrentHeadRound();
        } else {
            windowActive = false;
            AssignNextWindow();
        }
        isRunning = windowActive;
    }

    CATLASS_DEVICE
    const GDNFwdHHeadTask& GetHeadTask(uint32_t i) const {
        return headWindow.headTasks[i];
    }

    CATLASS_DEVICE
    uint32_t GetHeadsInRound() const {
        return currentHeadsInRound;
    }

    CATLASS_DEVICE
    uint32_t GetHeadsPerCore() const {
        return coreHeadCount;
    }

    CATLASS_DEVICE
    uint32_t GetCoreHeadBase() const {
        return coreHeadBase;
    }

    CATLASS_DEVICE
    uint32_t GetWindowId() const {
        return (currentTaskRound + currentHeadRoundIdx) % WORKSPACE_WINDOW_COUNT;
    }

    CATLASS_DEVICE
    const GDNFwdHOffsets& GetCurTaskOffsets(const GDNFwdHHeadTask& headTask) const {
        return headTask.offset;
    }

    CATLASS_DEVICE
    bool HeadTaskIsDone(const GDNFwdHHeadTask& headTask) const {
        return !headTask.active;
    }

    CATLASS_DEVICE
    bool NeedProcessStage0(const GDNFwdHHeadTask& headTask) const {
        return useInitialState || !headTask.offset.isInitialState;
    }

    CATLASS_DEVICE
    bool NeedProcessStage2(const GDNFwdHHeadTask& headTask) {
        return storeFinalState || !headTask.offset.isFinalState;
    }
};

struct BlockSchedulerGdnFwdHCube : public BlockSchedulerGdnFwdH {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdHCube() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling, GM_ADDR user) {
        BlockSchedulerGdnFwdH::Init(cu_seqlens, chunk_indices, tiling, user, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    }

    template <typename TilingData>
    CATLASS_DEVICE
    void InitFromData(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, const TilingData& tilingData, GM_ADDR user) {
        BlockSchedulerGdnFwdH::InitFromData(
            cu_seqlens, chunk_indices, tilingData, user, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    }

};

struct BlockSchedulerGdnFwdHVec : public BlockSchedulerGdnFwdH {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdHVec() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling, GM_ADDR user) {
        BlockSchedulerGdnFwdH::Init(
            cu_seqlens, chunk_indices, tiling, user,
            AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(),
            AscendC::GetBlockNum());
    }

    template <typename TilingData>
    CATLASS_DEVICE
    void InitFromData(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, const TilingData& tilingData, GM_ADDR user) {
        BlockSchedulerGdnFwdH::InitFromData(
            cu_seqlens, chunk_indices, tilingData, user,
            AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(),
            AscendC::GetBlockNum());
    }

};

}  // namespace Catlass::Gemm::Block

#endif  // CHUNK_GATED_DELTA_RULE_FWD_H_SCHEDULER_H
