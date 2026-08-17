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

#ifndef CATLASS_GEMM_SCHEDULER_GDN_FWD_H_HPP
#define CATLASS_GEMM_SCHEDULER_GDN_FWD_H_HPP

// constexpr uint32_t PING_PONG_STAGES = 1;
constexpr uint32_t PING_PONG_STAGES = 2;
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
    uint64_t hSrcOffset;
    uint64_t hDstOffset;
    uint64_t uvOffset;
    uint64_t wkOffset;
    uint64_t wOffset;
    uint64_t gOffset;
    uint64_t gkOffset;
    uint64_t hWorkOffset;
    uint64_t vWorkOffset;
    uint64_t kDecayWorkOffset;
    uint32_t vBlockOffset;
    uint32_t vBlockDim;
    uint64_t initialStateOffset;
    uint64_t finalStateOffset;
    bool isInitialState;
    bool isFinalState;
    uint32_t blockTokens;
    uint32_t streamId;
    // for debug
    uint32_t batchIdx;
    uint32_t headIdx;
    uint32_t chunkIdx;

};

struct GDNFwdHStream {
    uint32_t batchIdx;
    uint32_t chunkIdx{0};
    uint32_t vHeadIdx;
    uint32_t kHeadIdx;
    uint32_t shapeBatchIdx;
    uint32_t tokenBatchIdx;

    uint32_t chunkOffset;
    uint64_t tokenOffset;
    uint32_t batchChunks{0};
    uint64_t batchTokens;
    uint64_t nextTaskIdx{0};
    bool active{false};

    GDNFwdHOffsets offset;
};

struct GDNFwdHRunningQ {
    GDNFwdHStream streams[PING_PONG_STAGES];
};

struct BlockSchedulerGdnFwdH {
    uint32_t batch;
    uint64_t seqlen;
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
    uint64_t numSeqWorkspaceOffset;
    uint64_t numChunksWorkspaceOffset;

    uint64_t taskIdx;
    uint64_t taskStride;
    uint32_t cubeCoreIdx;
    uint32_t cubeCoreNum;
    uint64_t taskNum;
    uint32_t headGroups;
    uint32_t totalChunks;
    uint64_t totalTokens;
    bool useCompactSequencePlan{false};
    uint32_t sequenceCount{0};
    uint32_t compactTotalChunks{0};
    uint32_t fwdUsedCoreNum{0};
    GM_ADDR compactPlanAddr{nullptr};
    uint64_t seqChunkOffsetsOffset{0};
    uint32_t ownedHeadBegin{0};
    uint32_t ownedHeadEnd{0};
    uint32_t maxHeadsPerCore{0};

    bool cachedCompactSequenceValid{false};
    uint32_t cachedSequenceIdx{0};
    uint32_t cachedChunkOffset{0};
    uint64_t cachedTokenOffset{0};
    uint32_t cachedBatchChunks{0};
    uint64_t cachedBatchTokens{0};

    GDNFwdHRunningQ runningQ;

    bool isRunning;

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumSeq;
    AscendC::GlobalTensor<int64_t> gmNumChunks;
    AscendC::GlobalTensor<uint32_t> gmSeqChunkOffsets;

    Arch::CrossCoreFlag cube1Done[PING_PONG_STAGES] = {0, 1};
    Arch::CrossCoreFlag vec1Done[PING_PONG_STAGES] = {2, 3};
    Arch::CrossCoreFlag cube2Done[PING_PONG_STAGES] = {4, 5};
    Arch::CrossCoreFlag vec2Done[PING_PONG_STAGES] = {6, 7};

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
        useCompactSequencePlan = false;

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
        useCompactSequencePlan = tilingData.useCompactSequencePlan;
        sequenceCount = tilingData.sequenceCount;
        compactTotalChunks = tilingData.compactTotalChunks;
        fwdUsedCoreNum = tilingData.fwdUsedCoreNum;
        compactPlanAddr = tilingData.compactPlan;
        seqChunkOffsetsOffset = tilingData.seqChunkOffsetsOffset;

        InitRuntime(cu_seqlens, chunk_indices, user, coreIdx, coreNum);
    }

    CATLASS_DEVICE
    void InitRuntime(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR user,
                     uint32_t coreIdx, uint32_t coreNum) {

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        if (useCompactSequencePlan) {
            inputTokenBatch = sequenceCount;
            tokenBatch = sequenceCount;
            batch = sequenceCount;
            totalChunks = compactTotalChunks;
            totalTokens = seqlen;
            if (isVariedLen) {
                gmSeqChunkOffsets.SetGlobalBuffer(
                    (__gm__ uint32_t *)(compactPlanAddr + seqChunkOffsetsOffset));
            }
        } else if (isVariedLen) {
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
            totalTokens = static_cast<uint64_t>(prevSeq);
        } else {
            totalChunks = (seqlen + chunkSize - 1) / chunkSize;
            totalTokens = seqlen;
        }

        cubeCoreIdx = coreIdx;
        cubeCoreNum = coreNum;
        vBlockSize = vHeadDim;
        taskNum = static_cast<uint64_t>(batch) * vNumHead;
        headGroups = vNumHead / kNumHead;
        cachedCompactSequenceValid = false;
        if (useCompactSequencePlan) {
            fwdUsedCoreNum = Min(fwdUsedCoreNum, Min(vNumHead, cubeCoreNum));
            // Process synchronizes every wave, so inactive cores keep the same wave count.
            if (fwdUsedCoreNum != 0 && cubeCoreIdx < fwdUsedCoreNum) {
                ownedHeadBegin = static_cast<uint32_t>(
                    static_cast<uint64_t>(cubeCoreIdx) * vNumHead / fwdUsedCoreNum);
                ownedHeadEnd = static_cast<uint32_t>(
                    static_cast<uint64_t>(cubeCoreIdx + 1) * vNumHead / fwdUsedCoreNum);
                maxHeadsPerCore = (vNumHead + fwdUsedCoreNum - 1) / fwdUsedCoreNum;
            } else {
                ownedHeadBegin = 0;
                ownedHeadEnd = 0;
                maxHeadsPerCore = fwdUsedCoreNum == 0
                    ? 0
                    : (vNumHead + fwdUsedCoreNum - 1) / fwdUsedCoreNum;
            }
        }
        InitTaskWave(0);

    }

    CATLASS_DEVICE
    uint64_t ResolveWaveTask(uint64_t waveIdx) const {
        if (!useCompactSequencePlan) {
            return waveIdx * static_cast<uint64_t>(cubeCoreNum) + cubeCoreIdx;
        }
        if (maxHeadsPerCore == 0 || cubeCoreIdx >= fwdUsedCoreNum) {
            return taskNum;
        }
        const uint64_t sequenceIdx = waveIdx / maxHeadsPerCore;
        const uint32_t localHeadIdx =
            static_cast<uint32_t>(waveIdx % maxHeadsPerCore);
        const uint32_t headIdx = ownedHeadBegin + localHeadIdx;
        if (sequenceIdx >= sequenceCount || headIdx >= ownedHeadEnd) {
            return taskNum;
        }
        return sequenceIdx * static_cast<uint64_t>(vNumHead) + headIdx;
    }

    CATLASS_DEVICE
    void InitTaskWave(uint64_t waveIdx) {
        const uint64_t firstTaskIdx = ResolveWaveTask(waveIdx);
        taskStride = taskNum;
        for (uint32_t streamId = 0; streamId < PING_PONG_STAGES; ++streamId) {
            auto& stream = runningQ.streams[streamId];
            stream.nextTaskIdx = streamId == 0 ? firstTaskIdx : taskNum;
            stream.chunkIdx = 0;
            stream.batchChunks = 0;
            stream.active = false;
        }
        isRunning = firstTaskIdx < taskNum;
    }

    CATLASS_DEVICE
    uint64_t GetTaskWaveCount() const {
        if (useCompactSequencePlan) {
            return static_cast<uint64_t>(sequenceCount) * maxHeadsPerCore;
        }
        return CeilDiv(taskNum, static_cast<uint64_t>(cubeCoreNum));
    }

    CATLASS_DEVICE
    void ResolveCompactSequence(uint32_t sequenceIdx, GDNFwdHStream& stream) {
        if (cachedCompactSequenceValid && cachedSequenceIdx == sequenceIdx) {
            stream.chunkOffset = cachedChunkOffset;
            stream.tokenOffset = cachedTokenOffset;
            stream.batchChunks = cachedBatchChunks;
            stream.batchTokens = cachedBatchTokens;
            return;
        }
        const uint64_t tokenOffset =
            static_cast<uint64_t>(gmSeqlen.GetValue(sequenceIdx));
        const uint64_t tokenEnd =
            static_cast<uint64_t>(gmSeqlen.GetValue(sequenceIdx + 1));
        const uint32_t chunkOffset = gmSeqChunkOffsets.GetValue(sequenceIdx);
        const uint32_t chunkEnd = gmSeqChunkOffsets.GetValue(sequenceIdx + 1);

        stream.chunkOffset = chunkOffset;
        stream.tokenOffset = tokenOffset;
        stream.batchChunks = chunkEnd - chunkOffset;
        stream.batchTokens = tokenEnd - tokenOffset;
        cachedCompactSequenceValid = true;
        cachedSequenceIdx = sequenceIdx;
        cachedChunkOffset = stream.chunkOffset;
        cachedTokenOffset = stream.tokenOffset;
        cachedBatchChunks = stream.batchChunks;
        cachedBatchTokens = stream.batchTokens;
    }

    CATLASS_DEVICE
    void ResolveVarlenSequence(uint32_t compactBatchIdx, GDNFwdHStream& stream) {
        if (useCompactSequencePlan) {
            ResolveCompactSequence(compactBatchIdx, stream);
            return;
        }
        uint32_t actualBatch = 0;
        int64_t chunkPrefix = 0;
        int64_t prevSeq = 0;
        for (uint32_t b = 1; b <= inputTokenBatch; ++b) {
            int64_t currSeq = gmSeqlen.GetValue(b);
            int64_t batchTokens = currSeq - prevSeq;
            if (batchTokens > 0) {
                int64_t batchChunks = (batchTokens + chunkSize - 1) / chunkSize;
                if (actualBatch == compactBatchIdx) {
                    stream.chunkOffset = static_cast<uint32_t>(chunkPrefix);
                    stream.batchChunks = static_cast<uint32_t>(batchChunks);
                    stream.tokenOffset = static_cast<uint64_t>(prevSeq);
                    stream.batchTokens = static_cast<uint64_t>(batchTokens);
                    return;
                }
                ++actualBatch;
                chunkPrefix += batchChunks;
            }
            prevSeq = currSeq;
        }
        stream.chunkOffset = 0;
        stream.batchChunks = 0;
        stream.tokenOffset = 0;
        stream.batchTokens = 0;
    }

    CATLASS_DEVICE
    uint32_t GetVarlenChunkOffset(uint32_t compactBatchIdx) {
        GDNFwdHStream stream;
        ResolveVarlenSequence(compactBatchIdx, stream);
        return stream.chunkOffset;
    }

    CATLASS_DEVICE
    void InitNewStream(uint64_t newTaskIdx, GDNFwdHStream& newStream) {
        // Host tiling bounds dense batch and compact sequence IDs to uint32_t.
        newStream.batchIdx = static_cast<uint32_t>(newTaskIdx / vNumHead);
        newStream.vHeadIdx = static_cast<uint32_t>(newTaskIdx % vNumHead);
        newStream.kHeadIdx = newStream.vHeadIdx / headGroups;
        newStream.shapeBatchIdx = isVariedLen ? 0 : newStream.batchIdx;
        newStream.tokenBatchIdx = isVariedLen ? newStream.batchIdx : 0;
        if (isVariedLen) {
            ResolveVarlenSequence(newStream.tokenBatchIdx, newStream);
        } else {
            newStream.chunkOffset = 0;
            newStream.batchChunks = totalChunks;
            newStream.tokenOffset = 0;
            newStream.batchTokens = totalTokens;
        }
        newStream.chunkIdx = 0;
    }

    CATLASS_DEVICE
    void InitNewStream(GDNFwdHStream& newStream) {
        InitNewStream(taskIdx, newStream);
    }

    CATLASS_DEVICE
    bool ResolveWaveStream(uint64_t waveIdx, GDNFwdHStream& stream) {
        const uint64_t waveTaskIdx = ResolveWaveTask(waveIdx);
        if (waveTaskIdx >= taskNum) {
            stream.active = false;
            stream.batchChunks = 0;
            return false;
        }
        InitNewStream(waveTaskIdx, stream);
        stream.active = stream.batchChunks > 0;
        return true;
    }

    CATLASS_DEVICE
    void AssignNextStream(uint32_t streamId) {
        auto& stream = runningQ.streams[streamId];
        taskIdx = stream.nextTaskIdx;
        if (taskIdx >= taskNum) {
            stream.active = false;
            stream.batchChunks = 0;
            return;
        }

        const uint64_t remainingTasks = taskNum - taskIdx;
        stream.nextTaskIdx = taskStride >= remainingTasks
            ? taskNum
            : taskIdx + taskStride;
        InitNewStream(stream);
        stream.active = stream.batchChunks > 0;
        if (stream.active) {
            UpdateTask(streamId);
        }
    }

    CATLASS_DEVICE
    void UpdateTask(uint32_t streamId) {
        auto& stream = runningQ.streams[streamId];
        auto& offset = stream.offset;

        offset.isInitialState = stream.chunkIdx == 0;
        offset.isFinalState = stream.chunkIdx == (stream.batchChunks - 1);
        uint32_t vBlockOffset = 0;
        uint32_t vBlockDim = vBlockSize;
        const uint64_t stateSlot =
            static_cast<uint64_t>(stream.batchIdx) * vNumHead + stream.vHeadIdx;
        offset.initialStateOffset =
            stateSlot * kHeadDim * vHeadDim + vBlockOffset;
        offset.finalStateOffset = offset.initialStateOffset;
        const uint64_t chunkSlot =
            (static_cast<uint64_t>(stream.shapeBatchIdx) * vNumHead + stream.vHeadIdx) *
                totalChunks +
            stream.chunkOffset + stream.chunkIdx;
        offset.hSrcOffset =
            chunkSlot * kHeadDim * vHeadDim + vBlockOffset;
        offset.hDstOffset =
            offset.hSrcOffset + static_cast<uint64_t>(kHeadDim) * vHeadDim;
        if (storeFinalState && offset.isFinalState) {
            offset.hDstOffset = offset.hSrcOffset;
        }
        const uint64_t chunkTokenOffset =
            stream.tokenOffset + static_cast<uint64_t>(stream.chunkIdx) * chunkSize;
        const uint64_t vTokenSlot =
            (static_cast<uint64_t>(stream.shapeBatchIdx) * vNumHead + stream.vHeadIdx) *
                totalTokens +
            chunkTokenOffset;
        const uint64_t kTokenSlot =
            (static_cast<uint64_t>(stream.shapeBatchIdx) * kNumHead + stream.kHeadIdx) *
                totalTokens +
            chunkTokenOffset;
        offset.uvOffset = vTokenSlot * vHeadDim + vBlockOffset;
        offset.wkOffset = kTokenSlot * kHeadDim;
        offset.wOffset = vTokenSlot * kHeadDim;
        offset.gOffset = vTokenSlot;
        offset.gkOffset = vTokenSlot * kHeadDim;
        const uint64_t workspaceSlot =
            static_cast<uint64_t>(cubeCoreIdx) * PING_PONG_STAGES + streamId;
        offset.hWorkOffset = workspaceSlot * kHeadDim * vBlockSize;
        offset.vWorkOffset = workspaceSlot * chunkSize * vBlockSize;
        offset.kDecayWorkOffset = workspaceSlot * chunkSize * kHeadDim;
        offset.vBlockOffset = vBlockOffset;
        offset.vBlockDim = vBlockDim;
        offset.blockTokens = offset.isFinalState
            ? static_cast<uint32_t>(
                  stream.batchTokens - static_cast<uint64_t>(stream.chunkIdx) * chunkSize)
            : chunkSize;
        offset.streamId = streamId;
        offset.batchIdx = stream.batchIdx;
        offset.headIdx = stream.vHeadIdx;
        offset.chunkIdx = stream.chunkIdx;
    }

    CATLASS_DEVICE
    void InitTasks() {
        isRunning = false;
        for (uint32_t streamId = 0; streamId < PING_PONG_STAGES; ++streamId) {
            auto& stream = runningQ.streams[streamId];
            if (stream.active) {
                stream.chunkIdx += 1;
                if (stream.chunkIdx >= stream.batchChunks) {
                    stream.active = false;
                    stream.batchChunks = 0;
                }
            }
            if (!stream.active) {
                AssignNextStream(streamId);
            } else {
                UpdateTask(streamId);
            }
            if (stream.active) {
                isRunning = true;
            }
        }
    }

    CATLASS_DEVICE
    const GDNFwdHStream& GetStream(uint32_t i) const {
        return runningQ.streams[i];
    }

    CATLASS_DEVICE
    uint32_t GetStreamId(uint32_t i) const {
        return i;
    }

    CATLASS_DEVICE
    const GDNFwdHOffsets& GetCurTaskOffsets(const GDNFwdHStream& stream) const {
        return stream.offset;
    }

    CATLASS_DEVICE
    bool StreamIsDone(const GDNFwdHStream& stream) const {
        return !stream.active;
    }

    CATLASS_DEVICE
    bool NeedProcessStage2(const GDNFwdHStream& stream) {
        return storeFinalState || !stream.offset.isFinalState;
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

#endif  // CATLASS_GEMM_SCHEDULER_GDN_FWD_H_HPP
