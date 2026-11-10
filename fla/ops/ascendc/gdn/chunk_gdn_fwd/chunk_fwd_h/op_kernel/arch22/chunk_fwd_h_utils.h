/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file arch22/chunk_fwd_h_utils.h
 * \brief gated 复用引擎（仅 A2/arch22，由 chunk_fwd_h.cpp 非 arch35 分支引入）：
 * 复用 chunk_gated_delta_rule_fwd_h 的 preload kernel 链。原六个文件
 * （chunk_fwd_h_gated_struct.h、epilogue/gdn_fwd_h_epilogue_policies.hpp、
 * epilogue/block/block_epilogue_gdn_fwdh_{update,vnew}_preload.hpp、
 * gemm/block/block_scheduler_gdn_fwd_h_preload.hpp、gemm/kernel/gdn_fwd_h_kernel_preload.hpp）
 * 按原 include 展开顺序内联为单文件，芯片目录回归 cube.h/vec.h + 本文件布局；
 * gated 复用谓词/入口已归位回 chunk_fwd_h.cpp（优化前入口位置）；
 * 内核类已拆分为双引擎：ChunkFwdHGatedCubeArch22（arch22/chunk_fwd_h_cube.h）
 * 与 ChunkFwdHGatedVecArch22（arch22/chunk_fwd_h_vec.h），本文件保留
 * 镜像/调度器基座（V1/V2 epilogue 已解体入 vec 引擎）。纯搬移，无代码语义变化。
 */

#ifndef ARCH22_CHUNK_FWD_H_UTILS_H
#define ARCH22_CHUNK_FWD_H_UTILS_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif

// ---- 内联自原 chunk_fwd_h_gated_struct.h：本尊算子 kernel 侧 tiling 结构体镜像
// （字段序/类型对齐框架 BEGIN_TILING_DATA_DEF 宏生成的同名结构；fast kernel launch
// 扩展独立编译 kernel 时无该自动生成头，故提供同构 plain struct，可按值经 <<<>>> 传入）
struct ChunkGatedDeltaRuleFwdHTilingData {
    int64_t batch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    int64_t dataType;
    int64_t gDataType;
    int64_t stateDataType;
    int64_t isVariedLen;
    int64_t shapeBatch;
    int64_t tokenBatch;
    bool useG;
    bool useGk;
    int64_t vWorkspaceOffset;
    int64_t vUpdateWorkspaceOffset;
    int64_t kDecayWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t numSeqWorkspaceOffset;
    int64_t numChunksWorkspaceOffset;
};

// ---- 原 gemm/kernel/gdn_fwd_h_kernel_preload.hpp 的 catlass 头（补齐主模板声明等前置）
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"

#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_preload.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"

// ---- 内联自原 gemm/block/block_scheduler_gdn_fwd_h_preload.hpp
using namespace Catlass;

// gated 复用拷贝仅引入 preload 链：原版经非 preload scheduler（本拷贝未引入）
// 传递的 PING_PONG_STAGES 在此兜底定义。
#ifndef PING_PONG_STAGES
constexpr uint32_t PING_PONG_STAGES = 2;
#endif

namespace Catlass::Gemm::Block {

struct GDNFwdHOffsetsPreload {
    uint64_t hSrcOffset{0};
    uint64_t hDstOffset{0};
    uint64_t uvOffset{0};
    uint64_t wkOffset{0};
    uint64_t wOffset{0};
    uint64_t gOffset{0};
    uint64_t hWorkOffset{0};
    uint64_t vWorkOffset{0};
    uint64_t initialStateOffset{0};
    uint64_t finalStateOffset{0};
    bool isInitialState{false};
    bool isFinalState{false};
    uint32_t blockTokens{0};
    // for debug
    uint32_t batchIdx{0};
    uint32_t headIdx{0};
    uint32_t chunkIdx{0};

};

struct GDNFwdHStreamPreload {
    uint32_t vIdx{0};
    uint32_t batchIdx{0};
    uint32_t chunkIdx{0};
    uint32_t vHeadIdx{0};
    uint32_t kHeadIdx{0};
    uint32_t shapeBatchIdx{0};
    uint32_t tokenBatchIdx{0};
     
    uint32_t chunkOffset{0};
    uint32_t tokenOffset{0};
    uint32_t batchChunks{0};
    uint32_t batchTokens{0};
     
    GDNFwdHOffsetsPreload offset;
};
     
struct GDNFwdHRunningQPreload {
    GDNFwdHStreamPreload streams[PING_PONG_STAGES];
    uint32_t head{0};
};

struct BlockSchedulerGdnFwdHPreload {
    uint32_t batch{0};
    uint32_t seqlen{0};
    uint32_t kNumHead{0};
    uint32_t vNumHead{0};
    uint32_t kHeadDim{0};
    uint32_t vHeadDim{0};
    uint32_t chunkSize{0};
    uint32_t vBlockSize{128};
    uint32_t isVariedLen{0};
    uint32_t shapeBatch{0};
    uint32_t tokenBatch{0};
    bool useInitialState{false};
    bool storeFinalState{false};
    uint64_t numSeqWorkspaceOffset{0};
    uint64_t numChunksWorkspaceOffset{0};

    uint32_t taskIdx{0};
    uint32_t taskLoops{0};
    uint32_t cubeCoreIdx{0};
    uint32_t cubeCoreNum{0};
    uint32_t vLoops{0};
    uint32_t taskNum{0};
    uint32_t headGroups{0};
    uint32_t totalChunks{0};
    uint32_t totalTokens{0};
    bool hasDummyHead{false};

    GDNFwdHRunningQPreload runningQ;
    int64_t curLoopIdx{-1};
    uint32_t curLoopTaskBegin{0};
    uint32_t curLoopTaskCnt{0};
    uint32_t lastLoopTaskCnt{0};

    bool isRunning{false};

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumSeq;
    AscendC::GlobalTensor<int64_t> gmNumChunks;

    Arch::CrossCoreFlag cube1Done{0};
    Arch::CrossCoreFlag vec1Done{1};
    Arch::CrossCoreFlag cube2Done{2};
    Arch::CrossCoreFlag vec2Done[PING_PONG_STAGES] = {3, 4};

    CATLASS_DEVICE
    BlockSchedulerGdnFwdHPreload() {}

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

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        if (isVariedLen) {
            gmNumChunks.SetValue(0, 0);
            gmNumSeq.SetValue(0, 0);
            uint32_t actualBatch = 0;
            int64_t prevSeq = 0, currSeq;
            for (uint32_t b = 1; b <= tokenBatch; b++) {
                currSeq = gmSeqlen.GetValue(b);
                int64_t batchSeqLen = currSeq - prevSeq;
                if (batchSeqLen > 0) {
                    actualBatch++;
                    gmNumSeq.SetValue(actualBatch, currSeq);
                    int64_t batchChunk = (batchSeqLen + chunkSize - 1) / chunkSize;
                    gmNumChunks.SetValue(actualBatch, gmNumChunks.GetValue(actualBatch - 1) + batchChunk);
                }
                prevSeq = currSeq;
            }
            tokenBatch = actualBatch;
            batch = actualBatch;
            totalChunks = gmNumChunks.GetValue(tokenBatch);
            totalTokens = gmNumSeq.GetValue(tokenBatch);
        } else {
            totalChunks = (seqlen + chunkSize - 1) / chunkSize;
            totalTokens = seqlen;
        }

        cubeCoreIdx = coreIdx;
        cubeCoreNum = coreNum;
        vLoops = vHeadDim / vBlockSize;
        taskNum = vLoops * batch * vNumHead;
        headGroups = vNumHead / kNumHead;
        hasDummyHead = (taskNum % (PING_PONG_STAGES * cubeCoreNum) <= cubeCoreNum) && (taskNum % (PING_PONG_STAGES * cubeCoreNum) > 0);
        taskLoops = (taskNum + cubeCoreNum * PING_PONG_STAGES - 1) / (cubeCoreNum * PING_PONG_STAGES);
        uint32_t maxTaskCntPerLoop = taskNum > cubeCoreNum ? PING_PONG_STAGES : 1;
        curLoopTaskBegin = cubeCoreIdx * maxTaskCntPerLoop;
        uint32_t lastLoopTaskBegin = curLoopTaskBegin + (taskLoops - 1) * maxTaskCntPerLoop * cubeCoreNum;
        if (lastLoopTaskBegin >= taskNum) {
            lastLoopTaskCnt = 0;
        } else {
            uint32_t maxTaskCntLastLoop = hasDummyHead ? 1 : PING_PONG_STAGES;
            lastLoopTaskCnt = taskNum - lastLoopTaskBegin;
            if (lastLoopTaskCnt >= maxTaskCntLastLoop) {
                lastLoopTaskCnt = maxTaskCntLastLoop;
            }
        }
        curLoopTaskCnt = taskLoops > 1 ? PING_PONG_STAGES : lastLoopTaskCnt;
        curLoopIdx = -1; // -1: 第一次创建task时会将curLoopIdx加1
        taskIdx = curLoopTaskBegin + PING_PONG_STAGES; // 第一次创建task时重新初始化taskIdx
        isRunning = curLoopTaskBegin < taskNum;

    }


    CATLASS_DEVICE
    void InitNewStream(GDNFwdHStreamPreload& newStream) {
        newStream.vIdx = taskIdx / (batch * vNumHead);
        newStream.batchIdx = (taskIdx - newStream.vIdx * batch * vNumHead) / vNumHead;
        newStream.vHeadIdx = taskIdx % vNumHead;
        newStream.kHeadIdx = newStream.vHeadIdx / headGroups;
        newStream.shapeBatchIdx = isVariedLen ? 0 : newStream.batchIdx;
        newStream.tokenBatchIdx = isVariedLen ? newStream.batchIdx : 0;
        newStream.chunkOffset = isVariedLen ? gmNumChunks.GetValue(newStream.tokenBatchIdx) : 0;
        newStream.batchChunks = isVariedLen ? (gmNumChunks.GetValue(newStream.tokenBatchIdx + 1) - newStream.chunkOffset) : totalChunks;
        newStream.tokenOffset = isVariedLen ? gmNumSeq.GetValue(newStream.tokenBatchIdx) : 0;
        newStream.batchTokens = isVariedLen ? (gmNumSeq.GetValue(newStream.tokenBatchIdx + 1) - newStream.tokenOffset) : totalTokens;
        newStream.chunkIdx = 0;
    }
     
    CATLASS_DEVICE
    void UpdateTask(uint32_t streamId) {
        auto& stream = runningQ.streams[streamId];
        auto& offset = stream.offset;
     
        offset.isInitialState = stream.chunkIdx == 0; 
        offset.isFinalState = stream.chunkIdx == (stream.batchChunks - 1); 
        offset.initialStateOffset = (stream.batchIdx * vNumHead + stream.vHeadIdx) * kHeadDim * vHeadDim; 
        offset.finalStateOffset = (stream.batchIdx * vNumHead + stream.vHeadIdx) * kHeadDim * vHeadDim; 
        offset.hSrcOffset = ((stream.shapeBatchIdx * totalChunks + stream.chunkOffset + stream.chunkIdx) * vNumHead + stream.vHeadIdx) * kHeadDim * vHeadDim;
        offset.hDstOffset = offset.hSrcOffset + vNumHead * kHeadDim * vHeadDim;
        offset.uvOffset = (stream.shapeBatchIdx * vNumHead * totalTokens + stream.vHeadIdx * totalTokens + stream.tokenOffset + stream.chunkIdx * chunkSize) * vHeadDim;
        offset.wkOffset = (stream.shapeBatchIdx * kNumHead * totalTokens + stream.kHeadIdx * totalTokens + stream.tokenOffset + stream.chunkIdx * chunkSize) * kHeadDim;
        offset.wOffset = (stream.shapeBatchIdx * vNumHead * totalTokens + stream.vHeadIdx * totalTokens + stream.tokenOffset + stream.chunkIdx * chunkSize) * kHeadDim;
        offset.gOffset = stream.shapeBatchIdx * vNumHead * totalTokens + stream.vHeadIdx * totalTokens + stream.tokenOffset + stream.chunkIdx * chunkSize;
        offset.hWorkOffset = (cubeCoreIdx * PING_PONG_STAGES + streamId) * kHeadDim * vHeadDim;
        offset.vWorkOffset = (cubeCoreIdx * PING_PONG_STAGES + streamId) * chunkSize * vHeadDim;
        offset.blockTokens = offset.isFinalState ? (stream.batchTokens - stream.chunkIdx * chunkSize) : chunkSize;
        offset.batchIdx = stream.batchIdx; 
        offset.headIdx = stream.vHeadIdx; 
        offset.chunkIdx = stream.chunkIdx;
    }
     
    CATLASS_DEVICE
    void InitTasks() {
        //auto oldHead = runningQ.head;
        auto streamId = runningQ.head;
        for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
            //auto streamId = (oldHead + i) % PING_PONG_STAGES;
            auto& stream = runningQ.streams[streamId];
            stream.chunkIdx += 1;
            if (StreamIsDone(stream)) {
                // 当前stream已完成，用一个新stream替换它
                taskIdx += 1;
                if (taskIdx >= (curLoopTaskBegin + curLoopTaskCnt)) {
                    curLoopIdx += 1;
                    // curLoopTaskBegin = curLoopIdx * PING_PONG_STAGES * cubeCoreNum + PING_PONG_STAGES * cubeCoreIdx;
                    // if (curLoopIdx + 1 >= taskLoops) {
                    //     curLoopTaskCnt = lastLoopTaskCnt;
                    // }
                    if (curLoopIdx + 1 < taskLoops) {
                        curLoopTaskBegin = curLoopIdx * PING_PONG_STAGES * cubeCoreNum + PING_PONG_STAGES * cubeCoreIdx;
                    } else {
                        curLoopTaskBegin = curLoopIdx * PING_PONG_STAGES * cubeCoreNum + (hasDummyHead ? 1 : PING_PONG_STAGES) * cubeCoreIdx;
                        curLoopTaskCnt = lastLoopTaskCnt;
                    }
                    taskIdx = curLoopTaskBegin;
                }
     
                //runningQ.head = (streamId + 1) % PING_PONG_STAGES;
                stream.batchChunks = 0;
                if (taskIdx < taskNum) {
                    InitNewStream(stream);
                    UpdateTask(streamId);
                } 

                if (streamId == runningQ.head) {
                    runningQ.head = (runningQ.head + 1) % PING_PONG_STAGES;
                    if (taskIdx >= taskNum) {
                        // 没有新stream了，将head推进到下一个未完成的stream上
                        for (uint32_t j = 0; j < PING_PONG_STAGES && StreamIsDone(runningQ.streams[runningQ.head]); ++j) {
                            runningQ.head = (runningQ.head + 1) % PING_PONG_STAGES;
                        }
                        isRunning = ! StreamIsDone(runningQ.streams[runningQ.head]);
                    }
                    streamId = runningQ.head;
                }
                else {
                    streamId = (streamId + 1) % PING_PONG_STAGES;
                }

            } else {
                UpdateTask(streamId);
                streamId = (streamId + 1) % PING_PONG_STAGES;
            }
        }
    }
    
    CATLASS_DEVICE
    const GDNFwdHStreamPreload& GetStream(uint32_t i) const {
        return runningQ.streams[(runningQ.head + i) % PING_PONG_STAGES];
    }

    CATLASS_DEVICE
    const GDNFwdHOffsetsPreload& GetCurTaskOffsets(const GDNFwdHStreamPreload& stream) const {
        return stream.offset;
    }

    CATLASS_DEVICE
    bool StreamIsDone(const GDNFwdHStreamPreload& stream) const {
        return stream.chunkIdx >= stream.batchChunks;
    }

    CATLASS_DEVICE
    bool NeedProcessStage2(const GDNFwdHStreamPreload& stream) {
        return storeFinalState || !stream.offset.isFinalState;
    }
};

struct BlockSchedulerGdnFwdHPreloadCube : public BlockSchedulerGdnFwdHPreload {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdHPreloadCube() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling, GM_ADDR user) {
        BlockSchedulerGdnFwdHPreload::Init(cu_seqlens, chunk_indices, tiling, user, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    }

};

struct BlockSchedulerGdnFwdHPreloadVec : public BlockSchedulerGdnFwdHPreload {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdHPreloadVec() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling, GM_ADDR user) {
        BlockSchedulerGdnFwdHPreload::Init(cu_seqlens, chunk_indices, tiling, user, AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(), AscendC::GetBlockNum());
    }

};

}  // namespace Catlass::Gemm::Block

#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#endif // ARCH22_CHUNK_FWD_H_UTILS_H
