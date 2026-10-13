/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_fwd_h_tiling_processor.h
 * \brief Tiling computation decoupled from gert::TilingContext.
 *
 * The caller is responsible for resolving framework-specific information (shapes, dtypes,
 * platform core number, lib-api workspace size) into the plain context struct below. The
 * processor then fills ChunkFwdHPlainTilingData together with the block
 * dim and the total workspace size, mirroring exactly the original Tiling4ChunkFwdH.
 */

#ifndef CHUNK_FWD_H_TILING_PROCESSOR_H
#define CHUNK_FWD_H_TILING_PROCESSOR_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../op_kernel/chunk_fwd_h_struct.h"

namespace optiling {

static constexpr size_t CHUNK_FWD_H_WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
static constexpr size_t CHUNK_FWD_H_GM_ALIGN = 512;
static constexpr int64_t CHUNK_FWD_H_ROUND_HEAD_SLOTS = 4;
static constexpr int64_t CHUNK_FWD_H_CHUNK_SIZE = 64;
static constexpr int64_t CHUNK_FWD_H_K_DIM = 128;
static constexpr int64_t CHUNK_FWD_H_V_DIM = 128;

// Plain, framework-agnostic inputs needed to compute the tiling.
struct ChunkFwdHTilingContext {
    // shapes
    int64_t seqlen;        // k.dim(2)
    int64_t kNumHead;      // k.dim(1)
    int64_t kHeadDim;      // k.dim(3)
    int64_t vNumHead;      // u.dim(1)
    int64_t vHeadDim;      // u.dim(3)
    int64_t shapeBatchDim; // k.dim(0)
    // variable length
    bool hasCuSeqlens;
    int64_t cuSeqlensDim0; // length of cu_seqlens (only used when hasCuSeqlens)
    bool useInitialState;
    // attrs
    bool storeFinalState;
    int64_t chunkSize;
    // P1: per-sequence chunk counts read from cu_seqlens on host (aclnn/l0op layer);
    // nullptr / zero count means dense path — kernel keeps uniform head partition.
    const int64_t *seqChunkCounts;
    int64_t seqCount;
    // platform
    uint32_t aicCoreNum;
    size_t libApiWorkSpaceSize;
};

class ChunkFwdHTilingProcessor {
public:
    explicit ChunkFwdHTilingProcessor(const ChunkFwdHTilingContext &ctx) : ctx_(ctx) {}

    // Fills the plain tiling struct, the block dim and the total workspace size.
    void Process(::ChunkFwdHPlainTilingData &tiling,
                 uint32_t &blockDim, size_t &workspaceSize) const
    {
        int64_t isVariedLen;
        int64_t shapeBatch;
        int64_t tokenBatch;
        int64_t batch;

        if (!ctx_.hasCuSeqlens) {
            isVariedLen = 0;
            shapeBatch = ctx_.shapeBatchDim;
            tokenBatch = 1;
            batch = shapeBatch;
        } else {
            isVariedLen = 1;
            shapeBatch = 1;
            tokenBatch = ctx_.cuSeqlensDim0 - 1;
            batch = tokenBatch;
        }

        const uint64_t effectiveSequenceCount = static_cast<uint64_t>(batch);
        const uint64_t totalHeadTasks =
            effectiveSequenceCount * static_cast<uint64_t>(ctx_.vNumHead);
        const uint64_t physicalAicCores = static_cast<uint64_t>(ctx_.aicCoreNum);
        const uint64_t headsPerCore =
            (totalHeadTasks + physicalAicCores - 1) / physicalAicCores;
        blockDim = static_cast<uint32_t>(
            (totalHeadTasks + headsPerCore - 1) / headsPerCore);
        const int64_t activeAicCoreNum = static_cast<int64_t>(blockDim);
        constexpr int64_t chunkSize = CHUNK_FWD_H_CHUNK_SIZE;
        constexpr int64_t kHeadDim = CHUNK_FWD_H_K_DIM;
        constexpr int64_t vHeadDim = CHUNK_FWD_H_V_DIM;

        FillBalancedCoreBounds(tiling, blockDim, totalHeadTasks);

        size_t workspaceOffset = ctx_.libApiWorkSpaceSize;
        workspaceOffset += CHUNK_FWD_H_WORKSPACE_RSV_BYTE;

        tiling.vWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(activeAicCoreNum * CHUNK_FWD_H_ROUND_HEAD_SLOTS *
                                                        chunkSize * vHeadDim * sizeof(float)));

        tiling.vUpdateWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(activeAicCoreNum * CHUNK_FWD_H_ROUND_HEAD_SLOTS *
                                                        chunkSize * vHeadDim * sizeof(uint16_t)));

        tiling.kDecayWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(activeAicCoreNum * CHUNK_FWD_H_ROUND_HEAD_SLOTS *
                                                        kHeadDim * vHeadDim * sizeof(float)));

        tiling.hWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(activeAicCoreNum * CHUNK_FWD_H_ROUND_HEAD_SLOTS *
                                                        kHeadDim * vHeadDim * sizeof(float)));

        workspaceOffset += CHUNK_FWD_H_WORKSPACE_RSV_BYTE;
        workspaceSize = workspaceOffset;

        tiling.batch = batch;
        tiling.seqlen = ctx_.seqlen;
        tiling.kNumHead = ctx_.kNumHead;
        tiling.vNumHead = ctx_.vNumHead;
        tiling.kHeadDim = ctx_.kHeadDim;
        tiling.vHeadDim = ctx_.vHeadDim;
        tiling.chunkSize = chunkSize;
        tiling.useInitialState = ctx_.useInitialState;
        tiling.storeFinalState = ctx_.storeFinalState;
        tiling.isVariedLen = isVariedLen;
        tiling.shapeBatch = shapeBatch;
        tiling.tokenBatch = tokenBatch;
    }

private:
    // P1 varlen balanced partition: cut the seq-major head-task sequence into
    // blockDim contiguous spans of near-equal total chunk weight (capacity
    // binary search + greedy fill). blockDim itself is unchanged, so workspace
    // size and core launch count are untouched — only per-core [begin, end)
    // move. Falls back to the uniform partition (coreBoundCount=0) when the
    // sequence weights are unavailable or degenerate (dense / fewer tasks
    // than cores / greedy produced fewer spans than cores).
    void FillBalancedCoreBounds(::ChunkFwdHPlainTilingData &tiling,
                                uint32_t blockDim, uint64_t totalHeadTasks) const
    {
        tiling.coreBoundCount = 0;
        for (int64_t &bound : tiling.coreTaskBounds) {
            bound = 0;
        }
        if (ctx_.seqChunkCounts == nullptr || ctx_.seqCount <= 0 ||
            blockDim == 0 || totalHeadTasks < blockDim) {
            return;
        }
        std::vector<int64_t> weights;
        weights.reserve(totalHeadTasks);
        for (int64_t seq = 0; seq < ctx_.seqCount; ++seq) {
            const int64_t chunks = ctx_.seqChunkCounts[seq];
            if (chunks <= 0) {
                return; // malformed host table — fall back to uniform partition
            }
            for (int64_t head = 0; head < ctx_.vNumHead; ++head) {
                weights.push_back(chunks);
            }
        }
        if (static_cast<uint64_t>(weights.size()) != totalHeadTasks) {
            return;
        }
        int64_t maxWeight = 0;
        int64_t totalWeight = 0;
        for (const int64_t w : weights) {
            maxWeight = w > maxWeight ? w : maxWeight;
            totalWeight += w;
        }
        // Binary search the minimal capacity whose greedy span count <= blockDim.
        int64_t lo = maxWeight;
        int64_t hi = totalWeight;
        while (lo < hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            if (GreedySpanCount(weights, mid) <= blockDim) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        // Greedy cut at the minimal capacity; force exactly blockDim spans by
        // opening a new span whenever the remaining tasks equal the remaining
        // spans to place (keeps every span non-empty).
        const int64_t capacity = lo;
        std::vector<int64_t> bounds(static_cast<size_t>(blockDim), 0);
        int64_t spanCount = 1;
        int64_t sum = 0;
        for (uint64_t task = 0; task < totalHeadTasks; ++task) {
            const uint64_t remainingTasks = totalHeadTasks - task;
            const int64_t remainingSpans = static_cast<int64_t>(blockDim) - spanCount;
            const bool mustSplit = sum + weights[task] > capacity ||
                remainingTasks <= static_cast<uint64_t>(remainingSpans);
            if (mustSplit && spanCount < static_cast<int64_t>(blockDim)) {
                bounds[static_cast<size_t>(spanCount - 1)] = static_cast<int64_t>(task);
                spanCount += 1;
                sum = 0;
            }
            sum += weights[task];
        }
        bounds[static_cast<size_t>(spanCount - 1)] = static_cast<int64_t>(totalHeadTasks);
        if (spanCount != static_cast<int64_t>(blockDim)) {
            return; // degenerate — uniform partition is equivalent here
        }
        for (uint32_t core = 0; core < blockDim; ++core) {
            tiling.coreTaskBounds[core] = bounds[core];
        }
        tiling.coreBoundCount = static_cast<int64_t>(blockDim);
    }

    static int64_t GreedySpanCount(const std::vector<int64_t> &weights, int64_t capacity)
    {
        int64_t spanCount = 1;
        int64_t sum = 0;
        for (const int64_t w : weights) {
            if (sum + w > capacity) {
                spanCount += 1;
                sum = 0;
            }
            sum += w;
        }
        return spanCount;
    }

    // Mirrors the original "(x + GM_ALIGN) / GM_ALIGN * GM_ALIGN" alignment.
    static size_t AlignUp(size_t x)
    {
        return (x + CHUNK_FWD_H_GM_ALIGN - 1) / CHUNK_FWD_H_GM_ALIGN * CHUNK_FWD_H_GM_ALIGN;
    }

    const ChunkFwdHTilingContext &ctx_;
};

} // namespace optiling

#endif // CHUNK_FWD_H_TILING_PROCESSOR_H
