/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_fwd_h_tiling_processor.h
 * \brief Tiling computation decoupled from gert::TilingContext, reusable by both the
 *        aclnn tiling entry and the fast kernel launch C++ extension.
 *
 * The caller is responsible for resolving framework-specific information (shapes, dtypes,
 * platform core number, lib-api workspace size) into the plain context struct below. The
 * processor then fills the plain ChunkGatedDeltaRuleFwdHTilingData together with the block
 * dim and the total workspace size, mirroring exactly the original Tiling4ChunkGatedDeltaRuleFwdH.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_H_TILING_PROCESSOR_H
#define CHUNK_GATED_DELTA_RULE_FWD_H_TILING_PROCESSOR_H

#include <cstddef>
#include <cstdint>

#include "../op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"

namespace optiling {

static constexpr size_t GDN_FWD_H_WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
static constexpr size_t GDN_FWD_H_GM_ALIGN = 512;
static constexpr int64_t GDN_FWD_H_WORKSPACE_BUFFER_COUNT = 8;

inline bool ResolveFwdHHeadSharding(
    int64_t vNumHead, uint32_t availableCoreNum,
    uint32_t &maxHeadsPerCore, uint32_t &activeCoreNum)
{
    if (vNumHead <= 0 || availableCoreNum == 0) {
        return false;
    }
    maxHeadsPerCore = static_cast<uint32_t>(
        (vNumHead + static_cast<int64_t>(availableCoreNum) - 1) /
        static_cast<int64_t>(availableCoreNum));
    activeCoreNum = static_cast<uint32_t>(
        (vNumHead + static_cast<int64_t>(maxHeadsPerCore) - 1) /
        static_cast<int64_t>(maxHeadsPerCore));
    return activeCoreNum > 0 && activeCoreNum <= availableCoreNum;
}

// Plain, framework-agnostic inputs needed to compute the tiling.
struct ChunkGatedDeltaRuleFwdHTilingContext {
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
    // Host-only rolling-state storage contract. These fields are intentionally
    // not serialized into ChunkGatedDeltaRuleFwdHTilingData.
    size_t stateElementBytes = 0;
    bool useSeparateRollingState = false;
    int64_t chunkSize;
    // platform
    uint32_t aicCoreNum;
    size_t libApiWorkSpaceSize;
};

class ChunkGatedDeltaRuleFwdHTilingProcessor {
public:
    explicit ChunkGatedDeltaRuleFwdHTilingProcessor(const ChunkGatedDeltaRuleFwdHTilingContext &ctx) : ctx_(ctx) {}

    // Fills the plain tiling struct, the block dim and the total workspace size.
    void Process(::ChunkGatedDeltaRuleFwdHTilingData &tiling, uint32_t &blockDim, size_t &workspaceSize) const
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

        uint32_t maxHeadsPerCore = 0;
        uint32_t activeCoreNum = 0;
        if (!ResolveFwdHHeadSharding(
                ctx_.vNumHead, ctx_.aicCoreNum, maxHeadsPerCore, activeCoreNum)) {
            blockDim = 0;
            workspaceSize = 0;
            return;
        }
        (void)maxHeadsPerCore;
        blockDim = activeCoreNum;
        const int64_t aicCoreNum = static_cast<int64_t>(activeCoreNum);
        const int64_t chunkSize = ctx_.chunkSize;
        const int64_t kHeadDim = ctx_.kHeadDim;
        const int64_t vHeadDim = ctx_.vHeadDim;

        size_t workspaceOffset = ctx_.libApiWorkSpaceSize;
        workspaceOffset += GDN_FWD_H_WORKSPACE_RSV_BYTE;

        tiling.vWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(aicCoreNum * chunkSize * vHeadDim * static_cast<int64_t>(sizeof(float)) * GDN_FWD_H_WORKSPACE_BUFFER_COUNT));

        tiling.vUpdateWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(aicCoreNum * chunkSize * vHeadDim * static_cast<int64_t>(sizeof(float)) * GDN_FWD_H_WORKSPACE_BUFFER_COUNT));


        tiling.hWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>(aicCoreNum * kHeadDim * vHeadDim * static_cast<int64_t>(sizeof(float)) * GDN_FWD_H_WORKSPACE_BUFFER_COUNT));

        tiling.numSeqWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>((tokenBatch + 1) * static_cast<int64_t>(sizeof(int64_t))));

        tiling.numChunksWorkspaceOffset = static_cast<int64_t>(workspaceOffset);
        workspaceOffset += AlignUp(static_cast<size_t>((tokenBatch + 1) * static_cast<int64_t>(sizeof(int64_t))));

        // The hidden rolling state starts immediately after the aligned
        // numChunks buffer. The device entry can derive the same address from
        // numChunksWorkspaceOffset and tokenBatch, so no tiling ABI field is needed.
        if (ctx_.useSeparateRollingState && !ctx_.storeFinalState) {
            const size_t rollingStateBytes = static_cast<size_t>(batch) *
                static_cast<size_t>(ctx_.vNumHead) * static_cast<size_t>(kHeadDim) *
                static_cast<size_t>(vHeadDim) * ctx_.stateElementBytes;
            workspaceOffset += AlignUp(rollingStateBytes);
        }

        workspaceOffset += GDN_FWD_H_WORKSPACE_RSV_BYTE;
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
    // Mirrors the original "(x + GM_ALIGN) / GM_ALIGN * GM_ALIGN" alignment.
    static size_t AlignUp(size_t x)
    {
        return (x + GDN_FWD_H_GM_ALIGN) / GDN_FWD_H_GM_ALIGN * GDN_FWD_H_GM_ALIGN;
    }

    const ChunkGatedDeltaRuleFwdHTilingContext &ctx_;
};

} // namespace optiling

#endif // CHUNK_GATED_DELTA_RULE_FWD_H_TILING_PROCESSOR_H
