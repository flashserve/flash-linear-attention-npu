/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_bwd_dhu_common.h
 * \brief Common kernel helpers for chunk_gated_delta_rule_bwd_dhu.
 */

#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_COMMON_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_COMMON_H

#include "catlass/arch/cross_core_sync.hpp"
#include "chunk_gated_delta_rule_bwd_dhu_struct.h"
#include "kernel_operator.h"

namespace GDN {

constexpr uint64_t VEC_TO_CUBE_FLAG_READY = 2;
constexpr uint64_t CUBE_TO_VEC_FLAG_READY = 4;
// dh（GEMM0 唯一跨核输入）就绪专用 mode-2 flag。
// flagId 命名空间：mode-2 已占用 {2,4}，取未占用的 3。
constexpr uint64_t DH_READY_FLAG = 3;
constexpr int64_t HEADS_PER_TASK = 4;
constexpr int64_t WORKSPACE_BUFFER_COUNT = 8;
// 同步拆分/early-notify 仅在单 task chunk 链足够深时有利。
// cube/vector 两侧必须用同一判据（同 task 的
// chunkCnt 相同），否则 mode-2 计数错配死锁。
constexpr int64_t SYNC_SPLIT_MIN_CHUNKS = 32;

struct ChunkInfo {
    int64_t seqIdx = 0;
    int64_t chunkIdx = 0;
    int64_t bIdx = 0;
    int64_t tokenStart = 0;
    int64_t chunkLen = 0;
    int64_t outputChunkIdx = 0;
    bool valid = false;
};

struct SeqInfo {
    int64_t seqIdx = 0;
    int64_t bIdx = 0;
    int64_t tokenStart = 0;
    int64_t tokenEnd = 0;
    int64_t chunkCnt = 0;
    int64_t outputChunkBase = 0;
    bool valid = false;
};

__aicore__ inline int64_t Min(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

__aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return b == 0 ? 0 : (a + b - 1) / b;
}

__aicore__ inline void GetSeqInfo(
    GM_ADDR cuSeqlens, const ChunkGatedDeltaRuleBwdDhuTilingData &tiling, int64_t seqIdx, SeqInfo &seqInfo)
{
    seqInfo.valid = false;
    seqInfo.seqIdx = seqIdx;
    seqInfo.bIdx = 0;
    seqInfo.tokenStart = 0;
    seqInfo.tokenEnd = 0;
    seqInfo.chunkCnt = 0;
    seqInfo.outputChunkBase = 0;

    if (cuSeqlens == nullptr) {
        if (seqIdx < 0 || seqIdx >= tiling.B) {
            return;
        }

        seqInfo.bIdx = seqIdx;
        seqInfo.tokenStart = 0;
        seqInfo.tokenEnd = tiling.T;
        seqInfo.chunkCnt = tiling.chunkNumForT;
        seqInfo.valid = seqInfo.chunkCnt > 0;
        return;
    }

    if (seqIdx < 0 || seqIdx >= tiling.seqNum) {
        return;
    }

    AscendC::GlobalTensor<int64_t> cuSeqlensTensor;
    cuSeqlensTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
    int64_t prev = cuSeqlensTensor.GetValue(0);
    if (prev < 0 || prev > tiling.T) {
        return;
    }

    int64_t outputChunkBase = 0;
    for (int64_t curSeq = 0; curSeq < seqIdx; ++curSeq) {
        const int64_t next = cuSeqlensTensor.GetValue(curSeq + 1);
        if (next < prev || next > tiling.T) {
            return;
        }
        outputChunkBase += CeilDiv(next - prev, tiling.chunkSize);
        prev = next;
    }

    const int64_t seqEnd = cuSeqlensTensor.GetValue(seqIdx + 1);
    if (seqEnd < prev || seqEnd > tiling.T) {
        return;
    }

    seqInfo.bIdx = 0;
    seqInfo.tokenStart = prev;
    seqInfo.tokenEnd = seqEnd;
    seqInfo.chunkCnt = CeilDiv(seqEnd - prev, tiling.chunkSize);
    seqInfo.outputChunkBase = outputChunkBase;
    seqInfo.valid = seqInfo.chunkCnt > 0;
}

__aicore__ inline bool ChunkIndexMatches(
    GM_ADDR chunkIndices, int64_t outputIdx, int64_t seqIdx, int64_t chunkIdx)
{
    if (chunkIndices == nullptr || outputIdx < 0) {
        return false;
    }

    AscendC::GlobalTensor<int64_t> chunkIndicesTensor;
    chunkIndicesTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
    return chunkIndicesTensor.GetValue(2 * outputIdx) == seqIdx &&
           chunkIndicesTensor.GetValue(2 * outputIdx + 1) == chunkIdx;
}

__aicore__ inline int64_t FindVarlenChunkOutputIdx(
    GM_ADDR chunkIndices, const ChunkGatedDeltaRuleBwdDhuTilingData &tiling, int64_t seqIdx, int64_t chunkIdx)
{
    if (chunkIndices == nullptr) {
        return -1;
    }

    AscendC::GlobalTensor<int64_t> chunkIndicesTensor;
    chunkIndicesTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
    for (int64_t outputIdx = 0; outputIdx < tiling.totalChunkNum; ++outputIdx) {
        if (chunkIndicesTensor.GetValue(2 * outputIdx) == seqIdx &&
            chunkIndicesTensor.GetValue(2 * outputIdx + 1) == chunkIdx) {
            return outputIdx;
        }
    }
    return -1;
}

__aicore__ inline void GetChunkInfoBySeqChunk(
    GM_ADDR chunkIndices, const ChunkGatedDeltaRuleBwdDhuTilingData &tiling,
    const SeqInfo &seqInfo, int64_t localChunkIdx, ChunkInfo &chunkInfo)
{
    chunkInfo.valid = false;
    chunkInfo.seqIdx = seqInfo.seqIdx;
    chunkInfo.chunkIdx = localChunkIdx;
    chunkInfo.bIdx = 0;
    chunkInfo.tokenStart = 0;
    chunkInfo.chunkLen = 0;
    chunkInfo.outputChunkIdx = 0;

    if (!seqInfo.valid || localChunkIdx < 0 || localChunkIdx >= seqInfo.chunkCnt) {
        return;
    }

    const int64_t tokenStart = seqInfo.tokenStart + localChunkIdx * tiling.chunkSize;
    const int64_t tokenEnd = Min(tokenStart + tiling.chunkSize, seqInfo.tokenEnd);
    if (tokenStart < seqInfo.tokenStart || tokenStart >= seqInfo.tokenEnd || tokenEnd <= tokenStart) {
        return;
    }

    int64_t outputChunkIdx = localChunkIdx;
    if (chunkIndices != nullptr) {
        outputChunkIdx = seqInfo.outputChunkBase + localChunkIdx;
        if (outputChunkIdx >= tiling.totalChunkNum ||
            !ChunkIndexMatches(chunkIndices, outputChunkIdx, seqInfo.seqIdx, localChunkIdx)) {
            outputChunkIdx = FindVarlenChunkOutputIdx(chunkIndices, tiling, seqInfo.seqIdx, localChunkIdx);
        }
        if (outputChunkIdx < 0) {
            return;
        }
    }

    chunkInfo.bIdx = seqInfo.bIdx;
    chunkInfo.tokenStart = tokenStart;
    chunkInfo.chunkLen = tokenEnd - tokenStart;
    chunkInfo.outputChunkIdx = outputChunkIdx;
    chunkInfo.valid = true;
}

// 变长前缀增量缓存。task 循环内 taskIdx 严格递增 → seqIdx 单调不减，
// 相邻 task 的 GetSeqInfo 重复扫描 cuSeqlens[0..seqIdx]（每次 O(seqIdx) 的
// GetValue 往返）。缓存上次的 {seqIdx, tokenStart, outputChunkBase}，seqIdx
// 不变直接复用，前进则从缓存点增量累积。缓存初值 seqIdx=-1 表示无效。
// 固定长度模式（cuSeqlens == nullptr）不走此路径。
struct SeqInfoCache {
    int64_t seqIdx = -1;
    int64_t tokenStart = 0;
    int64_t outputChunkBase = 0;
};

__aicore__ inline void GetSeqInfoCached(
    GM_ADDR cuSeqlens, const ChunkGatedDeltaRuleBwdDhuTilingData &tiling, int64_t seqIdx,
    SeqInfoCache &cache, SeqInfo &seqInfo)
{
    if (cuSeqlens == nullptr) {
        GetSeqInfo(cuSeqlens, tiling, seqIdx, seqInfo);
        return;
    }
    if (seqIdx < 0 || seqIdx >= tiling.seqNum) {
        GetSeqInfo(cuSeqlens, tiling, seqIdx, seqInfo);
        return;
    }

    AscendC::GlobalTensor<int64_t> cuSeqlensTensor;
    cuSeqlensTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));

    int64_t curSeq = 0;
    int64_t prev = 0;
    int64_t outputChunkBase = 0;
    if (cache.seqIdx >= 0 && seqIdx >= cache.seqIdx) {
        // 单调前进：从缓存点继续累积（seqIdx 相同则零扫描）
        curSeq = cache.seqIdx;
        prev = cache.tokenStart;
        outputChunkBase = cache.outputChunkBase;
        if (curSeq == seqIdx) {
            const int64_t seqEnd = cuSeqlensTensor.GetValue(seqIdx + 1);
            if (seqEnd < prev || seqEnd > tiling.T) {
                GetSeqInfo(cuSeqlens, tiling, seqIdx, seqInfo);
                return;
            }
            seqInfo.seqIdx = seqIdx;
            seqInfo.bIdx = 0;
            seqInfo.tokenStart = prev;
            seqInfo.outputChunkBase = outputChunkBase;
            seqInfo.chunkCnt = CeilDiv(seqEnd - prev, tiling.chunkSize);
            seqInfo.tokenEnd = seqEnd;
            seqInfo.valid = seqInfo.chunkCnt > 0;
            return;
        }
    } else {
        prev = cuSeqlensTensor.GetValue(0);
        if (prev < 0 || prev > tiling.T) {
            GetSeqInfo(cuSeqlens, tiling, seqIdx, seqInfo);
            return;
        }
    }

    bool failed = false;
    while (curSeq < seqIdx) {
        const int64_t next = cuSeqlensTensor.GetValue(curSeq + 1);
        if (next < prev || next > tiling.T) {
            failed = true;
            break;
        }
        outputChunkBase += CeilDiv(next - prev, tiling.chunkSize);
        prev = next;
        ++curSeq;
    }
    if (failed) {
        GetSeqInfo(cuSeqlens, tiling, seqIdx, seqInfo);
        return;
    }

    const int64_t seqEnd = cuSeqlensTensor.GetValue(seqIdx + 1);
    if (seqEnd < prev || seqEnd > tiling.T) {
        GetSeqInfo(cuSeqlens, tiling, seqIdx, seqInfo);
        return;
    }

    // cache 须在 seqEnd 校验通过后再写入，否则非法 cuSeqlens 会把
    // 未验证的 {seqIdx, tokenStart, outputChunkBase} 留给后续 task 命中，
    // 绕过越界检查构造出 tokenEnd > T 的 SeqInfo。
    cache.seqIdx = seqIdx;
    cache.tokenStart = prev;
    cache.outputChunkBase = outputChunkBase;

    seqInfo.seqIdx = seqIdx;
    seqInfo.bIdx = 0;
    seqInfo.tokenStart = prev;
    seqInfo.tokenEnd = seqEnd;
    seqInfo.chunkCnt = CeilDiv(seqEnd - prev, tiling.chunkSize);
    seqInfo.outputChunkBase = outputChunkBase;
    seqInfo.valid = seqInfo.chunkCnt > 0;
}

} // namespace GDN

#endif // CHUNK_GATED_DELTA_RULE_BWD_DHU_COMMON_H
