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
 * \brief Common A5 kernel helpers for chunk_gated_delta_rule_bwd_dhu.
 */

#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_COMMON_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_COMMON_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "catlass/arch/cross_core_sync.hpp"
#include "chunk_gated_delta_rule_bwd_dhu_struct.h"
#include "kernel_operator.h"

#include <type_traits>

namespace GDN {

// Dhu-C2b：GM-dvState 路径判定（= phase2 需要消费 loop1 的 GM 产物）。
// vector/cube 两侧必须用本函数对同一 (DT, V, chunkLen) 逐 chunk 推导，保证 cube loop1 末 flag4 set
// 与 vector phase2 头 flag4 wait 严格配对——两侧条件不一致将信用失衡导致冻核（本项最高危点）。
// 判定公式与 useGmDvState（cube.h / vector.h 中 V==256 && chunkLen>64）必须保持同一，改动需同步。
// CV 路径（返回 false）phase2 仅读 dvState CV 片（bank flag 6/7 逐片 gate）与 dv 输入 GM（入口即稳定），
// 不读任何 loop1 GM 产物，故 loop1 末 flag4 set 与 phase2 头 wait 均可省。
template <typename DT>
__aicore__ inline bool IsGmDvStatePath(int64_t v, int64_t chunkLen)
{
    if constexpr (std::is_same<DT, bfloat16_t>::value) {
        return v == 256 && chunkLen > 64;
    }
    return true; // fp16：dvState 恒走 GM（cube/vector 的 fp16 分支均读 GM）
}

// Dhu-C5：劈分头判定（单一共享谓词——vector IsSplitHead / cube CvTargetSubBlock / dvState
// tokenHalf 三处消费必须同式，R-C5-1 配对纪律）。
// headCnt>1 门控（C5a 形状矩阵回归修复）：headCnt==1（如 H=8、headsPerTask=1）时劈分在
// 形状矩阵实测中回归（shape_t512_h8 六张 FAIL、t100_h8 trap），根因待 dump 实验定位；
// 退化回奇偶属主后各调用点与 pre-C5a 逐拍一致（+C2b bisect 已证该路径正确）。
// headCnt∈{2,4} 本就不劈分；故实际仅 headCnt==3（含 H=96 模型档）启用劈分。
__aicore__ inline bool IsDhuSplitHead(int64_t headCnt, int64_t headOffset)
{
    return headCnt > 1 && (headCnt & 1) == 1 && headOffset == headCnt - 1;
}

// Dhu-C5b 修复：劈分半界按 16 对齐（=min(RoundUp(⌊k/2⌋,16), k)）。L0C Fixpipe 源偏移必须落在
// fractal（16 行）边界——非 16 对齐起始偏移的 fixpipe 使 FIX 管停滞、数据 flag 永不置、全核互等
// 死锁（t8191_h96 末 chunk chunkLen=63 半界 31 实锤 507014；t100_h8 chunkLen=36 半界 18 同征）。
// k≤16 时半界=k（子块1 空段）；k=1 时半界=0（子块0 空段，保持原 ⌊1/2⌋=0 语义）；
// k=K=128 时半界=64 与原 K/2 相同。
// CvTargetSubBlock / cube dvState tokenHalf / vector TokenRowRange 三处共用（R-C5-1 单一谓词纪律）。
__aicore__ inline int64_t DhuSplitHalf(int64_t k)
{
    const int64_t half = (k / 2 + 15) / 16 * 16;
    return half < k ? half : k;
}

// Dhu-C5a/C5b：CV 片目标子块判定（cube 专用；AIV 侧无需对偶——各 AIV 只等自己的 bank flag）。
// 两种用法同一公式：C5-a termW 流传 k=K（K 维半界 DhuSplitHalf(K)=64）；C5-b dvState 流传
// k=chunkLen（token 维半界 DhuSplitHalf(chunkLen)，调用方须在半界处截断 cvRows 使片不跨界）。
// 劈分头（IsDhuSplitHead；subBlockNum==2 由 KERNEL_TYPE_MIX_AIC_1_2 保证）：rowIdx < 半界 → 子块0，
// 否则子块1；非劈分头维持属主 headOffset&1。
// AIC 侧 cvListId 必须按目标子块独立 ping-pong（cvListId[2]），与各 AIV 局部 cvListId
// （各自从 0 起逐片翻转）逐片配对——目标或计数错位即错子块收片/信用失衡/冻核（R-C5-1）。
__aicore__ inline uint32_t CvTargetSubBlock(int64_t headCnt, int64_t headOffset, uint32_t rowIdx, int64_t k)
{
    if (IsDhuSplitHead(headCnt, headOffset)) {
        return rowIdx < static_cast<uint32_t>(DhuSplitHalf(k)) ? 0U : 1U;
    }
    return static_cast<uint32_t>(headOffset & 1);
}

constexpr uint64_t VEC_TO_CUBE_FLAG_READY = 2;
// Dhu-C2a：dh ready / qg ready 拆分。flag3=dhReady（AIV 在 state/dh K 行循环结束即 set），
// flag2 语义收窄为 qgReady（AIV 在 qg staging 完成后 set）。ID 3 不与 {0,1,6,7} 的 CV 通道及
// {8,9,10} 的 catlass barrier 撞号；每 chunk set/wait 各 3 对 3（AND 配对），远小于 15 次上限。
constexpr uint64_t VEC_TO_CUBE_DH_FLAG_READY = 3;
constexpr uint64_t CUBE_TO_VEC_FLAG_READY = 4;
constexpr uint32_t CV_BUFFER_COUNT = 2;
constexpr uint64_t CV_SUBBLOCK_FLAG_STRIDE = 16;
constexpr uint64_t MATRIX_CV_AIV_TO_AIC_FLAG_BEGIN = 0;
constexpr uint64_t MATRIX_CV_AIC_TO_AIV_FLAG_BEGIN = 6;
constexpr int64_t HEADS_PER_TASK = 4;
constexpr int64_t WORKSPACE_BUFFER_COUNT = 8;

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

} // namespace GDN

#endif // CHUNK_GATED_DELTA_RULE_BWD_DHU_COMMON_H
