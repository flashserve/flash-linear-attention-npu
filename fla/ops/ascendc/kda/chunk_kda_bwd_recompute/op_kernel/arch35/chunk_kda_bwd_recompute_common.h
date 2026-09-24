/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_ARCH35_COMMON_H
#define CHUNK_KDA_BWD_RECOMPUTE_ARCH35_COMMON_H

#include "../chunk_kda_bwd_recompute_common.h"

namespace KdaBwdRecomputeArch35 {

constexpr uint32_t kBt = 64;
constexpr uint32_t kK = 128;
constexpr uint32_t kV = 128;
// Mix-core software pipeline: 2 AIV × 2-deep UB = 4 heads in flight.
constexpr uint32_t kHeadRotate = 4;

// Contiguous chunk-major (chunk, head) tasks per Mix core. Cyclic
// `task += coreNum` jumped 28 heads and hid the inner 4-head rotation.
__aicore__ inline void CoreTaskRange(
    uint32_t coreIdx, uint32_t coreNum, uint64_t total,
    uint64_t &begin, uint64_t &end)
{
    // Mix 1:2 may still launch all 28 AIC even when usedAic < 28.
    // coreIdx >= coreNum must be empty, not a wrapped extra chunk.
    if (coreNum == 0U || coreIdx >= coreNum || total == 0) {
        begin = 0;
        end = 0;
        return;
    }
    const uint64_t n = total / coreNum;
    const uint64_t r = total % coreNum;
    if (coreIdx < r) {
        begin = coreIdx * (n + 1U);
        end = begin + n + 1U;
    } else {
        begin = r * (n + 1U) + (coreIdx - r) * n;
        end = begin + n;
    }
}

__aicore__ inline bool NextChunkHead(
    uint64_t loopIdx, uint64_t h, uint64_t hEnd, uint64_t lastLoop,
    uint64_t &nxtLoop, uint64_t &nxtH)
{
    if (h + 1U < hEnd) {
        nxtLoop = loopIdx;
        nxtH = h + 1U;
        return true;
    }
    if (loopIdx < lastLoop) {
        nxtLoop = loopIdx + 1U;
        nxtH = 0;
        return true;
    }
    return false;
}

// Chunk-major [taskBegin, taskEnd) → heads on this chunk. lastLoop uses
// (taskEnd-1)/hv so a range that ends on a chunk boundary still has hEnd=hv.
__aicore__ inline void HeadsOnChunk(
    uint64_t taskBegin, uint64_t taskEnd, uint64_t hv, uint64_t loopIdx,
    uint64_t firstLoop, uint64_t lastLoop, uint64_t &hStart, uint64_t &hEnd)
{
    hStart = (loopIdx == firstLoop) ? (taskBegin % hv) : 0;
    if (loopIdx != lastLoop) {
        hEnd = hv;
        return;
    }
    const uint64_t rem = taskEnd % hv;
    hEnd = (rem == 0) ? hv : rem;
}

// R-Akk 方案A：A 区 8KiB 单缓冲 → 2×32KiB 组 ping-pong。每组装载 ≤kHeadRotate=4 个头的
// NZ 矩阵（ndNum=G 单条 DataCopy，dstNzMatrixStride=kL1AMatrixElems elem，矩阵在组内连续落位）。
// 槽位不相交：[0,64K) A0/A1，[64K,128K) kbg×2，[128K,192K) vb×2，合计 192KiB / 512KiB。
constexpr uint32_t kL1AOffset = 0;
constexpr uint32_t kL1AGroupBytes = 32 * 1024;
constexpr uint32_t kL1AMatrixElems = 64 * 64; // 4096 elem = 8KiB bf16 单头 NZ 落点
constexpr uint32_t kL1AGroupMatrixElems = kHeadRotate * kL1AMatrixElems; // 16384 elem = 32KiB 组
constexpr uint32_t kL1KbgSlot0Offset = 64 * 1024;
constexpr uint32_t kL1KbgSlot1Offset = 96 * 1024;
constexpr uint32_t kL1VbSlot0Offset = 128 * 1024;
constexpr uint32_t kL1VbSlot1Offset = 160 * 1024;

// Mix 1 AIC : 2 AIV, mode 4: AIV1 is flagId+16. Untemplated CrossCoreWaitFlag
// defaults to mode 0 (inter-core) and serializes every AIC onto one flag.
// Cube waits only the producing AIV (slot 0/1). Do not wait both flags per
// head (deadlock) and do not pre-post free credits (Set without Wait >15
// freezes the chip).
constexpr uint8_t kCrossCoreModeIndep = 0x4;
constexpr uint8_t kChunkReadyFlag = 6;
constexpr uint8_t kChunkFreeFlag = 7;
constexpr uint8_t kSubBlockFlagOffset = 16;

template <pipe_t PIPE>
__aicore__ inline void AivWaitChunkFree()
{
    AscendC::CrossCoreWaitFlag<kCrossCoreModeIndep, PIPE>(kChunkFreeFlag);
}

template <pipe_t PIPE>
__aicore__ inline void AivSetChunkReady()
{
    AscendC::CrossCoreSetFlag<kCrossCoreModeIndep, PIPE>(kChunkReadyFlag);
}

template <pipe_t PIPE>
__aicore__ inline void AicWaitChunkReady(uint16_t slot)
{
    AscendC::CrossCoreWaitFlag<kCrossCoreModeIndep, PIPE>(
        static_cast<uint16_t>(kChunkReadyFlag + slot * kSubBlockFlagOffset));
}

template <pipe_t PIPE>
__aicore__ inline void AicSetChunkFree(uint16_t slot)
{
    AscendC::CrossCoreSetFlag<kCrossCoreModeIndep, PIPE>(
        static_cast<uint16_t>(kChunkFreeFlag + slot * kSubBlockFlagOffset));
}

// Streaming GM: do not pollute L2. On 3510 this is encoded in the pointer
// high bits and picked up by DataCopy / Fixpipe.
template <typename T>
__aicore__ inline void BypassL2(AscendC::GlobalTensor<T> &tensor)
{
    tensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
}

// R-Akk：kEventA 单 ID → 组级 ping/pong 双 ID（仿 kEventL0C0/kEventL0C1；MTE1_MTE2 与
// MTE2_MTE1 两个方向各用 {0,1}，预算充足）。信用配对：free[b] 预置 1 次 + 每组末头 set 1 次
// vs 每组装载 wait 1 次；ready[b] 每组装载 set 1 次 vs 每组首头 wait 1 次。
constexpr int32_t kEventA0 = 0;
constexpr int32_t kEventA1 = 1;
constexpr int32_t kEventL0A = 0;
constexpr int32_t kEventL0B = 1;
constexpr int32_t kEventL0C0 = 0;
constexpr int32_t kEventL0C1 = 1;
constexpr int32_t kEventMte1M = 0;
constexpr uint32_t kL0CTileBytes = 128 * 1024;

__aicore__ inline int32_t AEvent(uint32_t groupBuf)
{
    return groupBuf == 0 ? kEventA0 : kEventA1;
}

// R-Akk：组级「下一组」描述。组不跨 chunk（hBase 循环嵌套在 loopIdx 内）；跨 chunk 时取下一
// chunk 的首组（hStart 经 HeadsOnChunk 重算），调用方按 nxtLoop 重算 chunk 形状。
__aicore__ inline bool NextGroup(
    uint64_t loopIdx, uint64_t hBase, uint64_t hEnd, uint64_t lastLoop,
    uint64_t taskBegin, uint64_t taskEnd, uint64_t hv, uint64_t firstLoop,
    uint64_t &nxtLoop, uint64_t &nxtHBase, uint64_t &nxtHEnd)
{
    if (hBase + kHeadRotate < hEnd) {
        nxtLoop = loopIdx;
        nxtHBase = hBase + kHeadRotate;
        nxtHEnd = hEnd;
        return true;
    }
    if (loopIdx < lastLoop) {
        nxtLoop = loopIdx + 1U;
        HeadsOnChunk(taskBegin, taskEnd, hv, nxtLoop, firstLoop, lastLoop, nxtHBase, nxtHEnd);
        return true;
    }
    return false;
}

__aicore__ inline uint32_t KbgSlotOffset(uint32_t slot)
{
    return slot == 0 ? kL1KbgSlot0Offset : kL1KbgSlot1Offset;
}

__aicore__ inline uint32_t VbSlotOffset(uint32_t slot)
{
    return slot == 0 ? kL1VbSlot0Offset : kL1VbSlot1Offset;
}

// Row-major UB → zN L1 (same fractal scatter as bwd_dhu qg, matches Catlass RowMajor→zN).
template <typename T>
__aicore__ inline void CopyUbNdToL1Zn(
    AscendC::LocalTensor<T> dstL1, AscendC::LocalTensor<T> srcUb,
    uint32_t rows, uint32_t cols, uint32_t paddedRows, uint32_t rowOffset)
{
    constexpr uint32_t c0Elems = 32 / sizeof(T);
    AscendC::DataCopyEnhancedParams enhanced;
    enhanced.blockMode = AscendC::BlockMode::BLOCK_MODE_VECTOR;
    const AscendC::DataCopyParams params{
        static_cast<uint16_t>(rows), 1,
        static_cast<uint16_t>(cols / c0Elems - 1), 0};
    #pragma unroll 8
    for (uint32_t colOffset = 0; colOffset < cols; colOffset += c0Elems) {
        const uint32_t l1Offset =
            (colOffset / c0Elems) * paddedRows * c0Elems + rowOffset * c0Elems;
        AscendC::DataCopy(dstL1[l1Offset], srcUb[colOffset], params, enhanced);
    }
}

} // namespace KdaBwdRecomputeArch35

#endif // CHUNK_KDA_BWD_RECOMPUTE_ARCH35_COMMON_H
