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

// A resident 8KiB, then 32KiB ping-pong regions for kbg/vb.
constexpr uint32_t kL1AOffset = 0;
constexpr uint32_t kL1KbgSlot0Offset = 8 * 1024;
constexpr uint32_t kL1KbgSlot1Offset = 40 * 1024;
constexpr uint32_t kL1VbSlot0Offset = 72 * 1024;
constexpr uint32_t kL1VbSlot1Offset = 104 * 1024;

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

constexpr int32_t kEventA = 0;
constexpr int32_t kEventL0A = 0;
constexpr int32_t kEventL0B = 1;
constexpr int32_t kEventL0C0 = 0;
constexpr int32_t kEventL0C1 = 1;
constexpr int32_t kEventMte1M = 0;
constexpr uint32_t kL0CTileBytes = 128 * 1024;

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
