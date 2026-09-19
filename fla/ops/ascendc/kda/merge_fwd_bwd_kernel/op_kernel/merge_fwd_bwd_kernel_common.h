/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef MERGE_FWD_BWD_KERNEL_COMMON_H
#define MERGE_FWD_BWD_KERNEL_COMMON_H

#include "merge_fwd_bwd_kernel_struct.h"
#include "kernel_operator.h"

namespace MergeFwBwd {

__aicore__ inline void CoreTaskRange(
    uint32_t coreIdx, uint32_t coreNum, uint64_t total, uint64_t &begin, uint64_t &end)
{
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

__aicore__ inline uint32_t RoundTaskCount(uint64_t roundBegin, uint64_t hEnd)
{
    const uint64_t remain = hEnd - roundBegin;
    return remain < static_cast<uint64_t>(kMaxTaskPerAic)
               ? static_cast<uint32_t>(remain)
               : kMaxTaskPerAic;
}

// AIV0: t=0,2 → local 0,1. AIV1: t=1,3 → local 0,1.
__aicore__ inline uint32_t TaskLocalSlot(uint32_t t)
{
    return t >> 1;
}

__aicore__ inline int64_t SrcRank(int64_t rank, int64_t N, int64_t i, bool forward, int64_t S)
{
    // Interface: S=1,N=1,rank=0 copies He[0] for both directions.
    if (S == 1 && N == 1) {
        return 0;
    }
    if (forward) {
        return rank - N + i;
    }
    return rank + N - i;
}

__aicore__ inline uint64_t AgHmHeadOffset(int64_t src, int64_t hv, int64_t h)
{
    return (static_cast<uint64_t>(src) * static_cast<uint64_t>(hv) + static_cast<uint64_t>(h)) *
           static_cast<uint64_t>(kKDim) * static_cast<uint64_t>(kRowStride);
}

__aicore__ inline void CSlotCoord(int64_t rank, int64_t slotIndex, int64_t &src, uint64_t &elemOff)
{
    const uint64_t linear =
        static_cast<uint64_t>(slotIndex) * static_cast<uint64_t>(kTileM) * static_cast<uint64_t>(kVDim);
    const uint64_t perHead = static_cast<uint64_t>(kKDim) * static_cast<uint64_t>(kRowStride);
    src = rank + static_cast<int64_t>(linear / perHead);
    elemOff = linear % perHead;
}

__aicore__ inline uint64_t SlotBase(uint32_t coreIdx, uint32_t slot)
{
    return static_cast<uint64_t>(coreIdx) * kCoreStride + static_cast<uint64_t>(slot) * kSlotBytes;
}

__aicore__ inline uint64_t CScratchElemOff(uint64_t h, int64_t tile, uint32_t usedAic)
{
    const uint64_t base =
        static_cast<uint64_t>(usedAic) * static_cast<uint64_t>(kCoreStride) / sizeof(float);
    return base +
           (h * static_cast<uint64_t>(kCTilesPerHead) + static_cast<uint64_t>(tile)) *
               static_cast<uint64_t>(kCTileElems);
}

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
// 910b / 910_93 Mix 1:2 Mode 0x2 (same FFTS aggregate as 950). Untemplated
// CrossCoreWaitFlag defaults to mode 0 (inter-core) and will hang or serialize
// every AIC — always pass <0x2, PIPE> on both Set and Wait (see chunk_fwd_h
// arch22). Idle AIV must DummyHandshake the same Ready/Free/C1 sequence.
constexpr uint8_t kA2CrossCoreMode = 0x2;
constexpr uint8_t kA2ChunkReadyFlag = 2;
constexpr uint8_t kA2ChunkFreeFlag = 4;
constexpr uint8_t kA2ChunkC1Flag = 3;

template <pipe_t PIPE>
__aicore__ inline void AivWaitChunkFree(uint16_t t)
{
    (void)t;
    AscendC::CrossCoreWaitFlag<kA2CrossCoreMode, PIPE>(kA2ChunkFreeFlag);
}

template <pipe_t PIPE>
__aicore__ inline void AivSetChunkReady(uint16_t t)
{
    (void)t;
    AscendC::CrossCoreSetFlag<kA2CrossCoreMode, PIPE>(kA2ChunkReadyFlag);
}

template <pipe_t PIPE>
__aicore__ inline void AicWaitChunkReady(uint16_t t)
{
    (void)t;
    AscendC::CrossCoreWaitFlag<kA2CrossCoreMode, PIPE>(kA2ChunkReadyFlag);
}

template <pipe_t PIPE>
__aicore__ inline void AicSetChunkFree(uint16_t t)
{
    (void)t;
    AscendC::CrossCoreSetFlag<kA2CrossCoreMode, PIPE>(kA2ChunkFreeFlag);
}

template <pipe_t PIPE>
__aicore__ inline void AivWaitChunkC1(uint16_t t)
{
    (void)t;
    AscendC::CrossCoreWaitFlag<kA2CrossCoreMode, PIPE>(kA2ChunkC1Flag);
}

template <pipe_t PIPE>
__aicore__ inline void AicSetChunkC1Done(uint16_t t)
{
    (void)t;
    AscendC::CrossCoreSetFlag<kA2CrossCoreMode, PIPE>(kA2ChunkC1Flag);
}
#endif

} // namespace MergeFwBwd

#endif
