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

// Design §6.1: A resident 8KiB, then 32KiB ping-pong regions for kbg/vb.
constexpr uint32_t kL1AOffset = 0;
constexpr uint32_t kL1KbgSlot0Offset = 8 * 1024;
constexpr uint32_t kL1KbgSlot1Offset = 40 * 1024;
constexpr uint32_t kL1VbSlot0Offset = 72 * 1024;
constexpr uint32_t kL1VbSlot1Offset = 104 * 1024;

// Mix 1 AIC : 2 AIV. Working Cube cores pre-post two free credits so both
// AIVs can Wait before the first C0 Set. Idle cores must not pre-post:
// unmatched CrossCoreSetFlag (>15) freezes the chip.
constexpr uint8_t kChunkReadyFlag = 6;
constexpr uint8_t kChunkFreeFlag = 7;

template <pipe_t PIPE>
__aicore__ inline void AivWaitChunkFree()
{
    (void)PIPE;
    AscendC::CrossCoreWaitFlag(kChunkFreeFlag);
}

template <pipe_t PIPE>
__aicore__ inline void AivSetChunkReady()
{
    AscendC::CrossCoreSetFlag<0x2, PIPE>(kChunkReadyFlag);
}

template <pipe_t PIPE>
__aicore__ inline void AicWaitChunkReady()
{
    (void)PIPE;
    AscendC::CrossCoreWaitFlag(kChunkReadyFlag);
}

template <pipe_t PIPE>
__aicore__ inline void AicSetChunkFree()
{
    AscendC::CrossCoreSetFlag<0x2, PIPE>(kChunkFreeFlag);
}

constexpr int32_t kEventA = 0;
constexpr int32_t kEventL0A = 0;
constexpr int32_t kEventL0B = 1;
constexpr int32_t kEventL0C = 0;
constexpr int32_t kEventMte1M = 0;

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
    for (uint32_t colOffset = 0; colOffset < cols; colOffset += c0Elems) {
        const uint32_t l1Offset =
            (colOffset / c0Elems) * paddedRows * c0Elems + rowOffset * c0Elems;
        AscendC::DataCopy(dstL1[l1Offset], srcUb[colOffset], params, enhanced);
    }
}

} // namespace KdaBwdRecomputeArch35

#endif // CHUNK_KDA_BWD_RECOMPUTE_ARCH35_COMMON_H
