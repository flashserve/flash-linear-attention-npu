/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef CHUNK_FWD_O_ARCH92_COMMON_H
#define CHUNK_FWD_O_ARCH92_COMMON_H

#include <cstdint>
#include "kernel_operator.h"
#include "../chunk_fwd_o_struct.h"
#include "../chunk_fwd_o_a6_constants.h"

namespace GDN {

// Single AIV, three independent head slots. Byte intervals follow docs/design.md.
constexpr uint32_t CHUNK_FWD_O_L1_RESIDENT_HEAD_COUNT = CHUNK_FWD_O_A6_HEAD_COUNT;
constexpr uint32_t CHUNK_FWD_O_UB_GATE_BASE = 0U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_GATE_O_SLOT_BYTES = 256U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_GATE_A_BASE = 768U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_GATE_A_SLOT_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_G_SCRATCH_BASE = 49920U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_G_SCRATCH_SLOT_BYTES = 256U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_OSPRIME_BASE = 50688U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_OSPRIME_SLOT_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_ARAW_BASE = 148992U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_ARAW_SLOT_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_OL_BASE = 198144U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_OL_SLOT_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_APRIME_BF16_OFFSET = 296448U;
constexpr uint32_t CHUNK_FWD_O_UB_APRIME_BF16_SLOT_BYTES = 8U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_TOTAL_BYTES = AscendC::TOTAL_UB_SIZE;

__aicore__ inline uint32_t ChunkFwdOGateOOffset(uint32_t slot)
{
    return CHUNK_FWD_O_UB_GATE_BASE + slot * CHUNK_FWD_O_UB_LAYOUT_GATE_O_SLOT_BYTES;
}

__aicore__ inline uint32_t ChunkFwdOGateAOffset(uint32_t slot)
{
    return CHUNK_FWD_O_UB_LAYOUT_GATE_A_BASE + slot * CHUNK_FWD_O_UB_LAYOUT_GATE_A_SLOT_BYTES;
}

__aicore__ inline uint32_t ChunkFwdOGScratchOffset(uint32_t streamSlot)
{
    return CHUNK_FWD_O_UB_LAYOUT_G_SCRATCH_BASE + streamSlot * CHUNK_FWD_O_UB_LAYOUT_G_SCRATCH_SLOT_BYTES;
}

__aicore__ inline uint32_t ChunkFwdOARawOffset(uint32_t slot)
{
    return CHUNK_FWD_O_UB_LAYOUT_ARAW_BASE + slot * CHUNK_FWD_O_UB_LAYOUT_ARAW_SLOT_BYTES;
}

__aicore__ inline uint32_t ChunkFwdOOSRawOffset(uint32_t slot)
{
    return CHUNK_FWD_O_UB_LAYOUT_OSPRIME_BASE + slot * CHUNK_FWD_O_UB_LAYOUT_OSPRIME_SLOT_BYTES;
}

__aicore__ inline uint32_t ChunkFwdOOsPrimeOffset(uint32_t slot)
{
    return ChunkFwdOOSRawOffset(slot);
}

__aicore__ inline uint32_t ChunkFwdOAPrimeBf16Offset(uint32_t streamSlot)
{
    return CHUNK_FWD_O_UB_APRIME_BF16_OFFSET +
           streamSlot * CHUNK_FWD_O_UB_APRIME_BF16_SLOT_BYTES;
}

__aicore__ inline uint32_t ChunkFwdOOlOffset(uint32_t slot)
{
    return CHUNK_FWD_O_UB_LAYOUT_OL_BASE + slot * CHUNK_FWD_O_UB_LAYOUT_OL_SLOT_BYTES;
}

__aicore__ inline uint32_t ChunkFwdOHeadGroupNum(const ChunkFwdOTilingData &tiling)
{
    return static_cast<uint32_t>(
        (tiling.vNumHead + tiling.taskGroupSize - 1) / tiling.taskGroupSize);
}

constexpr uint32_t CHUNK_FWD_O_L1_TOTAL_BYTES = AscendC::TOTAL_L1_SIZE;
constexpr uint32_t CHUNK_FWD_O_L1_Q_BASE = 0U;
constexpr uint32_t CHUNK_FWD_O_L1_K_BASE = 48U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_H_BASE = 96U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_Q_SLOT_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_K_SLOT_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_H_SLOT_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_APRIME_BASE = 192U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_APRIME_SLOT_BYTES = 8U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_V_BASE = 216U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_V_SLOT_BYTES = 16U * 1024U;

constexpr uint32_t CHUNK_FWD_O_L0_BUFFER_COUNT = 2U;
constexpr uint32_t CHUNK_FWD_O_L0_A_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L0_B_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L0_C_BYTES = 128U * 1024U;

__aicore__ inline uint32_t ChunkFwdOL1QOffset(uint32_t streamSlot)
{
    return CHUNK_FWD_O_L1_Q_BASE + streamSlot * CHUNK_FWD_O_L1_Q_SLOT_BYTES;
}

__aicore__ inline uint32_t ChunkFwdOL1KOffset(uint32_t streamSlot)
{
    return CHUNK_FWD_O_L1_K_BASE + streamSlot * CHUNK_FWD_O_L1_K_SLOT_BYTES;
}

__aicore__ inline uint32_t ChunkFwdOL1HOffset(uint32_t streamSlot)
{
    return CHUNK_FWD_O_L1_H_BASE + streamSlot * CHUNK_FWD_O_L1_H_SLOT_BYTES;
}

__aicore__ inline uint32_t ChunkFwdOL1APrimeOffset(uint32_t headOffset)
{
    return CHUNK_FWD_O_L1_APRIME_BASE + headOffset * CHUNK_FWD_O_L1_APRIME_SLOT_BYTES;
}

__aicore__ inline uint32_t ChunkFwdOL1VOffset(uint32_t headOffset)
{
    return CHUNK_FWD_O_L1_V_BASE + headOffset * CHUNK_FWD_O_L1_V_SLOT_BYTES;
}

__aicore__ inline uint32_t ChunkFwdOL0AOffset(uint32_t slot)
{
    return slot * CHUNK_FWD_O_L0_A_BYTES;
}

__aicore__ inline uint32_t ChunkFwdOL0BOffset(uint32_t slot)
{
    return slot * CHUNK_FWD_O_L0_B_BYTES;
}

__aicore__ inline uint32_t ChunkFwdOL0COffset(uint32_t slot)
{
    return slot * CHUNK_FWD_O_L0_C_BYTES;
}

// Each ready is scoped to one head slot; ROUND_DONE protects cross-round UB reuse.
constexpr uint16_t CHUNK_FWD_O_ARAW_READY = 0U;
constexpr uint16_t CHUNK_FWD_O_OSRAW_READY = 3U;
constexpr uint16_t CHUNK_FWD_O_APRIME_READY = 6U;
constexpr uint16_t CHUNK_FWD_O_OL_READY = 9U;
constexpr uint16_t CHUNK_FWD_O_ROUND_DONE = 12U;

struct ChunkFwdOChunkLoc {
    uint32_t batchIdx;
    uint32_t localChunkIdx;
    uint32_t globalChunkIdx;
    uint32_t tokenStart;
    uint32_t tokenEnd;
    uint32_t chunkLen;
};

__aicore__ inline void ChunkFwdOResolveChunkLoc(GM_ADDR cuSeqlens, GM_ADDR chunkOffsets,
                                                const ChunkFwdOTilingData &tiling, uint32_t loopIdx,
                                                ChunkFwdOChunkLoc &loc)
{
    const uint32_t chunkSize = static_cast<uint32_t>(tiling.chunkSize);
    if (tiling.isVariedLen == 0) {
        const uint32_t chunksPerBatch = static_cast<uint32_t>(tiling.numChunksPerBatch);
        loc.batchIdx = loopIdx / chunksPerBatch;
        loc.localChunkIdx = loopIdx % chunksPerBatch;
        loc.globalChunkIdx = loopIdx;
        loc.tokenStart = loc.localChunkIdx * chunkSize;
        const uint32_t seqlen = static_cast<uint32_t>(tiling.seqlen);
        loc.tokenEnd = loc.tokenStart + chunkSize > seqlen ? seqlen : loc.tokenStart + chunkSize;
    } else {
        AscendC::GlobalTensor<int64_t> cuSeqlensTensor;
        AscendC::GlobalTensor<int64_t> chunkOffsetsTensor;
        cuSeqlensTensor.SetGlobalBuffer((__gm__ int64_t *)cuSeqlens);
        chunkOffsetsTensor.SetGlobalBuffer((__gm__ int64_t *)chunkOffsets);
        const uint32_t seqIdx = static_cast<uint32_t>(chunkOffsetsTensor.GetValue(static_cast<int64_t>(loopIdx) * 2));
        loc.localChunkIdx = static_cast<uint32_t>(chunkOffsetsTensor.GetValue(static_cast<int64_t>(loopIdx) * 2 + 1));
        const uint32_t curSeqBegin = static_cast<uint32_t>(cuSeqlensTensor.GetValue(seqIdx));
        const uint32_t curSeqEnd = static_cast<uint32_t>(cuSeqlensTensor.GetValue(seqIdx + 1));
        loc.batchIdx = 0;
        loc.globalChunkIdx = loopIdx;
        loc.tokenStart = curSeqBegin + loc.localChunkIdx * chunkSize;
        loc.tokenEnd = loc.tokenStart + chunkSize > curSeqEnd ? curSeqEnd : loc.tokenStart + chunkSize;
    }
    loc.chunkLen = loc.tokenEnd - loc.tokenStart;
}

__aicore__ inline int64_t ChunkFwdOQKOffset(const ChunkFwdOTilingData &tiling, const ChunkFwdOChunkLoc &loc,
                                            int64_t hk)
{
    const int64_t row = static_cast<int64_t>(loc.batchIdx) * tiling.kNumHead + hk;
    return (row * tiling.seqlen + static_cast<int64_t>(loc.tokenStart)) * tiling.kHeadDim;
}

__aicore__ inline int64_t ChunkFwdOVOOffset(const ChunkFwdOTilingData &tiling, const ChunkFwdOChunkLoc &loc,
                                            int64_t hv)
{
    const int64_t row = static_cast<int64_t>(loc.batchIdx) * tiling.vNumHead + hv;
    return (row * tiling.seqlen + static_cast<int64_t>(loc.tokenStart)) * tiling.vHeadDim;
}

__aicore__ inline int64_t ChunkFwdOOOffset(const ChunkFwdOTilingData &tiling, const ChunkFwdOChunkLoc &loc,
                                           int64_t hv)
{
    // A6 writes O in sequence-major BSND/TND layout; V remains head-major.
    const int64_t token = static_cast<int64_t>(loc.batchIdx) * tiling.seqlen +
                          static_cast<int64_t>(loc.tokenStart);
    return (token * tiling.vNumHead + hv) * tiling.vHeadDim;
}

__aicore__ inline int64_t ChunkFwdOGOffset(const ChunkFwdOTilingData &tiling, const ChunkFwdOChunkLoc &loc,
                                           int64_t hv)
{
    const int64_t row = static_cast<int64_t>(loc.batchIdx) * tiling.vNumHead + hv;
    return row * tiling.seqlen + static_cast<int64_t>(loc.tokenStart);
}

__aicore__ inline int64_t ChunkFwdOHOffset(const ChunkFwdOTilingData &tiling, const ChunkFwdOChunkLoc &loc,
                                           int64_t hv)
{
    if (tiling.isVariedLen != 0) {
        return (static_cast<int64_t>(loc.globalChunkIdx) * tiling.vNumHead + hv) * tiling.kHeadDim * tiling.vHeadDim;
    }
    return ((static_cast<int64_t>(loc.batchIdx) * tiling.numChunksPerBatch +
             static_cast<int64_t>(loc.localChunkIdx)) * tiling.vNumHead + hv) *
           tiling.kHeadDim * tiling.vHeadDim;
}

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH92_COMMON_H
