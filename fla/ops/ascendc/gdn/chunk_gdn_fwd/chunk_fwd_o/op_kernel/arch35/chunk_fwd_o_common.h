/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef CHUNK_FWD_O_ARCH35_COMMON_H
#define CHUNK_FWD_O_ARCH35_COMMON_H

#include <cstdint>
#include "kernel_operator.h"
#include "../chunk_fwd_o_struct.h"
#include "../chunk_fwd_o_a5_constants.h"

namespace GDN {

constexpr int64_t CHUNK_FWD_O_PIPELINE_WINDOW = 4;

// UB layout (bytes, single AIV). Design §5.3:
// gate_o[0,0.5) gate_A[0.5,32.5) g[32.5,33) O_s'[33,97) A_raw[97,129) O_l[129,193).
constexpr uint32_t CHUNK_FWD_O_UB_GATE_BASE = 0U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_GATE_O_SLOT_BYTES = 256U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_GATE_A_BASE = 512U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_GATE_A_SLOT_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_G_SCRATCH_BASE = 33280U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_G_SCRATCH_SLOT_BYTES = 256U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_OSPRIME_BASE = 33792U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_OSPRIME_SLOT_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_ARAW_BASE = 99328U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_ARAW_SLOT_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_OL_BASE = 132096U;
constexpr uint32_t CHUNK_FWD_O_UB_LAYOUT_OL_SLOT_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_APRIME_FP32_OFFSET = 208U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_APRIME_FP32_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_APRIME_BF16_OFFSET =
    CHUNK_FWD_O_UB_APRIME_FP32_OFFSET + CHUNK_FWD_O_UB_APRIME_FP32_BYTES;
constexpr uint32_t CHUNK_FWD_O_STREAM_BANK_COUNT = 2U;
constexpr uint32_t CHUNK_FWD_O_L1_RESIDENT_HEAD_COUNT = 4U;
constexpr uint32_t CHUNK_FWD_O_UB_APRIME_BF16_SLOT_BYTES = CHUNK_FWD_O_APRIME_SLOT_BYTES;
constexpr uint32_t CHUNK_FWD_O_UB_APRIME_BF16_SLOT_COUNT = CHUNK_FWD_O_STREAM_BANK_COUNT;
constexpr uint32_t CHUNK_FWD_O_UB_TOTAL_BYTES = 248U * 1024U;
constexpr uint32_t CHUNK_FWD_O_UB_STAGE3_END = CHUNK_FWD_O_UB_TOTAL_BYTES;

static_assert(CHUNK_FWD_O_UB_APRIME_BF16_OFFSET +
                  CHUNK_FWD_O_UB_APRIME_BF16_SLOT_COUNT * CHUNK_FWD_O_UB_APRIME_BF16_SLOT_BYTES <=
              CHUNK_FWD_O_UB_TOTAL_BYTES,
              "Stage3 A-prime ping/pong exceeds UB capacity.");

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

__aicore__ inline int64_t ChunkFwdOGroupRound(uint32_t groupTaskIdx, uint32_t coreNum)
{
    return static_cast<int64_t>(groupTaskIdx / coreNum);
}

__aicore__ inline GM_ADDR ChunkFwdOAPrimeGmOffset(GM_ADDR workspace, const ChunkFwdOTilingData &tiling,
                                                  uint32_t coreIdx, int64_t groupRound, uint32_t aPrimeSlot)
{
    const uint32_t windowStart =
        static_cast<uint32_t>(groupRound & 1) * CHUNK_FWD_O_APRIME_HEADS_PER_WINDOW;
    const uint32_t workspaceSlot = windowStart + aPrimeSlot;
    return workspace + tiling.aPrimeWorkspaceOffset +
           static_cast<int64_t>(coreIdx) * static_cast<int64_t>(CHUNK_FWD_O_APRIME_WORKSPACE_BYTES) +
           static_cast<int64_t>(workspaceSlot) * static_cast<int64_t>(CHUNK_FWD_O_APRIME_SLOT_BYTES);
}

constexpr uint32_t CHUNK_FWD_O_L1_Q_BASE = 0U;
constexpr uint32_t CHUNK_FWD_O_L1_K_BASE = 64U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_H_BASE = 128U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_Q_SLOT_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_K_SLOT_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_H_SLOT_BYTES = 32U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_STAGE2_END =
    CHUNK_FWD_O_L1_H_BASE + CHUNK_FWD_O_L1_RESIDENT_HEAD_COUNT * CHUNK_FWD_O_L1_H_SLOT_BYTES;
constexpr uint32_t CHUNK_FWD_O_L1_APRIME_BASE = 256U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_APRIME_SLOT_BYTES = CHUNK_FWD_O_APRIME_SLOT_BYTES;
constexpr uint32_t CHUNK_FWD_O_L1_V_BASE = 288U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_V_SLOT_BYTES = 16U * 1024U;
constexpr uint32_t CHUNK_FWD_O_L1_STAGE4_END = 352U * 1024U;

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

// Ordered chains require both sides to walk every stage round in the same
// head order, so one fixed flag per direction is sufficient.
constexpr uint64_t CHUNK_FWD_O_VEC_TO_CUBE_READY_FLAG = 1;
constexpr uint64_t CHUNK_FWD_O_CUBE_TO_VEC_READY_FLAG = 3;

__aicore__ inline int64_t ChunkFwdOSyncIdx(int64_t chunkIdx)
{
    return chunkIdx % CHUNK_FWD_O_PIPELINE_WINDOW;
}

__aicore__ inline int64_t ChunkFwdOLaSlot(int64_t chunkIdx)
{
    return chunkIdx % CHUNK_FWD_O_PIPELINE_WINDOW;
}

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
    // A5 writes O in sequence-major BSND/TND layout; V remains head-major.
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
        return (hv * tiling.chunkNum + static_cast<int64_t>(loc.globalChunkIdx)) * tiling.kHeadDim * tiling.vHeadDim;
    }
    return (static_cast<int64_t>(loc.batchIdx) * tiling.vNumHead * tiling.numChunksPerBatch +
            hv * tiling.numChunksPerBatch + static_cast<int64_t>(loc.localChunkIdx)) *
           tiling.kHeadDim * tiling.vHeadDim;
}

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_COMMON_H
