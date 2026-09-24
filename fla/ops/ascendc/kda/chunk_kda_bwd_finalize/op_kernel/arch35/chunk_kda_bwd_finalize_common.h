/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH35_COMMON_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH35_COMMON_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "catlass/arch/cross_core_sync.hpp"
#include "../chunk_kda_bwd_finalize_struct.h"
#include "kernel_operator.h"

namespace KDA {

using FinalizeLocalType = bfloat16_t;

constexpr uint32_t KDA_FINALIZE_CHUNK = 64;
constexpr uint32_t KDA_FINALIZE_DIM = 128;
constexpr uint32_t KDA_FINALIZE_HEADS_PER_WINDOW = 2;
constexpr uint32_t KDA_FINALIZE_AIV_COUNT = 2;
constexpr uint32_t KDA_FINALIZE_AIV_SLOTS = 2;
constexpr uint32_t KDA_FINALIZE_WORKSPACE_SLOTS = 2 * KDA_FINALIZE_HEADS_PER_WINDOW;
constexpr uint32_t KDA_FINALIZE_SLOT_BYTES = 160 * 1024;

constexpr uint32_t KDA_FINALIZE_INTRA_ROWS = 32;

constexpr uint32_t KDA_FINALIZE_MATRIX_ELEMS = KDA_FINALIZE_CHUNK * KDA_FINALIZE_CHUNK;
constexpr uint32_t KDA_FINALIZE_VECTOR_ELEMS = KDA_FINALIZE_CHUNK * KDA_FINALIZE_DIM;
constexpr uint32_t KDA_FINALIZE_STATE_ELEMS = KDA_FINALIZE_DIM * KDA_FINALIZE_DIM;
constexpr uint32_t KDA_FINALIZE_MATRIX_BF16_BYTES = KDA_FINALIZE_MATRIX_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t KDA_FINALIZE_MATRIX_FP32_BYTES = KDA_FINALIZE_MATRIX_ELEMS * sizeof(float);
constexpr uint32_t KDA_FINALIZE_VECTOR_BF16_BYTES = KDA_FINALIZE_VECTOR_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t KDA_FINALIZE_VECTOR_FP32_BYTES = KDA_FINALIZE_VECTOR_ELEMS * sizeof(float);
constexpr uint32_t KDA_FINALIZE_STATE_BF16_BYTES = KDA_FINALIZE_STATE_ELEMS * sizeof(bfloat16_t);

// Keep the GM slot layout stable.  Only the three raw cube frames
// (DK_STATE_RAW/DVB/DKGB_RAW) are still read, as band slices by
// RunStateBaseSlice.  EXP2_GK/KE/GK_LAST/RH/GATE_STATE/DB_V and the
// *_BASE aliases are idle: the corresponding operands stay on chip.
constexpr uint32_t KDA_FINALIZE_WS_DK_STATE_RAW = 0;
constexpr uint32_t KDA_FINALIZE_WS_DVB = 32 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_DKGB_RAW = 64 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_EXP2_GK = 96 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_KE = 128 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_GK_LAST = 144 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_RH = 144 * 1024 + 512;
constexpr uint32_t KDA_FINALIZE_WS_GATE_STATE = 144 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_DB_V = 145 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_DK_BASE = KDA_FINALIZE_WS_DK_STATE_RAW;
constexpr uint32_t KDA_FINALIZE_WS_DQ_BASE = KDA_FINALIZE_WS_DVB;
constexpr uint32_t KDA_FINALIZE_WS_DG_BASE = KDA_FINALIZE_WS_DKGB_RAW;
constexpr uint32_t KDA_FINALIZE_WS_DB_BASE = KDA_FINALIZE_WS_DB_V;

// Per-AIV BuildZ fixed UB layout.
constexpr uint32_t KDA_FINALIZE_UB_ZV = 0;
constexpr uint32_t KDA_FINALIZE_UB_ZW = 32 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_ZB = 64 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_WORK = 81 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_BYTES = 248 * 1024;

// The Tza residual handoff grants the Stage4 Cube destination dAkk_raw
// at [0,16) KiB.
constexpr uint32_t KDA_FINALIZE_UB_DAKK_RAW = 0;

// Stage5: raw [0,16), paired dAqk/dAkk [16,48), then three paired vectors.
// Input/scratch begins at 144 KiB, disjoint from the live egress regions.
constexpr uint32_t KDA_FINALIZE_UB_K_NEG = 48 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_Q_POS = 80 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_BK_POS = 112 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_STAGE5_WORK = 144 * 1024;

// Stage6 writes dq_local_raw into the former zV FP32 ping/pong range. Stage7
// uses the remaining UB as one phase-wide working set.
constexpr uint32_t KDA_FINALIZE_UB_DQ_LOCAL_RAW = 0;
// One head per AIV per window: these inputs survive phase transitions.
// Q: Stage2 -> Stage9.  K and the raw log-gate frame: Stage0 -> Stage9
// (exp factors recomputed inline).  Beta: Stage2 -> Stage9.
// Stage11 puts rawG in the dead kNeg plane and its gate scalars in the dead
// beta strip, then prefetches the next task's Stage0 inputs into the dead
// [80,144)/[160,208) KiB regions (same fixed addresses RunStage0 uses).
constexpr uint32_t KDA_FINALIZE_UB_Q = 144 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_K = 160 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_EXP2_GK = 176 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_BETA = 208 * 1024;
// Per-band scalar working set: dbBase 211K, dbDelta 212K, qRstd 213K,
// kRstd 214K, dbOut 215K (see RunIntraBandResults).  The band center slot
// is persistent at [209.5,210) KiB (Stage5 -> Stage7/9), disjoint from rH
// [209,209.5) KiB and diagonal [210,210.5) KiB; gateState accumulates
// across bands at [210.5,211) KiB (see RunStateBaseSlice).
constexpr uint32_t KDA_FINALIZE_UB_STAGE7_Q_RSTD = 213 * 1024;
// Resident dg full frame: slice(b0) Stage4VF -> Stage7/9 both bands ->
// Stage11.  Reuses Stage5's dead dAqk input region; dAqkFp32 is band0-only
// and its V reads precede the slice on the same V pipe (program order).
constexpr uint32_t KDA_FINALIZE_UB_DG = 216 * 1024;

// Two 128-KiB owner slots occupy [64,320) KiB of L1. This range is disjoint
// from the only Stage4-live operands, Akk [0,16) and Tza [416,448), so each
// Stage5 head may publish immediately after its own dAkk becomes ready.
// Stage5 publishes all five operands directly from UB to L1. No Stage5
// operand is mirrored through GM; the owner slot stays live through Stage8.
constexpr uint32_t KDA_FINALIZE_LOCAL_BASE = 64 * 1024;
constexpr uint32_t KDA_FINALIZE_LOCAL_BYTES = 128 * 1024;
constexpr uint32_t KDA_FINALIZE_LOCAL_DAQK = 0;
constexpr uint32_t KDA_FINALIZE_LOCAL_DAKK = 16 * 1024;
constexpr uint32_t KDA_FINALIZE_LOCAL_K_NEG = 32 * 1024;
constexpr uint32_t KDA_FINALIZE_LOCAL_Q_POS = 64 * 1024;
constexpr uint32_t KDA_FINALIZE_LOCAL_BK_POS = 96 * 1024;

// KernelA-compatible directed 1C2V handshake.  AIV1's hardware flag bank is
// selected with the fixed +16 sub-block stride.
constexpr uint8_t KDA_FINALIZE_CROSS_MODE = 0x4;
constexpr uint64_t KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE = 16;
constexpr uint64_t KDA_FINALIZE_ZV_FREE_BASE = 0;
constexpr uint64_t KDA_FINALIZE_ZW_FREE_BASE = 2;
constexpr uint64_t KDA_FINALIZE_ZV_READY_BASE = 4;
constexpr uint64_t KDA_FINALIZE_ZW_READY_BASE = 6;
constexpr uint64_t KDA_FINALIZE_KE_READY_BASE = 8;
constexpr uint64_t KDA_FINALIZE_ZB_READY_BASE = 10;
constexpr uint64_t KDA_FINALIZE_ZB_FREE_BASE = 12;
// KE_READY is consumed at Stage1, then carries Tza-pair ready together with
// dAkk UB-free. ZW_READY is consumed at Stage2, then carries Tza raw, and
// finally dAkk raw only after the AIV has consumed the preceding payload.
constexpr uint64_t KDA_FINALIZE_DAKK_FREE_BASE = KDA_FINALIZE_KE_READY_BASE;
constexpr uint64_t KDA_FINALIZE_DAKK_READY_BASE = KDA_FINALIZE_ZW_READY_BASE;
// Keep the final Stage5 AIV->AIC publication on a dedicated pair.  This
// avoids aliasing earlier per-head handoffs without requiring a group-wide
// barrier inside the uneven multi-core work-task loop.
constexpr uint64_t KDA_FINALIZE_LOCAL_READY_BASE = 14;
// AIC->AIV only: LOCAL_READY uses the opposite direction. Keep flag 14
// unused in this direction because the final AIV-only SyncAll owns it.
constexpr uint64_t KDA_FINALIZE_TASK_L1_FREE = 15;
// Stage2 has consumed the zV pair before Stage5/6 starts. Stage5 republishes
// FREE only after its UB->L1 egress, and Stage6 consumes it before publishing
// dq_local_raw READY, so the pair remains balanced within each work task.
constexpr uint64_t KDA_FINALIZE_DQ_LOCAL_FREE_BASE = KDA_FINALIZE_ZV_FREE_BASE;
constexpr uint64_t KDA_FINALIZE_DQ_LOCAL_READY_BASE = KDA_FINALIZE_ZV_READY_BASE;

struct FinalizeChunkInfo {
    int64_t b = 0;
    int64_t seq = 0;
    int64_t localChunk = 0;
    int64_t stateIndex = 0;
    int64_t tokenStart = 0;
    int64_t validRows = 0;
    bool valid = false;
};

__aicore__ inline int64_t FinalizeMin(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

template <bool FULL_TILE>
__aicore__ inline void ResolveFinalizeChunk(
    int64_t task, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const ChunkKdaBwdFinalizeTilingData &tiling, FinalizeChunkInfo &info)
{
    info.valid = false;
    if (task < 0 || task >= tiling.chunkTaskNum) {
        return;
    }
    if constexpr (!FULL_TILE) {
        if (tiling.isVariable != 0) {
            if (cuSeqlens == nullptr || chunkIndices == nullptr) {
                return;
            }
            AscendC::GlobalTensor<int64_t> cu;
            AscendC::GlobalTensor<int64_t> indices;
            cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
            indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
            info.seq = indices.GetValue(2 * task);
            info.localChunk = indices.GetValue(2 * task + 1);
            if (info.seq < 0 || info.seq >= tiling.seqNum || info.localChunk < 0) {
                return;
            }
            const int64_t seqBegin = cu.GetValue(info.seq);
            const int64_t seqEnd = cu.GetValue(info.seq + 1);
            info.b = 0;
            info.stateIndex = task;
            info.tokenStart = seqBegin + info.localChunk * tiling.chunkSize;
            info.validRows = FinalizeMin(tiling.chunkSize, seqEnd - info.tokenStart);
            info.valid = seqBegin >= 0 && seqEnd >= seqBegin && seqEnd <= tiling.T && info.validRows > 0;
            return;
        }
    }
    info.b = task / tiling.denseChunkNum;
    info.seq = info.b;
    info.localChunk = task - info.b * tiling.denseChunkNum;
    info.stateIndex = info.localChunk;
    info.tokenStart = info.localChunk * tiling.chunkSize;
    if constexpr (FULL_TILE) {
        // Host guarantees dense with T % chunkSize == 0 on this key.
        info.validRows = KDA_FINALIZE_CHUNK;
        info.valid = true;
    } else {
        info.validRows = FinalizeMin(tiling.chunkSize, tiling.T - info.tokenStart);
        info.valid = info.b >= 0 && info.b < tiling.B && info.validRows > 0;
    }
}

template <bool FULL_TILE>
__aicore__ inline int64_t FinalizeTokenOffset(
    const ChunkKdaBwdFinalizeTilingData &tiling, const FinalizeChunkInfo &chunk,
    int64_t head, int64_t width)
{
    if constexpr (!FULL_TILE) {
        if (tiling.isVariable != 0) {
            return (head * tiling.T + chunk.tokenStart) * width;
        }
    }
    return ((chunk.b * tiling.NV + head) * tiling.T + chunk.tokenStart) * width;
}

// Saved h is chunk-major; dhu's internal dh remains head-major.
template <bool FULL_TILE>
__aicore__ inline int64_t FinalizeHOffset(
    const ChunkKdaBwdFinalizeTilingData &tiling, const FinalizeChunkInfo &chunk,
    int64_t head)
{
    if constexpr (!FULL_TILE) {
        if (tiling.isVariable != 0) {
            return (chunk.stateIndex * tiling.NV + head) * tiling.K * tiling.V;
        }
    }
    return ((chunk.b * tiling.denseChunkNum + chunk.stateIndex) * tiling.NV + head) *
        tiling.K * tiling.V;
}

template <bool FULL_TILE>
__aicore__ inline int64_t FinalizeDhOffset(
    const ChunkKdaBwdFinalizeTilingData &tiling, const FinalizeChunkInfo &chunk,
    int64_t head)
{
    if constexpr (!FULL_TILE) {
        if (tiling.isVariable != 0) {
            return (head * tiling.totalChunkNum + chunk.stateIndex) * tiling.K * tiling.V;
        }
    }
    return ((chunk.b * tiling.NV + head) * tiling.denseChunkNum + chunk.stateIndex) *
        tiling.K * tiling.V;
}

__aicore__ inline uint64_t FinalizeWorkspaceSlotBase(
    int64_t coreIdx, uint64_t groupGeneration, uint32_t owner)
{
    const uint64_t window = groupGeneration & 1U;
    const uint64_t slot = window * KDA_FINALIZE_HEADS_PER_WINDOW + owner;
    return (static_cast<uint64_t>(coreIdx) * KDA_FINALIZE_WORKSPACE_SLOTS + slot) *
        KDA_FINALIZE_SLOT_BYTES;
}

static_assert(KDA_FINALIZE_WS_DB_V + 512 <= KDA_FINALIZE_SLOT_BYTES,
              "Stage0--3 workspace slot exceeds 160 KiB.");
static_assert(KDA_FINALIZE_UB_WORK < KDA_FINALIZE_UB_BYTES,
              "BuildZ fixed UB handoff exceeds A5 UB.");
static_assert(KDA_FINALIZE_UB_STAGE5_WORK < KDA_FINALIZE_UB_BYTES,
              "Stage5 fixed UB handoff exceeds A5 UB.");
static_assert(KDA_FINALIZE_HEADS_PER_WINDOW == KDA_FINALIZE_AIV_COUNT,
              "Retained inputs require one head per AIV per window.");
// StateBaseSlice's X transient [16K+(1-slot)*16K,+16K) sits in the other
// slot's result half, which this task's cube Stage6/8 never writes; it
// overlaps the pre-band zV/zW/dAqkBf16/dAkkNd/TzaResidual-low transients,
// all drained by RunStage5's closing MTE3_V barrier and program order.
static_assert(16 * 1024 + 2 * KDA_FINALIZE_MATRIX_FP32_BYTES <= KDA_FINALIZE_UB_K_NEG,
              "StateBaseSlice X transient exceeds the pre-band scratch range.");
static_assert(KDA_FINALIZE_UB_Q + KDA_FINALIZE_VECTOR_BF16_BYTES <= KDA_FINALIZE_UB_K &&
                  KDA_FINALIZE_UB_K + KDA_FINALIZE_VECTOR_BF16_BYTES <= KDA_FINALIZE_UB_EXP2_GK &&
                  KDA_FINALIZE_UB_EXP2_GK + KDA_FINALIZE_VECTOR_FP32_BYTES <= KDA_FINALIZE_UB_BETA,
              "Retained head inputs overlap.");
static_assert(KDA_FINALIZE_UB_DG + KDA_FINALIZE_VECTOR_FP32_BYTES <= KDA_FINALIZE_UB_BYTES,
              "Retained dg exceeds A5 UB.");
// qRstd 213K / kRstd 214K / dbOut 215K each hold up to 64 scalars.
static_assert(KDA_FINALIZE_UB_STAGE7_Q_RSTD + 3 * 1024 <= KDA_FINALIZE_UB_DG,
              "Band scalar working set overlaps retained dg.");
static_assert(KDA_FINALIZE_HEADS_PER_WINDOW * KDA_FINALIZE_LOCAL_BYTES <= 256 * 1024,
              "Stage5 paired LocalOperand window exceeds 256 KiB.");
static_assert(KDA_FINALIZE_LOCAL_BASE +
                  KDA_FINALIZE_HEADS_PER_WINDOW * KDA_FINALIZE_LOCAL_BYTES <=
              512 * 1024,
              "Stage5 LocalOperand window exceeds A5 L1.");

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH35_COMMON_H
