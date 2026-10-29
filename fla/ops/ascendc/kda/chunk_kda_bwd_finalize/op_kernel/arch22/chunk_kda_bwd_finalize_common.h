/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH22_COMMON_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH22_COMMON_H

#include <cstdint>

namespace KDA::Arch22 {

constexpr uint32_t CHUNK_SIZE = 64;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t VECTOR_ROWS = 16;
constexpr uint32_t HEADS_PER_WORK = 2;
constexpr uint32_t FRACTAL_ELEMENTS = 16 * 16;

// Byte offsets from design/L1内存.md. Both heads keep their own L1 lane.
constexpr uint32_t L1_HEAD_BYTES = 192 * 1024;
constexpr uint32_t L1_BYTES = 512 * 1024;
constexpr uint32_t L1_AKK = 0;
constexpr uint32_t L1_DW = 8 * 1024;
constexpr uint32_t L1_KE = 24 * 1024;
constexpr uint32_t L1_ZB = 40 * 1024;
constexpr uint32_t L1_TZA = 48 * 1024;
constexpr uint32_t L1_AKK_BANK_STRIDE = 56 * 1024;
constexpr uint32_t L1_TZA_BANK_STRIDE = 128 * 1024;
constexpr uint32_t L1_H = 64 * 1024;
constexpr uint32_t L1_DH = 96 * 1024;
constexpr uint32_t L1_V_NEW = 128 * 1024;
constexpr uint32_t L1_DV_SCAN = 144 * 1024;
constexpr uint32_t L1_V = 160 * 1024;
constexpr uint32_t L0_AB_SLOT_BYTES = 32 * 1024;
constexpr uint32_t L0_C_SLOT_BYTES = 64 * 1024;

constexpr uint32_t UB_BYTES = 192 * 1024;
constexpr uint32_t UB_K = 16 * 1024;
constexpr uint32_t UB_EXP2_GK = 32 * 1024;
constexpr uint32_t UB_BETA = 80 * 1024;
constexpr uint32_t UB_DB_V = 80 * 1024 + 256;
constexpr uint32_t UB_RH = 81 * 1024 + 256;
constexpr uint32_t UB_GK_LAST = 81 * 1024 + 768;
// Production no-norm path uses a 512-byte strict-lower bitmask.
constexpr uint32_t UB_LOWER_BITS = 84 * 1024 + 256;
constexpr uint32_t UB_MASK_ZERO = 84 * 1024 + 896;
constexpr uint32_t UB_MASK_ADDR = 85 * 1024;
constexpr uint32_t UB_WORK = 88 * 1024;
constexpr uint32_t UB_WORK_SLOT_BYTES = 8 * 1024;

// Two reusable workspace generations per head.
constexpr uint32_t WS_BANK_BYTES = 384 * 1024;
constexpr uint32_t WS_HEAD_BYTES = 2 * WS_BANK_BYTES;
constexpr uint32_t WS_CORE_BYTES = HEADS_PER_WORK * WS_HEAD_BYTES;
constexpr uint32_t WS_DK_STATE_RAW = 0;
constexpr uint32_t WS_DV_B = 32 * 1024;
constexpr uint32_t WS_ZV = 64 * 1024;
constexpr uint32_t WS_DKGB_RAW = 80 * 1024;
constexpr uint32_t WS_ZW = 112 * 1024;
constexpr uint32_t WS_KE = 144 * 1024;
constexpr uint32_t WS_ZB = 160 * 1024;
// Stage 5 replaces consumed raw slots; both generations still fit 256 KiB.
constexpr uint32_t WS_DK_BASE = 0;
constexpr uint32_t WS_DQ_BASE = 32 * 1024;
constexpr uint32_t WS_DG_BASE = 168 * 1024;
constexpr uint32_t WS_DB_BASE = 200 * 1024;
constexpr uint32_t WS_RH = 208 * 1024;
constexpr uint32_t WS_DAKK_RAW = 216 * 1024;

// Stage 7 reuses dead raw slots after Stage 5 and Stage 6. Bases remain intact.
constexpr uint32_t WS_DAQK_LOWER = 64 * 1024;
constexpr uint32_t WS_DAKK_LOWER = 80 * 1024;
constexpr uint32_t WS_KNEG = 96 * 1024;
constexpr uint32_t WS_QPOS = 128 * 1024;
constexpr uint32_t WS_BKPOS = 224 * 1024;

// Stage 8 has independent L1 lanes, leaving both generations of akk intact.
constexpr uint32_t L1_STAGE8 = 384 * 1024;
constexpr uint32_t L1_STAGE8_HEAD_BYTES = 64 * 1024;
constexpr uint32_t WS_DQ_LOCAL = 256 * 1024;
constexpr uint32_t WS_LEFT = 288 * 1024;
constexpr uint32_t WS_DAQK_FP32 = 320 * 1024;
constexpr uint32_t WS_KNEG_FP32 = 336 * 1024;
constexpr uint32_t WS_DAKK_FP32 = 368 * 1024;
constexpr uint16_t BAND_READY = 1;
constexpr uint16_t BAND_LOADED = 2;
constexpr uint16_t S8_READY = 3;
constexpr uint16_t S8_SECOND_READY = 14;
// Flag 15 is reserved by the CANN MIX workspace initialization.
// Flag 11 is free on AIV receivers; VECTOR_ACK+1 targets the AIC receiver.
constexpr uint16_t S10_FIRST_READY = 11;

constexpr uint16_t KE_READY = 0;
constexpr uint16_t WORK_OPEN = 8;

// Host-prepared metadata: no sequence search or tail inference in the hot loop.
struct ChunkInfo {
    int64_t sequence;
    int64_t localChunk;
    int64_t tokenStart;
    int64_t validRows;
};

// A2/A3 tiling shared by the host and the single public kernel entry.
struct FinalizeTilingData {
    int64_t batch;
    int64_t heads;
    int64_t tokens;
    int64_t chunks;
    uint32_t usedCoreNum;
    uint32_t isVariable;
    uint32_t outputTokenMajor = 0;
    uint32_t hasQkL2Norm = 0;
    uint32_t betaMode = 0;
    float scale = 1.0f;
    float lowerBound = -5.0f;
    uint64_t partialOffset = 0;
};

struct FinalizeTailArgs {
    uint64_t rawG = 0, aLog = 0, dtBias = 0, kRstd = 0;
    uint64_t dk = 0, dg = 0, db = 0, partial = 0, dALog = 0, dDtBias = 0;
    float lowerBound = -5.0f;
    uint32_t betaMode = 0;
};
constexpr uint32_t PARTIAL_FLOATS = 136;
constexpr uint32_t UB11_BIAS = 82 * 1024 + 256;
constexpr uint32_t UB11_RSTD = 83 * 1024 + 512;
constexpr uint32_t UB11_BETA_FACTOR = 84 * 1024;
constexpr uint32_t UB11_SUFFIX = 86 * 1024;
constexpr uint32_t UB11_PARTIAL = 86 * 1024 + 512;
constexpr uint32_t UB11_EXP_A = 87 * 1024 + 512;

// Optional test snapshots, separate from the reusable production workspace.
// Each (head, chunk) owns a complete record, including padding and tail rows.
constexpr uint32_t CAPTURE_BYTES = 928 * 1024;
constexpr uint32_t CAPTURE_DK_STATE_RAW = 0;
constexpr uint32_t CAPTURE_DV_B = 32 * 1024;
constexpr uint32_t CAPTURE_ZV = 64 * 1024;
constexpr uint32_t CAPTURE_DW = 80 * 1024;
constexpr uint32_t CAPTURE_EXP2_GK = 96 * 1024;
constexpr uint32_t CAPTURE_KE = 128 * 1024;
constexpr uint32_t CAPTURE_RH = 144 * 1024;
constexpr uint32_t CAPTURE_GK_LAST = 144 * 1024 + 512;

constexpr uint32_t CAPTURE_DKGB_RAW = 160 * 1024;
constexpr uint32_t CAPTURE_ZW = 192 * 1024;
constexpr uint32_t CAPTURE_ZB = 208 * 1024;
constexpr uint32_t CAPTURE_DB_V = 216 * 1024;
constexpr uint32_t CAPTURE_TZA = 224 * 1024;
constexpr uint32_t CAPTURE_TZA_CHECK = 232 * 1024;
constexpr uint32_t CAPTURE_DQ_BASE = 256 * 1024;
constexpr uint32_t CAPTURE_DK_BASE = 288 * 1024;
constexpr uint32_t CAPTURE_DG_BASE = 320 * 1024;
constexpr uint32_t CAPTURE_DB_BASE = 352 * 1024;
constexpr uint32_t CAPTURE_DAKK_RAW = 360 * 1024;

constexpr uint32_t CAPTURE_DAQK_LOWER = 384 * 1024;
constexpr uint32_t CAPTURE_DAKK_LOWER = 400 * 1024;
constexpr uint32_t CAPTURE_DAQK_DIAG = 416 * 1024;
constexpr uint32_t CAPTURE_BANDS = 424 * 1024;
constexpr uint32_t CAPTURE_BAND_BYTES = 116 * 1024;
constexpr uint32_t CAPTURE_DQ_LOCAL = 656 * 1024;
constexpr uint32_t CAPTURE_LEFT = 688 * 1024;
constexpr uint32_t CAPTURE_DQ = 720 * 1024;
constexpr uint32_t CAPTURE_DG = 752 * 1024;
constexpr uint32_t CAPTURE_RIGHT = 784 * 1024;
constexpr uint32_t WS_RIGHT = WS_KNEG; // Stage 8 has consumed kNeg before Stage 10 writes.
constexpr uint32_t CAPTURE_DK = 816 * 1024;
constexpr uint32_t CAPTURE_DB = 848 * 1024;
constexpr uint32_t CAPTURE_SCAN = 856 * 1024;
constexpr uint32_t CAPTURE_GATE = 888 * 1024;
constexpr uint32_t CAPTURE_PARTIAL = 920 * 1024;
static_assert(CAPTURE_PARTIAL + PARTIAL_FLOATS * 4 <= CAPTURE_BYTES);
static_assert(CAPTURE_BANDS + 2 * CAPTURE_BAND_BYTES == CAPTURE_DQ_LOCAL);
static_assert(WS_LEFT + 32 * 1024 == WS_DAQK_FP32);
static_assert(WS_DAQK_FP32 + 16 * 1024 == WS_KNEG_FP32);
static_assert(WS_KNEG_FP32 + 32 * 1024 == WS_DAKK_FP32);
static_assert(WS_DAKK_FP32 + 16 * 1024 == WS_BANK_BYTES);

static_assert(WS_RH + HEAD_DIM * 4 <= WS_DAKK_RAW);
static_assert(WS_DAKK_RAW + CHUNK_SIZE * CHUNK_SIZE * 4 <= WS_BANK_BYTES);
static_assert(CAPTURE_DAKK_RAW + CHUNK_SIZE * CHUNK_SIZE * 4 <= CAPTURE_BYTES);
static_assert(sizeof(ChunkInfo) == 32);
static_assert(sizeof(FinalizeTilingData) == 72);
static_assert(L1_V + CHUNK_SIZE * HEAD_DIM * 2 <= L1_HEAD_BYTES);
static_assert(L1_BYTES <= 512 * 1024);
static_assert(2 * L0_AB_SLOT_BYTES == 64 * 1024);
static_assert(2 * L0_C_SLOT_BYTES == 128 * 1024);
static_assert(UB_WORK + 13 * UB_WORK_SLOT_BYTES == UB_BYTES);
static_assert(UB_RH + HEAD_DIM * 4 == UB_GK_LAST);
static_assert(WS_DKGB_RAW + CHUNK_SIZE * HEAD_DIM * 4 == WS_ZW);
static_assert(WS_ZW + CHUNK_SIZE * CHUNK_SIZE * 4 <= WS_KE);
static_assert(CAPTURE_ZW + CHUNK_SIZE * CHUNK_SIZE * 4 <= CAPTURE_BYTES);
static_assert(2 * WS_BANK_BYTES == WS_HEAD_BYTES);
static_assert(WS_ZB + CHUNK_SIZE * CHUNK_SIZE * 2 <= WS_BANK_BYTES);
static_assert(CAPTURE_DB_V + CHUNK_SIZE * 4 <= CAPTURE_BYTES);
static_assert(WS_KE + CHUNK_SIZE * HEAD_DIM * 2 <= WS_HEAD_BYTES);
static_assert(CAPTURE_GK_LAST + HEAD_DIM * 4 <= CAPTURE_BYTES);

} // namespace KDA::Arch22

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH22_COMMON_H
