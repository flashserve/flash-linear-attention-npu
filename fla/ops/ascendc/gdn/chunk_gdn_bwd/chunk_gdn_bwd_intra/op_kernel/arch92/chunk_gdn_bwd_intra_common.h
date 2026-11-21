/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_ARCH92_COMMON_H
#define CHUNK_GDN_BWD_INTRA_ARCH92_COMMON_H

#include <cstdint>

namespace GDN::Arch92 {

constexpr uint32_t SLOT_COUNT = 3;
constexpr uint32_t TILE_ROWS = 64;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t C0_ELEMENTS = 16;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t MAIN_BYTES = sizeof(uint16_t);
constexpr uint32_t MATRIX_BYTES = TILE_ROWS * TILE_ROWS * MAIN_BYTES;
constexpr uint32_t HEAD_BYTES = TILE_ROWS * HEAD_DIM * MAIN_BYTES;

constexpr uint32_t SCORE_SLOT_BYTES = TILE_ROWS * TILE_ROWS * sizeof(float);
constexpr uint32_t RECORD_SLOT_BYTES = 3 * MATRIX_BYTES;
constexpr uint32_t RAW_BETA_OFFSET = TILE_ROWS * sizeof(float);
constexpr uint32_t RAW_SLOT_BYTES = 2 * RAW_BETA_OFFSET;
constexpr uint32_t VDO_NZ_SLOT_BYTES = 2 * HEAD_BYTES;

constexpr uint32_t SCORE_UB_BASE = 0;
constexpr uint32_t A_UB_BASE = SCORE_UB_BASE + SLOT_COUNT * SCORE_SLOT_BYTES;
constexpr uint32_t RECORD_UB_BASE = A_UB_BASE + SLOT_COUNT * MATRIX_BYTES;
constexpr uint32_t RAW_UB_BASE = RECORD_UB_BASE + SLOT_COUNT * RECORD_SLOT_BYTES;
constexpr uint32_t V_INPUT_UB_BASE = RAW_UB_BASE + SLOT_COUNT * RAW_SLOT_BYTES;
constexpr uint32_t DO_INPUT_UB_BASE = V_INPUT_UB_BASE + SLOT_COUNT * HEAD_BYTES;
constexpr uint32_t VDO_NZ_UB_BASE = DO_INPUT_UB_BASE + SLOT_COUNT * HEAD_BYTES;
constexpr uint32_t UB_BYTES = VDO_NZ_UB_BASE + SLOT_COUNT * VDO_NZ_SLOT_BYTES;

constexpr uint32_t L1_K_BASE = 0;
constexpr uint32_t L1_Q_BASE = L1_K_BASE + SLOT_COUNT * HEAD_BYTES;
constexpr uint32_t L1_RECORD_BASE = L1_Q_BASE + SLOT_COUNT * HEAD_BYTES;
constexpr uint32_t L1_V_BASE = L1_RECORD_BASE + SLOT_COUNT * RECORD_SLOT_BYTES;
constexpr uint32_t L1_DO_BASE = L1_V_BASE + SLOT_COUNT * HEAD_BYTES;
constexpr uint32_t L1_BYTES = L1_DO_BASE + SLOT_COUNT * HEAD_BYTES;
constexpr uint32_t RECORD_D_OFFSET = 0;
constexpr uint32_t RECORD_A_BG_OFFSET = MATRIX_BYTES;
constexpr uint32_t RECORD_A_BETA_OFFSET = 2 * MATRIX_BYTES;

constexpr uint32_t L0_AB_SLOT_BYTES = HEAD_BYTES;
constexpr uint32_t L0_C_SLOT_BYTES = TILE_ROWS * HEAD_DIM * sizeof(float);
constexpr uint32_t L0_AB_BYTES = SLOT_COUNT * L0_AB_SLOT_BYTES;
constexpr uint32_t L0_C_BYTES = SLOT_COUNT * L0_C_SLOT_BYTES;

// One mode-4 flag per slot, reused for the return signal after ready is consumed.
constexpr uint16_t SCORE_FLAG_BASE = 0;
constexpr uint16_t RECORD_FLAG_BASE = 8;

static_assert(UB_BYTES == 0x54600 && L1_BYTES == 0x42000);
static_assert(L0_AB_BYTES == 48 * 1024 && L0_C_BYTES == 96 * 1024);

} // namespace GDN::Arch92

#endif // CHUNK_GDN_BWD_INTRA_ARCH92_COMMON_H
