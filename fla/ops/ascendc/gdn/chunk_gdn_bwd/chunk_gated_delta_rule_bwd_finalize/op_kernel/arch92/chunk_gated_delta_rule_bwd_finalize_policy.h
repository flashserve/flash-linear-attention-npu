#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ARCH92_POLICY_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ARCH92_POLICY_H

#include <cstdint>

namespace GDN::FinalizeArch92 {
constexpr uint32_t HEAD_COUNT = 3;
constexpr uint32_t SLOT_COUNT = 3;
constexpr uint32_t WINDOW_COUNT = 3;
constexpr uint32_t WORKSPACE_BUFFER_COUNT = HEAD_COUNT * WINDOW_COUNT;
constexpr uint32_t WORKSPACE_REGION_COUNT = 7;
constexpr uint32_t CHUNK_SIZE_64 = 64;
constexpr uint32_t K_SIZE_128 = 128;
constexpr uint32_t V_SIZE_128 = 128;
constexpr uint32_t UB_ALIGN_BYTES = 32;
constexpr uint32_t VECTOR_ELEMS = CHUNK_SIZE_64 * K_SIZE_128;
constexpr uint32_t MATRIX_ELEMS = CHUNK_SIZE_64 * CHUNK_SIZE_64;
constexpr uint32_t STATE_ELEMS = K_SIZE_128 * V_SIZE_128;
constexpr uint32_t VECTOR_BYTES = VECTOR_ELEMS * sizeof(uint16_t);
constexpr uint32_t MATRIX_BYTES = MATRIX_ELEMS * sizeof(uint16_t);
constexpr uint32_t STATE_BYTES = STATE_ELEMS * sizeof(uint16_t);
constexpr uint32_t SMALL_SLOT_BYTES = CHUNK_SIZE_64 * sizeof(float);
constexpr uint32_t SMALL_GROUP_BYTES = HEAD_COUNT * SMALL_SLOT_BYTES;
constexpr uint32_t L0_SLOT_BYTES = 16 * 1024;
constexpr uint32_t UB_TOTAL_BYTES = 0x5D000;
constexpr uint32_t L1_TOTAL_BYTES = 0x72000;
static_assert(2 * SLOT_COUNT <= 8, "Each local HardEvent direction has a bounded event pool.");

// Two ordered mode-4 chains; S14 free bounds the number of in-flight notifications.
constexpr uint8_t CROSS_MODE = 0x4;
constexpr uint16_t VEC_TO_CUBE_READY_FLAG = 1;
constexpr uint16_t CUBE_TO_VEC_READY_FLAG = 3;
constexpr uint32_t CUBE_TO_VEC_NOTIFICATIONS = 5 * HEAD_COUNT;
constexpr uint32_t VEC_TO_CUBE_NOTIFICATIONS = 4 * HEAD_COUNT + 1;
static_assert(CUBE_TO_VEC_NOTIFICATIONS <= 15);
static_assert(VEC_TO_CUBE_NOTIFICATIONS <= 15);

namespace UB {
constexpr uint32_t K = 0x00000;
constexpr uint32_t V = 0x0C000;
constexpr uint32_t GATE_DA = 0x18000;
constexpr uint32_t VB = 0x1E000;
constexpr uint32_t DAU = 0x2A000;
constexpr uint32_t KBG = 0x30000;
constexpr uint32_t KB = 0x3C000;
constexpr uint32_t DA2 = 0x48000;
constexpr uint32_t DKBG0 = 0x4E000;
constexpr uint32_t H_STATE = 0x1E000;
constexpr uint32_t DH_STATE = 0x36000;
constexpr uint32_t DS = V;
constexpr uint32_t DO_G = DAU;
constexpr uint32_t V_DECAY = DH_STATE;
constexpr uint32_t DKB = 0x42000;
constexpr uint32_t DK_BASE = V;
constexpr uint32_t DQ_HV = VB;
constexpr uint32_t DK_INTRA = V_DECAY;
constexpr uint32_t DQ_ACC = K;
constexpr uint32_t DK_ACC = DO_G;
constexpr uint32_t DQ_PARTIAL = GATE_DA;
constexpr uint32_t DK_PARTIAL = DKB;
constexpr uint32_t NORM_Q = DA2;
constexpr uint32_t NORM_K = DKBG0;
constexpr uint32_t BETA_RAW = 0x5B800;
constexpr uint32_t DB_V = BETA_RAW + SMALL_GROUP_BYTES;
constexpr uint32_t G = DB_V + SMALL_GROUP_BYTES;
constexpr uint32_t BETA = G + SMALL_GROUP_BYTES;
constexpr uint32_t G_EXP = BETA + SMALL_GROUP_BYTES;
constexpr uint32_t G_LAST = G_EXP + SMALL_GROUP_BYTES;
constexpr uint32_t DECAY = G_LAST + SMALL_GROUP_BYTES;
constexpr uint32_t BG = DECAY + SMALL_GROUP_BYTES;
}
namespace L1 {
constexpr uint32_t A = 0x00000;
constexpr uint32_t DH = 0x0C000;
constexpr uint32_t DA = 0x24000;
constexpr uint32_t H = 0x2A000;
constexpr uint32_t DU = 0x42000;
constexpr uint32_t KBG = 0x4E000;
constexpr uint32_t VB = 0x5A000;
constexpr uint32_t DW0 = 0x66000;
}
static_assert(UB::BG + SMALL_GROUP_BYTES == UB_TOTAL_BYTES);
static_assert(UB::H_STATE + HEAD_COUNT * STATE_BYTES == UB::DH_STATE);
static_assert(UB::DH_STATE + HEAD_COUNT * STATE_BYTES == UB::DKBG0);
static_assert(UB::DKBG0 + HEAD_COUNT * VECTOR_BYTES == 0x5A000);
static_assert(UB::DK_PARTIAL + VECTOR_ELEMS * sizeof(float) <= UB::DKBG0);
static_assert(UB::DQ_ACC + HEAD_COUNT * VECTOR_BYTES <= UB::DK_BASE);
static_assert(UB::DK_ACC + HEAD_COUNT * VECTOR_BYTES <= UB::DK_INTRA);
static_assert(L1::DW0 + HEAD_COUNT * VECTOR_BYTES == L1_TOTAL_BYTES);
} // namespace GDN::FinalizeArch92

#endif
