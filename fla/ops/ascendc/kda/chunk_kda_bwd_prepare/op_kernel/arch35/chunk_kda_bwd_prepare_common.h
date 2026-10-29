/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_PREPARE_ARCH35_COMMON_H
#define CHUNK_KDA_BWD_PREPARE_ARCH35_COMMON_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "catlass/arch/cross_core_sync.hpp"
#include "../chunk_kda_bwd_prepare_utils.h"
#include "kernel_operator.h"

namespace KDA {

// The A5 direct L0C-to-UB path requires a tightly coupled 1C2V block.  Follow
// the mature DHU convention: each AIV publishes local FREE ids 0/1, while the
// AIC observes AIV1 through the hardware sub-block flag stride.
constexpr uint8_t KDA_PREPARE_CROSS_CORE_MODE = 0x4;
constexpr uint64_t KDA_PREPARE_FREE_FLAG_BASE = 0;
constexpr uint64_t KDA_PREPARE_READY_FLAG_BASE = 6;
constexpr uint64_t KDA_PREPARE_Q_FREE_FLAG = 2;
constexpr uint64_t KDA_PREPARE_D_FREE_FLAG = 3;
constexpr uint64_t KDA_PREPARE_Q_READY_FLAG = 8;
constexpr uint64_t KDA_PREPARE_D_READY_FLAG = 9;
constexpr uint64_t KDA_PREPARE_SUBBLOCK_FLAG_STRIDE = 16;
constexpr uint32_t KDA_PREPARE_RAW_SLOT_COUNT = 2;
constexpr uint32_t KDA_PREPARE_CHUNK = 64;
constexpr uint32_t KDA_PREPARE_DIM = 128;
constexpr uint32_t KDA_PREPARE_RAW_FP32_BYTES =
    KDA_PREPARE_CHUNK * KDA_PREPARE_CHUNK * sizeof(float);
constexpr uint32_t KDA_PREPARE_FP32_BYTES =
    KDA_PREPARE_CHUNK * KDA_PREPARE_CHUNK * sizeof(float);
constexpr uint32_t KDA_PREPARE_A_UB_BYTES =
    2 * KDA_PREPARE_RAW_FP32_BYTES + 3 * KDA_PREPARE_FP32_BYTES;
constexpr uint32_t KDA_PREPARE_Q_FP32_BYTES =
    KDA_PREPARE_CHUNK * KDA_PREPARE_DIM * sizeof(float);
constexpr uint32_t KDA_PREPARE_D_BF16_BYTES =
    KDA_PREPARE_CHUNK * KDA_PREPARE_DIM * sizeof(bfloat16_t);
constexpr uint32_t KDA_PREPARE_Q_UB_OFFSET = KDA_PREPARE_A_UB_BYTES;
constexpr uint32_t KDA_PREPARE_D_UB_OFFSET =
    KDA_PREPARE_Q_UB_OFFSET + KDA_PREPARE_Q_FP32_BYTES;
constexpr uint32_t KDA_PREPARE_TOTAL_UB_BYTES =
    KDA_PREPARE_D_UB_OFFSET + KDA_PREPARE_D_BF16_BYTES;
constexpr uint32_t KDA_PREPARE_A5_UB_BUDGET_BYTES = 192 * 1024;
static_assert(KDA_PREPARE_TOTAL_UB_BYTES <= KDA_PREPARE_A5_UB_BUDGET_BYTES,
              "KernelA AIV buffers exceed the A5 192 KiB UB budget.");

} // namespace KDA

#endif
