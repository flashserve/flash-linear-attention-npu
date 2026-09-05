/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_TILING_PROCESSOR_H
#define CHUNK_GDN_BWD_INTRA_TILING_PROCESSOR_H

#include <cstddef>
#include <cstdint>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

constexpr size_t INTRA_Q_IDX = 0;
constexpr size_t INTRA_K_IDX = 1;
constexpr size_t INTRA_V_IDX = 2;
constexpr size_t INTRA_G_IDX = 3;
constexpr size_t INTRA_BETA_IDX = 4;
constexpr size_t INTRA_A_IDX = 5;
constexpr size_t INTRA_DO_IDX = 6;
constexpr size_t INTRA_CU_SEQLENS_IDX = 7;
constexpr size_t INTRA_CHUNK_INDICES_IDX = 8;
constexpr size_t INTRA_SCALE_ATTR_IDX = 0;
constexpr size_t INTRA_CHUNK_SIZE_ATTR_IDX = 1;
constexpr size_t INTRA_USE_EXP2_ATTR_IDX = 2;
constexpr size_t INTRA_STAGE_ATTR_IDX = 3;

inline uint64_t Align512(uint64_t value)
{
    return (value + 511U) / 512U * 512U;
}

} // namespace optiling

#endif // CHUNK_GDN_BWD_INTRA_TILING_PROCESSOR_H
