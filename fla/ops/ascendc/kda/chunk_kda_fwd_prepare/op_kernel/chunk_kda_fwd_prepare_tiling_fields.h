// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#ifndef CHUNK_KDA_FWD_PREPARE_TILING_FIELDS_H
#define CHUNK_KDA_FWD_PREPARE_TILING_FIELDS_H

#include <cstddef>
#include <cstdint>

// The host serializer and device wire type consume the same ordered fields.
// This is an internal launch contract; the public ACLNN signature is unchanged.
#define CHUNK_KDA_PREPARE_TILING_FIELDS(F) \
    F(uint32_t, batch); \
    F(uint32_t, seqNum); \
    F(uint32_t, seqLen); \
    F(uint32_t, qkHeadNum); \
    F(uint32_t, valueHeadNum); \
    F(uint32_t, totalChunks); \
    F(uint32_t, usedCoreNum); \
    F(uint32_t, headsPerPartition); \
    F(float, epsilon); \
    F(float, lowerBound); \
    F(float, scale); \
    F(bool, isVarLen); \
    F(bool, inputSequenceMajor); \
    F(bool, hasDtBias); \
    F(uint32_t, normMode); \
    F(uint32_t, betaMode); \
    F(uint32_t, gateMode); \
    F(uint32_t, outputMode); \
    F(uint32_t, pipelineVariant); \
    F(bool, useExp2)

namespace KdaPrepare {
struct PrepareTilingWire {
#define PREPARE_WIRE_FIELD(T, NAME) T NAME
    CHUNK_KDA_PREPARE_TILING_FIELDS(PREPARE_WIRE_FIELD);
#undef PREPARE_WIRE_FIELD
};
static_assert(sizeof(bool) == 1 && sizeof(float) == 4 && sizeof(uint32_t) == 4);
static_assert(offsetof(PrepareTilingWire, isVarLen) == 44);
static_assert(offsetof(PrepareTilingWire, normMode) == 48);
static_assert(offsetof(PrepareTilingWire, pipelineVariant) == 64);
static_assert(offsetof(PrepareTilingWire, useExp2) == 68);
static_assert(sizeof(PrepareTilingWire) == 72);
} // namespace KdaPrepare
#endif
