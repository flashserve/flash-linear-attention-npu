/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_full_tiling.h
 * \brief
 */
#ifndef PREPARE_WY_REPR_BWD_FULL_TILING_H
#define PREPARE_WY_REPR_BWD_FULL_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
static constexpr size_t INPUT_K_IDX = 0;
static constexpr size_t INPUT_V_IDX = 1;
static constexpr size_t INPUT_BETA_IDX = 2;
static constexpr size_t INPUT_A_IDX = 3;
static constexpr size_t INPUT_DA_IDX = 4;
static constexpr size_t INPUT_DW_IDX = 5;
static constexpr size_t INPUT_DU_IDX = 6;
static constexpr size_t INPUT_G_IDX = 7;
static constexpr size_t INPUT_SEQLENS_IDX = 8;
static constexpr size_t INPUT_CHUNK_INDICES_IDX = 9;

static constexpr size_t ATTR_CHUNK_SIZE_IDX = 0;

static constexpr size_t DIM_NUM_3 = 3;
static constexpr size_t DIM_NUM_4 = 4;

static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
static constexpr int64_t CHUNK_INDICES_DIM_1_SIZE = 2;

static constexpr int64_t CHUNK_SIZE_64 = 64;
static constexpr int64_t CHUNK_SIZE_128 = 128;

/** Value head dimension V supported by cube/vector tiling. */
static constexpr int64_t V_DIM_128 = 128;
static constexpr int64_t V_DIM_256 = 256;

static constexpr int64_t VAR_LEN_B_DIM_1 = 1;

static constexpr const char *const INPUT_K_NAME = "k";
static constexpr const char *const INPUT_V_NAME = "v";
static constexpr const char *const INPUT_BETA_NAME = "beta";
static constexpr const char *const INPUT_A_NAME = "A";
static constexpr const char *const INPUT_DA_NAME = "dA";
static constexpr const char *const INPUT_DW_NAME = "dw";
static constexpr const char *const INPUT_DU_NAME = "du";
static constexpr const char *const INPUT_G_NAME = "g";
static constexpr const char *const INPUT_CHUNK_INDICES_NAME = "chunk_indices";
static constexpr const char *const INPUT_SEQLENS_NAME = "cu_seqlens";
static constexpr uint64_t SIZE_HALF = 2;
static constexpr uint64_t SIZE_FP32 = 4;
constexpr uint64_t ONE_BLOCK_32 = 32;
constexpr uint64_t MAX_CUBE_VEC_SYNC_NUM = 5;

BEGIN_TILING_DATA_DEF(PrepareWyReprBwdFullTilingData)
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, HV);
TILING_DATA_FIELD_DEF(int64_t, HK);
TILING_DATA_FIELD_DEF(int64_t, T);
TILING_DATA_FIELD_DEF(int64_t, K);
TILING_DATA_FIELD_DEF(int64_t, V);
TILING_DATA_FIELD_DEF(int64_t, chunkNum);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(int64_t, kBeteVecRow); // kBeta计算流程时vector单次处理的行数
TILING_DATA_FIELD_DEF(int64_t, dkbVecRow);   // dk计算流程时vector单次处理的行数
TILING_DATA_FIELD_DEF(int64_t, dkbgVecRow);  // dkbg计算流程时vector单次处理的行数
TILING_DATA_FIELD_DEF(int64_t, dvbVecRow);   // dvb计算流程时vector单次处理的行数
TILING_DATA_FIELD_DEF(int64_t, kktVecRow);   // kkt计算流程时vector单次处理的行数
TILING_DATA_FIELD_DEF(int64_t, isVariable);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PrepareWyReprBwdFull, PrepareWyReprBwdFullTilingData)

} // namespace optiling

#endif // PREPARE_WY_REPR_BWD_FULL_TILING_H