/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef __SOLVE_TRIL_COMMON_H__
#define __SOLVE_TRIL_COMMON_H__

#include "ascendc/host_api/tiling/template_argument.h"
#include "solve_tril_tiling_data.h"

// ============================================================================
// Constants
// ============================================================================
constexpr uint32_t LEAF_BLOCK_SIZE = 16;       // MCH leaf block dimension
constexpr uint32_t MCH_ITERATIONS = 3;         // Number of MCH iterations
constexpr uint32_t LEAF_ELEMENTS = LEAF_BLOCK_SIZE * LEAF_BLOCK_SIZE; // 256

// ============================================================================
// UB Buffer Size Constants (compile-time boundary verification)
// ============================================================================
constexpr uint32_t MCH_WORK_SLOTS = 5;  // X, A, Y, temp, mmTmp

// MCH_ONLY (MBH_LEVELS == 0): 5 float32 16x16 blocks
constexpr uint32_t MCH_ONLY_WORK_ELEMENTS = MCH_WORK_SLOTS * LEAF_ELEMENTS;
constexpr uint32_t MCH_ONLY_WORK_BYTES = MCH_ONLY_WORK_ELEMENTS * sizeof(float);

// MBH_1 (MBH_LEVELS == 1): 10 float32 16x16 blocks
constexpr uint32_t MBH1_WORK_SLOTS = 10;
constexpr uint32_t MBH1_WORK_ELEMENTS = MBH1_WORK_SLOTS * LEAF_ELEMENTS;
constexpr uint32_t MBH1_WORK_BYTES = MBH1_WORK_ELEMENTS * sizeof(float);

// MBH_2 (MBH_LEVELS == 2): result(64*64) + MCH work(5*256)
constexpr uint32_t MBH2_RESULT_DIM = 64;
constexpr uint32_t MBH2_RESULT_SIZE = MBH2_RESULT_DIM * MBH2_RESULT_DIM;
constexpr uint32_t MBH2_WORK_TOTAL = MBH2_RESULT_SIZE + MCH_WORK_SLOTS * LEAF_ELEMENTS;
constexpr uint32_t MBH2_WORK_BYTES = MBH2_WORK_TOTAL * sizeof(float);

// MBH_3 (MBH_LEVELS == 3): result(128*128) + MCH work(5*256)
constexpr uint32_t MBH3_RESULT_DIM = 128;
constexpr uint32_t MBH3_RESULT_SIZE = MBH3_RESULT_DIM * MBH3_RESULT_DIM;
constexpr uint32_t MBH3_WORK_TOTAL = MBH3_RESULT_SIZE + MCH_WORK_SLOTS * LEAF_ELEMENTS;
constexpr uint32_t MBH3_WORK_BYTES = MBH3_WORK_TOTAL * sizeof(float);

// UB capacity limit for Ascend910B (192KB)
constexpr uint32_t UB_CAPACITY_BYTES = 192 * 1024;

// ============================================================================
// TilingKey definition
// ============================================================================
ASCENDC_TPL_ARGS_DECL(SolveTril,
    ASCENDC_TPL_DATATYPE_DECL(D_TYPE, C_DT_FLOAT16, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(MBH_LEVELS, 8, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_TYPE, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(MBH_LEVELS, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_TYPE, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(MBH_LEVELS, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3)
    ),
);

#endif // __SOLVE_TRIL_COMMON_H__
