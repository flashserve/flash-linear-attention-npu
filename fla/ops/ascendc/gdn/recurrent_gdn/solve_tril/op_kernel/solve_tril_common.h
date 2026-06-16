/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#ifndef __SOLVE_TRIL_COMMON_H__
#define __SOLVE_TRIL_COMMON_H__

#include "ascendc/host_api/tiling/template_argument.h"
#include "solve_tril_tiling_data.h"

// ============================================================================
// Constants
// ============================================================================
constexpr uint32_t LEAF_BLOCK_SIZE = 16;
constexpr uint32_t MCH_ITERATIONS = 3;
constexpr uint32_t LEAF_ELEMENTS = LEAF_BLOCK_SIZE * LEAF_BLOCK_SIZE;

constexpr uint32_t MCH_WORK_SLOTS = 5;
constexpr uint32_t MCH_ONLY_WORK_ELEMENTS = MCH_WORK_SLOTS * LEAF_ELEMENTS;
constexpr uint32_t MCH_ONLY_WORK_BYTES = MCH_ONLY_WORK_ELEMENTS * sizeof(float);

constexpr uint32_t MBH1_WORK_SLOTS = 10;
constexpr uint32_t MBH1_WORK_ELEMENTS = MBH1_WORK_SLOTS * LEAF_ELEMENTS;
constexpr uint32_t MBH1_WORK_BYTES = MBH1_WORK_ELEMENTS * sizeof(float);

constexpr uint32_t MBH2_RESULT_DIM = 64;
constexpr uint32_t MBH2_RESULT_SIZE = MBH2_RESULT_DIM * MBH2_RESULT_DIM;
constexpr uint32_t MBH2_WORK_TOTAL = MBH2_RESULT_SIZE + MCH_WORK_SLOTS * LEAF_ELEMENTS;
constexpr uint32_t MBH2_WORK_BYTES = MBH2_WORK_TOTAL * sizeof(float);

constexpr uint32_t MBH3_RESULT_DIM = 128;
constexpr uint32_t MBH3_RESULT_SIZE = MBH3_RESULT_DIM * MBH3_RESULT_DIM;
constexpr uint32_t MBH3_WORK_TOTAL = MBH3_RESULT_SIZE + MCH_WORK_SLOTS * LEAF_ELEMENTS;
constexpr uint32_t MBH3_WORK_BYTES = MBH3_WORK_TOTAL * sizeof(float);

constexpr uint32_t UB_CAPACITY_BYTES = 192 * 1024;

// ============================================================================
// TilingKey: D_TYPE (FLOAT16) x MBH_LEVELS (0,1,2,3)
// Note: BF16 kernel support requires ascend950 (bisheng for ascend910b
//       does not support bf16 type cast in backend).
// ============================================================================
ASCENDC_TPL_ARGS_DECL(SolveTril,
    ASCENDC_TPL_DATATYPE_DECL(D_TYPE, C_DT_FLOAT16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(MBH_LEVELS, 8, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_TYPE, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(MBH_LEVELS, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3)
    ),
);

#endif // __SOLVE_TRIL_COMMON_H__
