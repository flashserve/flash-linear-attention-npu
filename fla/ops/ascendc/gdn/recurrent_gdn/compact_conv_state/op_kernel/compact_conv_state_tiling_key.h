/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef COMPACT_CONV_STATE_TILING_KEY_H
#define COMPACT_CONV_STATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace GDN {

#define CCS_TPL_BF16 10
#define CCS_TPL_FP32 30
#define CCS_TPL_INT32 20
#define CCS_TPL_INT64 40

ASCENDC_TPL_ARGS_DECL(CompactConvState, ASCENDC_TPL_DTYPE_DECL(D_T_STATE, CCS_TPL_BF16, CCS_TPL_FP32),
                      ASCENDC_TPL_BOOL_DECL(ALIGN_MODE, 0, 1),
                      ASCENDC_TPL_DTYPE_DECL(D_T_PLAN, CCS_TPL_INT32, CCS_TPL_INT64),
                      ASCENDC_TPL_DTYPE_DECL(D_T_STATE_INDEX, CCS_TPL_INT32, CCS_TPL_INT64),
                      ASCENDC_TPL_BOOL_DECL(SHARED_DIM_STRIDE, 0, 1), );

#define CCS_TPL_SEL_ONE(STATE_TYPE, ALIGN_VALUE, PLAN_TYPE, INDEX_TYPE, SHARED_VALUE)                                  \
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_STATE, STATE_TYPE), ASCENDC_TPL_BOOL_SEL(ALIGN_MODE, ALIGN_VALUE),  \
                         ASCENDC_TPL_DTYPE_SEL(D_T_PLAN, PLAN_TYPE),                                                   \
                         ASCENDC_TPL_DTYPE_SEL(D_T_STATE_INDEX, INDEX_TYPE),                                           \
                         ASCENDC_TPL_BOOL_SEL(SHARED_DIM_STRIDE, SHARED_VALUE), )

#define CCS_TPL_SEL_PLAN_INDEX(STATE_TYPE, ALIGN_VALUE, SHARED_VALUE)                                                  \
    CCS_TPL_SEL_ONE(STATE_TYPE, ALIGN_VALUE, CCS_TPL_INT32, CCS_TPL_INT32, SHARED_VALUE),                              \
        CCS_TPL_SEL_ONE(STATE_TYPE, ALIGN_VALUE, CCS_TPL_INT32, CCS_TPL_INT64, SHARED_VALUE),                          \
        CCS_TPL_SEL_ONE(STATE_TYPE, ALIGN_VALUE, CCS_TPL_INT64, CCS_TPL_INT32, SHARED_VALUE),                          \
        CCS_TPL_SEL_ONE(STATE_TYPE, ALIGN_VALUE, CCS_TPL_INT64, CCS_TPL_INT64, SHARED_VALUE)

#define CCS_TPL_SEL_STATE(STATE_TYPE)                                                                                  \
    CCS_TPL_SEL_PLAN_INDEX(STATE_TYPE, 0, 0), CCS_TPL_SEL_PLAN_INDEX(STATE_TYPE, 0, 1),                                \
        CCS_TPL_SEL_PLAN_INDEX(STATE_TYPE, 1, 0), CCS_TPL_SEL_PLAN_INDEX(STATE_TYPE, 1, 1)

ASCENDC_TPL_SEL(CCS_TPL_SEL_STATE(CCS_TPL_BF16), CCS_TPL_SEL_STATE(CCS_TPL_FP32), );

#undef CCS_TPL_SEL_STATE
#undef CCS_TPL_SEL_PLAN_INDEX
#undef CCS_TPL_SEL_ONE

} // namespace GDN

#endif // COMPACT_CONV_STATE_TILING_KEY_H
