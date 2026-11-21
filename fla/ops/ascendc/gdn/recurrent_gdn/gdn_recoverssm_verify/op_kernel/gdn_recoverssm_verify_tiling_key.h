/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef GDN_RECOVERSSM_VERIFY_TILING_KEY_H
#define GDN_RECOVERSSM_VERIFY_TILING_KEY_H

#ifndef TORCH_MODE
#include "ascendc/host_api/tiling/template_argument.h"
#endif

namespace GDN {

#define GRSV_TPL_BF16 10
#define GRSV_TPL_FP32 30

#ifndef TORCH_MODE
ASCENDC_TPL_ARGS_DECL(GdnRecoverssmVerify, ASCENDC_TPL_DTYPE_DECL(D_T_STATE, GRSV_TPL_BF16, GRSV_TPL_FP32),
                      ASCENDC_TPL_BOOL_DECL(USE_QK_L2NORM, 0, 1), );

#define GRSV_TPL_SEL_ONE(STATE_TYPE, NORM_VALUE)                                                                       \
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_STATE, STATE_TYPE),                                                 \
                         ASCENDC_TPL_BOOL_SEL(USE_QK_L2NORM, NORM_VALUE), )

ASCENDC_TPL_SEL(GRSV_TPL_SEL_ONE(GRSV_TPL_BF16, 0), GRSV_TPL_SEL_ONE(GRSV_TPL_BF16, 1),
                GRSV_TPL_SEL_ONE(GRSV_TPL_FP32, 0), GRSV_TPL_SEL_ONE(GRSV_TPL_FP32, 1), );

#undef GRSV_TPL_SEL_ONE
#endif

} // namespace GDN

#endif // GDN_RECOVERSSM_VERIFY_TILING_KEY_H
