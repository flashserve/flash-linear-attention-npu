/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef GDN_RECOVERSSM_COMMIT_TILING_KEY_H
#define GDN_RECOVERSSM_COMMIT_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace GDN {

#define GRSC_TPL_BF16 10
#define GRSC_TPL_FP32 30

ASCENDC_TPL_ARGS_DECL(GdnRecoverssmCommit,
                      ASCENDC_TPL_DTYPE_DECL(D_T_STATE, GRSC_TPL_BF16, GRSC_TPL_FP32),
                      ASCENDC_TPL_BOOL_DECL(ALIGN_MODE, 0, 1), );

#define GRSC_TPL_SEL_ONE(STATE_TYPE, ALIGN_VALUE)                                                                     \
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_STATE, STATE_TYPE),                                                \
                         ASCENDC_TPL_BOOL_SEL(ALIGN_MODE, ALIGN_VALUE), )

ASCENDC_TPL_SEL(GRSC_TPL_SEL_ONE(GRSC_TPL_BF16, 0), GRSC_TPL_SEL_ONE(GRSC_TPL_BF16, 1),
                GRSC_TPL_SEL_ONE(GRSC_TPL_FP32, 0), GRSC_TPL_SEL_ONE(GRSC_TPL_FP32, 1), );

#undef GRSC_TPL_SEL_ONE

} // namespace GDN

#endif // GDN_RECOVERSSM_COMMIT_TILING_KEY_H
