/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef MERGE_FWD_BWD_TILING_KEY_H
#define MERGE_FWD_BWD_TILING_KEY_H

#ifndef TORCH_MODE
#include "ascendc/host_api/tiling/template_argument.h"
#endif

namespace MergeFwBwd {

#define MERGE_FWD_BWD_TPL_BF16 10
#define MERGE_FWD_BWD_TPL_FP32 30

#ifndef TORCH_MODE
ASCENDC_TPL_ARGS_DECL(MergeFwdBwd,
    ASCENDC_TPL_DTYPE_DECL(D_T, MERGE_FWD_BWD_TPL_BF16, MERGE_FWD_BWD_TPL_FP32),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T, MERGE_FWD_BWD_TPL_BF16)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T, MERGE_FWD_BWD_TPL_FP32)),
);
#endif

} // namespace MergeFwBwd

#endif
