/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CAUSAL_CONV1D_BWD_TILING_KEY_H_
#define CAUSAL_CONV1D_BWD_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(CausalConv1dBwd, ASCENDC_TPL_BOOL_DECL(IS_VARLEN, 0, 1));
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(IS_VARLEN, 0)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(IS_VARLEN, 1)));

#endif // CAUSAL_CONV1D_BWD_TILING_KEY_H_
