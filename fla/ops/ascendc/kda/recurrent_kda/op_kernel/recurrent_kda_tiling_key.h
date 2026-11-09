/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef RECURRENT_KDA_TILING_KEY_H_
#define RECURRENT_KDA_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(RecurrentKda, ASCENDC_TPL_BOOL_DECL(INPLACE_STATE, 0, 1));
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(INPLACE_STATE, 0)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(INPLACE_STATE, 1)));

#endif // RECURRENT_KDA_TILING_KEY_H_
