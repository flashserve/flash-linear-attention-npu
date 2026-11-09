/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_FINALIZE_TILING_KEY_H
#define CHUNK_KDA_FWD_FINALIZE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"
#include "chunk_kda_fwd_finalize_platform.h"

namespace KdaFinalize {

ASCENDC_TPL_ARGS_DECL(
    ChunkKdaFwdFinalize,
    ASCENDC_TPL_BOOL_DECL(USE_AIV_INPUT_MOVER, 0, 1));

// 核型由 platform.h 的 CHUNK_KDA_FWD_FINALIZE_MIX_AIC_KERNEL_TYPE 提供
// （A6=1:1 → MIX_AIC_1_1，其余 A5/A2A3 → MIX_AIC_1_2）。这里必须传宏而不是
// C++ 常量：ASCENDC_TPL_PRE 阶段会把实参交给代码生成器解析，宏会展开成数值。
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),
        ASCENDC_TPL_BOOL_SEL(USE_AIV_INPUT_MOVER, 0)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(CHUNK_KDA_FWD_FINALIZE_MIX_AIC_KERNEL_TYPE),
        ASCENDC_TPL_BOOL_SEL(USE_AIV_INPUT_MOVER, 1)));

} // namespace KdaFinalize

#endif // CHUNK_KDA_FWD_FINALIZE_TILING_KEY_H
