/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_FINALIZE_TILING_KEY_H
#define CHUNK_KDA_FWD_FINALIZE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"
#include "chunk_kda_fwd_finalize_platform.h"

// MIX 实例核型必须是可展开为数值字面量的宏：ASCENDC_TPL_PRE 阶段会把
// ASCENDC_TPL_KERNEL_TYPE_SEL 的实参原样交给代码生成器解析，C++ 限定名无法解析。
// A6 为 1:1 → MIX_AIC_1_1；A5 与 A2/A3 为 1:2 → MIX_AIC_1_2。
// 宏只在本文件使用，用完立即 #undef，避免泄漏到包含 tiling_key.h 的其他 TU。
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
#define CHUNK_KDA_FWD_FINALIZE_MIX_AIC_KERNEL_TYPE ASCENDC_TPL_MIX_AIC_1_1
#else
#define CHUNK_KDA_FWD_FINALIZE_MIX_AIC_KERNEL_TYPE ASCENDC_TPL_MIX_AIC_1_2
#endif

namespace KdaFinalize {

ASCENDC_TPL_ARGS_DECL(
    ChunkKdaFwdFinalize,
    ASCENDC_TPL_BOOL_DECL(USE_AIV_INPUT_MOVER, 0, 1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),
        ASCENDC_TPL_BOOL_SEL(USE_AIV_INPUT_MOVER, 0)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(CHUNK_KDA_FWD_FINALIZE_MIX_AIC_KERNEL_TYPE),
        ASCENDC_TPL_BOOL_SEL(USE_AIV_INPUT_MOVER, 1)));

} // namespace KdaFinalize

#undef CHUNK_KDA_FWD_FINALIZE_MIX_AIC_KERNEL_TYPE

#endif // CHUNK_KDA_FWD_FINALIZE_TILING_KEY_H
