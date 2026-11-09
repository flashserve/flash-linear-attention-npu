/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_FINALIZE_PLATFORM_H
#define CHUNK_KDA_FWD_FINALIZE_PLATFORM_H

#include <cstdint>

#include "ascendc/host_api/tiling/template_argument.h"

namespace KdaFinalize {

// MIX 实例核型必须保持“可展开为数值字面量的宏”：ASCENDC_TPL_PRE 阶段会把
// ASCENDC_TPL_KERNEL_TYPE_SEL 的实参原样交给代码生成器解析，只有宏会被展开成
// 数字，C++ 限定名（如 FinalizePlatform::kMixAicKernelType）无法被解析。
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
#define CHUNK_KDA_FWD_FINALIZE_MIX_AIC_KERNEL_TYPE ASCENDC_TPL_MIX_AIC_1_1
#else
#define CHUNK_KDA_FWD_FINALIZE_MIX_AIC_KERNEL_TYPE ASCENDC_TPL_MIX_AIC_1_2
#endif

// 三段门控的唯一判定点：把全工程散点的 __NPU_ARCH__==9201 / __CCE_AICORE__==310
// 收敛到这一处。struct/cube/tiling_key/kernel 的 per-arch 判定一律从这里的 bool 取，
// 不再各自 #if。
struct FinalizePlatform {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
    // A6（Ascend960DT）：AIC:AIV = 1:1，单 AIV 串行搬 4 head。
    static constexpr bool isSingleAiv = true;
    static constexpr bool isNonArch22 = true;
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    // A5（Ascend950）：AIC:AIV = 1:2，双 AIV 各搬 2 head。
    static constexpr bool isSingleAiv = false;
    static constexpr bool isNonArch22 = true;
#else
    // A2/A3（DAV_2201）：无 AIV input mover，AIC 独立完成搬运与计算。
    static constexpr bool isSingleAiv = false;
    static constexpr bool isNonArch22 = false;
#endif
};

} // namespace KdaFinalize

#endif // CHUNK_KDA_FWD_FINALIZE_PLATFORM_H