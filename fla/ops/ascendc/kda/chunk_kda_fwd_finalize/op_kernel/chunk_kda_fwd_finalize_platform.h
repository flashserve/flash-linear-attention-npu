/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_FINALIZE_PLATFORM_H
#define CHUNK_KDA_FWD_FINALIZE_PLATFORM_H

namespace KdaFinalize {

// 三段门控的唯一判定点：把全工程统一用 __NPU_ARCH__ 门控（9201/3510/2201），
// 收敛到这一处。struct/cube/kernel 的 per-arch 判定一律从这里的 bool 取，不再各自 #if。
// MIX 实例核型仍在 tiling_key.h 内以数值宏表达（ASCENDC_TPL_PRE 阶段只接受可展开成
// 数字的宏），属于同一平台契约的另一半。
struct FinalizePlatform {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
    // A6（Ascend960DT）：AIC:AIV = 1:1，单 AIV 串行搬 4 head。
    static constexpr bool isSingleAiv = true;
    static constexpr bool isNonArch22 = true;
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
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
