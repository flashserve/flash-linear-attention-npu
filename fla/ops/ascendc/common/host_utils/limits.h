/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 */
#ifndef FLA_NPU_ASCENDC_COMMON_HOST_UTILS_LIMITS_H
#define FLA_NPU_ASCENDC_COMMON_HOST_UTILS_LIMITS_H

#include <cstdint>
#include <limits>

// std::numeric_limits<T>::max() 这类三段模板限定统一收敛为宏，调用点不再出现 xx::xx::xx。
#define FLA_NPU_MAX(T)    std::numeric_limits<T>::max()
#define FLA_NPU_MIN(T)    std::numeric_limits<T>::min()
#define FLA_NPU_LOWEST(T) std::numeric_limits<T>::lowest()

#endif // FLA_NPU_ASCENDC_COMMON_HOST_UTILS_LIMITS_H
