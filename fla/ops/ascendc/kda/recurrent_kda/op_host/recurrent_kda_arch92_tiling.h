/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#pragma once

#include <cstdint>

namespace optiling {
namespace rkda_arch92 {
// The arch92 kernel packs one value head per beta DMA row and retains
// FP32 state across time windows. Keep these sizes paired with InitLocalBuffers.
constexpr int64_t WINDOW_TOKENS = 8;
constexpr int64_t BETA_ROW = 16;

inline int64_t FixedBytes(int64_t k, int64_t v, int64_t guard)
{
    return WINDOW_TOKENS * (8 * k + 2 * v + 4 * BETA_ROW) + 64 + guard;
}

inline int64_t WorkingBytes(int64_t k, int64_t v, int64_t guard)
{
    const int64_t windowScratch = WINDOW_TOKENS * (4 * v + 12 * k + 4 * BETA_ROW);
    const int64_t gateScratch = k == 128 ? WINDOW_TOKENS * k * sizeof(float) : 0;
    return FixedBytes(k, v, guard) + windowScratch + gateScratch;
}

inline int64_t ComputeBytesPerV(int64_t k)
{
    return (k == 128 ? 4 : 8) * k + 8;
}

inline uint32_t BlockDim(uint64_t tasks, uint64_t cores)
{
    const uint64_t wave = (tasks + cores - 1) / cores;
    return static_cast<uint32_t>((tasks + wave - 1) / wave);
}
} // namespace rkda_arch92
} // namespace optiling
