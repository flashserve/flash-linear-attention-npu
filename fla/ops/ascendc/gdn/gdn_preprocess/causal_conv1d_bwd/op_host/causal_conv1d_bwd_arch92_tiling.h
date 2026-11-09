/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#pragma once

#include <algorithm>
#include <cstdint>

namespace optiling {
namespace causal_conv_arch92 {
struct TimeTile {
    uint32_t rows;
    uint64_t bytes;
};

// BD is a multiple of 16, so each row already satisfies 32-byte DMA alignment.
inline TimeTile SelectTimeTile(uint32_t time, uint32_t tokens, uint32_t cores, uint32_t width,
                               uint32_t bd, bool b16, uint32_t activation, bool hasState, uint64_t ub)
{
    const bool direct = b16 && activation == 1 && !hasState && width > 0 && width <= 4 && bd == 64;
    const uint32_t tokensPerCore = tokens / cores + (tokens % cores != 0);
    const uint32_t limit = direct ? std::min(224U, std::max(32U, tokensPerCore)) : 64U;
    uint32_t rows = std::min(time, limit);
    uint64_t bytes = 0;
    while (rows > 0) {
        const uint64_t tile = static_cast<uint64_t>(rows) * bd;
        const uint64_t halo = (static_cast<uint64_t>(rows) + width - 1) * bd;
        const uint64_t weight = static_cast<uint64_t>(width) * bd;
        if (direct) {
            bytes = (3 * tile + 4 * halo + weight) * 2 + (tile + halo + weight + bd) * 4;
        } else {
            const uint64_t scratch = std::max(tile, std::max(halo, weight));
            bytes = (2 * tile + halo + 2 * scratch + 2 * weight + bd) * 4;
            if (b16) {
                bytes += scratch * 2;
            }
            if (activation == 0) {
                bytes += (width * tile + tile) * 4;
            } else if (activation == 1 || activation == 2) {
                bytes += 2 * halo * 4;
            }
            if (hasState) {
                bytes += bd * 4;
            }
        }
        if (bytes <= ub || rows == 1) {
            break;
        }
        --rows;
    }
    return {rows, bytes};
}
} // namespace causal_conv_arch92
} // namespace optiling
