// SPDX-License-Identifier: BSD-3-Clause
#ifndef VDN_BRIDGE_TILING_PROCESSOR_H
#define VDN_BRIDGE_TILING_PROCESSOR_H

#include <algorithm>
#include <initializer_list>
#include <limits>
#include "../op_kernel/vdn_bridge_struct.h"

namespace VdnBridgeHost {
template <typename Shape>
bool HasShape(const Shape &shape, std::initializer_list<int64_t> dims)
{
    if (shape.GetDimNum() != dims.size()) {
        return false;
    }
    size_t axis = 0;
    for (int64_t dim : dims) {
        if (shape.GetDim(axis++) != dim) {
            return false;
        }
    }
    return true;
}

inline bool ValidTile(int64_t bankRows, int64_t heads, int64_t start, int64_t count)
{
    // Both the largest FP32 byte address and every derived index must fit the signed runtime ABI.
    constexpr int64_t maxMatrices = std::numeric_limits<int64_t>::max() /
                                    (VdnBridgeKernel::HEAD_DIM * VdnBridgeKernel::HEAD_DIM * static_cast<int64_t>(sizeof(float)));
    return bankRows > 1 && heads > 0 && heads <= maxMatrices / bankRows && count > 0 &&
           count < bankRows && start >= 0 && start <= bankRows - 1 - count;
}

inline uint32_t BlockDim(int64_t heads, int64_t count, uint32_t available)
{
    return static_cast<uint32_t>(std::min<int64_t>(heads * count, available));
}
} // namespace VdnBridgeHost
#endif
