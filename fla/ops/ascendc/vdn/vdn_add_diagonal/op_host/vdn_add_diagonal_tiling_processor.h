// SPDX-License-Identifier: BSD-3-Clause
#ifndef VDN_ADD_DIAGONAL_TILING_PROCESSOR_H
#define VDN_ADD_DIAGONAL_TILING_PROCESSOR_H

#include <algorithm>
#include <cstdint>
#include <limits>

namespace VdnAddDiagonalHost {
constexpr int64_t kHeadDim = 128;

inline bool ValidBatch(int64_t count)
{
    return count > 0 && count <= std::numeric_limits<int32_t>::max() / 2;
}
inline uint32_t BlockDim(uint32_t count, uint32_t coreCount)
{
    return std::min(2U * count, coreCount);
}
} // namespace VdnAddDiagonalHost

#endif  // VDN_ADD_DIAGONAL_TILING_PROCESSOR_H
