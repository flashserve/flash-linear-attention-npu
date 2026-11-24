// SPDX-License-Identifier: BSD-3-Clause
#ifndef VDN_SCAN_TILING_PROCESSOR_H
#define VDN_SCAN_TILING_PROCESSOR_H

#include <algorithm>
#include <cstdint>
#include <limits>

namespace VdnScanHost {
constexpr int64_t kHeadDim = 128;

inline bool ValidDimensions(int64_t frames, int64_t heads)
{
    return frames > 0 && frames < std::numeric_limits<int32_t>::max() &&
           heads > 0 && heads < std::numeric_limits<int32_t>::max() / 2;
}
inline uint32_t BlockDim(uint32_t heads, uint32_t coreCount)
{
    return std::min(2U * heads, coreCount);
}
} // namespace VdnScanHost

#endif  // VDN_SCAN_TILING_PROCESSOR_H
