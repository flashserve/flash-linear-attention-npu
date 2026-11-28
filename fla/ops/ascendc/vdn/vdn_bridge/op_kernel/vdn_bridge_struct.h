// SPDX-License-Identifier: BSD-3-Clause
#ifndef VDN_BRIDGE_STRUCT_H
#define VDN_BRIDGE_STRUCT_H

#include <cstdint>

namespace VdnBridgeKernel {

constexpr int32_t HEAD_DIM = 128;
constexpr int32_t TILE_ROWS = 64;
struct VdnBridgeTilingData {
    int64_t frames;
    int64_t heads;
    int64_t start;
    int64_t count;
};

} // namespace VdnBridgeKernel

#endif
