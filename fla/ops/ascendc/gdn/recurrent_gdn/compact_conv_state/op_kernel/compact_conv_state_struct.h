/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef COMPACT_CONV_STATE_STRUCT_H
#define COMPACT_CONV_STATE_STRUCT_H

#include <cstdint>

namespace GDN {

constexpr uint32_t CCS_BUFFERS_2 = 2U;
constexpr uint32_t CCS_ROLE_BUFFERS_2 = 2U;
constexpr uint32_t CCS_CHANNEL_TILE_256 = 256U;
constexpr uint32_t CCS_HISTORY_MAX_32 = 32U;
constexpr uint32_t CCS_SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;

#pragma pack(push, 8)
struct alignas(8) CompactConvStateTilingData {
    uint32_t taskCount;
    uint32_t channelCount;
    uint32_t layerCount;
    uint32_t historyLen;
    uint32_t tilesPerRowLayer;
    uint64_t bufferElements;
    uint64_t sharedDimStride;
    int64_t nullBlockId;
    uint64_t stateIndicesStride;
};
#pragma pack(pop)

} // namespace GDN

#endif // COMPACT_CONV_STATE_STRUCT_H
