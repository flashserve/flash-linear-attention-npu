/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef GDN_RECOVERSSM_COMMIT_STRUCT_H
#define GDN_RECOVERSSM_COMMIT_STRUCT_H

#include <cstdint>

namespace GDN {

constexpr uint32_t GRSC_DIM_128 = 128U;
constexpr uint32_t GRSC_PAYLOAD_257 = 2U * GRSC_DIM_128 + 1U;
constexpr uint32_t GRSC_RECORD_STRIDE_264 = 264U;
constexpr uint32_t GRSC_VALUE_TILE_64 = 64U;
constexpr uint32_t GRSC_VALUE_TILES_2 = GRSC_DIM_128 / GRSC_VALUE_TILE_64;
constexpr uint32_t GRSC_SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;

#pragma pack(push, 8)
struct alignas(8) GdnRecoverssmCommitTilingData {
    uint32_t taskCount;
    uint32_t valueHeads;
    uint32_t layerCount;

    uint64_t stateHeadStride;
    uint64_t stateValueStride;
    uint64_t replayHeadStride;
    uint64_t replayPosStride;
};
#pragma pack(pop)

} // namespace GDN

#endif // GDN_RECOVERSSM_COMMIT_STRUCT_H
