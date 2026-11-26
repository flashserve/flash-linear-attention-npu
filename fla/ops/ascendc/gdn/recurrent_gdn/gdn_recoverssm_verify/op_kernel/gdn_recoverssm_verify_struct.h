/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef GDN_RECOVERSSM_VERIFY_STRUCT_H
#define GDN_RECOVERSSM_VERIFY_STRUCT_H

#include <cstdint>

namespace GDN {

constexpr uint32_t GRSV_REQUIRED_DIM_128 = 128U;
constexpr uint32_t GRSV_TOKEN_TILE_4 = 4U;
constexpr uint32_t GRSV_INPUT_BUFFERS_2 = 2U;
constexpr uint32_t GRSV_STATE_BANKS_3 = 3U;
constexpr uint32_t GRSV_SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;

#pragma pack(push, 8)
struct alignas(8) GdnRecoverssmVerifyTilingData {
    uint32_t vectorCoreNum;
    uint32_t ubCalSize;
    uint32_t taskCount;
    uint32_t batchSize;
    uint32_t totalTokens;
    uint32_t keyHeads;
    uint32_t valueHeads;
    uint32_t keyDim;
    uint32_t valueDim;
    uint32_t specQueryLen;
    uint32_t checkpointDtype;
    uint32_t metadataDtype;
    uint32_t useQkL2NormInKernel;
    float scale;
    float softplusBeta;
    float softplusThreshold;

    uint64_t queryStrides[4] = {};
    uint64_t keyStrides[4] = {};
    uint64_t valueStrides[4] = {};
    uint64_t outStrides[4] = {};
    uint64_t aStrides[2] = {};
    uint64_t bStrides[2] = {};
    uint64_t checkpointStrides[4] = {};
    uint64_t replayStrides[4] = {};
    uint64_t queryStartStride = 0;
    uint64_t stateIndexStride = 0;
};
#pragma pack(pop)

} // namespace GDN

#endif // GDN_RECOVERSSM_VERIFY_STRUCT_H
