/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef GDN_RECOVERSSM_COMMIT_STRUCT_H
#define GDN_RECOVERSSM_COMMIT_STRUCT_H

#include <cstdint>

namespace GDN {

constexpr uint32_t GRSC_DIM_128 = 128U;
constexpr uint32_t GRSC_PAYLOAD = 2U * GRSC_DIM_128 + 1U;
constexpr uint32_t GRSC_SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;

#pragma pack(push, 8)
struct alignas(8) GdnRecoverssmCommitTilingData {
    uint32_t vectorCoreNum;
    uint32_t ubCalSize;
    uint32_t taskCount;
    uint32_t batchSize;
    uint32_t valueHeads;
    uint32_t valueDim;
    uint32_t keyDim;
    uint32_t specQueryLen;
    uint32_t layerCount;
    uint32_t stateIndicesDtype;
    uint32_t planDtype;
    uint32_t alignMode;

    uint64_t stateHeadStride;
    uint64_t stateValueStride;
    uint64_t stateKeyStride;
    uint64_t replayHeadStride;
    uint64_t replayPosStride;
    uint64_t replayDimStride;
    uint64_t stateIndicesStride;
    uint64_t commitLensStride;
    uint64_t finalStateIndicesStride;
    uint64_t boundaryStateIndicesStride;
    uint64_t boundaryRecoveryLensStride;
};
#pragma pack(pop)

} // namespace GDN

#endif // GDN_RECOVERSSM_COMMIT_STRUCT_H
