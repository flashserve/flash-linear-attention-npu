/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */

#ifndef GDN_HO_PIPELINE_CONTEXT_H
#define GDN_HO_PIPELINE_CONTEXT_H

#include <cstdint>

namespace GDN {

#if defined(__CCE__)
#define GDN_HO_HOST_DEVICE __forceinline__ [host, aicore]
#else
#define GDN_HO_HOST_DEVICE
#endif

constexpr int64_t HO_PIPELINE_DTYPE_FP16 = 0;
constexpr int64_t HO_PIPELINE_DTYPE_BF16 = 1;
constexpr int64_t HO_PIPELINE_K_HEAD_DIM = 128;
constexpr int64_t HO_PIPELINE_V_HEAD_DIM = 128;
constexpr int64_t HO_PIPELINE_CHUNK_SIZE = 64;
constexpr int64_t HO_PIPELINE_V_HEAD_DIM_WIDE = 256;
constexpr int64_t HO_PIPELINE_CHUNK_SIZE_WIDE = 128;

GDN_HO_HOST_DEVICE constexpr bool HoPipelineLayoutEligible(
    bool isAscend950, int64_t dataType, int64_t kHeadDim, int64_t vHeadDim,
    int64_t chunkSize)
{
    const bool inputDtypeEligible = dataType == HO_PIPELINE_DTYPE_FP16 ||
                                    dataType == HO_PIPELINE_DTYPE_BF16;
    const bool vHeadDimEligible = vHeadDim == HO_PIPELINE_V_HEAD_DIM ||
                                  vHeadDim == HO_PIPELINE_V_HEAD_DIM_WIDE;
    const bool chunkSizeEligible = chunkSize == HO_PIPELINE_CHUNK_SIZE ||
                                   chunkSize == HO_PIPELINE_CHUNK_SIZE_WIDE;
    return isAscend950 && inputDtypeEligible &&
           kHeadDim == HO_PIPELINE_K_HEAD_DIM && vHeadDimEligible &&
           chunkSizeEligible;
}

struct HoPipelineContext {
    bool enabled{false};
    uint32_t producerGroups{0};
    uint32_t consumerGroups{0};
    uint64_t physicalShapeBatch{0};
    uint64_t chunksPerPhysicalBatch{0};
    uint64_t readyTaskCount{0};
};

}

#undef GDN_HO_HOST_DEVICE

#endif
