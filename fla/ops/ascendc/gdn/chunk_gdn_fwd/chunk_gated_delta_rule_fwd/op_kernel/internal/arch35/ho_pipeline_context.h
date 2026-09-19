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

// Keep host reservation and device enablement on one scalar predicate.  The
// header has no framework or serialized-tiling dependency, so it is safe to
// include from both sides of the arch35 private implementation.
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

// Private Phase-6 context shared by the arch35 fused H/O producer and
// consumer.  It is deliberately independent of serialized tiling data: host
// tiling reserves the eligible layout, while the device fills enabled and
// group counts from the same invocation metadata before entering H/O.
struct HoPipelineContext {
    bool enabled{false};
    uint32_t producerGroups{0};
    uint32_t consumerGroups{0};
    uint64_t physicalShapeBatch{0};
    uint64_t chunksPerPhysicalBatch{0};
    uint64_t readyTaskCount{0};
};

}  // namespace GDN

#undef GDN_HO_HOST_DEVICE

#endif  // GDN_HO_PIPELINE_CONTEXT_H
