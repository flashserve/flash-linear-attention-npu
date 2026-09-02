#ifndef KDA_GATE_BWD_POST_COMMON_H
#define KDA_GATE_BWD_POST_COMMON_H

#include <cstdint>

namespace KDA {

constexpr int32_t KDA_GATE_POST_MAX_CHUNKS = 1024;

// Private tiling contract for the delivery fallback that applies the raw-gate
// chain rule after the fused ABC kernel has produced chunk-local accumulated
// dg.  It is intentionally not exposed through the public L2 ABI.
#define KDA_GATE_POST_SCALAR_FIELDS \
    int32_t batch; \
    int32_t seqlen; \
    int32_t headNum; \
    int32_t keyDim; \
    int32_t chunkSize; \
    int32_t chunkNum; \
    int32_t chunkNumPerBatch; \
    int32_t usedCoreNum; \
    int8_t isVarLen; \
    int8_t hasDtBias; \
    int8_t inputScanned; \
    int8_t reserved1; \
    float lowerBound;

struct KdaGateBwdPostScalarTilingData {
    KDA_GATE_POST_SCALAR_FIELDS
};

struct KdaGateBwdPostTilingData {
    KDA_GATE_POST_SCALAR_FIELDS
};

#undef KDA_GATE_POST_SCALAR_FIELDS

} // namespace KDA

#endif // KDA_GATE_BWD_POST_COMMON_H
