#ifndef OP_API_INC_CHUNK_KDA_BWD_H
#define OP_API_INC_CHUNK_KDA_BWD_H

#include <array>

#include "aclnn/aclnn_base.h"

namespace l0op {

using ChunkKdaBwdOutputs = std::array<const aclTensor *, 7>;

ChunkKdaBwdOutputs ChunkKdaBwd(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *beta, const aclTensor *gk,
    const aclTensor *aqk, const aclTensor *akk,
    const aclTensor *wOptional, const aclTensor *qgOptional,
    const aclTensor *kgOptional, const aclTensor *vNewOptional,
    const aclTensor *hOptional, const aclTensor *dO,
    const aclTensor *rawGOptional, const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional, const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkIndicesOptional,
    float scale, int64_t chunkSize, bool safeGate,
    bool useGateInKernel, float lowerBound,
    bool disableRecompute, bool useExp2, bool stateVFirst,
    const aclTensor *dq,
    const aclTensor *dk, const aclTensor *dv,
    const aclTensor *db, const aclTensor *dg,
    const aclTensor *dAOptional,
    const aclTensor *dBiasOptional, aclOpExecutor *executor);

} // namespace l0op

#endif // OP_API_INC_CHUNK_KDA_BWD_H
