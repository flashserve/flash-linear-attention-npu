#ifndef OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_RECOMPUTE_H
#define OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_RECOMPUTE_H

#include <array>
#include "opdev/op_executor.h"

namespace l0op {
// inputLayout is reserved and accepts only BNSD/NTD. Tensors must still
// satisfy the existing rank-4 BNSD contract; callers own packed batch views.
const std::array<const aclTensor *, 5> ChunkKdaBwdRecompute(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    bool useGateInKernel,
    bool useExp2,
    double lowerBound,
    const char *inputLayout,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *gkOut,
    aclOpExecutor *executor, const char *modelLayout = "", bool qkHeadMajor = false);
} // namespace l0op

#endif
