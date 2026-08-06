#ifndef OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_INTRA_H
#define OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_INTRA_H

#include <array>
#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 4> ChunkKdaBwdIntra(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *gk,
    const aclTensor *beta,
    const aclTensor *dAqk,
    const aclTensor *dAkk,
    const aclTensor *dq,
    const aclTensor *dk,
    const aclTensor *db,
    const aclTensor *dg,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    bool safeGate,
    int64_t layoutMode,
    int64_t totalChunks,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dbOut,
    const aclTensor *dgOut,
    aclOpExecutor *executor);
} // namespace l0op

#endif
