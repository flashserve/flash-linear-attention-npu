#ifndef OP_API_INC_LEVEL0_OP_CHUNK_GDN_BWD_INTRA_H
#define OP_API_INC_LEVEL0_OP_CHUNK_GDN_BWD_INTRA_H

#include <array>
#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 3> ChunkGdnBwdIntra(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *dO,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    bool useExp2,
    int64_t stage,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *dvLocalOut,
    aclOpExecutor *executor);
} // namespace l0op

#endif
