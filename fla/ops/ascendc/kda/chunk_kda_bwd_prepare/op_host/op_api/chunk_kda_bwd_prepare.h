#ifndef OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_PREPARE_H
#define OP_API_INC_LEVEL0_OP_CHUNK_KDA_BWD_PREPARE_H

#include <array>
#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 3> ChunkKdaBwdPrepare(
    const aclTensor *aqk,
    const aclTensor *vNew,
    const aclTensor *dO,
    const aclTensor *h,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    bool stateVFirst,
    const aclTensor *dAqkOut,
    const aclTensor *dvOut,
    const aclTensor *dqRawOut,
    aclOpExecutor *executor);
} // namespace l0op

#endif
