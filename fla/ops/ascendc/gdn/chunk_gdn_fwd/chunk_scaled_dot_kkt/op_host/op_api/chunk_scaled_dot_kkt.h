#ifndef OP_API_INC_LEVEL0_CHUNK_SCALED_DOT_KKT_H
#define OP_API_INC_LEVEL0_CHUNK_SCALED_DOT_KKT_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *ChunkScaledDotKkt(
    const aclTensor *k,
    const aclTensor *g,
    const aclTensor *beta,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *aOut,
    aclOpExecutor *executor);
}

#endif
