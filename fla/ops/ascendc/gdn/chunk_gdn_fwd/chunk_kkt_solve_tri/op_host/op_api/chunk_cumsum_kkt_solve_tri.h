#ifndef OP_API_INC_LEVEL0_CHUNK_CUMSUM_KKT_SOLVE_TRI_H
#define OP_API_INC_LEVEL0_CHUNK_CUMSUM_KKT_SOLVE_TRI_H

#include <array>

#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 2> ChunkCumsumKktSolveTri(
    const aclTensor *k,
    const aclTensor *g,
    const aclTensor *beta,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *gCumsumOut,
    const aclTensor *aOut,
    aclOpExecutor *executor);
}

#endif
