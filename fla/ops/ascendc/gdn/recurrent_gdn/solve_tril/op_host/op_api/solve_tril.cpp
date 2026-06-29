/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "aclnn_solve_tril.h"
#include <string>

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SolveTril);

const aclTensor* SolveTril(
    const aclTensor *x,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const char *layout,
    const aclTensor *xOut,
    aclOpExecutor *executor)
{
    L0_DFX(SolveTril, x, cuSeqlensOptional, chunkIndicesOptional, chunkSize, layout, xOut);

    const aclTensor *actualCuSeqlens = nullptr;
    if (cuSeqlensOptional) {
        actualCuSeqlens = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualCuSeqlens)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetOriginalFormat(Format::FORMAT_ND);
    }

    const aclTensor *actualChunkIndices = nullptr;
    if (chunkIndicesOptional) {
        actualChunkIndices = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualChunkIndices)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetOriginalFormat(Format::FORMAT_ND);
    }

    std::string layoutStr(layout ? layout : "bsnd");

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SolveTril,
        OP_INPUT(x, actualCuSeqlens, actualChunkIndices),
        OP_OUTPUT(xOut),
        OP_ATTR(chunkSize, layoutStr));
    if (ret != ACLNN_SUCCESS) {
        return nullptr;
    }
    return xOut;
}

}  // namespace l0op
