#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "chunk_scaled_dot_kkt.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkScaledDotKkt);

const aclTensor* ChunkScaledDotKkt(
    const aclTensor *k,
    const aclTensor *g,
    const aclTensor *beta,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *aOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkScaledDotKkt, k, g, beta, cuSeqlensOptional, chunkIndicesOptional, chunkSize, aOut);

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

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ChunkScaledDotKkt,
        OP_INPUT(k, g, beta, actualCuSeqlens, actualChunkIndices),
        OP_OUTPUT(aOut),
        OP_ATTR(chunkSize));
    if (ret != ACLNN_SUCCESS) {
        return nullptr;
    }
    return aOut;
}

}  // namespace l0op