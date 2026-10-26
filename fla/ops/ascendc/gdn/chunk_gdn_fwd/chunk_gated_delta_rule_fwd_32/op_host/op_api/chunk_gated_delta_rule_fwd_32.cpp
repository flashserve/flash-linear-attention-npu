#include "aclnn_chunk_gated_delta_rule_fwd_32.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkGatedDeltaRuleFwd32);

const aclTensor *ChunkGatedDeltaRuleFwd32(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g, const aclTensor *beta,
    const aclTensor *initialStateOptional, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, double scale, const aclTensor *o, const aclTensor *finalState,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkGatedDeltaRuleFwd32, q, k, v, g, beta, initialStateOptional, cuSeqlensOptional, chunkIndicesOptional,
           scale, o, finalState);
    const aclTensor *cu = nullptr;
    if (cuSeqlensOptional != nullptr) {
        cu = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        if (cu == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert cu_seqlens to tensor failed.");
            return nullptr;
        }
        const_cast<aclTensor *>(cu)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(cu)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(cu)->SetOriginalFormat(Format::FORMAT_ND);
    }
    const aclTensor *chunks = nullptr;
    if (chunkIndicesOptional != nullptr) {
        chunks = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        if (chunks == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert chunk_indices to tensor failed.");
            return nullptr;
        }
        const_cast<aclTensor *>(chunks)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(chunks)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(chunks)->SetOriginalFormat(Format::FORMAT_ND);
    }
    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRuleFwd32, OP_INPUT(q, k, v, g, beta, initialStateOptional, cu, chunks),
        OP_OUTPUT(o, finalState), OP_ATTR(scale));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return o;
}
}  // namespace l0op
