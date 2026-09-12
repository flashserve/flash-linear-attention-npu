#include "chunk_kda_bwd_prepare.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwdPrepare);

const std::array<const aclTensor *, 3> ChunkKdaBwdPrepare(
    const aclTensor *aqk, const aclTensor *vNew, const aclTensor *dO, const aclTensor *h,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    double scale, int64_t chunkSize, bool stateVFirst,
    const aclTensor *dAqkOut, const aclTensor *dvOut, const aclTensor *dqRawOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaBwdPrepare, aqk, vNew, dO, h, cuSeqlensOptional,
           chunkIndicesOptional, scale, chunkSize, stateVFirst,
           dAqkOut, dvOut, dqRawOut);
    const aclTensor *cuTensor = nullptr;
    const aclTensor *indicesTensor = nullptr;
    if (cuSeqlensOptional != nullptr) {
        cuTensor = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        indicesTensor = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        if (cuTensor == nullptr || indicesTensor == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "failed to convert KernelA metadata");
            return {nullptr, nullptr, nullptr};
        }
        for (const aclTensor *tensor : {cuTensor, indicesTensor}) {
            auto *mutableTensor = const_cast<aclTensor *>(tensor);
            mutableTensor->SetStorageFormat(Format::FORMAT_ND);
            mutableTensor->SetViewFormat(Format::FORMAT_ND);
            mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
        }
    }
    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwdPrepare,
        OP_INPUT(aqk, vNew, dO, h, cuTensor, indicesTensor),
        OP_OUTPUT(dAqkOut, dvOut, dqRawOut),
        OP_ATTR(scale, chunkSize, stateVFirst));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwdPrepare failed");
        return {nullptr, nullptr, nullptr};
    }
    return {dAqkOut, dvOut, dqRawOut};
}

} // namespace l0op
