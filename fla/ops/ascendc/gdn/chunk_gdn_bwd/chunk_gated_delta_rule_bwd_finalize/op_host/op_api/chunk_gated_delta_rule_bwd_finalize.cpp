#include "chunk_gated_delta_rule_bwd_finalize.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ChunkGatedDeltaRuleBwdFinalize);

namespace {

const aclTensor *ConvertIntArray(const aclIntArray *array, aclOpExecutor *executor)
{
    if (array == nullptr) {
        return nullptr;
    }
    const aclTensor *tensor = executor->ConvertToTensor(array, DataType::DT_INT64);
    if (tensor == nullptr) {
        return nullptr;
    }
    auto *mutableTensor = const_cast<aclTensor *>(tensor);
    mutableTensor->SetStorageFormat(Format::FORMAT_ND);
    mutableTensor->SetViewFormat(Format::FORMAT_ND);
    mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
    return tensor;
}

} // namespace

const std::array<const aclTensor *, 5> ChunkGatedDeltaRuleBwdFinalize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *dO, const aclTensor *du,
    const aclTensor *g, const aclTensor *beta, const aclTensor *h,
    const aclTensor *dh, const aclTensor *a,
    const aclTensor *qRstdOptional, const aclTensor *kRstdOptional,
    const aclTensor *betaRawOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    double scale, int64_t chunkSize, bool useQkL2NormInKernel,
    bool useBetaSigmoidInKernel, bool allowNegEigval, bool useGateInKernel, bool stateVFirst,
    bool useExp2,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dbetaOut, const aclTensor *dgOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkGatedDeltaRuleBwdFinalize, q, k, v, vNew, dO, du, g, beta, h, dh, a,
           qRstdOptional, kRstdOptional, betaRawOptional,
           cuSeqlensOptional, chunkIndicesOptional,
           scale, chunkSize, useQkL2NormInKernel,
           useBetaSigmoidInKernel, allowNegEigval, useGateInKernel, stateVFirst, useExp2,
           dqOut, dkOut, dvOut, dbetaOut, dgOut);

    const aclTensor *cuSeqlens = ConvertIntArray(cuSeqlensOptional, executor);
    const aclTensor *chunkIndices = ConvertIntArray(chunkIndicesOptional, executor);
    if ((cuSeqlensOptional != nullptr && cuSeqlens == nullptr) ||
        (chunkIndicesOptional != nullptr && chunkIndices == nullptr)) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert varlen index array failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRuleBwdFinalize,
        OP_INPUT(q, k, v, vNew, dO, du, g, beta, h, dh, a,
                 qRstdOptional, kRstdOptional, betaRawOptional,
                 cuSeqlens, chunkIndices),
        OP_OUTPUT(dqOut, dkOut, dvOut, dbetaOut, dgOut),
        OP_ATTR(scale, chunkSize, useQkL2NormInKernel,
                useBetaSigmoidInKernel, allowNegEigval, useGateInKernel, stateVFirst, useExp2));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {dqOut, dkOut, dvOut, dbetaOut, dgOut};
}

} // namespace l0op
