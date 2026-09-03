#include "aclnn_chunk_gated_delta_rule_bwd_finalize.h"
#include "chunk_gated_delta_rule_bwd_finalize.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr || IsContiguous(tensor)) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnChunkGatedDeltaRuleBwdFinalizeGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *dO, const aclTensor *du,
    const aclTensor *g, const aclTensor *beta, const aclTensor *h,
    const aclTensor *dh, const aclTensor *a,
    const aclTensor *qRstdOptional, const aclTensor *kRstdOptional,
    const aclTensor *betaRawOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    double scale, int64_t chunkSize, bool useQkL2NormInKernel,
    bool useBetaSigmoidInKernel, bool useGateInKernel, bool stateVFirst,
    bool useExp2,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dbetaOut, const aclTensor *dgOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleBwdFinalize,
        DFX_IN(q, k, v, vNew, dO, du, g, beta, h, dh, a,
               qRstdOptional, kRstdOptional, betaRawOptional,
               cuSeqlensOptional, chunkIndicesOptional),
        DFX_OUT(dqOut, dkOut, dvOut, dbetaOut, dgOut));

    CHECK_COND(q != nullptr && k != nullptr && v != nullptr && vNew != nullptr &&
                   dO != nullptr && du != nullptr && g != nullptr && beta != nullptr &&
                   h != nullptr && dh != nullptr && a != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "Required input must not be nullptr.");
    CHECK_COND(dqOut != nullptr && dkOut != nullptr && dvOut != nullptr &&
                   dbetaOut != nullptr && dgOut != nullptr &&
                   workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "Output, workspaceSize and executor must not be nullptr.");
    CHECK_COND(chunkSize == 64, ACLNN_ERR_PARAM_INVALID, "chunkSize only supports 64.");
    CHECK_COND(!useGateInKernel, ACLNN_ERR_PARAM_INVALID,
               "useGateInKernel only supports false.");
    CHECK_COND(useExp2, ACLNN_ERR_PARAM_INVALID, "useExp2 only supports true.");
    CHECK_COND((cuSeqlensOptional == nullptr) == (chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional and chunkIndicesOptional must be both present or absent.");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorPtr = uniqueExecutor.get();

    CHECK_RET(MakeContiguous(q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(vNew, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(dO, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(du, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(beta, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(h, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(dh, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(a, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(qRstdOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(kRstdOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(betaRawOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    const auto result = l0op::ChunkGatedDeltaRuleBwdFinalize(
        q, k, v, vNew, dO, du, g, beta, h, dh, a,
        qRstdOptional, kRstdOptional, betaRawOptional,
        cuSeqlensOptional, chunkIndicesOptional,
        scale, chunkSize, useQkL2NormInKernel,
        useBetaSigmoidInKernel, useGateInKernel, stateVFirst, useExp2,
        dqOut, dkOut, dvOut, dbetaOut, dgOut,
        executorPtr);
    for (const aclTensor *tensor : result) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *outputs[] = {dqOut, dkOut, dvOut, dbetaOut, dgOut};
    for (size_t index = 0; index < result.size(); ++index) {
        const aclTensor *copied = l0op::ViewCopy(result[index], outputs[index], executorPtr);
        CHECK_RET(copied != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleBwdFinalize(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleBwdFinalize);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkGatedDeltaRuleBwdFinalize launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
