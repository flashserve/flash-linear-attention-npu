#include "aclnn_chunk_kda_bwd_recompute.h"
#include "chunk_kda_bwd_recompute.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

struct ChunkKdaBwdRecomputeParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *a = nullptr;
    const aclTensor *aLogOptional = nullptr;
    const aclTensor *dtBiasOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    int64_t chunkSize = 64;
    bool useExp2 = true;
    double lowerBound = -5.0;
    const aclTensor *wOut = nullptr;
    const aclTensor *uOut = nullptr;
    const aclTensor *qgOut = nullptr;
    const aclTensor *kgOut = nullptr;
    const aclTensor *gkOutOptional = nullptr;
};

aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus ParamsDataContiguous(ChunkKdaBwdRecomputeParams &params, aclOpExecutor *executor)
{
    CHECK_RET(DataContiguous(params.q, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.k, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.v, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.g, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.beta, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.a, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.aLogOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.dtBiasOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.wOut, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.uOut, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.qgOut, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.kgOut, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.gkOutOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnChunkKdaBwdRecomputeGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g, const aclTensor *beta,
    const aclTensor *a, const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional, int64_t chunkSize,
    bool useExp2, double lowerBound, const aclTensor *wOut, const aclTensor *uOut, const aclTensor *qgOut,
    const aclTensor *kgOut, const aclTensor *gkOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    ChunkKdaBwdRecomputeParams params{
        q, k, v, g, beta, a, aLogOptional, dtBiasOptional, cuSeqlensOptional, chunkIndicesOptional,
        chunkSize, useExp2, lowerBound, wOut, uOut, qgOut, kgOut, gkOutOptional};
    CHECK_COND(params.q != nullptr && params.k != nullptr && params.v != nullptr && params.g != nullptr &&
                   params.beta != nullptr && params.a != nullptr && params.wOut != nullptr &&
                   params.uOut != nullptr && params.qgOut != nullptr && params.kgOut != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "Required tensors must not be nullptr.");
    CHECK_COND(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    CHECK_RET(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const bool useGateInKernel = params.gkOutOptional != nullptr;
    auto outputs = l0op::ChunkKdaBwdRecompute(
        params.q, params.k, params.v, params.g, params.beta, params.a, params.aLogOptional,
        params.dtBiasOptional, params.cuSeqlensOptional, params.chunkIndicesOptional, params.chunkSize,
        useGateInKernel, params.useExp2, params.lowerBound, params.wOut, params.uOut, params.qgOut,
        params.kgOut, params.gkOutOptional, executorPtr);
    CHECK_COND(outputs[0] != nullptr, ACLNN_ERR_INNER_NULLPTR, "ChunkKdaBwdRecompute launch failed.");

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkKdaBwdRecompute(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
