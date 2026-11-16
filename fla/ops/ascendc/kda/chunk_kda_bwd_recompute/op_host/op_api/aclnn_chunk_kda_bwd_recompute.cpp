#include "op_common/log/log.h"
#include "aclnn_chunk_kda_bwd_recompute.h"
#include "chunk_kda_bwd_recompute.h"
#include <cstring>

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

static constexpr const char *OP_NAME = "aclnnChunkKdaBwdRecompute";

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
    const char *inputLayout = "BNSD";
    const aclTensor *wOut = nullptr;
    const aclTensor *uOut = nullptr;
    const aclTensor *qgOut = nullptr;
    const aclTensor *kgOut = nullptr;
    const aclTensor *gkOutOptional = nullptr;
};

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    if (!(tensor != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: tensor != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(ChunkKdaBwdRecomputeParams &params, aclOpExecutor *executor)
{
    if (!(DataContiguous(params.q, executor) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.k, executor) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.v, executor) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.g, executor) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.beta, executor) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.a, executor) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.aLogOptional, executor) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.dtBiasOptional, executor) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.wOut, executor) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.uOut, executor) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.qgOut, executor) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.kgOut, executor) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.gkOutOptional, executor) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnChunkKdaBwdRecomputeGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g, const aclTensor *beta,
    const aclTensor *a, const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional, int64_t chunkSize,
    bool useExp2, double lowerBound, const char *inputLayout,
    const aclTensor *wOut, const aclTensor *uOut, const aclTensor *qgOut,
    const aclTensor *kgOut, const aclTensor *gkOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    ChunkKdaBwdRecomputeParams params{
        q, k, v, g, beta, a, aLogOptional, dtBiasOptional, cuSeqlensOptional, chunkIndicesOptional,
        chunkSize, useExp2, lowerBound, inputLayout, wOut, uOut, qgOut, kgOut, gkOutOptional};
    if (!(inputLayout && (std::strcmp(inputLayout, "BNSD") == 0 ||
                          std::strcmp(inputLayout, "NTD") == 0))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "inputLayout", inputLayout == nullptr ? "nullptr" : inputLayout,
            "inputLayout must be BNSD or NTD");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.q != nullptr && params.k != nullptr && params.v != nullptr &&
          params.g != nullptr && params.beta != nullptr && params.a != nullptr &&
          params.wOut != nullptr && params.uOut != nullptr && params.qgOut != nullptr &&
          params.kgOut != nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "q,k,v,g,beta,a,wOut,uOut,qgOut,kgOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "workspaceSize,executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME,
                           "%s: CREATE_EXECUTOR returned nullptr; propagate the resource first error.", __func__);
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto executorPtr = uniqueExecutor.get();

    if (!(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool useGateInKernel = params.gkOutOptional != nullptr;
    const aclTensor *gkBuffer = params.gkOutOptional;
    if (gkBuffer == nullptr) {
        gkBuffer = executorPtr->AllocTensor(params.g->GetViewShape(), DataType::DT_FLOAT, Format::FORMAT_ND);
        if (!(gkBuffer != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "Unable to allocate internal gk output.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    }
    auto outputs = l0op::ChunkKdaBwdRecompute(
        params.q, params.k, params.v, params.g, params.beta, params.a, params.aLogOptional,
        params.dtBiasOptional, params.cuSeqlensOptional, params.chunkIndicesOptional, params.chunkSize,
        useGateInKernel, params.useExp2, params.lowerBound, params.inputLayout,
        params.wOut, params.uOut, params.qgOut,
        params.kgOut, gkBuffer, executorPtr);
    if (!(outputs[0] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "ChunkKdaBwdRecompute launch failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

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
