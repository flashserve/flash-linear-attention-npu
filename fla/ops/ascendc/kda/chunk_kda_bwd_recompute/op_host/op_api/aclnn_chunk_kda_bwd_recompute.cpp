#include "aclnn_chunk_kda_bwd_recompute.h"
#include "chunk_kda_bwd_recompute.h"
#include <cstring>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

#include "../../../../common/host_utils/log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

constexpr const char *CHUNK_KDA_BWD_RECOMPUTE_ENTITY = "aclnnChunkKdaBwdRecompute";

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
    if (tensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Making input contiguous failed in %s.", CHUNK_KDA_BWD_RECOMPUTE_ENTITY);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(ChunkKdaBwdRecomputeParams &params, aclOpExecutor *executor)
{
    aclnnStatus status = DataContiguous(params.q, executor);
    if (status != ACLNN_SUCCESS) return status;
    status = DataContiguous(params.k, executor);
    if (status != ACLNN_SUCCESS) return status;
    status = DataContiguous(params.v, executor);
    if (status != ACLNN_SUCCESS) return status;
    status = DataContiguous(params.g, executor);
    if (status != ACLNN_SUCCESS) return status;
    status = DataContiguous(params.beta, executor);
    if (status != ACLNN_SUCCESS) return status;
    status = DataContiguous(params.a, executor);
    if (status != ACLNN_SUCCESS) return status;
    status = DataContiguous(params.aLogOptional, executor);
    if (status != ACLNN_SUCCESS) return status;
    status = DataContiguous(params.dtBiasOptional, executor);
    if (status != ACLNN_SUCCESS) return status;
    status = DataContiguous(params.wOut, executor);
    if (status != ACLNN_SUCCESS) return status;
    status = DataContiguous(params.uOut, executor);
    if (status != ACLNN_SUCCESS) return status;
    status = DataContiguous(params.qgOut, executor);
    if (status != ACLNN_SUCCESS) return status;
    status = DataContiguous(params.kgOut, executor);
    if (status != ACLNN_SUCCESS) return status;
    status = DataContiguous(params.gkOutOptional, executor);
    if (status != ACLNN_SUCCESS) return status;
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
    if (inputLayout == nullptr || (std::strcmp(inputLayout, "BNSD") != 0 && std::strcmp(inputLayout, "NTD") != 0)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_RECOMPUTE_ENTITY, "inputLayout",
                                                   inputLayout == nullptr ? "null" : inputLayout, "BNSD or NTD");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const aclTensor *required[] = {params.q, params.k, params.v, params.g, params.beta, params.a,
                                   params.wOut, params.uOut, params.qgOut, params.kgOut};
    const char *names[] = {"q", "k", "v", "g", "beta", "a", "wOut", "uOut", "qgOut", "kgOut"};
    for (size_t idx = 0; idx < sizeof(required) / sizeof(required[0]); ++idx) {
        if (required[idx] == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_KDA_BWD_RECOMPUTE_ENTITY, names[idx]);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    if (workspaceSize == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_KDA_BWD_RECOMPUTE_ENTITY, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_KDA_BWD_RECOMPUTE_ENTITY, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    auto ret = ParamsDataContiguous(params, executorPtr);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    const bool useGateInKernel = params.gkOutOptional != nullptr;
    const aclTensor *gkBuffer = params.gkOutOptional;
    if (gkBuffer == nullptr) {
        gkBuffer = executorPtr->AllocTensor(params.g->GetViewShape(), DataType::DT_FLOAT, Format::FORMAT_ND);
        CHECK_COND(gkBuffer != nullptr, ACLNN_ERR_INNER_NULLPTR, "Unable to allocate internal gk output.");
    }
    auto outputs = l0op::ChunkKdaBwdRecompute(
        params.q, params.k, params.v, params.g, params.beta, params.a, params.aLogOptional,
        params.dtBiasOptional, params.cuSeqlensOptional, params.chunkIndicesOptional, params.chunkSize,
        useGateInKernel, params.useExp2, params.lowerBound, params.inputLayout,
        params.wOut, params.uOut, params.qgOut,
        params.kgOut, gkBuffer, executorPtr);
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
