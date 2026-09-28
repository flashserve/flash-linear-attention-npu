#include "aclnn_chunk_kda_bwd_finalize.h"
#include "chunk_kda_bwd_finalize.h"

#include <cmath>
#include <utility>

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

aclnnStatus CheckRequired(const aclTensor *tensor, DataType dtype, const char *name)
{
    CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "%s must not be nullptr", name);
    CHECK_COND(IsContiguous(tensor), ACLNN_ERR_PARAM_INVALID, "%s must be contiguous", name);
    CHECK_COND(tensor->GetDataType() == dtype, ACLNN_ERR_PARAM_INVALID,
               "%s has invalid dtype", name);
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" aclnnStatus aclnnChunkKdaBwdFinalizeGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *gk, const aclTensor *rawG, const aclTensor *beta,
    const aclTensor *aLog, const aclTensor *dtBias, const aclTensor *akk,
    const aclTensor *vNew, const aclTensor *h, const aclTensor *dh,
    const aclTensor *dvScan, const aclTensor *dAqk, const aclTensor *dqRaw,
    const aclTensor *qRstdOptional, const aclTensor *kRstdOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    double scale, double lowerBound, int64_t chunkSize,
    bool safeGate, bool useGateInKernel, bool useExp2, bool stateVFirst,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dBetaOut, const aclTensor *dGOut,
    const aclTensor *dALogOut, const aclTensor *dDtBiasOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "workspaceSize/executor must not be nullptr");
    for (const auto &item : {
             std::pair<const aclTensor *, const char *>(q, "q"),
             {k, "k"}, {v, "v"}, {akk, "akk"},
             {vNew, "v_new"}, {h, "h"}, {dh, "dh"}, {dvScan, "dv_scan"},
             {dqOut, "dq"}, {dkOut, "dk"}, {dvOut, "dv"}}) {
        CHECK_RET(CheckRequired(item.first, DataType::DT_BF16, item.second) == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }
    CHECK_COND(beta != nullptr &&
                   (beta->GetDataType() == DataType::DT_BF16 || beta->GetDataType() == DataType::DT_FLOAT),
               ACLNN_ERR_PARAM_INVALID, "beta must be BF16 or FP32");
    CHECK_RET(CheckRequired(beta, beta->GetDataType(), "beta") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckRequired(dBetaOut, beta->GetDataType(), "d_beta") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    for (const auto &item : {
             std::pair<const aclTensor *, const char *>(gk, "gk"),
             {rawG, "raw_g"}, {dtBias, "dt_bias"}, {dAqk, "d_aqk"},
             {dqRaw, "dq_raw"}, {dGOut, "d_g"}, {dALogOut, "d_a_log"},
             {dDtBiasOut, "d_dt_bias"}}) {
        CHECK_RET(CheckRequired(item.first, DataType::DT_FLOAT, item.second) == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }
    CHECK_COND(aLog != nullptr && IsContiguous(aLog), ACLNN_ERR_PARAM_INVALID,
               "a_log must be a contiguous tensor");
    CHECK_COND(aLog->GetDataType() == DataType::DT_FLOAT ||
                   aLog->GetDataType() == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "a_log must be FP32 or BF16");
    CHECK_COND((qRstdOptional == nullptr) == (kRstdOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID, "q_rstd and k_rstd must appear together");
    if (qRstdOptional != nullptr) {
        CHECK_RET(CheckRequired(qRstdOptional, DataType::DT_FLOAT, "q_rstd") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckRequired(kRstdOptional, DataType::DT_FLOAT, "k_rstd") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }
    CHECK_COND((cuSeqlensOptional == nullptr) == (chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens and chunk_indices must appear together");
    CHECK_COND(std::isfinite(scale) && std::isfinite(lowerBound),
               ACLNN_ERR_PARAM_INVALID, "scale/lower_bound must be finite");
    CHECK_COND(chunkSize == 64 && safeGate && useGateInKernel && useExp2 && !stateVFirst,
               ACLNN_ERR_PARAM_INVALID, "unsupported A5 v1 attribute combination");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto result = l0op::ChunkKdaBwdFinalize(
        q, k, v, gk, rawG, beta, aLog, dtBias, akk, vNew, h, dh,
        dvScan, dAqk, dqRaw, qRstdOptional, kRstdOptional,
        cuSeqlensOptional, chunkIndicesOptional,
        scale, lowerBound, chunkSize, safeGate, useGateInKernel, useExp2, stateVFirst,
        dqOut, dkOut, dvOut, dBetaOut, dGOut, dALogOut, dDtBiasOut,
        uniqueExecutor.get());
    for (const aclTensor *tensor : result) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwdFinalize(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkKdaBwdFinalize launch failed");
    return ACLNN_SUCCESS;
}
