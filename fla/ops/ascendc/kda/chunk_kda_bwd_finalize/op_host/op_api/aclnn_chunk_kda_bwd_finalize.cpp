#include "aclnn_chunk_kda_bwd_finalize.h"
#include "chunk_kda_bwd_finalize.h"

#include <cmath>
#include <cstring>
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
    bool useBetaSigmoidInKernel, bool allowNegEigval, const char *outputLayout,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dBetaOut, const aclTensor *dGOut,
    const aclTensor *dALogOut, const aclTensor *dDtBiasOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "workspaceSize/executor must not be nullptr");
    CHECK_COND(!allowNegEigval || useBetaSigmoidInKernel, ACLNN_ERR_PARAM_INVALID,
               "allowNegEigval=true requires useBetaSigmoidInKernel=true");
#ifndef KDA_BWD_STAGES_ARCH22
    CHECK_COND(!useBetaSigmoidInKernel, ACLNN_ERR_PARAM_INVALID,
               "A5 finalize: use_beta_sigmoid_in_kernel=true is reserved but not supported");
#endif
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
    CHECK_COND((cuSeqlensOptional != nullptr || chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "chunk_indices requires cu_seqlens");
    const auto rank = q->GetViewShape().GetDimNum();
    CHECK_COND(rank == 3 || rank == 4, ACLNN_ERR_PARAM_INVALID, "q must have rank 3 or 4");
    const bool rank3 = rank == 3;
    CHECK_COND(outputLayout != nullptr &&
                   (std::strcmp(outputLayout, rank3 ? "NTD" : "BNSD") == 0 ||
                    std::strcmp(outputLayout, rank3 ? "TND" : "BSND") == 0),
               ACLNN_ERR_PARAM_INVALID, "output_layout does not match input rank");
#ifdef KDA_BWD_STAGES_ARCH22
    CHECK_COND(aLog->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "A2/A3 requires FP32 a_log");
#endif
    CHECK_COND(!useBetaSigmoidInKernel || beta->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "beta sigmoid backward requires saved FP32 beta_eff");
    CHECK_COND(std::isfinite(static_cast<float>(scale)) && std::isfinite(static_cast<float>(lowerBound)) &&
                   lowerBound >= -5.0 && static_cast<float>(lowerBound) < 0.0f,
               ACLNN_ERR_PARAM_INVALID, "scale/lower_bound must be finite");
    CHECK_COND(chunkSize == 64 && safeGate && useGateInKernel && useExp2 && !stateVFirst,
               ACLNN_ERR_PARAM_INVALID, "public finalize requires C=64, safe_gate/use_gate/use_exp2=true and K-first state");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto result = l0op::ChunkKdaBwdFinalize(
        q, k, v, gk, rawG, beta, aLog, dtBias, akk, vNew, h, dh,
        dvScan, dAqk, dqRaw, qRstdOptional, kRstdOptional,
        cuSeqlensOptional, chunkIndicesOptional,
        scale, lowerBound, chunkSize, safeGate, useGateInKernel, useExp2, stateVFirst,
        useBetaSigmoidInKernel, allowNegEigval, outputLayout,
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
