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

#include "../../../../common/host_utils/log.h"

using namespace op;

namespace {

constexpr const char *CHUNK_KDA_BWD_FINALIZE_ENTITY = "aclnnChunkKdaBwdFinalize";

aclnnStatus CheckRequired(const aclTensor *tensor, DataType dtype, const char *name)
{
    if (tensor == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, name);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!IsContiguous(tensor)) {
        OP_LOGE_LIBOPAPI_REPORT_FORMAT_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, name, "non-contiguous", "contiguous");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (tensor->GetDataType() != dtype) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, name,
                                                   fla_npu::host_utils::DfxDtypeString(tensor->GetDataType()),
                                                   fla_npu::host_utils::DfxDtypeString(dtype));
        return ACLNN_ERR_PARAM_INVALID;
    }
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
    if (workspaceSize == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (allowNegEigval && !useBetaSigmoidInKernel) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "allowNegEigval", "true",
                                                   "requires useBetaSigmoidInKernel=true");
        return ACLNN_ERR_PARAM_INVALID;
    }
#ifndef KDA_BWD_STAGES_ARCH22
    if (useBetaSigmoidInKernel) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "useBetaSigmoidInKernel", "true",
                                                   "reserved but not supported on A5");
        return ACLNN_ERR_PARAM_INVALID;
    }
#endif
    aclnnStatus ret;
    for (const auto &item : {
             std::pair<const aclTensor *, const char *>(q, "q"),
             {k, "k"}, {v, "v"}, {akk, "akk"},
             {vNew, "v_new"}, {h, "h"}, {dh, "dh"}, {dvScan, "dv_scan"},
             {dqOut, "dq"}, {dkOut, "dk"}, {dvOut, "dv"}}) {
        ret = CheckRequired(item.first, DataType::DT_BF16, item.second);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    }
    if (beta == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "beta");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (beta->GetDataType() != DataType::DT_BF16 && beta->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "beta",
                                                   fla_npu::host_utils::DfxDtypeString(beta->GetDataType()), "BF16,FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    ret = CheckRequired(beta, beta->GetDataType(), "beta");
    if (ret != ACLNN_SUCCESS) return ret;
    ret = CheckRequired(dBetaOut, beta->GetDataType(), "d_beta");
    if (ret != ACLNN_SUCCESS) return ret;
    for (const auto &item : {
             std::pair<const aclTensor *, const char *>(gk, "gk"),

             {rawG, "raw_g"}, {dtBias, "dt_bias"}, {dAqk, "d_aqk"},
             {dqRaw, "dq_raw"}, {dGOut, "d_g"}, {dALogOut, "d_a_log"},
             {dDtBiasOut, "d_dt_bias"}}) {
        ret = CheckRequired(item.first, DataType::DT_FLOAT, item.second);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    }
    if (aLog != nullptr && !IsContiguous(aLog)) {
        OP_LOGE_LIBOPAPI_REPORT_FORMAT_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "a_log", "non-contiguous", "contiguous");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aLog != nullptr && aLog->GetDataType() != DataType::DT_FLOAT && aLog->GetDataType() != DataType::DT_BF16) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "a_log",
                                                   fla_npu::host_utils::DfxDtypeString(aLog->GetDataType()), "FP32,BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((qRstdOptional == nullptr) != (kRstdOptional == nullptr)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUES_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "qRstdOptional,kRstdOptional",
                                                    "not together", "must appear together");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qRstdOptional != nullptr) {
        ret = CheckRequired(qRstdOptional, DataType::DT_FLOAT, "q_rstd");
        if (ret != ACLNN_SUCCESS) return ret;
        ret = CheckRequired(kRstdOptional, DataType::DT_FLOAT, "k_rstd");
        if (ret != ACLNN_SUCCESS) return ret;
    }
    if (cuSeqlensOptional == nullptr && chunkIndicesOptional != nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "chunkIndicesOptional", "present",
                                                   "requires cuSeqlensOptional");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto rank = q->GetViewShape().GetDimNum();
    if (rank != 3 && rank != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "q",
                                                             "rank=" + std::to_string(rank), "rank 3 or 4");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool rank3 = rank == 3;

    if (outputLayout == nullptr ||
        (std::strcmp(outputLayout, rank3 ? "NTD" : "BNSD") != 0 &&
         std::strcmp(outputLayout, rank3 ? "TND" : "BSND") != 0)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "outputLayout",
                                                   outputLayout == nullptr ? "null" : outputLayout, rank3 ? "NTD or TND" : "BNSD or BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
#ifdef KDA_BWD_STAGES_ARCH22
    if (aLog != nullptr && aLog->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "a_log",
                                                   fla_npu::host_utils::DfxDtypeString(aLog->GetDataType()), "FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
#endif
    if (useBetaSigmoidInKernel && beta->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "beta",
                                                   fla_npu::host_utils::DfxDtypeString(beta->GetDataType()), "FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(std::isfinite(static_cast<float>(scale)) && std::isfinite(static_cast<float>(lowerBound)) &&
          lowerBound >= -5.0 && static_cast<float>(lowerBound) < 0.0f)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "lowerBound",
                                                   std::to_string(lowerBound), "finite, in [-5, 0)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(chunkSize == 64 && safeGate && useGateInKernel && useExp2 && !stateVFirst)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_FINALIZE_ENTITY, "chunkSize/safeGate/useGate/useExp2/stateVFirst",
                                                   "not matching", "C=64, safe_gate/use_gate/use_exp2=true, K-first state");
        return ACLNN_ERR_PARAM_INVALID;
    }


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
