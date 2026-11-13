#include "op_common/log/log.h"
#include "graph/utils/type_utils.h"
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

static constexpr const char *OP_NAME = "aclnnChunkKdaBwdFinalize";

aclnnStatus CheckRequired(const aclTensor *tensor, DataType dtype, const char *name)
{
    if (tensor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, name, "required tensor must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!IsContiguous(tensor)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, name, "tensor must be contiguous");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (tensor->GetDataType() != dtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, name, ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType()),
            ge::TypeUtils::DataTypeToSerialString(dtype));
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
    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "workspaceSize,executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (allowNegEigval && !useBetaSigmoidInKernel) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "allowNegEigval", "true",
            "allowNegEigval=true requires useBetaSigmoidInKernel=true");
        return ACLNN_ERR_PARAM_INVALID;
    }
#ifndef KDA_BWD_STAGES_ARCH22
    if (useBetaSigmoidInKernel) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "useBetaSigmoidInKernel", "true",
            "A5 finalize: useBetaSigmoidInKernel=true is reserved but not supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
#endif
    for (const auto &item : {
             std::pair<const aclTensor *, const char *>(q, "q"),
             {k, "k"}, {v, "v"}, {akk, "akk"},
             {vNew, "v_new"}, {h, "h"}, {dh, "dh"}, {dvScan, "dv_scan"},
             {dqOut, "dq"}, {dkOut, "dk"}, {dvOut, "dv"}}) {
        {
            const aclnnStatus checkStatus = CheckRequired(item.first, DataType::DT_BF16, item.second);
            if (checkStatus != ACLNN_SUCCESS) {
                return checkStatus;
            }
        };
    }
    if (beta == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "beta");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (beta->GetDataType() != DataType::DT_BF16 && beta->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "beta", ge::TypeUtils::DataTypeToSerialString(beta->GetDataType()),
            "beta must be BF16 or FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    {
        const aclnnStatus checkStatus = CheckRequired(beta, beta->GetDataType(), "beta");
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    {
        const aclnnStatus checkStatus = CheckRequired(dBetaOut, beta->GetDataType(), "d_beta");
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    for (const auto &item : {
             std::pair<const aclTensor *, const char *>(gk, "gk"),
             {dAqk, "d_aqk"}, {dqRaw, "dq_raw"}, {aLog, "a_log"},
             {dtBias, "dt_bias"}, {dALogOut, "d_a_log"}, {dDtBiasOut, "d_dt_bias"}}) {
        {
            const aclnnStatus checkStatus = CheckRequired(item.first, DataType::DT_FLOAT, item.second);
            if (checkStatus != ACLNN_SUCCESS) {
                return checkStatus;
            }
        };
    }
    for (const auto &item : {
             std::pair<const aclTensor *, const aclTensor *>(rawG, dGOut)}) {
        if (item.first == nullptr ||
            (item.first->GetDataType() != DataType::DT_FLOAT &&
             item.first->GetDataType() != DataType::DT_BF16)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "rawG",
                item.first == nullptr ? "nullptr" : ge::TypeUtils::DataTypeToSerialString(item.first->GetDataType()),
                "gate parameters must be BF16 or FP32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        {
            const aclnnStatus checkStatus = CheckRequired(item.first, item.first->GetDataType(), "gate parameter");
            if (checkStatus != ACLNN_SUCCESS) {
                return checkStatus;
            }
        };
        {
            const aclnnStatus checkStatus = CheckRequired(item.second, item.first->GetDataType(), "gate gradient");
            if (checkStatus != ACLNN_SUCCESS) {
                return checkStatus;
            }
        };
    }
    if ((qRstdOptional == nullptr) != (kRstdOptional == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "q_rstd,k_rstd",
            std::string(qRstdOptional == nullptr ? "q_rstd=absent" : "q_rstd=present") + "," +
                (kRstdOptional == nullptr ? "k_rstd=absent" : "k_rstd=present"),
            "q_rstd and k_rstd must appear together");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qRstdOptional != nullptr) {
        {
            const aclnnStatus checkStatus = CheckRequired(qRstdOptional, DataType::DT_FLOAT, "q_rstd");
            if (checkStatus != ACLNN_SUCCESS) {
                return checkStatus;
            }
        };
        {
            const aclnnStatus checkStatus = CheckRequired(kRstdOptional, DataType::DT_FLOAT, "k_rstd");
            if (checkStatus != ACLNN_SUCCESS) {
                return checkStatus;
            }
        };
    }
    if (cuSeqlensOptional == nullptr && chunkIndicesOptional != nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "chunkIndicesOptional", "chunkIndicesOptional requires cuSeqlensOptional");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto rank = q->GetViewShape().GetDimNum();
    if (rank != 3 && rank != 4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "q", std::to_string(rank), "q must have rank 3 or 4");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool rank3 = rank == 3;
    if (outputLayout == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "outputLayout");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (std::strcmp(outputLayout, rank3 ? "NTD" : "BNSD") != 0 &&
        std::strcmp(outputLayout, rank3 ? "TND" : "BSND") != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "outputLayout", outputLayout,
            rank3 ? "outputLayout must be NTD or TND for rank-3 q"
                  : "outputLayout must be BNSD or BSND for rank-4 q");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (useBetaSigmoidInKernel && beta->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "beta", ge::TypeUtils::DataTypeToSerialString(beta->GetDataType()),
            "beta sigmoid backward requires saved FP32 beta_eff");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(std::isfinite(static_cast<float>(scale)) && std::isfinite(static_cast<float>(lowerBound)) &&
          lowerBound >= -5.0 && static_cast<float>(lowerBound) < 0.0f)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "scale,lowerBound",
            std::to_string(scale) + "," + std::to_string(lowerBound),
            "scale must be finite and lowerBound must be in [-5,0)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(chunkSize == 64 && safeGate && useGateInKernel && useExp2 && !stateVFirst)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "chunkSize,safeGate,useGateInKernel,useExp2,stateVFirst",
            std::to_string(chunkSize) + "," + (safeGate ? "true" : "false") + "," +
                (useGateInKernel ? "true" : "false") + "," + (useExp2 ? "true" : "false") + "," +
                (stateVFirst ? "true" : "false"),
            "public finalize requires C=64, safeGate/useGateInKernel/useExp2=true and stateVFirst=false");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME,
                           "%s: CREATE_EXECUTOR returned nullptr; propagate the resource first error.", __func__);
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto result = l0op::ChunkKdaBwdFinalize(
        q, k, v, gk, rawG, beta, aLog, dtBias, akk, vNew, h, dh,
        dvScan, dAqk, dqRaw, qRstdOptional, kRstdOptional,
        cuSeqlensOptional, chunkIndicesOptional,
        scale, lowerBound, chunkSize, safeGate, useGateInKernel, useExp2, stateVFirst,
        useBetaSigmoidInKernel, allowNegEigval, outputLayout,
        dqOut, dkOut, dvOut, dBetaOut, dGOut, dALogOut, dDtBiasOut,
        uniqueExecutor.get());
    for (const aclTensor *tensor : result) {
        if (!(tensor != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: tensor != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwdFinalize(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (!(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "ChunkKdaBwdFinalize launch failed", __func__);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}
