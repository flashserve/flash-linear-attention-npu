/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include "aclnn_kda_gate_cumsum.h"
#include "kda_gate_cumsum.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

#include "../../../../common/host_utils/log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t MAX_KDA_K_DIM = 256;
constexpr const char *KDA_GATE_CUMSUM_ENTITY = "aclnnKdaGateCumsum";

aclnnStatus KdaGateDataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    if (tensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Making input contiguous failed in %s.", KDA_GATE_CUMSUM_ENTITY);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

int64_t KdaGateDim(const aclTensor *tensor, size_t idx)
{
    return tensor->GetViewShape().GetDim(idx);
}

size_t KdaGateRank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

bool KdaGateSameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    size_t rank = KdaGateRank(lhs);
    if (rank != KdaGateRank(rhs)) {
        return false;
    }
    for (size_t idx = 0; idx < rank; ++idx) {
        if (KdaGateDim(lhs, idx) != KdaGateDim(rhs, idx)) {
            return false;
        }
    }
    return true;
}

aclnnStatus KdaGateCheckCuSeqlens(const aclIntArray *cuSeqlensOptional, int64_t seqlen)
{
    if (cuSeqlensOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const aclIntArray &cu = *cuSeqlensOptional;
    if (cu.Size() < 2) {
        OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(KDA_GATE_CUMSUM_ENTITY, "cuSeqlensOptional",
                                                      std::to_string(cu.Size()), ">=2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (cu[0] != 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(KDA_GATE_CUMSUM_ENTITY, "cuSeqlensOptional[0]",
                                                   std::to_string(cu[0]), "0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (cu[cu.Size() - 1] != seqlen) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(KDA_GATE_CUMSUM_ENTITY, "cuSeqlensOptional[last]",
                                                   std::to_string(cu[cu.Size() - 1]), std::to_string(seqlen));
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t idx = 0; idx + 1 < cu.Size(); ++idx) {
        if (cu[idx] > cu[idx + 1]) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(KDA_GATE_CUMSUM_ENTITY, "cuSeqlensOptional", "non-monotonic",
                                                       "nondecreasing");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus KdaGateCheckParams(
    const aclTensor *g,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    int64_t chunkSize,
    bool useGateInKernel,
    bool safeGate,
    double lowerBound,
    const aclTensor *gkOut)
{
    if (g == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(KDA_GATE_CUMSUM_ENTITY, "g");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (gkOut == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(KDA_GATE_CUMSUM_ENTITY, "gkOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!(chunkSize == 32 || chunkSize == 64 || chunkSize == 128)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(KDA_GATE_CUMSUM_ENTITY, "chunkSize",
                                                   std::to_string(chunkSize), "32, 64 or 128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t rank = KdaGateRank(g);
    if (!(rank == 3 || rank == 4)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(KDA_GATE_CUMSUM_ENTITY, "g",
                                                             "rank=" + std::to_string(rank),
                                                             "head-major BNSD rank4 or NTD rank3");
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t kDimIdx = rank == 4 ? 3 : 2;
    if (KdaGateDim(g, kDimIdx) > MAX_KDA_K_DIM) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(KDA_GATE_CUMSUM_ENTITY, "g.shape[K]",
                                                   std::to_string(KdaGateDim(g, kDimIdx)), "<=256");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (gkOut->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(KDA_GATE_CUMSUM_ENTITY, "gkOut",
                                                   fla_npu::host_utils::DfxDtypeString(gkOut->GetDataType()), "FLOAT");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!KdaGateSameShape(gkOut, g)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(KDA_GATE_CUMSUM_ENTITY, "gkOut", "mismatch with g", "same as g");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t seqlen = KdaGateDim(g, rank == 4 ? 2 : 1);
    auto ret = KdaGateCheckCuSeqlens(cuSeqlensOptional, seqlen);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    if (!(cuSeqlensOptional == nullptr || rank == 3 || KdaGateDim(g, 0) == 1)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(KDA_GATE_CUMSUM_ENTITY, "g.shape[B]",
                                                   std::to_string(KdaGateDim(g, 0)),
                                                   "1 (rank4 varlen with cuSeqlens)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t hv = KdaGateDim(g, rank == 4 ? 1 : 0);
    if (useGateInKernel) {
        if (aLogOptional == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(KDA_GATE_CUMSUM_ENTITY, "aLogOptional");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (aLogOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(KDA_GATE_CUMSUM_ENTITY, "aLogOptional",
                                                       fla_npu::host_utils::DfxDtypeString(aLogOptional->GetDataType()), "FLOAT");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (safeGate) {
            if (!(lowerBound >= -5.0 && lowerBound < 0.0)) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(KDA_GATE_CUMSUM_ENTITY, "lowerBound",
                                                           std::to_string(lowerBound), "in [-5, 0)");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        int64_t k = KdaGateDim(g, kDimIdx);
        if (!(KdaGateRank(aLogOptional) == 1 && KdaGateDim(aLogOptional, 0) == hv)) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(KDA_GATE_CUMSUM_ENTITY, "aLogOptional",
                                                       "rank=" + std::to_string(KdaGateRank(aLogOptional)),
                                                       "[HV=" + std::to_string(hv) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (dtBiasOptional != nullptr) {
            if (!(dtBiasOptional->GetDataType() == DataType::DT_FLOAT &&
                  KdaGateRank(dtBiasOptional) == 1 &&
                  KdaGateDim(dtBiasOptional, 0) == hv * k)) {
                OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(KDA_GATE_CUMSUM_ENTITY, "dtBiasOptional",
                                                           "rank=" + std::to_string(KdaGateRank(dtBiasOptional)) + ",dim0=" +
                                                               std::to_string(KdaGateRank(dtBiasOptional) == 1 ? KdaGateDim(dtBiasOptional, 0) : -1),
                                                           "[HV*K=" + std::to_string(hv * k) + "]");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnKdaGateCumsumGetWorkspaceSize(
    const aclTensor *g,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    int64_t chunkSize,
    bool useGateInKernel,
    bool safeGate,
    double lowerBound,
    const aclTensor *gkOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    if (workspaceSize == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(KDA_GATE_CUMSUM_ENTITY, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(KDA_GATE_CUMSUM_ENTITY, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    L2_DFX_PHASE_1(
        aclnnKdaGateCumsum,
        DFX_IN(g, aLogOptional, dtBiasOptional, cuSeqlensOptional, chunkSize,
               useGateInKernel, safeGate, lowerBound),
        DFX_OUT(gkOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    OP_LOGW("npu_kda_gate_cumsum",
            "npu_kda_gate_cumsum is deprecated and will be removed on "
            "January 15, 2027 (v26.12.0); use the fused npu_chunk_kda_fwd instead.");

    auto ret = KdaGateCheckParams(g, aLogOptional, dtBiasOptional, cuSeqlensOptional, chunkSize, useGateInKernel,
                                  safeGate, lowerBound, gkOut);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    ret = KdaGateDataContiguous(g, executorPtr);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    ret = KdaGateDataContiguous(aLogOptional, executorPtr);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    ret = KdaGateDataContiguous(dtBiasOptional, executorPtr);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    auto result = l0op::KdaGateCumsum(g, aLogOptional, dtBiasOptional, cuSeqlensOptional, chunkSize,
                                      useGateInKernel, safeGate, lowerBound, gkOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKdaGateCumsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnKdaGateCumsum);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
