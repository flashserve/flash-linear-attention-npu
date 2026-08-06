/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_chunk_gated_delta_rule_fwd_h.h"
#include "chunk_gated_delta_rule_fwd_h.h"
#include <algorithm>
#include <dlfcn.h>
#include <new>
#include <vector>

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"


using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

struct ChunkGatedDeltaRuleFwdHParams {
    const aclTensor *k = nullptr;
    const aclTensor *w = nullptr;
    const aclTensor *u = nullptr;
    const aclTensor *gOptional = nullptr;
    const aclTensor *gkOptional = nullptr;
    const aclTensor *initialStateOptional = nullptr;
    bool outputFinalState = false;
    int64_t chunkSize = 64;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    bool stateVFirst = false;
    const aclTensor *hOut = nullptr;
    const aclTensor *vNewOut = nullptr;
    const aclTensor *finalStateOut = nullptr;
};

static op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

static op::Shape SwapLastTwo(const op::Shape &input)
{
    op::Shape output;
    const size_t rank = input.GetDimNum();
    for (size_t idx = 0; idx < rank; ++idx) {
        if (idx + 2 == rank) {
            output.AppendDim(input.GetDim(rank - 1));
        } else if (idx + 1 == rank) {
            output.AppendDim(input.GetDim(rank - 2));
        } else {
            output.AppendDim(input.GetDim(idx));
        }
    }
    return output;
}

static const aclTensor *TransposeLastTwo(const aclTensor *input, aclOpExecutor *executor)
{
    const size_t rank = input->GetViewShape().GetDimNum();
    std::vector<int64_t> perm(rank);
    for (size_t idx = 0; idx < rank; ++idx) {
        perm[idx] = static_cast<int64_t>(idx);
    }
    std::swap(perm[rank - 2], perm[rank - 1]);
    const aclIntArray *permArray = executor->AllocIntArray(perm.data(), perm.size());
    CHECK_RET(permArray != nullptr, nullptr);
    return l0op::Transpose(input, permArray, executor);
}

static aclnnStatus CheckNotNull(ChunkGatedDeltaRuleFwdHParams params)
{
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.w != nullptr, ACLNN_ERR_PARAM_NULLPTR, "w must not be nullptr.");
    CHECK_COND(params.u != nullptr, ACLNN_ERR_PARAM_NULLPTR, "u must not be nullptr.");

    CHECK_COND(params.hOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "hOut must not be nullptr.");
    CHECK_COND(params.vNewOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "vNewOut must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(ChunkGatedDeltaRuleFwdHParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(ChunkGatedDeltaRuleFwdHParams params)
{
    auto kShape = params.k->GetViewShape();
    auto wShape = params.w->GetViewShape();
    auto uShape = params.u->GetViewShape();
    CHECK_COND(kShape.GetDimNum() == 4 && wShape.GetDimNum() == 4 && uShape.GetDimNum() == 4,
               ACLNN_ERR_PARAM_INVALID, "k, w and u must be rank-4 BNSD tensors.");
    CHECK_COND(kShape.GetDim(0) == wShape.GetDim(0) && kShape.GetDim(0) == uShape.GetDim(0) &&
                   wShape.GetDim(1) == uShape.GetDim(1) && kShape.GetDim(2) == wShape.GetDim(2) &&
                   kShape.GetDim(2) == uShape.GetDim(2) && kShape.GetDim(3) == wShape.GetDim(3),
               ACLNN_ERR_PARAM_INVALID,
               "k, w and u must match in B/T, w and u must match in HV, and k and w must match in K.");
    CHECK_COND(uShape.GetDim(1) >= kShape.GetDim(1) && uShape.GetDim(1) % kShape.GetDim(1) == 0,
               ACLNN_ERR_PARAM_INVALID, "u HV must be greater than or equal to k H and divisible by H.");
    if (params.gOptional != nullptr) {
        auto gShape = params.gOptional->GetViewShape();
        CHECK_COND(gShape.GetDimNum() == 3 && gShape.GetDim(0) == uShape.GetDim(0) &&
                       gShape.GetDim(1) == uShape.GetDim(1) && gShape.GetDim(2) == uShape.GetDim(2),
                   ACLNN_ERR_PARAM_INVALID, "g must have shape [B, HV, T].");
    }
    const int64_t batch = kShape.GetDim(0);
    const int64_t hv = uShape.GetDim(1);
    const int64_t kDim = kShape.GetDim(3);
    const int64_t vDim = uShape.GetDim(3);
    const int64_t seqNum = params.cuSeqlensOptional == nullptr
                               ? batch
                               : static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;
    auto hShape = params.hOut->GetViewShape();
    CHECK_COND(hShape.GetDimNum() == 5 && hShape.GetDim(0) == batch && hShape.GetDim(1) == hv,
               ACLNN_ERR_PARAM_INVALID, "hOut must have prefix [B, HV, num_chunks].");
    const int64_t hK = params.stateVFirst ? hShape.GetDim(4) : hShape.GetDim(3);
    const int64_t hV = params.stateVFirst ? hShape.GetDim(3) : hShape.GetDim(4);
    CHECK_COND(hK == kDim && hV == vDim, ACLNN_ERR_PARAM_INVALID,
               "hOut state dimensions must be [K, V] when stateVFirst=false and [V, K] otherwise.");
    auto vNewShape = params.vNewOut->GetViewShape();
    CHECK_COND(vNewShape.GetDimNum() == 4 && vNewShape.GetDim(0) == batch &&
                   vNewShape.GetDim(1) == hv && vNewShape.GetDim(2) == kShape.GetDim(2) &&
                   vNewShape.GetDim(3) == vDim,
               ACLNN_ERR_PARAM_INVALID, "vNewOut must have shape [B, HV, T, V].");
    const aclTensor *states[] = {params.initialStateOptional, params.finalStateOut};
    const char *stateNames[] = {"initialStateOptional", "finalStateOut"};
    for (size_t idx = 0; idx < 2; ++idx) {
        if (states[idx] == nullptr) {
            continue;
        }
        auto stateShape = states[idx]->GetViewShape();
        CHECK_COND(stateShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID,
                   "%s must be rank 4.", stateNames[idx]);
        const int64_t stateK = params.stateVFirst ? stateShape.GetDim(3) : stateShape.GetDim(2);
        const int64_t stateV = params.stateVFirst ? stateShape.GetDim(2) : stateShape.GetDim(3);
        CHECK_COND(stateShape.GetDim(0) == seqNum &&
                       stateShape.GetDim(1) == hv && stateK == kDim && stateV == vDim,
                   ACLNN_ERR_PARAM_INVALID,
                   "%s must be [N, HV, K, V] when stateVFirst=false and [N, HV, V, K] otherwise.",
                   stateNames[idx]);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkGatedDeltaRuleFwdHParams params)
{
    auto inputDtype = params.k->GetDataType();
    CHECK_COND(inputDtype == DataType::DT_FLOAT16 || inputDtype == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "k dtype must be float16 or bfloat16.");
    CHECK_COND(params.w->GetDataType() == inputDtype && params.u->GetDataType() == inputDtype,
               ACLNN_ERR_PARAM_INVALID, "k, w and u must have the same dtype.");
    CHECK_COND(params.hOut->GetDataType() == inputDtype && params.vNewOut->GetDataType() == inputDtype,
               ACLNN_ERR_PARAM_INVALID, "hOut and vNewOut dtype must match k, w and u.");
    auto gateDtype = params.gOptional != nullptr ? params.gOptional->GetDataType() : params.gkOptional->GetDataType();
    CHECK_COND(gateDtype == DataType::DT_FLOAT || gateDtype == inputDtype,
               ACLNN_ERR_PARAM_INVALID, "g/gk dtype must be float32 or match k dtype.");
    if (params.gOptional != nullptr && params.gkOptional != nullptr) {
        CHECK_COND(params.gOptional->GetDataType() == params.gkOptional->GetDataType(),
                   ACLNN_ERR_PARAM_INVALID, "g and gk must have the same dtype when both are provided.");
    }
    if (params.outputFinalState) {
        CHECK_COND(params.finalStateOut != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "finalStateOut must be provided when outputFinalState is true.");
        auto stateDtype = params.initialStateOptional != nullptr ? params.initialStateOptional->GetDataType()
                                                                : DataType::DT_FLOAT;
        CHECK_COND(params.finalStateOut->GetDataType() == stateDtype, ACLNN_ERR_PARAM_INVALID,
                   "finalStateOut dtype must match initial state, or be float32 when initial state is absent.");
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(ChunkGatedDeltaRuleFwdHParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous k failed.");
    CHECK_COND(DataContiguous(params.w, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous w failed.");
    CHECK_COND(DataContiguous(params.u, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous u failed.");
    if (params.gOptional != nullptr) {
        CHECK_COND(DataContiguous(params.gOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "Contiguous gOptional failed.");
    }
    if (params.gkOptional != nullptr) {
        CHECK_COND(DataContiguous(params.gkOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "Contiguous gkOptional failed.");
    }
    if (params.initialStateOptional != nullptr) {
        CHECK_COND(DataContiguous(params.initialStateOptional, executorPtr) == ACLNN_SUCCESS,
                   ACLNN_ERR_PARAM_INVALID, "Contiguous initialStateOptional failed.");
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckGateOptionalNonNull(const ChunkGatedDeltaRuleFwdHParams &params)
{
    CHECK_COND(params.gOptional != nullptr || params.gkOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
               "Either g or gk must be provided.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckGkParams(const ChunkGatedDeltaRuleFwdHParams &params)
{
    if (params.gkOptional != nullptr) {
        auto gkShape = params.gkOptional->GetViewShape();
        CHECK_COND(gkShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID,
                   "gk must have rank 4 when provided, got rank %ld.", gkShape.GetDimNum());
        CHECK_COND(gkShape.GetDim(3) == params.k->GetViewShape().GetDim(3), ACLNN_ERR_PARAM_INVALID,
                   "gk.shape[3] (K) must match k.shape[3] (K).");
        CHECK_COND(gkShape.GetDim(2) == params.k->GetViewShape().GetDim(2), ACLNN_ERR_PARAM_INVALID,
                   "gk.shape[2] (T) must match k.shape[2] (T).");
        CHECK_COND(gkShape.GetDim(1) == params.u->GetViewShape().GetDim(1), ACLNN_ERR_PARAM_INVALID,
                   "gk.shape[1] (HV) must match u.shape[1] (HV).");
        CHECK_COND(gkShape.GetDim(0) == params.k->GetViewShape().GetDim(0), ACLNN_ERR_PARAM_INVALID,
                   "gk.shape[0] (B) must match k.shape[0] (B).");
        if (params.gOptional != nullptr) {
            CHECK_COND(params.gkOptional->GetDataType() == params.gOptional->GetDataType(), ACLNN_ERR_PARAM_INVALID,
                       "gk.dtype must match g.dtype when both are provided.");
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkGatedDeltaRuleFwdHParams params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGateOptionalNonNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGkParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    bool outputFinalState,
    int64_t chunkSize,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool stateVFirst,
    const aclTensor *hOut,
    const aclTensor *vNewOut,
    const aclTensor *finalStateOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkGatedDeltaRuleFwdHParams params{k,
                                         w,
                                         u,
                                         gOptional,
                                         gkOptional,
                                         initialStateOptional,
                                         outputFinalState,
                                         chunkSize,
                                         cuSeqlensOptional,
                                         chunkIndicesOptional,
                                         stateVFirst,
                                         hOut,
                                         vNewOut,
                                         finalStateOut};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleFwdH,
                   DFX_IN(k, w, u, gOptional, gkOptional, initialStateOptional, cuSeqlensOptional,
                          chunkIndicesOptional, outputFinalState, chunkSize, stateVFirst),
                   DFX_OUT(hOut, vNewOut, finalStateOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
    const auto kShape = params.k->GetViewShape();
    const auto uShape = params.u->GetViewShape();
    const int64_t seqNum = params.cuSeqlensOptional == nullptr
                               ? kShape.GetDim(0)
                               : static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;
    const aclTensor *initialStateCompute = params.initialStateOptional;
    const aclTensor *hCompute = params.hOut;
    const aclTensor *finalStateCompute = params.finalStateOut;
    if (params.stateVFirst && initialStateCompute != nullptr) {
        initialStateCompute = TransposeLastTwo(initialStateCompute, executorPtr);
        CHECK_RET(initialStateCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (params.stateVFirst) {
        hCompute = executorPtr->AllocTensor(
            SwapLastTwo(params.hOut->GetViewShape()), params.hOut->GetDataType(), Format::FORMAT_ND);
        CHECK_RET(hCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (!params.outputFinalState) {
        const DataType stateType = initialStateCompute == nullptr
                                       ? DataType::DT_FLOAT
                                       : initialStateCompute->GetDataType();
        finalStateCompute = executorPtr->AllocTensor(
            MakeShape({seqNum, uShape.GetDim(1), kShape.GetDim(3), uShape.GetDim(3)}),
            stateType, Format::FORMAT_ND);
    } else if (params.stateVFirst) {
        finalStateCompute = executorPtr->AllocTensor(
            SwapLastTwo(params.finalStateOut->GetViewShape()),
            params.finalStateOut->GetDataType(), Format::FORMAT_ND);
    }
    CHECK_RET(finalStateCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::ChunkGatedDeltaRuleFwdH(
        params.k, params.w, params.u, params.gOptional, params.gkOptional, initialStateCompute,
        params.cuSeqlensOptional, params.chunkIndicesOptional, params.outputFinalState, params.chunkSize,
        hCompute, params.vNewOut, finalStateCompute, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    const aclTensor *hResult = result[0];
    if (params.stateVFirst) {
        hResult = TransposeLastTwo(hResult, executorPtr);
        CHECK_RET(hResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto viewCopyResult0 = l0op::ViewCopy(hResult, params.hOut, executorPtr);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(result[1], params.vNewOut, executorPtr);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (outputFinalState && params.finalStateOut != nullptr) {
        const aclTensor *finalStateResult = result[2];
        if (params.stateVFirst) {
            finalStateResult = TransposeLastTwo(finalStateResult, executorPtr);
            CHECK_RET(finalStateResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        auto viewCopyResult2 = l0op::ViewCopy(finalStateResult, params.finalStateOut, executorPtr);
        CHECK_RET(viewCopyResult2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnChunkGatedDeltaRuleFwdH(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleFwdH);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in ChunkGatedDeltaRuleFwdH launch aicore.");
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
