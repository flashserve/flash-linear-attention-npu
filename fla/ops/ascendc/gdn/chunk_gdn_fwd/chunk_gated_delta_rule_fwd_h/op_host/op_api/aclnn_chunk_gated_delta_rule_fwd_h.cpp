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
#include <dlfcn.h>
#include <new>

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
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
    bool saveNewValue = true;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    bool useExp2 = false;
    bool stateVFirst = false;
    const aclTensor *hOut = nullptr;
    const aclTensor *vNewOut = nullptr;
    const aclTensor *finalStateOut = nullptr;
};

static aclnnStatus CheckNotNull(ChunkGatedDeltaRuleFwdHParams params)
{
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.w != nullptr, ACLNN_ERR_PARAM_NULLPTR, "w must not be nullptr.");
    CHECK_COND(params.u != nullptr, ACLNN_ERR_PARAM_NULLPTR, "u must not be nullptr.");

    CHECK_COND(params.hOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "hOut must not be nullptr.");
    CHECK_COND(params.vNewOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "vNewOut must not be nullptr.");
    CHECK_COND(params.finalStateOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "finalStateOut must not be nullptr.");
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
    CHECK_COND(vDim == 128, ACLNN_ERR_PARAM_INVALID,
               "u V dimension must be 128, but got %ld.", vDim);
    const int64_t seqNum = params.cuSeqlensOptional == nullptr
                               ? batch
                               : static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;
    const int64_t numChunks = params.chunkIndicesOptional == nullptr
                                  ? (kShape.GetDim(2) + params.chunkSize - 1) / params.chunkSize
                                  : static_cast<int64_t>(params.chunkIndicesOptional->Size()) / 2;
    auto hShape = params.hOut->GetViewShape();
    CHECK_COND(hShape.GetDimNum() == 5 && hShape.GetDim(0) == batch && hShape.GetDim(1) == hv &&
                   hShape.GetDim(2) == numChunks && hShape.GetDim(3) == kDim && hShape.GetDim(4) == vDim,
               ACLNN_ERR_PARAM_INVALID,
               "hOut must have shape [B, HV, num_chunks, K, V] with num_chunks=%ld.", numChunks);
    auto vNewShape = params.vNewOut->GetViewShape();
    CHECK_COND(vNewShape.GetDimNum() == 4 && vNewShape.GetDim(0) == batch &&
                   vNewShape.GetDim(1) == hv && vNewShape.GetDim(2) == kShape.GetDim(2) &&
                   vNewShape.GetDim(3) == vDim,
               ACLNN_ERR_PARAM_INVALID, "vNewOut must have shape [B, HV, T, V].");
    if (params.initialStateOptional != nullptr) {
        auto stateShape = params.initialStateOptional->GetViewShape();
        CHECK_COND(stateShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID,
                   "initialStateOptional must be rank 4.");
        CHECK_COND(stateShape.GetDim(0) == seqNum && stateShape.GetDim(1) == hv &&
                       stateShape.GetDim(2) == kDim && stateShape.GetDim(3) == vDim,
                   ACLNN_ERR_PARAM_INVALID,
                   "initialStateOptional must be [N, HV, K, V].");
    }
    auto finalStateShape = params.finalStateOut->GetViewShape();
    if (params.outputFinalState) {
        CHECK_COND(finalStateShape.GetDimNum() == 4 && finalStateShape.GetDim(0) == seqNum &&
                       finalStateShape.GetDim(1) == hv && finalStateShape.GetDim(2) == kDim &&
                       finalStateShape.GetDim(3) == vDim,
                   ACLNN_ERR_PARAM_INVALID, "finalStateOut must be [N, HV, K, V].");
    } else {
        CHECK_COND(finalStateShape.GetDimNum() == 1 && finalStateShape.GetDim(0) == 0,
                   ACLNN_ERR_PARAM_INVALID,
                   "finalStateOut must be an empty tensor with shape [0] when outputFinalState is false.");
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
    const auto finalStateDtype = params.finalStateOut->GetDataType();
    CHECK_COND(finalStateDtype == DataType::DT_FLOAT || finalStateDtype == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "finalStateOut dtype must be float32 or bfloat16.");
    if (params.initialStateOptional != nullptr) {
        const auto initialStateDtype = params.initialStateOptional->GetDataType();
        CHECK_COND(initialStateDtype == DataType::DT_FLOAT || initialStateDtype == DataType::DT_BF16,
                   ACLNN_ERR_PARAM_INVALID, "initialStateOptional dtype must be float32 or bfloat16.");
        CHECK_COND(initialStateDtype == finalStateDtype, ACLNN_ERR_PARAM_INVALID,
                   "initialStateOptional and finalStateOut must have the same dtype.");
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

static aclnnStatus CheckGateMode(const ChunkGatedDeltaRuleFwdHParams &params)
{
    const bool hasG = params.gOptional != nullptr;
    const bool hasGk = params.gkOptional != nullptr;
    CHECK_COND(hasG != hasGk, ACLNN_ERR_PARAM_INVALID,
               "Exactly one of g and gk must be provided: g-only selects GDN, while gk-only selects KDA/GDN2; "
               "has_g=%d, has_gk=%d.", hasG, hasGk);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOptions(const ChunkGatedDeltaRuleFwdHParams &params)
{
    CHECK_COND(params.chunkSize == 64, ACLNN_ERR_PARAM_INVALID,
               "chunkSize only supports 64 in the current version, but got %ld.", params.chunkSize);
    const bool hasCuSeqlens = params.cuSeqlensOptional != nullptr;
    const bool hasChunkIndices = params.chunkIndicesOptional != nullptr;
    CHECK_COND(hasCuSeqlens == hasChunkIndices, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional and chunkIndicesOptional must be both provided or both omitted.");
    if (hasCuSeqlens) {
        const auto kShape = params.k->GetViewShape();
        const int64_t batch = kShape.GetDim(0);
        const int64_t seqlen = kShape.GetDim(2);
        CHECK_COND(batch == 1, ACLNN_ERR_PARAM_INVALID,
                   "varlen BNSD input requires B=1, but got B=%ld.", batch);
        CHECK_COND(params.cuSeqlensOptional->Size() >= 2, ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional must contain at least two elements.");
        CHECK_COND((*params.cuSeqlensOptional)[0] == 0, ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional[0] must be 0.");
        CHECK_COND((*params.cuSeqlensOptional)[params.cuSeqlensOptional->Size() - 1] == seqlen,
                   ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional last element must equal T=%ld.", seqlen);
        int64_t totalChunks = 0;
        for (size_t seq = 0; seq + 1 < params.cuSeqlensOptional->Size(); ++seq) {
            const int64_t begin = (*params.cuSeqlensOptional)[seq];
            const int64_t end = (*params.cuSeqlensOptional)[seq + 1];
            CHECK_COND(begin < end, ACLNN_ERR_PARAM_INVALID,
                       "cuSeqlensOptional must be strictly increasing at sequence %zu.", seq);
            totalChunks += (end - begin + params.chunkSize - 1) / params.chunkSize;
        }
        CHECK_COND(params.chunkIndicesOptional->Size() == static_cast<size_t>(totalChunks) * 2,
                   ACLNN_ERR_PARAM_INVALID,
                   "chunkIndicesOptional must contain exactly one pair per chunk.");
        size_t offset = 0;
        for (size_t seq = 0; seq + 1 < params.cuSeqlensOptional->Size(); ++seq) {
            const int64_t length = (*params.cuSeqlensOptional)[seq + 1] - (*params.cuSeqlensOptional)[seq];
            const int64_t chunks = (length + params.chunkSize - 1) / params.chunkSize;
            for (int64_t chunk = 0; chunk < chunks; ++chunk) {
                CHECK_COND((*params.chunkIndicesOptional)[offset] == static_cast<int64_t>(seq) &&
                               (*params.chunkIndicesOptional)[offset + 1] == chunk,
                           ACLNN_ERR_PARAM_INVALID,
                           "chunkIndicesOptional must use canonical sequence-major order.");
                offset += 2;
            }
        }
    }
    CHECK_COND(params.saveNewValue, ACLNN_ERR_PARAM_INVALID,
               "saveNewValue is reserved and only true is supported.");
    CHECK_COND(!params.stateVFirst, ACLNN_ERR_PARAM_INVALID,
               "stateVFirst is reserved and only false is supported by the physical aclnn interface.");
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
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkGatedDeltaRuleFwdHParams params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGateMode(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptions(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
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
    bool saveNewValue,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool useExp2,
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
                                         saveNewValue,
                                         cuSeqlensOptional,
                                         chunkIndicesOptional,
                                         useExp2,
                                         stateVFirst,
                                         hOut,
                                         vNewOut,
                                         finalStateOut};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleFwdH,
                   DFX_IN(k, w, u, gOptional, gkOptional, initialStateOptional, cuSeqlensOptional,
                          chunkIndicesOptional, outputFinalState, chunkSize, saveNewValue, useExp2,
                          stateVFirst),
                   DFX_OUT(hOut, vNewOut, finalStateOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
    auto result = l0op::ChunkGatedDeltaRuleFwdH(
        params.k, params.w, params.u, params.gOptional, params.gkOptional, params.initialStateOptional,
        params.cuSeqlensOptional, params.chunkIndicesOptional, params.outputFinalState, params.chunkSize,
        params.useExp2, params.hOut, params.vNewOut, params.finalStateOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto viewCopyResult0 = l0op::ViewCopy(result[0], params.hOut, executorPtr);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(result[1], params.vNewOut, executorPtr);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (outputFinalState) {
        auto viewCopyResult2 = l0op::ViewCopy(result[2], params.finalStateOut, executorPtr);
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
