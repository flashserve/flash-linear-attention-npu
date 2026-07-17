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
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
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

namespace l0op {
const aclTensor *ZerosLike(const aclTensor *self, aclOpExecutor *executor);
}

static constexpr int64_t FWD_H_CHUNK_SIZE_64 = 64;
static constexpr int64_t FWD_H_CHUNK_SIZE_128 = 128;
static constexpr int64_t FWD_H_K_DIM = 128;
static constexpr int64_t FWD_H_V_DIM_128 = 128;
static constexpr int64_t FWD_H_V_DIM_256 = 256;

#ifdef __cplusplus
extern "C" {
#endif

struct ChunkGatedDeltaRuleFwdHParams {
    const aclTensor *k = nullptr;
    const aclTensor *w = nullptr;
    const aclTensor *u = nullptr;
    const aclTensor *gOptional = nullptr;
    const aclTensor *gkOptional = nullptr;
    const aclTensor *initalStateOptional = nullptr;
    bool outputFinalState = false;
    int64_t chunkSize = 64;
    bool saveNewValue = true;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    bool useExp2 = false;
    bool transposeStateLayout = false;
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
    CHECK_COND(kShape.GetDim(0) > 0 && kShape.GetDim(1) > 0 && kShape.GetDim(2) > 0 &&
                   kShape.GetDim(3) > 0 && uShape.GetDim(1) > 0 && uShape.GetDim(3) > 0,
               ACLNN_ERR_PARAM_INVALID, "B, H_k, H_v, T, K and V must be positive.");
    CHECK_COND(uShape.GetDim(1) >= kShape.GetDim(1) && uShape.GetDim(1) % kShape.GetDim(1) == 0,
               ACLNN_ERR_PARAM_INVALID, "u HV must be greater than or equal to k H and divisible by H.");
    CHECK_COND(kShape.GetDim(3) == FWD_H_K_DIM, ACLNN_ERR_PARAM_INVALID,
               "K must be %ld, but got %ld.", FWD_H_K_DIM, kShape.GetDim(3));
    CHECK_COND(uShape.GetDim(3) == FWD_H_V_DIM_128 || uShape.GetDim(3) == FWD_H_V_DIM_256,
               ACLNN_ERR_PARAM_INVALID, "V must be 128 or 256, but got %ld.", uShape.GetDim(3));
    if (params.gOptional != nullptr) {
        auto gShape = params.gOptional->GetViewShape();
        CHECK_COND(gShape.GetDimNum() == 3 && gShape.GetDim(0) == uShape.GetDim(0) &&
                       gShape.GetDim(1) == uShape.GetDim(1) && gShape.GetDim(2) == uShape.GetDim(2),
                   ACLNN_ERR_PARAM_INVALID, "g must have shape [B, HV, T].");
    }
    if (params.initalStateOptional != nullptr) {
        const auto initialShape = params.initalStateOptional->GetViewShape();
        CHECK_COND(initialShape.GetDimNum() == 4 && initialShape.GetDim(1) == uShape.GetDim(1) &&
                       initialShape.GetDim(2) == kShape.GetDim(3) && initialShape.GetDim(3) == uShape.GetDim(3),
                   ACLNN_ERR_PARAM_INVALID, "initial_state must be [N, HV, K, V].");
    }
    const auto hShape = params.hOut->GetViewShape();
    const auto vNewShape = params.vNewOut->GetViewShape();
    CHECK_COND(hShape.GetDimNum() == 5 && hShape.GetDim(0) == kShape.GetDim(0) &&
                   hShape.GetDim(1) == uShape.GetDim(1) && hShape.GetDim(2) > 0 &&
                   hShape.GetDim(3) == kShape.GetDim(3) && hShape.GetDim(4) == uShape.GetDim(3),
               ACLNN_ERR_PARAM_INVALID, "hOut must be [B, HV, NC, K, V] and NC must be positive.");
    CHECK_COND(vNewShape.GetDimNum() == 4 && vNewShape.GetDim(0) == uShape.GetDim(0) &&
                   vNewShape.GetDim(1) == uShape.GetDim(1) && vNewShape.GetDim(2) == uShape.GetDim(2) &&
                   vNewShape.GetDim(3) == uShape.GetDim(3),
               ACLNN_ERR_PARAM_INVALID, "vNewOut must match u shape [B, HV, T, V].");
    if (params.outputFinalState && params.finalStateOut != nullptr) {
        const auto finalShape = params.finalStateOut->GetViewShape();
        CHECK_COND(finalShape.GetDimNum() == 4 && finalShape.GetDim(1) == uShape.GetDim(1) &&
                       finalShape.GetDim(2) == kShape.GetDim(3) && finalShape.GetDim(3) == uShape.GetDim(3),
                   ACLNN_ERR_PARAM_INVALID, "finalStateOut must be [N, HV, K, V].");
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckChunkMetadataAndStateShape(ChunkGatedDeltaRuleFwdHParams params)
{
    const auto kShape = params.k->GetViewShape();
    const auto uShape = params.u->GetViewShape();
    const auto hShape = params.hOut->GetViewShape();
    const int64_t batch = kShape.GetDim(0);
    const int64_t seqlen = kShape.GetDim(2);
    int64_t logicalSequenceCount = batch;
    int64_t expectedChunks = (seqlen + params.chunkSize - 1) / params.chunkSize;

    if (params.cuSeqlensOptional != nullptr) {
        CHECK_COND(batch == 1, ACLNN_ERR_PARAM_INVALID,
                   "B must be 1 when cuSeqlensOptional is provided, but got %ld.", batch);
        const aclIntArray &cu = *params.cuSeqlensOptional;
        const aclIntArray &indices = *params.chunkIndicesOptional;
        CHECK_COND(cu.Size() >= 2, ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional must contain at least [0, total_tokens].");
        CHECK_COND(cu[0] == 0 && cu[cu.Size() - 1] == seqlen, ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional must start at 0 and end at T=%ld.", seqlen);
        CHECK_COND(indices.Size() % 2 == 0, ACLNN_ERR_PARAM_INVALID,
                   "chunkIndicesOptional must contain flattened (seq_id, chunk_id) pairs.");
        logicalSequenceCount = static_cast<int64_t>(cu.Size()) - 1;
        expectedChunks = 0;
        size_t index = 0;
        for (int64_t seq = 0; seq < logicalSequenceCount; ++seq) {
            CHECK_COND(cu[seq] >= 0 && cu[seq] <= cu[seq + 1] && cu[seq + 1] <= seqlen,
                       ACLNN_ERR_PARAM_INVALID,
                       "cuSeqlensOptional must be nondecreasing and within [0,T].");
            const int64_t localChunkCount =
                (cu[seq + 1] - cu[seq] + params.chunkSize - 1) / params.chunkSize;
            for (int64_t localChunk = 0; localChunk < localChunkCount; ++localChunk) {
                CHECK_COND(index + 1 < indices.Size(), ACLNN_ERR_PARAM_INVALID,
                           "chunkIndicesOptional contains fewer pairs than required by cuSeqlensOptional.");
                CHECK_COND(indices[index] == seq && indices[index + 1] == localChunk,
                           ACLNN_ERR_PARAM_INVALID,
                           "chunkIndicesOptional must use canonical sequence-major chunk order.");
                index += 2;
                ++expectedChunks;
            }
        }
        CHECK_COND(index == indices.Size(), ACLNN_ERR_PARAM_INVALID,
                   "chunkIndicesOptional pair count must match cuSeqlensOptional and chunkSize.");
    }

    CHECK_COND(hShape.GetDim(2) == expectedChunks, ACLNN_ERR_PARAM_INVALID,
               "hOut N_c must equal the chunk count derived from T/cuSeqlensOptional and chunkSize; expected %ld, got %ld.",
               expectedChunks, hShape.GetDim(2));
    if (params.initalStateOptional != nullptr) {
        CHECK_COND(params.initalStateOptional->GetViewShape().GetDim(0) == logicalSequenceCount,
                   ACLNN_ERR_PARAM_INVALID,
                   "initial_state N must equal the logical sequence count; expected %ld.", logicalSequenceCount);
    }
    if (params.outputFinalState) {
        CHECK_COND(params.finalStateOut != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "finalStateOut must be provided when outputFinalState is true.");
        CHECK_COND(params.finalStateOut->GetViewShape().GetDim(0) == logicalSequenceCount,
                   ACLNN_ERR_PARAM_INVALID,
                   "finalStateOut N must equal the logical sequence count; expected %ld.", logicalSequenceCount);
    }
    return ACLNN_SUCCESS;
}

static const aclTensor *MakeNeutralGate(const ChunkGatedDeltaRuleFwdHParams &params, aclOpExecutor *executor)
{
    auto gkShape = params.gkOptional->GetViewShape();
    int64_t offsetsData[] = {0, 0, 0, 0};
    int64_t sizesData[] = {gkShape.GetDim(0), gkShape.GetDim(1), gkShape.GetDim(2), 1};
    auto offsets = executor->AllocIntArray(offsetsData, 4);
    auto sizes = executor->AllocIntArray(sizesData, 4);
    if (offsets == nullptr || sizes == nullptr) {
        return nullptr;
    }
    auto gateLane = l0op::Slice(params.gkOptional, offsets, sizes, executor);
    if (gateLane == nullptr) {
        return nullptr;
    }
    gateLane = l0op::Contiguous(gateLane, executor);
    if (gateLane == nullptr) {
        return nullptr;
    }
    op::Shape gateShape;
    gateShape.AppendDim(gkShape.GetDim(0));
    gateShape.AppendDim(gkShape.GetDim(1));
    gateShape.AppendDim(gkShape.GetDim(2));
    gateLane = l0op::Reshape(gateLane, gateShape, executor);
    return gateLane == nullptr ? nullptr : l0op::ZerosLike(gateLane, executor);
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
        auto stateDtype = params.initalStateOptional != nullptr ? params.initalStateOptional->GetDataType()
                                                                : DataType::DT_FLOAT;
        CHECK_COND(params.finalStateOut->GetDataType() == stateDtype, ACLNN_ERR_PARAM_INVALID,
                   "finalStateOut dtype must match initial state, or be float32 when initial state is absent.");
    }
    if (params.initalStateOptional != nullptr) {
        const auto stateDtype = params.initalStateOptional->GetDataType();
        CHECK_COND(stateDtype == DataType::DT_FLOAT || stateDtype == inputDtype, ACLNN_ERR_PARAM_INVALID,
                   "initial_state dtype must be float32 or match k dtype.");
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
    if (params.initalStateOptional != nullptr) {
        CHECK_COND(DataContiguous(params.initalStateOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "Contiguous initalStateOptional failed.");
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckGateOptionalNonNull(const ChunkGatedDeltaRuleFwdHParams &params)
{
    CHECK_COND(params.gOptional != nullptr || params.gkOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
               "Either g or gk must be provided.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckReservedOptions(const ChunkGatedDeltaRuleFwdHParams &params)
{
    CHECK_COND(params.saveNewValue, ACLNN_ERR_PARAM_INVALID,
               "save_new_value is reserved and only true is supported.");
    CHECK_COND(!params.useExp2, ACLNN_ERR_PARAM_INVALID,
               "use_exp2 is reserved and only false is supported.");
    CHECK_COND(!params.transposeStateLayout, ACLNN_ERR_PARAM_INVALID,
               "transpose_state_layout is reserved and only false is supported.");
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
    CHECK_COND((params.cuSeqlensOptional == nullptr) == (params.chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional and chunkIndicesOptional must both be provided or both be nullptr.");
    CHECK_COND(params.chunkSize == FWD_H_CHUNK_SIZE_64 || params.chunkSize == FWD_H_CHUNK_SIZE_128,
               ACLNN_ERR_PARAM_INVALID, "chunkSize must be 64 or 128, but got %ld.", params.chunkSize);
    CHECK_RET(CheckGateOptionalNonNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckReservedOptions(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGkParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckChunkMetadataAndStateShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *initalStateOptional,
    bool outputFinalState,
    int64_t chunkSize,
    bool saveNewValue,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool useExp2,
    bool transposeStateLayout,
    const aclTensor *hOut,
    const aclTensor *vNewOut,
    const aclTensor *finalStateOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize must not be nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");
    ChunkGatedDeltaRuleFwdHParams params{k,
                                         w,
                                         u,
                                         gOptional,
                                         gkOptional,
                                         initalStateOptional,
                                         outputFinalState,
                                         chunkSize,
                                         saveNewValue,
                                         cuSeqlensOptional,
                                         chunkIndicesOptional,
                                         useExp2,
                                         transposeStateLayout,
                                         hOut,
                                         vNewOut,
                                         finalStateOut};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleFwdH,
                   DFX_IN(k, w, u, gOptional, gkOptional, initalStateOptional, cuSeqlensOptional, chunkIndicesOptional),
                   DFX_OUT(hOut, vNewOut, finalStateOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
    if (params.gOptional == nullptr) {
        params.gOptional = MakeNeutralGate(params, executorPtr);
        CHECK_RET(params.gOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto result = l0op::ChunkGatedDeltaRuleFwdH(params.k, params.w, params.u, params.gOptional, params.gkOptional, params.initalStateOptional, params.cuSeqlensOptional, params.chunkIndicesOptional, params.outputFinalState, params.chunkSize, params.hOut, params.vNewOut, params.finalStateOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
    auto viewCopyResult0 = l0op::ViewCopy(result[0], params.hOut, executorPtr);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(result[1], params.vNewOut, executorPtr);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (outputFinalState && params.finalStateOut != nullptr) {
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
