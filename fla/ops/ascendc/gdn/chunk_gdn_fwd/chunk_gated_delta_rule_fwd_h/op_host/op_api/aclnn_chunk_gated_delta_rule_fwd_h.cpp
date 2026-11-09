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
#include "../../../../../common/host_utils/log.h"
#include "opdev/make_op_executor.h"


using namespace op;

static constexpr const char *CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY = "aclnnChunkGatedDeltaRuleFwdH";

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
    if (params.k == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "k");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.w == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "w");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.u == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "u");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.hOut == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "hOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.vNewOut == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "vNewOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(ChunkGatedDeltaRuleFwdHParams params)
{
    return ACLNN_SUCCESS;
}

static int64_t CountChunks(const aclIntArray *cuSeqlens, int64_t seqlen, int64_t chunkSize)
{
    if (cuSeqlens == nullptr) {
        return (seqlen + chunkSize - 1) / chunkSize;
    }
    int64_t totalChunks = 0;
    for (size_t seq = 0; seq + 1 < cuSeqlens->Size(); ++seq) {
        const int64_t length = (*cuSeqlens)[seq + 1] - (*cuSeqlens)[seq];
        totalChunks += (length + chunkSize - 1) / chunkSize;
    }
    return totalChunks;
}

static aclnnStatus CheckShape(ChunkGatedDeltaRuleFwdHParams params)
{
    auto kShape = params.k->GetViewShape();
    auto wShape = params.w->GetViewShape();
    auto uShape = params.u->GetViewShape();
    if (kShape.GetDimNum() != 4 || wShape.GetDimNum() != 4 || uShape.GetDimNum() != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPES_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "k,w,u", "rank mismatch", "rank-4 BNSD tensors");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (kShape.GetDim(0) != wShape.GetDim(0) || kShape.GetDim(0) != uShape.GetDim(0) ||
        wShape.GetDim(1) != uShape.GetDim(1) || kShape.GetDim(2) != wShape.GetDim(2) ||
        kShape.GetDim(2) != uShape.GetDim(2) || kShape.GetDim(3) != wShape.GetDim(3)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPES_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "k,w,u", "mismatch",
            "match B/T, w/u HV, k/w K");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (uShape.GetDim(1) < kShape.GetDim(1) || uShape.GetDim(1) % kShape.GetDim(1) != 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "u.HV,k.H", "not divisible",
            "u HV >= k H and divisible by H");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.gOptional != nullptr) {
        auto gShape = params.gOptional->GetViewShape();
        if (gShape.GetDimNum() != 3 || gShape.GetDim(0) != uShape.GetDim(0) ||
            gShape.GetDim(1) != uShape.GetDim(1) || gShape.GetDim(2) != uShape.GetDim(2)) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "g", "mismatch", "[B, HV, T]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const int64_t batch = kShape.GetDim(0);
    const int64_t hv = uShape.GetDim(1);
    const int64_t kDim = kShape.GetDim(3);
    const int64_t vDim = uShape.GetDim(3);
    const int64_t seqNum = params.cuSeqlensOptional == nullptr
                               ? batch
                               : static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;
    auto hShape = params.hOut->GetViewShape();
    const int64_t chunks = CountChunks(params.cuSeqlensOptional, kShape.GetDim(2), params.chunkSize);
    if (hShape.GetDimNum() != 5 ||
        hShape.GetDim(0) != batch ||
        hShape.GetDim(1) != chunks || hShape.GetDim(2) != hv) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "hOut", "mismatch", "[B, NT, HV, K, V] with matching chunk count");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t hK = hShape.GetDim(params.stateVFirst ? 4 : 3);
    const int64_t hV = hShape.GetDim(params.stateVFirst ? 3 : 4);
    if (hK != kDim || hV != vDim) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "hOut", "state dim mismatch",
            "[K, V] when stateVFirst=false, [V, K] otherwise");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto vNewShape = params.vNewOut->GetViewShape();
    if (vNewShape.GetDimNum() != 4 || vNewShape.GetDim(0) != batch ||
        vNewShape.GetDim(1) != hv || vNewShape.GetDim(2) != kShape.GetDim(2) ||
        vNewShape.GetDim(3) != vDim) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "vNewOut", "mismatch", "[B, HV, T, V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const aclTensor *states[] = {params.initialStateOptional, params.finalStateOut};
    const char *stateNames[] = {"initialStateOptional", "finalStateOut"};
    for (size_t idx = 0; idx < 2; ++idx) {
        if (states[idx] == nullptr) {
            continue;
        }
        auto stateShape = states[idx]->GetViewShape();
        if (stateShape.GetDimNum() != 4) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, stateNames[idx], "rank mismatch", "rank 4");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t stateK = params.stateVFirst ? stateShape.GetDim(3) : stateShape.GetDim(2);
        const int64_t stateV = params.stateVFirst ? stateShape.GetDim(2) : stateShape.GetDim(3);
        if (stateShape.GetDim(0) != seqNum ||
            stateShape.GetDim(1) != hv || stateK != kDim || stateV != vDim) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, stateNames[idx], "mismatch",
                "[N, HV, K, V] or [N, HV, V, K] per stateVFirst");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkGatedDeltaRuleFwdHParams params)
{
    auto inputDtype = params.k->GetDataType();
    if (inputDtype != DataType::DT_FLOAT16 && inputDtype != DataType::DT_BF16) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "k",
            fla_npu::host_utils::DfxDtypeString(inputDtype), "FLOAT16,BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.w->GetDataType() != inputDtype || params.u->GetDataType() != inputDtype) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPES_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "w,u", "dtype mismatch", "match k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.hOut->GetDataType() != inputDtype || params.vNewOut->GetDataType() != inputDtype) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPES_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "hOut,vNewOut", "dtype mismatch", "match k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto gateDtype = params.gOptional != nullptr ? params.gOptional->GetDataType() : params.gkOptional->GetDataType();
    if (gateDtype != DataType::DT_FLOAT && gateDtype != inputDtype) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "g,gk",
            fla_npu::host_utils::DfxDtypeString(gateDtype), "FLOAT32 or match k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.gOptional != nullptr && params.gkOptional != nullptr &&
        params.gOptional->GetDataType() != params.gkOptional->GetDataType()) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPES_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "g,gk", "dtype mismatch", "same dtype");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.outputFinalState) {
        if (params.finalStateOut == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "finalStateOut");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        auto stateDtype = params.initialStateOptional != nullptr ? params.initialStateOptional->GetDataType()
                                                                : DataType::DT_FLOAT;
        if (params.finalStateOut->GetDataType() != stateDtype) {
            OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "finalStateOut",
                fla_npu::host_utils::DfxDtypeString(params.finalStateOut->GetDataType()),
                "match initial state, or FLOAT32 when absent");
            return ACLNN_ERR_PARAM_INVALID;
        }
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
    auto dfxRetStatus1 = DataContiguous(params.k, executorPtr);
    if (dfxRetStatus1 != ACLNN_SUCCESS) {
        return dfxRetStatus1;
    }
    auto dfxRetStatus2 = DataContiguous(params.w, executorPtr);
    if (dfxRetStatus2 != ACLNN_SUCCESS) {
        return dfxRetStatus2;
    }
    auto dfxRetStatus3 = DataContiguous(params.u, executorPtr);
    if (dfxRetStatus3 != ACLNN_SUCCESS) {
        return dfxRetStatus3;
    }
    if (params.gOptional != nullptr) {
        auto dfxRetStatus4 = DataContiguous(params.gOptional, executorPtr);
    if (dfxRetStatus4 != ACLNN_SUCCESS) {
        return dfxRetStatus4;
    }
    }
    if (params.gkOptional != nullptr) {
        auto dfxRetStatus5 = DataContiguous(params.gkOptional, executorPtr);
    if (dfxRetStatus5 != ACLNN_SUCCESS) {
        return dfxRetStatus5;
    }
    }
    if (params.initialStateOptional != nullptr) {
        CHECK_COND(DataContiguous(params.initialStateOptional, executorPtr) == ACLNN_SUCCESS,
                   ACLNN_ERR_PARAM_INVALID, "Contiguous initialStateOptional failed.");
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckGateOptionalNonNull(const ChunkGatedDeltaRuleFwdHParams &params)
{
    if (params.gOptional == nullptr && params.gkOptional == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "g,gk");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckGkParams(const ChunkGatedDeltaRuleFwdHParams &params)
{
    if (params.gkOptional != nullptr) {
        auto gkShape = params.gkOptional->GetViewShape();
        if (gkShape.GetDimNum() != 4) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "gk",
                "rank=" + std::to_string(gkShape.GetDimNum()), "rank 4");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (gkShape.GetDim(3) != params.k->GetViewShape().GetDim(3)) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "gk", "K mismatch", "match k K");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (gkShape.GetDim(2) != params.k->GetViewShape().GetDim(2)) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "gk", "T mismatch", "match k T");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (gkShape.GetDim(1) != params.u->GetViewShape().GetDim(1)) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "gk", "HV mismatch", "match u HV");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (gkShape.GetDim(0) != params.k->GetViewShape().GetDim(0)) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "gk", "B mismatch", "match k B");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (params.gOptional != nullptr &&
            params.gkOptional->GetDataType() != params.gOptional->GetDataType()) {
            OP_LOGE_LIBOPAPI_REPORT_DTYPES_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "g,gk", "dtype mismatch", "same dtype");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckVariableLength(const ChunkGatedDeltaRuleFwdHParams &params)
{
    if (params.cuSeqlensOptional == nullptr) {
        if (params.chunkIndicesOptional != nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "chunkIndicesOptional", "present",
                "requires cuSeqlensOptional");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    const auto kShape = params.k->GetViewShape();
    const int64_t seqNum = static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;
    if (seqNum < 1) {
        OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "cuSeqlensOptional",
            std::to_string(params.cuSeqlensOptional->Size()), "at least [0, total_tokens]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((*params.cuSeqlensOptional)[0] != 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "cuSeqlensOptional[0]",
            std::to_string((*params.cuSeqlensOptional)[0]), "0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((*params.cuSeqlensOptional)[seqNum] != kShape.GetDim(2)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "cuSeqlensOptional[-1]",
            std::to_string((*params.cuSeqlensOptional)[seqNum]),
            "T=" + std::to_string(kShape.GetDim(2)));
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (int64_t seq = 0; seq < seqNum; ++seq) {
        if ((*params.cuSeqlensOptional)[seq] > (*params.cuSeqlensOptional)[seq + 1]) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "cuSeqlensOptional", "non-decreasing",
                "non-decreasing; equal neighbours denote empty sequences");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (params.chunkIndicesOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const int64_t totalChunks =
        CountChunks(params.cuSeqlensOptional, kShape.GetDim(2), params.chunkSize);
    if (params.chunkIndicesOptional->Size() != static_cast<size_t>(totalChunks) * 2) {
        OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "chunkIndicesOptional",
            std::to_string(params.chunkIndicesOptional->Size()),
            std::to_string(totalChunks * 2));
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t offset = 0;
    for (int64_t seq = 0; seq < seqNum; ++seq) {
        const int64_t length =
            (*params.cuSeqlensOptional)[seq + 1] - (*params.cuSeqlensOptional)[seq];
        const int64_t chunks = (length + params.chunkSize - 1) / params.chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            if ((*params.chunkIndicesOptional)[offset] != seq ||
                (*params.chunkIndicesOptional)[offset + 1] != chunk) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_GATED_DELTA_RULE_FWD_H_ENTITY, "chunkIndicesOptional", "non-canonical",
                    "canonical sequence-major chunk order");
                return ACLNN_ERR_PARAM_INVALID;
            }
            offset += 2;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkGatedDeltaRuleFwdHParams params)
{
    auto status = CheckNotNull(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    CHECK_COND(params.chunkSize == 64 || params.chunkSize == 128, ACLNN_ERR_PARAM_INVALID,
               "chunkSize must be 64 or 128.");
    status = CheckGateOptionalNonNull(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGkParams(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckFormat(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckShape(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckDtype(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckVariableLength(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
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
                   DFX_IN(k, w, u, gOptional, gkOptional, initialStateOptional, outputFinalState, chunkSize,
                          cuSeqlensOptional, chunkIndicesOptional, stateVFirst),
                   DFX_OUT(hOut, vNewOut, finalStateOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    auto ret = CheckParams(params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    auto dfxRetStatus6 = ParamsDataContiguous(params, executorPtr);
    if (dfxRetStatus6 != ACLNN_SUCCESS) {
        return dfxRetStatus6;
    }
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
