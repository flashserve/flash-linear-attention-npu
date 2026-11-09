/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file aclnn_recurrent_kda.cpp
 * \brief
 */
#include "aclnn_recurrent_kda.h"
#include "recurrent_kda.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

#include "../../../../common/host_utils/log.h"

#include <cstring>
#include <limits>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr const char *RECURRENT_KDA_ENTITY = "aclnnRecurrentKda";
constexpr size_t DIM0 = 0;
constexpr size_t DIM1 = 1;
constexpr size_t DIM2 = 2;
constexpr size_t DIM3 = 3;

enum class RecurrentKdaLayout {
    BSND,
    TND,
};

struct RecurrentKdaParams {
    const aclTensor *query = nullptr;
    const aclTensor *key = nullptr;
    const aclTensor *value = nullptr;
    const aclTensor *gate = nullptr;
    const aclTensor *beta = nullptr;
    aclTensor *initialStateRef = nullptr;
    const aclTensor *cuSeqlensOptional = nullptr;
    const aclTensor *ssmStateIndicesOptional = nullptr;
    const aclTensor *aLogOptional = nullptr;
    const aclTensor *dtBiasOptional = nullptr;
    const aclTensor *numAcceptedTokensOptional = nullptr;
    const char *layout = "BSND";
    double scale = 1.0;
    bool outputFinalState = false;
    bool inplaceFinalState = true;
    bool useQkL2normInKernel = false;
    bool useGateInKernel = false;
    bool useBetaSigmoidInKernel = false;
    bool allowNegEigval = false;
    bool safeGate = false;
    double lowerBound = -5.0;
    bool stateVFirst = false;
    const aclTensor *attnOut = nullptr;
    const aclTensor *finalState = nullptr;
};

static const std::initializer_list<op::DataType> QKV_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> STATE_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16,
                                                                             op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> GATE_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                            op::DataType::DT_BF16,
                                                                            op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> F32_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> INT_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32,
                                                                          op::DataType::DT_INT64};

static size_t Rank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

static int64_t Dim(const aclTensor *tensor, size_t idx)
{
    return tensor->GetViewShape().GetDim(idx);
}

static bool SameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    if (Rank(lhs) != Rank(rhs)) {
        return false;
    }
    for (size_t i = 0; i < Rank(lhs); ++i) {
        if (Dim(lhs, i) != Dim(rhs, i)) {
            return false;
        }
    }
    return true;
}

static bool SameViewStrides(const aclTensor *lhs, const aclTensor *rhs)
{
    const auto &lhsStrides = lhs->GetViewStrides();
    const auto &rhsStrides = rhs->GetViewStrides();
    if (lhsStrides.size() != rhsStrides.size()) {
        return false;
    }
    for (size_t i = 0; i < lhsStrides.size(); ++i) {
        if (lhsStrides[i] != rhsStrides[i]) {
            return false;
        }
    }
    return true;
}

static bool SameTensorView(const aclTensor *lhs, const aclTensor *rhs)
{
    if (lhs == rhs) {
        return true;
    }
    void *lhsStorage = lhs->GetStorageAddr();
    if (lhsStorage == nullptr || lhsStorage != rhs->GetStorageAddr()) {
        return false;
    }
    return lhs->GetStorageOffset() == rhs->GetStorageOffset() &&
           lhs->GetViewOffset() == rhs->GetViewOffset() && SameShape(lhs, rhs) &&
           SameViewStrides(lhs, rhs);
}

static bool ParseLayout(const char *layout, RecurrentKdaLayout &parsed)
{
    if (layout == nullptr || std::strcmp(layout, "BSND") == 0) {
        parsed = RecurrentKdaLayout::BSND;
        return true;
    }
    if (std::strcmp(layout, "TND") == 0) {
        parsed = RecurrentKdaLayout::TND;
        return true;
    }
    OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(RECURRENT_KDA_ENTITY, "layout", layout, "BSND or TND");
    return false;
}

bool IsSupportedQkvView(const aclTensor *tensor, RecurrentKdaLayout layout)
{
    if (tensor == nullptr) {
        return false;
    }
    const size_t expectedRank = layout == RecurrentKdaLayout::TND ? 3 : 4;
    const auto &strides = tensor->GetViewStrides();
    if (Rank(tensor) != expectedRank || strides.size() != expectedRank) {
        return false;
    }
    for (size_t i = 0; i < expectedRank; ++i) {
        if (strides[i] <= 0) {
            return false;
        }
    }

    const size_t tokenDim = layout == RecurrentKdaLayout::TND ? DIM0 : DIM1;
    const size_t headDim = layout == RecurrentKdaLayout::TND ? DIM1 : DIM2;
    const size_t featureDim = layout == RecurrentKdaLayout::TND ? DIM2 : DIM3;
    const int64_t headCount = Dim(tensor, headDim);
    const int64_t featureCount = Dim(tensor, featureDim);
    const int64_t tokenStride = strides[tokenDim];
    const int64_t headStride = strides[headDim];
    if (strides[featureDim] != 1 || headStride < featureCount) {
        return false;
    }
    const int64_t maxInt64 = std::numeric_limits<int64_t>::max();
    if (headCount > 1 && headStride > (maxInt64 - featureCount) / (headCount - 1)) {
        return false;
    }
    const int64_t minTokenStride = (headCount - 1) * headStride + featureCount;
    if (tokenStride < minTokenStride) {
        return false;
    }
    const uint64_t srcGapElements = static_cast<uint64_t>(tokenStride - featureCount);
    if (srcGapElements > std::numeric_limits<uint32_t>::max() / sizeof(uint16_t)) {
        return false;
    }

    if (layout == RecurrentKdaLayout::BSND && Dim(tensor, DIM0) > 1) {
        const int64_t seqLen = Dim(tensor, DIM1);
        if (tokenStride > maxInt64 / seqLen || strides[DIM0] != seqLen * tokenStride) {
            return false;
        }
    }
    return true;
}

bool CheckQkvMaterializationSafety(
    const aclTensor *tensor, RecurrentKdaLayout layout, const char *name)
{
    const size_t expectedRank = layout == RecurrentKdaLayout::TND ? 3 : 4;
    const auto &strides = tensor->GetViewStrides();
    if (Rank(tensor) != expectedRank || strides.size() != expectedRank) {
        return true;
    }
    const size_t tokenDim = layout == RecurrentKdaLayout::TND ? DIM0 : DIM1;
    const size_t headDim = layout == RecurrentKdaLayout::TND ? DIM1 : DIM2;
    const size_t featureDim = layout == RecurrentKdaLayout::TND ? DIM2 : DIM3;
    const int64_t headCount = Dim(tensor, headDim);
    const int64_t featureCount = Dim(tensor, featureDim);
    const int64_t tokenStride = strides[tokenDim];
    const int64_t headStride = strides[headDim];
    if (strides[featureDim] != 1 || tokenStride <= 0 || headStride <= 0 ||
        tokenStride < featureCount || headStride < featureCount) {
        return true;
    }
    const int64_t maxInt64 = std::numeric_limits<int64_t>::max();
    if (headCount > 1 && headStride > (maxInt64 - featureCount) / (headCount - 1)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(RECURRENT_KDA_ENTITY, name, "head span overflow", "in int64 range");
        return false;
    }
    const int64_t minTokenStride = (headCount - 1) * headStride + featureCount;
    if (tokenStride < minTokenStride) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(RECURRENT_KDA_ENTITY, name, "token/head axis overlap", "supported order");
        return false;
    }
    return true;
}

bool CheckCuSeqlensShape(const aclTensor *cuSeqlens, const char *opName)
{
    if (cuSeqlens == nullptr) {
        return true;
    }
    if (Rank(cuSeqlens) != 1 || Dim(cuSeqlens, DIM0) < 2) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECURRENT_KDA_ENTITY, "cuSeqlensOptional", "mismatch", "1D, >=2 elements");
        return false;
    }
    return true;
}

bool CheckShape(const RecurrentKdaParams &params, RecurrentKdaLayout layout)
{
    if (!SameShape(params.query, params.key)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECURRENT_KDA_ENTITY, "query,key", "mismatch", "identical");
        return false;
    }
    int64_t totalTokens = 0;
    int64_t denseSeqLen = 0;
    int64_t batch = 1;
    int64_t h = 0;
    int64_t hv = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
    if (layout == RecurrentKdaLayout::TND) {
        if (Rank(params.query) != 3 || Rank(params.value) != 3 || Rank(params.gate) != 3 || Rank(params.beta) != 2) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECURRENT_KDA_ENTITY, "q,k,v,g,beta", "rank mismatch", "TND ranks");
            return false;
        }
        totalTokens = Dim(params.query, DIM0);
        denseSeqLen = totalTokens;
        h = Dim(params.query, DIM1);
        kDim = Dim(params.query, DIM2);
        hv = Dim(params.value, DIM1);
        vDim = Dim(params.value, DIM2);
        if (Dim(params.value, DIM0) != totalTokens || Dim(params.gate, DIM0) != totalTokens ||
            Dim(params.beta, DIM0) != totalTokens || Dim(params.gate, DIM1) != hv ||
            Dim(params.beta, DIM1) != hv || Dim(params.gate, DIM2) != kDim) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECURRENT_KDA_ENTITY, "v,g,beta", "mismatch", "TND shapes");
            return false;
        }
    } else {
        if (Rank(params.query) != 4 || Rank(params.value) != 4 || Rank(params.gate) != 4 || Rank(params.beta) != 3) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECURRENT_KDA_ENTITY, "q,k,v,g,beta", "rank mismatch", "BSND ranks");
            return false;
        }
        batch = Dim(params.query, DIM0);
        denseSeqLen = Dim(params.query, DIM1);
        totalTokens = batch * denseSeqLen;
        h = Dim(params.query, DIM2);
        kDim = Dim(params.query, DIM3);
        hv = Dim(params.value, DIM2);
        vDim = Dim(params.value, DIM3);
        if (Dim(params.value, DIM0) != batch || Dim(params.value, DIM1) != denseSeqLen ||
            Dim(params.gate, DIM0) != batch || Dim(params.gate, DIM1) != denseSeqLen ||
            Dim(params.gate, DIM2) != hv || Dim(params.gate, DIM3) != kDim ||
            Dim(params.beta, DIM0) != batch || Dim(params.beta, DIM1) != denseSeqLen ||
            Dim(params.beta, DIM2) != hv) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECURRENT_KDA_ENTITY, "v,g,beta", "mismatch", "BSND shapes");
            return false;
        }
    }
    if (h <= 0 || hv <= 0 || kDim <= 0 || vDim <= 0 || totalTokens <= 0 || denseSeqLen <= 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(RECURRENT_KDA_ENTITY, "shape dims", "non-positive", "positive");
        return false;
    }
    if (hv % h != 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(RECURRENT_KDA_ENTITY, "H,HV", "HV%H!=0", "HV%H==0");
        return false;
    }
    if (kDim != 128 || (vDim != 128 && vDim != 256)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUES_COMPAT(RECURRENT_KDA_ENTITY, "K,V",
            "K=" + std::to_string(kDim) + ",V=" + std::to_string(vDim), "K=128,V=128 or K=128,V=256");
        return false;
    }
    if (!CheckCuSeqlensShape(params.cuSeqlensOptional, "npu_recurrent_kda")) {
        return false;
    }
    int64_t seqNum = params.cuSeqlensOptional == nullptr ?
        ((layout == RecurrentKdaLayout::BSND) ? batch : 1) :
        Dim(params.cuSeqlensOptional, DIM0) - 1;
    if (Rank(params.initialStateRef) != 4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(RECURRENT_KDA_ENTITY, "initialStateRef",
            "rank=" + std::to_string(Rank(params.initialStateRef)), "rank 4");
        return false;
    }
    int64_t stateCapacity = Dim(params.initialStateRef, DIM0);
    bool stateTailMatches = params.stateVFirst ?
        (Dim(params.initialStateRef, DIM2) == vDim && Dim(params.initialStateRef, DIM3) == kDim) :
        (Dim(params.initialStateRef, DIM2) == kDim && Dim(params.initialStateRef, DIM3) == vDim);
    if (stateCapacity <= 0 || Dim(params.initialStateRef, DIM1) != hv || !stateTailMatches ||
        (params.ssmStateIndicesOptional == nullptr && stateCapacity != seqNum)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECURRENT_KDA_ENTITY, "initialStateRef", "mismatch",
            "[state_capacity,HV,V,K] or [state_capacity,HV,K,V]");
        return false;
    }
    if (!SameShape(params.attnOut, params.value)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECURRENT_KDA_ENTITY, "attnOut", "mismatch", "match value");
        return false;
    }
    if (!SameShape(params.finalState, params.initialStateRef)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECURRENT_KDA_ENTITY, "finalState", "mismatch", "match initialStateRef");
        return false;
    }
    if (params.ssmStateIndicesOptional != nullptr) {
        size_t rank = Rank(params.ssmStateIndicesOptional);
        bool packed1d = rank == 1 && Dim(params.ssmStateIndicesOptional, DIM0) >= totalTokens;
        bool speculative2d = rank == 2 && Dim(params.ssmStateIndicesOptional, DIM0) == seqNum &&
                             Dim(params.ssmStateIndicesOptional, DIM1) > 0;
        if (!packed1d && !speculative2d) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECURRENT_KDA_ENTITY, "ssmStateIndicesOptional", "mismatch",
                "packed [T] or speculative [seq_num,max_step]");
            return false;
        }
    }
    if (params.numAcceptedTokensOptional != nullptr &&
        (Rank(params.numAcceptedTokensOptional) != 1 || Dim(params.numAcceptedTokensOptional, DIM0) != seqNum)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECURRENT_KDA_ENTITY, "numAcceptedTokensOptional", "mismatch", "[seq_num]");
        return false;
    }
    if (params.useGateInKernel && params.aLogOptional == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECURRENT_KDA_ENTITY, "A_log");
        return false;
    }
    if (!params.useGateInKernel && (params.safeGate || params.aLogOptional != nullptr || params.dtBiasOptional != nullptr)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(RECURRENT_KDA_ENTITY, "A_log/dt_bias/safe_gate", "present",
            "require use_gate_in_kernel=True");
        return false;
    }
    if (params.aLogOptional != nullptr && (Rank(params.aLogOptional) != 1 || Dim(params.aLogOptional, DIM0) != hv)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECURRENT_KDA_ENTITY, "A_log", "mismatch", "float32 [HV]");
        return false;
    }
    if (params.dtBiasOptional != nullptr) {
        bool dtBiasOk = (Rank(params.dtBiasOptional) == 1 && Dim(params.dtBiasOptional, DIM0) == hv * kDim) ||
                        (Rank(params.dtBiasOptional) == 2 && Dim(params.dtBiasOptional, DIM0) == hv &&
                         Dim(params.dtBiasOptional, DIM1) == kDim);
        if (!dtBiasOk) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(RECURRENT_KDA_ENTITY, "dt_bias", "mismatch", "float32 [HV*K] or [HV,K]");
            return false;
        }
    }
    if (params.safeGate && (params.lowerBound < -5.0 || params.lowerBound >= 0.0)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(RECURRENT_KDA_ENTITY, "lower_bound", "out of range", "[-5, 0)");
        return false;
    }
    return true;
}

aclnnStatus CheckNotNull(const RecurrentKdaParams &params)
{
    if (params.query == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECURRENT_KDA_ENTITY, "query");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.key == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECURRENT_KDA_ENTITY, "key");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.value == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECURRENT_KDA_ENTITY, "value");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.gate == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECURRENT_KDA_ENTITY, "gate");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.beta == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECURRENT_KDA_ENTITY, "beta");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.initialStateRef == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECURRENT_KDA_ENTITY, "initialStateRef");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.attnOut == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECURRENT_KDA_ENTITY, "attnOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.finalState == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(RECURRENT_KDA_ENTITY, "finalState");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

bool CheckDtypeValid(const RecurrentKdaParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.query, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.key, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.value, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.gate, GATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.beta, GATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.initialStateRef, STATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.attnOut, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.finalState, STATE_TYPE_SUPPORT_LIST, return false);
    if (params.finalState->GetDataType() != params.initialStateRef->GetDataType()) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(RECURRENT_KDA_ENTITY, "finalState",
            fla_npu::host_utils::DfxDtypeString(params.finalState->GetDataType()), "match initialStateRef");
        return false;
    }
    if (params.cuSeqlensOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.cuSeqlensOptional, INT_TYPE_SUPPORT_LIST, return false);
    }
    if (params.ssmStateIndicesOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.ssmStateIndicesOptional, INT_TYPE_SUPPORT_LIST, return false);
    }
    if (params.aLogOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.aLogOptional, F32_TYPE_SUPPORT_LIST, return false);
    }
    if (params.dtBiasOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.dtBiasOptional, F32_TYPE_SUPPORT_LIST, return false);
    }
    if (params.numAcceptedTokensOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.numAcceptedTokensOptional, INT_TYPE_SUPPORT_LIST, return false);
    }
    return true;
}

aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus DataContiguousIfNeeded(
    const aclTensor *&tensor, bool needsContiguous, aclOpExecutor *executor)
{
    if (!needsContiguous) {
        return ACLNN_SUCCESS;
    }
    return DataContiguous(tensor, executor);
}

aclnnStatus CreateViewIfNonContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr || IsContiguous(tensor)) {
        return ACLNN_SUCCESS;
    }
    const aclTensor *view = executor->CreateView(
        tensor, tensor->GetViewShape(), tensor->GetStorageShape(),
        tensor->GetViewStrides(), tensor->GetViewOffset());
    CHECK_RET(view != nullptr, ACLNN_ERR_INNER_NULLPTR);
    tensor = view;
    return ACLNN_SUCCESS;
}

void SetTensorOriginalShape(const aclTensor *tensor)
{
    if (tensor != nullptr) {
        tensor->SetOriginalShape(tensor->GetViewShape());
    }
}

void SetInputOriginalShape(RecurrentKdaParams &params)
{
    SetTensorOriginalShape(params.query);
    SetTensorOriginalShape(params.key);
    SetTensorOriginalShape(params.value);
    SetTensorOriginalShape(params.gate);
    SetTensorOriginalShape(params.beta);
    SetTensorOriginalShape(params.initialStateRef);
    SetTensorOriginalShape(params.cuSeqlensOptional);
    SetTensorOriginalShape(params.ssmStateIndicesOptional);
    SetTensorOriginalShape(params.aLogOptional);
    SetTensorOriginalShape(params.dtBiasOptional);
    SetTensorOriginalShape(params.numAcceptedTokensOptional);
}

aclnnStatus PreProcess(
    RecurrentKdaParams &params,
    bool queryNeedsContiguous,
    bool keyNeedsContiguous,
    bool valueNeedsContiguous,
    aclOpExecutor *executor)
{
    SetInputOriginalShape(params);
    CHECK_RET(DataContiguousIfNeeded(params.query, queryNeedsContiguous, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguousIfNeeded(params.key, keyNeedsContiguous, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguousIfNeeded(params.value, valueNeedsContiguous, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.gate, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.beta, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.cuSeqlensOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.ssmStateIndicesOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.aLogOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.dtBiasOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.numAcceptedTokensOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    if (params.ssmStateIndicesOptional == nullptr && params.numAcceptedTokensOptional != nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(RECURRENT_KDA_ENTITY, "num_accepted_tokens", "present",
            "requires ssm_state_indices");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnRecurrentKdaGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *gate,
    const aclTensor *beta,
    aclTensor *initialStateRef,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *ssmStateIndicesOptional,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *numAcceptedTokensOptional,
    const char *layout,
    double scale,
    bool outputFinalState,
    bool inplaceFinalState,
    bool useQkL2normInKernel,
    bool useGateInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool safeGate,
    double lowerBound,
    bool stateVFirst,
    const aclTensor *attnOut,
    const aclTensor *finalState,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnRecurrentKda,
                   DFX_IN(query, key, value, gate, beta, initialStateRef, cuSeqlensOptional,
                          ssmStateIndicesOptional, aLogOptional, dtBiasOptional, numAcceptedTokensOptional,
                          layout, scale, outputFinalState, inplaceFinalState, useQkL2normInKernel,
                          useGateInKernel, useBetaSigmoidInKernel, allowNegEigval, safeGate, lowerBound,
                          stateVFirst),
                   DFX_OUT(initialStateRef, attnOut, finalState));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    RecurrentKdaParams params{query, key, value, gate, beta, initialStateRef, cuSeqlensOptional,
                              ssmStateIndicesOptional, aLogOptional, dtBiasOptional,
                              numAcceptedTokensOptional, layout, scale, outputFinalState, inplaceFinalState,
                              useQkL2normInKernel, useGateInKernel, useBetaSigmoidInKernel,
                              allowNegEigval, safeGate, lowerBound, stateVFirst, attnOut, finalState};

    auto status = CheckNotNull(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);
    RecurrentKdaLayout parsedLayout = RecurrentKdaLayout::BSND;
    CHECK_RET(ParseLayout(params.layout, parsedLayout), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params, parsedLayout), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckQkvMaterializationSafety(params.query, parsedLayout, "query") &&
                  CheckQkvMaterializationSafety(params.key, parsedLayout, "key") &&
                  CheckQkvMaterializationSafety(params.value, parsedLayout, "value"),
              ACLNN_ERR_PARAM_INVALID);
    const bool querySupportsDirect = IsSupportedQkvView(params.query, parsedLayout);
    const bool keySupportsDirect = IsSupportedQkvView(params.key, parsedLayout);
    const bool valueSupportsDirect = IsSupportedQkvView(params.value, parsedLayout);
    const bool queryUsesDirectView = !IsContiguous(params.query) && querySupportsDirect;
    const bool keyUsesDirectView = !IsContiguous(params.key) && keySupportsDirect;
    const bool valueUsesDirectView = !IsContiguous(params.value) && valueSupportsDirect;
    const bool queryNeedsContiguous = !querySupportsDirect;
    const bool keyNeedsContiguous = !keySupportsDirect;
    const bool valueNeedsContiguous = !valueSupportsDirect;
    CHECK_RET(PreProcess(params, queryNeedsContiguous, keyNeedsContiguous, valueNeedsContiguous, executorPtr) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    if (queryUsesDirectView) {
        CHECK_RET(CreateViewIfNonContiguous(params.query, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }
    if (keyUsesDirectView) {
        CHECK_RET(CreateViewIfNonContiguous(params.key, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }
    if (valueUsesDirectView) {
        CHECK_RET(CreateViewIfNonContiguous(params.value, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }

    aclTensor *initialStateForKernel = params.initialStateRef;
    if (!IsContiguous(initialStateForKernel)) {
        initialStateForKernel = executorPtr->CreateView(
            initialStateForKernel,
            initialStateForKernel->GetViewShape(),
            initialStateForKernel->GetStorageShape(),
            initialStateForKernel->GetViewStrides(),
            initialStateForKernel->GetViewOffset());
        CHECK_RET(initialStateForKernel != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *finalStateForKernel = params.finalState;
    if (!IsContiguous(finalStateForKernel)) {
        finalStateForKernel = executorPtr->CreateView(
            finalStateForKernel,
            finalStateForKernel->GetViewShape(),
            finalStateForKernel->GetStorageShape(),
            finalStateForKernel->GetViewStrides(),
            finalStateForKernel->GetViewOffset());
        CHECK_RET(finalStateForKernel != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto result = l0op::RecurrentKda(
        params.query, params.key, params.value, params.gate, params.beta, initialStateForKernel,
        params.cuSeqlensOptional, params.ssmStateIndicesOptional, params.aLogOptional,
        params.dtBiasOptional, params.numAcceptedTokensOptional, params.layout, params.scale,
        params.outputFinalState, params.inplaceFinalState, params.useQkL2normInKernel,
        params.useGateInKernel, params.useBetaSigmoidInKernel, params.allowNegEigval,
        params.safeGate, params.lowerBound, params.stateVFirst, params.attnOut,
        finalStateForKernel, executorPtr);
    CHECK_RET(result[0] != nullptr && result[1] != nullptr && result[2] != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    if (params.inplaceFinalState && params.outputFinalState &&
        !SameTensorView(params.finalState, params.initialStateRef)) {
        CHECK_RET(l0op::ViewCopy(result[1], params.finalState, executorPtr) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRecurrentKda(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRecurrentKda);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
