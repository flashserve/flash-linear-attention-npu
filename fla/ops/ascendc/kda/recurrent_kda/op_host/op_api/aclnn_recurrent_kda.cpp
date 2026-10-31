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
#include "op_common/log/log.h"
#include "aclnn_recurrent_kda.h"
#include "recurrent_kda.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "acl/acl_rt.h"

#include <cstring>
#include <algorithm>
#include <limits>
#include <vector>
#include <cstdio>
#include <cstdarg>
#include <utility>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
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

std::string TensorDescription(const aclTensor *tensor)
{
    if (tensor == nullptr) return "None";
    std::string result = "shape=[";
    for (size_t i = 0; i < tensor->GetViewShape().GetDimNum(); ++i) {
        if (i != 0) result += ",";
        result += std::to_string(tensor->GetViewShape().GetDim(i));
    }
    result += "],strides=[";
    for (size_t i = 0; i < tensor->GetViewStrides().size(); ++i) {
        if (i != 0) result += ",";
        result += std::to_string(tensor->GetViewStrides()[i]);
    }
    return result + "],dtype=" + std::to_string(static_cast<int>(tensor->GetDataType()));
}

void ReportInvalid(const RecurrentKdaParams &params, const char *name, const char *format, ...)
{
    char expected[768];
    va_list args;
    va_start(args, format);
    std::vsnprintf(expected, sizeof(expected), format, args);
    va_end(args);
    const std::string reason = std::string(expected) + " Actual: query " + TensorDescription(params.query) +
        "; key " + TensorDescription(params.key) + "; value " + TensorDescription(params.value) +
        "; gate " + TensorDescription(params.gate) + "; beta " + TensorDescription(params.beta) +
        "; initialStateRef " + TensorDescription(params.initialStateRef) +
        "; finalState " + TensorDescription(params.finalState) +
        "; cu_seqlens " + TensorDescription(params.cuSeqlensOptional) +
        "; ssm_state_indices " + TensorDescription(params.ssmStateIndicesOptional) +
        "; num_accepted_tokens " + TensorDescription(params.numAcceptedTokensOptional) +
        "; A_log " + TensorDescription(params.aLogOptional) + "; dt_bias " + TensorDescription(params.dtBiasOptional) +
        "; lower_bound=" + std::to_string(params.lowerBound);
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentKda", name, reason);
}

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

static bool SameTensorStorage(const aclTensor *lhs, const aclTensor *rhs)
{
    return lhs != nullptr && rhs != nullptr && lhs->GetStorageAddr() != nullptr &&
           lhs->GetStorageAddr() == rhs->GetStorageAddr();
}

static const uint8_t *ViewData(const aclTensor *tensor, uint64_t elementBytes)
{
    return static_cast<const uint8_t *>(tensor->GetStorageAddr()) +
           uint64_t(tensor->GetViewOffset()) * elementBytes;
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
    const int64_t tokenCount = Dim(tensor, tokenDim);
    if (strides[featureDim] != 1 || tokenStride < featureCount || headStride < featureCount) {
        return false;
    }
    const int64_t maxInt64 = std::numeric_limits<int64_t>::max();
    if (headCount > 1 && headStride > (maxInt64 - featureCount) / (headCount - 1)) {
        return false;
    }
    if (tokenCount > 1 && tokenStride > (maxInt64 - featureCount) / (tokenCount - 1)) {
        return false;
    }
    const int64_t tokenMajorSpan = (headCount - 1) * headStride + featureCount;
    const int64_t headMajorSpan = (tokenCount - 1) * tokenStride + featureCount;
    const bool tokenMajor = tokenStride >= tokenMajorSpan;
    const bool headMajor = headStride >= headMajorSpan;
    if (!tokenMajor && !headMajor) {
        return false;
    }
    const uint64_t srcGapElements = static_cast<uint64_t>(tokenStride - featureCount);
    if (srcGapElements > std::numeric_limits<uint32_t>::max() / sizeof(uint16_t)) {
        return false;
    }

    // Each sequence is a DMA segment with its independent physical batch base.
    if (layout == RecurrentKdaLayout::BSND && Dim(tensor, DIM0) > 1) {
        const int64_t seqLen = Dim(tensor, DIM1);
        if (!tokenMajor || tokenStride > maxInt64 / seqLen ||
            strides[DIM0] < (seqLen - 1) * tokenStride + tokenMajorSpan) {
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
    const int64_t tokenCount = Dim(tensor, tokenDim);
    const int64_t maxInt64 = std::numeric_limits<int64_t>::max();
    if (headCount > 1 && headStride > (maxInt64 - featureCount) / (headCount - 1)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentKda", name,
            "expected representable head span; actual " + TensorDescription(tensor));
        return false;
    }
    if (tokenCount > 1 && tokenStride > (maxInt64 - featureCount) / (tokenCount - 1)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentKda", name,
            "expected representable token span; actual " + TensorDescription(tensor));
        return false;
    }
    const int64_t tokenMajorSpan = (headCount - 1) * headStride + featureCount;
    const int64_t headMajorSpan = (tokenCount - 1) * tokenStride + featureCount;
    if (tokenStride < tokenMajorSpan && headStride < headMajorSpan) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentKda", name,
            "expected non-overlapping token/head axes; actual " + TensorDescription(tensor));
        return false;
    }
    if (layout == RecurrentKdaLayout::BSND && Dim(tensor, DIM0) > 1) {
        if (tokenCount > 1 && tokenStride >
            (maxInt64 - tokenMajorSpan) / (tokenCount - 1)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentKda", name,
                "expected representable full batch span; actual " + TensorDescription(tensor));
            return false;
        }
        const int64_t fullViewSpan = (tokenCount - 1) * tokenStride +
                                     (headCount - 1) * headStride + featureCount;
        const int64_t batchSpan = fullViewSpan;
        if (strides[DIM0] < batchSpan) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentKda", name,
                "expected batch stride covering the view; actual " + TensorDescription(tensor));
            return false;
        }
    }
    return true;
}

bool CheckCuSeqlensShape(const aclTensor *cuSeqlens, const char *opName)
{
    if (cuSeqlens == nullptr) {
        return true;
    }
    if (Rank(cuSeqlens) != 1 || Dim(cuSeqlens, DIM0) < 2) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "cu_seqlens",
            "expected 1D with at least two elements; actual " + TensorDescription(cuSeqlens));
        return false;
    }
    return true;
}

bool CheckShape(const RecurrentKdaParams &params, bool varlen)
{
    if (!SameShape(params.query, params.key)) {
        ReportInvalid(params, "query/key/value/gate/beta", "npu_recurrent_kda: query and key must have identical shape.");
        return false;
    }
    int64_t totalTokens = 0;
    int64_t denseSeqLen = 0;
    int64_t batch = 1;
    int64_t h = 0;
    int64_t hv = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
    if (varlen) {
        if (Rank(params.query) != 3 || Rank(params.value) != 3 || Rank(params.gate) != 3 || Rank(params.beta) != 2) {
            ReportInvalid(params, "query/key/value/gate/beta",
                    "npu_recurrent_kda: TND expects q/k [T,H,K], v [T,HV,V], g [T,HV,K], beta [T,HV].");
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
            ReportInvalid(params, "query/key/value/gate/beta", "npu_recurrent_kda: TND shape mismatch.");
            return false;
        }
    } else {
        if (Rank(params.query) != 4 || Rank(params.value) != 4 || Rank(params.gate) != 4 || Rank(params.beta) != 3) {
            ReportInvalid(params, "query/key/value/gate/beta",
                    "npu_recurrent_kda: BSND expects q/k [B,T,H,K], v [B,T,HV,V], g [B,T,HV,K], beta [B,T,HV].");
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
            ReportInvalid(params, "query/key/value/gate/beta", "npu_recurrent_kda: BSND shape mismatch.");
            return false;
        }
    }
    if (h <= 0 || hv <= 0 || kDim <= 0 || vDim <= 0 || totalTokens <= 0 || denseSeqLen <= 0) {
        ReportInvalid(params, "query/key/value/gate/beta", "npu_recurrent_kda: all shape dimensions must be positive.");
        return false;
    }
    if (hv % h != 0) {
        ReportInvalid(params, "value", "npu_recurrent_kda: HV must be divisible by H.");
        return false;
    }
    if (kDim != 128 || (vDim != 128 && vDim != 256)) {
        ReportInvalid(params, "query/value",
                "npu_recurrent_kda: K/V currently support only K=128,V=128 or K=128,V=256, but K=%ld,V=%ld.",
                kDim, vDim);
        return false;
    }
    if (!CheckCuSeqlensShape(params.cuSeqlensOptional, "npu_recurrent_kda")) {
        return false;
    }
    int64_t seqNum = varlen ? Dim(params.cuSeqlensOptional, DIM0) - 1 : batch;
    if (Rank(params.initialStateRef) != 4) {
        ReportInvalid(params, "initialStateRef", "npu_recurrent_kda: initialStateRef must be rank 4.");
        return false;
    }
    int64_t stateCapacity = Dim(params.initialStateRef, DIM0);
    bool stateTailMatches = params.stateVFirst ?
        (Dim(params.initialStateRef, DIM2) == vDim && Dim(params.initialStateRef, DIM3) == kDim) :
        (Dim(params.initialStateRef, DIM2) == kDim && Dim(params.initialStateRef, DIM3) == vDim);
    if (stateCapacity <= 0 || Dim(params.initialStateRef, DIM1) != hv || !stateTailMatches ||
        (params.ssmStateIndicesOptional == nullptr && stateCapacity != seqNum)) {
        ReportInvalid(params, "initialStateRef",
                "npu_recurrent_kda: state must be [state_capacity,HV,V,K] when stateVFirst=true or "
                "[state_capacity,HV,K,V] otherwise; without ssmStateIndicesOptional, state_capacity must equal seq_num.");
        return false;
    }
    if (!SameShape(params.attnOut, params.value)) {
        ReportInvalid(params, "attnOut", "npu_recurrent_kda: attnOut shape must match value.");
        return false;
    }
    if (!SameShape(params.finalState, params.initialStateRef)) {
        ReportInvalid(params, "finalState", "npu_recurrent_kda: finalState shape must match initialStateRef.");
        return false;
    }
    // Reject invalid public State views before enqueueing a copy or reaching
    // tiling. A bad stride is a parameter error, not an internal L0 failure.
    const aclTensor *stateViews[] = {params.initialStateRef, params.finalState};
    for (const auto *tensor : stateViews) {
        const auto &strides = tensor->GetViewStrides();
        const int64_t plane = kDim * vDim;
        bool valid = strides.size() == 4;
        if (valid) {
            valid = strides[0] > 0 && strides[1] >= plane &&
                    strides[2] == Dim(tensor, 3) && strides[3] == 1;
        }
        if (valid && hv > 1) {
            const int64_t limit = std::numeric_limits<int64_t>::max();
            valid = strides[1] <= (limit - plane) / (hv - 1);
        }
        if (valid) valid = strides[0] >= (hv - 1) * strides[1] + plane;
        if (!valid) {
            ReportInvalid(params, tensor == params.initialStateRef ? "initialStateRef" : "finalState",
                "State must keep its inner state matrix dense, with positive non-overlapping outer strides.");
            return false;
        }
    }
    if (params.ssmStateIndicesOptional != nullptr) {
        size_t rank = Rank(params.ssmStateIndicesOptional);
        bool packed1d = rank == 1 && Dim(params.ssmStateIndicesOptional, DIM0) >= totalTokens;
        bool speculative2d = rank == 2 && Dim(params.ssmStateIndicesOptional, DIM0) == seqNum &&
                             Dim(params.ssmStateIndicesOptional, DIM1) > 0;
        if (!packed1d && !speculative2d) {
            ReportInvalid(params, "ssm_state_indices",
                    "npu_recurrent_kda: ssm_state_indices must be packed [T] or speculative [seq_num,max_step].");
            return false;
        }
    }
    if (params.numAcceptedTokensOptional != nullptr &&
        (Rank(params.numAcceptedTokensOptional) != 1 || Dim(params.numAcceptedTokensOptional, DIM0) != seqNum)) {
        ReportInvalid(params, "num_accepted_tokens", "npu_recurrent_kda: num_accepted_tokens length must equal sequence number.");
        return false;
    }
    // Gate mode is determined solely by A_log presence.  The legacy
    // use_gate_in_kernel attribute is retained for ABI compatibility and is
    // intentionally ignored by the numeric path.
    if (params.aLogOptional != nullptr && (Rank(params.aLogOptional) != 1 || Dim(params.aLogOptional, DIM0) != hv)) {
        ReportInvalid(params, "A_log", "npu_recurrent_kda: A_log must be float32 with shape [HV].");
        return false;
    }
    if (params.dtBiasOptional != nullptr) {
        bool dtBiasOk = (Rank(params.dtBiasOptional) == 1 && Dim(params.dtBiasOptional, DIM0) == hv * kDim) ||
                        (Rank(params.dtBiasOptional) == 2 && Dim(params.dtBiasOptional, DIM0) == hv &&
                         Dim(params.dtBiasOptional, DIM1) == kDim);
        if (!dtBiasOk) {
            ReportInvalid(params, "dt_bias", "npu_recurrent_kda: dt_bias must be float32 with shape [HV*K] or [HV,K].");
            return false;
        }
    }
    if (params.safeGate && !(params.lowerBound >= -5.0 && params.lowerBound < 0.0)) {
        ReportInvalid(params, "lower_bound", "npu_recurrent_kda: lower_bound must be in [-5, 0) when safe_gate=True.");
        return false;
    }
    return true;
}

bool CheckNotNull(const RecurrentKdaParams &params)
{
    const std::pair<const aclTensor *, const char *> required[] = {
        {params.query, "query"}, {params.key, "key"}, {params.value, "value"},
        {params.gate, "gate"}, {params.beta, "beta"}, {params.initialStateRef, "initialStateRef"},
        {params.attnOut, "attnOut"}, {params.finalState, "finalState"},
    };
    for (const auto &item : required) {
        if (item.first == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentKda", item.second);
            return false;
        }
    }
    return true;
}

bool CheckDtypeValid(const RecurrentKdaParams &params)
{
    const auto check = [](const aclTensor *tensor, const char *name, const auto &supported) {
        if (tensor != nullptr && !CheckType(tensor->GetDataType(), supported)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("aclnnRecurrentKda", name,
                op::ToString(tensor->GetDataType()).GetString(),
                std::string("supported dtypes: ") + op::ToString(supported).GetString());
            return false;
        }
        return true;
    };
    if (!check(params.query, "query", QKV_TYPE_SUPPORT_LIST)) return false;
    if (!check(params.key, "key", QKV_TYPE_SUPPORT_LIST)) return false;
    if (!check(params.value, "value", QKV_TYPE_SUPPORT_LIST)) return false;
    if (!check(params.gate, "gate", GATE_TYPE_SUPPORT_LIST)) return false;
    if (!check(params.beta, "beta", GATE_TYPE_SUPPORT_LIST)) return false;
    if (!check(params.initialStateRef, "initialStateRef", STATE_TYPE_SUPPORT_LIST)) return false;
    if (!check(params.attnOut, "attnOut", QKV_TYPE_SUPPORT_LIST)) return false;
    if (!check(params.finalState, "finalState", STATE_TYPE_SUPPORT_LIST)) return false;
    if (params.finalState->GetDataType() != params.initialStateRef->GetDataType()) {
        ReportInvalid(params, "finalState", "npu_recurrent_kda: finalState dtype must match initialStateRef.");
        return false;
    }
    if (!check(params.cuSeqlensOptional, "cu_seqlens", INT_TYPE_SUPPORT_LIST)) return false;
    if (!check(params.ssmStateIndicesOptional, "ssm_state_indices", INT_TYPE_SUPPORT_LIST)) return false;
    if (!check(params.aLogOptional, "A_log", F32_TYPE_SUPPORT_LIST)) return false;
    if (!check(params.dtBiasOptional, "dt_bias", F32_TYPE_SUPPORT_LIST)) return false;
    if (!check(params.numAcceptedTokensOptional, "num_accepted_tokens", INT_TYPE_SUPPORT_LIST)) return false;
    return true;
}

aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    OP_CHECK_IF(tensor == nullptr,
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "DataContiguous/CreateView returned null; returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
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
    OP_CHECK_IF(view == nullptr,
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "CreateViewIfNonContiguous: CreateView returned null; returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    tensor = view;
    return ACLNN_SUCCESS;
}

// Deprecated shape aliases are normalized outside the canonical cu-only mode
// selection. No new attribute or kernel layout branch is introduced.
aclnnStatus NormalizePublishedShapes(RecurrentKdaParams &params, aclOpExecutor *executor)
{
    const size_t rank = Rank(params.query);
    const bool addBatch = rank == 3 && params.cuSeqlensOptional == nullptr;
    const bool flatten = rank == 4 && params.cuSeqlensOptional != nullptr;
    if (!addBatch && !flatten) return ACLNN_SUCCESS;
    const size_t prefix = addBatch ? 1 : 2;
    bool valid = SameShape(params.query, params.key) && SameShape(params.value, params.attnOut) &&
                 Rank(params.value) == rank && Rank(params.gate) == rank && Rank(params.beta) == rank - 1;
    if (valid) {
        for (const auto *tensor : {params.value, params.gate, params.beta}) {
            for (size_t i = 0; i < prefix; ++i) valid = valid && Dim(tensor, i) == Dim(params.query, i);
        }
    }
    if (!valid) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentKda", "query/key/value/gate/beta/attnOut",
            "legacy shape alias requires matching ranks, query/key shapes, token dimensions and value/attnOut shapes");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (flatten && (Dim(params.query, 0) <= 0 || Dim(params.query, 1) <= 0 ||
        Dim(params.query, 0) > std::numeric_limits<int64_t>::max() / Dim(params.query, 1))) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentKda", "query",
            "legacy B*T must be positive and representable; B=" + std::to_string(Dim(params.query, 0)) +
            ", T=" + std::to_string(Dim(params.query, 1)));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (flatten && !IsContiguous(params.attnOut)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentKda", "attnOut",
            "legacy rank4 packed output must be contiguous before flattening");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto reshape = [&](const aclTensor *&tensor, bool output) -> aclnnStatus {
        if (flatten && !output) {
            const auto status = DataContiguous(tensor, executor);
            if (status != ACLNN_SUCCESS) return status;
        }
        op::Shape shape;
        op::Strides strides;
        if (addBatch) {
            shape.AppendDim(1);
            strides.push_back(1); // A singleton batch has no address increment.
            for (size_t i = 0; i < Rank(tensor); ++i) {
                shape.AppendDim(Dim(tensor, i));
                strides.push_back(tensor->GetViewStrides()[i]);
            }
        } else {
            shape.AppendDim(Dim(tensor, 0) * Dim(tensor, 1));
            strides.push_back(tensor->GetViewStrides()[1]);
            for (size_t i = 2; i < Rank(tensor); ++i) {
                shape.AppendDim(Dim(tensor, i));
                strides.push_back(tensor->GetViewStrides()[i]);
            }
        }
        tensor = executor->CreateView(tensor, shape, tensor->GetStorageShape(), strides, tensor->GetViewOffset());
        OP_CHECK_IF(tensor == nullptr,
            OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
                "DataContiguous/CreateView returned null; returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
            return ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    };
    for (auto *tensor : {&params.query, &params.key, &params.value, &params.gate, &params.beta}) {
        const auto status = reshape(*tensor, false);
        if (status != ACLNN_SUCCESS) return status;
    }
    const auto status = reshape(params.attnOut, true);
    if (status != ACLNN_SUCCESS) return status;
    OP_LOGW("aclnnRecurrentKda", "Deprecated rank/cu_seqlens shape alias; supported through v26.12, "
            "removal no earlier than v27.1. Use rank4 without cu_seqlens or rank3 with cu_seqlens; layout is ignored.");
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

// GetWorkspaceSize has no producer-stream argument. Device metadata must be
// ready on entry (Python waits its framework stream). A synchronous D2H waits
// for this copy; do not add a device-wide producer barrier here.
class MetadataTransfer {
public:
    aclnnStatus Copy(void *destination, const void *source, uint64_t bytes, const char *name) const
    {
        const auto ret = aclrtMemcpy(destination, bytes, source, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        OP_CHECK_IF(ret != ACL_SUCCESS,
            OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
                "D2H for %s metadata failed; bytes=%lu, runtime status=%d.", name, bytes, ret),
            return ACLNN_ERR_RUNTIME_ERROR);
        return ACLNN_SUCCESS;
    }
};

aclnnStatus MaterializeCuSeqlens(RecurrentKdaParams &params, aclOpExecutor *executor, MetadataTransfer &transfer)
{
    const auto *cu = params.cuSeqlensOptional;
    if (cu == nullptr) return ACLNN_SUCCESS;
    const int64_t count = cu->GetViewShape().GetDim(0);
    const int64_t stride = cu->GetViewStrides()[0];
    const uint64_t width = cu->GetDataType() == DataType::DT_INT32 ? 4 : 8;
    const int64_t offset = cu->GetViewOffset();
    const uint64_t maxElements = std::numeric_limits<uint64_t>::max() / width;
    if (stride < 0 || offset < 0 || uint64_t(offset) >= maxElements ||
        uint64_t(count - 1) > (maxElements - uint64_t(offset) - 1) / std::max<int64_t>(stride, 1)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentKda", "cu_seqlens",
            "nonnegative stride/offset and representable storage span required; count=" +
            std::to_string(count) + ", stride=" + std::to_string(stride) + ", offset=" + std::to_string(offset));
        return ACLNN_ERR_PARAM_INVALID;
    }
    const uint64_t bytes = (uint64_t(count - 1) * uint64_t(stride) + 1) * width;
    std::vector<uint8_t> storage(bytes);
    const auto *source = ViewData(cu, width);
    if (gert::TensorPlacementUtils::IsOnHost(cu->GetPlacement())) {
        std::memcpy(storage.data(), source, bytes);
    } else {
        const auto status = transfer.Copy(storage.data(), source, bytes, "cu_seqlens");
        if (status != ACLNN_SUCCESS) return status;
    }
    const int64_t capacity = Dim(params.query, 0);
    int64_t previous = 0;
    for (int64_t i = 0; i < count; ++i) {
        int64_t current = 0;
        if (width == 4) {
            int32_t value = 0;
            std::memcpy(&value, storage.data() + uint64_t(i) * stride * width, sizeof(value));
            current = value;
        } else {
            std::memcpy(&current, storage.data() + uint64_t(i) * stride * width, sizeof(current));
        }
        if ((i == 0 && current != 0) || current < previous || current > capacity) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentKda", "cu_seqlens",
                "expected first=0 and monotonic offsets within token capacity; index=" + std::to_string(i) +
                ", previous=" + std::to_string(previous) + ", actual=" + std::to_string(current) +
                ", capacity=" + std::to_string(capacity));
            return ACLNN_ERR_PARAM_INVALID;
        }
        previous = current;
    }
    auto *host = executor->AllocHostTensor(cu->GetViewShape(), cu->GetDataType());
    OP_CHECK_IF(host == nullptr,
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "AllocHostTensor returned null; returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    auto *destination = static_cast<uint8_t *>(host->GetData());
    for (int64_t i = 0; i < count; ++i)
        std::memcpy(destination + i * width, storage.data() + uint64_t(i) * stride * width, width);
    params.cuSeqlensOptional = host;
    return ACLNN_SUCCESS;
}

// Describe only live entries: padding tokens and empty requests are not read.
struct MetadataSlice {
    uint64_t sourceOffset, destinationOffset, count, stride;
};

int64_t IntegerMetadataValue(const aclTensor *tensor, uint64_t index)
{
    const uint64_t width = tensor->GetDataType() == DataType::DT_INT32 ? sizeof(int32_t) : sizeof(int64_t);
    const auto *data = ViewData(tensor, width);
    if (tensor->GetDataType() == DataType::DT_INT32) {
        int32_t value;
        std::memcpy(&value, data + index * sizeof(value), sizeof(value));
        return value;
    }
    int64_t value;
    std::memcpy(&value, data + index * sizeof(value), sizeof(value));
    return value;
}

aclnnStatus SnapshotMetadata(const aclTensor *&tensor, const char *name,
                            const std::vector<MetadataSlice> &slices,
                            aclOpExecutor *executor, MetadataTransfer &transfer)
{
    if (tensor == nullptr) return ACLNN_SUCCESS;
    const uint64_t width = tensor->GetDataType() == DataType::DT_INT32 ? 4 : 8;
    const uint64_t limit = std::numeric_limits<uint64_t>::max() / width;
    uint64_t elements = 1, span = 1;
    bool representable = tensor->GetViewOffset() >= 0;
    for (size_t d = 0; d < tensor->GetViewShape().GetDimNum(); ++d) {
        const uint64_t size = tensor->GetViewShape().GetDim(d);
        const int64_t stride = tensor->GetViewStrides()[d];
        if (size == 0 || stride < 0 || elements > limit / size ||
            (size > 1 && uint64_t(stride) > (limit - span) / (size - 1))) {
            representable = false;
            break;
        }
        elements *= size;
        span += (size - 1) * uint64_t(stride);
    }
    if (!representable || uint64_t(tensor->GetViewOffset()) > limit - span) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentKda", name,
            "nonnegative strides/offset and representable metadata span required; actual " + TensorDescription(tensor));
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto *host = executor->AllocHostTensor(tensor->GetViewShape(), tensor->GetDataType());
    OP_CHECK_IF(host == nullptr,
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "AllocHostTensor returned null; returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    auto *destination = static_cast<uint8_t *>(host->GetData());
    std::memset(destination, 0, elements * width);
    const auto *source = ViewData(tensor, width);
    const bool onHost = gert::TensorPlacementUtils::IsOnHost(tensor->GetPlacement());
    for (const auto &slice : slices) {
        if (slice.count == 0) continue;
        const uint64_t bytes = ((slice.count - 1) * slice.stride + 1) * width;
        std::vector<uint8_t> storage(bytes);
        if (onHost) std::memcpy(storage.data(), source + slice.sourceOffset * width, bytes);
        else {
            const auto status = transfer.Copy(storage.data(), source + slice.sourceOffset * width, bytes, name);
            if (status != ACLNN_SUCCESS) return status;
        }
        for (uint64_t i = 0; i < slice.count; ++i)
            std::memcpy(destination + (slice.destinationOffset + i) * width,
                        storage.data() + i * slice.stride * width, width);
    }
    tensor = host;
    return ACLNN_SUCCESS;
}

aclnnStatus MaterializeStateMetadata(RecurrentKdaParams &params, aclOpExecutor *executor, MetadataTransfer &transfer)
{
    const auto *indices = params.ssmStateIndicesOptional;
    if (indices == nullptr) return ACLNN_SUCCESS;
    const bool packed = params.cuSeqlensOptional != nullptr;
    const int64_t requests = packed ? Dim(params.cuSeqlensOptional, 0) - 1 : Dim(params.query, 0);
    const int64_t denseLength = packed ? 0 : Dim(params.query, 1);
    const bool speculative = Rank(indices) == 2;
    std::vector<MetadataSlice> indexSlices, acceptedSlices;
    std::vector<int64_t> begins(requests), lengths(requests);
    for (int64_t b = 0; b < requests; ++b) {
        const int64_t begin = packed ? IntegerMetadataValue(params.cuSeqlensOptional, b) : b * denseLength;
        const int64_t end = packed ? IntegerMetadataValue(params.cuSeqlensOptional, b + 1) : begin + denseLength;
        begins[b] = begin;
        lengths[b] = end - begin;
        if (begin == end) continue;
        if (speculative && end - begin > Dim(indices, 1)) {
            ReportInvalid(params, "ssm_state_indices", "request=%ld, actual length=%ld exceeds max_step=%ld.",
                          b, end - begin, Dim(indices, 1));
            return ACLNN_ERR_PARAM_INVALID;
        }
        const uint64_t sourceOffset = speculative ? uint64_t(b) * uint64_t(indices->GetViewStrides()[0]) :
                                                  uint64_t(begin) * uint64_t(indices->GetViewStrides()[0]);
        const uint64_t targetOffset = speculative ? b * Dim(indices, 1) : begin;
        indexSlices.push_back({sourceOffset, targetOffset, uint64_t(end - begin),
                              uint64_t(indices->GetViewStrides()[speculative ? 1 : 0])});
        if (params.numAcceptedTokensOptional != nullptr)
            acceptedSlices.push_back({uint64_t(b) * uint64_t(params.numAcceptedTokensOptional->GetViewStrides()[0]),
                                      uint64_t(b), 1, 1});
    }
    auto status = SnapshotMetadata(params.ssmStateIndicesOptional, "ssm_state_indices", indexSlices,
                                   executor, transfer);
    if (status != ACLNN_SUCCESS) return status;
    status = SnapshotMetadata(params.numAcceptedTokensOptional, "num_accepted_tokens", acceptedSlices,
                              executor, transfer);
    if (status != ACLNN_SUCCESS) return status;
    const int64_t capacity = Dim(params.initialStateRef, 0);
    for (int64_t b = 0; b < requests; ++b) {
        if (lengths[b] == 0) continue;
        if (params.numAcceptedTokensOptional != nullptr) {
            const int64_t accepted = IntegerMetadataValue(params.numAcceptedTokensOptional, b);
            if (accepted <= 0 || accepted > lengths[b]) {
                ReportInvalid(params, "num_accepted_tokens", "request=%ld, actual=%ld, expected [1,length=%ld].",
                              b, accepted, lengths[b]);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        for (int64_t t = 0; t < lengths[b]; ++t) {
            const uint64_t offset = speculative ? b * Dim(params.ssmStateIndicesOptional, 1) + t : begins[b] + t;
            const int64_t slot = IntegerMetadataValue(params.ssmStateIndicesOptional, offset);
            if (slot < 0 || slot >= capacity) {
                ReportInvalid(params, "ssm_state_indices", "request=%ld, token=%ld, actual=%ld, expected [0,capacity=%ld).",
                              b, t, slot, capacity);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    return ACLNN_SUCCESS;
}

// Plain ND/rank-inferred formats share storage. Copy only the logical extent,
// while retaining the destination's offset, outer strides and format tags.
aclnnStatus CopyStateView(const aclTensor *source, const aclTensor *destination, aclOpExecutor *executor)
{
    source = l0op::Contiguous(source, executor);
    OP_CHECK_IF(source == nullptr,
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "CopyStateView: Contiguous returned null; returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    auto *copySource = executor->CreateView(source, source->GetViewShape(), source->GetViewOffset());
    OP_CHECK_IF(copySource == nullptr,
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "CopyStateView: CreateView returned null; returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    copySource->SetStorageShape(source->GetViewShape());
    copySource->SetOriginalShape(source->GetViewShape());
    copySource->SetStorageFormat(destination->GetStorageFormat());
    copySource->SetViewFormat(destination->GetViewFormat());
    copySource->SetOriginalFormat(destination->GetOriginalFormat());
    OP_CHECK_IF(!(l0op::ViewCopy(copySource, destination, executor) != nullptr),
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "l0op::ViewCopy(copySource, destination, executor) != nullptr failed; returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus PreProcess(
    RecurrentKdaParams &params,
    bool queryNeedsContiguous,
    bool keyNeedsContiguous,
    bool valueNeedsContiguous,
    aclOpExecutor *executor)
{
    SetInputOriginalShape(params);
    OP_CHECK_IF(!(DataContiguousIfNeeded(params.query, queryNeedsContiguous, executor) == ACLNN_SUCCESS),
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "DataContiguousIfNeeded(params.query, queryNeedsContiguous, executor) == ACLNN_SUCCESS "
            "failed; returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_IF(!(DataContiguousIfNeeded(params.key, keyNeedsContiguous, executor) == ACLNN_SUCCESS),
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "DataContiguousIfNeeded(params.key, keyNeedsContiguous, executor) == ACLNN_SUCCESS failed; "
            "returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_IF(!(DataContiguousIfNeeded(params.value, valueNeedsContiguous, executor) == ACLNN_SUCCESS),
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "DataContiguousIfNeeded(params.value, valueNeedsContiguous, executor) == ACLNN_SUCCESS "
            "failed; returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_IF(!(DataContiguous(params.gate, executor) == ACLNN_SUCCESS),
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "DataContiguous(params.gate, executor) == ACLNN_SUCCESS failed; returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_IF(!(DataContiguous(params.beta, executor) == ACLNN_SUCCESS),
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "DataContiguous(params.beta, executor) == ACLNN_SUCCESS failed; returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_IF(!(DataContiguous(params.cuSeqlensOptional, executor) == ACLNN_SUCCESS),
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "DataContiguous(params.cuSeqlensOptional, executor) == ACLNN_SUCCESS failed; returning "
            "status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_IF(!(DataContiguous(params.ssmStateIndicesOptional, executor) == ACLNN_SUCCESS),
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "DataContiguous(params.ssmStateIndicesOptional, executor) == ACLNN_SUCCESS failed; "
            "returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_IF(!(DataContiguous(params.aLogOptional, executor) == ACLNN_SUCCESS),
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "DataContiguous(params.aLogOptional, executor) == ACLNN_SUCCESS failed; returning "
            "status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_IF(!(DataContiguous(params.dtBiasOptional, executor) == ACLNN_SUCCESS),
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "DataContiguous(params.dtBiasOptional, executor) == ACLNN_SUCCESS failed; returning "
            "status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_IF(!(DataContiguous(params.numAcceptedTokensOptional, executor) == ACLNN_SUCCESS),
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "DataContiguous(params.numAcceptedTokensOptional, executor) == ACLNN_SUCCESS failed; "
            "returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    if (params.ssmStateIndicesOptional == nullptr && params.numAcceptedTokensOptional != nullptr) {
        ReportInvalid(params, "num_accepted_tokens",
                "npu_recurrent_kda: num_accepted_tokens requires ssm_state_indices.");
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
    // Device tensor descriptors do not include their values in the L2 cache
    // key. Packed A5 tiling depends on those values and owns a host snapshot,
    // so this path must rebuild instead of returning a cached executor.
    OP_CHECK_IF(executor == nullptr,
        OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentKdaGetWorkspaceSize", "executor"),
        return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_IF(workspaceSize == nullptr,
        OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentKdaGetWorkspaceSize", "workspaceSize"),
        return ACLNN_ERR_PARAM_NULLPTR);
    const bool valueDependent = ssmStateIndicesOptional != nullptr || cuSeqlensOptional != nullptr;
    L2_DFX_PHASE_1_WITHOUT_CACHE(aclnnRecurrentKda,
                   DFX_IN(query, key, value, gate, beta, initialStateRef, cuSeqlensOptional,
                          ssmStateIndicesOptional, aLogOptional, dtBiasOptional, numAcceptedTokensOptional,
                          layout, scale, outputFinalState, inplaceFinalState, useQkL2normInKernel,
                          useGateInKernel, useBetaSigmoidInKernel, allowNegEigval, safeGate, lowerBound,
                          stateVFirst),
                   DFX_OUT(initialStateRef, attnOut, finalState));

    if (!valueDependent && op::internal::GetFromCache(executor, workspaceSize, "aclnnRecurrentKda",
            DFX_IN(query, key, value, gate, beta, initialStateRef, cuSeqlensOptional,
                   ssmStateIndicesOptional, aLogOptional, dtBiasOptional, numAcceptedTokensOptional,
                   layout, scale, outputFinalState, inplaceFinalState, useQkL2normInKernel,
                   useGateInKernel, useBetaSigmoidInKernel, allowNegEigval, safeGate, lowerBound, stateVFirst),
            DFX_OUT(initialStateRef, attnOut, finalState))) {
        return ACLNN_SUCCESS;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    OP_CHECK_IF(!(uniqueExecutor.get() != nullptr),
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "uniqueExecutor.get() != nullptr failed; returning status=%d.", ACLNN_ERR_INNER_CREATE_EXECUTOR),
        return ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    if (valueDependent) executorPtr->AbandonCache();

    RecurrentKdaParams params{query, key, value, gate, beta, initialStateRef, cuSeqlensOptional,
                              ssmStateIndicesOptional, aLogOptional, dtBiasOptional,
                              numAcceptedTokensOptional, layout, scale, outputFinalState, inplaceFinalState,
                              useQkL2normInKernel, useGateInKernel, useBetaSigmoidInKernel,
                              allowNegEigval, safeGate, lowerBound, stateVFirst, attnOut, finalState};

    // Ignore inactive raw-gate inputs before validation or materialization.
    if (params.aLogOptional == nullptr) {
        params.dtBiasOptional = nullptr;
        params.safeGate = false;
        params.lowerBound = -5.0;
    }
    if (!CheckNotNull(params)) return ACLNN_ERR_PARAM_INVALID;
    if (!CheckDtypeValid(params)) return ACLNN_ERR_PARAM_INVALID;
    const auto normalizeStatus = NormalizePublishedShapes(params, executorPtr);
    if (normalizeStatus != ACLNN_SUCCESS) return normalizeStatus;
    // `cu_seqlens` is the only mode discriminator.  `layout` remains a
    // compatibility argument and does not affect rank or address semantics.
    const bool varlen = params.cuSeqlensOptional != nullptr;
    const RecurrentKdaLayout parsedLayout = varlen ? RecurrentKdaLayout::TND : RecurrentKdaLayout::BSND;
    if (!CheckShape(params, varlen)) return ACLNN_ERR_PARAM_INVALID;
    if (!CheckQkvMaterializationSafety(params.query, parsedLayout, "query") ||
        !CheckQkvMaterializationSafety(params.key, parsedLayout, "key") ||
        !CheckQkvMaterializationSafety(params.value, parsedLayout, "value"))
        return ACLNN_ERR_PARAM_INVALID;
    const bool querySupportsDirect = IsSupportedQkvView(params.query, parsedLayout);
    const bool keySupportsDirect = IsSupportedQkvView(params.key, parsedLayout);
    const bool valueSupportsDirect = IsSupportedQkvView(params.value, parsedLayout);
    const bool queryUsesDirectView = !IsContiguous(params.query) && querySupportsDirect;
    const bool keyUsesDirectView = !IsContiguous(params.key) && keySupportsDirect;
    const bool valueUsesDirectView = !IsContiguous(params.value) && valueSupportsDirect;
    const bool queryNeedsContiguous = !querySupportsDirect;
    const bool keyNeedsContiguous = !keySupportsDirect;
    const bool valueNeedsContiguous = !valueSupportsDirect;
    MetadataTransfer metadataTransfer;
    const auto cuStatus = MaterializeCuSeqlens(params, executorPtr, metadataTransfer);
    if (cuStatus != ACLNN_SUCCESS) return cuStatus;
    const auto metadataStatus = MaterializeStateMetadata(params, executorPtr, metadataTransfer);
    if (metadataStatus != ACLNN_SUCCESS) return metadataStatus;
    OP_CHECK_IF(!(PreProcess(params, queryNeedsContiguous, keyNeedsContiguous, valueNeedsContiguous, executorPtr) ==
        ACLNN_SUCCESS),
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "PreProcess(params, queryNeedsContiguous, keyNeedsContiguous, valueNeedsContiguous, "
            "executorPtr) == ACLNN_SUCCESS failed; returning status=%d.", ACLNN_ERR_PARAM_INVALID),
        return ACLNN_ERR_PARAM_INVALID);
    if (queryUsesDirectView) {
        OP_CHECK_IF(!(CreateViewIfNonContiguous(params.query, executorPtr) == ACLNN_SUCCESS),
            OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
                "CreateViewIfNonContiguous(params.query, executorPtr) == ACLNN_SUCCESS failed; returning "
                "status=%d.", ACLNN_ERR_INNER_NULLPTR),
            return ACLNN_ERR_INNER_NULLPTR);
    }
    if (keyUsesDirectView) {
        OP_CHECK_IF(!(CreateViewIfNonContiguous(params.key, executorPtr) == ACLNN_SUCCESS),
            OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
                "CreateViewIfNonContiguous(params.key, executorPtr) == ACLNN_SUCCESS failed; returning "
                "status=%d.", ACLNN_ERR_INNER_NULLPTR),
            return ACLNN_ERR_INNER_NULLPTR);
    }
    if (valueUsesDirectView) {
        OP_CHECK_IF(!(CreateViewIfNonContiguous(params.value, executorPtr) == ACLNN_SUCCESS),
            OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
                "CreateViewIfNonContiguous(params.value, executorPtr) == ACLNN_SUCCESS failed; returning "
                "status=%d.", ACLNN_ERR_INNER_NULLPTR),
            return ACLNN_ERR_INNER_NULLPTR);
    }

    aclTensor *initialStateForKernel = params.initialStateRef;
    if (!IsContiguous(initialStateForKernel)) {
        initialStateForKernel = executorPtr->CreateView(
            initialStateForKernel,
            initialStateForKernel->GetViewShape(),
            initialStateForKernel->GetStorageShape(),
            initialStateForKernel->GetViewStrides(),
            initialStateForKernel->GetViewOffset());
        OP_CHECK_IF(initialStateForKernel == nullptr,
            OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
                "initialStateRef: CreateView returned null; returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
            return ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *finalStateForKernel = params.finalState;
    if (!IsContiguous(finalStateForKernel)) {
        finalStateForKernel = executorPtr->CreateView(
            finalStateForKernel,
            finalStateForKernel->GetViewShape(),
            finalStateForKernel->GetStorageShape(),
            finalStateForKernel->GetViewStrides(),
            finalStateForKernel->GetViewOffset());
        OP_CHECK_IF(finalStateForKernel == nullptr,
            OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
                "finalState: CreateView returned null; returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
            return ACLNN_ERR_INNER_NULLPTR);
    }

    // Empty requests and untouched pool slots must preserve their initial value
    // on every public entry, including direct ACLNN and ctypes. Enqueue the copy
    // before recurrence; the executor orders both writes to the same output.
    if (!params.inplaceFinalState && params.outputFinalState &&
        SameTensorStorage(params.initialStateRef, params.finalState)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            "aclnnRecurrentKda", "finalState",
            "non-inplace output must use storage independent from initialStateRef");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!params.inplaceFinalState && params.outputFinalState &&
        !SameTensorView(params.initialStateRef, params.finalState)) {
        OP_CHECK_IF(!(CopyStateView(initialStateForKernel, finalStateForKernel, executorPtr) == ACLNN_SUCCESS),
            OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
                "CopyStateView(initialStateForKernel, finalStateForKernel, executorPtr) == ACLNN_SUCCESS "
                "failed; returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
            return ACLNN_ERR_INNER_NULLPTR);
    }
    auto result = l0op::RecurrentKda(
        params.query, params.key, params.value, params.gate, params.beta, initialStateForKernel,
        params.cuSeqlensOptional, params.ssmStateIndicesOptional, params.aLogOptional,
        params.dtBiasOptional, params.numAcceptedTokensOptional,
        params.layout == nullptr ? "BSND" : params.layout, params.scale,
        params.outputFinalState, params.inplaceFinalState, params.useQkL2normInKernel,
        params.aLogOptional != nullptr, params.useBetaSigmoidInKernel, params.allowNegEigval,
        params.safeGate, params.lowerBound, params.stateVFirst, params.attnOut,
        finalStateForKernel, executorPtr);
    OP_CHECK_IF(!(result[0] != nullptr && result[1] != nullptr && result[2] != nullptr),
        OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
            "result[0] != nullptr && result[1] != nullptr && result[2] != nullptr failed; returning "
            "status=%d.", ACLNN_ERR_INNER_NULLPTR),
        return ACLNN_ERR_INNER_NULLPTR);
    if (params.inplaceFinalState && params.outputFinalState &&
        !SameTensorView(params.finalState, params.initialStateRef)) {
        OP_CHECK_IF(!(CopyStateView(result[1], finalStateForKernel, executorPtr) == ACLNN_SUCCESS),
            OP_LOGE_LIBOPAPI_REPORT("aclnnRecurrentKda",
                "CopyStateView(result[1], finalStateForKernel, executorPtr) == ACLNN_SUCCESS failed; "
                "returning status=%d.", ACLNN_ERR_INNER_NULLPTR),
            return ACLNN_ERR_INNER_NULLPTR);
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
