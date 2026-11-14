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
#include <limits>
#include <cstdio>
#include <cstdarg>
#include <utility>
#include <string>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static constexpr const char *OP_NAME = "aclnnRecurrentKda";
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
        "; layout=" + (params.layout == nullptr ? std::string("nullptr") : std::string(params.layout)) +
        "; scale=" + std::to_string(params.scale) +
        "; output_final_state=" + std::to_string(params.outputFinalState) +
        "; inplace_final_state=" + std::to_string(params.inplaceFinalState) +
        "; use_qk_l2norm_in_kernel=" + std::to_string(params.useQkL2normInKernel) +
        "; use_gate_in_kernel=" + std::to_string(params.useGateInKernel) +
        "; use_beta_sigmoid_in_kernel=" + std::to_string(params.useBetaSigmoidInKernel) +
        "; allow_neg_eigval=" + std::to_string(params.allowNegEigval) +
        "; safe_gate=" + std::to_string(params.safeGate) +
        "; lower_bound=" + std::to_string(params.lowerBound) +
        "; state_v_first=" + std::to_string(params.stateVFirst);
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



// All platforms use the same token-major direct-view contract.
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
    const aclTensor *tensor, RecurrentKdaLayout layout, const char *name, bool &supportsDirect)
{
    supportsDirect = false;
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
    if ((headCount > 1 && headStride > (maxInt64 - featureCount) / (headCount - 1)) ||
        tokenStride < (headCount - 1) * headStride + featureCount) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentKda", name,
            "expected representable token-major head span; actual " + TensorDescription(tensor));
        return false;
    }
    supportsDirect = IsSupportedQkvView(tensor, layout);
    return true;
}

bool CheckCuSeqlensShape(const aclTensor *cuSeqlens, const char *opName)
{
    if (cuSeqlens == nullptr) {
        return true;
    }
    if (Rank(cuSeqlens) != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "cu_seqlens", std::to_string(Rank(cuSeqlens)), "1");
        return false;
    }
    if (Dim(cuSeqlens, DIM0) < 2) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "cu_seqlens",
            op::ToString(cuSeqlens->GetViewShape()).GetString(), "expected at least two elements");
        return false;
    }
    return true;
}

bool CheckShape(const RecurrentKdaParams &params, RecurrentKdaLayout layout, bool isA5)
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
    if (layout == RecurrentKdaLayout::TND) {
        if (Rank(params.query) != 3 || Rank(params.value) != 3 || Rank(params.gate) != 3 || Rank(params.beta) != 2) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON("aclnnRecurrentKda", "query/value/gate/beta",
                std::to_string(Rank(params.query)) + "/" + std::to_string(Rank(params.value)) + "/" +
                    std::to_string(Rank(params.gate)) + "/" + std::to_string(Rank(params.beta)),
                "TND expects q/k [T,H,K], v [T,HV,V], g [T,HV,K], beta [T,HV]");
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
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON("aclnnRecurrentKda", "query/value/gate/beta",
                std::to_string(Rank(params.query)) + "/" + std::to_string(Rank(params.value)) + "/" +
                    std::to_string(Rank(params.gate)) + "/" + std::to_string(Rank(params.beta)),
                "BSND expects q/k [B,T,H,K], v [B,T,HV,V], g [B,T,HV,K], beta [B,T,HV]");
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
    int64_t seqNum = params.cuSeqlensOptional != nullptr ?
        Dim(params.cuSeqlensOptional, DIM0) - 1 : batch;
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
    if (isA5) {
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
    if (isA5 && params.ssmStateIndicesOptional == nullptr && params.numAcceptedTokensOptional != nullptr) {
        ReportInvalid(params, "num_accepted_tokens",
                "npu_recurrent_kda: num_accepted_tokens requires ssm_state_indices.");
        return false;
    }
    if (isA5 && !params.inplaceFinalState && params.outputFinalState &&
        SameTensorStorage(params.initialStateRef, params.finalState)) {
        ReportInvalid(params, "finalState",
                "non-inplace output must use storage independent from initialStateRef");
        return false;
    }
    if (params.numAcceptedTokensOptional != nullptr &&
        (Rank(params.numAcceptedTokensOptional) != 1 || Dim(params.numAcceptedTokensOptional, DIM0) != seqNum)) {
        ReportInvalid(params, "num_accepted_tokens", "npu_recurrent_kda: num_accepted_tokens length must equal sequence number.");
        return false;
    }
    if (params.useGateInKernel && params.aLogOptional == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentKda", "A_log");
        return false;
    }
    if (!params.useGateInKernel &&
        (params.aLogOptional || params.dtBiasOptional || params.safeGate)) {
        ReportInvalid(params, "use_gate_in_kernel",
            "A_log, dt_bias and safe_gate require use_gate_in_kernel=True.");
        return false;
    }
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
    if (params.safeGate && (isA5 ? !(params.lowerBound >= -5.0 && params.lowerBound < 0.0) :
                                      (params.lowerBound < -5.0 || params.lowerBound >= 0.0))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnRecurrentKda", "lower_bound",
            std::to_string(params.lowerBound), "must be in [-5, 0) when safe_gate=True");
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
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentKda", item.second, "input argument must not be nullptr");
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
        OP_LOGE_FOR_INVALID_DTYPE("aclnnRecurrentKda", "finalState",
            op::ToString(params.finalState->GetDataType()).GetString(),
            op::ToString(params.initialStateRef->GetDataType()).GetString());
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
    // The callee reports the root error; propagate without a second report.
    tensor = l0op::Contiguous(tensor, executor);
    if (tensor == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
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
    if (view == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    tensor = view;
    return ACLNN_SUCCESS;
}

aclnnStatus CopyStateView(const aclTensor *source, const aclTensor *destination, aclOpExecutor *executor)
{
    source = l0op::Contiguous(source, executor);
    if (source == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto *copySource = executor->CreateView(source, source->GetViewShape(), source->GetViewOffset());
    if (copySource == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    copySource->SetStorageShape(source->GetViewShape());
    copySource->SetOriginalShape(source->GetViewShape());
    copySource->SetStorageFormat(destination->GetStorageFormat());
    copySource->SetViewFormat(destination->GetViewFormat());
    copySource->SetOriginalFormat(destination->GetOriginalFormat());
    if (l0op::ViewCopy(copySource, destination, executor) == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
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
    if (DataContiguousIfNeeded(params.query, queryNeedsContiguous, executor) != ACLNN_SUCCESS) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguousIfNeeded(params.key, keyNeedsContiguous, executor) != ACLNN_SUCCESS) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguousIfNeeded(params.value, valueNeedsContiguous, executor) != ACLNN_SUCCESS) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.gate, executor) != ACLNN_SUCCESS) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.beta, executor) != ACLNN_SUCCESS) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.cuSeqlensOptional, executor) != ACLNN_SUCCESS) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.ssmStateIndicesOptional, executor) != ACLNN_SUCCESS) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.aLogOptional, executor) != ACLNN_SUCCESS) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.dtBiasOptional, executor) != ACLNN_SUCCESS) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (DataContiguous(params.numAcceptedTokensOptional, executor) != ACLNN_SUCCESS) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (params.ssmStateIndicesOptional == nullptr && params.numAcceptedTokensOptional != nullptr) {
        ReportInvalid(params, "num_accepted_tokens", "num_accepted_tokens requires ssm_state_indices.");
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
    OP_CHECK_IF(executor == nullptr,
        OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentKdaGetWorkspaceSize", "executor"),
        return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_IF(workspaceSize == nullptr,
        OP_LOGE_WITH_INVALID_INPUT("aclnnRecurrentKdaGetWorkspaceSize", "workspaceSize"),
        return ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_1(aclnnRecurrentKda,
                   DFX_IN(query, key, value, gate, beta, initialStateRef, cuSeqlensOptional,
                          ssmStateIndicesOptional, aLogOptional, dtBiasOptional, numAcceptedTokensOptional,
                          layout, scale, outputFinalState, inplaceFinalState, useQkL2normInKernel,
                          useGateInKernel, useBetaSigmoidInKernel, allowNegEigval, safeGate, lowerBound,
                          stateVFirst),
                   DFX_OUT(initialStateRef, attnOut, finalState));

    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto executorPtr = uniqueExecutor.get();

    RecurrentKdaParams params{query, key, value, gate, beta, initialStateRef, cuSeqlensOptional,
                              ssmStateIndicesOptional, aLogOptional, dtBiasOptional,
                              numAcceptedTokensOptional, layout, scale, outputFinalState, inplaceFinalState,
                              useQkL2normInKernel, useGateInKernel, useBetaSigmoidInKernel,
                              allowNegEigval, safeGate, lowerBound, stateVFirst, attnOut, finalState};

    if (!CheckNotNull(params)) return ACLNN_ERR_PARAM_INVALID;
    if (!CheckDtypeValid(params)) return ACLNN_ERR_PARAM_INVALID;
    if (params.layout == nullptr) {
        ReportInvalid(params, "layout", "layout must be BSND or TND.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    RecurrentKdaLayout parsedLayout = RecurrentKdaLayout::BSND;
    if (std::strcmp(params.layout, "BSND") != 0) {
        if (std::strcmp(params.layout, "TND") != 0) {
            ReportInvalid(params, "layout", "layout must be BSND or TND.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        parsedLayout = RecurrentKdaLayout::TND;
    }
    const bool isA5 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    if (!CheckShape(params, parsedLayout, isA5)) return ACLNN_ERR_PARAM_INVALID;
    bool querySupportsDirect = false, keySupportsDirect = false, valueSupportsDirect = false;
    if (!CheckQkvMaterializationSafety(params.query, parsedLayout, "query", querySupportsDirect) ||
        !CheckQkvMaterializationSafety(params.key, parsedLayout, "key", keySupportsDirect) ||
        !CheckQkvMaterializationSafety(params.value, parsedLayout, "value", valueSupportsDirect))
        return ACLNN_ERR_PARAM_INVALID;
    const bool queryUsesDirectView = !IsContiguous(params.query) && querySupportsDirect;
    const bool keyUsesDirectView = !IsContiguous(params.key) && keySupportsDirect;
    const bool valueUsesDirectView = !IsContiguous(params.value) && valueSupportsDirect;
    const bool queryNeedsContiguous = !querySupportsDirect;
    const bool keyNeedsContiguous = !keySupportsDirect;
    const bool valueNeedsContiguous = !valueSupportsDirect;
    const aclnnStatus preprocessStatus =
        PreProcess(params, queryNeedsContiguous, keyNeedsContiguous, valueNeedsContiguous, executorPtr);
    if (preprocessStatus != ACLNN_SUCCESS) {
        return isA5 ? preprocessStatus : ACLNN_ERR_PARAM_INVALID;
    }
    if (queryUsesDirectView) {
        if (CreateViewIfNonContiguous(params.query, executorPtr) != ACLNN_SUCCESS) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    if (keyUsesDirectView) {
        if (CreateViewIfNonContiguous(params.key, executorPtr) != ACLNN_SUCCESS) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    if (valueUsesDirectView) {
        if (CreateViewIfNonContiguous(params.value, executorPtr) != ACLNN_SUCCESS) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    aclTensor *initialStateForKernel = params.initialStateRef;
    if (!IsContiguous(initialStateForKernel)) {
        initialStateForKernel = executorPtr->CreateView(
            initialStateForKernel,
            initialStateForKernel->GetViewShape(),
            initialStateForKernel->GetStorageShape(),
            initialStateForKernel->GetViewStrides(),
            initialStateForKernel->GetViewOffset());
        if (initialStateForKernel == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *finalStateForKernel = params.finalState;
    if (!IsContiguous(finalStateForKernel)) {
        finalStateForKernel = executorPtr->CreateView(
            finalStateForKernel,
            finalStateForKernel->GetViewShape(),
            finalStateForKernel->GetStorageShape(),
            finalStateForKernel->GetViewStrides(),
            finalStateForKernel->GetViewOffset());
        if (finalStateForKernel == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    // Non-inplace recurrence reads the caller's initial state directly and
    // writes only active destinations. Untouched output slots are unspecified,
    // matching the Stable-ABI state contract; do not prefill the output pool.
    auto result = l0op::RecurrentKda(
        params.query, params.key, params.value, params.gate, params.beta, initialStateForKernel,
        params.cuSeqlensOptional, params.ssmStateIndicesOptional, params.aLogOptional,
        params.dtBiasOptional, params.numAcceptedTokensOptional, params.layout, params.scale,
        params.outputFinalState, params.inplaceFinalState, params.useQkL2normInKernel,
        params.useGateInKernel, params.useBetaSigmoidInKernel, params.allowNegEigval,
        params.safeGate, params.lowerBound, params.stateVFirst, params.attnOut,
        finalStateForKernel, executorPtr);
    // The launcher reports the root error. Do not dereference failed outputs or
    // return success when all outputs are null.
    if (result[0] == nullptr || result[1] == nullptr || result[2] == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (params.inplaceFinalState && params.outputFinalState &&
        !SameTensorView(params.finalState, params.initialStateRef)) {
        const bool copyFailed = isA5 ?
            CopyStateView(result[1], finalStateForKernel, executorPtr) != ACLNN_SUCCESS :
            l0op::ViewCopy(result[1], params.finalState, executorPtr) == nullptr;
        if (copyFailed) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
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
