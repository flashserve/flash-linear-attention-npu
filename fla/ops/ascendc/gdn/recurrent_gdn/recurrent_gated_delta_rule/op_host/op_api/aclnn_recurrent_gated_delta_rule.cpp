/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "../recurrent_gated_delta_rule_error.h"
#include "aclnn_recurrent_gated_delta_rule.h"
#include "recurrent_gated_delta_rule.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "op_common/log/log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace op;

namespace {
namespace rgdn_check {
struct Params {
    const aclTensor *q, *k, *v, *beta;
    aclTensor *state;
    const aclTensor *lengths, *indices, *g, *gk, *accepted, *aLog, *bias;
    float scale;
    bool norm, rawGate, sigmoid, twice;
    aclTensor *out;
};

inline uint64_t GetElementSize(const aclTensor *t)
{
    return t->GetDataType() == op::DataType::DT_BF16 ? 2 : 4;
}

inline bool CheckTypeValid(const aclTensor *t, std::initializer_list<op::DataType> allowed,
                 const char *name, const char *expected, const char *entity)
{
    if (!t || std::find(allowed.begin(), allowed.end(), t->GetDataType()) != allowed.end()) return true;
    OP_LOGE_FOR_INVALID_DTYPE(entity, name, ge::TypeUtils::DataTypeToSerialString(t->GetDataType()), expected);
    return false;
}

inline bool CheckShapeValid(const aclTensor *t, std::initializer_list<int64_t> expected, const char *name)
{
    constexpr const char *entity = "aclnnRecurrentGatedDeltaRule";
    if (!t) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(entity, name, "input argument must not be nullptr"); return false; }
    bool valid = t->GetViewShape().GetDimNum() == expected.size();
    size_t i = 0;
    if (valid) for (auto n : expected) valid = (t->GetViewShape().GetDim(i++) == n) && valid;
    if (valid) return true;
    std::ostringstream correct;
    for (auto n : expected) { if (correct.tellp() != std::streampos(0)) correct << ','; correct << n; }
    const std::string actual = rgdn_error::ShapeText(t->GetViewShape());
    OP_LOGE_FOR_INVALID_SHAPE(entity, name, actual.substr(1, actual.size() - 2), correct.str());
    return false;
}

// Validate every multiply/add before using an address or narrowing a DMA gap.
inline bool CheckViewBoundsValid(const aclTensor *t)
{
    if (!t) return true;
    const auto &shape = t->GetViewShape();
    const auto &stride = t->GetViewStrides();
    const auto &storageShape = t->GetStorageShape();
    uint64_t storage = 1;
    const uint64_t limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / GetElementSize(t);
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        int64_t n = storageShape.GetDim(i);
        if (n <= 0 || static_cast<uint64_t>(n) > limit / storage) return false;
        storage *= static_cast<uint64_t>(n);
    }
    const int64_t offset = t->GetViewOffset();
    if (offset < 0 || static_cast<uint64_t>(offset) >= storage || stride.size() != shape.GetDimNum()) return false;
    uint64_t last = static_cast<uint64_t>(offset);
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t n = shape.GetDim(i), s = stride[i];
        if (n <= 0 || s < 0) return false;
        if (n > 1 && static_cast<uint64_t>(s) > (storage - 1 - last) / static_cast<uint64_t>(n - 1)) return false;
        last += static_cast<uint64_t>(n - 1) * static_cast<uint64_t>(s);
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(t->GetStorageAddr());
    return base != 0 && (last + 1) <= (std::numeric_limits<uintptr_t>::max() - base) / GetElementSize(t);
}

using Interval = std::pair<uintptr_t, uintptr_t>;
inline Interval GetAddressRange(const aclTensor *t)
{
    uint64_t last = 0;
    for (size_t i = 0; i < t->GetViewShape().GetDimNum(); ++i)
        last += (t->GetViewShape().GetDim(i) - 1) * t->GetViewStrides()[i];
    uintptr_t begin = reinterpret_cast<uintptr_t>(t->GetStorageAddr()) + t->GetViewOffset() * GetElementSize(t);
    return {begin, begin + (last + 1) * GetElementSize(t)};
}

inline std::vector<Interval> GetAddressIntervals(const aclTensor *t)
{
    std::vector<std::pair<uint64_t, uint64_t>> axes;
    for (size_t i = 0; i < t->GetViewShape().GetDimNum(); ++i)
        if (t->GetViewShape().GetDim(i) > 1)
            axes.emplace_back(t->GetViewStrides()[i], t->GetViewShape().GetDim(i));
    std::sort(axes.begin(), axes.end());
    uint64_t span = 1;
    size_t dense = 0;
    while (dense < axes.size() && axes[dense].first == span) {
        span *= axes[dense].second;
        ++dense;
    }
    std::vector<uintptr_t> starts{GetAddressRange(t).first};
    for (size_t i = dense; i < axes.size(); ++i) {
        std::vector<uintptr_t> expanded;
        for (auto start : starts)
            for (uint64_t j = 0; j < axes[i].second; ++j)
                expanded.push_back(start + j * axes[i].first * GetElementSize(t));
        starts.swap(expanded);
    }
    std::sort(starts.begin(), starts.end());
    std::vector<Interval> result;
    for (auto start : starts) result.emplace_back(start, start + span * GetElementSize(t));
    return result;
}

inline bool HasAddressOverlap(const std::vector<Interval> &a, const std::vector<Interval> &b)
{
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (std::max(a[i].first, b[j].first) < std::min(a[i].second, b[j].second)) return true;
        if (a[i].second <= b[j].second) ++i; else ++j;
    }
    return false;
}

inline bool HasTensorOverlap(const aclTensor *a, const aclTensor *b)
{
    if (!a || !b) return false;
    auto x = GetAddressRange(a), y = GetAddressRange(b);
    if (x.second <= y.first || y.second <= x.first) return false;
    return HasAddressOverlap(GetAddressIntervals(a), GetAddressIntervals(b));
}

inline bool CheckMatrixStridesValid(const aclTensor *t, size_t rowAxis)
{
    for (auto stride : t->GetViewStrides()) if (stride <= 0) return false;
    const auto &s = t->GetViewStrides();
    return s[s.size() - 1] == 1 && s[rowAxis] >= 128 &&
        static_cast<uint64_t>(s[rowAxis] - 128) <= std::numeric_limits<uint32_t>::max() / GetElementSize(t);
}

inline aclnnStatus CheckParams(const Params &p)
{
    using op::DataType;
    constexpr const char *entity = "aclnnRecurrentGatedDeltaRule";
    const std::pair<const char *, const aclTensor *> required[] = {
        {"query",p.q},{"key",p.k},{"value",p.v},{"beta",p.beta},{"state",p.state},
        {"actual_seq_lengths",p.lengths},{"ssm_state_indices",p.indices},{"out",p.out}};
    for (const auto &input : required) {
        if (!input.second) { OP_LOGE_WITH_INVALID_INPUT(entity, input.first); return ACLNN_ERR_PARAM_NULLPTR; }
    }
    if (p.gk) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(entity, "gk", "gk is not supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!std::isfinite(p.scale)) {
        OP_LOGE_FOR_INVALID_VALUE(entity, "scaleValue", std::to_string(p.scale), "finite float32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (p.rawGate && (!p.g || !p.aLog)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(entity, "use_gate_in_kernel",
            std::string("true requires non-null g and A_log; g=") + (p.g ? "present" : "null") +
            ", A_log=" + (p.aLog ? "present" : "null"));
        return ACLNN_ERR_PARAM_INVALID;
    }
    struct TensorRule {
        const char *first;
        const aclTensor *second;
        std::initializer_list<DataType> allowed;
        const char *expected;
        size_t rank; // Zero means the separately checked one/two-dimensional indices.
    };
    const TensorRule tensors[] = {
        {"query",p.q,{DataType::DT_BF16},"[bfloat16]",3},
        {"key",p.k,{DataType::DT_BF16},"[bfloat16]",3},
        {"value",p.v,{DataType::DT_BF16},"[bfloat16]",3},
        {"out",p.out,{DataType::DT_BF16},"[bfloat16]",3},
        {"state",p.state,{DataType::DT_BF16,DataType::DT_FLOAT},"[bfloat16, float32]",4},
        {"beta",p.beta,{DataType::DT_BF16,DataType::DT_FLOAT},"[bfloat16, float32]",2},
        {"g",p.g,{DataType::DT_BF16,DataType::DT_FLOAT},"[bfloat16, float32]",2},
        {"A_log",p.aLog,{DataType::DT_FLOAT},"[float32]",1},
        {"dt_bias",p.bias,{DataType::DT_FLOAT},"[float32]",1},
        {"actual_seq_lengths",p.lengths,{DataType::DT_INT32},"[int32]",1},
        {"ssm_state_indices",p.indices,{DataType::DT_INT32},"[int32]",0},
        {"num_accepted_tokens",p.accepted,{DataType::DT_INT32},"[int32]",1}};
    for (const auto &input : tensors) {
        if (!CheckTypeValid(input.second,input.allowed,input.first,input.expected,entity)) return ACLNN_ERR_PARAM_INVALID;
        if (input.second && input.second->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
            OP_LOGE_FOR_INVALID_FORMAT(entity, input.first, "FRACTAL_NZ", "any format except FRACTAL_NZ");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!input.second || input.rank == 0) continue;
        const size_t rank = input.second->GetViewShape().GetDimNum();
        if (rank != input.rank) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(entity,input.first,std::to_string(rank),std::to_string(input.rank));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const int64_t t = p.q->GetViewShape().GetDim(0), hk = p.q->GetViewShape().GetDim(1);
    const int64_t hv = p.v->GetViewShape().GetDim(1), blocks = p.state->GetViewShape().GetDim(0);
    const int64_t b = p.lengths->GetViewShape().GetDim(0) - 1;
    for (const auto &dim : {std::make_pair("query.T",t),std::make_pair("query.HK",hk),
         std::make_pair("value.HV",hv),std::make_pair("state.Cstate",blocks),std::make_pair("actual_seq_lengths.B",b)}) {
        if (dim.second <= 0) {
            OP_LOGE_FOR_INVALID_VALUE(entity,dim.first,std::to_string(dim.second),">0");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (hv % hk || hv / hk > 16) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(entity,"query.HK,value.HV",
            std::to_string(hk)+","+std::to_string(hv),"HV/HK must be an integer in [1,16]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (b > std::numeric_limits<int64_t>::max() / hv) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(entity,"actual_seq_lengths.B,value.HV",
            std::to_string(b)+","+std::to_string(hv),"B*HV must fit int64 task addressing");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShapeValid(p.q,{t,hk,128},"query") || !CheckShapeValid(p.k,{t,hk,128},"key") || !CheckShapeValid(p.v,{t,hv,128},"value") ||
        !CheckShapeValid(p.beta,{t,hv},"beta") || !CheckShapeValid(p.state,{blocks,hv,128,128},"state") || !CheckShapeValid(p.out,{t,hv,128},"out")) return ACLNN_ERR_PARAM_INVALID;
    if (p.indices->GetViewShape().GetDimNum() == 1) {
        if (!CheckShapeValid(p.indices,{t},"ssm_state_indices")) return ACLNN_ERR_PARAM_INVALID;
    } else if (p.indices->GetViewShape().GetDimNum() == 2) {
        if (p.indices->GetViewShape().GetDim(0) != b || p.indices->GetViewShape().GetDim(1) <= 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(entity,"ssm_state_indices",rgdn_error::ShapeText(p.indices->GetViewShape()),
                "expected [B="+std::to_string(b)+",Ckpt] with Ckpt>0");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM(entity,"ssm_state_indices",std::to_string(p.indices->GetViewShape().GetDimNum()),"1 or 2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((p.g && !CheckShapeValid(p.g,{t,hv},"g")) || (p.aLog && !CheckShapeValid(p.aLog,{hv},"A_log")) ||
        (p.bias && !CheckShapeValid(p.bias,{hv},"dt_bias")) || (p.accepted && !CheckShapeValid(p.accepted,{b},"num_accepted_tokens"))) return ACLNN_ERR_PARAM_INVALID;
    for (const auto &input : tensors) {
        if (input.second && !CheckViewBoundsValid(input.second)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(entity,input.first,
                "view exceeds storage bounds, has invalid dimensions/strides, or overflows address arithmetic; shape="+
                rgdn_error::ShapeText(input.second->GetViewShape())+", strides="+
                rgdn_error::StrideText(input.second->GetViewStrides())+", offset="+std::to_string(input.second->GetViewOffset())+
                ", storage="+rgdn_error::ShapeText(input.second->GetStorageShape()));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    for (const auto &input : required) {
        if (input.second != p.q && input.second != p.k && input.second != p.v && input.second != p.state) continue;
        if (!CheckMatrixStridesValid(input.second,input.second == p.state ? 2 : 0)) {
            OP_LOGE_FOR_INVALID_STRIDE(entity,input.first,rgdn_error::StrideText(input.second->GetViewStrides()),
                input.second == p.state ? "positive strides, K stride=1, V stride>=128, row byte gap<=UINT32_MAX" :
                "positive strides, last stride=1, token stride>=128, token byte gap<=UINT32_MAX");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!op::IsContiguous(p.out)) {
        OP_LOGE_FOR_INVALID_STRIDE(entity,"out",rgdn_error::StrideText(p.out->GetViewStrides()),"contiguous [T,HV,128]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // Fast proof for padding and permutations; exact intervals admit legal
    // interleaved State views instead of treating an inconclusive proof as failure.
    std::vector<std::pair<uint64_t,uint64_t>> axes;
    const auto &shape = p.state->GetViewShape();
    for (size_t i=0; i<4; ++i) if (shape.GetDim(i)>1)
        axes.emplace_back(p.state->GetViewStrides()[i],shape.GetDim(i));
    std::sort(axes.begin(),axes.end());
    uint64_t span=1; bool disjoint=true;
    for (auto axis:axes) {
        if (axis.first<span) {disjoint=false; break;}
        span+=(axis.second-1)*axis.first;
    }
    if (!disjoint) {
        auto intervals=GetAddressIntervals(p.state);
        for (size_t i=1;i<intervals.size();++i) if(intervals[i-1].second>intervals[i].first) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(entity,"state",
                "writable State elements overlap; strides="+rgdn_error::StrideText(p.state->GetViewStrides()));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    for (const auto &input : tensors) {
        if (std::string(input.first) == "state" || std::string(input.first) == "out") continue;
        if (HasTensorOverlap(p.state,input.second) || HasTensorOverlap(p.out,input.second)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(entity,input.first,
                std::string("read-only input overlaps writable ")+(HasTensorOverlap(p.state,input.second) ? "state" : "out"));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (HasTensorOverlap(p.state,p.out)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(entity,"state,out","writable State and out storage overlap");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
} // namespace rgdn_check
} // namespace

extern "C" {

aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta,
    aclTensor *stateRef, const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices,
    const aclTensor *g, const aclTensor *gk, const aclTensor *numAcceptedTokens,
    const aclTensor *aLog, const aclTensor *dtBias, float scaleValue,
    bool useQkL2NormInKernel, bool useGateInKernel, bool useBetaSigmoidInKernel,
    bool allowNegEigval, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // Validate before DFX: its phase-1 cache can return without executing
    // the function body, and its generic null guard loses the parameter name.
    if (workspaceSize == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentGatedDeltaRule", "workspaceSize",
            "output argument must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnRecurrentGatedDeltaRule", "executor",
            "output argument must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    const rgdn_check::Params params{query, key, value, beta, stateRef, actualSeqLengths,
        ssmStateIndices, g, gk, numAcceptedTokens, aLog, dtBias, scaleValue,
        useQkL2NormInKernel, useGateInKernel, useBetaSigmoidInKernel, allowNegEigval, out};
    const aclnnStatus checkRet = rgdn_check::CheckParams(params);
    if (checkRet != ACLNN_SUCCESS) {
        return checkRet;
    }

    L2_DFX_PHASE_1(aclnnRecurrentGatedDeltaRule,
                  DFX_IN(query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices,
                         g, gk, numAcceptedTokens, aLog, dtBias, scaleValue,
                         useQkL2NormInKernel, useGateInKernel, useBetaSigmoidInKernel, allowNegEigval),
                  DFX_OUT(stateRef, out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) { REPORT_INNER_ERR_MSG("EZ9999", "RecurrentGatedDeltaRule: CREATE_EXECUTOR failed"); return ACLNN_ERR_INNER_CREATE_EXECUTOR; }
    for (auto tensor : {query, key, value, beta, static_cast<const aclTensor *>(stateRef),
                        actualSeqLengths, ssmStateIndices, g, numAcceptedTokens, aLog, dtBias,
                        static_cast<const aclTensor *>(out)}) {
        if (tensor) tensor->SetOriginalShape(tensor->GetViewShape());
    }

    // Q/K/V and the writable State are genuine strided views. Never silently
    // materialize them: both their storage offset and their strides are ABI data.
    const auto preserveView = [&](const aclTensor *tensor) -> const aclTensor * {
        return uniqueExecutor->CreateView(tensor, tensor->GetViewShape(), tensor->GetStorageShape(),
                                          tensor->GetViewStrides(), tensor->GetViewOffset());
    };
    const auto queryView = preserveView(query);
    const auto keyView = preserveView(key);
    const auto valueView = preserveView(value);
    auto stateView = uniqueExecutor->CreateView(stateRef, stateRef->GetViewShape(), stateRef->GetStorageShape(),
                                               stateRef->GetViewStrides(), stateRef->GetViewOffset());
    if (!(queryView && keyView && valueView && stateView)) { REPORT_INNER_ERR_MSG("EZ9999", "RecurrentGatedDeltaRule: failed to preserve query/key/value/state views"); return ACLNN_ERR_INNER_NULLPTR; }

    const auto contiguous = [&](const aclTensor *tensor) -> const aclTensor * {
        return tensor ? l0op::Contiguous(tensor, uniqueExecutor.get()) : nullptr;
    };
    const auto betaContiguous = contiguous(beta);
    const auto lengthsContiguous = contiguous(actualSeqLengths);
    const auto indicesContiguous = contiguous(ssmStateIndices);
    const auto gContiguous = contiguous(g);
    const auto acceptedContiguous = contiguous(numAcceptedTokens);
    const auto aLogContiguous = useGateInKernel ? contiguous(aLog) : nullptr;
    const auto biasContiguous = useGateInKernel ? contiguous(dtBias) : nullptr;
    if (!(betaContiguous && lengthsContiguous && indicesContiguous && (!g || gContiguous) &&
              (!numAcceptedTokens || acceptedContiguous) && (!useGateInKernel || aLogContiguous) &&
              (!useGateInKernel || !dtBias || biasContiguous))) { REPORT_INNER_ERR_MSG("EZ9999", "RecurrentGatedDeltaRule: input Contiguous failed"); return ACLNN_ERR_INNER_NULLPTR; }
    const auto outRet = l0op::RecurrentGatedDeltaRule(
        queryView, keyView, valueView, betaContiguous, stateView, lengthsContiguous, indicesContiguous,
        gContiguous, nullptr, acceptedContiguous, aLogContiguous, biasContiguous, scaleValue,
        useQkL2NormInKernel, useGateInKernel, useBetaSigmoidInKernel, allowNegEigval, uniqueExecutor.get());
    if (outRet == nullptr) { return ACLNN_ERR_INNER_NULLPTR; }
    if (l0op::ViewCopy(outRet, out, uniqueExecutor.get()) == nullptr) { REPORT_INNER_ERR_MSG("EZ9999", "RecurrentGatedDeltaRule: output ViewCopy failed"); return ACLNN_ERR_INNER_NULLPTR; }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRecurrentGatedDeltaRule(void *workspace, uint64_t workspaceSize,
                                           aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRecurrentGatedDeltaRule);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // extern "C"
