/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_kda_tiling_processor.h
 * \brief Tiling processor shared by aclnn tiling and fast kernel launch.
 */

#ifndef RECURRENT_KDA_TILING_PROCESSOR_H
#define RECURRENT_KDA_TILING_PROCESSOR_H

#include "../op_kernel/recurrent_kda_struct.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <cstdio>
#include <cstdarg>
#include <utility>

using RecurrentKdaTilingData = RecurrentKda::RecurrentKdaTilingData;

namespace optiling {

static constexpr size_t RKDA_RANK3_QKV_DIM_NUM = 3;
static constexpr size_t RKDA_RANK4_QKV_DIM_NUM = 4;
static constexpr size_t RKDA_RANK2_BETA_DIM_NUM = 2;
static constexpr size_t RKDA_RANK3_BETA_DIM_NUM = 3;
static constexpr size_t RKDA_STATE_DIM_NUM = 4;
static constexpr size_t RKDA_METADATA_RANK1 = 1;
static constexpr size_t RKDA_METADATA_RANK2 = 2;

static constexpr size_t RKDA_DIM_0 = 0;
static constexpr size_t RKDA_DIM_1 = 1;
static constexpr size_t RKDA_DIM_2 = 2;
static constexpr size_t RKDA_DIM_3 = 3;

static constexpr uint32_t RKDA_LAYOUT_BSND = 0;
static constexpr uint32_t RKDA_LAYOUT_TND = 1;
static constexpr size_t RKDA_MAX_MTP = 8;
static constexpr size_t RKDA_A5_MAX_MTP = 4;
static constexpr int64_t RKDA_INPUT_BUFFER_NUM = 2;
static constexpr int64_t RKDA_UB_GUARD_BYTES = 2048;
static constexpr size_t RKDA_SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;
// On A5, broadTmp aliases state in UB when K hits this fused-reduce size.
static constexpr int64_t RKDA_FUSED_REDUCE_K = 128;

struct RecurrentKdaTilingContext {
    const char *nodeName = "RecurrentKda";
    gert::Shape queryShape;
    gert::Shape keyShape;
    gert::Shape valueShape;
    gert::Shape gateShape;
    gert::Shape betaShape;
    gert::Shape stateShape;
    gert::Shape cuSeqlensShape;
    gert::Shape ssmStateShape;
    gert::Shape aLogShape;
    gert::Shape dtBiasShape;
    gert::Shape acceptedTokensShape;
    float scale = 1.0f;
    float lowerBound = -5.0f;
    uint32_t layout = RKDA_LAYOUT_BSND;
    uint32_t hasCuSeqlens = 0;
    uint32_t hasSsmStateIndices = 0;
    uint32_t hasALog = 0;
    uint32_t hasDtBias = 0;
    uint32_t hasAcceptedTokens = 0;
    uint32_t useQkL2norm = 0;
    uint32_t useGateInKernel = 0;
    uint32_t useBetaSigmoid = 0;
    uint32_t allowNegEigval = 0;
    uint32_t safeGate = 0;
    uint32_t stateVFirst = 0;
    uint32_t outputFinalState = 0;
    uint32_t inplaceFinalState = 1;
    ge::DataType stateDtype = ge::DT_BF16;
    ge::DataType gateDtype = ge::DT_FLOAT;
    ge::DataType betaDtype = ge::DT_FLOAT;
    ge::DataType cuSeqlensDtype = ge::DT_INT64;
    ge::DataType ssmStateIndicesDtype = ge::DT_INT64;
    ge::DataType acceptedTokensDtype = ge::DT_INT64;
    std::array<int64_t, RKDA_RANK4_QKV_DIM_NUM> queryStrides = {};
    std::array<int64_t, RKDA_RANK4_QKV_DIM_NUM> keyStrides = {};
    std::array<int64_t, RKDA_RANK4_QKV_DIM_NUM> valueStrides = {};
    bool hasQueryStrides = false;
    bool hasKeyStrides = false;
    bool hasValueStrides = false;
    std::array<int64_t, RKDA_STATE_DIM_NUM> stateInStrides = {};
    std::array<int64_t, RKDA_STATE_DIM_NUM> stateOutStrides = {};
    bool hasStateInStrides = false;
    bool hasStateOutStrides = false;
    uint64_t aivNum = 0;
    uint64_t ubSize = 0;
    uint32_t isA5 = 0;
    const void *cuSeqlensData = nullptr; // Host value dependency, never a device pointer.
};

class RecurrentKdaTilingProcessor {
public:
    explicit RecurrentKdaTilingProcessor(const RecurrentKdaTilingContext &ctx) : ctx_(ctx) {}

    ge::graphStatus ProcessShapes(RecurrentKdaTilingData &tiling) const
    {
        // Validate byte products before dense-stride and device-field arithmetic.
        if (ctx_.isA5 != 0) {
            const uint64_t limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
            const std::array<const gert::Shape *, 6> shapes = {{
                &ctx_.queryShape, &ctx_.keyShape, &ctx_.valueShape,
                &ctx_.gateShape, &ctx_.betaShape, &ctx_.stateShape}};
            for (const auto *shape : shapes) {
                uint64_t bytes = 4; // Conservative maximum element width.
                for (size_t axis = 0; axis < shape->GetDimNum(); ++axis) {
                    const uint64_t dim = static_cast<uint64_t>(shape->GetDim(axis));
                    OP_CHECK_IF(dim == 0 || bytes > limit / dim,
                                ReportInvalid("query/key/value/gate/beta", "Tensor byte size exceeds the signed device address range."),
                                return ge::GRAPH_FAILED);
                    bytes *= dim;
                }
            }
        }

        tiling.vectorCoreNum = static_cast<uint32_t>(ctx_.aivNum);

        struct RuleItem {
            const char *name;
            ge::graphStatus (RecurrentKdaTilingProcessor::*fn)(RecurrentKdaTilingData &) const;
        };
        const std::array<RuleItem, 4> shapeRules = {{
            {"RuleCheckShapeDimAndRelation", &RecurrentKdaTilingProcessor::RuleCheckShapeDimAndRelation},
            {"RuleFillTilingShapeData", &RecurrentKdaTilingProcessor::RuleFillTilingShapeData},
            {"RuleCheckShapeValueRangeAndRule", &RecurrentKdaTilingProcessor::RuleCheckShapeValueRangeAndRule},
            {"RuleUpdateDynamicBlockDimByTaskUnits", &RecurrentKdaTilingProcessor::RuleUpdateDynamicBlockDimByTaskUnits},
        }};
        for (const auto &rule : shapeRules) {
            OP_CHECK_IF((this->*(rule.fn))(tiling) != ge::GRAPH_SUCCESS,
                        OP_LOGE_WITHOUT_REPORT(ctx_.nodeName, "ProcessShapes rule failed: %s", rule.name),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ProcessUb(RecurrentKdaTilingData &tiling) const
    {
        if (ctx_.isA5 != 0) {
            return PlanA5Ub(tiling);
        }
        struct RuleItem {
            const char *name;
            ge::graphStatus (RecurrentKdaTilingProcessor::*fn)(RecurrentKdaTilingData &, UbCalcContext &) const;
        };
        UbCalcContext ubCalcCtx;
        const std::array<RuleItem, 5> ubRules = {{
            {"RuleInitUbCalcContext", &RecurrentKdaTilingProcessor::RuleInitUbCalcContext},
            {"RuleCalcFixedUbBytes", &RecurrentKdaTilingProcessor::RuleCalcFixedUbBytes},
            {"RuleCalcWorkingUbBytes", &RecurrentKdaTilingProcessor::RuleCalcWorkingUbBytes},
            {"RuleCalcVStepCoeff", &RecurrentKdaTilingProcessor::RuleCalcVStepCoeff},
            {"RuleFinalizeVStepFromUb", &RecurrentKdaTilingProcessor::RuleFinalizeVStepFromUb},
        }};
        for (const auto &rule : ubRules) {
            OP_CHECK_IF((this->*(rule.fn))(tiling, ubCalcCtx) != ge::GRAPH_SUCCESS,
                        OP_LOGE_WITHOUT_REPORT(ctx_.nodeName, "ProcessUb rule failed: %s", rule.name),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

private:
    void ReportInvalid(const char *name, const char *format, ...) const
    {
        char expected[768];
        va_list args;
        va_start(args, format);
        std::vsnprintf(expected, sizeof(expected), format, args);
        va_end(args);
        std::string reason = expected;
        const std::array<std::pair<const char *, const gert::Shape *>, 10> shapes = {{
            {"query", &ctx_.queryShape}, {"key", &ctx_.keyShape}, {"value", &ctx_.valueShape},
            {"gate", &ctx_.gateShape}, {"beta", &ctx_.betaShape}, {"initial_state", &ctx_.stateShape},
            {"cu_seqlens", &ctx_.cuSeqlensShape}, {"ssm_state_indices", &ctx_.ssmStateShape},
            {"A_log", &ctx_.aLogShape}, {"dt_bias", &ctx_.dtBiasShape}}};
        for (const auto &entry : shapes) {
            reason += std::string("; actual ") + entry.first + "=[";
            for (size_t i = 0; i < entry.second->GetDimNum(); ++i) {
                if (i != 0) reason += ",";
                reason += std::to_string(entry.second->GetDim(i));
            }
            reason += "]";
        }
        reason += "; isA5=" + std::to_string(ctx_.isA5) + "; UB bytes=" + std::to_string(ctx_.ubSize) +
                  "; lower_bound=" + std::to_string(ctx_.lowerBound);
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ctx_.nodeName, name, reason);
    }

    struct UbCalcContext {
        int64_t ubSize = 0;
        int64_t aNv = 0;
        int64_t aDv = 0;
        int64_t aDk = 0;
        int64_t fixedUbBytes = 0;
        int64_t workingUbBytes = 0;
        int64_t coeff = 0;
    };

    struct BufferProfile {
        uint32_t stateOutBufferNum = 1;
        uint32_t attnOutBufferNum = 1;
        uint32_t vStep = 0;
        uint32_t repeatTime = 0;
        bool valid = false;
    };

    RecurrentKdaTilingContext ctx_;

    std::array<int64_t, RKDA_STATE_DIM_NUM> ResolveStateStrides(bool output) const
    {
        const bool hasStrides = output ? ctx_.hasStateOutStrides : ctx_.hasStateInStrides;
        if (hasStrides) {
            return output ? ctx_.stateOutStrides : ctx_.stateInStrides;
        }
        const auto &shape = ctx_.stateShape;
        std::array<int64_t, RKDA_STATE_DIM_NUM> strides = {};
        strides[RKDA_DIM_3] = 1;
        strides[RKDA_DIM_2] = shape.GetDim(RKDA_DIM_3);
        strides[RKDA_DIM_1] = shape.GetDim(RKDA_DIM_2) * strides[RKDA_DIM_2];
        strides[RKDA_DIM_0] = shape.GetDim(RKDA_DIM_1) * strides[RKDA_DIM_1];
        return strides;
    }

    std::array<int64_t, RKDA_RANK4_QKV_DIM_NUM> ResolveQkvStrides(
        const gert::Shape &shape,
        const std::array<int64_t, RKDA_RANK4_QKV_DIM_NUM> &viewStrides,
        bool hasStrides) const
    {
        if (hasStrides) {
            return viewStrides;
        }
        std::array<int64_t, RKDA_RANK4_QKV_DIM_NUM> strides = {};
        int64_t denseStride = 1;
        for (size_t dim = shape.GetDimNum(); dim > 0; --dim) {
            const size_t index = dim - 1;
            strides[index] = denseStride;
            denseStride *= shape.GetDim(index);
        }
        return strides;
    }

    ge::graphStatus ValidateQkvStrides(
        const gert::Shape &shape,
        const std::array<int64_t, RKDA_RANK4_QKV_DIM_NUM> &strides,
        const char *name) const
    {
        const size_t rank = shape.GetDimNum();
        for (size_t i = 0; i < rank; ++i) {
            OP_CHECK_IF(strides[i] <= 0,
                        ReportInvalid("query/key/value/gate/beta", "%s stride[%zu] must be positive, but it is %ld.",
                                name, i, strides[i]),
                        return ge::GRAPH_FAILED);
        }

        const size_t tokenDim = rank == RKDA_RANK3_QKV_DIM_NUM ? RKDA_DIM_0 : RKDA_DIM_1;
        const size_t headDim = rank == RKDA_RANK3_QKV_DIM_NUM ? RKDA_DIM_1 : RKDA_DIM_2;
        const size_t featureDim = rank == RKDA_RANK3_QKV_DIM_NUM ? RKDA_DIM_2 : RKDA_DIM_3;
        const uint64_t headCount = static_cast<uint64_t>(shape.GetDim(headDim));
        const uint64_t featureCount = static_cast<uint64_t>(shape.GetDim(featureDim));
        const uint64_t tokenStride = static_cast<uint64_t>(strides[tokenDim]);
        const uint64_t headStride = static_cast<uint64_t>(strides[headDim]);
        OP_CHECK_IF(strides[featureDim] != 1 || headStride < featureCount,
                    ReportInvalid("query/key/value/gate/beta",
                            "%s must keep feature elements dense and non-overlapping: "
                            "feature stride=1 and head stride>=%llu.",
                            name, static_cast<unsigned long long>(featureCount)),
                    return ge::GRAPH_FAILED);

        const uint64_t maxUint64 = std::numeric_limits<uint64_t>::max();
        OP_CHECK_IF(headCount > 1 &&
                        headStride > (maxUint64 - featureCount) / (headCount - 1),
                    ReportInvalid("query/key/value/gate/beta", "%s head stride calculation overflows.", name),
                    return ge::GRAPH_FAILED);
        const uint64_t tokenCount = static_cast<uint64_t>(shape.GetDim(tokenDim));
        OP_CHECK_IF(tokenCount > 1 &&
                        tokenStride > (maxUint64 - featureCount) / (tokenCount - 1),
                    ReportInvalid("query/key/value/gate/beta", "%s token span calculation overflows.", name),
                    return ge::GRAPH_FAILED);
        const uint64_t tokenMajorSpan = (headCount - 1) * headStride + featureCount;
        const uint64_t headMajorSpan = (tokenCount - 1) * tokenStride + featureCount;
        const bool tokenMajor = tokenStride >= tokenMajorSpan;
        const bool headMajor = headStride >= headMajorSpan;
        OP_CHECK_IF(!tokenMajor && !headMajor,
                    ReportInvalid("query/key/value/gate/beta", "%s token/head axes overlap.", name),
                    return ge::GRAPH_FAILED);

        const uint64_t srcGapElements = tokenStride - featureCount;
        OP_CHECK_IF(srcGapElements > std::numeric_limits<uint32_t>::max() / sizeof(uint16_t),
                    ReportInvalid("query/key/value/gate/beta",
                            "%s token gap exceeds DataCopy source-stride range: %llu elements.",
                            name, static_cast<unsigned long long>(srcGapElements)),
                    return ge::GRAPH_FAILED);

        if (rank == RKDA_RANK4_QKV_DIM_NUM && shape.GetDim(RKDA_DIM_0) > 1) {
            const uint64_t seqLen = static_cast<uint64_t>(shape.GetDim(RKDA_DIM_1));
            OP_CHECK_IF(tokenStride > maxUint64 / seqLen,
                        ReportInvalid("query/key/value/gate/beta", "%s batch stride calculation overflows.", name),
                        return ge::GRAPH_FAILED);
            const uint64_t expectedBatchStride = seqLen * tokenStride;
            OP_CHECK_IF(tokenMajor && static_cast<uint64_t>(strides[RKDA_DIM_0]) <
                                         expectedBatchStride - tokenStride + tokenMajorSpan,
                        ReportInvalid("query/key/value/gate/beta",
                                "%s batch stride overlaps a BSND sequence, "
                                "but got %llu and expected %llu.",
                                name, static_cast<unsigned long long>(strides[RKDA_DIM_0]),
                                static_cast<unsigned long long>(expectedBatchStride)),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(!tokenMajor && static_cast<uint64_t>(strides[RKDA_DIM_0]) <
                                     std::max(tokenMajorSpan, headMajorSpan),
                        ReportInvalid("query/key/value/gate/beta", "%s batch stride overlaps the view.", name),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckQkvStrides() const
    {
        const auto queryStrides = ResolveQkvStrides(ctx_.queryShape, ctx_.queryStrides, ctx_.hasQueryStrides);
        const auto keyStrides = ResolveQkvStrides(ctx_.keyShape, ctx_.keyStrides, ctx_.hasKeyStrides);
        const auto valueStrides = ResolveQkvStrides(ctx_.valueShape, ctx_.valueStrides, ctx_.hasValueStrides);
        if (ValidateQkvStrides(ctx_.queryShape, queryStrides, "query") != ge::GRAPH_SUCCESS ||
            ValidateQkvStrides(ctx_.keyShape, keyStrides, "key") != ge::GRAPH_SUCCESS ||
            ValidateQkvStrides(ctx_.valueShape, valueStrides, "value") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ValidateStateStrides(
        const std::array<int64_t, RKDA_STATE_DIM_NUM> &strides, const char *name) const
    {
        for (size_t i = 0; i < RKDA_STATE_DIM_NUM; ++i) {
            OP_CHECK_IF(strides[i] <= 0,
                        ReportInvalid("query/key/value/gate/beta", "%s stride[%zu] must be positive, but it is %ld.",
                                name, i, strides[i]),
                        return ge::GRAPH_FAILED);
        }
        const auto &shape = ctx_.stateShape;
        const int64_t denseRow = shape.GetDim(RKDA_DIM_3);
        const int64_t densePlane = shape.GetDim(RKDA_DIM_2) * denseRow;
        OP_CHECK_IF(strides[RKDA_DIM_3] != 1 || strides[RKDA_DIM_2] != denseRow,
                    ReportInvalid("initial_state",
                            "%s must keep its inner state matrix dense: stride[3]=1 and stride[2]=%ld, "
                            "but got [%ld, %ld, %ld, %ld].",
                            name, denseRow, strides[RKDA_DIM_0], strides[RKDA_DIM_1],
                            strides[RKDA_DIM_2], strides[RKDA_DIM_3]),
                    return ge::GRAPH_FAILED);
        const int64_t minSlotStride =
            (shape.GetDim(RKDA_DIM_1) - 1) * strides[RKDA_DIM_1] + densePlane;
        OP_CHECK_IF(strides[RKDA_DIM_1] < densePlane ||
                        strides[RKDA_DIM_0] < minSlotStride,
                    ReportInvalid("initial_state",
                            "%s outer strides overlap state rows or heads: [%ld, %ld, %ld, %ld].",
                            name, strides[RKDA_DIM_0], strides[RKDA_DIM_1],
                            strides[RKDA_DIM_2], strides[RKDA_DIM_3]),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckStateStrides() const
    {
        if (ValidateStateStrides(ResolveStateStrides(false), "initial_state") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (ValidateStateStrides(ResolveStateStrides(true), "state output") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    bool CheckDim(const gert::Shape &shape, const size_t dim, const std::string &dimDesc) const
    {
        if (shape.GetDimNum() != dim) {
            ReportInvalid("query/key/value/gate/beta", "The number of dimensions of %s should be %zu, but it is %zu",
                    dimDesc.c_str(), dim, shape.GetDimNum());
            return false;
        }
        return true;
    }

    bool CheckDimEqual(const gert::Shape &a, const int64_t dimA, const gert::Shape &b, const int64_t dimB,
                       const std::string &nameA, const std::string &nameB, const std::string &dimDesc) const
    {
        if (a.GetDim(dimA) != b.GetDim(dimB)) {
            ReportInvalid("query/key/value/gate/beta", "The %s of %s and %s should be the same, but %s is %ld while %s is %ld",
                    dimDesc.c_str(), nameA.c_str(), nameB.c_str(), nameA.c_str(), a.GetDim(dimA), nameB.c_str(),
                    b.GetDim(dimB));
            return false;
        }
        return true;
    }

    int64_t SeqNum(const gert::Shape &queryShape, const gert::Shape &cuSeqlensShape) const
    {
        if (ctx_.hasCuSeqlens) {
            return cuSeqlensShape.GetDim(RKDA_DIM_0) - 1;
        }
        return ctx_.layout == RKDA_LAYOUT_BSND ? queryShape.GetDim(RKDA_DIM_0) : 1;
    }

    ge::graphStatus CheckMetadataShapes(const gert::Shape &queryShape, const gert::Shape &cuSeqlensShape) const
    {
        if (ctx_.hasCuSeqlens) {
            if (!CheckDim(cuSeqlensShape, RKDA_METADATA_RANK1, "cu_seqlens")) {
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF(cuSeqlensShape.GetDim(RKDA_DIM_0) < 2,
                        ReportInvalid("cu_seqlens", "cu_seqlens must contain at least 2 elements."),
                        return ge::GRAPH_FAILED);
        }
        int64_t totalTokens =
            (ctx_.layout == RKDA_LAYOUT_TND) ? queryShape.GetDim(RKDA_DIM_0) :
            queryShape.GetDim(RKDA_DIM_0) * queryShape.GetDim(RKDA_DIM_1);
        int64_t seqNum = SeqNum(queryShape, cuSeqlensShape);

        if (ctx_.hasSsmStateIndices) {
            size_t rank = ctx_.ssmStateShape.GetDimNum();
            bool packed1d = rank == RKDA_METADATA_RANK1 &&
                            ctx_.ssmStateShape.GetDim(RKDA_DIM_0) >= totalTokens;
            bool speculative2d = rank == RKDA_METADATA_RANK2 &&
                                 ctx_.ssmStateShape.GetDim(RKDA_DIM_0) == seqNum &&
                                 ctx_.ssmStateShape.GetDim(RKDA_DIM_1) > 0;
            OP_CHECK_IF(!packed1d && !speculative2d,
                        ReportInvalid("ssm_state_indices",
                                "ssm_state_indices must be packed [T] or speculative [seq_num,max_step]."),
                        return ge::GRAPH_FAILED);
        }
        if (ctx_.hasAcceptedTokens) {
            if (!CheckDim(ctx_.acceptedTokensShape, RKDA_METADATA_RANK1, "num_accepted_tokens")) {
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF(ctx_.acceptedTokensShape.GetDim(RKDA_DIM_0) != seqNum,
                        ReportInvalid("num_accepted_tokens", "num_accepted_tokens length must equal sequence number."),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckOptionalGateShapes(int64_t hvNum, int64_t kDim) const
    {
        if (ctx_.hasALog) {
            if (!CheckDim(ctx_.aLogShape, RKDA_METADATA_RANK1, "A_log")) {
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF(ctx_.aLogShape.GetDim(RKDA_DIM_0) != hvNum,
                        ReportInvalid("A_log", "A_log shape must be [HV]."),
                        return ge::GRAPH_FAILED);
        }
        if (ctx_.hasALog && ctx_.hasDtBias) {
            size_t rank = ctx_.dtBiasShape.GetDimNum();
            bool valid = (rank == 1 && ctx_.dtBiasShape.GetDim(RKDA_DIM_0) == hvNum * kDim) ||
                         (rank == 2 && ctx_.dtBiasShape.GetDim(RKDA_DIM_0) == hvNum &&
                          ctx_.dtBiasShape.GetDim(RKDA_DIM_1) == kDim);
            OP_CHECK_IF(!valid, ReportInvalid("dt_bias", "dt_bias must be [HV*K] or [HV, K]."),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckShapeDimAndRelation(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                             const gert::Shape &valueShape, const gert::Shape &gateShape,
                                             const gert::Shape &betaShape, const gert::Shape &stateShape,
                                             const gert::Shape &cuSeqlensShape) const
    {
        int64_t totalTokens = 0;
        int64_t denseSeqLen = 0;
        int64_t hNum = 0;
        int64_t hvNum = 0;
        int64_t kDim = 0;
        int64_t vDim = 0;
        if (ctx_.layout == RKDA_LAYOUT_TND) {
            if (!CheckDim(queryShape, RKDA_RANK3_QKV_DIM_NUM, "query") ||
                !CheckDim(keyShape, RKDA_RANK3_QKV_DIM_NUM, "key") ||
                !CheckDim(valueShape, RKDA_RANK3_QKV_DIM_NUM, "value") ||
                !CheckDim(gateShape, RKDA_RANK3_QKV_DIM_NUM, "gate") ||
                !CheckDim(betaShape, RKDA_RANK2_BETA_DIM_NUM, "beta")) {
                return ge::GRAPH_FAILED;
            }
            totalTokens = queryShape.GetDim(RKDA_DIM_0);
            denseSeqLen = totalTokens;
            hNum = queryShape.GetDim(RKDA_DIM_1);
            kDim = queryShape.GetDim(RKDA_DIM_2);
            hvNum = valueShape.GetDim(RKDA_DIM_1);
            vDim = valueShape.GetDim(RKDA_DIM_2);
            OP_CHECK_IF(valueShape.GetDim(RKDA_DIM_0) != totalTokens ||
                            gateShape.GetDim(RKDA_DIM_0) != totalTokens ||
                            betaShape.GetDim(RKDA_DIM_0) != totalTokens ||
                            gateShape.GetDim(RKDA_DIM_1) != hvNum ||
                            betaShape.GetDim(RKDA_DIM_1) != hvNum ||
                            gateShape.GetDim(RKDA_DIM_2) != kDim,
                        ReportInvalid("query/key/value/gate/beta",
                                "TND expects q/k [T,H,K], value [T,HV,V], gate [T,HV,K], beta [T,HV]."),
                        return ge::GRAPH_FAILED);
        } else {
            if (!CheckDim(queryShape, RKDA_RANK4_QKV_DIM_NUM, "query") ||
                !CheckDim(keyShape, RKDA_RANK4_QKV_DIM_NUM, "key") ||
                !CheckDim(valueShape, RKDA_RANK4_QKV_DIM_NUM, "value") ||
                !CheckDim(gateShape, RKDA_RANK4_QKV_DIM_NUM, "gate") ||
                !CheckDim(betaShape, RKDA_RANK3_BETA_DIM_NUM, "beta")) {
                return ge::GRAPH_FAILED;
            }
            int64_t batch = queryShape.GetDim(RKDA_DIM_0);
            denseSeqLen = queryShape.GetDim(RKDA_DIM_1);
            totalTokens = batch * denseSeqLen;
            hNum = queryShape.GetDim(RKDA_DIM_2);
            kDim = queryShape.GetDim(RKDA_DIM_3);
            hvNum = valueShape.GetDim(RKDA_DIM_2);
            vDim = valueShape.GetDim(RKDA_DIM_3);
            OP_CHECK_IF(valueShape.GetDim(RKDA_DIM_0) != batch ||
                            valueShape.GetDim(RKDA_DIM_1) != denseSeqLen ||
                            gateShape.GetDim(RKDA_DIM_0) != batch ||
                            gateShape.GetDim(RKDA_DIM_1) != denseSeqLen ||
                            gateShape.GetDim(RKDA_DIM_2) != hvNum ||
                            gateShape.GetDim(RKDA_DIM_3) != kDim ||
                            betaShape.GetDim(RKDA_DIM_0) != batch ||
                            betaShape.GetDim(RKDA_DIM_1) != denseSeqLen ||
                            betaShape.GetDim(RKDA_DIM_2) != hvNum,
                        ReportInvalid("query/key/value/gate/beta",
                                "BSND expects q/k [B,T,H,K], value [B,T,HV,V], gate [B,T,HV,K], beta [B,T,HV]."),
                        return ge::GRAPH_FAILED);
        }

        if (!CheckDimEqual(queryShape, RKDA_DIM_0, keyShape, RKDA_DIM_0, "query", "key",
                           "leading token dimension") ||
            !CheckDimEqual(queryShape, queryShape.GetDimNum() - 2, keyShape, keyShape.GetDimNum() - 2,
                           "query", "key", "H dimension") ||
            !CheckDimEqual(queryShape, queryShape.GetDimNum() - 1, keyShape, keyShape.GetDimNum() - 1,
                           "query", "key", "K dimension")) {
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF(hNum <= 0 || hvNum <= 0 || kDim <= 0 || vDim <= 0 || totalTokens <= 0 || denseSeqLen <= 0,
                    ReportInvalid("query/key/value/gate/beta", "input shape dimensions must be positive."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(hvNum % hNum != 0,
                    ReportInvalid("query/key/value/gate/beta", "HV must be an integer multiple of H, but HV is %ld and H is %ld.",
                            hvNum, hNum),
                    return ge::GRAPH_FAILED);
        if (!CheckDim(stateShape, RKDA_STATE_DIM_NUM, "initial_state")) {
            return ge::GRAPH_FAILED;
        }
        int64_t seqNum = SeqNum(queryShape, cuSeqlensShape);
        bool stateTailMatches = ctx_.stateVFirst ?
            (stateShape.GetDim(RKDA_DIM_2) == vDim && stateShape.GetDim(RKDA_DIM_3) == kDim) :
            (stateShape.GetDim(RKDA_DIM_2) == kDim && stateShape.GetDim(RKDA_DIM_3) == vDim);
        OP_CHECK_IF(stateShape.GetDim(RKDA_DIM_0) <= 0 ||
                        (!ctx_.hasSsmStateIndices && stateShape.GetDim(RKDA_DIM_0) != seqNum) ||
                        stateShape.GetDim(RKDA_DIM_1) != hvNum || !stateTailMatches,
                    ReportInvalid("initial_state",
                            "state must be [state_capacity, HV, V, K] when state_v_first=true or "
                            "[state_capacity, HV, K, V] otherwise; without ssm_state_indices, "
                            "state_capacity must equal seq_num."),
                    return ge::GRAPH_FAILED);

        if (CheckQkvStrides() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        if (CheckStateStrides() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        if (CheckMetadataShapes(queryShape, cuSeqlensShape) != ge::GRAPH_SUCCESS ||
            CheckOptionalGateShapes(hvNum, kDim) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    void FillTilingShapeData(const gert::Shape &queryShape, const gert::Shape &valueShape,
                             const gert::Shape &stateShape, const gert::Shape &cuSeqlensShape,
                             RecurrentKdaTilingData &tiling) const
    {
        if (ctx_.layout == RKDA_LAYOUT_TND) {
            tiling.t = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_0));
            tiling.seqLen = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_0));
            tiling.nk = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_1));
            tiling.dk = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_2));
            tiling.nv = static_cast<uint32_t>(valueShape.GetDim(RKDA_DIM_1));
            tiling.dv = static_cast<uint32_t>(valueShape.GetDim(RKDA_DIM_2));
            tiling.b = static_cast<uint32_t>(SeqNum(queryShape, cuSeqlensShape));
        } else {
            tiling.seqLen = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_1));
            tiling.t = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_0) * queryShape.GetDim(RKDA_DIM_1));
            tiling.nk = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_2));
            tiling.dk = static_cast<uint32_t>(queryShape.GetDim(RKDA_DIM_3));
            tiling.nv = static_cast<uint32_t>(valueShape.GetDim(RKDA_DIM_2));
            tiling.dv = static_cast<uint32_t>(valueShape.GetDim(RKDA_DIM_3));
            tiling.b = static_cast<uint32_t>(SeqNum(queryShape, cuSeqlensShape));
        }
        tiling.sBlockNum = static_cast<uint32_t>(stateShape.GetDim(RKDA_DIM_0));
        tiling.ssmStateStride = (ctx_.hasSsmStateIndices && ctx_.ssmStateShape.GetDimNum() == RKDA_METADATA_RANK2) ?
                                static_cast<uint32_t>(ctx_.ssmStateShape.GetDim(RKDA_DIM_1)) : 0;
        const auto queryStrides = ResolveQkvStrides(queryShape, ctx_.queryStrides, ctx_.hasQueryStrides);
        const auto keyStrides = ResolveQkvStrides(ctx_.keyShape, ctx_.keyStrides, ctx_.hasKeyStrides);
        const auto valueStrides = ResolveQkvStrides(valueShape, ctx_.valueStrides, ctx_.hasValueStrides);
        const size_t tokenDim = ctx_.layout == RKDA_LAYOUT_TND ? RKDA_DIM_0 : RKDA_DIM_1;
        const size_t headDim = ctx_.layout == RKDA_LAYOUT_TND ? RKDA_DIM_1 : RKDA_DIM_2;
        tiling.queryTokenStride = static_cast<uint64_t>(queryStrides[tokenDim]);
        tiling.queryHeadStride = static_cast<uint64_t>(queryStrides[headDim]);
        tiling.keyTokenStride = static_cast<uint64_t>(keyStrides[tokenDim]);
        tiling.keyHeadStride = static_cast<uint64_t>(keyStrides[headDim]);
        tiling.valueTokenStride = static_cast<uint64_t>(valueStrides[tokenDim]);
        tiling.valueHeadStride = static_cast<uint64_t>(valueStrides[headDim]);
        tiling.queryBatchStride = ctx_.hasCuSeqlens ? 0 : static_cast<uint64_t>(queryStrides[0]);
        tiling.keyBatchStride = ctx_.hasCuSeqlens ? 0 : static_cast<uint64_t>(keyStrides[0]);
        tiling.valueBatchStride = ctx_.hasCuSeqlens ? 0 : static_cast<uint64_t>(valueStrides[0]);
        const auto stateInStrides = ResolveStateStrides(false);
        const auto stateOutStrides = ResolveStateStrides(true);
        tiling.stateInStride0 = static_cast<uint64_t>(stateInStrides[RKDA_DIM_0]);
        tiling.stateInStride1 = static_cast<uint64_t>(stateInStrides[RKDA_DIM_1]);
        tiling.stateInStride2 = static_cast<uint64_t>(stateInStrides[RKDA_DIM_2]);
        tiling.stateInStride3 = static_cast<uint64_t>(stateInStrides[RKDA_DIM_3]);
        tiling.stateOutStride0 = static_cast<uint64_t>(stateOutStrides[RKDA_DIM_0]);
        tiling.stateOutStride1 = static_cast<uint64_t>(stateOutStrides[RKDA_DIM_1]);
        tiling.stateOutStride2 = static_cast<uint64_t>(stateOutStrides[RKDA_DIM_2]);
        tiling.stateOutStride3 = static_cast<uint64_t>(stateOutStrides[RKDA_DIM_3]);
        tiling.scale = ctx_.scale;
        tiling.lowerBound = ctx_.lowerBound;
        tiling.layout = ctx_.layout;
        tiling.hasCuSeqlens = ctx_.hasCuSeqlens;
        tiling.hasSsmStateIndices = ctx_.hasSsmStateIndices;
        tiling.hasALog = ctx_.hasALog;
        tiling.hasDtBias = ctx_.hasALog && ctx_.hasDtBias;
        tiling.hasAcceptedTokens = ctx_.hasAcceptedTokens;
        tiling.useQkL2norm = ctx_.useQkL2norm;
        tiling.useGateInKernel = ctx_.useGateInKernel;
        tiling.useBetaSigmoid = ctx_.useBetaSigmoid;
        tiling.allowNegEigval = ctx_.allowNegEigval;
        tiling.safeGate = ctx_.hasALog && ctx_.safeGate;
        tiling.stateVFirst = ctx_.stateVFirst;
        tiling.outputFinalState = ctx_.outputFinalState;
        tiling.inplaceFinalState = ctx_.inplaceFinalState;
        tiling.gateDtype = ctx_.gateDtype == ge::DT_FLOAT ? 0 : (ctx_.gateDtype == ge::DT_BF16 ? 1 : 2);
        tiling.betaDtype = ctx_.betaDtype == ge::DT_FLOAT ? 0 : (ctx_.betaDtype == ge::DT_BF16 ? 1 : 2);
        tiling.cuSeqlensDtype = ctx_.cuSeqlensDtype == ge::DT_INT32 ? 0 : 1;
        tiling.ssmStateIndicesDtype = ctx_.ssmStateIndicesDtype == ge::DT_INT32 ? 0 : 1;
        tiling.acceptedTokensDtype = ctx_.acceptedTokensDtype == ge::DT_INT32 ? 0 : 1;
    }

    ge::graphStatus CheckShapeValueRangeAndRule(const RecurrentKdaTilingData &tiling) const
    {
        OP_CHECK_IF(ctx_.isA5 != 0 && tiling.dv != 128,
                    ReportInvalid("query/key/value/gate/beta", "A5 RegBase design requires K=V=128."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tiling.dk != 128 || (tiling.dv != 128 && tiling.dv != 256),
                    ReportInvalid("query/key/value/gate/beta",
                            "K/V currently support only K=128,V=128 or K=128,V=256, but K=%u, V=%u.",
                            tiling.dk, tiling.dv),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tiling.nv % tiling.nk != 0,
                    ReportInvalid("query/key/value/gate/beta", "HV must be divisible by H, but HV=%u and H=%u.",
                            tiling.nv, tiling.nk),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tiling.safeGate && !(tiling.lowerBound >= -5.0f && tiling.lowerBound < 0.0f),
                    ReportInvalid("lower_bound", "lower_bound must be in [-5, 0) when safe_gate is true."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    void UpdateDynamicBlockDimByTaskUnits(RecurrentKdaTilingData &tiling) const
    {
        uint64_t taskUnits = static_cast<uint64_t>(tiling.b) * static_cast<uint64_t>(tiling.nv);
        if (taskUnits == 0) {
            taskUnits = 1;
        }
        uint64_t maxCoreNum = (ctx_.aivNum > 0) ? ctx_.aivNum : 1;
        uint64_t selectedCoreNum = (taskUnits < maxCoreNum) ? taskUnits : maxCoreNum;
        tiling.vectorCoreNum = static_cast<uint32_t>(selectedCoreNum);
        OP_LOGD(ctx_.nodeName, "taskUnits: [%llu], selected vectorCoreNum: [%u]",
                static_cast<unsigned long long>(taskUnits), tiling.vectorCoreNum);
    }

    ge::graphStatus RuleCheckShapeDimAndRelation(RecurrentKdaTilingData &tiling) const
    {
        (void)tiling;
        return CheckShapeDimAndRelation(ctx_.queryShape, ctx_.keyShape, ctx_.valueShape, ctx_.gateShape,
                                        ctx_.betaShape, ctx_.stateShape, ctx_.cuSeqlensShape);
    }

    ge::graphStatus RuleFillTilingShapeData(RecurrentKdaTilingData &tiling) const
    {
        const int64_t maxUint32 = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
        const int64_t nk = ctx_.layout == RKDA_LAYOUT_TND ?
                               ctx_.queryShape.GetDim(RKDA_DIM_1) :
                               ctx_.queryShape.GetDim(RKDA_DIM_2);
        const int64_t nv = ctx_.layout == RKDA_LAYOUT_TND ?
                               ctx_.valueShape.GetDim(RKDA_DIM_1) :
                               ctx_.valueShape.GetDim(RKDA_DIM_2);
        OP_CHECK_IF(nk > maxUint32 || nv > maxUint32,
                    ReportInvalid("query/key/value/gate/beta",
                            "head counts exceed uint32 tiling fields: H=%ld, HV=%ld.", nk, nv),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.isA5 != 0 &&
                        static_cast<uint64_t>(nv - 1) > std::numeric_limits<uint32_t>::max() / (128ULL * 4),
                    ReportInvalid("gate", "Gate token gap exceeds the uint32 DMA stride field."),
                    return ge::GRAPH_FAILED);
        if (ctx_.isA5 != 0) {
            const int64_t maxCapacity = maxUint32;
            const int64_t seqLen = ctx_.queryShape.GetDim(
                ctx_.layout == RKDA_LAYOUT_TND ? RKDA_DIM_0 : RKDA_DIM_1);
            const int64_t batch = ctx_.layout == RKDA_LAYOUT_TND ? 1 :
                                  ctx_.queryShape.GetDim(RKDA_DIM_0);
            OP_CHECK_IF(seqLen > maxCapacity || batch > maxCapacity / seqLen ||
                            SeqNum(ctx_.queryShape, ctx_.cuSeqlensShape) > maxCapacity ||
                            ctx_.stateShape.GetDim(RKDA_DIM_0) > maxCapacity ||
                            (ctx_.hasSsmStateIndices &&
                             ctx_.ssmStateShape.GetDimNum() == RKDA_METADATA_RANK2 &&
                             ctx_.ssmStateShape.GetDim(RKDA_DIM_1) > maxCapacity),
                        ReportInvalid("initial_state", "Token/state capacities exceed uint32 tiling fields."),
                        return ge::GRAPH_FAILED);
        }
        FillTilingShapeData(ctx_.queryShape, ctx_.valueShape, ctx_.stateShape, ctx_.cuSeqlensShape, tiling);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleCheckShapeValueRangeAndRule(RecurrentKdaTilingData &tiling) const
    {
        return CheckShapeValueRangeAndRule(tiling);
    }

    ge::graphStatus RuleUpdateDynamicBlockDimByTaskUnits(RecurrentKdaTilingData &tiling) const
    {
        UpdateDynamicBlockDimByTaskUnits(tiling);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanA5Ub(RecurrentKdaTilingData &tiling) const
    {
        // 3 full FP32 banks + raw input/output double slots + parameter slots + guard.
        const uint64_t capacity = std::min<uint64_t>(ctx_.ubSize, 248ULL * 1024);
        uint64_t maxLength = tiling.seqLen;
        if (ctx_.hasCuSeqlens) {
            OP_CHECK_IF(ctx_.cuSeqlensData == nullptr,
                        ReportInvalid("cu_seqlens", "A5 X selection requires host cu_seqlens values."),
                        return ge::GRAPH_FAILED);
            auto value = [&](uint32_t i) -> int64_t {
                return ctx_.cuSeqlensDtype == ge::DT_INT32 ?
                    static_cast<const int32_t *>(ctx_.cuSeqlensData)[i] :
                    static_cast<const int64_t *>(ctx_.cuSeqlensData)[i];
            };
            int64_t previous = value(0);
            OP_CHECK_IF(previous != 0, ReportInvalid("cu_seqlens", "cu_seqlens[0] must be zero; actual=%ld.", previous),
                        return ge::GRAPH_FAILED);
            maxLength = 0;
            for (uint32_t b = 0; b < tiling.b; ++b) {
                const int64_t end = value(b + 1);
                OP_CHECK_IF(end < previous || end > tiling.t,
                            ReportInvalid("cu_seqlens", "cu_seqlens must be monotonic within token capacity; "
                                "index=%u, previous=%ld, actual=%ld, capacity=%u.", b + 1, previous, end, tiling.t),
                            return ge::GRAPH_FAILED);
                maxLength = std::max<uint64_t>(maxLength, static_cast<uint64_t>(end - previous));
                previous = end;
            }
        }
        const uint64_t selectionLength = std::min<uint64_t>(8, std::max<uint64_t>(1, maxLength));
        uint64_t selected = 0, bytes = 0;
        // Beta rows are padded to one 32-byte slot per token and double-buffered.
        // X=8 is the largest A5 tile that fits with the three state banks.
        for (uint64_t x : {1ULL, 2ULL, 4ULL, 8ULL}) {
            const uint64_t betaSlot = std::max<uint64_t>(128, 32 * x);
            const uint64_t beta = 2 * betaSlot;
            // K-first FP32: three 128x136 banks instead of 128x128.
            // BF16 padding fits in the existing 64 KiB bank allocation.
            const uint64_t statePadding = !ctx_.stateVFirst && ctx_.stateDtype == ge::DT_FLOAT ? 12288 : 0;
            const uint64_t peak = 199744 + statePadding + 3072 * x + beta;
            if (x > selectionLength || 32 * x > betaSlot || peak > capacity) continue;
            selected = x;
            bytes = peak;
        }
        OP_CHECK_IF(selected == 0 || bytes > std::numeric_limits<uint32_t>::max(),
                    ReportInvalid("UB/shape", "A5 UB cannot hold the specified recurrent KDA banks and raw slots."),
                    return ge::GRAPH_FAILED);
        tiling.windowSize = static_cast<uint32_t>(selected);
        tiling.ubCalSize = static_cast<uint32_t>(capacity);
        tiling.ubRestBytes = static_cast<uint32_t>(bytes);
        tiling.vStep = 128;
        tiling.stateOutBufferNum = 3;
        tiling.attnOutBufferNum = 2;
        return ge::GRAPH_SUCCESS;
    }

    int64_t CalcFixedUbBytes(int64_t aNv, int64_t aDv, int64_t aDk, uint32_t nv) const
    {
        const int64_t rawBytes = RKDA_MAX_MTP * (8 * aDk + 2 * aDv);
        const int64_t rawBufferNum = ctx_.isA5 != 0 ? 1 : 2;
        const int64_t betaHeads = ctx_.isA5 != 0 ? aNv : static_cast<int64_t>(nv);
        int64_t usedUbBytes = rawBufferNum * rawBytes;
        usedUbBytes += RKDA_MAX_MTP * 4 * betaHeads;
        usedUbBytes += 64 + RKDA_UB_GUARD_BYTES;
        return usedUbBytes;
    }

    int64_t CalcWorkingUbBytes(int64_t aNv, int64_t aDv, int64_t aDk, uint32_t nv) const
    {
        const int64_t betaHeads = ctx_.isA5 != 0 ? aNv : static_cast<int64_t>(nv);
        int64_t usedUbBytes = CalcFixedUbBytes(aNv, aDv, aDk, nv);
        usedUbBytes += RKDA_MAX_MTP * (4 * aDv + 12 * aDk + 4 * betaHeads);
        return usedUbBytes;
    }

    int64_t CalcComputeUbCoeff(int64_t aDk) const
    {
        int64_t coeff = 4 * aDk + 8; // state and row outputs.
        if (ctx_.isA5 == 0 || aDk != RKDA_FUSED_REDUCE_K) {
            coeff += 4 * aDk; // broadTmp for the generic matvec path.
        }
        return coeff;
    }

    int64_t CalcVStepCoeff(int64_t aDk, uint32_t stateOutBufferNum, uint32_t attnOutBufferNum) const
    {
        const int64_t maxMtp = ctx_.isA5 != 0 ? static_cast<int64_t>(RKDA_A5_MAX_MTP) :
                                               static_cast<int64_t>(RKDA_MAX_MTP);
        int64_t stateDtypeSize = (ctx_.stateDtype == ge::DT_FLOAT) ? 4 : 2;
        int64_t coeff = (ctx_.isA5 != 0 ? 1 : RKDA_INPUT_BUFFER_NUM) * stateDtypeSize * aDk; // state input queue.
        if (ctx_.stateVFirst == 0 && !(ctx_.isA5 != 0 && ctx_.stateDtype == ge::DT_FLOAT)) {
            coeff += stateDtypeSize * aDk; // BF16 or non-A5 K-first transpose buffer.
        }
        coeff += static_cast<int64_t>(stateOutBufferNum) * stateDtypeSize * aDk;
        coeff += static_cast<int64_t>(attnOutBufferNum) * 2 * (ctx_.isA5 != 0 ? maxMtp : 1);
        coeff += CalcComputeUbCoeff(aDk);
        return coeff;
    }

    bool EvaluateBufferProfile(int64_t ubSize, int64_t usedUbBytes, int64_t aDk, uint32_t stateOutBufferNum,
                               uint32_t attnOutBufferNum, const RecurrentKdaTilingData &tiling,
                               BufferProfile &profile) const
    {
        const int64_t maxMtp = ctx_.isA5 != 0 ? static_cast<int64_t>(RKDA_A5_MAX_MTP) :
                                               static_cast<int64_t>(RKDA_MAX_MTP);
        const int64_t vStepAlignment = ctx_.stateVFirst == 0 ? 16 : 8;
        int64_t coeff = CalcVStepCoeff(aDk, stateOutBufferNum, attnOutBufferNum);
        int64_t vStep = (ubSize - usedUbBytes) / coeff / vStepAlignment * vStepAlignment;
        if (vStep < maxMtp) {
            return false;
        }
        int64_t repeatTime = Ops::Base::CeilDiv(tiling.dv, static_cast<uint32_t>(vStep));
        vStep = Ops::Base::CeilAlign(Ops::Base::CeilDiv(tiling.dv, static_cast<uint32_t>(repeatTime)),
                                     static_cast<uint32_t>(vStepAlignment));
        if (vStep < maxMtp) {
            return false;
        }
        profile.stateOutBufferNum = stateOutBufferNum;
        profile.attnOutBufferNum = attnOutBufferNum;
        profile.vStep = static_cast<uint32_t>(vStep);
        profile.repeatTime = static_cast<uint32_t>(repeatTime);
        profile.valid = true;
        return true;
    }

    bool IsBetterProfile(const BufferProfile &candidate, const BufferProfile &current) const
    {
        if (!current.valid) {
            return true;
        }
        if (candidate.repeatTime != current.repeatTime) {
            return candidate.repeatTime < current.repeatTime;
        }
        uint32_t candidateDepth = candidate.stateOutBufferNum + candidate.attnOutBufferNum;
        uint32_t currentDepth = current.stateOutBufferNum + current.attnOutBufferNum;
        if (candidateDepth != currentDepth) {
            return candidateDepth > currentDepth;
        }
        return candidate.vStep > current.vStep;
    }

    ge::graphStatus FinalizeVStepFromUb(int64_t ubSize, int64_t usedUbBytes, int64_t coeff,
                                        RecurrentKdaTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        (void)coeff;
        int64_t aDk = Ops::Base::CeilAlign(tiling.dk, static_cast<uint32_t>(16));
        BufferProfile selected;
        const std::array<BufferProfile, 3> candidates = {{
            {1, 1, 0, 0, false},
            {1, 2, 0, 0, false},
            {2, 2, 0, 0, false},
        }};
        for (const auto &candidate : candidates) {
            BufferProfile profile;
            if (!EvaluateBufferProfile(ubSize, usedUbBytes, aDk, candidate.stateOutBufferNum,
                                       candidate.attnOutBufferNum, tiling, profile)) {
                continue;
            }
            if (IsBetterProfile(profile, selected)) {
                selected = profile;
            }
        }

        if (!selected.valid) {
            ReportInvalid("query/key/value/gate/beta", "vStep should be at least %zu, shape is too big",
                    ctx_.isA5 != 0 ? RKDA_A5_MAX_MTP : RKDA_MAX_MTP);
            return ge::GRAPH_FAILED;
        }

        int64_t queueCoeff = CalcVStepCoeff(aDk, selected.stateOutBufferNum, selected.attnOutBufferNum) -
                             CalcComputeUbCoeff(aDk);
        int64_t ubRestBytes = ubSize - ubCalcCtx.fixedUbBytes -
                              queueCoeff * static_cast<int64_t>(selected.vStep);
        if (ubRestBytes < 0) {
            ReportInvalid("UB/shape", "ubRestBytes should be non-negative, but got %ld", ubRestBytes);
            return ge::GRAPH_FAILED;
        }
        if (ctx_.isA5 == 0) {
            const int64_t tmpBytes = ubCalcCtx.workingUbBytes - ubCalcCtx.fixedUbBytes +
                                     CalcComputeUbCoeff(aDk) * static_cast<int64_t>(selected.vStep);
            OP_CHECK_IF(tmpBytes > ubRestBytes,
                        ReportInvalid("UB/shape", "compute UB bytes %ld exceed available bytes %ld",
                                      tmpBytes, ubRestBytes),
                        return ge::GRAPH_FAILED);
            // Allocate only the named live scratch slices used by the generic kernel.
            ubRestBytes = tmpBytes;
        }
        if (ctx_.isA5 != 0) {
            // Match arch35 InitLocalBuffers: fixed FP32 window tensors plus
            // the live state and row outputs. The unused UB tail is not scratch.
            const int64_t computeBytes = ubCalcCtx.workingUbBytes - ubCalcCtx.fixedUbBytes +
                                         CalcComputeUbCoeff(aDk) * static_cast<int64_t>(selected.vStep);
            const int64_t alignedComputeBytes = Ops::Base::CeilAlign(computeBytes, static_cast<int64_t>(32));
            if (alignedComputeBytes > ubRestBytes) {
                ReportInvalid("UB/shape", "compute UB bytes %ld exceed available bytes %ld",
                        alignedComputeBytes, ubRestBytes);
                return ge::GRAPH_FAILED;
            }
            ubRestBytes = alignedComputeBytes;
        }
        tiling.ubCalSize = static_cast<uint32_t>(ctx_.ubSize);
        tiling.vStep = selected.vStep;
        tiling.stateOutBufferNum = selected.stateOutBufferNum;
        tiling.attnOutBufferNum = selected.attnOutBufferNum;
        tiling.ubRestBytes = static_cast<uint32_t>(ubRestBytes);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleInitUbCalcContext(RecurrentKdaTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        ubCalcCtx.ubSize = static_cast<int64_t>(ctx_.ubSize);
        ubCalcCtx.aNv = Ops::Base::CeilAlign(tiling.nv, static_cast<uint32_t>(16));
        ubCalcCtx.aDv = Ops::Base::CeilAlign(tiling.dv, static_cast<uint32_t>(16));
        ubCalcCtx.aDk = Ops::Base::CeilAlign(tiling.dk, static_cast<uint32_t>(16));
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleCalcFixedUbBytes(RecurrentKdaTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        ubCalcCtx.fixedUbBytes = CalcFixedUbBytes(ubCalcCtx.aNv, ubCalcCtx.aDv, ubCalcCtx.aDk, tiling.nv);
        tiling.ubRestBytes = static_cast<uint32_t>(ubCalcCtx.ubSize - ubCalcCtx.fixedUbBytes);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleCalcWorkingUbBytes(RecurrentKdaTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        ubCalcCtx.workingUbBytes = CalcWorkingUbBytes(ubCalcCtx.aNv, ubCalcCtx.aDv, ubCalcCtx.aDk, tiling.nv);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleCalcVStepCoeff(RecurrentKdaTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        (void)tiling;
        ubCalcCtx.coeff = CalcVStepCoeff(ubCalcCtx.aDk, 1, 1);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleFinalizeVStepFromUb(RecurrentKdaTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        return FinalizeVStepFromUb(ubCalcCtx.ubSize, ubCalcCtx.workingUbBytes, ubCalcCtx.coeff, tiling, ubCalcCtx);
    }
};

} // namespace optiling

#endif // RECURRENT_KDA_TILING_PROCESSOR_H
