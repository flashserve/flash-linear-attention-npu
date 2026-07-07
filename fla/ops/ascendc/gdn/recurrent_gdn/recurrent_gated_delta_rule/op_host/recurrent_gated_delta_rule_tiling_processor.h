/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_gated_delta_rule_tiling_processor.h
 * \brief Tiling processor shared by aclnn tiling and fast kernel launch.
 */

#ifndef RECURRENT_GATED_DELTA_RULE_TILING_PROCESSOR_H
#define RECURRENT_GATED_DELTA_RULE_TILING_PROCESSOR_H

#include "../op_kernel/recurrent_gated_delta_rule_struct.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

using RecurrentGatedDeltaRuleTilingData = RecurrentGatedDeltaRule::RecurrentGatedDeltaRuleTilingData;

namespace optiling {

static constexpr size_t RGDR_QKV_DIM_NUM = 3;
static constexpr size_t RGDR_BETA_DIM_NUM = 2;
static constexpr size_t RGDR_STATE_DIM_NUM = 4;
static constexpr size_t RGDR_CUSEQLENS_DIM_NUM = 1;
static constexpr size_t RGDR_SSM_STATE_INDICES_DIM_NUM = 1;

static constexpr size_t RGDR_DIM_0 = 0;
static constexpr size_t RGDR_DIM_1 = 1;
static constexpr size_t RGDR_DIM_2 = 2;
static constexpr size_t RGDR_DIM_3 = 3;

static constexpr size_t RGDR_MAX_MTP = 8;
static constexpr size_t RGDR_SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;

struct RecurrentGatedDeltaRuleTilingContext {
    const char *nodeName = "RecurrentGatedDeltaRule";
    const gert::Shape *queryShape = nullptr;
    const gert::Shape *keyShape = nullptr;
    const gert::Shape *valueShape = nullptr;
    const gert::Shape *betaShape = nullptr;
    const gert::Shape *stateShape = nullptr;
    const gert::Shape *cuSeqlensShape = nullptr;
    const gert::Shape *ssmStateShape = nullptr;
    float scale = 1.0f;
    uint32_t hasGama = 0;
    uint32_t hasGamaK = 0;
    uint32_t hasAcceptedTokens = 0;
    ge::DataType stateDtype = ge::DT_BF16;
    uint64_t aivNum = 0;
    uint64_t ubSize = 0;
};

class RecurrentGatedDeltaRuleTilingProcessor {
public:
    explicit RecurrentGatedDeltaRuleTilingProcessor(const RecurrentGatedDeltaRuleTilingContext &ctx) : ctx_(ctx) {}

    ge::graphStatus ProcessShapes(RecurrentGatedDeltaRuleTilingData &tiling) const
    {
        tiling.vectorCoreNum = static_cast<uint32_t>(ctx_.aivNum);

        struct RuleItem {
            const char *name;
            ge::graphStatus (RecurrentGatedDeltaRuleTilingProcessor::*fn)(RecurrentGatedDeltaRuleTilingData &) const;
        };
        const std::array<RuleItem, 4> shapeRules = {{
            {"RuleCheckShapeDimAndRelation", &RecurrentGatedDeltaRuleTilingProcessor::RuleCheckShapeDimAndRelation},
            {"RuleFillTilingShapeData", &RecurrentGatedDeltaRuleTilingProcessor::RuleFillTilingShapeData},
            {"RuleCheckShapeValueRangeAndRule", &RecurrentGatedDeltaRuleTilingProcessor::RuleCheckShapeValueRangeAndRule},
            {"RuleUpdateDynamicBlockDimByTaskUnits", &RecurrentGatedDeltaRuleTilingProcessor::RuleUpdateDynamicBlockDimByTaskUnits},
        }};
        for (const auto &rule : shapeRules) {
            OP_CHECK_IF((this->*(rule.fn))(tiling) != ge::GRAPH_SUCCESS,
                        OP_LOGE(ctx_.nodeName, "ProcessShapes rule failed: %s", rule.name),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ProcessUb(RecurrentGatedDeltaRuleTilingData &tiling) const
    {
        struct RuleItem {
            const char *name;
            ge::graphStatus (RecurrentGatedDeltaRuleTilingProcessor::*fn)(RecurrentGatedDeltaRuleTilingData &,
                                                                          UbCalcContext &) const;
        };
        UbCalcContext ubCalcCtx;
        const std::array<RuleItem, 5> ubRules = {{
            {"RuleInitUbCalcContext", &RecurrentGatedDeltaRuleTilingProcessor::RuleInitUbCalcContext},
            {"RuleCalcFixedUbBytes", &RecurrentGatedDeltaRuleTilingProcessor::RuleCalcFixedUbBytes},
            {"RuleCalcWorkingUbBytes", &RecurrentGatedDeltaRuleTilingProcessor::RuleCalcWorkingUbBytes},
            {"RuleCalcVStepCoeff", &RecurrentGatedDeltaRuleTilingProcessor::RuleCalcVStepCoeff},
            {"RuleFinalizeVStepFromUb", &RecurrentGatedDeltaRuleTilingProcessor::RuleFinalizeVStepFromUb},
        }};
        for (const auto &rule : ubRules) {
            OP_CHECK_IF((this->*(rule.fn))(tiling, ubCalcCtx) != ge::GRAPH_SUCCESS,
                        OP_LOGE(ctx_.nodeName, "ProcessUb rule failed: %s", rule.name),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus Process(RecurrentGatedDeltaRuleTilingData &tiling, uint32_t &blockDim, size_t &workspaceSize) const
    {
        OP_CHECK_IF(ProcessShapes(tiling) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

        tiling.scale = ctx_.scale;
        tiling.hasGama = ctx_.hasGama;
        tiling.hasGamaK = ctx_.hasGamaK;
        tiling.hasAcceptedTokens = ctx_.hasAcceptedTokens;

        OP_CHECK_IF(ProcessUb(tiling) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

        blockDim = tiling.vectorCoreNum;
        workspaceSize = RGDR_SYS_WORKSPACE_SIZE;
        return ge::GRAPH_SUCCESS;
    }

private:
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

    const RecurrentGatedDeltaRuleTilingContext &ctx_;

    bool CheckDimEqual(const gert::Shape a, const int64_t dimA, gert::Shape b, const int64_t dimB,
                       const std::string &nameA, const std::string &nameB, const std::string &dimDesc) const
    {
        if (a.GetDim(dimA) != b.GetDim(dimB)) {
            OP_LOGE(ctx_.nodeName, "The %s of %s and %s should be the same, but %s is %ld while %s is %ld",
                    dimDesc.c_str(), nameA.c_str(), nameB.c_str(), nameA.c_str(), a.GetDim(dimA), nameB.c_str(),
                    b.GetDim(dimB));
            return false;
        }
        return true;
    }

    bool CheckDim(const gert::Shape shape, const size_t dim, const std::string &dimDesc) const
    {
        if (shape.GetDimNum() != dim) {
            OP_LOGE(ctx_.nodeName, "The number of dimensons of %s should be %zu, but it is %zu", dimDesc.c_str(), dim,
                    shape.GetDimNum());
            return false;
        }
        return true;
    }

    ge::graphStatus CheckShapeDimAndRelation(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                             const gert::Shape &valueShape, const gert::Shape &betaShape,
                                             const gert::Shape &stateShape, const gert::Shape &cuSeqlensShape,
                                             const gert::Shape &ssmStateShape) const
    {
        if (!CheckDim(queryShape, RGDR_QKV_DIM_NUM, "query") || !CheckDim(keyShape, RGDR_QKV_DIM_NUM, "key") ||
            !CheckDim(valueShape, RGDR_QKV_DIM_NUM, "value") || !CheckDim(betaShape, RGDR_BETA_DIM_NUM, "beta") ||
            !CheckDim(stateShape, RGDR_STATE_DIM_NUM, "state") ||
            !CheckDim(cuSeqlensShape, RGDR_CUSEQLENS_DIM_NUM, "actual_seq_lengths") ||
            !CheckDim(ssmStateShape, RGDR_SSM_STATE_INDICES_DIM_NUM, "ssm_state_indices")) {
            return ge::GRAPH_FAILED;
        }

        if (!CheckDimEqual(queryShape, RGDR_DIM_0, keyShape, RGDR_DIM_0, "query", "key", "T dimension") ||
            !CheckDimEqual(queryShape, RGDR_DIM_1, keyShape, RGDR_DIM_1, "query", "key", "Nk dimension") ||
            !CheckDimEqual(queryShape, RGDR_DIM_2, keyShape, RGDR_DIM_2, "query", "key", "Dk dimension") ||
            !CheckDimEqual(stateShape, RGDR_DIM_1, valueShape, RGDR_DIM_1, "state", "value", "Nv dimension") ||
            !CheckDimEqual(stateShape, RGDR_DIM_2, valueShape, RGDR_DIM_2, "state", "value", "Dv dimension") ||
            !CheckDimEqual(valueShape, RGDR_DIM_0, queryShape, RGDR_DIM_0, "value", "query", "T dimension") ||
            !CheckDimEqual(betaShape, RGDR_DIM_0, queryShape, RGDR_DIM_0, "beta", "query", "T dimension") ||
            !CheckDimEqual(betaShape, RGDR_DIM_1, valueShape, RGDR_DIM_1, "beta", "value", "Nv dimension") ||
            !CheckDimEqual(stateShape, RGDR_DIM_3, queryShape, RGDR_DIM_2, "state", "query", "Dk dimension")) {
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    void FillTilingShapeData(const gert::Shape &queryShape, const gert::Shape &valueShape, const gert::Shape &stateShape,
                             const gert::Shape &cuSeqlensShape, RecurrentGatedDeltaRuleTilingData &tiling) const
    {
        tiling.t = static_cast<uint32_t>(queryShape.GetDim(RGDR_DIM_0));
        tiling.nk = static_cast<uint32_t>(queryShape.GetDim(RGDR_DIM_1));
        tiling.dk = static_cast<uint32_t>(queryShape.GetDim(RGDR_DIM_2));
        tiling.nv = static_cast<uint32_t>(valueShape.GetDim(RGDR_DIM_1));
        tiling.dv = static_cast<uint32_t>(valueShape.GetDim(RGDR_DIM_2));
        tiling.sBlockNum = static_cast<uint32_t>(stateShape.GetDim(RGDR_DIM_0));
        tiling.b = static_cast<uint32_t>(cuSeqlensShape.GetDim(RGDR_DIM_0) - 1);
    }

    ge::graphStatus CheckShapeValueRangeAndRule(const RecurrentGatedDeltaRuleTilingData &tiling) const
    {
        OP_CHECK_IF(tiling.nk > 256 || tiling.nv > 256 || tiling.dk > 512 || tiling.dv > 512,
                    OP_LOGE(ctx_.nodeName,
                            "nk and nv should no bigger than 256, dk and dv should no bigger than 512, but nk is %u, "
                            "nv is %u, dk is %u, dv is %u",
                            tiling.nk, tiling.nv, tiling.dk, tiling.dv),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(tiling.nv % tiling.nk != 0,
                    OP_LOGE(ctx_.nodeName, "nv should be an integer multiple of nk, but nv is %u, nk is %u", tiling.nv,
                            tiling.nk),
                    return ge::GRAPH_FAILED);

        return ge::GRAPH_SUCCESS;
    }

    void UpdateDynamicBlockDimByTaskUnits(RecurrentGatedDeltaRuleTilingData &tiling) const
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

    ge::graphStatus RuleCheckShapeDimAndRelation(RecurrentGatedDeltaRuleTilingData &tiling) const
    {
        (void)tiling;
        OP_CHECK_IF(ctx_.queryShape == nullptr || ctx_.keyShape == nullptr || ctx_.valueShape == nullptr ||
                        ctx_.betaShape == nullptr || ctx_.stateShape == nullptr || ctx_.cuSeqlensShape == nullptr ||
                        ctx_.ssmStateShape == nullptr,
                    OP_LOGE(ctx_.nodeName, "Required shape pointer is null."), return ge::GRAPH_FAILED);
        return CheckShapeDimAndRelation(*ctx_.queryShape, *ctx_.keyShape, *ctx_.valueShape, *ctx_.betaShape,
                                        *ctx_.stateShape, *ctx_.cuSeqlensShape, *ctx_.ssmStateShape);
    }

    ge::graphStatus RuleFillTilingShapeData(RecurrentGatedDeltaRuleTilingData &tiling) const
    {
        (void)tiling;
        FillTilingShapeData(*ctx_.queryShape, *ctx_.valueShape, *ctx_.stateShape, *ctx_.cuSeqlensShape, tiling);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleCheckShapeValueRangeAndRule(RecurrentGatedDeltaRuleTilingData &tiling) const
    {
        return CheckShapeValueRangeAndRule(tiling);
    }

    ge::graphStatus RuleUpdateDynamicBlockDimByTaskUnits(RecurrentGatedDeltaRuleTilingData &tiling) const
    {
        UpdateDynamicBlockDimByTaskUnits(tiling);
        return ge::GRAPH_SUCCESS;
    }

    int64_t CalcFixedUbBytes(int64_t aNv, int64_t aDv, int64_t aDk, const RecurrentGatedDeltaRuleTilingData &tiling) const
    {
        int64_t usedUbBytes = RGDR_MAX_MTP * (4 * aDk + 2 * aDv);
        usedUbBytes += 128;
        if (tiling.hasGamaK) {
            usedUbBytes += RGDR_MAX_MTP * 4 * aDk;
        }
        if (tiling.hasGama) {
            usedUbBytes += RGDR_MAX_MTP * 4 * aNv;
        }
        usedUbBytes += RGDR_MAX_MTP * 2 * aNv;
        return usedUbBytes;
    }

    int64_t CalcWorkingUbBytes(int64_t aNv, int64_t aDv, int64_t aDk,
                               const RecurrentGatedDeltaRuleTilingData &tiling) const
    {
        int64_t usedUbBytes = CalcFixedUbBytes(aNv, aDv, aDk, tiling);
        usedUbBytes += RGDR_MAX_MTP * (8 * aDk + 4 * aDv + 4 * aNv);
        return usedUbBytes;
    }

    int64_t CalcVStepCoeff(int64_t aDk, uint32_t stateOutBufferNum, uint32_t attnOutBufferNum) const
    {
        int64_t stateDtypeSize = (ctx_.stateDtype == ge::DT_FLOAT) ? 4 : 2;
        int64_t coeff = (stateDtypeSize + static_cast<int64_t>(stateDtypeSize * stateOutBufferNum)) * aDk +
                        static_cast<int64_t>(4 * attnOutBufferNum);
        coeff += (4 + 4) * aDk + 4 + 4;
        return coeff;
    }

    bool EvaluateBufferProfile(int64_t ubSize, int64_t usedUbBytes, int64_t aDk, uint32_t stateOutBufferNum,
                               uint32_t attnOutBufferNum, const RecurrentGatedDeltaRuleTilingData &tiling,
                               BufferProfile &profile) const
    {
        int64_t coeff = CalcVStepCoeff(aDk, stateOutBufferNum, attnOutBufferNum);
        int64_t vStep = (ubSize - usedUbBytes) / coeff / 8 * 8;
        if (vStep < 8) {
            return false;
        }
        int64_t repeatTime = Ops::Base::CeilDiv(tiling.dv, static_cast<uint32_t>(vStep));
        vStep = Ops::Base::CeilAlign(Ops::Base::CeilDiv(tiling.dv, static_cast<uint32_t>(repeatTime)),
                                     static_cast<uint32_t>(8));
        if (vStep < 8) {
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
                                        RecurrentGatedDeltaRuleTilingData &tiling, UbCalcContext &ubCalcCtx) const
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
            if (!EvaluateBufferProfile(ubSize, usedUbBytes, aDk, candidate.stateOutBufferNum, candidate.attnOutBufferNum,
                                       tiling, profile)) {
                continue;
            }
            if (IsBetterProfile(profile, selected)) {
                selected = profile;
            }
        }

        OP_LOGD(ctx_.nodeName,
                "selected profile: stateOutBufferNum=[%u], attnOutBufferNum=[%u], vStep=[%u], repeatTime=[%u], "
                "valid=[%d]",
                selected.stateOutBufferNum, selected.attnOutBufferNum, selected.vStep, selected.repeatTime,
                selected.valid);

        if (!selected.valid) {
            OP_LOGE(ctx_.nodeName, "vStep should be bigger than 8, shape is too big");
            return ge::GRAPH_FAILED;
        }

        int64_t stateDtypeSize = (ctx_.stateDtype == ge::DT_FLOAT) ? 4 : 2;
        int64_t queueCoeff = (stateDtypeSize + static_cast<int64_t>(stateDtypeSize * selected.stateOutBufferNum)) * aDk +
                             static_cast<int64_t>(4 * selected.attnOutBufferNum);
        int64_t ubRestBytes = ubSize - ubCalcCtx.fixedUbBytes - queueCoeff * static_cast<int64_t>(selected.vStep);
        if (ubRestBytes < 0) {
            OP_LOGE(ctx_.nodeName, "ubRestBytes should be non-negative, but got %ld", ubRestBytes);
            return ge::GRAPH_FAILED;
        }
        tiling.ubCalSize = static_cast<uint32_t>(ctx_.ubSize);
        tiling.vStep = selected.vStep;
        tiling.stateOutBufferNum = selected.stateOutBufferNum;
        tiling.attnOutBufferNum = selected.attnOutBufferNum;
        tiling.ubRestBytes = static_cast<uint32_t>(ubRestBytes);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleInitUbCalcContext(RecurrentGatedDeltaRuleTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        ubCalcCtx.ubSize = static_cast<int64_t>(ctx_.ubSize);
        ubCalcCtx.aNv = Ops::Base::CeilAlign(tiling.nv, static_cast<uint32_t>(16));
        ubCalcCtx.aDv = Ops::Base::CeilAlign(tiling.dv, static_cast<uint32_t>(16));
        ubCalcCtx.aDk = Ops::Base::CeilAlign(tiling.dk, static_cast<uint32_t>(16));
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleCalcFixedUbBytes(RecurrentGatedDeltaRuleTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        ubCalcCtx.fixedUbBytes = CalcFixedUbBytes(ubCalcCtx.aNv, ubCalcCtx.aDv, ubCalcCtx.aDk, tiling);
        tiling.ubRestBytes = static_cast<uint32_t>(ubCalcCtx.ubSize - ubCalcCtx.fixedUbBytes);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleCalcWorkingUbBytes(RecurrentGatedDeltaRuleTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        ubCalcCtx.workingUbBytes = CalcWorkingUbBytes(ubCalcCtx.aNv, ubCalcCtx.aDv, ubCalcCtx.aDk, tiling);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleCalcVStepCoeff(RecurrentGatedDeltaRuleTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        (void)tiling;
        ubCalcCtx.coeff = CalcVStepCoeff(ubCalcCtx.aDk, 1, 1);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RuleFinalizeVStepFromUb(RecurrentGatedDeltaRuleTilingData &tiling, UbCalcContext &ubCalcCtx) const
    {
        return FinalizeVStepFromUb(ubCalcCtx.ubSize, ubCalcCtx.workingUbBytes, ubCalcCtx.coeff, tiling, ubCalcCtx);
    }
};

} // namespace optiling

#endif // RECURRENT_GATED_DELTA_RULE_TILING_PROCESSOR_H
