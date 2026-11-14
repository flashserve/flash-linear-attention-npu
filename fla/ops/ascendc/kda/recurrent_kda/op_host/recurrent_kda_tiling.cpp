/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_kda_tiling.cpp
 * \brief
 */
#include "recurrent_kda_tiling.h"
#include "../op_kernel/recurrent_kda_tiling_key.h"

#include "err/ops_err.h"
#include "platform/soc_spec.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
#include <cstring>
#include <string>

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(RecurrentKda, RecurrentKdaTiling, 0);

const size_t QUERY_INDEX = 0;
const size_t KEY_INDEX = 1;
const size_t VALUE_INDEX = 2;
const size_t GATE_INDEX = 3;
const size_t BETA_INDEX = 4;
const size_t STATE_INDEX = 5;
const size_t CU_SEQLENS_INDEX = 6;
const size_t SSM_STATE_INDICES_INDEX = 7;
const size_t A_LOG_INDEX = 8;
const size_t DT_BIAS_INDEX = 9;
const size_t ACC_TOKEN_INDEX = 10;

const size_t INITIAL_STATE_OUTPUT_INDEX = 1;
const size_t FINAL_STATE_OUTPUT_INDEX = 2;

const size_t ATTR_LAYOUT_INDEX = 0;
const size_t ATTR_SCALE_INDEX = 1;
const size_t ATTR_OUTPUT_FINAL_STATE_INDEX = 2;
const size_t ATTR_INPLACE_FINAL_STATE_INDEX = 3;
const size_t ATTR_USE_QK_L2NORM_INDEX = 4;
const size_t ATTR_USE_GATE_INDEX = 5;
const size_t ATTR_USE_BETA_SIGMOID_INDEX = 6;
const size_t ATTR_ALLOW_NEG_EIGVAL_INDEX = 7;
const size_t ATTR_SAFE_GATE_INDEX = 8;
const size_t ATTR_LOWER_BOUND_INDEX = 9;
const size_t ATTR_STATE_V_FIRST_INDEX = 10;

void RecurrentKdaTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "platform_info");
        return;
    }
    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo_.isA5 = ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510;

    if (compileInfo_.aivNum <= 0) {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(),
            "%s: platform reported no AIV core; UB bytes=%llu.", __func__,
            static_cast<unsigned long long>(compileInfo_.ubSize));
        return;
    }
    tilingData_.vectorCoreNum = static_cast<uint32_t>(compileInfo_.aivNum);
}

namespace {
void CopyOptionalOriginShape(gert::TilingContext *context, size_t index, gert::Shape &dst)
{
    auto shape = context->GetOptionalInputShape(index);
    if (shape != nullptr) {
        dst = shape->GetOriginShape();
    }
}

template <typename StrideType>
void CopyStateStrides(const StrideType *src, std::array<int64_t, RKDA_STATE_DIM_NUM> &dst, bool &hasStrides)
{
    if (src == nullptr || src->GetDimNum() != RKDA_STATE_DIM_NUM) {
        return;
    }
    for (size_t i = 0; i < RKDA_STATE_DIM_NUM; ++i) {
        dst[i] = src->GetStride(i);
    }
    hasStrides = true;
}

template <typename StrideType>
void CopyQkvStrides(const StrideType *src, size_t expectedDimNum,
                    std::array<int64_t, RKDA_RANK4_QKV_DIM_NUM> &dst, bool &hasStrides)
{
    if (src == nullptr || src->GetDimNum() != expectedDimNum ||
        expectedDimNum > RKDA_RANK4_QKV_DIM_NUM) {
        return;
    }
    for (size_t i = 0; i < expectedDimNum; ++i) {
        dst[i] = src->GetStride(i);
    }
    hasStrides = true;
}
} // namespace

RecurrentKdaTilingContext RecurrentKdaTiling::BuildProcessorContext() const
{
    RecurrentKdaTilingContext ctx;
    ctx.nodeName = context_->GetNodeName();
    ctx.queryShape = context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
    ctx.keyShape = context_->GetInputShape(KEY_INDEX)->GetOriginShape();
    ctx.valueShape = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
    CopyQkvStrides(context_->GetInputStride(QUERY_INDEX), ctx.queryShape.GetDimNum(),
                   ctx.queryStrides, ctx.hasQueryStrides);
    CopyQkvStrides(context_->GetInputStride(KEY_INDEX), ctx.keyShape.GetDimNum(),
                   ctx.keyStrides, ctx.hasKeyStrides);
    CopyQkvStrides(context_->GetInputStride(VALUE_INDEX), ctx.valueShape.GetDimNum(),
                   ctx.valueStrides, ctx.hasValueStrides);
    ctx.gateShape = context_->GetInputShape(GATE_INDEX)->GetOriginShape();
    ctx.betaShape = context_->GetInputShape(BETA_INDEX)->GetOriginShape();
    ctx.stateShape = context_->GetInputShape(STATE_INDEX)->GetOriginShape();
    CopyStateStrides(context_->GetInputStride(STATE_INDEX), ctx.stateInStrides, ctx.hasStateInStrides);
    const size_t stateOutputIndex = tilingData_.inplaceFinalState == 1 ?
                                    INITIAL_STATE_OUTPUT_INDEX : FINAL_STATE_OUTPUT_INDEX;
    CopyStateStrides(context_->GetOutputStride(stateOutputIndex), ctx.stateOutStrides, ctx.hasStateOutStrides);
    if (!ctx.hasStateOutStrides && tilingData_.inplaceFinalState == 1 && ctx.hasStateInStrides) {
        ctx.stateOutStrides = ctx.stateInStrides;
        ctx.hasStateOutStrides = true;
    }
    CopyOptionalOriginShape(context_, CU_SEQLENS_INDEX, ctx.cuSeqlensShape);
    CopyOptionalOriginShape(context_, SSM_STATE_INDICES_INDEX, ctx.ssmStateShape);
    CopyOptionalOriginShape(context_, A_LOG_INDEX, ctx.aLogShape);
    CopyOptionalOriginShape(context_, DT_BIAS_INDEX, ctx.dtBiasShape);
    CopyOptionalOriginShape(context_, ACC_TOKEN_INDEX, ctx.acceptedTokensShape);
    ctx.aivNum = compileInfo_.aivNum;
    ctx.ubSize = compileInfo_.ubSize;
    ctx.isA5 = compileInfo_.isA5 ? 1 : 0;
    ctx.stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
    ctx.gateDtype = context_->GetInputDesc(GATE_INDEX)->GetDataType();
    ctx.betaDtype = context_->GetInputDesc(BETA_INDEX)->GetDataType();
    if (context_->GetOptionalInputDesc(CU_SEQLENS_INDEX) != nullptr) {
        ctx.cuSeqlensDtype = context_->GetOptionalInputDesc(CU_SEQLENS_INDEX)->GetDataType();
    }
    if (context_->GetOptionalInputDesc(SSM_STATE_INDICES_INDEX) != nullptr) {
        ctx.ssmStateIndicesDtype = context_->GetOptionalInputDesc(SSM_STATE_INDICES_INDEX)->GetDataType();
    }
    if (context_->GetOptionalInputDesc(ACC_TOKEN_INDEX) != nullptr) {
        ctx.acceptedTokensDtype = context_->GetOptionalInputDesc(ACC_TOKEN_INDEX)->GetDataType();
    }
    ctx.scale = tilingData_.scale;
    ctx.lowerBound = tilingData_.lowerBound;
    ctx.hasCuSeqlens = tilingData_.hasCuSeqlens;
    ctx.layout = tilingData_.layout;
    ctx.hasSsmStateIndices = tilingData_.hasSsmStateIndices;
    ctx.hasALog = tilingData_.hasALog;
    ctx.hasDtBias = tilingData_.hasDtBias;
    ctx.hasAcceptedTokens = tilingData_.hasAcceptedTokens;
    ctx.useQkL2norm = tilingData_.useQkL2norm;
    ctx.useGateInKernel = tilingData_.useGateInKernel;
    ctx.useBetaSigmoid = tilingData_.useBetaSigmoid;
    ctx.allowNegEigval = tilingData_.allowNegEigval;
    ctx.safeGate = tilingData_.safeGate;
    ctx.stateVFirst = tilingData_.stateVFirst;
    ctx.outputFinalState = tilingData_.outputFinalState;
    ctx.inplaceFinalState = tilingData_.inplaceFinalState;
    return ctx;
}

ge::graphStatus RecurrentKdaTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus RecurrentKdaTiling::GetShapeAttrsInfo()
{
    if (CheckContext() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (GetOptionalInput() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (GetAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (AnalyzeDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (AnalyzeShapes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (AnalyzeFormat() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentKdaTiling::DoOpTiling()
{
    if (CalUbSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentKdaTiling::DoLibApiTiling()
{
    uint32_t kernelMode = RKDA_TPL_COMPAT;
    if (compileInfo_.isA5) {
        if (tilingData_.stateVFirst) {
            kernelMode = (tilingData_.a5VfFlags & RecurrentKda::RKDA_VF_FUSED) != 0
                ? RKDA_TPL_V_FIRST_FUSED : RKDA_TPL_V_FIRST;
        } else {
            kernelMode = (tilingData_.a5VfFlags & RecurrentKda::RKDA_VF_FUSED) != 0
                ? RKDA_TPL_K_FIRST_FUSED : RKDA_TPL_K_FIRST;
        }
    }
    tilingKey_ = GET_TPL_TILING_KEY(kernelMode);
    return ge::GRAPH_SUCCESS;
};

uint64_t RecurrentKdaTiling::GetTilingKey() const
{
    return tilingKey_;
};

ge::graphStatus RecurrentKdaTiling::GetWorkspaceSize()
{
    workspaceSize_ = static_cast<int64_t>(RKDA_SYS_WORKSPACE_SIZE);
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus RecurrentKdaTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.vectorCoreNum);
    auto tilingDataSize = sizeof(RecurrentKdaTilingData);
    auto *rawTilingData = context_->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetData() == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "raw_tiling_data");
        return ge::GRAPH_FAILED;
    }
    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "raw_tiling_data",
            "PostTiling serialization failed: memcpy_s rc=" + std::to_string(ret) +
            "; requiredBytes=" + std::to_string(tilingDataSize) +
            "; capacityBytes=" + std::to_string(rawTilingData->GetCapacity()) +
            "; tilingKey=" + std::to_string(tilingKey_) +
            ". Provide a writable tiling buffer with sufficient capacity.");
        return ge::GRAPH_FAILED;
    }
    rawTilingData->SetDataSize(tilingDataSize);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "workspace_sizes");
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentKdaTiling::CheckContext()
{
    const std::pair<size_t, const char *> required[] = {
        {QUERY_INDEX, "query"}, {KEY_INDEX, "key"}, {VALUE_INDEX, "value"},
        {GATE_INDEX, "gate"}, {BETA_INDEX, "beta"}, {STATE_INDEX, "initial_state"},
    };
    for (const auto &item : required) {
        if (context_->GetInputShape(item.first) == nullptr ||
                    context_->GetInputDesc(item.first) == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), item.second, "input argument must not be nullptr");
        return ge::GRAPH_FAILED;
    }
    }
     return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentKdaTiling::AnalyzeDtype()
{
    auto queryDtype = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto keyDtype = context_->GetInputDesc(KEY_INDEX)->GetDataType();
    auto valueDtype = context_->GetInputDesc(VALUE_INDEX)->GetDataType();
    if (queryDtype != ge::DT_BF16 || keyDtype != ge::DT_BF16 || valueDtype != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "query/key/value",
                    std::to_string(queryDtype) + "/" + std::to_string(keyDtype) + "/" + std::to_string(valueDtype),
                    "all three inputs must be BF16");
        return ge::GRAPH_FAILED;
    }

    auto gateDtype = context_->GetInputDesc(GATE_INDEX)->GetDataType();
    auto betaDtype = context_->GetInputDesc(BETA_INDEX)->GetDataType();
    auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
    if ((gateDtype != ge::DT_FLOAT && gateDtype != ge::DT_BF16 && gateDtype != ge::DT_FLOAT16) ||
                    (betaDtype != ge::DT_FLOAT && betaDtype != ge::DT_BF16 && betaDtype != ge::DT_FLOAT16)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "gate/beta",
                    std::to_string(gateDtype) + "/" + std::to_string(betaDtype),
                    "both inputs must be FP32/BF16/FP16");
        return ge::GRAPH_FAILED;
    }
    if (stateDtype != ge::DT_FLOAT && stateDtype != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "initial_state",
                    std::to_string(stateDtype), "BF16/FP32");
        return ge::GRAPH_FAILED;
    }
    if (context_->GetOptionalInputDesc(CU_SEQLENS_INDEX) != nullptr) {
        auto dtype = context_->GetOptionalInputDesc(CU_SEQLENS_INDEX)->GetDataType();
        if (dtype != ge::DT_INT32 && dtype != ge::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "cu_seqlens",
                        std::to_string(dtype), "INT32/INT64");
        return ge::GRAPH_FAILED;
    }
    }
    if (context_->GetOptionalInputDesc(SSM_STATE_INDICES_INDEX) != nullptr) {
        auto dtype = context_->GetOptionalInputDesc(SSM_STATE_INDICES_INDEX)->GetDataType();
        if (dtype != ge::DT_INT32 && dtype != ge::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "ssm_state_indices",
                        std::to_string(dtype), "INT32/INT64");
        return ge::GRAPH_FAILED;
    }
    }
    if (context_->GetOptionalInputDesc(A_LOG_INDEX) != nullptr) {
        if (context_->GetOptionalInputDesc(A_LOG_INDEX)->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "A_log",
                        std::to_string(context_->GetOptionalInputDesc(A_LOG_INDEX)->GetDataType()), "FP32");
        return ge::GRAPH_FAILED;
    }
    }
    if (tilingData_.hasDtBias) {
        if (context_->GetOptionalInputDesc(DT_BIAS_INDEX)->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dt_bias",
                        std::to_string(context_->GetOptionalInputDesc(DT_BIAS_INDEX)->GetDataType()), "FP32");
        return ge::GRAPH_FAILED;
    }
    }
    if (context_->GetOptionalInputDesc(ACC_TOKEN_INDEX) != nullptr) {
        auto dtype = context_->GetOptionalInputDesc(ACC_TOKEN_INDEX)->GetDataType();
        if (dtype != ge::DT_INT32 && dtype != ge::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "num_accepted_tokens",
                        std::to_string(dtype), "INT32/INT64");
        return ge::GRAPH_FAILED;
    }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentKdaTiling::AnalyzeShapes()
{
    RecurrentKdaTilingProcessor processor(BuildProcessorContext());
    return processor.ProcessShapes(tilingData_);
}

bool RecurrentKdaTiling::CheckFormat(ge::Format format, const std::string &desc)
{
    if (format == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context_->GetNodeName(), desc,
            "FORMAT_FRACTAL_NZ", "expected a plain format");
        return false;
    }
    return true;
}

ge::graphStatus RecurrentKdaTiling::AnalyzeFormat()
{
    if (!CheckFormat(context_->GetInputDesc(QUERY_INDEX)->GetStorageFormat(), "query") ||
        !CheckFormat(context_->GetInputDesc(KEY_INDEX)->GetStorageFormat(), "key") ||
        !CheckFormat(context_->GetInputDesc(VALUE_INDEX)->GetStorageFormat(), "value") ||
        !CheckFormat(context_->GetInputDesc(GATE_INDEX)->GetStorageFormat(), "gate") ||
        !CheckFormat(context_->GetInputDesc(BETA_INDEX)->GetStorageFormat(), "beta") ||
        !CheckFormat(context_->GetInputDesc(STATE_INDEX)->GetStorageFormat(), "initial_state")) {
        return ge::GRAPH_FAILED;
    }

    const std::array<std::pair<size_t, const char *>, 5> optionalInputs = {{
        {CU_SEQLENS_INDEX, "cu_seqlens"},
        {SSM_STATE_INDICES_INDEX, "ssm_state_indices"},
        {A_LOG_INDEX, "A_log"},
        {DT_BIAS_INDEX, "dt_bias"},
        {ACC_TOKEN_INDEX, "num_accepted_tokens"},
    }};
    for (const auto &item : optionalInputs) {
        if (compileInfo_.isA5 && item.first == DT_BIAS_INDEX && !tilingData_.hasALog) continue;
        auto desc = context_->GetOptionalInputDesc(item.first);
        if (desc != nullptr && !CheckFormat(desc->GetStorageFormat(), item.second)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentKdaTiling::GetAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "attrs");
        return ge::GRAPH_FAILED;
    }
    const char *layout = attrs->GetAttrPointer<char>(ATTR_LAYOUT_INDEX);
    if (layout == nullptr || std::strcmp(layout, "BSND") == 0) {
        tilingData_.layout = RKDA_LAYOUT_BSND;
    } else if (std::strcmp(layout, "TND") == 0) {
        tilingData_.layout = RKDA_LAYOUT_TND;
    } else {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "layout",
            "layout must be BSND or TND");
        return ge::GRAPH_FAILED;
    }
    tilingData_.scale = *attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    tilingData_.outputFinalState = *attrs->GetAttrPointer<bool>(ATTR_OUTPUT_FINAL_STATE_INDEX) ? 1 : 0;
    tilingData_.inplaceFinalState = *attrs->GetAttrPointer<bool>(ATTR_INPLACE_FINAL_STATE_INDEX) ? 1 : 0;
    tilingData_.useQkL2norm = *attrs->GetAttrPointer<bool>(ATTR_USE_QK_L2NORM_INDEX) ? 1 : 0;
    tilingData_.useGateInKernel = *attrs->GetAttrPointer<bool>(ATTR_USE_GATE_INDEX) ? 1 : 0;
    tilingData_.useBetaSigmoid = *attrs->GetAttrPointer<bool>(ATTR_USE_BETA_SIGMOID_INDEX) ? 1 : 0;
    tilingData_.allowNegEigval = *attrs->GetAttrPointer<bool>(ATTR_ALLOW_NEG_EIGVAL_INDEX) ? 1 : 0;
    tilingData_.safeGate = *attrs->GetAttrPointer<bool>(ATTR_SAFE_GATE_INDEX) ? 1 : 0;
    tilingData_.lowerBound = *attrs->GetAttrPointer<float>(ATTR_LOWER_BOUND_INDEX);
    tilingData_.stateVFirst = *attrs->GetAttrPointer<bool>(ATTR_STATE_V_FIRST_INDEX) ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentKdaTiling::GetOptionalInput()
{
    tilingData_.hasCuSeqlens = (context_->GetOptionalInputDesc(CU_SEQLENS_INDEX) == nullptr) ? 0 : 1;
    tilingData_.hasSsmStateIndices = (context_->GetOptionalInputDesc(SSM_STATE_INDICES_INDEX) == nullptr) ? 0 : 1;
    tilingData_.hasALog = (context_->GetOptionalInputDesc(A_LOG_INDEX) == nullptr) ? 0 : 1;
    tilingData_.hasDtBias = context_->GetOptionalInputDesc(DT_BIAS_INDEX) != nullptr ? 1 : 0;
    tilingData_.hasAcceptedTokens = (context_->GetOptionalInputDesc(ACC_TOKEN_INDEX) == nullptr) ? 0 : 1;
    return ge::GRAPH_SUCCESS;
}

void RecurrentKdaTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "vectorCoreNum: [%u]", tilingData_.vectorCoreNum);
    OP_LOGD(context_->GetNodeName(), "ubCalSize: [%u]", tilingData_.ubCalSize);
    OP_LOGD(context_->GetNodeName(), "ubRestBytes: [%u]", tilingData_.ubRestBytes);
    OP_LOGD(context_->GetNodeName(), "t: [%u]", tilingData_.t);
    OP_LOGD(context_->GetNodeName(), "seqLen: [%u]", tilingData_.seqLen);
    OP_LOGD(context_->GetNodeName(), "windowSize: [%u]", tilingData_.windowSize);
    OP_LOGD(context_->GetNodeName(), "nk: [%u]", tilingData_.nk);
    OP_LOGD(context_->GetNodeName(), "dk: [%u]", tilingData_.dk);
    OP_LOGD(context_->GetNodeName(), "nv: [%u]", tilingData_.nv);
    OP_LOGD(context_->GetNodeName(), "dv: [%u]", tilingData_.dv);
    OP_LOGD(context_->GetNodeName(), "sBlockNum: [%u]", tilingData_.sBlockNum);
    OP_LOGD(context_->GetNodeName(), "ssmStateStride: [%u]", tilingData_.ssmStateStride);
    OP_LOGD(context_->GetNodeName(), "b: [%u]", tilingData_.b);
    OP_LOGD(context_->GetNodeName(), "vStep: [%u]", tilingData_.vStep);
    OP_LOGD(context_->GetNodeName(), "stateOutBufferNum: [%u]", tilingData_.stateOutBufferNum);
    OP_LOGD(context_->GetNodeName(), "attnOutBufferNum: [%u]", tilingData_.attnOutBufferNum);
    OP_LOGD(context_->GetNodeName(), "scale: [%f]", tilingData_.scale);
    OP_LOGD(context_->GetNodeName(), "lowerBound: [%f]", tilingData_.lowerBound);
    OP_LOGD(context_->GetNodeName(), "layout: [%u]", tilingData_.layout);
    OP_LOGD(context_->GetNodeName(), "hasCuSeqlens: [%u]", tilingData_.hasCuSeqlens);
    OP_LOGD(context_->GetNodeName(), "hasSsmStateIndices: [%u]", tilingData_.hasSsmStateIndices);
    OP_LOGD(context_->GetNodeName(), "hasALog: [%u]", tilingData_.hasALog);
    OP_LOGD(context_->GetNodeName(), "hasDtBias: [%u]", tilingData_.hasDtBias);
    OP_LOGD(context_->GetNodeName(), "hasAcceptedTokens: [%u]", tilingData_.hasAcceptedTokens);
    OP_LOGD(context_->GetNodeName(), "useQkL2norm: [%u]", tilingData_.useQkL2norm);
    OP_LOGD(context_->GetNodeName(), "useGateInKernel: [%u]", tilingData_.useGateInKernel);
    OP_LOGD(context_->GetNodeName(), "useBetaSigmoid: [%u]", tilingData_.useBetaSigmoid);
    OP_LOGD(context_->GetNodeName(), "allowNegEigval: [%u]", tilingData_.allowNegEigval);
    OP_LOGD(context_->GetNodeName(), "safeGate: [%u]", tilingData_.safeGate);
    OP_LOGD(context_->GetNodeName(), "stateVFirst: [%u]", tilingData_.stateVFirst);
    OP_LOGD(context_->GetNodeName(), "outputFinalState: [%u]", tilingData_.outputFinalState);
    OP_LOGD(context_->GetNodeName(), "inplaceFinalState: [%u]", tilingData_.inplaceFinalState);
    OP_LOGD(context_->GetNodeName(), "gateDtype: [%u]", tilingData_.gateDtype);
    OP_LOGD(context_->GetNodeName(), "betaDtype: [%u]", tilingData_.betaDtype);
    OP_LOGD(context_->GetNodeName(), "cuSeqlensDtype: [%u]", tilingData_.cuSeqlensDtype);
    OP_LOGD(context_->GetNodeName(), "ssmStateIndicesDtype: [%u]", tilingData_.ssmStateIndicesDtype);
    OP_LOGD(context_->GetNodeName(), "acceptedTokensDtype: [%u]", tilingData_.acceptedTokensDtype);
    OP_LOGD(context_->GetNodeName(), "queryTokenStride: [%llu]",
            static_cast<unsigned long long>(tilingData_.queryTokenStride));
    OP_LOGD(context_->GetNodeName(), "queryHeadStride: [%llu]",
            static_cast<unsigned long long>(tilingData_.queryHeadStride));
    OP_LOGD(context_->GetNodeName(), "keyTokenStride: [%llu]",
            static_cast<unsigned long long>(tilingData_.keyTokenStride));
    OP_LOGD(context_->GetNodeName(), "keyHeadStride: [%llu]",
            static_cast<unsigned long long>(tilingData_.keyHeadStride));
    OP_LOGD(context_->GetNodeName(), "valueTokenStride: [%llu]",
            static_cast<unsigned long long>(tilingData_.valueTokenStride));
    OP_LOGD(context_->GetNodeName(), "valueHeadStride: [%llu]",
            static_cast<unsigned long long>(tilingData_.valueHeadStride));
}

ge::graphStatus RecurrentKdaTiling::CalUbSize()
{
    RecurrentKdaTilingProcessor processor(BuildProcessorContext());
    return processor.ProcessUb(tilingData_);
}

static ge::graphStatus RecurrentKdaTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("RecurrentKda", "tiling_context");
        return ge::GRAPH_FAILED;
    }
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForRecurrentKda(gert::TilingParseContext *context)
{
    if (context == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("RecurrentKda", "tiling_parse_context");
        return ge::GRAPH_FAILED;
    }

    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "platform_info");
        return ge::GRAPH_FAILED;
    }

    auto compileInfoPtr = context->GetCompiledInfo<RecurrentKdaCompileInfo>();
    if (compileInfoPtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "compile_info");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RecurrentKda)
    .Tiling(RecurrentKdaTilingFunc)
    .TilingParse<RecurrentKdaCompileInfo>(TilingPrepareForRecurrentKda);
} // namespace optiling
