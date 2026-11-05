/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_gated_delta_rule_tiling.cpp
 * \brief
 */
#include "recurrent_gated_delta_rule_tiling.h"

#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"
#include "recurrent_gated_delta_rule_error.h"
#include "tiling/platform/platform_ascendc.h"

#include <cmath>
#include <limits>

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(RecurrentGatedDeltaRule, RecurrentGatedDeltaRuleTiling, 0);

const size_t QUERY_INDEX = 0;
const size_t KEY_INDEX = 1;
const size_t VALUE_INDEX = 2;
const size_t BETA_INDEX = 3;
const size_t STATE_INDEX = 4;
const size_t CUSEQLENS_INDEX = 5;
const size_t SSM_STATE_INDICES_INDEX = 6;
const size_t G_INDEX = 7;
const size_t GK_INDEX = 8;
const size_t ACC_TO_INDEX = 9;
const size_t A_LOG_INDEX = 10;
const size_t DT_BIAS_INDEX = 11;

namespace {
bool HasOptionalTensor(gert::TilingContext *context, size_t index)
{
    const auto *shape = context->GetOptionalInputShape(index);
    return context->GetOptionalInputDesc(index) != nullptr && shape != nullptr &&
           shape->GetOriginShape().GetShapeSize() != 0;
}
} // namespace

void RecurrentGatedDeltaRuleTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "platform", rgdn_error::Format("platformInfoPtr is null"));
        return;
    }
    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo_.isA5 = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;

    if (compileInfo_.aivNum <= 0) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "platform.aivNum", "actual="+std::to_string(compileInfo_.aivNum)+"; expected >0");
        return;
    }
    tilingData_.vectorCoreNum = static_cast<uint32_t>(compileInfo_.aivNum);
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetShapeAttrsInfo()
{
    if (CheckContext() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; };

    if (GetVersion() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; };

    if (AnalyzeDtype() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; };

    if (AnalyzeShapes() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; };

    if (GetScale() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; };

    if (GetStateStrides() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; };

    if (GetOptionalInput() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; };

    if (AnalyzeFormat() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; };

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::DoOpTiling()
{
    if (CalUbSize() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; };

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::DoLibApiTiling()
{
    const auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
    const uint64_t stateDtypeKey = stateDtype == ge::DT_FLOAT ? RGDR_TPL_FP32 : RGDR_TPL_BF16;
    if (tilingData_.useV2) {
        tilingKey_ = GET_TPL_TILING_KEY(stateDtypeKey, tilingData_.betaDtype, tilingData_.gDtype);
    } else {
        // V1 keeps its original beta/g dtypes (BF16/FP32); the extra template
        // parameters are only present because the shared kernel entry now also
        // supports V2.
        tilingKey_ = GET_TPL_TILING_KEY(stateDtypeKey, RGDR_TPL_BF16, RGDR_TPL_FP32);
    }
    return ge::GRAPH_SUCCESS;
};

uint64_t RecurrentGatedDeltaRuleTiling::GetTilingKey() const
{
    return tilingKey_;
};

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetWorkspaceSize()
{
    workspaceSize_ = static_cast<int64_t>(RGDR_SYS_WORKSPACE_SIZE);
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus RecurrentGatedDeltaRuleTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.vectorCoreNum);
    auto tilingDataSize = sizeof(RecurrentGatedDeltaRuleTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "runtime", rgdn_error::Format("memcpy_s failed, ret=%d", ret));
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    if (workspaces == nullptr) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "workspace", rgdn_error::Format("workspaces is null")); return ge::GRAPH_FAILED; };
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::CheckContext()
{
    const std::pair<size_t,const char *> inputs[] = {
        {QUERY_INDEX,"query"},{KEY_INDEX,"key"},{VALUE_INDEX,"value"},{BETA_INDEX,"beta"},
        {STATE_INDEX,"state"},{CUSEQLENS_INDEX,"actual_seq_lengths"},{SSM_STATE_INDICES_INDEX,"ssm_state_indices"}};
    for (const auto &input : inputs) {
        if (!context_->GetInputShape(input.first) || !context_->GetInputDesc(input.first)) {
            OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(),input.second);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetVersion()
{
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "attributes", rgdn_error::Format("attrs is null")); return ge::GRAPH_FAILED; };
    const bool *useV2 = attrs->GetAttrPointer<bool>(5);
    // The V1 schema predates this hidden attribute.  Missing means the legacy
    // entry point and is intentionally equivalent to false.
    tilingData_.useV2 = useV2 != nullptr && *useV2;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeDtype()
{
    struct DtypeRule { size_t index; const char *name; ge::DataType first; ge::DataType second; const char *expected; };
    const DtypeRule required[] = {
        {QUERY_INDEX,"query",ge::DT_BF16,ge::DT_BF16,"[bfloat16]"},
        {KEY_INDEX,"key",ge::DT_BF16,ge::DT_BF16,"[bfloat16]"},
        {VALUE_INDEX,"value",ge::DT_BF16,ge::DT_BF16,"[bfloat16]"},
        {BETA_INDEX,"beta",ge::DT_BF16,tilingData_.useV2 ? ge::DT_FLOAT : ge::DT_BF16,
            tilingData_.useV2 ? "[bfloat16,float32]" : "[bfloat16]"},
        {STATE_INDEX,"state",ge::DT_BF16,ge::DT_FLOAT,"[bfloat16,float32]"},
        {CUSEQLENS_INDEX,"actual_seq_lengths",ge::DT_INT32,ge::DT_INT32,"[int32]"},
        {SSM_STATE_INDICES_INDEX,"ssm_state_indices",ge::DT_INT32,ge::DT_INT32,"[int32]"}};
    for (const auto &rule : required) {
        const auto dtype = context_->GetInputDesc(rule.index)->GetDataType();
        if (dtype != rule.first && dtype != rule.second) {
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(context_->GetNodeName(),rule.name,
                ge::TypeUtils::DataTypeToSerialString(dtype),rule.expected);
            return ge::GRAPH_FAILED;
        }
    }
    const auto beta = context_->GetInputDesc(BETA_INDEX)->GetDataType();
    tilingData_.betaDtype = beta == ge::DT_FLOAT ? RGDR_TPL_FP32 : RGDR_TPL_BF16;
    const DtypeRule optional[] = {
        {G_INDEX,"g",ge::DT_FLOAT,tilingData_.useV2 ? ge::DT_BF16 : ge::DT_FLOAT,
            tilingData_.useV2 ? "[bfloat16,float32]" : "[float32]"},
        {ACC_TO_INDEX,"num_accepted_tokens",ge::DT_INT32,ge::DT_INT32,"[int32]"},
        {A_LOG_INDEX,"A_log",ge::DT_FLOAT,ge::DT_FLOAT,"[float32]"},
        {DT_BIAS_INDEX,"dt_bias",ge::DT_FLOAT,ge::DT_FLOAT,"[float32]"}};
    for (const auto &rule : optional) {
        if (!HasOptionalTensor(context_,rule.index)) continue;
        const auto dtype = context_->GetOptionalInputDesc(rule.index)->GetDataType();
        if (dtype != rule.first && dtype != rule.second) {
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(context_->GetNodeName(),rule.name,
                ge::TypeUtils::DataTypeToSerialString(dtype),rule.expected);
            return ge::GRAPH_FAILED;
        }
        if (rule.index == G_INDEX) tilingData_.gDtype = dtype == ge::DT_FLOAT ? RGDR_TPL_FP32 : RGDR_TPL_BF16;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeShapes()
{
    const auto &query = context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
    const auto &key = context_->GetInputShape(KEY_INDEX)->GetOriginShape();
    const auto &value = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
    const auto &beta = context_->GetInputShape(BETA_INDEX)->GetOriginShape();
    const auto &state = context_->GetInputShape(STATE_INDEX)->GetOriginShape();
    const auto &lengths = context_->GetInputShape(CUSEQLENS_INDEX)->GetOriginShape();
    const auto &indices = context_->GetInputShape(SSM_STATE_INDICES_INDEX)->GetOriginShape();
    const std::pair<const gert::Shape *,std::pair<const char *,size_t>> ranks[] = {
        {&query,{"query",3}},{&key,{"key",3}},{&value,{"value",3}},{&beta,{"beta",2}},
        {&state,{"state",4}},{&lengths,{"actual_seq_lengths",1}}};
    for (const auto &input : ranks) {
        if (input.first->GetDimNum() != input.second.second) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(),input.second.first,
                std::to_string(input.first->GetDimNum()),std::to_string(input.second.second));
            return ge::GRAPH_FAILED;
        }
    }
    if (indices.GetDimNum() != 1 && indices.GetDimNum() != 2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(),"ssm_state_indices",
            std::to_string(indices.GetDimNum()),"1 or 2");
        return ge::GRAPH_FAILED;
    }
    if (query.GetDim(0) <= 0 || query.GetDim(1) <= 0 || value.GetDim(1) <= 0 ||
                state.GetDim(0) <= 0 || lengths.GetDim(0) < 2 ||
                query.GetDim(2) != RGDR_REQUIRED_DIM || value.GetDim(2) != RGDR_REQUIRED_DIM) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "input shapes", "T, heads and state slots must be positive; K and V must be 128; query="+rgdn_error::ShapeText(query)+", value="+rgdn_error::ShapeText(value)+", state="+rgdn_error::ShapeText(state)+", lengths="+rgdn_error::ShapeText(lengths)); return ge::GRAPH_FAILED; };
    if (key.GetDim(0) != query.GetDim(0) || key.GetDim(1) != query.GetDim(1) ||
                key.GetDim(2) != query.GetDim(2) || value.GetDim(0) != query.GetDim(0) ||
                beta.GetDim(0) != query.GetDim(0) || beta.GetDim(1) != value.GetDim(1) ||
                state.GetDim(1) != value.GetDim(1) || state.GetDim(2) != RGDR_REQUIRED_DIM ||
                state.GetDim(3) != RGDR_REQUIRED_DIM) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "input shapes", "inconsistent dimensions: query="+rgdn_error::ShapeText(query)+", key="+rgdn_error::ShapeText(key)+", value="+rgdn_error::ShapeText(value)+", beta="+rgdn_error::ShapeText(beta)+", state="+rgdn_error::ShapeText(state)); return ge::GRAPH_FAILED; };
    const uint64_t keyHeads = static_cast<uint64_t>(query.GetDim(1));
    const uint64_t valueHeads = static_cast<uint64_t>(value.GetDim(1));
    if (valueHeads % keyHeads != 0 || valueHeads / keyHeads > 16) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "query.HK,value.HV", "HK="+std::to_string(keyHeads)+", HV="+std::to_string(valueHeads)+"; HV/HK must be an integer in [1,16]"); return ge::GRAPH_FAILED; };

    tilingData_.totalTokens = static_cast<uint64_t>(query.GetDim(0));
    tilingData_.keyHeads = keyHeads;
    tilingData_.valueHeads = valueHeads;
    tilingData_.stateSlots = static_cast<uint64_t>(state.GetDim(0));
    tilingData_.batchSize = static_cast<uint64_t>(lengths.GetDim(0) - 1);
    tilingData_.checkpointWidth = indices.GetDimNum() == 2 ? static_cast<uint64_t>(indices.GetDim(1)) : 0;
    if ((indices.GetDimNum() == 1 && indices.GetDim(0) != query.GetDim(0)) ||
                (indices.GetDimNum() == 2 && (indices.GetDim(0) != lengths.GetDim(0) - 1 ||
                                            indices.GetDim(1) <= 0))) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "ssm_state_indices", "actual="+rgdn_error::ShapeText(indices)+"; expected [T="+std::to_string(tilingData_.totalTokens)+"] or [B="+std::to_string(tilingData_.batchSize)+",Ckpt>0]"); return ge::GRAPH_FAILED; };
    if (tilingData_.batchSize > std::numeric_limits<uint64_t>::max() / valueHeads ||
                compileInfo_.aivNum == 0 || compileInfo_.aivNum > std::numeric_limits<uint32_t>::max()) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "B,HV,aivNum", "B="+std::to_string(tilingData_.batchSize)+", HV="+std::to_string(valueHeads)+", aivNum="+std::to_string(compileInfo_.aivNum)+"; task count must fit uint64 and aivNum must be in [1,UINT32_MAX]"); return ge::GRAPH_FAILED; };
    tilingData_.taskCount = tilingData_.batchSize * valueHeads;
    const uint64_t tasksPerCore = (tilingData_.taskCount - 1) / compileInfo_.aivNum + 1;
    tilingData_.vectorCoreNum = static_cast<uint32_t>((tilingData_.taskCount - 1) / tasksPerCore + 1);

    // The prefix fields are used by the arch35 implementation.
    // The generic implementation uses the 64-bit dimensions and strides below.
    tilingData_.t = static_cast<uint32_t>(tilingData_.totalTokens);
    tilingData_.nk = static_cast<uint32_t>(keyHeads);
    tilingData_.nv = static_cast<uint32_t>(valueHeads);
    tilingData_.dk = RGDR_REQUIRED_DIM;
    tilingData_.dv = RGDR_REQUIRED_DIM;
    tilingData_.sBlockNum = static_cast<uint32_t>(tilingData_.stateSlots);
    tilingData_.b = static_cast<uint32_t>(tilingData_.batchSize);
    tilingData_.stateIndicesWidth = static_cast<uint32_t>(tilingData_.checkpointWidth);
    if (compileInfo_.isA5) {
        if (tilingData_.totalTokens > std::numeric_limits<uint32_t>::max() ||
                    tilingData_.stateSlots > std::numeric_limits<uint32_t>::max() ||
                    tilingData_.batchSize > std::numeric_limits<uint32_t>::max() ||
                    tilingData_.checkpointWidth > std::numeric_limits<uint32_t>::max()) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "input shapes", rgdn_error::Format("Ascend950 dimensions exceed uint32")); return ge::GRAPH_FAILED; };
    }
    return ge::GRAPH_SUCCESS;
}

bool RecurrentGatedDeltaRuleTiling::CheckFormat(ge::Format format, const std::string &Desc)
{
    if (format == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE_WITH_INVALID_INPUT_FORMAT(context_->GetNodeName(),Desc,ge::TypeUtils::FormatToSerialString(format),"any format except FRACTAL_NZ");
        return false;
    }
    return true;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeFormat()
{
    if (!CheckFormat(context_->GetInputDesc(QUERY_INDEX)->GetStorageFormat(), "query") ||
        !CheckFormat(context_->GetInputDesc(KEY_INDEX)->GetStorageFormat(), "key") ||
        !CheckFormat(context_->GetInputDesc(VALUE_INDEX)->GetStorageFormat(), "value") ||
        !CheckFormat(context_->GetInputDesc(STATE_INDEX)->GetStorageFormat(), "state") ||
        !CheckFormat(context_->GetInputDesc(CUSEQLENS_INDEX)->GetStorageFormat(), "actual_seq_lengths") ||
        !CheckFormat(context_->GetInputDesc(SSM_STATE_INDICES_INDEX)->GetStorageFormat(), "ssm_state_indices")) {
        return ge::GRAPH_FAILED;
    }

    if (context_->GetOptionalInputDesc(G_INDEX) != nullptr) {
        auto gamaFormat = context_->GetOptionalInputDesc(G_INDEX)->GetStorageFormat();
        if (gamaFormat == ge::FORMAT_FRACTAL_NZ) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "g", rgdn_error::Format("gama format not support NZ")); return ge::GRAPH_FAILED; };
    }
    if (context_->GetOptionalInputDesc(ACC_TO_INDEX) != nullptr) {
        auto numAcceptedTokensFormat = context_->GetOptionalInputDesc(ACC_TO_INDEX)->GetStorageFormat();
        if (numAcceptedTokensFormat == ge::FORMAT_FRACTAL_NZ) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "num_accepted_tokens", rgdn_error::Format("numAcceptedTokens format not support NZ")); return ge::GRAPH_FAILED; };
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetScale()
{
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "attributes", rgdn_error::Format("attrs is null")); return ge::GRAPH_FAILED; };
    const float *scalePtr = attrs->GetAttrPointer<float>(0);
    if (scalePtr == nullptr || !std::isfinite(*scalePtr)) { OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "scaleValue", scalePtr ? std::to_string(*scalePtr) : "null", "finite float32"); return ge::GRAPH_FAILED; };
    float scaleValue = *scalePtr;
    tilingData_.scale = scaleValue;
    const bool *norm = attrs->GetAttrPointer<bool>(1);
    const bool *gate = attrs->GetAttrPointer<bool>(2);
    const bool *sigmoid = attrs->GetAttrPointer<bool>(3);
    const bool *twice = attrs->GetAttrPointer<bool>(4);
    tilingData_.useQkL2NormInKernel = norm && *norm;
    tilingData_.useGateInKernel = gate && *gate;
    tilingData_.useBetaSigmoidInKernel = sigmoid && *sigmoid;
    tilingData_.allowNegEigval = twice && *twice;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetStateStrides()
{
    auto read = [&](size_t index, size_t rank, uint64_t *out) {
        const auto *stride = context_->GetInputStride(index);
        const bool explicitStrides = stride != nullptr && stride->GetDimNum() != 0;
        // Contiguous L0 intermediates may omit their stride metadata entirely.
        // Nonempty views must still carry the complete rank and positive strides.
        if (explicitStrides && stride->GetDimNum() != rank) {
            return false;
        }
        uint64_t dense = 1;
        uint64_t extent = 1;
        const auto &shape = context_->GetInputShape(index)->GetOriginShape();
        const uint64_t elementBytes = context_->GetInputDesc(index)->GetDataType() == ge::DT_FLOAT ? 4 : 2;
        const uint64_t maxExtent = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / elementBytes;
        for (size_t i = rank; i > 0; --i) {
            const int64_t value = explicitStrides ? stride->GetStride(i - 1) : static_cast<int64_t>(dense);
            if (value <= 0) {
                return false;
            }
            out[i - 1] = static_cast<uint64_t>(value);
            const uint64_t dim = static_cast<uint64_t>(shape.GetDim(i - 1));
            if (dim == 0 || (dim - 1) > (maxExtent - extent) / out[i - 1] ||
                dense > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / dim) {
                return false;
            }
            extent += (dim - 1) * out[i - 1];
            dense *= dim;
        }
        return true;
    };
    if (!read(QUERY_INDEX,3,tilingData_.queryStrides) || !read(KEY_INDEX,3,tilingData_.keyStrides) ||
                !read(VALUE_INDEX,3,tilingData_.valueStrides) || !read(STATE_INDEX,4,tilingData_.stateStrides)) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "input strides", rgdn_error::Format("strides must be positive and address arithmetic must fit int64")); return ge::GRAPH_FAILED; };
    for (const auto *strides : {tilingData_.queryStrides, tilingData_.keyStrides, tilingData_.valueStrides}) {
        if (strides[2] != 1 || strides[0] < RGDR_REQUIRED_DIM ||
                    strides[0] - RGDR_REQUIRED_DIM > std::numeric_limits<uint32_t>::max() / 2U) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "input strides", rgdn_error::Format("Q/K/V require a contiguous inner axis and a uint32 DMA token gap")); return ge::GRAPH_FAILED; };
    }
    const uint64_t stateBytes = context_->GetInputDesc(STATE_INDEX)->GetDataType() == ge::DT_FLOAT ? 4 : 2;
    if (tilingData_.stateStrides[3] != 1 || tilingData_.stateStrides[2] < RGDR_REQUIRED_DIM ||
                tilingData_.stateStrides[2] - RGDR_REQUIRED_DIM > std::numeric_limits<uint32_t>::max() / stateBytes) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "input strides", rgdn_error::Format("State requires a contiguous K axis and a uint32 DMA row gap")); return ge::GRAPH_FAILED; };
    tilingData_.stateStride0 = static_cast<uint32_t>(tilingData_.stateStrides[0]);
    tilingData_.stateStride1 = static_cast<uint32_t>(tilingData_.stateStrides[1]);
    tilingData_.stateStride2 = static_cast<uint32_t>(tilingData_.stateStrides[2]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetOptionalInput()
{
    tilingData_.hasGama = HasOptionalTensor(context_, G_INDEX);
    tilingData_.hasAcceptedTokens = HasOptionalTensor(context_, ACC_TO_INDEX);
    tilingData_.hasDtBias = HasOptionalTensor(context_, DT_BIAS_INDEX);
    tilingData_.hasALog = HasOptionalTensor(context_, A_LOG_INDEX);
    if (tilingData_.useGateInKernel && (!tilingData_.hasGama || !tilingData_.hasALog)) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "use_gate_in_kernel", "true requires g and A_log; g="+std::to_string(tilingData_.hasGama)+", A_log="+std::to_string(tilingData_.hasALog)); return ge::GRAPH_FAILED; };
    if (HasOptionalTensor(context_, GK_INDEX)) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "gk", rgdn_error::Format("gk is not supported")); return ge::GRAPH_FAILED; };
    for (size_t index : {G_INDEX, ACC_TO_INDEX, A_LOG_INDEX, DT_BIAS_INDEX}) {
        if (!HasOptionalTensor(context_, index)) {
            continue;
        }
        const auto &shape = context_->GetOptionalInputShape(index)->GetOriginShape();
        bool valid = false;
        if (index == G_INDEX) {
            valid = shape.GetDimNum() == 2 && shape.GetDim(0) == static_cast<int64_t>(tilingData_.totalTokens) &&
                    shape.GetDim(1) == static_cast<int64_t>(tilingData_.valueHeads);
        } else if (index == ACC_TO_INDEX) {
            valid = shape.GetDimNum() == 1 && shape.GetDim(0) == static_cast<int64_t>(tilingData_.batchSize);
        } else {
            valid = context_->GetOptionalInputDesc(index)->GetDataType() == ge::DT_FLOAT &&
                    shape.GetDimNum() == 1 && shape.GetDim(0) == static_cast<int64_t>(tilingData_.valueHeads);
        }
        if (!valid) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), index==G_INDEX ? "g" : index==ACC_TO_INDEX ? "num_accepted_tokens" : index==A_LOG_INDEX ? "A_log" : "dt_bias", "actual shape="+rgdn_error::ShapeText(shape)+"; expected "+(index==G_INDEX ? "[T,HV]" : index==ACC_TO_INDEX ? "[B]" : "[HV] FP32")); return ge::GRAPH_FAILED; };
    }
    if (!tilingData_.useV2 && (tilingData_.useQkL2NormInKernel || tilingData_.useGateInKernel ||
                                       tilingData_.useBetaSigmoidInKernel || tilingData_.allowNegEigval ||
                                       tilingData_.hasALog || tilingData_.hasDtBias ||
                                       tilingData_.betaDtype != RGDR_TPL_BF16 ||
                                       (tilingData_.hasGama && tilingData_.gDtype != RGDR_TPL_FP32))) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "input dtype", rgdn_error::Format("V1 does not accept V2 attributes or dtypes")); return ge::GRAPH_FAILED; };
    return ge::GRAPH_SUCCESS;
}

void RecurrentGatedDeltaRuleTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "vectorCoreNum: [%u]", tilingData_.vectorCoreNum);
    OP_LOGD(context_->GetNodeName(), "ubCalSize: [%u]", tilingData_.ubCalSize);
    OP_LOGD(context_->GetNodeName(), "ubRestBytes: [%u]", tilingData_.ubRestBytes);
    OP_LOGD(context_->GetNodeName(), "t: [%u]", tilingData_.t);
    OP_LOGD(context_->GetNodeName(), "nk: [%u]", tilingData_.nk);
    OP_LOGD(context_->GetNodeName(), "dk: [%u]", tilingData_.dk);
    OP_LOGD(context_->GetNodeName(), "nv: [%u]", tilingData_.nv);
    OP_LOGD(context_->GetNodeName(), "dv: [%u]", tilingData_.dv);
    OP_LOGD(context_->GetNodeName(), "sBlockNum: [%u]", tilingData_.sBlockNum);
    OP_LOGD(context_->GetNodeName(), "b: [%u]", tilingData_.b);
    OP_LOGD(context_->GetNodeName(), "vStep: [%u]", tilingData_.vStep);
    OP_LOGD(context_->GetNodeName(), "stateOutBufferNum: [%u]", tilingData_.stateOutBufferNum);
    OP_LOGD(context_->GetNodeName(), "attnOutBufferNum: [%u]", tilingData_.attnOutBufferNum);
    OP_LOGD(context_->GetNodeName(), "scale: [%f]", tilingData_.scale);
    OP_LOGD(context_->GetNodeName(), "hasGama: [%u]", tilingData_.hasGama);
    OP_LOGD(context_->GetNodeName(), "useV2: [%u]", tilingData_.useV2);
    OP_LOGD(context_->GetNodeName(), "hasAcceptedTokens: [%u]", tilingData_.hasAcceptedTokens);
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::CalUbSize()
{
    if (compileInfo_.isA5) {
        constexpr uint64_t tokenTile = 4;
        constexpr uint64_t stateBanks = 3;
        constexpr uint64_t inputBuffers = 2;
        constexpr uint64_t dim = RGDR_REQUIRED_DIM;
        constexpr uint64_t bf16Bytes = 2;
        constexpr uint64_t fp32Bytes = 4;
        constexpr uint64_t dataBlockBytes = 32;
        constexpr uint64_t ubGuardBytes = 4U * 1024U;
        const bool stateFp32 = context_->GetInputDesc(STATE_INDEX)->GetDataType() == ge::DT_FLOAT;
        const uint64_t pipelineBytes =
            stateBanks * dim * dim * fp32Bytes +
            (stateFp32 ? 0U : dim * dim * bf16Bytes) +
            inputBuffers * tokenTile * 3U * dim * bf16Bytes +
            inputBuffers * tokenTile * dataBlockBytes +
            inputBuffers * tokenTile * dataBlockBytes +
            inputBuffers * tokenTile * dim * bf16Bytes +
            dim * fp32Bytes + dim * dataBlockBytes + dim * sizeof(uint32_t);
        if (pipelineBytes + ubGuardBytes > compileInfo_.ubSize ||
            pipelineBytes > std::numeric_limits<uint32_t>::max()) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                context_->GetNodeName(), "UB capacity",
                rgdn_error::Format("UB cannot hold the Ascend950 V2 pipeline"));
            return ge::GRAPH_FAILED;
        }
        tilingData_.ubCalSize = static_cast<uint32_t>(pipelineBytes);
        tilingData_.ubRestBytes = 0;
        tilingData_.vStep = RGDR_REQUIRED_DIM;
        tilingData_.stateOutBufferNum = 1;
        tilingData_.attnOutBufferNum = 1;
        return ge::GRAPH_SUCCESS;
    }
    constexpr uint32_t fixedUbBytes = 162240;
    constexpr uint32_t ubGuardBytes = 2048;
    if (compileInfo_.ubSize < fixedUbBytes + ubGuardBytes) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "UB capacity", rgdn_error::Format("UB must hold the fixed 162240-byte layout and 2048-byte guard")); return ge::GRAPH_FAILED; };
    tilingData_.ubCalSize = fixedUbBytes;
    tilingData_.ubRestBytes = 0;
    tilingData_.vStep = RGDR_REQUIRED_DIM;
    tilingData_.stateOutBufferNum = 2;
    tilingData_.attnOutBufferNum = 2;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RecurrentGatedDeltaRuleTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("RecurrentGatedDeltaRule", "context", rgdn_error::Format("context is null")); return ge::GRAPH_FAILED; };
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForRecurrentGatedDeltaRule(gert::TilingParseContext *context)
{
    if (context == nullptr) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("RecurrentGatedDeltaRule", "context", rgdn_error::Format("context is null")); return ge::GRAPH_FAILED; };

    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform", rgdn_error::Format("platformInfoPtr is null")); return ge::GRAPH_FAILED; };

    auto compileInfoPtr = context->GetCompiledInfo<RecurrentGatedDeltaRuleCompileInfo>();
    if (compileInfoPtr == nullptr) { OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "runtime", rgdn_error::Format("compileInfoPtr is null")); return ge::GRAPH_FAILED; };

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RecurrentGatedDeltaRule)
    .Tiling(RecurrentGatedDeltaRuleTilingFunc)
    .TilingParse<RecurrentGatedDeltaRuleCompileInfo>(TilingPrepareForRecurrentGatedDeltaRule);
} // namespace optiling
