/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/* !
 * \file recurrent_gated_delta_rule_infershape.cpp
 * \brief
 */
#include <map>
#include <string>
#include <sstream>
#include <initializer_list>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "recurrent_gated_delta_rule_error.h"
#include "err/ops_err.h"

using namespace gert;
namespace ops {

const size_t VALUE_INDEX = 2;
const size_t STATE_INDEX = 4;
const size_t VALUE_DIM = 3;
const size_t STATE_DIM = 4;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

static ge::graphStatus InferShapeRecurrentGatedDeltaRule(InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("RecurrentGatedDeltaRule", "InferShapeContext");
        return ge::GRAPH_FAILED;
    }

    auto opName = context->GetNodeName();
    auto shapeValue = context->GetInputShape(VALUE_INDEX);
    auto shapeInitialState = context->GetInputShape(STATE_INDEX);
    auto shapeOut = context->GetOutputShape(DIM_0);
    auto shapeFinalState = context->GetOutputShape(DIM_1);
    const std::pair<const gert::Shape *,const char *> shapes[] = {
        {shapeValue,"value"},{shapeInitialState,"state"},{shapeOut,"out"},{shapeFinalState,"stateOut"}};
    for (const auto &shape : shapes) {
        if (!shape.first) {
            OP_LOGE_WITH_INVALID_INPUT(opName,shape.second);
            return ge::GRAPH_FAILED;
        }
    }
    for (const auto &input : {std::make_pair(shapeValue,VALUE_DIM),std::make_pair(shapeInitialState,STATE_DIM)}) {
        if (input.first->GetDimNum() != input.second) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName,input.first == shapeValue ? "value" : "state",
                std::to_string(input.first->GetDimNum()),std::to_string(input.second));
            return ge::GRAPH_FAILED;
        }
    }

    shapeOut->SetDimNum(VALUE_DIM);
    int64_t outDim0 = shapeValue->GetDim(DIM_0);
    int64_t outDim1 = shapeValue->GetDim(DIM_1);
    int64_t outDim2 = shapeValue->GetDim(DIM_2);
    shapeOut->SetDim(DIM_0, outDim0);
    shapeOut->SetDim(DIM_1, outDim1);
    shapeOut->SetDim(DIM_2, outDim2);

    shapeFinalState->SetDimNum(STATE_DIM);
    int64_t stateDim0 = shapeInitialState->GetDim(DIM_0);
    int64_t stateDim1 = shapeInitialState->GetDim(DIM_1);
    int64_t stateDim2 = shapeInitialState->GetDim(DIM_2);
    int64_t stateDim3 = shapeInitialState->GetDim(DIM_3);
    shapeFinalState->SetDim(DIM_0, stateDim0);
    shapeFinalState->SetDim(DIM_1, stateDim1);
    shapeFinalState->SetDim(DIM_2, stateDim2);
    shapeFinalState->SetDim(DIM_3, stateDim3);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeRecurrentGatedDeltaRule(gert::InferDataTypeContext *context)
{
    if (!context) {
        OP_LOGE_WITH_INVALID_INPUT("RecurrentGatedDeltaRule", "InferDataTypeContext");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, context->GetInputDataType(STATE_INDEX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RecurrentGatedDeltaRule)
    .InferShape(InferShapeRecurrentGatedDeltaRule)
    .InferDataType(InferDataTypeRecurrentGatedDeltaRule);
} // namespace ops
