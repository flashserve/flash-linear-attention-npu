/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/* !
 * \file recurrent_kda_infershape.cpp
 * \brief
 */
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace gert;
namespace ops {

const size_t VALUE_INDEX = 2;
const size_t STATE_INDEX = 5;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;

static ge::graphStatus InferShapeRecurrentKda(InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("RecurrentKda", "inference context is null");
        return ge::GRAPH_FAILED;
    }

    auto opName = context->GetNodeName();
    auto shapeValue = context->GetInputShape(VALUE_INDEX);
    auto shapeInitialState = context->GetInputShape(STATE_INDEX);
    auto shapeOut = context->GetOutputShape(DIM_0);
    auto shapeFinalState = context->GetOutputShape(DIM_1);
    if (shapeValue == nullptr || shapeInitialState == nullptr || shapeOut == nullptr || shapeFinalState == nullptr) {
        OP_LOGE(opName, "[InferShape] shape is null");
        return ge::GRAPH_FAILED;
    }

    *shapeOut = *shapeValue;
    *shapeFinalState = *shapeInitialState;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeRecurrentKda(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, context->GetInputDataType(STATE_INDEX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RecurrentKda)
    .InferShape(InferShapeRecurrentKda)
    .InferDataType(InferDataTypeRecurrentKda);
} // namespace ops
