/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/* !
 * \file gdn_recoverssm_verify_infershape.cpp
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
#include "gdn_recoverssm_verify_error.h"
#include "err/ops_err.h"

using namespace gert;
namespace ops {

const size_t VALUE_INDEX = 6;
const size_t CHECKPOINT_INDEX = 7;
const size_t REPLAY_INDEX = 8;
const size_t VALUE_DIM = 4;
const size_t STATE_DIM = 4;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

static ge::graphStatus InferShapeGdnRecoverssmVerify(InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("GdnRecoverssmVerify", "InferShapeContext");
        return ge::GRAPH_FAILED;
    }

    auto opName = context->GetNodeName();
    auto shapeValue = context->GetInputShape(VALUE_INDEX);
    auto shapeInitialState = context->GetInputShape(CHECKPOINT_INDEX);
    auto shapeReplay = context->GetInputShape(REPLAY_INDEX);
    auto shapeOut = context->GetOutputShape(DIM_0);
    auto shapeFinalState = context->GetOutputShape(DIM_1);
    const std::pair<const gert::Shape *, const char *> shapes[] = {{shapeValue, "v"},
                                                                   {shapeInitialState, "checkpoint_state"},
                                                                   {shapeReplay, "replay_cache"},
                                                                   {shapeOut, "out"},
                                                                   {shapeFinalState, "replay_cache_out"}};
    for (const auto &shape : shapes) {
        if (!shape.first) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, shape.second, "input argument must not be nullptr");
            return ge::GRAPH_FAILED;
        }
    }
    for (const auto &input : {std::make_pair(shapeValue, VALUE_DIM), std::make_pair(shapeInitialState, STATE_DIM),
                              std::make_pair(shapeReplay, STATE_DIM)}) {
        if (input.first->GetDimNum() != input.second) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName,
                                         input.first == shapeValue ?
                                             "v" :
                                             (input.first == shapeInitialState ? "checkpoint_state" : "replay_cache"),
                                         std::to_string(input.first->GetDimNum()), std::to_string(input.second));
            return ge::GRAPH_FAILED;
        }
    }

    shapeOut->SetDimNum(VALUE_DIM);
    for (size_t dim = 0; dim < VALUE_DIM; ++dim) {
        shapeOut->SetDim(dim, shapeValue->GetDim(dim));
    }

    shapeFinalState->SetDimNum(STATE_DIM);
    for (size_t dim = 0; dim < STATE_DIM; ++dim) {
        shapeFinalState->SetDim(dim, shapeReplay->GetDim(dim));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeGdnRecoverssmVerify(gert::InferDataTypeContext *context)
{
    if (!context) {
        OP_LOGE_WITH_INVALID_INPUT("GdnRecoverssmVerify", "InferDataTypeContext");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GdnRecoverssmVerify)
    .InferShape(InferShapeGdnRecoverssmVerify)
    .InferDataType(InferDataTypeGdnRecoverssmVerify);
} // namespace ops
