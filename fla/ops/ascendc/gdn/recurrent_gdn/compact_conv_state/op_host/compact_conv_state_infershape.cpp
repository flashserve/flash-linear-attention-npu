/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "err/ops_err.h"

using namespace gert;

namespace ops {

static ge::graphStatus InferShapeCompactConvState(InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *state = context->GetInputShape(0);
    auto *out = context->GetOutputShape(0);
    if (state == nullptr || out == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *out = *state;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeCompactConvState(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CompactConvState)
    .InferShape(InferShapeCompactConvState)
    .InferDataType(InferDataTypeCompactConvState);

} // namespace ops
