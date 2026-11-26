/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "err/ops_err.h"

using namespace gert;

namespace ops {

static ge::graphStatus InferShapeGdnRecoverssmCommit(InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("GdnRecoverssmCommit", "InferShapeContext");
        return ge::GRAPH_FAILED;
    }
    auto state = context->GetInputShape(0);
    auto out = context->GetOutputShape(0);
    if (state == nullptr || out == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "state/output",
                                                 "shape must not be nullptr");
        return ge::GRAPH_FAILED;
    }
    out->SetDimNum(state->GetDimNum());
    for (size_t dim = 0; dim < state->GetDimNum(); ++dim) {
        out->SetDim(dim, state->GetDim(dim));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeGdnRecoverssmCommit(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GdnRecoverssmCommit)
    .InferShape(InferShapeGdnRecoverssmCommit)
    .InferDataType(InferDataTypeGdnRecoverssmCommit);

} // namespace ops
