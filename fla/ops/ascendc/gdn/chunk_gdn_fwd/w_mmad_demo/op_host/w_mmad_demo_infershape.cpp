/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShapeWMmadDemo(gert::InferShapeContext *context)
{
    const gert::Shape *a = context->GetInputShape(0);
    gert::Shape *c = context->GetOutputShape(0);
    if (a == nullptr || c == nullptr) {
        return ge::GRAPH_FAILED;
    }
    c->SetDimNum(a->GetDimNum());
    for (size_t i = 0; i < a->GetDimNum(); ++i) {
        c->SetDim(i, a->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeWMmadDemo(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(WMmadDemo)
    .InferShape(InferShapeWMmadDemo)
    .InferDataType(InferDataTypeWMmadDemo);

} // namespace ops
