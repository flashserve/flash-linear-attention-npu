/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr int64_t kK = 128;
constexpr int64_t kV = 128;
constexpr size_t AG_HM_INDEX = 0;
constexpr size_t H_INDEX = 0;

static ge::graphStatus InferShapeMergeFwdBwd(gert::InferShapeContext *context)
{
    const gert::Shape *agShape = context->GetInputShape(AG_HM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, agShape);
    gert::Shape *hShape = context->GetOutputShape(H_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, hShape);
    if (agShape->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }
    const int64_t hv = agShape->GetDim(1);
    hShape->SetDimNum(3);
    hShape->SetDim(0, hv);
    hShape->SetDim(1, kK);
    hShape->SetDim(2, kV);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMergeFwdBwd(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(H_INDEX, context->GetInputDataType(AG_HM_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MergeFwdBwd)
    .InferShape(InferShapeMergeFwdBwd)
    .InferDataType(InferDataTypeMergeFwdBwd);
} // namespace ops
