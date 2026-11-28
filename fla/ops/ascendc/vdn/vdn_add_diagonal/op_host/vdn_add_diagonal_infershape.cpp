// SPDX-License-Identifier: BSD-3-Clause
#include "register/op_impl_registry.h"

namespace ops {
namespace {
ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    auto *correction = context->GetOutputShape(0);
    const auto *correctionIn = context->GetInputShape(0);
    if (correction == nullptr || correctionIn == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *correction = *correctionIn;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDtype(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(VdnAddDiagonal).InferShape(InferShape).InferDataType(InferDtype);
} // namespace ops
