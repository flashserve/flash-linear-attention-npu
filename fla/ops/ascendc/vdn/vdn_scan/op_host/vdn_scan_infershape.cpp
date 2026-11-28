// SPDX-License-Identifier: BSD-3-Clause
#include "register/op_impl_registry.h"

namespace ops {
namespace {
ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    auto *prefix = context->GetOutputShape(0);
    const auto *prefixIn = context->GetInputShape(1);
    if (prefix == nullptr || prefixIn == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *prefix = *prefixIn;
    auto *suffix = context->GetOutputShape(1);
    const auto *suffixIn = context->GetInputShape(2);
    if (suffix == nullptr || suffixIn == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *suffix = *suffixIn;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDtype(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(VdnScan).InferShape(InferShape).InferDataType(InferDtype);
} // namespace ops
