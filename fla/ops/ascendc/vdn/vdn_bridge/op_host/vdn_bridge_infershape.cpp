// SPDX-License-Identifier: BSD-3-Clause
#include "register/op_impl_registry.h"

namespace ops {
namespace {
ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *prefix = context->GetInputShape(0);
    auto *out = context->GetOutputShape(0);
    const auto *attrs = context->GetAttrs();
    const auto *count = attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(1);
    if (prefix == nullptr || out == nullptr || count == nullptr || *count < 1) {
        return ge::GRAPH_FAILED;
    }
    // For dynamic-rank graphs the head count is also dynamic. The bridge always returns four axes.
    out->SetDimNum(4);
    out->SetDim(0, *count);
    out->SetDim(1, prefix->GetDimNum() == 4 ? prefix->GetDim(1) : -1);
    out->SetDim(2, 128);
    out->SetDim(3, 128);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto *attrs = context->GetAttrs();
    const auto *bf16 = attrs == nullptr ? nullptr : attrs->GetAttrPointer<bool>(2);
    if (bf16 == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, *bf16 ? ge::DT_BF16 : ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(VdnBridge).InferShape(InferShape).InferDataType(InferDataType);
} // namespace ops
