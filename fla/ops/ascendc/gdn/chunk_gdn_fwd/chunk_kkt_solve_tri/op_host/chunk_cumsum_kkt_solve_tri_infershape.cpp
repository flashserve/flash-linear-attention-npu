#include "register/op_impl_registry.h"

namespace ops {
namespace {
ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetInputShape(1) == nullptr || context->GetOutputShape(0) == nullptr ||
        context->GetOutputShape(1) == nullptr || context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *k = context->GetInputShape(0);
    const gert::Shape *g = context->GetInputShape(1);
    const int64_t *chunkSize = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (k->GetDimNum() != 4 || g->GetDimNum() != 3 || chunkSize == nullptr ||
        (*chunkSize != 64 && *chunkSize != 128) || k->GetDim(0) != g->GetDim(0) ||
        k->GetDim(1) != g->GetDim(1) || k->GetDim(2) != g->GetDim(2)) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *g;
    gert::Shape *out = context->GetOutputShape(1);
    out->SetDimNum(4);
    out->SetDim(0, k->GetDim(0));
    out->SetDim(1, k->GetDim(1));
    out->SetDim(2, k->GetDim(2));
    out->SetDim(3, *chunkSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(ChunkCumsumKktSolveTri).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ops
