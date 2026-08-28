#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeChunkCumsumKkt(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
        context->GetInputShape(2) == nullptr || context->GetOutputShape(0) == nullptr ||
        context->GetOutputShape(1) == nullptr || context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *k = context->GetInputShape(0);
    const gert::Shape *g = context->GetInputShape(1);
    const gert::Shape *beta = context->GetInputShape(2);
    const int64_t *chunkSize = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (k->GetDimNum() != 4 || g->GetDimNum() != 3 || beta->GetDimNum() != 3 || chunkSize == nullptr ||
        (*chunkSize != 64 && *chunkSize != 128) || k->GetDim(0) != g->GetDim(0) ||
        k->GetDim(1) != g->GetDim(1) || k->GetDim(2) != g->GetDim(2) ||
        beta->GetDim(0) != g->GetDim(0) || beta->GetDim(1) != g->GetDim(1) ||
        beta->GetDim(2) != g->GetDim(2)) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *g;
    gert::Shape *a = context->GetOutputShape(1);
    a->SetDimNum(4);
    a->SetDim(0, k->GetDim(0));
    a->SetDim(1, k->GetDim(1));
    a->SetDim(2, k->GetDim(2));
    a->SetDim(3, *chunkSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeChunkCumsumKkt(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkCumsumKkt)
    .InferShape(InferShapeChunkCumsumKkt)
    .InferDataType(InferDataTypeChunkCumsumKkt);
}  // namespace ops
