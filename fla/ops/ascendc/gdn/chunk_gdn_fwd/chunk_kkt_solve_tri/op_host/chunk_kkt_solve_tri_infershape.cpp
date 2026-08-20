#include "register/op_impl_registry.h"

namespace ops {
namespace {
ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetOutputShape(0) == nullptr ||
        context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *k = context->GetInputShape(0);
    const int64_t *chunkSize = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (k->GetDimNum() != 4 || chunkSize == nullptr || (*chunkSize != 64 && *chunkSize != 128)) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape *out = context->GetOutputShape(0);
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
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(ChunkKktSolveTri).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ops
