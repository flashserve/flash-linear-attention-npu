#include "register/op_impl_registry.h"

namespace {
bool IsChunkSizeSupported(int64_t chunkSize)
{
    return chunkSize == 16 || chunkSize == 32 || chunkSize == 64 || chunkSize == 128;
}
}  // namespace

namespace ops {
static ge::graphStatus InferShapeChunkScaledDotKkt(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *kShape = context->GetInputShape(0);
    if (kShape->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    int64_t chunkSize = 64;
    if (context->GetAttrs() != nullptr && context->GetAttrs()->GetAttrPointer<int64_t>(0) != nullptr) {
        chunkSize = *context->GetAttrs()->GetAttrPointer<int64_t>(0);
    }
    if (!IsChunkSizeSupported(chunkSize)) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape *outShape = context->GetOutputShape(0);
    outShape->SetDimNum(4);
    outShape->SetDim(0, kShape->GetDim(0));
    outShape->SetDim(1, kShape->GetDim(1));
    outShape->SetDim(2, kShape->GetDim(2));
    outShape->SetDim(3, chunkSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeChunkScaledDotKkt(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkScaledDotKkt)
    .InferShape(InferShapeChunkScaledDotKkt)
    .InferDataType(InferDataTypeChunkScaledDotKkt);
}  // namespace ops
