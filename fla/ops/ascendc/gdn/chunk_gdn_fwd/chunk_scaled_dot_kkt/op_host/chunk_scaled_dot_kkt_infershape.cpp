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
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
        context->GetInputShape(2) == nullptr || context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *kShape = context->GetInputShape(0);
    const gert::Shape *gShape = context->GetInputShape(1);
    const gert::Shape *betaShape = context->GetInputShape(2);
    if (kShape->GetDimNum() != 4 || gShape->GetDimNum() != 3 || betaShape->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    const int64_t b = kShape->GetDim(0);
    const int64_t hk = kShape->GetDim(1);
    const int64_t t = kShape->GetDim(2);
    const int64_t hv = gShape->GetDim(1);
    if (b <= 0 || hk <= 0 || hv <= 0 || t <= 0 || gShape->GetDim(0) != b || gShape->GetDim(2) != t ||
        betaShape->GetDim(0) != b || betaShape->GetDim(1) != hv || betaShape->GetDim(2) != t ||
        (hv % hk) != 0) {
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
    outShape->SetDim(0, b);
    outShape->SetDim(1, hv);
    outShape->SetDim(2, t);
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
