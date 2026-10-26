#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeChunkGatedDeltaRuleFwd32(gert::InferShapeContext *context)
{
    const gert::Shape *qShape = context->GetInputShape(0);
    const gert::Shape *vShape = context->GetInputShape(2);
    gert::Shape *oShape = context->GetOutputShape(0);
    gert::Shape *stateShape = context->GetOutputShape(1);
    if (qShape == nullptr || vShape == nullptr || oShape == nullptr || stateShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (qShape->GetDimNum() != 4 || vShape->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    const int64_t b = qShape->GetDim(0);
    const int64_t t = qShape->GetDim(2);
    const int64_t k = qShape->GetDim(3);
    const int64_t hv = vShape->GetDim(1);
    const int64_t v = vShape->GetDim(3);
    int64_t seqNum = b;
    const gert::Shape *cuShape = context->GetInputShape(6);
    if (cuShape != nullptr && cuShape->GetDimNum() == 1 && cuShape->GetDim(0) > 1) {
        seqNum = cuShape->GetDim(0) - 1;
    }
    oShape->SetDimNum(4);
    oShape->SetDim(0, b);
    oShape->SetDim(1, hv);
    oShape->SetDim(2, t);
    oShape->SetDim(3, v);
    stateShape->SetDimNum(4);
    stateShape->SetDim(0, seqNum);
    stateShape->SetDim(1, hv);
    stateShape->SetDim(2, k);
    stateShape->SetDim(3, v);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeChunkGatedDeltaRuleFwd32(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkGatedDeltaRuleFwd32)
    .InferShape(InferShapeChunkGatedDeltaRuleFwd32)
    .InferDataType(InferDataTypeChunkGatedDeltaRuleFwd32);
}  // namespace ops
