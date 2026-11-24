#include "register/op_impl_registry.h"
#include "chunk_kda_bwd_prepare_validation.h"
namespace ops {
static ge::graphStatus InferShapeChunkKdaBwdPrepare(gert::InferShapeContext *context)
{
    auto *attrs = context->GetAttrs();
    auto *input = context->GetInputShape(0);
    uint32_t layout;
    KDA::TokenShape shape;
    if (!attrs || !input || !KDA::ParseLayout(attrs->GetStr(3), layout) ||
        !KDA::ParseTokenShape(*input, layout, shape)) return ge::GRAPH_FAILED;
    const bool packed = KDA::Rank3Layout(layout);
    for (size_t i = 0; i < 3; ++i) {
        auto *out = context->GetOutputShape(i);
        if (!out) return ge::GRAPH_FAILED;
        out->SetDimNum(packed ? 3 : 4);
        const size_t base = packed ? 0 : 1;
        if (!packed) out->SetDim(0, shape.B);
        out->SetDim(base, shape.H); out->SetDim(base + 1, shape.T);
        out->SetDim(base + 2, i == 0 ? 64 : 128);
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeChunkKdaBwdPrepare(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    // The optimized prepare accepts BF16 inputs only.
    context->SetOutputDataType(1, ge::DT_BF16);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(ChunkKdaBwdPrepare)
    .InferShape(InferShapeChunkKdaBwdPrepare).InferDataType(InferDataTypeChunkKdaBwdPrepare);
}
