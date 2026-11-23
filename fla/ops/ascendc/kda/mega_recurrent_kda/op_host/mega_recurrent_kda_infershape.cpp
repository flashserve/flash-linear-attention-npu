#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace {
constexpr size_t kSources[]={0,9,11,12,13};

ge::graphStatus InferShape(gert::InferShapeContext *ctx) {
    if(!ctx) return ge::GRAPH_FAILED;
    for(size_t i=0;i<5;++i) {
        auto src=ctx->GetInputShape(kSources[i]);
        auto dst=ctx->GetOutputShape(i);
        if(!src || !dst) return ge::GRAPH_FAILED;
        *dst=*src;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferType(gert::InferDataTypeContext *ctx) {
    if(!ctx) return ge::GRAPH_FAILED;
    for(size_t i=0;i<5;++i) ctx->SetOutputDataType(i,ctx->GetInputDataType(kSources[i]));
    return ge::GRAPH_SUCCESS;
}
}
IMPL_OP_INFERSHAPE(MegaRecurrentKda).InferShape(InferShape).InferDataType(InferType);
