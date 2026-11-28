#include "kda_metadata.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace {
constexpr std::array kOutputSources{kda::Input::HiddenStates, kda::Input::ConvState, kda::Input::ReplayU,
                                    kda::Input::ReplayK, kda::Input::ReplayDecay};

ge::graphStatus InferShape(gert::InferShapeContext *ctx)
{
    if (!ctx) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < kOutputSources.size(); ++i) {
        auto src = ctx->GetInputShape(static_cast<size_t>(kOutputSources[i]));
        auto dst = ctx->GetOutputShape(i);
        if (!src || !dst) {
            return ge::GRAPH_FAILED;
        }
        *dst = *src;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferType(gert::InferDataTypeContext *ctx)
{
    if (!ctx) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < kOutputSources.size(); ++i) {
        ctx->SetOutputDataType(i, ctx->GetInputDataType(static_cast<size_t>(kOutputSources[i])));
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace
IMPL_OP_INFERSHAPE(MegaRecurrentKda).InferShape(InferShape).InferDataType(InferType);
