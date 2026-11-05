#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeRecurrentGatedDeltaRuleV2(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(2) == nullptr || context->GetInputShape(4) == nullptr ||
        context->GetOutputShape(0) == nullptr || context->GetOutputShape(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(2);
    *context->GetOutputShape(1) = *context->GetInputShape(4);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeRecurrentGatedDeltaRuleV2(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, context->GetInputDataType(4));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RecurrentGatedDeltaRuleV2)
    .InferShape(InferShapeRecurrentGatedDeltaRuleV2)
    .InferDataType(InferDataTypeRecurrentGatedDeltaRuleV2);
} // namespace ops
