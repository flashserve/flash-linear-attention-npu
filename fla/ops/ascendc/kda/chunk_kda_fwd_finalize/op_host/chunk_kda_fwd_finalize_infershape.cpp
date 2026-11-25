#include <cstring>

#include "chunk_kda_fwd_finalize_tiling.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {

bool ResolveLayout(const char *layout, bool &packed, bool &sequenceMajor)
{
    if (layout == nullptr) {
        return false;
    }
    packed = std::strcmp(layout, "TND") == 0 ||
             std::strcmp(layout, "NTD") == 0;
    sequenceMajor = std::strcmp(layout, "BSND") == 0 ||
                    std::strcmp(layout, "TND") == 0;
    return packed || sequenceMajor || std::strcmp(layout, "BNSD") == 0;
}

} // namespace

static ge::graphStatus InferShapeChunkKdaFwdFinalize(
    gert::InferShapeContext *context)
{
    const auto *qShape = context->GetInputShape(
        optiling::FINALIZE_INPUT_QG_SCALED);
    const auto *vShape = context->GetInputShape(optiling::FINALIZE_INPUT_V_NEW);
    auto *out = context->GetOutputShape(0);
    if (qShape == nullptr || vShape == nullptr || vShape->GetDimNum() == 0 || out == nullptr || context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char *layout = context->GetAttrs()->GetStr(
        optiling::FINALIZE_ATTR_OUTPUT_LAYOUT);
    bool packed = false;
    bool sequenceMajor = false;
    if (!ResolveLayout(layout, packed, sequenceMajor)) {
        return ge::GRAPH_FAILED;
    }
    if (packed) {
        if (qShape->GetDimNum() != 3) {
            return ge::GRAPH_FAILED;
        }
        out->SetDimNum(3);
        out->SetDim(0, sequenceMajor ? qShape->GetDim(1) : qShape->GetDim(0));
        out->SetDim(1, sequenceMajor ? qShape->GetDim(0) : qShape->GetDim(1));
        out->SetDim(2, vShape->GetDim(vShape->GetDimNum()-1));
    } else {
        if (qShape->GetDimNum() != 4) {
            return ge::GRAPH_FAILED;
        }
        out->SetDimNum(4);
        out->SetDim(0, qShape->GetDim(0));
        out->SetDim(1, sequenceMajor ? qShape->GetDim(2) : qShape->GetDim(1));
        out->SetDim(2, sequenceMajor ? qShape->GetDim(1) : qShape->GetDim(2));
        out->SetDim(3, vShape->GetDim(vShape->GetDimNum()-1));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeChunkKdaFwdFinalize(
    gert::InferDataTypeContext *context)
{
    const auto *fp16 = context->GetAttrs()->GetAttrPointer<bool>(optiling::FINALIZE_ATTR_OUTPUT_FP16);
    context->SetOutputDataType(0, fp16 != nullptr && *fp16 ? ge::DT_FLOAT16 : ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkKdaFwdFinalize)
    .InferShape(InferShapeChunkKdaFwdFinalize)
    .InferDataType(InferDataTypeChunkKdaFwdFinalize);

} // namespace ops
