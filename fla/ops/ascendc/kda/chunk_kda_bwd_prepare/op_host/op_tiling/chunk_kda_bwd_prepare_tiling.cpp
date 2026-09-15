/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "chunk_kda_bwd_prepare_tiling.h"

#include <cmath>
#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"

namespace optiling {

ge::graphStatus Tiling4ChunkKdaBwdPrepare(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<KDA::ChunkKdaBwdPrepareTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto *aqkDesc = context->GetInputDesc(KDA_PREPARE_INPUT_AQK);
    auto *vNewDesc = context->GetInputDesc(KDA_PREPARE_INPUT_V_NEW);
    auto *dODesc = context->GetInputDesc(KDA_PREPARE_INPUT_D_O);
    auto *hDesc = context->GetInputDesc(KDA_PREPARE_INPUT_H);
    OP_CHECK_NULL_WITH_CONTEXT(context, aqkDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, vNewDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, dODesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, hDesc);

    const double *scale = attrs->GetAttrPointer<double>(KDA_PREPARE_ATTR_SCALE);
    const int64_t *chunkSize = attrs->GetAttrPointer<int64_t>(KDA_PREPARE_ATTR_CHUNK_SIZE);
    const bool *stateVFirst = attrs->GetAttrPointer<bool>(KDA_PREPARE_ATTR_STATE_V_FIRST);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSize);
    if (!std::isfinite(*scale)) {
        OP_LOGE(context->GetNodeName(), "scale must be finite");
        return ge::GRAPH_FAILED;
    }

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ChunkKdaBwdPrepareTilingContext ctx{
        context->GetNodeName(),
        context->GetRequiredInputShape(KDA_PREPARE_INPUT_AQK),
        context->GetRequiredInputShape(KDA_PREPARE_INPUT_V_NEW),
        context->GetRequiredInputShape(KDA_PREPARE_INPUT_D_O),
        context->GetRequiredInputShape(KDA_PREPARE_INPUT_H),
        context->GetOptionalInputShape(KDA_PREPARE_INPUT_CU_SEQLENS),
        context->GetOptionalInputShape(KDA_PREPARE_INPUT_CHUNK_INDICES),
        aqkDesc->GetDataType(), vNewDesc->GetDataType(), dODesc->GetDataType(), hDesc->GetDataType(),
        *scale, *chunkSize, stateVFirst != nullptr && *stateVFirst,
        static_cast<uint32_t>(platform.GetCoreNumAic()),
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize()),
    };
    ChunkKdaBwdPrepareTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    context->SetTilingKey(processor.GetTilingKey());
    context->SetBlockDim(processor.GetBlockDim());
    context->GetWorkspaceSizes(1)[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkKdaBwdPrepare(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwdPrepare)
    .Tiling(Tiling4ChunkKdaBwdPrepare)
    .TilingParse<ChunkKdaBwdPrepareCompileInfo>(TilingPrepareForChunkKdaBwdPrepare);

} // namespace optiling
