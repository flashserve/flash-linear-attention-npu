#include "chunk_kda_bwd_intra_tiling.h"
#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"

namespace optiling {

ge::graphStatus Tiling4ChunkKdaBwdIntra(gert::TilingContext *context)
{
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto *tiling = context->GetTilingData<KDA::ChunkKdaBwdIntraTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto *qDesc = context->GetInputDesc(KDA_BWD_Q_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qDesc);
    const auto *betaDesc = context->GetInputDesc(KDA_BWD_BETA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDesc);
    const auto *chunkSize =
        attrs->GetAttrPointer<int64_t>(KDA_BWD_CHUNK_SIZE_ATTR_IDX);
    const auto *safeGate =
        attrs->GetAttrPointer<bool>(KDA_BWD_SAFE_GATE_ATTR_IDX);
    const auto *layoutMode =
        attrs->GetAttrPointer<int64_t>(KDA_BWD_LAYOUT_MODE_ATTR_IDX);
    const auto *totalChunks =
        attrs->GetAttrPointer<int64_t>(KDA_BWD_TOTAL_CHUNKS_ATTR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, safeGate);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutMode);
    OP_CHECK_NULL_WITH_CONTEXT(context, totalChunks);
    const auto *cuSeqlensTensor =
        context->GetOptionalInputTensor(KDA_BWD_CU_SEQLENS_IDX);
    const auto *chunkMetadataTensor =
        context->GetOptionalInputTensor(KDA_BWD_CHUNK_METADATA_IDX);

    ChunkKdaBwdIntraTilingContext ctx{
        context->GetNodeName(),
        context->GetRequiredInputShape(KDA_BWD_Q_IDX),
        context->GetRequiredInputShape(KDA_BWD_K_IDX),
        context->GetRequiredInputShape(KDA_BWD_GK_IDX),
        context->GetRequiredInputShape(KDA_BWD_BETA_IDX),
        context->GetRequiredInputShape(KDA_BWD_DAQK_IDX),
        context->GetRequiredInputShape(KDA_BWD_DAKK_IDX),
        context->GetRequiredInputShape(KDA_BWD_DQ_IDX),
        context->GetRequiredInputShape(KDA_BWD_DK_IDX),
        context->GetRequiredInputShape(KDA_BWD_DB_IDX),
        context->GetRequiredInputShape(KDA_BWD_DG_IDX),
        qDesc->GetDataType(),
        betaDesc->GetDataType(),
        *chunkSize,
        *safeGate,
        *layoutMode,
        *totalChunks,
        cuSeqlensTensor != nullptr,
        chunkMetadataTensor != nullptr,
        cuSeqlensTensor == nullptr ? 0 :
            static_cast<int64_t>(
                cuSeqlensTensor->GetStorageShape().GetShapeSize()),
        chunkMetadataTensor == nullptr ? 0 :
            static_cast<int64_t>(
                chunkMetadataTensor->GetStorageShape().GetShapeSize()),
        static_cast<uint32_t>(platform.GetCoreNumAic()),
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize()),
    };
    ChunkKdaBwdIntraTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    context->SetTilingKey(processor.GetTilingKey());
    context->SetBlockDim(processor.GetBlockDim());
    context->GetWorkspaceSizes(1)[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaBwdIntra(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwdIntra)
    .Tiling(Tiling4ChunkKdaBwdIntra)
    .TilingParse<ChunkKdaBwdIntraCompileInfo>(TilingPrepare4ChunkKdaBwdIntra);

} // namespace optiling
