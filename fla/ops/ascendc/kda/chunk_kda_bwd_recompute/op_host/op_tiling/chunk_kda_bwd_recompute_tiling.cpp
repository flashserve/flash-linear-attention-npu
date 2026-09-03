/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "chunk_kda_bwd_recompute_tiling.h"
#include "../chunk_kda_bwd_recompute_tiling_processor.h"
#include <algorithm>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

ge::graphStatus Tiling4ChunkKdaBwdRecompute(gert::TilingContext *context)
{
    ChunkKdaBwdRecomputeTilingData *tiling = context->GetTilingData<ChunkKdaBwdRecomputeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const auto *gDesc = context->GetInputDesc(KDA_RECOMPUTE_G_IDX);
    const auto *betaDesc = context->GetInputDesc(KDA_RECOMPUTE_BETA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDesc);

    auto cuSeqlensTensor = context->GetOptionalInputTensor(KDA_RECOMPUTE_CU_SEQLENS_IDX);
    auto chunkIndicesTensor = context->GetOptionalInputTensor(KDA_RECOMPUTE_CHUNK_INDICES_IDX);
    const int64_t *cuSeqlensData = cuSeqlensTensor != nullptr ? cuSeqlensTensor->GetData<int64_t>() : nullptr;
    const int64_t *chunkIndicesData =
        chunkIndicesTensor != nullptr ? chunkIndicesTensor->GetData<int64_t>() : nullptr;

    ChunkKdaBwdRecomputeTilingContext ctx{
        context->GetNodeName(),
        context->GetRequiredInputShape(KDA_RECOMPUTE_Q_IDX),
        context->GetRequiredInputShape(KDA_RECOMPUTE_K_IDX),
        context->GetRequiredInputShape(KDA_RECOMPUTE_V_IDX),
        context->GetRequiredInputShape(KDA_RECOMPUTE_G_IDX),
        context->GetRequiredInputShape(KDA_RECOMPUTE_BETA_IDX),
        context->GetRequiredInputShape(KDA_RECOMPUTE_A_IDX),
        context->GetOptionalInputShape(KDA_RECOMPUTE_A_LOG_IDX),
        context->GetOptionalInputShape(KDA_RECOMPUTE_DT_BIAS_IDX),
        context->GetOptionalInputShape(KDA_RECOMPUTE_CU_SEQLENS_IDX),
        context->GetOptionalInputShape(KDA_RECOMPUTE_CHUNK_INDICES_IDX),
        cuSeqlensData,
        chunkIndicesData,
        gDesc->GetDataType(),
        betaDesc->GetDataType(),
        *(attrPtr->GetAttrPointer<int64_t>(KDA_RECOMPUTE_CHUNK_SIZE_ATTR)),
        *(attrPtr->GetAttrPointer<bool>(KDA_RECOMPUTE_USE_GATE_ATTR)),
        *(attrPtr->GetAttrPointer<bool>(KDA_RECOMPUTE_USE_EXP2_ATTR)),
        *(attrPtr->GetAttrPointer<float>(KDA_RECOMPUTE_LOWER_BOUND_ATTR)),
        ubSize,
        ascendcPlatform.GetLibApiWorkSpaceSize(),
    };

    ChunkKdaBwdRecomputeTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    context->SetTilingKey(processor.GetTilingKey());
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t usedAic = std::max(
        1u, std::min(aicNum, static_cast<uint32_t>(std::max<int64_t>(tiling->chunkNum, 1))));
    context->SetBlockDim(usedAic);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);
    auto *rawTiling = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTiling);
    rawTiling->SetDataSize(sizeof(ChunkKdaBwdRecomputeTilingData));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingRecomputeForChunkKdaBwdRecompute(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwdRecompute)
    .Tiling(Tiling4ChunkKdaBwdRecompute)
    .TilingParse<ChunkKdaBwdRecomputeCompileInfo>(TilingRecomputeForChunkKdaBwdRecompute);

} // namespace optiling
