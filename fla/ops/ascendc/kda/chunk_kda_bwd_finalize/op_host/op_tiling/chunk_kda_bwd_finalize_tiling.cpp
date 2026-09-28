#include "chunk_kda_bwd_finalize_tiling.h"

#include <cmath>
#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"

namespace optiling {

ge::graphStatus Tiling4ChunkKdaBwdFinalize(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<KDA::ChunkKdaBwdFinalizeTilingData>();
    auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const double *scale = attrs->GetAttrPointer<double>(ATTR_SCALE);
    const double *lowerBound = attrs->GetAttrPointer<double>(ATTR_LOWER_BOUND);
    const int64_t *chunkSize = attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE);
    const bool *safeGate = attrs->GetAttrPointer<bool>(ATTR_SAFE_GATE);
    const bool *useGate = attrs->GetAttrPointer<bool>(ATTR_USE_GATE_IN_KERNEL);
    const bool *useExp2 = attrs->GetAttrPointer<bool>(ATTR_USE_EXP2);
    const bool *stateVFirst = attrs->GetAttrPointer<bool>(ATTR_STATE_V_FIRST);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);
    OP_CHECK_NULL_WITH_CONTEXT(context, lowerBound);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, safeGate);
    OP_CHECK_NULL_WITH_CONTEXT(context, useGate);
    OP_CHECK_NULL_WITH_CONTEXT(context, useExp2);
    OP_CHECK_NULL_WITH_CONTEXT(context, stateVFirst);
    if (!std::isfinite(*scale) || !std::isfinite(*lowerBound)) {
        OP_LOGE(context->GetNodeName(), "scale/lower_bound must be finite");
        return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicCoreNum = static_cast<uint32_t>(platform.GetCoreNumAic());
    if (aicCoreNum == 0) {
        // The A5 aclnn direct-launch context may omit platform info.  This
        // checkpoint is A5-only and must still expose all 28 physical AICs.
        aicCoreNum = 28;
        OP_LOGW(context->GetNodeName(),
                "platform info did not report AIC count; use A5 fallback core count 28");
    }
    ChunkKdaBwdFinalizeTilingContext ctx{
        context->GetNodeName(),
        context->GetRequiredInputShape(INPUT_Q),
        context->GetRequiredInputShape(INPUT_K),
        context->GetRequiredInputShape(INPUT_V),
        context->GetRequiredInputShape(INPUT_GK),
        context->GetRequiredInputShape(INPUT_BETA),
        context->GetRequiredInputShape(INPUT_AKK),
        context->GetRequiredInputShape(INPUT_H),
        context->GetRequiredInputShape(INPUT_DH),
        context->GetRequiredInputShape(INPUT_DV_SCAN),
        context->GetOptionalInputShape(INPUT_Q_RSTD),
        context->GetOptionalInputShape(INPUT_K_RSTD),
        context->GetOptionalInputShape(INPUT_CU_SEQLENS),
        context->GetOptionalInputShape(INPUT_CHUNK_INDICES),
        *scale, *lowerBound, *chunkSize, *safeGate, *useGate, *useExp2, *stateVFirst,
        aicCoreNum,
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize())};
    ChunkKdaBwdFinalizeTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    context->SetTilingKey(processor.GetTilingKey());
    context->SetBlockDim(processor.GetBlockDim());
    context->GetWorkspaceSizes(1)[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkKdaBwdFinalize(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwdFinalize)
    .Tiling(Tiling4ChunkKdaBwdFinalize)
    .TilingParse<ChunkKdaBwdFinalizeCompileInfo>(TilingPrepareForChunkKdaBwdFinalize);

} // namespace optiling
