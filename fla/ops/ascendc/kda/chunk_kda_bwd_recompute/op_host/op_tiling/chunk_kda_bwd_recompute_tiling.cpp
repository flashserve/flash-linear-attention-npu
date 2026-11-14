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
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    constexpr uint32_t aivPerBlock = 2U;
    // Mix 1:2 platform info sometimes reports a single AIC/AIV pair instead of
    // the chip total. Fall back to AIV/2, then GetCoreNum, then A5's 28 AIC.
    if (isAscend950 && aicNum <= 2U) {
        const uint32_t coreNum = ascendcPlatform.GetCoreNum();
        if (aivNum >= 8U) {
            aicNum = aivNum / 2U;
        } else if (coreNum >= 8U) {
            aicNum = coreNum;
        } else {
            aicNum = 28U;
        }
    }
    if (isAscend950 && aivNum < aicNum * aivPerBlock) {
        aivNum = aicNum * aivPerBlock;
    }
    const uint32_t mixGroups = std::min(aicNum, aivNum / aivPerBlock);
    if (mixGroups == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "Mix core count", "invalid",
            "Recompute needs one AIC and two AIVs; platform reports no complete Mix group.");
        return ge::GRAPH_FAILED;
    }
    const int64_t headTasks = isAscend950 ? tiling->Hv :
        (tiling->Hv + KDA::RECOMPUTE_HEADS_PER_GROUP - 1) / KDA::RECOMPUTE_HEADS_PER_GROUP;
    const int64_t workUnits = tiling->chunkNum * headTasks;
    uint32_t usedAic = static_cast<uint32_t>(
        std::max<int64_t>(1, std::min(static_cast<int64_t>(mixGroups), workUnits)));
    tiling->usedAic = static_cast<int64_t>(usedAic);
    uint32_t blockDim = usedAic;
    if (isAscend950) {
        blockDim = ascendcPlatform.CalcTschBlockDim(usedAic * aivPerBlock, aicNum, aivNum);
    }
    if (blockDim < usedAic) {
        blockDim = usedAic;
    }
    context->SetBlockDim(blockDim);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = processor.GetWorkspaceSize(usedAic);
    context->SetScheduleMode(1);
    auto *rawTiling = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTiling);
    rawTiling->SetDataSize(sizeof(ChunkKdaBwdRecomputeTilingData));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingRecomputeForChunkKdaBwdRecompute(gert::TilingParseContext *context)
{
    auto *compileInfo = context->GetCompiledInfo<ChunkKdaBwdRecomputeCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    compileInfo->aicNum = 28;
    compileInfo->aivNum = 56;
    auto *platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
        const uint32_t aicNum = platform.GetCoreNumAic();
        const uint32_t aivNum = platform.GetCoreNumAiv();
        if (aicNum > 2U) {
            compileInfo->aicNum = aicNum;
        }
        if (aivNum >= 8U) {
            compileInfo->aivNum = aivNum;
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwdRecompute)
    .Tiling(Tiling4ChunkKdaBwdRecompute)
    .TilingParse<ChunkKdaBwdRecomputeCompileInfo>(TilingRecomputeForChunkKdaBwdRecompute);

} // namespace optiling
