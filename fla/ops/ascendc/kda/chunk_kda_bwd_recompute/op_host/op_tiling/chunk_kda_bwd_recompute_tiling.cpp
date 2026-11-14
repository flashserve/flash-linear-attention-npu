/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "chunk_kda_bwd_recompute_tiling.h"
#include "../chunk_kda_bwd_recompute_tiling_processor.h"
#include <algorithm>
#include <cstring>
#include <register/op_impl_registry.h>
#include "platform/soc_spec.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

ge::graphStatus Tiling4ChunkKdaBwdRecompute(gert::TilingContext *context)
{
    ChunkKdaBwdRecomputeTilingData *tiling = context->GetTilingData<ChunkKdaBwdRecomputeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);
    const char *inputLayout = attrPtr->GetStr(KDA_RECOMPUTE_INPUT_LAYOUT_ATTR);
    OP_CHECK_IF(inputLayout == nullptr ||
                    (std::strcmp(inputLayout, "BNSD") != 0 && std::strcmp(inputLayout, "NTD") != 0),
                OP_LOGE(context->GetNodeName(), "Recompute input_layout must be BNSD or NTD."),
                return ge::GRAPH_FAILED);

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

    const bool isAscend950 = ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510;
    const bool isAscend960 = ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_9201;
    const bool enableCompactWorkspace = isAscend950 || isAscend960;
    OP_CHECK_IF(!isAscend950 && !isAscend960 &&
                    context->GetOutputShape(KDA_RECOMPUTE_GK_OUTPUT_IDX) == nullptr,
                OP_LOGE(context->GetNodeName(), "gk output buffer is required on A2/A3."),
                return ge::GRAPH_FAILED);
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
        enableCompactWorkspace,
        isAscend960,
    };

    ChunkKdaBwdRecomputeTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    context->SetTilingKey(processor.GetTilingKey());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    const uint32_t coreNum = ascendcPlatform.GetCoreNum();
    // Mix 1:2 platform info sometimes reports a single AIC/AIV pair instead of
    // the chip total. Fall back to AIV/2, then GetCoreNum, then A5's 28 AIC.
    if (!isAscend960 && aicNum <= 2U) {
        if (aivNum >= 8U) {
            aicNum = aivNum / 2U;
        } else if (coreNum >= 8U) {
            aicNum = coreNum;
        } else {
            aicNum = 28U;
        }
    }
    // This A2/910B SKU exposes 20 complete 1:2 Mix groups. Platform metadata can
    // report more logical cores; scheduling group 21 leaves an unpaired waiter.
    if (!isAscend950 && !isAscend960) {
        aicNum = std::min(aicNum, 20U);
    }
    const uint32_t aivPerBlock = isAscend960 ? 1U : 2U;
    if (isAscend960) {
        // Like MergeFwdBwdKernel, consume the core counts saved by TilingParse
        // when the runtime platform reports zero. Preserve nonzero queries.
        const auto *compileInfo = context->GetCompileInfo<ChunkKdaBwdRecomputeCompileInfo>();
        if (aicNum == 0) {
            aicNum = compileInfo != nullptr && compileInfo->aicNum > 0 ?
                compileInfo->aicNum : kA6DefaultAicNum;
        }
        if (aivNum == 0) {
            aivNum = compileInfo != nullptr && compileInfo->aivNum > 0 ?
                compileInfo->aivNum : kA6DefaultAivNum;
        }
        aicNum = std::min(aicNum, aivNum);
        aivNum = aicNum;
    } else if (aivNum < aicNum * aivPerBlock) {
        aivNum = aicNum * aivPerBlock;
    }
    const int64_t workUnits = std::max<int64_t>(1, tiling->chunkNum * std::max<int64_t>(tiling->Hv, 1));
    uint32_t usedAic = static_cast<uint32_t>(
        std::max<int64_t>(1, std::min(static_cast<int64_t>(aicNum), workUnits)));
    if (isAscend960 && tiling->chunkNum == 0) {
        usedAic = 0;
    }
    tiling->usedAic = static_cast<int64_t>(usedAic);
    tiling->reserved = 0;
    // An empty A6 call launches one inert block; both engines drain only
    // their initialized local events and issue no cross-core notifications.
    uint32_t blockDim = std::max(usedAic, 1U);
    if (aivPerBlock == 2U) {
        blockDim = ascendcPlatform.CalcTschBlockDim(usedAic * aivPerBlock, aicNum, aivNum);
    }
    if (blockDim < usedAic) {
        blockDim = usedAic;
    }
    context->SetBlockDim(blockDim);
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
    auto *compileInfo = context->GetCompiledInfo<ChunkKdaBwdRecomputeCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    compileInfo->aicNum = 28;
    compileInfo->aivNum = 56;
    auto *platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
        const uint32_t aicNum = platform.GetCoreNumAic();
        const uint32_t aivNum = platform.GetCoreNumAiv();
        if (platform.GetCurNpuArch() == NpuArch::DAV_9201) {
            compileInfo->aicNum = kA6DefaultAicNum;
            compileInfo->aivNum = kA6DefaultAivNum;
            if (aicNum > 0) {
                compileInfo->aicNum = aicNum;
            }
            if (aivNum > 0) {
                compileInfo->aivNum = aivNum;
            }
            return ge::GRAPH_SUCCESS;
        }
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
