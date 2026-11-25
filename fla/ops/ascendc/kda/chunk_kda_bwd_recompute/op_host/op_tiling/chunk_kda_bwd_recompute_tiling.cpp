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
    if (inputLayout == nullptr ||
                    (std::strcmp(inputLayout, "BNSD") != 0 && std::strcmp(inputLayout, "NTD") != 0)) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context->GetNodeName(), "input_layout",
            inputLayout == nullptr ? "<null>" : inputLayout,
            "Recompute input_layout must be BNSD or NTD");
        return ge::GRAPH_FAILED;
    }

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const auto *gDesc = context->GetInputDesc(KDA_RECOMPUTE_G_IDX);
    const auto *betaDesc = context->GetInputDesc(KDA_RECOMPUTE_BETA_IDX);
    if (gDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "g");
        return ge::GRAPH_FAILED;
    }
    if (betaDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "beta");
        return ge::GRAPH_FAILED;
    }

    auto cuSeqlensTensor = context->GetOptionalInputTensor(KDA_RECOMPUTE_CU_SEQLENS_IDX);
    auto chunkIndicesTensor = context->GetOptionalInputTensor(KDA_RECOMPUTE_CHUNK_INDICES_IDX);
    const int64_t *cuSeqlensData = cuSeqlensTensor != nullptr ? cuSeqlensTensor->GetData<int64_t>() : nullptr;
    const int64_t *chunkIndicesData =
        chunkIndicesTensor != nullptr ? chunkIndicesTensor->GetData<int64_t>() : nullptr;

    const bool isAscend950 = ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510;
    const bool isAscend960 = ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_9201;
    const bool enableCompactWorkspace = isAscend950 || isAscend960;
    if (!enableCompactWorkspace && context->GetOutputShape(4) == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "buffer");
        return ge::GRAPH_FAILED;
    }
    const char *modelLayout = attrPtr->GetStr(5);
    const bool mixed = modelLayout && modelLayout[0];
    const bool modelTm = mixed && (std::strcmp(modelLayout, "BSND") == 0 || std::strcmp(modelLayout, "TND") == 0);
    const auto *qkHeadAttr = attrPtr->GetAttrPointer<bool>(6);
    const bool qkTm = modelTm && !(qkHeadAttr && *qkHeadAttr);
    if (mixed && (!modelTm && std::strcmp(modelLayout, "BNSD") != 0 &&
                    std::strcmp(modelLayout, "NTD") != 0)) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context->GetNodeName(), "model_layout",
            modelLayout == nullptr ? "<null>" : modelLayout,
            "Unsupported Recompute model_layout on this platform");
        return ge::GRAPH_FAILED;
    }
    if (mixed && isAscend960) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context->GetNodeName(), "model_layout", modelLayout,
                                               "A6 arch92 does not support mixed model layout yet");
        return ge::GRAPH_FAILED;
    }
    if(context->GetInputDesc(0)->GetDataType()==ge::DT_FLOAT){
        if(ascendcPlatform.GetCurNpuArch()!=NpuArch::DAV_2201 || qkTm)return ge::GRAPH_FAILED;
        const auto &q=context->GetRequiredInputShape(0)->GetStorageShape();
        const auto &v=context->GetRequiredInputShape(2)->GetStorageShape();
        if(q.GetDimNum()!=4 || v.GetDimNum()!=4)return ge::GRAPH_FAILED;
        *tiling={};tiling->B=q.GetDim(0);tiling->Hk=q.GetDim(1);tiling->Hv=tiling->Hk;
        tiling->T=q.GetDim(2);tiling->K=q.GetDim(3);tiling->V=v.GetDim(3);tiling->hvPerHk=1;
        const auto K=tiling->K,V=tiling->V;
        if(tiling->B<=0 || tiling->Hv<=0 || tiling->T<=0 ||
            !((K==V && (K==60 || K==64 || K==100 || K==128 || K==256)) || (K==64 && V==32)))return ge::GRAPH_FAILED;
        if(*attrPtr->GetAttrPointer<int64_t>(0)!=64)return ge::GRAPH_FAILED;
        tiling->chunkSize=64;tiling->useGateInKernel=*attrPtr->GetAttrPointer<bool>(1);
        tiling->useExp2=*attrPtr->GetAttrPointer<bool>(2);
        float lower=*attrPtr->GetAttrPointer<float>(3);uint32_t bits;std::memcpy(&bits,&lower,4);tiling->lowerBoundBits=bits;
        tiling->hasALog=context->GetOptionalInputDesc(6)!=nullptr;
        tiling->hasDtBias=context->GetOptionalInputDesc(7)!=nullptr;
        tiling->reserved=mixed?(1 | (modelTm?4:0)):0;
        tiling->isVariable=cuSeqlensData!=nullptr;tiling->vecRow=0;
        if(tiling->isVariable){
            auto count=cuSeqlensTensor->GetShapeSize();if(count<2 || tiling->B!=1)return ge::GRAPH_FAILED;
            tiling->vecRow=count-1;
            if(cuSeqlensData[0]!=0 || cuSeqlensData[count-1]!=tiling->T)return ge::GRAPH_FAILED;
            for(int64_t i=0;i<count-1;++i){
                if(cuSeqlensData[i+1]<cuSeqlensData[i])return ge::GRAPH_FAILED;
                tiling->chunkNum+=(cuSeqlensData[i+1]-cuSeqlensData[i]+63)/64;
            }
        }else tiling->chunkNum=tiling->B*((tiling->T+63)/64);
        tiling->usedAic=std::max<int64_t>(1,std::min<int64_t>(ascendcPlatform.GetCoreNumAiv(),tiling->chunkNum*tiling->Hv));
        using namespace KDA;
        context->SetTilingKey(GET_TPL_TILING_KEY(30ULL,30ULL,128ULL,1ULL));
        context->SetBlockDim(tiling->usedAic);context->GetWorkspaceSizes(1)[0]=ascendcPlatform.GetLibApiWorkSpaceSize();
        context->GetRawTilingData()->SetDataSize(sizeof(*tiling));return ge::GRAPH_SUCCESS;
    }
    if (mixed) {
        const auto &canonical = context->GetOutputShape(4)->GetStorageShape();
        if (canonical.GetDimNum() != 4) {
            return ge::GRAPH_FAILED;
        }
        static const char *kMixedInputNames[] = {"q", "k", "v", "g"};
        for (size_t i = 0; i < 4; ++i) {
            const auto *input = context->GetRequiredInputShape(i);
            if (input == nullptr) {
                OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), kMixedInputNames[i]);
                return ge::GRAPH_FAILED;
            }
            auto expected = canonical;
            const bool tm = i < 2 ? qkTm : (i == 2 ? modelTm : true);
            if (tm) { expected.SetDim(1, canonical.GetDim(2)); expected.SetDim(2, canonical.GetDim(1)); }
            const auto actualShape = input->GetStorageShape();
            if (actualShape != expected) {
                std::string actual = std::to_string(actualShape.GetDimNum());
                for (size_t d = 0; d < actualShape.GetDimNum(); ++d) {
                    actual += "," + std::to_string(actualShape.GetDim(d));
                }
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), kMixedInputNames[i], actual,
                    "mixed-layout Recompute input must match the canonical output shape");
                return ge::GRAPH_FAILED;
            }
        }
    }
    ChunkKdaBwdRecomputeTilingContext ctx{
        context->GetNodeName(),
        mixed ? context->GetOutputShape(4) : context->GetRequiredInputShape(KDA_RECOMPUTE_Q_IDX),
        mixed ? context->GetOutputShape(4) : context->GetRequiredInputShape(KDA_RECOMPUTE_K_IDX),
        mixed ? context->GetOutputShape(4) : context->GetRequiredInputShape(KDA_RECOMPUTE_V_IDX),
        mixed ? context->GetOutputShape(4) : context->GetRequiredInputShape(KDA_RECOMPUTE_G_IDX),
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
    if (processor.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tiling->reserved = mixed ? (1 | (qkTm ? 2 : 0) | (modelTm ? 4 : 0)) : 0;
    context->SetTilingKey(processor.GetTilingKey());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t aivNum = (isAscend950 || isAscend960) ? ascendcPlatform.GetCoreNumAiv() : 0;
    const uint32_t aivPerBlock = isAscend960 ? 1U : 2U;
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
    if (isAscend960) {
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
    } else if (isAscend950 && aivNum < aicNum * aivPerBlock) {
        aivNum = aicNum * aivPerBlock;
    }
    if (!isAscend950 && !isAscend960 && aicNum == 0) {
        // Some ACLNN tiling contexts expose incomplete Mix resource metadata.
        auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (platform != nullptr) {
            aicNum = platform->GetCoreNumAic();
        }
    }
    if (aicNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "aic_num",
            std::to_string(aicNum), "Platform reports no available AIC for recompute");
        return ge::GRAPH_FAILED;
    }
    const int64_t headTasks = (isAscend950 || isAscend960) ? tiling->Hv :
        (tiling->Hv + KDA::RECOMPUTE_HEADS_PER_GROUP - 1) / KDA::RECOMPUTE_HEADS_PER_GROUP;
    const int64_t workUnits = tiling->chunkNum * headTasks;
    uint32_t usedAic = static_cast<uint32_t>(
        std::max<int64_t>(1, std::min(static_cast<int64_t>(aicNum), workUnits)));
    if (isAscend960 && tiling->chunkNum == 0) {
        usedAic = 0;
    }
    tiling->usedAic = static_cast<int64_t>(usedAic);
    uint32_t blockDim = isAscend960 ? std::max(usedAic, 1U) : usedAic;
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
