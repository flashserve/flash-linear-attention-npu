/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_wu_fwd_tiling.cpp
 * \brief
 */

#include "recompute_wu_fwd_tiling.h"
#include "recompute_wu_fwd_tiling_processor.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

static constexpr size_t INPUT_GK_IDX = 5;
static constexpr size_t OUTPUT_W_IDX = 0;
static constexpr size_t OUTPUT_U_IDX = 1;

static void RecomputeWUFwdTilingDataPrint(gert::TilingContext *context, const RecomputeWUFwdTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print RecomputeWUFwd tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== B: %ld", tiling.B);
    OP_LOGD(nodeName, "=== Hk: %ld", tiling.Hk);
    OP_LOGD(nodeName, "=== Hv: %ld", tiling.Hv);
    OP_LOGD(nodeName, "=== hvPerHk: %ld", tiling.hvPerHk);
    OP_LOGD(nodeName, "=== T: %ld", tiling.T);
    OP_LOGD(nodeName, "=== K: %ld", tiling.K);
    OP_LOGD(nodeName, "=== V: %ld", tiling.V);
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling.chunkSize);
    OP_LOGD(nodeName, "=== vbVecRow: %ld", tiling.vbVecRow);
    OP_LOGD(nodeName, "=== kbgExpVecRow: %ld", tiling.kbgExpVecRow);
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print RecomputeWUFwd tiling data end <<<<<<<<<<<<<<<<");
}

ge::graphStatus Tiling4RecomputeWUFwd(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Tiling4RecomputeWUFwd start.");
    RecomputeWUFwdTilingData *tiling = context->GetTilingData<RecomputeWUFwdTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);

    auto kDesc = context->GetInputDesc(RECOMPUTE_WU_FWD_INPUT_K_IDX);
    auto vDesc = context->GetInputDesc(RECOMPUTE_WU_FWD_INPUT_V_IDX);
    auto betaDesc = context->GetInputDesc(RECOMPUTE_WU_FWD_INPUT_BETA_IDX);
    auto aDesc = context->GetInputDesc(RECOMPUTE_WU_FWD_INPUT_A_IDX);
    auto gDesc = context->GetInputDesc(RECOMPUTE_WU_FWD_INPUT_G_IDX);
    auto wDesc = context->GetOutputDesc(OUTPUT_W_IDX);
    auto uDesc = context->GetOutputDesc(OUTPUT_U_IDX);
    OP_CHECK_IF(kDesc == nullptr || vDesc == nullptr || betaDesc == nullptr || aDesc == nullptr ||
                    gDesc == nullptr || wDesc == nullptr || uDesc == nullptr,
                OP_LOGE(context->GetNodeName(), "required input/output dtype descriptors must be present."),
                return ge::GRAPH_FAILED);
    const ge::DataType inputDtype = kDesc->GetDataType();
    OP_CHECK_IF((inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_BF16) ||
                    vDesc->GetDataType() != inputDtype || aDesc->GetDataType() != inputDtype ||
                    wDesc->GetDataType() != inputDtype || uDesc->GetDataType() != inputDtype,
                OP_LOGE(context->GetNodeName(), "k/v/A/w/u must use matching float16 or bfloat16 dtype."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((betaDesc->GetDataType() != inputDtype && betaDesc->GetDataType() != ge::DT_FLOAT) ||
                    gDesc->GetDataType() != betaDesc->GetDataType(),
                OP_LOGE(context->GetNodeName(), "beta/g must share float32 or the main input dtype."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetOptionalInputDesc(INPUT_GK_IDX) != nullptr ||
                    context->GetOptionalInputTensor(INPUT_GK_IDX) != nullptr,
                OP_LOGE(context->GetNodeName(), "gk is reserved and must be absent."),
                return ge::GRAPH_FAILED);

    auto wShapePtr = context->GetOutputShape(OUTPUT_W_IDX);
    auto uShapePtr = context->GetOutputShape(OUTPUT_U_IDX);
    auto kShapePtr = context->GetRequiredInputShape(RECOMPUTE_WU_FWD_INPUT_K_IDX);
    auto vShapePtr = context->GetRequiredInputShape(RECOMPUTE_WU_FWD_INPUT_V_IDX);
    OP_CHECK_IF(wShapePtr == nullptr || uShapePtr == nullptr || kShapePtr == nullptr || vShapePtr == nullptr,
                OP_LOGE(context->GetNodeName(), "required input/output shapes must be present."),
                return ge::GRAPH_FAILED);
    const gert::Shape kShape = kShapePtr->GetStorageShape();
    const gert::Shape vShape = vShapePtr->GetStorageShape();
    const gert::Shape wShape = wShapePtr->GetStorageShape();
    const gert::Shape uShape = uShapePtr->GetStorageShape();
    OP_CHECK_IF(kShape.GetDimNum() != 4 || vShape.GetDimNum() != 4 || wShape.GetDimNum() != 4 ||
                    uShape.GetDimNum() != 4,
                OP_LOGE(context->GetNodeName(), "k/v/w/u must be rank-4 tensors."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(wShape.GetDim(0) != kShape.GetDim(0) || wShape.GetDim(1) != vShape.GetDim(1) ||
                    wShape.GetDim(2) != kShape.GetDim(2) || wShape.GetDim(3) != kShape.GetDim(3) ||
                    uShape.GetDim(0) != vShape.GetDim(0) || uShape.GetDim(1) != vShape.GetDim(1) ||
                    uShape.GetDim(2) != vShape.GetDim(2) || uShape.GetDim(3) != vShape.GetDim(3),
                OP_LOGE(context->GetNodeName(), "w must be [B,H_v,T,K] and u must match v."),
                return ge::GRAPH_FAILED);

    auto chunkSizePtr = attrPtr->GetAttrPointer<int32_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSizePtr);
    OP_CHECK_IF(*chunkSizePtr != 64 && *chunkSizePtr != 128,
                OP_LOGE(context->GetNodeName(), "chunk_size must be 64 or 128."),
                return ge::GRAPH_FAILED);

    auto cuDesc = context->GetOptionalInputDesc(RECOMPUTE_WU_FWD_INPUT_SEQLENS_IDX);
    auto indicesDesc = context->GetOptionalInputDesc(RECOMPUTE_WU_FWD_INPUT_CHUNK_INDICES_IDX);
    OP_CHECK_IF((cuDesc == nullptr) != (indicesDesc == nullptr),
                OP_LOGE(context->GetNodeName(), "cu_seqlens and chunk_indices must be provided together."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cuDesc != nullptr &&
                    (cuDesc->GetDataType() != ge::DT_INT64 || indicesDesc->GetDataType() != ge::DT_INT64),
                OP_LOGE(context->GetNodeName(), "cu_seqlens and chunk_indices must use int64."),
                return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    OP_CHECK_IF(ascendcPlatform.GetCoreNumAic() == 0,
                OP_LOGE(context->GetNodeName(), "AIC core count must be positive."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context->GetNodeName(), "UB size must be positive."),
                return ge::GRAPH_FAILED);
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    auto cuSeqlensTensor = context->GetOptionalInputTensor(RECOMPUTE_WU_FWD_INPUT_SEQLENS_IDX);
    auto chunkIndicesTensor = context->GetOptionalInputTensor(RECOMPUTE_WU_FWD_INPUT_CHUNK_INDICES_IDX);
    const int64_t *cuSeqlensData = cuSeqlensTensor != nullptr ? cuSeqlensTensor->GetData<int64_t>() : nullptr;
    const int64_t *chunkIndicesData =
        chunkIndicesTensor != nullptr ? chunkIndicesTensor->GetData<int64_t>() : nullptr;

    RecomputeWUFwdTilingContext ctx{
        context->GetNodeName(),
        context->GetRequiredInputShape(RECOMPUTE_WU_FWD_INPUT_K_IDX),
        context->GetRequiredInputShape(RECOMPUTE_WU_FWD_INPUT_V_IDX),
        context->GetRequiredInputShape(RECOMPUTE_WU_FWD_INPUT_BETA_IDX),
        context->GetRequiredInputShape(RECOMPUTE_WU_FWD_INPUT_A_IDX),
        context->GetRequiredInputShape(RECOMPUTE_WU_FWD_INPUT_G_IDX),
        context->GetOptionalInputShape(RECOMPUTE_WU_FWD_INPUT_SEQLENS_IDX),
        context->GetOptionalInputShape(RECOMPUTE_WU_FWD_INPUT_CHUNK_INDICES_IDX),
        cuSeqlensData,
        chunkIndicesData,
        *chunkSizePtr,
        kDesc->GetDataType(),
        betaDesc->GetDataType(),
        ubSize,
        sysWorkspaceSize,
    };

    RecomputeWUFwdTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    if (tiling->V == RECOMPUTE_WU_FWD_V_DIM_256) {
        context->SetTilingKey(2);
    } else {
        context->SetTilingKey(1);
    }
    OP_LOGD(context->GetNodeName(), "tilingKey: %d (V=%ld)", context->GetTilingKey(), tiling->V);
    RecomputeWUFwdTilingDataPrint(context, *tiling);

    context->SetBlockDim(ascendcPlatform.GetCoreNumAic());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);
    OP_LOGD(context->GetNodeName(), "Tiling4RecomputeWUFwd end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingRecomputeForRecomputeWUFwd(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RecomputeWUFwd)
    .Tiling(Tiling4RecomputeWUFwd)
    .TilingParse<RecomputeWUFwdCompileInfo>(TilingRecomputeForRecomputeWUFwd);

} // namespace optiling
