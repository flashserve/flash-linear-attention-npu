/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_bwd_dhu_tiling.cpp
 * \brief
 */

#include "chunk_gated_delta_rule_bwd_dhu_tiling_processor.h"
#include "chunk_gated_delta_rule_bwd_dhu_tiling.h"
#include "err/ops_err.h"

using namespace GDN;

namespace optiling {
namespace {
constexpr uint32_t INPUT_Q_IDX = 0;
constexpr uint32_t INPUT_K_IDX = 1;
constexpr uint32_t INPUT_W_IDX = 2;
constexpr uint32_t INPUT_DO_IDX = 3;
constexpr uint32_t INPUT_DV_IDX = 4;
constexpr uint32_t INPUT_G_IDX = 5;
constexpr uint32_t INPUT_CU_SEQLENS_IDX = 9;
constexpr uint32_t INPUT_CHUNK_INDICES_IDX = 10;

constexpr uint32_t ATTR_SCALE_IDX = 0;
constexpr uint32_t ATTR_CHUNK_SIZE_IDX = 1;

static void ChunkGatedDeltaRuleBwdDhuTilingDataPrint(gert::TilingContext *context,
                                                     const ChunkGatedDeltaRuleBwdDhuTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "End Run ChunkGatedDeltaRuleBwdDhu Tiling");
    OP_LOGD(nodeName, "B is %lu.", tiling.B);
    OP_LOGD(nodeName, "Hv is %lu.", tiling.Hv);
    OP_LOGD(nodeName, "Hk is %lu.", tiling.Hk);
    OP_LOGD(nodeName, "T is %lu.", tiling.T);
    OP_LOGD(nodeName, "K is %lu.", tiling.K);
    OP_LOGD(nodeName, "V is %lu.", tiling.V);
    OP_LOGD(nodeName, "chunkSize is %lu.", tiling.chunkSize);
    OP_LOGD(nodeName, "chunkNum is %lu.", tiling.chunkNum);
    OP_LOGD(nodeName, "seqNum is %lu.", tiling.seqNum);
    OP_LOGD(nodeName, "gBufSize is %lu.", tiling.gBufSize);
    OP_LOGD(nodeName, "dvBufSize is %lu.", tiling.dvBufSize);
    OP_LOGD(nodeName, "qBufSize is %lu.", tiling.qBufSize);
    OP_LOGD(nodeName, "dhBufSize is %lu.", tiling.dhBufSize);
    OP_LOGD(nodeName, "totalTbufByte is %lu.", tiling.totalTbufByte);
    OP_LOGD(nodeName, "bdvWs is %lu.", tiling.bdvWs);
    OP_LOGD(nodeName, "qWs is %lu.", tiling.qWs);
    OP_LOGD(nodeName, "wDv2Ws is %lu.", tiling.wDv2Ws);
    OP_LOGD(nodeName, "qDoWs is %lu.", tiling.qDoWs);
    OP_LOGD(nodeName, "isVarLen is %lu.", tiling.isVarLen);
    OP_LOGD(nodeName, "isScale is %lu.", tiling.isScale);
    OP_LOGD(nodeName, "usedCoreNum is %u.", tiling.usedCoreNum);
    OP_LOGD(nodeName, "scale is %f.", tiling.scale);
}
} // namespace

ASCENDC_EXTERN_C ge::graphStatus Tiling4ChunkGDRBwdDhu(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkGDRBwdDhu start.");
    ChunkGatedDeltaRuleBwdDhuTilingData *tiling = context->GetTilingData<ChunkGatedDeltaRuleBwdDhuTilingData>();

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context->GetNodeName(), "attrs is nullptr."), return ge::GRAPH_FAILED);
    const double *scalePtr = attrs->GetAttrPointer<double>(ATTR_SCALE_IDX);
    const uint32_t *chunkSizePtr = attrs->GetAttrPointer<uint32_t>(ATTR_CHUNK_SIZE_IDX);

    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null!"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint64_t maxUbSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize);

    const auto gInputDesc = context->GetOptionalInputDesc(INPUT_G_IDX);
    const auto qInputDesc = context->GetInputDesc(INPUT_Q_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qInputDesc);

    const gert::StorageShape *gShapePtr = nullptr;
    gert::StorageShape gShape;
    if (context->GetOptionalInputShape(INPUT_G_IDX) != nullptr) {
        gShape = *context->GetOptionalInputShape(INPUT_G_IDX);
        gShapePtr = &gShape;
    }

    ChunkGatedDeltaRuleBwdDhuTilingContext ctx{
        context->GetNodeName(),
        context->GetInputShape(INPUT_Q_IDX),
        context->GetInputShape(INPUT_K_IDX),
        context->GetInputShape(INPUT_W_IDX),
        context->GetInputShape(INPUT_DO_IDX),
        context->GetInputShape(INPUT_DV_IDX),
        gShapePtr,
        context->GetOptionalInputShape(INPUT_CU_SEQLENS_IDX),
        context->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX),
        qInputDesc->GetDataType(),
        gInputDesc != nullptr ? gInputDesc->GetDataType() : ge::DT_FLOAT,
        gShapePtr != nullptr,
        scalePtr != nullptr,
        scalePtr != nullptr ? *scalePtr : 1.0,
        chunkSizePtr != nullptr ? static_cast<int32_t>(*chunkSizePtr) : static_cast<int32_t>(64),
        maxUbSize,
        static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic()),
        static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize()),
    };

    ChunkGatedDeltaRuleBwdDhuTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "tiling process failed"),
                return ge::GRAPH_FAILED);

    context->SetTilingKey(processor.GetTilingKey());
    context->SetBlockDim(processor.GetBlockDim());
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = processor.GetWorkspaceSize();

    ChunkGatedDeltaRuleBwdDhuTilingDataPrint(context, *tiling);
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkGDRBwdDhu end.");
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepare4ChunkGDRBwdDhu(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleBwdDhu)
    .Tiling(Tiling4ChunkGDRBwdDhu)
    .TilingParse<ChunkGatedDeltaRuleBwdDhuCompileInfo>(TilingPrepare4ChunkGDRBwdDhu);

} // namespace optiling
