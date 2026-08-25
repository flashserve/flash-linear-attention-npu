/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_bwd_dhu_tiling.cpp
 * \brief Tiling implementation for chunk_gated_delta_rule_bwd_dhu.
 */

#include "chunk_gated_delta_rule_bwd_dhu_tiling.h"
#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"

namespace optiling {

namespace {

void ChunkGatedDeltaRuleBwdDhuTilingDataPrint(
    gert::TilingContext *context, const ChunkGatedDeltaRuleBwdDhuTilingData &tiling)
{
    const auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print ChunkGatedDeltaRuleBwdDhu tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== B: %ld", tiling.B);
    OP_LOGD(nodeName, "=== HK: %ld", tiling.HK);
    OP_LOGD(nodeName, "=== HV: %ld", tiling.HV);
    OP_LOGD(nodeName, "=== T: %ld", tiling.T);
    OP_LOGD(nodeName, "=== K: %ld", tiling.K);
    OP_LOGD(nodeName, "=== V: %ld", tiling.V);
    OP_LOGD(nodeName, "=== HRatio: %ld", tiling.HRatio);
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling.chunkSize);
    OP_LOGD(nodeName, "=== chunkNumForT: %ld", tiling.chunkNumForT);
    OP_LOGD(nodeName, "=== totalChunkNum: %ld", tiling.totalChunkNum);
    OP_LOGD(nodeName, "=== chunkTaskNum: %ld", tiling.chunkTaskNum);
    OP_LOGD(nodeName, "=== seqNum: %ld", tiling.seqNum);
    OP_LOGD(nodeName, "=== headsPerTask: %ld", tiling.headsPerTask);
    OP_LOGD(nodeName, "=== headWindowNum: %ld", tiling.headWindowNum);
    OP_LOGD(nodeName, "=== taskNum: %ld", tiling.taskNum);
    OP_LOGD(nodeName, "=== isVariable: %ld", tiling.isVariable);
    OP_LOGD(nodeName, "=== hasDh0: %ld", tiling.hasDh0);
    OP_LOGD(nodeName, "=== dh0ClearCoreNum: %ld", tiling.dh0ClearCoreNum);
    OP_LOGD(nodeName, "=== dh0ClearElemsPerCore: %ld", tiling.dh0ClearElemsPerCore);
    OP_LOGD(nodeName, "=== dh0ClearTailElems: %ld", tiling.dh0ClearTailElems);
    OP_LOGD(nodeName, "=== hasGk: %ld", tiling.hasGk);
    OP_LOGD(nodeName, "=== workspaceElemsPerSubBlock: %ld", tiling.workspaceElemsPerSubBlock);
    OP_LOGD(nodeName, "=== qgWorkspaceOffset: %ld", tiling.qgWorkspaceOffset);
    OP_LOGD(nodeName, "=== stateWorkspaceOffset: %ld", tiling.stateWorkspaceOffset);
    OP_LOGD(nodeName, "=== dvStateWorkspaceOffset: %ld", tiling.dvStateWorkspaceOffset);
    OP_LOGD(nodeName, "=== termQWorkspaceOffset: %ld", tiling.termQWorkspaceOffset);
    OP_LOGD(nodeName, "=== dv2WorkspaceOffset: %ld", tiling.dv2WorkspaceOffset);
    OP_LOGD(nodeName, "=== termWWorkspaceOffset: %ld", tiling.termWWorkspaceOffset);
    OP_LOGD(nodeName, "=== qgWorkspaceElems: %ld", tiling.qgWorkspaceElems);
    OP_LOGD(nodeName, "=== stateWorkspaceElems: %ld", tiling.stateWorkspaceElems);
    OP_LOGD(nodeName, "=== dvStateWorkspaceElems: %ld", tiling.dvStateWorkspaceElems);
    OP_LOGD(nodeName, "=== termQWorkspaceElems: %ld", tiling.termQWorkspaceElems);
    OP_LOGD(nodeName, "=== dv2WorkspaceElems: %ld", tiling.dv2WorkspaceElems);
    OP_LOGD(nodeName, "=== termWWorkspaceElems: %ld", tiling.termWWorkspaceElems);
    OP_LOGD(nodeName, "=== vecRow: %ld", tiling.vecRow);
    OP_LOGD(nodeName, "=== scale: %f", tiling.scale);
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print ChunkGatedDeltaRuleBwdDhu tiling data end <<<<<<<<<<<<<<<<");
}

} // namespace

ge::graphStatus Tiling4ChunkGatedDeltaRuleBwdDhu(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkGatedDeltaRuleBwdDhu start.");
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    ChunkGatedDeltaRuleBwdDhuTilingData *tiling =
        context->GetTilingData<ChunkGatedDeltaRuleBwdDhuTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);

    auto qInputDesc = context->GetInputDesc(CGDR_BWD_DHU_INPUT_Q_IDX);
    auto gInputDesc = context->GetOptionalInputDesc(CGDR_BWD_DHU_INPUT_G_IDX);
    auto gkInputDesc = context->GetOptionalInputDesc(CGDR_BWD_DHU_INPUT_GK_IDX);
    auto gInputShape = context->GetOptionalInputShape(CGDR_BWD_DHU_INPUT_G_IDX);
    auto gkInputShape = context->GetOptionalInputShape(CGDR_BWD_DHU_INPUT_GK_IDX);
    auto h0InputShape = context->GetOptionalInputShape(CGDR_BWD_DHU_INPUT_H0_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qInputDesc);

    const bool hasG = gInputShape != nullptr;
    const bool hasGk = gkInputShape != nullptr;
    const ge::DataType gateDataType =
        hasG ? gInputDesc->GetDataType() : (hasGk ? gkInputDesc->GetDataType() : ge::DT_FLOAT);

    const double *scalePtr = attrPtr->GetAttrPointer<double>(CGDR_BWD_DHU_ATTR_SCALE_IDX);
    const int32_t *chunkSizePtr = attrPtr->GetAttrPointer<int32_t>(CGDR_BWD_DHU_ATTR_CHUNK_SIZE_IDX);
    const bool *useExp2Ptr = attrPtr->GetAttrPointer<bool>(CGDR_BWD_DHU_ATTR_USE_EXP2_IDX);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    ChunkGatedDeltaRuleBwdDhuTilingContext ctx{
        context->GetNodeName(),
        context->GetRequiredInputShape(CGDR_BWD_DHU_INPUT_Q_IDX),
        context->GetRequiredInputShape(CGDR_BWD_DHU_INPUT_K_IDX),
        context->GetRequiredInputShape(CGDR_BWD_DHU_INPUT_W_IDX),
        context->GetRequiredInputShape(CGDR_BWD_DHU_INPUT_DO_IDX),
        context->GetRequiredInputShape(CGDR_BWD_DHU_INPUT_DV_IDX),
        gInputShape,
        gkInputShape,
        context->GetOptionalInputShape(CGDR_BWD_DHU_INPUT_CU_SEQLENS_IDX),
        context->GetOptionalInputShape(CGDR_BWD_DHU_INPUT_CHUNK_INDICES_IDX),
        qInputDesc->GetDataType(),
        gateDataType,
        hasG,
        hasGk,
        useExp2Ptr != nullptr ? *useExp2Ptr : false,
        h0InputShape != nullptr,
        true,
        scalePtr != nullptr ? static_cast<double>(*scalePtr) : 1.0,
        chunkSizePtr != nullptr ? *chunkSizePtr : 64,
        ubSize,
        static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic()),
        static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize()),
    };

    ChunkGatedDeltaRuleBwdDhuTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    context->SetTilingKey(processor.GetTilingKey());
    context->SetBlockDim(processor.GetBlockDim());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);

    OP_LOGD(context->GetNodeName(), "tilingKey: %u", context->GetTilingKey());
    ChunkGatedDeltaRuleBwdDhuTilingDataPrint(context, *tiling);
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkGatedDeltaRuleBwdDhu end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkGatedDeltaRuleBwdDhu(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleBwdDhu)
    .Tiling(Tiling4ChunkGatedDeltaRuleBwdDhu)
    .TilingParse<ChunkGatedDeltaRuleBwdDhuCompileInfo>(TilingPrepareForChunkGatedDeltaRuleBwdDhu);

} // namespace optiling
