/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_tiling.cpp
 * \brief Tiling implementation for fused prepare_wy_repr_bwd.
 */

#include "prepare_wy_repr_bwd_tiling.h"

#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"

namespace optiling {

namespace {

void PrepareWyReprBwdTilingDataPrint(gert::TilingContext *context, const PrepareWyReprBwdTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print PrepareWyReprBwd tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== B: %ld", tiling.B);
    OP_LOGD(nodeName, "=== HV: %ld", tiling.HV);
    OP_LOGD(nodeName, "=== HK: %ld", tiling.HK);
    OP_LOGD(nodeName, "=== groupSize: %ld", tiling.groupSize);
    OP_LOGD(nodeName, "=== T: %ld", tiling.T);
    OP_LOGD(nodeName, "=== K: %ld", tiling.K);
    OP_LOGD(nodeName, "=== V: %ld", tiling.V);
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling.chunkSize);
    OP_LOGD(nodeName, "=== chunkNum: %ld", tiling.chunkNum);
    OP_LOGD(nodeName, "=== chunkNumPerB: %ld", tiling.chunkNumPerB);
    OP_LOGD(nodeName, "=== isVariable: %ld", tiling.isVariable);
    OP_LOGD(nodeName, "=== workspaceBufferCount: %ld", tiling.workspaceBufferCount);
    OP_LOGD(nodeName, "=== workspaceSlotSize: %ld", tiling.workspaceSlotSize);
    OP_LOGD(nodeName, "=== workspaceCoreSize: %ld", tiling.workspaceCoreSize);
    OP_LOGD(nodeName, "=== kVecRow: %ld", tiling.kVecRow);
    OP_LOGD(nodeName, "=== vVecRow: %ld", tiling.vVecRow);
    OP_LOGD(nodeName, "=== mVecRow: %ld", tiling.mVecRow);
    OP_LOGD(nodeName, "=== kktVecRow: %ld", tiling.kktVecRow);
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print PrepareWyReprBwd tiling data end <<<<<<<<<<<<<<<<");
}

} // namespace

ge::graphStatus Tiling4PrepareWyReprBwd(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4PrepareWyReprBwd start.");
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    PrepareWyReprBwdTilingData *tiling = context->GetTilingData<PrepareWyReprBwdTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);

    auto kDesc = context->GetInputDesc(PREPARE_WY_REPR_BWD_INPUT_K_IDX);
    auto betaDesc = context->GetInputDesc(PREPARE_WY_REPR_BWD_INPUT_BETA_IDX);
    auto gDesc = context->GetInputDesc(PREPARE_WY_REPR_BWD_INPUT_G_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gDesc);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    auto cuSeqlensTensor = context->GetOptionalInputTensor(PREPARE_WY_REPR_BWD_INPUT_SEQLENS_IDX);
    auto chunkIndicesTensor = context->GetOptionalInputTensor(PREPARE_WY_REPR_BWD_INPUT_CHUNK_INDICES_IDX);
    const int64_t *cuSeqlensData = cuSeqlensTensor == nullptr ? nullptr : cuSeqlensTensor->GetData<int64_t>();
    const int64_t *chunkIndicesData =
        chunkIndicesTensor == nullptr ? nullptr : chunkIndicesTensor->GetData<int64_t>();

    PrepareWyReprBwdTilingContext ctx{
        context->GetNodeName(),
        context->GetRequiredInputShape(PREPARE_WY_REPR_BWD_INPUT_K_IDX),
        context->GetRequiredInputShape(PREPARE_WY_REPR_BWD_INPUT_V_IDX),
        context->GetRequiredInputShape(PREPARE_WY_REPR_BWD_INPUT_BETA_IDX),
        context->GetRequiredInputShape(PREPARE_WY_REPR_BWD_INPUT_A_IDX),
        context->GetRequiredInputShape(PREPARE_WY_REPR_BWD_INPUT_DW_IDX),
        context->GetRequiredInputShape(PREPARE_WY_REPR_BWD_INPUT_DU_IDX),
        context->GetRequiredInputShape(PREPARE_WY_REPR_BWD_INPUT_G_IDX),
        context->GetOptionalInputShape(PREPARE_WY_REPR_BWD_INPUT_SEQLENS_IDX),
        context->GetOptionalInputShape(PREPARE_WY_REPR_BWD_INPUT_CHUNK_INDICES_IDX),
        cuSeqlensData,
        chunkIndicesData,
        static_cast<int64_t>(*(attrPtr->GetAttrPointer<int32_t>(PREPARE_WY_REPR_BWD_ATTR_CHUNK_SIZE_IDX))),
        kDesc->GetDataType(),
        gDesc->GetDataType(),
        betaDesc->GetDataType(),
        ubSize,
        static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic()),
        static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize()),
    };

    PrepareWyReprBwdTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    context->SetTilingKey(processor.GetTilingKey());
    context->SetBlockDim(processor.GetBlockDim());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);

    OP_LOGD(context->GetNodeName(), "tilingKey: %lu (k dtype=%d, g dtype=%d, V=%ld, BT=%ld)",
            static_cast<unsigned long>(context->GetTilingKey()), static_cast<int32_t>(ctx.kDataType),
            static_cast<int32_t>(ctx.gDataType), tiling->V, tiling->chunkSize);
    PrepareWyReprBwdTilingDataPrint(context, *tiling);
    OP_LOGD(context->GetNodeName(), "Tiling4PrepareWyReprBwd end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForPrepareWyReprBwd(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PrepareWyReprBwd)
    .Tiling(Tiling4PrepareWyReprBwd)
    .TilingParse<PrepareWyReprBwdCompileInfo>(TilingPrepareForPrepareWyReprBwd);

} // namespace optiling
