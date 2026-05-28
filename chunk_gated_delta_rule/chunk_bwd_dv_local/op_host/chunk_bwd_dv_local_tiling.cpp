/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dv_local_tiling.cpp
 * \brief
 */

#include "chunk_bwd_dv_local_tiling_processor.h"
#include "../op_kernel/chunk_bwd_dv_local_struct.h"

using namespace GDN;

namespace optiling {

struct ChunkBwdDvLocalCompileInfo {};

static void ChunkBwdDvLocalTilingDataPrint(gert::TilingContext *context, const ChunkBwdDvLocalTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print ChunkBwdDvLocal tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== b: %ld", tiling.b);
    OP_LOGD(nodeName, "=== h: %ld", tiling.h);
    OP_LOGD(nodeName, "=== t: %ld", tiling.t);
    OP_LOGD(nodeName, "=== k: %ld", tiling.k);
    OP_LOGD(nodeName, "=== v: %ld", tiling.v);
    OP_LOGD(nodeName, "=== chunkNumForT: %ld", tiling.chunkNumForT);
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling.chunkSize);
    OP_LOGD(nodeName, "=== scale: %f", tiling.scale);
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print ChunkBwdDvLocal tiling data end <<<<<<<<<<<<<<<<");
}

ge::graphStatus Tiling4ChunkBwdDvLocal(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkBwdDvLocal start.");
    ChunkBwdDvLocalTilingData *tiling = context->GetTilingData<ChunkBwdDvLocalTilingData>();

    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);
    ChunkBwdDvLocalTilingContext ctx{
        context->GetNodeName(),
        context->GetOptionalInputShape(CHUNK_BWD_DV_LOCAL_INPUT_Q_IDX),
        context->GetOptionalInputShape(CHUNK_BWD_DV_LOCAL_INPUT_K_IDX),
        context->GetOptionalInputShape(CHUNK_BWD_DV_LOCAL_INPUT_DO_IDX),
        context->GetOptionalInputShape(CHUNK_BWD_DV_LOCAL_INPUT_G_IDX),
        context->GetOptionalInputShape(CHUNK_BWD_DV_LOCAL_INPUT_SEQLENS_IDX),
        context->GetOptionalInputShape(CHUNK_BWD_DV_LOCAL_INPUT_CHUNK_INDICES_IDX),
        *(attrPtr->GetAttrPointer<double>(CHUNK_BWD_DV_LOCAL_ATTR_SCALE_IDX)),
        *(attrPtr->GetAttrPointer<int32_t>(CHUNK_BWD_DV_LOCAL_ATTR_CHUNK_SIZE_IDX)),
    };

    ChunkBwdDvLocalTilingProcessor processor(ctx, *tiling);

    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    uint64_t strategyKey = processor.IsVariableLength() ? CHUNK_BWD_DV_LOCAL_STRATEGY_VAR_LEN : CHUNK_BWD_DV_LOCAL_STRATEGY_FIX_LEN;

    auto qInputDesc = context->GetInputDesc(CHUNK_BWD_DV_LOCAL_INPUT_Q_IDX);
    auto gInputDesc = context->GetInputDesc(CHUNK_BWD_DV_LOCAL_INPUT_G_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qInputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gInputDesc);
    ge::DataType qDtype = qInputDesc->GetDataType();
    ge::DataType gDtype = gInputDesc->GetDataType();

    int dTQ = (qDtype == ge::DT_BF16) ? CHUNK_BWD_DV_LOCAL_TPL_BF16 : CHUNK_BWD_DV_LOCAL_TPL_FP16;
    int dTG = (gDtype == ge::DT_FLOAT) ? CHUNK_BWD_DV_LOCAL_TPL_FP32 : dTQ;

    int v = static_cast<int>(tiling->v);
    OP_LOGD(context->GetNodeName(), "V value: %d", v);

    uint64_t tilingKey = GET_TPL_TILING_KEY(strategyKey, dTQ, dTG, v);
    context->SetTilingKey(tilingKey);

    OP_LOGD(context->GetNodeName(), "tilingKey: %d", context->GetTilingKey());
    ChunkBwdDvLocalTilingDataPrint(context, *tiling);

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAic());
    context->SetBlockDim(std::min(tiling->chunkNumForT * tiling->b, coreNum));

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint32_t userWorkspaceSize = QKV_DTYPE_SIZE * tiling->b * tiling->h * tiling->t * tiling->chunkSize;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize + userWorkspaceSize;
    context->SetScheduleMode(1);
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkBwdDvLocal end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkBwdDvLocal(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkBwdDvLocal)
    .Tiling(Tiling4ChunkBwdDvLocal)
    .TilingParse<ChunkBwdDvLocalCompileInfo>(TilingPrepareForChunkBwdDvLocal);

} // namespace optiling
