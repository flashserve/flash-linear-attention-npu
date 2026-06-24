/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "chunk_bwd_dv_local_tiling_a5.h"

using namespace GDN;

namespace optiling {

bool ChunkBwdDvLocalTilingA5::SetTiling(gert::TilingContext *context)
{
    ChunkBwdDvLocalTilingData *tiling = context->GetTilingData<ChunkBwdDvLocalTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

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
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return false);

    uint64_t strategyKey =
        processor.IsVariableLength() ? CHUNK_BWD_DV_LOCAL_STRATEGY_VAR_LEN : CHUNK_BWD_DV_LOCAL_STRATEGY_FIX_LEN;

    auto qInputDesc = context->GetInputDesc(CHUNK_BWD_DV_LOCAL_INPUT_Q_IDX);
    auto gInputDesc = context->GetInputDesc(CHUNK_BWD_DV_LOCAL_INPUT_G_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qInputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gInputDesc);
    ge::DataType qDtype = qInputDesc->GetDataType();
    ge::DataType gDtype = gInputDesc->GetDataType();

    int dTQ = (qDtype == ge::DT_BF16) ? CHUNK_BWD_DV_LOCAL_TPL_BF16 : CHUNK_BWD_DV_LOCAL_TPL_FP16;
    int dTG = (gDtype == ge::DT_FLOAT) ? CHUNK_BWD_DV_LOCAL_TPL_FP32 : dTQ;
    int v = static_cast<int>(tiling->v);
    context->SetTilingKey(GET_TPL_TILING_KEY(strategyKey, dTQ, dTG, v));
    OP_LOGD(context->GetNodeName(), "A5 tilingKey: %d", context->GetTilingKey());
    PrintTilingData(context, *tiling);

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAic());
    int64_t usedCoreNum = std::min(tiling->chunkNumForT * tiling->b, coreNum);
    context->SetBlockDim(usedCoreNum);

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint32_t userWorkspaceSize =
        QKV_DTYPE_SIZE * usedCoreNum * tiling->headBufNum * tiling->chunkSize * tiling->chunkSize;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize + userWorkspaceSize;
    context->SetScheduleMode(1);
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkBwdDvLocal A5 end.");
    return true;
}

void ChunkBwdDvLocalTilingA5::PrintTilingData(gert::TilingContext *context,
                                              const ChunkBwdDvLocalTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print ChunkBwdDvLocal A5 tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== b: %ld", tiling.b);
    OP_LOGD(nodeName, "=== hQk: %ld", tiling.hQk);
    OP_LOGD(nodeName, "=== hDo: %ld", tiling.hDo);
    OP_LOGD(nodeName, "=== hRatio: %ld", tiling.hRatio);
    OP_LOGD(nodeName, "=== headBufNum: %ld", tiling.headBufNum);
    OP_LOGD(nodeName, "=== t: %ld", tiling.t);
    OP_LOGD(nodeName, "=== k: %ld", tiling.k);
    OP_LOGD(nodeName, "=== v: %ld", tiling.v);
    OP_LOGD(nodeName, "=== chunkNumForT: %ld", tiling.chunkNumForT);
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling.chunkSize);
    OP_LOGD(nodeName, "=== scale: %f", tiling.scale);
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print ChunkBwdDvLocal A5 tiling data end <<<<<<<<<<<<<<<<");
}

} // namespace optiling
