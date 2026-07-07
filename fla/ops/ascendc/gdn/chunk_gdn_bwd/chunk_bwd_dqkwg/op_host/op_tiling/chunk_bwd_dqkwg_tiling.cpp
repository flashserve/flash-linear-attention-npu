/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dqkwg_tiling.cpp
 * \brief
 */

#include "chunk_bwd_dqkwg_tiling_processor.h"
#include "../../op_kernel/chunk_bwd_dqkwg_struct.h"

using namespace GDN;

namespace optiling {

struct ChunkBwdDqkwgCompileInfo {};

static void ChunkBwdDqkwgTilingDataPrint(gert::TilingContext *context, const ChunkBwdDqkwgTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print ChunkBwdDqkwg tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== B: %ld", tiling.B);
    OP_LOGD(nodeName, "=== HV: %ld", tiling.HV);
    OP_LOGD(nodeName, "=== HK: %ld", tiling.HK);
    OP_LOGD(nodeName, "=== T: %ld", tiling.T);
    OP_LOGD(nodeName, "=== K: %ld", tiling.K);
    OP_LOGD(nodeName, "=== V: %ld", tiling.V);
    OP_LOGD(nodeName, "=== BT: %ld", tiling.BT);
    OP_LOGD(nodeName, "=== numChunks: %ld", tiling.numChunks);
    OP_LOGD(nodeName, "=== scale: %f", tiling.scale);
    OP_LOGD(nodeName, "=== mul0RowNum: %ld", tiling.mul0RowNum);
    OP_LOGD(nodeName, "=== isVarLen: %ld", tiling.isVarLen);
    OP_LOGD(nodeName, "=== needReduce: %ld", tiling.needReduce);
    OP_LOGD(nodeName, "=== wsDqkvReduceOffset: %ld", tiling.wsDqkvReduceOffset);
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print ChunkBwdDqkwg tiling data end <<<<<<<<<<<<<<<<");
}

ASCENDC_EXTERN_C ge::graphStatus TilingChunkBwdDqkwg(gert::TilingContext* context) {
    OP_LOGD(context->GetNodeName(), "TilingChunkBwdDqkwg start.");
    ChunkBwdDqkwgTilingData *tiling = context->GetTilingData<ChunkBwdDqkwgTilingData>();

    auto attr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);

    const float* scalePtr = attr->GetAttrPointer<float>(CHUNK_BWD_DQKWG_ATTR_SCALE_IDX);
    OP_CHECK_IF(scalePtr == nullptr,
                OP_LOGE(context, "scale should not be nullptr."),
                return ge::GRAPH_FAILED);
    float scale = *scalePtr;

    const int32_t* chunkSizePtr = attr->GetAttrPointer<int32_t>(CHUNK_BWD_DQKWG_ATTR_CHUNK_SIZE_IDX);
    int32_t chunkSize = chunkSizePtr != nullptr ? *chunkSizePtr : 64;

    ChunkBwdDqkwgTilingContext ctx{
        context->GetNodeName(),
        context->GetOptionalInputShape(CHUNK_BWD_DQKWG_INPUT_Q_IDX),
        context->GetOptionalInputShape(CHUNK_BWD_DQKWG_INPUT_K_IDX),
        context->GetOptionalInputShape(CHUNK_BWD_DQKWG_INPUT_V_IDX),
        context->GetOptionalInputShape(CHUNK_BWD_DQKWG_INPUT_G_IDX),
        context->GetOptionalInputShape(CHUNK_BWD_DQKWG_INPUT_H_IDX),
        context->GetOptionalInputShape(CHUNK_BWD_DQKWG_INPUT_DO_IDX),
        context->GetOptionalInputShape(CHUNK_BWD_DQKWG_INPUT_DH_IDX),
        context->GetOptionalInputShape(CHUNK_BWD_DQKWG_INPUT_DV_IDX),
        context->GetOptionalInputShape(CHUNK_BWD_DQKWG_INPUT_CUSEQLENS_IDX),
        context->GetOptionalInputShape(CHUNK_BWD_DQKWG_INPUT_CHUNK_INDICES_IDX),
        scale,
        chunkSize,
    };

    ChunkBwdDqkwgTilingProcessor processor(ctx, *tiling);

    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    uint64_t strategyKey = processor.IsVariableLength() ? CHUNK_BWD_DQKWG_STRATEGY_VAR_LEN : CHUNK_BWD_DQKWG_STRATEGY_FIX_LEN;

    auto qInputDesc = context->GetInputDesc(CHUNK_BWD_DQKWG_INPUT_Q_IDX);
    auto gInputDesc = context->GetInputDesc(CHUNK_BWD_DQKWG_INPUT_G_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qInputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gInputDesc);
    ge::DataType qDtype = qInputDesc->GetDataType();
    ge::DataType gDtype = gInputDesc->GetDataType();

    int dTQ = (qDtype == ge::DT_BF16) ? CHUNK_BWD_DQKWG_TPL_BF16 : CHUNK_BWD_DQKWG_TPL_FP16;
    int dTG = (gDtype == ge::DT_FLOAT) ? CHUNK_BWD_DQKWG_TPL_FP32 : dTQ;

    int v = static_cast<int>(tiling->V);
    OP_LOGD(context->GetNodeName(), "V value: %d", v);

    uint64_t tilingKey = GET_TPL_TILING_KEY(strategyKey, dTQ, dTG, v);
    context->SetTilingKey(tilingKey);

    OP_LOGD(context->GetNodeName(), "tilingKey: %d", context->GetTilingKey());
    ChunkBwdDqkwgTilingDataPrint(context, *tiling);

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAic());
    int64_t usedCoreNum = std::min(tiling->B * tiling->numChunks, coreNum);
    context->SetBlockDim(usedCoreNum);

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t userWorkspaceSize = static_cast<size_t>(tiling->totalWorkspaceSize);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize + userWorkspaceSize;
    context->SetScheduleMode(1);
    OP_LOGD(context->GetNodeName(), "TilingChunkBwdDqkwg end.");
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingParseChunkBwdDqkwg(gert::TilingParseContext* context) {
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkBwdDqkwg)
    .Tiling(TilingChunkBwdDqkwg)
    .TilingParse<ChunkBwdDqkwgCompileInfo>(TilingParseChunkBwdDqkwg);

}  // namespace optiling
