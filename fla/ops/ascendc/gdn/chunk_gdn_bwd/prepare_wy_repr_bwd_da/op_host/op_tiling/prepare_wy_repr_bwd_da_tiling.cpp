/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_da_tiling.cpp
 * \brief
 */

#include "prepare_wy_repr_bwd_da_tiling.h"
#include "arch35/prepare_wy_repr_bwd_da_tiling_a5.h"
#include <register/op_impl_registry.h>
#include "platform/soc_spec.h"
#include "platform/platform_ascendc.h"

namespace optiling {

namespace {

int64_t ToPrepareWyReprBwdDaDtype(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return PREPARE_WY_REPR_BWD_DA_DTYPE_BF16;
    }
    if (dtype == ge::DT_FLOAT) {
        return PREPARE_WY_REPR_BWD_DA_DTYPE_FP32;
    }
    return PREPARE_WY_REPR_BWD_DA_DTYPE_FP16;
}

void PrepareWyReprBwdDaTilingDataPrint(gert::TilingContext *context, const PrepareWyReprBwdDaTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print PrepareWyReprBwdDa tiling data <<<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== B: %ld", tiling.B);
    OP_LOGD(nodeName, "=== HV: %ld", tiling.HV);
    OP_LOGD(nodeName, "=== HK: %ld", tiling.HK);
    OP_LOGD(nodeName, "=== T: %ld", tiling.T);
    OP_LOGD(nodeName, "=== K: %ld", tiling.K);
    OP_LOGD(nodeName, "=== V: %ld", tiling.V);
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling.chunkSize);
    OP_LOGD(nodeName, "=== chunkNum: %ld", tiling.chunkNum);
    OP_LOGD(nodeName, "=== rowNumKBetaG: %ld", tiling.rowNumKBetaG);
    OP_LOGD(nodeName, "=== rowNumVBeta: %ld", tiling.rowNumVBeta);
    OP_LOGD(nodeName, "=== rowNumMDuDw: %ld", tiling.rowNumMDuDw);
    OP_LOGD(nodeName, "=== rowNumG: %ld", tiling.rowNumG);
    OP_LOGD(nodeName, "=== isVariable: %ld", tiling.isVariable);
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print PrepareWyReprBwdDa tiling data end <<<<<<<<<<<<<<<<<");
}

} // namespace

ge::graphStatus Tiling4PrepareWyReprBwdDa(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4PrepareWyReprBwdDa start.");
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        PrepareWyReprBwdDaTilingA5 prepareWyReprBwdDaTilingA5;
        OP_CHECK_IF(!prepareWyReprBwdDaTilingA5.SetTiling(context),
                    OP_LOGE(context->GetNodeName(), "SetTiling failed."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    PrepareWyReprBwdDaTilingData *tiling = context->GetTilingData<PrepareWyReprBwdDaTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);

    auto kDesc = context->GetInputDesc(INPUT_K_IDX);
    auto betaDesc = context->GetInputDesc(INPUT_BETA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDesc);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    auto cuSeqlensTensor = context->GetOptionalInputTensor(INPUT_SEQLENS_IDX);
    auto chunkIndicesTensor = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
    const int64_t *cuSeqlensData = cuSeqlensTensor == nullptr ? nullptr : cuSeqlensTensor->GetData<int64_t>();
    const int64_t *chunkIndicesData =
        chunkIndicesTensor == nullptr ? nullptr : chunkIndicesTensor->GetData<int64_t>();

    PrepareWyReprBwdDaTilingContext ctx{
        context->GetNodeName(),
        context->GetRequiredInputShape(INPUT_K_IDX),
        context->GetRequiredInputShape(INPUT_V_IDX),
        context->GetRequiredInputShape(INPUT_BETA_IDX),
        context->GetRequiredInputShape(INPUT_A_IDX),
        context->GetRequiredInputShape(INPUT_DW_IDX),
        context->GetRequiredInputShape(INPUT_DU_IDX),
        context->GetRequiredInputShape(INPUT_G_IDX),
        context->GetOptionalInputShape(INPUT_SEQLENS_IDX),
        context->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX),
        cuSeqlensData,
        chunkIndicesData,
        static_cast<int64_t>(*(attrPtr->GetAttrPointer<int32_t>(ATTR_CHUNK_SIZE_IDX))),
        ToPrepareWyReprBwdDaDtype(kDesc->GetDataType()),
        ToPrepareWyReprBwdDaDtype(betaDesc->GetDataType()),
        ubSize,
        static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic()),
        static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize()),
    };

    PrepareWyReprBwdDaTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    context->SetTilingKey(processor.GetTilingKey());
    context->SetBlockDim(processor.GetBlockDim());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);

    OP_LOGD(context->GetNodeName(), "tilingKey: %d", context->GetTilingKey());
    PrepareWyReprBwdDaTilingDataPrint(context, *tiling);
    OP_LOGD(context->GetNodeName(), "Tiling4PrepareWyReprBwdDa end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForPrepareWyReprBwdDa(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PrepareWyReprBwdDa)
    .Tiling(Tiling4PrepareWyReprBwdDa)
    .TilingParse<PrepareWyReprBwdDaCompileInfo>(TilingPrepareForPrepareWyReprBwdDa);

} // namespace optiling
