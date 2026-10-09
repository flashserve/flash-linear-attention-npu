/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file pre_process_bwd_kernel_merged_tiling.cpp
 * \brief
 */

#include "pre_process_bwd_kernel_merged_tiling_processor.h"
#include "pre_process_bwd_kernel_merged_tiling.h"
#include "err/ops_err.h"

using namespace CP;

namespace optiling {
namespace {
static void PreProcessBwdKernelMergedTilingDataPrint(gert::TilingContext *context,
                                                    const PreProcessBwdKernelMergedTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "End Run PreProcessBwdKernelMerged Tiling");
    OP_LOGD(nodeName, "B is %lu.", tiling.B);
    OP_LOGD(nodeName, "T is %lu.", tiling.T);
    OP_LOGD(nodeName, "Hk is %lu.", tiling.Hk);
    OP_LOGD(nodeName, "Hv is %lu.", tiling.Hv);
    OP_LOGD(nodeName, "K is %lu.", tiling.K);
    OP_LOGD(nodeName, "V is %lu.", tiling.V);
    OP_LOGD(nodeName, "chunkSize is %lu.", tiling.chunkSize);
    OP_LOGD(nodeName, "chunkNum is %lu.", tiling.chunkNum);
    OP_LOGD(nodeName, "blockSize is %lu.", tiling.blockSize);
    OP_LOGD(nodeName, "tileV is %lu, tileK is %lu, tileNum is %lu.", tiling.tileV, tiling.tileK, tiling.tileNum);
    OP_LOGD(nodeName, "kGroupNum is %lu.", tiling.kGroupNum);
    OP_LOGD(nodeName, "isVarLen is %lu, useGateG is %lu, useGateGk is %lu.", tiling.isVarLen, tiling.useGateG,
            tiling.useGateGk);
    OP_LOGD(nodeName, "splitMode is %lu, groupHeads is %lu.", tiling.splitMode, tiling.groupHeads);
    OP_LOGD(nodeName, "slotNum is %lu, slotBytes is %lu.", tiling.slotNum, tiling.slotBytes);
    OP_LOGD(nodeName, "totalWsBytes is %lu.", tiling.totalWsBytes);
    OP_LOGD(nodeName, "usedCoreNum is %lu.", tiling.usedCoreNum);
    OP_LOGD(nodeName, "isScale is %lu, scale is %f.", tiling.isScale, tiling.scale);
}
} // namespace

ASCENDC_EXTERN_C ge::graphStatus Tiling4PreProcessBwdKernelMerged(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4PreProcessBwdKernelMerged start.");
    PreProcessBwdKernelMergedTilingData *tiling = context->GetTilingData<PreProcessBwdKernelMergedTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context->GetNodeName(), "attrs is nullptr."), return ge::GRAPH_FAILED);
    const double *scalePtr = attrs->GetAttrPointer<double>(CDHP_ATTR_SCALE_IDX);
    const uint32_t *chunkSizePtr = attrs->GetAttrPointer<uint32_t>(CDHP_ATTR_CHUNK_SIZE_IDX);

    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null!"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    // 1 AIC : 2 AIV（A5 专用）：一个 block 的两个 AIV 各承包一个 head 的整条逆序链，AIV 侧墙时间
    // 近似减半（实测 gva_cp2 34.6 → 17.5 ms）。A5 用 0x4 peer 寻址。
    //
    // A2/A3 用 1:1：910B 的 0x2 是"AIC:2*AIV 集合同步"，三次接 1:2 都在 smoke 用例上挂死
    // （共用 id 死锁 → per-AIV 段在 kda 挂 → P 常驻后再试仍在 gva 挂），详见 design.md §17/§21。
    const uint32_t aivPerBlock =
        (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) ? 2U : 1U;

    const auto gInputDesc = context->GetOptionalInputDesc(CDHP_INPUT_G_IDX);
    const auto gkInputDesc = context->GetOptionalInputDesc(CDHP_INPUT_GK_IDX);
    const auto qInputDesc = context->GetInputDesc(CDHP_INPUT_Q_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qInputDesc);

    const gert::StorageShape *gShapePtr = context->GetOptionalInputShape(CDHP_INPUT_G_IDX);
    const gert::StorageShape *cuSeqlensShapePtr = context->GetOptionalInputShape(CDHP_INPUT_CU_SEQLENS_IDX);

    PreProcessBwdKernelMergedTilingContext ctx{
        context->GetNodeName(),
        context->GetInputShape(CDHP_INPUT_Q_IDX),
        context->GetInputShape(CDHP_INPUT_K_IDX),
        context->GetInputShape(CDHP_INPUT_W_IDX),
        context->GetInputShape(CDHP_INPUT_DO_IDX),
        context->GetInputShape(CDHP_INPUT_DV_IDX),
        gShapePtr,
        context->GetOptionalInputShape(CDHP_INPUT_GK_IDX),
        cuSeqlensShapePtr,
        qInputDesc->GetDataType(),
        gInputDesc != nullptr ? gInputDesc->GetDataType() : ge::DT_FLOAT,
        gkInputDesc != nullptr ? gkInputDesc->GetDataType() : ge::DT_FLOAT,
        gShapePtr != nullptr,
        context->GetOptionalInputShape(CDHP_INPUT_GK_IDX) != nullptr,
        scalePtr != nullptr,
        scalePtr != nullptr ? *scalePtr : 1.0,
        chunkSizePtr != nullptr ? static_cast<int32_t>(*chunkSizePtr)
                                : static_cast<int32_t>(PRE_PROCESS_BWD_KERNEL_MERGED_CHUNK_SIZE),
        static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic()),
        aivPerBlock,
        static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize()),
    };

    PreProcessBwdKernelMergedTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "tiling process failed"),
                return ge::GRAPH_FAILED);

    // TPL tilingKey = 模板参数组合（dtype + gate 模式）：kernel 侧由该 key 选中模板实例，不走 TILING_KEY_IS 分支
    const int dTQ = (qInputDesc->GetDataType() == ge::DT_BF16) ? PRE_PROCESS_BWD_KERNEL_MERGED_TPL_BF16
                                                              : PRE_PROCESS_BWD_KERNEL_MERGED_TPL_FP16;
    int dTG = dTQ;
    if (processor.GetGateMode() == PRE_PROCESS_BWD_KERNEL_MERGED_GATE_MODE_SCALAR &&
        gInputDesc != nullptr && gInputDesc->GetDataType() == ge::DT_FLOAT) {
        dTG = PRE_PROCESS_BWD_KERNEL_MERGED_TPL_FP32;
    }
    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(static_cast<uint64_t>(dTQ), static_cast<uint64_t>(dTG),
                           static_cast<uint64_t>(processor.GetGateMode()));
    context->SetTilingKey(tilingKey);
    OP_LOGD(context->GetNodeName(), "tilingKey is %lu (dTQ=%d, dTG=%d, gateMode=%u).", tilingKey, dTQ, dTG,
            processor.GetGateMode());
    context->SetBlockDim(processor.GetBlockDim());
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = processor.GetWorkspaceSize();

    PreProcessBwdKernelMergedTilingDataPrint(context, *tiling);
    OP_LOGD(context->GetNodeName(), "Tiling4PreProcessBwdKernelMerged end.");
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepare4PreProcessBwdKernelMerged(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PreProcessBwdKernelMerged)
    .Tiling(Tiling4PreProcessBwdKernelMerged)
    .TilingParse<PreProcessBwdKernelMergedCompileInfo>(TilingPrepare4PreProcessBwdKernelMerged);

} // namespace optiling
