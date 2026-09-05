/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "chunk_gdn_bwd_intra_tiling.h"

#include <algorithm>
#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"
#include "../../op_kernel/chunk_gdn_bwd_intra_tiling_key.h"

namespace optiling {
namespace {

uint64_t DataTypeKey(ge::DataType dtype, ge::DataType fp32Type,
                     uint64_t lowPrecisionKey, uint64_t fp32Key)
{
    return dtype == fp32Type ? fp32Key : lowPrecisionKey;
}

} // namespace

ge::graphStatus Tiling4ChunkGdnBwdIntra(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<GDN::ChunkGdnBwdIntraTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto *scale = attrs->GetAttrPointer<double>(INTRA_SCALE_ATTR_IDX);
    const auto *chunkSize = attrs->GetAttrPointer<int64_t>(INTRA_CHUNK_SIZE_ATTR_IDX);
    const auto *useExp2 = attrs->GetAttrPointer<bool>(INTRA_USE_EXP2_ATTR_IDX);
    const auto *stage = attrs->GetAttrPointer<int64_t>(INTRA_STAGE_ATTR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, useExp2);
    OP_CHECK_NULL_WITH_CONTEXT(context, stage);

    const auto *qShape = context->GetRequiredInputShape(INTRA_Q_IDX);
    const auto *vShape = context->GetRequiredInputShape(INTRA_V_IDX);
    const auto *qDesc = context->GetInputDesc(INTRA_Q_IDX);
    const auto *gDesc = context->GetInputDesc(INTRA_G_IDX);
    const auto *betaDesc = context->GetInputDesc(INTRA_BETA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, qDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDesc);

    const auto qStorage = qShape->GetStorageShape();
    const auto vStorage = vShape->GetStorageShape();
    const int64_t batch = qStorage.GetDim(0);
    const int64_t qkHeads = qStorage.GetDim(1);
    const int64_t seqlen = qStorage.GetDim(2);
    const int64_t keyDim = qStorage.GetDim(3);
    const int64_t valueHeads = vStorage.GetDim(1);
    const int64_t valueDim = vStorage.GetDim(3);
    const int64_t headRatio = valueHeads / qkHeads;
    const int64_t cg = headRatio == 3 ? 3 : 4;
    const int64_t hvSliceCount = (valueHeads + cg - 1) / cg;

    const auto *cuTensor = context->GetOptionalInputTensor(INTRA_CU_SEQLENS_IDX);
    const auto *chunkTensor = context->GetOptionalInputTensor(INTRA_CHUNK_INDICES_IDX);
    const bool isVarlen = cuTensor != nullptr;
    const int64_t chunksPerBatch = (seqlen + *chunkSize - 1) / *chunkSize;
    int64_t chunkCount = batch * chunksPerBatch;
    if (isVarlen) {
        OP_CHECK_NULL_WITH_CONTEXT(context, chunkTensor);
        const auto *chunkShape = context->GetOptionalInputShape(INTRA_CHUNK_INDICES_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context, chunkShape);
        chunkCount = chunkShape->GetStorageShape().GetShapeSize() / 2;
    }
    const int64_t workCount = chunkCount * hvSliceCount;

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicCores = platform.GetCoreNumAic();
    OP_CHECK_IF(workCount <= 0 || aicCores == 0,
                OP_LOGE(context->GetNodeName(), "work count and AIC core count must be positive."),
                return ge::GRAPH_FAILED);
    const int64_t blockDim = std::min<int64_t>(workCount, aicCores);
    const uint64_t matrixStride = Align512(
        static_cast<uint64_t>(*chunkSize) * static_cast<uint64_t>(*chunkSize) * 2U);
    const uint64_t slotCount = static_cast<uint64_t>(blockDim) * static_cast<uint64_t>(cg);
    const uint64_t sectionBytes = slotCount * matrixStride;
    const uint64_t userWorkspace = *stage == 0 ? 0 : sectionBytes * 3U;

    tiling->batch = batch;
    tiling->qkHeads = qkHeads;
    tiling->valueHeads = valueHeads;
    tiling->seqlen = seqlen;
    tiling->keyDim = keyDim;
    tiling->valueDim = valueDim;
    tiling->chunkSize = *chunkSize;
    tiling->chunksPerBatch = chunksPerBatch;
    tiling->chunkCount = chunkCount;
    tiling->headRatio = headRatio;
    tiling->cg = cg;
    tiling->hvSliceCount = hvSliceCount;
    tiling->workCount = workCount;
    tiling->blockDim = blockDim;
    tiling->isVarlen = isVarlen ? 1 : 0;
    tiling->useExp2 = *useExp2 ? 1 : 0;
    tiling->stage = *stage;
    tiling->scale = static_cast<float>(*scale);
    tiling->reserved = 0;
    tiling->matrixStrideBytes = matrixStride;
    tiling->aBgWorkspaceOffset = 0;
    tiling->aBetaWorkspaceOffset = sectionBytes;
    tiling->dWorkspaceOffset = sectionBytes * 2U;
    tiling->userWorkspaceBytes = userWorkspace;

    const uint64_t strategyKey = isVarlen ?
        CHUNK_GDN_BWD_INTRA_VARLEN : CHUNK_GDN_BWD_INTRA_FIXED;
    const uint64_t mainKey = qDesc->GetDataType() == ge::DT_BF16 ?
        CHUNK_GDN_BWD_INTRA_MAIN_BF16 : CHUNK_GDN_BWD_INTRA_MAIN_FP16;
    const uint64_t gateKey = DataTypeKey(
        gDesc->GetDataType(), ge::DT_FLOAT,
        CHUNK_GDN_BWD_INTRA_GATE_BF16, CHUNK_GDN_BWD_INTRA_GATE_FP32);
    const uint64_t betaKey = DataTypeKey(
        betaDesc->GetDataType(), ge::DT_FLOAT,
        CHUNK_GDN_BWD_INTRA_BETA_BF16, CHUNK_GDN_BWD_INTRA_BETA_FP32);
    context->SetTilingKey(GET_TPL_TILING_KEY(strategyKey, mainKey, gateKey, betaKey));
    context->SetBlockDim(blockDim);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() + userWorkspace;
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkGdnBwdIntra(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGdnBwdIntra)
    .Tiling(Tiling4ChunkGdnBwdIntra)
    .TilingParse<ChunkGdnBwdIntraCompileInfo>(TilingPrepare4ChunkGdnBwdIntra);

} // namespace optiling
