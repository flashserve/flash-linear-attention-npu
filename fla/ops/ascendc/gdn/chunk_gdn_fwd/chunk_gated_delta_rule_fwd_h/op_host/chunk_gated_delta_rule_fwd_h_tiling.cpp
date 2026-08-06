/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_fwd_h_tiling.cpp
 * \brief
 */

#include "chunk_gated_delta_rule_fwd_h_tiling.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "chunk_gated_delta_rule_fwd_h_tiling_processor.h"

namespace optiling {

// Maps a ge::DataType to the {fp16:0, bf16:1, fp32:2} convention shared with the kernel.
static int64_t GdnFwdHDtypeToEnum(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return GDN_FWD_H_DTYPE_BF16;
    }
    if (dtype == ge::DT_FLOAT16) {
        return GDN_FWD_H_DTYPE_FP16;
    }
    return GDN_FWD_H_DTYPE_FP32;
}
static constexpr size_t INPUT_K_IDX = 0;
static constexpr size_t INPUT_W_IDX = 1;
static constexpr size_t INPUT_U_IDX = 2;
static constexpr size_t INPUT_G_IDX = 3;
static constexpr size_t INPUT_GK_IDX = 4;
static constexpr size_t INPUT_INITIAL_STATE_IDX = 5;
static constexpr size_t INPUT_SEQLENS_IDX = 6;
static constexpr size_t INPUT_CHUNK_INDICES_IDX = 7;

static constexpr size_t ATTR_STORE_FINAL_STATE_IDX = 0;
static constexpr size_t ATTR_CHUNK_SIZE_IDX = 1;
static constexpr size_t ATTR_LOGICAL_BATCH_IDX = 2;
static constexpr size_t ATTR_LOGICAL_SEQLEN_IDX = 3;
static constexpr size_t ATTR_LOGICAL_K_HEADS_IDX = 4;
static constexpr size_t ATTR_LOGICAL_V_HEADS_IDX = 5;
static constexpr size_t ATTR_LOGICAL_K_DIM_IDX = 6;
static constexpr size_t ATTR_LOGICAL_V_DIM_IDX = 7;

static constexpr uint32_t TILING_KEY_V128 = 1;
static constexpr uint32_t TILING_KEY_V256 = 2;
static constexpr int64_t V_DIM_128 = 128;
static constexpr int64_t V_DIM_256 = 256;

static void ChunkGatedDeltaRuleFwdHTilingDataPrint(gert::TilingContext *context, ChunkGatedDeltaRuleFwdHTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print ChunkGatedDeltaRuleFwdH tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== batch: %ld", tiling.get_batch());
    OP_LOGD(nodeName, "=== seqlen: %ld", tiling.get_seqlen());
    OP_LOGD(nodeName, "=== kNumHead: %ld", tiling.get_kNumHead());
    OP_LOGD(nodeName, "=== vNumHead: %ld", tiling.get_vNumHead());
    OP_LOGD(nodeName, "=== kHeadDim: %ld", tiling.get_kHeadDim());
    OP_LOGD(nodeName, "=== vHeadDim: %ld", tiling.get_vHeadDim());
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling.get_chunkSize());
    OP_LOGD(nodeName, "=== useInitialState: %ld", tiling.get_useInitialState());
    OP_LOGD(nodeName, "=== storeFinalState: %ld", tiling.get_storeFinalState());
    OP_LOGD(nodeName, "=== useG: %d", tiling.get_useG());
    OP_LOGD(nodeName, "=== useGk: %d", tiling.get_useGk());
    OP_LOGD(nodeName, "=== dataType: %ld", tiling.get_dataType());
    OP_LOGD(nodeName, "=== isVariedLen: %ld", tiling.get_isVariedLen());
    OP_LOGD(nodeName, "=== shapeBatch: %ld", tiling.get_shapeBatch());
    OP_LOGD(nodeName, "=== tokenBatch: %f", tiling.get_tokenBatch());
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print ChunkGatedDeltaRuleFwdH tiling data end <<<<<<<<<<<<<<<<");
}

ge::graphStatus Tiling4ChunkGatedDeltaRuleFwdH(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkGatedDeltaRuleFwdH start.");
    ChunkGatedDeltaRuleFwdHTilingData tiling;

    auto cuSeqlensTensor = context->GetOptionalInputTensor(INPUT_SEQLENS_IDX);
    auto initialStateTensor = context->GetOptionalInputTensor(INPUT_INITIAL_STATE_IDX);
    bool useInitialState = initialStateTensor != nullptr;
    auto gTensor = context->GetOptionalInputTensor(INPUT_G_IDX);
    auto gkTensor = context->GetOptionalInputTensor(INPUT_GK_IDX);
    bool useG = gTensor != nullptr;
    bool useGk = gkTensor != nullptr;
    OP_CHECK_IF(gTensor == nullptr && gkTensor == nullptr,
                OP_LOGE(context->GetNodeName(), "Either g or gk must be provided."),
                return ge::GRAPH_FAILED);
    auto gateTensor = useGk ? gkTensor : gTensor;

    auto attrPtr = context->GetAttrs();
    bool storeFinalState = *(attrPtr->GetAttrPointer<bool>(ATTR_STORE_FINAL_STATE_IDX));
    int64_t chunkSize = *(attrPtr->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX));
    int64_t logicalBatch = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_BATCH_IDX));
    int64_t logicalSeqlen = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_SEQLEN_IDX));
    int64_t logicalKHeads = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_K_HEADS_IDX));
    int64_t logicalVHeads = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_V_HEADS_IDX));
    int64_t logicalKDim = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_K_DIM_IDX));
    int64_t logicalVDim = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_V_DIM_IDX));

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    ChunkGatedDeltaRuleFwdHTilingContext tilingCtx{};
    tilingCtx.seqlen = logicalSeqlen;
    tilingCtx.kNumHead = logicalKHeads;
    tilingCtx.kHeadDim = logicalKDim;
    tilingCtx.vNumHead = logicalVHeads;
    tilingCtx.vHeadDim = logicalVDim;
    tilingCtx.shapeBatchDim = logicalBatch;
    tilingCtx.hasCuSeqlens = cuSeqlensTensor != nullptr;
    tilingCtx.cuSeqlensDim0 =
        cuSeqlensTensor != nullptr ? cuSeqlensTensor->GetStorageShape().GetDim(0) : 0;
    tilingCtx.dataType = GdnFwdHDtypeToEnum(context->GetInputTensor(0)->GetDataType());
    tilingCtx.gDataType = GdnFwdHDtypeToEnum(gateTensor->GetDataType());
    tilingCtx.useInitialState = useInitialState;
    tilingCtx.stateDataType =
        useInitialState ? GdnFwdHDtypeToEnum(initialStateTensor->GetDataType()) : GDN_FWD_H_DTYPE_FP32;
    tilingCtx.useG = useG;
    tilingCtx.storeFinalState = storeFinalState;
    tilingCtx.chunkSize = chunkSize;
    tilingCtx.useGk = useGk;
    tilingCtx.aicCoreNum = ascendcPlatform.GetCoreNumAic();
    tilingCtx.libApiWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    if (logicalBatch <= 0 || logicalSeqlen <= 0 || logicalKHeads <= 0 ||
        logicalVHeads <= 0 || logicalKDim <= 0 || logicalVDim <= 0 ||
        tilingCtx.vNumHead % tilingCtx.kNumHead != 0) {
        OP_LOGE(context->GetNodeName(),
                "Check logical shape failed: batch=%ld, seqlen=%ld, kDim=%ld, vDim=%ld; "
                "vNumHead (%ld) must be divisible by kNumHead (%ld).",
                logicalBatch, logicalSeqlen, logicalKDim, logicalVDim,
                tilingCtx.vNumHead, tilingCtx.kNumHead);
        return ge::GRAPH_FAILED;
    }
    if (tilingCtx.vHeadDim > V_DIM_256) {
        OP_LOGE(context->GetNodeName(), "Check u shape failed, vHeadDim should be <= %ld, but get %ld.",
                V_DIM_256, tilingCtx.vHeadDim);
        return ge::GRAPH_FAILED;
    }

    ::ChunkGatedDeltaRuleFwdHTilingData plainTiling{};
    uint32_t blockDim = 0;
    size_t workspaceSize = 0;
    ChunkGatedDeltaRuleFwdHTilingProcessor processor(tilingCtx);
    processor.Process(plainTiling, blockDim, workspaceSize);

    uint32_t tilingKey = plainTiling.vHeadDim > V_DIM_128 ? TILING_KEY_V256 : TILING_KEY_V128;
    context->SetTilingKey(tilingKey);
    OP_LOGD(context->GetNodeName(), "tilingKey: %u (vHeadDim=%ld)", tilingKey, plainTiling.vHeadDim);

    context->SetBlockDim(blockDim);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize;

    tiling.set_batch(plainTiling.batch);
    tiling.set_seqlen(plainTiling.seqlen);
    tiling.set_kNumHead(plainTiling.kNumHead);
    tiling.set_vNumHead(plainTiling.vNumHead);
    tiling.set_kHeadDim(plainTiling.kHeadDim);
    tiling.set_vHeadDim(plainTiling.vHeadDim);
    tiling.set_chunkSize(plainTiling.chunkSize);
    tiling.set_useInitialState(plainTiling.useInitialState);
    tiling.set_storeFinalState(plainTiling.storeFinalState);
    tiling.set_dataType(plainTiling.dataType);
    tiling.set_stateDataType(plainTiling.stateDataType);
    tiling.set_gDataType(plainTiling.gDataType);
    tiling.set_isVariedLen(plainTiling.isVariedLen);
    tiling.set_shapeBatch(plainTiling.shapeBatch);
    tiling.set_tokenBatch(plainTiling.tokenBatch);
    tiling.set_useG(plainTiling.useG);
    tiling.set_useGk(plainTiling.useGk);
    tiling.set_vWorkspaceOffset(plainTiling.vWorkspaceOffset);
    tiling.set_vUpdateWorkspaceOffset(plainTiling.vUpdateWorkspaceOffset);
    tiling.set_kDecayWorkspaceOffset(plainTiling.kDecayWorkspaceOffset);
    tiling.set_hWorkspaceOffset(plainTiling.hWorkspaceOffset);
    tiling.set_numSeqWorkspaceOffset(plainTiling.numSeqWorkspaceOffset);
    tiling.set_numChunksWorkspaceOffset(plainTiling.numChunksWorkspaceOffset);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    ChunkGatedDeltaRuleFwdHTilingDataPrint(context, tiling);
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkGatedDeltaRuleFwdH end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkGatedDeltaRuleFwdH(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleFwdH)
    .Tiling(Tiling4ChunkGatedDeltaRuleFwdH)
    .TilingParse<ChunkGatedDeltaRuleFwdHCompileInfo>(TilingPrepareForChunkGatedDeltaRuleFwdH);

} // namespace optiling
