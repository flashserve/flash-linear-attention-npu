/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_fwd_h_tiling.cpp
 * \brief
 */

#include "chunk_fwd_h_tiling.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "chunk_fwd_h_tiling_processor.h"

namespace optiling {

// Maps a ge::DataType to the {fp16:0, bf16:1, fp32:2} convention shared with the kernel.
static int64_t ChunkFwdHDtypeToEnum(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return CHUNK_FWD_H_DTYPE_BF16;
    }
    if (dtype == ge::DT_FLOAT16) {
        return CHUNK_FWD_H_DTYPE_FP16;
    }
    return CHUNK_FWD_H_DTYPE_FP32;
}
static constexpr size_t INPUT_K_IDX = 0;
static constexpr size_t INPUT_W_IDX = 1;
static constexpr size_t INPUT_U_IDX = 2;
static constexpr size_t INPUT_G_IDX = 3;
static constexpr size_t INPUT_GK_IDX = 4;
static constexpr size_t INPUT_INITIAL_STATE_IDX = 5;
static constexpr size_t INPUT_SEQLENS_IDX = 6;
static constexpr size_t INPUT_CHUNK_INDICES_IDX = 7;
static constexpr size_t OUTPUT_FINAL_STATE_IDX = 2;

static constexpr size_t ATTR_STORE_FINAL_STATE_IDX = 0;
static constexpr size_t ATTR_CHUNK_SIZE_IDX = 1;
static constexpr size_t ATTR_SAVE_NEW_VALUE_IDX = 2;
static constexpr size_t ATTR_USE_EXP2_IDX = 3;
static constexpr size_t ATTR_STATE_V_FIRST_IDX = 4;
static constexpr size_t ATTR_LOGICAL_BATCH_IDX = 5;
static constexpr size_t ATTR_LOGICAL_SEQLEN_IDX = 6;
static constexpr size_t ATTR_LOGICAL_K_HEADS_IDX = 7;
static constexpr size_t ATTR_LOGICAL_V_HEADS_IDX = 8;
static constexpr size_t ATTR_LOGICAL_K_DIM_IDX = 9;
static constexpr size_t ATTR_LOGICAL_V_DIM_IDX = 10;

static constexpr uint32_t TILING_KEY_V128 = 1;
static constexpr int64_t V_DIM_128 = 128;
static constexpr int64_t K_DIM_128 = 128;
static constexpr int64_t CHUNK_SIZE_64 = 64;

static void ChunkFwdHTilingDataPrint(gert::TilingContext *context, ChunkFwdHTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print ChunkFwdH tiling data <<<<<<<<<<<<<<<<");
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
    OP_LOGD(nodeName, "=== useExp2: %d", tiling.get_useExp2());
    OP_LOGD(nodeName, "=== stateVFirst: %d", tiling.get_stateVFirst());
    OP_LOGD(nodeName, "=== dataType: %ld", tiling.get_dataType());
    OP_LOGD(nodeName, "=== isVariedLen: %ld", tiling.get_isVariedLen());
    OP_LOGD(nodeName, "=== shapeBatch: %ld", tiling.get_shapeBatch());
    OP_LOGD(nodeName, "=== tokenBatch: %ld", tiling.get_tokenBatch());
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print ChunkFwdH tiling data end <<<<<<<<<<<<<<<<");
}

ge::graphStatus Tiling4ChunkFwdH(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkFwdH start.");
    ChunkFwdHTilingData tiling;

    auto cuSeqlensTensor = context->GetOptionalInputTensor(INPUT_SEQLENS_IDX);
    auto initialStateTensor = context->GetOptionalInputTensor(INPUT_INITIAL_STATE_IDX);
    bool useInitialState = initialStateTensor != nullptr;
    auto gTensor = context->GetOptionalInputTensor(INPUT_G_IDX);
    auto gkTensor = context->GetOptionalInputTensor(INPUT_GK_IDX);
    bool useG = gTensor != nullptr;
    bool useGk = gkTensor != nullptr;
    OP_CHECK_IF(useG == useGk,
                OP_LOGE(context->GetNodeName(), "Exactly one of g and gk must be provided."),
                return ge::GRAPH_FAILED);
    auto gateTensor = useGk ? gkTensor : gTensor;

    auto attrPtr = context->GetAttrs();
    bool storeFinalState = *(attrPtr->GetAttrPointer<bool>(ATTR_STORE_FINAL_STATE_IDX));
    int64_t chunkSize = *(attrPtr->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX));
    bool saveNewValue = *(attrPtr->GetAttrPointer<bool>(ATTR_SAVE_NEW_VALUE_IDX));
    bool useExp2 = *(attrPtr->GetAttrPointer<bool>(ATTR_USE_EXP2_IDX));
    bool stateVFirst = *(attrPtr->GetAttrPointer<bool>(ATTR_STATE_V_FIRST_IDX));
    int64_t logicalBatch = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_BATCH_IDX));
    int64_t logicalSeqlen = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_SEQLEN_IDX));
    int64_t logicalKHeads = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_K_HEADS_IDX));
    int64_t logicalVHeads = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_V_HEADS_IDX));
    int64_t logicalKDim = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_K_DIM_IDX));
    int64_t logicalVDim = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_V_DIM_IDX));

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    ChunkFwdHTilingContext tilingCtx{};
    tilingCtx.seqlen = logicalSeqlen;
    tilingCtx.kNumHead = logicalKHeads;
    tilingCtx.kHeadDim = logicalKDim;
    tilingCtx.vNumHead = logicalVHeads;
    tilingCtx.vHeadDim = logicalVDim;
    tilingCtx.shapeBatchDim = logicalBatch;
    tilingCtx.hasCuSeqlens = cuSeqlensTensor != nullptr;
    tilingCtx.cuSeqlensDim0 =
        cuSeqlensTensor != nullptr ? cuSeqlensTensor->GetStorageShape().GetDim(0) : 0;
    tilingCtx.dataType = ChunkFwdHDtypeToEnum(context->GetInputTensor(0)->GetDataType());
    tilingCtx.gDataType = ChunkFwdHDtypeToEnum(gateTensor->GetDataType());
    tilingCtx.useInitialState = useInitialState;
    tilingCtx.stateDataType = useInitialState
                                  ? ChunkFwdHDtypeToEnum(initialStateTensor->GetDataType())
                                  : (storeFinalState
                                         ? ChunkFwdHDtypeToEnum(
                                               context->GetOutputDesc(OUTPUT_FINAL_STATE_IDX)->GetDataType())
                                         : CHUNK_FWD_H_DTYPE_FP32);
    tilingCtx.useG = useG;
    tilingCtx.useGk = useGk;
    tilingCtx.useExp2 = useExp2;
    tilingCtx.stateVFirst = stateVFirst;
    tilingCtx.storeFinalState = storeFinalState;
    tilingCtx.chunkSize = chunkSize;
    tilingCtx.aicCoreNum = ascendcPlatform.GetCoreNumAic();
    tilingCtx.libApiWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    if (logicalBatch <= 0 || logicalSeqlen <= 0 || logicalKHeads <= 0 || logicalVHeads <= 0) {
        OP_LOGE(context->GetNodeName(),
                "Logical batch, sequence length and head counts must be positive, but got "
                "batch=%ld, seqlen=%ld, HK=%ld, HV=%ld.",
                logicalBatch, logicalSeqlen, logicalKHeads, logicalVHeads);
        return ge::GRAPH_FAILED;
    }
    if (logicalKDim != K_DIM_128 || logicalVDim != V_DIM_128 || chunkSize != CHUNK_SIZE_64) {
        OP_LOGE(context->GetNodeName(),
                "FwdH requires K=128, V=128 and chunk_size=64, but got K=%ld, V=%ld, chunk_size=%ld.",
                logicalKDim, logicalVDim, chunkSize);
        return ge::GRAPH_FAILED;
    }
    if (!saveNewValue) {
        OP_LOGE(context->GetNodeName(), "save_new_value must be true.");
        return ge::GRAPH_FAILED;
    }
    if ((useG && logicalVHeads % logicalKHeads != 0) || (useGk && logicalKHeads != logicalVHeads)) {
        OP_LOGE(context->GetNodeName(),
                "g-only requires HV %% HK == 0; gk-only requires k/ kg head count equal HV. "
                "Current mode useG=%d, useGk=%d, HK=%ld, HV=%ld.",
                useG, useGk, logicalKHeads, logicalVHeads);
        return ge::GRAPH_FAILED;
    }

    ::ChunkFwdHPlainTilingData plainTiling{};
    uint32_t blockDim = 0;
    size_t workspaceSize = 0;
    ChunkFwdHTilingProcessor processor(tilingCtx);
    processor.Process(plainTiling, blockDim, workspaceSize);

    context->SetTilingKey(TILING_KEY_V128);
    OP_LOGD(context->GetNodeName(), "tilingKey: %u", TILING_KEY_V128);

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
    tiling.set_useExp2(plainTiling.useExp2);
    tiling.set_stateVFirst(plainTiling.stateVFirst);
    tiling.set_vWorkspaceOffset(plainTiling.vWorkspaceOffset);
    tiling.set_vUpdateWorkspaceOffset(plainTiling.vUpdateWorkspaceOffset);
    tiling.set_kDecayWorkspaceOffset(plainTiling.kDecayWorkspaceOffset);
    tiling.set_hWorkspaceOffset(plainTiling.hWorkspaceOffset);
    tiling.set_numSeqWorkspaceOffset(plainTiling.numSeqWorkspaceOffset);
    tiling.set_numChunksWorkspaceOffset(plainTiling.numChunksWorkspaceOffset);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    ChunkFwdHTilingDataPrint(context, tiling);
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkFwdH end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkFwdH(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkFwdH)
    .Tiling(Tiling4ChunkFwdH)
    .TilingParse<ChunkFwdHCompileInfo>(TilingPrepareForChunkFwdH);

} // namespace optiling
