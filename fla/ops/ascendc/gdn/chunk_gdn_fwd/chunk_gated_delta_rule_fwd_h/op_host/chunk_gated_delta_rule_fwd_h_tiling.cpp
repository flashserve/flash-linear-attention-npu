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
static constexpr size_t OUTPUT_H_IDX = 0;

static constexpr size_t ATTR_STORE_FINAL_STATE_IDX = 0;
static constexpr size_t ATTR_CHUNK_SIZE_IDX = 1;
static constexpr size_t ATTR_USE_EXP2_IDX = 2;

static constexpr size_t DIM_BATCH = 0;
static constexpr size_t DIM_HEAD_NUM = 1;
static constexpr size_t DIM_SEQLEN = 2;
static constexpr size_t DIM_HEAD_DIM = 3;

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
    OP_LOGD(nodeName, "=== useExp2: %d", tiling.get_useExp2());
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

    auto kShapePtr = context->GetInputShape(INPUT_K_IDX);
    auto uShapePtr = context->GetInputShape(INPUT_U_IDX);
    OP_CHECK_IF(kShapePtr == nullptr || uShapePtr == nullptr,
                OP_LOGE(context->GetNodeName(), "k and u shapes must not be null."),
                return ge::GRAPH_FAILED);
    const gert::Shape &kShape = kShapePtr->GetOriginShape();
    const gert::Shape &uShape = uShapePtr->GetOriginShape();
    OP_CHECK_IF(kShape.GetDimNum() != 4 || uShape.GetDimNum() != 4,
                OP_LOGE(context->GetNodeName(),
                        "k and u origin shapes must be rank-4 BNSD tensors, but got k rank %ld and u rank %ld.",
                        static_cast<int64_t>(kShape.GetDimNum()), static_cast<int64_t>(uShape.GetDimNum())),
                return ge::GRAPH_FAILED);

    auto cuSeqlensTensor = context->GetOptionalInputTensor(INPUT_SEQLENS_IDX);
    auto chunkIndicesTensor = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
    OP_CHECK_IF((cuSeqlensTensor == nullptr) != (chunkIndicesTensor == nullptr),
                OP_LOGE(context->GetNodeName(),
                        "cu_seqlens and chunk_indices must be both provided or both omitted."),
                return ge::GRAPH_FAILED);
    if (cuSeqlensTensor != nullptr) {
        const auto &cuSeqlensShape = cuSeqlensTensor->GetStorageShape();
        OP_CHECK_IF(cuSeqlensShape.GetDimNum() != 1 || cuSeqlensShape.GetDim(0) < 2,
                    OP_LOGE(context->GetNodeName(),
                            "cu_seqlens must be rank 1 and contain at least two elements."),
                    return ge::GRAPH_FAILED);
    }
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
    bool useExp2 = *(attrPtr->GetAttrPointer<bool>(ATTR_USE_EXP2_IDX));
    OP_CHECK_IF(chunkSize != 64,
                OP_LOGE(context->GetNodeName(),
                        "chunk_size only supports 64 in the current version, but got %ld.", chunkSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(useExp2 && !useGk,
                OP_LOGE(context->GetNodeName(),
                        "use_exp2=true requires gk; the scalar g-only path uses natural-log decay."),
                return ge::GRAPH_FAILED);

    int64_t numChunks = (kShape.GetDim(DIM_SEQLEN) + chunkSize - 1) / chunkSize;
    if (chunkIndicesTensor != nullptr) {
        const auto &chunkIndicesShape = chunkIndicesTensor->GetStorageShape();
        if (chunkIndicesShape.GetDimNum() == 1) {
            OP_CHECK_IF(chunkIndicesShape.GetDim(0) % 2 != 0,
                        OP_LOGE(context->GetNodeName(),
                                "flat chunk_indices must contain an even number of elements."),
                        return ge::GRAPH_FAILED);
            numChunks = chunkIndicesShape.GetDim(0) / 2;
        } else if (chunkIndicesShape.GetDimNum() == 2) {
            OP_CHECK_IF(chunkIndicesShape.GetDim(1) != 2,
                        OP_LOGE(context->GetNodeName(),
                                "rank-2 chunk_indices must have shape [NT, 2]."),
                        return ge::GRAPH_FAILED);
            numChunks = chunkIndicesShape.GetDim(0);
        } else {
            OP_LOGE(context->GetNodeName(),
                    "chunk_indices must be flat [2*NT] or rank-2 [NT, 2].");
            return ge::GRAPH_FAILED;
        }
    }
    auto hShapePtr = context->GetOutputShape(OUTPUT_H_IDX);
    OP_CHECK_IF(hShapePtr == nullptr,
                OP_LOGE(context->GetNodeName(), "h output shape must not be null."),
                return ge::GRAPH_FAILED);
    const auto &hShape = hShapePtr->GetOriginShape();
    OP_CHECK_IF(hShape.GetDimNum() != 5 || hShape.GetDim(2) != numChunks,
                OP_LOGE(context->GetNodeName(),
                        "h must be rank 5 with dimension 2 equal to num_chunks=%ld.", numChunks),
                return ge::GRAPH_FAILED);

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    ChunkGatedDeltaRuleFwdHTilingContext tilingCtx{};
    tilingCtx.seqlen = kShape.GetDim(DIM_SEQLEN);
    tilingCtx.kNumHead = kShape.GetDim(DIM_HEAD_NUM);
    tilingCtx.kHeadDim = kShape.GetDim(DIM_HEAD_DIM);
    tilingCtx.vNumHead = uShape.GetDim(DIM_HEAD_NUM);
    tilingCtx.vHeadDim = uShape.GetDim(DIM_HEAD_DIM);
    tilingCtx.shapeBatchDim = kShape.GetDim(DIM_BATCH);
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
    tilingCtx.useExp2 = useExp2;
    tilingCtx.aicCoreNum = ascendcPlatform.GetCoreNumAic();
    tilingCtx.libApiWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    if (tilingCtx.shapeBatchDim <= 0 || tilingCtx.seqlen <= 0 || tilingCtx.kNumHead <= 0 ||
        tilingCtx.vNumHead <= 0 || tilingCtx.kHeadDim <= 0 || tilingCtx.vHeadDim <= 0 ||
        tilingCtx.vNumHead % tilingCtx.kNumHead != 0) {
        OP_LOGE(context->GetNodeName(),
                "Check logical shape failed: batch=%ld, seqlen=%ld, kDim=%ld, vDim=%ld; "
                "vNumHead (%ld) must be divisible by kNumHead (%ld).",
                tilingCtx.shapeBatchDim, tilingCtx.seqlen, tilingCtx.kHeadDim, tilingCtx.vHeadDim,
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
    tiling.set_useExp2(plainTiling.useExp2);
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
