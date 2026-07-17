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
static constexpr size_t OUTPUT_V_NEW_IDX = 1;
static constexpr size_t OUTPUT_FINAL_STATE_IDX = 2;

static constexpr size_t ATTR_STORE_FINAL_STATE_IDX = 0;
static constexpr size_t ATTR_CHUNK_SIZE_IDX = 1;

static constexpr size_t DIM_BATCH = 0;
static constexpr size_t DIM_HEAD_NUM = 1;
static constexpr size_t DIM_SEQLEN = 2;
static constexpr size_t DIM_HEAD_DIM = 3;

static constexpr uint32_t TILING_KEY_V128 = 1;
static constexpr uint32_t TILING_KEY_V256 = 2;
static constexpr int64_t V_DIM_128 = 128;
static constexpr int64_t V_DIM_256 = 256;
static constexpr int64_t K_DIM_128 = 128;

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
    OP_LOGD(nodeName, "=== useGk: %d", tiling.get_useGk());
    OP_LOGD(nodeName, "=== dataType: %ld", tiling.get_dataType());
    OP_LOGD(nodeName, "=== isVariedLen: %ld", tiling.get_isVariedLen());
    OP_LOGD(nodeName, "=== shapeBatch: %ld", tiling.get_shapeBatch());
    OP_LOGD(nodeName, "=== tokenBatch: %ld", tiling.get_tokenBatch());
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print ChunkGatedDeltaRuleFwdH tiling data end <<<<<<<<<<<<<<<<");
}

ge::graphStatus Tiling4ChunkGatedDeltaRuleFwdH(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkGatedDeltaRuleFwdH start.");
    ChunkGatedDeltaRuleFwdHTilingData tiling;

    auto kShapePtr = context->GetInputShape(INPUT_K_IDX);
    auto wShapePtr = context->GetInputShape(INPUT_W_IDX);
    auto uShapePtr = context->GetInputShape(INPUT_U_IDX);
    auto hShapePtr = context->GetOutputShape(OUTPUT_H_IDX);
    auto vNewShapePtr = context->GetOutputShape(OUTPUT_V_NEW_IDX);
    auto kDesc = context->GetInputDesc(INPUT_K_IDX);
    auto wDesc = context->GetInputDesc(INPUT_W_IDX);
    auto uDesc = context->GetInputDesc(INPUT_U_IDX);
    auto hDesc = context->GetOutputDesc(OUTPUT_H_IDX);
    auto vNewDesc = context->GetOutputDesc(OUTPUT_V_NEW_IDX);
    OP_CHECK_IF(kShapePtr == nullptr || wShapePtr == nullptr || uShapePtr == nullptr || hShapePtr == nullptr ||
                    vNewShapePtr == nullptr || kDesc == nullptr || wDesc == nullptr || uDesc == nullptr ||
                    hDesc == nullptr || vNewDesc == nullptr,
                OP_LOGE(context->GetNodeName(), "required input/output shape and dtype descriptors must be present."),
                return ge::GRAPH_FAILED);

    const gert::Shape kStorageShape = kShapePtr->GetStorageShape();
    const gert::Shape wStorageShape = wShapePtr->GetStorageShape();
    const gert::Shape uStorageShape = uShapePtr->GetStorageShape();
    const gert::Shape hStorageShape = hShapePtr->GetStorageShape();
    const gert::Shape vNewStorageShape = vNewShapePtr->GetStorageShape();
    OP_CHECK_IF(kStorageShape.GetDimNum() != 4 || wStorageShape.GetDimNum() != 4 ||
                    uStorageShape.GetDimNum() != 4 || hStorageShape.GetDimNum() != 5 ||
                    vNewStorageShape.GetDimNum() != 4,
                OP_LOGE(context->GetNodeName(), "k/w/u/v_new must be rank 4 and h must be rank 5."),
                return ge::GRAPH_FAILED);
    for (size_t dim = 0; dim < 4; ++dim) {
        OP_CHECK_IF(kStorageShape.GetDim(dim) <= 0 || wStorageShape.GetDim(dim) <= 0 ||
                        uStorageShape.GetDim(dim) <= 0 || vNewStorageShape.GetDim(dim) <= 0,
                    OP_LOGE(context->GetNodeName(), "k/w/u/v_new dimensions must be positive."),
                    return ge::GRAPH_FAILED);
    }
    for (size_t dim = 0; dim < 5; ++dim) {
        OP_CHECK_IF(hStorageShape.GetDim(dim) <= 0,
                    OP_LOGE(context->GetNodeName(), "h dimensions must be positive."),
                    return ge::GRAPH_FAILED);
    }

    const int64_t batch = kStorageShape.GetDim(DIM_BATCH);
    const int64_t seqlen = kStorageShape.GetDim(DIM_SEQLEN);
    const int64_t kNumHead = kStorageShape.GetDim(DIM_HEAD_NUM);
    const int64_t vNumHead = uStorageShape.GetDim(DIM_HEAD_NUM);
    const int64_t kHeadDim = kStorageShape.GetDim(DIM_HEAD_DIM);
    const int64_t vHeadDim = uStorageShape.GetDim(DIM_HEAD_DIM);
    OP_CHECK_IF(wStorageShape.GetDim(0) != batch || uStorageShape.GetDim(0) != batch ||
                    wStorageShape.GetDim(1) != vNumHead || wStorageShape.GetDim(2) != seqlen ||
                    uStorageShape.GetDim(2) != seqlen || wStorageShape.GetDim(3) != kHeadDim,
                OP_LOGE(context->GetNodeName(), "k/w/u shape relation does not match [B,H,T,K/V]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vNumHead < kNumHead || vNumHead % kNumHead != 0,
                OP_LOGE(context->GetNodeName(), "H_v must be divisible by H_k."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kHeadDim != K_DIM_128 || (vHeadDim != V_DIM_128 && vHeadDim != V_DIM_256),
                OP_LOGE(context->GetNodeName(), "K must be 128 and V must be 128 or 256."),
                return ge::GRAPH_FAILED);

    const ge::DataType inputDtype = kDesc->GetDataType();
    OP_CHECK_IF((inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_BF16) ||
                    wDesc->GetDataType() != inputDtype || uDesc->GetDataType() != inputDtype ||
                    hDesc->GetDataType() != inputDtype || vNewDesc->GetDataType() != inputDtype,
                OP_LOGE(context->GetNodeName(), "k/w/u/h/v_new must use matching float16 or bfloat16 dtype."),
                return ge::GRAPH_FAILED);

    auto gTensor = context->GetOptionalInputTensor(INPUT_G_IDX);
    auto gkTensor = context->GetOptionalInputTensor(INPUT_GK_IDX);
    auto gShapePtr = context->GetOptionalInputShape(INPUT_G_IDX);
    auto gkShapePtr = context->GetOptionalInputShape(INPUT_GK_IDX);
    auto gDesc = context->GetOptionalInputDesc(INPUT_G_IDX);
    auto gkDesc = context->GetOptionalInputDesc(INPUT_GK_IDX);
    OP_CHECK_IF(gTensor == nullptr && gkTensor == nullptr,
                OP_LOGE(context->GetNodeName(), "Either g or gk must be provided."),
                return ge::GRAPH_FAILED);
    auto checkGateDtype = [&](const gert::CompileTimeTensorDesc *desc) {
        return desc == nullptr || desc->GetDataType() == ge::DT_FLOAT || desc->GetDataType() == inputDtype;
    };
    OP_CHECK_IF(!checkGateDtype(gDesc) || !checkGateDtype(gkDesc) ||
                    (gDesc != nullptr && gkDesc != nullptr && gDesc->GetDataType() != gkDesc->GetDataType()),
                OP_LOGE(context->GetNodeName(), "g/gk must use float32 or input dtype and match when both exist."),
                return ge::GRAPH_FAILED);
    if (gShapePtr != nullptr) {
        const gert::Shape gShape = gShapePtr->GetStorageShape();
        OP_CHECK_IF(gShape.GetDimNum() != 3 || gShape.GetDim(0) != batch || gShape.GetDim(1) != vNumHead ||
                        gShape.GetDim(2) != seqlen,
                    OP_LOGE(context->GetNodeName(), "g must be [B,H_v,T]."),
                    return ge::GRAPH_FAILED);
    }
    if (gkShapePtr != nullptr) {
        const gert::Shape gkShape = gkShapePtr->GetStorageShape();
        OP_CHECK_IF(gkShape.GetDimNum() != 4 || gkShape.GetDim(0) != batch ||
                        gkShape.GetDim(1) != vNumHead || gkShape.GetDim(2) != seqlen ||
                        gkShape.GetDim(3) != kHeadDim,
                    OP_LOGE(context->GetNodeName(), "gk must be [B,H_v,T,K]."),
                    return ge::GRAPH_FAILED);
    }
    auto gateTensor = gTensor != nullptr ? gTensor : gkTensor;
    bool useGk = gkTensor != nullptr;

    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);
    auto storeFinalStatePtr = attrPtr->GetAttrPointer<bool>(ATTR_STORE_FINAL_STATE_IDX);
    auto chunkSizePtr = attrPtr->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, storeFinalStatePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSizePtr);
    const bool storeFinalState = *storeFinalStatePtr;
    const int64_t chunkSize = *chunkSizePtr;
    OP_CHECK_IF(chunkSize != 64 && chunkSize != 128,
                OP_LOGE(context->GetNodeName(), "chunk_size must be 64 or 128."),
                return ge::GRAPH_FAILED);

    auto cuSeqlensTensor = context->GetOptionalInputTensor(INPUT_SEQLENS_IDX);
    auto chunkIndicesTensor = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
    auto cuSeqlensShapePtr = context->GetOptionalInputShape(INPUT_SEQLENS_IDX);
    auto chunkIndicesShapePtr = context->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX);
    auto cuSeqlensDesc = context->GetOptionalInputDesc(INPUT_SEQLENS_IDX);
    auto chunkIndicesDesc = context->GetOptionalInputDesc(INPUT_CHUNK_INDICES_IDX);
    const bool hasCuSeqlens = cuSeqlensTensor != nullptr;
    const bool hasChunkIndices = chunkIndicesTensor != nullptr;
    OP_CHECK_IF(hasCuSeqlens != hasChunkIndices,
                OP_LOGE(context->GetNodeName(), "cu_seqlens and chunk_indices must be provided together."),
                return ge::GRAPH_FAILED);

    int64_t logicalSequenceCount = batch;
    int64_t expectedChunks = (seqlen + chunkSize - 1) / chunkSize;
    if (hasCuSeqlens) {
        OP_CHECK_IF(batch != 1 || cuSeqlensShapePtr == nullptr || chunkIndicesShapePtr == nullptr ||
                        cuSeqlensDesc == nullptr || chunkIndicesDesc == nullptr,
                    OP_LOGE(context->GetNodeName(), "varlen mode requires B=1 and complete metadata."),
                    return ge::GRAPH_FAILED);
        const gert::Shape cuShape = cuSeqlensShapePtr->GetStorageShape();
        const gert::Shape indicesShape = chunkIndicesShapePtr->GetStorageShape();
        OP_CHECK_IF(cuShape.GetDimNum() != 1 || cuShape.GetDim(0) < 2 || indicesShape.GetDimNum() != 1 ||
                        indicesShape.GetDim(0) <= 0 || indicesShape.GetDim(0) % 2 != 0 ||
                        cuSeqlensDesc->GetDataType() != ge::DT_INT64 ||
                        chunkIndicesDesc->GetDataType() != ge::DT_INT64,
                    OP_LOGE(context->GetNodeName(), "varlen metadata must be nonempty rank-1 int64 arrays."),
                    return ge::GRAPH_FAILED);
        const int64_t *cu = cuSeqlensTensor->GetData<int64_t>();
        const int64_t *indices = chunkIndicesTensor->GetData<int64_t>();
        OP_CHECK_IF(cu == nullptr || indices == nullptr,
                    OP_LOGE(context->GetNodeName(), "varlen metadata data must be available to tiling."),
                    return ge::GRAPH_FAILED);
        logicalSequenceCount = cuShape.GetDim(0) - 1;
        OP_CHECK_IF(cu[0] != 0 || cu[logicalSequenceCount] != seqlen,
                    OP_LOGE(context->GetNodeName(), "cu_seqlens must start at 0 and end at T."),
                    return ge::GRAPH_FAILED);
        expectedChunks = 0;
        const int64_t suppliedChunks = indicesShape.GetDim(0) / 2;
        for (int64_t seq = 0; seq < logicalSequenceCount; ++seq) {
            OP_CHECK_IF(cu[seq] < 0 || cu[seq] > cu[seq + 1] || cu[seq + 1] > seqlen,
                        OP_LOGE(context->GetNodeName(), "cu_seqlens must be nondecreasing and within [0,T]."),
                        return ge::GRAPH_FAILED);
            const int64_t localChunkCount = (cu[seq + 1] - cu[seq] + chunkSize - 1) / chunkSize;
            for (int64_t localChunk = 0; localChunk < localChunkCount; ++localChunk) {
                OP_CHECK_IF(expectedChunks >= suppliedChunks || indices[expectedChunks * 2] != seq ||
                                indices[expectedChunks * 2 + 1] != localChunk,
                            OP_LOGE(context->GetNodeName(),
                                    "chunk_indices must use canonical sequence-major chunk order."),
                            return ge::GRAPH_FAILED);
                ++expectedChunks;
            }
        }
        OP_CHECK_IF(expectedChunks != suppliedChunks,
                    OP_LOGE(context->GetNodeName(), "chunk_indices pair count does not match cu_seqlens."),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(hStorageShape.GetDim(0) != batch || hStorageShape.GetDim(1) != vNumHead ||
                    hStorageShape.GetDim(2) != expectedChunks || hStorageShape.GetDim(3) != kHeadDim ||
                    hStorageShape.GetDim(4) != vHeadDim,
                OP_LOGE(context->GetNodeName(), "h must be [B,H_v,N_c,K,V] with derived N_c."),
                return ge::GRAPH_FAILED);
    for (size_t dim = 0; dim < 4; ++dim) {
        OP_CHECK_IF(vNewStorageShape.GetDim(dim) != uStorageShape.GetDim(dim),
                    OP_LOGE(context->GetNodeName(), "v_new shape must match u."),
                    return ge::GRAPH_FAILED);
    }

    auto initialStateTensor = context->GetOptionalInputTensor(INPUT_INITIAL_STATE_IDX);
    auto initialStateShapePtr = context->GetOptionalInputShape(INPUT_INITIAL_STATE_IDX);
    auto initialStateDesc = context->GetOptionalInputDesc(INPUT_INITIAL_STATE_IDX);
    const bool useInitialState = initialStateTensor != nullptr;
    if (useInitialState) {
        OP_CHECK_IF(initialStateShapePtr == nullptr || initialStateDesc == nullptr,
                    OP_LOGE(context->GetNodeName(), "initial_state descriptor must be present."),
                    return ge::GRAPH_FAILED);
        const gert::Shape stateShape = initialStateShapePtr->GetStorageShape();
        const ge::DataType stateDtype = initialStateDesc->GetDataType();
        OP_CHECK_IF(stateShape.GetDimNum() != 4 || stateShape.GetDim(0) != logicalSequenceCount ||
                        stateShape.GetDim(1) != vNumHead || stateShape.GetDim(2) != kHeadDim ||
                        stateShape.GetDim(3) != vHeadDim ||
                        (stateDtype != ge::DT_FLOAT && stateDtype != inputDtype),
                    OP_LOGE(context->GetNodeName(),
                            "initial_state must be [N,H_v,K,V] with float32 or input dtype."),
                    return ge::GRAPH_FAILED);
    }
    if (storeFinalState) {
        auto finalStateShapePtr = context->GetOutputShape(OUTPUT_FINAL_STATE_IDX);
        auto finalStateDesc = context->GetOutputDesc(OUTPUT_FINAL_STATE_IDX);
        OP_CHECK_IF(finalStateShapePtr == nullptr || finalStateDesc == nullptr,
                    OP_LOGE(context->GetNodeName(), "final_state output is required when requested."),
                    return ge::GRAPH_FAILED);
        const gert::Shape finalShape = finalStateShapePtr->GetStorageShape();
        const ge::DataType expectedStateDtype = useInitialState ? initialStateDesc->GetDataType() : ge::DT_FLOAT;
        OP_CHECK_IF(finalShape.GetDimNum() != 4 || finalShape.GetDim(0) != logicalSequenceCount ||
                        finalShape.GetDim(1) != vNumHead || finalShape.GetDim(2) != kHeadDim ||
                        finalShape.GetDim(3) != vHeadDim || finalStateDesc->GetDataType() != expectedStateDtype,
                    OP_LOGE(context->GetNodeName(), "final_state shape/dtype must match the state contract."),
                    return ge::GRAPH_FAILED);
    }

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    OP_CHECK_IF(ascendcPlatform.GetCoreNumAic() == 0,
                OP_LOGE(context->GetNodeName(), "AIC core count must be positive."),
                return ge::GRAPH_FAILED);

    ChunkGatedDeltaRuleFwdHTilingContext tilingCtx{};
    tilingCtx.seqlen = seqlen;
    tilingCtx.kNumHead = kNumHead;
    tilingCtx.kHeadDim = kHeadDim;
    tilingCtx.vNumHead = vNumHead;
    tilingCtx.vHeadDim = vHeadDim;
    tilingCtx.shapeBatchDim = batch;
    tilingCtx.hasCuSeqlens = hasCuSeqlens;
    tilingCtx.cuSeqlensDim0 = hasCuSeqlens ? logicalSequenceCount + 1 : 0;
    tilingCtx.dataType = GdnFwdHDtypeToEnum(inputDtype);
    tilingCtx.gDataType = GdnFwdHDtypeToEnum(gateTensor->GetDataType());
    tilingCtx.useInitialState = useInitialState;
    tilingCtx.stateDataType =
        useInitialState ? GdnFwdHDtypeToEnum(initialStateTensor->GetDataType()) : GDN_FWD_H_DTYPE_FP32;
    tilingCtx.storeFinalState = storeFinalState;
    tilingCtx.chunkSize = chunkSize;
    tilingCtx.useGk = useGk;
    tilingCtx.aicCoreNum = ascendcPlatform.GetCoreNumAic();
    tilingCtx.libApiWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

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
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
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
    tiling.set_useGk(plainTiling.useGk);
    tiling.set_vWorkspaceOffset(plainTiling.vWorkspaceOffset);
    tiling.set_vUpdateWorkspaceOffset(plainTiling.vUpdateWorkspaceOffset);
    tiling.set_kDecayWorkspaceOffset(plainTiling.kDecayWorkspaceOffset);
    tiling.set_hWorkspaceOffset(plainTiling.hWorkspaceOffset);
    tiling.set_numSeqWorkspaceOffset(plainTiling.numSeqWorkspaceOffset);
    tiling.set_numChunksWorkspaceOffset(plainTiling.numChunksWorkspaceOffset);

    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr || rawTilingData->GetData() == nullptr ||
                    rawTilingData->GetCapacity() < tiling.GetDataSize(),
                OP_LOGE(context->GetNodeName(), "raw tiling buffer is null or too small."),
                return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tiling.GetDataSize());

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
