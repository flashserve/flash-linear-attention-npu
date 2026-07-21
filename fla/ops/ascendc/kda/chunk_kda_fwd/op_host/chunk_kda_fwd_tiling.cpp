/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "chunk_kda_fwd_tiling.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_Q_IDX = 0;
constexpr size_t INPUT_K_IDX = 1;
constexpr size_t INPUT_V_IDX = 2;
constexpr size_t INPUT_GK_IDX = 3;
constexpr size_t INPUT_BETA_IDX = 4;
constexpr size_t INPUT_INITIAL_IDX = 5;
constexpr size_t INPUT_CU_SEQLENS_IDX = 6;
constexpr size_t INPUT_CHUNK_INDICES_IDX = 7;
constexpr size_t INPUT_STAGE_QG_IDX = 8;
constexpr size_t INPUT_STAGE_AQK_IDX = 9;
constexpr size_t INPUT_STAGE_V_NEW_IDX = 10;
constexpr size_t INPUT_STAGE_H_IDX = 11;
constexpr size_t ATTR_SCALE_IDX = 0;
constexpr size_t ATTR_CHUNK_SIZE_IDX = 1;
constexpr size_t ATTR_OUTPUT_FINAL_STATE_IDX = 2;
constexpr size_t ATTR_TOTAL_CHUNKS_IDX = 3;
constexpr size_t ATTR_STAGE_IDX = 4;
constexpr uint64_t KDA_SOLVE_SCRATCH_SLOTS = 5;
constexpr uint64_t KDA_SCORE_QUEUE_SLOTS = 2;
constexpr uint64_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint64_t KDA_FP32_BYTES = sizeof(float);
constexpr uint64_t KDA_WORKSPACE_ALIGN = 512;

constexpr size_t DIM_B = 0;
constexpr size_t DIM_H = 1;
constexpr size_t DIM_T = 2;
constexpr size_t DIM_D = 3;

int64_t DTypeCode(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return 1;
    }
    if (dtype == ge::DT_FLOAT) {
        return 2;
    }
    return 0;
}

bool ShapeEquals(const gert::Shape &shape, std::initializer_list<int64_t> dims)
{
    if (shape.GetDimNum() != dims.size()) {
        return false;
    }
    size_t idx = 0;
    for (int64_t dim : dims) {
        if (shape.GetDim(idx++) != dim) {
            return false;
        }
    }
    return true;
}

bool OptionalTensorMatches(gert::TilingContext *context, size_t index, ge::DataType dtype,
                           std::initializer_list<int64_t> dims)
{
    const auto *desc = context->GetOptionalInputDesc(index);
    const auto *shape = context->GetOptionalInputShape(index);
    return desc != nullptr && shape != nullptr && desc->GetDataType() == dtype &&
           ShapeEquals(shape->GetStorageShape(), dims);
}
} // namespace

ge::graphStatus Tiling4ChunkKdaFwd(gert::TilingContext *context)
{
    if (context == nullptr || context->GetRequiredInputShape(INPUT_Q_IDX) == nullptr ||
        context->GetRequiredInputShape(INPUT_K_IDX) == nullptr ||
        context->GetRequiredInputShape(INPUT_V_IDX) == nullptr ||
        context->GetRequiredInputShape(INPUT_GK_IDX) == nullptr ||
        context->GetRequiredInputShape(INPUT_BETA_IDX) == nullptr || context->GetInputDesc(INPUT_Q_IDX) == nullptr ||
        context->GetInputDesc(INPUT_K_IDX) == nullptr || context->GetInputDesc(INPUT_V_IDX) == nullptr ||
        context->GetInputDesc(INPUT_GK_IDX) == nullptr || context->GetInputDesc(INPUT_BETA_IDX) == nullptr ||
        context->GetAttrs() == nullptr || context->GetRawTilingData() == nullptr ||
        context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ChunkKdaFwdTilingData tiling;

    auto qShape = context->GetRequiredInputShape(INPUT_Q_IDX)->GetStorageShape();
    auto kShape = context->GetRequiredInputShape(INPUT_K_IDX)->GetStorageShape();
    auto vShape = context->GetRequiredInputShape(INPUT_V_IDX)->GetStorageShape();
    auto gkShape = context->GetRequiredInputShape(INPUT_GK_IDX)->GetStorageShape();
    auto betaShape = context->GetRequiredInputShape(INPUT_BETA_IDX)->GetStorageShape();
    auto qDesc = context->GetInputDesc(INPUT_Q_IDX);
    auto kDesc = context->GetInputDesc(INPUT_K_IDX);
    auto vDesc = context->GetInputDesc(INPUT_V_IDX);
    auto gDesc = context->GetInputDesc(INPUT_GK_IDX);
    auto betaDesc = context->GetInputDesc(INPUT_BETA_IDX);
    const ge::DataType qDtype = qDesc->GetDataType();
    if ((qDtype != ge::DT_FLOAT16 && qDtype != ge::DT_BF16) || kDesc->GetDataType() != qDtype ||
        vDesc->GetDataType() != qDtype || gDesc->GetDataType() != ge::DT_FLOAT ||
        betaDesc->GetDataType() != ge::DT_FLOAT || qShape.GetDimNum() != 4 || kShape.GetDimNum() != 4 ||
        vShape.GetDimNum() != 4 || gkShape.GetDimNum() != 4 || betaShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = qShape.GetDim(DIM_B);
    const int64_t qHeadNum = qShape.GetDim(DIM_H);
    const int64_t seqlen = qShape.GetDim(DIM_T);
    const int64_t kDim = qShape.GetDim(DIM_D);
    const int64_t vHeadNum = vShape.GetDim(DIM_H);
    const int64_t vDim = vShape.GetDim(DIM_D);
    if (batch <= 0 || qHeadNum <= 0 || seqlen <= 0 || kDim < 16 || kDim > 256 || kDim % 16 != 0 ||
        vHeadNum < qHeadNum || vHeadNum > 128 || qHeadNum > 128 || vHeadNum % qHeadNum != 0 ||
        vDim < 16 || vDim > 256 || vDim % 16 != 0 ||
        !ShapeEquals(kShape, {batch, qHeadNum, seqlen, kDim}) ||
        !ShapeEquals(vShape, {batch, vHeadNum, seqlen, vDim}) ||
        !ShapeEquals(gkShape, {batch, vHeadNum, seqlen, kDim}) ||
        !ShapeEquals(betaShape, {batch, vHeadNum, seqlen})) {
        return ge::GRAPH_FAILED;
    }

    auto attrPtr = context->GetAttrs();
    const double *scalePtr = attrPtr->GetAttrPointer<double>(ATTR_SCALE_IDX);
    const int64_t *chunkSizePtr = attrPtr->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX);
    const bool *outputFinalStatePtr = attrPtr->GetAttrPointer<bool>(ATTR_OUTPUT_FINAL_STATE_IDX);
    const int64_t *totalChunksPtr = attrPtr->GetAttrPointer<int64_t>(ATTR_TOTAL_CHUNKS_IDX);
    if (scalePtr == nullptr || chunkSizePtr == nullptr || outputFinalStatePtr == nullptr || totalChunksPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    float scale = static_cast<float>(*scalePtr);
    int64_t chunkSize = *chunkSizePtr;
    bool outputFinalState = *outputFinalStatePtr;
    int64_t totalChunks = *totalChunksPtr;
    int64_t stage = 0;
    const int64_t *stagePtr = attrPtr->GetAttrPointer<int64_t>(ATTR_STAGE_IDX);
    if (stagePtr != nullptr) {
        stage = *stagePtr;
    }
    if (!std::isfinite(*scalePtr) || (chunkSize != 64 && chunkSize != 128) || totalChunks <= 0 ||
        (stage != 1 && stage != 2 && stage != 3)) {
        return ge::GRAPH_FAILED;
    }

    const bool hasCuSeqlens = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX) != nullptr;
    const bool hasChunkMetadata = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX) != nullptr;
    if (hasCuSeqlens != hasChunkMetadata) {
        return ge::GRAPH_FAILED;
    }
    bool isVarLen = hasCuSeqlens;
    int64_t seqNum = batch;
    std::array<int64_t, KDA_MAX_TILING_SEQUENCES> seqStart{};
    std::array<int64_t, KDA_MAX_TILING_SEQUENCES> seqEnd{};
    std::array<int64_t, KDA_MAX_TILING_SEQUENCE_OFFSETS> seqChunkOffset{};
    if (isVarLen) {
        auto cuTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX);
        auto chunkMetadata = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
        const auto *cuDesc = context->GetOptionalInputDesc(INPUT_CU_SEQLENS_IDX);
        const auto *chunkDesc = context->GetOptionalInputDesc(INPUT_CHUNK_INDICES_IDX);
        if (batch != 1 || cuDesc == nullptr || chunkDesc == nullptr ||
            cuDesc->GetDataType() != ge::DT_INT64 || chunkDesc->GetDataType() != ge::DT_INT64 ||
            cuTensor->GetStorageShape().GetDimNum() != 1 || chunkMetadata == nullptr ||
            chunkMetadata->GetStorageShape().GetDimNum() != 1) {
            return ge::GRAPH_FAILED;
        }
        seqNum = cuTensor->GetStorageShape().GetDim(0) - 1;
        if (seqNum <= 0 || seqNum > KDA_MAX_TILING_SEQUENCES ||
            chunkMetadata->GetStorageShape().GetShapeSize() != totalChunks * 4) {
            return ge::GRAPH_FAILED;
        }
        const int64_t *cu = cuTensor->GetData<int64_t>();
        if (cu == nullptr || cu[0] != 0 || cu[seqNum] != seqlen) {
            return ge::GRAPH_FAILED;
        }
        int64_t chunkOffset = 0;
        for (int64_t seq = 0; seq < seqNum; ++seq) {
            if (cu[seq] < 0 || cu[seq + 1] < cu[seq]) {
                return ge::GRAPH_FAILED;
            }
            seqStart[seq] = cu[seq];
            seqEnd[seq] = cu[seq + 1];
            seqChunkOffset[seq] = chunkOffset;
            const int64_t seqLength = cu[seq + 1] - cu[seq];
            chunkOffset += (seqLength + chunkSize - 1) / chunkSize;
        }
        seqChunkOffset[seqNum] = chunkOffset;
        if (chunkOffset != totalChunks) {
            return ge::GRAPH_FAILED;
        }
    } else if (totalChunks != (seqlen + chunkSize - 1) / chunkSize) {
        return ge::GRAPH_FAILED;
    }
    bool hasInitialState = context->GetOptionalInputTensor(INPUT_INITIAL_IDX) != nullptr;
    if (hasInitialState &&
        !OptionalTensorMatches(context, INPUT_INITIAL_IDX, ge::DT_FLOAT, {seqNum, vHeadNum, kDim, vDim})) {
        return ge::GRAPH_FAILED;
    }

    const bool hasStageQG = context->GetOptionalInputTensor(INPUT_STAGE_QG_IDX) != nullptr;
    const bool hasStageAqk = context->GetOptionalInputTensor(INPUT_STAGE_AQK_IDX) != nullptr;
    const bool hasStageVNew = context->GetOptionalInputTensor(INPUT_STAGE_V_NEW_IDX) != nullptr;
    const bool hasStageH = context->GetOptionalInputTensor(INPUT_STAGE_H_IDX) != nullptr;
    if (stage == 1 && (hasStageQG || hasStageAqk || hasStageVNew || hasStageH)) {
        return ge::GRAPH_FAILED;
    }
    if (stage == 3 &&
        (!OptionalTensorMatches(context, INPUT_STAGE_QG_IDX, qDtype, {batch, vHeadNum, seqlen, kDim}) ||
         !OptionalTensorMatches(context, INPUT_STAGE_AQK_IDX, qDtype, {batch, vHeadNum, seqlen, chunkSize}) ||
         !OptionalTensorMatches(context, INPUT_STAGE_V_NEW_IDX, qDtype, {batch, vHeadNum, seqlen, vDim}) ||
         hasStageH)) {
        return ge::GRAPH_FAILED;
    }
    if (stage == 2 &&
        (!OptionalTensorMatches(context, INPUT_STAGE_QG_IDX, qDtype, {batch, vHeadNum, seqlen, kDim}) ||
         !OptionalTensorMatches(context, INPUT_STAGE_AQK_IDX, qDtype, {batch, vHeadNum, seqlen, chunkSize}) ||
         !OptionalTensorMatches(context, INPUT_STAGE_V_NEW_IDX, qDtype, {batch, vHeadNum, seqlen, vDim}) ||
         !OptionalTensorMatches(context, INPUT_STAGE_H_IDX, qDtype,
                                {batch, vHeadNum, totalChunks, kDim, vDim}))) {
        return ge::GRAPH_FAILED;
    }

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAic();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t blockDim = coreNum;
    context->SetBlockDim(blockDim);
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint64_t kernelScratch = 0;
    if (stage == 1) {
        const uint64_t usedCoreNum = static_cast<uint64_t>(blockDim);
        const uint64_t solveScratch = usedCoreNum * KDA_SOLVE_SCRATCH_SLOTS *
                                      static_cast<uint64_t>(chunkSize) * static_cast<uint64_t>(chunkSize) *
                                      KDA_FP32_BYTES;
        const uint64_t alignedSolveScratch =
            (solveScratch + KDA_WORKSPACE_ALIGN - 1) / KDA_WORKSPACE_ALIGN * KDA_WORKSPACE_ALIGN;
        const uint64_t scoreElementBytes = qDesc->GetDataType() == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
        const uint64_t scoreScratch = usedCoreNum * KDA_SCORE_QUEUE_SLOTS * KDA_SCORE_SCRATCH_PLANES *
                                      static_cast<uint64_t>(chunkSize) *
                                      static_cast<uint64_t>(kDim) * scoreElementBytes;
        kernelScratch = alignedSolveScratch + scoreScratch;
    } else if (stage == 2) {
        const uint64_t outputElements = static_cast<uint64_t>(batch) *
                                        static_cast<uint64_t>(vHeadNum) *
                                        static_cast<uint64_t>(seqlen) *
                                        static_cast<uint64_t>(vDim);
        kernelScratch = 2 * outputElements * KDA_FP32_BYTES;
    }
    kernelScratch = (kernelScratch + KDA_WORKSPACE_ALIGN - 1) / KDA_WORKSPACE_ALIGN * KDA_WORKSPACE_ALIGN;
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + kernelScratch;

    tiling.set_batch(batch);
    tiling.set_seqNum(seqNum);
    tiling.set_qHeadNum(qHeadNum);
    tiling.set_vHeadNum(vHeadNum);
    tiling.set_seqlen(seqlen);
    tiling.set_kHeadDim(kDim);
    tiling.set_vHeadDim(vDim);
    tiling.set_chunkSize(chunkSize);
    tiling.set_totalChunks(totalChunks);
    tiling.set_scale(scale);
    tiling.set_hasInitialState(hasInitialState);
    tiling.set_outputFinalState(outputFinalState);
    tiling.set_isVarLen(isVarLen);
    tiling.set_dataType(DTypeCode(qDtype));
    tiling.set_gateDataType(DTypeCode(gDesc->GetDataType()));
    tiling.set_usedCoreNum(blockDim);
    tiling.set_stage(stage);
    tiling.set_seqStart(seqStart.data());
    tiling.set_seqEnd(seqEnd.data());
    tiling.set_seqChunkOffset(seqChunkOffset.data());

    context->SetTilingKey(1);
    auto rawTilingData = context->GetRawTilingData();
    if (rawTilingData->GetData() == nullptr || rawTilingData->GetCapacity() < tiling.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaFwd(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaFwd)
    .Tiling(Tiling4ChunkKdaFwd)
    .TilingParse<ChunkKdaFwdCompileInfo>(TilingPrepare4ChunkKdaFwd);

} // namespace optiling
