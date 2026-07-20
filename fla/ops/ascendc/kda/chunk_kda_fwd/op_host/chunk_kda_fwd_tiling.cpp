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
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"
#include "../../../gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling_processor.h"

namespace optiling {
namespace {
constexpr size_t INPUT_Q_IDX = 0;
constexpr size_t INPUT_V_IDX = 2;
constexpr size_t INPUT_GK_IDX = 3;
constexpr size_t INPUT_INITIAL_IDX = 5;
constexpr size_t INPUT_CU_SEQLENS_IDX = 6;
constexpr size_t INPUT_CHUNK_INDICES_IDX = 7;
constexpr size_t ATTR_SCALE_IDX = 0;
constexpr size_t ATTR_CHUNK_SIZE_IDX = 1;
constexpr size_t ATTR_OUTPUT_FINAL_STATE_IDX = 2;
constexpr size_t ATTR_TOTAL_CHUNKS_IDX = 3;
constexpr size_t ATTR_SAFE_GATE_IDX = 4;
constexpr uint64_t KDA_SOLVE_SCRATCH_SLOTS = 5;
constexpr uint64_t KDA_SCORE_QUEUE_SLOTS = 2;
constexpr uint64_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint64_t KDA_FP32_BYTES = sizeof(float);
constexpr uint64_t KDA_WORKSPACE_ALIGN = 512;
constexpr uint32_t KDA_BATCH_MODE = 1;

constexpr size_t DIM_B = 0;
constexpr size_t DIM_H = 1;
constexpr size_t DIM_T = 2;
constexpr size_t DIM_D = 3;

uint64_t AlignWorkspace(uint64_t bytes)
{
    return (bytes + KDA_WORKSPACE_ALIGN - 1) / KDA_WORKSPACE_ALIGN * KDA_WORKSPACE_ALIGN;
}

int64_t KdaDtypeToEnum(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return GDN_FWD_H_DTYPE_BF16;
    }
    if (dtype == ge::DT_FLOAT16) {
        return GDN_FWD_H_DTYPE_FP16;
    }
    return GDN_FWD_H_DTYPE_FP32;
}
} // namespace

ge::graphStatus Tiling4ChunkKdaFwd(gert::TilingContext *context)
{
    ChunkKdaFwdTilingData tiling;

    auto qShape = context->GetOptionalInputShape(INPUT_Q_IDX)->GetStorageShape();
    auto vShape = context->GetOptionalInputShape(INPUT_V_IDX)->GetStorageShape();
    auto qDesc = context->GetInputDesc(INPUT_Q_IDX);
    if (qDesc == nullptr || context->GetInputDesc(INPUT_GK_IDX) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto attrPtr = context->GetAttrs();
    if (attrPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    float scale = static_cast<float>(*(attrPtr->GetAttrPointer<double>(ATTR_SCALE_IDX)));
    int64_t chunkSize = *(attrPtr->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX));
    bool outputFinalState = *(attrPtr->GetAttrPointer<bool>(ATTR_OUTPUT_FINAL_STATE_IDX));
    int64_t totalChunks = *(attrPtr->GetAttrPointer<int64_t>(ATTR_TOTAL_CHUNKS_IDX));
    bool safeGate = *(attrPtr->GetAttrPointer<bool>(ATTR_SAFE_GATE_IDX));

    bool isVarLen = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX) != nullptr;
    int64_t batch = qShape.GetDim(DIM_B);
    int64_t seqNum = batch;
    if (isVarLen) {
        auto cuTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX);
        seqNum = cuTensor->GetStorageShape().GetDim(0) - 1;
        auto chunkMetadata = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
        if (seqNum <= 0 || chunkMetadata == nullptr ||
            chunkMetadata->GetStorageShape().GetShapeSize() != totalChunks * 4) {
            return ge::GRAPH_FAILED;
        }
        const int64_t *cu = cuTensor->GetData<int64_t>();
        if (cu == nullptr) {
            return ge::GRAPH_FAILED;
        }
        int64_t chunkOffset = 0;
        for (int64_t seq = 0; seq < seqNum; ++seq) {
            if (cu[seq] < 0 || cu[seq + 1] < cu[seq]) {
                return ge::GRAPH_FAILED;
            }
            const int64_t seqLength = cu[seq + 1] - cu[seq];
            chunkOffset += (seqLength + chunkSize - 1) / chunkSize;
        }
        if (chunkOffset != totalChunks) {
            return ge::GRAPH_FAILED;
        }
    }
    bool hasInitialState = context->GetOptionalInputTensor(INPUT_INITIAL_IDX) != nullptr;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = coreNum;
    context->SetBlockDim(blockDim == 0 ? 1 : blockDim);
    const uint64_t usedCoreNum = static_cast<uint64_t>(blockDim == 0 ? 1 : blockDim);
    const uint64_t dataBytes = qDesc->GetDataType() == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    const uint64_t tokenHeadCount = static_cast<uint64_t>(batch) *
                                    static_cast<uint64_t>(vShape.GetDim(DIM_H)) *
                                    static_cast<uint64_t>(qShape.GetDim(DIM_T));
    const uint64_t qgScaledOffset = 0;
    const uint64_t qgScaledBytes = tokenHeadCount * static_cast<uint64_t>(qShape.GetDim(DIM_D)) * dataBytes;
    const uint64_t phaseBaseOffset = AlignWorkspace(qgScaledBytes);
    const uint64_t wSeedOffset = phaseBaseOffset;
    const uint64_t wSeedBytes = qgScaledBytes;
    const uint64_t uSeedOffset = AlignWorkspace(wSeedOffset + wSeedBytes);
    const uint64_t uSeedBytes = tokenHeadCount * static_cast<uint64_t>(vShape.GetDim(DIM_D)) * dataBytes;
    const uint64_t aqkFp32Offset = AlignWorkspace(uSeedOffset + uSeedBytes);
    const uint64_t matrixFp32Bytes = tokenHeadCount * static_cast<uint64_t>(chunkSize) * KDA_FP32_BYTES;
    const uint64_t akkFp32Offset = AlignWorkspace(aqkFp32Offset + matrixFp32Bytes);
    const uint64_t prepareScratchOffset = AlignWorkspace(akkFp32Offset + matrixFp32Bytes);
    const uint64_t solveScratch = usedCoreNum * KDA_SOLVE_SCRATCH_SLOTS *
                                  static_cast<uint64_t>(chunkSize) * static_cast<uint64_t>(chunkSize) *
                                  KDA_FP32_BYTES;
    const uint64_t scoreScratch = usedCoreNum * KDA_SCORE_QUEUE_SLOTS * KDA_SCORE_SCRATCH_PLANES *
                                  static_cast<uint64_t>(chunkSize) *
                                  static_cast<uint64_t>(qShape.GetDim(DIM_D)) * dataBytes;
    const uint64_t prepareEnd = prepareScratchOffset + AlignWorkspace(solveScratch) + scoreScratch;

    const uint64_t postScratchOffset = AlignWorkspace(uSeedOffset + uSeedBytes);
    const uint64_t postScratchBytes = static_cast<uint64_t>(batch) *
                                      static_cast<uint64_t>(vShape.GetDim(DIM_H)) *
                                      static_cast<uint64_t>(totalChunks) * static_cast<uint64_t>(chunkSize) *
                                      static_cast<uint64_t>(qShape.GetDim(DIM_D)) * KDA_FP32_BYTES;
    const uint64_t postEnd = postScratchOffset + postScratchBytes;

    ChunkGatedDeltaRuleFwdHTilingContext fwdHContext{};
    fwdHContext.seqlen = qShape.GetDim(DIM_T);
    // POST_WU expands kg from H_k to H_v before state propagation.
    fwdHContext.kNumHead = vShape.GetDim(DIM_H);
    fwdHContext.kHeadDim = qShape.GetDim(DIM_D);
    fwdHContext.vNumHead = vShape.GetDim(DIM_H);
    fwdHContext.vHeadDim = vShape.GetDim(DIM_D);
    fwdHContext.shapeBatchDim = batch;
    fwdHContext.hasCuSeqlens = isVarLen;
    fwdHContext.cuSeqlensDim0 = isVarLen ? seqNum + 1 : 0;
    fwdHContext.dataType = KdaDtypeToEnum(qDesc->GetDataType());
    fwdHContext.gDataType = KdaDtypeToEnum(context->GetInputDesc(INPUT_GK_IDX)->GetDataType());
    fwdHContext.useInitialState = hasInitialState;
    fwdHContext.stateDataType = GDN_FWD_H_DTYPE_FP32;
    fwdHContext.useGk = true;
    fwdHContext.storeFinalState = outputFinalState;
    fwdHContext.chunkSize = chunkSize;
    fwdHContext.aicCoreNum = blockDim;
    // GetUserWorkspace() already skips the system workspace in the fused kernel.
    fwdHContext.libApiWorkSpaceSize = 0;
    ::ChunkGatedDeltaRuleFwdHTilingData fwdHTiling{};
    uint32_t fwdHBlockDim = 0;
    size_t fwdHWorkspaceBytes = 0;
    ChunkGatedDeltaRuleFwdHTilingProcessor(fwdHContext).Process(
        fwdHTiling, fwdHBlockDim, fwdHWorkspaceBytes);
    if (fwdHBlockDim != blockDim) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t fwdHEnd = phaseBaseOffset + static_cast<uint64_t>(fwdHWorkspaceBytes);

    const uint64_t outputElements = tokenHeadCount * static_cast<uint64_t>(vShape.GetDim(DIM_D));
    const uint64_t outputScratchOffset = phaseBaseOffset;
    const uint64_t outputEnd = outputScratchOffset + 2 * outputElements * KDA_FP32_BYTES;
    const uint64_t userWorkspaceBytes = AlignWorkspace(std::max({prepareEnd, postEnd, fwdHEnd, outputEnd}));
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + userWorkspaceBytes;

    tiling.set_batch(batch);
    tiling.set_seqNum(seqNum);
    tiling.set_qHeadNum(qShape.GetDim(DIM_H));
    tiling.set_vHeadNum(vShape.GetDim(DIM_H));
    tiling.set_seqlen(qShape.GetDim(DIM_T));
    tiling.set_kHeadDim(qShape.GetDim(DIM_D));
    tiling.set_vHeadDim(vShape.GetDim(DIM_D));
    tiling.set_chunkSize(chunkSize);
    tiling.set_totalChunks(totalChunks);
    tiling.set_scale(scale);
    tiling.set_hasInitialState(hasInitialState);
    tiling.set_outputFinalState(outputFinalState);
    tiling.set_isVarLen(isVarLen);
    tiling.set_safeGate(safeGate);
    const int64_t stageUsedCoreNum = static_cast<int64_t>(usedCoreNum);
    const ChunkKdaPrepareTilingData prepareTiling{
        stageUsedCoreNum, static_cast<int64_t>(qgScaledOffset), static_cast<int64_t>(wSeedOffset),
        static_cast<int64_t>(uSeedOffset), static_cast<int64_t>(aqkFp32Offset),
        static_cast<int64_t>(akkFp32Offset), static_cast<int64_t>(prepareScratchOffset)};
    const ChunkKdaPostWuTilingData postWuTiling{
        stageUsedCoreNum, static_cast<int64_t>(qgScaledOffset), static_cast<int64_t>(wSeedOffset),
        static_cast<int64_t>(uSeedOffset), static_cast<int64_t>(postScratchOffset)};
    const ChunkKdaFwdHStageTilingData fwdHStageTiling{
        fwdHTiling.batch, fwdHTiling.seqlen, fwdHTiling.kNumHead, fwdHTiling.vNumHead,
        fwdHTiling.kHeadDim, fwdHTiling.vHeadDim, fwdHTiling.chunkSize,
        fwdHTiling.useInitialState, fwdHTiling.storeFinalState, fwdHTiling.isVariedLen,
        fwdHTiling.shapeBatch, fwdHTiling.tokenBatch, fwdHTiling.vWorkspaceOffset,
        fwdHTiling.vUpdateWorkspaceOffset, fwdHTiling.kDecayWorkspaceOffset,
        fwdHTiling.hWorkspaceOffset, fwdHTiling.numSeqWorkspaceOffset,
        fwdHTiling.numChunksWorkspaceOffset, static_cast<int64_t>(phaseBaseOffset)};
    const ChunkKdaOutputTilingData outputTiling{
        stageUsedCoreNum, static_cast<int64_t>(qgScaledOffset), static_cast<int64_t>(outputScratchOffset)};

    tiling.set_prepareUsedCoreNum(prepareTiling.usedCoreNum);
    tiling.set_prepareQgScaledOffset(prepareTiling.qgScaledOffset);
    tiling.set_prepareWSeedOffset(prepareTiling.wSeedOffset);
    tiling.set_prepareUSeedOffset(prepareTiling.uSeedOffset);
    tiling.set_prepareAqkFp32Offset(prepareTiling.aqkFp32Offset);
    tiling.set_prepareAkkFp32Offset(prepareTiling.akkFp32Offset);
    tiling.set_prepareScratchOffset(prepareTiling.scratchOffset);
    tiling.set_postWuUsedCoreNum(postWuTiling.usedCoreNum);
    tiling.set_postWuQgScaledOffset(postWuTiling.qgScaledOffset);
    tiling.set_postWuWSeedOffset(postWuTiling.wSeedOffset);
    tiling.set_postWuUSeedOffset(postWuTiling.uSeedOffset);
    tiling.set_postWuScratchOffset(postWuTiling.scratchOffset);
    tiling.set_fwdHBatch(fwdHStageTiling.batch);
    tiling.set_fwdHSeqlen(fwdHStageTiling.seqlen);
    tiling.set_fwdHKNumHead(fwdHStageTiling.kNumHead);
    tiling.set_fwdHVNumHead(fwdHStageTiling.vNumHead);
    tiling.set_fwdHKHeadDim(fwdHStageTiling.kHeadDim);
    tiling.set_fwdHVHeadDim(fwdHStageTiling.vHeadDim);
    tiling.set_fwdHChunkSize(fwdHStageTiling.chunkSize);
    tiling.set_fwdHUseInitialState(fwdHStageTiling.useInitialState);
    tiling.set_fwdHStoreFinalState(fwdHStageTiling.storeFinalState);
    tiling.set_fwdHIsVariedLen(fwdHStageTiling.isVariedLen);
    tiling.set_fwdHShapeBatch(fwdHStageTiling.shapeBatch);
    tiling.set_fwdHTokenBatch(fwdHStageTiling.tokenBatch);
    tiling.set_fwdHVWorkspaceOffset(fwdHStageTiling.vWorkspaceOffset);
    tiling.set_fwdHVUpdateWorkspaceOffset(fwdHStageTiling.vUpdateWorkspaceOffset);
    tiling.set_fwdHKDecayWorkspaceOffset(fwdHStageTiling.kDecayWorkspaceOffset);
    tiling.set_fwdHHWorkspaceOffset(fwdHStageTiling.hWorkspaceOffset);
    tiling.set_fwdHNumSeqWorkspaceOffset(fwdHStageTiling.numSeqWorkspaceOffset);
    tiling.set_fwdHNumChunksWorkspaceOffset(fwdHStageTiling.numChunksWorkspaceOffset);
    tiling.set_fwdHWorkspaceBaseOffset(fwdHStageTiling.workspaceBaseOffset);
    tiling.set_outputUsedCoreNum(outputTiling.usedCoreNum);
    tiling.set_outputQgScaledOffset(outputTiling.qgScaledOffset);
    tiling.set_outputScratchOffset(outputTiling.scratchOffset);

    context->SetTilingKey(1);
    context->SetScheduleMode(KDA_BATCH_MODE);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
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
