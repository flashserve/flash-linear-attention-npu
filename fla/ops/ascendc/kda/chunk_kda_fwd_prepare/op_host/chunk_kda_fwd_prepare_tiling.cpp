/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "chunk_kda_fwd_prepare_tiling.h"

#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_Q_IDX = 0;
constexpr size_t INPUT_V_IDX = 2;
constexpr size_t INPUT_INITIAL_IDX = 5;
constexpr size_t INPUT_CU_SEQLENS_IDX = 6;
constexpr size_t INPUT_CHUNK_INDICES_IDX = 7;
constexpr size_t ATTR_SCALE_IDX = 0;
constexpr size_t ATTR_CHUNK_SIZE_IDX = 1;
constexpr size_t ATTR_TOTAL_CHUNKS_IDX = 3;
constexpr size_t ATTR_SAFE_GATE_IDX = 4;
constexpr uint64_t KDA_SOLVE_SCRATCH_SLOTS = 5;
constexpr uint64_t KDA_SCORE_QUEUE_SLOTS = 2;
constexpr uint64_t KDA_SCORE_SCRATCH_PLANES = 3;
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

bool ResolveSequenceInfo(gert::TilingContext *context, int64_t chunkSize, int64_t totalChunks,
                         int64_t batch, bool &isVarLen, int64_t &seqNum)
{
    isVarLen = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX) != nullptr;
    seqNum = batch;
    if (!isVarLen) {
        return true;
    }
    auto cuTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX);
    auto chunkMetadata = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
    seqNum = cuTensor->GetStorageShape().GetDim(0) - 1;
    if (seqNum <= 0 || chunkMetadata == nullptr ||
        chunkMetadata->GetStorageShape().GetShapeSize() != totalChunks * 4) {
        return false;
    }
    const int64_t *cu = cuTensor->GetData<int64_t>();
    if (cu == nullptr) {
        return false;
    }
    int64_t chunkCount = 0;
    for (int64_t seq = 0; seq < seqNum; ++seq) {
        if (cu[seq] < 0 || cu[seq + 1] < cu[seq]) {
            return false;
        }
        chunkCount += (cu[seq + 1] - cu[seq] + chunkSize - 1) / chunkSize;
    }
    return chunkCount == totalChunks;
}
} // namespace

ge::graphStatus Tiling4ChunkKdaFwdPrepare(gert::TilingContext *context)
{
    auto qDesc = context->GetInputDesc(INPUT_Q_IDX);
    auto attrPtr = context->GetAttrs();
    if (qDesc == nullptr || attrPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto qShape = context->GetOptionalInputShape(INPUT_Q_IDX)->GetStorageShape();
    auto vShape = context->GetOptionalInputShape(INPUT_V_IDX)->GetStorageShape();
    const float scale = static_cast<float>(*(attrPtr->GetAttrPointer<double>(ATTR_SCALE_IDX)));
    const int64_t chunkSize = *(attrPtr->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX));
    const int64_t totalChunks = *(attrPtr->GetAttrPointer<int64_t>(ATTR_TOTAL_CHUNKS_IDX));
    const bool safeGate = *(attrPtr->GetAttrPointer<bool>(ATTR_SAFE_GATE_IDX));
    const int64_t batch = qShape.GetDim(DIM_B);
    bool isVarLen = false;
    int64_t seqNum = 0;
    if (!ResolveSequenceInfo(context, chunkSize, totalChunks, batch, isVarLen, seqNum)) {
        return ge::GRAPH_FAILED;
    }

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t blockDim = platform.GetCoreNumAic() == 0 ? 1 : platform.GetCoreNumAic();
    context->SetBlockDim(blockDim);
    const uint64_t dataBytes = qDesc->GetDataType() == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    const uint64_t tokenHeadCount = static_cast<uint64_t>(batch) *
                                    static_cast<uint64_t>(vShape.GetDim(DIM_H)) *
                                    static_cast<uint64_t>(qShape.GetDim(DIM_T));
    const uint64_t matrixBytes = tokenHeadCount * static_cast<uint64_t>(chunkSize) * sizeof(float);
    const uint64_t aqkFp32Offset = 0;
    const uint64_t akkFp32Offset = AlignWorkspace(aqkFp32Offset + matrixBytes);
    const uint64_t scratchOffset = AlignWorkspace(akkFp32Offset + matrixBytes);
    const uint64_t solveBytes = static_cast<uint64_t>(blockDim) * KDA_SOLVE_SCRATCH_SLOTS *
                                static_cast<uint64_t>(chunkSize) * static_cast<uint64_t>(chunkSize) * sizeof(float);
    const uint64_t scoreBytes = static_cast<uint64_t>(blockDim) * KDA_SCORE_QUEUE_SLOTS *
                                KDA_SCORE_SCRATCH_PLANES * static_cast<uint64_t>(chunkSize) *
                                static_cast<uint64_t>(qShape.GetDim(DIM_D)) * dataBytes;
    const uint64_t userWorkspaceBytes = AlignWorkspace(scratchOffset + AlignWorkspace(solveBytes) + scoreBytes);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() + userWorkspaceBytes;

    ChunkKdaFwdPrepareTilingData tiling;
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
    tiling.set_hasInitialState(context->GetOptionalInputTensor(INPUT_INITIAL_IDX) != nullptr);
    tiling.set_isVarLen(isVarLen);
    tiling.set_safeGate(safeGate);
    tiling.set_prepareUsedCoreNum(blockDim);
    tiling.set_prepareAqkFp32Offset(aqkFp32Offset);
    tiling.set_prepareAkkFp32Offset(akkFp32Offset);
    tiling.set_prepareScratchOffset(scratchOffset);

    context->SetTilingKey(1);
    context->SetScheduleMode(KDA_BATCH_MODE);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaFwdPrepare(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaFwdPrepare)
    .Tiling(Tiling4ChunkKdaFwdPrepare)
    .TilingParse<ChunkKdaFwdPrepareCompileInfo>(TilingPrepare4ChunkKdaFwdPrepare);

} // namespace optiling
