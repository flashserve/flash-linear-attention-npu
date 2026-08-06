/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "chunk_kda_fwd_finalize_tiling.h"

#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_QG_SCALED_IDX = 0;
constexpr size_t INPUT_V_NEW_IDX = 2;
constexpr size_t INPUT_H_IDX = 3;
constexpr size_t INPUT_CU_SEQLENS_IDX = 4;
constexpr size_t INPUT_CHUNK_INDICES_IDX = 5;
constexpr size_t ATTR_CHUNK_SIZE_IDX = 0;
constexpr size_t ATTR_LOGICAL_BATCH_IDX = 1;
constexpr size_t ATTR_LOGICAL_SEQLEN_IDX = 2;
constexpr size_t ATTR_LOGICAL_Q_HEADS_IDX = 3;
constexpr size_t ATTR_LOGICAL_V_HEADS_IDX = 4;
constexpr size_t ATTR_LOGICAL_K_DIM_IDX = 5;
constexpr size_t ATTR_LOGICAL_V_DIM_IDX = 6;
constexpr size_t ATTR_LOGICAL_TOTAL_CHUNKS_IDX = 7;
constexpr uint64_t KDA_WORKSPACE_ALIGN = 512;
constexpr uint32_t KDA_BATCH_MODE = 1;

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

ge::graphStatus Tiling4ChunkKdaFwdFinalize(gert::TilingContext *context)
{
    auto attrPtr = context->GetAttrs();
    if (attrPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t chunkSize = *(attrPtr->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX));
    const int64_t batch = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_BATCH_IDX));
    const int64_t seqlen = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_SEQLEN_IDX));
    const int64_t qHeads = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_Q_HEADS_IDX));
    const int64_t vHeads = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_V_HEADS_IDX));
    const int64_t kDim = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_K_DIM_IDX));
    const int64_t vDim = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_V_DIM_IDX));
    const int64_t totalChunks = *(attrPtr->GetAttrPointer<int64_t>(ATTR_LOGICAL_TOTAL_CHUNKS_IDX));
    if (batch <= 0 || seqlen <= 0 || qHeads <= 0 || vHeads <= 0 ||
        kDim <= 0 || vDim <= 0 || totalChunks <= 0 || vHeads % qHeads != 0) {
        return ge::GRAPH_FAILED;
    }
    bool isVarLen = false;
    int64_t seqNum = 0;
    if (!ResolveSequenceInfo(context, chunkSize, totalChunks, batch, isVarLen, seqNum)) {
        return ge::GRAPH_FAILED;
    }

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t blockDim = platform.GetCoreNumAic() == 0 ? 1 : platform.GetCoreNumAic();
    context->SetBlockDim(blockDim);
    const uint64_t outputElements = static_cast<uint64_t>(batch) *
                                    static_cast<uint64_t>(vHeads) *
                                    static_cast<uint64_t>(seqlen) *
                                    static_cast<uint64_t>(vDim);
    const uint64_t scratchBytes = 2 * outputElements * sizeof(float);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() + AlignWorkspace(scratchBytes);

    ChunkKdaFwdFinalizeTilingData tiling;
    tiling.set_batch(batch);
    tiling.set_seqNum(seqNum);
    tiling.set_qHeadNum(qHeads);
    tiling.set_vHeadNum(vHeads);
    tiling.set_seqlen(seqlen);
    tiling.set_kHeadDim(kDim);
    tiling.set_vHeadDim(vDim);
    tiling.set_chunkSize(chunkSize);
    tiling.set_totalChunks(totalChunks);
    tiling.set_scale(1.0f);
    tiling.set_hasInitialState(false);
    tiling.set_isVarLen(isVarLen);
    tiling.set_outputUsedCoreNum(blockDim);
    tiling.set_outputScratchOffset(0);

    context->SetTilingKey(1);
    context->SetScheduleMode(KDA_BATCH_MODE);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaFwdFinalize(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaFwdFinalize)
    .Tiling(Tiling4ChunkKdaFwdFinalize)
    .TilingParse<ChunkKdaFwdFinalizeCompileInfo>(TilingPrepare4ChunkKdaFwdFinalize);

} // namespace optiling
