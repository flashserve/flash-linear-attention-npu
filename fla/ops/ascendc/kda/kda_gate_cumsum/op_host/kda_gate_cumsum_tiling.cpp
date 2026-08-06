/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include "kda_gate_cumsum_tiling.h"
#include <algorithm>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_G_IDX = 0;
constexpr size_t INPUT_A_LOG_IDX = 1;
constexpr size_t INPUT_DT_BIAS_IDX = 2;
constexpr size_t INPUT_CU_SEQLENS_IDX = 3;
constexpr size_t ATTR_CHUNK_SIZE_IDX = 0;
constexpr size_t ATTR_USE_GATE_IDX = 1;
constexpr size_t ATTR_SAFE_GATE_IDX = 2;
constexpr size_t ATTR_LOWER_BOUND_IDX = 3;
constexpr size_t ATTR_LOGICAL_RANK_IDX = 4;
constexpr size_t ATTR_LOGICAL_BATCH_IDX = 5;
constexpr size_t ATTR_LOGICAL_SEQLEN_IDX = 6;
constexpr size_t ATTR_LOGICAL_HEADS_IDX = 7;
constexpr size_t ATTR_LOGICAL_HEAD_DIM_IDX = 8;
constexpr int64_t MAX_K_DIM = 256;
} // namespace

ge::graphStatus Tiling4KdaGateCumsum(gert::TilingContext *context)
{
    KdaGateCumsumTilingData tiling;
    auto gDesc = context->GetInputDesc(INPUT_G_IDX);
    auto attrs = context->GetAttrs();
    if (gDesc == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t rank = *attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_RANK_IDX);
    const int64_t batch = *attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_BATCH_IDX);
    const int64_t t = *attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_SEQLEN_IDX);
    const int64_t hv = *attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_HEADS_IDX);
    const int64_t k = *attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_HEAD_DIM_IDX);
    if ((rank != 3 && rank != 4) || batch <= 0 || t <= 0 || hv <= 0 || k <= 0 || k > MAX_K_DIM) {
        return ge::GRAPH_FAILED;
    }

    int64_t chunkSize = *attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX);
    bool useGate = *attrs->GetAttrPointer<bool>(ATTR_USE_GATE_IDX);
    bool safeGate = *attrs->GetAttrPointer<bool>(ATTR_SAFE_GATE_IDX);
    float lowerBound = *attrs->GetAttrPointer<float>(ATTR_LOWER_BOUND_IDX);

    const auto cuShape = context->GetOptionalInputShape(INPUT_CU_SEQLENS_IDX);
    int64_t hasCuSeqlens = (cuShape != nullptr) ? 1 : 0;
    int64_t seqNum = hasCuSeqlens ? (cuShape->GetStorageShape().GetDim(0) - 1) : batch;
    int64_t maxChunks = (t + chunkSize - 1) / chunkSize;
    // Dense input keeps chunk-level parallelism. Varlen owns one (sequence, head) pair and
    // iterates only that sequence's real chunks, avoiding a rectangular grid of empty tasks.
    int64_t taskCount = hasCuSeqlens ? seqNum * hv : batch * hv * maxChunks;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(taskCount, coreNum));
    context->SetBlockDim(blockDim == 0 ? 1 : blockDim);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    tiling.set_batch(batch);
    tiling.set_t(t);
    tiling.set_hv(hv);
    tiling.set_k(k);
    tiling.set_rank(rank);
    tiling.set_chunkSize(chunkSize);
    tiling.set_seqNum(seqNum);
    tiling.set_hasCuSeqlens(hasCuSeqlens);
    tiling.set_hasALog(context->GetOptionalInputDesc(INPUT_A_LOG_IDX) != nullptr ? 1 : 0);
    tiling.set_hasDtBias(context->GetOptionalInputDesc(INPUT_DT_BIAS_IDX) != nullptr ? 1 : 0);
    int64_t dataType = 0;
    if (gDesc->GetDataType() == ge::DT_FLOAT) {
        dataType = 2;
    } else if (gDesc->GetDataType() == ge::DT_BF16) {
        dataType = 1;
    }
    tiling.set_dataType(dataType);
    tiling.set_useGateInKernel(useGate ? 1 : 0);
    tiling.set_safeGate(safeGate ? 1 : 0);
    tiling.set_lowerBound(lowerBound);
    tiling.set_usedCoreNum(blockDim == 0 ? 1 : blockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4KdaGateCumsum(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KdaGateCumsum)
    .Tiling(Tiling4KdaGateCumsum)
    .TilingParse<KdaGateCumsumCompileInfo>(TilingPrepare4KdaGateCumsum);

} // namespace optiling
