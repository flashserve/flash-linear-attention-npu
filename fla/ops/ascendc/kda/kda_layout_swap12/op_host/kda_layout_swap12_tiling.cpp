/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include "kda_layout_swap12_tiling.h"
#include <algorithm>
#include <limits>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t OUTPUT_Y_IDX = 0;
constexpr size_t DIM_B = 0;
constexpr size_t DIM_FIRST = 1;
constexpr size_t DIM_SECOND = 2;

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
} // namespace

ge::graphStatus Tiling4KdaLayoutSwap12(gert::TilingContext *context)
{
    if (context == nullptr || context->GetRequiredInputShape(INPUT_X_IDX) == nullptr ||
        context->GetOutputShape(OUTPUT_Y_IDX) == nullptr || context->GetInputDesc(INPUT_X_IDX) == nullptr ||
        context->GetOutputDesc(OUTPUT_Y_IDX) == nullptr || context->GetRawTilingData() == nullptr ||
        context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    KdaLayoutSwap12TilingData tiling;
    auto xShape = context->GetRequiredInputShape(INPUT_X_IDX)->GetStorageShape();
    auto yShape = context->GetOutputShape(OUTPUT_Y_IDX)->GetStorageShape();
    auto xDesc = context->GetInputDesc(INPUT_X_IDX);
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_IDX);
    if (xShape.GetDimNum() < 3 || yShape.GetDimNum() != xShape.GetDimNum() ||
        (xDesc->GetDataType() != ge::DT_FLOAT && xDesc->GetDataType() != ge::DT_FLOAT16 &&
         xDesc->GetDataType() != ge::DT_BF16) || yDesc->GetDataType() != xDesc->GetDataType()) {
        return ge::GRAPH_FAILED;
    }
    for (size_t idx = 0; idx < xShape.GetDimNum(); ++idx) {
        if (xShape.GetDim(idx) <= 0) {
            return ge::GRAPH_FAILED;
        }
        const size_t expectedInputIdx = xShape.GetDimNum() == 3 ?
            (idx == 0 ? 1 : idx == 1 ? 0 : idx) : (idx == 1 ? 2 : idx == 2 ? 1 : idx);
        if (yShape.GetDim(idx) != xShape.GetDim(expectedInputIdx)) {
            return ge::GRAPH_FAILED;
        }
    }

    int64_t batch = xShape.GetDim(DIM_B);
    int64_t firstDim = xShape.GetDim(DIM_FIRST);
    int64_t secondDim = xShape.GetDim(DIM_SECOND);
    int64_t tailDim = 1;
    for (size_t idx = 3; idx < xShape.GetDimNum(); ++idx) {
        if (tailDim > std::numeric_limits<int64_t>::max() / xShape.GetDim(idx)) {
            return ge::GRAPH_FAILED;
        }
        tailDim *= xShape.GetDim(idx);
    }

    if (xShape.GetDimNum() == 3) {
        batch = 1;
        firstDim = xShape.GetDim(0);
        secondDim = xShape.GetDim(1);
        tailDim = xShape.GetDim(2);
    }

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    if (batch > std::numeric_limits<int64_t>::max() / firstDim ||
        batch * firstDim > std::numeric_limits<int64_t>::max() / secondDim) {
        return ge::GRAPH_FAILED;
    }
    int64_t rowCount = batch * firstDim * secondDim;
    uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(rowCount, coreNum));
    context->SetBlockDim(blockDim);

    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    tiling.set_batch(batch);
    tiling.set_firstDim(firstDim);
    tiling.set_secondDim(secondDim);
    tiling.set_tailDim(tailDim);
    tiling.set_dataType(DTypeCode(xDesc->GetDataType()));
    tiling.set_usedCoreNum(blockDim);

    if (xDesc->GetDataType() == ge::DT_FLOAT) {
        context->SetTilingKey(0);
    } else if (xDesc->GetDataType() == ge::DT_BF16) {
        context->SetTilingKey(1);
    } else {
        context->SetTilingKey(2);
    }
    auto rawTilingData = context->GetRawTilingData();
    if (rawTilingData->GetData() == nullptr || rawTilingData->GetCapacity() < tiling.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4KdaLayoutSwap12(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KdaLayoutSwap12)
    .Tiling(Tiling4KdaLayoutSwap12)
    .TilingParse<KdaLayoutSwap12CompileInfo>(TilingPrepare4KdaLayoutSwap12);

} // namespace optiling
