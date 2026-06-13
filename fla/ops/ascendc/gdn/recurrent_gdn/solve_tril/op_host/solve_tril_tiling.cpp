/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file solve_tril_tiling.cpp
 * \brief Tiling computation for SolveTril operator (MCH_ONLY branch)
 */

#include "solve_tril_tiling.h"
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/solve_tril_common.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr size_t WORKSPACE_NUM = 1;
constexpr uint32_t WS_SYS_SIZE = 0U;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SolveTrilTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get input shape
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto shape = inputShape->GetStorageShape();
    int64_t dimNum = shape.GetDimNum();

    // Determine batch and matrix dimension
    int64_t batchSize = 1;
    int64_t n = 0;
    if (dimNum == 2) {
        n = shape.GetDim(0);
    } else if (dimNum == 3) {
        batchSize = shape.GetDim(0);
        n = shape.GetDim(1);
    } else {
        OP_LOGE(context, "SolveTril: unsupported rank %ld, expected 2 or 3", dimNum);
        return ge::GRAPH_FAILED;
    }

    // Validate n is multiple of LEAF_BLOCK_SIZE and supported
    if (n == 0 || n % LEAF_BLOCK_SIZE != 0) {
        OP_LOGE(context, "SolveTril: n=%ld must be positive multiple of %u", n, LEAF_BLOCK_SIZE);
        return ge::GRAPH_FAILED;
    }
    if (n != 16 && n != 32 && n != 64 && n != 128) {
        OP_LOGE(context, "SolveTril: n=%ld not supported, must be 16/32/64/128", n);
        return ge::GRAPH_FAILED;
    }

    // Get input dtype
    auto inputDtype = context->GetInputDesc(0)->GetDataType();

    int64_t numLeafBlocks = n / LEAF_BLOCK_SIZE;
    int64_t mbhLevels = 0;
    if (n == 32) {
        mbhLevels = 1;
    } else if (n == 64) {
        mbhLevels = 2;
    } else if (n == 128) {
        mbhLevels = 3;
    }

    // 3. Multi-core split
    int64_t totalTasks;
    if (mbhLevels == 0) {
        totalTasks = batchSize * numLeafBlocks;
    } else {
        totalTasks = batchSize;
    }
    int64_t usedCoreNum = totalTasks < coreNum ? totalTasks : coreNum;
    if (usedCoreNum == 0) usedCoreNum = 1;
    int64_t taskPerCore = CeilDiv(totalTasks, usedCoreNum);

    // 4. Set workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    size_t wsSize = WS_SYS_SIZE;
    if (mbhLevels >= 2) {
        size_t mainSize = static_cast<size_t>(n) * n * sizeof(float);
        wsSize = static_cast<size_t>(batchSize) * mainSize;
    }
    currentWorkspace[0] = wsSize;

    // 5. Fill TilingData
    SolveTrilTilingData* tiling = context->GetTilingData<SolveTrilTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->n = n;
    tiling->batchSize = batchSize;
    tiling->numLeafBlocks = numLeafBlocks;
    tiling->mbhLevels = mbhLevels;
    tiling->blockDim = usedCoreNum;
    tiling->taskPerCore = taskPerCore;
    tiling->workspaceOffset = 0LL;
    tiling->reserved = 0LL;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 6. Set TilingKey: D_TYPE + MBH_LEVELS
    uint32_t dType = static_cast<uint32_t>(inputDtype);
    ASCENDC_TPL_SEL_PARAM(context, dType, static_cast<uint32_t>(mbhLevels));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSolveTril([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SolveTrilCompileInfo {};

IMPL_OP_OPTILING(SolveTril)
    .Tiling(SolveTrilTilingFunc)
    .TilingParse<SolveTrilCompileInfo>(TilingParseForSolveTril);

} // namespace optiling
