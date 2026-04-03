/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file all_gather_add_tiling.cc
 * \brief
 */

#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "hccl/hccl_types.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/all_gather_add_tiling.h"

using namespace AscendC;
using namespace ge;

namespace {
    constexpr uint32_t TILE_NUM = 1;
    constexpr uint32_t COMM_TURN = 1;
}
namespace optiling {

static ge::graphStatus AllGatherParamsCheck(const gert::TilingContext* context)
{
    const gert::StorageShape* aShape = context->GetInputShape(0);
    uint64_t inputDim0 = aShape->GetStorageShape().GetDim(0);
    uint64_t inputDim1 = aShape->GetStorageShape().GetDim(1);

    OP_CHECK_IF(inputDim0 == 0 || inputDim1 == 0,
                OP_LOGE(context->GetNodeName(), "the value is invalid"), return ge::GRAPH_FAILED);
    
    if (context->GetAttrs() == nullptr) {
        OP_LOGE(context->GetNodeName(), "get attrs failed");
        return ge::GRAPH_FAILED;
    } 
    auto group = context->GetAttrs()->GetAttrPointer<char>(static_cast<int>(0));
    OP_CHECK_IF(group == nullptr, OP_LOGE(context->GetNodeName(), "group is nullptr. "),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 设置hccl高阶API Tiling结构体
static void InitHcclParam(AllGatherAddTilingData* tilingData, const char* group)
{
    std::string algConfig = "AllGather=level0:fullmesh";
    Mc2CcTilingConfig mc2CcTilingConfig(group, HCCL_CMD_ALLGATHER, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
}

static ge::graphStatus AllGatherAddTilingFunc(gert::TilingContext *context) {
    // 对参数进行校验
    OP_CHECK_IF(AllGatherParamsCheck(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "param is invalid"), return ge::GRAPH_FAILED);
    
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(ascendcPlatform.GetCoreNumAiv());

    // 设置TilingData
    AllGatherAddTilingData* tilingData = context->GetTilingData<AllGatherAddTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);

    tilingData->commTurn = COMM_TURN;
    tilingData->tileNum = TILE_NUM;
    tilingData->totalElemNum = context->GetInputTensor(1)->GetShapeSize();
    tilingData->blockElemNum = tilingData->totalElemNum / context->GetBlockDim();
    tilingData->addTileElemNum = tilingData->blockElemNum / tilingData->tileNum;
    uint32_t rankSize = *context->GetAttrs()->GetAttrPointer<uint32_t>(static_cast<int>(1));
    tilingData->gatherTileElemNum = tilingData->totalElemNum / rankSize;
    
    // 设置workspaceSize gather out需要额外的临时内存，大小与b输入一致
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context,currentWorkspace);
    // 如需使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace大小
    uint32_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // 预留18M + gather_out
    auto dataType = context->GetInputTensor(0)->GetDataType();
    currentWorkspace[0] = sysWorkSpaceSize + tilingData->totalElemNum * sizeof(dataType);

    auto group = context->GetAttrs()->GetAttrPointer<char>(static_cast<int>(0));
    InitHcclParam(tilingData, group);
    return ge::GRAPH_SUCCESS;
}

struct AllGatherAddCompileInfo {};

static ge::graphStatus TilingParseForAllGatherAdd([[maybe_unused]] gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AllGatherAdd)
    .Tiling(AllGatherAddTilingFunc)
    .TilingParse<AllGatherAddCompileInfo>(TilingParseForAllGatherAdd);
}  // namespace optiling