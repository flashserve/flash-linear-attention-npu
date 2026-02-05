/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prepare_wy_repr_bwd_da_tiling.cpp
 * \brief
 */
#include "prepare_wy_repr_bwd_da_tiling.h"
 #include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_type.h"

namespace optiling {
ASCENDC_EXTERN_C ge::graphStatus TilingWyReprBwdDa(gert::TilingContext* context) {
    int64_t B = 1;
    int64_t T = 2048;
    int64_t H = 4;
    int64_t V = 128;
    int64_t K = 128;
    int64_t BT = 64;
    context->SetTilingKey(1);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t userWorkspaceSize = B * T * H * BT * 2 + B * T * H * V;
    // printf("[TilingWyReprBwdDa]---hyh---\n");
    // auto baseM = tiling.mmTilingData.get_baseM();
    // auto baseN = tiling.mmTilingData.get_baseN();
    // uint32_t userWorkspaceSize = baseM * baseN * FP32_DATATYPE_SIZE * aicNum * AIC_AIV_RATION;
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = static_cast<size_t>(sysWorkspaceSize + userWorkspaceSize);
    const int64_t aicNum = ascendcPlatform.GetCoreNumAic();
    context->SetBlockDim(aicNum);
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForWyReprBwdDA(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PrepareWyReprBwdDa)
    .Tiling(TilingWyReprBwdDa);
} // namespace optiling
