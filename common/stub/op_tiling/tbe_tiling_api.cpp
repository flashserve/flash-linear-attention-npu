/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "tbe_tiling_api.h"

using namespace optiling;

namespace optiling {
bool GetTbeTiling(const gert::TilingContext* context, Conv3dBpFilterV2RunInfo& runInfoForV2, Conv3dBackpropV2TBETilingData& tbeTilingForV2)
{
    (void)context;
    (void)runInfoForV2;
    (void)tbeTilingForV2;
    return true;
}
bool GetTbeTiling(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2,
    Conv3dBackpropV2TBETilingData& tbeTilingForV2, const optiling::OpTypeV2 opType)
{
    (void)context;
    (void)runInfoV2;
    (void)tbeTilingForV2;
    (void)opType;
    return true;
}

bool GetTbeTiling(gert::TilingContext* context, Conv3dBackpropV2TBETilingData& tbeTilingForV2, const optiling::OpTypeV2 opType)
{
    (void)context;
    (void)tbeTilingForV2;
    (void)opType;
    return true;
}
}