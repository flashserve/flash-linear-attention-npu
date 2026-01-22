/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_bwd_dv_local_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <climits>
#include <register/tilingdata_base.h>
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include <tiling/tiling_api.h>
#include "err/ops_err.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ChunkGatedDeltaRuleBwdDhuTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);             // 物理总核数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkGatedDeltaRuleBwdDhu, ChunkGatedDeltaRuleBwdDhuTilingData)

struct ChunkGatedDeltaRuleBwdDhuCompileInfo {};

class ChunkGatedDeltaRuleBwdDhuTiling {
public:
    ChunkGatedDeltaRuleBwdDhuTilingData tilingData;
    bool Init(gert::TilingContext *context);
    bool CheckInputShape(gert::TilingContext *context);
    bool CalcUb(gert::TilingContext *context);
private:

    bool IS_SCALE = false;
    bool IS_VARIABLE_LEN = false; 
    uint32_t B = 0;
    uint32_t H = 0;
    uint32_t T = 0;
    uint32_t K = 0;
    uint32_t V = 0;
    uint32_t chunkNum = 0;
    uint32_t chunkSize = 64;
};

} // namespace optiling