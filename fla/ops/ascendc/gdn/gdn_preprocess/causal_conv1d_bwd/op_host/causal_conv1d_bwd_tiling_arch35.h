/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_bwd_tiling_arch35.h
 * \brief Causal Conv1D backward A5(arch35/DAV_3510) dedicated tiling interface
 *
 * A5 的 tiling 与 kernel 均已独立实现：本文件声明 A5 tiling 的入参结构体与入口函数，
 * 供共享 tiling 入口在检测到 DAV_3510 时调用。
 */

#ifndef ASCEND_OPS_CAUSAL_CONV1D_BWD_TILING_ARCH35_H
#define ASCEND_OPS_CAUSAL_CONV1D_BWD_TILING_ARCH35_H

#include <cstdint>
#include "exe_graph/runtime/tiling_context.h"
#include "../op_kernel/causal_conv1d_bwd_tiling_data.h"
#include "../op_kernel/causal_conv1d_bwd_tiling_key.h"

namespace optiling {

// A5 tiling 所需的、由共享入口完成解析与校验后的输入。
struct CausalConv1dBwdHostTilingArgs {
    gert::TilingContext *context = nullptr;
    uint32_t B = 0;
    uint32_t T = 0;
    uint32_t D = 0;
    uint32_t W = 0;
    uint32_t activation = 0;
    uint32_t inputMode = 0;
    uint32_t inputLayout = 0;
    uint32_t inputN = 1;
    uint32_t inputHeadDim = 0;
    uint32_t totalTokens = 0;
    bool isB16Input = false;
    bool isFp32Input = false;
    bool useInitialState = false;
    bool useFinalState = false;
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;              // 已扣除预留区并按 BYTE_BLOCK 对齐后的可用 UB
    uint64_t sysWorkspaceSize = 0;
};

// A5(arch35/DAV_3510) 专属 tiling：独立实现，负责选路与切分并写回 tiling 数据与 workspace。
ge::graphStatus CausalConv1dBwdTilingA5(const CausalConv1dBwdHostTilingArgs &args,
                                        CausalConv1dBwdTilingData *tiling);

} // namespace optiling

#endif // ASCEND_OPS_CAUSAL_CONV1D_BWD_TILING_ARCH35_H
