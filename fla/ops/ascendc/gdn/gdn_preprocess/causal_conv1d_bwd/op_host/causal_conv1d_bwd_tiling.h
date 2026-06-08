/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_bwd_tiling.h
 * \brief Causal Conv1D backward tiling compile info
 */

#ifndef ASCEND_OPS_CAUSAL_CONV1D_BWD_TILING_H
#define ASCEND_OPS_CAUSAL_CONV1D_BWD_TILING_H

#include "../op_kernel/causal_conv1d_bwd_tiling_key.h"

namespace optiling {

struct CausalConv1dBwdCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;
};

} // namespace optiling

#endif // ASCEND_OPS_CAUSAL_CONV1D_BWD_TILING_H
