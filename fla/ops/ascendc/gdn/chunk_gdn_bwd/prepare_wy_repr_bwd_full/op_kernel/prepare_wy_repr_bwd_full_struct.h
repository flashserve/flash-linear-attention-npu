/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_full_struct.h
 * \brief Shared tiling data for prepare_wy_repr_bwd_full.
 */

#ifndef PREPARE_WY_REPR_BWD_FULL_STRUCT_H
#define PREPARE_WY_REPR_BWD_FULL_STRUCT_H

#include <cstdint>

namespace GDN {

struct PrepareWyReprBwdFullTilingData {
    int64_t B;
    int64_t HV;
    int64_t HK;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t chunkNum;
    int64_t chunkSize;
    int64_t fusedKVecRow;
    int64_t fusedVVecRow;
    int64_t fusedKktVecRow;
    int64_t workspaceSlotSize;
    int64_t workspaceBufferCount;
    int64_t usedCoreNum;
    int64_t isVariable;
};

} // namespace GDN

#endif // PREPARE_WY_REPR_BWD_FULL_STRUCT_H
