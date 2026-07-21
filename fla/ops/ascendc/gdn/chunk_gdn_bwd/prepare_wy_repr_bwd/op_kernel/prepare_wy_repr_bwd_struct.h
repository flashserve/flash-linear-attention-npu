/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_struct.h
 * \brief Shared tiling data for fused prepare_wy_repr_bwd.
 */

#ifndef PREPARE_WY_REPR_BWD_STRUCT_H
#define PREPARE_WY_REPR_BWD_STRUCT_H

#include <cstdint>

namespace GDN {

struct PrepareWyReprBwdTilingData {
    int64_t B;
    int64_t HV;
    int64_t HK;
    int64_t groupSize;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t chunkSize;
    int64_t chunkNum;
    int64_t chunkNumPerB;
    int64_t isVariable;

    int64_t workspaceBufferCount;
    int64_t workspaceSlotSize;
    int64_t workspaceCoreSize;

    int64_t kBytes;
    int64_t vBytes;
    int64_t mBytes;

    int64_t kbgOffset;
    int64_t vbOffset;
    int64_t kbetaOffset;
    int64_t dkbgOffset;
    int64_t dvbOffset;
    int64_t da1Offset;
    int64_t da2Offset;
    int64_t da4Offset;
    int64_t da5Offset;
    int64_t da6Offset;
    int64_t dOffset;
    int64_t dkbOffset;
    int64_t dkOffset;
    int64_t kktOffset;

    int64_t kVecRow;
    int64_t vVecRow;
    int64_t mVecRow;
    int64_t kktVecRow;
};

} // namespace GDN

#endif // PREPARE_WY_REPR_BWD_STRUCT_H
