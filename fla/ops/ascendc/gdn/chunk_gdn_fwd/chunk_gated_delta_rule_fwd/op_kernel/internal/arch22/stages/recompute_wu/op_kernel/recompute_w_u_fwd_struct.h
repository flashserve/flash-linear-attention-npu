/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_w_u_fwd_struct.h
 * \brief Shared tiling data for recompute_w_u_fwd.
 */

#ifndef GDN_MEGA_ARCH22_RECOMPUTE_W_U_FWD_STRUCT_H
#define GDN_MEGA_ARCH22_RECOMPUTE_W_U_FWD_STRUCT_H

#include <cstdint>

namespace GDN {

constexpr uint32_t RECOMPUTE_WU_SYNC_WINDOW_TASKS = 4;

// 栈上传递的私有小批任务描述，不改变RecomputeWUFwdTilingData的GM布局。
struct RecomputeTaskRange {
    uint64_t begin;
    uint64_t end;
    uint64_t capacity;
    // 仅 owned FrontBatch 使用：全部 RHS 已按最终 Solve 窗口写入本批独立 GM。
    bool fullyPrepared = false;
};

struct GdnMegaArch22RecomputeWUTilingData {
    int64_t B;
    int64_t Hk;
    int64_t Hv;
    int64_t hvPerHk;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t chunkNum;
    int64_t chunkSize;
    int64_t interleavedMembaseVecRow;
    int64_t isVariable;
};

} // namespace GDN

#endif // GDN_MEGA_ARCH22_RECOMPUTE_W_U_FWD_STRUCT_H
