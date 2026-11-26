/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CHUNK_FWD_O_LAYOUT_H
#define CHUNK_FWD_O_LAYOUT_H

namespace GDN {
// Internal layout attribute shared by the fused ACLNN and FwdO tiling.
inline constexpr char kFusedA2BsndOutput[] = "BSND_A2_FUSED";
} // namespace GDN

#endif // CHUNK_FWD_O_LAYOUT_H
