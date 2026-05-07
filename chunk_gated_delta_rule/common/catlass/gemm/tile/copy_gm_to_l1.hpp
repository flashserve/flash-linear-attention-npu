/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_COPY_GM_TO_L1_HPP
#define CATLASS_GEMM_TILE_COPY_GM_TO_L1_HPP

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
#include "catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp"
#endif
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
#include "catlass/gemm/tile/ascend950/copy_gm_to_l1.hpp"
#endif

#endif
