/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CHUNK_BWD_DV_LOCAL_TILING_A5_H
#define CHUNK_BWD_DV_LOCAL_TILING_A5_H

#include "../../chunk_bwd_dv_local_tiling_processor.h"

namespace optiling {

class ChunkBwdDvLocalTilingA5 {
public:
    bool SetTiling(gert::TilingContext *context);

private:
    void PrintTilingData(gert::TilingContext *context, const GDN::ChunkBwdDvLocalTilingData &tiling);
};

} // namespace optiling

#endif // CHUNK_BWD_DV_LOCAL_TILING_A5_H
