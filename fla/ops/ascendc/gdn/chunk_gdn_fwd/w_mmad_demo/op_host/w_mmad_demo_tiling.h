/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef WMMADDEMO_TILING_H
#define WMMADDEMO_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(WMmadDemoTilingData)
    TILING_DATA_FIELD_DEF(int64_t, dummy);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(WMmadDemo, WMmadDemoTilingData)

} // namespace optiling

#endif
