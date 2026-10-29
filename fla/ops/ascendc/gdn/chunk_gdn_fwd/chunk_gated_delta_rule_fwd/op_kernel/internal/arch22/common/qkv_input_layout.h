/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_ARCH22_QKV_INPUT_LAYOUT_H
#define GDN_ARCH22_QKV_INPUT_LAYOUT_H

#include <cstdint>

namespace GDN {
// Schedulers keep their head-major logical offsets. Only raw Q/K/V GM loads
// convert them; W/U/H/vNew and all workspaces retain their existing layouts.
__aicore__ inline uint64_t QkvSequenceMajorOffset(
    uint64_t offset, uint64_t tokens, uint64_t heads, uint64_t channels)
{
    const uint64_t row = offset / channels;
    const uint64_t batchHead = row / tokens;
    return ((batchHead / heads * tokens + row % tokens) * heads +
            batchHead % heads) * channels + offset % channels;
}
} // namespace GDN
#endif
