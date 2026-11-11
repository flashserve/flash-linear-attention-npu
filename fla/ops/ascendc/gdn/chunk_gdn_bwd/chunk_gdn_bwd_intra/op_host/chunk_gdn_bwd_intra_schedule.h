/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_SCHEDULE_H
#define CHUNK_GDN_BWD_INTRA_SCHEDULE_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace optiling {

struct IntraSchedule {
    int64_t cg = 0;
    int64_t hvSlices = 0;
    int64_t workCount = 0;
    uint32_t blockDim = 0;
    size_t workspaceBytes = 0;
};

inline bool IntraCheckedMultiply(uint64_t a, uint64_t b, uint64_t &result)
{
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b) {
        return false;
    }
    result = a * b;
    return true;
}

inline bool BuildIntraSchedule(int64_t chunks, int64_t hv, int64_t hk,
                              uint32_t cores, bool arch22, size_t systemBytes,
                              IntraSchedule &result)
{
    if (chunks <= 0 || hv <= 0 || hk <= 0 || cores == 0 || hv % hk != 0 || hv / hk > 4) {
        return false;
    }
    IntraSchedule candidate;
    candidate.cg = hv / hk == 3 ? 3 : 4;
    candidate.hvSlices = hv / candidate.cg + (hv % candidate.cg != 0);
    uint64_t work = 0;
    if (!IntraCheckedMultiply(static_cast<uint64_t>(chunks), candidate.hvSlices, work) ||
        work > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    candidate.workCount = static_cast<int64_t>(work);
    candidate.blockDim = static_cast<uint32_t>(std::min<uint64_t>(work, cores));
    uint64_t userBytes = 0;
    if (arch22 && !IntraCheckedMultiply(candidate.blockDim, candidate.cg * 49152ULL, userBytes)) {
        return false;
    }
    if (userBytes > std::numeric_limits<size_t>::max() - systemBytes) {
        return false;
    }
    candidate.workspaceBytes = static_cast<size_t>(userBytes) + systemBytes;
    result = candidate;
    return true;
}

} // namespace optiling
#endif
