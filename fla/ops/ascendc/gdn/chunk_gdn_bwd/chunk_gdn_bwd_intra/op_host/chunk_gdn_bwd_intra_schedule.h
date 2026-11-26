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

#include "../op_kernel/arch22/chunk_gdn_bwd_intra_config.h"
#include "../op_kernel/arch92/chunk_gdn_bwd_intra_common.h"

namespace optiling {

// A2/A3 and A5 support runtime GVA up to 16; A6 uses the fixed GVA keys.
constexpr int64_t INTRA_MAX_GVA_RATIO_ARCH22 = 16;
constexpr int64_t INTRA_MAX_GVA_RATIO_ARCH35 = 16;
constexpr int64_t INTRA_MAX_GVA_RATIO_ARCH92 = 4;

constexpr int64_t IntraMaxGvaRatio(bool arch22, bool arch92 = false)
{
    if (arch92) {
        return INTRA_MAX_GVA_RATIO_ARCH92;
    }
    return arch22 ? INTRA_MAX_GVA_RATIO_ARCH22 : INTRA_MAX_GVA_RATIO_ARCH35;
}

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
                              uint32_t cores, bool arch22, bool arch92, size_t systemBytes,
                              IntraSchedule &result, bool fp32Score)
{
    if (chunks <= 0 || hv <= 0 || hk <= 0 || cores == 0 || hv % hk != 0 ||
        hv / hk < 1 || hv / hk > IntraMaxGvaRatio(arch22, arch92)) {
        return false;
    }
    IntraSchedule candidate;
    candidate.cg = arch92 ? GDN::Arch92::SLOT_COUNT : (hv / hk == 3 ? 3 : 4);
    candidate.hvSlices = hv / candidate.cg + (hv % candidate.cg != 0);
    uint64_t work = 0;
    if (!IntraCheckedMultiply(static_cast<uint64_t>(chunks), candidate.hvSlices, work) ||
        work > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    candidate.workCount = static_cast<int64_t>(work);
    candidate.blockDim = static_cast<uint32_t>(std::min<uint64_t>(work, cores));
    uint64_t userBytes = 0;
    const uint32_t recordBytes = fp32Score ? GDN::Arch22::WorkspaceLayout<sizeof(float)>::RECORD_BYTES :
        GDN::Arch22::WorkspaceLayout<sizeof(uint16_t)>::RECORD_BYTES;
    if (arch22 && !IntraCheckedMultiply(candidate.blockDim,
        static_cast<uint64_t>(candidate.cg) * recordBytes, userBytes)) {
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
