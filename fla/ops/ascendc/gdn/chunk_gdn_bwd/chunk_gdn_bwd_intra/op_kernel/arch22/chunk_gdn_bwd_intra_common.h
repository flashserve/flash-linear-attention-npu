/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_ARCH22_COMMON_H
#define CHUNK_GDN_BWD_INTRA_ARCH22_COMMON_H

#define CATLASS_ARCH 2201
#include "../chunk_gdn_bwd_intra_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "chunk_gdn_bwd_intra_config.h"

namespace GDN::Arch22 {
constexpr uint16_t INPUTS_READY = 4;
constexpr uint16_t INPUTS_FREE = 6;
constexpr uint16_t INPUT_SLOTS = 2;
static_assert(INPUTS_FREE + INPUT_SLOTS - 1 <= Catlass::Arch::FFTS_MAX_FLAG,
    "Cross-core flag budget exceeded");
static_assert(S_OFFSET + SCORE_ELEMENTS == RECORD_ELEMENTS, "Invalid workspace record");

template <AscendC::HardEvent EVENT>
__aicore__ inline void Fence()
{
    AscendC::SetFlag<EVENT>(0);
    AscendC::WaitFlag<EVENT>(0);
}

__aicore__ inline void Wait(uint16_t id)
{
    Catlass::Arch::CrossCoreFlag flag{id};
    Catlass::Arch::CrossCoreWaitFlag(flag);
}

template <pipe_t PIPE>
__aicore__ inline void Notify(uint16_t id)
{
    Catlass::Arch::CrossCoreFlag flag{id};
    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE>(flag);
}

__aicore__ inline uint64_t Record(int64_t core, int64_t cg, int64_t head)
{
    return (static_cast<uint64_t>(core) * cg + head) * RECORD_ELEMENTS;
}
} // namespace GDN::Arch22
#endif
