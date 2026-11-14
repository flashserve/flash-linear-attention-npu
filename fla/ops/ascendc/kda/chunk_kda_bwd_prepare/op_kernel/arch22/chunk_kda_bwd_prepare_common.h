#ifndef CHUNK_KDA_BWD_PREPARE_ARCH22_COMMON_H
#define CHUNK_KDA_BWD_PREPARE_ARCH22_COMMON_H
#include "../chunk_kda_bwd_prepare_utils.h"
namespace KDA {
constexpr uint32_t PREPARE_CHUNK = 64, PREPARE_DIM = 128;
constexpr uint32_t HEAD_OWNER_BYTES = 72 * 1024, L0_SLOT_BYTES = 32 * 1024;
constexpr uint32_t RAW_ELEMENTS = 64 * 64, RECORD_ELEMENTS = RAW_ELEMENTS;
constexpr uint32_t UB_BYTES = 80 * 1024;
constexpr uint8_t CROSS_MODE = 2;
constexpr uint16_t FREE_BASE = 0, READY_BASE = 4;
static_assert(5 * RAW_ELEMENTS * sizeof(float) == UB_BYTES, "UB regions must not overlap");
static_assert(4 * RECORD_ELEMENTS * sizeof(float) == 65536, "per-AIC workspace size");
static_assert(4 * HEAD_OWNER_BYTES <= 512 * 1024, "L1 capacity");
__aicore__ inline uint32_t Align16(uint32_t n) { return (n + 15) / 16 * 16; }
template <AscendC::HardEvent E>
__aicore__ inline void AllocateEvents(AscendC::TPipe *pipe, AscendC::TEventID *ids,
                                    uint32_t count, bool credit)
{
    for (uint32_t i = 0; i < count; ++i) {
        ids[i] = pipe->AllocEventID<E>();
        if (credit) AscendC::SetFlag<E>(ids[i]);
    }
}
template <AscendC::HardEvent E>
__aicore__ inline void ReleaseEvents(AscendC::TPipe *pipe, AscendC::TEventID *ids,
                                   uint32_t count, bool credit)
{
    for (uint32_t i = 0; i < count; ++i) {
        if (credit) AscendC::WaitFlag<E>(ids[i]);
        pipe->ReleaseEventID<E>(ids[i]);
    }
}
}
#endif
