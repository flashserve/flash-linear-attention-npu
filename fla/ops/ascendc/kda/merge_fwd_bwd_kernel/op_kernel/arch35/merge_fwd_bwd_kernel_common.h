/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 *
 * Ascend950 Mix 1:2 Mode-2 handshake (FFTS, both AIVs).
 */
#ifndef MERGE_FWD_BWD_KERNEL_ARCH35_COMMON_H
#define MERGE_FWD_BWD_KERNEL_ARCH35_COMMON_H

#include "../merge_fwd_bwd_kernel_common.h"

namespace MergeFwBwd {

__aicore__ inline uint16_t AivReadyFlag(uint16_t t)
{
    (void)t;
    return kChunkReadyFlag;
}

__aicore__ inline uint16_t AivFreeFlag(uint16_t t)
{
    (void)t;
    return kChunkFreeFlag;
}

__aicore__ inline uint16_t AivC1Flag(uint16_t t)
{
    (void)t;
    return kChunkC1Flag;
}

__aicore__ inline uint16_t AicReadyFlag(uint16_t t)
{
    (void)t;
    return kChunkReadyFlag;
}

__aicore__ inline uint16_t AicFreeFlag(uint16_t t)
{
    (void)t;
    return kChunkFreeFlag;
}

__aicore__ inline uint16_t AicC1Flag(uint16_t t)
{
    (void)t;
    return kChunkC1Flag;
}

template <pipe_t PIPE>
__aicore__ inline void AivWaitChunkFree(uint16_t t)
{
    AscendC::CrossCoreWaitFlag<kCrossCoreModeIndep, PIPE>(AivFreeFlag(t));
}

template <pipe_t PIPE>
__aicore__ inline void AivSetChunkReady(uint16_t t)
{
    AscendC::CrossCoreSetFlag<kCrossCoreModeIndep, PIPE>(AivReadyFlag(t));
}

template <pipe_t PIPE>
__aicore__ inline void AicWaitChunkReady(uint16_t t)
{
    AscendC::CrossCoreWaitFlag<kCrossCoreModeIndep, PIPE>(AicReadyFlag(t));
}

template <pipe_t PIPE>
__aicore__ inline void AicSetChunkFree(uint16_t t)
{
    AscendC::CrossCoreSetFlag<kCrossCoreModeIndep, PIPE>(AicFreeFlag(t));
}

template <pipe_t PIPE>
__aicore__ inline void AivWaitChunkC1(uint16_t t)
{
    AscendC::CrossCoreWaitFlag<kCrossCoreModeIndep, PIPE>(AivC1Flag(t));
}

template <pipe_t PIPE>
__aicore__ inline void AicSetChunkC1Done(uint16_t t)
{
    AscendC::CrossCoreSetFlag<kCrossCoreModeIndep, PIPE>(AicC1Flag(t));
}

} // namespace MergeFwBwd

#endif
