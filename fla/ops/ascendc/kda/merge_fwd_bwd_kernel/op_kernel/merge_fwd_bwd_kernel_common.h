/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef MERGE_FWD_BWD_KERNEL_COMMON_H
#define MERGE_FWD_BWD_KERNEL_COMMON_H

#include "merge_fwd_bwd_kernel_struct.h"
#include "kernel_operator.h"

namespace MergeFwBwd {

__aicore__ inline void CoreTaskRange(
    uint32_t coreIdx, uint32_t coreNum, uint64_t total, uint64_t &begin, uint64_t &end)
{
    if (coreNum == 0U || coreIdx >= coreNum || total == 0) {
        begin = 0;
        end = 0;
        return;
    }
    const uint64_t n = total / coreNum;
    const uint64_t r = total % coreNum;
    if (coreIdx < r) {
        begin = coreIdx * (n + 1U);
        end = begin + n + 1U;
    } else {
        begin = r * (n + 1U) + (coreIdx - r) * n;
        end = begin + n;
    }
}

__aicore__ inline int64_t SrcRank(int64_t rank, int64_t N, int64_t i, bool forward, int64_t S)
{
    if (S == 1 && N == 1) {
        return 0;
    }
    if (forward) {
        return rank - N + i;
    }
    return rank + N - i;
}

__aicore__ inline uint64_t AgHmHeadOffset(int64_t src, int64_t hv, int64_t h)
{
    return (static_cast<uint64_t>(src) * static_cast<uint64_t>(hv) + static_cast<uint64_t>(h)) *
           static_cast<uint64_t>(kKDim) * static_cast<uint64_t>(kRowStride);
}

// scratch [3, HV, 128, 128] FP32: bank 0 = H, bank 1/2 = M ping-pong.
// M of rank-step `step` (1-based, the M used by GEMM `step`) lives in bank (step-1)&1.
__aicore__ inline uint64_t ScratchHElemOff(uint64_t h)
{
    return h * static_cast<uint64_t>(kKvElems);
}

__aicore__ inline uint64_t ScratchMElemOff(uint64_t hv, uint64_t h, uint32_t mBank)
{
    return (1ULL + static_cast<uint64_t>(mBank & 1U)) * hv * static_cast<uint64_t>(kKvElems) +
           h * static_cast<uint64_t>(kKvElems);
}

// Per-head Mode-4 CV/VC. AIC/AIV counts must match: Vector Sets VC once per
// owned head per rank-step (plus prologue); Cube Waits VC then Sets CV.
__aicore__ inline uint16_t MixFlag(uint8_t flag, uint64_t odd)
{
    return static_cast<uint16_t>(flag + (odd ? kSubBlockFlagOffset : 0U));
}

__aicore__ inline void CubeSetCv0(uint64_t h)
{
    AscendC::CrossCoreSetFlag<kCrossCoreModeAiv, PIPE_FIX>(MixFlag(kCvReadyFlag, h & 1U));
}

__aicore__ inline void CubeSetCv1(uint64_t h)
{
    AscendC::CrossCoreSetFlag<kCrossCoreModeAiv, PIPE_FIX>(MixFlag(kCvC1ReadyFlag, h & 1U));
}

__aicore__ inline void CubeWaitVc(uint64_t h)
{
    AscendC::CrossCoreWaitFlag<kCrossCoreModeAiv, PIPE_MTE2>(MixFlag(kVcReadyFlag, h & 1U));
}

__aicore__ inline void CubeWaitCUbFree(uint64_t h)
{
    // Block Fixpipe until Vector drained fp32Buf on this AIV.
    AscendC::CrossCoreWaitFlag<kCrossCoreModeAiv, PIPE_FIX>(MixFlag(kCUbFreeFlag, h & 1U));
}

__aicore__ inline void VecWaitCv0(uint32_t subIdx)
{
    // Cube Fixpipe C into this AIV's UB; wait on V so Add can consume it.
    AscendC::CrossCoreWaitFlag<kCrossCoreModeAiv, PIPE_V>(MixFlag(kCvReadyFlag, subIdx));
}

__aicore__ inline void VecWaitCv1(uint32_t subIdx)
{
    AscendC::CrossCoreWaitFlag<kCrossCoreModeAiv, PIPE_V>(MixFlag(kCvC1ReadyFlag, subIdx));
}

__aicore__ inline void VecSetVc(uint32_t subIdx)
{
    AscendC::CrossCoreSetFlag<kCrossCoreModeAiv, PIPE_MTE3>(MixFlag(kVcReadyFlag, subIdx));
}

__aicore__ inline void VecSetCUbFree(uint32_t subIdx)
{
    AscendC::CrossCoreSetFlag<kCrossCoreModeAiv, PIPE_V>(MixFlag(kCUbFreeFlag, subIdx));
}

} // namespace MergeFwBwd

#endif
